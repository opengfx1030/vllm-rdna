# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert side of one V620 server of the heterogeneous MoE (``hx`` protocol).

``ExpertServerModel`` holds only the routed experts a placement plan assigns to
one device, for every target layer, in the memory layout of the production TP4-EP
V620 path, and runs the production MoE chain on them:

* weights: the checkpoint's N-first int4 words and bf16 scales are stacked per
  layer into the K-first layout that ``process_weights_after_loading`` of
  ``CompressedTensorsWNA16RDNA2MoEMethod`` leaves after its transpose step; the
  method's own ``process_k_first_weights`` then shuffles them and synthesizes the
  qzeros and the decode scratch (none of that is reimplemented here);
* ``moe()``: ``_rdna2_fused_moe`` of the same module with the device-side expert
  map ``local_map[layer]`` (global id -> local slot or -1) and
  ``ignore_invalid_experts=True``, so a pair whose expert is masked (-1) or lives
  on another server contributes zero: the output is the partial sum over the pairs
  this server holds.

The production module imports the compiled ops and the fused-MoE package. It is
imported lazily by ``load()`` and ``moe()`` (never at import time of this file),
so name resolution, the dequantization reference and the byte accounting work in a
process without the extensions.

Host memory while loading: one pinned staging set (about 2.5 MB per held expert of
the largest layer, reused for every layer) plus the page cache of the mmapped
shards. Nothing close to the checkpoint size is ever resident as anonymous memory.
"""

import contextlib
import importlib
import json
import os
import random
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, NamedTuple

import regex as re
import torch
from safetensors import safe_open

from vllm.logger import init_logger

logger = init_logger(__name__)

PRODUCTION_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2"
)
INDEX_NAME = "model.safetensors.index.json"

# Symmetric uint4b8: stored nibble = value + 8 (scalar_types.uint4b8.bias).
UINT4B8_BIAS = 8
INT4_PER_WORD = 8  # nibbles per int32, packed LSB-first along K

# Known-answer selftest: max abs error must stay below REL_TOL * max|reference|
# plus the floor (two fp16 ulps at 1.0). A zero output, a swapped gate/up, a wrong
# expert map or a silent no-op kernel is off by about 100 % of max|reference|; the
# fp16 kernel against the fp32 reference is off by about 0.1 %.
SELFTEST_REL_TOL = 0.03
SELFTEST_ABS_FLOOR = 2e-3
SELFTEST_ROWS = 6
SELFTEST_DECODE_ROWS = 3  # also run the first 3 rows alone: the M <= 4 tile

ROLES = (
    "gate_packed",
    "gate_scale",
    "up_packed",
    "up_scale",
    "down_packed",
    "down_scale",
)
_ROLE_OF = {
    ("gate_proj", "weight_packed"): "gate_packed",
    ("gate_proj", "weight_scale"): "gate_scale",
    ("up_proj", "weight_packed"): "up_packed",
    ("up_proj", "weight_scale"): "up_scale",
    ("down_proj", "weight_packed"): "down_packed",
    ("down_proj", "weight_scale"): "down_scale",
}
_EXPERT_KEY = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale)$"
)
_SCALE_DTYPES = ("BF16", "F16", "F32")

_HOLDER_TENSORS = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_qzeros",
    "w2_qzeros",
)
_SCRATCH = ("rdna2_w1_buf", "rdna2_act_buf", "rdna2_empty_tw")


def parse_expert_key(name: str) -> tuple[int, int, str] | None:
    """``(layer, expert, role)`` of a routed-expert tensor name, else None.

    Matches ``<any prefix>layers.L.mlp.experts.E.{gate,up,down}_proj.weight_{packed,
    scale}`` (prefixes seen: ``model.language_model.``, ``model.``, none). The MTP
    drafter's ``mtp.`` tensors are hub-local and never match; ``weight_shape`` is
    metadata and is ignored.
    """
    if name.startswith("mtp."):
        return None
    match = _EXPERT_KEY.search(name)
    if match is None:
        return None
    role = _ROLE_OF[(match.group(3), match.group(4))]
    return int(match.group(1)), int(match.group(2)), role


def resolve_expert_tensors(
    weight_map: dict[str, str], wanted: set[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, str]]:
    """Tensor names of every wanted ``(layer, expert)``, in one pass over the index.

    Raises:
        KeyError: a wanted expert lacks one of the six tensors.
        ValueError: the index lists the same tensor twice (two prefix variants).

    """
    found: dict[tuple[int, int], dict[str, str]] = {key: {} for key in wanted}
    for name in weight_map:
        parsed = parse_expert_key(name)
        if parsed is None:
            continue
        layer, expert, role = parsed
        roles = found.get((layer, expert))
        if roles is None:
            continue
        if role in roles:
            raise ValueError(
                f"layer {layer} expert {expert}: {role} is listed twice in the "
                f"index ({roles[role]!r} and {name!r})"
            )
        roles[role] = name
    for (layer, expert), roles in found.items():
        missing = [role for role in ROLES if role not in roles]
        if missing:
            raise KeyError(
                f"layer {layer} expert {expert}: the index has no "
                f"{', '.join(missing)} tensor"
            )
    return found


def expert_bytes(hidden: int, inter: int, group_size: int) -> tuple[int, int]:
    """Device bytes of one expert: ``(weights + scales, synthesized qzeros)``.

    K-first RDNA2 layout: w13 int32 [hidden/8, 2*inter], w2 int32 [inter/8, hidden],
    fp16 scales [hidden/g, 2*inter] and [inter/g, hidden], int32 qzeros
    [hidden/g, 2*inter/8] and [inter/g, hidden/8]. 2,534,400 + 19,200 bytes at
    hidden 2560, inter 640, group 128.
    """
    w13 = (hidden // 8) * (2 * inter) * 4
    w2 = (inter // 8) * hidden * 4
    scales = ((hidden // group_size) * (2 * inter) + (inter // group_size) * hidden) * 2
    qzeros = (
        (hidden // group_size) * (2 * inter // 8)
        + (inter // group_size) * (hidden // 8)
    ) * 4
    return w13 + w2 + scales, qzeros


def unpack_int4_words(packed: torch.Tensor) -> torch.Tensor:
    """int32 ``[..., W]`` words -> int32 ``[..., 8*W]`` nibbles in 0..15.

    Element ``8*j + i`` of a row is nibble ``i`` (bits ``4i..4i+3``) of word ``j``:
    the LSB-first order of ``pack_quantized_values_into_int32``. The mask makes the
    arithmetic right shift of negative words harmless.
    """
    shifts = torch.arange(
        0, 32, 32 // INT4_PER_WORD, dtype=torch.int32, device=packed.device
    )
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(*packed.shape[:-1], packed.shape[-1] * INT4_PER_WORD)


def dequantize_checkpoint_weight(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """fp32 ``[N, K]`` weight of one checkpoint projection (symmetric uint4b8).

    ``packed`` is the checkpoint's int32 ``[N, K/8]`` (packed along K), ``scale``
    its ``[N, K/group_size]`` scales (bf16 in the checkpoint); value =
    (nibble - 8) * scale.
    """
    values = unpack_int4_words(packed).to(torch.float32) - UINT4B8_BIAS
    scales = scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    if scales.shape != values.shape:
        raise ValueError(
            f"scale {tuple(scale.shape)} does not match packed {tuple(packed.shape)} "
            f"at group size {group_size}"
        )
    return values * scales


def reference_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    experts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """fp32 CPU MoE over the experts in ``experts`` (global id -> gate, up, down).

    ``gate``/``up`` are fp32 ``[inter, hidden]``, ``down`` ``[hidden, inter]``
    (``dequantize_checkpoint_weight`` outputs). Pairs whose id is not a key
    (masked -1, or an expert held elsewhere) contribute nothing.
    """
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    xf = x.to(torch.float32)
    for expert, (gate, up, down) in experts.items():
        rows, cols = torch.nonzero(topk_ids == expert, as_tuple=True)
        if rows.numel() == 0:
            continue
        xs = xf[rows]
        act = torch.nn.functional.silu(xs @ gate.T) * (xs @ up.T)
        weights = topk_w[rows, cols].to(torch.float32).unsqueeze(1)
        out.index_add_(0, rows, (act @ down.T) * weights)
    return out


class _TensorRef(NamedTuple):
    name: str
    shard: str
    offset: int  # first byte of the tensor data inside the shard's data section


def _read_safetensors_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _production() -> Any:
    return importlib.import_module(PRODUCTION_MODULE)


@contextlib.contextmanager
def _shuffle_override(
    prod: Any, shuffle: Callable[[torch.Tensor, int], None] | None
) -> Iterator[None]:
    """Run ``process_k_first_weights`` with ``shuffle`` as ``ops.gptq_shuffle``.

    The method calls ``vllm._custom_ops.gptq_shuffle`` through its module global;
    swapping that attribute for the duration is how a test replaces the HIP op.
    With ``shuffle=None`` (production) nothing is touched.
    """
    if shuffle is None:
        yield
        return
    ops = prod.ops
    original = ops.gptq_shuffle
    ops.gptq_shuffle = shuffle
    try:
        yield
    finally:
        ops.gptq_shuffle = original


class _ShardFiles:
    """One ``safe_open`` per shard, opened on first use, closed with the stack."""

    def __init__(self, model_dir: str, stack: contextlib.ExitStack) -> None:
        self._dir = model_dir
        self._stack = stack
        self._open: dict[str, Any] = {}

    def get_tensor(self, shard: str, name: str) -> torch.Tensor:
        handle = self._open.get(shard)
        if handle is None:
            path = os.path.join(self._dir, shard)
            handle = self._stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            self._open[shard] = handle
        return handle.get_tensor(name)


def _slot_view(
    role: str,
    slot: int,
    inter: int,
    w13p: torch.Tensor,
    w2p: torch.Tensor,
    w13s: torch.Tensor,
    w2s: torch.Tensor,
) -> torch.Tensor:
    """Where a checkpoint tensor lands in the K-first stacks: gate first, up second."""
    if role == "gate_packed":
        return w13p[slot, :, :inter]
    if role == "up_packed":
        return w13p[slot, :, inter:]
    if role == "down_packed":
        return w2p[slot]
    if role == "gate_scale":
        return w13s[slot, :, :inter]
    if role == "up_scale":
        return w13s[slot, :, inter:]
    return w2s[slot]


class ExpertServerModel:
    """The routed experts of one expert server and the chain that runs them."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        expert_sets: Sequence[Sequence[int]],
        device: str | torch.device,
        *,
        num_experts: int = 512,
        top_k: int = 10,
        max_capture_rows: int = 48,
        group_size: int = 128,
        shuffle: Callable[[torch.Tensor, int], None] | None = None,
    ) -> None:
        """Resolve and validate the experts against the checkpoint; load nothing.

        Reads the index and the headers of the shards that hold the requested
        experts, so a missing tensor or a shape that does not fit fails here, not
        after minutes of loading. ``hidden`` and ``inter`` come from those shapes.
        The ``local_map`` tensors are created on ``device`` and keep their address
        for the life of the object (graphs read them through a raw pointer).

        Args:
            model_dir: checkpoint directory (``model.safetensors.index.json`` and
                its shards, read-only).
            expert_sets: ``expert_sets[l]`` = strictly ascending global expert ids
                this server holds in layer ``l``; may be empty. The local slot of
                an expert is its index in the list. Its length is ``num_layers``;
                layers with no experts load nothing and ``moe()`` returns zeros
                for them.
            device: device of the weights, e.g. ``"cuda:0"`` (``"cpu"`` in tests).
            num_experts: routed experts per layer (size of every ``local_map``).
            top_k: routing fan-out; the ``ids``/``w`` width ``moe()`` accepts.
            max_capture_rows: largest graph-capture width; sizes the shared decode
                scratch (``max_capture_rows * top_k`` rows, at least 128).
            group_size: quantization group along K.
            shuffle: replacement for ``vllm._custom_ops.gptq_shuffle(tensor, 4)``
                (in-place exllama shuffle). ``None`` uses the HIP op; tests inject
                a fake because the op needs the compiled extension.

        """
        if num_experts < 1 or top_k < 1 or max_capture_rows < 1 or group_size < 8:
            raise ValueError(
                f"num_experts={num_experts}, top_k={top_k}, "
                f"max_capture_rows={max_capture_rows}, group_size={group_size}"
            )
        self.model_dir = os.fspath(model_dir)
        # Resolves "cuda" to "cuda:N" without naming the current device.
        self.device = torch.empty(0, device=torch.device(device)).device
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_capture_rows = max_capture_rows
        self.group_size = group_size
        self.hidden = 0
        self.inter = 0
        self.expert_sets = [
            self._checked_expert_set(layer, experts)
            for layer, experts in enumerate(expert_sets)
        ]
        if not any(self.expert_sets):
            raise ValueError("expert_sets holds no expert: nothing to serve")
        self.local_map: list[torch.Tensor] = [
            self._build_local_map(experts) for experts in self.expert_sets
        ]
        self._refs = self._plan_tensors()
        self._shuffle = shuffle
        self._prod: Any = None
        self._silu: Any = None
        self._method: Any = None
        self._layers: list[SimpleNamespace | None] = [None] * len(self.expert_sets)
        self._scratch: SimpleNamespace | None = None
        self._loaded = False

    @property
    def num_layers(self) -> int:
        return len(self.expert_sets)

    @classmethod
    def expert_tensor_names(
        cls, weight_map: dict[str, str], layer: int, expert: int
    ) -> dict[str, str]:
        """The six tensor names of one expert, found by regex on the index keys.

        Keys of the result: ``gate_packed``, ``gate_scale``, ``up_packed``,
        ``up_scale``, ``down_packed``, ``down_scale``. Works for every prefix
        variant (``model.language_model.layers.L``, ``model.layers.L``, ...).
        """
        return resolve_expert_tensors(weight_map, {(layer, expert)})[(layer, expert)]

    def layer_module(self, layer: int) -> SimpleNamespace | None:
        """The duck-typed layer the production chain reads (None: no local expert)."""
        return self._layers[layer]

    def _checked_expert_set(self, layer: int, experts: Sequence[int]) -> list[int]:
        ids = [int(e) for e in experts]
        if any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError(f"layer {layer}: expert ids must be strictly ascending")
        if ids and (ids[0] < 0 or ids[-1] >= self.num_experts):
            raise ValueError(
                f"layer {layer}: expert ids must be in [0, {self.num_experts})"
            )
        return ids

    def _build_local_map(self, experts: list[int]) -> torch.Tensor:
        table = torch.full((self.num_experts,), -1, dtype=torch.int32)
        if experts:
            table[torch.tensor(experts, dtype=torch.int64)] = torch.arange(
                len(experts), dtype=torch.int32
            )
        return table.to(self.device)

    def _plan_tensors(self) -> dict[tuple[int, int], dict[str, _TensorRef]]:
        with open(os.path.join(self.model_dir, INDEX_NAME), encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        wanted = {
            (layer, expert)
            for layer, experts in enumerate(self.expert_sets)
            for expert in experts
        }
        names = resolve_expert_tensors(weight_map, wanted)
        by_shard: dict[str, list[str]] = {}
        for roles in names.values():
            for name in roles.values():
                by_shard.setdefault(weight_map[name], []).append(name)
        infos: dict[str, tuple[str, tuple[int, ...], int]] = {}
        for shard, shard_names in by_shard.items():
            header = _read_safetensors_header(os.path.join(self.model_dir, shard))
            for name in shard_names:
                entry = header.get(name)
                if entry is None:
                    raise KeyError(f"{name!r} is in the index but not in {shard}")
                infos[name] = (
                    entry["dtype"],
                    tuple(entry["shape"]),
                    int(entry["data_offsets"][0]),
                )
        first = min(wanted)
        shape = infos[names[first]["gate_packed"]][1]
        if len(shape) != 2:
            raise ValueError(f"{names[first]['gate_packed']}: shape {shape} is not 2-D")
        self.inter, words = shape
        self.hidden = words * INT4_PER_WORD
        g = self.group_size
        if self.hidden % g or self.inter % g:
            raise ValueError(
                f"hidden {self.hidden} and inter {self.inter} must be multiples "
                f"of group_size {g}"
            )
        refs: dict[tuple[int, int], dict[str, _TensorRef]] = {}
        for key, roles in names.items():
            for role, name in roles.items():
                self._check_header(key, role, name, infos[name])
            refs[key] = {
                role: _TensorRef(name, weight_map[name], infos[name][2])
                for role, name in roles.items()
            }
        return refs

    def _check_header(
        self,
        key: tuple[int, int],
        role: str,
        name: str,
        info: tuple[str, tuple[int, ...], int],
    ) -> None:
        hidden, inter, g = self.hidden, self.inter, self.group_size
        expected = {
            "gate_packed": (("I32",), (inter, hidden // 8)),
            "up_packed": (("I32",), (inter, hidden // 8)),
            "down_packed": (("I32",), (hidden, inter // 8)),
            "gate_scale": (_SCALE_DTYPES, (inter, hidden // g)),
            "up_scale": (_SCALE_DTYPES, (inter, hidden // g)),
            "down_scale": (_SCALE_DTYPES, (hidden, inter // g)),
        }[role]
        dtype, shape, _ = info
        if dtype not in expected[0] or shape != expected[1]:
            raise ValueError(
                f"layer {key[0]} expert {key[1]}: {name} is {dtype}{list(shape)}, "
                f"expected {'/'.join(expected[0])}{list(expected[1])}"
            )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("ExpertServerModel.load() has not completed")

    def _check_arch(self) -> None:
        if self.device.type != "cuda":
            return
        arch = getattr(torch.cuda.get_device_properties(self.device), "gcnArchName", "")
        if not arch.startswith("gfx10"):
            raise RuntimeError(
                f"{self.device} is {arch or 'not a ROCm device'}: the RDNA2 MoE "
                "kernels only compile for gfx10 and run as silent no-ops elsewhere"
            )

    def _new_method(self, prod: Any) -> Any:
        """The production method object, built the way rdna2_moe_resident does."""
        method = object.__new__(prod.CompressedTensorsWNA16RDNA2MoEMethod)
        method.group_size = self.group_size
        method.symmetric = True
        method.moe = SimpleNamespace(
            experts_per_token=self.top_k, max_capture_size=self.max_capture_rows
        )
        return method

    def _to_device(self, host: torch.Tensor) -> torch.Tensor:
        out = torch.empty(host.shape, dtype=host.dtype, device=self.device)
        out.copy_(host)
        return out

    def _load_layer(
        self,
        layer: int,
        shards: _ShardFiles,
        staging: SimpleNamespace,
        method: Any,
        prod: Any,
    ) -> SimpleNamespace:
        experts = self.expert_sets[layer]
        n = len(experts)
        w13p, w2p = staging.w13p[:n], staging.w2p[:n]
        w13s, w2s = staging.w13s[:n], staging.w2s[:n]
        jobs = sorted(
            (ref.shard, ref.offset, slot, role, ref.name)
            for slot, expert in enumerate(experts)
            for role, ref in self._refs[(layer, expert)].items()
        )
        for shard, _, slot, role, name in jobs:
            tensor = shards.get_tensor(shard, name)
            dest = _slot_view(role, slot, self.inter, w13p, w2p, w13s, w2s)
            dest.copy_(tensor.t())  # N-first -> K-first; scales cast to fp16 here
        holder = SimpleNamespace(
            w13_weight_packed=self._to_device(w13p),
            w2_weight_packed=self._to_device(w2p),
            w13_weight_scale=self._to_device(w13s),
            w2_weight_scale=self._to_device(w2s),
        )
        with _shuffle_override(prod, self._shuffle):
            method.process_k_first_weights(holder)
        return holder

    def load(self, progress: Callable[[int, int], None] | None = None) -> None:
        """Read the held experts of every layer from the checkpoint onto the device.

        Per layer: the needed tensors are read through one ``safe_open`` per shard
        in file-offset order, stacked K-first into the reused (pinned on cuda)
        staging buffers, copied to the device in one transfer per parameter, and
        handed to the production ``process_k_first_weights``. The decode scratch
        of the first loaded layer is shared by all layers (they run one at a time
        on one stream). ``progress(layer, num_layers)`` is called after each layer
        (also the empty ones). A failed load leaves the model unusable.
        """
        if self._loaded:
            raise RuntimeError("ExpertServerModel.load() was already called")
        prod = _production()
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        self._check_arch()
        method = self._new_method(prod)
        start = time.perf_counter()
        layers: list[SimpleNamespace | None] = [None] * self.num_layers
        scratch: SimpleNamespace | None = None
        capacity = max(len(experts) for experts in self.expert_sets)
        pin = self.device.type == "cuda"
        hidden, inter, g = self.hidden, self.inter, self.group_size
        staging = SimpleNamespace(
            w13p=torch.empty(
                (capacity, hidden // 8, 2 * inter), dtype=torch.int32, pin_memory=pin
            ),
            w2p=torch.empty(
                (capacity, inter // 8, hidden), dtype=torch.int32, pin_memory=pin
            ),
            w13s=torch.empty(
                (capacity, hidden // g, 2 * inter), dtype=torch.float16, pin_memory=pin
            ),
            w2s=torch.empty(
                (capacity, inter // g, hidden), dtype=torch.float16, pin_memory=pin
            ),
        )
        with contextlib.ExitStack() as stack:
            shards = _ShardFiles(self.model_dir, stack)
            for layer, experts in enumerate(self.expert_sets):
                if experts:
                    holder = self._load_layer(layer, shards, staging, method, prod)
                    if scratch is None:
                        scratch = SimpleNamespace(
                            **{name: getattr(holder, name) for name in _SCRATCH}
                        )
                    else:
                        for name in _SCRATCH:
                            setattr(holder, name, getattr(scratch, name))
                    layers[layer] = holder
                    logger.debug(
                        "hx server model: layer %d loaded, %d experts",
                        layer,
                        len(experts),
                    )
                if progress is not None:
                    progress(layer, self.num_layers)
        del staging
        self._prod = prod
        self._silu = MoEActivation.SILU
        self._method = method
        self._layers = layers
        self._scratch = scratch
        self._loaded = True
        logger.info(
            "hx server model on %s: %d experts in %d layers, %.3f GiB, %.1f s",
            self.device,
            sum(len(experts) for experts in self.expert_sets),
            sum(1 for experts in self.expert_sets if experts),
            self.memory_bytes() / 2**30,
            time.perf_counter() - start,
        )

    def memory_bytes(self) -> int:
        """Bytes of the device tensors this object owns.

        Per expert ``expert_bytes()`` (2,534,400 + 19,200 B of qzeros at the
        production geometry), plus the shared decode scratch once and the
        ``local_map`` tables.
        """
        total = sum(_nbytes(table) for table in self.local_map)
        for holder in self._layers:
            if holder is not None:
                total += sum(_nbytes(getattr(holder, n)) for n in _HOLDER_TENSORS)
        if self._scratch is not None:
            total += sum(_nbytes(getattr(self._scratch, n)) for n in _SCRATCH)
        return total

    def _check_call(
        self,
        layer: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} outside [0, {self.num_layers})")
        rows = x.shape[0] if x.dim() == 2 else -1
        for what, tensor, dtype, shape in (
            ("x", x, torch.float16, (rows, self.hidden)),
            ("topk_ids", topk_ids, torch.int32, (rows, self.top_k)),
            ("topk_w", topk_w, torch.float32, (rows, self.top_k)),
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{what} must be a contiguous {dtype} {list(shape)} tensor on "
                    f"{self.device}, got {tensor.dtype} {list(tensor.shape)} on "
                    f"{tensor.device}"
                )

    def moe(
        self,
        layer: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> torch.Tensor:
        """Partial MoE sum of this server for one layer: a fresh fp16 ``[T, H]``.

        ``x`` is fp16 ``[T, hidden]`` (it may be a non-owning view of memory
        allocated outside torch), ``topk_ids`` int32 ``[T, top_k]`` of global
        expert ids (-1 masked; an id this server does not hold contributes zero),
        ``topk_w`` fp32 ``[T, top_k]``. Only shapes and dtypes are inspected on the
        host, so the call can be captured in a CUDA graph. A layer with no local
        expert returns zeros without running the chain.
        """
        self._require_loaded()
        self._check_call(layer, x, topk_ids, topk_w)
        holder = self._layers[layer]
        if holder is None or x.shape[0] == 0:
            return torch.zeros(x.shape[0], self.hidden, dtype=x.dtype, device=x.device)
        return self._prod._rdna2_fused_moe(
            x,
            topk_w,
            topk_ids,
            layer=holder,
            activation=self._silu,
            apply_router_weight_on_input=False,
            global_num_experts=self.num_experts,
            expert_map=self.local_map[layer],
            w4a8=self._method._w4a8,
            zero_v2=False,
            ignore_invalid_experts=True,
        )

    def _reference_experts(
        self, layer: int, experts: list[int]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """fp32 dequantized checkpoint weights, read again from the shards."""
        out = {}
        with contextlib.ExitStack() as stack:
            shards = _ShardFiles(self.model_dir, stack)
            for expert in experts:
                raw = {
                    role: shards.get_tensor(ref.shard, ref.name)
                    for role, ref in self._refs[(layer, expert)].items()
                }
                g = self.group_size
                out[expert] = (
                    dequantize_checkpoint_weight(
                        raw["gate_packed"], raw["gate_scale"], g
                    ),
                    dequantize_checkpoint_weight(raw["up_packed"], raw["up_scale"], g),
                    dequantize_checkpoint_weight(
                        raw["down_packed"], raw["down_scale"], g
                    ),
                )
        return out

    def _selftest_inputs(
        self, layer: int, experts_per_layer: int, seed: int
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        held = self.expert_sets[layer]
        key = seed * 1_000_003 + layer
        rng = random.Random(key)
        gen = torch.Generator().manual_seed(key)
        count = max(1, min(experts_per_layer, len(held), self.top_k))
        if count >= len(held):
            slots = list(range(len(held)))
        elif count == 1:
            slots = [rng.randrange(len(held))]
        else:  # both ends of the stack always, the rest seeded
            rest = list(range(1, len(held) - 1))
            slots = sorted({0, len(held) - 1, *rng.sample(rest, count - 2)})
        tested = [held[s] for s in slots]
        held_set = set(held)
        pool = rng.sample(
            [e for e in range(self.num_experts) if e not in held_set],
            min(self.top_k, self.num_experts - len(held)),
        )
        subsets = [
            tested,  # every selected expert
            tested[:1],
            tested[-1:],
            [],  # nothing held: this row must stay exactly zero
            rng.sample(tested, rng.randint(1, len(tested))),
            tested[::-1],
        ]
        ids = torch.empty(SELFTEST_ROWS, self.top_k, dtype=torch.int32)
        for r, subset in enumerate(subsets):
            fill = [
                pool[i // 2] if i % 2 and i // 2 < len(pool) else -1
                for i in range(self.top_k - len(subset))
            ]
            row = subset + fill
            rng.shuffle(row)
            ids[r] = torch.tensor(row, dtype=torch.int32)
        x = torch.randn(SELFTEST_ROWS, self.hidden, generator=gen).to(torch.float16)
        w = torch.softmax(torch.randn(SELFTEST_ROWS, self.top_k, generator=gen), -1)
        return tested, x, ids, w

    def _selftest_layer(
        self, layer: int, experts_per_layer: int, seed: int
    ) -> tuple[dict[str, Any], list[str]]:
        tested, x, ids, w = self._selftest_inputs(layer, experts_per_layer, seed)
        t0 = time.perf_counter()
        reference = reference_moe(x, ids, w, self._reference_experts(layer, tested))
        t1 = time.perf_counter()
        x_dev, ids_dev, w_dev = (t.to(self.device) for t in (x, ids, w))
        rows = SELFTEST_DECODE_ROWS
        got_full = self.moe(layer, x_dev, ids_dev, w_dev).float().cpu()
        got_decode = (
            self.moe(layer, x_dev[:rows], ids_dev[:rows], w_dev[:rows]).float().cpu()
        )
        t2 = time.perf_counter()
        peak = reference.abs().max().item()
        tol = SELFTEST_REL_TOL * peak + SELFTEST_ABS_FLOOR
        err_full = (got_full - reference).abs().max().item()
        err_decode = (got_decode - reference[:rows]).abs().max().item()
        problems = []
        for tag, got, err in (
            (f"{SELFTEST_ROWS} rows", got_full, err_full),
            (f"{rows} rows", got_decode, err_decode),
        ):
            if not torch.isfinite(got).all():
                problems.append(f"{tag}: non-finite output")
            elif got.abs().max().item() == 0.0:
                problems.append(f"{tag}: all zero (reference max {peak:.4g})")
            elif err > tol:
                problems.append(
                    f"{tag}: max abs error {err:.4g} > tolerance {tol:.4g} "
                    f"(reference max {peak:.4g})"
                )
        for r in range(SELFTEST_ROWS):
            holds_none = not set(ids[r].tolist()) & set(tested)
            if holds_none and got_full[r].abs().max().item() != 0.0:
                problems.append(f"row {r} holds no local expert but is not zero")
        record = {
            "experts": tested,
            "max_abs_ref": peak,
            "tolerance": tol,
            "max_abs_err": err_full,
            "max_abs_err_3rows": err_decode,
            "ref_ms": (t1 - t0) * 1e3,
            "moe_ms": (t2 - t1) * 1e3,
        }
        return record, problems

    def known_answer_selftest(
        self,
        layers: Sequence[int] | None = None,
        experts_per_layer: int = 2,
        seed: int = 1234,
    ) -> dict[str, Any]:
        """Run ``moe()`` on seeded inputs and compare with an fp32 CPU reference.

        Per layer: ``experts_per_layer`` of this server's experts (capped at
        ``top_k``; the first and last slot are always among them) are dequantized
        straight from the checkpoint shards (value = (nibble - 8) * scale), and a
        6-row batch is built from them plus masked -1 ids and ids of experts this
        server does not hold, with one row that holds no selected expert at all (it
        must come out exactly zero). The batch runs through ``moe()`` as 6 rows and
        as its first 3 rows (the other kernel tile). A result passes when it is
        finite, not all zero, and its max abs error is below
        ``SELFTEST_REL_TOL * max|reference| + SELFTEST_ABS_FLOOR``.

        Args:
            layers: layers to check; default every layer that holds experts.
            experts_per_layer: how many held experts each layer's batch uses.
            seed: seeds inputs and expert choice; the same seed repeats the run.

        Returns:
            Per-layer and overall max abs errors, tolerances and timings (ms).

        Raises:
            RuntimeError: some layer failed (all requested layers run first).
            ValueError: a requested layer holds no expert.

        """
        self._require_loaded()
        if experts_per_layer < 1:
            raise ValueError("experts_per_layer must be at least 1")
        if layers is None:
            layers = [i for i, experts in enumerate(self.expert_sets) if experts]
        for layer in layers:
            if not 0 <= layer < self.num_layers or not self.expert_sets[layer]:
                raise ValueError(f"layer {layer} holds no expert on this server")
        per_layer: dict[int, dict[str, Any]] = {}
        failures: list[str] = []
        for layer in layers:
            record, problems = self._selftest_layer(layer, experts_per_layer, seed)
            per_layer[layer] = record
            failures += [
                f"layer {layer} experts {record['experts']}: {p}" for p in problems
            ]
            logger.info(
                "hx server selftest layer %d experts %s: err %.3g / tol %.3g (%s)",
                layer,
                record["experts"],
                max(record["max_abs_err"], record["max_abs_err_3rows"]),
                record["tolerance"],
                "FAIL" if problems else "ok",
            )
        if failures:
            raise RuntimeError(
                "ExpertServerModel known-answer selftest failed:\n  "
                + "\n  ".join(failures)
            )
        records = list(per_layer.values())
        return {
            "layers": list(layers),
            "rows": SELFTEST_ROWS,
            "seed": seed,
            "rel_tol": SELFTEST_REL_TOL,
            "abs_floor": SELFTEST_ABS_FLOOR,
            "max_abs_err": max(
                max(r["max_abs_err"], r["max_abs_err_3rows"]) for r in records
            ),
            "max_abs_ref": max(r["max_abs_ref"] for r in records),
            "ref_ms": sum(r["ref_ms"] for r in records),
            "moe_ms": sum(r["moe_ms"] for r in records),
            "per_layer": per_layer,
        }
