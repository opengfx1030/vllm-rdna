# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rdna_hint_flex (HadamardINT2-FP16 spec v3.2) weight quantization.

Checkpoints come from the separate ``tools/hi2`` quantizer. Each quantized
linear ``<name>.weight`` is replaced by the flat buffers ``<name>.hi2_<field>``
of spec 2.4 (see ``utils/rdna_hint_flex_utils.py``), and ``config.json`` carries
``quantization_config = {"quant_method": <name>, "version": "3.2"}``
(the name and its aliases live in ``rdna_hint_flex_names.py``).
Every per-tensor choice (row widths, block sizes, Hadamard flags, smoothing
vector) lives in the tensors, so the reader needs no other metadata.

Which modules are quantized is read from the checkpoint index (a module whose
tensors are ``.hi2_*`` is rdna_hint_flex, one with ``.weight`` stays dense), so dense
leftovers such as vision towers, embeddings and the head need no list.

Runtime: on RDNA2 (gfx1030) the ``_rocm_C`` rdna_hint_flex kernels run the F16 compute
mode (spec 6.1): GEMV for decode, dequant + GEMM for prefill. Elsewhere the
weights are dequantized once at load and run through ``F.linear`` (correct,
not compact), so the format never breaks another platform.
"""

import json
import os
from typing import Any, cast

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.rdna_hint_flex_names import (
    RDNA_HINT_FLEX_QUANT_METHOD,
)
from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

SPEC_VERSIONS = ("3.2",)
# Compute modes (spec 6): "f16" (default, v_dot2_f32_f16 on fp16 activations),
# "i8" (v_dot4_i32_i8 on per-16-column int8 activations for M <= DECODE_MAX_M,
# the f16 prefill path above it), "i8_all" (the I8 kernel for every M; slow
# prefill, meant for accuracy evaluation). quantization_config["compute_mode"]
# sets it; the VLLM_RDNA_HINT_FLEX_COMPUTE_MODE environment variable overrides it.
COMPUTE_MODES = {"f16": 0, "i8": 1, "i8_all": 2}
# M at or below this runs the GEMV kernel; above it, dequant + rocBLAS GEMM.
# The GEMV re-reads the compact weights once per 8 tokens and stays faster
# than dequant + GEMM up to about M = 48-64 on Qwen3.5-4B shapes (V620).
DECODE_MAX_M = 48
_FIELD_PREFIX = "hi2_"


def _rdna2_kernels_available() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "rdna_hint_flex_linear_rdna2")
    )


def _norm_name(name: str) -> str:
    """Drop the ``model`` / ``language_model`` wrappers so checkpoint names
    (``model.language_model.layers.3.mlp.up_proj``) and vLLM prefixes
    (``language_model.model.layers.3.mlp.up_proj``) compare equal."""
    return ".".join(p for p in name.split(".") if p not in ("model", "language_model"))


def _checkpoint_tensor_names(model: str, revision: str | None) -> list[str] | None:
    """Tensor names of a local or Hugging Face checkpoint, from its index or
    its single safetensors header."""
    index = "model.safetensors.index.json"
    single = "model.safetensors"
    if os.path.isdir(model):
        path = os.path.join(model, index)
        if os.path.exists(path):
            with open(path) as f:
                return list(json.load(f)["weight_map"])
        path = os.path.join(model, single)
        if os.path.exists(path):
            from safetensors import safe_open

            with safe_open(path, "pt") as f:
                return list(f.keys())  # noqa: SIM118
        return None
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict, hf_api

    try:
        index_dict = get_hf_file_to_dict(index, model, revision)
        if index_dict is not None:
            return list(index_dict["weight_map"])
        meta = hf_api().get_safetensors_metadata(model, revision=revision)
        return [n for fm in meta.files_metadata.values() for n in fm.tensors]
    except Exception:
        return None


class RdnaHintFlexConfig(QuantizationConfig):
    """Config for rdna_hint_flex checkpoints (names: rdna_hint_flex_names.py)."""

    def __init__(
        self, version: str = "3.2", mode: str = "flex", compute_mode: str = "f16"
    ) -> None:
        super().__init__()
        compute_mode = os.environ.get("VLLM_RDNA_HINT_FLEX_COMPUTE_MODE", compute_mode)
        if compute_mode not in COMPUTE_MODES:
            raise ValueError(
                f"rdna_hint_flex compute_mode {compute_mode!r} is not one of "
                f"{sorted(COMPUTE_MODES)}"
            )
        self.compute_mode = compute_mode
        if version not in SPEC_VERSIONS:
            raise ValueError(
                f"rdna_hint_flex version {version!r} is not supported "
                f"(supported: {SPEC_VERSIONS})"
            )
        self.version = version
        self.mode = mode
        # normalized module name -> rdna_hint_flex fields present in the checkpoint.
        # None until maybe_update_config has read the checkpoint index.
        self._rdna_hint_flex_modules: dict[str, set[str]] | None = None

    def __repr__(self) -> str:
        return (
            f"RdnaHintFlexConfig(version={self.version}, mode={self.mode}, "
            f"compute_mode={self.compute_mode})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return cast(QuantizationMethods, RDNA_HINT_FLEX_QUANT_METHOD)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # fp16 only: the RDNA2 dot is v_dot2_f32_f16 (no bf16 on gfx1030).
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates on on_gfx10x() at load time; other platforms use the
        # dequantized fallback.
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RdnaHintFlexConfig":
        return cls(
            version=str(config.get("version", "3.2")),
            mode=str(config.get("mode", "flex")),
            compute_mode=str(config.get("compute_mode", "f16")),
        )

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        names = _checkpoint_tensor_names(model_name, revision)
        if names is None:
            return
        self.set_checkpoint_tensor_names(names)

    def set_checkpoint_tensor_names(self, names: list[str]) -> None:
        modules: dict[str, set[str]] = {}
        for n in names:
            base, sep, field = n.rpartition("." + _FIELD_PREFIX)
            if sep and field in hu.ALL_FIELDS:
                modules.setdefault(_norm_name(base), set()).add(field)
        self._rdna_hint_flex_modules = modules

    def _module_fields(self, prefix: str) -> list[set[str] | None]:
        """Fields present for every checkpoint tensor behind ``prefix`` (one
        entry per packed sub-module); None for a dense sub-module."""
        assert self._rdna_hint_flex_modules is not None
        head, _, last = prefix.rpartition(".")
        subs = self.packed_modules_mapping.get(last, [last])
        out = []
        for s in subs:
            full = f"{head}.{s}" if head else s
            out.append(self._rdna_hint_flex_modules.get(_norm_name(full)))
        return out

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )

        if isinstance(layer, RoutedExperts):
            if self._rdna_hint_flex_modules is None:
                return None
            head = _norm_name(prefix) + "."
            if not any(k.startswith(head) for k in self._rdna_hint_flex_modules):
                return None
            if not _rdna2_kernels_available():
                raise NotImplementedError(
                    "rdna_hint_flex routed experts need the RDNA2 kernels"
                )
            from vllm.model_executor.layers.quantization.rdna_hint_flex_moe import (
                RdnaHintFlexMoEMethod,
            )

            return RdnaHintFlexMoEMethod(
                layer.moe_config, COMPUTE_MODES[self.compute_mode]
            )
        if not isinstance(layer, LinearBase):
            return None
        if self._rdna_hint_flex_modules is None:
            raise ValueError(
                "rdna_hint_flex: the checkpoint index could not be read, so the "
                "quantized modules are unknown (expected a local directory or a "
                "Hugging Face repo with safetensors)."
            )
        fields = self._module_fields(prefix)
        present = [f for f in fields if f is not None]
        if not present:
            return UnquantizedLinearMethod()
        if len(present) != len(fields):
            raise ValueError(
                f"rdna_hint_flex: {prefix} fuses rdna_hint_flex and dense sub-modules; "
                "quantize all of them or none"
            )
        for f in present:
            missing = set(hu.FIELDS) - f
            if missing:
                raise ValueError(
                    f"rdna_hint_flex: {prefix} is missing {sorted(missing)}"
                )
        has_col_scale = any("col_scale" in f for f in present)
        return RdnaHintFlexLinearMethod(has_col_scale, COMPUTE_MODES[self.compute_mode])


class RdnaHintFlexLinearMethod(LinearMethodBase):
    """Linear method for rdna_hint_flex tensors.

    Loading collects each checkpoint tensor's flat buffers per output shard.
    ``process_weights_after_loading`` splits fused checkpoint tensors into
    their logical partitions, slices them for tensor parallelism (rows for
    column-parallel layers, 64-aligned columns for row-parallel layers),
    validates the offsets, and concatenates partitions that share an
    activation transform into one tensor ("segment"), so a fused q/k/v or
    gate/up layer is usually a single kernel call.
    """

    def __init__(self, has_col_scale: bool, compute_mode: int = 0):
        self.has_col_scale = has_col_scale
        self.compute_mode = compute_mode
        self.use_kernels = _rdna2_kernels_available()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % hu.K_MULTIPLE:
            raise ValueError(
                f"rdna_hint_flex: input size per partition {input_size_per_partition} "
                f"is not a multiple of {hu.K_MULTIPLE}"
            )
        layer.rdna_hint_flex_k = input_size_per_partition
        layer.rdna_hint_flex_n = sum(output_partition_sizes)
        layer._rdna_hint_flex_loaded = {}
        fields = hu.FIELDS + (("col_scale",) if self.has_col_scale else ())
        for f in fields:
            # Placeholder: the real buffers are ragged and are collected by
            # the loader, then assembled in process_weights_after_loading.
            p = torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            )
            set_weight_attrs(p, {"weight_loader": self._make_loader(layer, f)})
            layer.register_parameter(_FIELD_PREFIX + f, p)

    @staticmethod
    def _make_loader(layer: torch.nn.Module, field: str):
        def loader(param, loaded_weight: torch.Tensor, shard_id=None):
            if shard_id is None:
                shard_id = getattr(loaded_weight, "shard_id", None)
            entry = layer._rdna_hint_flex_loaded.setdefault(shard_id, {})
            entry[field] = loaded_weight.detach().to("cpu").clone()

        return loader

    # ---- partition bookkeeping -------------------------------------------
    @staticmethod
    def _partitions(layer) -> list[tuple[int, int, int]]:
        """``(checkpoint rows, rank start, rank rows)`` per logical partition."""
        rank, tp = layer.tp_rank, layer.tp_size
        if isinstance(layer, QKVParallelLinear):
            q = layer.num_heads * layer.head_size
            k = layer.num_kv_heads * layer.head_size
            v = layer.num_kv_heads * layer.v_head_size
            kv_rank = rank // layer.num_kv_head_replicas
            return [
                (layer.total_num_heads * layer.head_size, rank * q, q),
                (layer.total_num_kv_heads * layer.head_size, kv_rank * k, k),
                (layer.total_num_kv_heads * layer.v_head_size, kv_rank * v, v),
            ]
        if isinstance(layer, MergedColumnParallelLinear):
            return [(s, rank * (s // tp), s // tp) for s in layer.output_sizes]
        if isinstance(layer, (RowParallelLinear, ReplicatedLinear)):
            return [(layer.output_size, 0, layer.output_size)]
        # Plain column-parallel.
        n = layer.output_size_per_partition
        return [(layer.output_size, rank * n, n)]

    @staticmethod
    def _covered(layer, shard_id, n: int) -> list[int]:
        if shard_id is None:
            return list(range(n))
        if isinstance(shard_id, str):
            return [{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, tuple):
            return list(shard_id)
        return [int(shard_id)]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        loaded: dict = layer._rdna_hint_flex_loaded
        device = getattr(layer, _FIELD_PREFIX + "codes").device
        parts_spec = self._partitions(layer)
        found: list[hu.Hi2 | None] = [None] * len(parts_spec)
        for shard_id, buffers in loaded.items():
            missing = set(hu.FIELDS) - set(buffers)
            if missing:
                raise ValueError(
                    f"rdna_hint_flex: shard {shard_id} of "
                    f"{getattr(layer, 'prefix', '')}"
                    f" is missing {sorted(missing)}"
                )
            t = hu.canonicalize(buffers)
            hu.validate(t)
            idx = self._covered(layer, shard_id, len(parts_spec))
            rows = [parts_spec[i][0] for i in idx]
            if hu.num_rows(t) != sum(rows):
                raise ValueError(
                    f"rdna_hint_flex: shard {shard_id} has {hu.num_rows(t)} rows, "
                    f"expected {sum(rows)}"
                )
            off = 0
            for i, r in zip(idx, rows):
                found[i] = hu.select_rows(t, off, off + r)
                off += r
        parts = [p for p in found if p is not None]
        if len(parts) != len(found):
            raise ValueError("rdna_hint_flex: not every output partition was loaded")

        k_full = hu.num_cols(parts[0])
        k_part = layer.rdna_hint_flex_k
        sliced: list[hu.Hi2] = []
        for p, (full, start, n) in zip(parts, parts_spec):
            if (start, n) != (0, full):
                p = hu.select_rows(p, start, start + n)
            if k_part != k_full:
                if not isinstance(layer, RowParallelLinear):
                    raise ValueError(
                        f"rdna_hint_flex: K={k_full} in the checkpoint, {k_part} "
                        "in a layer that does not shard its input"
                    )
                k0 = layer.tp_rank * k_part
                p = hu.select_cols(p, k0, k0 + k_part)
            hu.validate(p, k_part)
            sliced.append(p)

        # Consecutive partitions with the same transform form one segment.
        groups: list[list[hu.Hi2]] = []
        for p in sliced:
            if groups and hu.same_transform(groups[-1][0], p):
                groups[-1].append(p)
            else:
                groups.append([p])
        segments = []
        for g in groups:
            t = hu.concat_rows(g)
            seg: dict[str, Any] = {
                "transform": hu.needs_transform(t),
                "had": t["had"].to(device),
                "col_scale": (
                    t["col_scale"].to(device)
                    if t.get("col_scale") is not None
                    else torch.empty(0, dtype=torch.float16, device=device)
                ),
            }
            if self.use_kernels:
                for f in hu.FIELDS:
                    if f != "had":
                        seg[f] = t[f].contiguous().to(device)
            else:
                seg["weight"] = hu.dequantize_stored(t).to(device)
            segments.append(seg)
        layer._rdna_hint_flex_segments = segments

        for f in hu.ALL_FIELDS:
            if hasattr(layer, _FIELD_PREFIX + f):
                delattr(layer, _FIELD_PREFIX + f)
        del layer._rdna_hint_flex_loaded
        if not self.use_kernels:
            logger.warning_once(
                "rdna_hint_flex: RDNA2 kernels unavailable; using dequantized "
                "fp16 weights (correct, not memory-saving)."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        outs = []
        for seg in layer._rdna_hint_flex_segments:
            if self.use_kernels:
                y = ops.rdna_hint_flex_linear_rdna2(
                    x2,
                    seg["col_scale"],
                    seg["had"],
                    seg["codes"],
                    seg["code_off"],
                    seg["blk_params"],
                    seg["param_off"],
                    seg["row_consts"],
                    seg["row_nbits"],
                    seg["row_block"],
                    seg["transform"],
                    DECODE_MAX_M,
                    self.compute_mode,
                )
            else:
                xt = x2
                if seg["transform"]:
                    xt = hu.act_transform(x2, seg)
                y = F.linear(xt, seg["weight"])
            outs.append(y)
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], y.shape[-1])
