# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in VRAM residency cache for routed MoE experts.

The idea follows Strata's expert cache: a hot subset stays in VRAM, and a
miss is copied from a pinned host copy. An optional popularity profile
chooses the experts that start hot.

This is not a Strata port. Strata's runtime, GGUF packs, CPU expert
kernels, and measured rates are intentionally absent. Once an expert row
is on device, the existing fused MoE kernel (including fdot2 / sdot4)
reads that row unchanged.

``VLLM_RDNA_MOE_RESIDENT`` is a different hook: it keeps the full W4A16
set in the native device layout. It is not reused here. If that flag is
on, this cache stays off.

The cache is default-off (``VLLM_RDNA_MOE_EXPERT_OFFLOAD=0``) and
serve-side. It is built only when the routed-expert bytes exceed the
VRAM budget. When they fit, no host pages and no residency table are
kept.

Only routed-expert parameters are eligible. Shared experts, embeddings,
the head, norms, the router, and PLE / n-gram / Engram tables are not
expert pages. The host copy is a pinned CPU buffer fetched on a side
stream. That fetch is refused while a HIP/CUDA graph is capturing.
Device slots are ordinary parameters of the routed-expert module; they
are not taken from GDN, QSA, PLE, or hc_combine workspace. Under tensor
or expert parallel each rank keeps its own table of local experts.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_INDEX = re.compile(r"layers\.(\d+)")
_EXCLUDED_BASENAMES = frozenset(
    {
        "e_score_correction_bias",
        "w13_input_scale",
        "w2_input_scale",
        "w13_input_global_scale",
        "w2_input_global_scale",
        "hash_indices_table",
    }
)
_EXCLUDED_PARTS = frozenset(
    {
        "ple",
        "engram",
        "ngram",
        "ngram_embedding",
        "shared_expert",
        "shared_experts",
    }
)


def expert_dram_offload_requested() -> bool:
    """True when the serve process asked for expert DRAM offload."""
    return bool(envs.VLLM_RDNA_MOE_EXPERT_OFFLOAD)


def full_resident_layout_requested() -> bool:
    """True when the native full-weight resident MoE path is selected."""
    return bool(envs.VLLM_RDNA_MOE_RESIDENT or envs.VLLM_RDNA_MOE_RESIDENT_SKINNY)


def expert_dram_offload_enabled() -> bool:
    """Offload is requested and the full-weight resident path is not.

    The resident flags keep every expert on device in a different layout.
    Running both would be a second, overlapping placement policy.
    """
    if not expert_dram_offload_requested():
        return False
    if full_resident_layout_requested():
        logger.warning_once(
            "VLLM_RDNA_MOE_EXPERT_OFFLOAD is set, but "
            "VLLM_RDNA_MOE_RESIDENT keeps the full expert set on device. "
            "Expert DRAM offload stays off."
        )
        return False
    return True


def should_engage_expert_offload(
    *,
    enabled: bool,
    expert_bytes: int,
    vram_budget: int,
) -> bool:
    """Return whether the residency cache should be built.

    Args:
        enabled: Master switch. Default off, so this is false in serve.
        expert_bytes: Bytes of routed-expert parameters on this rank.
        vram_budget: Bytes of VRAM those parameters may occupy.

    Returns:
        True only when the switch is on and the experts do not fit.
    """
    if not enabled or expert_bytes <= 0:
        return False
    return expert_bytes > max(int(vram_budget), 0)


def room_for_experts(
    *,
    total: int,
    allocated: int,
    expert_bytes_on_device: int,
    utilization: float,
) -> int:
    """VRAM bytes available for the routed-expert set.

    Args:
        total: Device memory in bytes.
        allocated: Bytes the caching allocator currently holds.
        expert_bytes_on_device: Routed-expert bytes already inside
            ``allocated``. They are added back so a set that is already
            resident is judged against the room it occupies.
        utilization: Same ceiling ``gpu_memory_utilization`` applies to KV.

    Returns:
        Non-negative byte budget for routed experts.
    """
    usable = int(total * utilization)
    other = max(int(allocated) - int(expert_bytes_on_device), 0)
    return max(usable - other, 0)


def experts_fit_in_vram(expert_bytes: int, vram_budget: int) -> bool:
    """True when the full routed-expert set fits in the budget."""
    return expert_bytes <= max(int(vram_budget), 0)


def assign_slots(
    per_expert_bytes: list[int],
    local_experts: list[int],
    budget: int,
) -> list[int]:
    """Split ``budget`` into per-layer resident slot counts.

    Every offloaded layer gets one slot when the budget can hold that
    minimum. Extra slots are handed out round-robin.

    Raises:
        RuntimeError: The budget cannot hold one expert from each layer.
    """
    count = len(per_expert_bytes)
    if count == 0:
        return []
    one = sum(per_expert_bytes)
    if one > budget:
        raise RuntimeError(
            "routed experts do not fit in VRAM, and the remaining budget "
            f"({budget} bytes) cannot hold one expert from each MoE layer "
            f"({one} bytes)"
        )
    slots = [1] * count
    remaining = budget - one
    progressed = True
    while progressed:
        progressed = False
        for index in range(count):
            if slots[index] >= local_experts[index]:
                continue
            cost = per_expert_bytes[index]
            if cost <= remaining:
                slots[index] += 1
                remaining -= cost
                progressed = True
    return slots


def layer_index_from_name(name: str, fallback: int) -> int:
    """Parse ``layers.{i}`` from a module name, else ``fallback``."""
    match = _LAYER_INDEX.search(name)
    if match is None:
        return fallback
    return int(match.group(1))


def _basename(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def is_routed_expert_parameter(
    name: str,
    tensor: torch.Tensor,
    local_num_experts: int,
) -> bool:
    """True for one rank-local routed-expert parameter.

    Shared experts, PLE / n-gram / Engram tables, and per-tensor scales
    fail this check even if they happen to live on the same module.
    """
    if local_num_experts <= 0 or tensor.ndim < 1:
        return False
    if tensor.shape[0] != local_num_experts or tensor.numel() == 0:
        return False
    base = _basename(name)
    if base in _EXCLUDED_BASENAMES:
        return False
    parts = set(name.lower().replace("-", "_").split("."))
    if parts & _EXCLUDED_PARTS:
        return False
    if any(part.startswith("ple") or "shared_expert" in part for part in parts):
        return False
    return True


def routed_expert_parameters(
    module: torch.nn.Module,
    local_num_experts: int,
) -> dict[str, torch.nn.Parameter]:
    """Routed-expert parameters on ``module``, in registration order."""
    selected: dict[str, torch.nn.Parameter] = {}
    for name, param in module.named_parameters(recurse=False):
        if is_routed_expert_parameter(name, param, local_num_experts):
            selected[name] = param
    return selected


def read_popularity_profile(path: str) -> list[tuple[int, int]]:
    """Read a hottest-first ``(layer, global_expert)`` ranking.

    Accepts a JSON object ``{"ranked": [[layer, expert], ...]}`` or a
    text file with one ``layer expert`` pair per line (``#`` comments
    and a trailing count are ignored). File order is the popularity
    order. This is not Strata's ``STRP`` container.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file has no usable pairs.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"expert popularity profile not found: {path}")
    text = open(path, encoding="utf-8").read()
    ranked: list[tuple[int, int]] = []
    stripped = text.lstrip()
    if stripped.startswith("{"):
        payload = json.loads(text)
        pairs = payload.get("ranked", [])
        for pair in pairs:
            if len(pair) < 2:
                continue
            ranked.append((int(pair[0]), int(pair[1])))
    else:
        for line in text.splitlines():
            body = line.split("#", 1)[0].strip()
            if not body:
                continue
            parts = body.split()
            if len(parts) < 2:
                continue
            ranked.append((int(parts[0]), int(parts[1])))
    if not ranked:
        raise ValueError(f"expert popularity profile {path} has no pairs")
    return ranked


def owned_local_expert(
    expert_map: torch.Tensor | None,
    global_expert: int,
    local_num_experts: int,
) -> int | None:
    """Map a global expert id to this rank's local id, or None."""
    if local_num_experts <= 0 or global_expert < 0:
        return None
    if expert_map is None:
        if global_expert < local_num_experts:
            return global_expert
        return None
    if global_expert >= expert_map.shape[0]:
        return None
    local = int(expert_map[global_expert].item())
    if local < 0 or local >= local_num_experts:
        return None
    return local


def profile_rank_for_layer(
    ranked: Iterable[tuple[int, int]],
    layer_index: int,
    expert_map: torch.Tensor | None,
    local_num_experts: int,
) -> dict[int, int]:
    """Local-expert popularity rank on this rank (0 is hottest)."""
    order: dict[int, int] = {}
    for layer, global_expert in ranked:
        if layer != layer_index:
            continue
        local = owned_local_expert(expert_map, global_expert, local_num_experts)
        if local is None or local in order:
            continue
        order[local] = len(order)
    return order


class SideStreamCopier:
    """Copy one expert row from pinned host memory onto a side stream.

    The copy is not recorded into a HIP/CUDA graph. Capture must be
    refused by the caller before ``copy_expert`` runs; the method also
    refuses ``capturing=True``.
    """

    def __init__(self) -> None:
        self._streams: dict[str, torch.cuda.Stream] = {}
        self.copies = 0

    def copy_expert(
        self,
        host: torch.Tensor,
        slots: torch.Tensor,
        expert_id: int,
        slot: int,
        *,
        capturing: bool,
    ) -> None:
        if capturing:
            raise RuntimeError(
                "RDNA expert host pages stay outside hipGraph capture"
            )
        self.copies += 1
        destination = slots[slot]
        source = host[expert_id].detach()
        if destination.device.type != "cuda":
            destination.copy_(source)
            return
        key = str(destination.device)
        stream = self._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=destination.device)
            self._streams[key] = stream
        current = torch.cuda.current_stream(destination.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            destination.copy_(source, non_blocking=True)
        current.wait_stream(stream)


def pinned_host_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous CPU copy, pinned when the CUDA runtime can pin it."""
    host = tensor.detach().to("cpu", copy=True).contiguous()
    if host.is_pinned() or not torch.cuda.is_available():
        return host
    try:
        return host.pin_memory()
    except (RuntimeError, AssertionError):
        return host


def rebind_tensor_aliases(
    root: object,
    replacements: Mapping[int, torch.Tensor],
) -> None:
    """Point attributes that still hold ``old`` tensors at ``new`` ones."""
    seen: set[int] = set()

    def walk(obj: object, depth: int) -> None:
        if obj is None or depth > 12 or id(obj) in seen:
            return
        if isinstance(obj, (str, bytes, int, float, torch.Tensor)):
            return
        seen.add(id(obj))
        items: list[tuple[Any, Any, Any]] = []
        mapping = getattr(obj, "__dict__", None)
        if isinstance(mapping, dict):
            items.extend((mapping, key, value) for key, value in list(mapping.items()))
        if isinstance(obj, dict):
            items.extend((obj, key, value) for key, value in list(obj.items()))
        elif isinstance(obj, list):
            items.extend((obj, index, value) for index, value in enumerate(obj))
        for container, key, value in items:
            if isinstance(value, torch.Tensor) and id(value) in replacements:
                container[key] = replacements[id(value)]
            else:
                walk(value, depth + 1)

    walk(root, 0)


def _default_device_type() -> str:
    getter = getattr(torch, "get_default_device", None)
    if getter is None:
        return "cpu"
    try:
        device = getter()
    except Exception:
        return "cpu"
    if device is None:
        return "cpu"
    return str(device.type)


def _quant_method_is_monolithic(quant_method: object) -> bool:
    try:
        return bool(quant_method.is_monolithic)  # type: ignore[attr-defined]
    except Exception:
        return False


def _eplb_enabled(quant_method: object) -> bool:
    moe = getattr(quant_method, "moe", None)
    parallel = getattr(moe, "moe_parallel_config", None)
    return bool(getattr(parallel, "enable_eplb", False))


@contextmanager
def routed_expert_cpu_weight_context(quant_method: object):
    """Place new routed-expert parameters on CPU when offload is enabled.

    Meta initialization is left alone. Monolithic kernels and EPLB keep
    the default device: their routing is not visible as ``topk_ids``, so
    this cache cannot fetch for them. The full-weight resident flags
    also keep the default device.
    """
    if not expert_dram_offload_enabled():
        yield
        return
    if _default_device_type() in ("meta", "cpu"):
        yield
        return
    if _quant_method_is_monolithic(quant_method) or _eplb_enabled(quant_method):
        yield
        return
    with torch.device("cpu"):
        yield


class ExpertResidencyCache:
    """Per-rank hot-expert table for one routed MoE layer.

    ``slot_of[local_expert]`` is the VRAM slot, or -1 when the expert is
    only in the pinned host pages. A miss copies every expert-major
    parameter of that expert into one slot on the side stream, then the
    caller remaps ``topk_ids`` onto those slots.
    """

    def __init__(
        self,
        *,
        layer_index: int,
        local_num_experts: int,
        host_pages: dict[str, torch.Tensor],
        slot_params: dict[str, torch.Tensor],
        n_slots: int,
        profile_rank: dict[int, int],
        copier: SideStreamCopier | None = None,
    ) -> None:
        if n_slots <= 0:
            raise ValueError("residency cache needs at least one slot")
        self.layer_index = layer_index
        self.local_num_experts = local_num_experts
        self.host_pages = host_pages
        self.slot_params = slot_params
        self.n_slots = n_slots
        self.profile_rank = profile_rank
        self.copier = copier or SideStreamCopier()
        self.heat = [0] * local_num_experts
        self.slot_expert = [-1] * n_slots
        self.free_slots = list(range(n_slots))
        device = next(iter(slot_params.values())).device
        self.slot_of = torch.full(
            (local_num_experts,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._slot_of_cpu = [-1] * local_num_experts
        self.engaged = True

    @property
    def resident_experts(self) -> list[int]:
        return [expert for expert, slot in enumerate(self._slot_of_cpu) if slot >= 0]

    def seed(self, local_experts: Iterable[int]) -> None:
        """Admit ``local_experts`` in order while free slots remain."""
        for expert in local_experts:
            if not self.free_slots:
                return
            self.ensure(int(expert), capturing=False)

    def ensure(self, local_expert: int, *, capturing: bool) -> None:
        """Make ``local_expert`` resident, fetching from host on a miss."""
        if self._slot_of_cpu[local_expert] >= 0:
            return
        if capturing:
            raise RuntimeError(
                "RDNA expert host pages stay outside hipGraph capture; "
                f"layer {self.layer_index} expert {local_expert} is not resident"
            )
        if self.free_slots:
            slot = self.free_slots.pop(0)
        else:
            slot = self._evict_coldest()
        self._copy_all(local_expert, slot, capturing=False)
        self._publish(local_expert, slot)

    def prepare(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        *,
        capturing: bool | None = None,
    ) -> torch.Tensor:
        """Return ``topk_ids`` remapped to resident slots.

        Args:
            topk_ids: Global expert ids, with negatives for padding.
            expert_map: Global-to-local map for this rank, or None when
                every id is already local.
            capturing: Force the hipGraph-capture refusal in tests.

        Returns:
            Slot ids. Non-local and padding rows stay -1.

        Raises:
            RuntimeError: A miss was required while a graph is capturing.
        """
        if capturing is None:
            capturing = _stream_is_capturing(self.slot_of.device)
        # Replay would route experts this capture never fetched. Refuse
        # the whole capturing forward so host pages cannot be recorded.
        if capturing:
            raise RuntimeError(
                "RDNA expert host pages stay outside hipGraph capture. "
                "Serve this offload eagerly so a miss is fetched from "
                "pinned host memory on the side stream."
            )
        local = local_ids_from_topk(
            topk_ids,
            expert_map,
            self.local_num_experts,
        )
        self._touch(local)
        safe = local.clamp(min=0).to(torch.long)
        slots = self.slot_of[safe]
        remapped = torch.where(local < 0, local, slots.to(local.dtype))
        return remapped

    def _touch(self, local: torch.Tensor) -> None:
        if local.numel() == 0:
            return
        for expert in torch.unique(local).tolist():
            expert_id = int(expert)
            if expert_id < 0:
                continue
            self.heat[expert_id] += 1
            self.ensure(expert_id, capturing=False)

    def _cold_key(self, local_expert: int) -> tuple[int, int]:
        # Smaller key is evicted. Higher heat stays. Profile rank 0 is
        # hottest, so a large rank (or none) sorts first.
        rank = self.profile_rank.get(local_expert, 10**9)
        return (self.heat[local_expert], -rank)

    def _evict_coldest(self) -> int:
        resident = self.resident_experts
        if not resident:
            raise RuntimeError("expert residency cache has no slot to evict")
        victim = min(resident, key=self._cold_key)
        slot = self._slot_of_cpu[victim]
        self._slot_of_cpu[victim] = -1
        self.slot_of[victim] = -1
        self.slot_expert[slot] = -1
        return slot

    def _copy_all(self, local_expert: int, slot: int, *, capturing: bool) -> None:
        for name, host in self.host_pages.items():
            self.copier.copy_expert(
                host,
                self.slot_params[name],
                local_expert,
                slot,
                capturing=capturing,
            )

    def _publish(self, local_expert: int, slot: int) -> None:
        self._slot_of_cpu[local_expert] = slot
        self.slot_of[local_expert] = slot
        self.slot_expert[slot] = local_expert


def local_ids_from_topk(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_num_experts: int,
) -> torch.Tensor:
    """Map global ``topk_ids`` to local ids, else -1."""
    invalid = torch.full_like(topk_ids, -1)
    if expert_map is None:
        valid = (topk_ids >= 0) & (topk_ids < local_num_experts)
        return torch.where(valid, topk_ids, invalid)
    valid = (topk_ids >= 0) & (topk_ids < expert_map.shape[0])
    gathered = invalid.clone()
    if bool(valid.any().item()):
        gathered[valid] = expert_map[topk_ids[valid]].to(gathered.dtype)
    owned = (gathered >= 0) & (gathered < local_num_experts)
    return torch.where(owned, gathered, invalid)


def _stream_is_capturing(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _gpu_memory_utilization() -> float:
    try:
        from vllm.config import get_current_vllm_config_or_none
    except Exception:
        return 0.9
    config = get_current_vllm_config_or_none()
    if config is None:
        return 0.9
    return float(config.cache_config.gpu_memory_utilization)


def _replace_parameter(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    from vllm.model_executor.utils import replace_parameter

    replace_parameter(module, name, tensor)


def _move_parameter_to_device(param: torch.nn.Parameter, device: torch.device) -> None:
    if param.device == torch.device(device):
        return
    param.data = param.data.to(device)


def install_expert_residency(
    layers: list[tuple[torch.nn.Module, int]],
    *,
    enabled: bool,
    vram_budget: int,
    profile: list[tuple[int, int]] | None,
    device: torch.device,
    copier: SideStreamCopier | None = None,
    slot_budget: int | None = None,
) -> bool:
    """Install per-layer caches, or leave weights in place when they fit.

    Args:
        layers: ``(module, fallback_layer_index)`` pairs. The module must
            expose ``local_num_experts`` and may expose ``expert_map``.
        enabled: Master switch.
        vram_budget: Physical bytes the full expert set is compared
            against. The cache stays dead when the experts fit here.
        slot_budget: Bytes of resident slots to allocate when the full
            set does not fit. Defaults to ``vram_budget``.
        profile: Hottest-first ``(layer, global_expert)`` pairs, or None.
        device: Device that should hold resident slots.
        copier: Host-to-slot copier. Tests pass one to count fetches.

    Returns:
        True when a cache was installed. False when offload stays dead:
        the switch is off, or the experts fit in ``vram_budget``.
    """
    if not enabled or not layers:
        return False
    specs: list[dict[str, Any]] = []
    expert_bytes = 0
    for module, fallback in layers:
        local_num = int(module.local_num_experts)
        params = routed_expert_parameters(module, local_num)
        if not params:
            continue
        if any(param.is_meta for param in params.values()):
            continue
        nbytes = sum(param.numel() * param.element_size() for param in params.values())
        per_expert = nbytes // local_num
        name = getattr(module, "layer_name", "") or ""
        specs.append(
            {
                "module": module,
                "layer_index": layer_index_from_name(name, fallback),
                "local_num": local_num,
                "params": params,
                "per_expert": per_expert,
                "expert_map": getattr(module, "expert_map", None),
            }
        )
        expert_bytes += nbytes
    if not specs:
        return False
    if not should_engage_expert_offload(
        enabled=True,
        expert_bytes=expert_bytes,
        vram_budget=vram_budget,
    ):
        for spec in specs:
            for param in spec["params"].values():
                _move_parameter_to_device(param, device)
        logger.info_once(
            "Routed experts fit in VRAM (%d bytes <= %d byte budget). "
            "Expert DRAM offload stays inactive.",
            expert_bytes,
            max(int(vram_budget), 0),
        )
        return False

    resident_budget = vram_budget if slot_budget is None else slot_budget
    slots = assign_slots(
        [int(spec["per_expert"]) for spec in specs],
        [int(spec["local_num"]) for spec in specs],
        max(int(resident_budget), 0),
    )
    shared_copier = copier or SideStreamCopier()
    for spec, n_slots in zip(specs, slots, strict=True):
        _install_layer_cache(spec, n_slots, profile or [], device, shared_copier)
    logger.info_once(
        "RDNA expert DRAM offload engaged for %d routed MoE layer(s); "
        "%d expert bytes exceed the %d byte VRAM budget. "
        "hipGraph capture cannot fetch host expert pages; serve eagerly.",
        len(specs),
        expert_bytes,
        max(int(vram_budget), 0),
    )
    return True


def _install_layer_cache(
    spec: dict[str, Any],
    n_slots: int,
    profile: list[tuple[int, int]],
    device: torch.device,
    copier: SideStreamCopier,
) -> None:
    module: torch.nn.Module = spec["module"]
    params: dict[str, torch.nn.Parameter] = spec["params"]
    host_pages = {name: pinned_host_copy(param) for name, param in params.items()}
    slot_params: dict[str, torch.Tensor] = {}
    replacements: dict[int, torch.Tensor] = {}
    old_params = dict(params)
    for name, param in params.items():
        slots = torch.zeros(
            (n_slots, *param.shape[1:]),
            dtype=param.dtype,
            device=device,
        )
        _replace_parameter(module, name, slots)
        new_param = module._parameters[name]
        slot_params[name] = new_param
        replacements[id(old_params[name])] = new_param
    rebind_tensor_aliases(module, replacements)
    rank = profile_rank_for_layer(
        profile,
        int(spec["layer_index"]),
        spec["expert_map"],
        int(spec["local_num"]),
    )
    cache = ExpertResidencyCache(
        layer_index=int(spec["layer_index"]),
        local_num_experts=int(spec["local_num"]),
        host_pages=host_pages,
        slot_params=slot_params,
        n_slots=n_slots,
        profile_rank=rank,
        copier=copier,
    )
    hot = sorted(rank, key=lambda expert: rank[expert])
    cache.seed(hot)
    module._rdna_expert_residency = cache  # type: ignore[attr-defined]


def _module_skipped(module: Any) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None and _quant_method_is_monolithic(quant_method):
        return True
    parallel = getattr(getattr(module, "moe_config", None), "moe_parallel_config", None)
    return bool(getattr(parallel, "enable_eplb", False))


def finalize_routed_expert_residency(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Build the cache after weights are packed, or leave the model alone.

    Called from the serve-side loader. Default off returns before any
    expert tensor is inspected. When the experts fit, CPU-staged routed
    weights are moved onto ``device`` and no host pages are retained.
    """
    if not expert_dram_offload_requested():
        return
    if not expert_dram_offload_enabled():
        return
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    restore: list[RoutedExperts] = []
    layers: list[tuple[torch.nn.Module, int]] = []
    for index, module in enumerate(model.modules()):
        if not isinstance(module, RoutedExperts):
            continue
        if _module_skipped(module):
            restore.append(module)
            continue
        layers.append((module, index))
    for module in restore:
        local_num = int(module.local_num_experts)
        for param in routed_expert_parameters(module, local_num).values():
            _move_parameter_to_device(param, device)
    if not layers:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        for module, _index in layers:
            local_num = int(module.local_num_experts)
            for param in routed_expert_parameters(module, local_num).values():
                _move_parameter_to_device(param, device)
        logger.warning_once(
            "VLLM_RDNA_MOE_EXPERT_OFFLOAD is set, but %s has no CUDA "
            "device. Expert DRAM offload stays inactive.",
            device,
        )
        return
    on_device = 0
    for module, _index in layers:
        for param in routed_expert_parameters(
            module, int(module.local_num_experts)
        ).values():
            if param.device.type == device.type:
                on_device += param.numel() * param.element_size()
    total = int(torch.cuda.get_device_properties(device).total_memory)
    allocated = int(torch.cuda.memory_allocated(device))
    physical = room_for_experts(
        total=total,
        allocated=allocated,
        expert_bytes_on_device=on_device,
        utilization=1.0,
    )
    slot_budget = room_for_experts(
        total=total,
        allocated=allocated,
        expert_bytes_on_device=on_device,
        utilization=_gpu_memory_utilization(),
    )
    profile_path = envs.VLLM_RDNA_MOE_EXPERT_PROFILE
    profile = read_popularity_profile(profile_path) if profile_path else None
    engaged = install_expert_residency(
        layers,
        enabled=True,
        vram_budget=physical,
        slot_budget=slot_budget,
        profile=profile,
        device=device,
    )
    if engaged and device.type == "cuda" and torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
