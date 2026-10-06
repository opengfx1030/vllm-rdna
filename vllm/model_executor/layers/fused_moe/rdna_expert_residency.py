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
the head, norms, the router, MoVA V-experts, and PLE / n-gram / Engram
tables are not expert pages. The host copy is a pinned CPU buffer
fetched with an async copy on a side stream (hipMemcpyAsync, or
``copy_`` on that stream). A resident GEMM does not wait for that
copy. The next routed MoE layer — skipping dense, linear, GDN, KDA,
and QSA-only blocks — waits on an event recorded after the copy, and
only if it will read those bytes. The copy lands in a second slot bank
so it does not overwrite a slot the in-flight GEMM is still reading.
The fetch is refused while a HIP/CUDA graph is capturing. Device slots
stay on the routed-expert module, not on GDN, QSA, PLE, or hc_combine
workspace. Under tensor or expert parallel each rank prefetches only
its own local experts.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
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
        "mova",
        "mova_v",
        "v_expert",
        "v_experts",
    }
)

_SKIP_LAYER_KINDS = frozenset(
    {
        "dense",
        "linear",
        "gdn",
        "kda",
        "qsa",
        "qsa_only",
        "shared",
        "shared_expert",
        "shared_experts",
        "mova",
        "mova_v",
        "v_expert",
        "v_experts",
    }
)
_MOE_LAYER_KINDS = frozenset({"moe", "routed", "routed_moe"})


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


def _layer_kind(kind: str) -> str:
    return kind.lower().replace("-", "_")


def next_moe_index(kinds: Sequence[str], index: int) -> int | None:
    """Index of the next routed MoE layer after ``index``.

    The successor is not ``index + 1`` when that neighbor is dense,
    linear, GDN, KDA, QSA-only, a shared expert, or a MoVA V-expert.
    """
    if index < 0 or index >= len(kinds):
        return None
    if _layer_kind(kinds[index]) not in _MOE_LAYER_KINDS:
        return None
    for nxt in range(index + 1, len(kinds)):
        if _layer_kind(kinds[nxt]) in _MOE_LAYER_KINDS:
            return nxt
    return None


def is_prefetch_moe_module(module: object) -> bool:
    """False for shared, MoVA V-expert, GDN, KDA, and QSA-only blocks."""
    label = " ".join(
        (
            str(getattr(module, "layer_name", "") or ""),
            type(module).__name__,
        )
    )
    parts = set(_layer_kind(label).replace(" ", ".").split("."))
    for part in parts:
        if part in _SKIP_LAYER_KINDS or part.startswith("qsa"):
            return False
        if part.startswith("mova") or "shared_expert" in part:
            return False
        if "v_expert" in part:
            return False
    return True


def module_layer_kind(module: object) -> str:
    """Kind used to find the next routed MoE layer."""
    if is_prefetch_moe_module(module):
        return "moe"
    label = " ".join(
        (
            str(getattr(module, "layer_name", "") or ""),
            type(module).__name__,
        )
    )
    parts = set(_layer_kind(label).replace(" ", ".").split("."))
    for kind in ("gdn", "kda", "qsa", "mova", "shared", "linear"):
        if kind in parts or any(part.startswith(kind) for part in parts):
            return "qsa" if kind == "qsa" else kind
    return "dense"


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
    if any(
        part.startswith("ple")
        or part.startswith("mova")
        or part.startswith("qsa")
        or part in {"gdn", "kda"}
        or "shared_expert" in part
        or "v_expert" in part
        for part in parts
    ):
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


class _CopyTicket:
    """Completion of one side-stream copy."""

    def wait(self, *, count: bool) -> None:
        """Make the compute stream wait for this copy.

        ``count`` records a consumer wait. Load-time seeding passes
        False so a hot GEMM is not charged for it.
        """
        raise NotImplementedError


class _ReadyTicket(_CopyTicket):
    def __init__(self, copier: "SideStreamCopier") -> None:
        self._copier = copier
        self.done = True

    def wait(self, *, count: bool) -> None:
        if count:
            self._copier.compute_waits += 1
            self._copier.log.append("wait")


class _EventTicket(_CopyTicket):
    def __init__(
        self,
        copier: "SideStreamCopier",
        event: torch.cuda.Event,
        device: torch.device,
    ) -> None:
        self._copier = copier
        self._event = event
        self._device = device
        self.done = False

    def wait(self, *, count: bool) -> None:
        if not self.done:
            torch.cuda.current_stream(self._device).wait_event(self._event)
            self.done = True
        if count:
            self._copier.compute_waits += 1
            self._copier.log.append("wait")


class SideStreamCopier:
    """Async host-to-device copies of pinned expert rows.

    Copies run on a side stream and record an event. Nothing here waits
    on the compute stream, and the compute stream is not told to wait
    unless a later consumer calls ``ticket.wait``. A blit or a stream
    sync on the compute queue would serialize the copy with fdot2.
    """

    def __init__(self) -> None:
        self._streams: dict[str, torch.cuda.Stream] = {}
        self.copies = 0
        self.compute_waits = 0
        self.stream_syncs = 0
        self.log: list[str] = []
        self.experts: list[int] = []
        self.dest_ptrs: list[int] = []

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        key = str(device)
        stream = self._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._streams[key] = stream
        return stream

    def copy_rows(
        self,
        rows: list[tuple[torch.Tensor, torch.Tensor, int, int]],
        *,
        capturing: bool,
    ) -> _CopyTicket:
        """Copy expert rows from pinned host pages onto ``slots[slot]``.

        Each row is ``(host, slots, expert_id, slot)``. On CUDA this is
        ``copy_(non_blocking=True)`` on the side stream, which is the
        hipMemcpyAsync path. The compute stream is not synchronized.
        """
        if capturing:
            raise RuntimeError(
                "RDNA expert host pages stay outside hipGraph capture"
            )
        if not rows:
            return _ReadyTicket(self)
        self.copies += len(rows)
        self.log.append("h2d")
        expert_id = int(rows[0][2])
        self.experts.append(expert_id)
        self.dest_ptrs.append(int(rows[0][1].data_ptr()))
        device = rows[0][1].device
        if device.type == "cuda" and torch.cuda.is_available():
            stream = self._stream(device)
            with torch.cuda.stream(stream):
                for host, slots, row_expert, slot in rows:
                    slots[slot].copy_(host[row_expert].detach(), non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
            return _EventTicket(self, event, device)
        for host, slots, row_expert, slot in rows:
            slots[slot].copy_(host[row_expert].detach())
        return _ReadyTicket(self)

    def copy_bound_rows(
        self,
        src: Mapping[str, torch.Tensor],
        src_slot: int,
        dst: Mapping[str, torch.Tensor],
        dst_slot: int,
        names: Sequence[str],
    ) -> _CopyTicket:
        """Publish a finished row onto the bank the next GEMM will read.

        The copy stays on the side stream. It is not a compute-queue blit.
        """
        self.log.append("bind")
        device = dst[names[0]].device
        if device.type == "cuda" and torch.cuda.is_available():
            stream = self._stream(device)
            with torch.cuda.stream(stream):
                for name in names:
                    dst[name][dst_slot].copy_(
                        src[name][src_slot].detach(),
                        non_blocking=True,
                    )
                event = torch.cuda.Event()
                event.record(stream)
            return _EventTicket(self, event, device)
        for name in names:
            dst[name][dst_slot].copy_(src[name][src_slot].detach())
        return _ReadyTicket(self)


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
    only in the pinned host pages. A hit on a resident expert does not
    wait. A miss for the next MoE layer is prefetched into the staging
    bank on the side stream while this layer's GEMM runs. The consumer
    GEMM waits on the copy event before it reads that slot.
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
        self.successor: ExpertResidencyCache | None = None
        self.last_routed_local: list[int] = []
        self.bank_id_bound = 0
        self.banks = [
            {name: param.data for name, param in slot_params.items()},
            {
                name: torch.zeros_like(param.data)
                for name, param in slot_params.items()
            },
        ]
        self.bank_expert = [[-1] * n_slots, [-1] * n_slots]
        self._staged: dict[int, int] = {}
        self._staged_ticket: dict[int, _CopyTicket] = {}
        # (event or False, storage ids). False is held until released.
        self._live: list[tuple[Any, set[int]]] = []

    @property
    def resident_experts(self) -> list[int]:
        return [expert for expert, slot in enumerate(self._slot_of_cpu) if slot >= 0]

    def bound_storage_ids(self) -> set[int]:
        bank = self.banks[self.bank_id_bound]
        return {int(tensor.data_ptr()) for tensor in bank.values()}

    def staging_storage_ids(self) -> set[int]:
        bank = self.banks[1 - self.bank_id_bound]
        return {int(tensor.data_ptr()) for tensor in bank.values()}

    def hold_live(self, storage_ids: Iterable[int]) -> None:
        """Keep these bytes until the in-flight GEMM event is released."""
        self._live.append((False, {int(item) for item in storage_ids}))

    def seed(self, local_experts: Iterable[int]) -> None:
        """Admit ``local_experts`` in order while free slots remain."""
        for expert in local_experts:
            if not self.free_slots:
                return
            self._fetch_into_bound(int(expert), count_wait=False)

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
        self._refresh_live()
        local = local_ids_from_topk(
            topk_ids,
            expert_map,
            self.local_num_experts,
        )
        needed = self._needed_locals(local)
        # Only ids this rank owns. Another rank's experts stay -1.
        self.last_routed_local = list(needed)
        for expert_id in needed:
            self.heat[expert_id] += 1
            if self._slot_of_cpu[expert_id] >= 0:
                continue
            if expert_id in self._staged:
                self._consume_staged(expert_id, set(needed))
                continue
            self._fetch_into_bound(expert_id, count_wait=True)
        safe = local.clamp(min=0).to(torch.long)
        slots = self.slot_of[safe]
        remapped = torch.where(local < 0, local, slots.to(local.dtype))
        return remapped

    def after_resident_gemm(self) -> None:
        """Record the GEMM event, then prefetch the next MoE layer.

        The compute stream is not synchronized. The copy overlaps this
        layer's already-launched resident GEMM.
        """
        if self.slot_of.device.type == "cuda" and torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.slot_of.device))
            self._live.append((event, self.bound_storage_ids()))
        self.prefetch_successor()

    def prefetch_local_ids(self) -> list[int]:
        """Local routed experts to stage for this rank.

        Uses the last routed set when this layer has run before, else
        the profile order. Experts already in the bound bank are hits,
        not misses. Ids outside this rank's local range are dropped.
        """
        if self.last_routed_local:
            raw: Iterable[int] = self.last_routed_local
        else:
            raw = sorted(
                self.profile_rank,
                key=lambda expert: self.profile_rank[expert],
            )
        misses: list[int] = []
        for expert in raw:
            expert_id = int(expert)
            if expert_id < 0 or expert_id >= self.local_num_experts:
                continue
            if self._slot_of_cpu[expert_id] >= 0 or expert_id in self._staged:
                continue
            misses.append(expert_id)
            if len(misses) >= self.n_slots:
                break
        return misses

    def prefetch_successor(self) -> None:
        """Stage the next MoE layer's misses. Do not wait."""
        nxt = self.successor
        if nxt is None:
            return
        if _stream_is_capturing(nxt.slot_of.device):
            raise RuntimeError(
                "RDNA expert host pages stay outside hipGraph capture. "
                "Serve this offload eagerly so a miss is fetched from "
                "pinned host memory on the side stream."
            )
        for expert in nxt.prefetch_local_ids():
            nxt.stage_prefetch(expert)

    def stage_prefetch(self, local_expert: int) -> None:
        """Async H2D into the staging bank. Never writes a live slot."""
        if local_expert < 0 or local_expert >= self.local_num_experts:
            return
        if self._slot_of_cpu[local_expert] >= 0 or local_expert in self._staged:
            return
        self._refresh_live()
        slot = self._take_staging_slot(protect=set())
        if slot is None:
            return
        bank = 1 - self.bank_id_bound
        ticket = self._copy_host(local_expert, bank, slot, capturing=False)
        self.bank_expert[bank][slot] = local_expert
        self._staged[local_expert] = slot
        self._staged_ticket[local_expert] = ticket

    def _needed_locals(self, local: torch.Tensor) -> list[int]:
        if local.numel() == 0:
            return []
        needed: list[int] = []
        for expert in torch.unique(local).tolist():
            expert_id = int(expert)
            if expert_id < 0 or expert_id >= self.local_num_experts:
                continue
            needed.append(expert_id)
        return needed

    def _refresh_live(self) -> None:
        kept: list[tuple[Any, set[int]]] = []
        for event, ids in self._live:
            if event is False:
                kept.append((event, ids))
                continue
            if event is None:
                continue
            try:
                done = bool(event.query())
            except Exception:
                done = False
            if not done:
                kept.append((event, ids))
        self._live = kept

    def _bank_is_live(self, bank: int) -> bool:
        self._refresh_live()
        ids = {int(tensor.data_ptr()) for tensor in self.banks[bank].values()}
        return any(bool(ids & live) for _event, live in self._live)

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
        self.bank_expert[self.bank_id_bound][slot] = -1
        self._slot_of_cpu[victim] = -1
        self.slot_of[victim] = -1
        self.slot_expert[slot] = -1
        return slot

    def _take_bound_slot(self) -> int:
        if self.free_slots:
            return self.free_slots.pop(0)
        return self._evict_coldest()

    def _take_staging_slot(self, protect: set[int]) -> int | None:
        bank = 1 - self.bank_id_bound
        if self._bank_is_live(bank):
            return None
        for slot, expert in enumerate(self.bank_expert[bank]):
            if expert < 0:
                return slot
        victims = [
            expert
            for expert in self.bank_expert[bank]
            if expert >= 0 and expert not in protect
        ]
        if not victims:
            return None
        victim = min(victims, key=self._cold_key)
        slot = self.bank_expert[bank].index(victim)
        self.bank_expert[bank][slot] = -1
        self._staged.pop(victim, None)
        self._staged_ticket.pop(victim, None)
        return slot

    def _copy_host(
        self,
        local_expert: int,
        bank: int,
        slot: int,
        *,
        capturing: bool,
    ) -> _CopyTicket:
        rows = [
            (host, self.banks[bank][name], local_expert, slot)
            for name, host in self.host_pages.items()
        ]
        return self.copier.copy_rows(rows, capturing=capturing)

    def _fetch_into_bound(self, local_expert: int, *, count_wait: bool) -> None:
        if self._slot_of_cpu[local_expert] >= 0:
            return
        if self._bank_is_live(self.bank_id_bound):
            self.stage_prefetch(local_expert)
            if local_expert not in self._staged:
                raise RuntimeError(
                    "expert residency cannot place a miss without "
                    "aliasing the in-flight GEMM slot"
                )
            self._consume_staged(local_expert, {local_expert})
            return
        slot = self._take_bound_slot()
        ticket = self._copy_host(
            local_expert,
            self.bank_id_bound,
            slot,
            capturing=False,
        )
        ticket.wait(count=count_wait)
        self._publish(local_expert, slot)

    def _consume_staged(self, local_expert: int, needed: set[int]) -> None:
        self._wait_staged(needed)
        if self._bank_is_live(self.bank_id_bound):
            self._mirror_needed_onto_staging(needed)
            self._swap_bound_to(1 - self.bank_id_bound)
            return
        self._bind_one(local_expert)

    def _wait_staged(self, experts: set[int]) -> None:
        for expert in experts:
            ticket = self._staged_ticket.pop(expert, None)
            if ticket is not None:
                ticket.wait(count=True)

    def _bind_one(self, local_expert: int) -> None:
        src = self._staged.pop(local_expert)
        staging = 1 - self.bank_id_bound
        self.bank_expert[staging][src] = -1
        slot = self._take_bound_slot()
        ticket = self.copier.copy_bound_rows(
            self.banks[staging],
            src,
            self.banks[self.bank_id_bound],
            slot,
            list(self.host_pages),
        )
        ticket.wait(count=True)
        self._publish(local_expert, slot)

    def _mirror_needed_onto_staging(self, needed: set[int]) -> None:
        staging = 1 - self.bank_id_bound
        for expert in needed:
            if expert in self._staged:
                continue
            src = self._slot_of_cpu[expert]
            if src < 0:
                continue
            dest = self._take_staging_slot(protect=needed)
            if dest is None:
                raise RuntimeError(
                    "expert residency cannot mirror a live expert without "
                    "aliasing the in-flight GEMM slot"
                )
            ticket = self.copier.copy_bound_rows(
                self.banks[self.bank_id_bound],
                src,
                self.banks[staging],
                dest,
                list(self.host_pages),
            )
            ticket.wait(count=False)
            self.bank_expert[staging][dest] = expert
            self._staged[expert] = dest

    def _swap_bound_to(self, new_bank: int) -> None:
        for name, param in self.slot_params.items():
            param.data = self.banks[new_bank][name]
        self.bank_id_bound = new_bank
        self._staged.clear()
        self._staged_ticket.clear()
        self.free_slots = []
        self._slot_of_cpu = [-1] * self.local_num_experts
        self.slot_expert = [-1] * self.n_slots
        self.slot_of.fill_(-1)
        for slot, expert in enumerate(self.bank_expert[new_bank]):
            if expert < 0:
                self.free_slots.append(slot)
                continue
            self._slot_of_cpu[expert] = slot
            self.slot_of[expert] = slot
            self.slot_expert[slot] = expert

    def _publish(self, local_expert: int, slot: int) -> None:
        bank = self.bank_id_bound
        previous = self.bank_expert[bank][slot]
        if previous >= 0 and previous != local_expert:
            self._slot_of_cpu[previous] = -1
            self.slot_of[previous] = -1
        old = self._slot_of_cpu[local_expert]
        if old >= 0 and old != slot:
            self.bank_expert[bank][old] = -1
            self.slot_expert[old] = -1
            if old not in self.free_slots:
                self.free_slots.append(old)
        self.bank_expert[bank][slot] = local_expert
        self._slot_of_cpu[local_expert] = slot
        self.slot_of[local_expert] = slot
        self.slot_expert[slot] = local_expert
        if slot in self.free_slots:
            self.free_slots.remove(slot)


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
    caches: list[ExpertResidencyCache] = []
    kinds: list[str] = []
    for spec, n_slots in zip(specs, slots, strict=True):
        caches.append(
            _install_layer_cache(spec, n_slots, profile or [], device, shared_copier)
        )
        kinds.append(module_layer_kind(spec["module"]))
    for index, cache in enumerate(caches):
        nxt = next_moe_index(kinds, index)
        cache.successor = caches[nxt] if nxt is not None else None
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
) -> ExpertResidencyCache:
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
    return cache


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
