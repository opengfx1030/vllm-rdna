# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, always-on-cheap expert load counters for expert-parallel balancing.

``VLLM_RDNA_EPLB_STATS_DIR=<dir>`` turns the counters on (unset or empty: off, and
the only cost left in the hot path is one ``is not None`` test per MoE call). They
answer the questions an expert-parallel load balancer needs before anything is
moved, per MoE layer and per rank of the placement in force:

* ``T_r``, the number of DISTINCT experts rank ``r`` touches in a step. The decode
  GEMM reads every touched expert's weights once, so this is the quantity decode
  time follows (``tools/rdna/eplb/objective.py`` fits it against kernel time);
* ``P_r``, the routed (row, expert) pairs rank ``r`` receives, which is what the
  compute-bound prefill GEMM follows;
* the per-expert totals behind both (``pairs``), the per-expert decode touch
  frequency per band of batch rows (``touch``), and a ring of the newest
  per-call records of every layer with an id hash that must agree across ranks.

Mechanism. The ids are the router's global expert ids, already identical on all
ranks (the router is replicated across the tensor-parallel group), so each rank
counts the whole call and derives every rank's load from its owner table; no
cross-rank exchange is needed. On the fused RDNA2 MoE path one extra CTA of the
existing routing launch (``rdna_moe_route_zero_stats``) builds the call's
histogram while CTA 0 routes: no extra launch and nothing on the routing critical
path. Calls the routing CTA does not take (more than 1024 routed entries, the
deterministic accumulator path, the resident skinny kernel) use the standalone
single-CTA ``rdna_eplb_stats``: one small launch per layer, amortised by a
prefill step. ``VLLM_RDNA_EPLB_STATS_KERNEL=torch`` selects the fixed-shape torch
formulation instead (about 60 tiny kernels per call): the CPU test oracle, the
build-free fallback and the parity reference of the HIP kernel, not for serving.

Counters are cumulative device tensors that are never reset, so a window is the
difference of two snapshots and there is no race with in-flight graphs. They are
allocated on the first eager call of a layer (vLLM's profile run precedes graph
capture); a layer that first runs inside a capture is skipped with a warning.
Draining is off the hot path: ``tick()`` is called once per step by the worker,
queues non-blocking copies into pinned buffers every
``VLLM_RDNA_EPLB_STATS_INTERVAL_S`` seconds (default 30) and writes the snapshot
when the event reports completion; ``close()`` writes a final one.

Output ``<dir>/eplb_stats-rank<R>-pid<P>.npz`` per rank (format ``eplb-stats/1``,
read by ``tools/rdna/eplb/summary.py``); the arrays and the ring record words are
specified there and in ``csrc/rocm/rdna_fused_glue.cu``.

Options (all optional):

* ``VLLM_RDNA_EPLB_STATS_RING`` records of the call ring (default 262144 = 32 MiB
  per rank, about 2 min of c=1 MTP2 decode; 0 = no ring);
* ``VLLM_RDNA_EPLB_STATS_HIST`` prefill histograms kept per layer (default 0; the
  per-call histograms feed ``placement.pack_empirical``);
* ``VLLM_RDNA_EPLB_STATS_EVERY`` count every N-th call of a layer (a power of two,
  default 1). The kernel-time join of ``objective.py`` needs 1;
* ``VLLM_RDNA_EPLB_STATS_DECODE_MAX`` largest decode call in valid rows (192);
* ``VLLM_RDNA_EPLB_STATS_KERNEL`` ``auto`` (HIP when built, else torch), ``hip`` or
  ``torch``.

Limits: one stream (the counters have a single writer per layer and per ring, so
two micro-batches on two streams would race: dual-batch overlap needs a ring per
micro-batch); at most 8 expert-parallel ranks; the Flash-Next ``--monokernel`` K2
and the expert-offload cache path do not route through ``_rdna2_fused_moe`` with
observable ids, so K2 stands down while the counters are on and the cache path is
not counted.
"""

import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import ExpertPlacementStrategy

logger = init_logger(__name__)

STATS_DIR_ENV = "VLLM_RDNA_EPLB_STATS_DIR"
# The online balancer (rdna_eplb_online) needs the counters but no snapshot files.
ONLINE_ENV = "VLLM_RDNA_EPLB"
RING_ENV = "VLLM_RDNA_EPLB_STATS_RING"
HIST_ENV = "VLLM_RDNA_EPLB_STATS_HIST"
EVERY_ENV = "VLLM_RDNA_EPLB_STATS_EVERY"
INTERVAL_ENV = "VLLM_RDNA_EPLB_STATS_INTERVAL_S"
DECODE_MAX_ENV = "VLLM_RDNA_EPLB_STATS_DECODE_MAX"
KERNEL_ENV = "VLLM_RDNA_EPLB_STATS_KERNEL"

FORMAT = "eplb-stats/1"
# These three equal kEplbWords / kEplbMaxRanks / kEplbMaxBands in
# csrc/rocm/rdna_fused_glue.cu (a test compares them).
WORDS = 32
MAX_RANKS = 8
MAX_BANDS = 32
# ``band b`` = valid rows in (edge[b-1], edge[b]]; the last edge is the decode limit.
DEFAULT_BAND_EDGES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
DEFAULT_DECODE_MAX = DEFAULT_BAND_EDGES[-1]
DEFAULT_RING = 262144
DEFAULT_INTERVAL_S = 30.0
MAX_HASH = 0xFFFFFFFF
HIST_CLIP = 32767

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_MTP = re.compile(r"(?:^|\.)mtp(?:\.|$)")


def layer_index(layer_name: str) -> int:
    """Decoder layer index parsed from a module name, or -1 when absent."""
    match = _LAYER_INDEX.search(layer_name)
    return int(match.group(1)) if match else -1


def layer_kind(layer_name: str) -> str:
    """``"mtp"`` for the draft model's layers, ``"target"`` otherwise."""
    return "mtp" if _MTP.search(layer_name) else "target"


def band_edges(decode_max: int) -> tuple[int, ...]:
    """The default bands cut at ``decode_max``, which becomes the last edge."""
    if not 1 <= decode_max <= 4096:
        raise ValueError(f"eplb stats decode limit must be 1..4096, got {decode_max}")
    return tuple(e for e in DEFAULT_BAND_EDGES if e < decode_max) + (decode_max,)


@dataclass(frozen=True)
class Offsets:
    """Word offsets of one layer's int64 ``state`` (see rdna_fused_glue.cu)."""

    touch: int  # touch[nb][E]
    calls: int  # calls[nb + 1]
    rows: int  # rows[2]
    ordinal: int  # calls of the layer so far, sampled or not
    hist_calls: int  # prefill histograms written
    hist_rows: int  # hist_rows[hist_n]
    size: int


def offsets(num_experts: int, nb: int, hist_n: int) -> Offsets:
    touch = 2 * num_experts  # pairs[2][E] comes first
    calls = touch + nb * num_experts
    rows = calls + nb + 1
    ordinal = rows + 2
    hist_calls = ordinal + 1
    hist_rows = hist_calls + 1
    return Offsets(
        touch, calls, rows, ordinal, hist_calls, hist_rows, hist_rows + hist_n
    )


def mix_hash(positions: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The kernel's per-entry hash (``eplb_mix``) in int64 arithmetic."""
    m = MAX_HASH
    x = ((positions * 0x9E3779B1) & m) ^ ((((ids + 1) & m) * 0x85EBCA6B) & m)
    x = x ^ (x >> 15)
    x = (x * 0x2C1B3C6D) & m
    return x ^ (x >> 12)


def build_owner(
    layer_name: str, ranks: int, num_experts: int, strategy: "ExpertPlacementStrategy"
) -> torch.Tensor:
    """Rank of every global expert under the placement in force (uint8 ``[E]``).

    Rebuilt with the same calls as ``ExpertMapManager`` (strategy, or the
    ``VLLM_RDNA_MOE_PLACEMENT`` file for the ``linear`` strategy).
    """
    owner = torch.zeros(num_experts, dtype=torch.uint8)
    if ranks == 1:
        return owner
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )
    from vllm.model_executor.layers.fused_moe.rdna_moe_placement import (
        placement_from_env,
    )

    for rank in range(ranks):
        explicit = (
            placement_from_env(layer_name, rank, num_experts)
            if strategy == "linear"
            else None
        )
        _, expert_map, _ = determine_expert_map(
            ranks, rank, num_experts, strategy, explicit_local_experts=explicit
        )
        assert expert_map is not None
        owner[expert_map >= 0] = rank
    return owner


def _padding_rows(rows: int, device: torch.device) -> torch.Tensor | None:
    """The forward context's padding mask when it covers exactly ``rows`` rows."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    padding = get_forward_context().is_padding
    if padding is None or padding.shape[0] != rows or padding.device != device:
        return None
    return padding


def _host_mirror(like: torch.Tensor) -> torch.Tensor:
    host = torch.zeros(like.shape, dtype=like.dtype)
    return host.pin_memory() if like.device.type == "cuda" else host


def _have_hip_ops() -> bool:
    ops = torch.ops._rocm_C
    return hasattr(ops, "rdna_moe_route_zero_stats") and hasattr(ops, "rdna_eplb_stats")


class LayerStats:
    """Device counters of one MoE layer and the calls that feed them."""

    def __init__(
        self,
        owner_stats: "EplbStats",
        name: str,
        pos: int,
        num_experts: int,
        top_k: int,
        ranks: int,
        owner: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.name = name
        # Set at registration: weak reference to the MoE module, its EP rank.
        self.module: Any = None
        self.rank = 0
        self.pos = pos
        self.num_experts = num_experts
        self.top_k = top_k
        self.ranks = ranks
        self.owner_host = owner
        self.owner = owner.to(device)
        c = owner_stats
        self.off = offsets(num_experts, c.nb, c.hist)
        self.state = torch.zeros(self.off.size, dtype=torch.int64, device=device)
        self.hist: torch.Tensor | None = None
        if c.hist:
            self.hist = torch.zeros(
                (c.hist, num_experts), dtype=torch.int16, device=device
            )
        self.host_state = _host_mirror(self.state)
        self.host_hist = _host_mirror(self.hist) if self.hist is not None else None
        self.ring = c.ring
        self.gseq = c.gseq
        assert c.edges is not None, "register the layer through EplbStats.layer_stats"
        self.edges = c.edges
        self.mask = c.every - 1
        self.nb = c.nb
        self.hist_n = c.hist
        self.use_hip = c.hip_for(device, num_experts)

    # ---- the two call sites in _rdna2_fused_moe ----------------------------

    def route_zero(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None,
        zero1: torch.Tensor,
        zero2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``ops.rdna_moe_route_zero`` and the counters of this call."""
        from vllm import _custom_ops as ops

        if self.use_hip:
            return ops.rdna_moe_route_zero_stats(
                topk_ids,
                block_size,
                num_experts,
                expert_map,
                zero1,
                zero2,
                _padding_rows(topk_ids.shape[0], topk_ids.device),
                self.state,
                self.ring,
                self.gseq,
                self.hist,
                self.owner,
                self.edges,
                self.pos,
                self.mask,
                self.ranks,
            )
        self.count_torch(topk_ids)
        return ops.rdna_moe_route_zero(
            topk_ids, block_size, num_experts, expert_map, zero1, zero2
        )

    def count(self, topk_ids: torch.Tensor) -> None:
        """Count one call that routes elsewhere (align kernel, resident decode)."""
        if topk_ids.numel() == 0:
            return
        if (
            self.use_hip
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and topk_ids.dim() == 2
        ):
            from vllm import _custom_ops as ops

            ops.rdna_eplb_stats(
                topk_ids,
                _padding_rows(topk_ids.shape[0], topk_ids.device),
                self.state,
                self.ring,
                self.gseq,
                self.hist,
                self.owner,
                self.edges,
                self.pos,
                self.mask,
                self.ranks,
                self.num_experts,
            )
            return
        self.count_torch(topk_ids)

    # ---- torch formulation: the reference of the HIP kernel ----------------

    def count_torch(self, topk_ids: torch.Tensor) -> None:
        """Fixed-shape device ops only (no host sync): capturable, CPU-testable."""
        e, k, nb = self.num_experts, self.top_k, self.nb
        off = self.off
        dev = self.state.device
        ids = topk_ids[:, :k].to(torch.int64)
        rows = ids.shape[0]
        padding = _padding_rows(rows, dev)
        row_ok = (
            torch.ones(rows, dtype=torch.bool, device=dev)
            if padding is None
            else ~padding
        )
        in_range = (ids >= 0) & (ids < e)
        flat = torch.where(in_range & row_ok[:, None], ids, e).reshape(-1)
        h = torch.zeros(e + 1, dtype=torch.int64, device=dev)
        h.index_add_(0, flat, torch.ones_like(flat))
        h = h[:e]
        valid = row_ok.sum()
        positions = torch.arange(rows * k, dtype=torch.int64, device=dev).reshape(
            rows, k
        )
        mixed = torch.where(row_ok[:, None], mix_hash(positions, ids), 0)
        id_hash = mixed.sum() & MAX_HASH
        id_hash = torch.where(id_hash >= 2**31, id_hash - 2**32, id_hash)

        ordinal = self.state[off.ordinal]
        sampled = (ordinal & self.mask) == 0
        counted = sampled & (valid > 0)
        gate = counted.to(torch.int64)
        phase = (valid > self.edges[nb - 1]).to(torch.int64)
        band = (self.edges < valid).sum().clamp(max=nb - 1)
        touched = (h > 0).to(torch.int64)
        pairs = self.state[: 2 * e].view(2, e)
        pairs.index_add_(0, phase.reshape(1), (h * gate)[None])
        touch = self.state[off.touch : off.calls].view(nb, e)
        decode_gate = gate * (1 - phase)
        touch.index_add_(0, band.reshape(1), (touched * decode_gate)[None])
        calls = self.state[off.calls : off.rows]
        call_slot = torch.where(phase == 0, band, nb).reshape(1)
        calls.index_add_(0, call_slot, gate.reshape(1))
        self.state[off.rows : off.ordinal].index_add_(
            0, phase.reshape(1), (valid * gate).reshape(1)
        )
        owner = self.owner.to(torch.int64)
        per_rank_t = torch.zeros(self.ranks, dtype=torch.int64, device=dev)
        per_rank_t.index_add_(0, owner, touched)
        per_rank_p = torch.zeros(self.ranks, dtype=torch.int64, device=dev)
        per_rank_p.index_add_(0, owner, h)

        if self.hist is not None:
            want = (counted & (phase == 1)).reshape(1)
            hcalls = self.state[off.hist_calls]
            slot = (hcalls % self.hist_n).reshape(1)
            clipped = h.clamp(max=HIST_CLIP).to(torch.int16)
            self.hist.index_copy_(
                0, slot, torch.where(want, clipped[None], self.hist[slot])
            )
            hrows = self.state[off.hist_rows : off.hist_rows + self.hist_n]
            hrows.index_copy_(0, slot, torch.where(want, valid.reshape(1), hrows[slot]))
            self.state[off.hist_calls] += want.reshape(()).to(torch.int64)

        if self.ring is not None and self.gseq is not None:
            zero = torch.zeros((), dtype=torch.int64, device=dev)
            head = torch.stack(
                [
                    zero + self.pos,
                    valid,
                    zero + rows,
                    id_hash,
                    zero,  # timestamp words: only the HIP kernel has a clock
                    zero,
                    phase,
                    ordinal,
                ]
            )
            pad = MAX_RANKS - self.ranks
            record = torch.cat(
                [
                    head,
                    torch.nn.functional.pad(per_rank_t, (0, pad)),
                    torch.nn.functional.pad(per_rank_p, (0, pad)),
                    torch.zeros(MAX_RANKS, dtype=torch.int64, device=dev),
                ]
            ).to(torch.int32)
            seq = self.gseq[0]
            slot = (seq % self.ring.shape[0]).reshape(1)
            self.ring.index_copy_(
                0, slot, torch.where(sampled, record[None], self.ring[slot])
            )
            self.gseq += sampled.to(torch.int64)
        self.state[off.ordinal] += 1


class EplbStats:
    """Per-process collector; see the module docstring."""

    def __init__(
        self,
        directory: str | None,
        *,
        ring: int = DEFAULT_RING,
        hist: int = 0,
        every: int = 1,
        interval_s: float = DEFAULT_INTERVAL_S,
        decode_max: int = DEFAULT_DECODE_MAX,
        kernel: str = "auto",
    ) -> None:
        if interval_s <= 0:
            raise ValueError("eplb stats interval must be positive")
        if ring < 0 or hist < 0:
            raise ValueError("eplb stats ring and hist sizes must be >= 0")
        if every < 1 or every & (every - 1):
            raise ValueError(f"eplb stats sampling must be a power of two: {every}")
        if kernel not in ("auto", "hip", "torch"):
            raise ValueError(f"{KERNEL_ENV} must be auto, hip or torch: {kernel!r}")
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        # None: counters only, no snapshot files (the online balancer reads them).
        self.directory = directory
        self.ring_n = ring
        self.hist = hist
        self.every = every
        self.interval_s = interval_s
        self.edges_host = band_edges(decode_max)
        self.nb = len(self.edges_host)
        if self.nb > MAX_BANDS:
            raise ValueError(f"at most {MAX_BANDS} bands, got {self.nb}")
        self.hip = kernel != "torch" and _have_hip_ops()
        if kernel == "hip" and not self.hip:
            raise RuntimeError(f"{KERNEL_ENV}=hip but the HIP ops are not built")
        self.ring: torch.Tensor | None = None
        self.gseq: torch.Tensor | None = None
        self.edges: torch.Tensor | None = None
        self.host_ring: torch.Tensor | None = None
        self.host_gseq: torch.Tensor | None = None
        self._layers: dict[int, LayerStats | None] = {}
        self._order: list[LayerStats] = []
        self._ranks = 1
        self._rank = 0
        self._tp_rank = 0
        self._dp_rank = 0
        self._num_experts = 0
        self._top_k = 0
        self._device: torch.device | None = None
        self._start = time.monotonic()
        self._start_wall = time.time()
        self._next_copy = self._start + interval_s
        # (event, wall time, layers) of a copy in flight.
        self._pending: tuple[Any, float, list[LayerStats]] | None = None
        self._seq = 0
        self._path: str | None = None

    @classmethod
    def from_env(cls) -> "EplbStats | None":
        directory = os.environ.get(STATS_DIR_ENV, "") or None
        if directory is None and os.environ.get(ONLINE_ENV, "0") != "1":
            return None
        stats = cls(
            directory,
            ring=int(os.environ.get(RING_ENV, DEFAULT_RING)),
            hist=int(os.environ.get(HIST_ENV, "0")),
            every=int(os.environ.get(EVERY_ENV, "1")),
            interval_s=float(os.environ.get(INTERVAL_ENV, DEFAULT_INTERVAL_S)),
            decode_max=int(os.environ.get(DECODE_MAX_ENV, DEFAULT_DECODE_MAX)),
            kernel=os.environ.get(KERNEL_ENV, "auto"),
        )
        logger.warning(
            "EPLB load counters are ON (%s=%s, every %d call(s), ring %d, kernel=%s):"
            " one extra CTA in the MoE routing launch; the Flash-Next mono K2"
            " stands down (its top-k is not observable)",
            STATS_DIR_ENV if directory is not None else ONLINE_ENV,
            directory if directory is not None else "1",
            stats.every,
            stats.ring_n,
            "hip" if stats.hip else "torch",
        )
        return stats

    def hip_for(self, device: torch.device, num_experts: int) -> bool:
        """Whether a layer takes the HIP counters (else the torch formulation)."""
        return self.hip and device.type == "cuda" and num_experts <= 1024

    @property
    def layers(self) -> list[LayerStats]:
        """The registered layers, in the order of their first call."""
        return list(self._order)

    # ---- registration (first eager call of a layer) -------------------------

    @torch.compiler.disable
    def layer_stats(self, layer: Any, topk_ids: torch.Tensor) -> LayerStats | None:
        """The counters of ``layer`` (registered on its first eager call).

        The registry is keyed by ``id(layer)`` and keeps no reference to it, so the
        layer must outlive its registration (a module does: it lives with the model).
        """
        key = id(layer)
        if key in self._layers:
            return self._layers[key]
        return self._register(key, layer, topk_ids)

    def _register(
        self, key: int, layer: Any, topk_ids: torch.Tensor
    ) -> LayerStats | None:
        name = getattr(layer, "layer_name", None) or f"moe{len(self._order)}"
        device = topk_ids.device
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            logger.warning(
                "EPLB stats: MoE layer %r first ran inside a CUDA graph capture; its"
                " counters would be re-zeroed on every replay, so it is not counted."
                " A warm-up forward pass must precede capture.",
                name,
            )
            self._layers[key] = None
            return None
        cfg = layer.moe_config
        ranks = int(getattr(cfg, "ep_size", 1) or 1)
        rank = int(getattr(cfg, "ep_rank", 0) or 0)
        if ranks > MAX_RANKS:
            raise ValueError(
                f"EPLB stats record up to {MAX_RANKS} expert-parallel ranks, "
                f"got {ranks}"
            )
        num_experts = int(layer.global_num_experts)
        top_k = int(topk_ids.shape[1])
        if self._order:
            first = self._order[0]
            if (first.num_experts, first.top_k, self._ranks) != (
                num_experts,
                top_k,
                ranks,
            ):
                raise ValueError(
                    f"EPLB stats: layer {name!r} has {num_experts} experts, top-k "
                    f"{top_k}, EP {ranks}; layer {first.name!r} has "
                    f"{first.num_experts}, {first.top_k}, {self._ranks}"
                )
            if device != self._device:
                raise ValueError("EPLB stats: layers on different devices")
        strategy = cast(
            "ExpertPlacementStrategy",
            getattr(layer, "expert_placement_strategy", "linear") or "linear",
        )
        owner = build_owner(name, ranks, num_experts, strategy)
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None and ranks > 1:
            mine = (expert_map.detach().cpu() >= 0) == (owner == rank)
            if not bool(mine.all()):
                raise RuntimeError(
                    f"EPLB stats: the owner table of {name!r} ({strategy}) disagrees "
                    "with the layer's expert_map; the per-rank loads would be wrong"
                )
        if not self._order:
            self._ranks, self._rank = ranks, rank
            self._tp_rank = int(getattr(cfg, "tp_rank", 0) or 0)
            self._dp_rank = int(getattr(cfg, "dp_rank", 0) or 0)
            self._num_experts, self._top_k, self._device = num_experts, top_k, device
            self._alloc_shared(device)
        state = LayerStats(
            self, name, len(self._order), num_experts, top_k, ranks, owner, device
        )
        # Weak: the registry must not keep a module alive (see layer_stats). A
        # layer object without weak references stays unknown to the balancer.
        try:
            state.module = weakref.ref(layer)
        except TypeError:
            state.module = None
        state.rank = rank
        self._layers[key] = state
        self._order.append(state)
        return state

    def _alloc_shared(self, device: torch.device) -> None:
        self.edges = torch.tensor(self.edges_host, dtype=torch.int32, device=device)
        if self.ring_n:
            self.ring = torch.zeros(
                (self.ring_n, WORDS), dtype=torch.int32, device=device
            )
            self.gseq = torch.zeros(1, dtype=torch.int64, device=device)
            self.host_ring = _host_mirror(self.ring)
            self.host_gseq = _host_mirror(self.gseq)

    # ---- draining (worker thread, between steps) ----------------------------

    def tick(self, now: float | None = None) -> None:
        """Complete a finished copy; start the next one when the interval is up.

        Never blocks on the device: the copies are non-blocking and a snapshot is
        written only once its event reports completion.
        """
        if not self._order or self.directory is None:
            return
        if self._pending is not None:
            event = self._pending[0]
            if event is not None and not event.query():
                return
            self._finish_pending()
        now = time.monotonic() if now is None else now
        if now < self._next_copy:
            return
        self._next_copy = now + self.interval_s
        self._launch_copy()

    def _copy_to_host(self, non_blocking: bool) -> list[LayerStats]:
        layers = list(self._order)
        for layer in layers:
            layer.host_state.copy_(layer.state, non_blocking=non_blocking)
            if layer.hist is not None and layer.host_hist is not None:
                layer.host_hist.copy_(layer.hist, non_blocking=non_blocking)
        if self.ring is not None and self.gseq is not None:
            assert self.host_ring is not None and self.host_gseq is not None
            self.host_ring.copy_(self.ring, non_blocking=non_blocking)
            self.host_gseq.copy_(self.gseq, non_blocking=non_blocking)
        return layers

    def _launch_copy(self) -> None:
        layers = self._copy_to_host(non_blocking=True)
        event = None
        if layers[0].state.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        self._pending = (event, time.time(), layers)

    def _finish_pending(self) -> None:
        assert self._pending is not None
        _, wall, layers = self._pending
        self._pending = None
        self._write(layers, wall)

    def close(self) -> None:
        """Write a final snapshot (blocking)."""
        if not self._order or self.directory is None:
            return
        if self._pending is not None:
            event = self._pending[0]
            if event is not None:
                event.synchronize()
            self._finish_pending()
        # The counters and their host mirrors were allocated inside the worker's
        # inference mode: updating them in place needs it here too.
        with torch.inference_mode():
            layers = self._copy_to_host(non_blocking=False)
        self._write(layers, time.time())

    # ---- output --------------------------------------------------------------

    def _meta(self, layers: list[LayerStats], wall: float) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "rank": self._rank,
            "ep_size": self._ranks,
            "tp_rank": self._tp_rank,
            "dp_rank": self._dp_rank,
            "num_experts": self._num_experts,
            "top_k": self._top_k,
            "decode_max_rows": self.edges_host[-1],
            "band_edges": list(self.edges_host),
            "ring": self.ring_n,
            "hist": self.hist,
            "every": self.every,
            "kernel": "hip" if self.hip else "torch",
            "layers": [
                {
                    "name": s.name,
                    "kind": layer_kind(s.name),
                    "layer": layer_index(s.name),
                }
                for s in layers
            ],
            "owner": [s.owner_host.tolist() for s in layers],
            "pid": os.getpid(),
            "start_time": self._start_wall,
            "time": wall,
            "seq": self._seq,
        }

    def _write(self, layers: list[LayerStats], wall: float) -> None:
        import numpy as np

        self._seq += 1
        e, nb, h = self._num_experts, self.nb, self.hist
        off = offsets(e, nb, h)
        n = len(layers)
        state = np.stack([s.host_state.numpy() for s in layers])
        arrays: dict[str, Any] = {
            "meta": np.array(
                json.dumps(self._meta(layers, wall), separators=(",", ":"))
            ),
            "pairs": state[:, : 2 * e].reshape(n, 2, e),
            "touch": state[:, off.touch : off.calls].reshape(n, nb, e),
            "calls": state[:, off.calls : off.rows],
            "rows": state[:, off.rows : off.ordinal],
        }
        if self.host_ring is not None and self.host_gseq is not None:
            arrays["ring"] = self.host_ring.numpy()
            arrays["ring_count"] = np.int64(self.host_gseq.numpy()[0])
        else:
            arrays["ring"] = np.zeros((0, WORDS), dtype=np.int32)
            arrays["ring_count"] = np.int64(0)
        if h:
            arrays["hist"] = np.stack(
                [s.host_hist.numpy() for s in layers if s.host_hist is not None]
            )
            arrays["hist_calls"] = state[:, off.hist_calls]
            arrays["hist_rows"] = state[:, off.hist_rows : off.hist_rows + h]
        if self._path is None:
            assert self.directory is not None  # tick and close return early
            self._path = os.path.join(
                self.directory, f"eplb_stats-rank{self._rank}-pid{os.getpid()}.npz"
            )
            logger.info("EPLB load counters -> %s", self._path)
        tmp = self._path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self._path)


# One collector per process; None (the default) is the whole cost when off.
COLLECTOR: EplbStats | None = EplbStats.from_env()
ENABLED = COLLECTOR is not None
