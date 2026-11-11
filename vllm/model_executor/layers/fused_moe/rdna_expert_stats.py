# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, CUDA-graph-safe routing statistics for MoE layers (hetero-MoE study).

``VLLM_EXPERT_STATS_DIR=<dir>`` turns collection on (unset or empty: off, and
the only cost left in the hot path is one ``is not None`` test per MoE call
in ``MoERunner._apply_quant_method`` and per step in ``Worker.execute_model``).
The collector counts, per MoE layer, how many token rows selected each expert
(``counts[e]`` = rows whose top-k set contains ``e``), split into a *decode*
phase (calls with at most ``VLLM_EXPERT_STATS_DECODE_MAX_TOKENS`` rows, default
64) and a *prefill* phase (larger calls). It also counts calls (``steps``) and,
for decode calls, how many valid rows each call had (``rows_hist``).
Optionally (``VLLM_EXPERT_STATS_PAIRS=1``) it counts how often two experts are
selected for the same row (decode phase only).

Mechanism. Every counter is a device tensor allocated on the first eager call
of its layer (vLLM's profile run precedes graph capture; registration during a
capture raises). A call does only fixed-shape device work -- mask padded and
out-of-range ids to a trash slot, then ``index_add_`` ones -- so it is captured
into FULL and piecewise graphs and replayed with the graph. Counters are
cumulative and never reset, which removes any race between a draining read and
in-flight graphs: a window is the difference of two snapshots.

Draining is off the hot path. The worker calls :meth:`ExpertStatsCollector.tick`
once per step (outside the forward pass, outside graph capture). Every
``VLLM_EXPERT_STATS_INTERVAL_S`` seconds (default 30) it queues non-blocking
copies of the counters into pinned host buffers and records an event; a later
tick that finds the event complete writes the snapshot. No thread issues CUDA
calls and the host never waits on the device, except in :meth:`close` (worker
shutdown), which writes a final snapshot synchronously.

Output (``<dir>/expert_stats-dp<D>-pid<P>.jsonl``, written by TP rank 0 of each
DP rank only: the router is replicated across TP ranks)::

    {"type": "meta", "schema": "expert-stats/1", "model": {"name": ...,
     "num_layers": 48, "num_experts": 512, "top_k": 10}, "num_experts": 512,
     "top_k": 10, "decode_max_tokens": 64, "pairs": false, "tp_size": 4,
     "dp_rank": 0, "pid": 1, "time": 1.7e9,
     "layers": [{"name": "model.layers.3.mlp.experts", "idx": 3,
                 "kind": "target"}, ...]}
    {"type": "snapshot", "seq": 1, "time": 1.7e9, "elapsed_s": 30.0,
     "counts": {"<layer name>": {"decode": [E ints], "prefill": [E ints]}},
     "tokens": {"<layer name>": {"decode": int, "prefill": int}},
     "steps": {"<layer name>": {"decode": int, "prefill": int}},
     "rows_hist": {"<layer name>": [decode_max_tokens + 1 ints]},
     "invalid": {"<layer name>": int}}

Everything is cumulative since process start. ``tokens`` is valid pairs / top-k
(rows, MTP-rejected draft rows included); ``steps`` counts calls of the layer;
``rows_hist[n]`` counts decode calls that had ``n`` valid (non-padding) rows;
``invalid`` counts padded rows and out-of-range ids (pairs). A new ``meta``
line is written whenever a layer registers. With pairs on,
``expert_stats-dp<D>-pid<P>.pairs.npz`` holds ``names`` (layer names) and
``pairs`` (int32 ``[L, E, E]``, entry ``[l, a, b]`` with ``a < b`` = decode rows
that selected both), rewritten every ``VLLM_EXPERT_STATS_PAIRS_EVERY``
snapshots (default 10) and at close. ``tools/rdna/hetero_moe/stats_io.py``
merges and converts both files.

With ``VLLM_EXPERT_STATS_TRACE_LAYERS=<i,j,...>`` (target layer indices) the
listed layers also keep a ring of the last ``VLLM_EXPERT_STATS_TRACE_STEPS``
(default 4096) decode calls' top-k ids (``-1`` for padding and invalid ids), to
test the row-independence assumption of the routing model. They are written to
``expert_stats-dp<D>-pid<P>.trace.npz``: ``names``, ``ids`` (int16
``[L, N, R, K]``, ``R`` = decode token limit), ``rows`` (int32 ``[L, N]``, rows of
the call in that slot, padding included), ``calls`` (int64 ``[L]``, decode calls
so far; call ``c`` is in slot ``c % N``, the newest ``min(calls, N)`` are valid)
and ``max_rows``; rewritten like the pairs file. Tracing adds about eight small
kernels per traced layer and call; it is for dedicated collection runs.

Requirements and limits:

* The collector hooks the router output in ``MoERunner._apply_quant_method``,
  so it covers every modular quant method (RDNA2 W4A16, the RDNA3 port, plain
  vLLM) but not monolithic kernels. The Flash-Next ``--monokernel`` K2 does its
  top-k inside a persistent kernel and exposes no ids: with collection on, K2
  stands down and the modules route (same router weights, same top-k).
* Expert ids are the router's logical ids; keep EPLB off (physical ids from
  redundant experts would be counted as out of range).
"""

import json
import os
import time
from typing import Any

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STATS_DIR_ENV = "VLLM_EXPERT_STATS_DIR"
INTERVAL_ENV = "VLLM_EXPERT_STATS_INTERVAL_S"
PAIRS_ENV = "VLLM_EXPERT_STATS_PAIRS"
PAIRS_EVERY_ENV = "VLLM_EXPERT_STATS_PAIRS_EVERY"
DECODE_MAX_ENV = "VLLM_EXPERT_STATS_DECODE_MAX_TOKENS"
ALL_RANKS_ENV = "VLLM_EXPERT_STATS_ALL_RANKS"  # debug: every TP rank records
# "fused" (default on CUDA/ROCm: 1-2 Triton launches per MoE call) or "torch" (the
# reference formulation, ~35 small kernels per call).
KERNEL_ENV = "VLLM_EXPERT_STATS_KERNEL"
TRACE_LAYERS_ENV = "VLLM_EXPERT_STATS_TRACE_LAYERS"
TRACE_STEPS_ENV = "VLLM_EXPERT_STATS_TRACE_STEPS"

SCHEMA = "expert-stats/1"
PHASES = ("decode", "prefill")
DEFAULT_INTERVAL_S = 30.0
DEFAULT_DECODE_MAX_TOKENS = 64
DEFAULT_PAIRS_EVERY = 10
DEFAULT_TRACE_STEPS = 4096

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index(layer_name: str) -> int:
    """Decoder layer index parsed from a module name, or -1 when absent."""
    match = _LAYER_INDEX.search(layer_name)
    return int(match.group(1)) if match else -1


def layer_kind(layer_name: str) -> str:
    """``"mtp"`` for the draft model's layers, ``"target"`` otherwise."""
    return "mtp" if re.search(r"(?:^|\.)mtp(?:\.|$)", layer_name) else "target"


def parse_layer_list(text: str) -> frozenset[int]:
    """``"0, 24,47"`` -> ``frozenset({0, 24, 47})`` (empty text: no layers)."""
    return frozenset(int(part) for part in text.replace(",", " ").split())


def _host_mirror(shape: tuple[int, ...], pin: bool) -> torch.Tensor:
    host = torch.zeros(shape, dtype=torch.int32)
    return host.pin_memory() if pin else host


class _LayerCounters:
    """Device counters and host mirrors of one MoE layer.

    ``counts`` is ``[phase, E + 2]``: experts, then the invalid-pair slot, then
    the call counter. ``rows_hist`` is ``[decode_max_tokens + 1]``.
    """

    def __init__(
        self,
        name: str,
        num_experts: int,
        top_k: int,
        device: torch.device,
        pairs: bool,
        decode_max_tokens: int,
        trace_steps: int = 0,
    ) -> None:
        self.name = name
        self.num_experts = num_experts
        self.top_k = top_k
        self.counts = torch.zeros(
            (len(PHASES), num_experts + 2), dtype=torch.int32, device=device
        )
        self.rows_hist = torch.zeros(
            decode_max_tokens + 1, dtype=torch.int32, device=device
        )
        self.pair_counts: torch.Tensor | None = None
        self.pair_index: tuple[torch.Tensor, torch.Tensor] | None = None
        if pairs and top_k > 1:
            # [a * E + b] for a < b; the last slot takes invalid pairs.
            self.pair_counts = torch.zeros(
                num_experts * num_experts + 1, dtype=torch.int32, device=device
            )
            rows, cols = torch.triu_indices(top_k, top_k, offset=1, device=device)
            self.pair_index = (rows, cols)
        pin = device.type == "cuda"
        self.host_counts = _host_mirror(tuple(self.counts.shape), pin)
        self.host_hist = _host_mirror(tuple(self.rows_hist.shape), pin)
        self.host_pairs: torch.Tensor | None = None
        if self.pair_counts is not None:
            self.host_pairs = _host_mirror(tuple(self.pair_counts.shape), pin)
        self.trace_steps = trace_steps
        self.trace_ids: torch.Tensor | None = None
        self.trace_rows: torch.Tensor | None = None
        self.trace_arange: torch.Tensor | None = None
        self.host_trace_ids: torch.Tensor | None = None
        self.host_trace_rows: torch.Tensor | None = None
        if trace_steps:
            # flat [steps * R, K] so one call is a single index_copy_ of its rows
            shape = (trace_steps * decode_max_tokens, top_k)
            self.trace_ids = torch.full(shape, -1, dtype=torch.int16, device=device)
            self.trace_rows = torch.zeros(trace_steps, dtype=torch.int32, device=device)
            self.trace_arange = torch.arange(
                decode_max_tokens, dtype=torch.int64, device=device
            )
            self.host_trace_ids = torch.zeros(shape, dtype=torch.int16)
            self.host_trace_rows = torch.zeros(trace_steps, dtype=torch.int32)
            if pin:
                self.host_trace_ids = self.host_trace_ids.pin_memory()
                self.host_trace_rows = self.host_trace_rows.pin_memory()


class ExpertStatsCollector:
    """Per-process collector; see the module docstring."""

    def __init__(
        self,
        directory: str,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        pairs: bool = False,
        pairs_every: int = DEFAULT_PAIRS_EVERY,
        decode_max_tokens: int = DEFAULT_DECODE_MAX_TOKENS,
        all_ranks: bool = False,
        fused: bool = False,
        trace_layers: frozenset[int] = frozenset(),
        trace_steps: int = DEFAULT_TRACE_STEPS,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("expert stats interval must be positive")
        self.all_ranks = all_ranks
        if decode_max_tokens < 1:
            raise ValueError("expert stats decode token limit must be >= 1")
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.interval_s = interval_s
        self.pairs = pairs
        self.pairs_every = max(1, pairs_every)
        if trace_layers and trace_steps < 1:
            raise ValueError("expert stats trace needs at least one step")
        self.trace_layers = frozenset(trace_layers)
        self.trace_steps = trace_steps
        self.decode_max_tokens = decode_max_tokens
        self._record_fused: Any = None
        if fused:
            from vllm.model_executor.layers.fused_moe import rdna_expert_stats_kernels

            if decode_max_tokens > rdna_expert_stats_kernels.BLOCK_ROWS:
                raise ValueError(
                    f"the fused expert-stats kernel handles decode calls of up to "
                    f"{rdna_expert_stats_kernels.BLOCK_ROWS} rows; "
                    f"{DECODE_MAX_ENV}={decode_max_tokens} needs {KERNEL_ENV}=torch"
                )
            self._record_fused = rdna_expert_stats_kernels.record_fused
        self._layers: dict[str, _LayerCounters] = {}
        self._tp_size = 1
        self._tp_rank = 0
        self._dp_rank = 0
        self._model_name: str | None = None
        self._path: str | None = None
        self._pairs_path: str | None = None
        self._trace_path: str | None = None
        self._file: Any = None
        self._meta_dirty = False
        self._seq = 0
        self._start = time.monotonic()
        self._next_copy = self._start + interval_s
        # (event, launch wall time, launch monotonic time, names) of a copy in flight.
        self._pending: tuple[Any, float, float, list[str]] | None = None
        self._pending_pairs = False
        self._pending_trace = False

    @classmethod
    def from_env(cls) -> "ExpertStatsCollector | None":
        directory = os.environ.get(STATS_DIR_ENV, "")
        if not directory:
            return None
        collector = cls(
            directory,
            interval_s=float(os.environ.get(INTERVAL_ENV, DEFAULT_INTERVAL_S)),
            pairs=os.environ.get(PAIRS_ENV, "0") not in ("", "0"),
            pairs_every=int(os.environ.get(PAIRS_EVERY_ENV, DEFAULT_PAIRS_EVERY)),
            decode_max_tokens=int(
                os.environ.get(DECODE_MAX_ENV, DEFAULT_DECODE_MAX_TOKENS)
            ),
            all_ranks=os.environ.get(ALL_RANKS_ENV, "0") not in ("", "0"),
            fused=os.environ.get(KERNEL_ENV, "fused") == "fused",
            trace_layers=parse_layer_list(os.environ.get(TRACE_LAYERS_ENV, "")),
            trace_steps=int(os.environ.get(TRACE_STEPS_ENV, DEFAULT_TRACE_STEPS)),
        )
        logger.warning(
            "MoE expert statistics are ON (%s=%s, every %.0f s, pairs=%s, "
            "kernel=%s): small device kernels per MoE call; the Flash-Next mono "
            "K2 stands down (its top-k is not observable)",
            STATS_DIR_ENV,
            directory,
            collector.interval_s,
            collector.pairs,
            "fused" if collector._record_fused is not None else "torch",
        )
        return collector

    # ---- recording (inside the forward pass, graph-safe) -------------------

    @torch.compiler.disable
    def record(
        self,
        layer_name: str,
        topk_ids: torch.Tensor,
        *,
        num_experts: int,
        top_k: int,
        tp_rank: int,
        tp_size: int,
        dp_rank: int,
    ) -> None:
        """Count one MoE call's routing. ``topk_ids`` is ``[T, >= top_k]``."""
        if (tp_rank != 0 and not self.all_ranks) or topk_ids.numel() == 0:
            return
        state = self._layers.get(layer_name)
        if state is None:
            state = self._register(
                layer_name,
                num_experts,
                top_k,
                topk_ids.device,
                tp_size,
                tp_rank,
                dp_rank,
            )
        ids = topk_ids[:, :top_k] if topk_ids.shape[1] > top_k else topk_ids
        rows = ids.shape[0]
        phase = 0 if rows <= self.decode_max_tokens else 1
        if phase == 0 and state.trace_ids is not None:
            self._record_trace(state, ids, rows)  # before the counters: slot = call #
        if self._record_fused is not None and ids.is_cuda:
            self._record_fused(
                ids,
                _padding_rows(rows, ids.device),
                state.counts[phase],
                state.rows_hist if phase == 0 else None,
                state.pair_counts if phase == 0 else None,
                num_experts=num_experts,
                top_k=top_k,
            )
            return
        idx = ids.to(torch.int64)
        invalid = (idx < 0) | (idx >= num_experts)
        padding = _padding_rows(rows, idx.device)
        if padding is not None:
            invalid = invalid | padding.unsqueeze(1)
        idx = torch.where(invalid, num_experts, idx)
        flat = idx.reshape(-1)
        counts = state.counts[phase]
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        counts[num_experts + 1] += 1  # one call (step)
        if phase == 0:
            valid_rows = (~invalid.all(dim=1)).sum().reshape(1)
            state.rows_hist.index_add_(
                0, valid_rows, torch.ones_like(valid_rows, dtype=torch.int32)
            )
            if state.pair_counts is not None:
                assert state.pair_index is not None
                first, second = state.pair_index
                a, b = idx[:, first], idx[:, second]
                low = torch.minimum(a, b)
                high = torch.maximum(a, b)
                bad = (a == num_experts) | (b == num_experts)
                pair = torch.where(
                    bad, num_experts * num_experts, low * num_experts + high
                )
                pair = pair.reshape(-1)
                state.pair_counts.index_add_(
                    0, pair, torch.ones_like(pair, dtype=torch.int32)
                )

    def _register(
        self,
        name: str,
        num_experts: int,
        top_k: int,
        device: torch.device,
        tp_size: int,
        tp_rank: int,
        dp_rank: int,
    ) -> _LayerCounters:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"expert stats: MoE layer {name!r} first ran inside a CUDA graph "
                "capture, so its counters would be re-zeroed on every replay. "
                "A warm-up forward pass must precede capture."
            )
        if self._layers:
            first = next(iter(self._layers.values()))
            if first.num_experts != num_experts or first.top_k != top_k:
                raise ValueError(
                    f"expert stats: layer {name!r} has {num_experts} experts and "
                    f"top-k {top_k}, layer {first.name!r} has {first.num_experts} "
                    f"and {first.top_k}"
                )
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._dp_rank = dp_rank
        if self._model_name is None:
            self._model_name = _model_name()
        traced = layer_kind(name) == "target" and layer_index(name) in self.trace_layers
        state = _LayerCounters(
            name,
            num_experts,
            top_k,
            device,
            self.pairs,
            self.decode_max_tokens,
            self.trace_steps if traced else 0,
        )
        self._layers[name] = state
        self._meta_dirty = True
        return state

    # ---- draining (worker thread, between steps) ---------------------------

    def tick(self, now: float | None = None) -> None:
        """Complete a finished copy; start the next one when the interval is up.

        Never blocks on the device: the copies are non-blocking and a snapshot is
        written only once its event reports completion.
        """
        if not self._layers:
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

    def _record_trace(
        self, state: _LayerCounters, ids: torch.Tensor, rows: int
    ) -> None:
        """Store this decode call's ids in slot ``call number % trace_steps``."""
        assert state.trace_ids is not None and state.trace_rows is not None
        assert state.trace_arange is not None
        e = state.num_experts
        idx = ids.to(torch.int64)
        bad = (idx < 0) | (idx >= e)
        padding = _padding_rows(rows, idx.device)
        if padding is not None:
            bad = bad | padding.unsqueeze(1)
        out = torch.where(bad, -1, idx).to(torch.int16)
        slot = state.counts[0, e + 1].to(torch.int64) % state.trace_steps
        dest = state.trace_arange[:rows] + slot * self.decode_max_tokens
        state.trace_ids.index_copy_(0, dest, out)
        state.trace_rows.index_fill_(0, slot.reshape(1), rows)

    def _copy_to_host(
        self, pairs: bool, non_blocking: bool, traces: bool = False
    ) -> list[str]:
        names = list(self._layers)
        for name in names:
            state = self._layers[name]
            state.host_counts.copy_(state.counts, non_blocking=non_blocking)
            state.host_hist.copy_(state.rows_hist, non_blocking=non_blocking)
            if pairs and state.pair_counts is not None:
                assert state.host_pairs is not None
                state.host_pairs.copy_(state.pair_counts, non_blocking=non_blocking)
            if traces and state.trace_ids is not None:
                assert state.host_trace_ids is not None
                assert (
                    state.host_trace_rows is not None and state.trace_rows is not None
                )
                state.host_trace_ids.copy_(state.trace_ids, non_blocking=non_blocking)
                state.host_trace_rows.copy_(state.trace_rows, non_blocking=non_blocking)
        return names

    def _launch_copy(self) -> None:
        due = (self._seq + 1) % self.pairs_every == 0
        pairs = self.pairs and due
        traces = bool(self.trace_layers) and due
        names = self._copy_to_host(pairs, non_blocking=True, traces=traces)
        event = None
        if any(self._layers[n].counts.device.type == "cuda" for n in names):
            event = torch.cuda.Event()
            event.record()
        self._pending_pairs = pairs
        self._pending_trace = traces
        self._pending = (event, time.time(), time.monotonic(), names)

    def _finish_pending(self) -> None:
        assert self._pending is not None
        _, wall, mono, names = self._pending
        self._pending = None
        self._write_snapshot(names, wall, mono - self._start)

    def close(self) -> None:
        """Write a final snapshot (blocking) and close the file."""
        if self._layers:
            if self._pending is not None:
                event = self._pending[0]
                if event is not None:
                    event.synchronize()
                self._finish_pending()
            # The counters and their host mirrors were allocated inside the
            # worker's inference mode: updating them in place needs it here too
            # (shutdown runs outside it).
            with torch.inference_mode():
                names = self._copy_to_host(
                    self.pairs, non_blocking=False, traces=bool(self.trace_layers)
                )
            self._pending_pairs = self.pairs
            self._pending_trace = bool(self.trace_layers)
            self._write_snapshot(names, time.time(), time.monotonic() - self._start)
        if self._file is not None:
            self._file.close()
            self._file = None

    # ---- output ------------------------------------------------------------

    def _open(self) -> Any:
        stem = f"expert_stats-dp{self._dp_rank}-pid{os.getpid()}"
        self._path = os.path.join(self.directory, stem + ".jsonl")
        self._pairs_path = os.path.join(self.directory, stem + ".pairs.npz")
        self._trace_path = os.path.join(self.directory, stem + ".trace.npz")
        self._file = open(self._path, "w", buffering=1)  # noqa: SIM115
        logger.info("MoE expert statistics -> %s", self._path)
        return self._file

    def _write_meta(self, out: Any, wall: float) -> None:
        any_state = next(iter(self._layers.values()))
        target_layers = sum(layer_kind(n) == "target" for n in self._layers)
        meta = {
            "type": "meta",
            "schema": SCHEMA,
            "model": {
                "name": self._model_name,
                "num_layers": target_layers,
                "num_experts": any_state.num_experts,
                "top_k": any_state.top_k,
            },
            "num_experts": any_state.num_experts,
            "top_k": any_state.top_k,
            "decode_max_tokens": self.decode_max_tokens,
            "pairs": self.pairs,
            "tp_size": self._tp_size,
            "tp_rank": self._tp_rank,
            "dp_rank": self._dp_rank,
            "pid": os.getpid(),
            "time": wall,
            "layers": [
                {"name": n, "idx": layer_index(n), "kind": layer_kind(n)}
                for n in self._layers
            ],
        }
        out.write(json.dumps(meta, separators=(",", ":")) + "\n")

    def _write_snapshot(self, names: list[str], wall: float, elapsed: float) -> None:
        out = self._file if self._file is not None else self._open()
        if self._meta_dirty:
            self._meta_dirty = False
            self._write_meta(out, wall)
        self._seq += 1
        counts: dict[str, dict[str, list[int]]] = {}
        tokens: dict[str, dict[str, int]] = {}
        steps: dict[str, dict[str, int]] = {}
        hist: dict[str, list[int]] = {}
        invalid: dict[str, int] = {}
        for name in names:
            state = self._layers[name]
            host = state.host_counts
            e = state.num_experts
            counts[name], tokens[name], steps[name] = {}, {}, {}
            for p, phase in enumerate(PHASES):
                row = host[p, :e].tolist()
                counts[name][phase] = row
                tokens[name][phase] = sum(row) // state.top_k
                steps[name][phase] = int(host[p, e + 1])
            invalid[name] = int(host[:, e].sum())
            hist[name] = state.host_hist.tolist()
        snapshot = {
            "type": "snapshot",
            "seq": self._seq,
            "time": wall,
            "elapsed_s": round(elapsed, 3),
            "counts": counts,
            "tokens": tokens,
            "steps": steps,
            "rows_hist": hist,
            "invalid": invalid,
        }
        out.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        if self._pending_pairs:
            self._write_pairs(names)
            self._pending_pairs = False
        if self._pending_trace:
            self._write_trace(names)
            self._pending_trace = False

    def _write_trace(self, names: list[str]) -> None:
        import numpy as np

        assert self._trace_path is not None
        traced = [n for n in names if self._layers[n].trace_ids is not None]
        if not traced:
            return
        r = self.decode_max_tokens
        ids, rows, calls = [], [], []
        for name in traced:
            state = self._layers[name]
            assert state.host_trace_ids is not None
            assert state.host_trace_rows is not None
            ids.append(
                state.host_trace_ids.numpy().reshape(state.trace_steps, r, state.top_k)
            )
            rows.append(state.host_trace_rows.numpy())
            calls.append(int(state.host_counts[0, state.num_experts + 1]))
        tmp = self._trace_path + ".tmp.npz"
        np.savez(
            tmp,
            names=np.array(traced),
            ids=np.stack(ids),
            rows=np.stack(rows),
            calls=np.array(calls, dtype=np.int64),
            max_rows=r,
        )
        os.replace(tmp, self._trace_path)

    def _write_pairs(self, names: list[str]) -> None:
        import numpy as np

        assert self._pairs_path is not None
        e = self._layers[names[0]].num_experts
        stack = np.zeros((len(names), e, e), dtype=np.int32)
        for i, name in enumerate(names):
            host = self._layers[name].host_pairs
            if host is not None:
                stack[i] = host[: e * e].numpy().reshape(e, e)
        tmp = self._pairs_path + ".tmp.npz"
        np.savez(tmp, names=np.array(names), pairs=stack)
        os.replace(tmp, self._pairs_path)


def _model_name() -> str | None:
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        return config.model_config.model if config is not None else None
    except Exception:  # best effort metadata only
        return None


def _padding_rows(rows: int, device: torch.device) -> torch.Tensor | None:
    """The forward context's padding mask when it covers exactly ``rows`` rows."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    padding = get_forward_context().is_padding
    if padding is None or padding.shape[0] != rows or padding.device != device:
        return None
    return padding


# One collector per process; None (the default) is the whole cost when off.
COLLECTOR: ExpertStatsCollector | None = ExpertStatsCollector.from_env()
ENABLED = COLLECTOR is not None
