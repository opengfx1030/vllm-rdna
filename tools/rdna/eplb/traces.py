# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Readers that turn what a run leaves behind into per-step expert counts.

Everything the simulator and the solver replay is ``counts[step, E]``: the number of
routed (row, slot) pairs of each global expert in one forward step of one layer.
Three sources exist today:

``id_trace``     ``expert_stats-*.trace.npz`` of the existing collector
                 (``VLLM_EXPERT_STATS_TRACE_LAYERS``): the real top-k ids of the newest
                 decode steps of a few layers (the only source of per-step decode sets);
``hist_jsonl``   ``moe_routing_hist.jsonl`` of ``VLLM_RDNA_MOE_HIST_DIR``: one
                 histogram per MoE layer call with at least 512 tokens (prefill chunks);
``eplb_stats``   ``eplb_stats-rank*.npz`` of ``VLLM_RDNA_EPLB_STATS_DIR`` (see
                 ``summary.py``): its optional prefill histogram ring.

The first two exist without any rebuild, so phase 0 can start with them.
"""

import json
from dataclasses import dataclass

import numpy as np

from tools.rdna.hetero_moe import trace_check


@dataclass
class StepCounts:
    """``counts[s, e]`` int32 pair counts and ``rows[s]`` valid rows of step ``s``."""

    layer: int
    counts: np.ndarray
    rows: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.counts.shape[0])

    @property
    def experts(self) -> int:
        return int(self.counts.shape[1])


def _bincount_rows(blocks: list[np.ndarray], experts: int) -> np.ndarray:
    out = np.zeros((len(blocks), experts), dtype=np.int32)
    for i, block in enumerate(blocks):
        out[i] = np.bincount(block.reshape(-1), minlength=experts)[:experts]
    return out


def id_trace(path: str, num_experts: int = 512) -> dict[int, StepCounts]:
    """Layer index -> counts of the decode steps in a collector id ring (oldest first).

    Rows the router wrote for padding are not distinguishable in the ring (it keeps the
    rows of the call, padding included); run the trace without CUDA-graph padding.
    """
    trace = trace_check.load_trace(path)
    out: dict[int, StepCounts] = {}
    for pos, name in enumerate(trace["names"]):
        steps = trace_check.valid_steps(trace, pos)
        if not steps:
            continue
        layer = trace_check.layer_index(name)
        out[layer] = StepCounts(
            layer,
            _bincount_rows(steps, num_experts),
            np.array([len(s) for s in steps], dtype=np.int32),
        )
    return out


def hist_jsonl(path: str, min_tokens: int = 0) -> tuple[dict, dict[int, StepCounts]]:
    """``(meta, layer -> counts)`` of a ``moe_routing_hist.jsonl`` capture.

    One row per recorded layer call, in file order; ``rows`` is the call's token
    count. A truncated last line is ignored (the file is written live).
    """
    meta: dict = {}
    counts: dict[int, list[list[int]]] = {}
    rows: dict[int, list[int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break
            if "meta" in obj:
                meta = obj["meta"]
                continue
            if "counts" not in obj or int(obj["tokens"]) < min_tokens:
                continue
            layer = int(obj["layer_idx"])
            counts.setdefault(layer, []).append(obj["counts"])
            rows.setdefault(layer, []).append(int(obj["tokens"]))
    out = {
        layer: StepCounts(
            layer,
            np.asarray(c, dtype=np.int32),
            np.asarray(rows[layer], dtype=np.int32),
        )
        for layer, c in counts.items()
    }
    return meta, out


def eplb_stats_hist(path: str) -> dict[int, StepCounts]:
    """Layer index -> prefill histograms of one ``eplb-stats/1`` rank file.

    The ring holds the newest calls; slot order is call order modulo the ring size, so
    the valid slots are re-ordered oldest first.
    """
    with np.load(path) as z:
        if "hist" not in z.files:
            return {}
        meta = json.loads(str(z["meta"]))
        hist, calls, rows = z["hist"], z["hist_calls"], z["hist_rows"]
    out: dict[int, StepCounts] = {}
    for pos, info in enumerate(meta["layers"]):
        n_slots = hist.shape[1]
        n = min(int(calls[pos]), n_slots)
        if n == 0:
            continue
        order = [(int(calls[pos]) - n + i) % n_slots for i in range(n)]
        out[int(info["layer"])] = StepCounts(
            int(info["layer"]),
            hist[pos][order].astype(np.int32),
            rows[pos][order].astype(np.int32),
        )
    return out


def split_steps(
    sc: StepCounts, folds: int = 2, fold: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Train / test split by contiguous blocks (neighbouring steps share a prompt, so
    random splitting would leak): fold ``fold`` of ``folds`` is the test block."""
    bounds = np.linspace(0, sc.steps, folds + 1).astype(int)
    lo, hi = bounds[fold], bounds[fold + 1]
    test = sc.counts[lo:hi]
    train = np.concatenate([sc.counts[:lo], sc.counts[hi:]])
    return train, test
