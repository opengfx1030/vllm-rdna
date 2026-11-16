# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step rank loads of one MoE layer: touched experts, routed pairs, time.

``counts[s, e]`` is the number of routed (row, slot) pairs of expert ``e`` in step
``s`` (a decode step is one forward pass of the batch; a prefill "step" is one
chunk). Under all-gather/reduce-scatter EP every rank sees the whole batch's top-k ids,
so a rank's work is fixed by which experts it owns:

``T[s, r]``   distinct experts of rank ``r`` with at least one pair (decode objective:
              bandwidth bound, one weight read per touched expert),
``P[s, r]``   pairs of rank ``r`` (prefill objective: compute bound),
``us[s, r]``  the roofline time of ``geometry.CostModel``.

The figure of merit is ``max_over_mean(x) = sum_s max_r x / sum_s mean_r x``: the
factor by which the slowest rank stretches the sum of step times (DeepInfer's
``sum_steps max_rank(load)`` objective divided by its balanced bound).
"""

import numpy as np

from tools.rdna.eplb.geometry import DEFAULT_COST, CostModel


def step_counts(
    ids: np.ndarray, step_of_row: np.ndarray, n_steps: int, experts: int
) -> np.ndarray:
    """``[n_steps, experts]`` int32 pair counts from ``[rows, K]`` expert ids."""
    rows, k = ids.shape
    flat = np.repeat(step_of_row.astype(np.int64), k) * experts + ids.reshape(-1)
    counts = np.bincount(flat, minlength=n_steps * experts)
    return counts.reshape(n_steps, experts).astype(np.int32)


def rank_loads(
    counts: np.ndarray,
    owner: np.ndarray,
    ranks: int,
    cost: CostModel = DEFAULT_COST,
) -> dict[str, np.ndarray]:
    """``T``, ``P`` and ``us`` of a permutation-only placement (``owner[e]`` = rank)."""
    onehot = np.zeros((owner.shape[0], ranks))
    onehot[np.arange(owner.shape[0]), owner] = 1.0
    return {
        "T": (counts > 0).astype(np.float64) @ onehot,
        "P": counts.astype(np.float64) @ onehot,
        "us": cost.rank_us(counts) @ onehot + cost.fixed_us,
    }


def max_over_mean(x: np.ndarray) -> float:
    """``E[max_r x] / E[mean_r x]`` over the steps of ``x[s, r]`` (1.0 = balanced)."""
    mean = x.mean(axis=1).sum()
    return float(x.max(axis=1).sum() / mean) if mean > 0 else 1.0


def summarize(loads: dict[str, np.ndarray]) -> dict[str, float]:
    """Max-over-mean of T, P and time, and the mean slowest-rank time (us)."""
    return {
        "T": max_over_mean(loads["T"]),
        "P": max_over_mean(loads["P"]),
        "us": max_over_mean(loads["us"]),
        "max_us": float(loads["us"].max(axis=1).mean()),
        "mean_us": float(loads["us"].mean(axis=1).mean()),
    }
