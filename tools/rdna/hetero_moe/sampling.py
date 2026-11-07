# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling of top-k expert sets, shared by the simulator and the synthesizer."""

import numpy as np


def inclusion_probs(weights: np.ndarray, top_k: int) -> np.ndarray:
    """Per-expert selection probability ``q`` with ``sum(q) == top_k``, ``q <= 1``.

    Scales ``weights`` by ``c`` and clips at 1, choosing ``c`` by bisection so the
    clipped sum is ``top_k`` (the water-filling that a systematic sample needs).
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    if top_k >= n:
        return np.ones(n)
    # Experts without mass still need a sliver so that top_k of them can be chosen.
    w = np.maximum(w, 1e-9 * max(float(w.max()), 1e-300))
    lo, hi = 0.0, float(top_k) / float(w.min())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, mid * w).sum() < top_k:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, 0.5 * (lo + hi) * w)


def systematic_sample(q: np.ndarray, rows: int, rng: np.random.Generator) -> np.ndarray:
    """``[rows, K]`` distinct expert ids with inclusion probabilities ``q``.

    Madow systematic sampling over a fresh random permutation per row. ``q`` must
    sum to the integer K and be <= 1; the marginals are exact.
    """
    n = q.shape[0]
    k = int(round(q.sum()))
    if rows == 0:
        return np.zeros((0, k), dtype=np.int32)
    perm = np.argsort(rng.random((rows, n)), axis=1)
    cum = np.cumsum(q[perm], axis=1)
    cum[:, -1] = k  # guard against rounding at the last bin
    start = rng.random(rows)
    points = start[:, None] + np.arange(k)[None, :]
    offset = (np.arange(rows) * (k + 2))[:, None]
    pos = np.searchsorted((cum + offset).ravel(), (points + offset).ravel(), "right")
    pos = pos.reshape(rows, k) - (np.arange(rows) * n)[:, None]
    pos = np.minimum(pos, n - 1)
    return np.take_along_axis(perm, pos, axis=1).astype(np.int32)


def gumbel_topk(logits: np.ndarray, top_k: int, rng: np.random.Generator) -> np.ndarray:
    """Plackett-Luce sample without replacement: ``[rows, K]`` ids.

    ``logits`` is ``[E]`` or ``[rows, E]``.
    """
    logits = np.atleast_2d(logits)
    rows = logits.shape[0]
    return gumbel_topk_rows(
        np.broadcast_to(logits, (rows, logits.shape[1])), top_k, rng
    )


def gumbel_topk_rows(
    logits: np.ndarray, top_k: int, rng: np.random.Generator
) -> np.ndarray:
    gumbel = -np.log(-np.log(rng.random(logits.shape).clip(1e-12, 1 - 1e-12)))
    keys = logits + gumbel
    idx = np.argpartition(-keys, top_k - 1, axis=1)[:, :top_k]
    # Order by key so slot 0 is the strongest pick (stable, deterministic).
    order = np.argsort(-np.take_along_axis(keys, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int32)
