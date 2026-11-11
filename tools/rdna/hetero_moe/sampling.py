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


def mix_rows(
    parent: np.ndarray, fresh: np.ndarray, keep: float, rng: np.random.Generator
) -> np.ndarray:
    """``[n, K]`` rows that keep each element of ``parent`` with probability ``keep``
    and take the missing ones from ``fresh`` (distinct ids per row, any order),
    skipping ids the row already holds. ``fresh`` rows must have K distinct ids."""
    n, k = parent.shape
    if keep <= 0.0:
        return fresh.copy()
    kept = np.where(rng.random((n, k)) < keep, parent, -1)
    need = k - (kept >= 0).sum(axis=1)
    free = ~(fresh[:, :, None] == kept[:, None, :]).any(axis=2)
    take = free & (np.cumsum(free, axis=1) <= need[:, None])
    # kept entries first, then the taken fresh ids, in a fixed-width row
    merged = np.concatenate([kept, np.where(take, fresh, -1)], axis=1)
    order = np.argsort(merged < 0, axis=1, kind="stable")
    return np.take_along_axis(merged, order, axis=1)[:, :k].astype(np.int32)


def correlated_sample(
    q: np.ndarray,
    step_of_row: np.ndarray,
    row_in_step: np.ndarray,
    group: int,
    rho: float,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``[rows, K]`` ids whose rows share experts like the rows of a real decode step.

    Marginals stay ``q`` (to the accuracy of the fill). Every step draws a base row;
    each MTP group of ``group`` consecutive rows (the verified token and its drafts of
    one sequence) keeps each base expert with probability ``tau``; each row keeps each
    expert of its group's base with probability ``rho``; the rest are fresh draws
    from ``q``. ``rho == tau == 0`` is the independent model.
    """
    rows = int(step_of_row.shape[0])
    k = int(round(q.sum()))
    if rows == 0:
        return np.zeros((0, k), dtype=np.int32)
    if rho <= 0.0:
        return systematic_sample(q, rows, rng)
    per_group = int(row_in_step.max()) // group + 1
    n_steps = int(step_of_row.max()) + 1
    group_of_row = step_of_row.astype(np.int64) * per_group + row_in_step // group
    used, group_index = np.unique(group_of_row, return_inverse=True)
    steps_of_group = used // per_group
    base = systematic_sample(q, len(used), rng)
    if tau > 0.0:
        step_base = systematic_sample(q, n_steps, rng)
        base = mix_rows(step_base[steps_of_group], base, tau, rng)
    return mix_rows(base[group_index], systematic_sample(q, rows, rng), rho, rng)
