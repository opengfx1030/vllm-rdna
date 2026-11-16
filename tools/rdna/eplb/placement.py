# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-to-rank placement policies and replica routing (numpy, deterministic).

A *placement* of one layer is ``owner[e]`` (the rank of the primary copy of logical
expert ``e``; every rank holds exactly ``E / R`` experts: permutation only) or, with
redundancy, a :class:`Holders` table (several ranks may hold an expert).

Policies
--------
``linear_owner`` / ``round_robin_owner``
    The two placements the engine ships (``expert_placement_strategy``).
``pack_owner``
    Equal-cardinality min-max packing of a weight per expert: longest-processing-time
    first (what DeepSeek's ``balanced_packing`` and the fork's
    ``tools/rdna/prof/moe_placement.py`` do) followed by pairwise swaps between the
    heaviest rank and every other rank until no swap lowers the maximum.
``repair_owner``
    The same swap descent started from the placement in force: the swaps in the order
    of their gain, so a prefix is the cheapest placement that captures that much (a
    swap moves two experts; ``pack_owner`` from scratch reshuffles about 3/4 of them).
``pack_empirical``
    Chunk-aware: swap hill climbing on ``sum_steps max_rank(cost)`` over measured
    per-step costs (DeepInfer's objective), starting from ``pack_owner``; sees which
    experts fire together, which the marginal cannot.
``replicate_pack``
    DeepSeek-EPLB style redundancy: replicas go to ``argmax(weight / copies)``, then
    the copies (weight ``w / copies``) are packed over ranks that do not hold them yet.
``hot_holders``
    The ``h`` heaviest experts on every rank (the flexible set for per-step routing).

Replica routing (which copy serves a pair) is in :func:`route_loads`:
``primary`` (copy 0 only), ``hash`` (the engine's: copy ``hash(row) % copies``, per
row, so one expert can be live on several ranks in a step) and ``min_touch`` (METRO,
arXiv:2512.09277: every touched expert runs on ONE copy, picked greedily so the
slowest rank has the least work; all ranks hold the whole batch's top-k ids under
all-gather/reduce-scatter EP, so every rank reaches the same decision locally).
"""

from dataclasses import dataclass

import numpy as np

from tools.rdna.eplb.geometry import DEFAULT_COST, CostModel

KNUTH = 2654435769  # the engine's replica hash multiplier


def linear_owner(experts: int, ranks: int) -> np.ndarray:
    """Rank ``e // (E / R)``: contiguous blocks (the engine's ``linear``)."""
    if experts % ranks:
        raise ValueError(f"{experts} experts do not divide over {ranks} ranks")
    return (np.arange(experts) // (experts // ranks)).astype(np.int16)


def round_robin_owner(experts: int, ranks: int) -> np.ndarray:
    if experts % ranks:
        raise ValueError(f"{experts} experts do not divide over {ranks} ranks")
    return (np.arange(experts) % ranks).astype(np.int16)


def rank_weight(weights: np.ndarray, owner: np.ndarray, ranks: int) -> np.ndarray:
    return np.bincount(owner, weights=weights, minlength=ranks)


def balance_ratio(weights: np.ndarray, owner: np.ndarray, ranks: int) -> float:
    """Max / mean of the per-rank weight sums (1.0 = perfectly balanced)."""
    per_rank = rank_weight(weights, owner, ranks)
    return float(per_rank.max() / per_rank.mean()) if per_rank.sum() > 0 else 1.0


def _swap_descent(
    w: np.ndarray, owner: np.ndarray, ranks: int, max_swaps: int
) -> list[tuple[int, int]]:
    """Best-improvement swap descent on ``owner`` (in place); the swaps it made.

    Each step swaps one expert of the heaviest rank with one of another rank, the pair
    that lowers the maximum the most. A swap is ``(a, b)``: expert ``a`` leaves the
    heaviest rank for the other rank, expert ``b`` takes its place.
    """
    done: list[tuple[int, int]] = []
    for _ in range(max_swaps):
        load = rank_weight(w, owner, ranks)
        hi = int(np.argmax(load))
        best: tuple[float, int, int, int] | None = None
        mem_hi = np.flatnonzero(owner == hi)
        for lo in range(ranks):
            gap = load[hi] - load[lo]
            if lo == hi or gap <= 0:
                continue
            mem_lo = np.flatnonzero(owner == lo)
            delta = w[mem_hi][:, None] - w[mem_lo][None, :]
            ok = (delta > 0) & (delta < gap)
            if not ok.any():
                continue
            new_max = np.where(
                ok, np.maximum(load[hi] - delta, load[lo] + delta), np.inf
            )
            i, j = np.unravel_index(int(np.argmin(new_max)), new_max.shape)
            cand = (float(new_max[i, j]), lo, int(mem_hi[i]), int(mem_lo[j]))
            if best is None or cand < best:
                best = cand
        if best is None or best[0] >= load[hi] - 1e-12 * max(load[hi], 1.0):
            break
        _, lo, a, b = best
        owner[a], owner[b] = lo, hi
        done.append((a, b))
    return done


def pack_owner(
    weights: np.ndarray, ranks: int, *, refine_iters: int = 400
) -> np.ndarray:
    """Equal-cardinality min-max packing of ``weights``: ``owner[e]`` in ``[0, R)``.

    Deterministic: ties break by expert id and then by rank.
    """
    w = np.asarray(weights, dtype=np.float64)
    experts = w.shape[0]
    if experts % ranks:
        raise ValueError(f"{experts} experts do not divide over {ranks} ranks")
    cap = experts // ranks
    owner = np.full(experts, -1, dtype=np.int16)
    load = np.zeros(ranks)
    room = np.full(ranks, cap)
    for e in np.lexsort((np.arange(experts), -w)):
        free = np.flatnonzero(room > 0)
        r = free[np.argmin(load[free])]
        owner[e] = r
        load[r] += w[e]
        room[r] -= 1
    _swap_descent(w, owner, ranks, refine_iters)
    return owner


def repair_owner(
    weights: np.ndarray, current: np.ndarray, ranks: int, *, max_swaps: int = 400
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Lower the heaviest rank of ``current`` by swaps only: ``(owner, swaps)``.

    ``pack_owner`` ignores where the experts are now and reshuffles about three
    quarters of them; a swap moves two experts, and the first swaps carry most of the
    gain, so the move cost (bytes over the link) is what the caller trades against the
    gain by taking a prefix of ``swaps`` (``a`` leaves, ``b`` arrives, in order).
    """
    w = np.asarray(weights, dtype=np.float64)
    owner = np.asarray(current, dtype=np.int16).copy()
    swaps = _swap_descent(w, owner, ranks, max_swaps)
    return owner, swaps


def step_max_sum(costs: np.ndarray, owner: np.ndarray, ranks: int) -> float:
    """``sum_s max_r sum_{e on r} costs[s, e]``: what the slowest rank adds up to."""
    onehot = np.zeros((owner.shape[0], ranks))
    onehot[np.arange(owner.shape[0]), owner] = 1.0
    return float((np.asarray(costs, dtype=np.float64) @ onehot).max(axis=1).sum())


def pack_empirical(
    costs: np.ndarray,
    ranks: int,
    init: np.ndarray,
    *,
    iters: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Chunk-aware packing: minimise ``step_max_sum(costs, owner)`` from ``init``.

    ``costs[s, e]`` is what expert ``e`` costs in step ``s`` (pairs, a touched flag, or
    time). The marginal packing balances the average over steps; this also sees which
    experts fire together, so a co-activating cluster is spread over the ranks.
    Randomised pairwise-swap hill climbing (one expert of the rank that is slowest in
    most steps against one expert of another rank); deterministic for a fixed ``seed``.
    Fit on training steps and judge on held-out ones: with few steps it overfits.
    """
    c = np.asarray(costs, dtype=np.float64)
    owner = np.asarray(init, dtype=np.int16).copy()
    load = np.stack([c[:, owner == r].sum(axis=1) for r in range(ranks)], axis=1)
    best = float(load.max(axis=1).sum())
    members = [np.flatnonzero(owner == r) for r in range(ranks)]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        slowest = np.bincount(load.argmax(axis=1), minlength=ranks).astype(np.float64)
        hi = int(rng.choice(ranks, p=slowest / slowest.sum()))
        lo = int(rng.integers(ranks - 1))
        lo += lo >= hi
        i = int(rng.integers(members[hi].size))
        j = int(rng.integers(members[lo].size))
        a, b = int(members[hi][i]), int(members[lo][j])
        d = c[:, a] - c[:, b]
        trial = load.copy()
        trial[:, hi] -= d
        trial[:, lo] += d
        obj = float(trial.max(axis=1).sum())
        if obj < best - 1e-12 * max(best, 1.0):
            best, load = obj, trial
            owner[a], owner[b] = lo, hi
            members[hi][i], members[lo][j] = b, a
    return owner


@dataclass(frozen=True)
class Holders:
    """Which ranks hold each expert: ``table[e, j]`` is the rank of copy ``j``.

    Copy 0 is the primary; unused columns are -1.
    """

    ranks: int
    table: np.ndarray  # int16 [E, ranks]

    @classmethod
    def from_owner(cls, owner: np.ndarray, ranks: int) -> "Holders":
        table = np.full((owner.shape[0], ranks), -1, dtype=np.int16)
        table[:, 0] = owner
        return cls(ranks, table)

    @property
    def copies(self) -> np.ndarray:
        return (self.table >= 0).sum(axis=1)

    @property
    def owner(self) -> np.ndarray:
        return self.table[:, 0]

    def slots_per_rank(self) -> np.ndarray:
        held = self.table[self.table >= 0]
        return np.bincount(held, minlength=self.ranks)

    def extra_slots_per_rank(self, experts: int) -> int:
        """Slots beyond the permutation-only ``E / R`` on the fullest rank."""
        return int(self.slots_per_rank().max() - experts // self.ranks)

    def validate(self) -> None:
        for e, row in enumerate(self.table):
            held = row[row >= 0]
            if held.size == 0 or len(set(held.tolist())) != held.size:
                raise ValueError(f"expert {e} has no copy or a duplicate rank")


def replicate_pack(
    weights: np.ndarray, ranks: int, extra_slots_per_rank: int
) -> Holders:
    """DeepSeek-EPLB redundancy: replicate by ``argmax(w / copies)``, pack the copies.

    Every rank ends with ``E / R + extra_slots_per_rank`` distinct experts.
    """
    w = np.asarray(weights, dtype=np.float64)
    experts = w.shape[0]
    if extra_slots_per_rank == 0:
        return Holders.from_owner(pack_owner(w, ranks), ranks)
    cap = experts // ranks + extra_slots_per_rank
    copies = np.ones(experts, dtype=np.int64)
    for _ in range(ranks * extra_slots_per_rank):
        score = np.where(copies < ranks, w / copies, -np.inf)
        copies[int(np.argmax(score))] += 1
    items = [(e, j) for e in range(experts) for j in range(int(copies[e]))]
    items.sort(key=lambda it: (-w[it[0]] / copies[it[0]], it[0], it[1]))
    load = np.zeros(ranks)
    room = np.full(ranks, cap)
    table = np.full((experts, ranks), -1, dtype=np.int16)
    for e, _ in items:
        has = set(table[e][table[e] >= 0].tolist())
        free = [r for r in range(ranks) if room[r] > 0 and r not in has]
        if not free:
            raise ValueError("replica packing ran out of ranks without the expert")
        r = min(free, key=lambda x: (load[x], x))
        table[e, int((table[e] >= 0).sum())] = r
        load[r] += w[e] / copies[e]
        room[r] -= 1
    return Holders(ranks, table)


def hot_holders(
    owner: np.ndarray, weights: np.ndarray, ranks: int, hot: int
) -> Holders:
    """The ``hot`` heaviest experts live on every rank (primaries stay put)."""
    holders = Holders.from_owner(owner, ranks)
    table = holders.table.copy()
    for e in np.lexsort((np.arange(len(weights)), -np.asarray(weights)))[:hot]:
        others = [r for r in range(ranks) if r != int(owner[e])]
        table[e, 1 : 1 + len(others)] = others
    return Holders(ranks, table)


def route_loads(
    ids: np.ndarray,
    step_of_row: np.ndarray,
    row_in_step: np.ndarray,
    n_steps: int,
    holders: Holders,
    policy: str,
    cost: CostModel = DEFAULT_COST,
) -> dict[str, np.ndarray]:
    """Per-step, per-rank loads ``T`` (live experts), ``P`` (pairs), ``us`` (time).

    ``ids`` is ``[rows, K]`` expert ids; ``policy`` is ``primary``, ``hash`` or
    ``min_touch``. Each (step, expert, rank) with at least one pair costs
    ``max(expert_us, pair_us * pairs)``; a rank adds ``fixed_us``.
    """
    ranks, experts = holders.ranks, holders.table.shape[0]
    copies = holders.copies
    rows, k = ids.shape
    e_flat = ids.reshape(-1).astype(np.int64)
    s_flat = np.repeat(step_of_row.astype(np.int64), k)
    if policy == "primary" or not (copies > 1).any():
        r_flat = holders.table[e_flat, 0].astype(np.int64)
    elif policy == "hash":
        tok = (row_in_step.astype(np.uint64) * np.uint64(KNUTH)) & np.uint64(0xFFFFFFFF)
        tok = np.repeat(tok, k)
        pick = (tok % copies[e_flat].astype(np.uint64)).astype(np.int64)
        r_flat = holders.table[e_flat, pick].astype(np.int64)
    elif policy == "min_touch":
        return _min_touch(e_flat, s_flat, n_steps, holders, cost)
    else:
        raise ValueError(f"unknown routing policy {policy!r}")
    key = (s_flat * experts + e_flat) * ranks + r_flat
    uniq, pairs = np.unique(key, return_counts=True)
    step = uniq // (experts * ranks)
    rank = uniq % ranks
    flat = step * ranks + rank
    size = n_steps * ranks
    t = np.bincount(flat, minlength=size).reshape(n_steps, ranks).astype(np.float64)
    p = np.bincount(flat, weights=pairs, minlength=size).reshape(n_steps, ranks)
    us = np.bincount(flat, weights=cost.rank_us(pairs), minlength=size)
    us = us.reshape(n_steps, ranks) + cost.fixed_us
    return {"T": t, "P": p, "us": us}


def _min_touch(
    e_flat: np.ndarray,
    s_flat: np.ndarray,
    n_steps: int,
    holders: Holders,
    cost: CostModel,
) -> dict[str, np.ndarray]:
    ranks, experts = holders.ranks, holders.table.shape[0]
    copies = holders.copies
    uniq, pairs = np.unique(s_flat * experts + e_flat, return_counts=True)
    step, expert = uniq // experts, uniq % experts
    per_expert_us = cost.rank_us(pairs)
    bounds = np.searchsorted(step, np.arange(n_steps + 1))
    t = np.zeros((n_steps, ranks))
    p = np.zeros((n_steps, ranks))
    us = np.zeros((n_steps, ranks))
    for s in range(n_steps):
        lo, hi = bounds[s], bounds[s + 1]
        ex, pr, tm = expert[lo:hi], pairs[lo:hi], per_expert_us[lo:hi]
        fixed = copies[ex] == 1
        prim = holders.table[ex[fixed], 0].astype(np.int64)
        np.add.at(t[s], prim, 1.0)
        np.add.at(p[s], prim, pr[fixed])
        np.add.at(us[s], prim, tm[fixed])
        flex = np.flatnonzero(~fixed)
        for i in flex[np.lexsort((ex[flex], -tm[flex]))]:
            cands = holders.table[ex[i], : copies[ex[i]]].astype(np.int64)
            r = int(cands[np.argmin(us[s][cands])])
            t[s, r] += 1.0
            p[s, r] += pr[i]
            us[s, r] += tm[i]
    return {"T": t, "P": p, "us": us + cost.fixed_us}
