# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebalance policy: EMA windows, hysteresis, and a replay of the whole schedule.

DeepInfer's schedule is one slow loop per phase: an EMA over a window of steps
(prefill 100, decode 1000) and a decision every ``interval`` steps (prefill 1000,
decode 10000), judged by ``slowest / mean`` of the per-step rank loads. Here a decision
is a pure function of the EMA, the loads the ranks really had and the placement in
force, so the same code serves the replay of a recorded trace (this module), the
boot-time solver (``solve.py``) and, in phase 2, the runtime balancer:

1. weight per expert: EMA of the pairs (prefill) or of the touched flag (decode) -- the
   decode objective is the count of DISTINCT activated experts per rank, because a
   decode kernel reads each touched expert's weights once;
2. candidate: the swap sequence of ``placement.repair_owner`` from the placement in
   force (``decide_swaps``: a prefix of it moves two experts per swap, and the length is
   the one with the best net benefit) or ``pack_owner`` of the weights, a reshuffle of
   about three quarters of the experts (``decide``, the ``expected`` predictor);
3. gain per step. ``realised`` (default): ``samples[c, r]`` are the loads the ranks
   really had in the window's calls, a candidate shifts rank ``r``'s expected load by
   ``delta[r]`` and the gain is ``mean_c max_r L - mean_c max_r (L + delta)``. A step is
   as slow as its slowest rank, and the slowest rank of a few-row decode step is mostly
   the maximum of random touch counts (or, in prefill, of a chunk's topic), which no
   placement changes; ``expected`` (``unit_us * (max_r W_r(now) - max_r W_r(cand))`` on
   the EMA alone) ignores that and overstates the gain several-fold;
4. move cost: bytes of the experts that change rank over the slowest link, times the
   share of the copy that shows up as added step latency;
5. rebalance only if (a) the gain over the horizon beats ``safety`` times the move cost,
   (b) the relative gain is at least ``min_rel_gain`` and (c) the candidate still wins
   on each half of the window (``holdout``), which stops the policy from chasing one
   transient.
"""

from dataclasses import dataclass

import numpy as np

from tools.rdna.eplb import loads, placement
from tools.rdna.eplb.geometry import DEFAULT_COST, FLASHNEXT, CostModel, Geometry


@dataclass(frozen=True)
class PolicyConfig:
    window: int = 1000  # EMA window in steps (alpha = 2 / (window + 1))
    interval: int = 10000  # steps between decisions
    objective: str = "touch"  # "touch" (decode) or "pairs" (prefill)
    min_rel_gain: float = 0.01  # required predicted gain of the slowest rank's time
    safety: float = 2.0  # gain over the horizon must beat safety x move cost
    holdout: bool = True  # candidate fitted on the older half must win on the newer
    bw_gbps: float = 14.0  # GB/s per link, the x8 P2P plateau [INFERENCE; 25 if x16]
    stall: float = 0.25  # share of the copy time visible as added step latency
    layer_fixed_us: float = 300.0  # per layer plan/commit overhead [INFERENCE]
    predictor: str = "realised"  # "realised" (decide_swaps) or "expected" (decide)
    max_swaps: int = 400  # longest swap sequence one decision considers


def ema_alpha(window: int) -> float:
    return 2.0 / (window + 1.0)


def unit_us(cfg: PolicyConfig, cost: CostModel) -> float:
    """Time one unit of the objective adds to its rank."""
    return cost.expert_us if cfg.objective == "touch" else cost.pair_us


def moved_bytes(
    old: np.ndarray, new: np.ndarray, ranks: int, geometry: Geometry = FLASHNEXT
) -> tuple[int, int]:
    """``(experts moved, bytes over the busiest rank)`` of one layer's permutation.

    Every expert that changes rank is written once by its old rank and received once by
    its new rank; the busiest rank's larger direction bounds the time of a push-style
    copy that runs all ranks concurrently.
    """
    moved = old != new
    n = int(moved.sum())
    sent = np.bincount(old[moved], minlength=ranks)
    recv = np.bincount(new[moved], minlength=ranks)
    per_rank = max(int(sent.max()), int(recv.max())) if n else 0
    return n, per_rank * geometry.param_expert_bytes


def move_us(per_rank_bytes: int, cfg: PolicyConfig) -> float:
    """Latency a move adds: the stalled share of the busiest link's copy time."""
    if per_rank_bytes == 0:
        return 0.0
    return cfg.stall * per_rank_bytes / (cfg.bw_gbps * 1e3) + cfg.layer_fixed_us


def predicted_us(
    weight: np.ndarray,
    owner: np.ndarray,
    ranks: int,
    cfg: PolicyConfig,
    cost: CostModel = DEFAULT_COST,
) -> float:
    """Slowest rank's time per step implied by a per-expert ``weight``."""
    return unit_us(cfg, cost) * float(placement.rank_weight(weight, owner, ranks).max())


@dataclass
class Decision:
    owner: np.ndarray | None  # candidate to move to, None = keep
    gain_us_per_step: float
    move_us: float
    experts_moved: int
    reason: str


def decide(
    weight: np.ndarray,
    current: np.ndarray,
    ranks: int,
    cfg: PolicyConfig,
    *,
    older: np.ndarray | None = None,
    newer: np.ndarray | None = None,
    cost: CostModel = DEFAULT_COST,
    geometry: Geometry = FLASHNEXT,
    steps_in_horizon: int | None = None,
) -> Decision:
    """Hysteresis decision for one layer (``weight`` = the window's per-expert EMA).

    ``older`` / ``newer`` are the two halves of the window for the holdout test.
    """
    cand = placement.pack_owner(weight, ranks)
    now = predicted_us(weight, current, ranks, cfg, cost)
    gain = now - predicted_us(weight, cand, ranks, cfg, cost)
    n_moved, per_rank = moved_bytes(current, cand, ranks, geometry)
    mv = move_us(per_rank, cfg)
    horizon = steps_in_horizon if steps_in_horizon is not None else cfg.interval
    if n_moved == 0:
        return Decision(None, 0.0, 0.0, 0, "same placement")
    if now <= 0 or gain < cfg.min_rel_gain * now:
        return Decision(None, gain, mv, n_moved, "gain below min_rel_gain")
    if gain * horizon < cfg.safety * mv:
        return Decision(None, gain, mv, n_moved, "gain does not pay for the move")
    if cfg.holdout and older is not None and newer is not None:
        fit = placement.pack_owner(older, ranks)
        held = predicted_us(newer, current, ranks, cfg, cost) - predicted_us(
            newer, fit, ranks, cfg, cost
        )
        if held < cfg.min_rel_gain * predicted_us(newer, current, ranks, cfg, cost):
            return Decision(None, gain, mv, n_moved, "holdout gain below min_rel_gain")
    return Decision(cand, gain, mv, n_moved, "rebalance")


def shifted_gain(samples: np.ndarray, delta: np.ndarray) -> float:
    """Mean drop of the slowest rank's load if rank ``r``'s load moved by ``delta[r]``.

    ``samples[c, r]`` is the load rank ``r`` really had in call ``c`` under the
    placement in force, sampling noise included; ``delta[r]`` is the change of its
    expected load under a candidate. The first-order counterfactual keeps the
    fluctuations and moves the means, so the noise floor of ``max_r`` stays in.
    """
    s = np.asarray(samples, dtype=np.float64)
    shifted = s + np.asarray(delta, dtype=np.float64)
    return float(s.max(axis=1).mean() - shifted.max(axis=1).mean())


def decide_swaps(
    weight: np.ndarray,
    current: np.ndarray,
    ranks: int,
    cfg: PolicyConfig,
    samples: np.ndarray,
    *,
    older: np.ndarray | None = None,
    newer: np.ndarray | None = None,
    cost: CostModel = DEFAULT_COST,
    geometry: Geometry = FLASHNEXT,
    steps_in_horizon: int | None = None,
) -> Decision:
    """Hysteresis decision for one layer from the loads the ranks really had.

    ``weight`` is the window's per-expert EMA, ``samples[c, r]`` the realised load of
    rank ``r`` in each call of the window (touched experts for decode, pairs for
    prefill: the ``T`` and ``P`` words of the counter ring). Every prefix of the swap
    sequence of ``repair_owner`` is a candidate; the one with the best
    ``horizon * gain - safety * move`` wins. ``older`` / ``newer`` are the EMA of the
    two halves of the window: with ``holdout`` the candidate must also gain on each
    half of ``samples``.
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] != ranks:
        raise ValueError("samples must be [calls, ranks] with at least one call")
    unit = unit_us(cfg, cost)
    now = unit * float(s.max(axis=1).mean())
    horizon = steps_in_horizon if steps_in_horizon is not None else cfg.interval
    _, swaps = placement.repair_owner(weight, current, ranks, max_swaps=cfg.max_swaps)
    if not swaps:
        return Decision(None, 0.0, 0.0, 0, "same placement")
    base = placement.rank_weight(weight, current, ranks)
    owner = np.asarray(current, dtype=np.int16).copy()
    best: tuple[float, Decision] | None = None
    last = Decision(None, 0.0, 0.0, 0, "gain below min_rel_gain")
    seen_gain = False
    for a, b in swaps:
        owner[a], owner[b] = owner[b], owner[a]
        gain = unit * shifted_gain(
            s, placement.rank_weight(weight, owner, ranks) - base
        )
        n, per_rank = moved_bytes(current, owner, ranks, geometry)
        mv = move_us(per_rank, cfg)
        last = Decision(None, gain, mv, n, "gain below min_rel_gain")
        if now <= 0 or gain < cfg.min_rel_gain * now:
            continue
        seen_gain = True
        net = gain * horizon - cfg.safety * mv
        if net > 0 and (best is None or net > best[0]):
            best = (net, Decision(owner.copy(), gain, mv, n, "rebalance"))
    if best is None:
        if seen_gain:
            last.reason = "gain does not pay for the move"
        return last
    cand = best[1]
    half = s.shape[0] // 2
    if cfg.holdout and older is not None and newer is not None and half >= 1:
        # cross-validation: each half of the calls is judged with the OTHER half's
        # weights, so the noise that shaped the candidate cannot vouch for it
        for seg, w in ((s[half:], older), (s[:half], newer)):
            delta = placement.rank_weight(w, cand.owner, ranks) - placement.rank_weight(
                w, current, ranks
            )
            if unit * shifted_gain(seg, delta) < cfg.min_rel_gain * unit * float(
                seg.max(axis=1).mean()
            ):
                return Decision(
                    None,
                    cand.gain_us_per_step,
                    cand.move_us,
                    cand.experts_moved,
                    "holdout gain below min_rel_gain",
                )
    return cand


def replay_schedule(
    counts: dict[int, np.ndarray],
    ranks: int,
    cfg: PolicyConfig,
    *,
    init: dict[int, np.ndarray] | None = None,
    cost: CostModel = DEFAULT_COST,
    geometry: Geometry = FLASHNEXT,
) -> dict:
    """Run the schedule over per-layer step counts ``counts[layer] = [steps, E]``.

    Returns the time (us) summed over steps and layers with (a) the linear placement,
    (b) the schedule (including its stalled move cost), (c) the best single static
    placement fitted on the whole trace (an in-sample bound for any static plan), plus
    the decisions taken. All layers see the same steps, so ``steps`` is shared.
    """
    layers = sorted(counts)
    steps = counts[layers[0]].shape[0]
    experts = counts[layers[0]].shape[1]
    alpha = ema_alpha(cfg.window)
    lin = placement.linear_owner(experts, ranks)
    owner = {li: (init[li].copy() if init else lin.copy()) for li in layers}
    total = {"linear": 0.0, "schedule": 0.0, "oracle": 0.0}
    moves: list[dict] = []
    last = 0

    def slowest_us(seg: np.ndarray, own: np.ndarray) -> float:
        return float(loads.rank_loads(seg, own, ranks, cost)["us"].max(axis=1).sum())

    def segment(a: int, b: int) -> None:
        for li in layers:
            total["linear"] += slowest_us(counts[li][a:b], lin)
            total["schedule"] += slowest_us(counts[li][a:b], owner[li])

    for t in range(cfg.interval, steps + 1, cfg.interval):
        segment(last, t)
        last = t
        for li in layers:
            c = counts[li][max(0, t - cfg.window) : t]
            span = (c > 0).astype(np.float64) if cfg.objective == "touch" else c
            # EMA over the newest window of steps, newest weighted highest
            wts = (1.0 - alpha) ** np.arange(span.shape[0])[::-1]
            weight = (span * wts[:, None]).sum(axis=0) / wts.sum()
            mid = span.shape[0] // 2
            older = span[:mid].mean(axis=0) if mid else weight
            newer = span[mid:].mean(axis=0)
            if cfg.predictor == "realised":
                stat = "T" if cfg.objective == "touch" else "P"
                d = decide_swaps(
                    weight,
                    owner[li],
                    ranks,
                    cfg,
                    loads.rank_loads(c, owner[li], ranks, cost)[stat],
                    older=older,
                    newer=newer,
                    cost=cost,
                    geometry=geometry,
                )
            else:
                d = decide(
                    weight,
                    owner[li],
                    ranks,
                    cfg,
                    older=older,
                    newer=newer,
                    cost=cost,
                    geometry=geometry,
                )
            if d.owner is not None:
                moves.append(
                    {
                        "step": t,
                        "layer": li,
                        "experts_moved": d.experts_moved,
                        "gain_us_per_step": d.gain_us_per_step,
                        "move_us": d.move_us,
                    }
                )
                total["schedule"] += d.move_us
                owner[li] = d.owner
    if last < steps:
        segment(last, steps)
    for li in layers:
        c = counts[li]
        w = (c > 0).mean(axis=0) if cfg.objective == "touch" else c.mean(axis=0)
        total["oracle"] += slowest_us(c, placement.pack_owner(w, ranks))
    return {
        "steps": steps,
        "layers": len(layers),
        "total_us": total,
        "rebalances": len(moves),
        "schedule_vs_linear_pct": 100.0 * (1.0 - total["schedule"] / total["linear"]),
        "oracle_vs_linear_pct": 100.0 * (1.0 - total["oracle"] / total["linear"]),
        "moves": moves,
    }
