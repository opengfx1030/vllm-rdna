# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebalance policy: EMA windows, hysteresis, and a replay of the whole schedule.

DeepInfer's schedule is one slow loop per phase: an EMA over a window of steps
(prefill 100, decode 1000) and a decision every ``interval`` steps (prefill 1000,
decode 10000), judged by ``slowest / mean`` of the per-step rank loads. Here a decision
is a pure function of the EMA, the loads the ranks really had and the placement in
force, so the same code serves the replay of a recorded trace (this module), the
boot-time solver (``solve.py``) and the runtime balancer. The shared part
(``PolicyConfig``, ``Decision``, ``decide_swaps``, ``shifted_gain``, ``moved_bytes``,
``move_us``, ``repair_owner``, the geometry and cost model) lives in
``vllm/model_executor/layers/fused_moe/rdna_eplb_policy.py`` (numpy only; the online
planner runs it in a helper process) and is used here through ``core``:

1. weight per expert: EMA of the pairs (prefill) or of the touched flag (decode) -- the
   decode objective is the count of DISTINCT activated experts per rank, because a
   decode kernel reads each touched expert's weights once;
2. candidate: the swap sequence of ``repair_owner`` from the placement in force
   (``core.decide_swaps``: a prefix of it moves two experts per swap, and the length is
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

import numpy as np

from tools.rdna.eplb import loads, placement
from tools.rdna.eplb._core import core


def predicted_us(
    weight: np.ndarray,
    owner: np.ndarray,
    ranks: int,
    cfg: core.PolicyConfig,
    cost: core.CostModel = core.DEFAULT_COST,
) -> float:
    """Slowest rank's time per step implied by a per-expert ``weight``."""
    return core.unit_us(cfg, cost) * float(core.rank_weight(weight, owner, ranks).max())


def decide(
    weight: np.ndarray,
    current: np.ndarray,
    ranks: int,
    cfg: core.PolicyConfig,
    *,
    older: np.ndarray | None = None,
    newer: np.ndarray | None = None,
    cost: core.CostModel = core.DEFAULT_COST,
    geometry: core.Geometry = core.FLASHNEXT,
    steps_in_horizon: int | None = None,
) -> core.Decision:
    """Hysteresis decision for one layer (``weight`` = the window's per-expert EMA).

    ``older`` / ``newer`` are the two halves of the window for the holdout test.
    """
    cand = placement.pack_owner(weight, ranks)
    now = predicted_us(weight, current, ranks, cfg, cost)
    gain = now - predicted_us(weight, cand, ranks, cfg, cost)
    n_moved, per_rank = core.moved_bytes(current, cand, ranks, geometry)
    mv = core.move_us(per_rank, cfg)
    horizon = steps_in_horizon if steps_in_horizon is not None else cfg.interval
    if n_moved == 0:
        return core.Decision(None, 0.0, 0.0, 0, "same placement")
    if now <= 0 or gain < cfg.min_rel_gain * now:
        return core.Decision(None, gain, mv, n_moved, "gain below min_rel_gain")
    if gain * horizon < cfg.safety * mv:
        return core.Decision(None, gain, mv, n_moved, "gain does not pay for the move")
    if cfg.holdout and older is not None and newer is not None:
        fit = placement.pack_owner(older, ranks)
        held = predicted_us(newer, current, ranks, cfg, cost) - predicted_us(
            newer, fit, ranks, cfg, cost
        )
        if held < cfg.min_rel_gain * predicted_us(newer, current, ranks, cfg, cost):
            return core.Decision(
                None, gain, mv, n_moved, "holdout gain below min_rel_gain"
            )
    return core.Decision(cand, gain, mv, n_moved, "rebalance")


def replay_schedule(
    counts: dict[int, np.ndarray],
    ranks: int,
    cfg: core.PolicyConfig,
    *,
    init: dict[int, np.ndarray] | None = None,
    cost: core.CostModel = core.DEFAULT_COST,
    geometry: core.Geometry = core.FLASHNEXT,
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
    alpha = core.ema_alpha(cfg.window)
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
                d = core.decide_swaps(
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
