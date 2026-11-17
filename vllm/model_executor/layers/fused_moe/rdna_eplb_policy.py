# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-numpy rebalance policy shared by the study tools and the online planner.

This module imports only numpy and the standard library, on purpose: the online
planner runs it in a helper process that must not import torch or the ``vllm``
package (``rdna_eplb_planner.py`` loads it by file path), and the CPU study tools
(``tools/rdna/eplb``) load it the same way, so there is one copy of the policy.

A decision is a pure function of the weights per expert, the loads the ranks really
had and the placement in force (see ``tools/rdna/eplb/policy.py`` for the schedule
around it: EMA windows, hysteresis, replay of a recorded trace):

1. weight per expert: the decode touch frequency or the prefill pairs per call;
2. candidate: the swap sequence of :func:`repair_owner` from the placement in force
   (a swap moves two experts; every prefix is a placement);
3. gain per step (:func:`decide_swaps`): ``samples[c, r]`` are the loads the ranks
   really had in the window's calls, a candidate shifts rank ``r``'s expected load
   by ``delta[r]`` and the gain is ``mean_c max_r L - mean_c max_r (L + delta)``;
4. move cost (:func:`move_us`): bytes of the experts that change rank over the
   slowest link, times the share of the copy that shows up as added step latency;
5. rebalance only if the gain over the horizon beats ``safety`` times the move cost,
   the relative gain is at least ``min_rel_gain`` and (``holdout``) the candidate
   still wins on each half of the window judged with the other half's weights.

``Geometry`` and ``CostModel`` carry the byte and time constants; every constant
keeps its source (see their docstrings).
"""

from dataclasses import dataclass

import numpy as np

# ---- geometry and cost model --------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Qwen3.8-Flash-Next W4A16 on a TP4/EP4 V620 island."""

    layers: int = 48
    experts: int = 512
    top_k: int = 10
    # packed int4 2,457,600 + fp16 scales 76,800 (symmetric, group 128)
    expert_bytes: int = 2_534_400
    # constant synthetic qzeros (0x77777777) the RDNA2 layout keeps as parameters
    resident_extra_bytes: int = 19_200
    # the MTP layer's 512 BF16 experts (kind "mtp", layer index 48)
    mtp_expert_bytes: int = 9_830_400
    # measured, TP4 flashnext-mtp2: 266,945 KV tokens in a 3.75 GiB KV budget
    kv_tokens: int = 266_945
    kv_bytes: int = 4_026_531_840

    @property
    def param_expert_bytes(self) -> int:
        """Bytes one expert occupies in the parameters (what a slot costs)."""
        return self.expert_bytes + self.resident_extra_bytes

    @property
    def slot_bytes_all_layers(self) -> int:
        """One extra expert slot on every layer, per rank."""
        return self.layers * self.param_expert_bytes

    def redundancy_bytes(self, extra_slots_per_rank: int) -> int:
        """Per-rank bytes of ``extra_slots_per_rank`` slots on every layer."""
        return int(extra_slots_per_rank) * self.slot_bytes_all_layers

    def kv_tokens_lost(self, nbytes: float) -> float:
        """KV capacity (tokens) that ``nbytes`` of weights take from the budget."""
        return nbytes * self.kv_tokens / self.kv_bytes

    def kv_fraction_lost(self, nbytes: float) -> float:
        return nbytes / self.kv_bytes

    def local_experts(self, ranks: int) -> int:
        if self.experts % ranks:
            raise ValueError(f"{self.experts} experts do not divide over {ranks} ranks")
        return self.experts // ranks

    def moved_bytes(self, experts_moved: int, layers: int | None = None) -> int:
        """Bytes a rebalance transfers: payload per moved (expert, layer)."""
        return int(experts_moved) * self.expert_bytes * (layers or 1)


@dataclass(frozen=True)
class CostModel:
    """Time model of one MoE layer on one rank.

    Two measured anchors:

    * decode (bandwidth bound): one touched expert costs ``expert_us`` whatever its
      row count, the weight read of 2.53 MB at about 380 GB/s (``hetero_moe``
      constants for the V620, ``--expert-us 6.7 --expert-fixed-us 3.1``). Check: at
      c=1 MTP2 (3 rows) the model gives 58 us per layer against 52 us measured in
      the decode-step trace (``docs/rdna2/async-stall-profile-2026-11-02.md``);
    * prefill (compute bound): ``pair_us`` per routed (row, expert) pair, derived
      from the 2026-11-03 rocprof table of a 2048-token step (MoE int4 GEMM tile 4,
      208.6 ms / 48 layers = 4.35 ms per layer, ~5,120 local pairs on 128 touched
      experts: (4.35 ms - 128 x 6.7 us) / 5,120 = 0.68 us). The shipped W4A8 v2
      tile-8 kernel is faster (1.75x at 2048 tokens), so absolute prefill times are
      an upper bound; only ratios between placements are used.

    A rank's time in a layer is ``fixed_us + sum over its touched experts of
    max(expert_us, pair_us * rows_of_the_expert)``: the same roofline for both
    phases. The layer finishes with its slowest rank.
    """

    expert_us: float = 6.7
    fixed_us: float = 3.1
    pair_us: float = 0.68

    def rank_us(self, rows_per_expert: np.ndarray) -> np.ndarray:
        """Per-expert time ``max(expert_us, pair_us * rows)``, zero for idle experts."""
        rows = np.asarray(rows_per_expert, dtype=np.float64)
        return np.where(rows > 0, np.maximum(self.expert_us, self.pair_us * rows), 0.0)


FLASHNEXT = Geometry()
DEFAULT_COST = CostModel()

# ---- placement primitives -----------------------------------------------------


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


# ---- policy -------------------------------------------------------------------


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


@dataclass
class Decision:
    owner: np.ndarray | None  # candidate to move to, None = keep
    gain_us_per_step: float
    move_us: float
    experts_moved: int
    reason: str
    # the prefix of the ``repair_owner`` swap sequence that reaches ``owner``
    swaps: tuple[tuple[int, int], ...] = ()


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
    distinct: bool = False,
) -> Decision:
    """Hysteresis decision for one layer from the loads the ranks really had.

    ``weight`` is the window's per-expert EMA, ``samples[c, r]`` the realised load of
    rank ``r`` in each call of the window (touched experts for decode, pairs for
    prefill: the ``T`` and ``P`` words of the counter ring). Every prefix of the swap
    sequence of ``repair_owner`` is a candidate; the one with the best
    ``horizon * gain - safety * move`` wins. ``older`` / ``newer`` are the EMA of the
    two halves of the window: with ``holdout`` the candidate must also gain on each
    half of ``samples``. With ``distinct`` the candidates stop before the first swap
    that would move an expert a second time (the online executor moves each expert at
    most once per plan).
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] != ranks:
        raise ValueError("samples must be [calls, ranks] with at least one call")
    unit = unit_us(cfg, cost)
    now = unit * float(s.max(axis=1).mean())
    horizon = steps_in_horizon if steps_in_horizon is not None else cfg.interval
    _, swaps = repair_owner(weight, current, ranks, max_swaps=cfg.max_swaps)
    if not swaps:
        return Decision(None, 0.0, 0.0, 0, "same placement")
    base = rank_weight(weight, current, ranks)
    owner = np.asarray(current, dtype=np.int16).copy()
    best: tuple[float, Decision] | None = None
    last = Decision(None, 0.0, 0.0, 0, "gain below min_rel_gain")
    seen_gain = False
    moved: set[int] = set()
    for i, (a, b) in enumerate(swaps):
        if distinct and (a in moved or b in moved):
            break  # a later swap would move an expert that already moved
        moved.update((a, b))
        owner[a], owner[b] = owner[b], owner[a]
        gain = unit * shifted_gain(s, rank_weight(weight, owner, ranks) - base)
        n, per_rank = moved_bytes(current, owner, ranks, geometry)
        mv = move_us(per_rank, cfg)
        last = Decision(None, gain, mv, n, "gain below min_rel_gain")
        if now <= 0 or gain < cfg.min_rel_gain * now:
            continue
        seen_gain = True
        net = gain * horizon - cfg.safety * mv
        if net > 0 and (best is None or net > best[0]):
            best = (
                net,
                Decision(owner.copy(), gain, mv, n, "rebalance", tuple(swaps[: i + 1])),
            )
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
            delta = rank_weight(w, cand.owner, ranks) - rank_weight(w, current, ranks)
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
