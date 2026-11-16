# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/rdna/eplb (geometry, placement, loads, simulator, policy).

Pure numpy; no torch, no GPU. Everything is seeded, so the numbers are exact.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.rdna.eplb import geometry, loads, placement, policy, simulate
from tools.rdna.eplb.geometry import FLASHNEXT, CostModel

ROOT = Path(__file__).resolve().parents[3]
T10 = ROOT / "bench_results" / "2026-11-07_expert-stats-t10"


# ---- geometry ---------------------------------------------------------------


def test_geometry_matches_the_checkpoint_bytes_table():
    assert geometry.from_bytes_table() == FLASHNEXT


def test_redundancy_costs_follow_from_the_byte_table():
    g = FLASHNEXT
    assert g.param_expert_bytes == 2_553_600
    assert g.slot_bytes_all_layers == 48 * 2_553_600 == 122_572_800
    assert g.redundancy_bytes(4) == 4 * 122_572_800
    assert round(g.kv_fraction_lost(g.slot_bytes_all_layers), 3) == 0.030
    assert round(g.kv_tokens_lost(g.slot_bytes_all_layers)) == 8126
    assert g.local_experts(4) == 128
    with pytest.raises(ValueError):
        g.local_experts(5)


def test_cost_model_is_a_roofline_with_idle_experts_free():
    c = CostModel()
    got = c.rank_us(np.array([0, 1, 3, 20, 100]))
    assert np.allclose(got, [0.0, 6.7, 6.7, 13.6, 68.0])


# ---- placement --------------------------------------------------------------


def test_linear_and_round_robin_owners():
    assert placement.linear_owner(8, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert placement.round_robin_owner(8, 4).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    with pytest.raises(ValueError):
        placement.linear_owner(10, 4)


def test_pack_owner_balances_the_textbook_case_with_equal_cardinality():
    w = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    owner = placement.pack_owner(w, 2)
    assert np.bincount(owner).tolist() == [4, 4]
    assert placement.rank_weight(w, owner, 2).tolist() == [18.0, 18.0]


def test_pack_owner_is_deterministic_and_beats_linear_on_zipf():
    rng = np.random.default_rng(3)
    w = (np.arange(512) + 1.0) ** -0.9
    w = w[rng.permutation(512)]
    a = placement.pack_owner(w, 4)
    b = placement.pack_owner(w, 4)
    assert np.array_equal(a, b)
    assert np.bincount(a).tolist() == [128] * 4
    lin = placement.balance_ratio(w, placement.linear_owner(512, 4), 4)
    assert placement.balance_ratio(w, a, 4) < min(lin, 1.01)


def test_pack_empirical_splits_a_co_activating_cluster_over_the_ranks():
    # Expert e belongs to cluster e % 4; one cluster fires per step, all its
    # experts at once. Marginals are uniform, so the marginal packing has nothing to
    # balance, but a start with each cluster on one rank is 4x too slow per step.
    e, r, steps = 32, 4, 64
    costs = np.zeros((steps, e))
    for s in range(steps):
        costs[s, np.arange(e) % 4 == s % 4] = 1.0
    init = placement.round_robin_owner(e, r)  # cluster c entirely on rank c
    before = placement.step_max_sum(costs, init, r)
    owner = placement.pack_empirical(costs, r, init, iters=3000, seed=1)
    after = placement.step_max_sum(costs, owner, r)
    assert before == 8.0 * steps
    assert after <= 0.5 * before
    assert np.bincount(owner, minlength=r).tolist() == [e // r] * r
    assert np.array_equal(
        owner, placement.pack_empirical(costs, r, init, iters=3000, seed=1)
    )


def test_replicate_pack_fills_every_rank_with_distinct_experts():
    w = (np.arange(64) + 1.0) ** -1.0
    h = placement.replicate_pack(w, 4, 3)
    h.validate()
    assert h.slots_per_rank().tolist() == [64 // 4 + 3] * 4
    assert h.copies.sum() == 64 + 4 * 3
    assert h.copies[0] > h.copies[-1] >= 1  # the heaviest expert gets the most copies
    assert h.extra_slots_per_rank(64) == 3


def test_hot_holders_put_the_heaviest_experts_on_every_rank():
    w = np.arange(16, dtype=float)
    owner = placement.round_robin_owner(16, 4)
    h = placement.hot_holders(owner, w, 4, 2)
    h.validate()
    assert (h.copies[[15, 14]] == 4).all() and (h.copies[:14] == 1).all()
    # e15 sits on ranks 0-2 besides its primary (3), e14 on ranks 0, 1, 3
    assert h.slots_per_rank().tolist() == [6, 6, 5, 5]
    assert h.extra_slots_per_rank(16) == 2


# ---- loads and replica routing -------------------------------------------------------


def test_rank_loads_of_a_hand_checked_step():
    ids = np.array([[0, 2], [1, 2]])  # e0, e1 on rank 0; e2 (two pairs) on rank 1
    counts = loads.step_counts(ids, np.array([0, 0]), 1, 4)
    assert counts.tolist() == [[1, 1, 2, 0]]
    owner = np.array([0, 0, 1, 1])
    res = loads.rank_loads(counts, owner, 2)
    assert res["T"].tolist() == [[2.0, 1.0]]
    assert res["P"].tolist() == [[2.0, 2.0]]
    assert np.allclose(res["us"], [[2 * 6.7 + 3.1, 6.7 + 3.1]])
    assert loads.max_over_mean(res["T"]) == pytest.approx(2.0 / 1.5)


def _two_rank_holders(replica_of_e0: bool) -> placement.Holders:
    table = np.full((4, 2), -1, dtype=np.int16)
    table[:, 0] = [1, 0, 1, 1]  # e0's primary is on rank 1, the loaded rank
    if replica_of_e0:
        table[0, 1] = 0
    return placement.Holders(2, table)


def test_min_touch_routes_the_replicated_expert_to_the_idle_rank():
    ids = np.array([[0, 1], [1, 2], [1, 3]])
    sor, ris = np.zeros(3, dtype=int), np.arange(3)
    holders = _two_rank_holders(True)
    primary = placement.route_loads(ids, sor, ris, 1, holders, "primary")
    assert primary["T"].tolist() == [[1.0, 3.0]]
    smart = placement.route_loads(ids, sor, ris, 1, holders, "min_touch")
    assert smart["T"].tolist() == [[2.0, 2.0]]
    assert smart["T"].sum() == primary["T"].sum()  # no activation inflation


def test_hash_routing_inflates_the_live_experts_min_touch_does_not():
    # e0 is in rows 0 and 1; the row hash sends them to different copies
    ids = np.array([[0, 1], [0, 2]])
    sor, ris = np.zeros(2, dtype=int), np.arange(2)
    holders = _two_rank_holders(True)
    hashed = placement.route_loads(ids, sor, ris, 1, holders, "hash")
    smart = placement.route_loads(ids, sor, ris, 1, holders, "min_touch")
    assert hashed["T"].sum() == 4.0  # 3 distinct experts, e0 live on both ranks
    assert smart["T"].sum() == 3.0
    with pytest.raises(ValueError):
        placement.route_loads(ids, sor, ris, 1, holders, "nope")


# ---- simulator ----------------------------------------------------------------------


def test_zipf_alpha_reproduces_the_requested_top_share():
    alpha = simulate.zipf_alpha_for_top_share(64, 0.388)
    q = np.sort(simulate.inclusion_probs((np.arange(512) + 1.0) ** -alpha, 10))[::-1]
    assert q[:64].sum() / 10 == pytest.approx(0.388, abs=2e-3)
    rec = simulate.zipf_record(layers=3, experts=64, top_k=4, alpha=0.8, seed=5)
    again = simulate.zipf_record(layers=3, experts=64, top_k=4, alpha=0.8, seed=5)
    assert np.array_equal(rec.layers[1].decode, again.layers[1].decode)
    assert not np.array_equal(rec.layers[0].decode, rec.layers[1].decode)


def _small_data(layers: int = 3):
    rec = simulate.zipf_record(layers=layers, alpha=0.9, seed=2)
    return simulate.layer_data(rec, rec, list(range(layers)))


def test_decode_study_is_deterministic_and_the_oracle_is_never_worse():
    data = _small_data()
    kw = dict(ranks=4, widths=[3, 24], mtp=2, rows_budget=3000, seed=4)
    a = simulate.decode_study(data, **kw)
    b = simulate.decode_study(data, **kw)
    assert a == b
    for width in ("3", "24"):
        row = a["widths"][width]
        assert row["oracle"]["step_us"] <= row["linear"]["step_us"] * 1.001
        for name in ("linear", "round_robin", "pack_pairs", "pack_touch", "oracle"):
            assert 1.0 <= row[name]["T"] < 3.0
    # more rows per step average the sampling noise out
    assert a["widths"]["24"]["linear"]["T"] < a["widths"]["3"]["linear"]["T"]


def test_redundancy_inflates_activations_only_with_hash_routing():
    data = _small_data(2)
    res = simulate.redundancy_study(
        data, ranks=4, width=24, extras=(4,), rows_budget=2400
    )
    by = {(r["kind"], r["policy"]): r for r in res["rows"]}
    live_primary = by[("replicate", "primary")]["live_experts_per_step"]
    assert by[("replicate", "hash")]["live_experts_per_step"] > live_primary
    assert by[("replicate", "min_touch")]["live_experts_per_step"] == pytest.approx(
        live_primary
    )
    assert by[("replicate", "min_touch")]["T"] < res["perm_T"]
    assert by[("replicate", "primary")]["extra_slots_per_rank"] == 4.0
    assert by[("replicate", "min_touch")]["vram_mib_per_rank"] == pytest.approx(
        4 * 122_572_800 / (1 << 20)
    )


def test_prefill_calibration_hits_the_requested_chunk_imbalance():
    data = _small_data(2)
    alpha = simulate.calibrate_alpha(data, 1.39, ranks=4, seed=1, chunks=120)
    got = simulate._linear_chunk_ratio(data, alpha, ranks=4, seed=1, chunks=120)
    assert got == pytest.approx(1.39, abs=0.03)


def test_replay_counts_scores_placements_on_measured_steps():
    rng = np.random.default_rng(0)
    p = np.where(np.arange(64) < 8, 0.9, 0.05)  # the hot experts all sit on rank 0
    counts = (rng.random((400, 64)) < p).astype(np.int32)
    res = simulate.replay_counts(counts, None, ranks=4, objective="touch")
    assert res["linear"]["T"] > 1.5
    assert res["pack"]["T"] < res["linear"]["T"]
    assert res["oracle"]["T"] <= res["pack"]["T"] + 1e-9


@pytest.mark.skipif(
    not (T10 / "W1.stats.json").exists(), reason="T10 stats not checked out"
)
def test_real_t10_marginals_linear_is_unbalanced_and_packing_generalises():
    from tools.rdna.hetero_moe.stats_io import load_stats

    w1 = load_stats([str(T10 / "W1.stats.json")])
    w2 = load_stats([str(T10 / "W2.stats.json")])
    data = simulate.layer_data(w1, w2, simulate.layer_ids("0:48:4", sorted(w1.layers)))
    res = simulate.prefill_study(data, ranks=4)["persistent"]
    assert 1.08 < res["linear"]["mean"] < 1.14  # measured: 1.112 over all 48 layers
    assert res["pack"]["mean"] < 1.02  # fitted on W1, scored on W2
    assert res["oracle"]["mean"] < 1.0001


# ---- policy ---------------------------------------------------------------------


def test_moved_bytes_and_move_cost_of_one_swap():
    old = placement.linear_owner(8, 2)
    new = old.copy()
    new[0], new[7] = 1, 0
    n, per_rank = policy.moved_bytes(old, new, 2)
    assert (n, per_rank) == (2, FLASHNEXT.param_expert_bytes)
    cfg = policy.PolicyConfig()
    expect = 0.25 * FLASHNEXT.param_expert_bytes / (14.0 * 1e3) + 300.0
    assert policy.move_us(per_rank, cfg) == pytest.approx(expect)
    assert policy.move_us(0, cfg) == 0.0


def _skewed_weight():
    w = np.full(64, 0.05)
    w[:8] = 0.9  # eight hot experts, all on linear rank 0
    return w


def test_decide_moves_only_when_the_gain_pays_for_the_move():
    w, lin = _skewed_weight(), placement.linear_owner(64, 4)
    cfg = policy.PolicyConfig(window=100, interval=1000, holdout=False)
    go = policy.decide(w, lin, 4, cfg)
    assert go.owner is not None and go.reason == "rebalance"
    assert go.gain_us_per_step > 0 and go.experts_moved > 0
    short = policy.decide(w, lin, 4, cfg, steps_in_horizon=1)
    assert short.owner is None and short.reason == "gain does not pay for the move"
    packed = placement.pack_owner(w, 4)
    assert policy.decide(w, packed, 4, cfg).reason == "same placement"
    tiny = policy.decide(np.full(64, 0.3), lin, 4, cfg)
    assert tiny.owner is None  # balanced weights: nothing to gain


def test_decide_holdout_rejects_a_gain_that_the_newer_half_does_not_show():
    w, lin = _skewed_weight(), placement.linear_owner(64, 4)
    cfg = policy.PolicyConfig(window=100, interval=1000, holdout=True)
    flat = np.full(64, 0.3)  # the newest half shows no skew at all
    d = policy.decide(w, lin, 4, cfg, older=w, newer=flat)
    assert d.owner is None and d.reason == "holdout gain below min_rel_gain"
    assert policy.decide(w, lin, 4, cfg, older=w, newer=w).owner is not None


def _drifting_counts(steps: int, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(2):
        first = np.where(np.arange(64) < 8, 0.9, 0.05)
        second = np.where(np.arange(64) >= 56, 0.9, 0.05)  # the hot set moves rank
        half = steps // 2
        p = np.vstack([np.tile(first, (half, 1)), np.tile(second, (steps - half, 1))])
        out[layer] = (rng.random((steps, 64)) < p).astype(np.int32)
    return out


def test_replay_schedule_follows_drift_and_stays_put_when_stationary():
    cfg = policy.PolicyConfig(window=200, interval=500, objective="touch")
    drift = policy.replay_schedule(_drifting_counts(4000), 4, cfg)
    assert drift["rebalances"] >= 2
    assert drift["total_us"]["schedule"] < drift["total_us"]["linear"]
    # the schedule re-planned after the hot set moved at step 2000
    assert any(m["step"] > 2000 for m in drift["moves"])

    rng = np.random.default_rng(1)
    p = np.where(np.arange(64) < 8, 0.9, 0.05)
    still = {li: (rng.random((4000, 64)) < p).astype(np.int32) for li in range(2)}
    res = policy.replay_schedule(still, 4, cfg)
    assert res["rebalances"] <= 2  # the first decision per layer only
    assert all(m["step"] == 500 for m in res["moves"])
    assert res["schedule_vs_linear_pct"] > 0


def test_replay_schedule_does_not_move_when_the_copy_is_too_expensive():
    slow = policy.PolicyConfig(window=200, interval=500, bw_gbps=1e-4, stall=1.0)
    res = policy.replay_schedule(_drifting_counts(2000), 4, slow)
    assert res["rebalances"] == 0
    assert res["total_us"]["schedule"] == pytest.approx(res["total_us"]["linear"])


def test_repair_owner_swaps_replay_to_the_result_and_move_fewer_than_a_repack():
    w, cur = _skewed_weight(), placement.linear_owner(64, 4)
    owner, swaps = placement.repair_owner(w, cur, 4)
    assert np.bincount(owner, minlength=4).tolist() == [16] * 4
    redo, peak = cur.copy(), [placement.rank_weight(w, cur, 4).max()]
    for a, b in swaps:
        redo[a], redo[b] = redo[b], redo[a]
        peak.append(placement.rank_weight(w, redo, 4).max())
    assert np.array_equal(redo, owner)
    assert all(x >= y for x, y in zip(peak, peak[1:]))  # no swap raises the maximum
    assert placement.repair_owner(w, cur, 4)[1] == swaps  # deterministic
    assert placement.repair_owner(w, cur, 4, max_swaps=2)[1] == swaps[:2]
    assert (owner != cur).sum() < (placement.pack_owner(w, 4) != cur).sum()


def test_shifted_gain_keeps_the_noise_floor():
    persistent = np.array([[3.0, 1.0]] * 4)
    assert policy.shifted_gain(persistent, np.array([-1.0, 1.0])) == pytest.approx(1.0)
    noise = np.array([[3.0, 1.0], [1.0, 3.0]])  # the slow rank alternates
    assert policy.shifted_gain(noise, np.array([-1.0, 1.0])) == pytest.approx(0.0)
    assert policy.shifted_gain(noise, np.zeros(2)) == 0.0


def _near_balanced_weight():
    w = np.full(64, 10.0 / 16.0)  # rank sums 10.5 / 10 / 10 / 9.5 on the linear split
    w[0] += 0.5
    w[48] -= 0.5
    return w


def test_decide_swaps_acts_on_a_skew_above_the_noise_and_not_below_it():
    w, cur = _near_balanced_weight(), placement.linear_owner(64, 4)
    sums = placement.rank_weight(w, cur, 4)
    cfg = policy.PolicyConfig(interval=1000, holdout=False, bw_gbps=1e6)
    quiet = np.random.default_rng(1).normal(sums, 0.5, size=(2000, 4))
    d = policy.decide_swaps(w, cur, 4, cfg, quiet)
    assert d.reason == "rebalance" and d.experts_moved == 2  # one swap
    noisy = np.random.default_rng(1).normal(sums, 4.0, size=(2000, 4))
    d = policy.decide_swaps(w, cur, 4, cfg, noisy)
    assert d.owner is None and d.reason == "gain below min_rel_gain"
    # the expectation-based predictor rebalances the same layer and reshuffles 48
    old = policy.decide(w, cur, 4, cfg)
    assert old.reason == "rebalance" and old.experts_moved == 48
    assert old.gain_us_per_step > 5 * d.gain_us_per_step
    with pytest.raises(ValueError, match="samples"):
        policy.decide_swaps(w, cur, 4, cfg, noisy[:, :3])


def test_decide_swaps_holdout_judges_each_half_with_the_other_halfs_weights():
    w, cur = _near_balanced_weight(), placement.linear_owner(64, 4)
    sums = placement.rank_weight(w, cur, 4)
    cfg = policy.PolicyConfig(interval=1000, holdout=True, bw_gbps=1e6)
    s = np.random.default_rng(1).normal(sums, 0.5, size=(2000, 4))
    flat = np.full(64, 10.0 / 16.0)  # the newer half shows no skew
    d = policy.decide_swaps(w, cur, 4, cfg, s, older=w, newer=flat)
    assert d.owner is None and d.reason == "holdout gain below min_rel_gain"
    assert policy.decide_swaps(w, cur, 4, cfg, s, older=w, newer=w).owner is not None
