# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hetero-MoE plan loader, the dispatch tables and the rule.

numpy only (the reference ops of the runtime tests need torch). Covers H1
(one island, replicate), H2 replicate (two islands, one engine each) and H2
partition (EP2 across the islands).
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.quantization.hetero_moe_support import hetero, solved_doc, stripe_doc
from tools.rdna.hetero_moe import simulate

plan_mod = hetero("plan")
tables_mod = hetero("tables")
PlanError = plan_mod.PlanError
STATS_PLANS = (
    Path(__file__).resolve().parents[2] / "bench_results/2026-11-07_expert-stats-t10"
)
STANDIN_PLAN = (
    Path(__file__).resolve().parents[2]
    / "bench_results/2026-11-10_hetero-hub-standin/H1v.W1.plan.json"
)


def _plan(**kw):
    return plan_mod.plan_from_doc(stripe_doc(**kw))


# ---- loading, validation, ids -----------------------------------------------------


def test_load_validates_and_ids_match_the_solver(tmp_path):
    doc = stripe_doc()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc))
    plan = plan_mod.load_plan(path)
    assert plan.plan_id == doc["plan_id"] == plan.stored_plan_id
    assert plan.num_experts == 64 and plan.top_k == 8
    assert plan.num_islands == 1 and plan.island_mode == "replicate"
    assert plan.target_layers == list(range(6))
    assert "1 island(s), 1 hub(s), 4 expert device(s)" in plan.describe()


def test_edited_plan_is_rejected_by_the_content_hash():
    doc = stripe_doc()
    doc["layers"][0]["owners"][0] = [0, 2]  # was [0, 1]
    with pytest.raises(PlanError, match="content hash"):
        plan_mod.plan_from_doc(doc, validate=False)
    # without the stored id there is nothing to compare
    doc.pop("plan_id")
    assert plan_mod.plan_from_doc(doc, validate=False).plan_id


def test_validator_violations_are_reported(tmp_path):
    doc = stripe_doc()
    doc["layers"][2]["owners"][3] = [1, 1]  # duplicate owner
    with pytest.raises(PlanError, match="layer 2 expert 3: duplicate owner"):
        plan_mod.plan_from_doc(doc)
    doc = stripe_doc()
    doc["devices"][0]["expert_budget_bytes"] = 1  # hub budget blown
    with pytest.raises(PlanError, match="exceeds its budget"):
        plan_mod.plan_from_doc(doc)
    with pytest.raises(PlanError, match="cannot read plan"):
        plan_mod.load_plan(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PlanError, match="cannot read plan"):
        plan_mod.load_plan(bad)


def test_runtime_limits():
    doc = stripe_doc()
    doc["layers"][0]["layer"] = plan_mod.MAX_LAYERS
    with pytest.raises(PlanError, match="below 64"):
        plan_mod.plan_from_doc(doc, validate=False, check_id=False)


def test_with_device_arch_retargets_and_rehashes():
    doc = stripe_doc()
    out = plan_mod.with_device_arch(doc, {"hub0": "gfx1030"})
    assert out["devices"][0]["arch"] == "gfx1030"
    assert doc["devices"][0]["arch"] == "gfx1100"  # the input is untouched
    assert out["plan_id"] != doc["plan_id"]
    plan = plan_mod.plan_from_doc(out)  # valid, hash consistent
    assert plan.arch(0) == "gfx1030"
    with pytest.raises(PlanError, match="unknown device"):
        plan_mod.with_device_arch(doc, {"nope": "gfx1030"})


def test_plan_layer_id_naming():
    f = plan_mod.plan_layer_id
    assert f("model.layers.12.mlp.experts", 48) == 12
    assert f("layers.0.mlp.experts", 48) == 0
    assert f("mtp.layers.0.mlp.experts", 48) == 48
    assert f("model.mtp.layers.1.mlp.experts", 48) == 49
    assert f("model.embed_tokens", 48) is None


# ---- H1: one island ----------------------------------------------------------------


def test_h1_views():
    plan = _plan()
    hub = plan.device_index("hub0")
    assert plan.hubs(0) == [hub] and len(plan.experts(0)) == 4
    for layer in plan.layer_ids:
        resident = plan.hub_resident(hub, layer)
        assert resident.dtype == np.int32
        assert list(resident) == sorted(resident)  # ascending without hub_order
        assert list(resident) == list(range(0, 64, 4))  # stripe 4
        held = np.zeros(64, int)
        for dev in plan.experts(0):
            held[plan.server_experts(dev, layer)] += 1
        assert (held >= 1).all()  # cold_complete: every expert has a server copy
        copies = plan.cold_copies(layer, 0)
        assert all(1 <= len(c) <= 4 for c in copies)
        for e, row in enumerate(copies):
            for s in row:
                dev = plan.experts(0)[s]
                assert e in plan.server_experts(dev, layer)
    assert [plan.server_index(d) for d in plan.experts(0)] == [0, 1, 2, 3]


def test_hub_order_overrides_ascending_slots():
    doc = stripe_doc()
    resident = [e for e, r in enumerate(doc["layers"][1]["owners"]) if 0 in r]
    hot_first = list(reversed(resident))
    doc["layers"][1]["hub_order"] = hot_first
    doc["plan_id"] = plan_mod.compute_plan_id(doc)
    plan = plan_mod.plan_from_doc(doc)
    assert list(plan.hub_resident(0, 1)) == hot_first
    assert list(plan.hub_resident(0, 0)) == sorted(resident)
    tabs = tables_mod.build_island_tables(plan, 0)
    lt = tabs.layers[1]
    assert list(lt.resident) == hot_first
    assert lt.slot_map[hot_first[0]] == 0 and lt.slot_map[hot_first[-1]] == 15


def test_runtime_spill_k_in_the_plan():
    doc = stripe_doc()
    doc["runtime"] = {"spill_k": {"3": 5, "12": 9}}
    plan = plan_mod.plan_from_doc(doc, validate=False, check_id=False)
    assert plan.spill_k() == {3: 5, 12: 9}
    tabs = tables_mod.build_island_tables(plan, 0)
    assert tabs.spill_k == {3: 5, 12: 9}
    assert (
        tabs.k_for(1) == 5 and tabs.k_for(4) == 9 and tabs.k_for(13) == tabs.prefill_k
    )
    override = tables_mod.build_island_tables(plan, 0, spill_k={3: 2})
    assert override.spill_k == {3: 2}  # the argument wins over the plan


# ---- tables ------------------------------------------------------------------------


def test_island_tables_layout():
    plan = _plan(mtp=True)
    tabs = tables_mod.build_island_tables(plan, 0)
    assert tabs.num_servers == 4 and tabs.server_ids == [f"exp{i}" for i in range(4)]
    assert sorted(tabs.layers) == list(range(7))
    for li, lt in tabs.layers.items():
        for arr in (lt.resident, lt.slot_map, lt.n_cold, lt.cold_srv):
            assert arr.dtype == np.int32
        assert lt.slot_map.shape == (64,) and lt.cold_srv.shape == (64, 8)
        assert lt.owned.dtype == np.uint8 and lt.owned.shape == (64,)
        assert (lt.owned == 1).all()  # H1: the island holds every expert
        # slot_map is the inverse of resident
        assert (lt.slot_map[lt.resident] == np.arange(lt.n_slots)).all()
        assert (lt.slot_map >= 0).sum() == lt.n_slots
        for e in range(64):
            n = lt.n_cold[e]
            assert (lt.cold_srv[e, n:] == -1).all() and (lt.cold_srv[e, :n] >= 0).all()
    mtp = tabs.layers[6]
    assert mtp.hub_only and not mtp.uses_hx and mtp.n_slots == 64
    assert (mtp.n_cold == 0).all()
    assert all(
        tabs.layers[li].uses_hx and tabs.layers[li].n_slots == 16 for li in range(6)
    )
    st = tabs.stacked()  # the hub-only MTP layer never reaches the servers
    assert st.slot_map.shape == (6, 64) and st.cold_srv.shape == (6, 64, 8)
    assert list(st.present) == [1] * 6 and list(st.hub_only) == [0] * 6
    assert list(st.n_slots) == [16] * 6
    assert (st.slot_map[3] == tabs.layers[3].slot_map).all()
    assert st.owned.shape == (6, 64) and (st.owned == 1).all()


def test_spill_table_helpers():
    assert tables_mod.parse_spill_k("3:7, 6:14,12:26") == {3: 7, 6: 14, 12: 26}
    for bad in ("", "3", "3:x", "0:4", "a:1"):
        with pytest.raises(PlanError):
            tables_mod.parse_spill_k(bad)
    assert tables_mod.default_spill_k(4) == {3: 10, 6: 21, 12: 39, 24: 65, 48: 110}
    assert tables_mod.default_spill_k(4, partition=True) == {
        3: 7, 6: 13, 12: 23, 24: 39, 48: 63,
    }  # fmt: skip
    assert tables_mod.default_spill_k(8)[24] == 130
    assert min(tables_mod.default_spill_k(1).values()) >= 1
    assert tables_mod.build_island_tables(_plan(), 0).spill_k[24] == 65
    part = _plan(islands=2, island_mode="partition")
    assert tables_mod.build_island_tables(part, part.hubs(1)[0]).spill_k[24] == 39
    tabs = tables_mod.build_island_tables(_plan(), 0, spill_k={3: 1, 12: 4, 6: 2})
    assert tabs.widths == [3, 6, 12]
    assert [tabs.bucket_for(r) for r in (1, 3, 4, 6, 7, 12, 13)] == [
        3,
        3,
        6,
        6,
        12,
        12,
        0,
    ]
    assert [tabs.k_for(r) for r in (1, 4, 13, 2048)] == [
        1,
        2,
        tabs.prefill_k,
        tabs.prefill_k,
    ]
    assert tabs.prefill_k == 16  # no spill above the largest width by default


def test_spill_is_disabled_without_server_copies():
    doc = stripe_doc()
    # drop the server copy of every hub-resident expert of layer 2
    for e, row in enumerate(doc["layers"][2]["owners"]):
        if 0 in row:
            doc["layers"][2]["owners"][e] = [0]
    doc["cold_complete"] = False
    doc["plan_id"] = plan_mod.compute_plan_id(doc)
    plan = plan_mod.plan_from_doc(doc, validate=False)
    tabs = tables_mod.build_island_tables(plan, 0)
    assert tabs.spill_disabled and "layers [2]" in tabs.spill_disabled
    assert set(tabs.spill_k.values()) == {16} and tabs.prefill_k == 16


def test_one_hub_per_island_is_required():
    doc = stripe_doc()
    doc["devices"][1]["role"] = "hub"  # a second hub in the island
    plan = plan_mod.plan_from_doc(doc, validate=False, check_id=False)
    with pytest.raises(PlanError, match="one hub per island"):
        tables_mod.build_island_tables(plan, 0)
    with pytest.raises(PlanError, match="not a hub"):
        tables_mod.build_island_tables(_plan(), 2)


# ---- H2: two islands ---------------------------------------------------------------


def test_h2_replicate_maps_the_island_by_position():
    plan = _plan(islands=2)
    assert plan.num_islands == 2
    hub0, hub1 = plan.hubs(0)[0], plan.hubs(1)[0]
    assert hub0 != hub1 and plan.island_of(hub1) == 1
    for layer in plan.layer_ids:
        assert list(plan.hub_resident(hub0, layer)) == list(
            plan.hub_resident(hub1, layer)
        )
        assert plan.cold_copies(layer, 0) == plan.cold_copies(layer, 1)
        for e, row in enumerate(plan.island_owners(layer, 1)):
            assert all(plan.island_of(d) == 1 for d in row)
    t0 = tables_mod.build_island_tables(plan, hub0)
    t1 = tables_mod.build_island_tables(plan, hub1)
    assert t0.plan_id == t1.plan_id and t1.island == 1
    assert (t0.layers[0].slot_map == t1.layers[0].slot_map).all()
    assert (t0.layers[0].cold_srv == t1.layers[0].cold_srv).all()


def test_h2_partition_drops_the_other_islands_experts():
    plan = _plan(islands=2, island_mode="partition", experts=64)
    assert plan.island_mode == "partition"
    hubs = [plan.hubs(k)[0] for k in range(2)]
    t = [tables_mod.build_island_tables(plan, h) for h in hubs]
    for layer in plan.layer_ids:
        owned: list[set[int]] = [set(), set()]
        for k in range(2):
            lt = t[k].layers[layer]
            for e in range(64):
                here = lt.slot_map[e] >= 0 or lt.n_cold[e] > 0
                assert bool(lt.owned[e]) == here
                if here:
                    owned[k].add(e)
                else:  # not ours: no slot, no copy -> the pair is dropped
                    assert lt.slot_map[e] == -1 and lt.n_cold[e] == 0
        assert owned[0] | owned[1] == set(range(64)) and not (owned[0] & owned[1])
        assert abs(len(owned[0]) - len(owned[1])) <= 1
        resident = [set(plan.hub_resident(h, layer)) for h in hubs]
        assert not (resident[0] & resident[1])
        for k in range(2):
            assert resident[k] <= owned[k]
            assert (t[k].layers[layer].n_cold[sorted(resident[k])] > 0).all()
    assert t[0].spill_disabled is None and t[1].spill_disabled is None


# ---- solver output -----------------------------------------------------------------


def test_plan_solved_from_synthetic_stats_loads(tmp_path):
    doc = solved_doc(tmp_path)
    plan = plan_mod.plan_from_doc(doc)
    tabs = tables_mod.build_island_tables(plan, plan.hubs(0)[0])
    assert plan.target_layers == list(range(6))
    for lt in tabs.layers.values():
        assert lt.n_slots > 0 and (lt.n_cold[lt.resident] > 0).all()
    assert tabs.spill_disabled is None


def test_stripe_plan_of_the_h1_command_line(tmp_path):
    """The recipe's solver command: 48 layers x 512 experts, 128 resident per layer."""
    plan = plan_mod.solve_plan(
        tmp_path / "h1.json",
        [
            "--islands", "1", "--hubs-per-island", "1", "--experts-per-island", "4",
            "--hub-hips", "0", "--expert-hips", "1,2,3,4", "--switches", "c0",
            "--island-mode", "replicate", "--hub-active-gib", "9.4",
            "--max-hub-per-layer", "130",
        ],
    )  # fmt: skip
    assert plan.num_layers == 48 and plan.num_experts == 512 and plan.top_k == 10
    hub = plan.hubs(0)[0]
    assert plan.arch(hub) == "gfx1100" and plan.arch(plan.experts(0)[0]) == "gfx1030"
    tabs = tables_mod.build_island_tables(plan, hub)
    assert all(lt.n_slots == 128 for lt in tabs.layers.values())
    assert tabs.stacked().cold_srv.shape == (48, 512, 8)
    # ~200 KB of int32 tables per hub (spec 6.2) before the int8 repack
    assert tabs.stacked().cold_srv.nbytes + tabs.stacked().slot_map.nbytes < 2_000_000


# ---- the rule: loop oracle ---------------------------------------------------------


def _layer(n_slots=4, e=16, servers=3):
    """Hand-built layer: slots hold experts 0,2,4,6 (in this order)."""
    resident = np.array([0, 2, 4, 6][:n_slots], np.int32)
    slot_map = np.full(e, -1, np.int32)
    slot_map[resident] = np.arange(n_slots, dtype=np.int32)
    n_cold = np.zeros(e, np.int32)
    cold = np.full((e, 8), -1, np.int32)
    for x in range(e):
        n_cold[x] = 1 + x % 2  # even experts: one copy, odd: two
        for j in range(n_cold[x]):
            cold[x, j] = (x + j) % servers
    owned = np.ones(e, np.uint8)
    return tables_mod.LayerTables(
        0, "target", False, resident, slot_map, n_cold, cold, owned
    )


def test_loop_oracle_rule_by_hand():
    lt = _layer()
    ids = np.array([[0, 2, 9, 4], [6, 2, 11, -1], [3, 99, 0, 6]], np.int32)
    # touched resident slots: experts 0,2,4,6 -> slots 0,1,2,3; K=2 keeps slots 0,1
    hub, dest = tables_mod.split_pairs_loop(ids, lt, layer=5, k_keep=2)
    assert hub.tolist() == [[0, 2, -1, -1], [-1, 2, -1, -1], [-1, -1, 0, -1]]
    # pair (row 0, expert 9): copies of 9 = (9+0)%3, (9+1)%3 = 0, 1; (0+5)%2 = 1 -> 1
    assert dest[0, 2] == lt.cold_srv[9, (0 + 5) % 2] == 1
    # spilled resident expert 4 (even, one copy) goes to its only copy
    assert dest[0, 3] == lt.cold_srv[4, 0] == 1
    assert dest[1, 0] == lt.cold_srv[6, 0] == 0
    assert dest[1, 3] == -1 and dest[2, 1] == -1  # -1 and 99 are dropped
    assert dest[2, 3] == lt.cold_srv[6, 0]
    # kept pairs have no destination
    assert (dest[hub >= 0] == -1).all()
    # K covers every touched slot: nothing spills, only non-resident experts go remote
    hub_all, dest_all = tables_mod.split_pairs_loop(ids, lt, layer=5, k_keep=4)
    assert hub_all.tolist() == [[0, 2, -1, 4], [6, 2, -1, -1], [-1, -1, 0, 6]]
    assert dest_all[0, 2] == 1 and dest_all[1, 2] == lt.cold_srv[11, (1 + 5) % 2]


def test_loop_oracle_drops_other_islands_and_flags_plan_errors():
    lt = _layer()
    lt.owned[[1, 3]] = 0  # experts 1 and 3 belong to the other island
    ids = np.array([[1, 3, 0, 5]], np.int32)
    hub, dest = tables_mod.split_pairs_loop(ids, lt, 0, k_keep=4)
    assert hub.tolist() == [[-1, -1, 0, -1]]
    assert dest[0, 0] == dest[0, 1] == -1 and dest[0, 3] == lt.cold_srv[5, 0]
    lt.n_cold[5] = 0  # an owned expert with no resident slot and no copy
    with pytest.raises(PlanError, match="expert 5 is owned"):
        tables_mod.split_pairs_loop(ids, lt, 0, k_keep=4)
    lt.n_cold[0] = 0  # a resident expert without a copy that is spilled
    with pytest.raises(PlanError, match="expert 0 is owned"):
        tables_mod.split_pairs_loop(np.array([[0]], np.int32), lt, 0, k_keep=0)


def test_rule_counts_touched_experts_not_pairs():
    lt = _layer()
    # expert 6 (slot 3) appears in every row: still ONE touched expert
    ids = np.array([[6, 0], [6, 0], [6, 0], [6, 0]], np.int32)
    hub, _ = tables_mod.split_pairs_loop(ids, lt, 0, k_keep=2)
    assert (hub == ids).all()
    hub, dest = tables_mod.split_pairs_loop(ids, lt, 0, k_keep=1)
    assert (hub[:, 1] == 0).all() and (hub[:, 0] == -1).all()
    assert (dest[:, 0] >= 0).all()  # all rows of the spilled expert leave together


def test_slot_order_decides_who_is_spilled():
    lt = _layer()
    ids = np.array([[6, 4, 2]], np.int32)  # touched slots 3, 2, 1
    hub, _ = tables_mod.split_pairs_loop(ids, lt, 0, k_keep=2)
    assert hub.tolist() == [[-1, 4, 2]]  # lowest two SLOTS stay, not the first pairs


def test_simulator_and_runtime_share_the_dispatch_rule(tmp_path):
    """evaluate_ids (cost model) and the dispatch tables classify every pair alike."""
    plan = _plan(experts=64, layers=3, top_k=8)
    tabs = tables_mod.build_island_tables(plan, 0)
    pa = simulate.parse_plan(plan.doc)
    rng = np.random.default_rng(3)
    steps, rows_per_step = 40, 5
    n_rows = steps * rows_per_step
    step_of_row = np.repeat(np.arange(steps), rows_per_step)
    row_in_step = np.tile(np.arange(rows_per_step), steps)
    ids = {
        li: np.stack([rng.choice(64, 8, replace=False) for _ in range(n_rows)]).astype(
            np.int32
        )
        for li in range(3)
    }
    sim = simulate.evaluate_ids(pa, ids, step_of_row, row_in_step, steps, spill=False)
    hub_hit = hub_touched = cold_mean = cold_max = 0.0
    for li, sel in ids.items():
        lt = tabs.layers[li]
        per_step = []
        for s in range(steps):
            rows = sel[s * rows_per_step : (s + 1) * rows_per_step]
            hub_ids, dest = tables_mod.split_pairs_loop(rows, lt, li, k_keep=10**6)
            hub_hit += (hub_ids >= 0).sum()
            hub_touched += len(np.unique(hub_ids[hub_ids >= 0])) / steps
            touched = [
                len(np.unique(rows[dest == srv])) for srv in range(tabs.num_servers)
            ]
            per_step.append(touched)
        per_step = np.array(per_step, float)
        cold_mean += per_step.mean(axis=1).mean()
        cold_max += per_step.max(axis=1).mean()
    assert sim["hub_hit"] == pytest.approx(hub_hit / (3 * n_rows * 8))
    assert sim["hub_touched_per_step"] == pytest.approx(hub_touched)
    assert sim["cold_mean_touched_per_step"] == pytest.approx(cold_mean)
    assert sim["cold_max_touched_per_step"] == pytest.approx(cold_max)


def test_plan_json_roundtrip_is_stable(tmp_path):
    doc = stripe_doc(mtp=True)
    again = json.loads(json.dumps(doc))
    assert plan_mod.plan_from_doc(again).plan_id == doc["plan_id"]
    assert plan_mod.plan_from_doc(copy.deepcopy(doc)).layer_kind(6) == "mtp"


# ---- the plan command line (launcher helper) --------------------------------------

CLI = [sys.executable, str(plan_mod.repo_root() / "vllm/distributed/hetero_moe/cli.py")]


def _cli(*args, ok=True):
    out = subprocess.run(
        [*CLI, *args], capture_output=True, text=True, cwd=plan_mod.repo_root()
    )
    assert (out.returncode == 0) == ok, out.stderr
    return out


def test_cli_lists_devices_and_buckets(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(stripe_doc(layers=3)))
    assert (
        "1 island(s), 1 hub(s), 4 expert device(s)"
        in _cli("validate", str(path)).stdout
    )
    hubs = [ln.split("\t") for ln in _cli("hubs", str(path)).stdout.splitlines()]
    assert hubs == [["0", "hub0", "0", "s0", "gfx1100"]]
    servers = [ln.split("\t") for ln in _cli("servers", str(path)).stdout.splitlines()]
    assert [s[1] for s in servers] == ["exp0", "exp1", "exp2", "exp3"]
    assert [s[2] for s in servers] == ["1", "2", "3", "4"]
    assert _cli("buckets", str(path)).stdout.strip() == "3,6,12,24,48"
    out = _cli("buckets", str(path), "--spill-k", "3:2,12:5")
    assert out.stdout.strip() == "3,12"


def test_cli_rejects_a_broken_plan(tmp_path):
    doc = stripe_doc(layers=3)
    doc["layers"][1]["owners"][2] = [1, 1]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc))
    out = _cli("validate", str(path), ok=False)
    assert "duplicate owner" in out.stderr and out.stdout == ""
    out = _cli("validate", str(tmp_path / "missing.json"), ok=False)
    assert "cannot read plan" in out.stderr


def test_cli_retarget_rehashes_the_plan(tmp_path):
    src = tmp_path / "plan.json"
    src.write_text(json.dumps(stripe_doc(layers=3)))
    dst = tmp_path / "v620.json"
    _cli("retarget", str(src), "--out", str(dst), "--arch", "hub0=gfx1030")
    hub = _cli("hubs", str(dst)).stdout.split("\t")
    assert hub[4].strip() == "gfx1030"
    new = json.loads(dst.read_text())
    assert new["plan_id"] != json.loads(src.read_text())["plan_id"]
    _cli("retarget", str(src), "--out", str(dst), "--arch", "nope=gfx1030", ok=False)


def test_cli_stripe_writes_the_h1_plan_and_the_stand_in(tmp_path):
    h1 = tmp_path / "h1.json"
    out = _cli("stripe", "--out", str(h1))
    plan = plan_mod.load_plan(h1)
    assert "plan " in out.stdout and plan.num_islands == 1
    assert (plan.num_experts, len(plan.target_layers)) == (512, 48)
    assert [plan.doc["devices"][d]["hip"] for d in plan.hubs(0)] == [0]
    assert [plan.doc["devices"][d]["hip"] for d in plan.experts(0)] == [1, 2, 3, 4]
    assert int((plan.hub_resident(plan.hubs(0)[0], 0) % 4 != 0).sum()) == 0
    stand = tmp_path / "standin.json"
    _cli(
        "stripe", "--out", str(stand), "--hub-hips", "1", "--expert-hips", "2,3,4",
        "--stripe", "16", "--hub-arch", "gfx1030",
    )  # fmt: skip
    plan = plan_mod.load_plan(stand)
    assert len(plan.experts(0)) == 3
    assert plan.arch(plan.hubs(0)[0]) == "gfx1030"
    assert plan.hub_resident(plan.hubs(0)[0], 0).shape[0] == 32
    assert plan.server_experts(plan.experts(0)[0], 0).shape[0] > 160


@pytest.mark.skipif(
    not STATS_PLANS.is_dir(), reason="the ExpertStats T10 plans are absent"
)
@pytest.mark.parametrize("name", ["H1.W1", "H1.W2", "H2.W1", "H2.W2"])
def test_plans_solved_on_real_routing_statistics_load(name):
    plan = plan_mod.load_plan(STATS_PLANS / f"{name}.plan.json")
    h2 = name.startswith("H2")
    assert plan.num_islands == (2 if h2 else 1)
    assert plan.stored_plan_id == plan.plan_id
    mtp = plan.layer_ids[-1]
    assert plan.layer_kind(mtp) == "mtp" and plan.is_hub_only(mtp)
    for island in range(plan.num_islands):
        it = tables_mod.build_island_tables(plan, plan.hubs(island)[0])
        assert it.spill_disabled is None
        target = [lt for lt in it.layers.values() if not lt.hub_only]
        assert len(target) == 48
        assert {lt.n_slots for lt in target} == {115 if h2 else 130}
        assert it.layers[mtp].n_slots == (256 if h2 else 512)
        if not h2:
            # cold_complete: a missed or spilled pair always finds a server copy
            assert all(int(lt.n_cold.min()) >= 1 for lt in target)
            assert all(bool(lt.owned.all()) for lt in target)


@pytest.mark.skipif(not STANDIN_PLAN.is_file(), reason="the stand-in plan is absent")
def test_standin_plan_is_a_v620_hub_with_four_servers():
    # bench_results/2026-11-10_hetero-hub-standin/make_plan.sh: the T10 W1 statistics
    # solved for the V620 hub (HIP 1) and the four V620 servers (HIP 2-5)
    plan = plan_mod.load_plan(STANDIN_PLAN)
    assert plan.num_islands == 1 and plan.stored_plan_id == plan.plan_id
    (hub,) = plan.hubs(0)
    servers = plan.experts(0)
    assert len(servers) == 4
    assert plan.arch(hub) == "gfx1030"
    assert {plan.arch(s) for s in servers} == {"gfx1030"}
    assert plan.devices[hub]["hip"] == 1
    assert [plan.devices[s]["hip"] for s in servers] == [2, 3, 4, 5]
    mtp = plan.layer_ids[-1]
    assert plan.layer_kind(mtp) == "mtp" and plan.is_hub_only(mtp)
    it = tables_mod.build_island_tables(plan, hub)
    assert it.spill_disabled is None
    target = [lt for lt in it.layers.values() if not lt.hub_only]
    assert len(target) == 48
    # 40 slots per layer: 6 GiB of the ~30 GiB hub with the MTP head's 512 experts
    assert {lt.n_slots for lt in target} == {40}
    assert it.layers[mtp].n_slots == 512
    # cold_complete: a missed or spilled pair always finds a server copy
    assert all(int(lt.n_cold.min()) >= 1 for lt in target)
    assert all(bool(lt.owned.all()) for lt in target)
    # every server fits its budget
    for s in servers:
        pairs = sum(
            plan.server_experts(s, layer).shape[0]
            for layer in plan.layer_ids
            if not plan.is_hub_only(layer)
        )
        assert pairs * plan.expert_bytes <= plan.devices[s]["expert_budget_bytes"]


def test_hub_only_layer_ids_are_not_table_indexes():
    # the stats collector numbers the MTP draft num_target_layers + N (96 here)
    doc = stripe_doc(mtp=True)
    doc["layers"][-1]["layer"] = 96
    doc["plan_id"] = plan_mod.compute_plan_id(doc)
    plan = plan_mod.plan_from_doc(doc)
    assert plan.layer_ids == [0, 1, 2, 3, 4, 5, 96] and plan.is_hub_only(96)
    tabs = tables_mod.build_island_tables(plan, plan.hubs(0)[0])
    st = tabs.stacked()
    assert st.slot_map.shape == (6, 64) and list(st.present) == [1] * 6
    # a server layer id beyond the protocol is still refused
    doc = stripe_doc(layers=6)
    doc["layers"][-1]["layer"] = 64
    doc["plan_id"] = plan_mod.compute_plan_id(doc)
    with pytest.raises(PlanError, match="below 64"):
        plan_mod.plan_from_doc(doc)
