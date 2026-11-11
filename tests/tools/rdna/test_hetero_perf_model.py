# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of the heterogeneous-MoE analytic performance model.

`tools/rdna/hetero_moe/perf_model.py` is calibrated to the measured V620 TP4
numbers. These tests pin that calibration, the held-out TP8 prediction, the
structural conclusions of `docs/rdna2/hetero-moe-design-2026-11-07.md` and the
byte-table identities, so a change to the model cannot silently move the spec.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "tools/rdna/hetero_moe/perf_model.py"


def _load():
    spec = importlib.util.spec_from_file_location("hetero_perf_model", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


pm = _load()


def _run(**knobs):
    a, ts = pm.setup(pm.Assume(**knobs))
    return a, ts, pm.results(a, ts)


@pytest.fixture(scope="module")
def central():
    return _run()


def test_tp4_calibration_is_exact(central):
    tp4 = central[2]["rows"]["TP4"]
    assert tp4["c1"]["step"] == pytest.approx(22.38, abs=0.05)
    assert tp4["c8"]["step"] == pytest.approx(43.97, abs=0.05)
    assert tp4["tps8"] == pytest.approx(262.6, abs=1.0)
    assert tp4["ttft16k"] == pytest.approx(5.52, abs=0.05)


def test_tp8_held_out_prediction(central):
    tp8 = central[2]["rows"]["TP8"]
    assert tp8["c1"]["step"] == pytest.approx(19.59, rel=0.05)
    assert tp8["c8"]["step"] == pytest.approx(37.2, rel=0.05)  # hierarchical AR
    assert tp8["tps8"] == pytest.approx(290.3, rel=0.06)


def test_h1_is_slower_than_tp4_slot_a_without_the_levers(central):
    rows = central[2]["rows"]
    tp4, h1, h2, w2 = rows["TP4"], rows["H1"], rows["H2"], rows["W2"]
    assert h1["c1"]["step"] > 1.3 * tp4["c1"]["step"]
    assert h1["c8"]["step"] > tp4["c8"]["step"]
    assert h2["c1"]["step"] < 1.1 * tp4["c1"]["step"]
    assert h2["c8"]["step"] < 0.9 * tp4["c8"]["step"]
    assert h2["c1"]["step"] < w2["c1"]["step"]
    assert h2["prefill"]["chunk"] < tp4["prefill"]["chunk"]


def test_spill_rule_balances_hub_and_servers(central):
    h2 = central[2]["rows"]["H2"]
    for mode in ("c1", "c8", "c16"):
        comp = h2[mode]["comp"]
        assert comp["n_hub"] <= comp["n_res"] + 1e-6
        assert abs(comp["hub"] - comp["remote"]) <= 0.1 * comp["moe"]


def test_machine_level_engine_split_beats_single_hetero_engine(central):
    o = central[2]
    deps = {d["name"].split(" ")[0]: d for d in pm.deployments(o)}
    two_tp4 = 2 * pm.TODAY["TP4"]["c8_tps"]
    assert deps["D0a"]["tps"] == pytest.approx(two_tp4)
    assert deps["D3"]["tps"] == pytest.approx(o["rows"]["W2"]["tps8"] + two_tp4)
    assert deps["D3"]["tps"] > deps["D2b"]["tps"] > deps["D2"]["tps"]
    assert deps["D3"]["tps"] > deps["D0a"]["tps"]


def test_dense_levers_help():
    base = _run()[2]["rows"]["H2"]["c1"]["step"]
    w8 = _run(hub_dense_scale=0.5)[2]["rows"]["H2"]["c1"]["step"]
    both = _run(hub_dense_scale=0.5, lmhead_remote=True)[2]["rows"]["H2"]["c1"]["step"]
    assert both < w8 < base


def test_pessimistic_port_hurts_the_hub_not_the_anchor():
    rows = _run(eta_h=0.65, kappa=1.4, phi=0.25, detect_us=30.0)[2]["rows"]
    assert rows["TP4"]["c1"]["step"] == pytest.approx(22.38, abs=0.05)
    assert rows["H2"]["c1"]["step"] > 1.2 * rows["TP4"]["c1"]["step"]


def test_byte_table_identities():
    g = pm.GIB
    routed = pm.EXPERT_B * pm.N_EXP * pm.N_LAYERS / pm.GiB
    assert routed == pytest.approx(g["experts"], abs=0.01)
    active = sum(
        g[k]
        for k in (
            "gdn", "qsa", "indexer", "hc", "router", "shared",
            "embed", "lm_head", "ple_proj", "mtp_dense",
        )
    )  # fmt: skip
    assert active == pytest.approx(9.385, abs=0.002)  # hub_active_bytes
    resident = sum(v for k, v in g.items() if k != "mtp_experts_w4")
    assert resident == pytest.approx(72.908, abs=0.005)  # checkpoint minus PLE table


def test_wire_sizes_match_the_spec_table():
    t3 = pm.pcie_traffic(3, 4)
    assert t3["down_kib"] == pytest.approx(60.7, abs=0.1)
    assert t3["up_kib"] == pytest.approx(60.0, abs=0.1)
    assert t3["push_us"] == pytest.approx(8.8, abs=0.1)
    assert pm.pcie_traffic(24, 4)["push_us"] == pytest.approx(26.2, abs=0.2)


def test_memory_fit(central):
    a = central[0]
    assert pm.mem_hub(2, 11.0, a)["kv_gib"] > 20.0  # H2 hub
    assert 4.0 < pm.mem_hub(2, 29.4, a)["kv_gib"] < 7.0  # W2 hub: 29.4 GiB experts
    assert 15.0 < pm.mem_hub(1, 12.0, a)["kv_gib"] < 20.0  # H1 hub
    assert pm.mem_v620(0.125, 1.0)["experts"] == pytest.approx(7.3, abs=0.1)
    assert pm.mem_v620(0.125, 3.5)["free"] > 0.0  # replicas fit the budget


def test_scalar_moe_prefill_range():
    # Spec 5.4: the W7800 has no WMMA MoE kernel yet; the scalar kernel is modelled
    # at eff_hub 0.14 and 0.10.
    for eff, w2, h2 in ((0.14, 1.01, 0.75), (0.10, 1.39, 1.03)):
        a = pm.Assume(eff_hub=eff)
        assert pm.prefill_chunk("W2", a)["chunk"] == pytest.approx(w2, abs=0.02)
        assert pm.prefill_chunk("H2", a)["chunk"] == pytest.approx(h2, abs=0.02)
    slow = pm.Assume(eff_hub=0.14)
    assert pm.prefill_chunk("W2", slow)["chunk"] > pm.TP4_CHUNK_S
    assert pm.prefill_chunk("H2", slow)["chunk"] < 1.10 * pm.TP4_CHUNK_S


def test_bf16_drafter_kv_capacity():
    # Spec R10: the checkpoint's drafter is bf16 until the W4A16 shard exists.
    def kv_ktok(mtp_w4, tp, hot):
        return pm.mem_hub(tp, hot, pm.Assume(mtp_w4=mtp_w4))["kv_tokens"] / 1e3

    assert kv_ktok(True, 2, 29.45) == pytest.approx(388, abs=2)
    assert kv_ktok(False, 2, 29.45) == pytest.approx(264, abs=2)
    assert kv_ktok(False, 2, 11.0) == pytest.approx(1577, abs=10)


def test_tp2_allreduce_sensitivity(monkeypatch):
    # Spec R15: 36 us per call (upper end of the RCCL estimate) instead of 18 us.
    base = _run()[2]["rows"]
    monkeypatch.setitem(pm.AR_US, 2, (36.0, 46.0))
    slow = _run()[2]["rows"]
    delta = slow["H2"]["c1"]["step"] - base["H2"]["c1"]["step"]
    assert delta == pytest.approx(1.8, abs=0.1)
    assert slow["W2"]["c1"]["step"] == pytest.approx(26.5, abs=0.1)


def test_cli_json_round_trip():
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = json.loads(out)["rows"]
    assert rows["TP4"]["c1"]["step"] == pytest.approx(22.38, abs=0.05)
    assert {"W2", "H1", "H2", "H2p"} <= set(rows)


def _s2(prefix, rows):
    (hit,) = [r for r in rows if r["name"].startswith(prefix)]
    return hit


def test_h1_ladder_reaches_tp4_parity_with_both_levers():
    # Stage-1 gates (spec section 10): H1 + W8A16 hub dense + lm_head on the servers
    # is within 10 % of measured TP4 slot A; the first boot is judged against the
    # central model with a 15 % band and the pessimistic scenario as the red line.
    base = pm.Assume()
    by = {r["name"]: r for r in pm.h1_ladder(base)}
    both, central_, pess = by["+ both levers"], by["central"], by["pessimistic"]
    assert both["c1_ratio"] < pm.GATE_PARITY and both["c8_ratio"] < pm.GATE_PARITY
    assert central_["c1_ratio"] > 1.3 and central_["c8_ratio"] > 1.1
    g = pm.h1_gates(base)
    assert central_["c1"] < g["green_c1"] < g["red_c1"] == pess["c1"]
    assert central_["c8"] < g["green_c8"] < g["red_c8"] == pess["c8"]
    assert g["levers_c1"] < g["parity_c1"] and g["levers_c8"] < g["parity_c8"]


def test_h1_fit_on_one_w7800(central):
    a = central[0]
    cap = pm.h1_fit(a, slots=130)  # the solver cap per layer for replicated islands
    assert cap["pool_gib"] == pytest.approx(14.7, abs=0.1)
    assert cap["kv_gib"] == pytest.approx(15.6, abs=0.1)
    assert cap["kv_tokens"] > 600e3  # TP4 slot A has 267k
    bf16 = pm.h1_fit(pm.Assume(mtp_w4=False), slots=130)
    assert 0 < bf16["kv_gib"] < cap["kv_gib"]  # the checkpoint's bf16 drafter fits
    reserve = pm.h1_fit(a, kv_gib=8.0)
    assert reserve["slots"] == pytest.approx(197, abs=1.5)
    total = pm.h1_fit(a, slots=reserve["slots"])
    assert total["kv_gib"] == pytest.approx(8.0, abs=1e-6)
    used = total["dense"] + total["mtp"] + total["fixed"] + total["pool_gib"]
    assert used + total["kv_gib"] == pytest.approx(44.98, abs=0.01)


def test_stage2_throughput_best_is_d3_and_hetero_is_not_better_per_gpu():
    rows = pm.stage2(pm.Assume())
    d3, d3l = _s2("D3 W2", rows), _s2("D3 with W8A16", rows)
    buildable = [r for r in rows if "blocked" not in r["basis"]]
    assert max(buildable, key=lambda r: r["tps"]) is d3l
    for prefix in ("D1 2 x H1", "D1 + both", "P2", "H2, 16", "H2 + both", "D4"):
        r = _s2(prefix, rows)
        assert r["tps"] < d3["tps"] and r["per_gpu"] < d3["per_gpu"]
    assert d3["tps"] == pytest.approx(803, abs=3)
    assert _s2("D0a", rows)["tps"] == pytest.approx(525, abs=3)


def test_pp2_across_islands_is_two_h1_for_decode_and_halves_ttft():
    rows = pm.stage2(pm.Assume())
    p2, d1 = _s2("P2", rows), _s2("D1 2 x H1", rows)
    assert p2["tps"] == pytest.approx(d1["tps"])
    assert p2["c1"] == pytest.approx(d1["c1"], abs=0.05)
    assert p2["ttft"] == pytest.approx(0.5625 * d1["ttft"], rel=1e-6)  # 4.5 / 8 chunks


def test_dense_shard_offload_to_the_servers_is_a_few_percent_at_best():
    gain = {(r["name"], r["mode"]): r for r in pm.dense_offload(pm.Assume())}
    assert 0.0 < gain[("H2", "c1")]["pct"] < 6.0
    assert gain[("H2", "c8")]["us_layer"] == 0.0
    assert gain[("H2 + W8A16 hub dense", "c1")]["us_layer"] == 0.0
    assert gain[("H2 + W8A16 hub dense", "c8")]["us_layer"] == 0.0


def test_pd_split_gain_is_bounded():
    rows = pm.stage2(pm.Assume())
    pd, d3 = _s2("D3-PD", rows), _s2("D3 W2", rows)
    assert "blocked" in pd["basis"]
    assert 1.10 * d3["tps"] < pd["tps"] < 1.30 * d3["tps"]


def test_step_at_users_matches_the_calibration_points():
    a, ts = pm.setup(pm.Assume())
    tp4 = ts["TP4"]
    assert pm.step_ms_at(tp4, 1, a) == pytest.approx(22.38, abs=0.05)
    assert pm.step_ms_at(tp4, 8, a) == pytest.approx(43.97, abs=0.05)
    assert pm.acc_at(1) == pm.ACC_C1 and pm.acc_at(8) == pm.acc_at(16) == pm.ACC
    steps = [pm.step_ms_at(tp4, u, a) for u in (1, 2, 4, 8, 16)]
    assert steps == sorted(steps)


def test_h2_wins_latency_only_at_low_concurrency():
    # Spec 5.7: on equal total users the engine split of D3 beats the single H2 engine
    # from about four users on; H2 + levers is best at one and two users (central).
    cur = pm.itl_curve(pm.Assume())
    h2, d3l = cur["H2 + both levers"], cur["D3 with W8A16 on W2"]
    assert h2[1] < d3l[1] and h2[2] < d3l[2]
    assert h2[4] > d3l[4] and h2[8] > d3l[8] and h2[16] > 1.15 * d3l[16]
    assert h2[24] is None and d3l[24] is not None
    assert h2[1] == pytest.approx(cur["TP8 (today)"][1], rel=0.03)  # TP8 is the floor
    for u in (4, 8, 16, 24):
        assert d3l[u] < cur["D0a 2 x TP4 (today)"][u]


def test_h2_latency_win_depends_on_the_w7800_port():
    best = {}
    for name, kw in pm.ITL_SCENARIOS:
        cur = pm.itl_curve(pm.Assume(**kw))
        best[name.split(" ")[0]] = {
            u: pm.ITL_TAG[min((n for n in cur if cur[n][u] is not None),
                              key=lambda n: cur[n][u])]
            for u in pm.ITL_USERS
        }  # fmt: skip
    assert best["optimistic"][8] == "H2L"  # kappa 0.8, eta 0.95: H2 wins up to 8 users
    assert "H2L" not in best["pessimistic"].values()  # kappa 1.4: never
    assert best["pessimistic"][1] == "TP8"
    assert best["central"][24] == "D3L"


def test_d5_is_the_latency_deployment_and_d3_the_throughput_one():
    # Spec 5.7: TP8 on the V620s plus W2 on the W7800 (no hx protocol) has the lowest
    # ITL from two users up to about four; the engine split of D3 wins from eight
    # users on and has the highest aggregate rate.
    cur = pm.itl_curve(pm.Assume())
    d5, d3l = cur["D5 TP8 + W2 + W8A16"], cur["D3 with W8A16 on W2"]
    assert d5[1] == pytest.approx(cur["TP8 (today)"][1])  # the first user is on TP8
    assert d5[2] < d3l[2] < cur["TP8 (today)"][2]
    assert d5[8] > d3l[8] and d5[24] > d3l[24]
    rows = pm.stage2(pm.Assume())
    assert _s2("D5 with W8A16", rows)["tps"] < 0.8 * _s2("D3 with W8A16", rows)["tps"]


def test_h2_never_clears_the_go_ratio_against_the_best_plain_deployment():
    # Spec 5.7.5: H2 is built only if its ITL is 8 % below the best deployment that
    # needs no hx protocol; in none of the three W7800 scenarios is it.
    for name, kw in pm.ITL_SCENARIOS:
        ratio = pm.h2_ratio(pm.itl_curve(pm.Assume(**kw)))
        assert min(v for v in ratio.values() if v is not None) > pm.H2_GO_RATIO, name
    central = pm.h2_ratio(pm.itl_curve(pm.Assume()))
    assert central[24] is None


def test_stage_plan_json_round_trip():
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    plan = json.loads(out)["stage_plan"]
    assert {
        "h1_ladder",
        "h1_gates",
        "h1_fit",
        "stage2",
        "spill_k",
        "itl",
        "h2_ratio",
        "dense_offload",
        "g7",
        "g7_stage2",
        "loopback",
        "loopback_python_stack",
        "t5",
        "t5_budget",
        "loopback_w7800",
        "loopback_python_stack_w7800",
        "loopback_refit_w7800",
        "t5_w7800",
        "routing",
        "routing_touched",
        "routing_stage2",
        "dispatch",
        "asbuilt",
        "standin",
        "loopback_interim",
    } <= set(plan)
    assert plan["t5"][0]["w"] == 3 and plan["g7_stage2"]["model"]["d3"] > 700
    assert plan["h1_gates"]["parity_c1"] == pytest.approx(22.38 * 1.10, abs=0.01)


def test_spill_k_defaults():
    # Spec 6.2: H2 per hub keeps 7 / 39 / 63 touched experts at W = 3 / 24 / 48;
    # H1's hub alone covers the whole expert set, so its balance points are higher.
    h2 = pm.spill_k(pm.Assume(), "H2")
    assert [round(h2[w]) for w in (3, 24, 48)] == [7, 39, 63]
    h1 = pm.spill_k(pm.Assume(), "H1")
    assert [round(h1[w]) for w in pm.SPILL_WIDTHS] == [10, 21, 39, 65, 110]
    assert all(h1[w] > h2[w] for w in pm.SPILL_WIDTHS)
    assert list(h1.values()) == sorted(h1.values())


def test_exchange_term_is_unchanged_by_the_transport_split():
    # `remote_us` = `transport_us` + server expert compute; at four servers the exchange
    # alone is 28.5 / 63.2 / 102.7 us at W = 3 / 24 / 48 (spec 11.2).
    a = pm.Assume()
    got = [pm.transport_us(w, 4, a) for w in (3, 24, 48)]
    assert got == pytest.approx([28.5, 63.2, 102.7], abs=0.1)


def test_model_exchange_is_optimistic_against_the_hx_loopback():
    # HxServer's loopback (V620 hub, three servers, graph mode, no expert compute):
    # measured 35.9 / 64.2 / 98.6 us against the model's 27.3 / 53.3 / 82.9 us.
    rows = {r["w"]: r for r in pm.loopback_rows(pm.Assume())}
    assert [round(rows[w]["model"], 1) for w in (3, 24, 48)] == [27.3, 53.3, 82.9]
    assert all(15.0 < r["pct"] < 35.0 for r in rows.values())
    # Raised to the measured exchange, the H1 steps move by about one percent: inside
    # the green band of the stage-1 gate (15 %).
    for r in pm.loopback_refit(pm.Assume()):
        assert 0.0 < r["after"] / r["before"] - 1.0 < 0.02


def test_t5_budget_belongs_to_the_plan_under_test():
    # HxServer's T5 on three V620 servers (plan_coldonly, graph mode): the old
    # 47 / 161 / 256 us budget is the H2 remote path; for the plan under test the model
    # is the exchange plus the busiest server's experts, and the server chain matches.
    rows = {r["w"]: r for r in pm.t5_rows(pm.Assume())}
    assert [round(rows[w]["model"]) for w in (3, 24, 48)] == [97, 329, 499]
    assert all(r["measured"] > 1.3 * r["old_gate"] for r in rows.values())
    # with the exchange measured through the Python stack the model is within 5 %
    assert all(0.95 < r["ratio_replay"] < 1.06 for r in rows.values())
    # the server compute (real minus synthetic servers) is the model's within 6 %
    assert all(0.93 < r["compute_meas"] / r["compute"] < 1.06 for r in rows.values())
    lim = {(r["d"], r["w"]): r["limit"] for r in pm.t5_budget(pm.Assume())}
    assert lim[(3, 3)] == pytest.approx(1.3 * rows[3]["model"])
    assert lim[(4, 24)] < lim[(3, 24)]  # four servers share the experts more thinly


def test_python_stack_exchange_costs_h1_a_few_percent():
    # The T5 replay's exchange (86 / 99 / 134 us against the loopback's 36 / 64 / 99)
    # moves the H1 decode steps by about 2-7 %; the green band of 5.4a absorbs it.
    gates = pm.h1_gates(pm.Assume())
    for r in pm.loopback_refit(pm.Assume(), pm.T5_SYNTH_US):
        assert 1.02 < r["after"] / r["before"] < 1.08
        assert r["after"] < gates["green_" + r["mode"]]


def test_w7800_hub_exchange_exceeds_the_model_by_a_growing_amount():
    # HxServer, W7800 hub + four V620 servers (hetero-hx doc 8.6): graph-mode loopback
    # 48 / 90 / 157 us per layer against the model's 28.5 / 63.2 / 102.7 us; through the
    # Python stack (T5 replay with synthetic servers) 80.4 / 123.0 / 178.2 us.
    a = pm.Assume()
    lb = pm.loopback_rows(a, pm.W7800_LOOPBACK_US, pm.W7800_D)
    assert [r["model"] for r in lb] == pytest.approx([28.5, 63.2, 102.7], abs=0.1)
    assert [r["delta"] for r in lb] == pytest.approx([19.5, 27.2, 54.6], abs=0.2)
    py = pm.loopback_rows(a, pm.W7800_T5_SYNTH_US, pm.W7800_D)
    assert [r["delta"] for r in py] == pytest.approx([51.9, 59.8, 75.5], abs=0.2)
    assert lb[-1]["delta"] > 2.0 * lb[0]["delta"]
    assert "T5 W7800" in pm.md_w7800(a)


def test_t5_on_the_w7800_hub_is_the_model_plus_the_python_exchange():
    # layer time / (1.3 x model) = 1.28 / 0.93 / 0.92 as HxServer reports; the server
    # chain (real minus synthetic servers) is the model's within 9 % and under its
    # 1.15 x n_max x 6.67 us limits (65 / 249.5 / 371 us); with the synthetic exchange
    # added the model is within 4 %.
    rows = {r["w"]: r for r in pm.t5_rows(pm.Assume(), "w7800")}
    assert [round(rows[w]["ratio"] / 1.3, 2) for w in (3, 24, 48)] == [1.28, 0.93, 0.92]
    assert all(0.98 < r["compute_meas"] / r["compute"] < 1.09 for r in rows.values())
    assert all(0.98 < r["ratio_replay"] < 1.04 for r in rows.values())
    limits = [1.15 * rows[w]["compute"] for w in (3, 24, 48)]
    assert limits == pytest.approx([65.0, 249.5, 371.1], abs=0.3)
    for w, lim in zip((3, 24, 48), limits, strict=True):
        assert rows[w]["compute_meas"] < lim


def test_w7800_exchange_moves_h1_by_about_two_ms_and_stays_green():
    # The Python-stack exchange of the W7800 hub adds 52 / 60 / 76 us per layer over the
    # model's; the hub's own experts hide part of it, so the H1 step moves by 1.5 to
    # 3 ms, and c1 and c8 stay inside the green limits of the stage-1 gate.
    gates = pm.h1_gates(pm.Assume())
    for routing in ("", "t10"):
        a = pm.Assume(routing=routing)
        rows = pm.loopback_refit(a, pm.W7800_T5_SYNTH_US, pm.W7800_D, pm.W7800_MODES)
        assert [r["mode"] for r in rows] == ["c1", "c8", "c16"]
        assert all(1.5 < r["after"] - r["before"] < 3.0 for r in rows)
        for r in rows[:2]:
            assert r["after"] < gates["green_" + r["mode"]]


def test_g7_interpolation_and_scope():
    assert pm.g7_at((1.0, 2.0, 4.0), 3) == 1.0
    assert pm.g7_at((1.0, 2.0, 4.0), 24) == 2.0
    assert pm.g7_at((1.0, 2.0, 4.0), 36) == pytest.approx(3.0)
    assert pm.g7_at((1.0, 2.0, 4.0), 99) == 4.0
    # V620 engines keep the model: only gfx1100 topologies at TP1 and TP2 take G7
    a, ts = pm.setup(pm.Assume(g7="best"))
    assert pm.g7_op(ts["TP4"], a, "hc", 24) is None
    assert pm.g7_op(ts["H1"], a, "hc", 24) == pytest.approx(41.7)
    base = _run()[2]["rows"]["TP4"]["c1"]["step"]
    assert pm.rows_for(pm.Assume(), g7="best")["TP4"]["c1"]["step"] == pytest.approx(
        base
    )


def test_g7_routes_decide_the_c8_step():
    # G7 (W7800, HIP 0): the torch HC preamble costs 88.5 us per call at W = 24 against
    # 41.7 us for the HIP family, and the shared-expert composition is 2x the fused op:
    # `VLLM_RDNA3_HIP_OPS=hc,se` is worth about 5 ms of the H1 c8 step.
    rows = {(r["engine"], r["name"]): r for r in pm.g7_refit(pm.Assume())}
    best = rows[("H1", "G7, best routes")]
    default = rows[("H1", "G7, default routes (torch HC and shared expert)")]
    assert 3.0 < default["c8"] - best["c8"] < 8.0
    gates = pm.h1_gates(pm.Assume())
    assert best["c1"] < gates["green_c1"] and best["c8"] < gates["green_c8"]
    # at c1 the HIP preamble is not slower than the model (33 us against 46 us)
    assert best["c1"] < rows[("H1", "model")]["c1"]
    # the lm_head and shared expert grow with W: c16 is the width G7 hurts most
    assert best["c16"] - rows[("H1", "model")]["c16"] > 3.0


def test_g7_refit_moves_the_one_user_case_to_the_edge_of_the_rule():
    # With the G7 per-op numbers (HIP hc and se routes) the W7800 engines get faster at
    # W = 3 (HC 33 us against the modelled 46) and slower at W = 24 and 48 (lm_head,
    # shared-expert fallback). D3 and D5 move by about 1 %; H2 with both levers at one
    # user falls below TP8 by 7.7 %, just short of the 8 % of the decision rule, and is
    # behind the best deployment without hx from two users on.
    s2 = pm.g7_stage2(pm.Assume())
    base, g7 = s2["model"], s2["G7, best routes"]
    assert abs(g7["d3"] / base["d3"] - 1.0) < 0.02
    assert abs(g7["d5l"] / base["d5l"] - 1.0) < 0.02
    assert 0.98 < base["ratio"][1] < 1.0  # the central model: 1 % below TP8
    assert pm.H2_GO_RATIO < g7["ratio"][1] < 0.93
    assert all(g7["ratio"][u] > 1.0 for u in (2, 4, 8, 16))


def _stats_json(tmp_path, name, probs, rows=1000):
    """A converted stats file (`hetero-moe-expert-stats/1`) with one selection
    probability per expert, the same in every one of three target layers, and an
    MTP layer that the tables must skip."""
    layer = {"kind": "target", "decode": {"counts": [p * rows for p in probs]}}
    mtp = {"kind": "mtp", "decode": {"counts": [rows] * len(probs)}}
    doc = {"phases": {"decode": {"tokens": rows}}, "layers": [layer] * 3 + [mtp]}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_t10_tables_of_uniform_routing_are_the_uniform_formulas(tmp_path):
    q = pm.TOPK / pm.N_EXP
    f = _stats_json(tmp_path, "u", [q] * pm.N_EXP)
    t = pm.t10_tables((f, f))
    assert t["layers"] == 3
    for w, n in zip(t["w"], t["n"]):
        assert n == pytest.approx(pm.N_EXP * (1.0 - (1.0 - q) ** w), rel=1e-9)
    # every expert equally popular: 130 hub slots hold 130 / 512 of what is touched
    assert t["hit"][1] == pytest.approx((130 / 512,) * len(t["w"]))
    assert t["hit"][2] == pytest.approx((115 / 256,) * len(t["w"]))


def test_t10_hit_is_scored_on_the_other_window(tmp_path):
    # Window A routes only to experts 0-255, window B only to 256-511 (10 selections
    # per row, so p = 10 / 256 each).  A hub chosen on A and scored on A holds 130 of
    # the 256 touched experts; scored on B it holds none.
    p = 10 / 256
    a = _stats_json(tmp_path, "a", [p] * 256 + [0.0] * 256)
    b = _stats_json(tmp_path, "b", [0.0] * 256 + [p] * 256)
    same = pm.t10_tables((a, a))["hit"][1]
    other = pm.t10_tables((a, b))["hit"][1]
    assert same == pytest.approx((130 / 256,) * len(same))
    assert other == pytest.approx((0.0,) * len(other))


def test_t10_constants_describe_a_skewed_router():
    q = pm.TOPK / pm.N_EXP
    n, w = pm.T10_N, pm.T10_W
    assert n[0] == pytest.approx(pm.TOPK) and w[0] == 1
    for wi, ni in zip(w[1:], n[1:]):
        assert ni < pm.N_EXP * (
            1.0 - (1.0 - q) ** wi
        )  # skew: fewer distinct than uniform
    slopes = [(n[i + 1] - n[i]) / (w[i + 1] - w[i]) for i in range(len(w) - 1)]
    assert slopes == sorted(slopes, reverse=True)  # concave
    for hit in pm.T10_HIT.values():
        assert list(hit) == sorted(hit, reverse=True)  # fewer hits as rows widen
    assert all(h2 > h1 for h1, h2 in zip(pm.T10_HIT[1], pm.T10_HIT[2]))
    # the central assumption was a little high at c8 and 11 % high at c16
    central = pm.Assume()
    assert 0.93 < pm.T10_N[w.index(24)] / pm.n_touched(24, central) < 0.96
    assert 0.86 < pm.T10_N[w.index(48)] / pm.n_touched(48, central) < 0.92
    # the table ends at 128 rows; beyond it the last measured ratio to uniform holds
    t10 = pm.Assume(routing="t10")
    assert pm.n_touched(128, t10) == pytest.approx(pm.T10_N[-1])
    assert pm.T10_N[-1] < pm.n_touched(256, t10) < pm.N_EXP
    with pytest.raises(ValueError):
        pm.n_touched(24, pm.Assume(routing="zipf"))


def test_t10_routing_keeps_the_anchors_and_moves_the_engines_little():
    # Measured routing has fewer distinct experts per layer than the central assumption,
    # but the TP4 anchors absorb the difference in the attention scale: the TP4 steps
    # stay put, the held-out TP8 prediction stays inside its 5 %, and the hub engines
    # move by at most 3 %, so H1 stays inside the green band of the stage-1 gate.
    base = pm.rows_for(pm.Assume())
    t10 = pm.rows_for(pm.Assume(), routing="t10")
    assert t10["TP4"]["c1"]["step"] == pytest.approx(22.38, abs=0.05)
    assert t10["TP4"]["c8"]["step"] == pytest.approx(43.97, abs=0.05)
    assert t10["TP8"]["c1"]["step"] == pytest.approx(19.59, rel=0.05)
    assert t10["TP8"]["c8"]["step"] == pytest.approx(37.2, rel=0.05)
    for kind in ("W2", "H1", "H2"):
        for mode in ("c1", "c8", "c16"):
            ratio = t10[kind][mode]["step"] / base[kind][mode]["step"]
            assert 0.97 < ratio < 1.03, (kind, mode, ratio)
    gates = pm.h1_gates(pm.Assume())
    assert t10["H1"]["c1"]["step"] < gates["green_c1"]
    assert t10["H1"]["c8"]["step"] < gates["green_c8"]


def test_t10_spill_k_seeds_are_lower_at_wide_steps():
    # fewer distinct experts per layer leave the hub less to keep: the balance point
    # of H1 at 48 rows drops from 110 to 98; the resident share (0.44) still covers it.
    a = pm.Assume(routing="t10")
    h1, h2 = pm.spill_k(a, "H1"), pm.spill_k(a, "H2")
    assert [h1[w] for w in pm.SPILL_WIDTHS] == pytest.approx(
        [10.3, 20.5, 36.5, 62.7, 98.1], abs=0.6
    )
    assert [h2[w] for w in pm.SPILL_WIDTHS] == pytest.approx(
        [7.0, 12.6, 22.1, 37.3, 57.7], abs=0.6
    )
    central = pm.spill_k(pm.Assume(), "H1")
    assert all(h1[w] <= central[w] for w in pm.SPILL_WIDTHS)
    ts = pm.setup(a)[1]
    for w in pm.SPILL_WIDTHS:  # K never exceeds the resident touched experts
        cap = pm.hit_at(ts["H1"], w, a) * pm.n_touched(w, a)
        assert h1[w] <= cap + 1e-6


def test_t10_routing_does_not_change_the_stage2_choice():
    s2 = pm.routing_stage2(pm.Assume())
    base, t10 = s2["central routing"], s2["T10 measured routing"]
    assert abs(t10["d3"] / base["d3"] - 1.0) < 0.01
    assert abs(t10["d5l"] / base["d5l"] - 1.0) < 0.01
    # with the G7 routes H2 + both levers at one user stays at the edge of the 8 % rule
    g7 = s2["T10 + G7 best routes"]
    assert pm.H2_GO_RATIO < g7["ratio"][1] < 0.93
    assert all(g7["ratio"][u] > 1.0 for u in (2, 4, 8, 16))


# ExpertStats' plan seeds (`runtime.spill_k_by_dispatch` of `H1.W1` and `H2.W1`,
# feat/expert-stats b0a52c197b): kept hub-resident experts per hub and layer under the
# correlated rows, by the decode dispatch rule, at W = 3 / 6 / 12 / 24 / 48.
ES_K = {
    ("H1", "row"): (9, 17, 31, 53, 87),
    ("H1", "primary"): (9, 16, 28, 48, 79),
    ("H2", "row"): (7, 11, 18, 31, 52),
    ("H2", "primary"): (7, 11, 17, 29, 48),
}


def test_t10c_constants_describe_correlated_rows():
    w = pm.T10C_W
    assert w[0] == 1 and pm.T10C_N[0] == pytest.approx(pm.TOPK)
    n3, w3 = (
        pm.T10C_N[1:],
        w[1:],
    )  # the first group of three rows is linear, not concave
    slopes = [(n3[i + 1] - n3[i]) / (w3[i + 1] - w3[i]) for i in range(len(w3) - 1)]
    assert slopes == sorted(slopes, reverse=True)  # concave from three rows on
    q = pm.TOPK / pm.N_EXP
    for wi, ni in zip(w, pm.T10C_N):
        assert ni <= pm.N_EXP * (1.0 - (1.0 - q) ** wi) + 1e-9
        if wi > 1:  # rows that share experts touch fewer distinct ones than T10
            ratio = ni / pm.interp_at(pm.T10_W, pm.T10_N, wi)
            assert 0.75 < ratio < 0.82, (wi, ratio)
    for key, hit in pm.T10C_HIT.items():
        assert list(hit) == sorted(hit, reverse=True), key
    for h2, h1, h1v in zip(pm.T10C_HIT["H2"], pm.T10C_HIT["H1"], pm.T10C_HIT["H1v"]):
        assert h2 > h1 > h1v
    # the marginal is right: the hub hit is the independent-row value within 0.035
    for wi, h1, h2 in zip(w, pm.T10C_HIT["H1"], pm.T10C_HIT["H2"]):
        assert abs(h1 - pm.interp_at(pm.T10_W, pm.T10_HIT[1], wi)) < 0.035
        assert abs(h2 - pm.interp_at(pm.T10_W, pm.T10_HIT[2], wi)) < 0.035
    for dup in pm.T10C_DUP.values():
        assert dup[0] == 1.0 and list(dup) == sorted(dup)
    assert all(h2 > h1 for h1, h2 in zip(pm.T10C_DUP["H1"][1:], pm.T10C_DUP["H2"][1:]))
    # beyond 48 rows the last ratio to uniform holds
    t10c = pm.Assume(routing="t10c")
    assert pm.n_touched(48, t10c) == pytest.approx(pm.T10C_N[-1])
    assert pm.T10C_N[-1] < pm.n_touched(96, t10c) < pm.N_EXP
    for rows, hubs, key in ((1, 1, "H1"), (1, 2, "H2")):  # keyed by hubs or plan
        t = pm.Topo("x", pm.W7800, 1, "hub", hubs)
        assert pm.plan_key(t) == key and rows == 1
    assert pm.plan_key(pm.Topo("x", pm.V620, 1, "hub", 1, plan="H1v")) == "H1v"


def test_row_dispatch_streams_replicated_experts_twice():
    t = pm.setup(pm.Assume(routing="t10c"))[1]["H1"]
    prim = pm.Assume(routing="t10c")
    row = pm.Assume(routing="t10c", dispatch="row")
    assert pm.dup_at(t, 24, prim) == 1.0
    assert pm.dup_at(t, 24, row) == pytest.approx(1.238)
    assert pm.dup_at(t, 100, row) == pytest.approx(pm.T10C_DUP["H1"][-1])  # flat
    for bad in (pm.Assume(dispatch="row"), pm.Assume(routing="t10", dispatch="row")):
        with pytest.raises(ValueError):
            pm.dup_at(t, 24, bad)
    with pytest.raises(ValueError):
        pm.dup_at(t, 24, pm.Assume(routing="t10c", dispatch="expert"))
    for kind in ("H1", "H2"):
        a, ts = pm.setup(row)
        for w in pm.SPILL_WIDTHS:
            s = pm.lat_scale(ts[kind], w // 3)
            p = pm.moe_hub_us(ts[kind], w, prim, s)
            r = pm.moe_hub_us(ts[kind], w, row, s)
            assert r[1]["dup"] > 1.0 and p[1]["dup"] == 1.0
            assert r[0] >= p[0]  # the servers stream more, never less
            assert r[1]["n_hub"] >= p[1]["n_hub"]  # the hub keeps more of its own


def test_dispatch_rows_price_decision_16():
    rows = pm.dispatch_rows(pm.Assume())
    assert len(rows) == 8
    by = {(r["engine"], r["calib"], r["rule"]): r for r in rows}
    limits = {  # row - primary, ms at c1 / c8 / c16
        "H1": ((0.0, 0.3), (0.5, 1.3), (1.0, 2.5)),
        "H2": ((0.0, 0.2), (0.2, 0.7), (0.4, 1.2)),
    }
    for kind in ("H1", "H2"):
        for calib, _ in pm.CALIBRATIONS:
            p, r = by[kind, calib, "primary"], by[kind, calib, "row"]
            for (lo, hi), a, b in zip(limits[kind], p["steps"], r["steps"]):
                assert lo < b - a < hi, (kind, calib, a, b)
            assert 1.005 < r["steps"][1] / p["steps"][1] < 1.03
    h1 = by["H1", "anchored", "row"]
    assert h1["dup"] == pytest.approx((1.181, 1.197, 1.216, 1.238, 1.283), abs=1e-6)
    assert by["H2", "anchored", "primary"]["dup"] == (1.0,) * 5


def test_model_spill_k_agrees_with_expertstats_plan_seeds():
    # Two implementations of the balance point (the model's closed form, ExpertStats'
    # simulated spill rule) agree within 12 % or 1.5 experts at every capture width.
    for (kind, rule), seeds in ES_K.items():
        a = pm.Assume(routing="t10c", dispatch=rule)
        k = pm.spill_k(a, kind)
        assert [k[w] for w in pm.SPILL_WIDTHS] == pytest.approx(
            seeds, rel=0.12, abs=1.5
        )
    t = pm.routing_touched(pm.Assume())
    for r in t:  # correlated rows leave the hub less to keep than independent ones
        assert r["k_h1c"][0] < r["k_h1"][1] and r["k_h2c"][0] < r["k_h2"][1]
        assert r["k_h1c"][1] >= r["k_h1c"][0] and r["k_h2c"][1] >= r["k_h2c"][0]


def test_digest_adds_its_measured_cost_to_every_exchange():
    off, on = pm.Assume(), pm.Assume(digest=True)
    for w, us in zip(pm.DIGEST_W, pm.DIGEST_US):
        assert pm.transport_us(w, 4, on) - pm.transport_us(w, 4, off) == pytest.approx(
            us
        )
    assert pm.digest_us(12) == pytest.approx(12.0 + (16.5 - 12.0) * 9 / 21)
    assert pm.digest_us(100) == pm.DIGEST_US[-1]  # flat beyond the table
    # HxServer: 0.6 to 0.8 ms per 48-layer step; the hub takes some of the work back
    # (the balance point moves), so the model's step grows by a little less
    base = pm.rows_for(pm.Assume(routing="t10c"))["H1"]
    dig = pm.rows_for(pm.Assume(routing="t10c", digest=True))["H1"]
    for mode, wd in (("c1", 3), ("c8", 24), ("c16", 48)):
        gain = dig[mode]["step"] - base[mode]["step"]
        assert 0.2 < gain < 48 * pm.digest_us(wd) / 1e3 + 1e-9, (mode, gain)
    # the central defaults are untouched by the new knobs
    d = pm.Assume()
    assert (d.dispatch, d.digest, d.pcie_bw, d.anchor) == ("primary", False, 25e9, None)


def test_pcie_bw_is_the_wire_term_of_the_exchange():
    x8 = pm.Assume(pcie_bw=pm.INTERIM_BW)
    for w in (3, 24, 48):
        extra = (
            4
            * w
            * (pm.ROW_REQ_B + pm.ROW_B)
            * (1 / pm.INTERIM_BW - 1 / pm.PCIE_BW)
            * 1e6
        )
        gap = pm.transport_us(w, 4, x8) - pm.transport_us(w, 4, pm.Assume())
        assert gap == pytest.approx(extra)
    # at x8 the four-server loopback exchange is 14 to 79 % above the model
    rows = pm.loopback_rows(x8, pm.INTERIM_LOOPBACK_US, pm.INTERIM_D)
    assert [round(r["pct"]) for r in rows] == [79, 22, 14]
    for w in pm.INTERIM_LOOPBACK_US:  # both loopback tables differ by the digest cost
        delta = pm.INTERIM_LOOPBACK_DIGEST_US[w] - pm.INTERIM_LOOPBACK_US[w]
        assert delta == pytest.approx(pm.digest_us(w), abs=0.05)


def test_anchor_picks_what_the_correlated_routing_is_allowed_to_move():
    # Default: TP4 stays on its two measured steps, so the lower expert time of the
    # correlated rows goes into the latency scale (0.484 -> 0.657 at c8) and every other
    # engine sees it.  anchor="": the central scale is kept and TP4 itself gets faster.
    t10c = pm.rows_for(pm.Assume(), routing="t10c")
    kept = pm.rows_for(pm.Assume(), routing="t10c", anchor="")
    central = pm.setup(pm.Assume())[1]["TP4"].s_l
    assert pm.setup(pm.Assume(routing="t10c"))[1]["TP4"].s_l["c8"] > 1.3 * central["c8"]
    assert pm.setup(pm.Assume(routing="t10c", anchor=""))[1]["TP4"].s_l == central
    assert t10c["TP4"]["c8"]["step"] == pytest.approx(43.97, abs=0.05)
    assert kept["TP4"]["c8"]["step"] < 41.0
    # held-out TP8 (hierarchical all-reduce, 37.2 ms): the anchored model stays inside
    # 5 %, keeping the calibration does not (-6 %); against the flat-tree 41.56 ms the
    # anchored model is nearer as well
    tp8, tp8_kept = t10c["TP8"]["c8"]["step"], kept["TP8"]["c8"]["step"]
    assert tp8 == pytest.approx(37.2, rel=0.05)
    assert tp8_kept != pytest.approx(37.2, rel=0.05)
    assert abs(tp8 - 41.56) < abs(tp8_kept - 41.56)
    # the sign of the effect on the hub engines depends on that choice, which the TP4
    # anchor cannot settle: H1 at c8 is +3.4 ms or -3.0 ms against the central model
    base = pm.rows_for(pm.Assume())["H1"]["c8"]["step"]
    assert t10c["H1"]["c8"]["step"] - base > 2.0
    assert kept["H1"]["c8"]["step"] - base < -2.0


def test_asbuilt_ladder_stays_inside_the_green_band_with_a_thin_c8_margin():
    gates = pm.h1_gates(pm.Assume())
    rungs = pm.asbuilt_ladder(pm.Assume())
    assert len(rungs) == 6
    for calib, _ in pm.CALIBRATIONS:
        for r in rungs:
            assert r[calib][0] < gates["green_c1"], (calib, r["name"])
            assert r[calib][1] < gates["green_c8"], (calib, r["name"])
    anchored = [r["anchored"] for r in rungs]
    assert anchored[0] == pytest.approx((32.05, 52.87, 75.32), abs=0.02)  # the design
    # correlated rows, row dispatch, the digest and the Python-stack exchange each cost
    # time in the anchored scenario
    for prev, nxt in zip(anchored[2:], anchored[3:]):
        assert all(b > a for a, b in zip(prev, nxt))
    last = anchored[-1]
    assert last == pytest.approx((35.1, 60.0, 86.6), abs=0.15)
    assert 0.97 * gates["green_c8"] < last[1] < gates["green_c8"]
    # keeping the calibration puts the same stack 6.4 ms lower at c8
    kept = rungs[-1]["calibration kept"]
    assert 5.0 < last[1] - kept[1] < 8.0


def test_standin_model_reproduces_the_v620_hub_run():
    # HubMethod's compiled stand-in (hub doc 13): c1 ITL 43.9 ms, c8 cell 50.2 ms (two
    # requests resident).  The model of that topology (V620 hub TP1, 40 slots, x8 links,
    # the run's routing, dispatch and digest) lands within 7 % on either calibration.
    rows = pm.standin_rows(pm.Assume())
    assert len(rows) == 8
    assert {r["measured"] for r in rows} == {43.9, 50.2}
    for r in rows:
        assert abs(r["pct"]) < 7.0, r
    own = [r for r in rows if r["exchange"] == "model"]
    assert all(abs(r["pct"]) < 4.0 for r in own)
    # the model overshoots c1 in every variant: the stand-in shows no sign of the
    # Python-stack gap of the T5 replay (+1.7 to +2.4 ms per step)
    assert all(r["pct"] > 0 for r in rows if r["case"] == "c1")
    # the 40-slot plan keeps few experts on the hub: low hit, K tops out at W = 24
    assert pm.T10C_HIT["H1v"][4] < 0.25


def test_routing_stage2_with_correlated_rows_does_not_help_h2():
    s2 = pm.routing_stage2(pm.Assume())
    base = s2["central routing"]
    anchored = s2["T10 correlated rows"]
    kept = s2["T10 correlated rows, calibration kept"]
    assert abs(anchored["d3"] / base["d3"] - 1.0) < 0.01
    assert all(anchored["ratio"][u] > 1.0 for u in (2, 4, 8, 16))
    assert anchored["ratio"][8] > base["ratio"][8]  # H2 looks worse, not better
    assert min(kept["ratio"][u] for u in (2, 4, 8, 16)) > 1.0
