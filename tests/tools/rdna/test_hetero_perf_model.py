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


def test_h1_is_dominated_and_h2_is_not(central):
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
