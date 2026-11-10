# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the T5 harness helpers (tools/rdna/hetero_moe/t5_common.py).

numpy only; the tests that need the plan/tables modules of the hetero_moe package
skip when ``regex`` is missing. Run on a machine without the repo conftest
requirements with ``--noconftest``.
"""

import numpy as np
import pytest

from tools.rdna.hetero_moe import t5_common as C

SHAPE = dict(nlayers=3, num_experts=32, top_k=4, hidden=16, x_pool=2)


def _inputs(seed=1, width=5, pattern=0, **kw):
    return C.make_inputs(seed, width, pattern, **{**SHAPE, **kw})


# ---- inputs -----


def test_make_inputs_is_deterministic_and_keyed_by_seed_width_pattern():
    a, b = _inputs(), _inputs()
    assert C.inputs_digest(a) == C.inputs_digest(b)
    assert a.x.dtype == np.float16 and a.x.shape == (2, 5, 16)
    assert a.ids.dtype == np.int32 and a.ids.shape == (3, 5, 4)
    assert a.w.dtype == np.float32 and a.w.shape == (3, 5, 4)
    digests = {
        C.inputs_digest(_inputs(seed=2)),
        C.inputs_digest(_inputs(width=6)),
        C.inputs_digest(_inputs(pattern=1)),
        C.inputs_digest(a),
    }
    assert len(digests) == 4


def test_x_of_layer_cycles_the_pool():
    inp = _inputs()
    assert inp.x_of(0) is not None
    assert np.array_equal(inp.x_of(2), inp.x[0])
    assert np.array_equal(inp.x_of(3), inp.x[1])


def test_ids_are_distinct_per_row_and_in_range():
    inp = _inputs(width=64, num_experts=512, top_k=10, nlayers=4)
    for layer in range(4):
        rows = inp.ids[layer]
        assert rows.min() >= 0 and rows.max() < 512
        assert all(len(set(r)) == 10 for r in rows.tolist())


def test_popularity_is_skewed_and_differs_by_layer():
    inp = _inputs(width=256, num_experts=64, top_k=4, nlayers=3)
    hot = []
    for layer in range(3):
        counts = np.bincount(inp.ids[layer].ravel(), minlength=64)
        assert counts.max() > 3 * counts.mean()  # Zipf: a few experts dominate
        hot.append(int(counts.argmax()))
    assert len(set(hot)) > 1  # the hot expert moves with the layer permutation


def test_weights_are_positive_and_normalised():
    inp = _inputs()
    assert (inp.w > 0).all()
    assert np.allclose(inp.w.sum(axis=2), 1.0, atol=1e-6)


def test_pad_rows_have_masked_ids():
    inp = _inputs(width=6, pad_rows=2)
    assert (inp.ids[:, 4:, :] == -1).all()
    assert (inp.ids[:, :4, :] >= 0).all()
    with pytest.raises(ValueError):
        _inputs(width=3, pad_rows=3)


def test_synthetic_y_hub_is_deterministic_small_fp16():
    a = C.synthetic_y_hub(1, 5, 0, 7, 5, 16)
    assert a.dtype == np.float16 and a.shape == (5, 16)
    assert np.array_equal(a, C.synthetic_y_hub(1, 5, 0, 7, 5, 16))
    assert not np.array_equal(a, C.synthetic_y_hub(1, 5, 0, 8, 5, 16))
    assert np.abs(a).max() < 1.0


# ---- digests -----


def test_digest_is_sensitive_to_one_bit_and_equal_for_bytes_and_arrays():
    y = np.arange(12, dtype=np.float16).reshape(3, 4)
    assert C.digest(y) == C.digest(y.tobytes()) == C.digest(np.copy(y))
    z = y.copy()
    z.view(np.uint16)[2, 3] ^= 1
    assert C.digest(z) != C.digest(y)
    assert len(C.digest(y)) == 32
    assert C.digest(y.T) == C.digest(np.ascontiguousarray(y.T))


# ---- wire rules -----


def test_combine_reference_matches_a_plain_loop_left_to_right():
    rng = np.random.default_rng(3)
    y_hub = rng.standard_normal((4, 6)).astype(np.float16)
    ys = [(rng.standard_normal((4, 6)) * 100).astype(np.float16) for _ in range(4)]
    out = C.combine_reference(y_hub, ys)
    assert out.dtype == np.float16
    for r in range(4):
        for c in range(6):
            acc = np.float32(y_hub[r, c])
            for y in ys:
                acc = np.float32(acc + np.float32(y[r, c]))
            assert out[r, c] == np.float16(acc)


def test_combine_order_matters_and_is_fp32_not_fp16():
    big, small = np.float16(2048), np.float16(1)
    y_hub = np.array([[big]], np.float16)
    left = C.combine_reference(y_hub, [np.array([[small]], np.float16)] * 2)
    # fp32 keeps both ones: 2050 is representable in fp16 (step 2 above 2048)
    assert left[0, 0] == np.float16(2050)
    # a pairwise fp16 sum would lose each 1 (2048 + 1 rounds to 2048)
    assert np.float16(big + small) == big


def test_server_ids_masks_other_destinations():
    ids = np.array([[5, 6, 7], [8, 9, -1]], np.int32)
    dest = np.array([[0, 1, -1], [1, 1, -1]], np.int32)
    assert C.server_ids(ids, dest, 0).tolist() == [[5, -1, -1], [-1, -1, -1]]
    assert C.server_ids(ids, dest, 1).tolist() == [[-1, 6, -1], [8, 9, -1]]
    assert C.server_ids(ids, dest, 1).dtype == np.int32


@pytest.fixture(scope="module")
def hx_env():
    pytest.importorskip("regex")
    from tests.quantization.hetero_moe_support import hetero, stripe_doc

    return hetero, stripe_doc


def test_expected_split_matches_the_rule_and_partitions_the_pairs(hx_env):
    hetero, stripe_doc = hx_env
    plan_mod, tables_mod = hetero("plan"), hetero("tables")
    plan = plan_mod.plan_from_doc(stripe_doc(experts=64, layers=3, top_k=4, mtp=True))
    tables = tables_mod.build_island_tables(
        plan, plan.hubs(0)[0], spill_k={3: 0, 24: 1}
    )
    assert C.hx_layers(tables) == [0, 1, 2]  # the MTP layer 3 is hub-only
    inp = C.make_inputs(1, 24, 0, nlayers=3, num_experts=64, top_k=4, hidden=8)
    for k_keep in (0, 1, 1000):
        for layer in range(3):
            hub_ids, dest = C.expected_split(tables, layer, inp.ids[layer], k_keep)
            want = tables_mod.split_pairs_loop(
                inp.ids[layer], tables.layers[layer], layer, k_keep
            )
            assert np.array_equal(hub_ids, want[0]) and np.array_equal(dest, want[1])
            on_servers = np.zeros_like(inp.ids[layer], dtype=bool)
            for s in range(tables.num_servers):
                part = C.server_ids(inp.ids[layer], dest, s)
                on_servers |= part >= 0
                assert np.array_equal(part[part >= 0], inp.ids[layer][part >= 0])
            on_hub = hub_ids >= 0
            assert not (on_hub & on_servers).any()
            assert (on_hub | on_servers).all()  # every pair is placed somewhere
    hub_ids, _ = C.expected_split(tables, 0, inp.ids[0], 0)
    assert (hub_ids == -1).all()  # K = 0: everything goes to the servers
    hub_ids, _ = C.expected_split(tables, 0, inp.ids[0], 1000)
    assert (hub_ids >= 0).any()  # hub-resident experts stay


# ---- check_layer -----


def _layer_fixture():
    rng = np.random.default_rng(4)
    tap = rng.standard_normal((2, 3, 8)).astype(np.float16)
    y_hub = rng.standard_normal((3, 8)).astype(np.float16)
    out = C.combine_reference(y_hub, tap)
    hub_ids = np.array([[1, -1], [-1, -1], [2, 3]], np.int32)
    digests = [C.digest(t) for t in tap]
    return tap, out, hub_ids, y_hub, digests


def test_check_layer_passes_on_consistent_data():
    tap, out, hub_ids, y_hub, digests = _layer_fixture()
    assert C.check_layer(7, tap, out, hub_ids, y_hub, hub_ids, digests, list(tap)) == []


def test_check_layer_reports_each_violation():
    tap, out, hub_ids, y_hub, digests = _layer_fixture()
    bad_tap = tap.copy()
    bad_tap[1, 2, 5] += np.float16(1)
    problems = C.check_layer(
        7, bad_tap, out, hub_ids, y_hub, hub_ids, digests, list(tap)
    )
    text = "\n".join(problems)
    assert "layer 7 server 1: reply digest" in text
    assert (
        "reply bytes differ from golden: 1 element(s) differ, first at [2, 5]" in text
    )
    assert "combine output differs" in text  # out was built from the good tap
    assert "server 0" not in text

    bad_out = out.copy()
    bad_out.view(np.uint16)[0, 0] ^= 0x8000
    problems = C.check_layer(7, tap, bad_out, hub_ids, y_hub, hub_ids, digests)
    assert len(problems) == 1 and "combine output differs" in problems[0]

    other = hub_ids.copy()
    other[2, 1] = 9
    problems = C.check_layer(7, tap, out, hub_ids, y_hub, other, digests)
    assert len(problems) == 1
    assert "hub_ids_out differs from the split rule at 1 pair(s)" in problems[0]
    assert "row 2 slot 1: got 3, want 9" in problems[0]


def test_first_diff_counts_and_locates():
    a = np.zeros((2, 3), np.float16)
    assert C.first_diff(a, a.copy()) is None
    b = a.copy()
    b[1, 0] = 1
    b[1, 2] = 2
    d = C.first_diff(b, a)
    assert d["count"] == 2 and d["index"] == (1, 0) and d["got"] == 1.0
    assert "note" in C.first_diff(a, a.astype(np.float32))
    # -0.0 and +0.0 differ bytewise
    z = a.copy()
    z[0, 0] = -0.0
    assert C.first_diff(z, a)["count"] == 1


# ---- golden file -----


def _golden(tmp_path, **kw):
    g = C.Golden.new("0123456789abcdef", "exp0", 5, [0, 1])
    inp = _inputs(nlayers=2, width=3)
    for pattern in range(2):
        ys = [np.full((3, 16), pattern + i, np.float16) for i in range(2)]
        g.add_pattern(3, pattern, ys)
    g.add_width(3, 2, 0, {0: C.inputs_digest(inp), 1: "ff"})
    big = [np.zeros((2048, 16), np.float16)] * 2
    g.add_pattern(2048, 0, big)
    g.add_width(2048, 1, 7, {0: "aa"})
    return g, g.save(tmp_path / "golden_exp0.npz")


def test_golden_round_trip(tmp_path):
    g, path = _golden(tmp_path)
    assert path.name == "golden_exp0.npz" == C.golden_path(tmp_path, "exp0").name
    h = C.load_golden(path, plan_id="0123456789abcdef", server_id="exp0")
    assert h.meta == g.meta and h.digests == g.digests
    assert h.meta["format"] == "hx-t5-golden/1"
    assert h.meta["numpy_version"] == np.__version__
    assert h.widths() == [3, 2048] and h.layers == [0, 1]
    assert h.patterns(3) == 2 and h.k_keep(2048) == 7
    assert h.layer_digest(3, 1, 0) == C.digest(np.full((3, 16), 1, np.float16))
    assert np.array_equal(h.layer_y(3, 1, 1), np.full((3, 16), 2, np.float16))
    assert h.layer_y(3, 1, 1).dtype == np.float16
    assert h.layer_y(2048, 0, 0) is None  # no raw bytes above width 48
    assert h.layer_digest(2048, 0, 1) == C.digest(np.zeros((2048, 16), np.float16))
    with pytest.raises(C.GoldenError, match="no width 24"):
        h.k_keep(24)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"plan_id": "ffffffffffffffff"}, "golden is for plan"),
        ({"server_id": "exp1"}, "golden is of server exp0"),
    ],
)
def test_golden_validation_errors(tmp_path, kw, message):
    _, path = _golden(tmp_path)
    with pytest.raises(C.GoldenError, match=message):
        C.load_golden(path, **kw)


def test_golden_refuses_another_numpy_or_format(tmp_path, monkeypatch):
    _, path = _golden(tmp_path)
    monkeypatch.setattr(np, "__version__", "0.0.1")
    with pytest.raises(C.GoldenError, match="input generation differs"):
        C.load_golden(path)
    monkeypatch.undo()
    g, _ = _golden(tmp_path)
    g.meta["format"] = "hx-t5-golden/0"
    g.save(tmp_path / "old.npz")
    with pytest.raises(C.GoldenError, match="format"):
        C.load_golden(tmp_path / "old.npz")
    with pytest.raises(C.GoldenError, match="cannot read golden"):
        C.load_golden(tmp_path / "missing.npz")


# ---- reports -----


def test_percentiles_and_latency_row_against_the_model():
    stat = C.percentiles([10.0, 20.0, 30.0, 40.0])
    assert stat["p50"] == 25.0 and stat["mean"] == 25.0 and stat["min"] == 10.0
    assert stat["n"] == 4 and stat["max"] == 40.0 and stat["p99"] > 39.0
    with pytest.raises(ValueError):
        C.percentiles([])
    row, rec = C.latency_row(24, "graph", [161.0, 161.0])
    assert row[:2] == [24, "graph"] and rec["mean"] == 161.0
    assert "limit_us" not in rec and row[5:] == ["-", "-", "-"]


def test_table_and_tag_line_format():
    table = C.format_table(("a", "bb"), [[1, 2.25], ["xyz", "-"]])
    lines = table.splitlines()
    assert lines[0].split() == ["a", "bb"] and lines[2].split() == ["1", "2.2"]
    assert len({len(line) for line in lines}) == 1
    assert C.tag_line("verify", True, "W=3") == "T5 verify PASS W=3"
    assert C.tag_line("verify", False) == "T5 verify FAIL"
    assert C.parse_int_list("3, 24,48,") == [3, 24, 48]
    assert C.status_line("bench chain", "NOT JUDGED", "W=3") == (
        "T5 bench chain NOT JUDGED W=3"
    )


# ---- latency gates (spec 11.2): chain (b) and plan-specific layer limit (c) ----


def test_exchange_model_and_plan_specific_limits_match_the_perf_model():
    # perf_model.transport_us / t5_budget of origin/feat/hetero-moe
    for d, want in ((3, (27.31, 53.26, 82.93)), (4, (28.54, 63.15, 102.7))):
        got = [C.transport_model_us(w, d) for w in (3, 24, 48)]
        assert got == pytest.approx(want, abs=0.01)
    nmax = {3: (10.5, 41.3, 62.4), 4: (8.7, 32.4, 48.4)}
    limits = {3: (126.5, 427.3, 648.8), 4: (112.5, 363.0, 553.2)}
    for d in (3, 4):
        for w, n, want in zip((3, 24, 48), nmax[d], limits[d]):
            assert C.layer_limit_us(w, d, n) == pytest.approx(want, abs=0.6)


def test_chain_gate_with_the_measured_t5_numbers():
    real = {3: 152.8, 24: 379.0, 48: 569.5}
    synth = {3: 86.1, 24: 99.3, 48: 134.0}
    n_max = {3: 10.5, 24: 41.3, 48: 62.4}
    ratios = {3: 0.96, 24: 1.02, 48: 1.05}
    for w in real:
        j = C.judge_chain(real[w], synth[w], n_max[w])
        assert j["chain_us"] == pytest.approx(real[w] - synth[w])
        assert j["ratio"] == pytest.approx(ratios[w], abs=0.01)
        assert j["limit_us"] == pytest.approx(1.15 * n_max[w] * 6.67)
        assert j["ok"] is True
    slow = C.judge_chain(150.0, 50.0, 10.5)  # 100 us against 80.5 us
    assert slow["ok"] is False and slow["ratio"] > 1.15
    edge = C.judge_chain(C.chain_limit_us(10.0) + 10.0, 10.0, 10.0)
    assert edge["ok"] is True  # exactly at the limit passes
    assert C.chain_limit_us(0.0) == 0.0


def test_chain_text_names_the_numbers():
    text = C.chain_text(3, C.judge_chain(152.8, 86.1, 10.5))
    assert text.startswith("W=3 server chain 152.8 - 86.1 (synthetic) = 66.7 us/layer")
    assert "limit 1.15 x 10.5 experts x 6.67 us = 80.5 us" in text
    assert "ratio to the model 0.95" in text
    assert C.tag_line("bench chain", True, text).startswith("T5 bench chain PASS W=3")


def test_latency_row_carries_n_max_and_the_plan_limit():
    row, rec = C.latency_row(24, "graph", [400.0, 440.0], 41.3, 427.3)
    assert row[:2] == [24, "graph"] and row[5] == 41.3 and row[6] == 427.3
    assert rec["n_max"] == 41.3 and rec["limit_us"] == 427.3
    assert rec["ratio_limit"] == pytest.approx(420.0 / 427.3)
    assert row[7] == "0.98" and len(row) == len(C.LATENCY_HEADERS)
    row, rec = C.latency_row(2048, "eager", [900.0])
    assert row[5:] == ["-", "-", "-"] and "limit_us" not in rec


def test_busiest_server_experts_counts_distinct_experts_of_the_split(hx_env):
    hetero, stripe_doc = hx_env
    plan = hetero("plan").plan_from_doc(stripe_doc(experts=64, layers=3, top_k=4))
    tables = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    layers = C.hx_layers(tables)
    inp = C.make_inputs(1, 24, 0, nlayers=3, num_experts=64, top_k=4, hidden=8)
    counts = C.busiest_server_experts(tables, layers, inp, 0, tables.num_servers)
    assert len(counts) == 3
    for i, layer in enumerate(layers):
        _, dest = C.expected_split(tables, layer, inp.ids[i], 0)
        per_server = [
            len(set(inp.ids[i][dest == s].tolist())) for s in range(tables.num_servers)
        ]
        assert counts[i] == max(per_server) > 0
    # a server never runs more experts than it holds, nor than 4 x 24 pairs
    assert max(counts) <= 24 * 4
    # K = everything: pairs of hub-resident experts stay on the hub, fewer remain
    kept = C.busiest_server_experts(tables, layers, inp, 10_000, tables.num_servers)
    assert sum(kept) < sum(counts)
    other = C.make_inputs(1, 24, 1, nlayers=3, num_experts=64, top_k=4, hidden=8)
    mean = C.mean_busiest_experts(tables, layers, [inp, other], 0, tables.num_servers)
    both = counts + C.busiest_server_experts(tables, layers, other, 0, 4)
    assert mean == pytest.approx(sum(both) / 6)
    with pytest.raises(ValueError):
        C.mean_busiest_experts(tables, layers, [], 0, 4)


def test_n_max_of_the_t5_coldonly_plan_matches_the_spec_table(hx_env):
    hetero, _ = hx_env
    from tools.rdna.hetero_moe import t5_plan

    # the three-server plan of the first real run: perf_model.T5_NMAX[3]
    doc = t5_plan.build_doc("coldonly", 1, [2, 3, 4], "c0")
    plan = hetero("plan").plan_from_doc(doc)
    tables = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    layers = C.hx_layers(tables)
    for w, want in ((3, 10.5), (24, 41.3)):
        inputs = [C.make_inputs(1, w, p, nlayers=48) for p in range(4)]
        got = C.mean_busiest_experts(tables, layers, inputs, tables.k_for(w), 3)
        assert got == pytest.approx(want, rel=0.06)


def _bench_doc(**over):
    doc = {
        "plan_id": "abc",
        "synthetic": True,
        "hub": {
            "modes": {
                "bench": {
                    "records": [
                        {"width": 3, "mode": "eager", "mean": 500.0},
                        {"width": 3, "mode": "graph", "mean": 86.1},
                        {"width": 24, "mode": "graph", "mean": 99.3},
                        {"width": 3, "mode": "chain", "mean": 1.0},
                    ]
                }
            }
        },
    }
    doc.update(over)
    return doc


def test_load_baseline_reads_t5_run_and_hub_result_layouts(tmp_path):
    import json

    run = tmp_path / "t5_run.json"
    run.write_text(json.dumps(_bench_doc()))
    assert C.load_baseline(run, "abc") == {3: 86.1, 24: 99.3}
    hub = tmp_path / "hub_result.json"
    doc = _bench_doc()["hub"] | {"plan_id": "abc", "servers_synthetic": True}
    hub.write_text(json.dumps(doc))
    assert C.load_baseline(hub, "abc") == {3: 86.1, 24: 99.3}
    old = _bench_doc()  # an earlier run without any synthetic marker is accepted
    del old["synthetic"]
    run.write_text(json.dumps(old))
    assert C.load_baseline(run) == {3: 86.1, 24: 99.3}


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        (_bench_doc(plan_id="other"), "baseline is for plan other"),
        (_bench_doc(synthetic=False), "real servers"),
        ({"plan_id": "abc", "hub": {"modes": {}}}, "no bench records"),
        ({"plan_id": "abc", "hub": None}, "no bench records"),
        (
            {"plan_id": "abc", "hub": {"modes": {"bench": {"records": []}}}},
            "no graph-mode",
        ),
    ],
)
def test_load_baseline_refuses_unusable_files(tmp_path, doc, message):
    import json

    path = tmp_path / "b.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(C.BaselineError, match=message):
        C.load_baseline(path, "abc")
    with pytest.raises(C.BaselineError, match="cannot read baseline"):
        C.load_baseline(tmp_path / "missing.json")


# ---- child process environments (GPU pin / CPU-only guards) -----------------------


@pytest.mark.parametrize("hip", [0, 1, 2, 3, 4, 5])  # interim box: six V620s
def test_gpu_child_env_pins_one_gpu_and_drops_the_selectors(hip):
    base = {
        "PATH": "/bin",
        "HIP_VISIBLE_DEVICES": "",
        "VLLM_TARGET_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "1",
    }
    env = C.gpu_child_env(base, hip)
    assert env == {"PATH": "/bin", "HIP_VISIBLE_DEVICES": str(hip)}
    assert "VLLM_TARGET_DEVICE" in base  # the caller's mapping is untouched


@pytest.mark.parametrize("hip", [6, 9, -1, 10, "2", None, 1.0, True, ""])
def test_gpu_child_env_refuses_anything_that_is_not_one_of_the_six_gpus(hip):
    with pytest.raises(C.SetupError, match="refused"):
        C.gpu_child_env({}, hip)
    assert issubclass(C.SetupError, RuntimeError)


def test_cpu_child_env_sees_no_gpu_and_short_circuits_the_platform_probe():
    base = {
        "PATH": "/bin",
        "HIP_VISIBLE_DEVICES": "3",
        "CUDA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "1",
    }
    assert C.cpu_child_env(base) == {
        "PATH": "/bin",
        "HIP_VISIBLE_DEVICES": "",
        "VLLM_TARGET_DEVICE": "cpu",
    }
    assert base["HIP_VISIBLE_DEVICES"] == "3"
