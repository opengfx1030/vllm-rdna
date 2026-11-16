# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/rdna/eplb IO: trace readers, counter summary, solver, objective.

Pure numpy; no torch, no GPU (the placement-loader check skips without vllm).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tools.rdna.eplb import (
    itl_gate,
    loads,
    objective,
    simulate,
    solve,
    summary,
    traces,
)
from tools.rdna.hetero_moe.stats_io import LayerStats, StatsRecord

E, K, RANKS, RING = 16, 4, 4, 64
LAYERS = [(3, "model.layers.3.mlp.experts"), (5, "model.layers.5.mlp.experts")]


def make_rank_file(
    path: Path,
    rank: int,
    *,
    records: int = 40,
    hash_flip: int | None = None,
    pid: int = 100,
    seed: int = 0,
) -> dict:
    """A synthetic ``eplb-stats/1`` file; returns the T/P arrays it was built from."""
    rng = np.random.default_rng(seed)
    edges = list(summary.DEFAULT_BAND_EDGES)
    ring = np.zeros((RING, summary.WORDS), dtype=np.int32)
    t = rng.integers(2, 6, size=(records, RANKS))
    p = rng.integers(3, 12, size=(records, RANKS))
    layer_pos = np.arange(records) % 2
    rows = np.where(np.arange(records) % 7 == 6, 300, 3)  # every 7th call is prefill
    phase = (rows > edges[-1]).astype(np.int32)
    for k in range(records):
        w = ring[k % RING]
        w[summary.W_LAYER] = layer_pos[k]
        w[summary.W_ROWS] = rows[k]
        w[summary.W_TOTAL] = rows[k]
        w[summary.W_HASH] = 1000 + k
        w[summary.W_PHASE] = phase[k]
        w[summary.W_CALL] = k // 2
        w[summary.W_TS_LO] = 5000 * k
        w[summary.W_T0 : summary.W_T0 + RANKS] = t[k]
        w[summary.W_P0 : summary.W_P0 + RANKS] = p[k]
        if hash_flip is not None and k == hash_flip:
            w[summary.W_HASH] ^= 0x55
    nb = len(edges)
    meta = {
        "format": summary.FORMAT,
        "rank": rank,
        "ep_size": RANKS,
        "num_experts": E,
        "top_k": K,
        "decode_max_rows": edges[-1],
        "band_edges": edges,
        "ring": RING,
        "layers": [{"name": n, "kind": "target", "layer": i} for i, n in LAYERS],
        "pid": pid,
    }
    pairs = rng.integers(0, 50, size=(2, 2, E)).astype(np.int64)
    touch = np.zeros((2, nb, E), dtype=np.int64)
    calls = np.zeros((2, nb + 1), dtype=np.int64)
    band3 = int(summary.band_of(3, tuple(edges)))
    touch[:, band3, :] = np.arange(E)  # expert e touched in e of 20 steps
    calls[:, band3] = 20
    calls[:, -1] = 5
    np.savez(
        path,
        meta=np.array(json.dumps(meta)),
        pairs=pairs,
        touch=touch,
        calls=calls,
        rows=np.array([[60, 1500], [60, 1500]], dtype=np.int64),
        ring=ring,
        ring_count=np.int64(records),
    )
    return {"t": t, "p": p, "layer_pos": layer_pos, "rows": rows, "phase": phase}


def edit_ring(path: Path, edits: dict[int, dict[int, int]]) -> None:
    """Add ``edits[record][word]`` to a rank file's ring (a rank that diverges)."""
    with np.load(path) as z:
        arrays = {name: z[name] for name in z.files}
    for k, words in edits.items():
        for w, delta in words.items():
            arrays["ring"][k % RING, w] += delta
    np.savez(path, **arrays)


# ---- summary -----------------------------------------------------------------------


def test_summary_ratios_match_the_ring(tmp_path):
    built = make_rank_file(tmp_path / "eplb_stats-rank0-pid100.npz", 0)
    rs = summary.load_dir(str(tmp_path))[0]
    rep = summary.summarize([rs], window=4)
    layer0 = rep["layers"][0]
    sel = (built["layer_pos"] == 0) & (built["phase"] == 0)
    band = layer0["decode_bands"]["<=3"]
    assert band["steps"] == int(sel.sum())
    assert band["T_max_over_mean"] == pytest.approx(
        loads.max_over_mean(built["t"][sel])
    )
    assert band["P_max_over_mean"] == pytest.approx(
        loads.max_over_mean(built["p"][sel])
    )
    pre = (built["layer_pos"] == 0) & (built["phase"] == 1)
    assert layer0["prefill"]["calls"] == int(pre.sum())
    assert layer0["prefill"]["P_max_over_mean"] == pytest.approx(
        loads.max_over_mean(built["p"][pre])
    )
    assert layer0["windows"]["decode_T"]["windows"] == int(sel.sum()) // 4
    assert "within_target" in layer0["windows"]["decode_T"]
    assert "T max/mean" in summary.format_table(rep)


def test_ring_wrap_is_read_oldest_first(tmp_path):
    make_rank_file(tmp_path / "eplb_stats-rank0-pid100.npz", 0, records=100)
    rs = summary.load_rank(str(tmp_path / "eplb_stats-rank0-pid100.npz"))
    assert rs.ring_total == 100 and rs.first_record == 100 - RING
    assert rs.ring.shape[0] == RING
    tab = summary.ring_table(rs)
    assert tab["k"][0] == 36 and tab["k"][-1] == 99
    assert tab["hash"][0] == 1036 and tab["hash"][-1] == 1099  # chronological
    assert (np.diff(tab["ts"].astype(np.int64)) == 5000).all()


def test_consensus_agrees_and_flags_a_diverging_rank(tmp_path):
    for r in range(2):
        make_rank_file(
            tmp_path / f"eplb_stats-rank{r}-pid{100 + r}.npz", r, pid=100 + r
        )
    assert summary.consensus(summary.load_dir(str(tmp_path))) == {
        "ranks": 2,
        "compared": 40,
        "mismatches": 0,
        "first_mismatch_record": None,
        "ok": True,
    }
    make_rank_file(tmp_path / "eplb_stats-rank1-pid101.npz", 1, pid=101, hash_flip=17)
    bad = summary.consensus(summary.load_dir(str(tmp_path)))
    assert (
        bad["ok"] is False
        and bad["mismatches"] == 1
        and bad["first_mismatch_record"] == 17
    )
    assert summary.main([str(tmp_path)]) == 2
    assert bad["detail"] == {
        "odd_rank": None,  # two ranks have no majority
        "phase": {"decode": 1},
        "rows": {"3": 1},
        "loads_differ": 0,
        "hash_only": 1,
        "passes": 1,
    }


def test_consensus_detail_names_the_odd_rank_and_separates_load_changes(tmp_path):
    for r in range(4):
        make_rank_file(
            tmp_path / f"eplb_stats-rank{r}-pid{100 + r}.npz", r, pid=100 + r
        )
    assert summary.consensus(summary.load_dir(str(tmp_path)))["ok"] is True
    # rank 2 disagrees on three prefill calls (records 6, 13, 34); on record 13 one
    # routed pair also moves from owner 1 to owner 0
    edit_ring(
        tmp_path / "eplb_stats-rank2-pid102.npz",
        {
            6: {summary.W_HASH: 0x55},
            13: {summary.W_HASH: 0x55, summary.W_P0: 1, summary.W_P0 + 1: -1},
            34: {summary.W_HASH: 7},
        },
    )
    bad = summary.consensus(summary.load_dir(str(tmp_path)))
    assert bad["mismatches"] == 3 and bad["first_mismatch_record"] == 6
    assert bad["detail"] == {
        "odd_rank": {"2": 3},
        "phase": {"prefill": 3},
        "rows": {"300": 3},
        "loads_differ": 1,
        "hash_only": 2,
        "passes": 3,
    }


def test_touch_weight_is_the_band_frequency_and_the_stats_export_is_consistent(
    tmp_path,
):
    make_rank_file(tmp_path / "eplb_stats-rank0-pid100.npz", 0)
    rs = summary.load_dir(str(tmp_path))[0]
    w, n = summary.touch_weight(rs, 0, 3)
    assert n == 20 and np.allclose(w, np.arange(E) / 20.0)
    w6, n6 = summary.touch_weight(rs, 0, 6)  # no steps of that width were seen
    assert n6 == 0 and not w6.any()
    rec = summary.to_stats_record(rs)
    assert rec.num_experts == E and rec.top_k == K
    assert sorted(rec.layers) == [3, 5]
    assert rec.layers[3].decode_tokens == 60 and rec.layers[3].prefill_tokens == 1500
    assert rec.steps["prefill"] == 5
    assert summary.topn_share(np.array([5, 3, 1, 1]), (1, 2)) == {"1": 0.5, "2": 0.8}


def test_wrong_format_is_rejected(tmp_path):
    np.savez(
        tmp_path / "eplb_stats-rank0-pid1.npz",
        meta=np.array(json.dumps({"format": "x"})),
    )
    with pytest.raises(ValueError, match="not an eplb-stats/1 file"):
        summary.load_dir(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        summary.load_dir(str(tmp_path / "empty"))


# ---- traces -----------------------------------------------------------------------


def test_id_trace_reads_the_ring_oldest_first_and_ignores_stale_rows(tmp_path):
    n_slots, r, k = 4, 3, 2
    ids = np.full((1, n_slots, r, k), -1, dtype=np.int16)
    rows = np.zeros((1, n_slots), dtype=np.int32)
    for call in range(6):  # 6 calls into 4 slots: the newest 4 are valid
        slot = call % n_slots
        n = 3 if call % 2 == 0 else 2
        ids[0, slot, :n] = (call, call + 1)  # same pair for every row of the call
        rows[0, slot] = n
    np.savez(
        tmp_path / "t.trace.npz",
        names=np.array(["model.layers.7.mlp.experts"]),
        ids=ids,
        rows=rows,
        calls=np.array([6]),
        max_rows=r,
    )
    out = traces.id_trace(str(tmp_path / "t.trace.npz"), num_experts=8)
    sc = out[7]
    assert sc.steps == 4 and sc.experts == 8
    assert sc.rows.tolist() == [3, 2, 3, 2]  # calls 2, 3, 4, 5
    assert sc.counts[0].tolist() == [0, 0, 3, 3, 0, 0, 0, 0]
    assert sc.counts[3].tolist() == [0, 0, 0, 0, 0, 2, 2, 0]


def test_hist_jsonl_tolerates_a_cut_last_line_and_filters_decode(tmp_path):
    lines = [
        {"meta": {"num_experts": 4, "ep_size": 2}},
        {
            "seq": 1,
            "step": 0,
            "layer": "a",
            "layer_idx": 3,
            "tokens": 600,
            "counts": [1, 2, 3, 4],
        },
        {
            "seq": 2,
            "step": 0,
            "layer": "a",
            "layer_idx": 3,
            "tokens": 8,
            "counts": [9, 9, 9, 9],
        },
        {
            "seq": 3,
            "step": 1,
            "layer": "a",
            "layer_idx": 3,
            "tokens": 700,
            "counts": [4, 3, 2, 1],
        },
    ]
    text = "\n".join(json.dumps(x) for x in lines) + '\n{"seq": 4, "ste'
    (tmp_path / "h.jsonl").write_text(text)
    meta, out = traces.hist_jsonl(str(tmp_path / "h.jsonl"), min_tokens=512)
    assert meta["ep_size"] == 2
    assert out[3].counts.tolist() == [[1, 2, 3, 4], [4, 3, 2, 1]]
    assert out[3].rows.tolist() == [600, 700]
    _, everything = traces.hist_jsonl(str(tmp_path / "h.jsonl"))
    assert everything[3].steps == 3


def test_replay_cli_scores_a_real_trace_out_of_fold(tmp_path, capsys):
    # every row routes to experts 0 and 1: both on rank 0 of the linear placement
    n_slots, r, k = 40, 4, 2
    ids = np.zeros((1, n_slots, r, k), dtype=np.int16)
    ids[..., 1] = 1
    ids[0, ::2, :, :] = (2, 3)  # even calls use the experts of the next block
    np.savez(
        tmp_path / "t.trace.npz",
        names=np.array(["model.layers.2.mlp.experts"]),
        ids=ids,
        rows=np.full((1, n_slots), r, dtype=np.int32),
        calls=np.array([n_slots]),
        max_rows=r,
    )
    out = tmp_path / "r.json"
    rc = simulate.main(
        [
            "replay",
            "--id-trace",
            str(tmp_path / "t.trace.npz"),
            "--num-experts",
            "8",
            "--ep",
            "2",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    rep = json.loads(out.read_text())["sources"]["decode (id trace)"]["2"]
    assert rep["steps"] == 40
    assert rep["pack"]["T"] < rep["linear"]["T"]
    assert "decode (id trace)" in capsys.readouterr().out
    assert simulate.main(["replay"]) == 2  # no source given


def test_split_steps_uses_contiguous_blocks():
    sc = traces.StepCounts(
        0, np.arange(40).reshape(10, 4).astype(np.int32), np.ones(10, np.int32)
    )
    train, test = traces.split_steps(sc, folds=2, fold=1)
    assert test[:, 0].tolist() == [20, 24, 28, 32, 36]
    assert train[:, 0].tolist() == [0, 4, 8, 12, 16]


def test_eplb_stats_hist_reorders_the_ring(tmp_path):
    make_rank_file(tmp_path / "f.npz", 0)
    with np.load(tmp_path / "f.npz") as z:
        data = {k: z[k] for k in z.files}
    hist = np.zeros((2, 4, E), dtype=np.uint16)
    for call in range(6):  # 6 calls into 4 slots
        hist[0, call % 4] = call
    data.update(
        hist=hist, hist_calls=np.array([6, 0]), hist_rows=np.tile(np.arange(4), (2, 1))
    )
    np.savez(tmp_path / "g.npz", **data)
    out = traces.eplb_stats_hist(str(tmp_path / "g.npz"))
    assert list(out) == [3]  # layer 5 has no calls
    assert out[3].counts[:, 0].tolist() == [2, 3, 4, 5]
    assert traces.eplb_stats_hist(str(tmp_path / "f.npz")) == {}


# ---- solve ----------------------------------------------------------------------


def _record(decode: np.ndarray, prefill: np.ndarray | None = None, top_k: int = K):
    layers = {}
    for li in range(decode.shape[0]):
        d = np.rint(decode[li] * 1e6).astype(np.int64)
        p = d if prefill is None else np.rint(prefill[li] * 1e6).astype(np.int64)
        layers[li] = LayerStats(
            li, "target", None, d.astype(float), p.astype(float), d, p, 1e6, 1e6
        )
    return StatsRecord(decode.shape[1], top_k, layers)


def test_solve_packs_skewed_layers_and_leaves_balanced_ones_linear():
    skew = simulate.zipf_record(layers=3, experts=64, top_k=4, alpha=1.0, seed=1)
    doc, report = solve.solve(skew, None, 4, objective="touch", rows=24)
    assert solve.validate_doc(doc) == []
    assert doc["version"] == 1 and doc["ep_size"] == 4 and doc["num_experts"] == 64
    assert set(doc["layers"]) <= {"0", "1", "2"} and doc["layers"]
    for key, entry in doc["layers"].items():
        assert (
            report[key]["objective_ratio_packed"]
            < report[key]["objective_ratio_linear"]
        )
    flat = _record(np.full((2, 64), 4 / 64))
    doc2, report2 = solve.solve(flat, None, 4)
    assert doc2["layers"] == {} and all(v["kept_linear"] for v in report2.values())


def test_solve_is_deterministic_and_the_gate_uses_the_held_out_capture():
    a = simulate.zipf_record(layers=2, experts=64, top_k=4, alpha=1.0, seed=1)
    b = simulate.zipf_record(layers=2, experts=64, top_k=4, alpha=1.0, seed=9)
    d1, r1 = solve.solve(a, b, 4)
    d2, r2 = solve.solve(a, b, 4)
    assert d1 == d2 and r1 == r2
    ins, _ = solve.solve(a, None, 4)
    oos_gain = sum(
        v["objective_ratio_linear"] - v["objective_ratio_packed"] for v in r1.values()
    )
    ins_gain = sum(
        v["objective_ratio_linear"] - v["objective_ratio_packed"]
        for v in solve.solve(a, None, 4)[1].values()
    )
    assert ins_gain > oos_gain and ins["layers"]


def test_solve_blend_and_measured_touch_paths():
    dec = (
        simulate.zipf_record(layers=1, experts=64, top_k=4, alpha=1.0, seed=2)
        .layers[0]
        .decode
    )
    pre = np.roll(dec, 17)  # prefill popularity differs from decode
    rec = _record(np.stack([dec / dec.sum() * 4]), np.stack([pre / pre.sum() * 4]))
    blend, rep = solve.solve(rec, None, 4, objective="blend", prefill_share=0.5)
    assert (
        solve.validate_doc(blend) == []
        and rep["0"]["prefill_ratio_packed"] < rep["0"]["prefill_ratio_linear"]
    )
    touch = np.linspace(0.0, 1.0, 64)
    measured, _ = solve.solve(rec, None, 4, touch={0: touch})
    model, _ = solve.solve(rec, None, 4)
    assert solve.validate_doc(measured) == [] and measured["layers"] != model["layers"]
    with pytest.raises(ValueError):
        solve.solve(rec, None, 4, objective="x")


def test_validate_doc_flags_non_permutations():
    doc = {
        "version": 1,
        "num_experts": 8,
        "ep_size": 2,
        "layers": {"0": [[0, 1, 2, 3], [4, 5, 6, 6]]},
    }
    assert solve.validate_doc(doc) == ["layer 0: not a permutation of 0..7"]


def test_solve_cli_writes_a_loadable_file(tmp_path):
    rec = simulate.zipf_record(layers=2, experts=64, top_k=4, alpha=1.0, seed=3)
    from tools.rdna.hetero_moe.stats_io import write_json

    write_json(rec, str(tmp_path / "s.json"))
    out = tmp_path / "p.json"
    assert (
        solve.main(
            ["--train", str(tmp_path / "s.json"), "--ep", "4", "--out", str(out)]
        )
        == 0
    )
    doc = json.loads(out.read_text())
    assert solve.validate_doc(doc) == []
    loader = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.rdna_moe_placement"
    )
    parsed = loader.load_placement(str(out))
    for key, per_rank in doc["layers"].items():
        for r in range(4):
            got = loader.local_experts_for_layer(
                parsed, f"model.layers.{key}.mlp.experts", r, 64
            )
            assert got == per_rank[r]


# ---- objective ------------------------------------------------------------------


def _synthetic_ring(
    tmp_path: Path, calls: int, y_of
) -> tuple[summary.RankStats, np.ndarray]:
    rng = np.random.default_rng(7)
    ring = np.zeros((calls, summary.WORDS), dtype=np.int32)
    t = rng.integers(20, 40, size=(calls, RANKS))
    p = rng.integers(30, 80, size=(calls, RANKS))  # independent of t on purpose
    ring[:, summary.W_ROWS] = 24
    ring[:, summary.W_LAYER] = np.arange(calls) % 2
    ring[:, summary.W_T0 : summary.W_T0 + RANKS] = t
    ring[:, summary.W_P0 : summary.W_P0 + RANKS] = p
    meta = {
        "rank": 1,
        "ep_size": RANKS,
        "band_edges": list(summary.DEFAULT_BAND_EDGES),
        "layers": [],
    }
    rs = summary.RankStats("x", meta, None, None, None, None, ring, calls)  # type: ignore[arg-type]
    return rs, y_of(t[:, 1], p[:, 1])


def _kernels_for(y_us: np.ndarray) -> np.ndarray:
    """Two back-to-back dispatches per call that add up to ``y_us``."""
    start = np.cumsum(np.full(y_us.shape[0], 50_000))  # 50 us gaps
    half = (y_us * 500).astype(np.int64)  # ns
    k = np.zeros((2 * y_us.shape[0], 2), dtype=np.int64)
    k[0::2, 0], k[0::2, 1] = start, start + half
    k[1::2, 0], k[1::2, 1] = start + half, start + 2 * half
    return k


def test_objective_fit_finds_the_load_the_time_follows(tmp_path):
    rs_t, y_t = _synthetic_ring(tmp_path, 400, lambda t, p: 5.0 + 6.7 * t)
    fit_t = objective.fit(
        objective.join(rs_t, objective.call_times_us(_kernels_for(y_t), 2))
    )
    assert fit_t["follows"] == "T"
    assert fit_t["us_per_touched_expert_T_only"] == pytest.approx(6.7, rel=0.01)
    assert fit_t["r2_T"] > 0.99 and not fit_t["ill_conditioned"]
    rs_p, y_p = _synthetic_ring(tmp_path, 400, lambda t, p: 2.0 + 0.68 * p)
    fit_p = objective.fit(
        objective.join(rs_p, objective.call_times_us(_kernels_for(y_p), 2))
    )
    assert fit_p["follows"] == "P"
    assert fit_p["us_per_pair_P_only"] == pytest.approx(0.68, rel=0.01)


def test_objective_join_refuses_to_guess(tmp_path):
    rs, y = _synthetic_ring(tmp_path, 100, lambda t, p: 5.0 + 6.7 * t)
    times = objective.call_times_us(_kernels_for(y), 2)
    with pytest.raises(objective.JoinError, match="100 ring records"):
        objective.join(rs, times[:-1])
    with pytest.raises(objective.JoinError, match="not a multiple of 2"):
        objective.call_times_us(_kernels_for(y)[:-1], 2)
    wrapped = summary.RankStats("x", rs.meta, None, None, None, None, rs.ring, 500)  # type: ignore[arg-type]
    with pytest.raises(objective.JoinError, match="ring wrapped"):
        objective.join(wrapped, times)
    assert objective.fit(objective.join(rs, times), rows=99)["error"] == "too few calls"


def test_read_kernels_keeps_only_the_moe_gemms(tmp_path):
    csv_text = "\n".join(
        [
            "Kernel_Name,Start_Timestamp,End_Timestamp",
            "void moe_gptq_gemm_rdna2<half>(...),200,260",
            "gemm_other,100,150",
            "moe_w4a8_gemm_rdna2_kernel,50,90",
        ]
    )
    (tmp_path / "123_kernel_trace.csv").write_text(csv_text)
    k = objective.read_kernels(str(tmp_path / "123_kernel_trace.csv"))
    assert k.tolist() == [[50, 90], [200, 260]]
    assert objective._pid(str(tmp_path / "123_kernel_trace.csv")) == 123


# ---- ITL gate ----------------------------------------------------------------


def _bench_dir(path: Path, cells: dict[str, tuple[float, float]], invalid=()) -> Path:
    """A serve_validate-style arm directory: cell name -> (ITL p50, ITL p99)."""
    path.mkdir(parents=True)
    for name, (p50, p99) in cells.items():
        doc = {
            "median_itl_ms": p50,
            "p99_itl_ms": p99,
            "median_ttft_ms": 900.0,
            "output_throughput": 100.0,
        }
        (path / f"bench-{name}.json").write_text(json.dumps(doc))
    if invalid:
        lines = [f"cell bench-{n} INVALID (KFD evictions)" for n in invalid]
        (path / "summary.txt").write_text("\n".join(lines))
    return path


def test_itl_gate_passes_inside_the_noise_and_fails_outside(tmp_path):
    cells = {"1024x512-c1": (22.40, 30.0), "1024x512-c8": (44.0, 60.0)}
    a1 = _bench_dir(tmp_path / "A1", cells)
    a2 = _bench_dir(
        tmp_path / "A2", {"1024x512-c1": (22.60, 30.4), "1024x512-c8": (44.2, 60.3)}
    )
    inside = {"1024x512-c1": (22.55, 30.1), "1024x512-c8": (44.1, 60.1)}
    b1 = _bench_dir(tmp_path / "B1", inside)
    b2 = _bench_dir(tmp_path / "B2", inside)
    rows = itl_gate.compare(
        [itl_gate.load_cells(str(d)) for d in (a1, a2)],
        [itl_gate.load_cells(str(d)) for d in (b1, b2)],
    )
    gated = [r for r in rows if r["metric"] in itl_gate.GATED]
    assert len(gated) == 4 and all(r["verdict"] == "pass" for r in gated)
    c1 = next(
        r for r in gated if r["cell"] == "1024x512-c1" and "median" in r["metric"]
    )
    assert c1["off"] == pytest.approx(22.5) and c1["noise"] == pytest.approx(0.2 / 22.5)
    slow = _bench_dir(
        tmp_path / "S", {"1024x512-c1": (24.0, 33.0), "1024x512-c8": (44.1, 60.1)}
    )
    bad = itl_gate.compare(
        [itl_gate.load_cells(str(d)) for d in (a1, a2)],
        [itl_gate.load_cells(str(slow))],
    )
    verdicts = {(r["cell"], r["metric"]): r["verdict"] for r in bad}
    assert verdicts[("1024x512-c1", "median_itl_ms")] == "FAIL"
    assert verdicts[("1024x512-c1", "p99_itl_ms")] == "FAIL"
    assert verdicts[("1024x512-c8", "median_itl_ms")] == "pass"
    assert itl_gate.main(["--off", str(a1), str(a2), "--on", str(slow)]) == 1
    assert itl_gate.main(["--off", str(a1), str(a2), "--on", str(b1), str(b2)]) == 0


def test_itl_gate_has_a_floor_unknown_verdicts_and_refuses_invalid_cells(tmp_path):
    cells = {"1024x512-c1": (22.5, 30.0)}
    a1 = _bench_dir(tmp_path / "A1", cells)
    a2 = _bench_dir(tmp_path / "A2", cells)  # identical: zero noise
    near = _bench_dir(tmp_path / "N", {"1024x512-c1": (22.5 * 1.004, 30.0 * 1.004)})
    rows = itl_gate.compare(
        [itl_gate.load_cells(str(d)) for d in (a1, a2)],
        [itl_gate.load_cells(str(near))],
    )
    assert all(r["verdict"] == "pass" for r in rows if r["metric"] in itl_gate.GATED)
    one = itl_gate.compare(
        [itl_gate.load_cells(str(a1))], [itl_gate.load_cells(str(near))]
    )
    assert {r["verdict"] for r in one if r["metric"] in itl_gate.GATED} == {"unknown"}
    dirty = _bench_dir(tmp_path / "D", cells, invalid=["1024x512-c1"])
    assert itl_gate.load_cells(str(dirty)) == {}
    assert itl_gate.main(["--off", str(a1), "--on", str(dirty)]) == 3
