# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/rdna/hetero_moe (stats IO, solver, simulator, validator).

Pure numpy; no torch, no GPU. Budgets are expressed in "expert slots" of 1000
bytes so the arithmetic is readable.
"""

import argparse
import copy
import dataclasses
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.rdna.hetero_moe import (
    model_bytes,
    simulate,
    solve,
    stats_io,
    synth,
    validate,
)
from tools.rdna.hetero_moe.placement import (
    SolverConfig,
    Topology,
    fallback_plan,
    plan_document,
    step_shape,
)
from tools.rdna.hetero_moe.placement import solve as solve_placement

ROOT = Path(__file__).resolve().parents[3]
E, L, K, D = 64, 6, 8, 4
SLOT = 1000


def _topo(n_hubs: int = 1) -> Topology:
    return Topology(islands=2, n_hubs=n_hubs, n_experts=D, switches=["a", "b"])


def _cfg(hub_slots: int, spare: int = 0, layers: int = L, **kw) -> SolverConfig:
    """Budgets in slots; ``spare`` = replica slots per expert device."""
    kw.setdefault("max_hub_per_layer", E)
    return SolverConfig(
        expert_bytes=SLOT,
        hub_budget_bytes=hub_slots * SLOT,
        expert_budget_bytes=((E // D) * layers + spare) * SLOT,
        **kw,
    )


def _record(tmp_path, name="s", **kw) -> stats_io.StatsRecord:
    params = dict(
        kind="zipf",
        layers=L,
        experts=E,
        top_k=K,
        alpha=1.1,
        decode_rows=3000,
        prefill_rows=2000,
        seed=1,
    )
    params.update(kw)
    data = synth.generate(**params)
    path = synth.write_raw(data, str(tmp_path / name))
    return stats_io.load_stats([path])


@pytest.fixture(scope="module")
def zipf(tmp_path_factory):
    return _record(tmp_path_factory.mktemp("zipf"))


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    return _record(tmp_path_factory.mktemp("uni"), kind="uniform")


def _solve(record, cfg, topo=None):
    topo = topo or _topo()
    placements, models, hub = solve_placement(record, topo, cfg)
    model = {"num_layers": L, "num_experts": E, "top_k": K}
    doc = plan_document(placements, topo, cfg, model, {}, "stats", [])
    return placements, doc, topo


def _arrays(placements, topo):
    return simulate.arrays_from_placements(placements, topo, E)


# ---- stats IO -----------------------------------------------------------------


def test_raw_to_json_roundtrip_and_counting_rule(tmp_path, zipf):
    path = tmp_path / "stats.json"
    stats_io.write_json(zipf, str(path))
    again = stats_io.read_json(str(path))
    assert again.num_experts == E and again.top_k == K
    assert sorted(again.layers) == list(range(L))
    for li, ls in zipf.layers.items():
        assert np.array_equal(again.layers[li].decode_raw, ls.decode_raw)
        # counts[e] = rows whose top-k set contains e: decode sums to top_k * rows
        assert ls.decode_raw.sum() == K * round(ls.decode_tokens)
    doc = json.loads(path.read_text())
    assert doc["format"] == "hetero-moe-expert-stats/1"
    assert doc["phases"]["decode"]["steps"] > 0
    assert sum(doc["step_hist"]["tokens_per_step"].values()) == pytest.approx(
        zipf.steps["decode"], rel=0.02
    )


def test_window_is_difference_of_cumulative_snapshots(tmp_path):
    data = synth.generate(layers=2, experts=E, top_k=K, decode_rows=2000, seed=4)
    path = synth.write_raw(data, str(tmp_path), snapshots=2)
    whole = stats_io.load_stats([path])
    second = stats_io.load_stats([path], t0=30.0)  # (snapshot 1, snapshot 2]
    first = stats_io.load_stats([path], t1=30.0)  # (0, snapshot 1]
    for li in whole.layers:
        assert np.array_equal(
            first.layers[li].decode_raw + second.layers[li].decode_raw,
            whole.layers[li].decode_raw,
        )
    assert second.layers[0].decode_raw.sum() > 0


def test_merge_is_elementwise_sum_with_weights(tmp_path):
    a = synth.write_raw(
        synth.generate(layers=2, experts=E, top_k=K, decode_rows=500, seed=1),
        str(tmp_path / "a"),
    )
    b = synth.write_raw(
        synth.generate(layers=2, experts=E, top_k=K, decode_rows=500, seed=2),
        str(tmp_path / "b"),
    )
    ra, rb = stats_io.load_stats([a]), stats_io.load_stats([b])
    merged = stats_io.load_stats([a, f"{b}:0.5"])
    for li in ra.layers:
        assert np.array_equal(
            merged.layers[li].decode_raw,
            ra.layers[li].decode_raw + rb.layers[li].decode_raw,
        )
        assert np.allclose(
            merged.layers[li].decode, ra.layers[li].decode + 0.5 * rb.layers[li].decode
        )
    doc = stats_io.to_json(merged)
    assert "weight" in doc["layers"][0]["decode"]
    assert len(merged.sources) == 2


def test_truncated_last_line_is_tolerated(tmp_path):
    data = synth.generate(layers=2, experts=E, top_k=K, decode_rows=300, seed=1)
    path = synth.write_raw(data, str(tmp_path), snapshots=2)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"type":"snapshot","seq":3,"time"')
    assert stats_io.load_stats([path]).layers[0].decode_raw.sum() > 0


def test_mtp_layer_gets_index_after_the_target_layers(tmp_path):
    path = tmp_path / "x.jsonl"
    names = [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
        "mtp.layers.0.mlp.experts",
    ]
    kinds = ["target", "target", "mtp"]
    meta = {
        "type": "meta",
        "schema": "expert-stats/1",
        "num_experts": 4,
        "top_k": 2,
        "decode_max_tokens": 4,
        "time": 1.0,
        "layers": [
            {"name": n, "idx": i % 2 if k == "target" else 0, "kind": k}
            for i, (n, k) in enumerate(zip(names, kinds, strict=True))
        ],
    }
    snap = {
        "type": "snapshot",
        "seq": 1,
        "time": 2.0,
        "elapsed_s": 1.0,
        "counts": {n: {"decode": [1, 1, 1, 1], "prefill": [0, 0, 0, 0]} for n in names},
        "tokens": {n: {"decode": 2, "prefill": 0} for n in names},
        "steps": {n: {"decode": 2, "prefill": 0} for n in names},
        "rows_hist": {n: [0, 0, 2, 0, 0] for n in names},
        "invalid": {n: 0 for n in names},
    }
    path.write_text(json.dumps(meta) + "\n" + json.dumps(snap) + "\n")
    record = stats_io.load_stats([str(path)])
    assert sorted(record.layers) == [0, 1, 2]
    assert [record.layers[i].kind for i in range(3)] == ["target", "target", "mtp"]
    assert record.model["num_layers"] == 2 and record.model["mtp_layers"] == 1
    assert record.rows_hist == {2: 2.0}


def test_pairs_file_becomes_a_coact_sidecar(tmp_path):
    data = synth.generate(
        layers=2, experts=32, top_k=4, decode_rows=400, pairs=True, clusters=3, seed=2
    )
    path = synth.write_raw(data, str(tmp_path / "p"))
    record = stats_io.load_stats([path])
    assert record.coact is not None and record.coact.shape == (2, 32, 32)
    out = tmp_path / "stats.json"
    stats_io.write_json(record, str(out))
    back = stats_io.read_json(str(out))
    assert back.coact is not None
    assert np.array_equal(back.coact, np.rint(record.coact).astype(np.uint32))
    assert json.loads(out.read_text())["coact"]["shape"] == [2, 32, 32]


def test_skew_summary_is_monotone_and_uniform_is_flat(zipf, uniform):
    skew = stats_io.skew_summary(zipf, (8, 16, 32))
    assert skew["mean"]["8"] < skew["mean"]["16"] < skew["mean"]["32"] <= 1.0
    flat = stats_io.skew_summary(uniform, (8, 32))
    assert flat["mean"]["8"] == pytest.approx(8 / E, abs=0.03)
    assert skew["mean"]["8"] > 2 * flat["mean"]["8"]


# ---- solver: objective, budgets, determinism -----------------------------------


def test_zipf_hub_beats_stripe_and_uniform_gains_nothing(zipf, uniform):
    slots = L * 16  # a quarter of every layer on average
    z_pl, _, topo = _solve(zipf, _cfg(slots))
    shape = step_shape(_cfg(slots), zipf)
    z_hit = simulate.analytic_metrics(_arrays(z_pl, topo), zipf, shape)[
        "decode_hub_hit"
    ]
    base = simulate.analytic_metrics(
        simulate.fallback_arrays(zipf, topo, _cfg(slots)), zipf, shape
    )["decode_hub_hit"]
    assert base == pytest.approx(0.25, abs=0.05)  # e % 4 == 0 stripe is blind
    assert z_hit > 2.0 * base
    # global water-filling takes at least what an even split per layer would
    per_layer = sum(
        np.sort(zipf.layers[li].decode / zipf.layers[li].decode_tokens)[::-1][:16].sum()
        for li in zipf.layers
    ) / (L * K)
    assert z_hit >= per_layer - 1e-9
    u_pl, _, _ = _solve(uniform, _cfg(slots))
    u_hit = simulate.analytic_metrics(_arrays(u_pl, topo), uniform, shape)[
        "decode_hub_hit"
    ]
    assert u_hit == pytest.approx(0.25, abs=0.04)


def test_global_budget_gives_skewed_layers_more_slots(tmp_path):
    data = synth.generate(
        layers=2, experts=E, top_k=K, alpha=1.6, alpha_jitter=0.0, seed=1
    )
    flat = synth.generate(layers=2, experts=E, top_k=K, kind="uniform", seed=1)
    # layer 0 skewed (zipf), layer 1 flat: splice the counts of two captures
    data.decode_counts[1] = flat.decode_counts[1]
    data.prefill_counts[1] = flat.prefill_counts[1]
    record = stats_io.load_stats([synth.write_raw(data, str(tmp_path))])
    cfg = _cfg(24, layers=2)
    placements, _, _ = solve_placement(record, _topo(), cfg)
    n0, n1 = (len(placements[i].hub[0]) for i in (0, 1))
    assert n0 + n1 == 24 and n0 > n1


def test_min_and_max_hub_per_layer_clamp(zipf):
    pl, _, _ = _solve(zipf, _cfg(L * 40, max_hub_per_layer=10))
    assert {len(p.hub[0]) for p in pl.values()} == {10}  # cap binds, budget unused
    pl, _, _ = _solve(zipf, _cfg(L * 8, min_hub_per_layer=5))
    assert min(len(p.hub[0]) for p in pl.values()) >= 5
    assert sum(len(p.hub[0]) for p in pl.values()) == L * 8
    with pytest.raises(ValueError, match="min_hub_per_layer"):
        solve_placement(zipf, _topo(), _cfg(L * 3, min_hub_per_layer=5))


def test_hub_fits_all_and_hub_fits_none(zipf):
    pl, doc, topo = _solve(zipf, _cfg(L * E))
    arr = _arrays(pl, topo)
    assert arr.hub.all()
    shape = step_shape(_cfg(0), zipf)
    m = simulate.analytic_metrics(arr, zipf, shape)
    assert (
        m["decode_hub_hit"] == pytest.approx(1.0)
        and m["cold_touched_critical_per_step"] == 0
    )
    assert validate.validate_plan(doc, zipf) == []
    # V620s still hold every expert (cold_complete): the hub is a cache
    assert all(any(d >= 1 for d in row) for row in doc["layers"][0]["owners"])

    pl, doc, topo = _solve(zipf, _cfg(0))
    arr = _arrays(pl, topo)
    assert not arr.hub.any()
    assert simulate.analytic_metrics(arr, zipf, shape)["decode_hub_hit"] == 0.0
    assert validate.validate_plan(doc, zipf) == []
    pl, _, _ = _solve(
        zipf,
        _cfg(0).__class__(
            **{**dataclasses.asdict(_cfg(0)), "hub_budget_bytes": SLOT - 1}
        ),
    )
    assert not any(p.hub[0] for p in pl.values())  # less than one expert fits


def test_solver_is_deterministic(tmp_path, zipf):
    cfg = _cfg(L * 12, spare=30)
    _, doc_a, _ = _solve(zipf, cfg)
    _, doc_b, _ = _solve(zipf, cfg)
    assert json.dumps(doc_a, sort_keys=True) == json.dumps(doc_b, sort_keys=True)
    assert doc_a["plan_id"] == doc_b["plan_id"]
    # the same statistics through the converted JSON give the same plan
    path = tmp_path / "stats.json"
    stats_io.write_json(zipf, str(path))
    _, doc_c, _ = _solve(stats_io.read_json(str(path)), cfg)
    assert doc_c["plan_id"] == doc_a["plan_id"]


def test_every_expert_placed_once_per_island_plus_replicas(zipf):
    cfg = _cfg(L * 12, spare=25)
    pl, doc, topo = _solve(zipf, cfg)
    assert validate.validate_plan(doc, zipf) == []
    assert doc["cold_complete"] and doc["islands"] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    for layer in doc["layers"]:
        primaries = np.zeros(D, dtype=int)
        for row in layer["owners"]:
            cold = [d for d in row if d >= 1]
            assert cold and len(set(cold)) == len(cold)  # >= 1 V620 copy, distinct
            primaries[cold[0] - 1] += 1
            hubs = [d for d in row if d < 1]
            assert row == hubs + cold and cold[1:] == sorted(cold[1:])
        assert primaries.tolist() == [E // D] * D  # exactly E/D primaries per device
    # island 1 maps by position: devices 5..9 mirror 0..4
    devices = doc["devices"]
    assert [devices[i]["role"] for i in doc["islands"][0]] == [
        devices[i]["role"] for i in doc["islands"][1]
    ]
    assert devices[0]["hip"] == 0 and devices[1]["hip"] == 1 and devices[5]["hip"] == 5


def test_replicas_use_the_budget_hottest_cold_first(zipf):
    spare = 20
    pl, doc, topo = _solve(zipf, _cfg(L * 12, spare=spare))
    n_rep = sum(sum(len(v) for v in p.replicas.values()) for p in pl.values())
    assert n_rep == D * spare  # every spare slot is used
    assert validate.validate_plan(doc, zipf) == []
    # replicas are only for experts that are not on the hub
    for p in pl.values():
        assert not set(p.replicas) & set(p.hub[0])
    # and the hottest cold expert of the plan got copied first
    hottest = max(
        ((li, e) for li, p in pl.items() for e in range(E) if e not in p.hub[0]),
        key=lambda x: zipf.layers[x[0]].decode[x[1]],
    )
    assert hottest[1] in pl[hottest[0]].replicas
    pl0, _, _ = _solve(zipf, _cfg(L * 12, spare=0))
    assert not any(p.replicas for p in pl0.values())
    pl1, _, _ = _solve(
        zipf, dataclasses.replace(_cfg(L * 12, spare=20), replicate=False)
    )
    assert not any(p.replicas for p in pl1.values())


def test_insufficient_expert_budget_is_reported(zipf):
    cfg = dataclasses.replace(_cfg(L * 12), expert_budget_bytes=SLOT * 10)
    with pytest.raises(ValueError, match="no-cold-complete"):
        solve_placement(zipf, _topo(), cfg)
    # without cold_complete only the cold experts need a V620 slot
    cfg = dataclasses.replace(
        _cfg(L * 40),
        cold_complete=False,
        expert_budget_bytes=SLOT * (E - 40 + 2) * L // D,
    )
    pl, doc, _ = _solve(zipf, cfg)
    assert validate.validate_plan(doc, zipf) == []
    assert any(not any(d >= 1 for d in row) for row in doc["layers"][0]["owners"])


def test_cold_load_is_balanced_across_expert_devices(zipf):
    pl, _, topo = _solve(zipf, _cfg(L * 8, spare=0))
    shape = step_shape(_cfg(0), zipf)
    m = simulate.analytic_metrics(_arrays(pl, topo), zipf, shape)
    assert m["cold_imbalance"] < 1.15  # LPT on expected touches
    base = simulate.analytic_metrics(
        simulate.fallback_arrays(zipf, topo, _cfg(L * 8)), zipf, shape
    )
    assert (
        m["cold_touched_critical_per_step"]
        < 0.8 * base["cold_touched_critical_per_step"]
    )


def test_hub_modes(zipf):
    topo = _topo(n_hubs=2)
    rep, doc_r, _ = _solve(zipf, _cfg(L * 8, hub_mode="replicate"), topo)
    shard, doc_s, _ = _solve(zipf, _cfg(L * 8, hub_mode="shard"), topo)
    part, doc_p, _ = _solve(zipf, _cfg(L * 8, hub_mode="partition"), topo)

    def total(pl):  # distinct hub-resident experts, summed over layers
        return sum(len(set().union(*p.hub)) for p in pl.values())

    assert total(rep) == L * 8 and total(shard) == total(part) == 2 * L * 8
    for li in range(L):
        assert shard[li].hub[0] == shard[li].hub[1]  # same ids, each holds a slice
        a, b = part[li].hub
        assert not set(a) & set(b) and abs(len(a) - len(b)) <= 1
    for doc in (doc_r, doc_s, doc_p):
        assert validate.validate_plan(doc, zipf) == []


def test_mtp_policies(tmp_path, zipf):
    record = copy.deepcopy(zipf)
    src = record.layers[0]
    record.layers[L] = dataclasses.replace(
        src, layer=L, kind="mtp", name="mtp.layers.0.mlp.experts"
    )
    base = _cfg(L * 12, max_hub_per_layer=E)
    omit, _, _ = solve_placement(
        record, _topo(), dataclasses.replace(base, mtp_policy="omit")
    )
    assert L not in omit
    cfg = dataclasses.replace(
        base,
        mtp_policy="hub",
        mtp_expert_bytes=2 * SLOT,
        hub_budget_bytes=(L * 12 + 2 * E) * SLOT,
    )
    hub = solve_placement(record, _topo(), cfg)[0]
    assert hub[L].hub_only and len(hub[L].hub[0]) == E
    assert (
        sum(len(hub[i].hub[0]) for i in range(L)) == L * 12
    )  # MTP took its bytes first
    too_small = dataclasses.replace(cfg, hub_budget_bytes=SLOT)
    with pytest.raises(ValueError, match="MTP"):
        solve_placement(record, _topo(), too_small)
    solved, _, _ = solve_placement(
        record,
        _topo(),
        dataclasses.replace(
            cfg, mtp_policy="solve", expert_budget_bytes=E // D * (L + 2) * SLOT
        ),
    )
    assert not solved[L].hub_only and any(solved[L].owner >= 0)


# ---- fallback -----------------------------------------------------------------


def test_fallback_stripe_when_no_stats():
    topo = _topo()
    cfg = _cfg(L * E)
    pl = fallback_plan(L, E, topo, cfg)
    for p in pl.values():
        assert p.hub[0] == [e for e in range(E) if e % 4 == 0]  # R = 4
        assert np.bincount(p.owner, minlength=D).tolist() == [E // D] * D
        assert not p.replicas
    doc = plan_document(
        pl,
        topo,
        cfg,
        {"num_layers": L, "num_experts": E, "top_k": K},
        {},
        "uniform-fallback",
        [],
    )
    assert doc["source"] == "uniform-fallback" and validate.validate_plan(doc) == []
    again = plan_document(
        fallback_plan(L, E, topo, cfg),
        topo,
        cfg,
        {"num_layers": L, "num_experts": E, "top_k": K},
        {},
        "uniform-fallback",
        [],
    )
    assert again["plan_id"] == doc["plan_id"]


def test_fallback_respects_a_small_hub_budget():
    topo = _topo()
    pl = fallback_plan(L, E, topo, _cfg(L * 5))
    for p in pl.values():
        assert len(p.hub[0]) == 5
        assert p.hub[0] == sorted(set(p.hub[0]))
        assert max(p.hub[0]) - min(p.hub[0]) > E // 2  # spread, not a prefix
    assert all(not p.hub[0] for p in fallback_plan(L, E, topo, _cfg(0)).values())


# ---- validator ----------------------------------------------------------------


@pytest.fixture(scope="module")
def good_doc(zipf):
    return _solve(zipf, _cfg(L * 12, spare=10))[1]


def test_validator_accepts_and_rejects(good_doc):
    assert validate.validate_plan(good_doc) == []

    def broken(mutate):
        doc = copy.deepcopy(good_doc)
        mutate(doc)
        return validate.validate_plan(doc)

    def drop_cold(doc):
        doc["layers"][0]["owners"][5] = [
            d for d in doc["layers"][0]["owners"][5] if d < 1
        ]

    assert any("no expert-device copy" in x for x in broken(drop_cold))
    assert any(
        "duplicate owner" in x
        for x in broken(
            lambda d: d["layers"][0]["owners"][3].append(d["layers"][0]["owners"][3][0])
        )
    )
    assert any(
        "outside island 0" in x
        for x in broken(lambda d: d["layers"][0]["owners"][3].append(7))
    )

    def reverse_a_hub_row(doc):
        row = next(r for r in doc["layers"][0]["owners"] if len(r) > 1 and r[0] < 1)
        row.reverse()

    assert any("hubs must precede" in x for x in broken(reverse_a_hub_row))
    assert any(
        "owner lists" in x for x in broken(lambda d: d["layers"][0]["owners"].pop())
    )

    def skew_primaries(doc):
        for row in doc["layers"][0]["owners"]:
            if len(row) == 1 and row[0] >= 1:
                row[0] = 1

    assert any("unbalanced" in x for x in broken(skew_primaries))
    assert any(
        "exceeds its budget" in x
        for x in broken(lambda d: d["devices"][1].update(expert_budget_bytes=1))
    )
    assert any(
        "exceeds its budget" in x
        for x in broken(lambda d: d["devices"][0].update(expert_budget_bytes=1))
    )
    assert any(
        "spans PCIe switches" in x
        for x in broken(lambda d: d["devices"][1].update(switch="zzz"))
    )
    assert any("partition" in x for x in broken(lambda d: d["islands"][1].pop()))
    assert any("format" in x for x in broken(lambda d: d.update(format="nope")))


def test_validator_flags_a_missing_stats_layer(zipf, good_doc):
    doc = copy.deepcopy(good_doc)
    doc["layers"].pop()
    assert any("missing from the plan" in x for x in validate.validate_plan(doc, zipf))


# ---- simulator ----------------------------------------------------------------


def _tiny_doc(layers_owners, e=8, hubs=1, d=2):
    devices = [
        {
            "id": f"hub{i}",
            "role": "hub",
            "arch": "gfx1100",
            "hip": i,
            "switch": "a",
            "expert_budget_bytes": 10**9,
        }
        for i in range(hubs)
    ]
    devices += [
        {
            "id": f"exp{i}",
            "role": "expert",
            "arch": "gfx1030",
            "hip": 10 + i,
            "switch": "a",
            "expert_budget_bytes": 10**9,
        }
        for i in range(d)
    ]
    return {
        "format": "hetero-moe-placement/1",
        "version": 1,
        "model": {"num_experts": e},
        "expert_bytes": 1,
        "devices": devices,
        "islands": [list(range(hubs + d))],
        "layers": [{"layer": i, "owners": o} for i, o in enumerate(layers_owners)],
    }


def test_evaluate_ids_hand_computed():
    # experts 0,1 on the hub (device 0); 2..5 on V620 0 (device 1); 6,7 on V620 1
    owners = [[0, 1], [0, 1], [1], [1], [1], [1], [2], [2]]
    pa = simulate.parse_plan(_tiny_doc([owners]))
    ids = {0: np.array([[0, 2, 6], [2, 3, 6]])}
    r = simulate.evaluate_ids(pa, ids, np.array([0, 0]), np.array([0, 1]), 1)
    assert r["hub_hit"] == pytest.approx(1 / 6)
    # device 0 touched {2, 3}; device 1 touched {6} -> max 2, mean 1.5
    assert (
        r["cold_max_touched_per_step"] == 2 and r["cold_mean_touched_per_step"] == 1.5
    )
    assert r["hub_touched_per_step"] == 1
    # rows sent: device 0 gets 2 rows, device 1 gets 2 rows -> 2 * (lat + 2*5120/24.5e3)
    kw = {"latency_us": 6.2, "gbps": 24.5}
    r = simulate.evaluate_ids(
        pa, ids, np.array([0, 0]), np.array([0, 1]), 1, exchange="rows", **kw
    )
    assert r["comm_us_per_step"] == pytest.approx(2 * (6.2 + 2 * 5120 / 24.5e3))
    # hx broadcast (the default): all T = 2 rows go to both V620s (D = 2)
    r = simulate.evaluate_ids(pa, ids, np.array([0, 0]), np.array([0, 1]), 1, **kw)
    assert r["comm_us_per_step"] == pytest.approx(2 * (6.2 + 2 * 2 * 5120 / 24.5e3))


def test_decode_dispatch_rules_and_replica_touch_ratio():
    owners = [[0, 1], [0, 1], [1, 2]] + [[1]] * 5  # expert 2 on both V620s
    pa = simulate.parse_plan(_tiny_doc([owners, owners]))
    ids = {0: np.array([[2], [2]]), 1: np.array([[2], [2]])}
    args = (pa, ids, np.array([0, 0]), np.array([0, 1]), 1)
    row = simulate.evaluate_ids(*args)
    assert row["cold_replica_touch_ratio"] == 2.0  # streamed on both copies
    for rule in ("expert", "primary"):  # one copy per expert: no duplication
        r = simulate.evaluate_ids(*args, dispatch=rule)
        assert r["cold_replica_touch_ratio"] == 1.0
        assert r["cold_max_touched_per_step"] == 2  # one touch per layer
    with pytest.raises(ValueError, match="dispatch"):
        simulate.evaluate_ids(*args, dispatch="nope")


def test_replicated_cold_expert_splits_rows_by_row_plus_layer():
    owners = [[0, 1], [0, 1], [1, 2]] + [[1]] * 5  # expert 2 on both V620s
    pa = simulate.parse_plan(_tiny_doc([owners, owners]))
    ids = {0: np.array([[2], [2]]), 1: np.array([[2], [2]])}
    r = simulate.evaluate_ids(pa, ids, np.array([0, 0]), np.array([0, 1]), 1)
    # layer 0: rows 0,1 -> copies 0,1 (one touch per device); layer 1 likewise
    assert (
        r["cold_max_touched_per_step"] == 2 and r["cold_mean_touched_per_step"] == 2.0
    )


def test_simulation_matches_the_analytic_hit_rate_and_beats_stripe(zipf):
    cfg = _cfg(L * 16, spare=20)
    pl, _, topo = _solve(zipf, cfg)
    arr = _arrays(pl, topo)
    shape = step_shape(cfg, zipf)
    analytic = simulate.analytic_metrics(arr, zipf, shape)
    sim = simulate.simulate(arr, zipf, steps=1500, seed=3, hub_us=2.0, expert_us=3.0)
    assert sim["hub_hit"] == pytest.approx(analytic["decode_hub_hit"], abs=0.03)
    assert sim["cold_imbalance"] >= 1.0 and sim["step_us"] > 0
    base = simulate.simulate(
        simulate.fallback_arrays(zipf, topo, cfg), zipf, steps=1500, seed=3
    )
    assert sim["cold_max_touched_per_step"] < 0.8 * base["cold_max_touched_per_step"]
    # same seed, same numbers
    assert simulate.simulate(arr, zipf, steps=200, seed=9) == simulate.simulate(
        arr, zipf, steps=200, seed=9
    )


def test_coactivation_refinement_lowers_the_cold_maximum(tmp_path):
    layers, experts, top_k = 4, 128, 8
    data = synth.generate(
        layers=layers,
        experts=experts,
        top_k=top_k,
        alpha=0.8,
        decode_rows=6000,
        prefill_rows=1000,
        clusters=8,
        beta=3.0,
        keep_trace=True,
        pairs=True,
        seed=3,
    )
    record = stats_io.load_stats([synth.write_raw(data, str(tmp_path))])
    assert record.coact is not None
    topo = _topo()
    step_of_row = np.repeat(np.arange(len(data.step_rows)), data.step_rows)
    starts = np.cumsum(data.step_rows) - data.step_rows
    row_in_step = np.arange(step_of_row.size) - np.repeat(starts, data.step_rows)
    ids = {i: data.trace[:, i, :].astype(np.int64) for i in range(layers)}
    result = {}
    for refine in (False, True):
        cfg = SolverConfig(
            expert_bytes=SLOT,
            hub_budget_bytes=SLOT * 24,
            expert_budget_bytes=SLOT * (experts // D) * layers,
            max_hub_per_layer=24,
            coact_refine=refine,
            refine_iters=200,
            replicate=False,
        )
        pl, _, _ = solve_placement(record, topo, cfg)
        arr = simulate.arrays_from_placements(pl, topo, experts)
        result[refine] = simulate.evaluate_ids(
            arr, ids, step_of_row, row_in_step, len(data.step_rows)
        )
    assert result[True]["hub_hit"] == pytest.approx(result[False]["hub_hit"])
    assert (
        result[True]["cold_max_touched_per_step"]
        < result[False]["cold_max_touched_per_step"]
    )
    # the mean touched count is unchanged: only the spread between devices moved
    assert result[True]["cold_mean_touched_per_step"] == pytest.approx(
        result[False]["cold_mean_touched_per_step"], rel=0.02
    )


# ---- byte table ---------------------------------------------------------------


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = model_bytes.tensor_bytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + n],
        }
        offset += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)  # header only: no data is read


def test_model_bytes_from_safetensors_headers(tmp_path):
    t = {}
    for layer in range(2):
        for e in range(4):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
            for proj, (n, k) in {
                "gate_proj": (16, 64),
                "up_proj": (16, 64),
                "down_proj": (64, 32),
            }.items():
                t[f"{base}.{proj}.weight_packed"] = ("I32", [n, k // 8])
                t[f"{base}.{proj}.weight_scale"] = ("BF16", [n, k // 32])
                t[f"{base}.{proj}.weight_shape"] = ("I64", [2])
        t[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = (
            "BF16",
            [8, 8],
        )
        t[f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"] = (
            "BF16",
            [8, 4],
        )
        t[f"model.language_model.layers.{layer}.mlp.gate.weight"] = ("BF16", [4, 8])
    t[
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    ] = ("BF16", [100, 10])
    t["lm_head.weight"] = ("BF16", [32, 8])
    t["model.language_model.embed_tokens.weight"] = ("BF16", [32, 8])
    t["model.visual.blocks.0.attn.qkv.weight"] = ("BF16", [6, 6])
    t["mtp.layers.0.mlp.experts.gate_up_proj"] = ("BF16", [4, 32, 8])
    t["mtp.layers.0.mlp.experts.down_proj"] = ("BF16", [4, 8, 16])
    t["mtp.layers.0.self_attn.q_proj.weight"] = ("BF16", [8, 8])
    _write_safetensors(tmp_path / "model-00001.safetensors", t)
    table = model_bytes.from_checkpoint(str(tmp_path))
    per_expert = 2 * (16 * 8 * 4 + 16 * 2 * 2) + 64 * 4 * 4 + 64 * 1 * 2  # no shapes
    assert table["expert_bytes"] == per_expert
    assert table["num_layers"] == 2 and table["num_experts"] == 4
    assert table["routed_expert_total_bytes"] == per_expert * 8
    assert table["mtp"] == {
        "layers": 1,
        "num_experts": 4,
        "expert_bytes": (32 * 8 + 8 * 16) * 2,
        "expert_dtype": "BF16",
    }
    ab = table["active_bytes"]
    assert ab["lm_head"] == ab["embed_tokens"] == 32 * 8 * 2
    assert (
        ab["attention"] == 2 * 128
        and ab["gdn"] == 2 * 64
        and ab["router_gate"] == 2 * 64
    )
    assert ab["mtp_non_expert"] == 128
    assert table["host_bytes"]["ple_ngram"] == 2000 and table["visual_bytes"] == 72
    assert table["hub_active_bytes"] == sum(ab.values())


def test_committed_flashnext_byte_table():
    table = model_bytes.load(str(ROOT / solve.DEFAULT_TABLE))
    assert table["expert_bytes"] == 2_534_400  # 3 x 819,200 packed + 3 x 25,600 scales
    assert table["num_layers"] == 48 and table["num_experts"] == 512
    assert table["routed_expert_total_bytes"] == 2_534_400 * 512 * 48
    assert (
        table["mtp"]["expert_bytes"] == 9_830_400
        and table["mtp"]["expert_dtype"] == "BF16"
    )
    assert table["expert_resident_extra_bytes_rdna2"] == 19_200


# ---- CLIs ---------------------------------------------------------------------


def test_cli_end_to_end(tmp_path, zipf, capsys):
    raw = synth.write_raw(
        synth.generate(
            layers=L,
            experts=E,
            top_k=K,
            decode_rows=2000,
            prefill_rows=1000,
            pairs=True,
            clusters=4,
            seed=5,
        ),
        str(tmp_path / "raw"),
    )
    stats = tmp_path / "stats.json"
    stats_io.main(
        [
            "convert",
            raw,
            "--out",
            str(stats),
            "--workload",
            "synthetic",
            "--engine-commit",
            "abc",
        ]
    )
    doc = json.loads(stats.read_text())
    assert (
        doc["window"]["workload"] == "synthetic"
        and doc["window"]["engine_commit"] == "abc"
    )
    table = tmp_path / "bytes.json"
    table.write_text(
        json.dumps(
            {
                "format": "hetero-moe-model-bytes/1",
                "model": "toy",
                "num_layers": L,
                "num_experts": E,
                "expert_bytes": SLOT,
                "hub_active_bytes": 0,
                "mtp": {"layers": 1, "expert_bytes": 4 * SLOT},
            }
        )
    )
    plan = tmp_path / "plan.json"
    gib = 1 << 30
    common = [
        "--bytes",
        str(table),
        "--hub-expert-budget-gib",
        str(L * 20 * SLOT / gib),
        "--expert-budget-gib",
        str((E // D * L + 30) * SLOT / gib),
    ]
    assert (
        solve.main(
            [
                "--stats",
                str(stats),
                *common,
                "--out",
                str(plan),
                "--simulate-steps",
                "300",
            ]
        )
        == 0
    )
    out = json.loads(plan.read_text())
    assert out["source"] == "stats" and out["devices"][0]["hip"] == 0
    assert [d["hip"] for d in out["devices"][:5]] == [0, 1, 2, 3, 4]
    assert [d["switch"] for d in out["devices"][4:6]] == ["c0", "00"]
    assert out["estimate"]["gain_vs_stripe"]["decode_hub_hit"] > 0
    assert validate.main([str(plan), "--stats", str(stats)]) == 0
    runtime = out["runtime"]["spill_k"]
    assert sorted(runtime, key=int) == ["3", "6", "12", "24", "48"]
    assert all(isinstance(v, int) and v >= 0 for v in runtime.values())
    assert runtime["3"] <= runtime["12"] <= runtime["48"]  # wider step, more kept
    variants = out["estimate"]["dispatch_variants"]
    assert sorted(variants["simulated_decode"]) == ["expert", "primary", "row"]
    # the primary copy never streams an expert twice in a step
    assert variants["simulated_decode"]["primary"]["cold_replica_touch_ratio"] == 1.0
    assert variants["prefill_pair_load_imbalance"]["replicas_spread"] > 0
    capsys.readouterr()
    assert simulate.main([str(plan), "--stats", str(stats), "--steps", "200"]) is None
    sim = json.loads(capsys.readouterr().out)
    assert sim["steps"] == 200 and 0 < sim["hub_hit"] < 1 and "analytic" in sim
    # raw files work directly too, with --no-coact
    plan2 = tmp_path / "plan2.json"
    assert (
        solve.main(
            [
                "--stats",
                raw,
                "--no-coact",
                *common,
                "--out",
                str(plan2),
                "--simulate-steps",
                "0",
            ]
        )
        == 0
    )
    assert json.loads(plan2.read_text())["layers"]


def test_cli_fallback_without_stats(tmp_path):
    table = tmp_path / "bytes.json"
    table.write_text(
        json.dumps(
            {
                "format": "hetero-moe-model-bytes/1",
                "model": "toy",
                "num_layers": L,
                "num_experts": E,
                "expert_bytes": SLOT,
                "hub_active_bytes": 0,
            }
        )
    )
    plan = tmp_path / "plan.json"
    assert (
        solve.main(
            ["--bytes", str(table), "--out", str(plan), "--hub-expert-budget-gib", "1"]
        )
        == 0
    )
    doc = json.loads(plan.read_text())
    assert doc["source"] == "uniform-fallback" and doc["stats_sha256"] == []
    assert len(doc["layers"]) == L
    assert [
        i for i, row in enumerate(doc["layers"][0]["owners"]) if row[0] < 1
    ] == list(range(0, E, 4))
    assert validate.main([str(plan)]) == 0


def test_cli_rejects_bad_plans(tmp_path, good_doc):
    bad = copy.deepcopy(good_doc)
    bad["layers"][0]["owners"][0] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    assert validate.main([str(path)]) == 1


# ---- cost model and island modes (HeteroDesign requests) -----------------------


def test_cost_model_defaults_and_fixed_terms():
    assert (simulate.HUB_US, simulate.EXPERT_US) == (3.7, 6.7)
    assert (simulate.PCIE_GBPS, simulate.PCIE_LATENCY_US) == (25.0, 13.3)
    owners = [[0, 1], [0, 1], [1], [1], [1], [1], [2], [2]]
    pa = simulate.parse_plan(_tiny_doc([owners]))
    ids = {0: np.array([[0, 2, 6], [2, 3, 6]])}
    args = (pa, ids, np.array([0, 0]), np.array([0, 1]), 1)
    kw = {
        "hub_us": 2.0,
        "expert_us": 3.0,
        "latency_us": 6.2,
        "gbps": 24.5,
        "exchange": "rows",
    }
    comm = 2 * (6.2 + 2 * 5120 / 24.5e3)
    # hub touched 1, V620 max touched 2: max(hf + 1 * 2, comm + ef + 2 * 3)
    r = simulate.evaluate_ids(*args, **kw)
    assert r["step_us"] == pytest.approx(comm + 6.0)
    # the hx broadcast pushes all T = 2 rows to both V620s (D = 2)
    bcast = 2 * (6.2 + 2 * 2 * 5120 / 24.5e3)
    r = simulate.evaluate_ids(*args, **{**kw, "exchange": "broadcast"})
    assert r["step_us"] == pytest.approx(bcast + 6.0)
    r = simulate.evaluate_ids(*args, **kw, expert_fixed_us=5.0)
    assert r["step_us"] == pytest.approx(comm + 11.0)
    slow_hub = {**kw, "hub_fixed_us": 100.0, "expert_fixed_us": 5.0}
    r = simulate.evaluate_ids(*args, **slow_hub, spill=False)
    assert r["step_us"] == pytest.approx(102.0) and r["hub_spilled_per_step"] == 0
    # with spill the hub's one expert moves to its V620 copy: the hub still pays
    # its fixed 100 us, the V620 side (2 + 1 experts over 2 devices) stays at 2
    r = simulate.evaluate_ids(*args, **slow_hub)
    assert r["step_us"] == pytest.approx(100.0)
    assert r["step_us_no_spill"] == pytest.approx(102.0)
    assert r["hub_spilled_per_step"] == 1
    # a spilled expert with a slow V620 is not worth it: hub 1 x 2 us vs 3 us + comm
    r = simulate.evaluate_ids(*args, **kw)
    assert r["hub_spilled_per_step"] == 0 and r["step_us"] == pytest.approx(comm + 6.0)


def _partition_cfg(hub_slots, **kw):
    kw.setdefault("max_hub_per_layer", E // 2)
    return _cfg(hub_slots, island_mode="partition", **kw)


def test_partition_island_mode(zipf):
    cfg = _partition_cfg(L * 8, spare=10)
    pl, doc, topo = _solve(zipf, cfg)
    assert doc["island_mode"] == "partition"
    assert validate.validate_plan(doc, zipf) == []
    groups = doc["islands"]
    shape = step_shape(cfg, zipf)
    for layer in doc["layers"]:
        owners = layer["owners"]
        isl = [
            {k for k, g in enumerate(groups) for d in row if d in g} for row in owners
        ]
        assert all(len(s) == 1 for s in isl)  # each expert in exactly one island
        per = [sum(1 for s in isl if s == {k}) for k in range(2)]
        assert per == [E // 2, E // 2]
        # expected decode touches balanced between the islands (LPT)
        q = (
            zipf.layers[layer["layer"]].decode
            / zipf.layers[layer["layer"]].decode_tokens
        )
        t = sum(p * (1 - (1 - np.clip(q, 0, 1)) ** r) for r, p in shape.items())
        load = [sum(t[e] for e, s in enumerate(isl) if s == {k}) for k in range(2)]
        assert abs(load[0] - load[1]) < 0.05 * sum(load)
    # hubs of both islands are used, each within its own budget
    for h in (0, 5):
        held = sum(1 for lay in doc["layers"] for row in lay["owners"] if h in row)
        assert 0 < held <= L * 8
    pa = _arrays(pl, topo)
    assert pa.n_devices == 2 * D and pa.n_hub_groups == 2
    again = _solve(zipf, cfg)[1]
    assert again["plan_id"] == doc["plan_id"]
    # the plan, parsed back from JSON, gives the same arrays as the solver output
    back = simulate.parse_plan(json.loads(json.dumps(doc)))
    assert np.array_equal(back.copies, pa.copies) and np.array_equal(back.hub, pa.hub)
    sim = simulate.simulate(pa, zipf, steps=800, seed=2)
    base = simulate.simulate(
        simulate.fallback_arrays(zipf, topo, cfg), zipf, steps=800, seed=2
    )
    assert sim["hub_hit"] > 2 * base["hub_hit"]
    assert sim["step_us"] < base["step_us"]


def test_partition_validator_rejects_cross_island_and_imbalance(zipf):
    _, doc, _ = _solve(zipf, _partition_cfg(L * 8, spare=10))
    bad = copy.deepcopy(doc)
    row = bad["layers"][0]["owners"][0]
    other = 6 if row[-1] < 5 else 1  # an expert device of the other island
    row.append(other)
    assert any("exactly one island" in x for x in validate.validate_plan(bad))
    bad = copy.deepcopy(doc)
    owners = bad["layers"][0]["owners"]
    moved = [e for e, r in enumerate(owners) if r[-1] in (1, 2, 3, 4)][0]
    owners[moved] = [owners[moved][-1] + 5]  # same expert device index, other island
    assert any("unbalanced" in x for x in validate.validate_plan(bad))
    replicate_doc = copy.deepcopy(doc)
    replicate_doc["island_mode"] = "replicate"  # owners outside island 0 now
    assert any("outside island 0" in x for x in validate.validate_plan(replicate_doc))


def test_partition_fallback_and_mtp_hub(zipf):
    cfg = _partition_cfg(L * 8)
    topo = _topo()
    fb = fallback_plan(L, E, topo, cfg)
    doc = plan_document(fb, topo, cfg, {"num_layers": L, "num_experts": E}, {}, "x", [])
    assert validate.validate_plan(doc) == []
    assert list(fb[0].island_of[:4]) == [0, 1, 0, 1]
    record = copy.deepcopy(zipf)
    record.layers[L] = dataclasses.replace(
        record.layers[0], layer=L, kind="mtp", name="mtp.layers.0.mlp.experts"
    )
    cfg = dataclasses.replace(
        _partition_cfg(0, spare=4 * E // D),
        mtp_policy="hub",
        mtp_expert_bytes=2 * SLOT,
        hub_budget_bytes=(L * 6 + E) * SLOT,  # E / 2 MTP experts x 2 slots per hub
    )
    pl, _, _ = solve_placement(record, topo, cfg)
    mtp = pl[L]
    assert mtp.hub_only and [len(x) for x in mtp.hub] == [E]  # both islands' halves
    assert sorted(mtp.hub[0]) == list(range(E))
    model = {"num_layers": L, "num_experts": E, "mtp_expert_bytes": 2 * SLOT}
    doc = plan_document(pl, topo, cfg, model, {}, "stats", [])
    assert validate.validate_plan(doc) == []
    for h in (0, 5):  # MTP halves first, then the target experts share the rest
        n_mtp = sum(1 for row in doc["layers"][-1]["owners"] if h in row)
        n_tgt = sum(
            1 for lay in doc["layers"][:-1] for row in lay["owners"] if h in row
        )
        assert n_mtp == E // 2 and n_tgt == L * 6


def test_cli_partition_and_defaults(tmp_path, zipf, capsys):
    raw = synth.write_raw(
        synth.generate(layers=L, experts=E, top_k=K, decode_rows=2000, seed=5),
        str(tmp_path / "raw"),
    )
    table = tmp_path / "bytes.json"
    table.write_text(
        json.dumps(
            {
                "format": "hetero-moe-model-bytes/1",
                "model": "toy",
                "num_layers": L,
                "num_experts": E,
                "expert_bytes": SLOT,
                "hub_active_bytes": 0,
            }
        )
    )
    gib = 1 << 30
    plan = tmp_path / "plan.json"
    common = [
        "--bytes", str(table),
        "--hub-expert-budget-gib", str(L * 8 * SLOT / gib),
        "--expert-budget-gib", str((E // D * L + 10) * SLOT / gib),
        "--island-mode", "partition",
        "--out", str(plan),
    ]  # fmt: skip
    assert solve.main(["--stats", raw, *common, "--simulate-steps", "200"]) == 0
    out = json.loads(plan.read_text())
    assert (
        out["island_mode"] == "partition"
        and out["meta"]["limits"]["max_hub_per_layer"] == 115
    )
    assert out["meta"]["cost_model"]["hub_us"] == 3.7
    assert "simulated_step_us_ratio" in out["estimate"]["gain_vs_stripe"]
    assert validate.main([str(plan)]) == 0
    fallback_args = ["--bytes", str(table), "--island-mode", "partition"]
    fallback_args += ["--out", str(plan), "--hub-expert-budget-gib", "1"]
    assert solve.main(fallback_args) == 0
    assert validate.main([str(plan)]) == 0
    # real-box defaults: 44.98 GiB hub minus the active part and 12.1 GiB reserve
    real = model_bytes.load(solve.DEFAULT_TABLE)
    args = argparse.Namespace(
        hub_expert_budget_gib=None,
        hub_vram_gib=44.98,
        hub_active_gib=None,
        hub_reserve_gib=12.1,
        expert_budget_gib=26.0,
    )
    hub, expert = solve.derive_budgets(args, real)
    assert expert == round(26 * gib)
    assert hub == round(44.98 * gib - real["hub_active_bytes"] - 12.1 * gib)


# ---- T10 report (tools/rdna/hetero_moe/t10_report.py) --------------------------


def _t10_raw(path, ranks_tamper=0):
    """A consistent two-target-layer + MTP capture: 3 snapshots of 4 decode steps."""
    names = ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]
    names.append("mtp.layers.0.mlp.experts")
    kinds = ["target", "target", "mtp"]
    meta = {
        "type": "meta",
        "schema": "expert-stats/1",
        "num_experts": 4,
        "top_k": 2,
        "decode_max_tokens": 4,
        "time": 1.0,
        "tp_rank": 0,
        "layers": [
            {"name": n, "idx": 0 if k == "mtp" else i, "kind": k}
            for i, (n, k) in enumerate(zip(names, kinds, strict=True))
        ],
    }
    lines = [meta]
    for seq in (1, 2, 3):
        counts = [seq * 2, seq * 2, seq * 2, seq * 2]  # 2 rows/step, k = 2
        counts[0] -= ranks_tamper if seq == 3 else 0
        snap = {
            "type": "snapshot",
            "seq": seq,
            "time": 1.0 + seq,
            "elapsed_s": 30.0 * seq,
            "counts": {
                n: {"decode": list(counts), "prefill": [0, 0, 0, 0]} for n in names
            },
            "tokens": {n: {"decode": seq * 4, "prefill": 0} for n in names},
            "steps": {n: {"decode": seq * 2, "prefill": 0} for n in names},
            "rows_hist": {n: [0, 0, seq * 2, 0, 0] for n in names},
            "invalid": {n: 0 for n in names},
        }
        lines.append(snap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_t10_check_passes_a_consistent_capture_and_flags_a_lost_update(tmp_path):
    from tools.rdna.hetero_moe import t10_report

    ok = tmp_path / "ok" / "stats" / "expert_stats-dp0-pid1.jsonl"
    _t10_raw(ok)
    (tmp_path / "ok" / "metrics-poll.txt").write_text(
        'epoch 5\nvllm:iteration_tokens_total_sum{engine="0"} 8.0\n'
    )
    res = t10_report.check_arm(str(tmp_path / "ok"))
    f = res["files"][0]
    assert f["problems"] == [] and f["targets"] == 2 and f["snapshots"] == 3
    assert f["target_tokens_final_minus_seq1"] == [8]
    assert res["target_rows_vs_server_final"]["diff"] == 0
    assert f["final_newer_than_last_periodic"]

    bad = tmp_path / "bad" / "stats" / "expert_stats-dp0-pid1.jsonl"
    _t10_raw(bad, ranks_tamper=1)  # one lost index_add_ update in the last snapshot
    probs = t10_report.check_arm(str(tmp_path / "bad"))["files"][0]["problems"]
    assert any("sum(counts)" in p for p in probs)
    assert any("rows_hist" in p for p in probs)


def test_t10_rank_agreement_reports_the_difference(tmp_path):
    from tools.rdna.hetero_moe import t10_report

    _t10_raw(tmp_path / "c" / "stats" / "expert_stats-dp0-pid1.jsonl")
    _t10_raw(tmp_path / "c" / "stats" / "expert_stats-dp0-pid2.jsonl", ranks_tamper=1)
    agree = t10_report.check_arm(str(tmp_path / "c"))["rank_agreement"]
    (entry,) = agree.values()
    assert entry["layers_differing"] == 3 and entry["max_abs_count_diff"] == 1


def test_t10_itl_rule_resolves_only_beyond_the_spread(tmp_path):
    from tools.rdna.hetero_moe import t10_report

    def cells(arm, med):
        d = tmp_path / "arms" / arm
        d.mkdir(parents=True)
        for cell in t10_report.CELLS:
            (d / f"{cell}.json").write_text(
                json.dumps({"median_itl_ms": med, "mean_itl_ms": med})
            )

    cells("A1", 20.0)
    cells("A2", 20.4)
    cells("B1", 21.9)
    cells("B2", 22.1)
    s = t10_report.itl_table(str(tmp_path))["summary"][
        "bench-1024x512-c1:median_itl_ms"
    ]
    assert s["delta"] == pytest.approx(1.8) and s["resolved"]
    assert s["spread_A"] == pytest.approx(0.4)
