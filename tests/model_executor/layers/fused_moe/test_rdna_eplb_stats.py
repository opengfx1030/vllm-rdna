# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the lean EPLB load counters.

The torch formulation of the counters is the reference of the HIP kernel (same
contract, fixed-shape ops), so the CPU run checks the semantics against an
independent plain-Python model, the ``eplb-stats/1`` writer against the reader in
``tools/rdna/eplb/summary.py``, and the source of the C++ kernel against the
constants of the Python side. The HIP kernel itself is checked by the GPU tests
below (skipped without a GPU and the built ops), which compare it with the torch
formulation call by call.
"""

import ast
import importlib
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import regex as re
import torch

from tools.rdna.eplb import summary, traces  # type: ignore[attr-defined]
from vllm.model_executor.layers.fused_moe import rdna_eplb_stats as es

E = 16
K = 4
RANKS = 4
DECODE_MAX = 8  # bands (1, 2, 3, 4, 6, 8)
NAME = "model.layers.3.mlp.experts"
NAME2 = "model.layers.4.mlp.experts"
ROOT = Path(__file__).resolve().parents[4]
CU = ROOT / "csrc/rocm/rdna_fused_glue.cu"
COMPRESSED = (
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py"
)


def _layer(name=NAME, rank=0, ranks=RANKS, strategy="linear"):
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )

    _, expert_map, _ = determine_expert_map(ranks, rank, E, strategy)
    cfg = SimpleNamespace(ep_size=ranks, ep_rank=rank, tp_rank=rank, dp_rank=0)
    return SimpleNamespace(
        layer_name=name,
        moe_config=cfg,
        global_num_experts=E,
        expert_map=expert_map,
        expert_placement_strategy=strategy,
    )


def _stats(tmp_path, **kwargs):
    kwargs.setdefault("kernel", "torch")
    kwargs.setdefault("decode_max", DECODE_MAX)
    kwargs.setdefault("ring", 64)
    return es.EplbStats(str(tmp_path), **kwargs)


def _ids(rows, seed=0):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, E, generator=gen)
    return scores.topk(K, dim=1).indices.to(torch.int32)


def _mix(i, e):
    """The kernel's per-entry hash with explicit 32-bit arithmetic."""
    m = 0xFFFFFFFF
    x = ((i * 0x9E3779B1) & m) ^ ((((e + 1) & m) * 0x85EBCA6B) & m)
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & m
    return x ^ (x >> 12)


class Model:
    """Plain-Python contract of one layer's counters (rdna_fused_glue.cu)."""

    def __init__(self, owner, edges, hist_n=0, every=1):
        self.owner, self.edges, self.hist_n, self.every = owner, edges, hist_n, every
        nb = len(edges)
        self.pairs = np.zeros((2, E), np.int64)
        self.touch = np.zeros((nb, E), np.int64)
        self.calls = np.zeros(nb + 1, np.int64)
        self.rows = np.zeros(2, np.int64)
        self.ordinal = 0
        self.hist = np.zeros((hist_n, E), np.int16)
        self.hist_rows = np.zeros(hist_n, np.int64)
        self.hist_calls = 0

    def call(self, pos, ids, padding=None):
        """Apply one call; returns its 32-word ring record, None if not sampled."""
        ids = ids.numpy()
        rows, k = ids.shape
        ok = np.ones(rows, bool) if padding is None else ~padding.numpy()
        h = np.zeros(E, np.int64)
        digest = 0
        for r in range(rows):
            if not ok[r]:
                continue
            for j in range(k):
                v = int(ids[r, j])
                if 0 <= v < E:
                    h[v] += 1
                digest = (digest + _mix(r * k + j, v)) & 0xFFFFFFFF
        valid = int(ok.sum())
        ordinal = self.ordinal
        self.ordinal += 1
        if ordinal & (self.every - 1):
            return None
        nb = len(self.edges)
        phase = 0 if valid <= self.edges[-1] else 1
        band = next((b for b, e in enumerate(self.edges) if e >= valid), nb - 1)
        if valid > 0:
            self.pairs[phase] += h
            if phase == 0:
                self.touch[band] += h > 0
            self.calls[band if phase == 0 else nb] += 1
            self.rows[phase] += valid
            if phase == 1 and self.hist_n:
                slot = self.hist_calls % self.hist_n
                self.hist[slot] = np.minimum(h, 32767)
                self.hist_rows[slot] = valid
                self.hist_calls += 1
        touched = np.zeros(RANKS, np.int64)
        pairs = np.zeros(RANKS, np.int64)
        for e in range(E):
            if h[e] > 0:
                touched[self.owner[e]] += 1
                pairs[self.owner[e]] += h[e]
        rec = np.zeros(es.WORDS, np.int32)
        rec[0], rec[1], rec[2] = pos, valid, rows
        rec[3] = np.uint32(digest).view(np.int32)
        rec[6], rec[7] = phase, ordinal
        rec[8 : 8 + RANKS] = touched
        rec[16 : 16 + RANKS] = pairs
        return rec


def _owner(strategy="linear"):
    return es.build_owner(NAME, RANKS, E, strategy).numpy()


def _models(n, **kwargs):
    edges = es.band_edges(DECODE_MAX)
    return [Model(_owner(), edges, **kwargs) for _ in range(n)]


def _drive(stats, layers, models, calls):
    """Run ``calls`` = [(layer index, ids[, padding])]; returns the ring records."""
    records = []
    for index, ids, *padding in calls:
        pad = padding[0] if padding else None
        handle = stats.layer_stats(layers[index], ids)
        assert handle is not None
        with mock.patch.object(es, "_padding_rows", lambda rows, device, p=pad: p):
            handle.count(ids)
        rec = models[index].call(handle.pos, ids, pad)
        if rec is not None:
            records.append(rec)
    return records


def _check_state(loaded, models):
    for pos, m in enumerate(models):
        assert loaded.pairs[pos].tolist() == m.pairs.tolist()
        assert loaded.touch[pos].tolist() == m.touch.tolist()
        assert loaded.calls[pos].tolist() == m.calls.tolist()
        assert loaded.rows[pos].tolist() == m.rows.tolist()


def _load(tmp_path, rank=0):
    (path,) = sorted(tmp_path.glob(f"eplb_stats-rank{rank}-pid*.npz"))
    return path, summary.load_rank(str(path))


CALLS = [
    (0, _ids(1, 1)),
    (1, _ids(3, 2)),
    (0, _ids(3, 3)),
    (1, _ids(5, 4)),
    (0, _ids(8, 5)),
    (0, _ids(9, 6)),  # above the decode limit: prefill
    (1, _ids(20, 7)),
    (0, _ids(2, 8)),
    (1, _ids(6, 9)),
]


def test_counters_and_ring_match_the_reference_model(tmp_path):
    stats = _stats(tmp_path)
    models = _models(2)
    records = _drive(stats, [_layer(NAME), _layer(NAME2)], models, CALLS)
    stats.close()
    _, loaded = _load(tmp_path)
    _check_state(loaded, models)
    assert loaded.ring_total == len(CALLS) == len(records)
    assert loaded.ring.tolist() == [r.tolist() for r in records]
    assert loaded.meta["format"] == "eplb-stats/1"
    assert loaded.meta["band_edges"] == [1, 2, 3, 4, 6, 8]
    assert [x["name"] for x in loaded.layers] == [NAME, NAME2]
    assert [x["layer"] for x in loaded.layers] == [3, 4]
    assert loaded.meta["owner"][0] == _owner().tolist()
    assert (loaded.rank, loaded.ranks) == (0, RANKS)


def test_padding_rows_and_invalid_ids_are_not_counted(tmp_path):
    stats = _stats(tmp_path)
    models = _models(1)
    ids = _ids(6, 11)
    ids[0, 0] = -1  # an invalid id in a valid row: hashed, not counted
    ids[1, 1] = E + 3
    padding = torch.tensor([False, False, False, False, True, True])
    _drive(stats, [_layer()], models, [(0, ids, padding)])
    handle = stats._order[0]
    assert handle.state[:E].sum().item() == 4 * K - 2  # 4 valid rows, 2 invalid ids
    assert models[0].rows.tolist() == [4, 0]
    stats.close()
    _check_state(_load(tmp_path)[1], models)


def test_a_call_without_valid_rows_gets_a_ring_record_but_no_counts(tmp_path):
    stats = _stats(tmp_path)
    models = _models(1)
    ids = _ids(3, 12)
    padding = torch.ones(3, dtype=torch.bool)
    records = _drive(stats, [_layer()], models, [(0, ids, padding)])
    stats.close()
    loaded = _load(tmp_path)[1]
    assert loaded.pairs.sum() == 0 and loaded.calls.sum() == 0
    assert loaded.ring.tolist() == [records[0].tolist()]
    assert loaded.ring[0, summary.W_ROWS] == 0 and loaded.ring[0, summary.W_TOTAL] == 3


def test_sampling_counts_every_nth_call_and_the_ordinal_counts_all(tmp_path):
    stats = _stats(tmp_path, every=2)
    models = _models(2, every=2)
    records = _drive(stats, [_layer(NAME), _layer(NAME2)], models, CALLS)
    stats.close()
    loaded = _load(tmp_path)[1]
    _check_state(loaded, models)
    assert loaded.ring.tolist() == [r.tolist() for r in records]
    ordinals = [int(h.state[h.off.ordinal]) for h in stats._order]
    assert ordinals == [5, 4]  # calls of each layer, sampled or not
    assert loaded.meta["every"] == 2


def test_ring_wraps_and_the_reader_returns_the_newest_in_order(tmp_path):
    stats = _stats(tmp_path, ring=4)
    models = _models(1)
    calls = [(0, _ids(rows, 20 + rows)) for rows in (1, 2, 3, 4, 5, 6, 2, 3, 1, 4)]
    records = _drive(stats, [_layer()], models, calls)
    stats.close()
    loaded = _load(tmp_path)[1]
    assert loaded.ring_total == 10 and loaded.first_record == 6
    assert loaded.ring.tolist() == [r.tolist() for r in records[-4:]]
    _check_state(loaded, models)


def test_no_ring_still_writes_a_loadable_file(tmp_path):
    stats = _stats(tmp_path, ring=0)
    models = _models(1)
    _drive(stats, [_layer()], models, [(0, _ids(2, 1)), (0, _ids(9, 2))])
    stats.close()
    loaded = _load(tmp_path)[1]
    assert loaded.ring.shape == (0, es.WORDS) and loaded.ring_total == 0
    _check_state(loaded, models)


def test_prefill_histograms_ring_matches_the_reader(tmp_path):
    stats = _stats(tmp_path, hist=2)
    models = _models(1, hist_n=2)
    calls = [(0, _ids(rows, 30 + rows)) for rows in (12, 3, 16, 9, 20)]
    _drive(stats, [_layer()], models, calls)
    stats.close()
    path, loaded = _load(tmp_path)
    _check_state(loaded, models)
    sc = traces.eplb_stats_hist(str(path))[3]  # layer index of NAME
    # four prefill calls (12, 16, 9, 20 rows): the ring of two keeps the newest two
    assert sc.counts.shape == (2, E) and sc.rows.tolist() == [9, 20]
    want = [np.bincount(calls[i][1].numpy().ravel(), minlength=E) for i in (3, 4)]
    assert sc.counts.tolist() == [w.tolist() for w in want]


def test_two_ranks_agree_and_a_disagreement_is_detected(tmp_path):
    ids_list = [_ids(r, 40 + r) for r in (1, 3, 5, 12)]
    for rank in (0, 1):
        stats = _stats(tmp_path)
        layer = _layer(rank=rank)
        for ids in ids_list:
            stats.layer_stats(layer, ids).count(ids)
        stats.close()
    ranks = summary.load_dir(str(tmp_path))
    assert [r.rank for r in ranks] == [0, 1]
    verdict = summary.consensus(ranks)
    assert (verdict["compared"], verdict["mismatches"], verdict["ok"]) == (4, 0, True)
    other = tmp_path / "bad"
    for rank in (0, 1):
        stats = _stats(other)
        layer = _layer(rank=rank)
        for n, ids in enumerate(ids_list):
            seen = ids.clone()
            if rank == 1 and n == 2:
                seen[0, 0] = (seen[0, 0] + 1) % E  # rank 1 routed one id differently
            stats.layer_stats(layer, seen).count(seen)
        stats.close()
    verdict = summary.consensus(summary.load_dir(str(other)))
    assert verdict["ok"] is False and verdict["mismatches"] == 1
    assert verdict["first_mismatch_record"] == 2


def test_summary_reports_the_rank_loads_of_the_records(tmp_path):
    stats = _stats(tmp_path)
    models = _models(1)
    ids = _ids(4, 50)
    _drive(stats, [_layer()], models, [(0, ids)])
    stats.close()
    report = summary.summarize(summary.load_dir(str(tmp_path)), window=1)
    (band,) = report["layers"][0]["decode_bands"].values()
    rec = models[0].call(0, ids)
    t = rec[8 : 8 + RANKS].astype(float)
    p = rec[16 : 16 + RANKS].astype(float)
    assert band["steps"] == 1
    assert band["T_max_over_mean"] == pytest.approx(t.max() / t.mean())
    assert band["P_max_over_mean"] == pytest.approx(p.max() / p.mean())


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_owner_table_follows_the_placement_strategy(strategy):
    owner = es.build_owner(NAME, RANKS, E, strategy).tolist()
    if strategy == "linear":
        assert owner == [r for r in range(RANKS) for _ in range(E // RANKS)]
    else:
        assert owner == [e % RANKS for e in range(E)]


def test_owner_table_follows_the_static_placement_file(tmp_path, monkeypatch):
    # rank r owns the experts congruent to r mod 4 (layer index 3 = NAME)
    layers = {"3": [[e for e in range(E) if e % RANKS == r] for r in range(RANKS)]}
    doc = {"version": 1, "num_experts": E, "ep_size": RANKS, "layers": layers}
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(doc))
    from vllm.model_executor.layers.fused_moe import rdna_moe_placement as pl

    monkeypatch.setenv(pl.PLACEMENT_ENV, str(path))
    assert es.build_owner(NAME, RANKS, E, "linear").tolist() == [
        e % RANKS for e in range(E)
    ]
    # a layer the file does not cover keeps the linear placement
    assert es.build_owner(NAME2, RANKS, E, "linear").tolist() == [
        r for r in range(RANKS) for _ in range(E // RANKS)
    ]


def test_a_layer_whose_expert_map_disagrees_with_the_owner_table_is_refused(tmp_path):
    stats = _stats(tmp_path)
    layer = _layer()
    layer.expert_map = torch.full((E,), -1, dtype=torch.int32)  # owns nothing
    with pytest.raises(RuntimeError, match="disagrees"):
        stats.layer_stats(layer, _ids(2))


def test_registration_order_kind_and_geometry_checks(tmp_path):
    stats = _stats(tmp_path)
    first_layer = _layer(NAME)
    first = stats.layer_stats(first_layer, _ids(2))
    mtp_layer = _layer("mtp.layers.0.mlp.experts")  # alive: the registry keys on id()
    mtp = stats.layer_stats(mtp_layer, _ids(2))
    assert (first.pos, mtp.pos) == (0, 1)
    assert es.layer_kind("mtp.layers.0.mlp.experts") == "mtp"
    assert es.layer_kind(NAME) == "target" and es.layer_index(NAME) == 3
    assert stats.layer_stats(first_layer, _ids(3)) is first  # keyed by the object
    wide = _layer(NAME2)
    wide.global_num_experts = E + 4
    with pytest.raises(ValueError, match="experts, top-k"):
        stats.layer_stats(wide, _ids(2))


def test_too_many_ranks_are_refused(tmp_path):
    stats = _stats(tmp_path)
    layer = _layer()
    layer.moe_config.ep_size = es.MAX_RANKS + 1
    with pytest.raises(ValueError, match="up to 8"):
        stats.layer_stats(layer, _ids(2))


def test_tick_drains_snapshots_without_closing(tmp_path):
    stats = _stats(tmp_path, interval_s=1.0)
    _drive(stats, [_layer()], _models(1), [(0, _ids(2, 1))])
    stats.tick(now=stats._start + 0.5)  # not due
    stats.tick(now=stats._start + 1.5)  # due: copies launched, nothing written yet
    assert not list(tmp_path.glob("*.npz"))
    stats.tick(now=stats._start + 1.6)  # the copy is complete: written
    path, loaded = _load(tmp_path)
    assert loaded.calls.sum() == 1
    with np.load(path) as z:
        assert json.loads(str(z["meta"]))["seq"] == 1


def test_close_outside_inference_mode_updates_inference_tensors(tmp_path):
    """The worker registers in inference mode; shutdown runs outside it."""
    stats = _stats(tmp_path)
    with torch.inference_mode():
        handle = stats.layer_stats(_layer(), _ids(3))
        handle.count(_ids(3))
    assert handle.host_state.is_inference()
    stats.close()
    assert _load(tmp_path)[1].calls.sum() == 1


def test_off_without_env(monkeypatch):
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    assert es.EplbStats.from_env() is None
    monkeypatch.setenv(es.STATS_DIR_ENV, "")
    assert es.EplbStats.from_env() is None


def test_module_singleton_is_none_when_env_unset(monkeypatch):
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    reloaded = importlib.reload(es)
    try:
        assert reloaded.COLLECTOR is None and reloaded.ENABLED is False
    finally:
        importlib.reload(es)


def test_env_enables_with_options(monkeypatch, tmp_path):
    monkeypatch.setenv(es.STATS_DIR_ENV, str(tmp_path / "out"))
    monkeypatch.setenv(es.RING_ENV, "128")
    monkeypatch.setenv(es.HIST_ENV, "3")
    monkeypatch.setenv(es.EVERY_ENV, "4")
    monkeypatch.setenv(es.INTERVAL_ENV, "5")
    monkeypatch.setenv(es.DECODE_MAX_ENV, "24")
    monkeypatch.setenv(es.KERNEL_ENV, "torch")
    stats = es.EplbStats.from_env()
    assert stats is not None and (tmp_path / "out").is_dir()
    assert (stats.ring_n, stats.hist, stats.every, stats.interval_s) == (128, 3, 4, 5.0)
    assert stats.edges_host == (1, 2, 3, 4, 6, 8, 12, 16, 24) and not stats.hip


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"every": 3}, "power of two"),
        ({"every": 0}, "power of two"),
        ({"ring": -1}, ">= 0"),
        ({"interval_s": 0}, "positive"),
        ({"decode_max": 0}, "1..4096"),
        ({"kernel": "cuda"}, "auto, hip or torch"),
    ],
)
def test_bad_options_are_refused(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _stats(tmp_path, **kwargs)


def test_hip_kernel_is_refused_when_the_ops_are_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "_have_hip_ops", lambda: False)
    with pytest.raises(RuntimeError, match="not built"):
        _stats(tmp_path, kernel="hip")
    assert _stats(tmp_path, kernel="auto").hip is False


# ---- off by default / zero cost / capture safety ----------------------------


def _guarded_uses(path: str, name: str) -> tuple[int, int]:
    """(attribute uses of ``name``, uses inside an ``is not None`` guard)."""
    tree = ast.parse((ROOT / path).read_text())
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == name
                and isinstance(test.ops[0], ast.IsNot)
            ):
                body = node.body if isinstance(node.body, list) else [node.body]
                for child in body:
                    for sub in ast.walk(child):
                        guarded.add(id(sub))
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == name
    ]
    return len(uses), sum(id(u) in guarded for u in uses)


@pytest.mark.parametrize(
    "path",
    [
        COMPRESSED,
        "vllm/model_executor/layers/quantization/rdna2_moe_resident.py",
        "vllm/v1/worker/gpu_worker.py",
    ],
)
def test_every_collector_use_is_behind_an_is_not_none_guard(path):
    total, guarded = _guarded_uses(path, "_EPLB_STATS")
    assert total >= 1 and total == guarded


@pytest.mark.parametrize(
    "path, minimum",
    [
        (COMPRESSED, 2),
        ("vllm/model_executor/layers/quantization/rdna2_moe_resident.py", 1),
    ],
)
def test_every_layer_handle_use_is_behind_an_is_not_none_guard(path, minimum):
    total, guarded = _guarded_uses(path, "eplb_stats")
    assert total >= minimum and total == guarded


def test_fused_moe_counts_each_call_once():
    """One routing branch counts (fused CTA or standalone); the recursion forwards."""
    source = (ROOT / COMPRESSED).read_text()
    assert source.count("eplb_stats.route_zero") == 1
    assert source.count("eplb_stats.count(topk_ids)") == 1
    assert source.count("eplb_stats=eplb_stats") == 2  # recursion + apply()
    # the expert-cache pass loop never passes it: those calls are not counted
    cached = source.split("def _apply_cached_moe", 1)[1]
    cached = cached.split("def _rdna2_fused_moe", 1)[0]
    assert "eplb_stats" not in cached


def test_k2_stands_down_only_through_the_import_time_flag():
    source = (ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/k2.py").read_text()
    assert "and not EPLB_STATS_ENABLED" in source
    assert source.count("EPLB_STATS_ENABLED") == 2  # import + the one gate


def test_the_torch_formulation_is_capture_safe_by_construction():
    """No host synchronization, dynamic shapes or allocation of persistent state."""
    source = inspect.getsource(es.LayerStats.count_torch)
    for forbidden in (
        ".item(",
        ".tolist(",
        ".cpu(",
        ".numpy(",
        "bincount",
        "nonzero",
        "unique",
        "synchronize",
        "masked_select",
        "print(",
        "logger.",
    ):
        assert forbidden not in source, forbidden
    assert source.count("index_add_") >= 5
    tree = ast.parse(textwrap.dedent(inspect.getsource(es.EplbStats.tick)))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "synchronize" not in names and "item" not in names


# ---- the C++ kernel and the Python side agree on the contract ----------------


def _cu_text() -> str:
    return CU.read_text()


@pytest.mark.parametrize(
    "cname, value",
    [
        ("kEplbWords", es.WORDS),
        ("kEplbMaxRanks", es.MAX_RANKS),
        ("kEplbMaxBands", es.MAX_BANDS),
    ],
)
def test_cu_constants_equal_the_python_constants(cname, value):
    match = re.search(rf"constexpr int {cname} = (\d+);", _cu_text())
    assert match is not None and int(match.group(1)) == value


def _eval_cxx(expr: str, scope: dict) -> int:
    expr = " ".join(expr.replace("(int64_t)", "").replace("LL", "").split())
    return int(eval(expr, {"__builtins__": {}}, scope))  # our own source text


@pytest.mark.parametrize("hist_n", [0, 3])
def test_cu_state_layout_equals_the_python_offsets(hist_n):
    text = _cu_text()
    body = text.split("__device__ void eplb_stats_cta", 1)[1]
    names = {
        "off_touch": "touch",
        "off_calls": "calls",
        "off_rows": "rows",
        "off_ord": "ordinal",
        "off_hcalls": "hist_calls",
        "off_hrows": "hist_rows",
    }
    nb = 6
    scope: dict = {"E": E, "nb": nb}
    for match in re.finditer(r"const int64_t (off_\w+) = ([^;]+);", body):
        scope[match.group(1)] = _eval_cxx(match.group(2), scope)
    py = es.offsets(E, nb, hist_n)
    for cname, field in names.items():
        assert scope[cname] == getattr(py, field), cname
    need = re.search(r"const int64_t need =([^;]+);", text, re.DOTALL)
    assert need is not None
    size = _eval_cxx(need.group(1), {"num_experts": E, "nb": nb, "hist_n": hist_n})
    assert size == py.size


def test_cu_hash_constants_equal_the_python_hash():
    text = _cu_text()
    fn = text.split("eplb_mix(unsigned i, int e)", 1)[1].split("\n}\n", 1)[0]
    assert [c.lower() for c in re.findall(r"0x[0-9A-Fa-f]+u", fn)] == [
        "0x9e3779b1u",
        "0x85ebca6bu",
        "0x2c1b3c6du",
    ]
    assert re.findall(r">> (\d+)", fn) == ["15", "12"]
    positions = torch.arange(6, dtype=torch.int64)
    ids = torch.tensor([0, 5, 511, -1, 7, 1023], dtype=torch.int64)
    got = es.mix_hash(positions, ids).tolist()
    assert got == [_mix(i, e) for i, e in zip(range(6), ids.tolist())]


def test_every_hip_op_is_declared_registered_and_wrapped():
    root = ROOT / "csrc/rocm"
    ops_h = (root / "ops.h").read_text()
    bindings = (root / "torch_bindings.cpp").read_text()
    custom = (ROOT / "vllm/_custom_ops.py").read_text()
    for op in ("rdna_moe_route_zero_stats", "rdna_eplb_stats"):
        assert op in ops_h
        assert f'"{op}(' in bindings and f'rocm_ops.impl("{op}"' in bindings
        assert f"def {op}(" in custom


# ---- GPU: the HIP kernel against the torch formulation ----------------------

needs_hip = pytest.mark.skipif(
    not torch.cuda.is_available() or not es._have_hip_ops(),
    reason="needs a GPU and the built rdna_eplb_stats ops",
)


def _gpu_calls():
    gen = torch.Generator().manual_seed(7)
    for rows in (1, 3, 6, 8, 9, 24, 100, 300):
        ids = torch.rand(rows, E, generator=gen).topk(K, dim=1).indices.to(torch.int32)
        if rows > 2:
            ids[0, 0] = -1  # an invalid id and a padded tail
            ids[rows - 1, K - 1] = E + 5
        padding = torch.zeros(rows, dtype=torch.bool)
        padding[rows // 2 :] = rows > 2
        yield ids, padding


@needs_hip
@pytest.mark.parametrize("every", [1, 2, 4])  # 2 and 4: the unsampled early return
@pytest.mark.parametrize("fused", [False, True])
def test_hip_counters_match_the_torch_formulation_on_the_gpu(tmp_path, fused, every):
    from vllm import _custom_ops as ops

    dev = torch.device("cuda")
    ref = _stats(tmp_path / "ref", hist=2, every=every)
    hip = _stats(tmp_path / "hip", hist=2, kernel="hip", every=every)
    layer = _layer()  # the registry is keyed by id(layer): one object for every call
    for ids, padding in list(_gpu_calls()) * 2:
        ids, padding = ids.to(dev), padding.to(dev)
        with mock.patch.object(es, "_padding_rows", lambda n, d, p=padding: p):
            ref.layer_stats(layer, ids).count_torch(ids)
            handle = hip.layer_stats(layer, ids)
            if fused and ids.shape[0] * K <= 1024:
                w1 = torch.full((ids.numel(), 64), 7, dtype=torch.float16, device=dev)
                out = torch.full((ids.shape[0], 32), 7, dtype=torch.float16, device=dev)
                want = ops.rdna_moe_route_zero(ids, 4, E, None, w1.clone(), out.clone())
                got = handle.route_zero(ids, 4, E, None, w1, out)
                assert all(torch.equal(a, b) for a, b in zip(want, got))
                assert not w1.any() and not out.any()
            else:
                handle.count(ids)
    assert len(ref._order) == len(hip._order) == 1
    a, b = ref._order[0], hip._order[0]
    assert torch.equal(a.state.cpu(), b.state.cpu())
    assert torch.equal(a.hist.cpu(), b.hist.cpu())
    n = int(ref.gseq.item())
    assert n == int(hip.gseq.item())
    keep = [i for i in range(es.WORDS) if i not in (4, 5)]  # the HIP clock words
    assert torch.equal(ref.ring.cpu()[:n][:, keep], hip.ring.cpu()[:n][:, keep])


@needs_hip
def test_hip_counters_survive_cuda_graph_replay_on_the_gpu(tmp_path):
    dev = torch.device("cuda")
    stats = _stats(tmp_path, kernel="hip")
    ids = _ids(5, 1).to(dev)
    handle = stats.layer_stats(_layer(), ids)
    handle.count(ids)  # eager: registers and compiles
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        handle.count(ids)  # warm-up of the captured code path
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        handle.count(ids)
    for _ in range(4):
        graph.replay()
    calls = 1 + 1 + 4  # eager, warm-up, replays (the capture itself does not run)
    assert handle.state[handle.off.ordinal].item() == calls
    assert int(stats.gseq.item()) == calls
    want = calls * np.bincount(ids.cpu().numpy().ravel(), minlength=E)
    assert handle.state[:E].cpu().tolist() == want.tolist()  # 5 rows: decode phase


@needs_hip
def test_a_layer_first_seen_inside_a_capture_is_skipped(tmp_path):
    dev = torch.device("cuda")
    stats = _stats(tmp_path, kernel="hip")
    ids = _ids(5, 1).to(dev)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.zeros(1, device=dev)
    torch.cuda.current_stream().wait_stream(side)
    layer = _layer()
    with torch.cuda.graph(graph):
        assert stats.layer_stats(layer, ids) is None
    assert stats.layer_stats(layer, ids) is None  # remembered, not retried
