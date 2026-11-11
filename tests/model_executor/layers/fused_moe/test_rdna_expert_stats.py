# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in MoE routing-statistics collector.

The device work is plain tensor ops, so the CPU run exercises the same code as
a GPU run; capture safety and zero cost when off are proven by inspecting the
source (no host sync inside ``record``; every hook sits behind ``is not None``).
"""

import ast
import importlib
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import rdna_expert_stats as es

E = 16
K = 4
ROOT = Path(__file__).resolve().parents[4]


def _collector(tmp_path, **kwargs):
    kwargs.setdefault("interval_s", 1.0)
    return es.ExpertStatsCollector(str(tmp_path), **kwargs)


def _record(collector, ids, name="model.layers.3.mlp.experts", tp_rank=0):
    collector.record(
        name,
        ids,
        num_experts=E,
        top_k=K,
        tp_rank=tp_rank,
        tp_size=4,
        dp_rank=0,
    )


def _ids(rows, seed=0):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, E, generator=gen)
    return scores.topk(K, dim=1).indices.to(torch.int32)


def _read(tmp_path):
    (path,) = sorted(tmp_path.glob("expert_stats-dp0-pid*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_counts_match_reference_and_split_phases(tmp_path):
    collector = _collector(tmp_path, decode_max_tokens=8)
    decode, prefill = _ids(4, 1), _ids(20, 2)
    _record(collector, decode)
    _record(collector, decode)
    _record(collector, prefill)
    state = collector._layers["model.layers.3.mlp.experts"]
    want_dec = 2 * np.bincount(decode.numpy().ravel(), minlength=E)
    want_pre = np.bincount(prefill.numpy().ravel(), minlength=E)
    assert state.counts[0, :E].tolist() == want_dec.tolist()
    assert state.counts[1, :E].tolist() == want_pre.tolist()
    assert state.counts[:, E].tolist() == [0, 0]  # no invalid pairs
    assert state.counts[:, E + 1].tolist() == [2, 1]  # calls
    assert state.rows_hist[4].item() == 2 and state.rows_hist.sum().item() == 2


def test_invalid_ids_and_padding_rows_are_not_counted(tmp_path):
    collector = _collector(tmp_path)
    ids = _ids(6, 3)
    ids[4:] = -1  # rows the router masked as padding
    ids[0, 0] = E + 7  # out of range (e.g. an appended fused shared expert)
    _record(collector, ids)
    state = collector._layers["model.layers.3.mlp.experts"]
    valid = ids[(ids >= 0) & (ids < E)]
    assert (
        state.counts[0, :E].tolist() == np.bincount(valid.numpy(), minlength=E).tolist()
    )
    assert state.counts[0, E].item() == 2 * K + 1
    assert state.rows_hist[4].item() == 1  # two padded rows are not valid rows


def test_padding_mask_from_forward_context(tmp_path, monkeypatch):
    collector = _collector(tmp_path)
    padding = torch.tensor([False, False, True, True])
    monkeypatch.setattr(es, "_padding_rows", lambda rows, device: padding)
    _record(collector, _ids(4, 4))
    state = collector._layers["model.layers.3.mlp.experts"]
    assert state.counts[0, :E].sum().item() == 2 * K
    assert state.counts[0, E].item() == 2 * K


def test_extra_columns_beyond_top_k_are_dropped(tmp_path):
    collector = _collector(tmp_path)
    ids = torch.cat([_ids(3, 5), torch.full((3, 1), E, dtype=torch.int32)], dim=1)
    _record(collector, ids)
    state = collector._layers["model.layers.3.mlp.experts"]
    assert state.counts[0, :E].sum().item() == 3 * K
    assert state.counts[0, E].item() == 0


def test_non_zero_tp_rank_records_nothing(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, _ids(4), tp_rank=1)
    assert not collector._layers


def test_all_ranks_debug_option_records_every_rank_and_writes_the_rank(tmp_path):
    collector = _collector(tmp_path, all_ranks=True)
    _record(collector, _ids(4), tp_rank=2)
    assert collector._layers and collector._tp_rank == 2
    collector.close()
    meta = json.loads((next(tmp_path.glob("*.jsonl"))).read_text().splitlines()[0])
    assert meta["tp_rank"] == 2


def test_pairs_count_unordered_coactivation(tmp_path):
    collector = _collector(tmp_path, pairs=True)
    ids = torch.tensor([[1, 5, 9, 2], [5, 1, 2, 3]], dtype=torch.int32)
    _record(collector, ids)
    state = collector._layers["model.layers.3.mlp.experts"]
    pairs = state.pair_counts[: E * E].reshape(E, E)
    assert pairs[1, 5].item() == 2 and pairs[5, 1].item() == 0
    assert pairs[2, 9].item() == 1 and pairs[1, 2].item() == 2
    assert pairs.sum().item() == 2 * (K * (K - 1) // 2)
    assert state.pair_counts[E * E].item() == 0


def test_tick_drains_cumulative_snapshots_and_close_is_final(tmp_path):
    collector = _collector(tmp_path, interval_s=10.0)
    ids = _ids(4, 6)
    _record(collector, ids)
    collector.tick(now=collector._start + 1)  # interval not up: nothing written
    assert not list(tmp_path.glob("*.jsonl"))
    collector.tick(now=collector._start + 11)  # launches the copy
    collector.tick(now=collector._start + 12)  # CPU copy is complete: write
    _record(collector, ids)
    collector.close()
    lines = _read(tmp_path)
    assert [x["type"] for x in lines] == ["meta", "snapshot", "snapshot"]
    meta, first, last = lines
    assert meta["schema"] == es.SCHEMA and meta["top_k"] == K
    assert meta["layers"] == [
        {"name": "model.layers.3.mlp.experts", "idx": 3, "kind": "target"}
    ]
    name = "model.layers.3.mlp.experts"
    one = np.bincount(ids.numpy().ravel(), minlength=E)
    assert first["counts"][name]["decode"] == one.tolist()
    assert last["counts"][name]["decode"] == (2 * one).tolist()
    assert last["tokens"][name] == {"decode": 8, "prefill": 0}
    assert last["steps"][name] == {"decode": 2, "prefill": 0}
    assert last["rows_hist"][name][4] == 2
    assert last["seq"] == 2 and last["invalid"][name] == 0


def test_new_layer_writes_a_new_meta_and_mtp_kind(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, _ids(2), name="model.layers.0.mlp.experts")
    collector.tick(now=collector._start + 2)  # launches the copy
    collector.tick(now=collector._start + 3)  # writes it (meta + snapshot)
    _record(collector, _ids(2), name="mtp.layers.0.mlp.experts")
    collector.close()
    metas = [x for x in _read(tmp_path) if x["type"] == "meta"]
    assert len(metas) == 2
    kinds = {x["name"]: x["kind"] for x in metas[-1]["layers"]}
    assert kinds == {
        "model.layers.0.mlp.experts": "target",
        "mtp.layers.0.mlp.experts": "mtp",
    }
    assert metas[-1]["model"]["num_layers"] == 1  # MTP layers are not target layers


def test_pairs_npz_written_at_close(tmp_path):
    collector = _collector(tmp_path, pairs=True)
    ids = _ids(5, 7)
    _record(collector, ids)
    collector.close()
    (path,) = tmp_path.glob("*.pairs.npz")
    data = np.load(path)
    assert data["names"].tolist() == ["model.layers.3.mlp.experts"]
    assert data["pairs"].shape == (1, E, E)
    assert data["pairs"].sum() == 5 * (K * (K - 1) // 2)


def test_registration_during_capture_is_refused(tmp_path, monkeypatch):
    collector = _collector(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA graph"):
        collector._register(
            "model.layers.0.mlp.experts", E, K, torch.device("cuda"), 4, 0, 0
        )


def test_mismatched_layer_geometry_is_refused(tmp_path):
    collector = _collector(tmp_path)
    _record(collector, _ids(2), name="model.layers.0.mlp.experts")
    with pytest.raises(ValueError, match="top-k"):
        collector.record(
            "model.layers.1.mlp.experts",
            _ids(2),
            num_experts=E,
            top_k=K + 1,
            tp_rank=0,
            tp_size=4,
            dp_rank=0,
        )


# ---- off by default / zero cost / capture safety ---------------------------


def test_off_without_env(monkeypatch):
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    assert es.ExpertStatsCollector.from_env() is None
    monkeypatch.setenv(es.STATS_DIR_ENV, "")
    assert es.ExpertStatsCollector.from_env() is None


def test_module_level_singleton_is_none_when_env_unset(monkeypatch):
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    reloaded = importlib.reload(es)
    try:
        assert reloaded.COLLECTOR is None and reloaded.ENABLED is False
    finally:
        importlib.reload(es)


def test_env_enables_with_options(monkeypatch, tmp_path):
    monkeypatch.setenv(es.STATS_DIR_ENV, str(tmp_path / "out"))
    monkeypatch.setenv(es.INTERVAL_ENV, "5")
    monkeypatch.setenv(es.PAIRS_ENV, "1")
    monkeypatch.setenv(es.DECODE_MAX_ENV, "24")
    collector = es.ExpertStatsCollector.from_env()
    assert collector is not None
    assert (collector.interval_s, collector.pairs, collector.decode_max_tokens) == (
        5.0,
        True,
        24,
    )
    assert (tmp_path / "out").is_dir()


def _guarded_uses(path: str, name: str) -> tuple[int, int]:
    """(attribute uses of ``name``, uses inside ``if name is not None``)."""
    tree = ast.parse((ROOT / path).read_text())
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == name
                and isinstance(test.ops[0], ast.IsNot)
            ):
                for child in node.body:
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
        "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
        "vllm/v1/worker/gpu_worker.py",
    ],
)
def test_every_hook_is_behind_an_is_not_none_guard(path):
    total, guarded = _guarded_uses(path, "_EXPERT_STATS")
    assert total >= 1 and total == guarded


def test_k2_stands_down_only_through_the_import_time_flag():
    source = (ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/k2.py").read_text()
    assert "and not EXPERT_STATS_ENABLED" in source
    assert source.count("EXPERT_STATS_ENABLED") == 2  # import + the one gate


def test_record_is_capture_safe_by_construction():
    """No host synchronization, dynamic shapes or allocation of persistent state."""
    source = inspect.getsource(es.ExpertStatsCollector.record)
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
    # Fixed-shape device ops only: scatter-adds into preallocated counters.
    assert source.count("index_add_") >= 2
    tree = ast.parse(textwrap.dedent(inspect.getsource(es.ExpertStatsCollector.tick)))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "synchronize" not in names and "item" not in names


def test_env_selects_the_fused_kernel_unless_torch_is_asked(monkeypatch, tmp_path):
    monkeypatch.setenv(es.STATS_DIR_ENV, str(tmp_path))
    fused = es.ExpertStatsCollector.from_env()
    assert fused is not None and fused._record_fused is not None
    monkeypatch.setenv(es.KERNEL_ENV, "torch")
    torch_only = es.ExpertStatsCollector.from_env()
    assert torch_only is not None and torch_only._record_fused is None
    monkeypatch.setenv(es.KERNEL_ENV, "fused")
    monkeypatch.setenv(es.DECODE_MAX_ENV, "128")
    with pytest.raises(ValueError, match="up to 64 rows"):
        es.ExpertStatsCollector.from_env()


def test_trace_ring_keeps_the_newest_decode_calls(tmp_path):
    collector = _collector(
        tmp_path, decode_max_tokens=8, trace_layers=frozenset({3}), trace_steps=4
    )
    calls = [
        _ids(rows, seed=rows * 10 + i) for i, rows in enumerate([2, 3, 1, 5, 4, 2])
    ]
    calls[3][2, :] = -1  # a masked row is stored as -1
    for ids in calls:
        _record(collector, ids)
    _record(collector, _ids(20, 9))  # a prefill call is not traced
    _record(collector, _ids(3, 8), name="model.layers.4.mlp.experts")  # not selected
    assert collector._layers["model.layers.4.mlp.experts"].trace_ids is None
    collector.close()
    (path,) = tmp_path.glob("expert_stats-dp0-pid*.trace.npz")
    data = np.load(path)
    assert data["names"].tolist() == ["model.layers.3.mlp.experts"]
    assert data["calls"].tolist() == [6] and int(data["max_rows"]) == 8
    ids, rows = data["ids"][0], data["rows"][0]
    assert ids.shape == (4, 8, K) and ids.dtype == np.int16
    for call in (2, 3, 4, 5):  # the ring holds calls 2..5, call c in slot c % 4
        slot, want = call % 4, calls[call].numpy()
        assert rows[slot] == len(want)
        assert (ids[slot, : len(want)] == want).all()
    # slot 0 holds the 4-row call 4 (it replaced the 2-row call 0), slot 1 call 5
    assert rows[0] == 4 and rows[1] == 2


def test_trace_masks_padding_and_env_parses_layers(tmp_path, monkeypatch):
    assert es.parse_layer_list("0, 24,47") == frozenset({0, 24, 47})
    assert es.parse_layer_list("") == frozenset()
    collector = _collector(
        tmp_path, decode_max_tokens=8, trace_layers=frozenset({3}), trace_steps=2
    )
    padding = torch.tensor([False, False, True])
    monkeypatch.setattr(es, "_padding_rows", lambda n, d: padding)
    _record(collector, _ids(3, 5))
    collector.close()
    (path,) = tmp_path.glob("expert_stats-dp0-pid*.trace.npz")
    got = np.load(path)["ids"][0, 0, :3]
    assert (got[2] == -1).all() and (got[:2] >= 0).all()
    monkeypatch.setenv(es.STATS_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(es.TRACE_LAYERS_ENV, "3 7")
    monkeypatch.setenv(es.TRACE_STEPS_ENV, "16")
    from_env = es.ExpertStatsCollector.from_env()
    assert from_env is not None
    assert from_env.trace_layers == frozenset({3, 7}) and from_env.trace_steps == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fused", [False, True])
def test_counters_and_trace_survive_cuda_graph_replay_on_the_gpu(tmp_path, fused):
    """One eager warm-up call, one captured call, replayed: every replay counts and
    advances the trace slot (the slot comes from the device-side call counter)."""
    dev = torch.device("cuda")
    collector = _collector(
        tmp_path,
        decode_max_tokens=64,
        fused=fused,
        trace_layers=frozenset({3}),
        trace_steps=4,
    )
    ids = _ids(5, 1).to(dev)
    _record(collector, ids)  # registers the layer, compiles the kernels
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _record(collector, ids)  # warm-up of the captured code path
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        _record(collector, ids)
    for _ in range(4):
        graph.replay()
    state = collector._layers["model.layers.3.mlp.experts"]
    calls = 1 + 1 + 4  # eager, warm-up, replays (capture itself does not execute)
    assert state.counts[0, E + 1].cpu().item() == calls
    want = calls * np.bincount(ids.cpu().numpy().ravel(), minlength=E)
    assert state.counts[0, :E].cpu().tolist() == want.tolist()
    assert state.trace_rows is not None and state.trace_ids is not None
    assert state.trace_rows.cpu().tolist() == [5, 5, 5, 5]  # 6 calls fill the ring
    ring = state.trace_ids.cpu().numpy().reshape(4, 64, K)
    for slot in range(4):
        assert (ring[slot, :5] == ids.cpu().numpy()).all()
        assert (ring[slot, 5:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("pairs", [False, True])
def test_fused_kernel_matches_the_torch_reference_on_the_gpu(
    tmp_path, monkeypatch, pairs
):
    """Same counters from both formulations, invalid ids and padding rows included."""
    dev = torch.device("cuda")
    ref = _collector(tmp_path / "ref", decode_max_tokens=64, pairs=pairs)
    fus = _collector(tmp_path / "fus", decode_max_tokens=64, pairs=pairs, fused=True)
    gen = torch.Generator().manual_seed(7)
    for rows in (1, 3, 6, 24, 64, 65, 300):
        ids = torch.rand(rows, E, generator=gen).topk(K, dim=1).indices.to(torch.int32)
        ids[0, 0] = -1  # invalid ids
        ids[rows - 1, K - 1] = E + 5
        ids = ids.to(dev)
        padding = torch.zeros(rows, dtype=torch.bool, device=dev)
        padding[rows // 2 :] = rows > 2  # padded tail
        monkeypatch.setattr(es, "_padding_rows", lambda n, d, p=padding: p)
        for collector in (ref, fus):
            _record(collector, ids)
    name = "model.layers.3.mlp.experts"
    a, b = ref._layers[name], fus._layers[name]
    assert torch.equal(a.counts.cpu(), b.counts.cpu())
    assert torch.equal(a.rows_hist.cpu(), b.rows_hist.cpu())
    if pairs:
        assert torch.equal(a.pair_counts.cpu(), b.pair_counts.cpu())


def test_close_outside_inference_mode_updates_inference_tensors(tmp_path):
    """The worker allocates the counters in inference mode; shutdown is outside it
    (the T10 run: 'Inplace update to inference tensor outside InferenceMode')."""
    collector = _collector(tmp_path, decode_max_tokens=8, pairs=True)
    with torch.inference_mode():
        _record(collector, _ids(4, 1))
    state = collector._layers["model.layers.3.mlp.experts"]
    assert state.host_counts.is_inference()
    collector.close()
    last = _read(tmp_path)[-1]
    assert last["type"] == "snapshot"
    assert sum(last["counts"]["model.layers.3.mlp.experts"]["decode"]) == 4 * K
