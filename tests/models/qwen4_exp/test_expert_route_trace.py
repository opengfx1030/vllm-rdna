# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route receipts retain complete demand and explicit workload scopes."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_RDNA_EXPERT_ROUTE_DIR", str(tmp_path))
    return _load(
        "route_trace_test",
        _ROOT / ("vllm/model_executor/layers/fused_moe/rdna_expert_route_trace.py"),
    )


@pytest.fixture
def sweep():
    return _load("route_sweep_test", _ROOT / "tools/rdna/mono/sweep_offload_policy.py")


def test_disabled_recorder_and_helper_touch_no_inputs_or_runtime(trace):
    trace.ENABLED = False
    assert trace.record_routes(None, None, None, phase=None, rank=None) is False
    assert trace.record_cache_routes(None, None, None, None) is False


def test_complete_routes_bounded_calls_and_count_manifest(
    trace, sweep, tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_RDNA_EXPERT_ROUTE_MAX_CALLS", "2")
    routes = [0, 0, 1, 2, 3, -1, -256, 256]
    for _ in range(2):
        assert trace.record_routes(
            "model.layers.7.mlp.experts",
            routes,
            4,
            phase="decode",
            rank=1,
            expert_bytes=100,
        )
    assert not trace.record_routes(
        "model.layers.7.mlp.experts",
        routes,
        4,
        phase="decode",
        rank=1,
        expert_bytes=100,
    )
    records = sweep.read_calls(tmp_path, local_experts=4)
    assert records[(1, 7)] == [[0, 0, 1, 2, 3, -1, -1, -1]] * 2
    counts = sweep.simulate(records[(1, 7)], 2, "lru", local_experts=4)
    assert counts["expert_demands"] == counts["hits"] + counts["misses"] == 8
    assert counts["routed_pairs"] == 10 and counts["passes"] == 4
    manifest = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    assert manifest["recorded_calls"] == 2
    assert manifest["routed_pairs"] == 16 and manifest["local_pairs"] == 10
    assert manifest["expert_bytes"] == 100


def test_phase_scope_never_silently_mixes(trace, sweep, tmp_path):
    for phase in ("prefill", "decode", "profile"):
        trace.record_routes(
            "model.layers.0.mlp.experts",
            [0, 1],
            256,
            phase=phase,
            rank=0,
            expert_bytes=10,
        )
    with pytest.raises(ValueError, match="mixed route phases"):
        sweep.read_calls(tmp_path)
    records = sweep.read_calls(tmp_path, phase="prefill")
    report = sweep.sweep(records, [1], ["lru"])
    assert report["trace_phase"] == "prefill"
    assert report["excluded_trace_phases"] == ("decode", "profile")
    assert report["scope"].startswith("prefill-only")
    plan = sweep.plan_slots(
        records, {(0, 0): {"expert_bytes": 10}}, {0: 1034}, [1], baseline_capacity=1
    )
    assert plan["trace_phase"] == "prefill"


@pytest.mark.parametrize(
    "mutation",
    ["missing_manifest", "count", "bool_id", "duplicate_call", "wrong_geometry"],
)
def test_incomplete_or_invalid_jsonl_fails_closed(trace, sweep, tmp_path, mutation):
    for _ in range(2):
        trace.record_routes(
            "model.layers.0.mlp.experts", [0, 1], 256, phase="decode", rank=0
        )
    path = next(tmp_path.glob("*.jsonl"))
    manifest = next(tmp_path.glob("*.manifest.json"))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_manifest":
        manifest.unlink()
    elif mutation == "count":
        data = json.loads(manifest.read_text())
        data["recorded_calls"] = 1
        manifest.write_text(json.dumps(data))
    elif mutation == "bool_id":
        lines[1]["ids"] = [True]
    elif mutation == "duplicate_call":
        lines[2]["call"] = 0
    elif mutation == "wrong_geometry":
        lines[0]["logical_experts"] = 128
    elif mutation == "bool_version":
        lines[0]["version"] = True
    elif mutation == "bad_layer_name":
        lines[0]["layer_name"] = "model.layers.99.mlp.experts"
    else:
        lines[1]["call"] = False
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    with pytest.raises(ValueError):
        sweep.read_calls(tmp_path)


def test_jsonl_and_legacy_pt_rejected_before_loading_torch(trace, sweep, tmp_path):
    trace.record_routes("model.layers.0.mlp.experts", [0], 256, phase="decode", rank=0)
    (tmp_path / "rank0-layer0-call0.pt").write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="mixed .pt and JSONL"):
        sweep.read_calls(tmp_path)


def test_cpu_integer_contract_rejects_tensor_like_without_iteration(trace):
    class Poison:
        def __iter__(self):
            raise AssertionError("must not implicitly iterate a device object")

    with pytest.raises(ValueError, match="CPU Python list"):
        trace.record_routes(
            "model.layers.0.mlp.experts", Poison(), 256, phase="decode", rank=0
        )
    with pytest.raises(ValueError, match="CPU Python integers"):
        trace.record_routes(
            "model.layers.0.mlp.experts", [True], 256, phase="decode", rank=0
        )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("decode", "decode"),
        ("prefill", "prefill"),
        ("verify", "decode"),
        ("capture", None),
        ("missing", None),
        ("profile", None),
        ("ambiguous", None),
    ],
)
def test_runtime_helper_gates_and_classifies_without_device_reads(
    trace, monkeypatch, tmp_path, case, expected
):
    metadata = SimpleNamespace(num_decodes=1)
    if case == "prefill":
        metadata = SimpleNamespace(num_prefills=1)
    elif case == "verify":
        metadata = SimpleNamespace(num_spec_decodes=1)
    elif case == "ambiguous":
        metadata = SimpleNamespace(num_actual_tokens=3)
    context = SimpleNamespace(
        attn_metadata={"attn": metadata}, is_profile=case == "profile"
    )
    capture = SimpleNamespace(replaying=case != "capture")
    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.breakable_cudagraph",
        SimpleNamespace(
            BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: capture)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: context,
            is_forward_context_available=lambda: case != "missing",
        ),
    )
    cache = SimpleNamespace(logical_num_experts=256, _expert_bytes=100)
    assert trace.record_cache_routes(
        cache, "model.layers.2.mlp.experts", [0, 1], 0
    ) is (expected is not None)
    manifests = list(tmp_path.glob("*.manifest.json"))
    if expected:
        assert json.loads(manifests[0].read_text())["phase"] == expected
    else:
        assert manifests == []


def test_arm_file_and_phase_filter_exclude_unmeasured_work(
    trace, tmp_path, monkeypatch
):
    arm = tmp_path / "armed"
    monkeypatch.setenv("VLLM_RDNA_EXPERT_ROUTE_ARM_FILE", str(arm))
    monkeypatch.setenv("VLLM_RDNA_EXPERT_ROUTE_PHASE", "decode")
    kwargs = dict(
        layer_name="model.layers.0.mlp", local_ids=[0], logical_experts=4, rank=0
    )
    assert not trace.record_routes(**kwargs, phase="decode")
    arm.touch()
    assert not trace.record_routes(**kwargs, phase="prefill")
    assert trace.record_routes(**kwargs, phase="decode")
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
