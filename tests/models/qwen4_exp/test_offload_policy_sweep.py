# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "tools/rdna/mono/sweep_offload_policy.py"
_SPEC = importlib.util.spec_from_file_location("policy_sweep", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_counts_unique_transfers_but_observes_duplicate_routes():
    counts = sweep.simulate([[0, 0, 1, -1, 256], [0, 1], [2, 3]], 2, "lfu_lru")
    assert counts == dict(
        calls=3,
        routed_pairs=7,
        expert_demands=6,
        hits=2,
        misses=4,
        passes=3,
        copied_expert_units=4,
    )


def test_overflow_and_empty_calls_preserve_accounting():
    counts = sweep.simulate([[0, 1, 2], [], [0, 1, 2]], 2, "lru")
    assert counts["passes"] == 4
    assert counts["hits"] + counts["misses"] == 6
    assert counts["calls"] == 3


def test_rank_mapping_numeric_call_order_and_safe_load(tmp_path):
    for call, ids in [(10, [257]), (2, [256, 257, 511, 0])]:
        torch.save(
            dict(
                rank=1, layer=7, call=call, tensors={"route_ids": torch.tensor([ids])}
            ),
            tmp_path / f"rank1-layer7-call{call}.pt",
        )
    records = sweep.read_calls(tmp_path)
    assert records[(1, 7)] == [[0, 1, 255, -256], [1]]
    report = sweep.sweep(records, [2], ["lru"], payload_bytes=100)
    result = report["results"][0]
    assert result["modeled_copy_bytes"] == result["totals"]["misses"] * 100
    assert result["per_rank"]["1"]["calls"] == 2


def test_invalid_global_ids_fail(tmp_path):
    torch.save(
        dict(rank=0, layer=1, call=1, tensors={"route_ids": torch.tensor([512])}),
        tmp_path / "rank0-layer1-call1.pt",
    )
    with pytest.raises(ValueError, match="invalid global"):
        sweep.read_calls(tmp_path)


def _profiles(keys, expert_bytes=10, **kwargs):
    return {key: dict(expert_bytes=expert_bytes, **kwargs) for key in keys}


def test_nonuniform_plan_spends_only_on_measured_savings_and_accounts_maps():
    records = {(0, 0): [[0], [1], [0], [1]], (0, 1): [[0]] * 4}
    profiles = _profiles(records)
    profiles[0, 1]["expert_bytes"] = 100
    report = sweep.plan_slots(
        records,
        profiles,
        {0: 152},
        [1, 2],
        policy="lru",
        baseline_capacity=2,
        local_experts=4,
    )
    rank = report["per_rank"]["0"]
    assert report["layer_slots"] == {"0": {"0": 2, "1": 1}}
    assert rank["layer_slots"] == {"0": 2, "1": 1}
    assert rank["map_bytes"] == 32
    assert rank["payload_bytes"] == 120 and rank["used_bytes"] == 152
    assert rank["modeled_copy_bytes"] == 120
    assert rank["gpu_bytes_saved_vs_baseline"] == 100
    assert rank["modeled_copy_bytes_saved_vs_baseline"] == 0
    assert rank["baseline"]["feasible"] is False
    assert "not a global optimum" in report["scope"]


@pytest.mark.parametrize("weighted", [False, True])
def test_plan_ties_and_supplied_weights_are_deterministic(weighted):
    calls = [[0], [1], [0], [1]]
    records = {(0, 7): calls, (0, 2): calls}
    profiles = _profiles(records, map_bytes=0)
    if weighted:
        profiles[0, 7]["objective_weight"] = 2
    kwargs = dict(
        capacities=[2, 1, 2], policy="lru", baseline_capacity=2, local_experts=4
    )
    first = sweep.plan_slots(records, profiles, {0: 30}, **kwargs)
    second = sweep.plan_slots(
        dict(reversed(list(records.items()))),
        dict(reversed(list(profiles.items()))),
        {0: 30},
        **kwargs,
    )
    assert first == second
    assert first["per_rank"]["0"]["layer_slots"] == (
        {"2": 1, "7": 2} if weighted else {"2": 2, "7": 1}
    )


def test_rank_budgets_cannot_borrow_and_overflow_routes_are_retained():
    records = {(0, 0): [[0, 0, 1, 2, -1, 99]], (1, 0): [[0], [1], [0], [1]]}
    profiles = _profiles(records, map_bytes=4)
    result = sweep.plan_slots(
        records,
        profiles,
        {0: 14, 1: 100},
        [1, 2, 3],
        baseline_capacity=3,
        local_experts=4,
    )
    assert result["per_rank"]["0"]["layer_slots"] == {"0": 1}
    for rank in result["per_rank"].values():
        assert rank["used_bytes"] <= rank["budget_bytes"]
    layer = result["per_layer"][0]
    assert layer["curve"][0]["passes"] == 3
    for point in layer["curve"]:
        assert point["expert_demands"] == point["hits"] + point["misses"] == 3
        assert point["routed_pairs"] == 4


def test_available_profile_capacities_and_bounds_are_enforced():
    records = {(0, 0): [[0], [1], [0], [1]]}
    profiles = _profiles(
        records, map_bytes=0, min_capacity=2, max_capacity=3, capacities=[1, 2, 4]
    )
    rank = sweep.plan_slots(
        records, profiles, {0: 40}, [1, 3], baseline_capacity=3, local_experts=4
    )["per_rank"]["0"]
    assert rank["layer_slots"] == {"0": 2}
    assert rank["baseline"]["capacities_available"] is False
    with pytest.raises(ValueError, match="minimum plan needs"):
        sweep.plan_slots(
            records, profiles, {0: 19}, [1, 3], local_experts=4, baseline_capacity=3
        )


@pytest.mark.parametrize(
    "change",
    [
        {"expert_bytes": 0},
        {"expert_bytes": True},
        {"map_bytes": -1},
        {"min_capacity": 3, "max_capacity": 2},
        {"max_capacity": 5},
        {"capacities": []},
        {"capacities": [5]},
        {"objective_weight": float("nan")},
        {"objective_weight": float("inf")},
        {"objective_weight": 0},
    ],
)
def test_invalid_profiles_fail_closed(change):
    records = {(0, 0): [[0]]}
    profiles = _profiles(records)
    profiles[0, 0].update(change)
    with pytest.raises(ValueError):
        sweep.plan_slots(
            records, profiles, {0: 100}, [1, 2], local_experts=4, baseline_capacity=2
        )


def test_missing_profiles_ranks_and_empty_traces_fail_closed():
    records = {(0, 0): [[0]]}
    with pytest.raises(ValueError, match="profiles must cover exactly"):
        sweep.plan_slots(records, {}, {0: 100}, [1], baseline_capacity=1)
    with pytest.raises(ValueError, match="rank_budgets must cover exactly"):
        sweep.plan_slots(
            records, _profiles(records), {1: 100}, [1], baseline_capacity=1
        )
    with pytest.raises(ValueError, match="empty trace"):
        sweep.plan_slots(
            {(0, 0): []}, _profiles(records), {0: 100}, [1], baseline_capacity=1
        )


def test_profile_file_rejects_duplicate_layer_and_noncanonical_rank(tmp_path):
    import json

    path = tmp_path / "profile.json"
    entry = dict(rank=0, layer=2, expert_bytes=100)
    path.write_text(json.dumps(dict(layers=[entry, entry], rank_budgets={"0": 200})))
    with pytest.raises(ValueError, match="duplicate layer profile"):
        sweep.read_plan_profile(path)
    path.write_text(json.dumps(dict(layers=[entry], rank_budgets={"00": 200})))
    with pytest.raises(ValueError, match="canonical"):
        sweep.read_plan_profile(path)


def test_cli_keeps_sweep_report_and_writes_separate_plan(tmp_path, monkeypatch):
    import json
    import sys

    torch.save(
        dict(rank=0, layer=2, call=0, tensors={"route_ids": torch.tensor([[0, 1]])}),
        tmp_path / "rank0-layer2-call0.pt",
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            dict(
                layers=[dict(rank=0, layer=2, expert_bytes=100, map_bytes=1024)],
                rank_budgets={"0": 1224},
            )
        )
    )
    output, plan = tmp_path / "sweep.json", tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            str(tmp_path),
            "--output",
            str(output),
            "--capacities",
            "1",
            "2",
            "--plan-profile",
            str(profile),
            "--plan-output",
            str(plan),
            "--baseline-capacity",
            "2",
        ],
    )
    sweep.main()
    assert len(json.loads(output.read_text())["results"]) == 4
    saved = json.loads(plan.read_text())
    assert saved["per_rank"]["0"]["used_bytes"] <= 1224
    assert saved["per_rank"]["0"]["layer_slots"] == {"2": 1}


def test_planner_can_jump_across_flat_miss_curve():
    records = {(0, 0): [[0], [1], [2], [0]]}
    report = sweep.plan_slots(
        records,
        _profiles(records, expert_bytes=1, map_bytes=0),
        {0: 3},
        [1, 2, 3],
        policy="lru",
        baseline_capacity=1,
        local_experts=4,
    )
    rank = report["per_rank"]["0"]
    curve = report["per_layer"][0]["curve"]
    assert [point["misses"] for point in curve] == [4, 4, 3]
    assert rank["layer_slots"] == {"0": 3}
    assert rank["modeled_copy_bytes_saved_vs_baseline"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"unsupported_cost": 2},
        {"objective_weight": 10**400},
        {"expert_bytes": None},
    ],
)
def test_profile_typos_and_unrepresentable_weights_are_rejected(change):
    records = {(0, 0): [[0]]}
    profiles = _profiles(records)
    profiles[0, 0].update(change)
    with pytest.raises(ValueError):
        sweep.plan_slots(records, profiles, {0: 100}, [1], baseline_capacity=1)


def _recorded_costs(tmp_path, monkeypatch, expert_bytes=100):
    monkeypatch.setenv("VLLM_RDNA_EXPERT_ROUTE_DIR", str(tmp_path))
    path = _PATH.parents[3] / (
        "vllm/model_executor/layers/fused_moe/rdna_expert_route_trace.py"
    )
    spec = importlib.util.spec_from_file_location("cost_route_writer", path)
    assert spec is not None and spec.loader is not None
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    for layer in (0, 1):
        writer.record_routes(
            f"model.layers.{layer}.mlp.experts",
            [0, 1],
            4,
            phase="decode",
            rank=0,
            expert_bytes=expert_bytes,
        )
    return sweep.read_calls(tmp_path, local_experts=4)


@pytest.mark.parametrize(
    "change", [{"expert_bytes": 1}, {"map_bytes": 0}, {"map_bytes": 17}]
)
def test_jsonl_cost_contradictions_rejected_before_any_replay(
    tmp_path, monkeypatch, change
):
    records = _recorded_costs(tmp_path, monkeypatch)
    assert records.runtime_costs == {
        key: {"expert_bytes": 100, "map_bytes": 16, "logical_experts": 4}
        for key in records
    }
    profiles = _profiles(records, expert_bytes=100)
    profiles[0, 1].update(change)  # Later layer must reject before first replay.

    def forbidden_replay(*args, **kwargs):
        raise AssertionError("contradictory costs must fail before simulation")

    monkeypatch.setattr(sweep, "simulate", forbidden_replay)
    with pytest.raises(ValueError, match="contradict measured runtime"):
        sweep.plan_slots(
            records, profiles, {0: 1000}, [1, 2], baseline_capacity=1, local_experts=4
        )


def test_jsonl_plan_reports_verified_payload_and_fixed_maps(tmp_path, monkeypatch):
    records = _recorded_costs(tmp_path, monkeypatch)
    report = sweep.plan_slots(
        records,
        _profiles(records, expert_bytes=100),
        {0: 232},
        [1, 2],
        baseline_capacity=1,
        local_experts=4,
    )
    assert report["runtime_costs_verified"] is True
    assert "JSONL" in report["cost_scope"]
    rank = report["per_rank"]["0"]
    assert rank["map_bytes"] == 32 and rank["used_bytes"] == 232
    assert rank["used_bytes"] <= rank["budget_bytes"]


def test_jsonl_missing_measured_payload_cost_cannot_claim_feasible_plan(
    tmp_path, monkeypatch
):
    records = _recorded_costs(tmp_path, monkeypatch, expert_bytes=None)
    with pytest.raises(ValueError, match="missing measured expert_bytes"):
        sweep.plan_slots(
            records,
            _profiles(records),
            {0: 1000},
            [1],
            baseline_capacity=1,
            local_experts=4,
        )


def test_manual_costs_remain_explicitly_modeled_and_customizable():
    records = {(0, 0): [[0]]}
    report = sweep.plan_slots(
        records, _profiles(records, map_bytes=0), {0: 10}, [1], baseline_capacity=1
    )
    assert report["runtime_costs_verified"] is False
    assert "Caller-supplied modeled costs" in report["cost_scope"]
    assert report["per_rank"]["0"]["used_bytes"] == 10
