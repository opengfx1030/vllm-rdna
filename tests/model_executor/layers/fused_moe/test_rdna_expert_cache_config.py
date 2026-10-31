# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup plans must not silently exceed a rank's intended expert capacity."""

import importlib.util
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_config.py"
)
_SPEC = importlib.util.spec_from_file_location("slot_plan_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config)


def test_rank_and_layer_select_different_fixed_capacities():
    plan = {"0": {"7": 32, "8": 128}, "1": {"7": 64}}
    assert config.select_slots(96, plan, 0, "model.layers.7.mlp.experts") == 32
    assert config.select_slots(96, plan, 0, "model.layers.8.mlp.experts") == 128
    assert config.select_slots(96, plan, 1, "model.layers.7.mlp.experts") == 64
    assert config.select_slots(96, {}, 1, "arbitrary") == 96


@pytest.mark.parametrize(
    "rank,name",
    [(1, "model.layers.8.mlp"), (2, "model.layers.7.mlp"), (0, "mtp.experts")],
)
def test_incomplete_plan_fails_closed(rank, name):
    with pytest.raises(ValueError):
        config.select_slots(96, {"0": {"7": 32}}, rank, name)


@pytest.mark.parametrize(
    "plan",
    [
        [],
        {0: {"1": 32}},
        {"00": {"1": 32}},
        {"0": {}},
        {"0": {1: 32}},
        {"0": {"01": 32}},
        {"0": {"1": 0}},
        {"0": {"1": 257}},
        {"0": {"1": True}},
        {"0": {"1": 32.5}},
    ],
)
def test_malformed_plan_rejected(plan):
    with pytest.raises(ValueError):
        config.validate_slot_plan(plan)
