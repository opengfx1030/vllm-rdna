# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in static expert placement (VLLM_RDNA_MOE_PLACEMENT)."""

import json

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.rdna_moe_placement import (
    PLACEMENT_ENV,
    experts_to_expert_map,
    load_placement,
    local_experts_for_layer,
    placement_from_env,
)

NUM_EXPERTS = 8
EP_SIZE = 2
LAYER = "model.layers.3.mlp.experts"
# rank 0 keeps the odd experts, rank 1 the even ones (a permutation, not a range)
PLACEMENT = {
    "version": 1,
    "num_experts": NUM_EXPERTS,
    "ep_size": EP_SIZE,
    "method": "lpt+swap",
    "layers": {"3": [[1, 3, 5, 7], [0, 2, 4, 6]]},
}


def _write(tmp_path, doc, name="placement.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_local_experts_for_layer_round_robin_style():
    assert local_experts_for_layer(PLACEMENT, LAYER, 0, NUM_EXPERTS) == [1, 3, 5, 7]
    assert local_experts_for_layer(PLACEMENT, LAYER, 1, NUM_EXPERTS) == [0, 2, 4, 6]
    assert (
        local_experts_for_layer(PLACEMENT, "model.layers.4.mlp.experts", 0, NUM_EXPERTS)
        is None
    )
    # a file for another expert count, or another module without a layer index
    assert local_experts_for_layer(PLACEMENT, LAYER, 0, 16) is None
    assert local_experts_for_layer(PLACEMENT, "model.layers.3.mlp", 0, NUM_EXPERTS) == [
        1,
        3,
        5,
        7,
    ]
    assert local_experts_for_layer(PLACEMENT, "model.norm", 0, NUM_EXPERTS) is None


def test_local_experts_for_layer_rejects_non_uniform_rank():
    doc = {
        **PLACEMENT,
        "layers": {"3": [[1, 3, 5, 7], [0, 2, 4]]},
    }
    assert local_experts_for_layer(doc, LAYER, 1, NUM_EXPERTS) is None


def test_load_placement_validates(tmp_path):
    path = _write(tmp_path, PLACEMENT)
    assert load_placement(path)["layers"]["3"][1] == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="version"):
        load_placement(_write(tmp_path, {**PLACEMENT, "version": 2}, "v2.json"))
    with pytest.raises(ValueError, match="expected 4"):
        load_placement(_write(tmp_path, {**PLACEMENT, "ep_size": 4}, "ep4.json"))
    with pytest.raises(ValueError, match="layers"):
        load_placement(_write(tmp_path, {"version": 1}, "nolayers.json"))


def test_experts_to_expert_map():
    expert_map = experts_to_expert_map([1, 3, 5, 7], NUM_EXPERTS)
    assert expert_map.tolist() == [-1, 0, -1, 1, -1, 2, -1, 3]
    assert expert_map.dtype == torch.int32
    with pytest.raises(ValueError, match="repeats"):
        experts_to_expert_map([1, 1], NUM_EXPERTS)
    with pytest.raises(ValueError, match="outside"):
        experts_to_expert_map([0, NUM_EXPERTS], NUM_EXPERTS)


def test_determine_expert_map_explicit_matches_linear_count():
    linear_n, linear_map, _ = determine_expert_map(EP_SIZE, 1, NUM_EXPERTS)
    n, expert_map, mask = determine_expert_map(
        EP_SIZE, 1, NUM_EXPERTS, explicit_local_experts=[0, 2, 4, 6]
    )
    assert n == linear_n == NUM_EXPERTS // EP_SIZE
    assert mask is None
    assert expert_map is not None and linear_map is not None
    # same local slot count, different ownership: expert 0 stays in slot 0,
    # expert 4 is local here although the linear map puts it on rank 0.
    assert expert_map[0].item() == 0
    assert expert_map[4].item() == 2
    assert expert_map[5].item() == -1
    assert linear_map[4].item() == 0  # linear rank 1 owns experts 4..7
    assert linear_map[1].item() == -1  # and expert 1 lives on rank 0
    with pytest.raises(ValueError, match="fused shared experts"):
        determine_expert_map(
            EP_SIZE,
            0,
            NUM_EXPERTS,
            num_fused_shared_experts=1,
            explicit_local_experts=[0, 1, 2, 3],
        )


def test_placement_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv(PLACEMENT_ENV, raising=False)
    assert placement_from_env(LAYER, 0, NUM_EXPERTS) is None
    monkeypatch.setenv(PLACEMENT_ENV, _write(tmp_path, PLACEMENT))
    assert placement_from_env(LAYER, 1, NUM_EXPERTS) == [0, 2, 4, 6]
    assert placement_from_env(LAYER, 0, NUM_EXPERTS) == [1, 3, 5, 7]
