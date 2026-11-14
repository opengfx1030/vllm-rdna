# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[3]
    / "vllm/model_executor/model_loader/qwen4exp_gguf_mapping.py"
)
spec = importlib.util.spec_from_file_location("qmap", MODULE)
qmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qmap)
CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures/qwen38_gguf_contract.json").read_text()
)


def test_native_head_and_embedding_destinations_have_exact_shapes():
    tensors = [
        t
        for t in CONTRACT["tensors"]
        if not t["name"].startswith("blk.")
        and t["name"] != "per_layer_token_embd.weight"
    ]
    plan = qmap.plan_tensors(tensors, native_parameters=CONTRACT["native_parameters"])
    assert not plan.unsupported
    assert plan.by_source["output.weight"].destination == "lm_head.weight"
    assert (
        plan.by_source["token_embd.weight"].destination == "model.embed_tokens.weight"
    )
    assert (
        plan.by_source["output_hc_down.weight"].destination
        == "model.hyper_connection_mixer.input_mix_weight_down.weight"
    )
    assert all(
        e.destination_shape
        == tuple(CONTRACT["native_parameters"][e.destination]["shape"])
        for e in plan.entries
    )
    assert plan.by_source["output_hc_norm.weight"].transform == "subtract_one"


def test_inventory_planning_accounts_for_every_tensor_without_silent_drops():
    tensors = CONTRACT["tensors"]
    plan = qmap.plan_tensors(tensors, native_parameters=CONTRACT["native_parameters"])
    assert len(plan.entries) == 1224
    assert set(plan.by_source) == {t["name"] for t in tensors}
    assert len(plan.unsupported) == 0


def test_gdn_linear_attention_destinations_slices_and_transforms():
    names = {
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_a",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
    }
    plan = qmap.plan_tensors(
        [t for t in CONTRACT["tensors"] if t["name"] in names],
        native_parameters=CONTRACT["native_parameters"],
    )
    assert not plan.unsupported
    assert (
        plan.by_source["blk.0.attn_qkv.weight"].destination
        == "model.layers.0.linear_attn.in_proj_qkvz.weight"
    )
    assert plan.by_source["blk.0.attn_qkv.weight"].slice_start == 0
    assert plan.by_source["blk.0.attn_qkv.weight"].slice_size == 10240
    assert plan.by_source["blk.0.attn_qkv.weight"].transform == "gdn_inverse"
    assert (
        plan.by_source["blk.0.attn_gate.weight"].destination
        == "model.layers.0.linear_attn.in_proj_qkvz.weight"
    )
    assert plan.by_source["blk.0.attn_gate.weight"].slice_start == 10240
    assert plan.by_source["blk.0.attn_gate.weight"].slice_size == 6144
    assert plan.by_source["blk.0.attn_gate.weight"].transform == "gdn_inverse"
    assert (
        plan.by_source["blk.0.ssm_beta.weight"].destination
        == "model.layers.0.linear_attn.in_proj_ba.weight"
    )
    assert plan.by_source["blk.0.ssm_beta.weight"].slice_start == 0
    assert plan.by_source["blk.0.ssm_beta.weight"].slice_size == 48
    assert plan.by_source["blk.0.ssm_alpha.weight"].slice_start == 48
    assert plan.by_source["blk.0.ssm_alpha.weight"].slice_size == 48
    assert plan.by_source["blk.0.ssm_conv1d.weight"].destination_shape == (10240, 1, 4)
    assert plan.by_source["blk.0.ssm_conv1d.weight"].transform == "gdn_conv_inverse"
    assert (
        plan.by_source["blk.0.ssm_a"].destination == "model.layers.0.linear_attn.A_log"
    )
    assert plan.by_source["blk.0.ssm_a"].transform == "gdn_inverse"
    assert (
        plan.by_source["blk.0.ssm_norm.weight"].destination
        == "model.layers.0.linear_attn.norm.weight"
    )
    assert plan.by_source["blk.0.ssm_norm.weight"].transform == "identity"
    assert (
        plan.by_source["blk.0.ssm_out.weight"].destination
        == "model.layers.0.linear_attn.out_proj.weight"
    )
    assert plan.by_source["blk.0.ssm_out.weight"].transform == "gdn_inverse"


def test_direct_layer_destinations_and_reshape_contracts():
    names = {
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.hc_attn_norm.weight",
        "blk.0.hc_ffn_up.weight",
        "blk.3.attn_output.weight",
        "blk.3.attn_q_norm.weight",
        "blk.1.ple_norm_key.weight",
        "blk.1.ple_conv1d.weight",
    }
    tensors = [t for t in CONTRACT["tensors"] if t["name"] in names]
    plan = qmap.plan_tensors(tensors, native_parameters=CONTRACT["native_parameters"])
    assert not plan.unsupported
    assert plan.by_source["blk.0.ffn_gate_inp_shexp.weight"].destination_shape == (
        1,
        2560,
    )
    assert plan.by_source["blk.0.ffn_gate_inp_shexp.weight"].transform == "reshape"
    assert plan.by_source["blk.3.attn_q_norm.weight"].transform == "subtract_one"
    assert plan.by_source["blk.1.ple_conv1d.weight"].transform == "unsqueeze_conv"


def test_merged_dense_and_packed_expert_slices():
    names = {
        "blk.0.hc_attn_down.weight",
        "blk.0.hc_attn_inject.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.3.indexer.q_proj.weight",
        "blk.3.indexer.k_proj.weight",
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "per_layer_token_embd.weight",
    }
    plan = qmap.plan_tensors(
        [t for t in CONTRACT["tensors"] if t["name"] in names],
        native_parameters=CONTRACT["native_parameters"],
    )
    assert not plan.unsupported
    assert plan.by_source["blk.0.hc_attn_inject.weight"].slice_start == 320
    assert plan.by_source["blk.0.ffn_up_exps.weight"].slice_axis == 1
    assert plan.by_source["blk.0.ffn_up_exps.weight"].slice_start == 640
    assert plan.by_source["blk.3.attn_v.weight"].slice_start == 12800
    assert plan.by_source["blk.0.ffn_down_exps.weight"].delivery == "packed_expert"
    assert plan.by_source["per_layer_token_embd.weight"].delivery == "packed_ple"


@pytest.mark.parametrize(
    "dims,typ", [([2560.5, 248320], 8), ([True, 248320], 8), ([2560, 248320], "8")]
)
def test_header_values_are_not_normalized(dims, typ):
    with pytest.raises(qmap.MappingError):
        qmap.plan_tensors([{"name": "output.weight", "dims": dims, "type": typ}])


def test_zero_centered_norm_inverse_is_subtract_one():
    import torch

    assert torch.equal(
        qmap.inverse_zero_centered_norm(torch.tensor([1.0, 2.0])),
        torch.tensor([0.0, 1.0]),
    )


def test_unknown_tensor_is_explicitly_unsupported_with_contract():
    entry = qmap.plan_tensors(
        [{"name": "mystery.weight", "dims": [2, 3], "type": 0}]
    ).entries[0]
    assert entry.destination is None
    assert (
        entry.source == "mystery.weight"
        and entry.shape == (2, 3)
        and entry.ggml_type == 0
    )
    assert "not yet verified" in entry.reason
