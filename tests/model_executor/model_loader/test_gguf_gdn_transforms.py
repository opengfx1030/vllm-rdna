# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

MODULE = (
    Path(__file__).parents[3]
    / "vllm/model_executor/model_loader/gguf_gdn_transforms.py"
)
spec = importlib.util.spec_from_file_location("gdn_inverse", MODULE)
gdn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdn)
ORACLE = json.loads(
    (Path(__file__).parent / "fixtures/qwen4exp_gdn_v_head_oracle.json").read_text()
)


def forward_v_order(tensor, dim=0, heads=16, repeat=3, width=128):
    case = next(
        c for c in ORACLE["cases"] if c["k_heads"] == heads and c["v_per_k"] == repeat
    )
    indices = (
        torch.tensor(case["forward_head_order"])[:, None] * width + torch.arange(width)
    ).flatten()
    return tensor.index_select(dim, indices)


def test_inverse_v_head_matches_frozen_converter_oracle_round_trip():
    for heads, repeat, width, dim, shape in [
        (2, 3, 4, 0, (24, 5)),
        (16, 3, 128, 0, (6144, 1)),
        (16, 3, 128, 1, (2, 6144, 3)),
        (16, 3, 128, -1, (2, 6144)),
    ]:
        original = torch.arange(math.prod(shape)).reshape(shape)
        gguf_order = forward_v_order(original, dim, heads, repeat, width)
        assert torch.equal(
            gdn.inverse_v_heads(gguf_order, dim, heads, repeat, width), original
        )


def test_inverse_ssm_a_uses_converter_log_inverse():
    original = torch.linspace(-0.25, -4.0, 48, dtype=torch.float32)
    gguf = -torch.exp(forward_v_order(original, width=1))
    restored = gdn.inverse_gdn_tensor("blk.0.ssm_a", gguf)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, original, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bad", [0.0, 1.0, float("nan"), float("inf"), -float("inf")])
def test_ssm_a_rejects_bad_values_in_correctly_shaped_tensor(bad):
    values = torch.full((48,), -1.0)
    values[7] = bad
    with pytest.raises(ValueError, match="finite and negative"):
        gdn.inverse_gdn_tensor("blk.0.ssm_a", values)


@pytest.mark.parametrize("role", ["ssm_alpha.weight", "ssm_beta.weight"])
def test_alpha_beta_reorder_matrix_rows_not_vector_elements(role):
    original = torch.arange(48 * 3).reshape(48, 3)
    got = gdn.inverse_gdn_tensor("blk.0." + role, forward_v_order(original, width=1))
    assert torch.equal(got, original)


def test_dt_bias_reorders_vector_heads():
    original = torch.arange(48)
    assert torch.equal(
        gdn.inverse_gdn_tensor("blk.0.ssm_dt.bias", forward_v_order(original, width=1)),
        original,
    )


def test_out_projection_reorders_input_columns():
    original = torch.arange(3 * 6144).reshape(3, 6144)
    assert torch.equal(
        gdn.inverse_gdn_tensor(
            "blk.0.ssm_out.weight", forward_v_order(original, dim=1)
        ),
        original,
    )


def test_conv_preserves_qk_and_inverts_only_v_channels():
    original = torch.arange(10240 * 4).reshape(10240, 1, 4)
    squeezed = original[:, 0, :]
    gguf = torch.cat([squeezed[:4096], forward_v_order(squeezed[4096:])])
    got = gdn.inverse_gdn_tensor("blk.0.ssm_conv1d.weight", gguf)
    assert torch.equal(got, original)


def test_qkv_preserves_qk_and_inverts_only_v_rows():
    original = torch.arange(10240 * 3).reshape(10240, 3)
    gguf = torch.cat([original[:4096], forward_v_order(original[4096:])])
    assert torch.equal(gdn.inverse_gdn_tensor("blk.0.attn_qkv.weight", gguf), original)


def test_gate_uses_exact_real_gguf_name_and_reorders_rows():
    original = torch.arange(6144 * 3).reshape(6144, 3)
    got = gdn.inverse_gdn_tensor("blk.0.attn_gate.weight", forward_v_order(original))
    assert torch.equal(got, original)


@pytest.mark.parametrize(
    "name",
    [
        "blk.00.attn_gate.weight",
        "blk.48.attn_gate.weight",
        "blk.3.attn_gate.weight",
        "attn_gate.weight",
    ],
)
def test_rejects_noncanonical_or_non_gdn_layer_names(name):
    with pytest.raises(ValueError, match="invalid GDN tensor name"):
        gdn.inverse_gdn_tensor(name, torch.zeros(6144, 1))
