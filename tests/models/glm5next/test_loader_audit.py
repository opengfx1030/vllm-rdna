# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.glm5next.common.model import _match_param_name


def test_match_param_name_accepts_loader_prefixes():
    params = {
        "layers.0.hc_attn_base": object(),
        "layers.1.hc_attn_base": object(),
    }

    assert _match_param_name(params, "layers.0.hc_attn_base") == (
        "layers.0.hc_attn_base"
    )
    assert _match_param_name(params, "model.layers.1.hc_attn_base") == (
        "layers.1.hc_attn_base"
    )
    assert _match_param_name(
        params, "language_model.model.layers.0.hc_attn_base"
    ) == "layers.0.hc_attn_base"


def test_match_param_name_does_not_cross_match_layers():
    params = {"layers.1.hc_attn_base": object()}

    assert _match_param_name(params, "layers.0.hc_attn_base") is None
    assert _match_param_name(params, "layers.0.hc_attn_scale") is None
