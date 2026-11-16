# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.models.qwen4_exp.amd.qsa import _qsa_tp3_ordinal

from ...utils import spawn_new_process_for_each_test
from .test_config import _text_config


@pytest.mark.parametrize("label", ["qwen_sparse_attention", "indexed_attention"])
def test_rotating_owners_follow_target_and_mtp_sparse_layers(label):
    config = SimpleNamespace(
        layer_types=["linear_attention", label, "linear_attention", label],
        num_hidden_layers=4,
        mtp_num_hidden_layers=2,
    )
    assert [_qsa_tp3_ordinal(config, i) for i in (1, 3, 4, 5)] == [0, 1, 2, 3]
    for invalid_layer in (-1, 0, 2, 6):
        with pytest.raises(ValueError):
            _qsa_tp3_ordinal(config, invalid_layer)


@spawn_new_process_for_each_test
@pytest.mark.parametrize("tp_size", [1, 3])
def test_amd_mtp_fusion_preserves_full_width_before_decoder(tp_size: int) -> None:
    """The real fusion loaders/math must retain all 2560 rows at text TP3."""
    import contextlib

    from vllm.config.compilation import CompilationMode
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.qwen4_exp.amd import model as model_module
    from vllm.models.qwen4_exp.amd import mtp

    width, branches, tokens = 2560, 2, 2
    config = _text_config(hidden_size=width, hc_count=branches)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.float32),
        quant_config=None,
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )
    captured = []

    class ReachedDecoder(Exception):
        pass

    class DecoderBoundary(torch.nn.Module):
        def forward(self, **kwargs):
            captured.append(kwargs["hidden_states"].detach().clone())
            raise ReachedDecoder

    # Only the heavy decoder, unused embedding/mixer and runtime context are
    # isolated. Norms, projections, checkpoint loading and fusion stay real.
    with (
        torch.device("cpu"),
        patch.object(mtp, "get_tensor_model_parallel_world_size", return_value=tp_size),
        patch.object(
            model_module, "get_tensor_model_parallel_world_size", return_value=tp_size
        ),
        patch.object(
            linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ),
        patch.object(linear, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(
            parameter, "get_tensor_model_parallel_world_size", return_value=tp_size
        ),
        patch.object(linear, "get_current_vllm_config_or_none", return_value=None),
        patch.object(
            linear,
            "dispatch_unquantized_gemm",
            return_value=lambda layer, x, weight, bias: torch.nn.functional.linear(
                x, weight, bias
            ),
        ),
        patch.object(
            mtp.GemmaRMSNorm,
            "dispatch_forward",
            lambda self, **kwargs: self.forward_native,
        ),
        patch.object(mtp, "_off_gpu_under_pp", contextlib.nullcontext),
        patch.object(mtp, "_make_draft_vllm_config", return_value=vllm_config),
        patch.object(
            mtp,
            "set_current_vllm_config",
            side_effect=lambda *a, **k: contextlib.nullcontext(),
        ),
        patch.object(
            mtp,
            "VocabParallelEmbedding",
            side_effect=lambda *a, **k: torch.nn.Identity(),
        ),
        patch.object(
            mtp, "Qwen4ExpDecoderLayer", side_effect=lambda *a, **k: DecoderBoundary()
        ),
        patch.object(
            mtp, "is_model_fused_shared_expert_compatible", return_value=False
        ),
        patch.object(
            mtp, "GatedResidual", side_effect=lambda *a, **k: torch.nn.Identity()
        ),
    ):
        model = mtp.Qwen4ExpMultiTokenPredictor(vllm_config=vllm_config)
        # Nonuniform and non-diagonal maps distinguish branch sharing, transpose,
        # embedding residual and hidden/embedding order errors.
        row = torch.arange(width)
        embedding_weight = torch.diag(torch.linspace(0.2, 1.2, width))
        hidden_weight = torch.diag(torch.linspace(-0.8, 0.4, width))
        embedding_weight[row, (row + 7) % width] = 0.15
        hidden_weight[row, (row + 13) % width] = -0.3
        embedding_norm = torch.linspace(-0.2, 0.3, width)
        hidden_norm = torch.linspace(0.1, 0.6, width * branches)
        weights = [
            ("fc_embedding.weight", embedding_weight),
            ("fc_hidden.weight", hidden_weight),
            ("pre_fc_norm_embedding.weight", embedding_norm),
            ("pre_fc_norm_hidden.weight", hidden_norm),
        ]
        assert model.load_weights(weights) == {name for name, _ in weights}

        def normalize(x, weight):
            return (
                x
                * torch.rsqrt(
                    x.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
                )
                * (1 + weight)
            )

        for step in range(2):
            embedding = (
                torch.arange(tokens * width, dtype=torch.float32)
                .reshape(tokens, width)
                .remainder(31)
                .sub(11 + step)
                .div(9)
            )
            hidden = (
                torch.arange(tokens * branches * width, dtype=torch.float32)
                .reshape(tokens, branches * width)
                .remainder(43)
                .sub(17 - step)
                .div(7)
            )
            expected_embedding = (
                normalize(embedding, embedding_norm) @ embedding_weight.T
            )
            expected_hidden = (
                normalize(hidden, hidden_norm).reshape(tokens, branches, width)
                @ hidden_weight.T
            )
            expected = (expected_hidden + expected_embedding[:, None, :]).flatten(-2)
            with pytest.raises(ReachedDecoder):
                model(
                    input_ids=None,
                    positions=torch.arange(tokens),
                    inputs_embeds=embedding,
                    hidden_states=hidden,
                    spec_step_idx=step,
                )
            assert captured[-1].shape == (tokens, branches * width)
            torch.testing.assert_close(captured[-1], expected)
