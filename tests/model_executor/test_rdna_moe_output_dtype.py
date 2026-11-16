# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU precision boundaries; arithmetic experts do not qualify native HIP."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
    RDNA2W4A16MoEExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.platforms import current_platform


class ArithmeticExperts(RDNA2W4A16MoEExperts):
    """Exercise the real allocator/finalizer with CPU FP16/FP32 arithmetic."""

    def __init__(self, output_dtype):
        self.moe_config = NS(output_dtype=output_dtype, moe_parallel_config=None)
        self.quant_config = NS(
            **dict.fromkeys(
                (
                    "a1_scale",
                    "a2_scale",
                    "a1_gscale",
                    "a2_gscale",
                    "w1_scale",
                    "w2_scale",
                    "w1_zp",
                    "w2_zp",
                    "w1_bias",
                    "w2_bias",
                )
            )
        )
        self.seen = None

    def workspace_shapes(self, M, N, K, topk, *args):
        return (M * topk, K), (M * topk, K), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        workspace13,
        workspace2,
        **kwargs,
    ):
        self.seen = output, workspace13, workspace2
        assert (
            hidden_states.dtype
            == workspace13.dtype
            == workspace2.dtype
            == torch.float16
        )
        assert not torch._C._overlaps(output, hidden_states)
        if output.dtype != workspace13.dtype:
            assert not torch._C._overlaps(output, workspace13)
            assert not torch._C._overlaps(output, workspace2)
        topk = topk_ids.shape[1]
        routed = hidden_states.repeat_interleave(topk, dim=0)
        workspace13.copy_(
            torch.bmm(routed.unsqueeze(1), w1[topk_ids.reshape(-1)]).squeeze(1)
        )
        workspace2.copy_(workspace13)
        down = torch.bmm(
            workspace2.float().unsqueeze(1), w2[topk_ids.reshape(-1)].float()
        ).squeeze(1)
        weighted = down.view(hidden_states.shape[0], topk, -1)
        output.copy_((weighted * topk_weights.unsqueeze(-1)).sum(dim=1))


@pytest.fixture(autouse=True)
def cpu_precision(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(ops, "_RDNA2_MOE_FP32_ACCUM", True)
    monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "0")


def _inputs(rows=2):
    hidden = torch.tensor([[1.0, 2**-12]], dtype=torch.float16).repeat(rows, 1)
    w1 = torch.eye(2, dtype=torch.float16).repeat(2, 1, 1)
    w2 = torch.ones(2, 2, 2, dtype=torch.float16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64).repeat(rows, 1)
    weights = torch.full((rows, 2), 0.5, dtype=torch.float32)
    return hidden, w1, w2, ids, weights


def _kernel(dtype=torch.float32):
    return mk.FusedMoEKernelModularImpl(
        MoEPrepareAndFinalizeNoDPEPModular(), ArithmeticExperts(dtype)
    )


def test_real_finalize_preserves_sub_half_ulp_without_widening_scratch():
    args = _inputs()
    before = args[0].clone()
    kernel = _kernel()
    result = kernel.apply(*args)
    expected = torch.full((2, 2), 1 + 2**-12, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)
    assert torch.equal(args[0], before)
    assert kernel.fused_experts.seen[0] is result
    assert torch.equal(_kernel(None).apply(*args), expected.half())
    assert not torch.equal(expected, expected.half().float())


@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_empty_rows_negotiate_native_and_final_precision(dtype):
    kernel = _kernel(dtype)
    result = kernel.apply(*_inputs(0))
    assert result.shape == (0, 2)
    assert result.dtype == (dtype or torch.float16)
    assert kernel.fused_experts.seen is None


@pytest.mark.parametrize(
    "kind", ["matching", "shape", "dtype", "stride", "input", "weights"]
)
def test_native_output_alias_never_overwrites_live_inputs(kind):
    hidden, w1, w2, ids, weights = _inputs()
    alias = torch.empty(2, 2, dtype=torch.float32)
    if kind == "shape":
        alias = torch.empty(1, 2, dtype=torch.float32)
    elif kind == "dtype":
        alias = torch.empty(2, 2, dtype=torch.float16)
    elif kind == "stride":
        alias = alias.T
    elif kind == "input":
        backing = torch.zeros(8, dtype=torch.float16)
        view = backing.view(2, 4)[:, :2]
        view.copy_(hidden)
        hidden = view
        alias = backing.view(torch.float32).view(2, 2)
    elif kind == "weights":
        alias = w2.view(torch.float32).view(2, 2)
    before, weights_before = hidden.clone(), w2.clone()
    result = _kernel()._fused_experts(
        hidden.dtype,
        hidden,
        None,
        w1,
        w2,
        weights,
        ids,
        MoEActivation.SILU,
        2,
        2,
        None,
        False,
        None,
        alias,
    )
    assert (result is alias) == (kind == "matching")
    assert torch.equal(hidden, before)
    assert torch.equal(w2, weights_before)
    assert torch.equal(result, torch.full_like(result, 1 + 2**-12))


@pytest.mark.parametrize(
    "failure", ["unordered", "w4a8", "input", "native", "finalize"]
)
def test_explicit_fp32_fails_closed(monkeypatch, failure):
    kernel = _kernel()
    args = _inputs()
    if failure == "unordered":
        monkeypatch.setattr(ops, "_RDNA2_MOE_FP32_ACCUM", False)
    elif failure == "w4a8":
        monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "1")
    elif failure == "input":
        args = (args[0].bfloat16(), *args[1:])
    elif failure == "native":
        kernel.fused_experts.native_output_dtype = lambda dtype: dtype
    else:

        class Unsupported(MoEPrepareAndFinalizeNoDPEPModular):
            pass

        kernel.prepare_finalize = Unsupported()
    with pytest.raises(ValueError):
        kernel.apply(*args)


def _runner(monkeypatch, partial, shared=None):
    config = NS(
        output_dtype=torch.float32,
        hidden_dim=2,
        hidden_dim_unpadded=2,
        is_sequence_parallel=False,
        pcp_size=1,
        tp_size=3,
        ep_size=1,
        dp_size=1,
        moe_parallel_config=NS(use_ep=False),
        skip_final_all_reduce=False,
        use_deferred_moe_finalize=False,
    )
    kernel = NS(
        is_monolithic=False,
        fused_experts=ArithmeticExperts(torch.float32),
        output_is_reduced=lambda: False,
    )
    method = NS(
        moe_kernel=kernel, has_unpadded_output=False, skip_forward_padding=False
    )
    routed = NS(quant_method=method, _ensure_moe_quant_config_init=lambda: None)
    monkeypatch.setattr(moe_runner, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner, "register_layer_for_moe_forward_op", lambda *args: None
    )
    monkeypatch.setattr(moe_runner, "is_forward_context_available", lambda: False)
    runner = moe_runner.MoERunner("precision", config, NS(), routed)
    runner._forward_entry = lambda *args: (
        partial.clone() if shared is None else (shared, partial.clone())
    )
    return runner


def test_runner_keeps_fp32_through_shared_add_and_allreduce(monkeypatch):
    partial = torch.full((2, 2), 1 + 2**-12, dtype=torch.float32)
    shared = torch.full_like(partial, -1, dtype=torch.float16)
    calls = []

    def reduce(value):
        calls.append(value.dtype)
        return value

    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", reduce)
    runner = _runner(monkeypatch, partial, shared)
    hidden = torch.ones_like(shared)
    result = runner(hidden, hidden)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.full_like(result, 2**-12))
    assert calls == [torch.float32]


def test_qwen_rounds_only_after_external_replicated_shared_add(monkeypatch):
    partial = torch.full((2, 2), 1 + 2**-12, dtype=torch.float32)
    runner = _runner(monkeypatch, partial)
    monkeypatch.setattr(
        moe_runner, "tensor_model_parallel_all_reduce", lambda value: value
    )
    model = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(model)
    model.output_dtype = torch.float32
    model.is_sequence_parallel = False
    model.replicate_shared_expert = True

    class Shared(torch.nn.Module):
        def forward(self, hidden):
            return torch.full_like(hidden, -1)

    model.shared_expert = Shared()
    model.experts = runner
    hidden = torch.ones_like(partial, dtype=torch.float16)
    result = model(hidden)
    assert result.dtype == torch.float16
    assert torch.equal(result, torch.full_like(hidden, 2**-12))


def test_native_packed_layout_allocates_fp16_gate_up_and_fp32_down():
    experts = RDNA2W4A16MoEExperts.__new__(RDNA2W4A16MoEExperts)
    settings = ArithmeticExperts(torch.float32)
    experts.moe_config = settings.moe_config
    experts.quant_config = settings.quant_config
    hidden = torch.zeros(2, 8, dtype=torch.float16)
    w1 = torch.empty(2, 1, 32, dtype=torch.int32)
    w2 = torch.empty(2, 2, 8, dtype=torch.int32)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    assert experts.moe_problem_size(hidden, w1, w2, ids) == (2, 2, 32, 8, 2)
    kernel = mk.FusedMoEKernelModularImpl(MoEPrepareAndFinalizeNoDPEPModular(), experts)
    workspace13, workspace2, output, specialized = kernel._allocate_buffers(
        torch.float16, hidden.device, 2, 2, 32, 8, 2, 2, 2, None, MoEActivation.SILU
    )
    assert workspace13.shape == (4, 32)
    assert workspace13.dtype == workspace2.dtype == torch.float16
    assert output.shape == (2, 8) and output.dtype == torch.float32
    assert not torch._C._overlaps(workspace13, output)
    assert not specialized


def test_fake_runner_output_keeps_routed_and_shared_dtypes_distinct():
    hidden = torch.zeros(2, 8, dtype=torch.float16)
    routed = moe_runner._moe_forward_fake(
        hidden, hidden, None, None, "precision", 0, torch.float32
    )
    shared, routed_shared = moe_runner._moe_forward_shared_fake(
        hidden, hidden, hidden, None, "precision", 0, torch.float32
    )
    assert routed.dtype == routed_shared.dtype == torch.float32
    assert shared.dtype == torch.float16


def test_default_precision_does_not_require_fp32_or_disable_w4a8(monkeypatch):
    monkeypatch.setattr(ops, "_RDNA2_MOE_FP32_ACCUM", False)
    monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "1")
    result = _kernel(None).apply(*_inputs())
    assert result.dtype == torch.float16
    assert torch.equal(result, torch.ones_like(result))
