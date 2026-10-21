# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils


@pytest.mark.skipif(current_platform.is_rocm(), reason="Non-ROCm import guard")
def test_rocm_bf16_skinny_guard_on_other_platforms():
    x = torch.empty(3, 1024, dtype=torch.bfloat16)
    weight = torch.empty(2560, 1024, dtype=torch.bfloat16)
    assert not utils.rocm_bf16_skinny_gemm_compatible(x, weight, None)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize(
    "shape", [(3, 1024), (4, 1024), (24, 1024), (1, 3, 1024), (2, 2, 1024)]
)
def test_rocm_bf16_skinny_draft_dispatch(monkeypatch, shape):
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("Requires gfx10x")
    if not hasattr(torch.ops._rocm_C, "gemv_bf16_rdna2"):
        pytest.skip("Requires rebuilt gemv_bf16_rdna2 op")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, 1024, device="cuda", dtype=torch.bfloat16) * 0.05
    bias = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    original = utils.ops.gemv_bf16_rdna2
    skinny = MagicMock(side_effect=original)
    monkeypatch.setattr(utils.ops, "gemv_bf16_rdna2", skinny)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    skinny.assert_called_once()
    x_view, passed_weight, passed_bias = skinny.call_args.args
    assert x_view.shape == (x.numel() // 1024, 1024)
    assert x_view.data_ptr() == x.data_ptr()  # the gate must not copy
    assert passed_weight is weight and passed_bias is bias
    assert out.shape == (*shape[:-1], 2560)
    ref = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
def test_rocm_bf16_skinny_rejects_unmeasured_and_unaligned(monkeypatch):
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("Requires gfx10x")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    weight = torch.randn(2560, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16)
    unaligned = torch.empty(3 * 1024 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
        3, 1024
    )
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 != 0
    assert not utils.rocm_bf16_skinny_gemm_compatible(unaligned, weight, None)
    assert not utils.rocm_bf16_skinny_gemm_compatible(x[:2], weight, None)
    assert not utils.rocm_bf16_skinny_gemm_compatible(x, weight[:128], None)
    assert not utils.rocm_bf16_skinny_gemm_compatible(x[:, :0], weight[:, :0], None)
    assert not utils.rocm_bf16_skinny_gemm_compatible(x.t(), weight, None)
    noncontiguous_3d = torch.empty(
        1, 1024, 3, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    assert not noncontiguous_3d.is_contiguous()
    assert not utils.rocm_bf16_skinny_gemm_compatible(noncontiguous_3d, weight, None)
    assert not utils.rocm_bf16_skinny_gemm_compatible(
        x, weight, torch.empty(1, 2560, device="cuda", dtype=torch.bfloat16)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("rows", [3, 4, 24])
def test_rocm_bf16_skinny_fullgraph_cuda_graph_route(monkeypatch, rows):
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("Requires gfx10x")
    if not hasattr(torch.ops._rocm_C, "gemv_bf16_rdna2"):
        pytest.skip("Requires rebuilt gemv_bf16_rdna2 op")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    x = torch.randn(1, rows, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, 1024, device="cuda", dtype=torch.bfloat16) * 0.05
    bias = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    original = utils.ops.gemv_bf16_rdna2
    skinny = MagicMock(side_effect=original)
    monkeypatch.setattr(utils.ops, "gemv_bf16_rdna2", skinny)
    traced_ops: list[object] = []

    def backend(graph_module, _example_inputs):
        traced_ops.extend(node.target for node in graph_module.graph.nodes)
        return torch._inductor.compile(graph_module, _example_inputs)

    def project(activation):
        return torch.ops.vllm.rocm_unquantized_gemm(activation, weight, bias)

    compiled = torch.compile(project, backend=backend, fullgraph=True)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            compiled(x)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.accelerator.synchronize()
    assert any(
        target
        in (
            torch.ops.vllm.rocm_unquantized_gemm,
            torch.ops.vllm.rocm_unquantized_gemm.default,
        )
        for target in traced_ops
    )
    assert skinny.call_count >= 1  # opaque op implementation ran after tracing

    graph = torch.cuda.CUDAGraph()
    before_capture = skinny.call_count
    with torch.cuda.graph(graph):
        out = compiled(x)
    assert skinny.call_count > before_capture
    before_replay = skinny.call_count
    x.add_(0.5)
    graph.replay()
    torch.accelerator.synchronize()
    assert skinny.call_count == before_replay  # replay executes the captured kernel
    ref = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=2e-2)


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch, m):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", range(1, 6))
def test_rocm_unquantized_gemm_single_output_real_kernel(dtype, n):
    torch.manual_seed(0)
    x = torch.randn(n, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(1, 2048, device="cuda", dtype=dtype)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    from vllm.platforms.rocm import on_gfx10x

    if on_gfx10x():
        # RDNA2 uses its own FP16 kernel and rocBLAS for this BF16 shape.
        wvsplitk_mock.assert_not_called()
    else:
        wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch, m):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # fall back to torch.nn.functional.linear.
    x = torch.randn(6, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
