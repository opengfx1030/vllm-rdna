# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gemv_f16_rdna2 (gfx10x fp16 skinny GEMM) matches torch for M in 1..32."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("gemv_f16_rdna2 is ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gemv_f16_rdna2")
    ),
    reason="Requires gfx10x with gemv_f16_rdna2",
)


# 1..8 use exact-row kernels, 9..32 the 16/24/32 row capacities.
@pytest.mark.parametrize("M", [1, 3, 8, 9, 12, 16, 17, 24, 32])
@pytest.mark.parametrize("N, K", [(4096, 512), (300, 1024), (62080, 256)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemv_f16_rdna2_matches_linear(M, N, K, with_bias):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.half, device="cuda")
    w = torch.randn(N, K, dtype=torch.half, device="cuda") * 0.05
    bias = torch.randn(N, dtype=torch.half, device="cuda") if with_bias else None
    out = torch.ops._rocm_C.gemv_f16_rdna2(x, w, bias)
    ref = torch.nn.functional.linear(x.float(), w.float(), bias=None)
    if bias is not None:
        ref = ref + bias.float()
    assert out.shape == (M, N)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not hasattr(torch.ops._rocm_C, "gemv_bf16_rdna2"),
    reason="Requires rebuilt gemv_bf16_rdna2 op",
)
@pytest.mark.parametrize("M", [3, 4, 24])
@pytest.mark.parametrize("N, K", [(2560, 1024), (640, 2560), (256, 2560)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemv_bf16_rdna2_matches_fp32_linear(M, N, K, with_bias):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda") if with_bias else None
    out = torch.ops._rocm_C.gemv_bf16_rdna2(x, w, bias)
    ref = torch.nn.functional.linear(
        x.float(), w.float(), bias.float() if bias is not None else None
    )
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=2e-2)


@pytest.mark.skipif(
    not hasattr(torch.ops._rocm_C, "gemv_bf16_rdna2"),
    reason="Requires rebuilt gemv_bf16_rdna2 op",
)
def test_gemv_bf16_rdna2_preserves_full_bf16_range_and_alignment():
    # A half shadow of x would overflow here; BF16 and FP32 remain finite.
    x = torch.full((3, 1024), 2**20, dtype=torch.bfloat16, device="cuda")
    w = torch.full((256, 1024), 2**-20, dtype=torch.bfloat16, device="cuda")
    out = torch.ops._rocm_C.gemv_bf16_rdna2(x, w, None)
    torch.testing.assert_close(out.float(), torch.full_like(out.float(), 1024))
    large_out = torch.ops._rocm_C.gemv_bf16_rdna2(x, torch.ones_like(w), None)
    torch.testing.assert_close(
        large_out.float(), torch.full_like(large_out.float(), 2**30)
    )

    # A contiguous view can still start two bytes past a 16-byte boundary.
    shifted = torch.empty(3 * 1024 + 1, dtype=torch.bfloat16, device="cuda")
    unaligned = shifted[1:].view(3, 1024)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 != 0
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops._rocm_C.gemv_bf16_rdna2(unaligned, w, None)


@pytest.mark.skipif(
    not hasattr(torch.ops._rocm_C, "gemv_bf16_rdna2"),
    reason="Requires rebuilt gemv_bf16_rdna2 op",
)
@pytest.mark.parametrize("N", [640, 256])
def test_gemv_bf16_rdna2_m24_row_tiles_graph_replay(N):
    torch.manual_seed(1)
    x = torch.randn(24, 2560, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, 2560, dtype=torch.bfloat16, device="cuda") * 0.05
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        torch.ops._rocm_C.gemv_bf16_rdna2(x, w, bias)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops._rocm_C.gemv_bf16_rdna2(x, w, bias)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.accelerator.synchronize()
    ref = torch.nn.functional.linear(x.float(), w.float(), bias.float())
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=2e-2)
