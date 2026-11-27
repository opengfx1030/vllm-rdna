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
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gemv_i8_rdna2")
    ),
    reason="Requires gfx10x with gemv_i8_rdna2",
)
@pytest.mark.parametrize("M", [1, 4, 8, 9, 12, 13, 16, 17, 24, 25, 32])
@pytest.mark.parametrize("N, K", [(320, 1024), (1152, 2560)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemv_i8_rdna2_matches_linear(M, N, K, with_bias):
    torch.manual_seed(10 + M + N)
    x = torch.randn(M, K, dtype=torch.half, device="cuda")
    w_i8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    scale = (torch.rand(N, dtype=torch.half, device="cuda") + 0.1) * 0.01
    bias = torch.randn(N, dtype=torch.half, device="cuda") if with_bias else None
    out = torch.ops._rocm_C.gemv_i8_rdna2(x, w_i8, scale, bias)
    ref = (x.float() @ w_i8.float().T) * scale.float()
    if bias is not None:
        ref = ref + bias.float()
    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)
