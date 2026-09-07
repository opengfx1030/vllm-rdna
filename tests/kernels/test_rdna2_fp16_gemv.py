# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the RDNA2 (gfx1030) fp16 GEMV vs torch.linear.

Runs in seconds (no model load). Use this to confirm the Triton GEMV produces
correct output and does not trap on gfx1030 before restarting the full server.
"""

import pytest
import torch

from vllm.model_executor.layers.rdna2_gemm import rdna2_gemv
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x

DEVICE = torch.device("cuda")

# Shapes covering the unquantized decode projections (incl. Qwen3_5 GDN/attn).
SHAPES = [
    (5120, 5120),
    (13824, 5120),
    (5120, 13824),
    (4096, 4096),
    (1024, 4096),
    (256, 256),
]


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_rdna2_gemv_vs_linear(N, K, with_bias):
    torch.manual_seed(0)
    x = torch.randn(K, dtype=torch.float16, device=DEVICE)
    w = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
    b = torch.randn(N, dtype=torch.float16, device=DEVICE) if with_bias else None

    ref = torch.nn.functional.linear(x, w, b)
    out = rdna2_gemv(x, w, b)

    assert out.shape == ref.shape
    assert out.isfinite().all(), "GEMV produced inf/nan (kernel may have trapped)"
    max_diff = (out.float() - ref.float()).abs().max().item()
    # fp16 GEMV vs fp32-accum reference: tolerant bound for large K.
    assert max_diff < 1.0, f"N={N} K={K} bias={with_bias}: max_diff={max_diff}"


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
def test_rdna2_gemv_smoke():
    """Smoke test: just confirm it runs without faulting on gfx1030."""
    x = torch.randn(128, dtype=torch.float16, device=DEVICE)
    w = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    out = rdna2_gemv(x, w, None)
    assert out.shape == (64,)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("M", [2, 3, 5, 8, 16])
@pytest.mark.parametrize("N,K", [(512, 1024), (1536, 2048), (4096, 2176)])
def test_rdna2_skinny_matmul_vs_linear(M, N, K):
    """Small-M matmul (spec-decode verify / MTP batches) vs F.linear."""
    from vllm.model_executor.layers.rdna2_gemm import rdna2_skinny_matmul

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1

    ref = torch.nn.functional.linear(a, b)
    out = rdna2_skinny_matmul(a, b)

    assert out.shape == ref.shape
    assert out.isfinite().all(), "skinny matmul produced inf/nan"
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 1.0, f"M={M} N={N} K={K}: max_diff={max_diff}"


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
def test_rdna2_skinny_matmul_matches_gemv():
    """Per-row consistency between the M=1 GEMV and the tl.dot matmul."""
    from vllm.model_executor.layers.rdna2_gemm import rdna2_skinny_matmul

    torch.manual_seed(0)
    N, K, M = 1024, 2048, 4
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
    out = rdna2_skinny_matmul(a, b)
    ref = torch.stack([rdna2_gemv(a[i], b, None) for i in range(M)])
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 1.0, f"max_diff={max_diff}"


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize(
    "N,K,gs",
    [(8704, 5120, 32), (5120, 8704, 32), (4096, 5120, 32),
     (5120, 4096, 32), (1024, 2048, 64), (512, 1024, 128)],
)
@pytest.mark.parametrize("has_zp", [True, False])
def test_rdna2_w4a16_gemv(N, K, gs, has_zp):
    """M=1 W4A16 GEMV vs dequant reference (fp32 math)."""
    from vllm.model_executor.layers.rdna2_gemm import rdna2_w4a16_gemv

    def pack_along_last(t):
        R, C = t.shape
        out = torch.zeros(R, C // 8, dtype=torch.int32, device=t.device)
        for i in range(8):
            out |= t[:, i::8].to(torch.int32) << (4 * i)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    qweight = pack_along_last(w4.t().contiguous())  # (K, N//8)
    scales = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    zp_bias = 8
    if has_zp:
        zeros_v = torch.randint(
            0, 16, (K // gs, N), dtype=torch.int32, device=DEVICE
        )
        qzeros = pack_along_last(zeros_v)  # (K//gs, N//8)
        zero_ref = zeros_v
    else:
        qzeros = None
        zero_ref = torch.full(
            (K // gs, N), zp_bias, dtype=torch.int32, device=DEVICE
        )

    x = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1
    out = rdna2_w4a16_gemv(x, qweight, scales, qzeros, gs, zp_bias)

    # dequant: w[n, k] = (w4[n, k] - zero[k // gs, n]) * scale[k // gs, n]
    z_nk = zero_ref.t().to(torch.float32).repeat_interleave(gs, dim=1)
    s_nk = scales.t().to(torch.float32).repeat_interleave(gs, dim=1)
    w_deq = (w4.to(torch.float32) - z_nk) * s_nk
    ref = x.float() @ w_deq.T
    torch.testing.assert_close(out.float(), ref, atol=2e-1, rtol=2e-2)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("M", [2, 3, 4])
@pytest.mark.parametrize("N,K,gs", [(8704, 5120, 32), (5120, 8704, 32)])
def test_rdna2_w4a16_gemv_m(M, N, K, gs):
    """Small-M W4A16 GEMV (spec-decode verify batches) vs dequant ref."""
    from vllm.model_executor.layers.rdna2_gemm import rdna2_w4a16_gemv_m

    def pack_along_last(t):
        R, C = t.shape
        out = torch.zeros(R, C // 8, dtype=torch.int32, device=t.device)
        for i in range(8):
            out |= t[:, i::8].to(torch.int32) << (4 * i)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    qweight = pack_along_last(w4.t().contiguous())
    scales = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    zeros_v = torch.randint(0, 16, (K // gs, N), dtype=torch.int32, device=DEVICE)
    qzeros = pack_along_last(zeros_v)
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE) * 0.1
    out = rdna2_w4a16_gemv_m(x, qweight, scales, qzeros, gs, 8)
    z_nk = zeros_v.t().to(torch.float32).repeat_interleave(gs, dim=1)
    s_nk = scales.t().to(torch.float32).repeat_interleave(gs, dim=1)
    w_deq = (w4.to(torch.float32) - z_nk) * s_nk
    ref = x.float() @ w_deq.T
    torch.testing.assert_close(out.float(), ref, atol=2e-1, rtol=2e-2)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize(
    "N,K,gs",
    [(8704, 5120, 32), (5120, 8704, 32), (4096, 5120, 32),
     (5120, 4096, 32), (1024, 2048, 64)],
)
def test_gemv_w4_kpack(N, K, gs):
    """HIP K-packed M=1 GEMV vs chunked dequant reference."""
    from vllm import _custom_ops as ops

    def pack_k(t_nk):
        t = t_nk.t().contiguous().to(torch.int32)
        out = torch.zeros(K // 8, N, dtype=torch.int32, device=t.device)
        for j in range(8):
            out |= t[j::8] << (4 * j)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    wq = pack_k(w4)
    s = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    z8 = torch.randint(0, 16, (K // gs, N), dtype=torch.uint8, device=DEVICE)
    x = torch.randn(1, K, dtype=torch.float16, device=DEVICE) * 0.1

    out = ops.gemv_w4_kpack(x, wq, s, z8, gs, 8)[0].float()

    z_nk = z8.t().to(torch.float32).repeat_interleave(gs, dim=1)
    s_nk = s.t().to(torch.float32).repeat_interleave(gs, dim=1)
    ref = x[0].float() @ ((w4.float() - z_nk) * s_nk).T
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("M", [2, 3, 4, 33])
@pytest.mark.parametrize("N,K,gs", [(8704, 5120, 32), (5120, 8704, 32)])
def test_w4a16_kpack_gemm(M, N, K, gs):
    """K-packed Triton GEMM (M>1) vs chunked dequant reference."""
    from vllm.model_executor.layers.rdna2_gemm import w4a16_kpack_gemm

    def pack_k(t_nk):
        t = t_nk.t().contiguous().to(torch.int32)
        out = torch.zeros(K // 8, N, dtype=torch.int32, device=t.device)
        for j in range(8):
            out |= t[j::8] << (4 * j)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    wq = pack_k(w4)
    s = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    z8 = torch.randint(0, 16, (K // gs, N), dtype=torch.uint8, device=DEVICE)
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE) * 0.1

    out = w4a16_kpack_gemm(x, wq, s, z8, gs).float()

    z_nk = z8.t().to(torch.float32).repeat_interleave(gs, dim=1)
    s_nk = s.t().to(torch.float32).repeat_interleave(gs, dim=1)
    ref = x.float() @ ((w4.float() - z_nk) * s_nk).T
    torch.testing.assert_close(out, ref, atol=3e-1, rtol=3e-2)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("M", [2, 3, 4])
@pytest.mark.parametrize("N,K,gs", [(8704, 5120, 32), (5120, 8704, 32)])
def test_rdna2_w4a16_kpack_gemv_m(M, N, K, gs):
    """Small-M K-packed GEMV (MTP verify path under kpack) vs dequant ref."""
    from vllm.model_executor.layers.rdna2_gemm import (
        rdna2_w4a16_kpack_gemv_m,
    )

    def pack_k(t_nk):
        t = t_nk.t().contiguous().to(torch.int32)
        out = torch.zeros(K // 8, N, dtype=torch.int32, device=t.device)
        for j in range(8):
            out |= t[j::8] << (4 * j)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    wq = pack_k(w4)
    s = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    z8 = torch.randint(0, 16, (K // gs, N), dtype=torch.uint8, device=DEVICE)
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE) * 0.1

    out = rdna2_w4a16_kpack_gemv_m(x, wq, s, z8, gs).float()

    z_nk = z8.t().to(torch.float32).repeat_interleave(gs, dim=1)
    s_nk = s.t().to(torch.float32).repeat_interleave(gs, dim=1)
    ref = x.float() @ ((w4.float() - z_nk) * s_nk).T
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)
    assert out.isfinite().all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("M", [2, 3])
@pytest.mark.parametrize("N,K,gs", [(8704, 5120, 32)])
def test_rdna2_w4a16_kpack_gemv_m_nozp(M, N, K, gs):
    """Small-M K-packed GEMV, symmetric path (ZP_BIAS, no zeros tensor)."""
    from vllm.model_executor.layers.rdna2_gemm import (
        rdna2_w4a16_kpack_gemv_m,
    )

    def pack_k(t_nk):
        t = t_nk.t().contiguous().to(torch.int32)
        out = torch.zeros(K // 8, N, dtype=torch.int32, device=t.device)
        for j in range(8):
            out |= t[j::8] << (4 * j)
        return out.contiguous()

    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEVICE)
    wq = pack_k(w4)
    s = torch.rand(K // gs, N, dtype=torch.float16, device=DEVICE) * 0.01 + 0.005
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE) * 0.1

    out = rdna2_w4a16_kpack_gemv_m(x, wq, s, None, gs, zp_bias=8.0).float()

    ref = x.float() @ ((w4.float() - 8.0) * s.t().repeat_interleave(gs, dim=1)).T
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)
    assert out.isfinite().all()
