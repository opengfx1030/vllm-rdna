#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 fused MoE W8A16-FP8 HIP kernel.

Tests ``moe_w8a16_fp8_gemm_rdna2``. Storage stays FP8 (no requant on
disk); per-tile FP8->fp16 conversion inside the kernel before v_dot2.

bf16 is not supported on gfx1030 (lacks v_dot2_f32_bf16 intrinsic).

Run `pytest tests/kernels/quantization/test_rdna2_w8a16_fp8_moe.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE W8A16-FP8 kernel is ROCm-only",
                allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402

device = "cuda"

gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_w8a16_fp8_gemm_rdna2")
    ),
    reason="Requires gfx1030 with moe_w8a16_fp8_gemm_rdna2 op",
)

MODEL_CONFIGS = [
    pytest.param(16, 2048, 768, 8, 32, id="Qwen3-30B-A3B"),
    pytest.param(16, 2048, 512, 8, 32, id="Qwen3.6-35B-A3B"),
]

NUM_TOKENS = [1, 4, 16, 64, 256]


# FP8 e4m3 max representable: 448.0. Block-quantized scales are
# typically small (< 1.0 for activations and weights after per-tensor
# or per-channel quant). Use a small range to keep FP8 representable.
FP8_MAX = 448.0


def _make_fp8_weights(E, K, N, scale=0.01):
    """Create random FP8 e4m3 weights [E, K, N]."""
    w = torch.randn(E, K, N, dtype=torch.float32, device=device) * scale
    w = w.clamp(-FP8_MAX, FP8_MAX)
    return w.to(torch.float8_e4m3fn)


def _make_fp8_zeros(E, groups, N, scale=0.01):
    """Create zero tensor with FP8 dtype (asymmetric quant can use this)."""
    z = torch.zeros(E, groups, N, dtype=torch.float32, device=device)
    return z.to(torch.float8_e4m3fn)


def _make_scales_fp16(E, groups, N):
    return torch.rand(E, groups, N, dtype=torch.float16, device=device) * 0.01 + 0.001


def _fp8_ref(x_fp16, w_fp8, scale, zero, group_size):
    """Reference: dequantize FP8 -> fp16, then GEMM."""
    E, K, N = w_fp8.shape
    K_groups = K // group_size
    w_dequant = w_fp8.to(torch.float32).view(E, K_groups, group_size, N)
    s = scale.to(torch.float32).view(E, K_groups, 1, N)
    z = zero.to(torch.float32).view(E, K_groups, 1, N)
    w_dequant = (w_dequant - z) * s
    w_dequant = w_dequant.view(E, K, N)
    return (x_fp16.float() @ w_dequant.transpose(1, 2))


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_moe_w8a16_fp8_w1_matches_dense(
    E, K, N_inter, top_k, group_size, M, block_size_m
):
    """w1 GEMM via fused kernel matches per-expert dense reference."""
    N_gate_up = N_inter * 2
    groups = K // group_size

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w13 = _make_fp8_weights(E, K, N_gate_up)
    w13_s = _make_scales_fp16(E, groups, N_gate_up)
    w13_z = _make_fp8_zeros(E, groups, N_gate_up)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_w8a16_fp8_gemm_rdna2(
        x,
        fused_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
    )

    # Per-expert dense reference
    for m in range(M):
        for k in range(top_k):
            e = topk_ids[m, k].item()
            flat = m * top_k + k
            w_fp = _fp8_ref(
                x[m : m + 1], w13[e : e + 1], w13_s[e : e + 1],
                w13_z[e : e + 1], group_size,
            ).squeeze(0)
            torch.testing.assert_close(
                fused_out[flat].float().cpu(),
                w_fp.float().cpu(),
                atol=3.0,
                rtol=0.05,
            )


@gfx1030_only
def test_w8a16_fp8_bf16_rejected():
    """Passing bf16 activations must fail at the kernel."""
    if not on_gfx10x():
        pytest.skip("gfx1030 only")
    N = 8
    K = 8
    E = 1
    G = 1
    M = 1

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.zeros(E, K, N, dtype=torch.float8_e4m3fn, device=device)
    s = torch.ones(E, G, N, dtype=torch.float16, device=device)
    z = torch.zeros(E, G, N, dtype=torch.float8_e4m3fn, device=device)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    si, ei, ntp = moe_align_block_size(
        torch.zeros(M, 1, dtype=torch.int32, device=device), 1, E)

    with pytest.raises(RuntimeError, match="fp16"):
        ops.moe_w8a16_fp8_gemm_rdna2(x, out, w, s, z,
                                      torch.empty(0, device=device),
                                      si, ei, ntp, 1, 1, False, 0)
