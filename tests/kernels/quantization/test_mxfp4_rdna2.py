#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 fused MoE W4A4 MXFP4 HIP kernel (gfx1030).

Tests ``moe_mxfp4_gemm_rdna2`` (fused MoE) and ``mxfp4_gemm_rdna2`` (dense)
against a pure-Python reference that:
  1. Dequantizes 8 E2M1 nibbles → 8 fp16 values via the OCP MXFP4 LUT.
  2. Applies the per-32-block UE8M0 power-of-two scale.
  3. Performs the GEMM with ``numpy.dot``.

The kernel is fp16-only (gfx1030 has no v_dot2_f32_bf16). bf16 variants
are explicitly skipped.

The kernel uses 64-bit CAS-loop atomic adds for the output reduction;
fp16 accumulation across split-K tiles can introduce ~1 ULP rounding
differences, so we use a generous ``atol`` (similar to the W4A16 RDNA2
tests).

Model parameters taken from DeepSeek V4 Flash:
  - hidden=2048, intermediate=512, E=128, top_k=8 (experts capped at 16
    for test memory; kernel behavior is E-independent)

Run ``pytest tests/kernels/quantization/test_mxfp4_rdna2.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MXFP4 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402

device = "cuda"

# Skip the entire module unless we're on gfx1030 AND the ops are registered.
gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_mxfp4_gemm_rdna2")
        and hasattr(torch.ops._rocm_C, "mxfp4_gemm_rdna2")
    ),
    reason="Requires gfx1030 with moe_mxfp4_gemm_rdna2 + mxfp4_gemm_rdna2 ops",
)

# --- OCP MXFP4 E2M1 → fp16 LUT (matches mxfp4_dot2_common.cuh:e2m1_lut) ---
# E2M1 element layout (4 bits, LSB first): s e e m
# FP16 representation of each of the 16 possible nibble values.
E2M1_LUT_FP16 = torch.tensor(
    [
        0x0000,  # 0 0 0 0 = +0
        0x3800,  # 0 0 0 1 = +0.5
        0x3C00,  # 0 0 1 0 = +1.0
        0x3E00,  # 0 0 1 1 = +1.5
        0x4000,  # 0 1 0 0 = +2.0
        0x4400,  # 0 1 0 1 = +3.0
        0x4800,  # 0 1 1 0 = +4.0
        0x4C00,  # 0 1 1 1 = +6.0
        0x8000,  # 1 0 0 0 = -0
        0xB800,  # 1 0 0 1 = -0.5
        0xBC00,  # 1 0 1 0 = -1.0
        0xBE00,  # 1 0 1 1 = -1.5
        0xC000,  # 1 1 0 0 = -2.0
        0xC400,  # 1 1 0 1 = -3.0
        0xC800,  # 1 1 1 0 = -4.0
        0xCC00,  # 1 1 1 1 = -6.0
    ],
    dtype=torch.int32,
)

# Model configurations: real K/N/top_k/group_size=32 from DeepSeek V4 Flash,
# with E capped to 16 for test memory. Kernel behavior is E-independent.
MODEL_CONFIGS = [
    pytest.param(16, 2048, 512, 8, id="DSV4-Flash-like"),
]

NUM_TOKENS = [1, 4, 16, 64]


def _ue8m0_to_fp16_scale(scale_byte: torch.Tensor) -> torch.Tensor:
    """Convert uint8 UE8M0 power-of-two exponent to fp16 scale.

    Mirrors ``ue8m0_to_fp16`` in mxfp4_dot2_common.cuh:
      bits = (scale_byte - 112) << 10

    Returns a float16 tensor with the same shape as ``scale_byte``.
    """
    bits = (scale_byte.to(torch.int32) - 112) << 10
    return bits.to(torch.int32).view(torch.float16)


def _make_e2m1_weights(E: int, K: int, N: int) -> torch.Tensor:
    """Create random E2M1-packed weights [E, K/8, N] uint32.

    8 E2M1 nibbles per uint32 word, LSB-first (matches mxfp4_dot2_moe.cu
    and OCP MX spec).
    """
    # Generate random fp16 values, round to nearest E2M1, repack.
    import numpy as np

    rng = np.random.default_rng(seed=42)
    # Random fp16 in [-1, 1]; skip subnormals (rare in E2M1 anyway).
    vals = rng.uniform(-1.0, 1.0, size=(E, K, N)).astype(np.float16)
    # Quantize to nearest E2M1 value: pick the LUT entry with min |diff|.
    lut = E2M1_LUT_FP16.to(torch.float32).numpy().view(np.float32)
    # Reshape for broadcast compare: (E, K, N, 1) vs (16,)
    vals_f32 = vals.astype(np.float32)
    diffs = np.abs(vals_f32[..., None] - lut[None, None, None, :])
    nibbles = diffs.argmin(axis=-1).astype(np.uint8)  # (E, K, N)

    # Pack 8 nibbles per uint32, LSB-first.
    packed = np.zeros((E, K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= nibbles[:, i::8, :].astype(np.uint32) << (i * 4)
    return torch.from_numpy(packed).to(device)


def _make_ue8m0_scales(E: int, K: int, N: int) -> torch.Tensor:
    """Create random UE8M0 scales [E, K/32, N] uint8 in [120, 140].

    Range chosen so the resulting fp16 scales cover the typical LLM
    weight magnitude (≈ 2^-7 to 2^7) — fits cleanly in fp16 normal range.
    """
    import numpy as np

    rng = np.random.default_rng(seed=43)
    return torch.from_numpy(
        rng.integers(120, 141, size=(E, K // 32, N), dtype=np.uint8)
    ).to(device)


def _dequant_reference(
    packed_weights: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize [E, K/8, N] uint32 + [E, K/32, N] uint8 -> [E, K, N] fp16.

    Pure-Python reference using the same LUT/scale arithmetic as the
    kernel. Used to build the numpy.dot reference GEMM.
    """
    import numpy as np

    E, K8, N = packed_weights.shape
    K = K8 * 8

    # Unpack 8 nibbles per uint32, LSB-first.
    nibbles = np.zeros((E, K, N), dtype=np.int64)
    words = packed_weights.cpu().numpy().view(np.uint32)
    for i in range(8):
        nibbles[:, i::8, :] = (words >> (i * 4)) & 0xF
    nibbles_t = torch.from_numpy(nibbles).to(device).to(torch.long)

    # LUT lookup: (E, K, N) -> fp16
    lut = E2M1_LUT_FP16.to(device).to(torch.int32)
    # Build sign-extension properly. The high bit is sign; we want the
    # fp16 value, which already has the sign bit set in the LUT, so we
    # just index.
    e2m1_vals = lut[nibbles_t].to(torch.float32)

    # Scale: [E, K/32, N] -> broadcast to [E, K, N]
    scale_fp16 = _ue8m0_to_fp16_scale(scales).to(torch.float32)  # (E, K/32, N)
    # Per-block scale broadcast
    scale_broadcast = scale_fp16.repeat_interleave(32, dim=1)  # (E, K, N)

    # Final dequantized weight (E, K, N) fp16
    out = (e2m1_vals * scale_broadcast).to(torch.float16)
    assert out.shape == (E, K, N)
    return out


def _reference_moe_gemm(
    x: torch.Tensor,  # [M, K]
    w_packed: torch.Tensor,  # [E, K/8, N] uint32
    scales: torch.Tensor,  # [E, K/32, N] uint8
    topk_ids: torch.Tensor,  # [M, top_k] int32
    top_k: int,
) -> torch.Tensor:
    """Pure-Python reference: per-expert dense GEMM with moe_sum fusion.

    Output shape: [M, N] fp16 (after moe_sum).
    """
    import numpy as np

    M, K = x.shape
    E, _, N = w_packed.shape

    # Dequantize all expert weights once.
    w_dq = _dequant_reference(w_packed, scales)  # (E, K, N) fp16

    # Per-token routing: out[m] = sum over k of x[m] @ w_dq[topk_ids[m, k]]
    x_np = x.to(torch.float32).cpu().numpy().astype(np.float16)
    w_dq_np = w_dq.cpu().numpy()
    out = np.zeros((M, N), dtype=np.float16)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            # x[m]: (K,), w_dq[e]: (K, N) → (N,)
            out[m] += (x_np[m].astype(np.float32) @ w_dq_np[e].astype(np.float32)).astype(np.float16)
    return torch.from_numpy(out).to(device).to(torch.float16)


# ---------------------------------------------------------------------------
# Dense GEMM tests (no MoE routing, no topk)
# ---------------------------------------------------------------------------


@gfx1030_only
@pytest.mark.parametrize("K, N", [(256, 256), (2048, 512), (1024, 256)])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
def test_dense_mxfp4_matches_reference(K, N, M):
    """Dense ``mxfp4_gemm_rdna2`` matches pure-Python E2M1 dequant + numpy.dot."""
    import numpy as np

    torch.manual_seed(0)
    # Single expert (dense layer)
    w_packed = _make_e2m1_weights(1, K, N)
    scales = _make_ue8m0_scales(1, K, N)
    # Note: for the dense kernel the weight is [K/8, N] (no E dim). Squeeze.
    w_packed_2d = w_packed.squeeze(0).contiguous()  # [K/8, N]
    scales_2d = scales.squeeze(0).contiguous()  # [K/32, N]

    x = torch.randn(M, K, dtype=torch.float16, device=device)

    # Pre-zero output (kernel uses atomic adds)
    c = torch.zeros(M, N, dtype=torch.float16, device=device)

    ops.mxfp4_gemm_rdna2(x, c, w_packed_2d, scales_2d, M, N, K)
    torch.cuda.synchronize()

    # Reference: dequant + numpy.dot
    w_dq = _dequant_reference(w_packed, scales).squeeze(0).to(torch.float32)
    x_f32 = x.to(torch.float32)
    ref = (x_f32 @ w_dq).to(torch.float16)

    # fp16 accumulation + split-K atomic adds → generous tolerance
    atol = 0.5
    assert torch.allclose(c, ref, atol=atol, rtol=0.05), (
        f"max diff: {(c - ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Fused MoE tests (routing + w1 + activation + w2 with moe_sum)
# ---------------------------------------------------------------------------


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_mxfp4_moe_w1_matches_reference(E, K, N_inter, top_k, M, block_size_m):
    """Fused MoE w1 matches per-expert dense kernel + moe_sum."""
    N_gate_up = N_inter * 2

    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w13_packed = _make_e2m1_weights(E, K, N_gate_up)
    w13_scales = _make_ue8m0_scales(E, K, N_gate_up)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    # Fused kernel (no output_topk: write [M*top_k, N])
    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        x,
        fused_out,
        w13_packed,
        w13_scales,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
    )

    # Reference: per-token routing + numpy.dot
    ref_flat = _reference_moe_gemm(x, w13_packed, w13_scales, topk_ids, top_k)
    # _reference_moe_gemm returns [M, N] (after moe_sum). For comparison
    # without moe_sum, build the per-(m, k) result separately.
    import numpy as np

    w_dq = _dequant_reference(w13_packed, w13_scales).to(torch.float32)
    x_np = x.to(torch.float32)
    ref_no_sum = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            flat = m * top_k + k
            ref_no_sum[flat] = (x_np[m] @ w_dq[e]).to(torch.float16)

    atol = 0.5
    assert torch.allclose(fused_out, ref_no_sum, atol=atol, rtol=0.05), (
        f"max diff: {(fused_out - ref_no_sum).abs().max().item()}"
    )


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
def test_fused_mxfp4_moe_output_topk_reduces(E, K, N_inter, top_k, M):
    """output_topk fuses moe_sum into the w2 kernel epilogue."""
    torch.manual_seed(2)
    x = torch.randn(M * top_k, K, dtype=torch.float16, device=device)
    w = _make_e2m1_weights(E, K, N_inter)
    ws = _make_ue8m0_scales(E, K, N_inter)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Without output_topk: write to [M*top_k, N] then moe_sum (reference)
    flat_out = torch.zeros(M * top_k, N_inter, dtype=torch.float16, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        x,
        flat_out,
        w,
        ws,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        0,
    )
    ref = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_sum(flat_out.view(M, top_k, N_inter), ref)

    # With output_topk: write directly to [M, N] (fused moe_sum)
    fused = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        x,
        fused,
        w,
        ws,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    atol = 0.5
    assert torch.allclose(fused, ref, atol=atol, rtol=0.05), (
        f"max diff: {(fused - ref).abs().max().item()}"
    )


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k", MODEL_CONFIGS)
@pytest.mark.parametrize("M", [4, 16])
def test_full_mxfp4_moe_e2e(E, K, N_inter, top_k, M):
    """Full MoE forward: w1 + silu_and_mul + w2 with output_topk reduce."""
    N_gate_up = N_inter * 2
    hidden = K

    torch.manual_seed(3)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w13_packed = _make_e2m1_weights(E, K, N_gate_up)
    w13_scales = _make_ue8m0_scales(E, K, N_gate_up)
    w2_packed = _make_e2m1_weights(E, N_inter, hidden)
    w2_scales = _make_ue8m0_scales(E, N_inter, hidden)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Fused path
    w1_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        x,
        w1_out,
        w13_packed,
        w13_scales,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        1,
        False,
        0,
    )
    act_out = torch.empty(M * top_k, N_inter, dtype=torch.float16, device=device)
    apply_moe_activation(MoEActivation.SILU, act_out, w1_out)

    out = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        act_out,
        out,
        w2_packed,
        w2_scales,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    # Reference: dequant + numpy
    w13_dq = _dequant_reference(w13_packed, w13_scales).to(torch.float32)
    w2_dq = _dequant_reference(w2_packed, w2_scales).to(torch.float32)
    x_f32 = x.to(torch.float32)
    ref = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    for m in range(M):
        # w1 contribution
        gate_up = torch.zeros(N_gate_up, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            gate_up += x_f32[m] @ w13_dq[e]
        # silu_and_mul
        gate = gate_up[:N_inter]
        up = gate_up[N_inter:]
        act = gate * torch.nn.functional.silu(gate * up)  # gated silu
        # w2 contribution (with topk weight)
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            ref[m] += (topk_w[m, k] * (act.to(torch.float32) @ w2_dq[e])).to(torch.float16)

    atol = 1.0  # larger tolerance: 2 GEMMs + activation + topk_weight
    assert torch.allclose(out, ref, atol=atol, rtol=0.1), (
        f"max diff: {(out - ref).abs().max().item()}"
    )
