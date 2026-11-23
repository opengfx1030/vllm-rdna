#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the RDNA2 (gfx1030) MoE kernels used by
Inkling-2-Mini experts: ``moe_nvfp4_gemm_rdna2`` and ``moe_mxfp8_gemm_rdna2``.

Both formats are quantised here from a random fp32 weight with the exact
formula the checkpoint uses, then dequantised again in torch as the reference:

  NVFP4   value = e2m1(nibble) * e4m3(scale[k/16]) * (amax / 2688)
  MXFP8   value = e4m3(byte) * 2^(ue8m0(scale[k/32]) - 127)

The kernel accumulates fp16 via 64-bit CAS atomics across split-K tiles, so the
tolerance is the same generous one the MXFP4/W4A16 RDNA2 tests use.

Run ``pytest tests/kernels/quantization/test_rdna2_moe_lowp.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "RDNA2 low-precision MoE kernels are ROCm-only", allow_module_level=True
    )

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402

device = "cuda"

_have_ops = (
    hasattr(torch.ops, "_rocm_C")
    and hasattr(torch.ops._rocm_C, "moe_nvfp4_gemm_rdna2")
    and hasattr(torch.ops._rocm_C, "moe_mxfp8_gemm_rdna2")
)
gfx1030_only = pytest.mark.skipif(
    not (on_gfx10x() and _have_ops),
    reason="Requires gfx1030 with moe_nvfp4_gemm_rdna2 + moe_mxfp8_gemm_rdna2",
)

# OCP E2M1 magnitudes; index is the 3-bit magnitude field.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    """[..., ] int nibbles -> float values (bit 3 is the sign)."""
    mag = E2M1.to(nibbles.device)[nibbles & 0x7]
    return torch.where(nibbles & 0x8 != 0, -mag, mag)


def _e4m3_decode(bytes_u8: torch.Tensor) -> torch.Tensor:
    b = bytes_u8.to(torch.int32)
    sign = torch.where(b & 0x80 != 0, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 0x7).to(torch.float32) / 8.0
    val = torch.where(e == 0, m * 2.0**-6, (1.0 + m) * torch.pow(2.0, (e - 7).float()))
    return sign * val


def _e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Nearest E4M3 byte (normals only, magnitudes in [2^-6, 448])."""
    ax = x.abs().clamp(min=2.0**-9, max=448.0)
    e = torch.floor(torch.log2(ax)).clamp(-6, 8)
    m = ((ax / torch.pow(2.0, e) - 1.0) * 8.0).round().clamp(0, 7)
    e = torch.where(m == 8, e + 1, e)
    m = torch.where(m == 8, torch.zeros_like(m), m)
    byte = (((e + 7).to(torch.int32) & 0xF) << 3) | m.to(torch.int32)
    return torch.where(x < 0, byte | 0x80, byte).to(torch.uint8)


def quantise_nvfp4(w: torch.Tensor, block: int = 16):
    """[E, K, N] fp32 -> packed [E, K/8, N] uint32, [E, K/16, N] e4m3, [E] f32."""
    E, K, N = w.shape
    amax = w.abs().amax(dim=(1, 2))
    glob = amax / 2688.0
    wb = w.reshape(E, K // block, block, N)
    bmax = wb.abs().amax(dim=2)
    s = (bmax / 6.0 / glob.view(E, 1, 1)).clamp(min=1e-30)
    scales = _e4m3_encode(s)
    s_val = _e4m3_decode(scales).to(torch.float32)
    unit = s_val * glob.view(E, 1, 1)  # per block value of one LUT step
    step = unit.unsqueeze(2)  # [E, K/block, 1, N]
    norm = (wb / step.clamp(min=1e-30)).clamp(-6, 6)
    mag = norm.abs()
    # nearest LUT magnitude (0, .5, 1, 1.5, 2, 3, 4, 6); index == code
    idx = (mag.unsqueeze(-1) - E2M1.to(mag.device)).abs().argmin(-1)
    code = torch.where(norm < 0, idx | 0x8, idx).to(torch.int64)
    code = code.reshape(E, K, N)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int64, device=w.device)
    for i in range(8):
        packed[:, :, :] |= code[:, i::8, :] << (4 * i)
    return packed.to(torch.uint32).contiguous(), scales.contiguous(), glob.float()


def quantise_mxfp8(w: torch.Tensor, block: int = 32):
    """[E, K, N] fp32 -> [E, K/4, N] uint32 e4m3 words, [E, K/32, N] uint8."""
    E, K, N = w.shape
    wb = w.reshape(E, K // block, block, N)
    bmax = wb.abs().amax(dim=2).clamp(min=1e-30)
    exp = torch.round(torch.log2(bmax / 6.0)).clamp(-14, 3)
    scales = (exp + 127).to(torch.uint8)
    sc = torch.pow(2.0, exp)
    bytes_u8 = _e4m3_encode(wb / sc.unsqueeze(2).clamp(min=1e-30)).reshape(E, K, N)
    # pack 4 consecutive K bytes per uint32 word, K-first
    words = bytes_u8.to(torch.int32).reshape(E, K // 4, 4, N)
    packed = (
        words[:, :, 0, :]
        | (words[:, :, 1, :] << 8)
        | (words[:, :, 2, :] << 16)
        | (words[:, :, 3, :] << 24)
    )
    return packed.to(torch.uint32).contiguous(), scales.contiguous()


def dequant_nvfp4(packed, scales, glob) -> torch.Tensor:
    E, K8, N = packed.shape
    words = packed.to(torch.int64)
    vals = torch.empty(E, K8 * 8, N, dtype=torch.float32, device=packed.device)
    for i in range(8):
        vals[:, i::8, :] = _e2m1_decode((words >> (4 * i)) & 0xF)
    s = _e4m3_decode(scales).repeat_interleave(16, dim=1)
    return vals * s * glob.view(E, 1, 1)


def dequant_mxfp8(packed, scales) -> torch.Tensor:
    E, K4, N = packed.shape
    words = packed.to(torch.int64)
    vals = torch.empty(E, K4 * 4, N, dtype=torch.float32, device=packed.device)
    for i in range(4):
        vals[:, i::4, :] = _e4m3_decode((words >> (8 * i)) & 0xFF)
    s = torch.pow(2.0, scales.to(torch.float32) - 127.0).repeat_interleave(32, dim=1)
    return vals * s


def _run_moe(x, packed, scales, topk_ids, top_k, block_size_m, globals=None):
    M, K = x.shape
    N = packed.shape[2]
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, packed.shape[0])
    out = torch.zeros(M * top_k, N, dtype=torch.float16, device=device)
    if globals is None:
        ops.moe_mxfp8_gemm_rdna2(
            x,
            out,
            packed,
            scales,
            torch.empty(0, device=device),
            si,
            ei,
            ntp,
            top_k,
            block_size_m,
            False,
            0,
        )
    else:
        ops.moe_nvfp4_gemm_rdna2(
            x,
            out,
            packed,
            scales,
            globals,
            torch.empty(0, device=device),
            si,
            ei,
            ntp,
            top_k,
            block_size_m,
            False,
            0,
        )
    torch.accelerator.synchronize()
    return out


def _reference_moe(x, w_dq, topk_ids, top_k):
    M = x.shape[0]
    N = w_dq.shape[2]
    ref = torch.zeros(M * top_k, N, dtype=torch.float16, device=device)
    xf = x.to(torch.float32)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            ref[m * top_k + k] = (xf[m] @ w_dq[e]).to(torch.float16)
    return ref


CONFIGS = [(4, 256, 128, 2), (8, 512, 256, 2)]
TOKENS = [1, 4, 8]


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k", CONFIGS)
@pytest.mark.parametrize("M", TOKENS)
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_moe_nvfp4_matches_reference(E, K, N_inter, top_k, M, block_size_m):
    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randn(E, K, 2 * N_inter, dtype=torch.float32, device=device) * 0.02
    packed, scales, glob = quantise_nvfp4(w)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)

    out = _run_moe(x, packed, scales, topk_ids, top_k, block_size_m, globals=glob)
    ref = _reference_moe(x, dequant_nvfp4(packed, scales, glob), topk_ids, top_k)

    diff = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out, ref, atol=0.5, rtol=0.05), f"max diff {diff}"


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k", CONFIGS)
@pytest.mark.parametrize("M", TOKENS)
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_moe_mxfp8_matches_reference(E, K, N_inter, top_k, M, block_size_m):
    torch.manual_seed(9)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randn(E, K, 2 * N_inter, dtype=torch.float32, device=device) * 0.02
    packed, scales = quantise_mxfp8(w)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)

    out = _run_moe(x, packed, scales, topk_ids, top_k, block_size_m)
    ref = _reference_moe(x, dequant_mxfp8(packed, scales), topk_ids, top_k)

    diff = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out, ref, atol=0.5, rtol=0.05), f"max diff {diff}"
