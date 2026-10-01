#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 fused MoE W4A8 (sdot4) HIP kernel.

Mirrors ``test_rdna2_moe_w4a16.py`` (real expert shapes, packed RDNA2 weights,
fused routing) and the dense W4A8 suite:

* the W4A8 op against a per-(token, group) fp32 quantized reference,
* the W4A8 op against ``moe_gptq_gemm_rdna2`` on the same packed buffer,
* ``output_topk`` fused reduce vs ``moe_sum``,
* invalid-shape fallback byte-identical to ``moe_gptq_gemm_rdna2``,
* the wiring latch: default-off, env-on, and the fused forward with the
  flag True/False.

Run ``pytest tests/kernels/quantization/test_rdna2_moe_w4a8.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE W4A8 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"

GPTQV1_ZERO = 8  # uint4b8 bias: effective zero for stored nibble 7 (+1 quirk)


def _has_w4a8_moe_ops() -> bool:
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    try:
        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("moe_w4a8_gemm_rdna2" in str(s) for s in schemas)


gfx1030_w4a8_moe = pytest.mark.skipif(
    not (on_gfx10x() and _has_w4a8_moe_ops()),
    reason="requires gfx1030 with moe_w4a8_gemm_rdna2 built in",
)


def _make_weights(E, K, N):
    """Random 4-bit weights [E, K, N] plus the packed+shuffled RDNA2 form."""
    q = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q[:, i::8, :] & 0xF) << (i * 4)
    g_idx = torch.empty(0, dtype=torch.int32, device=device)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, g_idx, 4)
        packed[e] = we
    return q, packed


def _make_scales(E, groups, N):
    return (0.05 * torch.rand((E, groups, N), device=device) + 0.01).to(
        torch.float16
    )


def _make_qzeros(E, groups, N):
    zeros = torch.full(
        (groups, N),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    packed = pack_quantized_values_into_int32(
        zeros, scalar_types.uint4b8, packed_dim=1
    )
    return packed.unsqueeze(0).expand(E, -1, -1).contiguous()


def _quant_act(x, group_size):
    """Per-(token, group) absmax/127 int8 quant, matching the kernel."""
    m, k = x.shape
    g = k // group_size
    xf = x.float().reshape(m, g, group_size)
    gmax = xf.abs().amax(dim=2, keepdim=True)
    inv = torch.where(gmax > 0, 127.0 / gmax, torch.zeros_like(gmax))
    q = torch.round(xf * inv).clamp_(-128, 127)
    return q, (gmax.squeeze(-1) / 127.0)


def _fp32_reference(x, q_int4, scales, topk_ids, group_size):
    """Independent per-(token, group) int8-A / fp32-W MoE w1 reference."""
    e_count, k, n = q_int4.shape
    m, topk = topk_ids.shape
    groups = k // group_size
    aq, ascale = _quant_act(x, group_size)
    w = (q_int4.float() - GPTQV1_ZERO).reshape(e_count, groups, group_size, n)
    w = (w * scales.float().unsqueeze(2)).reshape(e_count, k, n)
    a_scaled = (aq.reshape(m, groups, group_size) * ascale.unsqueeze(2)).reshape(
        m, k
    )
    out = torch.zeros(m * topk, n, dtype=torch.float32, device=device)
    for row in range(m):
        for slot in range(topk):
            expert = int(topk_ids[row, slot])
            out[row * topk + slot] = a_scaled[row] @ w[expert]
    return out.to(torch.float16)


def _run_w4a8(a, c, w, scales, zeros, topk_weights, si, ei, ntp, top_k,
              block_size_m, mul_topk_weight, output_topk):
    ops.moe_w4a8_gemm_rdna2(
        a, c, w, scales, zeros, topk_weights, si, ei, ntp, top_k, block_size_m,
        mul_topk_weight, output_topk, False,
    )


def _run_w4a16(a, c, w, scales, zeros, topk_weights, si, ei, ntp, top_k,
               block_size_m, mul_topk_weight, output_topk):
    ops.moe_gptq_gemm_rdna2(
        a, c, w, scales, zeros, topk_weights, si, ei, ntp, top_k, block_size_m,
        mul_topk_weight, output_topk,
    )


def _rel_l2(got, ref):
    got_f = got.float()
    ref_f = ref.float()
    denom = ref_f.norm()
    return ((got_f - ref_f).norm() / denom).item() if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# W4A8 vs W4A16 on real expert shapes
# ---------------------------------------------------------------------------

# E, K, N_inter, top_k, group_size, M, block_size_m
REAL_CASES = [
    (4, 2048, 512, 8, 32, 1, 1),
    (4, 2048, 512, 8, 32, 4, 4),
    (4, 2048, 512, 8, 64, 16, 4),
    (16, 2048, 768, 8, 32, 64, 4),
    (4, 2560, 640, 10, 128, 64, 8),
    (4, 2560, 640, 10, 64, 256, 8),
    (4, 2560, 640, 10, 32, 512, 8),
    (4, 2560, 512, 10, 128, 2048, 8),
]


@gfx1030_w4a8_moe
@pytest.mark.parametrize(
    "E, K, N_inter, top_k, group_size, M, block_size_m", REAL_CASES
)
def test_w4a8_moe_w1_matches_w4a16(
    E, K, N_inter, top_k, group_size, M, block_size_m
):
    """The sdot4 MoE w1 GEMM tracks the qualified W4A16 MoE kernel."""
    n_gate_up = N_inter * 2
    groups = K // group_size
    torch.manual_seed(42)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    _, w = _make_weights(E, K, n_gate_up)
    scales = _make_scales(E, groups, n_gate_up)
    zeros = _make_qzeros(E, groups, n_gate_up)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)
    empty = torch.empty(0, device=device)

    w4a8_out = torch.zeros(M * top_k, n_gate_up, dtype=torch.float16, device=device)
    _run_w4a8(
        x, w4a8_out, w, scales, zeros, empty, si, ei, ntp, top_k, block_size_m,
        False, 0,
    )
    w4a16_out = torch.zeros(M * top_k, n_gate_up, dtype=torch.float16, device=device)
    _run_w4a16(
        x, w4a16_out, w, scales, zeros, empty, si, ei, ntp, top_k, block_size_m,
        False, 0,
    )

    assert torch.isfinite(w4a8_out).all()
    rel = _rel_l2(w4a8_out, w4a16_out)
    assert rel < 0.1, f"W4A8 vs W4A16 rel-L2 {rel:.4f} >= 0.1"


# ---------------------------------------------------------------------------
# fp32 quantized reference
# ---------------------------------------------------------------------------

FP32_CASES = [
    (4, 512, 256, 4, 32, 16, 4),
    (4, 2048, 512, 8, 32, 64, 4),
    (4, 2048, 512, 8, 64, 32, 4),
    (4, 2560, 640, 10, 128, 64, 8),
]


@gfx1030_w4a8_moe
@pytest.mark.parametrize(
    "E, K, N_inter, top_k, group_size, M, block_size_m", FP32_CASES
)
def test_w4a8_moe_w1_matches_fp32_reference(
    E, K, N_inter, top_k, group_size, M, block_size_m
):
    """The kernel matches an independent per-(token, group) fp32 reference."""
    n_gate_up = N_inter * 2
    groups = K // group_size
    torch.manual_seed(7)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    q_int4, w = _make_weights(E, K, n_gate_up)
    scales = _make_scales(E, groups, n_gate_up)
    zeros = _make_qzeros(E, groups, n_gate_up)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    got = torch.zeros(M * top_k, n_gate_up, dtype=torch.float16, device=device)
    _run_w4a8(
        x, got, w, scales, zeros, torch.empty(0, device=device), si, ei, ntp,
        top_k, block_size_m, False, 0,
    )
    ref = _fp32_reference(x, q_int4, scales, topk_ids, group_size)

    rel = _rel_l2(got, ref)
    assert rel < 1e-2, f"rel-L2 {rel:.5f} >= 1e-2"


# ---------------------------------------------------------------------------
# output_topk reduce
# ---------------------------------------------------------------------------


@gfx1030_w4a8_moe
@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("top_k", [8, 10])
def test_w4a8_moe_output_topk_reduces(M, top_k):
    """output_topk fuses moe_sum: several experts write to the same row."""
    E, K, N_inter, group_size = 8, 2048, 512, 64
    groups = K // group_size
    torch.manual_seed(123)

    x = torch.randn(M * top_k, K, dtype=torch.float16, device=device)
    _, w = _make_weights(E, K, N_inter)
    scales = _make_scales(E, groups, N_inter)
    zeros = _make_qzeros(E, groups, N_inter)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()
    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    flat = torch.zeros(M * top_k, N_inter, dtype=torch.float16, device=device)
    _run_w4a8(
        x, flat, w, scales, zeros, topk_w.view(-1), si, ei, ntp, 1, 1, True, 0,
    )
    ref = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_sum(flat.view(M, top_k, N_inter), ref)

    fused = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    _run_w4a8(
        x, fused, w, scales, zeros, topk_w.view(-1), si, ei, ntp, 1, 1, True,
        top_k,
    )
    rel = _rel_l2(fused, ref)
    assert rel < 0.05, f"fused reduce rel-L2 {rel:.5f} >= 0.05"


# ---------------------------------------------------------------------------
# invalid-shape fallback
# ---------------------------------------------------------------------------


@gfx1030_w4a8_moe
def test_w4a8_moe_invalid_shape_falls_back_byte_identical():
    """group_size 256 is W4A8-ineligible; the internal fallback must match.

    K=256 keeps grid.z=1, so each output element is written by exactly one
    block and the packed CAS order is deterministic (byte-identical).
    """
    E, K, N, group_size, M, top_k, block_size_m = 4, 256, 128, 256, 8, 2, 1
    groups = K // group_size
    torch.manual_seed(11)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    _, w = _make_weights(E, K, N)
    scales = _make_scales(E, groups, N)
    zeros = _make_qzeros(E, groups, N)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)
    empty = torch.empty(0, device=device)

    via_w4a8 = torch.zeros(M * top_k, N, dtype=torch.float16, device=device)
    _run_w4a8(
        x, via_w4a8, w, scales, zeros, empty, si, ei, ntp, top_k, block_size_m,
        False, 0,
    )
    direct = torch.zeros(M * top_k, N, dtype=torch.float16, device=device)
    _run_w4a16(
        x, direct, w, scales, zeros, empty, si, ei, ntp, top_k, block_size_m,
        False, 0,
    )
    assert torch.equal(via_w4a8, direct), "fallback must be byte-identical"


# ---------------------------------------------------------------------------
# end-to-end MoE + wiring latch
# ---------------------------------------------------------------------------


@gfx1030_w4a8_moe
def test_w4a8_moe_full_forward_matches_w4a16():
    """w1 + SwiGLU + w2 with output_topk through the wiring helper."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        _rdna2_fused_moe,
    )

    E, K, N_inter, top_k, group_size, M = 4, 512, 256, 4, 64, 32
    groups13 = K // group_size
    groups2 = N_inter // group_size
    torch.manual_seed(3)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    _, w13 = _make_weights(E, K, 2 * N_inter)
    _, w2 = _make_weights(E, N_inter, K)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1)

    # Scales and zeros must be identical across the on/off runs (they are
    # drawn once here), otherwise the A/B compares different weights.
    s13 = _make_scales(E, groups13, 2 * N_inter)
    s2 = _make_scales(E, groups2, K)
    z13 = _make_qzeros(E, groups13, 2 * N_inter)
    z2 = _make_qzeros(E, groups2, K)

    def _layer():
        return SimpleNamespace(
            w13_weight_packed=w13,
            w2_weight_packed=w2,
            w13_weight_scale=s13,
            w2_weight_scale=s2,
            w13_qzeros=z13,
            w2_qzeros=z2,
            rdna2_w1_buf=torch.zeros(
                M * top_k, 2 * N_inter, dtype=torch.float16, device=device
            ),
            rdna2_act_buf=torch.empty(
                M * top_k, N_inter, dtype=torch.float16, device=device
            ),
            rdna2_empty_tw=torch.empty(0, device=device),
        )

    def _run(w4a8):
        return _rdna2_fused_moe(
            x,
            topk_w,
            topk_ids,
            layer=_layer(),
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=E,
            expert_map=None,
            w4a8=w4a8,
        )

    off = _run(False)
    on = _run(True)
    assert torch.isfinite(on).all()
    rel = _rel_l2(on, off)
    assert 1e-6 < rel < 0.1, f"fused W4A8 vs W4A16 rel-L2 {rel:.6f} out of band"


@gfx1030_w4a8_moe
def test_w4a8_moe_latch_default_off(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
        resolve_w4a8_moe,
    )

    monkeypatch.delenv("VLLM_RDNA2_W4A8_SDOT4", raising=False)
    assert resolve_w4a8_moe() is False


@gfx1030_w4a8_moe
def test_w4a8_moe_latch_env_on(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
        resolve_w4a8_moe,
    )

    monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "1")
    assert resolve_w4a8_moe() is True


# ---------------------------------------------------------------------------
# w2 shape (intermediate -> hidden) through the op
# ---------------------------------------------------------------------------


@gfx1030_w4a8_moe
@pytest.mark.parametrize("M", [1, 16, 256])
def test_w4a8_moe_w2_matches_w4a16(M):
    """The down-projection shape runs on the same kernel path."""
    E, N_inter, hidden, top_k, group_size = 4, 640, 2560, 10, 128
    groups = N_inter // group_size
    torch.manual_seed(5)

    act = torch.randn(M * top_k, N_inter, dtype=torch.float16, device=device)
    _, w = _make_weights(E, N_inter, hidden)
    scales = _make_scales(E, groups, hidden)
    zeros = _make_qzeros(E, groups, hidden)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1)
    si, ei, ntp = moe_align_block_size(topk_ids, 4, E)

    w4a8_out = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    _run_w4a8(
        act, w4a8_out, w, scales, zeros, topk_w.view(-1), si, ei, ntp, 1, 4,
        True, top_k,
    )
    w4a16_out = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    _run_w4a16(
        act, w4a16_out, w, scales, zeros, topk_w.view(-1), si, ei, ntp, 1, 4,
        True, top_k,
    )
    rel = _rel_l2(w4a8_out, w4a16_out)
    assert rel < 0.1, f"w2 rel-L2 {rel:.4f} >= 0.1"
