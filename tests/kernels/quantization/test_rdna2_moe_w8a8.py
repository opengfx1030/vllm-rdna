# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 fused MoE W8A8 (sdot4) HIP kernel.

Mirrors ``test_rdna2_moe_w4a8.py`` on int8 experts in the
``rdna2_w8a8.pack_moe_weight`` layout:

* the op against a per-(token, group) int8-A / fp32-W reference, in both
  accumulation modes,
* ``output_topk`` fused reduce vs ``moe_sum``,
* the down-projection shape,
* invalid shapes raise instead of falling back.

Run ``pytest tests/kernels/quantization/test_rdna2_moe_w8a8.py``.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization import rdna2_w8a8
from vllm.model_executor.layers.quantization.rdna2_w8a8 import pack_moe_weight


def _has_ops() -> bool:
    return rdna2_w8a8.on_gfx1030() and rdna2_w8a8.moe_ops_built()


gfx1030_w8a8_moe = pytest.mark.skipif(
    not _has_ops(), reason="requires gfx1030 with moe_w8a8_gemm_rdna2 built in"
)
device = "cuda"


def _make_experts(E, N, K, w_group):
    """int8 experts [E, N, K], fp16 scales [E, K/w_group, N] and the packed form."""
    w = torch.randint(-128, 128, (E, N, K), dtype=torch.int8, device=device)
    scales = (0.02 * torch.rand(E, K // w_group, N, device=device) + 0.005).half()
    return w, scales, pack_moe_weight(w)


def _quant_act(x, group):
    m, k = x.shape
    g = k // group
    xf = x.float().reshape(m, g, group)
    gmax = xf.abs().amax(dim=2, keepdim=True)
    inv = torch.where(gmax > 0, 127.0 / gmax, torch.zeros_like(gmax))
    q = torch.round(xf * inv).clamp_(-128, 127)
    return (q * (gmax / 127.0)).reshape(m, k)


def _dequant(w, scales):
    """int8 [E, N, K] x fp16 [E, K/WG, N] -> fp32 [E, N, K]."""
    e_count, n, k = w.shape
    wg = k // scales.shape[1]
    s = scales.float().transpose(1, 2).repeat_interleave(wg, dim=2)
    return w.float() * s


def _reference(x, w, scales, topk_ids, a_group):
    """Per-(token, group) int8-A / fp32-W MoE GEMM, one row per (token, slot)."""
    m, topk = topk_ids.shape
    a = _quant_act(x, a_group)
    w_deq = _dequant(w, scales)
    out = torch.zeros(m * topk, w.shape[1], dtype=torch.float32, device=device)
    for row in range(m):
        for slot in range(topk):
            out[row * topk + slot] = a[row] @ w_deq[int(topk_ids[row, slot])].t()
    return out.to(torch.float16)


def _run(
    a,
    c,
    packed,
    scales,
    topk_w,
    si,
    ei,
    ntp,
    top_k,
    bsm,
    mul_tw,
    out_topk,
    a_group,
    fp32_accum=True,
):
    from vllm import _custom_ops as ops

    ops.moe_w8a8_gemm_rdna2(
        a,
        c,
        packed,
        scales,
        topk_w,
        si,
        ei,
        ntp,
        top_k,
        bsm,
        mul_tw,
        out_topk,
        a_group,
        fp32_accum,
    )


def _rel_l2(got, ref):
    got_f = got.float()
    ref_f = ref.float()
    denom = ref_f.norm()
    return ((got_f - ref_f).norm() / denom).item() if denom > 0 else 0.0


# E, K, N_inter, top_k, a_group, w_group, M, block_size_m
CASES = [
    (4, 2048, 512, 8, 128, 2048, 1, 1),
    (4, 2048, 512, 8, 128, 2048, 4, 4),
    (4, 2048, 512, 8, 64, 2048, 16, 4),
    (16, 2048, 768, 8, 32, 32, 64, 4),
    (4, 2560, 640, 10, 128, 2560, 64, 8),
    (4, 2560, 640, 10, 128, 128, 256, 8),
    (4, 2560, 512, 10, 128, 2560, 2048, 8),
]


@gfx1030_w8a8_moe
@pytest.mark.parametrize("E, K, N_inter, top_k, a_group, w_group, M, bsm", CASES)
@pytest.mark.parametrize("fp32_accum", [True, False])
def test_w8a8_moe_w1_matches_reference(
    E, K, N_inter, top_k, a_group, w_group, M, bsm, fp32_accum
):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    n_gate_up = 2 * N_inter
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w, scales, packed = _make_experts(E, n_gate_up, K, w_group)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, bsm, E)
    empty = torch.empty(0, device=device)

    out = torch.zeros(M * top_k, n_gate_up, dtype=torch.float16, device=device)
    _run(
        x,
        out,
        packed,
        scales,
        empty,
        si,
        ei,
        ntp,
        top_k,
        bsm,
        False,
        0,
        a_group,
        fp32_accum,
    )
    assert torch.isfinite(out).all()
    rel = _rel_l2(out, _reference(x, w, scales, topk_ids, a_group))
    # fp32 atomics: accumulation order only; fp16 CAS: fp16 partial rounding.
    bound = 2e-3 if fp32_accum else 1e-2
    assert rel < bound, f"rel-L2 {rel:.5f} >= {bound}"


@gfx1030_w8a8_moe
@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("top_k", [8, 10])
def test_w8a8_moe_output_topk_reduces(M, top_k):
    """output_topk fuses moe_sum: several experts write to the same row."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    E, K, N_inter, a_group = 8, 2048, 512, 128
    torch.manual_seed(123)
    x = torch.randn(M * top_k, K, dtype=torch.float16, device=device)
    _, scales, packed = _make_experts(E, N_inter, K, K)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()
    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    flat = torch.zeros(M * top_k, N_inter, dtype=torch.float16, device=device)
    _run(x, flat, packed, scales, topk_w.view(-1), si, ei, ntp, 1, 1, True, 0, a_group)
    ref = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_sum(flat.view(M, top_k, N_inter), ref)

    fused = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    _run(
        x,
        fused,
        packed,
        scales,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
        a_group,
    )
    rel = _rel_l2(fused, ref)
    assert rel < 0.05, f"fused reduce rel-L2 {rel:.5f} >= 0.05"


@gfx1030_w8a8_moe
@pytest.mark.parametrize("M", [1, 16, 256])
def test_w8a8_moe_w2_matches_reference(M):
    """The down-projection shape (intermediate -> hidden, fused reduce)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    E, N_inter, hidden, top_k, a_group = 4, 640, 2560, 10, 128
    torch.manual_seed(5)
    act = torch.randn(M * top_k, N_inter, dtype=torch.float16, device=device)
    w, scales, packed = _make_experts(E, hidden, N_inter, N_inter)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()
    si, ei, ntp = moe_align_block_size(topk_ids, 4, E)

    out = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    _run(
        act,
        out,
        packed,
        scales,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        4,
        True,
        top_k,
        a_group,
    )

    flat = _reference(act, w, scales, topk_ids.reshape(-1, 1), a_group)
    flat = (flat.float() * topk_w.view(-1, 1)).half()
    ref = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    ops.moe_sum(flat.view(M, top_k, hidden), ref)
    rel = _rel_l2(out, ref)
    assert rel < 0.05, f"w2 rel-L2 {rel:.4f} >= 0.05"


@gfx1030_w8a8_moe
def test_w8a8_moe_rejects_bad_shapes():
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    E, K, N, M, top_k = 4, 256, 128, 8, 2
    torch.manual_seed(11)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    _, scales, packed = _make_experts(E, N, K, 64)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)
    empty = torch.empty(0, device=device)
    out = torch.zeros(M * top_k, N, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError):  # a_group 128 does not divide w_group 64
        _run(x, out, packed, scales, empty, si, ei, ntp, top_k, 1, False, 0, 128)
    with pytest.raises(RuntimeError):  # block_size_m 16 is not instantiated
        _run(x, out, packed, scales, empty, si, ei, ntp, top_k, 16, False, 0, 64)
