#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_RDNA2_MOE_DETERMINISTIC: the RDNA2 routed MoE is run-to-run exact.

The default epilogues add K-slice and top-k partials with packed-fp16 CAS or
fp32 atomics, so the rounding follows the arrival order of the blocks. The
deterministic mode sums in a fixed order:

* W4A8 with one K block per GEMM: w2 stores one row per (token, expert) and
  ``moe_sum`` adds the local rows in k order. The test rebuilds that sum from
  the kernel's own rows and requires a bit-exact match.
* everything else: W4A16 into the Q32 fixed-point accumulators.

Run ``pytest tests/kernels/quantization/test_rdna2_moe_deterministic.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE kernels are ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    compressed_tensors_moe_wna16_rdna2 as wiring,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"


def _has_op(name: str) -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, name)


gfx1030_moe = pytest.mark.skipif(
    not (on_gfx10x() and _has_op("moe_gptq_gemm_rdna2")),
    reason="requires gfx1030 with moe_gptq_gemm_rdna2 built in",
)
gfx1030_w4a8_moe = pytest.mark.skipif(
    not (on_gfx10x() and _has_op("moe_w4a8_gemm_rdna2")),
    reason="requires gfx1030 with moe_w4a8_gemm_rdna2 built in",
)

REPEATS = 8


def _packed_weights(E, K, N):
    q = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q[:, i::8, :] & 0xF) << (i * 4)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, 4)
        packed[e] = we
    return packed


def _scales(E, groups, N):
    return (0.05 * torch.rand((E, groups, N), device=device) + 0.01).to(torch.float16)


def _qzeros(E, groups, N):
    zeros = torch.full(
        (groups, N), scalar_types.uint4b8.bias - 1, dtype=torch.int32, device=device
    )
    packed = pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)
    return packed.unsqueeze(0).expand(E, -1, -1).contiguous()


def _problem(M, K, N_inter, top_k, local_experts, global_experts, group_size, seed):
    """Inputs plus a layer namespace; EP when local < global experts."""
    torch.manual_seed(seed)
    layer = SimpleNamespace(
        w13_weight_packed=_packed_weights(local_experts, K, 2 * N_inter),
        w2_weight_packed=_packed_weights(local_experts, N_inter, K),
        w13_weight_scale=_scales(local_experts, K // group_size, 2 * N_inter),
        w2_weight_scale=_scales(local_experts, N_inter // group_size, K),
        w13_qzeros=_qzeros(local_experts, K // group_size, 2 * N_inter),
        w2_qzeros=_qzeros(local_experts, N_inter // group_size, K),
        rdna2_w1_buf=torch.zeros(
            M * top_k, 2 * N_inter, dtype=torch.float16, device=device
        ),
        rdna2_act_buf=torch.empty(
            M * top_k, N_inter, dtype=torch.float16, device=device
        ),
        rdna2_empty_tw=torch.empty(0, device=device),
    )
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    # Distinct experts per token, as the router produces.
    topk_ids = (
        torch.rand(M, global_experts, device=device)
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
        .contiguous()
    )
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1)
    expert_map = None
    if local_experts < global_experts:
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
        local = torch.randperm(global_experts, device=device)[:local_experts]
        expert_map[local] = torch.arange(
            local_experts, dtype=torch.int32, device=device
        )
    return layer, x, topk_ids, topk_w, expert_map


def _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, w4a8):
    return wiring._rdna2_fused_moe(
        x,
        topk_w,
        topk_ids,
        layer=layer,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=global_experts,
        expert_map=expert_map,
        w4a8=w4a8,
    )


def _rel_l2(got, ref):
    ref_f = ref.float()
    return ((got.float() - ref_f).norm() / ref_f.norm()).item()


def _assert_repeats_identical(run):
    first = run()
    assert torch.isfinite(first).all()
    for i in range(1, REPEATS):
        again = run()
        assert torch.equal(again, first), (
            f"repeat {i} differs: max |d| "
            f"{(again.float() - first.float()).abs().max().item():.3e}"
        )
    return first


def _set_w4a8_kblock(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_RDNA_W4A8_KBLOCK", raising=False)
    else:
        monkeypatch.setenv("VLLM_RDNA_W4A8_KBLOCK", value)


def test_w4a8_single_k_block_follows_kernel_resolution(monkeypatch):
    _set_w4a8_kblock(monkeypatch, None)
    assert not wiring._w4a8_single_k_block(8, 2560, 640)
    _set_w4a8_kblock(monkeypatch, "256")
    assert not wiring._w4a8_single_k_block(8, 2560, 640)
    _set_w4a8_kblock(monkeypatch, "-1")
    assert wiring._w4a8_single_k_block(8, 2560, 640)
    assert wiring._w4a8_single_k_block(4, 2560, 640)
    # 16 rows cap the A window at 2048 K positions: 2560 splits in two.
    assert not wiring._w4a8_single_k_block(16, 2560, 640)
    assert wiring._w4a8_single_k_block(16, 2048, 640)
    assert not wiring._w4a8_single_k_block(1, 2560, 640)


@gfx1030_w4a8_moe
@pytest.mark.parametrize("global_experts", [16, 64])
def test_deterministic_w4a8_is_exact_ordered_sum(monkeypatch, global_experts):
    """Prefill W4A8: repeats are bit-identical and equal a k-ordered fp32 sum
    of the kernel's own per-(token, expert) rows."""
    M, K, N_inter, top_k, local, group = 1024, 2560, 640, 10, 16, 128
    layer, x, topk_ids, topk_w, expert_map = _problem(
        M, K, N_inter, top_k, local, global_experts, group, seed=7
    )
    monkeypatch.setattr(wiring, "_W4A8_BLOCK_M", 8)
    _set_w4a8_kblock(monkeypatch, "-1")
    monkeypatch.setattr(wiring, "_DETERMINISTIC", True)
    det = _assert_repeats_identical(
        lambda: _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, True)
    )

    # Reference: the same w1/w2 kernels with output_topk=0, then an explicit
    # fp32 sum over the local experts of each token in k order.
    si, ei, ntp = moe_align_block_size(
        topk_ids,
        8,
        global_experts,
        expert_map,
        ignore_invalid_experts=expert_map is not None,
    )
    w1_out = torch.zeros(M * top_k, 2 * N_inter, dtype=torch.float16, device=device)
    empty_tw = torch.empty(0, device=device)
    ops.moe_w4a8_gemm_rdna2(
        x,
        w1_out,
        layer.w13_weight_packed,
        layer.w13_weight_scale,
        layer.w13_qzeros,
        empty_tw,
        si,
        ei,
        ntp,
        top_k,
        8,
        False,
        0,
        False,
    )
    act = torch.empty(M * top_k, N_inter, dtype=torch.float16, device=device)
    apply_moe_activation(MoEActivation.SILU, act, w1_out)
    rows = torch.zeros(M * top_k, K, dtype=torch.float16, device=device)
    ops.moe_w4a8_gemm_rdna2(
        act,
        rows,
        layer.w2_weight_packed,
        layer.w2_weight_scale,
        layer.w2_qzeros,
        topk_w.view(-1).float(),
        si,
        ei,
        ntp,
        1,
        8,
        True,
        0,
        False,
    )
    rows = rows.view(M, top_k, K).float()
    local_slot = (
        expert_map[topk_ids.long()] >= 0
        if expert_map is not None
        else torch.ones_like(topk_ids, dtype=torch.bool)
    )
    acc = torch.zeros(M, K, dtype=torch.float32, device=device)
    for k in range(top_k):
        acc = torch.where(local_slot[:, k : k + 1], acc + rows[:, k], acc)
    assert torch.equal(det, acc.to(torch.float16))

    # Same math as the atomic epilogue up to the rounding order.
    monkeypatch.setattr(wiring, "_DETERMINISTIC", False)
    cas = _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, True)
    assert _rel_l2(det, cas) < 2e-3


@gfx1030_moe
@pytest.mark.parametrize("M", [1, 3, 24, 100, 600])
@pytest.mark.parametrize("global_experts", [16, 64])
def test_deterministic_w4a16_fixed_point_repeats(monkeypatch, M, global_experts):
    """W4A16 steps (decode, short prefill) go through the Q32 fixed point."""
    K, N_inter, top_k, local, group = 2560, 640, 10, 16, 128
    layer, x, topk_ids, topk_w, expert_map = _problem(
        M, K, N_inter, top_k, local, global_experts, group, seed=11
    )
    monkeypatch.setattr(wiring, "_DETERMINISTIC", True)
    det = _assert_repeats_identical(
        lambda: _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, False)
    )
    monkeypatch.setattr(wiring, "_DETERMINISTIC", False)
    cas = _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, False)
    assert _rel_l2(det, cas) < 2e-3


@gfx1030_w4a8_moe
def test_deterministic_w4a8_without_single_k_block_uses_fixed_point(monkeypatch):
    """A split-K W4A8 configuration falls back to the exact W4A16 path."""
    M, K, N_inter, top_k, local, global_experts, group = 600, 2560, 640, 10, 16, 64, 128
    layer, x, topk_ids, topk_w, expert_map = _problem(
        M, K, N_inter, top_k, local, global_experts, group, seed=5
    )
    monkeypatch.setattr(wiring, "_DETERMINISTIC", True)
    monkeypatch.setattr(wiring, "_W4A8_BLOCK_M", 8)
    _set_w4a8_kblock(monkeypatch, None)
    split_k = _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, True)
    w4a16 = _forward(layer, x, topk_ids, topk_w, expert_map, global_experts, False)
    assert torch.equal(split_k, w4a16)
