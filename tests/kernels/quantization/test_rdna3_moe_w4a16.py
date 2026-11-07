#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA3 fused MoE W4A16 HIP kernel (gfx1100).

Tests ``moe_gptq_gemm_rdna3`` against the dense ``gptq_gemm_rdna3`` as
reference: builds RDNA3-format weights (shuffled int32, synthesized qzeros),
runs the fused MoE kernel, and compares per-expert results.

Model parameters taken from:
  - cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
    (hidden=2048, inter=768, E=128, top_k=8, G=32)
  - Qwen3.6-35B-A3B-GPTQ-W4A16-G32
    (hidden=2048, inter=512, E=256, top_k=8, G=32)

Run `pytest tests/kernels/quantization/test_rdna3_moe_w4a16.py`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 MoE W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (  # noqa: E402
    Rdna3WNA16Experts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (  # noqa: E402
    _process_weights_rdna3,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"

gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")
    ),
    reason="Requires gfx1100 with moe_gptq_gemm_rdna3 op",
)

# Model configurations: real K/N/top_k/group_size dims, E capped at 16 to
# fit in test GPU memory (full E=128/256 would need >20GB for weights alone).
# Kernel behavior is E-independent (per-expert tiling), so E=16 is sufficient.
MODEL_CONFIGS = [
    # cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit dims (E capped)
    pytest.param(16, 2048, 768, 8, 32, id="Qwen3-30B-A3B"),
    # Qwen3.6-35B-A3B-GPTQ-W4A16-G32 dims (E capped)
    pytest.param(16, 2048, 512, 8, 32, id="Qwen3.6-35B-A3B"),
]

# Token counts: decode (1), small batch (4), medium (16), prefill (64)
NUM_TOKENS = [1, 4, 16, 64, 256, 512]


def _make_packed_weights(E, K, N):
    """Create random 4-bit packed weights [E, K/8, N] int32 + shuffle."""
    w = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (w[:, i::8, :] & 0xF) << (i * 4)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, 4)
        packed[e] = we
    return packed


def _make_scales(E, groups, N, dtype):
    return torch.rand(E, groups, N, dtype=dtype, device=device) * 0.1


def _make_qzeros(E, groups, N):
    zeros = torch.full(
        (groups, N),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    qz = pack_quantized_values_into_int32(
        zeros,
        scalar_types.uint4b8,
        packed_dim=1,
    )
    return qz.unsqueeze(0).expand(E, -1, -1).contiguous()


def _make_ct_weights(E, N, K, group_size, dtype):
    """Compressed-tensors canonical N-first layout: packed ``[E, N, K // 8]``
    and scales ``[E, N, K // group_size]``, plus the dequantized weights."""
    q = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=device)
    s = torch.rand(E, N, K // group_size, dtype=dtype, device=device) * 0.1 + 0.01
    packed = pack_quantized_values_into_int32(q, scalar_types.uint4b8, packed_dim=2)
    w = (q - 8).float() * s.float().repeat_interleave(group_size, dim=2)
    return packed.contiguous(), s, w


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_moe_w1_matches_dense(
    E, K, N_inter, top_k, group_size, M, dtype, block_size_m
):
    """w1 GEMM via fused kernel matches per-expert dense kernel."""
    N_gate_up = N_inter * 2
    groups = K // group_size

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=dtype, device=device)
    w13 = _make_packed_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, groups, N_gate_up, dtype)
    w13_z = _make_qzeros(E, groups, N_gate_up)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    # Fused kernel
    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
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
    ref_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    for m in range(M):
        for k in range(top_k):
            e = topk_ids[m, k].item()
            flat = m * top_k + k
            ref = ops.gptq_gemm_rdna3(
                x[m : m + 1],
                w13[e],
                w13_z[e],
                w13_s[e],
                False,
            )
            ref_out[flat] = ref.squeeze()

    # Split-K atomics can cause minor fp16/bf16 rounding differences
    # at large K (e.g. K=2048 → 8 K-blocks). Use allclose, not equal.
    atol = 0.5 if dtype == torch.bfloat16 else 0.1
    assert torch.allclose(fused_out, ref_out, atol=atol, rtol=0.01), (
        f"max diff: {(fused_out - ref_out).abs().max().item()}"
    )


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_output_topk_reduces(E, K, N_inter, top_k, group_size, M, dtype):
    """output_topk fuses moe_sum: multiple experts write to same output row."""
    groups = K // group_size

    torch.manual_seed(123)
    x = torch.randn(M * top_k, K, dtype=dtype, device=device)
    w = _make_packed_weights(E, K, N_inter)
    ws = _make_scales(E, groups, N_inter, dtype)
    wz = _make_qzeros(E, groups, N_inter)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Without output_topk: write to [M*top_k, N] then moe_sum
    flat_out = torch.zeros(M * top_k, N_inter, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        flat_out,
        w,
        ws,
        wz,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        0,
    )
    ref = torch.zeros(M, N_inter, dtype=dtype, device=device)
    ops.moe_sum(flat_out.view(M, top_k, N_inter), ref)

    # With output_topk: write directly to [M, N]
    fused = torch.zeros(M, N_inter, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        fused,
        w,
        ws,
        wz,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    atol = 1.0 if dtype == torch.bfloat16 else 0.1
    assert torch.allclose(fused, ref, atol=atol, rtol=0.01), (
        f"max diff: {(fused - ref).abs().max().item()}"
    )


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_moe_e2e(E, K, N_inter, top_k, group_size, M, dtype):
    """Full MoE forward: w1 + silu_and_mul + w2 with output_topk reduce."""
    N_gate_up = N_inter * 2
    hidden = K

    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=dtype, device=device)
    w13 = _make_packed_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, K // group_size, N_gate_up, dtype)
    w13_z = _make_qzeros(E, K // group_size, N_gate_up)
    w2 = _make_packed_weights(E, N_inter, hidden)
    w2_s = _make_scales(E, N_inter // group_size, hidden, dtype)
    w2_z = _make_qzeros(E, N_inter // group_size, hidden)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Fused path (what apply() does)
    w1_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        w1_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        1,
        False,
        0,
    )
    act_out = torch.empty(M * top_k, N_inter, dtype=dtype, device=device)
    apply_moe_activation(MoEActivation.SILU, act_out, w1_out)
    fused = torch.zeros(M, hidden, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        act_out,
        fused,
        w2,
        w2_s,
        w2_z,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    # Per-expert reference
    ref = torch.zeros(M, hidden, dtype=dtype, device=device)
    for m_idx in range(M):
        for k_idx in range(top_k):
            e = topk_ids[m_idx, k_idx].item()
            w = topk_w[m_idx, k_idx].item()
            r1 = ops.gptq_gemm_rdna3(
                x[m_idx : m_idx + 1],
                w13[e],
                w13_z[e],
                w13_s[e],
                False,
            )
            a = torch.empty(1, N_inter, dtype=dtype, device=device)
            apply_moe_activation(MoEActivation.SILU, a, r1)
            r2 = ops.gptq_gemm_rdna3(
                a,
                w2[e],
                w2_z[e],
                w2_s[e],
                False,
            )
            ref[m_idx] += r2.squeeze() * w

    # E2E chains w1 + activation + w2 + topk_w + output_topk reduce.
    # Each step accumulates rounding error (split-K atomics, topk_w
    # multiply order). Use relative L2 norm like the dense kernel test.
    diff_l2 = torch.norm(fused.float() - ref.float())
    ref_l2 = torch.norm(ref.float())
    rel_l2 = (diff_l2 / ref_l2).item() if ref_l2 > 0 else 0.0
    threshold = 0.05 if dtype == torch.float16 else 0.10
    assert rel_l2 < threshold, (
        f"rel L2 = {rel_l2:.4f} (threshold {threshold}), "
        f"max abs diff: {(fused - ref).abs().max().item()}"
    )


@gfx1100_only
def test_expert_id_minus_one():
    """Kernel handles expert_id == -1 (expert parallelism) without crash."""
    # Qwen3-30B-A3B dims (E capped for memory)
    E, K, N = 16, 2048, 768
    groups = K // 32

    w = _make_packed_weights(E, K, N)
    ws = _make_scales(E, groups, N, torch.bfloat16)
    wz = _make_qzeros(E, groups, N)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)

    # Manually create sorted_token_ids/expert_ids with -1
    sorted_ids = torch.tensor([0], dtype=torch.int32, device=device)
    expert_ids = torch.tensor([-1], dtype=torch.int32, device=device)
    ntp = torch.tensor([1], dtype=torch.int32, device=device)

    out = torch.zeros(1, N, dtype=torch.bfloat16, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        out,
        w,
        ws,
        wz,
        torch.empty(0, device=device),
        sorted_ids,
        expert_ids,
        ntp,
        1,
        1,
        False,
        0,
    )
    current_platform.synchronize()

    # Output should remain zero (expert skipped)
    assert torch.equal(out, torch.zeros_like(out))


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_process_weights_from_compressed_tensors_layout(dtype):
    """Weights in the layout compressed-tensors hands to the oracle go through
    the RDNA3 post-processing and a full MoE forward, against a dequantized
    reference. The tests above build the kernel layout directly, so they do
    not notice when the layout upstream of the kernel changes (#52798)."""
    E, K, N_inter, top_k, group_size, M = 8, 512, 256, 2, 32, 16
    torch.manual_seed(5)
    w13, w13_s, w13_ref = _make_ct_weights(E, 2 * N_inter, K, group_size, dtype)
    w2, w2_s, w2_ref = _make_ct_weights(E, K, N_inter, group_size, dtype)
    w13, w2, w13_s, w2_s, w13_z, w2_z, *_ = _process_weights_rdna3(
        w13, w2, w13_s, w2_s, group_size
    )
    assert w13_z is not None and w2_z is not None

    x = torch.randn(M, K, dtype=dtype, device=device)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()
    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    w1_out = torch.zeros(M * top_k, 2 * N_inter, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        w1_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        1,
        False,
        0,
    )
    act_out = torch.empty(M * top_k, N_inter, dtype=dtype, device=device)
    apply_moe_activation(MoEActivation.SILU, act_out, w1_out)
    out = torch.zeros(M, K, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        act_out,
        out,
        w2,
        w2_s,
        w2_z,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    ref = torch.zeros(M, K, device=device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k])
            h = w13_ref[e] @ x[m].float()
            a = F.silu(h[:N_inter]) * h[N_inter:]
            ref[m] += (w2_ref[e] @ a) * topk_w[m, k]
    rel_l2 = (torch.norm(out.float() - ref) / torch.norm(ref)).item()
    assert rel_l2 < (0.02 if dtype == torch.float16 else 0.05), rel_l2


# --- Q32 mode, hub masking and block sizes through Rdna3WNA16Experts.apply ---
#
# The hub contract: topk_ids == -1 marks a masked pair, slot_map is passed as
# expert_map (-1 = expert not resident), the call is graph captured for
# M <= 48 and M goes up to 2048. Everything below runs the shipping experts
# class, so the Python flow is tested together with the kernels.

HUB_K, HUB_INTER, HUB_TOP_K, HUB_GROUP = 2560, 640, 10, 32
# Resident experts per test (E capped like MODEL_CONFIGS to fit test memory).
HUB_SLOTS = 16
HUB_TOKENS = [1, 3, 24, 48, 2048]
MODES = ["default", "q32"]


def _expert_weights(dtype=torch.float16):
    """Kernel-layout weights for HUB_SLOTS experts and their dequantized refs."""
    w13, w13_s, w13_ref = _make_ct_weights(
        HUB_SLOTS, 2 * HUB_INTER, HUB_K, HUB_GROUP, dtype
    )
    w2, w2_s, w2_ref = _make_ct_weights(HUB_SLOTS, HUB_K, HUB_INTER, HUB_GROUP, dtype)
    w13, w2, w13_s, w2_s, w13_z, w2_z, *_ = _process_weights_rdna3(
        w13, w2, w13_s, w2_s, HUB_GROUP
    )
    assert w13_z is not None and w2_z is not None
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        w13_s=w13_s,
        w2_s=w2_s,
        w13_z=w13_z,
        w2_z=w2_z,
        w13_ref=w13_ref,
        w2_ref=w2_ref,
    )


@pytest.fixture(scope="module")
def hub_weights():
    torch.manual_seed(7)
    return _expert_weights()


def _hub_experts(monkeypatch, weights, num_experts, *, q32, block_m=None):
    """The shipping experts class; it reads its env knobs at construction."""
    if q32:
        monkeypatch.setenv("VLLM_RDNA3_MOE_DETERMINISTIC", "1")
    else:
        monkeypatch.delenv("VLLM_RDNA3_MOE_DETERMINISTIC", raising=False)
    if block_m is None:
        monkeypatch.delenv("VLLM_RDNA3_MOE_BLOCK_M", raising=False)
    else:
        monkeypatch.setenv("VLLM_RDNA3_MOE_BLOCK_M", str(block_m))
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=HUB_TOP_K,
        hidden_dim=HUB_K,
        intermediate_size=HUB_INTER,
        num_local_experts=HUB_SLOTS,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.float16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=2048,
    )
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=weights.w13_s,
        w2_scale=weights.w2_s,
        w1_zp=weights.w13_z,
        w2_zp=weights.w2_z,
    )
    return Rdna3WNA16Experts(moe_config, quant_config)


def _slot_map(num_experts, resident_global_ids=None):
    """Hub slot_map: global expert id -> local slot, -1 when not resident."""
    slot_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    if resident_global_ids is None:
        generator = torch.Generator(device=device).manual_seed(3)
        resident_global_ids = torch.randperm(
            num_experts, generator=generator, device=device
        )[:HUB_SLOTS]
    slots = torch.arange(len(resident_global_ids), dtype=torch.int32, device=device)
    slot_map[resident_global_ids] = slots
    return slot_map


def _routing(M, num_experts, slot_map, *, masked=0.0, seed=0):
    """fp16 hidden states, [M, top_k] int32 ids with a fraction ``masked`` set
    to -1, and fp32 router weights. Column 0 always names a resident expert so
    that no token's result is trivially zero."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, HUB_K, generator=generator, device=device).to(torch.float16)
    scores = torch.rand(M, num_experts, generator=generator, device=device)
    ids = scores.argsort(dim=1)[:, :HUB_TOP_K].to(torch.int32)
    ids[torch.rand(ids.shape, generator=generator, device=device) < masked] = -1
    resident = (slot_map >= 0).nonzero().flatten().to(torch.int32)
    ids[:, 0] = resident[torch.arange(M, device=device) % resident.numel()]
    logits = torch.randn(M, HUB_TOP_K, generator=generator, device=device)
    return x, ids, torch.softmax(logits, dim=-1)


def _dense_moe(x, ids, topk_w, slot_map, weights):
    """float32 reference of the hub contract: pairs with id -1 or an expert
    absent from slot_map contribute nothing."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1).long()
    slots = torch.where(flat >= 0, slot_map[flat.clamp(min=0)].long(), -1)
    flat_w = topk_w.reshape(-1)
    for slot in range(weights.w13_ref.size(0)):
        pairs = (slots == slot).nonzero().flatten()
        if pairs.numel() == 0:
            continue
        tokens = pairs // ids.size(1)
        h = x[tokens].float() @ weights.w13_ref[slot].T
        act = F.silu(h[:, :HUB_INTER]) * h[:, HUB_INTER:]
        out.index_add_(0, tokens, (act @ weights.w2_ref[slot].T) * flat_w[pairs, None])
    return out


def _rel_l2(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def _run_experts(experts, weights, x, ids, topk_w, num_experts, expert_map):
    """One ``apply`` on NaN-filled buffers, so that any element the op sequence
    fails to define shows up as NaN."""
    M = x.size(0)
    shapes = experts.workspace_shapes(
        M, HUB_INTER, HUB_K, HUB_TOP_K, num_experts, HUB_SLOTS, None, MoEActivation.SILU
    )
    nan = float("nan")
    output = torch.full(shapes[2], nan, dtype=x.dtype, device=x.device)
    experts.apply(
        output,
        x,
        weights.w13,
        weights.w2,
        topk_w,
        ids,
        MoEActivation.SILU,
        num_experts,
        expert_map,
        None,
        None,
        torch.full(shapes[0], nan, dtype=x.dtype, device=x.device),
        torch.full(shapes[1], nan, dtype=x.dtype, device=x.device),
        None,
        False,
    )
    return output


def _q32_finalize_reference(accumulator):
    """int64 -> fp32 -> fp16, NaN for a flagged row or an fp16 overflow."""
    value = (accumulator[:, :-1].float() * (1.0 / (1 << 32))).half()
    poisoned = (accumulator[:, -1:] != 0) | ~torch.isfinite(value)
    return value.masked_fill(poisoned, float("nan"))


@gfx1100_only
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M", HUB_TOKENS)
def test_hub_moe_matches_dense_reference(monkeypatch, hub_weights, mode, M):
    identity = _slot_map(HUB_SLOTS, torch.arange(HUB_SLOTS, device=device))
    x, ids, topk_w = _routing(M, HUB_SLOTS, identity, seed=M)
    experts = _hub_experts(monkeypatch, hub_weights, HUB_SLOTS, q32=mode == "q32")
    out = _run_experts(experts, hub_weights, x, ids, topk_w, HUB_SLOTS, None)
    ref = _dense_moe(x, ids, topk_w, identity, hub_weights)
    assert _rel_l2(out, ref) < 0.02, mode


@gfx1100_only
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M", HUB_TOKENS)
def test_hub_masked_ids_and_slot_map_match_dense_reference(
    monkeypatch, hub_weights, mode, M
):
    num_experts = 64
    slot_map = _slot_map(num_experts)
    x, ids, topk_w = _routing(M, num_experts, slot_map, masked=0.25, seed=M)
    experts = _hub_experts(monkeypatch, hub_weights, num_experts, q32=mode == "q32")
    out = _run_experts(experts, hub_weights, x, ids, topk_w, num_experts, slot_map)
    ref = _dense_moe(x, ids, topk_w, slot_map, hub_weights)
    assert _rel_l2(out, ref) < 0.02, mode


@gfx1100_only
@pytest.mark.parametrize("M", HUB_TOKENS)
def test_q32_mode_is_bit_identical_over_50_repeats(monkeypatch, hub_weights, M):
    num_experts = 64
    slot_map = _slot_map(num_experts)
    x, ids, topk_w = _routing(M, num_experts, slot_map, masked=0.25, seed=M)
    experts = _hub_experts(monkeypatch, hub_weights, num_experts, q32=True)
    first = _run_experts(experts, hub_weights, x, ids, topk_w, num_experts, slot_map)
    assert not first.isnan().any()
    for _ in range(49):
        again = _run_experts(
            experts, hub_weights, x, ids, topk_w, num_experts, slot_map
        )
        assert torch.equal(again, first)


@gfx1100_only
@pytest.mark.parametrize("M", HUB_TOKENS)
def test_q32_mode_agrees_with_default_mode_within_fp16_noise(
    monkeypatch, hub_weights, M
):
    num_experts = 64
    slot_map = _slot_map(num_experts)
    x, ids, topk_w = _routing(M, num_experts, slot_map, masked=0.25, seed=M)
    fixed = _hub_experts(monkeypatch, hub_weights, num_experts, q32=True)
    q32 = _run_experts(fixed, hub_weights, x, ids, topk_w, num_experts, slot_map)
    default = _hub_experts(monkeypatch, hub_weights, num_experts, q32=False)
    atomic = _run_experts(default, hub_weights, x, ids, topk_w, num_experts, slot_map)
    assert _rel_l2(q32, atomic) < 5e-3


@gfx1100_only
@pytest.mark.parametrize("M", [24, 48])
def test_q32_block_size_override_sweep(monkeypatch, hub_weights, M):
    identity = _slot_map(HUB_SLOTS, torch.arange(HUB_SLOTS, device=device))
    x, ids, topk_w = _routing(M, HUB_SLOTS, identity, seed=M)
    ref = _dense_moe(x, ids, topk_w, identity, hub_weights)
    outputs = {}
    for block_m in (None, 1, 2, 4, 8):
        experts = _hub_experts(
            monkeypatch, hub_weights, HUB_SLOTS, q32=True, block_m=block_m
        )
        outputs[block_m] = _run_experts(
            experts, hub_weights, x, ids, topk_w, HUB_SLOTS, None
        )
        assert _rel_l2(outputs[block_m], ref) < 0.02, block_m
    # None picks 4 for M > 4, so it is the BLOCK_M=4 result; the sizes differ
    # only in how rows share a weight tile, not in the arithmetic.
    assert torch.equal(outputs[None], outputs[4])
    for block_m in (1, 2, 8):
        assert _rel_l2(outputs[block_m], outputs[4]) < 2e-3, block_m


@gfx1100_only
@pytest.mark.parametrize("mode", MODES)
def test_partial_sums_over_split_experts_equal_the_full_result(
    monkeypatch, hub_weights, mode
):
    M, half = 48, HUB_SLOTS // 2
    slots = torch.arange(HUB_SLOTS, device=device)
    full_map = _slot_map(HUB_SLOTS, slots)
    map_a = _slot_map(HUB_SLOTS, slots[:half])
    map_b = full_map.clone()
    map_b[:half] = -1
    x, ids, topk_w = _routing(M, HUB_SLOTS, full_map, masked=0.1, seed=11)
    experts = _hub_experts(monkeypatch, hub_weights, HUB_SLOTS, q32=mode == "q32")

    def run(slot_map):
        return _run_experts(experts, hub_weights, x, ids, topk_w, HUB_SLOTS, slot_map)

    full, part_a, part_b = run(full_map), run(map_a), run(map_b)
    assert _rel_l2(part_a.float() + part_b.float(), full) < 5e-3
    if mode == "q32":
        # Three fp16 roundings at most: each part, and their sum.
        bound = 2.0**-10 * (part_a.abs() + part_b.abs() + full.abs()).float() + 1e-6
        assert ((part_a + part_b).float() - full.float()).abs().le(bound).all()


@gfx1100_only
def test_q32_mode_nonfinite_input_poisons_only_its_token(monkeypatch, hub_weights):
    M, bad_token = 8, 3
    identity = _slot_map(HUB_SLOTS, torch.arange(HUB_SLOTS, device=device))
    x, ids, topk_w = _routing(M, HUB_SLOTS, identity, seed=5)
    x[bad_token, 11] = float("inf")
    experts = _hub_experts(monkeypatch, hub_weights, HUB_SLOTS, q32=True)
    out = _run_experts(experts, hub_weights, x, ids, topk_w, HUB_SLOTS, None)
    assert out.isnan().all(dim=1).tolist() == [t == bad_token for t in range(M)]
    clean = [t for t in range(M) if t != bad_token]
    ref = _dense_moe(x, ids, topk_w, identity, hub_weights)
    assert _rel_l2(out[clean], ref[clean]) < 0.02


@gfx1100_only
def test_fixedpoint_cast_matches_the_int64_to_fp32_to_fp16_reference():
    generator = torch.Generator(device=device).manual_seed(0)
    values = torch.randint(
        -(1 << 44), 1 << 44, (64, 1000), generator=generator, device=device
    )
    specials = [0, 1 << 32, -(1 << 31), 1 << 44, -(1 << 44), 1 << 29, 1, -1]
    # fp32 lands exactly on an fp16 tie, and the fp16 range limits.
    specials += [4297064449, -4297064449, 65504 << 32, 65520 << 32, -(65520 << 32)]
    values[0, : len(specials)] = torch.tensor(specials, device=device)
    accumulator = torch.zeros(64, 1001, dtype=torch.int64, device=device)
    accumulator[:, :-1] = values
    accumulator[5, -1] = 1
    accumulator[6, -1] = 7
    output = torch.full((64, 1000), 5.0, dtype=torch.float16, device=device)
    original = accumulator.clone()
    ops.moe_fixedpoint_to_half_rdna3(accumulator, output)
    torch.testing.assert_close(
        output, _q32_finalize_reference(accumulator), rtol=0, atol=0, equal_nan=True
    )
    assert output[5].isnan().all() and output[6].isnan().all()
    assert torch.equal(accumulator, original)


@gfx1100_only
def test_q32_kernel_rejects_malformed_accumulators(hub_weights):
    M, size_n = 4, 2 * HUB_INTER
    x = torch.randn(M, HUB_K, dtype=torch.float16, device=device)
    si, ei, ntp = moe_align_block_size(
        torch.zeros(M, 1, dtype=torch.int32, device=device), 1, HUB_SLOTS
    )

    def call(a, c, weights, scales, zeros):
        ops.moe_gptq_gemm_rdna3(
            a,
            c,
            weights,
            scales,
            zeros,
            torch.empty(0, device=device),
            si,
            ei,
            ntp,
            1,
            1,
            False,
            0,
        )

    w, s, z = hub_weights.w13, hub_weights.w13_s, hub_weights.w13_z
    call(x, torch.zeros(M, size_n + 1, dtype=torch.int64, device=device), w, s, z)
    with pytest.raises(RuntimeError, match="Q32 c must be"):
        call(x, torch.zeros(M, size_n, dtype=torch.int64, device=device), w, s, z)
    # bf16 activations cannot use the int64 accumulator.
    E, K, N = 2, 256, 128
    w_bf16 = _make_packed_weights(E, K, N)
    s_bf16 = _make_scales(E, K // 32, N, torch.bfloat16)
    z_bf16 = _make_qzeros(E, K // 32, N)
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="fp16 activations only"):
        call(
            x_bf16,
            torch.zeros(M, N + 1, dtype=torch.int64, device=device),
            w_bf16,
            s_bf16,
            z_bf16,
        )


@gfx1100_only
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M", [1, 3, 24, 48])
def test_apply_is_graph_capturable_and_reads_new_inputs_on_replay(
    monkeypatch, hub_weights, mode, M
):
    """Capture ``apply`` once, then change routing, hidden states and the
    slot_map in place: the replay must follow them (device-resident ids and
    map, no host reads, no shapes taken from data)."""
    num_experts = 64
    q32 = mode == "q32"
    slot_map = _slot_map(num_experts)
    x, ids, topk_w = _routing(M, num_experts, slot_map, masked=0.25, seed=M)
    experts = _hub_experts(monkeypatch, hub_weights, num_experts, q32=q32)
    shapes = experts.workspace_shapes(
        M, HUB_INTER, HUB_K, HUB_TOP_K, num_experts, HUB_SLOTS, None, MoEActivation.SILU
    )
    output = torch.empty(shapes[2], dtype=x.dtype, device=device)
    workspace13 = torch.empty(shapes[0], dtype=x.dtype, device=device)
    workspace2 = torch.empty(shapes[1], dtype=x.dtype, device=device)

    def run():
        experts.apply(
            output,
            x,
            hub_weights.w13,
            hub_weights.w2,
            topk_w,
            ids,
            MoEActivation.SILU,
            num_experts,
            slot_map,
            None,
            None,
            workspace13,
            workspace2,
            None,
            False,
        )

    run()
    current_platform.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    other_map = _slot_map(num_experts, torch.arange(num_experts, device=device)[-16:])
    new_x, new_ids, new_w = _routing(M, num_experts, other_map, masked=0.25, seed=99)
    x.copy_(new_x)
    ids.copy_(new_ids)
    topk_w.copy_(new_w)
    slot_map.copy_(other_map)
    graph.replay()
    current_platform.synchronize()
    expected = _run_experts(
        experts, hub_weights, new_x, new_ids, new_w, num_experts, other_map
    )
    if q32:
        assert torch.equal(output, expected)
    else:
        assert _rel_l2(output, expected) < 5e-3
