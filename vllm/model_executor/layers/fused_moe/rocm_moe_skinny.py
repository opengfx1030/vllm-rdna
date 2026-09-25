# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x wave-per-row W4A16 MoE skinny GEMV dispatch (recipe 0008).

Ported from leapdragon/vllm-rdna2-recipe (Aron Hsiao).

This path is for **Triton WNA16 sequential packing only**:

  qweight: k-sequential nibbles, value = (nibble - 8) * scale (uint4b8)
  w1: ``[E, 2N, K/8]`` (or uint8 ``[E, 2N, K/2]`` with the same bytes)
  scales: ``[E, rows, K/G]``

It must **not** run on ``CompressedTensorsWNA16RDNA2MoEMethod`` weights.
Those are shuffled Exllama ``[E, K/8, N]``. Calling this kernel on them
is silently wrong, not a launch failure.

Gated to gfx10x, fp16, symmetric int4, SILU, M<=8, optional EP expert mapping,
no ``apply_router_weight_on_input``. ``VLLM_ROCM_MOE_SKINNY=0`` disables.

Handover A/B (do this on gfx1030 before making skinny the production
Triton-WNA16 decode path, and before writing a shuffled-layout sibling):

Recipe measured 15.6 -> 26.9 t/s on Intel Qwen3.5-122B-A10B PP=3 after
replacing tile Triton at decode. This fork's live W4 MoE is
``moe_gptq_gemm_rdna2`` (tile, V_DOT2, atomics, shuffled). Need a
bandwidth A/B at M=1..8, not just e2e tok/s.

How to fire this kernel:

  1. Disable the RDNA2 fused MoE method so Triton WNA16 runs
     (e.g. ``VLLM_DISABLED_KERNELS`` / skip
     ``CompressedTensorsWNA16RDNA2MoEMethod``).
  2. ``VLLM_ROCM_MOE_SKINNY=1`` vs ``0`` on the same Triton path.
  3. Log line ``rocm_moe_skinny: using moe_skinny_int4_decode`` must
     appear. A hook only on ``TritonExperts.apply`` is dead for WNA16
     (``TritonWNA16Experts.apply`` overrides it); both are hooked here,
     plus ``fused_experts()``.

Correctness:

  * Greedy vs the Triton tile baseline (byte-identical is the bar the
    recipe used; this fork should at least match greedy probes).
  * ``tests/kernels/quantization/test_rdna2_moe_w4a16.py`` is the
    **shuffled HIP** kernel — it does not cover this layout. Use
    ``test_rocm_moe_skinny.py`` (sequential packing).
  * EP uses the global-to-local expert map; nonlocal experts contribute zero.
  * Asymmetric zp must not.
  * Prefill M>8 must stay tile Triton / RDNA2 HIP.

Measure:

  * Effective GB/s vs ~506 GB/s DRAM ceiling (recipe claimed ~432 GB/s,
    85% of ceiling, linear in M from 1 to 8).
  * SILU fused into gate_up epilogue; down combine has no atomics.
  * M=1 vs M=8 linearity.
  * CUDA-graph capture: expert ``apply`` reuses workspace; the
    ``fused_experts()`` fallback still ``torch.empty``s (avoid that
    path under capture).
  * If skinny wins on Triton but RDNA2 HIP is still faster, do **not**
    switch production. If Triton+skinny beats HIP tile at M=1, then a
    shuffled-layout skinny follow-up is worth writing.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform

logger = init_logger(__name__)

_ROCM_MOE_SKINNY: bool | None = None
_LOGGED_USE = False

# Dispatch M cap (kernel itself allows 1..16). Recipe gated at 8.
_MAX_M = 8


def rocm_moe_skinny_available() -> bool:
    """True when the HIP op is present, env is on, and this is gfx10x."""
    global _ROCM_MOE_SKINNY
    if _ROCM_MOE_SKINNY is None:
        from vllm.platforms.rocm import on_gfx10x

        _ROCM_MOE_SKINNY = (
            current_platform.is_rocm()
            and bool(envs.VLLM_ROCM_MOE_SKINNY)
            and on_gfx10x()
            and hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "moe_skinny_int4_decode")
        )
    return _ROCM_MOE_SKINNY


def moe_skinny_decode_supported(
    *,
    use_int4_w4a16: bool,
    hidden_dtype: torch.dtype,
    num_tokens: int,
    activation: MoEActivation,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    global_num_experts: int,
    num_local_experts: int,
) -> bool:
    """Pure predicate for the skinny decode launch (no HIP / no alloc)."""
    if not use_int4_w4a16:
        return False
    if hidden_dtype != torch.float16:
        return False
    if num_tokens < 1 or num_tokens > _MAX_M:
        return False
    if activation != MoEActivation.SILU:
        return False
    if apply_router_weight_on_input:
        return False
    if w1_zp is not None or w2_zp is not None:
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if block_shape is None or len(block_shape) < 2:
        return False
    if expert_map is None:
        return global_num_experts in (-1, num_local_experts)
    return True


def try_rocm_moe_skinny_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    use_int4_w4a16: bool,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None,
    activation: MoEActivation,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    global_num_experts: int = -1,
    act_workspace: torch.Tensor | None = None,
) -> bool:
    """Run skinny GEMV into ``output`` if eligible. Returns True if it ran."""
    if not rocm_moe_skinny_available():
        return False
    if not moe_skinny_decode_supported(
        use_int4_w4a16=use_int4_w4a16,
        hidden_dtype=hidden_states.dtype,
        num_tokens=hidden_states.shape[0],
        activation=activation,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        global_num_experts=global_num_experts,
        num_local_experts=w1.shape[0],
    ):
        return False

    assert w1_scale is not None and w2_scale is not None and block_shape is not None
    m = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    inter = w1.size(1) // 2
    act_shape = (m, topk, inter)
    if act_workspace is not None:
        act_buf = _resize_cache(act_workspace, act_shape)
    else:
        act_buf = torch.empty(
            act_shape, dtype=torch.float16, device=hidden_states.device
        )

    global _LOGGED_USE
    if not _LOGGED_USE:
        _LOGGED_USE = True
        logger.info(
            "rocm_moe_skinny: using moe_skinny_int4_decode "
            "(M=%d, topk=%d, N=%d, sequential Triton WNA16 layout)",
            m,
            topk,
            inter,
        )

    ops.moe_skinny_int4_decode(
        hidden_states.contiguous(),
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        act_buf,
        output,
        block_shape[1],
        expert_map,
    )
    return True
