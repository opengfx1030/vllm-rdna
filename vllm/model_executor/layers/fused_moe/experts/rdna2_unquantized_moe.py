# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP unquantized (fp16 weight / fp16 activation) MoE experts for gfx1030.

Replaces the two ``fused_moe_kernel`` launches of the Triton unquantized path
(``vllm/model_executor/layers/fused_moe/fused_moe.py``) plus the alignment and
the separate activation launch on RDNA2. The top-k reduction is the shared fp32
``ops.moe_sum``, so this backend rounds exactly where the Triton path rounds:
once per (token, top-k) pair after the down projection, then only at the end.
Selected as the "RDNA2 HIP" unquantized MoE backend by
``oracle/unquantized.py`` when ``on_rdna2()`` is true; TRITON stays the backend
on every other platform.

Layout contract (see ``csrc/rocm/moe_unquantized_rdna2.cu``): ``w1`` is the
checkpoint-layout ``[E, 2N, K]`` and ``w2`` is ``[E, K, N]``, so no weight
conversion is needed (``convert_to_unquantized_kernel_format`` is a no-op here).

Deploy geometry (Qwen4ExpMTP drafter, W4A16 target + fp16 drafter, TP4/EP4):
E=128 local (512 global), N=640, K=2560, top_k=10, M=8..24. This is a batch of
GEMVs that is bound by the weight bytes read per routed expert; the kernels
block by (token, top-k slot) instead of sorting tokens by expert, so they need
no ``moe_align_block_size`` and no intermediate cache1.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms.rdna import on_rdna2

_GATE_UP_OP = "moe_gate_up_silu_unquantized_rdna2"
_DOWN_OP = "moe_down_unquantized_rdna2"


def rdna2_unquantized_moe_available() -> bool:
    """True when both HIP ops are registered in ``_rocm_C``."""
    return (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, _GATE_UP_OP)
        and hasattr(torch.ops._rocm_C, _DOWN_OP)
    )


def _require_ops() -> None:
    """Raise rather than silently leaving the unquantized path on TRITON."""
    if not rdna2_unquantized_moe_available():
        raise RuntimeError(
            "RDNA2 unquantized MoE: "
            f"{_GATE_UP_OP}/{_DOWN_OP} are missing from _rocm_C; "
            "refusing to fall back to the Triton fused_moe kernel"
        )


class Rdna2UnquantizedExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def is_monolithic() -> bool:
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # Selected on gfx10x whether or not the binding is present: a missing
        # binding must fail loudly in apply() rather than silently leaving the
        # unquantized path on TRITON (see _require_ops).
        return on_rdna2()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        # The kernel computes silu(gate)*up from one [E, 2N, K] projection.
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_batch_invariance() -> bool:
        # The kernels themselves are batch-invariant (per-pair blocks, fixed
        # reduction order), but that has not been verified end to end here.
        return False

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        # fp16 only: the kernels load 16-byte fp16 tiles and use
        # v_dot2_f32_f16. bf16 (and every other dtype) falls through to
        # TRITON, the same way any other unsupported config does.
        if moe_config.in_dtype != torch.float16:
            return False, "kernel requires fp16 activations"
        hidden = moe_config.hidden_dim
        inter = moe_config.intermediate_size_per_partition
        if hidden % 8 or inter % 8:
            return False, "kernel requires hidden_dim and intermediate_size % 8 == 0"
        return True, None

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Identical to TritonExperts: workspace13 holds the activation cache,
        # workspace2 the `[M, topk, K]` per-slot down results that `moe_sum`
        # reduces. workspace2 must stay disjoint from workspace13 -- `output`
        # is the first M*K elements of the latter.
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert activation == MoEActivation.SILU, activation
        assert a1q_scale is None and a2_scale is None
        assert hidden_states.dtype == torch.float16, hidden_states.dtype
        assert hidden_states.is_contiguous()
        assert hidden_states.dim() == 2
        assert output.is_contiguous()

        _require_ops()
        _, num_tokens, N, K, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        n_dim = self.adjust_N_for_activation(N, activation)
        assert w1.shape[1] == N and w1.shape[2] == K and N == 2 * n_dim

        act = _resize_cache(workspace13, (num_tokens * top_k, n_dim))
        # Per-(token, top-k) down result, rounded once to fp16, then reduced in
        # fp32 by the same `moe_sum` the Triton path uses.
        slot_out = _resize_cache(workspace2, (num_tokens, top_k, K))
        ids = topk_ids.reshape(-1)

        torch.ops._rocm_C.moe_gate_up_silu_unquantized_rdna2(
            hidden_states, w1, ids, expert_map, act, top_k, n_dim, K
        )
        # apply_router_weight_on_input folds the router weight into the layer
        # input (and the nonlinearity changes with it), so it must not be
        # applied a second time here.
        mul_weight = not apply_router_weight_on_input
        torch.ops._rocm_C.moe_down_unquantized_rdna2(
            act,
            w2,
            ids,
            expert_map,
            topk_weights.reshape(-1) if mul_weight else topk_weights.new_empty(0),
            mul_weight,
            slot_out,
            top_k,
            n_dim,
            K,
        )
        ops.moe_sum(slot_out, output, topk_ids, expert_map)
