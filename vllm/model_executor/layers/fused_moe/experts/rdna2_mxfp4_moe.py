# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) MXFP4 fused MoE experts class.

This wraps the custom HIP kernel ``moe_mxfp4_gemm_rdna2`` (registered in
``vllm/_custom_ops.py``) into vLLM's ``FusedMoEExpertsModular`` interface
so the oracle can select it for DeepSeek-V4 on gfx1030.

The kernel is called twice per MoE layer:
  1. w1+w3 GEMM: ``moe_mxfp4_gemm_rdna2(x, w13_packed, ..., output_topk=0)``
     -> ``[M*top_k, 2*N_inter]``  (no activation, no topk reduction)
  2. SwiGLU activation: applied in Python (or via a torch._C op).
  3. w2 GEMM: ``moe_mxfp4_gemm_rdna2(activated, w2_packed, ...,
     output_topk=1)`` -> ``[M, K]``  (topk reduction fused into epilogue)

This pattern matches the standard vLLM MoE flow but with our native
HIP kernel instead of AITER or Triton.
"""
import os

import torch

from vllm.logger import init_logger
from vllm import _custom_ops as ops

logger = init_logger(__name__)

_DEBUG_NAN = os.environ.get("VLLM_RDNA2_MOE_DEBUG_NAN", "0") == "1"
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


def _swiglu_split(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: split last dim into (gate, up), apply silu(gate) * up."""
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


class RDNA2Mxfp4MoEExperts(FusedMoEExpertsModular):
    """RDNA2 (gfx1030) MXFP4 fused MoE experts using our custom HIP kernel."""

    expects_unquantized_inputs: bool = True

    @staticmethod
    def _supports_current_device() -> bool:
        if not hasattr(torch.ops, "_rocm_C"):
            return False
        return hasattr(torch.ops._rocm_C, "moe_mxfp4_gemm_rdna2")

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Our kernel: MXFP4 weights, FP8 dense activations (DeepSeek-V4)
        # or unquantized activations (smoke tests).
        return (weight_key, activation_key) in (
            (kMxfp4Static, kFp8StaticTensorSym),
            (kMxfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method == RoutingMethodType.DeepseekV4

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype,
        routing_method: RoutingMethodType,
    ) -> bool:
        return router_logits_dtype in (torch.float16, torch.bfloat16, torch.float32)

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 256 == 0

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # intermediate buffer holds the activated w1+w3 output
        N_inter = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, N_inter)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output)

    def finalize_weight_and_reduce_impl(self):
        # Kernel fuses topk reduction in output_topk=1 epilogue
        return TopKWeightAndReduceNoOP()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Cache weight scales from the FusedMoE parent layer. These are
        # the per-block UE8M0 scales for the MXFP4 weights, which the
        # kernel needs but the modular apply() signature does not pass
        # (a1q_scale/a2_scale are input quant scales, not weight scales).
        self.w13_weight_scale = layer.w13_weight_scale
        self.w2_weight_scale = layer.w2_weight_scale

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
        expert_tokens_meta,
        apply_router_weight_on_input: bool,
    ):
        if _DEBUG_NAN:
            hs = hidden_states
            logger.warning(
                "MoE in: dtype=%s max=%.1f nan=%d inf=%d M=%d K=%d topk=%d",
                hs.dtype, hs.float().abs().max().item(),
                torch.isnan(hs).sum().item(), torch.isinf(hs).sum().item(),
                hs.size(0), hs.size(-1), topk_ids.size(1))

        # Kernel requires fp16 activations (V_DOT2_F32_F16; gfx1030 has no bf16 dot).
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        local_num_experts = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        topk = topk_ids.size(1)
        K = hidden_states.size(-1)
        # w1 is [E, K/8, 2*N_inter] int32 after the convert step.
        N_inter = self.adjust_N_for_activation(w1.shape[2], activation)

        # Kernel is decode-oriented: only block_size_m in {1, 2, 4, 8}.
        block_size_m = 8
        M = hidden_states.size(0)

        # Routing prep: sort tokens by expert, pad to block alignment
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, local_num_experts, expert_map,
            ignore_invalid_experts=True,
        )

        # --- Pass 1: w1+w3 GEMM (gate+up), output [M*topk, 2*N_inter] ---
        w13_packed = w1  # [E, K/8, 2*N_inter] int32
        w13_scales = getattr(self, "w13_weight_scale", None)
        if w13_scales is None:
            w13_scales = a1q_scale
        if w13_scales is None:
            raise RuntimeError(
                "RDNA2Mxfp4MoEExperts: w13_weight_scale not available "
                "(process_weights_after_loading did not cache it and "
                "a1q_scale is None). The kernel requires per-block UE8M0 "
                "weight scales for the MXFP4 weights."
            )
        w1_out = torch.zeros(M * topk, w1.shape[2], dtype=hidden_states.dtype,
                             device=hidden_states.device)
        ops.moe_mxfp4_gemm_rdna2(
            hidden_states,
            w1_out,
            w13_packed,
            w13_scales,
            torch.empty(0, device=hidden_states.device),  # topk_weights unused here
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk,
            block_size_m,
            False,  # mul_topk_weight
            0,      # output_topk
        )

        # --- Activation: SwiGLU split -> [M*topk, N_inter] ---
        activated = _swiglu_split(w1_out) if activation == MoEActivation.SILU else w1_out

        if _DEBUG_NAN:
            logger.warning(
                "MoE pass1: w1_out max=%.1f nan=%d inf=%d; act max=%.1f nan=%d; "
                "w13_s dtype=%s min=%d max=%d nz=%d/%d; w13_w nz=%d/%d",
                w1_out.float().abs().max().item(), torch.isnan(w1_out).sum().item(),
                torch.isinf(w1_out).sum().item(),
                activated.float().abs().max().item(), torch.isnan(activated).sum().item(),
                w13_scales.dtype, w13_scales.min().item(), w13_scales.max().item(),
                (w13_scales != 0).sum().item(), w13_scales.numel(),
                (w13_packed != 0).sum().item(), w13_packed.numel())

        # --- Pass 2: w2 GEMM (down) with topk reduction, output [M, K] ---
        # w2 shape: [E, N_inter, K]; we treat it as the "b_q_weight" for the
        # second call, with output_topk=1 to fuse moe_sum.
        # Re-align routing for the second pass (same routing, but the kernel
        # reads from the activated buffer which is already in routed order).
        w2_scales = getattr(self, "w2_weight_scale", None)
        if w2_scales is None:
            w2_scales = a2_scale
        if w2_scales is None:
            raise RuntimeError(
                "RDNA2Mxfp4MoEExperts: w2_weight_scale not available "
                "(process_weights_after_loading did not cache it and "
                "a2_scale is None)."
            )

        out_buf = (
            output
            if output.dtype == torch.float16
            else torch.empty(output.shape, dtype=torch.float16, device=output.device)
        )
        # Epilogue is atomic-add; the buffer must start at zero.
        out_buf.zero_()
        ops.moe_mxfp4_gemm_rdna2(
            activated,
            out_buf,
            w2,
            w2_scales,
            topk_weights.view(-1) if topk_weights.numel() > 0 else torch.empty(0, device=hidden_states.device),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,      # top_k=1: sorted tokens map 1:1 to activated rows
            block_size_m,
            True,   # mul_topk_weight
            topk,   # output_topk: reduce token_id/topk back to [M, K] rows
        )
        if out_buf is not output:
            output.copy_(out_buf.to(output.dtype))

        if _DEBUG_NAN:
            logger.warning(
                "MoE out: max=%.1f nan=%d inf=%d",
                output.float().abs().max().item(), torch.isnan(output).sum().item(),
                torch.isinf(output).sum().item())
