# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) W4A16 fused MoE experts class.

Wraps the custom HIP kernel ``moe_gptq_gemm_rdna2`` (registered in
``vllm/_custom_ops.py``) into vLLM's ``FusedMoEExpertsModular`` interface
so the oracle can select it for Qwen3.8-Flash-Next-AWQ (and similar W4A16
MoEs) on gfx1030.

The kernel is called twice per MoE layer:
  1. w1+w3 GEMM (gate+up) -> [M*top_k, N_gate_up]
  2. SwiGLU activation (Python) -> [M*top_k, N_inter]
  3. w2 GEMM (down) with topk reduction fused -> [M, K]

This pattern mirrors the RDNA2 MXFP4 experts class but with int4 weights
(scales/zeros in activation dtype).
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
)

logger = init_logger(__name__)


def _swiglu_split(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: split last dim into (gate, up), apply silu(gate) * up."""
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


class RDNA2W4A16MoEExperts(FusedMoEExpertsModular):
    """RDNA2 (gfx1030) W4A16 fused MoE experts using our custom HIP kernel."""

    expects_unquantized_inputs: bool = True

    @staticmethod
    def _supports_current_device() -> bool:
        if not hasattr(torch.ops, "_rocm_C"):
            return False
        return hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2")

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key,
        activation_key,
    ) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt4Static,
        )
        return weight_key == kInt4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method,
        weight_key,
        activation_key,
    ) -> bool:
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype,
        routing_method,
    ) -> bool:
        return router_logits_dtype in (torch.float16, torch.bfloat16, torch.float32)

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return True

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
    ) -> tuple:
        N_inter = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, N_inter)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The HIP kernel expects shuffled exllama-format int32 weights.
        # The quant_method (CompressedTensorsWNA16RDNA2MoEMethod) handles
        # the shuffle in its own process_weights_after_loading; we just
        # need to expose the weight tensors to the kernel here.
        pass

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
        # Kernel requires fp16 activations (V_DOT2_F32_F16; gfx1030 has no bf16 dot).
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        local_num_experts = w1.shape[0]
        if global_num_experts <= 0:
            global_num_experts = local_num_experts

        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        N_gate_up = w1.shape[2]

        # BLOCK_SIZE_M=1 for decode (small M), 4 for prefill
        block_size_m = 1 if num_tokens <= 4 else 4

        # Routing prep: sort tokens by expert, pad to block alignment
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, local_num_experts, expert_map,
            ignore_invalid_experts=True,
        )

        # The w1/w2 tensors from the modular interface are the packed int32
        # weights in [E, K/8, N] layout (shuffled by the quant_method).
        # The scales/zeros come from the layer's named buffers.
        w13_scales = getattr(self, "w13_weight_scale", None)
        w13_qzeros = getattr(self, "w13_weight_scale_zeros", None)
        w2_scales = getattr(self, "w2_weight_scale", None)
        w2_qzeros = getattr(self, "w2_weight_scale_zeros", None)

        # --- Pass 1: w1+w3 GEMM (gate+up) ---
        w1_out = torch.zeros(
            num_tokens * top_k, N_gate_up,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        topk_w_float = (
            topk_weights.view(-1).float()
            if topk_weights.numel() > 0
            else torch.empty(0, device=hidden_states.device)
        )
        empty_tw = torch.empty(0, device=hidden_states.device)
        ops.moe_gptq_gemm_rdna2(
            hidden_states,
            w1_out,
            w1,
            w13_scales,
            w13_qzeros,
            topk_w_float if apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            False,  # mul_topk_weight
            0,      # output_topk
        )

        # --- Activation: SwiGLU split ---
        if activation == MoEActivation.SILU:
            activated = _swiglu_split(w1_out)
        else:
            activated = w1_out

        # --- Pass 2: w2 GEMM (down) with topk reduction ---
        K = hidden_states.shape[-1]
        out_buf = output
        if out_buf.dtype != torch.float16:
            out_buf = torch.empty(output.shape, dtype=torch.float16, device=output.device)
        out_buf.zero_()
        ops.moe_gptq_gemm_rdna2(
            activated,
            out_buf,
            w2,
            w2_scales,
            w2_qzeros,
            topk_w_float,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,      # top_k=1: sorted tokens map 1:1 to activated rows
            block_size_m,
            True,   # mul_topk_weight
            top_k,  # output_topk: reduce back to [M, K]
        )
        if out_buf is not output:
            output.copy_(out_buf.to(output.dtype))
