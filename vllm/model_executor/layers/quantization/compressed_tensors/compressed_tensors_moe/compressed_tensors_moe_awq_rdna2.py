# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A16 using the fused RDNA2 (gfx1030) HIP kernel.

Uses ``moe_awq_gemm_rdna2`` — a single HIP kernel launch per GEMM that
handles expert routing + W4A16 AWQ dequant + dot product with atomic output.

Weight format (per expert):
  - Packed int32 ``[E, K, N/8]`` sequential (no shuffle)
  - Scales ``[E, groups, N]`` fp16
  - Zero points ``[E, groups, N/8]`` packed int32
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)

logger = init_logger(__name__)


class CompressedTensorsAWQMoERDNA2Method(CompressedTensorsWNA16MoEMethod):
    """W4A16 MoE using the fused RDNA2 HIP kernel (moe_awq_gemm_rdna2).

    Weights are in sequential int32 [E, K, N/8] format,
    scales [E, groups, N] fp16, zeros [E, groups, N/8] int32.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert CompressedTensors weights to RDNA2 kernel format.

        CompressedTensors loads weights as [E, K/8, N] int32.
        RDNA2 kernel expects [E, K, N/8] int32 (transposed last two dims).
        """
        device = layer.w13_weight_packed.device

        # Transpose w13: [E, K/8, N] -> [E, K, N/8]
        # First convert to uint8 for proper handling
        w13 = layer.w13_weight_packed
        if w13.dtype == torch.int32:
            # CompressedTensors format: [E, K/8, N]
            # Need to transpose to [E, K, N/8] for RDNA2
            w13_t = w13.transpose(1, 2).contiguous()
            layer.w13_weight_packed = torch.nn.Parameter(
                w13_t, requires_grad=False
            )

        w2 = layer.w2_weight_packed
        if w2.dtype == torch.int32:
            w2_t = w2.transpose(1, 2).contiguous()
            layer.w2_weight_packed = torch.nn.Parameter(
                w2_t, requires_grad=False
            )

        # Ensure scales are [E, groups, N] fp16
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(torch.float16).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(torch.float16).contiguous(),
            requires_grad=False,
        )

        # Create zero points for symmetric quant (all zeros)
        num_experts = layer.w13_weight_packed.shape[0]
        w13_groups = layer.w13_weight_scale.shape[1]
        w13_n_packed = layer.w13_weight_packed.shape[2]
        w2_groups = layer.w2_weight_scale.shape[1]
        w2_n_packed = layer.w2_weight_packed.shape[2]

        layer.w13_qzeros = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                w13_groups,
                w13_n_packed,
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )
        layer.w2_qzeros = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                w2_groups,
                w2_n_packed,
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )

        logger.info_once(
            "CompressedTensorsAWQMoERDNA2Method: weights converted to RDNA2 format"
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        return _rdna2_awq_fused_moe(
            x,
            topk_weights,
            topk_ids,
            layer=layer,
            activation=activation,
            apply_router_weight_on_input=(layer.apply_router_weight_on_input),
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )


def _rdna2_awq_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: RoutedExperts,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA2 W4A16 AWQ HIP kernel."""
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    total_tokens = num_tokens * top_k
    dtype = hidden_states.dtype
    device = hidden_states.device

    N_gate_up = layer.w13_weight_packed.shape[2] * 8
    hidden_size = layer.w2_weight_packed.shape[2] * 8
    intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

    if global_num_experts <= 0:
        global_num_experts = layer.w13_weight_packed.shape[0]

    block_size_m = 1 if num_tokens <= 4 else 4

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
    )

    w1_out = torch.zeros(
        total_tokens,
        N_gate_up,
        dtype=dtype,
        device=device,
    )
    act_out = torch.empty(
        total_tokens,
        intermediate_size,
        dtype=dtype,
        device=device,
    )

    topk_w_float = topk_weights.view(-1).float()
    empty_tw = torch.empty(0, dtype=torch.float32, device=device)

    ops.moe_awq_gemm_rdna2(
        hidden_states,
        w1_out,
        layer.w13_weight_packed,
        layer.w13_weight_scale,
        layer.w13_qzeros,
        topk_w_float if apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        apply_router_weight_on_input,
    )

    apply_moe_activation(activation, act_out, w1_out)

    out = torch.zeros(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    ops.moe_awq_gemm_rdna2(
        act_out,
        out,
        layer.w2_weight_packed,
        layer.w2_weight_scale,
        layer.w2_qzeros,
        topk_w_float if not apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
        block_size_m,
        not apply_router_weight_on_input,
        output_topk=top_k,
    )

    return out
