# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W8A16 using the fused RDNA2 (gfx1030) HIP kernel.

Uses ``moe_w8a16_gemm_rdna2`` — a single HIP kernel launch per GEMM that
handles expert routing + INT8 dequant + dot product with atomic output.

Weight format (per expert):
  - Stored int8 [E, K, N] (no packing)
  - Scales [E, groups, N] in activation dtype
  - Zero points [E, groups, N] in activation dtype (optional, fp16 zero)
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


class CompressedTensorsWNA16RDNA2W8A16MoEMethod(CompressedTensorsWNA16MoEMethod):
    """W8A16 MoE (INT8 weight + fp16 act) using the fused RDNA2 HIP kernel.

    Weights are already in the right storage layout at layer load
    (no nibble packing, no exllama shuffle). apply() dispatches through
    the fused HIP kernel directly.

    Supports per-channel (group_size == size_n) and per-group
    (group_size < size_n) quantization, both with optional fp16 zeros.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Override the base class to allocate the correct buffer size.
        # The base class uses packed_factor = 32 / num_bits = 4 for
        # num_bits=8, which would store 4 int8 per int32 (4 bytes for
        # 4 weights). For W8A16 we want one byte per weight, so
        # packed_factor = 1.
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w13 = hidden_size // self.group_size
            num_groups_w2 = intermediate_size_per_partition // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, num_groups_w2, hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        # g_idx for W8A16 is unused (no actorder). Register empty
        # buffers so the upstream loader's view()/reshape() calls don't
        # fail. The fused kernel does not consume g_idx.
        layer.register_parameter("w13_weight_g_idx",
            torch.nn.Parameter(torch.empty(0, dtype=torch.int32),
                               requires_grad=False))
        layer.register_parameter("w2_weight_g_idx",
            torch.nn.Parameter(torch.empty(0, dtype=torch.int32),
                               requires_grad=False))
        layer.register_parameter("w13_g_idx_sort_indices",
            torch.nn.Parameter(torch.empty(0, dtype=torch.int32),
                               requires_grad=False))
        layer.register_parameter("w2_g_idx_sort_indices",
            torch.nn.Parameter(torch.empty(0, dtype=torch.int32),
                               requires_grad=False))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight_packed.device
        act_dtype = layer.w13_weight_scale.dtype

        N_gate_up = layer.w13_weight_packed.shape[2]
        hidden_size = layer.w2_weight_packed.shape[2]
        intermediate = N_gate_up // 2

        max_decode_tokens = 16
        top_k = 8
        buf_size = max_decode_tokens * top_k
        layer.rdna2_w1_buf = torch.zeros(
            buf_size, N_gate_up, dtype=act_dtype, device=device,
        )
        layer.rdna2_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device,
        )
        layer.rdna2_out_buf = torch.zeros(
            max_decode_tokens, hidden_size, dtype=act_dtype, device=device,
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

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
        intermediate_size = (
            layer.intermediate_size_per_partition
            if hasattr(layer, "intermediate_size_per_partition")
            else layer.intermediate_size
        )
        N_gate_up = layer.w13_weight_packed.shape[2]
        intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

        num_tokens = x.shape[0]
        hidden_size = layer.w2_weight_packed.shape[2]
        device = x.device
        dtype = x.dtype

        if hasattr(layer, "global_num_experts") and layer.global_num_experts > 0:
            global_num_experts = layer.global_num_experts
        else:
            global_num_experts = layer.w13_weight_packed.shape[0]

        expert_map = (
            layer.expert_map if hasattr(layer, "expert_map") else None
        )

        top_k = topk_weights.shape[-1] if topk_weights.numel() > 0 else 1
        apply_router_weight_on_input = (
            getattr(layer, "apply_router_weight_on_input", False)
        )

        total_tokens = num_tokens * top_k

        block_size_m = 1 if num_tokens <= 4 else 4

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
        )

        if total_tokens <= layer.rdna2_w1_buf.shape[0]:
            w1_out = layer.rdna2_w1_buf[:total_tokens]
            w1_out.zero_()
            act_out = layer.rdna2_act_buf[:total_tokens]
        else:
            w1_out = torch.zeros(
                total_tokens, N_gate_up, dtype=dtype, device=device,
            )
            act_out = torch.empty(
                total_tokens, intermediate_size, dtype=dtype, device=device,
            )

        topk_w_float = topk_weights.view(-1).float()
        empty_tw = layer.rdna2_empty_tw

        ops.moe_w8a16_gemm_rdna2(
            x,
            w1_out,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            getattr(layer, "w13_qzeros", layer.rdna2_empty_tw),
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
            num_tokens, hidden_size, dtype=dtype, device=device,
        )
        ops.moe_w8a16_gemm_rdna2(
            act_out,
            out,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            getattr(layer, "w2_qzeros", layer.rdna2_empty_tw),
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
