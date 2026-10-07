# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W8A16-FP8 using the fused RDNA2 (gfx1030) HIP kernel.

Uses ``moe_w8a16_fp8_gemm_rdna2`` — a single HIP kernel launch per GEMM that
handles expert routing + FP8 dequant (per-tile FP8->fp32->fp16) + dot product
with atomic output.

Storage format (per expert):
  - Stored float8_e4m3fn [E, K, N] (no packing, no requant on disk)
  - Scales [E, groups, N] in activation dtype (fp16)
  - Zero points [E, groups, N] in float8_e4m3fn (optional)

Activation dtype is fp16; the kernel does per-tile FP8->fp16 conversion
before v_dot2_f32_f16. This is the only FP8 path that runs correctly on
gfx1030 (no native FP8 multiply; the conversion is unavoidable).
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
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsFP8RDNA2MoEMethod(CompressedTensorsWNA16MoEMethod):
    """W8A16-FP8 MoE (FP8 weight + fp16 act) using the fused RDNA2 HIP kernel.

    Storage stays FP8 (no requant on disk); per-tile FP8->fp16 conversion
    inside the kernel before v_dot2_f32_f16.

    Mirrors the W8A16 INT8 method's structure but with FP8 weight dtype.
    """

    def __init__(
        self,
        weight_quant,
        input_quant,
        moe,
        layer_name: str | None = None,
    ):
        # We accept input_quant but ignore it (FP8 W8A16 == fp16 act).
        # The base __init__ requires input_quant to be passed and
        # validates it; pass through.
        super().__init__(
            weight_quant=weight_quant,
            input_quant=input_quant,
            moe=moe,
            layer_name=layer_name,
        )
        # Override num_bits to 8 (the base class derives it from
        # weight_quant; we want FP8 weight which is also 8 bits).
        self.num_bits = 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # Allocate weights as float8_e4m3fn (1 byte per element).
        # OCP fp8 on gfx1030 (rocm.py:fp8_dtype returns torch.float8_e4m3fn).
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
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
                dtype=torch.float8_e4m3fn,
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

        # Empty g_idx (no actorder support)
        for name in ("w13_weight_g_idx", "w2_weight_g_idx",
                     "w13_g_idx_sort_indices", "w2_g_idx_sort_indices"):
            layer.register_parameter(name,
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
        N_gate_up = layer.w13_weight_packed.shape[2]
        intermediate_size = (
            N_gate_up // 2 if activation.is_gated else N_gate_up
        )

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

        top_k = (topk_weights.shape[-1]
                 if topk_weights.numel() > 0 else 1)
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
        zeros = getattr(layer, "w13_qzeros", empty_tw)

        ops.moe_w8a16_fp8_gemm_rdna2(
            x,
            w1_out,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            zeros,
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
        ops.moe_w8a16_fp8_gemm_rdna2(
            act_out,
            out,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            getattr(layer, "w2_qzeros", empty_tw),
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
