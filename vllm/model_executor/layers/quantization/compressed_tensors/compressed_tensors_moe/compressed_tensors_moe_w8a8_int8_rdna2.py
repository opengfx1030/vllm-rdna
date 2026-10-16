# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W8A8 int8 through the fused RDNA2 (gfx1030) sdot4 kernel.

Uses ``moe_w8a8_gemm_rdna2``: one HIP launch per GEMM that quantizes the
activation batch per (token, group), handles expert routing and accumulates
the int8 x int8 dot products with the router weight folded in.

Weight format (per expert), built in ``process_weights_after_loading`` from the
compressed-tensors W8A8 int8 checkpoint layout:
  - Packed int32 ``[E, K/4, N]`` (``rdna2_w8a8.pack_moe_weight``)
  - Scales fp16 ``[E, 1, N]`` (per-channel, symmetric)

Opt-in: ``VLLM_RDNA2_W8A8_SDOT4=1`` on gfx1030 (``rocm_moe_rdna``), for
symmetric per-channel weights with symmetric dynamic per-token activations.
"""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

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
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization import rdna2_w8a8
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

# Prefill tile width (rows per block) and the token count it kicks in at;
# decode keeps the narrow tiles like the other RDNA2 MoE methods.
_BSM_LARGE = 8
_BSM_LARGE_MIN_TOKENS = 512


class CompressedTensorsW8A8Int8RDNA2MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    """W8A8 int8 MoE on the fused RDNA2 sdot4 HIP kernel.

    Reuses the parent's checkpoint-format ``create_weights`` (int8 [E, N, K]
    weights, fp32 [E, N, 1] scales) and replaces the int8 MoE backend
    selection with a direct launch of ``moe_w8a8_gemm_rdna2``.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        # Skip CompressedTensorsW8A8Int8MoEMethod.__init__: it picks a generic
        # int8 MoE backend this method does not use.
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.static_input_scales = not self.input_quant.dynamic

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel or self.static_input_scales:
            raise ValueError(
                "RDNA2 W8A8 MoE requires channelwise weights with dynamic "
                f"per-token activations. Found {self.weight_quant}, "
                f"{self.input_quant}"
            )
        if not (self.weight_quant.symmetric and self.input_quant.symmetric):
            raise ValueError("RDNA2 W8A8 MoE requires symmetric quantization")
        if self.moe.has_bias:
            raise ValueError("RDNA2 W8A8 MoE does not support expert biases")

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        for w_name, s_name in (
            ("w13_weight", "w13_weight_scale"),
            ("w2_weight", "w2_weight_scale"),
        ):
            w = getattr(layer, w_name).data  # int8 [E, N, K]
            s = getattr(layer, s_name).data  # fp32 [E, N, 1]
            n = w.shape[1]
            replace_parameter(
                layer,
                w_name,
                torch.nn.Parameter(rdna2_w8a8.pack_moe_weight(w), requires_grad=False),
            )
            replace_parameter(
                layer,
                s_name,
                torch.nn.Parameter(
                    s.reshape(num_experts, 1, n).to(torch.float16).contiguous(),
                    requires_grad=False,
                ),
            )

        hidden_size = layer.w2_weight.shape[2]
        n_gate_up = layer.w13_weight.shape[2]
        intermediate = layer.w2_weight.shape[1] * 4
        self._a_group_w13 = rdna2_w8a8.act_group(hidden_size)
        self._a_group_w2 = rdna2_w8a8.act_group(intermediate)

        # Decode-sized buffers, reused when the batch fits (as the W8A16
        # RDNA2 method does); larger batches allocate per call.
        max_decode_tokens = 16
        top_k = 8
        buf_size = max_decode_tokens * top_k
        layer.rdna2_w1_buf = torch.zeros(
            buf_size, n_gate_up, dtype=torch.float16, device=device
        )
        layer.rdna2_act_buf = torch.empty(
            buf_size, intermediate, dtype=torch.float16, device=device
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # Metadata only: the fused op quantizes activations itself and reads
        # the packed scales directly.
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
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
        assert not self.is_monolithic
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        n_gate_up = layer.w13_weight.shape[2]
        intermediate_size = n_gate_up // 2 if activation.is_gated else n_gate_up
        hidden_size = layer.w2_weight.shape[2]

        num_tokens = x.shape[0]
        device = x.device
        dtype = x.dtype
        if dtype != torch.float16:
            x = x.to(torch.float16)

        global_num_experts = (
            layer.global_num_experts
            if getattr(layer, "global_num_experts", 0) > 0
            else layer.w13_weight.shape[0]
        )
        expert_map = getattr(layer, "expert_map", None)
        top_k = topk_weights.shape[-1] if topk_weights.numel() > 0 else 1
        apply_router_weight_on_input = getattr(
            layer, "apply_router_weight_on_input", False
        )
        total_tokens = num_tokens * top_k

        if num_tokens <= 4:
            block_size_m = 1
        elif num_tokens < _BSM_LARGE_MIN_TOKENS:
            block_size_m = 4
        else:
            block_size_m = _BSM_LARGE

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts, expert_map
        )

        if total_tokens <= layer.rdna2_w1_buf.shape[0]:
            w1_out = layer.rdna2_w1_buf[:total_tokens]
            w1_out.zero_()
            act_out = layer.rdna2_act_buf[:total_tokens]
        else:
            w1_out = torch.zeros(
                total_tokens, n_gate_up, dtype=torch.float16, device=device
            )
            act_out = torch.empty(
                total_tokens, intermediate_size, dtype=torch.float16, device=device
            )

        topk_w_float = topk_weights.view(-1).float()
        empty_tw = layer.rdna2_empty_tw

        ops.moe_w8a8_gemm_rdna2(
            x,
            w1_out,
            layer.w13_weight,
            layer.w13_weight_scale,
            topk_w_float if apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
            0,
            self._a_group_w13,
        )

        apply_moe_activation(activation, act_out, w1_out)

        # output_topk=top_k fuses moe_sum into the atomic accumulation.
        out = torch.zeros(num_tokens, hidden_size, dtype=torch.float16, device=device)
        ops.moe_w8a8_gemm_rdna2(
            act_out,
            out,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_w_float if not apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            top_k,
            self._a_group_w2,
        )
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def apply_monolithic(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("RDNA2 W8A8 MoE is not a monolithic kernel")
