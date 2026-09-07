# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
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
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _is_rdna2_awq_supported() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        return False
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_awq_gemm_rdna2"
    )


class AWQMoERDNA2Method(FusedMoEMethodBase):
    """W4A16 MoE using the fused RDNA2 (gfx1030) HIP kernel.

    Uses MoeWNA16Method for weight creation/loading (handles AWQ checkpoint
    format), then repacks in process_weights_after_loading to RDNA2 kernel
    layout: [E, K, N/8] int32 sequential.
    """

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

        # Delegate weight creation to MoeWNA16Method which handles
        # the AWQ checkpoint loading format correctly.
        from .moe_wna16 import MoeWNA16Config, MoeWNA16Method

        self._wna16_config = MoeWNA16Config(
            linear_quant_method="awq",
            weight_bits=quant_config.weight_bits,
            group_size=quant_config.group_size,
            has_zp=quant_config.zero_point,
            lm_head_quantized=False,
            modules_to_not_convert=quant_config.modules_to_not_convert,
            full_config={
                "quant_method": "awq",
                "bits": quant_config.weight_bits,
                "group_size": quant_config.group_size,
                "zero_point": quant_config.zero_point,
                "lm_head": False,
                "modules_to_not_convert": quant_config.modules_to_not_convert,
            },
        )
        self._wna16_method = MoeWNA16Method(self._wna16_config, moe)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._wna16_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Repack from MoeWNA16 uint8 format to RDNA2 int32 kernel format.

        MoeWNA16 format (after AWQ checkpoint load):
          w13_qweight: [E, 2*intermediate, hidden_size//2] uint8
          w2_qweight:  [E, hidden_size, intermediate//2] uint8
          w13_scales:  [E, 2*intermediate, hidden_size//group_size] fp16
          w2_scales:   [E, hidden_size, intermediate//group_size] fp16
          w13_qzeros:  [E, 2*intermediate//2, hidden_size//group_size] uint8
          w2_qzeros:   [E, hidden_size//2, intermediate//group_size] uint8

        RDNA2 kernel format:
          w13_qweight: [E, K, N/8] int32  (K=hidden_size, N=2*intermediate)
          w2_qweight:  [E, K, N/8] int32  (K=intermediate, N=hidden_size)
          w13_scales:  [E, groups, N] fp16
          w2_scales:   [E, groups, N] fp16
          w13_qzeros:  [E, groups, N/8] int32
          w2_qzeros:   [E, groups, N/8] int32
        """
        device = layer.w13_qweight.device

        # --- Repack w13_qweight ---
        # MoeWNA16: [E, N, K//2] uint8 where N=2*intermediate, K=hidden_size
        # RDNA2:    [E, K, N//8] int32
        w13_uint8 = layer.w13_qweight  # [E, 2*intermediate, hidden_size//2]
        E, N, K_half = w13_uint8.shape
        K = K_half * 2  # hidden_size

        # Unpack uint8 -> 2 nibbles per byte
        unpacked = torch.zeros(E, N, K, dtype=torch.int32, device=device)
        unpacked[:, :, 0::2] = w13_uint8 & 0xF
        unpacked[:, :, 1::2] = (w13_uint8 >> 4) & 0xF

        # Transpose to [E, K, N] then pack N into int32
        unpacked = unpacked.permute(0, 2, 1)  # [E, K, N]
        unpacked = unpacked.reshape(E, K, N // 8, 8)
        shifts = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
        )
        w13_int32 = (unpacked << shifts).sum(dim=3, dtype=torch.int32)

        layer.w13_qweight = torch.nn.Parameter(w13_int32, requires_grad=False)

        # --- Repack w2_qweight ---
        # MoeWNA16: [E, N, K//2] uint8 where N=hidden_size, K=intermediate
        # RDNA2:    [E, K, N//8] int32
        w2_uint8 = layer.w2_qweight  # [E, hidden_size, intermediate//2]
        E, N, K_half = w2_uint8.shape
        K = K_half * 2  # intermediate

        unpacked = torch.zeros(E, N, K, dtype=torch.int32, device=device)
        unpacked[:, :, 0::2] = w2_uint8 & 0xF
        unpacked[:, :, 1::2] = (w2_uint8 >> 4) & 0xF

        unpacked = unpacked.permute(0, 2, 1)  # [E, K, N]
        unpacked = unpacked.reshape(E, K, N // 8, 8)
        w2_int32 = (unpacked << shifts).sum(dim=3, dtype=torch.int32)

        layer.w2_qweight = torch.nn.Parameter(w2_int32, requires_grad=False)

        # --- Reshape scales: [E, N, groups] -> [E, groups, N] ---
        # w13_scales: [E, 2*intermediate, hidden_size//group_size]
        layer.w13_scales = torch.nn.Parameter(
            layer.w13_scales.permute(0, 2, 1).contiguous().to(torch.float16),
            requires_grad=False,
        )
        # w2_scales: [E, hidden_size, intermediate//group_size]
        layer.w2_scales = torch.nn.Parameter(
            layer.w2_scales.permute(0, 2, 1).contiguous().to(torch.float16),
            requires_grad=False,
        )

        # --- Repack qzeros ---
        # MoeWNA16 qzeros: [E, N//2, G] uint8 where N=output features, G=groups
        # RDNA2 qzeros:    [E, G, N//8] int32
        if hasattr(layer, "w13_qzeros") and layer.w13_qzeros.numel() > 0:
            w13_z_uint8 = layer.w13_qzeros  # [E, intermediate, hidden_size//group_size]
            E, N_half, G = w13_z_uint8.shape
            N_full = N_half * 2  # 2*intermediate

            unpacked_z = torch.zeros(E, N_full, G, dtype=torch.int32, device=device)
            unpacked_z[:, 0::2, :] = w13_z_uint8 & 0xF
            unpacked_z[:, 1::2, :] = (w13_z_uint8 >> 4) & 0xF

            # Transpose to [E, G, N] then pack N into int32
            unpacked_z = unpacked_z.permute(0, 2, 1)  # [E, G, N_full]
            unpacked_z = unpacked_z.reshape(E, G, N_full // 8, 8)
            w13_z_int32 = (unpacked_z << shifts).sum(dim=3, dtype=torch.int32)

            layer.w13_qzeros = torch.nn.Parameter(w13_z_int32, requires_grad=False)

        if hasattr(layer, "w2_qzeros") and layer.w2_qzeros.numel() > 0:
            # [E, hidden_size//2, intermediate//group_size]
            w2_z_uint8 = layer.w2_qzeros
            E, N_half, G = w2_z_uint8.shape
            N_full = N_half * 2  # hidden_size

            unpacked_z = torch.zeros(E, N_full, G, dtype=torch.int32, device=device)
            unpacked_z[:, 0::2, :] = w2_z_uint8 & 0xF
            unpacked_z[:, 1::2, :] = (w2_z_uint8 >> 4) & 0xF

            unpacked_z = unpacked_z.permute(0, 2, 1)  # [E, G, N_full]
            unpacked_z = unpacked_z.reshape(E, G, N_full // 8, 8)
            w2_z_int32 = (unpacked_z << shifts).sum(dim=3, dtype=torch.int32)

            layer.w2_qzeros = torch.nn.Parameter(w2_z_int32, requires_grad=False)

        logger.warning_once(
            "RDNA2 Experimental: weights repacked to RDNA2 kernel format "
            f"w13={layer.w13_qweight.shape} w2={layer.w2_qweight.shape} "
            f"scales_w13={layer.w13_scales.shape} scales_w2={layer.w2_scales.shape}"
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        from vllm.model_executor.layers.fused_moe.config import (
            int4_w4a16_moe_quant_config,
        )

        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if self.quant_config.zero_point else None,
            w2_zp=layer.w2_qzeros if self.quant_config.zero_point else None,
            block_shape=[0, layer.group_size],
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
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            has_zp=self.quant_config.zero_point,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
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
    has_zp: bool,
    shared_experts: SharedExperts | None,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA2 W4A16 AWQ HIP kernel."""
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    total_tokens = num_tokens * top_k
    dtype = hidden_states.dtype
    device = hidden_states.device

    N_gate_up = layer.w13_qweight.shape[2] * 8
    hidden_size = layer.w2_qweight.shape[2] * 8
    intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

    if global_num_experts <= 0:
        global_num_experts = layer.w13_qweight.shape[0]

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

    w13_qzeros = layer.w13_qzeros if has_zp else torch.empty(0, device=device)
    w2_qzeros = layer.w2_qzeros if has_zp else torch.empty(0, device=device)

    topk_w_float = topk_weights.view(-1).float()
    empty_tw = torch.empty(0, dtype=torch.float32, device=device)

    logger.warning_once(
        "RDNA2 Experimental: moe_awq_gemm_rdna2 w13 kernel "
        f"M={num_tokens} top_k={top_k} block_size_m={block_size_m} "
        f"weights={layer.w13_qweight.shape}"
    )
    ops.moe_awq_gemm_rdna2(
        hidden_states,
        w1_out,
        layer.w13_qweight,
        layer.w13_scales,
        w13_qzeros,
        topk_w_float if apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        apply_router_weight_on_input,
    )

    apply_moe_activation(activation, act_out, w1_out)

    logger.warning_once(
        "RDNA2 Experimental: moe_awq_gemm_rdna2 w2 kernel "
        f"M={num_tokens} top_k={top_k} block_size_m={block_size_m} "
        f"weights={layer.w2_qweight.shape}"
    )
    out = torch.zeros(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    ops.moe_awq_gemm_rdna2(
        act_out,
        out,
        layer.w2_qweight,
        layer.w2_scales,
        w2_qzeros,
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
