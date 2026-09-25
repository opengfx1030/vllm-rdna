# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A4 MXFP4 using the fused RDNA2 (gfx1030) HIP kernel.

Bridge between the parent ``CompressedTensorsW4A4Mxfp4MoEMethod`` (which
allocates weights in the standard CUTLASS/Marlin ``[E, N, K/2]`` layout)
and the RDNA2 native kernel (which expects ``[E, K/8, N]`` uint32, K-major
for V_DOT2_F32_F16 coalescing).

``process_weights_after_loading`` transposes the weights and scales in-place
to the kernel's native layout. The transpose is a one-time amortized cost
on model load; the kernel's compute path is then optimal for every token.

Weight format (per expert, after RDNA2 transpose):
  - Packed E2M1 ``[E, K/8, N]`` uint32 (8 nibbles per word, LSB-first)
  - Scales ``[E, K/32, N]`` uint8 (UE8M0, one byte per 32-element K-block)

No zero point — E2M1 is signed-magnitude so the high nibble encodes sign.
fp16-only (gfx1030 lacks v_dot2_f32_bf16).
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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)

logger = init_logger(__name__)


class CompressedTensorsW4A4Mxfp4RDNA2MoEMethod(
    CompressedTensorsW4A4Mxfp4MoEMethod
):
    """W4A4 MXFP4 MoE using the fused RDNA2 HIP kernel (moe_mxfp4_gemm_rdna2).

    Inherits ``create_weights`` from the parent (which allocates the
    canonical ``[E, N, K/2]`` MXFP4 layout). Overrides
    ``process_weights_after_loading`` to transpose into the kernel's
    ``[E, K/8, N]`` native layout and overrides ``apply`` to dispatch
    through the fused HIP kernel.
    """

    def __init__(self, moe):
        super().__init__(moe)
        # Disable parent CUTLASS/Marlin selection — we use the RDNA2 kernel.
        self.use_cutlass_mxfp4 = False
        self.mxfp4_backend = None
        self.experts_cls = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight_packed.device
        num_experts = layer.w13_weight_packed.shape[0]

        # Parent's create_weights lays out:
        #   w13_weight_packed: [E, 2*intermediate, hidden_size//2]       uint8
        #   w13_weight_scale:  [E, 2*intermediate, hidden_size//32]      uint8
        #   w2_weight_packed:  [E, hidden_size, intermediate//2]         uint8
        #   w2_weight_scale:   [E, hidden_size, intermediate//32]        uint8
        # Kernel expects [E, K/8, N] (K-major for V_DOT2 coalescing).
        # Transpose dims (1, 2) to swap N and K onto the kernel's layout.
        w13_packed_T = layer.w13_weight_packed.data.transpose(1, 2).contiguous()
        w13_scale_T = (
            layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        )
        w2_packed_T = layer.w2_weight_packed.data.transpose(1, 2).contiguous()
        w2_scale_T = layer.w2_weight_scale.data.transpose(1, 2).contiguous()

        layer.w13_weight_packed = torch.nn.Parameter(
            w13_packed_T, requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale_T, requires_grad=False
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            w2_packed_T, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale_T, requires_grad=False
        )

        # Kernel-readable dims (post-transpose):
        # w13: [E, hidden_size//2, 2*intermediate]     — K=hidden, N=2*inter
        # w2:  [E, intermediate//2, hidden_size]       — K=inter, N=hidden
        N_gate_up = w13_packed_T.shape[2]
        hidden_size = w2_packed_T.shape[2]
        intermediate = N_gate_up // 2 if N_gate_up > 0 else 0

        # Pre-allocate buffers for decode (BLOCK_SIZE_M=1, top_k=8).
        max_decode_tokens = 16
        top_k = 8
        buf_size = max_decode_tokens * top_k
        act_dtype = torch.float16
        layer.rdna2_w1_buf = torch.zeros(
            buf_size, N_gate_up, dtype=act_dtype, device=device
        )
        layer.rdna2_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna2_out_buf = torch.zeros(
            max_decode_tokens, hidden_size, dtype=act_dtype, device=device
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

        # Parent's apply() asserts self.moe_kernel is not None. We override
        # apply() so this never fires — set to None explicitly so any
        # future caller that bypasses our override fails loudly.
        self.moe_kernel = None

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
        return _rdna2_fused_mxfp4_moe(
            x,
            topk_weights,
            topk_ids,
            layer=layer,
            activation=activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )


def _rdna2_fused_mxfp4_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: RoutedExperts,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA2 W4A4 MXFP4 HIP kernel.

    fp16 only — bf16 is not supported on gfx1030 (no v_dot2_f32_bf16).
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    total_tokens = num_tokens * top_k
    N_gate_up = layer.w13_weight_packed.shape[2]
    hidden_size = layer.w2_weight_packed.shape[2]
    dtype = hidden_states.dtype
    device = hidden_states.device

    intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

    if global_num_experts <= 0:
        global_num_experts = layer.w13_weight_packed.shape[0]

    block_size_m = 1 if num_tokens <= 4 else 4

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts, expert_map
    )

    if total_tokens <= layer.rdna2_w1_buf.shape[0]:
        w1_out = layer.rdna2_w1_buf[:total_tokens]
        w1_out.zero_()
        act_out = layer.rdna2_act_buf[:total_tokens]
    else:
        w1_out = torch.zeros(
            total_tokens, N_gate_up, dtype=dtype, device=device
        )
        act_out = torch.empty(
            total_tokens, intermediate_size, dtype=dtype, device=device
        )

    topk_w_float = topk_weights.view(-1).float()
    empty_tw = layer.rdna2_empty_tw

    # w1 GEMM: topk_weight on input if apply_router_weight_on_input=True.
    ops.moe_mxfp4_gemm_rdna2(
        hidden_states,
        w1_out,
        layer.w13_weight_packed,
        layer.w13_weight_scale,
        topk_w_float if apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        apply_router_weight_on_input,
    )

    apply_moe_activation(activation, act_out, w1_out)

    # w2 GEMM: topk_weight on output if apply_router_weight_on_input=False.
    # output_topk=top_k fuses moe_sum into the atomic accumulation.
    out = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    ops.moe_mxfp4_gemm_rdna2(
        act_out,
        out,
        layer.w2_weight_packed,
        layer.w2_weight_scale,
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
