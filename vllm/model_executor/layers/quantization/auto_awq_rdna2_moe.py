# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AWQ INT4 (W4A16) routed-MoE method for the RDNA2 (gfx1030) HIP kernel.

Written BLIND (no runtime verification available at authoring time). This
module mirrors ``compressed_tensors/compressed_tensors_moe/
compressed_tensors_moe_wna16_rdna2.py`` (the existing GPTQ RDNA2 MoE
method) and reuses the exact same kernel contract and apply() flow:

  * ``ops.moe_gptq_gemm_rdna2`` two-call w1/w2 fused-MoE flow
    (``moe_align_block_size`` + ``silu_and_mul`` + fused top-k reduce via
    ``output_topk``), pre-zeroed output tensors, fp32 topk weights,
    ``block_size_m = 1`` for <= 4 tokens else ``4``.
  * Weight format after ``process_weights_after_loading``:
    packed int32 ``[E, K/8, N]`` with the exllama shuffle
    (``ops.gptq_shuffle`` with an empty ``g_idx``), scales
    ``[E, groups, N]`` fp16, packed zeros ``[E, groups, N/8]`` int32.

AWQ zero points are handled per the kernel decode contract in
``csrc/rocm/moe_q_gemm_rdna2.cu`` + ``q_gemm_rdna2_common.cuh``: the
kernel applies the GPTQv1 ``+1`` zero offset (``refresh_group`` passes
``zero_offset = 1`` into ``prep_zero_scale_fp16``), i.e. it dequantizes
as ``(q - (stored_zero + 1)) * scale``. AWQ dequantizes as
``(q - awq_zero) * scale``, so at load time we store
``(awq_zero - 1) & 0xF``. If the checkpoint has no qzeros
(``zero_point=False``), symmetric GPTQv1 zeros are synthesized
(effective zero 8 == uint4b8 bias => stored nibble 7).

This method is strictly env-gated: it is only returned by
``AutoAWQConfig.get_quant_method`` when ``VLLM_FORCE_RDNA2_W4A16_HIP``
is set, the platform is ROCm gfx10x, and the ``_rocm_C.moe_gptq_gemm_rdna2``
op is registered. Without that, AWQ MoE behavior (Marlin/WNA16) is
byte-identical to before. Dense AWQ is untouched.

Target workload: GLM-5.3-Flash routed experts (288 experts, hidden 4096,
moe_intermediate 2048, top-8, group_size 128), fp16 activations only.
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import (
        FusedMoEConfig,
        RoutedExperts,
        SharedExperts,
    )
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

logger = init_logger(__name__)

# AWQ uses a non-standard packing order within int32 values.
# For 4-bit: standard order stores values at bit positions [0,4,8,12,16,20,24,28]
# for indices [0,1,2,3,4,5,6,7], while AWQ stores them for indices
# [0,4,1,5,2,6,3,7]. This permutation reverses that ordering.
# (Same constant as auto_awq._REVERSE_AWQ_PACK_ORDER; duplicated here to keep
# this module import-cycle-free and self-contained.)
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

_SIZE_BITS = 4
_PACK_FACTOR = 32 // _SIZE_BITS  # 8 int4 values per int32
_MASK = (1 << _SIZE_BITS) - 1

# The MoE HIP kernel decodes zeros with the GPTQv1 +1 offset
# (zero_offset=1 in moe_q_gemm_rdna2.cu). Symmetric quant (no qzeros in the
# checkpoint) wants effective zero == uint4b8 bias (8), so store 8 - 1 = 7
# in every nibble.
_SYM_ZERO_STORED_NIBBLE = scalar_types.uint4b8.bias - 1
_SYM_ZERO_PACKED_WORD = sum(
    _SYM_ZERO_STORED_NIBBLE << (_SIZE_BITS * i) for i in range(_PACK_FACTOR)
)


def _awq_qweight_to_gptq(
    qw: torch.Tensor,
    reverse_order: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Convert one expert's AWQ qweight ``[K, N/8]`` (AWQ interleave, packed
    along the output dim) to standard GPTQ packing ``[K/8, N]`` (standard bit
    order, packed along the input dim).

    Adapted from ``auto_awq._convert_awq_to_standard_format`` for a single
    2-D expert slice.
    """
    K, N_packed = qw.shape
    N = N_packed * _PACK_FACTOR

    # Unpack int32 -> individual 4-bit values, fix AWQ ordering.
    unpacked = (qw.unsqueeze(-1) >> shifts) & _MASK  # (K, N/8, 8)
    unpacked = unpacked[:, :, reverse_order]
    unpacked = unpacked.reshape(K, N)  # (K, N)

    # Repack along the input dim (dim 0).
    unpacked = unpacked.reshape(K // _PACK_FACTOR, _PACK_FACTOR, N)
    return (unpacked.to(torch.int32) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32
    )


def _awq_qzeros_to_kernel_format(
    qz: torch.Tensor,
    reverse_order: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Convert one expert's AWQ qzeros ``[G, N/8]`` (AWQ interleave) into the
    kernel's packed format ``[G, N/8]`` (standard nibble order, GPTQv1 offset
    applied): stored nibble = ``(awq_zero - 1) & 0xF`` because the kernel
    decodes ``stored + 1``.
    """
    G, N_packed = qz.shape
    N = N_packed * _PACK_FACTOR

    unpacked = (qz.unsqueeze(-1) >> shifts) & _MASK  # (G, N/8, 8)
    unpacked = unpacked[:, :, reverse_order].reshape(G, N)

    stored = (unpacked - 1) & _MASK  # GPTQv1: kernel adds 1 back

    # Repack along the output dim: nibble i of word j holds column j*8 + i,
    # matching load4_zeros() in q_gemm_rdna2_common.cuh.
    stored = stored.reshape(G, N_packed, _PACK_FACTOR)
    return (stored.to(torch.int32) << shifts[None, None, :]).sum(
        dim=2, dtype=torch.int32
    )


class AutoAWQRDNA2MoEMethod(FusedMoEMethodBase):
    """AWQ W4A16 routed-MoE using the fused RDNA2 HIP kernel
    (``moe_gptq_gemm_rdna2``).

    Weights are loaded in the AWQ checkpoint layout (qweight packed along the
    output dim with the AWQ interleave), converted per expert to the kernel's
    GPTQ-shuffled ``[E, K/8, N]`` contract in ``process_weights_after_loading``,
    and executed with the same two-call w1/w2 flow as
    ``CompressedTensorsWNA16RDNA2MoEMethod``.

    Written blind; mirrors compressed_tensors_moe_wna16_rdna2.py. Env-gated via
    ``VLLM_FORCE_RDNA2_W4A16_HIP`` (see ``AutoAWQConfig.get_quant_method``).
    """

    def __init__(
        self,
        quant_config: "AutoAWQConfig",
        moe: "FusedMoEConfig",
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        if quant_config.weight_bits != 4:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod supports only 4-bit AWQ checkpoints, "
                f"got weight_bits={quant_config.weight_bits}."
            )
        group_size = quant_config.group_size
        if group_size == -1 or group_size < 32:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod requires group_size >= 32 "
                f"(GLM-5.3-Flash AWQ uses 128), got group_size={group_size}."
            )
        self.group_size = group_size
        self.pack_factor = quant_config.pack_factor  # 8 for 4-bit
        self.quant_type = scalar_types.uint4
        self.input_dtype = None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Not consumed by this method (no actorder g_idx); pop so it does not
        # leak onto parameter weight attrs (same as AutoAWQMoEMethod).
        extra_weight_attrs.pop("intermediate_size_full", None)
        extra_weight_attrs.update(
            {
                # AWQ checkpoints pack qweight along the OUTPUT dim, so the
                # sharded (intermediate) dim is dim-1 of the checkpoint
                # tensors: shard_dim must be flipped for w13/w2 loading.
                # NOTE: the loader only transposes the loaded tensor for a
                # hard-coded set of compressed-tensors method classes; this
                # class is intentionally NOT in that set — AWQ tensors are
                # stored in checkpoint orientation and converted in
                # process_weights_after_loading.
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        pack_factor = self.pack_factor
        group_size = self.group_size
        w13_num_shards = self.moe.w13_num_shards
        n_gate_up = w13_num_shards * intermediate_size_per_partition

        # --- Kernel-contract shape checks (TORCH-style runtime checks) ---
        if hidden_size % group_size != 0:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod: hidden_size "
                f"({hidden_size}) must be divisible by group_size "
                f"({group_size}) for w13 scales/zeros."
            )
        if intermediate_size_per_partition % group_size != 0:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod: intermediate_size_per_partition "
                f"({intermediate_size_per_partition}) must be divisible by "
                f"group_size ({group_size}) for w2 scales/zeros."
            )
        if hidden_size % _PACK_FACTOR != 0:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod: hidden_size "
                f"({hidden_size}) must be divisible by {_PACK_FACTOR} "
                "(K of w2 and packing of w13 must be whole int32 words)."
            )
        if n_gate_up % _PACK_FACTOR != 0:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod: gate+up output size "
                f"({n_gate_up}) must be divisible by {_PACK_FACTOR}."
            )
        if n_gate_up % 8 != 0 or hidden_size % 8 != 0:
            raise ValueError(
                "AutoAWQRDNA2MoEMethod: kernel requires N to be a multiple "
                f"of 8 (got gate+up={n_gate_up}, hidden={hidden_size})."
            )

        num_groups_w13 = hidden_size // group_size
        num_groups_w2 = intermediate_size_per_partition // group_size
        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        # --- Weights: AWQ checkpoint orientation ---
        # Per-expert checkpoint qweight is [K, N/8] (packed along N, AWQ
        # interleave); fused over experts and the gate/up shards:
        #   w13_qweight: [E, hidden, (w13_num_shards * inter) / pack]
        #   w2_qweight:  [E, inter, hidden / pack]
        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                n_gate_up // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # --- Scales: [E, groups, N] (loaded per gate/up shard on dim 1) ---
        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                n_gate_up,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # --- Zero points (AWQ stores them; if zero_point=False the
        # checkpoint has no qzeros and we synthesize in
        # process_weights_after_loading instead — do NOT register params the
        # checkpoint will not fill). ---
        if self.quant_config.zero_point:
            w13_qzeros = Parameter(
                torch.empty(
                    num_experts,
                    num_groups_w13,
                    n_gate_up // pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = Parameter(
                torch.empty(
                    num_experts,
                    num_groups_w2,
                    hidden_size // pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        """Per-expert AWQ -> standard (GPTQ) conversion + exllama shuffle +
        zero-point synthesis into the kernel's packed format."""
        w13_awq = layer.w13_qweight.data  # [E, K, N_gate_up/8]
        w2_awq = layer.w2_qweight.data  # [E, I, hidden/8]
        num_experts = w13_awq.shape[0]
        hidden_size = w13_awq.shape[1]
        n_gate_up = w13_awq.shape[2] * _PACK_FACTOR
        inter_size = w2_awq.shape[1]
        device = w13_awq.device

        shifts = torch.arange(0, 32, _SIZE_BITS, dtype=torch.int32, device=device)
        reverse_order = torch.tensor(
            _REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=device
        )
        empty_g_idx = torch.empty(0, dtype=torch.int32, device=device)

        # Target kernel layout: packed int32 [E, K/8, N].
        w13_packed = torch.empty(
            num_experts,
            hidden_size // _PACK_FACTOR,
            n_gate_up,
            dtype=torch.int32,
            device=device,
        )
        w2_packed = torch.empty(
            num_experts,
            inter_size // _PACK_FACTOR,
            hidden_size,
            dtype=torch.int32,
            device=device,
        )

        has_qzeros = self.quant_config.zero_point and hasattr(layer, "w13_qzeros")
        w13_qzeros_new = torch.empty(
            num_experts,
            layer.num_groups_w13,
            n_gate_up // _PACK_FACTOR,
            dtype=torch.int32,
            device=device,
        )
        w2_qzeros_new = torch.empty(
            num_experts,
            layer.num_groups_w2,
            hidden_size // _PACK_FACTOR,
            dtype=torch.int32,
            device=device,
        )

        # Convert one expert at a time to keep the unpack temporaries small
        # (288 experts x 4096x4096 would be several GB if batched).
        for e in range(num_experts):
            # --- w13 (fused gate+up) ---
            w13_e = _awq_qweight_to_gptq(
                w13_awq[e].contiguous(), reverse_order, shifts
            ).contiguous()
            ops.gptq_shuffle(w13_e, empty_g_idx, 4)
            w13_packed[e] = w13_e

            if has_qzeros:
                w13_qzeros_new[e] = _awq_qzeros_to_kernel_format(
                    layer.w13_qzeros.data[e].contiguous(), reverse_order, shifts
                )
            else:
                w13_qzeros_new[e] = _SYM_ZERO_PACKED_WORD

            # --- w2 (down) ---
            w2_e = _awq_qweight_to_gptq(
                w2_awq[e].contiguous(), reverse_order, shifts
            ).contiguous()
            ops.gptq_shuffle(w2_e, empty_g_idx, 4)
            w2_packed[e] = w2_e

            if has_qzeros:
                w2_qzeros_new[e] = _awq_qzeros_to_kernel_format(
                    layer.w2_qzeros.data[e].contiguous(), reverse_order, shifts
                )
            else:
                w2_qzeros_new[e] = _SYM_ZERO_PACKED_WORD

        # Scales: kernel requires dtype == activation dtype (fp16 on gfx1030).
        w13_scales = layer.w13_scales.data.to(torch.float16).contiguous()
        w2_scales = layer.w2_scales.data.to(torch.float16).contiguous()

        # Publish kernel-format parameters.
        layer.w13_weight_packed = Parameter(w13_packed, requires_grad=False)
        layer.w2_weight_packed = Parameter(w2_packed, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_scales, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_scales, requires_grad=False)
        layer.w13_qzeros = Parameter(w13_qzeros_new, requires_grad=False)
        layer.w2_qzeros = Parameter(w2_qzeros_new, requires_grad=False)

        # Free the AWQ-oriented tensors (replaced above).
        for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales"):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)

        # Pre-allocate reusable buffers for decode (sizes based on top_k=8),
        # mirroring CompressedTensorsWNA16RDNA2MoEMethod.
        act_dtype = torch.float16
        intermediate = n_gate_up // 2  # gated activation
        max_decode_tokens = 16
        top_k = 8  # conservative default
        buf_size = max_decode_tokens * top_k
        layer.rdna2_w1_buf = torch.zeros(
            buf_size, n_gate_up, dtype=act_dtype, device=device
        )
        layer.rdna2_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna2_out_buf = torch.zeros(
            max_decode_tokens, hidden_size, dtype=act_dtype, device=device
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

    def get_fused_moe_quant_config(self, layer: "RoutedExperts"):
        # This method executes via its own apply() (direct HIP dispatch), not
        # via the modular-kernel machinery; no FusedMoEQuantConfig is needed.
        return None

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        return _awq_rdna2_fused_moe(
            x,
            topk_weights,
            topk_ids,
            layer=layer,
            activation=activation,
            apply_router_weight_on_input=(layer.apply_router_weight_on_input),
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )


def _awq_rdna2_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: "RoutedExperts",
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA2 W4A16 HIP kernel — identical flow to
    ``compressed_tensors_moe_wna16_rdna2._rdna2_fused_moe``:

      - BLOCK_SIZE_M=1 for decode (small M), 4 otherwise
      - Pre-allocated buffers (no torch.zeros per call) where possible
      - moe_sum fused into the w2 output accumulation via output_topk
      - Pre-zeroed outputs (kernel accumulates via packed CAS atomics)
    """
    # gfx1030 kernels are fp16-only (v_dot2_f32_f16); coerce defensively.
    if hidden_states.dtype != torch.float16:
        logger.warning_once(
            "AutoAWQRDNA2MoEMethod requires fp16 activations; coercing %s "
            "to float16.",
            hidden_states.dtype,
        )
        hidden_states = hidden_states.to(torch.float16)

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

    # BLOCK_SIZE_M=1 for decode (small M), 4 for prefill
    # (kernel supports {1, 2, 4, 8}).
    block_size_m = 1 if num_tokens <= 4 else 4
    if block_size_m not in (1, 2, 4, 8):
        raise ValueError(
            f"moe_gptq_gemm_rdna2: block_size_m must be one of 1/2/4/8, "
            f"got {block_size_m}."
        )

    # --- Token routing ---
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
    )

    # --- Reuse pre-allocated buffers when possible ---
    if total_tokens <= layer.rdna2_w1_buf.shape[0]:
        w1_out = layer.rdna2_w1_buf[:total_tokens]
        w1_out.zero_()
        act_out = layer.rdna2_act_buf[:total_tokens]
    else:
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

    # --- topk weights (pre-cast to float32 for kernel) ---
    topk_w_float = topk_weights.view(-1).float()
    empty_tw = layer.rdna2_empty_tw

    # --- w1 GEMM: [M, K] -> [M*top_k, N_gate_up] ---
    ops.moe_gptq_gemm_rdna2(
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

    # --- Activation (silu_and_mul etc.) ---
    apply_moe_activation(activation, act_out, w1_out)

    # --- w2 GEMM: [M*top_k, intermediate] -> [M, hidden] (fused reduce) ---
    # output_topk=top_k: kernel writes to out[token_id / top_k] directly,
    # fusing moe_sum into the atomic accumulation — saves one kernel launch
    # and the w2_out intermediate buffer.
    out = torch.zeros(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    ops.moe_gptq_gemm_rdna2(
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
