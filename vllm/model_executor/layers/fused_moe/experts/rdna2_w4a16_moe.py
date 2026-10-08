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

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    TopKWeightAndReduce,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

logger = init_logger(__name__)

# W4A8 (int4 weights, int8 activations) MoE opt-in. Default OFF; resolved once
# at weight-load time so the traced forward only reads a Python bool.
W4A8_MOE_ENV_VAR = "VLLM_RDNA2_W4A8_SDOT4"

# moe_gptq_gemm_rdna2 supports block_size_m in {1, 2, 4, 8} (the kernel's
# TORCH_CHECK). The pre-allocated routing buffers must be large enough for the
# worst case, so size them for the kernel's maximum.
_MAX_BLOCK_SIZE_M = 8

# Overflow tile width for the MoE prefill path. The decode-shaped default caps
# block_size_m at 4; for large token counts the kernel's 8-row tile amortises the
# per-block overhead over twice the work. Set to 0 to keep the decode default.
_BLOCK_SIZE_M_LARGE = int(os.environ.get("VLLM_RDNA2_MOE_BSM_LARGE", "8"))
_BLOCK_SIZE_M_MIN_TOKENS = int(os.environ.get("VLLM_RDNA2_MOE_BSM_MIN_TOKENS", "512"))


def _resolve_block_size_m(num_tokens: int) -> int:
    if num_tokens <= 4:
        return 1
    if _BLOCK_SIZE_M_LARGE > 4 and num_tokens >= _BLOCK_SIZE_M_MIN_TOKENS:
        return min(_BLOCK_SIZE_M_LARGE, _MAX_BLOCK_SIZE_M)
    return 4


def _w4a8_moe_ops_built() -> bool:
    """True when the gfx1030 MoE W4A8 op is registered.

    Probe by enumerating the registered schemas: ``dir(torch.ops._rocm_C)``
    only reports ``name`` for the namespace object.
    """
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    try:
        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("moe_w4a8_gemm_rdna2" in str(s) for s in schemas)


def resolve_w4a8_moe() -> bool:
    """Resolve the W4A8 MoE opt-in once, at weight-load time.

    The resident MoE layouts own the same weight buffers with a different
    packing, so W4A8 stays hard-off whenever a resident variant is active.
    """
    if envs.VLLM_RDNA_MOE_RESIDENT or envs.VLLM_RDNA_MOE_RESIDENT_SKINNY:
        return False
    return os.environ.get(W4A8_MOE_ENV_VAR) == "1" and _w4a8_moe_ops_built()


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

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        # apply() multiplies by the top-k weights and reduces over top-k in
        # the w2 GEMM (output_topk), so nothing is left for the finalize step.
        return TopKWeightAndReduceNoOP()

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
        workspace1 = (M * topk, N)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.w13_weight_scale = layer.w13_weight_scale
        self.w13_qzeros = getattr(layer, "w13_qzeros", None)
        if self.w13_qzeros is None:
            self.w13_qzeros = getattr(layer, "w13_weight_scale_zeros", None)
        self.w2_weight_scale = layer.w2_weight_scale
        self.w2_qzeros = getattr(layer, "w2_qzeros", None)
        if self.w2_qzeros is None:
            self.w2_qzeros = getattr(layer, "w2_weight_scale_zeros", None)
        device = layer.w13_weight_scale.device
        self._empty_tw = torch.empty(0, device=device)
        self._topk_w_buf = torch.empty(
            layer.moe_config.max_num_tokens * layer.top_k,
            dtype=torch.float32,
            device=device,
        )
        max_tokens = layer.moe_config.max_num_tokens * layer.top_k
        max_padded = max_tokens + layer.moe_config.num_experts * (_MAX_BLOCK_SIZE_M - 1)
        self._sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        self._expert_ids = torch.empty(
            (max_padded + _MAX_BLOCK_SIZE_M - 1) // _MAX_BLOCK_SIZE_M,
            dtype=torch.int32,
            device=device,
        )
        self._num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
        self._w4a8 = resolve_w4a8_moe()

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
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        local_num_experts = w1.shape[0]
        if global_num_experts <= 0:
            global_num_experts = local_num_experts

        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        N_gate_up = w1.shape[2]

        block_size_m = _resolve_block_size_m(num_tokens)

        # The modular-kernel path never calls process_weights_after_loading
        # here: scales and zero points come from the FusedMoEQuantConfig
        # (set by the WNA16 oracle's RDNA2 conversion), routing buffers are
        # per call.
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            local_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        w13_scales = self.w1_scale
        w13_qzeros = self.w1_zp
        w2_scales = self.w2_scale
        w2_qzeros = self.w2_zp
        if not hasattr(self, "_w4a8"):
            self._w4a8 = resolve_w4a8_moe()

        total_tokens = num_tokens * top_k
        if total_tokens <= workspace13.shape[0] and N_gate_up <= workspace13.shape[1]:
            w1_out = workspace13[:total_tokens, :N_gate_up]
            w1_out.zero_()
        else:
            w1_out = torch.zeros(
                total_tokens,
                N_gate_up,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        topk_w_buf = topk_weights.reshape(-1).float()
        empty_tw = topk_w_buf.new_empty(0)
        if getattr(self, "_w4a8", False):
            ops.moe_w4a8_gemm_rdna2(
                hidden_states,
                w1_out,
                w1,
                w13_scales,
                w13_qzeros,
                topk_w_buf if apply_router_weight_on_input else empty_tw,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                False,
                0,
                False,
            )
        else:
            ops.moe_gptq_gemm_rdna2(
                hidden_states,
                w1_out,
                w1,
                w13_scales,
                w13_qzeros,
                topk_w_buf if apply_router_weight_on_input else empty_tw,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                False,
                0,
            )

        if activation == MoEActivation.SILU:
            activated = _swiglu_split(w1_out)
        else:
            activated = w1_out

        out_buf = output
        if out_buf.dtype != torch.float16:
            out_buf = torch.empty(
                output.shape, dtype=torch.float16, device=output.device
            )
        out_buf.zero_()
        if getattr(self, "_w4a8", False):
            ops.moe_w4a8_gemm_rdna2(
                activated,
                out_buf,
                w2,
                w2_scales,
                w2_qzeros,
                topk_w_buf,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                1,
                block_size_m,
                True,
                top_k,
                False,
            )
        else:
            ops.moe_gptq_gemm_rdna2(
                activated,
                out_buf,
                w2,
                w2_scales,
                w2_qzeros,
                topk_w_buf,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                1,
                block_size_m,
                True,
                top_k,
            )
        if out_buf is not output:
            output.copy_(out_buf.to(output.dtype))
