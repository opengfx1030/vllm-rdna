# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE W4A16 experts on the RDNA3 (gfx1100) HIP kernel.

``moe_gptq_gemm_rdna3`` is a single HIP kernel launch per GEMM that handles
expert routing + W4A16 dequant + dot product with atomic output accumulation.
``VLLM_RDNA3_MOE_DETERMINISTIC=1`` accumulates in Q32 fixed point instead of
packed fp16 CAS atomics, which makes the result run-to-run bit-identical.
Prefill-sized batches (``VLLM_RDNA3_MOE_WMMA=1``, at least
``VLLM_RDNA3_MOE_WMMA_MIN_M`` tokens) use the WMMA grouped GEMM
``moe_gptq_gemm_rdna3_wmma`` instead; it is deterministic by construction.

Weight format (per expert, same as the dense RDNA3 W4A16 kernel):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` in activation dtype
  - Zero points ``[E, groups, N/8]`` packed int32 (synthesized, symmetric only)
"""

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Token-block sizes moe_gptq_gemm_rdna3 is instantiated for.
_BLOCK_SIZES_M = (1, 2, 4, 8)
# Token-block (WMMA M tile) sizes moe_gptq_gemm_rdna3_wmma is instantiated for.
_WMMA_BLOCK_SIZES_M = (16, 32, 64)


def _block_size_m_override() -> int:
    """``VLLM_RDNA3_MOE_BLOCK_M``: 0 (automatic) or a kernel block size."""
    block_size_m = envs.VLLM_RDNA3_MOE_BLOCK_M
    if block_size_m != 0 and block_size_m not in _BLOCK_SIZES_M:
        raise ValueError(
            "VLLM_RDNA3_MOE_BLOCK_M must be 0 (automatic) or one of "
            f"{_BLOCK_SIZES_M}, got {block_size_m}"
        )
    return block_size_m


def _wmma_block_size_m_override() -> int:
    """``VLLM_RDNA3_MOE_WMMA_BLOCK_M``: 0 (automatic) or a WMMA tile height."""
    block_size_m = envs.VLLM_RDNA3_MOE_WMMA_BLOCK_M
    if block_size_m != 0 and block_size_m not in _WMMA_BLOCK_SIZES_M:
        raise ValueError(
            "VLLM_RDNA3_MOE_WMMA_BLOCK_M must be 0 (automatic) or one of "
            f"{_WMMA_BLOCK_SIZES_M}, got {block_size_m}"
        )
    return block_size_m


def _q32_scratch(destination: torch.Tensor) -> torch.Tensor:
    """Zeroed int64 Q32 accumulator for a ``[rows, N]`` fp16 destination.

    The extra last column is the sticky row status.
    """
    rows, columns = destination.shape
    return torch.zeros(rows, columns + 1, dtype=torch.int64, device=destination.device)


def rdna3_moe_kernel_available() -> bool:
    """Whether the fused RDNA3 MoE HIP kernel is built into this binary."""
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_gfx1100

    return (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")
    )


def rdna3_moe_wmma_available() -> bool:
    """Whether the WMMA MoE prefill GEMM is built into this binary."""
    return rdna3_moe_kernel_available() and hasattr(
        torch.ops._rocm_C, "moe_gptq_gemm_rdna3_wmma"
    )


class Rdna3WNA16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts backed by ``moe_gptq_gemm_rdna3`` (gfx1100).

    By default both GEMMs accumulate with packed fp16 CAS atomics, so their
    destinations are zeroed first and every partial sum is rounded to fp16 in
    arrival order. With ``VLLM_RDNA3_MOE_DETERMINISTIC=1`` they accumulate into
    zeroed int64 Q32 fixed-point scratch instead (exact, order-independent
    integer adds) and ``moe_fixedpoint_to_half_rdna3`` rounds each result to
    fp16 once.

    With ``VLLM_RDNA3_MOE_WMMA=1`` batches of at least
    ``VLLM_RDNA3_MOE_WMMA_MIN_M`` tokens run both GEMMs on
    ``moe_gptq_gemm_rdna3_wmma`` (S1b): GEMM 1 stores each scheduled pair row
    once, GEMM 2 reduces the top-k dimension through Q32 scratch, so that path
    is bit-deterministic whatever ``VLLM_RDNA3_MOE_DETERMINISTIC`` says.
    Smaller batches (decode, speculative verify) stay on the scalar kernel.

    The second GEMM is given ``output_topk``, which makes it reduce over the
    top-k dimension while it accumulates — the ``moe_sum`` launch and the
    ``[M * top_k, K]`` intermediate are fused away, hence
    ``TopKWeightAndReduceNoOP``.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16, "Supports only int4_w4a16"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._empty_topk_weights: torch.Tensor | None = None
        # Resolved once: a captured graph replays the path it recorded.
        self._deterministic = envs.VLLM_RDNA3_MOE_DETERMINISTIC
        self._block_size_m_override = _block_size_m_override()
        self._wmma = envs.VLLM_RDNA3_MOE_WMMA
        self._wmma_min_m = envs.VLLM_RDNA3_MOE_WMMA_MIN_M
        self._wmma_block_size_m_override = _wmma_block_size_m_override()
        if self._wmma:
            if not rdna3_moe_wmma_available():
                raise RuntimeError(
                    "VLLM_RDNA3_MOE_WMMA=1 but this build has no "
                    "moe_gptq_gemm_rdna3_wmma (needs a gfx1100 build with S1b)"
                )
            logger.info_once(
                "RDNA3 W4A16 MoE: WMMA grouped GEMM for batches of >= %d tokens "
                "(VLLM_RDNA3_MOE_WMMA=1), scalar kernel below",
                self._wmma_min_m,
            )
        if self._deterministic:
            logger.info_once(
                "RDNA3 W4A16 MoE: Q32 fixed-point accumulation "
                "(VLLM_RDNA3_MOE_DETERMINISTIC=1)"
            )

    def _block_size_m(self, num_tokens: int) -> int:
        """1 for decode (no padding waste), 4 for prefill, unless overridden."""
        return self._block_size_m_override or (1 if num_tokens <= 4 else 4)

    def _use_wmma(self, num_tokens: int, dtype: torch.dtype) -> bool:
        """Whether this batch runs on the WMMA GEMM (enabled, fp16, big enough)."""
        if not self._wmma:
            return False
        if dtype != torch.float16:
            raise ValueError(
                "VLLM_RDNA3_MOE_WMMA=1 needs fp16 activations (the top-k "
                f"reduction is Q32, fp16-only), got {dtype}"
            )
        return num_tokens >= self._wmma_min_m

    def _wmma_block_size_m(self, num_tokens: int, top_k: int, num_experts: int) -> int:
        """WMMA M tile from the mean pair rows per expert (GPU-tunable guess).

        A tile is padded per expert, so a tall tile on a sparse expert wastes
        MACs while a short one re-reads the expert weights once per tile.
        """
        if self._wmma_block_size_m_override:
            return self._wmma_block_size_m_override
        rows_per_expert = num_tokens * top_k / max(num_experts, 1)
        if rows_per_expert < 20:
            return 16
        return 32 if rows_per_expert < 80 else 64

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna3_moe_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Symmetric int4 weights only: the kernel consumes synthesized zero
        # points and has no path for checkpoint zero points.
        return activation_key is None and weight_key in (kInt4Static, kInt4Static32)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Weights are packed K-first: w1 is [E, K // 8, w13_shards * N] and
        # w2 is [E, N // 8, K], so neither N nor K can be read off a trailing
        # dimension the way the base implementation does.
        assert w1.dim() == 3 and w2.dim() == 3
        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        return w1.size(0), a1.size(0), w2.size(1) * 8, a1.size(-1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The modular kernel provisions the output buffer out of workspace13,
        # so both GEMM-1 buffers have to live in workspace2 — the second GEMM
        # reads the activation output while accumulating into the output.
        gate_up = N if not activation.is_gated else 2 * N
        return ((M, K), (M * topk * (gate_up + N),), (M, K))

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
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.w1_zp is not None and self.w2_zp is not None

        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E
        if self._deterministic and hidden_states.dtype != torch.float16:
            raise ValueError(
                "VLLM_RDNA3_MOE_DETERMINISTIC=1 needs fp16 activations (Q32 "
                f"accumulation is fp16-only), got {hidden_states.dtype}"
            )
        use_wmma = self._use_wmma(M, hidden_states.dtype)
        if expert_map is not None:
            # The align kernel reads the map through a raw int32 pointer.
            assert expert_map.dtype == torch.int32 and expert_map.is_contiguous()
            assert expert_map.numel() >= global_num_experts

        gate_up = w1.size(2)
        act_n = self.adjust_N_for_activation(gate_up, activation)
        rows = M * top_k

        scratch = workspace2.view(-1)
        gate_up_out = scratch[: rows * gate_up].view(rows, gate_up)
        act_out = scratch[rows * gate_up : rows * (gate_up + act_n)].view(rows, act_n)

        # Masked ids (-1) and experts absent from expert_map (slot -1) are
        # dropped from counting and ranking, so no block is scheduled for them.
        # num_experts is the global count: the kernel range-checks the global
        # ids before mapping them to local slots.
        block_size_m = (
            self._wmma_block_size_m(M, top_k, global_num_experts)
            if use_wmma
            else self._block_size_m(M)
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        if (
            self._empty_topk_weights is None
            or self._empty_topk_weights.device != hidden_states.device
        ):
            self._empty_topk_weights = torch.empty(
                0, dtype=torch.float32, device=hidden_states.device
            )
        topk_weights_f32 = topk_weights.reshape(-1).float()
        no_topk_weights = self._empty_topk_weights

        if use_wmma:
            # GEMM 1 stores every scheduled pair row exactly once: no zeroing
            # and no atomics. Rows of dropped experts are never read by GEMM 2.
            ops.moe_gptq_gemm_rdna3_wmma(
                hidden_states,
                gate_up_out,
                w1,
                self.w1_scale,
                self.w1_zp,
                topk_weights_f32 if apply_router_weight_on_input else no_topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                apply_router_weight_on_input,
            )
            self.activation(activation, act_out, gate_up_out)
            output_q32 = _q32_scratch(output)
            ops.moe_gptq_gemm_rdna3_wmma(
                act_out,
                output_q32,
                w2,
                self.w2_scale,
                self.w2_zp,
                no_topk_weights if apply_router_weight_on_input else topk_weights_f32,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                1,
                block_size_m,
                not apply_router_weight_on_input,
                output_topk=top_k,
            )
            ops.moe_fixedpoint_to_half_rdna3(output_q32, output)
            return

        gate_up_target = (
            _q32_scratch(gate_up_out) if self._deterministic else gate_up_out.zero_()
        )
        ops.moe_gptq_gemm_rdna3(
            hidden_states,
            gate_up_target,
            w1,
            self.w1_scale,
            self.w1_zp,
            topk_weights_f32 if apply_router_weight_on_input else no_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
        )
        if self._deterministic:
            ops.moe_fixedpoint_to_half_rdna3(gate_up_target, gate_up_out)

        self.activation(activation, act_out, gate_up_out)

        # output_topk=top_k makes the kernel accumulate into out[token_id],
        # fusing the top-k reduction into the atomic write-back.
        output_target = _q32_scratch(output) if self._deterministic else output.zero_()
        ops.moe_gptq_gemm_rdna3(
            act_out,
            output_target,
            w2,
            self.w2_scale,
            self.w2_zp,
            no_topk_weights if apply_router_weight_on_input else topk_weights_f32,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            output_topk=top_k,
        )
        if self._deterministic:
            ops.moe_fixedpoint_to_half_rdna3(output_target, output)
