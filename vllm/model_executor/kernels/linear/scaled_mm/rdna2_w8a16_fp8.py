# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A16-FP8 dense linear kernel for AMD RDNA2 (gfx1030).

FP8 (E4M3) weights + fp16 activations, per-tile FP8 -> fp16 conversion
via a 256-entry constant-memory LUT (gfx1030 has no FP8 hardware), then
v_dot2_f32_f16 for the inner dot. fp32 accumulator, packed CAS-64
atomic-write epilogue.

Weight storage: float8_e4m3fn [K, N] as uint8 (1 byte per element).
Scale storage: fp16 [groups, N] (block_structure=[N_block, K_block]).
Activations: fp16 [M, K].

gfx1030 lacks v_dot2_f32_bf16 (RDNA3+ feature), so this kernel rejects
bf16 activations at can_implement time. bf16 checkpoints must be cast
to fp16 (or quantized to W4A16 which already has bf16 disabled).

The kernel is registered in _POSSIBLE_WFP8A16_KERNELS[ROCM] -- but only
fires if the C++ op is registered (rebuild required). Without the
rebuild, falls through to the next kernel in the registry (no-op since
no other RDNA2 FP8 kernels exist; the model will fail to load rather
than silently use wrong math).
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class RDNA2W8A16FP8LinearKernel(FP8ScaledMMLinearKernel):
    SUPPORTED_WEIGHT_DTYPES = [scalar_types.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx10x() in can_implement.
        return 60

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W8A16 FP8 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 W8A16 FP8 kernel requires gfx1030"

        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gemm_w8a16_fp8_dense")
        ):
            return (
                False,
                "torch.ops._rocm_C.gemm_w8a16_fp8_dense missing -- rebuild "
                "C++ extension (CUTLASS blocker; see CUTLASS_BLOCKER.md)",
            )

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        supported, reason = cls.is_supported()
        if not supported:
            return False, reason

        # Act type: fp16 only (gfx1030 has no v_dot2_f32_bf16).
        if c.input_dtype != torch.float16:
            return (
                False,
                f"RDNA2 W8A16 FP8 kernel only supports fp16 activations on "
                f"gfx1030 (got {c.input_dtype})",
            )

        # Out dtype: fp16 only (atomic_add_pk4_f16 emits packed CAS on fp16).
        if c.out_dtype != torch.float16:
            return (
                False,
                f"RDNA2 W8A16 FP8 kernel only supports fp16 output on gfx1030 "
                f"(got {c.out_dtype})",
            )

        # Block-strategy weight quant: row = N-block, col = K-block.
        # We need a positive K-axis block (col > 0) since the kernel
        # uses col as group_size. Per-tensor (row=-1,col=-1) and
        # per-channel (row=-1,col=1) and per-token (row=1,col=-1) do
        # not match.
        ws = c.weight_quant_key.scale.group_shape
        if not (ws.row > 0 and ws.col > 0):
            return (
                False,
                "RDNA2 W8A16 FP8 kernel requires block-quant weight scales "
                f"with positive N-block and K-block (got group_shape={ws})",
            )

        # Activation quant: static per-tensor only (no dynamic quant on gfx1030).
        if not c.activation_quant_key.scale.group_shape.is_per_tensor():
            return (
                False,
                "RDNA2 W8A16 FP8 kernel only handles static per-tensor "
                "activation scales (no dynamic per-token act quant on gfx1030)",
            )

        # Symmetric quant only (kernel assumes zero=null).
        if not c.weight_quant_key.symmetric:
            return (
                False,
                "RDNA2 W8A16 FP8 kernel only supports symmetric weight quant "
                f"(asymmetric={not c.weight_quant_key.symmetric})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # One-time weight prep. Without it, apply_scaled_mm re-transposes
        # the (N, K) fp8 weight and re-expands the block scales on EVERY
        # forward (the BLOCK strategy skips the scheme's (K, N)
        # canonicalization). Those transposed-contiguous copies run as
        # pathological fp8 byte-copies (~7ms for large projections at
        # decode, ~38% of decode GPU time by profiling). Materializing
        # (K, N) weights and [K_groups, N] fp16 scales here turns every
        # per-forward fallback in apply_scaled_mm into a no-op.
        if getattr(layer, "_rdna2_w8a16_fp8_prepared", False):
            return
        w_name, w_s_name = self.layer_param_names[0], self.layer_param_names[1]
        N_out, K = self.config.weight_shape

        w = getattr(layer, w_name)
        if w.shape != (K, N_out):
            replace_parameter(layer, w_name, w.t().contiguous())
            prepared = getattr(layer, w_name)
            prepared.input_dim = 0
            prepared.output_dim = 1

        ws = getattr(layer, w_s_name)
        if ws.dim() == 2:
            grp = self.config.weight_quant_key.scale.group_shape
            n_groups = N_out // grp.row
            k_groups = K // grp.col
            if ws.shape == (n_groups, k_groups):
                # [N_groups, K_groups] -> [K_groups, N] (kernel layout)
                ws_prepared = ws.t().contiguous()
                ws_prepared = ws_prepared.repeat_interleave(grp.row, dim=1)
                replace_parameter(
                    layer, w_s_name, ws_prepared.to(torch.float16)
                )
        layer._rdna2_w8a16_fp8_prepared = True

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        if A.dtype == torch.float8_e4m3fn:
            A = A.to(torch.float16)
        assert A.dtype == torch.float16, (
            f"RDNA2 W8A16 FP8 kernel: A must be fp16 or fp8 (got {A.dtype})"
        )
        assert B.dtype == torch.float8_e4m3fn, (
            f"RDNA2 W8A16 FP8 kernel: B must be fp8 e4m3fn (got {B.dtype})"
        )
        if Bs.dtype != torch.float16:
            Bs = Bs.to(torch.float16)

        M, K = A.shape
        B_in, N = B.shape
        # vLLM stores linear weights as (out_features, in_features).
        # C++ kernel expects (K, N).
        if B_in != K:
            B = B.t().contiguous()
            # Match the transpose for the block-quant scale. CT stores
            # scales as [N_groups, K_groups] matching weight layout;
            # C++ kernel expects [K_groups, N].
            if Bs.dim() == 2 and Bs.shape[0] != K // 128:
                Bs = Bs.t().contiguous()
        K_b, N = B.shape
        assert K == K_b, f"K mismatch: A={K}, B={K_b}"

        b_bytes = B.view(torch.uint8)

        # output_shape was computed by the caller from pre-transpose w
        # as [*x.shape[:-1], w.shape[1]]; replace last dim with N.
        output_shape = [*output_shape[:-1], N]

        # C++ kernel reads b_scales[group * N + n], expecting shape
        # [K_groups, N]. The compressed-tensors BLOCK scheme stores
        # [K_groups, N_blocks] -- one scale per (K_block, N_block) cell.
        # Broadcast to [K_groups, N] by repeating each block's value
        # block_n times.
        if Bs.dim() == 2 and Bs.shape[0] > 1 and Bs.shape[1] != N:
            # Block-quant scale: [K_groups, N_blocks] -> broadcast -> [K_groups, N]
            block_n = N // Bs.shape[1]
            Bs = Bs.repeat_interleave(block_n, dim=1).contiguous()
        # Now Bs should be [K_groups, N]
        K_groups = Bs.shape[0]
        assert K % K_groups == 0, (
            f"K={K} not divisible by K_groups={K_groups}"
        )
        group_size = K // K_groups

        output = torch.zeros((M, N), dtype=out_dtype, device=A.device)

        ops.gemm_w8a16_fp8_dense(A, b_bytes, Bs, output, group_size)

        if bias is not None:
            output.add_(bias)

        return output.view(*output_shape)
