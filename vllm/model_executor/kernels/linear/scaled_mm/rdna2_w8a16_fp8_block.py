# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A16-FP8 dense linear kernel for AMD RDNA2 (gfx1030), BLOCK-scheme variant.

Reuses the C++ op `gemm_w8a16_fp8_dense` (FP16 activations + FP8 weights +
block weight scales, v_dot2_f32_f16 inner loop). This wrapper is for the
compressed-tensors BLOCK scheme (`CompressedTensorsW8A8Fp8` with
`is_static_input_scheme=False`), which by default would call
`quant_fp8` to quantize FP16 -> FP8 per-token then call a Triton FP8
block-scaled MM kernel. The Triton JIT path fails on gfx1030 (Wave32,
no FP8 hardware).

Strategy: register this kernel with `apply_input_quant=False` so the
upstream `Fp8BlockScaledMMLinearKernel.apply_weights` skips the FP8
activation quant and passes the original FP16 `x` to `apply_block_scaled_mm`.
We then call the existing FP16-input C++ op directly, ignoring the FP8
`A` and per-token `As` placeholder tensors.

gfx1030 has no FP8 hardware, so quantizing FP16 -> FP8 only to dequant
back to FP16 in the kernel is wasted work. Skipping the act quant is a
strict win on gfx1030 (same compute, half the activation bandwidth).
"""

import torch

from typing import ClassVar

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel


class RDNA2W8A16Fp8BlockLinearKernel(Fp8BlockScaledMMLinearKernel):
    # gfx1030 has no FP8 hardware. The upstream `apply_weights` would
    # call `quant_fp8` to convert FP16 -> FP8 per-token, but we consume
    # FP16 directly. The `As` placeholder passed to `apply_block_scaled_mm`
    # is ignored.
    apply_input_quant: ClassVar[bool] = False  # type: ignore[name-defined]

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W8A16 FP8 block kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 W8A16 FP8 block kernel requires gfx1030"

        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gemm_w8a16_fp8_dense")
        ):
            return (
                False,
                "torch.ops._rocm_C.gemm_w8a16_fp8_dense missing -- rebuild "
                "C++ extension (CUTLASS blocker)",
            )

        return True, None

    @classmethod
    def can_implement(cls, c) -> tuple[bool, str | None]:
        supported, reason = cls.is_supported()
        if not supported:
            return False, reason

        # fp16 activations only (gfx1030 has no v_dot2_f32_bf16).
        if c.input_dtype != torch.float16:
            return (
                False,
                f"RDNA2 W8A16 FP8 block kernel only supports fp16 input "
                f"(got {c.input_dtype})",
            )

        # fp16 output only (atomic_add_pk4_f16 emits packed CAS on fp16).
        if c.out_dtype != torch.float16:
            return (
                False,
                f"RDNA2 W8A16 FP8 block kernel only supports fp16 output "
                f"(got {c.out_dtype})",
            )

        # Block-strategy weight quant: positive N-block AND K-block.
        ws = c.weight_quant_key.scale.group_shape
        if not (ws.row > 0 and ws.col > 0):
            return (
                False,
                "RDNA2 W8A16 FP8 block kernel requires block-quant weight "
                f"scales with positive N-block and K-block (got "
                f"group_shape={ws})",
            )

        # Symmetric weight quant only (kernel assumes zero=null).
        if not c.weight_quant_key.symmetric:
            return (
                False,
                "RDNA2 W8A16 FP8 block kernel only supports symmetric "
                f"weight quant (asymmetric={not c.weight_quant_key.symmetric})",
            )

        # Activation quant: the BLOCK scheme's activation quant key has a
        # per-group scale (1, K_block). We ignore the act quant entirely,
        # but we still accept the key shape so we get selected over
        # Triton fallback.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # The compressed-tensors BLOCK FP8 scheme stores weight as [N, K]
        # (because the upstream Triton block-scaled MM kernel reads B as
        # [N, K], see fp8_utils.w8a8_triton_block_scaled_mm: N, K = B.shape).
        # The base class's process_weights_after_loading renames
        # weight_scale -> weight_scale_inv and reshapes it. We leave the
        # weight un-transposed here so the base class's output_shape
        # computation (weight.shape[0] = N) stays correct, and transpose
        # B inside apply_block_scaled_mm to feed our C++ W8A16 kernel
        # which expects [K, N].
        super().process_weights_after_loading(layer)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # `apply_input_quant=False` means `A` arrives as the original
        # FP16 input (not the post-quant FP8). `As` is a placeholder
        # tensor we must not consume. Some paths (indexer, MLA) pass
        # bf16 -- gfx1030 has no v_dot2_f32_bf16 so we cast to fp16
        # (no precision loss vs the model's intended fp16 path).
        if A.dtype == torch.bfloat16:
            A = A.to(torch.float16)
        assert A.dtype == torch.float16, (
            f"RDNA2 W8A16 FP8 block kernel: A must be fp16 when "
            f"apply_input_quant=False (got {A.dtype})"
        )
        assert B.dtype == torch.float8_e4m3fn, (
            f"RDNA2 W8A16 FP8 block kernel: B must be fp8 e4m3fn "
            f"(got {B.dtype})"
        )

        # B arrives as [N, K] (un-transposed). Transpose to [K, N] for
        # our C++ kernel.
        N, K = B.shape
        M, K_a = A.shape
        assert K == K_a, f"K mismatch: A={K_a}, B={K}"

        B_kn = B.t().contiguous()  # [K, N]

        # Bs is the block weight scale. The BLOCK scheme stores it as
        # either [K_groups, N_blocks] or [N_blocks, K_groups]
        # (one scale per (K_block, N_block) cell). Our C++ kernel reads
        # it as b_scales[group * N + n] -- a full [K_groups, N] tensor
        # where each block's value is repeated block_n times across the
        # N dimension. Broadcast in Python.
        block_n = self.weight_group_shape.row
        if Bs.dim() == 2:
            if Bs.shape == (N // block_n, K // block_n):
                Bs = Bs.t().contiguous()
            if Bs.shape[1] == N // block_n:
                Bs = Bs.repeat_interleave(block_n, dim=1).contiguous()
        # C++ kernel requires fp16 scales; compressed-tensors stores
        # them as fp32 by default.
        if Bs.dtype != torch.float16:
            Bs = Bs.to(torch.float16)
        # Now Bs should be [K_groups, N].
        K_groups = Bs.shape[0]
        assert K % K_groups == 0, (
            f"K={K} not divisible by K_groups={K_groups}"
        )
        group_size = K // K_groups
        assert Bs.shape == (K_groups, N), (
            f"Bs shape mismatch: got {Bs.shape}, expected "
            f"({K_groups}, {N})"
        )

        # torch's float8_e4m3fn storage is byte-identical to uint8.
        b_bytes = B_kn.view(torch.uint8)

        # Pre-zeroed fp16 output; the kernel atomic-adds into it.
        output = torch.zeros((M, N), dtype=torch.float16, device=A.device)

        ops.gemm_w8a16_fp8_dense(A, b_bytes, Bs, output, group_size)

        return output