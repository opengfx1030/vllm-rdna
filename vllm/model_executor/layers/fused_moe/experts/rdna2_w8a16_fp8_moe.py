# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) W8A16-FP8 fused MoE experts class.

This is a stub for the FusedMoEExpertsModular interface. The full
implementation requires:
- workspace_shapes: sized for the RDNA2 fused kernel's two GEMM calls
- apply: orchestrates the two HIP kernel launches (w1, w2) with the
  RDNA2 fused MoE kernel ops.moe_w8a16_fp8_gemm_rdna2
- is_supported_config: gate on gfx1030 + FP8 + non-quantized activations
- _supports_* classmethods for routing/dtype compatibility

Until the full implementation lands, the kernel and dispatcher are
registered but the FP8 MoE path will fail at is_supported_config with
NotImplementedError. The kernel itself is fully functional and can be
called directly via torch.ops._rocm_C.moe_w8a16_fp8_gemm_rdna2.
"""

import torch

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
)


class RDNA2W8A16FP8Experts(FusedMoEExpertsModular):
    """RDNA2 (gfx1030) W8A16-FP8 fused MoE experts - STUB.

    See module docstring for what's needed to complete this class.
    """

    @staticmethod
    def is_supported_config(
        cls, config, weight_key, activation_key, activation_format
    ):
        raise NotImplementedError(
            "RDNA2W8A16FP8Experts is not yet implemented as a "
            "FusedMoEExpertsModular. The underlying HIP kernel "
            "moe_w8a16_fp8_gemm_rdna2 is fully functional and can be "
            "called directly via torch.ops._rocm_C.moe_w8a16_fp8_gemm_rdna2."
        )
