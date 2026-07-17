# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE kernel dispatcher.

Selects architecture-specific native HIP MoE kernels in priority order.
Falls back to the Triton WNA16 path when no native kernel is available.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_supported(weight_quant) -> bool:
    """Check if a native ROCm MoE kernel is available for this config."""
    if weight_quant.num_bits != 4:
        return False

    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    if not hasattr(torch.ops, "_rocm_C"):
        return False

    # RDNA3 (gfx1100)
    if on_gfx1100() and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
        return True

    # RDNA2 (gfx1030)
    if on_gfx10x() and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2"):
        return True

    return False


def make_method(weight_quant, input_quant, moe_config):
    """Create the native ROCm MoE method. Call only after is_supported()."""
    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    if on_gfx1100() and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
        from .compressed_tensors_moe_wna16_rdna3 import (
            CompressedTensorsWNA16RDNA3MoEMethod,
        )

        logger.info_once(
            "Using CompressedTensorsWNA16RDNA3MoEMethod (native RDNA3 HIP kernel)"
        )
        return CompressedTensorsWNA16RDNA3MoEMethod(
            weight_quant, input_quant, moe_config
        )

    if on_gfx10x() and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2"):
        from .compressed_tensors_moe_wna16_rdna2 import (
            CompressedTensorsWNA16RDNA2MoEMethod,
        )

        logger.info_once(
            "Using CompressedTensorsWNA16RDNA2MoEMethod (native RDNA2 HIP kernel)"
        )
        return CompressedTensorsWNA16RDNA2MoEMethod(
            weight_quant, input_quant, moe_config
        )

    raise RuntimeError("is_supported() returned True but no kernel matched")
