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
    if weight_quant.num_bits not in (4, 8):
        return False

    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    if not hasattr(torch.ops, "_rocm_C"):
        return False

    # RDNA3 (gfx1100): W4A16 only
    if on_gfx1100() and weight_quant.num_bits == 4 and \
            hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
        return True

    # RDNA2 (gfx1030): W4A16 AWQ, W4A16 GPTQ, W8A16 INT8, W8A16 FP8
    if on_gfx10x():
        if weight_quant.num_bits == 4 and \
                hasattr(torch.ops._rocm_C, "moe_awq_gemm_rdna2"):
            return True
        if weight_quant.num_bits == 4 and \
                hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2"):
            return True
        if weight_quant.num_bits == 8 and \
                hasattr(torch.ops._rocm_C, "moe_w8a16_gemm_rdna2"):
            return True

    return False


def is_supported_fp8(weight_quant) -> bool:
    """Check if the native RDNA2 FP8 W8A16 MoE kernel is available."""
    if weight_quant.num_bits != 8:
        return False
    from vllm.platforms.rocm import on_gfx10x
    if not on_gfx10x():
        return False
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    return hasattr(torch.ops._rocm_C, "moe_w8a16_fp8_gemm_rdna2")


def is_supported_mxfp4(weight_quant) -> bool:
    """Check if the native RDNA2 W4A4 MXFP4 MoE kernel is available.

    DeepSeek V4 Flash uses MXFP4 (E2M1 + UE8M0) for expert weights.
    gfx1030 cannot use CUTLASS MXFP4 (CDNA-only) or Marlin MXFP4
    (slow at M=1..15 decode). The native RDNA2 V_DOT2 kernel is the
    only compute-optimal path.
    """
    if weight_quant.num_bits != 4:
        return False
    from compressed_tensors.quantization import QuantizationType
    if weight_quant.type != QuantizationType.FLOAT:
        return False
    from vllm.platforms.rocm import on_gfx10x
    if not on_gfx10x():
        return False
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    return hasattr(torch.ops._rocm_C, "moe_mxfp4_gemm_rdna2")


def make_method_mxfp4(weight_quant, input_quant, moe_config):
    """Create the native RDNA2 W4A4 MXFP4 MoE method."""
    from vllm.platforms.rocm import on_gfx10x
    if on_gfx10x():
        from .compressed_tensors_moe_w4a4_mxfp4_rdna2 import (
            CompressedTensorsW4A4Mxfp4RDNA2MoEMethod,
        )
        logger.info_once(
            "Using CompressedTensorsW4A4Mxfp4RDNA2MoEMethod "
            "(native RDNA2 W4A4 MXFP4 HIP kernel)"
        )
        return CompressedTensorsW4A4Mxfp4RDNA2MoEMethod(moe_config)
    raise RuntimeError("is_supported_mxfp4() returned True but no kernel matched")


def make_method(weight_quant, input_quant, moe_config):
    """Create the native ROCm MoE method. Call only after is_supported()."""
    from vllm.platforms.rocm import on_gfx10x, on_gfx1100

    if on_gfx1100() and weight_quant.num_bits == 4 and \
            hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
        from .compressed_tensors_moe_wna16_rdna3 import (
            CompressedTensorsWNA16RDNA3MoEMethod,
        )

        logger.info_once(
            "Using CompressedTensorsWNA16RDNA3MoEMethod (native RDNA3 HIP kernel)"
        )
        return CompressedTensorsWNA16RDNA3MoEMethod(
            weight_quant, input_quant, moe_config
        )

    if on_gfx10x():
        if weight_quant.num_bits == 4 and \
                hasattr(torch.ops._rocm_C, "moe_awq_gemm_rdna2"):
            from .compressed_tensors_moe_awq_rdna2 import (
                CompressedTensorsAWQMoERDNA2Method,
            )

            logger.info_once(
                "Using CompressedTensorsAWQMoERDNA2Method "
                "(native RDNA2 AWQ W4A16 HIP kernel)"
            )
            return CompressedTensorsAWQMoERDNA2Method(moe_config)

        if weight_quant.num_bits == 4 and \
                hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2"):
            from .compressed_tensors_moe_wna16_rdna2 import (
                CompressedTensorsWNA16RDNA2MoEMethod,
            )

            logger.info_once(
                "Using CompressedTensorsWNA16RDNA2MoEMethod "
                "(native RDNA2 W4A16 HIP kernel)"
            )
            return CompressedTensorsWNA16RDNA2MoEMethod(
                weight_quant, input_quant, moe_config
            )

        if weight_quant.num_bits == 8 and \
                hasattr(torch.ops._rocm_C, "moe_w8a16_gemm_rdna2"):
            from .compressed_tensors_moe_wna16_rdna2_w8a16 import (
                CompressedTensorsWNA16RDNA2W8A16MoEMethod,
            )

            logger.info_once(
                "Using CompressedTensorsWNA16RDNA2W8A16MoEMethod "
                "(native RDNA2 W8A16 INT8 HIP kernel)"
            )
            return CompressedTensorsWNA16RDNA2W8A16MoEMethod(
                weight_quant, input_quant, moe_config
            )

    raise RuntimeError("is_supported() returned True but no kernel matched")
