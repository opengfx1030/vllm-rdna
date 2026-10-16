# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fatbin isolation. No new kernels, and no gfx1030 objects on gfx1100.

The hetero cold path is the existing gfx1030 W4A16 kernel
(``moe_gptq_gemm_rdna2``, fdot2). v0.31 also has a same-arch W4A8
sdot4 kernel (``moe_w4a8_gemm_rdna2``) and gfx1100 WMMA units
(``gptq_gemm_rdna3_wmma``, ``moe_gptq_gemm_rdna3``) gated by
``__HIP__RDNA3__``. This selector does not launch either of those.
"""

GFX1030_DOT_OBJECTS = frozenset({"moe_gptq_gemm_rdna2", "RDNA2W4A16MoEExperts"})
# Same-arch gfx1100 quant. Hetero must not place these on gfx1030.
RDNA3_WMMA_OBJECTS = frozenset(
    {
        "gptq_gemm_rdna3",
        "gptq_gemm_rdna3_wmma",
        "moe_gptq_gemm_rdna3",
    }
)
FAST_W4A16_KERNEL = "triton_wna16"
COLD_W4A16_KERNEL = "moe_gptq_gemm_rdna2"


def assert_device_kernel(arch: str, kernel: str) -> None:
    """Reject a kernel that belongs on the other fatbin.

    Args:
        arch: Device arch, ``gfx1100`` or ``gfx1030``.
        kernel: Kernel or experts-class name about to be launched.

    Raises:
        RuntimeError: The object would be loaded on the wrong arch.

    """
    on_gfx1030 = arch == "gfx1030" or arch.startswith("gfx103")
    if on_gfx1030 and (kernel in RDNA3_WMMA_OBJECTS or "wmma" in kernel):
        raise RuntimeError(
            f"refusing RDNA3 WMMA object {kernel} on {arch}; "
            f"cold W4A16 stays on {COLD_W4A16_KERNEL} (fdot2)"
        )
    if arch.startswith("gfx110") and kernel in GFX1030_DOT_OBJECTS:
        raise RuntimeError(f"refusing to load gfx1030 object {kernel} on {arch}")
    if on_gfx1030 and kernel not in GFX1030_DOT_OBJECTS:
        raise RuntimeError(
            f"gfx1030 cold experts use {COLD_W4A16_KERNEL} (fdot2), not {kernel}"
        )


def select_hot_kernel(arch: str) -> str:
    """Existing Triton W4A16 path for the fast tier.

    Args:
        arch: Fast-tier arch.

    Returns:
        ``triton_wna16``.

    Raises:
        RuntimeError: A gfx1030 device was asked to run the fast path.
        NotImplementedError: The arch is not the implemented gfx1100 tier.

    """
    if arch == "gfx1030" or arch.startswith("gfx103"):
        raise RuntimeError("fast tier must not run on the gfx1030 fatbin")
    if not arch.startswith("gfx110"):
        raise NotImplementedError(
            f"fast-tier W4A16 is implemented for gfx1100, not {arch}"
        )
    assert_device_kernel(arch, FAST_W4A16_KERNEL)
    return FAST_W4A16_KERNEL


def select_cold_kernel(arch: str) -> str:
    """Existing gfx1030 W4A16 MoE kernel.

    Args:
        arch: Cold-tier arch.

    Returns:
        ``moe_gptq_gemm_rdna2``.

    """
    assert_device_kernel(arch, COLD_W4A16_KERNEL)
    return COLD_W4A16_KERNEL
