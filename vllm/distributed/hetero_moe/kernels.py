# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fatbin isolation. No new kernels, and no gfx1030 objects on gfx1100."""

GFX1030_DOT_OBJECTS = frozenset({"moe_gptq_gemm_rdna2", "RDNA2W4A16MoEExperts"})
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
    if arch.startswith("gfx110") and kernel in GFX1030_DOT_OBJECTS:
        raise RuntimeError(f"refusing to load gfx1030 object {kernel} on {arch}")
    if arch == "gfx1030" and kernel not in GFX1030_DOT_OBJECTS:
        raise RuntimeError(
            f"gfx1030 cold experts use {COLD_W4A16_KERNEL}, not {kernel}"
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
