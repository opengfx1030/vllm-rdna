# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 attention support for AMD RDNA GPUs (gfx10.3 / gfx11 / gfx12).

RDNA has no AITER and (on gfx1030) no bf16 math, so this package carries the
RDNA-only pieces of the DeepSeek-V4 AMD path: thin wrappers over the
``_rocm_C`` dsv4 HIP kernels built from ``csrc/rocm/rdna/dsv4/`` (``ops``).
Nothing here is imported on CUDA or CDNA.
"""
