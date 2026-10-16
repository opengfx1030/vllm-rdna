# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 (int8 x int8 sdot4) opt-in for gfx1030: env latch and weight packing.

The HIP kernels (``csrc/rocm/w8a8_sdot4_rdna2.cu``, ``csrc/rocm/moe_w8a8_rdna2.cu``)
read int8 weights in a K-major dword layout that pairs byte for byte with the
activation tile ``w4a8_act_quant_kernel`` writes: chunk ``c`` of 8 K positions is
two dword rows per column, row ``2c`` holding ``w[8c + {0, 4, 1, 5}]`` and row
``2c + 1`` holding ``w[8c + {2, 6, 3, 7}]`` (``A_PERM``, byte 0 lowest).
``pack_weight`` builds it once at load time.

Gate: ``VLLM_RDNA2_W8A8_SDOT4=1`` on gfx1030 with the ops built. Read once at
construction, never in a forward.
"""

import os

import torch

ENV_VAR = "VLLM_RDNA2_W8A8_SDOT4"
# kAPerm in csrc/rocm/w4a8_sdot4_rdna2.cuh: byte order of each 8-K chunk.
A_PERM = (0, 4, 1, 5, 2, 6, 3, 7)
# Activation quant groups the kernels instantiate, largest first.
ACT_GROUPS = (128, 64, 32)


def _op_built(name: str) -> bool:
    # hasattr on torch.ops namespaces is unreliable; probe the attribute.
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    try:
        getattr(torch.ops._rocm_C, name)
        return True
    except (AttributeError, RuntimeError):
        return False


def dense_ops_built() -> bool:
    return _op_built("w8a8_gemm_rdna2")


def moe_ops_built() -> bool:
    return _op_built("moe_w8a8_gemm_rdna2")


def on_gfx1030() -> bool:
    """True on gfx1030 exactly: the kernel bodies are compiled for it alone."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    try:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return False
    return arch.startswith("gfx1030")


def enabled() -> bool:
    return os.environ.get(ENV_VAR) == "1" and on_gfx1030()


def act_group(k: int) -> int:
    """Largest kernel activation group dividing K."""
    for g in ACT_GROUPS:
        if k % g == 0:
            return g
    raise ValueError(f"RDNA2 W8A8 needs K % 32 == 0, got K={k}")


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """int8 [N, K] -> int32 [K/4, N] in the kernel's dword layout."""
    assert w.dtype == torch.int8 and w.dim() == 2, "expected int8 [N, K]"
    n, k = w.shape
    assert k % 32 == 0 and n % 4 == 0, (
        f"needs K % 32 == 0 and N % 4 == 0, got {w.shape}"
    )
    chunks = w.reshape(n, k // 8, 8)[:, :, list(A_PERM)]
    planes = chunks.reshape(n, k // 8, 2, 4).permute(1, 2, 0, 3).contiguous()
    return planes.view(torch.int32).reshape(k // 4, n)


def unpack_weight(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_weight``: int32 [K/4, N] -> int8 [N, K]."""
    k4, n = packed.shape
    k = k4 * 4
    planes = packed.reshape(k // 8, 2, n, 1).view(torch.int8)
    chunks = planes.reshape(k // 8, 2, n, 4).permute(2, 0, 1, 3).reshape(n, k // 8, 8)
    inv = [A_PERM.index(i) for i in range(8)]
    return chunks[:, :, inv].reshape(n, k).contiguous()


def pack_moe_weight(w: torch.Tensor) -> torch.Tensor:
    """int8 [E, N, K] -> int32 [E, K/4, N], one ``pack_weight`` per expert."""
    assert w.dtype == torch.int8 and w.dim() == 3, "expected int8 [E, N, K]"
    return torch.stack([pack_weight(w[e]) for e in range(w.shape[0])]).contiguous()
