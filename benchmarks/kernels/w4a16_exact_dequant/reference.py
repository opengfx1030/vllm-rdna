# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy model of the gfx1030 W4A16 fp16 dequant, baked and exact.

Weights are ``[K, N]`` with per-group ``zeros`` (effective, stored + offset)
and fp16 ``scales``, both ``[K / G, N]``. The kernels read exllama-shuffled
dwords; the shuffle only decides which K offsets take the ``(1024 + q)``
"low" formula and which the ``(1024 + 16 q)`` "high" one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOW_OFFSETS = (0, 1, 4, 5)  # K % 8 served by the (1024 + q) half2 pairs
SLOT_TO_K = (0, 2, 4, 6, 1, 3, 5, 7)  # gptq_shuffle: nibble slot -> K offset


@dataclass(frozen=True)
class Problem:
    x: np.ndarray  # [M, K] fp16
    q: np.ndarray  # [K, N] 0..15
    zeros: np.ndarray  # [G, N] effective zero points
    scales: np.ndarray  # [G, N] fp16
    group_size: int


def make_problem(
    m: int,
    n: int,
    k: int,
    group_size: int,
    zeros: str = "random",
    seed: int = 0,
    scale_range: tuple[float, float] = (0.01, 0.06),
) -> Problem:
    """Random problem in the style of tests/kernels/quantization/
    test_rdna2_w4a16.py: x ~ 0.25 N(0, 1), uniform nibbles and scales.
    ``zeros`` is "random" (AWQ-like, 0..15) or "symmetric" (8)."""
    rng = np.random.default_rng(seed)
    groups = k // group_size
    x = (0.25 * rng.standard_normal((m, k))).astype(np.float16)
    q = rng.integers(0, 16, size=(k, n))
    if zeros == "random":
        z = rng.integers(0, 16, size=(groups, n))
    elif zeros == "symmetric":
        z = np.full((groups, n), 8)
    else:
        raise ValueError(zeros)
    lo, hi = scale_range
    s = (lo + (hi - lo) * rng.random((groups, n))).astype(np.float16)
    return Problem(x, q, z, s, group_size)


def _per_k(p_gn: np.ndarray, group_size: int) -> np.ndarray:
    return np.repeat(p_gn.astype(np.float64), group_size, axis=0)


def _low_rows(k: int) -> np.ndarray:
    return np.isin(np.arange(k) % 8, LOW_OFFSETS)[:, None]


def dequant_exact(q, zeros, scales, group_size) -> np.ndarray:
    """(q - z) * s in float64."""
    return (q - _per_k(zeros, group_size)) * _per_k(scales, group_size)


def dequant_baked(q, zeros, scales, group_size) -> np.ndarray:
    """fp16 weights of the default build, bit for bit.

    ``prep_zero_scale_fp16`` rounds ``s * (-1024 - z)`` and ``s * (-64 - z)``
    to fp16; ``dequant_4bit_8_fp16`` then does one fp16 FMA per weight.
    """
    s = _per_k(scales, group_size)
    z = _per_k(zeros, group_size)
    z1 = (s * (-1024.0 - z)).astype(np.float16).astype(np.float64)
    z16 = (s * (-64.0 - z)).astype(np.float16).astype(np.float64)
    y16 = (s * 0.0625).astype(np.float16).astype(np.float64)
    low = ((1024.0 + q) * s + z1).astype(np.float16)
    high = ((1024.0 + 16.0 * q) * y16 + z16).astype(np.float16)
    return np.where(_low_rows(q.shape[0]), low, high)


def dequant_exact_fp16(q, zeros, scales, group_size) -> np.ndarray:
    """fp16 weights with VLLM_RDNA2_W4A16_EXACT_DEQUANT=1, bit for bit.

    ``(1024 + q) - (1024 + z)`` and ``(1024 + 16q) - (1024 + 16z)`` are exact
    in fp16, so each weight is one rounding of ``(q - z) * s`` (the high pairs
    multiply ``16 (q - z)`` by the fp16 ``s / 16``).
    """
    s = _per_k(scales, group_size)
    z = _per_k(zeros, group_size)
    y16 = (s * 0.0625).astype(np.float16).astype(np.float64)
    low = ((q - z) * s).astype(np.float16)
    high = (16.0 * (q - z) * y16).astype(np.float16)
    return np.where(_low_rows(q.shape[0]), low, high)


def exllama_shuffle(packed: np.ndarray) -> np.ndarray:
    """gptq_shuffle(bits=4) of K-packed dwords: slot s gets K offset
    SLOT_TO_K[s], so the kernel's 0x000F000F masks read K pairs (0, 1), ..."""
    u = packed.astype(np.uint32)
    out = np.zeros_like(u)
    for slot, k in enumerate(SLOT_TO_K):
        out |= ((u >> np.uint32(4 * k)) & np.uint32(0xF)) << np.uint32(4 * slot)
    return out


def unpack_shuffled(shuffled: np.ndarray) -> np.ndarray:
    """[K/8, N] shuffled dwords -> [K, N] nibbles, as the kernels read them."""
    u = shuffled.astype(np.uint32)
    q = np.zeros((u.shape[0], 8, u.shape[1]), dtype=np.int64)
    for slot, k in enumerate(SLOT_TO_K):
        q[:, k] = (u >> np.uint32(4 * slot)) & np.uint32(0xF)
    return q.reshape(-1, u.shape[1])


def unpack_zeros(qzeros: np.ndarray) -> np.ndarray:
    """[G, N/8] N-packed zeros -> [G, N], as load4_zeros reads them."""
    u = qzeros.astype(np.uint32)
    z = [(u >> np.uint32(4 * j)) & np.uint32(0xF) for j in range(8)]
    return np.stack(z, axis=-1).reshape(u.shape[0], -1).astype(np.int64)


def fp16_from_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def rel_l2(a: np.ndarray, ref: np.ndarray) -> float:
    a64, r64 = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    return float(np.linalg.norm(a64 - r64) / max(np.linalg.norm(r64), 1e-30))


def matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ np.asarray(w, np.float64)
