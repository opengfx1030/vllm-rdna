# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy model of the W4A8 sdot4 explore contract (gfx1030).

This is the oracle for both the CPU checks an agent can run
(``test_reference.py``) and the V620 harness (``bench.py``). It models:

* the in-memory W4 pack that ``RDNA2W4A16LinearKernel`` leaves behind:
  GPTQ ``[K/8, N]`` int32 packed along K, then ``gptq_shuffle``;
* the SWAR nibble unpack used by the HIP draft (``w & 0x0F0F0F0F`` and
  ``(w >> 4) & 0x0F0F0F0F``) and the A byte order that matches it;
* vLLM's dynamic per-token int8 quant (``scale = absmax / 127``, round half
  to even, saturate) plus per-group activation sums;
* the per-group i32 ``sdot4`` accumulate, ``z * asum`` zero fold and f32
  group flush;
* an f64 oracle and error bounds for the f32 and fp16 output paths;
* the VALU budget and split-K rules the docs quote.

Only NumPy is needed, so everything here runs without torch or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Slot s (bits 4s..4s+3) of a shuffled dword holds K offset SHUFFLE_SLOTS[s].
# Mirrors shuffle_4bit_8 in csrc/libtorch_stable/quantization/gptq/qdq_4.cuh.
SHUFFLE_SLOTS = (0, 2, 4, 6, 1, 3, 5, 7)
# Byte j of every 8-byte A chunk holds a[8c + A_PERM[j]], so that
# sdot4(a_lo, w & LO_MASK) and sdot4(a_hi, (w >> 4) & LO_MASK) line up in K.
A_PERM = tuple(SHUFFLE_SLOTS[2 * b] for b in range(4)) + tuple(
    SHUFFLE_SLOTS[2 * b + 1] for b in range(4)
)
LO_MASK = np.uint32(0x0F0F0F0F)
F32_EPS = 2.0**-24
F16_EPS = 2.0**-11
F16_MIN_SUBNORMAL = 2.0**-24
I24_MAX = (1 << 23) - 1
SUPPORTED_GROUP_SIZES = (32, 64, 128)
K_STEPS = (16, 32)
ELIGIBLE_LAYERS = ("gate_up_proj", "down_proj")
ELIGIBLE_WEIGHT_TYPES = ("uint4", "uint4b8")


# ---------------------------------------------------------------------------
# Weight packing (what RDNA2W4A16LinearKernel leaves in memory)
# ---------------------------------------------------------------------------


def pack_k_major(q_kn: np.ndarray) -> np.ndarray:
    """Packs ``[K, N]`` uint4 values into the GPTQ ``[K/8, N]`` layout.

    K offset ``i`` of each 8-row chunk lands in bits ``4i..4i+3``.
    """
    k, n = q_kn.shape
    if k % 8:
        raise ValueError(f"K={k} must be a multiple of 8")
    q = q_kn.astype(np.uint32).reshape(k // 8, 8, n)
    out = np.zeros((k // 8, n), dtype=np.uint32)
    for i in range(8):
        out |= q[:, i, :] << np.uint32(4 * i)
    return out


def exllama_shuffle(packed: np.ndarray) -> np.ndarray:
    """Applies ``gptq_shuffle`` (4-bit, no act-order) to a ``[K/8, N]`` pack."""
    packed = packed.astype(np.uint32)
    out = np.zeros_like(packed)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        nibble = (packed >> np.uint32(4 * k_off)) & np.uint32(0xF)
        out |= nibble << np.uint32(4 * slot)
    return out


def unpack_shuffled(shuffled: np.ndarray) -> np.ndarray:
    """Inverts ``exllama_shuffle(pack_k_major(q))`` back to ``[K, N]``."""
    k8, n = shuffled.shape
    q = np.zeros((k8, 8, n), dtype=np.uint8)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        q[:, k_off, :] = (shuffled >> np.uint32(4 * slot)) & np.uint32(0xF)
    return q.reshape(k8 * 8, n)


def pack_zeros_n_major(z_gn: np.ndarray) -> np.ndarray:
    """Packs ``[G, N]`` stored zeros into the ``[G, N/8]`` layout kernels read.

    Nibble ``j`` of word ``i`` is column ``8i + j`` (``load4_zeros``).
    """
    g, n = z_gn.shape
    if n % 8:
        raise ValueError(f"N={n} must be a multiple of 8")
    z = z_gn.astype(np.uint32).reshape(g, n // 8, 8)
    out = np.zeros((g, n // 8), dtype=np.uint32)
    for j in range(8):
        out |= z[:, :, j] << np.uint32(4 * j)
    return out


def w4a16_dequant_k_order() -> tuple[int, ...]:
    """K offsets, in emit order, that ``dequant_4bit_8_fp16`` reads.

    The fp16 path masks ``0x000F000F`` / ``0x00F000F0`` on ``w`` and on
    ``w >> 8``; lane 0 of each half2 takes the low half-word, lane 1 the
    high one. On a correctly shuffled dword this must be ``0..7``.
    """
    order = []
    for shift in (0, 8):
        for nibble_bit in (0, 4):
            for lane_bit in (0, 16):
                order.append(SHUFFLE_SLOTS[(shift + nibble_bit + lane_bit) // 4])
    return tuple(order)


# ---------------------------------------------------------------------------
# sdot4 inner math
# ---------------------------------------------------------------------------


def swar_unpack(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-extends the 8 nibbles of shuffled dwords into two i8x4 packs.

    ``lo`` bytes hold K offsets ``(0, 4, 1, 5)`` and ``hi`` bytes hold
    ``(2, 6, 3, 7)``, which is why A is stored in :data:`A_PERM` order.
    Costs 3 VALU per dword on gfx1030 (and, lshr, and).
    """
    w = np.asarray(w, dtype=np.uint32)
    return w & LO_MASK, (w >> np.uint32(4)) & LO_MASK


def sign_extend_nibbles(w: np.ndarray) -> np.ndarray:
    """Per-nibble two's-complement view, shape ``[..., 8]`` in slot order.

    Only used to document why sign extension is the wrong expansion for
    uint4/uint4b8 packs.
    """
    w = np.asarray(w, dtype=np.uint32)
    nib = np.stack([(w >> np.uint32(4 * s)) & np.uint32(0xF) for s in range(8)], -1)
    nib = nib.astype(np.int64)
    return np.where(nib >= 8, nib - 16, nib)


def _bytes_i8(packs: np.ndarray) -> np.ndarray:
    packs = np.ascontiguousarray(packs, dtype="<u4")
    return packs.view(np.int8).reshape(*packs.shape, 4).astype(np.int64)


def wrap_i32(x: np.ndarray) -> np.ndarray:
    """Two's-complement i32 wraparound on an int64 array."""
    return (np.asarray(x, dtype=np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)


def sdot4(a_pack: np.ndarray, b_pack: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """``v_dot4_i32_i8`` with ``clamp=false``; broadcasts like NumPy."""
    prod = (_bytes_i8(a_pack) * _bytes_i8(b_pack)).sum(axis=-1)
    return wrap_i32(np.asarray(acc, dtype=np.int64) + prod)


def _check_i24(x: np.ndarray, what: str) -> None:
    if np.abs(np.asarray(x, dtype=np.int64)).max(initial=0) > I24_MAX:
        raise OverflowError(f"{what} does not fit a v_mad_i32_i24 operand")


# ---------------------------------------------------------------------------
# Activation quant (A8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActQuant:
    """int8 activations in the forms the kernels consume."""

    a_i8: np.ndarray  # [M, K] int8, natural K order
    a_perm: np.ndarray  # [M, K] int8, A_PERM order inside every 8-K chunk
    scale: np.ndarray  # [M] per token, or [M, K / group_size] per token-group
    asum: np.ndarray  # [M, K / group_size] int32

    @property
    def per_group_scale(self) -> bool:
        return self.scale.ndim == 2


def quantize_per_token(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matches vLLM ``dynamic_scaled_int8_quant`` (symmetric, per token).

    ``scale = absmax / 127``; values are ``rint(x * (127 / absmax))`` in f32
    with ties to even, saturated to int8; all-zero rows give zeros.
    """
    xf = np.asarray(x).astype(np.float32)
    absmax = np.abs(xf).max(axis=1)
    scale = absmax / np.float32(127.0)
    inv = np.zeros_like(absmax)
    nonzero = absmax != 0
    inv[nonzero] = np.float32(127.0) / absmax[nonzero]
    q = np.clip(np.rint(xf * inv[:, None]), -128, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def quantize_per_token_group(
    x: np.ndarray, group_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Same rounding as :func:`quantize_per_token`, one scale per K group.

    This is llama.cpp's Q8_1 idea (a scale and a sum per 32 values): an
    outlier channel only coarsens its own group instead of the whole token.
    """
    xf = np.asarray(x).astype(np.float32)
    m, k = xf.shape
    xg = xf.reshape(m, k // group_size, group_size)
    absmax = np.abs(xg).max(axis=2)
    scale = absmax / np.float32(127.0)
    inv = np.zeros_like(absmax)
    nonzero = absmax != 0
    inv[nonzero] = np.float32(127.0) / absmax[nonzero]
    q = np.clip(np.rint(xg * inv[:, :, None]), -128, 127).astype(np.int8)
    return q.reshape(m, k), scale.astype(np.float32)


def permute_a(a_i8: np.ndarray) -> np.ndarray:
    """Reorders every 8-K chunk of ``a_i8`` into :data:`A_PERM` order."""
    m, k = a_i8.shape
    if k % 8:
        raise ValueError(f"K={k} must be a multiple of 8")
    chunks = a_i8.reshape(m, k // 8, 8)[:, :, list(A_PERM)]
    return np.ascontiguousarray(chunks.reshape(m, k))


def group_sums(a_i8: np.ndarray, group_size: int) -> np.ndarray:
    """Per-(token, group) sums of the int8 activations as int32."""
    m, k = a_i8.shape
    if k % group_size:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    grouped = a_i8.astype(np.int32).reshape(m, k // group_size, group_size)
    return grouped.sum(axis=2, dtype=np.int32)


def quantize_act(
    x: np.ndarray, group_size: int, per_group_scale: bool = False
) -> ActQuant:
    """Runs the act-quant step the draft's quant kernel performs."""
    if per_group_scale:
        a_i8, scale = quantize_per_token_group(x, group_size)
    else:
        a_i8, scale = quantize_per_token(x)
    return ActQuant(
        a_i8=a_i8,
        a_perm=permute_a(a_i8),
        scale=scale,
        asum=group_sums(a_i8, group_size),
    )


def _pad_rows(x: np.ndarray, m_tile: int) -> np.ndarray:
    pad = -x.shape[0] % m_tile
    return np.concatenate([x, np.zeros((pad, *x.shape[1:]), x.dtype)]) if pad else x


def tile_a(a_perm: np.ndarray, m_tile: int) -> np.ndarray:
    """``[M, K]`` -> the kernels' ``[T][K/8][MT][8]`` layout (rows >= M zero).

    Byte ``(t * K/8 + c) * MT * 8 + m * 8 + j`` is ``a_perm[t*MT + m, 8c + j]``.
    """
    a = _pad_rows(a_perm, m_tile)
    t, k = a.shape[0] // m_tile, a.shape[1]
    tiled = a.reshape(t, m_tile, k // 8, 8).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(tiled).reshape(-1)


def untile_a(tiled: np.ndarray, m: int, k: int, m_tile: int) -> np.ndarray:
    """Inverse of :func:`tile_a`, dropping the padding rows."""
    t = -(-m // m_tile)
    a = tiled.reshape(t, k // 8, m_tile, 8).transpose(0, 2, 1, 3)
    return a.reshape(t * m_tile, k)[:m]


def tile_groups(values_mg: np.ndarray, m_tile: int) -> np.ndarray:
    """``[M, G]`` -> ``[T][G][MT]`` (rows >= M zero): Σa and group A scales."""
    s = _pad_rows(values_mg, m_tile)
    t, g = s.shape[0] // m_tile, s.shape[1]
    return np.ascontiguousarray(s.reshape(t, m_tile, g).transpose(0, 2, 1)).ravel()


# ---------------------------------------------------------------------------
# Oracle and error bounds
# ---------------------------------------------------------------------------


def effective_zeros(stored_gn: np.ndarray, zero_offset: int) -> np.ndarray:
    """``stored + zero_offset``: 0 for AWQ ``uint4``, 1 for GPTQv1 ``uint4b8``."""
    return stored_gn.astype(np.int64) + zero_offset


def dequant_weights_f64(
    q_kn: np.ndarray, zeros_eff_gn: np.ndarray, scales_gn: np.ndarray, group_size: int
) -> np.ndarray:
    """``(q - z) * s`` in f64 (exact for fp16 scales)."""
    z = np.repeat(zeros_eff_gn.astype(np.float64), group_size, axis=0)
    s = np.repeat(scales_gn.astype(np.float64), group_size, axis=0)
    return (q_kn.astype(np.float64) - z) * s


@dataclass(frozen=True)
class Oracle:
    """f64 W4A8 result plus the magnitude used to scale error bounds."""

    c: np.ndarray  # [M, N] float64
    mag: np.ndarray  # [M, N] float64, s_a * (|a| @ |W|)


def oracle_w4a8(
    act: ActQuant,
    q_kn: np.ndarray,
    zeros_eff_gn: np.ndarray,
    scales_gn: np.ndarray,
    group_size: int,
) -> Oracle:
    """Exact-in-practice W4A8 result: int8 A times dequantized W in f64."""
    w = dequant_weights_f64(q_kn, zeros_eff_gn, scales_gn, group_size)
    a = act.a_i8.astype(np.float64)
    if act.per_group_scale:
        a = a * np.repeat(act.scale.astype(np.float64), group_size, axis=1)
        return Oracle(c=a @ w, mag=np.abs(a) @ np.abs(w))
    s_a = act.scale.astype(np.float64)[:, None]
    return Oracle(c=(a @ w) * s_a, mag=(np.abs(a) @ np.abs(w)) * s_a)


def w4a16_reference(
    x: np.ndarray,
    q_kn: np.ndarray,
    zeros_eff_gn: np.ndarray,
    scales_gn: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """fp16-activation result the W4A16 path approximates (f64)."""
    w = dequant_weights_f64(q_kn, zeros_eff_gn, scales_gn, group_size)
    return x.astype(np.float64) @ w


def w4a16_rdna2_weights(
    q_kn: np.ndarray, zeros_eff_gn: np.ndarray, scales_gn: np.ndarray, group_size: int
) -> np.ndarray:
    """fp16 weights exactly as the gfx1030 W4A16 kernels materialize them.

    ``prep_zero_scale_fp16`` (qdq_4_rdna2.cuh) rounds ``s·(−1024 − z)`` and
    ``s·(−64 − z)`` to fp16; ``dequant_4bit_8_fp16`` then does one fp16 FMA,
    ``(1024 + q)·s + z1`` for K offsets {0,1,4,5} and
    ``(1024 + 16q)·(s/16) + z16`` for {2,3,6,7}. The fp16 constant ``z1``
    (~s/2 ulp) is the baked-bias error JartX measured on gfx1100 (dafcde3).
    """
    s = np.repeat(scales_gn.astype(np.float64), group_size, axis=0)
    z = np.repeat(zeros_eff_gn.astype(np.float64), group_size, axis=0)
    z1 = (s * (-1024.0 - z)).astype(np.float16).astype(np.float64)
    z16 = (s * (-64.0 - z)).astype(np.float16).astype(np.float64)
    y16 = (s * 0.0625).astype(np.float16).astype(np.float64)
    q = q_kn.astype(np.float64)
    low = ((1024.0 + q) * s + z1).astype(np.float16)
    high = ((1024.0 + 16.0 * q) * y16 + z16).astype(np.float16)
    use_low = np.isin(np.arange(q.shape[0]) % 8, (0, 1, 4, 5))[:, None]
    return np.where(use_low, low, high)


def f32_flush_bound(
    mag: np.ndarray, groups: int, per_group_scale: bool = False
) -> np.ndarray:
    """Error bound for the f32 output path: one rounding per group flush.

    A per-group activation scale adds a multiply (a second rounding) per flush.
    """
    return ((2 if per_group_scale else 1) * groups + 4) * F32_EPS * mag


def f16_output_bound(
    mag: np.ndarray, groups: int, split_k: int, per_group_scale: bool = False
) -> np.ndarray:
    """Error bound for fp16 output: per-split cast plus fp16 atomic adds."""
    return (
        f32_flush_bound(mag, groups, per_group_scale)
        + (2 * split_k + 1) * F16_EPS * mag
        + 2 * split_k * F16_MIN_SUBNORMAL
    )


# ---------------------------------------------------------------------------
# Kernel replay
# ---------------------------------------------------------------------------


def emulate_kernel(
    act: ActQuant,
    w_shuf: np.ndarray,
    zeros_eff_gn: np.ndarray,
    scales_gn: np.ndarray,
    group_size: int,
    k_step: int = 32,
    split_k: int = 1,
    out_f16: bool = False,
    dwords_per_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replays the HIP draft's arithmetic on the CPU (small shapes only).

    Args:
        act: Quantized activations (the kernel reads ``a_perm`` and ``asum``).
        w_shuf: ``[K/8, N]`` shuffled W4 pack.
        zeros_eff_gn: ``[G, N]`` effective zero points.
        scales_gn: ``[G, N]`` fp16 weight scales.
        group_size: Weight group size.
        k_step: K advanced per unrolled step.
        split_k: Number of group-aligned K splits.
        out_f16: Cast each split to fp16 and accumulate in fp16 (the pk4
            atomic epilogue); otherwise return the f32 result.
        dwords_per_step: W dwords the body consumes per ``k_step``; defaults
            to ``k_step // 8``. A smaller value replays the ConfigH bug.

    Returns:
        ``(c [M, N], partials [M, G, N])``; partials are the int32 values
        ``sum(a * q) - z * asum`` handed to each group flush.
    """
    m, k = act.a_perm.shape
    n = w_shuf.shape[1]
    groups = k // group_size
    if k_step not in K_STEPS or group_size % k_step:
        raise ValueError(f"k_step={k_step} must be in {K_STEPS} and divide G")
    if groups % split_k:
        raise ValueError(f"split_k={split_k} does not divide {groups} groups")
    if split_k > 1 and not out_f16:
        raise ValueError("the f32 output path is single-split (plain stores)")
    per_step = k_step // 8 if dwords_per_step is None else dwords_per_step

    a_dw = np.ascontiguousarray(act.a_perm).view("<u4").reshape(m, k // 4)
    lo, hi = swar_unpack(w_shuf)
    partials = np.zeros((m, groups, n), dtype=np.int64)
    out = np.zeros((m, n), dtype=np.float16 if out_f16 else np.float32)
    groups_per_split = groups // split_k
    for split in range(split_k):
        c = np.zeros((m, n), dtype=np.float32)
        for g in range(split * groups_per_split, (split + 1) * groups_per_split):
            acc = np.zeros((m, n), dtype=np.int64)
            for k0 in range(g * group_size, (g + 1) * group_size, k_step):
                for j in range(per_step):
                    d = k0 // 8 + j
                    acc = sdot4(a_dw[:, 2 * d, None], lo[None, d, :], acc)
                    acc = sdot4(a_dw[:, 2 * d + 1, None], hi[None, d, :], acc)
            z = zeros_eff_gn[g].astype(np.int64)
            asum = act.asum[:, g].astype(np.int64)
            _check_i24(z, "zero point")
            _check_i24(asum, "activation group sum")
            t = wrap_i32(acc - z[None, :] * asum[:, None])
            partials[:, g, :] = t
            # fmaf(float(t), s, c): t and s are exact in f64, one rounding.
            s = scales_gn[g].astype(np.float64)[None, :]
            tf = t.astype(np.float32)
            if act.per_group_scale:  # fmaf(float(t) * sa, s, c)
                tf = tf * act.scale[:, g : g + 1]
            c = (tf.astype(np.float64) * s + c.astype(np.float64)).astype(np.float32)
        if not act.per_group_scale:
            c = c * act.scale[:, None]
        if out_f16:
            out = (out.astype(np.float32) + c.astype(np.float16)).astype(np.float16)
        else:
            out = c
    return out, partials


# ---------------------------------------------------------------------------
# Budget, split-K and eligibility rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopBudget:
    """VALU instructions per thread per weight group in the K loop."""

    sdot4: int
    unpack: int
    flush: int
    w4a16_fdot2: int
    w4a16_dequant: int

    @property
    def w4a8(self) -> int:
        return self.sdot4 + self.unpack + self.flush

    @property
    def w4a16(self) -> int:
        return self.w4a16_fdot2 + self.w4a16_dequant

    @property
    def ratio(self) -> float:
        return self.w4a16 / self.w4a8


def loop_budget(
    m_tile: int,
    n_per_thread: int,
    group_size: int,
    unpack_per_dword: int = 3,
    flush_per_output: int = 3,
    w4a16_dequant_per_dword: int = 9,
) -> LoopBudget:
    """VALU budget of the W4A8 draft vs the W4A16 ConfigA-class loop.

    W4A8 per W dword: ``unpack_per_dword`` (shared by the M tile) plus two
    ``sdot4`` per row; per output and group a ``flush_per_output`` flush
    (mad_i32_i24, cvt_f32_i32, fmac_f32). W4A16 per W dword: 9 VALU of
    exllama dequant plus four ``fdot2`` per row.
    """
    dwords = group_size // 8
    return LoopBudget(
        sdot4=2 * m_tile * n_per_thread * dwords,
        unpack=unpack_per_dword * n_per_thread * dwords,
        flush=flush_per_output * m_tile * n_per_thread,
        w4a16_fdot2=4 * m_tile * n_per_thread * dwords,
        w4a16_dequant=w4a16_dequant_per_dword * n_per_thread * dwords,
    )


def weight_reread_bytes(m: int, n: int, k: int, m_tile: int) -> int:
    """W4 bytes streamed when every M tile re-reads the packed weight."""
    return -(-m // m_tile) * n * k // 2


def valid_split_ks(k: int, group_size: int, max_split: int = 16) -> list[int]:
    """Split-K factors whose K ranges start and end on group boundaries."""
    groups = k // group_size
    return [s for s in range(1, max_split + 1) if groups % s == 0]


def lds_bytes(
    m_tile: int, k_per_split: int, group_size: int, a_group: bool = False
) -> int:
    """LDS for the A-in-LDS variant: int8 A tile, int32 group sums, and the
    f32 group A scales when the config uses per-(token, group) scales."""
    per_group = (2 if a_group else 1) * 4
    return m_tile * k_per_split + m_tile * (k_per_split // group_size) * per_group


def pick_split_k(
    m: int,
    n: int,
    k: int,
    group_size: int,
    m_tile: int = 16,
    n_tile: int = 1024,
    max_split: int = 16,
    a_group: bool = False,
) -> int:
    """ConfigA's split-K heuristic restricted to group-aligned splits.

    Mirrors ``compute_split_k`` in ``q_gemm_rdna2_prefill.cu`` (LDS budget
    16/64/32 KiB by grid size, then grow the split while the grid is small
    or the K range is long); the HIP host code implements the same rule.
    """
    blocks = -(-m // m_tile) * -(-n // n_tile)
    if blocks > 1024:
        budget = 16 * 1024
    elif blocks > 256:
        budget = 64 * 1024
    else:
        budget = 32 * 1024
    splits = valid_split_ks(k, group_size, max_split)

    def lds(split: int) -> int:
        return lds_bytes(m_tile, k // split, group_size, a_group)

    i = 0
    while i + 1 < len(splits) and lds(splits[i]) > budget:
        i += 1
    while i + 1 < len(splits) and (blocks * splits[i] < 2048 or k // splits[i] > 2048):
        if lds(splits[i + 1]) > budget:
            break
        i += 1
    return splits[i]


def eligibility(
    weight_type: str,
    group_size: int,
    k: int,
    n: int,
    has_g_idx: bool,
    layer: str,
) -> str | None:
    """Returns ``None`` if a layer may take the W4A8 path, else why not."""
    if weight_type not in ELIGIBLE_WEIGHT_TYPES:
        return f"weight type {weight_type} is not a 4-bit zero-point pack"
    if has_g_idx:
        return "act-order (g_idx) packs are out of scope"
    if group_size not in SUPPORTED_GROUP_SIZES:
        return f"group size {group_size} not in {SUPPORTED_GROUP_SIZES}"
    if k % group_size or k % 32:
        return "K must be a multiple of the group size and of 32"
    if n % 8:
        return "N must be a multiple of 8"
    if not layer.endswith(ELIGIBLE_LAYERS):
        return "only dense FFN gate_up_proj / down_proj are in scope"
    return None


# ---------------------------------------------------------------------------
# Problem generation shared by tests and the GPU harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """A random W4A8 GEMM in the exact layouts the kernels read."""

    x: np.ndarray  # [M, K] float16 activations
    q_kn: np.ndarray  # [K, N] uint8 values in [0, 15]
    zeros_stored_gn: np.ndarray  # [G, N] stored zero nibbles
    zero_offset: int  # 0 for uint4 (AWQ), 1 for uint4b8 (GPTQv1)
    scales_gn: np.ndarray  # [G, N] float16
    group_size: int

    @property
    def zeros_eff_gn(self) -> np.ndarray:
        return effective_zeros(self.zeros_stored_gn, self.zero_offset)

    @property
    def w_shuf(self) -> np.ndarray:
        return exllama_shuffle(pack_k_major(self.q_kn))

    @property
    def qzeros(self) -> np.ndarray:
        return pack_zeros_n_major(self.zeros_stored_gn)


def make_problem(
    m: int,
    n: int,
    k: int,
    group_size: int,
    weight_type: str,
    seed: int = 0,
    x_scale: float = 0.5,
    outlier_channels: int = 0,
) -> Problem:
    """Builds a random problem; ``uint4b8`` uses the synthesized zero (7 + 1).

    ``outlier_channels`` scales a few K channels by 20x to mimic the
    activation outliers that make per-token int8 lossy.
    """
    rng = np.random.default_rng(seed)
    groups = k // group_size
    x = rng.standard_normal((m, k)).astype(np.float32) * x_scale
    if outlier_channels:
        cols = rng.choice(k, size=outlier_channels, replace=False)
        x[:, cols] *= 20.0
    q = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    scales = (rng.random((groups, n)) * 0.02 + 0.002).astype(np.float16)
    if weight_type == "uint4":
        zeros, offset = rng.integers(0, 16, size=(groups, n), dtype=np.uint8), 0
    elif weight_type == "uint4b8":
        zeros, offset = np.full((groups, n), 7, dtype=np.uint8), 1
    else:
        raise ValueError(f"unsupported weight type {weight_type}")
    return Problem(
        x=x.astype(np.float16),
        q_kn=q,
        zeros_stored_gn=zeros,
        zero_offset=offset,
        scales_gn=scales,
        group_size=group_size,
    )
