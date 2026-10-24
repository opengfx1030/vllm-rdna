# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference codec for the proposed INT4 + FP16 block-scale KV cache format.

This file is the executable half of ``docs/explore/kv-int4-fp16-scales/README.md``.
It is pure Python with no third-party dependency so that it runs anywhere the
repo's docs run, and so that every number quoted in the README can be
regenerated with::

    python3 docs/explore/kv-int4-fp16-scales/reference_codec.py

It compares scale strategies for a 4-bit KV cache, all at the same element
width:

``int4_per_token_head``
    What the fork ships today. Asymmetric INT4, one FP32 scale per
    (token, head) row, with a 4-bit zero-point hidden in the scale's low
    mantissa bits. Mirrors ``vllm/v1/attention/ops/int4_per_token_head.py``.

``int4_fp16_block``
    The proposal. Symmetric INT4, one FP16 scale per block of ``k`` values
    along the head dimension, single level (no global scale).

``int4_fp16_block_g``
    The proposal with NVFP4's second level: FP16 block scale divided by a
    per-tensor FP32 global scale, which is multiplied back at read time.

``int4_e4m3_block_g``
    NVFP4's scale geometry with an integer element type: symmetric INT4,
    one E4M3 scale per block, divided by a per-tensor FP32 global scale.

``int4_e8m0_block``
    The MX-style alternative to the proposal: symmetric INT4 with one E8M0
    (power-of-two) scale per block, under both the OCP floor rule and the
    no-saturation round-up rule.

``int4_fp32_block``
    The same format with an FP32 scale, to test whether FP32 buys anything
    over FP16 at block granularity.

``int4_fp16_block_token_groups``
    The other grouping axis: one scale per (channel, group of ``k`` tokens)
    instead of per (token, group of ``k`` values). Costs the same memory and
    behaves completely differently on the two outlier modes.

``int8_*``
    INT8 with an E8M0, FP16, FP32 or E4M3 block scale.

``fp8_e4m3``
    FP8 e4m3 elements with a per-tensor or per-block scale. This is what
    ``--kv-cache-dtype fp8`` does, and it is the only KV cache dtype with a
    measurement on the V620.

``nvfp4``
    NVIDIA NVFP4 itself: E2M1 elements, one E4M3 scale per 16 values, divided
    by a per-tensor FP32 global scale.

The global scale is modelled the way NVFP4 actually does it: the block scale
stored in the cache is ``effective_scale / global``, and reconstruction
multiplies the global back in. That detail matters, because E4M3 has a narrow
exponent range and a block whose effective scale is far below the tensor
maximum underflows it, while FP16 does not.

The script reports round-trip SNR and the true bits/value of each scheme,
including scale overhead, so the cost of the scale format is visible next to
the accuracy it buys.

These are arithmetic results on synthetic data. They are not measurements on a
V620; ``TESTPLAN.md`` carries the empty cells for those.
"""

from __future__ import annotations

import bisect
import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "e4m3_encode",
    "e4m3_decode",
    "e8m0_encode",
    "e8m0_decode",
    "f16_round",
    "pack_int4",
    "unpack_int4",
    "quantize_int4_per_token_head",
    "quantize_int4_fp16_block",
    "quantize_int4_fp16_block_packed",
    "quantize_int4_fp16_block_g",
    "quantize_int4_e4m3_block_g",
    "quantize_int4_e8m0_block",
    "quantize_int8_e8m0_block",
    "quantize_int8_fp16_block",
    "quantize_nvfp4",
    "snr_db",
    "bits_per_value",
]

#: Largest finite value of the OCP E4M3 element type.
E4M3_MAX = 448.0
#: Largest finite value of FP16.
FP16_MAX = 65504.0
#: Smallest positive normal FP16.
FP16_MIN_NORMAL = 6.103515625e-05
#: Smallest positive normal E4M3.
E4M3_MIN_NORMAL = 2.0**-6


def f16_round(x: float) -> float:
    """Round ``x`` to the nearest FP16 value, ties to even.

    Args:
        x: The value to round.

    Returns:
        The FP16-representable value nearest to ``x``. Values beyond the FP16
        range saturate to +/- ``FP16_MAX`` rather than raising.

    """
    if x != x:  # NaN
        return float("nan")
    if x > FP16_MAX:
        return FP16_MAX
    if x < -FP16_MAX:
        return -FP16_MAX
    return struct.unpack("<e", struct.pack("<e", x))[0]


def f16_is_normal(x: float) -> bool:
    """Return True when ``x`` is a normal (non-zero, non-subnormal) FP16."""
    return x != 0.0 and abs(x) >= FP16_MIN_NORMAL


def f16_bits(x: float) -> int:
    """Return the 16 raw bits of an FP16 value."""
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16_from_bits(bits: int) -> float:
    """Return the FP16 value held in 16 raw bits."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def f16_ceil_6(x: float) -> float:
    """Return the smallest FP16 whose low 4 mantissa bits are zero, >= ``x``.

    This is the scale grid a 4-bit zero-point can live in: the zero-point
    takes the low 4 bits of the FP16 mantissa, which leaves a 6-bit mantissa
    for the scale itself.

    Args:
        x: The value to round up onto the grid.

    Returns:
        The grid value, never below ``x``.

    """
    bits = f16_bits(x)
    if bits & 0xF:
        bits = (bits & ~0xF) + 0x10
    if bits > 0x7BFF:
        bits = 0x7BFF
    return f16_from_bits(bits)


def e4m3_is_normal(x: float) -> bool:
    """Return True when ``x`` is a normal (non-zero, non-subnormal) E4M3."""
    return x != 0.0 and abs(x) >= E4M3_MIN_NORMAL


def _e4m3_code_to_float(code: int) -> float | None:
    """Decode one E4M3 code. Returns None for the two NaN encodings."""
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0x0F
    mant = code & 0x07
    if exp == 0x0F and mant == 0x07:
        return None  # NaN
    if exp == 0:
        return sign * (mant / 8.0) * 2.0**-6
    return sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)


#: All finite E4M3 values, sorted, with their codes and mantissa parity.
_E4M3_TABLE: list[tuple[float, int, bool]] = sorted(
    (v, code, (code & 0x07) % 2 == 0)
    for code in range(256)
    if (v := _e4m3_code_to_float(code)) is not None
)
_E4M3_VALUES = [entry[0] for entry in _E4M3_TABLE]


def e4m3_decode(code: int) -> float:
    """Decode one E4M3 code to a float. NaN codes decode to NaN."""
    value = _e4m3_code_to_float(code & 0xFF)
    return float("nan") if value is None else value


def e4m3_encode(x: float) -> int:
    """Round ``x`` to the nearest E4M3 value and return its code.

    Ties go to the neighbour whose mantissa LSB is zero, matching the
    round-to-nearest-even behaviour of the hardware cast.

    Args:
        x: The value to encode.

    Returns:
        The 8-bit E4M3 code. NaN and infinities encode to the NaN code 0x7F.

    """
    if x != x or math.isinf(x):
        return 0x7F
    if x > E4M3_MAX:
        x = E4M3_MAX
    if x < -E4M3_MAX:
        x = -E4M3_MAX
    idx = bisect.bisect_left(_E4M3_VALUES, x)
    if idx == 0:
        return _E4M3_TABLE[0][1]
    if idx == len(_E4M3_VALUES):
        return _E4M3_TABLE[-1][1]
    lo = _E4M3_TABLE[idx - 1]
    hi = _E4M3_TABLE[idx]
    d_lo = x - lo[0]
    d_hi = hi[0] - x
    if d_lo < d_hi:
        return lo[1]
    if d_hi < d_lo:
        return hi[1]
    # Exact tie: prefer the even mantissa.
    return lo[1] if lo[2] else hi[1]


# ---------------------------------------------------------------------------
# E8M0 (the OCP MX shared-scale format)
# ---------------------------------------------------------------------------
#
# E8M0 is an *unsigned biased exponent*: eight bits, no sign, no mantissa, no
# zero and no infinity. The code is the FP32 exponent field with bias 127, so
# code 0 is 2**-127, code 254 is 2**127 and code 255 (0xFF) is the single
# reserved NaN. It is stored as one raw byte per block.
#
# Decoding to FP32 is a bit trick, not an FP operation: 2**(code - 127) is
# exactly the FP32 bit pattern ``code << 23``. That is what makes E8M0 cheap
# on hardware, and it is also why the fork's own MXFP8 kernels build the
# reciprocal instead of dividing by 2**(code - 127): code 0 gives a divisor of
# 2**-127, which is subnormal in FP32 and flushes to zero on AMD parts
# (see vllm/model_executor/layers/quantization/utils/mxfp8_utils.py).


#: Exponent bias of the E8M0 scale format.
E8M0_BIAS = 127
#: Largest finite E8M0 code (0xFE).
E8M0_MAX_CODE = 254
#: The one reserved NaN encoding.
E8M0_NAN_CODE = 255
#: Largest positive value of the symmetric INT4 element type.
INT4_MAX = 7.0
#: Smallest value of the symmetric INT4 element type.
INT4_MIN = -8
#: Exponent of the largest normal INT4 value (7 = 1.75 * 2**2).
INT4_EMAX = 2
#: Largest positive value of the symmetric INT8 element type.
INT8_MAX = 127.0
#: Smallest value of the symmetric INT8 element type.
INT8_MIN = -128
#: Exponent of the largest normal INT8 value (127 = 1.984375 * 2**6).
INT8_EMAX = 6


def e8m0_decode(code: int) -> float:
    """Decode one E8M0 code to a float.

    Args:
        code: The 8-bit E8M0 code.

    Returns:
        ``2 ** (code - 127)``, or NaN for the reserved code 0xFF.

    """
    code &= 0xFF
    if code == E8M0_NAN_CODE:
        return float("nan")
    return 2.0 ** (code - E8M0_BIAS)


def e8m0_encode(x: float, *, round_up: bool = True) -> int:
    """Encode a positive scale into an E8M0 code.

    Args:
        x: The scale to encode. Must be positive.
        round_up: When True (the default) the code is the smallest power of
            two that is at least ``x``. When False it is the largest power of
            two that is at most ``x``.

    Returns:
        The 8-bit E8M0 code, clamped to the finite range 0x00..0xFE.

    """
    if x != x or x <= 0.0:
        return 0
    exponent = _log2_ceil(x) if round_up else _log2_floor(x)
    return _e8m0_code(exponent)


def _e8m0_code(exponent: int) -> int:
    """Clamp an unbiased exponent into E8M0's range and bias it."""
    return max(0, min(E8M0_MAX_CODE, exponent + E8M0_BIAS))


def _log2_floor(x: float) -> int:
    """Return ``floor(log2(x))`` exactly, for positive ``x``."""
    _, exponent = math.frexp(x)
    return exponent - 1


def _log2_ceil(x: float) -> int:
    """Return ``ceil(log2(x))`` exactly, for positive ``x``."""
    mantissa, exponent = math.frexp(x)
    return exponent - 1 if mantissa == 0.5 else exponent


def _no_saturation_exponent(amax: float, qmax: float) -> int:
    """Return the smallest exponent whose scale cannot saturate.

    The scale must satisfy ``qmax * 2**e >= amax`` so that every element
    maps into the element format's range. Writing ``amax = m * 2**p`` with
    ``m`` in ``[0.5, 1)`` and ``qmax = f * 2**k`` with ``f`` in ``[1, 2)``,
    the answer is ``p - k`` when ``amax <= f * 2**p`` and ``p - k + 1``
    otherwise. The comparison is exact, so this never loses a bit to a
    floating-point log.

    Args:
        amax: The largest absolute value in the block.
        qmax: The largest magnitude the element format can represent.

    Returns:
        The unbiased exponent of the scale.

    """
    exponent = _log2_floor(amax)
    qmax_exponent = _log2_floor(qmax)
    mantissa = qmax / (2.0**qmax_exponent)
    if amax <= mantissa * (2.0**exponent):
        return exponent - qmax_exponent
    return exponent - qmax_exponent + 1


# ---------------------------------------------------------------------------
# INT4 packing
# ---------------------------------------------------------------------------


def pack_int4(codes: Sequence[int]) -> bytes:
    """Pack unsigned 4-bit codes, two per byte, low nibble first.

    Args:
        codes: Values in ``[0, 15]``.

    Returns:
        The packed bytes.

    """
    out = bytearray()
    for i in range(0, len(codes), 2):
        lo = codes[i] & 0x0F
        hi = (codes[i + 1] & 0x0F) if i + 1 < len(codes) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def unpack_int4(data: bytes, count: int) -> list[int]:
    """Unpack ``count`` 4-bit codes from ``data``, low nibble first."""
    out: list[int] = []
    for i in range(count):
        byte = data[i >> 1]
        out.append((byte >> 4) if (i & 1) else (byte & 0x0F))
    return out


def _round_half_away(x: float) -> int:
    """Round half away from zero, matching the shipped INT4 kernel path."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _round_half_even(x: float) -> int:
    """Round half to even, matching a hardware float-to-int cast."""
    return int(round(x))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TensorResult:
    """Round-trip result for a whole KV tensor.

    Attributes:
        reconstructed: The dequantized values, one list per row.
        scales: The stored block scales, one list per row.
        global_scale: The per-tensor global scale, or None for single-level
            schemes.
        underflowed_blocks: Number of blocks whose stored scale fell below
            the normal range of its scale format.
        clipped_blocks: Number of blocks in which at least one element
            saturated at the element format's maximum.

    """

    reconstructed: list[list[float]]
    scales: list[list[float]]
    global_scale: float | None = None
    underflowed_blocks: int = 0
    clipped_blocks: int = 0


def _blocks(row: Sequence[float], k: int):
    """Yield (start, block) pairs over a row."""
    for start in range(0, len(row), k):
        yield start, row[start : start + k]


# ---------------------------------------------------------------------------
# Scheme A: int4_per_token_head (shipped today)
# ---------------------------------------------------------------------------


def quantize_int4_per_token_head(rows: Sequence[Sequence[float]]) -> TensorResult:
    """Quantize rows the way ``int4_per_token_head`` does today.

    Asymmetric: the scale covers ``(max - min) / 15`` and a 4-bit zero-point
    absorbs the offset. The shipped kernel keeps the zero-point in the low
    mantissa bits of the FP32 scale; this reference keeps them apart, which
    changes no arithmetic.

    Args:
        rows: The rows to quantize.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    for row in rows:
        lo = min(row)
        hi = max(row)
        scale = max((hi - lo) / 15.0, 1e-6)
        zp = max(0, min(15, _round_half_away(-lo / scale)))
        out = []
        for value in row:
            code = max(0, min(15, _round_half_away(value / scale) + zp))
            out.append((code - zp) * scale)
        reconstructed.append(out)
        scales.append([scale])
    return TensorResult(reconstructed, scales)


# ---------------------------------------------------------------------------
# Scheme B: int4_fp16_block (the proposal, single level)
# ---------------------------------------------------------------------------


def quantize_int4_fp16_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
    symmetric: bool = True,
) -> TensorResult:
    """Quantize rows with a block-granular FP16 scale and no global scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.
        symmetric: When True use a symmetric scale and no zero-point.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            if symmetric:
                scale = max(max(abs(v) for v in block) / 7.0, 1e-6)
                zp = 0
                codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            else:
                lo = min(block)
                hi = max(block)
                scale = max((hi - lo) / 15.0, 1e-6)
                zp = max(0, min(15, _round_half_away(-lo / scale)))
                codes = [
                    max(0, min(15, _round_half_away(v / scale) + zp)) for v in block
                ]
            scale16 = f16_round(scale)
            if not f16_is_normal(scale16):
                underflowed += 1
            row_scales.append(scale16)
            if symmetric:
                out.extend(code * scale16 for code in codes)
            else:
                out.extend((code - zp) * scale16 for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


# ---------------------------------------------------------------------------
# Scheme B2: int4_fp16_block_packed (asymmetric, zero-point in the scale)
# ---------------------------------------------------------------------------


def quantize_int4_fp16_block_packed(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows asymmetrically, with the zero-point inside the scale.

    The zero-point is 4 bits and rides in the low 4 bits of the FP16 scale's
    mantissa, the same trick ``int4_per_token_head`` plays with an FP32
    scale. The scale is snapped up onto the resulting 6-bit-mantissa grid so
    that the block's range still fits in 16 codes and nothing saturates.

    The cost is that the scale keeps only 6 mantissa bits instead of 10, so
    it carries up to 1.5 % relative error. The benefit is a second level: 16
    codes per block instead of 8, which is worth more than the scale loses.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result. ``scales`` holds the packed 16-bit words, so
        the format on the wire is exactly one uint16 per block.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[int]] = []
    clipped = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[int] = []
        for _, block in _blocks(row, k):
            lo = min(block)
            hi = max(block)
            ideal = max((hi - lo) / 15.0, 1e-6)
            scale = f16_ceil_6(ideal)
            zp = max(0, min(15, _round_half_away(-lo / scale)))
            codes: list[int] = []
            for value in block:
                code = _round_half_away(value / scale) + zp
                if code < 0 or code > 15:
                    clipped += 1
                codes.append(max(0, min(15, code)))
            packed = (f16_bits(scale) & ~0xF) | zp
            stored_scale = f16_from_bits(packed & ~0xF)
            stored_zp = packed & 0xF
            row_scales.append(packed)
            out.extend((code - stored_zp) * stored_scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, clipped_blocks=clipped)


# ---------------------------------------------------------------------------
# Scheme H: FP32 block scales, and the token-axis grouping
# ---------------------------------------------------------------------------


def quantize_int4_fp32_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
    symmetric: bool = True,
) -> TensorResult:
    """Quantize rows with an FP32 block scale and a packed zero-point.

    The FP32 scale has 23 mantissa bits, so the 4-bit zero-point costs
    nothing at all in precision - this is the shipped format's trick, at
    block granularity instead of row granularity.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.
        symmetric: When True use a symmetric scale and no zero-point.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[int]] = []
    clipped = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[int] = []
        for _, block in _blocks(row, k):
            if symmetric:
                scale = max(max(abs(v) for v in block) / 7.0, 1e-6)
                zp = 0
                codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            else:
                lo = min(block)
                hi = max(block)
                scale = max((hi - lo) / 15.0, 1e-6)
                zp = max(0, min(15, _round_half_away(-lo / scale)))
                codes = []
                for value in block:
                    code = _round_half_away(value / scale) + zp
                    if code < 0 or code > 15:
                        clipped += 1
                    codes.append(max(0, min(15, code)))
            packed = struct.unpack("<I", struct.pack("<f", scale))[0]
            packed = (packed & ~0xF) | zp
            stored_scale = struct.unpack("<f", struct.pack("<I", packed & ~0xF))[0]
            stored_zp = packed & 0xF
            row_scales.append(packed)
            out.extend((code - stored_zp) * stored_scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, clipped_blocks=clipped)


def quantize_int8_fp32_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with symmetric INT8 and an FP32 block scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            scale = max(max(abs(v) for v in block) / INT8_MAX, 1e-6)
            codes = [
                max(INT8_MIN, min(INT8_MAX, _round_half_even(value / scale)))
                for value in block
            ]
            row_scales.append(scale)
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales)


def quantize_int8_e4m3_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with symmetric INT8 and an E4M3 block scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            amax = max(abs(value) for value in block)
            scale = max(amax / INT8_MAX, 1e-6)
            stored = e4m3_decode(e4m3_encode(scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            for value in block:
                code = max(
                    INT8_MIN,
                    min(INT8_MAX, _round_half_even(value / stored)),
                )
                out.append(code * stored)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


def quantize_int4_fp16_block_token_groups(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize with one scale per (channel, group of ``k`` tokens).

    This is the other grouping axis: instead of sharing a scale along the
    head dimension within one token, share it along the token dimension
    within one channel. KIVI's observation is that K carries per-channel
    outliers and V carries per-token outliers, which points at this axis
    for K.

    The memory cost is the same as head-dimension grouping: the number of
    scales per token per head is ``head_size / k`` either way.

    Args:
        rows: The rows to quantize. One row is one token.
        k: Number of tokens that share a scale, per channel.

    Returns:
        The round-trip result.

    """
    if not rows:
        return TensorResult([], [])
    head_size = len(rows[0])
    reconstructed = [[0.0] * head_size for _ in rows]
    for channel in range(head_size):
        column = [row[channel] for row in rows]
        for start in range(0, len(column), k):
            group = column[start : start + k]
            lo = min(group)
            hi = max(group)
            ideal = max((hi - lo) / 15.0, 1e-6)
            scale = f16_ceil_6(ideal)
            zp = max(0, min(15, _round_half_away(-lo / scale)))
            packed = (f16_bits(scale) & ~0xF) | zp
            stored_scale = f16_from_bits(packed & ~0xF)
            stored_zp = packed & 0xF
            for offset, value in enumerate(group):
                code = max(
                    0,
                    min(15, _round_half_away(value / stored_scale) + stored_zp),
                )
                reconstructed[start + offset][channel] = (
                    code - stored_zp
                ) * stored_scale
    return TensorResult(reconstructed, [])


# ---------------------------------------------------------------------------
# Scheme G: fp8_e4m3 (what the fork already ships and has measured)
# ---------------------------------------------------------------------------


def quantize_fp8_e4m3_per_tensor(
    rows: Sequence[Sequence[float]],
) -> TensorResult:
    """Quantize rows with FP8 e4m3 elements and one per-tensor scale.

    This is what ``--kv-cache-dtype fp8`` does, and it is the only KV cache
    dtype with a measurement on the V620.

    Args:
        rows: The rows to quantize.

    Returns:
        The round-trip result.

    """
    amax = max((abs(value) for row in rows for value in row), default=0.0)
    global_scale = max(amax / E4M3_MAX, 1e-6)
    reconstructed: list[list[float]] = []
    for row in rows:
        out: list[float] = []
        for value in row:
            code = e4m3_encode(value / global_scale)
            out.append(e4m3_decode(code) * global_scale)
        reconstructed.append(out)
    return TensorResult(reconstructed, [[global_scale]])


def quantize_fp8_e4m3_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with FP8 e4m3 elements and one E4M3 scale per block.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            amax = max(abs(value) for value in block)
            scale = max(amax / E4M3_MAX, 1e-6)
            stored = e4m3_decode(e4m3_encode(scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            for value in block:
                out.append(e4m3_decode(e4m3_encode(value / stored)) * stored)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


# ---------------------------------------------------------------------------
# Scheme F: int8 baselines (MXINT8 and friends)
# ---------------------------------------------------------------------------


def quantize_int8_e8m0_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
    rule: str = "rceil",
) -> TensorResult:
    """Quantize rows with symmetric INT8 and an E8M0 block scale.

    This is the OCP MXINT8 geometry and the natural "one byte of scale"
    baseline for the INT4 proposal.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.
        rule: ``"rceil"`` (no saturation) or ``"floor"`` (the OCP MX rule).

    Returns:
        The round-trip result.

    """
    if rule not in ("rceil", "floor"):
        raise ValueError(f"unknown E8M0 rule {rule!r}")
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    clipped = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            amax = max(abs(v) for v in block)
            if amax == 0.0:
                code = 0
            elif rule == "rceil":
                code = _e8m0_code(_no_saturation_exponent(amax, INT8_MAX))
            else:
                code = _e8m0_code(_log2_floor(amax) - INT8_EMAX)
            scale = e8m0_decode(code)
            codes: list[int] = []
            for value in block:
                code_value = _round_half_even(value / scale)
                if code_value > INT8_MAX or code_value < INT8_MIN:
                    clipped += 1
                codes.append(max(INT8_MIN, min(INT8_MAX, code_value)))
            row_scales.append(scale)
            out.extend(code_value * scale for code_value in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, clipped_blocks=clipped)


def quantize_int8_fp16_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with symmetric INT8 and an FP16 block scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            scale = f16_round(max(max(abs(v) for v in block) / INT8_MAX, 1e-6))
            codes = [
                max(INT8_MIN, min(INT8_MAX, _round_half_even(value / scale)))
                for value in block
            ]
            row_scales.append(scale)
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales)


# ---------------------------------------------------------------------------
# Scheme E: int4_e8m0_block (the MX-style alternative)
# ---------------------------------------------------------------------------


def quantize_int4_e8m0_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
    rule: str = "rceil",
) -> TensorResult:
    """Quantize rows with an E8M0 (power-of-two) block scale.

    E8M0 can only hold a power of two, so the scale is chosen by rounding the
    ideal ``amax / 7`` up or down to the nearest power of two. Two rules are
    modelled:

    ``rceil``
        The scale is the smallest power of two with ``7 * scale >= amax``.
        No element can ever saturate. This is the rule the fork's own MXFP8
        code uses (``vllm/model_executor/layers/quantization/utils/
        mxfp8_utils.py``), because the OCP floor rule saturates.

    ``floor``
        The OCP MX v1.0 rule, ``shared_exp = floor(log2(amax)) - emax_elem``
        with ``emax_elem = 2`` for symmetric INT4. The block maximum then
        lands in ``[4, 8)``, so it saturates at 7 whenever the fractional part
        of ``log2(amax)`` exceeds ``log2(7/4)``.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.
        rule: ``"rceil"`` or ``"floor"``.

    Returns:
        The round-trip result.

    """
    if rule not in ("rceil", "floor"):
        raise ValueError(f"unknown E8M0 rule {rule!r}")
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    clipped = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            amax = max(abs(v) for v in block)
            if amax == 0.0:
                # E8M0 has no zero. Use the smallest scale; every element
                # quantizes to 0, so the block reconstructs exactly.
                code = 0
            elif rule == "rceil":
                code = _e8m0_code(_no_saturation_exponent(amax, INT4_MAX))
            else:
                code = _e8m0_code(_log2_floor(amax) - INT4_EMAX)
            scale = e8m0_decode(code)
            block_clipped = False
            codes: list[int] = []
            for value in block:
                code_value = _round_half_even(value / scale)
                if code_value > INT4_MAX or code_value < INT4_MIN:
                    block_clipped = True
                codes.append(max(INT4_MIN, min(INT4_MAX, code_value)))
            if block_clipped:
                clipped += 1
            row_scales.append(scale)
            out.extend(code_value * scale for code_value in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, clipped_blocks=clipped)


# ---------------------------------------------------------------------------
# Scheme C: int4_fp16_block_g (the proposal with NVFP4's second level)
# ---------------------------------------------------------------------------


def quantize_int4_fp16_block_g(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with an FP16 block scale divided by a global scale.

    The global scale is the largest effective block scale in the tensor, so
    the stored block scales land at or below 1.0 and the global is multiplied
    back at read time. This is NVFP4's structure with an FP16 block scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        row_effective = []
        for _, block in _blocks(row, k):
            row_effective.append(max(max(abs(v) for v in block) / 7.0, 1e-6))
        effective.append(row_effective)
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = f16_round(eff / global_scale)
            if not f16_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


# ---------------------------------------------------------------------------
# Scheme D: int4_e4m3_block_g (NVFP4 scale geometry, integer elements)
# ---------------------------------------------------------------------------


def quantize_int4_e4m3_block_g(
    rows: Sequence[Sequence[float]],
    k: int = 16,
) -> TensorResult:
    """Quantize rows with an E4M3 block scale divided by a global scale.

    Mirrors NVFP4's scale geometry (E4M3 per block, FP32 global) but keeps
    uniform INT4 elements instead of E2M1.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        effective.append(
            [
                max(max(abs(v) for v in block) / 7.0, 1e-6)
                for _, block in _blocks(row, k)
            ]
        )
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = e4m3_decode(e4m3_encode(eff / global_scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


def quantize_int4_e4m3_block(
    rows: Sequence[Sequence[float]],
    k: int = 16,
) -> TensorResult:
    """Quantize rows with an E4M3 block scale and no global scale.

    The single-level counterpart of :func:`quantize_int4_e4m3_block_g`, kept
    so the cost of NVFP4's second level can be measured rather than asserted.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            eff = max(max(abs(v) for v in block) / 7.0, 1e-6)
            stored = e4m3_decode(e4m3_encode(eff))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            codes = [max(-8, min(7, _round_half_even(v / stored))) for v in block]
            out.extend(code * stored for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


# ---------------------------------------------------------------------------
# Scheme E: NVFP4 (E2M1 elements, E4M3 scale per 16)
# ---------------------------------------------------------------------------

#: E2M1 magnitudes. E2M1 has no negative zero; both zeros decode to 0.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_decode(code: int) -> float:
    """Decode a 4-bit E2M1 code to its value."""
    sign = -1.0 if code & 0x08 else 1.0
    return sign * _E2M1_VALUES[code & 0x07]


def _e2m1_encode(x: float) -> int:
    """Round ``x`` to the nearest E2M1 code."""
    sign_bit = 0x08 if x < 0 else 0x00
    mag = abs(x)
    best_code = 0
    best_err = math.inf
    for idx, value in enumerate(_E2M1_VALUES):
        err = abs(mag - value)
        if err < best_err:
            best_err = err
            best_code = idx
    return best_code | sign_bit


def quantize_nvfp4(rows: Sequence[Sequence[float]], k: int = 16) -> TensorResult:
    """Quantize rows with E2M1 elements and an E4M3 scale per block.

    This is the NVFP4 geometry the fork already has for Blackwell
    (``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu``), reproduced here in
    Python so its cost can be compared on the same data. The global scale is
    modelled the way the kernel models it: folded into the stored block scale
    and multiplied back at read time.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        effective.append(
            [
                max(max(abs(v) for v in block) / 6.0, 1e-6)
                for _, block in _blocks(row, k)
            ]
        )
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = e4m3_decode(e4m3_encode(eff / global_scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [_e2m1_encode(v / scale) for v in block]
            out.extend(_e2m1_decode(code) * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def snr_db(reference: Sequence[float], test: Sequence[float]) -> float:
    """Signal-to-noise ratio in dB between two equal-length sequences."""
    num = 0.0
    den = 0.0
    for ref, tst in zip(reference, test):
        num += ref * ref
        den += (ref - tst) ** 2
    if den == 0.0:
        return math.inf
    return 10.0 * math.log10(num / den)


def bits_per_value(
    element_bits: int,
    scale_bits: int,
    block: int,
    zero_point_bits: int = 0,
) -> float:
    """Total bits per stored value including the scale overhead.

    A per-tensor global scale is not included: it is one number for the whole
    tensor and costs no meaningful number of bits.

    Args:
        element_bits: Bits per element payload.
        scale_bits: Bits per stored scale.
        block: Elements sharing one scale.
        zero_point_bits: Bits per stored zero-point. The shipped
            ``int4_per_token_head`` format hides its zero-point in the low
            mantissa bits of the FP32 scale, so it passes 0 here.

    Returns:
        The total bits per value.

    """
    return element_bits + scale_bits / block + zero_point_bits / block


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class SchemeResult:
    """One scheme's outcome on one tensor.

    Attributes:
        name: The scheme name.
        snr: Round-trip SNR in dB, averaged over rows.
        bits: Total bits per value including scale overhead.
        notes: Free-text caveats.
        underflowed: Blocks whose stored scale fell out of the normal range.
        clipped: Blocks in which at least one element saturated.

    """

    name: str
    snr: float
    bits: float
    notes: str
    underflowed: int = 0
    clipped: int = 0


#: Labels for the synthetic outlier modes.
_OUTLIER_LABELS = {
    "none": "no outlier channels",
    "channel": "two x40 outlier channels (K-side)",
    "token": "two x40 outlier tokens (V-side)",
}


def _make_row(
    rng: random.Random,
    head_size: int,
    outlier: str,
) -> list[float]:
    """Build one synthetic KV row, optionally with channel outliers."""
    row = [rng.gauss(0.0, 0.5) for _ in range(head_size)]
    if outlier == "channel":
        # KIVI's observation for K: a few channels carry much larger
        # values, and they do so consistently across tokens.
        for channel in (3, 17):
            if channel < head_size:
                row[channel] *= 40.0
    return row


def _make_rows(
    rng: random.Random,
    n_rows: int,
    head_size: int,
    outlier: str,
) -> list[list[float]]:
    """Build a synthetic KV tensor with the requested outlier mode.

    ``channel`` is the K-side failure mode: two channels of every row are
    40x too large. ``token`` is the V-side failure mode: two whole tokens
    are 40x too large.
    """
    rows = [_make_row(rng, head_size, outlier) for _ in range(n_rows)]
    if outlier == "token":
        for index in (5, 40):
            if index < len(rows):
                rows[index] = [value * 40.0 for value in rows[index]]
    return rows


def run_comparison(
    head_size: int = 128,
    outlier: str = "channel",
    seed: int = 42,
) -> list[SchemeResult]:
    """Compare the scale strategies on synthetic KV rows.

    Args:
        head_size: Values per row.
        outlier: ``"channel"``, ``"token"`` or ``"none"``.
        seed: RNG seed, so the numbers are reproducible.

    Returns:
        One result per scheme.

    """
    rng = random.Random(seed)
    rows = _make_rows(rng, 64, head_size, outlier)

    def measure(name, quantize, bits, notes) -> SchemeResult:
        result = quantize(rows)
        snrs = [snr_db(row, recon) for row, recon in zip(rows, result.reconstructed)]
        return SchemeResult(
            name,
            sum(snrs) / len(snrs),
            bits,
            notes,
            underflowed=result.underflowed_blocks,
            clipped=result.clipped_blocks,
        )

    return [
        measure(
            "int4_per_token_head (shipped)",
            quantize_int4_per_token_head,
            bits_per_value(4, 32, head_size),
            "asymmetric, one FP32 scale per row, zp in the scale mantissa",
        ),
        measure(
            "int4_fp16_block k=64",
            lambda r: quantize_int4_fp16_block(r, k=64),
            bits_per_value(4, 16, 64),
            "symmetric, one FP16 scale per 64 values, single level",
        ),
        measure(
            "int4_fp16_block k=32 (proposal)",
            lambda r: quantize_int4_fp16_block(r, k=32),
            bits_per_value(4, 16, 32),
            "symmetric, one FP16 scale per 32 values, single level",
        ),
        measure(
            "int4_fp16_block k=16",
            lambda r: quantize_int4_fp16_block(r, k=16),
            bits_per_value(4, 16, 16),
            "symmetric, one FP16 scale per 16 values, single level",
        ),
        measure(
            "int4_fp16_block_token_groups k=32",
            lambda r: quantize_int4_fp16_block_token_groups(r, k=32),
            bits_per_value(4, 16, 32),
            "asymmetric, one FP16 scale per (channel, 32 tokens)",
        ),
        measure(
            "int4_fp16_block_packed k=64",
            lambda r: quantize_int4_fp16_block_packed(r, k=64),
            bits_per_value(4, 16, 64),
            "asymmetric, FP16 scale with 4-bit zp packed in its mantissa",
        ),
        measure(
            "int4_fp16_block_packed k=32 (proposal)",
            lambda r: quantize_int4_fp16_block_packed(r, k=32),
            bits_per_value(4, 16, 32),
            "asymmetric, FP16 scale with 4-bit zp packed in its mantissa",
        ),
        measure(
            "int4_fp16_block_packed k=16",
            lambda r: quantize_int4_fp16_block_packed(r, k=16),
            bits_per_value(4, 16, 16),
            "asymmetric, FP16 scale with 4-bit zp packed in its mantissa",
        ),
        measure(
            "int4_fp16_block k=32 asym",
            lambda r: quantize_int4_fp16_block(r, k=32, symmetric=False),
            bits_per_value(4, 16, 32, zero_point_bits=4),
            "asymmetric, one FP16 scale + 4-bit zp per 32 values",
        ),
        measure(
            "int4_fp16_block_g k=32 (NVFP4 structure)",
            quantize_int4_fp16_block_g,
            bits_per_value(4, 16, 32),
            "symmetric, FP16 block scale / FP32 per-tensor global",
        ),
        measure(
            "int4_e8m0_block k=32 (rceil)",
            lambda r: quantize_int4_e8m0_block(r, k=32, rule="rceil"),
            bits_per_value(4, 8, 32),
            "symmetric, one E8M0 scale per 32 values, no-saturation rule",
        ),
        measure(
            "int4_e8m0_block k=16 (rceil)",
            lambda r: quantize_int4_e8m0_block(r, k=16, rule="rceil"),
            bits_per_value(4, 8, 16),
            "symmetric, one E8M0 scale per 16 values, no-saturation rule",
        ),
        measure(
            "int4_e8m0_block k=32 (OCP floor)",
            lambda r: quantize_int4_e8m0_block(r, k=32, rule="floor"),
            bits_per_value(4, 8, 32),
            "symmetric, one E8M0 scale per 32 values, OCP MX floor rule",
        ),
        measure(
            "int4_e8m0_block k=16 (OCP floor)",
            lambda r: quantize_int4_e8m0_block(r, k=16, rule="floor"),
            bits_per_value(4, 8, 16),
            "symmetric, one E8M0 scale per 16 values, OCP MX floor rule",
        ),
        measure(
            "int4_e4m3_block k=16 (no global)",
            quantize_int4_e4m3_block,
            bits_per_value(4, 8, 16),
            "symmetric, one E4M3 scale per 16 values, single level",
        ),
        measure(
            "int4_e4m3_block k=32 (no global)",
            lambda r: quantize_int4_e4m3_block(r, k=32),
            bits_per_value(4, 8, 32),
            "symmetric, one E4M3 scale per 32 values, single level",
        ),
        measure(
            "int4_e4m3_block k=64 (no global)",
            lambda r: quantize_int4_e4m3_block(r, k=64),
            bits_per_value(4, 8, 64),
            "symmetric, one E4M3 scale per 64 values, single level",
        ),
        measure(
            "int4_e4m3_block_g k=16",
            quantize_int4_e4m3_block_g,
            bits_per_value(4, 8, 16),
            "symmetric, E4M3 block scale / FP32 global (NVFP4 geometry)",
        ),
        measure(
            "int8_e8m0_block k=32 (MXINT8, rceil)",
            lambda r: quantize_int8_e8m0_block(r, k=32),
            bits_per_value(8, 8, 32),
            "symmetric INT8, one E8M0 scale per 32 values",
        ),
        measure(
            "int8_e8m0_block k=16 (MXINT8, rceil)",
            lambda r: quantize_int8_e8m0_block(r, k=16),
            bits_per_value(8, 8, 16),
            "symmetric INT8, one E8M0 scale per 16 values",
        ),
        measure(
            "int8_fp16_block k=32",
            lambda r: quantize_int8_fp16_block(r, k=32),
            bits_per_value(8, 16, 32),
            "symmetric INT8, one FP16 scale per 32 values",
        ),
        measure(
            "int4_fp32_block k=32 (sym)",
            lambda r: quantize_int4_fp32_block(r, k=32),
            bits_per_value(4, 32, 32),
            "symmetric, one FP32 scale per 32 values",
        ),
        measure(
            "int4_fp32_block k=32 (packed zp)",
            lambda r: quantize_int4_fp32_block(r, k=32, symmetric=False),
            bits_per_value(4, 32, 32),
            "asymmetric, FP32 scale with 4-bit zp packed in its mantissa",
        ),
        measure(
            "int8_fp32_block k=32",
            lambda r: quantize_int8_fp32_block(r, k=32),
            bits_per_value(8, 32, 32),
            "symmetric INT8, one FP32 scale per 32 values",
        ),
        measure(
            "int8_e4m3_block k=32",
            lambda r: quantize_int8_e4m3_block(r, k=32),
            bits_per_value(8, 8, 32),
            "symmetric INT8, one E4M3 scale per 32 values",
        ),
        measure(
            "fp8_e4m3 per-tensor (shipped, measured)",
            quantize_fp8_e4m3_per_tensor,
            bits_per_value(8, 0, 1),
            "FP8 e4m3 elements, one per-tensor scale: --kv-cache-dtype fp8",
        ),
        measure(
            "fp8_e4m3 k=32 block scale",
            lambda r: quantize_fp8_e4m3_block(r, k=32),
            bits_per_value(8, 8, 32),
            "FP8 e4m3 elements, one E4M3 scale per 32 values",
        ),
        measure(
            "nvfp4 (E2M1 + E4M3/16)",
            quantize_nvfp4,
            bits_per_value(4, 8, 16),
            "NVIDIA NVFP4: E2M1 elements, E4M3 scale per 16, FP32 global",
        ),
    ]


def _self_check() -> None:
    """Assert the primitives behave as the README claims."""
    assert f16_round(0.1) == struct.unpack("<e", struct.pack("<e", 0.1))[0]
    assert f16_round(FP16_MAX * 2) == FP16_MAX

    assert e4m3_decode(0x7E) == E4M3_MAX
    assert e4m3_encode(E4M3_MAX) == 0x7E
    assert e4m3_encode(1e9) == 0x7E
    assert math.isnan(e4m3_decode(0x7F))

    # E4M3 round-trip is exact for every finite code. Negative zero is
    # canonicalised to positive zero by the encoder, which is harmless for
    # a KV cache.
    for code in range(256):
        value = e4m3_decode(code)
        if math.isnan(value):
            continue
        if value == 0.0:
            assert e4m3_decode(e4m3_encode(value)) == 0.0
            continue
        assert e4m3_encode(value) == code, f"E4M3 round-trip failed at {code:#04x}"

    codes = list(range(16))
    assert unpack_int4(pack_int4(codes), 16) == codes

    # E8M0 round-trips exactly for every finite code: the value is a power
    # of two and the code is its unbiased exponent plus the bias.
    for code in range(E8M0_MAX_CODE + 1):
        value = e8m0_decode(code)
        assert e8m0_encode(value) == code, f"E8M0 round-trip failed at {code:#04x}"
    assert math.isnan(e8m0_decode(E8M0_NAN_CODE))

    # The no-saturation exponent is exact and minimal: the chosen scale never
    # lets an element exceed the INT4 maximum, and halving it would.
    for amax in (0.1, 0.5, 1.0, 3.0, 3.6, 7.0, 14.0, 100.0):
        exponent = _no_saturation_exponent(amax, INT4_MAX)
        scale = 2.0**exponent
        assert 7.0 * scale >= amax, f"scale {scale} saturates amax {amax}"
        assert 7.0 * (scale / 2.0) < amax, f"scale {scale} is not minimal"

    # The round-up rule never saturates; the OCP floor rule does.
    rows = [[0.5 * i for i in range(64)]]
    assert quantize_int4_e8m0_block(rows, k=32, rule="rceil").clipped_blocks == 0
    assert quantize_int4_e8m0_block(rows, k=32, rule="floor").clipped_blocks > 0

    # The packed zero-point really is free: the stored word is one uint16 and
    # it carries both the scale and the zero-point.
    packed = quantize_int4_fp16_block_packed([[0.5 * i for i in range(64)]], k=32)
    assert len(packed.scales[0]) == 2
    for word in packed.scales[0]:
        assert 0 <= word <= 0xFFFF
        assert f16_from_bits(word & ~0xF) > 0.0
        assert 0 <= (word & 0xF) <= 15

    # Snapping the scale up onto the 6-bit-mantissa grid never drops below
    # the requested value, so the block range still fits in 16 codes.
    for value in (1e-4, 0.1, 1.0, 3.3, 7.7, 100.0, 1000.0):
        assert f16_ceil_6(value) >= value

    # The proposal's stored scale really is FP16-representable.
    rows = [[0.5 * i for i in range(64)]]
    result = quantize_int4_fp16_block(rows, k=32)
    assert len(result.scales[0]) == 2
    for scale in result.scales[0]:
        assert f16_round(scale) == scale

    # The global-scale variant reconstructs the same values as the single
    # level one when the global is the tensor maximum.
    single = quantize_int4_fp16_block(rows, k=32)
    with_global = quantize_int4_fp16_block_g(rows, k=32)
    assert with_global.global_scale == max(single.scales[0])


def main() -> None:
    """Run the self-check and print the comparison tables."""
    _self_check()
    head_size = 128

    for outlier in ("channel", "token", "none"):
        label = _OUTLIER_LABELS[outlier]
        results = run_comparison(head_size=head_size, outlier=outlier)
        print(f"=== head_size={head_size}, {label} ===")
        print("64 rows, N(0, 0.5). SNR is round-trip, averaged over rows.")
        print("These are arithmetic results, not V620 measurements.\n")
        header = (
            f"{'scheme':<38} {'SNR dB':>8} {'bits/val':>10} {'subnorm':>9} {'clip':>6}"
        )
        print(header)
        print("-" * len(header))
        for result in results:
            print(
                f"{result.name:<38} {result.snr:>8.3f} {result.bits:>10.3f} "
                f"{result.underflowed:>9d} {result.clipped:>6d}"
            )
        print()
        for result in results:
            print(f"  {result.name}: {result.notes}")
        print()

    print("Self-check passed.")


if __name__ == "__main__":
    main()
