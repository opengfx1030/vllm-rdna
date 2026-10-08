// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Scalar conversion helpers for the csrc/rocm/rdna/** kernels.
//
// gfx1030 has no bf16 or fp8 hardware conversions, and the software paths in
// <hip/hip_fp8.h> differ between ROCm releases. These helpers are explicit
// bit-level conversions with fixed rounding, so cache rows written on RDNA
// are byte-identical to a torch reference regardless of the toolchain:
//
//   fp16   <-> fp32   round-to-nearest-even (hardware v_cvt_f16_f32)
//   bf16   <-> fp32   round-to-nearest-even, NaN kept quiet
//   e4m3fn <-  fp32   OCP E4M3 (bias 7, no inf, max 448), RNE, saturating
//   e4m3fn  -> fp32
//   e8m0   <-> fp32   UE8M0 power-of-two scale (bias 127, 0xFF = NaN)
//
// Everything is `__host__ __device__` so host-side reference code and unit
// tests can share the definitions.

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace vllm {
namespace rdna {

#define VLLM_RDNA_HD __host__ __device__ __forceinline__

VLLM_RDNA_HD uint32_t float_as_u32(float f) {
  return __builtin_bit_cast(uint32_t, f);
}
VLLM_RDNA_HD float u32_as_float(uint32_t u) {
  return __builtin_bit_cast(float, u);
}

// ── fp16 ────────────────────────────────────────────────────────────────────
VLLM_RDNA_HD float f16_bits_to_float(uint16_t h) {
  return static_cast<float>(__builtin_bit_cast(_Float16, h));
}
VLLM_RDNA_HD uint16_t float_to_f16_bits(float f) {
  return __builtin_bit_cast(uint16_t, static_cast<_Float16>(f));
}

// ── bf16 ────────────────────────────────────────────────────────────────────
VLLM_RDNA_HD float bf16_bits_to_float(uint16_t b) {
  return u32_as_float(static_cast<uint32_t>(b) << 16);
}
VLLM_RDNA_HD uint16_t float_to_bf16_bits(float f) {
  uint32_t const u = float_as_u32(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN: keep sign, force quiet
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  uint32_t const rounding_bias = 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// 16-bit float storage tags. `bits` is the raw storage; `to_float` /
// `from_float` are the RNE conversions above.
struct Fp16 {
  static VLLM_RDNA_HD float to_float(uint16_t b) {
    return f16_bits_to_float(b);
  }
  static VLLM_RDNA_HD uint16_t from_float(float f) {
    return float_to_f16_bits(f);
  }
  // Round-trip through the storage type (used to mirror kernels that
  // round an fp32 intermediate to the activation dtype).
  static VLLM_RDNA_HD float round(float f) { return to_float(from_float(f)); }
};
struct Bf16 {
  static VLLM_RDNA_HD float to_float(uint16_t b) {
    return bf16_bits_to_float(b);
  }
  static VLLM_RDNA_HD uint16_t from_float(float f) {
    return float_to_bf16_bits(f);
  }
  static VLLM_RDNA_HD float round(float f) { return to_float(from_float(f)); }
};

// ── fp8 e4m3fn (OCP) ────────────────────────────────────────────────────────
constexpr float kFp8E4m3Max = 448.0f;

VLLM_RDNA_HD float fp8_e4m3fn_to_float(uint8_t v) {
  uint32_t const sign = static_cast<uint32_t>(v & 0x80u) << 24;
  uint32_t const exp = (v >> 3) & 0xfu;
  uint32_t const man = v & 0x7u;
  if (exp == 0xfu && man == 0x7u) return u32_as_float(sign | 0x7fc00000u);
  float mag;
  if (exp == 0) {
    mag = static_cast<float>(man) * 0.001953125f;  // man * 2^-9
  } else {
    mag = u32_as_float(((exp + 120u) << 23) | (man << 20));
  }
  return u32_as_float(sign | float_as_u32(mag));
}

// Saturating RNE float -> e4m3fn. |x| >= 448 (and +-inf) map to +-448; NaN
// maps to 0x7f. -0.0 keeps its sign bit (0x80), like torch.
VLLM_RDNA_HD uint8_t float_to_fp8_e4m3fn(float x) {
  uint32_t const u = float_as_u32(x);
  uint8_t const sign = static_cast<uint8_t>((u >> 24) & 0x80u);
  uint32_t const abs_u = u & 0x7fffffffu;
  if (abs_u > 0x7f800000u) return 0x7f;  // NaN
  float const a = u32_as_float(abs_u);
  if (a >= kFp8E4m3Max) return sign | 0x7e;
  if (a < 0.015625f) {  // below the smallest normal (2^-6): 2^-9 steps
    // a * 2^9 is exact; rintf rounds half to even.
    return sign | static_cast<uint8_t>(rintf(a * 512.0f));
  }
  // Normal: round the 23-bit mantissa to 3 bits (RNE), carry into exponent.
  uint32_t const r = (abs_u + 0x7ffffu + ((abs_u >> 20) & 1u)) >> 20;
  uint32_t const code = r - (120u << 3);  // rebias 127 -> 7
  return sign | static_cast<uint8_t>(code > 0x7eu ? 0x7eu : code);
}

// ── e8m0 (UE8M0 power-of-two scale) ────────────────────────────────────────
VLLM_RDNA_HD float e8m0_to_float(uint8_t e) {
  if (e == 0xffu) return u32_as_float(0x7fc00000u);
  if (e == 0) return u32_as_float(0x00400000u);  // 2^-127 (subnormal)
  return u32_as_float(static_cast<uint32_t>(e) << 23);
}

// ceil(log2(x)) for a positive, normal, finite fp32 `x`, computed exactly
// from the bit pattern (device log2f is approximate near powers of two).
VLLM_RDNA_HD int ceil_log2_exact(float x) {
  uint32_t const u = float_as_u32(x);
  int const e = static_cast<int>((u >> 23) & 0xffu) - 127;
  return (u & 0x7fffffu) ? e + 1 : e;
}

// Encode an unbiased power-of-two exponent as UE8M0, clamped to [0, 254]
// (255 is NaN, never produced).
VLLM_RDNA_HD uint8_t e8m0_from_exponent(int exponent) {
  int const biased = exponent + 127;
  return static_cast<uint8_t>(biased < 0 ? 0 : (biased > 254 ? 254 : biased));
}

#undef VLLM_RDNA_HD

}  // namespace rdna
}  // namespace vllm
