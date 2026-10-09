// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// FP8 (E4M3) weight dequantization primitives for RDNA2 (gfx1030) W8A16-FP8
// kernels. Inline bit-trick conversion (no LUT, no __constant__, no host init).
//
// gfx1030 (RDNA2) has NO FP8 hardware support. We convert FP8 -> fp16
// inline using the same bit-trick pattern as the W4A16 dequant function
// (qdq_4_rdna2.cuh): shift exponent, shift mantissa, combine into fp16
// bits. This avoids the cross-TU __constant__ visibility issues that
// plagued the previous LUT-based design.
//
// The OCP FP8 E4M3 bit layout (from "FP8 Formats for Deep Learning",
// Micikevicius et al. 2022, arXiv:2209.05433):
//   bit 7: sign | bits 6-3: exponent (bias 7) | bits 2-0: mantissa
// fp16 layout: 1 sign | 5 exp | 10 mant
// For normal fp8: fp16_exp = fp8_exp + 8 (shift bias 7 -> 15)
//                 fp16_mant = fp8_mant << 7

#ifndef _QDQ_FP8_RDNA2_CUH
#define _QDQ_FP8_RDNA2_CUH

#include <cstdint>

// Inline FP8 (E4M3) byte -> fp16 conversion. Uses the same bit-trick pattern
// as qdq_4_rdna2.cuh:dequant_4bit_8_fp16 (exllamav2 style).
//
// Returns the fp16 raw bits for one OCP E4M3 (e4m3fn) byte. E4M3 has no
// Inf: exponent 0xF is a normal binade (256..448) except S.1111.111, the only
// NaN.
//
// Branch-free: the 7 magnitude bits placed at fp16 bit 7 are the fp16
// encoding of value * 2^-8 for normals and subnormals alike (the exponent
// rebias 7 -> 15 is a power of two), so one exact fp16 multiply by 2^8
// restores the value. The NaN code is selected, not branched. This replaced
// a per-byte if/else ladder that dominated the RDNA2 fp8 GEMMs.
__forceinline__ __device__ uint16_t fp8_e4m3_to_fp16_bits(uint8_t fp8) {
  const uint32_t mag = fp8 & 0x7Fu;
  const half scaled = __hmul(__ushort_as_half((uint16_t)(mag << 7)),
                             __ushort_as_half((uint16_t)0x5C00));  // 256
  uint16_t bits = __half_as_ushort(scaled);
  bits = mag == 0x7Fu ? (uint16_t)0x7E00 : bits;  // NaN
  return (uint16_t)(bits | ((uint16_t)(fp8 & 0x80u) << 8));
}

namespace vllm {
namespace w8a16_fp8_rdna2 {

// Dequantize 8 FP8 weights (one uint64 word) into 4 half2 pairs.
// qa[0..7] are 8 FP8 weights packed as bytes 0..7 of a uint64.
// sh/zh carry this N col's scale and zero (caller fetches per N col).
// dq[0..3] are 4 half2 pairs (= 8 fp16 values) suitable for dot22_8_f.
//
// Per-tile conversion: 8 inline fp8->fp16 conversions + 4 hfma2 per K-iter.
// Same shape as the W8A16 INT8 dequant (qdq_8_rdna2.cuh:dequant_8bit_8_fp16)
// so an outer kernel can dispatch by dequant-function pointer.
__forceinline__ __device__ void dequant_fp8_8_fp16(uint64_t qa, half2 sh,
                                                   half2 zh, half2 (&dq)[4]) {
  half2 w01 = __halves2half2(
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)(qa & 0xFFu))),
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 8) & 0xFFu))));
  half2 w23 = __halves2half2(
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 16) & 0xFFu))),
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 24) & 0xFFu))));
  half2 w45 = __halves2half2(
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 32) & 0xFFu))),
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 40) & 0xFFu))));
  half2 w67 = __halves2half2(
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 48) & 0xFFu))),
      __ushort_as_half(fp8_e4m3_to_fp16_bits((uint8_t)((qa >> 56) & 0xFFu))));

  dq[0] = __hfma2(w01, sh, zh);
  dq[1] = __hfma2(w23, sh, zh);
  dq[2] = __hfma2(w45, sh, zh);
  dq[3] = __hfma2(w67, sh, zh);
}

}  // namespace w8a16_fp8_rdna2
}  // namespace vllm

#endif  // _QDQ_FP8_RDNA2_CUH