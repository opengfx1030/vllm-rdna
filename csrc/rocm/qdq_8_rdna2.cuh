// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// INT8 weight dequantization primitives for RDNA2 (gfx1030) W8A16 kernels.
//
// Used by:
//   * csrc/rocm/moe_w8a16_rdna2.cu (fused MoE kernel)
//
// Storage format: weights are stored as int8 (1 byte per element); no nibble
// packing. The kernel reads 8 int8s as a uint64, converts each to fp16,
// applies per-group scale and zero, then dot-products against fp16
// activations via v_dot2_f32_f16 (the same primitive as W4A16).
//
// Activation dtype is fp16; scale dtype is fp16; zero dtype is fp16.

#ifndef _QDQ_8_RDNA2_CUH
#define _QDQ_8_RDNA2_CUH

#include <cstdint>


namespace vllm {
namespace w8a16_rdna2 {

// ---------------------------------------------------------------------------
// fp16 path
// ---------------------------------------------------------------------------

// Precompute scale-baked (z, scale) pair for a single zero/scale.
__forceinline__ __device__ void prep_scale_zero_int8(
    int8_t zero, half scale,
    half2 (&zh)[1], half2 (&sh)[1]) {
  zh[0] = __half2half2(__int2half_rn((int)zero));
  sh[0] = __half2half2(scale);
}

// Dequantize 8 int8 weights (one uint64 word) into 4 half2 pairs.
//   qa[0..7] are 8 int8 weights packed as bytes 0..7 of a uint64.
//   dq[0..3] are 4 half2 pairs (= 8 fp16 values) suitable for dot22_8_f.
//
// Conversion: each int8 -> fp16 via __int2half_rn (round-to-nearest),
// then grouped into half2 for the 2-wide dot.
// Scale + zero are applied as fp16 FMA (storage precision: fp16).
__forceinline__ __device__ void dequant_8bit_8_fp16(
    uint64_t qa, half2 sh, half2 zh,
    half2 (&dq)[4]) {
  half2 w01 = __halves2half2(
      __int2half_rn((int)(int8_t)(qa & 0xFFu)),
      __int2half_rn((int)(int8_t)((qa >> 8) & 0xFFu)));
  half2 w23 = __halves2half2(
      __int2half_rn((int)(int8_t)((qa >> 16) & 0xFFu)),
      __int2half_rn((int)(int8_t)((qa >> 24) & 0xFFu)));
  half2 w45 = __halves2half2(
      __int2half_rn((int)(int8_t)((qa >> 32) & 0xFFu)),
      __int2half_rn((int)(int8_t)((qa >> 40) & 0xFFu)));
  half2 w67 = __halves2half2(
      __int2half_rn((int)(int8_t)((qa >> 48) & 0xFFu)),
      __int2half_rn((int)(int8_t)((qa >> 56) & 0xFFu)));
  dq[0] = __hfma2(w01, sh, zh);
  dq[1] = __hfma2(w23, sh, zh);
  dq[2] = __hfma2(w45, sh, zh);
  dq[3] = __hfma2(w67, sh, zh);
}

// Variant: per-channel scale and zero passed as raw fp16 (no dequant vector
// precomputed). Equivalent to dequant_8bit_8_fp16 but with one fewer
// register pressure. Used when group_size == size_n (per-channel quant).
__forceinline__ __device__ void dequant_8bit_8_fp16_per_ch(
    uint64_t qa, half scale, half zero,
    half2 (&dq)[4]) {
  half2 s = __half2half2(scale);
  half2 z = __half2half2(zero);
  dequant_8bit_8_fp16(qa, s, z, dq);
}

}  // namespace w8a16_rdna2
}  // namespace vllm

#endif  // _QDQ_8_RDNA2_CUH
