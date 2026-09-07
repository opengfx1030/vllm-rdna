// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 AWQ dequant for RDNA2 (gfx1030), fp16 only, scalar per-nibble.
//
// Unlike the GPTQ kernel which dequantizes 8 nibbles at once using the
// exllama half2 bit-trick, this processes one nibble at a time. Each
// nibble is widened to fp32 and the dequant FMA is done in fp32:
//   w_fp32 = (nibble - zero) * scale_f

#ifndef _qdq_awq_rdna2_cuh
#define _qdq_awq_rdna2_cuh

#include <cstdint>
#include <hip/hip_fp16.h>

namespace vllm {
namespace awq_rdna2 {

// Precompute fp32 dequant constants for one zero/scale pair.
//   z_f = -zero * scale_f
//   y_f = scale_f
// Then: w_fp32 = nibble * y_f + z_f = (nibble - zero) * scale_f
__forceinline__ __device__ void prep_zero_scale_fp32(uint32_t zero, half scale,
                                                       float& z_f, float& y_f) {
  float scale_f = __half2float(scale);
  z_f = -(float)zero * scale_f;
  y_f = scale_f;
}

// Dequantize a single 4-bit nibble to fp32 using precomputed constants.
__forceinline__ __device__ float dequant_nibble(int nibble, float z_f, float y_f) {
  return __fmaf_rn((float)nibble, y_f, z_f);
}

// Pair-wise fp16 to fp32 widening: 2 fp16 -> 2 fp32 in one call.
__forceinline__ __device__ void widen_half2_to_float2(half2 h, float& f0, float& f1) {
  float2 f2 = __half22float2(h);
  f0 = f2.x;
  f1 = f2.y;
}

} // namespace awq_rdna2
} // namespace vllm

#endif // _qdq_awq_rdna2_cuh
