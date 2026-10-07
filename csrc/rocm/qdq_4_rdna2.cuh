// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA2 (gfx1030),
// using the activation/scale dtype (half).
// The fp16 path reuses the classic exllamav2 bit-trick:
//
//   (qa & 0x000F000F) | 0x64006400  ->  half2(1024+q_lo, 1024+q_hi)
//   (qa & 0x00F000F0) | 0x64006400  ->  half2(1024+q_lo*16, 1024+q_hi*16)
//
// The "*16 then divide by 16 in the FMA" trick for the upper-nibble pairs
// works in fp16 because the mantissa (10 bits) is wide enough to hold a value
// shifted by 4 bits.

#ifndef _qdq_4_rdna2_cuh
#define _qdq_4_rdna2_cuh

#include <cstdint>

#include <hip/hip_fp16.h>

namespace vllm {
namespace gptq_rdna2 {

// ---------------------------------------------------------------------------
// fp16 path
// ---------------------------------------------------------------------------

// Precompute scale-baked constants for a single zero/scale pair.
//   z1z16[0] = scale * (-1024 - zero)            (used for "low" pairs)
//   z1z16[1] = scale * (-64   - zero)            (used for "high" pairs)
//   y1y16[0] = scale * 1                          (low pairs are q + 1024)
//   y1y16[1] = scale * (1/16)                     (high pairs are q*16 + 1024)
__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  // half(-1024 - zero) via the exllamav2 bit-trick:
  //   half bits 0xE400 == -1024.0 ; ORing the zero into mantissa subtracts it.
  union {
    uint16_t u;
    half h;
  } z1u;
  z1u.u = (uint16_t)(0xE400 | zero);
  half z1 = z1u.h;
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn((int)zero));

  half2 scale2 = __half2half2(scale);
  z1z16[0] = __hmul2(scale2, __half2half2(z1));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  half y1 = __float2half_rn(1.0f);
  half y16 = __float2half_rn(1.0f / 16.0f);
  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

// Dequantize one int32 (8 shuffled 4-bit weights) into 4 half2 pairs:
//   dq[0] = (q[0], q[1]) * scale - zero*scale
//   dq[1] = (q[2], q[3]) * scale - zero*scale
//   dq[2] = (q[4], q[5]) * scale - zero*scale
//   dq[3] = (q[6], q[7]) * scale - zero*scale
__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa, half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;

  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;  // half2(q[0]+1024, q[1]+1024)
  q1.u = (qa & 0x00F000F0) | c0;  // half2(q[2]*16+1024, q[3]*16+1024)
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;  // half2(q[4]+1024, q[5]+1024)
  q3.u = (qa_hi & 0x00F000F0) | c0;  // half2(q[6]*16+1024, q[7]*16+1024)

  dq[0] = __hfma2(q0.h2, y1y16[0], z1z16[0]);
  dq[1] = __hfma2(q1.h2, y1y16[1], z1z16[1]);
  dq[2] = __hfma2(q2.h2, y1y16[0], z1z16[0]);
  dq[3] = __hfma2(q3.h2, y1y16[1], z1z16[1]);
}

}  // namespace gptq_rdna2
}  // namespace vllm

#endif  // _qdq_4_rdna2_cuh
