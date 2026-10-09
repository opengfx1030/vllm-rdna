// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 dequant primitives for RDNA2 (gfx1030):
//   - E2M1 nibble → FP16 LUT (16 entries)
//   - UE8M0 → FP16 conversion (power-of-two scale)
//   - V_DOT2_F32_F16 wrapper (4 calls × 2 elements = 8 elements)
//   - 64-bit CAS atomic-add for output reduction
//
// Used by:
//   * csrc/rocm/mxfp4_dot2_dense.cu
//   * csrc/rocm/mxfp4_dot2_moe.cu
//
// Activation dtype is fp16; scale dtype is uint8 (UE8M0 power-of-two).
// E2M1 has no zero point — the FP4 element already encodes signed magnitude.

#ifndef _MXFP4_DOT2_RDNA2_CUH
#define _MXFP4_DOT2_RDNA2_CUH

#include <cstdint>

#include <hip/hip_fp16.h>

namespace vllm {
namespace mxfp4_dot2 {

// ---------------------------------------------------------------------------
// E2M1 → FP16 LUT (16 entries, OCP MXFP4 spec)
// ---------------------------------------------------------------------------
// E2M1 element layout (4 bits, LSB first):
//   s e e m
//   0 0 0 0 = 0
//   0 0 0 1 = 0.5     (subnormal, only valid for exp=0, mantissa=1)
//   0 0 1 0 = 1.0
//   0 0 1 1 = 1.5
//   0 1 0 0 = 2.0
//   0 1 0 1 = 3.0
//   0 1 1 0 = 4.0
//   0 1 1 1 = 6.0
//   1 0 0 0 = -0
//   1 0 0 1 = -0.5
//   1 0 1 0 = -1.0
//   1 0 1 1 = -1.5
//   1 1 0 0 = -2.0
//   1 1 0 1 = -3.0
//   1 1 1 0 = -4.0
//   1 1 1 1 = -6.0
//
// FP16 representations (IEEE 754 binary16, sign:1 + exp:5 + mantissa:10):
//   0x0000 = +0,    0x3800 = +0.5,  0x3C00 = +1.0,  0x3E00 = +1.5,
//   0x4000 = +2.0,  0x4200 = +3.0,  0x4400 = +4.0,  0x4600 = +6.0,
//   0x8000 = -0,    0xB800 = -0.5,  0xBC00 = -1.0,  0xBE00 = -1.5,
//   0xC000 = -2.0,  0xC200 = -3.0,  0xC400 = -4.0,  0xC600 = -6.0
//
// Stored in __constant__ memory for fast LUT access (1-cycle per lookup).
// AMDGPU hipcc 7.14 on gfx1030 rejects `__device__` with any brace
// initializer ("dynamic initialization is not supported") AND rejects
// plain `constexpr` references from `__device__` functions. The only
// form that compiles is a `__device__` function returning the LUT
// value by switch — hipcc accepts this because the function body
// itself is a constant expression folded at compile time, with the
// returned values placed in constant memory.
__device__ __forceinline__ half e2m1_lut_fn(int i) {
  // The values are fp16 *bit patterns*: `return 0x3800;` would convert
  // the integer 14336 to half numerically (14336.0, not 0.5).
  unsigned short bits;
  switch (i) {
    case 1:
      bits = 0x3800;
      break;
    case 2:
      bits = 0x3C00;
      break;
    case 3:
      bits = 0x3E00;
      break;
    case 4:
      bits = 0x4000;
      break;
    case 5:
      bits = 0x4200;
      break;
    case 6:
      bits = 0x4400;
      break;
    case 7:
      bits = 0x4600;
      break;
    case 8:
      bits = 0x8000;
      break;
    case 9:
      bits = 0xB800;
      break;
    case 10:
      bits = 0xBC00;
      break;
    case 11:
      bits = 0xBE00;
      break;
    case 12:
      bits = 0xC000;
      break;
    case 13:
      bits = 0xC200;
      break;
    case 14:
      bits = 0xC400;
      break;
    case 15:
      bits = 0xC600;
      break;
    default:
      bits = 0x0000;
      break;
  }
  return __ushort_as_half(bits);
}
// Consumers MUST call e2m1_lut_fn(q) (function-call syntax), not
// e2m1_lut[q] (subscript syntax). A `#define e2m1_lut(i) e2m1_lut_fn(i)`
// macro would not expand for `e2m1_lut[q0]` because the preprocessor
// treats that as array-subscript syntax, not a macro invocation.

// ---------------------------------------------------------------------------
// UE8M0 → FP16 conversion
// ---------------------------------------------------------------------------
// UE8M0 is 8-bit unsigned exponent (power-of-two scale): scale = 2^(k-127).
// FP16 representation of 2^(scale-127):
//   sign=0, exp=(scale-127+15)=(scale-112), mantissa=0
// So the FP16 bits are: (scale - 112) << 10
//
// Cycle cost: 1 cycle (constant-time shift, no memory access).
// Range: FP16 normal range is [2^-14, 2^15] = [6.1e-5, 32768].
// Subnormal UE8M0 (scale < 113) → FP16 subnormal (rare, may underflow).
// Overflow UE8M0 (scale > 142) → FP16 infinity (very rare in practice).
__forceinline__ __device__ half ue8m0_to_fp16(uint8_t scale_byte) {
  // Without subnormal handling (UE8M0 range is [2^-127, 2^127])
  // For typical LLM weights (scale in [113, 140]), this is exact FP16
  unsigned short bits = (unsigned short)((scale_byte - 112) << 10);
  return __ushort_as_half(bits);
}

// ---------------------------------------------------------------------------
// dot22_8_f — 4 × V_DOT2_F32_F16 calls covering 8 consecutive K positions
// ---------------------------------------------------------------------------
// Use __builtin_amdgcn_fdot2 explicitly. hipcc does not lower the obvious
// hfma2+cast+add pattern to v_dot2 on gfx1030, and keeping the accumulator
// in fp32 avoids the ~3 bits of precision loss from accumulating in fp16.
__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  float result = 0.0f;
  const half2* a2_ptr = reinterpret_cast<const half2*>(a_ptr);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

// ---------------------------------------------------------------------------
// 64-bit CAS atomic-add for 4 fp16 output columns
// ---------------------------------------------------------------------------
// Packed atomic-add via CAS-loop on a 64-bit word (4 fp16 lanes per CAS).
// RDNA2 (gfx1030) does NOT have native v_global_atomic_pk_add_f16 (that
// landed on gfx940), so this lowers to global_atomic_cmpswap_b64 plus retry.
// Writes 4 output columns per row in one atomic op. The (m, n) target is
// 8-byte aligned because n is a multiple of 4 and N is a multiple of 8.
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// ---------------------------------------------------------------------------
// Dequantize one int32 (8 E2M1 nibbles) into 4 half2 pairs (8 FP16 values)
// ---------------------------------------------------------------------------
// Branch-free: an E2M1 nibble's 3 magnitude bits (e e m) placed at fp16 bits
// 11..9 are the fp16 encoding of value * 2^-14, for the normals (fp16
// exponent field = e) and the subnormal 0.5 (fp16 subnormal) alike; the sign
// moves to bit 15. The 2^14 lives in the scale: pass
// ue8m0_to_fp16_e2m1(scale_byte), not ue8m0_to_fp16. This replaced a
// 16-way switch per nibble (the e2m1_lut_fn lookup) that dominated the
// MoE decode GEMMs.
__forceinline__ __device__ void dequant_e2m1_8_fp16(
    uint32_t qa,    // 8 E2M1 nibbles packed LSBs-first
    half2 scale2,   // ue8m0_to_fp16_e2m1(scale) broadcast to half2
    half2 (&dq)[4]  // output: 4 half2 pairs = 8 FP16 values
) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t b = (qa >> (8 * i)) & 0xFFu;  // nibbles 2i (lo), 2i+1 (hi)
    const uint32_t h2 = ((b & 0x08u) << 12) | ((b & 0x07u) << 9) |
                        ((b & 0x80u) << 24) | ((b & 0x70u) << 21);
    dq[i] = __hmul2(__builtin_bit_cast(half2, h2), scale2);
  }
}

// UE8M0 scale times 2^14 (the factor dequant_e2m1_8_fp16 leaves out):
// fp16 bits (scale_byte - 127 + 14 + 15) << 10. Exact for scale bytes 99..128
// (2^-28 .. 2^1), which covers real MXFP4 checkpoints (DeepSeek-V4-Flash
// experts: 119..122).
__forceinline__ __device__ half ue8m0_to_fp16_e2m1(uint8_t scale_byte) {
  return __ushort_as_half((unsigned short)((scale_byte - 98) << 10));
}

// ---------------------------------------------------------------------------
// zero helper for FP16
// ---------------------------------------------------------------------------
template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

}  // namespace mxfp4_dot2
}  // namespace vllm

#endif  // _MXFP4_DOT2_RDNA2_CUH
