// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared W4A16 GPTQ primitives for RDNA2 (gfx1030) kernels.
//
// Used by:
//   * csrc/rocm/q_gemm_rdna2.cu       (decode kernel)
//   * csrc/rocm/q_gemm_rdna2_prefill.cu (prefill kernel)
//
// All helpers assume fp16 activations / scales. The bit-trick dequant lives
// in qdq_4_rdna2.cuh.

#ifndef _Q_GEMM_RDNA2_COMMON_CUH
#define _Q_GEMM_RDNA2_COMMON_CUH

#include <cstdint>

#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

namespace vllm {
namespace gptq_rdna2 {

// Type-generic zero — half in HIP/ROCm has a converting constructor from
// float, but going through __float2half_rn is the unambiguously correct
// path on every ROCm version.
template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

// 4 V_DOT2_F32_F16 calls covering 8 consecutive K positions.
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

// Precondition: n is a multiple of 4, so the 4 nibbles for columns n..n+3
// fit in one uint32 from the [groups, N/8] packed-zeros tensor.
__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                             T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

// Refresh the (z1z16, y1y16) dequant constants for group g and N_COLS
// consecutive columns starting at n.
template <int N_COLS>
__forceinline__ __device__ void refresh_group(
    int g, int n, const uint32_t* b_qzeros, const half* b_scales,
    int size_n, int zero_offset,
    half2 (&z1z16_h)[N_COLS][2], half2 (&y1y16_h)[N_COLS][2]) {
  const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
  const half* sc_row = b_scales + g * size_n;
  int zeros[N_COLS];
  half scales[N_COLS];
  load4_zeros(qz_row, n, zeros);
  load4_scales<half>(sc_row, n, scales);
  #pragma unroll
  for (int i = 0; i < N_COLS; ++i) {
    prep_zero_scale_fp16(static_cast<uint32_t>(zeros[i] + zero_offset),
                         scales[i], z1z16_h[i], y1y16_h[i]);
  }
}

// Epilogue: write M_TILE rows of 4 consecutive N-columns via packed f16 CAS.
// Output tensor c must be zero-initialized.
template <int M_TILE>
__forceinline__ __device__ void epilogue(
    const float block_c[M_TILE][4], int m_tile, int size_m, int size_n,
    int n, half* c) {
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    const int m_row = m_tile + m;
    if (m_row >= size_m) continue;
    half* c_row = c + m_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    atomic_add_pk4_f16(c_row, r01, r23);
  }
}

}  // namespace gptq_rdna2
}  // namespace vllm

#endif  // _Q_GEMM_RDNA2_COMMON_CUH
