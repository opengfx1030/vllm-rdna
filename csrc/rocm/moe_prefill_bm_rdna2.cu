// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Prefill-sized NVFP4 / MXFP8 expert GEMMs for RDNA2 (gfx1030), fp16
// activations, V_DOT2_F32_F16.  Same weight layouts, same dequantised fp16
// values and the same fp32 dot2 accumulation as moe_lowp_dot2_rdna2.cu, but a
// block owns up to kRows (32) routed rows of ONE expert and the WHOLE K range:
//
//   * every weight word is dequantised once per block and reused by up to 32
//     rows (the decode kernels dequantise it once per <= 8 rows and once per K
//     split), and the K range is not split, so the w13 stage needs no zeroing
//     and no atomics and the w2 stage only the top-k reduction;
//   * the work list (``moe_prefill_cta_rdna2``) groups up to kRows /
//   block_size_m
//     consecutive blocks of one expert of the router's block_size_m-padded
//     layout (8 for inkling2_glue_moe_route), so no padding beyond the router's
//     is computed: row groups of 8 past the last valid row are skipped;
//   * the activation tile (rows x 128 / 256 K) sits in LDS, K order permuted
//     inside every 8-K word (0,4,1,5,2,6,3,7) for NVFP4 so that nibble j and
//     nibble j + 4 of a weight word share one half2 (one shift + mask handles
//     both halves: ~20 VALU per word instead of ~40);
//   * every (row, column) keeps ONE fp32 accumulator chained through V_DOT2.
//
//   NVFP4  b_q_weight [E, K/8, N] uint32, b_scales [E, K/16, N] e4m3,
//          b_globals [E] float; K % 32 == 0
//   MXFP8  b_q_weight [E, K/4, N] uint32, b_scales [E, K/32, N] ue8m0; K % 32
//   == 0
//
// Contract (as moe_nvfp4_gemm_rdna2): w13: top_k = 8, output_topk = 0 (plain
// stores, one row per routed id), w2: top_k = 1, mul_topk_weight, output_topk =
// 8 (router weight, atomic fp16 add of the 8 rows of a token into a zeroed
// output).

#include <cstdint>
#include <tuple>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "mxfp4_dot2_common.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace moe_prefill_bm {

constexpr int kRows = 32;      // rows of one block
constexpr int kNt = 128;       // threads of one block (4 columns each)
constexpr int kNvfp4Kc = 128;  // K staged per LDS chunk (NVFP4)
constexpr int kMxfp8Kc = 256;  // K staged per LDS chunk (MXFP8)
constexpr int kI8Kc = 256;     // K staged per LDS chunk (W4A8 NVFP4)
constexpr int kMetaThreads = 256;

using vllm::mxfp4_dot2::atomic_add_pk4_f16;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// nibble J (low half) and nibble J + 4 (high half) of an 8-nibble word as the
// half2 of (value * 2^-14) * scale2, scale2 = e4m3 * global * 2^14.  The three
// magnitude bits (e e m) at fp16 bits 11..9 are the fp16 encoding of
// value * 2^-14 (normals and the subnormal 0.5 alike), the sign goes to bit 15.
template <int J>
__device__ __forceinline__ half2 dq_pair(uint32_t w, half2 s) {
  constexpr int SM = 9 - 4 * J;   // nibble J magnitude -> bits 11..9
  constexpr int SS = 12 - 4 * J;  // nibble J sign (bit 4J+3) -> bit 15
  const uint32_t mag = (SM >= 0 ? (w << SM) : (w >> (-SM))) & 0x0E000E00u;
  const uint32_t h = ((w << SS) & 0x80008000u) | mag;
  return __hmul2(__builtin_bit_cast(half2, h), s);
}

// e4m3 byte -> half2(value * global * 2^14), g22 = global * 2^22.  (b & 0x7F)
// << 7 is the fp16 encoding of value * 2^-8 for normals and subnormals; g22
// undoes it.
__device__ __forceinline__ half2 scale2_e4m3(uint32_t b, float g22) {
  const uint32_t hb = ((b << 7) & 0x3F80u) | ((b << 8) & 0x8000u);
  const float f = __half2float(__ushort_as_half((unsigned short)hb)) * g22;
  return __float2half2_rn(f);
}

// K order inside every 8-K word of an activation chunk: (0,4,1,5,2,6,3,7)
__device__ __forceinline__ uint4 perm_k8(uint4 v) {
  uint4 p;
  p.x = __builtin_amdgcn_perm(v.z, v.x, 0x05040100u);
  p.y = __builtin_amdgcn_perm(v.z, v.x, 0x07060302u);
  p.z = __builtin_amdgcn_perm(v.w, v.y, 0x05040100u);
  p.w = __builtin_amdgcn_perm(v.w, v.y, 0x07060302u);
  return p;
}

// e4m3 bytes -> fp16 on the integer pipe (see moe_lowp_dot2_rdna2.cu): the 7
// magnitude bits moved to fp16 bits 13..7 are the fp16 encoding of value *
// 2^-8.
__device__ __forceinline__ half2 e4m3_block_factor(uint32_t ue8m0) {
  return __half2half2(__float2half_rn(__uint_as_float(ue8m0 << 23) * 256.0f));
}

template <uint32_t SEL>
__device__ __forceinline__ half2 e4m3x2_bits(uint32_t w) {
  const uint32_t lanes = __builtin_amdgcn_perm(0u, w, SEL);
  const uint32_t h =
      ((lanes << 8) & 0x80008000u) | ((lanes << 7) & 0x3F803F80u);
  return __builtin_bit_cast(half2, h);
}

__device__ __forceinline__ void deq_e4m3_4(uint32_t w, half2 factor,
                                           half2* dq) {
  dq[0] = __hmul2(e4m3x2_bits<0x0C010C00u>(w), factor);  // bytes 0, 1
  dq[1] = __hmul2(e4m3x2_bits<0x0C030C02u>(w), factor);  // bytes 2, 3
}

// ---- W4A8 helpers (see nvfp4_i8_prefill_kernel)
// ---------------------------------

// First dot of a chain: the untied VOP3P form takes the inline constant 0 (the
// tied v_dot4c the compiler picks for sdot4(a, b, 0) needs a v_mov 0 of the
// accumulator).
__device__ __forceinline__ int sdot4_init(uint32_t a, uint32_t b) {
  int r;
  asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

__device__ __forceinline__ int sdot4_acc(uint32_t a, uint32_t b, int c) {
  return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), c,
                                false);
}

// Four e2m1 codes (nibble SH/4 of every byte of w >> SH, bit 3 = sign) -> four
// signed int8 = 2 * value, through three V_PERM_B32 (positive table, negative
// table, select by the sign bit).
template <int SH>
__device__ __forceinline__ uint32_t lut_e2m1_x2(uint32_t w) {
  const uint32_t z = w >> SH;
  const uint32_t idx = z & 0x07070707u;
  const uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, idx);
  const uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, idx);
  const uint32_t sel = ((z >> 1) & 0x04040404u) | 0x03020100u;
  return __builtin_amdgcn_perm(neg, pos, sel);
}

// Block head: the work entry, the routed ids of its rows (-1: invalid row) and
// the number of 8-row groups that hold at least one valid row.  Returns false
// when the block has no work (uniform over the block).
template <int BM>
__device__ __forceinline__ bool load_cta(const int4* __restrict__ cta_info,
                                         const int32_t* __restrict__ sorted_ids,
                                         int* sh_id, int size_m, int top_k,
                                         int& expert_id, int& nrg) {
  const int4 ci = cta_info[blockIdx.x];
  if (ci.z <= 0) return false;
  expert_id = ci.x;
  if (threadIdx.x < BM) {
    int id = -1;
    if (threadIdx.x < ci.z) {
      id = sorted_ids[ci.y + threadIdx.x];
      if (id / top_k >= size_m) id = -1;
    }
    sh_id[threadIdx.x] = id;
  }
  __syncthreads();
  int n = 0;
  #pragma unroll
  for (int g = 0; g < BM / 8; ++g) n += (sh_id[8 * g] >= 0) ? 1 : 0;
  nrg = __builtin_amdgcn_readfirstlane(n);
  return true;
}

// router weight, fp16 rounding, plain store (w13) or fp16 atomic add (w2)
template <int BM>
__device__ __forceinline__ void store_rows(const float (&acc)[BM][4],
                                           const int* sh_id,
                                           half* __restrict__ c,
                                           const float* __restrict__ topk_w,
                                           int n, int size_n, bool mul_tw,
                                           int out_topk) {
  #pragma unroll
  for (int m = 0; m < BM; ++m) {
    const int id = sh_id[m];
    if (id < 0) continue;
    float r[4] = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
    if (mul_tw && topk_w != nullptr) {
      const float tw = topk_w[id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) r[j] *= tw;
    }
    const int64_t out_row =
        (out_topk > 0) ? (int64_t)(id / out_topk) : (int64_t)id;
    half* out = c + out_row * size_n + n;
    const half2 r01 =
        __halves2half2(__float2half_rn(r[0]), __float2half_rn(r[1]));
    const half2 r23 =
        __halves2half2(__float2half_rn(r[2]), __float2half_rn(r[3]));
    if (out_topk > 0) {
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      uint2 u;
      u.x = __builtin_bit_cast(uint32_t, r01);
      u.y = __builtin_bit_cast(uint32_t, r23);
      *(uint2*)out = u;
    }
  }
}

// ---------------------------------------------------------------------------
// NVFP4.  Weights prefetched two 16-K groups ahead across the whole K range
// (two register sets), the activation tile double-buffered in LDS (next chunk
// loaded into registers before the compute of the current one, stored after it,
// one barrier per chunk).
// ---------------------------------------------------------------------------
template <int BM, int NT, int KC>
__global__ __launch_bounds__(NT) void nvfp4_prefill_kernel(
    const half* __restrict__ a, half* __restrict__ c,
    const uint32_t* __restrict__ b_q,  // [E, K/8, N]
    const uint8_t* __restrict__ b_s,   // [E, K/16, N] e4m3
    const float* __restrict__ b_g,     // [E]
    const float* __restrict__ topk_w, const int32_t* __restrict__ sorted_ids,
    const int4* __restrict__ cta_info, const int size_m, const int size_n,
    const int size_k, const int top_k, const int ew_stride, const int es_stride,
    const bool mul_tw, const int out_topk) {
  constexpr int CPR = KC / 8;
  constexpr int NCH = BM * CPR;
  constexpr int NS = NCH / NT;
  constexpr int NG = KC / 16;
  static_assert(NCH % NT == 0, "NT must divide BM * KC / 8");
  static_assert(NG % 2 == 0 && BM % 8 == 0 && KC % 16 == 0, "shape");
  const int t = threadIdx.x;
  const int n = blockIdx.y * (NT * 4) + t * 4;

  __shared__ __attribute__((aligned(16))) half lds_a[2][BM * KC];
  __shared__ int sh_id[BM];
  int expert_id, nrg;
  if (!load_cta<BM>(cta_info, sorted_ids, sh_id, size_m, top_k, expert_id, nrg))
    return;
  const int nrows = nrg * 8;

  const bool active = n < size_n;
  const uint32_t* ew = b_q + (int64_t)expert_id * ew_stride;
  const uint8_t* es = b_s + (int64_t)expert_id * es_stride;
  const float g22 = b_g[expert_id] * 4194304.0f;  // global * 2^22
  const int nchunks = (size_k + KC - 1) / KC;
  const int total_groups = size_k >> 4;

  // staging descriptors: thread t moves chunks t + i * NT of the BM x CPR grid
  int soff[NS];
  int loff[NS];
  int sk_off[NS];
  bool sval[NS];
  #pragma unroll
  for (int i = 0; i < NS; ++i) {
    const int q = t + i * NT;
    const int r = q / CPR;
    const int ch = q - r * CPR;
    const int id = sh_id[r];
    sval[i] = (id >= 0) && (r < nrows);
    sk_off[i] = 8 * ch;
    soff[i] = sval[i] ? (id / top_k) * size_k + 8 * ch : 0;
    loff[i] = r * KC + 8 * ch;
  }
  uint4 sreg[NS];
  #pragma unroll
  for (int i = 0; i < NS; ++i)
    sreg[i] = (sval[i] && sk_off[i] < size_k) ? *(const uint4*)(a + soff[i])
                                              : make_uint4(0u, 0u, 0u, 0u);
  #pragma unroll
  for (int i = 0; i < NS; ++i)
    *(uint4*)(&lds_a[0][0] + loff[i]) = perm_k8(sreg[i]);

  const uint32_t* b_ptr = ew + n;
  const uint8_t* s_ptr = es + n;
  // two prefetch sets: groups 0 and 1 (total_groups >= 2 since size_k % 32 ==
  // 0)
  uint4 wa0 = make_uint4(0u, 0u, 0u, 0u), wa1 = wa0;
  uint4 wb0 = wa0, wb1 = wa0;
  uint32_t sa = 0u, sbb = 0u;
  if (active) {
    wa0 = *(const uint4*)(b_ptr);
    wa1 = *(const uint4*)(b_ptr + size_n);
    sa = *(const uint32_t*)(s_ptr);
    wb0 = *(const uint4*)(b_ptr + 2 * (int64_t)size_n);
    wb1 = *(const uint4*)(b_ptr + 3 * (int64_t)size_n);
    sbb = *(const uint32_t*)(s_ptr + size_n);
  }
  __syncthreads();

  float acc[BM][4];
  #pragma unroll
  for (int m = 0; m < BM; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  // one 16-K group of the 4 columns of this thread against the rows of the tile
  auto group = [&](const uint4& w0, const uint4& w1, uint32_t sb,
                   const half* arow) {
    half2 sc[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j)
      sc[j] = scale2_e4m3((sb >> (8 * j)) & 0xFFu, g22);
  #pragma unroll
    for (int ws = 0; ws < 2; ++ws) {
      const uint4 w = ws ? w1 : w0;
      const uint32_t wc[4] = {w.x, w.y, w.z, w.w};
      half2 dq[4][4];
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        dq[j][0] = dq_pair<0>(wc[j], sc[j]);
        dq[j][1] = dq_pair<1>(wc[j], sc[j]);
        dq[j][2] = dq_pair<2>(wc[j], sc[j]);
        dq[j][3] = dq_pair<3>(wc[j], sc[j]);
      }
  #pragma unroll
      for (int rg = 0; rg < BM / 8; ++rg) {
        if (rg < nrg) {
  #pragma unroll
          for (int mi = 0; mi < 8; ++mi) {
            const int m = rg * 8 + mi;
            const uint4 av = *(const uint4*)(arow + m * KC + 8 * ws);
            const half2 a0 = __builtin_bit_cast(half2, av.x);
            const half2 a1 = __builtin_bit_cast(half2, av.y);
            const half2 a2 = __builtin_bit_cast(half2, av.z);
            const half2 a3 = __builtin_bit_cast(half2, av.w);
  #pragma unroll
            for (int j = 0; j < 4; ++j) {
              float r = acc[m][j];
              r = __builtin_amdgcn_fdot2(dq[j][0], a0, r, false);
              r = __builtin_amdgcn_fdot2(dq[j][1], a1, r, false);
              r = __builtin_amdgcn_fdot2(dq[j][2], a2, r, false);
              r = __builtin_amdgcn_fdot2(dq[j][3], a3, r, false);
              acc[m][j] = r;
            }
          }
        }
      }
    }
  };

  for (int ck = 0; ck < nchunks; ++ck) {
    if (ck + 1 < nchunks) {
      const int kn = (ck + 1) * KC;
  #pragma unroll
      for (int i = 0; i < NS; ++i)
        sreg[i] = (sval[i] && kn + sk_off[i] < size_k)
                      ? *(const uint4*)(a + soff[i] + kn)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
    const half* cur = &lds_a[ck & 1][0];
    const int ng = min(KC, size_k - ck * KC) >> 4;
    if (active) {
  #pragma unroll 1
      for (int g = 0; g < ng; g += 2) {
        const int G = ck * NG + g;
        {
          const uint4 w0 = wa0, w1 = wa1;
          const uint32_t sb = sa;
          const int gn = min(G + 2, total_groups - 1);
          wa0 = *(const uint4*)(b_ptr + (int64_t)(2 * gn) * size_n);
          wa1 = *(const uint4*)(b_ptr + (int64_t)(2 * gn + 1) * size_n);
          sa = *(const uint32_t*)(s_ptr + (int64_t)gn * size_n);
          group(w0, w1, sb, cur + 16 * g);
        }
        {
          const uint4 w0 = wb0, w1 = wb1;
          const uint32_t sb = sbb;
          const int gn = min(G + 3, total_groups - 1);
          wb0 = *(const uint4*)(b_ptr + (int64_t)(2 * gn) * size_n);
          wb1 = *(const uint4*)(b_ptr + (int64_t)(2 * gn + 1) * size_n);
          sbb = *(const uint32_t*)(s_ptr + (int64_t)gn * size_n);
          group(w0, w1, sb, cur + 16 * (g + 1));
        }
      }
    }
    if (ck + 1 < nchunks) {
      half* nxt = &lds_a[(ck + 1) & 1][0];
  #pragma unroll
      for (int i = 0; i < NS; ++i) *(uint4*)(nxt + loff[i]) = perm_k8(sreg[i]);
    }
    __syncthreads();
  }

  if (active)
    store_rows<BM>(acc, sh_id, c, topk_w, n, size_n, mul_tw, out_topk);
}

// ---------------------------------------------------------------------------
// MXFP8: 32 K per iteration (one scale byte per column), activations in natural
// K order (two bytes dequantise into one half2 of adjacent K).
// ---------------------------------------------------------------------------
template <int BM, int NT, int KC>
__global__ __launch_bounds__(NT) void mxfp8_prefill_kernel(
    const half* __restrict__ a, half* __restrict__ c,
    const uint32_t* __restrict__ b_q,  // [E, K/4, N]
    const uint8_t* __restrict__ b_s,   // [E, K/32, N] ue8m0
    const float* __restrict__ topk_w, const int32_t* __restrict__ sorted_ids,
    const int4* __restrict__ cta_info, const int size_m, const int size_n,
    const int size_k, const int top_k, const int ew_stride, const int es_stride,
    const bool mul_tw, const int out_topk) {
  static_assert(NT >= BM, "NT >= BM");
  static_assert(KC % 32 == 0 && BM % 8 == 0, "KC % 32, BM % 8");
  const int t = threadIdx.x;
  const int n = blockIdx.y * (NT * 4) + t * 4;

  __shared__ __attribute__((aligned(16))) half lds_a[BM * KC];
  __shared__ int sh_id[BM];
  int expert_id, nrg;
  if (!load_cta<BM>(cta_info, sorted_ids, sh_id, size_m, top_k, expert_id, nrg))
    return;
  const int nrows = nrg * 8;

  const bool active = n < size_n;
  const uint32_t* ew = b_q + (int64_t)expert_id * ew_stride;
  const uint8_t* es = b_s + (int64_t)expert_id * es_stride;

  float acc[BM][4];
  #pragma unroll
  for (int m = 0; m < BM; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  for (int k0 = 0; k0 < size_k; k0 += KC) {
    const int kc = min(KC, size_k - k0);
    const int cpr = kc >> 3;
    for (int q = t; q < nrows * cpr; q += NT) {
      const int r = q / cpr;
      const int ch = q - r * cpr;
      const int id = sh_id[r];
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (id >= 0)
        v = *(const uint4*)(a + (int64_t)(id / top_k) * size_k + k0 + 8 * ch);
      *(uint4*)(lds_a + r * KC + 8 * ch) = v;
    }
    __syncthreads();

    if (active) {
      const uint32_t* b_ptr = ew + (int64_t)(k0 >> 2) * size_n + n;
      const uint8_t* s_ptr = es + (int64_t)(k0 >> 5) * size_n + n;
      const int ng = kc >> 5;  // 32-K iterations
      uint4 wn[8];
  #pragma unroll
      for (int i = 0; i < 8; ++i)
        wn[i] = *(const uint4*)(b_ptr + (int64_t)i * size_n);
      uint32_t sbn = *(const uint32_t*)(s_ptr);
  #pragma unroll 1
      for (int g = 0; g < ng; ++g) {
        uint4 w[8];
  #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = wn[i];
        const uint32_t sb = sbn;
        const int gn = (g + 1 < ng) ? g + 1 : g;
  #pragma unroll
        for (int i = 0; i < 8; ++i)
          wn[i] = *(const uint4*)(b_ptr + (int64_t)(8 * gn + i) * size_n);
        sbn = *(const uint32_t*)(s_ptr + (int64_t)gn * size_n);
        half2 factor[4];
  #pragma unroll
        for (int j = 0; j < 4; ++j)
          factor[j] = e4m3_block_factor((sb >> (8 * j)) & 0xFFu);

        const half* arow = lds_a + 32 * g;
  #pragma unroll
        for (int s = 0; s < 4; ++s) {  // 8 K per slice
          const uint32_t wlo[4] = {w[2 * s].x, w[2 * s].y, w[2 * s].z,
                                   w[2 * s].w};
          const uint32_t whi[4] = {w[2 * s + 1].x, w[2 * s + 1].y,
                                   w[2 * s + 1].z, w[2 * s + 1].w};
          half2 dq[4][4];
  #pragma unroll
          for (int j = 0; j < 4; ++j) {
            deq_e4m3_4(wlo[j], factor[j], &dq[j][0]);
            deq_e4m3_4(whi[j], factor[j], &dq[j][2]);
          }
  #pragma unroll
          for (int rg = 0; rg < BM / 8; ++rg) {
            if (rg < nrg) {
  #pragma unroll
              for (int mi = 0; mi < 8; ++mi) {
                const int m = rg * 8 + mi;
                const uint4 av = *(const uint4*)(arow + m * KC + 8 * s);
                const half2 a0 = __builtin_bit_cast(half2, av.x);
                const half2 a1 = __builtin_bit_cast(half2, av.y);
                const half2 a2 = __builtin_bit_cast(half2, av.z);
                const half2 a3 = __builtin_bit_cast(half2, av.w);
  #pragma unroll
                for (int j = 0; j < 4; ++j) {
                  float r = acc[m][j];
                  r = __builtin_amdgcn_fdot2(dq[j][0], a0, r, false);
                  r = __builtin_amdgcn_fdot2(dq[j][1], a1, r, false);
                  r = __builtin_amdgcn_fdot2(dq[j][2], a2, r, false);
                  r = __builtin_amdgcn_fdot2(dq[j][3], a3, r, false);
                  acc[m][j] = r;
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (active)
    store_rows<BM>(acc, sh_id, c, topk_w, n, size_n, mul_tw, out_topk);
}

// ---------------------------------------------------------------------------
// W4A8 NVFP4 (flag VLLM_INKLING2_PREFILL_INT8, OFF by default): the activations
// are int8, one fp32 scale per (row, 32 K) (moe_prefill_act_quant_rdna2, K
// order inside every 8-K word permuted to (0,2,4,6 | 1,3,5,7)), and
// V_DOT4_I32_I8 does the K reduction.
//
//   weight word (8 nibbles): E = w & 0x0F0F0F0F, O = (w >> 4) & 0x0F0F0F0F hold
//   the nibbles (0,2,4,6) / (1,3,5,7); lut_e2m1_x2 turns 4 codes into 4 int8 =
//   2 * e2m1. dot4(E, A_even) + dot4(O, A_odd) over the 16 K of a group give
//   the integer P_g.
//
//   The e4m3 scale of group g is M_g * 2^(E_g - 10) (M 0..15, E 1..15).  Over
//   the TWO groups of a 32-K window (the activation scale window) the scales
//   are aligned to the larger exponent: Mm_g = (M_g << 6) >> (E_max - E_g)
//   (exact for E_max - E_g
//   <= 2, truncated below that), T = Mm_0 * P_0 + Mm_1 * P_1 (|P| <= 24384 and
//   Mm <= 960 keep T in 24-bit multiplies: v_mul_i32_i24 + v_mad_i32_i24), and
//   one fp32 flush per (row, column, window):
//       acc += float(T) * (sa_row * wsc_col),  wsc = global * 2^(E_max - 17).
// Blocking, work list, row groups and epilogue as nvfp4_prefill_kernel.
// ---------------------------------------------------------------------------
template <int BM, int NT, int KC>
__global__ __launch_bounds__(NT) void nvfp4_i8_prefill_kernel(
    const int8_t* __restrict__ a8,  // [rows, K] int8, K permuted inside 8-K
                                    // words
    const float* __restrict__ sa,   // [rows, K / 32]
    half* __restrict__ c,
    const uint32_t* __restrict__ b_q,  // [E, K/8, N]
    const uint8_t* __restrict__ b_s,   // [E, K/16, N] e4m3
    const float* __restrict__ b_g,     // [E]
    const float* __restrict__ topk_w, const int32_t* __restrict__ sorted_ids,
    const int4* __restrict__ cta_info, const int size_m, const int size_n,
    const int size_k, const int top_k, const int ew_stride, const int es_stride,
    const bool mul_tw, const int out_topk) {
  static_assert(NT >= BM && KC % 32 == 0 && BM % 8 == 0, "shape");
  constexpr int NW = KC / 32;  // windows per chunk
  const int t = threadIdx.x;
  const int n = blockIdx.y * (NT * 4) + t * 4;

  __shared__ __attribute__((aligned(16))) int8_t lds_a[BM * KC];
  __shared__ float lds_s[BM * NW];
  __shared__ int sh_id[BM];
  int expert_id, nrg;
  if (!load_cta<BM>(cta_info, sorted_ids, sh_id, size_m, top_k, expert_id, nrg))
    return;
  const int nrows = nrg * 8;

  const bool active = n < size_n;
  const uint32_t* ew = b_q + (int64_t)expert_id * ew_stride;
  const uint8_t* es = b_s + (int64_t)expert_id * es_stride;
  const float gl = b_g[expert_id];
  const int kwin = size_k >> 5;  // 32-K windows per row

  float acc[BM][4];
  #pragma unroll
  for (int m = 0; m < BM; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  for (int k0 = 0; k0 < size_k; k0 += KC) {
    const int kc = min(KC, size_k - k0);
    const int cpr = kc >> 4;  // 16-K pieces of a row
    for (int q = t; q < nrows * cpr; q += NT) {
      const int r = q / cpr;
      const int ch = q - r * cpr;
      const int id = sh_id[r];
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (id >= 0)
        v = *(const uint4*)(a8 + (int64_t)(id / top_k) * size_k + k0 + 16 * ch);
      *(uint4*)(lds_a + r * KC + 16 * ch) = v;
    }
    const int wpc = kc >> 5;  // windows in this chunk
    for (int q = t; q < nrows * wpc; q += NT) {
      const int r = q / wpc;
      const int wi = q - r * wpc;
      const int id = sh_id[r];
      lds_s[r * NW + wi] =
          (id >= 0) ? sa[(int64_t)(id / top_k) * kwin + (k0 >> 5) + wi] : 0.0f;
    }
    __syncthreads();

    if (active) {
      const uint32_t* b_ptr = ew + (int64_t)(k0 >> 3) * size_n + n;
      const uint8_t* s_ptr = es + (int64_t)(k0 >> 4) * size_n + n;
      uint4 wn[4];
      uint32_t sn[2];
  #pragma unroll
      for (int i = 0; i < 4; ++i)
        wn[i] = *(const uint4*)(b_ptr + (int64_t)i * size_n);
  #pragma unroll
      for (int i = 0; i < 2; ++i)
        sn[i] = *(const uint32_t*)(s_ptr + (int64_t)i * size_n);
  #pragma unroll 1
      for (int w = 0; w < wpc; ++w) {
        uint4 wq[4];
        uint32_t sb[2];
  #pragma unroll
        for (int i = 0; i < 4; ++i) wq[i] = wn[i];
  #pragma unroll
        for (int i = 0; i < 2; ++i) sb[i] = sn[i];
        {
          const int wnx = (w + 1 < wpc) ? w + 1 : w;
  #pragma unroll
          for (int i = 0; i < 4; ++i)
            wn[i] = *(const uint4*)(b_ptr + (int64_t)(4 * wnx + i) * size_n);
  #pragma unroll
          for (int i = 0; i < 2; ++i)
            sn[i] = *(const uint32_t*)(s_ptr + (int64_t)(2 * wnx + i) * size_n);
        }

        // scale decode of the two groups of the window, per column
        int mm0[4], mm1[4];
        float wsc[4];
  #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t b0 = (sb[0] >> (8 * j)) & 0xFFu;
          const uint32_t b1 = (sb[1] >> (8 * j)) & 0xFFu;
          const uint32_t e0 = (b0 >> 3) & 15u, e1 = (b1 >> 3) & 15u;
          const uint32_t M0 = (b0 & 7u) | (e0 != 0u ? 8u : 0u);
          const uint32_t M1 = (b1 & 7u) | (e1 != 0u ? 8u : 0u);
          const uint32_t E0 = max(e0, 1u), E1 = max(e1, 1u);
          const uint32_t Em = max(E0, E1);
          mm0[j] = (int)((M0 << 6) >> (Em - E0));
          mm1[j] = (int)((M1 << 6) >> (Em - E1));
          wsc[j] =
              gl * __uint_as_float((Em + 110u) << 23);  // global * 2^(Em - 17)
        }

        // int8 weights of the 4 columns: group g, word wd, even / odd K
        uint32_t dq[2][2][4][2];
  #pragma unroll
        for (int g = 0; g < 2; ++g)
  #pragma unroll
          for (int wd = 0; wd < 2; ++wd) {
            const uint4 x = wq[2 * g + wd];
            const uint32_t wc[4] = {x.x, x.y, x.z, x.w};
  #pragma unroll
            for (int j = 0; j < 4; ++j) {
              dq[g][wd][j][0] = lut_e2m1_x2<0>(wc[j]);
              dq[g][wd][j][1] = lut_e2m1_x2<4>(wc[j]);
            }
          }

        const int8_t* arow = lds_a + 32 * w;
        const float* srow = lds_s + w;
  #pragma unroll
        for (int rg = 0; rg < BM / 8; ++rg) {
          if (rg < nrg) {
  #pragma unroll
            for (int mi = 0; mi < 8; ++mi) {
              const int m = rg * 8 + mi;
              const uint4 a0 = *(const uint4*)(arow + m * KC);
              const uint4 a1 = *(const uint4*)(arow + m * KC + 16);
              const float sam = srow[m * NW];
  #pragma unroll
              for (int j = 0; j < 4; ++j) {
                int p0 = sdot4_init(dq[0][0][j][0], a0.x);
                p0 = sdot4_acc(dq[0][0][j][1], a0.y, p0);
                p0 = sdot4_acc(dq[0][1][j][0], a0.z, p0);
                p0 = sdot4_acc(dq[0][1][j][1], a0.w, p0);
                int p1 = sdot4_init(dq[1][0][j][0], a1.x);
                p1 = sdot4_acc(dq[1][0][j][1], a1.y, p1);
                p1 = sdot4_acc(dq[1][1][j][0], a1.z, p1);
                p1 = sdot4_acc(dq[1][1][j][1], a1.w, p1);
                const int T = __mul24(p0, mm0[j]) + __mul24(p1, mm1[j]);
                acc[m][j] = fmaf((float)T, sam * wsc[j], acc[m][j]);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (active)
    store_rows<BM>(acc, sh_id, c, topk_w, n, size_n, mul_tw, out_topk);
}

#else  // non-RDNA2: empty stubs for symbol parity

template <int BM, int NT, int KC>
__global__ void nvfp4_prefill_kernel(const half*, half*, const uint32_t*,
                                     const uint8_t*, const float*, const float*,
                                     const int32_t*, const int4*, const int,
                                     const int, const int, const int, const int,
                                     const int, const bool, const int) {}

template <int BM, int NT, int KC>
__global__ void mxfp8_prefill_kernel(const half*, half*, const uint32_t*,
                                     const uint8_t*, const float*,
                                     const int32_t*, const int4*, const int,
                                     const int, const int, const int, const int,
                                     const int, const bool, const int) {}

template <int BM, int NT, int KC>
__global__ void nvfp4_i8_prefill_kernel(const int8_t*, const float*, half*,
                                        const uint32_t*, const uint8_t*,
                                        const float*, const float*,
                                        const int32_t*, const int4*, const int,
                                        const int, const int, const int,
                                        const int, const int, const bool,
                                        const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// fp16 [rows, K] -> int8 [rows, K] (K permuted inside every 8-K word to
// (0,2,4,6 | 1,3,5,7)) + fp32 scales [rows, K / 32] = absmax / 127; one thread
// per 32-K window.  Round to nearest even, clamp to +-127.
// ---------------------------------------------------------------------------
template <int DUMMY = 0>
__global__ void moe_prefill_act_quant_kernel(const half* __restrict__ a,
                                             int8_t* __restrict__ a8,
                                             float* __restrict__ sa,
                                             const int64_t nwin) {
  const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwin) return;
  const uint4* src = reinterpret_cast<const uint4*>(a + w * 32);
  uint4 v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = src[i];
  float f[32];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const half2 h = __builtin_bit_cast(half2, u[j]);
      f[8 * i + 2 * j] = __half2float(__low2half(h));
      f[8 * i + 2 * j + 1] = __half2float(__high2half(h));
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int q[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float r = rintf(f[i] * inv);
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    q[i] = static_cast<int>(r);
  }
  uint32_t out[8];
#pragma unroll
  for (int kw = 0; kw < 4; ++kw) {
    auto pk = [&](int a0, int a1, int a2, int a3) {
      return (uint32_t)(q[a0] & 0xFF) | ((uint32_t)(q[a1] & 0xFF) << 8) |
             ((uint32_t)(q[a2] & 0xFF) << 16) |
             ((uint32_t)(q[a3] & 0xFF) << 24);
    };
    out[2 * kw] = pk(8 * kw + 0, 8 * kw + 2, 8 * kw + 4, 8 * kw + 6);  // even K
    out[2 * kw + 1] =
        pk(8 * kw + 1, 8 * kw + 3, 8 * kw + 5, 8 * kw + 7);  // odd K
  }
  uint4* dst = reinterpret_cast<uint4*>(a8 + w * 32);
  dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
  dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
  sa[w] = amax * (1.0f / 127.0f);
}

// ---------------------------------------------------------------------------
// Work list for a ``ba``-padded layout (ba = the router's block_size_m): one
// entry {expert, first sorted row, valid rows, 0} per work block of up to
// R = kRows / ba consecutive blocks of one expert.  Full work blocks come first
// (expert order), the partial last work block of each expert after them,
// longest first, so the tail of the grid is made of the shortest blocks.
// Entries past the last block have rows == 0.  expert_ids must be ascending (as
// every moe_align_block_size layout).  Single workgroup.
// ---------------------------------------------------------------------------
template <int DUMMY = 0>
__global__ __launch_bounds__(kMetaThreads) void moe_prefill_cta_kernel(
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int4* __restrict__ cta_info, const int num_experts, const int ba,
    const int R, const int max_ctas) {
  __shared__ int s_lb[kMetaThreads + 1];  // first block of expert e
  __shared__ int
      s_nf[kMetaThreads + 1];        // exclusive prefix of the full work blocks
  __shared__ int s_f[kMetaThreads];  // full work blocks of expert e
  __shared__ int s_p[kMetaThreads];  // blocks of the partial one (0: none)
  __shared__ int s_inv[kMetaThreads];  // partial order -> expert
  __shared__ int s_tot[2];
  const int t = threadIdx.x;
  const int nblk = num_tokens_post_padded[0] / ba;
  for (int e = t; e <= num_experts; e += kMetaThreads) {
    int lo = 0, hi = nblk;  // first block whose expert id is >= e
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (expert_ids[mid] < e)
        lo = mid + 1;
      else
        hi = mid;
    }
    s_lb[e] = lo;
  }
  __syncthreads();
  for (int e = t; e < num_experts; e += kMetaThreads) {
    const int cnt = s_lb[e + 1] - s_lb[e];
    const int f = cnt / R;
    s_f[e] = f;
    s_p[e] = cnt - f * R;
  }
  __syncthreads();
  if (t == 0) {
    int acc = 0, np = 0;
    for (int e = 0; e < num_experts; ++e) {
      s_nf[e] = acc;
      acc += s_f[e];
      np += (s_p[e] > 0) ? 1 : 0;
    }
    s_nf[num_experts] = acc;
    s_tot[0] = acc;
    s_tot[1] = np;
  }
  __syncthreads();
  for (int e = t; e < num_experts; e += kMetaThreads) {
    const int p = s_p[e];
    if (p > 0) {
      int rank = 0;
      for (int e2 = 0; e2 < num_experts; ++e2) {
        const int p2 = s_p[e2];
        rank += (p2 > p || (p2 == p && e2 < e)) ? 1 : 0;
      }
      s_inv[rank] = e;
    }
  }
  __syncthreads();
  const int F = s_tot[0], P = s_tot[1];
  for (int c = t; c < max_ctas; c += kMetaThreads) {
    int4 info = make_int4(-1, 0, 0, 0);
    if (c < F) {
      int lo = 0, hi = num_experts;  // last expert with s_nf[e] <= c
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (s_nf[mid] <= c)
          lo = mid;
        else
          hi = mid;
      }
      info = make_int4(lo, (s_lb[lo] + (c - s_nf[lo]) * R) * ba, R * ba, 0);
    } else if (c < F + P) {
      const int e = s_inv[c - F];
      info = make_int4(e, (s_lb[e] + s_f[e] * R) * ba, s_p[e] * ba, 0);
    }
    cta_info[c] = info;
  }
}

}  // namespace moe_prefill_bm

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

// Work list of an ``block_size_m``-padded layout: int32 [max_blocks, 4].
torch::Tensor moe_prefill_cta_rdna2(torch::Tensor expert_ids,
                                    torch::Tensor num_tokens_post_padded,
                                    int64_t num_experts, int64_t block_size_m) {
  using namespace moe_prefill_bm;
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == torch::kInt,
              "expert_ids must be int32 on the GPU");
  TORCH_CHECK(num_tokens_post_padded.is_cuda() &&
                  num_tokens_post_padded.scalar_type() == torch::kInt &&
                  num_tokens_post_padded.numel() >= 1,
              "num_tokens_post_padded must be an int32 GPU tensor");
  TORCH_CHECK(num_experts >= 1 && num_experts <= kMetaThreads,
              "moe_prefill_cta_rdna2: num_experts must be in [1, ",
              kMetaThreads, "]");
  TORCH_CHECK(
      block_size_m >= 1 && block_size_m <= kRows && kRows % block_size_m == 0,
      "moe_prefill_cta_rdna2: block_size_m must divide ", kRows, ", got ",
      block_size_m);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_ids));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int R = (int)(kRows / block_size_m);
  const int max_ctas = (int)((expert_ids.size(0) + R - 1) / R + num_experts);
  auto cta_info = torch::empty({max_ctas, 4}, expert_ids.options());
  moe_prefill_cta_kernel<0><<<1, kMetaThreads, 0, stream>>>(
      expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(),
      reinterpret_cast<int4*>(cta_info.data_ptr()), (int)num_experts,
      (int)block_size_m, R, max_ctas);
  return cta_info;
}

void moe_nvfp4_bm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor b_globals, torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids, torch::Tensor cta_info,
                        int64_t top_k, bool mul_topk_weight,
                        int64_t output_topk) {
  using namespace moe_prefill_bm;
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf && c.scalar_type() == torch::kHalf,
      "a and c must be fp16");
  TORCH_CHECK(b_q_weight.dim() == 3 && b_q_weight.is_contiguous(),
              "b_q_weight must be a contiguous 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3 && b_scales.is_contiguous(),
              "b_scales must be a contiguous 3D [E, K/16, N]");
  TORCH_CHECK(cta_info.is_cuda() && cta_info.scalar_type() == torch::kInt &&
                  cta_info.dim() == 2 && cta_info.size(1) == 4 &&
                  cta_info.is_contiguous(),
              "cta_info must come from moe_prefill_cta_rdna2");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(2);
  TORCH_CHECK((int)b_q_weight.size(1) * 8 == size_k,
              "b_q_weight K-dim * 8 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(1) * 16 == size_k,
              "b_scales K-dim * 16 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(2) == size_n, "b_scales N-dim must equal N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(size_k % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(b_globals.numel() == (int64_t)b_q_weight.size(0),
              "b_globals must hold one scale per expert");
  TORCH_CHECK(top_k >= 1 && output_topk >= 0, "top_k / output_topk");
  const int ew_stride = (int)(b_q_weight.size(1) * size_n);
  const int es_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const float* tw =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  dim3 block(kNt);
  dim3 grid((unsigned)cta_info.size(0), (size_n + kNt * 4 - 1) / (kNt * 4), 1);
  nvfp4_prefill_kernel<kRows, kNt, kNvfp4Kc><<<grid, block, 0, stream>>>(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      (const uint8_t*)b_scales.data_ptr(), b_globals.data_ptr<float>(), tw,
      sorted_token_ids.data_ptr<int32_t>(),
      reinterpret_cast<const int4*>(cta_info.data_ptr()), size_m, size_n,
      size_k, (int)top_k, ew_stride, es_stride, mul_topk_weight,
      (int)output_topk);
}

void moe_mxfp8_bm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids, torch::Tensor cta_info,
                        int64_t top_k, bool mul_topk_weight,
                        int64_t output_topk) {
  using namespace moe_prefill_bm;
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf && c.scalar_type() == torch::kHalf,
      "a and c must be fp16");
  TORCH_CHECK(b_q_weight.dim() == 3 && b_q_weight.is_contiguous(),
              "b_q_weight must be a contiguous 3D [E, K/4, N]");
  TORCH_CHECK(b_scales.dim() == 3 && b_scales.is_contiguous(),
              "b_scales must be a contiguous 3D [E, K/32, N]");
  TORCH_CHECK(cta_info.is_cuda() && cta_info.scalar_type() == torch::kInt &&
                  cta_info.dim() == 2 && cta_info.size(1) == 4 &&
                  cta_info.is_contiguous(),
              "cta_info must come from moe_prefill_cta_rdna2");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(2);
  TORCH_CHECK((int)b_q_weight.size(1) * 4 == size_k,
              "b_q_weight K-dim * 4 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(1) * 32 == size_k,
              "b_scales K-dim * 32 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(2) == size_n, "b_scales N-dim must equal N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(size_k % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(top_k >= 1 && output_topk >= 0, "top_k / output_topk");
  const int ew_stride = (int)(b_q_weight.size(1) * size_n);
  const int es_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const float* tw =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  dim3 block(kNt);
  dim3 grid((unsigned)cta_info.size(0), (size_n + kNt * 4 - 1) / (kNt * 4), 1);
  mxfp8_prefill_kernel<kRows, kNt, kMxfp8Kc><<<grid, block, 0, stream>>>(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      (const uint8_t*)b_scales.data_ptr(), tw,
      sorted_token_ids.data_ptr<int32_t>(),
      reinterpret_cast<const int4*>(cta_info.data_ptr()), size_m, size_n,
      size_k, (int)top_k, ew_stride, es_stride, mul_topk_weight,
      (int)output_topk);
}

// fp16 [rows, K] -> (int8 [rows, K], fp32 [rows, K / 32]) for
// moe_nvfp4_bm_i8_rdna2.
std::tuple<torch::Tensor, torch::Tensor> moe_prefill_act_quant_rdna2(
    torch::Tensor a) {
  using namespace moe_prefill_bm;
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.is_contiguous() &&
                  a.scalar_type() == torch::kHalf,
              "a must be a contiguous fp16 [rows, K] GPU tensor");
  const int64_t rows = a.size(0), k = a.size(1);
  TORCH_CHECK(k % 32 == 0, "K must be a multiple of 32");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto a8 = torch::empty({rows, k}, a.options().dtype(torch::kChar));
  auto sa = torch::empty({rows, k / 32}, a.options().dtype(torch::kFloat));
  const int64_t nwin = rows * (k / 32);
  if (nwin > 0) {
    moe_prefill_act_quant_kernel<0>
        <<<(unsigned)((nwin + 255) / 256), 256, 0, stream>>>(
            (const half*)a.data_ptr(), a8.data_ptr<int8_t>(),
            sa.data_ptr<float>(), nwin);
  }
  return {a8, sa};
}

// W4A8 variant of moe_nvfp4_bm_rdna2: ``a8`` / ``sa`` from
// moe_prefill_act_quant_rdna2.
void moe_nvfp4_bm_i8_rdna2(torch::Tensor a8, torch::Tensor sa, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor b_globals, torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor cta_info, int64_t top_k,
                           bool mul_topk_weight, int64_t output_topk) {
  using namespace moe_prefill_bm;
  TORCH_CHECK(a8.is_cuda() && c.is_cuda() && sa.is_cuda(),
              "a8, sa and c must be GPU tensors");
  TORCH_CHECK(
      a8.dim() == 2 && a8.is_contiguous() && a8.scalar_type() == torch::kChar,
      "a8 must be a contiguous int8 [rows, K]");
  TORCH_CHECK(sa.dim() == 2 && sa.is_contiguous() &&
                  sa.scalar_type() == torch::kFloat &&
                  sa.size(0) == a8.size(0) && sa.size(1) * 32 == a8.size(1),
              "sa must be a contiguous fp32 [rows, K / 32]");
  TORCH_CHECK(c.dim() == 2 && c.scalar_type() == torch::kHalf,
              "c must be fp16 2D");
  TORCH_CHECK(b_q_weight.dim() == 3 && b_q_weight.is_contiguous(),
              "b_q_weight must be a contiguous 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3 && b_scales.is_contiguous(),
              "b_scales must be a contiguous 3D [E, K/16, N]");
  TORCH_CHECK(cta_info.is_cuda() && cta_info.scalar_type() == torch::kInt &&
                  cta_info.dim() == 2 && cta_info.size(1) == 4 &&
                  cta_info.is_contiguous(),
              "cta_info must come from moe_prefill_cta_rdna2");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a8));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a8.size(0);
  const int size_k = (int)a8.size(1);
  const int size_n = (int)b_q_weight.size(2);
  TORCH_CHECK((int)b_q_weight.size(1) * 8 == size_k,
              "b_q_weight K-dim * 8 must equal a8.size(1)");
  TORCH_CHECK((int)b_scales.size(1) * 16 == size_k,
              "b_scales K-dim * 16 must equal a8.size(1)");
  TORCH_CHECK((int)b_scales.size(2) == size_n, "b_scales N-dim must equal N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(size_k % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(b_globals.numel() == (int64_t)b_q_weight.size(0),
              "b_globals must hold one scale per expert");
  TORCH_CHECK(top_k >= 1 && output_topk >= 0, "top_k / output_topk");
  const int ew_stride = (int)(b_q_weight.size(1) * size_n);
  const int es_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const float* tw =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  dim3 block(kNt);
  dim3 grid((unsigned)cta_info.size(0), (size_n + kNt * 4 - 1) / (kNt * 4), 1);
  nvfp4_i8_prefill_kernel<kRows, kNt, kI8Kc><<<grid, block, 0, stream>>>(
      a8.data_ptr<int8_t>(), sa.data_ptr<float>(), (half*)c.data_ptr(),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      (const uint8_t*)b_scales.data_ptr(), b_globals.data_ptr<float>(), tw,
      sorted_token_ids.data_ptr<int32_t>(),
      reinterpret_cast<const int4*>(cta_info.data_ptr()), size_m, size_n,
      size_k, (int)top_k, ew_stride, es_stride, mul_topk_weight,
      (int)output_topk);
}
