// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (QTIP bitshift trellis) dense GEMM for RDNA2/RDNA3 (gfx1030/gfx1100),
// fp16-only. REAL tile layout (locked 2026-08-26 against exllamav3_ext
// reconstruct, max-err 0.0 on 112 real + 32 synthetic tiles):
//
//   trellis: (k/16, n/16, 256*bits/16) int16, packed tail-biting stream.
//   Window p (16-bit codebook input) sits at bit (p+1)*bits - 16 mod 256*bits.
//   Weight value = decode_3inst<cb>(window_p).
//   Tile (row r, col c): window position p = (c%8)<<5 | (off(r) mod 32)
//     K=3: off(r) = 8*(r/2) + (r%2) + 2*(r>=8) + 4*(c/8)
//     K=4: off(r) = 8*(r/2) + sel(r) - 4*(c/8), sel = 7,6,5,4 by parity/half
//   suh (k,) / svh (n,) half: caller-side scales + Hadamard flips; NOT in
//   the K-dot (wiki kernels/exl3.md).
//
// Kernel geometry follows the RDNA2 W4A16 dense kernel (q_gemm_rdna2.cu):
//   THREADS_X=256, 4 N-columns per thread => 1024 N-cols per block, 8 waves
//   wave32. A is staged once into LDS for BLOCK_KN (256) K-elements per
//   block; the K-loop iterates 32 K at a time (4 x 8-col sub-tiles), and
//   B/decode is register-resident per thread with 4x fdot2 ILP. No barrier
//   inside the K loop.
//
// Helpers from exl3_dot2_common.cuh: decode_3inst, exl3_window_pos,
// exl3_window_at, atomic_add_pk4_f16.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "exl3_dot2_common.cuh"

#if defined(__HIPCC__) &&                                                    \
    (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1100__) || \
     defined(__gfx1101__) || defined(__gfx1150__) || defined(__gfx1151__) || \
     defined(__gfx1200__) || defined(__gfx1201__))
  #define __HIP__RDNA__
#endif

#define THREADS_X 256
#define BLOCK_N 1024  // N-cols per block (4 per thread)
#define BLOCK_K 256   // K-elements staged per block
#define K_TILE 16     // EXL3 tile depth
#define COL_PER_THREAD 4

namespace vllm {
namespace exl3_dot2 {

__forceinline__ __device__ int m_global(int m, int bz, int mc) {
  return bz * mc + m;
}

// v3 kernel geometry (see gemm_exl3_v3_kernel_rdna).
#define V3_THREADS 128
#define V3_TILES_PER_BLOCK 32  // one N-tile per lane
#define V3_MAX_BLOCK_K 256

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

template <int M_PER, int bits, int cb>
__global__ void gemm_exl3_kernel_rdna(
    const half* __restrict__ a,           // [size_m, size_k]
    const int16_t* __restrict__ trellis,  // [k/16, n/16, 256*bits/16]
    half* __restrict__ c,                 // [size_m, size_n]
    const int size_m, const int size_n, const int size_k) {
  constexpr int TILE_WORDS = 8 * bits;      // uint32 per 16x16 tile
  constexpr int TILE_I16 = 2 * TILE_WORDS;  // int16 per tile
  const int t = threadIdx.x;
  const int n0 = blockIdx.x * BLOCK_N + t * COL_PER_THREAD;
  const int n_tiles_total = size_n / 16;
  const int offset_k = blockIdx.y * BLOCK_K;
  const int end_k = min(offset_k + BLOCK_K, size_k);

  // A staging: [M_PER][BLOCK_K] halves in LDS, PAD for bank conflicts.
  constexpr int LDS_PAD = 8;
  __shared__ half s_a[M_PER][BLOCK_K + LDS_PAD];

  // Each thread loads 4 fp16 per M row (BLOCK_K/THREADS_X = 1 K each).
  static_assert(BLOCK_K == THREADS_X, "one K-elem per thread");
  for (int m = 0; m < M_PER; ++m) {
    const int mr = m_global(m, blockIdx.z, M_PER);
    half av = (mr < size_m) ? a[(int64_t)mr * size_k + offset_k + t]
                            : __float2half_rn(0.0f);
    s_a[m][t] = av;
  }
  __syncthreads();

  if (n0 >= size_n) return;

  // Per-thread accumulators: M_PER x 4 N-cols.
  float acc[M_PER][4];
  #pragma unroll
  for (int m = 0; m < M_PER; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  // n-tile(s) touched by this thread's 4 columns (cols n0..n0+3 may span
  // one or two 16-col tiles).
  const int tile_idx0 = (n0) / 16;
  const int tile_idx1 = (n0 + 3) / 16;
  const bool two_tiles = tile_idx1 != tile_idx0;

  // K-loop: each iteration decodes a 16x(4 cols) slice in the codebook
  // domain and does M_PER*4 dot products. No barrier inside.
  for (int k_tile = 0; k_tile < (end_k - offset_k) / K_TILE; ++k_tile) {
    const int16_t* tile0 =
        trellis +
        ((int64_t)(offset_k / K_TILE + k_tile) * n_tiles_total + tile_idx0) *
            TILE_I16;
    const int16_t* tile1 = two_tiles ? tile0 + TILE_I16 : nullptr;

    // Decode 4 cols x 16 K deltas.
    half w0[4][16], w1[4][16];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int n_here = n0 + j;
      int nt = (n_here / 16) - tile_idx0;  // 0 or 1
      int ccol = n_here % 16;
      const int16_t* tp = nt ? tile1 : tile0;
  #pragma unroll
      for (int r = 0; r < 16; ++r) {
        const int p = exl3_window_pos<bits>(r, ccol);
        const uint32_t win =
            exl3_window_at<bits>(reinterpret_cast<const uint32_t*>(tp), p);
        (nt ? w1[j] : w0[j])[r] = decode_3inst<cb>(win);
      }
    }

    // Accumulate M rows x 4 cols via 8 fdot2 each.
  #pragma unroll
    for (int m = 0; m < M_PER; ++m) {
      const half* ak = &s_a[m][k_tile * K_TILE];
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const half* wjk = ((n0 + j) / 16 != tile_idx0) ? w1[j] : w0[j];
  #pragma unroll
        for (int h = 0; h < K_TILE / 2; ++h) {
          half2 a2 = __halves2half2(ak[2 * h], ak[2 * h + 1]);
          half2 w2 = __halves2half2(wjk[2 * h], wjk[2 * h + 1]);
          acc[m][j] = __builtin_amdgcn_fdot2(w2, a2, acc[m][j], false);
        }
      }
    }
  }

  // Epilogue: atomically accumulate 4 columns per row (multi-K-block adds).
  #pragma unroll
  for (int m = 0; m < M_PER; ++m) {
    const int mr = m_global(m, blockIdx.z, M_PER);
    if (mr >= size_m) continue;
    half* out = c + (int64_t)mr * size_n + n0;
    half2 r01 =
        __halves2half2(__float2half_rn(acc[m][0]), __float2half_rn(acc[m][1]));
    half2 r23 =
        __halves2half2(__float2half_rn(acc[m][2]), __float2half_rn(acc[m][3]));
    if (gridDim.y > 1) {
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      union {
        unsigned long long u;
        half2 h2[2];
      } v;
      v.h2[0] = r01;
      v.h2[1] = r23;
      *reinterpret_cast<unsigned long long*>(out) = v.u;
    }
  }
}

  // Small-M (decode) variant for bits=3: grain-based decode with per-k_tile
  // tile-word register staging. A thread owns 16 N-cols = exactly one 16x16
  // tile; per k_tile it loads the tile's 24 uint32 words once and derives all
  // 32 grains (8 windows each) via fshift from registers. Cuts trellis load
  // instructions ~8x vs the per-weight window reads above: bit-identical
  // output, 4.2-4.8x faster at M=1 on gfx1030 (microbench 2026-09-04).
  // Grain g = j*4+mg covers windows p = 8g..8g+7:
  //   i=0,1 -> rows 2mg,2mg+1       col j    (c/8 = 0)
  //   i=2,3 -> rows 8+2mg,8+2mg+1   col j    (c/8 = 0)
  //   i=4,5 -> rows 2mg,2mg+1       col j+8  (c/8 = 1)
  //   i=6,7 -> rows 8+2mg,8+2mg+1   col j+8  (c/8 = 1)
  #define V2_THREADS_X 64
  #define V2_BLOCK_N 1024  // 16 cols x 64 threads
  #define V2_BLOCK_K 128
template <int M_PER, int cb>
__global__ void gemm_exl3_v2_kernel_rdna(const half* __restrict__ a,
                                         const int16_t* __restrict__ trellis,
                                         half* __restrict__ c, const int size_m,
                                         const int size_n, const int size_k) {
  constexpr int V2_COL = 16;
  constexpr int V2_KTILE = 16;
  constexpr int NW = 24;  // bits=3 tile words
  const int t = threadIdx.x;
  const int n0 = blockIdx.x * V2_BLOCK_N + t * V2_COL;
  const int n_tiles_total = size_n / 16;
  const int offset_k = blockIdx.y * V2_BLOCK_K;
  const int end_k = min(offset_k + V2_BLOCK_K, size_k);
  constexpr int LDS_PAD = 8;
  __shared__ half s_a[M_PER][V2_BLOCK_K + LDS_PAD];
  #pragma unroll 1
  for (int m = 0; m < M_PER; ++m) {
    const int mr = blockIdx.z * M_PER + m;
    for (int kk = t; kk < V2_BLOCK_K; kk += V2_THREADS_X) {
      s_a[m][kk] = (mr < size_m) ? a[(int64_t)mr * size_k + offset_k + kk]
                                 : __float2half_rn(0.0f);
    }
  }
  __syncthreads();
  if (n0 >= size_n) return;
  float acc[M_PER][V2_COL];
  #pragma unroll
  for (int m = 0; m < M_PER; ++m)
  #pragma unroll
    for (int j = 0; j < V2_COL; ++j) acc[m][j] = 0.0f;

  const int tile_idx = n0 / 16;

  for (int kt = 0; kt < (end_k - offset_k) / V2_KTILE; ++kt) {
    const uint32_t* tp = reinterpret_cast<const uint32_t*>(
        trellis +
        ((int64_t)(offset_k / V2_KTILE + kt) * n_tiles_total + tile_idx) * 2 *
            NW);
    uint32_t tw[NW];
  #pragma unroll
    for (int w = 0; w < NW; ++w) tw[w] = tp[w];

  #pragma unroll
    for (int g = 0; g < 32; ++g) {
      const int j = g >> 2;
      const int mg = g & 3;
      half2 wpair[4];
  #pragma unroll
      for (int q = 0; q < 4; ++q) {
        // even window position p = 8g + 2q; tail-biting pair read at tpos=p/2
        const int tpos = 4 * g + q;
        const int b0 = tpos * 6 + 755;  // tpos*2*bits + bits - 16 + 256*bits
        const int b2 = b0 + 19;         // b0 + bits + 16
        const int i1_raw = (b2 - 1) >> 5;
        const int i0 = (b0 >> 5) % NW;
        const int i1 = i1_raw % NW;
        // s1 must use the pre-modulo word index (tail-biting wrap): the
        // shift count is only valid in [0,31]; a negative count is UB.
        const int s1 = (i1_raw + 1) * 32 - b2;
        uint32_t w1f = fshift(tw[i1], tw[i0], s1);
        wpair[q] = __halves2half2(
            decode_3inst<cb>((w1f >> 3) & 0xffffu),  // even p -> w0
            decode_3inst<cb>(w1f & 0xffffu));        // odd p  -> w1
      }
      const int r_lo = 2 * mg, r_hi = 8 + 2 * mg;
  #pragma unroll
      for (int m = 0; m < M_PER; ++m) {
        const int mr = blockIdx.z * M_PER + m;
        if (mr >= size_m) continue;
        const half* ak = &s_a[m][kt * V2_KTILE];
        half2 a_lo = __halves2half2(ak[r_lo], ak[r_lo + 1]);
        half2 a_hi = __halves2half2(ak[r_hi], ak[r_hi + 1]);
        acc[m][j] = __builtin_amdgcn_fdot2(wpair[0], a_lo, acc[m][j], false);
        acc[m][j] = __builtin_amdgcn_fdot2(wpair[1], a_hi, acc[m][j], false);
        acc[m][j + 8] =
            __builtin_amdgcn_fdot2(wpair[2], a_lo, acc[m][j + 8], false);
        acc[m][j + 8] =
            __builtin_amdgcn_fdot2(wpair[3], a_hi, acc[m][j + 8], false);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M_PER; ++m) {
    const int mr = blockIdx.z * M_PER + m;
    if (mr >= size_m) continue;
    half* out = c + (int64_t)mr * size_n + n0;
  #pragma unroll
    for (int jj = 0; jj < V2_COL; jj += 4) {
      half2 r01 = __halves2half2(__float2half_rn(acc[m][jj]),
                                 __float2half_rn(acc[m][jj + 1]));
      half2 r23 = __halves2half2(__float2half_rn(acc[m][jj + 2]),
                                 __float2half_rn(acc[m][jj + 3]));
      if (gridDim.y > 1) {
        atomic_add_pk4_f16(out + jj, r01, r23);
      } else {
        union {
          unsigned long long u;
          half2 h2[2];
        } v;
        v.h2[0] = r01;
        v.h2[1] = r23;
        *reinterpret_cast<unsigned long long*>(out + jj) = v.u;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v3: decode each tile once for all M rows (verify / small-batch decode).
//
// The v2 kernel keeps a whole 16x16 tile per thread, so its accumulators
// grow as 16 * M_PER and it z-splits M in blocks of 4: an MTP verify step
// (M = 3 * num_seqs, e.g. 24) decodes every trellis tile 6 times, and the
// generic kernel used above M = 8 decodes it 3 times with a much slower
// per-weight window read. Here a 128-thread block is 4 waves; each lane
// owns one N-tile and each wave one 4-column quarter of it, chosen
// wave-uniformly (readfirstlane) so every window position and tile-word
// index stays a compile-time constant (registers, no scratch). A thread
// decodes 4 cols x 16 K = 32 half2 per k-tile ONCE and dots them against
// up to M_PER rows of A read from LDS with 2 x 128-bit broadcasts per row.
// Rows beyond size_m are skipped with a wave-uniform branch, so M_PER is a
// capacity, not a cost. The tile words are read by all 4 waves of the
// block (L0 hits). The 4 output columns of a thread are contiguous, so the
// split-K epilogue is one 64-bit CAS per row (atomic_add_pk4_f16).
// ---------------------------------------------------------------------------
// Window position of weight (r, c) inside a 16x16 tile: exl3_window_pos as
// a constexpr so it folds once the loops below are unrolled.
template <int bits>
__forceinline__ __device__ constexpr int exl3_wpos_c(int r, int c) {
  int off = (bits == 4)
                ? 8 * (r / 2) +
                      ((r & 1) ? ((r < 8) ? 6 : 4) : ((r < 8) ? 7 : 5)) -
                      4 * (c / 8)
                : 8 * (r / 2) + (r & 1) + ((r >= 8) ? 2 : 0) + 4 * (c / 8);
  off %= 32;
  if (off < 0) off += 32;
  return ((c & 7) << 5) | off;
}

// 16 + extra stream bits ending at window p+extra/bits: low 16 bits hold
// window p + (extra ? 1 : 0); window p sits `extra` bits higher. Big-endian
// bit order across tile words, tail-biting (see the v2 kernel).
template <int bits>
__forceinline__ __device__ uint32_t exl3_funnel(const uint32_t* tw, int p,
                                                int extra) {
  constexpr int NW = 8 * bits;
  const int b0 = (p + 1) * bits - 16 + 256 * bits;
  const int b2 = b0 + 16 + extra;
  const int i1_raw = (b2 - 1) >> 5;
  const int i0 = (b0 >> 5) % NW;
  const int i1 = i1_raw % NW;
  const int s1 = (i1_raw + 1) * 32 - b2;
  return fshift(tw[i1], tw[i0], s1);
}

// Two 16-bit states -> {d(x0), d(x1)}. mul1 uses v_dot4_u32_u8 for the byte
// sum (bit-identical to the 4 shifted adds in decode_3inst<2>) and one
// packed fma for both halves.
template <int cb>
__forceinline__ __device__ half2 exl3_decode_pair(uint32_t x0, uint32_t x1) {
  if constexpr (cb == 2) {
  #if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    const uint32_t s0 =
        __builtin_amdgcn_udot4(x0 * 0x83DCD12Du, 0x01010101u, 0x6400u, false);
    const uint32_t s1 =
        __builtin_amdgcn_udot4(x1 * 0x83DCD12Du, 0x01010101u, 0x6400u, false);
    const uint32_t packed = (s0 & 0xffffu) | (s1 << 16);
    const half2 v = __builtin_bit_cast(half2, packed);
    const half2 k_inv = __builtin_bit_cast(half2, 0x1eee1eeeu);
    const half2 k_bias = __builtin_bit_cast(half2, 0xc931c931u);
    return __hfma2(v, k_inv, k_bias);
  #else
    return __halves2half2(decode_3inst<cb>(x0), decode_3inst<cb>(x1));
  #endif
  } else {
    return __halves2half2(decode_3inst<cb>(x0), decode_3inst<cb>(x1));
  }
}

// Weights (rows 2h, 2h+1) of column c as one half2, from the tile words.
template <int bits, int cb>
__forceinline__ __device__ half2 exl3_rowpair(const uint32_t* tw, int h,
                                              int c) {
  // Stream window of tile position p (exl3_window_at). bits 4 follows the
  // dq8_aligned reader: positions run backwards inside each group of 8;
  // bits 3 and 6 are in stream order.
  const int p0 = exl3_wpos_c<bits>(2 * h, c) ^ (bits == 4 ? 7 : 0);
  const int p1 = exl3_wpos_c<bits>(2 * h + 1, c) ^ (bits == 4 ? 7 : 0);
  if (p1 == p0 + 1) {  // row 2h+1 is the next window (bits 3 and 4)
    const uint32_t f = exl3_funnel<bits>(tw, p0, bits);
    return exl3_decode_pair<cb>((f >> bits) & 0xffffu, f & 0xffffu);
  }
  if (p0 == p1 + 1) {
    const uint32_t f = exl3_funnel<bits>(tw, p1, bits);
    return exl3_decode_pair<cb>(f & 0xffffu, (f >> bits) & 0xffffu);
  }
  const uint32_t f0 = exl3_funnel<bits>(tw, p0, 0) & 0xffffu;
  const uint32_t f1 = exl3_funnel<bits>(tw, p1, 0) & 0xffffu;
  return exl3_decode_pair<cb>(f0, f1);
}

template <int M_PER, int bits, int cb, int Q>
__forceinline__ __device__ void gemm_exl3_v3_body(
    const half (*s_a)[V3_MAX_BLOCK_K + 8], const int16_t* __restrict__ trellis,
    half* __restrict__ c, const int rows, const int m0, const int size_n,
    const int k_tile0, const int k_tiles, const int nt, const bool split_k) {
  constexpr int NW = 8 * bits;
  constexpr int CBASE = (Q & 1) * 4 + (Q >> 1) * 8;
  const int n_tiles_total = size_n / 16;
  float acc[M_PER][4];
  #pragma unroll
  for (int m = 0; m < M_PER; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  for (int kt = 0; kt < k_tiles; ++kt) {
    const uint4* tp4 = reinterpret_cast<const uint4*>(
        trellis + ((int64_t)(k_tile0 + kt) * n_tiles_total + nt) * (2 * NW));
    uint32_t tw[NW];
  #pragma unroll
    for (int w = 0; w < NW / 4; ++w) {
      const uint4 v = tp4[w];
      tw[4 * w + 0] = v.x;
      tw[4 * w + 1] = v.y;
      tw[4 * w + 2] = v.z;
      tw[4 * w + 3] = v.w;
    }
    half2 wv[4][8];
  #pragma unroll
    for (int j = 0; j < 4; ++j)
  #pragma unroll
      for (int h = 0; h < 8; ++h)
        wv[j][h] = exl3_rowpair<bits, cb>(tw, h, CBASE + j);

  #pragma unroll
    for (int m = 0; m < M_PER; ++m) {
      // Wave-uniform guard; a `break` here would stop the unroll and push
      // acc[][] to scratch.
      if (m >= rows) continue;
      const uint4* ar = reinterpret_cast<const uint4*>(&s_a[m][kt * K_TILE]);
      const uint4 lo = ar[0];
      const uint4 hi = ar[1];
      const half2 a2[8] = {
          __builtin_bit_cast(half2, lo.x), __builtin_bit_cast(half2, lo.y),
          __builtin_bit_cast(half2, lo.z), __builtin_bit_cast(half2, lo.w),
          __builtin_bit_cast(half2, hi.x), __builtin_bit_cast(half2, hi.y),
          __builtin_bit_cast(half2, hi.z), __builtin_bit_cast(half2, hi.w)};
  #pragma unroll
      for (int j = 0; j < 4; ++j)
  #pragma unroll
        for (int h = 0; h < 8; ++h)
          acc[m][j] = __builtin_amdgcn_fdot2(wv[j][h], a2[h], acc[m][j], false);
    }
  }

  #pragma unroll
  for (int m = 0; m < M_PER; ++m) {
    if (m >= rows) continue;
    half* out = c + (int64_t)(m0 + m) * size_n + nt * 16 + CBASE;
    const half2 r01 =
        __halves2half2(__float2half_rn(acc[m][0]), __float2half_rn(acc[m][1]));
    const half2 r23 =
        __halves2half2(__float2half_rn(acc[m][2]), __float2half_rn(acc[m][3]));
    if (split_k) {
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      union {
        unsigned long long u;
        half2 h2[2];
      } v;
      v.h2[0] = r01;
      v.h2[1] = r23;
      *reinterpret_cast<unsigned long long*>(out) = v.u;
    }
  }
}

template <int M_PER, int bits, int cb>
__global__ void __launch_bounds__(V3_THREADS)
    gemm_exl3_v3_kernel_rdna(const half* __restrict__ a,
                             const int16_t* __restrict__ trellis,
                             half* __restrict__ c, const int size_m,
                             const int size_n, const int size_k,
                             const int block_k) {
  __shared__ __attribute__((aligned(16))) half s_a[M_PER][V3_MAX_BLOCK_K + 8];
  const int t = threadIdx.x;
  const int offset_k = blockIdx.y * block_k;
  const int len_k = min(block_k, size_k - offset_k);
  const int m0 = blockIdx.z * M_PER;
  const int rows = min(M_PER, size_m - m0);
  // A rows -> LDS, 8 halves per thread per step (K is a multiple of 16).
  for (int idx = t * 8; idx < rows * len_k; idx += V3_THREADS * 8) {
    const int m = idx / len_k;
    const int kk = idx - m * len_k;
    *reinterpret_cast<uint4*>(&s_a[m][kk]) = *reinterpret_cast<const uint4*>(
        &a[(int64_t)(m0 + m) * size_k + offset_k + kk]);
  }
  __syncthreads();
  const int nt = blockIdx.x * V3_TILES_PER_BLOCK + (t & 31);
  if (nt >= size_n / 16) return;
  const int q = __builtin_amdgcn_readfirstlane(t >> 5);
  const int k_tile0 = offset_k / K_TILE;
  const int k_tiles = len_k / K_TILE;
  const bool split_k = gridDim.y > 1;
  switch (q) {
    case 0:
      gemm_exl3_v3_body<M_PER, bits, cb, 0>(s_a, trellis, c, rows, m0, size_n,
                                            k_tile0, k_tiles, nt, split_k);
      break;
    case 1:
      gemm_exl3_v3_body<M_PER, bits, cb, 1>(s_a, trellis, c, rows, m0, size_n,
                                            k_tile0, k_tiles, nt, split_k);
      break;
    case 2:
      gemm_exl3_v3_body<M_PER, bits, cb, 2>(s_a, trellis, c, rows, m0, size_n,
                                            k_tile0, k_tiles, nt, split_k);
      break;
    default:
      gemm_exl3_v3_body<M_PER, bits, cb, 3>(s_a, trellis, c, rows, m0, size_n,
                                            k_tile0, k_tiles, nt, split_k);
      break;
  }
}

#else  // non-RDNA: empty stub for symbol parity

template <int M_PER, int bits, int cb>
__global__ void gemm_exl3_kernel_rdna(const half*, const int16_t*, half*,
                                      const int, const int, const int) {}
template <int M_PER, int cb>
__global__ void gemm_exl3_v2_kernel_rdna(const half*, const int16_t*, half*,
                                         const int, const int, const int) {}
template <int M_PER, int bits, int cb>
__global__ void gemm_exl3_v3_kernel_rdna(const half*, const int16_t*, half*,
                                         const int, const int, const int,
                                         const int) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

__forceinline__ int divide_up(int x, int y) { return (x + y - 1) / y; }

template <int M_PER, int bits, int cb>
void launch_mcb(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                int sk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(divide_up(sn, BLOCK_N), divide_up(sk, BLOCK_K),
            divide_up(sm, M_PER));
  gemm_exl3_kernel_rdna<M_PER, bits, cb>
      <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
}

template <int M_PER, int bits>
void launch_mb(const half* a, const int16_t* trellis, half* c, int sm, int sn,
               int sk, int cb, cudaStream_t stream) {
  if (cb == 0)
    launch_mcb<M_PER, bits, 0>(a, trellis, c, sm, sn, sk, stream);
  else if (cb == 1)
    launch_mcb<M_PER, bits, 1>(a, trellis, c, sm, sn, sk, stream);
  else if (cb == 2)
    launch_mcb<M_PER, bits, 2>(a, trellis, c, sm, sn, sk, stream);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_PER>
void launch_v2(const half* a, const int16_t* trellis, half* c, int sm, int sn,
               int sk, int cb, cudaStream_t stream) {
  dim3 block(V2_THREADS_X);
  dim3 grid(divide_up(sn, V2_BLOCK_N), divide_up(sk, V2_BLOCK_K),
            divide_up(sm, M_PER));
  if (cb == 0)
    gemm_exl3_v2_kernel_rdna<M_PER, 0>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else if (cb == 1)
    gemm_exl3_v2_kernel_rdna<M_PER, 1>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else if (cb == 2)
    gemm_exl3_v2_kernel_rdna<M_PER, 2>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_PER>
void launch_m(const half* a, const int16_t* trellis, half* c, int sm, int sn,
              int sk, int bits, int cb, cudaStream_t stream) {
  switch (bits) {
    case 1:
      launch_mb<M_PER, 1>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 2:
      launch_mb<M_PER, 2>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 3:
      launch_mb<M_PER, 3>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 4:
      launch_mb<M_PER, 4>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 5:
      launch_mb<M_PER, 5>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 6:
      launch_mb<M_PER, 6>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 7:
      launch_mb<M_PER, 7>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 8:
      launch_mb<M_PER, 8>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    default:
      TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported bits=", bits);
  }
}

template <int M_PER, int bits>
void launch_v3_mb(const half* a, const int16_t* trellis, half* c, int sm,
                  int sn, int sk, int cb, int block_k, cudaStream_t stream) {
  dim3 block(V3_THREADS);
  dim3 grid(divide_up(sn / 16, V3_TILES_PER_BLOCK), divide_up(sk, block_k),
            divide_up(sm, M_PER));
  if (cb == 0)
    gemm_exl3_v3_kernel_rdna<M_PER, bits, 0>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk, block_k);
  else if (cb == 1)
    gemm_exl3_v3_kernel_rdna<M_PER, bits, 1>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk, block_k);
  else if (cb == 2)
    gemm_exl3_v3_kernel_rdna<M_PER, bits, 2>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk, block_k);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_PER>
void launch_v3_m(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                 int sk, int bits, int cb, int block_k, cudaStream_t stream) {
  if (bits == 3)
    launch_v3_mb<M_PER, 3>(a, trellis, c, sm, sn, sk, cb, block_k, stream);
  else if (bits == 4)
    launch_v3_mb<M_PER, 4>(a, trellis, c, sm, sn, sk, cb, block_k, stream);
  else
    launch_v3_mb<M_PER, 6>(a, trellis, c, sm, sn, sk, cb, block_k, stream);
}

// The v3 kernel is tuned and validated on RDNA2 (gfx103x) only; other RDNA
// targets keep the v2 / generic kernels.
static bool exl3_v3_enabled() {
  static const int env = [] {
    const char* e = std::getenv("VLLM_EXL3_GEMM_V3");
    return e ? std::atoi(e) : 1;
  }();
  if (env == 0) return false;
  thread_local int cached_dev = -1;
  thread_local bool cached_ok = false;
  const int dev = at::cuda::current_device();
  if (dev != cached_dev) {
    hipDeviceProp_t prop;
    cached_ok = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx103", 6) == 0;
    cached_dev = dev;
  }
  return cached_ok;
}

// Split K so the grid fills the GPU (72 CUs on a V620) without paying more
// split-K atomics than needed. VLLM_EXL3_V3_BLOCK_K overrides (tuning).
static int exl3_v3_block_k(int sn, int sk) {
  static const int env = [] {
    const char* e = std::getenv("VLLM_EXL3_V3_BLOCK_K");
    return e ? std::atoi(e) : 0;
  }();
  if (env >= 16 && env <= V3_MAX_BLOCK_K && env % 16 == 0) return env;
  const int gx = divide_up(sn / 16, V3_TILES_PER_BLOCK);
  return gx * divide_up(sk, 256) >= 160 ? 256 : 128;
}

void launch_tile(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                 int sk, int bits, int cb, cudaStream_t stream) {
  // Decode-once kernel for decode and MTP-verify batches (M <= 64, the
  // CG-path buffer size; larger M goes through decode + rocBLAS in Python).
  // bits 3: body, 4: MTP draft layer, 6: lm_head.
  if ((bits == 3 || bits == 4 || bits == 6) && sm <= 64 && exl3_v3_enabled()) {
    const int block_k = exl3_v3_block_k(sn, sk);
    if (sm <= 8)
      launch_v3_m<8>(a, trellis, c, sm, sn, sk, bits, cb, block_k, stream);
    else if (sm <= 16)
      launch_v3_m<16>(a, trellis, c, sm, sn, sk, bits, cb, block_k, stream);
    else
      launch_v3_m<32>(a, trellis, c, sm, sn, sk, bits, cb, block_k, stream);
    return;
  }
  // bits=3 decode batches (sm <= max_num_seqs): the grain-based v2 kernel.
  // M_PER=4 z-split beats M_PER=8 (register pressure); prefill chunks
  // (sm > 8) keep the original kernel.
  if (bits == 3 && sm <= 8) {
    if (sm == 1)
      launch_v2<1>(a, trellis, c, sm, sn, sk, cb, stream);
    else if (sm == 2)
      launch_v2<2>(a, trellis, c, sm, sn, sk, cb, stream);
    else
      launch_v2<4>(a, trellis, c, sm, sn, sk, cb, stream);
    return;
  }
  if (sm == 1)
    launch_m<1>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 3)
    launch_m<2>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 7)
    launch_m<4>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else
    launch_m<8>(a, trellis, c, sm, sn, sk, bits, cb, stream);
}

}  // namespace exl3_dot2
}  // namespace vllm

namespace vllm {
namespace exl3_dot2 {

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

// One block per 16x16 tile, one output element per thread. The GEMM
// kernel re-runs this exact decode per M-block (M_PER=8 cap); pulling it
// out lets prefill decode each tile once and hand the dot work to rocBLAS.
template <int bits, int cb>
__global__ void decode_trellis_kernel_rdna(const int16_t* __restrict__ trellis,
                                           half* __restrict__ out,
                                           const int size_k, const int size_n) {
  const int kt = blockIdx.x;
  const int nt = blockIdx.y;
  const int n_tiles = size_n / 16;
  const int16_t* tile = trellis + ((int64_t)kt * n_tiles + nt) * (2 * 8 * bits);
  const int t = threadIdx.x;
  const int r = t / 16;
  const int c = t % 16;
  const int p = exl3_window_pos<bits>(r, c);
  const uint32_t win =
      exl3_window_at<bits>(reinterpret_cast<const uint32_t*>(tile), p);
  out[(int64_t)(kt * 16 + r) * size_n + (nt * 16 + c)] = decode_3inst<cb>(win);
}

#else  // non-RDNA: empty stub for symbol parity

template <int bits, int cb>
__global__ void decode_trellis_kernel_rdna(const int16_t*, half*, const int,
                                           const int) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

template <int bits, int cb>
void launch_decode_trellis(const int16_t* trellis, half* out, int sk, int sn,
                           cudaStream_t stream) {
  dim3 grid(sk / 16, sn / 16);
  decode_trellis_kernel_rdna<bits, cb>
      <<<grid, dim3(256), 0, stream>>>(trellis, out, sk, sn);
}

template <int bits>
void launch_decode_cb(const int16_t* trellis, half* out, int sk, int sn, int cb,
                      cudaStream_t stream) {
  if (cb == 0)
    launch_decode_trellis<bits, 0>(trellis, out, sk, sn, stream);
  else if (cb == 1)
    launch_decode_trellis<bits, 1>(trellis, out, sk, sn, stream);
  else if (cb == 2)
    launch_decode_trellis<bits, 2>(trellis, out, sk, sn, stream);
  else
    TORCH_CHECK(false, "exl3_decode_trellis_rdna2: unsupported cb=", cb);
}

}  // namespace exl3_dot2
}  // namespace vllm

void exl3_decode_trellis_rdna2(torch::Tensor trellis, torch::Tensor out,
                               int64_t bits, int64_t cb) {
  const int64_t size_k = trellis.size(0) * 16;
  const int64_t size_n = trellis.size(1) * 16;
  TORCH_CHECK(trellis.is_cuda() && out.is_cuda(), "tensors must be CUDA/HIP");
  TORCH_CHECK(trellis.dim() == 3, "trellis 3D [K/16, N/16, W]");
  TORCH_CHECK(out.scalar_type() == torch::kHalf && out.size(0) == size_k &&
                  out.size(1) == size_n,
              "out must be fp16 [K, N]");
  TORCH_CHECK(bits >= 1 && bits <= 8,
              "exl3_decode_trellis_rdna2: bits must be 1..8");
  const at::cuda::OptionalCUDAGuard dg(device_of(trellis));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (bits) {
    case 1:
      vllm::exl3_dot2::launch_decode_cb<1>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 2:
      vllm::exl3_dot2::launch_decode_cb<2>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 3:
      vllm::exl3_dot2::launch_decode_cb<3>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 4:
      vllm::exl3_dot2::launch_decode_cb<4>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 5:
      vllm::exl3_dot2::launch_decode_cb<5>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 6:
      vllm::exl3_dot2::launch_decode_cb<6>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    case 7:
      vllm::exl3_dot2::launch_decode_cb<7>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
    default:
      vllm::exl3_dot2::launch_decode_cb<8>((const int16_t*)trellis.data_ptr(),
                                           (half*)out.data_ptr(), (int)size_k,
                                           (int)size_n, (int)cb, stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

void exl3_hadamard_128(torch::Tensor input, torch::Tensor output,
                       torch::optional<torch::Tensor> pre_scale,
                       torch::optional<torch::Tensor> post_scale, double scale);
void exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
                     int64_t bits, int64_t cb);

// One decode linear: H_K(x, suh) -> trellis GEMM -> H_N(mid, svh).
// The three kernels stay the checked implementations; they are queued on
// the current stream so Python does not allocate or dispatch between them.
void exl3_project_rdna2(torch::Tensor x, torch::Tensor xh, torch::Tensor mid,
                        torch::Tensor out, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t bits,
                        int64_t cb) {
  TORCH_CHECK(x.is_cuda() && xh.is_cuda() && mid.is_cuda() && out.is_cuda(),
              "exl3_project_rdna2 tensors must be CUDA/HIP");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "exl3_project_rdna2 fp16 only");
  TORCH_CHECK(x.dim() == 2 && xh.dim() == 2 && mid.dim() == 2 && out.dim() == 2,
              "exl3_project_rdna2 expects 2D activations");
  TORCH_CHECK(xh.size(0) == x.size(0) && xh.size(1) == x.size(1),
              "xh must be [M, K]");
  TORCH_CHECK(mid.size(0) == x.size(0) && out.size(0) == x.size(0),
              "mid/out row count must match M");
  TORCH_CHECK(mid.size(1) == out.size(1), "mid and out N must match");
  TORCH_CHECK(x.is_contiguous() && xh.is_contiguous() && mid.is_contiguous() &&
                  out.is_contiguous(),
              "exl3_project_rdna2 activations must be contiguous");
  TORCH_CHECK(trellis.is_contiguous(), "trellis slice must be contiguous");
  TORCH_CHECK(suh.is_contiguous() && svh.is_contiguous(),
              "suh and svh must be contiguous");
  const at::cuda::OptionalCUDAGuard dg(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(mid.data_ptr(), 0,
                                 mid.numel() * mid.element_size(), stream));
  exl3_hadamard_128(x, xh, suh, c10::nullopt, 1.0);
  exl3_gemm_rdna2(xh, mid, trellis, bits, cb);
  exl3_hadamard_128(mid, out, c10::nullopt, svh, 1.0);
}

void exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
                     int64_t bits, int64_t cb) {
  // Derive sizes from tensors. Taking them as Python ints at the call
  // site would force dynamo to specialize the symbolic
  // input_ids.size()[0] (= a.size(0)) to the trace-time batch (2048),
  // firing ConstraintViolationError against V2's dynamic marker.
  const int64_t size_m = a.size(0);
  const int64_t size_n = c.size(1);
  const int64_t size_k = a.size(1);
  TORCH_CHECK(a.is_cuda() && c.is_cuda() && trellis.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(a.dim() == 2 && trellis.dim() == 3,
              "a 2D, trellis 3D [K/16, N/16, W]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "exl3_gemm_rdna2 fp16 only");
  TORCH_CHECK(size_k % 16 == 0 && size_n % 16 == 0, "K and N multiples of 16");
  const at::cuda::OptionalCUDAGuard dg(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Caller must pre-zero c (atomic accumulation, W4A16/mxfp4 convention).
  vllm::exl3_dot2::launch_tile((const half*)a.data_ptr(),
                               (const int16_t*)trellis.data_ptr(),
                               (half*)c.data_ptr(), (int)size_m, (int)size_n,
                               (int)size_k, (int)bits, (int)cb, stream);
}