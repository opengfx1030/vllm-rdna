// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// WMMA grouped W4A16 GPTQ GEMM for MoE prefill on RDNA3 (gfx1100).
//
// Sibling of moe_q_gemm_rdna3.cu, which stays the decode path and the
// fallback (scalar v_dot2 kernel, K split over blockIdx.z, atomic write-back).
// This kernel runs the experts of a token block on the matrix unit
// (v_wmma_f32_16x16x16_{f16,bf16}_w32) and reduces the whole K dimension in
// registers: fp32 accumulators, no split-K, no atomics on the GEMM result.
// It lives in its own translation unit for the reason given in
// q_gemm_rdna3_wmma.cu (a shared TU with the scalar kernels miscompiled the
// scalar path) and keeps its dequant helpers private for the same reason.
//
// Weight format (identical to moe_gptq_gemm_rdna3): b_q_weight [E, K/8, N]
// uint32 GPTQ-shuffled, b_scales [E, groups, N], b_qzeros [E, groups, N/8]
// (4-bit packed, GPTQ v1: effective zero = stored zero + 1).
//
// Routing contract (identical): sorted_token_ids[tb * BM + r] is the pair id
// of row r of token block tb; the row of `a` is pair / top_k and the padding
// sentinel is any pair with pair / top_k >= size_m; expert_ids[tb] == -1 skips
// the block; blocks at or past num_tokens_post_padded are skipped.
//
// Outputs (picked by the dtype of c):
//   * c is fp16/bf16 [pairs, N], output_topk == 0: every scheduled pair row
//     (valid pair, block not skipped) is written exactly once, as
//     round_T(acc * topk_weight) with ONE rounding. Rows of padding and of
//     skipped blocks are not touched.
//   * c is int64 [rows, N + 1], fp16 activations only: the Q32 fixed-point
//     accumulator of the scalar kernel (32 fractional bits, sticky row status
//     in column N), one int64 atomic add per output element and pair.
//     output_topk > 0 folds pair p into row p / output_topk (the top-k
//     reduction), output_topk == 0 keeps one row per pair.
//     moe_fixedpoint_to_half_rdna3 finishes the accumulator.
//
// ---------------------------------------------------------------------------
// Mapping (tests/kernels/quantization/test_rdna3_moe_wmma_model.py emulates
// exactly this, index for index, and parses the Tile table below)
// ---------------------------------------------------------------------------
// BM = block_size_m in {16, 32, 64}, MT = BM / 16, W waves of 32 lanes,
// NF fragments of 16 columns per wave, TILE_N = W * NF * 16, KC = K columns of
// the activation tile staged in LDS per chunk (KC / 32 "sub steps" of 32 K,
// each sub step = two WMMA K steps of 16).
//
//   grid  = (num_token_blocks, ceil(N / TILE_N)), block = W * 32 threads
//   block (tb, ny): rows  r in [0, BM): pair = sorted_token_ids[tb * BM + r]
//                   cols  [ny * TILE_N, ny * TILE_N + TILE_N)
//   wave w, fragment j  : columns F(w, j) + [0, 16),
//                         F(w, j) = ny * TILE_N + (w * NF + j) * 16.
//                         F >= N is clamped to N - 16 for every load (N % 16 ==
//                         0, so a fragment is whole or absent); a clamped
//                         fragment runs the same instructions and stores
//                         nothing.
//
// WMMA fragments of v_wmma_f32_16x16x16 (lane l, lo = l & 15, hi = l >> 4;
// validated on gfx1100 by the dense kernel in q_gemm_rdna3_wmma.cu):
//   A: 16 values per lane, lanes l and l + 16 hold the same data:
//        a[s] = A[m = lo][k = s]
//   B: 16 values per lane, lanes l and l + 16 hold the same data:
//        b[s] = B[k = s][n = lo]
//   D: 8 fp32 per lane:  d[i] = D[m = 2 * i + hi][n = lo]
//
// Activation tile (A): the BM x KC chunk is gathered by all threads, thread t
// copying 16 bytes (8 values) per pass p of PASSES:
//     row r = p * ROWS_PER_PASS + t / VEC_PER_ROW,  vector v = t % VEC_PER_ROW
//     global  a[pair(r) / top_k][k_base + 8 v .. +7]  ->  LDS a_lds[buf][r][v]
//   (VEC_PER_ROW = KC / 8, ROWS_PER_PASS = W * 32 / VEC_PER_ROW; invalid rows
//   store zeros; a row has VEC_PER_ROW + 1 vectors: the pad vector keeps the
//   16 rows a fragment load touches on different banks). Chunk c + 1 is loaded
//   to registers before chunk c is computed and stored to the other buffer
//   after it; one __syncthreads per chunk.
//   A fragment (mt, K step at chunk offset kk, a multiple of 16):
//     lane l reads a_lds[buf][16 * mt + lo][kk / 8] and [..][kk / 8 + 1]
//     (slots 0..7 and 8..15).
//
// Weights (B): K word row kw = k / 8 of column n holds, after gptq_shuffle,
//   K = 8 kw + {0..7} in the nibbles {0, 4, 1, 5, 2, 6, 3, 7}
//   (nibble p < 4 = K 2p, nibble 4 + p = K 2p + 1). A K step of 16 at k
//   uses the words of rows k / 8 (slots 0..7) and k / 8 + 1 (slots 8..15) of
//   column n = F(w, j) + lo; each lane dequantizes its own two words, so both
//   lane halves of a wave repeat the same work (the doubled layout).
//   Dequant (as in q_gemm_rdna3_wmma.cu): w = scale * (nibble - zero) with
//   (nibble - zero) exact and ONE rounding to T (bf16: one fp32 fma, one
//   round to nearest even).
//   Scales and zeros of group g = k / group_size are read from
//   [g * N + n] and [g * N/8 + n / 8] (nibble n % 8); the next group is
//   loaded one group ahead.
//
// Output (D -> c): lane l of fragment (mt, j), slot i holds row
//   r = 16 * mt + 2 i + hi, column n = F(w, j) + lo
//   pair = sorted_token_ids[tb * BM + r]; value = acc[i] * topk_weights[pair]
//   (if mul_topk_weight); direct store c[pair * N + n], or Q32 row
//   (output_topk > 0 ? pair / output_topk : pair).
//
// Numerics: per output element the K sum is one fp32 chain of K / 16 WMMA
// accumulations in ascending k, identical for every BM / tile shape, so the
// direct-store result is bitwise independent of the schedule. The Q32 result
// differs from the scalar kernel's only through the fp32 value of one pair
// (the scalar kernel sums K slices of 256 in a different order and applies the
// exact factored (y, z) form, this one rounds each dequantized weight once);
// the Q32 integer sums themselves are exact. The Q32 limit is
// 2^30 / max(1, output_topk): one output element receives exactly one partial
// per reduced pair here (the scalar kernel splits K over gridDim.z and divides
// by that too). A pair value of magnitude >= 2^30 / output_topk is far outside
// fp16, so the final fp16 cast turns it into NaN either way.

#include <cstdint>
#include <type_traits>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace moe_gptq_rdna3_wmma {

using gptq_rdna3::bf162_t;
using gptq_rdna3::bf16_t;

// Tile shape per block_size_m. WAVES waves, NF 16-column fragments per wave,
// KC activation columns per LDS chunk. Every wave multiplies all BM rows
// (MT A fragments) by its own columns, so each dequantized B fragment is
// produced once per block and used MT times.
template <int BLOCK_M>
struct Tile;
template <>
struct Tile<16> {
  static constexpr int WAVES = 4, NF = 4, KC = 128;
};
template <>
struct Tile<32> {
  static constexpr int WAVES = 8, NF = 2, KC = 128;
};
template <>
struct Tile<64> {
  static constexpr int WAVES = 8, NF = 2, KC = 64;
};

template <int BLOCK_M>
struct Shape {
  static constexpr int WAVES = Tile<BLOCK_M>::WAVES;
  static constexpr int NF = Tile<BLOCK_M>::NF;
  static constexpr int KC = Tile<BLOCK_M>::KC;
  static constexpr int THREADS = WAVES * 32;
  static constexpr int MT = BLOCK_M / 16;
  static constexpr int TILE_N = WAVES * NF * 16;
  static constexpr int SUBS = KC / 32;
  static constexpr int VEC_PER_ROW = KC / 8;
  static constexpr int ROWS_PER_PASS = THREADS / VEC_PER_ROW;
  static constexpr int PASSES = BLOCK_M / ROWS_PER_PASS;
  static constexpr int LDS_ROW_VEC = VEC_PER_ROW + 1;
  static constexpr int LDS_BYTES = 2 * BLOCK_M * LDS_ROW_VEC * 16;

  static_assert(BLOCK_M % 16 == 0, "WMMA M tile is 16");
  static_assert(KC % 64 == 0, "the K loop handles sub steps in pairs");
  static_assert(THREADS % VEC_PER_ROW == 0 && BLOCK_M % ROWS_PER_PASS == 0,
                "activation staging must cover the tile in whole passes");
  static_assert(BLOCK_M <= THREADS,
                "row metadata is staged one row per thread");
  static_assert(LDS_BYTES <= 32 * 1024, "LDS budget per block");
};

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// Native AMDGPU vector types expected by the WMMA built-ins.
using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));
using v4u32 = uint32_t __attribute__((ext_vector_type(4)));
using v8u32 = uint32_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// HIP wrapper types -> native compiler types of the WMMA built-ins.
template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using v16 = v16bf16;
};

// --- Dequant helpers: verbatim from q_gemm_rdna3_wmma.cu (see the numerics
// notes there) so that this kernel dequantizes bit for bit like the dense
// WMMA kernel. Copied, not shared, to leave qdq_4_rdna3.cuh and the scalar
// kernels' code generation alone. ---

// (nibble - zero) is formed as an exact integer in fp16 (1024 + nibble minus
// 1024 + zero) and then multiplied by the scale: one rounding.
__forceinline__ __device__ void prep_zero_scale_fp16_precise(uint32_t zero,
                                                             half scale,
                                                             half2& z_prep,
                                                             half2& y_prep) {
  union {
    uint16_t u;
    half h;
  } zu;
  zu.u = (uint16_t)(0x6400 | zero);
  z_prep = __half2half2(zu.h);
  y_prep = __half2half2(scale);
}

__forceinline__ __device__ void dequant_4bit_8_fp16_precise(uint32_t qa,
                                                            half2 (&dq)[4],
                                                            half2 z_prep,
                                                            half2 y_prep) {
  const uint32_t c0 = 0x64006400;
  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = ((qa >> 0) & 0x000F000F) | c0;
  q1.u = ((qa >> 4) & 0x000F000F) | c0;
  q2.u = ((qa >> 8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;
  dq[0] = __hmul2(__hsub2(q0.h2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.h2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.h2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.h2, z_prep), y_prep);
}

// bf16: z_prep = -(128 + zero) * scale, y_prep = scale; the fma
// q * y_prep + z_prep is exact (<= 12 significant bits), narrowed once.
__forceinline__ __device__ void prep_zero_scale_bf16_f32(uint32_t zero,
                                                         bf16_t scale,
                                                         float& z_prep,
                                                         float& y_prep) {
  float scale_f = __bfloat162float(scale);
  z_prep = -(128.0f + (float)zero) * scale_f;
  y_prep = scale_f;
}

// fp32 -> bf16 narrow without the NaN canonicalisation __float2bfloat16
// emits: dequant outputs are bounded products and never NaN/Inf.
__forceinline__ __device__ bf16_t f32_to_bf16_no_canon(float f) {
  uint32_t fu = __float_as_uint(f);
  uint32_t lsb = (fu >> 16) & 1u;
  uint16_t out_u = (uint16_t)((fu + 0x7FFFu + lsb) >> 16);
  bf16_t out;
  __builtin_memcpy(&out, &out_u, sizeof(out));
  return out;
}

__forceinline__ __device__ void dequant_4bit_8_bf16_to_bf16(uint32_t qa,
                                                            bf162_t (&dq)[4],
                                                            float z_prep,
                                                            float y_prep) {
  const uint32_t c0 = 0x43004300;
  const uint32_t q0 = ((qa >> 0) & 0x000F000F) | c0;
  const uint32_t q1 = ((qa >> 4) & 0x000F000F) | c0;
  const uint32_t q2 = ((qa >> 8) & 0x000F000F) | c0;
  const uint32_t q3 = ((qa >> 12) & 0x000F000F) | c0;
  const float q0x = __uint_as_float((q0 & 0xFFFFu) << 16);
  const float q0y = __uint_as_float(q0 & 0xFFFF0000u);
  const float q1x = __uint_as_float((q1 & 0xFFFFu) << 16);
  const float q1y = __uint_as_float(q1 & 0xFFFF0000u);
  const float q2x = __uint_as_float((q2 & 0xFFFFu) << 16);
  const float q2y = __uint_as_float(q2 & 0xFFFF0000u);
  const float q3x = __uint_as_float((q3 & 0xFFFFu) << 16);
  const float q3y = __uint_as_float(q3 & 0xFFFF0000u);
  dq[0].x = f32_to_bf16_no_canon(__fmaf_rn(q0x, y_prep, z_prep));
  dq[0].y = f32_to_bf16_no_canon(__fmaf_rn(q0y, y_prep, z_prep));
  dq[1].x = f32_to_bf16_no_canon(__fmaf_rn(q1x, y_prep, z_prep));
  dq[1].y = f32_to_bf16_no_canon(__fmaf_rn(q1y, y_prep, z_prep));
  dq[2].x = f32_to_bf16_no_canon(__fmaf_rn(q2x, y_prep, z_prep));
  dq[2].y = f32_to_bf16_no_canon(__fmaf_rn(q2y, y_prep, z_prep));
  dq[3].x = f32_to_bf16_no_canon(__fmaf_rn(q3x, y_prep, z_prep));
  dq[3].y = f32_to_bf16_no_canon(__fmaf_rn(q3y, y_prep, z_prep));
}

template <typename To, typename From>
__device__ __forceinline__ To bit_cast_to(const From& x) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast_to needs equal sizes");
  To r;
  __builtin_memcpy(&r, &x, sizeof(To));
  return r;
}

// Per-column dequant constants of one group and the dequantization of one
// WMMA B fragment (16 K values of one column = two packed words).
template <typename T>
struct Dequant;

template <>
struct Dequant<half> {
  struct Prep {
    half2 z;
    half2 y;
  };
  static __device__ __forceinline__ Prep prep(uint32_t zero, half scale) {
    Prep p;
    prep_zero_scale_fp16_precise(zero, scale, p.z, p.y);
    return p;
  }
  // w0 = K slots 0..7, w1 = K slots 8..15 of the fragment.
  static __device__ __forceinline__ v16fp16 frag(uint32_t w0, uint32_t w1,
                                                 const Prep& p) {
    half2 d0[4], d1[4];
    dequant_4bit_8_fp16_precise(w0, d0, p.z, p.y);
    dequant_4bit_8_fp16_precise(w1, d1, p.z, p.y);
    const v8u32 t = {
        bit_cast_to<uint32_t>(d0[0]), bit_cast_to<uint32_t>(d0[1]),
        bit_cast_to<uint32_t>(d0[2]), bit_cast_to<uint32_t>(d0[3]),
        bit_cast_to<uint32_t>(d1[0]), bit_cast_to<uint32_t>(d1[1]),
        bit_cast_to<uint32_t>(d1[2]), bit_cast_to<uint32_t>(d1[3])};
    return __builtin_bit_cast(v16fp16, t);
  }
};

template <>
struct Dequant<bf16_t> {
  struct Prep {
    float z;
    float y;
  };
  static __device__ __forceinline__ Prep prep(uint32_t zero, bf16_t scale) {
    Prep p;
    prep_zero_scale_bf16_f32(zero, scale, p.z, p.y);
    return p;
  }
  static __device__ __forceinline__ v16bf16 frag(uint32_t w0, uint32_t w1,
                                                 const Prep& p) {
    bf162_t d0[4], d1[4];
    dequant_4bit_8_bf16_to_bf16(w0, d0, p.z, p.y);
    dequant_4bit_8_bf16_to_bf16(w1, d1, p.z, p.y);
    const v8u32 t = {
        bit_cast_to<uint32_t>(d0[0]), bit_cast_to<uint32_t>(d0[1]),
        bit_cast_to<uint32_t>(d0[2]), bit_cast_to<uint32_t>(d0[3]),
        bit_cast_to<uint32_t>(d1[0]), bit_cast_to<uint32_t>(d1[1]),
        bit_cast_to<uint32_t>(d1[2]), bit_cast_to<uint32_t>(d1[3])};
    return __builtin_bit_cast(v16bf16, t);
  }
};

// Q32 accumulation of one value: a partial that is non-finite or reaches
// ``fixed_limit`` in magnitude sets the sticky row status instead of being
// added (the per-element rule of fixed_accum_row in moe_q_gemm_rdna3.cu).
__forceinline__ __device__ void fixed_accum(float value, long long* out,
                                            long long* row_status,
                                            float fixed_limit) {
  if (!isfinite(value) || fabsf(value) >= fixed_limit) {
    atomicOr(reinterpret_cast<unsigned long long*>(row_status), 1ull);
    return;
  }
  const long long fixed = __float2ll_rn(value * 4294967296.f);
  atomicAdd(reinterpret_cast<unsigned long long*>(out),
            static_cast<unsigned long long>(fixed));
}

// ---------------------------------------------------------------------------
// Grouped GEMM kernel.
// ---------------------------------------------------------------------------

template <typename T, typename C_T, int BLOCK_M>
__global__ void __launch_bounds__(Shape<BLOCK_M>::THREADS)
    moe_gemm_q4_wmma_kernel_rdna3(
        const T* __restrict__ a,                  // [size_m, size_k]
        C_T* __restrict__ c,                      // [pairs, N] or Q32 int64
        const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] packed
        const T* __restrict__ b_scales,           // [E, groups, N]
        const uint32_t* __restrict__ b_qzeros,    // [E, groups, N/8] packed
        const float* __restrict__ topk_weights,   // [pairs] or nullptr
        const int32_t* __restrict__ sorted_token_ids,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
        const int size_n, const int size_k, const int groups, const int top_k,
        const int num_experts, const int expert_weight_stride,
        const int expert_scales_stride, const int expert_zeros_stride,
        const int output_topk) {
  using S = Shape<BLOCK_M>;
  constexpr int NF = S::NF;
  constexpr int MT = S::MT;
  constexpr int KC = S::KC;
  constexpr int SUBS = S::SUBS;
  constexpr int VEC_PER_ROW = S::VEC_PER_ROW;
  constexpr int ROWS_PER_PASS = S::ROWS_PER_PASS;
  constexpr int PASSES = S::PASSES;
  constexpr bool Q32 = std::is_same<C_T, long long>::value;
  static_assert(!Q32 || std::is_same<T, half>::value,
                "Q32 fixed-point accumulation is fp16-only");
  static_assert(Q32 || std::is_same<C_T, T>::value,
                "the direct store writes the activation dtype");
  using V16 = typename WmmaNative<T>::v16;
  using DQ = Dequant<T>;

  // Whole-block early exits, taken before any barrier.
  const int token_block = blockIdx.x;
  if (token_block * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int row_base = token_block * BLOCK_M;
  const int tile_n0 = blockIdx.y * S::TILE_N;

  __shared__ v4u32 a_lds[2][BLOCK_M][S::LDS_ROW_VEC];
  __shared__ int s_out_row[BLOCK_M];  // output row of each tile row, -1: none
  __shared__ float s_weight[BLOCK_M];

  // Epilogue metadata of tile row `tid`; made visible by the barrier below.
  if (tid < BLOCK_M) {
    const int pair = sorted_token_ids[row_base + tid];
    const bool valid = pair >= 0 && pair / top_k < size_m;
    s_out_row[tid] = valid ? (output_topk > 0 ? pair / output_topk : pair) : -1;
    s_weight[tid] =
        (valid && topk_weights != nullptr) ? topk_weights[pair] : 0.0f;
  }

  // Activation staging: this thread's rows and column vector (see mapping).
  const int stage_vec = tid % VEC_PER_ROW;
  const int stage_row = tid / VEC_PER_ROW;
  const v4u32* a_src[PASSES];
  #pragma unroll
  for (int p = 0; p < PASSES; ++p) {
    const int pair = sorted_token_ids[row_base + stage_row + p * ROWS_PER_PASS];
    const int token_row = pair / top_k;
    a_src[p] =
        (pair >= 0 && token_row < size_m)
            ? reinterpret_cast<const v4u32*>(a + (int64_t)token_row * size_k) +
                  stage_vec
            : nullptr;
  }
  const v4u32 zero_vec = {0u, 0u, 0u, 0u};
  auto load_a_chunk = [&](v4u32(&dst)[PASSES], int k_base) {
    const int k_rem = size_k - k_base;
  #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      dst[p] = (a_src[p] != nullptr && stage_vec * 8 < k_rem)
                   ? a_src[p][k_base >> 3]
                   : zero_vec;
    }
  };
  auto store_a_chunk = [&](int buf, const v4u32(&src)[PASSES]) {
  #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      a_lds[buf][stage_row + p * ROWS_PER_PASS][stage_vec] = src[p];
    }
  };

  // Weight columns of this wave's fragments. A fragment at or past N reads
  // the last 16 columns instead and never stores. Every lane address below is
  // a wave-uniform 64-bit base plus a zero-extended 32-bit byte offset; N <=
  // 2^26 (host check) keeps the offsets, up to 16 N bytes, inside 32 bits.
  const char* b_expert = reinterpret_cast<const char*>(
      b_q_weight + (int64_t)expert_id * expert_weight_stride);
  const char* sc_expert = reinterpret_cast<const char*>(
      b_scales + (int64_t)expert_id * expert_scales_stride);
  const char* qz_expert = reinterpret_cast<const char*>(
      b_qzeros + (int64_t)expert_id * expert_zeros_stride);
  const int64_t b_row_bytes = (int64_t)size_n * 4;
  const uint32_t b_row_bytes32 = (uint32_t)size_n * 4u;
  int frag_n0[NF];
  int n_load[NF];
  uint32_t b_col_byte[NF];
  uint32_t zero_col_byte[NF];
  uint32_t scale_col_byte[NF];
  #pragma unroll
  for (int j = 0; j < NF; ++j) {
    frag_n0[j] = tile_n0 + (wave * NF + j) * 16;
    n_load[j] = min(frag_n0[j], size_n - 16) + lane_lo;
    b_col_byte[j] = (uint32_t)n_load[j] * 4u;
    zero_col_byte[j] = (uint32_t)(n_load[j] >> 3) * 4u;
    scale_col_byte[j] = (uint32_t)n_load[j] * (uint32_t)sizeof(T);
  }

  // Packed words of the 32 K starting at k: four K-word rows per fragment.
  auto load_words = [&](uint32_t (&dst)[NF][4], int k) {
    const char* row = b_expert + (int64_t)(k >> 3) * b_row_bytes;
  #pragma unroll
    for (int j = 0; j < NF; ++j) {
  #pragma unroll
      for (int r = 0; r < 4; ++r) {
        dst[j][r] = *reinterpret_cast<const uint32_t*>(
            row + (b_col_byte[j] + (uint32_t)r * b_row_bytes32));
      }
    }
  };

  // Per-column constants of the current group; the raw zero word and scale of
  // the NEXT group are always in flight.
  const int groupsize = size_k / groups;
  typename DQ::Prep prep[NF];
  uint32_t zero_raw[NF];
  T scale_raw[NF];
  auto load_group_raw = [&](int g) {
    g = min(g, groups - 1);
    const char* zero_row = qz_expert + (int64_t)g * (size_n >> 3) * 4;
    const char* scale_row =
        sc_expert + (int64_t)g * size_n * (int64_t)sizeof(T);
  #pragma unroll
    for (int j = 0; j < NF; ++j) {
      zero_raw[j] =
          *reinterpret_cast<const uint32_t*>(zero_row + zero_col_byte[j]);
      scale_raw[j] = *reinterpret_cast<const T*>(scale_row + scale_col_byte[j]);
    }
  };
  auto make_prep = [&]() {
  #pragma unroll
    for (int j = 0; j < NF; ++j) {
      // GPTQ v1: effective zero = stored zero + 1.
      const uint32_t zero = ((zero_raw[j] >> ((n_load[j] & 7) * 4)) & 0xF) + 1;
      prep[j] = DQ::prep(zero, scale_raw[j]);
    }
  };

  v8fp32 acc[MT][NF];
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
  #pragma unroll
    for (int j = 0; j < NF; ++j) {
      acc[mt][j] = v8fp32{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  int group = 0;
  int next_group_k = groupsize;
  load_group_raw(0);
  make_prep();
  load_group_raw(1);

  // One sub step = 32 K = two WMMA K steps. `cur` holds the packed words of
  // this sub step, the words of the next one are loaded into `nxt` first.
  auto sub_step = [&](int k, uint32_t (&cur)[NF][4], uint32_t (&nxt)[NF][4],
                      int kvec0, int buf) {
    if (k >= size_k) return;  // tail chunk: only whole 32-K steps exist
    if (k == next_group_k) {
      ++group;
      next_group_k += groupsize;
      make_prep();
      load_group_raw(group + 1);
    }
    load_words(nxt, min(k + 32, size_k - 32));
  #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      V16 a_frag[MT];
  #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const v4u32 lo = a_lds[buf][16 * mt + lane_lo][kvec0 + 2 * ks];
        const v4u32 hi = a_lds[buf][16 * mt + lane_lo][kvec0 + 2 * ks + 1];
        a_frag[mt] = __builtin_bit_cast(
            V16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
      }
  #pragma unroll
      for (int j = 0; j < NF; ++j) {
        const V16 b_frag =
            DQ::frag(cur[j][2 * ks], cur[j][2 * ks + 1], prep[j]);
  #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          acc[mt][j] = wmma_mma(a_frag[mt], b_frag, acc[mt][j]);
        }
      }
    }
  };

  // Prologue: chunk 0 of A and the first words of B.
  v4u32 a_pref[PASSES];
  uint32_t words_a[NF][4];
  uint32_t words_b[NF][4];
  load_a_chunk(a_pref, 0);
  load_words(words_a, 0);
  store_a_chunk(0, a_pref);
  __syncthreads();

  int buf = 0;
  for (int k_base = 0; k_base < size_k; k_base += KC) {
    const bool has_next = k_base + KC < size_k;
    if (has_next) load_a_chunk(a_pref, k_base + KC);
  #pragma unroll
    for (int pr = 0; pr < SUBS / 2; ++pr) {
      sub_step(k_base + 64 * pr, words_a, words_b, 8 * pr, buf);
      sub_step(k_base + 64 * pr + 32, words_b, words_a, 8 * pr + 4, buf);
    }
    if (has_next) store_a_chunk(buf ^ 1, a_pref);
    __syncthreads();
    buf ^= 1;
  }

  // Epilogue: scatter the D fragments (see mapping).
  [[maybe_unused]] const float fixed_limit =
      1073741824.f / (float)max(1, output_topk);
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 16 * mt + 2 * i + lane_hi;
      const int out_row = s_out_row[r];
      if (out_row < 0) continue;
      const float weight = topk_weights != nullptr ? s_weight[r] : 1.0f;
  #pragma unroll
      for (int j = 0; j < NF; ++j) {
        if (frag_n0[j] >= size_n) continue;
        const int n = frag_n0[j] + lane_lo;
        const float value = acc[mt][j][i] * weight;
        if constexpr (Q32) {
          long long* row = c + (int64_t)out_row * (size_n + 1);
          fixed_accum(value, row + n, row + size_n, fixed_limit);
        } else if constexpr (std::is_same<T, half>::value) {
          c[(int64_t)out_row * size_n + n] = __float2half_rn(value);
        } else {
          c[(int64_t)out_row * size_n + n] = __float2bfloat16(value);
        }
      }
    }
  }
}

#else  // non-RDNA3 device pass: empty stub for symbol parity

// Keep the template and parameter lists identical, in the same order, to the
// RDNA3 kernel above: the mangled name of the stub is what the gfx1030 device
// image exports for the host-side launch, so a parameter added, removed or
// reordered in the real kernel must be mirrored here.
template <typename T, typename C_T, int BLOCK_M>
__global__ void moe_gemm_q4_wmma_kernel_rdna3(
    const T*, C_T*, const uint32_t*, const T*, const uint32_t*, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const int, const int, const int,
    const int) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, typename C_T, int BLOCK_M>
void launch_moe_gemm_q4_wmma(
    const T* a, C_T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int num_experts,
    int expert_weight_stride, int expert_scales_stride, int expert_zeros_stride,
    int output_topk, cudaStream_t stream) {
  using S = Shape<BLOCK_M>;
  dim3 block(S::THREADS);
  dim3 grid(num_token_blocks, (size_n + S::TILE_N - 1) / S::TILE_N, 1);
  moe_gemm_q4_wmma_kernel_rdna3<T, C_T, BLOCK_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      num_experts, expert_weight_stride, expert_scales_stride,
      expert_zeros_stride, output_topk);
}

template <typename T, typename C_T>
void dispatch_moe_gemm_q4_wmma(
    const T* a, C_T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int block_size_m,
    int num_experts, int expert_weight_stride, int expert_scales_stride,
    int expert_zeros_stride, int output_topk, cudaStream_t stream) {
  switch (block_size_m) {
    case 16:
      launch_moe_gemm_q4_wmma<T, C_T, 16>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, num_experts, expert_weight_stride,
          expert_scales_stride, expert_zeros_stride, output_topk, stream);
      break;
    case 32:
      launch_moe_gemm_q4_wmma<T, C_T, 32>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, num_experts, expert_weight_stride,
          expert_scales_stride, expert_zeros_stride, output_topk, stream);
      break;
    case 64:
      launch_moe_gemm_q4_wmma<T, C_T, 64>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, num_experts, expert_weight_stride,
          expert_scales_stride, expert_zeros_stride, output_topk, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_gptq_gemm_rdna3_wmma: block_size_m must be 16, 32, or "
                  "64, got ",
                  block_size_m);
  }
}

}  // namespace moe_gptq_rdna3_wmma
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
//
// Inputs:
//   a                      [M, K]                   half or bfloat16, rows
//                          16-byte aligned (K % 8 == 0)
//   c                      output_topk == 0, direct store: [M*top_k, N], the
//                          dtype of a; every scheduled pair row is written,
//                          other rows are not touched (no zeroing needed).
//                          Q32 (fp16 only): a zeroed int64 [rows, N+1], N
//                          fixed-point columns and a sticky row status,
//                          rows = M*top_k, or ceil(M*top_k / output_topk) when
//                          output_topk > 0. output_topk > 0 requires Q32.
//   b_q_weight             [E, K/8, N]              uint32 (shuffled)
//   b_scales               [E, groups, N]           same dtype as a
//   b_qzeros               [E, groups, N/8]         uint32 (packed 4-bit)
//   topk_weights           [>= M*top_k] or empty    float32; used when
//                          mul_topk_weight
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [>= num_blocks]          int32 (-1: skip the block)
//   num_tokens_post_padded [1]                      int32
//   top_k                  int
//   block_size_m           int (16, 32, or 64)
//   mul_topk_weight        bool
//   output_topk            int (0: none)

void moe_gptq_gemm_rdna3_wmma(
    torch::Tensor a, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor b_qzeros, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_topk_weight, int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda() && b_qzeros.is_cuda() &&
                  sorted_token_ids.is_cuda() && expert_ids.is_cuda() &&
                  num_tokens_post_padded.is_cuda(),
              "moe_gptq_gemm_rdna3_wmma: every tensor must be a CUDA/HIP "
              "tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3, "b_qzeros must be 3D [E, groups, N/8]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kInt &&
                  b_qzeros.scalar_type() == torch::kInt,
              "b_q_weight and b_qzeros must be int32");
  TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt &&
                  expert_ids.scalar_type() == torch::kInt &&
                  num_tokens_post_padded.scalar_type() == torch::kInt,
              "sorted_token_ids, expert_ids and num_tokens_post_padded must "
              "be int32");
  TORCH_CHECK(top_k >= 1, "moe_gptq_gemm_rdna3_wmma: top_k must be >= 1, got ",
              top_k);
  TORCH_CHECK(output_topk >= 0,
              "moe_gptq_gemm_rdna3_wmma: output_topk must be >= 0, got ",
              output_topk);
  TORCH_CHECK(block_size_m == 16 || block_size_m == 32 || block_size_m == 64,
              "moe_gptq_gemm_rdna3_wmma: block_size_m must be 16, 32, or 64, "
              "got ",
              block_size_m);
  TORCH_CHECK(
      a.device() == b_q_weight.device() && a.device() == b_scales.device() &&
          a.device() == b_qzeros.device() &&
          a.device() == sorted_token_ids.device() &&
          a.device() == expert_ids.device() &&
          a.device() == num_tokens_post_padded.device(),
      "moe_gptq_gemm_rdna3_wmma: every tensor must be on the device of a");
  TORCH_CHECK(a.is_contiguous() && b_q_weight.is_contiguous() &&
                  b_scales.is_contiguous() && b_qzeros.is_contiguous() &&
                  sorted_token_ids.is_contiguous() &&
                  expert_ids.is_contiguous() &&
                  num_tokens_post_padded.is_contiguous(),
              "moe_gptq_gemm_rdna3_wmma: tensors must be contiguous");

  const bool fixed_point = c.scalar_type() == torch::kLong;
  if (fixed_point) {
    TORCH_CHECK(a.scalar_type() == torch::kHalf,
                "moe_gptq_gemm_rdna3_wmma: Q32 fixed-point accumulation (int64 "
                "c) supports fp16 activations only, got ",
                a.scalar_type());
  } else {
    TORCH_CHECK(output_topk == 0,
                "moe_gptq_gemm_rdna3_wmma: output_topk > 0 reduces through "
                "the int64 Q32 accumulator; c must be int64, got ",
                c.scalar_type());
    TORCH_CHECK(c.scalar_type() == a.scalar_type(),
                "moe_gptq_gemm_rdna3_wmma: c must have the dtype of a (or be "
                "int64 for Q32 accumulation), got ",
                c.scalar_type());
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t size_m = a.size(0);
  const int64_t size_k = a.size(1);
  const int64_t size_n = b_q_weight.size(2);
  const int64_t groups = b_scales.size(1);
  const int64_t num_experts = b_q_weight.size(0);
  TORCH_CHECK(size_n > 0 && size_n % 16 == 0 && size_n <= (1 << 26),
              "moe_gptq_gemm_rdna3_wmma: N must be a positive multiple of 16 "
              "(WMMA tile) and at most 2^26 (32-bit byte offsets), got ",
              size_n);
  TORCH_CHECK(size_k > 0 && size_k % 32 == 0,
              "moe_gptq_gemm_rdna3_wmma: K must be a positive multiple of 32, "
              "got ",
              size_k);
  TORCH_CHECK(groups > 0 && size_k % (groups * 32) == 0,
              "group size (K/groups = ", size_k / std::max<int64_t>(groups, 1),
              ") must be a multiple of 32: the kernel checks group "
              "transitions at 32-K granularity");
  TORCH_CHECK(b_q_weight.size(1) * 8 == size_k,
              "b_q_weight must be [E, K/8, N] with K = a.size(1) = ", size_k,
              ", got ", b_q_weight.sizes());
  TORCH_CHECK(b_scales.size(0) == num_experts && b_scales.size(2) == size_n,
              "b_scales must be [E, groups, N] = [", num_experts, ", ", groups,
              ", ", size_n, "], got ", b_scales.sizes());
  TORCH_CHECK(b_qzeros.size(0) == num_experts && b_qzeros.size(1) == groups &&
                  b_qzeros.size(2) * 8 == size_n,
              "b_qzeros must be [E, groups, N/8] = [", num_experts, ", ",
              groups, ", ", size_n / 8, "], got ", b_qzeros.sizes());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "moe_gptq_gemm_rdna3_wmma: a must be 16-byte aligned");
  const int64_t pairs = size_m * top_k;
  TORCH_CHECK(pairs <= INT32_MAX && (size_k / 8) * size_n <= INT32_MAX &&
                  num_experts <= INT32_MAX,
              "moe_gptq_gemm_rdna3_wmma: problem too large for 32-bit "
              "indexing");
  TORCH_CHECK(sorted_token_ids.numel() % block_size_m == 0,
              "sorted_token_ids must hold whole token blocks of block_size_m "
              "= ",
              block_size_m, ", got ", sorted_token_ids.numel());

  // Per-expert strides
  const int expert_weight_stride =
      (int)(b_q_weight.size(1) * b_q_weight.size(2));
  const int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const int expert_zeros_stride = (int)(b_qzeros.size(1) * b_qzeros.size(2));

  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  // Nothing is routed: skip the launch instead of requesting an empty grid.
  if (num_token_blocks == 0) return;
  TORCH_CHECK(expert_ids.numel() >= num_token_blocks &&
                  num_tokens_post_padded.numel() >= 1,
              "expert_ids must hold one entry per token block (",
              num_token_blocks, ") and num_tokens_post_padded one element");

  const float* topk_w_ptr = nullptr;
  if (mul_topk_weight) {
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.is_contiguous() &&
                    topk_weights.scalar_type() == torch::kFloat &&
                    topk_weights.device() == a.device() &&
                    topk_weights.numel() >= pairs,
                "moe_gptq_gemm_rdna3_wmma: mul_topk_weight needs a contiguous "
                "float32 topk_weights of >= M*top_k = ",
                pairs, " elements on the device of a, got ",
                topk_weights.sizes(), " ", topk_weights.scalar_type());
    topk_w_ptr = topk_weights.data_ptr<float>();
  }

  using vllm::gptq_rdna3::bf16_t;

  auto dispatch = [&](auto* a_ptr, auto* c_ptr, const auto* s_ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(a_ptr)>>;
    using C_T = std::remove_pointer_t<decltype(c_ptr)>;
    vllm::moe_gptq_rdna3_wmma::dispatch_moe_gemm_q4_wmma<T, C_T>(
        a_ptr, c_ptr, (const uint32_t*)b_q_weight.data_ptr<int32_t>(), s_ptr,
        (const uint32_t*)b_qzeros.data_ptr<int32_t>(), topk_w_ptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks,
        (int)size_m, (int)size_n, (int)size_k, (int)groups, (int)top_k,
        (int)block_size_m, (int)num_experts, expert_weight_stride,
        expert_scales_stride, expert_zeros_stride, (int)output_topk, stream);
  };

  if (fixed_point) {
    const int64_t rows =
        output_topk > 0 ? (pairs + output_topk - 1) / output_topk : pairs;
    TORCH_CHECK(c.device() == a.device() && c.is_contiguous() &&
                    c.size(0) == rows && c.size(1) == size_n + 1,
                "moe_gptq_gemm_rdna3_wmma: Q32 c must be a contiguous int64 [",
                rows, ", ", size_n + 1, "] tensor on the device of a, got ",
                c.sizes());
    dispatch((const half*)a.data_ptr(),
             reinterpret_cast<long long*>(c.data_ptr<int64_t>()),
             (const half*)b_scales.data_ptr());
  } else {
    TORCH_CHECK(c.device() == a.device() && c.is_contiguous() &&
                    c.size(0) == pairs && c.size(1) == size_n,
                "moe_gptq_gemm_rdna3_wmma: c must be a contiguous [", pairs,
                ", ", size_n,
                "] tensor of the dtype of a on the device of a, "
                "got ",
                c.sizes());
    if (a.scalar_type() == torch::kHalf) {
      dispatch((const half*)a.data_ptr(), (half*)c.data_ptr(),
               (const half*)b_scales.data_ptr());
    } else {
      dispatch((const bf16_t*)a.data_ptr(), (bf16_t*)c.data_ptr(),
               (const bf16_t*)b_scales.data_ptr());
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
