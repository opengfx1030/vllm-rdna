// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// rdna_hint_flex (HadamardINT2-FP16 spec v3.2) weight kernels for RDNA2
// (gfx1030), F16 compute mode (spec 6.1), fp16 only.
//
// Storage (one tensor W[R, K], spec 2.4; all buffers flat, per-row ragged):
//   codes      int32 [sum_r K*nbits_r/32]  codes LSB-first, word aligned
//                                          (16 INT2, 8 INT4 or 4 INT8 per word)
//   code_off   int32 [R+1]   first word of each row
//   blk_params uint8 [2*sum_r K/B_r]  (lo_u8, step_u8) per block
//   param_off  int32 [R+1]   first block of each row
//   row_consts fp16  [R, 3]  lo_min, lo_scale, step_scale
//   row_nbits  uint8 [R]     2, 4 or 8
//   row_block  uint8 [R]     16, 32 or 64
//   had        uint8 [K/16]  order-16 Walsh-Hadamard flag per column group
//   col_scale  fp16  [K]     optional smoothing vector s (numel 0 = none)
//
// Decode (spec 2.2):
//   lo   = fp16(fma_f32(lo_u8, lo_scale, lo_min))
//   step = fp16(step_u8 * step_scale)
//   w    = fp16(lo + code * step)       one rounding: v_fma_f16
// The stored basis is H(W * s); the runtime activation is x' = H(x / s) on
// the flagged groups (spec 2.3), so y = x' . W_stored^T.
//
// Kernels:
//   act_transform : x' = H16(x / s), fp32 math, one fp16 rounding.
//   gemv          : decode path (small M). One wave32 per output row,
//                       lanes stride over the row's words (coalesced), code ->
//                       fp16 by the 0x6400 magic, v_pk_fma_f16 with the block
//                       lo/step, v_dot2_f32_f16 against x', wave reduction.
//   dequant       : stored-basis fp16 W [R, K]; the prefill path is
//                       dequant + rocBLAS GEMM (correct and simple, not tuned).
//   rdna_hint_flex_linear_rdna2  : composite (transform + gemv or
//   dequant+GEMM), chosen
//                       at execution time from M so torch.compile cannot bake
//                       the warmup shape into the dispatch.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace rdna_hint_flex {

constexpr int kGroup = 16;        // Hadamard group (spec 2.3)
constexpr int kWave = 32;         // RDNA2 wave32
constexpr int kRowsPerBlock = 8;  // 8 waves x 32 lanes = 256 threads
constexpr int kThreads = kRowsPerBlock * kWave;
// Activation quantization group (columns). Declared here, outside the
// RDNA2/host guard below, because the host-side launchers also size grids
// with it: the non-RDNA2 device pass must still see the definition.
constexpr int kQG = 64;

// Row header of one weight row, as cached in LDS by the prefill GEMM.
struct RowHdrS {
  float lo_min, lo_scale, step_scale;
  int32_t code_off, param_off;
  int nb, lb;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ void decode_params(uint32_t lo_u8, uint32_t step_u8,
                                              float lo_min, float lo_scale,
                                              float step_scale, half& lo,
                                              half& step) {
  lo = __float2half_rn(__fmaf_rn(static_cast<float>(lo_u8), lo_scale, lo_min));
  // u8 * fp16 has at most 19 significant bits: exact in fp32, so this is one
  // rounding (to fp16), as in the spec.
  step = __float2half_rn(static_cast<float>(step_u8) * step_scale);
}

// Two codes (< 1024) -> half2 exactly: half(0x6400 | c) = 1024 + c.
__device__ __forceinline__ half2 codes_to_half2(uint32_t c0, uint32_t c1) {
  union {
    uint32_t u;
    half2 h;
  } v;
  v.u = (0x6400u | c0) | ((0x6400u | c1) << 16);
  return __hsub2(v.h, __float2half2_rn(1024.0f));
}

__device__ __forceinline__ half code_to_half(uint32_t c) {
  union {
    uint16_t u;
    half h;
  } v;
  v.u = static_cast<uint16_t>(0x6400u | c);
  return __hsub(v.h, __float2half(1024.0f));
}

__device__ __forceinline__ int log2_block(int b) {
  return b == 16 ? 4 : (b == 32 ? 5 : 6);
}

// ---------------------------------------------------------------------------
// Activation transform: y[m, g*16 + i] = (H16 (x / s))_i on flagged groups.
// One thread per (row, group).
// ---------------------------------------------------------------------------
__global__ void act_transform_kernel(const half* __restrict__ x,
                                     const int64_t ldx,
                                     const half* __restrict__ col_scale,
                                     const uint8_t* __restrict__ had,
                                     half* __restrict__ y, const int M,
                                     const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(M) * groups) return;
  const int m = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const half* xp = x + m * ldx + gi * kGroup;
  float v[kGroup];
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
  if (col_scale != nullptr) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i)
      v[i] = v[i] / __half2float(col_scale[gi * kGroup + i]);
  }
  if (had[gi]) {
    // In-place butterflies give the Sylvester (natural) order, i.e. the
    // symmetric matrix built by [[h, h], [h, -h]] in the reference.
  #pragma unroll
    for (int len = 1; len < kGroup; len <<= 1) {
  #pragma unroll
      for (int i = 0; i < kGroup; ++i) {
        if ((i & len) == 0) {
          const float a = v[i], b = v[i + len];
          v[i] = a + b;
          v[i + len] = a - b;
        }
      }
    }
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;  // 1/sqrt(16)
  }
  half* yp = y + static_cast<int64_t>(m) * K + gi * kGroup;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) yp[i] = __float2half_rn(v[i]);
}

// ---------------------------------------------------------------------------
// GEMV: one wave per output row. NB = code width of the row.
// ---------------------------------------------------------------------------
template <int MT, int NB>
__device__ __forceinline__ void word_dot(
    const uint32_t q, const int w, const half* __restrict__ x, const int K,
    const int m0, const int M, const uint8_t* __restrict__ params, const int lb,
    const float lo_min, const float lo_scale, const float step_scale,
    float (&acc)[MT]) {
  constexpr int CPW = 32 / NB;  // codes per word; a word never crosses a block
  constexpr uint32_t MASK = (1u << NB) - 1u;
  const int k0 = w * CPW;
  const int blk = k0 >> lb;
  half lo, step;
  decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                step_scale, lo, step);
  const half2 lo2 = __half2half2(lo);
  const half2 st2 = __half2half2(step);
  half2 wv[CPW / 2];
  #pragma unroll
  for (int i = 0; i < CPW / 2; ++i) {
    const uint32_t c0 = (q >> (NB * (2 * i))) & MASK;
    const uint32_t c1 = (q >> (NB * (2 * i + 1))) & MASK;
    wv[i] = __hfma2(codes_to_half2(c0, c1), st2, lo2);
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m0 + m < M) {
      const half2* xp = reinterpret_cast<const half2*>(
          x + static_cast<int64_t>(m0 + m) * K + k0);
      float a = acc[m];
  #pragma unroll
      for (int i = 0; i < CPW / 2; ++i)
        a = __builtin_amdgcn_fdot2(wv[i], xp[i], a, /*clamp=*/false);
      acc[m] = a;
    }
  }
}

// Words [w_lo, w_hi) of one row; lanes take consecutive words (coalesced),
// two per step so two weight loads are in flight per lane.
template <int MT, int NB>
__device__ __forceinline__ void row_dot_range(
    const half* __restrict__ x, const int K, const int m0, const int M,
    const uint32_t* __restrict__ words, const uint8_t* __restrict__ params,
    const int lb, const float lo_min, const float lo_scale,
    const float step_scale, const int lane, const int w_lo, const int w_hi,
    float (&acc)[MT]) {
  int w = w_lo + lane;
  for (; w + kWave < w_hi; w += 2 * kWave) {
    const uint32_t qa = words[w];
    const uint32_t qb = words[w + kWave];
    word_dot<MT, NB>(qa, w, x, K, m0, M, params, lb, lo_min, lo_scale,
                     step_scale, acc);
    word_dot<MT, NB>(qb, w + kWave, x, K, m0, M, params, lb, lo_min, lo_scale,
                     step_scale, acc);
  }
  if (w < w_hi)
    word_dot<MT, NB>(words[w], w, x, K, m0, M, params, lb, lo_min, lo_scale,
                     step_scale, acc);
}

template <int MT, int NB>
__device__ __forceinline__ void row_dot(
    const half* __restrict__ x, const int K, const int m0, const int M,
    const uint32_t* __restrict__ words, const uint8_t* __restrict__ params,
    const int lb, const float lo_min, const float lo_scale,
    const float step_scale, const int lane, float (&acc)[MT]) {
  row_dot_range<MT, NB>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                        step_scale, lane, 0, K / (32 / NB), acc);
}

// SPLIT waves share one row, each taking a contiguous share of its words;
// partial sums meet in LDS. Small-R tensors use it to keep enough waves in
// flight; SPLIT = 1 is one wave per row.
template <int MT, int SPLIT>
__global__ void __launch_bounds__(kThreads)
    gemv_kernel(const half* __restrict__ x, const int M, const int K,
                const int32_t* __restrict__ codes,
                const int32_t* __restrict__ code_off,
                const uint8_t* __restrict__ blk_params,
                const int32_t* __restrict__ param_off,
                const half* __restrict__ row_consts,
                const uint8_t* __restrict__ row_nbits,
                const uint8_t* __restrict__ row_block, half* __restrict__ out,
                const int64_t ldo, const int R) {
  constexpr int RPB = kRowsPerBlock / SPLIT;
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * RPB + wave / SPLIT;
  const int part = wave % SPLIT;
  const int m0 = blockIdx.y * MT;
  float acc[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
  if (r < R) {
    const int nb = row_nbits[r];
    const int lb = log2_block(row_block[r]);
    const float lo_min = __half2float(row_consts[3 * r + 0]);
    const float lo_scale = __half2float(row_consts[3 * r + 1]);
    const float step_scale = __half2float(row_consts[3 * r + 2]);
    const uint32_t* words =
        reinterpret_cast<const uint32_t*>(codes) + code_off[r];
    const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[r]);
    const int nw = K * nb / 32;
    const int share = (nw + SPLIT * kWave - 1) / (SPLIT * kWave) * kWave;
    const int w_lo = part * share;
    const int w_hi = min(nw, w_lo + share);
    switch (nb) {
      case 2:
        row_dot_range<MT, 2>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                             step_scale, lane, w_lo, w_hi, acc);
        break;
      case 4:
        row_dot_range<MT, 4>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                             step_scale, lane, w_lo, w_hi, acc);
        break;
      default:
        row_dot_range<MT, 8>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                             step_scale, lane, w_lo, w_hi, acc);
        break;
    }
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
  #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      acc[m] += __shfl_xor(acc[m], off, kWave);
  }
  if constexpr (SPLIT == 1) {
    if (r < R && lane == 0) {
  #pragma unroll
      for (int m = 0; m < MT; ++m)
        if (m0 + m < M)
          out[static_cast<int64_t>(m0 + m) * ldo + r] = __float2half_rn(acc[m]);
    }
  } else {
    __shared__ float red[kRowsPerBlock][MT];
    if (lane == 0) {
  #pragma unroll
      for (int m = 0; m < MT; ++m) red[wave][m] = acc[m];
    }
    __syncthreads();
    if (part == 0 && r < R && lane == 0) {
  #pragma unroll
      for (int m = 0; m < MT; ++m) {
        float v = red[wave][m];
  #pragma unroll
        for (int p = 1; p < SPLIT; ++p) v += red[wave + p][m];
        if (m0 + m < M)
          out[static_cast<int64_t>(m0 + m) * ldo + r] = __float2half_rn(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dequant: W_stored[r, :] in fp16. grid.x = rows, grid.y strides the words.
// ---------------------------------------------------------------------------
// The 16 codes of one 16-column group (one INT2 word, two INT4 words, four
// INT8 words), decoded to fp16 pairs: w = fma(code, step, lo).
template <int NB>
__device__ __forceinline__ void decode16(const uint32_t* __restrict__ w,
                                         const half2 lo2, const half2 st2,
                                         half2 (&wv)[8]) {
  constexpr uint32_t MASK = (1u << NB) - 1u;
  constexpr int CPW = 32 / NB;
  uint32_t q[NB / 2];
  if constexpr (NB == 2) {
    q[0] = w[0];
  } else if constexpr (NB == 4) {
    const uint2 v = *reinterpret_cast<const uint2*>(w);
    q[0] = v.x;
    q[1] = v.y;
  } else {
    const uint4 v = *reinterpret_cast<const uint4*>(w);
    q[0] = v.x;
    q[1] = v.y;
    q[2] = v.z;
    q[3] = v.w;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = 2 * i;
    const uint32_t ca = (q[c / CPW] >> (NB * (c % CPW))) & MASK;
    const uint32_t cb = (q[(c + 1) / CPW] >> (NB * ((c + 1) % CPW))) & MASK;
    wv[i] = __hfma2(codes_to_half2(ca, cb), st2, lo2);
  }
}

// Dequant: W_stored[r, :] in fp16, one 16-column group per thread with two
// 16-byte stores. grid.x = rows.
__global__ void __launch_bounds__(kThreads) dequant_kernel(
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, half* __restrict__ w, const int K) {
  const int r = blockIdx.x;
  const int nb = row_nbits[r];
  const int lb = log2_block(row_block[r]);
  const float lo_min = __half2float(row_consts[3 * r + 0]);
  const float lo_scale = __half2float(row_consts[3 * r + 1]);
  const float step_scale = __half2float(row_consts[3 * r + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[r];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[r]);
  half* wr = w + static_cast<int64_t>(r) * K;
  const int groups = K / kGroup;
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    const int blk = (g * kGroup) >> lb;
    half lo, step;
    decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                  step_scale, lo, step);
    const half2 lo2 = __half2half2(lo), st2 = __half2half2(step);
    union {
      half2 h[8];
      uint4 u[2];
    } o;
    if (nb == 2)
      decode16<2>(words + g, lo2, st2, o.h);
    else if (nb == 4)
      decode16<4>(words + 2 * g, lo2, st2, o.h);
    else
      decode16<8>(words + 4 * g, lo2, st2, o.h);
    uint4* dst = reinterpret_cast<uint4*>(wr + g * kGroup);
    dst[0] = o.u[0];
    dst[1] = o.u[1];
  }
}

// ---------------------------------------------------------------------------
// I8 compute mode (spec 6.2).
//
// Activation: per token and 16-column group g of the transformed activation
//   sx_g = fp16(absmax / 127), xq = clamp(rint(x / sx_g), -127, 127),
//   sumq_g = sum xq. The transform and the quantization group are the same
//   16 columns, so one thread does both.
// Weights: per 16-column chunk of a row, S_g = sum code * xq through four
//   v_dot4_i32_i8; y += sx_g * (step * S_g + lo * sumq_g) in fp32.
// INT8 codes (0..255) do not fit a signed byte: they enter the dot as
//   code - 128 and S_g gets 128 * sumq_g back, exactly, in int32.
// ---------------------------------------------------------------------------
__global__ void act_quant_kernel(const half* __restrict__ x, const int64_t ldx,
                                 const half* __restrict__ col_scale,
                                 const uint8_t* __restrict__ had,
                                 int8_t* __restrict__ xq, half* __restrict__ sx,
                                 int32_t* __restrict__ sumq, const int M,
                                 const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(M) * groups) return;
  const int m = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const half* xp = x + m * ldx + gi * kGroup;
  float v[kGroup];
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
  if (col_scale != nullptr) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i)
      v[i] = v[i] / __half2float(col_scale[gi * kGroup + i]);
  }
  if (had != nullptr && had[gi]) {
  #pragma unroll
    for (int len = 1; len < kGroup; len <<= 1) {
  #pragma unroll
      for (int i = 0; i < kGroup; ++i) {
        if ((i & len) == 0) {
          const float a = v[i], b = v[i + len];
          v[i] = a + b;
          v[i + len] = a - b;
        }
      }
    }
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;
  }
  float amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const half sh = __float2half_rn(amax / 127.0f);
  const float sf = __half2float(sh);
  union {
    int8_t b[kGroup];
    uint4 u;
  } q;
  int sum = 0;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    float r = sf > 0.0f ? rintf(v[i] / sf) : 0.0f;
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    q.b[i] = static_cast<int8_t>(r);
    sum += q.b[i];
  }
  *reinterpret_cast<uint4*>(xq + static_cast<int64_t>(m) * K + gi * kGroup) =
      q.u;
  sx[static_cast<int64_t>(m) * groups + gi] = sh;
  sumq[static_cast<int64_t>(m) * groups + gi] = sum;
}

// Four consecutive codes as the bytes of one dword (code j in byte j).
__device__ __forceinline__ int spread2(uint32_t x) {
  return static_cast<int>((x & 0x03u) | ((x & 0x0Cu) << 6) |
                          ((x & 0x30u) << 12) | ((x & 0xC0u) << 18));
}
__device__ __forceinline__ int spread4(uint32_t x) {
  return static_cast<int>((x & 0x000Fu) | ((x & 0x00F0u) << 4) |
                          ((x & 0x0F00u) << 8) | ((x & 0xF000u) << 12));
}

template <int NB>
__device__ __forceinline__ void group_codes(const uint32_t* __restrict__ w,
                                            int (&d)[4]) {
  if constexpr (NB == 2) {
    const uint32_t q = w[0];
    d[0] = spread2(q);
    d[1] = spread2(q >> 8);
    d[2] = spread2(q >> 16);
    d[3] = spread2(q >> 24);
  } else if constexpr (NB == 4) {
    const uint2 q = *reinterpret_cast<const uint2*>(w);
    d[0] = spread4(q.x);
    d[1] = spread4(q.x >> 16);
    d[2] = spread4(q.y);
    d[3] = spread4(q.y >> 16);
  } else {
    const uint4 q = *reinterpret_cast<const uint4*>(w);
    d[0] = static_cast<int>(q.x ^ 0x80808080u);
    d[1] = static_cast<int>(q.y ^ 0x80808080u);
    d[2] = static_cast<int>(q.z ^ 0x80808080u);
    d[3] = static_cast<int>(q.w ^ 0x80808080u);
  }
}

template <int MT, int NB>
__device__ __forceinline__ void row_dot_i8(
    const int8_t* __restrict__ xq, const half* __restrict__ sx,
    const int32_t* __restrict__ sumq, const int K, const int m0, const int M,
    const uint32_t* __restrict__ words, const uint8_t* __restrict__ params,
    const int lb, const float lo_min, const float lo_scale,
    const float step_scale, const int lane, float (&acc)[MT]) {
  constexpr int WPG = NB / 2;  // words per 16-column group
  const int groups = K / kGroup;
  for (int g = lane; g < groups; g += kWave) {
    int d[4];
    group_codes<NB>(words + g * WPG, d);
    const int blk = (g * kGroup) >> lb;
    half lo, step;
    decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                  step_scale, lo, step);
    const float lof = __half2float(lo), stf = __half2float(step);
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m < M) {
        const int64_t row = static_cast<int64_t>(m0 + m);
        const int4 xv =
            *reinterpret_cast<const int4*>(xq + row * K + g * kGroup);
        const int sq = sumq[row * groups + g];
        int S = __builtin_amdgcn_sdot4(d[0], xv.x, 0, false);
        S = __builtin_amdgcn_sdot4(d[1], xv.y, S, false);
        S = __builtin_amdgcn_sdot4(d[2], xv.z, S, false);
        S = __builtin_amdgcn_sdot4(d[3], xv.w, S, false);
        if constexpr (NB == 8) S += 128 * sq;
        const float sxf = __half2float(sx[row * groups + g]);
        acc[m] +=
            sxf * (stf * static_cast<float>(S) + lof * static_cast<float>(sq));
      }
    }
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads)
    gemv_i8_kernel(const int8_t* __restrict__ xq, const half* __restrict__ sx,
                   const int32_t* __restrict__ sumq, const int M, const int K,
                   const int32_t* __restrict__ codes,
                   const int32_t* __restrict__ code_off,
                   const uint8_t* __restrict__ blk_params,
                   const int32_t* __restrict__ param_off,
                   const half* __restrict__ row_consts,
                   const uint8_t* __restrict__ row_nbits,
                   const uint8_t* __restrict__ row_block,
                   half* __restrict__ out, const int64_t ldo, const int R) {
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * kRowsPerBlock + wave;
  if (r >= R) return;
  const int m0 = blockIdx.y * MT;
  const int nb = row_nbits[r];
  const int lb = log2_block(row_block[r]);
  const float lo_min = __half2float(row_consts[3 * r + 0]);
  const float lo_scale = __half2float(row_consts[3 * r + 1]);
  const float step_scale = __half2float(row_consts[3 * r + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[r];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[r]);
  float acc[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
  switch (nb) {
    case 2:
      row_dot_i8<MT, 2>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
      break;
    case 4:
      row_dot_i8<MT, 4>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
      break;
    default:
      row_dot_i8<MT, 8>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
      break;
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = acc[m];
  #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      v += __shfl_xor(v, off, kWave);
    if (lane == 0 && m0 + m < M)
      out[static_cast<int64_t>(m0 + m) * ldo + r] = __float2half_rn(v);
  }
}

// ---------------------------------------------------------------------------
// Routed experts (F16 mode). Experts are stacked row-wise: expert e owns rows
// [e * n_out, (e + 1) * n_out) of one ragged tensor, and has its own
// smoothing vector and Hadamard flags (the quantizer calibrates each expert
// on the tokens routed to it). A (token, k) pair is p = t * top_k + k.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void transform16(float (&v)[kGroup],
                                            const half* __restrict__ s,
                                            const bool rotate) {
  if (s != nullptr) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] = v[i] / __half2float(s[i]);
  }
  if (rotate) {
  #pragma unroll
    for (int len = 1; len < kGroup; len <<= 1) {
  #pragma unroll
      for (int i = 0; i < kGroup; ++i) {
        if ((i & len) == 0) {
          const float a = v[i], b = v[i + len];
          v[i] = a + b;
          v[i + len] = a - b;
        }
      }
    }
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;
  }
}

// y[p] = transform_{expert[p]}(x[p / src_div]); rows of unrouted pairs
// (expert < 0) are zero.
__global__ void moe_act_transform_kernel(const half* __restrict__ x,
                                         const int64_t ldx, const int src_div,
                                         const int32_t* __restrict__ expert,
                                         const half* __restrict__ col_scale,
                                         const uint8_t* __restrict__ had,
                                         half* __restrict__ y, const int P,
                                         const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(P) * groups) return;
  const int p = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const int e = expert[p];
  half* yp = y + static_cast<int64_t>(p) * K + gi * kGroup;
  if (e < 0) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) yp[i] = __float2half(0.0f);
    return;
  }
  const half* xp = x + (p / src_div) * ldx + gi * kGroup;
  float v[kGroup];
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
  const half* s = col_scale != nullptr
                      ? col_scale + static_cast<int64_t>(e) * K + gi * kGroup
                      : nullptr;
  const bool rot = had != nullptr && had[static_cast<int64_t>(e) * groups + gi];
  transform16(v, s, rot);
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) yp[i] = __float2half_rn(v[i]);
}

// Per-lane partial dot of one fp16 activation row with weight row gr.
__device__ __forceinline__ float row_dot1(
    const half* __restrict__ x, const int K, const int gr,
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, const int lane) {
  const int nb = row_nbits[gr];
  const int lb = log2_block(row_block[gr]);
  const float lo_min = __half2float(row_consts[3 * gr + 0]);
  const float lo_scale = __half2float(row_consts[3 * gr + 1]);
  const float step_scale = __half2float(row_consts[3 * gr + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[gr];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[gr]);
  float acc[1] = {0.0f};
  switch (nb) {
    case 2:
      row_dot<1, 2>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
    case 4:
      row_dot<1, 4>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
    default:
      row_dot<1, 8>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
  }
  return acc[0];
}

// REDUCE=false: out[p, r] = x[p] . W_{expert[p]}[r]            (gate/up)
// REDUCE=true:  out[t, r] = sum_k w[t, k] x[t*top_k+k] . W_e[r] (down)
template <bool REDUCE>
__global__ void __launch_bounds__(kThreads)
    moe_gemv_kernel(const half* __restrict__ x, const int K, const int top_k,
                    const int32_t* __restrict__ expert,
                    const float* __restrict__ topk_w,
                    const int32_t* __restrict__ codes,
                    const int32_t* __restrict__ code_off,
                    const uint8_t* __restrict__ blk_params,
                    const int32_t* __restrict__ param_off,
                    const half* __restrict__ row_consts,
                    const uint8_t* __restrict__ row_nbits,
                    const uint8_t* __restrict__ row_block, const int n_out,
                    half* __restrict__ out, const int64_t ldo) {
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * kRowsPerBlock + wave;
  if (r >= n_out) return;
  const int row = blockIdx.y;
  float acc = 0.0f;
  if constexpr (REDUCE) {
    for (int k = 0; k < top_k; ++k) {
      const int p = row * top_k + k;
      const int e = expert[p];
      if (e < 0) continue;
      acc += topk_w[p] * row_dot1(x + static_cast<int64_t>(p) * K, K,
                                  e * n_out + r, codes, code_off, blk_params,
                                  param_off, row_consts, row_nbits, row_block,
                                  lane);
    }
  } else {
    const int e = expert[row];
    if (e >= 0)
      acc = row_dot1(x + static_cast<int64_t>(row) * K, K, e * n_out + r, codes,
                     code_off, blk_params, param_off, row_consts, row_nbits,
                     row_block, lane);
  }
  #pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off, kWave);
  if (lane == 0)
    out[static_cast<int64_t>(row) * ldo + r] = __float2half_rn(acc);
}

// ---------------------------------------------------------------------------
// Prefill GEMM with the decode fused in (F16 mode): C[M, R] = X'[M, K] W^T.
// A 256-thread workgroup computes a GM x GN tile (16 x 16 threads, each a
// GM/16 x GN/16 micro-tile accumulated with v_dot2_f32_f16 from 16-byte LDS
// reads). The weight row headers are cached in LDS once per workgroup.
// K steps of 64 columns: the next step's activation and weight words (and
// block parameters) are loaded into registers while the current step is
// computed from LDS; the decode happens when the prefetched words are
// written to LDS.
// ---------------------------------------------------------------------------
constexpr int kGemmPad = 8;  // halves of padding per LDS row
constexpr int kGemm2BK = 64;

// MOE: grouped GEMM over the (token, expert) pairs of moe_align_block_size:
// tile row block y covers sorted_ids[y*GM .. y*GM+GM) (pair indices; >= M is
// padding), all of one expert e = expert_ids[y], whose weight rows are
// [e*R, (e+1)*R); output rows are the pair indices.
template <int GM, int GN, bool MOE = false>
__global__ void __launch_bounds__(kThreads)
    gemm2_kernel(const half* __restrict__ x, const int M, const int K,
                 const int32_t* __restrict__ codes,
                 const int32_t* __restrict__ code_off,
                 const uint8_t* __restrict__ blk_params,
                 const int32_t* __restrict__ param_off,
                 const half* __restrict__ row_consts,
                 const uint8_t* __restrict__ row_nbits,
                 const uint8_t* __restrict__ row_block, half* __restrict__ out,
                 const int64_t ldo, const int R,
                 const int32_t* __restrict__ sorted_ids = nullptr,
                 const int32_t* __restrict__ expert_ids = nullptr,
                 const int32_t* __restrict__ ntpp = nullptr) {
  constexpr int TM = GM / 16;
  constexpr int TN = GN / 16;
  constexpr int BK = kGemm2BK;
  constexpr int LD = BK + kGemmPad;
  constexpr int XV = GM * (BK / 8) / kThreads;       // uint4 of X per thread
  constexpr int WG = GN * (BK / kGroup) / kThreads;  // groups per thread
  static_assert(XV >= 1 && WG >= 1, "tile too small for 256 threads");
  __shared__ __align__(16) half xs[GM * LD];
  __shared__ __align__(16) half ws[GN * LD];
  __shared__ RowHdrS hdr[GN];
  const int tid = threadIdx.x;
  const int m_base = blockIdx.y * GM;
  const int r_base = blockIdx.x * GN;
  int row_off = 0;
  if constexpr (MOE) {
    if (m_base >= ntpp[0]) return;  // block-uniform, before any barrier
    const int e = expert_ids[blockIdx.y];
    if (e < 0) return;
    row_off = e * R;
  }
  // Activation / output row of tile row m (pair index under MOE).
  auto xrow = [&](int m) -> int {
    if constexpr (MOE) {
      const int p = sorted_ids[m_base + m];
      return p < M ? p : -1;
    } else {
      return m_base + m < M ? m_base + m : -1;
    }
  };
  for (int i = tid; i < GN; i += kThreads) {
    const int r = r_base + i;
    RowHdrS h;
    if (r < R) {
      const int gr = row_off + r;
      h.nb = row_nbits[gr];
      h.lb = log2_block(row_block[gr]);
      h.lo_min = __half2float(row_consts[3 * gr + 0]);
      h.lo_scale = __half2float(row_consts[3 * gr + 1]);
      h.step_scale = __half2float(row_consts[3 * gr + 2]);
      h.code_off = code_off[gr];
      h.param_off = param_off[gr];
    } else {
      h.nb = 0;
    }
    hdr[i] = h;
  }
  __syncthreads();

  uint4 xr[XV];
  uint4 wr[WG];     // up to four words per group (INT8)
  uint32_t pr[WG];  // lo_u8 | step_u8 << 8
  auto load = [&](int k0) {
  #pragma unroll
    for (int v = 0; v < XV; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / 8), c = (i % (BK / 8)) * 8;
      xr[v] = make_uint4(0, 0, 0, 0);
      const int xm = xrow(row);
      if (xm >= 0)
        xr[v] = *reinterpret_cast<const uint4*>(
            x + static_cast<int64_t>(xm) * K + k0 + c);
    }
  #pragma unroll
    for (int v = 0; v < WG; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / kGroup), gl = i % (BK / kGroup);
      const RowHdrS& h = hdr[row];
      wr[v] = make_uint4(0, 0, 0, 0);
      pr[v] = 0;
      if (h.nb != 0) {
        const int g = k0 / kGroup + gl;
        const uint32_t* words =
            reinterpret_cast<const uint32_t*>(codes) + h.code_off;
        if (h.nb == 2)
          wr[v].x = words[g];
        else if (h.nb == 4) {
          const uint2 t = *reinterpret_cast<const uint2*>(words + 2 * g);
          wr[v].x = t.x;
          wr[v].y = t.y;
        } else
          wr[v] = *reinterpret_cast<const uint4*>(words + 4 * g);
        const int blk = (g * kGroup) >> h.lb;
        const uint8_t* p =
            blk_params + 2 * (static_cast<int64_t>(h.param_off) + blk);
        pr[v] =
            static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
      }
    }
  };
  auto store = [&]() {
  #pragma unroll
    for (int v = 0; v < XV; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / 8), c = (i % (BK / 8)) * 8;
      *reinterpret_cast<uint4*>(xs + row * LD + c) = xr[v];
    }
  #pragma unroll
    for (int v = 0; v < WG; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / kGroup), gl = i % (BK / kGroup);
      const RowHdrS& h = hdr[row];
      union {
        half2 h2[8];
        uint4 u[2];
      } o;
      if (h.nb == 0) {
        o.u[0] = make_uint4(0, 0, 0, 0);
        o.u[1] = o.u[0];
      } else {
        half lo, step;
        decode_params(pr[v] & 0xFFu, pr[v] >> 8, h.lo_min, h.lo_scale,
                      h.step_scale, lo, step);
        const half2 lo2 = __half2half2(lo), st2 = __half2half2(step);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(&wr[v]);
        if (h.nb == 2)
          decode16<2>(q, lo2, st2, o.h2);
        else if (h.nb == 4)
          decode16<4>(q, lo2, st2, o.h2);
        else
          decode16<8>(q, lo2, st2, o.h2);
      }
      uint4* dst = reinterpret_cast<uint4*>(ws + row * LD + gl * kGroup);
      dst[0] = o.u[0];
      dst[1] = o.u[1];
    }
  };

  const int tx = tid % 16, ty = tid / 16;
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  load(0);
  for (int k0 = 0; k0 < K; k0 += BK) {
    __syncthreads();
    store();
    __syncthreads();
    if (k0 + BK < K) load(k0 + BK);
  #pragma unroll 2
    for (int kk = 0; kk < BK; kk += 8) {
      half2 a[TM][4], b[TN][4];
  #pragma unroll
      for (int i = 0; i < TM; ++i) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(xs + (ty + 16 * i) * LD + kk);
        const half2* hv = reinterpret_cast<const half2*>(&v);
  #pragma unroll
        for (int t = 0; t < 4; ++t) a[i][t] = hv[t];
      }
  #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(ws + (tx + 16 * j) * LD + kk);
        const half2* hv = reinterpret_cast<const half2*>(&v);
  #pragma unroll
        for (int t = 0; t < 4; ++t) b[j][t] = hv[t];
      }
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j) {
          float c = acc[i][j];
  #pragma unroll
          for (int t = 0; t < 4; ++t)
            c = __builtin_amdgcn_fdot2(a[i][t], b[j][t], c, false);
          acc[i][j] = c;
        }
    }
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = xrow(ty + 16 * i);
    if (m < 0) continue;
  #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = r_base + tx + 16 * j;
      if (r < R)
        out[static_cast<int64_t>(m) * ldo + r] = __float2half_rn(acc[i][j]);
    }
  }
}

// I8 routed experts: per (token, expert) pair, the expert's transform then the
// per-16-column int8 quantization of spec 6.2 (pairs with expert < 0 get
// zeros); the GEMV reads the pair's quantized row.
__global__ void moe_act_quant_kernel(
    const half* __restrict__ x, const int64_t ldx, const int src_div,
    const int32_t* __restrict__ expert, const half* __restrict__ col_scale,
    const uint8_t* __restrict__ had, int8_t* __restrict__ xq,
    half* __restrict__ sx, int32_t* __restrict__ sumq, const int P,
    const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(P) * groups) return;
  const int p = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const int e = expert[p];
  float v[kGroup];
  if (e >= 0) {
    const half* xp = x + (p / src_div) * ldx + gi * kGroup;
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
    const half* s = col_scale != nullptr
                        ? col_scale + static_cast<int64_t>(e) * K + gi * kGroup
                        : nullptr;
    transform16(v, s,
                had != nullptr && had[static_cast<int64_t>(e) * groups + gi]);
  } else {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] = 0.0f;
  }
  float amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const half sh = __float2half_rn(amax / 127.0f);
  const float sf = __half2float(sh);
  union {
    int8_t b[kGroup];
    uint4 u;
  } q;
  int sum = 0;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    float r = sf > 0.0f ? rintf(v[i] / sf) : 0.0f;
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    q.b[i] = static_cast<int8_t>(r);
    sum += q.b[i];
  }
  *reinterpret_cast<uint4*>(xq + static_cast<int64_t>(p) * K + gi * kGroup) =
      q.u;
  sx[static_cast<int64_t>(p) * groups + gi] = sh;
  sumq[static_cast<int64_t>(p) * groups + gi] = sum;
}

__device__ __forceinline__ float row_dot1_i8(
    const int8_t* __restrict__ xq, const half* __restrict__ sx,
    const int32_t* __restrict__ sumq, const int K, const int gr,
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, const int lane) {
  const int nb = row_nbits[gr];
  const int lb = log2_block(row_block[gr]);
  const float lo_min = __half2float(row_consts[3 * gr + 0]);
  const float lo_scale = __half2float(row_consts[3 * gr + 1]);
  const float step_scale = __half2float(row_consts[3 * gr + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[gr];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[gr]);
  float acc[1] = {0.0f};
  switch (nb) {
    case 2:
      row_dot_i8<1, 2>(xq, sx, sumq, K, 0, 1, words, params, lb, lo_min,
                       lo_scale, step_scale, lane, acc);
      break;
    case 4:
      row_dot_i8<1, 4>(xq, sx, sumq, K, 0, 1, words, params, lb, lo_min,
                       lo_scale, step_scale, lane, acc);
      break;
    default:
      row_dot_i8<1, 8>(xq, sx, sumq, K, 0, 1, words, params, lb, lo_min,
                       lo_scale, step_scale, lane, acc);
      break;
  }
  return acc[0];
}

template <bool REDUCE>
__global__ void __launch_bounds__(kThreads) moe_gemv_i8_kernel(
    const int8_t* __restrict__ xq, const half* __restrict__ sx,
    const int32_t* __restrict__ sumq, const int K, const int top_k,
    const int32_t* __restrict__ expert, const float* __restrict__ topk_w,
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, const int n_out,
    half* __restrict__ out, const int64_t ldo) {
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * kRowsPerBlock + wave;
  if (r >= n_out) return;
  const int row = blockIdx.y;
  const int groups = K / kGroup;
  float acc = 0.0f;
  auto pair = [&](int p, int e) {
    return row_dot1_i8(xq + static_cast<int64_t>(p) * K,
                       sx + static_cast<int64_t>(p) * groups,
                       sumq + static_cast<int64_t>(p) * groups, K,
                       e * n_out + r, codes, code_off, blk_params, param_off,
                       row_consts, row_nbits, row_block, lane);
  };
  if constexpr (REDUCE) {
    for (int k = 0; k < top_k; ++k) {
      const int p = row * top_k + k;
      const int e = expert[p];
      if (e >= 0) acc += topk_w[p] * pair(p, e);
    }
  } else {
    const int e = expert[row];
    if (e >= 0) acc = pair(row, e);
  }
  #pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off, kWave);
  if (lane == 0)
    out[static_cast<int64_t>(row) * ldo + r] = __float2half_rn(acc);
}

// ---------------------------------------------------------------------------
// I8 prefill (activation scale per 64 columns).
//
// Pre-pass, one thread per (token, 64-column group): x' = H16(x / s) per
// 16-column sub-group (fp32), sx = fp16(absmax / 127) over the 64 columns,
// xq = clamp(rint(x' / sx), -127, 127), and per 16-column chunk
// sq16 = sx * sum(xq) (float). Per weight block b:
//   y += sx * (step_b * sum(code * xq) + lo_b * sum(xq))
// with step_b = step_scale * u_b and lo_b = lo_min + lo_scale * lou_b taken
// unrounded (exactly linear in the stored bytes). INT2/INT4 codes are already
// valid int8; INT8 codes enter as code - 128 and lo_b gains 128 * step_b.
// ---------------------------------------------------------------------------

__global__ void act_quant64_kernel(
    const half* __restrict__ x, const int64_t ldx,
    const half* __restrict__ col_scale, const uint8_t* __restrict__ had,
    int8_t* __restrict__ xq, float* __restrict__ sx, float* __restrict__ sq16,
    const int M, const int K) {
  const int groups = K / kQG;
  const int64_t gid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= static_cast<int64_t>(M) * groups) return;
  const int m = static_cast<int>(gid / groups);
  const int g = static_cast<int>(gid % groups);
  float v[kQG];
  #pragma unroll
  for (int c = 0; c < kQG / kGroup; ++c) {
    const int col = g * kQG + c * kGroup;
    const half* xp = x + m * ldx + col;
    float t[kGroup];
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) t[i] = __half2float(xp[i]);
    if (col_scale != nullptr || had != nullptr)
      transform16(t, col_scale != nullptr ? col_scale + col : nullptr,
                  had != nullptr && had[col / kGroup]);
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[c * kGroup + i] = t[i];
  }
  float amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kQG; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const float sf = __half2float(__float2half_rn(amax / 127.0f));
  union {
    int8_t b[kQG];
    uint4 u[kQG / 16];
  } q;
  int sums[kQG / kGroup];
  #pragma unroll
  for (int c = 0; c < kQG / kGroup; ++c) {
    int sum = 0;
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) {
      float r = sf > 0.0f ? rintf(v[c * kGroup + i] / sf) : 0.0f;
      r = fminf(fmaxf(r, -127.0f), 127.0f);
      q.b[c * kGroup + i] = static_cast<int8_t>(r);
      sum += q.b[c * kGroup + i];
    }
    sums[c] = sum;
  }
  uint4* dst =
      reinterpret_cast<uint4*>(xq + static_cast<int64_t>(m) * K + g * kQG);
  #pragma unroll
  for (int i = 0; i < kQG / 16; ++i) dst[i] = q.u[i];
  sx[static_cast<int64_t>(m) * groups + g] = sf;
  #pragma unroll
  for (int c = 0; c < kQG / kGroup; ++c)
    sq16[static_cast<int64_t>(m) * (K / kGroup) + g * 4 + c] =
        sf * static_cast<float>(sums[c]);
}

// I8 prefill GEMM: GM x GN tile, 16 x 16 threads with TM x TN micro-tiles,
// one 64-column activation group per K step. Per 16-column chunk:
// d = 4 x v_dot4_i32_i8, S1 += u_chunk * d (integer), acc += lo'_chunk * sq16;
// per group: acc += sx * step_scale * float(S1).
template <int GM, int GN>
__global__ void __launch_bounds__(kThreads)
    i8gemm_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx,
                  const float* __restrict__ sq16, const int M, const int K,
                  const int32_t* __restrict__ codes,
                  const int32_t* __restrict__ code_off,
                  const uint8_t* __restrict__ blk_params,
                  const int32_t* __restrict__ param_off,
                  const half* __restrict__ row_consts,
                  const uint8_t* __restrict__ row_nbits,
                  const uint8_t* __restrict__ row_block, half* __restrict__ out,
                  const int64_t ldo, const int R) {
  constexpr int TM = GM / 16, TN = GN / 16;
  constexpr int BK = kQG;
  constexpr int LDB = BK + 16;  // bytes per LDS row (16-byte pad)
  constexpr int XV = GM * (BK / 16) / kThreads;      // uint4 of xq per thread
  constexpr int WC = GN * (BK / kGroup) / kThreads;  // weight chunks per thread
  static_assert(XV >= 1 && WC >= 1, "tile too small for 256 threads");
  __shared__ __align__(16) int8_t xs[GM * LDB];
  __shared__ __align__(16) int8_t ws[GN * LDB];
  __shared__ int wu[GN][4];
  __shared__ float wl[GN][4];
  __shared__ float sxs[GM];
  __shared__ float sqs[GM][4];
  __shared__ RowHdrS hdr[GN];
  const int tid = threadIdx.x;
  const int m_base = blockIdx.y * GM;
  const int r_base = blockIdx.x * GN;
  const int groups = K / kQG;
  for (int i = tid; i < GN; i += kThreads) {
    const int r = r_base + i;
    RowHdrS h;
    if (r < R) {
      h.nb = row_nbits[r];
      h.lb = log2_block(row_block[r]);
      h.lo_min = __half2float(row_consts[3 * r + 0]);
      h.lo_scale = __half2float(row_consts[3 * r + 1]);
      h.step_scale = __half2float(row_consts[3 * r + 2]);
      h.code_off = code_off[r];
      h.param_off = param_off[r];
    } else {
      h.nb = 0;
    }
    hdr[i] = h;
  }
  __syncthreads();

  uint4 xr[XV];
  uint4 wr[WC];
  uint32_t pr[WC];
  float sxr = 0.0f;
  float4 sqr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  auto load = [&](int g) {
    const int k0 = g * BK;
  #pragma unroll
    for (int v = 0; v < XV; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / 16), c = (i % (BK / 16)) * 16;
      xr[v] = make_uint4(0, 0, 0, 0);
      if (m_base + row < M)
        xr[v] = *reinterpret_cast<const uint4*>(
            xq + static_cast<int64_t>(m_base + row) * K + k0 + c);
    }
  #pragma unroll
    for (int v = 0; v < WC; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / 4, ch = i % 4;
      const RowHdrS& h = hdr[row];
      wr[v] = make_uint4(0, 0, 0, 0);
      pr[v] = 0;
      if (h.nb != 0) {
        const int g16 = k0 / kGroup + ch;
        const uint32_t* words =
            reinterpret_cast<const uint32_t*>(codes) + h.code_off;
        if (h.nb == 2)
          wr[v].x = words[g16];
        else if (h.nb == 4) {
          const uint2 t = *reinterpret_cast<const uint2*>(words + 2 * g16);
          wr[v].x = t.x;
          wr[v].y = t.y;
        } else
          wr[v] = *reinterpret_cast<const uint4*>(words + 4 * g16);
        const int blk = (g16 * kGroup) >> h.lb;
        const uint8_t* p =
            blk_params + 2 * (static_cast<int64_t>(h.param_off) + blk);
        pr[v] =
            static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
      }
    }
    if (tid < GM) {
      const int m = m_base + tid;
      sxr = 0.0f;
      sqr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (m < M) {
        sxr = sx[static_cast<int64_t>(m) * groups + g];
        sqr = *reinterpret_cast<const float4*>(
            sq16 + static_cast<int64_t>(m) * (K / kGroup) + g * 4);
      }
    }
  };
  auto store = [&]() {
  #pragma unroll
    for (int v = 0; v < XV; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / (BK / 16), c = (i % (BK / 16)) * 16;
      *reinterpret_cast<uint4*>(xs + row * LDB + c) = xr[v];
    }
  #pragma unroll
    for (int v = 0; v < WC; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / 4, ch = i % 4;
      const RowHdrS& h = hdr[row];
      int d[4] = {0, 0, 0, 0};
      int u = 0;
      float lo = 0.0f;
      if (h.nb != 0) {
        const uint32_t* q = reinterpret_cast<const uint32_t*>(&wr[v]);
        if (h.nb == 2)
          group_codes<2>(q, d);
        else if (h.nb == 4)
          group_codes<4>(q, d);
        else
          group_codes<8>(q, d);
        u = static_cast<int>(pr[v] >> 8);
        const float step = h.step_scale * static_cast<float>(u);
        lo = __fmaf_rn(static_cast<float>(pr[v] & 0xFFu), h.lo_scale, h.lo_min);
        if (h.nb == 8) lo = __fmaf_rn(128.0f, step, lo);
      }
      *reinterpret_cast<int4*>(ws + row * LDB + ch * 16) =
          make_int4(d[0], d[1], d[2], d[3]);
      wu[row][ch] = u;
      wl[row][ch] = lo;
    }
    if (tid < GM) {
      sxs[tid] = sxr;
      sqs[tid][0] = sqr.x;
      sqs[tid][1] = sqr.y;
      sqs[tid][2] = sqr.z;
      sqs[tid][3] = sqr.w;
    }
  };

  const int tx = tid % 16, ty = tid / 16;
  float ss[TN];
  #pragma unroll
  for (int j = 0; j < TN; ++j) ss[j] = hdr[tx + 16 * j].step_scale;
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  load(0);
  for (int g = 0; g < groups; ++g) {
    __syncthreads();
    store();
    __syncthreads();
    if (g + 1 < groups) load(g + 1);
    int s1[TM][TN];
  #pragma unroll
    for (int i = 0; i < TM; ++i)
  #pragma unroll
      for (int j = 0; j < TN; ++j) s1[i][j] = 0;
  #pragma unroll
    for (int c = 0; c < 4; ++c) {
      int4 a[TM], b[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
        a[i] =
            *reinterpret_cast<const int4*>(xs + (ty + 16 * i) * LDB + c * 16);
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        b[j] =
            *reinterpret_cast<const int4*>(ws + (tx + 16 * j) * LDB + c * 16);
  #pragma unroll
      for (int i = 0; i < TM; ++i) {
        const float sq = sqs[ty + 16 * i][c];
  #pragma unroll
        for (int j = 0; j < TN; ++j) {
          int dd = __builtin_amdgcn_sdot4(a[i].x, b[j].x, 0, false);
          dd = __builtin_amdgcn_sdot4(a[i].y, b[j].y, dd, false);
          dd = __builtin_amdgcn_sdot4(a[i].z, b[j].z, dd, false);
          dd = __builtin_amdgcn_sdot4(a[i].w, b[j].w, dd, false);
          s1[i][j] += __mul24(wu[tx + 16 * j][c], dd);
          acc[i][j] = __fmaf_rn(wl[tx + 16 * j][c], sq, acc[i][j]);
        }
      }
    }
  #pragma unroll
    for (int i = 0; i < TM; ++i) {
      const float sxi = sxs[ty + 16 * i];
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        acc[i][j] =
            __fmaf_rn(sxi * ss[j], static_cast<float>(s1[i][j]), acc[i][j]);
    }
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m_base + ty + 16 * i;
    if (m >= M) continue;
  #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = r_base + tx + 16 * j;
      if (r < R)
        out[static_cast<int64_t>(m) * ldo + r] = __float2half_rn(acc[i][j]);
    }
  }
}

#else  // non-RDNA2 device pass: empty kernels for symbol parity.

__global__ void act_quant64_kernel(const half*, const int64_t, const half*,
                                   const uint8_t*, int8_t*, float*, float*,
                                   const int, const int) {}
template <int GM, int GN>
__global__ void i8gemm_kernel(const int8_t*, const float*, const float*,
                              const int, const int, const int32_t*,
                              const int32_t*, const uint8_t*, const int32_t*,
                              const half*, const uint8_t*, const uint8_t*,
                              half*, const int64_t, const int) {}

__global__ void moe_act_quant_kernel(const half*, const int64_t, const int,
                                     const int32_t*, const half*,
                                     const uint8_t*, int8_t*, half*, int32_t*,
                                     const int, const int) {}
template <bool REDUCE>
__global__ void moe_gemv_i8_kernel(const int8_t*, const half*, const int32_t*,
                                   const int, const int, const int32_t*,
                                   const float*, const int32_t*, const int32_t*,
                                   const uint8_t*, const int32_t*, const half*,
                                   const uint8_t*, const uint8_t*, const int,
                                   half*, const int64_t) {}

template <int GM, int GN, bool MOE = false>
__global__ void gemm2_kernel(const half*, const int, const int, const int32_t*,
                             const int32_t*, const uint8_t*, const int32_t*,
                             const half*, const uint8_t*, const uint8_t*, half*,
                             const int64_t, const int, const int32_t* = nullptr,
                             const int32_t* = nullptr,
                             const int32_t* = nullptr) {}

__global__ void moe_act_transform_kernel(const half*, const int64_t, const int,
                                         const int32_t*, const half*,
                                         const uint8_t*, half*, const int,
                                         const int) {}
template <bool REDUCE>
__global__ void moe_gemv_kernel(const half*, const int, const int,
                                const int32_t*, const float*, const int32_t*,
                                const int32_t*, const uint8_t*, const int32_t*,
                                const half*, const uint8_t*, const uint8_t*,
                                const int, half*, const int64_t) {}

__global__ void act_quant_kernel(const half*, const int64_t, const half*,
                                 const uint8_t*, int8_t*, half*, int32_t*,
                                 const int, const int) {}
template <int MT>
__global__ void gemv_i8_kernel(const int8_t*, const half*, const int32_t*,
                               const int, const int, const int32_t*,
                               const int32_t*, const uint8_t*, const int32_t*,
                               const half*, const uint8_t*, const uint8_t*,
                               half*, const int64_t, const int) {}

__global__ void act_transform_kernel(const half*, const int64_t, const half*,
                                     const uint8_t*, half*, const int,
                                     const int) {}
template <int MT, int SPLIT>
__global__ void gemv_kernel(const half*, const int, const int, const int32_t*,
                            const int32_t*, const uint8_t*, const int32_t*,
                            const half*, const uint8_t*, const uint8_t*, half*,
                            const int64_t, const int) {}
__global__ void dequant_kernel(const int32_t*, const int32_t*, const uint8_t*,
                               const int32_t*, const half*, const uint8_t*,
                               const uint8_t*, half*, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace rdna_hint_flex
}  // namespace vllm

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
namespace {

namespace hf = vllm::rdna_hint_flex;

struct Hi2Weight {
  const torch::Tensor& codes;
  const torch::Tensor& code_off;
  const torch::Tensor& blk_params;
  const torch::Tensor& param_off;
  const torch::Tensor& row_consts;
  const torch::Tensor& row_nbits;
  const torch::Tensor& row_block;
};

void check_dev(const torch::Tensor& t, const char* name, at::ScalarType dt) {
  TORCH_CHECK(t.is_cuda(), "hi2: ", name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(t.scalar_type() == dt, "hi2: ", name, " must be ", dt, ", got ",
              t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), "hi2: ", name, " must be contiguous");
}

// Shape/dtype contract. The per-row offsets are validated on the host when
// the checkpoint is loaded (vllm/model_executor/layers/quantization/hi2.py);
// reading them back here would sync the stream.
int64_t check_weight(const Hi2Weight& w) {
  check_dev(w.codes, "codes", torch::kInt32);
  check_dev(w.code_off, "code_off", torch::kInt32);
  check_dev(w.blk_params, "blk_params", torch::kUInt8);
  check_dev(w.param_off, "param_off", torch::kInt32);
  check_dev(w.row_consts, "row_consts", torch::kHalf);
  check_dev(w.row_nbits, "row_nbits", torch::kUInt8);
  check_dev(w.row_block, "row_block", torch::kUInt8);
  const int64_t R = w.row_nbits.numel();
  TORCH_CHECK(w.row_block.numel() == R, "hi2: row_block must have R entries");
  TORCH_CHECK(w.code_off.numel() == R + 1 && w.param_off.numel() == R + 1,
              "hi2: code_off/param_off must have R+1 entries");
  TORCH_CHECK(w.row_consts.numel() == 3 * R, "hi2: row_consts must be [R, 3]");
  TORCH_CHECK(R < (int64_t{1} << 31) / hf::kRowsPerBlock, "hi2: too many rows");
  return R;
}

void check_act(const torch::Tensor& x, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf,
              "hi2: activations must be fp16 CUDA/HIP tensors (RDNA2 has no "
              "bf16 dot)");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "hi2: x must be [M, ", K, "]");
  TORCH_CHECK(x.stride(1) == 1, "hi2: x must be row-major");
  TORCH_CHECK(K % 64 == 0 && K > 0, "hi2: K must be a positive multiple of 64");
}

void launch_transform(const torch::Tensor& x, const torch::Tensor& col_scale,
                      const torch::Tensor& had, torch::Tensor& y,
                      cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  const half* s = nullptr;
  if (col_scale.numel() > 0) {
    check_dev(col_scale, "col_scale", torch::kHalf);
    TORCH_CHECK(col_scale.numel() == K, "hi2: col_scale must have K entries");
    s = reinterpret_cast<const half*>(col_scale.data_ptr());
  }
  check_dev(had, "had", torch::kUInt8);
  TORCH_CHECK(had.numel() == K / hf::kGroup, "hi2: had must have K/16 entries");
  const int64_t n = M * (K / hf::kGroup);
  if (n == 0) return;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  hf::act_transform_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s,
      had.data_ptr<uint8_t>(), reinterpret_cast<half*>(y.data_ptr()),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT, int SPLIT>
void launch_gemv_ms(const torch::Tensor& x, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  constexpr int RPB = hf::kRowsPerBlock / SPLIT;
  dim3 grid((R + RPB - 1) / RPB, (M + MT - 1) / MT);
  hf::gemv_kernel<MT, SPLIT><<<grid, hf::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(M),
      static_cast<int>(K), w.codes.data_ptr<int32_t>(),
      w.code_off.data_ptr<int32_t>(), w.blk_params.data_ptr<uint8_t>(),
      w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Waves per row: enough row-waves to fill the GPU, then split K. The
// RDNA_HINT_FLEX_GEMV_SPLIT environment variable (1, 2, 4, 8) overrides it
// for tuning.
int gemv_split(int64_t R) {
  static const int forced = [] {
    const char* e = std::getenv("RDNA_HINT_FLEX_GEMV_SPLIT");
    return e ? std::atoi(e) : 0;
  }();
  if (forced > 0) return forced;
  return R >= 8192 ? 1 : (R >= 4096 ? 2 : 4);
}

template <int MT>
void launch_gemv_mt(const torch::Tensor& x, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  switch (gemv_split(R)) {
    case 1:
      launch_gemv_ms<MT, 1>(x, w, out, R, stream);
      break;
    case 2:
      launch_gemv_ms<MT, 2>(x, w, out, R, stream);
      break;
    case 4:
      launch_gemv_ms<MT, 4>(x, w, out, R, stream);
      break;
    default:
      launch_gemv_ms<MT, 8>(x, w, out, R, stream);
      break;
  }
}

void launch_gemv(const torch::Tensor& x, const Hi2Weight& w, torch::Tensor& out,
                 int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0);
  if (M == 0 || R == 0) return;
  TORCH_CHECK(x.is_contiguous(), "hi2: gemv input must be contiguous");
  if (M == 1) {
    launch_gemv_mt<1>(x, w, out, R, stream);
  } else if (M == 2) {
    launch_gemv_mt<2>(x, w, out, R, stream);
  } else if (M <= 4) {
    launch_gemv_mt<4>(x, w, out, R, stream);
  } else {
    launch_gemv_mt<8>(x, w, out, R, stream);
  }
}

void launch_dequant(const Hi2Weight& w, torch::Tensor& out, int64_t R,
                    int64_t K, cudaStream_t stream) {
  if (R == 0) return;
  dim3 grid(static_cast<unsigned>(R));
  hf::dequant_kernel<<<grid, hf::kThreads, 0, stream>>>(
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// I8 activation: xq [M, K] int8, sx [M, K/16] fp16, sumq [M, K/16] int32.
void launch_act_quant(const torch::Tensor& x, const torch::Tensor& col_scale,
                      const torch::Tensor& had, bool transform,
                      torch::Tensor& xq, torch::Tensor& sx, torch::Tensor& sumq,
                      cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  const half* s = nullptr;
  const uint8_t* h = nullptr;
  if (transform) {
    if (col_scale.numel() > 0) {
      check_dev(col_scale, "col_scale", torch::kHalf);
      TORCH_CHECK(col_scale.numel() == K, "hi2: col_scale must have K entries");
      s = reinterpret_cast<const half*>(col_scale.data_ptr());
    }
    check_dev(had, "had", torch::kUInt8);
    TORCH_CHECK(had.numel() == K / hf::kGroup,
                "hi2: had must have K/16 entries");
    h = had.data_ptr<uint8_t>();
  }
  const int64_t n = M * (K / hf::kGroup);
  if (n == 0) return;
  const int threads = 256;
  hf::act_quant_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s, h,
      xq.data_ptr<int8_t>(), reinterpret_cast<half*>(sx.data_ptr()),
      sumq.data_ptr<int32_t>(), static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT>
void launch_gemv_i8_mt(const torch::Tensor& xq, const torch::Tensor& sx,
                       const torch::Tensor& sumq, const Hi2Weight& w,
                       torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = xq.size(0), K = xq.size(1);
  dim3 grid((R + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, (M + MT - 1) / MT);
  hf::gemv_i8_kernel<MT><<<grid, hf::kThreads, 0, stream>>>(
      xq.data_ptr<int8_t>(), reinterpret_cast<const half*>(sx.data_ptr()),
      sumq.data_ptr<int32_t>(), static_cast<int>(M), static_cast<int>(K),
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemv_i8(const torch::Tensor& xq, const torch::Tensor& sx,
                    const torch::Tensor& sumq, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = xq.size(0);
  if (M == 0 || R == 0) return;
  if (M == 1) {
    launch_gemv_i8_mt<1>(xq, sx, sumq, w, out, R, stream);
  } else if (M == 2) {
    launch_gemv_i8_mt<2>(xq, sx, sumq, w, out, R, stream);
  } else if (M <= 4) {
    launch_gemv_i8_mt<4>(xq, sx, sumq, w, out, R, stream);
  } else {
    launch_gemv_i8_mt<8>(xq, sx, sumq, w, out, R, stream);
  }
}

void launch_moe_transform(const torch::Tensor& x, int src_div,
                          const torch::Tensor& expert,
                          const torch::Tensor& col_scale,
                          const torch::Tensor& had, bool transform,
                          torch::Tensor& y, cudaStream_t stream) {
  const int64_t P = y.size(0), K = y.size(1);
  const int64_t n = P * (K / hf::kGroup);
  if (n == 0) return;
  const half* s = transform && col_scale.numel() > 0
                      ? reinterpret_cast<const half*>(col_scale.data_ptr())
                      : nullptr;
  const uint8_t* h =
      transform && had.numel() > 0 ? had.data_ptr<uint8_t>() : nullptr;
  const int threads = 256;
  hf::moe_act_transform_kernel<<<(n + threads - 1) / threads, threads, 0,
                                 stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), src_div,
      expert.data_ptr<int32_t>(), s, h, reinterpret_cast<half*>(y.data_ptr()),
      static_cast<int>(P), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool REDUCE>
void launch_moe_gemv(const torch::Tensor& x, int top_k,
                     const torch::Tensor& expert, const torch::Tensor& topk_w,
                     const Hi2Weight& w, int64_t n_out, torch::Tensor& out,
                     cudaStream_t stream) {
  const int64_t rows = out.size(0);
  if (rows == 0 || n_out == 0) return;
  dim3 grid((n_out + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, rows);
  hf::moe_gemv_kernel<REDUCE><<<grid, hf::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(x.size(1)),
      top_k, expert.data_ptr<int32_t>(), topk_w.data_ptr<float>(),
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      static_cast<int>(n_out), reinterpret_cast<half*>(out.data_ptr()),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GM, int GN>
void launch_gemm_t(const torch::Tensor& x, const Hi2Weight& w,
                   torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  dim3 grid((R + GN - 1) / GN, (M + GM - 1) / GM);
  hf::gemm2_kernel<GM, GN><<<grid, hf::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(M),
      static_cast<int>(K), w.codes.data_ptr<int32_t>(),
      w.code_off.data_ptr<int32_t>(), w.blk_params.data_ptr<uint8_t>(),
      w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Prefill path for M > decode_max_m: 0 = dequant + rocBLAS GEMM (default),
// 1 = the fused decode-in-GEMM kernel. The RDNA_HINT_FLEX_PREFILL
// environment variable selects it (A/B only; see the docs for the numbers).
int prefill_variant() {
  static const int v = [] {
    const char* e = std::getenv("RDNA_HINT_FLEX_PREFILL");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

void launch_gemm(const torch::Tensor& x, const Hi2Weight& w, torch::Tensor& out,
                 int64_t R, cudaStream_t stream) {
  TORCH_CHECK(x.is_contiguous(),
              "rdna_hint_flex: gemm input must be contiguous");
  launch_gemm_t<128, 128>(x, w, out, R, stream);
}

void launch_moe_act_quant(const torch::Tensor& x, int src_div,
                          const torch::Tensor& expert,
                          const torch::Tensor& col_scale,
                          const torch::Tensor& had, bool transform,
                          torch::Tensor& xq, torch::Tensor& sx,
                          torch::Tensor& sumq, cudaStream_t stream) {
  const int64_t P = xq.size(0), K = xq.size(1);
  const int64_t n = P * (K / hf::kGroup);
  if (n == 0) return;
  const half* s = transform && col_scale.numel() > 0
                      ? reinterpret_cast<const half*>(col_scale.data_ptr())
                      : nullptr;
  const uint8_t* h =
      transform && had.numel() > 0 ? had.data_ptr<uint8_t>() : nullptr;
  const int threads = 256;
  hf::moe_act_quant_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), src_div,
      expert.data_ptr<int32_t>(), s, h, xq.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sx.data_ptr()), sumq.data_ptr<int32_t>(),
      static_cast<int>(P), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool REDUCE>
void launch_moe_gemv_i8(const torch::Tensor& xq, const torch::Tensor& sx,
                        const torch::Tensor& sumq, int top_k,
                        const torch::Tensor& expert,
                        const torch::Tensor& topk_w, const Hi2Weight& w,
                        int64_t n_out, torch::Tensor& out,
                        cudaStream_t stream) {
  const int64_t rows = out.size(0);
  if (rows == 0 || n_out == 0) return;
  dim3 grid((n_out + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, rows);
  hf::moe_gemv_i8_kernel<REDUCE><<<grid, hf::kThreads, 0, stream>>>(
      xq.data_ptr<int8_t>(), reinterpret_cast<const half*>(sx.data_ptr()),
      sumq.data_ptr<int32_t>(), static_cast<int>(xq.size(1)), top_k,
      expert.data_ptr<int32_t>(), topk_w.data_ptr<float>(),
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      static_cast<int>(n_out), reinterpret_cast<half*>(out.data_ptr()),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_act_quant64(const torch::Tensor& x, const torch::Tensor& col_scale,
                        const torch::Tensor& had, bool transform,
                        torch::Tensor& xq, torch::Tensor& sx,
                        torch::Tensor& sq16, cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  const half* s = nullptr;
  const uint8_t* h = nullptr;
  if (transform) {
    if (col_scale.numel() > 0)
      s = reinterpret_cast<const half*>(col_scale.data_ptr());
    h = had.data_ptr<uint8_t>();
  }
  const int64_t n = M * (K / hf::kQG);
  if (n == 0) return;
  const int threads = 128;
  hf::act_quant64_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s, h,
      xq.data_ptr<int8_t>(), sx.data_ptr<float>(), sq16.data_ptr<float>(),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GM, int GN>
void launch_i8gemm_t(const torch::Tensor& xq, const torch::Tensor& sx,
                     const torch::Tensor& sq16, const Hi2Weight& w,
                     torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = xq.size(0), K = xq.size(1);
  dim3 grid((R + GN - 1) / GN, (M + GM - 1) / GM);
  hf::i8gemm_kernel<GM, GN><<<grid, hf::kThreads, 0, stream>>>(
      xq.data_ptr<int8_t>(), sx.data_ptr<float>(), sq16.data_ptr<float>(),
      static_cast<int>(M), static_cast<int>(K), w.codes.data_ptr<int32_t>(),
      w.code_off.data_ptr<int32_t>(), w.blk_params.data_ptr<uint8_t>(),
      w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_i8gemm(const torch::Tensor& xq, const torch::Tensor& sx,
                   const torch::Tensor& sq16, const Hi2Weight& w,
                   torch::Tensor& out, int64_t R, int tile,
                   cudaStream_t stream) {
  if (tile == 1)
    launch_i8gemm_t<64, 128>(xq, sx, sq16, w, out, R, stream);
  else
    launch_i8gemm_t<128, 128>(xq, sx, sq16, w, out, R, stream);
}

// Grouped MoE GEMM (prefill): see gemm2_kernel's MOE mode.
constexpr int kMoeBlockM = 64;

void launch_moe_gemm(const torch::Tensor& x, const torch::Tensor& sorted_ids,
                     const torch::Tensor& expert_ids, const torch::Tensor& ntpp,
                     const Hi2Weight& w, int64_t n_out, torch::Tensor& out,
                     cudaStream_t stream) {
  const int64_t P = x.size(0), K = x.size(1);
  const int64_t nblk = sorted_ids.numel() / kMoeBlockM;
  if (P == 0 || nblk == 0) return;
  dim3 grid((n_out + 127) / 128, nblk);
  hf::gemm2_kernel<kMoeBlockM, 128, true><<<grid, hf::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(P),
      static_cast<int>(K), w.codes.data_ptr<int32_t>(),
      w.code_off.data_ptr<int32_t>(), w.blk_params.data_ptr<uint8_t>(),
      w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(n_out), sorted_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(), ntpp.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// x [M, K] fp16 -> out [M, K] fp16 = H16(x / s) on flagged groups.
void rdna_hint_flex_act_transform_rdna2(torch::Tensor x,
                                        torch::Tensor col_scale,
                                        torch::Tensor had, torch::Tensor out) {
  check_act(x, x.size(1));
  check_dev(out, "out", torch::kHalf);
  TORCH_CHECK(out.sizes() == x.sizes(), "hi2: out must match x");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_transform(x, col_scale, had, out, stream);
}

// x [M, K] fp16 (already transformed) -> out [M, R] fp16 (written, not
// accumulated; out may be a row-strided view with unit column stride).
void rdna_hint_flex_gemv_rdna2(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor code_off, torch::Tensor blk_params,
                               torch::Tensor param_off,
                               torch::Tensor row_consts,
                               torch::Tensor row_nbits, torch::Tensor row_block,
                               torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_act(x, x.size(1));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.dim() == 2 && out.size(0) == x.size(0) &&
                  out.size(1) == R && out.stride(1) == 1,
              "hi2: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv(x.contiguous(), w, out, R, stream);
}

// Stored-basis reconstruction W [R, K] fp16.
void rdna_hint_flex_dequant_rdna2(torch::Tensor codes, torch::Tensor code_off,
                                  torch::Tensor blk_params,
                                  torch::Tensor param_off,
                                  torch::Tensor row_consts,
                                  torch::Tensor row_nbits,
                                  torch::Tensor row_block, torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_dev(out, "out", torch::kHalf);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == R && out.size(1) % 64 == 0,
              "hi2: out must be [R, K] with K a multiple of 64");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_dequant(w, out, R, out.size(1), stream);
}

// Routed experts: out[t] = sum_k w[t,k] * down_e(silu(gate_e(x_t)) * up_e(x_t))
// with e = topk_ids[t, k] (local expert id, < 0 = not on this rank). w13 rows
// of expert e: [e*2N, e*2N+N) gate, [e*2N+N, (e+1)*2N) up; w2 rows of expert
// e: [e*H, (e+1)*H). Per-expert col_scale [E, K] (or empty) and had [E, K/16].
torch::Tensor rdna_hint_flex_moe_rdna2(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_codes, torch::Tensor w13_code_off,
    torch::Tensor w13_blk_params, torch::Tensor w13_param_off,
    torch::Tensor w13_row_consts, torch::Tensor w13_row_nbits,
    torch::Tensor w13_row_block, torch::Tensor w13_col_scale,
    torch::Tensor w13_had, bool w13_transform, torch::Tensor w2_codes,
    torch::Tensor w2_code_off, torch::Tensor w2_blk_params,
    torch::Tensor w2_param_off, torch::Tensor w2_row_consts,
    torch::Tensor w2_row_nbits, torch::Tensor w2_row_block,
    torch::Tensor w2_col_scale, torch::Tensor w2_had, bool w2_transform,
    int64_t num_experts, int64_t intermediate, int64_t mode,
    int64_t decode_max_m) {
  const Hi2Weight w13{w13_codes,     w13_code_off,   w13_blk_params,
                      w13_param_off, w13_row_consts, w13_row_nbits,
                      w13_row_block};
  const Hi2Weight w2{w2_codes,      w2_code_off,  w2_blk_params, w2_param_off,
                     w2_row_consts, w2_row_nbits, w2_row_block};
  const int64_t R13 = check_weight(w13), R2 = check_weight(w2);
  const int64_t T = x.size(0), H = x.size(1), N = intermediate;
  check_act(x, H);
  TORCH_CHECK(N % 64 == 0,
              "rdna_hint_flex: intermediate must be a multiple of 64");
  TORCH_CHECK(R13 == num_experts * 2 * N && R2 == num_experts * H,
              "rdna_hint_flex: stacked expert rows do not match E, N, H");
  check_dev(topk_ids, "topk_ids", torch::kInt32);
  check_dev(topk_weights, "topk_weights", torch::kFloat);
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == T &&
                  topk_weights.sizes() == topk_ids.sizes(),
              "rdna_hint_flex: topk_ids/topk_weights must be [T, top_k]");
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int64_t P = T * top_k;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({T, H}, x.options());
  if (T == 0) return out;
  if (mode == 2 || (mode == 1 && T <= decode_max_m)) {
    const auto i8 = x.options().dtype(torch::kInt8);
    const auto i32 = x.options().dtype(torch::kInt32);
    auto xq = torch::empty({P, H}, i8);
    auto sx = torch::empty({P, H / 16}, x.options());
    auto sq = torch::empty({P, H / 16}, i32);
    launch_moe_act_quant(x, top_k, topk_ids, w13_col_scale, w13_had,
                         w13_transform, xq, sx, sq, stream);
    auto g13 = torch::empty({P, 2 * N}, x.options());
    launch_moe_gemv_i8<false>(xq, sx, sq, top_k, topk_ids, topk_weights, w13,
                              2 * N, g13, stream);
    auto h = at::silu(g13.narrow(1, 0, N)) * g13.narrow(1, N, N);
    auto hq = torch::empty({P, N}, i8);
    auto hs = torch::empty({P, N / 16}, x.options());
    auto hsq = torch::empty({P, N / 16}, i32);
    launch_moe_act_quant(h, 1, topk_ids, w2_col_scale, w2_had, w2_transform, hq,
                         hs, hsq, stream);
    launch_moe_gemv_i8<true>(hq, hs, hsq, top_k, topk_ids, topk_weights, w2, H,
                             out, stream);
    return out;
  }
  auto xp = torch::empty({P, H}, x.options());
  launch_moe_transform(x, top_k, topk_ids, w13_col_scale, w13_had,
                       w13_transform, xp, stream);
  auto g13 = torch::empty({P, 2 * N}, x.options());
  launch_moe_gemv<false>(xp, top_k, topk_ids, topk_weights, w13, 2 * N, g13,
                         stream);
  auto h = at::silu(g13.narrow(1, 0, N)) * g13.narrow(1, N, N);
  auto hp = torch::empty({P, N}, x.options());
  launch_moe_transform(h, 1, topk_ids, w2_col_scale, w2_had, w2_transform, hp,
                       stream);
  launch_moe_gemv<true>(hp, top_k, topk_ids, topk_weights, w2, H, out, stream);
  return out;
}

// Fused decode-in-GEMM on an already transformed activation -> out [M, R].
void rdna_hint_flex_gemm_rdna2(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor code_off, torch::Tensor blk_params,
                               torch::Tensor param_off,
                               torch::Tensor row_consts,
                               torch::Tensor row_nbits, torch::Tensor row_block,
                               torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_act(x, x.size(1));
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kHalf && out.dim() == 2 &&
          out.size(0) == x.size(0) && out.size(1) == R && out.stride(1) == 1,
      "rdna_hint_flex: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (x.size(0) > 0 && R > 0) launch_gemm(x.contiguous(), w, out, R, stream);
}

// I8 prefill: per-64-column activation quantization (transform included when
// `transform`) -> xq [M, K] int8, sx [M, K/64] float, sq16 [M, K/16] float.
void rdna_hint_flex_act_quant64_rdna2(torch::Tensor x, torch::Tensor col_scale,
                                      torch::Tensor had, bool transform,
                                      torch::Tensor xq, torch::Tensor sx,
                                      torch::Tensor sq16) {
  check_act(x, x.size(1));
  const int64_t M = x.size(0), K = x.size(1);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kFloat);
  check_dev(sq16, "sq16", torch::kFloat);
  TORCH_CHECK(
      xq.numel() == M * K && sx.numel() == M * K / 64 &&
          sq16.numel() == M * K / 16,
      "rdna_hint_flex: xq/sx/sq16 must be [M, K], [M, K/64], [M, K/16]");
  if (transform) {
    check_dev(had, "had", torch::kUInt8);
    TORCH_CHECK(had.numel() == K / 16, "rdna_hint_flex: had must be K/16");
    if (col_scale.numel() > 0) {
      check_dev(col_scale, "col_scale", torch::kHalf);
      TORCH_CHECK(col_scale.numel() == K, "rdna_hint_flex: col_scale size");
    }
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_act_quant64(x, col_scale, had, transform, xq, sx, sq16, stream);
}

// I8 prefill GEMM on a pre-quantized activation -> out [M, R] fp16.
void rdna_hint_flex_i8gemm_rdna2(
    torch::Tensor xq, torch::Tensor sx, torch::Tensor sq16, torch::Tensor codes,
    torch::Tensor code_off, torch::Tensor blk_params, torch::Tensor param_off,
    torch::Tensor row_consts, torch::Tensor row_nbits, torch::Tensor row_block,
    torch::Tensor out, int64_t tile) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kFloat);
  check_dev(sq16, "sq16", torch::kFloat);
  TORCH_CHECK(xq.dim() == 2 && xq.size(1) % 64 == 0,
              "rdna_hint_flex: xq [M, K]");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kHalf && out.dim() == 2 &&
          out.size(0) == xq.size(0) && out.size(1) == R && out.stride(1) == 1,
      "rdna_hint_flex: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xq));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (xq.size(0) > 0 && R > 0)
    launch_i8gemm(xq, sx, sq16, w, out, R, static_cast<int>(tile), stream);
}

// Routed experts, prefill building blocks. Per-pair transform with the
// expert's smoothing/rotation: out[p] = T_{expert[p]}(x[p / src_div]).
void rdna_hint_flex_moe_transform_rdna2(torch::Tensor x, int64_t src_div,
                                        torch::Tensor expert,
                                        torch::Tensor col_scale,
                                        torch::Tensor had, bool transform,
                                        torch::Tensor out) {
  check_act(x, x.size(1));
  check_dev(expert, "expert", torch::kInt32);
  check_dev(out, "out", torch::kHalf);
  TORCH_CHECK(out.size(0) == expert.numel() && out.size(1) == x.size(1),
              "rdna_hint_flex: out must be [P, K]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_moe_transform(x, static_cast<int>(src_div), expert, col_scale, had,
                       transform, out, stream);
}

// Grouped GEMM of the pairs of moe_align_block_size (block 64):
// out[p] = x[p] . W_{expert(p)}^T, stacked expert rows [e*n_out, (e+1)*n_out).
void rdna_hint_flex_moe_gemm_rdna2(
    torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor expert_ids,
    torch::Tensor ntpp, torch::Tensor codes, torch::Tensor code_off,
    torch::Tensor blk_params, torch::Tensor param_off, torch::Tensor row_consts,
    torch::Tensor row_nbits, torch::Tensor row_block, int64_t n_out,
    torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  check_weight(w);
  check_act(x, x.size(1));
  TORCH_CHECK(x.is_contiguous(), "rdna_hint_flex: x must be contiguous");
  check_dev(sorted_ids, "sorted_ids", torch::kInt32);
  check_dev(expert_ids, "expert_ids", torch::kInt32);
  check_dev(ntpp, "ntpp", torch::kInt32);
  TORCH_CHECK(sorted_ids.numel() % kMoeBlockM == 0 &&
                  expert_ids.numel() * kMoeBlockM >= sorted_ids.numel(),
              "rdna_hint_flex: moe_align_block_size must use block size 64");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.size(0) == x.size(0) && out.size(1) == n_out &&
                  out.stride(1) == 1,
              "rdna_hint_flex: out must be fp16 [P, n_out]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_moe_gemm(x, sorted_ids, expert_ids, ntpp, w, n_out, out, stream);
}

// I8 activation quantization (transform included when `transform`).
void rdna_hint_flex_act_quant_rdna2(torch::Tensor x, torch::Tensor col_scale,
                                    torch::Tensor had, bool transform,
                                    torch::Tensor xq, torch::Tensor sx,
                                    torch::Tensor sumq) {
  check_act(x, x.size(1));
  const int64_t M = x.size(0), K = x.size(1);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kHalf);
  check_dev(sumq, "sumq", torch::kInt32);
  TORCH_CHECK(xq.numel() == M * K && sx.numel() == M * K / 16 &&
                  sumq.numel() == M * K / 16,
              "hi2: xq/sx/sumq must be [M, K], [M, K/16], [M, K/16]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_act_quant(x, col_scale, had, transform, xq, sx, sumq, stream);
}

// I8 GEMV from a quantized activation -> out [M, R] fp16.
void rdna_hint_flex_gemv_i8_rdna2(
    torch::Tensor xq, torch::Tensor sx, torch::Tensor sumq, torch::Tensor codes,
    torch::Tensor code_off, torch::Tensor blk_params, torch::Tensor param_off,
    torch::Tensor row_consts, torch::Tensor row_nbits, torch::Tensor row_block,
    torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kHalf);
  check_dev(sumq, "sumq", torch::kInt32);
  TORCH_CHECK(xq.dim() == 2 && xq.size(1) % 64 == 0, "hi2: xq must be [M, K]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.dim() == 2 && out.size(0) == xq.size(0) &&
                  out.size(1) == R && out.stride(1) == 1,
              "hi2: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xq));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv_i8(xq, sx, sumq, w, out, R, stream);
}

// y = x . W^T for an rdna_hint_flex tensor (transform included when
// `transform`). mode 0 (F16): M <= decode_max_m runs the F16 GEMV, otherwise
// dequant + rocBLAS GEMM. mode 1 (I8 decode): M <= decode_max_m runs the I8
// path, larger M as mode 0. mode 2 (I8 everywhere): the I8 path for every M.
torch::Tensor rdna_hint_flex_linear_rdna2(
    torch::Tensor x, torch::Tensor col_scale, torch::Tensor had,
    torch::Tensor codes, torch::Tensor code_off, torch::Tensor blk_params,
    torch::Tensor param_off, torch::Tensor row_consts, torch::Tensor row_nbits,
    torch::Tensor row_block, bool transform, int64_t decode_max_m,
    int64_t mode) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  const int64_t K = x.size(1);
  check_act(x, K);
  TORCH_CHECK(mode >= 0 && mode <= 2, "hi2: mode must be 0, 1 or 2");
  const int64_t M = x.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({M, R}, x.options());
  if (M == 0 || R == 0) return out;

  if (mode == 2 || (mode == 1 && M <= decode_max_m)) {
    const auto opts = x.options();
    auto xq = torch::empty({M, K}, opts.dtype(torch::kInt8));
    auto sx = torch::empty({M, K / 16}, opts);
    auto sumq = torch::empty({M, K / 16}, opts.dtype(torch::kInt32));
    launch_act_quant(x, col_scale, had, transform, xq, sx, sumq, stream);
    launch_gemv_i8(xq, sx, sumq, w, out, R, stream);
    return out;
  }

  torch::Tensor xt;
  if (transform) {
    xt = torch::empty({M, K}, x.options());
    launch_transform(x, col_scale, had, xt, stream);
  } else {
    xt = x.contiguous();
  }
  if (M <= decode_max_m) {
    launch_gemv(xt, w, out, R, stream);
  } else if (prefill_variant() > 0) {
    launch_gemm(xt, w, out, R, stream);
  } else {
    auto wd = torch::empty({R, K}, x.options());
    launch_dequant(w, wd, R, K, stream);
    at::mm_out(out, xt, wd.t());
  }
  return out;
}
