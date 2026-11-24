// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native HIP kernels of the Inkling-2-Mini DSpark draft (mtp.safetensors) for
// RDNA2 (gfx1030, wave32, V_DOT2_F32_F16 as the matrix primitive).  Everything
// the draft runs per proposal lives here: tap capture, the DFlare mix +
// RMSNorm, the skinny and dense projections, per-head Q/K RMSNorm + RoPE fused
// with the paged KV write, the bidirectional block attention (split-K over the
// paged cache), the residual/RMSNorm glue, and the recurrent Markov head with
// the hidden_out-folded LM-head correction and its softcap.
//
// Numerics (see docs/rdna2/inkling2-dspark.md): fp16 weights and GEMM operands,
// fp32 accumulation (dot2), an fp32 residual stream and fp32 reductions,
// softmax and gating; fp16 only where a tensor feeds a GEMM or is the cached
// draft logits. The taps are stored as bf16 bits (the precision the draft was
// trained on, and no fp16 overflow of the target's block states).  RoPE angles
// are formed in float64 and reduced before the fp32 sincos, so long positions
// stay exact.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.  No kernel synchronizes with the host or
// allocates: every entry point is capturable in a HIP graph.
//
// Kernels exist only for gfx1030 (__HIP__RDNA2__); other device passes compile
// nothing from this file and the host entry points are host-pass only.

#include <cstdint>
#include <cmath>
#include <cstring>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rdna_hint_flex_dense_gemm_kernel.cuh"
#include "rdna_hint_flex_dense_gemm_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #ifndef __HIP__RDNA2__
    #define __HIP__RDNA2__
  #endif
#endif

namespace inkling2_dspark {

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

constexpr int kHead = 128;     // head_dim
constexpr int kBlockRows = 8;  // queries (anchor + masks) per request
constexpr int kQPerKv = 3;     // GQA group: 24 q heads / 8 kv heads
constexpr int kNumQ = kQPerKv * kBlockRows;  // 24 query vectors per KV head
constexpr int kTile = 64;                    // keys per attention tile
constexpr int kRank = 256;                   // Markov rank
constexpr int kGateWidth = 3 * kRank;        // gate | candidate | output
constexpr int kPartStride = kHead + 2;       // O[128], m, l per query and split
constexpr float kHalfMax = 65504.0f;

// ---------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
  return __uint_as_float(static_cast<unsigned int>(b) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);                     // round to nearest even
  return static_cast<unsigned short>(u >> 16);
}

__device__ __forceinline__ unsigned short to_bf16_bits(float v) {
  return f32_to_bf16(v);
}
__device__ __forceinline__ unsigned short to_bf16_bits(__half v) {
  return f32_to_bf16(__half2float(v));
}
__device__ __forceinline__ unsigned short to_bf16_bits(unsigned short v) {
  return v;  // already bf16
}

// fp32 -> fp16 saturating at the finite range (NaN stays NaN).
__device__ __forceinline__ __half to_half_sat(float x) {
  x = x > kHalfMax ? kHalfMax : (x < -kHalfMax ? -kHalfMax : x);
  return __float2half_rn(x);
}

__device__ __forceinline__ __half2 as_half2(unsigned int u) {
  return *reinterpret_cast<const __half2*>(&u);
}

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));

// One dword = two fp16 values: fp32 a.x * b.x + a.y * b.y + c (v_dot2_f32_f16).
// The operands are taken by value on purpose (see dot2u in
// rdna_hint_flex_dense_gemm_kernel.cuh: bit_cast of a vector-element lvalue
// misreads on clang 23 / gfx1030).
__device__ __forceinline__ float dot2u(unsigned int a, unsigned int b,
                                       float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
}

// 8 fp16 x 8 fp16 dot product on V_DOT2_F32_F16, fp32 accumulation.
__device__ __forceinline__ float dot8(const uint4& a, const uint4& b,
                                      float acc) {
  acc = dot2u(a.x, b.x, acc);
  acc = dot2u(a.y, b.y, acc);
  acc = dot2u(a.z, b.z, acc);
  acc = dot2u(a.w, b.w, acc);
  return acc;
}

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  const unsigned int w[4] = {v.x, v.y, v.z, v.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 p = __half22float2(as_half2(w[i]));
    f[2 * i] = p.x;
    f[2 * i + 1] = p.y;
  }
}

__device__ __forceinline__ unsigned int pack2(float a, float b) {
  const __half2 h = __halves2half2(to_half_sat(a), to_half_sat(b));
  return *reinterpret_cast<const unsigned int*>(&h);
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 v;
  v.x = pack2(f[0], f[1]);
  v.y = pack2(f[2], f[3]);
  v.z = pack2(f[4], f[5]);
  v.w = pack2(f[6], f[7]);
  return v;
}

// 8 consecutive values of a weight vector of dtype T as fp32.
__device__ __forceinline__ void load8(const __half* p, float* f) {
  unpack8(*reinterpret_cast<const uint4*>(p), f);
}
__device__ __forceinline__ void load8(const float* p, float* f) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x;
  f[1] = a.y;
  f[2] = a.z;
  f[3] = a.w;
  f[4] = b.x;
  f[5] = b.y;
  f[6] = b.z;
  f[7] = b.w;
}

  #if defined(__HIP__RDNA2__)
template <int O>
__device__ __forceinline__ float dspark_xor_lane(float v) {
  const int x = __float_as_int(v);
  int r;
  if constexpr (O == 16)
    r = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false,
                                     false);
  else if constexpr (O == 8)
    r = __builtin_amdgcn_update_dpp(x, x, 0x168, 0xF, 0xF, false);
  else if constexpr (O == 4)
    r = __builtin_amdgcn_update_dpp(x, x, 0x164, 0xF, 0xF, false);
  else if constexpr (O == 2)
    r = __builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false);
  else
    r = __builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false);
  return __int_as_float(r);
}
  #else
template <int O>
__device__ __forceinline__ float dspark_xor_lane(float v) {
  return __shfl_xor(v, O);
}
  #endif

__device__ __forceinline__ float warp_sum(float v) {
  v += dspark_xor_lane<16>(v);
  v += dspark_xor_lane<8>(v);
  v += dspark_xor_lane<4>(v);
  v += dspark_xor_lane<2>(v);
  v += dspark_xor_lane<1>(v);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  v = fmaxf(v, dspark_xor_lane<16>(v));
  v = fmaxf(v, dspark_xor_lane<8>(v));
  v = fmaxf(v, dspark_xor_lane<4>(v));
  v = fmaxf(v, dspark_xor_lane<2>(v));
  v = fmaxf(v, dspark_xor_lane<1>(v));
  return v;
}

// Sum over the block of NV values per thread (one LDS slot per wave and value).
// Every thread returns the totals.  `red` holds NV * 32 floats.
template <int NV>
__device__ __forceinline__ void block_sum(float* v, float* red) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int nwaves = (blockDim.x + 31) >> 5;
  #pragma unroll
  for (int i = 0; i < NV; ++i) v[i] = warp_sum(v[i]);
  __syncthreads();
  if (lane == 0) {
  #pragma unroll
    for (int i = 0; i < NV; ++i) red[i * 32 + wave] = v[i];
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < NV; ++i) {
    float t = 0.f;
    for (int w = 0; w < nwaves; ++w) t += red[i * 32 + w];
    v[i] = t;
  }
}

// ---------------------------------------------------------------------------
// bf16 tap capture & integer index helpers
// ---------------------------------------------------------------------------
// An int32 / int64 device vector read through one type.
struct IntVec {
  const void* p;
  int wide;
  __device__ __forceinline__ long long operator[](long long i) const {
    return wide ? static_cast<const long long*>(p)[i]
                : static_cast<const int*>(p)[i];
  }
};

// An int32 / int64 device vector written through one type.
struct IntOut {
  void* p;
  int wide;
  __device__ __forceinline__ void set(long long i, long long v) const {
    if (wide) {
      static_cast<long long*>(p)[i] = v;
    } else {
      static_cast<int*>(p)[i] = static_cast<int>(v);
    }
  }
};

struct TapSources {
  const void* p[4];
  long long ld[4];
  int kind[4];  // 0 fp32, 1 fp16, 2 bf16
  int n;
};

// dst[row, t * hidden + c] = bf16(src_t[row, c]); every tap has its own dtype
// (the target hands the AttnRes partial blocks in fp32 and its final output in
// fp16).
__global__ void store_taps_kernel(TapSources srcs,
                                  unsigned short* __restrict__ dst,
                                  long long ld_dst, int hidden) {
  const long long row = blockIdx.x;
  for (int t = 0; t < srcs.n; ++t) {
    unsigned short* d = dst + row * ld_dst + static_cast<long long>(t) * hidden;
    if (srcs.kind[t] == 0) {
      const float* s = static_cast<const float*>(srcs.p[t]) + row * srcs.ld[t];
      for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        d[c] = to_bf16_bits(s[c]);
    } else if (srcs.kind[t] == 1) {
      const __half* s =
          static_cast<const __half*>(srcs.p[t]) + row * srcs.ld[t];
      for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        d[c] = to_bf16_bits(s[c]);
    } else {
      const unsigned short* s =
          static_cast<const unsigned short*>(srcs.p[t]) + row * srcs.ld[t];
      for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        d[c] = to_bf16_bits(s[c]);
    }
  }
}

// ---------------------------------------------------------------------------
// DFlare mix + RMSNorm: out[l, row, :] = rmsnorm(sum_t alpha[l][t] tap_t) *
// w[l] alpha = softmax(fuse_weight[alpha_row0 + l, :]).  One block per row, H/8
// threads.
// ---------------------------------------------------------------------------
template <int NT, int NL>
__global__ void mix_rms_kernel(
    const unsigned short* __restrict__ taps, long long ld_taps,
    const float* __restrict__ fuse_weight, int alpha_row0,
    const __half* __restrict__ norm_w, __half* __restrict__ out,
    const int* __restrict__ qsl, const int* __restrict__ rej, int num_rows,
    int hidden, float eps) {
  __shared__ float red[NL * 32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  long long src = row;
  if (qsl != nullptr) {
    const int e = qsl[row + 1] - rej[row] - 1;
    src = e < 0 ? 0 : e;
  }
  float a[NL][NT];
  #pragma unroll
  for (int l = 0; l < NL; ++l) {
    float w[NT], mx = -INFINITY, sum = 0.f;
  #pragma unroll
    for (int t = 0; t < NT; ++t) {
      w[t] = fuse_weight[(alpha_row0 + l) * NT + t];
      mx = fmaxf(mx, w[t]);
    }
  #pragma unroll
    for (int t = 0; t < NT; ++t) {
      w[t] = expf(w[t] - mx);
      sum += w[t];
    }
  #pragma unroll
    for (int t = 0; t < NT; ++t) a[l][t] = w[t] / sum;
  }
  float x[NT][8];
  #pragma unroll
  for (int t = 0; t < NT; ++t) {
    const uint4 v = *reinterpret_cast<const uint4*>(
        taps + src * ld_taps + static_cast<long long>(t) * hidden + tid * 8);
    const unsigned int w4[4] = {v.x, v.y, v.z, v.w};
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[t][2 * i] = bf16_to_f32(static_cast<unsigned short>(w4[i] & 0xffffu));
      x[t][2 * i + 1] = bf16_to_f32(static_cast<unsigned short>(w4[i] >> 16));
    }
  }
  float mixed[NL][8], ss[NL];
  #pragma unroll
  for (int l = 0; l < NL; ++l) {
    float s = 0.f;
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      float m = 0.f;
  #pragma unroll
      for (int t = 0; t < NT; ++t) m += a[l][t] * x[t][j];
      mixed[l][j] = m;
      s += m * m;
    }
    ss[l] = s;
  }
  block_sum<NL>(ss, red);
  #pragma unroll
  for (int l = 0; l < NL; ++l) {
    const float inv = rsqrtf(ss[l] / hidden + eps);
    float w[8], o[8];
    load8(norm_w + static_cast<long long>(l) * hidden + tid * 8, w);
  #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = mixed[l][j] * inv * w[j];
    *reinterpret_cast<uint4*>(
        out + (static_cast<long long>(l) * num_rows + row) * hidden + tid * 8) =
        pack8(o);
  }
}

// ---------------------------------------------------------------------------
// Block input: row % block == 0 -> embed_norm(embed[id]); else mask_embed.
// residual (fp32) = that row; xn (fp16) = rmsnorm(row) * in_norm_w.
// ---------------------------------------------------------------------------
template <typename IdT, typename WT>
__global__ void embed_block_kernel(
    const IdT* __restrict__ ids, const __half* __restrict__ embed,
    long long vocab_rows, const WT* __restrict__ embed_norm_w,
    const __half* __restrict__ mask_embed, const __half* __restrict__ in_norm_w,
    float* __restrict__ resid, __half* __restrict__ xn, int hidden, int block,
    float eps) {
  __shared__ float red[32];
  const long long row = blockIdx.x;
  const int tid = threadIdx.x;
  float x[8];
  if (row % block == 0) {
    long long id = static_cast<long long>(ids[row]);
    id = id < 0 ? 0 : (id >= vocab_rows ? vocab_rows - 1 : id);
    float e[8], w[8];
    load8(embed + id * hidden + tid * 8, e);
    load8(embed_norm_w + tid * 8, w);
    float s = 0.f;
  #pragma unroll
    for (int j = 0; j < 8; ++j) s += e[j] * e[j];
    float v[1] = {s};
    block_sum<1>(v, red);
    const float inv = rsqrtf(v[0] / hidden + eps);
  #pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = e[j] * inv * w[j];
  } else {
    load8(mask_embed + tid * 8, x);
  }
  float4* r4 = reinterpret_cast<float4*>(resid + row * hidden + tid * 8);
  r4[0] = make_float4(x[0], x[1], x[2], x[3]);
  r4[1] = make_float4(x[4], x[5], x[6], x[7]);
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) s += x[j] * x[j];
  float v[1] = {s};
  block_sum<1>(v, red);
  const float inv = rsqrtf(v[0] / hidden + eps);
  float w[8], o[8];
  load8(in_norm_w + tid * 8, w);
  #pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = x[j] * inv * w[j];
  *reinterpret_cast<uint4*>(xn + row * hidden + tid * 8) = pack8(o);
}

// ---------------------------------------------------------------------------
// residual += delta; out = rmsnorm(residual) * w.  delta is fp16 [rows, H]
// (splits == 0) or fp32 split-K partials [splits, rows, H] summed in order.
// ---------------------------------------------------------------------------
template <bool kPartials>
__global__ void add_rmsnorm_kernel(float* __restrict__ resid,
                                   const void* __restrict__ delta, int splits,
                                   long long split_stride,
                                   const __half* __restrict__ norm_w,
                                   __half* __restrict__ out, int hidden,
                                   float eps) {
  __shared__ float red[32];
  const long long row = blockIdx.x;
  const int tid = threadIdx.x;
  float4* r4 = reinterpret_cast<float4*>(resid + row * hidden + tid * 8);
  float4 lo = r4[0], hi = r4[1];
  float r[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  if (kPartials) {
    for (int s = 0; s < splits; ++s) {
      const float4* p = reinterpret_cast<const float4*>(
          static_cast<const float*>(delta) + s * split_stride + row * hidden +
          tid * 8);
      const float4 a = p[0], b = p[1];
      r[0] += a.x;
      r[1] += a.y;
      r[2] += a.z;
      r[3] += a.w;
      r[4] += b.x;
      r[5] += b.y;
      r[6] += b.z;
      r[7] += b.w;
    }
  } else {
    float d[8];
    load8(static_cast<const __half*>(delta) + row * hidden + tid * 8, d);
  #pragma unroll
    for (int j = 0; j < 8; ++j) r[j] += d[j];
  }
  r4[0] = make_float4(r[0], r[1], r[2], r[3]);
  r4[1] = make_float4(r[4], r[5], r[6], r[7]);
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) s += r[j] * r[j];
  float v[1] = {s};
  block_sum<1>(v, red);
  const float inv = rsqrtf(v[0] / hidden + eps);
  float w[8], o[8];
  load8(norm_w + tid * 8, w);
  #pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = r[j] * inv * w[j];
  *reinterpret_cast<uint4*>(out + row * hidden + tid * 8) = pack8(o);
}

// out[t, j] = silu(gu[t, 2j]) * gu[t, 2j + 1]  (gate/up rows interleaved)
__global__ void silu_mul_kernel(const __half* __restrict__ gu,
                                __half* __restrict__ out,
                                long long total_quads) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_quads) return;
  float f[8], o[4];
  load8(gu + i * 8, f);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float g = f[2 * j], u = f[2 * j + 1];
    o[j] = g / (1.0f + expf(-g)) * u;
  }
  uint2 v;
  v.x = pack2(o[0], o[1]);
  v.y = pack2(o[2], o[3]);
  *reinterpret_cast<uint2*>(out + i * 4) = v;
}

// dst[i, :] = src[idx[i], :]
__global__ void gather_rows_kernel(const __half* __restrict__ src,
                                   long long ld_src,
                                   const int64_t* __restrict__ idx,
                                   __half* __restrict__ dst, long long ld_dst,
                                   int hidden) {
  const long long i = blockIdx.x;
  const long long s = idx[i];
  for (int c = threadIdx.x * 8; c < hidden; c += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + i * ld_dst + c) =
        *reinterpret_cast<const uint4*>(src + s * ld_src + c);
}

// ---------------------------------------------------------------------------
// Skinny GEMM: y[M, N] = x[M, K] . W[N, K]^T for M <= MT (fp16 in, fp32
// accumulate).
//
// One wave owns NR weight rows and streams them once; the activations of the
// block's K chunk sit in LDS and are shared by all its waves.  Lanes split K in
// 16-byte pieces, the per-lane partial sums of the NR * MT outputs are reduced
// across the wave with a reduce-scatter butterfly (NV - 1 shuffles instead of
// 5 * NV).  gridDim.y splits K: the F32 epilogue stores the fp32 partial sums
// [split, M, N] (the consumer adds them in a fixed order), F16 requires one
// split.
// ---------------------------------------------------------------------------
constexpr int kGemvWaves = 8;
constexpr int kGemvKc = 512;

template <int MT, int NR, bool kF32>
__global__ __launch_bounds__(256) void gemv_kernel(
    const __half* __restrict__ x, long long ldx,
    const int64_t* __restrict__ x_index, const __half* __restrict__ w,
    long long ldw, IntVec w_row_ids, void* __restrict__ out, long long ldo,
    long long split_stride, int M, int N, int K, int k_per_split) {
  constexpr int NV = MT * NR;
  static_assert(NV == 4 || NV == 8 || NV == 16 || NV == 32, "NV must be 4..32");
  __shared__ __align__(16) __half xs[MT][kGemvKc];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int row0 = (blockIdx.x * kGemvWaves + wave) * NR;
  const int k_begin = blockIdx.y * k_per_split;
  const int k_end = min(K, k_begin + k_per_split);
  long long wr[NR];
  #pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int r = row0 + i;
    wr[i] = r < N ? (w_row_ids.p != nullptr ? w_row_ids[r]
                                            : static_cast<long long>(r))
                  : -1;
  }
  float acc[NR][MT];
  #pragma unroll
  for (int i = 0; i < NR; ++i)
  #pragma unroll
    for (int m = 0; m < MT; ++m) acc[i][m] = 0.f;

  for (int k0 = k_begin; k0 < k_end; k0 += kGemvKc) {
    const int kc = min(kGemvKc, k_end - k0);
    __syncthreads();
    for (int idx = tid; idx < MT * (kGemvKc / 8); idx += 256) {
      const int m = idx / (kGemvKc / 8), c8 = idx % (kGemvKc / 8);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (m < M && c8 * 8 < kc) {
        const long long srow = x_index != nullptr ? x_index[m] : m;
        v = *reinterpret_cast<const uint4*>(x + srow * ldx + k0 + c8 * 8);
      }
      *reinterpret_cast<uint4*>(&xs[m][c8 * 8]) = v;
    }
    __syncthreads();
    if (row0 < N) {
      for (int c = lane * 8; c < kc; c += 256) {
        uint4 wv[NR];
  #pragma unroll
        for (int i = 0; i < NR; ++i) {
          wv[i] =
              wr[i] >= 0
                  ? *reinterpret_cast<const uint4*>(w + wr[i] * ldw + k0 + c)
                  : make_uint4(0, 0, 0, 0);
        }
  #pragma unroll
        for (int m = 0; m < MT; ++m) {
          const uint4 xv = *reinterpret_cast<const uint4*>(&xs[m][c]);
  #pragma unroll
          for (int i = 0; i < NR; ++i) acc[i][m] = dot8(wv[i], xv, acc[i][m]);
        }
      }
    }
  }

  float v[NV];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
  #pragma unroll
    for (int i = 0; i < NR; ++i) v[m * NR + i] = acc[i][m];
  // reduce-scatter: after log2(NV) steps lane holds the sum of value
  // idx = lane >> (5 - log2 NV) over lanes sharing the remaining low bits
  #pragma unroll
  for (int n = NV, step = 16; n > 1; n >>= 1, step >>= 1) {
    const int half_n = n >> 1;
    const bool upper = (lane & step) != 0;
  #pragma unroll
    for (int k = 0; k < half_n; ++k) {
      const float send = upper ? v[k] : v[k + half_n];
      const float keep = upper ? v[k + half_n] : v[k];
      v[k] = keep + __shfl_xor(send, step);
    }
  }
  constexpr int kLog2Nv = NV == 4 ? 2 : (NV == 8 ? 3 : (NV == 16 ? 4 : 5));
  #pragma unroll
  for (int step = 16 >> kLog2Nv; step >= 1; step >>= 1)
    v[0] += __shfl_xor(v[0], step);

  if ((lane & ((32 >> kLog2Nv) - 1)) == 0) {
    const int idx = lane >> (5 - kLog2Nv);
    const int m = idx / NR, r = row0 + idx % NR;
    if (m < M && r < N) {
      if (kF32) {
        static_cast<float*>(out)[blockIdx.y * split_stride +
                                 static_cast<long long>(m) * ldo + r] = v[0];
      } else {
        static_cast<__half*>(out)[static_cast<long long>(m) * ldo + r] =
            to_half_sat(v[0]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Indexed dense GEMM: C[M, N] = A[M, K] . W[w_row_ids[0..N), K]^T on the
// rdna_hint_flex tile designs (D1..D6, DMA == 0), preserving exact fp32
// accumulation order and single fp16 store rounding of the materialized path.
// ---------------------------------------------------------------------------
template <class P>
__global__ __launch_bounds__(P::T) void indexed_gemm_kernel(
    const char* __restrict__ A, const char* __restrict__ W, IntVec w_row_ids,
    _Float16* __restrict__ Cout, int M, int N, int W_rows, int K, int lda,
    int ldw, int ldc) {
  #if defined(__HIP__RDNA2__)
  using Chunk = vllm::rdna_hint_flex::gemm::u32x4;
  using Frag = typename vllm::rdna_hint_flex::gemm::FragT<P::LW>::type;
  constexpr int BM = P::BM, BN = P::BN, BK = P::BK, TM = P::TM, TN = P::TN;
  constexpr int TX = P::TX, TY = P::TY, T = P::T, LD = P::LD, CH = P::CH;
  constexpr int CPR = P::CPR, NBUF = P::NBUF;
  constexpr int A_PER_T = P::A_PER_T, B_PER_T = P::B_PER_T;

  __shared__ __attribute__((aligned(16))) char lds[P::LDS_BYTES];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int swa = P::SWZ ? (ty / P::SRP) & (CPR - 1) : 0;
  const int swb = P::SWZ ? (tx / P::SRP) & (CPR - 1) : 0;
  auto store_off = [](int r, int kcc) {
    if constexpr (P::SWZ)
      return r * LD * 2 + ((kcc ^ ((r / P::SRP) & (CPR - 1))) << 4);
    else
      return (r * LD + kcc * CH) * 2;
  };
  auto frag_off = [](int kk, int sw) {
    if constexpr (P::SWZ)
      return (((kk >> 3) ^ sw) << 4) + ((kk & 7) << 1);
    else
      return kk * 2;
  };

  const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
  int tm, tn;
  const int bid = blockIdx.x;
  if constexpr (P::RAST == 0) {
    tn = bid % ntn;
    tm = bid / ntn;
  } else {
    constexpr int GM = P::RAST;
    const int per_group = GM * ntn;
    const int group = bid / per_group;
    const int first_m = group * GM;
    const int gsz = min(ntm - first_m, GM);
    tm = first_m + (bid % per_group) % gsz;
    tn = (bid % per_group) / gsz;
  }
  const int m0 = tm * BM, n0 = tn * BN;
  const int nk = K / BK;

  bool wave_live = true;
  if constexpr (P::SKIP) {
    const int w0 = __builtin_amdgcn_readfirstlane(tid) & ~31;
    wave_live = (m0 + w0 / TX < M) && (n0 + (TX >= 32 ? w0 % TX : 0) < N);
  }

  uint32_t aoff[A_PER_T], boff[B_PER_T];
    #pragma unroll
  for (int v = 0; v < A_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BM - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(m0 + row, M - 1);
    aoff[v] = static_cast<uint32_t>((grow * lda + kc) * 2);
  }
    #pragma unroll
  for (int v = 0; v < B_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BN - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(n0 + row, N - 1);
    const long long wrow = w_row_ids[grow];
    boff[v] = static_cast<uint32_t>((wrow * ldw + kc) * 2);
  }

  Chunk ra[A_PER_T], rb[B_PER_T];
  const __amdgpu_buffer_rsrc_t rsa = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(A), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(M) * static_cast<uint32_t>(lda) *
                       2u),
      0x31014000);
  const __amdgpu_buffer_rsrc_t rsw = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(W), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(W_rows) *
                       static_cast<uint32_t>(ldw) * 2u),
      0x31014000);
  auto gload = [&](int kt) {
    const uint32_t kb = static_cast<uint32_t>(kt * BK * 2);
    #pragma unroll
    for (int v = 0; v < A_PER_T; ++v)
      if (A_PER_T * T == P::A_CHUNKS || tid + v * T < P::A_CHUNKS) {
        if constexpr (P::BUF)
          ra[v] = __builtin_amdgcn_raw_buffer_load_b128(rsa, aoff[v], kb, 0);
        else
          ra[v] = *reinterpret_cast<const Chunk*>(A + aoff[v] + kb);
      }
    #pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      if (B_PER_T * T == P::B_CHUNKS || tid + v * T < P::B_CHUNKS) {
        if constexpr (P::BUF)
          rb[v] = __builtin_amdgcn_raw_buffer_load_b128(rsw, boff[v], kb, 0);
        else
          rb[v] = *reinterpret_cast<const Chunk*>(W + boff[v] + kb);
      }
  };
  auto lstore = [&](int buf) {
    char* as = lds + buf * P::STAGE * 2;
    char* bs = as + P::A_LDS * 2;
    #pragma unroll
    for (int v = 0; v < A_PER_T; ++v) {
      const int c = tid + v * T;
      if (A_PER_T * T == P::A_CHUNKS || c < P::A_CHUNKS)
        *reinterpret_cast<Chunk*>(as + store_off(c / CPR, c % CPR)) = ra[v];
    }
    #pragma unroll
    for (int v = 0; v < B_PER_T; ++v) {
      const int c = tid + v * T;
      if (B_PER_T * T == P::B_CHUNKS || c < P::B_CHUNKS) {
        char* dst = bs + store_off(c / CPR, c % CPR);
        if constexpr (P::BROT == 0) {
          *reinterpret_cast<Chunk*>(dst) = rb[v];
        } else {
          const Chunk q = rb[v];
          if constexpr (P::BROT == 1) {
            if constexpr (P::LW == 8)
              *reinterpret_cast<Chunk*>(dst) = Chunk{q.y, q.x, q.w, q.z};
            else
              *reinterpret_cast<Chunk*>(dst) = Chunk{q.w, q.x, q.y, q.z};
          } else {
            volatile uint32_t* d32 = reinterpret_cast<volatile uint32_t*>(dst);
            if constexpr (P::LW == 8) {
              d32[0] = q.y;
              d32[1] = q.x;
              d32[2] = q.w;
              d32[3] = q.z;
            } else {
              d32[0] = q.w;
              d32[1] = q.x;
              d32[2] = q.y;
              d32[3] = q.z;
            }
          }
        }
      }
    }
  };

  float acc[TM][TN];
    #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  auto compute = [&](int buf) {
    const char* as = lds + buf * P::STAGE * 2;
    const char* bs = as + P::A_LDS * 2;
    auto kstep = [&](int kk) {
      Frag af[TM], bf[TN];
    #pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const Frag*>(as + (ty + TY * i) * LD * 2 +
                                               frag_off(kk, swa));
    #pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const Frag*>(bs + (tx + TX * j) * LD * 2 +
                                               frag_off(kk, swb));
    #pragma unroll
      for (int i = 0; i < TM; ++i)
    #pragma unroll
        for (int j = 0; j < TN; ++j)
    #pragma unroll
          for (int t = 0; t < P::ND; ++t)
            acc[i][j] = dot2u(af[i][t], bf[j][P::BROT ? (t + 1) % P::ND : t],
                              acc[i][j]);
    };
    if constexpr (P::KU == 0) {
    #pragma unroll
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    } else {
    #pragma unroll P::KU
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    }
  };
  auto run = [&](int buf) {
    if constexpr (P::SKIP) {
      if (!wave_live) return;
    }
    if constexpr (P::PRIO == 1) __builtin_amdgcn_s_setprio(1);
    compute(buf);
    if constexpr (P::PRIO == 1) __builtin_amdgcn_s_setprio(0);
  };
  auto mem_begin = [&]() {
    if constexpr (P::PRIO == 2) __builtin_amdgcn_s_setprio(1);
  };
  auto mem_end = [&]() {
    if constexpr (P::PRIO == 2) __builtin_amdgcn_s_setprio(0);
  };

  if constexpr (NBUF == 2) {
    mem_begin();
    gload(0);
    lstore(0);
    mem_end();
    __syncthreads();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      mem_begin();
      gload(kt + 1);
      mem_end();
      run(0);
      mem_begin();
      lstore(1);
      mem_end();
      __syncthreads();
      mem_begin();
      if (kt + 2 < nk) gload(kt + 2);
      mem_end();
      run(1);
      mem_begin();
      if (kt + 2 < nk) lstore(0);
      mem_end();
      __syncthreads();
    }
    if (kt < nk) run(0);
  } else {
    mem_begin();
    gload(0);
    mem_end();
    for (int kt = 0; kt < nk; ++kt) {
      __syncthreads();
      mem_begin();
      lstore(0);
      mem_end();
      __syncthreads();
      mem_begin();
      if (kt + 1 < nk) gload(kt + 1);
      mem_end();
      run(0);
    }
  }

    #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + TY * i;
    if (row < M) {
    #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + tx + TX * j;
        if (col < N)
          Cout[static_cast<size_t>(row) * ldc + col] =
              static_cast<_Float16>(acc[i][j]);
      }
    }
  }
  #else
  (void)A;
  (void)W;
  (void)w_row_ids;
  (void)Cout;
  (void)M;
  (void)N;
  (void)W_rows;
  (void)K;
  (void)lda;
  (void)ldw;
  (void)ldc;
  #endif
}

// ---------------------------------------------------------------------------
// Per-head Q/K RMSNorm + neox RoPE + paged KV write.
//
// qkv row = [q heads | k heads | v heads] (q_heads may be 0: the context K/V
// rows [k | v] of the target taps).  K goes through the same norm and RoPE; V
// is stored as is.  RoPE: pairs (d, d + 64), angle = pos * theta^(-2 d / 128)
// formed in float64.  The cache page is [nb, H, bs, 2 * D] (K then V per slot);
// a slot < 0 writes nothing (padding rows).  Q (normed, roped) goes to q_out
// [T, q_heads * D]. One block per token, one wave per head.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void qkv_prep_kernel(
    const __half* __restrict__ qkv, long long ld_qkv,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ slots,
    const __half* __restrict__ q_w, const __half* __restrict__ k_w,
    const double* __restrict__ inv_freq, __half* __restrict__ q_out,
    long long ld_q, __half* __restrict__ cache, long long s_blk,
    long long s_head, long long s_tok, int block_size, int q_heads,
    int kv_heads, float eps) {
  const int t = blockIdx.x;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const long long pos = positions[t];
  const long long slot = slots[t];
  // cos / sin of this token for the lane's two frequency indices
  float cs[2], sn[2];
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const double ang = static_cast<double>(pos) * inv_freq[lane + 32 * u];
    const double two_pi = 6.283185307179586476925286766559;
    const double red = ang - rint(ang * (1.0 / two_pi)) * two_pi;
    sincosf(static_cast<float>(red), &sn[u], &cs[u]);
  }
  const __half* row = qkv + static_cast<long long>(t) * ld_qkv;
  __half* kv_slot = nullptr;
  long long blk = 0, off = 0;
  if (slot >= 0) {
    blk = slot / block_size;
    off = slot - blk * block_size;
  }
  const int items = q_heads + 2 * kv_heads;
  for (int item = wave; item < items; item += 8) {
    if (item >= q_heads + kv_heads) {  // V head: raw copy
      const int h = item - q_heads - kv_heads;
      if (slot >= 0) {
        kv_slot = cache + blk * s_blk + h * s_head + off * s_tok + kHead;
        const __half* src = row + static_cast<long long>(item) * kHead;
        for (int d = lane; d < kHead; d += 32) kv_slot[d] = src[d];
      }
      continue;
    }
    const bool is_q = item < q_heads;
    const __half* src = row + static_cast<long long>(item) * kHead;
    const __half* wt = is_q ? q_w : k_w;
    float xv[4], y[4];
  #pragma unroll
    for (int u = 0; u < 4; ++u) xv[u] = __half2float(src[lane + 32 * u]);
    float ss = xv[0] * xv[0] + xv[1] * xv[1] + xv[2] * xv[2] + xv[3] * xv[3];
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / kHead + eps);
  #pragma unroll
    for (int u = 0; u < 4; ++u)
      y[u] = xv[u] * inv * __half2float(wt[lane + 32 * u]);
    // pairs (lane, lane + 64) with frequency lane; (lane + 32, lane + 96) with
    // lane + 32
    const float o0 = y[0] * cs[0] - y[2] * sn[0];
    const float o2 = y[2] * cs[0] + y[0] * sn[0];
    const float o1 = y[1] * cs[1] - y[3] * sn[1];
    const float o3 = y[3] * cs[1] + y[1] * sn[1];
    __half* dst;
    if (is_q) {
      dst = q_out + static_cast<long long>(t) * ld_q +
            static_cast<long long>(item) * kHead;
    } else {
      if (slot < 0) continue;
      dst = cache + blk * s_blk + (item - q_heads) * s_head + off * s_tok;
    }
    dst[lane] = to_half_sat(o0);
    dst[lane + 32] = to_half_sat(o1);
    dst[lane + 64] = to_half_sat(o2);
    dst[lane + 96] = to_half_sat(o3);
  }
}

// ---------------------------------------------------------------------------
// Bidirectional block attention over the paged KV cache.
//
// Per request and KV head: 3 q heads x 8 block rows = 24 query vectors attend
// to the keys [start, seq_len): start = 0 (global) or max(0, seq_len - 8 -
// window) (local layers: `window` context positions before the block plus the
// block). grid = (splits, kv_heads, requests); a split owns whole 64-key tiles,
// writes its unnormalized O[24][128], running max (log2 domain) and sum;
// attn_combine merges.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tile_geometry(int seq_len, int window,
                                              int splits, int* start,
                                              int* n_keys,
                                              int* tiles_per_split) {
  const int p = seq_len - kBlockRows;
  const int s = window > 0 ? max(0, p - window) : 0;
  const int n = seq_len - s;
  *start = s;
  *n_keys = n;
  const int tiles = n > 0 ? (n + kTile - 1) / kTile : 0;
  *tiles_per_split = tiles > 0 ? (tiles + splits - 1) / splits : 0;
}

__global__ __launch_bounds__(256) void attn_split_kernel(
    const __half* __restrict__ q, long long ld_q,
    const __half* __restrict__ cache, long long s_blk, long long s_head,
    long long s_tok, int block_size, const int* __restrict__ block_table,
    long long bt_stride, const int* __restrict__ seq_lens,
    float* __restrict__ part, int kv_heads, int window, int splits,
    float scale_log2) {
  // Pad row stride by 8 halfs (16B) to avoid LDS bank conflicts while keeping
  // 16B alignment. Total LDS: qs 6.5 KB + kv_s 17.4 KB + ss 6.1 KB + meta 0.8
  // KB = ~30.8 KB (< 32 KB -> 4 CTAs/WGP).
  constexpr int kHeadPad = kHead + 8;
  __shared__ __align__(16) __half qs[kNumQ][kHeadPad];
  __shared__ __align__(16) __half kv_s[kTile][kHeadPad];
  __shared__ float ss[kNumQ][kTile];
  __shared__ long long koff[kTile];
  __shared__ float m_run[kNumQ], l_run[kNumQ], alpha_s[kNumQ];

  const int sp = blockIdx.x, kvh = blockIdx.y, req = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int seq_len = seq_lens[req];
  int start, n_keys, tiles_per_split;
  tile_geometry(seq_len, window, splits, &start, &n_keys, &tiles_per_split);
  const int k_lo = start + sp * tiles_per_split * kTile;
  if (n_keys <= 0 || k_lo >= seq_len) return;
  const int k_hi = min(seq_len, k_lo + tiles_per_split * kTile);

  // queries: qi = g * 8 + r  <->  q[(req * 8 + r) * ld_q + (kvh * 3 + g) * 128]
  for (int idx = tid; idx < kNumQ * (kHead / 8); idx += 256) {
    const int qi = idx / (kHead / 8), c = idx % (kHead / 8);
    const int g = qi / kBlockRows, r = qi % kBlockRows;
    *reinterpret_cast<uint4*>(&qs[qi][c * 8]) = *reinterpret_cast<const uint4*>(
        q + (static_cast<long long>(req) * kBlockRows + r) * ld_q +
        (kvh * kQPerKv + g) * kHead + c * 8);
  }
  if (tid < kNumQ) {
    m_run[tid] = -INFINITY;
    l_run[tid] = 0.f;
  }

  // Coalesced K/V staging geometry: 16 threads per key (8 halfs = 16B each),
  // 16 keys per pass x 4 passes = 64 keys loaded once from DRAM.
  const int ld_k_base = tid >> 4;  // 0..15
  const int ld_c8 = tid & 15;      // 0..15 (16B vector along kHead)
  // QK scoring geometry: 64 keys x 4 thread groups (6 queries each).
  const int kk = tid >> 2;
  const int g4 = tid & 3;
  // PV geometry: 8 waves x 3 queries/wave, 32 lanes x 4 dims/lane (8B loads).
  float o[3][4];
  #pragma unroll
  for (int j = 0; j < 3; ++j)
  #pragma unroll
    for (int u = 0; u < 4; ++u) o[j][u] = 0.f;

  for (int k0 = k_lo; k0 < k_hi; k0 += kTile) {
    const int nt = min(kTile, k_hi - k0);
    __syncthreads();
    if (tid < kTile) {
      long long offs = 0;
      if (tid < nt) {
        const int pos = k0 + tid;
        const long long blk =
            block_table[static_cast<long long>(req) * bt_stride +
                        pos / block_size];
        offs = blk * s_blk + kvh * s_head +
               static_cast<long long>(pos % block_size) * s_tok;
      }
      koff[tid] = offs;
    }
    __syncthreads();
    // Stage K tile once from DRAM with coalesced 16-byte loads.
  #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      const int k_idx = pass * 16 + ld_k_base;
      uint4 kv = make_uint4(0u, 0u, 0u, 0u);
      if (k_idx < nt) {
        kv = *reinterpret_cast<const uint4*>(cache + koff[k_idx] + ld_c8 * 8);
      }
      *reinterpret_cast<uint4*>(&kv_s[k_idx][ld_c8 * 8]) = kv;
    }
    __syncthreads();
    // scores (reading K and Q from LDS)
    {
      float sc[6];
  #pragma unroll
      for (int j = 0; j < 6; ++j) sc[j] = 0.f;
      if (kk < nt) {
        const uint4* kp = reinterpret_cast<const uint4*>(&kv_s[kk][0]);
  #pragma unroll
        for (int c = 0; c < kHead / 8; ++c) {
          const uint4 kv = kp[c];
  #pragma unroll
          for (int j = 0; j < 6; ++j) {
            const uint4 qv =
                *reinterpret_cast<const uint4*>(&qs[g4 * 6 + j][c * 8]);
            sc[j] = dot8(kv, qv, sc[j]);
          }
        }
      }
  #pragma unroll
      for (int j = 0; j < 6; ++j)
        ss[g4 * 6 + j][kk] = kk < nt ? sc[j] * scale_log2 : -INFINITY;
    }
    __syncthreads();
    // Stage V tile once from DRAM into the reused kv_s buffer while computing
    // online softmax (independent of kv_s).
    uint4 v_reg[4];
  #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      const int k_idx = pass * 16 + ld_k_base;
      v_reg[pass] = k_idx < nt ? *reinterpret_cast<const uint4*>(
                                     cache + koff[k_idx] + kHead + ld_c8 * 8)
                               : make_uint4(0u, 0u, 0u, 0u);
    }
    // online softmax: wave w owns queries w, w + 8, w + 16
    {
      for (int qi = wave; qi < kNumQ; qi += 8) {
        const float s0 = ss[qi][lane], s1 = ss[qi][lane + 32];
        const float tm = warp_max(fmaxf(s0, s1));
        const float mo = m_run[qi];
        const float mn = fmaxf(mo, tm);
        float p0 = 0.f, p1 = 0.f, al = 1.f;
        if (mn != -INFINITY) {
          p0 = exp2f(s0 - mn);
          p1 = exp2f(s1 - mn);
          al = mo == -INFINITY ? 0.f : exp2f(mo - mn);
        }
        const float ts = warp_sum(p0 + p1);
        ss[qi][lane] = p0;
        ss[qi][lane + 32] = p1;
        if (lane == 0) {
          alpha_s[qi] = al;
          m_run[qi] = mn;
          l_run[qi] = l_run[qi] * al + ts;
        }
      }
    }
  #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      const int k_idx = pass * 16 + ld_k_base;
      *reinterpret_cast<uint4*>(&kv_s[k_idx][ld_c8 * 8]) = v_reg[pass];
    }
    __syncthreads();
    // P . V: wave w owns queries 3*w .. 3*w + 2, lane l owns dims 4*l .. 4*l+3.
  #pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float al = alpha_s[wave * 3 + j];
  #pragma unroll
      for (int u = 0; u < 4; ++u) o[j][u] *= al;
    }
    for (int k = 0; k < nt; ++k) {
      const uint2 vv = *reinterpret_cast<const uint2*>(&kv_s[k][lane * 4]);
      const float2 vf0 = __half22float2(as_half2(vv.x));
      const float2 vf1 = __half22float2(as_half2(vv.y));
  #pragma unroll
      for (int j = 0; j < 3; ++j) {
        const float p = ss[wave * 3 + j][k];
        o[j][0] = fmaf(p, vf0.x, o[j][0]);
        o[j][1] = fmaf(p, vf0.y, o[j][1]);
        o[j][2] = fmaf(p, vf1.x, o[j][2]);
        o[j][3] = fmaf(p, vf1.y, o[j][3]);
      }
    }
  }
  __syncthreads();
  float* pbase =
      part + ((static_cast<long long>(req) * kv_heads + kvh) * splits + sp) *
                 kNumQ * kPartStride;
  // kPartStride = 130 floats (8-byte aligned for every qi; odd qi is not 16B).
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    float* p = pbase + (wave * 3 + j) * kPartStride + lane * 4;
    *reinterpret_cast<float2*>(p) = make_float2(o[j][0], o[j][1]);
    *reinterpret_cast<float2*>(p + 2) = make_float2(o[j][2], o[j][3]);
  }
  if (tid < kNumQ) {
    float* p = pbase + tid * kPartStride;
    p[kHead] = m_run[tid];
    p[kHead + 1] = l_run[tid];
  }
}

// out[(req * 8 + r), (kvh * 3 + g) * 128 + d]: merge the splits of one query.
// One CTA per (req, kvh, qi) -> 192 CTAs at b=1 instead of 8.
__global__ __launch_bounds__(128) void attn_combine_kernel(
    const float* __restrict__ part, const int* __restrict__ seq_lens,
    __half* __restrict__ out, long long ld_out, int kv_heads, int window,
    int splits) {
  const int qi = blockIdx.x % kNumQ;
  const int rk = blockIdx.x / kNumQ;
  const int kvh = rk % kv_heads, req = rk / kv_heads;
  const int d = threadIdx.x;
  int start, n_keys, tiles_per_split;
  tile_geometry(seq_lens[req], window, splits, &start, &n_keys,
                &tiles_per_split);
  const int tiles = n_keys > 0 ? (n_keys + kTile - 1) / kTile : 0;
  const int active =
      tiles_per_split > 0 ? (tiles + tiles_per_split - 1) / tiles_per_split : 0;
  const float* pbase = part + (static_cast<long long>(req) * kv_heads + kvh) *
                                  splits * kNumQ * kPartStride;
  float mx = -INFINITY;
  for (int s = 0; s < active; ++s)
    mx = fmaxf(
        mx,
        pbase[(static_cast<long long>(s) * kNumQ + qi) * kPartStride + kHead]);
  float acc = 0.f, l = 0.f;
  for (int s = 0; s < active; ++s) {
    const float* p =
        pbase + (static_cast<long long>(s) * kNumQ + qi) * kPartStride;
    const float wgt = exp2f(p[kHead] - mx);
    l += p[kHead + 1] * wgt;
    acc += p[d] * wgt;
  }
  const int g = qi / kBlockRows, r = qi % kBlockRows;
  out[(static_cast<long long>(req) * kBlockRows + r) * ld_out +
      (kvh * kQPerKv + g) * kHead + d] =
      to_half_sat(active > 0 && l > 0.f ? acc / l : 0.f);
}

// ---------------------------------------------------------------------------
// Markov head
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ long long load_token(const void* p, bool wide,
                                                long long i) {
  return wide ? static_cast<const int64_t*>(p)[i]
              : static_cast<const int*>(p)[i];
}

// One recurrence step of every request (block = request, thread = rank index
// k).
//   raw = bias + ctx_h[r, step] + ctx_f[r] + Wt[:, :] . [state ;
//   markov_w1[prev]] gate | cand | out = raw split; state' = sigmoid(gate)
//   state + (1-sigmoid(gate)) tanh(cand) latent = tanh(out)
// wt is rnn_proj.weight[:, :2R] transposed, [2R, 3R] fp16 (coalesced over k).
// `first` reads a zero state (the previous token is the anchor).
__global__ __launch_bounds__(256) void markov_step_kernel(
    const void* __restrict__ prev, bool prev_wide, long long prev_stride,
    float* __restrict__ state, const __half* __restrict__ w1, long long w1_rows,
    const __half* __restrict__ wt, const float* __restrict__ bias,
    const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
    __half* __restrict__ latent, int step, int nsteps, bool first) {
  __shared__ float xs[2 * kRank];
  const int r = blockIdx.x, k = threadIdx.x;
  long long tok =
      load_token(prev, prev_wide, static_cast<long long>(r) * prev_stride);
  tok = tok < 0 ? 0 : (tok >= w1_rows ? w1_rows - 1 : tok);
  const float st_old = first ? 0.f : state[r * kRank + k];
  xs[k] = st_old;
  xs[kRank + k] = __half2float(w1[tok * kRank + k]);
  __syncthreads();
  float ag = 0.f, ac = 0.f, ao = 0.f;
  #pragma unroll 8
  for (int c = 0; c < 2 * kRank; ++c) {
    const float xv = xs[c];
    const __half* wr = wt + static_cast<long long>(c) * kGateWidth;
    ag += __half2float(wr[k]) * xv;
    ac += __half2float(wr[kRank + k]) * xv;
    ao += __half2float(wr[2 * kRank + k]) * xv;
  }
  const float* ch =
      ctx_h + (static_cast<long long>(r) * nsteps + step) * kGateWidth;
  const float* cf = ctx_f + static_cast<long long>(r) * kGateWidth;
  ag += bias[k] + ch[k] + cf[k];
  ac += bias[kRank + k] + ch[kRank + k] + cf[kRank + k];
  ao += bias[2 * kRank + k] + ch[2 * kRank + k] + cf[2 * kRank + k];
  const float gate = sigmoid_f(ag);
  state[r * kRank + k] = gate * st_old + (1.0f - gate) * tanhf(ac);
  latent[r * kRank + k] = __float2half_rn(tanhf(ao));
}

// logits[r, v] = cap * tanh((base[r * nsteps + step, v] + fold_b[v] +
//                            fold_w[v, :] . latent[r, :]) / cap)
// A wave owns one vocabulary row at a time (one coalesced 512-byte read of its
// fold_w row), dotted with the latents of all R <= RP requests.
template <int RP>
__global__ __launch_bounds__(256) void head_logits_kernel(
    const __half* __restrict__ base, long long ld_base, int step, int nsteps,
    const __half* __restrict__ latent, const __half* __restrict__ fold_w,
    const float* __restrict__ fold_b, __half* __restrict__ logits,
    long long ld_logits, int R, int V, float cap) {
  constexpr int RB = 32;  // vocabulary rows per block
  __shared__ __align__(16) __half lat[RP][kRank];
  __shared__ float res[RP][RB];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const long long v0 = static_cast<long long>(blockIdx.x) * RB;
  for (int idx = tid; idx < RP * (kRank / 8); idx += 256) {
    const int r = idx / (kRank / 8), c = idx % (kRank / 8);
    *reinterpret_cast<uint4*>(&lat[r][c * 8]) =
        r < R ? *reinterpret_cast<const uint4*>(
                    latent + static_cast<long long>(r) * kRank + c * 8)
              : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  constexpr int kLog2 =
      RP == 1
          ? 0
          : (RP == 2 ? 1 : (RP == 4 ? 2 : (RP == 8 ? 3 : (RP == 16 ? 4 : 5))));
  for (int j = 0; j < RB / 8; ++j) {
    const int rl = wave * (RB / 8) + j;
    const long long v = v0 + rl;
    uint4 fw = make_uint4(0, 0, 0, 0);
    if (v < V) fw = reinterpret_cast<const uint4*>(fold_w + v * kRank)[lane];
    float p[RP];
  #pragma unroll
    for (int r = 0; r < RP; ++r)
      p[r] = dot8(fw, reinterpret_cast<const uint4*>(&lat[r][0])[lane], 0.f);
    // reduce-scatter over the wave (see gemv_kernel)
  #pragma unroll
    for (int n = RP, step_m = 16; n > 1; n >>= 1, step_m >>= 1) {
      const int half_n = n >> 1;
      const bool upper = (lane & step_m) != 0;
  #pragma unroll
      for (int k = 0; k < half_n; ++k) {
        const float send = upper ? p[k] : p[k + half_n];
        const float keep = upper ? p[k + half_n] : p[k];
        p[k] = keep + __shfl_xor(send, step_m);
      }
    }
  #pragma unroll
    for (int step_m = 16 >> kLog2; step_m >= 1; step_m >>= 1)
      p[0] += __shfl_xor(p[0], step_m);
    if ((lane & ((32 >> kLog2) - 1)) == 0) res[lane >> (5 - kLog2)][rl] = p[0];
  }
  __syncthreads();
  for (int idx = tid; idx < R * RB; idx += 256) {
    const int r = idx / RB, c = idx % RB;
    const long long v = v0 + c;
    if (v >= V) continue;
    const float b = __half2float(
        base[(static_cast<long long>(r) * nsteps + step) * ld_base + v]);
    const float x = b + fold_b[v] + res[r][c];
    logits[static_cast<long long>(r) * ld_logits + v] =
        to_half_sat(cap * tanhf(x / cap));
  }
}

// fold_b[v] = lm_head[v, :] . b  (fp32 accumulate), one wave per row
__global__ __launch_bounds__(256) void fold_bias_kernel(
    const __half* __restrict__ head, long long ld_head,
    const __half* __restrict__ b, float* __restrict__ out, int V, int hidden) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const long long v = static_cast<long long>(blockIdx.x) * 8 + wave;
  if (v >= V) return;
  float acc = 0.f;
  for (int c = lane * 8; c < hidden; c += 256) {
    acc = dot8(*reinterpret_cast<const uint4*>(head + v * ld_head + c),
               *reinterpret_cast<const uint4*>(b + c), acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) out[v] = acc;
}

// out[r * out_stride] = lowest index of the maximum of logits[r, 0:V] (fp16).
// NaN counts as -inf and ties resolve to the lowest index, as
// ``logits.masked_fill(logits.isnan(), -inf).argmax(-1)`` does.  One block per
// row.  With ``part`` (the partial of a vocabulary shard) ``out`` is not
// written: part[r] = the maximum (fp32) and part[part_rows + r] = the int32
// bits of
// ``first_id`` + its index, the (value, id) pair a cross-rank reduction takes.
__global__ __launch_bounds__(512) void argmax_rows_kernel(
    const __half* __restrict__ logits, long long ld, int V, bool vec,
    long long* __restrict__ out, long long out_stride, IntVec token_map,
    float* __restrict__ part, int part_rows, int first_id) {
  __shared__ float sv[16];
  __shared__ int si[16];
  const __half* row = logits + static_cast<long long>(blockIdx.x) * ld;
  const int tid = threadIdx.x;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  const int v8 = vec ? V / 8 : 0;
  for (int c = tid; c < v8; c += 512) {
    float f[8];
    unpack8(reinterpret_cast<const uint4*>(row)[c], f);
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = f[j] != f[j] ? -INFINITY : f[j];
      if (v > bv || (v == bv && c * 8 + j < bi)) {
        bv = v;
        bi = c * 8 + j;
      }
    }
  }
  for (int i = v8 * 8 + tid; i < V; i += 512) {
    const float x = __half2float(row[i]);
    const float v = x != x ? -INFINITY : x;
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  for (int m = 16; m >= 1; m >>= 1) {
    const float ov = __shfl_xor(bv, m);
    const int oi = __shfl_xor(bi, m);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if ((tid & 31) == 0) {
    sv[tid >> 5] = bv;
    si[tid >> 5] = bi;
  }
  __syncthreads();
  if (tid < 32) {
    bv = tid < 16 ? sv[tid] : -INFINITY;
    bi = tid < 16 ? si[tid] : 0x7fffffff;
    for (int m = 16; m >= 1; m >>= 1) {
      const float ov = __shfl_xor(bv, m);
      const int oi = __shfl_xor(bi, m);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (tid == 0) {
      const long long mapped = token_map.p != nullptr
                                   ? token_map[bi]
                                   : static_cast<long long>(first_id + bi);
      if (part == nullptr) {
        out[static_cast<long long>(blockIdx.x) * out_stride] = mapped;
      } else {
        part[blockIdx.x] = bv;
        part[part_rows + blockIdx.x] = __int_as_float(static_cast<int>(mapped));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DFlash / DSpark input preparation (the HIP port of the runner's
// ``_prepare_dflash_inputs_kernel`` and its padding programs, CP size 1).
//
// Programs (req, blk) with req < num_reqs write request ``req``'s context
// positions / slots, its query block (input ids, positions, slots) and its
// sample rows, 128 columns of the context + query span per ``blk``.  The 32
// programs past the batch (blk == 0) pad every buffer up to its maximum with
// inert values so a replayed graph never reads a stale one.  A null block
// (physical block 0, an evicted sliding-window entry) is never a writable slot.
// ---------------------------------------------------------------------------
constexpr int kPrepBlock = 128;
constexpr int kPadPrograms = 32;
constexpr long long kPadSlot = -1;

struct PrepareArgs {
  // outputs
  int* input_ids;
  long long* query_positions;
  bool* is_padding;
  int* query_start_loc;
  int* seq_lens;
  long long* query_slot;
  long long* ctx_positions;
  long long* ctx_slot;
  long long* sample_indices;
  long long* sample_pos;
  IntOut sample_idx_mapping;
  float* out_temperature;
  long long* out_seeds;
  // inputs of the target batch
  const long long* target_positions;
  const int* target_qsl;
  IntVec idx_mapping;
  const long long* last_sampled;
  IntVec next_prefill;
  IntVec num_sampled;
  IntVec num_rejected;
  const float* temperature;
  const long long* seeds;
  const int* block_table;
  long long block_table_stride;
  // scalars
  int block_size;
  int parallel_token;
  int nq;  // query rows per request (anchor + masks)
  int steps;
  int max_reqs;
  int max_tokens;
  long long max_model_len;
  int num_reqs;
  bool sample_from_anchor;
};

__device__ __forceinline__ void pad_inputs(const PrepareArgs& a, int pad_idx) {
  const int t = threadIdx.x;
  const int start = pad_idx * kPrepBlock;
  const int stride = kPadPrograms * kPrepBlock;
  const int last_query_end = a.num_reqs * a.nq;
  for (int i = a.num_reqs + start; i < a.max_reqs + 1; i += stride) {
    if (i + t < a.max_reqs + 1) a.query_start_loc[i + t] = last_query_end;
  }
  for (int i = a.num_reqs + start; i < a.max_reqs; i += stride) {
    if (i + t < a.max_reqs) a.seq_lens[i + t] = 0;
  }
  // Padded sample rows point at query row 0 (valid in the hidden states) and
  // map to request -1, which sampling ignores.
  const int pad_end = a.max_reqs * a.steps;
  for (int i = a.num_reqs * a.steps + start; i < pad_end; i += stride) {
    if (i + t < pad_end) {
      a.sample_indices[i + t] = 0;
      a.sample_pos[i + t] = 0;
      a.sample_idx_mapping.set(i + t, -1);
    }
  }
  for (int i = last_query_end + start; i < a.max_tokens; i += stride) {
    if (i + t < a.max_tokens) {
      a.input_ids[i + t] = 0;
      a.query_positions[i + t] = 0;
      a.is_padding[i + t] = true;
      a.query_slot[i + t] = kPadSlot;
    }
  }
}

__global__ __launch_bounds__(kPrepBlock) void prepare_inputs_kernel(
    PrepareArgs a) {
  const int req = blockIdx.x;
  const int t = threadIdx.x;
  if (req >= a.num_reqs) {
    if (blockIdx.y == 0) pad_inputs(a, req - a.num_reqs);
    return;
  }
  const long long req_state = a.idx_mapping[req];
  const int ctx_start = a.target_qsl[req];
  const int ctx_end = a.target_qsl[req + 1];
  const int num_ctx = ctx_end - ctx_start;
  const int valid_end = ctx_end - static_cast<int>(a.num_rejected[req]);
  const int num_valid = valid_end - ctx_start;
  // The bonus token is the last sampled one; a chunked prefill splices in the
  // next prompt token instead.
  const int bonus = a.num_sampled[req] > 0
                        ? static_cast<int>(a.last_sampled[req_state])
                        : static_cast<int>(a.next_prefill[req_state]);
  const long long last_valid_pos = a.target_positions[valid_end - 1];
  const int query_base = req * a.nq;
  const int j = blockIdx.y * kPrepBlock + t;
  const bool is_ctx = j < num_ctx;
  const bool is_valid_ctx = j < num_valid;
  const bool is_query = j >= num_valid && j < num_valid + a.nq;
  const int query_off = j - num_valid;

  // Context rows: positions of the valid rows, PAD slots for the rejected
  // suffix. The whole [0, num_ctx) span is written so no value survives from an
  // earlier batch.
  if (is_ctx) {
    long long pos = 0, slot = kPadSlot;
    if (is_valid_ctx) {
      pos = a.target_positions[ctx_start + j];
      long long b = pos / a.block_size;
      b = b < a.block_table_stride - 1 ? b : a.block_table_stride - 1;
      const long long block = a.block_table[req * a.block_table_stride + b];
      if (block != 0) slot = block * a.block_size + pos % a.block_size;
    }
    a.ctx_positions[ctx_start + j] = pos;
    a.ctx_slot[ctx_start + j] = slot;
  }

  const long long query_pos = last_valid_pos + 1 + query_off;
  const int query_idx = query_base + query_off;
  if (is_query) {
    long long b = query_pos / a.block_size;
    b = b < a.block_table_stride - 1 ? b : a.block_table_stride - 1;
    const long long block = a.block_table[req * a.block_table_stride + b];
    a.input_ids[query_idx] = query_off == 0 ? bonus : a.parallel_token;
    a.is_padding[query_idx] = false;
    a.query_positions[query_idx] =
        query_pos < a.max_model_len - 1 ? query_pos : a.max_model_len - 1;
    a.query_slot[query_idx] =
        block != 0 ? block * a.block_size + query_pos % a.block_size : kPadSlot;
    // DSpark with sample_from_anchor samples every query row (row k predicts
    // position query_pos + 1); otherwise the anchor is the bonus token and only
    // the masks (offsets >= 1) are sampled, each at its own position.
    const int sample_off = a.sample_from_anchor ? 0 : 1;
    if (query_off >= sample_off) {
      const int row = req * a.steps + (query_off - sample_off);
      a.sample_indices[row] = query_idx;
      a.sample_pos[row] = a.sample_from_anchor ? query_pos + 1 : query_pos;
      a.sample_idx_mapping.set(row, req_state);
    }
  }
  if (blockIdx.y == 0 && t == 0) {
    a.query_start_loc[req] = query_base;
    // the absolute length the draft attention reads up to (context + query)
    const long long len = last_valid_pos + 1 + a.nq;
    a.seq_lens[req] =
        static_cast<int>(len < a.max_model_len ? len : a.max_model_len);
    a.out_temperature[req_state] = a.temperature[req_state];
    a.out_seeds[req_state] = a.seeds[req_state];
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace inkling2_dspark

// ---------------------------------------------------------------------------
// Host side.  Compiled in the host pass and in the gfx1030 device pass (the
// launches below are what instantiates the template kernels for the device);
// the device passes of other architectures see nothing of this file.
// ---------------------------------------------------------------------------
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__HIP__RDNA2__)

namespace hg = vllm::rdna_hint_flex::gemm;

namespace {

using namespace inkling2_dspark;

inline void check_gpu(const torch::Tensor& t, const char* name,
                      c10::ScalarType type) {
  TORCH_CHECK(t.is_cuda(), name, " must be a GPU tensor");
  TORCH_CHECK(t.scalar_type() == type, name, " has dtype ", t.scalar_type(),
              ", expected ", type);
}

inline void check_rows(const torch::Tensor& t, const char* name,
                       int64_t align) {
  TORCH_CHECK(t.stride(-1) == 1, name, " needs unit column stride");
  TORCH_CHECK(t.dim() < 2 || t.stride(0) % align == 0, name,
              " row stride must be a multiple of ", align);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
}

inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

// dst[row, t * H + c] = bf16(srcs[t][row, c]); each source is an fp32 / fp16 /
// bf16 [rows, H] tensor with its own dtype.
void inkling2_dspark_store_taps(std::vector<torch::Tensor> const& srcs,
                                torch::Tensor dst, int64_t num_rows) {
  TORCH_CHECK(!srcs.empty() && srcs.size() <= 4, "1..4 tap sources");
  check_gpu(dst, "dst", torch::kBFloat16);
  const int64_t hidden = srcs[0].size(1);
  TORCH_CHECK(dst.dim() == 2 && dst.size(1) == hidden * (int64_t)srcs.size() &&
                  dst.size(0) >= num_rows,
              "dst must be [>= num_rows, taps * hidden]");
  TapSources ts{};
  ts.n = static_cast<int>(srcs.size());
  for (size_t i = 0; i < srcs.size(); ++i) {
    const auto type = srcs[i].scalar_type();
    TORCH_CHECK(type == torch::kFloat || type == torch::kHalf ||
                    type == torch::kBFloat16,
                "tap sources must be fp32, fp16 or bf16");
    TORCH_CHECK(srcs[i].is_cuda() && srcs[i].dim() == 2 &&
                    srcs[i].size(1) == hidden && srcs[i].size(0) >= num_rows &&
                    srcs[i].stride(1) == 1,
                "tap sources must be [>= num_rows, hidden]");
    ts.p[i] = srcs[i].data_ptr();
    ts.ld[i] = srcs[i].stride(0);
    ts.kind[i] = type == torch::kFloat ? 0 : (type == torch::kHalf ? 1 : 2);
  }
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(dst));
  auto stream = at::cuda::getCurrentCUDAStream();
  store_taps_kernel<<<dim3(static_cast<unsigned>(num_rows)), dim3(256), 0,
                      stream>>>(
      ts, reinterpret_cast<unsigned short*>(dst.data_ptr()), dst.stride(0),
      (int)hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[l, row, :] = rmsnorm(sum_t softmax(fuse_weight[row0 + l])[t] tap_t) *
// norm_w[l]. With qsl / num_rejected the source row of output row i is the last
// committed row of request i (qsl[i + 1] - num_rejected[i] - 1).
void inkling2_dspark_mix_rms(torch::Tensor taps, torch::Tensor fuse_weight,
                             torch::Tensor norm_w, torch::Tensor out,
                             std::optional<torch::Tensor> qsl,
                             std::optional<torch::Tensor> num_rejected,
                             int64_t num_rows, int64_t alpha_row0, double eps) {
  check_gpu(taps, "taps", torch::kBFloat16);
  check_gpu(fuse_weight, "fuse_weight", torch::kFloat);
  check_gpu(norm_w, "norm_w", torch::kHalf);
  check_gpu(out, "out", torch::kHalf);
  const int64_t NT = fuse_weight.size(1), NL = norm_w.size(0),
                H = norm_w.size(1);
  TORCH_CHECK(NT == 4 && (NL == 1 || NL == 3),
              "4 taps and 1 or 3 layers supported");
  TORCH_CHECK(taps.dim() == 2 && taps.size(1) == NT * H &&
                  taps.stride(1) == 1 && taps.stride(0) % 8 == 0,
              "taps must be [rows, taps * hidden]");
  TORCH_CHECK(H % 8 == 0 && H / 8 <= 1024,
              "hidden must be a multiple of 8, <= 8192");
  TORCH_CHECK(out.is_contiguous() && out.numel() >= NL * num_rows * H,
              "out must hold [layers, num_rows, hidden]");
  TORCH_CHECK(
      fuse_weight.is_contiguous() && alpha_row0 + NL <= fuse_weight.size(0),
      "fuse_weight rows out of range");
  TORCH_CHECK(qsl.has_value() == num_rejected.has_value(),
              "qsl and num_rejected go together");
  if (num_rows == 0) return;
  const int* qsl_p = nullptr;
  const int* rej_p = nullptr;
  if (qsl.has_value()) {
    TORCH_CHECK(qsl->scalar_type() == torch::kInt &&
                    num_rejected->scalar_type() == torch::kInt,
                "qsl / num_rejected must be int32");
    qsl_p = qsl->data_ptr<int>();
    rej_p = num_rejected->data_ptr<int>();
  } else {
    TORCH_CHECK(taps.size(0) >= num_rows, "taps has fewer than num_rows rows");
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(taps));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(num_rows)),
      block(static_cast<unsigned>(H / 8));
  if (NL == 3) {
    mix_rms_kernel<4, 3><<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned short*>(taps.data_ptr()),
        taps.stride(0), fuse_weight.data_ptr<float>(), (int)alpha_row0,
        reinterpret_cast<const __half*>(norm_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), qsl_p, rej_p, (int)num_rows,
        (int)H, (float)eps);
  } else {
    mix_rms_kernel<4, 1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned short*>(taps.data_ptr()),
        taps.stride(0), fuse_weight.data_ptr<float>(), (int)alpha_row0,
        reinterpret_cast<const __half*>(norm_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), qsl_p, rej_p, (int)num_rows,
        (int)H, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Block input of every request: row % block == 0 -> embed_norm(embed[id]), else
// mask_embed; residual (fp32) and xn = rmsnorm(residual) * in_norm_w (fp16).
void inkling2_dspark_embed_block(torch::Tensor ids, torch::Tensor embed,
                                 torch::Tensor embed_norm_w,
                                 torch::Tensor mask_embed,
                                 torch::Tensor in_norm_w, torch::Tensor resid,
                                 torch::Tensor xn, int64_t num_rows,
                                 int64_t block, double eps) {
  check_gpu(embed, "embed", torch::kHalf);
  check_gpu(mask_embed, "mask_embed", torch::kHalf);
  check_gpu(in_norm_w, "in_norm_w", torch::kHalf);
  check_gpu(resid, "resid", torch::kFloat);
  check_gpu(xn, "xn", torch::kHalf);
  const int64_t H = embed.size(1);
  TORCH_CHECK(H % 8 == 0 && H / 8 <= 1024 && embed.stride(1) == 1 &&
                  embed.stride(0) % 8 == 0,
              "embed must be [vocab, hidden] rows of a multiple of 8");
  TORCH_CHECK(resid.is_contiguous() && xn.is_contiguous() &&
                  resid.size(1) == H && xn.size(1) == H &&
                  resid.size(0) >= num_rows && xn.size(0) >= num_rows,
              "resid / xn must be contiguous [>= num_rows, hidden]");
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) >= num_rows,
              "ids must be [>= num_rows]");
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(embed));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(num_rows)),
      block_dim(static_cast<unsigned>(H / 8));
  const bool ids64 = ids.scalar_type() == torch::kLong;
  TORCH_CHECK(ids64 || ids.scalar_type() == torch::kInt,
              "ids must be int32 / int64");
  const bool w32 = embed_norm_w.scalar_type() == torch::kFloat;
  TORCH_CHECK(w32 || embed_norm_w.scalar_type() == torch::kHalf,
              "embed_norm_w must be fp32 or fp16");
  #define I2D_EMBED(IDT, WT)                                               \
    embed_block_kernel<IDT, WT><<<grid, block_dim, 0, stream>>>(           \
        static_cast<const IDT*>(ids.data_ptr()),                           \
        reinterpret_cast<const __half*>(embed.data_ptr()), embed.size(0),  \
        static_cast<const WT*>(embed_norm_w.data_ptr()),                   \
        reinterpret_cast<const __half*>(mask_embed.data_ptr()),            \
        reinterpret_cast<const __half*>(in_norm_w.data_ptr()),             \
        resid.data_ptr<float>(), reinterpret_cast<__half*>(xn.data_ptr()), \
        (int)H, (int)block, (float)eps)
  if (ids64) {
    if (w32) {
      I2D_EMBED(int64_t, float);
    } else {
      I2D_EMBED(int64_t, __half);
    }
  } else {
    if (w32) {
      I2D_EMBED(int, float);
    } else {
      I2D_EMBED(int, __half);
    }
  }
  #undef I2D_EMBED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// residual += delta; out = rmsnorm(residual) * norm_w.  delta: fp16 [rows, H],
// or fp32 [splits, rows, H] split-K partials when splits > 0.
void inkling2_dspark_add_rmsnorm(torch::Tensor resid, torch::Tensor delta,
                                 torch::Tensor norm_w, torch::Tensor out,
                                 int64_t num_rows, int64_t splits, double eps) {
  check_gpu(resid, "resid", torch::kFloat);
  check_gpu(norm_w, "norm_w", torch::kHalf);
  check_gpu(out, "out", torch::kHalf);
  const int64_t H = norm_w.numel();
  TORCH_CHECK(H % 8 == 0 && H / 8 <= 1024,
              "hidden must be a multiple of 8, <= 8192");
  TORCH_CHECK(resid.is_contiguous() && out.is_contiguous() &&
                  resid.size(1) == H && out.size(1) == H &&
                  resid.size(0) >= num_rows && out.size(0) >= num_rows,
              "resid / out must be contiguous [>= num_rows, hidden]");
  const bool partials = splits > 0;
  if (partials) {
    check_gpu(delta, "delta", torch::kFloat);
    TORCH_CHECK(delta.is_contiguous() && delta.dim() == 3 &&
                    delta.size(0) >= splits && delta.size(2) == H &&
                    delta.size(1) >= num_rows,
                "delta must be [splits, >= num_rows, hidden] fp32");
  } else {
    check_gpu(delta, "delta", torch::kHalf);
    TORCH_CHECK(delta.is_contiguous() && delta.size(-1) == H &&
                    delta.size(0) >= num_rows,
                "delta must be [>= num_rows, hidden] fp16");
  }
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(resid));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(num_rows)),
      block(static_cast<unsigned>(H / 8));
  const long long split_stride = partials ? delta.size(1) * H : 0;
  if (partials) {
    add_rmsnorm_kernel<true><<<grid, block, 0, stream>>>(
        resid.data_ptr<float>(), delta.data_ptr(), (int)splits, split_stride,
        reinterpret_cast<const __half*>(norm_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), (int)H, (float)eps);
  } else {
    add_rmsnorm_kernel<false><<<grid, block, 0, stream>>>(
        resid.data_ptr<float>(), delta.data_ptr(), 0, 0,
        reinterpret_cast<const __half*>(norm_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), (int)H, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[t, j] = silu(gu[t, 2j]) * gu[t, 2j + 1]
void inkling2_dspark_silu_mul(torch::Tensor gu, torch::Tensor out,
                              int64_t num_rows) {
  check_gpu(gu, "gu", torch::kHalf);
  check_gpu(out, "out", torch::kHalf);
  TORCH_CHECK(gu.is_contiguous() && out.is_contiguous() && gu.dim() == 2 &&
                  out.dim() == 2 && gu.size(1) == 2 * out.size(1) &&
                  out.size(1) % 4 == 0 && gu.size(0) >= num_rows &&
                  out.size(0) >= num_rows,
              "gu [rows, 2 * I], out [rows, I], I % 4 == 0");
  const int64_t quads = num_rows * out.size(1) / 4;
  if (quads == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(gu));
  auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_kernel<<<dim3(static_cast<unsigned>(cdiv(quads, 256))), dim3(256), 0,
                    stream>>>(reinterpret_cast<const __half*>(gu.data_ptr()),
                              reinterpret_cast<__half*>(out.data_ptr()), quads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dst[i, :] = src[idx[i], :]
void inkling2_dspark_gather_rows(torch::Tensor src, torch::Tensor idx,
                                 torch::Tensor dst, int64_t num_rows) {
  check_gpu(src, "src", torch::kHalf);
  check_gpu(dst, "dst", torch::kHalf);
  check_gpu(idx, "idx", torch::kLong);
  const int64_t H = src.size(1);
  check_rows(src, "src", 8);
  check_rows(dst, "dst", 8);
  TORCH_CHECK(H % 8 == 0 && dst.size(1) == H && dst.size(0) >= num_rows &&
                  idx.size(0) >= num_rows,
              "dst [>= num_rows, hidden], idx [>= num_rows]");
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(src));
  auto stream = at::cuda::getCurrentCUDAStream();
  gather_rows_kernel<<<dim3(static_cast<unsigned>(num_rows)), dim3(256), 0,
                       stream>>>(
      reinterpret_cast<const __half*>(src.data_ptr()), src.stride(0),
      idx.data_ptr<int64_t>(), reinterpret_cast<__half*>(dst.data_ptr()),
      dst.stride(0), (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

inline IntVec int_vec_1d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.is_contiguous(), name,
              " must be a contiguous 1D GPU tensor");
  TORCH_CHECK(t.scalar_type() == torch::kInt || t.scalar_type() == torch::kLong,
              name, " must be int32 or int64");
  return IntVec{t.data_ptr(), t.scalar_type() == torch::kLong ? 1 : 0};
}

template <int MT, int NR, bool F32>
void launch_gemv(const torch::Tensor& x,
                 const std::optional<torch::Tensor>& x_index,
                 const torch::Tensor& w, IntVec w_ids, const torch::Tensor& out,
                 int64_t M, int64_t N, int64_t K, int64_t splits, int64_t ldo,
                 int64_t split_stride, hipStream_t stream) {
  const int64_t rows_per_block = kGemvWaves * NR;
  const dim3 grid(static_cast<unsigned>(cdiv(N, rows_per_block)),
                  static_cast<unsigned>(splits));
  const int64_t kps = K / splits;
  gemv_kernel<MT, NR, F32><<<grid, dim3(256), 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr()), x.stride(0),
      x_index.has_value() ? x_index->data_ptr<int64_t>() : nullptr,
      reinterpret_cast<const __half*>(w.data_ptr()), w.stride(0), w_ids,
      out.data_ptr(), ldo, split_stride, (int)M, (int)N, (int)K, (int)kps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

using IdxD1 =
    hg::Design<128, 128, 32, 8, 16, 2, 0, 8, 8, true, 1, 1, 1, 0, 0, 1>;
using IdxD2 =
    hg::Design<128, 128, 32, 8, 16, 2, 0, 4, 8, true, 1, 1, 1, 0, 0, 1>;
using IdxD3 =
    hg::Design<128, 64, 64, 8, 8, 1, 8, 4, 8, false, 1, 0, 1, 0, 0, 1>;
using IdxD4 =
    hg::Design<256, 128, 32, 8, 16, 2, 0, 4, 8, true, 1, 1, 1, 0, 0, 1>;
using IdxD5 = hg::Design<32, 128, 64, 4, 8, 2, 0, 0, 8, true, 1, 0, 1, 0, 0, 1>;
using IdxD6 = hg::Design<64, 128, 64, 8, 8, 2, 0, 0, 8, true, 1, 0, 1, 0, 0, 1>;

template <class P>
void launch_indexed_gemm_design(const void* A, const void* W, IntVec w_ids,
                                void* C, int64_t M, int64_t N, int64_t W_rows,
                                int64_t K, int64_t lda, int64_t ldw,
                                int64_t ldc, hipStream_t stream) {
  const int64_t ntm = (M + P::BM - 1) / P::BM, ntn = (N + P::BN - 1) / P::BN;
  indexed_gemm_kernel<P>
      <<<dim3(static_cast<unsigned>(ntm * ntn)), dim3(P::T), 0, stream>>>(
          static_cast<const char*>(A), static_cast<const char*>(W), w_ids,
          static_cast<_Float16*>(C), static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(W_rows), static_cast<int>(K), static_cast<int>(lda),
          static_cast<int>(ldw), static_cast<int>(ldc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool launch_indexed_gemm(int design, const void* A, const void* W, IntVec w_ids,
                         void* C, int64_t M, int64_t N, int64_t W_rows,
                         int64_t K, int64_t lda, int64_t ldw, int64_t ldc,
                         hipStream_t stream) {
  constexpr int kBks[6] = {32, 32, 64, 32, 64, 64};
  if (design < 0 || design >= 6) return false;
  if (M == 0 || N == 0) return true;
  constexpr int64_t kMaxElems = int64_t{1} << 30;
  const hipDeviceProp_t* prop = at::cuda::getCurrentDeviceProperties();
  if (M < 0 || N < 0 || W_rows <= 0 || K <= 0 || K % kBks[design] != 0 ||
      lda % 8 != 0 || ldw % 8 != 0 || lda < K || ldw < K || ldc < N ||
      M * lda >= kMaxElems || W_rows * ldw >= kMaxElems ||
      ldc >= (int64_t{1} << 31) || (reinterpret_cast<uintptr_t>(A) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(W) & 15) != 0 ||
      std::strncmp(prop->gcnArchName, "gfx1030", 7) != 0)
    return false;
  switch (design) {
    case 0:
      launch_indexed_gemm_design<IdxD1>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
    case 1:
      launch_indexed_gemm_design<IdxD2>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
    case 2:
      launch_indexed_gemm_design<IdxD3>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
    case 3:
      launch_indexed_gemm_design<IdxD4>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
    case 4:
      launch_indexed_gemm_design<IdxD5>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
    default:
      launch_indexed_gemm_design<IdxD6>(A, W, w_ids, C, M, N, W_rows, K, lda,
                                        ldw, ldc, stream);
      break;
  }
  return true;
}

}  // namespace

// y = x . w^T with M <= 32 rows.  fp16 operands, fp32 accumulation.
//   splits == 0: out is fp16 [>= M, N] (K is not split).
//   splits  > 0: out is fp32 [>= splits, >= M, N] partial sums over K / splits
//                (K / splits a multiple of 256), summed by the consumer.
// x_index (int64 [M]) gathers the activation rows; w_row_ids (int32/int64 [N])
// gathers the weight rows directly from w without copying a subset table.
// `variant` forces the rows per wave (0: automatic); tests and benchmarks use
// it.
void inkling2_dspark_gemv_v2(torch::Tensor x, torch::Tensor w,
                             torch::Tensor out,
                             std::optional<torch::Tensor> x_index,
                             int64_t num_rows, int64_t splits, int64_t variant,
                             std::optional<torch::Tensor> w_row_ids) {
  check_gpu(x, "x", torch::kHalf);
  check_gpu(w, "w", torch::kHalf);
  const int64_t M = num_rows, K = w.size(1);
  IntVec w_ids{nullptr, 0};
  int64_t N = w.size(0);
  if (w_row_ids.has_value()) {
    w_ids = int_vec_1d(*w_row_ids, "w_row_ids");
    N = w_row_ids->size(0);
    TORCH_CHECK(N == 0 || w.size(0) > 0,
                "w has 0 rows for non-empty w_row_ids");
  }
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == K,
              "x [M, K], w [W_rows, K]");
  TORCH_CHECK(M >= 0 && M <= 32,
              "the skinny kernel takes at most 32 rows, got ", M);
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  check_rows(x, "x", 8);
  check_rows(w, "w", 8);
  if (x_index.has_value()) {
    check_gpu(*x_index, "x_index", torch::kLong);
    TORCH_CHECK(x_index->size(0) >= M,
                "x_index has fewer than num_rows entries");
  } else {
    TORCH_CHECK(x.size(0) >= M, "x has fewer than num_rows rows");
  }
  const bool f32 = splits > 0;
  const int64_t nsplit = f32 ? splits : 1;
  TORCH_CHECK(K % nsplit == 0 && (K / nsplit) % 256 == 0,
              "K / splits must be a multiple of 256");
  int64_t ldo, split_stride = 0;
  if (f32) {
    check_gpu(out, "out", torch::kFloat);
    TORCH_CHECK(out.dim() == 3 && out.size(0) >= splits && out.size(1) >= M &&
                    out.size(2) == N && out.is_contiguous(),
                "out must be fp32 [splits, >= M, N]");
    ldo = N;
    split_stride = out.size(1) * N;
  } else {
    check_gpu(out, "out", torch::kHalf);
    TORCH_CHECK(out.dim() == 2 && out.size(0) >= M && out.size(1) == N &&
                    out.stride(1) == 1,
                "out must be fp16 [>= M, N]");
    ldo = out.stride(0);
  }
  if (M == 0 || N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t mt = M <= 8 ? 8 : (M <= 16 ? 16 : 32);
  int64_t nr = variant;
  if (nr == 0) nr = mt == 8 ? (N >= 4096 ? 4 : 2) : (mt == 16 ? 2 : 1);
  #define I2D_GEMV(MT, NR, F32)                                               \
    launch_gemv<MT, NR, F32>(x, x_index, w, w_ids, out, M, N, K, nsplit, ldo, \
                             split_stride, stream)
  if (mt == 8 && nr == 2) {
    f32 ? I2D_GEMV(8, 2, true) : I2D_GEMV(8, 2, false);
  } else if (mt == 8 && nr == 4) {
    f32 ? I2D_GEMV(8, 4, true) : I2D_GEMV(8, 4, false);
  } else if (mt == 16 && nr == 2) {
    f32 ? I2D_GEMV(16, 2, true) : I2D_GEMV(16, 2, false);
  } else if (mt == 16 && nr == 1) {
    f32 ? I2D_GEMV(16, 1, true) : I2D_GEMV(16, 1, false);
  } else if (mt == 32 && nr == 1) {
    f32 ? I2D_GEMV(32, 1, true) : I2D_GEMV(32, 1, false);
  } else
    TORCH_CHECK(false, "no skinny kernel for M<=", mt, " with ", nr,
                " rows per wave");
  #undef I2D_GEMV
}

void inkling2_dspark_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                          std::optional<torch::Tensor> x_index,
                          int64_t num_rows, int64_t splits, int64_t variant) {
  inkling2_dspark_gemv_v2(std::move(x), std::move(w), std::move(out),
                          std::move(x_index), num_rows, splits, variant,
                          std::nullopt);
}

// y = x . w^T on the dense RDNA2 GEMM `design` (rdna_hint_flex designs 0..5),
// fp16 in / out, fp32 accumulation, with optional zero-copy `w_row_ids` [N].
// The launch fails loudly (no rocBLAS fallback).
void inkling2_dspark_gemm_v2(torch::Tensor x, torch::Tensor w,
                             torch::Tensor out, int64_t num_rows,
                             int64_t design,
                             std::optional<torch::Tensor> w_row_ids) {
  check_gpu(x, "x", torch::kHalf);
  check_gpu(w, "w", torch::kHalf);
  check_gpu(out, "out", torch::kHalf);
  const int64_t M = num_rows, W_rows = w.size(0), K = w.size(1);
  IntVec w_ids{nullptr, 0};
  int64_t N = W_rows;
  if (w_row_ids.has_value()) {
    w_ids = int_vec_1d(*w_row_ids, "w_row_ids");
    N = w_row_ids->size(0);
    TORCH_CHECK(N == 0 || W_rows > 0, "w has 0 rows for non-empty w_row_ids");
  }
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2 &&
                  x.size(1) == K && x.size(0) >= M && out.size(0) >= M &&
                  out.size(1) == N,
              "x [>= M, K], w [W_rows, K], out [>= M, N]");
  check_rows(x, "x", 8);
  check_rows(w, "w", 8);
  check_rows(out, "out", 1);
  if (M == 0 || N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool ok =
      w_row_ids.has_value()
          ? launch_indexed_gemm(static_cast<int>(design), x.data_ptr(),
                                w.data_ptr(), w_ids, out.data_ptr(), M, N,
                                W_rows, K, x.stride(0), w.stride(0),
                                out.stride(0), stream)
          : hg::launch(static_cast<int>(design), x.data_ptr(), w.data_ptr(),
                       out.data_ptr(), M, N, K, x.stride(0), w.stride(0),
                       out.stride(0), stream);
  TORCH_CHECK(ok, "inkling2_dspark_gemm: design ", design, " cannot run M=", M,
              " N=", N, " K=", K,
              " (needs gfx1030, K a multiple of the design's K step, "
              "16-byte aligned operands)");
}

void inkling2_dspark_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                          int64_t num_rows, int64_t design) {
  inkling2_dspark_gemm_v2(std::move(x), std::move(w), std::move(out), num_rows,
                          design, std::nullopt);
}

// Per-head Q/K RMSNorm + RoPE + paged KV write (see qkv_prep_kernel).
void inkling2_dspark_qkv_prep(torch::Tensor qkv, torch::Tensor positions,
                              torch::Tensor slots, torch::Tensor q_norm_w,
                              torch::Tensor k_norm_w, torch::Tensor inv_freq,
                              std::optional<torch::Tensor> q_out,
                              torch::Tensor kv_cache, int64_t num_tokens,
                              int64_t q_heads, int64_t kv_heads, double eps) {
  check_gpu(qkv, "qkv", torch::kHalf);
  check_gpu(positions, "positions", torch::kLong);
  check_gpu(slots, "slots", torch::kLong);
  check_gpu(q_norm_w, "q_norm_w", torch::kHalf);
  check_gpu(k_norm_w, "k_norm_w", torch::kHalf);
  check_gpu(inv_freq, "inv_freq", torch::kDouble);
  check_gpu(kv_cache, "kv_cache", torch::kHalf);
  TORCH_CHECK(qkv.dim() == 2 &&
                  qkv.size(1) == (q_heads + 2 * kv_heads) * kHead &&
                  qkv.size(0) >= num_tokens,
              "qkv must be [>= T, (q_heads + 2 * kv_heads) * 128]");
  check_rows(qkv, "qkv", 8);
  TORCH_CHECK(inv_freq.numel() == kHead / 2 && inv_freq.is_contiguous(),
              "inv_freq must be float64 [64]");
  TORCH_CHECK(positions.numel() >= num_tokens && slots.numel() >= num_tokens,
              "positions / slots are shorter than num_tokens");
  TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(1) >= kv_heads &&
                  kv_cache.size(3) == 2 * kHead && kv_cache.stride(3) == 1,
              "kv_cache must be [nb, H, bs, 256]");
  __half* q_ptr = nullptr;
  long long ld_q = 0;
  if (q_heads > 0) {
    TORCH_CHECK(q_out.has_value(), "q_out is required with q heads");
    check_gpu(*q_out, "q_out", torch::kHalf);
    TORCH_CHECK(q_out->dim() == 2 && q_out->size(1) == q_heads * kHead &&
                    q_out->size(0) >= num_tokens && q_out->stride(1) == 1,
                "q_out must be [>= T, q_heads * 128]");
    q_ptr = reinterpret_cast<__half*>(q_out->data_ptr());
    ld_q = q_out->stride(0);
  }
  if (num_tokens == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_prep_kernel<<<dim3(static_cast<unsigned>(num_tokens)), dim3(256), 0,
                    stream>>>(
      reinterpret_cast<const __half*>(qkv.data_ptr()), qkv.stride(0),
      positions.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(q_norm_w.data_ptr()),
      reinterpret_cast<const __half*>(k_norm_w.data_ptr()),
      inv_freq.data_ptr<double>(), q_ptr, ld_q,
      reinterpret_cast<__half*>(kv_cache.data_ptr()), kv_cache.stride(0),
      kv_cache.stride(1), kv_cache.stride(2), (int)kv_cache.size(2),
      (int)q_heads, (int)kv_heads, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Bidirectional block attention of 8-row blocks over the paged cache.
// q [T, 3072] -> out [T, 3072] (T = 8 * num_reqs); `part` is the fp32 split
// scratch
// [>= num_reqs * kv_heads * splits * 24 * 130].
void inkling2_dspark_attention(torch::Tensor q, torch::Tensor kv_cache,
                               torch::Tensor block_table,
                               torch::Tensor seq_lens, torch::Tensor out,
                               torch::Tensor part, int64_t num_reqs,
                               int64_t window, int64_t splits) {
  check_gpu(q, "q", torch::kHalf);
  check_gpu(kv_cache, "kv_cache", torch::kHalf);
  check_gpu(block_table, "block_table", torch::kInt);
  check_gpu(seq_lens, "seq_lens", torch::kInt);
  check_gpu(out, "out", torch::kHalf);
  check_gpu(part, "part", torch::kFloat);
  const int64_t kv_heads = kv_cache.size(1);
  TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(3) == 2 * kHead &&
                  kv_cache.stride(3) == 1,
              "kv_cache must be [nb, H, bs, 256]");
  TORCH_CHECK(q.dim() == 2 && q.size(1) == kv_heads * kQPerKv * kHead &&
                  q.size(0) >= num_reqs * kBlockRows && q.stride(1) == 1 &&
                  q.stride(0) % 8 == 0,
              "q must be [>= 8 * num_reqs, 24 * 128]");
  TORCH_CHECK(out.dim() == 2 && out.size(1) == q.size(1) &&
                  out.size(0) >= num_reqs * kBlockRows && out.stride(1) == 1,
              "out must match q");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= num_reqs &&
                  block_table.stride(1) == 1 && seq_lens.numel() >= num_reqs,
              "block_table [>= R, max_blocks] int32, seq_lens [>= R] int32");
  TORCH_CHECK(splits >= 1 && window >= 0, "splits >= 1, window >= 0");
  TORCH_CHECK(
      part.is_contiguous() &&
          part.numel() >= num_reqs * kv_heads * splits * kNumQ * kPartStride,
      "part scratch too small");
  if (num_reqs == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const float scale_log2 =
      1.4426950408889634f / sqrtf(static_cast<float>(kHead));
  attn_split_kernel<<<dim3((unsigned)splits, (unsigned)kv_heads,
                           (unsigned)num_reqs),
                      dim3(256), 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), q.stride(0),
      reinterpret_cast<const __half*>(kv_cache.data_ptr()), kv_cache.stride(0),
      kv_cache.stride(1), kv_cache.stride(2), (int)kv_cache.size(2),
      block_table.data_ptr<int>(), block_table.stride(0),
      seq_lens.data_ptr<int>(), part.data_ptr<float>(), (int)kv_heads,
      (int)window, (int)splits, scale_log2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  attn_combine_kernel<<<dim3((unsigned)(num_reqs * kv_heads * kNumQ)),
                        dim3(kHead), 0, stream>>>(
      part.data_ptr<float>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr()), out.stride(0), (int)kv_heads,
      (int)window, (int)splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One Markov recurrence step of `num_reqs` requests (see markov_step_kernel).
void inkling2_dspark_markov_step(torch::Tensor prev, int64_t prev_stride,
                                 torch::Tensor state, torch::Tensor w1,
                                 torch::Tensor wt, torch::Tensor bias,
                                 torch::Tensor ctx_h, torch::Tensor ctx_f,
                                 torch::Tensor latent, int64_t num_reqs,
                                 int64_t step, int64_t nsteps, bool first) {
  check_gpu(state, "state", torch::kFloat);
  check_gpu(w1, "w1", torch::kHalf);
  check_gpu(wt, "wt", torch::kHalf);
  check_gpu(bias, "bias", torch::kFloat);
  check_gpu(ctx_h, "ctx_h", torch::kFloat);
  check_gpu(ctx_f, "ctx_f", torch::kFloat);
  check_gpu(latent, "latent", torch::kHalf);
  const bool wide = prev.scalar_type() == torch::kLong;
  TORCH_CHECK(wide || prev.scalar_type() == torch::kInt,
              "prev must be int32 / int64");
  TORCH_CHECK(w1.dim() == 2 && w1.size(1) == kRank && w1.is_contiguous(),
              "w1 must be [vocab, 256]");
  TORCH_CHECK(wt.dim() == 2 && wt.size(0) == 2 * kRank &&
                  wt.size(1) == kGateWidth && wt.is_contiguous(),
              "wt must be [512, 768]");
  TORCH_CHECK(bias.numel() == kGateWidth && ctx_h.size(-1) == kGateWidth &&
                  ctx_f.size(-1) == kGateWidth,
              "bias / ctx widths must be 768");
  TORCH_CHECK(
      state.size(0) >= num_reqs && latent.size(0) >= num_reqs &&
          ctx_f.size(0) >= num_reqs && ctx_h.size(0) >= num_reqs * nsteps &&
          state.is_contiguous() && latent.is_contiguous() &&
          ctx_h.is_contiguous() && ctx_f.is_contiguous(),
      "state / latent / ctx_f need num_reqs rows, ctx_h num_reqs * nsteps");
  TORCH_CHECK(step >= 0 && step < nsteps, "step out of range");
  if (num_reqs == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(state));
  auto stream = at::cuda::getCurrentCUDAStream();
  markov_step_kernel<<<dim3(static_cast<unsigned>(num_reqs)), dim3(256), 0,
                       stream>>>(
      prev.data_ptr(), wide, prev_stride, state.data_ptr<float>(),
      reinterpret_cast<const __half*>(w1.data_ptr()), w1.size(0),
      reinterpret_cast<const __half*>(wt.data_ptr()), bias.data_ptr<float>(),
      ctx_h.data_ptr<float>(), ctx_f.data_ptr<float>(),
      reinterpret_cast<__half*>(latent.data_ptr()), (int)step, (int)nsteps,
      first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// logits[r, :] = cap * tanh((base[r * nsteps + step, :] + fold_b + fold_w .
// latent[r]) / cap)
void inkling2_dspark_head_logits(torch::Tensor base, torch::Tensor latent,
                                 torch::Tensor fold_w, torch::Tensor fold_b,
                                 torch::Tensor logits, int64_t num_reqs,
                                 int64_t step, int64_t nsteps, double softcap) {
  check_gpu(base, "base", torch::kHalf);
  check_gpu(latent, "latent", torch::kHalf);
  check_gpu(fold_w, "fold_w", torch::kHalf);
  check_gpu(fold_b, "fold_b", torch::kFloat);
  check_gpu(logits, "logits", torch::kHalf);
  const int64_t V = fold_w.size(0);
  TORCH_CHECK(fold_w.dim() == 2 && fold_w.size(1) == kRank &&
                  fold_w.is_contiguous() && fold_b.numel() >= V &&
                  latent.is_contiguous() && latent.size(0) >= num_reqs,
              "fold_w [V, 256], fold_b [V], latent [>= R, 256]");
  TORCH_CHECK(base.dim() == 2 && base.size(1) >= V &&
                  base.size(0) >= num_reqs * nsteps && base.stride(1) == 1,
              "base must be [>= R * nsteps, >= V]");
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) >= V &&
                  logits.size(0) >= num_reqs && logits.stride(1) == 1,
              "logits must be [>= R, >= V]");
  TORCH_CHECK(num_reqs >= 0 && num_reqs <= 32,
              "at most 32 requests per head launch");
  TORCH_CHECK(step >= 0 && step < nsteps, "step out of range");
  if (num_reqs == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(base));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(cdiv(V, 32)));
  #define I2D_HEAD(RP)                                                    \
    head_logits_kernel<RP><<<grid, dim3(256), 0, stream>>>(               \
        reinterpret_cast<const __half*>(base.data_ptr()), base.stride(0), \
        (int)step, (int)nsteps,                                           \
        reinterpret_cast<const __half*>(latent.data_ptr()),               \
        reinterpret_cast<const __half*>(fold_w.data_ptr()),               \
        fold_b.data_ptr<float>(),                                         \
        reinterpret_cast<__half*>(logits.data_ptr()), logits.stride(0),   \
        (int)num_reqs, (int)V, (float)softcap)
  if (num_reqs <= 1) {
    I2D_HEAD(1);
  } else if (num_reqs <= 2) {
    I2D_HEAD(2);
  } else if (num_reqs <= 4) {
    I2D_HEAD(4);
  } else if (num_reqs <= 8) {
    I2D_HEAD(8);
  } else if (num_reqs <= 16) {
    I2D_HEAD(16);
  } else {
    I2D_HEAD(32);
  }
  #undef I2D_HEAD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// fold_b[v] = head[v, :] . b  (fp32), the bias of the folded hidden_out
// projection.
void inkling2_dspark_fold_bias(torch::Tensor head, torch::Tensor b,
                               torch::Tensor out) {
  check_gpu(head, "head", torch::kHalf);
  check_gpu(b, "b", torch::kHalf);
  check_gpu(out, "out", torch::kFloat);
  const int64_t V = head.size(0), H = head.size(1);
  check_rows(head, "head", 8);
  TORCH_CHECK(H % 256 == 0 && b.numel() == H && b.is_contiguous() &&
                  out.numel() >= V &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0,
              "head [V, H], b [H] (H a multiple of 256), out [>= V]");
  const at::cuda::OptionalCUDAGuard guard(device_of(head));
  auto stream = at::cuda::getCurrentCUDAStream();
  fold_bias_kernel<<<dim3(static_cast<unsigned>(cdiv(V, 8))), dim3(256), 0,
                     stream>>>(reinterpret_cast<const __half*>(head.data_ptr()),
                               head.stride(0),
                               reinterpret_cast<const __half*>(b.data_ptr()),
                               out.data_ptr<float>(), (int)V, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[r] = argmax of logits[r, :vocab] (fp16 rows, NaN as -inf, lowest index on
// ties), optionally remapped through `token_map` (int32/int64 [>= vocab]).
// `out` is an int64 vector or [rows, 1] strided column of a token table.
void inkling2_dspark_argmax_v2(torch::Tensor logits, torch::Tensor out,
                               int64_t num_rows, int64_t vocab,
                               std::optional<torch::Tensor> token_map) {
  check_gpu(logits, "logits", torch::kHalf);
  check_gpu(out, "out", torch::kLong);
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= num_rows &&
                  logits.size(1) >= vocab && logits.stride(1) == 1 &&
                  vocab > 0 && vocab < (int64_t{1} << 31),
              "logits must be [>= rows, >= vocab] with a unit column stride");
  TORCH_CHECK((out.dim() == 1 || (out.dim() == 2 && out.size(1) == 1)) &&
                  out.size(0) >= num_rows,
              "out must be an int64 vector (or [rows, 1] column) of at least "
              "num_rows entries");
  IntVec tmap{nullptr, 0};
  if (token_map.has_value()) {
    tmap = int_vec_1d(*token_map, "token_map");
    TORCH_CHECK(token_map->size(0) >= vocab,
                "token_map has fewer than vocab entries");
  }
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vec = logits.stride(0) % 8 == 0 &&
                   reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0;
  argmax_rows_kernel<<<dim3(static_cast<unsigned>(num_rows)), dim3(512), 0,
                       stream>>>(
      reinterpret_cast<const __half*>(logits.data_ptr()), logits.stride(0),
      (int)vocab, vec, static_cast<long long*>(out.data_ptr()), out.stride(0),
      tmap, nullptr, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_dspark_argmax(torch::Tensor logits, torch::Tensor out,
                            int64_t num_rows, int64_t vocab) {
  inkling2_dspark_argmax_v2(std::move(logits), std::move(out), num_rows, vocab,
                            std::nullopt);
}

// The (value, id) partial of one vocabulary shard: part[r] = max logits[r,
// :vocab] (fp32), part[num_rows + r] = the int32 bits of (token_map ?
// token_map[bi] : first_id + bi) for the lowest index `bi` of that maximum.
void inkling2_dspark_argmax_partial_v2(torch::Tensor logits, torch::Tensor part,
                                       int64_t num_rows, int64_t vocab,
                                       int64_t first_id,
                                       std::optional<torch::Tensor> token_map) {
  check_gpu(logits, "logits", torch::kHalf);
  check_gpu(part, "part", torch::kFloat);
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= num_rows &&
                  logits.size(1) >= vocab && logits.stride(1) == 1 &&
                  vocab > 0 && vocab < (int64_t{1} << 31),
              "logits must be [>= rows, >= vocab] with a unit column stride");
  IntVec tmap{nullptr, 0};
  if (token_map.has_value()) {
    tmap = int_vec_1d(*token_map, "token_map");
    TORCH_CHECK(token_map->size(0) >= vocab,
                "token_map has fewer than vocab entries");
  } else {
    TORCH_CHECK(first_id >= 0 && first_id + vocab < (int64_t{1} << 31),
                "first_id + vocab must fit an int32");
  }
  TORCH_CHECK(
      part.is_contiguous() && part.numel() >= 2 * num_rows,
      "part must be a contiguous fp32 tensor of >= 2 * num_rows entries");
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vec = logits.stride(0) % 8 == 0 &&
                   reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0;
  argmax_rows_kernel<<<dim3(static_cast<unsigned>(num_rows)), dim3(512), 0,
                       stream>>>(
      reinterpret_cast<const __half*>(logits.data_ptr()), logits.stride(0),
      (int)vocab, vec, nullptr, 0, tmap, part.data_ptr<float>(), (int)num_rows,
      (int)first_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_dspark_argmax_partial(torch::Tensor logits, torch::Tensor part,
                                    int64_t num_rows, int64_t vocab,
                                    int64_t first_id) {
  inkling2_dspark_argmax_partial_v2(std::move(logits), std::move(part),
                                    num_rows, vocab, first_id, std::nullopt);
}

namespace {

inline IntVec int_vec(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name,
              " must be a contiguous GPU tensor");
  TORCH_CHECK(t.scalar_type() == torch::kInt || t.scalar_type() == torch::kLong,
              name, " must be int32 or int64");
  return IntVec{t.data_ptr(), t.scalar_type() == torch::kLong ? 1 : 0};
}

inline IntOut int_out(torch::Tensor& t, const char* name, int64_t min_numel) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.numel() >= min_numel, name,
              " must be a contiguous GPU tensor of at least ", min_numel,
              " entries");
  TORCH_CHECK(t.scalar_type() == torch::kInt || t.scalar_type() == torch::kLong,
              name, " must be int32 or int64");
  return IntOut{t.data_ptr(), t.scalar_type() == torch::kLong ? 1 : 0};
}

inline void check_vec(const torch::Tensor& t, const char* name,
                      c10::ScalarType type, int64_t min_numel) {
  check_gpu(t, name, type);
  TORCH_CHECK(t.is_contiguous() && t.numel() >= min_numel, name,
              " must be a contiguous tensor of at least ", min_numel,
              " entries");
}

}  // namespace

// The HIP port of the runner's DFlash input preparation for a single-rank
// context
// (``prepare_dflash_inputs``): the query block, context positions / slots and
// sample rows of every request from the target batch, plus the padding of all
// buffers up to their maximum.
void inkling2_dspark_prepare_inputs(
    torch::Tensor input_ids, torch::Tensor positions, torch::Tensor is_padding,
    torch::Tensor query_start_loc, torch::Tensor seq_lens,
    torch::Tensor query_slot_mapping, torch::Tensor context_positions,
    torch::Tensor context_slot_mapping, torch::Tensor sample_indices,
    torch::Tensor sample_pos, torch::Tensor sample_idx_mapping,
    torch::Tensor out_temperature, torch::Tensor out_seeds,
    torch::Tensor target_positions, torch::Tensor target_query_start_loc,
    torch::Tensor idx_mapping, torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
    torch::Tensor num_rejected, torch::Tensor temperature, torch::Tensor seeds,
    torch::Tensor block_table, int64_t block_size,
    int64_t parallel_drafting_token_id, int64_t num_query_per_req,
    int64_t num_speculative_steps, int64_t max_num_reqs, int64_t max_num_tokens,
    int64_t max_model_len, int64_t num_reqs, int64_t max_tokens_per_req,
    bool sample_from_anchor) {
  TORCH_CHECK(num_reqs > 0 && num_reqs <= max_num_reqs,
              "0 < num_reqs <= max_num_reqs");
  TORCH_CHECK(block_size > 0 && num_query_per_req > 0 &&
                  num_speculative_steps > 0 && max_model_len > 0 &&
                  max_tokens_per_req > 0,
              "block_size, query width, steps, max_model_len and the span must "
              "be positive");
  TORCH_CHECK(
      num_query_per_req == num_speculative_steps + (sample_from_anchor ? 0 : 1),
      "a block holds one sample row per step: num_query_per_req must be steps "
      "(sample_from_anchor) or steps + 1");
  TORCH_CHECK(num_reqs * num_query_per_req <= max_num_tokens,
              "the query blocks do not fit max_num_tokens");
  check_vec(input_ids, "input_ids", torch::kInt, max_num_tokens);
  check_vec(positions, "positions", torch::kLong, max_num_tokens);
  check_vec(is_padding, "is_padding", torch::kBool, max_num_tokens);
  check_vec(query_start_loc, "query_start_loc", torch::kInt, max_num_reqs + 1);
  check_vec(seq_lens, "seq_lens", torch::kInt, max_num_reqs);
  check_vec(query_slot_mapping, "query_slot_mapping", torch::kLong,
            max_num_tokens);
  check_vec(context_positions, "context_positions", torch::kLong,
            max_num_tokens);
  check_vec(context_slot_mapping, "context_slot_mapping", torch::kLong,
            max_num_tokens);
  const int64_t sample_rows = max_num_reqs * num_speculative_steps;
  check_vec(sample_indices, "sample_indices", torch::kLong, sample_rows);
  check_vec(sample_pos, "sample_pos", torch::kLong, sample_rows);
  const IntOut sample_idx_out =
      int_out(sample_idx_mapping, "sample_idx_mapping", sample_rows);
  check_vec(out_temperature, "out_temperature", torch::kFloat, 1);
  check_vec(out_seeds, "out_seeds", torch::kLong, 1);
  check_vec(target_positions, "target_positions", torch::kLong, 1);
  check_vec(target_query_start_loc, "target_query_start_loc", torch::kInt,
            num_reqs + 1);
  check_vec(last_sampled, "last_sampled", torch::kLong, 1);
  check_vec(temperature, "temperature", torch::kFloat, 1);
  check_vec(seeds, "seeds", torch::kLong, 1);
  check_gpu(block_table, "block_table", torch::kInt);
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.stride(1) == 1 &&
          block_table.size(0) >= num_reqs,
      "block_table must be [>= num_reqs, blocks] with a unit column stride");
  PrepareArgs a;
  a.input_ids = input_ids.data_ptr<int>();
  a.query_positions = static_cast<long long*>(positions.data_ptr());
  a.is_padding = reinterpret_cast<bool*>(is_padding.data_ptr());
  a.query_start_loc = query_start_loc.data_ptr<int>();
  a.seq_lens = seq_lens.data_ptr<int>();
  a.query_slot = static_cast<long long*>(query_slot_mapping.data_ptr());
  a.ctx_positions = static_cast<long long*>(context_positions.data_ptr());
  a.ctx_slot = static_cast<long long*>(context_slot_mapping.data_ptr());
  a.sample_indices = static_cast<long long*>(sample_indices.data_ptr());
  a.sample_pos = static_cast<long long*>(sample_pos.data_ptr());
  a.sample_idx_mapping = sample_idx_out;
  a.out_temperature = out_temperature.data_ptr<float>();
  a.out_seeds = static_cast<long long*>(out_seeds.data_ptr());
  a.target_positions =
      static_cast<const long long*>(target_positions.data_ptr());
  a.target_qsl = target_query_start_loc.data_ptr<int>();
  a.idx_mapping = int_vec(idx_mapping, "idx_mapping");
  a.last_sampled = static_cast<const long long*>(last_sampled.data_ptr());
  a.next_prefill = int_vec(next_prefill_tokens, "next_prefill_tokens");
  a.num_sampled = int_vec(num_sampled, "num_sampled");
  a.num_rejected = int_vec(num_rejected, "num_rejected");
  a.temperature = temperature.data_ptr<float>();
  a.seeds = static_cast<const long long*>(seeds.data_ptr());
  a.block_table = block_table.data_ptr<int>();
  a.block_table_stride = block_table.stride(0);
  a.block_size = static_cast<int>(block_size);
  a.parallel_token = static_cast<int>(parallel_drafting_token_id);
  a.nq = static_cast<int>(num_query_per_req);
  a.steps = static_cast<int>(num_speculative_steps);
  a.max_reqs = static_cast<int>(max_num_reqs);
  a.max_tokens = static_cast<int>(max_num_tokens);
  a.max_model_len = max_model_len;
  a.num_reqs = static_cast<int>(num_reqs);
  a.sample_from_anchor = sample_from_anchor;
  const at::cuda::OptionalCUDAGuard guard(device_of(input_ids));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(num_reqs + kPadPrograms),
                  static_cast<unsigned>(cdiv(max_tokens_per_req, kPrepBlock)));
  prepare_inputs_kernel<<<grid, dim3(kPrepBlock), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif  // !__HIP_DEVICE_COMPILE__ || __HIP__RDNA2__
