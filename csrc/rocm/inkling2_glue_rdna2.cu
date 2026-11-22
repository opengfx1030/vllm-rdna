// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Inkling-2-Mini glue kernels for RDNA2 (gfx1030, wave32).
//
// Everything of a layer visit that is not a GEMM, an attention op, a short
// convolution or an expert GEMM, fused into a handful of CUDA-graph-capturable
// launches.  The math is exactly that of
// vllm/models/inkling2_loop/amd/blocks.py and moe.py (fp32 residual stream and
// block states, fp16 activations, fp32 accumulation); only the order of fp32
// reductions differs.
//
//   inkling2_glue_attn_res        Block/MessageAttentionResidual: unweighted
//                                 RMS norm of the block states, learned-query
//                                 softmax attention over them, optional fused
//                                 (x0 + x1) partial state, block-state append
//                                 and per-query weighted RMSNorm -> fp16.
//   inkling2_glue_msg_gate_down   MessageGate step 1: both unweighted norms
//                                 and the fp16 down projection (V_DOT2, fp16
//                                 result).
//   inkling2_glue_msg_gate_mix    MessageGate step 2: forget / gain
//                                 projections, tanh, (1 - f) * h + g * m and
//                                 the fused RMSNorm -> fp16 (cross-CTA
//                                 reduction by a last-arriver CTA).
//   inkling2_glue_rms_norm        Inkling2RMSNorm / unweighted RMS norm with
//                                 an optional fp32 add and the fp16 cast.
//   inkling2_glue_embed_norm      embedding gather + embed_norm (+ fp16
//                                 rounding).
//   inkling2_glue_moe_route       router logits GEMV, sigmoid, biased top-k,
//                                 normalisation, route scale AND the
//                                 moe_align_block_size layout (+ zero fills)
//                                 in ONE launch.
//   inkling2_glue_swiglu          softcapped SwiGLU between the two expert
//                                 GEMMs.
//   inkling2_glue_copy_states     fp32 [T, (P,) H] snapshot copy (persistent
//                                 block states that a later layer overwrites).
//
// Not here (already fused by the other Inkling-2 kernel files): the q / k / rel
// scaled L2 clip, log scale, v norm and the sigmoid output gate
// (inkling2_attention / inkling2_kv_prep), the router-only GEMV and the lm_head
// softcap (inkling2_linear_*).
//
// Device code is gfx1030-only (V_DOT2_F32_F16); the kernels of the other
// architectures are empty bodies with the same signature (same namespace), so
// the translation unit compiles for every arch of a multi-arch build.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <torch/all.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace inkling2_glue {

constexpr int kWave = 32;
constexpr int kMaxStates = 64;    // block states (+ append + partial) per call
constexpr int kMaxExperts = 255;  // expert id is a byte, 0xFF is the pad
constexpr int kMaxTopK = 16;
constexpr int kRouteThreads = 256;      // 8 waves, one expert row per wave
constexpr int kRouteMaxEntries = 8192;  // tokens * top_k of one call
// align scratch in the router's dynamic LDS: per-wave expert histograms
// [waves][256], the wave sums of the expert scan, the routed ids as bytes
constexpr int kAlignOffWs = (kRouteThreads / kWave) * 256 * 4;
constexpr int kAlignOffKeys = kAlignOffWs + 16 * 4;
constexpr int kAlignSmemBytes = kAlignOffKeys + kRouteMaxEntries;
constexpr int kGateWaves = 4;  // output rows of the down projection per CTA
constexpr int kMinCtas = 216;  // CTAs kept in flight (3 per CU of a 72 CU V620)
constexpr int kMaxZeroCtas = 32;  // router zero-fill CTAs (each reserves LDS)
constexpr int kStageV = 32;       // float4 per lane of a staged row: H <= 4096

// ---------------------------------------------------------------------------
// Small helpers (arch independent)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

// All threads of the block get the sum (same order in every thread).
// `scratch` holds one float per wave.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  v = wave_sum(v);
  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int nw = blockDim.x / kWave;
  __syncthreads();
  if (lane == 0) scratch[wave] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < nw; ++w) t += scratch[w];
  return t;
}

__device__ __forceinline__ float dot4(const float4 a, const float4 b) {
  return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, a.x * b.x)));
}

__device__ __forceinline__ float4 fma4(float c, const float4 a, float4 acc) {
  acc.x = fmaf(c, a.x, acc.x);
  acc.y = fmaf(c, a.y, acc.y);
  acc.z = fmaf(c, a.z, acc.z);
  acc.w = fmaf(c, a.w, acc.w);
  return acc;
}

__device__ __forceinline__ float h2f_lo(uint32_t u) {
  return __half2float(__ushort_as_half((unsigned short)(u & 0xFFFFu)));
}
__device__ __forceinline__ float h2f_hi(uint32_t u) {
  return __half2float(__ushort_as_half((unsigned short)(u >> 16)));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  return (uint32_t)__half_as_ushort(__float2half_rn(lo)) |
         ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}
__device__ __forceinline__ float round_h(float x) {
  return __half2float(__float2half_rn(x));
}

__device__ __forceinline__ float4 load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 load4(const __half* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(h2f_lo(u.x), h2f_hi(u.x), h2f_lo(u.y), h2f_hi(u.y));
}
__device__ __forceinline__ void store4(float* p, const float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}
__device__ __forceinline__ void store4(__half* p, const float4 v) {
  uint2 u;
  u.x = pack_h2(v.x, v.y);
  u.y = pack_h2(v.z, v.w);
  *reinterpret_cast<uint2*>(p) = u;
}

// Element type tag selected by a uniform runtime flag.
__device__ __forceinline__ float4 load4_dyn(const void* base, int64_t idx,
                                            int is_half) {
  return is_half ? load4(reinterpret_cast<const __half*>(base) + idx)
                 : load4(reinterpret_cast<const float*>(base) + idx);
}

__device__ __forceinline__ float sigmoidf_(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// attn_res
// ---------------------------------------------------------------------------
struct AttnResArgs {
  float* blocks;  // [T, S, H] fp32; the append slot is written here
  int64_t bs_t, bs_s;
  const float* x0;  // partial state = x0 (+ x1)
  int64_t x0_t;
  const float* x1;
  int64_t x1_t;
  float* partial_out;  // receives the partial state
  int64_t po_t;
  const float* append;  // state `n_blocks` (also copied into blocks)
  int64_t ap_t;
  const float* query;  // [Q, H] fp32
  int64_t q_s;
  const void* norm_w;  // [1 or Q, H] fp16 / fp32 (null: fp32 output)
  int64_t nw_s;
  int nw_f16;
  int nw_rows;
  void* out;  // [T, Q, H] fp32 (or fp16 with norm_w)
  int64_t o_t, o_q;
  int H, n_blocks;
  float eps, norm_eps, inv_h, scale;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int Q, int ITER>
__device__ __forceinline__ void state_partials(const float4 (&s)[ITER],
                                               const float4 (&qv)[Q][ITER],
                                               float (&v)[Q + 1]) {
  #pragma unroll
  for (int i = 0; i < Q + 1; ++i) v[i] = 0.f;
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    v[0] += dot4(s[k], s[k]);
  #pragma unroll
    for (int q = 0; q < Q; ++q) v[1 + q] += dot4(qv[q][k], s[k]);
  }
}

template <int Q, int ITER, bool NORM>
__device__ __forceinline__ void attn_res_body(const AttnResArgs& a) {
  constexpr int kNReg = 6;  // states kept in registers between the two passes
  constexpr int kQ1 = Q + 1;
  const int tid = threadIdx.x;
  const int nth = blockDim.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int nwaves = nth / kWave;
  const int64_t t = blockIdx.x;
  const int V = a.H >> 2;

  __shared__ float s_red[8][kMaxStates * kQ1];
  __shared__ float s_tot[kMaxStates * kQ1];
  __shared__ float s_coef[Q][kMaxStates];

  const bool has_x1 = a.x1 != nullptr;
  const bool has_app = a.append != nullptr;
  const int NS = a.n_blocks + (has_app ? 1 : 0) + 1;
  float* brow = a.blocks != nullptr ? a.blocks + t * a.bs_t : nullptr;

  // partial state (last of the stack) and appended block state
  float4 P[ITER], A[ITER], Qv[Q][ITER];
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
    float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 ap = p;
    if (v < V) {
      p = load4(a.x0 + t * a.x0_t + 4 * v);
      if (has_x1) {
        const float4 q1 = load4(a.x1 + t * a.x1_t + 4 * v);
        p.x += q1.x;
        p.y += q1.y;
        p.z += q1.z;
        p.w += q1.w;
      }
      if (a.partial_out != nullptr)
        store4(a.partial_out + t * a.po_t + 4 * v, p);
      if (has_app) {
        ap = load4(a.append + t * a.ap_t + 4 * v);
        store4(brow + (int64_t)a.n_blocks * a.bs_s + 4 * v, ap);
      }
    }
    P[k] = p;
    A[k] = ap;
  #pragma unroll
    for (int q = 0; q < Q; ++q)
      Qv[q][k] = v < V ? load4(a.query + (int64_t)q * a.q_s + 4 * v)
                       : make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // state j: blocks[j] (j < n_blocks), then the appended state, then partial
  auto fetch = [&](int j, float4(&s)[ITER]) {
  #pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int v = tid + k * nth;
      if (j < a.n_blocks)
        s[k] = v < V ? load4(brow + (int64_t)j * a.bs_s + 4 * v)
                     : make_float4(0.f, 0.f, 0.f, 0.f);
      else if (has_app && j == a.n_blocks)
        s[k] = A[k];
      else
        s[k] = P[k];
    }
  };

  // pass 1: per state sum of squares and the dot product with every query
  float4 S[kNReg][ITER];
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
    if (j < NS) {
      fetch(j, S[j]);
      float v[kQ1];
      state_partials<Q, ITER>(S[j], Qv, v);
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) v[i] = wave_sum(v[i]);
      if (lane == 0) {
  #pragma unroll
        for (int i = 0; i < kQ1; ++i) s_red[wave][j * kQ1 + i] = v[i];
      }
    }
  }
  for (int j = kNReg; j < NS; ++j) {
    float4 s[ITER];
    fetch(j, s);
    float v[kQ1];
    state_partials<Q, ITER>(s, Qv, v);
  #pragma unroll
    for (int i = 0; i < kQ1; ++i) v[i] = wave_sum(v[i]);
    if (lane == 0) {
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) s_red[wave][j * kQ1 + i] = v[i];
    }
  }
  __syncthreads();
  for (int i = tid; i < NS * kQ1; i += nth) {
    float acc = 0.f;
    for (int w = 0; w < nwaves; ++w) acc += s_red[w][i];
    s_tot[i] = acc;
  }
  __syncthreads();

  // scores -> softmax over the states -> coefficients c[q][j] = p[q][j] * r[j]
  if (wave == 0) {
    float rr[2];
  #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = lane + kWave * u;
      rr[u] = j < NS ? rsqrtf(s_tot[j * kQ1] * a.inv_h + a.eps) : 0.f;
    }
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      float sc[2], e[2];
  #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = lane + kWave * u;
        sc[u] = j < NS ? (s_tot[j * kQ1 + 1 + q] * rr[u]) * a.scale : -INFINITY;
      }
      const float m = wave_max(fmaxf(sc[0], sc[1]));
  #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = lane + kWave * u;
        e[u] = j < NS ? expf(sc[u] - m) : 0.f;
      }
      const float den = wave_sum(e[0] + e[1]);
  #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = lane + kWave * u;
        if (j < NS) s_coef[q][j] = (e[u] / den) * rr[u];
      }
    }
  }
  __syncthreads();

  // pass 2: out[q] = sum_j c[q][j] * state_j
  float4 acc[Q][ITER];
  #pragma unroll
  for (int q = 0; q < Q; ++q)
  #pragma unroll
    for (int k = 0; k < ITER; ++k) acc[q][k] = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
    if (j < NS) {
  #pragma unroll
      for (int q = 0; q < Q; ++q) {
        const float c = s_coef[q][j];
  #pragma unroll
        for (int k = 0; k < ITER; ++k) acc[q][k] = fma4(c, S[j][k], acc[q][k]);
      }
    }
  }
  for (int j = kNReg; j < NS; ++j) {
    float4 s[ITER];
    fetch(j, s);
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      const float c = s_coef[q][j];
  #pragma unroll
      for (int k = 0; k < ITER; ++k) acc[q][k] = fma4(c, s[k], acc[q][k]);
    }
  }

  if constexpr (!NORM) {
    float* out = reinterpret_cast<float*>(a.out) + t * a.o_t;
  #pragma unroll
    for (int q = 0; q < Q; ++q)
  #pragma unroll
      for (int k = 0; k < ITER; ++k) {
        const int v = tid + k * nth;
        if (v < V) store4(out + (int64_t)q * a.o_q + 4 * v, acc[q][k]);
      }
  } else {
    __shared__ float s_nrm[8][Q];
    float ssq[Q];
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      float s = 0.f;
  #pragma unroll
      for (int k = 0; k < ITER; ++k) s += dot4(acc[q][k], acc[q][k]);
      ssq[q] = wave_sum(s);
    }
    __syncthreads();
    if (lane == 0) {
  #pragma unroll
      for (int q = 0; q < Q; ++q) s_nrm[wave][q] = ssq[q];
    }
    __syncthreads();
    float rs[Q];
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      float tot = 0.f;
      for (int w = 0; w < nwaves; ++w) tot += s_nrm[w][q];
      rs[q] = rsqrtf(tot * a.inv_h + a.norm_eps);
    }
    __half* out = reinterpret_cast<__half*>(a.out) + t * a.o_t;
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t wrow = (a.nw_rows == 1 ? 0 : q) * a.nw_s;
  #pragma unroll
      for (int k = 0; k < ITER; ++k) {
        const int v = tid + k * nth;
        if (v < V) {
          const float4 w = load4_dyn(a.norm_w, wrow + 4 * v, a.nw_f16);
          const float4 y = make_float4(
              w.x * (acc[q][k].x * rs[q]), w.y * (acc[q][k].y * rs[q]),
              w.z * (acc[q][k].z * rs[q]), w.w * (acc[q][k].w * rs[q]));
          store4(out + (int64_t)q * a.o_q + 4 * v, y);
        }
      }
    }
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int Q, int ITER, bool NORM>
__global__ void __launch_bounds__(256) attn_res_kernel(const AttnResArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  attn_res_body<Q, ITER, NORM>(a);
#endif
}

// ---------------------------------------------------------------------------
// MessageGate, step 1: norms + down projection
// ---------------------------------------------------------------------------
struct GateDownArgs {
  const float* h;  // [T, SH, H]; h_s == 0 broadcasts one h to every gate
  int64_t h_t, h_s;
  const float* m;  // [T, G, H]
  int64_t m_t, m_s;
  const __half* wd;  // [G, R, 2H]
  __half* d;         // [G, T, R]  (fp16-rounded projection)
  int T, R, H;
  float eps, inv_h;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// dst = fp16(src * rsqrt(mean(src^2) + eps)) of one fp32 row, by one wave.  All
// the loads are issued before the first use: the row pays one memory latency
// instead of one per pass and per group of 32 float4.
__device__ __forceinline__ void stage_unit_row(const float* src, __half* dst,
                                               int V, float inv_h, float eps,
                                               int lane) {
  float4 buf[kStageV];
  #pragma unroll
  for (int i = 0; i < kStageV; ++i) {
    const int c4 = lane + kWave * i;
    buf[i] = c4 < V ? load4(src + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float ss = 0.f;
  #pragma unroll
  for (int i = 0; i < kStageV; ++i) ss += dot4(buf[i], buf[i]);
  ss = wave_sum(ss);
  const float rs = rsqrtf(ss * inv_h + eps);
  #pragma unroll
  for (int i = 0; i < kStageV; ++i) {
    const int c4 = lane + kWave * i;
    if (c4 < V) {
      uint2 u;
      u.x = pack_h2(buf[i].x * rs, buf[i].y * rs);
      u.y = pack_h2(buf[i].z * rs, buf[i].w * rs);
      *reinterpret_cast<uint2*>(dst + 4 * c4) = u;
    }
  }
}

template <int TB>
__device__ __forceinline__ void gate_down_body(const GateDownArgs& a) {
  extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);  // [TB][2H]
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  constexpr int nwaves = kGateWaves;
  const int s = blockIdx.y;
  const int t0 = blockIdx.z * TB;
  const int H = a.H, V = H >> 2, K2 = 2 * H;

  // stage x = fp16([norm(h), norm(m)]) of the TB tokens
  for (int vi = wave; vi < 2 * TB; vi += nwaves) {
    const int tb = vi >> 1, which = vi & 1;
    const int t = t0 + tb;
    __half* dst = xs + (int64_t)tb * K2 + which * H;
    if (t >= a.T) {
      for (int c4 = lane; c4 < V; c4 += kWave)
        *reinterpret_cast<uint2*>(dst + 4 * c4) = make_uint2(0u, 0u);
      continue;
    }
    const float* src = which ? a.m + t * a.m_t + (int64_t)s * a.m_s
                             : a.h + t * a.h_t + (int64_t)s * a.h_s;
    stage_unit_row(src, dst, V, a.inv_h, a.eps, lane);
  }
  __syncthreads();

  const uint4* x4 = reinterpret_cast<const uint4*>(xs);
  const int K8 = K2 >> 3;
  // output rows of this wave: r = blockIdx.x * nwaves + wave + k * stride
  for (int r = blockIdx.x * nwaves + wave; r < a.R; r += gridDim.x * nwaves) {
    const uint4* w4 =
        reinterpret_cast<const uint4*>(a.wd + ((int64_t)s * a.R + r) * K2);
    float acc[TB];
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) acc[tb] = 0.f;
  #pragma unroll 6
    for (int i = lane; i < K8; i += kWave) {
      const uint4 wv = w4[i];
      const half2* wh = reinterpret_cast<const half2*>(&wv);
  #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const uint4 xv = x4[(int64_t)tb * K8 + i];
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        float c = acc[tb];
        c = __builtin_amdgcn_fdot2(wh[0], xh[0], c, false);
        c = __builtin_amdgcn_fdot2(wh[1], xh[1], c, false);
        c = __builtin_amdgcn_fdot2(wh[2], xh[2], c, false);
        c = __builtin_amdgcn_fdot2(wh[3], xh[3], c, false);
        acc[tb] = c;
      }
    }
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) {
      const float sum = wave_sum(acc[tb]);
      const int t = t0 + tb;
      if (lane == 0 && t < a.T)
        a.d[((int64_t)s * a.T + t) * a.R + r] = __float2half_rn(sum);
    }
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int TB>
__global__ void __launch_bounds__(kGateWaves* kWave)
    gate_down_kernel(const GateDownArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  gate_down_body<TB>(a);
#endif
}

// ---------------------------------------------------------------------------
// MessageGate, step 2: forget / gain projections, mix, optional RMSNorm -> fp16
// ---------------------------------------------------------------------------
struct GateMixArgs {
  const float* h;
  int64_t h_t, h_s;
  const float* m;
  int64_t m_t, m_s;
  const __half* d;  // [G, T, R]
  const void* wf;   // fp32 [G, R, H]  or  fp16 [G, R/2, H, 2]
  const void* wg;
  const void* nw;  // [1 or G, H] fp16 / fp32 (NORM)
  int64_t nw_s;
  int nw_f16, nw_rows;
  float* out32;  // [G, T, H] mix result (the output without NORM, scratch with)
  __half* out16;  // [G, T, H] normalised fp16 (NORM)
  float* part;    // [G, T, C] per-chunk sums of squares (NORM)
  int* counters;  // [G, ceil(T/TB)] (NORM), zero on entry and exit
  int T, R, H, C;
  float norm_eps, inv_h;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename WT, int TB, bool NORM>
__device__ __forceinline__ void gate_mix_body(const GateMixArgs& a) {
  __shared__ int s_last;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int nwaves = blockDim.x / kWave;
  const int s = blockIdx.y;
  const int tile = blockIdx.z;
  const int t0 = tile * TB;
  const int cc = blockIdx.x * nwaves + wave;  // 128-column chunk of this wave
  const int c = cc * 128 + lane * 4;
  const bool active = c < a.H;
  const int H = a.H, R = a.R, T = a.T;

  float accf[TB][4], accg[TB][4];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb)
  #pragma unroll
    for (int j = 0; j < 4; ++j) accf[tb][j] = accg[tb][j] = 0.f;

  if (active) {
    int tclamp[TB];
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) tclamp[tb] = min(t0 + tb, T - 1);
    if constexpr (std::is_same_v<WT, float>) {
      const float* wf =
          reinterpret_cast<const float*>(a.wf) + (int64_t)s * R * H + c;
      const float* wg =
          reinterpret_cast<const float*>(a.wg) + (int64_t)s * R * H + c;
  #pragma unroll 8
      for (int r = 0; r < R; ++r) {
        const float4 fv = load4(wf + (int64_t)r * H);
        const float4 gv = load4(wg + (int64_t)r * H);
  #pragma unroll
        for (int tb = 0; tb < TB; ++tb) {
          const float dr =
              __half2float(a.d[((int64_t)s * T + tclamp[tb]) * R + r]);
          accf[tb][0] = fmaf(dr, fv.x, accf[tb][0]);
          accf[tb][1] = fmaf(dr, fv.y, accf[tb][1]);
          accf[tb][2] = fmaf(dr, fv.z, accf[tb][2]);
          accf[tb][3] = fmaf(dr, fv.w, accf[tb][3]);
          accg[tb][0] = fmaf(dr, gv.x, accg[tb][0]);
          accg[tb][1] = fmaf(dr, gv.y, accg[tb][1]);
          accg[tb][2] = fmaf(dr, gv.z, accg[tb][2]);
          accg[tb][3] = fmaf(dr, gv.w, accg[tb][3]);
        }
      }
    } else {
      const int R2 = R >> 1;
      const __half* wf =
          reinterpret_cast<const __half*>(a.wf) + ((int64_t)s * R2 * H + c) * 2;
      const __half* wg =
          reinterpret_cast<const __half*>(a.wg) + ((int64_t)s * R2 * H + c) * 2;
  #pragma unroll 8
      for (int p = 0; p < R2; ++p) {
        const uint4 fv =
            *reinterpret_cast<const uint4*>(wf + (int64_t)p * H * 2);
        const uint4 gv =
            *reinterpret_cast<const uint4*>(wg + (int64_t)p * H * 2);
        const half2* fh = reinterpret_cast<const half2*>(&fv);
        const half2* gh = reinterpret_cast<const half2*>(&gv);
  #pragma unroll
        for (int tb = 0; tb < TB; ++tb) {
          const half2 dd = *reinterpret_cast<const half2*>(
              a.d + ((int64_t)s * T + tclamp[tb]) * R + 2 * p);
  #pragma unroll
          for (int j = 0; j < 4; ++j) {
            accf[tb][j] = __builtin_amdgcn_fdot2(fh[j], dd, accf[tb][j], false);
            accg[tb][j] = __builtin_amdgcn_fdot2(gh[j], dd, accg[tb][j], false);
          }
        }
      }
    }
  }

  // mix: out = (1 - tanh(f)) * h + tanh(g) * m
  float ss[TB];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    ss[tb] = 0.f;
    const int t = t0 + tb;
    if (active && t < T) {
      const float4 hv = load4(a.h + t * a.h_t + (int64_t)s * a.h_s + c);
      const float4 mv = load4(a.m + t * a.m_t + (int64_t)s * a.m_s + c);
      float4 o;
      o.x = (1.0f - tanhf(accf[tb][0])) * hv.x + tanhf(accg[tb][0]) * mv.x;
      o.y = (1.0f - tanhf(accf[tb][1])) * hv.y + tanhf(accg[tb][1]) * mv.y;
      o.z = (1.0f - tanhf(accf[tb][2])) * hv.z + tanhf(accg[tb][2]) * mv.z;
      o.w = (1.0f - tanhf(accf[tb][3])) * hv.w + tanhf(accg[tb][3]) * mv.w;
      store4(a.out32 + ((int64_t)s * T + t) * H + c, o);
      ss[tb] = dot4(o, o);
    }
  }
  if constexpr (!NORM) return;

  // partial sums of squares, then the last CTA of (gate, token tile) normalises
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    const float tot = wave_sum(ss[tb]);
    const int t = t0 + tb;
    if (lane == 0 && t < T && cc < a.C)
      a.part[((int64_t)s * T + t) * a.C + cc] = tot;
  }
  __threadfence();
  __syncthreads();
  const int ntiles = gridDim.z;
  int* counter = a.counters + (int64_t)s * ntiles + tile;
  if (tid == 0) {
    const int old = atomicAdd(counter, 1);
    s_last = (old == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  for (int tb = wave; tb < TB; tb += nwaves) {
    const int t = t0 + tb;
    if (t >= T) continue;
    const float* pp = a.part + ((int64_t)s * T + t) * a.C;
    float tot = 0.f;
    for (int i = lane; i < a.C; i += kWave) tot += pp[i];
    tot = wave_sum(tot);
    const float rs = rsqrtf(tot * a.inv_h + a.norm_eps);
    const int64_t row = ((int64_t)s * T + t) * H;
    const int64_t wrow = (a.nw_rows == 1 ? 0 : s) * a.nw_s;
    for (int c4 = lane; c4 < (H >> 2); c4 += kWave) {
      const float4 v = load4(a.out32 + row + 4 * c4);
      const float4 w = load4_dyn(a.nw, wrow + 4 * c4, a.nw_f16);
      store4(a.out16 + row + 4 * c4,
             make_float4(w.x * (v.x * rs), w.y * (v.y * rs), w.z * (v.z * rs),
                         w.w * (v.w * rs)));
    }
  }
  __syncthreads();
  if (tid == 0) *counter = 0;
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <typename WT, int TB, bool NORM>
__global__ void __launch_bounds__(2 * kWave)
    gate_mix_kernel(const GateMixArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  gate_mix_body<WT, TB, NORM>(a);
#endif
}

// ---------------------------------------------------------------------------
// rms_norm / embed_norm
// ---------------------------------------------------------------------------
struct RmsArgs {
  const void* x;  // [T, H] fp16 / fp32, or the embedding table
  int x_f16;
  int64_t x_t;
  const void* ids;  // embedding gather: [T] int32 / int64 (null: no gather)
  int ids_i64;
  int64_t vocab;
  const float* x1;  // optional fp32 addend
  int64_t x1_t;
  float* sum_out;  // optional: receives x + x1
  int64_t so_t;
  const void* w;  // [H] fp16 / fp32 (null: unweighted)
  int w_f16;
  void* out;
  int out_f16;
  int64_t o_t;
  int H, round_h;
  float eps, inv_h;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int ITER>
__device__ __forceinline__ void rms_norm_body(const RmsArgs& a) {
  __shared__ float s_scratch[8];
  const int tid = threadIdx.x, nth = blockDim.x;
  const int64_t t = blockIdx.x;
  const int V = a.H >> 2;
  int64_t xrow = t * a.x_t;
  if (a.ids != nullptr) {
    int64_t id = a.ids_i64
                     ? reinterpret_cast<const int64_t*>(a.ids)[t]
                     : (int64_t)reinterpret_cast<const int32_t*>(a.ids)[t];
    id = id < 0 ? 0 : (id >= a.vocab ? a.vocab - 1 : id);
    xrow = id * a.x_t;
  }
  float4 v[ITER];
  float ss = 0.f;
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int i = tid + k * nth;
    float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < V) {
      p = load4_dyn(a.x, xrow + 4 * i, a.x_f16);
      if (a.x1 != nullptr) {
        const float4 q1 = load4(a.x1 + t * a.x1_t + 4 * i);
        p.x += q1.x;
        p.y += q1.y;
        p.z += q1.z;
        p.w += q1.w;
      }
      if (a.sum_out != nullptr) store4(a.sum_out + t * a.so_t + 4 * i, p);
    }
    v[k] = p;
    ss += dot4(p, p);
  }
  ss = block_sum(ss, s_scratch);
  const float rs = rsqrtf(ss * a.inv_h + a.eps);
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int i = tid + k * nth;
    if (i >= V) continue;
    float4 y = make_float4(v[k].x * rs, v[k].y * rs, v[k].z * rs, v[k].w * rs);
    if (a.w != nullptr) {
      const float4 w = load4_dyn(a.w, 4 * i, a.w_f16);
      y = make_float4(w.x * y.x, w.y * y.y, w.z * y.z, w.w * y.w);
    }
    if (a.round_h) {
      y = make_float4(round_h(y.x), round_h(y.y), round_h(y.z), round_h(y.w));
    }
    if (a.out_f16)
      store4(reinterpret_cast<__half*>(a.out) + t * a.o_t + 4 * i, y);
    else
      store4(reinterpret_cast<float*>(a.out) + t * a.o_t + 4 * i, y);
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int ITER>
__global__ void __launch_bounds__(256) rms_norm_kernel(const RmsArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  rms_norm_body<ITER>(a);
#endif
}

// ---------------------------------------------------------------------------
// swiglu: softcapped SwiGLU between the two expert GEMMs
// ---------------------------------------------------------------------------
struct SwigluArgs {
  const __half* gate_up;  // [R, 2I]
  __half* out;            // [R, I]
  int64_t n8;             // R * I / 8
  int I8;                 // I / 8
  float gate_cap, up_cap;
};

__device__ __forceinline__ float swiglu1(float g, float u, float gc, float uc) {
  return ((tanhf(g / gc) * gc) * sigmoidf_(g)) * (tanhf(u / uc) * uc);
}

template <int DUMMY = 0>
__global__ void __launch_bounds__(256) swiglu_kernel(const SwigluArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  for (int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x; i < a.n8;
       i += (int64_t)gridDim.x * 256) {
    const int64_t row = i / a.I8;
    const int64_t col8 = i - row * a.I8;
    const uint4* g4 =
        reinterpret_cast<const uint4*>(a.gate_up) + row * (2 * a.I8);
    const uint4 gq = g4[col8];
    const uint4 uq = g4[a.I8 + col8];
    const uint32_t* gw = reinterpret_cast<const uint32_t*>(&gq);
    const uint32_t* uw = reinterpret_cast<const uint32_t*>(&uq);
    uint4 oq;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&oq);
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      ow[j] =
          pack_h2(swiglu1(h2f_lo(gw[j]), h2f_lo(uw[j]), a.gate_cap, a.up_cap),
                  swiglu1(h2f_hi(gw[j]), h2f_hi(uw[j]), a.gate_cap, a.up_cap));
    }
    reinterpret_cast<uint4*>(a.out)[i] = oq;
  }
#endif
}

// ---------------------------------------------------------------------------
// copy_states: [T, P, H] fp32 -> [T, P, H] fp32 (snapshots of persistent
// states)
// ---------------------------------------------------------------------------
struct CopyArgs {
  const float* src;
  float* dst;
  int64_t s_t, s_p, d_t, d_p;  // row strides in elements
  int P, H;
};

template <int DUMMY = 0>
__global__ void __launch_bounds__(256) copy_states_kernel(const CopyArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int t = blockIdx.x / a.P, p = blockIdx.x - t * a.P;
  const float4* s =
      reinterpret_cast<const float4*>(a.src + t * a.s_t + p * a.s_p);
  float4* d = reinterpret_cast<float4*>(a.dst + t * a.d_t + p * a.d_p);
  for (int i = threadIdx.x; i < (a.H >> 2); i += 256) d[i] = s[i];
#endif
}

// ---------------------------------------------------------------------------
// moe_route: logits, sigmoid, biased top-k, normalisation, align, zero fills
// ---------------------------------------------------------------------------
struct RouteArgs {
  const __half* x;  // [T, H]
  int64_t x_t;
  const void* w;  // [E, H] fp32 or fp16
  const float* bias;
  float* logits;  // [T, E]
  float* topk_w;  // [T, K]
  int32_t* topk_ids;
  int32_t* sorted_ids;
  int32_t* expert_ids;
  int32_t* post_pad;
  int32_t* counters;  // [NT + 1]
  __half* zero1;
  int64_t zero1_n;
  __half* zero2;
  int64_t zero2_n;
  int T, H, E, K, BM, max_padded, max_blocks, NEG, NT, nzero;
  float route_scale;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// Top-k of the token handled by this wave (experts lane, lane + 32, ...).
__device__ __forceinline__ void route_topk_wave(const RouteArgs& a, int t) {
  constexpr int kMaxEpl = (kMaxExperts + kWave - 1) / kWave;
  const int lane = threadIdx.x & (kWave - 1);
  const int epl = (a.E + kWave - 1) / kWave;
  float sc[kMaxEpl], bs[kMaxEpl];
  #pragma unroll
  for (int i = 0; i < kMaxEpl; ++i) {
    const int e = lane + kWave * i;
    sc[i] = 0.f;
    bs[i] = -INFINITY;
    if (i < epl && e < a.E) {
      const float s = sigmoidf_(a.logits[(int64_t)t * a.E + e]);
      sc[i] = s;
      bs[i] = s + a.bias[e];
    }
  }
  __shared__ int s_sel_id[kRouteThreads / kWave][kMaxTopK];
  __shared__ float s_sel_w[kRouteThreads / kWave][kMaxTopK];
  const int wave = threadIdx.x / kWave;
  for (int k = 0; k < a.K; ++k) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
  #pragma unroll
    for (int i = 0; i < kMaxEpl; ++i) {
      if (bs[i] > bv) {
        bv = bs[i];
        bi = lane + kWave * i;
      }
    }
  #pragma unroll
    for (int o = kWave / 2; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o);
      const int oi = __shfl_xor(bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    // NaN scores never win a comparison: keep the id a valid expert index
    if (bi >= a.E) bi = a.E - 1;
    float mine = 0.f;
  #pragma unroll
    for (int i = 0; i < kMaxEpl; ++i) {
      if (lane + kWave * i == bi) {
        mine = sc[i];
        bs[i] = -INFINITY;
      }
    }
    const float sval = wave_sum(mine);  // only the owner lane is non-zero
    if (lane == 0) {
      s_sel_id[wave][k] = bi;
      s_sel_w[wave][k] = sval;
    }
  }
  if (lane == 0) {
    float sum = 0.f;
    for (int k = 0; k < a.K; ++k) sum += s_sel_w[wave][k];
    const float den = sum + 1e-20f;
    for (int k = 0; k < a.K; ++k) {
      a.topk_ids[(int64_t)t * a.K + k] = s_sel_id[wave][k];
      a.topk_w[(int64_t)t * a.K + k] = (s_sel_w[wave][k] / den) * a.route_scale;
    }
  }
}

// Lanes (of this wave) whose 8-bit `key` equals mine, as a lane mask: a
// bitwise ballot match, 8 votes instead of one compare per distinct key.
__device__ __forceinline__ uint32_t match_key(uint32_t key) {
  uint32_t m = 0xFFFFFFFFu;
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    const uint32_t bit = (key >> b) & 1u;
    const uint32_t vote = (uint32_t)__ballot((int)bit);
    m &= bit ? vote : ~vote;
  }
  return m;
}

// One CTA: stable counting sort of the routed ids into expert blocks.
// Pass 1: every wave histograms a contiguous range of the entries, 32 per
// step.  The expert block layout and, for every expert, the start of each
// wave's share follow from the histograms; pass 2 scatters in the same order,
// so the order inside an expert stays (token, k).
__device__ __forceinline__ void route_align_cta(const RouteArgs& a,
                                                unsigned char* smem) {
  constexpr int kWaves = kRouteThreads / kWave;
  constexpr int kPer = kRouteMaxEntries / kRouteThreads;  // ids per thread
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int numel = a.T * a.K;
  const int E = a.E, BM = a.BM;
  int* s_hist = reinterpret_cast<int*>(smem);  // [kWaves][256]
  int* s_ws = reinterpret_cast<int*>(smem + kAlignOffWs);
  uint8_t* s_key = smem + kAlignOffKeys;

  // the routed ids as bytes; every load of a thread is in flight before the
  // first one is used
  int ids[kPer];
  #pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = tid + j * kRouteThreads;
    ids[j] = i < numel ? a.topk_ids[i] : 0;
  }
  for (int i = tid; i < kWaves * 256; i += kRouteThreads) s_hist[i] = 0;
  for (int p = tid; p < a.max_padded; p += kRouteThreads)
    a.sorted_ids[p] = numel;
  #pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = tid + j * kRouteThreads;
    if (i < numel) s_key[i] = (uint8_t)ids[j];
  }
  __syncthreads();

  // entries [lo, hi) of this wave
  const int span = (((numel + kWaves - 1) / kWaves) + kWave - 1) & ~(kWave - 1);
  const int lo = wave * span, hi = min(lo + span, numel);
  const uint32_t lower = (1u << lane) - 1u;
  int* hist = s_hist + wave * 256;
  for (int g = lo; g < hi; g += kWave) {
    const int i = g + lane;
    const bool valid = i < hi;
    const uint32_t key = valid ? s_key[i] : 0xFFu;
    const uint32_t m = match_key(key);
    if (valid && (m & lower) == 0)
      hist[key] += __popc(m);  // first lane of a key
  }
  __syncthreads();

  int cnt = 0;
  if (tid < E) {
    for (int w = 0; w < kWaves; ++w) cnt += s_hist[w * 256 + tid];
  }
  const int nb = (cnt + BM - 1) / BM;
  int v = nb;
  #pragma unroll
  for (int d = 1; d < kWave; d <<= 1) {
    const int n = __shfl_up(v, d);
    if (lane >= d) v += n;
  }
  if (lane == kWave - 1) s_ws[wave] = v;
  __syncthreads();
  int base = 0, total = 0;
  for (int w = 0; w < kWaves; ++w) {
    if (w < wave) base += s_ws[w];
    total += s_ws[w];
  }
  const int first_block = base + v - nb;
  if (tid < E) {
    for (int b = 0; b < nb; ++b) a.expert_ids[first_block + b] = tid;
    // cursor of every wave inside the segment of this expert
    int run = first_block * BM;
    for (int w = 0; w < kWaves; ++w) {
      const int h = s_hist[w * 256 + tid];
      s_hist[w * 256 + tid] = run;
      run += h;
    }
  }
  for (int b = total + tid; b < a.max_blocks; b += kRouteThreads)
    a.expert_ids[b] = -1;
  if (tid == 0) a.post_pad[0] = total * BM;
  __syncthreads();

  for (int g = lo; g < hi; g += kWave) {
    const int i = g + lane;
    const bool valid = i < hi;
    const uint32_t key = valid ? s_key[i] : 0xFFu;
    const uint32_t m = match_key(key);
    if (valid) a.sorted_ids[hist[key] + __popc(m & lower)] = i;
    if (valid && (m & lower) == 0) hist[key] += __popc(m);
  }
}

template <typename WT, int TB>
__device__ __forceinline__ void moe_route_body(const RouteArgs& a) {
  extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
  __shared__ int s_flag;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int bid = blockIdx.x;

  if (bid < a.nzero) {  // zero the two MoE accumulators; these are the first
                        // CTAs so they overlap with the GEMV of the others
    const int64_t n1 = a.zero1_n >> 3, n2 = a.zero2_n >> 3;
    for (int64_t i = (int64_t)bid * kRouteThreads + tid; i < n1 + n2;
         i += (int64_t)a.nzero * kRouteThreads) {
      uint4* p = i < n1 ? reinterpret_cast<uint4*>(a.zero1) + i
                        : reinterpret_cast<uint4*>(a.zero2) + (i - n1);
      *p = make_uint4(0u, 0u, 0u, 0u);
    }
    return;
  }

  using XT = std::conditional_t<std::is_same_v<WT, float>, float, __half>;
  XT* xs = reinterpret_cast<XT*>(smem);  // [TB][H]
  const int rb = bid - a.nzero;
  const int tile = rb / a.NEG;
  const int eg = rb - tile * a.NEG;
  const int t0 = tile * TB;
  const int H = a.H, V = H >> 2, E = a.E;

  for (int i = tid; i < TB * V; i += kRouteThreads) {
    const int tb = i / V, c4 = i - tb * V;
    const int t = t0 + tb;
    uint2 u = make_uint2(0u, 0u);
    if (t < a.T)
      u = *reinterpret_cast<const uint2*>(a.x + (int64_t)t * a.x_t + 4 * c4);
    if constexpr (std::is_same_v<XT, float>) {
      *reinterpret_cast<float4*>(xs + (int64_t)tb * H + 4 * c4) =
          make_float4(h2f_lo(u.x), h2f_hi(u.x), h2f_lo(u.y), h2f_hi(u.y));
    } else {
      *reinterpret_cast<uint2*>(xs + (int64_t)tb * H + 4 * c4) = u;
    }
  }
  __syncthreads();

  // expert rows of this wave: e = eg * waves + wave + k * NEG * waves
  for (int e = eg * (kRouteThreads / kWave) + wave; e < E;
       e += a.NEG * (kRouteThreads / kWave)) {
    float acc[TB];
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) acc[tb] = 0.f;
    if constexpr (std::is_same_v<WT, float>) {
      const float4* w4 = reinterpret_cast<const float4*>(a.w) + (int64_t)e * V;
  #pragma unroll 4
      for (int i = lane; i < V; i += kWave) {
        const float4 wv = w4[i];
  #pragma unroll
        for (int tb = 0; tb < TB; ++tb) {
          const float4 xv =
              *reinterpret_cast<const float4*>(xs + (int64_t)tb * H + 4 * i);
          acc[tb] = fmaf(wv.x, xv.x, acc[tb]);
          acc[tb] = fmaf(wv.y, xv.y, acc[tb]);
          acc[tb] = fmaf(wv.z, xv.z, acc[tb]);
          acc[tb] = fmaf(wv.w, xv.w, acc[tb]);
        }
      }
    } else {
      const uint4* w8 =
          reinterpret_cast<const uint4*>(a.w) + (int64_t)e * (H >> 3);
  #pragma unroll 4
      for (int i = lane; i < (H >> 3); i += kWave) {
        const uint4 wv = w8[i];
        const half2* wh = reinterpret_cast<const half2*>(&wv);
  #pragma unroll
        for (int tb = 0; tb < TB; ++tb) {
          const uint4 xv =
              *reinterpret_cast<const uint4*>(xs + (int64_t)tb * H + 8 * i);
          const half2* xh = reinterpret_cast<const half2*>(&xv);
          float c = acc[tb];
          c = __builtin_amdgcn_fdot2(wh[0], xh[0], c, false);
          c = __builtin_amdgcn_fdot2(wh[1], xh[1], c, false);
          c = __builtin_amdgcn_fdot2(wh[2], xh[2], c, false);
          c = __builtin_amdgcn_fdot2(wh[3], xh[3], c, false);
          acc[tb] = c;
        }
      }
    }
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) {
      const float sum = wave_sum(acc[tb]);
      if (lane == 0 && t0 + tb < a.T)
        a.logits[(int64_t)(t0 + tb) * E + e] = sum;
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int old = atomicAdd(a.counters + tile, 1);
    s_flag = (old == a.NEG - 1);
  }
  __syncthreads();
  if (!s_flag) return;
  __threadfence();

  // last CTA of this token tile: top-k per token (one wave each)
  if (wave < TB && t0 + wave < a.T) route_topk_wave(a, t0 + wave);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    a.counters[tile] = 0;
    const int old = atomicAdd(a.counters + a.NT, 1);
    s_flag = (old == a.NT - 1);
  }
  __syncthreads();
  if (!s_flag) return;
  __threadfence();

  // last tile overall: the expert-block layout of every routed row
  route_align_cta(a, smem);
  if (tid == 0) a.counters[a.NT] = 0;
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <typename WT, int TB>
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_kernel(const RouteArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  moe_route_body<WT, TB>(a);
#endif
}

}  // namespace inkling2_glue

// ===========================================================================
// Host side
// ===========================================================================
namespace {

using namespace inkling2_glue;

void check_gpu(const at::Tensor& t, const char* who, const char* name) {
  TORCH_CHECK(t.defined() && t.is_cuda(), who, ": ", name,
              " must be a GPU tensor");
}

void check_dtype(const at::Tensor& t, const char* who, const char* name,
                 at::ScalarType dt) {
  TORCH_CHECK(t.scalar_type() == dt, who, ": ", name, " must be ",
              c10::toString(dt), ", got ", c10::toString(t.scalar_type()));
}

// unit stride on the last dim, every other stride and the base pointer aligned
// to `vec` elements (so 16 / 8 byte vector accesses are legal)
void check_vec_rows(const at::Tensor& t, const char* who, const char* name,
                    int64_t vec) {
  TORCH_CHECK(t.dim() >= 1 && t.stride(-1) == 1, who, ": ", name,
              " must have unit stride on the last dim");
  for (int64_t i = 0; i + 1 < t.dim(); ++i)
    TORCH_CHECK(t.size(i) <= 1 || t.stride(i) % vec == 0, who, ": ", name,
                " strides must be multiples of ", vec, " elements");
  TORCH_CHECK(t.size(-1) % vec == 0, who, ": ", name,
              " last dim must be a multiple of ", vec);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(t.data_ptr()) % (vec * t.element_size()) == 0,
      who, ": ", name, " must be ", vec * t.element_size(), "-byte aligned");
}

bool is_f16(const at::Tensor& t, const char* who, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat,
              who, ": ", name, " must be fp16 or fp32");
  return t.scalar_type() == at::kHalf;
}

template <typename T>
const T* opt_ptr(const std::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined())
             ? reinterpret_cast<const T*>(t->data_ptr())
             : nullptr;
}

// 1..4 float4 per thread: threads = ceil(V / iter) rounded up to a wave
void row_launch_shape(int64_t H, const char* who, int& iter, int& block) {
  TORCH_CHECK(H % 4 == 0 && H >= 4, who,
              ": hidden size must be a multiple of 4");
  const int64_t V = H / 4;
  iter = (int)((V + 255) / 256);
  TORCH_CHECK(iter >= 1 && iter <= 4, who, ": hidden size ", H,
              " > 4096 unsupported");
  block = (int)(((V + iter - 1) / iter + kWave - 1) / kWave * kWave);
}

}  // namespace

// ---------------------------------------------------------------------------
// inkling2_glue_attn_res
// ---------------------------------------------------------------------------
template <int Q, bool NORM>
static void launch_attn_res(int iter, int block, int64_t T,
                            const AttnResArgs& a, hipStream_t stream) {
#define I2G_AR(IT)                                          \
  case IT:                                                  \
    inkling2_glue::attn_res_kernel<Q, IT, NORM>             \
        <<<dim3((unsigned)T), dim3(block), 0, stream>>>(a); \
    break
  switch (iter) {
    I2G_AR(1);
    I2G_AR(2);
    I2G_AR(3);
    I2G_AR(4);
    default:
      TORCH_CHECK(false, "inkling2_glue_attn_res: bad iter ", iter);
  }
#undef I2G_AR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_glue_attn_res(const at::Tensor& blocks, int64_t n_blocks,
                            const at::Tensor& x0,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& partial_out,
                            const std::optional<at::Tensor>& append,
                            const at::Tensor& query,
                            const std::optional<at::Tensor>& norm_w,
                            const at::Tensor& out, double eps,
                            double norm_eps) {
  const char* who = "inkling2_glue_attn_res";
  check_gpu(x0, who, "x0");
  check_gpu(query, who, "query");
  check_gpu(out, who, "out");
  TORCH_CHECK(x0.dim() == 2, who, ": x0 must be [T, H]");
  const int64_t T = x0.size(0), H = x0.size(1);
  check_dtype(x0, who, "x0", at::kFloat);
  check_vec_rows(x0, who, "x0", 4);
  check_dtype(query, who, "query", at::kFloat);
  TORCH_CHECK(query.dim() == 2 && query.size(1) == H, who,
              ": query must be [Q, H]");
  check_vec_rows(query, who, "query", 4);
  const int64_t Q = query.size(0);
  TORCH_CHECK(Q == 1 || Q == 3 || Q == 4, who, ": Q must be 1, 3 or 4, got ",
              Q);
  TORCH_CHECK(out.dim() == 3 && out.size(0) == T && out.size(1) == Q &&
                  out.size(2) == H,
              who, ": out must be [T, Q, H]");
  const bool norm = norm_w.has_value() && norm_w->defined();
  check_dtype(out, who, "out", norm ? at::kHalf : at::kFloat);
  check_vec_rows(out, who, "out", 4);
  TORCH_CHECK(n_blocks >= 0, who, ": n_blocks must be >= 0");

  AttnResArgs a{};
  a.x0 = x0.data_ptr<float>();
  a.x0_t = x0.stride(0);
  a.query = query.data_ptr<float>();
  a.q_s = query.stride(0);
  a.out = out.data_ptr();
  a.o_t = out.stride(0);
  a.o_q = out.stride(1);
  a.H = (int)H;
  a.n_blocks = (int)n_blocks;
  a.eps = (float)eps;
  a.norm_eps = (float)norm_eps;
  a.inv_h = 1.0f / (float)H;
  a.scale = (float)std::pow((double)H, -0.5);

  if (x1.has_value() && x1->defined()) {
    check_gpu(*x1, who, "x1");
    check_dtype(*x1, who, "x1", at::kFloat);
    TORCH_CHECK(x1->dim() == 2 && x1->size(0) == T && x1->size(1) == H, who,
                ": x1 must be [T, H]");
    check_vec_rows(*x1, who, "x1", 4);
    a.x1 = x1->data_ptr<float>();
    a.x1_t = x1->stride(0);
  }
  if (partial_out.has_value() && partial_out->defined()) {
    check_gpu(*partial_out, who, "partial_out");
    check_dtype(*partial_out, who, "partial_out", at::kFloat);
    TORCH_CHECK(partial_out->dim() == 2 && partial_out->size(0) == T &&
                    partial_out->size(1) == H,
                who, ": partial_out must be [T, H]");
    check_vec_rows(*partial_out, who, "partial_out", 4);
    a.partial_out = partial_out->data_ptr<float>();
    a.po_t = partial_out->stride(0);
  }
  const bool has_app = append.has_value() && append->defined();
  if (has_app) {
    check_gpu(*append, who, "append");
    check_dtype(*append, who, "append", at::kFloat);
    TORCH_CHECK(
        append->dim() == 2 && append->size(0) == T && append->size(1) == H, who,
        ": append must be [T, H]");
    check_vec_rows(*append, who, "append", 4);
    a.append = append->data_ptr<float>();
    a.ap_t = append->stride(0);
  }
  const int64_t NS = n_blocks + (has_app ? 1 : 0) + 1;
  TORCH_CHECK(NS <= kMaxStates, who, ": ", NS, " states exceed the maximum ",
              kMaxStates);
  if (n_blocks > 0 || has_app) {
    check_gpu(blocks, who, "blocks");
    check_dtype(blocks, who, "blocks", at::kFloat);
    TORCH_CHECK(blocks.dim() == 3 && blocks.size(0) == T &&
                    blocks.size(2) == H &&
                    blocks.size(1) >= n_blocks + (has_app ? 1 : 0),
                who, ": blocks must be [T, >= n_blocks (+1 with append), H]");
    check_vec_rows(blocks, who, "blocks", 4);
    a.blocks = blocks.data_ptr<float>();
    a.bs_t = blocks.stride(0);
    a.bs_s = blocks.stride(1);
  }
  if (norm) {
    check_gpu(*norm_w, who, "norm_w");
    a.nw_f16 = is_f16(*norm_w, who, "norm_w");
    TORCH_CHECK(norm_w->dim() == 2 && norm_w->size(1) == H &&
                    (norm_w->size(0) == 1 || norm_w->size(0) == Q),
                who, ": norm_w must be [1 or Q, H]");
    check_vec_rows(*norm_w, who, "norm_w", 4);
    a.norm_w = norm_w->data_ptr();
    a.nw_s = norm_w->stride(0);
    a.nw_rows = (int)norm_w->size(0);
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x0));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  int iter, block;
  row_launch_shape(H, who, iter, block);
#define I2G_ARQ(QQ)                                          \
  case QQ:                                                   \
    if (norm)                                                \
      launch_attn_res<QQ, true>(iter, block, T, a, stream);  \
    else                                                     \
      launch_attn_res<QQ, false>(iter, block, T, a, stream); \
    break
  switch (Q) {
    I2G_ARQ(1);
    I2G_ARQ(3);
    I2G_ARQ(4);
    default:
      break;
  }
#undef I2G_ARQ
}

// ---------------------------------------------------------------------------
// inkling2_glue_msg_gate_down / _mix
// ---------------------------------------------------------------------------
static int gate_tb(int64_t T, int max_tb) {
  int tb = 1;
  while (tb < max_tb && tb < T) tb <<= 1;
  return tb;
}

// tokens per gate_mix CTA: the weights are re-read once per token tile, so take
// as many as possible while at least kMinCtas CTAs stay in flight
static int gate_mix_tb(int64_t T, int64_t ctas_per_tile, int max_tb) {
  int tb = gate_tb(T, max_tb);
  while (tb > 1 && ctas_per_tile * ((T + tb - 1) / tb) < kMinCtas) tb >>= 1;
  return tb;
}

void inkling2_glue_msg_gate_down(const at::Tensor& h, const at::Tensor& m,
                                 const at::Tensor& wd, const at::Tensor& d,
                                 double eps) {
  const char* who = "inkling2_glue_msg_gate_down";
  check_gpu(h, who, "h");
  check_gpu(m, who, "m");
  check_gpu(wd, who, "wd");
  check_gpu(d, who, "d");
  check_dtype(h, who, "h", at::kFloat);
  check_dtype(m, who, "m", at::kFloat);
  check_dtype(wd, who, "wd", at::kHalf);
  check_dtype(d, who, "d", at::kHalf);
  TORCH_CHECK(m.dim() == 3 && h.dim() == 3, who,
              ": h and m must be [T, G|1, H]");
  const int64_t T = m.size(0), G = m.size(1), H = m.size(2);
  TORCH_CHECK(
      h.size(0) == T && h.size(2) == H && (h.size(1) == G || h.size(1) == 1),
      who, ": h must be [T, G or 1, H]");
  check_vec_rows(h, who, "h", 4);
  check_vec_rows(m, who, "m", 4);
  TORCH_CHECK(wd.dim() == 3 && wd.size(0) == G && wd.size(2) == 2 * H &&
                  wd.is_contiguous(),
              who, ": wd must be contiguous [G, R, 2H]");
  const int64_t R = wd.size(1);
  TORCH_CHECK(R % 2 == 0 && R <= 256, who, ": rank must be even and <= 256");
  TORCH_CHECK(d.dim() == 3 && d.size(0) == G && d.size(1) == T &&
                  d.size(2) == R && d.is_contiguous(),
              who, ": d must be contiguous [G, T, R]");
  TORCH_CHECK((2 * H) % 8 == 0 && H <= kStageV * kWave * 4, who,
              ": 2H must be a multiple of 8 and H <= ", kStageV * kWave * 4);

  GateDownArgs a{};
  a.h = h.data_ptr<float>();
  a.h_t = h.stride(0);
  a.h_s = h.size(1) == 1 ? 0 : h.stride(1);
  a.m = m.data_ptr<float>();
  a.m_t = m.stride(0);
  a.m_s = m.stride(1);
  a.wd = reinterpret_cast<const __half*>(wd.data_ptr());
  a.d = reinterpret_cast<__half*>(d.data_ptr());
  a.T = (int)T;
  a.R = (int)R;
  a.H = (int)H;
  a.eps = (float)eps;
  a.inv_h = 1.0f / (float)H;

  const at::cuda::OptionalCUDAGuard guard(device_of(m));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  const int tb = gate_tb(T, 4);
  const int64_t tiles = (T + tb - 1) / tb;
  const size_t shmem = (size_t)tb * 2 * H * sizeof(__half);
  TORCH_CHECK(shmem <= 64 * 1024, who,
              ": hidden size too large for the LDS tile");
  // CTAs per (gate, token tile): one output row per wave at most, as few as
  // fill the GPU (every CTA stages its own x tile)
  const int nwaves = kGateWaves;
  const int64_t nx_max = (R + nwaves - 1) / nwaves;
  const int64_t nx =
      std::clamp<int64_t>((kMinCtas + G * tiles - 1) / (G * tiles), 1, nx_max);
  const dim3 grid((unsigned)nx, (unsigned)G, (unsigned)tiles);
  const dim3 block(nwaves * kWave);
  switch (tb) {
    case 1:
      inkling2_glue::gate_down_kernel<1><<<grid, block, shmem, stream>>>(a);
      break;
    case 2:
      inkling2_glue::gate_down_kernel<2><<<grid, block, shmem, stream>>>(a);
      break;
    default:
      inkling2_glue::gate_down_kernel<4><<<grid, block, shmem, stream>>>(a);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename WT, bool NORM>
static void launch_gate_mix(int tb, dim3 grid, dim3 block, const GateMixArgs& a,
                            hipStream_t stream) {
  switch (tb) {
    case 1:
      inkling2_glue::gate_mix_kernel<WT, 1, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    case 2:
      inkling2_glue::gate_mix_kernel<WT, 2, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    case 4:
      inkling2_glue::gate_mix_kernel<WT, 4, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    case 8:
      inkling2_glue::gate_mix_kernel<WT, 8, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    default:
      if constexpr (std::is_same_v<WT, __half>) {
        inkling2_glue::gate_mix_kernel<WT, 16, NORM>
            <<<grid, block, 0, stream>>>(a);
      } else {
        TORCH_CHECK(false, "inkling2_glue_msg_gate_mix: unsupported tile ", tb);
      }
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_glue_msg_gate_mix(const at::Tensor& h, const at::Tensor& m,
                                const at::Tensor& d, const at::Tensor& wf,
                                const at::Tensor& wg,
                                const std::optional<at::Tensor>& norm_w,
                                const at::Tensor& out,
                                const std::optional<at::Tensor>& scratch,
                                const std::optional<at::Tensor>& partials,
                                const std::optional<at::Tensor>& counters,
                                double norm_eps) {
  const char* who = "inkling2_glue_msg_gate_mix";
  check_gpu(h, who, "h");
  check_gpu(m, who, "m");
  check_gpu(d, who, "d");
  check_gpu(wf, who, "wf");
  check_gpu(wg, who, "wg");
  check_gpu(out, who, "out");
  check_dtype(h, who, "h", at::kFloat);
  check_dtype(m, who, "m", at::kFloat);
  check_dtype(d, who, "d", at::kHalf);
  TORCH_CHECK(m.dim() == 3 && h.dim() == 3, who,
              ": h and m must be [T, G|1, H]");
  const int64_t T = m.size(0), G = m.size(1), H = m.size(2);
  TORCH_CHECK(
      h.size(0) == T && h.size(2) == H && (h.size(1) == G || h.size(1) == 1),
      who, ": h must be [T, G or 1, H]");
  check_vec_rows(h, who, "h", 4);
  check_vec_rows(m, who, "m", 4);
  TORCH_CHECK(
      d.dim() == 3 && d.size(0) == G && d.size(1) == T && d.is_contiguous(),
      who, ": d must be contiguous [G, T, R]");
  const int64_t R = d.size(2);
  TORCH_CHECK(R % 2 == 0, who, ": rank must be even");
  TORCH_CHECK(H % 4 == 0, who, ": hidden size must be a multiple of 4");
  const bool w16 = wf.scalar_type() == at::kHalf;
  TORCH_CHECK(wg.scalar_type() == wf.scalar_type(), who,
              ": wf and wg dtypes differ");
  if (w16) {
    TORCH_CHECK(wf.dim() == 4 && wf.size(0) == G && wf.size(1) == R / 2 &&
                    wf.size(2) == H && wf.size(3) == 2 && wf.is_contiguous() &&
                    wg.sizes() == wf.sizes() && wg.is_contiguous(),
                who, ": fp16 wf / wg must be contiguous [G, R/2, H, 2]");
  } else {
    check_dtype(wf, who, "wf", at::kFloat);
    TORCH_CHECK(wf.dim() == 3 && wf.size(0) == G && wf.size(1) == R &&
                    wf.size(2) == H && wf.is_contiguous() &&
                    wg.sizes() == wf.sizes() && wg.is_contiguous(),
                who, ": fp32 wf / wg must be contiguous [G, R, H]");
  }
  TORCH_CHECK(out.dim() == 3 && out.size(0) == G && out.size(1) == T &&
                  out.size(2) == H && out.is_contiguous(),
              who, ": out must be contiguous [G, T, H]");
  const bool norm = out.scalar_type() == at::kHalf;
  if (!norm) check_dtype(out, who, "out", at::kFloat);

  GateMixArgs a{};
  a.h = h.data_ptr<float>();
  a.h_t = h.stride(0);
  a.h_s = h.size(1) == 1 ? 0 : h.stride(1);
  a.m = m.data_ptr<float>();
  a.m_t = m.stride(0);
  a.m_s = m.stride(1);
  a.d = reinterpret_cast<const __half*>(d.data_ptr());
  a.wf = wf.data_ptr();
  a.wg = wg.data_ptr();
  a.T = (int)T;
  a.R = (int)R;
  a.H = (int)H;
  a.C = (int)((H + 127) / 128);
  a.norm_eps = (float)norm_eps;
  a.inv_h = 1.0f / (float)H;

  const int tb = gate_mix_tb(T, G * a.C, w16 ? 16 : 8);
  const int64_t ntiles = (T + tb - 1) / tb;
  if (norm) {
    TORCH_CHECK(norm_w.has_value() && norm_w->defined(), who,
                ": fp16 out (RMSNorm) needs norm_w");
    check_gpu(*norm_w, who, "norm_w");
    a.nw_f16 = is_f16(*norm_w, who, "norm_w");
    TORCH_CHECK(norm_w->dim() == 2 && norm_w->size(1) == H &&
                    (norm_w->size(0) == 1 || norm_w->size(0) == G),
                who, ": norm_w must be [1 or G, H]");
    check_vec_rows(*norm_w, who, "norm_w", 4);
    a.nw = norm_w->data_ptr();
    a.nw_s = norm_w->stride(0);
    a.nw_rows = (int)norm_w->size(0);
    TORCH_CHECK(scratch.has_value() && scratch->defined() &&
                    scratch->scalar_type() == at::kFloat &&
                    scratch->is_contiguous() && scratch->numel() >= G * T * H,
                who, ": scratch must be fp32 with >= G*T*H elements");
    TORCH_CHECK(
        partials.has_value() && partials->defined() &&
            partials->scalar_type() == at::kFloat &&
            partials->is_contiguous() && partials->numel() >= G * T * a.C,
        who, ": partials must be fp32 with >= G*T*ceil(H/128) elements");
    TORCH_CHECK(
        counters.has_value() && counters->defined() &&
            counters->scalar_type() == at::kInt && counters->is_contiguous() &&
            counters->numel() >= G * ntiles,
        who, ": counters must be int32 with >= G*ceil(T/TB) elements, zero");
    a.out32 = scratch->data_ptr<float>();
    a.out16 = reinterpret_cast<__half*>(out.data_ptr());
    a.part = partials->data_ptr<float>();
    a.counters = counters->data_ptr<int>();
  } else {
    a.out32 = out.data_ptr<float>();
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(m));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  const dim3 grid((unsigned)a.C, (unsigned)G, (unsigned)ntiles);
  const dim3 block(kWave);
  if (w16) {
    if (norm)
      launch_gate_mix<__half, true>(tb, grid, block, a, stream);
    else
      launch_gate_mix<__half, false>(tb, grid, block, a, stream);
  } else {
    if (norm)
      launch_gate_mix<float, true>(tb, grid, block, a, stream);
    else
      launch_gate_mix<float, false>(tb, grid, block, a, stream);
  }
}

// ---------------------------------------------------------------------------
// inkling2_glue_rms_norm / _embed_norm
// ---------------------------------------------------------------------------
static void launch_rms(int64_t T, int64_t H, const RmsArgs& a, const char* who,
                       hipStream_t stream) {
  int iter, block;
  row_launch_shape(H, who, iter, block);
  const dim3 grid((unsigned)T), blk(block);
  switch (iter) {
    case 1:
      inkling2_glue::rms_norm_kernel<1><<<grid, blk, 0, stream>>>(a);
      break;
    case 2:
      inkling2_glue::rms_norm_kernel<2><<<grid, blk, 0, stream>>>(a);
      break;
    case 3:
      inkling2_glue::rms_norm_kernel<3><<<grid, blk, 0, stream>>>(a);
      break;
    default:
      inkling2_glue::rms_norm_kernel<4><<<grid, blk, 0, stream>>>(a);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_glue_rms_norm(const at::Tensor& x,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& weight,
                            const at::Tensor& out,
                            const std::optional<at::Tensor>& sum_out,
                            double eps, bool round_fp16) {
  const char* who = "inkling2_glue_rms_norm";
  check_gpu(x, who, "x");
  check_gpu(out, who, "out");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && out.sizes() == x.sizes(), who,
              ": x and out must be [T, H]");
  const int64_t T = x.size(0), H = x.size(1);
  RmsArgs a{};
  a.x = x.data_ptr();
  a.x_f16 = is_f16(x, who, "x");
  a.x_t = x.stride(0);
  check_vec_rows(x, who, "x", 4);
  a.out = out.data_ptr();
  a.out_f16 = is_f16(out, who, "out");
  a.o_t = out.stride(0);
  check_vec_rows(out, who, "out", 4);
  a.H = (int)H;
  a.eps = (float)eps;
  a.inv_h = 1.0f / (float)H;
  a.round_h = round_fp16 ? 1 : 0;
  if (x1.has_value() && x1->defined()) {
    check_gpu(*x1, who, "x1");
    check_dtype(*x1, who, "x1", at::kFloat);
    TORCH_CHECK(x1->sizes() == x.sizes(), who, ": x1 must be [T, H]");
    check_vec_rows(*x1, who, "x1", 4);
    a.x1 = x1->data_ptr<float>();
    a.x1_t = x1->stride(0);
  }
  if (sum_out.has_value() && sum_out->defined()) {
    check_gpu(*sum_out, who, "sum_out");
    check_dtype(*sum_out, who, "sum_out", at::kFloat);
    TORCH_CHECK(sum_out->sizes() == x.sizes(), who, ": sum_out must be [T, H]");
    check_vec_rows(*sum_out, who, "sum_out", 4);
    a.sum_out = sum_out->data_ptr<float>();
    a.so_t = sum_out->stride(0);
  }
  if (weight.has_value() && weight->defined()) {
    check_gpu(*weight, who, "weight");
    a.w_f16 = is_f16(*weight, who, "weight");
    TORCH_CHECK(
        weight->dim() == 1 && weight->size(0) == H && weight->stride(0) == 1,
        who, ": weight must be a contiguous [H]");
    check_vec_rows(*weight, who, "weight", 4);
    a.w = weight->data_ptr();
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  launch_rms(T, H, a, who, stream);
}

void inkling2_glue_embed_norm(const at::Tensor& ids, const at::Tensor& table,
                              const at::Tensor& weight, const at::Tensor& out,
                              double eps) {
  const char* who = "inkling2_glue_embed_norm";
  check_gpu(ids, who, "ids");
  check_gpu(table, who, "table");
  check_gpu(weight, who, "weight");
  check_gpu(out, who, "out");
  TORCH_CHECK(ids.dim() == 1 && (ids.scalar_type() == at::kInt ||
                                 ids.scalar_type() == at::kLong),
              who, ": ids must be a 1-D int32 / int64 tensor");
  TORCH_CHECK(ids.stride(0) == 1, who, ": ids must be contiguous");
  TORCH_CHECK(table.dim() == 2, who, ": table must be [V, H]");
  const int64_t T = ids.size(0), H = table.size(1);
  RmsArgs a{};
  a.x = table.data_ptr();
  a.x_f16 = is_f16(table, who, "table");
  a.x_t = table.stride(0);
  check_vec_rows(table, who, "table", 4);
  a.ids = ids.data_ptr();
  a.ids_i64 = ids.scalar_type() == at::kLong ? 1 : 0;
  a.vocab = table.size(0);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == T && out.size(1) == H, who,
              ": out must be [T, H]");
  a.out = out.data_ptr();
  a.out_f16 = is_f16(out, who, "out");
  a.o_t = out.stride(0);
  check_vec_rows(out, who, "out", 4);
  a.w_f16 = is_f16(weight, who, "weight");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == H && weight.stride(0) == 1,
              who, ": weight must be a contiguous [H]");
  check_vec_rows(weight, who, "weight", 4);
  a.w = weight.data_ptr();
  a.H = (int)H;
  a.eps = (float)eps;
  a.inv_h = 1.0f / (float)H;
  a.round_h = 1;  // embed_norm returns the table dtype (fp16)
  TORCH_CHECK(a.x_f16, who, ": the embedding table must be fp16");
  const at::cuda::OptionalCUDAGuard guard(device_of(table));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  launch_rms(T, H, a, who, stream);
}

// ---------------------------------------------------------------------------
// inkling2_glue_swiglu
// ---------------------------------------------------------------------------
static unsigned elementwise_grid(int64_t work) {
  const int64_t blocks = (work + 255) / 256;
  return (unsigned)(blocks < 4096 ? (blocks < 1 ? 1 : blocks) : 4096);
}

void inkling2_glue_swiglu(const at::Tensor& gate_up, const at::Tensor& out,
                          double gate_cap, double up_cap) {
  const char* who = "inkling2_glue_swiglu";
  check_gpu(gate_up, who, "gate_up");
  check_gpu(out, who, "out");
  check_dtype(gate_up, who, "gate_up", at::kHalf);
  check_dtype(out, who, "out", at::kHalf);
  TORCH_CHECK(gate_up.dim() == 2 && out.dim() == 2 &&
                  gate_up.size(0) == out.size(0) &&
                  gate_up.size(1) == 2 * out.size(1) &&
                  gate_up.is_contiguous() && out.is_contiguous(),
              who, ": gate_up [R, 2I] and out [R, I] must be contiguous");
  const int64_t R = out.size(0), I = out.size(1);
  TORCH_CHECK(I % 8 == 0, who, ": intermediate size must be a multiple of 8");
  for (const at::Tensor* t : {&gate_up, &out})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, who,
                ": tensors must be 16-byte aligned");
  SwigluArgs a{};
  a.gate_up = reinterpret_cast<const __half*>(gate_up.data_ptr());
  a.out = reinterpret_cast<__half*>(out.data_ptr());
  a.I8 = (int)(I / 8);
  a.n8 = R * (I / 8);
  a.gate_cap = (float)gate_cap;
  a.up_cap = (float)up_cap;
  const at::cuda::OptionalCUDAGuard guard(device_of(gate_up));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (a.n8 == 0) return;
  inkling2_glue::swiglu_kernel<0>
      <<<dim3(elementwise_grid(a.n8)), dim3(256), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_glue_copy_states
// ---------------------------------------------------------------------------
void inkling2_glue_copy_states(const at::Tensor& src, const at::Tensor& dst) {
  const char* who = "inkling2_glue_copy_states";
  check_gpu(src, who, "src");
  check_gpu(dst, who, "dst");
  check_dtype(src, who, "src", at::kFloat);
  check_dtype(dst, who, "dst", at::kFloat);
  TORCH_CHECK((src.dim() == 2 || src.dim() == 3) && dst.sizes() == src.sizes(),
              who, ": src and dst must be [T, H] or [T, P, H] of one shape");
  check_vec_rows(src, who, "src", 4);
  check_vec_rows(dst, who, "dst", 4);
  TORCH_CHECK(src.data_ptr() != dst.data_ptr(), who,
              ": src and dst must not alias");
  const int64_t T = src.size(0), H = src.size(-1);
  const int64_t P = src.dim() == 3 ? src.size(1) : 1;
  CopyArgs a{};
  a.src = src.data_ptr<float>();
  a.dst = dst.data_ptr<float>();
  a.s_t = src.stride(0);
  a.d_t = dst.stride(0);
  a.s_p = src.dim() == 3 ? src.stride(1) : 0;
  a.d_p = dst.dim() == 3 ? dst.stride(1) : 0;
  a.P = (int)P;
  a.H = (int)H;
  const at::cuda::OptionalCUDAGuard guard(device_of(src));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T * P == 0) return;
  inkling2_glue::copy_states_kernel<0>
      <<<dim3((unsigned)(T * P)), dim3(256), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_glue_moe_route
// ---------------------------------------------------------------------------
template <typename WT>
static void launch_route(int tb, unsigned grid, size_t shmem,
                         const RouteArgs& a, hipStream_t stream) {
  switch (tb) {
    case 1:
      inkling2_glue::moe_route_kernel<WT, 1>
          <<<dim3(grid), dim3(kRouteThreads), shmem, stream>>>(a);
      break;
    case 2:
      inkling2_glue::moe_route_kernel<WT, 2>
          <<<dim3(grid), dim3(kRouteThreads), shmem, stream>>>(a);
      break;
    case 4:
      inkling2_glue::moe_route_kernel<WT, 4>
          <<<dim3(grid), dim3(kRouteThreads), shmem, stream>>>(a);
      break;
    default:
      inkling2_glue::moe_route_kernel<WT, 8>
          <<<dim3(grid), dim3(kRouteThreads), shmem, stream>>>(a);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_glue_moe_route(
    const at::Tensor& x, const at::Tensor& gate_w, const at::Tensor& bias,
    const at::Tensor& logits, const at::Tensor& topk_weights,
    const at::Tensor& topk_ids, const at::Tensor& sorted_ids,
    const at::Tensor& expert_ids, const at::Tensor& post_pad,
    const at::Tensor& counters, const std::optional<at::Tensor>& zero1,
    const std::optional<at::Tensor>& zero2, int64_t block_size_m,
    double route_scale) {
  const char* who = "inkling2_glue_moe_route";
  const std::pair<const at::Tensor*, const char*> on_gpu[] = {
      {&x, "x"},
      {&gate_w, "gate_w"},
      {&bias, "bias"},
      {&logits, "logits"},
      {&topk_weights, "topk_weights"},
      {&topk_ids, "topk_ids"},
      {&sorted_ids, "sorted_ids"},
      {&expert_ids, "expert_ids"},
      {&post_pad, "post_pad"},
      {&counters, "counters"}};
  for (const auto& p : on_gpu) check_gpu(*p.first, who, p.second);
  check_dtype(x, who, "x", at::kHalf);
  TORCH_CHECK(x.dim() == 2, who, ": x must be [T, H]");
  const int64_t T = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 8 == 0, who, ": hidden size must be a multiple of 8");
  check_vec_rows(x, who, "x", 8);
  const bool w16 = gate_w.scalar_type() == at::kHalf;
  if (!w16) check_dtype(gate_w, who, "gate_w", at::kFloat);
  TORCH_CHECK(
      gate_w.dim() == 2 && gate_w.size(1) == H && gate_w.is_contiguous(), who,
      ": gate_w must be a contiguous [E, H]");
  const int64_t E = gate_w.size(0);
  TORCH_CHECK(E >= 1 && E <= kMaxExperts, who, ": 1..", kMaxExperts,
              " experts");
  check_dtype(bias, who, "bias", at::kFloat);
  TORCH_CHECK(bias.numel() == E, who, ": bias must hold one value per expert");
  check_dtype(logits, who, "logits", at::kFloat);
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) == T && logits.size(1) == E &&
                  logits.is_contiguous(),
              who, ": logits must be contiguous [T, E]");
  check_dtype(topk_weights, who, "topk_weights", at::kFloat);
  check_dtype(topk_ids, who, "topk_ids", at::kInt);
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == T &&
                  topk_ids.sizes() == topk_weights.sizes() &&
                  topk_weights.is_contiguous() && topk_ids.is_contiguous(),
              who, ": topk_weights / topk_ids must be contiguous [T, K]");
  const int64_t K = topk_ids.size(1);
  TORCH_CHECK(K >= 1 && K <= kMaxTopK && K <= E, who,
              ": 1 <= top_k <= ", kMaxTopK);
  TORCH_CHECK(T * K <= kRouteMaxEntries, who, ": at most ", kRouteMaxEntries,
              " routed entries per call, got ", T * K);
  TORCH_CHECK(block_size_m == 1 || block_size_m == 2 || block_size_m == 4 ||
                  block_size_m == 8,
              who, ": block_size_m must be 1, 2, 4 or 8");
  check_dtype(sorted_ids, who, "sorted_ids", at::kInt);
  check_dtype(expert_ids, who, "expert_ids", at::kInt);
  check_dtype(post_pad, who, "post_pad", at::kInt);
  check_dtype(counters, who, "counters", at::kInt);
  TORCH_CHECK(sorted_ids.is_contiguous() && expert_ids.is_contiguous() &&
                  post_pad.numel() >= 1 && counters.is_contiguous(),
              who, ": sorted_ids / expert_ids / counters must be contiguous");
  const int64_t numel = T * K;
  const int64_t need_padded = numel + E * (block_size_m - 1);
  const int64_t cap_padded =
      numel < E ? std::min<int64_t>(numel * block_size_m, need_padded)
                : need_padded;
  TORCH_CHECK(sorted_ids.numel() >= cap_padded &&
                  expert_ids.numel() >= cap_padded / block_size_m,
              who, ": sorted_ids needs >= ", cap_padded,
              " and expert_ids >= ", cap_padded / block_size_m,
              " entries (moe_align_block_size sizes)");

  RouteArgs a{};
  a.x = reinterpret_cast<const __half*>(x.data_ptr());
  a.x_t = x.stride(0);
  a.w = gate_w.data_ptr();
  a.bias = bias.data_ptr<float>();
  a.logits = logits.data_ptr<float>();
  a.topk_w = topk_weights.data_ptr<float>();
  a.topk_ids = topk_ids.data_ptr<int32_t>();
  a.sorted_ids = sorted_ids.data_ptr<int32_t>();
  a.expert_ids = expert_ids.data_ptr<int32_t>();
  a.post_pad = post_pad.data_ptr<int32_t>();
  a.counters = counters.data_ptr<int32_t>();
  a.T = (int)T;
  a.H = (int)H;
  a.E = (int)E;
  a.K = (int)K;
  a.BM = (int)block_size_m;
  a.max_padded = (int)sorted_ids.numel();
  a.max_blocks = (int)expert_ids.numel();
  a.route_scale = (float)route_scale;
  int64_t zero_bytes = 0;
  const std::pair<const std::optional<at::Tensor>*, int> zeros[] = {
      {&zero1, 0}, {&zero2, 1}};
  for (const auto& zp : zeros) {
    const std::optional<at::Tensor>& z = *zp.first;
    if (!(z.has_value() && z->defined())) continue;
    check_gpu(*z, who, "zero");
    check_dtype(*z, who, "zero", at::kHalf);
    TORCH_CHECK(
        z->is_contiguous() && z->numel() % 8 == 0 &&
            reinterpret_cast<uintptr_t>(z->data_ptr()) % 16 == 0,
        who,
        ": zero buffers must be contiguous, 16-byte aligned, numel % 8 == 0");
    zero_bytes += z->numel() * 2;
    if (zp.second == 0) {
      a.zero1 = reinterpret_cast<__half*>(z->data_ptr());
      a.zero1_n = z->numel();
    } else {
      a.zero2 = reinterpret_cast<__half*>(z->data_ptr());
      a.zero2_n = z->numel();
    }
  }

  // tokens per CTA tile: fp32 weights stage x as fp32 (LDS), fp16 as fp16
  const int xsz = w16 ? 2 : 4;
  int tb = 1;
  const int max_tb = w16 ? 8 : 4;
  while (tb < max_tb && tb < T) tb <<= 1;
  while (tb > 1 && (size_t)tb * H * xsz > 48 * 1024) tb >>= 1;
  TORCH_CHECK((size_t)tb * H * xsz <= 48 * 1024, who,
              ": hidden size too large");
  a.NT = (int)((T + tb - 1) / tb);
  // CTAs per token tile: one expert row per wave at most, as few as fill the
  // GPU (every CTA stages its own x tile)
  const int64_t waves = kRouteThreads / kWave;
  const int64_t neg_max = (E + waves - 1) / waves;
  a.NEG = (int)std::clamp<int64_t>((kMinCtas + a.NT - 1) / a.NT, 1, neg_max);
  TORCH_CHECK(counters.numel() >= a.NT + 1, who,
              ": counters needs >= ", a.NT + 1, " int32 (zero on entry)");
  // the zeroing CTAs come first: they overlap with the GEMV of the others (few
  // of them, every CTA of this launch reserves the x tile of LDS)
  a.nzero = (int)std::min<int64_t>((zero_bytes + 16 * 256 - 1) / (16 * 256),
                                   kMaxZeroCtas);
  const unsigned grid = (unsigned)(a.nzero + (int64_t)a.NT * a.NEG);
  const size_t shmem = std::max<size_t>((size_t)tb * H * xsz, kAlignSmemBytes);

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  if (w16)
    launch_route<__half>(tb, grid, shmem, a, stream);
  else
    launch_route<float>(tb, grid, shmem, a, stream);
}
