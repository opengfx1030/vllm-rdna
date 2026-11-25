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
#include <cstdlib>
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
constexpr int kRouteThreads = 256;       // 8 waves, one expert row per wave
constexpr int kRouteSmallThreads = 128;  // 4 waves per expert CTA (T <= 4)
constexpr int kRouteMaxEntries = 8192;   // tokens * top_k of one call
// align scratch in the router's dynamic LDS: per-wave expert histograms
// [waves][256], the wave sums of the expert scan, the routed ids as bytes
constexpr int kAlignOffWs = (kRouteThreads / kWave) * 256 * 4;
constexpr int kAlignOffKeys = kAlignOffWs + 16 * 4;
constexpr int kAlignSmemBytes = kAlignOffKeys + kRouteMaxEntries;
constexpr int kGateWaves = 4;  // output rows of the down projection per CTA
constexpr int kMinCtas = 216;  // CTAs kept in flight (3 per CU of a 72 CU V620)
constexpr int kMaxZeroCtas = 32;  // router zero-fill CTAs (each reserves LDS)
constexpr int kStageV = 32;       // float4 per lane of a staged row: H <= 4096
constexpr int kMixWaves = 8;      // waves of a split gate_mix CTA
constexpr int kMixRpw = 8;        // rows of the gate rank per wave: R = 64

// ---------------------------------------------------------------------------
// Small helpers (wave32 DPP butterflies on gfx1030, bit-identical to shfl_xor)
// ---------------------------------------------------------------------------
#if defined(__HIP__RDNA2__)
template <int O>
__device__ __forceinline__ int xor_lane_i32(int x) {
  if constexpr (O == 16)
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false,
                                        false);
  else if constexpr (O == 8)
    return __builtin_amdgcn_update_dpp(x, x, 0x168, 0xF, 0xF, false);
  else if constexpr (O == 4)
    return __builtin_amdgcn_update_dpp(x, x, 0x164, 0xF, 0xF, false);
  else if constexpr (O == 2)
    return __builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false);
  else
    return __builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false);
}
#else
template <int O>
__device__ __forceinline__ int xor_lane_i32(int x) {
  return __shfl_xor(x, O);
}
#endif

template <int O>
__device__ __forceinline__ float xor_lane(float v) {
  return __int_as_float(xor_lane_i32<O>(__float_as_int(v)));
}

__device__ __forceinline__ float wave_sum(float v) {
  v += xor_lane<16>(v);
  v += xor_lane<8>(v);
  v += xor_lane<4>(v);
  v += xor_lane<2>(v);
  v += xor_lane<1>(v);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
  v = fmaxf(v, xor_lane<16>(v));
  v = fmaxf(v, xor_lane<8>(v));
  v = fmaxf(v, xor_lane<4>(v));
  v = fmaxf(v, xor_lane<2>(v));
  v = fmaxf(v, xor_lane<1>(v));
  return v;
}

template <int N>
__device__ __forceinline__ void wave_sum_n(float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] += xor_lane<16>(v[i]);
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] += xor_lane<8>(v[i]);
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] += xor_lane<4>(v[i]);
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] += xor_lane<2>(v[i]);
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] += xor_lane<1>(v[i]);
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

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = static_cast<uint32_t>(__float_as_int(f));
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);                      // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  return (uint32_t)f32_to_bf16_rne(lo) | ((uint32_t)f32_to_bf16_rne(hi) << 16);
}
__device__ __forceinline__ void store4_bf16(uint16_t* p, const float4 v) {
  uint2 u;
  u.x = pack_bf2(v.x, v.y);
  u.y = pack_bf2(v.z, v.w);
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
  void* norm_out2;  // optional [T, H] fp32 or bf16 secondary norm output (Q=1)
  int64_t no2_t;
  int no2_bf16;
  int H, n_blocks;
  float eps, norm_eps, inv_h, scale;
};

struct GlueConvMeta {
  const int32_t* block_table;  // [R, bt_stride]
  const int32_t* seq_idx;      // [T_md]
  const int32_t* qsl;          // [R + 1]
  const int32_t* seq_lens;     // [R]
  __half* cache;               // [blocks, nkv, ring, W] fp16
  int64_t cs0, cs1, cs2;       // strides in halfs
  int bt_stride, blk_shift, ring, T_md;
};

struct AttnResSconvArgs {
  AttnResArgs base;
  GlueConvMeta cm;
  const __half* cx;  // [T, H] fp16 conv input
  int64_t cx_t;
  const float* cw;   // [H, 1, 4] fp32 conv weight
  float* sconv_out;  // optional [T, H] fp32 conv output
  int64_t sco_t;
  int conv_width;  // H / num_kv_heads
  int conv_o0;     // stream_offset
  int conv_wr;     // 1 if write_state else 0
  float conv_inv;  // 1 / sqrt(5)
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
__device__ __forceinline__ void attn_res_core(const AttnResArgs& a,
                                              const float4 (&P)[ITER]) {
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

  const bool has_app = a.append != nullptr;
  const int NS = a.n_blocks + (has_app ? 1 : 0) + 1;
  float* brow = a.blocks != nullptr ? a.blocks + t * a.bs_t : nullptr;

  float4 A[ITER], Qv[Q][ITER];
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
    float4 ap = make_float4(0.f, 0.f, 0.f, 0.f);
    if (v < V && has_app) {
      ap = load4(a.append + t * a.ap_t + 4 * v);
      store4(brow + (int64_t)a.n_blocks * a.bs_s + 4 * v, ap);
    }
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

  // pass 1: issue all register-resident state loads together, then reduce
  float4 S[kNReg][ITER];
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
    if (j < NS) fetch(j, S[j]);
  }
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
    if (j < NS) {
      float v[kQ1];
      state_partials<Q, ITER>(S[j], Qv, v);
      wave_sum_n<kQ1>(v);
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
    wave_sum_n<kQ1>(v);
    if (lane == 0) {
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) s_red[wave][j * kQ1 + i] = v[i];
    }
  }
  __syncthreads();

  // Every wave independently reduces s_red across waves and computes softmax
  // coefficients in registers (eliminates s_tot / s_coef LDS and 2 barriers).
  float rr[2];
  float tot_q[Q][2];
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int j = lane + kWave * u;
    if (j < NS) {
      float s0 = 0.f;
      for (int w = 0; w < nwaves; ++w) s0 += s_red[w][j * kQ1];
      rr[u] = rsqrtf(s0 * a.inv_h + a.eps);
  #pragma unroll
      for (int q = 0; q < Q; ++q) {
        float sq = 0.f;
        for (int w = 0; w < nwaves; ++w) sq += s_red[w][j * kQ1 + 1 + q];
        tot_q[q][u] = sq;
      }
    } else {
      rr[u] = 0.f;
  #pragma unroll
      for (int q = 0; q < Q; ++q) tot_q[q][u] = 0.f;
    }
  }
  float coef[Q][2];
  #pragma unroll
  for (int q = 0; q < Q; ++q) {
    float sc[2], e[2];
  #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = lane + kWave * u;
      sc[u] = j < NS ? (tot_q[q][u] * rr[u]) * a.scale : -INFINITY;
    }
    const float m = wave_max(fmaxf(sc[0], sc[1]));
  #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = lane + kWave * u;
      e[u] = j < NS ? expf(sc[u] - m) : 0.f;
    }
    const float den = wave_sum(e[0] + e[1]);
  #pragma unroll
    for (int u = 0; u < 2; ++u) coef[q][u] = (e[u] / den) * rr[u];
  }

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
        const float c = __shfl(coef[q][0], j);
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
      const float c =
          (j < kWave) ? __shfl(coef[q][0], j) : __shfl(coef[q][1], j - kWave);
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
      ssq[q] = s;
    }
    wave_sum_n<Q>(ssq);
    if (lane == 0) {
  #pragma unroll
      for (int q = 0; q < Q; ++q) s_nrm[wave][q] = ssq[q];
    }
    // Prefetch norm weights while waves arrive at the single norm barrier.
    float4 nw[Q][ITER];
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      if (q == 0 || a.nw_rows > 1) {
        const int64_t wrow = (a.nw_rows == 1 ? 0 : q) * a.nw_s;
  #pragma unroll
        for (int k = 0; k < ITER; ++k) {
          const int v = tid + k * nth;
          nw[q][k] = v < V ? load4_dyn(a.norm_w, wrow + 4 * v, a.nw_f16)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
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
  #pragma unroll
      for (int k = 0; k < ITER; ++k) {
        const int v = tid + k * nth;
        if (v < V) {
          const float4 w = (a.nw_rows == 1) ? nw[0][k] : nw[q][k];
          const float4 y = make_float4(
              w.x * (acc[q][k].x * rs[q]), w.y * (acc[q][k].y * rs[q]),
              w.z * (acc[q][k].z * rs[q]), w.w * (acc[q][k].w * rs[q]));
          store4(out + (int64_t)q * a.o_q + 4 * v, y);
          if constexpr (Q == 1) {
            if (a.norm_out2 != nullptr) {
              if (a.no2_bf16) {
                store4_bf16(reinterpret_cast<uint16_t*>(a.norm_out2) +
                                t * a.no2_t + 4 * v,
                            y);
              } else {
                store4(
                    reinterpret_cast<float*>(a.norm_out2) + t * a.no2_t + 4 * v,
                    y);
              }
            }
          }
        }
      }
    }
  }
}

template <int Q, int ITER, bool NORM>
__device__ __forceinline__ void attn_res_body(const AttnResArgs& a) {
  const int tid = threadIdx.x;
  const int nth = blockDim.x;
  const int64_t t = blockIdx.x;
  const int V = a.H >> 2;
  const bool has_x1 = a.x1 != nullptr;
  float4 P[ITER];
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
    float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
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
    }
    P[k] = p;
  }
  attn_res_core<Q, ITER, NORM>(a, P);
}

// SCONV_MODE 1: partial = x0 + sconv(cx)
// SCONV_MODE 2: partial = sconv(cx) (boundary layer mlp_input, x0 == nullptr)
// Computing P[ITER] in an isolated prologue scope before loading Qv / S keeps
// conv weights + history taps dead during the state reduction (~104 VGPRs vs
// 164 VGPRs when all loads overlap).
template <int Q, int ITER, bool NORM, int SCONV_MODE>
__device__ __forceinline__ void attn_res_sconv_body(
    const AttnResSconvArgs& sa) {
  const AttnResArgs& a = sa.base;
  const int tid = threadIdx.x;
  const int nth = blockDim.x;
  const int64_t t = blockIdx.x;
  const int V = a.H >> 2;
  float4 P[ITER];
  {
    int cr = -1;
    int qs = 0, tp_p = 0, tp_p1 = 0;
    bool keep = false;
    int blk_cur = 0;
    bool h_use_x[3] = {false, false, false};
    bool h_valid[3] = {false, false, false};
    int64_t h_base[3] = {0, 0, 0};
    if (t < sa.cm.T_md) {
      cr = sa.cm.seq_idx[t];
      if (cr >= 0) {
        qs = sa.cm.qsl[cr];
        tp_p1 = sa.cm.seq_lens[cr];
        tp_p = tp_p1 - (sa.cm.qsl[cr + 1] - qs) + ((int)t - qs);
        if (sa.conv_wr && tp_p >= 0 && tp_p < tp_p1) {
          const int blk_end = ((tp_p >> sa.cm.blk_shift) + 1)
                              << sa.cm.blk_shift;
          keep = (tp_p + sa.cm.ring >= min(tp_p1, blk_end));
        }
        const int bt_row = cr * sa.cm.bt_stride;
        if (keep) {
          blk_cur =
              sa.cm.block_table[bt_row + (max(tp_p, 0) >> sa.cm.blk_shift)];
        }
  #pragma unroll
        for (int lag = 1; lag < 4; ++lag) {
          const int sr = (int)t - lag;
          const int sp = tp_p - lag;
          const bool ux = sr >= qs;
          h_use_x[lag - 1] = ux;
          h_valid[lag - 1] = ux || sp >= 0;
          if (ux) {
            h_base[lag - 1] = (int64_t)sr * sa.cx_t;
          } else if (sp >= 0) {
            const int b = sa.cm.block_table[bt_row + (sp >> sa.cm.blk_shift)];
            h_base[lag - 1] = (int64_t)b * sa.cm.cs0 +
                              (int64_t)(sp % sa.cm.ring) * sa.cm.cs2 +
                              sa.conv_o0;
          }
        }
      }
    }

    uint2 xh[ITER];
    uint2 hh[ITER][3];
    float4 wc[ITER][4];
    float4 x0v[ITER];
  #pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int v = tid + k * nth;
      xh[k] = make_uint2(0u, 0u);
      hh[k][0] = hh[k][1] = hh[k][2] = make_uint2(0u, 0u);
      x0v[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (v < V) {
        const int c = 4 * v;
        if constexpr (SCONV_MODE == 1) {
          x0v[k] = load4(a.x0 + t * a.x0_t + c);
        }
        if (cr >= 0) {
          xh[k] = *reinterpret_cast<const uint2*>(sa.cx + t * sa.cx_t + c);
          const int h = c / sa.conv_width;
          const int jch = c - h * sa.conv_width;
  #pragma unroll
          for (int i = 0; i < 3; ++i) {
            if (h_valid[i]) {
              const __half* src = h_use_x[i] ? (sa.cx + h_base[i] + c)
                                             : (sa.cm.cache + h_base[i] +
                                                (int64_t)h * sa.cm.cs1 + jch);
              hh[k][i] = *reinterpret_cast<const uint2*>(src);
            }
          }
          const float* wp = sa.cw + (int64_t)c * 4;
          wc[k][0] = load4(wp);
          wc[k][1] = load4(wp + 4);
          wc[k][2] = load4(wp + 8);
          wc[k][3] = load4(wp + 12);
        }
      }
    }

  #pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int v = tid + k * nth;
      float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
      if (v < V) {
        const int c = 4 * v;
        float4 cv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (cr >= 0) {
          const float4 t0 = make_float4(h2f_lo(xh[k].x), h2f_hi(xh[k].x),
                                        h2f_lo(xh[k].y), h2f_hi(xh[k].y));
          const float4 t1 = make_float4(h2f_lo(hh[k][0].x), h2f_hi(hh[k][0].x),
                                        h2f_lo(hh[k][0].y), h2f_hi(hh[k][0].y));
          const float4 t2 = make_float4(h2f_lo(hh[k][1].x), h2f_hi(hh[k][1].x),
                                        h2f_lo(hh[k][1].y), h2f_hi(hh[k][1].y));
          const float4 t3 = make_float4(h2f_lo(hh[k][2].x), h2f_hi(hh[k][2].x),
                                        h2f_lo(hh[k][2].y), h2f_hi(hh[k][2].y));
          cv.x = t0.x * wc[k][0].w + t1.x * wc[k][0].z + t2.x * wc[k][0].y +
                 t3.x * wc[k][0].x;
          cv.y = t0.y * wc[k][1].w + t1.y * wc[k][1].z + t2.y * wc[k][1].y +
                 t3.y * wc[k][1].x;
          cv.z = t0.z * wc[k][2].w + t1.z * wc[k][2].z + t2.z * wc[k][2].y +
                 t3.z * wc[k][2].x;
          cv.w = t0.w * wc[k][3].w + t1.w * wc[k][3].z + t2.w * wc[k][3].y +
                 t3.w * wc[k][3].x;
          cv.x = (cv.x + t0.x) * sa.conv_inv;
          cv.y = (cv.y + t0.y) * sa.conv_inv;
          cv.z = (cv.z + t0.z) * sa.conv_inv;
          cv.w = (cv.w + t0.w) * sa.conv_inv;
          if (keep && blk_cur > 0) {
            const int h = c / sa.conv_width;
            const int jch = c - h * sa.conv_width;
            __half* dst = sa.cm.cache + (int64_t)blk_cur * sa.cm.cs0 +
                          (int64_t)h * sa.cm.cs1 +
                          (int64_t)(tp_p % sa.cm.ring) * sa.cm.cs2 +
                          sa.conv_o0 + jch;
            store4(dst, t0);
          }
        }
        if (sa.sconv_out != nullptr) {
          store4(sa.sconv_out + t * sa.sco_t + c, cv);
        }
        if constexpr (SCONV_MODE == 1) {
          p = make_float4(x0v[k].x + cv.x, x0v[k].y + cv.y, x0v[k].z + cv.z,
                          x0v[k].w + cv.w);
        } else {
          p = cv;
        }
        if (a.partial_out != nullptr) {
          store4(a.partial_out + t * a.po_t + c, p);
        }
      }
      P[k] = p;
    }
  }
  attn_res_core<Q, ITER, NORM>(a, P);
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int Q, int ITER, bool NORM>
__global__ void __launch_bounds__(256) attn_res_kernel(const AttnResArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  attn_res_body<Q, ITER, NORM>(a);
#endif
}

template <int Q, int ITER, bool NORM, int SCONV_MODE>
__global__ void __launch_bounds__(256)
    attn_res_sconv_kernel(const AttnResSconvArgs sa) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  attn_res_sconv_body<Q, ITER, NORM, SCONV_MODE>(sa);
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

// gate_mix for a handful of tokens (decode, speculative verification) with fp32
// weights.  The R rows of the projections are split over the kMixWaves waves of
// the CTA, so every wave has 2 * RPW float4 weight loads in flight at once: the
// one-wave kernel above walks all R rows in dependent groups of 8 loads and is
// latency bound (54 waves on a 72 CU GPU at T = 1).  The partial sums of the
// waves are added in wave order through LDS, so the result differs from the
// sequential kernel only in the order of the fp32 additions.  The last-arriving
// CTA of a (gate, token tile) normalises, with its waves split over the tokens
// and the columns.
template <int TB, int RPW, bool NORM>
__device__ __forceinline__ void gate_mix_split_body(const GateMixArgs& a) {
  constexpr int NW = kMixWaves;
  constexpr int WPT = NW / TB;  // waves per token of the normalisation
  __shared__ int s_last;
  __shared__ float4 s_part[NW][TB][2][kWave];
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int s = blockIdx.y;
  const int tile = blockIdx.z;
  const int t0 = tile * TB;
  const int cc = blockIdx.x;  // 128-column chunk
  const int c = cc * 128 + lane * 4;
  const bool active = c < a.H;
  const int H = a.H, R = a.R, T = a.T;
  const int r0 = wave * RPW;

  // every load of the CTA is in flight before the first one is used
  float4 fv[RPW], gv[RPW];
  if (active) {
    const float* wf =
        reinterpret_cast<const float*>(a.wf) + (int64_t)s * R * H + c;
    const float* wg =
        reinterpret_cast<const float*>(a.wg) + (int64_t)s * R * H + c;
  #pragma unroll
    for (int i = 0; i < RPW; ++i) {
      fv[i] = load4(wf + (int64_t)(r0 + i) * H);
      gv[i] = load4(wg + (int64_t)(r0 + i) * H);
    }
  }
  float dd[TB][RPW];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    const int tc = min(t0 + tb, T - 1);
  #pragma unroll
    for (int i = 0; i < RPW; ++i)
      dd[tb][i] = __half2float(a.d[((int64_t)s * T + tc) * R + r0 + i]);
  }
  float4 hv = make_float4(0.f, 0.f, 0.f, 0.f), mv = hv;
  if (wave < TB && active && t0 + wave < T) {  // token `wave` of the epilogue
    hv = load4(a.h + (t0 + wave) * a.h_t + (int64_t)s * a.h_s + c);
    mv = load4(a.m + (t0 + wave) * a.m_t + (int64_t)s * a.m_s + c);
  }

  float accf[TB][4], accg[TB][4];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb)
  #pragma unroll
    for (int j = 0; j < 4; ++j) accf[tb][j] = accg[tb][j] = 0.f;
  if (active) {
  #pragma unroll
    for (int i = 0; i < RPW; ++i) {
  #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const float dr = dd[tb][i];
        accf[tb][0] = fmaf(dr, fv[i].x, accf[tb][0]);
        accf[tb][1] = fmaf(dr, fv[i].y, accf[tb][1]);
        accf[tb][2] = fmaf(dr, fv[i].z, accf[tb][2]);
        accf[tb][3] = fmaf(dr, fv[i].w, accf[tb][3]);
        accg[tb][0] = fmaf(dr, gv[i].x, accg[tb][0]);
        accg[tb][1] = fmaf(dr, gv[i].y, accg[tb][1]);
        accg[tb][2] = fmaf(dr, gv[i].z, accg[tb][2]);
        accg[tb][3] = fmaf(dr, gv[i].w, accg[tb][3]);
      }
    }
  }
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    s_part[wave][tb][0][lane] =
        make_float4(accf[tb][0], accf[tb][1], accf[tb][2], accf[tb][3]);
    s_part[wave][tb][1][lane] =
        make_float4(accg[tb][0], accg[tb][1], accg[tb][2], accg[tb][3]);
  }
  __syncthreads();

  // mix of token `wave`: out = (1 - tanh(f)) * h + tanh(g) * m
  if (wave < TB) {
    const int t = t0 + wave;
    float4 f = make_float4(0.f, 0.f, 0.f, 0.f), g = f;
  #pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 pf = s_part[w][wave][0][lane];
      const float4 pg = s_part[w][wave][1][lane];
      f.x += pf.x;
      f.y += pf.y;
      f.z += pf.z;
      f.w += pf.w;
      g.x += pg.x;
      g.y += pg.y;
      g.z += pg.z;
      g.w += pg.w;
    }
    float ss = 0.f;
    if (active && t < T) {
      float4 o;
      o.x = (1.0f - tanhf(f.x)) * hv.x + tanhf(g.x) * mv.x;
      o.y = (1.0f - tanhf(f.y)) * hv.y + tanhf(g.y) * mv.y;
      o.z = (1.0f - tanhf(f.z)) * hv.z + tanhf(g.z) * mv.z;
      o.w = (1.0f - tanhf(f.w)) * hv.w + tanhf(g.w) * mv.w;
      store4(a.out32 + ((int64_t)s * T + t) * H + c, o);
      ss = dot4(o, o);
    }
    if constexpr (NORM) {
      const float tot = wave_sum(ss);
      if (lane == 0 && t < T && cc < a.C)
        a.part[((int64_t)s * T + t) * a.C + cc] = tot;
    }
  }
  if constexpr (!NORM) return;

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

  {
    const int tb = wave / WPT, sw = wave - tb * WPT;
    const int t = t0 + tb;
    if (t < T) {
      const float* pp = a.part + ((int64_t)s * T + t) * a.C;
      float tot = 0.f;
      for (int i = lane; i < a.C; i += kWave) tot += pp[i];
      tot = wave_sum(tot);
      const float rs = rsqrtf(tot * a.inv_h + a.norm_eps);
      const int64_t row = ((int64_t)s * T + t) * H;
      const int64_t wrow = (a.nw_rows == 1 ? 0 : s) * a.nw_s;
      for (int c4 = sw * kWave + lane; c4 < (H >> 2); c4 += WPT * kWave) {
        const float4 v = load4(a.out32 + row + 4 * c4);
        const float4 w = load4_dyn(a.nw, wrow + 4 * c4, a.nw_f16);
        store4(a.out16 + row + 4 * c4,
               make_float4(w.x * (v.x * rs), w.y * (v.y * rs), w.z * (v.z * rs),
                           w.w * (v.w * rs)));
      }
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

template <int TB, int RPW, bool NORM>
__global__ void __launch_bounds__(kMixWaves* kWave)
    gate_mix_split_kernel(const GateMixArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  gate_mix_split_body<TB, RPW, NORM>(a);
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
// embed_lookup: the rows of a vocabulary-sharded embedding table that this rank
// owns.  The ids of the other ranks give zero rows; the all-reduce of the
// tensor-parallel group adds the owner's row back (exactly: x + 0 = x).
// ---------------------------------------------------------------------------
struct LookupArgs {
  const uint4* table;  // [rows, H / 8] 16-byte units: this rank's shard
  const void* ids;     // [T] int32 / int64
  int ids_i64;
  int64_t first;   // vocabulary id of the shard's row 0
  int64_t rows;    // rows of the shard
  int64_t total;   // ids are clamped to [0, total) like embed_norm
  int64_t t_row8;  // table row stride in 16-byte units
  uint4* out;      // [T, H / 8] fp16
  int64_t o_row8;
  int H8;
};

template <int DUMMY = 0>
__global__ void __launch_bounds__(256) embed_lookup_kernel(const LookupArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int64_t t = blockIdx.x;
  int64_t id = a.ids_i64 ? reinterpret_cast<const int64_t*>(a.ids)[t]
                         : (int64_t)reinterpret_cast<const int32_t*>(a.ids)[t];
  id = id < 0 ? 0 : (id >= a.total ? a.total - 1 : id);
  const int64_t r = id - a.first;
  uint4* dst = a.out + t * a.o_row8;
  if (r >= 0 && r < a.rows) {
    const uint4* src = a.table + r * a.t_row8;
    for (int i = threadIdx.x; i < a.H8; i += blockDim.x) dst[i] = src[i];
  } else {
    for (int i = threadIdx.x; i < a.H8; i += blockDim.x)
      dst[i] = make_uint4(0u, 0u, 0u, 0u);
  }
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
// copy_states: [T, P, H] fp32 -> [T, P, H] fp32 or bf16 (snapshots of
// persistent states and DSpark partial-block taps)
// ---------------------------------------------------------------------------
struct CopyArgs {
  const float* src;
  void* dst;
  int64_t s_t, s_p, d_t, d_p;  // row strides in elements
  int P, H;
};

template <bool DST_BF16 = false>
__global__ void __launch_bounds__(256) copy_states_kernel(const CopyArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int t = blockIdx.x / a.P, p = blockIdx.x - t * a.P;
  const float4* s =
      reinterpret_cast<const float4*>(a.src + t * a.s_t + p * a.s_p);
  if constexpr (DST_BF16) {
    uint16_t* d = reinterpret_cast<uint16_t*>(a.dst) + (int64_t)t * a.d_t +
                  (int64_t)p * a.d_p;
    for (int i = threadIdx.x; i < (a.H >> 2); i += 256)
      store4_bf16(d + 4 * i, s[i]);
  } else {
    float4* d =
        reinterpret_cast<float4*>(reinterpret_cast<float*>(a.dst) +
                                  (int64_t)t * a.d_t + (int64_t)p * a.d_p);
    for (int i = threadIdx.x; i < (a.H >> 2); i += 256) d[i] = s[i];
  }
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
// When `flat_ids_out != nullptr`, also writes the selected expert ids as bytes
// into LDS (`flat_ids_out[t * K + k]`) so small-T align avoids a global
// readback.
__device__ __forceinline__ void route_topk_wave(
    const RouteArgs& a, int t, uint8_t* flat_ids_out = nullptr) {
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
  auto step_max = [](float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  };
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
    step_max(bv, bi, xor_lane<16>(bv), xor_lane_i32<16>(bi));
    step_max(bv, bi, xor_lane<8>(bv), xor_lane_i32<8>(bi));
    step_max(bv, bi, xor_lane<4>(bv), xor_lane_i32<4>(bi));
    step_max(bv, bi, xor_lane<2>(bv), xor_lane_i32<2>(bi));
    step_max(bv, bi, xor_lane<1>(bv), xor_lane_i32<1>(bi));
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
    const float sval = __shfl(mine, bi & (kWave - 1));
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
      const int id = s_sel_id[wave][k];
      a.topk_ids[(int64_t)t * a.K + k] = id;
      a.topk_w[(int64_t)t * a.K + k] = (s_sel_w[wave][k] / den) * a.route_scale;
      if (flat_ids_out != nullptr) flat_ids_out[t * a.K + k] = (uint8_t)id;
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

// Small-T router (T <= 8, decode & speculative verification): 1 expert per CTA
// (4 waves = 128 threads splitting H across all E CTAs -> 128 CTAs across 72
// CUs instead of 16, zero dynamic LDS, direct cached x reads), ONE global
// barrier, and an in-LDS O(T*K) align without histogram/ballot/sort passes.
template <typename WT, int TB>
__device__ __forceinline__ void moe_route_small_body(const RouteArgs& a) {
  __shared__ float s_part[4][TB];
  __shared__ int s_flag;
  __shared__ uint8_t s_flat_ids[8 * kMaxTopK];
  __shared__ int s_ws[8];
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int bid = blockIdx.x;

  if (bid < a.nzero) {
    const int64_t n1 = a.zero1_n >> 3, n2 = a.zero2_n >> 3;
    for (int64_t i = (int64_t)bid * kRouteSmallThreads + tid; i < n1 + n2;
         i += (int64_t)a.nzero * kRouteSmallThreads) {
      uint4* p = i < n1 ? reinterpret_cast<uint4*>(a.zero1) + i
                        : reinterpret_cast<uint4*>(a.zero2) + (i - n1);
      *p = make_uint4(0u, 0u, 0u, 0u);
    }
    return;
  }

  const int e = bid - a.nzero;
  const int H = a.H, E = a.E;
  float acc[TB];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) acc[tb] = 0.f;

  if constexpr (std::is_same_v<WT, float>) {
    const int V = H >> 2;
    const float4* w4 = reinterpret_cast<const float4*>(a.w) + (int64_t)e * V;
  #pragma unroll 4
    for (int i = tid; i < V; i += kRouteSmallThreads) {
      const float4 wv = w4[i];
  #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        if (tb < a.T) {
          const uint2 u = *reinterpret_cast<const uint2*>(
              a.x + (int64_t)tb * a.x_t + 4 * i);
          acc[tb] = fmaf(wv.x, h2f_lo(u.x), acc[tb]);
          acc[tb] = fmaf(wv.y, h2f_hi(u.x), acc[tb]);
          acc[tb] = fmaf(wv.z, h2f_lo(u.y), acc[tb]);
          acc[tb] = fmaf(wv.w, h2f_hi(u.y), acc[tb]);
        }
      }
    }
  } else {
    const int H8 = H >> 3;
    const uint4* w8 = reinterpret_cast<const uint4*>(a.w) + (int64_t)e * H8;
  #pragma unroll 4
    for (int i = tid; i < H8; i += kRouteSmallThreads) {
      const uint4 wv = w8[i];
      const half2* wh = reinterpret_cast<const half2*>(&wv);
  #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        if (tb < a.T) {
          const uint4 xv = *reinterpret_cast<const uint4*>(
              a.x + (int64_t)tb * a.x_t + 8 * i);
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
  }

  wave_sum_n<TB>(acc);
  if (lane == 0) {
  #pragma unroll
    for (int tb = 0; tb < TB; ++tb) s_part[wave][tb] = acc[tb];
  }
  __syncthreads();

  if (wave == 0) {
    if (lane < TB && lane < a.T) {
      const float sum = (s_part[0][lane] + s_part[1][lane]) +
                        (s_part[2][lane] + s_part[3][lane]);
      a.logits[(int64_t)lane * E + e] = sum;
    }
    __threadfence();
    if (lane == 0) {
      const int old = atomicAdd(a.counters, 1);
      s_flag = (old == E - 1);
    }
  }
  __syncthreads();
  if (!s_flag) return;
  __threadfence();

  // Last-arriver CTA: top-k per token (4 waves cover up to 8 tokens) + direct
  // small-numel align
  for (int t = wave; t < a.T; t += 4) route_topk_wave(a, t, s_flat_ids);
  const int numel = a.T * a.K;
  for (int p = tid; p < a.max_padded; p += kRouteSmallThreads)
    a.sorted_ids[p] = numel;
  __syncthreads();

  const int BM = a.BM;
  const int e0 = tid;
  const int e1 = tid + kRouteSmallThreads;
  int cnt0 = 0, cnt1 = 0;
  for (int i = 0; i < numel; ++i) {
    const int key = (int)s_flat_ids[i];
    cnt0 += (key == e0) ? 1 : 0;
    cnt1 += (key == e1) ? 1 : 0;
  }
  const int nb0 = (e0 < E) ? (cnt0 + BM - 1) / BM : 0;
  const int nb1 = (e1 < E) ? (cnt1 + BM - 1) / BM : 0;
  int v0 = nb0, v1 = nb1;
  #pragma unroll
  for (int d = 1; d < kWave; d <<= 1) {
    const int n0 = __shfl_up(v0, d);
    const int n1 = __shfl_up(v1, d);
    if (lane >= d) {
      v0 += n0;
      v1 += n1;
    }
  }
  if (lane == kWave - 1) {
    s_ws[wave] = v0;
    s_ws[4 + wave] = v1;
  }
  __syncthreads();
  int base0 = 0, base1 = 0, total = 0;
  #pragma unroll
  for (int w = 0; w < 8; ++w) {
    if (w < wave) base0 += s_ws[w];
    if (w < 4 + wave) base1 += s_ws[w];
    total += s_ws[w];
  }
  if (nb0 > 0) {
    const int fb = base0 + v0 - nb0;
    for (int b = 0; b < nb0; ++b) a.expert_ids[fb + b] = e0;
    int pos = fb * BM;
    for (int i = 0; i < numel; ++i) {
      if ((int)s_flat_ids[i] == e0) a.sorted_ids[pos++] = i;
    }
  }
  if (nb1 > 0) {
    const int fb = base1 + v1 - nb1;
    for (int b = 0; b < nb1; ++b) a.expert_ids[fb + b] = e1;
    int pos = fb * BM;
    for (int i = 0; i < numel; ++i) {
      if ((int)s_flat_ids[i] == e1) a.sorted_ids[pos++] = i;
    }
  }
  for (int b = total + tid; b < a.max_blocks; b += kRouteSmallThreads)
    a.expert_ids[b] = -1;
  if (tid == 0) {
    a.post_pad[0] = total * BM;
    a.counters[0] = 0;
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <typename WT, int TB>
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_kernel(const RouteArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  moe_route_body<WT, TB>(a);
#endif
}

template <typename WT, int TB>
__global__ void __launch_bounds__(kRouteSmallThreads)
    moe_route_small_kernel(const RouteArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  moe_route_small_body<WT, TB>(a);
#endif
}

// ---------------------------------------------------------------------------
// LoopCD (arXiv:2610.02185): hidden-state blend, gated readout, metadata prep,
// and 2-pass contrastive logits
// ---------------------------------------------------------------------------
struct LoopcdBlendArgs {
  float* x0;
  int64_t x0_t;
  float* x1;
  int64_t x1_t;
  const float* px0;
  int64_t px0_t;
  const float* px1;
  int64_t px1_t;
  float* blk;
  int64_t b_t, b_p;
  const float* pblk;
  int64_t pb_t, pb_p;
  const float* omega;
  const float* omega_blk;
  int P, H;
};

template <int DUMMY = 0>
__global__ void __launch_bounds__(256)
    loopcd_blend_kernel(const LoopcdBlendArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int64_t t = blockIdx.x;
  const int slot = blockIdx.y;
  if (slot == 0) {
    const float w = a.omega[t];
    if (w == 0.0f) return;
    for (int i = threadIdx.x; i < (a.H >> 2); i += 256) {
      float4 hf = load4(a.x0 + t * a.x0_t + 4 * i);
      if (a.x1 != nullptr) {
        const float4 qf = load4(a.x1 + t * a.x1_t + 4 * i);
        hf = make_float4(hf.x + qf.x, hf.y + qf.y, hf.z + qf.z, hf.w + qf.w);
      }
      float4 hp = load4(a.px0 + t * a.px0_t + 4 * i);
      const float4 qp = load4(a.px1 + t * a.px1_t + 4 * i);
      hp = make_float4(hp.x + qp.x, hp.y + qp.y, hp.z + qp.z, hp.w + qp.w);
      const float4 out =
          make_float4(fmaf(w, hf.x - hp.x, hf.x), fmaf(w, hf.y - hp.y, hf.y),
                      fmaf(w, hf.z - hp.z, hf.z), fmaf(w, hf.w - hp.w, hf.w));
      store4(a.x0 + t * a.x0_t + 4 * i, out);
      if (a.x1 != nullptr) {
        store4(a.x1 + t * a.x1_t + 4 * i, make_float4(0.f, 0.f, 0.f, 0.f));
      }
    }
  } else {
    const int p = slot - 1;
    if (p < 0 || p >= a.P) return;
    const float wb = (a.omega_blk != nullptr) ? a.omega_blk[t] : a.omega[t];
    if (wb == 0.0f) return;
    for (int i = threadIdx.x; i < (a.H >> 2); i += 256) {
      const float4 bf = load4(a.blk + t * a.b_t + (int64_t)p * a.b_p + 4 * i);
      const float4 bp =
          load4(a.pblk + t * a.pb_t + (int64_t)p * a.pb_p + 4 * i);
      store4(a.blk + t * a.b_t + (int64_t)p * a.b_p + 4 * i,
             make_float4(
                 fmaf(wb, bf.x - bp.x, bf.x), fmaf(wb, bf.y - bp.y, bf.y),
                 fmaf(wb, bf.z - bp.z, bf.z), fmaf(wb, bf.w - bp.w, bf.w)));
    }
  }
#endif
}

template <int ITER>
__global__ void __launch_bounds__(256)
    loopcd_readout_kernel(const AttnResArgs a, const int32_t* gate) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  if (gate != nullptr && gate[0] == 0) return;
  attn_res_body<1, ITER, true>(a);
#endif
}

struct LoopcdPrepHiddenArgs {
  float* omega_out;
  float* omega_blk_out;
  const void* idx;
  const int32_t* qsl;
  const int32_t* seq_lens;
  const int32_t* prefill_len;
  const float* req_alpha;
  const int32_t* req_mode;
  const int32_t* req_blend_blk;
  int num_reqs;
  int max_reqs;
  int num_tokens_padded;
  int ids_i64;
};

template <int DUMMY = 0>
__global__ void __launch_bounds__(256)
    loopcd_prepare_hidden_kernel(const LoopcdPrepHiddenArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int r = blockIdx.x;
  if (r < a.num_reqs) {
    const int qs = a.qsl[r];
    const int qe = a.qsl[r + 1];
    const int qlen = qe - qs;
    const int64_t s = a.ids_i64
                          ? reinterpret_cast<const int64_t*>(a.idx)[r]
                          : (int64_t)reinterpret_cast<const int32_t*>(a.idx)[r];
    if (s < 0 || s >= a.max_reqs) {
      for (int k = threadIdx.x; k < qlen; k += 256) {
        const int t = qs + k;
        if (t >= 0 && t < a.num_tokens_padded) {
          a.omega_out[t] = 0.0f;
          a.omega_blk_out[t] = 0.0f;
        }
      }
      return;
    }
    const int slen = a.seq_lens[r];
    const int plen = a.prefill_len[s];
    const float alpha = (a.req_mode[s] == 1) ? a.req_alpha[s] : 0.0f;
    const bool bflag = a.req_blend_blk[s] != 0;
    for (int k = threadIdx.x; k < qlen; k += 256) {
      const int t = qs + k;
      if (t >= 0 && t < a.num_tokens_padded) {
        const int pos = (slen - qlen) + k;
        const float w = (pos < plen - 1) ? 0.0f : alpha;
        a.omega_out[t] = w;
        a.omega_blk_out[t] = bflag ? w : 0.0f;
      }
    }
  } else {
    int end = (a.num_reqs > 0) ? a.qsl[a.num_reqs] : 0;
    if (end < 0) end = 0;
    for (int t = end + (int)threadIdx.x; t < a.num_tokens_padded; t += 256) {
      a.omega_out[t] = 0.0f;
      a.omega_blk_out[t] = 0.0f;
    }
  }
#endif
}

struct LoopcdLogitsArgs {
  float* zf;
  int64_t zf_s;
  const float* zw;
  int64_t zw_s;
  const float* omega;
  const float* log_beta;
  const void* adap;
  float* stats;
  int V, S;
  int adap_bool;
};

template <int DUMMY = 0>
__global__ void __launch_bounds__(256)
    loopcd_logits_stats_kernel(const LoopcdLogitsArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int s = blockIdx.x;
  const int64_t b = blockIdx.y;
  if (a.omega[b] == 0.0f) return;

  __shared__ float s_m[8];
  __shared__ float s_l[8];

  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const float* zf_row = a.zf + b * a.zf_s;
  const bool aligned = (reinterpret_cast<uintptr_t>(zf_row) & 15u) == 0u;

  float m_loc = -INFINITY;
  float l_loc = 0.0f;
  auto upd = [&](float x) {
    if (x > m_loc) {
      l_loc = (m_loc > -INFINITY ? l_loc * expf(m_loc - x) : 0.0f) + 1.0f;
      m_loc = x;
    } else if (x > -INFINITY) {
      l_loc += expf(x - m_loc);
    }
  };

  if (aligned) {
    const int V4 = a.V >> 2;
    for (int v4 = s * 256 + tid; v4 < V4; v4 += a.S * 256) {
      const float4 q = load4(zf_row + 4 * v4);
      upd(q.x);
      upd(q.y);
      upd(q.z);
      upd(q.w);
    }
    if (s == 0 && tid < (a.V & 3)) {
      upd(zf_row[(a.V & ~3) + tid]);
    }
  } else {
    for (int v = s * 256 + tid; v < a.V; v += a.S * 256) {
      upd(zf_row[v]);
    }
  }

  const float m_w = wave_max(m_loc);
  const float l_w = wave_sum((m_loc > -INFINITY && m_w > -INFINITY)
                                 ? l_loc * expf(m_loc - m_w)
                                 : 0.0f);
  if (lane == 0) {
    s_m[wave] = m_w;
    s_l[wave] = l_w;
  }
  __syncthreads();

  if (wave == 0) {
    const float mw = (lane < 8) ? s_m[lane] : -INFINITY;
    const float lw = (lane < 8) ? s_l[lane] : 0.0f;
    const float m_cta = wave_max(mw);
    const float l_cta = wave_sum(
        (mw > -INFINITY && m_cta > -INFINITY) ? lw * expf(mw - m_cta) : 0.0f);
    if (lane == 0) {
      float* dst = a.stats + (b * 32 + s) * 2;
      dst[0] = m_cta;
      dst[1] = l_cta;
    }
  }
#endif
}

template <int DUMMY = 0>
__global__ void __launch_bounds__(256)
    loopcd_logits_apply_kernel(const LoopcdLogitsArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int s = blockIdx.x;
  const int64_t b = blockIdx.y;
  const float w0 = a.omega[b];
  if (w0 == 0.0f) return;

  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  float ms = -INFINITY;
  float ls = 0.0f;
  if (lane < a.S) {
    const float* st = a.stats + (b * 32 + lane) * 2;
    ms = st[0];
    ls = st[1];
  }
  const float m_b = wave_max(ms);
  if (!(m_b > -INFINITY)) return;
  const float denom = wave_sum((ms > -INFINITY) ? ls * expf(ms - m_b) : 0.0f);

  const bool adap = a.adap_bool
                        ? reinterpret_cast<const bool*>(a.adap)[b]
                        : (reinterpret_cast<const int32_t*>(a.adap)[b] != 0);
  const float p_max = (denom > 0.0f) ? (1.0f / denom) : 1.0f;
  const float wb = adap ? (w0 * fmaxf(0.0f, 1.0f - p_max)) : w0;
  const float cutoff = m_b + a.log_beta[b];

  float* zf_row = a.zf + b * a.zf_s;
  const float* zw_row = a.zw + b * a.zw_s;
  const bool aligned = ((reinterpret_cast<uintptr_t>(zf_row) |
                         reinterpret_cast<uintptr_t>(zw_row)) &
                        15u) == 0u;

  auto apply1 = [&](float zf, float zw) -> float {
    return (zf >= cutoff && zf > -INFINITY)
               ? ((isfinite(zf) && isfinite(zw)) ? fmaf(wb, zf - zw, zf) : zf)
               : -INFINITY;
  };

  if (aligned) {
    const int V4 = a.V >> 2;
    for (int v4 = s * 256 + tid; v4 < V4; v4 += a.S * 256) {
      const float4 qf = load4(zf_row + 4 * v4);
      const float4 qw = load4(zw_row + 4 * v4);
      store4(zf_row + 4 * v4,
             make_float4(apply1(qf.x, qw.x), apply1(qf.y, qw.y),
                         apply1(qf.z, qw.z), apply1(qf.w, qw.w)));
    }
    if (s == 0 && tid < (a.V & 3)) {
      const int v = (a.V & ~3) + tid;
      zf_row[v] = apply1(zf_row[v], zw_row[v]);
    }
  } else {
    for (int v = s * 256 + tid; v < a.V; v += a.S * 256) {
      zf_row[v] = apply1(zf_row[v], zw_row[v]);
    }
  }
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

// VLLM_INKLING2_FUSE_GLUE (default on, ``0`` or empty turns it off): the
// decode-latency variant of the glue kernels (split-R gate_mix for a few tokens
// with fp32 weights).  Read once.
bool fuse_glue() {
  static const bool on = [] {
    const char* e = std::getenv("VLLM_INKLING2_FUSE_GLUE");
    return e == nullptr || (e[0] != '\0' && e[0] != '0');
  }();
  return on;
}

}  // namespace

// ---------------------------------------------------------------------------
// inkling2_glue_attn_res / _dual / _sconv
// ---------------------------------------------------------------------------
static int glue_log2_exact(int64_t v, const char* who) {
  TORCH_CHECK(v >= 1 && (v & (v - 1)) == 0, who,
              ": the cache block size must be a power of two, got ", v);
  int s = 0;
  while ((int64_t(1) << s) < v) ++s;
  return s;
}

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

template <int Q, bool NORM, int SCONV_MODE>
static void launch_attn_res_sconv(int iter, int block, int64_t T,
                                  const AttnResSconvArgs& sa,
                                  hipStream_t stream) {
#define I2G_ARS(IT)                                               \
  case IT:                                                        \
    inkling2_glue::attn_res_sconv_kernel<Q, IT, NORM, SCONV_MODE> \
        <<<dim3((unsigned)T), dim3(block), 0, stream>>>(sa);      \
    break
  switch (iter) {
    I2G_ARS(1);
    I2G_ARS(2);
    I2G_ARS(3);
    I2G_ARS(4);
    default:
      TORCH_CHECK(false, "inkling2_glue_attn_res_sconv: bad iter ", iter);
  }
#undef I2G_ARS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void fill_attn_res_common(AttnResArgs& a, const char* who,
                                 const at::Tensor& blocks, int64_t n_blocks,
                                 int64_t T, int64_t H, int64_t Q,
                                 const std::optional<at::Tensor>& partial_out,
                                 const std::optional<at::Tensor>& append,
                                 const at::Tensor& query,
                                 const std::optional<at::Tensor>& norm_w,
                                 const at::Tensor& out,
                                 const std::optional<at::Tensor>& norm_out2,
                                 double eps, double norm_eps, bool& norm) {
  check_gpu(query, who, "query");
  check_gpu(out, who, "out");
  check_dtype(query, who, "query", at::kFloat);
  TORCH_CHECK(query.dim() == 2 && query.size(1) == H, who,
              ": query must be [Q, H]");
  check_vec_rows(query, who, "query", 4);
  TORCH_CHECK(out.dim() == 3 && out.size(0) == T && out.size(1) == Q &&
                  out.size(2) == H,
              who, ": out must be [T, Q, H]");
  norm = norm_w.has_value() && norm_w->defined();
  check_dtype(out, who, "out", norm ? at::kHalf : at::kFloat);
  check_vec_rows(out, who, "out", 4);
  TORCH_CHECK(n_blocks >= 0, who, ": n_blocks must be >= 0");

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
  if (norm_out2.has_value() && norm_out2->defined()) {
    TORCH_CHECK(norm && Q == 1, who, ": norm_out2 requires norm_w and Q == 1");
    check_gpu(*norm_out2, who, "norm_out2");
    TORCH_CHECK(norm_out2->scalar_type() == at::kFloat ||
                    norm_out2->scalar_type() == at::kBFloat16,
                who, ": norm_out2 must be Float or BFloat16");
    TORCH_CHECK(norm_out2->dim() == 2 && norm_out2->size(0) == T &&
                    norm_out2->size(1) == H,
                who, ": norm_out2 must be [T, H]");
    check_vec_rows(*norm_out2, who, "norm_out2", 4);
    a.norm_out2 = norm_out2->data_ptr();
    a.no2_t = norm_out2->stride(0);
    a.no2_bf16 = norm_out2->scalar_type() == at::kBFloat16 ? 1 : 0;
  }
}

void inkling2_glue_attn_res_dual(
    const at::Tensor& blocks, int64_t n_blocks, const at::Tensor& x0,
    const std::optional<at::Tensor>& x1,
    const std::optional<at::Tensor>& partial_out,
    const std::optional<at::Tensor>& append, const at::Tensor& query,
    const std::optional<at::Tensor>& norm_w, const at::Tensor& out,
    const std::optional<at::Tensor>& norm_out2, double eps, double norm_eps) {
  const char* who = "inkling2_glue_attn_res";
  check_gpu(x0, who, "x0");
  TORCH_CHECK(x0.dim() == 2, who, ": x0 must be [T, H]");
  const int64_t T = x0.size(0), H = x0.size(1);
  check_dtype(x0, who, "x0", at::kFloat);
  check_vec_rows(x0, who, "x0", 4);
  const int64_t Q = query.dim() == 2 ? query.size(0) : -1;
  TORCH_CHECK(Q == 1 || Q == 3 || Q == 4, who, ": Q must be 1, 3 or 4, got ",
              Q);

  AttnResArgs a{};
  bool norm = false;
  fill_attn_res_common(a, who, blocks, n_blocks, T, H, Q, partial_out, append,
                       query, norm_w, out, norm_out2, eps, norm_eps, norm);
  a.x0 = x0.data_ptr<float>();
  a.x0_t = x0.stride(0);
  if (x1.has_value() && x1->defined()) {
    check_gpu(*x1, who, "x1");
    check_dtype(*x1, who, "x1", at::kFloat);
    TORCH_CHECK(x1->dim() == 2 && x1->size(0) == T && x1->size(1) == H, who,
                ": x1 must be [T, H]");
    check_vec_rows(*x1, who, "x1", 4);
    a.x1 = x1->data_ptr<float>();
    a.x1_t = x1->stride(0);
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

void inkling2_glue_attn_res(const at::Tensor& blocks, int64_t n_blocks,
                            const at::Tensor& x0,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& partial_out,
                            const std::optional<at::Tensor>& append,
                            const at::Tensor& query,
                            const std::optional<at::Tensor>& norm_w,
                            const at::Tensor& out, double eps,
                            double norm_eps) {
  inkling2_glue_attn_res_dual(blocks, n_blocks, x0, x1, partial_out, append,
                              query, norm_w, out, std::nullopt, eps, norm_eps);
}

void inkling2_glue_attn_res_sconv(
    const at::Tensor& blocks, int64_t n_blocks,
    const std::optional<at::Tensor>& x0, const at::Tensor& conv_x,
    const at::Tensor& conv_w, const at::Tensor& conv_cache,
    const at::Tensor& block_table, const at::Tensor& seq_idx,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    int64_t stream_offset, int64_t num_kv_heads, int64_t block_size,
    bool write_state, const std::optional<at::Tensor>& sconv_out,
    const std::optional<at::Tensor>& partial_out,
    const std::optional<at::Tensor>& append, const at::Tensor& query,
    const std::optional<at::Tensor>& norm_w, const at::Tensor& out,
    const std::optional<at::Tensor>& norm_out2, double eps, double norm_eps) {
  const char* who = "inkling2_glue_attn_res_sconv";
  check_gpu(conv_x, who, "conv_x");
  check_dtype(conv_x, who, "conv_x", at::kHalf);
  TORCH_CHECK(conv_x.dim() == 2, who, ": conv_x must be [T, H]");
  const int64_t T = conv_x.size(0), H = conv_x.size(1);
  check_vec_rows(conv_x, who, "conv_x", 4);
  const bool has_x0 = x0.has_value() && x0->defined();
  const int64_t Q = query.dim() == 2 ? query.size(0) : -1;
  TORCH_CHECK(Q == 1 || (Q == 3 && has_x0), who,
              ": Q must be 1 (or 3 with x0), got ", Q);

  AttnResSconvArgs sa{};
  bool norm = false;
  fill_attn_res_common(sa.base, who, blocks, n_blocks, T, H, Q, partial_out,
                       append, query, norm_w, out, norm_out2, eps, norm_eps,
                       norm);
  if (has_x0) {
    check_gpu(*x0, who, "x0");
    check_dtype(*x0, who, "x0", at::kFloat);
    TORCH_CHECK(x0->dim() == 2 && x0->size(0) == T && x0->size(1) == H, who,
                ": x0 must be [T, H]");
    check_vec_rows(*x0, who, "x0", 4);
    sa.base.x0 = x0->data_ptr<float>();
    sa.base.x0_t = x0->stride(0);
  }
  if (sconv_out.has_value() && sconv_out->defined()) {
    check_gpu(*sconv_out, who, "sconv_out");
    check_dtype(*sconv_out, who, "sconv_out", at::kFloat);
    TORCH_CHECK(sconv_out->dim() == 2 && sconv_out->size(0) == T &&
                    sconv_out->size(1) == H,
                who, ": sconv_out must be [T, H]");
    check_vec_rows(*sconv_out, who, "sconv_out", 4);
    sa.sconv_out = sconv_out->data_ptr<float>();
    sa.sco_t = sconv_out->stride(0);
  }

  check_gpu(conv_cache, who, "conv_cache");
  TORCH_CHECK(conv_cache.scalar_type() == at::kHalf && conv_cache.dim() == 4 &&
                  conv_cache.stride(3) == 1,
              who, ": conv_cache must be fp16 [blocks, H, ring_rows, W]");
  TORCH_CHECK(conv_cache.size(2) >= 3, who,
              ": the conv state keeps at least k - 1 = 3 ring rows per block");
  for (const auto& p :
       std::initializer_list<std::pair<const at::Tensor*, const char*>>{
           {&block_table, "block_table"},
           {&seq_idx, "seq_idx"},
           {&query_start_loc, "query_start_loc"},
           {&seq_lens, "seq_lens"}}) {
    check_gpu(*p.first, who, p.second);
    check_dtype(*p.first, who, p.second, at::kInt);
    TORCH_CHECK(p.first->is_contiguous(), who, ": ", p.second,
                " must be contiguous");
  }
  TORCH_CHECK(block_table.dim() == 2, who, ": block_table must be [R, blocks]");
  TORCH_CHECK(conv_cache.stride(0) % 4 == 0 && conv_cache.stride(1) % 4 == 0 &&
                  conv_cache.stride(2) % 4 == 0 &&
                  (reinterpret_cast<uintptr_t>(conv_cache.data_ptr()) & 7) == 0,
              who, ": conv_cache must be 8-byte aligned");
  TORCH_CHECK(num_kv_heads > 0 && H % num_kv_heads == 0, who,
              ": channels must be divisible by num_kv_heads");
  const int64_t width = H / num_kv_heads;
  TORCH_CHECK(width % 4 == 0 && stream_offset % 4 == 0, who,
              ": stream width and offset must be multiples of 4");
  TORCH_CHECK(conv_cache.size(1) == num_kv_heads &&
                  stream_offset + width <= conv_cache.size(3),
              who, ": conv_cache heads / width do not hold this stream");
  check_gpu(conv_w, who, "conv_w");
  TORCH_CHECK(conv_w.scalar_type() == at::kFloat && conv_w.is_contiguous() &&
                  conv_w.numel() == 4 * H &&
                  (reinterpret_cast<uintptr_t>(conv_w.data_ptr()) & 15) == 0,
              who, ": conv_w must be a contiguous fp32 [H, 1, 4] tensor");

  sa.cm.block_table = block_table.data_ptr<int32_t>();
  sa.cm.seq_idx = seq_idx.data_ptr<int32_t>();
  sa.cm.qsl = query_start_loc.data_ptr<int32_t>();
  sa.cm.seq_lens = seq_lens.data_ptr<int32_t>();
  sa.cm.cache = reinterpret_cast<__half*>(conv_cache.data_ptr());
  sa.cm.cs0 = conv_cache.stride(0);
  sa.cm.cs1 = conv_cache.stride(1);
  sa.cm.cs2 = conv_cache.stride(2);
  sa.cm.bt_stride = (int)block_table.stride(0);
  sa.cm.blk_shift = glue_log2_exact(block_size, who);
  sa.cm.ring = (int)conv_cache.size(2);
  sa.cm.T_md = (int)std::min<int64_t>(seq_idx.size(0), T);
  sa.cx = reinterpret_cast<const __half*>(conv_x.data_ptr());
  sa.cx_t = conv_x.stride(0);
  sa.cw = conv_w.data_ptr<float>();
  sa.conv_width = (int)width;
  sa.conv_o0 = (int)stream_offset;
  sa.conv_wr = write_state ? 1 : 0;
  sa.conv_inv = (float)(1.0 / std::sqrt(5.0));

  const at::cuda::OptionalCUDAGuard guard(device_of(conv_x));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  int iter, block;
  row_launch_shape(H, who, iter, block);
#define I2G_SC(QQ, NN, SM) \
  launch_attn_res_sconv<QQ, NN, SM>(iter, block, T, sa, stream)
  if (has_x0) {
    if (Q == 1) {
      if (norm)
        I2G_SC(1, true, 1);
      else
        I2G_SC(1, false, 1);
    } else {
      if (norm)
        I2G_SC(3, true, 1);
      else
        I2G_SC(3, false, 1);
    }
  } else {
    if (norm)
      I2G_SC(1, true, 2);
    else
      I2G_SC(1, false, 2);
  }
#undef I2G_SC
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

template <bool NORM>
static void launch_gate_mix_split(int tb, dim3 grid, dim3 block,
                                  const GateMixArgs& a, hipStream_t stream) {
  switch (tb) {
    case 1:
      inkling2_glue::gate_mix_split_kernel<1, kMixRpw, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    case 2:
      inkling2_glue::gate_mix_split_kernel<2, kMixRpw, NORM>
          <<<grid, block, 0, stream>>>(a);
      break;
    default:
      inkling2_glue::gate_mix_split_kernel<4, kMixRpw, NORM>
          <<<grid, block, 0, stream>>>(a);
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

  // a handful of tokens with fp32 weights: the R-split kernel (opt-in)
  const bool split = fuse_glue() && !w16 && T <= 16 && R == kMixWaves * kMixRpw;
  const int tb = split ? gate_tb(T, 4) : gate_mix_tb(T, G * a.C, w16 ? 16 : 8);
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
  if (split) {
    const dim3 sblock(kMixWaves * kWave);
    if (norm)
      launch_gate_mix_split<true>(tb, grid, sblock, a, stream);
    else
      launch_gate_mix_split<false>(tb, grid, sblock, a, stream);
    return;
  }
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
// inkling2_glue_embed_lookup
// ---------------------------------------------------------------------------
void inkling2_glue_embed_lookup(const at::Tensor& ids, const at::Tensor& table,
                                const at::Tensor& out, int64_t vocab_first,
                                int64_t vocab_total) {
  const char* who = "inkling2_glue_embed_lookup";
  check_gpu(ids, who, "ids");
  check_gpu(table, who, "table");
  check_gpu(out, who, "out");
  TORCH_CHECK(ids.dim() == 1 && (ids.scalar_type() == at::kInt ||
                                 ids.scalar_type() == at::kLong),
              who, ": ids must be a 1-D int32 / int64 tensor");
  TORCH_CHECK(ids.stride(0) == 1, who, ": ids must be contiguous");
  check_dtype(table, who, "table", at::kHalf);
  check_dtype(out, who, "out", at::kHalf);
  TORCH_CHECK(table.dim() == 2 && table.size(0) > 0, who,
              ": table must be a non-empty [rows, H]");
  const int64_t T = ids.size(0), H = table.size(1);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == T && out.size(1) == H, who,
              ": out must be [T, H]");
  check_vec_rows(table, who, "table", 8);
  check_vec_rows(out, who, "out", 8);
  TORCH_CHECK(vocab_first >= 0 && vocab_total > 0, who,
              ": vocab_first must be >= 0 and vocab_total > 0");
  LookupArgs a{};
  a.table = reinterpret_cast<const uint4*>(table.data_ptr());
  a.ids = ids.data_ptr();
  a.ids_i64 = ids.scalar_type() == at::kLong ? 1 : 0;
  a.first = vocab_first;
  a.rows = table.size(0);
  a.total = vocab_total;
  a.t_row8 = table.stride(0) / 8;
  a.out = reinterpret_cast<uint4*>(out.data_ptr());
  a.o_row8 = out.stride(0) / 8;
  a.H8 = (int)(H / 8);
  const at::cuda::OptionalCUDAGuard guard(device_of(table));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  inkling2_glue::embed_lookup_kernel<0>
      <<<dim3((unsigned)T), dim3(256), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const bool dst_bf16 = dst.scalar_type() == at::kBFloat16;
  if (!dst_bf16) check_dtype(dst, who, "dst", at::kFloat);
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
  a.dst = dst.data_ptr();
  a.s_t = src.stride(0);
  a.d_t = dst.stride(0);
  a.s_p = src.dim() == 3 ? src.stride(1) : 0;
  a.d_p = dst.dim() == 3 ? dst.stride(1) : 0;
  a.P = (int)P;
  a.H = (int)H;
  const at::cuda::OptionalCUDAGuard guard(device_of(src));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T * P == 0) return;
  if (dst_bf16) {
    inkling2_glue::copy_states_kernel<true>
        <<<dim3((unsigned)(T * P)), dim3(256), 0, stream>>>(a);
  } else {
    inkling2_glue::copy_states_kernel<false>
        <<<dim3((unsigned)(T * P)), dim3(256), 0, stream>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_glue_moe_route
// ---------------------------------------------------------------------------
template <typename WT>
static void launch_route_small(int tb, unsigned grid, const RouteArgs& a,
                               hipStream_t stream) {
  switch (tb) {
    case 1:
      inkling2_glue::moe_route_small_kernel<WT, 1>
          <<<dim3(grid), dim3(kRouteSmallThreads), 0, stream>>>(a);
      break;
    case 2:
      inkling2_glue::moe_route_small_kernel<WT, 2>
          <<<dim3(grid), dim3(kRouteSmallThreads), 0, stream>>>(a);
      break;
    default:
      inkling2_glue::moe_route_small_kernel<WT, 4>
          <<<dim3(grid), dim3(kRouteSmallThreads), 0, stream>>>(a);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;

  // Small-T path (decode and speculative verify steps): 1 expert per CTA,
  // zero dynamic LDS, single global barrier, in-LDS align.  Capped at T <= 4:
  // at T=5..8 per-CTA register pressure and the serial align loop make the
  // 16-CTA path faster.
  if (fuse_glue() && T <= 4) {
    const int stb = T == 1 ? 1 : (T == 2 ? 2 : 4);
    a.NT = 1;
    a.NEG = (int)E;
    TORCH_CHECK(counters.numel() >= a.NT + 1, who,
                ": counters needs >= ", a.NT + 1, " int32 (zero on entry)");
    a.nzero = (int)std::min<int64_t>(
        (zero_bytes + 16 * kRouteSmallThreads - 1) / (16 * kRouteSmallThreads),
        kMaxZeroCtas);
    const unsigned sgrid = (unsigned)(a.nzero + E);
    if (w16)
      launch_route_small<__half>(stb, sgrid, a, stream);
    else
      launch_route_small<float>(stb, sgrid, a, stream);
    return;
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

  if (w16)
    launch_route<__half>(tb, grid, shmem, a, stream);
  else
    launch_route<float>(tb, grid, shmem, a, stream);
}

// ---------------------------------------------------------------------------
// inkling2_glue_loopcd_blend
// ---------------------------------------------------------------------------
void inkling2_glue_loopcd_blend(const torch::Tensor& x0,
                                const std::optional<torch::Tensor>& x1,
                                const torch::Tensor& prev_x0,
                                const torch::Tensor& prev_x1,
                                const std::optional<torch::Tensor>& blk,
                                const std::optional<torch::Tensor>& prev_blk,
                                const torch::Tensor& omega,
                                const std::optional<torch::Tensor>& omega_blk) {
  const char* who = "inkling2_glue_loopcd_blend";
  check_gpu(x0, who, "x0");
  check_gpu(prev_x0, who, "prev_x0");
  check_gpu(prev_x1, who, "prev_x1");
  check_gpu(omega, who, "omega");
  check_dtype(x0, who, "x0", at::kFloat);
  check_dtype(prev_x0, who, "prev_x0", at::kFloat);
  check_dtype(prev_x1, who, "prev_x1", at::kFloat);
  check_dtype(omega, who, "omega", at::kFloat);
  TORCH_CHECK(x0.dim() == 2, who, ": x0 must be [T, H]");
  const int64_t T = x0.size(0), H = x0.size(1);
  TORCH_CHECK(H % 4 == 0, who, ": hidden size must be a multiple of 4");
  TORCH_CHECK(
      prev_x0.dim() == 2 && prev_x0.size(0) == T && prev_x0.size(1) == H, who,
      ": prev_x0 must be [T, H]");
  TORCH_CHECK(
      prev_x1.dim() == 2 && prev_x1.size(0) == T && prev_x1.size(1) == H, who,
      ": prev_x1 must be [T, H]");
  check_vec_rows(x0, who, "x0", 4);
  check_vec_rows(prev_x0, who, "prev_x0", 4);
  check_vec_rows(prev_x1, who, "prev_x1", 4);
  TORCH_CHECK(omega.dim() == 1 && omega.is_contiguous() && omega.numel() >= T,
              who, ": omega must be 1-D contiguous with numel() >= T");

  LoopcdBlendArgs a{};
  a.x0 = x0.data_ptr<float>();
  a.x0_t = x0.stride(0);
  a.px0 = prev_x0.data_ptr<float>();
  a.px0_t = prev_x0.stride(0);
  a.px1 = prev_x1.data_ptr<float>();
  a.px1_t = prev_x1.stride(0);
  a.omega = omega.data_ptr<float>();
  a.H = (int)H;

  if (x1.has_value() && x1->defined()) {
    check_gpu(*x1, who, "x1");
    check_dtype(*x1, who, "x1", at::kFloat);
    TORCH_CHECK(x1->dim() == 2 && x1->size(0) == T && x1->size(1) == H, who,
                ": x1 must be [T, H]");
    check_vec_rows(*x1, who, "x1", 4);
    a.x1 = x1->data_ptr<float>();
    a.x1_t = x1->stride(0);
  }

  int64_t P = 0;
  if (blk.has_value() && blk->defined()) {
    check_gpu(*blk, who, "blk");
    check_dtype(*blk, who, "blk", at::kFloat);
    if (blk->dim() >= 2 && blk->size(1) > 0) {
      TORCH_CHECK(blk->dim() == 3 && blk->size(0) == T && blk->size(2) == H,
                  who, ": blk must be [T, P, H]");
      P = blk->size(1);
      check_vec_rows(*blk, who, "blk", 4);
      TORCH_CHECK(prev_blk.has_value() && prev_blk->defined(), who,
                  ": prev_blk must be defined when blk has P > 0");
      check_gpu(*prev_blk, who, "prev_blk");
      check_dtype(*prev_blk, who, "prev_blk", at::kFloat);
      TORCH_CHECK(prev_blk->dim() == 3 && prev_blk->size(0) == T &&
                      prev_blk->size(1) >= P && prev_blk->size(2) == H,
                  who, ": prev_blk must be [T, >= P, H]");
      check_vec_rows(*prev_blk, who, "prev_blk", 4);
      a.blk = blk->data_ptr<float>();
      a.b_t = blk->stride(0);
      a.b_p = blk->stride(1);
      a.pblk = prev_blk->data_ptr<float>();
      a.pb_t = prev_blk->stride(0);
      a.pb_p = prev_blk->stride(1);
    } else {
      TORCH_CHECK(blk->numel() == 0, who, ": blk must be [T, P, H]");
    }
  }
  a.P = (int)P;

  if (omega_blk.has_value() && omega_blk->defined()) {
    check_gpu(*omega_blk, who, "omega_blk");
    check_dtype(*omega_blk, who, "omega_blk", at::kFloat);
    TORCH_CHECK(omega_blk->dim() == 1 && omega_blk->is_contiguous() &&
                    omega_blk->numel() >= T,
                who, ": omega_blk must be 1-D contiguous with numel() >= T");
    a.omega_blk = omega_blk->data_ptr<float>();
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x0));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  inkling2_glue::loopcd_blend_kernel<0>
      <<<dim3((unsigned)T, (unsigned)(1 + P)), dim3(256), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_glue_loopcd_readout
// ---------------------------------------------------------------------------
void inkling2_glue_loopcd_readout(const torch::Tensor& blocks, int64_t n_blocks,
                                  const torch::Tensor& x0,
                                  const std::optional<torch::Tensor>& x1,
                                  const torch::Tensor& query,
                                  const torch::Tensor& norm_w,
                                  const torch::Tensor& out,
                                  const std::optional<torch::Tensor>& gate_flag,
                                  double eps, double norm_eps) {
  const char* who = "inkling2_glue_loopcd_readout";
  check_gpu(x0, who, "x0");
  TORCH_CHECK(x0.dim() == 2, who, ": x0 must be [T, H]");
  const int64_t T = x0.size(0), H = x0.size(1);
  check_dtype(x0, who, "x0", at::kFloat);
  check_vec_rows(x0, who, "x0", 4);
  check_gpu(query, who, "query");
  TORCH_CHECK(query.dim() == 2 && query.size(0) == 1, who,
              ": query must be [1, H] (Q == 1)");
  check_gpu(norm_w, who, "norm_w");

  AttnResArgs a{};
  bool norm = false;
  const std::optional<at::Tensor> opt_norm_w(norm_w);
  fill_attn_res_common(a, who, blocks, n_blocks, T, H, 1, std::nullopt,
                       std::nullopt, query, opt_norm_w, out, std::nullopt, eps,
                       norm_eps, norm);
  TORCH_CHECK(norm, who, ": norm_w must be defined");
  a.x0 = x0.data_ptr<float>();
  a.x0_t = x0.stride(0);
  if (x1.has_value() && x1->defined()) {
    check_gpu(*x1, who, "x1");
    check_dtype(*x1, who, "x1", at::kFloat);
    TORCH_CHECK(x1->dim() == 2 && x1->size(0) == T && x1->size(1) == H, who,
                ": x1 must be [T, H]");
    check_vec_rows(*x1, who, "x1", 4);
    a.x1 = x1->data_ptr<float>();
    a.x1_t = x1->stride(0);
  }
  const int32_t* gate = nullptr;
  if (gate_flag.has_value() && gate_flag->defined()) {
    check_gpu(*gate_flag, who, "gate_flag");
    check_dtype(*gate_flag, who, "gate_flag", at::kInt);
    TORCH_CHECK(gate_flag->numel() >= 1, who,
                ": gate_flag must have numel() >= 1");
    gate = gate_flag->data_ptr<int32_t>();
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x0));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  int iter, block;
  row_launch_shape(H, who, iter, block);
#define I2G_LRO(IT)                                               \
  case IT:                                                        \
    inkling2_glue::loopcd_readout_kernel<IT>                      \
        <<<dim3((unsigned)T), dim3(block), 0, stream>>>(a, gate); \
    break
  switch (iter) {
    I2G_LRO(1);
    I2G_LRO(2);
    I2G_LRO(3);
    I2G_LRO(4);
    default:
      TORCH_CHECK(false, "inkling2_glue_loopcd_readout: bad iter ", iter);
  }
#undef I2G_LRO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_loopcd_prepare_hidden
// ---------------------------------------------------------------------------
void inkling2_loopcd_prepare_hidden(
    const torch::Tensor& omega_out, const torch::Tensor& omega_blk_out,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens, const torch::Tensor& prefill_len,
    const torch::Tensor& req_alpha, const torch::Tensor& req_mode,
    const torch::Tensor& req_blend_blk, int64_t num_tokens_padded) {
  const char* who = "inkling2_loopcd_prepare_hidden";
  const std::pair<const at::Tensor*, const char*> on_gpu[] = {
      {&omega_out, "omega_out"},        {&omega_blk_out, "omega_blk_out"},
      {&idx_mapping, "idx_mapping"},    {&query_start_loc, "query_start_loc"},
      {&seq_lens, "seq_lens"},          {&prefill_len, "prefill_len"},
      {&req_alpha, "req_alpha"},        {&req_mode, "req_mode"},
      {&req_blend_blk, "req_blend_blk"}};
  for (const auto& p : on_gpu) check_gpu(*p.first, who, p.second);

  check_dtype(omega_out, who, "omega_out", at::kFloat);
  check_dtype(omega_blk_out, who, "omega_blk_out", at::kFloat);
  TORCH_CHECK(idx_mapping.dim() == 1 && idx_mapping.is_contiguous() &&
                  (idx_mapping.scalar_type() == at::kInt ||
                   idx_mapping.scalar_type() == at::kLong),
              who, ": idx_mapping must be 1-D contiguous int32 or int64");
  const int64_t num_reqs = idx_mapping.size(0);
  check_dtype(query_start_loc, who, "query_start_loc", at::kInt);
  check_dtype(seq_lens, who, "seq_lens", at::kInt);
  check_dtype(prefill_len, who, "prefill_len", at::kInt);
  check_dtype(req_alpha, who, "req_alpha", at::kFloat);
  check_dtype(req_mode, who, "req_mode", at::kInt);
  check_dtype(req_blend_blk, who, "req_blend_blk", at::kInt);

  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.is_contiguous() &&
                  query_start_loc.numel() >= num_reqs + 1,
              who, ": query_start_loc must be 1-D contiguous [>= num_reqs+1]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  seq_lens.numel() >= num_reqs,
              who, ": seq_lens must be 1-D contiguous [>= num_reqs]");
  TORCH_CHECK(prefill_len.dim() == 1 && prefill_len.is_contiguous() &&
                  req_alpha.dim() == 1 && req_alpha.is_contiguous() &&
                  req_mode.dim() == 1 && req_mode.is_contiguous() &&
                  req_blend_blk.dim() == 1 && req_blend_blk.is_contiguous(),
              who, ": request state tables must be 1-D contiguous");
  const int64_t max_reqs = std::min<int64_t>(
      std::min<int64_t>(prefill_len.numel(), req_alpha.numel()),
      std::min<int64_t>(req_mode.numel(), req_blend_blk.numel()));
  TORCH_CHECK(omega_out.dim() == 1 && omega_out.is_contiguous() &&
                  omega_blk_out.dim() == 1 && omega_blk_out.is_contiguous(),
              who, ": omega_out and omega_blk_out must be 1-D contiguous");
  if (num_tokens_padded <= 0) return;
  TORCH_CHECK(omega_out.numel() >= num_tokens_padded &&
                  omega_blk_out.numel() >= num_tokens_padded,
              who, ": omega outputs must have numel() >= num_tokens_padded");

  LoopcdPrepHiddenArgs a{};
  a.omega_out = omega_out.data_ptr<float>();
  a.omega_blk_out = omega_blk_out.data_ptr<float>();
  a.idx = idx_mapping.data_ptr();
  a.qsl = query_start_loc.data_ptr<int32_t>();
  a.seq_lens = seq_lens.data_ptr<int32_t>();
  a.prefill_len = prefill_len.data_ptr<int32_t>();
  a.req_alpha = req_alpha.data_ptr<float>();
  a.req_mode = req_mode.data_ptr<int32_t>();
  a.req_blend_blk = req_blend_blk.data_ptr<int32_t>();
  a.num_reqs = (int)num_reqs;
  a.max_reqs = (int)max_reqs;
  a.num_tokens_padded = (int)num_tokens_padded;
  a.ids_i64 = idx_mapping.scalar_type() == at::kLong ? 1 : 0;

  const at::cuda::OptionalCUDAGuard guard(device_of(omega_out));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  inkling2_glue::loopcd_prepare_hidden_kernel<0>
      <<<dim3((unsigned)(num_reqs + 1)), dim3(256), 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// inkling2_loopcd_logits
// ---------------------------------------------------------------------------
void inkling2_loopcd_logits(const torch::Tensor& z_final,
                            const torch::Tensor& z_first,
                            const torch::Tensor& omega,
                            const torch::Tensor& log_beta,
                            const torch::Tensor& adaptive,
                            const torch::Tensor& stats_scratch) {
  const char* who = "inkling2_loopcd_logits";
  check_gpu(z_final, who, "z_final");
  check_gpu(z_first, who, "z_first");
  check_gpu(omega, who, "omega");
  check_gpu(log_beta, who, "log_beta");
  check_gpu(adaptive, who, "adaptive");
  check_gpu(stats_scratch, who, "stats_scratch");

  check_dtype(z_final, who, "z_final", at::kFloat);
  check_dtype(z_first, who, "z_first", at::kFloat);
  check_dtype(omega, who, "omega", at::kFloat);
  check_dtype(log_beta, who, "log_beta", at::kFloat);
  TORCH_CHECK(
      adaptive.scalar_type() == at::kInt || adaptive.scalar_type() == at::kBool,
      who, ": adaptive must be int32 or bool");
  check_dtype(stats_scratch, who, "stats_scratch", at::kFloat);

  TORCH_CHECK(z_final.dim() == 2 && z_first.dim() == 2 &&
                  z_first.sizes() == z_final.sizes(),
              who, ": z_final and z_first must be [B, V] of one shape");
  const int64_t B = z_final.size(0), V = z_final.size(1);
  TORCH_CHECK(z_final.stride(1) == 1 && z_first.stride(1) == 1, who,
              ": z_final and z_first must have unit stride on V");
  TORCH_CHECK(B <= 1 || z_final.stride(0) >= V, who,
              ": z_final row stride must be >= V");
  TORCH_CHECK(omega.dim() == 1 && omega.is_contiguous() && omega.numel() >= B,
              who, ": omega must be 1-D contiguous [>= B]");
  TORCH_CHECK(
      log_beta.dim() == 1 && log_beta.is_contiguous() && log_beta.numel() >= B,
      who, ": log_beta must be 1-D contiguous [>= B]");
  TORCH_CHECK(
      adaptive.dim() == 1 && adaptive.is_contiguous() && adaptive.numel() >= B,
      who, ": adaptive must be 1-D contiguous [>= B]");
  TORCH_CHECK(
      stats_scratch.is_contiguous() && stats_scratch.numel() >= B * 32 * 2, who,
      ": stats_scratch must be contiguous fp32 [>= B*32*2]");

  const at::cuda::OptionalCUDAGuard guard(device_of(z_final));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (B == 0 || V == 0) return;

  int S = std::clamp<int>((int)(72 / std::max<int64_t>(B, 1)), 1, 32);
  S = std::min<int>(S, (int)((V + 255) / 256));
  S = std::max<int>(S, 1);

  LoopcdLogitsArgs a{};
  a.zf = z_final.data_ptr<float>();
  a.zf_s = z_final.stride(0);
  a.zw = z_first.data_ptr<float>();
  a.zw_s = z_first.stride(0);
  a.omega = omega.data_ptr<float>();
  a.log_beta = log_beta.data_ptr<float>();
  a.adap = adaptive.data_ptr();
  a.stats = stats_scratch.data_ptr<float>();
  a.V = (int)V;
  a.S = S;
  a.adap_bool = adaptive.scalar_type() == at::kBool ? 1 : 0;

  const dim3 grid((unsigned)S, (unsigned)B);
  const dim3 block(256);
  inkling2_glue::loopcd_logits_stats_kernel<0><<<grid, block, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  inkling2_glue::loopcd_logits_apply_kernel<0><<<grid, block, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
