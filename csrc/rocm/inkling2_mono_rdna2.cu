// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Inkling-2-Mini Persistent Decode MoE Monokernel (RDNA2 / gfx1030, Wave32).
//
// Fuses all four decode MoE stages for T <= mono_max_rows (default 8) into a
// single persistent grid (CTAS=56 default for async continuous-batching CU
// headroom, or 72 full-CU):
//   1. Sigmoid router GEMV (fp16/fp32 weights) + bias top-8 + renorm
//   2. NVFP4 (L0-19) / MXFP8 (L20-23) w13 GEMM + softcapped SwiGLU ->
//   act[T*8,I]
//   3. NVFP4 / MXFP8 w2 GEMM + top-8 weighted sum (zero global atomics)
//   4. Optional inline Wave-7 BAR1 P2P AllReduce (via RdnaArDev)
//
// Prefill chunks (T > mono_max_rows or prefill mode) stay on moe_prefill_bm so
// chunked prefill never stalls decode streams.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "rdna_allreduce.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace inkling2_mono {

constexpr int kThreads = 256;
constexpr int kMaxT = 8;
constexpr int kTopK = 8;
constexpr int kMaxE = 128;
constexpr int kMaxH = 2304;
constexpr int kMaxI = 1152;

struct MonoParams {
  const half* x;         // [T, H]
  const void* router_w;  // [E, H] fp16 or fp32
  int router_f16;
  const float* bias;  // [E]
  const uint32_t* w13_q;
  const uint8_t* w13_s;
  const float* w13_g;  // null if MXFP8
  const uint32_t* w2_q;
  const uint8_t* w2_s;
  const float* w2_g;  // null if MXFP8
  float* logits;      // [T, E]
  int32_t* topk_ids;  // [T, K]
  float* topk_w;      // [T, K]
  half* act;          // [T*K, I]
  int* ctrl;          // [4]
  half* out;          // [T, H]
  int T, H, I, E, K;
  int w13_wstride, w13_sstride;
  int w2_wstride, w2_sstride;
  float route_scale, gate_cap, up_cap;
  int nvfp4;
  int nctas;
  int ar_on;
  RdnaArDev ar;
  long long ar_wait_ticks;
  long long ar_ticks_per_ms;
};

}  // namespace inkling2_mono

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

namespace inkling2_mono {

constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;  // 8
constexpr int kActLdsStride = kMaxI + 8;

__device__ __forceinline__ float half_to_f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float swiglu_act(float g, float u, float gate_cap,
                                            float up_cap) {
  const float sig = 1.0f / (1.0f + expf(-g));
  const float gc = tanhf(g / gate_cap) * gate_cap;
  const float uc = tanhf(u / up_cap) * up_cap;
  return (gc * sig) * uc;
}

__device__ __forceinline__ float e4m3_scale_to_f32(uint32_t b) {
  const uint32_t mag = b & 0x7Fu;
  const float normal = __uint_as_float((mag << 20) + 0x3C000000u);
  const float sub = static_cast<float>(mag) * 0.001953125f;  // 2^-9
  const uint32_t bits = __float_as_uint(mag < 8u ? sub : normal);
  return __uint_as_float(bits | ((b & 0x80u) << 24));
}

__device__ __forceinline__ void dequant_e2m1_8_fp16_fast(uint32_t qa,
                                                         half2 scale2,
                                                         half2 (&dq)[4]) {
  const uint32_t ev = qa & 0x0F0F0F0Fu;         // nibbles 0, 2, 4, 6
  const uint32_t od = (qa >> 4) & 0x0F0F0F0Fu;  // nibbles 1, 3, 5, 7
  const uint32_t hb_e = ((ev << 4) & 0x80808080u) | ((ev << 1) & 0x0E0E0E0Eu);
  const uint32_t hb_o = ((od << 4) & 0x80808080u) | ((od << 1) & 0x0E0E0E0Eu);
  dq[0] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x040C000Cu)),
      scale2);
  dq[1] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x050C010Cu)),
      scale2);
  dq[2] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x060C020Cu)),
      scale2);
  dq[3] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x070C030Cu)),
      scale2);
}

__device__ __forceinline__ float dot22_8_f(const half2 (&dq)[4],
                                           const half* a) {
  const uint4 av = *reinterpret_cast<const uint4*>(a);
  float r = 0.0f;
  r = __builtin_amdgcn_fdot2(dq[0], __builtin_bit_cast(half2, av.x), r, false);
  r = __builtin_amdgcn_fdot2(dq[1], __builtin_bit_cast(half2, av.y), r, false);
  r = __builtin_amdgcn_fdot2(dq[2], __builtin_bit_cast(half2, av.z), r, false);
  r = __builtin_amdgcn_fdot2(dq[3], __builtin_bit_cast(half2, av.w), r, false);
  return r;
}

__device__ __forceinline__ half2 e4m3_block_factor(unsigned int ue8m0) {
  const float f = (ue8m0 == 0u) ? ldexpf(1.0f, -127)
                                : ((ue8m0 < 255u) ? __uint_as_float(ue8m0 << 23)
                                                  : __int_as_float(0x7FC00000));
  return __half2half2(__float2half_rn(f * 256.0f));
}

template <unsigned int SEL>
__device__ __forceinline__ half2 e4m3x2_bits(uint32_t w) {
  const uint32_t lanes = __builtin_amdgcn_perm(0u, w, SEL);
  const uint32_t h =
      ((lanes << 8) & 0x80008000u) | ((lanes << 7) & 0x3F803F80u);
  return __builtin_bit_cast(half2, h);
}

__device__ __forceinline__ void deq_e4m3_4(uint32_t w, half2 factor,
                                           half2* dq) {
  dq[0] = __hmul2(e4m3x2_bits<0x0C010C00u>(w), factor);
  dq[1] = __hmul2(e4m3x2_bits<0x0C030C02u>(w), factor);
}

// Sense-reversing HIP-graph-safe grid barrier. Every wave calls __threadfence()
// BEFORE __syncthreads() so prior GL2 stores from all waves are visible to all
// CTAs before thread 0 increments ctrl[0] with agent-scope release.
__device__ __forceinline__ void mono_grid_barrier(int* ctrl, int nctas,
                                                  int& sense) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int expected = sense ^ 1;
    if (__hip_atomic_fetch_add(&ctrl[0], 1, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT) == nctas - 1) {
      __hip_atomic_store(&ctrl[0], 0, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&ctrl[1], expected, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (__hip_atomic_load(&ctrl[1], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) != expected) {
        __builtin_amdgcn_s_sleep(2);
      }
    }
  }
  __syncthreads();
  sense ^= 1;
}

template <int W>
__device__ __forceinline__ void mono_ar_reduce_vec(
    half* __restrict__ out, const half* __restrict__ mine_base, int rank,
    long long max_elems, int nvec, int gid, int gstride) {
  uint4* out4 = reinterpret_cast<uint4*>(out);
  for (int v = gid; v < nvec; v += gstride) {
    uint4 x[W];
  #pragma unroll
    for (int r = 0; r < W; r++) {
      const uint4* src4 = (r == rank)
                              ? reinterpret_cast<const uint4*>(out)
                              : reinterpret_cast<const uint4*>(
                                    mine_base + (long long)r * max_elems);
      x[r] = src4[v];
    }
    float acc[8];
  #pragma unroll
    for (int m = 0; m < 8; m++) acc[m] = 0.0f;
  #pragma unroll
    for (int r = 0; r < W; r++) {
      const half* h = reinterpret_cast<const half*>(&x[r]);
  #pragma unroll
      for (int m = 0; m < 8; m++) acc[m] += __half2float(h[m]);
    }
    union {
      uint4 q;
      half h[8];
    } res;
  #pragma unroll
    for (int m = 0; m < 8; m++) res.h[m] = __float2half_rn(acc[m]);
    out4[v] = res.q;
  }
}

template <bool NVFP4>
__global__ void __launch_bounds__(kThreads)
    inkling2_mono_moe_kernel(const MonoParams p) {
  __shared__ union {
    struct {
      half s_x[kMaxH + 8];
      float s_red[kWaves][32][4];
    } p12;
    struct {
      half s_act[kTopK * kActLdsStride];
      float s_red[kWaves][16][4];
      float s_out[64];
      float s_w[kTopK];
    } p3;
  } sm;
  __shared__ int s_sense;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int H = p.H, I = p.I, E = p.E, K = p.K, T = p.T;

  if (tid == 0) {
    s_sense = __hip_atomic_load(&p.ctrl[1], __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  int sense = s_sense;

  // =========================================================================
  // Phase 1a: Router GEMV across (T * (E / 8)) work items
  // =========================================================================
  const int eg_cnt = E / kWaves;
  const int r_items = T * eg_cnt;
  for (int wi = bid; wi < r_items; wi += p.nctas) {
    const int t = wi / eg_cnt;
    const int eg = wi % eg_cnt;
    const uint4* x4 = reinterpret_cast<const uint4*>(p.x + (int64_t)t * H);
    uint4* sx4 = reinterpret_cast<uint4*>(sm.p12.s_x);
    for (int i = tid; i < (H >> 3); i += kThreads) sx4[i] = x4[i];
    __syncthreads();

    const int e = eg * kWaves + wave;
    float acc = 0.0f;
    if (p.router_f16) {
      const uint4* wr = reinterpret_cast<const uint4*>(
          reinterpret_cast<const half*>(p.router_w) + (int64_t)e * H);
      for (int i = lane; i < (H >> 3); i += kWave) {
        const uint4 wv = wr[i];
        const uint4 xv = sx4[i];
        acc =
            __builtin_amdgcn_fdot2(__builtin_bit_cast(half2, wv.x),
                                   __builtin_bit_cast(half2, xv.x), acc, false);
        acc =
            __builtin_amdgcn_fdot2(__builtin_bit_cast(half2, wv.y),
                                   __builtin_bit_cast(half2, xv.y), acc, false);
        acc =
            __builtin_amdgcn_fdot2(__builtin_bit_cast(half2, wv.z),
                                   __builtin_bit_cast(half2, xv.z), acc, false);
        acc =
            __builtin_amdgcn_fdot2(__builtin_bit_cast(half2, wv.w),
                                   __builtin_bit_cast(half2, xv.w), acc, false);
      }
    } else {
      const float4* wr = reinterpret_cast<const float4*>(
          reinterpret_cast<const float*>(p.router_w) + (int64_t)e * H);
      const half* sx = sm.p12.s_x;
      for (int i = lane; i < (H >> 2); i += kWave) {
        const float4 wv = wr[i];
        const uint2 xv = *reinterpret_cast<const uint2*>(sx + 4 * i);
        acc = fmaf(wv.x, half_to_f((uint16_t)(xv.x & 0xFFFFu)), acc);
        acc = fmaf(wv.y, half_to_f((uint16_t)(xv.x >> 16)), acc);
        acc = fmaf(wv.z, half_to_f((uint16_t)(xv.y & 0xFFFFu)), acc);
        acc = fmaf(wv.w, half_to_f((uint16_t)(xv.y >> 16)), acc);
      }
    }
    acc = wave_sum(acc);
    if (lane == 0) p.logits[t * E + e] = acc;
    __syncthreads();
  }

  mono_grid_barrier(p.ctrl, p.nctas, sense);

  // =========================================================================
  // Phase 1b: CTA 0 selects top-K across all T <= 8 tokens (1 wave per token)
  // =========================================================================
  if (bid == 0 && wave < T) {
    const float* row = p.logits + wave * E;
    float sc[4], sel[4];
  #pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = lane + kWave * k;
      if (e < E) {
        sc[k] = 1.0f / (1.0f + expf(-row[e]));
        sel[k] = sc[k] + p.bias[e];
      } else {
        sc[k] = 0.0f;
        sel[k] = -__builtin_inff();
      }
    }
    float chosen_w[kTopK];
    float sum_w = 0.0f;
    for (int m = 0; m < K; m++) {
      float bv = -__builtin_inff();
      int be = E;
  #pragma unroll
      for (int k = 0; k < 4; k++) {
        const int e = lane + kWave * k;
        if (sel[k] > bv || (sel[k] == bv && e < be)) {
          bv = sel[k];
          be = e;
        }
      }
  #pragma unroll
      for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor(bv, o);
        const int oe = __shfl_xor(be, o);
        if (ov > bv || (ov == bv && oe < be)) {
          bv = ov;
          be = oe;
        }
      }
      float ws = 0.0f;
  #pragma unroll
      for (int k = 0; k < 4; k++) {
        if (lane + kWave * k == be) {
          ws = sc[k];
          sel[k] = -__builtin_inff();
        }
      }
      // Sum across wave so every lane holds winner score
      ws = wave_sum(ws);
      chosen_w[m] = ws;
      sum_w += ws;
      if (lane == 0) p.topk_ids[wave * K + m] = be;
    }
    const float inv = p.route_scale / (sum_w + 1e-20f);
    if (lane < K) p.topk_w[wave * K + lane] = chosen_w[lane] * inv;
  }

  mono_grid_barrier(p.ctrl, p.nctas, sense);

  // =========================================================================
  // Phase 2: w13 + softcapped SwiGLU -> act[T * K, I] (zero global atomics)
  // =========================================================================
  const int n13 = 2 * I;
  const int w13_tiles = (I + 63) / 64;
  const int p2_items = T * K * w13_tiles;
  const int H_blks = H / 32;  // 72 for H=2304

  for (int wi = bid; wi < p2_items; wi += p.nctas) {
    const int pair = wi / w13_tiles;
    const int nt = wi % w13_tiles;
    const int t = pair / K;
    const int c0 = nt * 64;

    const uint4* x4 = reinterpret_cast<const uint4*>(p.x + (int64_t)t * H);
    uint4* sx4 = reinterpret_cast<uint4*>(sm.p12.s_x);
    for (int i = tid; i < (H >> 3); i += kThreads) sx4[i] = x4[i];
    __syncthreads();

    const int eid = p.topk_ids[pair];
    const int is_up = (lane >= 16) ? 1 : 0;
    const int q_local = (lane & 15) * 4;
    const int col = (is_up ? I : 0) + c0 + q_local;
    const bool valid_col = (c0 + q_local) < I;

    const int kb = ((H_blks * wave) / kWaves) * 32;
    const int ke = ((H_blks * (wave + 1)) / kWaves) * 32;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (valid_col) {
      if constexpr (NVFP4) {
        const float global = p.w13_g[eid];
        const uint32_t* bp = p.w13_q + (int64_t)eid * p.w13_wstride +
                             (int64_t)(kb / 8) * n13 + col;
        const uint8_t* sp = p.w13_s + (int64_t)eid * p.w13_sstride +
                            (int64_t)(kb / 16) * n13 + col;
        for (int k = kb; k < ke; k += 32) {
          const uint32_t sw0 = *(const uint32_t*)sp;
          const uint32_t sw1 = *(const uint32_t*)(sp + n13);
          sp += 2 * n13;
          half2 sc[4][2];
  #pragma unroll
          for (int j = 0; j < 4; j++) {
            sc[j][0] =
                __float2half2_rn(e4m3_scale_to_f32((sw0 >> (8 * j)) & 0xFFu) *
                                 global * 16384.0f);
            sc[j][1] =
                __float2half2_rn(e4m3_scale_to_f32((sw1 >> (8 * j)) & 0xFFu) *
                                 global * 16384.0f);
          }
          int4 bw[4];
  #pragma unroll
          for (int j = 0; j < 4; j++) bw[j] = *(const int4*)(bp + j * n13);
          bp += 4 * n13;
  #pragma unroll
          for (int j = 0; j < 4; j++) {
            const half* ap = &sm.p12.s_x[k + 8 * j];
            half2 dq[4];
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].x, sc[0][j / 2], dq);
            acc[0] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].y, sc[1][j / 2], dq);
            acc[1] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].z, sc[2][j / 2], dq);
            acc[2] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].w, sc[3][j / 2], dq);
            acc[3] += dot22_8_f(dq, ap);
          }
        }
      } else {
        const uint32_t* bp = p.w13_q + (int64_t)eid * p.w13_wstride +
                             (int64_t)(kb / 4) * n13 + col;
        const uint8_t* sp = p.w13_s + (int64_t)eid * p.w13_sstride +
                            (int64_t)(kb / 32) * n13 + col;
        for (int k = kb; k < ke; k += 32) {
          half2 factor[4];
  #pragma unroll
          for (int j = 0; j < 4; j++) factor[j] = e4m3_block_factor(sp[j]);
          sp += n13;
          uint32_t w[8][4];
  #pragma unroll
          for (int i = 0; i < 8; i++) {
            const int4 v = *(const int4*)(bp + i * n13);
            w[i][0] = (uint32_t)v.x;
            w[i][1] = (uint32_t)v.y;
            w[i][2] = (uint32_t)v.z;
            w[i][3] = (uint32_t)v.w;
          }
          bp += 8 * n13;
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            const uint4 av =
                *reinterpret_cast<const uint4*>(&sm.p12.s_x[k + 8 * g]);
            const half* ap = reinterpret_cast<const half*>(&av);
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              half2 dq[4];
              deq_e4m3_4(w[2 * g + 0][j], factor[j], dq + 0);
              deq_e4m3_4(w[2 * g + 1][j], factor[j], dq + 2);
              acc[j] += dot22_8_f(dq, ap);
            }
          }
        }
      }
    }
  #pragma unroll
    for (int j = 0; j < 4; j++) sm.p12.s_red[wave][lane][j] = acc[j];
    __syncthreads();

    if (tid < 64 && c0 + tid < I) {
      const int lane_g = tid >> 2;
      const int sub = tid & 3;
      float gf = 0.0f, uf = 0.0f;
  #pragma unroll
      for (int wv = 0; wv < kWaves; wv++) {
        gf += sm.p12.s_red[wv][lane_g][sub];
        uf += sm.p12.s_red[wv][16 + lane_g][sub];
      }
      const float g16 = __half2float(__float2half_rn(gf));
      const float u16 = __half2float(__float2half_rn(uf));
      p.act[(int64_t)pair * I + c0 + tid] =
          __float2half_rn(swiglu_act(g16, u16, p.gate_cap, p.up_cap));
    }
    __syncthreads();
  }

  mono_grid_barrier(p.ctrl, p.nctas, sense);

  // =========================================================================
  // Phase 3: w2 across all K=8 active experts -> out[T, H] (zero atomics!)
  // =========================================================================
  const int w2_tiles = (H + 63) / 64;
  const int p3_items = T * w2_tiles;
  const int I_blks = I / 32;  // 9 (TP4), 18 (TP2), 36 (TP1)

  for (int wi = bid; wi < p3_items; wi += p.nctas) {
    const int t = wi / w2_tiles;
    const int nt = wi % w2_tiles;
    const int c0 = nt * 64;

    if (tid < K) sm.p3.s_w[tid] = p.topk_w[t * K + tid];
    const int iv8 = I >> 3;
    const half* act_t = p.act + (int64_t)(t * K) * I;
    for (int idx = tid; idx < K * iv8; idx += kThreads) {
      const int ei = idx / iv8;
      const int v = idx % iv8;
      const uint4 q =
          reinterpret_cast<const uint4*>(act_t + (int64_t)ei * I)[v];
      reinterpret_cast<uint4*>(sm.p3.s_act + ei * kActLdsStride)[v] = q;
    }
    __syncthreads();

    // 8 waves handle 4 pairs of experts split over K2 in 2 halves:
    // wave w -> (expert_pair = w & 3, half_k = w >> 2)
    // Within each wave: lanes 0..15 handle expert (2 * ep + 0),
    //                   lanes 16..31 handle expert (2 * ep + 1).
    const int ep = wave & 3;
    const int kh = wave >> 2;
    const int ei = 2 * ep + (lane >= 16 ? 1 : 0);
    const int col = c0 + (lane & 15) * 4;
    const int kb = (kh == 0) ? 0 : ((I_blks / 2) * 32);
    const int ke = (kh == 0) ? ((I_blks / 2) * 32) : I;
    const int eid = p.topk_ids[t * K + ei];
    const half* sa = sm.p3.s_act + ei * kActLdsStride;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (col < H && kb < ke) {
      if constexpr (NVFP4) {
        const float global = p.w2_g[eid];
        const uint32_t* bp =
            p.w2_q + (int64_t)eid * p.w2_wstride + (int64_t)(kb / 8) * H + col;
        const uint8_t* sp =
            p.w2_s + (int64_t)eid * p.w2_sstride + (int64_t)(kb / 16) * H + col;
        for (int k = kb; k < ke; k += 32) {
          const uint32_t sw0 = *(const uint32_t*)sp;
          const uint32_t sw1 = *(const uint32_t*)(sp + H);
          sp += 2 * H;
          half2 sc[4][2];
  #pragma unroll
          for (int j = 0; j < 4; j++) {
            sc[j][0] =
                __float2half2_rn(e4m3_scale_to_f32((sw0 >> (8 * j)) & 0xFFu) *
                                 global * 16384.0f);
            sc[j][1] =
                __float2half2_rn(e4m3_scale_to_f32((sw1 >> (8 * j)) & 0xFFu) *
                                 global * 16384.0f);
          }
          int4 bw[4];
  #pragma unroll
          for (int j = 0; j < 4; j++) bw[j] = *(const int4*)(bp + j * H);
          bp += 4 * H;
  #pragma unroll
          for (int j = 0; j < 4; j++) {
            const half* ap = &sa[k + 8 * j];
            half2 dq[4];
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].x, sc[0][j / 2], dq);
            acc[0] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].y, sc[1][j / 2], dq);
            acc[1] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].z, sc[2][j / 2], dq);
            acc[2] += dot22_8_f(dq, ap);
            dequant_e2m1_8_fp16_fast((uint32_t)bw[j].w, sc[3][j / 2], dq);
            acc[3] += dot22_8_f(dq, ap);
          }
        }
      } else {
        const uint32_t* bp =
            p.w2_q + (int64_t)eid * p.w2_wstride + (int64_t)(kb / 4) * H + col;
        const uint8_t* sp =
            p.w2_s + (int64_t)eid * p.w2_sstride + (int64_t)(kb / 32) * H + col;
        for (int k = kb; k < ke; k += 32) {
          half2 factor[4];
  #pragma unroll
          for (int j = 0; j < 4; j++) factor[j] = e4m3_block_factor(sp[j]);
          sp += H;
          uint32_t w[8][4];
  #pragma unroll
          for (int i = 0; i < 8; i++) {
            const int4 v = *(const int4*)(bp + i * H);
            w[i][0] = (uint32_t)v.x;
            w[i][1] = (uint32_t)v.y;
            w[i][2] = (uint32_t)v.z;
            w[i][3] = (uint32_t)v.w;
          }
          bp += 8 * H;
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            const uint4 av = *reinterpret_cast<const uint4*>(&sa[k + 8 * g]);
            const half* ap = reinterpret_cast<const half*>(&av);
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              half2 dq[4];
              deq_e4m3_4(w[2 * g + 0][j], factor[j], dq + 0);
              deq_e4m3_4(w[2 * g + 1][j], factor[j], dq + 2);
              acc[j] += dot22_8_f(dq, ap);
            }
          }
        }
      }
    }
    const int slot = wave;  // 0..3: kh=0, 4..7: kh=1
    const int ql = lane & 15;
    const int e_half = lane >= 16 ? 1 : 0;

    if (e_half == 0) {
  #pragma unroll
      for (int j = 0; j < 4; j++) sm.p3.s_red[slot][ql][j] = acc[j];
    }
    __syncthreads();
    if (tid < 64) {
      const int qid = tid >> 2;
      const int sub = tid & 3;
      float tot = 0.0f;
  #pragma unroll
      for (int ep2 = 0; ep2 < 4; ep2++) {
        const float sum_e =
            sm.p3.s_red[ep2][qid][sub] + sm.p3.s_red[ep2 + 4][qid][sub];
        tot += sum_e * sm.p3.s_w[2 * ep2];
      }
      sm.p3.s_out[tid] = tot;
    }
    __syncthreads();
    if (e_half == 1) {
  #pragma unroll
      for (int j = 0; j < 4; j++) sm.p3.s_red[slot][ql][j] = acc[j];
    }
    __syncthreads();
    if (tid < 64 && c0 + tid < H) {
      const int qid = tid >> 2;
      const int sub = tid & 3;
      float tot = sm.p3.s_out[tid];
  #pragma unroll
      for (int ep2 = 0; ep2 < 4; ep2++) {
        const float sum_e =
            sm.p3.s_red[ep2][qid][sub] + sm.p3.s_red[ep2 + 4][qid][sub];
        tot += sum_e * sm.p3.s_w[2 * ep2 + 1];
      }
      p.out[(int64_t)t * H + c0 + tid] = __float2half_rn(tot);
    }
    __syncthreads();
  }

  // =========================================================================
  // Phase 4 (optional, TP > 1): inline Wave-7 BAR1 P2P AllReduce on out[T, H]
  // =========================================================================
  if (p.ar_on) {
    mono_grid_barrier(p.ctrl, p.nctas, sense);
    const int n = T * H;
    const long long max_elems = p.ar.max_bytes / (long long)sizeof(half);
    __shared__ int s_ar_seq;
    if (tid == 0) {
      s_ar_seq =
          (int)((unsigned)__hip_atomic_load(p.ar.seqbuf, __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_AGENT) +
                1u);
    }
    __syncthreads();
    const int ar_seq = s_ar_seq;
    const int ar_p = ar_seq & 1;
    const int gid = bid * kThreads + tid;
    const int gstride = p.nctas * kThreads;

    // Push out[0..n) to every peer's staging slot (parity, rank) as 16B uint4s
    for (int k = 1; k < p.ar.world; k++) {
      const int j = (p.ar.rank + k) % p.ar.world;
      half* dst = reinterpret_cast<half*>(p.ar.peers.stage[j]) +
                  ((long long)ar_p * p.ar.world + p.ar.rank) * max_elems;
      if (((uintptr_t)dst & 15) == 0 && ((uintptr_t)p.out & 15) == 0 &&
          n >= 8) {
        const int nvec = n >> 3;
        uint4* dst4 = reinterpret_cast<uint4*>(dst);
        const uint4* src4 = reinterpret_cast<const uint4*>(p.out);
        for (int v = gid; v < nvec; v += gstride) dst4[v] = src4[v];
        for (int i = (nvec << 3) + gid; i < n; i += gstride) dst[i] = p.out[i];
      } else {
        for (int i = gid; i < n; i += gstride) dst[i] = p.out[i];
      }
    }

    // All threads flush posted PCIe writes before barrier + CTA 0 flag
    // announce.
    __threadfence_system();
    mono_grid_barrier(p.ctrl, p.nctas, sense);
    if (bid == 0 && tid == 0) {
      __hip_atomic_store(p.ar.seqbuf, ar_seq, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
      rdna_ar_announce(p.ar.peers, p.ar.rank, p.ar.world, ar_seq);
      int* myflags = p.ar.peers.flags[p.ar.rank];
      for (int j = 0; j < p.ar.world; j++) {
        if (j == p.ar.rank) continue;
        const long long t0 = wall_clock64();
        while (rdna_ar_before(__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE,
                                                __HIP_MEMORY_SCOPE_SYSTEM),
                              ar_seq)) {
          RDNA_AR_POLL_PAUSE();
          if (wall_clock64() - t0 > p.ar_wait_ticks) break;
        }
      }
    }
    mono_grid_barrier(p.ctrl, p.nctas, sense);

    // Reduce rank 0..world-1 in ascending rank order in fp32 (vectorized)
    const half* mine_base =
        reinterpret_cast<const half*>(p.ar.peers.stage[p.ar.rank]) +
        ((long long)ar_p * p.ar.world) * max_elems;
    const int W = p.ar.world;
    if (((uintptr_t)p.out & 15) == 0 && (max_elems & 7) == 0 && n >= 8) {
      const int nvec = n >> 3;
      if (W == 2) {
        mono_ar_reduce_vec<2>(p.out, mine_base, p.ar.rank, max_elems, nvec, gid,
                              gstride);
      } else if (W == 4) {
        mono_ar_reduce_vec<4>(p.out, mine_base, p.ar.rank, max_elems, nvec, gid,
                              gstride);
      } else {
        uint4* out4 = reinterpret_cast<uint4*>(p.out);
        for (int v = gid; v < nvec; v += gstride) {
          float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          for (int r = 0; r < W; r++) {
            const uint4* src4 = (r == p.ar.rank)
                                    ? reinterpret_cast<const uint4*>(p.out)
                                    : reinterpret_cast<const uint4*>(
                                          mine_base + (long long)r * max_elems);
            union {
              uint4 q;
              half h[8];
            } u;
            u.q = src4[v];
  #pragma unroll
            for (int m = 0; m < 8; m++) acc[m] += __half2float(u.h[m]);
          }
          union {
            uint4 q;
            half h[8];
          } res;
  #pragma unroll
          for (int m = 0; m < 8; m++) res.h[m] = __float2half_rn(acc[m]);
          out4[v] = res.q;
        }
      }
      for (int i = (nvec << 3) + gid; i < n; i += gstride) {
        float acc = 0.0f;
        for (int r = 0; r < W; r++) {
          const half hv = (r == p.ar.rank)
                              ? p.out[i]
                              : mine_base[(long long)r * max_elems + i];
          acc += __half2float(hv);
        }
        p.out[i] = __float2half_rn(acc);
      }
    } else {
      for (int i = gid; i < n; i += gstride) {
        float acc = 0.0f;
        for (int r = 0; r < W; r++) {
          const half hv = (r == p.ar.rank)
                              ? p.out[i]
                              : mine_base[(long long)r * max_elems + i];
          acc += __half2float(hv);
        }
        p.out[i] = __float2half_rn(acc);
      }
    }
  }
}

}  // namespace inkling2_mono

#else  // non-RDNA2 device pass: empty stub

namespace inkling2_mono {

template <bool NVFP4>
__global__ void __launch_bounds__(256)
    inkling2_mono_moe_kernel(const MonoParams) {}

}  // namespace inkling2_mono

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

namespace {

int query_max_resident_ctas(int dev, bool nvfp4) {
  static int cached[16][2] = {};
  if (dev >= 0 && dev < 16 && cached[dev][nvfp4 ? 1 : 0] > 0) {
    return cached[dev][nvfp4 ? 1 : 0];
  }
  int cus = 72;
  int max_per_cu = 1;
  (void)hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev);
  if (nvfp4) {
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_per_cu, inkling2_mono::inkling2_mono_moe_kernel<true>, 256, 0);
  } else {
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_per_cu, inkling2_mono::inkling2_mono_moe_kernel<false>, 256, 0);
  }
  int cap = std::max(1, std::max(1, max_per_cu) * std::max(1, cus));
  if (dev >= 0 && dev < 16) cached[dev][nvfp4 ? 1 : 0] = cap;
  return cap;
}

long long query_ticks_per_ms(int dev) {
  static long long cached[16] = {};
  if (dev >= 0 && dev < 16 && cached[dev] > 0) return cached[dev];
  int wc_khz = 0;
  (void)hipDeviceGetAttribute(&wc_khz, hipDeviceAttributeWallClockRate, dev);
  long long rate = (wc_khz > 0) ? (long long)wc_khz : 100000ll;
  if (dev >= 0 && dev < 16) cached[dev] = rate;
  return rate;
}

}  // namespace

void inkling2_mono_moe_k2(
    const torch::Tensor& x, const torch::Tensor& router_w,
    const torch::Tensor& bias, const torch::Tensor& w13_qweight,
    const torch::Tensor& w13_scales,
    const std::optional<torch::Tensor>& w13_globals,
    const torch::Tensor& w2_qweight, const torch::Tensor& w2_scales,
    const std::optional<torch::Tensor>& w2_globals, torch::Tensor logits_ws,
    torch::Tensor topk_ids_ws, torch::Tensor topk_w_ws, torch::Tensor act_ws,
    torch::Tensor ctrl_ws, torch::Tensor out, int64_t top_k, double route_scale,
    double gate_cap, double up_cap, int64_t ar_handle, int64_t n_ctas) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "inkling2_mono_moe_k2: x must be 2D fp16 GPU tensor");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat,
              "inkling2_mono_moe_k2: bias must be fp32 GPU tensor");
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)router_w.size(0);
  const int I = (int)w2_qweight.size(1) *
                (w13_globals.has_value() && w13_globals->defined() ? 8 : 4);
  TORCH_CHECK(T >= 1 && T <= inkling2_mono::kMaxT,
              "inkling2_mono_moe_k2: T must be 1..", inkling2_mono::kMaxT);
  TORCH_CHECK(H <= inkling2_mono::kMaxH && H % 32 == 0,
              "inkling2_mono_moe_k2: H must be <= 2304 and % 32 == 0");
  TORCH_CHECK(I <= inkling2_mono::kMaxI && I % 32 == 0,
              "inkling2_mono_moe_k2: I must be <= 1152 and % 32 == 0");
  TORCH_CHECK(E <= inkling2_mono::kMaxE && E % 32 == 0,
              "inkling2_mono_moe_k2: E must be <= 128 and % 32 == 0");
  TORCH_CHECK(top_k == inkling2_mono::kTopK, "inkling2_mono_moe_k2: top_k=8");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ctrl_ws.numel() >= 4,
              "inkling2_mono_moe_k2: invalid buffers");

  const bool nvfp4 = w13_globals.has_value() && w13_globals->defined() &&
                     w2_globals.has_value() && w2_globals->defined();
  const bool router_f16 = router_w.scalar_type() == at::kHalf;
  TORCH_CHECK(router_f16 || router_w.scalar_type() == at::kFloat,
              "inkling2_mono_moe_k2: router_w must be fp16 or fp32");

  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int dev = x.device().index();
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  inkling2_mono::MonoParams p{};
  p.x = reinterpret_cast<const half*>(x.data_ptr());
  p.router_w = router_w.data_ptr();
  p.router_f16 = router_f16 ? 1 : 0;
  p.bias = bias.data_ptr<float>();
  p.w13_q = reinterpret_cast<const uint32_t*>(w13_qweight.data_ptr());
  p.w13_s = reinterpret_cast<const uint8_t*>(w13_scales.data_ptr());
  p.w13_g = nvfp4 ? w13_globals->data_ptr<float>() : nullptr;
  p.w2_q = reinterpret_cast<const uint32_t*>(w2_qweight.data_ptr());
  p.w2_s = reinterpret_cast<const uint8_t*>(w2_scales.data_ptr());
  p.w2_g = nvfp4 ? w2_globals->data_ptr<float>() : nullptr;
  p.logits = logits_ws.data_ptr<float>();
  p.topk_ids = topk_ids_ws.data_ptr<int32_t>();
  p.topk_w = topk_w_ws.data_ptr<float>();
  p.act = reinterpret_cast<half*>(act_ws.data_ptr());
  p.ctrl = ctrl_ws.data_ptr<int32_t>();
  p.out = reinterpret_cast<half*>(out.data_ptr());
  p.T = T;
  p.H = H;
  p.I = I;
  p.E = E;
  p.K = (int)top_k;
  p.w13_wstride = (int)(w13_qweight.size(1) * w13_qweight.size(2));
  p.w13_sstride = (int)(w13_scales.size(1) * w13_scales.size(2));
  p.w2_wstride = (int)(w2_qweight.size(1) * w2_qweight.size(2));
  p.w2_sstride = (int)(w2_scales.size(1) * w2_scales.size(2));
  p.route_scale = (float)route_scale;
  p.gate_cap = (float)gate_cap;
  p.up_cap = (float)up_cap;
  p.nvfp4 = nvfp4 ? 1 : 0;
  const int req_ctas = (int)(n_ctas > 0 ? n_ctas : 56);
  const int cap_ctas = query_max_resident_ctas(dev, nvfp4);
  p.nctas = std::max(1, std::min(req_ctas, cap_ctas));
  p.ar_on = 0;
  if (ar_handle >= 0 && rdna_ar_device((long long)ar_handle, &p.ar) &&
      (int64_t)T * H * (int64_t)sizeof(half) <= p.ar.max_bytes) {
    p.ar_on = 1;
    p.ar_ticks_per_ms = query_ticks_per_ms(dev);
    p.ar_wait_ticks = RDNA_AR_WAIT_MS * p.ar_ticks_per_ms;
  }

  if (nvfp4) {
    inkling2_mono::inkling2_mono_moe_kernel<true>
        <<<dim3(p.nctas), dim3(inkling2_mono::kThreads), 0, stream>>>(p);
  } else {
    inkling2_mono::inkling2_mono_moe_kernel<false>
        <<<dim3(p.nctas), dim3(inkling2_mono::kThreads), 0, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
