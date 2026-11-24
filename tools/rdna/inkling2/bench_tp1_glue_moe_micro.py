#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TP=1 Glue + MoE parity & HIP-graph microbenchmark on RDNA2 (gfx1030).

Compiles ``csrc/rocm/inkling2_glue_rdna2.cu``, ``csrc/rocm/inkling2_attn_rdna2.cu``
(for ``inkling2_sconv_static`` / ``inkling2_conv_state_write`` reference parity),
and ``csrc/rocm/moe_lowp_dot2_rdna2.cu`` via ``torch.utils.cpp_extension.load``
(including ``-Rpass-analysis=kernel-resource-usage`` VGPR/LDS diagnostics and
experimental ``gate_mix_split`` occupancy / acquire-release atomic mutants) so it
can be run directly on ``par1-cs25`` without a full editable vLLM rebuild:

    HIP_VISIBLE_DEVICES=1 python3 tools/rdna/inkling2/bench_tp1_glue_moe_micro.py \\
        --section all --json /tmp/bench_tp1_glue_moe_micro.json

Sections:
  1. ``sconv_attn_res``:
     Bit-exact parity & HIP-graph latency of standalone ``inkling2_sconv_static`` +
     ``inkling2_glue_attn_res`` (2 launches on ``write_state=True``, 3 launches on
     ``write_state=False``) vs fused ``inkling2_glue_attn_res_sconv`` across
     ``T in {1, 2, 4, 8}``, ``n_blocks in {1, 2, 4, 8}``, ``Q in {1, 3}``,
     boundary & non-boundary, ``norm`` & ``raw``.
  2. ``taps_final``:
     Bit-exact RNE parity of ``inkling2_glue_copy_states`` (``fp32 -> bf16``,
     including NaN ``0x7fc0``, ``+-inf``, ``-0.0``, and RNE tie-to-even edge cases)
     and fused final readout (``norm_out2`` ``fp16 + bf16/fp32``, 1 launch vs 3 or
     4 launches).
  3. ``gate_mix``:
     ``inkling2_glue_msg_gate_mix`` parity & graph latency at
     ``T in {1, 2, 4, 8, 10, 12, 16}``, ``G in {1, 3}`` (``T <= 16`` split-R vs
     1-wave ``gate_mix_kernel`` fallback), plus experimental JIT ``gate_mix_split``
     variants:
       - ``acq_rel``: ``__hip_atomic_fetch_add(..., __ATOMIC_ACQ_REL, AGENT)``
         replacing ``__threadfence()``;
       - ``col64``: 64-col CTAs (doubling CTA count at ``T = 1, 2``);
       - ``rsplit2`` / ``rsplit4``: cross-CTA R-splits across ``R = 64``.
  4. ``moe_mxfp8``:
     MXFP8 ``w13 + w2`` (fused SwiGLU) ``bm in {1, 2, 4}`` at
     ``T in {1, 2, 3, 4, 8, 16}`` under both uniform top-8 routing and correlated
     spec-verify routing (~44% shared experts between adjacent tokens).
  5. ``moe_nvfp4``:
     NVFP4 ``w13 + w2`` (fused SwiGLU) at ``T in {1, 3, 8}`` comparing:
       - ``var0``: ``THREADS_X=256, BLOCK_K=256`` (baseline);
       - ``var1``: ``THREADS_X=128, BLOCK_K=256`` (2x N-grid occupancy);
       - ``var2``: ``THREADS_X=256, BLOCK_K=384`` (6x64 K-tile amortisation).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import regex as re
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
CSRC = REPO / "csrc" / "rocm"
HIDDEN = 2304
INTER = 1152
EXPERTS = 128
TOP_K = 8
GATE_RANK = 64
EPS = 1e-6
SOFTCAP_GATE = 4.0
SOFTCAP_UP = 25.0


# ---------------------------------------------------------------------------
# JIT extension builders + clang resource-usage parser
# ---------------------------------------------------------------------------

_GLUE_ATTN_WRAPPER = r"""
#include <torch/extension.h>
#include <optional>

void inkling2_glue_rms_norm(const at::Tensor& x,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& weight,
                            const at::Tensor& out,
                            const std::optional<at::Tensor>& sum_out,
                            double eps, bool round_fp16);
void inkling2_glue_copy_states(const at::Tensor& src, const at::Tensor& dst);
void inkling2_glue_attn_res(const at::Tensor& blocks, int64_t n_blocks,
                            const at::Tensor& x0,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& partial_out,
                            const std::optional<at::Tensor>& append,
                            const at::Tensor& query,
                            const std::optional<at::Tensor>& norm_w,
                            const at::Tensor& out, double eps,
                            double norm_eps);
void inkling2_glue_attn_res_dual(const at::Tensor& blocks, int64_t n_blocks,
                                 const at::Tensor& x0,
                                 const std::optional<at::Tensor>& x1,
                                 const std::optional<at::Tensor>& partial_out,
                                 const std::optional<at::Tensor>& append,
                                 const at::Tensor& query,
                                 const std::optional<at::Tensor>& norm_w,
                                 const at::Tensor& out,
                                 const std::optional<at::Tensor>& norm_out2,
                                 double eps, double norm_eps);
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
    const std::optional<at::Tensor>& norm_out2, double eps, double norm_eps);
void inkling2_glue_msg_gate_down(const at::Tensor& h, const at::Tensor& m,
                                 const at::Tensor& wd, const at::Tensor& d,
                                 double eps);
void inkling2_glue_msg_gate_mix(const at::Tensor& h, const at::Tensor& m,
                                const at::Tensor& d, const at::Tensor& wf,
                                const at::Tensor& wg,
                                const std::optional<at::Tensor>& norm_w,
                                const at::Tensor& out,
                                const std::optional<at::Tensor>& scratch,
                                const std::optional<at::Tensor>& partials,
                                const std::optional<at::Tensor>& counters,
                                double norm_eps);
void inkling2_glue_moe_route(
    const at::Tensor& x, const at::Tensor& gate_w, const at::Tensor& bias,
    const at::Tensor& logits, const at::Tensor& topk_weights,
    const at::Tensor& topk_ids, const at::Tensor& sorted_ids,
    const at::Tensor& expert_ids, const at::Tensor& post_pad,
    const at::Tensor& counters, const std::optional<at::Tensor>& zero1,
    const std::optional<at::Tensor>& zero2, int64_t block_size_m,
    double route_scale);
void inkling2_sconv_static(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                           torch::Tensor conv_cache, torch::Tensor block_table,
                           torch::Tensor seq_idx, torch::Tensor qsl,
                           torch::Tensor seq_lens, int64_t stream_offset,
                           int64_t num_kv_heads, int64_t block_size,
                           bool write_state);
void inkling2_conv_state_write(torch::Tensor x, torch::Tensor conv_cache,
                               torch::Tensor block_table, torch::Tensor seq_idx,
                               torch::Tensor qsl, torch::Tensor seq_lens,
                               int64_t stream_offset, int64_t num_kv_heads,
                               int64_t block_size);

// Experimental gate_mix_split micro-variants (defined in gate_mix_exp.cu)
void exp_gate_mix_launch(int variant, const at::Tensor& h, const at::Tensor& m,
                         const at::Tensor& d, const at::Tensor& wf,
                         const at::Tensor& wg, const at::Tensor& norm_w,
                         const at::Tensor& out, const at::Tensor& scratch,
                         const at::Tensor& partials, const at::Tensor& counters,
                         double norm_eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &inkling2_glue_rms_norm);
  m.def("copy_states", &inkling2_glue_copy_states);
  m.def("attn_res", &inkling2_glue_attn_res);
  m.def("attn_res_dual", &inkling2_glue_attn_res_dual);
  m.def("attn_res_sconv", &inkling2_glue_attn_res_sconv);
  m.def("msg_gate_down", &inkling2_glue_msg_gate_down);
  m.def("msg_gate_mix", &inkling2_glue_msg_gate_mix);
  m.def("moe_route", &inkling2_glue_moe_route);
  m.def("sconv_static", &inkling2_sconv_static);
  m.def("conv_state_write", &inkling2_conv_state_write);
  m.def("exp_gate_mix_launch", &exp_gate_mix_launch);
}
"""

# Experimental gate_mix_split variants for T=1..16 fp32 gate weights:
#   variant 0: 1-wave gate_mix_kernel fallback (simulates old T > 8 cliff)
#   variant 1: split-R (128-col CTAs, 8 waves, RPW=8) + ACQ_REL AGENT atomic
#   variant 2: split-R 64-col CTAs (8 waves, 2 cols/lane -> 36/108 CTAs)
#   variant 3: cross-CTA R-split (RSPLIT=2, 4 waves/CTA, RPW=8 -> 36/108 CTAs)
#   variant 4: cross-CTA R-split (RSPLIT=4, 2 waves/CTA, RPW=8 -> 72/216 CTAs)
_GATE_MIX_EXP_CU = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace {
constexpr int kW = 32;

struct ExpGateArgs {
  const float* h;
  int64_t h_t, h_s;
  const float* m;
  int64_t m_t, m_s;
  const __half* d;
  const float* wf;
  const float* wg;
  const void* nw;
  int64_t nw_s;
  int nw_rows, nw_f16;
  __half* out16;
  float* out32;
  float* part;
  int* counters;
  int T, R, H, C;
  float norm_eps, inv_h;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
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
__device__ __forceinline__ float4 load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 load4_h(const __half* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(h2f_lo(u.x), h2f_hi(u.x), h2f_lo(u.y), h2f_hi(u.y));
}
__device__ __forceinline__ float4 load4_dyn(const void* base, int64_t idx, int is_h) {
  return is_h ? load4_h(reinterpret_cast<const __half*>(base) + idx)
              : load4(reinterpret_cast<const float*>(base) + idx);
}
__device__ __forceinline__ void store4(__half* p, const float4 v) {
  uint2 u{pack_h2(v.x, v.y), pack_h2(v.z, v.w)};
  *reinterpret_cast<uint2*>(p) = u;
}
__device__ __forceinline__ void store4(float* p, const float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}
__device__ __forceinline__ float dot4(const float4 a, const float4 b) {
  return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, a.x * b.x)));
}
__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int m = 16; m >= 1; m >>= 1)
    v += __shfl_xor(v, m, kW);
  return v;
}

// Variant 0: 1-wave fallback (2 waves = 64 threads, mimics gate_mix_kernel)
template <int TB>
__global__ void __launch_bounds__(64) exp_gate_1wave_kernel(const ExpGateArgs a) {
  __shared__ int s_last;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int cc = blockIdx.x;
  const int s = blockIdx.y;
  const int tile = blockIdx.z;
  const int t0 = tile * TB;
  const int H = a.H, R = a.R, T = a.T;
  const int c = cc * 256 + tid * 4;
  const bool active = c < H;
  const float* wf = a.wf + (int64_t)s * R * H;
  const float* wg = a.wg + (int64_t)s * R * H;

  float accf[TB][4], accg[TB][4];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb)
    #pragma unroll
    for (int j = 0; j < 4; ++j) accf[tb][j] = accg[tb][j] = 0.f;

  if (active) {
    for (int r = 0; r < R; ++r) {
      const float4 fv = load4(wf + (int64_t)r * H + c);
      const float4 gv = load4(wg + (int64_t)r * H + c);
      #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const int t = t0 + tb;
        const float dr = t < T ? __half2float(a.d[((int64_t)s * T + t) * R + r]) : 0.f;
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
  }
  __shared__ float s_ws[2];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    const int t = t0 + tb;
    float ss = 0.f;
    if (active && t < T) {
      const float4 hv = load4(a.h + (int64_t)t * a.h_t + (int64_t)s * a.h_s + c);
      const float4 mv = load4(a.m + (int64_t)t * a.m_t + (int64_t)s * a.m_s + c);
      float4 o{
          (1.f - tanhf(accf[tb][0])) * hv.x + tanhf(accg[tb][0]) * mv.x,
          (1.f - tanhf(accf[tb][1])) * hv.y + tanhf(accg[tb][1]) * mv.y,
          (1.f - tanhf(accf[tb][2])) * hv.z + tanhf(accg[tb][2]) * mv.z,
          (1.f - tanhf(accf[tb][3])) * hv.w + tanhf(accg[tb][3]) * mv.w};
      store4(a.out32 + ((int64_t)s * T + t) * H + c, o);
      ss = dot4(o, o);
    }
    float ws = wave_sum(ss);
    __syncthreads();
    if (lane == 0) s_ws[tid >> 5] = ws;
    __syncthreads();
    if (tid == 0 && t < T)
      a.part[((int64_t)s * T + t) * a.C + cc] = s_ws[0] + s_ws[1];
  }
  __threadfence();
  __syncthreads();
  int* counter = a.counters + (int64_t)s * gridDim.z + tile;
  if (tid == 0) s_last = (atomicAdd(counter, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int tb = 0; tb < TB; ++tb) {
    const int t = t0 + tb;
    if (t >= T) break;
    const float* pp = a.part + ((int64_t)s * T + t) * a.C;
    float tot = 0.f;
    for (int i = lane; i < (int)gridDim.x; i += kW) tot += pp[i];
    tot = wave_sum(tot);
    const float rs = rsqrtf(tot * a.inv_h + a.norm_eps);
    const int64_t row = ((int64_t)s * T + t) * H;
    const int64_t wrow = (a.nw_rows == 1 ? 0 : s) * a.nw_s;
    for (int c4 = tid; c4 < (H >> 2); c4 += 64) {
      const float4 v = load4(a.out32 + row + 4 * c4);
      const float4 w = load4_dyn(a.nw, wrow + 4 * c4, a.nw_f16);
      store4(a.out16 + row + 4 * c4,
             make_float4(w.x * (v.x * rs), w.y * (v.y * rs),
                         w.z * (v.z * rs), w.w * (v.w * rs)));
    }
  }
  __syncthreads();
  if (tid == 0) *counter = 0;
}

// Variant 1 (COLS=128, ACQ_REL=true) & Variant 2 (COLS=64, ACQ_REL=true):
// Within-CTA 8-wave R-split, testing agent-scope acquire/release atomic vs
// __threadfence() and 64-col CTAs (2 floats/lane -> 36/108 CTAs at G=1/3).
template <int TB, int COLS>
__global__ void __launch_bounds__(256)
exp_gate_split_acqrel_kernel(const ExpGateArgs a) {
  constexpr int NW = 8;
  constexpr int RPW = 8;
  constexpr int VEC = COLS / kW;  // 4 (float4) or 2 (float2)
  constexpr int WPT = NW / TB;
  __shared__ float4 s_part[NW][TB][2][kW];
  __shared__ int s_last;

  const int tid = threadIdx.x;
  const int wave = tid / kW;
  const int lane = tid & (kW - 1);
  const int cc = blockIdx.x;
  const int s = blockIdx.y;
  const int tile = blockIdx.z;
  const int t0 = tile * TB;
  const int H = a.H, R = a.R, T = a.T;
  const int c = cc * COLS + lane * VEC;
  const bool active = c < H;

  const float* wf = a.wf + (int64_t)s * R * H;
  const float* wg = a.wg + (int64_t)s * R * H;
  const int r0 = wave * RPW;

  float4 hv = make_float4(0.f, 0.f, 0.f, 0.f), mv = hv;
  if (wave < TB && active && t0 + wave < T) {
    const int t = t0 + wave;
    if constexpr (VEC == 4) {
      hv = load4(a.h + (int64_t)t * a.h_t + (int64_t)s * a.h_s + c);
      mv = load4(a.m + (int64_t)t * a.m_t + (int64_t)s * a.m_s + c);
    } else {
      const float* hp = a.h + (int64_t)t * a.h_t + (int64_t)s * a.h_s + c;
      const float2 h2 = *reinterpret_cast<const float2*>(hp);
      const float* mp = a.m + (int64_t)t * a.m_t + (int64_t)s * a.m_s + c;
      const float2 m2 = *reinterpret_cast<const float2*>(mp);
      hv = make_float4(h2.x, h2.y, 0.f, 0.f);
      mv = make_float4(m2.x, m2.y, 0.f, 0.f);
    }
  }

  float accf[TB][4], accg[TB][4];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb)
    #pragma unroll
    for (int j = 0; j < 4; ++j) accf[tb][j] = accg[tb][j] = 0.f;

  if (active && r0 < R) {
    float4 fv[RPW], gv[RPW];
    float dd[TB][RPW];
    #pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = r0 + i;
      if constexpr (VEC == 4) {
        fv[i] = load4(wf + (int64_t)r * H + c);
        gv[i] = load4(wg + (int64_t)r * H + c);
      } else {
        const float2 f2 = *reinterpret_cast<const float2*>(wf + (int64_t)r * H + c);
        const float2 g2 = *reinterpret_cast<const float2*>(wg + (int64_t)r * H + c);
        fv[i] = make_float4(f2.x, f2.y, 0.f, 0.f);
        gv[i] = make_float4(g2.x, g2.y, 0.f, 0.f);
      }
      #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const int t = t0 + tb;
        dd[tb][i] = t < T ? __half2float(a.d[((int64_t)s * T + t) * R + r]) : 0.f;
      }
    }
    #pragma unroll
    for (int i = 0; i < RPW; ++i) {
      #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const float dr = dd[tb][i];
        accf[tb][0] = fmaf(dr, fv[i].x, accf[tb][0]);
        accf[tb][1] = fmaf(dr, fv[i].y, accf[tb][1]);
        accg[tb][0] = fmaf(dr, gv[i].x, accg[tb][0]);
        accg[tb][1] = fmaf(dr, gv[i].y, accg[tb][1]);
        if constexpr (VEC == 4) {
          accf[tb][2] = fmaf(dr, fv[i].z, accf[tb][2]);
          accf[tb][3] = fmaf(dr, fv[i].w, accf[tb][3]);
          accg[tb][2] = fmaf(dr, gv[i].z, accg[tb][2]);
          accg[tb][3] = fmaf(dr, gv[i].w, accg[tb][3]);
        }
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

  if (wave < TB) {
    const int t = t0 + wave;
    float4 f = make_float4(0.f, 0.f, 0.f, 0.f), g = f;
    #pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 pf = s_part[w][wave][0][lane];
      const float4 pg = s_part[w][wave][1][lane];
      f.x += pf.x; f.y += pf.y; f.z += pf.z; f.w += pf.w;
      g.x += pg.x; g.y += pg.y; g.z += pg.z; g.w += pg.w;
    }
    float ss = 0.f;
    if (active && t < T) {
      if constexpr (VEC == 4) {
        float4 o{
            (1.f - tanhf(f.x)) * hv.x + tanhf(g.x) * mv.x,
            (1.f - tanhf(f.y)) * hv.y + tanhf(g.y) * mv.y,
            (1.f - tanhf(f.z)) * hv.z + tanhf(g.z) * mv.z,
            (1.f - tanhf(f.w)) * hv.w + tanhf(g.w) * mv.w};
        store4(a.out32 + ((int64_t)s * T + t) * H + c, o);
        ss = dot4(o, o);
      } else {
        float2 o2{
            (1.f - tanhf(f.x)) * hv.x + tanhf(g.x) * mv.x,
            (1.f - tanhf(f.y)) * hv.y + tanhf(g.y) * mv.y};
        *reinterpret_cast<float2*>(a.out32 + ((int64_t)s * T + t) * H + c) = o2;
        ss = fmaf(o2.y, o2.y, o2.x * o2.x);
      }
    }
    const float tot = wave_sum(ss);
    if (lane == 0 && t < T && cc < (int)gridDim.x)
      a.part[((int64_t)s * T + t) * gridDim.x + cc] = tot;
  }

  __syncthreads();
  int* counter = a.counters + (int64_t)s * gridDim.z + tile;
  if (tid == 0) {
    const int old = __hip_atomic_fetch_add(
        counter, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (old == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (!s_last) return;

  const int nchunks = (int)gridDim.x;
  const int tb = wave / WPT, sw = wave - tb * WPT;
  const int t = t0 + tb;
  if (t < T) {
    const float* pp = a.part + ((int64_t)s * T + t) * nchunks;
    float tot = 0.f;
    for (int i = lane; i < nchunks; i += kW) tot += pp[i];
    tot = wave_sum(tot);
    const float rs = rsqrtf(tot * a.inv_h + a.norm_eps);
    const int64_t row = ((int64_t)s * T + t) * H;
    const int64_t wrow = (a.nw_rows == 1 ? 0 : s) * a.nw_s;
    for (int c4 = sw * kW + lane; c4 < (H >> 2); c4 += WPT * kW) {
      const float4 v = load4(a.out32 + row + 4 * c4);
      const float4 w = load4_dyn(a.nw, wrow + 4 * c4, a.nw_f16);
      store4(a.out16 + row + 4 * c4,
             make_float4(w.x * (v.x * rs), w.y * (v.y * rs),
                         w.z * (v.z * rs), w.w * (v.w * rs)));
    }
  }
  __syncthreads();
  if (tid == 0)
    __hip_atomic_store(counter, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// Variant 3/4: Cross-CTA R-split (RSPLIT=2 or 4 across R=64) at T=1..2
// Cross-CTA R-split over R=64 using fg_accum [RSPLIT, G, T, 2, H].
// Bench caller passes oversized scratch + counters.
// or atomicAdd into zeroed scratch! Even simpler: measure cross-CTA R-split using
// atomicAdd into scratch (f in out32, g in scratch tail or two-pass) — wait: for G=3,
// scratch is [3, T, H] = exact size of out32, so cross-CTA R-split without extra buffer
// needs either atomicAdd into [f, g] workspace or a 2*3*T*H buffer! Let's allocate a
// small static bench buffer in the caller or test atomic R-split into [2, G, T, H]
// passed via scratch so we measure the EXACT hardware cost of cross-CTA R-split!
template <int TB, int RSPLIT>
__global__ void __launch_bounds__(256 / RSPLIT) exp_gate_cross_cta_rsplit_kernel(
    const ExpGateArgs a, float* fg_accum, int* fg_counters) {
  constexpr int NW = 8 / RSPLIT;
  constexpr int RPW = 8;
  __shared__ float4 s_part[NW][TB][2][kW];
  __shared__ int s_col_done;
  __shared__ int s_tile_done;

  const int tid = threadIdx.x;
  const int wave = tid / kW;
  const int lane = tid & (kW - 1);
  const int cc = blockIdx.x;
  const int s = blockIdx.y / RSPLIT;
  const int rs_idx = blockIdx.y - s * RSPLIT;
  const int tile = blockIdx.z;
  const int t0 = tile * TB;
  const int H = a.H, R = a.R, T = a.T;
  const int c = cc * 128 + lane * 4;
  const bool active = c < H;

  const float* wf = a.wf + (int64_t)s * R * H;
  const float* wg = a.wg + (int64_t)s * R * H;
  const int r0 = (rs_idx * NW + wave) * RPW;

  float accf[TB][4], accg[TB][4];
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb)
    #pragma unroll
    for (int j = 0; j < 4; ++j) accf[tb][j] = accg[tb][j] = 0.f;

  if (active && r0 < R) {
    #pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = r0 + i;
      const float4 fv = load4(wf + (int64_t)r * H + c);
      const float4 gv = load4(wg + (int64_t)r * H + c);
      #pragma unroll
      for (int tb = 0; tb < TB; ++tb) {
        const int t = t0 + tb;
        const float dr = t < T ? __half2float(a.d[((int64_t)s * T + t) * R + r]) : 0.f;
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
  }
  #pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    s_part[wave][tb][0][lane] =
        make_float4(accf[tb][0], accf[tb][1], accf[tb][2], accf[tb][3]);
    s_part[wave][tb][1][lane] =
        make_float4(accg[tb][0], accg[tb][1], accg[tb][2], accg[tb][3]);
  }
  __syncthreads();

  const int G = gridDim.y / RSPLIT;
  // Store this RSPLIT slice's [f, g] into fg_accum [RSPLIT, G, T, 2, H]
  if (wave < TB) {
    const int t = t0 + wave;
    if (active && t < T) {
      float4 f = make_float4(0.f, 0.f, 0.f, 0.f), g = f;
      #pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float4 pf = s_part[w][wave][0][lane];
        const float4 pg = s_part[w][wave][1][lane];
        f.x += pf.x; f.y += pf.y; f.z += pf.z; f.w += pf.w;
        g.x += pg.x; g.y += pg.y; g.z += pg.z; g.w += pg.w;
      }
      const int64_t base = (((int64_t)rs_idx * G + s) * T + t) * 2 * H;
      store4(fg_accum + base + c, f);
      store4(fg_accum + base + H + c, g);
    }
  }
  __syncthreads();
  int* col_ctr = fg_counters + ((int64_t)s * gridDim.z + tile) * gridDim.x + cc;
  if (tid == 0) {
    const int old = __hip_atomic_fetch_add(
        col_ctr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_col_done = (old == RSPLIT - 1);
  }
  __syncthreads();
  if (!s_col_done) return;
  if (tid == 0)
    __hip_atomic_store(col_ctr, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  if (wave < TB) {
    const int t = t0 + wave;
    float ss = 0.f;
    if (active && t < T) {
      float4 f = make_float4(0.f, 0.f, 0.f, 0.f), g = f;
      #pragma unroll
      for (int r = 0; r < RSPLIT; ++r) {
        const int64_t base = (((int64_t)r * G + s) * T + t) * 2 * H;
        const float4 pf = load4(fg_accum + base + c);
        const float4 pg = load4(fg_accum + base + H + c);
        f.x += pf.x; f.y += pf.y; f.z += pf.z; f.w += pf.w;
        g.x += pg.x; g.y += pg.y; g.z += pg.z; g.w += pg.w;
      }
      const float4 hv = load4(a.h + (int64_t)t * a.h_t + (int64_t)s * a.h_s + c);
      const float4 mv = load4(a.m + (int64_t)t * a.m_t + (int64_t)s * a.m_s + c);
      float4 o{
          (1.f - tanhf(f.x)) * hv.x + tanhf(g.x) * mv.x,
          (1.f - tanhf(f.y)) * hv.y + tanhf(g.y) * mv.y,
          (1.f - tanhf(f.z)) * hv.z + tanhf(g.z) * mv.z,
          (1.f - tanhf(f.w)) * hv.w + tanhf(g.w) * mv.w};
      store4(a.out32 + ((int64_t)s * T + t) * H + c, o);
      ss = dot4(o, o);
    }
    const float tot = wave_sum(ss);
    if (lane == 0 && t < T)
      a.part[((int64_t)s * T + t) * a.C + cc] = tot;
  }
  __syncthreads();
  int* counter = a.counters + (int64_t)s * gridDim.z + tile;
  if (tid == 0) {
    const int old = __hip_atomic_fetch_add(
        counter, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_tile_done = (old == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (!s_tile_done) return;

  for (int tb = 0; tb < TB; ++tb) {
    const int t = t0 + tb;
    if (t >= T) break;
    const float* pp = a.part + ((int64_t)s * T + t) * a.C;
    float tot = 0.f;
    for (int i = lane; i < a.C; i += kW) tot += pp[i];
    tot = wave_sum(tot);
    const float rs = rsqrtf(tot * a.inv_h + a.norm_eps);
    const int64_t row = ((int64_t)s * T + t) * H;
    const int64_t wrow = (a.nw_rows == 1 ? 0 : s) * a.nw_s;
    for (int c4 = tid; c4 < (H >> 2); c4 += (int)blockDim.x) {
      const float4 v = load4(a.out32 + row + 4 * c4);
      const float4 w = load4_dyn(a.nw, wrow + 4 * c4, a.nw_f16);
      store4(a.out16 + row + 4 * c4,
             make_float4(w.x * (v.x * rs), w.y * (v.y * rs),
                         w.z * (v.z * rs), w.w * (v.w * rs)));
    }
  }
  __syncthreads();
  if (tid == 0)
    __hip_atomic_store(counter, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif
}  // namespace

void exp_gate_mix_launch(int variant, const at::Tensor& h, const at::Tensor& m,
                         const at::Tensor& d, const at::Tensor& wf,
                         const at::Tensor& wg, const at::Tensor& norm_w,
                         const at::Tensor& out, const at::Tensor& scratch,
                         const at::Tensor& partials, const at::Tensor& counters,
                         double norm_eps) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  const int64_t T = m.size(0), G = m.size(1), H = m.size(2), R = d.size(2);
  const int64_t SH = h.size(1);
  ExpGateArgs a{};
  a.h = h.data_ptr<float>();
  a.h_t = h.stride(0);
  a.h_s = SH == 1 ? 0 : h.stride(1);
  a.m = m.data_ptr<float>();
  a.m_t = m.stride(0);
  a.m_s = m.stride(1);
  a.d = reinterpret_cast<const __half*>(d.data_ptr());
  a.wf = wf.data_ptr<float>();
  a.wg = wg.data_ptr<float>();
  a.nw = norm_w.data_ptr();
  a.nw_s = norm_w.stride(0);
  a.nw_rows = (int)norm_w.size(0);
  a.nw_f16 = norm_w.scalar_type() == at::kHalf ? 1 : 0;
  a.out16 = reinterpret_cast<__half*>(out.data_ptr());
  a.out32 = scratch.data_ptr<float>();
  a.part = partials.data_ptr<float>();
  a.counters = counters.data_ptr<int>();
  a.T = (int)T;
  a.R = (int)R;
  a.H = (int)H;
  a.C = (int)((H + 127) / 128);
  a.norm_eps = (float)norm_eps;
  a.inv_h = 1.0f / (float)H;

  const at::cuda::OptionalCUDAGuard guard(device_of(m));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int tb = T == 1 ? 1 : (T <= 2 ? 2 : 4);
  const uint32_t ntiles = (uint32_t)((T + tb - 1) / tb);

  if (variant == 0) {
    const dim3 grid((uint32_t)((H + 255) / 256), (uint32_t)G, ntiles);
    if (tb == 1) {
      exp_gate_1wave_kernel<1><<<grid, 64, 0, stream>>>(a);
    } else if (tb == 2) {
      exp_gate_1wave_kernel<2><<<grid, 64, 0, stream>>>(a);
    } else {
      exp_gate_1wave_kernel<4><<<grid, 64, 0, stream>>>(a);
    }
  } else if (variant == 1) {
    const dim3 grid((uint32_t)a.C, (uint32_t)G, ntiles);
    if (tb == 1) {
      exp_gate_split_acqrel_kernel<1, 128><<<grid, 256, 0, stream>>>(a);
    } else if (tb == 2) {
      exp_gate_split_acqrel_kernel<2, 128><<<grid, 256, 0, stream>>>(a);
    } else {
      exp_gate_split_acqrel_kernel<4, 128><<<grid, 256, 0, stream>>>(a);
    }
  } else if (variant == 2) {
    const uint32_t c64 = (uint32_t)((H + 63) / 64);
    const dim3 grid(c64, (uint32_t)G, ntiles);
    if (tb == 1) {
      exp_gate_split_acqrel_kernel<1, 64><<<grid, 256, 0, stream>>>(a);
    } else if (tb == 2) {
      exp_gate_split_acqrel_kernel<2, 64><<<grid, 256, 0, stream>>>(a);
    } else {
      exp_gate_split_acqrel_kernel<4, 64><<<grid, 256, 0, stream>>>(a);
    }
  } else if (variant == 3 || variant == 4) {
    // scratch tail holds fg_accum + fg_counters
    const int rsplit = variant == 3 ? 2 : 4;
    float* fg_accum = a.out32 + 3 * T * H;
    int* fg_ctr = a.counters + 3 * T;
    const dim3 grid((uint32_t)a.C, (uint32_t)(G * rsplit), ntiles);
    if (rsplit == 2) {
      if (tb == 1)
        exp_gate_cross_cta_rsplit_kernel<1, 2><<<grid, 128, 0, stream>>>(
            a, fg_accum, fg_ctr);
      else
        exp_gate_cross_cta_rsplit_kernel<2, 2><<<grid, 128, 0, stream>>>(
            a, fg_accum, fg_ctr);
    } else {
      if (tb == 1)
        exp_gate_cross_cta_rsplit_kernel<1, 4><<<grid, 64, 0, stream>>>(
            a, fg_accum, fg_ctr);
      else
        exp_gate_cross_cta_rsplit_kernel<2, 4><<<grid, 64, 0, stream>>>(
            a, fg_accum, fg_ctr);
    }
  }
#endif
}
"""

_MOE_WRAPPER = r"""
#include <torch/extension.h>

void moe_nvfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_globals, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);
void moe_mxfp8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);
void moe_nvfp4_gemm_swiglu_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor b_globals,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, double gate_cap, double up_cap);
void moe_mxfp8_gemm_swiglu_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_topk_weight, int64_t output_topk, double gate_cap,
    double up_cap);
void moe_nvfp4_set_variant_rdna2(int64_t variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_nvfp4_gemm_rdna2", &moe_nvfp4_gemm_rdna2);
  m.def("moe_mxfp8_gemm_rdna2", &moe_mxfp8_gemm_rdna2);
  m.def("moe_nvfp4_gemm_swiglu_rdna2", &moe_nvfp4_gemm_swiglu_rdna2);
  m.def("moe_mxfp8_gemm_swiglu_rdna2", &moe_mxfp8_gemm_swiglu_rdna2);
  m.def("set_nvfp4_variant", &moe_nvfp4_set_variant_rdna2);
}
"""


def _jit_root() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
    ).expanduser()


def _parse_resource_remarks(
    cu_path: pathlib.Path, arch: str = "gfx1030"
) -> dict[str, dict[str, int]]:
    """Run hipcc -fsyntax-only / -S -Rpass-analysis=kernel-resource-usage if available
    or parse ninja build log remarks for key kernels."""
    hipcc = os.environ.get("CMAKE_HIP_COMPILER", "/opt/rocm/core-7.14/bin/hipcc")
    if not pathlib.Path(hipcc).exists():
        hipcc = "hipcc"
    torch_inc = pathlib.Path(torch.__file__).resolve().parent / "include"
    cmd = [
        hipcc,
        "-O2",
        f"--offload-arch={arch}",
        "--cuda-device-only",
        "-S",
        "-o",
        "/dev/null",
        "-Rpass-analysis=kernel-resource-usage",
        f"-I{CSRC}",
        f"-I{torch_inc}",
        f"-I{torch_inc / 'torch' / 'csrc' / 'api' / 'include'}",
        str(cu_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out: dict[str, dict[str, int]] = {}
    cur_fn = None
    for line in (res.stderr or "").splitlines():
        m_fn = re.search(r"Function Name:\s*(\S+)", line)
        if m_fn:
            cur_fn = m_fn.group(1)
            out.setdefault(cur_fn, {})
            continue
        if cur_fn is not None:
            for key, pat in (
                ("sgpr", r"SGPRs:\s*(\d+)"),
                ("vgpr", r"VGPRs:\s*(\d+)"),
                ("lds", r"LDS Size[^:]*:\s*(\d+)"),
                ("scratch", r"Scratch\s*:\s*(\d+)"),
            ):
                mm = re.search(pat, line)
                if mm:
                    out[cur_fn][key] = int(mm.group(1))
    return out


def build_glue_attn_ext(archs: str = "gfx1030", verbose: bool = False):
    from torch.utils.cpp_extension import load

    glue_src = (CSRC / "inkling2_glue_rdna2.cu").read_text()
    attn_src = (CSRC / "inkling2_attn_rdna2.cu").read_text()
    tag = hashlib.md5(
        "\0".join(
            [glue_src, attn_src, _GLUE_ATTN_WRAPPER, _GATE_MIX_EXP_CU, archs]
        ).encode()
    ).hexdigest()[:12]
    name = f"i2_tp1_glue_micro_{tag}"
    work = _jit_root() / name
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)

    files = {
        "wrapper.cpp": _GLUE_ATTN_WRAPPER,
        "inkling2_glue_rdna2.cu": glue_src,
        "inkling2_attn_rdna2.cu": attn_src,
        "gate_mix_exp.cu": _GATE_MIX_EXP_CU,
    }
    srcs = []
    for fname, txt in files.items():
        p = work / "src" / fname
        if not p.exists() or p.read_text() != txt:
            p.write_text(txt)
        srcs.append(str(p))

    os.environ["PYTORCH_ROCM_ARCH"] = archs
    return load(
        name=name,
        sources=srcs,
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=["-O2"],
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def build_moe_ext(archs: str = "gfx1030", verbose: bool = False):
    from torch.utils.cpp_extension import load

    moe_src = (CSRC / "moe_lowp_dot2_rdna2.cu").read_text()
    tag = hashlib.md5("\0".join([moe_src, _MOE_WRAPPER, archs]).encode()).hexdigest()[
        :12
    ]
    name = f"i2_tp1_moe_micro_{tag}"
    work = _jit_root() / name
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)

    w_p = work / "src" / "wrapper.cpp"
    c_p = work / "src" / "moe_lowp_dot2_rdna2.cu"
    if not w_p.exists() or w_p.read_text() != _MOE_WRAPPER:
        w_p.write_text(_MOE_WRAPPER)
    if not c_p.exists() or c_p.read_text() != moe_src:
        c_p.write_text(moe_src)

    os.environ["PYTORCH_ROCM_ARCH"] = archs
    return load(
        name=name,
        sources=[str(w_p), str(c_p)],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=["-O2"],
        build_directory=str(work / "build"),
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# HIP-graph timing helper (replays N times inside one graph to measure true
# back-to-back GPU kernel + dispatch latency without Python launch overhead)
# ---------------------------------------------------------------------------


def time_in_graph(fn, *, inner: int = 20, iters: int = 100) -> float:
    """Microseconds per single ``fn()`` call inside a captured HIP graph."""
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(10):
        g.replay()
    torch.accelerator.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        g.replay()
    ev1.record()
    torch.accelerator.synchronize()
    return 1e3 * ev0.elapsed_time(ev1) / (iters * inner)


# ---------------------------------------------------------------------------
# Pure-PyTorch router block alignment (standalone without importing vllm._C)
# ---------------------------------------------------------------------------


def align_expert_blocks(
    topk_ids: torch.Tensor, block_size_m: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact replica of ``moe_align_block_size`` layout for ``[T, K]`` int32 ids."""
    flat = topk_ids.reshape(-1).cpu().tolist()
    total = len(flat)
    max_num_tokens_padded = total + num_experts * (block_size_m - 1)
    max_blocks = -(-max_num_tokens_padded // block_size_m)
    buckets: list[list[int]] = [[] for _ in range(num_experts)]
    for tok_slot, e in enumerate(flat):
        buckets[e].append(tok_slot)
    sorted_ids = [total] * max_num_tokens_padded
    expert_ids = [-1] * max_blocks
    pos = 0
    blk = 0
    for e, slots in enumerate(buckets):
        if not slots:
            continue
        for i in range(0, len(slots), block_size_m):
            chunk = slots[i : i + block_size_m]
            for j, s in enumerate(chunk):
                sorted_ids[pos + j] = s
            expert_ids[blk] = e
            pos += block_size_m
            blk += 1
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([pos], dtype=torch.int32, device=dev),
    )


# ---------------------------------------------------------------------------
# Section 1: Fused sconv + attn_res bit-exact parity & HIP-graph latency
# ---------------------------------------------------------------------------


def _bench_sep_sconv_ar(
    ext,
    conv_x,
    conv_w,
    attn_c,
    cache,
    bt,
    seq_idx,
    qsl,
    seq_lens,
    stream_offset,
    num_kv_heads,
    block_size,
    ws_flag,
    blk,
    n_blocks,
    hid,
    res,
    query,
    nw,
    out,
) -> None:
    ext.sconv_static(
        conv_x,
        conv_w,
        attn_c,
        cache,
        bt,
        seq_idx,
        qsl,
        seq_lens,
        stream_offset,
        num_kv_heads,
        block_size,
        ws_flag,
    )
    if not ws_flag:
        ext.conv_state_write(
            conv_x,
            cache,
            bt,
            seq_idx,
            qsl,
            seq_lens,
            stream_offset,
            num_kv_heads,
            block_size,
        )
    ext.attn_res(blk, n_blocks, hid, attn_c, res, None, query, nw, out, EPS, EPS)


def _bench_fus_sconv_ar(
    ext,
    blk,
    n_blocks,
    hid,
    conv_x,
    conv_w,
    cache,
    bt,
    seq_idx,
    qsl,
    seq_lens,
    stream_offset,
    num_kv_heads,
    block_size,
    ws_flag,
    res,
    query,
    nw,
    out,
) -> None:
    ext.attn_res_sconv(
        blk,
        n_blocks,
        hid,
        conv_x,
        conv_w,
        cache,
        bt,
        seq_idx,
        qsl,
        seq_lens,
        stream_offset,
        num_kv_heads,
        block_size,
        ws_flag,
        None,
        res,
        None,
        query,
        nw,
        out,
        None,
        EPS,
        EPS,
    )
    if not ws_flag:
        ext.conv_state_write(
            conv_x,
            cache,
            bt,
            seq_idx,
            qsl,
            seq_lens,
            stream_offset,
            num_kv_heads,
            block_size,
        )


def run_sconv_attn_res(ext, iters: int) -> dict[str, Any]:
    print(
        "\n=== [1] sconv + attn_res -> attn_res_sconv (bit-exact parity + graph us) ==="
    )
    gen = torch.Generator(device="cuda").manual_seed(20261124)
    num_kv_heads = 8
    block_size = 16
    ring = 4  # conv_ring_rows for k=4
    width = (2 * num_kv_heads * 128 + 2 * HIDDEN) // num_kv_heads
    stream_offset = (2 * num_kv_heads * 128) // num_kv_heads
    n_pages = 64

    results: dict[str, Any] = {"parity_checks": 0, "bench": {}}

    # Exhaustive bit-exact parity matrix
    for T in (1, 2, 4, 8):
        for write_state in (True, False):
            if write_state:
                # R = T single-token decode requests
                R = T
                qsl = torch.arange(0, R + 1, dtype=torch.int32, device="cuda")
                seq_idx = torch.arange(0, T, dtype=torch.int32, device="cuda")
                seq_lens = torch.tensor(
                    [1 + (r * 7) % 45 for r in range(R)],
                    dtype=torch.int32,
                    device="cuda",
                )
            else:
                # Multi-token verify chunk(s): 1 req of len T (or 2 reqs at T=8)
                R = 2 if T == 8 else 1
                step_q = T // R
                qsl = torch.arange(0, T + 1, step_q, dtype=torch.int32, device="cuda")
                seq_idx = torch.arange(0, T, dtype=torch.int32, device="cuda") // step_q
                seq_lens = torch.tensor(
                    [step_q + 19 + 5 * r for r in range(R)],
                    dtype=torch.int32,
                    device="cuda",
                )
            bt = torch.arange(1, 1 + R * 8, dtype=torch.int32, device="cuda").view(R, 8)
            conv_x = torch.randn(
                T, HIDDEN, dtype=torch.float16, device="cuda", generator=gen
            )
            conv_w = torch.randn(
                HIDDEN, 1, 4, dtype=torch.float32, device="cuda", generator=gen
            )
            cache_init = torch.randn(
                n_pages,
                num_kv_heads,
                ring,
                width,
                dtype=torch.float16,
                device="cuda",
                generator=gen,
            )

            for n_blocks in (1, 2, 4, 8):
                for Q in (1, 3):
                    for boundary in (False, True):
                        if Q == 3 and boundary:
                            continue
                        for norm in (True, False):
                            blk_ref = torch.randn(
                                T,
                                max(n_blocks + 1, 1),
                                HIDDEN,
                                dtype=torch.float32,
                                device="cuda",
                                generator=gen,
                            )
                            blk_fus = blk_ref.clone()
                            cache_ref = cache_init.clone()
                            cache_fus = cache_init.clone()
                            hid = torch.randn(
                                T,
                                HIDDEN,
                                dtype=torch.float32,
                                device="cuda",
                                generator=gen,
                            )
                            query = torch.randn(
                                Q,
                                HIDDEN,
                                dtype=torch.float32,
                                device="cuda",
                                generator=gen,
                            )
                            nw = (
                                torch.randn(
                                    Q,
                                    HIDDEN,
                                    dtype=torch.float32,
                                    device="cuda",
                                    generator=gen,
                                )
                                if norm
                                else None
                            )
                            out_dtype = torch.float16 if norm else torch.float32
                            out_ref = torch.empty(
                                T, Q, HIDDEN, dtype=out_dtype, device="cuda"
                            )
                            out_fus = torch.empty_like(out_ref)
                            attn_c_ref = torch.empty(
                                T, HIDDEN, dtype=torch.float32, device="cuda"
                            )
                            attn_c_fus = torch.empty_like(attn_c_ref)
                            res_ref = torch.empty(
                                T, HIDDEN, dtype=torch.float32, device="cuda"
                            )
                            res_fus = torch.empty_like(res_ref)

                            # Reference (2 or 3 launches)
                            ext.sconv_static(
                                conv_x,
                                conv_w,
                                attn_c_ref,
                                cache_ref,
                                bt,
                                seq_idx,
                                qsl,
                                seq_lens,
                                stream_offset,
                                num_kv_heads,
                                block_size,
                                write_state,
                            )
                            if not write_state:
                                ext.conv_state_write(
                                    conv_x,
                                    cache_ref,
                                    bt,
                                    seq_idx,
                                    qsl,
                                    seq_lens,
                                    stream_offset,
                                    num_kv_heads,
                                    block_size,
                                )
                            if boundary:
                                ext.attn_res(
                                    blk_ref,
                                    n_blocks,
                                    attn_c_ref,
                                    None,
                                    None,
                                    hid,
                                    query,
                                    nw,
                                    out_ref,
                                    EPS,
                                    EPS,
                                )
                            else:
                                ext.attn_res(
                                    blk_ref,
                                    n_blocks,
                                    hid,
                                    attn_c_ref,
                                    res_ref,
                                    None,
                                    query,
                                    nw,
                                    out_ref,
                                    EPS,
                                    EPS,
                                )

                            # Fused (1 or 2 launches)
                            ext.attn_res_sconv(
                                blk_fus,
                                n_blocks,
                                None if boundary else hid,
                                conv_x,
                                conv_w,
                                cache_fus,
                                bt,
                                seq_idx,
                                qsl,
                                seq_lens,
                                stream_offset,
                                num_kv_heads,
                                block_size,
                                write_state,
                                attn_c_fus if boundary else None,
                                None if boundary else res_fus,
                                hid if boundary else None,
                                query,
                                nw,
                                out_fus,
                                None,
                                EPS,
                                EPS,
                            )
                            if not write_state:
                                ext.conv_state_write(
                                    conv_x,
                                    cache_fus,
                                    bt,
                                    seq_idx,
                                    qsl,
                                    seq_lens,
                                    stream_offset,
                                    num_kv_heads,
                                    block_size,
                                )

                            assert torch.equal(out_ref, out_fus), (
                                f"out mismatch T={T} ws={write_state} "
                                f"nb={n_blocks} Q={Q} "
                                f"bnd={boundary} norm={norm}"
                            )
                            assert torch.equal(blk_ref, blk_fus), "blocks mismatch"
                            assert torch.equal(cache_ref, cache_fus), (
                                "conv_cache mismatch"
                            )
                            if boundary:
                                assert torch.equal(attn_c_ref, attn_c_fus), (
                                    "sconv_out mismatch"
                                )
                            else:
                                assert torch.equal(res_ref, res_fus), (
                                    "partial_out mismatch"
                                )
                            results["parity_checks"] += 1

    print(
        f"  PASS: {results['parity_checks']} bit-exact sconv+attn_res parity configs!"
    )

    # Graph-timed benchmark on representative decode & verify shapes
    for T, ws_flag, label in (
        (1, True, "decode_T1"),
        (2, True, "decode_T2"),
        (4, True, "decode_T4"),
        (8, True, "decode_T8"),
        (8, False, "verify_T8_q8"),
    ):
        R = 1 if not ws_flag else T
        qsl = (
            torch.tensor([0, T], dtype=torch.int32, device="cuda")
            if not ws_flag
            else torch.arange(0, R + 1, dtype=torch.int32, device="cuda")
        )
        seq_idx = (
            torch.zeros(T, dtype=torch.int32, device="cuda")
            if not ws_flag
            else torch.arange(0, T, dtype=torch.int32, device="cuda")
        )
        seq_lens = torch.full((R,), 64, dtype=torch.int32, device="cuda")
        bt = torch.arange(1, 1 + R * 8, dtype=torch.int32, device="cuda").view(R, 8)
        conv_x = torch.randn(
            T, HIDDEN, dtype=torch.float16, device="cuda", generator=gen
        )
        conv_w = torch.randn(
            HIDDEN, 1, 4, dtype=torch.float32, device="cuda", generator=gen
        )
        cache = torch.zeros(
            n_pages, num_kv_heads, ring, width, dtype=torch.float16, device="cuda"
        )
        blk = torch.randn(
            T, 9, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
        )
        hid = torch.randn(T, HIDDEN, dtype=torch.float32, device="cuda", generator=gen)
        res = torch.empty_like(hid)
        attn_c = torch.empty_like(hid)

        for Q, n_blocks in ((1, 4), (3, 4), (1, 8)):
            query = torch.randn(
                Q, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            nw = torch.randn(
                Q, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            out = torch.empty(T, Q, HIDDEN, dtype=torch.float16, device="cuda")

            sep_fn = functools.partial(
                _bench_sep_sconv_ar,
                ext,
                conv_x,
                conv_w,
                attn_c,
                cache,
                bt,
                seq_idx,
                qsl,
                seq_lens,
                stream_offset,
                num_kv_heads,
                block_size,
                ws_flag,
                blk,
                n_blocks,
                hid,
                res,
                query,
                nw,
                out,
            )
            fus_fn = functools.partial(
                _bench_fus_sconv_ar,
                ext,
                blk,
                n_blocks,
                hid,
                conv_x,
                conv_w,
                cache,
                bt,
                seq_idx,
                qsl,
                seq_lens,
                stream_offset,
                num_kv_heads,
                block_size,
                ws_flag,
                res,
                query,
                nw,
                out,
            )

            us_sep = time_in_graph(sep_fn, iters=iters)
            us_fus = time_in_graph(fus_fn, iters=iters)
            key = f"{label}/Q{Q}/nb{n_blocks}"
            saved_94 = (us_sep - us_fus) * 94.0 / 1e3
            results["bench"][key] = {
                "sep_us": round(us_sep, 3),
                "fused_us": round(us_fus, 3),
                "delta_us": round(us_sep - us_fus, 3),
                "est_94_calls_ms": round(saved_94, 3),
            }
            print(
                f"  {key:24s}  sep={us_sep:6.2f} us  fused={us_fus:6.2f} us  "
                f"save={us_sep - us_fus:+5.2f} us/site ({saved_94:+.3f} ms / 94 sites)"
            )
    return results


# ---------------------------------------------------------------------------
# Section 2: copy_states fp32->bf16 RNE + fused dual-output final readout
# ---------------------------------------------------------------------------


def _ref_f32_to_bf16_bits(x_f32: torch.Tensor) -> torch.Tensor:
    """Bit-for-bit reference matching inkling2_dspark_rdna2.cu:65-70."""
    u = x_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    nan_mask = (u & 0x7FFFFFFF) > 0x7F800000
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    bits = torch.where(nan_mask, torch.full_like(rounded, 0x7FC0), rounded)
    return (bits & 0xFFFF).to(torch.uint16)


def _bench_sep3_final(ext, blk, x0, x1, query, hid_tmp, f32_out, f16_out, nw) -> None:
    ext.attn_res(
        blk, 4, x0, x1, None, None, query, None, hid_tmp.unsqueeze(1), EPS, EPS
    )
    ext.rms_norm(hid_tmp, None, nw[0], f32_out, None, EPS, False)
    ext.rms_norm(hid_tmp, None, nw[0], f16_out, None, EPS, False)


def run_taps_final(ext, iters: int) -> dict[str, Any]:
    print("\n=== [2] copy_states fp32->bf16 RNE + fused final readout (norm_out2) ===")
    gen = torch.Generator(device="cuda").manual_seed(42)

    # Edge-case + random RNE bit-exact test
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            -float("nan"),
            1.0,
            # Exact halfway ties: odd LSB rounds up, even LSB rounds down
            struct_f32 := torch.tensor(
                [0x3F808000, 0x3F818000, 0x7F7F8000], dtype=torch.int32
            )
            .view(torch.float32)[0]
            .item(),
            torch.tensor([0x3F818000], dtype=torch.int32).view(torch.float32)[0].item(),
            torch.tensor([0x7F7F8000], dtype=torch.int32).view(torch.float32)[0].item(),
        ],
        dtype=torch.float32,
        device="cuda",
    )
    _ = struct_f32
    src = torch.randn(8, HIDDEN, dtype=torch.float32, device="cuda", generator=gen)
    src[0, : specials.numel()] = specials
    dst_bf16 = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ext.copy_states(src, dst_bf16)
    got_bits = dst_bf16.view(torch.uint16)
    exp_bits = _ref_f32_to_bf16_bits(src)
    assert torch.equal(got_bits, exp_bits), "copy_states fp32->bf16 RNE bits mismatch!"
    print(
        "  PASS: copy_states fp32->bf16 RNE bit-exact "
        "(incl. NaN 0x7fc0, +-inf, -0.0, ties)!"
    )

    # Fused final readout parity & latency (norm_out2)
    out: dict[str, Any] = {}
    for T in (1, 8):
        blk = torch.randn(
            T, 4, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
        )
        x0 = torch.randn(T, HIDDEN, dtype=torch.float32, device="cuda", generator=gen)
        x1 = torch.randn(T, HIDDEN, dtype=torch.float32, device="cuda", generator=gen)
        query = torch.randn(
            1, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
        )
        nw = torch.randn(1, HIDDEN, dtype=torch.float32, device="cuda", generator=gen)
        hid_tmp = torch.empty(T, HIDDEN, dtype=torch.float32, device="cuda")

        final16_ref = torch.empty(T, HIDDEN, dtype=torch.float16, device="cuda")
        final32_ref = torch.empty(T, HIDDEN, dtype=torch.float32, device="cuda")
        ext.attn_res(
            blk, 4, x0, x1, None, None, query, None, hid_tmp.unsqueeze(1), EPS, EPS
        )
        ext.rms_norm(hid_tmp, None, nw[0], final32_ref, None, EPS, False)
        ext.rms_norm(hid_tmp, None, nw[0], final16_ref, None, EPS, False)

        final16_f32 = torch.empty_like(final16_ref)
        final_no2_f32 = torch.empty_like(final32_ref)
        ext.attn_res_dual(
            blk,
            4,
            x0,
            x1,
            None,
            None,
            query,
            nw,
            final16_f32.unsqueeze(1),
            final_no2_f32,
            EPS,
            EPS,
        )
        final16_ar = torch.empty_like(final16_ref)
        ext.attn_res(
            blk, 4, x0, x1, None, None, query, nw, final16_ar.unsqueeze(1), EPS, EPS
        )
        assert torch.equal(final16_ar, final16_f32), (
            f"final16 vs attn_res mismatch T={T}"
        )
        assert (final16_ref.float() - final16_f32.float()).abs().max().item() <= 2e-3, (
            f"final16 vs sep rms_norm mismatch T={T}"
        )
        assert (final32_ref - final_no2_f32).abs().max().item() <= 1e-5, (
            f"final32 norm_out2 mismatch T={T}"
        )

        final16_bf = torch.empty_like(final16_ref)
        final_no2_bf = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ext.attn_res_dual(
            blk,
            4,
            x0,
            x1,
            None,
            None,
            query,
            nw,
            final16_bf.unsqueeze(1),
            final_no2_bf,
            EPS,
            EPS,
        )
        assert torch.equal(final16_ar, final16_bf), f"final16 (bf16 tap) mismatch T={T}"
        assert torch.equal(
            final_no2_bf.view(torch.uint16), _ref_f32_to_bf16_bits(final_no2_f32)
        ), f"final_bf16 norm_out2 RNE mismatch T={T}"

        sep3 = functools.partial(
            _bench_sep3_final,
            ext,
            blk,
            x0,
            x1,
            query,
            hid_tmp,
            final32_ref,
            final16_ref,
            nw,
        )
        fus1 = functools.partial(
            ext.attn_res_dual,
            blk,
            4,
            x0,
            x1,
            None,
            None,
            query,
            nw,
            final16_bf.unsqueeze(1),
            final_no2_bf,
            EPS,
            EPS,
        )

        us_3 = time_in_graph(sep3, iters=iters)
        us_1 = time_in_graph(fus1, iters=iters)
        out[f"T{T}"] = {"sep3_us": round(us_3, 3), "fused1_bf16_us": round(us_1, 3)}
        print(
            f"  final_readout T={T:2d}: 3-launch fp32={us_3:6.2f} us -> "
            f"1-launch bf16={us_1:6.2f} us (save {us_3 - us_1:+.2f} us)"
        )
    return out


# ---------------------------------------------------------------------------
# Section 3: gate_mix_split T <= 16 cliff fix + occupancy / atomic variants
# ---------------------------------------------------------------------------


def run_gate_mix(ext, iters: int) -> dict[str, Any]:
    print("\n=== [3] msg_gate_mix T=1..16 (split-R vs 1-wave cliff + exp variants) ===")
    gen = torch.Generator(device="cuda").manual_seed(99)
    C = (HIDDEN + 127) // 128
    C64 = (HIDDEN + 63) // 64
    out: dict[str, Any] = {}

    for G in (3, 1):
        wf = (
            torch.randn(
                G, GATE_RANK, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            * 0.1
        )
        wg = (
            torch.randn(
                G, GATE_RANK, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            * 0.1
        )
        nw = torch.ones(G, HIDDEN, dtype=torch.float32, device="cuda")

        for T in (1, 2, 4, 8, 10, 12, 16):
            h = torch.randn(
                T, G, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            m = torch.randn(
                T, G, HIDDEN, dtype=torch.float32, device="cuda", generator=gen
            )
            d = torch.randn(
                G, T, GATE_RANK, dtype=torch.float16, device="cuda", generator=gen
            )
            out_prod = torch.empty(G, T, HIDDEN, dtype=torch.float16, device="cuda")
            out_1w = torch.empty_like(out_prod)
            # Oversized scratch/counters so experimental variants 2/3/4 can run too
            scratch = torch.zeros(
                3 * T * HIDDEN + 4 * 3 * T * 2 * HIDDEN,
                dtype=torch.float32,
                device="cuda",
            )
            part = torch.zeros(3 * T * C64, dtype=torch.float32, device="cuda")
            counters = torch.zeros(
                3 * T + 3 * T * C64, dtype=torch.int32, device="cuda"
            )

            ext.msg_gate_mix(
                h,
                m,
                d,
                wf,
                wg,
                nw,
                out_prod,
                scratch[: 3 * T * HIDDEN],
                part[: 3 * T * C],
                counters[: 3 * T],
                EPS,
            )
            assert int(counters[: 3 * T].abs().max().item()) == 0, (
                f"counters not restored to 0 at T={T}, G={G}"
            )
            # Exact PyTorch fp32 reference for msg_gate_mix + RMSNorm -> fp16:
            d32 = d.float()  # [G, T, R]
            f_ref = torch.einsum("gtr,grh->gth", d32, wf)
            g_ref = torch.einsum("gtr,grh->gth", d32, wg)
            h_gt = h.permute(1, 0, 2)  # [G, T, H]
            m_gt = m.permute(1, 0, 2)  # [G, T, H]
            mix_ref = (1.0 - torch.tanh(f_ref)) * h_gt + torch.tanh(g_ref) * m_gt
            rs_ref = torch.rsqrt(mix_ref.pow(2).mean(dim=-1, keepdim=True) + EPS)
            out_ref = (mix_ref * rs_ref * nw[:, None, :]).to(torch.float16)
            max_err = (out_prod.float() - out_ref.float()).abs().max().item()
            assert max_err <= 2e-3, (
                f"gate_mix split vs torch ref err {max_err} at T={T}, G={G}"
            )

            us_split = time_in_graph(
                functools.partial(
                    ext.msg_gate_mix,
                    h,
                    m,
                    d,
                    wf,
                    wg,
                    nw,
                    out_prod,
                    scratch[: 3 * T * HIDDEN],
                    part[: 3 * T * C],
                    counters[: 3 * T],
                    EPS,
                ),
                iters=iters,
            )
            us_1w = time_in_graph(
                functools.partial(
                    ext.exp_gate_mix_launch,
                    0,
                    h,
                    m,
                    d,
                    wf,
                    wg,
                    nw,
                    out_1w,
                    scratch,
                    part,
                    counters,
                    EPS,
                ),
                iters=iters,
            )
            us_acqrel = time_in_graph(
                functools.partial(
                    ext.exp_gate_mix_launch,
                    1,
                    h,
                    m,
                    d,
                    wf,
                    wg,
                    nw,
                    out_1w,
                    scratch,
                    part,
                    counters,
                    EPS,
                ),
                iters=iters,
            )
            row: dict[str, Any] = {
                "prod_split_us": round(us_split, 3),
                "old_1wave_us": round(us_1w, 3),
                "acqrel_us": round(us_acqrel, 3),
            }
            extra_str = ""
            if T in (1, 2):
                out_v = torch.empty_like(out_prod)
                for vid, vname in (
                    (2, "col64_us"),
                    (3, "rsplit2_us"),
                    (4, "rsplit4_us"),
                ):
                    ext.exp_gate_mix_launch(
                        vid, h, m, d, wf, wg, nw, out_v, scratch, part, counters, EPS
                    )
                    err_v = (out_prod.float() - out_v.float()).abs().max().item()
                    assert err_v <= 2e-3, f"{vname} parity err {err_v}"
                    assert int(counters.abs().max().item()) == 0, (
                        f"{vname} counter leak"
                    )
                    row[vname] = round(
                        time_in_graph(
                            functools.partial(
                                ext.exp_gate_mix_launch,
                                vid,
                                h,
                                m,
                                d,
                                wf,
                                wg,
                                nw,
                                out_v,
                                scratch,
                                part,
                                counters,
                                EPS,
                            ),
                            iters=iters,
                        ),
                        3,
                    )
                extra_str = (
                    f"  col64={row['col64_us']:5.2f}  "
                    f"rs2={row['rsplit2_us']:5.2f}  rs4={row['rsplit4_us']:5.2f}"
                )

            key = f"G{G}/T{T}"
            out[key] = row
            # 24 QKV (G=3) or 24 MLP (G=1) calls across passes 1 & 2 (when fp32 weights)
            print(
                f"  {key:8s}  split={us_split:6.2f} us  1wave={us_1w:6.2f} us "
                f"(save {us_1w - us_split:+6.2f} us)  "
                f"acqrel={us_acqrel:6.2f} us{extra_str}"
            )
    return out


# ---------------------------------------------------------------------------
# Section 4 & 5: MoE MXFP8 bm sweep + NVFP4 (TX, BK) variant sweep
# ---------------------------------------------------------------------------


def _routing_modes(tokens: int, mode: str, gen: torch.Generator) -> torch.Tensor:
    """``uniform`` vs ``correlated`` (~44% shared experts between verify rows)."""
    if mode == "uniform" or tokens == 1:
        logits = torch.randn(tokens, EXPERTS, device="cuda", generator=gen)
        return logits.topk(TOP_K, dim=-1).indices.to(torch.int32).contiguous()
    # Correlated verify chain: adjacent rows share ~44% of top-8 experts
    base = torch.randn(1, EXPERTS, device="cuda", generator=gen)
    rows = [base[0]]
    for _ in range(1, tokens):
        noise = torch.randn(EXPERTS, device="cuda", generator=gen)
        rows.append(0.72 * rows[-1] + 0.69 * noise)
    logits = torch.stack(rows, dim=0)
    return logits.topk(TOP_K, dim=-1).indices.to(torch.int32).contiguous()


def _rand_moe_weights(fmt: str, k: int, n: int, gen: torch.Generator):
    def rnd_i32(shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.int32, device="cuda", generator=gen
        )

    if fmt == "nvfp4":
        q = rnd_i32((EXPERTS, k // 8, n), -(2**31), 2**31 - 1).to(torch.int32)
        scales = rnd_i32((EXPERTS, k // 16, n), 0x34, 0x42).to(torch.uint8)
        glob = torch.full((EXPERTS,), 2e-2, dtype=torch.float32, device="cuda")
        return q, scales, glob
    raw = rnd_i32((EXPERTS, k // 4, n, 4), 0, 0x70) | (
        rnd_i32((EXPERTS, k // 4, n, 4), 0, 2) << 7
    )
    q = (
        (raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16) | (raw[..., 3] << 24))
        .to(torch.int64)
        .to(torch.int32)
    )
    scales = rnd_i32((EXPERTS, k // 32, n), 114, 119).to(torch.uint8)
    return q, scales, None


def _zero_pair(gu: torch.Tensor, c: torch.Tensor) -> None:
    gu.zero_()
    c.zero_()


def _bench_mxfp8_step(
    moe_ext, x, gu, c, w13_q, w13_s, w2_q, w2_s, no_tw, tw, s_ids, e_ids, ntpp, bm
) -> None:
    gu.zero_()
    c.zero_()
    moe_ext.moe_mxfp8_gemm_rdna2(
        x, gu, w13_q, w13_s, no_tw, s_ids, e_ids, ntpp, TOP_K, bm, False, 0
    )
    moe_ext.moe_mxfp8_gemm_swiglu_rdna2(
        gu,
        c,
        w2_q,
        w2_s,
        tw,
        s_ids,
        e_ids,
        ntpp,
        1,
        bm,
        True,
        TOP_K,
        SOFTCAP_GATE,
        SOFTCAP_UP,
    )


def _bench_nvfp4_step(
    moe_ext,
    x,
    gu,
    c,
    w13_q,
    w13_s,
    w13_g,
    w2_q,
    w2_s,
    w2_g,
    no_tw,
    tw,
    s_ids,
    e_ids,
    ntpp,
    bm,
) -> None:
    gu.zero_()
    c.zero_()
    moe_ext.moe_nvfp4_gemm_rdna2(
        x, gu, w13_q, w13_s, w13_g, no_tw, s_ids, e_ids, ntpp, TOP_K, bm, False, 0
    )
    moe_ext.moe_nvfp4_gemm_swiglu_rdna2(
        gu,
        c,
        w2_q,
        w2_s,
        w2_g,
        tw,
        s_ids,
        e_ids,
        ntpp,
        1,
        bm,
        True,
        TOP_K,
        SOFTCAP_GATE,
        SOFTCAP_UP,
    )


def run_moe_mxfp8(moe_ext, iters: int) -> dict[str, Any]:
    print("\n=== [4] MoE MXFP8 (w13 + w2_swiglu) bm=1 vs bm=2 vs bm=4 ===")
    gen = torch.Generator(device="cuda").manual_seed(7)
    w13_q, w13_s, _ = _rand_moe_weights("mxfp8", HIDDEN, 2 * INTER, gen)
    w2_q, w2_s, _ = _rand_moe_weights("mxfp8", INTER, HIDDEN, gen)
    no_tw = torch.empty(0, dtype=torch.float32, device="cuda")
    out: dict[str, Any] = {}

    for mode in ("correlated", "uniform"):
        for T in (1, 2, 3, 4, 8, 16):
            ids = _routing_modes(T, mode, gen)
            x = torch.randn(
                T, HIDDEN, dtype=torch.float16, device="cuda", generator=gen
            )
            tw = torch.softmax(
                torch.randn(
                    T, TOP_K, dtype=torch.float32, device="cuda", generator=gen
                ),
                dim=-1,
            ).reshape(-1)
            ref_c = None
            row_res: dict[str, Any] = {}
            for bm in (1, 2, 4):
                sorted_ids, expert_ids, ntpp = align_expert_blocks(ids, bm, EXPERTS)
                active_blks = int(ntpp.item()) // bm
                gu = torch.zeros(
                    T * TOP_K, 2 * INTER, dtype=torch.float16, device="cuda"
                )
                c = torch.zeros(T, HIDDEN, dtype=torch.float16, device="cuda")

                step = functools.partial(
                    _bench_mxfp8_step,
                    moe_ext,
                    x,
                    gu,
                    c,
                    w13_q,
                    w13_s,
                    w2_q,
                    w2_s,
                    no_tw,
                    tw,
                    sorted_ids,
                    expert_ids,
                    ntpp,
                    bm,
                )
                zero_only = functools.partial(_zero_pair, gu, c)

                step()
                if ref_c is None:
                    ref_c = c.clone()
                else:
                    err = (c.float() - ref_c.float()).abs().max().item()
                    assert err <= 5e-2, f"MXFP8 bm={bm} vs bm=1 err={err} at T={T}"

                us = time_in_graph(step, iters=iters) - time_in_graph(
                    zero_only, iters=iters
                )
                row_res[f"bm{bm}"] = {"us": round(us, 2), "active_blocks": active_blks}
            key = f"{mode}/T{T}"
            out[key] = row_res
            print(
                f"  {key:15s}  "
                + "  ".join(
                    f"bm={bm}: {row_res[f'bm{bm}']['us']:6.1f} us "
                    f"({row_res[f'bm{bm}']['active_blocks']:3d} blk)"
                    for bm in (1, 2, 4)
                )
            )
    return out


def run_moe_nvfp4(moe_ext, iters: int) -> dict[str, Any]:
    print("\n=== [5] MoE NVFP4 (w13 + w2_swiglu) (TX, BK) variants at T=1, 3, 8 ===")
    remarks = _parse_resource_remarks(CSRC / "moe_lowp_dot2_rdna2.cu")
    nvfp4_remarks = {k: v for k, v in remarks.items() if "moe_gemm_nvfp4" in k}
    if nvfp4_remarks:
        print("  Resource usage (-Rpass-analysis=kernel-resource-usage):")
        for k, v in sorted(nvfp4_remarks.items()):
            print(f"    {k[:84]}: {v}")

    gen = torch.Generator(device="cuda").manual_seed(11)
    w13_q, w13_s, w13_g = _rand_moe_weights("nvfp4", HIDDEN, 2 * INTER, gen)
    w2_q, w2_s, w2_g = _rand_moe_weights("nvfp4", INTER, HIDDEN, gen)
    no_tw = torch.empty(0, dtype=torch.float32, device="cuda")

    # Bytes per active expert block across w13 (K=2304, N=2304) + w2 (K=1152, N=2304):
    # NVFP4 weight = 0.5 B/elem + E4M3 scale = 1/16 B/elem = 9/16 B/elem
    bytes_per_expert = int((HIDDEN * 2 * INTER + INTER * HIDDEN) * (9.0 / 16.0))
    out: dict[str, Any] = {"resource_remarks": nvfp4_remarks, "bench": {}}

    for T, bm in ((1, 1), (3, 2), (8, 2), (8, 4)):
        ids = _routing_modes(T, "correlated", gen)
        sorted_ids, expert_ids, ntpp = align_expert_blocks(ids, bm, EXPERTS)
        active_blks = int(ntpp.item()) // bm
        x = (
            torch.randn(T, HIDDEN, dtype=torch.float16, device="cuda", generator=gen)
            * 0.2
        )
        tw = torch.softmax(
            torch.randn(T, TOP_K, dtype=torch.float32, device="cuda", generator=gen),
            dim=-1,
        ).reshape(-1)
        gu = torch.zeros(T * TOP_K, 2 * INTER, dtype=torch.float16, device="cuda")
        c = torch.zeros(T, HIDDEN, dtype=torch.float16, device="cuda")

        zero_only = functools.partial(_zero_pair, gu, c)

        us_zero = time_in_graph(zero_only, iters=iters)
        ref_c = None
        row: dict[str, Any] = {"active_blocks": active_blks}

        for var, vlabel in (
            (0, "TX256_BK256"),
            (1, "TX128_BK256"),
        ):
            moe_ext.set_nvfp4_variant(var)

            step = functools.partial(
                _bench_nvfp4_step,
                moe_ext,
                x,
                gu,
                c,
                w13_q,
                w13_s,
                w13_g,
                w2_q,
                w2_s,
                w2_g,
                no_tw,
                tw,
                sorted_ids,
                expert_ids,
                ntpp,
                bm,
            )

            step()
            torch.accelerator.synchronize()
            if ref_c is None:
                ref_c = c.clone()
                max_abs, rel_l2 = 0.0, 0.0
            else:
                diff = (c.float() - ref_c.float()).abs()
                max_abs = float(diff.max().item())
                rel_l2 = float(
                    diff.norm().item() / (ref_c.float().norm().item() + 1e-12)
                )
                assert rel_l2 < 5e-3, (
                    f"NVFP4 {vlabel} rel_l2={rel_l2} at T={T}, bm={bm}"
                )

            us = time_in_graph(step, iters=iters) - us_zero
            gbps = (active_blks * bytes_per_expert) / (us * 1e3)
            row[vlabel] = {
                "us": round(us, 2),
                "gbps": round(gbps, 1),
                "max_abs": round(max_abs, 6),
                "rel_l2": round(rel_l2, 6),
            }
        moe_ext.set_nvfp4_variant(-1)
        key = f"T{T}_bm{bm}"
        out["bench"][key] = row
        print(
            f"  {key:10s} ({active_blks:2d} blk)  "
            + "  ".join(
                f"{vl}: {row[vl]['us']:6.1f} us "
                f"({row[vl]['gbps']:5.1f} GB/s, rel={row[vl]['rel_l2']:.1e})"
                for vl in ("TX256_BK256", "TX128_BK256")
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--section",
        choices=(
            "all",
            "glue",
            "moe",
            "sconv_attn_res",
            "taps_final",
            "gate_mix",
            "moe_mxfp8",
            "moe_nvfp4",
        ),
        default="all",
    )
    ap.add_argument("--archs", default="gfx1030")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--verbose-jit", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    need_glue = args.section in (
        "all",
        "glue",
        "sconv_attn_res",
        "taps_final",
        "gate_mix",
    )
    need_moe = args.section in ("all", "moe", "moe_mxfp8", "moe_nvfp4")

    summary: dict[str, Any] = {}
    if need_glue:
        print("Building Glue+Attn JIT extension...")
        glue_ext = build_glue_attn_ext(args.archs, verbose=args.verbose_jit)
        glue_remarks = _parse_resource_remarks(
            CSRC / "inkling2_glue_rdna2.cu", args.archs
        )
        sconv_res = {
            k: v
            for k, v in glue_remarks.items()
            if "attn_res" in k or "gate_mix_split" in k
        }
        if sconv_res:
            summary["glue_resource_remarks"] = sconv_res
            print("  Glue resource usage (-Rpass-analysis=kernel-resource-usage):")
            for k, v in sorted(sconv_res.items()):
                print(f"    {k[:84]}: {v}")
        if args.section in ("all", "glue", "sconv_attn_res"):
            summary["sconv_attn_res"] = run_sconv_attn_res(glue_ext, args.iters)
        if args.section in ("all", "glue", "taps_final"):
            summary["taps_final"] = run_taps_final(glue_ext, args.iters)
        if args.section in ("all", "glue", "gate_mix"):
            summary["gate_mix"] = run_gate_mix(glue_ext, args.iters)

    if need_moe:
        print("\nBuilding MoE JIT extension...")
        moe_ext = build_moe_ext(args.archs, verbose=args.verbose_jit)
        if args.section in ("all", "moe", "moe_mxfp8"):
            summary["moe_mxfp8"] = run_moe_mxfp8(moe_ext, args.iters)
        if args.section in ("all", "moe", "moe_nvfp4"):
            summary["moe_nvfp4"] = run_moe_nvfp4(moe_ext, args.iters)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote JSON results to {args.json}")
    print("\nALL REQUESTED SECTIONS PASSED.")


if __name__ == "__main__":
    sys.exit(main())
