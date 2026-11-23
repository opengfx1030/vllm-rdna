// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Inkling-2-Mini decode node fusions of the attention side of a layer visit on
// RDNA2 (gfx1030, wave32): fewer graph nodes for bit-identical results
// (VLLM_INKLING2_FUSE_ATTN_CHAIN, docs/rdna2/inkling2-fuse-attn-chain.md).
//
//   inkling2_fuse_attn_res_conv
//       A static short convolution (inkling2_sconv_static with write_state) of
//       an fp16 [T, H] stream fused into the attn_res kernel that consumes its
//       fp32 output: the conv result is computed in the load prologue of the
//       attn_res CTA, never stored unless asked (x1_out), and the conv state is
//       written there.  Two shapes cover the layer visit:
//         * post-MLP conv of visit i + the attention-input attn_res of visit
//           i + 1:        partial state = x0 + conv(moe_out)
//         * post-attention conv + the MLP-input attn_res of the same visit:
//           partial state = hid + conv(o_out) (x0 given) or conv(o_out) alone
//           at a block boundary (no x0, ``append`` = hid, x1_out = attn_c)
//       The arithmetic of both kernels is copied verbatim (the same fp32
//       expressions in the same order), so the results are bit-identical to
//       the two launches.  Only valid when every request has one token
//       (conv_writes_in_op): the state row a token writes can be a row the
//       first tokens of another chunk position still read.
//
//   inkling2_fuse_msgs_fill
//       The message attn_res (4 learned queries, raw fp32 result) of every
//       layer of a loop pass in ONE launch (grid = (T, layers)): the messages
//       of a pass depend only on the states of the previous pass, which are
//       fixed while the pass runs.
//
// Not here: the decode attention kernel and its combine (owned by the fused
// decode kernel), the MoE side and the message gates (glue kernels).
//
// Device code is gfx1030-only like the kernels it copies; the kernels of the
// other architectures are empty bodies with the same signature, so the
// translation unit compiles for every arch of a multi-arch build.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <torch/all.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace inkling2_fuse {

constexpr int kWave = 32;
constexpr int kMaxStates = 64;  // block states (+ append + partial) per call

// ---------------------------------------------------------------------------
// Small helpers: verbatim copies of inkling2_glue_rdna2.cu
// ---------------------------------------------------------------------------
// Lane `lane ^ O` of the wave, for a compile-time O: DPP / permlanex16 moves (a
// few cycles) instead of the LDS crossbar of __shfl_xor (tens of cycles each,
// in series).
template <int O>
__device__ __forceinline__ float xor_lane(float v) {
  const int x = __float_as_int(v);
  int r;
  if constexpr (O == 16)  // the other row of 16 lanes
    r = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false,
                                     false);
  else if constexpr (O == 8)  // row_xmask:8
    r = __builtin_amdgcn_update_dpp(x, x, 0x168, 0xF, 0xF, false);
  else if constexpr (O == 4)  // row_xmask:4
    r = __builtin_amdgcn_update_dpp(x, x, 0x164, 0xF, 0xF, false);
  else if constexpr (O == 2)  // quad_perm:[2,3,0,1]
    r = __builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false);
  else  // quad_perm:[1,0,3,2]
    r = __builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false);
  return __int_as_float(r);
}

// v += v[lane ^ 16], then ^ 8, 4, 2, 1: the butterfly of the shipped kernels
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

// N independent wave_sum: the moves of one step are issued together
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

// ---------------------------------------------------------------------------
// Paged conv state: verbatim copies of inkling2_attn_rdna2.cu (fp16 input only)
// ---------------------------------------------------------------------------
struct ConvMeta {
  const int* block_table;  // [R, bt_stride]
  const int* seq_idx;      // [T_md]
  const int* qsl;          // [R + 1]
  const int* seq_lens;     // [R]
  _Float16* cache;
  int64_t cs0, cs1, cs2;  // block / head / ring-row strides, in halfs
  int bt_stride;
  int blk_shift;  // log2 of the tokens per cache block
  int ring;       // ring rows per block
  int T_md;       // tokens covered by seq_idx
};

struct TokPos {
  int qs;  // first token row of the request
  int p;   // position of this token
  int p1;  // context length after this step
};

struct FuseArgs {
  float* blocks;  // [T, S, H] fp32; the append slot is written here
  int64_t bs_t, bs_s;
  const float* x0;  // partial state = x0 (+ x1) | x0 + conv | conv
  int64_t x0_t;
  const float* x1;  // XSRC == 0 only
  int64_t x1_t;
  float* x1_out;  // XSRC != 0: receives the conv result (optional)
  int64_t x1o_t;
  float* partial_out;  // receives the partial state
  int64_t po_t;
  const float* append;  // state `n_blocks` (also copied into blocks)
  int64_t ap_t;
  const float* query;  // [Q, H] fp32 (+ blockIdx.y * q_layer)
  int64_t q_s, q_layer;
  const void* norm_w;  // [1 or Q, H] fp16 / fp32 (null: fp32 output)
  int64_t nw_s;
  int nw_f16;
  int nw_rows;
  void* out;  // [T, Q, H] fp32 (or fp16 with norm_w) (+ blockIdx.y * o_layer)
  int64_t o_t, o_q, o_layer;
  ConvMeta m;          // conv state (XSRC != 0)
  const _Float16* cx;  // conv input, fp16 [T, H]
  int64_t cx_t;
  const float* cw;  // conv weight [H, 4]
  int width;        // channels per kv head
  int o0;           // column of the stream in the state rows
  float inv;        // 1 / sqrt(5)
  int H, n_blocks;
  float eps, norm_eps, inv_h, scale;
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ TokPos tok_pos(const ConvMeta& m, int t, int r) {
  const int qs = m.qsl[r];
  const int p1 = m.seq_lens[r];
  return {qs, p1 - (m.qsl[r + 1] - qs) + (t - qs), p1};
}

// Raw fp16 x4: the loads of the conv inputs and of the norm weights stay
// unconverted until their value is needed: a conversion right behind a load
// makes the wave wait for every load in flight (the whole prologue would run in
// series).
__device__ __forceinline__ uint2 load8(const void* p) {
  return *reinterpret_cast<const uint2*>(p);
}

__device__ __forceinline__ float4 cvt4h(const uint2 u) {
  return make_float4(h2f_lo(u.x), h2f_hi(u.x), h2f_lo(u.y), h2f_hi(u.y));
}

// Where the three history taps (lag 1..3) of a token come from.  Uniform:
// computed from the conv metadata once per token; the three block-table loads
// do not depend on each other and are issued together (a lookup per tap and per
// thread group would cost one full memory latency each, in series).
struct HistPlan {
  bool use_x[3];  // the tap is a row of the same chunk (the conv input), else a
                  // state row
  bool valid[3];  // false: before the start of the sequence (zero)
  int64_t base[3];  // element offset of the row (state rows: without head and
                    // channel)
  int blk_cur;      // state page of the position of the token (its ring row is
                    // written)
};

__device__ __forceinline__ HistPlan hist_plan(const ConvMeta& m,
                                              const TokPos& tp,
                                              int64_t x_stride, int t, int r,
                                              int o0) {
  // 32-bit indices keep the address arithmetic on the scalar unit
  const int row = r * m.bt_stride;
  int blk[3];
  HistPlan hp;
  hp.blk_cur = m.block_table[row + (max(tp.p, 0) >> m.blk_shift)];
  #pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    const int sp = tp.p - lag;
    blk[lag - 1] = m.block_table[row + (max(sp, 0) >> m.blk_shift)];
  }
  #pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    const int sr = t - lag;
    const int sp = tp.p - lag;
    const bool ux = sr >= tp.qs;
    hp.use_x[lag - 1] = ux;
    hp.valid[lag - 1] = ux || sp >= 0;
    hp.base[lag - 1] = ux ? (int64_t)sr * x_stride
                          : (int64_t)blk[lag - 1] * m.cs0 +
                                (int64_t)(sp % m.ring) * m.cs2 + o0;
  }
  return hp;
}

// h[lag - 1] = raw fp16 input `lag` tokens back for channels [c, c + 4) of the
// token (head hd, channel jch of the head)
__device__ __forceinline__ void hist_load(uint2 (&h)[3], const HistPlan& hp,
                                          const ConvMeta& m,
                                          const _Float16* __restrict__ x, int c,
                                          int hd, int jch) {
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    uint2 v = make_uint2(0u, 0u);
    if (hp.valid[i])
      v = load8(hp.use_x[i] ? x + hp.base[i] + c
                            : m.cache + hp.base[i] + (int64_t)hd * m.cs1 + jch);
    h[i] = v;
  }
}

// Whether position p of a request that ends this step at context length p1 is
// one of the last `ring` positions of its block inside the chunk.
__device__ __forceinline__ bool ring_keeps(const ConvMeta& m, int p, int p1) {
  if (p < 0 || p >= p1) return false;
  const int blk_end = ((p >> m.blk_shift) + 1) << m.blk_shift;
  return p + m.ring >= min(p1, blk_end);
}

__device__ __forceinline__ void write_ring(const ConvMeta& m, int blk, int p,
                                           int h, int jch, int o0, float4 cur) {
  // Block 0 is the null block, not a page: dummy batches write nothing.
  if (blk <= 0) return;
  _Float16* dst = m.cache + (int64_t)blk * m.cs0 + (int64_t)h * m.cs1 +
                  (int64_t)(p % m.ring) * m.cs2 + o0 + jch;
  const auto pk = [](float lo, float hi) {
    const uint16_t a = __builtin_bit_cast(uint16_t, (_Float16)lo);
    const uint16_t b = __builtin_bit_cast(uint16_t, (_Float16)hi);
    return (uint32_t)a | ((uint32_t)b << 16);
  };
  *reinterpret_cast<uint2*>(dst) =
      make_uint2(pk(cur.x, cur.y), pk(cur.z, cur.w));
}

// static taps: w is [C, 4] (conv1d.weight [C, 1, 4]); w[c, 3 - lag] weights
// the input `lag` tokens back.  `w[i]` is the row of channel i of the group.
__device__ __forceinline__ float4 conv4_static(const float4 (&tap)[4],
                                               const float4 (&w)[4],
                                               float inv) {
  const float4 w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
  float4 o;
  o.x = tap[0].x * w0.w + tap[1].x * w0.z + tap[2].x * w0.y + tap[3].x * w0.x;
  o.y = tap[0].y * w1.w + tap[1].y * w1.z + tap[2].y * w1.y + tap[3].y * w1.x;
  o.z = tap[0].z * w2.w + tap[1].z * w2.z + tap[2].z * w2.y + tap[3].z * w2.x;
  o.w = tap[0].w * w3.w + tap[1].w * w3.z + tap[2].w * w3.y + tap[3].w * w3.x;
  o.x = (o.x + tap[0].x) * inv;
  o.y = (o.y + tap[0].y) * inv;
  o.z = (o.z + tap[0].z) * inv;
  o.w = (o.w + tap[0].w) * inv;
  return o;
}

// ---------------------------------------------------------------------------
// attn_res with the partial state produced in the load prologue
//   XSRC 0: x0 (+ x1)           (the shipped kernel; one layer per grid row)
//   XSRC 1: x0 + conv(cx)
//   XSRC 2: conv(cx)
//
// The same fp32 arithmetic in the same order as the shipped attn_res / conv
// kernels, scheduled for latency (a visit is one CTA, so its time is the length
// of its dependent chain, not its work):
//   * every load that needs nothing from another load (queries, stored block
//   states,
//     x0 / x1 / appended state, the conv input of the token and the conv
//     weights, the norm weights) is issued first and consumed late;
//   * the conv metadata chain (seq_idx -> qsl / seq_lens -> block table ->
//   state rows)
//     runs on scalar loads next to them, and the partial products of the stored
//     states are computed while the state rows are in flight;
//   * the softmax over the states runs in every wave from the per-wave partial
//   sums
//     (same additions in the same order), so only the exchange of those sums
//     and the final norm need a block barrier.
// ---------------------------------------------------------------------------
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

// state j of the stack: blocks[j] (j < n_blocks), then the appended state, then
// the partial state
template <int ITER>
__device__ __forceinline__ void fetch_state(int j, float4 (&s)[ITER],
                                            const float4 (&A)[ITER],
                                            const float4 (&P)[ITER],
                                            const float* brow, int64_t bs_s,
                                            int n_blocks, bool has_app, int V) {
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = threadIdx.x + k * blockDim.x;
    if (j < n_blocks)
      s[k] = v < V ? load4(brow + (int64_t)j * bs_s + 4 * v)
                   : make_float4(0.f, 0.f, 0.f, 0.f);
    else if (has_app && j == n_blocks)
      s[k] = A[k];
    else
      s[k] = P[k];
  }
}

// the value of `x` in lane `lane` of the wave (a uniform lane)
__device__ __forceinline__ float bcast_lane(float x, int lane) {
  return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), lane));
}

template <int Q, int ITER, bool NORM, int XSRC>
__device__ __forceinline__ void attn_res_fused_body(const FuseArgs& a) {
  constexpr int kNReg = 5;  // states kept in registers between the two passes
  constexpr int kQ1 = Q + 1;
  constexpr bool kPrefW = NORM && Q == 1;  // norm weights held in registers
  constexpr int kConv = XSRC != 0 ? ITER : 1;
  const int tid = threadIdx.x;
  const int nth = blockDim.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int nwaves = nth / kWave;
  const int64_t t = blockIdx.x;
  const int V = a.H >> 2;

  __shared__ float s_red[8][kMaxStates * kQ1];

  const bool has_x1 = a.x1 != nullptr;
  const bool has_app = a.append != nullptr;
  const int NS = a.n_blocks + (has_app ? 1 : 0) + 1;
  float* brow = a.blocks != nullptr ? a.blocks + t * a.bs_t : nullptr;
  const float* query = a.query + (int64_t)blockIdx.y * a.q_layer;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  const uint2 z2 = make_uint2(0u, 0u);

  // ---- 1. every load that needs nothing from the conv metadata
  float4 Qv[Q][ITER], S[kNReg][ITER], P[ITER], A[ITER], X1[ITER];
  float4 Wc[kConv][4];
  uint2 Xh[ITER], Wa[ITER], Wb[ITER];
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
  #pragma unroll
    for (int q = 0; q < Q; ++q)
      Qv[q][k] = v < V ? load4(query + (int64_t)q * a.q_s + 4 * v) : z4;
  }
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
  #pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int v = tid + k * nth;
      S[j][k] = (j < a.n_blocks && v < V)
                    ? load4(brow + (int64_t)j * a.bs_s + 4 * v)
                    : z4;
    }
  }
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
    P[k] = z4;
    A[k] = z4;
    X1[k] = z4;
    Xh[k] = z2;
    Wa[k] = z2;
    Wb[k] = z2;
    if (v < V) {
      if constexpr (XSRC != 2) P[k] = load4(a.x0 + t * a.x0_t + 4 * v);
      if constexpr (XSRC == 0) {
        if (has_x1) X1[k] = load4(a.x1 + t * a.x1_t + 4 * v);
      } else {
        Xh[k] = load8(a.cx + t * a.cx_t + 4 * v);  // tap 0 of the conv
      }
      if (has_app) A[k] = load4(a.append + t * a.ap_t + 4 * v);
      if constexpr (kPrefW) {
        // fp16 weights: the 8 bytes of this thread's 4 values (the second load
        // re-reads in-bounds bytes and is not used); fp32 weights: its 16 bytes
        // as two halves. No branch around a load here: its result would be
        // merged into the registers of the other path with a copy, and that
        // copy waits for every load in flight.
        const char* wbase = reinterpret_cast<const char*>(a.norm_w);
        const int64_t esz = a.nw_f16 ? 2 : 4;
        const int64_t w_off = (int64_t)4 * v * esz;
        Wa[k] = load8(wbase + w_off);
        Wb[k] = load8(wbase + min(w_off + 8, (int64_t)a.H * esz - 8));
      }
      if constexpr (XSRC != 0) {
        const float* wc = a.cw + (int64_t)(4 * v) * 4;
        Wc[k][0] = *reinterpret_cast<const float4*>(wc);
        Wc[k][1] = *reinterpret_cast<const float4*>(wc + 4);
        Wc[k][2] = *reinterpret_cast<const float4*>(wc + 8);
        Wc[k][3] = *reinterpret_cast<const float4*>(wc + 12);
      }
    }
  }

  // ---- 2. conv metadata chain (uniform) and the state rows it addresses
  int cr = -1;
  TokPos tp = {0, 0, 0};
  bool keep = false;
  uint2 Hh[kConv][3];
  HistPlan hp = {};
  if constexpr (XSRC != 0) {
    if (t < a.m.T_md) {
      cr = a.m.seq_idx[t];
      if (cr >= 0) {
        tp = tok_pos(a.m, (int)t, cr);
        keep = ring_keeps(a.m, tp.p, tp.p1);
        hp = hist_plan(a.m, tp, a.cx_t, (int)t, cr, a.o0);
  #pragma unroll
        for (int k = 0; k < ITER; ++k) {
          const int v = tid + k * nth;
          if (v < V) {
            const int c = 4 * v;
            const int h = c / a.width;
            const int jch = c - h * a.width;
            hist_load(Hh[k], hp, a.m, a.cx, c, h, jch);
          }
        }
      }
    }
  }

  // ---- 3. pass 1 over the stored states: sum of squares and the dot product
  // with every query (needs no partial state); the unused slots are zero
  {
    float vs[kNReg * kQ1];
  #pragma unroll
    for (int j = 0; j < kNReg; ++j) {
      float v[kQ1];
      state_partials<Q, ITER>(S[j], Qv, v);
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) vs[j * kQ1 + i] = v[i];
    }
    wave_sum_n<kNReg * kQ1>(vs);
    if (lane == 0) {
  #pragma unroll
      for (int j = 0; j < kNReg; ++j)
        if (j < a.n_blocks)
  #pragma unroll
          for (int i = 0; i < kQ1; ++i)
            s_red[wave][j * kQ1 + i] = vs[j * kQ1 + i];
    }
  }

  // ---- 4. partial state (last of the stack) and appended block state
  #pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int v = tid + k * nth;
    if (v < V) {
      float4 p = P[k];
      if constexpr (XSRC == 0) {
        if (has_x1) {
          const float4 q1 = X1[k];
          p.x += q1.x;
          p.y += q1.y;
          p.z += q1.z;
          p.w += q1.w;
        }
      } else {
        float4 cv = z4;
        if (cr >= 0) {
          const int c = 4 * v;
          const int h = c / a.width;
          const int jch = c - h * a.width;
          const float4 tap[4] = {cvt4h(Xh[k]), cvt4h(Hh[k][0]), cvt4h(Hh[k][1]),
                                 cvt4h(Hh[k][2])};
          cv = conv4_static(tap, Wc[k], a.inv);
          if (keep) write_ring(a.m, hp.blk_cur, tp.p, h, jch, a.o0, tap[0]);
        }
        if (a.x1_out != nullptr) store4(a.x1_out + t * a.x1o_t + 4 * v, cv);
        if constexpr (XSRC == 1) {
          p.x += cv.x;
          p.y += cv.y;
          p.z += cv.z;
          p.w += cv.w;
        } else {
          p = cv;
        }
      }
      if (a.partial_out != nullptr)
        store4(a.partial_out + t * a.po_t + 4 * v, p);
      if (has_app) store4(brow + (int64_t)a.n_blocks * a.bs_s + 4 * v, A[k]);
      P[k] = p;
    }
  }

  // ---- 5. pass 1 over the appended and the partial state (one reduction for
  // both)
  {
    float vb[2 * kQ1];
    float va[kQ1], vq[kQ1];
    state_partials<Q, ITER>(A, Qv, va);  // zero without an appended state
    state_partials<Q, ITER>(P, Qv, vq);
  #pragma unroll
    for (int i = 0; i < kQ1; ++i) {
      vb[i] = va[i];
      vb[kQ1 + i] = vq[i];
    }
    wave_sum_n<2 * kQ1>(vb);
    if (lane == 0) {
      if (has_app && a.n_blocks < kNReg)
  #pragma unroll
        for (int i = 0; i < kQ1; ++i) s_red[wave][a.n_blocks * kQ1 + i] = vb[i];
      if (NS - 1 < kNReg)
  #pragma unroll
        for (int i = 0; i < kQ1; ++i)
          s_red[wave][(NS - 1) * kQ1 + i] = vb[kQ1 + i];
    }
  }
  // states of the stack beyond the registers (more than kNReg: tests only)
  #pragma unroll
  for (int j = 0; j < kNReg; ++j)
    if (j >= a.n_blocks && j < NS)
      fetch_state<ITER>(j, S[j], A, P, brow, a.bs_s, a.n_blocks, has_app, V);
  float vp[kQ1];
  for (int j = kNReg; j < NS; ++j) {
    float4 s[ITER];
    fetch_state<ITER>(j, s, A, P, brow, a.bs_s, a.n_blocks, has_app, V);
    state_partials<Q, ITER>(s, Qv, vp);
  #pragma unroll
    for (int i = 0; i < kQ1; ++i) vp[i] = wave_sum(vp[i]);
    if (lane == 0) {
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) s_red[wave][j * kQ1 + i] = vp[i];
    }
  }
  __syncthreads();

  // ---- 6. scores -> softmax over the states -> coefficients c[q][j] = p[q][j]
  // * r[j]; every wave reduces the per-wave sums of the states it owns (lane l:
  // states l and l + 32) in the order of the waves and keeps the coefficients
  // in registers
  float coef[Q][2];
  {
    float rr[2];
    float tot[2][kQ1];
  #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = lane + kWave * u;
  #pragma unroll
      for (int i = 0; i < kQ1; ++i) {
        float acc = 0.f;
        if (j < NS)
          for (int w = 0; w < nwaves; ++w) acc += s_red[w][j * kQ1 + i];
        tot[u][i] = acc;
      }
      rr[u] = j < NS ? rsqrtf(tot[u][0] * a.inv_h + a.eps) : 0.f;
    }
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      float sc[2], e[2];
  #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = lane + kWave * u;
        sc[u] = j < NS ? (tot[u][1 + q] * rr[u]) * a.scale : -INFINITY;
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
  }

  // ---- 7. pass 2: out[q] = sum_j c[q][j] * state_j
  float4 acc[Q][ITER];
  #pragma unroll
  for (int q = 0; q < Q; ++q)
  #pragma unroll
    for (int k = 0; k < ITER; ++k) acc[q][k] = z4;
  #pragma unroll
  for (int j = 0; j < kNReg; ++j) {
    if (j < NS) {
  #pragma unroll
      for (int q = 0; q < Q; ++q) {
        const float c = bcast_lane(coef[q][j >> 5], j & (kWave - 1));
  #pragma unroll
        for (int k = 0; k < ITER; ++k) acc[q][k] = fma4(c, S[j][k], acc[q][k]);
      }
    }
  }
  for (int j = kNReg; j < NS; ++j) {
    float4 s[ITER];
    fetch_state<ITER>(j, s, A, P, brow, a.bs_s, a.n_blocks, has_app, V);
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      const float c =
          bcast_lane(j < kWave ? coef[q][0] : coef[q][1], j & (kWave - 1));
  #pragma unroll
      for (int k = 0; k < ITER; ++k) acc[q][k] = fma4(c, s[k], acc[q][k]);
    }
  }

  // ---- 8. epilogue
  if constexpr (!NORM) {
    float* out = reinterpret_cast<float*>(a.out) + t * a.o_t +
                 (int64_t)blockIdx.y * a.o_layer;
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
    __half* out = reinterpret_cast<__half*>(a.out) + t * a.o_t +
                  (int64_t)blockIdx.y * a.o_layer;
  #pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t wrow = (a.nw_rows == 1 ? 0 : q) * a.nw_s;
  #pragma unroll
      for (int k = 0; k < ITER; ++k) {
        const int v = tid + k * nth;
        if (v < V) {
          float4 w;
          if constexpr (kPrefW)
            w = a.nw_f16
                    ? cvt4h(Wa[k])
                    : make_float4(
                          __uint_as_float(Wa[k].x), __uint_as_float(Wa[k].y),
                          __uint_as_float(Wb[k].x), __uint_as_float(Wb[k].y));
          else
            w = load4_dyn(a.norm_w, wrow + 4 * v, a.nw_f16);
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

template <int Q, int ITER, bool NORM, int XSRC>
__global__ void __launch_bounds__(256) attn_res_fused_kernel(const FuseArgs a) {
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  attn_res_fused_body<Q, ITER, NORM, XSRC>(a);
#endif
}

}  // namespace inkling2_fuse

// ===========================================================================
// Host side
// ===========================================================================
namespace {

using namespace inkling2_fuse;

void check_gpu(const at::Tensor& t, const char* who, const char* name) {
  TORCH_CHECK(t.defined() && t.is_cuda(), who, ": ", name,
              " must be a GPU tensor");
}

void check_dtype(const at::Tensor& t, const char* who, const char* name,
                 at::ScalarType dt) {
  TORCH_CHECK(t.scalar_type() == dt, who, ": ", name, " must be ",
              c10::toString(dt), ", got ", c10::toString(t.scalar_type()));
}

void check_i32(const at::Tensor& t, const char* who, const char* name) {
  check_gpu(t, who, name);
  check_dtype(t, who, name, at::kInt);
  TORCH_CHECK(t.is_contiguous(), who, ": ", name, " must be contiguous");
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

int log2_exact(int64_t v, const char* who) {
  TORCH_CHECK(v >= 1 && (v & (v - 1)) == 0, who,
              ": the cache block size must be a power of two, got ", v);
  int s = 0;
  while ((int64_t(1) << s) < v) ++s;
  return s;
}

template <int Q, bool NORM, int XSRC>
void launch_fused(int iter, int block, int64_t T, int64_t layers,
                  const FuseArgs& a, hipStream_t stream) {
#define I2F_LAUNCH(IT)                                                        \
  case IT:                                                                    \
    inkling2_fuse::attn_res_fused_kernel<Q, IT, NORM, XSRC>                   \
        <<<dim3((unsigned)T, (unsigned)layers), dim3(block), 0, stream>>>(a); \
    break
  switch (iter) {
    I2F_LAUNCH(1);
    I2F_LAUNCH(2);
    I2F_LAUNCH(3);
    I2F_LAUNCH(4);
    default:
      TORCH_CHECK(false, "inkling2_fuse: bad iter ", iter);
  }
#undef I2F_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// fields shared by the two entry points: block states, partial_out, append,
// query, norm weights, out
void fill_common(FuseArgs& a, const char* who, const at::Tensor& blocks,
                 int64_t n_blocks, int64_t T, int64_t H, int64_t Q,
                 const std::optional<at::Tensor>& partial_out,
                 const std::optional<at::Tensor>& append,
                 const std::optional<at::Tensor>& norm_w, const at::Tensor& out,
                 bool& norm) {
  norm = norm_w.has_value() && norm_w->defined();
  a.out = out.data_ptr();
  a.o_t = out.stride(0);
  a.H = (int)H;
  a.n_blocks = (int)n_blocks;
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
}

}  // namespace

// ---------------------------------------------------------------------------
// inkling2_fuse_attn_res_conv
// ---------------------------------------------------------------------------
void inkling2_fuse_attn_res_conv(
    const at::Tensor& blocks, int64_t n_blocks,
    const std::optional<at::Tensor>& x0, const at::Tensor& conv_x,
    const at::Tensor& conv_w, const at::Tensor& conv_cache,
    const at::Tensor& block_table, const at::Tensor& seq_idx,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    int64_t stream_offset, int64_t num_kv_heads, int64_t block_size,
    const std::optional<at::Tensor>& x1_out,
    const std::optional<at::Tensor>& partial_out,
    const std::optional<at::Tensor>& append, const at::Tensor& query,
    const std::optional<at::Tensor>& norm_w, const at::Tensor& out, double eps,
    double norm_eps) {
  const char* who = "inkling2_fuse_attn_res_conv";
  check_gpu(conv_x, who, "conv_x");
  check_gpu(query, who, "query");
  check_gpu(out, who, "out");
  check_dtype(conv_x, who, "conv_x", at::kHalf);
  TORCH_CHECK(conv_x.dim() == 2, who, ": conv_x must be [T, H]");
  const int64_t T = conv_x.size(0), H = conv_x.size(1);
  TORCH_CHECK(conv_x.stride(1) == 1 && conv_x.stride(0) % 8 == 0 &&
                  (reinterpret_cast<uintptr_t>(conv_x.data_ptr()) & 15) == 0,
              who, ": conv_x rows must be 16-byte aligned");
  check_dtype(query, who, "query", at::kFloat);
  TORCH_CHECK(query.dim() == 2 && query.size(1) == H, who,
              ": query must be [Q, H]");
  check_vec_rows(query, who, "query", 4);
  const int64_t Q = query.size(0);
  const bool has_x0 = x0.has_value() && x0->defined();
  TORCH_CHECK(Q == 1 || (Q == 3 && has_x0), who,
              ": Q must be 1 (or 3 with x0), got ", Q);
  TORCH_CHECK(out.dim() == 3 && out.size(0) == T && out.size(1) == Q &&
                  out.size(2) == H,
              who, ": out must be [T, Q, H]");
  const bool norm = norm_w.has_value() && norm_w->defined();
  check_dtype(out, who, "out", norm ? at::kHalf : at::kFloat);
  check_vec_rows(out, who, "out", 4);
  TORCH_CHECK(n_blocks >= 0, who, ": n_blocks must be >= 0");

  FuseArgs a{};
  bool norm_f = false;
  fill_common(a, who, blocks, n_blocks, T, H, Q, partial_out, append, norm_w,
              out, norm_f);
  a.o_q = out.stride(1);
  a.o_layer = 0;
  a.query = query.data_ptr<float>();
  a.q_s = query.stride(0);
  a.q_layer = 0;
  a.eps = (float)eps;
  a.norm_eps = (float)norm_eps;
  if (has_x0) {
    check_gpu(*x0, who, "x0");
    check_dtype(*x0, who, "x0", at::kFloat);
    TORCH_CHECK(x0->dim() == 2 && x0->size(0) == T && x0->size(1) == H, who,
                ": x0 must be [T, H]");
    check_vec_rows(*x0, who, "x0", 4);
    a.x0 = x0->data_ptr<float>();
    a.x0_t = x0->stride(0);
  }
  if (x1_out.has_value() && x1_out->defined()) {
    check_gpu(*x1_out, who, "x1_out");
    check_dtype(*x1_out, who, "x1_out", at::kFloat);
    TORCH_CHECK(
        x1_out->dim() == 2 && x1_out->size(0) == T && x1_out->size(1) == H, who,
        ": x1_out must be [T, H]");
    check_vec_rows(*x1_out, who, "x1_out", 4);
    a.x1_out = x1_out->data_ptr<float>();
    a.x1o_t = x1_out->stride(0);
  }

  // conv state (the checks of the shipped i2_conv_meta / i2_sconv_launch)
  check_gpu(conv_cache, who, "conv_cache");
  TORCH_CHECK(conv_cache.scalar_type() == at::kHalf && conv_cache.dim() == 4 &&
                  conv_cache.stride(3) == 1,
              who, ": conv_cache must be fp16 [blocks, H, ring_rows, W]");
  TORCH_CHECK(conv_cache.size(2) >= 3, who,
              ": the conv state keeps at least k - 1 = 3 ring rows per block");
  check_i32(block_table, who, "block_table");
  check_i32(seq_idx, who, "seq_idx");
  check_i32(query_start_loc, who, "query_start_loc");
  check_i32(seq_lens, who, "seq_lens");
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
  a.m.block_table = block_table.data_ptr<int32_t>();
  a.m.seq_idx = seq_idx.data_ptr<int32_t>();
  a.m.qsl = query_start_loc.data_ptr<int32_t>();
  a.m.seq_lens = seq_lens.data_ptr<int32_t>();
  a.m.cache = reinterpret_cast<_Float16*>(conv_cache.data_ptr());
  a.m.cs0 = conv_cache.stride(0);
  a.m.cs1 = conv_cache.stride(1);
  a.m.cs2 = conv_cache.stride(2);
  a.m.bt_stride = (int)block_table.stride(0);
  a.m.blk_shift = log2_exact(block_size, who);
  a.m.ring = (int)conv_cache.size(2);
  a.m.T_md = (int)std::min<int64_t>(seq_idx.size(0), T);
  a.cx = reinterpret_cast<const _Float16*>(conv_x.data_ptr());
  a.cx_t = conv_x.stride(0);
  a.cw = conv_w.data_ptr<float>();
  a.width = (int)width;
  a.o0 = (int)stream_offset;
  a.inv = (float)(1.0 / std::sqrt(5.0));

  const at::cuda::OptionalCUDAGuard guard(device_of(conv_x));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  int iter, block;
  row_launch_shape(H, who, iter, block);
#define I2F_CASE(QQ, NN, XS) \
  launch_fused<QQ, NN, XS>(iter, block, T, 1, a, stream)
  if (has_x0) {
    if (Q == 1) {
      if (norm)
        I2F_CASE(1, true, 1);
      else
        I2F_CASE(1, false, 1);
    } else {
      if (norm)
        I2F_CASE(3, true, 1);
      else
        I2F_CASE(3, false, 1);
    }
  } else {
    if (norm)
      I2F_CASE(1, true, 2);
    else
      I2F_CASE(1, false, 2);
  }
#undef I2F_CASE
}

// ---------------------------------------------------------------------------
// inkling2_fuse_msgs_fill
// ---------------------------------------------------------------------------
void inkling2_fuse_msgs_fill(const at::Tensor& blocks, int64_t n_blocks,
                             const at::Tensor& x0, const at::Tensor& x1,
                             const at::Tensor& queries, const at::Tensor& out,
                             double eps) {
  const char* who = "inkling2_fuse_msgs_fill";
  check_gpu(x0, who, "x0");
  check_gpu(x1, who, "x1");
  check_gpu(queries, who, "queries");
  check_gpu(out, who, "out");
  check_dtype(x0, who, "x0", at::kFloat);
  check_dtype(x1, who, "x1", at::kFloat);
  TORCH_CHECK(x0.dim() == 2, who, ": x0 must be [T, H]");
  const int64_t T = x0.size(0), H = x0.size(1);
  check_vec_rows(x0, who, "x0", 4);
  TORCH_CHECK(x1.dim() == 2 && x1.size(0) == T && x1.size(1) == H, who,
              ": x1 must be [T, H]");
  check_vec_rows(x1, who, "x1", 4);
  check_dtype(queries, who, "queries", at::kFloat);
  TORCH_CHECK(
      queries.dim() == 3 && queries.size(1) == 4 && queries.size(2) == H, who,
      ": queries must be [L, 4, H]");
  check_vec_rows(queries, who, "queries", 4);
  const int64_t L = queries.size(0);
  check_dtype(out, who, "out", at::kFloat);
  TORCH_CHECK(out.dim() == 4 && out.size(0) >= T && out.size(1) == L &&
                  out.size(2) == 4 && out.size(3) == H,
              who, ": out must be [>= T, L, 4, H]");
  check_vec_rows(out, who, "out", 4);
  TORCH_CHECK(n_blocks >= 0, who, ": n_blocks must be >= 0");

  FuseArgs a{};
  bool norm_f = false;
  fill_common(a, who, blocks, n_blocks, T, H, 4, std::nullopt, std::nullopt,
              std::nullopt, out, norm_f);
  a.o_q = out.stride(2);
  a.o_layer = out.stride(1);
  a.query = queries.data_ptr<float>();
  a.q_s = queries.stride(1);
  a.q_layer = queries.stride(0);
  a.eps = (float)eps;
  a.norm_eps = (float)eps;
  a.x0 = x0.data_ptr<float>();
  a.x0_t = x0.stride(0);
  a.x1 = x1.data_ptr<float>();
  a.x1_t = x1.stride(0);

  const at::cuda::OptionalCUDAGuard guard(device_of(x0));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (T == 0 || L == 0) return;
  int iter, block;
  row_launch_shape(H, who, iter, block);
  launch_fused<4, false, 0>(iter, block, T, L, a, stream);
}
