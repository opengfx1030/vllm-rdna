// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Inkling-2-Mini attention core for RDNA2 (gfx1030, wave32, V_DOT2_F32_F16).
//
// Three op families (all fp16 activations, fp32 accumulation, native HIP):
//
//  1. inkling2_sconv_static / inkling2_sconv_dynamic
//       depthwise causal short convolution (k = 4) over the paged conv state.
//       The previous k-1 raw inputs of a token come from the activations when
//       they are in the same query chunk and from the conv state otherwise
//       (positions < 0 read as zero).  With write_state (at most one token per
//       request) the same kernel then writes this step's raw input back;
//       larger chunks write it with inkling2_conv_state_write once the
//       history has been read.  Output is fp32.
//         static : out = (sum_lag x[t-lag] * w[c, k-1-lag] + x[t]) / sqrt(k+1)
//         dynamic: out = (sum_lag x[t-lag] * (sw[lag, c] + dw[t, h, lag])
//                         + x[t]) / sqrt(2k+1)
//
//  2. inkling2_kv_prep
//       k and v short convs fused with the per-(token, kv head) k L2 clip /
//       v unweighted RMS norm and the scatter into the paged K|V cache.
//
//  3. inkling2_attention
//       attention over the paged K|V cache with the hidden-conditioned learned
//       relative bias, the per-head sink logit, the sliding window or the
//       log-scaled full attention, softmax in fp32, and the sigmoid output
//       gate fused into the epilogue.  The prologue fuses the scaled L2 clip
//       (+ log scale) of q and of the relative features.
//         * q_len == 1 requests : split-K decode kernel, one wave per
//           (request, kv head, split), 8 lanes per key, K/V streamed straight
//           from the paged cache (no LDS).
//         * q_len >= 2 requests : tiled kernel, 16 query tokens x GQA group
//           per CTA, K/V tiles of 64 keys staged in LDS, QK^T and PV through
//           V_DOT2 with fp32 accumulation, optional split-K over keys.
//         * a combine kernel merges the split partials, adds the sink and
//           applies the gate.
//
// KV cache page: [num_blocks, H_kv, block_size, 2 * 128] fp16, K at [0, 128),
// V at [128, 256).  Conv state page: [num_blocks, H_kv, S, W] fp16, a ring of S
// rows per cache block (row = position % S, see I2ConvMeta); a stream of C
// channels occupies the columns [offset, offset + C / H_kv) of every row.
//
// All kernels are graph-capture safe: no host reads of device data, no
// allocation (the split-K partials live in a caller-provided workspace), grid
// sizes depend only on host integers.

#include <algorithm>
#include <cstdint>
#include <optional>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE) && \
    (__AMDGCN_WAVEFRONT_SIZE != 32)
  #error "inkling2_attn_rdna2.cu assumes wave32"
#endif

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// Tile constants at global scope: HIP does not reliably resolve namespace
// scoped constexpr values inside templated __global__ functions.
constexpr int kI2D = 128;  // head dim
constexpr int kI2BR = 16;  // query tokens per tile CTA
constexpr int kI2BC = 64;  // keys per tile
constexpr int kI2TileThreads = 128;
constexpr int kI2SubTile = 16;     // keys per decode super-tile
constexpr int kI2MaxSplits = 256;  // split-K partials per (request, kv head)

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float i2_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2, a),
                                __builtin_bit_cast(half2, b), c, false);
}

__device__ __forceinline__ float i2_lo(uint32_t w) {
  return (float)__builtin_bit_cast(_Float16, (uint16_t)(w & 0xFFFFu));
}

__device__ __forceinline__ float i2_hi(uint32_t w) {
  return (float)__builtin_bit_cast(_Float16, (uint16_t)(w >> 16));
}

__device__ __forceinline__ float i2_round_half(float x) {
  return (float)(_Float16)x;
}

__device__ __forceinline__ uint32_t i2_pack(float lo, float hi) {
  const uint16_t a = __builtin_bit_cast(uint16_t, (_Float16)lo);
  const uint16_t b = __builtin_bit_cast(uint16_t, (_Float16)hi);
  return (uint32_t)a | ((uint32_t)b << 16);
}

__device__ __forceinline__ float i2_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float i2_xor(float v, int m) {
  return __shfl_xor(v, m);
}

// ---------------------------------------------------------------------------
// short convolution
// ---------------------------------------------------------------------------
// Conv state.  The raw inputs of the previous k - 1 = 3 tokens of a request are
// kept position addressed: position p lives in the block block_table[r, p >>
// blk_shift] at ring row p % ring of a [H, ring, W] fp16 page.  A step writes
// the last `ring` positions of every (block, chunk) intersection
// (i2_ring_keeps), so every ring row has one writer, the rows a following step
// reads (history of the next chunk, tail of a completed block a prefix hit
// resumes from) are always among them, and ring >= k - 1 + the speculative
// draft length keeps the rows of the accepted prefix intact after a rejection.
struct I2ConvMeta {
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

template <typename XT>
struct I2Load4;

template <>
struct I2Load4<float> {
  static __device__ __forceinline__ float4 get(const float* p) {
    return *reinterpret_cast<const float4*>(p);
  }
};

template <>
struct I2Load4<_Float16> {
  static __device__ __forceinline__ float4 get(const _Float16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    return make_float4(i2_lo(u.x), i2_hi(u.x), i2_lo(u.y), i2_hi(u.y));
  }
};

struct I2TokPos {
  int qs;  // first token row of the request
  int p;   // position of this token
  int p1;  // context length after this step
};

__device__ __forceinline__ I2TokPos i2_tok_pos(const I2ConvMeta& m, int t,
                                               int r) {
  const int qs = m.qsl[r];
  const int p1 = m.seq_lens[r];
  return {qs, p1 - (m.qsl[r + 1] - qs) + (t - qs), p1};
}

// tap[lag] = raw input `lag` tokens back for channels [c, c + 4) of token t.
template <typename XT>
__device__ __forceinline__ void i2_gather_taps(float4 (&tap)[4],
                                               const I2ConvMeta& m,
                                               const I2TokPos& tp,
                                               const XT* __restrict__ x,
                                               int64_t x_stride, int t, int r,
                                               int c, int h, int jch, int o0) {
  tap[0] = I2Load4<XT>::get(x + (int64_t)t * x_stride + c);
  #pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    const int sr = t - lag;
    if (sr >= tp.qs) {
      v = I2Load4<XT>::get(x + (int64_t)sr * x_stride + c);
    } else {
      const int sp = tp.p - lag;
      if (sp >= 0) {
        const int blk =
            m.block_table[(int64_t)r * m.bt_stride + (sp >> m.blk_shift)];
        v = I2Load4<_Float16>::get(m.cache + (int64_t)blk * m.cs0 +
                                   (int64_t)h * m.cs1 +
                                   (int64_t)(sp % m.ring) * m.cs2 + o0 + jch);
      }
    }
    tap[lag] = v;
  }
}

// Whether position p of a request that ends this step at context length p1 is
// one of the last `ring` positions of its block inside the chunk.  Only the
// positions 0 <= p < p1 exist.  A token that is not one writes nothing: the
// padding of a graph that the engine's token -> request map gives to request 0
// past the last chunk (p >= p1), and the rows of a capture's dummy batch whose
// seq_len is shorter than its query (p < 0: no block, no ring row).
__device__ __forceinline__ bool i2_ring_keeps(const I2ConvMeta& m, int p,
                                              int p1) {
  if (p < 0 || p >= p1) return false;
  const int blk_end = ((p >> m.blk_shift) + 1) << m.blk_shift;
  return p + m.ring >= min(p1, blk_end);
}

__device__ __forceinline__ void i2_write_ring(const I2ConvMeta& m, int r, int p,
                                              int h, int jch, int o0,
                                              float4 cur) {
  const int blk = m.block_table[(int64_t)r * m.bt_stride + (p >> m.blk_shift)];
  // Block 0 is the null block (BlockPool.null_block), not a page: the dummy
  // batches of a graph capture and of the memory profile have zeroed block
  // tables and must write nothing, as their -1 slots do for the KV cache.
  if (blk <= 0) return;
  _Float16* dst = m.cache + (int64_t)blk * m.cs0 + (int64_t)h * m.cs1 +
                  (int64_t)(p % m.ring) * m.cs2 + o0 + jch;
  *reinterpret_cast<uint2*>(dst) =
      make_uint2(i2_pack(cur.x, cur.y), i2_pack(cur.z, cur.w));
}

// static taps: w is [C, 4] (conv1d.weight [C, 1, 4]); w[c, 3 - lag] weights
// the input `lag` tokens back.  `wc` points at w[c, 0].
__device__ __forceinline__ float4 i2_conv4_static(const float4 (&tap)[4],
                                                  const float* __restrict__ wc,
                                                  float inv) {
  const float4 w0 = *reinterpret_cast<const float4*>(wc);
  const float4 w1 = *reinterpret_cast<const float4*>(wc + 4);
  const float4 w2 = *reinterpret_cast<const float4*>(wc + 8);
  const float4 w3 = *reinterpret_cast<const float4*>(wc + 12);
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

// dynamic taps: sw is [4, C] (static_weight), dw[lag] the per-token per-head
// tap added to every channel of the head.  `swc` points at sw[0, c].
__device__ __forceinline__ float4
i2_conv4_dynamic(const float4 (&tap)[4], const float* __restrict__ swc,
                 int64_t C, uint2 dw_raw, float inv) {
  const float d0 = i2_lo(dw_raw.x), d1 = i2_hi(dw_raw.x);
  const float d2 = i2_lo(dw_raw.y), d3 = i2_hi(dw_raw.y);
  const float4 s0 = *reinterpret_cast<const float4*>(swc);
  const float4 s1 = *reinterpret_cast<const float4*>(swc + C);
  const float4 s2 = *reinterpret_cast<const float4*>(swc + 2 * C);
  const float4 s3 = *reinterpret_cast<const float4*>(swc + 3 * C);
  float4 o;
  o.x = tap[0].x * (s0.x + d0) + tap[1].x * (s1.x + d1) +
        tap[2].x * (s2.x + d2) + tap[3].x * (s3.x + d3);
  o.y = tap[0].y * (s0.y + d0) + tap[1].y * (s1.y + d1) +
        tap[2].y * (s2.y + d2) + tap[3].y * (s3.y + d3);
  o.z = tap[0].z * (s0.z + d0) + tap[1].z * (s1.z + d1) +
        tap[2].z * (s2.z + d2) + tap[3].z * (s3.z + d3);
  o.w = tap[0].w * (s0.w + d0) + tap[1].w * (s1.w + d1) +
        tap[2].w * (s2.w + d2) + tap[3].w * (s3.w + d3);
  o.x = (o.x + tap[0].x) * inv;
  o.y = (o.y + tap[0].y) * inv;
  o.z = (o.z + tap[0].z) * inv;
  o.w = (o.w + tap[0].w) * inv;
  return o;
}

// grid (T, ceil(C / 512)), block 128: one thread = 4 channels of one token.
// WR writes the raw input of the token into the conv state after reading the
// history; it is only valid when no request has more than one token (the row a
// token writes can be a row the first tokens of another chunk position read).
template <typename XT, bool DYN, bool WR>
__global__ __launch_bounds__(128) void i2_sconv_kernel(
    I2ConvMeta m, const XT* __restrict__ x, int64_t x_stride,
    const float* __restrict__ w, const _Float16* __restrict__ dyn,
    int64_t dyn_stride, float* __restrict__ out, int64_t out_stride, int C,
    int width, int o0, float inv) {
  const int t = blockIdx.x;
  const int c = (blockIdx.y * 128 + threadIdx.x) * 4;
  if (c >= C) return;
  float4 res = make_float4(0.f, 0.f, 0.f, 0.f);
  if (t < m.T_md) {
    const int r = m.seq_idx[t];
    if (r >= 0) {
      const int h = c / width;
      const int jch = c - h * width;
      const I2TokPos tp = i2_tok_pos(m, t, r);
      float4 tap[4];
      i2_gather_taps<XT>(tap, m, tp, x, x_stride, t, r, c, h, jch, o0);
      if constexpr (DYN) {
        const uint2 dw = *reinterpret_cast<const uint2*>(
            dyn + (int64_t)t * dyn_stride + h * 4);
        res = i2_conv4_dynamic(tap, w + c, C, dw, inv);
      } else {
        res = i2_conv4_static(tap, w + (int64_t)c * 4, inv);
      }
      if constexpr (WR) {
        if (i2_ring_keeps(m, tp.p, tp.p1))
          i2_write_ring(m, r, tp.p, h, jch, o0, tap[0]);
      }
    }
  }
  *reinterpret_cast<float4*>(out + (int64_t)t * out_stride + c) = res;
}

// grid (T, ceil(C / 512)), block 128: write the raw input of every token that
// is among the last `ring` positions of its block inside the chunk.
template <typename XT>
__global__ __launch_bounds__(128) void i2_state_write_kernel(
    I2ConvMeta m, const XT* __restrict__ x, int64_t x_stride, int C, int width,
    int o0) {
  const int t = blockIdx.x;
  const int c = (blockIdx.y * 128 + threadIdx.x) * 4;
  if (c >= C || t >= m.T_md) return;
  const int r = m.seq_idx[t];
  if (r < 0) return;
  const I2TokPos tp = i2_tok_pos(m, t, r);
  if (!i2_ring_keeps(m, tp.p, tp.p1)) return;
  const int h = c / width;
  i2_write_ring(m, r, tp.p, h, c - h * width, o0,
                I2Load4<XT>::get(x + (int64_t)t * x_stride + c));
}

// grid (T, ceil(H_kv / 4)), block 128: one wave = one (token, kv head); the
// 32 lanes own 4 channels each of the 128-wide head, K stream then V stream.
template <bool DYN, bool WR>
__global__ __launch_bounds__(128) void i2_kv_prep_kernel(
    I2ConvMeta m, const _Float16* __restrict__ xk, int64_t xk_stride,
    const _Float16* __restrict__ xv, int64_t xv_stride,
    const float* __restrict__ kw, const float* __restrict__ vw,
    const _Float16* __restrict__ kdyn, const _Float16* __restrict__ vdyn,
    int64_t dyn_stride, _Float16* __restrict__ kv, int64_t kv_s0, int64_t kv_s1,
    int64_t kv_s2, int kv_shift, const int64_t* __restrict__ kv_slots, int C,
    int H, int ko0, int vo0, float inv, float k_clip, float rms_eps) {
  const int t = blockIdx.x;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.y * 4 + wave;
  if (h >= H || t >= m.T_md) return;
  const int r = m.seq_idx[t];
  if (r < 0) return;
  const int jch = lane * 4;
  const int c = h * kI2D + jch;
  const I2TokPos tp = i2_tok_pos(m, t, r);
  const bool keep = WR && i2_ring_keeps(m, tp.p, tp.p1);

  float4 tap[4];
  float4 ko, vo;
  i2_gather_taps<_Float16>(tap, m, tp, xk, xk_stride, t, r, c, h, jch, ko0);
  if constexpr (DYN) {
    const uint2 dw =
        *reinterpret_cast<const uint2*>(kdyn + (int64_t)t * dyn_stride + h * 4);
    ko = i2_conv4_dynamic(tap, kw + c, C, dw, inv);
  } else {
    ko = i2_conv4_static(tap, kw + (int64_t)c * 4, inv);
  }
  if (keep) i2_write_ring(m, r, tp.p, h, jch, ko0, tap[0]);

  i2_gather_taps<_Float16>(tap, m, tp, xv, xv_stride, t, r, c, h, jch, vo0);
  if constexpr (DYN) {
    const uint2 dw =
        *reinterpret_cast<const uint2*>(vdyn + (int64_t)t * dyn_stride + h * 4);
    vo = i2_conv4_dynamic(tap, vw + c, C, dw, inv);
  } else {
    vo = i2_conv4_static(tap, vw + (int64_t)c * 4, inv);
  }
  if (keep) i2_write_ring(m, r, tp.p, h, jch, vo0, tap[0]);

  float kss = ko.x * ko.x + ko.y * ko.y + ko.z * ko.z + ko.w * ko.w;
  float vss = vo.x * vo.x + vo.y * vo.y + vo.z * vo.z + vo.w * vo.w;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    kss += i2_xor(kss, off);
    vss += i2_xor(vss, off);
  }
  const float kf = fminf(1.0f, k_clip / (sqrtf(kss) + 1e-6f));
  const float vf = rsqrtf(vss * (1.0f / kI2D) + rms_eps);

  const int64_t slot = kv_slots[t];
  if (slot >= 0) {
    const int mask = (1 << kv_shift) - 1;
    _Float16* dst = kv + (slot >> kv_shift) * kv_s0 + (int64_t)h * kv_s1 +
                    (int64_t)(slot & mask) * kv_s2;
    *reinterpret_cast<uint2*>(dst + jch) = make_uint2(
        i2_pack(ko.x * kf, ko.y * kf), i2_pack(ko.z * kf, ko.w * kf));
    *reinterpret_cast<uint2*>(dst + kI2D + jch) = make_uint2(
        i2_pack(vo.x * vf, vo.y * vf), i2_pack(vo.z * vf, vo.w * vf));
  }
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------
struct I2AttnArgs {
  const _Float16* q;
  int64_t q_stride;  // [T, nh * 128]
  const _Float16* rel;
  int64_t rel_stride;  // [T, nh * 16]
  const _Float16* sink;
  int64_t sink_stride;  // [T, nh]
  const _Float16* gate;
  int64_t gate_stride;     // [T, nh * 128] or nullptr
  const float* q_scale;    // [nh, 128]
  const float* rel_scale;  // [nh, 16]
  const _Float16* proj_t;  // [rel_extent, 16]
  const void* positions;
  int pos64;
  _Float16* out;
  int64_t out_stride;  // [T, nh * 128]
  const _Float16* kv;
  int64_t kv_s0, kv_s1, kv_s2;
  const int* block_table;
  int bt_stride;
  const int* seq_lens;
  const int* qsl;
  float* part_o;   // [S, T, nh, 128]
  float* part_ml;  // [S, T, nh, 2]
  int T, R, nh, nkv, bs_shift, window, rel_extent, log_scale;
  int splits_a, splits_b, tile_row0;
  float log_alpha, log_nfloor, q_clip, rel_clip, scale;
};

__device__ __forceinline__ float i2_log_tau(const I2AttnArgs& a, int t) {
  const int64_t pos =
      a.pos64 ? reinterpret_cast<const int64_t*>(a.positions)[t]
              : (int64_t)reinterpret_cast<const int*>(a.positions)[t];
  return 1.0f +
         a.log_alpha * logf(fmaxf((float)(pos + 1) / a.log_nfloor, 1.0f));
}

// Key range of a q_len == 1 request split into 16-key super-tiles.
struct I2DecodeSplit {
  int lo, lo16, nst, tps, n_active;
};

__device__ __forceinline__ I2DecodeSplit i2_decode_split(int L, int window,
                                                         int S) {
  I2DecodeSplit d;
  d.lo = (window > 0) ? max(0, L - window) : 0;
  d.lo16 = d.lo & ~(kI2SubTile - 1);
  d.nst = (L - d.lo16 + kI2SubTile - 1) / kI2SubTile;
  d.tps = (d.nst + S - 1) / S;
  d.n_active = (d.nst + d.tps - 1) / d.tps;
  return d;
}

// Key range of a tile of kI2BR query tokens split into 64-key tiles.
struct I2TileSplit {
  int p_first, ntok, lo, t_lo, t_hi, tps, n_active;
};

__device__ __forceinline__ I2TileSplit i2_tile_split(int L, int q_len, int i0,
                                                     int window, int S) {
  I2TileSplit s;
  s.p_first = L - q_len + i0;
  s.ntok = min(kI2BR, q_len - i0);
  const int p_last = s.p_first + s.ntok - 1;
  s.lo = (window > 0) ? max(0, s.p_first - window + 1) : 0;
  s.t_lo = s.lo / kI2BC;
  s.t_hi = (p_last + kI2BC) / kI2BC;
  const int nt = s.t_hi - s.t_lo;
  s.tps = (nt + S - 1) / S;
  s.n_active = (nt + s.tps - 1) / s.tps;
  return s;
}

// --------------------------- decode (q_len == 1) ---------------------------
// Grid (H_kv, ceil(S / 4), R), block 128.  One wave = one (request, kv head,
// split).  The kv head is the fastest grid dimension: the CTAs of all heads of
// one split group are dispatched back to back, so they stream the same
// token-major pages ([B, bs, H, C] physical layout, 4 KB per token) in step.
// With the heads in the slow dimension a grid larger than the resident CTAs
// ran 1.9x slower (128k-token full layer, 1024 waves: 2.3 ms vs 1.2 ms;
// sibling heads no longer share DRAM rows, [INFERENCE]).
// Lane = (octet o = lane / 8, c = lane % 8): the 4 octets stream 4
// keys at a time; octet lane c owns head dims [8c, 8c + 8) and [64 + 8c,
// 64 + 8c + 8) of K and V, so one b128 load per lane covers a contiguous
// 128-byte run of a key row.  Every octet runs its own online softmax over its
// keys; the octets are merged at the end of the wave's key range.
template <int G>
__global__ __launch_bounds__(128) void i2_decode_kernel(I2AttnArgs a) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int split = blockIdx.y * 4 + wave;
  const int kvh = blockIdx.x;
  const int r = blockIdx.z;
  const int qs = a.qsl[r];
  if (a.qsl[r + 1] - qs != 1) return;
  const int t = qs;
  const int L = a.seq_lens[r];
  const int c = lane & 7;
  const int oc = lane >> 3;
  const int hq0 = kvh * G;

  if (L <= 0) {  // padded request: define the output
    if (split == 0 && oc == 0) {
  #pragma unroll
      for (int g = 0; g < G; ++g) {
        _Float16* o = a.out + (int64_t)t * a.out_stride + (hq0 + g) * kI2D;
        *reinterpret_cast<uint4*>(o + c * 8) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(o + 64 + c * 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    return;
  }
  if (split >= a.splits_a) return;
  const I2DecodeSplit ds = i2_decode_split(L, a.window, a.splits_a);
  const int st0 = split * ds.tps;
  if (st0 >= ds.nst) return;
  const int st1 = min(ds.nst, st0 + ds.tps);

  // ---- q / relative-feature prologue (scaled L2 clip, optional log scale)
  const float tau = a.log_scale ? i2_log_tau(a, t) : 1.0f;
  uint32_t qreg[G][8];
  uint32_t relw[G];
  #pragma unroll
  for (int g = 0; g < G; ++g) {
    const int hq = hq0 + g;
    const _Float16* qrow = a.q + (int64_t)t * a.q_stride + hq * kI2D;
    const uint4 u0 = *reinterpret_cast<const uint4*>(qrow + c * 8);
    const uint4 u1 = *reinterpret_cast<const uint4*>(qrow + 64 + c * 8);
    const float* sc = a.q_scale + hq * kI2D;
    const float4 s0 = *reinterpret_cast<const float4*>(sc + c * 8);
    const float4 s1 = *reinterpret_cast<const float4*>(sc + c * 8 + 4);
    const float4 s2 = *reinterpret_cast<const float4*>(sc + 64 + c * 8);
    const float4 s3 = *reinterpret_cast<const float4*>(sc + 64 + c * 8 + 4);
    float x[16];
    x[0] = i2_lo(u0.x) * s0.x;
    x[1] = i2_hi(u0.x) * s0.y;
    x[2] = i2_lo(u0.y) * s0.z;
    x[3] = i2_hi(u0.y) * s0.w;
    x[4] = i2_lo(u0.z) * s1.x;
    x[5] = i2_hi(u0.z) * s1.y;
    x[6] = i2_lo(u0.w) * s1.z;
    x[7] = i2_hi(u0.w) * s1.w;
    x[8] = i2_lo(u1.x) * s2.x;
    x[9] = i2_hi(u1.x) * s2.y;
    x[10] = i2_lo(u1.y) * s2.z;
    x[11] = i2_hi(u1.y) * s2.w;
    x[12] = i2_lo(u1.z) * s3.x;
    x[13] = i2_hi(u1.z) * s3.y;
    x[14] = i2_lo(u1.w) * s3.z;
    x[15] = i2_hi(u1.w) * s3.w;
    float ss = 0.f;
  #pragma unroll
    for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
    ss += i2_xor(ss, 1);
    ss += i2_xor(ss, 2);
    ss += i2_xor(ss, 4);
    const float f = fminf(1.0f, a.q_clip / (sqrtf(ss) + 1e-6f));
  #pragma unroll
    for (int w = 0; w < 8; ++w) {
      float y0 = i2_round_half(x[2 * w] * f);
      float y1 = i2_round_half(x[2 * w + 1] * f);
      if (a.log_scale) {
        y0 = i2_round_half(y0 * tau);
        y1 = i2_round_half(y1 * tau);
      }
      qreg[g][w] = i2_pack(y0, y1);
    }

    const uint32_t rraw = *reinterpret_cast<const uint32_t*>(
        a.rel + (int64_t)t * a.rel_stride + hq * 16 + 2 * c);
    const float2 rs =
        *reinterpret_cast<const float2*>(a.rel_scale + hq * 16 + 2 * c);
    const float r0 = i2_lo(rraw) * rs.x, r1 = i2_hi(rraw) * rs.y;
    float rss = r0 * r0 + r1 * r1;
    rss += i2_xor(rss, 1);
    rss += i2_xor(rss, 2);
    rss += i2_xor(rss, 4);
    const float rf = fminf(1.0f, a.rel_clip / (sqrtf(rss) + 1e-6f));
    float y0 = i2_round_half(r0 * rf), y1 = i2_round_half(r1 * rf);
    if (a.log_scale) {
      y0 = i2_round_half(y0 * tau);
      y1 = i2_round_half(y1 * tau);
    }
    relw[g] = i2_pack(y0, y1);
  }

  float m_run[G], l_run[G], acc[G][16];
  #pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = -INFINITY;
    l_run[g] = 0.f;
  #pragma unroll
    for (int i = 0; i < 16; ++i) acc[g][i] = 0.f;
  }

  const int pq = L - 1;
  const int bs_mask = (1 << a.bs_shift) - 1;
  const int64_t head_off = (int64_t)kvh * a.kv_s1;
  for (int st = st0; st < st1; ++st) {
    const int kb = ds.lo16 + st * kI2SubTile;
    const int page = __builtin_amdgcn_readfirstlane(
        a.block_table[(int64_t)r * a.bt_stride + (kb >> a.bs_shift)]);
    const int slot0 = kb & bs_mask;
    uint4 kk[4][2], vv[4][2];
    bool valid[4];
  #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      const int j = kb + rho * 4 + oc;
      valid[rho] = (j >= ds.lo) && (j < L);
      if (valid[rho]) {
        const _Float16* p = a.kv + (int64_t)page * a.kv_s0 + head_off +
                            (int64_t)(slot0 + rho * 4 + oc) * a.kv_s2;
        kk[rho][0] = *reinterpret_cast<const uint4*>(p + c * 8);
        kk[rho][1] = *reinterpret_cast<const uint4*>(p + 64 + c * 8);
        vv[rho][0] = *reinterpret_cast<const uint4*>(p + 128 + c * 8);
        vv[rho][1] = *reinterpret_cast<const uint4*>(p + 192 + c * 8);
      } else {
        kk[rho][0] = kk[rho][1] = make_uint4(0u, 0u, 0u, 0u);
        vv[rho][0] = vv[rho][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      const int j = kb + rho * 4 + oc;
      const int dist = pq - j;
      uint32_t pw = 0u;
      if (valid[rho] && dist < a.rel_extent)
        pw = *reinterpret_cast<const uint32_t*>(a.proj_t + (int64_t)dist * 16 +
                                                2 * c);
      float v[16];
      v[0] = i2_lo(vv[rho][0].x);
      v[1] = i2_hi(vv[rho][0].x);
      v[2] = i2_lo(vv[rho][0].y);
      v[3] = i2_hi(vv[rho][0].y);
      v[4] = i2_lo(vv[rho][0].z);
      v[5] = i2_hi(vv[rho][0].z);
      v[6] = i2_lo(vv[rho][0].w);
      v[7] = i2_hi(vv[rho][0].w);
      v[8] = i2_lo(vv[rho][1].x);
      v[9] = i2_hi(vv[rho][1].x);
      v[10] = i2_lo(vv[rho][1].y);
      v[11] = i2_hi(vv[rho][1].y);
      v[12] = i2_lo(vv[rho][1].z);
      v[13] = i2_hi(vv[rho][1].z);
      v[14] = i2_lo(vv[rho][1].w);
      v[15] = i2_hi(vv[rho][1].w);
  #pragma unroll
      for (int g = 0; g < G; ++g) {
        float qk = 0.f;
        qk = i2_dot2(qreg[g][0], kk[rho][0].x, qk);
        qk = i2_dot2(qreg[g][1], kk[rho][0].y, qk);
        qk = i2_dot2(qreg[g][2], kk[rho][0].z, qk);
        qk = i2_dot2(qreg[g][3], kk[rho][0].w, qk);
        qk = i2_dot2(qreg[g][4], kk[rho][1].x, qk);
        qk = i2_dot2(qreg[g][5], kk[rho][1].y, qk);
        qk = i2_dot2(qreg[g][6], kk[rho][1].z, qk);
        qk = i2_dot2(qreg[g][7], kk[rho][1].w, qk);
        float sp = fmaf(qk, a.scale, i2_dot2(relw[g], pw, 0.f));
        sp += i2_xor(sp, 1);
        sp += i2_xor(sp, 2);
        sp += i2_xor(sp, 4);
        const float s = valid[rho] ? sp : -INFINITY;
        const float m_new = fmaxf(m_run[g], s);
        const float alpha =
            (m_run[g] == -INFINITY) ? 0.f : __expf(m_run[g] - m_new);
        const float p = (s == -INFINITY) ? 0.f : __expf(s - m_new);
        l_run[g] = fmaf(l_run[g], alpha, p);
  #pragma unroll
        for (int i = 0; i < 16; ++i)
          acc[g][i] = fmaf(p, v[i], acc[g][i] * alpha);
        m_run[g] = m_new;
      }
    }
  }

  // ---- merge the four octets of the wave
  #pragma unroll
  for (int g = 0; g < G; ++g) {
  #pragma unroll
    for (int off = 8; off <= 16; off <<= 1) {
      const float m2 = i2_xor(m_run[g], off);
      const float l2 = i2_xor(l_run[g], off);
      const float mn = fmaxf(m_run[g], m2);
      const float f1 = (m_run[g] == -INFINITY) ? 0.f : __expf(m_run[g] - mn);
      const float f2 = (m2 == -INFINITY) ? 0.f : __expf(m2 - mn);
      l_run[g] = l_run[g] * f1 + l2 * f2;
  #pragma unroll
      for (int i = 0; i < 16; ++i)
        acc[g][i] = acc[g][i] * f1 + i2_xor(acc[g][i], off) * f2;
      m_run[g] = mn;
    }
  }
  if (oc != 0) return;

  #pragma unroll
  for (int g = 0; g < G; ++g) {
    const int hq = hq0 + g;
    if (ds.n_active == 1) {
      const float sink = (float)a.sink[(int64_t)t * a.sink_stride + hq];
      const float mg = fmaxf(m_run[g], sink);
      const float w = (m_run[g] == -INFINITY) ? 0.f : __expf(m_run[g] - mg);
      const float den = fmaf(l_run[g], w, __expf(sink - mg));
      const float f = w / den;
      float o[16];
  #pragma unroll
      for (int i = 0; i < 16; ++i) o[i] = acc[g][i] * f;
      if (a.gate != nullptr) {
        const _Float16* gr = a.gate + (int64_t)t * a.gate_stride + hq * kI2D;
        const uint4 g0 = *reinterpret_cast<const uint4*>(gr + c * 8);
        const uint4 g1 = *reinterpret_cast<const uint4*>(gr + 64 + c * 8);
        const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
  #pragma unroll
        for (int i = 0; i < 8; ++i) {
          o[2 * i] *= i2_sigmoid(i2_lo(gw[i]));
          o[2 * i + 1] *= i2_sigmoid(i2_hi(gw[i]));
        }
      }
      _Float16* dst = a.out + (int64_t)t * a.out_stride + hq * kI2D;
      *reinterpret_cast<uint4*>(dst + c * 8) =
          make_uint4(i2_pack(o[0], o[1]), i2_pack(o[2], o[3]),
                     i2_pack(o[4], o[5]), i2_pack(o[6], o[7]));
      *reinterpret_cast<uint4*>(dst + 64 + c * 8) =
          make_uint4(i2_pack(o[8], o[9]), i2_pack(o[10], o[11]),
                     i2_pack(o[12], o[13]), i2_pack(o[14], o[15]));
    } else {
      const int64_t row = ((int64_t)split * a.R + r) * a.nh + hq;
      float* po = a.part_o + row * kI2D;
      *reinterpret_cast<float4*>(po + c * 8) =
          make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
      *reinterpret_cast<float4*>(po + c * 8 + 4) =
          make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
      *reinterpret_cast<float4*>(po + 64 + c * 8) =
          make_float4(acc[g][8], acc[g][9], acc[g][10], acc[g][11]);
      *reinterpret_cast<float4*>(po + 64 + c * 8 + 4) =
          make_float4(acc[g][12], acc[g][13], acc[g][14], acc[g][15]);
      if (c == 0)
        *reinterpret_cast<float2*>(a.part_ml + row * 2) =
            make_float2(m_run[g], l_run[g]);
    }
  }
}

// ------------------------------ tile (q_len >= 2) --------------------------
// Grid (ceil(max_q / 16), H_kv * S, R), block kI2TileThreads.  The CTA owns 16
// query tokens x G heads of one kv head (ROWS = 16 G rows) and walks 64-key
// tiles of its key range.  Thread (rg, kg): rg = tid / 16 (RGS row groups of
// TPT tokens x G heads), kg = tid % 16 (16 key / dim groups).
//   QK^T : S[row][key] for keys {kg + 16 u}, u < 4, all 128 dims, V_DOT2.
//   PV   : O[row][8 dims at 8 kg], keys paired as (m, m + 32) so the V tile is
//          stored pair-interleaved and the probabilities as half2 pairs.
template <int G>
__global__ __launch_bounds__(kI2TileThreads) void i2_tile_kernel(I2AttnArgs a) {
  constexpr int ROWS = kI2BR * G;
  constexpr int RGS = kI2TileThreads / 16;  // row groups per CTA
  constexpr int TPT = kI2BR / RGS;          // query tokens per thread
  constexpr int RPT = TPT * G;              // (token, head) rows per thread
  constexpr int NKR = kI2BC / RGS;          // K rows staged per thread
  constexpr int NVR = 32 / RGS;             // V row pairs staged per thread
  static_assert(kI2TileThreads % 32 == 0 && kI2BR % RGS == 0 &&
                    kI2BC % RGS == 0 && 32 % RGS == 0 &&
                    ROWS * 2 <= kI2TileThreads,
                "tile geometry");
  constexpr int DK = 136;  // halves per sQ / sK row (16B aligned, bank skew)
  constexpr int SPS = 36;  // dwords per sP row

  __shared__ __attribute__((aligned(16))) _Float16 sQ[ROWS * DK];
  __shared__ __attribute__((aligned(16))) _Float16 sK[kI2BC * DK];
  __shared__ __attribute__((aligned(16))) uint32_t sV[32 * kI2D];
  __shared__ __attribute__((aligned(16))) uint32_t sP[ROWS * SPS];
  __shared__ __attribute__((aligned(16))) _Float16 sRel[ROWS * 16];
  __shared__ int s_page[kI2BC];

  const int tid = threadIdx.x;
  const int kvh = blockIdx.y % a.nkv;
  const int split = blockIdx.y / a.nkv;
  const int r = blockIdx.z;
  const int qs = a.qsl[r];
  const int q_len = a.qsl[r + 1] - qs;
  if (q_len < 2) return;
  const int L = a.seq_lens[r];
  const int i0 = blockIdx.x * kI2BR;
  if (L < q_len || i0 >= q_len) return;
  const I2TileSplit ts = i2_tile_split(L, q_len, i0, a.window, a.splits_b);
  const int t0 = ts.t_lo + split * ts.tps;
  if (t0 >= ts.t_hi) return;
  const int t1 = min(ts.t_hi, t0 + ts.tps);
  const int ntok = ts.ntok;
  const int p_first = ts.p_first;
  const int lo_all = ts.lo;
  const int hi_all = p_first + ntok;
  const int bs_mask = (1 << a.bs_shift) - 1;
  const int64_t head_off = (int64_t)kvh * a.kv_s1;

  // ---- q tile: scaled L2 clip (+ log scale) -> fp16 in LDS
  for (int i = tid; i < ROWS * 16; i += kI2TileThreads) {
    const int row = i >> 4, cc = i & 15;
    const int tq = row / G, g = row - tq * G;
    float x[8];
    float ss = 0.f, tau = 1.0f;
    if (tq < ntok) {
      const int tg = qs + i0 + tq;
      const int hq = kvh * G + g;
      const uint4 u = *reinterpret_cast<const uint4*>(
          a.q + (int64_t)tg * a.q_stride + hq * kI2D + cc * 8);
      const float* sc = a.q_scale + hq * kI2D + cc * 8;
      const float4 s0 = *reinterpret_cast<const float4*>(sc);
      const float4 s1 = *reinterpret_cast<const float4*>(sc + 4);
      x[0] = i2_lo(u.x) * s0.x;
      x[1] = i2_hi(u.x) * s0.y;
      x[2] = i2_lo(u.y) * s0.z;
      x[3] = i2_hi(u.y) * s0.w;
      x[4] = i2_lo(u.z) * s1.x;
      x[5] = i2_hi(u.z) * s1.y;
      x[6] = i2_lo(u.w) * s1.z;
      x[7] = i2_hi(u.w) * s1.w;
  #pragma unroll
      for (int k = 0; k < 8; ++k) ss += x[k] * x[k];
      if (a.log_scale) tau = i2_log_tau(a, tg);
    } else {
  #pragma unroll
      for (int k = 0; k < 8; ++k) x[k] = 0.f;
    }
    ss += i2_xor(ss, 1);
    ss += i2_xor(ss, 2);
    ss += i2_xor(ss, 4);
    ss += i2_xor(ss, 8);
    const float f = fminf(1.0f, a.q_clip / (sqrtf(ss) + 1e-6f));
    uint32_t w[4];
  #pragma unroll
    for (int k = 0; k < 4; ++k) {
      float y0 = i2_round_half(x[2 * k] * f);
      float y1 = i2_round_half(x[2 * k + 1] * f);
      if (a.log_scale) {
        y0 = i2_round_half(y0 * tau);
        y1 = i2_round_half(y1 * tau);
      }
      w[k] = i2_pack(y0, y1);
    }
    *reinterpret_cast<uint4*>(sQ + row * DK + cc * 8) =
        make_uint4(w[0], w[1], w[2], w[3]);
  }
  // ---- relative features: scaled L2 clip (+ log scale) -> fp16 in LDS
  {
    const bool act = tid < ROWS * 2;
    const int row = tid >> 1, hf = tid & 1;
    float x[8];
    float ss = 0.f, tau = 1.0f;
  #pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = 0.f;
    if (act) {
      const int tq = row / G, g = row - tq * G;
      if (tq < ntok) {
        const int tg = qs + i0 + tq;
        const int hq = kvh * G + g;
        const uint4 u = *reinterpret_cast<const uint4*>(
            a.rel + (int64_t)tg * a.rel_stride + hq * 16 + hf * 8);
        const float* sc = a.rel_scale + hq * 16 + hf * 8;
        const float4 s0 = *reinterpret_cast<const float4*>(sc);
        const float4 s1 = *reinterpret_cast<const float4*>(sc + 4);
        x[0] = i2_lo(u.x) * s0.x;
        x[1] = i2_hi(u.x) * s0.y;
        x[2] = i2_lo(u.y) * s0.z;
        x[3] = i2_hi(u.y) * s0.w;
        x[4] = i2_lo(u.z) * s1.x;
        x[5] = i2_hi(u.z) * s1.y;
        x[6] = i2_lo(u.w) * s1.z;
        x[7] = i2_hi(u.w) * s1.w;
  #pragma unroll
        for (int k = 0; k < 8; ++k) ss += x[k] * x[k];
        if (a.log_scale) tau = i2_log_tau(a, tg);
      }
    }
    ss += i2_xor(ss, 1);
    if (act) {
      const float f = fminf(1.0f, a.rel_clip / (sqrtf(ss) + 1e-6f));
      uint32_t w[4];
  #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float y0 = i2_round_half(x[2 * k] * f);
        float y1 = i2_round_half(x[2 * k + 1] * f);
        if (a.log_scale) {
          y0 = i2_round_half(y0 * tau);
          y1 = i2_round_half(y1 * tau);
        }
        w[k] = i2_pack(y0, y1);
      }
      *reinterpret_cast<uint4*>(sRel + row * 16 + hf * 8) =
          make_uint4(w[0], w[1], w[2], w[3]);
    }
  }

  const int wave = tid >> 5, lane = tid & 31;
  const int rg = (wave << 1) | (lane >> 4);
  const int kg = lane & 15;
  const int tq0 = TPT * rg;

  float sink_r[RPT];
  float m_run[RPT], l_run[RPT], o_acc[RPT][8];
  #pragma unroll
  for (int ri = 0; ri < RPT; ++ri) {
    const int tq = tq0 + ri / G;
    const int g = ri % G;
    sink_r[ri] =
        (tq < ntok)
            ? (float)
                  a.sink[(int64_t)(qs + i0 + tq) * a.sink_stride + kvh * G + g]
            : 0.f;
    m_run[ri] = -INFINITY;
    l_run[ri] = 0.f;
  #pragma unroll
    for (int d = 0; d < 8; ++d) o_acc[ri][d] = 0.f;
  }

  if (tid < kI2BC)
    s_page[tid] = a.block_table[(int64_t)r * a.bt_stride +
                                (min(t0 * kI2BC + tid, L - 1) >> a.bs_shift)];
  __syncthreads();

  for (int tile = t0; tile < t1; ++tile) {
    const int n0 = tile * kI2BC;
    const int cc = tid & 15;

    // ---- global -> registers: K rows (NKR x 16 B) and V row pairs (NVR x 2 x
    // 16 B)
    uint4 kreg[NKR];
  #pragma unroll
    for (int u = 0; u < NKR; ++u) {
      const int key = (tid >> 4) + RGS * u;
      const int n = n0 + key;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (n >= lo_all && n < hi_all) {
        const _Float16* p = a.kv + (int64_t)s_page[key] * a.kv_s0 + head_off +
                            (int64_t)(n & bs_mask) * a.kv_s2;
        v = *reinterpret_cast<const uint4*>(p + cc * 8);
      }
      kreg[u] = v;
    }
    uint4 vra[NVR], vrb[NVR];
  #pragma unroll
    for (int u = 0; u < NVR; ++u) {
      const int mp = (tid >> 4) + RGS * u;
      const int na = n0 + mp, nb = n0 + mp + 32;
      uint4 va = make_uint4(0u, 0u, 0u, 0u), vb = make_uint4(0u, 0u, 0u, 0u);
      if (na >= lo_all && na < hi_all) {
        const _Float16* p = a.kv + (int64_t)s_page[mp] * a.kv_s0 + head_off +
                            (int64_t)(na & bs_mask) * a.kv_s2;
        va = *reinterpret_cast<const uint4*>(p + kI2D + cc * 8);
      }
      if (nb >= lo_all && nb < hi_all) {
        const _Float16* p = a.kv + (int64_t)s_page[mp + 32] * a.kv_s0 +
                            head_off + (int64_t)(nb & bs_mask) * a.kv_s2;
        vb = *reinterpret_cast<const uint4*>(p + kI2D + cc * 8);
      }
      vra[u] = va;
      vrb[u] = vb;
    }
    __syncthreads();  // (1) previous tile's LDS readers are done
  #pragma unroll
    for (int u = 0; u < NKR; ++u)
      *reinterpret_cast<uint4*>(sK + ((tid >> 4) + RGS * u) * DK + cc * 8) =
          kreg[u];
  #pragma unroll
    for (int u = 0; u < NVR; ++u) {
      const int mp = (tid >> 4) + RGS * u;
      const uint4 va = vra[u], vb = vrb[u];
      uint4 lo4, hi4;
      lo4.x = (va.x & 0xFFFFu) | (vb.x << 16);
      lo4.y = (va.x >> 16) | (vb.x & 0xFFFF0000u);
      lo4.z = (va.y & 0xFFFFu) | (vb.y << 16);
      lo4.w = (va.y >> 16) | (vb.y & 0xFFFF0000u);
      hi4.x = (va.z & 0xFFFFu) | (vb.z << 16);
      hi4.y = (va.z >> 16) | (vb.z & 0xFFFF0000u);
      hi4.z = (va.w & 0xFFFFu) | (vb.w << 16);
      hi4.w = (va.w >> 16) | (vb.w & 0xFFFF0000u);
      *reinterpret_cast<uint4*>(sV + mp * kI2D + cc * 8) = lo4;
      *reinterpret_cast<uint4*>(sV + mp * kI2D + cc * 8 + 4) = hi4;
    }
    __syncthreads();  // (2) K / V tile visible
    if (tid < kI2BC && tile + 1 < t1)
      s_page[tid] =
          a.block_table[(int64_t)r * a.bt_stride +
                        (min((tile + 1) * kI2BC + tid, L - 1) >> a.bs_shift)];

    // ---- S = Q K^T
    float sacc[RPT][4];
  #pragma unroll
    for (int ri = 0; ri < RPT; ++ri)
  #pragma unroll
      for (int u = 0; u < 4; ++u) sacc[ri][u] = 0.f;
  #pragma unroll 2
    for (int dch = 0; dch < 16; ++dch) {
      uint4 kc[4];
  #pragma unroll
      for (int u = 0; u < 4; ++u)
        kc[u] =
            *reinterpret_cast<const uint4*>(sK + (kg + 16 * u) * DK + dch * 8);
  #pragma unroll
      for (int ri = 0; ri < RPT; ++ri) {
        const uint4 qc =
            *reinterpret_cast<const uint4*>(sQ + (tq0 * G + ri) * DK + dch * 8);
  #pragma unroll
        for (int u = 0; u < 4; ++u) {
          float s = sacc[ri][u];
          s = i2_dot2(qc.x, kc[u].x, s);
          s = i2_dot2(qc.y, kc[u].y, s);
          s = i2_dot2(qc.z, kc[u].z, s);
          s = i2_dot2(qc.w, kc[u].w, s);
          sacc[ri][u] = s;
        }
      }
    }
    // ---- scale + relative bias
  #pragma unroll
    for (int tqi = 0; tqi < TPT; ++tqi) {
      const int pq = p_first + tq0 + tqi;
      uint4 rw[G][2];
  #pragma unroll
      for (int g = 0; g < G; ++g) {
        rw[g][0] =
            *reinterpret_cast<const uint4*>(sRel + ((tq0 + tqi) * G + g) * 16);
        rw[g][1] = *reinterpret_cast<const uint4*>(
            sRel + ((tq0 + tqi) * G + g) * 16 + 8);
      }
  #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int dist = pq - (n0 + kg + 16 * u);
        const int idx = min(max(dist, 0), a.rel_extent - 1);
        const uint4 p0 =
            *reinterpret_cast<const uint4*>(a.proj_t + (int64_t)idx * 16);
        const uint4 p1 =
            *reinterpret_cast<const uint4*>(a.proj_t + (int64_t)idx * 16 + 8);
        const bool inb = dist < a.rel_extent;
  #pragma unroll
        for (int g = 0; g < G; ++g) {
          float b = 0.f;
          b = i2_dot2(rw[g][0].x, p0.x, b);
          b = i2_dot2(rw[g][0].y, p0.y, b);
          b = i2_dot2(rw[g][0].z, p0.z, b);
          b = i2_dot2(rw[g][0].w, p0.w, b);
          b = i2_dot2(rw[g][1].x, p1.x, b);
          b = i2_dot2(rw[g][1].y, p1.y, b);
          b = i2_dot2(rw[g][1].z, p1.z, b);
          b = i2_dot2(rw[g][1].w, p1.w, b);
          sacc[tqi * G + g][u] =
              fmaf(sacc[tqi * G + g][u], a.scale, inb ? b : 0.f);
        }
      }
    }
    // ---- mask, online softmax, probabilities -> LDS (fp16 pairs)
  #pragma unroll
    for (int ri = 0; ri < RPT; ++ri) {
      const int tqi = ri / G;
      const int pq = p_first + tq0 + tqi;
      const bool rv = (tq0 + tqi) < ntok;
      float tmax = -INFINITY;
  #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int j = n0 + kg + 16 * u;
        const bool vis =
            rv && (j <= pq) && (a.window <= 0 || j > pq - a.window);
        const float s = vis ? sacc[ri][u] : -INFINITY;
        sacc[ri][u] = s;
        tmax = fmaxf(tmax, s);
      }
      tmax = fmaxf(tmax, i2_xor(tmax, 1));
      tmax = fmaxf(tmax, i2_xor(tmax, 2));
      tmax = fmaxf(tmax, i2_xor(tmax, 4));
      tmax = fmaxf(tmax, i2_xor(tmax, 8));
      float o_scale = 1.0f, psum = 0.f;
      float p[4] = {0.f, 0.f, 0.f, 0.f};
      if (tmax > -INFINITY) {
        const float m_new = fmaxf(m_run[ri], tmax);
        o_scale = (m_run[ri] == -INFINITY) ? 0.f : __expf(m_run[ri] - m_new);
        m_run[ri] = m_new;
  #pragma unroll
        for (int u = 0; u < 4; ++u) {
          p[u] = (sacc[ri][u] == -INFINITY) ? 0.f : __expf(sacc[ri][u] - m_new);
          psum += p[u];
        }
      }
      l_run[ri] = fmaf(l_run[ri], o_scale, psum);
  #pragma unroll
      for (int d = 0; d < 8; ++d) o_acc[ri][d] *= o_scale;
      sP[(tq0 * G + ri) * SPS + kg] = i2_pack(p[0], p[2]);
      sP[(tq0 * G + ri) * SPS + kg + 16] = i2_pack(p[1], p[3]);
    }
    __syncthreads();  // (3) probabilities visible

    // ---- O += P V
  #pragma unroll
    for (int m4 = 0; m4 < 8; ++m4) {
      uint4 pw[RPT];
  #pragma unroll
      for (int ri = 0; ri < RPT; ++ri)
        pw[ri] =
            *reinterpret_cast<const uint4*>(sP + (tq0 * G + ri) * SPS + m4 * 4);
  #pragma unroll
      for (int mm = 0; mm < 4; ++mm) {
        const int mp = m4 * 4 + mm;
        const uint4 va =
            *reinterpret_cast<const uint4*>(sV + mp * kI2D + kg * 8);
        const uint4 vb =
            *reinterpret_cast<const uint4*>(sV + mp * kI2D + kg * 8 + 4);
  #pragma unroll
        for (int ri = 0; ri < RPT; ++ri) {
          const uint32_t pwd = (mm == 0)   ? pw[ri].x
                               : (mm == 1) ? pw[ri].y
                               : (mm == 2) ? pw[ri].z
                                           : pw[ri].w;
          o_acc[ri][0] = i2_dot2(pwd, va.x, o_acc[ri][0]);
          o_acc[ri][1] = i2_dot2(pwd, va.y, o_acc[ri][1]);
          o_acc[ri][2] = i2_dot2(pwd, va.z, o_acc[ri][2]);
          o_acc[ri][3] = i2_dot2(pwd, va.w, o_acc[ri][3]);
          o_acc[ri][4] = i2_dot2(pwd, vb.x, o_acc[ri][4]);
          o_acc[ri][5] = i2_dot2(pwd, vb.y, o_acc[ri][5]);
          o_acc[ri][6] = i2_dot2(pwd, vb.z, o_acc[ri][6]);
          o_acc[ri][7] = i2_dot2(pwd, vb.w, o_acc[ri][7]);
        }
      }
    }
  }

  // ---- epilogue
  #pragma unroll
  for (int ri = 0; ri < RPT; ++ri) {
    float l = l_run[ri];
    l += i2_xor(l, 1);
    l += i2_xor(l, 2);
    l += i2_xor(l, 4);
    l += i2_xor(l, 8);
    const int tq = tq0 + ri / G;
    if (tq >= ntok) continue;
    const int tg = qs + i0 + tq;
    const int hq = kvh * G + ri % G;
    if (ts.n_active == 1) {
      const float m = m_run[ri];
      const float mg = fmaxf(m, sink_r[ri]);
      const float w = (m == -INFINITY) ? 0.f : __expf(m - mg);
      const float den = fmaf(l, w, __expf(sink_r[ri] - mg));
      const float f = w / den;
      float o[8];
  #pragma unroll
      for (int d = 0; d < 8; ++d) o[d] = o_acc[ri][d] * f;
      if (a.gate != nullptr) {
        const uint4 gr = *reinterpret_cast<const uint4*>(
            a.gate + (int64_t)tg * a.gate_stride + hq * kI2D + kg * 8);
        o[0] *= i2_sigmoid(i2_lo(gr.x));
        o[1] *= i2_sigmoid(i2_hi(gr.x));
        o[2] *= i2_sigmoid(i2_lo(gr.y));
        o[3] *= i2_sigmoid(i2_hi(gr.y));
        o[4] *= i2_sigmoid(i2_lo(gr.z));
        o[5] *= i2_sigmoid(i2_hi(gr.z));
        o[6] *= i2_sigmoid(i2_lo(gr.w));
        o[7] *= i2_sigmoid(i2_hi(gr.w));
      }
      *reinterpret_cast<uint4*>(a.out + (int64_t)tg * a.out_stride + hq * kI2D +
                                kg * 8) =
          make_uint4(i2_pack(o[0], o[1]), i2_pack(o[2], o[3]),
                     i2_pack(o[4], o[5]), i2_pack(o[6], o[7]));
    } else {
      const int64_t row =
          ((int64_t)a.tile_row0 + (int64_t)split * a.T + tg) * a.nh + hq;
      float* po = a.part_o + row * kI2D + kg * 8;
      *reinterpret_cast<float4*>(po) =
          make_float4(o_acc[ri][0], o_acc[ri][1], o_acc[ri][2], o_acc[ri][3]);
      *reinterpret_cast<float4*>(po + 4) =
          make_float4(o_acc[ri][4], o_acc[ri][5], o_acc[ri][6], o_acc[ri][7]);
      if (kg == 0)
        *reinterpret_cast<float2*>(a.part_ml + row * 2) =
            make_float2(m_run[ri], l);
    }
  }
}

// ------------------------------- split combine -----------------------------
// Grid (ceil(max_q / 16), H_kv, R), block 128: wave g merges the split partials
// of head g of the kv head, one token at a time; lane l owns the head dims
// [4 l, 4 l + 4).  Pass 1 takes the max over the sink and the split maxima,
// pass 2 forms the split weights (kept in LDS) and the denominator, pass 3
// sums the weighted partial outputs, adds the sink to the denominator and
// applies the output gate.  Tokens whose kernel wrote the final output
// directly (a single active split) are skipped.
template <int G>
__global__ __launch_bounds__(128) void i2_combine_kernel(I2AttnArgs a) {
  __shared__ float s_w[4][kI2MaxSplits];
  const int lane = threadIdx.x & 31;
  const int g = threadIdx.x >> 5;
  const int r = blockIdx.z;
  const int kvh = blockIdx.y;
  const int qs = a.qsl[r];
  const int q_len = a.qsl[r + 1] - qs;
  const int L = a.seq_lens[r];
  const int i0 = blockIdx.x * kI2BR;
  if (q_len <= 0 || L < q_len || i0 >= q_len) return;
  int n_active, ntok;
  if (q_len == 1) {
    n_active = i2_decode_split(L, a.window, a.splits_a).n_active;
    ntok = 1;
  } else {
    const I2TileSplit ts = i2_tile_split(L, q_len, i0, a.window, a.splits_b);
    n_active = ts.n_active;
    ntok = ts.ntok;
  }
  if (n_active <= 1) return;
  const bool active = g < G;  // wave-uniform; every wave reaches the barriers
  const int hq = kvh * G + g;
  // partial row of split s: decode requests keep one row per request, tile
  // requests one row per token behind the decode rows
  const int64_t row0 = (q_len == 1) ? 0 : a.tile_row0;
  const int64_t row_stride = (q_len == 1) ? a.R : a.T;
  for (int tq = 0; tq < ntok; ++tq) {
    const int t = qs + i0 + tq;
    const int64_t tok = (q_len == 1) ? r : t;
    auto prow = [&](int s) {
      return (row0 + s * row_stride + tok) * a.nh + hq;
    };
    float sink = 0.f, mg = 0.f, den = 0.f;
    if (active) {
      sink = (float)a.sink[(int64_t)t * a.sink_stride + hq];
      mg = sink;
      for (int s = lane; s < n_active; s += 32)
        mg = fmaxf(mg, a.part_ml[prow(s) * 2]);
  #pragma unroll
      for (int off = 16; off >= 1; off >>= 1) mg = fmaxf(mg, i2_xor(mg, off));
      for (int s = lane; s < n_active; s += 32) {
        const int64_t row = prow(s);
        const float2 ml = *reinterpret_cast<const float2*>(a.part_ml + row * 2);
        const float w = (ml.x == -INFINITY) ? 0.f : __expf(ml.x - mg);
        s_w[g][s] = w;
        den = fmaf(ml.y, w, den);
      }
  #pragma unroll
      for (int off = 16; off >= 1; off >>= 1) den += i2_xor(den, off);
      den += __expf(sink - mg);
    }
    __syncthreads();
    if (active) {
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll 8
      for (int s = 0; s < n_active; ++s) {
        const float w = s_w[g][s];
        const float4 p = *reinterpret_cast<const float4*>(
            a.part_o + prow(s) * kI2D + 4 * lane);
        o.x = fmaf(p.x, w, o.x);
        o.y = fmaf(p.y, w, o.y);
        o.z = fmaf(p.z, w, o.z);
        o.w = fmaf(p.w, w, o.w);
      }
      o.x /= den;
      o.y /= den;
      o.z /= den;
      o.w /= den;
      if (a.gate != nullptr) {
        const uint2 gw = *reinterpret_cast<const uint2*>(
            a.gate + (int64_t)t * a.gate_stride + hq * kI2D + 4 * lane);
        o.x *= i2_sigmoid(i2_lo(gw.x));
        o.y *= i2_sigmoid(i2_hi(gw.x));
        o.z *= i2_sigmoid(i2_lo(gw.y));
        o.w *= i2_sigmoid(i2_hi(gw.y));
      }
      *reinterpret_cast<uint2*>(a.out + (int64_t)t * a.out_stride + hq * kI2D +
                                4 * lane) =
          make_uint2(i2_pack(o.x, o.y), i2_pack(o.z, o.w));
    }
    __syncthreads();  // s_w is reused by the next token
  }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
static int i2_log2_exact(int64_t v, int64_t min_v, const char* what) {
  TORCH_CHECK(v >= min_v && (v & (v - 1)) == 0, what,
              " must be a power of two >= ", min_v, ", got ", v);
  int s = 0;
  while ((int64_t(1) << s) < v) ++s;
  return s;
}

static void i2_check_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a GPU tensor");
}

static void i2_check_i32(const torch::Tensor& t, const char* name) {
  i2_check_cuda(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.is_contiguous(), name,
              " must be a contiguous int32 tensor");
}

static void i2_check_rows(const torch::Tensor& t, const char* name,
                          int dtype_bytes, int64_t cols) {
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1, name,
              " must be [T, ", cols, "] with unit column stride");
  const int64_t align = 16 / dtype_bytes;
  TORCH_CHECK(t.stride(0) % align == 0 &&
                  (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0,
              name, " rows must be 16-byte aligned");
}

static I2ConvMeta i2_conv_meta(const torch::Tensor& conv_cache,
                               const torch::Tensor& block_table,
                               const torch::Tensor& seq_idx,
                               const torch::Tensor& qsl,
                               const torch::Tensor& seq_lens,
                               int64_t block_size) {
  i2_check_cuda(conv_cache, "conv_cache");
  TORCH_CHECK(conv_cache.scalar_type() == torch::kHalf &&
                  conv_cache.dim() == 4 && conv_cache.stride(3) == 1,
              "conv_cache must be fp16 [blocks, H, ring_rows, W]");
  TORCH_CHECK(conv_cache.size(2) >= 3,
              "the conv state keeps at least k - 1 = 3 ring rows per block");
  i2_check_i32(block_table, "block_table");
  i2_check_i32(seq_idx, "seq_idx");
  i2_check_i32(qsl, "query_start_loc");
  i2_check_i32(seq_lens, "seq_lens");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [R, max_blocks]");
  TORCH_CHECK(conv_cache.stride(0) % 4 == 0 && conv_cache.stride(1) % 4 == 0 &&
                  conv_cache.stride(2) % 4 == 0 &&
                  (reinterpret_cast<uintptr_t>(conv_cache.data_ptr()) & 7) == 0,
              "conv_cache must be 8-byte aligned");
  I2ConvMeta m;
  m.block_table = block_table.data_ptr<int32_t>();
  m.seq_idx = seq_idx.data_ptr<int32_t>();
  m.qsl = qsl.data_ptr<int32_t>();
  m.seq_lens = seq_lens.data_ptr<int32_t>();
  m.cache = reinterpret_cast<_Float16*>(conv_cache.data_ptr());
  m.cs0 = conv_cache.stride(0);
  m.cs1 = conv_cache.stride(1);
  m.cs2 = conv_cache.stride(2);
  m.bt_stride = (int)block_table.stride(0);
  m.blk_shift = i2_log2_exact(block_size, 1, "conv block size");
  m.ring = (int)conv_cache.size(2);
  m.T_md = (int)seq_idx.size(0);
  return m;
}

template <typename XT, bool DYN, bool WR>
static void i2_sconv_launch(torch::Tensor x, torch::Tensor w,
                            std::optional<torch::Tensor> dyn, torch::Tensor out,
                            torch::Tensor conv_cache, torch::Tensor block_table,
                            torch::Tensor seq_idx, torch::Tensor qsl,
                            torch::Tensor seq_lens, int64_t stream_offset,
                            int64_t num_kv_heads, int64_t block_size) {
  const int T = (int)x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK(num_kv_heads > 0 && C % num_kv_heads == 0,
              "channels must be divisible by num_kv_heads");
  const int width = C / (int)num_kv_heads;
  TORCH_CHECK(width % 4 == 0 && stream_offset % 4 == 0,
              "stream width and offset must be multiples of 4");
  TORCH_CHECK(conv_cache.size(1) == num_kv_heads &&
                  stream_offset + width <= conv_cache.size(3),
              "conv_cache heads / width do not hold this stream");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.dim() == 2 &&
                  out.size(0) == T && out.size(1) == C && out.stride(1) == 1 &&
                  out.stride(0) % 4 == 0 &&
                  (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              "out must be fp32 [T, C]");
  I2ConvMeta m =
      i2_conv_meta(conv_cache, block_table, seq_idx, qsl, seq_lens, block_size);
  m.T_md = std::min<int>(m.T_md, T);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(T, (C / 4 + 127) / 128);
  const _Float16* dyn_ptr = nullptr;
  int64_t dyn_stride = 0;
  float inv;
  if constexpr (DYN) {
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() &&
                    w.numel() == 4 * (int64_t)C,
                "static_weight must be a contiguous fp32 [4, C] tensor");
    TORCH_CHECK(dyn.has_value(), "dynamic taps required");
    const torch::Tensor& d = *dyn;
    TORCH_CHECK(d.scalar_type() == torch::kHalf && d.dim() == 2 &&
                    d.size(0) >= T && d.size(1) == 4 * num_kv_heads &&
                    d.stride(1) == 1 && d.stride(0) % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(d.data_ptr()) & 7) == 0,
                "dyn must be fp16 [T, H_kv * 4]");
    dyn_ptr = reinterpret_cast<const _Float16*>(d.data_ptr());
    dyn_stride = d.stride(0);
    inv = (float)(1.0 / std::sqrt(9.0));
  } else {
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() &&
                    w.numel() == 4 * (int64_t)C,
                "conv weight must be a contiguous fp32 [C, 1, 4] tensor");
    inv = (float)(1.0 / std::sqrt(5.0));
  }
  i2_sconv_kernel<XT, DYN, WR><<<grid, 128, 0, stream>>>(
      m, reinterpret_cast<const XT*>(x.data_ptr()), x.stride(0),
      w.data_ptr<float>(), dyn_ptr, dyn_stride, out.data_ptr<float>(),
      out.stride(0), C, width, (int)stream_offset, inv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename XT>
static void i2_sconv_modes(bool dynamic, bool write_state, torch::Tensor x,
                           torch::Tensor w, std::optional<torch::Tensor> dyn,
                           torch::Tensor out, torch::Tensor conv_cache,
                           torch::Tensor block_table, torch::Tensor seq_idx,
                           torch::Tensor qsl, torch::Tensor seq_lens,
                           int64_t stream_offset, int64_t num_kv_heads,
                           int64_t block_size) {
  #define I2_SCONV_MODE(DYN, WR)                                          \
    i2_sconv_launch<XT, DYN, WR>(x, w, dyn, out, conv_cache, block_table, \
                                 seq_idx, qsl, seq_lens, stream_offset,   \
                                 num_kv_heads, block_size)
  if (dynamic) {
    if (write_state)
      I2_SCONV_MODE(true, true);
    else
      I2_SCONV_MODE(true, false);
  } else {
    if (write_state)
      I2_SCONV_MODE(false, true);
    else
      I2_SCONV_MODE(false, false);
  }
  #undef I2_SCONV_MODE
}

static void i2_check_conv_x(const torch::Tensor& x) {
  i2_check_cuda(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 &&
                  (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0,
              "x must be [T, C] with unit column stride");
  if (x.scalar_type() == torch::kHalf) {
    TORCH_CHECK(x.stride(0) % 8 == 0, "fp16 x rows must be 16-byte aligned");
  } else {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.stride(0) % 4 == 0,
                "x must be fp16 or fp32");
  }
}

static void i2_sconv_dispatch(bool dynamic, torch::Tensor x, torch::Tensor w,
                              std::optional<torch::Tensor> dyn,
                              torch::Tensor out, torch::Tensor conv_cache,
                              torch::Tensor block_table, torch::Tensor seq_idx,
                              torch::Tensor qsl, torch::Tensor seq_lens,
                              int64_t stream_offset, int64_t num_kv_heads,
                              int64_t block_size, bool write_state) {
  i2_check_conv_x(x);
  if (x.scalar_type() == torch::kHalf)
    i2_sconv_modes<_Float16>(dynamic, write_state, x, w, dyn, out, conv_cache,
                             block_table, seq_idx, qsl, seq_lens, stream_offset,
                             num_kv_heads, block_size);
  else
    i2_sconv_modes<float>(dynamic, write_state, x, w, dyn, out, conv_cache,
                          block_table, seq_idx, qsl, seq_lens, stream_offset,
                          num_kv_heads, block_size);
}

void inkling2_sconv_static(torch::Tensor x, torch::Tensor weight,
                           torch::Tensor out, torch::Tensor conv_cache,
                           torch::Tensor block_table, torch::Tensor seq_idx,
                           torch::Tensor query_start_loc,
                           torch::Tensor seq_lens, int64_t stream_offset,
                           int64_t num_kv_heads, int64_t block_size,
                           bool write_state) {
  i2_sconv_dispatch(false, x, weight, std::nullopt, out, conv_cache,
                    block_table, seq_idx, query_start_loc, seq_lens,
                    stream_offset, num_kv_heads, block_size, write_state);
}

void inkling2_sconv_dynamic(torch::Tensor x, torch::Tensor static_weight,
                            torch::Tensor dyn, torch::Tensor out,
                            torch::Tensor conv_cache, torch::Tensor block_table,
                            torch::Tensor seq_idx,
                            torch::Tensor query_start_loc,
                            torch::Tensor seq_lens, int64_t stream_offset,
                            int64_t num_kv_heads, int64_t block_size,
                            bool write_state) {
  i2_sconv_dispatch(true, x, static_weight, dyn, out, conv_cache, block_table,
                    seq_idx, query_start_loc, seq_lens, stream_offset,
                    num_kv_heads, block_size, write_state);
}

template <typename XT>
static void i2_state_write_launch(const torch::Tensor& x, const I2ConvMeta& m,
                                  int T, int C, int width,
                                  int64_t stream_offset) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(T, (C / 4 + 127) / 128);
  i2_state_write_kernel<XT>
      <<<grid, 128, 0, stream>>>(m, reinterpret_cast<const XT*>(x.data_ptr()),
                                 x.stride(0), C, width, (int)stream_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Writes the raw inputs of one stream into the conv state (chunks of more than
// one token per request: call it after the op that reads the history).
void inkling2_conv_state_write(torch::Tensor x, torch::Tensor conv_cache,
                               torch::Tensor block_table, torch::Tensor seq_idx,
                               torch::Tensor query_start_loc,
                               torch::Tensor seq_lens, int64_t stream_offset,
                               int64_t num_kv_heads, int64_t block_size) {
  i2_check_conv_x(x);
  const int T = (int)x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK(num_kv_heads > 0 && C % num_kv_heads == 0,
              "channels must be divisible by num_kv_heads");
  const int width = C / (int)num_kv_heads;
  TORCH_CHECK(width % 4 == 0 && stream_offset % 4 == 0,
              "stream width and offset must be multiples of 4");
  TORCH_CHECK(conv_cache.dim() == 4 && conv_cache.size(1) == num_kv_heads &&
                  stream_offset + width <= conv_cache.size(3),
              "conv_cache heads / width do not hold this stream");
  I2ConvMeta m = i2_conv_meta(conv_cache, block_table, seq_idx, query_start_loc,
                              seq_lens, block_size);
  m.T_md = std::min<int>(m.T_md, T);
  if (x.scalar_type() == torch::kHalf)
    i2_state_write_launch<_Float16>(x, m, T, C, width, stream_offset);
  else
    i2_state_write_launch<float>(x, m, T, C, width, stream_offset);
}

static void i2_check_kv_cache(const torch::Tensor& kv, int& shift) {
  i2_check_cuda(kv, "kv_cache");
  TORCH_CHECK(kv.scalar_type() == torch::kHalf && kv.dim() == 4 &&
                  kv.size(3) == 2 * kI2D && kv.stride(3) == 1,
              "kv_cache must be fp16 [blocks, H_kv, block_size, 256]");
  TORCH_CHECK(kv.stride(0) % 8 == 0 && kv.stride(1) % 8 == 0 &&
                  kv.stride(2) % 8 == 0 &&
                  (reinterpret_cast<uintptr_t>(kv.data_ptr()) & 15) == 0,
              "kv_cache must be 16-byte aligned");
  shift = i2_log2_exact(kv.size(2), 16, "kv block size");
}

void inkling2_kv_prep(torch::Tensor xk, torch::Tensor xv,
                      torch::Tensor k_weight, torch::Tensor v_weight,
                      std::optional<torch::Tensor> k_dyn,
                      std::optional<torch::Tensor> v_dyn,
                      torch::Tensor conv_cache, torch::Tensor conv_block_table,
                      torch::Tensor seq_idx, torch::Tensor query_start_loc,
                      torch::Tensor seq_lens, torch::Tensor kv_cache,
                      torch::Tensor kv_slot_mapping, int64_t k_stream_offset,
                      int64_t v_stream_offset, int64_t conv_block_size,
                      bool write_state, double k_clip, double rms_eps) {
  int kv_shift;
  i2_check_kv_cache(kv_cache, kv_shift);
  const int H = (int)kv_cache.size(1);
  const int T = (int)xk.size(0);
  const int C = H * kI2D;
  i2_check_rows(xk, "xk", 2, C);
  i2_check_rows(xv, "xv", 2, C);
  TORCH_CHECK(xk.scalar_type() == torch::kHalf &&
                  xv.scalar_type() == torch::kHalf && xv.size(0) == T,
              "xk / xv must be fp16 [T, H_kv * 128]");
  TORCH_CHECK(conv_cache.dim() == 4 && conv_cache.size(1) == H &&
                  k_stream_offset % 4 == 0 && v_stream_offset % 4 == 0 &&
                  k_stream_offset + kI2D <= conv_cache.size(3) &&
                  v_stream_offset + kI2D <= conv_cache.size(3),
              "conv_cache does not hold the k / v streams");
  const bool dynamic = k_dyn.has_value();
  TORCH_CHECK(dynamic == v_dyn.has_value(),
              "k_dyn and v_dyn must be given together");
  for (const torch::Tensor* w : {&k_weight, &v_weight})
    TORCH_CHECK(w->scalar_type() == torch::kFloat32 && w->is_contiguous() &&
                    w->numel() == 4 * (int64_t)C,
                "k / v conv weights must be contiguous fp32 with 4 * C values");
  i2_check_cuda(kv_slot_mapping, "kv_slot_mapping");
  TORCH_CHECK(kv_slot_mapping.scalar_type() == torch::kInt64 &&
                  kv_slot_mapping.is_contiguous() &&
                  kv_slot_mapping.size(0) >= seq_idx.size(0),
              "kv_slot_mapping must be a contiguous int64 [T] tensor");
  I2ConvMeta m = i2_conv_meta(conv_cache, conv_block_table, seq_idx,
                              query_start_loc, seq_lens, conv_block_size);
  m.T_md = std::min<int>(m.T_md, T);
  const _Float16* kd = nullptr;
  const _Float16* vd = nullptr;
  int64_t dyn_stride = 0;
  float inv = (float)(1.0 / std::sqrt(5.0));
  if (dynamic) {
    for (const torch::Tensor* d : {&*k_dyn, &*v_dyn})
      TORCH_CHECK(d->scalar_type() == torch::kHalf && d->dim() == 2 &&
                      d->size(0) >= T && d->size(1) == 4 * H &&
                      d->stride(1) == 1 && d->stride(0) % 4 == 0 &&
                      (reinterpret_cast<uintptr_t>(d->data_ptr()) & 7) == 0,
                  "dyn must be fp16 [T, H_kv * 4]");
    TORCH_CHECK(k_dyn->stride(0) == v_dyn->stride(0),
                "k_dyn and v_dyn must share their row stride");
    kd = reinterpret_cast<const _Float16*>(k_dyn->data_ptr());
    vd = reinterpret_cast<const _Float16*>(v_dyn->data_ptr());
    dyn_stride = k_dyn->stride(0);
    inv = (float)(1.0 / std::sqrt(9.0));
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xk));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(T, (H + 3) / 4);
  auto* kvp = reinterpret_cast<_Float16*>(kv_cache.data_ptr());
  const _Float16* xkp = reinterpret_cast<const _Float16*>(xk.data_ptr());
  const _Float16* xvp = reinterpret_cast<const _Float16*>(xv.data_ptr());
  #define I2_KV_PREP(DYN, WR)                                                 \
    i2_kv_prep_kernel<DYN, WR><<<grid, 128, 0, stream>>>(                     \
        m, xkp, xk.stride(0), xvp, xv.stride(0), k_weight.data_ptr<float>(),  \
        v_weight.data_ptr<float>(), kd, vd, dyn_stride, kvp,                  \
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_shift, \
        kv_slot_mapping.data_ptr<int64_t>(), C, H, (int)k_stream_offset,      \
        (int)v_stream_offset, inv, (float)k_clip, (float)rms_eps)
  if (dynamic) {
    if (write_state) {
      I2_KV_PREP(true, true);
    } else {
      I2_KV_PREP(true, false);
    }
  } else {
    if (write_state) {
      I2_KV_PREP(false, true);
    } else {
      I2_KV_PREP(false, false);
    }
  }
  #undef I2_KV_PREP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_attention(torch::Tensor q, torch::Tensor rel, torch::Tensor sink,
                        std::optional<torch::Tensor> gate,
                        torch::Tensor q_scale, torch::Tensor rel_scale,
                        torch::Tensor rel_proj_t, torch::Tensor positions,
                        torch::Tensor kv_cache, torch::Tensor block_table,
                        torch::Tensor seq_lens, torch::Tensor query_start_loc,
                        torch::Tensor out, torch::Tensor workspace,
                        int64_t window, int64_t rel_extent, bool log_scale,
                        double log_alpha, double log_n_floor, double q_clip,
                        double rel_clip, int64_t max_query_len,
                        int64_t splits_decode, int64_t splits_tile) {
  int bs_shift;
  i2_check_kv_cache(kv_cache, bs_shift);
  const int nkv = (int)kv_cache.size(1);
  const int T = (int)q.size(0);
  TORCH_CHECK(q.dim() == 2 && q.size(1) % kI2D == 0, "q must be [T, nh * 128]");
  const int nh = (int)(q.size(1) / kI2D);
  TORCH_CHECK(nh % nkv == 0, "num q heads must be a multiple of num kv heads");
  const int G = nh / nkv;
  TORCH_CHECK(G >= 1 && G <= 4, "GQA group size must be in [1, 4], got ", G);
  for (const torch::Tensor* t : {&q, &rel, &sink, &out})
    TORCH_CHECK(t->scalar_type() == torch::kHalf && t->is_cuda(),
                "q / rel / sink / out must be fp16 GPU tensors");
  i2_check_rows(q, "q", 2, nh * kI2D);
  i2_check_rows(rel, "rel", 2, nh * 16);
  i2_check_rows(out, "out", 2, nh * kI2D);
  TORCH_CHECK(rel.size(0) == T && out.size(0) == T && sink.size(0) == T,
              "q / rel / sink / out must share the token count");
  TORCH_CHECK(sink.dim() == 2 && sink.size(1) == nh && sink.stride(1) == 1 &&
                  sink.stride(0) % 2 == 0,
              "sink must be [T, nh] with unit column stride");
  int64_t gate_stride = 0;
  const _Float16* gate_ptr = nullptr;
  if (gate.has_value()) {
    TORCH_CHECK(gate->scalar_type() == torch::kHalf && gate->size(0) == T,
                "gate must be fp16 [T, nh * 128]");
    i2_check_rows(*gate, "gate", 2, nh * kI2D);
    gate_ptr = reinterpret_cast<const _Float16*>(gate->data_ptr());
    gate_stride = gate->stride(0);
  }
  TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32 &&
                  q_scale.is_contiguous() &&
                  q_scale.numel() == (int64_t)nh * kI2D &&
                  (reinterpret_cast<uintptr_t>(q_scale.data_ptr()) & 15) == 0,
              "q_scale must be a contiguous fp32 [nh, 128] tensor");
  TORCH_CHECK(rel_scale.scalar_type() == torch::kFloat32 &&
                  rel_scale.is_contiguous() &&
                  rel_scale.numel() == (int64_t)nh * 16 &&
                  (reinterpret_cast<uintptr_t>(rel_scale.data_ptr()) & 15) == 0,
              "rel_scale must be a contiguous fp32 [nh, 16] tensor");
  TORCH_CHECK(
      rel_proj_t.scalar_type() == torch::kHalf && rel_proj_t.dim() == 2 &&
          rel_proj_t.size(1) == 16 && rel_proj_t.is_contiguous() &&
          rel_proj_t.size(0) >= rel_extent && rel_extent >= 1 &&
          (reinterpret_cast<uintptr_t>(rel_proj_t.data_ptr()) & 15) == 0,
      "rel_proj_t must be a contiguous fp16 [rel_extent, 16] tensor");
  i2_check_cuda(positions, "positions");
  TORCH_CHECK((positions.scalar_type() == torch::kInt64 ||
               positions.scalar_type() == torch::kInt32) &&
                  positions.is_contiguous() && positions.size(0) >= T,
              "positions must be a contiguous int32 / int64 [T] tensor");
  i2_check_i32(block_table, "block_table");
  i2_check_i32(seq_lens, "seq_lens");
  i2_check_i32(query_start_loc, "query_start_loc");
  TORCH_CHECK(
      block_table.dim() == 2 &&
          block_table.size(0) + 1 == query_start_loc.size(0) &&
          seq_lens.size(0) == block_table.size(0),
      "block_table / seq_lens / query_start_loc disagree on the request count");
  const int R = (int)block_table.size(0);
  TORCH_CHECK(window >= 0 && max_query_len >= 1, "bad window / max_query_len");

  const int S_A = (int)splits_decode, S_B = (int)splits_tile;
  TORCH_CHECK(splits_decode >= 1 && splits_decode <= kI2MaxSplits &&
                  splits_tile >= 1 && splits_tile <= kI2MaxSplits,
              "splits must be in [1, ", kI2MaxSplits, "]");
  const int tiles = (int)((max_query_len + kI2BR - 1) / kI2BR);
  const bool has_tile = max_query_len > 1;
  // partial rows: S_A per decode request, then S_B per token of tile requests
  const int64_t rows_a = S_A > 1 ? (int64_t)S_A * R : 0;
  const int64_t rows_b = (has_tile && S_B > 1) ? (int64_t)S_B * T : 0;
  const bool need_ws = rows_a + rows_b > 0;
  if (need_ws) {
    i2_check_cuda(workspace, "workspace");
    TORCH_CHECK(workspace.scalar_type() == torch::kFloat32 &&
                    workspace.is_contiguous() &&
                    workspace.numel() >= (rows_a + rows_b) * nh * (kI2D + 2),
                "workspace must be fp32 with at least (splits_decode * "
                "num_reqs + splits_tile * num_tokens) * num_q_heads * 130 "
                "elements (inkling2_attention_workspace_numel)");
  }
  const int64_t part_o_numel = (rows_a + rows_b) * nh * kI2D;

  I2AttnArgs a;
  a.q = reinterpret_cast<const _Float16*>(q.data_ptr());
  a.q_stride = q.stride(0);
  a.rel = reinterpret_cast<const _Float16*>(rel.data_ptr());
  a.rel_stride = rel.stride(0);
  a.sink = reinterpret_cast<const _Float16*>(sink.data_ptr());
  a.sink_stride = sink.stride(0);
  a.gate = gate_ptr;
  a.gate_stride = gate_stride;
  a.q_scale = q_scale.data_ptr<float>();
  a.rel_scale = rel_scale.data_ptr<float>();
  a.proj_t = reinterpret_cast<const _Float16*>(rel_proj_t.data_ptr());
  a.positions = positions.data_ptr();
  a.pos64 = positions.scalar_type() == torch::kInt64;
  a.out = reinterpret_cast<_Float16*>(out.data_ptr());
  a.out_stride = out.stride(0);
  a.kv = reinterpret_cast<const _Float16*>(kv_cache.data_ptr());
  a.kv_s0 = kv_cache.stride(0);
  a.kv_s1 = kv_cache.stride(1);
  a.kv_s2 = kv_cache.stride(2);
  a.block_table = block_table.data_ptr<int32_t>();
  a.bt_stride = (int)block_table.stride(0);
  a.seq_lens = seq_lens.data_ptr<int32_t>();
  a.qsl = query_start_loc.data_ptr<int32_t>();
  a.part_o = need_ws ? workspace.data_ptr<float>() : nullptr;
  a.part_ml = need_ws ? workspace.data_ptr<float>() + part_o_numel : nullptr;
  a.T = T;
  a.R = R;
  a.nh = nh;
  a.nkv = nkv;
  a.bs_shift = bs_shift;
  a.window = (int)window;
  a.rel_extent = (int)rel_extent;
  a.log_scale = log_scale ? 1 : 0;
  a.splits_a = S_A;
  a.splits_b = S_B;
  a.tile_row0 = (int)rows_a;
  a.log_alpha = (float)log_alpha;
  a.log_nfloor = (float)log_n_floor;
  a.q_clip = (float)q_clip;
  a.rel_clip = (float)rel_clip;
  a.scale = (float)std::pow((double)kI2D, -0.5);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid_a(nkv, (S_A + 3) / 4, R);
  const dim3 grid_b(tiles, nkv * S_B, R);
  const dim3 grid_c(tiles, nkv, R);
  if (R == 0) return;
  #define I2_ATTN_LAUNCH(GG)                                          \
    do {                                                              \
      i2_decode_kernel<GG><<<grid_a, 128, 0, stream>>>(a);            \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
      if (has_tile) {                                                 \
        i2_tile_kernel<GG><<<grid_b, kI2TileThreads, 0, stream>>>(a); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                               \
      }                                                               \
      if (need_ws) {                                                  \
        i2_combine_kernel<GG><<<grid_c, 128, 0, stream>>>(a);         \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                               \
      }                                                               \
    } while (0)
  switch (G) {
    case 1:
      I2_ATTN_LAUNCH(1);
      break;
    case 2:
      I2_ATTN_LAUNCH(2);
      break;
    case 3:
      I2_ATTN_LAUNCH(3);
      break;
    default:
      I2_ATTN_LAUNCH(4);
      break;
  }
  #undef I2_ATTN_LAUNCH
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__
