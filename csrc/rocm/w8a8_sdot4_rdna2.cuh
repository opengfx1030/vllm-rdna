// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 (int8 weights x int8 activations) sdot4 kernels for AMD RDNA2
// (gfx1030): the int8-weight sibling of w4a8_sdot4_rdna2.cuh.
//
// Both kernels consume the activation tile that w4a8_act_quant_kernel
// (PER_GROUP, MT = 8) writes, so the activation side is shared with W4A8
// unchanged. Weights are repacked once at load time (Python,
// vllm/model_executor/layers/quantization/rdna2_w8a8.py) into the K-major
// dword layout below, so one u32x4 load brings 4 columns x 4 K values and
// pairs byte for byte with the A tile.
//
// Contract:
//   w        [K/4, N]        uint32  int8 weights. Chunk c covers K
//                                    positions 8c..8c+7 as two dword rows:
//                                    row 2c holds bytes w[8c+{0,4,1,5}][n],
//                                    row 2c+1 holds w[8c+{2,6,3,7}][n]
//                                    (kAPerm order, byte 0 lowest). Row 2c
//                                    pairs with a8.x, row 2c+1 with a8.y.
//   scales   [K/WG, N]       fp16    symmetric weight scale per (weight
//                                    group, column). WG = K for per-channel
//                                    weights; WG % AG == 0.
//   a        [T][K/8][8][8]  int8    w4a8_act_quant_kernel tile (MT = 8)
//   a_scale  [T][K/AG][8]    f32     per-(token, activation group)
//
//   out[m, n] = sum_g a_scale[m, g] * scales[g*AG/WG, n]
//                     * sum_{k in g} a[m, k] * w[k, n]
//
// i32 accumulates within one activation group (|acc| <= AG * 127 * 128) and
// is flushed to f32 per group with both scales, so the per-(token, group)
// activation scale (llama.cpp's Q8_1 idea) costs one multiply per flush.
//
// Dense: grid (ceil(N/1024), ceil(M/8), split_k). A block walks its K range
// in 256-wide windows staged through 2 KiB of LDS, so no shape is LDS-bound;
// split_k only exists to fill the GPU at small M (fp32 atomics into a zeroed
// scratch the host casts once).
// MoE: grid (num_token_blocks, ceil(N/1024), ceil(K/256)), the
// moe_w4a8_rdna2 launch shape and epilogue.
//
// The W8A8 scheme (symmetric per-channel int8 weights, dynamic per-token int8
// activations) follows curvedinf/int8-vllm@c7fe970 (docs/recipes/README.md,
// INT8_AUDIT_RESULTS.md); no code is taken from it.
//
// Depends only on <cstdint> and clang builtins (plus the W4A8 header), so it
// compiles under hipcc and under plain clang with a builtin shim.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "w4a8_sdot4_rdna2.cuh"

namespace vllm {
namespace w8a8_rdna2 {

namespace ex = vllm::explore_w4a8;
using ex::f16_t;
using ex::f16x4_t;
using ex::u32x2_t;
using ex::u32x4_t;

constexpr int kThreads = 256;
constexpr int kNPerThread = 4;
constexpr int kNTile = kThreads * kNPerThread;      // 1024 output columns
constexpr int kKStep = 32;                          // K positions per step
constexpr int kKWindow = 256;                       // K positions per sweep
constexpr int kMTile = 8;                           // act_quant row tile
constexpr int kChunkBytes = kMTile * 8;             // bytes per 8-K chunk
constexpr int kChunksPerWindow = kKWindow / 8;      // 32
constexpr int kMaxGroupsPerWindow = kKWindow / 32;  // 8 at AG = 32

#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)

// fp16 scales of this thread's 4 columns in weight group wg, as f32.
__device__ __forceinline__ void load_scales(const f16_t* __restrict__ scales,
                                            int wg, int n, int size_n,
                                            float (&s)[kNPerThread]) {
  const f16x4_t v = *reinterpret_cast<const f16x4_t*>(
      scales + static_cast<size_t>(wg) * size_n + n);
  s[0] = static_cast<float>(v.x);
  s[1] = static_cast<float>(v.y);
  s[2] = static_cast<float>(v.z);
  s[3] = static_cast<float>(v.w);
}

// One staged K window: ROWS LDS rows ([ROWS][kKWindow] int8, kAPerm byte
// order within each 8-K chunk) against this thread's 4 columns. `wp` is the
// window's first dword row of column n; `lds_as` holds [groups][ROWS] f32
// activation scales; `k_window` is the window's first K position. The
// weight scale is fetched per activation group, so a per-channel scale
// (w_group == K) and a per-group scale (w_group % AG == 0) share the loop.
template <int ROWS, int AG>
__device__ __forceinline__ void mac_window(
    const int8_t* __restrict__ lds_a, const float* __restrict__ lds_as,
    int groups_in_window, const uint32_t* __restrict__ wp, int size_n,
    const f16_t* __restrict__ scales, int k_window, int w_group, int n,
    float (&cf)[ROWS][kNPerThread]) {
  static_assert(AG == 32 || AG == 64 || AG == 128, "activation group");
  static_assert(ROWS >= 1 && ROWS <= kMTile, "rows per block");
  constexpr int STEPS = AG / kKStep;
  constexpr int DW = kKStep / 8;  // chunks per step

  for (int gi = 0; gi < groups_in_window; ++gi) {
    float s[kNPerThread];
    load_scales(scales, (k_window + gi * AG) / w_group, n, size_n, s);

    int32_t acc[ROWS][kNPerThread];
  #pragma unroll
    for (int m = 0; m < ROWS; ++m) {
  #pragma unroll
      for (int c = 0; c < kNPerThread; ++c) {
        acc[m][c] = 0;
      }
    }

  #pragma unroll
    for (int st = 0; st < STEPS; ++st) {
      uint32_t wx[DW][kNPerThread];
      uint32_t wy[DW][kNPerThread];
  #pragma unroll
      for (int j = 0; j < DW; ++j) {
        ex::load_w<kNPerThread>(wp + static_cast<size_t>(2 * j) * size_n,
                                wx[j]);
        ex::load_w<kNPerThread>(wp + static_cast<size_t>(2 * j + 1) * size_n,
                                wy[j]);
      }
      wp += static_cast<size_t>(2 * DW) * size_n;
      // Row-major tile: every A read of the step is base + immediate.
      const int8_t* a_step = lds_a + gi * AG + st * kKStep;
  #pragma unroll
      for (int j = 0; j < DW; ++j) {
  #pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          const u32x2_t a8 =
              *reinterpret_cast<const u32x2_t*>(a_step + m * kKWindow + j * 8);
  #pragma unroll
          for (int c = 0; c < kNPerThread; ++c) {
            acc[m][c] = ex::sdot4(a8.x, wx[j][c], acc[m][c]);
            acc[m][c] = ex::sdot4(a8.y, wy[j][c], acc[m][c]);
          }
        }
      }
    }

    // Group flush: cvt_f32_i32, per-(token, group) A scale, weight scale.
  #pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      const float sa = lds_as[gi * ROWS + m];
  #pragma unroll
      for (int c = 0; c < kNPerThread; ++c) {
        cf[m][c] =
            __builtin_fmaf(static_cast<float>(acc[m][c]) * sa, s[c], cf[m][c]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dense GEMM: grid (ceil(N / 1024), ceil(M / 8), split_k), 256 threads.
// Each block owns an 8-row tile and the K range [z * k_per_split, +k_per_split)
// (a whole number of 256-K windows except at the K tail). out_f32 != 0 means
// fp32 atomic adds into a zeroed scratch (split_k > 1); otherwise plain fp16
// stores.
// ---------------------------------------------------------------------------
template <int AG>
__global__ __launch_bounds__(kThreads) void w8a8_gemm_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ a_scale,
    const uint32_t* __restrict__ w, const f16_t* __restrict__ scales,
    void* __restrict__ out, int size_m, int size_n, int size_k, int w_group,
    int k_per_split, int out_f32) {
  const int t = ex::thread_x();
  const int n = ex::block_x() * kNTile + t * kNPerThread;
  const int tile = ex::block_y();
  const int k_begin = ex::block_z() * k_per_split;
  const int k_end =
      k_begin + k_per_split < size_k ? k_begin + k_per_split : size_k;
  const int groups_total = size_k / AG;
  const int chunks_total = size_k / 8;
  const int8_t* a_tile =
      a + static_cast<size_t>(tile) * chunks_total * kChunkBytes;
  const float* as_tile =
      a_scale + static_cast<size_t>(tile) * groups_total * kMTile;
  const bool active = n < size_n;

  __shared__ int8_t lds_a[kMTile * kKWindow];
  __shared__ float lds_as[kMaxGroupsPerWindow * kMTile];

  float cf[kMTile][kNPerThread];
  #pragma unroll
  for (int m = 0; m < kMTile; ++m) {
  #pragma unroll
    for (int c = 0; c < kNPerThread; ++c) {
      cf[m][c] = 0.0f;
    }
  }

  for (int kw = k_begin; kw < k_end; kw += kKWindow) {
    const int win = k_end - kw < kKWindow ? k_end - kw : kKWindow;
    const int chunks = win / 8;
    const int groups = win / AG;
    const int chunk0 = kw / 8;
    {
      // 256 threads = 8 rows x 32 chunks: thread t stages chunk t % 32 of
      // row t / 32 from the chunk-major global tile into row-major LDS.
      const int m = t / kChunksPerWindow;
      const int c = t % kChunksPerWindow;
      u32x2_t v = {0u, 0u};
      if (c < chunks) {
        v = *reinterpret_cast<const u32x2_t*>(
            a_tile + static_cast<size_t>(chunk0 + c) * kChunkBytes + m * 8);
      }
      *reinterpret_cast<u32x2_t*>(lds_a + m * kKWindow + c * 8) = v;
    }
    // [groups][8] scales of this window are contiguous in the global tile.
    if (t < groups * kMTile) {
      lds_as[t] = as_tile[static_cast<size_t>(kw / AG) * kMTile + t];
    }
    ex::block_sync();
    if (active) {
      mac_window<kMTile, AG>(lds_a, lds_as, groups,
                             w + static_cast<size_t>(chunk0) * 2 * size_n + n,
                             size_n, scales, kw, w_group, n, cf);
    }
    ex::block_sync();
  }

  if (!active) {
    return;
  }
  #pragma unroll
  for (int m = 0; m < kMTile; ++m) {
    const int row = tile * kMTile + m;
    if (row >= size_m) {
      continue;
    }
    const size_t off = static_cast<size_t>(row) * size_n + n;
    if (out_f32) {
      float* o = static_cast<float*>(out) + off;
  #pragma unroll
      for (int c = 0; c < kNPerThread; ++c) {
        __hip_atomic_fetch_add(o + c, cf[m][c], __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      const f16x4_t v = {
          static_cast<f16_t>(cf[m][0]), static_cast<f16_t>(cf[m][1]),
          static_cast<f16_t>(cf[m][2]), static_cast<f16_t>(cf[m][3])};
      *reinterpret_cast<f16x4_t*>(static_cast<f16_t*>(out) + off) = v;
    }
  }
}

// ---------------------------------------------------------------------------
// Fused MoE: grid (num_token_blocks, ceil(N / 1024), ceil(K / 256)), 256
// threads. Each block owns BLOCK_M sorted tokens of one expert and one 256-K
// window; the epilogue is moe_w4a8_rdna2's (router weight in fp32, then
// fp32 atomics into the scratch for C_T == float or the packed fp16 CAS add
// for C_T == f16_t, with the output_topk row reduce fused).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int AG, typename C_T>
__global__ __launch_bounds__(kThreads) void moe_w8a8_gemm_kernel(
    const int8_t* __restrict__ a,             // [T][K/8][8][8] int8
    const float* __restrict__ a_scale,        // [T][K/AG][8] f32
    const uint32_t* __restrict__ b_q_weight,  // [E, K/4, N]
    const f16_t* __restrict__ b_scales,       // [E, K/WG, N]
    C_T* __restrict__ c,                      // [M*topk or M, N] accumulator
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, int size_m, int size_n,
    int size_k, int w_group, int top_k, bool mul_topk_weight, int output_topk) {
  static_assert(BLOCK_M == 1 || BLOCK_M == 2 || BLOCK_M == 4 || BLOCK_M == 8,
                "token block");
  static_assert(BLOCK_M * kChunksPerWindow <= kThreads,
                "staging needs <= THREADS chunks");
  const int t = ex::thread_x();
  const int token_block = ex::block_x();
  const int n = ex::block_y() * kNTile + t * kNPerThread;
  const int kw = ex::block_z() * kKWindow;
  const int win = size_k - kw < kKWindow ? size_k - kw : kKWindow;
  const int chunks = win / 8;
  const int groups = win / AG;
  const int chunk0 = kw / 8;
  const int groups_total = size_k / AG;
  const int chunks_total = size_k / 8;

  // Early exit for padding blocks / invalid experts (EP expert_map = -1).
  if (token_block * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_w =
      b_q_weight + static_cast<size_t>(expert_id) * (size_k / 4) * size_n;
  const f16_t* expert_s =
      b_scales + static_cast<size_t>(expert_id) * (size_k / w_group) * size_n;

  __shared__ int8_t lds_a[BLOCK_M * kKWindow];
  __shared__ float lds_as[kMaxGroupsPerWindow * BLOCK_M];
  const int offset_m_base = token_block * BLOCK_M;

  // Stage the block's A window (BLOCK_M x 256 int8) and its [groups][BLOCK_M]
  // scales. Rows are addressed through the [T][K/8][8][8] tile as
  // (tile_r = row / 8, m_r = row % 8); padding rows clamp to tile 0 / row 0
  // so reads stay in bounds, and the epilogue skips their write.
  if (t < BLOCK_M * kChunksPerWindow) {
    const int m = t / kChunksPerWindow;
    const int cc = t % kChunksPerWindow;
    const int token_row = sorted_token_ids[offset_m_base + m] / top_k;
    const int tr = token_row < size_m ? (token_row >> 3) : 0;
    const int mr = token_row < size_m ? (token_row & 7) : 0;
    u32x2_t v = {0u, 0u};
    if (cc < chunks) {
      v = *reinterpret_cast<const u32x2_t*>(
          a +
          (static_cast<size_t>(tr) * chunks_total + chunk0 + cc) * kChunkBytes +
          mr * 8);
    }
    *reinterpret_cast<u32x2_t*>(lds_a + m * kKWindow + cc * 8) = v;
  }
  if (t < groups * BLOCK_M) {
    const int gi = t / BLOCK_M;
    const int m = t % BLOCK_M;
    const int token_row = sorted_token_ids[offset_m_base + m] / top_k;
    const int tr = token_row < size_m ? (token_row >> 3) : 0;
    const int mr = token_row < size_m ? (token_row & 7) : 0;
    lds_as[gi * BLOCK_M + m] =
        a_scale[(static_cast<size_t>(tr) * groups_total + kw / AG + gi) *
                    kMTile +
                mr];
  }
  ex::block_sync();

  // Every thread owns 4 contiguous N columns; with N % 4 == 0 an in-range n
  // keeps all four columns in range. No barrier follows this point.
  if (n >= size_n) return;

  float cf[BLOCK_M][kNPerThread];
  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
  #pragma unroll
    for (int cc = 0; cc < kNPerThread; ++cc) {
      cf[m][cc] = 0.0f;
    }
  }
  mac_window<BLOCK_M, AG>(
      lds_a, lds_as, groups,
      expert_w + static_cast<size_t>(chunk0) * 2 * size_n + n, size_n, expert_s,
      kw, w_group, n, cf);

  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
  #pragma unroll
      for (int cc = 0; cc < kNPerThread; ++cc) {
        cf[m][cc] *= tw;
      }
    }

    const int64_t out_row = (output_topk > 0)
                                ? static_cast<int64_t>(token_id / output_topk)
                                : static_cast<int64_t>(token_id);
    C_T* o = c + out_row * size_n + n;
    if constexpr (std::is_same_v<C_T, float>) {
  #pragma unroll
      for (int cc = 0; cc < kNPerThread; ++cc) {
        __hip_atomic_fetch_add(o + cc, cf[m][cc], __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      const f16x4_t v = {
          static_cast<f16_t>(cf[m][0]), static_cast<f16_t>(cf[m][1]),
          static_cast<f16_t>(cf[m][2]), static_cast<f16_t>(cf[m][3])};
      ex::atomic_add_f16x4(reinterpret_cast<f16_t*>(o), v);
    }
  }
}

#else  // non-gfx1030 device pass: same signatures, no body.

template <int AG>
__global__ __launch_bounds__(kThreads) void w8a8_gemm_kernel(
    const int8_t*, const float*, const uint32_t*, const f16_t*, void*, int, int,
    int, int, int, int) {}

template <int BLOCK_M, int AG, typename C_T>
__global__ __launch_bounds__(kThreads) void moe_w8a8_gemm_kernel(
    const int8_t*, const float*, const uint32_t*, const f16_t*, C_T*,
    const float*, const int32_t*, const int32_t*, const int32_t*, int, int, int,
    int, int, bool, int) {}

#endif  // __gfx1030__ || !__HIP_DEVICE_COMPILE__

}  // namespace w8a8_rdna2
}  // namespace vllm
