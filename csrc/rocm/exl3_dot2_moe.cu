// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (QTIP bitshift trellis) fused MoE GEMM for RDNA2/RDNA3
// (gfx1030/gfx1100), fp16-only. Real tile layout (locked 2026-08-26,
// see exl3_dot2_common.cuh):
//
//   trellis: [E, k/16, n/16, 256*bits/16] int16 (per-expert packed tiles)
//   Weight(r,c) = decode_3inst<cb>(window at exl3_window_pos<bits>(r,c))
//
// Mirrors moe_q_gemm_rdna2's sorted-token-id MoE structure: 4 N-cols per
// thread, BLOCK_SIZE_M token rows per block, block loops all k-tiles in
// registers, atomic_add_pk4_f16 epilogue (gfx1030 lacks native fp16
// atomics; 64-bit CAS emulation).
//
// Decode is done once per (thread, k-tile) for the thread's 4 columns, then
// reused across all BLOCK_SIZE_M token rows (A-side differs per row).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "exl3_dot2_common.cuh"

#if defined(__HIPCC__) && (defined(__gfx1030__) || defined(__gfx1100__))
  #define __HIP__RDNA__
#endif

#define THREADS_X 256
#define BLOCK_N_COLS 256
#define K_TILE 16

namespace vllm {
namespace exl3_dot2 {

// 64-bit CAS half2-add for 4 consecutive fp16 columns (mxfp4 pattern).
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ int token_row_of(int32_t tid, int top_k) {
  return tid / top_k;
}

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

// One block: BLOCK_SIZE_M token rows (from one expert, sorted) x 256 N-cols,
// all K (looped in registers). grid = (num_token_blocks, n_blks, 1).
template <int BLOCK_SIZE_M, int bits, int cb>
__global__ void moe_gemm_exl3_tile_kernel_rdna(
    const half* __restrict__ a,             // [size_m, size_k]
    half* __restrict__ c,                   // [size_m*top_k, size_n] or [size_m, size_n]
    const int16_t* __restrict__ trellis,    // [E, k/16, n/16, 256*bits/16]
    const float* __restrict__ topk_weights, // [size_m*top_k] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m, const int size_n, const int size_k, const int top_k,
    const int output_topk, const bool mul_topk_weight) {
  constexpr int TILE_WORDS = 8 * bits;
  constexpr int TILE_I16 = 2 * TILE_WORDS;
  constexpr int TILES_PER_BLOCK = BLOCK_N_COLS / 16;  // 16

  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_N_COLS;
  const int n = offset_n + t * 4;
  const int n_tiles = size_n / 16;
  const int n_k_tiles = size_k / K_TILE;
  const int offset_m_base = token_block * BLOCK_SIZE_M;

  if (offset_m_base >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const int16_t* ex_trellis = trellis +
      (int64_t)expert_id * (n_k_tiles * n_tiles * TILE_I16);

  __shared__ half s_a[BLOCK_SIZE_M][K_TILE];
  __shared__ uint32_t s_tile[TILES_PER_BLOCK][TILE_WORDS];

  float acc[BLOCK_SIZE_M][4];
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  for (int k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
    // Stage trellis: this block's 16 n-tiles at k_tile for the expert.
    {
      constexpr int TOTAL_U32 = TILES_PER_BLOCK * TILE_WORDS;
      const int16_t* base = ex_trellis + (int64_t)k_tile * (n_tiles * TILE_I16);
#pragma unroll
      for (int i = t; i < TOTAL_U32; i += THREADS_X) {
        const int tt = i / TILE_WORDS;
        const int wi = i % TILE_WORDS;
        const int16_t* src = base + (blockIdx.y * TILES_PER_BLOCK + tt) * TILE_I16;
        s_tile[tt][wi] = reinterpret_cast<const uint32_t*>(src)[wi];
      }
    }

    // Stage A: BLOCK_SIZE_M rows x 16 K-elements.
    {
#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        half av = __float2half_rn(0.0f);
        const int32_t token_id = sorted_token_ids[offset_m_base + m];
        const int row = token_row_of(token_id, top_k);
        if (token_id >= 0 && row < size_m) {
          av = a[(int64_t)row * size_k + k_tile * K_TILE + (t % K_TILE)];
        }
        s_a[m][t % K_TILE] = av;
      }
    }
    __syncthreads();

    if (n < size_n) {
      // Decode 4 columns x 16 windows (4 N-tiles may be shared).
      half w[4][K_TILE];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = (t * 4 + j) % 16;
        const int n_tile = (t * 4 + j) / 16;
#pragma unroll
        for (int r = 0; r < K_TILE; ++r) {
          const int p = exl3_window_pos<bits>(r, col);
          const uint32_t win = exl3_window_at<bits>(s_tile[n_tile], p);
          w[j][r] = decode_3inst<cb>(win);
        }
      }

#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_row = s_a[m];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int i = 0; i < K_TILE / 2; ++i) {
            half2 w2 = __halves2half2(w[j][2 * i], w[j][2 * i + 1]);
            half2 a2 = __halves2half2(a_row[2 * i], a_row[2 * i + 1]);
            acc[m][j] = __builtin_amdgcn_fdot2(w2, a2, acc[m][j], false);
          }
        }
      }
    }
    __syncthreads();
  }

  if (n >= size_n) return;

  // Epilogue: topk weight + (optional) moe_sum-reduce via atomics.
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    const int row = token_row_of(token_id, top_k);
    if (token_id < 0 || row >= size_m) continue;

    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][j] *= tw;
    }

    const int64_t out_row = (output_topk > 0)
                                ? (int64_t)token_id / output_topk
                                : (int64_t)token_id;
    half* out = c + out_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(acc[m][0]),
                               __float2half_rn(acc[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(acc[m][2]),
                               __float2half_rn(acc[m][3]));
    if (output_topk > 0) {
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      union {
        unsigned long long u;
        half2 h2[2];
      } v;
      v.h2[0] = r01;
      v.h2[1] = r23;
      *reinterpret_cast<unsigned long long*>(out) = v.u;
    }
  }
}

#else  // non-RDNA: empty stub for symbol parity

template <int BLOCK_SIZE_M, int bits, int cb>
__global__ void moe_gemm_exl3_tile_kernel_rdna(
    const half*, half*, const int16_t*, const float*, const int32_t*,
    const int32_t*, const int32_t*, const int, const int, const int, const int,
    const int, const bool) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

__forceinline__ int divide_up(int x, int y) { return (x + y - 1) / y; }

template <int BLOCK_SIZE_M, int bits, int cb>
void launch_moe_mcb(const half* a, half* c, const int16_t* trellis,
                    const float* topk_weights, const int32_t* sorted_token_ids,
                    const int32_t* expert_ids,
                    const int32_t* num_tokens_post_padded, int num_token_blocks,
                    int sm, int sn, int sk, int top_k, int output_topk,
                    bool mul_topk_weight, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks, divide_up(sn, BLOCK_N_COLS), 1);
  moe_gemm_exl3_tile_kernel_rdna<BLOCK_SIZE_M, bits, cb>
      <<<grid, block, 0, stream>>>(a, c, trellis, topk_weights,
                                   sorted_token_ids, expert_ids,
                                   num_tokens_post_padded, sm, sn, sk, top_k,
                                   output_topk, mul_topk_weight);
}

template <int BLOCK_SIZE_M, int bits>
void launch_moe_mb(const half* a, half* c, const int16_t* trellis,
                   const float* topk_weights, const int32_t* sorted_token_ids,
                   const int32_t* expert_ids,
                   const int32_t* num_tokens_post_padded, int num_token_blocks,
                   int sm, int sn, int sk, int top_k, int output_topk,
                   bool mul_topk_weight, int cb, cudaStream_t stream) {
  if (cb == 0)
    launch_moe_mcb<BLOCK_SIZE_M, bits, 0>(
        a, c, trellis, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, num_token_blocks, sm, sn, sk, top_k,
        output_topk, mul_topk_weight, stream);
  else if (cb == 1)
    launch_moe_mcb<BLOCK_SIZE_M, bits, 1>(
        a, c, trellis, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, num_token_blocks, sm, sn, sk, top_k,
        output_topk, mul_topk_weight, stream);
  else
    TORCH_CHECK(false, "moe_exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int BLOCK_SIZE_M>
void launch_moe_m(const half* a, half* c, const int16_t* trellis,
                  const float* topk_weights, const int32_t* sorted_token_ids,
                  const int32_t* expert_ids,
                  const int32_t* num_tokens_post_padded, int num_token_blocks,
                  int sm, int sn, int sk, int top_k, int output_topk,
                  bool mul_topk_weight, int bits, int cb, cudaStream_t stream) {
  switch (bits) {
    case 2:
      launch_moe_mb<BLOCK_SIZE_M, 2>(a, c, trellis, topk_weights,
                                     sorted_token_ids, expert_ids,
                                     num_tokens_post_padded, num_token_blocks,
                                     sm, sn, sk, top_k, output_topk,
                                     mul_topk_weight, cb, stream);
      break;
    case 3:
      launch_moe_mb<BLOCK_SIZE_M, 3>(a, c, trellis, topk_weights,
                                     sorted_token_ids, expert_ids,
                                     num_tokens_post_padded, num_token_blocks,
                                     sm, sn, sk, top_k, output_topk,
                                     mul_topk_weight, cb, stream);
      break;
    case 4:
      launch_moe_mb<BLOCK_SIZE_M, 4>(a, c, trellis, topk_weights,
                                     sorted_token_ids, expert_ids,
                                     num_tokens_post_padded, num_token_blocks,
                                     sm, sn, sk, top_k, output_topk,
                                     mul_topk_weight, cb, stream);
      break;
    default:
      TORCH_CHECK(false, "moe_exl3_gemm_rdna2: unsupported bits=", bits);
  }
}

void launch_moe(const half* a, half* c, const int16_t* trellis,
                const float* topk_weights, const int32_t* sorted_token_ids,
                const int32_t* expert_ids,
                const int32_t* num_tokens_post_padded, int num_token_blocks,
                int sm, int sn, int sk, int top_k, int block_size_m,
                int output_topk, bool mul_topk_weight, int bits, int cb,
                cudaStream_t stream) {
  switch (block_size_m) {
    case 1:
      launch_moe_m<1>(a, c, trellis, topk_weights, sorted_token_ids,
                      expert_ids, num_tokens_post_padded, num_token_blocks,
                      sm, sn, sk, top_k, output_topk, mul_topk_weight, bits,
                      cb, stream);
      break;
    case 2:
      launch_moe_m<2>(a, c, trellis, topk_weights, sorted_token_ids,
                      expert_ids, num_tokens_post_padded, num_token_blocks,
                      sm, sn, sk, top_k, output_topk, mul_topk_weight, bits,
                      cb, stream);
      break;
    case 4:
      launch_moe_m<4>(a, c, trellis, topk_weights, sorted_token_ids,
                      expert_ids, num_tokens_post_padded, num_token_blocks,
                      sm, sn, sk, top_k, output_topk, mul_topk_weight, bits,
                      cb, stream);
      break;
    case 8:
      launch_moe_m<8>(a, c, trellis, topk_weights, sorted_token_ids,
                      expert_ids, num_tokens_post_padded, num_token_blocks,
                      sm, sn, sk, top_k, output_topk, mul_topk_weight, bits,
                      cb, stream);
      break;
    default:
      TORCH_CHECK(false, "moe_exl3_gemm_rdna2: block_size_m must be 1/2/4/8");
  }
}

}  // namespace exl3_dot2
}  // namespace vllm

void moe_exl3_gemm_rdna2(
    torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, int64_t bits, int64_t cb) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(trellis.is_cuda(), "trellis must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2 && trellis.dim() == 4,
              "a 2D, trellis [E, K/16, N/16, 256*bits/16]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "moe_exl3_gemm_rdna2 only supports fp16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int sm = (int)a.size(0);
  const int sk = (int)a.size(1);
  const int sn = (int)c.size(1);
  int qk = (int)trellis.size(1);
  TORCH_CHECK(qk * 16 == sk, "trellis K-dim * 16 must equal a.size(1)");
  TORCH_CHECK(sn % 16 == 0 && sk % 16 == 0, "K and N must be multiples of 16");

  int expert_weight_stride = (int)(trellis.size(1) * trellis.size(2) *
                                   trellis.size(3));
  (void)expert_weight_stride;
  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  vllm::exl3_dot2::launch_moe(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      (const int16_t*)trellis.data_ptr(), topk_w_ptr,
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, sm, sn, sk,
      (int)top_k, (int)block_size_m, (int)output_topk, mul_topk_weight,
      (int)bits, (int)cb, stream);
}