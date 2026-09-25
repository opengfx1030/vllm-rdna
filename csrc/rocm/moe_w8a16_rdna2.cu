// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W8A16 (INT8 weight + fp16 act) kernel for AMD RDNA2 (gfx1030).
//
// Uses v_dot2_f32_f16 for the inner dot product (native RDNA2 fp16 dot).
// fp32 accumulator, packed CAS-64 atomic-write epilogue. Block grid and
// per-thread N mapping mirror moe_q_gemm_rdna2 (W4A16) so we share the
// (num_token_blocks, n_blocks, k_tiles) launch shape.
//
// Storage: int8 [E, K, N] (one byte per element, no nibble packing).
// Per-group fp16 scale and zero (block_shape=[0, group_size]).
//
// gfx1030 lacks v_dot2_f32_bf16; bf16 inputs are rejected at host.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "q_gemm_rdna2_common.cuh"  // dot22_8_f, atomic_add_pk4_f16, epilogue
#include "ops.h"
#include "qdq_8_rdna2.cuh"

namespace vllm {
namespace w8a16_rdna2 {

constexpr int THREADS = 256;
constexpr int BLOCK_KN_SIZE = 256;
constexpr int LDS_PAD = 8;

template <int BLOCK_SIZE_M>
__global__ void moe_w8a16_gemm_kernel(
    const half* __restrict__ a,            // [M, K] fp16
    half* __restrict__ c,                  // [M, N] fp16
    const int8_t* __restrict__ b_q,        // [E, K, N] int8
    const half* __restrict__ b_scales,     // [E, groups, N] fp16
    const half* __restrict__ b_zeros,      // [E, groups, N] fp16 or nullptr
    const float* __restrict__ topk_w,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k,
    int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int n_block = blockIdx.y;
  const int k_tile_idx = blockIdx.z;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;

  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  // Each thread handles 4 N columns (4 fdot2 calls per inner K-iter).
  // Grid: 256 threads per block * 4 N cols = 1024 N cols per block.
  const int offset_n = n_block * THREADS * 4;
  const int n = offset_n + t * 4;
  const int offset_k = k_tile_idx * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int group_size = size_k / groups;
  const int K_unroll = 32;  // 4 j-iterations * 8 K-positions each

  const int8_t* expert_b = b_q + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const half* expert_zeros =
      (b_zeros != nullptr)
          ? b_zeros + (int64_t)expert_id * expert_zeros_stride
          : nullptr;

  // LDS for activations: pad to avoid bank conflicts when different
  // M rows read the same K offset.
  __shared__ half block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD];

  // LDS for per-group scale and zero, shared across all threads in
  // the block. The group index depends only on K position (not on
  // thread), so all threads can share the same scale/zero for a given
  // group. MAX_GROUPS_PER_TILE=4 covers the common case where the
  // inner j-loop's 4 groups (k/gs, (k+8)/gs, (k+16)/gs, (k+24)/gs)
  // are consecutive. For K-tiles where the group index exceeds 3
  // (group_size < K_unroll/4 = 8), the inner loop falls back to
  // inline global loads.
  constexpr int MAX_GROUPS_PER_TILE = 4;
  __shared__ half2 sh_lds[MAX_GROUPS_PER_TILE][4];  // [group][N col 0..3]
  __shared__ half2 zh_lds[MAX_GROUPS_PER_TILE][4];

  float block_c[BLOCK_SIZE_M][4] = {0};

  for (int k = offset_k; k < end_k; k += K_unroll) {
    // Stage activations: [BLOCK_SIZE_M, K_unroll] fp16. All threads
    // participate cooperatively: thread t stages position t in each
    // row. Since THREADS=256 > K_unroll=32, only threads 0..K_unroll-1
    // do useful work for each row; the rest are idle (no-op).
    #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      if (t < K_unroll) {
        const int token_id =
            sorted_token_ids[token_block * BLOCK_SIZE_M + m];
        const int row = token_id / top_k;
        if (row < size_m && (k + t) < end_k) {
          block_a[m][t] = a[row * size_k + k + t];
        } else {
          block_a[m][t] = __float2half_rn(0.0f);
        }
      }
    }

    // Pre-load per-group scale/zero into shared memory (shared across
    // all threads in the block). The inner j-loop touches groups
    // g0, g0+1, g0+2, g0+3 where g0 = k / group_size. We pre-load
    // up to MAX_GROUPS_PER_TILE=4 groups starting at g0. Threads
    // cooperatively load: thread t handles N cols at positions
    // offset_n + t (one N col per thread, since MAX_GROUPS_PER_TILE
    // groups * 4 N cols = 16 N cols need to be loaded but we have
    // 256 threads, so threads 0..15 do the actual loads, rest idle).
    // For K-tiles where g0+3 >= groups, the inner loop falls back
    // to inline global loads (see inner loop below).
    if (t < MAX_GROUPS_PER_TILE * 4) {
      const int group_local = t / 4;  // 0..3
      const int kk = t % 4;            // 0..3
      const int k_off = k + 8 * group_local;
      const int group = k_off / group_size;
      const int n_chunk = n + kk;
      half scale_v = __float2half_rn(0.0f);
      half zero_v = __float2half_rn(0.0f);
      if (k_off < end_k && n_chunk < size_n && group < groups) {
        scale_v = expert_scales[group * size_n + n_chunk];
        if (expert_zeros != nullptr) {
          zero_v = expert_zeros[group * size_n + n_chunk];
        }
      }
      sh_lds[group_local][kk] = __half2half2(scale_v);
      zh_lds[group_local][kk] = __half2half2(zero_v);
    }
    __syncthreads();

    if (n >= size_n) {
      __syncthreads();
      continue;
    }

    // Inner loop: 4 j-iterations, each processes 4 N cols × 8 K positions
    for (int j = 0; j < 4; ++j) {
      const int k_off = k + 8 * j;
      const int group = k_off / group_size;

      // Load 8 int8 weights (1 N col × 8 K positions) for each of 4 N cols.
      // For each kk in 0..3, the N col is n + kk; weights are at byte
      // addresses (k_off, n + kk), (k_off + 1, n + kk), ..., (k_off + 7,
      // n + kk). Since storage is K-major with N-major within a row
      // (b_q[expert][k * size_n + n]), the 8 K positions for 1 N col
      // Gather 8 K positions for 1 N col from K-major INT8 storage.
// Storage layout: b_q[expert][k * size_n + n] is at (k, n). The 8 K
// positions for N col n+kk at K positions k_off..k_off+7 are at
// byte addresses (k_off+i) * size_n + (n+kk) for i=0..7. These are
// size_n bytes apart (NOT consecutive), so we must gather 8 bytes
// rather than load a single uint64.
      uint64_t w_packed[4];
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        w_packed[kk] = 0;
        if (n + kk < size_n) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int k_pos = k_off + i;
            if (k_pos < end_k) {
              const int8_t byte =
                  expert_b[(int64_t)k_pos * size_n + (n + kk)];
              w_packed[kk] |= ((uint64_t)(uint8_t)byte) << (i * 8);
            }
          }
        }
      }

      // Dequant 4 N cols. For groups < MAX_GROUPS_PER_TILE, use the
      // LDS-resident scale/zero (pre-loaded by threads 0..15 above).
      // For groups >= MAX_GROUPS_PER_TILE (rare: group_size < 8),
      // load scale/zero inline from global memory.
      half2 dq[4][4];
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        half2 sh_v, zh_v;
        if (group < MAX_GROUPS_PER_TILE) {
          sh_v = sh_lds[j][kk];
          zh_v = zh_lds[j][kk];
        } else {
          // Inline fallback: load from global memory
          const int n_chunk = n + kk;
          half scale_v = __float2half_rn(0.0f);
          half zero_v = __float2half_rn(0.0f);
          if (k_off < end_k && n_chunk < size_n && group < groups) {
            scale_v = expert_scales[group * size_n + n_chunk];
            if (expert_zeros != nullptr) {
              zero_v = expert_zeros[group * size_n + n_chunk];
            }
          }
          sh_v = __half2half2(scale_v);
          zh_v = __half2half2(zero_v);
        }
        dequant_8bit_8_fp16(w_packed[kk], sh_v, zh_v, dq[kk]);
      }

      // Dot each M row against the 4 N cols. dot22_8_f(dq[kk], a_ptr)
      // issues 4 v_dot2_f32_f16 calls covering 8 K positions for the
      // single N col stored in dq[kk][0..3]. a_ptr points into block_a
      // at the 8 K positions for this j-iteration.
      const half* a_ptr_base = &block_a[0][8 * j];
      #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr = a_ptr_base + m * (BLOCK_KN_SIZE + LDS_PAD);
        block_c[m][0] += vllm::gptq_rdna2::dot22_8_f(dq[0], a_ptr);
        block_c[m][1] += vllm::gptq_rdna2::dot22_8_f(dq[1], a_ptr);
        block_c[m][2] += vllm::gptq_rdna2::dot22_8_f(dq[2], a_ptr);
        block_c[m][3] += vllm::gptq_rdna2::dot22_8_f(dq[3], a_ptr);
      }
    }
    __syncthreads();
  }

  // --- Epilogue: apply topk_weight and atomic-add to output ---
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[token_block * BLOCK_SIZE_M + m];
    if (token_id / top_k >= size_m) continue;

    if (mul_topk_weight && topk_w != nullptr) {
      float tw = topk_w[token_id];
      #pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }

    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    half* out = c + out_row * size_n + n;
    if (n + 3 < size_n) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                  __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                  __float2half_rn(block_c[m][3]));
      vllm::gptq_rdna2::atomic_add_pk4_f16(out, r01, r23);
    }
  }
}

template <int BLOCK_SIZE_M>
void launch_w8a16_moe(
    const half* a, half* c, const int8_t* b_q,
    const half* b_scales, const half* b_zeros,
    const float* topk_w, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k,
    int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  const int n_blocks = (size_n + THREADS * 4 - 1) / (THREADS * 4);
  const int k_tiles = (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE;
  dim3 grid(num_token_blocks, n_blocks, k_tiles);
  dim3 block(THREADS);
  moe_w8a16_gemm_kernel<BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q, b_scales, b_zeros, topk_w,
      sorted_token_ids, expert_ids, num_tokens_post_padded,
      num_token_blocks, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, output_topk);
}

void dispatch_moe_w8a16(
    const half* a, half* c, const int8_t* b_q,
    const half* b_scales, const half* b_zeros,
    const float* topk_w, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k,
    int groups, int top_k, int block_size_m, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  switch (block_size_m) {
    case 1:
      launch_w8a16_moe<1>(a, c, b_q, b_scales, b_zeros, topk_w,
          sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, groups, top_k,
          expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 2:
      launch_w8a16_moe<2>(a, c, b_q, b_scales, b_zeros, topk_w,
          sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, groups, top_k,
          expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 4:
      launch_w8a16_moe<4>(a, c, b_q, b_scales, b_zeros, topk_w,
          sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, groups, top_k,
          expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 8:
      launch_w8a16_moe<8>(a, c, b_q, b_scales, b_zeros, topk_w,
          sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, groups, top_k,
          expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk, stream);
      break;
    default:
      TORCH_CHECK(false, "moe_w8a16_gemm_rdna2: block_size_m must be in "
                  "{1, 2, 4, 8}; got ", block_size_m);
  }
}

}  // namespace w8a16_rdna2
}  // namespace vllm

void moe_w8a16_gemm_rdna2(
    torch::Tensor a, torch::Tensor c,
    torch::Tensor b_q_weight, torch::Tensor b_scales,
    torch::Tensor b_qzeros, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA/HIP");
  TORCH_CHECK(c.is_cuda(), "c must be CUDA/HIP");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be CUDA/HIP");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be CUDA/HIP");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K, N] (int8)");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3 || b_qzeros.numel() == 0,
              "b_qzeros must be 3D [E, groups, N] or empty");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "moe_w8a16_gemm_rdna2 only supports fp16 activations");
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kInt8,
              "b_q_weight must be int8");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");
  TORCH_CHECK(b_qzeros.numel() == 0 ||
              b_qzeros.scalar_type() == b_scales.scalar_type(),
              "b_qzeros dtype must match b_scales");

  TORCH_CHECK(topk_weights.numel() == 0 ||
              topk_weights.scalar_type() == torch::kFloat32,
              "topk_weights must be fp32 or empty");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales.size(1);

  TORCH_CHECK(size_k % 8 == 0,
              "moe_w8a16_gemm_rdna2: size_k must be a multiple of 8; got ",
              size_k);
  TORCH_CHECK(size_n % 4 == 0,
              "moe_w8a16_gemm_rdna2: size_n must be a multiple of 4; got ",
              size_n);
  TORCH_CHECK(groups > 0,
              "moe_w8a16_gemm_rdna2: groups must be > 0");
  TORCH_CHECK(size_k % groups == 0,
              "moe_w8a16_gemm_rdna2: size_k (", size_k,
              ") must be divisible by groups (", groups, ")");
  const int group_size = size_k / groups;
  TORCH_CHECK(group_size % 8 == 0,
              "moe_w8a16_gemm_rdna2: group_size must be a multiple of 8; got ",
              group_size);

  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = (b_qzeros.numel() > 0)
      ? (int)(b_qzeros.size(1) * b_qzeros.size(2))
      : 0;

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;
  // data_ptr<at::Half>() returns c10::Half*, cast to __half* (alias
  // for the AMD half type). at::Half and __half are layout-compatible
  // (both are 16-bit IEEE-754 fp16).
  const half* zeros_ptr =
      (b_qzeros.numel() > 0)
          ? reinterpret_cast<const half*>(b_qzeros.data_ptr<at::Half>())
          : nullptr;

  vllm::w8a16_rdna2::dispatch_moe_w8a16(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      (const int8_t*)b_q_weight.data_ptr(),
      (const half*)b_scales.data_ptr(), zeros_ptr,
      topk_w_ptr,
      sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
      num_tokens_post_padded.data_ptr<int>(),
      num_token_blocks, size_m, size_n, size_k, groups, (int)top_k,
      (int)block_size_m, expert_weight_stride, expert_scales_stride,
      expert_zeros_stride, mul_topk_weight, (int)output_topk, stream);
}
