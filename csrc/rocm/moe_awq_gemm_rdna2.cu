// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A16 AWQ kernel for RDNA2 (gfx1030).
//
// Combines expert routing (sorted_token_ids / expert_ids) with the RDNA2
// W4A16 AWQ dequant+dot from awq_gemm_rdna2.cu into a single kernel launch.
// Each block processes BLOCK_SIZE_M tokens assigned to one expert, covering
// a tile of N output columns and K input positions.
//
// Weight format: sequential [E, K, N/8] uint32, [E, groups, N/8] uint32 zeros,
// [E, groups, N] half scales.
//
// Dequant: w_fp = (nibble - zero) * scale  (no +1 zero offset for AWQ)
//
// Design: wave32, THREADS_X=256, BLOCK_KN_SIZE=256. Each thread handles 4 N
// columns. All arithmetic in fp32 (RDNA2 has no v_dot2). Output via 64-bit
// packed CAS atomic-add directly to the pre-zeroed output tensor.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "qdq_awq_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__MOE_AWQ_RDNA2__
#endif

namespace vllm {
namespace moe_awq_rdna2 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// Packed atomic-add via CAS-loop.
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

#if defined(__HIP__MOE_AWQ_RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int BLOCK_SIZE_M>
__global__ void moe_awq_gemm_q4_kernel_rdna2(
    const half* __restrict__ a,                  // [size_m, size_k] or [M*topk, K]
    half* __restrict__ c,                        // [M*topk, size_n] pre-zeroed
    const uint32_t* __restrict__ b_q_weight,     // [E, K, N/8] sequential
    const uint32_t* __restrict__ b_qzeros,       // [E, groups, N/8] packed
    const half* __restrict__ b_scales,           // [E, groups, N]
    const float* __restrict__ topk_weights,      // [M*topk] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m,    // total tokens (original M, or M*topk for w2)
    const int size_n,    // output features per expert
    const int size_k,    // input features
    const int groups,    // K / group_size
    const int top_k,     // routing top-k (1 for w2 pass)
    // Per-expert strides (in elements, not bytes)
    const int expert_weight_stride,  // K * (N/8)
    const int expert_scales_stride,  // groups * N
    const int expert_zeros_stride,   // groups * (N/8)
    const bool mul_topk_weight,
    const int output_topk) {           // >0: reduce output by token_id/output_topk

  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * 4;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // Early exit for padding blocks or invalid experts
  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;

  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  // Expert-specific pointers
  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales = b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  // LDS for activations
  constexpr int LDS_PAD = 8;
  __shared__ half block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X");

  const int offset_m_base = token_block * BLOCK_SIZE_M;

  // Stage A through LDS
  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      int32_t token_id = sorted_token_ids[offset_m_base + m];
      int token_row = token_id / top_k;
      half av;
      if (token_row < size_m) {
        av = a[(int64_t)token_row * size_k + offset_k + t];
      } else {
        av = __float2half_rn(0.0f);
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  // Per-column nibble extraction parameters
  int n_packed[4], shift[4];
#pragma unroll
  for (int col = 0; col < 4; ++col) {
    int nc = n + col;
    n_packed[col] = nc / 8;
    shift[col] = (nc & 0x07) * 4;
  }

  // Group bookkeeping
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Per-column fp32 dequant constants: w_fp32 = nibble * y_f + z_f
  float z_f[4], y_f[4];

  auto refresh_group = [&](int g) {
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      int nc = n + col;
      // Extract zero from packed qzeros row
      int qcol = nc / 8;
      int s = (nc & 0x07) * 4;
      uint32_t d = expert_qzeros[g * (size_n / 8) + qcol] >> s;
      int zero = (int)(d & 0xF);
      // Get scale for this column
      half scale = expert_scales[g * size_n + nc];
      awq_rdna2::prep_zero_scale_fp32(zero, scale, z_f[col], y_f[col]);
    }
  };

  refresh_group(group);

  float block_c[BLOCK_SIZE_M][4];
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // --- Main K-loop: process 8 K positions at a time ---
  int k = offset_k;
  while (k + 7 < end_k) {
    // Check group boundary
    if (k >= nextgroup || k + 7 >= nextgroup) {
      // Process up to boundary one-at-a-time
      while (k < nextgroup && k + 7 < end_k) {
#pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          float a_f = __half2float(block_a[m][k - offset_k]);
#pragma unroll
          for (int col = 0; col < 4; ++col) {
            uint32_t qa = expert_weights[k * (size_n / 8) + n_packed[col]];
            int nibble = (qa >> shift[col]) & 0xF;
            float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
            block_c[m][col] = __fmaf_rn(w, a_f, block_c[m][col]);
          }
        }
        k++;
      }
      if (k >= nextgroup) {
        group++;
        nextgroup += groupsize;
        refresh_group(group);
      }
      continue;
    }

    // Process 8 K positions at once with fp16->fp32 widening
    float a_f[8][BLOCK_SIZE_M];
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        half2 av = __halves2half2(block_a[m][k + i - offset_k],
                                  block_a[m][k + i + 1 - offset_k]);
        float2 f2 = __half22float2(av);
        a_f[i][m] = f2.x;
        a_f[i + 1][m] = f2.y;
      }
    }

#pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
#pragma unroll
      for (int col = 0; col < 4; ++col) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint32_t qa = expert_weights[(k + i) * (size_n / 8) + n_packed[col]];
          int nibble = (qa >> shift[col]) & 0xF;
          float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
          block_c[m][col] = __fmaf_rn(w, a_f[i][m], block_c[m][col]);
        }
      }
    }

    k += 8;
  }

  // Tail: remaining K positions
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

#pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      float a_f = __half2float(block_a[m][k - offset_k]);
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        uint32_t qa = expert_weights[k * (size_n / 8) + n_packed[col]];
        int nibble = (qa >> shift[col]) & 0xF;
        float w = __fmaf_rn((float)nibble, y_f[col], z_f[col]);
        block_c[m][col] = __fmaf_rn(w, a_f, block_c[m][col]);
      }
    }
    k++;
  }

  // --- Epilogue: apply topk_weight and atomic-add to output ---
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
#pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }

    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    half* out = c + out_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2: empty stub

template <int BLOCK_SIZE_M>
__global__ void moe_awq_gemm_q4_kernel_rdna2(
    const half*, half*, const uint32_t*, const uint32_t*, const half*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    const int, const int, const int, const int, const int, const int,
    const int, const int, const int, const bool, const int) {}

#endif  // __HIP__MOE_AWQ_RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int BLOCK_SIZE_M>
void launch_moe_awq_gemm_q4(
    const half* a, half* c, const uint32_t* b_q_weight,
    const uint32_t* b_qzeros, const half* b_scales,
    const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  moe_awq_gemm_q4_kernel_rdna2<BLOCK_SIZE_M>
      <<<grid, block, 0, stream>>>(
          a, c, b_q_weight, b_qzeros, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups,
          top_k, expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk);
}

void dispatch_moe_awq_gemm_q4(
    const half* a, half* c, const uint32_t* b_q_weight,
    const uint32_t* b_qzeros, const half* b_scales,
    const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int block_size_m,
    int expert_weight_stride, int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  switch (block_size_m) {
    case 1:
      launch_moe_awq_gemm_q4<1>(a, c, b_q_weight, b_qzeros, b_scales,
                                 topk_weights, sorted_token_ids, expert_ids,
                                 num_tokens_post_padded, num_token_blocks,
                                 size_m, size_n, size_k, groups, top_k,
                                 expert_weight_stride, expert_scales_stride,
                                 expert_zeros_stride, mul_topk_weight,
                                 output_topk, stream);
      break;
    case 2:
      launch_moe_awq_gemm_q4<2>(a, c, b_q_weight, b_qzeros, b_scales,
                                 topk_weights, sorted_token_ids, expert_ids,
                                 num_tokens_post_padded, num_token_blocks,
                                 size_m, size_n, size_k, groups, top_k,
                                 expert_weight_stride, expert_scales_stride,
                                 expert_zeros_stride, mul_topk_weight,
                                 output_topk, stream);
      break;
    case 4:
      launch_moe_awq_gemm_q4<4>(a, c, b_q_weight, b_qzeros, b_scales,
                                 topk_weights, sorted_token_ids, expert_ids,
                                 num_tokens_post_padded, num_token_blocks,
                                 size_m, size_n, size_k, groups, top_k,
                                 expert_weight_stride, expert_scales_stride,
                                 expert_zeros_stride, mul_topk_weight,
                                 output_topk, stream);
      break;
    case 8:
      launch_moe_awq_gemm_q4<8>(a, c, b_q_weight, b_qzeros, b_scales,
                                 topk_weights, sorted_token_ids, expert_ids,
                                 num_tokens_post_padded, num_token_blocks,
                                 size_m, size_n, size_k, groups, top_k,
                                 expert_weight_stride, expert_scales_stride,
                                 expert_zeros_stride, mul_topk_weight,
                                 output_topk, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_awq_gemm_rdna2: block_size_m must be 1, 2, 4, or 8, "
                  "got ",
                  block_size_m);
  }
}

}  // namespace moe_awq_rdna2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
//
// Inputs:
//   a                      [M, K] or [M*top_k, K]  half
//   c                      [M*top_k, N]             half (pre-zeroed!)
//   b_q_weight             [E, K, N/8]              uint32 (sequential)
//   b_scales               [E, groups, N]           half
//   b_qzeros               [E, groups, N/8]         uint32 (packed 4-bit)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]              int32
//   num_tokens_post_padded [1]                       int32
//   top_k                  int
//   block_size_m           int (1, 2, 4, or 8)
//   mul_topk_weight        bool
//   output_topk            int (>0 for output reduction)

void moe_awq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor b_qzeros, torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids,
                        torch::Tensor expert_ids,
                        torch::Tensor num_tokens_post_padded, int64_t top_k,
                        int64_t block_size_m, bool mul_topk_weight,
                        int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K, N/8]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3, "b_qzeros must be 3D [E, groups, N/8]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "a must be half (fp16)");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf,
              "b_scales must be half (fp16)");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2) * 8;
  int groups = (int)b_scales.size(1);

  // Per-expert strides
  int expert_weight_stride =
      (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = (int)(b_qzeros.size(1) * b_qzeros.size(2));

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  vllm::moe_awq_rdna2::dispatch_moe_awq_gemm_q4(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr<int32_t>(),
      (const uint32_t*)b_qzeros.data_ptr<int32_t>(),
      (const half*)b_scales.data_ptr(), topk_w_ptr,
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
      size_n, size_k, groups, (int)top_k, (int)block_size_m,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, (int)output_topk, stream);
}
