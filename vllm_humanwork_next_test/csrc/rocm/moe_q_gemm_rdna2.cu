// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A16 GPTQ kernel for RDNA2 (gfx1030), fp16-only.
//
// Combines expert routing (sorted_token_ids / expert_ids) with the RDNA2
// W4A16 dequant+dot from q_gemm_rdna2.cu into a single kernel launch.
// Each block processes BLOCK_SIZE_M tokens assigned to one expert, covering
// a tile of N output columns and K input positions.
//
// Weight format: same as the dense kernel — [E, K/8, N] uint32 shuffled,
// [E, groups, N] scales, [E, groups, N/8] packed zeros.
//
// Scope notes:
//   1. Decode only. block_size_m ∈ {1, 2, 4, 8} covers M ∈ [1, 15] in
//      tiles; the larger MoE block sizes used in upstream vLLM
//      (M ∈ {16, 32, 64, 128}) are out of scope and fall through to
//      the upstream MoE path. gfx1030 has no WMMA ISA, so a prefill
//      MoE kernel on gfx1030 would also be a scalar V_DOT2 design.
//   2. Packed atomic output via CAS-loop on a 64-bit word
//      (atomic_add_pk4_f16). gfx1030 has no native
//      v_global_atomic_pk_add_f16 (that landed on gfx940). The kernel
//      emulates packed atomic add with global_atomic_cmpswap_b64 plus
//      retry.
//
// Design: THREADS_X=256 (8 waves on wave32), BLOCK_KN_SIZE=256, each thread
// handles 4 N columns. fp16 uses v_dot2_f32_f16 (__builtin_amdgcn_fdot2)
// directly. Output via 64-bit packed CAS atomic-add directly to the
// pre-zeroed output tensor (no FP32 scratch buffer).
//
// Shared helpers (tzero, dot22_8_f, atomic_add_pk4_f16, load4_zeros,
// load4_scales, refresh_group, epilogue, prep_zero_scale_fp16,
// dequant_4bit_8_fp16) come from q_gemm_rdna2_common.cuh and
// qdq_4_rdna2.cuh in the vllm::gptq_rdna2:: namespace.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"
#include "qdq_4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Fused MoE kernel.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_q4_kernel_rdna2(
    const T* __restrict__ a,                  // [size_m, size_k] or [M*topk, K]
    T* __restrict__ c,                        // [M*topk, size_n] pre-zeroed
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] packed
    const T* __restrict__ b_scales,           // [E, groups, N]
    const uint32_t* __restrict__ b_qzeros,    // [E, groups, N/8] packed
    const float* __restrict__ topk_weights,   // [M*topk] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m,  // total tokens (original M, or M*topk for w2)
    const int size_n,  // output features per expert
    const int size_k,  // input features
    const int groups,  // K / group_size
    const int top_k,   // routing top-k (1 for w2 pass)
    // Per-expert strides (in elements, not bytes)
    const int expert_weight_stride,  // (K/8) * N
    const int expert_scales_stride,  // groups * N
    const int expert_zeros_stride,   // groups * (N/8)
    const bool mul_topk_weight,
    const int output_topk) {  // >0: reduce output by token_id/output_topk
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * 4;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // Early exit for padding blocks or invalid experts (expert_map = -1)
  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;

  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  // Expert-specific pointers
  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const T* expert_scales = b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  // LDS for activations
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X");

  const int offset_m_base = token_block * BLOCK_SIZE_M;

  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      int32_t token_id = sorted_token_ids[offset_m_base + m];
      int token_row = token_id / top_k;
      T av;
      if (token_row < size_m) {
        av = a[(int64_t)token_row * size_k + offset_k + t];
      } else {
        av = vllm::gptq_rdna2::tzero<T>();
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  // Group bookkeeping
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Weight pointer for this expert
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  // Per-column dequant constants (4 columns per thread)
  half2 z1z16_h[4][2], y1y16_h[4][2];

  // GPTQv1: zero_offset = 1
  constexpr int zero_offset = 1;

  // Refresh dequant constants for current group + 4 columns at n
  vllm::gptq_rdna2::refresh_group<4>(group, n, expert_qzeros,
                                     reinterpret_cast<const half*>(expert_scales),
                                     size_n, zero_offset, z1z16_h, y1y16_h);

  float block_c[BLOCK_SIZE_M][4];
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // --- Main K-loop ---
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      vllm::gptq_rdna2::refresh_group<4>(group, n, expert_qzeros,
                                         reinterpret_cast<const half*>(expert_scales),
                                         size_n, zero_offset, z1z16_h, y1y16_h);
    }

    // Prefetch 4 weight words (128 bytes)
    int4 b_w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      // fp16 path: dequant via bit-trick, dot via v_dot2_f32_f16
      half2 dq[4][4];
      vllm::gptq_rdna2::dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0],
                                           y1y16_h[0]);
      vllm::gptq_rdna2::dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1],
                                           y1y16_h[1]);
      vllm::gptq_rdna2::dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2],
                                           y1y16_h[2]);
      vllm::gptq_rdna2::dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3],
                                           y1y16_h[3]);

#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += vllm::gptq_rdna2::dot22_8_f(dq[0], a_ptr);
        block_c[m][1] += vllm::gptq_rdna2::dot22_8_f(dq[1], a_ptr);
        block_c[m][2] += vllm::gptq_rdna2::dot22_8_f(dq[2], a_ptr);
        block_c[m][3] += vllm::gptq_rdna2::dot22_8_f(dq[3], a_ptr);
      }
    }
    k += 32;
  }

  // --- Epilogue: apply topk_weight and atomic-add to output ---
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    // Apply router weight
    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
#pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }

    // output_topk > 0: reduce by mapping token_id back to original token
    // (multiple experts write to the same row via atomics)
    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    vllm::gptq_rdna2::atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2: empty stub for symbol parity

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_q4_kernel_rdna2(
    const T*, T*, const uint32_t*, const T*, const uint32_t*, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const int, const int,
    const bool, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
void launch_moe_gemm_q4(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  moe_gemm_q4_kernel_rdna2<T, BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, output_topk);
}

template <typename T>
void dispatch_moe_gemm_q4(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int block_size_m,
    int expert_weight_stride, int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  switch (block_size_m) {
    case 1:
      launch_moe_gemm_q4<T, 1>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 2:
      launch_moe_gemm_q4<T, 2>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 4:
      launch_moe_gemm_q4<T, 4>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 8:
      launch_moe_gemm_q4<T, 8>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_gptq_gemm_rdna2: block_size_m must be 1, 2, 4, or 8, "
                  "got ",
                  block_size_m);
  }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
//
// Inputs:
//   a                      [M, K] or [M*top_k, K]  half
//   c                      [M*top_k, N]             half (pre-zeroed!)
//   b_q_weight             [E, K/8, N]              uint32 (shuffled)
//   b_scales               [E, groups, N]           half
//   b_qzeros               [E, groups, N/8]         uint32 (packed 4-bit)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]              int32
//   num_tokens_post_padded [1]                       int32
//   top_k                  int
//   block_size_m           int (1, 2, 4, or 8)
//   mul_topk_weight        bool

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
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
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3, "b_qzeros must be 3D [E, groups, N/8]");
  // gfx1030 lacks native BF16 hardware; the RDNA2 W4A16 kernels use
  // __builtin_amdgcn_fdot2 (v_dot2_f32_f16) which is fp16-only.
  // Force fp16 input on gfx1030.
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be half");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales.size(1);

  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = (int)(b_qzeros.size(1) * b_qzeros.size(2));

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  if (a.scalar_type() == torch::kHalf) {
    dispatch_moe_gemm_q4<half>(
        (const half*)a.data_ptr(), (half*)c.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr<int32_t>(),
        (const half*)b_scales.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr<int32_t>(), topk_w_ptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
        size_n, size_k, groups, (int)top_k, (int)block_size_m,
        expert_weight_stride, expert_scales_stride, expert_zeros_stride,
        mul_topk_weight, (int)output_topk, stream);
  } else if (a.scalar_type() == torch::kBFloat16) {
    // gfx1030 has no v_dot2_f32_bf16 (RDNA3+); fallback to fp16 dot accumulator
    // via __nv_bfloat16 -> half promotion. Path not used by production
    // (gfx1030 serves only fp16 weights); kept as compile-only fallback so
    // the op does not silently segfault on rare bf16 inputs.
    TORCH_CHECK(false, "bfloat16 path is not yet implemented for gfx1030; "
                "the W4A16 helper kernel uses fp16 v_dot2 instructions only");
  }
}
