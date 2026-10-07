// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A4 MXFP4 kernel for RDNA2 (gfx1030), fp16-only.
//
// Mirrors the W4A16 INT4 fused-MoE kernel (csrc/rocm/moe_q_gemm_rdna2.cu)
// one-to-one. Same block geometry, same V_DOT2 compute path, same atomic-
// add pattern — only the dequant changes (E2M1 LUT + UE8M0 scale instead
// of INT4 bit-trick + per-group zero).
//
// Quantization format (DeepSeek V4 native):
//   - b_q_weight [E, K/8, N]   uint32 (8 E2M1 nibbles packed LSB-first)
//   - b_scales   [E, K/32, N]  uint8  (UE8M0, one per 32-element K-block,
//                                    per N column — no zero point)
//
// Compute: a [M*topk, K] fp16 -> c [M*topk, N] fp16 (pre-zeroed).
//
// Scale refresh frequency is higher than INT4 W4A16 (every 32 K elements
// vs every group_size=128) but no `refresh_group` is needed — one UE8M0
// byte per iter, converted to a single fp16 scale that broadcasts across
// 4 N columns.
//
// Shared helpers come from mxfp4_dot2_common.cuh:
//   - e2m1_lut[16]      (constant memory E2M1 -> fp16 LUT)
//   - ue8m0_to_fp16     (power-of-two scale -> fp16)
//   - dot22_8_f         (4x v_dot2_f32_f16 covering 8 K positions)
//   - atomic_add_pk4_f16 (64-bit CAS-loop fp16 atomic-add for 4 columns)
//   - dequant_e2m1_8_fp16 (8 E2M1 nibbles -> 4 fp16 pairs, scaled)
//   - tzero<T>()
//
// Scope: decode only. block_size_m ∈ {1, 2, 4, 8} covers M ∈ [1, 15] in
// tiles, matching the INT4 W4A16 MoE kernel. gfx1030 has no WMMA ISA, so
// a prefill MoE kernel on gfx1030 would be a scalar V_DOT2 design (out
// of scope; falls through to upstream).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "mxfp4_dot2_common.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Fused MoE MXFP4 kernel.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna2(
    const T* __restrict__ a,                  // [size_m, size_k] or [M*topk, K]
    T* __restrict__ c,                        // [M*topk, size_n] pre-zeroed
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] packed E2M1
    const uint8_t* __restrict__ b_scales,     // [E, K/32, N] UE8M0
    const float* __restrict__ topk_weights,   // [M*topk] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m,  // total tokens (original M, or M*topk for w2)
    const int size_n,  // output features per expert
    const int size_k,  // input features
    const int top_k,   // routing top-k (1 for w2 pass)
    // Per-expert strides (in elements, not bytes)
    const int expert_weight_stride,  // (K/8) * N
    const int expert_scales_stride,  // (K/32) * N
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
  const uint8_t* expert_scales_u8 =
      b_scales + (int64_t)expert_id * expert_scales_stride;

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
        av = vllm::mxfp4_dot2::tzero<T>();
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  // MXFP4 scale layout: [K/32, N] — each K-block of 32 elements has 1 scale
  // per N column. The inner K-loop advances 32 elements at a time (4 weight
  // words × 8 nibbles = 32 K), so the scale refresh lines up exactly with
  // the weight iter — no need for the multi-group refresh_group() used by
  // W4A16 INT4. We just load 4 scale bytes (one per N column) per iter.
  //
  // qk indexes the packed weight (K/8 per row); sk indexes the scales
  // (K/32 per row). Same offset_k → qk = offset_k/8, sk = offset_k/32.
  int qk = offset_k / 8;
  int sk = offset_k / 32;
  const uint32_t* b_ptr = expert_weights + (int64_t)qk * size_n + n;
  const uint8_t* s_ptr = expert_scales_u8 + (int64_t)sk * size_n + n;

  float block_c[BLOCK_SIZE_M][4];
#pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // --- Main K-loop: 32 elements per iter (4 weight words × 8 E2M1) ---
  int k = offset_k;
  while (k < end_k) {
    // Load 4 scale bytes for the 4 N columns handled by this thread.
    // Convert each to fp16 and pack into 2 half2s that broadcast across
    // j-iterations (each scale applies to all 8 K elements of one weight
    // word, but the scale is per-32-block so it applies to the full iter).
    half s0 = vllm::mxfp4_dot2::ue8m0_to_fp16(s_ptr[0]);
    half s1 = vllm::mxfp4_dot2::ue8m0_to_fp16(s_ptr[1]);
    half s2 = vllm::mxfp4_dot2::ue8m0_to_fp16(s_ptr[2]);
    half s3 = vllm::mxfp4_dot2::ue8m0_to_fp16(s_ptr[3]);
    // scale01 covers N columns [n+0, n+1], scale23 covers [n+2, n+3]
    half2 scale01 = __halves2half2(s0, s1);
    half2 scale23 = __halves2half2(s2, s3);
    s_ptr += 4 * size_n;  // advance 4 N columns' worth of scales

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

      // fp16 path: E2M1 LUT dequant, dot via v_dot2_f32_f16
      half2 dq[4];
      vllm::mxfp4_dot2::dequant_e2m1_8_fp16((uint32_t)b_w[j].x, scale01, dq);
      // dq holds 4 half2 pairs (8 fp16 values) for column pair (n+0, n+1)
      // We re-use them across 4 m rows below.
#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr =
            reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += vllm::mxfp4_dot2::dot22_8_f(dq, a_ptr);
      }

      half2 dq2[4];
      vllm::mxfp4_dot2::dequant_e2m1_8_fp16((uint32_t)b_w[j].y, scale01, dq2);
#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr =
            reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][1] += vllm::mxfp4_dot2::dot22_8_f(dq2, a_ptr);
      }

      half2 dq3[4];
      vllm::mxfp4_dot2::dequant_e2m1_8_fp16((uint32_t)b_w[j].z, scale23, dq3);
#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr =
            reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][2] += vllm::mxfp4_dot2::dot22_8_f(dq3, a_ptr);
      }

      half2 dq4[4];
      vllm::mxfp4_dot2::dequant_e2m1_8_fp16((uint32_t)b_w[j].w, scale23, dq4);
#pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr =
            reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][3] += vllm::mxfp4_dot2::dot22_8_f(dq4, a_ptr);
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
    vllm::mxfp4_dot2::atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2: empty stub for symbol parity

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna2(
    const T*, T*, const uint32_t*, const uint8_t*, const float*, const int32_t*,
    const int32_t*, const int32_t*, const int, const int, const int, const int,
    const int, const int, const bool, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
void launch_moe_gemm_mxfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  moe_gemm_mxfp4_kernel_rdna2<T, BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_padded, size_m, size_n, size_k, top_k,
      expert_weight_stride, expert_scales_stride, mul_topk_weight,
      output_topk);
}

template <typename T>
void dispatch_moe_gemm_mxfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int block_size_m, int expert_weight_stride, int expert_scales_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  switch (block_size_m) {
    case 1:
      launch_moe_gemm_mxfp4<T, 1>(
          a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 2:
      launch_moe_gemm_mxfp4<T, 2>(
          a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 4:
      launch_moe_gemm_mxfp4<T, 4>(
          a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, stream);
      break;
    case 8:
      launch_moe_gemm_mxfp4<T, 8>(
          a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_mxfp4_gemm_rdna2: block_size_m must be 1, 2, 4, or 8, "
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
//   b_q_weight             [E, K/8, N]              uint32 (8 E2M1 nibbles)
//   b_scales               [E, K/32, N]             uint8 (UE8M0)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]              int32
//   num_tokens_post_padded [1]                       int32
//   top_k                  int
//   block_size_m           int (1, 2, 4, or 8)
//   mul_topk_weight        bool
//   output_topk            int (0 or top_k for fused moe_sum)

void moe_mxfp4_gemm_rdna2(
    torch::Tensor a, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_topk_weight, int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  // b_q_weight: viewed as [E, K/8, N] uint32 OR as [E, K/2, N] uint8 —
  // both are byte-equivalent. Accept either rank/dtype by accepting the
  // uint8 byte view internally and treating the storage as uint32.
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, K/32, N]");
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kInt32 ||
                  b_q_weight.scalar_type() == torch::kUInt32 ||
                  b_q_weight.scalar_type() == torch::kUInt8 ||
                  b_q_weight.scalar_type() == torch::kInt8,
              "b_q_weight must be uint8, int8, uint32, or int32 "
              "(byte-equivalent views of 8 packed E2M1 nibbles)");
  TORCH_CHECK(b_scales.scalar_type() == torch::kUInt8,
              "b_scales must be uint8 (UE8M0 power-of-two exponent)");
  // gfx1030 lacks native BF16 hardware; the RDNA2 W4A4 MXFP4 kernel uses
  // __builtin_amdgcn_fdot2 (v_dot2_f32_f16) which is fp16-only. Force fp16.
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be half");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  // K/8 elements * N must produce a multiple of size_n's N stride; sanity
  // check that b_q_weight's K-dim is consistent with a's K.
  int qk = (int)b_q_weight.size(1);
  TORCH_CHECK(qk * 8 == size_k,
              "b_q_weight K-dim (", qk,
              ") * 8 must equal a.size(1) (=", size_k, ")");
  int sk = (int)b_scales.size(1);
  TORCH_CHECK(sk * 32 == size_k,
              "b_scales K-dim (", sk,
              ") * 32 must equal a.size(1) (=", size_k, ")");
  TORCH_CHECK(size_n % 8 == 0,
              "N must be a multiple of 8 (64-bit atomic CAS alignment)");

  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  TORCH_CHECK(topk_weights.numel() == 0 ||
              topk_weights.scalar_type() == torch::kFloat32,
              "topk_weights must be fp32 or empty");

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  // b_q_weight is byte-equivalent between [E, K/8, N] uint32 and
  // [E, K/2, N] uint8. Both forms point at the same first byte. Treat it
  // as uint8 storage for the data_ptr cast — the kernel reads it as
  // uint32 (which is byte-equivalent on a little-endian GPU).
  const uint8_t* b_qw_bytes = b_q_weight.data_ptr<uint8_t>();

  if (a.scalar_type() == torch::kHalf) {
    dispatch_moe_gemm_mxfp4<half>(
        (const half*)a.data_ptr(), (half*)c.data_ptr(),
        reinterpret_cast<const uint32_t*>(b_qw_bytes),
        (const uint8_t*)b_scales.data_ptr(), topk_w_ptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
        size_n, size_k, (int)top_k, (int)block_size_m, expert_weight_stride,
        expert_scales_stride, mul_topk_weight, (int)output_topk, stream);
  } else {
    TORCH_CHECK(false, "moe_mxfp4_gemm_rdna2 only supports fp16; got dtype=",
                a.scalar_type());
  }
}
