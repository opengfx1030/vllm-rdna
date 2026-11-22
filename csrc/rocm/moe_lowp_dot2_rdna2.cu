// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A16 / W8A16 kernels for RDNA2 (gfx1030), fp16 activations.
//
// Two block-scaled formats share the skeleton of moe_mxfp4_gemm_rdna2 (same
// block geometry, same V_DOT2 compute path, same atomic-add epilogue):
//
//   NVFP4 (modelopt; Inkling-2-Mini layers 0-19)
//     b_q_weight  [E, K/8, N]  uint32    8 E2M1 nibbles, LSB-first
//     b_scales    [E, K/16, N] fp8_e4m3  one scale per 16 K elements
//     b_globals   [E]          float32   per-expert amax / 2688
//     value = e2m1(nibble) * e4m3(scale) * global
//
//   MXFP8 (layers 20-23)
//     b_q_weight  [E, K/4, N]  uint32    4 E4M3 bytes (one byte per element)
//     b_scales    [E, K/32, N] uint8     UE8M0, one per 32 K elements
//     value = e4m3(byte) * 2^(scale - 127)
//
// Both dequantise on the fly into the V_DOT2 accumulator, so no weight wider
// than 8 bits is ever materialised.  Scope: decode-sized tiles
// (block_size_m 1/2/4/8), exactly like the MXFP4 kernel.
//
// The checkpoint stores N-first tensors ([E, N, K/2] / [E, N, K]); the Python
// wrapper repacks them into the K-first word layout indexed here.

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

namespace lowp_dot2 {

using vllm::mxfp4_dot2::atomic_add_pk4_f16;
using vllm::mxfp4_dot2::dequant_e2m1_8_fp16;
using vllm::mxfp4_dot2::dot22_8_f;
using vllm::mxfp4_dot2::tzero;

// E4M3 (e4m3fn) byte -> float, exact: normals are (1+m/8)*2^(e-7) and the
// subnormal range is m*2^-9.  gfx1030 has no fp8 conversion instruction.
__device__ __forceinline__ float e4m3_to_f32(unsigned int b) {
  const unsigned int s = (b >> 7) & 1u;
  const unsigned int e = (b >> 3) & 0xFu;
  const unsigned int m = b & 0x7u;
  const float v = (e == 0) ? (static_cast<float>(m) * 0.001953125f)  // 2^-9
                           : __uint_as_float(((e + 120u) << 23) | (m << 20));
  return s ? -v : v;
}

__device__ __forceinline__ float ue8m0_to_f32(unsigned int b) {
  return __uint_as_float(b << 23);  // 2^(b - 127)
}

// epilogue shared by both formats: router weight, optional top-k reduction,
// packed-fp16 CAS atomic add into the pre-zeroed output.
template <typename T, int BLOCK_SIZE_M>
__device__ __forceinline__ void write_out(
    T* __restrict__ c, const float (&block_c)[BLOCK_SIZE_M][4],
    const int32_t* __restrict__ sorted_token_ids,
    const float* __restrict__ topk_weights, int offset_m_base, int size_m,
    int top_k, int size_n, int n, bool mul_topk_weight, int output_topk) {
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;
    float r[4] = {block_c[m][0], block_c[m][1], block_c[m][2], block_c[m][3]};
    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) r[j] *= tw;
    }
    const int64_t out_row = (output_topk > 0)
                                ? (int64_t)(token_id / output_topk)
                                : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    const half2 r01 =
        __halves2half2(__float2half_rn(r[0]), __float2half_rn(r[1]));
    const half2 r23 =
        __halves2half2(__float2half_rn(r[2]), __float2half_rn(r[3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

// Loads the activation tile, returns false when the thread has no columns.
template <typename T, int BLOCK_SIZE_M>
__device__ __forceinline__ void load_activations(
    const T* __restrict__ a, T (&block_a)[BLOCK_SIZE_M][BLOCK_KN_SIZE + 8],
    const int32_t* __restrict__ sorted_token_ids, int offset_m_base,
    int offset_k, int end_k, int t, int size_m, int size_k, int top_k) {
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      const int32_t token_id = sorted_token_ids[offset_m_base + m];
      const int token_row = token_id / top_k;
      block_a[m][t] = (token_row < size_m)
                          ? a[(int64_t)token_row * size_k + offset_k + t]
                          : tzero<T>();
    }
  }
}

// ---------------------------------------------------------------------------
// NVFP4
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_nvfp4_kernel_rdna2(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N]
    const uint8_t* __restrict__ b_scales,     // [E, K/16, N] e4m3
    const float* __restrict__ b_globals,      // [E]
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * 4;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const uint8_t* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const float global = b_globals[expert_id];

  __shared__ T block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + 8];
  const int offset_m_base = token_block * BLOCK_SIZE_M;
  load_activations<T, BLOCK_SIZE_M>(a, block_a, sorted_token_ids, offset_m_base,
                                    offset_k, end_k, t, size_m, size_k, top_k);
  __syncthreads();
  if (n >= size_n) return;

  // 32 K per iter = 4 words per column (2 words per 16-K scale block).
  int qk = offset_k / 8;
  int sk = offset_k / 16;
  const uint32_t* b_ptr = expert_weights + (int64_t)qk * size_n + n;
  const uint8_t* s_ptr = expert_scales + (int64_t)sk * size_n + n;

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    // Two 16-K scale rows x 4 columns.
    const uint8_t* s0 = s_ptr;
    const uint8_t* s1 = (k + 16 < size_k) ? s_ptr + size_n : s_ptr;
    half2 sc[4][2];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      sc[j][0] = __float2half2_rn(e4m3_to_f32(s0[j]) * global * 16384.0f);
      sc[j][1] = __float2half2_rn(e4m3_to_f32(s1[j]) * global * 16384.0f);
    }
    s_ptr += 2 * size_n;

    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) b_w[j] = *(const int4*)(b_ptr + j * size_n);
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      // One scale per 16 K: words 0,1 use block 0, words 2,3 block 1.
      const half2 s_col0 = sc[0][j / 2], s_col1 = sc[1][j / 2];
      const half2 s_col2 = sc[2][j / 2], s_col3 = sc[3][j / 2];
      half2 dq[4];
      dequant_e2m1_8_fp16((uint32_t)b_w[j].x, s_col0, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][0] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16((uint32_t)b_w[j].y, s_col1, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][1] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16((uint32_t)b_w[j].z, s_col2, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][2] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16((uint32_t)b_w[j].w, s_col3, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][3] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
    }
    k += 32;
  }

  write_out<T, BLOCK_SIZE_M>(c, block_c, sorted_token_ids, topk_weights,
                             offset_m_base, size_m, top_k, size_n, n,
                             mul_topk_weight, output_topk);
}

}  // namespace lowp_dot2

#else  // non-RDNA2: empty stubs for symbol parity

namespace lowp_dot2 {

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_nvfp4_kernel_rdna2(
    const T*, T*, const uint32_t*, const uint8_t*, const float*, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const bool, const int) {}

}  // namespace lowp_dot2

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
void launch_moe_nvfp4(const T* a, T* c, const uint32_t* b_q_weight,
                      const uint8_t* b_scales, const float* b_globals,
                      const float* topk_weights,
                      const int32_t* sorted_token_ids,
                      const int32_t* expert_ids,
                      const int32_t* num_tokens_post_padded,
                      int num_token_blocks, int size_m, int size_n, int size_k,
                      int top_k, int expert_weight_stride,
                      int expert_scales_stride, bool mul_topk_weight,
                      int output_topk, hipStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);
  lowp_dot2::moe_gemm_nvfp4_kernel_rdna2<T, BLOCK_SIZE_M>
      <<<grid, block, 0, stream>>>(
          a, c, b_q_weight, b_scales, b_globals, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, top_k,
          expert_weight_stride, expert_scales_stride, mul_topk_weight,
          output_topk);
}

template <typename T>
void dispatch_moe_nvfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales,
    const float* b_globals, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int top_k, int block_size_m,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, hipStream_t stream) {
#define LAUNCH(BM)                                                           \
  launch_moe_nvfp4<T, BM>(                                                   \
      a, c, b_q_weight, b_scales, b_globals, topk_weights, sorted_token_ids, \
      expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,  \
      size_k, top_k, expert_weight_stride, expert_scales_stride,             \
      mul_topk_weight, output_topk, stream)
  switch (block_size_m) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    default:
      TORCH_CHECK(
          false,
          "moe_nvfp4_gemm_rdna2: block_size_m must be 1, 2, 4 or 8, got ",
          block_size_m);
  }
#undef LAUNCH
}

// ---------------------------------------------------------------------------
// Public entry point (NVFP4)
// ---------------------------------------------------------------------------
void moe_nvfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_globals, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight must be contiguous");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, K/16, N]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(2);
  TORCH_CHECK((int)b_q_weight.size(1) * 8 == size_k,
              "b_q_weight K-dim * 8 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(1) * 16 == size_k,
              "b_scales K-dim * 16 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(2) == size_n, "b_scales N-dim must equal N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(size_k % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(b_globals.numel() == (int64_t)b_q_weight.size(0),
              "b_globals must hold one scale per expert");

  const int expert_weight_stride = (int)(b_q_weight.size(1) * size_n);
  const int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  dispatch_moe_nvfp4<half>(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      (const uint8_t*)b_scales.data_ptr(), b_globals.data_ptr<float>(),
      topk_w_ptr, sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
      size_n, size_k, (int)top_k, (int)block_size_m, expert_weight_stride,
      expert_scales_stride, mul_topk_weight, (int)output_topk, stream);
}

// ---------------------------------------------------------------------------
// MXFP8
// ---------------------------------------------------------------------------
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

namespace lowp_dot2 {

// 4 E4M3 bytes -> 2 half2, scaled by a power of two (exact in fp16).
__device__ __forceinline__ void deq_e4m3_4(uint32_t w, float sc, half2* dq) {
  dq[0] = __halves2half2(__float2half_rn(e4m3_to_f32((w >> 0) & 0xFFu) * sc),
                         __float2half_rn(e4m3_to_f32((w >> 8) & 0xFFu) * sc));
  dq[1] = __halves2half2(__float2half_rn(e4m3_to_f32((w >> 16) & 0xFFu) * sc),
                         __float2half_rn(e4m3_to_f32((w >> 24) & 0xFFu) * sc));
}

// One thread per 32 K elements of one N row: 8 words per column, 1 scale byte.
template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp8_kernel_rdna2(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N]
    const uint8_t* __restrict__ b_scales,     // [E, K/32, N] ue8m0
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * 4;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const uint8_t* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;

  __shared__ T block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + 8];
  const int offset_m_base = token_block * BLOCK_SIZE_M;
  load_activations<T, BLOCK_SIZE_M>(a, block_a, sorted_token_ids, offset_m_base,
                                    offset_k, end_k, t, size_m, size_k, top_k);
  __syncthreads();
  if (n >= size_n) return;

  const uint32_t* b_ptr = expert_weights + (int64_t)(offset_k / 4) * size_n + n;
  const uint8_t* s_ptr = expert_scales + (int64_t)(offset_k / 32) * size_n + n;

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    float sc[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) sc[j] = ue8m0_to_f32(s_ptr[j]);
    s_ptr += size_n;

    // K-major packing [E, K/4, N]: 8 words per column per 32 K, i.e. two
    // words per 8-K group.  One int4 load per K-word spans 4 N columns.
    uint32_t w[8][4];  // w[k_word][column]
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int4 v = *(const int4*)(b_ptr + i * size_n);
      w[i][0] = (uint32_t)v.x;
      w[i][1] = (uint32_t)v.y;
      w[i][2] = (uint32_t)v.z;
      w[i][3] = (uint32_t)v.w;
    }
    b_ptr += 8 * size_n;

  #pragma unroll
    for (int g = 0; g < 4; ++g) {  // 8 K per group
      const int a_off = (k - offset_k) + 8 * g;
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        half2 dq[4];
        deq_e4m3_4(w[2 * g + 0][j], sc[j], dq + 0);
        deq_e4m3_4(w[2 * g + 1][j], sc[j], dq + 2);
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m)
          block_c[m][j] +=
              dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      }
    }
    k += 32;
  }

  write_out<T, BLOCK_SIZE_M>(c, block_c, sorted_token_ids, topk_weights,
                             offset_m_base, size_m, top_k, size_n, n,
                             mul_topk_weight, output_topk);
}

}  // namespace lowp_dot2

#else  // non-RDNA2 stubs

namespace lowp_dot2 {

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp8_kernel_rdna2(
    const T*, T*, const uint32_t*, const uint8_t*, const float*, const int32_t*,
    const int32_t*, const int32_t*, const int, const int, const int, const int,
    const int, const int, const bool, const int) {}

}  // namespace lowp_dot2

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// MXFP8 launcher and entry point
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE_M>
void launch_moe_mxfp8(const T* a, T* c, const uint32_t* b_q_weight,
                      const uint8_t* b_scales, const float* topk_weights,
                      const int32_t* sorted_token_ids,
                      const int32_t* expert_ids,
                      const int32_t* num_tokens_post_padded,
                      int num_token_blocks, int size_m, int size_n, int size_k,
                      int top_k, int expert_weight_stride,
                      int expert_scales_stride, bool mul_topk_weight,
                      int output_topk, hipStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);
  lowp_dot2::moe_gemm_mxfp8_kernel_rdna2<T, BLOCK_SIZE_M>
      <<<grid, block, 0, stream>>>(
          a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, top_k,
          expert_weight_stride, expert_scales_stride, mul_topk_weight,
          output_topk);
}

template <typename T>
void dispatch_moe_mxfp8(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int block_size_m, int expert_weight_stride, int expert_scales_stride,
    bool mul_topk_weight, int output_topk, hipStream_t stream) {
#define LAUNCH_MX8(BM)                                                         \
  launch_moe_mxfp8<T, BM>(                                                     \
      a, c, b_q_weight, b_scales, topk_weights, sorted_token_ids, expert_ids,  \
      num_tokens_post_padded, num_token_blocks, size_m, size_n, size_k, top_k, \
      expert_weight_stride, expert_scales_stride, mul_topk_weight,             \
      output_topk, stream)
  switch (block_size_m) {
    case 1:
      LAUNCH_MX8(1);
      break;
    case 2:
      LAUNCH_MX8(2);
      break;
    case 4:
      LAUNCH_MX8(4);
      break;
    case 8:
      LAUNCH_MX8(8);
      break;
    default:
      TORCH_CHECK(
          false,
          "moe_mxfp8_gemm_rdna2: block_size_m must be 1, 2, 4 or 8, got ",
          block_size_m);
  }
#undef LAUNCH_MX8
}

void moe_mxfp8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/4, N]");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight must be contiguous");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, K/32, N]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(2);
  TORCH_CHECK((int)b_q_weight.size(1) * 4 == size_k,
              "b_q_weight K-dim * 4 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(1) * 32 == size_k,
              "b_scales K-dim * 32 must equal a.size(1)");
  TORCH_CHECK((int)b_scales.size(2) == size_n, "b_scales N-dim must equal N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(size_k % 32 == 0, "K must be a multiple of 32");

  const int expert_weight_stride = (int)(b_q_weight.size(1) * size_n);
  const int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  dispatch_moe_mxfp8<half>(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      (const uint8_t*)b_scales.data_ptr(), topk_w_ptr,
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
      size_n, size_k, (int)top_k, (int)block_size_m, expert_weight_stride,
      expert_scales_stride, mul_topk_weight, (int)output_topk, stream);
}
