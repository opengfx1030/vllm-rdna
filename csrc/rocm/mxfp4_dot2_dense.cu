// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A4 MXFP4 dense GEMM kernel for RDNA2 (gfx1030), fp16-only.
//
// Mirrors csrc/rocm/q_gemm_rdna2.cu (W4A16 INT4) one-to-one. Same block
// geometry (THREADS_X=256, BLOCK_KN_SIZE=256, 4 N columns per thread),
// same V_DOT2_F32_F16 compute path, same atomic-add output pattern —
// only the dequant changes (E2M1 LUT + UE8M0 scale, no zero).
//
// Used for non-MoE MXFP4 layers (attention, shared experts, etc.) that
// share the same [E, K/8, N] uint32 + [E, K/32, N] uint8 weight format
// as the MoE experts. The dense kernel collapses E=1 — it does NOT
// take an E parameter and treats the weight tensor as [K/8, N].
//
// Quantization format:
//   - b_q_weight [K/8, N]  uint32 (8 E2M1 nibbles packed LSB-first)
//   - b_scales   [K/32, N] uint8 (UE8M0, one per 32-element K-block)
//
// M_COUNT ∈ {1, 2, 4, 8} covers M ∈ [1, 15] in tiles; larger M is correct
// but decode-optimized (gfx1030 has no WMMA ISA — that landed on RDNA3).
//
// Shared helpers come from mxfp4_dot2_common.cuh:
//   - e2m1_lut[16], ue8m0_to_fp16, dot22_8_f, atomic_add_pk4_f16,
//     dequant_e2m1_8_fp16, tzero<T>

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "mxfp4_dot2_common.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace mxfp4_dot2 {

// BLOCK_KN_SIZE=256 (was 128 in older exllama designs). Each block covers
// 256 K elements and THREADS_X*4 = 1024 N columns. For Qwen-class K=4096
// this halves gridDim.z (32 -> 16) and halves the atomic count per output
// position vs the exllama default. THREADS_X=256 = 8 waves on RDNA2 wave32.
#define BLOCK_KN_SIZE 256
#define THREADS_X 256

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
__global__ void gemm_mxfp4_kernel_rdna2(
    const T* __restrict__ a,                 // [size_m, size_k]
    const uint32_t* __restrict__ b_q_weight, // [size_k/8, size_n] packed E2M1
    const uint8_t* __restrict__ b_scales,    // [size_k/32, size_n] UE8M0
    T* __restrict__ c,                       // [size_m, size_n] pre-zeroed
    const int size_m, const int size_n, const int size_k) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // LDS layout: [M_COUNT][BLOCK_KN_SIZE + LDS_PAD]. PAD=8 avoids bank
  // conflicts when different M rows read the same K offset.
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A into LDS: each thread loads one K element per M row. Invalid
  // rows past size_m are zero-padded. LDS staging is required even at M=1
  // because the inner loop indexes block_a[m][a_off] unconditionally.
  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X (1 K element per thread)");
  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      T av;
      if (offset_m + m < size_m) {
        const T* a_row = a + (offset_m + m) * size_k;
        av = a_row[offset_k + t];
      } else {
        av = tzero<T>();  // zero-pad invalid M rows
      }
      block_a[m][t] = av;
    }
  }

  // Threads beyond the right edge of N have nothing to do, but ALL
  // THREADS_X (=256) threads participate in the LDS load above, so we
  // cannot return before __syncthreads().
  __syncthreads();

  if (n >= size_n) return;

  // MXFP4: scale refresh lines up exactly with the K-loop iter — every
  // 32 K elements, one scale byte per N column. No refresh_group needed.
  int qk = offset_k / 8;
  int sk = offset_k / 32;
  const uint32_t* b_ptr = b_q_weight + (int64_t)qk * size_n + n;
  const uint8_t* s_ptr = b_scales + (int64_t)sk * size_n + n;

  float block_c[M_COUNT][4];
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    // Load 4 scale bytes for the 4 N columns handled by this thread.
    half s0 = ue8m0_to_fp16(s_ptr[0]);
    half s1 = ue8m0_to_fp16(s_ptr[1]);
    half s2 = ue8m0_to_fp16(s_ptr[2]);
    half s3 = ue8m0_to_fp16(s_ptr[3]);
    half2 scale01 = __halves2half2(s0, s1);
    half2 scale23 = __halves2half2(s2, s3);
    s_ptr += 4 * size_n;

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

      half2 dq[4], dq2[4], dq3[4], dq4[4];
      dequant_e2m1_8_fp16((uint32_t)b_w[j].x, scale01, dq);
      dequant_e2m1_8_fp16((uint32_t)b_w[j].y, scale01, dq2);
      dequant_e2m1_8_fp16((uint32_t)b_w[j].z, scale23, dq3);
      dequant_e2m1_8_fp16((uint32_t)b_w[j].w, scale23, dq4);

#pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += dot22_8_f(dq, a_ptr);
        block_c[m][1] += dot22_8_f(dq2, a_ptr);
        block_c[m][2] += dot22_8_f(dq3, a_ptr);
        block_c[m][3] += dot22_8_f(dq4, a_ptr);
      }
    }
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // Pack partial sums into two half2 pairs and atomically add to the
  // pre-zeroed fp16 output (64-bit CAS-loop).
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;
    T* out = c + (int64_t)(offset_m + m) * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <typename T, int M_COUNT>
__global__ void gemm_mxfp4_kernel_rdna2(
    const T*, const uint32_t*, const uint8_t*, T*, const int, const int,
    const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_mxfp4_for_mcount(const T* a, const uint32_t* b_q_weight,
                                  const uint8_t* b_scales, T* c, int size_m,
                                  int size_n, int size_k,
                                  cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_mxfp4_kernel_rdna2<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_scales, c, size_m, size_n, size_k);
}

// Dispatch to the largest M_COUNT template that tiles size_m without
// wasting more than half the last tile. M_COUNT is capped at 8.
template <typename T>
void launch_gemm_mxfp4(const T* a, const uint32_t* b_q_weight,
                       const uint8_t* b_scales, T* c, int size_m, int size_n,
                       int size_k, cudaStream_t stream) {
  if (size_m == 1) {
    launch_gemm_mxfp4_for_mcount<T, 1>(a, b_q_weight, b_scales, c, size_m,
                                       size_n, size_k, stream);
  } else if (size_m <= 3) {
    launch_gemm_mxfp4_for_mcount<T, 2>(a, b_q_weight, b_scales, c, size_m,
                                       size_n, size_k, stream);
  } else if (size_m <= 7) {
    launch_gemm_mxfp4_for_mcount<T, 4>(a, b_q_weight, b_scales, c, size_m,
                                       size_n, size_k, stream);
  } else {
    launch_gemm_mxfp4_for_mcount<T, 8>(a, b_q_weight, b_scales, c, size_m,
                                       size_n, size_k, stream);
  }
}

}  // namespace mxfp4_dot2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a          [M, K]            half
//   b_q_weight [K/8, N]          uint32 (8 E2M1 nibbles packed LSB-first)
//                                OR uint8 byte-equivalent view
//   b_scales   [K/32, N]         uint8 (UE8M0 power-of-two exponent)
//   size_m, size_n, size_k       ints
//
// Output:
//   c          [M, N]            half (pre-zeroed; kernel atomically adds)

void mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                      torch::Tensor b_q_weight, torch::Tensor b_scales,
                      int64_t size_m, int64_t size_n, int64_t size_k) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  // Accept either uint8 ([K/2, N]) or uint32 ([K/8, N]) — byte-equivalent
  // views of the same packed E2M1 storage.
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N] uint32");
  TORCH_CHECK(b_scales.dim() == 2, "b_scales must be 2D [K/32, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "mxfp4_gemm_rdna2 only supports fp16");

  int sm = (int)size_m;
  int sk = (int)size_k;
  int sn = (int)size_n;

  // Validate strides against tensor dims
  TORCH_CHECK(b_q_weight.size(0) * 8 == sk,
              "b_q_weight K-dim (", b_q_weight.size(0),
              ") * 8 must equal size_k (=", sk, ")");
  TORCH_CHECK(b_q_weight.size(1) == sn,
              "b_q_weight N-dim (", b_q_weight.size(1),
              ") must equal size_n (=", sn, ")");
  TORCH_CHECK(b_scales.size(0) * 32 == sk,
              "b_scales K-dim (", b_scales.size(0),
              ") * 32 must equal size_k (=", sk, ")");
  TORCH_CHECK(b_scales.size(1) == sn,
              "b_scales N-dim (", b_scales.size(1),
              ") must equal size_n (=", sn, ")");
  TORCH_CHECK(sn % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");
  TORCH_CHECK(b_scales.scalar_type() == torch::kUInt8,
              "b_scales must be uint8 (UE8M0)");

  // Caller MUST pre-zero c (kernel does atomic adds, not writes).
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  // Byte-equivalent cast: uint8 and uint32 views of the same storage.
  const uint8_t* b_qw_bytes = b_q_weight.data_ptr<uint8_t>();

  if (a.scalar_type() == torch::kHalf) {
    vllm::mxfp4_dot2::launch_gemm_mxfp4<half>(
        (const half*)a.data_ptr(),
        reinterpret_cast<const uint32_t*>(b_qw_bytes),
        (const uint8_t*)b_scales.data_ptr(), (half*)c.data_ptr(), sm, sn, sk,
        stream);
  } else {
    TORCH_CHECK(false, "mxfp4_gemm_rdna2 only supports fp16; got dtype=",
                a.scalar_type());
  }
}
