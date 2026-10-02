// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Explore-only host + kernel sketch for W4A4 shape (1) on gfx1030.
//
// NOT WIRED:
//   - not in VLLM_ROCM_EXT_SRC (keep VLLM_RDNA2_W4A4_SDOT8=0)
//   - not registered in csrc/rocm/torch_bindings.cpp
//   - not referenced by can_implement / serve scripts
//
// Shape (1): ConfigA-class LDS=0, both sides packed signed i4,
// K_STEP ∈ {16,32}, one sdot8 per K-dword, full-K_STEP inner loop.
// Packing is K-contiguous nibbles, clamp=false.
// Scales belong in the epilogue; this sketch writes i32 C.
//
// A is packed i4 (runtime quant happens before this launcher).
// Do not unpack to i8 / call sdot4. Not E2M1. Not a torch op.

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "w4a4_sdot8_shape1.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
#define __HIP__RDNA2__
#endif

namespace vllm {
namespace explore_w4a4_sdot8 {

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int K_STEP>
__global__ __launch_bounds__(Shape1Config<K_STEP>::THREADS) void
gemm_w4a4_sdot8_shape1_kernel(const uint32_t* __restrict__ a_q,
                              const uint32_t* __restrict__ b_q_weight,
                              int32_t* __restrict__ c, int size_m, int size_n,
                              int size_k) {
  using Cfg = Shape1Config<K_STEP>;
  const int t = threadIdx.x;
  const int n = blockIdx.x * Cfg::N_TILE + t * Cfg::N_PER_THREAD;
  const int m_tile = blockIdx.y * Cfg::M_TILE;
  if (n >= size_n) {
    return;
  }

  int32_t acc[Cfg::M_TILE][Cfg::N_PER_THREAD];
#pragma unroll
  for (int m = 0; m < Cfg::M_TILE; ++m) {
#pragma unroll
    for (int col = 0; col < Cfg::N_PER_THREAD; ++col) {
      acc[m][col] = 0;
    }
  }

  // Inner loop covers every 8-K dword in K_STEP, then k += K_STEP.
  // Do not advance K by more than the unrolled body consumes.
  // One sdot8 per dword; no nibble→i8 expand.
  for (int k = 0; k < size_k; k += K_STEP) {
#pragma unroll
    for (int j = 0; j < Cfg::DWORDS_PER_STEP; ++j) {
      const int k8 = k + 8 * j;
      const int k_dword = k8 / 8;
#pragma unroll
      for (int col = 0; col < Cfg::N_PER_THREAD; ++col) {
        const int ncol = n + col;
        if (ncol >= size_n) {
          continue;
        }
        const uint32_t w = b_q_weight[k_dword * size_n + ncol];
#pragma unroll
        for (int m = 0; m < Cfg::M_TILE; ++m) {
          const int m_row = m_tile + m;
          if (m_row >= size_m) {
            continue;
          }
          const uint32_t a_pack = a_q[m_row * (size_k / 8) + k_dword];
          acc[m][col] = sdot8_from_i4_dwords(a_pack, w, acc[m][col]);
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < Cfg::M_TILE; ++m) {
    const int m_row = m_tile + m;
    if (m_row >= size_m) {
      continue;
    }
#pragma unroll
    for (int col = 0; col < Cfg::N_PER_THREAD; ++col) {
      const int ncol = n + col;
      if (ncol < size_n) {
        c[m_row * size_n + ncol] = acc[m][col];
      }
    }
  }
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// Explore host launcher. Not a torch custom op. Call only from a
// standalone research binary / opt-in compile.
void launch_w4a4_sdot8_shape1(const torch::Tensor& a_q_i4,
                              const torch::Tensor& b_q_weight,
                              torch::Tensor& c_i32, int k_step) {
  TORCH_CHECK(a_q_i4.scalar_type() == torch::kInt ||
                  a_q_i4.scalar_type() == torch::kUInt32,
              "A pack must be uint32 / int32 dwords (packed i4)");
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kInt ||
                  b_q_weight.scalar_type() == torch::kUInt32,
              "W pack must be uint32 / int32 dwords");
  TORCH_CHECK(c_i32.scalar_type() == torch::kInt, "C must be int32");
  TORCH_CHECK(a_q_i4.is_contiguous() && b_q_weight.is_contiguous() &&
                  c_i32.is_contiguous(),
              "explore W4A4 tensors must be contiguous");
  TORCH_CHECK(k_step == 16 || k_step == 32, "shape (1): K_STEP in {16,32}");

  const int size_m = static_cast<int>(a_q_i4.size(0));
  const int packed_k = static_cast<int>(a_q_i4.size(1));
  const int size_k = packed_k * 8;
  const int size_n = static_cast<int>(c_i32.size(1));
  TORCH_CHECK(size_k % 8 == 0, "contract: K % 8 == 0");
  TORCH_CHECK(c_i32.size(0) == size_m, "C M mismatch");
  TORCH_CHECK(b_q_weight.size(0) == packed_k, "W pack is [K/8, N]");
  TORCH_CHECK(b_q_weight.size(1) == size_n, "W pack N mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a_q_i4));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(Shape1Config<16>::THREADS);
  dim3 grid((size_n + Shape1Config<16>::N_TILE - 1) / Shape1Config<16>::N_TILE,
            (size_m + Shape1Config<16>::M_TILE - 1) / Shape1Config<16>::M_TILE);

  const uint32_t* a =
      reinterpret_cast<const uint32_t*>(a_q_i4.data_ptr());
  const uint32_t* w =
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr());
  int32_t* c = c_i32.data_ptr<int32_t>();

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  if (k_step == 16) {
    gemm_w4a4_sdot8_shape1_kernel<16>
        <<<grid, block, 0, stream>>>(a, w, c, size_m, size_n, size_k);
  } else {
    gemm_w4a4_sdot8_shape1_kernel<32>
        <<<grid, block, 0, stream>>>(a, w, c, size_m, size_n, size_k);
  }
#else
  (void)a;
  (void)w;
  (void)c;
  (void)grid;
  (void)block;
  (void)stream;
  TORCH_CHECK(false, "explore W4A4 sdot8 shape (1) is gfx1030-only");
#endif
}

}  // namespace explore_w4a4_sdot8
}  // namespace vllm
