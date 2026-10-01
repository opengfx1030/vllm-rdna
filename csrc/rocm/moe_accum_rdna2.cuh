// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared MoE output epilogue for the RDNA2 (gfx1030) fused MoE kernels.
//
// ``moe_q_gemm_rdna2.cu`` (W4A16) and ``moe_w4a8_rdna2.cu`` (W4A8) emit the
// same output contract: one token-block row of 4 consecutive N columns with
// the router weight already folded in fp32, atomically reduced across experts
// when ``output_topk > 0``. Two accumulation modes:
//
//   * fp32 (C_T == float, default): native v_global_atomic_add_f32 into a
//     cached fp32 scratch -- no CAS, no intermediate fp16 rounding. A single
//     fp32 -> fp16 cast runs once at the end, so the result does not depend
//     on the atomic ordering at fp16 precision (run-to-run stable in
//     practice).
//   * fp16 CAS (C_T == 16-bit half, opt-in): packed fp16 CAS-64 atomic add
//     into the caller's pre-zeroed fp16 output. Order-dependent at fp16
//     precision.
//
// The fp32 scratch is a persistent per-(rows, n, device) allocation: the
// device pointer baked into a captured HIP graph stays valid across replays,
// and the per-call zero is a hipMemsetAsync (a capture-legal node), never an
// allocation inside the captured region.

#ifndef _MOE_ACCUM_RDNA2_CUH
#define _MOE_ACCUM_RDNA2_CUH

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"

namespace vllm {
namespace gptq_rdna2 {

// Emit one token's 4 N-columns into accumulator row ``out``.
//   C_T == float          -> native fp32 atomic add (no CAS) into the scratch.
//   C_T == 16-bit half    -> packed fp16 CAS-64 atomic add into the output.
// The 16-bit case accepts any layout-compatible half (HIP ``half`` /
// ``_Float16``); the packed CAS helper needs a ``half*``.
template <typename C_T>
__forceinline__ __device__ void moe_accum_row(const float (&vals)[4],
                                              C_T* __restrict__ out) {
  if constexpr (std::is_same_v<C_T, float>) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      atomicAdd(out + j, vals[j]);
    }
  } else {
    half* h = reinterpret_cast<half*>(out);
    half2 r01 =
        __halves2half2(__float2half_rn(vals[0]), __float2half_rn(vals[1]));
    half2 r23 =
        __halves2half2(__float2half_rn(vals[2]), __float2half_rn(vals[3]));
    atomic_add_pk4_f16(h, r01, r23);
  }
}

}  // namespace gptq_rdna2
}  // namespace vllm

// fp32 scratch -> fp16, one element per thread. Deterministic; a plain kernel
// launch, so it is capture-safe.
static __global__ void moe_cast_f32_to_f16_kernel(const float* __restrict__ src,
                                                  half* __restrict__ dst,
                                                  int64_t numel) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    dst[i] = __float2half_rn(src[i]);
  }
}

// Persistent per-(rows, n, device) fp32 accumulator. Allocated on first use
// (eager, before graph capture) and never freed, so the device pointer baked
// into a captured graph stays valid across replays. Shared by both MoE ops;
// the per-call zero is hipMemsetAsync, never an allocation.
inline float* moe_fp32_scratch(int64_t rows, int64_t n, int64_t device) {
  static std::mutex mu;
  static std::map<std::tuple<int64_t, int64_t, int64_t>, at::Tensor> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(rows, n, device);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second.data_ptr<float>();
  }
  const at::Device dev(at::kCUDA, device);
  at::Tensor t =
      at::empty({rows, n}, at::TensorOptions().device(dev).dtype(at::kFloat));
  return cache.emplace(key, std::move(t)).first->second.data_ptr<float>();
}

inline void moe_cast_f32_to_f16(const float* src, half* dst, int64_t numel,
                                hipStream_t stream) {
  if (numel <= 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (numel + kThreads - 1) / kThreads;
  moe_cast_f32_to_f16_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                               stream>>>(src, dst, numel);
}

#endif  // _MOE_ACCUM_RDNA2_CUH
