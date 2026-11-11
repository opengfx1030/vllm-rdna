// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared MoE output epilogue for the RDNA2 (gfx1030) fused MoE kernels.
//
// ``moe_q_gemm_rdna2.cu`` (W4A16) and ``moe_w4a8_rdna2.cu`` (W4A8) emit the
// same output contract: one token-block row of 4 consecutive N columns with
// the router weight already folded in fp32, atomically reduced across experts
// when ``output_topk > 0``. Accumulation modes:
//
//   * fp16 CAS (C_T == 16-bit half, default): packed fp16 CAS-64 atomic add
//     into the caller's pre-zeroed fp16 output -- one retry loop per row.
//     Order-dependent at fp16 precision.
//   * fp32 (C_T == float, opt-in via VLLM_RDNA2_MOE_FP32_ACCUM=1): atomicAdd
//     into an fp32 scratch, no intermediate fp16 rounding. gfx1030 has
//     no native fp32 atomic add for global memory here: the built library
//     compiles it to a load + v_add_f32 + global_atomic_cmpswap retry loop
//     per column (four loops per row). A single fp32 -> fp16 cast runs once
//     at the end. The fp32 adds still round in arrival order, so the result
//     is still order-dependent; the final cast only hides most of it.
//   * cache fixed point (C_T == long long): native integer atomic add with
//     32 fractional bits into caller-zeroed [rows,N+1] storage. Integer adds
//     are associative, so this mode is order-independent (it backs
//     VLLM_RDNA2_MOE_DETERMINISTIC=1 for W4A16). The last column is a sticky
//     row status. Nonfinite/out-of-bound partials set it and skip
//     conversion; finalization poisons that row. The caller derives
//     a conservative bound from K-slices and top-k to prevent signed overflow.
//
// The fp32 scratch is a per-call block from the caching allocator
// (moe_fp32_scratch below): eager calls keep nothing, and under HIP graph
// capture the block comes from the graph's private pool, which keeps the
// address valid across replays. The per-call zero is a hipMemsetAsync (a
// capture-legal node).

#ifndef _MOE_ACCUM_RDNA2_CUH
#define _MOE_ACCUM_RDNA2_CUH

#include <cstdint>
#include <cstdlib>
#include <cstring>
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
//   C_T == float          -> atomicAdd into the scratch (CAS loop on gfx1030).
//   C_T == long long      -> caller-owned fixed point with 32 fractional bits.
//   C_T == 16-bit half    -> packed fp16 CAS-64 atomic add into the output.
// The 16-bit case accepts any layout-compatible half (HIP ``half`` /
// ``_Float16``); the packed CAS helper needs a ``half*``.
template <typename C_T>
__forceinline__ __device__ void moe_accum_row(const float (&vals)[4],
                                              C_T* __restrict__ out,
                                              long long* row_status = nullptr,
                                              float fixed_limit = 0.f) {
  if constexpr (std::is_same_v<C_T, float>) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      atomicAdd(out + j, vals[j]);
    }
  } else if constexpr (std::is_same_v<C_T, long long>) {
    fixed_accum_row(vals, out, row_status, fixed_limit);
  } else {
    half* h = reinterpret_cast<half*>(out);
    half2 r01 =
        __halves2half2(__float2half_rn(vals[0]), __float2half_rn(vals[1]));
    half2 r23 =
        __halves2half2(__float2half_rn(vals[2]), __float2half_rn(vals[3]));
    atomic_add_pk4_f16(h, r01, r23);
  }
}

// Direct (non-atomic) store of one row's 4 N-columns. Valid only when exactly
// one block writes each output element (no K split, no top-k reduce).
template <typename C_T>
__forceinline__ __device__ void moe_store_row(const float (&vals)[4],
                                              C_T* __restrict__ out) {
  static_assert(!std::is_same_v<C_T, long long>, "fixed point needs atomics");
  if constexpr (std::is_same_v<C_T, float>) {
    *reinterpret_cast<float4*>(out) =
        make_float4(vals[0], vals[1], vals[2], vals[3]);
  } else {
    union {
      half2 h2[2];
      unsigned long long u;
    } pk;
    pk.h2[0] =
        __halves2half2(__float2half_rn(vals[0]), __float2half_rn(vals[1]));
    pk.h2[1] =
        __halves2half2(__float2half_rn(vals[2]), __float2half_rn(vals[3]));
    *reinterpret_cast<unsigned long long*>(out) = pk.u;
  }
}

// R rows of one thread, reduced into the output together. The per-row helper
// above chains two dependent global round trips per row (plain load, then
// CAS), so a block of 8 rows pays ~16 serial latencies at the end of its
// life. Here all R loads are issued first, then all R CASes, and only rows
// whose CAS lost a race (another expert or K split wrote the same words in
// between) retry. Same arithmetic and same atomic granularity as
// ``moe_accum_row``; only the issue order differs. Rows with ``live[r]`` false
// are skipped. The fp32 and fixed-point accumulators keep the per-row helper.
template <int R, typename C_T>
__forceinline__ __device__ void moe_accum_rows(const float (&vals)[R][4],
                                               C_T* const (&out)[R],
                                               const bool (&live)[R]) {
  if constexpr (std::is_same_v<C_T, float> || std::is_same_v<C_T, long long>) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (live[r]) moe_accum_row<C_T>(vals[r], out[r]);
    }
  } else {
    union Pk {
      half2 h2[2];
      unsigned long long u;
    };
    unsigned long long* addr[R];
    unsigned long long old[R];
    unsigned long long prev[R];
    Pk add[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      addr[r] = reinterpret_cast<unsigned long long*>(out[r]);
      add[r].h2[0] = __halves2half2(__float2half_rn(vals[r][0]),
                                    __float2half_rn(vals[r][1]));
      add[r].h2[1] = __halves2half2(__float2half_rn(vals[r][2]),
                                    __float2half_rn(vals[r][3]));
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (live[r]) old[r] = *addr[r];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (live[r]) {
        Pk cur, sum;
        cur.u = old[r];
        sum.h2[0] = __hadd2(cur.h2[0], add[r].h2[0]);
        sum.h2[1] = __hadd2(cur.h2[1], add[r].h2[1]);
        prev[r] = atomicCAS(addr[r], old[r], sum.u);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (live[r]) {
        while (prev[r] != old[r]) {
          old[r] = prev[r];
          Pk cur, sum;
          cur.u = old[r];
          sum.h2[0] = __hadd2(cur.h2[0], add[r].h2[0]);
          sum.h2[1] = __hadd2(cur.h2[1], add[r].h2[1]);
          prev[r] = atomicCAS(addr[r], old[r], sum.u);
        }
      }
    }
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

// fp32 accumulator shaped like the fp16 output `c`, for one call of either MoE
// op. The returned tensor owns it; keep it alive until the launches that use
// it are enqueued. Releasing it lets the next call on the stream reuse the
// block, so eager calls keep nothing whatever their token count. Under HIP
// graph capture the caching allocator serves it from the graph's private
// pool, which reserves the address for the graph's replays.
inline at::Tensor moe_fp32_scratch(const at::Tensor& c) {
  return at::empty(c.sizes(), c.options().dtype(at::kFloat));
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
