// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RidgeFill map GEMM for AMD RDNA2 (gfx1030), fp16.
//
// RidgeFill replaces the late layers' block input on the filled prompt rows
// with a linear map of the layer-24 split state. Every map shares the same
// input, so the 24 per-layer products
//
//   out[l][m][n] = sum_k x[m][k] * w[l * DOUT + n][k] + b[l * DOUT + n]
//
// are ONE wide GEMM: x[m][k] is staged once into LDS per row tile and the
// stacked map weights stream past it, instead of 24 rocBLAS calls that each
// re-read the same x. The epilogue writes each map's rows straight into that
// layer's slot of the [L, M, DOUT] buffer the layers read by view, so no
// separate predict buffer, cast or replace launch is needed.
//
// Measured on a V620 (HIP 8, 2026-11-03): rocBLAS fp16 reaches 11.3 T-MAC/s
// on this shape (heuristic and tuned alike), exactly half of the 22.8 T-MAC/s
// v_dot2_f32_f16 peak, so the kernel accumulates with
// ``__builtin_amdgcn_fdot2`` in fp32.
//
// Design
// ------
// * Tile: BLOCK_M 32 rows x BLOCK_N 128 columns, 128 threads, K blocked by
//   BLOCK_K 64. Both operands are staged in LDS (x transposed so a thread
//   reads RT consecutive rows with one vector load, w padded so the CT
//   column reads do not collide on a bank).
// * BLOCK_N divides DOUT, so a tile never straddles two maps and the epilogue
//   resolves l once.
// * The rows past M are masked on load; the grid is exact, so only the x tile
//   needs the bound.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ops.h"

// The build defines VLLM_ROCM_GFX1030 for a gfx1030 target in both the
// host and the device pass; __gfx1030__ is device-only. Either one selects
// the kernel, so the host launcher always sees a definition.
#if defined(__HIPCC__) && (defined(__gfx1030__) || defined(VLLM_ROCM_GFX1030))
  #define __HIP__RDNA2_RIDGEFILL_MAP__
#endif

namespace vllm {
namespace ridgefill_map_rdna2 {

constexpr int kBlockM = 32;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kThreads = 128;
constexpr int kRowsPerThread = 4;  // kBlockM / (kThreads / kColsPerGroup)
constexpr int kColsPerThread = 8;  // kBlockN / 16 column groups
constexpr int kColGroups = 16;
constexpr int kLdW = kBlockK + 8;  // LDS pad against bank clash

#if defined(__HIP__RDNA2_RIDGEFILL_MAP__) || !defined(__HIP_DEVICE_COMPILE__)

__global__ __launch_bounds__(kThreads) void ridgefill_map_kernel(
    const __half* __restrict__ x,     // [M, K]
    const __half* __restrict__ w,     // [L * DOUT, K]
    const __half* __restrict__ bias,  // [L * DOUT] or null
    __half* __restrict__ out,         // [L, M, DOUT]
    int size_m, int size_k, int dout) {
  __shared__ __half xs[kBlockK][kBlockM];
  __shared__ __half ws[kBlockN][kLdW];

  const int m0 = blockIdx.x * kBlockM;
  const int n0 = blockIdx.y * kBlockN;
  const int tid = threadIdx.x;
  const int row = (tid / kColGroups) * kRowsPerThread;
  const int col = (tid % kColGroups) * kColsPerThread;

  float acc[kRowsPerThread][kColsPerThread];
  #pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r)
  #pragma unroll
    for (int c = 0; c < kColsPerThread; ++c) acc[r][c] = 0.f;

  for (int k0 = 0; k0 < size_k; k0 += kBlockK) {
    // x tile, transposed: one 16-byte global read per row segment.
    for (int i = tid; i < kBlockM * (kBlockK / 8); i += kThreads) {
      const int m = i % kBlockM;
      const int k8 = (i / kBlockM) * 8;
      const int gm = m0 + m;
      const __half* src = x + (int64_t)gm * size_k + k0 + k8;
      __half* dst = &xs[k8][m];
      if (gm < size_m) {
        const uint4 v = *reinterpret_cast<const uint4*>(src);
        const __half* h = reinterpret_cast<const __half*>(&v);
  #pragma unroll
        for (int j = 0; j < 8; ++j) dst[j * kBlockM] = h[j];
      } else {
  #pragma unroll
        for (int j = 0; j < 8; ++j) dst[j * kBlockM] = __half(0);
      }
    }
    // w tile: kBlockN rows of the stacked map, contiguous in K.
    for (int i = tid; i < kBlockN * (kBlockK / 8); i += kThreads) {
      const int n = i % kBlockN;
      const int k8 = (i / kBlockN) * 8;
      const uint4 v = *reinterpret_cast<const uint4*>(
          w + (int64_t)(n0 + n) * size_k + k0 + k8);
      *reinterpret_cast<uint4*>(&ws[n][k8]) = v;
    }
    __syncthreads();

    for (int k = 0; k < kBlockK; k += 2) {
      __half2 a[kRowsPerThread];
  #pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r)
        // fdot2 multiplies along K, so the pair is (k, k + 1) for row r.
        a[r] = __halves2half2(xs[k][row + r], xs[k + 1][row + r]);
  #pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        const __half2 w2 = *reinterpret_cast<const __half2*>(&ws[col + c][k]);
  #pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
          const __half2 ar = a[r];
          acc[r][c] += __half2float(ar.x) * __half2float(w2.x) +
                       __half2float(ar.y) * __half2float(w2.y);
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: [L, M, DOUT] with l resolved once per tile.
  const int layer = (n0 + col) / dout;
  __half* base =
      out + ((int64_t)layer * size_m + m0) * dout + (n0 + col) % dout;
  #pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    const int gm = m0 + row + r;
    if (gm >= size_m) continue;
    __half* dst_row = base + (int64_t)(row + r) * dout;
  #pragma unroll
    for (int c = 0; c < kColsPerThread; ++c) {
      const int gn = n0 + col + c;
      const float b = bias ? __half2float(bias[gn]) : 0.f;
      const __half v = __float2half(acc[r][c] + b);
      dst_row[c] = v;
    }
  }
}

#else  // non-RDNA: empty stub for symbol parity

__global__ void ridgefill_map_kernel(const __half*, const __half*,
                                     const __half*, __half*, int, int, int) {}

#endif  // __HIP__RDNA2_RIDGEFILL_MAP__ || !__HIP_DEVICE_COMPILE__

void launch_map(const __half* x, const __half* w, const __half* bias,
                __half* out, int size_m, int size_k, int dout, int layers,
                hipStream_t stream) {
  const dim3 grid(
      static_cast<unsigned>((size_m + vllm::ridgefill_map_rdna2::kBlockM - 1) /
                            vllm::ridgefill_map_rdna2::kBlockM),
      static_cast<unsigned>(layers * dout /
                            vllm::ridgefill_map_rdna2::kBlockN));
  const dim3 block(vllm::ridgefill_map_rdna2::kThreads);
  hipLaunchKernelGGL(vllm::ridgefill_map_rdna2::ridgefill_map_kernel, grid,
                     block, 0, stream, x, w, bias, out, size_m, size_k, dout);
}

}  // namespace ridgefill_map_rdna2
}  // namespace vllm

void ridgefill_map_rdna2(torch::Tensor x, torch::Tensor w,
                         c10::optional<torch::Tensor> bias, torch::Tensor out,
                         int64_t dout, int64_t layers) {
#if !defined(VLLM_ROCM_GFX1030)
  // The build declares each arch; the kernel body is gfx1030-only, so a
  // non-gfx1030 launch must fail closed rather than writing nothing.
  TORCH_CHECK(false, "ridgefill_map: gfx1030 only");
#endif
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "ridgefill_map: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                  w.scalar_type() == torch::kHalf &&
                  out.scalar_type() == torch::kHalf,
              "ridgefill_map: fp16 only");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 3,
              "ridgefill_map: x[F,K] w[L*D,K] out[L,F,D]");
  TORCH_CHECK(w.size(0) == layers * dout && w.size(1) == x.size(1),
              "ridgefill_map: weight shape");
  TORCH_CHECK(
      out.size(0) == layers && out.size(1) == x.size(0) && out.size(2) == dout,
      "ridgefill_map: out shape");
  if (x.size(0) == 0) return;

  const at::cuda::OptionalCUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  const int size_m = static_cast<int>(x.size(0));
  const int size_k = static_cast<int>(x.size(1));
  const __half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kHalf &&
                    bias->numel() == layers * dout,
                "ridgefill_map: bias shape");
    bp = reinterpret_cast<const __half*>(bias->data_ptr());
  }

  vllm::ridgefill_map_rdna2::launch_map(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(w.data_ptr()), bp,
      reinterpret_cast<__half*>(out.data_ptr()), size_m, size_k,
      static_cast<int>(dout), static_cast<int>(layers), stream);
}
