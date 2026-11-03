// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RidgeFill map GEMM for AMD RDNA2 (gfx1030), fp16.
//
// RidgeFill replaces the late layers' block input on the filled prompt rows
// with a linear map of the layer-24 split state. Every map shares the same
// input, so the 24 per-layer products
//
//   out[l][m][d] = sum_k x[m][k] * w[l * DOUT + d][k] + b[l * DOUT + d]
//
// are ONE wide GEMM: the 24 maps stack along the weight rows (N = L * DOUT)
// and each layer reads its own [M, DOUT] slot of the [L, M, DOUT] output.
//
// rocBLAS fp16 reaches 11.3 T-MAC/s on this shape on a V620 (HIP 4,
// 2026-11-03), half of the 22.8 T-MAC/s v_dot2_f32_f16 peak, so headroom
// exists for a tiled HIP GEMM.
//
// Design
// ------
// * Tile: BLOCK_M 128 rows x BLOCK_N 128 columns, 1024 threads (32 wave32s).
//   Each thread holds a 4 x 4 register micro-tile (16 fp32 accumulators).
//   gfx1030 gives a lane 32 VGPRs at one resident wave, so an 8 x 8 tile
//   (64 accumulators) spills; that was the first version and it measured 10x
//   slower than rocBLAS.
// * K is blocked by BLOCK_K 32 (16 fdot2 pairs). A and B tiles are staged in
//   LDS already as half2 pairs along K, so the inner loop reads the pair
//   directly and issues one fdot2 per accumulator.
// * The staging reads 128 bits of a row at a time and writes four half2.
// * Double buffering: tile k+1 is staged while tile k is multiplied.
// * BLOCK_N divides DOUT (2560 / 128 = 20), so a tile never straddles two
//   stacked maps and the epilogue resolves the layer once.
// * Rows past M and k past K are zero-filled in LDS, so the compute has no
//   branch.

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
  #define __HIP__RDNA2_RIDGEFILL_MAP__ 1
#endif

namespace vllm {
namespace ridgefill_map_rdna2 {

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 1024;  // 128 x 128 tile / (4 x 4) micro-tile
constexpr int kRowsPerThread = 4;
constexpr int kColsPerThread = 4;
constexpr int kColGroups = kBlockN / kColsPerThread;  // 32
constexpr int kKPair = kBlockK / 2;                   // 16 pairs along K
constexpr int kSeg = 8;                               // halves per vector read

#if defined(__HIP__RDNA2_RIDGEFILL_MAP__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ void stage_rows(__half2* __restrict__ dst,
                                           const __half* __restrict__ src,
                                           int rows, int valid_rows, int k0,
                                           int size_k, int tid, int nthreads) {
  // One 128-bit read per 8 K halves of a row, then four half2 stores. Rows
  // past the M bound and K past the K bound are zero-filled.
  constexpr int kSegs = kBlockK / kSeg;
  const bool vector_ok = (size_k % kSeg) == 0;
  for (int i = tid; i < rows * kSegs; i += nthreads) {
    const int m = i / kSegs;
    const int seg = i % kSegs;
    const int k = k0 + seg * kSeg;
    __half2* out = dst + (int64_t)(seg * (kSeg / 2)) * rows + m;
    if (m < valid_rows && k + kSeg <= size_k && vector_ok) {
      const uint4 v =
          *reinterpret_cast<const uint4*>(src + (int64_t)m * size_k + k);
      const __half* h = reinterpret_cast<const __half*>(&v);
  #pragma unroll
      for (int j = 0; j < kSeg / 2; ++j)
        out[(int64_t)j * rows] = __halves2half2(h[2 * j], h[2 * j + 1]);
    } else {
  #pragma unroll
      for (int j = 0; j < kSeg / 2; ++j) {
        const int kk = k + 2 * j;
        __half2 p;
        p.x = (m < valid_rows && kk < size_k) ? src[(int64_t)m * size_k + kk]
                                              : __float2half(0.f);
        p.y = (m < valid_rows && kk + 1 < size_k)
                  ? src[(int64_t)m * size_k + kk + 1]
                  : __float2half(0.f);
        out[(int64_t)j * rows] = p;
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void ridgefill_map_kernel(
    const __half* __restrict__ x,     // [M, K]
    const __half* __restrict__ w,     // [L * DOUT, K]
    const __half* __restrict__ bias,  // [L * DOUT] or null
    __half* __restrict__ out,         // [L, M, DOUT]
    int size_m, int size_k, int dout) {
  // [kKPair][row] keeps the warp's operand reads conflict-free: the lanes of
  // a wave share tm (a[r] broadcasts) and read consecutive tn (b[c]).
  __shared__ __half2 sA[2][kKPair][kBlockM];
  __shared__ __half2 sB[2][kKPair][kBlockN];

  const int m0 = blockIdx.x * kBlockM;
  const int n0 = blockIdx.y * kBlockN;
  const int tid = threadIdx.x;
  const int tm = (tid / kColGroups) * kRowsPerThread;
  const int tn = (tid % kColGroups) * kColsPerThread;

  const __half* a_base = x + (int64_t)m0 * size_k;
  const __half* b_base = w + (int64_t)n0 * size_k;
  const int a_rows = size_m - m0;

  float acc[kRowsPerThread][kColsPerThread];
  #pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r)
  #pragma unroll
    for (int c = 0; c < kColsPerThread; ++c) acc[r][c] = 0.f;

  const int k_tiles = (size_k + kBlockK - 1) / kBlockK;
  stage_rows(&sA[0][0][0], a_base, kBlockM, a_rows, 0, size_k, tid, kThreads);
  stage_rows(&sB[0][0][0], b_base, kBlockN, kBlockN, 0, size_k, tid, kThreads);
  __syncthreads();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int cur = kt & 1;
    const int nxt = cur ^ 1;
    if (kt + 1 < k_tiles) {
      const int k_next = (kt + 1) * kBlockK;
      stage_rows(&sA[nxt][0][0], a_base, kBlockM, a_rows, k_next, size_k, tid,
                 kThreads);
      stage_rows(&sB[nxt][0][0], b_base, kBlockN, kBlockN, k_next, size_k, tid,
                 kThreads);
    }
  #pragma unroll
    for (int kp = 0; kp < kKPair; ++kp) {
      __half2 a[kRowsPerThread];
      __half2 b[kColsPerThread];
  #pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) a[r] = sA[cur][kp][tm + r];
  #pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) b[c] = sB[cur][kp][tn + c];
  #pragma unroll
      for (int c = 0; c < kColsPerThread; ++c)
  #pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r)
          acc[r][c] = __builtin_amdgcn_fdot2(b[c], a[r], acc[r][c], false);
    }
    if (kt + 1 < k_tiles) {
      __syncthreads();
    }
  }

  // Epilogue: [L, M, DOUT]. BLOCK_N divides DOUT, so l is one per tile.
  const int gn = n0 + tn;
  const int layer = gn / dout;
  const int d0 = gn % dout;
  __half* base = out + ((int64_t)layer * size_m + m0 + tm) * dout + d0;
  #pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    if (m0 + tm + r >= size_m) continue;
    __half2* dst = reinterpret_cast<__half2*>(base + (int64_t)r * dout);
  #pragma unroll
    for (int c = 0; c < kColsPerThread / 2; ++c) {
      const int d = 2 * c;
      const float b0 = bias ? __half2float(bias[gn + d]) : 0.f;
      const float b1 = bias ? __half2float(bias[gn + d + 1]) : 0.f;
      dst[c] = __halves2half2(__float2half(acc[r][d] + b0),
                              __float2half(acc[r][d + 1] + b1));
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
  const dim3 grid(static_cast<unsigned>((size_m + kBlockM - 1) / kBlockM),
                  static_cast<unsigned>(layers * dout / kBlockN));
  const dim3 block(kThreads);
  hipLaunchKernelGGL(ridgefill_map_kernel, grid, block, 0, stream, x, w, bias,
                     out, size_m, size_k, dout);
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
  TORCH_CHECK((layers * dout) % vllm::ridgefill_map_rdna2::kBlockN == 0,
              "ridgefill_map: BLOCK_N must divide layers * dout");
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
