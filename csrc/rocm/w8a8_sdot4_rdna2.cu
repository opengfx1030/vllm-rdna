// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 (int8 weights, int8 activations) dense GEMM for AMD RDNA2 (gfx1030).
//
// Opt-in dense linear for compressed-tensors W8A8 int8 checkpoints
// (RDNA2Int8ScaledMMLinearKernel). One torch op:
//
//   w8a8_gemm_rdna2(a, w_packed, w_scales, a_group) -> Tensor
//     a         [M, K]      fp16 activations, quantized in here per
//                           (token, a_group) by the W4A8 act_quant kernel
//     w_packed  [K/4, N]    int32, rdna2_w8a8.pack_weight layout
//     w_scales  [K/WG, N]   fp16 (WG = K for per-channel)
//     returns   [M, N]      fp16
//
// Unlike the W4A8 entry there is no fallback kernel for this weight layout,
// so every ineligible shape is a TORCH_CHECK, and the Python side keeps
// the shape rules (K % 32, N % 4, a_group | K, a_group | WG).
//
// All M run here. Prefill tiles the M x N grid; at small M (decode) the
// grid is filled by splitting K across blocks into a zeroed fp32 scratch
// (native fp32 atomics), cast to fp16 once. The kernel body lives in
// w8a8_sdot4_rdna2.cuh.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "moe_accum_rdna2.cuh"  // moe_cast_f32_to_f16
#include "ops.h"
#include "w8a8_sdot4_rdna2.cuh"

namespace ex = vllm::explore_w4a8;
namespace w8 = vllm::w8a8_rdna2;

namespace {

// Split K across blocks until the grid reaches this many blocks (V620: 72
// CUs). Prefill shapes reach it from the M x N tiling alone and keep plain
// fp16 stores.
constexpr int64_t kTargetBlocks = 256;

bool on_gfx1030() {
  thread_local int cached_dev = -1;
  thread_local bool cached_ok = false;
  const int dev = at::cuda::current_device();
  if (dev != cached_dev) {
    hipDeviceProp_t prop;
    cached_ok = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx1030", 7) == 0;
    cached_dev = dev;
  }
  return cached_ok;
}

int act_group_index(int64_t a_group) {
  switch (a_group) {
    case 32:
      return 0;
    case 64:
      return 1;
    case 128:
      return 2;
    default:
      return -1;
  }
}

// Env-gated diagnostic: VLLM_RDNA2_W8A8_DEBUG=1 logs each distinct
// (m, k, n, a_group, split) once per process. Read once so the traced
// forward never touches the env.
bool w8a8_debug_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("VLLM_RDNA2_W8A8_DEBUG");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

void w8a8_log_shape(int m, int k, int n, int a_group, int split) {
  static std::mutex mu;
  static std::set<std::tuple<int, int, int, int, int>> seen;
  const std::tuple<int, int, int, int, int> key{m, k, n, a_group, split};
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
      return;
    }
  }
  fprintf(stderr, "[W8A8-DEBUG] dense: m=%d k=%d n=%d a_group=%d split=%d\n", m,
          k, n, a_group, split);
  fflush(stderr);
}

struct ActWorkspace {
  at::Tensor a_i8;     // [T, K/8, 8, 8]
  at::Tensor a_scale;  // [T, K/AG, 8]
  at::Tensor a_sum;    // [T, K/AG, 8] (written by act_quant, unused here)
};

// Per-(token, a_group) int8 quant of x [M, K] fp16 into the W4A8 tile
// layout. Per-call allocations go through the caching allocator (graph
// pools during capture), as the W4A8 paths do.
ActWorkspace quant_act(const at::Tensor& x, int64_t a_group,
                       hipStream_t stream) {
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t tiles = (m + w8::kMTile - 1) / w8::kMTile;
  const int64_t groups = k / a_group;
  const auto opts = x.options();
  ActWorkspace ws{
      at::empty({tiles, k / 8, w8::kMTile, 8}, opts.dtype(at::kChar)),
      at::empty({tiles, groups, w8::kMTile}, opts.dtype(at::kFloat)),
      at::empty({tiles, groups, w8::kMTile}, opts.dtype(at::kInt))};
  ex::w4a8_act_quant_kernel<w8::kThreads, w8::kMTile, /*PER_GROUP=*/true>
      <<<dim3(static_cast<unsigned>(tiles)), dim3(w8::kThreads), 0, stream>>>(
          reinterpret_cast<const ex::f16_t*>(x.data_ptr()), x.stride(0),
          reinterpret_cast<int8_t*>(ws.a_i8.data_ptr()),
          reinterpret_cast<float*>(ws.a_scale.data_ptr()),
          reinterpret_cast<int32_t*>(ws.a_sum.data_ptr()), static_cast<int>(m),
          static_cast<int>(k), static_cast<int>(a_group));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ws;
}

struct GemmArgs {
  const int8_t* a;
  const float* a_scale;
  const uint32_t* w;
  const ex::f16_t* scales;
  void* out;
  int m, n, k, w_group, k_per_split, grid_z, out_f32;
};

template <int AG>
void launch_gemm(const GemmArgs& p, hipStream_t stream) {
  const dim3 grid((p.n + w8::kNTile - 1) / w8::kNTile,
                  (p.m + w8::kMTile - 1) / w8::kMTile, p.grid_z);
  w8::w8a8_gemm_kernel<AG><<<grid, dim3(w8::kThreads), 0, stream>>>(
      p.a, p.a_scale, p.w, p.scales, p.out, p.m, p.n, p.k, p.w_group,
      p.k_per_split, p.out_f32);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

using LaunchFn = void (*)(const GemmArgs&, hipStream_t);
const LaunchFn kLaunch[3] = {&launch_gemm<32>, &launch_gemm<64>,
                             &launch_gemm<128>};

}  // namespace

at::Tensor w8a8_gemm_rdna2(const at::Tensor& a, const at::Tensor& w_packed,
                           const at::Tensor& w_scales, int64_t a_group) {
  TORCH_CHECK(on_gfx1030(), "w8a8_gemm_rdna2: gfx1030 only");
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == at::kHalf &&
                  a.stride(1) == 1,
              "w8a8_gemm_rdna2: a must be a fp16 [M, K] HIP tensor with "
              "contiguous rows");
  const int64_t size_m = a.size(0);
  const int64_t size_k = a.size(1);
  TORCH_CHECK(size_m > 0 && size_k > 0 && size_k % 32 == 0,
              "w8a8_gemm_rdna2: K must be a positive multiple of 32");
  TORCH_CHECK(a.stride(0) % 8 == 0 && a.stride(0) >= size_k,
              "w8a8_gemm_rdna2: a rows must be 16-byte aligned");
  TORCH_CHECK(w_packed.is_cuda() && w_packed.dim() == 2 &&
                  w_packed.scalar_type() == at::kInt &&
                  w_packed.is_contiguous() && w_packed.size(0) * 4 == size_k,
              "w8a8_gemm_rdna2: w_packed must be int32 [K/4, N] contiguous");
  const int64_t size_n = w_packed.size(1);
  TORCH_CHECK(size_n > 0 && size_n % w8::kNPerThread == 0,
              "w8a8_gemm_rdna2: N must be a positive multiple of 4");
  TORCH_CHECK(w_scales.is_cuda() && w_scales.dim() == 2 &&
                  w_scales.scalar_type() == at::kHalf &&
                  w_scales.is_contiguous() && w_scales.size(1) == size_n &&
                  w_scales.size(0) > 0 && size_k % w_scales.size(0) == 0,
              "w8a8_gemm_rdna2: w_scales must be fp16 [K/WG, N] contiguous");
  const int64_t w_group = size_k / w_scales.size(0);
  const int gi = act_group_index(a_group);
  TORCH_CHECK(gi >= 0, "w8a8_gemm_rdna2: a_group must be 32, 64 or 128");
  TORCH_CHECK(size_k % a_group == 0 && w_group % a_group == 0,
              "w8a8_gemm_rdna2: a_group must divide K and the weight group");
  TORCH_CHECK(size_m * size_n < (int64_t{1} << 31),
              "w8a8_gemm_rdna2: M * N must fit 32-bit offsets");

  const at::cuda::OptionalCUDAGuard guard(a.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  const ActWorkspace ws = quant_act(a, a_group, stream);
  auto out = at::empty({size_m, size_n}, a.options());

  // Fill the GPU at small M by splitting K across blocks. Each split owns a
  // whole number of 256-K windows; split 1 keeps plain fp16 stores.
  const int64_t blocks_mn = ((size_n + w8::kNTile - 1) / w8::kNTile) *
                            ((size_m + w8::kMTile - 1) / w8::kMTile);
  const int64_t windows = (size_k + w8::kKWindow - 1) / w8::kKWindow;
  int64_t split = 1;
  if (blocks_mn < kTargetBlocks) {
    split = std::min(windows, (kTargetBlocks + blocks_mn - 1) / blocks_mn);
  }
  const int64_t windows_per_block = (windows + split - 1) / split;
  const int64_t k_per_split = windows_per_block * w8::kKWindow;
  const int64_t grid_z = (windows + windows_per_block - 1) / windows_per_block;
  const bool out_f32 = grid_z > 1;

  at::Tensor scratch;
  void* out_ptr = out.data_ptr();
  if (out_f32) {
    scratch = at::zeros({size_m, size_n}, a.options().dtype(at::kFloat));
    out_ptr = scratch.data_ptr();
  }

  const GemmArgs p{static_cast<const int8_t*>(ws.a_i8.data_ptr()),
                   static_cast<const float*>(ws.a_scale.data_ptr()),
                   static_cast<const uint32_t*>(w_packed.data_ptr()),
                   static_cast<const ex::f16_t*>(w_scales.data_ptr()),
                   out_ptr,
                   static_cast<int>(size_m),
                   static_cast<int>(size_n),
                   static_cast<int>(size_k),
                   static_cast<int>(w_group),
                   static_cast<int>(k_per_split),
                   static_cast<int>(grid_z),
                   out_f32 ? 1 : 0};
  kLaunch[gi](p, stream);
  if (out_f32) {
    moe_cast_f32_to_f16(static_cast<const float*>(out_ptr),
                        reinterpret_cast<half*>(out.data_ptr()), out.numel(),
                        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  TORCH_WARN_ONCE("RDNA2 W8A8 sdot4 dense path active");
  if (w8a8_debug_enabled()) {
    w8a8_log_shape(static_cast<int>(size_m), static_cast<int>(size_k),
                   static_cast<int>(size_n), static_cast<int>(a_group),
                   static_cast<int>(grid_z));
  }
  return out;
}
