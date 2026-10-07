// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A8 (int4 weights, int8 activations) prefill GEMM for AMD RDNA2 (gfx1030).
//
// Opt-in drop-in replacement for the dense W4A16 prefill GEMM. Two torch ops:
//
//   w4a8_act_quant_rdna2(x, group_size, a_i8, a_scale, a_asum) -> Tensor
//     Per-(token, group) int8 quant of the fp16 activations plus per-token
//     (or per-(token,group), per config) fp32 scales and per-group int32 sums,
//     written in the tiled layout the GEMM below consumes. Returns the int8
//     buffer on success or an empty tensor when ineligible (used by the
//     standalone test).
//
//   w4a8_gemm_rdna2(a, w_packed, qzeros, scales, b_g_idx, use_v2_format)
//       -> Tensor
//     Self-contained: allocates the int8 A, A scales, A group sums and the
//     output internally, then fires the a8_lds_k32_ag GEMM or falls back to
//     gptq_gemm_rdna2_prefill when the shape/LDS is not eligible. Weights are
//     the SAME packed buffer RDNA2W4A16LinearKernel already produces
//     (zero-extended nibbles + gptq_shuffle).
//
// The kernel body lives in the sibling header `w4a8_sdot4_rdna2.cuh`.
// This TU is the production wrapper: the gemm entry owns the shape/LDS
// eligibility and its own internal W4A16 fallback, so Python never branches
// on a runtime value in the traced forward.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "w4a8_sdot4_rdna2.cuh"

// Forward declaration — defined in csrc/rocm/q_gemm_rdna2_prefill.cu.
torch::Tensor gptq_gemm_rdna2_prefill(torch::Tensor a, torch::Tensor b_q_weight,
                                      torch::Tensor b_qzeros,
                                      torch::Tensor b_scales,
                                      torch::Tensor b_g_idx,
                                      bool use_v2_format);

namespace ex = vllm::explore_w4a8;

namespace {

enum Err : int {
  kBadShape = -1,
  kBadConfig = -2,
  kNotGfx1030 = -3,
  kLdsTooBig = -4,
  kBadSplit = -5,
  kBadGroup = -6,
};

constexpr int kMaxSplit = 16;

// Prefill-only: decode (M < kW4a8MinRows) keeps the W4A16 arms; the wired
// W4A8 config is a8_lds_k32_ag (M_TILE 8).
constexpr int kW4a8MinRows = 33;

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

int group_index(int64_t group_size) {
  switch (group_size) {
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

// Mirror of pick_split_k's LDS cap: some group-aligned split <= 16 must bring
// M_TILE=8 rows of K plus the per-(token, group) scales under 64 KiB of LDS.
//
// The K_STEP-alignment guard mirrors pick_split_k/compute_split_k: the kernel
// walks K in K_STEP-wide chunks and never clamps the tail to k_per_split, so a
// split whose k_per_split is not a whole multiple of K_STEP would read past
// the split. `k % 32 == 0` is part of the eligibility gate, so split=1 is
// always aligned and the scan still terminates on every eligible shape.
bool w4a8_lds_fits(int k, int group_size) {
  const int groups = k / group_size;
  for (int split = 16; split >= 1; --split) {
    if (groups % split) {
      continue;
    }
    const int kps = k / split;
    if (kps % 32) {
      continue;
    }
    if (8 * kps + 8 * (kps / group_size) * 8 <= 64 * 1024) {
      return true;
    }
  }
  return false;
}

struct GemmArgs {
  const int8_t* a;
  const uint32_t* w;
  const uint32_t* qzeros;
  const ex::f16_t* scales;
  const float* a_scale;
  const int32_t* asum;
  void* out;
  int m, n, k, zero_offset, split_k, out_f32;
};

// Same rule as reference.pick_split_k: LDS budget by grid size, then grow
// while the grid is small or the K range long, restricted to group-aligned
// splits (K % (group * split) == 0).
template <class C>
int pick_split_k(int m, int n, int k) {
  const int blocks =
      ((m + C::M_TILE - 1) / C::M_TILE) * ((n + C::N_TILE - 1) / C::N_TILE);
  const int budget = blocks > 1024  ? 16 * 1024
                     : blocks > 256 ? 64 * 1024
                                    : 32 * 1024;
  const int groups = k / C::GROUP;
  int splits[kMaxSplit];
  int count = 0;
  for (int s = 1; s <= kMaxSplit; ++s) {
    // Group-aligned AND K_STEP-aligned: the kernel's K_STEP-wide inner loop
    // never clamps the tail to k_per_split, so a split with
    // (k / s) % K_STEP != 0 would over-read the split. `k % 32 == 0` is
    // checked by the caller, so s=1 is always aligned and the list is
    // non-empty.
    if (groups % s == 0 && (k / s) % C::K_STEP == 0) {
      splits[count++] = s;
    }
  }
  auto lds = [&](int s) {
    const int kps = k / s;
    return C::M_TILE * kps +
           C::M_TILE * (kps / C::GROUP) * (C::A_GROUP ? 8 : 4);
  };
  int i = 0;
  while (i + 1 < count && lds(splits[i]) > budget) {
    ++i;
  }
  while (i + 1 < count && (blocks * splits[i] < 2048 || k / splits[i] > 2048)) {
    if (lds(splits[i + 1]) > budget) {
      break;
    }
    ++i;
  }
  while (i + 1 < count && ex::lds_bytes<C>(k / splits[i]) > 64 * 1024) {
    ++i;
  }
  return splits[i];
}

template <class C>
int launch_gemm(const GemmArgs& p, hipStream_t stream) {
  if (p.m <= 0 || p.n <= 0 || p.n % 8 || p.k % 32 || p.k % C::GROUP) {
    return kBadShape;
  }
  const int groups = p.k / C::GROUP;
  const int split = p.split_k > 0 ? p.split_k : pick_split_k<C>(p.m, p.n, p.k);
  if (split > kMaxSplit || groups % split || (p.out_f32 && split != 1)) {
    return kBadSplit;
  }
  const int k_per_split = p.k / split;
  const int lds = ex::lds_bytes<C>(k_per_split);
  if (lds > 64 * 1024) {
    return kLdsTooBig;
  }
  if (!p.out_f32 && split > 1) {
    const hipError_t e = hipMemsetAsync(
        p.out, 0, static_cast<size_t>(p.m) * p.n * sizeof(ex::f16_t), stream);
    if (e != hipSuccess) {
      return static_cast<int>(e);
    }
  }
  const dim3 grid((p.n + C::N_TILE - 1) / C::N_TILE,
                  (p.m + C::M_TILE - 1) / C::M_TILE, split);
  auto* kernel = ex::w4a8_gemm_kernel<C>;
  kernel<<<grid, dim3(C::THREADS), lds, stream>>>(
      p.a, p.w, p.qzeros, p.scales, p.a_scale, p.asum, p.out, p.m, p.n, p.k,
      p.zero_offset, k_per_split, split, p.out_f32);
  const hipError_t e = hipGetLastError();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e == hipSuccess ? 0 : static_cast<int>(e);
}

using LaunchFn = int (*)(const GemmArgs&, hipStream_t);
using SplitFn = int (*)(int, int, int);

const LaunchFn kLaunchA8LdsK32Ag[3] = {
    &launch_gemm<ex::Cfg<256, 4, 32, 8, 32, ex::ASrc::kLds, true>>,
    &launch_gemm<ex::Cfg<256, 4, 32, 8, 64, ex::ASrc::kLds, true>>,
    &launch_gemm<ex::Cfg<256, 4, 32, 8, 128, ex::ASrc::kLds, true>>,
};

// Thread t owns row (t % MT) of its block's row tile; the per-group activation
// scale variant (A_GROUP) needs no block reduction, so the launch is identical
// for both — only the a_scale layout differs (per token vs [T][K/G][MT]).
template <int MT, bool PER_GROUP>
int launch_act_quant(const void* x, int64_t x_row_stride, void* a, void* a_scale,
                     void* asum, int m, int k, int group_size,
                     hipStream_t stream) {
  auto* kernel = ex::w4a8_act_quant_kernel<256, MT, PER_GROUP>;
  kernel<<<dim3((m + MT - 1) / MT), dim3(256), 0, stream>>>(
      static_cast<const ex::f16_t*>(x), x_row_stride, static_cast<int8_t*>(a),
      static_cast<float*>(a_scale), static_cast<int32_t*>(asum), m, k,
      group_size);
  const hipError_t e = hipGetLastError();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e == hipSuccess ? 0 : static_cast<int>(e);
}

}  // namespace

// ---------------------------------------------------------------------------
// torch ops
// ---------------------------------------------------------------------------

// x [M, K] fp16 -> a_i8 [T][K/8][MT][8], a_scale [T][K/G][MT] f32 (per-(token,
// group) for the A_GROUP config) and a_asum [T][K/G][MT] int32. MT is the M
// tile of the configured GEMM (a8_lds_k32_ag uses 8). Returns 0 or an error.
at::Tensor w4a8_act_quant_rdna2(const at::Tensor& x, int64_t group_size,
                         at::Tensor& a_i8, at::Tensor& a_scale,
                         at::Tensor& a_asum) {
  if (!on_gfx1030()) {
    return at::Tensor();
  }
  if (group_index(group_size) < 0) {
    return at::Tensor();
  }
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t row_stride = x.stride(0);
  if (m <= 0 || k % 32 || k % group_size || row_stride % 8 || row_stride < k) {
    return at::Tensor();
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // The wired path uses the A_GROUP config (per-(token, group) scales) with an
  // 8-row M tile; keep the M tile selection here so a config change only needs
  // one edit, and so Python never branches on M.
  constexpr int kMTile = 8;
  constexpr bool kPerGroup = true;
  const int64_t rc = launch_act_quant<kMTile, kPerGroup>(
      x.data_ptr(), row_stride, a_i8.data_ptr(), a_scale.data_ptr(),
      a_asum.data_ptr(), static_cast<int>(m), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  return rc == 0 ? a_i8 : at::Tensor();
}

at::Tensor w4a8_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                           torch::Tensor b_qzeros, torch::Tensor b_scales,
                           torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(a.scalar_type() == at::kHalf,
              "w4a8_gemm_rdna2 only supports fp16");

  const int size_m = a.size(0);
  const int size_k = a.size(1);
  const int size_n = b_q_weight.size(1);
  const int groups = b_qzeros.size(0);
  const int group_size = groups > 0 ? size_k / groups : 0;
  const bool has_g_idx = b_g_idx.numel() > 0;

  // Eligibility gate — mirrors the Python pre-check (m >= kW4a8MinRows,
  // k % 32 == 0, k % group_size == 0, LDS fits) plus the C++-only checks
  // (gfx1030, fp16, g_idx-free, group 32/64/128, contiguous rows). The
  // Python selector is only an optimisation; this entry always returns a
  // correct [M, N] tensor, falling back to the W4A16 prefill internally.
  const bool eligible =
      on_gfx1030() && size_m >= kW4a8MinRows && !has_g_idx &&
      size_k % 32 == 0 && group_index(group_size) >= 0 &&
      size_k % group_size == 0 && a.stride(0) % 8 == 0 &&
      a.stride(0) >= size_k && w4a8_lds_fits(size_k, group_size);

  if (!eligible) {
    return gptq_gemm_rdna2_prefill(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                   use_v2_format);
  }

  const at::cuda::OptionalCUDAGuard guard(a.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kMTile = 8;
  constexpr bool kPerGroup = true;
  const int num_tiles = (size_m + kMTile - 1) / kMTile;
  auto a_i8 = at::empty({num_tiles, size_k / 8, kMTile, 8},
                        a.options().dtype(at::kChar));
  auto a_scale = at::empty({num_tiles, groups, kMTile},
                           a.options().dtype(at::kFloat));
  auto a_asum = at::empty({num_tiles, groups, kMTile},
                          a.options().dtype(at::kInt));
  auto out = at::empty({size_m, size_n}, a.options());

  const int zero_offset = use_v2_format ? 0 : 1;

  if (launch_act_quant<kMTile, kPerGroup>(
          a.data_ptr(), a.stride(0), a_i8.data_ptr(), a_scale.data_ptr(),
          a_asum.data_ptr(), size_m, size_k, group_size, stream) != 0) {
    return gptq_gemm_rdna2_prefill(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                   use_v2_format);
  }

  const int gi = group_index(group_size);
  const GemmArgs p{static_cast<const int8_t*>(a_i8.data_ptr()),
                   static_cast<const uint32_t*>(b_q_weight.data_ptr()),
                   static_cast<const uint32_t*>(b_qzeros.data_ptr()),
                   static_cast<const ex::f16_t*>(b_scales.data_ptr()),
                   static_cast<const float*>(a_scale.data_ptr()),
                   static_cast<const int32_t*>(a_asum.data_ptr()),
                   out.data_ptr(),
                   size_m, size_n, size_k, zero_offset,
                   /*split_k=*/1, /*out_f32=*/0};
  if (kLaunchA8LdsK32Ag[gi](p, stream) != 0) {
    return gptq_gemm_rdna2_prefill(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                   use_v2_format);
  }

  TORCH_WARN_ONCE("RDNA2 W4A8 sdot4 path active (config a8_lds_k32_ag)");
  return out;
}
