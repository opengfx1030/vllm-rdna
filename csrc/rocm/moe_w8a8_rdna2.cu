// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W8A8 (int8 weights x int8 activations) kernel for AMD RDNA2
// (gfx1030), fp16 only.
//
// The MoE half of the W8A8 sdot4 path, for compressed-tensors W8A8 int8
// expert checkpoints (CompressedTensorsW8A8Int8RDNA2MoEMethod). Same
// output contract as moe_w4a8_gemm_rdna2 / moe_gptq_gemm_rdna2: it
// accumulates into the caller's pre-zeroed ``c`` with the router weight
// folded and ``output_topk`` reducing rows. The activation batch is
// quantized once per call by the W4A8 act_quant kernel; the weights are the
// [E, K/4, N] int32 layout rdna2_w8a8.pack_moe_weight produces (see
// w8a8_sdot4_rdna2.cuh for the contract).
//
// There is no int8 fallback for this weight layout, so every ineligible
// shape is a TORCH_CHECK rather than a silent fallback.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "moe_accum_rdna2.cuh"
#include "ops.h"
#include "w8a8_sdot4_rdna2.cuh"

namespace ex = vllm::explore_w4a8;
namespace w8 = vllm::w8a8_rdna2;

namespace {

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

bool valid_act_group(int64_t a_group) {
  return a_group == 32 || a_group == 64 || a_group == 128;
}

bool w8a8_moe_debug_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("VLLM_RDNA2_W8A8_MOE_DEBUG");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

void w8a8_moe_log_shape(int m, int k, int n, int a_group, int block_m) {
  static std::mutex mu;
  static std::map<std::tuple<int, int, int, int, int>, bool> seen;
  const std::tuple<int, int, int, int, int> key{m, k, n, a_group, block_m};
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.emplace(key, true).second) {
      return;
    }
  }
  fprintf(stderr, "[W8A8-MOE-DEBUG] M=%d K=%d N=%d a_group=%d bsm=%d\n", m, k,
          n, a_group, block_m);
  fflush(stderr);
}

struct MoeArgs {
  const int8_t* a_i8;
  const float* a_scale;
  const uint32_t* w;
  const ex::f16_t* scales;
  void* c;  // accumulator: fp16 (CAS) or fp32 (scratch) by C_T
  const float* topk_weights;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_post_padded;
  int num_token_blocks;
  int size_m;
  int size_n;
  int size_k;
  int w_group;
  int top_k;
  bool mul_topk_weight;
  int output_topk;
};

template <int BLOCK_M, int AG, typename C_T>
void launch_kernel(const MoeArgs& p, hipStream_t stream) {
  const dim3 grid(p.num_token_blocks, (p.size_n + w8::kNTile - 1) / w8::kNTile,
                  (p.size_k + w8::kKWindow - 1) / w8::kKWindow);
  w8::moe_w8a8_gemm_kernel<BLOCK_M, AG, C_T>
      <<<grid, dim3(w8::kThreads), 0, stream>>>(
          p.a_i8, p.a_scale, p.w, p.scales, static_cast<C_T*>(p.c),
          p.topk_weights, p.sorted_token_ids, p.expert_ids,
          p.num_tokens_post_padded, p.size_m, p.size_n, p.size_k, p.w_group,
          p.top_k, p.mul_topk_weight, p.output_topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BLOCK_M, typename C_T>
void dispatch_group(int64_t a_group, const MoeArgs& p, hipStream_t stream) {
  switch (a_group) {
    case 32:
      launch_kernel<BLOCK_M, 32, C_T>(p, stream);
      break;
    case 64:
      launch_kernel<BLOCK_M, 64, C_T>(p, stream);
      break;
    default:
      launch_kernel<BLOCK_M, 128, C_T>(p, stream);
      break;
  }
}

template <typename C_T>
void dispatch_block_m(int64_t block_size_m, int64_t a_group, const MoeArgs& p,
                      hipStream_t stream) {
  switch (block_size_m) {
    case 1:
      dispatch_group<1, C_T>(a_group, p, stream);
      break;
    case 2:
      dispatch_group<2, C_T>(a_group, p, stream);
      break;
    case 4:
      dispatch_group<4, C_T>(a_group, p, stream);
      break;
    default:
      dispatch_group<8, C_T>(a_group, p, stream);
      break;
  }
}

}  // namespace

void moe_w8a8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, int64_t a_group,
                         bool fp32_accum) {
  TORCH_CHECK(on_gfx1030(), "moe_w8a8_gemm_rdna2: gfx1030 only");
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == at::kHalf &&
                  a.stride(1) == 1,
              "moe_w8a8_gemm_rdna2: a must be a fp16 [M, K] HIP tensor with "
              "contiguous rows");
  TORCH_CHECK(c.is_cuda() && c.dim() == 2 && c.scalar_type() == at::kHalf &&
                  c.is_contiguous(),
              "moe_w8a8_gemm_rdna2: c must be a contiguous fp16 2D tensor");
  const int64_t size_m = a.size(0);
  const int64_t size_k = a.size(1);
  TORCH_CHECK(size_m > 0 && size_k > 0 && size_k % 32 == 0,
              "moe_w8a8_gemm_rdna2: K must be a positive multiple of 32");
  TORCH_CHECK(a.stride(0) % 8 == 0 && a.stride(0) >= size_k,
              "moe_w8a8_gemm_rdna2: a rows must be 16-byte aligned");
  TORCH_CHECK(b_q_weight.is_cuda() && b_q_weight.dim() == 3 &&
                  b_q_weight.scalar_type() == at::kInt &&
                  b_q_weight.is_contiguous() &&
                  b_q_weight.size(1) * 4 == size_k,
              "moe_w8a8_gemm_rdna2: b_q_weight must be int32 [E, K/4, N] "
              "contiguous");
  const int64_t num_experts = b_q_weight.size(0);
  const int64_t size_n = b_q_weight.size(2);
  TORCH_CHECK(size_n > 0 && size_n % w8::kNPerThread == 0,
              "moe_w8a8_gemm_rdna2: N must be a positive multiple of 4");
  TORCH_CHECK(c.size(1) == size_n,
              "moe_w8a8_gemm_rdna2: c must have N columns");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.dim() == 3 &&
                  b_scales.scalar_type() == at::kHalf &&
                  b_scales.is_contiguous() && b_scales.size(0) == num_experts &&
                  b_scales.size(1) > 0 && size_k % b_scales.size(1) == 0 &&
                  b_scales.size(2) == size_n,
              "moe_w8a8_gemm_rdna2: b_scales must be fp16 [E, K/WG, N] "
              "contiguous");
  const int64_t w_group = size_k / b_scales.size(1);
  TORCH_CHECK(valid_act_group(a_group),
              "moe_w8a8_gemm_rdna2: a_group must be 32, 64 or 128");
  TORCH_CHECK(size_k % a_group == 0 && w_group % a_group == 0,
              "moe_w8a8_gemm_rdna2: a_group must divide K and the weight "
              "group");
  TORCH_CHECK(block_size_m == 1 || block_size_m == 2 || block_size_m == 4 ||
                  block_size_m == 8,
              "moe_w8a8_gemm_rdna2: block_size_m must be in {1, 2, 4, 8}");
  TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                  expert_ids.scalar_type() == at::kInt &&
                  num_tokens_post_padded.scalar_type() == at::kInt,
              "moe_w8a8_gemm_rdna2: routing tensors must be int32");
  TORCH_CHECK(
      topk_weights.numel() == 0 || topk_weights.scalar_type() == at::kFloat,
      "moe_w8a8_gemm_rdna2: topk_weights must be fp32 or empty");
  TORCH_CHECK(c.numel() < (int64_t{1} << 31) &&
                  num_experts * (size_k / 4) * size_n < (int64_t{1} << 31),
              "moe_w8a8_gemm_rdna2: tensors must fit 32-bit offsets");

  const at::cuda::OptionalCUDAGuard guard(a.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  // int8 A: per-(token, a_group) quant in the W4A8 tile layout, once for
  // the whole batch. Per-call workspace through the caching allocator, as
  // moe_w4a8_gemm_rdna2 does.
  const int64_t tiles = (size_m + w8::kMTile - 1) / w8::kMTile;
  const int64_t groups = size_k / a_group;
  const auto opts = a.options();
  struct {
    at::Tensor a_i8, a_scale, a_sum;
  } ws{at::empty({tiles, size_k / 8, w8::kMTile, 8}, opts.dtype(at::kChar)),
       at::empty({tiles, groups, w8::kMTile}, opts.dtype(at::kFloat)),
       at::empty({tiles, groups, w8::kMTile}, opts.dtype(at::kInt))};
  ex::w4a8_act_quant_kernel<w8::kThreads, w8::kMTile, /*PER_GROUP=*/true>
      <<<dim3(static_cast<unsigned>(tiles)), dim3(w8::kThreads), 0, stream>>>(
          reinterpret_cast<const ex::f16_t*>(a.data_ptr()), a.stride(0),
          reinterpret_cast<int8_t*>(ws.a_i8.data_ptr()),
          reinterpret_cast<float*>(ws.a_scale.data_ptr()),
          reinterpret_cast<int32_t*>(ws.a_sum.data_ptr()),
          static_cast<int>(size_m), static_cast<int>(size_k),
          static_cast<int>(a_group));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const MoeArgs p{
      reinterpret_cast<const int8_t*>(ws.a_i8.data_ptr()),
      reinterpret_cast<const float*>(ws.a_scale.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      reinterpret_cast<const ex::f16_t*>(b_scales.data_ptr()),
      c.data_ptr(),
      topk_weights.numel() > 0 ? topk_weights.data_ptr<float>() : nullptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(),
      static_cast<int>(sorted_token_ids.size(0) / block_size_m),
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      static_cast<int>(w_group),
      static_cast<int>(top_k),
      mul_topk_weight,
      static_cast<int>(output_topk)};

  if (fp32_accum) {
    // fp32 accumulation: partials land in the cached fp32 scratch, then one
    // elementwise cast rounds to fp16. Shared with the W4A16/W4A8 MoE ops.
    MoeArgs p32 = p;
    float* scratch = moe_fp32_scratch(c.size(0), c.size(1), c.get_device());
    const hipError_t memset_err = hipMemsetAsync(
        scratch, 0, static_cast<size_t>(c.numel()) * sizeof(float), stream);
    TORCH_CHECK(memset_err == hipSuccess,
                "moe_w8a8_gemm_rdna2: scratch memset failed");
    p32.c = scratch;
    dispatch_block_m<float>(block_size_m, a_group, p32, stream);
    moe_cast_f32_to_f16(scratch, reinterpret_cast<half*>(c.data_ptr()),
                        c.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    dispatch_block_m<ex::f16_t>(block_size_m, a_group, p, stream);
  }

  TORCH_WARN_ONCE("RDNA2 W8A8 sdot4 MoE path active");
  if (w8a8_moe_debug_enabled()) {
    w8a8_moe_log_shape(static_cast<int>(size_m), static_cast<int>(size_k),
                       static_cast<int>(size_n), static_cast<int>(a_group),
                       static_cast<int>(block_size_m));
  }
}
