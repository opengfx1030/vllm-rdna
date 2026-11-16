// SPDX-License-Identifier: MIT
// Packed extraction/wave32 tiling derived from lablup/mlxcel
// 955335a38e13856ad205cc17150f261e40434d5a.
// Copyright (c) 2023 Apple Inc.
// Copyright (c) the MLX and NripeshN/mlx contributors.
// Copyright (c) 2025-2026 Lablup Inc. and contributors.
// Copyright contributors to the vLLM project.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "oq_affine_rdna2.cuh"

namespace {
using namespace vllm::oq_affine;

template <int BM, int TN, bool Aligned, typename Id>
void launch(const torch::Tensor& x, const torch::Tensor& codes,
            const torch::Tensor& scales, const torch::Tensor& biases,
            const torch::Tensor& topk, const torch::Tensor& map,
            const torch::Tensor& sorted, const torch::Tensor& experts,
            const torch::Tensor& post_pad, torch::Tensor& out, int bits,
            int group, int divisor, hipStream_t stream) {
  const int64_t blocks = Aligned ? experts.numel() : topk.numel();
  projection<BM, TN, true, Aligned, Id>
      <<<dim3(blocks, (codes.size(1) + TN - 1) / TN), dim3(32, TN), 0,
         stream>>>(reinterpret_cast<const half*>(x.data_ptr()),
                   reinterpret_cast<const uint32_t*>(codes.data_ptr<int32_t>()),
                   reinterpret_cast<const uint16_t*>(scales.data_ptr()),
                   reinterpret_cast<const uint16_t*>(biases.data_ptr()),
                   out.data_ptr(), topk.data_ptr<Id>(),
                   map.numel() ? map.data_ptr<int32_t>() : nullptr,
                   Aligned ? sorted.data_ptr<int32_t>() : nullptr,
                   Aligned ? experts.data_ptr<int32_t>() : nullptr,
                   Aligned ? post_pad.data_ptr<int32_t>() : nullptr,
                   topk.numel(), divisor, 512, codes.size(0), x.size(0),
                   codes.size(1), x.size(1), codes.size(2), scales.size(2),
                   bits, group, 0, 0, 0, out.size(1), 0, sorted.numel());
}

template <typename Id>
void dispatch(const torch::Tensor& x, const torch::Tensor& codes,
              const torch::Tensor& scales, const torch::Tensor& biases,
              const torch::Tensor& topk, const torch::Tensor& map,
              const torch::Tensor& sorted, const torch::Tensor& experts,
              const torch::Tensor& post_pad, torch::Tensor& out, int bits,
              int group, int divisor, int bm, int tn, bool aligned,
              hipStream_t stream) {
#define RUN(BM, TN, ALIGNED)                                                \
  launch<BM, TN, ALIGNED, Id>(x, codes, scales, biases, topk, map, sorted,  \
                              experts, post_pad, out, bits, group, divisor, \
                              stream)
  if (!aligned) {
    if (tn == 4) {
      RUN(1, 4, false);
    } else if (tn == 8) {
      RUN(1, 8, false);
    } else {
      RUN(1, 16, false);
    }
  } else if (bm == 2) {
    if (tn == 4) {
      RUN(2, 4, true);
    } else {
      RUN(2, 8, true);
    }
  } else if (bm == 4) {
    if (tn == 4) {
      RUN(4, 4, true);
    } else {
      RUN(4, 16, true);
    }
  } else {
    if (tn == 4) {
      RUN(8, 4, true);
    } else {
      RUN(8, 8, true);
    }
  }
#undef RUN
}
}  // namespace

void moe_oq_affine_gemm_rdna2(
    const torch::Tensor& x, const torch::Tensor& codes,
    const torch::Tensor& scales, const torch::Tensor& biases,
    const torch::Tensor& topk_ids, const torch::Tensor& expert_map,
    const torch::Tensor& sorted_ids, const torch::Tensor& expert_ids,
    const torch::Tensor& post_pad, torch::Tensor& out, int64_t bits,
    int64_t group_size, int64_t input_divisor, int64_t block_m,
    int64_t tile_n) {
  using namespace vllm::oq_affine;
  format(bits, group_size);
  tensor(x, x, at::kHalf, "x");
  tensor(codes, x, at::kInt, "codes");
  tensor(scales, x, at::kBFloat16, "scales");
  tensor(biases, x, at::kBFloat16, "biases");
  tensor(out, x, at::kFloat, "out");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
      "topk_ids must be int32 or int64");
  tensor(topk_ids, x, topk_ids.scalar_type(), "topk_ids");
  tensor(expert_map, x, at::kInt, "expert_map");
  tensor(sorted_ids, x, at::kInt, "sorted_ids");
  tensor(expert_ids, x, at::kInt, "expert_ids");
  tensor(post_pad, x, at::kInt, "post_pad");
  disjoint(out, {x, codes, scales, biases, topk_ids, expert_map, sorted_ids,
                 expert_ids, post_pad});
  TORCH_CHECK(x.dim() == 2 && codes.dim() == 3 && scales.dim() == 3 &&
                  biases.sizes() == scales.sizes() && out.dim() == 2 &&
                  topk_ids.dim() == 2 && topk_ids.size(1) > 0 &&
                  expert_map.dim() == 1 && sorted_ids.dim() == 1 &&
                  expert_ids.dim() == 1 && post_pad.dim() == 1,
              "invalid native affine MoE tensor ranks");
  TORCH_CHECK(codes.size(0) <= 512 &&
                  (expert_map.numel() == 512 ||
                   (expert_map.numel() == 0 && codes.size(0) == 512)),
              "EP needs the real global512 expert map; empty map requires "
              "all512 experts");
  TORCH_CHECK(
      topk_ids.numel() <= INT_MAX && sorted_ids.numel() <= INT_MAX &&
          x.size(1) > 0 &&
          (input_divisor == 1 || input_divisor == topk_ids.size(1)) &&
          x.size(0) * input_divisor == topk_ids.numel() &&
          out.size(0) == topk_ids.numel() && out.size(1) == codes.size(1) &&
          scales.size(0) == codes.size(0) && scales.size(1) == codes.size(1) &&
          codes.size(2) == (x.size(1) * bits + 31) / 32 &&
          scales.size(2) == (x.size(1) + group_size - 1) / group_size,
      "MoE requires codes[E,N,words], BF16 metadata[E,N,groups], "
      "FP16 x[routes/input_divisor,K], FP32 out[routes,N]");
  const bool aligned = sorted_ids.numel() != 0;
  geometry(block_m, tile_n, !aligned);
  if (aligned) {
    TORCH_CHECK(
        post_pad.numel() == 1 &&
            expert_ids.numel() == (sorted_ids.numel() + block_m - 1) / block_m,
        "alignment buffers do not cover the selected block_m");
  } else {
    TORCH_CHECK(expert_ids.numel() == 0 && post_pad.numel() == 0 &&
                    topk_ids.size(0) <= 32,
                "direct routing requires M<=32 and empty alignment buffers");
  }
  if (!out.numel()) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Alignment deliberately omits nonlocal routes; zero every destination on
  // every invocation. This capture-legal node belongs to the caller's graph.
  const hipError_t err =
      hipMemsetAsync(out.data_ptr(), 0, out.numel() * sizeof(float), stream);
  TORCH_CHECK(err == hipSuccess, "native affine MoE output zero failed: ", err);
#define DISPATCH(ID)                                                       \
  dispatch<ID>(x, codes, scales, biases, topk_ids, expert_map, sorted_ids, \
               expert_ids, post_pad, out, bits, group_size, input_divisor, \
               block_m, tile_n, aligned, stream)
  if (topk_ids.scalar_type() == at::kLong) {
    DISPATCH(int64_t);
  } else {
    DISPATCH(int32_t);
  }
#undef DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
