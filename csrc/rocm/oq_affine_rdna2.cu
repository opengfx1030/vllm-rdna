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

#include <rocblas/rocblas.h>

namespace {
using namespace vllm::oq_affine;

void dense_validate(const torch::Tensor& x, const torch::Tensor& codes,
                    const torch::Tensor& scales, const torch::Tensor& biases,
                    const torch::Tensor& out, int64_t bits, int64_t group_size,
                    int64_t logical_k_start, int64_t code_bit_offset,
                    int64_t metadata_group_start, int64_t output_offset) {
  format(bits, group_size);
  tensor(x, x, at::kHalf, "x");
  tensor(codes, x, at::kInt, "codes");
  tensor(scales, x, at::kBFloat16, "scales");
  tensor(biases, x, at::kBFloat16, "biases");
  tensor(out, x, at::kHalf, "out");
  disjoint(out, {x, codes, scales, biases});
  TORCH_CHECK(
      x.dim() == 2 && codes.dim() == 2 && scales.dim() == 2 &&
          biases.sizes() == scales.sizes() && out.dim() == 2 &&
          scales.size(0) == codes.size(0) && out.size(0) == x.size(0),
      "dense requires x[M,K], codes[N,words], metadata[N,groups], out[M,O]");
  TORCH_CHECK(
      x.size(1) > 0 && logical_k_start >= 0 &&
          logical_k_start <= INT_MAX - x.size(1) &&
          code_bit_offset == (logical_k_start * bits) % 32 &&
          metadata_group_start == logical_k_start / group_size &&
          codes.size(1) == (code_bit_offset + x.size(1) * bits + 31) / 32 &&
          scales.size(1) ==
              (logical_k_start + x.size(1) + group_size - 1) / group_size -
                  metadata_group_start &&
          output_offset >= 0 && output_offset <= out.size(1) &&
          codes.size(0) <= out.size(1) - output_offset,
      "invalid dense global-K shard, metadata interval, or output slice");
}

void blas_check(rocblas_status status) {
  TORCH_CHECK(status == rocblas_status_success,
              "staged dense rocBLAS failure: ", static_cast<int>(status));
}

class BlasPrecisionGuard {
 public:
  explicit BlasPrecisionGuard(rocblas_handle handle) : handle_(handle) {
    blas_check(rocblas_get_pointer_mode(handle_, &pointer_mode_));
    blas_check(rocblas_get_atomics_mode(handle_, &atomics_mode_));
    blas_check(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    const auto status =
        rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    if (status != rocblas_status_success)
      rocblas_set_pointer_mode(handle_, pointer_mode_);
    blas_check(status);
  }
  ~BlasPrecisionGuard() {
    rocblas_set_atomics_mode(handle_, atomics_mode_);
    rocblas_set_pointer_mode(handle_, pointer_mode_);
  }
  BlasPrecisionGuard(const BlasPrecisionGuard&) = delete;
  BlasPrecisionGuard& operator=(const BlasPrecisionGuard&) = delete;

 private:
  rocblas_handle handle_;
  rocblas_pointer_mode pointer_mode_;
  rocblas_atomics_mode atomics_mode_;
};

__global__ void dense_reconstruct_kernel(
    const uint32_t* codes, const uint16_t* scales, const uint16_t* biases,
    half* workspace, int64_t count, int width, int words, int groups, int bits,
    int group, int logical_start, int bit_offset, int metadata_start) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / width;
    const int col = i % width;
    const uint32_t q =
        code(codes + row * words, words,
             static_cast<int64_t>(col) * bits + bit_offset, bits);
    const int64_t g = row * groups +
                      (static_cast<int64_t>(logical_start) + col) / group -
                      metadata_start;
    workspace[i] = reconstruct(q, scales[g], biases[g]);
  }
}

template <int BM, int TN>
void dense_launch(const torch::Tensor& x, const torch::Tensor& codes,
                  const torch::Tensor& scales, const torch::Tensor& biases,
                  torch::Tensor& out, int bits, int group, int logical_start,
                  int bit_offset, int metadata_start, int output_offset,
                  hipStream_t stream) {
  projection<BM, TN, false, false, int32_t>
      <<<dim3((x.size(0) + BM - 1) / BM, (codes.size(0) + TN - 1) / TN),
         dim3(32, TN), 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(codes.data_ptr<int32_t>()),
          reinterpret_cast<const uint16_t*>(scales.data_ptr()),
          reinterpret_cast<const uint16_t*>(biases.data_ptr()), out.data_ptr(),
          nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, 1, x.size(0),
          codes.size(0), x.size(1), codes.size(1), scales.size(1), bits, group,
          logical_start, bit_offset, metadata_start, out.size(1), output_offset,
          0);
}

template <typename Id>
__global__ void embedding_kernel(const Id* ids, const uint32_t* codes,
                                 const uint16_t* scales, const uint16_t* biases,
                                 half* out, int64_t count, int rows, int width,
                                 int words, int groups, int bits, int group) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count * width) return;
  const int64_t row = static_cast<int64_t>(ids[i / width]);
  const int col = i % width;
  if (row < 0 || row >= rows) {
    out[i] = __float2half_rn(0.f);  // Including affine bias: no table read.
    return;
  }
  const uint32_t q =
      code(codes + row * words, words, static_cast<int64_t>(col) * bits, bits);
  const int64_t g = row * groups + col / group;
  out[i] = reconstruct(q, scales[g], biases[g]);
}

__global__ void silu_kernel(const float* gate_up, half* act, int64_t rows,
                            int width) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const int64_t start = (i / width) * (2LL * width) + i % width;
  const float gate = gate_up[start];
  const float activated = gate / (1.f + expf(-gate));
  const float product = activated * gate_up[start + width];
  act[i] = __float2half_rn(product);
}

__global__ void reduce_kernel(const float* slots, const float* weights,
                              half* out, int64_t rows, int topk, int width,
                              bool multiply) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const int64_t row = i / width;
  const int col = i % width;
  float sum = 0.f;
  // Ascending route slots, separate router multiply and sum rounding.
  for (int slot = 0; slot < topk; ++slot) {
    const int64_t route = row * topk + slot;
    float value = slots[route * width + col];
    if (multiply) value = value * weights[route];
    sum = sum + value;
  }
  out[i] = __float2half_rn(sum);
}
}  // namespace

void oq_affine_linear_rdna2(const torch::Tensor& x, const torch::Tensor& codes,
                            const torch::Tensor& scales,
                            const torch::Tensor& biases, torch::Tensor& out,
                            int64_t bits, int64_t group_size,
                            int64_t logical_k_start, int64_t code_bit_offset,
                            int64_t metadata_group_start, int64_t output_offset,
                            int64_t block_m, int64_t tile_n) {
  using namespace vllm::oq_affine;
  dense_validate(x, codes, scales, biases, out, bits, group_size,
                 logical_k_start, code_bit_offset, metadata_group_start,
                 output_offset);
  // Dynamo may freeze Python's symbolic-M branch while capturing small graphs.
  // Select the existing direct-M1 policy from actual tensor metadata instead.
  const bool direct = x.size(0) <= 32;
  TORCH_CHECK(block_m != 1 || direct, "direct dense execution requires M<=32");
  if (direct)
    TORCH_CHECK(block_m == 1 || block_m == 2 || block_m == 4 || block_m == 8,
                "unsupported native affine tile geometry");
  geometry(block_m, tile_n, direct);
  if (!x.size(0) || !codes.size(0)) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define DENSE(BM, TN)                                                          \
  dense_launch<BM, TN>(x, codes, scales, biases, out, bits, group_size,        \
                       logical_k_start, code_bit_offset, metadata_group_start, \
                       output_offset, stream)
  if (direct) {
    if (tile_n == 4) {
      DENSE(1, 4);
    } else if (tile_n == 8) {
      DENSE(1, 8);
    } else {
      DENSE(1, 16);
    }
  } else if (block_m == 2) {
    if (tile_n == 4) {
      DENSE(2, 4);
    } else {
      DENSE(2, 8);
    }
  } else if (block_m == 4) {
    if (tile_n == 4) {
      DENSE(4, 4);
    } else {
      DENSE(4, 8);
    }
  } else {
    if (tile_n == 4) {
      DENSE(8, 4);
    } else {
      DENSE(8, 8);
    }
  }
#undef DENSE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void oq_affine_linear_staged_rdna2(
    const torch::Tensor& x, const torch::Tensor& codes,
    const torch::Tensor& scales, const torch::Tensor& biases,
    torch::Tensor& out, torch::Tensor& workspace, int64_t bits,
    int64_t group_size, int64_t logical_k_start, int64_t code_bit_offset,
    int64_t metadata_group_start, int64_t output_offset) {
  using namespace vllm::oq_affine;
  dense_validate(x, codes, scales, biases, out, bits, group_size,
                 logical_k_start, code_bit_offset, metadata_group_start,
                 output_offset);
  tensor(workspace, x, at::kHalf, "workspace");
  TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == codes.size(0) &&
                  workspace.size(1) == x.size(1),
              "staged dense requires exact FP16 workspace[N,K]");
  disjoint(workspace, {x, codes, scales, biases, out});
  if (!x.size(0) || !codes.size(0)) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (workspace.numel() + 255) / 256;
  const unsigned int grid =
      static_cast<unsigned int>(blocks < 65535 ? blocks : 65535);
  dense_reconstruct_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(codes.data_ptr<int32_t>()),
      reinterpret_cast<const uint16_t*>(scales.data_ptr()),
      reinterpret_cast<const uint16_t*>(biases.data_ptr()),
      reinterpret_cast<half*>(workspace.data_ptr()), workspace.numel(),
      x.size(1), codes.size(1), scales.size(1), bits, group_size,
      logical_k_start, code_bit_offset, metadata_group_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Use ATen's pooled handle and its current-stream library workspace, not an
  // expanded-weight cache or an allocated GEMM result. Warm up BLAS before
  // graph capture, as for torch.mm. No global precision/backend flags change.
  const auto handle =
      reinterpret_cast<rocblas_handle>(at::cuda::getCurrentCUDABlasHandle());
  const BlasPrecisionGuard precision(handle);
  const float alpha = 1.f, beta = 0.f;
  auto* destination = out.data_ptr<at::Half>() + output_offset;
  // Column-major GEMM computes out_slice^T = workspace * x^T. The physical
  // leading dimension is the full output width, including untouched segments.
  // f32 compute selects rocBLAS HPA; disallow atomics so split reductions
  // cannot accumulate into FP16 output. FP16 conversion is final storage only.
  // No globally configurable ATen GEMM or alternate (BF16) implementation.
  blas_check(rocblas_gemm_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none,
      codes.size(0), x.size(0), x.size(1), &alpha, workspace.data_ptr(),
      rocblas_datatype_f16_r, x.size(1), x.data_ptr(), rocblas_datatype_f16_r,
      x.size(1), &beta, destination, rocblas_datatype_f16_r, out.size(1),
      destination, rocblas_datatype_f16_r, out.size(1), rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_none));
}

void oq_affine_embedding_rdna2(const torch::Tensor& ids,
                               const torch::Tensor& codes,
                               const torch::Tensor& scales,
                               const torch::Tensor& biases, torch::Tensor& out,
                               int64_t bits, int64_t group_size,
                               int64_t embedding_dim) {
  using namespace vllm::oq_affine;
  format(bits, group_size);
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong,
              "ids must be int32 or int64");
  tensor(ids, ids, ids.scalar_type(), "ids");
  tensor(codes, ids, at::kInt, "codes");
  tensor(scales, ids, at::kBFloat16, "scales");
  tensor(biases, ids, at::kBFloat16, "biases");
  tensor(out, ids, at::kHalf, "out");
  disjoint(out, {ids, codes, scales, biases});
  TORCH_CHECK(
      embedding_dim > 0 && embedding_dim <= INT_MAX && codes.dim() == 2 &&
          scales.dim() == 2 && biases.sizes() == scales.sizes() &&
          scales.size(0) == codes.size(0) &&
          codes.size(1) == (embedding_dim * bits + 31) / 32 &&
          scales.size(1) == (embedding_dim + group_size - 1) / group_size &&
          out.numel() == ids.numel() * embedding_dim,
      "invalid compact embedding geometry");
  if (!out.numel()) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(ids));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LOOKUP(ID)                                                         \
  embedding_kernel<ID><<<(out.numel() + 255) / 256, 256, 0, stream>>>(     \
      ids.data_ptr<ID>(),                                                  \
      reinterpret_cast<const uint32_t*>(codes.data_ptr<int32_t>()),        \
      reinterpret_cast<const uint16_t*>(scales.data_ptr()),                \
      reinterpret_cast<const uint16_t*>(biases.data_ptr()),                \
      reinterpret_cast<half*>(out.data_ptr()), ids.numel(), codes.size(0), \
      embedding_dim, codes.size(1), scales.size(1), bits, group_size)
  if (ids.scalar_type() == at::kLong) {
    LOOKUP(int64_t);
  } else {
    LOOKUP(int32_t);
  }
#undef LOOKUP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void oq_silu_mul_rdna2(const torch::Tensor& gate_up, torch::Tensor& act) {
  using namespace vllm::oq_affine;
  tensor(gate_up, gate_up, at::kFloat, "gate_up");
  tensor(act, gate_up, at::kHalf, "act");
  disjoint(act, {gate_up});
  TORCH_CHECK(gate_up.dim() == 2 && act.dim() == 2 &&
                  gate_up.size(0) == act.size(0) &&
                  gate_up.size(1) == 2 * act.size(1),
              "SiLU requires FP32 gate_up[R,2H], FP16 act[R,H]");
  if (!act.numel()) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(act));
  const auto stream = at::cuda::getCurrentCUDAStream();
  silu_kernel<<<(act.numel() + 255) / 256, 256, 0, stream>>>(
      gate_up.data_ptr<float>(), reinterpret_cast<half*>(act.data_ptr()),
      act.size(0), act.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void oq_moe_reduce_rdna2(const torch::Tensor& slots,
                         const torch::Tensor& topk_weights,
                         torch::Tensor& output, bool mul_router_weight) {
  using namespace vllm::oq_affine;
  tensor(slots, slots, at::kFloat, "slots");
  tensor(topk_weights, slots, at::kFloat, "topk_weights");
  tensor(output, slots, at::kHalf, "output");
  disjoint(output, {slots, topk_weights});
  TORCH_CHECK(
      slots.dim() == 3 && topk_weights.dim() == 2 && output.dim() == 2 &&
          slots.size(0) == topk_weights.size(0) &&
          slots.size(1) == topk_weights.size(1) &&
          slots.size(0) == output.size(0) && slots.size(2) == output.size(1) &&
          slots.size(1) > 0,
      "reduce requires FP32 slots[M,T,N], weights[M,T], FP16 output[M,N]");
  if (!output.numel()) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  const auto stream = at::cuda::getCurrentCUDAStream();
  reduce_kernel<<<(output.numel() + 255) / 256, 256, 0, stream>>>(
      slots.data_ptr<float>(), topk_weights.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr()), output.size(0), slots.size(1),
      output.size(1), mul_router_weight);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
