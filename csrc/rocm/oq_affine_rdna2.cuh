// SPDX-License-Identifier: MIT
// Packed extraction and wave32 tiling adapted from lablup/mlxcel
// 955335a38e13856ad205cc17150f261e40434d5a, qdequant.hpp and
// qmv_tiled_kernel.hip. The donor's factored affine arithmetic is NOT used.
//
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

#pragma once

#include <cstdint>
#include <climits>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace vllm::oq_affine {

inline void tensor(const torch::Tensor& t, const torch::Tensor& reference,
                   at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == reference.device() &&
                  t.scalar_type() == dtype && t.is_contiguous(),
              name, " must be contiguous, on the input GPU, with dtype ",
              dtype);
  for (auto extent : t.sizes())
    TORCH_CHECK(extent <= INT_MAX, name, " extent exceeds int32 indexing");
}

inline void disjoint(const torch::Tensor& out,
                     std::initializer_list<torch::Tensor> inputs) {
  for (const auto& input : inputs) at::assert_no_overlap(out, input);
}

inline void format(int64_t bits, int64_t group) {
  TORCH_CHECK(
      (bits == 6 || bits == 8) && (group == 32 || group == 64 || group == 128),
      "native affine requires bits 6/8 and group size 32/64/128");
}

inline void geometry(int64_t bm, int64_t tn, bool direct = false) {
  TORCH_CHECK(direct
                  ? (tn == 4 || tn == 8 || tn == 16)
                  : ((bm == 2 || bm == 4 || bm == 8) && (tn == 4 || tn == 8)),
              "unsupported native affine tile geometry");
}

__device__ __forceinline__ float bf16_float(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ half reconstruct(uint32_t q, uint16_t s,
                                            uint16_t b) {
  // Strict per-source flags preserve two separately rounded FP32 operations:
  // no fast math, no contraction, IEEE denormals; no volatile memory traffic.
  const float product = bf16_float(s) * static_cast<float>(q);
  const float affine = product + bf16_float(b);
  return __float2half_rn(affine);
}

__device__ __forceinline__ uint32_t code(const uint32_t* row, int words,
                                         int64_t bit, int bits) {
  const int word = bit >> 5;
  const int shift = bit & 31;
  if (word < 0 || word >= words) return 0u;
  uint32_t value = row[word] >> shift;
  if (shift + bits > 32 && word + 1 < words)
    value |= row[word + 1] << (32 - shift);
  return value & ((1u << bits) - 1u);
}

template <int Bits>
__device__ __forceinline__ void weights16(const uint32_t* row,
                                          const uint16_t* scales,
                                          const uint16_t* biases, int words,
                                          int k0, int k, int group,
                                          int logical_start, int bit_offset,
                                          int metadata_start, half (&w)[16]) {
  const int group_shift = group == 32 ? 5 : group == 64 ? 6 : 7;
  const int64_t bit = static_cast<int64_t>(k0) * Bits + bit_offset;
  const int64_t logical_k0 = static_cast<int64_t>(logical_start) + k0;
  // Keep the complete same-group path free of per-weight tail/group branches.
  // Only metadata is shared; reconstruct preserves each FP32 multiply/add.
  if (k0 <= k - 16 && (bit & 31) == 0 &&
      (logical_k0 >> group_shift) == ((logical_k0 + 15) >> group_shift)) {
    const int word = bit >> 5;
    uint32_t packed[4];
    // Six-bit lanes consume precisely three words, never four.
    packed[0] = row[word];
    packed[1] = row[word + 1];
    packed[2] = row[word + 2];
    if constexpr (Bits == 8) packed[3] = row[word + 3];
    const int64_t g = (logical_k0 >> group_shift) - metadata_start;
    const uint16_t scale = scales[g];
    const uint16_t bias = biases[g];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int shift = i * Bits;
      uint32_t q = packed[shift >> 5] >> (shift & 31);
      if ((shift & 31) + Bits > 32)
        q |= packed[(shift >> 5) + 1] << (32 - (shift & 31));
      q &= (1u << Bits) - 1u;
      w[i] = reconstruct(q, scale, bias);
    }
    return;
  }
  // Partial blocks, unaligned packed words and group crossings stay generic.
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (k0 + i < k) {
      const uint32_t q = code(row, words, bit + i * Bits, Bits);
      const int64_t g = ((logical_k0 + i) >> group_shift) - metadata_start;
      w[i] = reconstruct(q, scales[g], biases[g]);
    } else {
      w[i] = __float2half_rn(0.f);
    }
  }
}

__device__ __forceinline__ float dot2(half2 w, half2 x, float acc) {
#if defined(__gfx1030__)
  return __builtin_amdgcn_fdot2(w, x, acc, false);
#else
  // Functional compilation on other device images; native dispatch is RDNA2.
  acc += __half2float(__low2half(w)) * __half2float(__low2half(x));
  return acc + __half2float(__high2half(w)) * __half2float(__high2half(x));
#endif
}

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_xor(value, offset, 32);
  return value;
}

// One wave per output column; reconstruct each K tile once for BM rows.
// Aligned expert IDs are LOCAL, as returned by moe_align_block_size with map.
// Direct output rows and aligned scatter destinations are original route slots.
template <int BM, int TN, bool Routed, bool Aligned, typename Id = int32_t>
__global__ __launch_bounds__(32 * TN) void projection(
    const half* x, const uint32_t* codes, const uint16_t* scales,
    const uint16_t* biases, void* output, const Id* topk, const int32_t* map,
    const int32_t* sorted, const int32_t* experts, const int32_t* post_pad,
    int routes, int input_divisor, int global_experts, int local_experts, int m,
    int n, int k, int words, int groups, int bits, int group, int logical_start,
    int bit_offset, int metadata_start, int out_width, int out_offset,
    int sorted_capacity) {
  constexpr int BK = 512;
  __shared__ half xs[BM][BK];
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int col = blockIdx.y * TN + threadIdx.y;
  int rows[BM];
  bool live[BM];
  int expert = 0;
  if constexpr (Aligned) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * BM;
    if (start >= sorted_capacity || start >= post_pad[0]) return;
    expert = experts[blockIdx.x];
    if (expert < 0 || expert >= local_experts) return;
  }
#pragma unroll
  for (int r = 0; r < BM; ++r) {
    if constexpr (Routed) {
      const int64_t position = static_cast<int64_t>(blockIdx.x) * BM + r;
      int route;
      if constexpr (Aligned)
        route = position < sorted_capacity && position < post_pad[0]
                    ? sorted[position]
                    : -1;
      else
        route = static_cast<int>(position);
      live[r] = route >= 0 && route < routes;
      int local = -1;
      if (live[r]) {
        const int64_t global = static_cast<int64_t>(topk[route]);
        if (global >= 0 && global < global_experts)
          local = map ? map[global] : static_cast<int>(global);
      }
      live[r] = live[r] && local >= 0 && local < local_experts;
      if constexpr (Aligned)
        live[r] = live[r] && local == expert;
      else
        expert = local;
      rows[r] = route;
    } else {
      rows[r] = blockIdx.x * BM + r;
      live[r] = rows[r] < m;
    }
  }
  float sums[BM] = {};
  for (int base = 0; base < k; base += BK) {
    __syncthreads();
    for (int i = tid; i < BM * BK; i += TN * 32) {
      const int r = i / BK, j = i % BK;
      xs[r][j] = live[r] && base + j < k
                     ? x[static_cast<int64_t>(Routed ? rows[r] / input_divisor
                                                     : rows[r]) *
                             k +
                         base + j]
                     : __float2half_rn(0.f);
    }
    __syncthreads();
    if (col < n && expert >= 0 && expert < local_experts) {
      const int64_t row = static_cast<int64_t>(expert) * n + col;
      half w[16];
      if (bits == 6)
        weights16<6>(codes + row * words, scales + row * groups,
                     biases + row * groups, words, base + lane * 16, k, group,
                     logical_start, bit_offset, metadata_start, w);
      else
        weights16<8>(codes + row * words, scales + row * groups,
                     biases + row * groups, words, base + lane * 16, k, group,
                     logical_start, bit_offset, metadata_start, w);
#pragma unroll
      for (int r = 0; r < BM; ++r) {
        if (live[r]) {
#pragma unroll
          for (int p = 0; p < 8; ++p)
            sums[r] = dot2(__halves2half2(w[2 * p], w[2 * p + 1]),
                           __halves2half2(xs[r][lane * 16 + 2 * p],
                                          xs[r][lane * 16 + 2 * p + 1]),
                           sums[r]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < BM; ++r) {
    const float sum = wave_sum(sums[r]);
    if (lane == 0 && col < n && live[r]) {
      const int64_t index =
          static_cast<int64_t>(rows[r]) * out_width + out_offset + col;
      if constexpr (Routed)
        static_cast<float*>(output)[index] = sum;
      else
        static_cast<half*>(output)[index] = __float2half_rn(sum);
    }
  }
}

}  // namespace vllm::oq_affine

// Public mutation-only ABI. Only the named output storage is mutated.
// All inputs must be disjoint from the output; shared read-only storage is
// permitted.
void oq_affine_linear_rdna2(const torch::Tensor& x, const torch::Tensor& codes,
                            const torch::Tensor& scales,
                            const torch::Tensor& biases, torch::Tensor& out,
                            int64_t bits, int64_t group_size,
                            int64_t logical_k_start, int64_t code_bit_offset,
                            int64_t metadata_group_start, int64_t output_offset,
                            int64_t block_m, int64_t tile_n);
void oq_affine_embedding_rdna2(const torch::Tensor& ids,
                               const torch::Tensor& codes,
                               const torch::Tensor& scales,
                               const torch::Tensor& biases, torch::Tensor& out,
                               int64_t bits, int64_t group_size,
                               int64_t embedding_dim);
void moe_oq_affine_gemm_rdna2(
    const torch::Tensor& x, const torch::Tensor& codes,
    const torch::Tensor& scales, const torch::Tensor& biases,
    const torch::Tensor& topk_ids, const torch::Tensor& expert_map,
    const torch::Tensor& sorted_ids, const torch::Tensor& expert_ids,
    const torch::Tensor& post_pad, torch::Tensor& out, int64_t bits,
    int64_t group_size, int64_t input_divisor, int64_t block_m, int64_t tile_n);
void oq_silu_mul_rdna2(const torch::Tensor& gate_up, torch::Tensor& act);
void oq_moe_reduce_rdna2(const torch::Tensor& slots,
                         const torch::Tensor& topk_weights,
                         torch::Tensor& output, bool mul_router_weight);
