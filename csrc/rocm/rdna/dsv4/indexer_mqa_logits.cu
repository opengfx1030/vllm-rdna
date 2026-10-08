// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged MQA logits kernel for AMD RDNA (gfx1030) -- DeepSeek V4 Lightning
// Indexer decode path. AITER is CDNA-only; this kernel computes the FP8 MQA
// logits that AITER's `paged_mqa_logits` would have produced, using the same
// algorithm as `fp8_paged_mqa_logits_torch` in a single fused kernel.
//
// Layout:
//   q_fp8:        [B, next_n, H, D]    float8_e4m3fn, packed
//   kv_cache:     [num_pages, page_size, 1, D + 4]  uint8, one of
//                 - per-slot (block_flat=false): each slot is D fp8 bytes
//                   followed by its fp32 dequant scale
//                 - block-flat (block_flat=true, DeepSeek-V4 C4A indexer):
//                   page = [page_size * D fp8 bytes][page_size fp32 scales]
//   weights:      [B * next_n, H]      float32 (per-head weight)
//   context_lens: [B]                  int32
//   block_tables: [B, max_blocks]      int32 (row stride may exceed width)
//
// Output: logits [B * next_n, max_model_len] float32, with -inf in padded
// slots (positions >= context_lens[b]).
//
// Algorithm per output row (b, p):
//   for kv in [0, context_lens[b]):
//     score[h] = sum_d q_fp8[b,p,h,d] * fp8_to_fp16(k_fp8[kv,d]) * k_scale[kv]
//     score[h] = relu(score[h]) * weights[b,p,h]
//     logits[b,p,kv] = sum_h score[h]
//
// Parallelization: grid (rows, splits). Split s handles KV tiles
// s, s + splits, ... so a few decode rows still fill the GPU. The split
// count depends only on the launch shape (rows, max_model_len), never on
// context lengths, so the launch is CUDA-graph safe. Output positions of
// different splits are disjoint: no cross-CTA reduction.

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <limits>

#include "../../qdq_fp8_rdna2.cuh"

// One thread block processes one (row, split). Thread h (< N_HEADS) owns
// head h's dot products for a tile of BLOCK_K kv positions; partials go to
// shared memory and every thread then reduces a column across heads.
template <int HEAD_DIM, int N_HEADS, int BLOCK_THREADS, int BLOCK_K>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_mqa_logits_decode_kernel(
    const uint8_t* __restrict__ q_packed,      // [B*next_n, N_HEADS, HEAD_DIM]
    const uint8_t* __restrict__ kv_cache,      // paged
    const float* __restrict__ weights,         // [B*next_n, N_HEADS]
    const int32_t* __restrict__ context_lens,  // [B]
    const int32_t* __restrict__ block_tables,  // [B, bt_stride]
    int32_t block_size, int32_t max_blocks_per_seq, int32_t bt_stride,
    int64_t page_stride,         // bytes per page in kv_cache
    int32_t value_slot_bytes,    // bytes between consecutive slots' values
    int32_t scale_base,          // byte offset of slot 0's scale in a page
    int32_t scale_slot_bytes,    // bytes between consecutive slots' scales
    float* __restrict__ logits,  // [B*next_n, max_model_len]
    int32_t max_model_len, int32_t next_n, int32_t num_pages,
    int32_t debug_oob) {
  static_assert(N_HEADS <= BLOCK_THREADS, "one thread per head");
  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int num_splits = gridDim.y;
  const int b = row / next_n;
  const int tid = threadIdx.x;

  const int32_t seq_len = min(context_lens[b], max_model_len);
  if (seq_len <= 0) return;

  extern __shared__ char smem_raw[];
  uint16_t* q_shared = reinterpret_cast<uint16_t*>(smem_raw);
  float* partial = reinterpret_cast<float*>(q_shared + N_HEADS * HEAD_DIM);

  const int64_t q_row_offset = (int64_t)row * N_HEADS * HEAD_DIM;
  for (int idx = tid; idx < N_HEADS * HEAD_DIM; idx += BLOCK_THREADS) {
    q_shared[idx] = fp8_e4m3_to_fp16_bits(q_packed[q_row_offset + idx]);
  }
  __syncthreads();

  const float w_h = tid < N_HEADS ? weights[(int64_t)row * N_HEADS + tid] : 0.f;
  const uint16_t* q_h = q_shared + (tid < N_HEADS ? tid : 0) * HEAD_DIM;
  const int32_t* bt_row = block_tables + (int64_t)b * bt_stride;
  float* out_row = logits + (int64_t)row * max_model_len;

  for (int tile_start = split * BLOCK_K; tile_start < seq_len;
       tile_start += num_splits * BLOCK_K) {
    const int tile_len = min(BLOCK_K, seq_len - tile_start);
    if (tid < N_HEADS) {
      for (int k = 0; k < tile_len; ++k) {
        const int kv = tile_start + k;
        const int page_idx = kv / block_size;
        const int slot = kv % block_size;
        const int32_t page_id =
            page_idx < max_blocks_per_seq ? bt_row[page_idx] : -1;
        float dot = 0.0f;
        if (page_id >= 0 && page_id < num_pages) {
          const uint8_t* page = kv_cache + (int64_t)page_id * page_stride;
          const uint8_t* kv_vals = page + (int64_t)slot * value_slot_bytes;
          float k_scale;
          __builtin_memcpy(&k_scale,
                           page + scale_base + (int64_t)slot * scale_slot_bytes,
                           4);
#pragma unroll 16
          for (int d = 0; d < HEAD_DIM; ++d) {
            dot += __half2float(__ushort_as_half(q_h[d])) *
                   __half2float(
                       __ushort_as_half(fp8_e4m3_to_fp16_bits(kv_vals[d])));
          }
          dot = fmaxf(dot * k_scale, 0.0f) * w_h;
        } else if (debug_oob && tid == 0) {
          printf(
              "[paged_mqa OOB] row=%d b=%d kv=%d seq_len=%d page_idx=%d "
              "page_id=%d num_pages=%d\n",
              row, b, kv, seq_len, page_idx, page_id, num_pages);
        }
        partial[tid * BLOCK_K + k] = dot;
      }
    }
    __syncthreads();
    for (int k = tid; k < tile_len; k += BLOCK_THREADS) {
      float total = 0.0f;
#pragma unroll 8
      for (int hh = 0; hh < N_HEADS; ++hh) total += partial[hh * BLOCK_K + k];
      out_row[tile_start + k] = total;
    }
    __syncthreads();
  }
}

torch::Tensor paged_mqa_logits_decode_rdna2(
    torch::Tensor q_fp8,         // [B, next_n, H, D] float8_e4m3fn
    torch::Tensor kv_cache,      // paged [num_pages, page_size, 1, D+4] uint8
    torch::Tensor weights,       // [B*next_n, H] float32
    torch::Tensor context_lens,  // [B] int32
    torch::Tensor block_tables,  // [B, max_blocks] int32
    int64_t max_model_len, bool block_flat) {
  TORCH_CHECK(q_fp8.is_cuda(), "q_fp8 must be on HIP device");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on HIP device");
  TORCH_CHECK(weights.is_cuda(), "weights must be on HIP device");
  TORCH_CHECK(context_lens.is_cuda(), "context_lens must be on HIP device");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be on HIP device");
  TORCH_CHECK(
      q_fp8.dtype() == torch::kFloat8_e4m3fn || q_fp8.dtype() == torch::kUInt8,
      "q_fp8 must be FP8 e4m3fn or uint8 (got ", q_fp8.dtype(), ")");
  TORCH_CHECK(q_fp8.dim() == 4 && q_fp8.is_contiguous(),
              "q_fp8 must be contiguous [B, next_n, H, D]");
  TORCH_CHECK(weights.dtype() == torch::kFloat32 && weights.stride(1) == 1,
              "weights must be fp32 [rows, H] with unit head stride");
  TORCH_CHECK(context_lens.dim() == 1 &&
                  context_lens.scalar_type() == torch::kInt32 &&
                  context_lens.is_contiguous(),
              "context_lens must be contiguous int32 [B]");
  TORCH_CHECK(block_tables.dim() == 2 &&
                  block_tables.scalar_type() == torch::kInt32 &&
                  block_tables.stride(1) == 1,
              "block_tables must be int32 [B, max_blocks], unit column stride");

  auto stream = at::hip::getCurrentHIPStream();
  const int B = q_fp8.size(0);
  const int next_n = q_fp8.size(1);
  const int H = q_fp8.size(2);
  const int D = q_fp8.size(3);
  const int block_size = kv_cache.size(1);
  const int max_blocks_per_seq = block_tables.size(1);
  const int num_pages = kv_cache.size(0);

  TORCH_CHECK(context_lens.size(0) == B, "context_lens length ",
              context_lens.size(0), " must match B=", B);
  TORCH_CHECK(block_tables.size(0) >= B, "block_tables rows ",
              block_tables.size(0), " must be >= B=", B);
  TORCH_CHECK(weights.size(0) >= B * next_n, "weights rows ", weights.size(0),
              " must be >= B*next_n=", B * next_n);
  TORCH_CHECK(weights.stride(0) == H, "weights rows must be contiguous");

  auto logits = torch::full(
      {B * next_n, max_model_len}, -std::numeric_limits<float>::infinity(),
      torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (B == 0 || max_model_len == 0) return logits;

  const int64_t page_stride = kv_cache.stride(0);
  int value_slot_bytes, scale_base, scale_slot_bytes;
  if (block_flat) {
    value_slot_bytes = D;
    scale_base = block_size * D;
    scale_slot_bytes = 4;
  } else {
    value_slot_bytes = static_cast<int>(kv_cache.stride(1));
    scale_base = D;
    scale_slot_bytes = static_cast<int>(kv_cache.stride(1));
  }

  const int debug_oob = (getenv("VLLM_PAGED_MQA_DEBUG_OOB") != nullptr &&
                         getenv("VLLM_PAGED_MQA_DEBUG_OOB")[0] == '1')
                            ? 1
                            : 0;

  const int rows = B * next_n;
  if (H == 64 && D == 128) {
    constexpr int kHeadDim = 128, kHeads = 64, kThreads = 64, kBlockK = 128;
    const int64_t tiles = (max_model_len + kBlockK - 1) / kBlockK;
    const int splits = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(tiles, 512 / std::max(rows, 1))));
    const size_t smem_bytes =
        sizeof(uint16_t) * kHeads * kHeadDim + sizeof(float) * kHeads * kBlockK;
    auto kernel_ptr =
        paged_mqa_logits_decode_kernel<kHeadDim, kHeads, kThreads, kBlockK>;
    (void)hipFuncSetAttribute((const void*)kernel_ptr,
                              hipFuncAttributeMaxDynamicSharedMemorySize,
                              (int)smem_bytes);
    kernel_ptr<<<dim3(rows, splits), dim3(kThreads), smem_bytes, stream>>>(
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        kv_cache.data_ptr<uint8_t>(), weights.data_ptr<float>(),
        context_lens.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
        block_size, max_blocks_per_seq,
        static_cast<int>(block_tables.stride(0)), page_stride, value_slot_bytes,
        scale_base, scale_slot_bytes, logits.data_ptr<float>(),
        static_cast<int>(max_model_len), next_n, num_pages, debug_oob);
  } else {
    TORCH_CHECK(false,
                "paged_mqa_logits_decode_rdna2 only supports H=64, D=128 "
                "(got H=",
                H, ", D=", D, ")");
  }
  return logits;
}
