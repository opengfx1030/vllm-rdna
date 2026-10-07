// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged MQA logits kernel for AMD RDNA2 (gfx1030) -- DeepSeek V4 Lightning
// Indexer fallback path. AITER is CDNA-only and crashes on gfx1030; this
// kernel computes the FP8 MQA logits that AITER's `paged_mqa_logits`
// would have produced, using the same algorithm as
// `fp8_paged_mqa_logits_torch` but in a single fused kernel.
//
// Layout:
//   q_fp8:        [B, next_n, H, D]    float8_e4m3fn, packed
//   kv_cache:     [num_pages, page_size, 1, D + 4]  uint8
//                 - first D bytes per (page, slot): FP8 K values
//                 - last 4 bytes per (page, slot): fp32 dequant scale
//   weights:      [B * next_n, H]      float32 (per-head weight)
//   context_lens: [B]                  int32
//   block_tables: [B, max_blocks]      int32
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
// This is the MQA "multi-query attention logits" stage. The downstream
// top_k_per_row kernel picks the top-K indices per row.

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include "qdq_fp8_rdna2.cuh"

#ifndef VLLM_ROCM_GFX1030_PAGED_MQA_MAX_HEAD_DIM
#define VLLM_ROCM_GFX1030_PAGED_MQA_MAX_HEAD_DIM 256
#endif

// One thread block processes one (batch, position) row. The block iterates
// over kv slots in [0, context_len), accumulating logits.
//
// Block layout:
//   threadIdx.x = head index (one thread per head, N_HEADS lanes)
//   blockIdx.x  = flat row index = b * next_n + p
// Each thread computes ONE head's dot product for BLOCK_K kv tokens
// at a time, accumulating into shared-mem scratch. Then a single
// thread reduces across heads and writes to global logits.
template <int HEAD_DIM, int N_HEADS, int BLOCK_THREADS, int BLOCK_K>
__global__ void paged_mqa_logits_decode_kernel(
    const uint8_t* __restrict__ q_packed,   // [B*next_n, N_HEADS, HEAD_DIM]
    const uint8_t* __restrict__ kv_cache,   // paged
    const float* __restrict__ weights,      // [B*next_n, N_HEADS]
    const int32_t* __restrict__ context_lens,    // [B]
    const int32_t* __restrict__ block_tables,    // [B, max_blocks]
    int32_t block_size,
    int32_t max_blocks_per_seq,
    int32_t page_stride,        // bytes per page in kv_cache
    int32_t slot_stride,        // bytes per slot in a page
    int32_t scale_offset,       // byte offset of fp32 scale within a slot
    float* __restrict__ logits,  // [B*next_n, max_model_len]
    int32_t max_model_len,
    int32_t next_n,
    int32_t num_pages,
    int32_t debug_oob) {
  static_assert(N_HEADS <= BLOCK_THREADS,
                "Each thread handles at least one head");
  const int row = blockIdx.x;
  const int b = row / next_n;
  const int tid = threadIdx.x;

  const int32_t seq_len = context_lens[b];
  if (seq_len <= 0) return;

  // Shared memory:
  //   q_shared[N_HEADS, HEAD_DIM] in fp16 bits
  //   partial_logits[N_HEADS, BLOCK_K] in float
  extern __shared__ char smem_raw[];
  uint16_t* q_shared = reinterpret_cast<uint16_t*>(smem_raw);
  float* partial = reinterpret_cast<float*>(
      q_shared + N_HEADS * HEAD_DIM);

  const int q_row_offset = row * N_HEADS * HEAD_DIM;
  // Cooperatively dequantize q for this row.
  for (int idx = tid; idx < N_HEADS * HEAD_DIM; idx += BLOCK_THREADS) {
    int h = idx / HEAD_DIM;
    int d = idx % HEAD_DIM;
    uint8_t qb = q_packed[q_row_offset + h * HEAD_DIM + d];
    q_shared[h * HEAD_DIM + d] = fp8_e4m3_to_fp16_bits(qb);
  }
  __syncthreads();

  // Each thread handles one head (tid < N_HEADS). For N_HEADS=64
  // and BLOCK_THREADS=64, tid == h. Threads with tid >= N_HEADS
  // are idle (kept for warpsize alignment).
  if (tid < N_HEADS) {
    const int h = tid;
    const uint16_t* q_h = q_shared + h * HEAD_DIM;
    const float w_h = weights[row * N_HEADS + h];

    // Process kv tokens in BLOCK_K-sized tiles.
    for (int tile_start = 0; tile_start < seq_len; tile_start += BLOCK_K) {
      int tile_end = min(tile_start + BLOCK_K, seq_len);

      // Each thread computes its head's contribution to BLOCK_K tokens.
      for (int kv = tile_start; kv < tile_end; ++kv) {
        int page_idx = kv / block_size;
        int slot_in_page = kv % block_size;
        int32_t page_id = block_tables[b * max_blocks_per_seq + page_idx];
        float dot = 0.0f;
        if (page_id < 0 || page_id >= num_pages) {
          if (debug_oob && tid == 0) {
            printf(
                "[paged_mqa OOB] row=%d b=%d kv=%d seq_len=%d page_idx=%d "
                "page_id=%d num_pages=%d max_blocks=%d bs=%d\n",
                row, b, kv, seq_len, page_idx, page_id, num_pages,
                max_blocks_per_seq, block_size);
          }
          partial[h * BLOCK_K + (kv - tile_start)] = 0.0f;
          continue;
        }
        // slot_base must be int64: page_id * page_stride can exceed 2^31 for
        // large caches (page_stride ~1MB on the strided indexer layout).
        const int64_t slot_base =
            (int64_t)page_id * page_stride + (int64_t)slot_in_page * slot_stride;

        float k_scale;
        __builtin_memcpy(&k_scale, kv_cache + slot_base + scale_offset, 4);

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          uint8_t kb = kv_cache[slot_base + d];
          __half k_half = __ushort_as_half(fp8_e4m3_to_fp16_bits(kb));
          __half q_half = __ushort_as_half(q_h[d]);
          dot += __half2float(q_half) * __half2float(k_half);
        }
        dot *= k_scale;
        dot = fmaxf(dot, 0.0f);
        dot *= w_h;

        partial[h * BLOCK_K + (kv - tile_start)] = dot;
      }
      __syncthreads();

      // Single thread reduces across heads for each kv in the tile.
      if (tid == 0) {
        for (int k = 0; k < tile_end - tile_start; ++k) {
          float total = 0.0f;
          for (int hh = 0; hh < N_HEADS; ++hh) {
            total += partial[hh * BLOCK_K + k];
          }
          logits[row * max_model_len + tile_start + k] = total;
        }
      }
      __syncthreads();
    }
  } else {
    // Idle threads: zero-init partials to avoid stale data.
    for (int j = 0; j < N_HEADS * BLOCK_K; ++j) {
      partial[j] = 0.0f;
    }
  }
}

#define LAUNCH_DECODE_KERNEL(HEAD_DIM, N_HEADS, BLOCK_THREADS, BLOCK_K)         \
  do {                                                                          \
    size_t smem_bytes =                                                         \
        sizeof(uint16_t) * N_HEADS * HEAD_DIM +                                  \
        sizeof(float) * N_HEADS * BLOCK_K;                                       \
    auto kernel_ptr = paged_mqa_logits_decode_kernel<                            \
        HEAD_DIM, N_HEADS, BLOCK_THREADS, BLOCK_K>;                              \
    (void)hipFuncSetAttribute(                                                   \
        (const void*)kernel_ptr,                                                \
        hipFuncAttributeMaxDynamicSharedMemorySize,                              \
        (int)smem_bytes);                                                        \
    kernel_ptr<<<grid, dim3(BLOCK_THREADS), smem_bytes, stream>>>(               \
        q_packed, kv_ptr, w_ptr, cl_ptr, bt_ptr,                                 \
        block_size, max_blocks_per_seq, page_stride, slot_stride,                \
        scale_offset, out_ptr, max_model_len, next_n, num_pages, debug_oob);     \
  } while (0)

torch::Tensor paged_mqa_logits_decode_rdna2(
    torch::Tensor q_fp8,             // [B, next_n, H, D] float8_e4m3fn
    torch::Tensor kv_cache,          // paged [num_pages, page_size, 1, D+4] uint8
    torch::Tensor weights,           // [B*next_n, H] float32
    torch::Tensor context_lens,      // [B] int32
    torch::Tensor block_tables,      // [B, max_blocks] int32
    int64_t max_model_len) {
  TORCH_CHECK(q_fp8.is_cuda(), "q_fp8 must be on HIP device");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on HIP device");
  TORCH_CHECK(weights.is_cuda(), "weights must be on HIP device");
  TORCH_CHECK(context_lens.is_cuda(), "context_lens must be on HIP device");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be on HIP device");
  TORCH_CHECK(
      q_fp8.dtype() == torch::kFloat8_e4m3fn || q_fp8.dtype() == torch::kUInt8,
      "q_fp8 must be FP8 e4m3fn or uint8 (got ", q_fp8.dtype(), ")");
  TORCH_CHECK(q_fp8.dim() == 4, "q_fp8 must be [B, next_n, H, D]");
  TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be fp32");
  TORCH_CHECK(context_lens.dim() == 1, "context_lens must be 1D [B]");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D [B, max_blocks]");

  auto stream = at::hip::getCurrentHIPStream();
  const int B = q_fp8.size(0);
  const int next_n = q_fp8.size(1);
  const int H = q_fp8.size(2);
  const int D = q_fp8.size(3);
  const int block_size = kv_cache.size(1);
  const int max_blocks_per_seq = block_tables.size(1);
  const int num_pages = kv_cache.size(0);

  TORCH_CHECK(context_lens.size(0) == B,
              "context_lens length ", context_lens.size(0),
              " must match B=", B);
  TORCH_CHECK(block_tables.size(0) == B,
              "block_tables rows ", block_tables.size(0),
              " must match B=", B);
  TORCH_CHECK(weights.size(0) >= B * next_n,
              "weights rows ", weights.size(0),
              " must be >= B*next_n=", B * next_n);

  auto logits = torch::full(
      {B * next_n, max_model_len}, -std::numeric_limits<float>::infinity(),
      torch::dtype(torch::kFloat32).device(q_fp8.device()));

  const int64_t page_stride = kv_cache.stride(0);
  const int64_t slot_stride = kv_cache.stride(1);
  const int scale_offset = D;

  const uint8_t* q_packed = q_fp8.data_ptr<uint8_t>();
  const uint8_t* kv_ptr = kv_cache.data_ptr<uint8_t>();
  const float* w_ptr = weights.data_ptr<float>();
  const int32_t* cl_ptr = context_lens.data_ptr<int32_t>();
  const int32_t* bt_ptr = block_tables.data_ptr<int32_t>();
  float* out_ptr = logits.data_ptr<float>();

  const int debug_oob =
      (getenv("VLLM_PAGED_MQA_DEBUG_OOB") != nullptr &&
       getenv("VLLM_PAGED_MQA_DEBUG_OOB")[0] == '1')
          ? 1
          : 0;

  const int grid = B * next_n;
  if (H == 64 && D == 128) {
    constexpr int BLOCK_THREADS = 64;
    constexpr int BLOCK_K = 128;
    LAUNCH_DECODE_KERNEL(128, 64, BLOCK_THREADS, BLOCK_K);
  } else {
    TORCH_CHECK(false, "paged_mqa_logits_decode_rdna2 only supports H=64, D=128 "
                       "(got H=", H, ", D=", D, ")");
  }

  return logits;
}