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

// One thread block processes one (row, split): KV tiles of BLOCK_K
// positions are dequantized to fp16 in shared memory (row padded by one
// dword: conflict-free column reads), q stays in shared memory as fp16, and
// each thread owns one tile position x N_HEADS / HEAD_GROUPS heads, using
// v_dot2_f32_f16 with an fp32 accumulator. Head-group partials are summed
// through shared memory. (The previous layout walked a whole tile serially in
// one thread per head: ~210 us per decode call at a 256-position context.)
typedef _Float16 logits_f16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t fp8x2_to_f16x2_bits(uint32_t two) {
  return static_cast<uint32_t>(fp8_e4m3_to_fp16_bits(two & 0xFFu)) |
         (static_cast<uint32_t>(fp8_e4m3_to_fp16_bits((two >> 8) & 0xFFu))
          << 16);
}

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
  constexpr int D2 = HEAD_DIM / 2;          // half2 per row
  constexpr int K_STRIDE = D2 + 1;          // padded k row (dwords)
  constexpr int GROUPS = BLOCK_THREADS / BLOCK_K;
  constexpr int HEADS_PER = N_HEADS / GROUPS;
  constexpr int LOAD_THREADS_PER_ROW = BLOCK_THREADS / BLOCK_K;
  constexpr int BYTES_PER_LOAD = HEAD_DIM / LOAD_THREADS_PER_ROW;
  static_assert(BLOCK_THREADS % BLOCK_K == 0 && N_HEADS % GROUPS == 0, "");
  static_assert(BYTES_PER_LOAD % 4 == 0, "dword loads");

  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int num_splits = gridDim.y;
  const int b = row / next_n;
  const int tid = threadIdx.x;

  const int32_t seq_len = min(context_lens[b], max_model_len);
  if (seq_len <= 0 || split * BLOCK_K >= seq_len) return;

  __shared__ uint32_t q_sh[N_HEADS * D2];
  __shared__ uint32_t k_sh[BLOCK_K * K_STRIDE];
  __shared__ float k_scale[BLOCK_K];
  __shared__ float red[GROUPS][BLOCK_K];

  const uint32_t* q_row = reinterpret_cast<const uint32_t*>(
      q_packed + (int64_t)row * N_HEADS * HEAD_DIM);
  for (int i = tid; i < N_HEADS * HEAD_DIM / 4; i += BLOCK_THREADS) {
    const uint32_t four = q_row[i];
    q_sh[2 * i] = fp8x2_to_f16x2_bits(four & 0xFFFFu);
    q_sh[2 * i + 1] = fp8x2_to_f16x2_bits(four >> 16);
  }

  const int kk = tid % BLOCK_K;
  const int hg = tid / BLOCK_K;
  float w[HEADS_PER];
#pragma unroll
  for (int j = 0; j < HEADS_PER; ++j)
    w[j] = weights[(int64_t)row * N_HEADS + hg + GROUPS * j];

  const int32_t* bt_row = block_tables + (int64_t)b * bt_stride;
  float* out_row = logits + (int64_t)row * max_model_len;
  const int lr = tid / LOAD_THREADS_PER_ROW;  // tile row this thread loads
  const int lc = (tid % LOAD_THREADS_PER_ROW) * BYTES_PER_LOAD;

  for (int tile_start = split * BLOCK_K; tile_start < seq_len;
       tile_start += num_splits * BLOCK_K) {
    const int tile_len = min(BLOCK_K, seq_len - tile_start);
    __syncthreads();  // q_sh ready / previous tile consumed
    {
      const int kv = tile_start + lr;
      const uint8_t* page = nullptr;
      int slot = 0;
      if (lr < tile_len) {
        const int page_idx = kv / block_size;
        slot = kv % block_size;
        const int32_t page_id =
            page_idx < max_blocks_per_seq ? bt_row[page_idx] : -1;
        if (page_id >= 0 && page_id < num_pages) {
          page = kv_cache + (int64_t)page_id * page_stride;
        } else if (debug_oob && lc == 0) {
          printf(
              "[paged_mqa OOB] row=%d b=%d kv=%d seq_len=%d page_idx=%d "
              "page_id=%d num_pages=%d\n",
              row, b, kv, seq_len, page_idx, page_id, num_pages);
        }
      }
      uint32_t* dst = k_sh + lr * K_STRIDE + lc / 2;
      if (page != nullptr) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(
            page + (int64_t)slot * value_slot_bytes + lc);
#pragma unroll
        for (int i = 0; i < BYTES_PER_LOAD / 4; ++i) {
          const uint32_t four = src[i];
          dst[2 * i] = fp8x2_to_f16x2_bits(four & 0xFFFFu);
          dst[2 * i + 1] = fp8x2_to_f16x2_bits(four >> 16);
        }
        if (lc == 0) {
          float sc;
          __builtin_memcpy(
              &sc, page + scale_base + (int64_t)slot * scale_slot_bytes, 4);
          k_scale[lr] = sc;
        }
      } else {
#pragma unroll
        for (int i = 0; i < BYTES_PER_LOAD / 2; ++i) dst[i] = 0u;
        if (lc == 0) k_scale[lr] = 0.0f;
      }
    }
    __syncthreads();

    const logits_f16x2* krow =
        reinterpret_cast<const logits_f16x2*>(k_sh + kk * K_STRIDE);
    const float sc = k_scale[kk];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < HEADS_PER; ++j) {
      const logits_f16x2* qrow =
          reinterpret_cast<const logits_f16x2*>(q_sh + (hg + GROUPS * j) * D2);
      float dot = 0.0f;
#pragma unroll 16
      for (int d = 0; d < D2; ++d)
        dot = __builtin_amdgcn_fdot2(qrow[d], krow[d], dot, /*clamp=*/false);
      acc += fmaxf(dot * sc, 0.0f) * w[j];
    }
    red[hg][kk] = acc;
    __syncthreads();
    if (tid < tile_len) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < GROUPS; ++g) total += red[g][tid];
      out_row[tile_start + tid] = total;
    }
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
    constexpr int kHeadDim = 128, kHeads = 64, kThreads = 256, kBlockK = 32;
    const int64_t tiles = (max_model_len + kBlockK - 1) / kBlockK;
    const int splits = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(tiles, 1024 / std::max(rows, 1))));
    auto kernel_ptr =
        paged_mqa_logits_decode_kernel<kHeadDim, kHeads, kThreads, kBlockK>;
    kernel_ptr<<<dim3(rows, splits), dim3(kThreads), 0, stream>>>(
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
