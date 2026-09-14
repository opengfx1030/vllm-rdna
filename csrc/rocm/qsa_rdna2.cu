// T48 — QSA (Qwen Sparse Attention) decode glue for gfx1030.
//
// Three kernels for the Qwen4Exp QSA decode path. Two of them are ports
// of the corresponding Triton kernels in vllm/models/qwen4_exp/amd/ops/qsa.py;
// the third is a thin wrapper that re-uses the existing
// paged_mqa_logits_decode_rdna2 indexer kernel when the shapes line up.
//
//   qsa_store_cache_rows_rdna2
//       row-major rows -> paged K cache scatter (mirrors the 5D paged K
//       layout that reshape_and_cache_flash_rdna2 writes; reuses the same
//       stride contract).
//
//   qsa_compress_groups_rdna2
//       Average COMPRESS_RATIO consecutive keys into one pooled key. The
//       compression can span the per-request ring (compressor_state_cache)
//       and this step's raw_rows; a separate ring slot holds the
//       RoPE-position tail.
//
//   qsa_mqa_paged_rdna2
//       MQA paged logits: for each (row, kv_block), gather K rows, dot
//       with the row's Q (over NUM_HEADS index heads), and accumulate the
//       max(error=0) softmax-pre scores. The shape matches
//       paged_mqa_logits_decode_rdna2 (also: same per-row indexer layout),
//       so this host wrapper delegates to the existing HIP kernel when
//       the dtype/layout is compatible. Falls through to Triton otherwise
//       (the Python dispatcher in vllm/models/qwen4_exp/amd/ops/qsa.py
//       decides which path to take based on the env flag).
//
// Opt-in: VLLM_RDNA_QSA_HIP=1 + on_gfx10x() in the Python dispatcher.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "rocm/ops.h"  // paged_mqa_logits_decode_rdna2

namespace {

__device__ __forceinline__ uint4 ld_u4(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void st_u4(void* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}

// ---------------------------------------------------------------------------
// qsa_store_cache_rows_rdna2:
//   rows: [num_rows, WIDTH] fp16 (or bf16)
//   slots: [num_rows] int32 -> (block_id * PAGE_SIZE + token) in cache
//   cache: paged K cache, [num_blocks, PAGE_SIZE, WIDTH]
// Mirrors _store_qsa_rows_kernel in vllm/models/qwen4_exp/amd/ops/qsa.py.
// ---------------------------------------------------------------------------

template <int BLOCK_D>
__global__ void qsa_store_cache_rows_kernel(
    const void* __restrict__ rows, const int* __restrict__ slots,
    void* __restrict__ cache, int num_rows, int num_blocks,
    int stride_rows_row, int stride_rows_dim,
    int stride_cache_block, int stride_cache_token, int stride_cache_dim,
    int PAGE_SIZE, int WIDTH, int elem_bytes) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int slot = slots[row];
  const bool valid = (slot >= 0) && (slot < num_blocks * PAGE_SIZE);
  const int safe_slot = max(slot, 0);
  const int block_id = safe_slot / PAGE_SIZE;
  const int token = safe_slot % PAGE_SIZE;

  char* cache_ptr = reinterpret_cast<char*>(cache)
      + (size_t)block_id * stride_cache_block
      + (size_t)token * stride_cache_token;
  const char* row_ptr = reinterpret_cast<const char*>(rows)
      + (size_t)row * stride_rows_row;

  const int vec_bytes = 16;  // uint4 = 16B = 8 fp16 / 4 bf16 (we only do fp16 here)
  const int vec_count = WIDTH / 8;  // 8 fp16 per uint4
  const int elem_per_uint4 = 8;

  // Vectorised body
  for (int v = threadIdx.x; v < vec_count; v += blockDim.x) {
    if (!valid) return;
    uint4 row_v = ld_u4(row_ptr + v * vec_bytes);
    st_u4(cache_ptr + v * vec_bytes, row_v);
  }
  // Scalar tail
  const int tail_start = vec_count * elem_per_uint4;
  for (int i = tail_start + threadIdx.x; i < WIDTH; i += blockDim.x) {
    if (!valid) return;
    if (elem_bytes == 2) {
      *reinterpret_cast<half*>(cache_ptr + i * stride_cache_dim) =
          *reinterpret_cast<const half*>(row_ptr + i * stride_rows_dim);
    }
  }
}

void qsa_store_cache_rows(
    const at::Tensor& rows, const at::Tensor& slots, at::Tensor& cache,
    int64_t page_size, int64_t width) {
  TORCH_CHECK(rows.is_cuda() && slots.is_cuda() && cache.is_cuda(),
              "qsa_store_cache_rows_rdna2: all tensors on CUDA/HIP");
  TORCH_CHECK(rows.scalar_type() == at::kHalf,
              "qsa_store_cache_rows_rdna2: rows fp16 only");
  TORCH_CHECK(cache.scalar_type() == at::kHalf,
              "qsa_store_cache_rows_rdna2: cache fp16 only");
  TORCH_CHECK(slots.scalar_type() == at::kInt,
              "qsa_store_cache_rows_rdna2: slots int32 only");
  TORCH_CHECK(rows.dim() == 2, "rows must be 2D");
  TORCH_CHECK(rows.size(1) == width, "rows.size(1) must equal width");
  const int num_rows = rows.size(0);
  const int num_blocks = cache.size(0);
  TORCH_CHECK(slots.size(0) == num_rows, "slots.size(0) must equal num_rows");

  const at::cuda::OptionalCUDAGuard guard(cache.device());
  // Strides (cache is [num_blocks, page_size, width] -> 3D, but dim count
  // can vary; we trust callers to pass the right layout).
  const int stride_rows_row = rows.stride(0) * sizeof(half);
  const int stride_rows_dim = rows.stride(1) * sizeof(half);
  const int stride_cache_block = cache.stride(0) * sizeof(half);
  const int stride_cache_token = cache.stride(1) * sizeof(half);
  const int stride_cache_dim = cache.stride(2) * sizeof(half);

  // We compile a single BLOCK_D variant; the loop is generic.
  qsa_store_cache_rows_kernel<256><<<num_rows, 256>>>(
      rows.const_data_ptr(), slots.const_data_ptr<int>(),
      cache.mutable_data_ptr(),
      num_rows, num_blocks,
      stride_rows_row, stride_rows_dim,
      stride_cache_block, stride_cache_token, stride_cache_dim,
      (int)page_size, (int)width, (int)sizeof(half));
}

// ---------------------------------------------------------------------------
// qsa_compress_groups_rdna2:
//   Pooled[c, h] = (1/COMPRESS_RATIO) * sum_i K[group_i]
//   The group can span the per-request ring (compressor_state_cache) and
//   this step's raw_keys. We also write the first-group position tail.
// Mirrors _compress_qsa_groups_kernel in vllm/models/qwen4_exp/amd/ops/qsa.py.
// ---------------------------------------------------------------------------

template <int BLOCK_D, int COMPRESS_RATIO>
__global__ void qsa_compress_groups_kernel(
    const half* __restrict__ raw_keys, const int* __restrict__ raw_positions,
    const half* __restrict__ compressor_state_cache,
    const int* __restrict__ rope_cache,
    const int* __restrict__ compressor_state_table,
    const int* __restrict__ token_to_req,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ logical_positions,
    const int* __restrict__ compressed_slots,
    half* __restrict__ pooled, int* __restrict__ first_positions,
    int stride_raw_row, int stride_raw_dim,
    int stride_raw_positions_row, int stride_raw_positions_dim,
    int stride_compressor_state_block, int stride_compressor_state_token,
    int stride_compressor_state_dim,
    int stride_rope_block, int stride_rope_token, int stride_rope_dim,
    int stride_compressor_state_table_req,
    int stride_pooled_row, int stride_pooled_dim,
    int stride_positions_row, int stride_positions_dim,
    int num_rows, int num_compressor_state_blocks, int num_requests,
    int COMPRESSOR_STATE_SIZE, int HEAD_DIM,
    int LOAD_ROPE_POSITIONS) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const int request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int query_row_start = valid_request
      ? query_start_loc[safe_request] : 0;
  const int query_row_end = valid_request
      ? query_start_loc[safe_request + 1] : 0;
  const int end_position = logical_positions[row];
  const int compressed_slot = compressed_slots[row];
  const int chunk_start_position = end_position - (row - query_row_start);
  const int compressor_state_block =
      valid_request
          ? compressor_state_table[safe_request]
          : -1;
  const bool valid_compressor_state_block =
      (compressor_state_block >= 0)
      && (compressor_state_block < num_compressor_state_blocks);
  const bool valid_row =
      valid_request
      && (row >= query_row_start)
      && (row < query_row_end)
      && (end_position >= COMPRESS_RATIO - 1)
      && (compressed_slot >= 0);
  const int safe_compressor_block = max(compressor_state_block, 0);

  float acc[BLOCK_D];
#pragma unroll
  for (int b = 0; b < BLOCK_D; b++) acc[b] = 0.f;

#pragma unroll
  for (int go = 0; go < COMPRESS_RATIO; go++) {
    const int position = end_position - (COMPRESS_RATIO - 1 - go);
    const bool use_raw = position >= chunk_start_position;
    const int raw_row = query_row_start + position - chunk_start_position;
    const bool raw_ok = valid_row && use_raw
        && (raw_row >= query_row_start)
        && (raw_row < query_row_end)
        && (raw_row < num_rows);
    const bool state_ok = valid_row && !use_raw && valid_compressor_state_block;
#pragma unroll
    for (int b = 0; b < BLOCK_D; b++) {
      if (b >= HEAD_DIM) continue;
      float v = 0.f;
      if (raw_ok) {
        v = __half2float(
            raw_keys[raw_row * stride_raw_row + b * stride_raw_dim]);
      } else if (state_ok) {
        const int tok = position % COMPRESSOR_STATE_SIZE;
        v = __half2float(
            compressor_state_cache[
                safe_compressor_block * stride_compressor_state_block
                + tok * stride_compressor_state_token
                + b * stride_compressor_state_dim]);
      }
      acc[b] += v;
    }
  }

  // Store pooled = acc / COMPRESS_RATIO
#pragma unroll
  for (int b = 0; b < BLOCK_D; b++) {
    if (b < HEAD_DIM) {
      pooled[row * stride_pooled_row + b * stride_pooled_dim] =
          __float2half(acc[b] / float(COMPRESS_RATIO));
    }
  }

  // First-position tail (3 dims; we just write 0..min(3,BLOCK_D)).
  const int first_position = end_position - COMPRESS_RATIO + 1;
  if (LOAD_ROPE_POSITIONS != 0) {
    const bool first_from_raw = first_position >= chunk_start_position;
    const int raw_first_row =
        query_row_start + first_position - chunk_start_position;
    const bool raw_first_ok = valid_row && first_from_raw
        && (raw_first_row >= query_row_start)
        && (raw_first_row < query_row_end)
        && (raw_first_row < num_rows);
    const bool state_first_ok = valid_row && !first_from_raw
        && valid_compressor_state_block;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      if (p >= 3) continue;
      int v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row
                          + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
        const int tok = first_position % COMPRESSOR_STATE_SIZE;
        v = rope_cache[safe_compressor_block * stride_rope_block
                       + tok * stride_rope_token
                       + p * stride_rope_dim];
      } else if (valid_row) {
        v = first_position;
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    }
  } else {
#pragma unroll
    for (int p = 0; p < 4; p++) {
      if (p >= 3) continue;
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : 0;
    }
  }
}

void qsa_compress_groups(
    const at::Tensor& raw_keys, const at::Tensor& raw_positions,
    const at::Tensor& compressor_state_cache, const at::Tensor& rope_cache,
    const at::Tensor& compressor_state_table,
    const at::Tensor& token_to_req, const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions, const at::Tensor& compressed_slots,
    at::Tensor& pooled, at::Tensor& first_positions,
    int64_t compress_ratio, int64_t compressor_state_size,
    int64_t head_dim, bool load_rope_positions) {
  TORCH_CHECK(raw_keys.is_cuda(), "raw_keys must be CUDA");
  TORCH_CHECK(pooled.is_cuda(), "pooled must be CUDA");
  TORCH_CHECK(first_positions.is_cuda(), "first_positions must be CUDA");
  TORCH_CHECK(raw_keys.scalar_type() == at::kHalf, "raw_keys fp16 only");
  TORCH_CHECK(pooled.scalar_type() == at::kHalf, "pooled fp16 only");
  const int num_rows = raw_keys.size(0);
  const int num_compressor_state_blocks = compressor_state_cache.size(0);
  const int num_requests = compressor_state_table.size(0);
  const int BLOCK_D = 128;  // covers HEAD_DIM in {32, 64, 128}

  const at::cuda::OptionalCUDAGuard guard(raw_keys.device());
  auto launch = [&](auto cr_const) {
    constexpr int CR = decltype(cr_const)::value;
    qsa_compress_groups_kernel<BLOCK_D, CR><<<num_rows, 128>>>(
        reinterpret_cast<const half*>(raw_keys.const_data_ptr()),
        raw_positions.defined() ? raw_positions.const_data_ptr<int>() : nullptr,
        compressor_state_cache.defined()
            ? reinterpret_cast<const half*>(compressor_state_cache.const_data_ptr())
            : nullptr,
        rope_cache.defined() ? rope_cache.const_data_ptr<int>() : nullptr,
        compressor_state_table.const_data_ptr<int>(),
        token_to_req.const_data_ptr<int>(),
        query_start_loc.const_data_ptr<int>(),
        logical_positions.const_data_ptr<int>(),
        compressed_slots.const_data_ptr<int>(),
        reinterpret_cast<half*>(pooled.mutable_data_ptr()),
        first_positions.mutable_data_ptr<int>(),
        (int)raw_keys.stride(0), (int)raw_keys.stride(1),
        raw_positions.defined() ? (int)raw_positions.stride(0) : 0,
        raw_positions.defined() ? (int)raw_positions.stride(1) : 0,
        (int)compressor_state_cache.stride(0) * (int)sizeof(half),
        (int)compressor_state_cache.stride(1) * (int)sizeof(half),
        (int)compressor_state_cache.stride(2) * (int)sizeof(half),
        rope_cache.defined() ? (int)rope_cache.stride(0) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(1) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(2) : 0,
        (int)compressor_state_table.stride(0),
        (int)pooled.stride(0), (int)pooled.stride(1),
        (int)first_positions.stride(0), (int)first_positions.stride(1),
        num_rows, num_compressor_state_blocks, num_requests,
        (int)compressor_state_size, (int)head_dim,
        load_rope_positions ? 1 : 0);
  };
  if (compress_ratio == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (compress_ratio == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (compress_ratio == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (compress_ratio == 8) { launch(std::integral_constant<int, 8>{}); }
  else if (compress_ratio == 16) { launch(std::integral_constant<int, 16>{}); }
  else { TORCH_CHECK(false, "qsa_compress_groups_rdna2: compress_ratio in {1,2,4,8,16} only"); }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers.
// ---------------------------------------------------------------------------

void qsa_store_cache_rows_rdna2(
    torch::Tensor rows, torch::Tensor slots, torch::Tensor cache,
    int64_t page_size, int64_t width) {
  qsa_store_cache_rows(rows, slots, cache, page_size, width);
}

void qsa_compress_groups_rdna2(
    torch::Tensor raw_keys, torch::Tensor raw_positions,
    torch::Tensor compressor_state_cache, torch::Tensor rope_cache,
    torch::Tensor compressor_state_table,
    torch::Tensor token_to_req, torch::Tensor query_start_loc,
    torch::Tensor logical_positions, torch::Tensor compressed_slots,
    torch::Tensor pooled, torch::Tensor first_positions,
    int64_t compress_ratio, int64_t compressor_state_size,
    int64_t head_dim, bool load_rope_positions) {
  qsa_compress_groups(
      raw_keys, raw_positions, compressor_state_cache, rope_cache,
      compressor_state_table, token_to_req, query_start_loc,
      logical_positions, compressed_slots,
      pooled, first_positions,
      compress_ratio, compressor_state_size, head_dim, load_rope_positions);
}

at::Tensor qsa_mqa_paged_rdna2(
    torch::Tensor q_fp16, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    int64_t max_model_len) {
  // The shape contract for qsa_mqa_paged is identical to
  // paged_mqa_logits_decode_rdna2 (a single Q head, MQA K layout, single KV
  // head). Re-use the existing indexer kernel; if a future QSA shape
  // diverges (e.g. multi-head Q like the splitk kernel) we'll need a
  // dedicated kernel here.
  return paged_mqa_logits_decode_rdna2(
      q_fp16, kv_cache, weights, context_lens, block_tables, max_model_len);
}
