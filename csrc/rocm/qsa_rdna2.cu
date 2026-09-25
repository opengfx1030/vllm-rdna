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

#include <algorithm>
#include <cstdio>
#include <cstdlib>

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
    const char* __restrict__ rows, const int64_t* __restrict__ slots,
    char* __restrict__ cache, int num_rows, int num_blocks,
    int64_t stride_rows_row, int64_t stride_rows_dim,
    int64_t stride_cache_block, int64_t stride_cache_token,
    int64_t stride_cache_dim,
    int PAGE_SIZE, int WIDTH, int elem_bytes) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int64_t slot = slots[row];
  const bool valid = (slot >= 0) && (slot < (int64_t)num_blocks * PAGE_SIZE);
  if (!valid) return;
  const int64_t safe_slot = slot;
  const int block_id = (int)(safe_slot / PAGE_SIZE);
  const int token = (int)(safe_slot % PAGE_SIZE);

  char* cache_ptr = cache + (size_t)block_id * stride_cache_block
      + (size_t)token * stride_cache_token;
  const char* row_ptr = rows + (size_t)row * stride_rows_row;

  const int elems_per_vec = 16 / elem_bytes;
  const int vec_count = WIDTH / elems_per_vec;
  for (int v = threadIdx.x; v < vec_count; v += blockDim.x) {
    const uint4 row_v = ld_u4(
        reinterpret_cast<const uint4*>(row_ptr + (size_t)v * 16));
    st_u4(reinterpret_cast<uint4*>(cache_ptr + (size_t)v * 16), row_v);
  }
  const int tail_start = vec_count * elems_per_vec;
  for (int i = tail_start + threadIdx.x; i < WIDTH; i += blockDim.x) {
    const char* src = row_ptr + (size_t)i * stride_rows_dim;
    char* dst = cache_ptr + (size_t)i * stride_cache_dim;
    if (elem_bytes == 2) {
      *reinterpret_cast<half*>(dst) = *reinterpret_cast<const half*>(src);
    } else if (elem_bytes == 4) {
      *reinterpret_cast<int32_t*>(dst) = *reinterpret_cast<const int32_t*>(src);
    } else if (elem_bytes == 8) {
      *reinterpret_cast<int64_t*>(dst) = *reinterpret_cast<const int64_t*>(src);
    }
  }
}

void qsa_store_cache_rows(
    const at::Tensor& rows, const at::Tensor& slots, at::Tensor& cache,
    int64_t page_size, int64_t width) {
  TORCH_CHECK(rows.is_cuda() && slots.is_cuda() && cache.is_cuda(),
              "qsa_store_cache_rows_rdna2: all tensors on CUDA/HIP");
  TORCH_CHECK(rows.scalar_type() == cache.scalar_type(),
              "qsa_store_cache_rows_rdna2: rows and cache dtype must match");
  TORCH_CHECK(slots.scalar_type() == at::kLong,
              "qsa_store_cache_rows_rdna2: slots int64 only");
  TORCH_CHECK(rows.dim() == 2, "rows must be 2D");
  TORCH_CHECK(rows.size(1) == width, "rows.size(1) must equal width");
  const int num_rows = rows.size(0);
  const int num_blocks = cache.size(0);
  TORCH_CHECK(slots.size(0) == num_rows, "slots.size(0) must equal num_rows");

  const at::cuda::OptionalCUDAGuard guard(cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int elem_bytes = rows.element_size();
  TORCH_CHECK(cache.element_size() == elem_bytes,
              "rows and cache element sizes must match");
  TORCH_CHECK(elem_bytes == 2 || elem_bytes == 4 || elem_bytes == 8,
              "elem size must be 2, 4, or 8 bytes");
  const int64_t stride_rows_row = rows.stride(0) * elem_bytes;
  const int64_t stride_rows_dim = rows.stride(rows.dim() - 1) * elem_bytes;
  const int64_t stride_cache_block = cache.stride(0) * elem_bytes;
  const int64_t stride_cache_token = cache.stride(1) * elem_bytes;
  const int64_t stride_cache_dim = cache.stride(3) * elem_bytes;

  qsa_store_cache_rows_kernel<256><<<num_rows, 256, 0, stream>>>(
      reinterpret_cast<const char*>(rows.const_data_ptr()),
      slots.const_data_ptr<int64_t>(),
      reinterpret_cast<char*>(cache.mutable_data_ptr()),
      num_rows, num_blocks,
      stride_rows_row, stride_rows_dim,
      stride_cache_block, stride_cache_token, stride_cache_dim,
      (int)page_size, (int)width, elem_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    const half* __restrict__ raw_keys, const int64_t* __restrict__ raw_positions,
    const half* __restrict__ compressor_state_cache,
    const int64_t* __restrict__ rope_cache,
    const int* __restrict__ compressor_state_table,
    const int* __restrict__ token_to_req,
    const int* __restrict__ query_start_loc,
    const int64_t* __restrict__ logical_positions,
    const int64_t* __restrict__ compressed_slots,
    half* __restrict__ pooled, int64_t* __restrict__ first_positions,
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
    int LOAD_ROPE_POSITIONS, int DEBUG) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const int request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int query_row_start = valid_request
      ? query_start_loc[safe_request] : 0;
  const int query_row_end = valid_request
      ? query_start_loc[safe_request + 1] : 0;
  const int64_t end_position = logical_positions[row];
  const int64_t compressed_slot = compressed_slots[row];
  const int64_t chunk_start_position = end_position - (row - query_row_start);
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
    const int64_t position = end_position - (COMPRESS_RATIO - 1 - go);
    const bool use_raw = position >= chunk_start_position;
    const int raw_row =
        (int)(query_row_start + position - chunk_start_position);
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
        const int tok = (int)(position % COMPRESSOR_STATE_SIZE);
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
  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (LOAD_ROPE_POSITIONS != 0) {
    const bool first_from_raw = first_position >= chunk_start_position;
    const int raw_first_row =
        (int)(query_row_start + first_position - chunk_start_position);
    const bool raw_first_ok = valid_row && first_from_raw
        && (raw_first_row >= query_row_start)
        && (raw_first_row < query_row_end)
        && (raw_first_row < num_rows);
    const bool state_first_ok = valid_row && !first_from_raw
        && valid_compressor_state_block;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      if (p >= 3) continue;
      int64_t v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row
                          + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
        const int tok = (int)(first_position % COMPRESSOR_STATE_SIZE);
        v = rope_cache[safe_compressor_block * stride_rope_block
                       + tok * stride_rope_token
                       + p * stride_rope_dim];
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    }
  } else {
#pragma unroll
    for (int p = 0; p < 4; p++) {
      if (p >= 3) continue;
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
  if (DEBUG) {
    const int64_t v0 = first_positions[row * stride_positions_row + 0];
    const int64_t v1 = first_positions[row * stride_positions_row + 1];
    const int64_t v2 = first_positions[row * stride_positions_row + 2];
    const bool bad = (v0 < 0) || (v0 > 1000000) || (v1 < 0) || (v1 > 1000000)
        || (v2 < 0) || (v2 > 1000000);
    if (bad) {
      const int64_t r0 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 0 * stride_rope_dim];
      const int64_t r1 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 1 * stride_rope_dim];
      const int64_t r2 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 2 * stride_rope_dim];
      printf(
          "[QSA-KBAD] row=%d end=%lld chunk_start=%lld first_pos=%lld "
          "from_raw=%d raw_row=%d blk=%d tok=%d valid=%d "
          "v=(%lld,%lld,%lld) ring=(%lld,%lld,%lld) load_rope=%d\n",
          row, (long long)end_position, (long long)chunk_start_position,
          (long long)first_position,
          (int)(first_position >= chunk_start_position),
          (int)(query_row_start + first_position - chunk_start_position),
          safe_compressor_block, (int)(first_position % COMPRESSOR_STATE_SIZE),
          (int)valid_row, (long long)v0, (long long)v1, (long long)v2,
          (long long)r0, (long long)r1, (long long)r2, LOAD_ROPE_POSITIONS);
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
  TORCH_CHECK(first_positions.scalar_type() == at::kLong,
              "first_positions int64 only");
  TORCH_CHECK(logical_positions.scalar_type() == at::kLong,
              "logical_positions int64 only");
  TORCH_CHECK(compressed_slots.scalar_type() == at::kLong,
              "compressed_slots int64 only");
  if (raw_positions.defined()) {
    TORCH_CHECK(raw_positions.scalar_type() == at::kLong,
                "raw_positions int64 only");
  }
  if (rope_cache.defined()) {
    TORCH_CHECK(rope_cache.scalar_type() == at::kLong,
                "rope_cache int64 only");
  }
  int64_t num_rows_i = raw_keys.size(0);
  num_rows_i = std::min<int64_t>(num_rows_i, logical_positions.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, token_to_req.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, compressed_slots.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, first_positions.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, pooled.size(0));
  if (raw_positions.defined()) {
    num_rows_i = std::min<int64_t>(num_rows_i, raw_positions.size(0));
  }
  if (num_rows_i != raw_keys.size(0)) {
    printf(
        "[QSA-WARN] row count mismatch: raw_keys=%lld clamped to %lld "
        "(logical_positions=%lld token_to_req=%lld compressed_slots=%lld "
        "first_positions=%lld pooled=%lld) — metadata buffers shorter than "
        "raw_keys\n",
        (long long)raw_keys.size(0), (long long)num_rows_i,
        (long long)logical_positions.size(0), (long long)token_to_req.size(0),
        (long long)compressed_slots.size(0), (long long)first_positions.size(0),
        (long long)pooled.size(0));
  }
  const int num_rows = (int)num_rows_i;
  const int num_compressor_state_blocks = compressor_state_cache.size(0);
  const int num_requests = compressor_state_table.size(0);
  const int BLOCK_D = 128;  // covers HEAD_DIM in {32, 64, 128}

  const at::cuda::OptionalCUDAGuard guard(raw_keys.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool qsa_dbg = ::getenv("VLLM_QSA_RDNA2_DEBUG") != nullptr;
  if (qsa_dbg) {
    printf(
        "[QSA-H] num_rows=%d cr=%lld cs=%lld hd=%lld stream=%p dev=%d "
        "fp=%p pooled=%p rk=%p\n",
        num_rows, (long long)compress_ratio, (long long)compressor_state_size,
        (long long)head_dim, (void*)stream,
        raw_keys.get_device(), (void*)first_positions.data_ptr(),
        (void*)pooled.data_ptr(), (void*)raw_keys.data_ptr());
  }
  auto launch = [&](auto cr_const) {
    constexpr int CR = decltype(cr_const)::value;
    qsa_compress_groups_kernel<BLOCK_D, CR><<<num_rows, 128, 0, stream>>>(
        reinterpret_cast<const half*>(raw_keys.const_data_ptr()),
        raw_positions.defined()
            ? raw_positions.const_data_ptr<int64_t>() : nullptr,
        compressor_state_cache.defined()
            ? reinterpret_cast<const half*>(compressor_state_cache.const_data_ptr())
            : nullptr,
        rope_cache.defined() ? rope_cache.const_data_ptr<int64_t>() : nullptr,
        compressor_state_table.const_data_ptr<int>(),
        token_to_req.const_data_ptr<int>(),
        query_start_loc.const_data_ptr<int>(),
        logical_positions.const_data_ptr<int64_t>(),
        compressed_slots.const_data_ptr<int64_t>(),
        reinterpret_cast<half*>(pooled.mutable_data_ptr()),
        first_positions.mutable_data_ptr<int64_t>(),
        (int)raw_keys.stride(0),
        (int)raw_keys.stride(2),
        raw_positions.defined() ? (int)raw_positions.stride(0) : 0,
        raw_positions.defined() ? (int)raw_positions.stride(2) : 0,
        (int)compressor_state_cache.stride(0),
        (int)compressor_state_cache.stride(1),
        (int)compressor_state_cache.stride(3),
        rope_cache.defined() ? (int)rope_cache.stride(0) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(1) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(3) : 0,
        (int)compressor_state_table.stride(0),
        (int)pooled.stride(0),
        (int)pooled.stride(2),
        (int)first_positions.stride(0),
        (int)first_positions.stride(1),
        num_rows, num_compressor_state_blocks, num_requests,
        (int)compressor_state_size, (int)head_dim,
        load_rope_positions ? 1 : 0, qsa_dbg ? 1 : 0);
  };
  if (compress_ratio == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (compress_ratio == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (compress_ratio == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (compress_ratio == 8) { launch(std::integral_constant<int, 8>{}); }
  else if (compress_ratio == 16) { launch(std::integral_constant<int, 16>{}); }
  else { TORCH_CHECK(false, "qsa_compress_groups_rdna2: compress_ratio in {1,2,4,8,16} only"); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers.
// ---------------------------------------------------------------------------

void qsa_store_cache_rows_rdna2(
    torch::Tensor rows, torch::Tensor slots, torch::Tensor cache,
    const at::Tensor& page_size, const at::Tensor& width) {
  qsa_store_cache_rows(rows, slots, cache,
                       page_size.item<int64_t>(), width.item<int64_t>());
}

void qsa_compress_groups_rdna2(
    torch::Tensor raw_keys, torch::Tensor raw_positions,
    torch::Tensor compressor_state_cache, torch::Tensor rope_cache,
    torch::Tensor compressor_state_table,
    torch::Tensor token_to_req, torch::Tensor query_start_loc,
    torch::Tensor logical_positions, torch::Tensor compressed_slots,
    torch::Tensor pooled, torch::Tensor first_positions,
    const at::Tensor& compress_ratio, const at::Tensor& compressor_state_size,
    const at::Tensor& head_dim, bool load_rope_positions) {
  qsa_compress_groups(
      raw_keys, raw_positions, compressor_state_cache, rope_cache,
      compressor_state_table, token_to_req, query_start_loc,
      logical_positions, compressed_slots,
      pooled, first_positions,
      compress_ratio.item<int64_t>(), compressor_state_size.item<int64_t>(),
      head_dim.item<int64_t>(), load_rope_positions);
}

at::Tensor qsa_mqa_paged_rdna2(
    torch::Tensor q_fp16, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    const at::Tensor& max_model_len) {
  // The shape contract for qsa_mqa_paged is identical to
  // paged_mqa_logits_decode_rdna2 (a single Q head, MQA K layout, single KV
  // head). Re-use the existing indexer kernel; if a future QSA shape
  // diverges (e.g. multi-head Q like the splitk kernel) we'll need a
  // dedicated kernel here.
  return paged_mqa_logits_decode_rdna2(
      q_fp16, kv_cache, weights, context_lens, block_tables,
      max_model_len.item<int64_t>());
}
