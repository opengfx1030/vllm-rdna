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
    int PAGE_SIZE, int WIDTH, int elem_bytes, int rows_per_block) {
  // The vectorized stride-1 path is only valid when both the row width and
  // the cache width are dense (element stride == 1). The server passes a
  // transposed ``canonical_qsa_rope_positions`` view whose last-dim stride is
  // ``tokens``, so fall back to the stride-aware scalar loop otherwise.
  const int elems_per_vec = 16 / elem_bytes;
  const bool row_dense = (stride_rows_dim == elem_bytes);
  const bool cache_dense = (stride_cache_dim == elem_bytes);
  const bool vec_exact = (WIDTH % elems_per_vec) == 0;
  if (row_dense && cache_dense && vec_exact && WIDTH >= elems_per_vec) {
    // Dense path: vec_count threads cover one row, so a block spans
    // blockDim.x / vec_count rows. One row per block left 240 of 256 threads
    // idle and cost ~2.5x the Triton kernel.
    const int vec_count = WIDTH / elems_per_vec;
    const int row = blockIdx.x * rows_per_block + threadIdx.x / vec_count;
    if (row >= num_rows) return;
    const int64_t slot = slots[row];
    if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
    char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
        + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
    const char* row_ptr = rows + (size_t)row * stride_rows_row;
    const int v = threadIdx.x % vec_count;
    st_u4(reinterpret_cast<uint4*>(cache_ptr + (size_t)v * 16),
          ld_u4(reinterpret_cast<const uint4*>(row_ptr + (size_t)v * 16)));
    return;
  }

  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int64_t slot = slots[row];
  if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
  char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
      + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
  const char* row_ptr = rows + (size_t)row * stride_rows_row;
  for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {
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

  const int elems_per_vec = 16 / elem_bytes;
  const bool dense = (stride_rows_dim == elem_bytes) &&
      (stride_cache_dim == elem_bytes) && (width % elems_per_vec == 0) &&
      (width >= elems_per_vec);
  const int vec_count = dense ? (int)(width / elems_per_vec) : 1;
  const int rows_per_block = dense ? (256 / vec_count) : 1;
  const int grid = (num_rows + rows_per_block - 1) / rows_per_block;
  qsa_store_cache_rows_kernel<256><<<grid, 256, 0, stream>>>(
      reinterpret_cast<const char*>(rows.const_data_ptr()),
      slots.const_data_ptr<int64_t>(),
      reinterpret_cast<char*>(cache.mutable_data_ptr()),
      num_rows, num_blocks,
      stride_rows_row, stride_rows_dim,
      stride_cache_block, stride_cache_token, stride_cache_dim,
      (int)page_size, (int)width, elem_bytes, rows_per_block);
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

  // One thread per output dim. A per-thread ``float acc[BLOCK_D]`` array spilled
  // to local memory, and every thread in the block recomputed the whole row.
  for (int b = threadIdx.x; b < HEAD_DIM; b += blockDim.x) {
    float acc = 0.f;
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
      acc += v;
    }
    pooled[row * stride_pooled_row + b * stride_pooled_dim] =
        __float2half(acc / float(COMPRESS_RATIO));
  }

  // First-position tail: threads 0..2 own one dim each.
  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (threadIdx.x < 3) {
    const int p = (int)threadIdx.x;
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
    } else {
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
  __syncthreads();
  if (DEBUG && threadIdx.x == 0) {
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
  TORCH_CHECK(head_dim <= 128,
              "qsa_compress_groups_rdna2: head_dim <= 128 only (the kernel pads "
              "to BLOCK_D=128 and would silently drop wider rows); got ",
              head_dim);
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

// ---------------------------------------------------------------------------
// qsa_mqa_paged_rdna2: QSA indexer scoring, exact parity with the Triton
// `_qsa_mqa_paged_kernel` in vllm/models/qwen4_exp/amd/ops/qsa.py.
//
// The previous implementation delegated to `paged_mqa_logits_decode_rdna2`
// (the DeepSeek V4 indexer), whose contract is different in every respect:
// fp8 e4m3 q/K with per-slot dequant scales, H == 64 and D == 128 only,
// per-head weights, `block_tables` indexed by row, visibility from
// `context_lens`, and no `visible_blocks` output. QSA needs fp16, H = 4,
// D = 128, unit weights, a request-indirected page table, visibility derived
// from `compress_ratio`, and the `visible_blocks` side channel, so it gets a
// dedicated kernel here.
//
// One thread per output column, BLOCK_N columns per CTA.
// ---------------------------------------------------------------------------
namespace {

typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));

template <int BLOCK_N, typename TPos, typename TSeq>
__global__ void qsa_mqa_paged_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ token_to_req,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    int32_t* __restrict__ visible_blocks, float* __restrict__ logits,
    int64_t stride_q_row, int64_t stride_q_head, int64_t stride_q_dim,
    int64_t stride_cache_block, int64_t stride_cache_token,
    int64_t stride_cache_dim, int64_t stride_table_req,
    int64_t stride_table_page, int64_t stride_logits_row, int num_rows,
    int num_columns, int num_pages, int num_requests, int table_width,
    int page_size, int num_heads, int head_dim, int compress_ratio,
    float score_divisor) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int col = (int)blockIdx.y * BLOCK_N + (int)threadIdx.x;

  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int64_t query_position = (int64_t)query_positions[row];
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t visible = min((query_position + 1) / compress_ratio,
                              sequence_length / compress_ratio);
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    visible_blocks[row] = (int32_t)visible;
  }
  if (col >= num_columns) return;

  const int logical_page = col / page_size;
  const int page_offset = col % page_size;
  bool valid = valid_request && (col < visible) && (logical_page < table_width);
  int32_t physical_page = -1;
  if (valid) {
    const int safe_logical_page = min(logical_page, table_width - 1);
    physical_page = page_table[safe_request * stride_table_req
                               + safe_logical_page * stride_table_page];
    valid = (physical_page >= 0) && (physical_page < num_pages);
  }

  float score = 0.0f;
  if (valid) {
    const half* k_row = k_cache + (int64_t)physical_page * stride_cache_block
        + (int64_t)page_offset * stride_cache_token;
    const half* q_row = q + (int64_t)row * stride_q_row;
    // Dense contiguous rows: two MACs per instruction via v_dot2. The indexer
    // layout is [rows, heads, head_dim] contiguous with head_dim even, so this
    // is the production path; anything else falls back to scalar half loads.
    // The dense path takes 128 bits at a time and keeps four independent
    // fdot2 accumulator chains. The previous form issued one 4-byte load per
    // f16x2 operand (512 loads per column per row) and accumulated all 64
    // fdot2 of a head into a single register, so the dot was a 64-deep
    // dependent chain. That is why HIP lost to Triton once most columns were
    // visible: the early-chunk win came only from skipping masked columns, and
    // the dense case was simply a worse kernel.
    const bool dense = (stride_q_dim == 1) && (stride_cache_dim == 1)
        && (stride_q_head % 8 == 0) && (stride_cache_token % 8 == 0)
        && (head_dim % 8 == 0)
        && ((reinterpret_cast<uintptr_t>(k_row)
             | reinterpret_cast<uintptr_t>(q_row)) % 16 == 0);
    for (int h = 0; h < num_heads; ++h) {
      const half* q_head = q_row + (int64_t)h * stride_q_head;
      float dot = 0.0f;
      if (dense) {
        const uint4* q_vec = reinterpret_cast<const uint4*>(q_head);
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_row);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int d8 = 0; d8 < head_dim / 8; ++d8) {
          const uint4 qq = q_vec[d8];
          const uint4 kk = k_vec[d8];
          const f16x2* qh = reinterpret_cast<const f16x2*>(&qq);
          const f16x2* kh = reinterpret_cast<const f16x2*>(&kk);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[j] = __builtin_amdgcn_fdot2(qh[j], kh[j], acc[j], false);
          }
        }
        dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      } else {
        for (int d = 0; d < head_dim; ++d) {
          dot += __half2float(q_head[(int64_t)d * stride_q_dim])
              * __half2float(k_row[(int64_t)d * stride_cache_dim]);
        }
      }
      score += fmaxf(dot, 0.0f);
    }
    score /= score_divisor;
  }
  const float neg_inf = __int_as_float(0xff800000);
  logits[(int64_t)row * stride_logits_row + col] = valid ? score : neg_inf;
}

template <typename TPos, typename TSeq>
void qsa_mqa_paged_launch(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& page_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor& visible_blocks,
    at::Tensor& logits, int64_t compress_ratio, double score_divisor,
    cudaStream_t stream) {
  constexpr int kBlockN = 128;
  const int num_rows = q.size(0);
  const int num_columns = logits.size(1);
  const dim3 grid(num_rows, (num_columns + kBlockN - 1) / kBlockN);
  qsa_mqa_paged_kernel<kBlockN, TPos, TSeq><<<grid, kBlockN, 0, stream>>>(
      reinterpret_cast<const half*>(q.const_data_ptr()),
      reinterpret_cast<const half*>(k_cache.const_data_ptr()),
      page_table.const_data_ptr<int32_t>(),
      token_to_req.const_data_ptr<int32_t>(),
      query_positions.const_data_ptr<TPos>(),
      sequence_lengths.const_data_ptr<TSeq>(),
      visible_blocks.mutable_data_ptr<int32_t>(), logits.mutable_data_ptr<float>(),
      q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1),
      k_cache.stride(3), page_table.stride(0), page_table.stride(1),
      logits.stride(0), num_rows, num_columns, k_cache.size(0),
      page_table.size(0), page_table.size(1), k_cache.size(1), q.size(1),
      q.size(2), (int)compress_ratio, (float)score_divisor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


std::tuple<at::Tensor, at::Tensor> qsa_mqa_paged_rdna2(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& page_table, const at::Tensor& token_to_req,
    const at::Tensor& query_positions, const at::Tensor& sequence_lengths,
    int64_t compress_ratio, int64_t num_columns, double score_scale) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && page_table.is_cuda()
                  && token_to_req.is_cuda() && query_positions.is_cuda()
                  && sequence_lengths.is_cuda(),
              "qsa_mqa_paged_rdna2: all tensors on CUDA/HIP");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf,
              "qsa_mqa_paged_rdna2: fp16 q/k_cache only");
  TORCH_CHECK(q.dim() == 3, "qsa_mqa_paged_rdna2: q must be [rows, heads, dim]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1
                  && k_cache.size(3) == q.size(2),
              "qsa_mqa_paged_rdna2: k_cache must be [pages, page, 1, head_dim]");
  TORCH_CHECK(page_table.scalar_type() == at::kInt
                  && token_to_req.scalar_type() == at::kInt,
              "qsa_mqa_paged_rdna2: page_table/token_to_req int32 only");
  TORCH_CHECK(page_table.dim() == 2, "qsa_mqa_paged_rdna2: page_table must be 2D");
  TORCH_CHECK(token_to_req.numel() == q.size(0),
              "qsa_mqa_paged_rdna2: token_to_req must have one entry per row");
  TORCH_CHECK(query_positions.numel() == q.size(0),
              "qsa_mqa_paged_rdna2: query_positions must have one entry per row");
  TORCH_CHECK(sequence_lengths.numel() == page_table.size(0),
              "qsa_mqa_paged_rdna2: sequence_lengths must match page_table rows");
  TORCH_CHECK(compress_ratio > 0, "qsa_mqa_paged_rdna2: compress_ratio > 0");
  TORCH_CHECK(num_columns >= 0, "qsa_mqa_paged_rdna2: num_columns >= 0");
  TORCH_CHECK(score_scale > 0.0, "qsa_mqa_paged_rdna2: score_scale > 0");

  const auto pos_dtype = query_positions.scalar_type();
  const auto seq_dtype = sequence_lengths.scalar_type();
  TORCH_CHECK((pos_dtype == at::kLong || pos_dtype == at::kInt)
                  && (seq_dtype == at::kLong || seq_dtype == at::kInt),
              "qsa_mqa_paged_rdna2: query_positions/sequence_lengths must be int32 "
              "or int64");

  const int64_t rows = q.size(0);
  auto logits = at::empty({rows, num_columns},
                          at::dtype(at::kFloat).device(q.device()));
  auto visible_blocks = at::empty({rows}, at::dtype(at::kInt).device(q.device()));
  if (rows == 0 || num_columns == 0 || page_table.size(0) == 0) {
    return {logits, visible_blocks};
  }

  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (pos_dtype == at::kLong && seq_dtype == at::kInt) {
    qsa_mqa_paged_launch<int64_t, int32_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else if (pos_dtype == at::kLong) {
    qsa_mqa_paged_launch<int64_t, int64_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else if (seq_dtype == at::kInt) {
    qsa_mqa_paged_launch<int32_t, int32_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else {
    qsa_mqa_paged_launch<int32_t, int64_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  }
  return {logits, visible_blocks};
}

// ---------------------------------------------------------------------------
// qsa_expand_indices_rdna2: expand compressed block indices into token indices
// and compact the causal tail of the open group. Port of the Triton
// `_expand_qsa_indices_kernel` in vllm/models/qwen4_exp/amd/ops/qsa.py.
//
// output_width = token_topk + compress_ratio - 1, block_topk =
// token_topk / compress_ratio. One thread per output element; slots that are
// neither an expanded block entry nor a valid tail entry are written as -1.
// ---------------------------------------------------------------------------
namespace {

// Columns per thread. The op is dominated by the per-row preamble and by the
// workgroup dispatch rate (the production grid is rows * 9 workgroups of 256
// threads), so a thread covering four strided columns cuts both by 4. Measured
// on gfx1030 at ratio 4 / topk 512: 4 is the knee (2 leaves ~6 us on the table
// at rows=2048, 8 costs ~0.3 us at rows<=24).
constexpr int kColsPerThread = 4;

template <typename TPos, typename TSeq, int CRE>
__global__ void qsa_expand_indices_kernel(
    const int32_t* __restrict__ block_indices,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    const int32_t* __restrict__ token_to_req, int32_t* __restrict__ out,
    int64_t stride_blocks_row, int64_t stride_blocks_column,
    int64_t stride_out_row, int64_t stride_out_column, int rows,
    int num_requests, int block_topk, int cr_rt, int output_width) {
  // compress_ratio as a compile-time constant turns the four runtime integer
  // divisions in the per-thread preamble (two of them int64) into shifts, which
  // is what Triton's COMPRESS_RATIO constexpr already gives its kernel. The
  // generic runtime path is kept for ratios outside the instantiated set.
  constexpr int CR = CRE;
  const int compress_ratio = (CR > 0) ? CR : cr_rt;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int col_base =
      (int)blockIdx.y * (blockDim.x * kColsPerThread) + (int)threadIdx.x;

  const int64_t query_position = (int64_t)query_positions[row];
  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t complete_blocks =
      min(min((query_position + 1) / compress_ratio,
              sequence_length / compress_ratio),
          (int64_t)block_topk);
  const int expanded_count = (int)complete_blocks * compress_ratio;
  const int tail_start = (int)((query_position + 1) / compress_ratio) * compress_ratio;
  const int tail_count = (int)((query_position + 1) - tail_start);

  // Each thread owns kColsPerThread columns strided by blockDim.x: the stores
  // stay fully coalesced, while the per-row preamble above and the workgroup
  // count are both kColsPerThread times smaller. The strided form measured far
  // better than giving one thread a whole contiguous rank (that layout writes
  // 4 bytes every 16 within a warp and loses store coalescing).
#pragma unroll
  for (int o = 0; o < kColsPerThread; o++) {
    const int col = col_base + o * (int)blockDim.x;
    if (col >= output_width) continue;
    int token = -1;
    if (col < expanded_count) {
      // Expanded block entry. Guarded exactly like Triton's masked load, so a
      // zero block_topk never reads (its expanded_count is 0 and nothing takes
      // this branch).
      const int block_rank = col / compress_ratio;
      const int offset = col - block_rank * compress_ratio;
      const int safe_rank = min(block_rank, block_topk - 1);
      const int64_t block =
          block_indices[(int64_t)row * stride_blocks_row
                        + (int64_t)safe_rank * stride_blocks_column];
      token = (int)(block * compress_ratio + offset);
    } else {
      // Compaction of the causal tail of the open group.
      const int tail_offset = col - expanded_count;
      if ((tail_offset < tail_count) && (tail_offset < compress_ratio - 1)) {
        token = tail_start + tail_offset;
      }
    }
    const bool valid = (token >= 0) && ((int64_t)token < sequence_length);
    out[(int64_t)row * stride_out_row + (int64_t)col * stride_out_column] =
        valid ? token : -1;
  }
}

}  // namespace

torch::Tensor qsa_expand_indices_rdna2(
    const at::Tensor& block_indices, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, const at::Tensor& token_to_req,
    int64_t compress_ratio, int64_t token_topk,
    c10::optional<at::Tensor> out) {
  TORCH_CHECK(block_indices.scalar_type() == at::kInt,
              "qsa_expand_indices_rdna2: block_indices int32 only");
  TORCH_CHECK(block_indices.dim() == 2,
              "qsa_expand_indices_rdna2: block_indices must be 2D");
  TORCH_CHECK(token_to_req.scalar_type() == at::kInt,
              "qsa_expand_indices_rdna2: token_to_req int32 only");
  TORCH_CHECK(compress_ratio > 0,
              "qsa_expand_indices_rdna2: compress_ratio > 0");
  TORCH_CHECK(token_topk >= 0,
              "qsa_expand_indices_rdna2: token_topk >= 0");
  TORCH_CHECK(compress_ratio == 0 || token_topk % compress_ratio == 0,
              "qsa_expand_indices_rdna2: token_topk must be divisible by "
              "compress_ratio");

  const int64_t rows = block_indices.size(0);
  const int64_t block_topk = token_topk / compress_ratio;
  const int64_t output_width = token_topk + compress_ratio - 1;
  TORCH_CHECK(block_indices.size(1) == block_topk,
              "qsa_expand_indices_rdna2: block_indices must be [rows, "
              "token_topk / compress_ratio]");
  TORCH_CHECK(query_positions.numel() == rows,
              "qsa_expand_indices_rdna2: query_positions must have one entry "
              "per row");
  TORCH_CHECK(token_to_req.numel() == rows,
              "qsa_expand_indices_rdna2: token_to_req must have one entry per "
              "row");
  TORCH_CHECK(sequence_lengths.dim() == 1 && sequence_lengths.numel() > 0,
              "qsa_expand_indices_rdna2: sequence_lengths must be nonempty 1D");
  const auto pos_dtype = query_positions.scalar_type();
  const auto seq_dtype = sequence_lengths.scalar_type();
  TORCH_CHECK((pos_dtype == at::kLong || pos_dtype == at::kInt) &&
                  (seq_dtype == at::kLong || seq_dtype == at::kInt),
              "qsa_expand_indices_rdna2: query_positions/sequence_lengths must "
              "be int32 or int64");

  at::Tensor result;
  if (out.has_value() && out->defined()) {
    result = *out;
    TORCH_CHECK(result.scalar_type() == at::kInt,
                "qsa_expand_indices_rdna2: out int32 only");
    TORCH_CHECK(result.dim() == 2 && result.size(0) == rows &&
                    result.size(1) == output_width,
                "qsa_expand_indices_rdna2: out must be [rows, token_topk + "
                "compress_ratio - 1]");
  } else {
    result = at::empty({rows, output_width},
                       at::dtype(at::kInt).device(block_indices.device()));
  }
  if (rows == 0 || output_width == 0) return result;

  constexpr int kColumnBlock = 256;
  const at::cuda::OptionalCUDAGuard guard(block_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_requests = sequence_lengths.numel();
  dim3 grid((unsigned)rows,
            (unsigned)((output_width + kColumnBlock * kColsPerThread - 1)
                       / (kColumnBlock * kColsPerThread)));

  auto launch_one = [&](auto pos_tag, auto seq_tag, auto cr_tag) {
    using TPos = decltype(pos_tag);
    using TSeq = decltype(seq_tag);
    constexpr int CRE = decltype(cr_tag)::value;
    qsa_expand_indices_kernel<TPos, TSeq, CRE><<<grid, kColumnBlock, 0, stream>>>(
        block_indices.const_data_ptr<int32_t>(),
        reinterpret_cast<const TPos*>(query_positions.const_data_ptr()),
        reinterpret_cast<const TSeq*>(sequence_lengths.const_data_ptr()),
        token_to_req.const_data_ptr<int32_t>(),
        result.mutable_data_ptr<int32_t>(), block_indices.stride(0),
        block_indices.stride(1), result.stride(0), result.stride(1), (int)rows,
        (int)num_requests, (int)block_topk, (int)compress_ratio,
        (int)output_width);
  };
  auto launch = [&](auto pos_tag, auto seq_tag) {
    switch ((int)compress_ratio) {
      case 1: launch_one(pos_tag, seq_tag, std::integral_constant<int, 1>{}); break;
      case 2: launch_one(pos_tag, seq_tag, std::integral_constant<int, 2>{}); break;
      case 4: launch_one(pos_tag, seq_tag, std::integral_constant<int, 4>{}); break;
      case 8: launch_one(pos_tag, seq_tag, std::integral_constant<int, 8>{}); break;
      case 16: launch_one(pos_tag, seq_tag, std::integral_constant<int, 16>{}); break;
      default: launch_one(pos_tag, seq_tag, std::integral_constant<int, 0>{}); break;
    }
  };
  if (pos_dtype == at::kLong && seq_dtype == at::kInt) {
    launch((int64_t)0, (int32_t)0);
  } else if (pos_dtype == at::kLong) {
    launch((int64_t)0, (int64_t)0);
  } else if (seq_dtype == at::kInt) {
    launch((int32_t)0, (int32_t)0);
  } else {
    launch((int32_t)0, (int64_t)0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

// ---------------------------------------------------------------------------
// qsa_sparse_paged_attention_rdna2: the QSA sparse paged GQA attention
// (split-k) and its reduction. Ports `_qsa_sparse_paged_gqa_splitk_kernel` and
// `_qsa_merge_splitk_kernel` in vllm/models/qwen4_exp/amd/ops/qsa.py.
//
// One program per (row, kv_head, split). Each group of GROUP_SIZE query heads
// shares one KV head; a warp owns one query head, and each lane owns the
// columns lane, lane+32, ... so the online softmax max/sum is a warp shuffle
// reduction and needs no LDS round trip. Scores pair over HEAD_DIM (k is
// token-major with dim contiguous, exactly as the paged cache stores it);
// P @ V pairs over the column instead, so v is read scalar per column (the
// transposed-fdot2 variant is an optimisation left for later).
// ---------------------------------------------------------------------------
namespace {

// The sparse-attention kernel runs one warp per query head, so its block size
// follows group_size (6 for this model at TP4: 24 query heads / 4 ranks, and
// max(1, 2 kv heads / 4) = 1 kv head per rank). QSA_MAX_GROUP bounds the LDS
// probability tile and the launch bounds.
constexpr int QSA_MAX_GROUP = 8;
constexpr int QSA_MAX_THREADS = QSA_MAX_GROUP * 32;
// LDS row stride for the staged K/V tiles: 8 halves of padding so the row start
// stays 16-byte aligned (uint4 loads) and the strided rows do not all land in the
// same bank. Two sizes: the 128-wide tile keeps three workgroups per CU, the
// 256-wide one (the production head_dim) needs ~39 KB and drops to one, so it is
// only instantiated when the head is that wide.
constexpr int QSA_LDS_DIM_128 = 128 + 8;
constexpr int QSA_LDS_DIM_256 = 256 + 8;

__device__ __forceinline__ float qsa_exp2(float x) { return exp2f(x); }

template <int BLOCK_N, int LDS_DIM, typename TIdx>
__global__ void __launch_bounds__(QSA_MAX_THREADS)
    qsa_sparse_paged_gqa_splitk_kernel(
        const half* __restrict__ q, const half* __restrict__ k_cache,
        const half* __restrict__ v_cache,
        const TIdx* __restrict__ logical_indices,
        const int32_t* __restrict__ block_table,
        const int32_t* __restrict__ token_to_req,
        float* __restrict__ partial_output, float* __restrict__ partial_lse,
        half* __restrict__ output, int64_t stride_q_row, int64_t stride_q_head,
        int64_t stride_k_block, int64_t stride_k_token, int64_t stride_k_head,
        int64_t stride_v_block, int64_t stride_v_token, int64_t stride_v_head,
        int64_t stride_indices_row, int64_t stride_table_req,
        int64_t stride_output_row, int64_t stride_output_head, int num_rows,
        int64_t num_cache_blocks, int num_requests, int topk, int page_size,
        int page_table_width, int group_size, int head_dim, int num_query_heads,
        int num_splits, int num_tiles, int single_split, int vec8) {
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_id = blockIdx.z;
  const int warp = (int)threadIdx.x >> 5;
  const int lane = (int)threadIdx.x & 31;
  if (row >= num_rows) return;

  const int first_head = kv_head * group_size;
  const int last_head = min(first_head + group_size, num_query_heads);

  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t table_row = (int64_t)safe_request * stride_table_req;

  // Each lane owns ACC_DIMS contiguous dims, so 32 lanes cover 128 dims with 4
  // and the production 256 with 8.
  constexpr int ACC_DIMS = (LDS_DIM > QSA_LDS_DIM_128) ? 8 : 4;
  const int nwarps = (int)(blockDim.x >> 5);

  // The index -> logical page -> physical page resolution does not depend on the
  // query head, so warp 0 resolves it once per tile into LDS and the whole group
  // shares it. Both loops used to redo it per column in every warp.
  __shared__ int64_t s_koff[BLOCK_N];
  __shared__ int64_t s_voff[BLOCK_N];
  // Probabilities, kept as fp16 to match Triton's `probabilities.to(dtype)`.
  __shared__ half s_p[QSA_MAX_GROUP][BLOCK_N];
  // This tile's K and V, staged once and shared by every warp in the group.
  __shared__ half s_k[BLOCK_N][LDS_DIM];
  __shared__ half s_v[BLOCK_N][LDS_DIM];
  // This warp's query row, staged once per head: the score loop re-reads it for
  // every column, so it must not be a global load there.
  __shared__ half s_q[QSA_MAX_GROUP][LDS_DIM];

  const int tile_begin = split_id * num_tiles / num_splits;
  const int tile_end = (split_id + 1) * num_tiles / num_splits;

  // One warp per query head, but group_size is not bounded by the warp count:
  // gfx1030 TP4 gives 6 query heads to 1 KV head, so every warp walks the heads
  // `warp, warp + NWARPS, ...`. All warps run the same iteration count even when
  // they run out of heads, because the tile loop contains barriers.
  const int head_iters = (group_size + nwarps - 1) / nwarps;

  for (int hi = 0; hi < head_iters; ++hi) {
    const int head = first_head + warp + hi * nwarps;
    const bool active = head < last_head;

    const half* q_row = active
        ? q + (int64_t)row * stride_q_row + (int64_t)head * stride_q_head
        : q;
    if (active) {
      if (vec8 != 0) {
        for (int i = lane; i < (head_dim >> 3); i += 32) {
          *reinterpret_cast<uint4*>(&s_q[warp][8 * i]) =
              *reinterpret_cast<const uint4*>(q_row + 8 * i);
        }
      } else {
        for (int i = lane; i < head_dim; i += 32) {
          s_q[warp][i] = q_row[i];
        }
      }
    }

    float running_max = -1.0e20f;
    float running_sum = 0.0f;
    float acc[ACC_DIMS];
#pragma unroll
    for (int k = 0; k < ACC_DIMS; ++k) acc[k] = 0.0f;

    for (int tile = tile_begin; tile < tile_end; ++tile) {
      const int tile_col0 = tile * BLOCK_N;
      if (warp == 0) {
        for (int c = lane; c < BLOCK_N; c += 32) {
          const int col = tile_col0 + c;
          int32_t logical_token = -1;
          if (col < topk) {
            logical_token = (int32_t)logical_indices[
                (int64_t)row * stride_indices_row + col];
          }
          const int32_t safe_token = max(logical_token, 0);
          const int logical_page = safe_token / page_size;
          const int page_offset = safe_token % page_size;
          bool valid = valid_request && (logical_token >= 0) &&
              (logical_page < page_table_width);
          int32_t physical_page = -1;
          if (valid) {
            physical_page = block_table[table_row +
                min(logical_page, page_table_width - 1)];
            valid = (physical_page >= 0) && (physical_page < num_cache_blocks);
          }
          s_koff[c] = valid
              ? (int64_t)physical_page * stride_k_block +
                    (int64_t)page_offset * stride_k_token +
                    (int64_t)kv_head * stride_k_head
              : (int64_t)-1;
          s_voff[c] = valid
              ? (int64_t)physical_page * stride_v_block +
                    (int64_t)page_offset * stride_v_token +
                    (int64_t)kv_head * stride_v_head
              : (int64_t)-1;
        }
      }
      __syncthreads();

      // Stage this tile's K and V into LDS once for the whole group. Without
      // this each of the group_size warps re-reads the same rows from global
      // (6x on this model), and the loads are the measured bottleneck.
      const int nthreads = (int)blockDim.x;
      if (vec8 != 0) {
        const int n8 = head_dim >> 3;
        const int chunks = BLOCK_N * n8;
        for (int i = (int)threadIdx.x; i < chunks; i += nthreads) {
          const int c = i / n8;
          const int j = i - c * n8;
          const int64_t koff = s_koff[c];
          uint4 kv = make_uint4(0u, 0u, 0u, 0u);
          if (koff >= 0) {
            kv = *reinterpret_cast<const uint4*>(k_cache + koff + 8 * j);
          }
          *reinterpret_cast<uint4*>(&s_k[c][8 * j]) = kv;
          const int64_t voff = s_voff[c];
          uint4 vv = make_uint4(0u, 0u, 0u, 0u);
          if (voff >= 0) {
            vv = *reinterpret_cast<const uint4*>(v_cache + voff + 8 * j);
          }
          *reinterpret_cast<uint4*>(&s_v[c][8 * j]) = vv;
        }
      } else {
        for (int i = (int)threadIdx.x; i < BLOCK_N * head_dim; i += nthreads) {
          const int c = i / head_dim;
          const int d = i - c * head_dim;
          const int64_t koff = s_koff[c];
          s_k[c][d] = (koff >= 0) ? k_cache[koff + d] : __float2half(0.0f);
          const int64_t voff = s_voff[c];
          s_v[c][d] = (voff >= 0) ? v_cache[voff + d] : __float2half(0.0f);
        }
      }
      __syncthreads();

      // Warp-level softmax statistics for this tile; every lane computes its own
      // columns' scores first, so max/sum need one shuffle reduction each.
      float col_score[BLOCK_N / 32 > 0 ? BLOCK_N / 32 : 1];
      // Softmax is computed in the log2 domain. Scale each score before the
      // max reduction so `p = score - max` subtracts two values from the same
      // domain: scaling a -1e20 masked score after the max leaves a rounding
      // residue that exp2 turns into inf.
      const float scale_log2 = (float)(1.4426950408889634 / sqrt((double)head_dim));
      float lane_max = -1.0e20f;
      int n_cols = 0;
      for (int c = lane; c < BLOCK_N; c += 32, ++n_cols) {
        float score = -INFINITY;
        const int64_t koff = s_koff[c];
        if (active && (koff >= 0)) {
          const half* k_row = &s_k[c][0];
          float dot = 0.0f;
          if (vec8 != 0) {
            // 8 halves (four fdot2) per 16-byte load instead of one fdot2 per
            // 4-byte load. Same accumulation order, so results are unchanged.
            const uint4* k4 = reinterpret_cast<const uint4*>(k_row);
            const uint4* q4 = reinterpret_cast<const uint4*>(&s_q[warp][0]);
            const int n8 = head_dim >> 3;
            for (int i = 0; i < n8; ++i) {
              const uint4 kv = k4[i];
              const uint4 qv = q4[i];
              dot = __builtin_amdgcn_fdot2(
                  *reinterpret_cast<const half2*>(&kv.x),
                  *reinterpret_cast<const half2*>(&qv.x), dot, false);
              dot = __builtin_amdgcn_fdot2(
                  *reinterpret_cast<const half2*>(&kv.y),
                  *reinterpret_cast<const half2*>(&qv.y), dot, false);
              dot = __builtin_amdgcn_fdot2(
                  *reinterpret_cast<const half2*>(&kv.z),
                  *reinterpret_cast<const half2*>(&qv.z), dot, false);
              dot = __builtin_amdgcn_fdot2(
                  *reinterpret_cast<const half2*>(&kv.w),
                  *reinterpret_cast<const half2*>(&qv.w), dot, false);
            }
          } else {
#pragma unroll 4
            for (int d2 = 0; d2 < head_dim / 2; ++d2) {
              dot = __builtin_amdgcn_fdot2(
                  *reinterpret_cast<const half2*>(k_row + 2 * d2),
                  *reinterpret_cast<const half2*>(&s_q[warp][2 * d2]), dot, false);
            }
          }
          score = dot;
        }
        const float scaled = score * scale_log2;
        col_score[n_cols] = scaled;
        lane_max = fmaxf(lane_max, scaled);
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        lane_max = fmaxf(lane_max, __shfl_xor(lane_max, s));
      }
      // Rescale the running statistics and accumulator onto the new maximum.
      const float next_max = fmaxf(running_max, lane_max);
      const float alpha = qsa_exp2(running_max - next_max);
      running_sum *= alpha;
#pragma unroll
      for (int k = 0; k < ACC_DIMS; ++k) acc[k] *= alpha;
      running_max = next_max;

      float lane_sum = 0.0f;
      n_cols = 0;
      for (int c = lane; c < BLOCK_N; c += 32, ++n_cols) {
        const float p = col_score[n_cols] - next_max;
        const float prob = qsa_exp2(p);
        lane_sum += prob;
        s_p[warp][c] = __float2half_rn(prob);
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        lane_sum += __shfl_xor(lane_sum, s);
      }
      running_sum += lane_sum;
      __syncthreads();

      // P @ V: this lane owns the contiguous dims [ACC_DIMS*lane, ACC_DIMS*(lane+1)).
      // An invalid column has p == 0 exactly (its score is -1e20) and its staged V
      // row is zero-filled, so the loop needs no per-column validity test and can
      // be unrolled to keep several LDS reads in flight.
      const int cols_here = max(0, min(BLOCK_N, topk - tile_col0));
      if (active) {
#pragma unroll 4
        for (int c = 0; c < cols_here; ++c) {
          const float p = __half2float(s_p[warp][c]);
          const half* v_row = &s_v[c][0];
          const int d0 = ACC_DIMS * lane;
          if ((vec8 != 0) && (d0 + ACC_DIMS <= head_dim)) {
            // One 16-byte LDS read covers all of this lane's dims; the 8-dim
            // case needs the wider load (and is the only one that is 16-byte
            // aligned, since d0 is 8*lane then).
            if constexpr (ACC_DIMS > 4) {
              const uint4 vv = *reinterpret_cast<const uint4*>(v_row + d0);
              const half2 h0 = *reinterpret_cast<const half2*>(&vv.x);
              const half2 h1 = *reinterpret_cast<const half2*>(&vv.y);
              const half2 h2 = *reinterpret_cast<const half2*>(&vv.z);
              const half2 h3 = *reinterpret_cast<const half2*>(&vv.w);
              acc[0] += p * __half2float(__low2half(h0));
              acc[1] += p * __half2float(__high2half(h0));
              acc[2] += p * __half2float(__low2half(h1));
              acc[3] += p * __half2float(__high2half(h1));
              acc[4] += p * __half2float(__low2half(h2));
              acc[5] += p * __half2float(__high2half(h2));
              acc[6] += p * __half2float(__low2half(h3));
              acc[7] += p * __half2float(__high2half(h3));
            } else {
              const uint2 vv = *reinterpret_cast<const uint2*>(v_row + d0);
              const half2 lo = *reinterpret_cast<const half2*>(&vv.x);
              const half2 hi = *reinterpret_cast<const half2*>(&vv.y);
              acc[0] += p * __half2float(__low2half(lo));
              acc[1] += p * __half2float(__high2half(lo));
              acc[2] += p * __half2float(__low2half(hi));
              acc[3] += p * __half2float(__high2half(hi));
            }
          } else {
#pragma unroll
            for (int k = 0; k < ACC_DIMS; ++k) {
              const int d = d0 + k;
              if (d < head_dim) acc[k] += p * __half2float(v_row[d]);
            }
          }
        }
      }
      __syncthreads();
    }

    if (!active) continue;

    const bool has_values = running_sum > 0.0f;
    float normalized[ACC_DIMS];
#pragma unroll
    for (int k = 0; k < ACC_DIMS; ++k) {
      normalized[k] = has_values ? acc[k] / fmaxf(running_sum, 1.0e-20f) : 0.0f;
    }
    if (single_split != 0) {
#pragma unroll
      for (int k = 0; k < ACC_DIMS; ++k) {
        const int d = ACC_DIMS * lane + k;
        if (d < head_dim) {
          output[(int64_t)row * stride_output_row +
                 (int64_t)head * stride_output_head + d] =
              __float2half_rn(normalized[k]);
        }
      }
      continue;
    }
    const int64_t partial_base =
        ((int64_t)split_id * num_rows + row) * num_query_heads + head;
#pragma unroll
    for (int k = 0; k < ACC_DIMS; ++k) {
      const int d = ACC_DIMS * lane + k;
      if (d < head_dim) {
        partial_output[partial_base * head_dim + d] = normalized[k];
      }
    }
    if (lane == 0) {
      partial_lse[(int64_t)((split_id * num_rows + row)) * num_query_heads + head] =
          has_values ? running_max + log2f(fmaxf(running_sum, 1.0e-20f))
                     : -INFINITY;
    }
  }
}

__global__ void qsa_merge_splitk_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_lse, half* __restrict__ output,
    int64_t stride_output_row, int64_t stride_output_head, int num_rows,
    int num_query_heads, int num_splits, int head_dim) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int d = (int)threadIdx.x;
  if ((row >= num_rows) || (head >= num_query_heads) || (d >= head_dim)) return;

  // The per-split log-sum-exp is identical for every thread in the block, and
  // the split loop is the whole cost of this kernel on small batches (it was
  // 66 us of 110 us at rows=3): stage the lse once in LDS and give the weighted
  // sum four independent accumulator chains instead of one 64-deep chain.
  extern __shared__ float s_lse[];
  const int64_t lse_base = (int64_t)row * num_query_heads + head;
  const int64_t lse_stride = (int64_t)num_rows * num_query_heads;
  for (int s = (int)threadIdx.x; s < num_splits; s += (int)blockDim.x) {
    s_lse[s] = partial_lse[(int64_t)s * lse_stride + lse_base];
  }
  __syncthreads();

  float lse_max0 = -INFINITY;
  float lse_max1 = -INFINITY;
  float lse_max2 = -INFINITY;
  float lse_max3 = -INFINITY;
  int s = 0;
  for (; s + 3 < num_splits; s += 4) {
    lse_max0 = fmaxf(lse_max0, s_lse[s]);
    lse_max1 = fmaxf(lse_max1, s_lse[s + 1]);
    lse_max2 = fmaxf(lse_max2, s_lse[s + 2]);
    lse_max3 = fmaxf(lse_max3, s_lse[s + 3]);
  }
  for (; s < num_splits; ++s) lse_max0 = fmaxf(lse_max0, s_lse[s]);
  const float lse_max =
      fmaxf(fmaxf(lse_max0, lse_max1), fmaxf(lse_max2, lse_max3));

  float denominator = 0.0f;
  float merged = 0.0f;
  if (lse_max > -INFINITY) {
    float den1 = 0.0f;
    float den2 = 0.0f;
    float den3 = 0.0f;
    float mer1 = 0.0f;
    float mer2 = 0.0f;
    float mer3 = 0.0f;
    const float* pout = partial_output + lse_base * head_dim + d;
    const int64_t pout_stride = lse_stride * head_dim;
    s = 0;
    for (; s + 3 < num_splits; s += 4) {
      const float w0 = qsa_exp2(s_lse[s] - lse_max);
      const float w1 = qsa_exp2(s_lse[s + 1] - lse_max);
      const float w2 = qsa_exp2(s_lse[s + 2] - lse_max);
      const float w3 = qsa_exp2(s_lse[s + 3] - lse_max);
      denominator += w0;
      den1 += w1;
      den2 += w2;
      den3 += w3;
      merged += pout[(int64_t)s * pout_stride] * w0;
      mer1 += pout[(int64_t)(s + 1) * pout_stride] * w1;
      mer2 += pout[(int64_t)(s + 2) * pout_stride] * w2;
      mer3 += pout[(int64_t)(s + 3) * pout_stride] * w3;
    }
    for (; s < num_splits; ++s) {
      const float w = qsa_exp2(s_lse[s] - lse_max);
      denominator += w;
      merged += pout[(int64_t)s * pout_stride] * w;
    }
    denominator += (den1 + den2) + den3;
    merged += (mer1 + mer2) + mer3;
  }
  output[(int64_t)row * stride_output_row + (int64_t)head * stride_output_head + d] =
      __float2half_rn(denominator > 0.0f ? merged / denominator : 0.0f);
}

}  // namespace

torch::Tensor qsa_sparse_paged_attention_rdna2(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    const at::Tensor& logical_indices, const at::Tensor& block_table,
    const at::Tensor& token_to_req, c10::optional<at::Tensor> out) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf &&
                  v_cache.scalar_type() == at::kHalf,
              "qsa_sparse_paged_attention_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(q.dim() == 3, "qsa_sparse_paged_attention_rdna2: q must be 3D");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
              "qsa_sparse_paged_attention_rdna2: k_cache/v_cache must share a "
              "4D shape");
  TORCH_CHECK(logical_indices.dim() == 2 &&
                  logical_indices.size(0) == q.size(0),
              "qsa_sparse_paged_attention_rdna2: logical_indices must be "
              "[rows, topk]");
  TORCH_CHECK(logical_indices.scalar_type() == at::kInt ||
                  logical_indices.scalar_type() == at::kLong,
              "qsa_sparse_paged_attention_rdna2: logical_indices must be int32 "
              "or int64");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  token_to_req.scalar_type() == at::kInt,
              "qsa_sparse_paged_attention_rdna2: block_table/token_to_req int32 "
              "only");
  TORCH_CHECK(token_to_req.numel() == q.size(0),
              "qsa_sparse_paged_attention_rdna2: token_to_req must have one "
              "entry per row");
  TORCH_CHECK(q.stride(2) == 1 && k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "qsa_sparse_paged_attention_rdna2: q/K/V must be contiguous in the "
              "last dim");
  TORCH_CHECK(logical_indices.stride(1) == 1 && block_table.stride(1) == 1,
              "qsa_sparse_paged_attention_rdna2: indices/block_table must be "
              "contiguous in the last dim");
  TORCH_CHECK(logical_indices.size(1) > 0 && k_cache.size(0) > 0 &&
                  k_cache.size(1) > 0 && k_cache.size(2) > 0 &&
                  block_table.size(0) > 0 && block_table.size(1) > 0,
              "qsa_sparse_paged_attention_rdna2: cache, table and selection must "
              "be nonempty");

  at::Tensor result;
  if (out.has_value() && out->defined()) {
    result = *out;
    TORCH_CHECK(result.sizes() == q.sizes() && result.scalar_type() == at::kHalf,
                "qsa_sparse_paged_attention_rdna2: out must match q");
    TORCH_CHECK(result.stride(2) == 1,
                "qsa_sparse_paged_attention_rdna2: out must be contiguous in the "
                "last dim");
  } else {
    result = at::empty_like(q);
  }
  const int64_t rows = q.size(0);
  if (rows == 0) return result;

  const int64_t num_query_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t num_kv_heads = k_cache.size(2);
  TORCH_CHECK(num_kv_heads > 0 && num_query_heads % num_kv_heads == 0,
              "qsa_sparse_paged_attention_rdna2: query heads must be a multiple "
              "of kv heads (GQA)");
  const int64_t group_size = num_query_heads / num_kv_heads;
  TORCH_CHECK(group_size >= 1,
              "qsa_sparse_paged_attention_rdna2: group_size must be >= 1");
  const int64_t topk = logical_indices.size(1);
  const int64_t page_size = k_cache.size(1);
  const int64_t page_table_width = block_table.size(1);
  const int64_t num_requests = block_table.size(0);
  const int64_t num_cache_blocks = k_cache.size(0);

  // Tiling choice only -- any BLOCK_N is correct. BLOCK_N is capped at 32 because
  // the staged K/V tile costs 2 * BLOCK_N * LDS_DIM halves of LDS: BLOCK_N=64
  // needs ~35 KB at head_dim 128 and drops the kernel to one workgroup per CU,
  // which measured worse than the load redundancy it removes. head_dim 256 needs
  // ~39 KB at BLOCK_N=32, so that width also runs at one workgroup per CU.
  const int64_t base_programs = rows * num_kv_heads;
  int block_n = 32;
  int target_splits = 1;
  if (base_programs <= 24) {
    target_splits = 64;
  } else if (base_programs <= 32) {
    target_splits = 16;
  } else if (base_programs <= 64) {
    target_splits = 8;
  } else if (base_programs <= 128) {
    target_splits = 4;
  } else if (base_programs <= 256) {
    target_splits = 8;
  } else if (base_programs <= 512) {
    target_splits = 4;
  }
  const int64_t num_tiles = (topk + block_n - 1) / block_n;
  int64_t max_useful_splits = 1;
  while (max_useful_splits * 2 <= num_tiles) max_useful_splits *= 2;
  const int64_t num_splits = std::min<int64_t>(max_useful_splits, target_splits);
  TORCH_CHECK(num_splits >= 1 && num_tiles >= 1, "internal tiling error");

  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor partial_output;
  at::Tensor partial_lse;
  float* partial_output_ptr = nullptr;
  float* partial_lse_ptr = nullptr;
  if (num_splits > 1) {
    partial_output = at::empty({num_splits, rows, num_query_heads, head_dim},
                               at::dtype(at::kFloat).device(q.device()));
    partial_lse = at::empty({num_splits, rows, num_query_heads},
                            at::dtype(at::kFloat).device(q.device()));
    partial_output_ptr = partial_output.mutable_data_ptr<float>();
    partial_lse_ptr = partial_lse.mutable_data_ptr<float>();
  }
  // The 8-wide (16-byte) load paths need every address they touch to be
  // 16-byte aligned. That holds whenever head_dim and every cache stride is a
  // multiple of 8 halves; otherwise the kernel falls back to half2 loads.
  const bool vec8 =
      (head_dim % 8 == 0) && (q.stride(0) % 8 == 0) && (q.stride(1) % 8 == 0) &&
      (k_cache.stride(0) % 8 == 0) && (k_cache.stride(1) % 8 == 0) &&
      (k_cache.stride(2) % 8 == 0) && (v_cache.stride(0) % 8 == 0) &&
      (v_cache.stride(1) % 8 == 0) && (v_cache.stride(2) % 8 == 0) &&
      ((reinterpret_cast<uintptr_t>(q.const_data_ptr()) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(k_cache.const_data_ptr()) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(v_cache.const_data_ptr()) & 15) == 0);
  // The staged K/V tiles live in LDS rows of QSA_LDS_DIM_{128,256} halves.
  TORCH_CHECK(head_dim <= 256,
              "qsa_sparse_paged_attention_rdna2: head_dim must be <= 256");
  dim3 grid((unsigned)rows, (unsigned)num_kv_heads, (unsigned)num_splits);
  const unsigned attn_threads =
      (unsigned)(std::min<int64_t>(group_size, (int64_t)QSA_MAX_GROUP) * 32);
  half* out_ptr = reinterpret_cast<half*>(result.mutable_data_ptr());

  auto launch = [&](auto idx_tag) {
    using TIdx = decltype(idx_tag);
    auto* idx_ptr = reinterpret_cast<const TIdx*>(
        logical_indices.const_data_ptr());
#define QSA_ATTN_LAUNCH(BN, LD)                                                \
  qsa_sparse_paged_gqa_splitk_kernel<BN, LD, TIdx>                             \
      <<<grid, attn_threads, 0, stream>>>(                                    \
          reinterpret_cast<const half*>(q.const_data_ptr()),                   \
          reinterpret_cast<const half*>(k_cache.const_data_ptr()),             \
          reinterpret_cast<const half*>(v_cache.const_data_ptr()), idx_ptr,    \
          block_table.const_data_ptr<int32_t>(),                               \
          token_to_req.const_data_ptr<int32_t>(), partial_output_ptr,          \
          partial_lse_ptr, out_ptr, q.stride(0), q.stride(1),                  \
          k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),             \
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),             \
          logical_indices.stride(0), block_table.stride(0), result.stride(0),  \
          result.stride(1), (int)rows, num_cache_blocks, (int)num_requests,    \
          (int)topk, (int)page_size, (int)page_table_width, (int)group_size,   \
          (int)head_dim, (int)num_query_heads, (int)num_splits, (int)num_tiles, \
          (num_splits == 1) ? 1 : 0, vec8 ? 1 : 0)
    if (head_dim <= 128) {
      if (block_n <= 16) {
        QSA_ATTN_LAUNCH(16, QSA_LDS_DIM_128);
      } else if (block_n <= 32) {
        QSA_ATTN_LAUNCH(32, QSA_LDS_DIM_128);
      } else {
        QSA_ATTN_LAUNCH(64, QSA_LDS_DIM_128);
      }
    } else {
      // 64 columns of 256-wide LDS rows need 74 KB and do not fit.
      TORCH_CHECK(block_n <= 32,
                  "internal tiling error: BLOCK_N > 32 at head_dim 256");
      if (block_n <= 16) {
        QSA_ATTN_LAUNCH(16, QSA_LDS_DIM_256);
      } else {
        QSA_ATTN_LAUNCH(32, QSA_LDS_DIM_256);
      }
    }
#undef QSA_ATTN_LAUNCH
  };
  if (logical_indices.scalar_type() == at::kInt) {
    launch((int32_t)0);
  } else {
    launch((int64_t)0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (num_splits > 1) {
    const int64_t merge_threads =
        std::min<int64_t>(1024, ((head_dim + 31) / 32) * 32);
    dim3 merge_grid((unsigned)rows, (unsigned)num_query_heads);
    qsa_merge_splitk_kernel<<<merge_grid, (unsigned)merge_threads,
                              (size_t)num_splits * sizeof(float), stream>>>(
        partial_output_ptr, partial_lse_ptr, out_ptr, result.stride(0),
        result.stride(1), (int)rows, (int)num_query_heads, (int)num_splits,
        (int)head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}
