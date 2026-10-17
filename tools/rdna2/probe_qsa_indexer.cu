// Standalone hipcc probe for the three QSA indexer kernels in
// csrc/rocm/qsa_rdna2.cu.  Extracted verbatim (plus -DVARIANT switches) to
// iterate in ~5 s instead of a 30 s-2 min ninja pass.  Clock-warm + min-of-5.
//
//   hipcc -O3 --offload-arch=gfx1030 -DVARIANT=1 probe_qsa_indexer.cu -o probe
//   HIP_VISIBLE_DEVICES=2 ./probe
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

static const int ROWS[] = {3, 6, 12, 24, 64, 256, 512, 1024, 2048};
static const int NROWS = 9;
static const int REPS = 5;

// ---------------------------------------------------------------------------
// clock warm
// ---------------------------------------------------------------------------
__global__ void warm_kernel(float* out, int iters) {
  float a = threadIdx.x * 1e-6f, b = 1.0001f;
  for (int i = 0; i < iters; i++) { a = fmaf(a, b, 1e-7f); b = fmaf(b, a, 1e-7f); }
  if (a == 12345.f) out[blockIdx.x] = a + b;
}

static void warm(int ms) {
  float* p; CK(hipMalloc(&p, 4096));
  hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
  CK(hipEventRecord(s));
  while (true) {
    warm_kernel<<<1024, 256>>>(p, 20000);
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    float t; CK(hipEventElapsedTime(&t, s, e));
    if (t > ms) break;
  }
  CK(hipFree(p));
}

template <typename F>
static float timeit(F launch, int iters) {
  hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
  float best = 1e30f;
  for (int r = 0; r < REPS; r++) {
    CK(hipEventRecord(s));
    for (int i = 0; i < iters; i++) launch();
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    float t; CK(hipEventElapsedTime(&t, s, e));
    best = std::min(best, t * 1000.0f / iters);
  }
  CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
  return best;
}

static int iters_for(int rows) { return std::max(20, 200000 / rows); }

// ===========================================================================
// 1. store_cache_rows
// ===========================================================================
__device__ __forceinline__ uint4 ld_u4_(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void st_u4_(void* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }

template <int VEC_COUNT>
__global__ void store_tmpl(const char* __restrict__ rows, const int64_t* __restrict__ slots,
                           char* __restrict__ cache, int num_rows, int num_blocks,
                           int64_t stride_rows_row, int64_t stride_cache_block,
                           int64_t stride_cache_token, int PAGE_SIZE, int rows_per_block) {
  const int row = blockIdx.x * rows_per_block + threadIdx.x / VEC_COUNT;
  if (row >= num_rows) return;
  const int64_t slot = slots[row];
  if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
  char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
      + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
  const char* row_ptr = rows + (size_t)row * stride_rows_row;
  const int v = threadIdx.x % VEC_COUNT;
  st_u4_(reinterpret_cast<uint4*>(cache_ptr + (size_t)v * 16),
         ld_u4_(reinterpret_cast<const uint4*>(row_ptr + (size_t)v * 16)));
}

__global__ void store_base(const char* __restrict__ rows, const int64_t* __restrict__ slots,
                           char* __restrict__ cache, int num_rows, int num_blocks,
                           int64_t stride_rows_row, int64_t stride_cache_block,
                           int64_t stride_cache_token, int PAGE_SIZE, int rows_per_block,
                           int vec_count) {
  const int row = blockIdx.x * rows_per_block + threadIdx.x / vec_count;
  if (row >= num_rows) return;
  const int64_t slot = slots[row];
  if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
  char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
      + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
  const char* row_ptr = rows + (size_t)row * stride_rows_row;
  const int v = threadIdx.x % vec_count;
  st_u4_(reinterpret_cast<uint4*>(cache_ptr + (size_t)v * 16),
         ld_u4_(reinterpret_cast<const uint4*>(row_ptr + (size_t)v * 16)));
}

// ===========================================================================
// 2. compress_groups
// ===========================================================================
template <int BLOCK_D, int COMPRESS_RATIO, int VARIANT>
__global__ void compress_kernel(
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
  const int query_row_start = valid_request ? query_start_loc[safe_request] : 0;
  const int query_row_end = valid_request ? query_start_loc[safe_request + 1] : 0;
  const int64_t end_position = logical_positions[row];
  const int64_t compressed_slot = compressed_slots[row];
  const int64_t chunk_start_position = end_position - (row - query_row_start);
  const int compressor_state_block =
      valid_request ? compressor_state_table[safe_request] : -1;
  const bool valid_compressor_state_block =
      (compressor_state_block >= 0) && (compressor_state_block < num_compressor_state_blocks);
  const bool valid_row = valid_request && (row >= query_row_start) && (row < query_row_end)
      && (end_position >= COMPRESS_RATIO - 1) && (compressed_slot >= 0);
  const int safe_compressor_block = max(compressor_state_block, 0);

#if VARIANT >= 2
  // state ring size is a power of two -> replace the int64 software modulo.
  const int state_mask = COMPRESSOR_STATE_SIZE - 1;
#endif
  for (int b = threadIdx.x; b < HEAD_DIM; b += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int go = 0; go < COMPRESS_RATIO; go++) {
      const int64_t position = end_position - (COMPRESS_RATIO - 1 - go);
      const bool use_raw = position >= chunk_start_position;
      const int raw_row = (int)(query_row_start + position - chunk_start_position);
      const bool raw_ok = valid_row && use_raw && (raw_row >= query_row_start)
          && (raw_row < query_row_end) && (raw_row < num_rows);
      const bool state_ok = valid_row && !use_raw && valid_compressor_state_block;
      float v = 0.f;
      if (raw_ok) {
        v = __half2float(raw_keys[raw_row * stride_raw_row + b * stride_raw_dim]);
      } else if (state_ok) {
#if VARIANT >= 2
        const int tok = (int)(position & state_mask);
#else
        const int tok = (int)(position % COMPRESSOR_STATE_SIZE);
#endif
        v = __half2float(compressor_state_cache[
            safe_compressor_block * stride_compressor_state_block
            + tok * stride_compressor_state_token + b * stride_compressor_state_dim]);
      }
      acc += v;
    }
    pooled[row * stride_pooled_row + b * stride_pooled_dim] = __float2half(acc / float(COMPRESS_RATIO));
  }

  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (threadIdx.x < 3 && VARIANT < 3) {
    const int p = (int)threadIdx.x;
    if (LOAD_ROPE_POSITIONS != 0) {
      const bool first_from_raw = first_position >= chunk_start_position;
      const int raw_first_row = (int)(query_row_start + first_position - chunk_start_position);
      const bool raw_first_ok = valid_row && first_from_raw
          && (raw_first_row >= query_row_start) && (raw_first_row < query_row_end)
          && (raw_first_row < num_rows);
      const bool state_first_ok = valid_row && !first_from_raw && valid_compressor_state_block;
      int64_t v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
#if VARIANT >= 2
        const int tok = (int)(first_position & state_mask);
#else
        const int tok = (int)(first_position % COMPRESSOR_STATE_SIZE);
#endif
        v = rope_cache[safe_compressor_block * stride_rope_block + tok * stride_rope_token
                       + p * stride_rope_dim];
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    } else {
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
}

// ===========================================================================
// 3. expand_indices
// ===========================================================================
template <typename TPos, typename TSeq, int CR, int VARIANT>
__global__ void expand_kernel(
    const int32_t* __restrict__ block_indices,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    const int32_t* __restrict__ token_to_req, int32_t* __restrict__ out,
    int64_t stride_blocks_row, int64_t stride_blocks_column,
    int64_t stride_out_row, int64_t stride_out_column, int rows,
    int num_requests, int block_topk, int output_width) {
  const int row = blockIdx.x;
  const int col = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
  if ((row >= rows) || (col >= output_width)) return;

  const int64_t query_position = (int64_t)query_positions[row];
  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t complete_blocks =
      min(min((query_position + 1) / CR, sequence_length / CR),
          (int64_t)block_topk);
  const int expanded_count = (int)complete_blocks * CR;
  const int tail_start = (int)((query_position + 1) / CR) * CR;
  const int tail_count = (int)((query_position + 1) - tail_start);

  int token = -1;
  if (col < expanded_count) {
    const int block_rank = col / CR;
    const int offset = col - block_rank * CR;
    const int safe_rank = min(block_rank, block_topk - 1);
    const int64_t block = block_indices[(int64_t)row * stride_blocks_row
                                        + (int64_t)safe_rank * stride_blocks_column];
    token = (int)(block * CR + offset);
  } else {
    const int tail_offset = col - expanded_count;
    if ((tail_offset < tail_count) && (tail_offset < CR - 1)) {
      token = tail_start + tail_offset;
    }
  }
  const bool valid = (token >= 0) && ((int64_t)token < sequence_length);
  out[(int64_t)row * stride_out_row + (int64_t)col * stride_out_column] = valid ? token : -1;
}

// same, with the runtime-division generic path (baseline behaviour)
template <typename TPos, typename TSeq>
__global__ void expand_generic(
    const int32_t* __restrict__ block_indices,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    const int32_t* __restrict__ token_to_req, int32_t* __restrict__ out,
    int64_t stride_blocks_row, int64_t stride_blocks_column,
    int64_t stride_out_row, int64_t stride_out_column, int rows,
    int num_requests, int block_topk, int compress_ratio, int output_width) {
  const int row = blockIdx.x;
  const int col = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
  if ((row >= rows) || (col >= output_width)) return;

  const int64_t query_position = (int64_t)query_positions[row];
  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t complete_blocks =
      min(min((query_position + 1) / compress_ratio, sequence_length / compress_ratio),
          (int64_t)block_topk);
  const int expanded_count = (int)complete_blocks * compress_ratio;
  const int tail_start = (int)((query_position + 1) / compress_ratio) * compress_ratio;
  const int tail_count = (int)((query_position + 1) - tail_start);

  int token = -1;
  if (col < expanded_count) {
    const int block_rank = col / compress_ratio;
    const int offset = col - block_rank * compress_ratio;
    const int safe_rank = min(block_rank, block_topk - 1);
    const int64_t block = block_indices[(int64_t)row * stride_blocks_row
                                        + (int64_t)safe_rank * stride_blocks_column];
    token = (int)(block * compress_ratio + offset);
  } else {
    const int tail_offset = col - expanded_count;
    if ((tail_offset < tail_count) && (tail_offset < compress_ratio - 1)) {
      token = tail_start + tail_offset;
    }
  }
  const bool valid = (token >= 0) && ((int64_t)token < sequence_length);
  out[(int64_t)row * stride_out_row + (int64_t)col * stride_out_column] = valid ? token : -1;
}

// One thread per compressed rank: it owns CR consecutive output columns, so the
// block index is loaded once and the preamble is computed CR times less often.
// Each thread owns CPT columns strided by blockDim.x, so the stores stay fully
// coalesced while the preamble and the grid are both CPT times smaller.
template <typename TPos, typename TSeq, int CR, int CPT>
__global__ void expand_multi(
    const int32_t* __restrict__ block_indices,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    const int32_t* __restrict__ token_to_req, int32_t* __restrict__ out,
    int64_t stride_blocks_row, int64_t stride_blocks_column,
    int64_t stride_out_row, int64_t stride_out_column, int rows,
    int num_requests, int block_topk, int output_width) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int base = (int)blockIdx.y * (blockDim.x * CPT) + (int)threadIdx.x;

  const int64_t query_position = (int64_t)query_positions[row];
  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t complete_blocks =
      min(min((query_position + 1) / CR, sequence_length / CR), (int64_t)block_topk);
  const int expanded_count = (int)complete_blocks * CR;
  const int tail_start = (int)((query_position + 1) / CR) * CR;
  const int tail_count = (int)((query_position + 1) - tail_start);

#pragma unroll
  for (int o = 0; o < CPT; o++) {
    const int col = base + o * (int)blockDim.x;
    if (col >= output_width) continue;
    int token = -1;
    if (col < expanded_count) {
      const int block_rank = col / CR;
      const int safe_rank = min(block_rank, block_topk - 1);
      const int64_t block = block_indices[(int64_t)row * stride_blocks_row
                                          + (int64_t)safe_rank * stride_blocks_column];
      token = (int)(block * CR + (col - block_rank * CR));
    } else {
      const int tail_offset = col - expanded_count;
      if ((tail_offset < tail_count) && (tail_offset < CR - 1)) {
        token = tail_start + tail_offset;
      }
    }
    out[(int64_t)row * stride_out_row + (int64_t)col * stride_out_column] =
        ((token >= 0) && ((int64_t)token < sequence_length)) ? token : -1;
  }
}

// One thread per compressed rank: it owns CR consecutive output columns, so the
// block index is loaded once and the preamble is computed CR times less often.
template <typename TPos, typename TSeq, int CR>
__global__ void expand_vec(
    const int32_t* __restrict__ block_indices,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    const int32_t* __restrict__ token_to_req, int32_t* __restrict__ out,
    int64_t stride_blocks_row, int64_t stride_blocks_column,
    int64_t stride_out_row, int64_t stride_out_column, int rows,
    int num_requests, int block_topk, int output_width) {
  const int row = blockIdx.x;
  const int rank = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
  const int col0 = rank * CR;
  if ((row >= rows) || (col0 >= output_width)) return;

  const int64_t query_position = (int64_t)query_positions[row];
  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max((int)request, 0), num_requests - 1);
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t complete_blocks =
      min(min((query_position + 1) / CR, sequence_length / CR), (int64_t)block_topk);
  const int expanded_count = (int)complete_blocks * CR;
  const int tail_start = (int)((query_position + 1) / CR) * CR;
  const int tail_count = (int)((query_position + 1) - tail_start);

  int tok[CR];
#pragma unroll
  for (int o = 0; o < CR; o++) {
    const int col = col0 + o;
    int token = -1;
    if (col < expanded_count) {
      const int safe_rank = min(rank, block_topk - 1);
      const int64_t block = block_indices[(int64_t)row * stride_blocks_row
                                          + (int64_t)safe_rank * stride_blocks_column];
      token = (int)(block * CR + o);
    } else {
      const int tail_offset = col - expanded_count;
      if ((tail_offset < tail_count) && (tail_offset < CR - 1)) {
        token = tail_start + tail_offset;
      }
    }
    tok[o] = ((token >= 0) && ((int64_t)token < sequence_length)) ? token : -1;
  }
#pragma unroll
  for (int o = 0; o < CR; o++) {
    if (col0 + o < output_width) {
      out[(int64_t)row * stride_out_row + (int64_t)(col0 + o) * stride_out_column] = tok[o];
    }
  }
}

template <int BLOCK_D, int COMPRESS_RATIO, int VARIANT>
__global__ void compress_lds_kernel(
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

  // The preamble depends only on `row`, and the grid is one block per row, so
  // thread 0 computes it once and broadcasts instead of all 128 threads
  // recomputing it (it is the bulk of the kernel's instruction count).
  __shared__ int64_t s_pos[2];
  __shared__ int s_pre[4];
  if (threadIdx.x == 0) {
    const int request = token_to_req[row];
    const bool valid_request = (request >= 0) && (request < num_requests);
    const int safe_request = min(max(request, 0), num_requests - 1);
    const int query_row_start = valid_request ? query_start_loc[safe_request] : 0;
    const int query_row_end = valid_request ? query_start_loc[safe_request + 1] : 0;
    const int64_t end_position = logical_positions[row];
    const int64_t compressed_slot = compressed_slots[row];
    const int compressor_state_block =
        valid_request ? compressor_state_table[safe_request] : -1;
    const bool valid_csb = (compressor_state_block >= 0)
        && (compressor_state_block < num_compressor_state_blocks);
    const bool valid_row = valid_request && (row >= query_row_start)
        && (row < query_row_end) && (end_position >= COMPRESS_RATIO - 1)
        && (compressed_slot >= 0);
    s_pos[0] = end_position;
    s_pos[1] = end_position - (row - query_row_start);
    s_pre[0] = query_row_start;
    s_pre[1] = query_row_end;
    s_pre[2] = (valid_row ? 1 : 0) | (valid_csb ? 2 : 0)
        | (max(compressor_state_block, 0) << 2);
  }
  __syncthreads();
  const int64_t end_position = s_pos[0];
  const int64_t chunk_start_position = s_pos[1];
  const int query_row_start = s_pre[0];
  const int query_row_end = s_pre[1];
  const bool valid_row = (s_pre[2] & 1) != 0;
  const bool valid_compressor_state_block = (s_pre[2] & 2) != 0;
  const int safe_compressor_block = s_pre[2] >> 2;

  for (int b = threadIdx.x; b < HEAD_DIM; b += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int go = 0; go < COMPRESS_RATIO; go++) {
      const int64_t position = end_position - (COMPRESS_RATIO - 1 - go);
      const bool use_raw = position >= chunk_start_position;
      const int raw_row = (int)(query_row_start + position - chunk_start_position);
      const bool raw_ok = valid_row && use_raw && (raw_row >= query_row_start)
          && (raw_row < query_row_end) && (raw_row < num_rows);
      const bool state_ok = valid_row && !use_raw && valid_compressor_state_block;
      float v = 0.f;
      if (raw_ok) {
        v = __half2float(raw_keys[raw_row * stride_raw_row + b * stride_raw_dim]);
      } else if (state_ok) {
        const int tok = (int)(position % COMPRESSOR_STATE_SIZE);
        v = __half2float(compressor_state_cache[
            safe_compressor_block * stride_compressor_state_block
            + tok * stride_compressor_state_token + b * stride_compressor_state_dim]);
      }
      acc += v;
    }
    pooled[row * stride_pooled_row + b * stride_pooled_dim] = __float2half(acc / float(COMPRESS_RATIO));
  }

  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (threadIdx.x < 3) {
    const int p = (int)threadIdx.x;
    if (LOAD_ROPE_POSITIONS != 0) {
      const bool first_from_raw = first_position >= chunk_start_position;
      const int raw_first_row = (int)(query_row_start + first_position - chunk_start_position);
      const bool raw_first_ok = valid_row && first_from_raw
          && (raw_first_row >= query_row_start) && (raw_first_row < query_row_end)
          && (raw_first_row < num_rows);
      const bool state_first_ok = valid_row && !first_from_raw && valid_compressor_state_block;
      int64_t v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
        const int tok = (int)(first_position % COMPRESSOR_STATE_SIZE);
        v = rope_cache[safe_compressor_block * stride_rope_block + tok * stride_rope_token
                       + p * stride_rope_dim];
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    } else {
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
}

template <int BLOCK_D, int COMPRESS_RATIO, int VARIANT, int RPB>
__global__ void compress_rpb_kernel(
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
  // RPB rows share one workgroup: the same threads-per-row, half the
  // workgroups, and the per-row preamble is unchanged.
  const int dims_per_row = 128;
  const int sub = threadIdx.x % dims_per_row;
  const int row = blockIdx.x * RPB + threadIdx.x / dims_per_row;
  if (row >= num_rows) return;

  const int request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int query_row_start = valid_request ? query_start_loc[safe_request] : 0;
  const int query_row_end = valid_request ? query_start_loc[safe_request + 1] : 0;
  const int64_t end_position = logical_positions[row];
  const int64_t compressed_slot = compressed_slots[row];
  const int64_t chunk_start_position = end_position - (row - query_row_start);
  const int compressor_state_block =
      valid_request ? compressor_state_table[safe_request] : -1;
  const bool valid_compressor_state_block =
      (compressor_state_block >= 0) && (compressor_state_block < num_compressor_state_blocks);
  const bool valid_row = valid_request && (row >= query_row_start) && (row < query_row_end)
      && (end_position >= COMPRESS_RATIO - 1) && (compressed_slot >= 0);
  const int safe_compressor_block = max(compressor_state_block, 0);

#if VARIANT >= 2
  // state ring size is a power of two -> replace the int64 software modulo.
  const int state_mask = COMPRESSOR_STATE_SIZE - 1;
#endif
  for (int b = sub; b < HEAD_DIM; b += dims_per_row) {
    float acc = 0.f;
#pragma unroll
    for (int go = 0; go < COMPRESS_RATIO; go++) {
      const int64_t position = end_position - (COMPRESS_RATIO - 1 - go);
      const bool use_raw = position >= chunk_start_position;
      const int raw_row = (int)(query_row_start + position - chunk_start_position);
      const bool raw_ok = valid_row && use_raw && (raw_row >= query_row_start)
          && (raw_row < query_row_end) && (raw_row < num_rows);
      const bool state_ok = valid_row && !use_raw && valid_compressor_state_block;
      float v = 0.f;
      if (raw_ok) {
        v = __half2float(raw_keys[raw_row * stride_raw_row + b * stride_raw_dim]);
      } else if (state_ok) {
#if VARIANT >= 2
        const int tok = (int)(position & state_mask);
#else
        const int tok = (int)(position % COMPRESSOR_STATE_SIZE);
#endif
        v = __half2float(compressor_state_cache[
            safe_compressor_block * stride_compressor_state_block
            + tok * stride_compressor_state_token + b * stride_compressor_state_dim]);
      }
      acc += v;
    }
    pooled[row * stride_pooled_row + b * stride_pooled_dim] = __float2half(acc / float(COMPRESS_RATIO));
  }

  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (sub < 3 && VARIANT < 3) {
    const int p = sub;
    if (LOAD_ROPE_POSITIONS != 0) {
      const bool first_from_raw = first_position >= chunk_start_position;
      const int raw_first_row = (int)(query_row_start + first_position - chunk_start_position);
      const bool raw_first_ok = valid_row && first_from_raw
          && (raw_first_row >= query_row_start) && (raw_first_row < query_row_end)
          && (raw_first_row < num_rows);
      const bool state_first_ok = valid_row && !first_from_raw && valid_compressor_state_block;
      int64_t v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
#if VARIANT >= 2
        const int tok = (int)(first_position & state_mask);
#else
        const int tok = (int)(first_position % COMPRESSOR_STATE_SIZE);
#endif
        v = rope_cache[safe_compressor_block * stride_rope_block + tok * stride_rope_token
                       + p * stride_rope_dim];
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    } else {
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
}

__global__ void empty_kernel(int* p) { if (p) *p = 1; }

// ===========================================================================

int main(int argc, char** argv) {
  int which = (argc > 1) ? atoi(argv[1]) : 0;
  const int PAGE_SIZE = 256, WIDTH = 128, W = WIDTH / 2;  // 8 halves per 16 B
  const int NB = 64;
  const int BLOCK_TOPK = 512, CR = 4;
  const int OUT_W = BLOCK_TOPK * CR + CR - 1;
  const int NR = 8;

  // ---- store inputs
  std::vector<int64_t> h_slots(2048);
  srand(1);
  for (int i = 0; i < 2048; i++) h_slots[i] = (int64_t)(rand() % (NB * PAGE_SIZE));
  int64_t* slots; CK(hipMalloc(&slots, 2048 * 8)); CK(hipMemcpy(slots, h_slots.data(), 2048 * 8, hipMemcpyHostToDevice));
  half* rows_buf; CK(hipMalloc(&rows_buf, (size_t)2048 * WIDTH * 2));
  CK(hipMemset(rows_buf, 0, (size_t)2048 * WIDTH * 2));
  half* cache_buf; CK(hipMalloc(&cache_buf, (size_t)NB * PAGE_SIZE * WIDTH * 2));
  CK(hipMemset(cache_buf, 0, (size_t)NB * PAGE_SIZE * WIDTH * 2));

  // ---- compress inputs
  const int STATE = 8;
  half* raw_keys; CK(hipMalloc(&raw_keys, (size_t)2048 * WIDTH * 2)); CK(hipMemset(raw_keys, 0, (size_t)2048*WIDTH*2));
  int64_t* raw_pos; CK(hipMalloc(&raw_pos, (size_t)2048 * 3 * 8)); CK(hipMemset(raw_pos, 0, (size_t)2048*3*8));
  half* state_cache; CK(hipMalloc(&state_cache, (size_t)NB * STATE * WIDTH * 2)); CK(hipMemset(state_cache, 0, (size_t)NB*STATE*WIDTH*2));
  int64_t* rope; CK(hipMalloc(&rope, (size_t)NB * STATE * 3 * 8)); CK(hipMemset(rope, 0, (size_t)NB*STATE*3*8));
  int* stable; CK(hipMalloc(&stable, (size_t)NR * 4)); CK(hipMemset(stable, 0, NR*4));
  int* t2r; CK(hipMalloc(&t2r, (size_t)2048 * 4)); CK(hipMemset(t2r, 0, 2048*4));
  int* qsl; CK(hipMalloc(&qsl, (size_t)(NR + 1) * 4)); CK(hipMemset(qsl, 0, (NR+1)*4));
  int64_t* lpos; CK(hipMalloc(&lpos, (size_t)2048 * 8)); CK(hipMemset(lpos, 0, 2048*8));
  int64_t* cslot; CK(hipMalloc(&cslot, (size_t)2048 * 8)); CK(hipMemset(cslot, 0, 2048*8));
  half* pooled; CK(hipMalloc(&pooled, (size_t)2048 * WIDTH * 2)); CK(hipMemset(pooled, 0, (size_t)2048*WIDTH*2));
  int64_t* fpos; CK(hipMalloc(&fpos, (size_t)2048 * 3 * 8)); CK(hipMemset(fpos, 0, (size_t)2048*3*8));

  // ---- expand inputs
  int32_t* blocks; CK(hipMalloc(&blocks, (size_t)2048 * BLOCK_TOPK * 4));
  CK(hipMemset(blocks, 0, (size_t)2048 * BLOCK_TOPK * 4));
  int64_t* qpos; CK(hipMalloc(&qpos, (size_t)2048 * 8));
  { std::vector<int64_t> h(2048); for (int i = 0; i < 2048; i++) h[i] = (int64_t)i * CR + 1;
    CK(hipMemcpy(qpos, h.data(), 2048 * 8, hipMemcpyHostToDevice)); }
  int32_t* seqlen; CK(hipMalloc(&seqlen, (size_t)NR * 4));
  { std::vector<int32_t> h(NR); for (int i = 0; i < NR; i++) h[i] = (i + 1) * 7000;
    CK(hipMemcpy(seqlen, h.data(), NR * 4, hipMemcpyHostToDevice)); }
  int32_t* out; CK(hipMalloc(&out, (size_t)2048 * OUT_W * 4));

  warm(2000);

  if (which == 0 || which == 1) {
    printf("store_cache_rows w=128 page=%d rows_per_block=%d\n", PAGE_SIZE, 256 / (WIDTH / 8));
    const int rows_per_block = 256 / (WIDTH / 8);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int grid = (rows + rows_per_block - 1) / rows_per_block;
      const int iters = iters_for(rows);
      float t;
      if (which == 0) {
        t = timeit([&] { store_base<<<grid, 256>>>((const char*)rows_buf, slots, (char*)cache_buf,
              rows, NB, WIDTH * 2, (int64_t)PAGE_SIZE * WIDTH * 2, (int64_t)WIDTH * 2,
              PAGE_SIZE, rows_per_block, WIDTH / 8); }, iters);
      } else {
        t = timeit([&] { store_tmpl<16><<<grid, 256>>>((const char*)rows_buf, slots, (char*)cache_buf,
              rows, NB, WIDTH * 2, (int64_t)PAGE_SIZE * WIDTH * 2, (int64_t)WIDTH * 2,
              PAGE_SIZE, rows_per_block); }, iters);
      }
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }

  if (which == 0 || which == 2) {
    printf("compress_groups d=128 cr=%d state=%d\n", CR, STATE);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      float t = timeit([&] { compress_kernel<128, CR, 0><<<rows, 128>>>(
            raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
            pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
            WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 3) {
    printf("compress_groups v2 (pow2 state mask) d=128 cr=%d state=%d\n", CR, STATE);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      float t = timeit([&] { compress_kernel<128, CR, 2><<<rows, 128>>>(
            raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
            pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
            WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 4) {
    printf("compress_groups v3 (no rope tail) d=128 cr=%d state=%d\n", CR, STATE);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      float t = timeit([&] { compress_kernel<128, CR, 3><<<rows, 128>>>(
            raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
            pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
            WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }

  if (which == 0 || which == 5) {
    printf("expand_indices block_topk=%d out_w=%d (generic runtime cr)\n", BLOCK_TOPK, OUT_W);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      const dim3 grid(rows, (OUT_W + 255) / 256);
      float t = timeit([&] { expand_generic<int64_t, int32_t><<<grid, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, CR, OUT_W); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 8) {
    printf("expand_indices vec (1 rank/thread) block_topk=%d out_w=%d cr=%d\n", BLOCK_TOPK, OUT_W, CR);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      const dim3 grid(rows, (OUT_W / CR + 255) / 256);
      float t = timeit([&] { expand_vec<int64_t, int32_t, CR><<<grid, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 9) {
    printf("expand_indices multi (stride-256 cols/thread) block_topk=%d out_w=%d cr=%d\n", BLOCK_TOPK, OUT_W, CR);
    const int cpt = (argc > 2) ? atoi(argv[2]) : 4;
    printf("cpt=%d\n", cpt);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      float t;
      if (cpt == 1) { const dim3 g(rows, (OUT_W + 255) / 256);
        t = timeit([&] { expand_multi<int64_t, int32_t, CR, 1><<<g, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters); }
      else if (cpt == 2) { const dim3 g(rows, (OUT_W + 511) / 512);
        t = timeit([&] { expand_multi<int64_t, int32_t, CR, 2><<<g, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters); }
      else if (cpt == 4) { const dim3 g(rows, (OUT_W + 1023) / 1024);
        t = timeit([&] { expand_multi<int64_t, int32_t, CR, 4><<<g, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters); }
      else { const dim3 g(rows, (OUT_W + 2047) / 2048);
        t = timeit([&] { expand_multi<int64_t, int32_t, CR, 8><<<g, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters); }
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 10) {
    printf("compress_groups LDS preamble d=128 cr=%d state=%d\n", CR, STATE);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      float t = timeit([&] { compress_lds_kernel<128, CR, 0><<<rows, 128>>>(
            raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
            pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
            WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 11) {
    const int rpb = (argc > 2) ? atoi(argv[2]) : 2;
    printf("compress_groups rows/block=%d d=128 cr=%d state=%d\n", rpb, CR, STATE);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      const int grid = (rows + rpb - 1) / rpb;
      float t;
      if (rpb == 2) {
        t = timeit([&] { compress_rpb_kernel<128, CR, 0, 2><<<grid, 256>>>(
              raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
              pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
              WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      } else {
        t = timeit([&] { compress_rpb_kernel<128, CR, 0, 4><<<grid, 512>>>(
              raw_keys, raw_pos, state_cache, rope, stable, t2r, qsl, lpos, cslot,
              pooled, fpos, WIDTH, 1, 3, 1, STATE, 1, WIDTH, STATE * WIDTH, STATE, 1, 0,
              WIDTH, 1, 3, 1, rows, NB, NR, STATE, WIDTH, 1, 0); }, iters);
      }
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  if (which == 7) {
    printf("empty kernel reference (launch-bound floor)\n");
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      const dim3 g2(rows, (OUT_W + 255) / 256);
      float t1 = timeit([&] { empty_kernel<<<1, 256>>>(out); }, iters);
      float t9 = timeit([&] { empty_kernel<<<g2, 256>>>(out); }, iters);
      float tb = timeit([&] { empty_kernel<<<(rows + 15) / 16, 256>>>(out); }, iters);
      printf("  rows %5d  store_grid %.3f  expand_grid %.3f  grid1 %.3f\n", rows, tb, t9, t1);
    }
  }
  if (which == 6) {
    printf("expand_indices block_topk=%d out_w=%d (constexpr cr=%d)\n", BLOCK_TOPK, OUT_W, CR);
    for (int ri = 0; ri < NROWS; ri++) {
      const int rows = ROWS[ri];
      const int iters = iters_for(rows);
      const dim3 grid(rows, (OUT_W + 255) / 256);
      float t = timeit([&] { expand_kernel<int64_t, int32_t, CR, 1><<<grid, 256>>>(
            blocks, qpos, seqlen, t2r, out, BLOCK_TOPK, 1, OUT_W, 1, rows, NR, BLOCK_TOPK, OUT_W); }, iters);
      printf("  rows %5d  us %.3f\n", rows, t);
    }
  }
  return 0;
}
