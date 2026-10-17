// HIP replacements for the v1 worker Mamba Triton kernels on gfx1030 (RDNA2):
//
//   .../gpu/model_states/mamba_hybrid.py::preprocess_mamba_align_fused_kernel
//   .../gpu/model_states/mamba_hybrid.py::_scatter_num_accepted_kernel
//   .../gpu/model_states/mamba_hybrid.py::_fill_num_accepted_kernel
//   vllm/v1/worker/mamba_utils.py::precopy_mamba_align_fused_kernel
//   vllm/v1/worker/mamba_utils.py::postprocess_mamba_fused_kernel
//   vllm/v1/worker/mamba_utils.py::batch_memcpy_kernel
//
// All of these run once per decode step inside a FULL cudagraph capture, so the
// host wrappers only launch: no host syncs, no `.item()`, no allocations. Every
// write goes into a caller-provided buffer.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace vllm {
namespace rdna2_mamba {

constexpr int kThreads = 256;
// batch_memcpy_kernel mirrors the Triton BLOCK_SIZE=1024 lane tile one-to-one.
constexpr int kCopyBlock = 1024;

__device__ __forceinline__ int64_t cdiv_i64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ int64_t imin_i64(int64_t a, int64_t b) {
  return a < b ? a : b;
}

// Triton's `//` is floor division; C's `/` truncates.
__device__ __forceinline__ int32_t floor_div_i32(int32_t a, int32_t b) {
  int32_t q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0))) {
    --q;
  }
  return q;
}

// ---------------------------------------------------------------------------
// _memcpy_u64_tiled
//   Head/body/tail memcpy with the u64 body split across `num_tiles` CTAs.
//   Matches vllm/v1/worker/mamba_utils.py::_memcpy_u64_tiled.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void memcpy_u64_tiled(uint64_t src_addr,
                                                 uint64_t dst_addr,
                                                 int64_t copy_size,
                                                 int tile_idx, int num_tiles) {
  if (copy_size <= 0) {
    return;
  }
  if (((src_addr ^ dst_addr) & 7ull) == 0ull) {
    // Fast path: shared sub-8B alignment. Tile 0 owns the byte head that lifts
    // dst to 8B and the trailing 0-7 byte tail; the body is u64 vectorized.
    const int64_t head_raw = (int64_t)((8ull - (dst_addr & 7ull)) & 7ull);
    const int64_t head_bytes = imin_i64(head_raw, copy_size);
    if (tile_idx == 0 && (int64_t)threadIdx.x < head_bytes) {
      ((uint8_t*)(uintptr_t)dst_addr)[threadIdx.x] =
          ((const uint8_t*)(uintptr_t)src_addr)[threadIdx.x];
    }

    const int64_t body_u64 = (copy_size - head_bytes) / 8;
    const int64_t per_tile =
        cdiv_i64(cdiv_i64(body_u64, num_tiles), kCopyBlock) * kCopyBlock;
    const int64_t tile_start = (int64_t)tile_idx * per_tile;
    const int64_t tile_end = imin_i64(tile_start + per_tile, body_u64);

    const uint64_t* src_u64 =
        (const uint64_t*)(uintptr_t)(src_addr + (uint64_t)head_bytes);
    uint64_t* dst_u64 = (uint64_t*)(uintptr_t)(dst_addr + (uint64_t)head_bytes);
    for (int64_t o = tile_start + threadIdx.x; o < tile_end;
         o += (int64_t)blockDim.x) {
      dst_u64[o] = src_u64[o];
    }

    if (tile_idx == 0) {
      const int64_t tail_start = head_bytes + body_u64 * 8;
      const int64_t tail_bytes = copy_size - tail_start;
      if ((int64_t)threadIdx.x < tail_bytes) {
        ((uint8_t*)(uintptr_t)(dst_addr + (uint64_t)tail_start))[threadIdx.x] =
            ((const uint8_t*)(uintptr_t)(src_addr +
                                         (uint64_t)tail_start))[threadIdx.x];
      }
    }
  } else {
    // Slow path: mismatched sub-8B alignment, byte-wide tiled copy.
    const int64_t per_tile =
        cdiv_i64(cdiv_i64(copy_size, num_tiles), kCopyBlock) * kCopyBlock;
    const int64_t tile_start = (int64_t)tile_idx * per_tile;
    const int64_t tile_end = imin_i64(tile_start + per_tile, copy_size);
    const uint8_t* src_u8 = (const uint8_t*)(uintptr_t)src_addr;
    uint8_t* dst_u8 = (uint8_t*)(uintptr_t)dst_addr;
    for (int64_t o = tile_start + threadIdx.x; o < tile_end;
         o += (int64_t)blockDim.x) {
      dst_u8[o] = src_u8[o];
    }
  }
}

// ---------------------------------------------------------------------------
// Flattened per-(layer, state-type) metadata, one entry per state tensor.
// Mirrors the flattened Triton metadata arrays byte-for-byte.
// ---------------------------------------------------------------------------
struct MambaCopyMeta {
  const int64_t* base_addrs;        // int64: base address of each state tensor
  const int64_t* block_strides;     // int64: bytes per block
  const int32_t* elem_sizes;        // int32: element size in bytes
  const int64_t* inner_sizes;       // int64: elements in inner dimensions
  const int32_t* conv_widths;       // int32: conv width (0 = temporal state)
  const int32_t* group_indices;     // int32: state_idx -> block-table group
  const int32_t* dim_row_count;     // int32: per-block dim row count
  const int64_t* dim_row_stride;    // int64: bytes between dim rows
  const int64_t* block_table_ptrs;  // int64: per-group block-table base addrs
  int64_t block_table_stride_req;
};

// ---------------------------------------------------------------------------
// _copy_mamba_state_block
//   conv state (conv_width > 0): shift the window by `token_bias` tokens,
//     state[bt[src_col], token_bias:] -> state[bt[dst_col], :conv_width - bias]
//   temporal state: state[bt[src_col + token_bias]] -> state[bt[dst_col]]
// ---------------------------------------------------------------------------
template <bool kConvDimFirst>
__device__ __forceinline__ void copy_mamba_state_block(
    const MambaCopyMeta& m, int state_idx, int64_t bt_row_idx, int32_t src_col,
    int32_t dst_col, int32_t token_bias, int tile_idx, int num_temporal_tiles) {
  const uint64_t base = (uint64_t)(int64_t)m.base_addrs[state_idx];
  const int64_t block_stride = m.block_strides[state_idx];
  const int32_t elem_size = m.elem_sizes[state_idx];
  const int64_t inner_size = m.inner_sizes[state_idx];
  const int32_t conv_width = m.conv_widths[state_idx];
  const int64_t group_idx = (int64_t)m.group_indices[state_idx];
  const int32_t* block_table = reinterpret_cast<const int32_t*>(
      (uintptr_t)m.block_table_ptrs[group_idx]);
  block_table += bt_row_idx * m.block_table_stride_req;

  const int64_t dest_block_id = (int64_t)block_table[dst_col];
  const uint64_t dst_addr = base + (uint64_t)(dest_block_id * block_stride);

  if (conv_width > 0 && kConvDimFirst) {
    // DS conv layout: state_len is the slide axis, so copy per dim row. Only
    // tile 0 copies conv state; higher tiles contribute nothing.
    if (tile_idx > 0) {
      return;
    }
    const int64_t src_block_id = (int64_t)block_table[src_col];
    const int32_t dim_rows = m.dim_row_count[state_idx];
    const int64_t row_stride = m.dim_row_stride[state_idx];
    const uint64_t src_block = base + (uint64_t)(src_block_id * block_stride);
    const int32_t num_dst_tokens = conv_width - token_bias;

    // Stable row-to-lane ownership keeps left shifts memmove-safe: a row's
    // tokens are copied in increasing order by the lane that owns the row.
    for (int32_t t = 0; t < num_dst_tokens; ++t) {
      for (int32_t r = threadIdx.x; r < dim_rows; r += blockDim.x) {
        const uint64_t s =
            src_block + (uint64_t)((int64_t)r * row_stride +
                                   (int64_t)(t + token_bias) * elem_size);
        const uint64_t d = dst_addr + (uint64_t)((int64_t)r * row_stride +
                                                 (int64_t)t * elem_size);
        if (elem_size == 2) {
          *(uint16_t*)(uintptr_t)d = *(const uint16_t*)(uintptr_t)s;
        } else if (elem_size == 4) {
          *(uint32_t*)(uintptr_t)d = *(const uint32_t*)(uintptr_t)s;
        } else {
          for (int32_t b = 0; b < elem_size; ++b) {
            ((uint8_t*)(uintptr_t)d)[b] = ((const uint8_t*)(uintptr_t)s)[b];
          }
        }
      }
    }
    return;
  }

  if (conv_width > 0) {
    // SD conv layout: dim is contiguous and conv_width is the slide axis.
    if (tile_idx > 0) {
      return;
    }
    const int64_t src_block_id = (int64_t)block_table[src_col];
    const uint64_t src_block = base + (uint64_t)(src_block_id * block_stride);
    const int64_t token_bytes = inner_size * (int64_t)elem_size;
    const int32_t num_dst_tokens = conv_width - token_bias;

    // Distinct blocks and exact self-copies have no destructive overlap, so one
    // vectorized memcpy suffices.
    if (src_block_id != dest_block_id || token_bias == 0) {
      memcpy_u64_tiled(
          src_block + (uint64_t)((int64_t)token_bias * token_bytes), dst_addr,
          (int64_t)num_dst_tokens * token_bytes, tile_idx, 1);
      return;
    }

    // Same-block left shift: per-token copies from low to high.
    // shortcut: Triton issues these copies without a barrier and can race
    // (a leading warp overwrites a token a lagging warp has yet to read); we
    // sync between tokens so the memmove is deterministic. Upgrade if the
    // Triton fallback is ever fixed to match.
    for (int32_t t = 0; t < num_dst_tokens; ++t) {
      memcpy_u64_tiled(
          src_block + (uint64_t)((int64_t)(t + token_bias) * token_bytes),
          dst_addr + (uint64_t)((int64_t)t * token_bytes), token_bytes,
          tile_idx, 1);
      __syncthreads();
    }
    return;
  }

  // Temporal state: the u64 body is partitioned across num_temporal_tiles CTAs
  // to keep the SMs filled at small batch.
  const int64_t actual_src_block_id =
      (int64_t)block_table[src_col + token_bias];
  const uint64_t src_addr =
      base + (uint64_t)(actual_src_block_id * block_stride);
  // Natural data size, not the page stride: the state tensor may be as_strided
  // with padded page strides.
  memcpy_u64_tiled(src_addr, dst_addr, inner_size * (int64_t)elem_size,
                   tile_idx, num_temporal_tiles);
}

// ---------------------------------------------------------------------------
// preprocess_mamba_align_fused_kernel   grid = (cdiv(num_reqs, kThreads),)
// Emits the pre-copy src column/offset AND advances state_idx (with the
// accepted-token reset) for the align preprocess.
// ---------------------------------------------------------------------------
__global__ void preprocess_mamba_align_fused_kernel(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ state_idx_ptr,
    const int32_t* __restrict__ num_computed_tokens_ptr,
    const int32_t* __restrict__ query_start_loc_ptr,
    int32_t* __restrict__ num_accepted_tokens_ptr,
    int32_t* __restrict__ src_col_ptr, int32_t* __restrict__ src_off_ptr,
    int32_t num_reqs, int32_t mamba_block_size) {
  for (int32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < num_reqs;
       row += gridDim.x * blockDim.x) {
    const int32_t req = idx_mapping[row];
    const int32_t state_idx = state_idx_ptr[req];
    const int32_t num_accepted = num_accepted_tokens_ptr[req];

    const int32_t src_off = num_accepted - 1 > 0 ? num_accepted - 1 : 0;
    src_col_ptr[req] = state_idx;
    src_off_ptr[req] = src_off;

    const int32_t num_computed = num_computed_tokens_ptr[req];
    const int32_t query_start = query_start_loc_ptr[row];
    const int32_t query_end = query_start_loc_ptr[row + 1];
    const int32_t computed_after = num_computed + query_end - query_start;
    const int32_t new_state_idx =
        (computed_after + mamba_block_size - 1) / mamba_block_size - 1;
    state_idx_ptr[req] = new_state_idx;

    if (state_idx >= 0 && state_idx != new_state_idx) {
      num_accepted_tokens_ptr[req] = 1;
    }
  }
}

// ---------------------------------------------------------------------------
// _scatter_num_accepted_kernel / _fill_num_accepted_kernel
// ---------------------------------------------------------------------------
__global__ void scatter_num_accepted_kernel(
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ num_sampled, int32_t* __restrict__ num_accepted,
    int32_t num_reqs) {
  const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_reqs) {
    return;
  }
  const int32_t req_state_idx = idx_mapping[row];
  if (req_state_idx < 0) {
    return;
  }
  const int32_t sampled = num_sampled[row];
  num_accepted[req_state_idx] = sampled > 1 ? sampled : 1;
}

__global__ void fill_num_accepted_kernel(
    const int32_t* __restrict__ idx_mapping, int32_t* __restrict__ num_accepted,
    int32_t value, int32_t num_reqs) {
  const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_reqs) {
    return;
  }
  const int32_t req_state_idx = idx_mapping[row];
  if (req_state_idx < 0) {
    return;
  }
  num_accepted[req_state_idx] = value;
}

// ---------------------------------------------------------------------------
// precopy_mamba_align_fused_kernel
//   grid = (num_reqs, num_states, num_temporal_tiles)
// ---------------------------------------------------------------------------
template <bool kHasIdxMapping, bool kConvDimFirst>
__global__ void precopy_mamba_align_fused_kernel(
    const int32_t* __restrict__ mamba_state_idx,
    const int32_t* __restrict__ src_col_ptr,
    const int32_t* __restrict__ token_bias_ptr,
    const int32_t* __restrict__ idx_mapping, int32_t num_reqs,
    int32_t num_temporal_tiles, MambaCopyMeta m) {
  const int32_t batch_idx = blockIdx.x;
  const int32_t state_idx = blockIdx.y;
  const int32_t tile_idx = blockIdx.z;
  if (batch_idx >= num_reqs) {
    return;
  }
  int32_t req_idx;
  if (kHasIdxMapping) {
    req_idx = idx_mapping[batch_idx];
    if (req_idx < 0) {
      return;
    }
  } else {
    req_idx = batch_idx;
  }

  const int32_t src_col = src_col_ptr[req_idx];
  const int32_t dst_col = mamba_state_idx[req_idx];
  // Fresh state, or still writing the same block: the kernels locate the
  // initial state in-block via num_accepted, so there is nothing to copy.
  if (src_col < 0 || src_col == dst_col) {
    return;
  }
  const int32_t token_bias = token_bias_ptr[req_idx];
  copy_mamba_state_block<kConvDimFirst>(m, state_idx, req_idx, src_col, dst_col,
                                        token_bias, tile_idx,
                                        num_temporal_tiles);
}

// ---------------------------------------------------------------------------
// postprocess_mamba_fused_kernel
//   grid = (num_reqs, num_states, num_temporal_tiles)
// ---------------------------------------------------------------------------
template <bool kHasIdxMapping, bool kPrecomputed, bool kConvDimFirst>
__global__ void postprocess_mamba_fused_kernel(
    const int32_t* __restrict__ num_accepted_tokens,
    const int32_t* __restrict__ mamba_state_idx,
    const int32_t* __restrict__ num_scheduled_tokens,
    const int32_t* __restrict__ num_computed_tokens,
    const int32_t* __restrict__ num_draft_tokens,
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ num_accepted_tokens_out, int32_t num_reqs,
    int32_t block_size, int32_t num_temporal_tiles, MambaCopyMeta m) {
  const int32_t batch_idx = blockIdx.x;
  const int32_t state_idx = blockIdx.y;
  const int32_t tile_idx = blockIdx.z;
  if (batch_idx >= num_reqs) {
    return;
  }
  int32_t req_idx;
  if (kHasIdxMapping) {
    req_idx = idx_mapping[batch_idx];
    if (req_idx < 0) {
      return;
    }
  } else {
    req_idx = batch_idx;
  }

  const int32_t num_accepted = num_accepted_tokens[req_idx];
  const int32_t src_block_idx = mamba_state_idx[req_idx];

  int32_t new_num_computed;
  int32_t num_tokens_running_state;
  if (kPrecomputed) {
    new_num_computed = num_computed_tokens[req_idx];
    num_tokens_running_state = new_num_computed - num_accepted + 1;
  } else {
    const int32_t num_scheduled = num_scheduled_tokens[req_idx];
    const int32_t num_computed = num_computed_tokens[req_idx];
    const int32_t num_draft = num_draft_tokens[req_idx];
    num_tokens_running_state = num_computed + num_scheduled - num_draft;
    new_num_computed = num_tokens_running_state + num_accepted - 1;
  }

  const int32_t aligned_new_computed =
      floor_div_i32(new_num_computed, block_size) * block_size;
  if (aligned_new_computed < num_tokens_running_state) {
    return;
  }

  const int32_t accept_token_bias =
      aligned_new_computed - num_tokens_running_state;
  const int32_t dest_block_idx =
      floor_div_i32(aligned_new_computed, block_size) - 1;

  // Per-request update, so only state_idx == 0 writes; guard on tile_idx == 0
  // so higher temporal tiles do not duplicate the store.
  if (src_block_idx == dest_block_idx && state_idx == 0 && tile_idx == 0 &&
      threadIdx.x == 0) {
    num_accepted_tokens_out[req_idx] = 1;
  }

  if (src_block_idx == dest_block_idx && accept_token_bias == 0) {
    return;
  }

  copy_mamba_state_block<kConvDimFirst>(m, state_idx, req_idx, src_block_idx,
                                        dest_block_idx, accept_token_bias,
                                        tile_idx, num_temporal_tiles);
}

// ---------------------------------------------------------------------------
// batch_memcpy_kernel   grid = (batch,)
// ---------------------------------------------------------------------------
__global__ void batch_memcpy_kernel(const uint64_t* __restrict__ src_ptrs,
                                    const uint64_t* __restrict__ dst_ptrs,
                                    const int32_t* __restrict__ sizes) {
  const int32_t pid = blockIdx.x;
  const uint64_t src_ptr = src_ptrs[pid];
  const uint64_t dst_ptr = dst_ptrs[pid];
  const int32_t size = sizes[pid];
  // Preserve each lane's source before a lower-address lane stores over it.
  // The condition is uniform within the program.
  const bool is_left_overlap =
      dst_ptr < src_ptr && dst_ptr + (uint64_t)(int64_t)size > src_ptr;
  const uint8_t* src_u8 = (const uint8_t*)(uintptr_t)src_ptr;
  uint8_t* dst_u8 = (uint8_t*)(uintptr_t)dst_ptr;

  for (int32_t i = 0; i < size; i += kCopyBlock) {
    const int32_t off = i + (int32_t)threadIdx.x;
    const bool active = off < size;
    uint8_t data = 0;
    if (active) {
      data = src_u8[off];
    }
    if (is_left_overlap) {
      __syncthreads();
    }
    if (active) {
      dst_u8[off] = data;
    }
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
namespace {

MambaCopyMeta build_meta(const torch::Tensor& block_table_ptrs,
                         int64_t bt_stride,
                         const torch::Tensor& state_base_addrs,
                         const torch::Tensor& state_block_strides,
                         const torch::Tensor& state_elem_sizes,
                         const torch::Tensor& state_inner_sizes,
                         const torch::Tensor& state_conv_widths,
                         const torch::Tensor& state_group_indices,
                         const torch::Tensor& state_dim_row_count,
                         const torch::Tensor& state_dim_row_stride) {
  MambaCopyMeta m;
  m.base_addrs = reinterpret_cast<const int64_t*>(state_base_addrs.data_ptr());
  m.block_strides =
      reinterpret_cast<const int64_t*>(state_block_strides.data_ptr());
  m.elem_sizes = reinterpret_cast<const int32_t*>(state_elem_sizes.data_ptr());
  m.inner_sizes =
      reinterpret_cast<const int64_t*>(state_inner_sizes.data_ptr());
  m.conv_widths =
      reinterpret_cast<const int32_t*>(state_conv_widths.data_ptr());
  m.group_indices =
      reinterpret_cast<const int32_t*>(state_group_indices.data_ptr());
  m.dim_row_count =
      reinterpret_cast<const int32_t*>(state_dim_row_count.data_ptr());
  m.dim_row_stride =
      reinterpret_cast<const int64_t*>(state_dim_row_stride.data_ptr());
  m.block_table_ptrs =
      reinterpret_cast<const int64_t*>(block_table_ptrs.data_ptr());
  m.block_table_stride_req = bt_stride;
  return m;
}

void check_meta_dtypes(const torch::Tensor& state_base_addrs,
                       const torch::Tensor& state_block_strides,
                       const torch::Tensor& state_elem_sizes,
                       const torch::Tensor& state_inner_sizes,
                       const torch::Tensor& state_conv_widths,
                       const torch::Tensor& state_group_indices,
                       const torch::Tensor& state_dim_row_count,
                       const torch::Tensor& state_dim_row_stride) {
  TORCH_CHECK(state_base_addrs.scalar_type() == at::kLong &&
                  state_block_strides.scalar_type() == at::kLong &&
                  state_inner_sizes.scalar_type() == at::kLong &&
                  state_dim_row_stride.scalar_type() == at::kLong,
              "mamba rdna2: int64 metadata expected");
  TORCH_CHECK(state_elem_sizes.scalar_type() == at::kInt &&
                  state_conv_widths.scalar_type() == at::kInt &&
                  state_group_indices.scalar_type() == at::kInt &&
                  state_dim_row_count.scalar_type() == at::kInt,
              "mamba rdna2: int32 metadata expected");
}

}  // namespace

}  // namespace rdna2_mamba
}  // namespace vllm

using namespace vllm::rdna2_mamba;

void preprocess_mamba_align_fused_rdna2(
    torch::Tensor idx_mapping,  // [num_reqs] int32
    torch::Tensor state_idx,    // [max_num_reqs] int32 (in/out)
    torch::Tensor num_computed_tokens, torch::Tensor query_start_loc,
    torch::Tensor num_accepted_tokens,  // [max_num_reqs] int32 (in/out)
    torch::Tensor src_col,              // [max_num_reqs] int32 (out)
    torch::Tensor src_off,              // [max_num_reqs] int32 (out)
    int64_t num_reqs, int64_t mamba_block_size) {
  TORCH_CHECK(idx_mapping.scalar_type() == at::kInt &&
                  state_idx.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt &&
                  num_accepted_tokens.scalar_type() == at::kInt &&
                  src_col.scalar_type() == at::kInt &&
                  src_off.scalar_type() == at::kInt,
              "preprocess_mamba_align_fused_rdna2: all tensors must be int32");
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = (int)((num_reqs + kThreads - 1) / kThreads);
  preprocess_mamba_align_fused_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<int32_t*>(state_idx.data_ptr()),
      reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<int32_t*>(num_accepted_tokens.data_ptr()),
      reinterpret_cast<int32_t*>(src_col.data_ptr()),
      reinterpret_cast<int32_t*>(src_off.data_ptr()), (int32_t)num_reqs,
      (int32_t)mamba_block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_num_accepted_rdna2(torch::Tensor idx_mapping,
                                torch::Tensor num_sampled,
                                torch::Tensor num_accepted, int64_t num_reqs) {
  TORCH_CHECK(idx_mapping.scalar_type() == at::kInt &&
                  num_sampled.scalar_type() == at::kInt &&
                  num_accepted.scalar_type() == at::kInt,
              "scatter_num_accepted_rdna2: all tensors must be int32");
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = (int)((num_reqs + kThreads - 1) / kThreads);
  scatter_num_accepted_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(num_sampled.data_ptr()),
      reinterpret_cast<int32_t*>(num_accepted.data_ptr()), (int32_t)num_reqs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_num_accepted_rdna2(torch::Tensor idx_mapping,
                             torch::Tensor num_accepted, int64_t value,
                             int64_t num_reqs) {
  TORCH_CHECK(idx_mapping.scalar_type() == at::kInt &&
                  num_accepted.scalar_type() == at::kInt,
              "fill_num_accepted_rdna2: all tensors must be int32");
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = (int)((num_reqs + kThreads - 1) / kThreads);
  fill_num_accepted_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<int32_t*>(num_accepted.data_ptr()), (int32_t)value,
      (int32_t)num_reqs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void precopy_mamba_align_fused_rdna2(
    torch::Tensor mamba_state_idx, torch::Tensor src_col,
    torch::Tensor token_bias, torch::Tensor block_table_ptrs,
    int64_t block_table_stride_req, torch::Tensor state_base_addrs,
    torch::Tensor state_block_strides, torch::Tensor state_elem_sizes,
    torch::Tensor state_inner_sizes, torch::Tensor state_conv_widths,
    torch::Tensor state_group_indices, torch::Tensor state_dim_row_count,
    torch::Tensor state_dim_row_stride,
    const std::optional<torch::Tensor>& idx_mapping, int64_t num_reqs,
    bool conv_state_dim_first, int64_t temporal_tiles) {
  TORCH_CHECK(mamba_state_idx.scalar_type() == at::kInt &&
                  src_col.scalar_type() == at::kInt &&
                  token_bias.scalar_type() == at::kInt,
              "precopy_mamba_align_fused_rdna2: per-request inputs int32");
  TORCH_CHECK(block_table_ptrs.scalar_type() == at::kLong,
              "precopy_mamba_align_fused_rdna2: block_table_ptrs int64");
  check_meta_dtypes(state_base_addrs, state_block_strides, state_elem_sizes,
                    state_inner_sizes, state_conv_widths, state_group_indices,
                    state_dim_row_count, state_dim_row_stride);
  if (num_reqs == 0) {
    return;
  }
  const int64_t num_states = state_base_addrs.numel();
  if (num_states == 0) {
    return;
  }
  const MambaCopyMeta meta =
      build_meta(block_table_ptrs, block_table_stride_req, state_base_addrs,
                 state_block_strides, state_elem_sizes, state_inner_sizes,
                 state_conv_widths, state_group_indices, state_dim_row_count,
                 state_dim_row_stride);

  const bool has_idx_mapping = idx_mapping.has_value();
  const int32_t* idx_mapping_ptr =
      has_idx_mapping
          ? reinterpret_cast<const int32_t*>(idx_mapping->data_ptr())
          : nullptr;

  const at::cuda::OptionalCUDAGuard guard(device_of(mamba_state_idx));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_reqs, (unsigned)num_states,
                  (unsigned)std::max<int64_t>(temporal_tiles, 1));

#define VLLM_RDNA2_PRECOPY(IM, CF)                                      \
  precopy_mamba_align_fused_kernel<IM, CF>                              \
      <<<grid, kThreads, 0, stream.stream()>>>(                         \
          reinterpret_cast<const int32_t*>(mamba_state_idx.data_ptr()), \
          reinterpret_cast<const int32_t*>(src_col.data_ptr()),         \
          reinterpret_cast<const int32_t*>(token_bias.data_ptr()),      \
          idx_mapping_ptr, (int32_t)num_reqs, (int32_t)temporal_tiles, meta)

  if (has_idx_mapping) {
    if (conv_state_dim_first) {
      VLLM_RDNA2_PRECOPY(true, true);
    } else {
      VLLM_RDNA2_PRECOPY(true, false);
    }
  } else if (conv_state_dim_first) {
    VLLM_RDNA2_PRECOPY(false, true);
  } else {
    VLLM_RDNA2_PRECOPY(false, false);
  }
#undef VLLM_RDNA2_PRECOPY
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void postprocess_mamba_fused_rdna2(
    torch::Tensor num_accepted_tokens, torch::Tensor mamba_state_idx,
    const std::optional<torch::Tensor>& num_scheduled_tokens,
    torch::Tensor num_computed_tokens,
    const std::optional<torch::Tensor>& num_draft_tokens,
    torch::Tensor block_table_ptrs, int64_t block_table_stride_req,
    torch::Tensor state_base_addrs, torch::Tensor state_block_strides,
    torch::Tensor state_elem_sizes, torch::Tensor state_inner_sizes,
    torch::Tensor state_conv_widths, torch::Tensor state_group_indices,
    torch::Tensor state_dim_row_count, torch::Tensor state_dim_row_stride,
    torch::Tensor num_accepted_tokens_out,
    const std::optional<torch::Tensor>& idx_mapping, int64_t num_reqs,
    int64_t block_size, bool conv_state_dim_first,
    bool precomputed_new_computed, int64_t temporal_tiles) {
  TORCH_CHECK(num_accepted_tokens.scalar_type() == at::kInt &&
                  mamba_state_idx.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt &&
                  num_accepted_tokens_out.scalar_type() == at::kInt,
              "postprocess_mamba_fused_rdna2: per-request inputs int32");
  TORCH_CHECK(block_table_ptrs.scalar_type() == at::kLong,
              "postprocess_mamba_fused_rdna2: block_table_ptrs int64");
  check_meta_dtypes(state_base_addrs, state_block_strides, state_elem_sizes,
                    state_inner_sizes, state_conv_widths, state_group_indices,
                    state_dim_row_count, state_dim_row_stride);
  if (num_reqs == 0) {
    return;
  }
  const int64_t num_states = state_base_addrs.numel();
  if (num_states == 0) {
    return;
  }
  const bool has_idx_mapping = idx_mapping.has_value();
  if (!precomputed_new_computed) {
    TORCH_CHECK(
        num_scheduled_tokens.has_value() && num_draft_tokens.has_value(),
        "postprocess_mamba_fused_rdna2: num_scheduled_tokens and "
        "num_draft_tokens are required without "
        "precomputed_new_computed");
  }
  const MambaCopyMeta meta =
      build_meta(block_table_ptrs, block_table_stride_req, state_base_addrs,
                 state_block_strides, state_elem_sizes, state_inner_sizes,
                 state_conv_widths, state_group_indices, state_dim_row_count,
                 state_dim_row_stride);

  const int32_t* num_scheduled_ptr =
      num_scheduled_tokens.has_value()
          ? reinterpret_cast<const int32_t*>(num_scheduled_tokens->data_ptr())
          : nullptr;
  const int32_t* num_draft_ptr =
      num_draft_tokens.has_value()
          ? reinterpret_cast<const int32_t*>(num_draft_tokens->data_ptr())
          : nullptr;
  const int32_t* idx_mapping_ptr =
      has_idx_mapping
          ? reinterpret_cast<const int32_t*>(idx_mapping->data_ptr())
          : nullptr;

  const at::cuda::OptionalCUDAGuard guard(device_of(num_accepted_tokens));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_reqs, (unsigned)num_states,
                  (unsigned)std::max<int64_t>(temporal_tiles, 1));

#define VLLM_RDNA2_POSTPROCESS(IM, PC, CF)                                  \
  postprocess_mamba_fused_kernel<IM, PC, CF>                                \
      <<<grid, kThreads, 0, stream.stream()>>>(                             \
          reinterpret_cast<const int32_t*>(num_accepted_tokens.data_ptr()), \
          reinterpret_cast<const int32_t*>(mamba_state_idx.data_ptr()),     \
          num_scheduled_ptr,                                                \
          reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()), \
          num_draft_ptr, idx_mapping_ptr,                                   \
          reinterpret_cast<int32_t*>(num_accepted_tokens_out.data_ptr()),   \
          (int32_t)num_reqs, (int32_t)block_size, (int32_t)temporal_tiles,  \
          meta)

  if (has_idx_mapping) {
    if (precomputed_new_computed) {
      if (conv_state_dim_first) {
        VLLM_RDNA2_POSTPROCESS(true, true, true);
      } else {
        VLLM_RDNA2_POSTPROCESS(true, true, false);
      }
    } else if (conv_state_dim_first) {
      VLLM_RDNA2_POSTPROCESS(true, false, true);
    } else {
      VLLM_RDNA2_POSTPROCESS(true, false, false);
    }
  } else if (precomputed_new_computed) {
    if (conv_state_dim_first) {
      VLLM_RDNA2_POSTPROCESS(false, true, true);
    } else {
      VLLM_RDNA2_POSTPROCESS(false, true, false);
    }
  } else if (conv_state_dim_first) {
    VLLM_RDNA2_POSTPROCESS(false, false, true);
  } else {
    VLLM_RDNA2_POSTPROCESS(false, false, false);
  }
#undef VLLM_RDNA2_POSTPROCESS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void batch_memcpy_rdna2(torch::Tensor src_ptrs, torch::Tensor dst_ptrs,
                        torch::Tensor sizes) {
  TORCH_CHECK(src_ptrs.scalar_type() == at::kUInt64 &&
                  dst_ptrs.scalar_type() == at::kUInt64 &&
                  sizes.scalar_type() == at::kInt,
              "batch_memcpy_rdna2: expected uint64 ptrs and int32 sizes");
  const int64_t batch = src_ptrs.numel();
  TORCH_CHECK(dst_ptrs.numel() == batch && sizes.numel() == batch,
              "batch_memcpy_rdna2: batch size mismatch");
  if (batch == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(src_ptrs));
  auto stream = at::cuda::getCurrentCUDAStream();
  batch_memcpy_kernel<<<(unsigned)batch, kCopyBlock, 0, stream.stream()>>>(
      reinterpret_cast<const uint64_t*>(src_ptrs.data_ptr()),
      reinterpret_cast<const uint64_t*>(dst_ptrs.data_ptr()),
      reinterpret_cast<const int32_t*>(sizes.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
