// HIP replacement for the QSA metadata builder Triton kernel on gfx1030
// (RDNA2):
//
//   vllm/models/qwen4_exp/common/qsa_cache.py::_build_qsa_metadata_kernel
//   (launcher build_qsa_metadata_triton)
//
// One CTA covers one TOKEN_BLOCK_SIZE=128 token tile and one
// WORK_BLOCK_SIZE=256 work tile, exactly like the Triton program: the Triton
// launch uses num_warps=4 (128 threads) for both shapes, so the token tile maps
// 1:1 onto the threads and the work tile is walked twice.
//
// The three slot-mapping rules of the Triton kernel are `constexpr`-selected
// there and template-selected here:
//   kSlotCircular   circular_buffer_size > 0  (raw key ring, block column 0)
//   kSlotCompressed compress_ratio != 1       (one row per complete group)
//   kSlotNone       neither: no store at all; the caller keeps
//                   common_attn_metadata.slot_mapping instead.
// The optional pre-indexer work metadata (cumsum over the request axis plus a
// descending binary search per work item) is built when k_work_metadata is not
// null, i.e. only for the compressed cache.
//
// This runs twice per decode step (raw + compressed cache) inside a cudagraph
// capture, so the host wrapper only launches into caller-provided buffers: no
// host syncs, no `.item()`, no allocations. `launch_pdl` is not replicated:
// `is_arch_support_pdl()` is CUDA-only and therefore False on RDNA2.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>
#include <optional>

namespace vllm {
namespace rdna2_qsa_metadata {

constexpr int kTokenBlock = 128;  // Triton TOKEN_BLOCK_SIZE
constexpr int kWorkBlock = 256;   // Triton WORK_BLOCK_SIZE
constexpr int kThreads = 128;     // Triton num_warps=4

enum SlotRule {
  kSlotNone = 0,
  kSlotCircular = 1,
  kSlotCompressed = 2,
};

__device__ __forceinline__ int32_t mini32(int32_t a, int32_t b) {
  return a < b ? a : b;
}

__device__ __forceinline__ int32_t maxi32(int32_t a, int32_t b) {
  return a > b ? a : b;
}

// Triton's integer `//` and `%` truncate toward zero, NOT Python floor
// division: `(-127) // 8` is -15 in a Triton kernel and -16 in Python.
// Verified against this build's Triton on negative operands.
__device__ __forceinline__ int32_t tdiv(int32_t a, int32_t b) {
  return a / b;
}

__device__ __forceinline__ int32_t tmod(int32_t a, int32_t b) {
  return a % b;
}

struct MetadataArgs {
  const int32_t* __restrict__ query_start_loc;
  const int32_t* __restrict__ seq_lens;
  const int64_t* __restrict__ common_slot_mapping;
  const int32_t* __restrict__ block_table;
  int32_t* __restrict__ token_to_req;
  int64_t* __restrict__ logical_positions;
  int32_t* __restrict__ visible_blocks;
  int64_t* __restrict__ slot_mapping;
  int32_t* __restrict__ k_work_metadata;
  int64_t block_table_stride_0;
  int64_t block_table_stride_1;
  int32_t num_reqs;
  int32_t num_mapped_tokens;
  int32_t num_tokens;
  int32_t max_num_work;
  int32_t num_search_steps;
  int32_t work_search_steps;
  int32_t request_scan_size;
  int32_t storage_block_size;
  int32_t compress_ratio;
  int32_t circular_buffer_size;
  int32_t num_block_table_columns;
  int32_t num_block_table_rows;
};

template <int kRule>
__global__ void build_qsa_metadata_kernel(const MetadataArgs args) {
  extern __shared__ int32_t shared_work_ends[];

  const int tid = threadIdx.x;
  const int32_t token_idx = (int32_t)((int64_t)blockIdx.x * kTokenBlock) + tid;
  const bool store_mask = token_idx < args.num_tokens;
  const bool mapped = token_idx < args.num_mapped_tokens;
  const int32_t search_token_idx =
      mini32(token_idx, args.num_mapped_tokens - 1);

  // Last query start at or before this token: descending binary lifting.
  int32_t request_idx = 0;
  for (int step = 0; step < args.num_search_steps; ++step) {
    const int32_t candidate =
        request_idx + (1 << (args.num_search_steps - step - 1));
    if (candidate < args.num_reqs &&
        args.query_start_loc[candidate] <= search_token_idx) {
      request_idx = candidate;
    }
  }

  // request_idx is always < num_reqs, so these are in bounds either way; the
  // Triton masked loads return 0 for unmapped tokens and the value is only used
  // through `mapped` below.
  const int32_t query_start =
      mapped ? args.query_start_loc[request_idx] : 0;
  const int32_t query_end = mapped ? args.query_start_loc[request_idx + 1] : 0;
  const int32_t seq_len = mapped ? args.seq_lens[request_idx] : 0;
  int32_t logical_position =
      seq_len - (query_end - query_start) + token_idx - query_start;
  if (!mapped) {
    logical_position = -1;
  }

  if (store_mask) {
    args.token_to_req[token_idx] = mapped ? request_idx : 0;
    args.logical_positions[token_idx] = (int64_t)logical_position;
    args.visible_blocks[token_idx] =
        maxi32(0, mini32(tdiv(logical_position + 1, args.compress_ratio),
                         tdiv(seq_len, args.compress_ratio)));
  }

  if (kRule == kSlotCircular) {
    const int32_t circular_buffer_size = args.circular_buffer_size;
    bool valid = mapped && (logical_position >= 0) &&
                 (token_idx + circular_buffer_size >= query_end) &&
                 (args.num_block_table_columns > 0) &&
                 (request_idx < args.num_block_table_rows);
    int32_t physical_block = -1;
    if (valid) {
      physical_block = args.block_table[(int64_t)request_idx *
                                        args.block_table_stride_0];
    }
    valid = valid && (physical_block >= 0);
    const int32_t slot = physical_block * circular_buffer_size +
                         tmod(logical_position, circular_buffer_size);
    if (store_mask) {
      args.slot_mapping[token_idx] = valid ? (int64_t)slot : (int64_t)-1;
    }
  } else if (kRule == kSlotCompressed) {
    const int32_t compressed_position =
        tdiv(maxi32(logical_position, 0), args.compress_ratio);
    const int32_t logical_block =
        tdiv(compressed_position, args.storage_block_size);
    bool valid = mapped && (logical_position >= 0) &&
                 (tmod(logical_position + 1, args.compress_ratio) == 0) &&
                 (logical_block < args.num_block_table_columns) &&
                 (request_idx < args.num_block_table_rows);
    int32_t physical_block = -1;
    if (valid) {
      physical_block =
          args.block_table[(int64_t)request_idx * args.block_table_stride_0 +
                           (int64_t)logical_block * args.block_table_stride_1];
    }
    valid = valid && (physical_block >= 0);
    if (mapped && args.common_slot_mapping[token_idx] < 0) {
      valid = false;
    }
    const int32_t slot = physical_block * args.storage_block_size +
                         tmod(compressed_position, args.storage_block_size);
    if (store_mask) {
      args.slot_mapping[token_idx] = valid ? (int64_t)slot : (int64_t)-1;
    }
  }

  // Pre-indexer work metadata: one item per (request, completed compression
  // group), plus item zero of every non-empty request for the raw K suffix.
  const int32_t work_tile_start = (int32_t)((int64_t)blockIdx.x * kWorkBlock);
  if (args.k_work_metadata != nullptr && work_tile_start < args.max_num_work) {
    const int32_t scan_size = args.request_scan_size;
    if (tid < scan_size) {
      int32_t count = 0;
      if (tid < args.num_reqs) {
        const int32_t query_len =
            args.query_start_loc[tid + 1] - args.query_start_loc[tid];
        const int32_t chunk_start = args.seq_lens[tid] - query_len;
        const int32_t num_groups =
            tdiv(args.seq_lens[tid], args.compress_ratio) -
            tdiv(chunk_start, args.compress_ratio);
        count = query_len > 0 ? maxi32(num_groups, 1) : 0;
      }
      shared_work_ends[tid] = count;
    }
    __syncthreads();
    // Inclusive scan of a vector that is at most max_num_seqs wide; the Triton
    // kernel pays for a full tl.cumsum here on every work CTA.
    if (tid == 0) {
      for (int i = 1; i < scan_size; ++i) {
        shared_work_ends[i] += shared_work_ends[i - 1];
      }
    }
    __syncthreads();

    const int32_t total_work = shared_work_ends[scan_size - 1];
    for (int item = tid; item < kWorkBlock; item += kThreads) {
      const int32_t work = work_tile_start + item;
      const bool in_bounds = work < args.max_num_work;
      const bool active = in_bounds && (work < total_work);
      int32_t request = 0;
      for (int step_idx = 0; step_idx < args.work_search_steps; ++step_idx) {
        const int32_t candidate =
            request + (1 << (args.work_search_steps - step_idx - 1));
        if (active && candidate < args.num_reqs) {
          const int32_t idx = mini32(candidate - 1, scan_size - 1);
          if (shared_work_ends[idx] <= work) {
            request = candidate;
          }
        }
      }
      int32_t owner_work_start = 0;
      if (request != 0) {
        owner_work_start = shared_work_ends[mini32(request - 1, scan_size - 1)];
      }
      if (in_bounds) {
        args.k_work_metadata[work * 2] = active ? request : -1;
        args.k_work_metadata[work * 2 + 1] = active ? work - owner_work_start : -1;
      }
    }
  }
}

}  // namespace rdna2_qsa_metadata
}  // namespace vllm

void build_qsa_metadata_rdna2(
    torch::Tensor query_start_loc, torch::Tensor seq_lens,
    torch::Tensor common_slot_mapping, torch::Tensor block_table,
    torch::Tensor token_to_req, torch::Tensor logical_positions,
    torch::Tensor visible_blocks, torch::Tensor slot_mapping,
    const std::optional<torch::Tensor>& k_work_metadata, int64_t num_reqs,
    int64_t num_mapped_tokens, int64_t num_tokens, int64_t max_num_work,
    int64_t num_search_steps, int64_t work_search_steps,
    int64_t request_scan_size, int64_t storage_block_size,
    int64_t compress_ratio, int64_t circular_buffer_size) {
  using namespace vllm::rdna2_qsa_metadata;
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt &&
                  token_to_req.scalar_type() == at::kInt &&
                  visible_blocks.scalar_type() == at::kInt,
              "build_qsa_metadata_rdna2: expected int32 inputs");
  TORCH_CHECK(common_slot_mapping.scalar_type() == at::kLong &&
                  logical_positions.scalar_type() == at::kLong &&
                  slot_mapping.scalar_type() == at::kLong,
              "build_qsa_metadata_rdna2: expected int64 slot tensors");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(num_reqs > 0, "build_qsa_metadata_rdna2 needs at least one request");
  TORCH_CHECK(compress_ratio > 0 && storage_block_size > 0,
              "compress_ratio and storage_block_size must be positive");

  const at::cuda::OptionalCUDAGuard guard(device_of(query_start_loc));
  auto stream = at::cuda::getCurrentCUDAStream();

  int32_t* k_work = nullptr;
  if (k_work_metadata.has_value() && k_work_metadata->numel() > 0) {
    TORCH_CHECK(k_work_metadata->scalar_type() == at::kInt &&
                    k_work_metadata->dim() == 2 && k_work_metadata->size(1) == 2,
                "k_work_metadata must be [num_work, 2] int32");
    k_work = reinterpret_cast<int32_t*>(k_work_metadata->data_ptr());
  }

  MetadataArgs args;
  args.query_start_loc =
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr());
  args.seq_lens = reinterpret_cast<const int32_t*>(seq_lens.data_ptr());
  args.common_slot_mapping =
      reinterpret_cast<const int64_t*>(common_slot_mapping.data_ptr());
  args.block_table = reinterpret_cast<const int32_t*>(block_table.data_ptr());
  args.token_to_req = reinterpret_cast<int32_t*>(token_to_req.data_ptr());
  args.logical_positions =
      reinterpret_cast<int64_t*>(logical_positions.data_ptr());
  args.visible_blocks = reinterpret_cast<int32_t*>(visible_blocks.data_ptr());
  args.slot_mapping = reinterpret_cast<int64_t*>(slot_mapping.data_ptr());
  args.k_work_metadata = k_work;
  args.block_table_stride_0 = block_table.stride(0);
  args.block_table_stride_1 = block_table.stride(1);
  args.num_reqs = (int32_t)num_reqs;
  args.num_mapped_tokens = (int32_t)num_mapped_tokens;
  args.num_tokens = (int32_t)num_tokens;
  args.max_num_work = (int32_t)max_num_work;
  args.num_search_steps = (int32_t)num_search_steps;
  args.work_search_steps = (int32_t)work_search_steps;
  args.request_scan_size = (int32_t)std::max<int64_t>(request_scan_size, 1);
  args.storage_block_size = (int32_t)storage_block_size;
  args.compress_ratio = (int32_t)compress_ratio;
  args.circular_buffer_size = (int32_t)circular_buffer_size;
  args.num_block_table_columns = (int32_t)block_table.size(1);
  args.num_block_table_rows = (int32_t)block_table.size(0);

  const int64_t num_token_blocks = (num_tokens + kTokenBlock - 1) / kTokenBlock;
  const int64_t num_work_blocks =
      (k_work != nullptr && max_num_work > 0)
          ? (max_num_work + kWorkBlock - 1) / kWorkBlock
          : 0;
  const int64_t grid = std::max<int64_t>(
      {num_token_blocks, num_work_blocks, static_cast<int64_t>(1)});
  const size_t smem = (size_t)args.request_scan_size * sizeof(int32_t);

  if (circular_buffer_size > 0) {
    build_qsa_metadata_kernel<kSlotCircular>
        <<<grid, kThreads, smem, stream.stream()>>>(args);
  } else if (compress_ratio != 1) {
    build_qsa_metadata_kernel<kSlotCompressed>
        <<<grid, kThreads, smem, stream.stream()>>>(args);
  } else {
    build_qsa_metadata_kernel<kSlotNone>
        <<<grid, kThreads, smem, stream.stream()>>>(args);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
