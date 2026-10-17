// HIP replacements for the v1 worker's per-step Triton index/arithmetic
// kernels on gfx1030 (RDNA2):
//
//   vllm/v1/worker/gpu/block_table.py::_gather_block_tables_kernel
//   vllm/v1/worker/gpu/block_table.py::_compute_slot_mappings_kernel
//   vllm/v1/worker/gpu/buffer_utils.py::_apply_write_kernel  (single group)
//   vllm/v1/worker/gpu/buffer_utils.py::_apply_write_kernel  (MULTI_GROUP)
//   vllm/v1/worker/utils.py::_zero_kv_blocks_kernel
//
// All five run once per decode step inside a FULL cudagraph capture, so the
// host wrappers only launch: no host syncs, no `.item()`, no allocations.
// Every write goes into a caller-provided buffer.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace vllm {
namespace rdna2_v1_worker {

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// _gather_block_tables_kernel
//   grid = (num_kv_cache_groups, num_reqs_padded)
// Rows >= num_reqs are zeroed over the full row width (= block_table stride).
// ---------------------------------------------------------------------------
__global__ void gather_block_tables_kernel(
    const int32_t* __restrict__ idx_mapping,
    const uint64_t* __restrict__ src_ptrs,
    const uint64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ block_table_strides,
    const int32_t* __restrict__ num_blocks_ptr,
    const int64_t num_blocks_stride, const int num_reqs) {
  const int group = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int stride = (int)block_table_strides[group];

  int32_t* __restrict__ dst = reinterpret_cast<int32_t*>(
                                  (uintptr_t)dst_ptrs[group]) +
                              (int64_t)batch_idx * stride;

  if (batch_idx >= num_reqs) {
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
      dst[i] = 0;
    }
    return;
  }

  const int req_idx = idx_mapping[batch_idx];
  const int num_blocks =
      num_blocks_ptr[(int64_t)group * num_blocks_stride + req_idx];
  const int32_t* __restrict__ src =
      reinterpret_cast<const int32_t*>((uintptr_t)src_ptrs[group]) +
      (int64_t)req_idx * stride;

  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ---------------------------------------------------------------------------
// _compute_slot_mappings_kernel
//   grid = (num_kv_cache_groups, num_reqs + 1)
// The last row (batch_idx == num_reqs) pads [actual_num_tokens, max_num_tokens)
// with PAD_SLOT_ID. slot_mappings is int64.
// ---------------------------------------------------------------------------
__global__ void compute_slot_mappings_kernel(
    const int64_t max_num_tokens,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ pos,
    const uint64_t* __restrict__ block_table_ptrs,
    const int64_t* __restrict__ block_table_strides,
    const int32_t* __restrict__ block_sizes,
    const int32_t* __restrict__ kernel_block_sizes,
    const uint8_t* __restrict__ slot_mapping_enabled,
    const uint8_t* __restrict__ dcp_sharded,
    int64_t* __restrict__ slot_mappings,
    const int64_t slot_mappings_stride, const int cp_rank, const int cp_size,
    const int cp_interleave, const int64_t pad_id) {
  const int group = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_reqs = gridDim.y - 1;

  int64_t* __restrict__ out = slot_mappings + (int64_t)group * slot_mappings_stride;

  if (batch_idx == num_reqs) {
    const int actual_num_tokens = query_start_loc[batch_idx];
    for (int64_t i = actual_num_tokens + threadIdx.x; i < max_num_tokens;
         i += blockDim.x) {
      out[i] = pad_id;
    }
    return;
  }

  const int32_t* __restrict__ block_table =
      reinterpret_cast<const int32_t*>((uintptr_t)block_table_ptrs[group]);
  const int64_t block_table_stride = block_table_strides[group];
  const int64_t kv_block_size = block_sizes[group];
  const int64_t kernel_block_size = kernel_block_sizes[group];
  const bool mapping_enabled = slot_mapping_enabled[group] != 0;
  bool sharded = false;
  if (cp_size != 1) {
    sharded = dcp_sharded[group] != 0;
  }

  const int req_state_idx = idx_mapping[batch_idx];
  const bool is_real_req = req_state_idx >= 0;
  const int start_idx = query_start_loc[batch_idx];
  const int end_idx = query_start_loc[batch_idx + 1];

  const int64_t virtual_block_size = kv_block_size * cp_size;
  const int64_t cp_interleave_total = (int64_t)cp_interleave * cp_size;

  for (int64_t t = (int64_t)start_idx + threadIdx.x; t < end_idx;
       t += blockDim.x) {
    const int64_t position = pos[t];
    int64_t local_positions = position;
    bool is_local = true;
    if (cp_size != 1) {
      const int64_t virtual_block_index = position / virtual_block_size;
      const int64_t virtual_block_offset = position % virtual_block_size;
      const bool local =
          (virtual_block_offset / cp_interleave) % cp_size == cp_rank;
      const int64_t rounds = virtual_block_offset / cp_interleave_total;
      const int64_t remainder = virtual_block_offset % cp_interleave;
      const int64_t local_offsets = rounds * cp_interleave + remainder;
      local_positions = virtual_block_index * kv_block_size + local_offsets;
      if (!sharded) {
        local_positions = position;
      }
      is_local = (!sharded) || local;
    }

    const int64_t block_index =
        mapping_enabled ? local_positions / kernel_block_size : 0;
    const int64_t block_offset = local_positions % kernel_block_size;

    int64_t block_number = 0;
    if (is_local && is_real_req) {
      block_number =
          block_table[(int64_t)req_state_idx * block_table_stride + block_index];
    }

    int64_t slot_id = block_number * kernel_block_size + block_offset;
    if (cp_size != 1 && !is_local) {
      slot_id = pad_id;
    }
    if (!(mapping_enabled && is_real_req)) {
      slot_id = pad_id;
    }
    out[t] = slot_id;
  }
}

// ---------------------------------------------------------------------------
// _apply_write_kernel (single group)
//   grid = (num_writes,)
// `out` is the target tensor's data pointer, `out_stride` its row stride.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void apply_write_kernel(
    T* __restrict__ out, const int64_t out_stride,
    const int32_t* __restrict__ write_indices,
    const int32_t* __restrict__ write_starts,
    const T* __restrict__ write_contents,
    const int32_t* __restrict__ write_cu_lens) {
  const int pid = blockIdx.x;
  const int64_t row_idx = write_indices[pid];
  const int64_t start_idx = write_starts[pid];
  const int cu_start = pid > 0 ? write_cu_lens[pid - 1] : 0;
  const int cu_end = write_cu_lens[pid];
  const int content_len = cu_end - cu_start;

  T* __restrict__ row_ptr = out + row_idx * out_stride + start_idx;
  for (int i = threadIdx.x; i < content_len; i += blockDim.x) {
    row_ptr[i] = write_contents[cu_start + i];
  }
}

// ---------------------------------------------------------------------------
// _apply_write_kernel (MULTI_GROUP)
//   grid = (num_writes,)
// Each write targets a different output tensor; its base pointer and row stride
// are resolved per write from the pointer/stride tables.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_apply_write_kernel(
    const uint64_t* __restrict__ output_ptrs,
    const int64_t* __restrict__ output_strides,
    const int32_t* __restrict__ write_indices,
    const int32_t* __restrict__ write_starts,
    const T* __restrict__ write_contents,
    const int32_t* __restrict__ write_cu_lens,
    const int32_t* __restrict__ write_group_ids) {
  const int pid = blockIdx.x;
  const int group_id = write_group_ids[pid];
  const int64_t row_idx = write_indices[pid];
  const int64_t start_idx = write_starts[pid];
  const int cu_start = pid > 0 ? write_cu_lens[pid - 1] : 0;
  const int cu_end = write_cu_lens[pid];
  const int content_len = cu_end - cu_start;

  T* __restrict__ out = reinterpret_cast<T*>((uintptr_t)output_ptrs[group_id]);
  const int64_t row_stride = output_strides[group_id];
  T* __restrict__ row_ptr = out + row_idx * row_stride + start_idx;
  for (int i = threadIdx.x; i < content_len; i += blockDim.x) {
    row_ptr[i] = write_contents[cu_start + i];
  }
}

// ---------------------------------------------------------------------------
// _zero_kv_blocks_kernel
//   grid = (num_blocks, num_segments, max_chunks)
// Segment addresses are absolute byte addresses; block strides and page sizes
// are in 4-byte words.
// ---------------------------------------------------------------------------
__global__ void zero_kv_blocks_kernel(
    const uint64_t* __restrict__ seg_addrs,
    const int64_t* __restrict__ seg_block_strides,
    const int64_t* __restrict__ seg_page_sizes,
    const int64_t* __restrict__ block_ids,
    const int64_t chunk_size) {
  const int seg_index = blockIdx.y;
  const int64_t chunk_begin = (int64_t)blockIdx.z * chunk_size;
  const int64_t page_size = seg_page_sizes[seg_index];
  if (chunk_begin >= page_size) {
    return;
  }

  const int64_t block_id = block_ids[blockIdx.x];
  const int64_t block_stride = seg_block_strides[seg_index];
  const int64_t block_offset = block_id * block_stride;

  int32_t* __restrict__ ptr =
      reinterpret_cast<int32_t*>((uintptr_t)seg_addrs[seg_index]) + block_offset;

  const int64_t chunk_limit = chunk_begin + chunk_size;
  const int64_t chunk_end = chunk_limit < page_size ? chunk_limit : page_size;
  const int64_t count = chunk_end - chunk_begin;
  // The block base and the chunk start are 4-word aligned for every real KV
  // page, so the bulk of the clear can use 128-bit stores.
  if (((block_offset + chunk_begin) & 3) == 0) {
    int4* __restrict__ vec = reinterpret_cast<int4*>(ptr + chunk_begin);
    const int64_t n_vec = count >> 2;
    for (int64_t k = threadIdx.x; k < n_vec; k += blockDim.x) {
      vec[k] = make_int4(0, 0, 0, 0);
    }
    for (int64_t c = (n_vec << 2) + threadIdx.x; c < count; c += blockDim.x) {
      ptr[chunk_begin + c] = 0;
    }
  } else {
    for (int64_t c = threadIdx.x; c < count; c += blockDim.x) {
      ptr[chunk_begin + c] = 0;
    }
  }
}

template <typename T>
void dispatch_apply_write(torch::Tensor& out, const int64_t out_stride,
                          const torch::Tensor& indices,
                          const torch::Tensor& starts,
                          const torch::Tensor& contents,
                          const torch::Tensor& cu_lens, const int n,
                          hipStream_t stream) {
  apply_write_kernel<T><<<n, kThreads, 0, stream>>>(
      reinterpret_cast<T*>(out.data_ptr()), out_stride,
      reinterpret_cast<const int32_t*>(indices.data_ptr()),
      reinterpret_cast<const int32_t*>(starts.data_ptr()),
      reinterpret_cast<const T*>(contents.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_lens.data_ptr()));
}

template <typename T>
void dispatch_fused_apply_write(
    const torch::Tensor& output_ptrs, const torch::Tensor& output_strides,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& contents, const torch::Tensor& cu_lens,
    const torch::Tensor& group_ids, const int n, hipStream_t stream) {
  fused_apply_write_kernel<T><<<n, kThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(output_ptrs.data_ptr()),
      reinterpret_cast<const int64_t*>(output_strides.data_ptr()),
      reinterpret_cast<const int32_t*>(indices.data_ptr()),
      reinterpret_cast<const int32_t*>(starts.data_ptr()),
      reinterpret_cast<const T*>(contents.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(group_ids.data_ptr()));
}

inline bool supported_write_dtype(const torch::Tensor& t) {
  return t.scalar_type() == at::kInt || t.scalar_type() == at::kLong ||
         t.scalar_type() == at::kFloat;
}

}  // namespace rdna2_v1_worker
}  // namespace vllm

// Host wrappers stay at global scope so the torch_bindings importer resolves
// them (same pattern as causal_conv1d_rdna2.cu / mrope_rdna2.cu).
void gather_block_tables_rdna2(torch::Tensor batch_idx_to_req_idx,
                               torch::Tensor src_block_table_ptrs,
                               torch::Tensor dst_block_table_ptrs,
                               torch::Tensor block_table_strides,
                               torch::Tensor num_blocks_ptr,
                               int64_t num_blocks_stride, int64_t num_reqs,
                               int64_t num_reqs_padded) {
  using namespace vllm::rdna2_v1_worker;
  TORCH_CHECK(batch_idx_to_req_idx.is_cuda() && src_block_table_ptrs.is_cuda() &&
                  dst_block_table_ptrs.is_cuda() &&
                  block_table_strides.is_cuda() && num_blocks_ptr.is_cuda(),
              "gather_block_tables_rdna2: all tensors must be on HIP");
  TORCH_CHECK(batch_idx_to_req_idx.scalar_type() == at::kInt,
              "idx_mapping must be int32");
  TORCH_CHECK(num_blocks_ptr.scalar_type() == at::kInt,
              "num_blocks must be int32");

  const int num_groups = (int)src_block_table_ptrs.numel();
  if (num_groups == 0 || num_reqs_padded == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(batch_idx_to_req_idx));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)num_groups, (unsigned)num_reqs_padded);
  gather_block_tables_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(batch_idx_to_req_idx.data_ptr()),
      reinterpret_cast<const uint64_t*>(src_block_table_ptrs.data_ptr()),
      reinterpret_cast<const uint64_t*>(dst_block_table_ptrs.data_ptr()),
      reinterpret_cast<const int64_t*>(block_table_strides.data_ptr()),
      reinterpret_cast<const int32_t*>(num_blocks_ptr.data_ptr()),
      num_blocks_stride, (int)num_reqs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_slot_mappings_rdna2(
    int64_t max_num_tokens, torch::Tensor idx_mapping,
    torch::Tensor query_start_loc, torch::Tensor pos,
    torch::Tensor block_table_ptrs, torch::Tensor block_table_strides,
    torch::Tensor block_sizes, torch::Tensor kernel_block_sizes,
    torch::Tensor slot_mapping_enabled, torch::Tensor dcp_sharded,
    torch::Tensor slot_mappings, int64_t slot_mappings_stride,
    int64_t cp_rank, int64_t cp_size, int64_t cp_interleave, int64_t pad_id) {
  using namespace vllm::rdna2_v1_worker;
  TORCH_CHECK(pos.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK(slot_mappings.scalar_type() == at::kLong,
              "slot_mappings must be int64");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
              "query_start_loc must be int32");

  const int num_groups = (int)block_table_ptrs.numel();
  const int num_reqs = (int)idx_mapping.numel();
  if (num_groups == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)num_groups, (unsigned)(num_reqs + 1));
  compute_slot_mappings_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      max_num_tokens,
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int64_t*>(pos.data_ptr()),
      reinterpret_cast<const uint64_t*>(block_table_ptrs.data_ptr()),
      reinterpret_cast<const int64_t*>(block_table_strides.data_ptr()),
      reinterpret_cast<const int32_t*>(block_sizes.data_ptr()),
      reinterpret_cast<const int32_t*>(kernel_block_sizes.data_ptr()),
      reinterpret_cast<const uint8_t*>(slot_mapping_enabled.data_ptr()),
      reinterpret_cast<const uint8_t*>(dcp_sharded.data_ptr()),
      reinterpret_cast<int64_t*>(slot_mappings.data_ptr()),
      slot_mappings_stride, (int)cp_rank, (int)cp_size, (int)cp_interleave,
      pad_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_write_rdna2(torch::Tensor out, int64_t out_stride,
                       torch::Tensor write_indices, torch::Tensor write_starts,
                       torch::Tensor write_contents,
                       torch::Tensor write_cu_lens) {
  using namespace vllm::rdna2_v1_worker;
  const int n = (int)write_indices.numel();
  if (n == 0) {
    return;
  }
  TORCH_CHECK(supported_write_dtype(write_contents),
              "apply_write_rdna2: unsupported write dtype");
  TORCH_CHECK(write_contents.scalar_type() == out.scalar_type(),
              "apply_write_rdna2: contents/out dtype mismatch");
  const at::cuda::OptionalCUDAGuard guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (write_contents.scalar_type()) {
    case at::kInt:
      dispatch_apply_write<int32_t>(out, out_stride, write_indices,
                                    write_starts, write_contents, write_cu_lens,
                                    n, stream.stream());
      break;
    case at::kLong:
      dispatch_apply_write<int64_t>(out, out_stride, write_indices,
                                    write_starts, write_contents, write_cu_lens,
                                    n, stream.stream());
      break;
    default:
      dispatch_apply_write<float>(out, out_stride, write_indices, write_starts,
                                  write_contents, write_cu_lens, n,
                                  stream.stream());
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_apply_write_rdna2(torch::Tensor output_ptrs,
                             torch::Tensor output_strides,
                             torch::Tensor write_indices,
                             torch::Tensor write_starts,
                             torch::Tensor write_contents,
                             torch::Tensor write_cu_lens,
                             torch::Tensor write_group_ids) {
  using namespace vllm::rdna2_v1_worker;
  const int n = (int)write_indices.numel();
  if (n == 0) {
    return;
  }
  TORCH_CHECK(supported_write_dtype(write_contents),
              "fused_apply_write_rdna2: unsupported write dtype");
  const at::cuda::OptionalCUDAGuard guard(device_of(write_contents));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (write_contents.scalar_type()) {
    case at::kInt:
      dispatch_fused_apply_write<int32_t>(
          output_ptrs, output_strides, write_indices, write_starts,
          write_contents, write_cu_lens, write_group_ids, n, stream.stream());
      break;
    case at::kLong:
      dispatch_fused_apply_write<int64_t>(
          output_ptrs, output_strides, write_indices, write_starts,
          write_contents, write_cu_lens, write_group_ids, n, stream.stream());
      break;
    default:
      dispatch_fused_apply_write<float>(
          output_ptrs, output_strides, write_indices, write_starts,
          write_contents, write_cu_lens, write_group_ids, n, stream.stream());
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void zero_kv_blocks_rdna2(torch::Tensor seg_addrs, torch::Tensor seg_block_strides,
                          torch::Tensor seg_page_sizes, torch::Tensor block_ids,
                          int64_t chunk_size, int64_t max_chunks) {
  using namespace vllm::rdna2_v1_worker;
  const int64_t n_blocks = block_ids.numel();
  const int64_t n_segs = seg_addrs.numel();
  if (n_blocks == 0 || n_segs == 0 || max_chunks == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(seg_addrs));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)n_blocks, (unsigned)n_segs, (unsigned)max_chunks);
  zero_kv_blocks_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const uint64_t*>(seg_addrs.data_ptr()),
      reinterpret_cast<const int64_t*>(seg_block_strides.data_ptr()),
      reinterpret_cast<const int64_t*>(seg_page_sizes.data_ptr()),
      reinterpret_cast<const int64_t*>(block_ids.data_ptr()), chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
