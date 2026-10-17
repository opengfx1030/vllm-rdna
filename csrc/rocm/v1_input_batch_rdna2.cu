// HIP replacements for the v1 worker input-batch Triton kernels on gfx1030
// (RDNA2):
//
//   vllm/v1/worker/gpu/input_batch.py::_prepare_prefill_inputs_kernel
//   vllm/v1/worker/gpu/input_batch.py::_prepare_pos_seq_lens_kernel
//   vllm/v1/worker/gpu/input_batch.py::_combine_sampled_and_draft_tokens_kernel
//   vllm/v1/worker/gpu/input_batch.py::_get_num_sampled_and_rejected_kernel
//   vllm/v1/worker/gpu/input_batch.py::_post_update_kernel
//   vllm/v1/worker/gpu/input_batch.py::_expand_idx_mapping_kernel
//
// `_post_update_num_computed_tokens_kernel` is NOT ported: it is only reached
// from ModelRunner.postprocess_num_computed_tokens, which is guarded by
// `not is_last_pp_rank` (pp_size == 1 here) or by `is_pooling_model` (this is a
// generation model), and it never appears in the Triton launch audit.
//
// All of these run once per decode step inside a FULL cudagraph capture, so the
// host wrappers only launch: no host syncs, no `.item()`, no allocations.
// Every write goes into a caller-provided buffer.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <optional>

namespace vllm {
namespace rdna2_input_batch {

constexpr int kThreads = 256;
// Kernels whose Triton body is a scalar (non-tensor) loop: one logical writer.
constexpr int kScalarThreads = 32;

// ---------------------------------------------------------------------------
// _prepare_prefill_inputs_kernel
//   grid = (num_reqs,)
// Copy this request's scheduled prompt tokens into input_ids, then stage the
// num_lookahead next prefill tokens (0 when past prefill_len).
// ---------------------------------------------------------------------------
__global__ void prepare_prefill_inputs_kernel(
    int32_t* __restrict__ input_ids, int32_t* __restrict__ next_prefill_tokens,
    const int64_t next_prefill_tokens_stride, const int32_t num_lookahead,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ all_token_ids,
    const int64_t all_token_ids_stride,
    const int32_t* __restrict__ prefill_lens,
    const int32_t* __restrict__ num_computed_tokens) {
  const int batch_idx = blockIdx.x;
  const int req_state_idx = idx_mapping[batch_idx];
  const int32_t prefill_len = prefill_lens[req_state_idx];
  const int32_t num_computed = num_computed_tokens[req_state_idx];
  if (num_computed >= prefill_len) {
    return;
  }

  const int32_t query_start = query_start_loc[batch_idx];
  const int32_t query_len = query_start_loc[batch_idx + 1] - query_start;
  const int32_t* __restrict__ request =
      all_token_ids + (int64_t)req_state_idx * all_token_ids_stride;

  for (int i = threadIdx.x; i < query_len; i += blockDim.x) {
    input_ids[query_start + i] = request[num_computed + i];
  }

  for (int k = threadIdx.x; k < num_lookahead; k += blockDim.x) {
    const int32_t pos = num_computed + query_len + k;
    next_prefill_tokens[(int64_t)k * next_prefill_tokens_stride +
                        req_state_idx] = (pos < prefill_len) ? request[pos] : 0;
  }
}

// ---------------------------------------------------------------------------
// _prepare_pos_seq_lens_kernel
//   grid = (num_reqs + 1,)
// The last block (req_id == num_reqs) zeroes the unused tail of seq_lens so a
// full CUDA graph sees a stable buffer.
// ---------------------------------------------------------------------------
__global__ void prepare_pos_seq_lens_kernel(
    int64_t* __restrict__ pos, int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ num_computed_tokens,
    const int64_t max_num_reqs) {
  const int req_id = blockIdx.x;
  const int num_reqs = gridDim.x - 1;
  if (req_id == num_reqs) {
    for (int64_t i = (int64_t)num_reqs + threadIdx.x; i < max_num_reqs;
         i += blockDim.x) {
      seq_lens[i] = 0;
    }
    return;
  }

  const int req_state_idx = idx_mapping[req_id];
  const int32_t num_computed = num_computed_tokens[req_state_idx];
  const int32_t start = query_start_loc[req_id];
  const int32_t query_len = query_start_loc[req_id + 1] - start;

  seq_lens[req_id] = num_computed + query_len;
  for (int i = threadIdx.x; i < query_len; i += blockDim.x) {
    pos[start + i] = (int64_t)(num_computed + i);
  }
}

// ---------------------------------------------------------------------------
// _combine_sampled_and_draft_tokens_kernel
//   grid = (num_reqs,)
// ---------------------------------------------------------------------------
__global__ void combine_sampled_and_draft_tokens_kernel(
    int32_t* __restrict__ input_ids, const int32_t* __restrict__ idx_mapping,
    const int64_t* __restrict__ last_sampled_tokens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefill_len,
    const int64_t* __restrict__ draft_tokens, const int64_t draft_tokens_stride,
    const int32_t* __restrict__ cu_num_logits,
    int64_t* __restrict__ logits_indices,
    const int32_t num_new_sampled_tokens) {
  const int batch_idx = blockIdx.x;
  const int req_state_idx = idx_mapping[batch_idx];

  const int32_t cu_start = cu_num_logits[batch_idx];
  const int32_t num_logits = cu_num_logits[batch_idx + 1] - cu_start;
  const int32_t num_draft_tokens = num_logits - num_new_sampled_tokens;

  const int32_t query_end = query_start_loc[batch_idx + 1];
  const int32_t logits_start = query_end - num_logits;
  for (int i = threadIdx.x; i < num_logits; i += blockDim.x) {
    logits_indices[cu_start + i] = (int64_t)(logits_start + i);
  }

  const int32_t seq_len = seq_lens[batch_idx];
  const int32_t prefill_len_r = prefill_len[req_state_idx];
  if (seq_len <= prefill_len_r) {
    // Handling prefill tokens. No sampled or draft tokens.
    return;
  }

  // Keep prompt-tail slots intact; only rewrite generated-token slots.
  if (num_new_sampled_tokens > 0 && seq_len - num_logits >= prefill_len_r) {
    if (threadIdx.x == 0) {
      input_ids[logits_start] = (int32_t)last_sampled_tokens[req_state_idx];
    }
  }

  if (num_draft_tokens > 0) {
    const int64_t* __restrict__ draft_row =
        draft_tokens + (int64_t)req_state_idx * draft_tokens_stride;
    const int32_t dst = query_end - num_draft_tokens;
    for (int i = threadIdx.x; i < num_draft_tokens; i += blockDim.x) {
      input_ids[dst + i] = (int32_t)draft_row[i];
    }
  }
}

// ---------------------------------------------------------------------------
// _get_num_sampled_and_rejected_kernel
//   grid = (num_reqs,)
// Chunked-prefill requests produce no output token, so both counts are forced
// to 0. num_sampled is updated in place.
// ---------------------------------------------------------------------------
__global__ void get_num_sampled_and_rejected_kernel(
    int32_t* __restrict__ num_sampled, int32_t* __restrict__ num_rejected,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ prefill_len) {
  const int batch_idx = blockIdx.x;
  const int req_state_idx = idx_mapping[batch_idx];
  const bool is_chunked_prefilling =
      seq_lens[batch_idx] < prefill_len[req_state_idx];

  int32_t ns = is_chunked_prefilling ? 0 : num_sampled[batch_idx];
  num_sampled[batch_idx] = ns;

  const int32_t num_logits =
      cu_num_logits[batch_idx + 1] - cu_num_logits[batch_idx];
  num_rejected[batch_idx] = is_chunked_prefilling ? 0 : (num_logits - ns);
}

// ---------------------------------------------------------------------------
// _post_update_kernel
//   grid = (num_reqs,)
// Triton runs the sampled-token loop serially (`for i in range(num_sampled)`
// with a scalar bound), so lane 0 alone reproduces it bit-for-bit.
// ---------------------------------------------------------------------------
// sampled_tokens is int64 on the MTP spec path (rejection_sampler_utils.py
// allocates it with dtype=torch.int64) and int32 on the plain sampler path.
template <typename S>
__global__ void post_update_kernel(const int32_t* __restrict__ idx_mapping,
                                   int32_t* __restrict__ num_computed_tokens,
                                   int64_t* __restrict__ last_sampled_tokens,
                                   int32_t* __restrict__ output_bin_counts,
                                   const int64_t output_bin_counts_stride,
                                   const S* __restrict__ sampled_tokens,
                                   const int64_t sampled_tokens_stride,
                                   const int32_t* __restrict__ num_sampled,
                                   const int32_t* __restrict__ num_rejected,
                                   const int32_t* __restrict__ query_start_loc,
                                   int32_t* __restrict__ all_token_ids,
                                   const int64_t all_token_ids_stride,
                                   int32_t* __restrict__ total_len) {
  const int req_id = blockIdx.x;
  const int req_state_idx = idx_mapping[req_id];
  if (req_state_idx < 0) {
    // Filter rows with negative index entries.
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }

  const int32_t total_len_v = total_len[req_state_idx];
  const int32_t ns = num_sampled[req_id];
  if (ns > 0) {
    const int32_t token_id =
        (int32_t)sampled_tokens[req_id * sampled_tokens_stride + ns - 1];
    last_sampled_tokens[req_state_idx] = (int64_t)token_id;
    total_len[req_state_idx] = total_len_v + ns;
  }

  for (int32_t i = 0; i < ns; i++) {
    const int32_t token_id =
        (int32_t)sampled_tokens[req_id * sampled_tokens_stride + i];
    all_token_ids[(int64_t)req_state_idx * all_token_ids_stride + total_len_v +
                  i] = token_id;
    if (output_bin_counts != nullptr) {
      int32_t* __restrict__ bin =
          output_bin_counts +
          (int64_t)req_state_idx * output_bin_counts_stride + token_id;
      *bin = *bin + 1;
    }
  }

  int32_t query_len = 0;
  if (query_start_loc != nullptr) {
    query_len = query_start_loc[req_id + 1] - query_start_loc[req_id];
  }
  const int32_t computed_delta = query_len - num_rejected[req_id];
  if (computed_delta != 0) {
    num_computed_tokens[req_state_idx] += computed_delta;
  }
}

// ---------------------------------------------------------------------------
// _expand_idx_mapping_kernel
//   grid = (num_reqs,)
// ---------------------------------------------------------------------------
__global__ void expand_idx_mapping_kernel(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ expanded_idx_mapping,
    int32_t* __restrict__ expanded_local_pos,
    const int32_t* __restrict__ cu_num_logits) {
  const int req_idx = blockIdx.x;
  const int32_t start_idx = cu_num_logits[req_idx];
  const int32_t num_tokens = cu_num_logits[req_idx + 1] - start_idx;
  const int32_t req_state_idx = idx_mapping[req_idx];
  for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
    expanded_idx_mapping[start_idx + i] = req_state_idx;
    expanded_local_pos[start_idx + i] = i;
  }
}

}  // namespace rdna2_input_batch
}  // namespace vllm

// Host wrappers stay at global scope so the torch_bindings importer resolves
// them (same pattern as v1_worker_rdna2.cu).
void prepare_prefill_inputs_rdna2(
    torch::Tensor input_ids, torch::Tensor next_prefill_tokens,
    int64_t next_prefill_tokens_stride, int64_t num_lookahead,
    torch::Tensor idx_mapping, torch::Tensor query_start_loc,
    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor prefill_len, torch::Tensor num_computed_tokens) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(input_ids.scalar_type() == at::kInt &&
                  next_prefill_tokens.scalar_type() == at::kInt &&
                  all_token_ids.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt &&
                  prefill_len.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt,
              "prepare_prefill_inputs_rdna2: all tensors must be int32");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  prepare_prefill_inputs_kernel<<<num_reqs, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<int32_t*>(input_ids.data_ptr()),
      reinterpret_cast<int32_t*>(next_prefill_tokens.data_ptr()),
      next_prefill_tokens_stride, (int32_t)num_lookahead,
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(all_token_ids.data_ptr()),
      all_token_ids_stride,
      reinterpret_cast<const int32_t*>(prefill_len.data_ptr()),
      reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_pos_seq_lens_rdna2(torch::Tensor pos, torch::Tensor seq_lens,
                                torch::Tensor idx_mapping,
                                torch::Tensor query_start_loc,
                                torch::Tensor num_computed_tokens,
                                int64_t max_num_reqs) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(pos.scalar_type() == at::kLong, "pos must be int64");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt,
              "prepare_pos_seq_lens_rdna2: index tensors must be int32");

  const int num_reqs = (int)idx_mapping.numel();
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  prepare_pos_seq_lens_kernel<<<num_reqs + 1, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<int64_t*>(pos.data_ptr()),
      reinterpret_cast<int32_t*>(seq_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()),
      max_num_reqs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void combine_sampled_and_draft_tokens_rdna2(
    torch::Tensor input_ids, torch::Tensor idx_mapping,
    torch::Tensor last_sampled_tokens, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor prefill_len,
    torch::Tensor draft_tokens, int64_t draft_tokens_stride,
    torch::Tensor cu_num_logits, torch::Tensor logits_indices,
    int64_t num_new_sampled_tokens) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(input_ids.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  prefill_len.scalar_type() == at::kInt &&
                  cu_num_logits.scalar_type() == at::kInt,
              "combine_sampled_and_draft_tokens_rdna2: index tensors int32");
  TORCH_CHECK(last_sampled_tokens.scalar_type() == at::kLong &&
                  draft_tokens.scalar_type() == at::kLong,
              "last_sampled_tokens/draft_tokens must be int64");
  TORCH_CHECK(logits_indices.scalar_type() == at::kLong,
              "logits_indices must be int64");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  combine_sampled_and_draft_tokens_kernel<<<num_reqs, kThreads, 0,
                                            stream.stream()>>>(
      reinterpret_cast<int32_t*>(input_ids.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int64_t*>(last_sampled_tokens.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(seq_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(prefill_len.data_ptr()),
      reinterpret_cast<const int64_t*>(draft_tokens.data_ptr()),
      draft_tokens_stride,
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
      reinterpret_cast<int64_t*>(logits_indices.data_ptr()),
      (int32_t)num_new_sampled_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void get_num_sampled_and_rejected_rdna2(torch::Tensor num_sampled,
                                        torch::Tensor num_rejected,
                                        torch::Tensor seq_lens,
                                        torch::Tensor cu_num_logits,
                                        torch::Tensor idx_mapping,
                                        torch::Tensor prefill_len) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(num_sampled.scalar_type() == at::kInt &&
                  num_rejected.scalar_type() == at::kInt,
              "num_sampled/num_rejected must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt &&
                  cu_num_logits.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  prefill_len.scalar_type() == at::kInt,
              "get_num_sampled_and_rejected_rdna2: index tensors int32");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  get_num_sampled_and_rejected_kernel<<<num_reqs, kScalarThreads, 0,
                                        stream.stream()>>>(
      reinterpret_cast<int32_t*>(num_sampled.data_ptr()),
      reinterpret_cast<int32_t*>(num_rejected.data_ptr()),
      reinterpret_cast<const int32_t*>(seq_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(prefill_len.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void post_update_rdna2(torch::Tensor idx_mapping,
                       torch::Tensor num_computed_tokens,
                       torch::Tensor last_sampled_tokens,
                       const std::optional<torch::Tensor>& output_bin_counts,
                       int64_t output_bin_counts_stride,
                       torch::Tensor sampled_tokens,
                       int64_t sampled_tokens_stride, torch::Tensor num_sampled,
                       torch::Tensor num_rejected,
                       const std::optional<torch::Tensor>& query_start_loc,
                       torch::Tensor all_token_ids,
                       int64_t all_token_ids_stride, torch::Tensor total_len) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt &&
                  (sampled_tokens.scalar_type() == at::kInt ||
                   sampled_tokens.scalar_type() == at::kLong) &&
                  num_sampled.scalar_type() == at::kInt &&
                  num_rejected.scalar_type() == at::kInt &&
                  all_token_ids.scalar_type() == at::kInt &&
                  total_len.scalar_type() == at::kInt,
              "post_update_rdna2: unexpected dtype");
  TORCH_CHECK(last_sampled_tokens.scalar_type() == at::kLong,
              "last_sampled_tokens must be int64");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();

  int32_t* bin_counts = nullptr;
  if (output_bin_counts.has_value()) {
    TORCH_CHECK(output_bin_counts->scalar_type() == at::kInt,
                "output_bin_counts must be int32");
    bin_counts = reinterpret_cast<int32_t*>(output_bin_counts->data_ptr());
  }
  const int32_t* qsl = nullptr;
  if (query_start_loc.has_value()) {
    TORCH_CHECK(query_start_loc->scalar_type() == at::kInt,
                "query_start_loc must be int32");
    qsl = reinterpret_cast<const int32_t*>(query_start_loc->data_ptr());
  }

  auto launch = [&](auto sampled_tag) {
    using S = decltype(sampled_tag);
    post_update_kernel<S><<<num_reqs, kScalarThreads, 0, stream.stream()>>>(
        reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
        reinterpret_cast<int32_t*>(num_computed_tokens.data_ptr()),
        reinterpret_cast<int64_t*>(last_sampled_tokens.data_ptr()), bin_counts,
        output_bin_counts_stride,
        reinterpret_cast<const S*>(sampled_tokens.data_ptr()),
        sampled_tokens_stride,
        reinterpret_cast<const int32_t*>(num_sampled.data_ptr()),
        reinterpret_cast<const int32_t*>(num_rejected.data_ptr()), qsl,
        reinterpret_cast<int32_t*>(all_token_ids.data_ptr()),
        all_token_ids_stride,
        reinterpret_cast<int32_t*>(total_len.data_ptr()));
  };
  if (sampled_tokens.scalar_type() == at::kLong) {
    launch(int64_t{});
  } else {
    launch(int32_t{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand_idx_mapping_rdna2(torch::Tensor idx_mapping,
                              torch::Tensor expanded_idx_mapping,
                              torch::Tensor expanded_local_pos,
                              torch::Tensor cu_num_logits) {
  using namespace vllm::rdna2_input_batch;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kInt &&
                  expanded_idx_mapping.scalar_type() == at::kInt &&
                  expanded_local_pos.scalar_type() == at::kInt &&
                  cu_num_logits.scalar_type() == at::kInt,
              "expand_idx_mapping_rdna2: all tensors must be int32");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  expand_idx_mapping_kernel<<<num_reqs, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<int32_t*>(expanded_local_pos.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
