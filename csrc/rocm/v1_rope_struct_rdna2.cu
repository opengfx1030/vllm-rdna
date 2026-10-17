// HIP replacements for two per-decode-step v1 worker Triton kernels on gfx1030
// (RDNA2):
//
//   vllm/v1/worker/gpu/mm/rope.py::_prepare_rope_positions_kernel
//   vllm/v1/worker/gpu/structured_outputs.py::_apply_grammar_bitmask_kernel
//
// Both run inside a FULL cudagraph capture, so the host wrappers only launch:
// no host syncs, no `.item()`, no allocations.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>

namespace vllm {
namespace rdna2_rope_struct {

constexpr int kThreads = 256;
// _apply_grammar_bitmask_kernel: BLOCK_SIZE=8192 logits per CTA, one int32
// packed bitmask word per thread.
constexpr int kBitmaskBlock = 8192;
constexpr int kBitsPerWord = 32;
constexpr int kWordsPerBlock = kBitmaskBlock / kBitsPerWord;

// ---------------------------------------------------------------------------
// _prepare_rope_positions_kernel   grid = (num_reqs,)
// M-RoPE positions: prefill rows read the staged per-request position table,
// decode rows are num_computed + delta.
// ---------------------------------------------------------------------------
__global__ void prepare_rope_positions_kernel(
    int64_t* __restrict__ positions, int64_t positions_stride,
    const int32_t* __restrict__ prefill_positions, int64_t prefill_stride0,
    int64_t prefill_stride1, const int32_t* __restrict__ prefill_delta,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ prefill_lens,
    const int32_t* __restrict__ num_computed_tokens, int32_t num_dims) {
  const int32_t batch_idx = blockIdx.x;
  const int32_t req_state_idx = idx_mapping[batch_idx];

  const int32_t prefill_len = prefill_lens[req_state_idx];
  const int32_t num_computed = num_computed_tokens[req_state_idx];
  const bool is_prefill = num_computed < prefill_len;

  const int32_t query_start = query_start_loc[batch_idx];
  const int32_t query_len = query_start_loc[batch_idx + 1] - query_start;
  const int32_t delta = prefill_delta[req_state_idx];

  const int32_t* request =
      prefill_positions + (int64_t)req_state_idx * prefill_stride0;

  for (int32_t i = threadIdx.x; i < query_len; i += blockDim.x) {
    const int32_t orig_pos = num_computed + i;
    for (int32_t j = 0; j < num_dims; ++j) {
      int64_t pos;
      if (is_prefill) {
        pos = (int64_t)request[(int64_t)j * prefill_stride1 + orig_pos];
      } else {
        pos = (int64_t)(orig_pos + delta);
      }
      positions[(int64_t)j * positions_stride + query_start + i] = pos;
    }
  }
}

// ---------------------------------------------------------------------------
// _apply_grammar_bitmask_kernel
//   grid = (num_masks, cdiv(vocab_size, kBitmaskBlock))
// One thread per 32-bit packed bitmask word, staged through shared memory so
// the -inf stores are issued with consecutive lanes on consecutive elements
// (per-thread consecutive elements would stride each warp's stores by 64B).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void apply_grammar_bitmask_kernel(
    scalar_t* __restrict__ logits, int64_t logits_stride,
    const int32_t* __restrict__ logits_indices,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ bitmask, int64_t bitmask_stride,
    int64_t vocab_size, int32_t mask_stride) {
  constexpr int kBlockThreads = kWordsPerBlock;  // 8192 logits per CTA
  constexpr int kChunks = kBitmaskBlock / kBlockThreads;

  __shared__ int32_t s_words[kWordsPerBlock];

  const int32_t bitmask_idx = blockIdx.x;
  const int32_t block_id = blockIdx.y;

  const int32_t mapping_idx = logits_indices[bitmask_idx];
  const int32_t req_idx = mapping_idx / mask_stride;
  const int32_t position_idx = mapping_idx % mask_stride;

  const int32_t cu_start = cu_num_logits[req_idx];
  const int32_t num_req_logits = cu_num_logits[req_idx + 1] - cu_start;
  const int64_t logits_idx = (int64_t)cu_start + position_idx;
  const bool position_is_active = position_idx < num_req_logits;

  // Uniform within the CTA, so the early exit cannot strand a barrier.
  if (!position_is_active) {
    return;
  }

  const int64_t bitmask_offset =
      (int64_t)block_id * kWordsPerBlock + threadIdx.x;
  s_words[threadIdx.x] =
      bitmask_offset < bitmask_stride
          ? bitmask[(int64_t)bitmask_idx * bitmask_stride + bitmask_offset]
          : 0;
  __syncthreads();

  scalar_t* row = logits + logits_idx * logits_stride;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    // One bitmask word covers kBitsPerWord consecutive elements; with 256
    // threads covering 8 words per chunk, each warp reads a single (broadcast)
    // word and the 32 lanes write 32 consecutive elements.
    const int32_t word =
        s_words[chunk * (kBlockThreads / kBitsPerWord) + (threadIdx.x >> 5)];
    const int64_t offset =
        (int64_t)block_id * kBitmaskBlock + chunk * kBlockThreads + threadIdx.x;
    if (offset < vocab_size && ((word >> (threadIdx.x & 31)) & 1) == 0) {
      row[offset] = (scalar_t)(-INFINITY);
    }
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

}  // namespace rdna2_rope_struct
}  // namespace vllm

using namespace vllm::rdna2_rope_struct;
void prepare_rope_positions_rdna2(
    torch::Tensor positions, int64_t positions_stride,
    torch::Tensor prefill_positions, int64_t prefill_stride0,
    int64_t prefill_stride1, torch::Tensor prefill_delta,
    torch::Tensor idx_mapping, torch::Tensor query_start_loc,
    torch::Tensor prefill_lens, torch::Tensor num_computed_tokens,
    int64_t num_dims) {
  TORCH_CHECK(positions.scalar_type() == at::kLong,
              "prepare_rope_positions_rdna2: positions must be int64");
  TORCH_CHECK(prefill_positions.scalar_type() == at::kInt &&
                  prefill_delta.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt &&
                  prefill_lens.scalar_type() == at::kInt &&
                  num_computed_tokens.scalar_type() == at::kInt,
              "prepare_rope_positions_rdna2: int32 inputs expected");
  const int64_t num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  prepare_rope_positions_kernel<<<(unsigned)num_reqs, kThreads, 0,
                                  stream.stream()>>>(
      reinterpret_cast<int64_t*>(positions.data_ptr()), positions_stride,
      reinterpret_cast<const int32_t*>(prefill_positions.data_ptr()),
      prefill_stride0, prefill_stride1,
      reinterpret_cast<const int32_t*>(prefill_delta.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(prefill_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()),
      (int32_t)num_dims);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_grammar_bitmask_rdna2(torch::Tensor logits, int64_t logits_stride,
                                 torch::Tensor logits_indices,
                                 torch::Tensor cu_num_logits,
                                 torch::Tensor bitmask, int64_t bitmask_stride,
                                 int64_t vocab_size, int64_t mask_stride) {
  TORCH_CHECK(logits_indices.scalar_type() == at::kInt &&
                  cu_num_logits.scalar_type() == at::kInt &&
                  bitmask.scalar_type() == at::kInt,
              "apply_grammar_bitmask_rdna2: int32 index/bitmask expected");
  const int64_t num_masks = logits_indices.numel();
  if (num_masks == 0 || vocab_size == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_masks,
                  (unsigned)((vocab_size + kBitmaskBlock - 1) / kBitmaskBlock));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "apply_grammar_bitmask_rdna2", [&] {
        apply_grammar_bitmask_kernel<scalar_t>
            <<<grid, kWordsPerBlock, 0, stream.stream()>>>(
                reinterpret_cast<scalar_t*>(logits.data_ptr()), logits_stride,
                reinterpret_cast<const int32_t*>(logits_indices.data_ptr()),
                reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
                reinterpret_cast<const int32_t*>(bitmask.data_ptr()),
                bitmask_stride, vocab_size, (int32_t)mask_stride);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
