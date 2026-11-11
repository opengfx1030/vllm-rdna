// HIP replacements for the v1 sampler's Triton kernels on gfx1030 (RDNA2):
//
//   vllm/v1/worker/gpu/sample/gumbel.py::_temperature_kernel
//   vllm/v1/worker/gpu/sample/gumbel.py::_gumbel_sample_kernel
//   vllm/v1/worker/gpu/sample/penalties.py::_penalties_kernel
//   vllm/v1/worker/gpu/sample/penalties.py::_bincount_kernel
//   vllm/v1/worker/gpu/sample/min_p.py::_min_p_kernel
//   vllm/v1/worker/gpu/sample/logit_bias.py::_bias_kernel
//   vllm/v1/worker/gpu/sample/bad_words.py::_bad_words_kernel
//   vllm/v1/worker/gpu/sample/logprob.py::_topk_log_softmax_kernel
//   vllm/v1/worker/gpu/sample/logprob.py::_ranks_kernel
//   vllm/v1/worker/gpu/sample/prompt_logprob.py::_prompt_logprobs_token_ids_kernel
//
// All of them run once per decode step inside a FULL cudagraph capture, so the
// host wrappers only launch: no host syncs, no `.item()`, no allocations.
// Every write goes into a caller-provided buffer.
//
// Bit-exactness notes (all verified against the Triton kernels by
// tests/v1/worker/test_sample_ops_rdna2.py):
//   * tl.exp(x) is `exp2f(x * log2e)`; Triton's AMD lowering adds a denormal
//     rescale around the raw v_exp_f32, which is what exp2f() also emits.
//   * tl.log(x) is NOT HIP's `__logf` (that form adds the product's rounded
//     value, Triton adds the exact product through v_fmac). The exact Triton
//     sequence is re-implemented in tl_log().
//   * tl.sum over a BLOCK_SIZE tensor with 4 warps reduces in Triton's blocked
//     layout: a balanced tree over the thread's 4-element vector groups, then a
//     16/8/4/2/1 lane butterfly, then ((w0+w2)+(w1+w3)) across warps. Only the
//     _topk_log_softmax lse is order sensitive; reduce_block_sum_1024()
//     reproduces it exactly.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace vllm {
namespace rdna2_sample {

constexpr int kThreads = 256;
constexpr int kSampleThreads = 128;  // matches Triton's num_warps = 4
constexpr int kGumbelBlock = 1024;
constexpr int kTopkBlock = 1024;

// ---------------------------------------------------------------------------
// Triton math lowering helpers (gfx1030)
// ---------------------------------------------------------------------------

// Triton's `math.exp` for fp32: the AMDGPU `v_exp_f32` hardware op on a scaled
// argument, plus a 2**-64 rescale so subnormal results are not flushed.
__device__ __forceinline__ float tl_exp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

// Triton's `math.log` for fp32: `v_log_f32` (log2) followed by an FMA-based
// refinement, exactly the sequence Triton emits. HIP's `__logf` uses a
// different final association and is off by 1 ulp on some inputs.
__device__ __forceinline__ float tl_log(float x) {
  const float c = __int_as_float(0x3f317217);  // ln(2)
  const float d = __int_as_float(0x3377d1cf);
  const bool denorm = x < __int_as_float(0x00800000);
  const float raw = __log2f(__builtin_amdgcn_ldexp(x, denorm ? 32 : 0));
  const float r = c * raw;
  const float e = __fmaf_rn(c, raw, -r);
  const float refined = __fmaf_rn(raw, c, __fmaf_rn(raw, d, e));
  const float y = (fabsf(raw) < __int_as_float(0x7f800000)) ? refined : raw;
  return denorm ? (y - __int_as_float(0x41b17218)) : y;
}

// gumbel.py::_log1p_neg_stable
__device__ __forceinline__ float tl_log1p_neg_stable(float value) {
  float p = 1.0f / 8.0f;
  p = __fmaf_rn(value, p, 1.0f / 7.0f);
  p = __fmaf_rn(value, p, 1.0f / 6.0f);
  p = __fmaf_rn(value, p, 1.0f / 5.0f);
  p = __fmaf_rn(value, p, 1.0f / 4.0f);
  p = __fmaf_rn(value, p, 1.0f / 3.0f);
  p = __fmaf_rn(value, p, 1.0f / 2.0f);
  p = __fmaf_rn(value, p, 1.0f);
  const float series = -value * p;
  const float direct = tl_log(fmaxf(1.0f - value, 5.960464477539063e-08f));
  return value < 0.25f ? series : direct;
}

// gumbel.py murmur3 helpers (uint32 wraparound arithmetic).
__device__ __forceinline__ uint32_t murmur3_rotl32(uint32_t v, int s) {
  return (v << s) | (v >> (32 - s));
}

__device__ __forceinline__ uint32_t murmur3_mix(uint32_t h, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = murmur3_rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = murmur3_rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}

__device__ __forceinline__ uint32_t murmur3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}

__device__ __forceinline__ uint32_t murmur3_hash32(int64_t seed, int64_t pos,
                                                   uint32_t offset) {
  const uint64_t s = (uint64_t)seed;
  const uint64_t p = (uint64_t)pos;
  uint32_t h = 0u;
  h = murmur3_mix(h, (uint32_t)(s & 0xFFFFFFFFu));
  h = murmur3_mix(h, (uint32_t)((s >> 32) & 0xFFFFFFFFu));
  h = murmur3_mix(h, (uint32_t)(p & 0xFFFFFFFFu));
  h = murmur3_mix(h, offset);
  return murmur3_fmix32(h ^ 16u);
}

__device__ __forceinline__ float murmur3_uniform32(int64_t seed, int64_t pos,
                                                   uint32_t offset) {
  const uint32_t random32 = murmur3_hash32(seed, pos, offset);
  const float hi16 = (float)(int32_t)(random32 >> 16);
  const float lo16 = (float)(int32_t)(random32 & 0xFFFFu);
  return hi16 * 1.52587890625e-05f + (lo16 + 0.5f) * 2.3283064365386963e-10f;
}

// ---------------------------------------------------------------------------
// Block reductions
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, off));
  return v;
}

__device__ __forceinline__ float block_max(float v) {
  __shared__ float smem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (int)(blockDim.x >> 5);
  v = warp_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < nwarps) ? smem[lane] : -INFINITY;
    t = warp_max(t);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ int32_t block_sum_int(int32_t v) {
  __shared__ int32_t smem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (int)(blockDim.x >> 5);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    int32_t t = (lane < nwarps) ? smem[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      t += __shfl_xor_sync(0xffffffffffffffffULL, t, off);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const int32_t r = smem[0];
  __syncthreads();
  return r;
}

// Reproduces Triton's `tl.sum` over a 1024-wide tensor compiled with 4 warps
// (blocked layout: sizePerThread = 4, threadsPerWarp = 32, warpsPerCTA = 4):
// the per-thread partial is the balanced tree over the thread's two 4-element
// vector groups, then a 16/8/4/2/1 lane butterfly, then ((w0+w2)+(w1+w3)).
// `partial` must already be reduced over one 4-element group pair per thread.
// Must be called by exactly 128 threads.
// `parity` ping-pongs between two shared buffers so one barrier per call is
// enough (the caller alternates it); Triton emits a single s_barrier too.
__device__ __forceinline__ float reduce_block_sum_1024(float partial,
                                                       const int parity) {
  __shared__ float smem[2][4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);
  if (lane == 0) smem[parity][warp] = partial;
  __syncthreads();
  return (smem[parity][0] + smem[parity][2]) +
         (smem[parity][1] + smem[parity][3]);
}

__device__ __forceinline__ float reduce_block_max_1024(float partial) {
  __shared__ float smem[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial =
        fmaxf(partial, __shfl_xor_sync(0xffffffffffffffffULL, partial, off));
  if (lane == 0) smem[warp] = partial;
  __syncthreads();
  const float r = fmaxf(fmaxf(smem[0], smem[1]), fmaxf(smem[2], smem[3]));
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------
// 16-byte vector loads. The fp32 sampler path is bandwidth bound, so every
// whole-row pass uses float4 when the row base and the length allow it; fp16
// (never used by the serving sampler, which copies logits to fp32 first) stays
// scalar. Only the *access* is vectorised; every arithmetic association is
// unchanged.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool vec4_ok(const void* row, const int64_t count) {
  return ((uintptr_t)row & 15) == 0 && (count & 3) == 0;
}

__device__ __forceinline__ float4 load4f(const float* __restrict__ p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ void store4f(float* __restrict__ p, const float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}

// Per-thread max over row[0, vocab_size).
template <typename T>
__device__ __forceinline__ float thread_max(const T* __restrict__ row,
                                            const int64_t vocab_size) {
  float m = -INFINITY;
  if constexpr (std::is_same_v<T, float>) {
    if (vec4_ok(row, vocab_size)) {
      const int64_t n4 = vocab_size >> 2;
      const int64_t stride = 8 * blockDim.x;
      int64_t i = threadIdx.x;
      for (; i + 7 * blockDim.x < n4; i += stride) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const float4 v = load4f(row + ((i + u * blockDim.x) << 2));
          m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
      }
      for (; i < n4; i += blockDim.x) {
        const float4 v = load4f(row + (i << 2));
        m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
      }
      return m;
    }
  }
  for (int64_t i = threadIdx.x; i < vocab_size; i += blockDim.x)
    m = fmaxf(m, (float)row[i]);
  return m;
}

// row[i] = -inf where row[i] < threshold.
template <typename T>
__device__ __forceinline__ void thread_mask_below(T* __restrict__ row,
                                                  const int64_t vocab_size,
                                                  const float threshold) {
  if constexpr (std::is_same_v<T, float>) {
    if (vec4_ok(row, vocab_size)) {
      const int64_t n4 = vocab_size >> 2;
      for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = load4f(row + (i << 2));
        if (v.x < threshold) v.x = -INFINITY;
        if (v.y < threshold) v.y = -INFINITY;
        if (v.z < threshold) v.z = -INFINITY;
        if (v.w < threshold) v.w = -INFINITY;
        store4f(row + (i << 2), v);
      }
      return;
    }
  }
  for (int64_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    if ((float)row[i] < threshold) row[i] = (T)(-INFINITY);
  }
}

// ---------------------------------------------------------------------------
// _temperature_kernel
//   grid = (num_tokens, cdiv(vocab_size, 8192)), BLOCK_SIZE = 8192
// ---------------------------------------------------------------------------
template <typename T>
__global__ void temperature_kernel(
    T* __restrict__ logits, const int64_t logits_stride,
    const int32_t* __restrict__ expanded_idx_mapping,
    const float* __restrict__ temperature_ptr, const int64_t vocab_size) {
  const int64_t token_idx = blockIdx.x;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const float temperature = temperature_ptr[req_state_idx];
  if (temperature == 0.0f || temperature == 1.0f) return;

  const int64_t begin = (int64_t)blockIdx.y * 8192;
  const int64_t end = min(begin + 8192, vocab_size);
  T* __restrict__ row = logits + token_idx * logits_stride;
  const int64_t count = end - begin;
  if constexpr (std::is_same_v<T, float>) {
    if (vec4_ok(row + begin, count)) {
      const int64_t n4 = count >> 2;
      for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = load4f(row + begin + (i << 2));
        v.x /= temperature;
        v.y /= temperature;
        v.z /= temperature;
        v.w /= temperature;
        store4f(row + begin + (i << 2), v);
      }
      return;
    }
  }
  for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    row[i] = (T)((float)row[i] / temperature);
  }
}

// ---------------------------------------------------------------------------
// _gumbel_sample_kernel
//   grid = (num_tokens, cdiv(vocab_size, 1024)), BLOCK_SIZE = 1024
// One program reduces one 1024-wide logits block to (argmax token id, value).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gumbel_sample_kernel(
    int64_t* __restrict__ local_argmax, const int64_t local_argmax_stride,
    float* __restrict__ local_max, const int64_t local_max_stride,
    T* __restrict__ logits_cache, const int64_t cache_stride_0,
    const int64_t cache_stride_1, const int32_t* __restrict__ cache_col,
    const bool has_cache, const bool per_token_col,
    const T* __restrict__ logits, const int64_t logits_stride,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int64_t* __restrict__ seeds_ptr, const int64_t* __restrict__ pos_ptr,
    const float* __restrict__ temperature_ptr, const int64_t vocab_size,
    const bool is_drafting, const bool apply_temperature) {
  const int64_t token_idx = blockIdx.x;
  const int64_t begin = (int64_t)blockIdx.y * kGumbelBlock;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const bool is_valid_req = req_state_idx >= 0;
  const float temperature =
      is_valid_req ? temperature_ptr[req_state_idx] : 0.0f;

  const T* __restrict__ row = logits + token_idx * logits_stride;

  if (has_cache && is_valid_req) {
    const int32_t col = per_token_col ? cache_col[token_idx] : cache_col[0];
    T* __restrict__ dst = logits_cache +
                          (int64_t)req_state_idx * cache_stride_0 +
                          (int64_t)col * cache_stride_1;
    for (int64_t i = begin + threadIdx.x;
         i < min(begin + kGumbelBlock, vocab_size); i += blockDim.x) {
      dst[i] = row[i];
    }
  }

  const int64_t seed = is_valid_req ? seeds_ptr[req_state_idx] : 0;
  int64_t pos = pos_ptr[token_idx];
  if (is_drafting) pos += (int64_t)1 << 30;

  // 8 elements per thread: Triton's #blocked layout groups them as
  // {4t..4t+3, 512+4t..512+4t+3}; max/argmax are layout independent.
  float val[8];
  int32_t idx[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int64_t j =
        (k < 4) ? (4 * threadIdx.x + k) : (512 + 4 * threadIdx.x + (k - 4));
    const int64_t global = begin + j;
    idx[k] = (int32_t)global;
    if (global < vocab_size) {
      float v = (float)row[global];
      if (temperature != 0.0f && apply_temperature) v = v / temperature;
      if (temperature != 0.0f) {
        const float u =
            murmur3_uniform32(seed, pos, (uint32_t)(uint64_t)global);
        v = v + (-tl_log(-tl_log1p_neg_stable(u)));
      }
      val[k] = v;
    } else {
      val[k] = -INFINITY;
    }
  }

  // tl.max(..., return_indices=True) breaks ties toward the lower index.
  float best = -INFINITY;
  int32_t best_idx = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (val[k] > best) {
      best = val[k];
      best_idx = idx[k];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffffffffffULL, best, off);
    const int32_t other_idx =
        __shfl_xor_sync(0xffffffffffffffffULL, best_idx, off);
    if (other > best || (other == best && other_idx < best_idx)) {
      best = other;
      best_idx = other_idx;
    }
  }

  __shared__ float smem_val[4];
  __shared__ int32_t smem_idx[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    smem_val[warp] = best;
    smem_idx[warp] = best_idx;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv = smem_val[0];
    int32_t bi = smem_idx[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) {
      if (smem_val[w] > bv || (smem_val[w] == bv && smem_idx[w] < bi)) {
        bv = smem_val[w];
        bi = smem_idx[w];
      }
    }
    local_argmax[token_idx * local_argmax_stride + blockIdx.y] = (int64_t)bi;
    local_max[token_idx * local_max_stride + blockIdx.y] = bv;
  }
}

// ---------------------------------------------------------------------------
// _penalties_kernel
//   grid = (num_tokens, cdiv(vocab_size, 8192)), BLOCK_SIZE = 8192
// ---------------------------------------------------------------------------
template <typename T>
__global__ void penalties_kernel(
    T* __restrict__ logits, const int64_t logits_stride,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ expanded_local_pos,
    const float* __restrict__ repetition_penalty_ptr,
    const float* __restrict__ frequency_penalty_ptr,
    const float* __restrict__ presence_penalty_ptr,
    const int32_t* __restrict__ prompt_bin_mask,
    const int64_t prompt_bin_mask_stride,
    const int32_t* __restrict__ output_bin_counts,
    const int64_t output_bin_counts_stride, const int64_t vocab_size) {
  const int64_t token_idx = blockIdx.x;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const float rep_penalty = repetition_penalty_ptr[req_state_idx];
  const float freq_penalty = frequency_penalty_ptr[req_state_idx];
  const float pres_penalty = presence_penalty_ptr[req_state_idx];

  const bool use_rep_penalty = rep_penalty != 1.0f;
  const bool use_freq_penalty = freq_penalty != 0.0f;
  const bool use_pres_penalty = pres_penalty != 0.0f;
  if (!use_rep_penalty && !use_freq_penalty && !use_pres_penalty) return;

  const int32_t pos = expanded_local_pos[token_idx];
  const int64_t start_idx = token_idx - pos;
  const int32_t* __restrict__ bin_mask =
      prompt_bin_mask + (int64_t)req_state_idx * prompt_bin_mask_stride;
  const int32_t* __restrict__ base_counts =
      output_bin_counts + (int64_t)req_state_idx * output_bin_counts_stride;

  T* __restrict__ row = logits + token_idx * logits_stride;
  const int64_t begin = (int64_t)blockIdx.y * 8192;
  const int64_t end = min(begin + 8192, vocab_size);

  auto penalise = [&](float value, const int64_t i) -> float {
    int32_t count = base_counts[i];
    for (int32_t p = 0; p < pos; ++p) {
      const int32_t prev_token = token_ids[start_idx + p + 1];
      if ((int32_t)i == prev_token) count += 1;
    }
    const bool bin_mask_set = count > 0;
    if (use_rep_penalty) {
      const int32_t word = (int32_t)i >> 5;
      const int32_t bit = (int32_t)i & 31;
      const bool in_prompt = ((bin_mask[word] >> bit) & 1) != 0;
      const float scale = (in_prompt || bin_mask_set) ? rep_penalty : 1.0f;
      value *= (value > 0.0f) ? (1.0f / scale) : scale;
    }
    if (use_freq_penalty) value = __fmaf_rn(-freq_penalty, (float)count, value);
    if (use_pres_penalty) value -= bin_mask_set ? pres_penalty : 0.0f;
    return value;
  };

  if constexpr (std::is_same_v<T, float>) {
    if (vec4_ok(row + begin, end - begin)) {
      const int64_t n4 = (end - begin) >> 2;
      for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const int64_t base = begin + (i << 2);
        float4 v = load4f(row + base);
        v.x = penalise(v.x, base + 0);
        v.y = penalise(v.y, base + 1);
        v.z = penalise(v.z, base + 2);
        v.w = penalise(v.w, base + 3);
        store4f(row + base, v);
      }
      return;
    }
  }
  for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    row[i] = (T)penalise((float)row[i], i);
  }
}

// ---------------------------------------------------------------------------
// _bincount_kernel
//   grid = (num_tokens, cdiv(max_prefill_len, 1024)), BLOCK_SIZE = 1024
// ---------------------------------------------------------------------------
__global__ void bincount_kernel(
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ all_token_ids,
    const int64_t all_token_ids_stride,
    const int32_t* __restrict__ prompt_len_ptr,
    const int32_t* __restrict__ prefill_len_ptr,
    int32_t* __restrict__ prompt_bin_mask, const int64_t prompt_bin_mask_stride,
    int32_t* __restrict__ output_bin_counts,
    const int64_t output_bin_counts_stride) {
  const int32_t token_idx = blockIdx.x;
  const int64_t begin = (int64_t)blockIdx.y * 1024;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const int32_t prefill_len = prefill_len_ptr[req_state_idx];
  if (begin >= prefill_len) return;

  const int32_t prompt_len = prompt_len_ptr[req_state_idx];
  const int32_t* __restrict__ row =
      all_token_ids + (int64_t)req_state_idx * all_token_ids_stride;

  if (begin < prompt_len) {
    const int64_t n = min((int64_t)1024, (int64_t)prompt_len - begin);
    int32_t* __restrict__ bin_mask =
        prompt_bin_mask + (int64_t)req_state_idx * prompt_bin_mask_stride;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
      const int32_t tok = row[begin + j];
      atomicOr((unsigned int*)(bin_mask + (tok >> 5)), 1u << (tok & 31));
    }
  }

  if (begin + 1024 >= (int64_t)prompt_len) {
    const int64_t start = max(begin, (int64_t)prompt_len);
    const int64_t n = min((int64_t)1024, (int64_t)prefill_len - start);
    int32_t* __restrict__ counts =
        output_bin_counts + (int64_t)req_state_idx * output_bin_counts_stride;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
      atomicAdd(counts + row[start + j], 1);
    }
  }
}

// ---------------------------------------------------------------------------
// _min_p_kernel
//   grid = (num_tokens,), BLOCK_SIZE = 1024
// ---------------------------------------------------------------------------
template <typename T>
__global__ void min_p_kernel(T* __restrict__ logits,
                             const int64_t logits_stride,
                             const int32_t* __restrict__ expanded_idx_mapping,
                             const float* __restrict__ min_p_ptr,
                             const int64_t vocab_size) {
  const int64_t token_idx = blockIdx.x;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const float min_p = min_p_ptr[req_state_idx];
  if (min_p == 0.0f) return;

  T* __restrict__ row = logits + token_idx * logits_stride;
  const float max_val = block_max(thread_max(row, vocab_size));
  thread_mask_below(row, vocab_size, max_val + tl_log(min_p));
}

// ---------------------------------------------------------------------------
// _bias_kernel
//   grid = (num_tokens,), BLOCK_SIZE = next_pow2(list widths),
//   LOGITS_BLOCK_SIZE = 8192. VPT = BLOCK_SIZE / blockDim.x register slots per
//   thread hold the saved allowed-token logits.
// ---------------------------------------------------------------------------
template <typename T, int VPT>
__global__ void bias_kernel(
    T* __restrict__ logits, const int64_t logits_stride,
    const int64_t vocab_size, const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ num_allowed_ptr,
    const int32_t* __restrict__ allowed_token_ids,
    const int64_t allowed_token_ids_stride,
    const int32_t* __restrict__ num_logit_bias_ptr,
    const int32_t* __restrict__ bias_token_ids,
    const int64_t bias_token_ids_stride, const float* __restrict__ bias,
    const int64_t bias_stride, const void* __restrict__ pos_ptr,
    const bool pos_is_i64, const int32_t* __restrict__ min_lens_ptr,
    const int32_t* __restrict__ num_stop_token_ids_ptr,
    const uint8_t* __restrict__ restore_when_all_masked_ptr,
    const int32_t* __restrict__ stop_token_ids,
    const int64_t stop_token_ids_stride, const int64_t block_size,
    const bool check_all_masked_rows) {
  const int64_t token_idx = blockIdx.x;
  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  T* __restrict__ row = logits + token_idx * logits_stride;
  const int tid = threadIdx.x;
  const int nthreads = (int)blockDim.x;

  // ---- Allowed token IDs: keep the listed logits, set the rest to -inf. ----
  int32_t num_allowed = num_allowed_ptr[req_state_idx];
  if (num_allowed > 0) {
    if (num_allowed > (int32_t)block_size) num_allowed = (int32_t)block_size;
    T saved[VPT];
    int32_t saved_ids[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int i = tid + k * nthreads;
      if (i < num_allowed) {
        const int32_t id = allowed_token_ids[(int64_t)req_state_idx *
                                                 allowed_token_ids_stride +
                                             i];
        saved_ids[k] = id;
        saved[k] = row[id];
      }
    }
    __syncthreads();
    for (int64_t i = tid; i < vocab_size; i += nthreads)
      row[i] = (T)(-INFINITY);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int i = tid + k * nthreads;
      if (i < num_allowed) row[saved_ids[k]] = saved[k];
    }
  }

  // ---- Logit bias. ----
  int32_t num_logit_bias = num_logit_bias_ptr[req_state_idx];
  if (num_logit_bias > (int32_t)block_size)
    num_logit_bias = (int32_t)block_size;
  for (int i = tid; i < num_logit_bias; i += nthreads) {
    const int32_t id =
        bias_token_ids[(int64_t)req_state_idx * bias_token_ids_stride + i];
    const float b = bias[(int64_t)req_state_idx * bias_stride + i];
    row[id] = (T)((float)row[id] + b);
  }

  // ---- Min tokens. ----
  int32_t num_stop_token_ids = num_stop_token_ids_ptr[req_state_idx];
  if (num_stop_token_ids > (int32_t)block_size)
    num_stop_token_ids = (int32_t)block_size;
  const int64_t pos = pos_is_i64
                          ? ((const int64_t*)pos_ptr)[token_idx]
                          : (int64_t)((const int32_t*)pos_ptr)[token_idx];
  const int32_t min_len = min_lens_ptr[req_state_idx];
  const int32_t* __restrict__ stop_row =
      stop_token_ids + (int64_t)req_state_idx * stop_token_ids_stride;
  if (num_stop_token_ids <= 0 || pos + 1 >= (int64_t)min_len) return;

  const bool should_restore = check_all_masked_rows &&
                              (restore_when_all_masked_ptr[req_state_idx] != 0);
  if (!should_restore) {
    for (int i = tid; i < num_stop_token_ids; i += nthreads)
      row[stop_row[i]] = (T)(-INFINITY);
    return;
  }

  T stop_saved[VPT];
  int32_t stop_ids[VPT];
  bool stop_ok[VPT];
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = tid + k * nthreads;
    stop_ok[k] = false;
    if (i < num_stop_token_ids) {
      const int32_t id = stop_row[i];
      const float v = (float)row[id];
      stop_ids[k] = id;
      stop_saved[k] = row[id];
      stop_ok[k] = (v > -INFINITY) && (v < INFINITY);
    }
  }
  __syncthreads();
  for (int i = tid; i < num_stop_token_ids; i += nthreads)
    row[stop_row[i]] = (T)(-INFINITY);
  __syncthreads();

  float row_max = -INFINITY;
  for (int64_t i = tid; i < vocab_size; i += nthreads)
    row_max = fmaxf(row_max, (float)row[i]);
  row_max = block_max(row_max);
  if (row_max == -INFINITY) {
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int i = tid + k * nthreads;
      if (i < num_stop_token_ids && stop_ok[k])
        row[stop_ids[k]] = stop_saved[k];
    }
  }
}

// ---------------------------------------------------------------------------
// _bad_words_kernel
//   grid = ceil(num_tokens * max_num_bad_words / 256); one thread per program.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void bad_words_kernel(
    T* __restrict__ logits, const int64_t logits_stride,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ bad_word_token_ids,
    const int64_t bad_word_token_ids_stride,
    const int32_t* __restrict__ bad_word_offsets,
    const int64_t bad_word_offsets_stride,
    const int32_t* __restrict__ num_bad_words_ptr,
    const int32_t* __restrict__ all_token_ids,
    const int64_t all_token_ids_stride,
    const int32_t* __restrict__ prompt_len_ptr,
    const int32_t* __restrict__ total_len_ptr,
    const int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ expanded_local_pos,
    const int32_t max_num_bad_words, const int64_t num_tokens) {
  const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t bw_idx = (int32_t)(gid % max_num_bad_words);
  const int64_t token_idx = gid / max_num_bad_words;
  if (token_idx >= num_tokens) return;

  const int32_t req_state_idx = expanded_idx_mapping[token_idx];
  const int32_t num_bad_words = num_bad_words_ptr[req_state_idx];
  if (bw_idx >= num_bad_words) return;

  const int32_t pos = expanded_local_pos[token_idx];
  const int64_t cur_req_first_pos = token_idx - pos;
  const int32_t prompt_len = prompt_len_ptr[req_state_idx];
  const int32_t total_len = total_len_ptr[req_state_idx];
  const int32_t output_len = total_len - prompt_len;
  const int32_t effective_len = output_len + pos;

  const int32_t* __restrict__ offsets_row =
      bad_word_offsets + (int64_t)req_state_idx * bad_word_offsets_stride;
  const int32_t* __restrict__ tokens_row =
      bad_word_token_ids + (int64_t)req_state_idx * bad_word_token_ids_stride;
  const int32_t* __restrict__ output_row =
      all_token_ids + (int64_t)req_state_idx * all_token_ids_stride +
      prompt_len;

  const int32_t start = offsets_row[bw_idx];
  const int32_t end = offsets_row[bw_idx + 1];
  const int32_t prefix_len = (end - start) - 1;
  if (prefix_len > effective_len) return;

  const int32_t last_token = tokens_row[end - 1];
  bool match = true;
  for (int32_t i = 0; i < prefix_len; ++i) {
    const int32_t expected = tokens_row[start + i];
    const int32_t actual_pos = effective_len - prefix_len + i;
    int32_t actual;
    if (actual_pos >= output_len) {
      const int32_t spec_offset = actual_pos - output_len;
      actual = input_ids[cur_req_first_pos + spec_offset + 1];
    } else {
      actual = output_row[actual_pos];
    }
    match = match && (expected == actual);
  }
  if (match) {
    logits[token_idx * logits_stride + last_token] = (T)(-INFINITY);
  }
}

// ---------------------------------------------------------------------------
// _topk_log_softmax_kernel
//   grid = (batch_size,), BLOCK_SIZE = 1024, 128 threads.
// ---------------------------------------------------------------------------
// Triton picks the logits load layout from the axis info of the row pointer and
// the mask: a 16-byte vector (4 fp32 / 8 fp16) when both the row stride and
// vocab_size are multiples of 16 elements, otherwise scalar. The `tl.sum`
// association follows that layout (see the fp32/vec4 case in the header), so
// the per-thread grouping is templated on VEC.
template <typename T, int VEC>
__global__ void topk_log_softmax_kernel(
    float* __restrict__ output, const T* __restrict__ logits,
    const int64_t logits_stride, const int64_t* __restrict__ topk_ids,
    const int64_t topk_ids_stride, const int64_t topk, const int64_t vocab_size,
    const int64_t topk_block_size) {
  const int64_t req_idx = blockIdx.x;
  const T* __restrict__ row = logits + req_idx * logits_stride;

  const float max_val = reduce_block_max_1024(thread_max(row, vocab_size));

  float se = 0.0f;
  for (int64_t begin = 0; begin < vocab_size; begin += kTopkBlock) {
    const int64_t limit = min(begin + kTopkBlock, vocab_size);
    float e[8];
    // Triton's blocked layout: thread t owns, for rep r in [0, 8/VEC),
    // elements [r * 128 * VEC + t * VEC, + VEC) in register order.
    bool vectorised = false;
    if constexpr (std::is_same_v<T, float>) {
      if (VEC == 4 && begin + 4 * threadIdx.x < limit) {
#pragma unroll
        for (int g = 0; g < 8; g += 4) {
          const int64_t base = begin + (g / 4) * 512 + 4 * threadIdx.x;
          const float4 v = load4f(row + base);
          e[g + 0] = tl_exp(v.x - max_val);
          e[g + 1] = tl_exp(v.y - max_val);
          e[g + 2] = tl_exp(v.z - max_val);
          e[g + 3] = tl_exp(v.w - max_val);
        }
        vectorised = true;
      }
    }
    if (!vectorised) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t j =
            (k / VEC) * (128 * VEC) + threadIdx.x * VEC + (k % VEC);
        if (begin + j < limit) {
          e[k] = tl_exp((float)row[begin + j] - max_val);
        } else {
          e[k] = 0.0f;
        }
      }
    }
    // Adjacent-pair tree over the thread's registers.
    const float a0 = e[0] + e[1];
    const float a1 = e[2] + e[3];
    const float a2 = e[4] + e[5];
    const float a3 = e[6] + e[7];
    const float partial = (a0 + a1) + (a2 + a3);
    se += reduce_block_sum_1024(partial, (int)((begin / kTopkBlock) & 1));
  }
  const float lse = tl_log(se);

  (void)topk_block_size;
  for (int64_t k = threadIdx.x; k < topk; k += blockDim.x) {
    const int64_t token_id = topk_ids[req_idx * topk_ids_stride + k];
    const float value = (float)row[token_id];
    output[req_idx * topk + k] = (value - max_val) - lse;
  }
}

// ---------------------------------------------------------------------------
// _ranks_kernel
//   grid = (batch_size,), BLOCK_SIZE = 8192. Integer counts: the reduction
//   order cannot change the result.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void ranks_kernel(int64_t* __restrict__ output,
                             const T* __restrict__ logits,
                             const int64_t logits_stride,
                             const int64_t* __restrict__ token_ids,
                             const int64_t vocab_size) {
  const int64_t req_idx = blockIdx.x;
  const T* __restrict__ row = logits + req_idx * logits_stride;
  const T x = row[token_ids[req_idx]];

  int32_t n = 0;
  if constexpr (std::is_same_v<T, float>) {
    if (vec4_ok(row, vocab_size)) {
      const float xf = (float)x;
      const int64_t n4 = vocab_size >> 2;
      // 8 independent vec4 loads per iteration (8192 elements per block
      // iteration, as in Triton's 64-elements-per-thread loop): the row is
      // latency bound, not ALU bound.
      const int64_t stride = 8 * blockDim.x;
      int64_t i = threadIdx.x;
      for (; i + 7 * blockDim.x < n4; i += stride) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const float4 v = load4f(row + ((i + u * blockDim.x) << 2));
          n += (v.x >= xf) + (v.y >= xf) + (v.z >= xf) + (v.w >= xf);
        }
      }
      for (; i < n4; i += blockDim.x) {
        const float4 v = load4f(row + (i << 2));
        n += (v.x >= xf) + (v.y >= xf) + (v.z >= xf) + (v.w >= xf);
      }
      output[req_idx] = (int64_t)block_sum_int(n);
      return;
    }
  }
  for (int64_t i = threadIdx.x; i < vocab_size; i += blockDim.x)
    n += ((float)row[i] >= (float)x) ? 1 : 0;
  output[req_idx] = (int64_t)block_sum_int(n);
}

// ---------------------------------------------------------------------------
// _prompt_logprobs_token_ids_kernel
//   grid = (num_reqs,), BLOCK_SIZE = 1024
// ---------------------------------------------------------------------------
__global__ void prompt_logprobs_token_ids_kernel(
    int64_t* __restrict__ prompt_logprobs_token_ids,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ num_computed_tokens,
    const int32_t* __restrict__ all_token_ids,
    const int64_t all_token_ids_stride) {
  const int32_t batch_idx = blockIdx.x;
  const int32_t req_state_idx = idx_mapping[batch_idx];
  const int32_t query_start = query_start_loc[batch_idx];
  const int32_t query_len = query_start_loc[batch_idx + 1] - query_start;
  const int32_t num_computed = num_computed_tokens[req_state_idx];
  const int32_t* __restrict__ row =
      all_token_ids + (int64_t)req_state_idx * all_token_ids_stride;
  for (int32_t i = threadIdx.x; i < query_len; i += blockDim.x) {
    prompt_logprobs_token_ids[query_start + i] =
        (int64_t)row[num_computed + 1 + i];
  }
}

inline bool supported_logits_dtype(const torch::Tensor& t) {
  const auto dt = t.scalar_type();
  return dt == at::kFloat || dt == at::kHalf || dt == at::kBFloat16;
}

// The draft head's logits are bf16 in the real serving path, so every logits
// kernel is instantiated for fp32/fp16/bf16.
template <typename F>
void dispatch_logits_dtype(const torch::Tensor& logits, F&& f) {
  switch (logits.scalar_type()) {
    case at::kFloat:
      f.template operator()<float>();
      break;
    case at::kHalf:
      f.template operator()<at::Half>();
      break;
    default:
      f.template operator()<at::BFloat16>();
      break;
  }
}

inline void check_device(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, ": tensor must be on HIP");
}

}  // namespace rdna2_sample
}  // namespace vllm

// Host wrappers stay at global scope so the torch_bindings importer resolves
// them (same pattern as v1_worker_rdna2.cu / v1_input_batch_rdna2.cu).
void temperature_rdna2(torch::Tensor logits, int64_t logits_stride,
                       torch::Tensor expanded_idx_mapping,
                       torch::Tensor temperature, int64_t vocab_size) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "temperature_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "temperature_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(expanded_idx_mapping.scalar_type() == at::kInt,
              "temperature_rdna2: expanded_idx_mapping must be int32");
  const int64_t num_tokens = logits.size(0);
  const int64_t num_blocks = (vocab_size + 8191) / 8192;
  if (num_tokens == 0 || num_blocks == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_tokens, (unsigned)num_blocks);
  dispatch_logits_dtype(logits, [&]<typename T>() {
    temperature_kernel<T><<<grid, kThreads, 0, stream.stream()>>>(
        reinterpret_cast<T*>(logits.data_ptr()), logits_stride,
        reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
        reinterpret_cast<const float*>(temperature.data_ptr()), vocab_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gumbel_sample_rdna2(
    torch::Tensor local_argmax, int64_t local_argmax_stride,
    torch::Tensor local_max, int64_t local_max_stride,
    std::optional<torch::Tensor> logits_cache, int64_t logits_cache_stride_0,
    int64_t logits_cache_stride_1,
    std::optional<torch::Tensor> logits_cache_col, torch::Tensor logits,
    int64_t logits_stride, torch::Tensor expanded_idx_mapping,
    torch::Tensor seeds, torch::Tensor pos, torch::Tensor temperature,
    int64_t vocab_size, bool is_drafting, bool apply_temperature, bool use_fp64,
    bool per_token_col) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "gumbel_sample_rdna2");
  TORCH_CHECK(!use_fp64, "gumbel_sample_rdna2: fp64 path is Triton-only");
  TORCH_CHECK(supported_logits_dtype(logits),
              "gumbel_sample_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(local_max.scalar_type() == at::kFloat,
              "gumbel_sample_rdna2: local_max must be fp32");
  TORCH_CHECK(local_argmax.scalar_type() == at::kLong,
              "gumbel_sample_rdna2: local_argmax must be int64");
  TORCH_CHECK(pos.scalar_type() == at::kLong,
              "gumbel_sample_rdna2: pos must be int64");
  TORCH_CHECK(seeds.scalar_type() == at::kLong,
              "gumbel_sample_rdna2: seeds must be int64");
  const int64_t num_tokens = logits.size(0);
  const int64_t num_blocks = (vocab_size + kGumbelBlock - 1) / kGumbelBlock;
  if (num_tokens == 0 || num_blocks == 0) return;
  const bool has_cache = logits_cache.has_value() && logits_cache->defined();
  if (has_cache) {
    TORCH_CHECK(logits_cache->scalar_type() == logits.scalar_type(),
                "gumbel_sample_rdna2: logits_cache dtype mismatch");
    TORCH_CHECK(logits_cache_col.has_value() && logits_cache_col->defined(),
                "gumbel_sample_rdna2: logits_cache_col required with a cache");
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_tokens, (unsigned)num_blocks);
  auto* argmax_ptr = reinterpret_cast<int64_t*>(local_argmax.data_ptr());
  auto* max_ptr = reinterpret_cast<float*>(local_max.data_ptr());
  auto* mapping_ptr =
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr());
  auto* seeds_ptr = reinterpret_cast<const int64_t*>(seeds.data_ptr());
  auto* pos_ptr = reinterpret_cast<const int64_t*>(pos.data_ptr());
  auto* temp_ptr = reinterpret_cast<const float*>(temperature.data_ptr());
  // The column tensor is required whenever a cache is passed; for the shared
  // column case it is a 0-dim device tensor (never a host pointer).
  const int32_t* col_ptr = nullptr;
  if (has_cache)
    col_ptr = reinterpret_cast<const int32_t*>(logits_cache_col->data_ptr());

  dispatch_logits_dtype(logits, [&]<typename T>() {
    auto* cache_ptr =
        has_cache ? reinterpret_cast<T*>(logits_cache->data_ptr()) : nullptr;
    gumbel_sample_kernel<T><<<grid, kSampleThreads, 0, stream.stream()>>>(
        argmax_ptr, local_argmax_stride, max_ptr, local_max_stride, cache_ptr,
        logits_cache_stride_0, logits_cache_stride_1, col_ptr, has_cache,
        per_token_col, reinterpret_cast<const T*>(logits.data_ptr()),
        logits_stride, mapping_ptr, seeds_ptr, pos_ptr, temp_ptr, vocab_size,
        is_drafting, apply_temperature);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void penalties_rdna2(torch::Tensor logits, int64_t logits_stride,
                     torch::Tensor expanded_idx_mapping,
                     torch::Tensor token_ids, torch::Tensor expanded_local_pos,
                     torch::Tensor repetition_penalty,
                     torch::Tensor frequency_penalty,
                     torch::Tensor presence_penalty,
                     torch::Tensor prompt_bin_mask,
                     int64_t prompt_bin_mask_stride,
                     torch::Tensor output_bin_counts,
                     int64_t output_bin_counts_stride, int64_t vocab_size) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "penalties_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "penalties_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(token_ids.scalar_type() == at::kInt,
              "penalties_rdna2: token_ids must be int32");
  TORCH_CHECK(prompt_bin_mask.scalar_type() == at::kInt &&
                  output_bin_counts.scalar_type() == at::kInt,
              "penalties_rdna2: bin tensors must be int32");
  const int64_t num_tokens = logits.size(0);
  const int64_t num_blocks = (vocab_size + 8191) / 8192;
  if (num_tokens == 0 || num_blocks == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)num_tokens, (unsigned)num_blocks);
  dispatch_logits_dtype(logits, [&]<typename T>() {
    penalties_kernel<T><<<grid, kThreads, 0, stream.stream()>>>(
        reinterpret_cast<T*>(logits.data_ptr()), logits_stride,
        reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
        reinterpret_cast<const int32_t*>(token_ids.data_ptr()),
        reinterpret_cast<const int32_t*>(expanded_local_pos.data_ptr()),
        reinterpret_cast<const float*>(repetition_penalty.data_ptr()),
        reinterpret_cast<const float*>(frequency_penalty.data_ptr()),
        reinterpret_cast<const float*>(presence_penalty.data_ptr()),
        reinterpret_cast<const int32_t*>(prompt_bin_mask.data_ptr()),
        prompt_bin_mask_stride,
        reinterpret_cast<const int32_t*>(output_bin_counts.data_ptr()),
        output_bin_counts_stride, vocab_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bincount_rdna2(torch::Tensor expanded_idx_mapping,
                    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
                    torch::Tensor prompt_len, torch::Tensor prefill_len,
                    torch::Tensor prompt_bin_mask,
                    int64_t prompt_bin_mask_stride,
                    torch::Tensor output_bin_counts,
                    int64_t output_bin_counts_stride, int64_t max_prefill_len) {
  using namespace vllm::rdna2_sample;
  const int64_t num_tokens = expanded_idx_mapping.numel();
  const int64_t num_blocks = (max_prefill_len + 1023) / 1024;
  if (num_tokens == 0 || num_blocks == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(expanded_idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)num_tokens, (unsigned)num_blocks);
  bincount_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(all_token_ids.data_ptr()),
      all_token_ids_stride,
      reinterpret_cast<const int32_t*>(prompt_len.data_ptr()),
      reinterpret_cast<const int32_t*>(prefill_len.data_ptr()),
      reinterpret_cast<int32_t*>(prompt_bin_mask.data_ptr()),
      prompt_bin_mask_stride,
      reinterpret_cast<int32_t*>(output_bin_counts.data_ptr()),
      output_bin_counts_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void min_p_rdna2(torch::Tensor logits, int64_t logits_stride,
                 torch::Tensor expanded_idx_mapping, torch::Tensor min_p,
                 int64_t vocab_size) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "min_p_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "min_p_rdna2: logits must be fp32/fp16/bf16");
  const int64_t num_tokens = logits.size(0);
  if (num_tokens == 0 || vocab_size == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_logits_dtype(logits, [&]<typename T>() {
    min_p_kernel<T><<<(unsigned)num_tokens, kThreads, 0, stream.stream()>>>(
        reinterpret_cast<T*>(logits.data_ptr()), logits_stride,
        reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
        reinterpret_cast<const float*>(min_p.data_ptr()), vocab_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

constexpr int64_t kBiasMaxBlock = 4096;

template <typename T>
void launch_bias(torch::Tensor& logits, int64_t logits_stride,
                 int64_t vocab_size, const torch::Tensor& expanded_idx_mapping,
                 const torch::Tensor& num_allowed,
                 const torch::Tensor& allowed_ids, int64_t allowed_ids_stride,
                 const torch::Tensor& num_bias, const torch::Tensor& bias_ids,
                 int64_t bias_ids_stride, const torch::Tensor& bias,
                 int64_t bias_stride, const torch::Tensor& pos,
                 const torch::Tensor& min_lens, const torch::Tensor& num_stop,
                 const torch::Tensor& restore, const torch::Tensor& stop_ids,
                 int64_t stop_ids_stride, int64_t block_size,
                 bool check_all_masked, bool pos_is_i64, hipStream_t stream) {
  const int64_t num_tokens = logits.size(0);
  const unsigned grid = (unsigned)num_tokens;
  const int vpt = (int)((block_size + vllm::rdna2_sample::kThreads - 1) /
                        vllm::rdna2_sample::kThreads);
  auto* logits_ptr = reinterpret_cast<T*>(logits.data_ptr());
  auto* mapping_ptr =
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr());
  auto* num_allowed_ptr =
      reinterpret_cast<const int32_t*>(num_allowed.data_ptr());
  auto* allowed_ptr = reinterpret_cast<const int32_t*>(allowed_ids.data_ptr());
  auto* num_bias_ptr = reinterpret_cast<const int32_t*>(num_bias.data_ptr());
  auto* bias_ids_ptr = reinterpret_cast<const int32_t*>(bias_ids.data_ptr());
  auto* bias_ptr = reinterpret_cast<const float*>(bias.data_ptr());
  const void* pos_ptr = pos.data_ptr();
  auto* min_lens_ptr = reinterpret_cast<const int32_t*>(min_lens.data_ptr());
  auto* num_stop_ptr = reinterpret_cast<const int32_t*>(num_stop.data_ptr());
  auto* restore_ptr = reinterpret_cast<const uint8_t*>(restore.data_ptr());
  auto* stop_ids_ptr = reinterpret_cast<const int32_t*>(stop_ids.data_ptr());
#define VLLM_RDNA2_BIAS_CASE(VPT)                                              \
  case VPT:                                                                    \
    vllm::rdna2_sample::bias_kernel<T, VPT>                                    \
        <<<grid, vllm::rdna2_sample::kThreads, 0, stream>>>(                   \
            logits_ptr, logits_stride, vocab_size, mapping_ptr,                \
            num_allowed_ptr, allowed_ptr, allowed_ids_stride, num_bias_ptr,    \
            bias_ids_ptr, bias_ids_stride, bias_ptr, bias_stride, pos_ptr,     \
            pos_is_i64, min_lens_ptr, num_stop_ptr, restore_ptr, stop_ids_ptr, \
            stop_ids_stride, block_size, check_all_masked);                    \
    break;
  switch (vpt) {
    VLLM_RDNA2_BIAS_CASE(1)
    VLLM_RDNA2_BIAS_CASE(2)
    VLLM_RDNA2_BIAS_CASE(4)
    VLLM_RDNA2_BIAS_CASE(8)
    VLLM_RDNA2_BIAS_CASE(16)
    default:
      TORCH_CHECK(false, "logit_bias_rdna2: unsupported block size ",
                  block_size);
  }
#undef VLLM_RDNA2_BIAS_CASE
}

}  // namespace

void logit_bias_rdna2(torch::Tensor logits, int64_t logits_stride,
                      int64_t vocab_size, torch::Tensor expanded_idx_mapping,
                      torch::Tensor num_allowed, torch::Tensor allowed_ids,
                      int64_t allowed_ids_stride, torch::Tensor num_bias,
                      torch::Tensor bias_ids, int64_t bias_ids_stride,
                      torch::Tensor bias, int64_t bias_stride,
                      torch::Tensor pos, torch::Tensor min_lens,
                      torch::Tensor num_stop,
                      torch::Tensor restore_when_all_masked,
                      torch::Tensor stop_ids, int64_t stop_ids_stride,
                      int64_t block_size, bool check_all_masked) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "logit_bias_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "logit_bias_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt,
              "logit_bias_rdna2: pos must be int32/int64");
  TORCH_CHECK(block_size > 0 && block_size <= kBiasMaxBlock,
              "logit_bias_rdna2: unsupported block size ", block_size);
  const int64_t num_tokens = logits.size(0);
  if (num_tokens == 0 || vocab_size == 0) return;
  const bool pos_is_i64 = pos.scalar_type() == at::kLong;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_logits_dtype(logits, [&]<typename T>() {
    launch_bias<T>(logits, logits_stride, vocab_size, expanded_idx_mapping,
                   num_allowed, allowed_ids, allowed_ids_stride, num_bias,
                   bias_ids, bias_ids_stride, bias, bias_stride, pos, min_lens,
                   num_stop, restore_when_all_masked, stop_ids, stop_ids_stride,
                   block_size, check_all_masked, pos_is_i64, stream.stream());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bad_words_rdna2(
    torch::Tensor logits, int64_t logits_stride,
    torch::Tensor expanded_idx_mapping, torch::Tensor bad_word_token_ids,
    int64_t bad_word_token_ids_stride, torch::Tensor bad_word_offsets,
    int64_t bad_word_offsets_stride, torch::Tensor num_bad_words,
    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor prompt_len, torch::Tensor total_len, torch::Tensor input_ids,
    torch::Tensor expanded_local_pos, int64_t max_num_bad_words) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "bad_words_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "bad_words_rdna2: logits must be fp32/fp16/bf16");
  const int64_t num_tokens = logits.size(0);
  if (num_tokens == 0 || max_num_bad_words <= 0) return;
  const int64_t total = num_tokens * max_num_bad_words;
  const unsigned grid = (unsigned)((total + kThreads - 1) / kThreads);
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_logits_dtype(logits, [&]<typename T>() {
    bad_words_kernel<T><<<grid, kThreads, 0, stream.stream()>>>(
        reinterpret_cast<T*>(logits.data_ptr()), logits_stride,
        reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
        reinterpret_cast<const int32_t*>(bad_word_token_ids.data_ptr()),
        bad_word_token_ids_stride,
        reinterpret_cast<const int32_t*>(bad_word_offsets.data_ptr()),
        bad_word_offsets_stride,
        reinterpret_cast<const int32_t*>(num_bad_words.data_ptr()),
        reinterpret_cast<const int32_t*>(all_token_ids.data_ptr()),
        all_token_ids_stride,
        reinterpret_cast<const int32_t*>(prompt_len.data_ptr()),
        reinterpret_cast<const int32_t*>(total_len.data_ptr()),
        reinterpret_cast<const int32_t*>(input_ids.data_ptr()),
        reinterpret_cast<const int32_t*>(expanded_local_pos.data_ptr()),
        (int32_t)max_num_bad_words, num_tokens);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void topk_log_softmax_rdna2(torch::Tensor output, torch::Tensor logits,
                            int64_t logits_stride, torch::Tensor topk_ids,
                            int64_t topk_ids_stride, int64_t topk,
                            int64_t vocab_size, int64_t topk_block_size) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "topk_log_softmax_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "topk_log_softmax_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(output.scalar_type() == at::kFloat,
              "topk_log_softmax_rdna2: output must be fp32");
  const int64_t batch_size = logits.size(0);
  if (batch_size == 0 || topk == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* out_ptr = reinterpret_cast<float*>(output.data_ptr());
  auto* ids_ptr = reinterpret_cast<const int64_t*>(topk_ids.data_ptr());
  // 16-byte vectorization requires both the row stride and the mask boundary
  // to be 16-element aligned (Triton's AxisInfo); otherwise the load layout is
  // scalar and the reduction association changes.
  const bool vectorized = (vocab_size % 16 == 0 && logits_stride % 16 == 0);
  const dim3 grid((unsigned)batch_size);
  dispatch_logits_dtype(logits, [&]<typename T>() {
    constexpr int VEC = sizeof(T) == 4 ? 4 : 8;
    if (vectorized) {
      topk_log_softmax_kernel<T, VEC>
          <<<grid, kSampleThreads, 0, stream.stream()>>>(
              out_ptr, reinterpret_cast<const T*>(logits.data_ptr()),
              logits_stride, ids_ptr, topk_ids_stride, topk, vocab_size,
              topk_block_size);
    } else {
      topk_log_softmax_kernel<T, 1>
          <<<grid, kSampleThreads, 0, stream.stream()>>>(
              out_ptr, reinterpret_cast<const T*>(logits.data_ptr()),
              logits_stride, ids_ptr, topk_ids_stride, topk, vocab_size,
              topk_block_size);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void token_ranks_rdna2(torch::Tensor token_ranks, torch::Tensor logits,
                       int64_t logits_stride, torch::Tensor sampled_token_ids,
                       int64_t vocab_size) {
  using namespace vllm::rdna2_sample;
  check_device(logits, "token_ranks_rdna2");
  TORCH_CHECK(supported_logits_dtype(logits),
              "token_ranks_rdna2: logits must be fp32/fp16/bf16");
  TORCH_CHECK(token_ranks.scalar_type() == at::kLong,
              "token_ranks_rdna2: token_ranks must be int64");
  TORCH_CHECK(sampled_token_ids.scalar_type() == at::kLong,
              "token_ranks_rdna2: sampled_token_ids must be int64");
  const int64_t batch_size = logits.size(0);
  if (batch_size == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* out_ptr = reinterpret_cast<int64_t*>(token_ranks.data_ptr());
  auto* ids_ptr =
      reinterpret_cast<const int64_t*>(sampled_token_ids.data_ptr());
  dispatch_logits_dtype(logits, [&]<typename T>() {
    ranks_kernel<T><<<(unsigned)batch_size, kThreads, 0, stream.stream()>>>(
        out_ptr, reinterpret_cast<const T*>(logits.data_ptr()), logits_stride,
        ids_ptr, vocab_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prompt_logprobs_token_ids_rdna2(torch::Tensor prompt_logprobs_token_ids,
                                     torch::Tensor query_start_loc,
                                     torch::Tensor idx_mapping,
                                     torch::Tensor num_computed_tokens,
                                     torch::Tensor all_token_ids,
                                     int64_t all_token_ids_stride) {
  using namespace vllm::rdna2_sample;
  const int64_t num_reqs = idx_mapping.numel();
  if (num_reqs == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  prompt_logprobs_token_ids_kernel<<<(unsigned)num_reqs, kThreads, 0,
                                     stream.stream()>>>(
      reinterpret_cast<int64_t*>(prompt_logprobs_token_ids.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(num_computed_tokens.data_ptr()),
      reinterpret_cast<const int32_t*>(all_token_ids.data_ptr()),
      all_token_ids_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
