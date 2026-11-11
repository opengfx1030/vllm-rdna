// HIP replacements for the gfx1030 (RDNA2) speculative-decoding Triton kernels:
//
//   gpu/spec_decode/autoregressive/speculator.py::_prepare_prefill_inputs_kernel
//   gpu/spec_decode/autoregressive/speculator.py::_prepare_decode_inputs_kernel
//   gpu/spec_decode/autoregressive/speculator.py::_update_draft_inputs_kernel
//   gpu/spec_decode/rejection_sampler.py::_gather_draft_sampled_kernel
//   gpu/spec_decode/rejection_sampler.py::_flatten_sampled_kernel
//   gpu/spec_decode/rejection_sampler_utils.py::_compute_local_logits_stats_kernel
//   gpu/spec_decode/rejection_sampler_utils.py::_rejection_kernel
//   gpu/spec_decode/rejection_sampler_utils.py::_resample_kernel
//   gpu/spec_decode/rejection_sampler_utils.py::_insert_resampled_kernel
//
// All of these run once per decode step inside a FULL cudagraph capture, so the
// host wrappers only launch: no host syncs, no `.item()`, no allocations.
//
// Bit-exactness against the Triton path is the requirement, and float
// reductions make that non-obvious. The device helpers below reproduce
// Triton's AMD lowering exactly:
//   * tl.exp(x) -> v_exp_f32(x * log2e), tl.log(x) -> logf(x)
//     (verified: Triton emits `fmul x, 0x3FB8AA3B` + `llvm.exp2.f32`, and
//     `llvm.log.f32` lowers to the V_LOG_F32 instruction on gfx1030)
//   * block reductions: in-thread balanced tree over the register file (in
//     register order), then a 32-lane butterfly with masks 16,8,4,2,1, then a
//     warp butterfly with masks 2,1 over the per-warp partials. Verified
//     bit-exact on 100 random 8192-element sums and 200 random 32-element sums
//     against `tl.sum` (see the simulator used while porting).
//   * tl.rand(seed, offset) is Philox-4x32-10 over (offset_lo, offset_hi, 0, 0)
//     with the seed split 32/32, then uint32 -> uniform fp32.
//
// The watermark variant of `_resample_kernel` (philox_gumbel_block_argmax) is
// NOT ported: the Python dispatcher keeps the Triton kernel for it. It is off
// unless `watermark_config` is set.
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <type_traits>
#include <optional>

namespace vllm {
namespace rdna2_spec_decode {

// ---------------------------------------------------------------------------
// Triton AMD math primitives
// ---------------------------------------------------------------------------
__device__ __forceinline__ float tl_exp(float x) {
  float t = x * 1.4426950408889634f;  // log2(e), the literal Triton uses
  float y;
  asm("v_exp_f32 %0, %1" : "=v"(y) : "v"(t));
  return y;
}

// tl.log is LLVM's `llvm.log.f32` AMDGPU expansion (v_log_f32 gives log2, then
// it is multiplied by ln2 with an FMA refinement; denormals are pre-scaled by
// 2**32 and corrected afterwards). Neither `logf` (ocml) nor a bare
// v_log_f32*ln2 reproduces it: both differ by 1 ulp on O(0.3) of inputs.
// Verified bit-for-bit against `tl.log` on 2048 random values spanning
// 1e-35..1e35 plus 0/-0/+inf/-inf/NaN/denormals. Keep the expression shape:
// clang contracts `e + t * d` into an FMA exactly like the intrinsic lowering,
// and the final `+ t*c` is fused as well: the lowering emits
// `fma(t, c, fma(t, d, e))`, one rounding, not `fma(t, d, e) + fl(t*c)`.
__device__ __forceinline__ float tl_log(float x) {
  const float c = __int_as_float(0x3f317217);  // ln2, truncated
  const float d = __int_as_float(0x3377d1cf);  // ln2 residual
  const bool normal = fabsf(x) >= 0x1p-126f;
  const float t = [&] {
    float y;
    asm("v_log_f32 %0, %1" : "=v"(y) : "v"(ldexpf(x, normal ? 0 : 32)));
    return y;
  }();
  const float r = t * c;
  const float e = __fmaf_rn(t, c, -r);
  const float refined = __fmaf_rn(t, c, __fmaf_rn(t, d, e));
  const float finite = fabsf(t) < __int_as_float(0x7f800000) ? refined : t;
  return finite - (normal ? 0.0f : __int_as_float(0x41b17218));
}

// uint_to_uniform_float: negative int32 -> ~x, else x, times 2**-31.
__device__ __forceinline__ float tl_uint_to_uniform(uint32_t v) {
  int32_t x = (int32_t)v;
  x = x < 0 ? ~x : x;
  return (float)x * 4.6566127342e-10f;
}

__device__ __forceinline__ uint32_t tl_umulhi(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

// Philox-4x32-10 with the key split 32/32 out of the int64 seed, matching
// triton/language/random.py:randint4x + `rand`.
__device__ __forceinline__ float tl_rand32(int64_t seed, int64_t offset) {
  uint32_t c0 = (uint32_t)((uint64_t)offset & 0xFFFFFFFFu);
  uint32_t c1 = (uint32_t)(((uint64_t)offset >> 32) & 0xFFFFFFFFu);
  uint32_t c2 = 0u, c3 = 0u;
  uint32_t k0 = (uint32_t)((uint64_t)seed & 0xFFFFFFFFu);
  uint32_t k1 = (uint32_t)(((uint64_t)seed >> 32) & 0xFFFFFFFFu);
  constexpr uint32_t kKeyA = 0x9E3779B9u, kKeyB = 0xBB67AE85u;
  constexpr uint32_t kRoundA = 0xD2511F53u, kRoundB = 0xCD9E8D57u;
#pragma unroll
  for (int i = 0; i < 10; i++) {
    uint32_t _c0 = c0, _c2 = c2;
    c0 = tl_umulhi(kRoundB, _c2) ^ c1 ^ k0;
    c2 = tl_umulhi(kRoundA, _c0) ^ c3 ^ k1;
    c1 = kRoundB * _c2;
    c3 = kRoundA * _c0;
    k0 += kKeyA;
    k1 += kKeyB;
  }
  return tl_uint_to_uniform(c0);
}

// tl_rand32(..., includes_zero=False): clamp away from zero.
__device__ __forceinline__ float tl_rand32_nozero(int64_t seed, int64_t offset) {
  float u = tl_rand32(seed, offset);
  return u < 4.6566127342e-10f ? 4.6566127342e-10f : u;
}

// ---------------------------------------------------------------------------
// murmur3 used by gumbel_block_argmax (gumbel.py)
// ---------------------------------------------------------------------------
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
                                                  uint32_t offset,
                                                  uint32_t domain = 0u) {
  uint32_t h = (uint32_t)(seed ^ seed);
  h ^= domain;
  h = murmur3_mix(h, (uint32_t)((uint64_t)seed & 0xFFFFFFFFu));
  h = murmur3_mix(h, (uint32_t)(((uint64_t)seed >> 32) & 0xFFFFFFFFu));
  h = murmur3_mix(h, (uint32_t)((uint64_t)pos & 0xFFFFFFFFu));
  h = murmur3_mix(h, offset);
  return murmur3_fmix32(h ^ 16u);
}

__device__ __forceinline__ float murmur3_uniform32(int64_t seed, int64_t pos,
                                                   uint32_t offset) {
  uint32_t r = murmur3_hash32(seed, pos, offset);
  float hi = (float)(int32_t)(r >> 16) * 1.52587890625e-05f;
  float lo = ((float)(int32_t)(r & 0xFFFFu) + 0.5f) * 2.3283064365386963e-10f;
  return hi + lo;
}

__device__ __forceinline__ double murmur3_uniform64(int64_t seed, int64_t pos,
                                                    uint32_t offset) {
  uint64_t lo = (uint64_t)murmur3_hash32(seed, pos, offset, 0u);
  uint64_t hi = (uint64_t)murmur3_hash32(seed, pos, offset, 0x9E3779B9u);
  uint64_t random53 = ((hi << 32) | lo) >> 11;
  double u = ((double)random53 + 0.5) * 1.1102230246251565e-16;
  return u == 1.0 ? u - 1.1102230246251565e-16 : u;
}

// _log1p_neg_stable: degree-8 series on [0, 0.25), direct log elsewhere.
__device__ __forceinline__ float log1p_neg_stable(float value) {
  float p = 1.0f / 8.0f;
  p = 1.0f / 7.0f + value * p;
  p = 1.0f / 6.0f + value * p;
  p = 1.0f / 5.0f + value * p;
  p = 1.0f / 4.0f + value * p;
  p = 1.0f / 3.0f + value * p;
  p = 1.0f / 2.0f + value * p;
  p = 1.0f + value * p;
  float series = -value * p;
  float direct = tl_log(fmaxf(1.0f - value, 5.960464477539063e-08f));
  return value < 0.25f ? series : direct;
}

// ---------------------------------------------------------------------------
// Warp reductions matching Triton's AMD codegen
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_bfly_sum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, o);
  }
  return v;
}

__device__ __forceinline__ float warp_bfly_max(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    float o_v = __shfl_xor_sync(0xffffffffffffffffULL, v, o);
    v = fmaxf(v, o_v);
  }
  return v;
}

// max with Triton's `_argmax_combine_tie_break_left` (lowest index wins ties).
struct ValIdx {
  float v;
  int64_t i;
};

__device__ __forceinline__ ValIdx argmax_combine(ValIdx a, ValIdx b) {
  bool take_a = (a.v > b.v) || (a.v == b.v && a.i < b.i);
  return take_a ? a : b;
}

__device__ __forceinline__ ValIdx warp_bfly_argmax(ValIdx a) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    ValIdx b;
    b.v = __shfl_xor_sync(0xffffffffffffffffULL, a.v, o);
    b.i = __shfl_xor_sync(0xffffffffffffffffULL, a.i, o);
    a = argmax_combine(a, b);
  }
  return a;
}

// Vectorised 4-wide load; 16 bytes for fp32, 8 bytes for fp16/bf16.
template <typename T>
__device__ __forceinline__ void load4(const T* __restrict__ p, float out[4]) {
  if constexpr (sizeof(T) == 4) {
    const float4 c = *reinterpret_cast<const float4*>(p);
    out[0] = c.x;
    out[1] = c.y;
    out[2] = c.z;
    out[3] = c.w;
  } else {
    const int2 c = *reinterpret_cast<const int2*>(p);
    const T* t = reinterpret_cast<const T*>(&c);
    out[0] = (float)t[0];
    out[1] = (float)t[1];
    out[2] = (float)t[2];
    out[3] = (float)t[3];
  }
}
// 8192-element reduction helpers (4 warps, 64 registers per thread, float4
// loads). The in-thread balanced tree is over the register file in the order
// Triton's blocked layout assigns it: register k holds element
// 512*(k/4) + 128*warp + 4*lane + (k%4).
struct Block8192 {
  static constexpr int kRegs = 64;
  float v[kRegs];
  // Triton picks the load/register layout from the divisibility hints it can
  // prove: a 4-wide vectorised load (sizePerThread=4) when the row pointer,
  // the row stride and the vocab size all have 16-byte divisibility (which is
  // what Triton's specialization records; 17000 = 4*4250 is *not* enough for
  // it), otherwise a
  // scalar load (sizePerThread=1). The two layouts pair up *different*
  // elements in the in-thread reduction tree, so the association must match.
  bool vec4;

  template <typename T>
  __device__ __forceinline__ void load(const T* __restrict__ row, int64_t base,
                                       int vocab_size, int lane, int warp,
                                       bool vectorised) {
    vec4 = vectorised;
    if (!vectorised) {
      // sizePerThread == 1: register k holds element 128*k + 32*warp + lane.
#pragma unroll
      for (int k = 0; k < kRegs; k++) {
        const int64_t abs = base + 128 * k + 32 * warp + lane;
        v[k] = abs < vocab_size ? (float)row[abs] : -INFINITY;
      }
      return;
    }
#pragma unroll
    for (int p = 0; p < 16; p++) {
      int64_t off = 512 * p + 128 * warp + 4 * lane;
      int64_t abs = base + off;
      if (abs + 3 < vocab_size) {
        float c[4];
        load4<T>(row + abs, c);
        v[4 * p + 0] = c[0];
        v[4 * p + 1] = c[1];
        v[4 * p + 2] = c[2];
        v[4 * p + 3] = c[3];
      } else {
        float c[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int64_t a = abs + r;
          c[r] = a < vocab_size ? (float)row[a] : -INFINITY;
        }
        v[4 * p + 0] = c[0];
        v[4 * p + 1] = c[1];
        v[4 * p + 2] = c[2];
        v[4 * p + 3] = c[3];
      }
    }
  }

  __device__ __forceinline__ float tree_sum() const {
    float a[kRegs];
#pragma unroll
    for (int k = 0; k < kRegs; k++) a[k] = v[k];
#pragma unroll
    for (int n = kRegs; n > 1; n >>= 1) {
#pragma unroll
      for (int i = 0; i < n / 2; i++) a[i] = a[2 * i] + a[2 * i + 1];
    }
    return a[0];
  }

  __device__ __forceinline__ float max() const {
    float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < kRegs; k++) m = fmaxf(m, v[k]);
    return m;
  }

  // argmax over the block; index is the element offset inside the block.
  __device__ __forceinline__ ValIdx argmax(int lane, int warp) const {
    ValIdx a[kRegs];
#pragma unroll
    for (int k = 0; k < kRegs; k++) {
      a[k].v = v[k];
      a[k].i = vec4 ? 512 * (k >> 2) + 128 * warp + 4 * lane + (k & 3)
                    : 128 * k + 32 * warp + lane;
    }
#pragma unroll
    for (int n = kRegs; n > 1; n >>= 1) {
#pragma unroll
      for (int i = 0; i < n / 2; i++) a[i] = argmax_combine(a[2 * i], a[2 * i + 1]);
    }
    return a[0];
  }
};

// Cross-warp step: butterfly masks 2,1 over the 4 per-warp partials.
__device__ __forceinline__ float warp4_sum(float local,
                                           float* __restrict__ scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  if (lane == 0) scratch[warp] = local;
  __syncthreads();
  float r = lane < 4 ? scratch[lane] : 0.0f;
  __syncthreads();
#pragma unroll
  for (int o = 2; o >= 1; o >>= 1) {
    r += __shfl_xor_sync(0xffffffffffffffffULL, r, o);
  }
  // The butterfly only combines lanes 0..3, so broadcast lane 0's total: every
  // lane must see the same value (the sumpaths use it per-thread).
  return __shfl_sync(0xffffffffffffffffULL, r, 0);
}

__device__ __forceinline__ float warp4_max(float local,
                                           float* __restrict__ scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  if (lane == 0) scratch[warp] = local;
  __syncthreads();
  float r = lane < 4 ? scratch[lane] : -INFINITY;
  __syncthreads();
#pragma unroll
  for (int o = 2; o >= 1; o >>= 1) {
    r = fmaxf(r, __shfl_xor_sync(0xffffffffffffffffULL, r, o));
  }
  return __shfl_sync(0xffffffffffffffffULL, r, 0);
}

__device__ __forceinline__ ValIdx warp4_argmax(ValIdx local,
                                               float* __restrict__ scratch_f,
                                               int64_t* __restrict__ scratch_i) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  if (lane == 0) {
    scratch_f[warp] = local.v;
    scratch_i[warp] = local.i;
  }
  __syncthreads();
  ValIdx r;
  r.v = lane < 4 ? scratch_f[lane] : -INFINITY;
  r.i = lane < 4 ? scratch_i[lane] : 0;
  __syncthreads();
#pragma unroll
  for (int o = 2; o >= 1; o >>= 1) {
    ValIdx other;
    other.v = __shfl_xor_sync(0xffffffffffffffffULL, r.v, o);
    other.i = __shfl_xor_sync(0xffffffffffffffffULL, r.i, o);
    r = argmax_combine(r, other);
  }
  r.v = __shfl_sync(0xffffffffffffffffULL, r.v, 0);
  r.i = __shfl_sync(0xffffffffffffffffULL, r.i, 0);
  return r;
}

// ---------------------------------------------------------------------------
// _prepare_prefill_inputs_kernel -- grid = (num_reqs,)
// ---------------------------------------------------------------------------
__global__ void prepare_prefill_inputs_kernel(
    int64_t* __restrict__ last_token_indices,
    int64_t* __restrict__ draft_current_step,
    int32_t* __restrict__ draft_input_ids,
    int64_t* __restrict__ draft_positions,
    int32_t* __restrict__ draft_query_start_loc,
    int32_t* __restrict__ draft_seq_lens,
    const int32_t* __restrict__ target_input_ids,
    const int64_t* __restrict__ target_positions,
    const int32_t* __restrict__ idx_mapping,
    const int64_t* __restrict__ last_sampled,
    const int32_t* __restrict__ next_prefill_tokens,
    const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ num_rejected,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens, const int64_t max_num_reqs) {
  const int req_idx = blockIdx.x;
  const int num_reqs = gridDim.x;
  const int req_state_idx = idx_mapping[req_idx];

  const int32_t query_start = query_start_loc[req_idx];
  const int32_t query_end = query_start_loc[req_idx + 1];
  const int32_t seq_len = seq_lens[req_idx];
  const int32_t query_len = query_end - query_start - num_rejected[req_idx];

  int32_t next_token;
  if (num_sampled[req_idx] > 0) {
    next_token = (int32_t)last_sampled[req_state_idx];
  } else {
    next_token = next_prefill_tokens[req_state_idx];
  }

  for (int32_t i = 1 + threadIdx.x; i < query_len; i += blockDim.x) {
    draft_input_ids[query_start + i - 1] = target_input_ids[query_start + i];
  }
  const int32_t last_token_index = query_start + query_len - 1;
  if (threadIdx.x == 0) {
    last_token_indices[req_idx] = last_token_index;
    draft_input_ids[last_token_index] = next_token;
  }

  for (int32_t i = threadIdx.x; i < query_len; i += blockDim.x) {
    draft_positions[query_start + i] = target_positions[query_start + i];
  }

  if (threadIdx.x == 0) {
    draft_query_start_loc[req_idx] = query_start;
    draft_seq_lens[req_idx] = seq_len;
  }

  if (req_idx == num_reqs - 1) {
    if (threadIdx.x == 0) {
      *draft_current_step = 0;
    }
    for (int64_t i = num_reqs + threadIdx.x; i < max_num_reqs + 1;
         i += blockDim.x) {
      draft_query_start_loc[i] = query_end;
    }
    for (int64_t i = num_reqs + threadIdx.x; i < max_num_reqs;
         i += blockDim.x) {
      draft_seq_lens[i] = 0;
      last_token_indices[i] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// _prepare_decode_inputs_kernel -- grid = (num_reqs + 1,)
// ---------------------------------------------------------------------------
template <bool ADVANCE_DRAFT_POSITIONS>
__global__ void prepare_decode_inputs_kernel(
    const int64_t* __restrict__ draft_tokens,
    const int64_t draft_tokens_stride,
    const int32_t* __restrict__ target_seq_lens,
    const int32_t* __restrict__ num_rejected,
    int32_t* __restrict__ input_ids, int64_t* __restrict__ positions,
    int64_t* __restrict__ sample_src_positions,
    int32_t* __restrict__ query_start_loc, int32_t* __restrict__ seq_lens,
    const int64_t max_model_len, const int64_t max_num_reqs) {
  const int req_idx = blockIdx.x;
  const int num_reqs = gridDim.x - 1;

  if (req_idx == num_reqs) {
    for (int64_t i = threadIdx.x; i < max_num_reqs + 1; i += blockDim.x) {
      query_start_loc[i] = (int32_t)(i < num_reqs ? i : num_reqs);
    }
    for (int64_t i = num_reqs + threadIdx.x; i < max_num_reqs;
         i += blockDim.x) {
      seq_lens[i] = 0;
    }
    return;
  }

  input_ids[req_idx] = (int32_t)draft_tokens[req_idx * draft_tokens_stride];
  sample_src_positions[req_idx] += 1;

  int64_t seq_len = (int64_t)target_seq_lens[req_idx] - num_rejected[req_idx];
  if (ADVANCE_DRAFT_POSITIONS) {
    int64_t position = positions[req_idx];
    position = position + 1 < max_model_len - 1 ? position + 1
                                                : max_model_len - 1;
    positions[req_idx] = position;
    seq_len = seq_len + 1 < max_model_len ? seq_len + 1 : max_model_len;
  }
  seq_lens[req_idx] = (int32_t)seq_len;
}

// ---------------------------------------------------------------------------
// _update_draft_inputs_kernel -- grid = (num_reqs,)
// ---------------------------------------------------------------------------
template <typename T, bool ADVANCE_DRAFT_POSITIONS>
__global__ void update_draft_inputs_kernel(
    int64_t* __restrict__ output_draft_tokens,
    const int64_t output_draft_tokens_stride,
    T* __restrict__ next_input_hidden_states,
    const int64_t next_input_hidden_states_stride,
    int32_t* __restrict__ input_ids, int64_t* __restrict__ positions,
    int64_t* __restrict__ sample_src_positions,
    int32_t* __restrict__ seq_lens, const int64_t* __restrict__ draft_tokens,
    const int64_t* __restrict__ current_draft_step,
    const T* __restrict__ hidden_states, const int64_t hidden_states_stride,
    const int64_t hidden_size, const int64_t max_model_len,
    const int64_t num_speculative_steps) {
  const int req_idx = blockIdx.x;
  const int64_t draft_token = draft_tokens[req_idx];
  const int64_t step = *current_draft_step;
  output_draft_tokens[req_idx * output_draft_tokens_stride + step] =
      draft_token;

  if (step >= num_speculative_steps - 1) {
    return;
  }

  sample_src_positions[req_idx] += 1;
  input_ids[req_idx] = (int32_t)draft_token;

  const T* __restrict__ src = hidden_states + req_idx * hidden_states_stride;
  T* __restrict__ dst =
      next_input_hidden_states + req_idx * next_input_hidden_states_stride;
  for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    dst[i] = src[i];
  }

  if (ADVANCE_DRAFT_POSITIONS) {
    int64_t position = positions[req_idx];
    position = position + 1 < max_model_len - 1 ? position + 1
                                                : max_model_len - 1;
    positions[req_idx] = position;
    int64_t seq_len = seq_lens[req_idx];
    seq_len = seq_len + 1 < max_model_len ? seq_len + 1 : max_model_len;
    seq_lens[req_idx] = (int32_t)seq_len;
  }
}

// ---------------------------------------------------------------------------
// _gather_draft_sampled_kernel -- elementwise
// ---------------------------------------------------------------------------
__global__ void gather_draft_sampled_kernel(
    int32_t* __restrict__ draft_sampled, int64_t* __restrict__ pos_out,
    const int32_t* __restrict__ input_ids,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ logits_indices,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ expanded_local_pos,
    const int32_t* __restrict__ prefill_len, const int64_t num_logits) {
  const int64_t block = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= num_logits) {
    return;
  }
  const int64_t token_idx = logits_indices[block];
  const int32_t token_id = input_ids[token_idx];
  const int64_t pos = positions[token_idx];
  const int32_t req_state_idx = expanded_idx_mapping[block];
  const int32_t local_pos = expanded_local_pos[block];
  const int64_t prefill = prefill_len[req_state_idx];
  const bool is_placeholder = (local_pos > 0) && (pos - local_pos < prefill);
  draft_sampled[block] = is_placeholder ? -1 : token_id;
  pos_out[block] = pos;
}

// ---------------------------------------------------------------------------
// _flatten_sampled_kernel -- grid = (num_reqs,), scalar serial loop
// ---------------------------------------------------------------------------
__global__ void flatten_sampled_kernel(
    int64_t* __restrict__ flat_sampled, const int64_t* __restrict__ sampled,
    const int64_t sampled_stride, const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ cu_num_logits) {
  if (threadIdx.x != 0) {
    return;
  }
  const int req_idx = blockIdx.x;
  const int32_t start_idx = cu_num_logits[req_idx];
  const int32_t num = num_sampled[req_idx];
  for (int32_t i = 0; i < num; i++) {
    flat_sampled[start_idx + i] = sampled[req_idx * sampled_stride + i];
  }
}

// ---------------------------------------------------------------------------
// _compute_local_logits_stats_kernel -- grid = (num_logits, vocab_num_blocks)
// ---------------------------------------------------------------------------

template <typename T_TARGET, typename T_DRAFT>
__global__ void compute_local_logits_stats_kernel(
    int64_t* __restrict__ target_local_argmax,
    const int64_t target_local_argmax_stride,
    float* __restrict__ target_local_max,
    const int64_t target_local_max_stride,
    float* __restrict__ target_local_sumexp,
    const int64_t target_local_sumexp_stride,
    float* __restrict__ draft_local_max, const int64_t draft_local_max_stride,
    float* __restrict__ draft_local_sumexp,
    const int64_t draft_local_sumexp_stride,
    const T_TARGET* __restrict__ target_logits, const int64_t target_logits_stride,
    const T_DRAFT* __restrict__ draft_logits,
    const int64_t draft_logits_stride_0, const int64_t draft_logits_stride_1,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ expanded_local_pos,
    const float* __restrict__ temp, const int32_t vocab_size,
    const int32_t num_speculative_steps, const bool has_draft_logits,
    const bool vec4, const bool vec4_draft) {
  const int64_t logit_idx = blockIdx.x;
  const int32_t draft_step_idx = expanded_local_pos[logit_idx];
  if (draft_step_idx >= num_speculative_steps) {
    return;
  }
  const int32_t req_state_idx = expanded_idx_mapping[logit_idx];
  const float t = temp[req_state_idx];

  const int block_idx = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  const int64_t base = (int64_t)block_idx * 8192;
  const T_TARGET* __restrict__ trow = target_logits + logit_idx * target_logits_stride;
  __shared__ float scratch_f[8];
  __shared__ int64_t scratch_i[8];

  if (t == 0.0f) {
    Block8192 blk;
    blk.load(trow, base, vocab_size, lane, warp, vec4);
    ValIdx local = blk.argmax(lane, warp);
    ValIdx best = warp_bfly_argmax(local);
    ValIdx overall = warp4_argmax(best, scratch_f, scratch_i);
    if (threadIdx.x == 0) {
      target_local_argmax[logit_idx * target_local_argmax_stride + block_idx] =
          (int64_t)base + overall.i;
      target_local_max[logit_idx * target_local_max_stride + block_idx] =
          overall.v;
    }
    return;
  }

  Block8192 blk;
  blk.load(trow, base, vocab_size, lane, warp, vec4);
  const float max_value = warp4_max(warp_bfly_max(blk.max()), scratch_f);
  Block8192 blk_exp;
  if (max_value > -INFINITY) {
#pragma unroll
    for (int k = 0; k < Block8192::kRegs; k++) {
      blk_exp.v[k] = tl_exp(blk.v[k] - max_value);
    }
  } else {
#pragma unroll
    for (int k = 0; k < Block8192::kRegs; k++) blk_exp.v[k] = 0.0f;
  }
  float sumexp = warp_bfly_sum(blk_exp.tree_sum());
  sumexp = warp4_sum(sumexp, scratch_f);
  if (threadIdx.x == 0) {
    target_local_max[logit_idx * target_local_max_stride + block_idx] =
        max_value;
    target_local_sumexp[logit_idx * target_local_sumexp_stride + block_idx] =
        max_value > -INFINITY ? sumexp : 0.0f;
  }

  if (has_draft_logits) {
    const T_DRAFT* __restrict__ drow =
        draft_logits + (int64_t)req_state_idx * draft_logits_stride_0 +
        (int64_t)draft_step_idx * draft_logits_stride_1;
    Block8192 dblk;
    dblk.vec4 = vec4_draft;
    if (!vec4_draft) {
#pragma unroll
      for (int k = 0; k < Block8192::kRegs; k++) {
        const int64_t abs = base + 128 * k + 32 * warp + lane;
        dblk.v[k] = abs < vocab_size ? (float)drow[abs] / t : -INFINITY;
      }
    } else {
#pragma unroll
    for (int p = 0; p < 16; p++) {
      const int64_t abs = base + 512 * p + 128 * warp + 4 * lane;
      if (abs + 3 < vocab_size) {
        float c[4];
        load4<T_DRAFT>(drow + abs, c);
#pragma unroll
        for (int r = 0; r < 4; r++) dblk.v[4 * p + r] = c[r] / t;
      } else {
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const bool in_range = abs + r < vocab_size;
          const float value = in_range ? (float)drow[abs + r] : -INFINITY;
          dblk.v[4 * p + r] = in_range ? value / t : -INFINITY;
        }
      }
    }
    }
    const float dmax = warp4_max(warp_bfly_max(dblk.max()), scratch_f);
    Block8192 dexpb;
    if (dmax > -INFINITY) {
#pragma unroll
      for (int k = 0; k < Block8192::kRegs; k++) {
        dexpb.v[k] = tl_exp(dblk.v[k] - dmax);
      }
    } else {
#pragma unroll
      for (int k = 0; k < Block8192::kRegs; k++) dexpb.v[k] = 0.0f;
    }
    float dsum = warp_bfly_sum(dexpb.tree_sum());
    dsum = warp4_sum(dsum, scratch_f);
    if (threadIdx.x == 0) {
      draft_local_max[logit_idx * draft_local_max_stride + block_idx] = dmax;
      draft_local_sumexp[logit_idx * draft_local_sumexp_stride + block_idx] =
          dmax > -INFINITY ? dsum : 0.0f;
    }
  }
}


// ---------------------------------------------------------------------------
// Per-request reductions over the padded vocab blocks (one warp per request,
// lane == block index), matching Triton's `tl.sum`/`tl.max` over
// `tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)` with num_warps=1.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float global_logsumexp(
    const float* __restrict__ local_max, const int64_t max_stride,
    const float* __restrict__ local_sumexp, const int64_t sumexp_stride,
    const int64_t logit_idx, const int32_t vocab_num_blocks,
    const int32_t padded_vocab_num_blocks, const int lane) {
  const bool in_range = lane < vocab_num_blocks;
  const float mx =
      in_range ? local_max[logit_idx * max_stride + lane] : -INFINITY;
  const float sx =
      in_range ? local_sumexp[logit_idx * sumexp_stride + lane] : 0.0f;
  const float gmax = warp_bfly_max(mx);
  const float e = tl_exp(mx - gmax);
  const float prod = sx * e;
  // Triton's `tl.sum(sumexps * exp(...))` is reduced over PADDED_VOCAB_NUM_BLOCKS
  // lanes and LLVM contracts the elementwise multiply into the *first* butterfly
  // add (`fma(sumexps, exp, shfl(prod, n/2))`); the remaining steps are plain
  // adds. Reproduce that so the product is rounded exactly once, as Triton does.
  float s;
  if (padded_vocab_num_blocks >= 2) {
    int o = padded_vocab_num_blocks / 2;
    s = __fmaf_rn(
        sx, e, __shfl_xor_sync(0xffffffffffffffffULL, prod, o));
    for (o >>= 1; o >= 1; o >>= 1) {
      s += __shfl_xor_sync(0xffffffffffffffffULL, s, o);
    }
  } else {
    s = prod;
  }
  // Triton's reduce result is a uniform scalar; broadcast lane 0's total.
  s = __shfl_sync(0xffffffffffffffffULL, s, 0);
  return gmax + tl_log(s);
}

__device__ __forceinline__ int64_t global_target_argmax(
    const float* __restrict__ local_max, const int64_t max_stride,
    const int64_t* __restrict__ local_argmax, const int64_t argmax_stride,
    const int64_t logit_idx, const int32_t vocab_num_blocks,
    const int32_t padded_vocab_num_blocks, const int lane) {
  float mx = lane < padded_vocab_num_blocks && lane < vocab_num_blocks
                 ? local_max[logit_idx * max_stride + lane]
                 : -INFINITY;
  mx = mx != mx ? -INFINITY : mx;
  // argmax over the padded region, lowest block index wins ties.
  ValIdx best;
  best.v = mx;
  best.i = lane;
  ValIdx r = warp_bfly_argmax(best);
  const int64_t block_idx = r.i < padded_vocab_num_blocks ? r.i : 0;
  return local_argmax[logit_idx * argmax_stride + block_idx];
}

// target_log_prob, draft_log_prob, target_lse, draft_lse for one token.
struct LogProbs {
  float target_log_prob;
  float draft_log_prob;
  float target_lse;
  float draft_lse;
};

template <typename T_TARGET, typename T_DRAFT>
__device__ __forceinline__ LogProbs global_logprobs_and_lse(
    const int64_t token, const bool mask, const int64_t logit_idx,
    const int32_t req_state_idx, const int32_t draft_step, const float temp,
    const T_TARGET* __restrict__ target_logits, const int64_t target_logits_stride,
    const float* __restrict__ target_local_max,
    const int64_t target_local_max_stride,
    const float* __restrict__ target_local_sumexp,
    const int64_t target_local_sumexp_stride,
    const T_DRAFT* __restrict__ draft_logits,
    const int64_t draft_logits_stride_0, const int64_t draft_logits_stride_1,
    const float* __restrict__ draft_local_max,
    const int64_t draft_local_max_stride,
    const float* __restrict__ draft_local_sumexp,
    const int64_t draft_local_sumexp_stride, const int32_t vocab_num_blocks,
    const int32_t padded_vocab_num_blocks, const bool has_draft_logits,
    const int lane) {
  LogProbs out;
  const float target_logit =
      mask ? (float)target_logits[logit_idx * target_logits_stride + token]
           : -INFINITY;
  out.target_lse = global_logsumexp(
      target_local_max, target_local_max_stride, target_local_sumexp,
      target_local_sumexp_stride, logit_idx, vocab_num_blocks,
      padded_vocab_num_blocks, lane);
  out.target_log_prob = target_logit - out.target_lse;
  if (has_draft_logits) {
    const T_DRAFT* __restrict__ row =
        draft_logits + (int64_t)req_state_idx * draft_logits_stride_0 +
        (int64_t)draft_step * draft_logits_stride_1;
    const float draft_logit =
        mask ? (float)row[token] / temp : -INFINITY;
    out.draft_lse = global_logsumexp(
        draft_local_max, draft_local_max_stride, draft_local_sumexp,
        draft_local_sumexp_stride, logit_idx, vocab_num_blocks,
        padded_vocab_num_blocks, lane);
    out.draft_log_prob = draft_logit - out.draft_lse;
  } else {
    out.draft_log_prob = 0.0f;
    out.draft_lse = 0.0f;
  }
  return out;
}

// max(p_i * M_b(x) - M_s(x), 0) summed over the vocab blocks.
template <typename T_TARGET, typename T_DRAFT>
__device__ __forceinline__ float global_residual_mass(
    const float* __restrict__ local_residual_mass, const int64_t lrm_stride,
    const float prefix_joint_ratio, const T_TARGET* __restrict__ target_logits,
    const int64_t target_logits_stride,
    const float* __restrict__ target_local_max,
    const int64_t target_local_max_stride,
    const float* __restrict__ target_local_sumexp,
    const int64_t target_local_sumexp_stride, const int64_t draft_token,
    const int64_t logit_idx, const int32_t vocab_num_blocks,
    const int32_t padded_vocab_num_blocks, const bool has_draft_logits,
    const int lane) {
  if (has_draft_logits) {
    const float p = lane < vocab_num_blocks
                        ? local_residual_mass[logit_idx * lrm_stride + lane]
                        : 0.0f;
    return warp_bfly_sum(p);
  }
  const float target_lse = global_logsumexp(
      target_local_max, target_local_max_stride, target_local_sumexp,
      target_local_sumexp_stride, logit_idx, vocab_num_blocks,
      padded_vocab_num_blocks, lane);
  const float target_logit =
      (float)target_logits[logit_idx * target_logits_stride + draft_token];
  const float m_b = tl_exp(target_logit - target_lse);
  return prefix_joint_ratio * (1.0f - m_b);
}

// ---------------------------------------------------------------------------
// _rejection_kernel -- grid = (num_reqs,), one warp per request
// ---------------------------------------------------------------------------
template <typename T_TARGET, typename T_DRAFT>
__global__ void rejection_kernel(
    int64_t* __restrict__ sampled, const int64_t sampled_stride,
    int32_t* __restrict__ rejected_steps,
    float* __restrict__ target_rejected_logsumexp,
    float* __restrict__ draft_rejected_logsumexp,
    const T_TARGET* __restrict__ target_logits, const int64_t target_logits_stride,
    const int64_t* __restrict__ target_local_argmax,
    const int64_t target_local_argmax_stride,
    const float* __restrict__ target_local_max,
    const int64_t target_local_max_stride,
    const float* __restrict__ target_local_sumexp,
    const int64_t target_local_sumexp_stride,
    const int32_t* __restrict__ draft_sampled,
    const T_DRAFT* __restrict__ draft_logits,
    const int64_t draft_logits_stride_0, const int64_t draft_logits_stride_1,
    const float* __restrict__ draft_local_max,
    const int64_t draft_local_max_stride,
    const float* __restrict__ draft_local_sumexp,
    const int64_t draft_local_sumexp_stride,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ idx_mapping, const float* __restrict__ temp_ptr,
    const int64_t* __restrict__ seed_ptr, const int64_t* __restrict__ pos_ptr,
    const float* __restrict__ synthetic_conditional_rates,
    const float* __restrict__ cumulative_log_p,
    const float* __restrict__ local_residual_mass,
    const int64_t local_residual_mass_stride,
    const int32_t vocab_num_blocks, const int32_t padded_vocab_num_blocks,
    const bool has_draft_logits, const bool synthetic_mode,
    const bool use_block_verification) {
  const int req_idx = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int32_t req_state_idx = idx_mapping[req_idx];
  const int64_t start_idx = cu_num_logits[req_idx];
  const int64_t end_idx = cu_num_logits[req_idx + 1];
  const int64_t num_draft_tokens = end_idx - start_idx - 1;
  const int64_t seed = seed_ptr[req_state_idx];
  const float t = temp_ptr[req_state_idx];
  const bool is_greedy = t == 0.0f;

  int64_t accepted_length = 0;
  float target_lse = 0.0f;
  float draft_lse = 0.0f;
  bool verifying = true;

  for (int64_t i = 0; i < num_draft_tokens; i++) {
    const int64_t logit_idx = start_idx + i;
    int64_t draft_token = draft_sampled[logit_idx + 1];
    const bool is_valid_draft = draft_token >= 0;
    if (draft_token < 0) draft_token = 0;
    if (!is_greedy) {
      verifying = verifying && is_valid_draft;
    }
    if (!verifying) {
      continue;
    }
    const int64_t pos = pos_ptr[logit_idx];
    const float u = tl_rand32_nozero(seed, pos);

    if (is_greedy) {
      const int64_t target_argmax = global_target_argmax(
          target_local_max, target_local_max_stride, target_local_argmax,
          target_local_argmax_stride, logit_idx, vocab_num_blocks,
          padded_vocab_num_blocks, lane);
      bool accepted;
      if (synthetic_mode) {
        accepted = u < synthetic_conditional_rates[i];
      } else {
        accepted = target_argmax == draft_token;
      }
      accepted = accepted && is_valid_draft;
      verifying = accepted;
      accepted_length += accepted ? 1 : 0;
      if (threadIdx.x == 0) {
        sampled[req_idx * sampled_stride + i] =
            accepted ? draft_token : target_argmax;
      }
    } else if (use_block_verification) {
      const float prefix_joint_ratio =
          tl_exp(cumulative_log_p[logit_idx]);
      int64_t next_draft_token = -1;
      if (i < num_draft_tokens - 1) {
        next_draft_token = draft_sampled[logit_idx + 2];
      }
      float h;
      if (next_draft_token >= 0) {
        const float residual_mass = global_residual_mass<T_TARGET, T_DRAFT>(
            local_residual_mass, local_residual_mass_stride,
            prefix_joint_ratio, target_logits, target_logits_stride,
            target_local_max, target_local_max_stride, target_local_sumexp,
            target_local_sumexp_stride, next_draft_token, logit_idx + 1,
            vocab_num_blocks, padded_vocab_num_blocks, has_draft_logits, lane);
        const float denom = residual_mass + 1.0f - prefix_joint_ratio;
        h = denom > 0.0f ? residual_mass / denom : 1.0f;
      } else {
        h = prefix_joint_ratio;
      }
      accepted_length = u <= h ? i + 1 : accepted_length;
      if (threadIdx.x == 0) {
        sampled[req_idx * sampled_stride + i] = draft_token;
      }
    } else {
      const LogProbs lp = global_logprobs_and_lse<T_TARGET, T_DRAFT>(
          (float)draft_token, true, logit_idx, req_state_idx, (int32_t)i, t,
          target_logits, target_logits_stride, target_local_max,
          target_local_max_stride, target_local_sumexp,
          target_local_sumexp_stride, draft_logits, draft_logits_stride_0,
          draft_logits_stride_1, draft_local_max, draft_local_max_stride,
          draft_local_sumexp, draft_local_sumexp_stride, vocab_num_blocks,
          padded_vocab_num_blocks, has_draft_logits, lane);
      // Triton rebinds target_lse/draft_lse from these helpers, and they are
      // stored into the rejected-logsumexp outputs below.
      target_lse = lp.target_lse;
      draft_lse = lp.draft_lse;
      bool accepted;
      if (synthetic_mode) {
        accepted = u < synthetic_conditional_rates[i];
      } else {
        accepted = lp.target_log_prob > tl_log(u) + lp.draft_log_prob;
      }
      verifying = accepted;
      accepted_length += accepted ? 1 : 0;
      if (threadIdx.x == 0) {
        sampled[req_idx * sampled_stride + i] = draft_token;
      }
    }
  }

  if (threadIdx.x == 0) {
    rejected_steps[req_idx] = (int32_t)accepted_length;
  }

  if (use_block_verification && !is_greedy &&
      accepted_length < num_draft_tokens) {
    const int64_t rejected_idx = start_idx + accepted_length;
    target_lse = global_logsumexp(
        target_local_max, target_local_max_stride, target_local_sumexp,
        target_local_sumexp_stride, rejected_idx, vocab_num_blocks,
        padded_vocab_num_blocks, lane);
    if (has_draft_logits) {
      draft_lse = global_logsumexp(
          draft_local_max, draft_local_max_stride, draft_local_sumexp,
          draft_local_sumexp_stride, rejected_idx, vocab_num_blocks,
          padded_vocab_num_blocks, lane);
    }
  }
  if (threadIdx.x == 0) {
    target_rejected_logsumexp[req_idx] = target_lse;
    draft_rejected_logsumexp[req_idx] = draft_lse;
  }
}

// ---------------------------------------------------------------------------
// _resample_kernel -- grid = (num_reqs, resample_num_blocks)
// ---------------------------------------------------------------------------
template <bool USE_FP64>
struct ResampleVal {
  using type = typename std::conditional<USE_FP64, double, float>::type;
};

template <bool USE_FP64, typename T_TARGET, typename T_DRAFT>
__global__ void resample_kernel(
    int64_t* __restrict__ resampled_local_argmax,
    const int64_t resampled_local_argmax_stride,
    void* __restrict__ resampled_local_max,
    const int64_t resampled_local_max_stride,
    const T_TARGET* __restrict__ target_logits, const int64_t target_logits_stride,
    const float* __restrict__ target_rejected_logsumexp,
    const T_DRAFT* __restrict__ draft_logits,
    const int64_t draft_logits_stride_0, const int64_t draft_logits_stride_1,
    const float* __restrict__ draft_rejected_logsumexp,
    const int32_t* __restrict__ rejected_step,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int32_t* __restrict__ draft_sampled,
    const float* __restrict__ temp_ptr, const int64_t* __restrict__ seed_ptr,
    const int64_t* __restrict__ pos_ptr,
    const float* __restrict__ cumulative_log_p,
    const int32_t vocab_size, const bool has_draft_logits,
    const bool use_block_verification) {
  using VT = typename ResampleVal<USE_FP64>::type;
  const int req_idx = blockIdx.x;
  const int32_t resample_idx = rejected_step[req_idx];
  const int64_t start_idx = cu_num_logits[req_idx];
  const int64_t end_idx = cu_num_logits[req_idx + 1];
  const int64_t resample_token_idx = start_idx + resample_idx;
  const int32_t req_state_idx = expanded_idx_mapping[resample_token_idx];

  const float t = temp_ptr[req_state_idx];
  const bool is_bonus = resample_token_idx == end_idx - 1;
  if (t == 0.0f && !is_bonus) {
    return;
  }

  const int64_t rejected_draft_token =
      is_bonus ? 0 : (int64_t)draft_sampled[resample_token_idx + 1];
  const bool is_valid_rejected_draft = rejected_draft_token >= 0;

  // One 1024-wide vocabulary block per program, 8 elements per thread (the
  // 128-thread block covers all 1024, matching Triton's sizePerThread=8).
  const int64_t block_base = (int64_t)blockIdx.y * 1024;
  const int64_t stride = blockDim.x;
  VT best = VT(-INFINITY);
  int64_t best_idx = block_base;
  for (int e = 0; e < 8; e++) {
    const int64_t block = block_base + (int64_t)e * stride + threadIdx.x;
    const bool mask = block < vocab_size;
    const float target_logit =
        mask ? (float)
                   target_logits[resample_token_idx * target_logits_stride + block]
             : -INFINITY;

    float residual_logits;
    if (is_bonus || !is_valid_rejected_draft) {
      // Bonus token (no rejections) or -1 placeholder: use the target logits.
      residual_logits = target_logit;
    } else if (has_draft_logits) {
      const float draft_logit =
          mask ? (float)draft_logits[(int64_t)req_state_idx *
                                         draft_logits_stride_0 +
                                     (int64_t)resample_idx *
                                         draft_logits_stride_1 +
                                     block] /
                     t
               : -INFINITY;
      const float target_lse = target_rejected_logsumexp[req_idx];
      const float draft_lse = draft_rejected_logsumexp[req_idx];
      float target_log_probs = target_logit - target_lse;
      if (use_block_verification) {
        float log_p_tau = 0.0f;
        if (resample_idx > 0) {
          log_p_tau = cumulative_log_p[resample_token_idx - 1];
        }
        target_log_probs += log_p_tau;
      }
      const float draft_log_probs = draft_logit - draft_lse;
      const float ratio = tl_exp(draft_log_probs - target_log_probs);
      residual_logits =
          ratio < 1.0f ? target_log_probs + log1pf(-ratio) : -INFINITY;
    } else {
      // One-hot draft: zero out the rejected token.
      residual_logits =
          block != rejected_draft_token ? target_logit : -INFINITY;
    }

    const bool valid_req = req_state_idx >= 0;
    const int64_t seed = valid_req ? seed_ptr[req_state_idx] : 0;
    const int64_t pos = pos_ptr[resample_token_idx];

    VT value;
    if constexpr (USE_FP64) {
      const double logits = (double)residual_logits;
      if (t != 0.0f) {
        const double u = murmur3_uniform64(seed, pos, (uint32_t)block);
        const double noise = -log(-log(u));
        value = mask ? logits + noise : -INFINITY;
      } else {
        value = mask ? logits : -INFINITY;
      }
    } else {
      if (t != 0.0f) {
        const float u = murmur3_uniform32(seed, pos, (uint32_t)block);
        const float noise = -tl_log(-log1p_neg_stable(u));
        value = mask ? residual_logits + noise : -INFINITY;
      } else {
        value = mask ? residual_logits : -INFINITY;
      }
    }
    if (value > best) {
      best = value;
      best_idx = block;
    }
  }

  // Block argmax. Triton reduces with tl.max(..., return_indices=True), whose
  // tie-break keeps the lowest index; a max is order independent, so any tree
  // reproduces it.
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    VT other = __shfl_xor_sync(0xffffffffffffffffULL, best, o);
    int64_t other_idx = __shfl_xor_sync(0xffffffffffffffffULL, best_idx, o);
    const bool take = (other > best) || (other == best && other_idx < best_idx);
    if (take) {
      best = other;
      best_idx = other_idx;
    }
  }
  __shared__ VT s_val[4];
  __shared__ int64_t s_idx[4];
  if (lane == 0) {
    s_val[warp] = best;
    s_idx[warp] = best_idx;
  }
  __syncthreads();
  if (warp == 0) {
    VT v = lane < 4 ? s_val[lane] : VT(-INFINITY);
    int64_t idx = lane < 4 ? s_idx[lane] : 0;
#pragma unroll
    for (int o = 2; o >= 1; o >>= 1) {
      VT other = __shfl_xor_sync(0xffffffffffffffffULL, v, o);
      int64_t other_idx = __shfl_xor_sync(0xffffffffffffffffULL, idx, o);
      const bool take = (other > v) || (other == v && other_idx < idx);
      if (take) {
        v = other;
        idx = other_idx;
      }
    }
    if (lane == 0) {
      reinterpret_cast<VT*>(resampled_local_max)
          [req_idx * resampled_local_max_stride + blockIdx.y] = v;
      resampled_local_argmax[req_idx * resampled_local_argmax_stride +
                             blockIdx.y] = idx;
    }
  }
}

// ---------------------------------------------------------------------------
// _insert_resampled_kernel -- grid = (num_reqs,), warp-parallel block argmax
// ---------------------------------------------------------------------------
template <typename T_MAX>
__global__ void insert_resampled_kernel(
    int64_t* __restrict__ sampled, const int64_t sampled_stride,
    int32_t* __restrict__ num_sampled,
    const int64_t* __restrict__ resampled_local_argmax,
    const int64_t resampled_local_argmax_stride,
    const T_MAX* __restrict__ resampled_local_max,
    const int64_t resampled_local_max_stride,
    const int32_t resample_num_blocks,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ expanded_idx_mapping,
    const float* __restrict__ temp_ptr) {
  const int tid = threadIdx.x;
  const int req_idx = blockIdx.x;
  const int32_t num = num_sampled[req_idx];
  const int32_t start_idx = cu_num_logits[req_idx];
  const int32_t end_idx = cu_num_logits[req_idx + 1];
  const int32_t resample_token_idx = start_idx + num;
  const int32_t req_state_idx = expanded_idx_mapping[resample_token_idx];

  if (tid == 0) {
    num_sampled[req_idx] = num + 1;
  }

  const float t = temp_ptr[req_state_idx];
  const bool is_bonus = resample_token_idx == end_idx - 1;
  if (t == 0.0f && !is_bonus) {
    return;
  }

  // argmax over the padded block maxes; NaN maps to -inf and ties keep the
  // lowest block index (matching tl.argmax's tie-break-left). The 149-block
  // scan is spread over the warp instead of serialised in thread 0.
  float best = -INFINITY;
  int best_idx = 0x7fffffff;
  for (int b = tid; b < resample_num_blocks; b += blockDim.x) {
    float v = (float)resampled_local_max[req_idx * resampled_local_max_stride +
                                         b];
    v = v != v ? -INFINITY : v;
    if (v > best) {
      best = v;
      best_idx = b;
    }
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffffffffffULL, best, o);
    const int oi = __shfl_xor_sync(0xffffffffffffffffULL, best_idx, o);
    if (ov > best || (ov == best && oi < best_idx)) {
      best = ov;
      best_idx = oi;
    }
  }
  if (tid == 0) {
    sampled[req_idx * sampled_stride + num] =
        resampled_local_argmax
            [req_idx * resampled_local_argmax_stride +
             (best_idx == 0x7fffffff ? 0 : best_idx)];
  }
}

}  // namespace rdna2_spec_decode
}  // namespace vllm

// ===========================================================================
// Host wrappers (global scope, resolved by the torch_bindings importer)
// ===========================================================================
namespace {

using at::Tensor;
using vllm::rdna2_spec_decode::insert_resampled_kernel;
using vllm::rdna2_spec_decode::compute_local_logits_stats_kernel;
using vllm::rdna2_spec_decode::flatten_sampled_kernel;
using vllm::rdna2_spec_decode::gather_draft_sampled_kernel;
using vllm::rdna2_spec_decode::prepare_decode_inputs_kernel;
using vllm::rdna2_spec_decode::prepare_prefill_inputs_kernel;
using vllm::rdna2_spec_decode::rejection_kernel;
using vllm::rdna2_spec_decode::resample_kernel;
using vllm::rdna2_spec_decode::update_draft_inputs_kernel;

constexpr int kThreads = 256;
constexpr int kScalarThreads = 32;
constexpr int kReduceThreads = 128;

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

#define LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()

// Triton only emits a 4-wide vector load (and with it the sizePerThread=4
// register layout) when the pointer, the row stride and the mask bound are all
// provably 4-element friendly; otherwise the layout collapses to
// sizePerThread=1 and the in-thread reduction tree pairs different elements.
bool wide_load_ok(const Tensor& t, int64_t vocab_size) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
         t.stride(0) % 16 == 0 && vocab_size % 16 == 0;
}

// Row-major 3-D (max_num_reqs, num_speculative_steps, vocab): the inner stride
// is what the load walks.
bool wide_load_ok_2d(const Tensor& t, int64_t vocab_size) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
         t.stride(0) % 16 == 0 && t.stride(1) % 16 == 0 && vocab_size % 16 == 0;
}

void check_dtype(const Tensor& t, at::ScalarType expected, const char* what) {
  TORCH_CHECK(t.scalar_type() == expected, "rdna2 spec-decode: ", what,
              " has unexpected dtype");
}

// The target logits reach the rejection sampler in whatever dtype the sampler
// produced them in: apply_sampling_params materialises an fp32 buffer only
// when a request needs logits processing, so greedy/no-penalty requests hand
// over the model dtype (fp16) directly. Triton loads them generically and
// casts to fp32, so the HIP kernels template the element type the same way.
template <typename F>
void dispatch_float_dtype(const Tensor& t, F&& f) {
  switch (t.scalar_type()) {
    case at::kFloat:
      f(float{});
      return;
    case at::kHalf:
      f(at::Half{});
      return;
    case at::kBFloat16:
      f(at::BFloat16{});
      return;
    default:
      TORCH_CHECK(false,
                  "rdna2 spec-decode: unsupported target logits dtype");
  }
}

}  // namespace

void spec_prepare_prefill_inputs_rdna2(
    torch::Tensor last_token_indices, torch::Tensor draft_current_step,
    torch::Tensor draft_input_ids, torch::Tensor draft_positions,
    torch::Tensor draft_query_start_loc, torch::Tensor draft_seq_lens,
    torch::Tensor target_input_ids, torch::Tensor target_positions,
    torch::Tensor idx_mapping, torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
    torch::Tensor num_rejected, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, int64_t max_num_reqs) {
  check_dtype(last_token_indices, at::kLong, "last_token_indices");
  check_dtype(draft_current_step, at::kLong, "draft_current_step");
  check_dtype(draft_input_ids, at::kInt, "draft_input_ids");
  check_dtype(draft_positions, at::kLong, "draft_positions");
  check_dtype(draft_query_start_loc, at::kInt, "draft_query_start_loc");
  check_dtype(draft_seq_lens, at::kInt, "draft_seq_lens");
  check_dtype(target_input_ids, at::kInt, "target_input_ids");
  check_dtype(target_positions, at::kLong, "target_positions");
  check_dtype(idx_mapping, at::kInt, "idx_mapping");
  check_dtype(last_sampled, at::kLong, "last_sampled");
  check_dtype(next_prefill_tokens, at::kInt, "next_prefill_tokens");
  check_dtype(num_sampled, at::kInt, "num_sampled");
  check_dtype(num_rejected, at::kInt, "num_rejected");
  check_dtype(query_start_loc, at::kInt, "query_start_loc");
  check_dtype(seq_lens, at::kInt, "seq_lens");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  auto stream = at::cuda::getCurrentCUDAStream();
  prepare_prefill_inputs_kernel<<<num_reqs, kThreads, 0, stream.stream()>>>(
      reinterpret_cast<int64_t*>(last_token_indices.data_ptr()),
      reinterpret_cast<int64_t*>(draft_current_step.data_ptr()),
      reinterpret_cast<int32_t*>(draft_input_ids.data_ptr()),
      reinterpret_cast<int64_t*>(draft_positions.data_ptr()),
      reinterpret_cast<int32_t*>(draft_query_start_loc.data_ptr()),
      reinterpret_cast<int32_t*>(draft_seq_lens.data_ptr()),
      reinterpret_cast<const int32_t*>(target_input_ids.data_ptr()),
      reinterpret_cast<const int64_t*>(target_positions.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const int64_t*>(last_sampled.data_ptr()),
      reinterpret_cast<const int32_t*>(next_prefill_tokens.data_ptr()),
      reinterpret_cast<const int32_t*>(num_sampled.data_ptr()),
      reinterpret_cast<const int32_t*>(num_rejected.data_ptr()),
      reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
      reinterpret_cast<const int32_t*>(seq_lens.data_ptr()), max_num_reqs);
  LAUNCH_CHECK();
}

void spec_prepare_decode_inputs_rdna2(
    torch::Tensor draft_tokens, torch::Tensor target_seq_lens,
    torch::Tensor num_rejected, torch::Tensor input_ids,
    torch::Tensor positions, torch::Tensor sample_src_positions,
    torch::Tensor query_start_loc, torch::Tensor seq_lens,
    int64_t max_model_len, int64_t max_num_reqs,
    bool advance_draft_positions) {
  check_dtype(draft_tokens, at::kLong, "draft_tokens");
  check_dtype(target_seq_lens, at::kInt, "target_seq_lens");
  check_dtype(num_rejected, at::kInt, "num_rejected");
  check_dtype(input_ids, at::kInt, "input_ids");
  check_dtype(positions, at::kLong, "positions");
  check_dtype(sample_src_positions, at::kLong, "sample_src_positions");
  check_dtype(query_start_loc, at::kInt, "query_start_loc");
  check_dtype(seq_lens, at::kInt, "seq_lens");

  const int num_reqs = (int)draft_tokens.size(0);
  const at::cuda::OptionalCUDAGuard guard(device_of(draft_tokens));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t stride = draft_tokens.stride(0);
  if (advance_draft_positions) {
    prepare_decode_inputs_kernel<true><<<num_reqs + 1, kThreads, 0,
                                         stream.stream()>>>(
        reinterpret_cast<const int64_t*>(draft_tokens.data_ptr()), stride,
        reinterpret_cast<const int32_t*>(target_seq_lens.data_ptr()),
        reinterpret_cast<const int32_t*>(num_rejected.data_ptr()),
        reinterpret_cast<int32_t*>(input_ids.data_ptr()),
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<int64_t*>(sample_src_positions.data_ptr()),
        reinterpret_cast<int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<int32_t*>(seq_lens.data_ptr()), max_model_len,
        max_num_reqs);
  } else {
    prepare_decode_inputs_kernel<false><<<num_reqs + 1, kThreads, 0,
                                          stream.stream()>>>(
        reinterpret_cast<const int64_t*>(draft_tokens.data_ptr()), stride,
        reinterpret_cast<const int32_t*>(target_seq_lens.data_ptr()),
        reinterpret_cast<const int32_t*>(num_rejected.data_ptr()),
        reinterpret_cast<int32_t*>(input_ids.data_ptr()),
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<int64_t*>(sample_src_positions.data_ptr()),
        reinterpret_cast<int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<int32_t*>(seq_lens.data_ptr()), max_model_len,
        max_num_reqs);
  }
  LAUNCH_CHECK();
}

template <typename T>
static void launch_update_draft_inputs(
    const Tensor& output_draft_tokens, Tensor& next_input_hidden_states,
    Tensor& input_ids, Tensor& positions, Tensor& sample_src_positions,
    Tensor& seq_lens, const Tensor& draft_tokens,
    const Tensor& current_draft_step, const Tensor& hidden_states,
    int64_t hidden_size, int64_t max_model_len, int64_t num_speculative_steps,
    bool advance_draft_positions) {
  const int num_reqs = (int)draft_tokens.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_stride = output_draft_tokens.stride(0);
  const int64_t nihs_stride = next_input_hidden_states.stride(0);
  const int64_t hs_stride = hidden_states.stride(0);
  if (advance_draft_positions) {
    update_draft_inputs_kernel<T, true><<<num_reqs, kThreads, 0,
                                          stream.stream()>>>(
        reinterpret_cast<int64_t*>(output_draft_tokens.data_ptr()), out_stride,
        reinterpret_cast<T*>(next_input_hidden_states.data_ptr()),
        nihs_stride, reinterpret_cast<int32_t*>(input_ids.data_ptr()),
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<int64_t*>(sample_src_positions.data_ptr()),
        reinterpret_cast<int32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const int64_t*>(draft_tokens.data_ptr()),
        reinterpret_cast<const int64_t*>(current_draft_step.data_ptr()),
        reinterpret_cast<const T*>(hidden_states.data_ptr()), hs_stride,
        hidden_size, max_model_len, num_speculative_steps);
  } else {
    update_draft_inputs_kernel<T, false><<<num_reqs, kThreads, 0,
                                           stream.stream()>>>(
        reinterpret_cast<int64_t*>(output_draft_tokens.data_ptr()), out_stride,
        reinterpret_cast<T*>(next_input_hidden_states.data_ptr()),
        nihs_stride, reinterpret_cast<int32_t*>(input_ids.data_ptr()),
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<int64_t*>(sample_src_positions.data_ptr()),
        reinterpret_cast<int32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const int64_t*>(draft_tokens.data_ptr()),
        reinterpret_cast<const int64_t*>(current_draft_step.data_ptr()),
        reinterpret_cast<const T*>(hidden_states.data_ptr()), hs_stride,
        hidden_size, max_model_len, num_speculative_steps);
  }
  LAUNCH_CHECK();
}

void spec_update_draft_inputs_rdna2(
    torch::Tensor output_draft_tokens,
    torch::Tensor next_input_hidden_states, torch::Tensor input_ids,
    torch::Tensor positions, torch::Tensor sample_src_positions,
    torch::Tensor seq_lens, torch::Tensor draft_tokens,
    torch::Tensor current_draft_step, torch::Tensor hidden_states,
    int64_t hidden_size, int64_t max_model_len, int64_t num_speculative_steps,
    bool advance_draft_positions) {
  check_dtype(output_draft_tokens, at::kLong, "output_draft_tokens");
  check_dtype(input_ids, at::kInt, "input_ids");
  check_dtype(positions, at::kLong, "positions");
  check_dtype(sample_src_positions, at::kLong, "sample_src_positions");
  check_dtype(seq_lens, at::kInt, "seq_lens");
  check_dtype(draft_tokens, at::kLong, "draft_tokens");
  check_dtype(current_draft_step, at::kLong, "current_draft_step");
  TORCH_CHECK(hidden_states.scalar_type() ==
                  next_input_hidden_states.scalar_type(),
              "rdna2 spec-decode: hidden-state dtype mismatch");

  const at::cuda::OptionalCUDAGuard guard(device_of(draft_tokens));
  if (hidden_states.scalar_type() == at::kFloat) {
    launch_update_draft_inputs<float>(
        output_draft_tokens, next_input_hidden_states, input_ids, positions,
        sample_src_positions, seq_lens, draft_tokens, current_draft_step,
        hidden_states, hidden_size, max_model_len, num_speculative_steps,
        advance_draft_positions);
  } else if (hidden_states.scalar_type() == at::kHalf) {
    launch_update_draft_inputs<at::Half>(
        output_draft_tokens, next_input_hidden_states, input_ids, positions,
        sample_src_positions, seq_lens, draft_tokens, current_draft_step,
        hidden_states, hidden_size, max_model_len, num_speculative_steps,
        advance_draft_positions);
  } else {
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16,
                "rdna2 spec-decode: unsupported hidden-state dtype");
    launch_update_draft_inputs<at::BFloat16>(
        output_draft_tokens, next_input_hidden_states, input_ids, positions,
        sample_src_positions, seq_lens, draft_tokens, current_draft_step,
        hidden_states, hidden_size, max_model_len, num_speculative_steps,
        advance_draft_positions);
  }
}

void spec_gather_draft_sampled_rdna2(
    torch::Tensor draft_sampled, torch::Tensor pos, torch::Tensor input_ids,
    torch::Tensor positions, torch::Tensor logits_indices,
    torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
    torch::Tensor prefill_len) {
  check_dtype(draft_sampled, at::kInt, "draft_sampled");
  check_dtype(pos, at::kLong, "pos");
  check_dtype(input_ids, at::kInt, "input_ids");
  check_dtype(positions, at::kLong, "positions");
  check_dtype(logits_indices, at::kLong, "logits_indices");
  check_dtype(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  check_dtype(expanded_local_pos, at::kInt, "expanded_local_pos");
  check_dtype(prefill_len, at::kInt, "prefill_len");

  const int64_t num_logits = logits_indices.numel();
  if (num_logits == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(logits_indices));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 1024;
  gather_draft_sampled_kernel<<<(int)cdiv(num_logits, threads), threads, 0,
                                stream.stream()>>>(
      reinterpret_cast<int32_t*>(draft_sampled.data_ptr()),
      reinterpret_cast<int64_t*>(pos.data_ptr()),
      reinterpret_cast<const int32_t*>(input_ids.data_ptr()),
      reinterpret_cast<const int64_t*>(positions.data_ptr()),
      reinterpret_cast<const int64_t*>(logits_indices.data_ptr()),
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(expanded_local_pos.data_ptr()),
      reinterpret_cast<const int32_t*>(prefill_len.data_ptr()), num_logits);
  LAUNCH_CHECK();
}

void spec_flatten_sampled_rdna2(torch::Tensor flat_sampled, torch::Tensor sampled,
                           torch::Tensor num_sampled,
                           torch::Tensor cu_num_logits) {
  check_dtype(flat_sampled, at::kLong, "flat_sampled");
  check_dtype(sampled, at::kLong, "sampled");
  check_dtype(num_sampled, at::kInt, "num_sampled");
  check_dtype(cu_num_logits, at::kInt, "cu_num_logits");

  const int num_reqs = (int)num_sampled.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(sampled));
  auto stream = at::cuda::getCurrentCUDAStream();
  flatten_sampled_kernel<<<num_reqs, kScalarThreads, 0, stream.stream()>>>(
      reinterpret_cast<int64_t*>(flat_sampled.data_ptr()),
      reinterpret_cast<const int64_t*>(sampled.data_ptr()), sampled.stride(0),
      reinterpret_cast<const int32_t*>(num_sampled.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()));
  LAUNCH_CHECK();
}

template <typename T_TARGET, typename T_DRAFT>
static void launch_compute_local_logits_stats(
    const Tensor& target_local_argmax, Tensor& target_local_max,
    Tensor& target_local_sumexp, Tensor& draft_local_max,
    Tensor& draft_local_sumexp, const Tensor& target_logits,
    const std::optional<Tensor>& draft_logits, const Tensor& expanded_idx_mapping,
    const Tensor& expanded_local_pos, const Tensor& temp, int64_t vocab_size,
    int64_t num_speculative_steps, bool has_draft_logits) {
  const int64_t num_logits = expanded_idx_mapping.numel();
  const int64_t vocab_num_blocks = cdiv(vocab_size, 8192);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vec4 = wide_load_ok(target_logits, vocab_size);
  bool vec4_draft = vec4;
  if (has_draft_logits) {
    vec4_draft = wide_load_ok_2d(*draft_logits, vocab_size);
  }
  const T_DRAFT* draft_ptr =
      has_draft_logits
          ? reinterpret_cast<const T_DRAFT*>(draft_logits->data_ptr())
          : nullptr;
  const int64_t draft_stride_0 =
      has_draft_logits ? draft_logits->stride(0) : 0;
  const int64_t draft_stride_1 =
      has_draft_logits ? draft_logits->stride(1) : 0;
  dim3 grid((unsigned)num_logits, (unsigned)vocab_num_blocks);
  compute_local_logits_stats_kernel<T_TARGET, T_DRAFT><<<grid, kReduceThreads, 0,
                                               stream.stream()>>>(
      reinterpret_cast<int64_t*>(target_local_argmax.data_ptr()),
      target_local_argmax.stride(0),
      reinterpret_cast<float*>(target_local_max.data_ptr()),
      target_local_max.stride(0),
      reinterpret_cast<float*>(target_local_sumexp.data_ptr()),
      target_local_sumexp.stride(0),
      reinterpret_cast<float*>(draft_local_max.data_ptr()),
      draft_local_max.stride(0),
      reinterpret_cast<float*>(draft_local_sumexp.data_ptr()),
      draft_local_sumexp.stride(0),
      reinterpret_cast<const T_TARGET*>(target_logits.data_ptr()),
      target_logits.stride(0), draft_ptr, draft_stride_0, draft_stride_1,
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(expanded_local_pos.data_ptr()),
      reinterpret_cast<const float*>(temp.data_ptr()), (int32_t)vocab_size,
      (int32_t)num_speculative_steps, has_draft_logits, vec4, vec4_draft);
  LAUNCH_CHECK();
}

void spec_compute_local_logits_stats_rdna2(
    torch::Tensor target_local_argmax, torch::Tensor target_local_max,
    torch::Tensor target_local_sumexp, torch::Tensor draft_local_max,
    torch::Tensor draft_local_sumexp, torch::Tensor target_logits,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
    torch::Tensor temp, int64_t vocab_size, int64_t num_speculative_steps,
    bool has_draft_logits) {
  check_dtype(target_local_argmax, at::kLong, "target_local_argmax");
  check_dtype(target_local_max, at::kFloat, "target_local_max");
  check_dtype(target_local_sumexp, at::kFloat, "target_local_sumexp");
  check_dtype(draft_local_max, at::kFloat, "draft_local_max");
  check_dtype(draft_local_sumexp, at::kFloat, "draft_local_sumexp");
  TORCH_CHECK(target_logits.scalar_type() == at::kFloat ||
                  target_logits.scalar_type() == at::kHalf ||
                  target_logits.scalar_type() == at::kBFloat16,
              "rdna2 spec-decode: unsupported target logits dtype");
  check_dtype(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  check_dtype(expanded_local_pos, at::kInt, "expanded_local_pos");
  check_dtype(temp, at::kFloat, "temp");

  const int64_t num_logits = expanded_idx_mapping.numel();
  if (num_logits == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  dispatch_float_dtype(target_logits, [&](auto ttag) {
    using T_TARGET = decltype(ttag);
    if (!has_draft_logits) {
      launch_compute_local_logits_stats<T_TARGET, float>(
          target_local_argmax, target_local_max, target_local_sumexp,
          draft_local_max, draft_local_sumexp, target_logits, draft_logits,
          expanded_idx_mapping, expanded_local_pos, temp, vocab_size,
          num_speculative_steps, false);
    } else if (draft_logits->scalar_type() == at::kFloat) {
      launch_compute_local_logits_stats<T_TARGET, float>(
          target_local_argmax, target_local_max, target_local_sumexp,
          draft_local_max, draft_local_sumexp, target_logits, draft_logits,
          expanded_idx_mapping, expanded_local_pos, temp, vocab_size,
          num_speculative_steps, true);
    } else if (draft_logits->scalar_type() == at::kHalf) {
      launch_compute_local_logits_stats<T_TARGET, at::Half>(
          target_local_argmax, target_local_max, target_local_sumexp,
          draft_local_max, draft_local_sumexp, target_logits, draft_logits,
          expanded_idx_mapping, expanded_local_pos, temp, vocab_size,
          num_speculative_steps, true);
    } else {
      TORCH_CHECK(draft_logits->scalar_type() == at::kBFloat16,
                  "rdna2 spec-decode: unsupported draft logits dtype");
      launch_compute_local_logits_stats<T_TARGET, at::BFloat16>(
          target_local_argmax, target_local_max, target_local_sumexp,
          draft_local_max, draft_local_sumexp, target_logits, draft_logits,
          expanded_idx_mapping, expanded_local_pos, temp, vocab_size,
          num_speculative_steps, true);
    }
  });
}

template <typename T_TARGET, typename T_DRAFT>
static void launch_rejection(
    const Tensor& sampled, Tensor& rejected_steps,
    Tensor& target_rejected_logsumexp, Tensor& draft_rejected_logsumexp,
    const Tensor& target_logits, const Tensor& target_local_argmax,
    const Tensor& target_local_max, const Tensor& target_local_sumexp,
    const Tensor& draft_sampled, const std::optional<Tensor>& draft_logits,
    const Tensor& draft_local_max, const Tensor& draft_local_sumexp,
    const Tensor& cu_num_logits, const Tensor& idx_mapping,
    const Tensor& temperature, const Tensor& seed, const Tensor& pos,
    const std::optional<Tensor>& synthetic_conditional_rates,
    const std::optional<Tensor>& cumulative_log_p,
    const std::optional<Tensor>& local_residual_mass, int64_t vocab_num_blocks,
    int64_t padded_vocab_num_blocks, bool has_draft_logits,
    bool synthetic_mode, bool use_block_verification) {
  const int num_reqs = (int)idx_mapping.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const T_DRAFT* draft_ptr =
      has_draft_logits
          ? reinterpret_cast<const T_DRAFT*>(draft_logits->data_ptr())
          : nullptr;
  const float* synthetic_ptr =
      synthetic_mode
          ? reinterpret_cast<const float*>(
                synthetic_conditional_rates->data_ptr())
          : nullptr;
  const float* cumulative_ptr =
      cumulative_log_p.has_value()
          ? reinterpret_cast<const float*>(cumulative_log_p->data_ptr())
          : nullptr;
  const float* residual_ptr =
      local_residual_mass.has_value()
          ? reinterpret_cast<const float*>(local_residual_mass->data_ptr())
          : nullptr;
  const int64_t residual_stride =
      local_residual_mass.has_value() ? local_residual_mass->stride(0) : 0;
  rejection_kernel<T_TARGET, T_DRAFT><<<num_reqs, kScalarThreads, 0, stream.stream()>>>(
      reinterpret_cast<int64_t*>(sampled.data_ptr()), sampled.stride(0),
      reinterpret_cast<int32_t*>(rejected_steps.data_ptr()),
      reinterpret_cast<float*>(target_rejected_logsumexp.data_ptr()),
      reinterpret_cast<float*>(draft_rejected_logsumexp.data_ptr()),
      reinterpret_cast<const T_TARGET*>(target_logits.data_ptr()),
      target_logits.stride(0),
      reinterpret_cast<const int64_t*>(target_local_argmax.data_ptr()),
      target_local_argmax.stride(0),
      reinterpret_cast<const float*>(target_local_max.data_ptr()),
      target_local_max.stride(0),
      reinterpret_cast<const float*>(target_local_sumexp.data_ptr()),
      target_local_sumexp.stride(0),
      reinterpret_cast<const int32_t*>(draft_sampled.data_ptr()), draft_ptr,
      has_draft_logits ? draft_logits->stride(0) : 0,
      has_draft_logits ? draft_logits->stride(1) : 0,
      reinterpret_cast<const float*>(draft_local_max.data_ptr()),
      draft_local_max.stride(0),
      reinterpret_cast<const float*>(draft_local_sumexp.data_ptr()),
      draft_local_sumexp.stride(0),
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
      reinterpret_cast<const int32_t*>(idx_mapping.data_ptr()),
      reinterpret_cast<const float*>(temperature.data_ptr()),
      reinterpret_cast<const int64_t*>(seed.data_ptr()),
      reinterpret_cast<const int64_t*>(pos.data_ptr()), synthetic_ptr,
      cumulative_ptr, residual_ptr, residual_stride,
      (int32_t)vocab_num_blocks, (int32_t)padded_vocab_num_blocks,
      has_draft_logits, synthetic_mode, use_block_verification);
  LAUNCH_CHECK();
}

void spec_rejection_rdna2(
    torch::Tensor sampled, torch::Tensor rejected_steps,
    torch::Tensor target_rejected_logsumexp,
    torch::Tensor draft_rejected_logsumexp, torch::Tensor target_logits,
    torch::Tensor target_local_argmax, torch::Tensor target_local_max,
    torch::Tensor target_local_sumexp, torch::Tensor draft_sampled,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor draft_local_max, torch::Tensor draft_local_sumexp,
    torch::Tensor cu_num_logits, torch::Tensor idx_mapping,
    torch::Tensor temperature, torch::Tensor seed, torch::Tensor pos,
    const std::optional<torch::Tensor>& synthetic_conditional_rates,
    const std::optional<torch::Tensor>& cumulative_log_p,
    const std::optional<torch::Tensor>& local_residual_mass,
    int64_t vocab_num_blocks, int64_t padded_vocab_num_blocks,
    bool has_draft_logits, bool synthetic_mode, bool use_block_verification) {
  check_dtype(sampled, at::kLong, "sampled");
  check_dtype(rejected_steps, at::kInt, "rejected_steps");
  check_dtype(target_rejected_logsumexp, at::kFloat,
              "target_rejected_logsumexp");
  check_dtype(draft_rejected_logsumexp, at::kFloat,
              "draft_rejected_logsumexp");
  TORCH_CHECK(target_logits.scalar_type() == at::kFloat ||
                  target_logits.scalar_type() == at::kHalf ||
                  target_logits.scalar_type() == at::kBFloat16,
              "rdna2 spec-decode: unsupported target logits dtype");
  check_dtype(target_local_argmax, at::kLong, "target_local_argmax");
  check_dtype(target_local_max, at::kFloat, "target_local_max");
  check_dtype(target_local_sumexp, at::kFloat, "target_local_sumexp");
  check_dtype(draft_sampled, at::kInt, "draft_sampled");
  check_dtype(cu_num_logits, at::kInt, "cu_num_logits");
  check_dtype(idx_mapping, at::kInt, "idx_mapping");
  check_dtype(temperature, at::kFloat, "temperature");
  check_dtype(seed, at::kLong, "seed");
  check_dtype(pos, at::kLong, "pos");
  TORCH_CHECK(padded_vocab_num_blocks <= 32,
              "rdna2 spec-decode: rejection_reduce needs at most 32 vocab "
              "blocks per warp");

  const int num_reqs = (int)idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  dispatch_float_dtype(target_logits, [&](auto ttag) {
    using T_TARGET = decltype(ttag);
    if (!has_draft_logits) {
      launch_rejection<T_TARGET, float>(
          sampled, rejected_steps, target_rejected_logsumexp,
          draft_rejected_logsumexp, target_logits, target_local_argmax,
          target_local_max, target_local_sumexp, draft_sampled, draft_logits,
          draft_local_max, draft_local_sumexp, cu_num_logits, idx_mapping,
          temperature, seed, pos, synthetic_conditional_rates,
          cumulative_log_p, local_residual_mass, vocab_num_blocks,
          padded_vocab_num_blocks, false, synthetic_mode,
          use_block_verification);
    } else if (draft_logits->scalar_type() == at::kFloat) {
      launch_rejection<T_TARGET, float>(
          sampled, rejected_steps, target_rejected_logsumexp,
          draft_rejected_logsumexp, target_logits, target_local_argmax,
          target_local_max, target_local_sumexp, draft_sampled, draft_logits,
          draft_local_max, draft_local_sumexp, cu_num_logits, idx_mapping,
          temperature, seed, pos, synthetic_conditional_rates,
          cumulative_log_p, local_residual_mass, vocab_num_blocks,
          padded_vocab_num_blocks, true, synthetic_mode,
          use_block_verification);
    } else if (draft_logits->scalar_type() == at::kHalf) {
      launch_rejection<T_TARGET, at::Half>(
          sampled, rejected_steps, target_rejected_logsumexp,
          draft_rejected_logsumexp, target_logits, target_local_argmax,
          target_local_max, target_local_sumexp, draft_sampled, draft_logits,
          draft_local_max, draft_local_sumexp, cu_num_logits, idx_mapping,
          temperature, seed, pos, synthetic_conditional_rates,
          cumulative_log_p, local_residual_mass, vocab_num_blocks,
          padded_vocab_num_blocks, true, synthetic_mode,
          use_block_verification);
    } else {
      TORCH_CHECK(draft_logits->scalar_type() == at::kBFloat16,
                  "rdna2 spec-decode: unsupported draft logits dtype");
      launch_rejection<T_TARGET, at::BFloat16>(
          sampled, rejected_steps, target_rejected_logsumexp,
          draft_rejected_logsumexp, target_logits, target_local_argmax,
          target_local_max, target_local_sumexp, draft_sampled, draft_logits,
          draft_local_max, draft_local_sumexp, cu_num_logits, idx_mapping,
          temperature, seed, pos, synthetic_conditional_rates,
          cumulative_log_p, local_residual_mass, vocab_num_blocks,
          padded_vocab_num_blocks, true, synthetic_mode,
          use_block_verification);
    }
  });
}

template <typename T_TARGET, typename T_DRAFT, bool USE_FP64>
static void launch_resample(
    const Tensor& resampled_local_argmax, Tensor& resampled_local_max,
    const Tensor& target_logits, const Tensor& target_rejected_logsumexp,
    const std::optional<Tensor>& draft_logits,
    const Tensor& draft_rejected_logsumexp, const Tensor& rejected_step,
    const Tensor& cu_num_logits, const Tensor& expanded_idx_mapping,
    const Tensor& draft_sampled, const Tensor& temperature, const Tensor& seed,
    const Tensor& pos, const std::optional<Tensor>& cumulative_log_p,
    int64_t vocab_size, bool has_draft_logits, bool use_block_verification) {
  const int64_t num_reqs = rejected_step.numel();
  const int64_t resample_num_blocks = cdiv(vocab_size, 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  const T_DRAFT* draft_ptr =
      has_draft_logits
          ? reinterpret_cast<const T_DRAFT*>(draft_logits->data_ptr())
          : nullptr;
  const float* cumulative_ptr =
      cumulative_log_p.has_value()
          ? reinterpret_cast<const float*>(cumulative_log_p->data_ptr())
          : nullptr;
  dim3 grid((unsigned)num_reqs, (unsigned)resample_num_blocks);
  resample_kernel<USE_FP64, T_TARGET, T_DRAFT><<<grid, kReduceThreads, 0,
                                       stream.stream()>>>(
      reinterpret_cast<int64_t*>(resampled_local_argmax.data_ptr()),
      resampled_local_argmax.stride(0), resampled_local_max.data_ptr(),
      resampled_local_max.stride(0),
      reinterpret_cast<const T_TARGET*>(target_logits.data_ptr()),
      target_logits.stride(0),
      reinterpret_cast<const float*>(target_rejected_logsumexp.data_ptr()),
      draft_ptr, has_draft_logits ? draft_logits->stride(0) : 0,
      has_draft_logits ? draft_logits->stride(1) : 0,
      reinterpret_cast<const float*>(draft_rejected_logsumexp.data_ptr()),
      reinterpret_cast<const int32_t*>(rejected_step.data_ptr()),
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<const int32_t*>(draft_sampled.data_ptr()),
      reinterpret_cast<const float*>(temperature.data_ptr()),
      reinterpret_cast<const int64_t*>(seed.data_ptr()),
      reinterpret_cast<const int64_t*>(pos.data_ptr()), cumulative_ptr,
      (int32_t)vocab_size, has_draft_logits, use_block_verification);
  LAUNCH_CHECK();
}

template <typename T_TARGET, typename T_DRAFT>
static void dispatch_resample_dtype(
    const Tensor& resampled_local_argmax, Tensor& resampled_local_max,
    const Tensor& target_logits, const Tensor& target_rejected_logsumexp,
    const std::optional<Tensor>& draft_logits,
    const Tensor& draft_rejected_logsumexp, const Tensor& rejected_step,
    const Tensor& cu_num_logits, const Tensor& expanded_idx_mapping,
    const Tensor& draft_sampled, const Tensor& temperature, const Tensor& seed,
    const Tensor& pos, const std::optional<Tensor>& cumulative_log_p,
    int64_t vocab_size, bool has_draft_logits, bool use_fp64,
    bool use_block_verification) {
  if (use_fp64) {
    launch_resample<T_TARGET, T_DRAFT, true>(
        resampled_local_argmax, resampled_local_max, target_logits,
        target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
        rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
        temperature, seed, pos, cumulative_log_p, vocab_size, has_draft_logits,
        use_block_verification);
  } else {
    launch_resample<T_TARGET, T_DRAFT, false>(
        resampled_local_argmax, resampled_local_max, target_logits,
        target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
        rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
        temperature, seed, pos, cumulative_log_p, vocab_size, has_draft_logits,
        use_block_verification);
  }
}

void spec_resample_rdna2(
    torch::Tensor resampled_local_argmax, torch::Tensor resampled_local_max,
    torch::Tensor target_logits, torch::Tensor target_rejected_logsumexp,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor draft_rejected_logsumexp, torch::Tensor rejected_step,
    torch::Tensor cu_num_logits, torch::Tensor expanded_idx_mapping,
    torch::Tensor draft_sampled, torch::Tensor temperature, torch::Tensor seed,
    torch::Tensor pos, const std::optional<torch::Tensor>& cumulative_log_p,
    int64_t vocab_size, bool has_draft_logits, bool use_fp64,
    bool use_block_verification) {
  check_dtype(resampled_local_argmax, at::kLong, "resampled_local_argmax");
  TORCH_CHECK(target_logits.scalar_type() == at::kFloat ||
                  target_logits.scalar_type() == at::kHalf ||
                  target_logits.scalar_type() == at::kBFloat16,
              "rdna2 spec-decode: unsupported target logits dtype");
  check_dtype(target_rejected_logsumexp, at::kFloat,
              "target_rejected_logsumexp");
  check_dtype(draft_rejected_logsumexp, at::kFloat,
              "draft_rejected_logsumexp");
  check_dtype(rejected_step, at::kInt, "rejected_step");
  check_dtype(cu_num_logits, at::kInt, "cu_num_logits");
  check_dtype(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  check_dtype(draft_sampled, at::kInt, "draft_sampled");
  check_dtype(temperature, at::kFloat, "temperature");
  check_dtype(seed, at::kLong, "seed");
  check_dtype(pos, at::kLong, "pos");
  TORCH_CHECK(resampled_local_max.scalar_type() ==
                  (use_fp64 ? at::kDouble : at::kFloat),
              "rdna2 spec-decode: resampled_local_max dtype mismatch");

  const int64_t num_reqs = rejected_step.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  dispatch_float_dtype(target_logits, [&](auto ttag) {
    using T_TARGET = decltype(ttag);
    if (!has_draft_logits) {
      dispatch_resample_dtype<T_TARGET, float>(
          resampled_local_argmax, resampled_local_max, target_logits,
          target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
          rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
          temperature, seed, pos, cumulative_log_p, vocab_size, false, use_fp64,
          use_block_verification);
    } else if (draft_logits->scalar_type() == at::kFloat) {
      dispatch_resample_dtype<T_TARGET, float>(
          resampled_local_argmax, resampled_local_max, target_logits,
          target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
          rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
          temperature, seed, pos, cumulative_log_p, vocab_size, true, use_fp64,
          use_block_verification);
    } else if (draft_logits->scalar_type() == at::kHalf) {
      dispatch_resample_dtype<T_TARGET, at::Half>(
          resampled_local_argmax, resampled_local_max, target_logits,
          target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
          rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
          temperature, seed, pos, cumulative_log_p, vocab_size, true, use_fp64,
          use_block_verification);
    } else {
      TORCH_CHECK(draft_logits->scalar_type() == at::kBFloat16,
                  "rdna2 spec-decode: unsupported draft logits dtype");
      dispatch_resample_dtype<T_TARGET, at::BFloat16>(
          resampled_local_argmax, resampled_local_max, target_logits,
          target_rejected_logsumexp, draft_logits, draft_rejected_logsumexp,
          rejected_step, cu_num_logits, expanded_idx_mapping, draft_sampled,
          temperature, seed, pos, cumulative_log_p, vocab_size, true, use_fp64,
          use_block_verification);
    }
  });
}

template <typename T_MAX>
static void launch_insert_resampled(
    Tensor& sampled, Tensor& num_sampled, const Tensor& resampled_local_argmax,
    const Tensor& resampled_local_max, int64_t resample_num_blocks,
    const Tensor& cu_num_logits, const Tensor& expanded_idx_mapping,
    const Tensor& temperature) {
  const int num_reqs = (int)num_sampled.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  insert_resampled_kernel<T_MAX><<<num_reqs, kScalarThreads, 0,
                                   stream.stream()>>>(
      reinterpret_cast<int64_t*>(sampled.data_ptr()), sampled.stride(0),
      reinterpret_cast<int32_t*>(num_sampled.data_ptr()),
      reinterpret_cast<const int64_t*>(resampled_local_argmax.data_ptr()),
      resampled_local_argmax.stride(0),
      reinterpret_cast<const T_MAX*>(resampled_local_max.data_ptr()),
      resampled_local_max.stride(0), (int32_t)resample_num_blocks,
      reinterpret_cast<const int32_t*>(cu_num_logits.data_ptr()),
      reinterpret_cast<const int32_t*>(expanded_idx_mapping.data_ptr()),
      reinterpret_cast<const float*>(temperature.data_ptr()));
  LAUNCH_CHECK();
}

void spec_insert_resampled_rdna2(
    torch::Tensor sampled, torch::Tensor num_sampled,
    torch::Tensor resampled_local_argmax, torch::Tensor resampled_local_max,
    int64_t resample_num_blocks, torch::Tensor cu_num_logits,
    torch::Tensor expanded_idx_mapping, torch::Tensor temperature) {
  check_dtype(sampled, at::kLong, "sampled");
  check_dtype(num_sampled, at::kInt, "num_sampled");
  check_dtype(resampled_local_argmax, at::kLong, "resampled_local_argmax");
  check_dtype(cu_num_logits, at::kInt, "cu_num_logits");
  check_dtype(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  check_dtype(temperature, at::kFloat, "temperature");

  const int num_reqs = (int)num_sampled.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(sampled));
  if (resampled_local_max.scalar_type() == at::kFloat) {
    launch_insert_resampled<float>(sampled, num_sampled,
                                   resampled_local_argmax, resampled_local_max,
                                   resample_num_blocks, cu_num_logits,
                                   expanded_idx_mapping, temperature);
  } else {
    TORCH_CHECK(resampled_local_max.scalar_type() == at::kDouble,
                "rdna2 spec-decode: resampled_local_max dtype mismatch");
    launch_insert_resampled<double>(sampled, num_sampled,
                                    resampled_local_argmax,
                                    resampled_local_max, resample_num_blocks,
                                    cu_num_logits, expanded_idx_mapping,
                                    temperature);
  }
}

