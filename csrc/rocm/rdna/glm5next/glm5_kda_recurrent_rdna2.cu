// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Post-conv KDA recurrence for Glm5Next on gfx1030.
//
// Upstream ``fused_recurrent_kda`` already receives q/k/v after the causal
// conv and writes the pre-norm core output; ``o_norm`` runs in the caller.
// This kernel is that recurrence only (l2norm, bounded gate, delta rule).
// It does not touch conv state and does not apply RMSNorm.
//
// State layout matches the Triton kernel: ``[slots, H, V, K]`` with K
// contiguous (``o_v * K + o_k``). NULL_BLOCK_ID (0) and PAD_SLOT_ID (-1)
// are empty: output is zeroed and the state is left untouched, matching
// fused_recurrent_kda (state_idx <= 0). Activations are fp16 or bf16.
// bf16 values are widened to fp32 with a bit shift and stored back with
// round-to-nearest-even. Dots are scalar fp32 multiplies. Packed bf16
// dots are not used.
//
// One wave32-friendly workgroup per (token, head): 1024 threads, 128 v-rows
// by 8 k-slices of 16. gfx1030 is wave32; the k-slice butterfly uses mask
// 0xffffffffffffffffULL.

#include <cstdint>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "glm5_kda_slot.cuh"

namespace {

constexpr int kD = 128;
constexpr int kThreads = 1024;
constexpr float kL2Eps = 1e-6f;
constexpr float kQkScale = 0.08838834764831845f;  // 1/sqrt(128)

enum class ActKind { kF16, kBF16 };

__device__ __forceinline__ float ksum8(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

__device__ __forceinline__ float load_f16(const void* p, long idx) {
  return __half2float(reinterpret_cast<const __half*>(p)[idx]);
}

__device__ __forceinline__ float load_bf16(const void* p, long idx) {
  const uint32_t bits =
      static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[idx]) << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ void store_f16(void* p, long idx, float v) {
  reinterpret_cast<__half*>(p)[idx] = __float2half(v);
}

__device__ __forceinline__ void store_bf16(void* p, long idx, float v) {
  const uint32_t u = __float_as_uint(v);
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t rounded = (u + 0x7fffu + lsb) >> 16;
  reinterpret_cast<uint16_t*>(p)[idx] = static_cast<uint16_t>(rounded);
}

template <ActKind Kind>
__device__ __forceinline__ float load_act(const void* p, long idx) {
  if constexpr (Kind == ActKind::kF16) {
    return load_f16(p, idx);
  } else {
    return load_bf16(p, idx);
  }
}

template <ActKind Kind>
__device__ __forceinline__ void store_act(void* p, long idx, float v) {
  if constexpr (Kind == ActKind::kF16) {
    store_f16(p, idx, v);
  } else {
    store_bf16(p, idx, v);
  }
}

template <ActKind Kind>
__global__ void glm5_kda_recurrent_rdna2_kernel(
    const void* __restrict__ q, const void* __restrict__ k,
    const void* __restrict__ v, const void* __restrict__ g,
    const void* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, const int* __restrict__ state_indices,
    float* __restrict__ ssm_state, void* __restrict__ out, long q_s0, long q_s1,
    long k_s0, long k_s1, long v_s0, long v_s1, long g_s0, long g_s1,
    long beta_s0, long beta_s1, long idx_s0, long state_s0, long state_s1,
    long out_s0, long out_s1, long num_slots, float lower_bound) {
  const int i_h = blockIdx.x;
  const int i_n = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_local = tid >> 3;
  const int ks = tid & 7;
  const int k0 = ks * 16;

  const long slot = static_cast<long>(state_indices[i_n * idx_s0]);
  if (glm5_kda_slot_is_empty(slot, num_slots)) {
    if (ks == 0) {
      store_act<Kind>(out, i_n * out_s0 + i_h * out_s1 + v_local, 0.0f);
    }
    return;
  }

  __shared__ float s_decay[kD];
  __shared__ float s_beta;

  float qv[16], kk[16];
  const long q_base = i_n * q_s0 + i_h * q_s1 + k0;
  const long k_base = i_n * k_s0 + i_h * k_s1 + k0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    qv[j] = load_act<Kind>(q, q_base + j);
    kk[j] = load_act<Kind>(k, k_base + j);
  }
  float sq = 0.0f, sk = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    sq += qv[j] * qv[j];
    sk += kk[j] * kk[j];
  }
  sq = ksum8(sq);
  sk = ksum8(sk);
  const float q_scale = kQkScale / sqrtf(sq + kL2Eps);
  const float k_scale = 1.0f / sqrtf(sk + kL2Eps);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    qv[j] *= q_scale;
    kk[j] *= k_scale;
  }

  if (tid == 0) {
    const float b_val = load_act<Kind>(beta, i_n * beta_s0 + i_h * beta_s1);
    s_beta = 1.0f / (1.0f + expf(-b_val));
  }
  if (tid < kD) {
    const float g_raw =
        load_act<Kind>(g, i_n * g_s0 + i_h * g_s1 + tid) +
        dt_bias[static_cast<long>(i_h) * kD + tid];
    const float amp = expf(A_log[i_h]);
    const float gate = lower_bound / (1.0f + expf(-amp * g_raw));
    s_decay[tid] = expf(gate);
  }
  const float vv = load_act<Kind>(v, i_n * v_s0 + i_h * v_s1 + v_local);
  __syncthreads();
  const float beta_v = s_beta;

  // [slots, H, V, K], K contiguous. This thread owns k0:k0+16 of one v row.
  float* row = ssm_state + slot * state_s0 +
               static_cast<long>(i_h) * state_s1 +
               static_cast<long>(v_local) * kD + k0;
  float h[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] = row[j];
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] *= s_decay[k0 + j];

  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += h[j] * kk[j];
  acc = ksum8(acc);
  const float delta_v = (vv - acc) * beta_v;
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] += kk[j] * delta_v;

  float o_acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc += h[j] * qv[j];
  o_acc = ksum8(o_acc);

#pragma unroll
  for (int j = 0; j < 16; ++j) row[j] = h[j];
  if (ks == 0) {
    store_act<Kind>(out, i_n * out_s0 + i_h * out_s1 + v_local, o_acc);
  }
}

// The decode path hands over [1, T, H, D] activations and [1, T, H] beta
// (fused_recurrent_kda's layout); the kernel indexes [T, H, D] and [T, H].
torch::Tensor squeeze_leading(const torch::Tensor& t, int64_t rank) {
  if (t.dim() == rank + 1) {
    TORCH_CHECK(t.size(0) == 1, "leading dimension must be 1, got ", t.size(0));
    return t.squeeze(0);
  }
  return t;
}

template <ActKind Kind>
void launch_recurrent(const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& g,
                      const torch::Tensor& beta, const torch::Tensor& A_log,
                      const torch::Tensor& dt_bias,
                      const torch::Tensor& state_indices,
                      torch::Tensor& ssm_state, torch::Tensor& out,
                      float lower_bound, cudaStream_t stream) {
  const long B = q.size(0);
  const long H = q.size(1);
  dim3 grid(static_cast<unsigned>(H), static_cast<unsigned>(B));
  glm5_kda_recurrent_rdna2_kernel<Kind><<<grid, kThreads, 0, stream>>>(
          q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
          beta.data_ptr(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
          state_indices.data_ptr<int>(), ssm_state.data_ptr<float>(),
          out.data_ptr(), q.stride(0), q.stride(1), k.stride(0), k.stride(1),
          v.stride(0), v.stride(1), g.stride(0), g.stride(1), beta.stride(0),
          beta.stride(1), state_indices.stride(0), ssm_state.stride(0),
          ssm_state.stride(1), out.stride(0), out.stride(1), ssm_state.size(0),
          lower_bound);
}

}  // namespace

void glm5_kda_recurrent_rdna2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor g, torch::Tensor beta,
                              torch::Tensor A_log, torch::Tensor dt_bias,
                              torch::Tensor state_indices,
                              torch::Tensor ssm_state, torch::Tensor out,
                              double lower_bound) {
  q = squeeze_leading(q, 3);
  k = squeeze_leading(k, 3);
  v = squeeze_leading(v, 3);
  g = squeeze_leading(g, 3);
  beta = squeeze_leading(beta, 2);
  out = squeeze_leading(out, 3);

  const auto dt = q.scalar_type();
  TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16,
              "q must be fp16 or bf16");
  TORCH_CHECK(k.scalar_type() == dt && v.scalar_type() == dt &&
                  g.scalar_type() == dt && beta.scalar_type() == dt &&
                  out.scalar_type() == dt,
              "k/v/g/beta/out must match q dtype");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kD && q.stride(2) == 1,
              "q must be [B, H, 128] with contiguous last dim");
  TORCH_CHECK(k.sizes() == q.sizes() && k.stride(2) == 1, "k shape mismatch");
  TORCH_CHECK(v.sizes() == q.sizes() && v.stride(2) == 1, "v shape mismatch");
  TORCH_CHECK(g.sizes() == q.sizes() && g.stride(2) == 1, "g shape mismatch");
  TORCH_CHECK(beta.dim() == 2 && beta.size(0) == q.size(0) &&
                  beta.size(1) == q.size(1),
              "beta must be [B, H]");
  TORCH_CHECK(out.sizes() == q.sizes() && out.stride(2) == 1,
              "out must be [B, H, 128] with contiguous last dim");
  TORCH_CHECK(A_log.is_contiguous() && A_log.scalar_type() == at::kFloat &&
                  A_log.numel() == q.size(1),
              "A_log must be contiguous fp32 [H]");
  TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.scalar_type() == at::kFloat &&
                  dt_bias.numel() == q.size(1) * kD,
              "dt_bias must be contiguous fp32 [H*128]");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == q.size(0) &&
                  state_indices.scalar_type() == at::kInt,
              "state_indices must be int32 [B]");
  TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.scalar_type() == at::kFloat &&
                  ssm_state.size(1) == q.size(1) && ssm_state.size(2) == kD &&
                  ssm_state.size(3) == kD && ssm_state.stride(3) == 1 &&
                  ssm_state.stride(2) == kD,
              "ssm_state must be fp32 [slots, H, 128, 128] with K contiguous");
  if (q.size(0) == 0) {
    return;
  }
  TORCH_CHECK(q.size(0) <= 65535 && q.size(1) <= 65535,
              "glm5_kda_recurrent_rdna2 grid exceeds 65535");

  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dt == at::kHalf) {
    launch_recurrent<ActKind::kF16>(q, k, v, g, beta, A_log, dt_bias,
                                    state_indices, ssm_state, out,
                                    static_cast<float>(lower_bound), stream);
  } else {
    launch_recurrent<ActKind::kBF16>(q, k, v, g, beta, A_log, dt_bias,
                                     state_indices, ssm_state, out,
                                     static_cast<float>(lower_bound), stream);
  }
}
