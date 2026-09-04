// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA (Kimi Delta Attention) packed single-token decode
// kernel for AMD RDNA2 (gfx1030). Hand implementation of the torch
// reference in modeling_glm5_next.py:
//   * causal_conv1d_update            (lines 373-390)
//   * Glm5NextTextForgetGate          (lines 304-334)
//   * l2norm                          (lines 416-424, eps = 1e-6 fixed)
//   * recurrent_kimi_delta_attention  (lines 427-478, recurrence 464-476)
//   * Glm5NextTextRMSNormGated        (lines 338-358)
//
// One workgroup per (batch, head): 1024 threads = 128 v-rows x 8 k-slices;
// thread tid owns v_local = tid >> 3 (0..127), ks = tid & 7, a 16-wide
// k-slice k0 = ks*16, i.e. the fp32 state registers h[k0:k0+16, v_local]
// of the head's [D, D] = [128, 128] state tile in [k, v] ordering (NOTE:
// this is transposed vs the Qwen GDN kernel's [V, K] ordering -- GLM's
// reference state is [B, H, K, V], modeling_glm5_next.py:457-458). All K
// reductions (q/k L2 norms, kv_mem matvec, output matvec) are wave-local
// __shfl_xor over the 8 k-slice lanes (4 v-groups per 32-lane wave).
//
// Phases:
//   0. causal conv update: shift conv_state[slot, c, 0..2] by one tap,
//      y = silu(sum_w conv_state_old[w]*conv_w[c,w] + x*conv_w[c,3]),
//      fp16 round-trip of y for HF parity (reference conv runs in fp16).
//   1. l2norm q, k in fp32 with the fixed FLA eps 1e-6 (NOT norm_eps),
//      then q *= 1/sqrt(128) (reference query*scale, line 451-452).
//   2. gate = lower_bound * sigmoid(exp(A_log[h]) * (f + dt_bias)) per
//      (head, k-dim) (line 328); beta = sigmoid(b) with fp16 round-trip
//      (torch sigmoid of an fp16 b_proj output is computed in fp16).
//   3. recurrence (lines 471-476): h[k,:] *= exp(gate[k]);
//      kv_mem[v] = sum_k h[k,v]*k[k]; delta = (v - kv_mem)*beta;
//      h[k,v] += k[k]*delta[v]; o[v] = sum_k h[k,v]*q[k].
//   4. RMSNormGated over D=128 per head (lines 346-358): fp32 variance,
//      rsqrt(var + norm_eps), * norm_w (fp32), * sigmoid(out_gate) --
//      the gate is widened fp16 -> fp32 and sigmoided in fp32 (NO fp16
//      round-trip; reference gate.to(torch.float32)), result fp16.
//
// Invalid state_indices (< 0 or >= slots): the head's 128 output rows are
// zeroed and BOTH states (conv_state, ssm_state) are left untouched. The
// guard depends only on i_n, so the early return is workgroup-uniform.
//
// Differences from gdn_decode_rdna2.cu (kept for reviewer orientation):
//   * state ordering [k, v] (GDN: [V, K]);
//   * per-(head, k-dim) gate vector (GDN: per-head scalar);
//   * no GQA (H k-heads == H v-heads == H);
//   * fused depthwise conv (width 4) + silu in-kernel;
//   * lower_bound sigmoid gate formula (GDN: softplus);
//   * fused RMSNormGated epilogue;
//   * 1024 threads, one workgroup per (batch, head);
//   * no __launch_bounds__ / amdgpu_waves_per_eu occupancy pins
//     (project rule; occupancy-first, tune later).
//
// UNVERIFIED NOTES (first cut, no HIP compile run yet):
//   * fp16 round-trip parity of silu/sigmoid assumed to match torch's
//     fp16 elementwise kernels bitwise on ROCm; if not, outputs differ by
//     <= 1 ulp fp16.
//   * 1/sqrtf used for both norms; torch uses sqrt+divide -- <= 1 ulp fp32.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_KDA_D = 128;          // head dim (K == V == D)
constexpr int GLM5_KDA_CONV_W = 4;       // causal depthwise conv width
constexpr int GLM5_KDA_CONV_STATE = 3;   // conv taps kept between steps
constexpr int GLM5_KDA_THREADS = 1024;   // 128 v-rows x 8 k-slices
constexpr float GLM5_KDA_L2NORM_EPS = 1e-6f;  // fixed FLA l2norm eps
constexpr float GLM5_KDA_QK_SCALE =
    0.08838834764831845f;  // 1/sqrt(128)

// Reduce across the 8 k-slice lanes sharing one v-row. Lane layout is
// v_local = lane >> 3, ks = lane & 7 inside each 32-lane wave (4 v-groups
// per wave), so xor masks 1/2/4 stay inside the group.
__device__ __forceinline__ float glm5_kda_ksum(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

__global__ void glm5_kda_decode_rdna2_kernel(
    const __half* __restrict__ qkv_raw,    // [B, 3*H*D] pre-conv q|k|v
    const __half* __restrict__ conv_w,     // [3*H*D, 4]
    __half* __restrict__ conv_state,       // [slots, 3*H*D, 3] in-place
    const float* __restrict__ A_log,       // [H]
    const float* __restrict__ dt_bias,     // [H*D]
    const __half* __restrict__ f,          // [B, H*D] forget pre-activation
    const __half* __restrict__ beta_raw,   // [B, H] pre-sigmoid
    const __half* __restrict__ out_gate,   // [B, H*D] pre-sigmoid
    const float* __restrict__ norm_w,      // [D]
    const int* __restrict__ state_indices, // [B]
    float* __restrict__ ssm_state,         // [slots, H, D, D] (k, v)
    __half* __restrict__ out,              // [B, H*D]
    long stride_qkv_tok, long stride_f_tok, long stride_beta_tok,
    long stride_gate_tok, long stride_indices_seq, long num_slots,
    long stride_conv_slot, long stride_state_slot, int H,
    float lower_bound, float norm_eps) {
  const int i_h = blockIdx.x;
  const int i_n = blockIdx.y;
  const int tid = threadIdx.x;

  // [k, v] thread decomposition (uniform for the whole workgroup).
  const int v_local = tid >> 3;   // 0..127
  const int ks = tid & 7;         // 0..7
  const int o_v = v_local;        // one workgroup covers all V rows
  const int k0 = ks * 16;

  // NULL-slot guard: zero this head's outputs, leave both states alone.
  const long slot = (long)state_indices[i_n * stride_indices_seq];
  if (slot < 0 || slot >= num_slots) {
    if (tid < GLM5_KDA_D) {
      out[(long)i_n * H * GLM5_KDA_D + (long)i_h * GLM5_KDA_D + tid] =
          __float2half(0.0f);
    }
    return;
  }

  __shared__ __half s_qkv[3 * GLM5_KDA_D];  // post-conv silu'd q|k|v (fp16)
  __shared__ float s_decay[GLM5_KDA_D];     // exp(gate[k]) per k-dim
  __shared__ float s_o[GLM5_KDA_D];         // pre-norm output
  __shared__ float s_beta;                  // sigmoid'd beta (fp16 parity)
  __shared__ float s_red[40];               // RMS reduction scratch

  // ---- Phase 0: causal depthwise conv update + silu -------------------
  // Only 3*D = 384 channels per head; lanes 0..383 work, the rest idle.
  for (int idx = tid; idx < 3 * GLM5_KDA_D; idx += GLM5_KDA_THREADS) {
    const int block = idx / GLM5_KDA_D;         // 0=q, 1=k, 2=v
    const int d = idx - block * GLM5_KDA_D;
    const long c_global =
        (long)block * H * GLM5_KDA_D + (long)i_h * GLM5_KDA_D + d;
    __half* p_cs = conv_state + slot * stride_conv_slot + c_global * 3;
    const __half x_h = qkv_raw[(long)i_n * stride_qkv_tok + c_global];
    const float s0 = __half2float(p_cs[0]);
    const float s1 = __half2float(p_cs[1]);
    const float s2 = __half2float(p_cs[2]);
    const float w0 = __half2float(conv_w[c_global * GLM5_KDA_CONV_W + 0]);
    const float w1 = __half2float(conv_w[c_global * GLM5_KDA_CONV_W + 1]);
    const float w2 = __half2float(conv_w[c_global * GLM5_KDA_CONV_W + 2]);
    const float w3 = __half2float(conv_w[c_global * GLM5_KDA_CONV_W + 3]);
    const float acc =
        s0 * w0 + s1 * w1 + s2 * w2 + __half2float(x_h) * w3;
    // Shift the tap history, then store the new sample raw (fp16).
    p_cs[0] = p_cs[1];
    p_cs[1] = p_cs[2];
    p_cs[2] = x_h;
    // silu in fp32, fp16 round-trip (reference conv output is fp16).
    const float y = acc / (1.0f + expf(-acc));
    s_qkv[idx] = __float2half(y);
  }
  __syncthreads();

  // ---- Phase 1: load + l2norm q, k for this thread's 16-wide k-slice --
  float q[16], kk[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    q[j] = __half2float(s_qkv[k0 + j]);
    kk[j] = __half2float(s_qkv[GLM5_KDA_D + k0 + j]);
  }
  float sq = 0.0f, sk = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    sq += q[j] * q[j];
    sk += kk[j] * kk[j];
  }
  sq = glm5_kda_ksum(sq);
  sk = glm5_kda_ksum(sk);
  // Fixed FLA eps 1e-6 (l2norm, modeling_glm5_next.py:445); scale folded
  // into q exactly like the reference's query = l2norm(query) * scale.
  const float q_scale = GLM5_KDA_QK_SCALE / sqrtf(sq + GLM5_KDA_L2NORM_EPS);
  const float k_scale = 1.0f / sqrtf(sk + GLM5_KDA_L2NORM_EPS);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    q[j] *= q_scale;
    kk[j] *= k_scale;
  }

  // ---- Phase 2: per-(head, k-dim) decay, beta, v value ----------------
  if (tid == 0) {
    const float b_val =
        __half2float(beta_raw[(long)i_n * stride_beta_tok + i_h]);
    // torch sigmoid on fp16 computes in fp16 -> round-trip parity.
    s_beta = __half2float(__float2half(1.0f / (1.0f + expf(-b_val))));
  }
  if (tid < GLM5_KDA_D) {
    const int d = tid;
    const float g_raw =
        __half2float(f[(long)i_n * stride_f_tok + (long)i_h * GLM5_KDA_D + d]) +
        dt_bias[(long)i_h * GLM5_KDA_D + d];
    const float decay_rate = expf(A_log[i_h]);
    const float gate =
        lower_bound / (1.0f + expf(-decay_rate * g_raw));  // lb*sigmoid
    s_decay[d] = expf(gate);
  }
  const float vv = __half2float(s_qkv[2 * GLM5_KDA_D + o_v]);
  __syncthreads();
  const float beta_v = s_beta;

  // ---- Phase 3: delta-rule recurrence on h[k0:k0+16, o_v] -------------
  float* p_h = ssm_state + slot * stride_state_slot +
               (long)i_h * GLM5_KDA_D * GLM5_KDA_D +
               (long)k0 * GLM5_KDA_D + o_v;
  float h[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] = p_h[(long)j * GLM5_KDA_D];

  // decay: h[k, :] *= exp(gate[k])  (per k-row)
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] *= s_decay[k0 + j];

  // kv_mem[v] = sum_k h[k, v] * k[k]
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += h[j] * kk[j];
  acc = glm5_kda_ksum(acc);

  const float delta_v = (vv - acc) * beta_v;
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] += kk[j] * delta_v;

  // o[v] = sum_k h[k, v] * q[k]
  float o_acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc += h[j] * q[j];
  o_acc = glm5_kda_ksum(o_acc);

  // Exclusive ownership: this (batch, head) tile is written by exactly
  // one workgroup -- no atomics needed.
#pragma unroll
  for (int j = 0; j < 16; ++j) p_h[(long)j * GLM5_KDA_D] = h[j];

  if (ks == 0) s_o[o_v] = o_acc;
  __syncthreads();

  // ---- Phase 4: RMSNormGated over D per head ---------------------------
  if (tid < GLM5_KDA_D) {
    const float val = s_o[tid];
    float sumsq = val * val;
    sumsq += __shfl_xor_sync(0xffffffffffffffffULL, sumsq, 1);
    sumsq += __shfl_xor_sync(0xffffffffffffffffULL, sumsq, 2);
    sumsq += __shfl_xor_sync(0xffffffffffffffffULL, sumsq, 4);
    sumsq += __shfl_xor_sync(0xffffffffffffffffULL, sumsq, 8);
    sumsq += __shfl_xor_sync(0xffffffffffffffffULL, sumsq, 16);
    if ((tid & 31) == 0) s_red[tid >> 5] = sumsq;
  }
  __syncthreads();
  if (tid == 0) {
    const float total = s_red[0] + s_red[1] + s_red[2] + s_red[3];
    s_red[4] = 1.0f / sqrtf(total / GLM5_KDA_D + norm_eps);
  }
  __syncthreads();
  if (tid < GLM5_KDA_D) {
    const float rstd = s_red[4];
    // gate sigmoided in fp32 on the widened fp16 input (no round-trip;
    // reference: ACT2FN["sigmoid"](gate.to(torch.float32))).
    const float gate = 1.0f / (1.0f + expf(-__half2float(
        out_gate[(long)i_n * stride_gate_tok + (long)i_h * GLM5_KDA_D + tid])));
    out[(long)i_n * H * GLM5_KDA_D + (long)i_h * GLM5_KDA_D + tid] =
        __float2half(s_o[tid] * rstd * norm_w[tid] * gate);
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- public symbol registered as
// torch.ops._rocm_C.glm5_kda_decode_rdna2 (see report / torch_bindings).
// Argument order is exactly the DESIGN.md "K (KDA) ops" contract.
// ============================================================================

void glm5_kda_decode_rdna2(torch::Tensor qkv_raw, torch::Tensor conv_w,
                           torch::Tensor conv_state, torch::Tensor A_log,
                           torch::Tensor dt_bias, torch::Tensor f,
                           torch::Tensor beta, torch::Tensor out_gate,
                           torch::Tensor norm_w, torch::Tensor state_indices,
                           torch::Tensor ssm_state, torch::Tensor out,
                           double lower_bound, double norm_eps) {
  TORCH_CHECK(qkv_raw.dim() == 2 && qkv_raw.stride(-1) == 1 &&
                  qkv_raw.scalar_type() == at::kHalf,
              "qkv_raw must be fp16 [B, 3*H*D], contiguous in last dim");
  TORCH_CHECK(conv_w.dim() == 2 && conv_w.is_contiguous() &&
                  conv_w.scalar_type() == at::kHalf &&
                  conv_w.size(1) == GLM5_KDA_CONV_W,
              "conv_w must be contiguous fp16 [3*H*D, 4]");
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.stride(-1) == 1 &&
                  conv_state.scalar_type() == at::kHalf &&
                  conv_state.size(2) == GLM5_KDA_CONV_STATE,
              "conv_state must be fp16 [slots, 3*H*D, 3], contiguous in "
              "last dim");
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  A_log.scalar_type() == at::kFloat,
              "A_log must be contiguous fp32 [H]");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  dt_bias.scalar_type() == at::kFloat,
              "dt_bias must be contiguous fp32 [H*D]");
  TORCH_CHECK(f.dim() == 2 && f.stride(-1) == 1 &&
                  f.scalar_type() == at::kHalf,
              "f must be fp16 [B, H*D], contiguous in last dim");
  TORCH_CHECK(beta.dim() == 2 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kHalf,
              "beta must be fp16 [B, H], contiguous in last dim");
  TORCH_CHECK(out_gate.dim() == 2 && out_gate.stride(-1) == 1 &&
                  out_gate.scalar_type() == at::kHalf,
              "out_gate must be fp16 [B, H*D], contiguous in last dim");
  TORCH_CHECK(norm_w.dim() == 1 && norm_w.is_contiguous() &&
                  norm_w.scalar_type() == at::kFloat,
              "norm_w must be contiguous fp32 [D]");
  TORCH_CHECK(state_indices.dim() == 1 &&
                  state_indices.scalar_type() == at::kInt,
              "state_indices must be int32 [B]");
  TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.stride(-1) == 1 &&
                  ssm_state.scalar_type() == at::kFloat,
              "ssm_state must be fp32 [slots, H, D, D], contiguous in last "
              "dim");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() &&
                  out.scalar_type() == at::kHalf,
              "out must be contiguous fp16 [B, H*D]");

  const long B = qkv_raw.size(0);
  const long qkv_dim = qkv_raw.size(1);
  TORCH_CHECK(qkv_dim % (3 * GLM5_KDA_D) == 0,
              "qkv_raw last dim must be divisible by 3*D");
  const long H = qkv_dim / (3 * GLM5_KDA_D);
  TORCH_CHECK(H >= 1, "glm5_kda_decode_rdna2 needs at least one head");
  TORCH_CHECK(conv_w.size(0) == qkv_dim, "conv_w rows must equal 3*H*D");
  TORCH_CHECK(conv_state.size(0) == ssm_state.size(0),
              "conv_state and ssm_state slot counts must match");
  TORCH_CHECK(conv_state.size(1) == qkv_dim,
              "conv_state channel dim must equal 3*H*D");
  TORCH_CHECK(A_log.size(0) == H, "A_log size mismatch");
  TORCH_CHECK(dt_bias.size(0) == H * GLM5_KDA_D, "dt_bias size mismatch");
  TORCH_CHECK(f.size(0) == B && f.size(1) == H * GLM5_KDA_D,
              "f shape mismatch");
  TORCH_CHECK(beta.size(0) == B && beta.size(1) == H, "beta shape mismatch");
  TORCH_CHECK(out_gate.size(0) == B && out_gate.size(1) == H * GLM5_KDA_D,
              "out_gate shape mismatch");
  TORCH_CHECK(norm_w.size(0) == GLM5_KDA_D, "norm_w size mismatch");
  TORCH_CHECK(state_indices.size(0) == B, "state_indices shape mismatch");
  TORCH_CHECK(ssm_state.size(1) == H && ssm_state.size(2) == GLM5_KDA_D &&
                  ssm_state.size(3) == GLM5_KDA_D,
              "ssm_state shape mismatch (expected [slots, H, 128, 128])");
  TORCH_CHECK(out.size(0) == B && out.size(1) == H * GLM5_KDA_D,
              "out shape mismatch");
  if (B == 0) return;

  const at::cuda::OptionalCUDAGuard guard(qkv_raw.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned int)H, (unsigned int)B);
  glm5_kda_decode_rdna2_kernel<<<grid, GLM5_KDA_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(qkv_raw.data_ptr()),
      reinterpret_cast<const __half*>(conv_w.data_ptr()),
      reinterpret_cast<__half*>(conv_state.data_ptr()),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      reinterpret_cast<const __half*>(f.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()),
      reinterpret_cast<const __half*>(out_gate.data_ptr()),
      norm_w.data_ptr<float>(), state_indices.data_ptr<int>(),
      ssm_state.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr()), qkv_raw.stride(0),
      f.stride(0), beta.stride(0), out_gate.stride(0),
      state_indices.stride(0), conv_state.size(0), conv_state.stride(0),
      ssm_state.stride(0), (int)H, (float)lower_bound, (float)norm_eps);
}
