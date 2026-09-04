// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA chunked-prefill prep kernel for AMD RDNA2 (gfx1030).
// Replaces the torch reference path:
//   * causal_conv1d_fn          (modeling_glm5_next.py:393-413)
//   * Glm5NextTextForgetGate    (lines 304-334, lower_bound branch only)
//   * l2norm                    (lines 416-424, eps = 1e-6 fixed)
//   * g.cumsum(dim=-2)          (line 530, INCLUSIVE cumsum, reset at
//                                every chunk boundary)
//
// Workgroup = one (chunk, head); grid = (NT, H); 128 threads (4 wave32).
// Varlen-only contract (vLLM v1 always provides varlen metadata):
// cu_seqlens [N+1] int32 and chunk_indices [NT, 2] int32 are REQUIRED;
// a single sequence can be expressed as cu_seqlens = [0, L].
//
// Per (chunk, head) the workgroup does, for each of the q|k|v blocks
// (channel range block*H*D + i_h*D .. +D):
//   1. cooperatively stages 67 tokens x 128 channels of the block into
//      LDS (tokens t_base-3 .. t_base+63, i.e. the chunk plus the 3-tap
//      causal halo; positions before bos are zero-filled so the conv
//      padding is causal per SEGMENT, matching causal_conv1d_fn's left
//      padding of 3);
//   2. computes the width-4 causal depthwise conv
//      y[t] = sum_{w=0..3} conv_w[c, w] * x[t-3+w]  in fp32, silu, then
//      fp16 round-trip (reference conv/silu run in fp16);
//   3. for q and k only: l2norm in fp32 (eps 1e-6), token owned by a
//      pair of lanes (64 dims each, paired via __shfl_xor delta=1);
//      q is NOT scaled by 1/sqrt(128) here -- the scale is applied in
//      glm5_kda_prefill_o_rdna2 (like the GDN chain);
//   4. stores q_out/k_out/v_out [L, H, D] fp16 for valid tokens only
//      (tail tokens stay caller-zeroed and are unused downstream).
// Then:
//   5. g path: thread d = tid owns k-dim d and runs a serial 64-token
//      INCLUSIVE cumsum of gate = lower_bound * sigmoid(exp(A_log[h]) *
//      (f[t,h,d] + dt_bias[h,d])) WITHIN the chunk (resets at chunk
//      boundaries, line 530), writing g_out [L, H, D] fp32;
//   6. beta path: lanes 0..63 write beta_out[t, h] = sigmoid(beta_raw)
//      WITH fp16 round-trip (torch sigmoid of the fp16 b_proj output is
//      fp16). Tail tokens (>= eos) are skipped -> stay 0.0, which is
//      safe: k/v/q of tail tokens are 0 so k_beta/v_beta rows vanish
//      regardless of beta (documented: reference pads beta with 0 then
//      sigmoids to 0.5 -- irrelevant because the padded k/v rows are 0).
//
// UNVERIFIED NOTES (first cut): fp16 round-trip parity of silu/sigmoid
// vs torch assumed bitwise on ROCm (<= 1 ulp otherwise).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_PREP_BT = 64;        // chunk size
constexpr int GLM5_PREP_D = 128;        // head dim
constexpr int GLM5_PREP_CONV_W = 4;     // causal conv width
constexpr int GLM5_PREP_HALO = 3;       // conv taps before chunk start
constexpr int GLM5_PREP_THREADS = 128;  // 4 warps (wave32)
constexpr float GLM5_PREP_L2NORM_EPS = 1e-6f;  // fixed FLA l2norm eps

__global__ void glm5_kda_prefill_prep_rdna2_kernel(
    const __half* __restrict__ mixed_qkv,    // [L, 3*H*D] pre-conv q|k|v
    const __half* __restrict__ conv_w,       // [3*H*D, 4]
    const float* __restrict__ A_log,         // [H]
    const float* __restrict__ dt_bias,       // [H*D]
    const __half* __restrict__ f,            // [L, H*D]
    const __half* __restrict__ beta_raw,     // [L, H]
    __half* __restrict__ q_out,              // [L, H, D] fp16 out
    __half* __restrict__ k_out,              // [L, H, D] fp16 out
    __half* __restrict__ v_out,              // [L, H, D] fp16 out
    float* __restrict__ g_out,               // [L, H, D] fp32 out (cumsum)
    float* __restrict__ beta_out,            // [L, H] fp32 out (sigmoid)
    long stride_x_tok,                       // = 3*H*D
    long stride_f_tok,                       // = H*D
    long stride_beta_tok,                    // = H
    int H, float lower_bound,
    const int* __restrict__ cu_seqlens,      // [N+1]
    const int* __restrict__ chunk_indices) { // [NT, 2]
  const int i_tb = blockIdx.x;  // flat chunk index
  const int i_h = blockIdx.y;   // head
  const int tid = threadIdx.x;

  // Halo staging tile: rows 0..66 map to tokens t_base-3 .. t_base+63.
  __shared__ __half s_x[(GLM5_PREP_BT + GLM5_PREP_HALO) * GLM5_PREP_D];

  // ---- Resolve this chunk's (bos, eos, t_base, t_chunk) ---------------
  const int i_n = chunk_indices[i_tb * 2];
  const int i_t = chunk_indices[i_tb * 2 + 1];
  const long bos = (long)cu_seqlens[i_n];
  const long eos = (long)cu_seqlens[i_n + 1];
  const long t_base = bos + (long)i_t * GLM5_PREP_BT;
  const long remaining = eos - t_base;
  const int t_chunk =
      (remaining < GLM5_PREP_BT) ? (int)remaining : GLM5_PREP_BT;
  if (t_chunk <= 0) return;  // defensive; caller emits no such chunks

  // ---- Conv + silu (+l2norm for q/k) per block ------------------------
  // Token-pair decomposition: 2 lanes per token, 64 dims each.
  const int kv_token = tid >> 1;   // 0..63
  const int half = tid & 1;
  const int d_start = half * (GLM5_PREP_D / 2);

  for (int block = 0; block < 3; ++block) {
    const long c_block = (long)block * H * GLM5_PREP_D;

    // Stage 67 x 128 channels: tokens [t_base-3, t_base+63], zero OOB
    // before bos (causal per-segment padding) and after eos. (Plain
    // loop: 67 iterations, unrolling would bloat for no gain.)
    for (int it = 0;
         it < (GLM5_PREP_BT + GLM5_PREP_HALO) * GLM5_PREP_D /
                  GLM5_PREP_THREADS;
         ++it) {
      const int idx = tid + it * GLM5_PREP_THREADS;
      const int tok_rel = idx / GLM5_PREP_D;   // 0..66
      const int d = idx - tok_rel * GLM5_PREP_D;
      const long t_abs = t_base - GLM5_PREP_HALO + tok_rel;
      __half val = __float2half(0.0f);
      if (t_abs >= bos && t_abs < eos) {
        val = mixed_qkv[t_abs * stride_x_tok + c_block +
                        (long)i_h * GLM5_PREP_D + d];
      }
      s_x[idx] = val;
    }
    __syncthreads();

    // Conv: token kv_token lives at s_x row kv_token + HALO; tap
    // x[t-3+w] is at row (kv_token + HALO) - 3 + w = kv_token + w.
    float y_local[GLM5_PREP_D / 2];
#pragma unroll
    for (int j = 0; j < GLM5_PREP_D / 2; ++j) {
      const int d = d_start + j;
      const long c_global = c_block + (long)i_h * GLM5_PREP_D + d;
      const float w0 = __half2float(conv_w[c_global * GLM5_PREP_CONV_W + 0]);
      const float w1 = __half2float(conv_w[c_global * GLM5_PREP_CONV_W + 1]);
      const float w2 = __half2float(conv_w[c_global * GLM5_PREP_CONV_W + 2]);
      const float w3 = __half2float(conv_w[c_global * GLM5_PREP_CONV_W + 3]);
      const float x0 = __half2float(s_x[(kv_token + 0) * GLM5_PREP_D + d]);
      const float x1 = __half2float(s_x[(kv_token + 1) * GLM5_PREP_D + d]);
      const float x2 = __half2float(s_x[(kv_token + 2) * GLM5_PREP_D + d]);
      const float x3 = __half2float(s_x[(kv_token + 3) * GLM5_PREP_D + d]);
      const float acc = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
      // silu in fp32, fp16 round-trip, back to fp32 (reference parity).
      const float y32 = acc / (1.0f + expf(-acc));
      y_local[j] = __half2float(__float2half(y32));
    }

    // l2norm q and k over D=128: pair-reduce the two 64-dim halves.
    if (block < 2) {
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < GLM5_PREP_D / 2; ++j) ss += y_local[j] * y_local[j];
      ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, 1);
      const float inv = 1.0f / sqrtf(ss + GLM5_PREP_L2NORM_EPS);
#pragma unroll
      for (int j = 0; j < GLM5_PREP_D / 2; ++j) y_local[j] *= inv;
    }

    // Store (valid tokens only; tail stays caller-zeroed).
    if (kv_token < t_chunk) {
      const long t_abs = t_base + kv_token;
      __half* dst = (block == 0)   ? q_out
                    : (block == 1) ? k_out
                                   : v_out;
      const long off = (t_abs * H + i_h) * GLM5_PREP_D + d_start;
#pragma unroll
      for (int j = 0; j < GLM5_PREP_D / 2; ++j) {
        dst[off + j] = __float2half(y_local[j]);
      }
    }
    __syncthreads();  // before the next block overwrites s_x
  }

  // ---- g path: per-dim INCLUSIVE cumsum within the chunk --------------
  // Thread tid owns k-dim d = tid; serial over the chunk's tokens.
  {
    const int d = tid;
    const float dt_b = dt_bias[(long)i_h * GLM5_PREP_D + d];
    const float decay_rate = expf(A_log[i_h]);
    float running = 0.0f;
    for (int tok = 0; tok < t_chunk; ++tok) {
      const long t_abs = t_base + tok;
      const float g_raw =
          __half2float(f[t_abs * stride_f_tok + (long)i_h * GLM5_PREP_D + d]) +
          dt_b;
      // gate = lower_bound * sigmoid(decay_rate * g_raw); lower_bound is
      // negative -> gate <= 0 (log-decay). Softplus fallback branch of
      // the torch reference is NOT supported here: configs without
      // lower_bound must use the torch path (documented).
      const float gate = lower_bound / (1.0f + expf(-decay_rate * g_raw));
      running += gate;
      g_out[(t_abs * H + i_h) * GLM5_PREP_D + d] = running;
    }
  }

  // ---- beta path: one token per lane (lanes 0..63) --------------------
  if (tid < GLM5_PREP_BT && tid < t_chunk) {
    const long t_abs = t_base + tid;
    const float b_raw =
        __half2float(beta_raw[t_abs * stride_beta_tok + i_h]);
    // fp16 round-trip parity with torch sigmoid on the fp16 b_proj out.
    beta_out[t_abs * H + i_h] =
        __half2float(__float2half(1.0f / (1.0f + expf(-b_raw))));
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- public symbol torch.ops._rocm_C.glm5_kda_prefill_prep_rdna2.
// Argument order is exactly the DESIGN.md contract. Varlen-only: both
// cu_seqlens and chunk_indices must be defined.
// ============================================================================

void glm5_kda_prefill_prep_rdna2(
    torch::Tensor mixed_qkv, torch::Tensor conv_w, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor f, torch::Tensor beta_raw,
    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
    torch::Tensor g_out, torch::Tensor beta_out, torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices, double lower_bound) {
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.stride(-1) == 1 &&
                  mixed_qkv.scalar_type() == at::kHalf,
              "mixed_qkv must be fp16 [L, 3*H*D], contiguous in last dim");
  TORCH_CHECK(conv_w.dim() == 2 && conv_w.is_contiguous() &&
                  conv_w.scalar_type() == at::kHalf &&
                  conv_w.size(1) == GLM5_PREP_CONV_W,
              "conv_w must be contiguous fp16 [3*H*D, 4]");
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  A_log.scalar_type() == at::kFloat,
              "A_log must be contiguous fp32 [H]");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  dt_bias.scalar_type() == at::kFloat,
              "dt_bias must be contiguous fp32 [H*D]");
  TORCH_CHECK(f.dim() == 2 && f.stride(-1) == 1 &&
                  f.scalar_type() == at::kHalf,
              "f must be fp16 [L, H*D], contiguous in last dim");
  TORCH_CHECK(beta_raw.dim() == 2 && beta_raw.stride(-1) == 1 &&
                  beta_raw.scalar_type() == at::kHalf,
              "beta_raw must be fp16 [L, H], contiguous in last dim");
  TORCH_CHECK(q_out.dim() == 3 && q_out.stride(-1) == 1 &&
                  q_out.scalar_type() == at::kHalf,
              "q_out must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(k_out.dim() == 3 && k_out.stride(-1) == 1 &&
                  k_out.scalar_type() == at::kHalf,
              "k_out must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(v_out.dim() == 3 && v_out.stride(-1) == 1 &&
                  v_out.scalar_type() == at::kHalf,
              "v_out must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(g_out.dim() == 3 && g_out.stride(-1) == 1 &&
                  g_out.scalar_type() == at::kFloat,
              "g_out must be fp32 [L, H, D], contiguous in last dim");
  TORCH_CHECK(beta_out.dim() == 2 && beta_out.stride(-1) == 1 &&
                  beta_out.scalar_type() == at::kFloat,
              "beta_out must be fp32 [L, H], contiguous in last dim");
  TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be int32 [N+1] (varlen-only chain)");
  TORCH_CHECK(chunk_indices.defined() && chunk_indices.dim() == 2 &&
                  chunk_indices.size(1) == 2 &&
                  chunk_indices.scalar_type() == at::kInt,
              "chunk_indices must be int32 [NT, 2] (varlen-only chain)");

  const long L = mixed_qkv.size(0);
  const long qkv_dim = mixed_qkv.size(1);
  TORCH_CHECK(qkv_dim % (3 * GLM5_PREP_D) == 0,
              "mixed_qkv last dim must be divisible by 3*D");
  const long H = qkv_dim / (3 * GLM5_PREP_D);
  TORCH_CHECK(q_out.size(0) == L && q_out.size(1) == H &&
                  q_out.size(2) == GLM5_PREP_D,
              "q_out shape mismatch");
  TORCH_CHECK(k_out.size(0) == L && k_out.size(1) == H &&
                  k_out.size(2) == GLM5_PREP_D,
              "k_out shape mismatch");
  TORCH_CHECK(v_out.size(0) == L && v_out.size(1) == H &&
                  v_out.size(2) == GLM5_PREP_D,
              "v_out shape mismatch");
  TORCH_CHECK(g_out.size(0) == L && g_out.size(1) == H &&
                  g_out.size(2) == GLM5_PREP_D,
              "g_out shape mismatch");
  TORCH_CHECK(beta_out.size(0) == L && beta_out.size(1) == H,
              "beta_out shape mismatch");
  TORCH_CHECK(conv_w.size(0) == qkv_dim, "conv_w rows must equal 3*H*D");
  TORCH_CHECK(A_log.size(0) == H, "A_log size mismatch");
  TORCH_CHECK(dt_bias.size(0) == H * GLM5_PREP_D, "dt_bias size mismatch");
  TORCH_CHECK(f.size(0) == L && f.size(1) == H * GLM5_PREP_D,
              "f shape mismatch");
  TORCH_CHECK(beta_raw.size(0) == L && beta_raw.size(1) == H,
              "beta_raw shape mismatch");

  const long NT = chunk_indices.size(0);
  if (L == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned int)NT, (unsigned int)H);
  glm5_kda_prefill_prep_rdna2_kernel<<<grid, GLM5_PREP_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(mixed_qkv.data_ptr()),
      reinterpret_cast<const __half*>(conv_w.data_ptr()),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      reinterpret_cast<const __half*>(f.data_ptr()),
      reinterpret_cast<const __half*>(beta_raw.data_ptr()),
      reinterpret_cast<__half*>(q_out.data_ptr()),
      reinterpret_cast<__half*>(k_out.data_ptr()),
      reinterpret_cast<__half*>(v_out.data_ptr()),
      g_out.data_ptr<float>(), beta_out.data_ptr<float>(),
      mixed_qkv.stride(0), f.stride(0), beta_raw.stride(0), (int)H,
      (float)lower_bound, cu_seqlens.data_ptr<int>(),
      chunk_indices.data_ptr<int>());
}
