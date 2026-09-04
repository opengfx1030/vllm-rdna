// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA chunked-prefill KKT kernel for AMD RDNA2 (gfx1030).
// Implements the torch reference at modeling_glm5_next.py:531-533
// (attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask)
//          .sum(-1).masked_fill(triu(diag=0), 0)).
//
// THE SIGN BRIDGE (critical, read this first):
// The HF reference builds the STRICT-lower matrix
//   A_signed[i,j] = -beta_i * sum_d k_i[d]*k_j[d]*exp(g_i[d]-g_j[d])
// for i > j (negative), then forward-substitutes (I - A_signed)^-1.
// This kernel instead emits the POSITIVE matrix
//   P[i,j] = beta_i * sum_d k_i[d]*k_j[d]*exp(g_i[d]-g_j[d])  (i > j)
// so that the GDN-style solver in glm5_kda_prefill_solve_wy_rdna2.cu,
// which computes (I + A_input)^-1 by construction, yields
//   (I + P)^-1 == (I - A_signed)^-1 == T
// exactly the matrix the reference uses for value = T @ (v*beta) and
// k_cumdecay = T @ (k*beta*exp(g)). The negation is thus absorbed by
// the solver's sign convention; downstream math is unchanged.
//
// g here is the per-(head, dim) INCLUSIVE cumsum of the log-decay within
// the chunk (produced by glm5_kda_prefill_prep_rdna2). Because gate <= 0,
// the cumsum is non-increasing in the token index, so for i > j we have
// g_i[d] <= g_j[d] and exp(g_i[d] - g_j[d]) <= 1: the DIRECT form
// expf(g_i - g_j) is used inside the dot (never the factorized
// exp(g_i)*exp(-g_j) form, whose exp(-g_j) factor can overflow fp32 for
// deeply negative cumsums even though the product is <= 1).
//
// No V_DOT2: the per-dim exp factor varies inside the reduction axis, so
// the dot is scalar fp32 FMA (also matching the gfx1030 GDN kkt parity
// rule "_CAST_DOT_TO_K_DTYPE = False": beta*k stays fp32, fp32 accum).
//
// Workgroup = one (chunk, head); grid = (NT, H); 256 threads, each owns
// a 4x4 sub-tile of the [64, 64] output. LDS: s_k [64,128] fp16 (16 KB)
// + s_g [64,128] fp32 (32 KB) = 48 KB. Output layout (documented):
//   A_out [NT, H, 64, 64] fp32, A_out[((i_tg*H + i_h)*64 + i)*64 + j].
//
// PERF NOTE (first cut): 16 expf per inner dim per thread = 2048
// accurate expf per thread. Prefill correctness first; a chunk-16
// variant (FlashKDA-inspired) or precomputed eg tiles are future work.
// Varlen-only chain: cu_seqlens [N+1], chunk_indices [NT, 2] required.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_KKT_BT = 64;       // chunk size
constexpr int GLM5_KKT_K = 128;       // head dim
constexpr int GLM5_KKT_THREADS = 256; // 16 row-tiles x 16 col-tiles of 4x4

__global__ void glm5_kda_prefill_kkt_rdna2_kernel(
    const __half* __restrict__ k,           // [L, H, D] fp16
    const float* __restrict__ beta,         // [L, H] fp32 (sigmoid'd)
    const float* __restrict__ g,            // [L, H, D] fp32 (cumsum'd)
    float* __restrict__ A,                  // [NT, H, BT, BT] fp32 out
    int H,
    const int* __restrict__ cu_seqlens,     // [N+1]
    const int* __restrict__ chunk_indices) { // [NT, 2]
  const int i_tg = blockIdx.x;  // flat chunk index
  const int i_h = blockIdx.y;   // head
  const int tid = threadIdx.x;

  // LDS tiles for this (chunk, head).
  __shared__ __half s_k[GLM5_KKT_BT * GLM5_KKT_K];  // 16 KB
  __shared__ float s_g[GLM5_KKT_BT * GLM5_KKT_K];   // 32 KB

  // Resolve chunk bounds.
  const int i_n = chunk_indices[i_tg * 2];
  const int i_t = chunk_indices[i_tg * 2 + 1];
  const long bos = (long)cu_seqlens[i_n];
  const long eos = (long)cu_seqlens[i_n + 1];
  const long t_base = bos + (long)i_t * GLM5_KKT_BT;
  const long remaining = eos - t_base;
  const int t_chunk =
      (remaining < GLM5_KKT_BT) ? (int)remaining : GLM5_KKT_BT;

  // Stage k and g tiles (fp16 / fp32); tail rows zero-filled.
#pragma unroll
  for (int it = 0; it < GLM5_KKT_BT * GLM5_KKT_K / GLM5_KKT_THREADS; ++it) {
    const int idx = tid + it * GLM5_KKT_THREADS;
    const int row = idx / GLM5_KKT_K;
    const int col = idx - row * GLM5_KKT_K;
    __half kv = __float2half(0.0f);
    float gv = 0.0f;
    if (row < t_chunk) {
      const long tok = t_base + row;
      kv = k[(tok * H + i_h) * GLM5_KKT_K + col];
      gv = g[(tok * H + i_h) * GLM5_KKT_K + col];
    }
    s_k[idx] = kv;
    s_g[idx] = gv;
  }
  __syncthreads();

  // Thread -> 4x4 sub-tile of the [64, 64] output.
  const int row_tile = tid / 16;
  const int col_tile = tid % 16;
  const int i0 = row_tile * 4;
  const int j0 = col_tile * 4;

  // Per-row beta (0 for tail rows -> P row vanishes).
  float beta_r[4];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int i = i0 + l;
    beta_r[l] = (i < t_chunk) ? beta[(t_base + i) * H + i_h] : 0.0f;
  }

  // P[i,j] = beta_i * sum_d k_i[d]*k_j[d]*exp(g_i[d]-g_j[d]).
  // fp32 FMA; direct-form exp (see header for overflow rationale).
  float acc[4][4];
#pragma unroll
  for (int l = 0; l < 4; ++l)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[l][m] = 0.0f;

#pragma unroll 1
  for (int d = 0; d < GLM5_KKT_K; ++d) {
    float gi[4], kb[4], gj[4], kj[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int i = i0 + l;
      gi[l] = s_g[i * GLM5_KKT_K + d];
      kb[l] = beta_r[l] * __half2float(s_k[i * GLM5_KKT_K + d]);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int j = j0 + m;
      gj[m] = s_g[j * GLM5_KKT_K + d];
      kj[m] = __half2float(s_k[j * GLM5_KKT_K + d]);
    }
#pragma unroll
    for (int l = 0; l < 4; ++l)
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        acc[l][m] += kb[l] * kj[m] * expf(gi[l] - gj[m]);
      }
  }

  // Store with the strict-lower causal mask (i > j) and chunk bounds.
  const long a_head = ((long)i_tg * H + i_h) * GLM5_KKT_BT * GLM5_KKT_BT;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int i = i0 + l;
    if (i >= t_chunk) continue;  // tail rows: leave caller-zeroed
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int j = j0 + m;
      const bool valid = (i > j) && (j < t_chunk);
      A[a_head + (long)i * GLM5_KKT_BT + j] = valid ? acc[l][m] : 0.0f;
    }
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- torch.ops._rocm_C.glm5_kda_prefill_kkt_rdna2.
// Varlen-only: cu_seqlens and chunk_indices must be defined.
// ============================================================================

void glm5_kda_prefill_kkt_rdna2(torch::Tensor k, torch::Tensor beta,
                                torch::Tensor g, torch::Tensor A,
                                torch::Tensor cu_seqlens,
                                torch::Tensor chunk_indices) {
  TORCH_CHECK(k.dim() == 3 && k.stride(-1) == 1 &&
                  k.scalar_type() == at::kHalf,
              "k must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(beta.dim() == 2 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kFloat,
              "beta must be fp32 [L, H], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1 &&
                  g.scalar_type() == at::kFloat,
              "g must be fp32 [L, H, D], contiguous in last dim");
  TORCH_CHECK(A.dim() == 4 && A.is_contiguous() &&
                  A.scalar_type() == at::kFloat &&
                  A.size(2) == GLM5_KKT_BT && A.size(3) == GLM5_KKT_BT,
              "A must be contiguous fp32 [NT, H, 64, 64]");
  TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be int32 [N+1] (varlen-only chain)");
  TORCH_CHECK(chunk_indices.defined() && chunk_indices.dim() == 2 &&
                  chunk_indices.size(1) == 2 &&
                  chunk_indices.scalar_type() == at::kInt,
              "chunk_indices must be int32 [NT, 2] (varlen-only chain)");

  const long L = k.size(0);
  const long H = k.size(1);
  const long K = k.size(2);
  TORCH_CHECK(K == GLM5_KKT_K, "glm5_kda_prefill_kkt_rdna2 requires D == 128");
  TORCH_CHECK(beta.size(0) == L && beta.size(1) == H, "beta shape mismatch");
  TORCH_CHECK(g.size(0) == L && g.size(1) == H && g.size(2) == K,
              "g shape mismatch");
  TORCH_CHECK(A.size(0) == chunk_indices.size(0) && A.size(1) == H,
              "A shape mismatch with chunk_indices / k");

  const long NT = chunk_indices.size(0);
  if (L == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned int)NT, (unsigned int)H);
  glm5_kda_prefill_kkt_rdna2_kernel<<<grid, GLM5_KKT_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(k.data_ptr()),
      beta.data_ptr<float>(), g.data_ptr<float>(), A.data_ptr<float>(),
      (int)H, cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>());
}
