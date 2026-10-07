// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GatedDeltaNet (GDN) chunked-prefill KKT kernel for AMD RDNA2 (gfx1030).
// Hand port of chunk_scaled_dot_kkt_fwd_kernel
// (vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py).
//
// Computes, per (chunk, head), the strictly-lower-triangular matrix
//
//   A[i, j] = beta[i] * exp(g[i] - g[j]) * (k[i] . k[j])   for i > j
//   A[i, j] = 0                                             otherwise
//
// where the dot is over the K=128 key dimension. This is the "KKT system"
// matrix (beta * K * K^T) that solve_tril later inverts.
//
// CRITICAL PARITY RULE (gfx1030): the call site sets
// _CAST_DOT_TO_K_DTYPE = on_gfx1x() = False on gfx1030, so beta-scaled k
// stays in fp32 and the [64,128] @ [128,64] dot is a pure fp32 x fp32 FMA
// accumulation. Do NOT use V_DOT2 here -- it would change the result.
// k is loaded from memory as fp16, cast to fp32, scaled by the fp32 beta,
// then accumulated with plain fp32 FMA.
//
// Workgroup = one (chunk, head). 256 threads. The k tile [64,128] (fp16,
// 16 KB) is staged in LDS; each thread owns a 4x4 sub-tile of the [64,64]
// fp32 output A (16 x 16 sub-tiles = 256 threads).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int KKT_BT = 64;       // chunk size (FLA_CHUNK_SIZE)
constexpr int KKT_K = 128;       // head_k_dim
constexpr int KKT_THREADS = 256; // 16 row-tiles x 16 col-tiles of 4x4

template <bool IS_VARLEN>
__global__ void __launch_bounds__(KKT_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_prefill_kkt_rdna2_kernel(
        const __half* __restrict__ k,          // [B, T, Hg, K]
        const float* __restrict__ beta,        // [B, T, H]
        const float* __restrict__ g,           // [B, T, H] (cumulative log gate)
        float* __restrict__ A,                 // [B, T, H, BT] output
        const int* __restrict__ cu_seqlens,    // [N+1] (varlen only)
        const int* __restrict__ chunk_indices, // [N_chunks, 2] (varlen only)
        long T, long H, long Hg, long K_dim) {
  const int i_tg = blockIdx.x;   // global chunk index (or chunk_indices row)
  const int i_bh = blockIdx.y;   // b * H + h
  const int i_b = i_bh / (int)H;
  const int i_h = i_bh % (int)H;

  long bos, eos;
  int i_t;  // chunk index within the sequence
  if (IS_VARLEN) {
    const int i_n = chunk_indices[i_tg * 2];
    i_t = chunk_indices[i_tg * 2 + 1];
    bos = cu_seqlens[i_n];
    eos = cu_seqlens[i_n + 1];
  } else {
    i_t = i_tg;
    bos = i_b * T;
    eos = bos + T;
  }
  const long T_local = eos - bos;
  const long t_row_base = bos + (long)i_t * KKT_BT;
  // Valid rows in this chunk (last chunk may be a tail < 64).
  const int t_chunk =
      (int)((T_local - (long)i_t * KKT_BT) < KKT_BT
                ? (T_local - (long)i_t * KKT_BT)
                : KKT_BT);

  const int hg = i_h / (int)(H / Hg);  // key head index (H >= Hg, H % Hg == 0)

  // Stage the k tile [BT, K] (fp16) in LDS: 16 KB.
  __shared__ __half s_k[KKT_BT * KKT_K];
  const int tid = threadIdx.x;
#pragma unroll
  for (int it = 0; it < KKT_BT * KKT_K / KKT_THREADS; ++it) {
    const int idx = tid + it * KKT_THREADS;
    const int row = idx / KKT_K;
    const int col = idx - row * KKT_K;
    __half val = __float2half(0.0f);
    if (row < t_chunk) {
      val = k[(t_row_base + row) * Hg * K_dim + (long)hg * K_dim + col];
    }
    s_k[row * KKT_K + col] = val;
  }
  __syncthreads();

  // Thread -> 4x4 sub-tile of the [64,64] output.
  const int row_tile = tid / 16;
  const int col_tile = tid % 16;
  const int i0 = row_tile * 4;
  const int j0 = col_tile * 4;

  // Per-row beta and per-row/col gate values (fp32, zero-padded past T_local).
  float beta_r[4], g_r[4], g_c[4];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const long t_i = t_row_base + i0 + l;
    const long t_j = t_row_base + j0 + l;
    beta_r[l] = (t_i < eos) ? beta[t_i * H + i_h] : 0.0f;
    g_r[l] = (t_i < eos) ? g[t_i * H + i_h] : 0.0f;
    g_c[l] = (t_j < eos) ? g[t_j * H + i_h] : 0.0f;
  }

#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int i = i0 + l;
    const float beta_i = beta_r[l];
    const float g_i = g_r[l];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int kk = 0; kk < KKT_K; ++kk) {
      // beta applied inside the dot exactly as the reference (b_kb = b_k *
      // beta[:, None]); the dot is fp32 x fp32 FMA.
      const float kb = beta_i * __half2float(s_k[i * KKT_K + kk]);
      acc[0] += kb * __half2float(s_k[(j0 + 0) * KKT_K + kk]);
      acc[1] += kb * __half2float(s_k[(j0 + 1) * KKT_K + kk]);
      acc[2] += kb * __half2float(s_k[(j0 + 2) * KKT_K + kk]);
      acc[3] += kb * __half2float(s_k[(j0 + 3) * KKT_K + kk]);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int j = j0 + m;
      // Rows past the sequence end (tail chunk) are out of bounds in A;
      // skip the write rather than storing 0.0f to an OOB address.
      if (i < t_chunk) {
        const float val = acc[m] * expf(g_i - g_c[m]);
        // Strict '>' causal mask (o_t[i] > o_t[j]) plus the m_t bounds.
        const bool valid = (i > j) && (j < t_chunk);
        A[(t_row_base + i) * H * KKT_BT + (long)i_h * KKT_BT + j] =
            valid ? val : 0.0f;
      }
    }
  }
}

}  // namespace

void gdn_prefill_kkt_rdna2(torch::Tensor k, torch::Tensor beta,
                           torch::Tensor g, torch::Tensor A,
                           torch::Tensor cu_seqlens,
                           torch::Tensor chunk_indices) {
  TORCH_CHECK(k.dim() == 4 && k.stride(-1) == 1 &&
                  k.scalar_type() == at::kHalf,
              "k must be fp16 [B, T, Hg, K], contiguous in last dim");
  TORCH_CHECK(beta.dim() == 3 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kFloat,
              "beta must be fp32 [B, T, H], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1 &&
                  g.scalar_type() == at::kFloat,
              "g must be fp32 [B, T, H], contiguous in last dim");
  TORCH_CHECK(A.dim() == 4 && A.stride(-1) == 1 &&
                  A.scalar_type() == at::kFloat,
              "A must be fp32 [B, T, H, BT], contiguous in last dim");

  const long B = k.size(0);
  const long T = k.size(1);
  const long Hg = k.size(2);
  const long K = k.size(3);
  const long H = beta.size(2);
  const long BT = A.size(3);
  TORCH_CHECK(K == KKT_K, "gdn_prefill_kkt_rdna2 requires K == 128");
  TORCH_CHECK(BT == KKT_BT, "gdn_prefill_kkt_rdna2 requires BT == 64");
  TORCH_CHECK(H % Hg == 0, "H must be divisible by Hg");
  TORCH_CHECK(beta.size(0) == B && beta.size(1) == T,
              "beta shape mismatch");
  TORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(2) == H,
              "g shape mismatch");
  TORCH_CHECK(A.size(0) == B && A.size(1) == T && A.size(2) == H,
              "A shape mismatch");

  const bool is_varlen = cu_seqlens.defined();
  TORCH_CHECK(is_varlen == chunk_indices.defined(),
              "cu_seqlens and chunk_indices must both be provided (varlen) "
              "or both absent");
  long NT;
  if (is_varlen) {
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == at::kInt,
                "cu_seqlens must be int32 [N+1]");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                    chunk_indices.scalar_type() == at::kInt,
                "chunk_indices must be int32 [N_chunks, 2]");
    NT = chunk_indices.size(0);
  } else {
    NT = (T + KKT_BT - 1) / KKT_BT;
  }
  if (B == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(NT, B * H);
  const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
  const float* beta_ptr = beta.data_ptr<float>();
  const float* g_ptr = g.data_ptr<float>();
  float* A_ptr = A.data_ptr<float>();
  if (is_varlen) {
    gdn_prefill_kkt_rdna2_kernel<true><<<grid, KKT_THREADS, 0, stream>>>(
        k_ptr, beta_ptr, g_ptr, A_ptr, cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), T, H, Hg, K);
  } else {
    gdn_prefill_kkt_rdna2_kernel<false><<<grid, KKT_THREADS, 0, stream>>>(
        k_ptr, beta_ptr, g_ptr, A_ptr, nullptr, nullptr, T, H, Hg, K);
  }
}
