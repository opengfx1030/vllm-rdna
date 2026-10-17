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
// _CAST_DOT_TO_K_DTYPE = on_gfx1x() or on_gfx10x() -- True on gfx1030 -- so
// the reference computes b_kb = b_k * beta in fp32, rounds it back to k's
// fp16 storage dtype, and only then does the dot:
//
//   b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
//
// The beta-scaled operand therefore carries an fp16 rounding and the dot is
// fp16 x fp16 with an fp32 accumulator. An earlier version of this kernel
// kept b_kb in fp32 (the comment above used to claim CAST_DOT_TO_K_DTYPE was
// False on gfx1030, which stopped being true when the `or on_gfx10x()` was
// added upstream), and every parity case failed by up to 0.07 absolute.
//
// Workgroup = one (chunk, head). 256 threads. The k tile [64,128] (fp16,
// 16 KB) is staged in LDS; each thread owns a 4x4 sub-tile of the [64,64]
// fp32 output A (16 x 16 sub-tiles = 256 threads).
//
// The dot is the v_dot2c_f32_f16 instruction, which is what the reference's
// tl.dot(b_kb.to(fp16), tl.trans(b_k)) lowers to on gfx1030, so the pairwise
// summation order over k (and hence the fp32 accumulation error) matches the
// reference rather than merely being within tolerance of it.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int KKT_BT = 64;        // chunk size (FLA_CHUNK_SIZE)
constexpr int KKT_K = 128;        // head_k_dim
constexpr int KKT_THREADS = 256;  // 16 row-tiles x 16 col-tiles of 4x4
// LDS row stride, padded. The column reads of the inner loop walk rows at a
// stride of KKT_K halves = 256 B = 64 dwords, a multiple of the 32 LDS banks,
// so all 16 column lanes hit one bank (16-way conflict). The lanes are 4 rows
// apart, so their dword stride is 2*KKT_KP; KKT_KP = K + 2 makes that 260,
// i.e. 4 (mod 32) -- the widest spread a 4-row lane stride can reach, giving
// 8 distinct banks (2-way conflict) instead of 1.
constexpr int KKT_KP = KKT_K + 2;

// Two adjacent halves of one row. KKT_KP is even, so a 4-byte LDS read at an
// even column is aligned.
__device__ __forceinline__ unsigned kkt_ld2(const __half* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

// b_kb = round_fp16(beta * k) for two adjacent k, packed in one register:
// v_fma_mixlo_f16 / v_fma_mixhi_f16 write the two halves without a repack.
__device__ __forceinline__ unsigned kkt_scale2(float beta, unsigned raw) {
  const __half2 h = *reinterpret_cast<const __half2*>(&raw);
  const __half2 out = __halves2half2(
      __float2half_rn(beta * __half2float(__low2half(h))),
      __float2half_rn(beta * __half2float(__high2half(h))));
  return *reinterpret_cast<const unsigned*>(&out);
}

// acc + a.lo*b.lo + a.hi*b.hi in fp32. This is the instruction the reference
// lowers tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k)) to on gfx1030, so the
// pairwise summation order over k matches the reference.
__device__ __forceinline__ float kkt_dot2(unsigned a, unsigned b, float acc) {
  asm("v_dot2c_f32_f16 %0, %1, %2" : "+v"(acc) : "v"(a), "v"(b));
  return acc;
}
template <bool IS_VARLEN>
__global__ void __launch_bounds__(KKT_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_prefill_kkt_rdna2_kernel(
        const __half* __restrict__ k,        // [B, T, Hg, K]
        const float* __restrict__ beta,      // [B, T, H]
        const float* __restrict__ g,         // [B, T, H] (cumulative log gate)
        float* __restrict__ A,               // [B, T, H, BT] output
        const int* __restrict__ cu_seqlens,  // [N+1] (varlen only)
        const int* __restrict__ chunk_indices,  // [N_chunks, 2] (varlen only)
        long T, long H, long Hg, long K_dim) {
  const int i_tg = blockIdx.x;  // global chunk index (or chunk_indices row)
  const int i_bh = blockIdx.y;  // b * H + h
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
  const int t_chunk = (int)((T_local - (long)i_t * KKT_BT) < KKT_BT
                                ? (T_local - (long)i_t * KKT_BT)
                                : KKT_BT);

  const int hg = i_h / (int)(H / Hg);  // key head index (H >= Hg, H % Hg == 0)

  // Stage the k tile [BT, K] (fp16) in LDS: 16 KB.
  __shared__ __half s_k[KKT_BT * KKT_KP];
  const int tid = threadIdx.x;
  // One 16-byte global load per step. The elementwise form spent ~1170 of the
  // kernel's ~2350 instructions on 64-bit index math for 2 bytes at a time;
  // the chunk index makes row/col shifts and the row base loop-invariant. The
  // LDS stores stay 16-bit: 16-byte stores need KKT_KP % 8 == 0, which forces
  // 2*KKT_KP % 32 == 0 and so brings the 16-way column bank conflict back.
  const __half* __restrict__ k_row =
      k + t_row_base * (Hg * K_dim) + (long)hg * K_dim;
#pragma unroll
  for (int it = 0; it < KKT_BT * KKT_K / 8 / KKT_THREADS; ++it) {
    const int chunk = tid + it * KKT_THREADS;
    const int row = chunk >> 4;         // 16 x 8-half chunks per 128-wide row
    const int col = (chunk & 15) << 3;
    __half* dst = s_k + row * KKT_KP + col;
    if (row < t_chunk) {
      const int4 v = *reinterpret_cast<const int4*>(
          k_row + (long)row * (Hg * K_dim) + col);
      const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
      for (int q = 0; q < 8; ++q) dst[q] = h[q];
    } else {
#pragma unroll
      for (int q = 0; q < 8; ++q) dst[q] = __float2half(0.0f);
    }
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

  // [row l][col m] accumulators: 16 fp32 registers per thread.
  float acc[4][4] = {};
  // 8 is the best of 8/16/32/64 measured (dot2 needs the four column operands
  // live, so a deeper unroll only costs VGPRs).
#pragma unroll 8
  for (int kk = 0; kk < KKT_K; kk += 2) {
    // The four column operands are shared by all four rows, so they are
    // loaded once per k pair; beta is applied inside the dot exactly as the
    // reference does it (b_kb = b_k * beta[:, None], rounded to fp16).
    unsigned k_col[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      k_col[m] = kkt_ld2(&s_k[(j0 + m) * KKT_KP + kk]);
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const unsigned k_row =
          kkt_scale2(beta_r[l], kkt_ld2(&s_k[(i0 + l) * KKT_KP + kk]));
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        acc[l][m] = kkt_dot2(k_row, k_col[m], acc[l][m]);
      }
    }
  }
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int i = i0 + l;
    // Rows past the sequence end (tail chunk) are out of bounds in A; skip
    // the write rather than storing 0.0f to an OOB address.
    if (i < t_chunk) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int j = j0 + m;
        const float val = acc[l][m] * expf(g_r[l] - g_c[m]);
        // Strict '>' causal mask (o_t[i] > o_t[j]) plus the m_t bounds.
        const bool valid = (i > j) && (j < t_chunk);
        A[(t_row_base + i) * H * KKT_BT + (long)i_h * KKT_BT + j] =
            valid ? val : 0.0f;
      }
    }
  }
}

}  // namespace

void gdn_prefill_kkt_rdna2(torch::Tensor k, torch::Tensor beta, torch::Tensor g,
                           torch::Tensor A, torch::Tensor cu_seqlens,
                           torch::Tensor chunk_indices) {
  TORCH_CHECK(k.dim() == 4 && k.stride(-1) == 1 && k.scalar_type() == at::kHalf,
              "k must be fp16 [B, T, Hg, K], contiguous in last dim");
  // The k tile is staged with 16-byte loads, which assume the packed [B, T,
  // Hg, K] layout the kernel indexes with (it never consults k.stride()).
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(beta.dim() == 3 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kFloat,
              "beta must be fp32 [B, T, H], contiguous in last dim");
  TORCH_CHECK(
      g.dim() == 3 && g.stride(-1) == 1 && g.scalar_type() == at::kFloat,
      "g must be fp32 [B, T, H], contiguous in last dim");
  TORCH_CHECK(
      A.dim() == 4 && A.stride(-1) == 1 && A.scalar_type() == at::kFloat,
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
  TORCH_CHECK(beta.size(0) == B && beta.size(1) == T, "beta shape mismatch");
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
