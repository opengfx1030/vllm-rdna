// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GatedDeltaNet (GDN) chunked-prefill fused kernel for AMD RDNA2 (gfx1030).
// Hand port of two Triton references from
// vllm/third_party/flash_linear_attention/ops/, fused into one launch:
//   1. merge_16x16_to_64x64_inverse_kernel  (solve_tril.py, BT = 64)
//   2. recompute_w_u_fwd_kernel             (wy_fast.py)
//
// Fusion rationale: both kernels share the exact same (NT, B*H) grid and
// the exact same (chunk, head) partition. Keeping A_inv = (I + A_lower)^-1
// resident in LDS between the two phases eliminates a ~64 MB write + read
// round-trip of the fp16 intermediate (BT = 64, fp16 A_inv per chunk).
//
// Workgroup = one (chunk, head). 256 threads. LDS layout (rounded down):
//   s_A    [64][64] fp32   = 16 KB   -- raw A tile (Phase 1 input + Schur)
//   s_Ai   [64][64] fp32   = 16 KB   -- evolving inverse (lower-tri + 0)
//   s_tmp  [16][16] fp32   =  1 KB   -- Schur intermediate
//   s_Aih  [64][64] fp16   =  8 KB   -- cast of s_Ai for Phase 2 (rtne)
//   s_rhsT [128][66] fp16  = 16.5 KB -- transposed + bank-padded rhs
//   s_beta [64]     fp32   = 0.25 KB
//   s_eg   [64]     fp32   = 0.25 KB
//                              total ~58 KB  (fits 64 KB / CU budget)
//
// PARITY (critical, matched against the references exactly):
//
// Phase 1 -- solve_tril (merge_16x16_to_64x64_inverse_kernel)
//   * A is fp32 input (output of chunk_scaled_dot_kkt_fwd, output_dtype=fp32).
//   * Diagonal block init: A_inv[r][c] = (r > c) ? -A[r][c] : 0  within each
//     16x16 diagonal block; upper blocks stay 0 until Schur (zero-initialized
//     s_Ai gives these for free).
//   * Forward substitution: for each i in [block_base+2, block_base+16):
//       a[k]      = -A[i][k]
//       new[j]    = a[j] + sum_k a[k] * A_inv[k][j]
//       A_inv[i][j] = new[j]
//     then add identity to the block's diagonal. fp32 FMA only -- the
//     operands are fp32 (this matches Triton's `input_precision="ieee"`
//     for tl.dot on AMD, which lowers to a software FMA sequence).
//   * Schur (6 off-diagonal blocks):
//       Ai_21 = -Ai_22 @ A_21 @ Ai_11
//       Ai_32 = -Ai_33 @ A_32 @ Ai_22
//       Ai_43 = -Ai_44 @ A_43 @ Ai_33
//       Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21)
//       Ai_42 = -Ai_44 @ (A_42 @ Ai_22 + A_43 @ Ai_32)
//       Ai_41 = -Ai_44 @ (A_41 @ Ai_11 + A_42 @ Ai_21 + A_43 @ Ai_31)
//     All fp32 FMA. Same dot association order as the Triton reference.
//   * A_inv global store: rtne cast to fp16, write full 64-col rows for
//     valid chunk rows. OOB chunk rows (last chunk of a sequence) are
//     skipped via the row bound, exactly like the reference's
//     boundary_check=(0, 1) on the make_block_ptr store. The upper-tri
//     blocks of valid rows are written 0 (matches reference zeros_like).
//
// Phase 2 -- recompute_w_u
//   * A_inv is loaded as fp16 (V_DOT2_F32_F16 operand).
//   * u:  rhs[r][c] = __float2half_rn( v[t0+r][i_h][c] * beta[r] )
//   * w:  rhs[r][c] = __float2half_rn( (k[t0+r][i_hg][c] * beta[r]) * eg[r] )
//     where eg = exp(g_cumsum). The explicit fp16 rtne cast of these
//     intermediate products is part of the reference numerics:
//        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)     <-- rtne fp16
//        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
//     Both match. Dot is V_DOT2_F32_F16 (__builtin_amdgcn_fdot2).
//   * The right-hand side is staged in LDS transposed (s_rhsT[c][r]) so
//     V_DOT2 can pair operands as __half2 vectors on the k-axis. Row-pad
//     stride (64 + 2 = 66 halves) to break LDS bank conflicts on the
//     inner-loop B reads.
//   * expf (accurate __ocml_exp) is used; NEVER __expf.
//
// Registration: this op is gated by VLLM_ROCM_GFX1030 in torch_bindings.cpp.
// No in-file ifdefs -- the conditional-compilation pattern is external.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GDN_BT = 64;          // FLA_CHUNK_SIZE
constexpr int GDN_BLOCK = 16;       // 16x16 sub-block within BT
constexpr int GDN_K = 128;          // head_k_dim
constexpr int GDN_V = 128;          // head_v_dim
constexpr int GDN_THREADS = 256;    // 1 workgroup per (chunk, head)

// Pad on the r-axis of the transposed rhs tile (64 -> 66 halves) to avoid
// LDS bank conflicts when warp lanes stride through the c dimension.
constexpr int GDN_RHS_R_PAD = 2;
constexpr int GDN_RHS_C = 128;      // = max(K, V)

// ----- Shared-memory block-origin helpers ------------------------------------

__device__ __forceinline__ float* gdn_blk(float* base, int br, int bc) {
  return base + br * GDN_BLOCK * GDN_BT + bc * GDN_BLOCK;
}
__device__ __forceinline__ const float* gdn_blk(const float* base, int br,
                                                int bc) {
  return base + br * GDN_BLOCK * GDN_BT + bc * GDN_BLOCK;
}

// One element of a 16x16 fp32 matmul, fp32 FMA. Accumulator or assigner;
// negate applies sign at the final write so the value mirrors the
// reference's `-tl.dot(...)` convention.
__device__ __forceinline__ void gdn_mm16(float* Z, int ldz,
                                         const float* X, int ldx,
                                         const float* Y, int ldy,
                                         bool accumulate, bool negate,
                                         int tid) {
  const int r = tid >> 4;
  const int c = tid & 15;
  float acc = 0.0f;
#pragma unroll
  for (int m = 0; m < GDN_BLOCK; ++m) {
    acc += X[r * ldx + m] * Y[m * ldy + c];
  }
  if (negate) acc = -acc;
  if (accumulate)
    Z[r * ldz + c] += acc;
  else
    Z[r * ldz + c] = acc;
}

// ============================================================================
// Main kernel
// ============================================================================

template <bool IS_VARLEN>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_prefill_solve_wy_rdna2_kernel(
        const float* __restrict__ A,             // [B, T, H, BT] fp32 (from kkt)
        const __half* __restrict__ k,            // [B, T, Hg, K] fp16
        const __half* __restrict__ v,            // [B, T, H, V]  fp16
        const float* __restrict__ beta,          // [B, T, H] fp32
        const float* __restrict__ g,             // [B, T, H] fp32 (cumsum)
        __half* __restrict__ A_inv,              // [B, T, H, BT] fp16 out
        __half* __restrict__ w,                  // [B, T, H, K]  fp16 out
        __half* __restrict__ u,                  // [B, T, H, V]  fp16 out
        const int* __restrict__ cu_seqlens,      // [N+1] or null
        const int* __restrict__ chunk_indices,   // [NT, 2] or null
        long T, long H, long Hg) {
  const int i_tg = blockIdx.x;
  const int i_bh = blockIdx.y;
  const int i_b = i_bh / (int)H;
  const int i_h = i_bh % (int)H;

  // Resolve (bos, i_t_within_seq, T_local) -- varlen remaps the global chunk
  // index to a per-sequence local one and pulls the sequence length.
  long bos;
  int i_t;
  long T_local;
  if (IS_VARLEN) {
    const int i_n = chunk_indices[i_tg * 2];
    i_t = chunk_indices[i_tg * 2 + 1];
    bos = (long)cu_seqlens[i_n];
    T_local = (long)cu_seqlens[i_n + 1] - bos;
  } else {
    i_t = i_tg;
    bos = (long)i_b * T;
    T_local = T;
  }
  const int rows = (int)min((long)GDN_BT, T_local - (long)i_t * GDN_BT);
  const long t0 = bos + (long)i_t * GDN_BT;   // first token of this chunk
  const int hg = i_h / (int)(H / Hg);         // k-head index

  const int tid = threadIdx.x;

  // --------------------------------------------------------------------------
  // LDS layout
  // --------------------------------------------------------------------------
  __shared__ float s_A[GDN_BT * GDN_BT];      // raw A (fp32)
  __shared__ float s_Ai[GDN_BT * GDN_BT];     // inverse (fp32, lower-tri only)
  __shared__ float s_tmp[GDN_BLOCK * GDN_BLOCK]; // Schur intermediate
  __shared__ __half s_Aih[GDN_BT * GDN_BT];   // fp16 cast of s_Ai (rtne)
  __shared__ __half s_rhsT[GDN_RHS_C * (GDN_BT + GDN_RHS_R_PAD)];
  __shared__ float s_beta[GDN_BT];
  __shared__ float s_eg[GDN_BT];

  // --------------------------------------------------------------------------
  // Load s_A: cooperative row-major copy of the chunk's A tile.
  // OOB chunk rows (rows_valid < 64) are zero-filled, matching the
  // reference's boundary_check on the make_block_ptr load.
  // --------------------------------------------------------------------------
#pragma unroll
  for (int it = 0; it < (GDN_BT * GDN_BT) / GDN_THREADS; ++it) {
    const int idx = tid + it * GDN_THREADS;   // 0..4095
    const int r = idx >> 6;
    const int c = idx & 63;
    s_A[idx] = (r < rows)
                   ? A[((t0 + r) * H + i_h) * GDN_BT + c]
                   : 0.0f;
  }

  // Initialize s_Ai: zero everywhere (default for off-diag blocks); for the
  // 4 diagonal 16x16 blocks set A_inv[r][c] = -A[r][c] where r > c else 0.
#pragma unroll
  for (int it = 0; it < (GDN_BT * GDN_BT) / GDN_THREADS; ++it) {
    const int idx = tid + it * GDN_THREADS;
    const int r = idx >> 6;
    const int c = idx & 63;
    const int br = r >> 4;
    const int bc = c >> 4;
    const int lr = r & 15;
    const int lc = c & 15;
    s_Ai[idx] = (br == bc && lr > lc) ? -s_A[idx] : 0.0f;
  }

  // Phase 2 broadcast caches (beta + exp(g)) -- computed by lanes 0..63
  // once now, reused per element in the rhs-staging loop.
  if (tid < GDN_BT) {
    s_beta[tid] = (tid < rows) ? beta[(t0 + tid) * H + i_h] : 0.0f;
    s_eg[tid] = (tid < rows) ? expf(g[(t0 + tid) * H + i_h]) : 0.0f;
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Phase 1a -- 4 diagonal forward-substitution blocks.
  // Each block runs sequentially using all 256 threads. At iteration i:
  //   thread t = (j, k) where j = t/16, k = t&15 computes
  //     a_k = -A[16b+i][16b+k]
  //     partial = a_k * A_inv[16b+k][16b+j]
  //   then a butterfly shfl reduces the 16-lane partials over k. Lane k==0
  //   adds a_j and writes A_inv[16b+i][16b+j]. __syncthreads between
  //   iterations so next iteration sees the freshly-written row.
  // --------------------------------------------------------------------------
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int row_end = min(GDN_BLOCK, rows - b * GDN_BLOCK);  // <= 16
    const int base = b * GDN_BLOCK;

    for (int i = 2; i < row_end; ++i) {
      const int grow = base + i;            // global row inside the tile
      const int j = tid >> 4;               // output column (local 0..15)
      const int k = tid & 15;               // reduction lane
      const float a_k = -s_A[grow * GDN_BT + base + k];
      float partial = a_k * s_Ai[(base + k) * GDN_BT + base + j];
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 1);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 2);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 4);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 8);
      if (k == 0) {
        const float a_j = -s_A[grow * GDN_BT + base + j];
        s_Ai[grow * GDN_BT + base + j] = a_j + partial;
      }
      __syncthreads();
    }

    // Add identity to the block's diagonal (after the loop, exactly like
    // the reference). Threads with r == c within the local 16x16 block
    // own the diagonal element.
    {
      const int lr = tid >> 4;
      const int lc = tid & 15;
      if (lr == lc) s_Ai[(base + lr) * GDN_BT + (base + lc)] += 1.0f;
    }
    __syncthreads();
  }

  // --------------------------------------------------------------------------
  // Phase 1b -- 6 Schur-complement blocks (each a 16x16 fp32 chain).
  // Dot association order matches the Triton reference exactly so the
  // summation-order delta stays within the rtol budget on fp16 outputs.
  // --------------------------------------------------------------------------

  // Ai_21 = -Ai_22 @ A_21 @ Ai_11
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_Ai, 1, 1), GDN_BT,
           gdn_blk(s_A, 1, 0), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 1, 0), GDN_BT, s_tmp, GDN_BLOCK,
           gdn_blk(s_Ai, 0, 0), GDN_BT, false, true, tid);
  __syncthreads();

  // Ai_32 = -Ai_33 @ A_32 @ Ai_22
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_Ai, 2, 2), GDN_BT,
           gdn_blk(s_A, 2, 1), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 2, 1), GDN_BT, s_tmp, GDN_BLOCK,
           gdn_blk(s_Ai, 1, 1), GDN_BT, false, true, tid);
  __syncthreads();

  // Ai_43 = -Ai_44 @ A_43 @ Ai_33
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_Ai, 3, 3), GDN_BT,
           gdn_blk(s_A, 3, 2), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 3, 2), GDN_BT, s_tmp, GDN_BLOCK,
           gdn_blk(s_Ai, 2, 2), GDN_BT, false, true, tid);
  __syncthreads();

  // Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21)
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 2, 0), GDN_BT,
           gdn_blk(s_Ai, 0, 0), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 2, 1), GDN_BT,
           gdn_blk(s_Ai, 1, 0), GDN_BT, true, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 2, 0), GDN_BT, gdn_blk(s_Ai, 2, 2), GDN_BT,
           s_tmp, GDN_BLOCK, false, true, tid);
  __syncthreads();

  // Ai_42 = -Ai_44 @ (A_42 @ Ai_22 + A_43 @ Ai_32)
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 3, 1), GDN_BT,
           gdn_blk(s_Ai, 1, 1), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 3, 2), GDN_BT,
           gdn_blk(s_Ai, 2, 1), GDN_BT, true, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 3, 1), GDN_BT, gdn_blk(s_Ai, 3, 3), GDN_BT,
           s_tmp, GDN_BLOCK, false, true, tid);
  __syncthreads();

  // Ai_41 = -Ai_44 @ (A_41 @ Ai_11 + A_42 @ Ai_21 + A_43 @ Ai_31)
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 3, 0), GDN_BT,
           gdn_blk(s_Ai, 0, 0), GDN_BT, false, false, tid);
  __syncthreads();
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 3, 1), GDN_BT,
           gdn_blk(s_Ai, 1, 0), GDN_BT, true, false, tid);
  __syncthreads();
  gdn_mm16(s_tmp, GDN_BLOCK, gdn_blk(s_A, 3, 2), GDN_BT,
           gdn_blk(s_Ai, 2, 0), GDN_BT, true, false, tid);
  __syncthreads();
  gdn_mm16(gdn_blk(s_Ai, 3, 0), GDN_BT, gdn_blk(s_Ai, 3, 3), GDN_BT,
           s_tmp, GDN_BLOCK, false, true, tid);
  __syncthreads();

  // --------------------------------------------------------------------------
  // Phase 1c -- Store A_inv to global AND cast to fp16 on-chip for Phase 2.
  // The on-chip s_Aih must be 0 for OOB rows so Phase 2 dot output is 0
  // for OOB rows (matches the reference's boundary_checked A reload).
  // --------------------------------------------------------------------------
#pragma unroll
  for (int it = 0; it < (GDN_BT * GDN_BT) / GDN_THREADS; ++it) {
    const int idx = tid + it * GDN_THREADS;
    const int r = idx >> 6;
    const int c = idx & 63;
    const float v32 = s_Ai[idx];
    const __half vh = __float2half_rn(v32);
    s_Aih[idx] = (r < rows) ? vh : __float2half_rn(0.0f);
    if (r < rows) {
      A_inv[((t0 + r) * H + i_h) * GDN_BT + c] = vh;
    }
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Phase 2 -- recompute w and u. Two passes over the same [64, K|V] rhs
  // tile (re-staged between passes, with sync). The transposed + padded
  // tile layout enables vectorised __half2 loads on the k-axis inside the
  // V_DOT2 inner loop.
  // --------------------------------------------------------------------------
  const int t_rhs = GDN_BT * GDN_RHS_C / GDN_THREADS;  // = 32 for V=128
  for (int pass = 0; pass < 2; ++pass) {
    // Stage rhs[r][c] -> s_rhsT[c][r] (transposed, row-stride 66 halves).
#pragma unroll 4
    for (int it = 0; it < t_rhs; ++it) {
      const int idx = tid + it * GDN_THREADS;
      const int r = idx >> 7;          // 0..63
      const int c = idx & 127;         // 0..127
      __half val = __float2half_rn(0.0f);
      if (r < rows) {
        float x;
        if (pass == 0) {
          x = __half2float(v[((t0 + r) * H + i_h) * GDN_V + c]) *
              s_beta[r];
        } else {
          x = __half2float(k[((t0 + r) * Hg + hg) * GDN_K + c]) *
              s_beta[r] * s_eg[r];
        }
        val = __float2half_rn(x);
      }
      s_rhsT[c * (GDN_BT + GDN_RHS_R_PAD) + r] = val;
    }
    __syncthreads();

    // Dot: each thread owns two output columns (c0, c0+64) over a strip
    // of 16 contiguous rows. rg = tid / 64, c0 = tid % 64. Inner loop
    // pairs over k (m = 0..31) using V_DOT2_F32_F16 on __half2 operands.
    const int c0 = tid & 63;
    const int rg = tid >> 6;
    const int rhs_stride = GDN_BT + GDN_RHS_R_PAD;
#pragma unroll
    for (int i = 0; i < GDN_BLOCK; ++i) {
      const int r = rg * GDN_BLOCK + i;
      const int s_r0 = r * GDN_BT;
      const int s_c0 = c0 * rhs_stride;
      const int s_c1 = (c0 + 64) * rhs_stride;
      float acc0 = 0.0f;
      float acc1 = 0.0f;
#pragma unroll
      for (int m = 0; m < GDN_BT / 2; ++m) {
        const __half2 a2 = *reinterpret_cast<const __half2*>(&s_Aih[s_r0 + 2 * m]);
        const __half2 b20 = *reinterpret_cast<const __half2*>(&s_rhsT[s_c0 + 2 * m]);
        const __half2 b21 = *reinterpret_cast<const __half2*>(&s_rhsT[s_c1 + 2 * m]);
        acc0 = __builtin_amdgcn_fdot2(a2, b20, acc0, /*clamp=*/false);
        acc1 = __builtin_amdgcn_fdot2(a2, b21, acc1, /*clamp=*/false);
      }
      if (r < rows) {
        const long out_row = t0 + r;
        if (pass == 0) {
          const long base = (out_row * H + i_h) * GDN_V;
          u[base + c0] = __float2half_rn(acc0);
          u[base + c0 + 64] = __float2half_rn(acc1);
        } else {
          const long base = (out_row * H + i_h) * GDN_K;
          w[base + c0] = __float2half_rn(acc0);
          w[base + c0 + 64] = __float2half_rn(acc1);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- public symbol the orchestrator registers as
// torch.ops._rocm_C.gdn_prefill_solve_wy_rdna2. torch::Tensor arguments
// for cu_seqlens / chunk_indices may be undefined (Python None) for the
// fixed-length path.
// ============================================================================

void gdn_prefill_solve_wy_rdna2(torch::Tensor A, torch::Tensor k,
                                torch::Tensor v, torch::Tensor beta,
                                torch::Tensor g, torch::Tensor A_inv,
                                torch::Tensor w, torch::Tensor u,
                                torch::Tensor cu_seqlens,
                                torch::Tensor chunk_indices) {
  TORCH_CHECK(A.dim() == 4 && A.is_contiguous() &&
                  A.scalar_type() == at::kFloat,
              "A must be contiguous fp32 [B, T, H, BT]");
  TORCH_CHECK(k.dim() == 4 && k.is_contiguous() &&
                  k.scalar_type() == at::kHalf,
              "k must be contiguous fp16 [B, T, Hg, K]");
  TORCH_CHECK(v.dim() == 4 && v.is_contiguous() &&
                  v.scalar_type() == at::kHalf,
              "v must be contiguous fp16 [B, T, H, V]");
  TORCH_CHECK(beta.dim() == 3 && beta.is_contiguous() &&
                  beta.scalar_type() == at::kFloat,
              "beta must be contiguous fp32 [B, T, H]");
  TORCH_CHECK(g.dim() == 3 && g.is_contiguous() &&
                  g.scalar_type() == at::kFloat,
              "g must be contiguous fp32 [B, T, H]");
  TORCH_CHECK(A_inv.dim() == 4 && A_inv.is_contiguous() &&
                  A_inv.scalar_type() == at::kHalf,
              "A_inv must be contiguous fp16 [B, T, H, BT]");
  TORCH_CHECK(w.dim() == 4 && w.is_contiguous() &&
                  w.scalar_type() == at::kHalf,
              "w must be contiguous fp16 [B, T, H, K]");
  TORCH_CHECK(u.dim() == 4 && u.is_contiguous() &&
                  u.scalar_type() == at::kHalf,
              "u must be contiguous fp16 [B, T, H, V]");

  const long B = A.size(0);
  const long T = A.size(1);
  const long H = A.size(2);
  const long BT = A.size(3);
  const long Hg = k.size(2);
  const long K = k.size(3);
  const long V = v.size(3);
  TORCH_CHECK(BT == GDN_BT, "BT must be 64");
  TORCH_CHECK(K == GDN_K, "K must be 128");
  TORCH_CHECK(V == GDN_V, "V must be 128");
  TORCH_CHECK(H % Hg == 0, "H must be divisible by Hg");

  TORCH_CHECK(k.size(0) == B && k.size(1) == T,
              "k shape mismatch with A");
  TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == H,
              "v shape mismatch with A");
  TORCH_CHECK(beta.size(0) == B && beta.size(1) == T && beta.size(2) == H,
              "beta shape mismatch with A");
  TORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(2) == H,
              "g shape mismatch with A");
  TORCH_CHECK(A_inv.size(0) == B && A_inv.size(1) == T &&
                  A_inv.size(2) == H && A_inv.size(3) == BT,
              "A_inv shape mismatch");
  TORCH_CHECK(w.size(0) == B && w.size(1) == T && w.size(2) == H &&
                  w.size(3) == K,
              "w shape mismatch");
  TORCH_CHECK(u.size(0) == B && u.size(1) == T && u.size(2) == H &&
                  u.size(3) == V,
              "u shape mismatch");

  const bool is_varlen = cu_seqlens.defined();
  TORCH_CHECK(is_varlen == chunk_indices.defined(),
              "cu_seqlens and chunk_indices must both be provided (varlen) "
              "or both be None");
  long NT;
  if (is_varlen) {
    TORCH_CHECK(B == 1,
                "varlen path requires the flattened-batch convention B == 1");
    TORCH_CHECK(cu_seqlens.dim() == 1 &&
                    cu_seqlens.scalar_type() == at::kInt,
                "cu_seqlens must be int32 [N+1]");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                    chunk_indices.scalar_type() == at::kInt,
                "chunk_indices must be int32 [N_chunks, 2]");
    NT = chunk_indices.size(0);
  } else {
    NT = (T + GDN_BT - 1) / GDN_BT;
  }
  if (B == 0 || T == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned int)NT, (unsigned int)(B * H));
  auto A_ptr = A.data_ptr<float>();
  auto k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
  auto v_ptr = reinterpret_cast<const __half*>(v.data_ptr());
  auto beta_ptr = beta.data_ptr<float>();
  auto g_ptr = g.data_ptr<float>();
  auto Ai_ptr = reinterpret_cast<__half*>(A_inv.data_ptr());
  auto w_ptr = reinterpret_cast<__half*>(w.data_ptr());
  auto u_ptr = reinterpret_cast<__half*>(u.data_ptr());
  if (is_varlen) {
    gdn_prefill_solve_wy_rdna2_kernel<true>
        <<<grid, GDN_THREADS, 0, stream>>>(A_ptr, k_ptr, v_ptr, beta_ptr,
                                            g_ptr, Ai_ptr, w_ptr, u_ptr,
                                            cu_seqlens.data_ptr<int>(),
                                            chunk_indices.data_ptr<int>(), T, H,
                                            Hg);
  } else {
    gdn_prefill_solve_wy_rdna2_kernel<false>
        <<<grid, GDN_THREADS, 0, stream>>>(A_ptr, k_ptr, v_ptr, beta_ptr,
                                            g_ptr, Ai_ptr, w_ptr, u_ptr,
                                            nullptr, nullptr, T, H, Hg);
  }
}
