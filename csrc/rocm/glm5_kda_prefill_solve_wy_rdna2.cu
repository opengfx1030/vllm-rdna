// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA chunked-prefill WY-solve kernel for AMD RDNA2
// (gfx1030). GDN-style fusion of two phases per (chunk, head) workgroup:
//
// Phase 1 -- forward substitution + Schur (structure copied from
// gdn_prefill_solve_wy_rdna2.cu, which ports merge_16x16_to_64x64_inverse
// from FLA solve_tril.py): computes
//   A_inv = (I + P)^-1
// for P the POSITIVE kkt output of glm5_kda_prefill_kkt_rdna2.cu. By the
// sign bridge documented there, (I + P)^-1 == (I - A_signed)^-1 == T,
// the exact matrix the HF reference obtains from its forward-substitution
// loop (modeling_glm5_next.py:534-539). All Phase-1 math is fp32 FMA
// (fp32 operands -- no V_DOT2), identical in association order to the
// GDN kernel.
//
// Phase 2 -- WY products (modeling_glm5_next.py:540-541):
//   u[t] = T @ (v[t] * beta[t])                       ("value")
//   w[t] = T @ (k[t] * beta[t] * exp(g[t, :]))        ("k_cumdecay")
// where g is the per-(head, dim) cumsum'd log-decay (per-dim exp, the
// only structural difference from the GDN kernel whose exp(g) is a
// per-token scalar). Dots are V_DOT2_F32_F16 (__builtin_amdgcn_fdot2)
// with rhs staged transposed + bank-padded in LDS, exactly like GDN:
// the explicit fp16 rtne cast of the rhs products is the chain's data
// model (k, v are fp16 throughout the chain); the pure-fp32 torch
// reference is matched within fp16 tolerance. Documented deviation.
//
// Outputs:
//   A_inv [NT, H, 64, 64] fp16 (also kept for debugging / downstream),
//   w     [L, H, D] fp16,
//   u     [L, H, D] fp16.
//
// Workgroup = one (chunk, head); grid = (NT, H); 256 threads.
// LDS layout (~58 KB, fits the gfx1030 64 KB budget):
//   s_A    [64][64] fp32  = 16 KB    raw P tile
//   s_Ai   [64][64] fp32  = 16 KB    evolving inverse
//   s_tmp  [16][16] fp32  =  1 KB    Schur intermediate
//   s_Aih  [64][64] fp16  =  8 KB    fp16 cast for Phase 2
//   s_rhsT [128][66] fp16 = 16.5 KB  transposed + padded rhs
//   s_beta [64] fp32      = 0.25 KB
//
// Varlen-only chain: cu_seqlens [N+1], chunk_indices [NT, 2] required.
// No __launch_bounds__ / occupancy pins (project rule).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_WY_BT = 64;        // chunk size
constexpr int GLM5_WY_BLOCK = 16;     // 16x16 sub-block within BT
constexpr int GLM5_WY_D = 128;        // head dim (K == V == D)
constexpr int GLM5_WY_THREADS = 256;  // 1 workgroup per (chunk, head)
constexpr int GLM5_WY_RHS_R_PAD = 2;  // bank-conflict pad on transposed rhs

// ----- Shared-memory block-origin helpers (GDN structure) ---------------

__device__ __forceinline__ float* glm5_blk(float* base, int br, int bc) {
  return base + br * GLM5_WY_BLOCK * GLM5_WY_BT + bc * GLM5_WY_BLOCK;
}
__device__ __forceinline__ const float* glm5_blk(const float* base, int br,
                                                 int bc) {
  return base + br * GLM5_WY_BLOCK * GLM5_WY_BT + bc * GLM5_WY_BLOCK;
}

// One element of a 16x16 fp32 matmul, fp32 FMA (per thread: r = tid>>4,
// c = tid&15). accumulate adds into Z; negate applies the reference's
// `-tl.dot(...)` sign at the final write.
__device__ __forceinline__ void glm5_mm16(float* Z, int ldz,
                                          const float* X, int ldx,
                                          const float* Y, int ldy,
                                          bool accumulate, bool negate,
                                          int tid) {
  const int r = tid >> 4;
  const int c = tid & 15;
  float acc = 0.0f;
#pragma unroll
  for (int m = 0; m < GLM5_WY_BLOCK; ++m) {
    acc += X[r * ldx + m] * Y[m * ldy + c];
  }
  if (negate) acc = -acc;
  if (accumulate)
    Z[r * ldz + c] += acc;
  else
    Z[r * ldz + c] = acc;
}

__global__ void glm5_kda_prefill_solve_wy_rdna2_kernel(
    const float* __restrict__ A,            // [NT, H, BT, BT] fp32 (= P)
    const __half* __restrict__ k,           // [L, H, D] fp16
    const __half* __restrict__ v,           // [L, H, D] fp16
    const float* __restrict__ beta,         // [L, H] fp32 (sigmoid'd)
    const float* __restrict__ g,            // [L, H, D] fp32 (cumsum'd)
    __half* __restrict__ A_inv,             // [NT, H, BT, BT] fp16 out
    __half* __restrict__ w,                 // [L, H, D] fp16 out
    __half* __restrict__ u,                 // [L, H, D] fp16 out
    int H,
    const int* __restrict__ cu_seqlens,     // [N+1]
    const int* __restrict__ chunk_indices) { // [NT, 2]
  const int i_tg = blockIdx.x;  // flat chunk index
  const int i_h = blockIdx.y;   // head
  const int tid = threadIdx.x;

  // Resolve (bos, chunk-local index, rows valid).
  const int i_n = chunk_indices[i_tg * 2];
  const int i_t = chunk_indices[i_tg * 2 + 1];
  const long bos = (long)cu_seqlens[i_n];
  const long T_local = (long)cu_seqlens[i_n + 1] - bos;
  const int rows =
      (int)min((long)GLM5_WY_BT, T_local - (long)i_t * GLM5_WY_BT);
  const long t0 = bos + (long)i_t * GLM5_WY_BT;  // first token of chunk

  // ---- LDS layout ------------------------------------------------------
  __shared__ float s_A[GLM5_WY_BT * GLM5_WY_BT];
  __shared__ float s_Ai[GLM5_WY_BT * GLM5_WY_BT];
  __shared__ float s_tmp[GLM5_WY_BLOCK * GLM5_WY_BLOCK];
  __shared__ __half s_Aih[GLM5_WY_BT * GLM5_WY_BT];
  __shared__ __half s_rhsT[GLM5_WY_D * (GLM5_WY_BT + GLM5_WY_RHS_R_PAD)];
  __shared__ float s_beta[GLM5_WY_BT];

  // ---- Load s_A: contiguous [64,64] P tile for this (chunk, head) ----
#pragma unroll
  for (int it = 0; it < (GLM5_WY_BT * GLM5_WY_BT) / GLM5_WY_THREADS; ++it) {
    const int idx = tid + it * GLM5_WY_THREADS;
    const int r = idx >> 6;
    const int c = idx & 63;
    s_A[idx] =
        (r < rows)
            ? A[(((long)i_tg * H + i_h) * GLM5_WY_BT + r) * GLM5_WY_BT + c]
            : 0.0f;
  }

  // s_Ai init: diagonal blocks seeded with -P (strict lower), rest 0.
#pragma unroll
  for (int it = 0; it < (GLM5_WY_BT * GLM5_WY_BT) / GLM5_WY_THREADS; ++it) {
    const int idx = tid + it * GLM5_WY_THREADS;
    const int r = idx >> 6;
    const int c = idx & 63;
    const int br = r >> 4;
    const int bc = c >> 4;
    const int lr = r & 15;
    const int lc = c & 15;
    s_Ai[idx] = (br == bc && lr > lc) ? -s_A[idx] : 0.0f;
  }

  // Phase-2 broadcast cache: per-token beta (tail tokens -> 0).
  if (tid < GLM5_WY_BT) {
    s_beta[tid] = (tid < rows) ? beta[(t0 + tid) * H + i_h] : 0.0f;
  }
  __syncthreads();

  // ---- Phase 1a: 4 diagonal forward-substitution blocks ---------------
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int row_end = min(GLM5_WY_BLOCK, rows - b * GLM5_WY_BLOCK);
    const int base = b * GLM5_WY_BLOCK;

    for (int i = 2; i < row_end; ++i) {
      const int grow = base + i;
      const int j = tid >> 4;   // output column (local 0..15)
      const int kk = tid & 15;  // reduction lane
      const float a_k = -s_A[grow * GLM5_WY_BT + base + kk];
      float partial = a_k * s_Ai[(base + kk) * GLM5_WY_BT + base + j];
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 1);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 2);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 4);
      partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, 8);
      if (kk == 0) {
        const float a_j = -s_A[grow * GLM5_WY_BT + base + j];
        s_Ai[grow * GLM5_WY_BT + base + j] = a_j + partial;
      }
      __syncthreads();
    }

    // Identity on the block diagonal (reference adds eye after the loop).
    {
      const int lr = tid >> 4;
      const int lc = tid & 15;
      if (lr == lc) s_Ai[(base + lr) * GLM5_WY_BT + (base + lc)] += 1.0f;
    }
    __syncthreads();
  }

  // ---- Phase 1b: 6 Schur-complement blocks (fp32 FMA, GDN order) ------
  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_Ai, 1, 1), GLM5_WY_BT,
            glm5_blk(s_A, 1, 0), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 1, 0), GLM5_WY_BT, s_tmp, GLM5_WY_BLOCK,
            glm5_blk(s_Ai, 0, 0), GLM5_WY_BT, false, true, tid);
  __syncthreads();

  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_Ai, 2, 2), GLM5_WY_BT,
            glm5_blk(s_A, 2, 1), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 2, 1), GLM5_WY_BT, s_tmp, GLM5_WY_BLOCK,
            glm5_blk(s_Ai, 1, 1), GLM5_WY_BT, false, true, tid);
  __syncthreads();

  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_Ai, 3, 3), GLM5_WY_BT,
            glm5_blk(s_A, 3, 2), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 3, 2), GLM5_WY_BT, s_tmp, GLM5_WY_BLOCK,
            glm5_blk(s_Ai, 2, 2), GLM5_WY_BT, false, true, tid);
  __syncthreads();

  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 2, 0), GLM5_WY_BT,
            glm5_blk(s_Ai, 0, 0), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 2, 1), GLM5_WY_BT,
            glm5_blk(s_Ai, 1, 0), GLM5_WY_BT, true, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 2, 0), GLM5_WY_BT, glm5_blk(s_Ai, 2, 2), GLM5_WY_BT,
            s_tmp, GLM5_WY_BLOCK, false, true, tid);
  __syncthreads();

  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 3, 1), GLM5_WY_BT,
            glm5_blk(s_Ai, 1, 1), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 3, 2), GLM5_WY_BT,
            glm5_blk(s_Ai, 2, 1), GLM5_WY_BT, true, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 3, 1), GLM5_WY_BT, glm5_blk(s_Ai, 3, 3), GLM5_WY_BT,
            s_tmp, GLM5_WY_BLOCK, false, true, tid);
  __syncthreads();

  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 3, 0), GLM5_WY_BT,
            glm5_blk(s_Ai, 0, 0), GLM5_WY_BT, false, false, tid);
  __syncthreads();
  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 3, 1), GLM5_WY_BT,
            glm5_blk(s_Ai, 1, 0), GLM5_WY_BT, true, false, tid);
  __syncthreads();
  glm5_mm16(s_tmp, GLM5_WY_BLOCK, glm5_blk(s_A, 3, 2), GLM5_WY_BT,
            glm5_blk(s_Ai, 2, 0), GLM5_WY_BT, true, false, tid);
  __syncthreads();
  glm5_mm16(glm5_blk(s_Ai, 3, 0), GLM5_WY_BT, glm5_blk(s_Ai, 3, 3), GLM5_WY_BT,
            s_tmp, GLM5_WY_BLOCK, false, true, tid);
  __syncthreads();

  // ---- Phase 1c: store A_inv (fp16) + keep fp16 cast on-chip ----------
#pragma unroll
  for (int it = 0; it < (GLM5_WY_BT * GLM5_WY_BT) / GLM5_WY_THREADS; ++it) {
    const int idx = tid + it * GLM5_WY_THREADS;
    const int r = idx >> 6;
    const int c = idx & 63;
    const __half vh = __float2half_rn(s_Ai[idx]);
    s_Aih[idx] = (r < rows) ? vh : __float2half_rn(0.0f);
    if (r < rows) {
      A_inv[(((long)i_tg * H + i_h) * GLM5_WY_BT + r) * GLM5_WY_BT + c] = vh;
    }
  }
  __syncthreads();

  // ---- Phase 2: u and w via V_DOT2 over the transposed rhs ------------
  // pass 0 (u):  rhs[r][c] = v[t0+r, h, c] * beta[r]
  // pass 1 (w):  rhs[r][c] = k[t0+r, h, c] * beta[r] * exp(g[t0+r, h, c])
  // (per-dim exp -- the GLM generalization of GDN's per-token scalar).
  constexpr int rhs_stride = GLM5_WY_BT + GLM5_WY_RHS_R_PAD;
  constexpr int t_rhs = GLM5_WY_BT * GLM5_WY_D / GLM5_WY_THREADS;  // = 32
  for (int pass = 0; pass < 2; ++pass) {
    // Stage rhs[r][c] -> s_rhsT[c][r] (transposed, padded row stride).
#pragma unroll 4
    for (int it = 0; it < t_rhs; ++it) {
      const int idx = tid + it * GLM5_WY_THREADS;
      const int r = idx >> 7;    // 0..63
      const int c = idx & 127;   // 0..127
      __half val = __float2half_rn(0.0f);
      if (r < rows) {
        const long tok = t0 + r;
        const long base = (tok * H + i_h) * GLM5_WY_D + c;
        float x;
        if (pass == 0) {
          x = __half2float(v[base]) * s_beta[r];
        } else {
          x = __half2float(k[base]) * s_beta[r] * expf(g[base]);
        }
        val = __float2half_rn(x);  // rtne cast (chain data model)
      }
      s_rhsT[c * rhs_stride + r] = val;
    }
    __syncthreads();

    // Dot: thread owns a 16-row strip (rg = tid>>6) and two output
    // columns (c0 = tid&63, c0+64). Pair over the 64-token axis.
    const int c0 = tid & 63;
    const int rg = tid >> 6;
#pragma unroll
    for (int i = 0; i < GLM5_WY_BLOCK; ++i) {
      const int r = rg * GLM5_WY_BLOCK + i;
      const int s_r0 = r * GLM5_WY_BT;
      const int s_c0 = c0 * rhs_stride;
      const int s_c1 = (c0 + 64) * rhs_stride;
      float acc0 = 0.0f;
      float acc1 = 0.0f;
#pragma unroll
      for (int m = 0; m < GLM5_WY_BT / 2; ++m) {
        const __half2 a2 =
            *reinterpret_cast<const __half2*>(&s_Aih[s_r0 + 2 * m]);
        const __half2 b20 =
            *reinterpret_cast<const __half2*>(&s_rhsT[s_c0 + 2 * m]);
        const __half2 b21 =
            *reinterpret_cast<const __half2*>(&s_rhsT[s_c1 + 2 * m]);
        acc0 = __builtin_amdgcn_fdot2(a2, b20, acc0, /*clamp=*/false);
        acc1 = __builtin_amdgcn_fdot2(a2, b21, acc1, /*clamp=*/false);
      }
      if (r < rows) {
        const long base = ((t0 + r) * H + i_h) * GLM5_WY_D;
        if (pass == 0) {
          u[base + c0] = __float2half_rn(acc0);
          u[base + c0 + 64] = __float2half_rn(acc1);
        } else {
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
// Host wrapper -- torch.ops._rocm_C.glm5_kda_prefill_solve_wy_rdna2.
// Varlen-only: cu_seqlens and chunk_indices must be defined.
// ============================================================================

void glm5_kda_prefill_solve_wy_rdna2(torch::Tensor A, torch::Tensor k,
                                     torch::Tensor v, torch::Tensor beta,
                                     torch::Tensor g, torch::Tensor A_inv,
                                     torch::Tensor w, torch::Tensor u,
                                     torch::Tensor cu_seqlens,
                                     torch::Tensor chunk_indices) {
  TORCH_CHECK(A.dim() == 4 && A.is_contiguous() &&
                  A.scalar_type() == at::kFloat &&
                  A.size(2) == GLM5_WY_BT && A.size(3) == GLM5_WY_BT,
              "A must be contiguous fp32 [NT, H, 64, 64]");
  TORCH_CHECK(k.dim() == 3 && k.stride(-1) == 1 &&
                  k.scalar_type() == at::kHalf,
              "k must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(v.dim() == 3 && v.stride(-1) == 1 &&
                  v.scalar_type() == at::kHalf,
              "v must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(beta.dim() == 2 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kFloat,
              "beta must be fp32 [L, H], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1 &&
                  g.scalar_type() == at::kFloat,
              "g must be fp32 [L, H, D], contiguous in last dim");
  TORCH_CHECK(A_inv.dim() == 4 && A_inv.is_contiguous() &&
                  A_inv.scalar_type() == at::kHalf &&
                  A_inv.size(2) == GLM5_WY_BT && A_inv.size(3) == GLM5_WY_BT,
              "A_inv must be contiguous fp16 [NT, H, 64, 64]");
  TORCH_CHECK(w.dim() == 3 && w.stride(-1) == 1 &&
                  w.scalar_type() == at::kHalf,
              "w must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(u.dim() == 3 && u.stride(-1) == 1 &&
                  u.scalar_type() == at::kHalf,
              "u must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be int32 [N+1] (varlen-only chain)");
  TORCH_CHECK(chunk_indices.defined() && chunk_indices.dim() == 2 &&
                  chunk_indices.size(1) == 2 &&
                  chunk_indices.scalar_type() == at::kInt,
              "chunk_indices must be int32 [NT, 2] (varlen-only chain)");

  const long NT = A.size(0);
  const long H = A.size(1);
  const long L = k.size(0);
  const long K = k.size(2);
  const long V = v.size(2);
  TORCH_CHECK(K == GLM5_WY_D && V == GLM5_WY_D,
              "glm5_kda_prefill_solve_wy_rdna2 requires D == 128");
  TORCH_CHECK(chunk_indices.size(0) == NT,
              "chunk_indices rows must equal A's NT dim");
  TORCH_CHECK(v.size(0) == L && v.size(1) == H, "v shape mismatch");
  TORCH_CHECK(beta.size(0) == L && beta.size(1) == H, "beta shape mismatch");
  TORCH_CHECK(g.size(0) == L && g.size(1) == H && g.size(2) == K,
              "g shape mismatch");
  TORCH_CHECK(A_inv.size(0) == NT && A_inv.size(1) == H,
              "A_inv shape mismatch");
  TORCH_CHECK(w.size(0) == L && w.size(1) == H && w.size(2) == K,
              "w shape mismatch");
  TORCH_CHECK(u.size(0) == L && u.size(1) == H && u.size(2) == V,
              "u shape mismatch");
  if (L == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned int)NT, (unsigned int)H);
  glm5_kda_prefill_solve_wy_rdna2_kernel<<<grid, GLM5_WY_THREADS, 0,
                                           stream>>>(
      A.data_ptr<float>(), reinterpret_cast<const __half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(v.data_ptr()), beta.data_ptr<float>(),
      g.data_ptr<float>(),
      reinterpret_cast<__half*>(A_inv.data_ptr()),
      reinterpret_cast<__half*>(w.data_ptr()),
      reinterpret_cast<__half*>(u.data_ptr()), (int)H,
      cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>());
}
