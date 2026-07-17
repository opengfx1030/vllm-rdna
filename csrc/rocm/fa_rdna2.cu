// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Flash Attention v2 kernels for RDNA2 (gfx1030)
//
// Two kernels are provided:
//   - Decode: Br=1 per CTA, split-K across multiple CTAs per head.
//   - Prefill: Br=16 per CTA, no split-K (one CTA per (b, h_q, q_block)).
//
// Decode layout: Q[B, H_q, D], K[N, H_kv, D], V[N, H_kv, D], O[B, H_q, D].
// Prefill layout: Q[B, H_q, N_q, D], K[N, H_kv, D], V[N, H_kv, D], O[B, H_q, N_q, D].
//
// GQA: H_q may be a multiple of H_kv (kv_group_num = H_q / H_kv).
// Each query head H_q[h] attends to H_kv[h / kv_group_num].
//
// Decode algorithm (FA2 split-K, Br=1):
//   Stage 1 (per CTA, per (batch, head, kv_split)):
//     Initialize m_i = -inf, l_i = 0, O_i = 0
//     For each K/V block in this split:
//       Load K_j[BC][D], V_j[BC][D] into shared memory
//       S[BC] = Q . K_j^T * scale
//       m_new = max(m_i, max(S))
//       P[BC] = exp(S - m_new)   (write to shared memory)
//       l_new = exp(m_i - m_new) * l_i + sum(P)
//       O_i = exp(m_i - m_new) * O_i + P . V_j
//       m_i = m_new, l_i = l_new
//     Write (O_i, m_i, l_i) to global
//
//   Stage 2 (per (batch, head)):
//     Load all splits' (O_k, m_k, l_k)
//     m_global = max(m_k)
//     O_final = sum(exp(m_k - m_global) * O_k) / sum(exp(m_k - m_global) * l_k)
//
// Prefill algorithm (FA2, Br=16, no split-K):
//   Per CTA (per (batch, head, q_block)):
//     Load Q[Br x D] into shared memory
//     Initialize m[Br] = -inf, l[Br] = 0, O[Br x D] = 0
//     For each K/V block:
//       Load K[BC x D], V[BC x D] into shared memory
//       S[Br x BC] = Q . K^T * scale
//       For each row br:
//         m_new[br] = max(m[br], max_k S[br, k])
//         P[br, k] = exp(S[br, k] - m_new[br])
//         l_new[br] = exp(m[br] - m_new[br]) * l[br] + sum_k P[br, k]
//         O[br, :] = exp(m[br] - m_new[br]) * O[br, :] + sum_k P[br, k] * V[k, :]
//         m[br] = m_new[br], l[br] = l_new[br]
//     O[br, :] /= l[br]
//
// Tile sizes for gfx1030 (V620, 72 CUs, 4MB L2):
//   Decode:  Br = 1,  Bc = 64, head_dim = 128, THREADS = 128 (~33 KB smem)
//   Prefill: Br = 16, Bc = 64, head_dim = 128, THREADS = 128 (~48 KB smem)
//
// Each thread owns one O element (D=128, THREADS=128) and one P[k] element
// for the online softmax. P[k] is written to shared memory so all threads
// can use it for the O update (PV dot product).
//
// V_DOT2_F32_F16 intrinsic: 2 fp16 multiply-adds per instruction.
// fp32 accumulators for m, l, o_acc (numerical stability).

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---- Tile constants for gfx1030 -----------------------------------------
// Defined at global scope (not inside namespace vllm::fa_rdna2) because
// __global__ kernel functions in HIP do not reliably resolve namespace-
// scoped constexpr values at template instantiation time. Keeping these
// at global scope lets every kernel reference them unqualified.
constexpr int BC = 64;
constexpr int BC_256 = 32;
constexpr int MAX_SPLITS = 16;
constexpr int BR_PREFILL = 16;
constexpr int THREADS_PREFILL = 128;
constexpr int HEAD_DIM_PAGED_128 = 128;
__device__ __forceinline__ float fdot2(half2 q, half2 k, float acc) {
  return __builtin_amdgcn_fdot2(q, k, acc, false);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

// Block reductions across 4 warps (THREADS=128). shared[4] is scratch.
// Result is broadcast to all threads in the block via shared memory.
// Caller must size shared as 5 floats: shared[0..3] for warp partials,
// shared[4] for the broadcast result. The caller is responsible for
// not clobbering shared[4] between the reduction and the broadcast read.
__device__ __forceinline__ float block_reduce_max(float v, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < 4) ? shared[lane] : -INFINITY;
  if (wid == 0) v = warp_reduce_max(v);
  if (wid == 0 && lane == 0) shared[4] = v;
  __syncthreads();
  return shared[4];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < 4) ? shared[lane] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  if (wid == 0 && lane == 0) shared[4] = v;
  __syncthreads();
  return shared[4];
}

// XOR swizzle for sK/sV shared-memory indexing to reduce LDS bank conflicts.
// Pattern: d_swizzled = d ^ ((k & 7) << 4). For HEAD_DIM=128 / RDNA2 64-bank
// LDS, this shifts access patterns so 32-thread lanes reading different rows
// (k) but the same column (d) hit distinct banks rather than colliding on one.
// Smem cost: zero (same storage layout, remapped indexing on read & write).
__device__ __forceinline__ int fa_swz_d(int d, int k) {
  return d ^ ((k & 7) << 4);
}


// =====================================================================
// DECODE STAGE 1: per-CTA partials (split-K)
// =====================================================================

// =====================================================================
// DECODE STAGE 1 (PAGED): per-CTA partials, K/V read from paged blocks
// =====================================================================
//
// Mirrors fa_decode_splitk_kernel but reads K/V from a paged cache
// (vLLM's KV cache layout: [num_blocks, H_kv, D/x, block_size, x]).
// Each query token is a separate sequence in the batch with its own
// block_table and seq_len.
//
// Grid: (num_tokens, H_q, kv_splits)
//   blockIdx.x = token_idx (== seq_idx for decode — one query per seq)
//   blockIdx.y = h_q
//   blockIdx.z = split
//
// Strides for the 5D paged cache:
//   stride_kc0 = bytes/element * stride(0) = one block
//   stride_kc1 = one head within a block
//   stride_kc2 = one D/x sub-dim
//   stride_kc3 = one slot within a block
//   stride_kc4 = one x-element (usually 1)
//   x_dim = packing factor (typically 8 for fp16)
//
// HEAD_DIM = 128 specialization (the original kernel). A HEAD_DIM = 256
// variant `fa_decode_paged_splitk_kernel_256` follows below for Qwen3.5.

__global__ __launch_bounds__(128, 1) void fa_decode_paged_splitk_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int num_tokens,
    const int H_q,
    const int H_kv,
    const int kv_splits,
    const int kv_group_num,
    const float scale,
    const int sliding_window) {

  const int token_idx = blockIdx.x;
  const int h_q = blockIdx.y;
  const int split = blockIdx.z;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (token_idx >= num_tokens || h_q >= H_q || split >= kv_splits) return;

  // Per-query sequence length and block table base.
  const int seq_len = seq_lens[token_idx];
  const int* my_block_table = block_table + token_idx * max_blocks;
  if (seq_len <= 0) {
    // Empty sequence: write zeros and skip.
    if (t < HEAD_DIM_PAGED_128) {
      O_partial[((token_idx * H_q + h_q) * kv_splits + split) * HEAD_DIM_PAGED_128 + t] = 0.0f;
    }
    if (t == 0) {
      M_partial[(token_idx * H_q + h_q) * kv_splits + split] = -INFINITY;
      L_partial[(token_idx * H_q + h_q) * kv_splits + split] = 0.0f;
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ   = reinterpret_cast<half*>(smem_raw);
  half*  sK   = sQ + HEAD_DIM_PAGED_128;
  half*  sV   = sK + BC * HEAD_DIM_PAGED_128;
  float* sP   = reinterpret_cast<float*>(sV + BC * HEAD_DIM_PAGED_128);
  float* sRed = sP + BC;

  // Load Q for this query token.
  sQ[t] = Q[(token_idx * H_q + h_q) * HEAD_DIM_PAGED_128 + t];
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_acc = 0.0f;

  // Split this sequence's KV range across kv_splits CTAs.
  const int blocks_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int blk_start = split * blocks_per_split;
  const int blk_end   = min(blk_start + blocks_per_split, seq_len);

  for (int n = blk_start; n < blk_end; n += BC) {
    const int blk_size = min(BC, blk_end - n);

    // Cooperative load K[BC][D] and V[BC][D] from paged cache.
    // For each n_local in [0, blk_size), compute paged address.
    #pragma unroll
    for (int i = t; i < BC * HEAD_DIM_PAGED_128; i += 128) {
      const int n_local = i / HEAD_DIM_PAGED_128;
      const int d = i % HEAD_DIM_PAGED_128;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = my_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    __syncthreads();

    // Compute S[k] = Q . K[k]^T * scale for k in [0, blk_size).
    // For paged decode, q_idx is always at the END of the sequence (the
    // current token). So sliding_window mask is:
    // if (seq_len - 1 - kv_idx) >= sliding_window: mask. I.e. kv_idx < seq_len - sliding_window.
    float s_k = -INFINITY;
    if (t < blk_size) {
      const int kv_idx = n + t;
      const bool in_window = (sliding_window <= 0) || (kv_idx >= seq_len - sliding_window);
      if (in_window) {
        float acc = 0.0f;
        const half* sK_row = sK + t * HEAD_DIM_PAGED_128;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM_PAGED_128; d += 2) {
          half2 q2 = *reinterpret_cast<const half2*>(&sQ[d]);
          half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
          acc = fdot2(q2, k2, acc);
        }
        s_k = acc * scale;
      }
    }
    if (t < BC) sP[t] = s_k;
    __syncthreads();

    // Online softmax: block reduce max, then exp + accumulate.
    float s_for_max = (t < BC) ? sP[t] : -INFINITY;
    float m_new = block_reduce_max(s_for_max, sRed);
    m_new = fmaxf(m_i, m_new);  // also fold in previous m_i

    // Skip the online-softmax update when the entire block is masked
    // (causal or sliding window). Without this guard, exp(-INFINITY -
    // (-INFINITY)) = exp(NaN) = NaN corrupts sL and propagates to output.
    // m_new is broadcast across the block so this branch is uniform —
    // one predicated instruction, no extra syncs.
    if (m_new > -INFINITY) {
      // PV dot product: o_acc *= exp(m_i - m_new)
      float alpha = expf(m_i - m_new);
      o_acc *= alpha;

      // l_i update
      float p_k = 0.0f;
      if (t < BC) {
        p_k = (t < blk_size) ? __expf(sP[t] - m_new) : 0.0f;
        sP[t] = p_k;
      }
      __syncthreads();
      float l_new = alpha * l_i + block_reduce_sum(p_k, sRed);

      // PV accumulation: o_acc += sum_k sP[k] * sV[k, t]
      // Each thread t owns output dim t.
      float pv = 0.0f;
      if (t < HEAD_DIM_PAGED_128) {
        for (int k = 0; k < BC; k++) {
          // sV layout: [BC][HEAD_DIM]; thread t accumulates V[k][t].
          pv += sP[k] * __half2float(sV[k * HEAD_DIM_PAGED_128 + t]);
        }
      }
      o_acc += pv;

      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  // Write partial outputs.
  if (t < HEAD_DIM_PAGED_128) {
    O_partial[((token_idx * H_q + h_q) * kv_splits + split) * HEAD_DIM_PAGED_128 + t] = o_acc;
  }
  if (t == 0) {
    M_partial[(token_idx * H_q + h_q) * kv_splits + split] = m_i;
    L_partial[(token_idx * H_q + h_q) * kv_splits + split] = l_i;
  }
}

// HEAD_DIM = 256 specialization of the paged decode kernel for Qwen3.5 /
// GDN hybrid models. Same algorithm as the 128 variant but with 256-element
// query/key/value vectors and 256 threads per block.
__global__ __launch_bounds__(256, 1) void fa_decode_paged_splitk_kernel_256(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int num_tokens,
    const int H_q,
    const int H_kv,
    const int kv_splits,
    const int kv_group_num,
    const float scale,
    const int sliding_window) {

  const int token_idx = blockIdx.x;
  const int h_q = blockIdx.y;
  const int split = blockIdx.z;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (token_idx >= num_tokens || h_q >= H_q || split >= kv_splits) return;

  const int seq_len = seq_lens[token_idx];
  const int* my_block_table = block_table + token_idx * max_blocks;
  if (seq_len <= 0) {
    if (t < 256) {
      O_partial[((token_idx * H_q + h_q) * kv_splits + split) * 256 + t] = 0.0f;
    }
    if (t == 0) {
      M_partial[(token_idx * H_q + h_q) * kv_splits + split] = -INFINITY;
      L_partial[(token_idx * H_q + h_q) * kv_splits + split] = 0.0f;
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ   = reinterpret_cast<half*>(smem_raw);
  half*  sK   = sQ + 256;
  half*  sV   = sK + BC_256 * 256;
  float* sP   = reinterpret_cast<float*>(sV + BC_256 * 256);
  float* sRed = sP + BC_256;

  sQ[t] = Q[(token_idx * H_q + h_q) * 256 + t];
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_acc = 0.0f;

  const int blocks_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int blk_start = split * blocks_per_split;
  const int blk_end   = min(blk_start + blocks_per_split, seq_len);

  for (int n = blk_start; n < blk_end; n += BC_256) {
    const int blk_size = min(BC_256, blk_end - n);

    for (int i = t; i < BC_256 * 256; i += 256) {
      const int n_local = i / 256;
      const int d = i % 256;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = my_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    __syncthreads();

    // Compute S[k] = Q . K[k]^T * scale for k in [0, blk_size).
    // For paged decode, q_idx is at the END of the sequence.
    // Sliding window mask: kv_idx >= seq_len - sliding_window.
    float s_k = -INFINITY;
    if (t < blk_size) {
      const int kv_idx = n + t;
      const bool in_window = (sliding_window <= 0) || (kv_idx >= seq_len - sliding_window);
      if (in_window) {
        float acc = 0.0f;
        const half* sK_row = sK + t * 256;
        #pragma unroll
        for (int d = 0; d < 256; d += 2) {
          half2 q2 = *reinterpret_cast<const half2*>(&sQ[d]);
          half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
          acc = fdot2(q2, k2, acc);
        }
        s_k = acc * scale;
      }
    }

    float s_for_max = (t < blk_size) ? s_k : -INFINITY;
    float m_new = block_reduce_max(s_for_max, sRed);
    m_new = fmaxf(m_i, m_new);

    // Skip the online-softmax update when the entire block is masked
    // (causal or sliding window). exp(-INFINITY - (-INFINITY)) = exp(NaN)
    // would otherwise corrupt sL and propagate to output. Uniform branch.
    if (m_new > -INFINITY) {
      float exp_diff = expf(m_i - m_new);

      float p_k = (t < blk_size) ? expf(s_k - m_new) : 0.0f;
      if (t < BC_256) sP[t] = p_k;
      __syncthreads();

      float sum_p = block_reduce_sum(p_k, sRed);
      float l_new = exp_diff * l_i + sum_p;

      if (t < 256) {
        float pv = 0.0f;
        for (int k = 0; k < BC_256; k++) {
          pv += sP[k] * __half2float(sV[k * 256 + t]);
        }
        o_acc = exp_diff * o_acc + pv;
      }

      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  if (t < 256) {
    O_partial[((token_idx * H_q + h_q) * kv_splits + split) * 256 + t] = o_acc;
  }
  if (t == 0) {
    M_partial[(token_idx * H_q + h_q) * kv_splits + split] = m_i;
    L_partial[(token_idx * H_q + h_q) * kv_splits + split] = l_i;
  }
}

// =====================================================================
// DECODE STAGE 2: combine partials across splits
// =====================================================================

__global__ void fa_decode_combine_kernel(
    const float* __restrict__ O_partial,
    const float* __restrict__ M_partial,
    const float* __restrict__ L_partial,
    half* __restrict__ O,
    const int B,
    const int H_q,
    const int kv_splits,
    const int D) {

  const int b = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  if (b >= B || h_q >= H_q) return;

  extern __shared__ unsigned char smem_raw[];
  float* sM  = reinterpret_cast<float*>(smem_raw);
  float* sL  = sM + kv_splits;
  float* sWg = sL + kv_splits;

  if (t < kv_splits) {
    sM[t] = M_partial[(b * H_q + h_q) * kv_splits + t];
    sL[t] = L_partial[(b * H_q + h_q) * kv_splits + t];
  }
  __syncthreads();

  float m_global = -INFINITY;
  if (t == 0) {
    for (int s = 0; s < kv_splits; ++s) {
      m_global = fmaxf(m_global, sM[s]);
    }
    float den = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      float w = expf(sM[s] - m_global);
      sWg[s] = w;
      den += w * sL[s];
    }
    sWg[kv_splits] = den;
  }
  __syncthreads();

  if (t < D) {
    float num = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      num += sWg[s] * O_partial[((b * H_q + h_q) * kv_splits + s) * D + t];
    }
    float den = sWg[kv_splits];
    O[(b * H_q + h_q) * D + t] = __float2half_rn(num / den);
  }
}

// =====================================================================
// PREFILL STAGE 1: Br > 1, no split-K
// =====================================================================
//
// Tile sizes for prefill (gfx1030):
//   Br = 16 (Q rows per CTA — 16 * 128 halves = 4 KB for sQ)
//   Bc = 64 (K/V block, same as decode)
//   head_dim = 128
//   THREADS = 128 (4 warps)
//
// One CTA processes (b, h_q, q_block) for one kv_group. All Q rows in
// the Br block share the same K/V tiles (Q is local to the CTA,
// K/V are streamed).
//
// Memory layout (per CTA):
//   sQ  [Br x D] fp16          = Br * 128 * 2  = 4 KB
//   sK  [Bc x D] fp16          = Bc * 128 * 2  = 16 KB
//   sV  [Bc x D] fp16          = Bc * 128 * 2  = 16 KB
//   sP  [Bc x Br] fp32         = Bc * Br * 4   = 4 KB (Br=16)
//   sM  [Br] fp32              = Br * 4        = 64 B
//   sL  [Br] fp32              = Br * 4        = 64 B
//   sO  [Br x D] fp32          = Br * D * 4    = 8 KB
//   sRed [5] fp32              = 20 B
//   Total smem ≈ 48 KB (fits gfx1030 64 KB per-CU limit with 1 block/CU)
//
// Each thread owns one element of sO[br, t] where br = t / D and
// t_local = t % D. Block reductions operate along the Br dimension
// (for m, l per row).

// =====================================================================
// PREFILL (NON-PAGED) for HEAD_DIM=72 (vision ViT encoder, e.g. Qwen3.5/3.6)
// =====================================================================
//
// Cloned from fa_prefill_kernel but specialized for HEAD_DIM=72 (vision
// encoder in Qwen3.5/3.6 VL models). Key differences vs the HEAD=128
// variant:
//   * HEAD_DIM=72 (not a power of 2). 72 is even so fdot2 still works
//     (d += 2 inner loop). 72 % 32 != 0 so lane-to-lane dot accumulation
//     pattern is uneven — we still use half2 reads but threads handle
//     fractional strided indices.
//   * BC=32 (vs 64 for HEAD=128). Halves sK+sV smem cost. 72 not
//     divisible by 32, so we still use BC=32 with d += 2 chunking.
//   * BR=16, THREADS=128 — same as HEAD=128 to keep launch shape.
//   * NO swizzle: 72 isn't a power of 2 so XOR swizzle bits would alias.
//     Measure-first; apply if proven ≥5% gain.
//
// Smem budget:
//   sQ: 16*72 halves      = 2.25 KB
//   sK: 32*72 halves      = 4.50 KB
//   sV: 32*72 halves      = 4.50 KB
//   sP: 16*32 floats      = 2.00 KB
//   sM+sL+sO: 16+16+16*72 floats = 4.75 KB
//   Total                  ≈ 18.0 KB (well under 64 KB gfx1030 limit)
//

// =====================================================================
// PREFILL (PAGED): Br > 1, reads K/V from paged KV cache
// =====================================================================
//
// Mirrors fa_prefill_kernel but reads K/V from vLLM's paged KV cache
// (5D layout: [num_blocks, H_kv, D/x, block_size, x]). Each query block
// of BR_PREFILL consecutive query tokens belongs to one sequence; the
// block_table for that sequence maps KV positions to physical blocks.
//
// Supports HEAD_DIM=128 and HEAD_DIM=256. For HEAD_DIM=256 we use
// BC_256=32 to stay under gfx1030's 64KB shared memory limit.
//
// Block grid:
//   x: query block index (B * num_q_blocks_per_seq)
//   y: head index (H_q)
//   z: batch index (B)  — simplified, assumes one block per (b, q_block)
// =====================================================================

// =====================================================================
// PREFILL (PAGED, VARLEN): Multiple sequences per launch
// =====================================================================
//
// Extends the paged prefill kernel to handle multiple sequences in
// one launch. Each query block of BR_PREFILL consecutive tokens may
// span a different sequence. The CTA uses cu_query_lens to determine
// which sequence it belongs to, then reads that sequence's seq_lens and
// block_table slice.
//
// Grid:
//   x: global query block index (ceil(N_q / BR_PREFILL))
//   y: head index (H_q)
//   z: 1
//
// Per-CTA: linear search cu_query_lens to find seq_idx, then use
// seq_lens[seq_idx] and block_table[seq_idx * max_blocks + ...].
// Causal masking uses sequence-local query position.
// =====================================================================

__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_kernel_128(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    const int num_seqs,
    half* __restrict__ O,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  // Grid: (max_q_blocks_per_seq, H_q, num_seqs). Each CTA handles one
  // sequence's query block — blockIdx.z = seq_idx, blockIdx.x = q_block
  // within that sequence. This guarantees every token is covered by
  // exactly one CTA (no boundary gaps when q_block spans two sequences).
  const int seq_idx = blockIdx.z;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);

  // block_table is [num_seqs, max_blocks] — slice to this sequence.
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  // O and Q have shape [N_q, H_q, D]. Stride: token dim = H_q*D, head dim = D.
  const int stride_qo_tok = H_q * 128;
  const int stride_qo_h = 128;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
      const int br = idx / 128;
      const int d = idx % 128;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 128;
  half*  sV  = sK + BC * 128;
  float* sP  = reinterpret_cast<float*>(sV + BC * 128);
  float* sM  = sP + BC * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  // Load Q[Br x D] into shared memory.
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
      const int br = i / 128;
      const int d = i % 128;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  // Stream over KV blocks for this sequence.
  for (int n = 0; n < seq_len; n += BC) {
    const int blk_size = min(BC, seq_len - n);

    // Cooperative load sK[BC][D] and sV[BC][D] from paged cache.
    // STORAGE: write at swizzled offset to match the QK^T / PV reads below.
    for (int i = t; i < BC * 128; i += THREADS_PREFILL) {
      const int n_local = i / 128;
      const int d = i % 128;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const int d_swz = fa_swz_d(d, n_local);
        sK[n_local * 128 + d_swz] = *k_ptr;
        sV[n_local * 128 + d_swz] = *v_ptr;
      }
    }
    __syncthreads();

    // Compute sP[br, k] = sum_d sQ[br, d] * sK[k, d] * scale.
    for (int idx = t; idx < BR_PREFILL * BC; idx += THREADS_PREFILL) {
      const int br = idx / BC;
      const int k = idx % BC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        // Causal mask uses sequence-local query position.
        // Sliding window: k_global must be >= q_local - sliding_window.
        const int q_local = q_start_in_seq + br;
        const int k_global = n + k;
        const bool masked_causal = causal && (k_global > q_local);
        const bool masked_window = (sliding_window > 0)
                                   && (k_global < q_local - sliding_window);
        const bool masked = masked_causal || masked_window;
        if (!masked) {
          const half* sQ_row = sQ + br * 128;
          const half* sK_row = sK + k * 128;
          #pragma unroll
          for (int d = 0; d < 128; d += 2) {
            // Read sK at the SAME swizzled offset it was stored at.
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[fa_swz_d(d, k)]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC + k] = acc * scale;
        } else {
          sP[br * BC + k] = -INFINITY;
        }
      } else {
        sP[br * BC + k] = 0.0f;
      }
    }
    __syncthreads();

    // Online softmax update.
    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC + k]);
      }
      // Skip update if all positions in this block are masked (causal or
      // sliding window). Without this guard, exp(-INFINITY - (-INFINITY)) =
      // exp(NaN) = NaN, which corrupts sL and propagates to output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 128; ++d) {
          sO[t * 128 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = expf(sP[t * BC + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // PV dot product. Scalar half V reads at swizzled offset to match the
    // K/V storage layout and avoid bank conflicts when lanes read different
    // rows (k) at the same column (d).
    for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
      const int br = idx / 128;
      const int d = idx % 128;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC + k];
            float v_val = __half2float(sV[k * 128 + fa_swz_d(d, k)]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 128 + d] += pv;
      }
    }
    __syncthreads();
  }

  // Write output.
  for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
    const int br = idx / 128;
    const int d = idx % 128;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final_val = sO[br * 128 + d] * inv_l;
      O_row[d] = __float2half_rn(final_val);
    }
  }
}

// Sub-4096 optimized variant: BR_PREFILL=32, THREADS_PREFILL=256.
// Larger BR (32 vs 16) processes more query tokens per CTA, reducing
// grid overhead. More threads (256 vs 128) improve warp utilization
// for the larger BR. Shared memory usage: ~48 KB (fits 64 KB limit).
// Trade-off: fewer CTAs (1024/32=32 q_blocks for N=1024, so
// 32*H_q=32*16=512 CTAs total for H_q=16). Triton autotunes this
// shape and finds similar configs, so this variant is designed to
// match or beat Triton at N<4096 where the default BR=16 kernel
// underutilizes the 72 CUs of V620.
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_128_short(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    const int num_seqs,
    half* __restrict__ O,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  constexpr int BR_PREFILL = 32;
  constexpr int THREADS_PREFILL = 256;
  constexpr int BC = 32;
  constexpr int HEAD_DIM = 128;

  const int seq_idx = blockIdx.z;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);

  const int* seq_block_table = block_table + seq_idx * max_blocks;

  const int stride_qo_tok = H_q * HEAD_DIM;
  const int stride_qo_h = HEAD_DIM;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * HEAD_DIM; idx += THREADS_PREFILL) {
      const int br = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * HEAD_DIM;
  half*  sV  = sK + BC * HEAD_DIM;
  float* sP  = reinterpret_cast<float*>(sV + BC * HEAD_DIM);
  float* sM  = sP + BC * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  // Load Q[Br x D] into shared memory. Vectorized half2 loads.
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    const int N_HALVES = BR_PREFILL * HEAD_DIM;
    for (int i = t; i < N_HALVES / 2; i += THREADS_PREFILL) {
      const int br = (i * 2) / HEAD_DIM;
      const int d = (i * 2) % HEAD_DIM;
      half2 q2;
      if (br < br_size) {
        q2 = *reinterpret_cast<const half2*>(Q_row + br * stride_qo_tok + d);
      } else {
        q2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
      }
      *reinterpret_cast<half2*>(sQ + i * 2) = q2;
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  // Initialize sO (vectorized).
  for (int i = t; i < BR_PREFILL * HEAD_DIM / 2; i += THREADS_PREFILL) {
    *reinterpret_cast<float2*>(sO + i * 2) = make_float2(0.0f, 0.0f);
  }
  __syncthreads();

  // Stream over KV blocks for this sequence.
  for (int n = 0; n < seq_len; n += BC) {
    const int blk_size = min(BC, seq_len - n);

    // Vectorized half2 K/V loads. Two consecutive d values share the same
    // d_sub when x_dim >= 2, so we can load 4 contiguous bytes per thread.
    // x_dim stride_kc4 = 1 (packed), so d and d+1 are adjacent in memory.
    // STORAGE: write to sK[n_local * HEAD_DIM + fa_swz_d(d, n_local)] so the
    // swizzled index puts each thread's half2 on a different bank.
    for (int i = t; i < (BC * HEAD_DIM) / 2; i += THREADS_PREFILL) {
      const int n_local = (i * 2) / HEAD_DIM;
      const int d = (i * 2) % HEAD_DIM;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const int d_swz = fa_swz_d(d, n_local);
        *reinterpret_cast<half2*>(sK + n_local * HEAD_DIM + d_swz) =
            *reinterpret_cast<const half2*>(k_ptr);
        *reinterpret_cast<half2*>(sV + n_local * HEAD_DIM + d_swz) =
            *reinterpret_cast<const half2*>(v_ptr);
      }
    }
    __syncthreads();

    // Compute sP[br, k] = sum_d sQ[br, d] * sK[k, d] * scale.
    for (int idx = t; idx < BR_PREFILL * BC; idx += THREADS_PREFILL) {
      const int br = idx / BC;
      const int k = idx % BC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        const int q_local = q_start_in_seq + br;
        const int k_global = n + k;
        const bool masked_causal = causal && (k_global > q_local);
        const bool masked_window = (sliding_window > 0)
                                   && (k_global < q_local - sliding_window);
        const bool masked = masked_causal || masked_window;
        if (!masked) {
          const half* sQ_row = sQ + br * HEAD_DIM;
          const half* sK_row = sK + k * HEAD_DIM;
          #pragma unroll
          for (int d = 0; d < HEAD_DIM; d += 2) {
            // Read sK at the SAME swizzled offset it was stored at.
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[fa_swz_d(d, k)]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC + k] = acc * scale;
        } else {
          sP[br * BC + k] = -INFINITY;
        }
      } else {
        sP[br * BC + k] = 0.0f;
      }
    }
    __syncthreads();

    // Online softmax update. FA2-style: process BR_PREFILL rows using
    // one wavefront at a time. Each row is reduced via 32-lane butterfly
    // (__shfl_xor, which compiles to DPP v_mov_b32_dpp on RDNA2).
    // Rows are processed sequentially: each iteration uses one wave to
    // reduce one row. This keeps the butterfly correct (all 32 lanes on
    // the same row) while still using only 5 shfl instructions per reduce.
    for (int br = 0; br < BR_PREFILL; ++br) {
      if (t < 32 && br < br_size) {
        const int lane = t;  // 0..31
        float p_val = (lane < blk_size) ? sP[br * BC + lane] : -INFINITY;

        // 32-lane butterfly max reduction (5 steps, all 32 lanes participate).
        float row_max = p_val;
        row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 8));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 16));
        // All 32 lanes now hold row_max.

        if (row_max > -INFINITY) {
          float old_m = sM[br];
          float new_m = fmaxf(old_m, row_max);
          float exp_diff = expf(old_m - new_m);

          float exp_p = (lane < blk_size) ? expf(p_val - new_m) : 0.0f;

          // 32-lane butterfly sum reduction.
          float sum_p = exp_p;
          sum_p += __shfl_xor(sum_p, 1);
          sum_p += __shfl_xor(sum_p, 2);
          sum_p += __shfl_xor(sum_p, 4);
          sum_p += __shfl_xor(sum_p, 8);
          sum_p += __shfl_xor(sum_p, 16);

          // Lane 0 writes sM/sL.
          if (lane == 0) {
            sL[br] = exp_diff * sL[br] + sum_p;
            sM[br] = new_m;
          }
          // Scale sO by exp_diff. All 32 lanes stride through HEAD_DIM.
          // With HEAD_DIM=128 and 32 lanes, each lane handles 4 d-values.
          for (int d = lane; d < HEAD_DIM; d += 32) {
            sO[br * HEAD_DIM + d] *= exp_diff;
          }
          // Write exp_p back to sP for PV loop.
          if (lane < blk_size) {
            sP[br * BC + lane] = exp_p;
          }
        } else {
          // All-masked block: zero sP so PV loop contributes nothing.
          if (lane < blk_size) {
            sP[br * BC + lane] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // PV dot product. Vectorized half2 V reads with XOR swizzle to match
    // the K/V storage layout and eliminate bank conflicts when lanes read
    // different rows (k) at the same column (d).
    for (int idx = t; idx < BR_PREFILL * (HEAD_DIM / 2); idx += THREADS_PREFILL) {
      const int br = idx / (HEAD_DIM / 2);
      const int d_pair = idx % (HEAD_DIM / 2);
      const int d = d_pair * 2;
      if (br < br_size) {
        float pv0 = 0.0f;
        float pv1 = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC + k];
            half2 v2 = *reinterpret_cast<const half2*>(sV + k * HEAD_DIM + fa_swz_d(d, k));
            float2 v_f = __half22float2(v2);
            pv0 = fmaf(p_val, v_f.x, pv0);
            pv1 = fmaf(p_val, v_f.y, pv1);
          }
        }
        sO[br * HEAD_DIM + d] += pv0;
        sO[br * HEAD_DIM + d + 1] += pv1;
      }
    }
    __syncthreads();
  }

  // Write output. Vectorized half2 writes.
  for (int idx = t; idx < BR_PREFILL * (HEAD_DIM / 2); idx += THREADS_PREFILL) {
    const int br = idx / (HEAD_DIM / 2);
    const int d_pair = idx % (HEAD_DIM / 2);
    const int d = d_pair * 2;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final0 = sO[br * HEAD_DIM + d] * inv_l;
      float final1 = sO[br * HEAD_DIM + d + 1] * inv_l;
      *reinterpret_cast<half2*>(O_row + d) = __floats2half2_rn(final0, final1);
    }
  }
}

// HEAD_DIM = 256 varlen variant.
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_256(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    const int num_seqs,
    half* __restrict__ O,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  // Grid: (max_q_blocks_per_seq, H_q, num_seqs). Each CTA handles one
  // sequence's query block — blockIdx.z = seq_idx, blockIdx.x = q_block
  // within that sequence. This guarantees every token is covered by
  // exactly one CTA (no boundary gaps when q_block spans two sequences).
  const int seq_idx = blockIdx.z;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  const int stride_qo_tok = H_q * 256;
  const int stride_qo_h = 256;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
      const int br = idx / 256;
      const int d = idx % 256;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 256;
  half*  sV  = sK + BC_256 * 256;
  float* sP  = reinterpret_cast<float*>(sV + BC_256 * 256);
  float* sM  = sP + BC_256 * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL * 256; i += 256) {
      const int br = i / 256;
      const int d = i % 256;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 256; i += 256) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  for (int n = 0; n < seq_len; n += BC_256) {
    const int blk_size = min(BC_256, seq_len - n);

    for (int i = t; i < BC_256 * 256; i += 256) {
      const int n_local = i / 256;
      const int d = i % 256;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * BC_256; idx += 256) {
      const int br = idx / BC_256;
      const int k = idx % BC_256;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        const int q_local = q_start_in_seq + br;
        const int k_global = n + k;
        const bool masked_causal = causal && (k_global > q_local);
        const bool masked_window = (sliding_window > 0)
                                   && (k_global < q_local - sliding_window);
        const bool masked = masked_causal || masked_window;
        if (!masked) {
          const half* sQ_row = sQ + br * 256;
          const half* sK_row = sK + k * 256;
          #pragma unroll
          for (int d = 0; d < 256; d += 2) {
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC_256 + k] = acc * scale;
        } else {
          sP[br * BC_256 + k] = -INFINITY;
        }
      } else {
        sP[br * BC_256 + k] = 0.0f;
      }
    }
    __syncthreads();

    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC_256 + k]);
      }
      // Skip update if all KV positions in this block are masked (causal).
      // Without this guard, exp(-INFINITY - (-INFINITY)) = exp(NaN) = NaN,
      // which corrupts sL and propagates to L_partial -> NaN output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC_256 + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 256; ++d) {
          sO[t * 256 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = expf(sP[t * BC_256 + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
      const int br = idx / 256;
      const int d = idx % 256;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC_256; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC_256 + k];
            float v_val = __half2float(sV[k * 256 + d]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 256 + d] += pv;
      }
    }
    __syncthreads();
  }

  for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
    const int br = idx / 256;
    const int d = idx % 256;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final_val = sO[br * 256 + d] * inv_l;
      O_row[d] = __float2half_rn(final_val);
    }
  }
}

// =====================================================================
// SPLIT-K PREFILL (PAGED, VARLEN): Multiple CTAs per (seq, q_block, h_q)
// =====================================================================
//
// Partitions the KV sequence dimension across multiple CTAs when
// seq_len is large. Each split CTA processes a different KV range and
// outputs partial O (unnormalized), M (row max), L (row sum).
// A separate reduction kernel combines the partials using the standard
// online softmax merge formula.
//
// Grid:
//   x: max_q_blocks_per_seq
//   y: H_q
//   z: num_seqs * kv_splits
// where the z dim encodes both seq_idx and split_idx.
//
// This increases grid utilization from H_q to H_q * kv_splits, which is
// important for large seq_len where the non-split kernel only launches
// H_q CTAs per q_block (40 for Qwen3.5, underutilizing 72 CUs).
// =====================================================================

__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_splitk_kernel_128(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    const int num_seqs,
    const int kv_splits,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  const int seq_idx = blockIdx.z / kv_splits;
  const int split_idx = blockIdx.z % kv_splits;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  // Split the KV range [0, seq_len) into kv_splits chunks.
  const int kv_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int kv_start = split_idx * kv_per_split;
  const int kv_end = min(kv_start + kv_per_split, seq_len);
  if (kv_start >= kv_end) {
    // Empty split: write zero partials.
    const int partial_base_empty = ((q_start_global * H_q + h_q) * BR_PREFILL * kv_splits) + split_idx;
    for (int br = 0; br < BR_PREFILL; ++br) {
      M_partial[partial_base_empty + br * kv_splits] = -INFINITY;
      L_partial[partial_base_empty + br * kv_splits] = 0.0f;
      for (int d = t; d < 128; d += 128) {
        O_partial[(partial_base_empty + br * kv_splits) * 128 + d] = 0.0f;
      }
    }
    return;
  }

  // O and Q have shape [N_q, H_q, D]. Stride: token dim = H_q*D, head dim = D.
  const int stride_qo_tok = H_q * 128;
  const int stride_qo_h = 128;

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 128;
  half*  sV  = sK + BC * 128;
  float* sP  = reinterpret_cast<float*>(sV + BC * 128);
  float* sM  = sP + BC * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  // Load Q[Br x D] into shared memory.
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
      const int br = i / 128;
      const int d = i % 128;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  // Stream over this split's KV range.
  for (int n = kv_start; n < kv_end; n += BC) {
    const int blk_size = min(BC, kv_end - n);

    for (int i = t; i < BC * 128; i += THREADS_PREFILL) {
      const int n_local = i / 128;
      const int d = i % 128;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * BC; idx += THREADS_PREFILL) {
      const int br = idx / BC;
      const int k = idx % BC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        const int q_local = q_start_in_seq + br;
        const int k_global = n + k;
        const bool masked_causal = causal && (k_global > q_local);
        const bool masked_window = (sliding_window > 0)
                                   && (k_global < q_local - sliding_window);
        const bool masked = masked_causal || masked_window;
        if (!masked) {
          const half* sQ_row = sQ + br * 128;
          const half* sK_row = sK + k * 128;
          #pragma unroll
          for (int d = 0; d < 128; d += 2) {
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC + k] = acc * scale;
        } else {
          sP[br * BC + k] = -INFINITY;
        }
      } else {
        sP[br * BC + k] = 0.0f;
      }
    }
    __syncthreads();

    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC + k]);
      }
      // Skip update if all KV positions in this block are masked (causal).
      // Without this guard, exp(-INFINITY - (-INFINITY)) = exp(NaN) = NaN,
      // which corrupts sL and propagates to L_partial -> NaN output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 128; ++d) {
          sO[t * 128 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = expf(sP[t * BC + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
      const int br = idx / 128;
      const int d = idx % 128;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC + k];
            float v_val = __half2float(sV[k * 128 + d]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 128 + d] += pv;
      }
    }
    __syncthreads();
  }

  // Write partial O (unnormalized), M, L.
  // Layout: [N, H_q, BR_PREFILL, kv_splits, D] so each br row has its own slot.
  // Each thread t (t < BR_PREFILL) writes its own br row, and all threads
  // cooperate to write the D-dim row using a strided loop.
  const int partial_base = ((q_start_global * H_q + h_q) * BR_PREFILL * kv_splits) + split_idx;
  if (t < BR_PREFILL) {
    const int br = t;
    const int64_t slot = partial_base + br * kv_splits;
    const bool active = (br < br_size);
    if (active) {
      M_partial[slot] = sM[br];
      L_partial[slot] = sL[br];
    } else {
      M_partial[slot] = -INFINITY;
      L_partial[slot] = 0.0f;
    }
  }
  for (int br = 0; br < BR_PREFILL; ++br) {
    const int64_t slot = partial_base + br * kv_splits;
    const bool active = (br < br_size);
    for (int d = t; d < 128; d += THREADS_PREFILL) {
      O_partial[slot * 128 + d] = active ? sO[br * 128 + d] : 0.0f;
    }
  }
}

// Reduction kernel: combine kv_splits partials per (token, h_q, br) into final O.
// Partial layout: [N, H_q, BR_PREFILL, kv_splits, D].
// Grid: (max_q_blocks, H_q, 1). One CTA per (q_block, h_q); loops over BR_PREFILL
// query rows within the block.
//
// IMPORTANT: The partial write address uses q_start_global (= q_block*BR_PREFILL),
// NOT the per-br token. All br rows in a q_block share the same q_start_global.
// The write address for (q_start_global, h_q, br, split) is:
//   ((q_start_global * H_q + h_q) * BR_PREFILL + br) * kv_splits + split
// The output address uses the per-br token: token = q_start_global + br.
__global__ void fa_prefill_paged_varlen_splitk_reduce_kernel_128(
    const float* __restrict__ O_partial,
    const float* __restrict__ M_partial,
    const float* __restrict__ L_partial,
    half* __restrict__ O,
    const int max_q_blocks,
    const int H_q,
    const int kv_splits,
    const int stride_qo_tok,
    const int stride_qo_h) {
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int d = threadIdx.x;
  if (q_block >= max_q_blocks || h_q >= H_q || d >= 128) return;

  const int q_start_global = q_block * BR_PREFILL;

  for (int br = 0; br < BR_PREFILL; ++br) {
    const int token = q_start_global + br;
    // Partial slot base: uses q_start_global (shared across all br in this q_block)
    // + br offset within the BR_PREFILL dimension. NOT token.
    const int64_t slot_base =
        ((int64_t)(q_start_global * H_q + h_q) * BR_PREFILL + br) * (int64_t)kv_splits;

    float m_global = -INFINITY;
    for (int s = 0; s < kv_splits; ++s) {
      m_global = fmaxf(m_global, M_partial[slot_base + s]);
    }
    if (m_global == -INFINITY) {
      O[token * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      continue;
    }
    float l_combined = 0.0f;
    float o_combined = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      const float w = expf(M_partial[slot_base + s] - m_global);
      l_combined += w * L_partial[slot_base + s];
      o_combined += w * O_partial[(slot_base + s) * 128 + d];
    }
    O[token * stride_qo_tok + h_q * stride_qo_h + d] =
        __float2half_rn(o_combined / l_combined);
  }
}

// HEAD_DIM = 256 split-K prefill kernel.
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_splitk_kernel_256(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    const int num_seqs,
    const int kv_splits,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  const int seq_idx = blockIdx.z / kv_splits;
  const int split_idx = blockIdx.z % kv_splits;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  const int kv_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int kv_start = split_idx * kv_per_split;
  const int kv_end = min(kv_start + kv_per_split, seq_len);
  if (kv_start >= kv_end) {
    const int partial_base_empty = ((q_start_global * H_q + h_q) * BR_PREFILL * kv_splits) + split_idx;
    for (int br = 0; br < BR_PREFILL; ++br) {
      M_partial[partial_base_empty + br * kv_splits] = -INFINITY;
      L_partial[partial_base_empty + br * kv_splits] = 0.0f;
      for (int d = t; d < 256; d += 256) {
        O_partial[(partial_base_empty + br * kv_splits) * 256 + d] = 0.0f;
      }
    }
    return;
  }

  const int stride_qo_tok = H_q * 256;
  const int stride_qo_h = 256;

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 256;
  half*  sV  = sK + BC_256 * 256;
  float* sP  = reinterpret_cast<float*>(sV + BC_256 * 256);
  float* sM  = sP + BC_256 * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL * 256; i += 256) {
      const int br = i / 256;
      const int d = i % 256;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 256; i += 256) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  for (int n = kv_start; n < kv_end; n += BC_256) {
    const int blk_size = min(BC_256, kv_end - n);

    for (int i = t; i < BC_256 * 256; i += 256) {
      const int n_local = i / 256;
      const int d = i % 256;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * BC_256; idx += 256) {
      const int br = idx / BC_256;
      const int k = idx % BC_256;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        const int q_local = q_start_in_seq + br;
        const int k_global = n + k;
        const bool masked_causal = causal && (k_global > q_local);
        const bool masked_window = (sliding_window > 0)
                                   && (k_global < q_local - sliding_window);
        const bool masked = masked_causal || masked_window;
        if (!masked) {
          const half* sQ_row = sQ + br * 256;
          const half* sK_row = sK + k * 256;
          #pragma unroll
          for (int d = 0; d < 256; d += 2) {
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC_256 + k] = acc * scale;
        } else {
          sP[br * BC_256 + k] = -INFINITY;
        }
      } else {
        sP[br * BC_256 + k] = 0.0f;
      }
    }
    __syncthreads();

    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC_256 + k]);
      }
      // Skip update if all KV positions in this block are masked (causal).
      // Without this guard, exp(-INFINITY - (-INFINITY)) = exp(NaN) = NaN,
      // which corrupts sL and propagates to L_partial -> NaN output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC_256 + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 256; ++d) {
          sO[t * 256 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = expf(sP[t * BC_256 + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
      const int br = idx / 256;
      const int d = idx % 256;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC_256; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC_256 + k];
            float v_val = __half2float(sV[k * 256 + d]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 256 + d] += pv;
      }
    }
    __syncthreads();
  }

  const int partial_base = ((q_start_global * H_q + h_q) * BR_PREFILL * kv_splits) + split_idx;
  if (t < BR_PREFILL) {
    const int br = t;
    const int64_t slot = partial_base + br * kv_splits;
    const bool active = (br < br_size);
    if (active) {
      M_partial[slot] = sM[br];
      L_partial[slot] = sL[br];
    } else {
      M_partial[slot] = -INFINITY;
      L_partial[slot] = 0.0f;
    }
  }
  for (int br = 0; br < BR_PREFILL; ++br) {
    const int64_t slot = partial_base + br * kv_splits;
    const bool active = (br < br_size);
    for (int d = t; d < 256; d += 256) {
      O_partial[slot * 256 + d] = active ? sO[br * 256 + d] : 0.0f;
    }
  }
}

// HEAD_DIM = 256 reduction kernel.
// Partial layout: [N, H_q, BR_PREFILL, kv_splits, D].
// Grid: (max_q_blocks, H_q, 1). One CTA per (q_block, h_q); loops over BR_PREFILL
// query rows within the block.
//
// IMPORTANT: The partial write address uses q_start_global (= q_block*BR_PREFILL),
// NOT the per-br token. All br rows in a q_block share the same q_start_global.
__global__ void fa_prefill_paged_varlen_splitk_reduce_kernel_256(
    const float* __restrict__ O_partial,
    const float* __restrict__ M_partial,
    const float* __restrict__ L_partial,
    half* __restrict__ O,
    const int max_q_blocks,
    const int H_q,
    const int kv_splits,
    const int stride_qo_tok,
    const int stride_qo_h) {
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int d = threadIdx.x;
  if (q_block >= max_q_blocks || h_q >= H_q || d >= 256) return;

  const int q_start_global = q_block * BR_PREFILL;

  for (int br = 0; br < BR_PREFILL; ++br) {
    const int token = q_start_global + br;
    // Partial slot base: uses q_start_global (shared across all br in this q_block)
    // + br offset within the BR_PREFILL dimension. NOT token.
    const int64_t slot_base =
        ((int64_t)(q_start_global * H_q + h_q) * BR_PREFILL + br) * (int64_t)kv_splits;

    float m_global = -INFINITY;
    for (int s = 0; s < kv_splits; ++s) {
      m_global = fmaxf(m_global, M_partial[slot_base + s]);
    }
    if (m_global == -INFINITY) {
      O[token * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      continue;
    }
    float l_combined = 0.0f;
    float o_combined = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      const float w = expf(M_partial[slot_base + s] - m_global);
      l_combined += w * L_partial[slot_base + s];
      o_combined += w * O_partial[(slot_base + s) * 256 + d];
    }
    O[token * stride_qo_tok + h_q * stride_qo_h + d] =
        __float2half_rn(o_combined / l_combined);
  }
}

// =====================================================================
// Public entry point
// =====================================================================

// =====================================================================
// PAGED DECODE HOST WRAPPER
// =====================================================================
//
// Reads K/V from vLLM's paged KV cache (5D layout):
//   key_cache:   [num_blocks, H_kv, D/x, block_size, x]
//   value_cache: [num_blocks, H_kv, D/x, block_size, x]
//   block_table: [num_tokens, max_blocks] (int32)
//   seq_lens:    [num_tokens] (int32) — KV length per query token
//
// Output: O [num_tokens, H_q, D] fp16
 torch::Tensor fa_rdna2_decode_paged(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t kv_splits,
    int64_t sliding_window) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D [num_blocks, H_kv, D/x, block_size, x]");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256,
              "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(kv_splits >= 1 && kv_splits <= MAX_SPLITS,
              "kv_splits must be in [1, 16]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  const int x_dim = key_cache.size(4);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);

  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());

  auto O_partial = torch::empty({num_tokens, H_q, (int)kv_splits, D}, float_opts);
  auto M_partial = torch::empty({num_tokens, H_q, (int)kv_splits}, float_opts);
  auto L_partial = torch::zeros({num_tokens, H_q, (int)kv_splits}, float_opts);
  auto O = torch::empty({num_tokens, H_q, D}, half_opts);

  dim3 grid1(num_tokens, H_q, (int)kv_splits);
  const float reduction_bytes = (float)((D + D + D) * sizeof(float) + D * sizeof(float) * 2 + D * sizeof(float));
  (void)reduction_bytes;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC * HEAD_DIM * sizeof(half) * 2
                 + BC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel<<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC_LOC * HEAD_DIM * sizeof(half) * 2
                 + BC_LOC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel_256<<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window);
  }
  hipError_t err1 = hipGetLastError();
  TORCH_CHECK(err1 == hipSuccess,
              "fa_rdna2 paged splitk launch failed: ",
              hipGetErrorString(err1),
              " (grid=", grid1.x, ",", grid1.y, ",", grid1.z, ")");

  // Combine kernel — same as contiguous case, just `B` → `num_tokens`.
  dim3 grid2(num_tokens, H_q);
  dim3 block2(D);  // one thread per output dim element
  size_t smem2 = (3 * (int)kv_splits + 1) * sizeof(float);
  fa_decode_combine_kernel<<<grid2, block2, smem2, stream.stream()>>>(
      (const float*)O_partial.data_ptr(),
      (const float*)M_partial.data_ptr(),
      (const float*)L_partial.data_ptr(),
      (half*)O.data_ptr(),
      num_tokens, H_q, (int)kv_splits, D);
  hipError_t err2 = hipGetLastError();
  TORCH_CHECK(err2 == hipSuccess, "fa_rdna2 paged combine launch failed: ",
              hipGetErrorString(err2));

  return O;
}


// =====================================================================
// PAGED PREFILL HOST WRAPPER
// =====================================================================
//
// Reads K/V from vLLM's paged KV cache (5D layout):
//   key_cache:   [num_blocks, H_kv, D/x, block_size, x]
//   value_cache: [num_blocks, H_kv, D/x, block_size, x]  (after reinterp)
//   block_table: [num_tokens, max_blocks] (int32) — per-token block indices
//   seq_lens:    [num_tokens] (int32) — KV length per query token
//   Q:           [num_tokens, H_q, D] fp16 query tensor (Br > 1 for prefill)
//
// Output: O [num_tokens, H_q, D] fp16 attention output
//
// Supports HEAD_DIM=128 and HEAD_DIM=256.
//
torch::Tensor fa_rdna2_prefill_paged_varlen(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  const int x_dim = key_cache.size(4);
  const int num_seqs = seq_lens.size(0);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");

  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());
  auto O = torch::zeros({num_tokens, H_q, D}, half_opts);

  // Grid: (max_q_blocks_per_seq, H_q, num_seqs). max_q_blocks_per_seq must
  // be large enough for the longest sequence's query blocks. We compute it
  // from the maximum per-sequence query length derived from cu_query_lens
  // and seq_lens (the max is stored implicitly in cu_query_lens[num_seqs]).
  // For simplicity we use ceil(num_tokens / BR_PREFILL) which is an upper
  // bound — some CTAs will early-exit when q_block >= seq_query_len.
  const int max_q_blocks = (num_tokens + BR_PREFILL - 1)
                           / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC * HEAD_DIM * sizeof(half) * 2
                + BC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_128),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_128<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (half*)O.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC_LOC * HEAD_DIM * sizeof(half) * 2
                + BC_LOC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_256<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (half*)O.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen launch failed: ",
              hipGetErrorString(err));
  return O;
}

// Sub-4096 optimized varlen prefill host wrapper (HEAD_DIM=128 only).
// Uses BR_PREFILL=32, THREADS_PREFILL=256 for better grid utilization
// at short sequence lengths. Only valid for D=128; for D=256 the caller
// should use fa_rdna2_prefill_paged_varlen with the >=4096 path.
torch::Tensor fa_rdna2_prefill_paged_varlen_short(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128, "D must be 128 for short variant");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  const int x_dim = key_cache.size(4);
  const int num_seqs = seq_lens.size(0);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");

  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());
  auto O = torch::zeros({num_tokens, H_q, D}, half_opts);

  constexpr int BR_PREFILL = 32;
  constexpr int HEAD_DIM = 128;
  constexpr int BC = 32;
  constexpr int THREADS = 256;

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1) / BR_PREFILL;
  dim3 grid(max_q_blocks, H_q, num_seqs);
  dim3 block(THREADS);
  size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
              + BC * HEAD_DIM * sizeof(half) * 2
              + BC * BR_PREFILL * sizeof(float)
              + BR_PREFILL * sizeof(float) * 3
              + BR_PREFILL * HEAD_DIM * sizeof(float)
              + (THREADS / 32 + 1) * sizeof(float);
  hipFuncSetAttribute(
      reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_128_short),
      hipFuncAttributeMaxDynamicSharedMemorySize, smem);
  fa_prefill_paged_varlen_kernel_128_short<<<grid, block, smem, stream.stream()>>>(
      (const half*)Q.data_ptr(),
      (const half*)key_cache.data_ptr(),
      (const half*)value_cache.data_ptr(),
      (const int*)block_table.data_ptr(),
      (const int*)cu_query_lens.data_ptr(),
      (const int*)seq_lens.data_ptr(),
      (int)key_cache.stride(0),
      (int)key_cache.stride(1),
      (int)key_cache.stride(2),
      (int)key_cache.stride(3),
      (int)key_cache.stride(4),
      max_blocks,
      (int)block_size,
      x_dim,
      num_seqs,
      (half*)O.data_ptr(),
      H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);

  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen short launch failed: ",
              hipGetErrorString(err));
  return O;
}

// Split-K paged prefill varlen host wrapper.
// Partitions the KV sequence across kv_splits CTAs, each producing
// partial O/M/L. A reduction kernel combines them into the final O.
torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t kv_splits,
    int64_t sliding_window) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
  TORCH_CHECK(kv_splits >= 1 && kv_splits <= MAX_SPLITS,
              "kv_splits must be in [1, 16]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  const int x_dim = key_cache.size(4);
  const int num_seqs = seq_lens.size(0);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");

  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto O = torch::empty({num_tokens, H_q, D}, half_opts);
  // Partial layout: [N, H_q, kv_splits, D] — each query token owns one
  // slot per (head, kv_split). The splitk kernel indexes
  //   ((token_idx * H_q + h_q) * kv_splits + split) * D + t
  // which matches this layout. (The earlier [N, H_q, BR_PREFILL, kv_splits,
  // D] shape allocated BR_PREFILL extra rows per token, wasting
  // BR_PREFILL x memory and OOM-ing at 16k prefill with cudagraphs.)
  auto O_partial = torch::empty({num_tokens, H_q,
                                 (int)kv_splits, D}, float_opts);
  auto M_partial = torch::empty({num_tokens, H_q,
                                 (int)kv_splits}, float_opts);
  auto L_partial = torch::empty({num_tokens, H_q,
                                 (int)kv_splits}, float_opts);

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1)
                           / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC * HEAD_DIM * sizeof(half) * 2
                + BC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_128),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_128<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (int)kv_splits,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
    // Reduction: one CTA per (q_block, h_q), 128 threads; loops over BR_PREFILL rows.
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_128<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)O.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC_LOC * HEAD_DIM * sizeof(half) * 2
                + BC_LOC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_256<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (int)kv_splits,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_256<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)O.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen splitk launch failed: ",
              hipGetErrorString(err));
  return O;
}

