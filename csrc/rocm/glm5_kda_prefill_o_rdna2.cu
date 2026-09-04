// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA chunked-prefill final-output kernel for AMD RDNA2
// (gfx1030). Implements modeling_glm5_next.py:552-565 per chunk:
//   attn_inter[t, v] = sum_k q[t,k]*exp(g[t,k]) * h_start[k, v]   (558)
//   attn_intra[t, s] = sum_k q[t,k]*k[s,k]*exp(g[t,k]-g[s,k])     (560)
//                      for s <= t (INCLUSIVE; triu(diag=1) zeroing)
//   o[t, v] = attn_inter[t,v] + sum_{s<=t} attn_intra[t,s]*v_new[s,v] (565)
// with q pre-scaled by 1/sqrt(128) (line 514 scales query once before
// both terms; the scale is applied to q inside this kernel, passed as an
// argument like the GDN o-kernel). g is the per-(head, dim) INCLUSIVE
// within-chunk cumsum from glm5_kda_prefill_prep_rdna2. Output is the
// RAW o [L, H, D] fp16; the caller applies RMSNormGated.
//
// NUMERICAL DESIGN (read before optimizing):
// The saved-analysis first draft staged qe = q*exp(g) and ke = k*exp(-g)
// to run both intra/inter dots through V_DOT2. ke = k*exp(-g) OVERFLOWS
// for cumsum'd g below ~-88 (at lower_bound=-5 that happens after ~18
// tokens of a 64-token chunk), and the 0*inf products would produce NaN
// in valid (s <= r) entries. The factorized form is therefore NOT used.
// Instead, mirroring what the kkt analysis chose for the same hazard:
//   * attn_intra (dot1) is a scalar fp32 FMA loop over the 128 dims with
//     the DIRECT decay expf(g_r[d] - g_s[d]). For kept entries (s <= r)
//     the cumsum is non-increasing so g_r <= g_s and exp <= 1 -- no
//     overflow. Entries with s > r can overflow (exp up to exp(range)),
//     possibly to 0*inf = NaN when staged rows are zero, but those
//     entries are overwritten with a literal 0.0f by the INCLUSIVE mask
//     step (assignment, not multiply), so no NaN leaks.
//   * attn_inter (dot2) folds exp(g_r[d]) (which is <= 1: only UNDERflow,
//     fp16 staging of qe is safe) into an fp16 qe tile and runs V_DOT2
//     against the transposed h tile h[k, v] -> s_hT[oc, k]. Underflowed
//     inter contributions are below fp16 output resolution (they are
//     either negligible vs the intra term or below the RMSNorm eps floor
//     after normalization) -- documented honest approximation.
//   * dot3 (masked attn_intra @ v_new) is V_DOT2 with bA rtne-cast to
//     fp16 (chain data model, GDN parity).
//
// Workgroup = one (V-tile, chunk, head); grid = (V/BV, NT_total, H);
// 256 threads; each owns a 4x4 fp32 sub-block of bA and a 4x2 fp32
// sub-block of bo (rb = tid/16, cb = tid%16).
//
// LDS layout (60 KB, fits the gfx1030 64 KB budget):
//   s_q  [64][128] fp16 = 16 KB  plain q (scale applied in-loop fp32)
//   s_k  [64][128] fp16 = 16 KB
//   s_hT [32][128] fp16 =  8 KB  h[k, v] transposed: s_hT[oc][k]
//   s_vT [32][64]  fp16 =  4 KB  v_new transposed: s_vT[oc][t]
//   s_gh [64][64]  fp32 = 16 KB  g half-tile (staged twice: dims
//                                0..63 then 64..127 -- a full fp32 g
//                                tile would push LDS to 76 KB)
//   s_bA aliases s_q's 16 KB slot after dot1/dot2 are done.
//
// Varlen-only chain: cu_seqlens [N+1], chunk_offsets [N+1] required.
// No __launch_bounds__ / occupancy pins (project rule).
//
// PERF NOTE (first cut): dot1 costs ~2048 accurate expf per thread plus
// ~512 for dot2's row gates. A chunk-16 variant (FlashKDA-inspired) or a
// split-d pass with precomputed decay tiles is future work.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_O_BT = 64;
constexpr int GLM5_O_BV = 32;
constexpr int GLM5_O_D = 128;         // K == V == D
constexpr int GLM5_O_THREADS = 256;
constexpr int GLM5_O_AROW = 4;
constexpr int GLM5_O_ACOL = 4;
constexpr int GLM5_O_OCOL = 2;
static_assert(GLM5_O_BT / GLM5_O_AROW * GLM5_O_BT / GLM5_O_ACOL ==
                  GLM5_O_THREADS,
              "thread count must equal (BT/AROW)*(BT/ACOL)");
static_assert(GLM5_O_BT / GLM5_O_AROW * GLM5_O_BV / GLM5_O_OCOL ==
                  GLM5_O_THREADS,
              "thread count must also equal (BT/AROW)*(BV/OCOL)");

// LDS byte sizes / offsets (extern shared).
constexpr int GLM5_LDS_Q_BYTES = GLM5_O_BT * GLM5_O_D * 2;      // 16384
constexpr int GLM5_LDS_K_BYTES = GLM5_O_BT * GLM5_O_D * 2;      // 16384
constexpr int GLM5_LDS_HT_BYTES = GLM5_O_BV * GLM5_O_D * 2;     //  8192
constexpr int GLM5_LDS_VT_BYTES = GLM5_O_BV * GLM5_O_BT * 2;    //  4096
constexpr int GLM5_LDS_GH_BYTES = GLM5_O_BT * (GLM5_O_D / 2) * 4; // 16384
constexpr int GLM5_LDS_TOTAL_BYTES =
    GLM5_LDS_Q_BYTES + GLM5_LDS_K_BYTES + GLM5_LDS_HT_BYTES +
    GLM5_LDS_VT_BYTES + GLM5_LDS_GH_BYTES;                       // 61440

__device__ __forceinline__ float glm5_fdot2(__half2 a, __half2 b,
                                            float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, /*clamp=*/false);
}

__global__ void glm5_kda_prefill_o_rdna2_kernel(
    const __half* __restrict__ q,        // [L, H, D] fp16
    const __half* __restrict__ k,        // [L, H, D] fp16
    const __half* __restrict__ v_new,    // [L, H, D] fp16 (ungated)
    const __half* __restrict__ h,        // [NT_total, H, D, D] fp16 (k, v)
    const float* __restrict__ g,         // [L, H, D] fp32 (cumsum'd)
    __half* __restrict__ o,              // [L, H, D] fp16 out
    int N_seqs, float scale, int H,
    const int* __restrict__ cu_seqlens,    // [N+1]
    const int* __restrict__ chunk_offsets) { // [N+1]
  const int i_v = blockIdx.x;
  const int i_tg = blockIdx.y;  // flat chunk index
  const int i_h = blockIdx.z;

  // Map the flat chunk to its sequence (chunk_offsets walk, GDN style).
  int i_n = 0;
  while (i_n < N_seqs && chunk_offsets[i_n + 1] <= i_tg) i_n++;
  const long bos = (long)cu_seqlens[i_n];
  const long T_local = (long)cu_seqlens[i_n + 1] - bos;
  const int i_t = i_tg - chunk_offsets[i_n];
  const long t0 = bos + (long)i_t * GLM5_O_BT;

  const int t_chunk = (int)min((long)GLM5_O_BT, T_local - (long)i_t * GLM5_O_BT);
  if (t_chunk <= 0) return;  // out is caller-zeroed; nothing to write

  const int tid = threadIdx.x;
  const int rb = tid / 16;  // 0..15 row blocks
  const int cb = tid % 16;  // 0..15 col blocks

  extern __shared__ __half smem[];
  __half* s_q = smem;                                        // [BT, D]
  __half* s_k = smem + GLM5_LDS_Q_BYTES / 2;                 // [BT, D]
  __half* s_hT = smem + (GLM5_LDS_Q_BYTES + GLM5_LDS_K_BYTES) / 2; // [BV, D]
  __half* s_vT = smem + (GLM5_LDS_Q_BYTES + GLM5_LDS_K_BYTES +
                         GLM5_LDS_HT_BYTES) / 2;             // [BV, BT]
  float* s_gh = reinterpret_cast<float*>(
      smem + (GLM5_LDS_Q_BYTES + GLM5_LDS_K_BYTES + GLM5_LDS_HT_BYTES +
              GLM5_LDS_VT_BYTES) / 2);                       // [BT, D/2]
  float* s_bA = reinterpret_cast<float*>(smem);              // aliases s_q

  const long HD = (long)H * GLM5_O_D;
  const long DD = (long)GLM5_O_D * GLM5_O_D;
  const int o_v0 = i_v * GLM5_O_BV;

  // ---- Stage s_q, s_k (plain fp16; tail rows zero-filled) -------------
#pragma unroll
  for (int it = 0; it < GLM5_O_BT * GLM5_O_D / GLM5_O_THREADS; ++it) {
    const int idx = tid + it * GLM5_O_THREADS;
    const int row = idx >> 7;
    const int d = idx & 127;
    __half qv = __float2half(0.0f);
    __half kv = __float2half(0.0f);
    if (row < t_chunk) {
      const long base = (t0 + row) * HD + (long)i_h * GLM5_O_D + d;
      qv = q[base];
      kv = k[base];
    }
    s_q[idx] = qv;
    s_k[idx] = kv;
  }

  // ---- Stage s_hT[oc][kd] = h[kd, o_v0+oc] (transpose) ----------------
#pragma unroll
  for (int it = 0; it < GLM5_O_BV * GLM5_O_D / GLM5_O_THREADS; ++it) {
    const int idx = tid + it * GLM5_O_THREADS;
    const int oc = idx / GLM5_O_D;
    const int kd = idx - oc * GLM5_O_D;
    s_hT[idx] = h[((long)i_tg * H + i_h) * DD + (long)kd * GLM5_O_D +
                  o_v0 + oc];
  }

  // ---- Stage s_vT[oc][t] = v_new[t, o_v0+oc] (transpose) --------------
#pragma unroll
  for (int it = 0; it < GLM5_O_BV * GLM5_O_BT / GLM5_O_THREADS; ++it) {
    const int idx = tid + it * GLM5_O_THREADS;
    const int oc = idx / GLM5_O_BT;
    const int tt = idx - oc * GLM5_O_BT;
    __half vv = __float2half(0.0f);
    if (tt < t_chunk) {
      vv = v_new[(t0 + tt) * HD + (long)i_h * GLM5_O_D + o_v0 + oc];
    }
    s_vT[idx] = vv;
  }
  __syncthreads();

  // ---- Accumulators ----------------------------------------------------
  float bA[GLM5_O_AROW][GLM5_O_ACOL];
  float bo[GLM5_O_AROW][GLM5_O_OCOL];
#pragma unroll
  for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
#pragma unroll
    for (int dc = 0; dc < GLM5_O_ACOL; ++dc) bA[dr][dc] = 0.0f;
#pragma unroll
    for (int do_ = 0; do_ < GLM5_O_OCOL; ++do_) bo[dr][do_] = 0.0f;
  }

  // ---- dot1 (intra, scalar fp32 FMA + direct exp) and dot2 (inter,
  // fp32 FMA with row gate exp(g_r)) over the two staged g halves ------
  for (int pass = 0; pass < 2; ++pass) {
    // Stage s_gh[row][dl] = g[t0+row, i_h, pass*64 + dl] (tail -> 0).
#pragma unroll
    for (int it = 0;
         it < GLM5_O_BT * (GLM5_O_D / 2) / GLM5_O_THREADS; ++it) {
      const int idx = tid + it * GLM5_O_THREADS;
      const int row = idx >> 6;
      const int dl = idx & 63;
      float gv = 0.0f;
      if (row < t_chunk) {
        gv = g[(t0 + row) * HD + (long)i_h * GLM5_O_D + pass * 64 + dl];
      }
      s_gh[idx] = gv;
    }
    __syncthreads();

#pragma unroll 1
    for (int dl = 0; dl < GLM5_O_D / 2; ++dl) {
      const int d = pass * (GLM5_O_D / 2) + dl;
      float qr[GLM5_O_AROW], gr[GLM5_O_AROW], egr[GLM5_O_AROW];
#pragma unroll
      for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
        const int r = rb * GLM5_O_AROW + dr;
        // scale folded into q in fp32 (reference: query*scale, fp32).
        qr[dr] = __half2float(s_q[r * GLM5_O_D + d]) * scale;
        gr[dr] = s_gh[r * (GLM5_O_D / 2) + dl];
        egr[dr] = expf(gr[dr]);  // <= 1 (dot2 row gate)
      }
      float kcol[GLM5_O_ACOL], gcol[GLM5_O_ACOL];
#pragma unroll
      for (int dc = 0; dc < GLM5_O_ACOL; ++dc) {
        const int s = cb * GLM5_O_ACOL + dc;
        kcol[dc] = __half2float(s_k[s * GLM5_O_D + d]);
        gcol[dc] = s_gh[s * (GLM5_O_D / 2) + dl];
      }
      // dot1: bA += q[r,d]*k[s,d]*exp(g_r[d]-g_s[d]) (safe direct form).
#pragma unroll
      for (int dr = 0; dr < GLM5_O_AROW; ++dr)
#pragma unroll
        for (int dc = 0; dc < GLM5_O_ACOL; ++dc) {
          bA[dr][dc] += qr[dr] * kcol[dc] * expf(gr[dr] - gcol[dc]);
        }
      // dot2: bo += q[r,d]*exp(g_r[d]) * hT[oc, d].
#pragma unroll
      for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
        const float qg = qr[dr] * egr[dr];
#pragma unroll
        for (int do_ = 0; do_ < GLM5_O_OCOL; ++do_) {
          const int oc = cb * GLM5_O_OCOL + do_;
          bo[dr][do_] +=
              qg * __half2float(s_hT[oc * GLM5_O_D + d]);
        }
      }
    }
    __syncthreads();  // before re-staging s_gh (and before s_bA spill on
                      // the last pass)
  }

  // ---- INCLUSIVE s <= r mask on bA + spill to s_bA (aliases s_q) ------
  // s_q is dead now (dot1/dot2 done). Mask zeroes via assignment so any
  // overflow/NaN from s > r entries never propagates.
#pragma unroll
  for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
    const int r = rb * GLM5_O_AROW + dr;
#pragma unroll
    for (int dc = 0; dc < GLM5_O_ACOL; ++dc) {
      const int s = cb * GLM5_O_ACOL + dc;
      const bool ok = (s <= r) && (r < t_chunk) && (s < t_chunk);
      s_bA[(long)r * GLM5_O_BT + s] = ok ? bA[dr][dc] : 0.0f;
    }
  }
  __syncthreads();

  // ---- dot3: bo += (bA.to(fp16) @ v_new) via V_DOT2 over s pairs ------
  // No extra scale: scale was folded into q for bA as well.
#pragma unroll
  for (int cp = 0; cp < GLM5_O_BT / 2; ++cp) {
    half2 bA_pair[GLM5_O_AROW];
#pragma unroll
    for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
      const long off = (long)(rb * GLM5_O_AROW + dr) * GLM5_O_BT + cp * 2;
      bA_pair[dr] = __halves2half2(__float2half_rn(s_bA[off]),
                                   __float2half_rn(s_bA[off + 1]));
    }
#pragma unroll
    for (int dr = 0; dr < GLM5_O_AROW; ++dr)
#pragma unroll
      for (int do_ = 0; do_ < GLM5_O_OCOL; ++do_) {
        const int oc = cb * GLM5_O_OCOL + do_;
        const half2 v_pair =
            *reinterpret_cast<const half2*>(&s_vT[oc * GLM5_O_BT + cp * 2]);
        bo[dr][do_] = glm5_fdot2(bA_pair[dr], v_pair, bo[dr][do_]);
      }
  }

  // ---- Store o[t, i_h, o_v0+oc] fp16 (valid rows only) ----------------
#pragma unroll
  for (int dr = 0; dr < GLM5_O_AROW; ++dr) {
    const int r = rb * GLM5_O_AROW + dr;
    if (r >= t_chunk) continue;
#pragma unroll
    for (int do_ = 0; do_ < GLM5_O_OCOL; ++do_) {
      const int oc = cb * GLM5_O_OCOL + do_;
      o[(t0 + r) * HD + (long)i_h * GLM5_O_D + o_v0 + oc] =
          __float2half(bo[dr][do_]);
    }
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- torch.ops._rocm_C.glm5_kda_prefill_o_rdna2.
// Varlen-only: cu_seqlens and chunk_offsets must be defined.
// ============================================================================

void glm5_kda_prefill_o_rdna2(torch::Tensor q, torch::Tensor k,
                              torch::Tensor v_new, torch::Tensor h,
                              torch::Tensor g, torch::Tensor o, double scale,
                              torch::Tensor cu_seqlens,
                              torch::Tensor chunk_offsets) {
  TORCH_CHECK(q.dim() == 3 && q.stride(-1) == 1 &&
                  q.scalar_type() == at::kHalf,
              "q must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(k.dim() == 3 && k.stride(-1) == 1 &&
                  k.scalar_type() == at::kHalf,
              "k must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(v_new.dim() == 3 && v_new.stride(-1) == 1 &&
                  v_new.scalar_type() == at::kHalf,
              "v_new must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(h.dim() == 4 && h.stride(-1) == 1 &&
                  h.scalar_type() == at::kHalf,
              "h must be fp16 [NT_total, H, D, D], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1 &&
                  g.scalar_type() == at::kFloat,
              "g must be fp32 [L, H, D], contiguous in last dim");
  TORCH_CHECK(o.dim() == 3 && o.stride(-1) == 1 &&
                  o.scalar_type() == at::kHalf,
              "o must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be int32 [N+1] (varlen-only chain)");
  TORCH_CHECK(chunk_offsets.defined() && chunk_offsets.dim() == 1 &&
                  chunk_offsets.scalar_type() == at::kInt &&
                  chunk_offsets.size(0) == cu_seqlens.size(0),
              "chunk_offsets must be int32 [N+1] of equal length");

  const long L = q.size(0);
  const long H = q.size(1);
  const long K = q.size(2);
  TORCH_CHECK(K == GLM5_O_D, "glm5_kda_prefill_o_rdna2 requires D == 128");
  TORCH_CHECK(k.size(0) == L && k.size(1) == H && k.size(2) == K,
              "k shape mismatch");
  TORCH_CHECK(v_new.size(0) == L && v_new.size(1) == H &&
                  v_new.size(2) == K,
              "v_new shape mismatch");
  TORCH_CHECK(g.size(0) == L && g.size(1) == H && g.size(2) == K,
              "g shape mismatch");
  TORCH_CHECK(o.size(0) == L && o.size(1) == H && o.size(2) == K,
              "o shape mismatch");
  TORCH_CHECK(h.size(1) == H && h.size(2) == GLM5_O_D &&
                  h.size(3) == GLM5_O_D,
              "h shape mismatch (expected [NT_total, H, 128, 128])");

  const long NT = h.size(0);
  const int N_seqs = (int)cu_seqlens.size(0) - 1;
  if (L == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((GLM5_O_D + GLM5_O_BV - 1) / GLM5_O_BV, (unsigned int)NT,
            (unsigned int)H);
  glm5_kda_prefill_o_rdna2_kernel<<<grid, GLM5_O_THREADS,
                                    GLM5_LDS_TOTAL_BYTES, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const __half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(v_new.data_ptr()),
      reinterpret_cast<const __half*>(h.data_ptr()),
      g.data_ptr<float>(), reinterpret_cast<__half*>(o.data_ptr()), N_seqs,
      (float)scale, (int)H, cu_seqlens.data_ptr<int>(),
      chunk_offsets.data_ptr<int>());
}
