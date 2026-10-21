// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA chunked-prefill inter-chunk recurrence kernel for
// AMD RDNA2 (gfx1030). Implements the torch reference loop at
// modeling_glm5_next.py:551-569, the parts that span chunks:
//   v_prime  = k_cumdecay @ state_start        (line 562)
//   v_new    = value - v_prime                 (line 563)  [value == u]
//   state    = state * exp(g_last)             (line 566-567)
//            + (k * exp(g_last - g))^T @ v_new (line 568)
// with g the per-(head, dim) cumsum'd log-decay (per-dim g_last).
//
// Workgroup = one (V-tile, sequence x head); grid = (V/BV, N*H);
// 256 threads = 32 v-rows x 8 k-slices, same lane layout as
// gdn_prefill_delta_h_rdna2.cu (lane_v = tid>>3, lane_ks = tid&7), but
// with the GLM state ordering [k, v]: each thread owns the 16 fp32
// registers h[k0 : k0+16, o_v] of the head's [D, D] = [128, 128] state
// (GDN owns h[v, k0:k0+16]). State stays fp32 register-resident across
// the whole serial chunk loop; the pre-update per-chunk copy persisted
// for the o-kernel is fp16 rtne.
//
// Per chunk (t_len valid tokens):
//   1. store h tile (fp32 -> fp16 rtne) into h [NT_total, H, D, D] at
//      [k, v] offsets h[(chunk*H + i_h)*D*D + k*D + v];
//   2. stage s_glast[128] = g at the chunk's last valid token (per dim);
//   3. stage s_kg[64, 128] fp16, s_kg[t, d] = k[t, d] *
//      exp(g_last[d] - g[t, d]) for t < t_len else 0 -- staging the
//      per-dim gated keys once in LDS turns the h-update into a pure
//      V_DOT2 pass (8192 expf / workgroup / chunk, cooperative);
//   4. v-correction: v_raw[t] = u[t, v] - sum_k w[t, k]*fp16(h[k, v]),
//      K-reduction via the 8 k-slice shfl; the UNGATED v_raw is written
//      to v_new [L, H, D] fp16 (the o-kernel consumes ungated v_new);
//      v_corr[t] kept fp32 in registers (64 VGPR, same as GDN);
//   5. decay: hreg[j] *= exp(g_last[k0+j]) (per k-row);
//   6. h-update: hreg[j] += sum_t s_kg[t, k0+j] * fp16(v_corr[t]),
//      paired over t via V_DOT2_F32_F16.
// Correction uses the PRE-decay state (snapshot at step 4 before step 5)
// exactly like the reference ordering.
//
// Parity notes: h is rounded to fp16 (rtne) before the correction dot
// and v_corr before the update dot (chain data model; the pure-fp32
// torch reference is matched within fp16 tolerance -- documented).
// expf is accurate __ocml_exp, never __expf.
//
// Varlen-only chain: cu_seqlens [N+1] int32 and chunk_offsets [N+1]
// int32 (cumulative chunk counts) are REQUIRED; initial_state /
// final_state are optional fp32 [N, H, D, D] tensors ([k, v] ordering),
// mirroring gdn_prefill_delta_h_rdna2.
// No __launch_bounds__ / occupancy pins (project rule; ~90 VGPR/thread
// is accepted like the GDN kernel -- occupancy-first, tune later).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GLM5_DH_D = 128;        // head dim (K == V == D)
constexpr int GLM5_DH_BT = 64;        // chunk size
constexpr int GLM5_DH_BV = 32;        // v-rows per workgroup
constexpr int GLM5_DH_THREADS = 256;  // 32 v-rows x 8 k-slices

// Reduction across the 8 k-slice lanes sharing one v-row.
__device__ __forceinline__ float glm5_kda_ksum(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

__device__ __forceinline__ float glm5_fdot2(__half2 a, __half2 b,
                                            float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, /*clamp=*/false);
}

__device__ __forceinline__ __half2 glm5_load_f16x2(const __half* p) {
  return *reinterpret_cast<const __half2*>(p);
}

__global__ void glm5_kda_prefill_delta_h_rdna2_kernel(
    const __half* __restrict__ k,           // [L, H, D] fp16
    const __half* __restrict__ u,           // [L, H, D] fp16
    const __half* __restrict__ w,           // [L, H, D] fp16
    const float* __restrict__ g,            // [L, H, D] fp32 (cumsum'd)
    const float* __restrict__ h0,           // [N, H, D, D] fp32, nullable
    __half* __restrict__ h,                 // [NT_total, H, D, D] fp16 out
    float* __restrict__ ht,                 // [N, H, D, D] fp32, nullable
    __half* __restrict__ v_new,             // [L, H, D] fp16 out
    int H, int N, int store_final_state,
    const int* __restrict__ cu_seqlens,     // [N+1]
    const int* __restrict__ chunk_offsets) { // [N+1]
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / H;
  const int i_h = i_nh % H;

  const int lane = threadIdx.x;
  const int lane_v = lane >> 3;               // 0..31
  const int lane_ks = lane & 7;               // 0..7
  const int o_v = i_v * GLM5_DH_BV + lane_v;  // V row owned (always < D)
  const int k0 = lane_ks * 16;                // 16-wide K slice

  // LDS: per-chunk gated-key tile + per-dim last gate.
  __shared__ __half s_kg[GLM5_DH_BT * GLM5_DH_D];  // 16 KB
  __shared__ float s_glast[GLM5_DH_D];             // 0.5 KB

  // Sequence bounds (varlen-only).
  const int bos = cu_seqlens[i_n];
  const int T_seq = cu_seqlens[i_n + 1] - bos;
  const int NT_seq = (T_seq + GLM5_DH_BT - 1) / GLM5_DH_BT;
  if (NT_seq <= 0) return;

  // Flat chunk index of this sequence's first chunk.
  const long h_base_t = (long)chunk_offsets[i_n];
  const long DD = (long)GLM5_DH_D * GLM5_DH_D;

  // Initial state tile h[k0:k0+16, o_v] -> 16 fp32 registers ([k, v]).
  float hreg[16];
  if (h0 != nullptr) {
    const float* p_h0 = h0 + (long)i_nh * DD + (long)k0 * GLM5_DH_D + o_v;
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = p_h0[(long)j * GLM5_DH_D];
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = 0.0f;
  }

  // Streaming base pointers for this (sequence, head). p_k spans the
  // full D columns (the kg staging loop covers all dims; the k0 slice
  // is applied inside the staging index math).
  const __half* p_k = k + ((long)bos * H + i_h) * GLM5_DH_D;
  const __half* p_w = w + ((long)bos * H + i_h) * GLM5_DH_D + k0;
  const __half* p_u = u + ((long)bos * H + i_h) * GLM5_DH_D + o_v;
  const float* p_g = g + ((long)bos * H + i_h) * GLM5_DH_D;
  __half* p_vnew = v_new + ((long)bos * H + i_h) * GLM5_DH_D + o_v;
  __half* p_h = h + ((h_base_t * H + i_h) * DD + (long)k0 * GLM5_DH_D) + o_v;
  const long stride_h = (long)H * DD;
  const long stride_tok = (long)H * GLM5_DH_D;

  // ---- Serial recurrence over this sequence's chunks ------------------
  for (int i_t = 0; i_t < NT_seq; ++i_t) {
    const int chunk_start = i_t * GLM5_DH_BT;
    const int t_len = (chunk_start + GLM5_DH_BT <= T_seq)
                          ? GLM5_DH_BT
                          : (T_seq - chunk_start);

    // 1. Store pre-update h tile (fp16 rtne) for the o-kernel.
#pragma unroll
    for (int j = 0; j < 16; ++j) p_h[(long)j * GLM5_DH_D] = __float2half_rn(hreg[j]);

    // 2. Per-dim gate at the chunk's last valid token.
    if (lane < GLM5_DH_D) {
      s_glast[lane] = p_g[(long)(chunk_start + t_len - 1) * GLM5_DH_D + lane];
    }
    __syncthreads();

    // 3. Stage the gated-key tile: kg[t, d] = k[t, d]*exp(g_last[d]-g[t,d]).
    for (int it = 0; it < GLM5_DH_BT * GLM5_DH_D / GLM5_DH_THREADS; ++it) {
      const int idx = lane + it * GLM5_DH_THREADS;
      const int t = idx / GLM5_DH_D;
      const int d = idx - t * GLM5_DH_D;
      __half val = __float2half(0.0f);
      if (t < t_len) {
        const float kval = __half2float(p_k[(long)(chunk_start + t) * stride_tok + d]);
        const float g_t = p_g[(long)(chunk_start + t) * GLM5_DH_D + d];
        val = __float2half(kval * expf(s_glast[d] - g_t));
      }
      s_kg[idx] = val;
    }
    __syncthreads();

    // 4. v-correction with the PRE-decay state (fp16-rounded snapshot).
    __half h_fp16[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h_fp16[j] = __float2half_rn(hreg[j]);

    float v_corr[GLM5_DH_BT];
#pragma unroll 1
    for (int t = 0; t < t_len; ++t) {
      float acc = 0.0f;
      const __half* p_wt = p_w + (long)t * stride_tok;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const __half2 a = glm5_load_f16x2(p_wt + 2 * j);
        const __half2 b =
            __halves2half2(h_fp16[2 * j], h_fp16[2 * j + 1]);
        acc = glm5_fdot2(a, b, acc);
      }
      acc = glm5_kda_ksum(acc);
      const float u_val = __half2float(p_u[(long)t * stride_tok]);
      const float v_raw = u_val - acc;
      v_corr[t] = v_raw;
      p_vnew[(long)t * stride_tok] = __float2half_rn(v_raw);
    }
    for (int t = t_len; t < GLM5_DH_BT; ++t) v_corr[t] = 0.0f;

    // 5. Decay: h[k, :] *= exp(g_last[k]) per k-row.
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] *= expf(s_glast[k0 + j]);

    // 6. h-update: hreg[j] += sum_t kg[t, k0+j] * v_corr[t], paired
    //    over t so V_DOT2 pairs the reduction axis (t, t+1).
#pragma unroll
    for (int tp = 0; tp < GLM5_DH_BT / 2; ++tp) {
      const int t0 = 2 * tp;
      const __half2 vb = __halves2half2(__float2half_rn(v_corr[t0]),
                                        __float2half_rn(v_corr[t0 + 1]));
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const __half2 a = __halves2half2(s_kg[t0 * GLM5_DH_D + k0 + j],
                                         s_kg[(t0 + 1) * GLM5_DH_D + k0 + j]);
        hreg[j] = glm5_fdot2(a, vb, hreg[j]);
      }
    }

    __syncthreads();  // before the next chunk re-stages s_glast / s_kg
    p_h += stride_h;
  }

  // Epilogue: persist post-final state (fp32, [k, v] ordering).
  if (store_final_state) {
    float* p_ht = ht + (long)i_nh * DD + (long)k0 * GLM5_DH_D + o_v;
#pragma unroll
    for (int j = 0; j < 16; ++j) p_ht[(long)j * GLM5_DH_D] = hreg[j];
  }
}

}  // namespace

// ============================================================================
// Host wrapper -- torch.ops._rocm_C.glm5_kda_prefill_delta_h_rdna2.
// Optional initial_state/final_state mirror gdn_prefill_delta_h_rdna2;
// cu_seqlens/chunk_offsets are REQUIRED (varlen-only chain) but typed
// optional for binding parity with the GDN op.
// ============================================================================

void glm5_kda_prefill_delta_h_rdna2(
    torch::Tensor k, torch::Tensor u, torch::Tensor w, torch::Tensor g,
    torch::Tensor h, torch::Tensor v_new,
    c10::optional<torch::Tensor> initial_state,
    c10::optional<torch::Tensor> final_state,
    c10::optional<torch::Tensor> cu_seqlens,
    c10::optional<torch::Tensor> chunk_offsets, int64_t chunk_size) {
  TORCH_CHECK(chunk_size == GLM5_DH_BT,
              "glm5_kda_prefill_delta_h_rdna2 requires chunk_size == 64");
  TORCH_CHECK(k.scalar_type() == at::kHalf && u.scalar_type() == at::kHalf &&
                  w.scalar_type() == at::kHalf &&
                  v_new.scalar_type() == at::kHalf &&
                  h.scalar_type() == at::kHalf,
              "glm5_kda_prefill_delta_h_rdna2 is fp16-only (gfx1030) for "
              "k/u/w/h/v_new");
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be fp32 [L, H, D]");
  TORCH_CHECK(k.dim() == 3 && k.stride(-1) == 1,
              "k must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(u.dim() == 3 && u.stride(-1) == 1,
              "u must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(w.dim() == 3 && w.stride(-1) == 1,
              "w must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1,
              "g must be fp32 [L, H, D], contiguous in last dim");
  TORCH_CHECK(v_new.dim() == 3 && v_new.stride(-1) == 1,
              "v_new must be fp16 [L, H, D], contiguous in last dim");
  TORCH_CHECK(h.dim() == 4 && h.stride(-1) == 1,
              "h must be fp16 [NT_total, H, D, D], contiguous in last dim");

  const long L = k.size(0);
  const long H = k.size(1);
  const long K = k.size(2);
  TORCH_CHECK(K == GLM5_DH_D,
              "glm5_kda_prefill_delta_h_rdna2 requires D == 128");
  TORCH_CHECK(u.size(0) == L && u.size(1) == H && u.size(2) == K &&
                  w.size(0) == L && w.size(1) == H && w.size(2) == K &&
                  g.size(0) == L && g.size(1) == H && g.size(2) == K &&
                  v_new.size(0) == L && v_new.size(1) == H &&
                  v_new.size(2) == K,
              "u/w/g/v_new shape mismatch with k");
  TORCH_CHECK(h.size(1) == H && h.size(2) == GLM5_DH_D &&
                  h.size(3) == GLM5_DH_D,
              "h shape mismatch (expected [NT_total, H, 128, 128])");

  // Varlen-only: cu_seqlens + chunk_offsets required.
  TORCH_CHECK(cu_seqlens.has_value() && cu_seqlens->defined(),
              "varlen-only chain: cu_seqlens must be provided");
  TORCH_CHECK(chunk_offsets.has_value() && chunk_offsets->defined(),
              "varlen-only chain: chunk_offsets must be provided");
  TORCH_CHECK(cu_seqlens->dim() == 1 &&
                  cu_seqlens->scalar_type() == at::kInt,
              "cu_seqlens must be int32 [N+1]");
  TORCH_CHECK(chunk_offsets->dim() == 1 &&
                  chunk_offsets->scalar_type() == at::kInt &&
                  chunk_offsets->size(0) == cu_seqlens->size(0),
              "chunk_offsets must be int32 [N+1] of equal length");
  const int N = (int)cu_seqlens->size(0) - 1;
  TORCH_CHECK(N >= 1, "need at least one sequence");

  const bool has_h0 = initial_state.has_value() && initial_state->defined();
  if (has_h0) {
    TORCH_CHECK(initial_state->scalar_type() == at::kFloat &&
                    initial_state->dim() == 4 &&
                    initial_state->stride(-1) == 1 &&
                    initial_state->size(0) == N &&
                    initial_state->size(1) == H &&
                    initial_state->size(2) == GLM5_DH_D &&
                    initial_state->size(3) == GLM5_DH_D,
                "initial_state must be fp32 [N, H, 128, 128] ([k, v] "
                "ordering), contiguous in last dim");
  }
  const bool store_final = final_state.has_value() && final_state->defined();
  if (store_final) {
    TORCH_CHECK(final_state->scalar_type() == at::kFloat &&
                    final_state->dim() == 4 &&
                    final_state->stride(-1) == 1 &&
                    final_state->size(0) == N &&
                    final_state->size(1) == H &&
                    final_state->size(2) == GLM5_DH_D &&
                    final_state->size(3) == GLM5_DH_D,
                "final_state must be fp32 [N, H, 128, 128] ([k, v] "
                "ordering), contiguous in last dim");
  }

  if (L == 0) return;
  // NT_total = h.size(0): reading chunk_offsets[N] on the host would
  // dereference a device pointer (same reasoning as the GDN wrapper).

  const at::cuda::OptionalCUDAGuard guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((GLM5_DH_D + GLM5_DH_BV - 1) / GLM5_DH_BV, (unsigned int)(N * H));
  glm5_kda_prefill_delta_h_rdna2_kernel<<<grid, GLM5_DH_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(u.data_ptr()),
      reinterpret_cast<const __half*>(w.data_ptr()),
      g.data_ptr<float>(),
      has_h0 ? initial_state->data_ptr<float>() : nullptr,
      reinterpret_cast<__half*>(h.data_ptr()),
      store_final ? final_state->data_ptr<float>() : nullptr,
      reinterpret_cast<__half*>(v_new.data_ptr()), (int)H, N,
      store_final ? 1 : 0, cu_seqlens->data_ptr<int>(),
      chunk_offsets->data_ptr<int>());
}
