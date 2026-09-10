// GatedDeltaNet (GDN) chunked-prefill inter-chunk recurrence kernel for
// AMD RDNA2 (gfx1030). Hand port of
// chunk_gated_delta_rule_fwd_kernel_h_blockdim64
// (vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py).
//
// Workgroup = one (V-tile, sequence x value-head). 256 threads = 32 v-rows
// x 8 k-slices, the exact lane layout of gdn_decode_rdna2.cu
// (v = lane >> 3, ks = lane & 7); each thread owns 16 fp32 state
// registers h[v, ks*16 : ks*16+16], register-resident across the entire
// serial chunk loop (no LDS for h, no per-chunk sync). Per chunk:
//   1. store start-of-chunk h (fp32 -> fp16 rtne) for chunk_fwd_o;
//   2. v-correction  v_raw[t] = u[t] - sum_k w[t,k] * fp16(h[bv,k]),
//      reduction over K=128 via the 8-k-slice gdn_ksum shfl. The
//      UNGATED v_raw (fp16 rtne) is written to v_new here, BEFORE
//      gating -- chunk_fwd_o consumes the ungated v_new;
//   3. USE_G scaling: v_corr[t] *= exp(g_last - g[t]); h *= exp(g_last);
//   4. h-update  h[bv,k] += sum_t fp16(k[t,k]) * fp16(v_corr_gated[t]),
//      reduction over BT=64 as a serial t-loop, each thread owning
//      distinct h elements, paired-over-t V_DOT2_F32_F16 inner dots.
//
// Parity-critical fp16 roundings (rtne) of the Triton reference are
// preserved: fp32 h state is rounded to fp16 before the v-correction
// dot, and the gated fp32 v_corr is rounded to fp16 before the h-update
// dot. expf (accurate __ocml_exp) is used throughout, never __expf.
// num_stages=4 pipelining is known-broken for this kernel on the Triton
// AMD backend (chunk_delta_h.py:21); the HIP loop is deliberately
// straightforward (no software pipelining, no double-buffering).
//
// USE_GK (per-key gating, gk) is dead code on the Qwen3.5/3.6 call
// path -- chunk.py:61-71 never passes it -- and is therefore not a
// parameter of the wrapper: it is structurally impossible to silently
// diverge on it. h0 (initial_state) IS supported (default None == zeros).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GDN_K = 128;        // head_k_dim; fixed by the 8x16 slice layout
constexpr int GDN_V = 128;        // head_v_dim
constexpr int GDN_BT = 64;        // FLA_CHUNK_SIZE
constexpr int GDN_BV = 32;        // V rows per workgroup (V/GDN_BV = 4 tiles)
constexpr int GDN_THREADS = 256;  // 32 v-rows * 8 k-slices

// Reduction across the 8 k-slice lanes sharing one v-row. Lane layout is
// v = lane >> 3, ks = lane & 7, so masks 1/2/4 stay inside the group.
__device__ __forceinline__ float gdn_ksum(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

// V_DOT2_F32_F16: acc += a.x*b.x + a.y*b.y (fp32 accumulate, no clamp).
__device__ __forceinline__ float gdn_fdot2(__half2 a, __half2 b, float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, /*clamp=*/false);
}

__device__ __forceinline__ __half2 gdn_load_f16x2(const __half* p) {
  return *reinterpret_cast<const __half2*>(p);
}

__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4)))
    gdn_prefill_delta_h_packed_kernel(
        const __half* __restrict__ k,            // [B*T, Hg, K]
        const __half* __restrict__ u,            // [B*T, H, V]
        const __half* __restrict__ w,            // [B*T, H, K]
        const float* __restrict__ g,             // [B*T, H] g_cumsum, fp32
        const float* __restrict__ h0,            // [N, H, V, K] fp32, nullable
        __half* __restrict__ h,                  // [B, NT, H, V, K] fp16 out (5D always)
        float* __restrict__ ht,                  // [N, H, V, K] fp32, nullable
        __half* __restrict__ v_new,              // [B*T, H, V] fp16 out
        const int* __restrict__ cu_seqlens,      // [N+1] varlen, or nullptr
        const int* __restrict__ chunk_offsets,   // [N+1] varlen, or nullptr
        int T, int H, int Hg, int N, int B, int NT, int store_final_state) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / H;
  const int i_h = i_nh % H;                  // value-head (H == HV)
  const int i_hk = i_h / (H / Hg);           // key-head (GQA grouping)

  const int lane = threadIdx.x;
  const int lane_v = lane >> 3;              // [0, 32)
  const int lane_ks = lane & 7;              // [0, 8)
  const int o_v = i_v * GDN_BV + lane_v;     // V row this thread owns
  const bool v_ok = o_v < GDN_V;
  const int k0 = lane_ks * 16;               // this thread's 16-wide K slice

  // Sequence bounds. cu_seqlens present => varlen (B == 1, flattened);
  // otherwise uniform-length with per-seq length T.
  int bos, T_seq;
  if (cu_seqlens != nullptr) {
    bos = cu_seqlens[i_n];
    const int eos = cu_seqlens[i_n + 1];
    T_seq = eos - bos;
  } else {
    bos = i_n * T;
    T_seq = T;
  }
  const int NT_seq = (T_seq + GDN_BT - 1) / GDN_BT;
  if (NT_seq <= 0) return;

  // Per-batch base offset into 5D h. The 5D layout is [B, NT, H, V, K].
  // For non-varlen: h[i_b, i_t, i_h, ...] -> flat = (i_b*NT + i_t)*H*V*K + ...
  // For varlen: h[0, flat_i_t, i_h, ...] -> flat = flat_i_t*H*V*K + ...
  long h_base_t;  // flat chunk index added to i_t inside the loop
  if (cu_seqlens != nullptr) {
    // Cumulative chunks before sequence i_n (boh equivalent for 5D)
    h_base_t = (long)chunk_offsets[i_n];
  } else {
    h_base_t = (long)i_n * NT_seq;
  }

  // h tile h[o_v, k0 : k0+16] -> registers (16 fp32). Either from h0 or
  // zeros (the wrapper normalizes h0=None to == zeros up front if it
  // wants, but we also handle it here for convenience).
  float hreg[16];
  if (h0 != nullptr) {
    const float* p_h0 =
        h0 + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = v_ok ? p_h0[j] : 0.0f;
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = 0.0f;
  }

  // Streaming base pointers for this (sequence, head). Triton strides:
  // k/w advance by (head*K), u/v_new by (head*V), k uses Hg not H (GQA).
  const __half* p_k =
      k + (long)bos * Hg * GDN_K + (long)i_hk * GDN_K + k0;
  const __half* p_w =
      w + (long)bos * H * GDN_K + (long)i_h * GDN_K + k0;
  const __half* p_u =
      u + (long)bos * H * GDN_V + (long)i_h * GDN_V + o_v;
  const float* p_g = g + (long)bos * H + i_h;
  __half* p_vnew =
      v_new + (long)bos * H * GDN_V + (long)i_h * GDN_V + o_v;
  __half* p_h =
      h + ((h_base_t * H + i_h) * GDN_V * GDN_K + (long)o_v * GDN_K) + k0;
  const long stride_h = (long)H * GDN_V * GDN_K;

  // ---- Serial recurrence over this sequence's chunks -----------------
  for (int i_t = 0; i_t < NT_seq; ++i_t) {
    const int chunk_start = i_t * GDN_BT;
    const int t_len =
        (chunk_start + GDN_BT <= T_seq) ? GDN_BT : (T_seq - chunk_start);
    // The loop body's k/w/u/v_new indexing is chunk-local (t in
    // [0, t_len)); rebase the row pointers by chunk_start so chunk i_t
    // reads/writes its own rows, not chunk 0's.
    const __half* p_w_c = p_w + (long)chunk_start * H * GDN_K;
    const __half* p_u_c = p_u + (long)chunk_start * H * GDN_V;
    __half* p_vnew_c = p_vnew + (long)chunk_start * H * GDN_V;
    const __half* p_k_c = p_k + (long)chunk_start * Hg * GDN_K;

    // 1. Store the pre-update h tile (fp16, rtne) for chunk_fwd_o. The
    //    register state stays fp32; only the persisted copy is rounded.
    if (v_ok) {
#pragma unroll
      for (int j = 0; j < 16; ++j) p_h[j] = __float2half_rn(hreg[j]);
    }

    // 2. v-correction: v_raw[t] = u[t] - sum_k w[t,k] * fp16(h).
    //    h is rounded to fp16 (rtne) once per chunk before the dot;
    //    reduction over K=128 is the 8 k-slice shfl.
    //    The UNGATED v_raw is stored to v_new (fp16 rtne) here, BEFORE
    //    gating -- chunk_fwd_o consumes the ungated v_new.
    __half h_fp16[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h_fp16[j] = __float2half_rn(hreg[j]);

    float v_corr[GDN_BT];
#pragma unroll
    for (int t = 0; t < t_len; ++t) {
      float acc = 0.0f;
      const __half* p_wt = p_w_c + (long)t * H * GDN_K;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const __half2 a = gdn_load_f16x2(p_wt + 2 * j);
        const __half2 b =
            __halves2half2(h_fp16[2 * j], h_fp16[2 * j + 1]);
        acc = gdn_fdot2(a, b, acc);
      }
      acc = gdn_ksum(acc);
      const float u_val =
          v_ok ? __half2float(p_u_c[(long)t * H * GDN_V]) : 0.0f;
      const float v_raw = u_val - acc;
      v_corr[t] = v_raw;
      // Persist UNGATED v_new (fp16, rtne). Invalid v-rows skip the
      // store but still need the dot + shfl for warp uniformity.
      if (v_ok) p_vnew_c[(long)t * H * GDN_V] = __float2half_rn(v_raw);
    }
    for (int t = t_len; t < GDN_BT; ++t) v_corr[t] = 0.0f;

    // 3. USE_G gating. g_last is the cumulative gate at the last valid
    //    token of this chunk; v_corr is scaled by exp(g_last - g[t])
    //    (zero for tail tokens since v_corr[t] == 0 there), and h is
    //    scaled by exp(g_last).
    const int last_idx = chunk_start + t_len - 1;
    const float g_last = p_g[(long)last_idx * H];
    for (int t = 0; t < t_len; ++t) {
      const float g_t = p_g[(long)(chunk_start + t) * H];
      v_corr[t] *= expf(g_last - g_t);
    }
    const float decay = expf(g_last);
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] *= decay;

    // 4. h-update: h[o_v, k0+j] += sum_t k[t, k0+j] * fp16(v_corr[t]).
    //    v_corr (fp32, gated) is rounded to fp16 (rtne) before the dot.
    //    Pairing over t matches Triton's V_DOT2 lowering of
    //    tl.dot(b_k, b_v) with fp16 operands; each thread owns its
    //    16 h elements, no shfl.
#pragma unroll
    for (int tp = 0; tp < GDN_BT / 2; ++tp) {
      const int t0 = 2 * tp;
      const __half2 vb = __halves2half2(__float2half_rn(v_corr[t0]),
                                        __float2half_rn(v_corr[t0 + 1]));
      __half2 a0[8], a1[8];
      const bool valid0 = (t0 < t_len);
      const bool valid1 = (t0 + 1 < t_len);
      if (valid0) {
        const __half* p_k0 = p_k_c + (long)t0 * Hg * GDN_K;
#pragma unroll
        for (int j = 0; j < 8; ++j) a0[j] = gdn_load_f16x2(p_k0 + 2 * j);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          a0[j] = __halves2half2(__float2half_rn(0.0f),
                                 __float2half_rn(0.0f));
      }
      if (valid1) {
        const __half* p_k1 = p_k_c + (long)(t0 + 1) * Hg * GDN_K;
#pragma unroll
        for (int j = 0; j < 8; ++j) a1[j] = gdn_load_f16x2(p_k1 + 2 * j);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          a1[j] = __halves2half2(__float2half_rn(0.0f),
                                 __float2half_rn(0.0f));
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        hreg[2 * j] =
            gdn_fdot2(__halves2half2(__low2half(a0[j]), __low2half(a1[j])),
                      vb, hreg[2 * j]);
        hreg[2 * j + 1] = gdn_fdot2(
            __halves2half2(__high2half(a0[j]), __high2half(a1[j])), vb,
            hreg[2 * j + 1]);
      }
    }

    p_h += stride_h;
  }

  // Epilogue: persist the post-final h register state to the float
  // final_state slot (fp32, no rounding).
  if (store_final_state && v_ok) {
    float* p_ht = ht + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) p_ht[j] = hreg[j];
  }
}

}  // namespace

void gdn_prefill_delta_h_rdna2(torch::Tensor k, torch::Tensor u,
                               torch::Tensor w, torch::Tensor g,
                               torch::Tensor h, torch::Tensor v_new,
                               c10::optional<torch::Tensor> initial_state,
                               c10::optional<torch::Tensor> final_state,
                               c10::optional<torch::Tensor> cu_seqlens,
                               c10::optional<torch::Tensor> chunk_offsets,
                               int64_t chunk_size) {
  TORCH_CHECK(chunk_size == GDN_BT,
              "gdn_prefill_delta_h_rdna2 requires chunk_size == 64");
  TORCH_CHECK(k.scalar_type() == at::kHalf && u.scalar_type() == at::kHalf &&
                  w.scalar_type() == at::kHalf,
              "gdn_prefill_delta_h_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(g.scalar_type() == at::kFloat,
              "g (g_cumsum) must be fp32 [B, T, H]");
  TORCH_CHECK(h.scalar_type() == at::kHalf,
              "h (per-chunk states) must be fp16 [B, NT, H, V, K] (5D)");
  TORCH_CHECK(v_new.scalar_type() == at::kHalf,
              "v_new must be fp16 [B, T, H, V]");
  TORCH_CHECK(k.dim() == 4 && k.stride(-1) == 1,
              "k must be [B, T, Hg, K] contiguous in last dim");
  TORCH_CHECK(u.dim() == 4 && u.stride(-1) == 1,
              "u must be [B, T, H, V] contiguous in last dim");
  TORCH_CHECK(w.dim() == 4 && w.stride(-1) == 1,
              "w must be [B, T, H, K] contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1,
              "g must be [B, T, H] contiguous in last dim");

  const int B = k.size(0);
  const int T = k.size(1);
  const int Hg = k.size(2);
  const int K = k.size(3);
  const int H = u.size(2);
  const int V = u.size(3);
  TORCH_CHECK(K == GDN_K,
              "gdn_prefill_delta_h_rdna2 requires head_k_dim == 128");
  TORCH_CHECK(V == GDN_V,
              "gdn_prefill_delta_h_rdna2 requires head_v_dim == 128");
  TORCH_CHECK(H % Hg == 0,
              "H must be divisible by Hg (GQA); got H=", H, " Hg=", Hg);
  TORCH_CHECK(k.size(1) == u.size(1) && u.size(1) == w.size(1) &&
                  w.size(1) == g.size(1),
              "k/u/w/g sequence length mismatch");

  const bool varlen = cu_seqlens.has_value() && cu_seqlens->defined();
  int N;
  if (varlen) {
    TORCH_CHECK(B == 1, "varlen requires flattened batch B == 1");
    TORCH_CHECK(cu_seqlens->scalar_type() == at::kInt,
                "cu_seqlens must be int32 [N+1]");
    TORCH_CHECK(chunk_offsets.has_value() && chunk_offsets->defined(),
                "varlen requires chunk_offsets");
    TORCH_CHECK(chunk_offsets->scalar_type() == at::kInt,
                "chunk_offsets must be int32 [N+1]");
    N = cu_seqlens->size(0) - 1;
  } else {
    N = B;
  }
  TORCH_CHECK(N >= 1, "need at least one sequence");

  const bool has_h0 = initial_state.has_value() && initial_state->defined();
  if (has_h0) {
    TORCH_CHECK(initial_state->scalar_type() == at::kFloat &&
                    initial_state->dim() == 4 &&
                    initial_state->stride(-1) == 1,
                "initial_state must be fp32 [N, H, V, K] contiguous in last "
                "dim");
    TORCH_CHECK(initial_state->size(0) == N &&
                    initial_state->size(1) == H &&
                    initial_state->size(2) == V &&
                    initial_state->size(3) == K,
                "initial_state shape mismatch");
  }

  const bool store_final = final_state.has_value() && final_state->defined();
  if (store_final) {
    TORCH_CHECK(final_state->scalar_type() == at::kFloat &&
                    final_state->dim() == 4 &&
                    final_state->stride(-1) == 1,
                "final_state must be fp32 [N, H, V, K] contiguous in last "
                "dim");
    TORCH_CHECK(final_state->size(0) == N && final_state->size(1) == H &&
                    final_state->size(2) == V && final_state->size(3) == K,
                "final_state shape mismatch");
  }

  if (T == 0 || N == 0) return;

  // NT_total = total number of chunks across all sequences in this batch.
  // Non-varlen: NT_total = B * cdiv(T, BT)
  // Varlen: NT_total = chunk_offsets[N] (cumulative total at end of array)
  int NT_total;
  if (varlen) {
    // h.size(1) is the NT dim of the 5D h tensor. Reading chunk_offsets[N]
    // here would dereference a device pointer on the host and yield garbage.
    NT_total = h.size(1);
  } else {
    NT_total = B * ((T + GDN_BT - 1) / GDN_BT);
  }

  const at::cuda::OptionalCUDAGuard guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((V + GDN_BV - 1) / GDN_BV, N * H);
  gdn_prefill_delta_h_packed_kernel<<<grid, GDN_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(u.data_ptr()),
      reinterpret_cast<const __half*>(w.data_ptr()),
      g.data_ptr<float>(),
      has_h0 ? initial_state->data_ptr<float>() : nullptr,
      reinterpret_cast<__half*>(h.data_ptr()),
      store_final ? final_state->data_ptr<float>() : nullptr,
      reinterpret_cast<__half*>(v_new.data_ptr()),
      varlen ? cu_seqlens->data_ptr<int>() : nullptr,
      varlen ? chunk_offsets->data_ptr<int>() : nullptr,
      T, H, Hg, N, B, NT_total, store_final ? 1 : 0);
}
