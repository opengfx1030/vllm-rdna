// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet (GDN) fused post-conv1d prep + chunk-local cumsum kernel
// for AMD RDNA2 (gfx1030). Hand port of
//   fused_post_conv_prep (vllm/third_party/flash_linear_attention/ops/
//                         fused_gdn_prefill_post_conv.py)
// fused with
//   chunk_local_cumsum_scalar (vllm/third_party/flash_linear_attention/ops/
//                               cumsum.py, chunk_size=64, reverse=False).
//
// The V-head path of fused_post_conv_prep writes the per-(token, V-head)
// gating values g [L, HV] (fp32) and beta [L, HV] (fp32). We fold the
// per-chunk inclusive cumsum of g into the same kernel so the orchestrator
// only launches one grid for the post-conv prep + cumsum pair.
//
// Per (chunk, V-head) the V-head workgroup:
//   1. copies v [BT, V] from mixed_qkv to the v output (passthrough, fp16);
//   2. computes g [BT] (stable softplus, threshold 20) and beta [BT]
//      (fp32 sigmoid, NO fp16 round-trip) from a, b, A_log, dt_bias;
//   3. does an inclusive cumsum of g over BT in-warp (shfl_up), the result
//      is the per-chunk g_cumsum [BT] (matches cumsum.py:66-71 exactly:
//      the cumsum resets at every chunk boundary, so no inter-chunk carry);
//   4. writes g_cumsum and beta to the [L, HV] outputs.
//
// The Q/K-head workgroups just do load + L2-normalise + store, identical to
// the Triton reference (fused_gdn_prefill_post_conv.py:69-107).
//
// Workgroup = one (token-block, head). BlockT=64 (FLA_CHUNK_SIZE) tokens
// per workgroup, 128 threads = 4 warps (wave32).
//
// Thread decomposition:
//   * For Q/K load + L2norm + store and for v passthrough, each token is
//     owned by a pair of lanes (half = lane & 1), each handling 64
//     contiguous K (or V) elements. The L2 reduction pairs the two lanes
//     via __shfl_xor with delta=1 (pair stays within one warp on RDNA2).
//   * For the V-head gating/cumsum, lanes 0..63 each own one token of the
//     chunk (scan_tok = lane). The 64-wide inclusive scan is a
//     2-warp scan: warp-local shfl_up + cross-warp carry via 1-element LDS.
//
// fp16 inputs, fp32 intermediates, fp32 g/beta outputs. expf/log1pf only.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int PREP_BT = 64;            // FLA_CHUNK_SIZE
constexpr int PREP_THREADS = 128;      // 4 warps (wave32) x 32 lanes
constexpr int PREP_LANES_PER_TOK = 2;  // 2 lanes share K/V=128 (64 elems each)
constexpr int PREP_ELEMS_PER_LANE = 64;
constexpr float PREP_SOFTPLUS_THRESHOLD = 20.0f;
constexpr float PREP_L2NORM_EPS = 1e-6f;

__device__ __forceinline__ float gdn_to_f32(float x) { return x; }
__device__ __forceinline__ float gdn_to_f32(__half x) {
  return __half2float(x);
}

// Inclusive scan of a value across the lanes of a single warp.
__device__ __forceinline__ float warp_inclusive_scan(float x) {
  // shfl_up with delta = k adds the value from lane (lane - k) to lane.
  // delta = 1, 2, 4, 8, 16 covers a 32-wide inclusive scan.
  // Guard the addition: lane < delta means the source is out of range (self-add
  // without the guard produces 32x inflation for lane 0, 16x for lane 1, etc.)
  const int lane = threadIdx.x & 31;
  float n;
  n = __shfl_up_sync(0xffffffffffffffffULL, x, 1);  if (lane >= 1)  x += n;
  n = __shfl_up_sync(0xffffffffffffffffULL, x, 2);  if (lane >= 2)  x += n;
  n = __shfl_up_sync(0xffffffffffffffffULL, x, 4);  if (lane >= 4)  x += n;
  n = __shfl_up_sync(0xffffffffffffffffULL, x, 8);  if (lane >= 8)  x += n;
  n = __shfl_up_sync(0xffffffffffffffffULL, x, 16); if (lane >= 16) x += n;
  return x;
}

template <bool IS_VARLEN, typename AT, typename DT>
__global__ void __launch_bounds__(PREP_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_prefill_prep_rdna2_kernel(
        const __half* __restrict__ mixed_qkv,   // [L, qkv_dim]
        const __half* __restrict__ a,           // [L, HV]
        const __half* __restrict__ b,           // [L, HV]
        const AT* __restrict__ A_log,           // [HV]
        const DT* __restrict__ dt_bias,         // [HV]
        __half* __restrict__ q,                 // [L, H, K]
        __half* __restrict__ k_out,             // [L, H, K]
        __half* __restrict__ v,                 // [L, HV, V]
        float* __restrict__ g_cumsum,           // [L, HV]
        float* __restrict__ beta,               // [L, HV]
        long stride_x_tok,                      // = qkv_dim
        long stride_a_tok,                      // = HV
        long stride_b_tok,                      // = HV
        long stride_q_tok,                      // = H * K
        long stride_k_tok,                      // = H * K
        long stride_v_tok,                      // = HV * V
        long L, long H, long HV,
        const int* __restrict__ cu_seqlens,     // [N+1] or null
        const int* __restrict__ chunk_indices,  // [NT, 2] or null
        long NT                                 // = chunk_indices.size(0)
                                                  //   or ceil(L, PREP_BT)
) {
  // 1-element LDS carry for the cross-warp portion of the 64-wide cumsum.
  // Used in the V-head path only; harmless in the Q/K path.
  __shared__ float s_scan_carry;

  const int i_tb = blockIdx.x;    // chunk index within the flattened batch
  const int i_head = blockIdx.y;  // head index: [0, H) = Q/K path,
                                  //               [H, H+HV) = V+gating path

  const int tid = threadIdx.x;
  // ---- Two distinct token decompositions ----
  // 1) For Q/K and v (token-pair work over K or V):
  //    kv_token = tid / 2 (0..63), kv_half = tid & 1 (0 or 1).
  //    Each pair of lanes handles one token's 128 K or V elements,
  //    split into halves of 64 elements each.
  const int kv_token = tid / PREP_LANES_PER_TOK;   // 0..63
  const int kv_half = tid & 1;                     // 0 or 1
  const int kv_k_start = kv_half * PREP_ELEMS_PER_LANE;

  // 2) For g/beta/cumsum (one lane per token, lanes 0..63 only):
  //    scan_tok = tid (0..63), valid only when tid < 64.
  const int scan_tok = tid;  // 0..63; meaningful only when is_scan_lane
  const bool is_scan_lane = (tid < 64);

  // ---- Resolve the (eos, t_global_base) for this chunk. ----
  long eos;                       // token offset of the sequence end (exclusive)
  long t_global_base;             // absolute token index of the first token in chunk
  if (IS_VARLEN) {
    const int i_n = chunk_indices[i_tb * 2];
    const int i_t = chunk_indices[i_tb * 2 + 1];
    const long bos = (long)cu_seqlens[i_n];
    eos = (long)cu_seqlens[i_n + 1];
    t_global_base = bos + (long)i_t * PREP_BT;
  } else {
    eos = L;
    t_global_base = (long)i_tb * PREP_BT;
  }
  const long t_remaining = eos - t_global_base;
  const int t_chunk =
      (t_remaining < PREP_BT) ? (int)t_remaining : PREP_BT;

  // ---- Per-token validity for the KV/token-pair decomposition ----
  const bool kv_tok_valid = (kv_token < t_chunk);
  const long kv_t_abs = t_global_base + (long)kv_token;

  if (i_head < H) {
    // ============ Q/K path ============
    const int i_h = i_head;
    const long HK = H * 128;
    const long qkv_row = kv_t_abs * stride_x_tok;
    const long q_off_row = qkv_row + (long)i_h * 128 + kv_k_start;
    const long k_off_row = qkv_row + HK + (long)i_h * 128 + kv_k_start;

    // Load this lane's 64 q elements and 64 k elements (masked past t_chunk).
    float q_local[PREP_ELEMS_PER_LANE];
    float k_local[PREP_ELEMS_PER_LANE];
#pragma unroll
    for (int j = 0; j < PREP_ELEMS_PER_LANE; ++j) {
      q_local[j] = 0.0f;
      k_local[j] = 0.0f;
      if (kv_tok_valid) {
        q_local[j] = __half2float(mixed_qkv[q_off_row + j]);
        k_local[j] = __half2float(mixed_qkv[k_off_row + j]);
      }
    }

    // L2 norm reduction over K=128: each token is owned by a pair of lanes.
    // Each lane sums its 64 elements, then a single shfl_xor with delta=1
    // produces the full-K sum on both lanes of the pair.
    float sq = 0.0f;
    float sk = 0.0f;
#pragma unroll
    for (int j = 0; j < PREP_ELEMS_PER_LANE; ++j) {
      sq += q_local[j] * q_local[j];
      sk += k_local[j] * k_local[j];
    }
    sq += __shfl_xor_sync(0xffffffffffffffffULL, sq, 1);
    sk += __shfl_xor_sync(0xffffffffffffffffULL, sk, 1);
    const float q_inv = 1.0f / sqrtf(sq + PREP_L2NORM_EPS);
    const float k_inv = 1.0f / sqrtf(sk + PREP_L2NORM_EPS);
#pragma unroll
    for (int j = 0; j < PREP_ELEMS_PER_LANE; ++j) {
      q_local[j] *= q_inv;
      k_local[j] *= k_inv;
    }

    // Store q and k.
    if (kv_tok_valid) {
      const long q_out_row =
          kv_t_abs * stride_q_tok + (long)i_h * 128 + kv_k_start;
      const long k_out_row =
          kv_t_abs * stride_k_tok + (long)i_h * 128 + kv_k_start;
#pragma unroll
      for (int j = 0; j < PREP_ELEMS_PER_LANE; ++j) {
        q[q_out_row + j] = __float2half(q_local[j]);
        k_out[k_out_row + j] = __float2half(k_local[j]);
      }
    }
  } else {
    // ============ V + gating + cumsum path ============
    const int i_hv = i_head - H;
    const long HK = H * 128;
    const long V_OFFSET = 2 * HK;

    // ---- v passthrough (all 128 threads; token-pair decomposition) ----
    const long v_in_row =
        kv_t_abs * stride_x_tok + V_OFFSET + (long)i_hv * 128 + kv_k_start;
    const long v_out_row =
        kv_t_abs * stride_v_tok + (long)i_hv * 128 + kv_k_start;
    if (kv_tok_valid) {
#pragma unroll
      for (int j = 0; j < PREP_ELEMS_PER_LANE; ++j) {
        v[v_out_row + j] = mixed_qkv[v_in_row + j];
      }
    }

    // ---- g, beta, and per-chunk cumsum of g ----
    // Lanes 0..63 each own one token of the chunk (scan_tok = tid).
    float g_val = 0.0f;   // default 0 for out-of-bound tokens (cumsum-safe)
    float b_val = 0.0f;
    if (is_scan_lane) {
      const bool tok_valid = (scan_tok < t_chunk);
      const long t_abs_scan = t_global_base + (long)scan_tok;
      const long a_off = t_abs_scan * stride_a_tok + i_hv;
      const long b_off = t_abs_scan * stride_b_tok + i_hv;
      const float a_raw = tok_valid ? __half2float(a[a_off]) : 0.0f;
      b_val = tok_valid ? __half2float(b[b_off]) : 0.0f;
      const float A_log_val = gdn_to_f32(A_log[i_hv]);
      const float dt_bias_val = gdn_to_f32(dt_bias[i_hv]);

      // Stable softplus: x > 0 ? x + log(1+exp(-x)) : log(1+exp(x))
      // Threshold 20: above that log(1+exp(-x)) underflows in fp32 so the
      // branch flips to sp = x (the textbook numerical identity that
      // matches the Triton reference's tl.where).
      const float x = a_raw + dt_bias_val;
      float sp;
      if (x <= PREP_SOFTPLUS_THRESHOLD) {
        sp = (x > 0.0f) ? (x + log1pf(expf(-x))) : log1pf(expf(x));
      } else {
        sp = x;
      }
      // g = -exp(A_log) * sp
      g_val = -expf(A_log_val) * sp;
    }

    // Inclusive cumsum over the 64-element chunk held by lanes 0..63.
    // Phase 1: warp-local scan within each of the first two warps.
    float local_scan = 0.0f;
    if (is_scan_lane) {
      local_scan = warp_inclusive_scan(g_val);
    }
    // Phase 2: cross-warp carry via LDS. Lane 31 of warp 0 has the total
    // of warp 0; warp 1 (lanes 32..63) adds it to every element.
    if (tid == 31) {
      s_scan_carry = local_scan;
    }
    __syncthreads();  // ALL 128 threads must reach this barrier
    float cum = 0.0f;
    if (is_scan_lane) {
      cum = (tid < 32) ? local_scan : (local_scan + s_scan_carry);
    }
    __syncthreads();  // before any thread reuses s_scan_carry (defensive)

    // ---- Store g_cumsum and beta ----
    if (is_scan_lane) {
      const bool tok_valid = (scan_tok < t_chunk);
      if (tok_valid) {
        const long t_abs_scan = t_global_base + (long)scan_tok;
        const long gb_off = t_abs_scan * HV + i_hv;
        g_cumsum[gb_off] = cum;
        // fp32 sigmoid (NO fp16 round-trip — decode rounds, prefill must not).
        const float beta_val = 1.0f / (1.0f + expf(-b_val));
        beta[gb_off] = beta_val;
      }
    }
  }
}

}  // namespace

void gdn_prefill_prep_rdna2(
    torch::Tensor mixed_qkv,       // [L, qkv_dim] fp16
    torch::Tensor a,               // [L, HV] fp16
    torch::Tensor b,               // [L, HV] fp16
    torch::Tensor A_log,           // [HV] fp32/fp16
    torch::Tensor dt_bias,         // [HV] fp32/fp16
    torch::Tensor q,               // [L, H, K] fp16 (output)
    torch::Tensor k_out,           // [L, H, K] fp16 (output)
    torch::Tensor v,               // [L, HV, V] fp16 (output)
    torch::Tensor g_cumsum,        // [L, HV] fp32 (output)
    torch::Tensor beta,            // [L, HV] fp32 (output)
    torch::Tensor cu_seqlens,      // [N+1] int32 (empty for non-varlen)
    torch::Tensor chunk_indices) { // [NT, 2] int32 (empty for non-varlen)
  // ---- Shape / dtype validation ----
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.stride(-1) == 1 &&
                  mixed_qkv.scalar_type() == at::kHalf,
              "mixed_qkv must be fp16 [L, qkv_dim], contiguous in last dim");
  TORCH_CHECK(a.dim() == 2 && a.stride(-1) == 1 &&
                  a.scalar_type() == at::kHalf,
              "a must be fp16 [L, HV], contiguous in last dim");
  TORCH_CHECK(b.dim() == 2 && b.stride(-1) == 1 &&
                  b.scalar_type() == at::kHalf,
              "b must be fp16 [L, HV], contiguous in last dim");
  const auto a_ty = A_log.scalar_type();
  const auto d_ty = dt_bias.scalar_type();
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  (a_ty == at::kFloat || a_ty == at::kHalf),
              "A_log must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  (d_ty == at::kFloat || d_ty == at::kHalf),
              "dt_bias must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(q.dim() == 3 && q.stride(-1) == 1 &&
                  q.scalar_type() == at::kHalf,
              "q must be fp16 [L, H, K], contiguous in last dim");
  TORCH_CHECK(k_out.dim() == 3 && k_out.stride(-1) == 1 &&
                  k_out.scalar_type() == at::kHalf,
              "k_out must be fp16 [L, H, K], contiguous in last dim");
  TORCH_CHECK(v.dim() == 3 && v.stride(-1) == 1 &&
                  v.scalar_type() == at::kHalf,
              "v must be fp16 [L, HV, V], contiguous in last dim");
  TORCH_CHECK(g_cumsum.dim() == 2 && g_cumsum.stride(-1) == 1 &&
                  g_cumsum.scalar_type() == at::kFloat,
              "g_cumsum must be fp32 [L, HV], contiguous in last dim");
  TORCH_CHECK(beta.dim() == 2 && beta.stride(-1) == 1 &&
                  beta.scalar_type() == at::kFloat,
              "beta must be fp32 [L, HV], contiguous in last dim");

  const long L = mixed_qkv.size(0);
  const long qkv_dim = mixed_qkv.size(1);
  const long HV = a.size(1);
  const long H = q.size(1);
  const long K = q.size(2);
  const long V = v.size(2);
  TORCH_CHECK(K == 128, "gdn_prefill_prep_rdna2 requires head_k_dim == 128");
  TORCH_CHECK(V == 128, "gdn_prefill_prep_rdna2 requires head_v_dim == 128");
  TORCH_CHECK(qkv_dim == 2 * H * K + HV * V,
              "mixed_qkv last dim must equal 2*H*K + HV*V");
  TORCH_CHECK(b.size(0) == L && b.size(1) == HV, "b shape mismatch");
  TORCH_CHECK(A_log.size(0) == HV, "A_log size mismatch");
  TORCH_CHECK(dt_bias.size(0) == HV, "dt_bias size mismatch");
  TORCH_CHECK(k_out.size(0) == L && k_out.size(1) == H &&
                  k_out.size(2) == K,
              "k_out shape mismatch");
  TORCH_CHECK(v.size(0) == L && v.size(1) == HV, "v shape mismatch");
  TORCH_CHECK(g_cumsum.size(0) == L && g_cumsum.size(1) == HV,
              "g_cumsum shape mismatch");
  TORCH_CHECK(beta.size(0) == L && beta.size(1) == HV,
              "beta size mismatch");
  if (L == 0) return;

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
    NT = (L + PREP_BT - 1) / PREP_BT;
  }
  if (NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(NT, (unsigned int)(H + HV));
#define PREP_LAUNCH(AT, DT, APTR, DPTR)                                  \
  gdn_prefill_prep_rdna2_kernel<false, AT, DT>                           \
      <<<grid, PREP_THREADS, 0, stream>>>(                               \
          reinterpret_cast<const __half*>(mixed_qkv.data_ptr()),         \
          reinterpret_cast<const __half*>(a.data_ptr()),                 \
          reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,     \
          reinterpret_cast<__half*>(q.data_ptr()),                       \
          reinterpret_cast<__half*>(k_out.data_ptr()),                   \
          reinterpret_cast<__half*>(v.data_ptr()),                       \
          g_cumsum.data_ptr<float>(), beta.data_ptr<float>(),            \
          mixed_qkv.stride(0), a.stride(0), b.stride(0), q.stride(0),    \
          k_out.stride(0), v.stride(0), L, H, HV, nullptr, nullptr, NT)
#define PREP_LAUNCH_VARLEN(AT, DT, APTR, DPTR)                           \
  gdn_prefill_prep_rdna2_kernel<true, AT, DT>                            \
      <<<grid, PREP_THREADS, 0, stream>>>(                               \
          reinterpret_cast<const __half*>(mixed_qkv.data_ptr()),         \
          reinterpret_cast<const __half*>(a.data_ptr()),                 \
          reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,     \
          reinterpret_cast<__half*>(q.data_ptr()),                       \
          reinterpret_cast<__half*>(k_out.data_ptr()),                   \
          reinterpret_cast<__half*>(v.data_ptr()),                       \
          g_cumsum.data_ptr<float>(), beta.data_ptr<float>(),            \
          mixed_qkv.stride(0), a.stride(0), b.stride(0), q.stride(0),    \
          k_out.stride(0), v.stride(0), L, H, HV,                        \
          cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>(), NT)
  if (is_varlen) {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      PREP_LAUNCH_VARLEN(float, float, A_log.data_ptr<float>(),
                         dt_bias.data_ptr<float>());
    } else if (a_ty == at::kFloat) {
      PREP_LAUNCH_VARLEN(
          float, __half, A_log.data_ptr<float>(),
          reinterpret_cast<const __half*>(dt_bias.data_ptr()));
    } else if (d_ty == at::kFloat) {
      PREP_LAUNCH_VARLEN(__half, float,
                         reinterpret_cast<const __half*>(A_log.data_ptr()),
                         dt_bias.data_ptr<float>());
    } else {
      PREP_LAUNCH_VARLEN(
          __half, __half, reinterpret_cast<const __half*>(A_log.data_ptr()),
          reinterpret_cast<const __half*>(dt_bias.data_ptr()));
    }
  } else {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      PREP_LAUNCH(float, float, A_log.data_ptr<float>(),
                  dt_bias.data_ptr<float>());
    } else if (a_ty == at::kFloat) {
      PREP_LAUNCH(float, __half, A_log.data_ptr<float>(),
                  reinterpret_cast<const __half*>(dt_bias.data_ptr()));
    } else if (d_ty == at::kFloat) {
      PREP_LAUNCH(__half, float,
                  reinterpret_cast<const __half*>(A_log.data_ptr()),
                  dt_bias.data_ptr<float>());
    } else {
      PREP_LAUNCH(__half, __half,
                  reinterpret_cast<const __half*>(A_log.data_ptr()),
                  reinterpret_cast<const __half*>(dt_bias.data_ptr()));
    }
  }
#undef PREP_LAUNCH
#undef PREP_LAUNCH_VARLEN
}
