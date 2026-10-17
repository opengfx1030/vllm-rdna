// GatedDeltaNet (GDN) packed single-token decode kernel for AMD RDNA2
// (gfx1030). Hand port of
// fused_recurrent_gated_delta_rule_packed_decode_kernel
// (vllm/third_party/flash_linear_attention/ops/fused_recurrent.py),
// is_kda=False path with scalar per-head sigmoid gating.
//
// Workgroup = one (token, value-head, V-tile). 256 threads = 32 v-rows x
// 8 k-slices; each thread owns 16 fp32 state registers
// h[v, ks*16 : ks*16+16]. All four K-reductions (q/k L2 norms, delta-rule
// matvec, output matvec) are warp-local __shfl_xor over the 8 k-slice
// lanes. State is register-resident (16 VGPR/thread); no LDS.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GDN_K = 128;        // head_k_dim; packed decode requires NK == 1
constexpr int GDN_BV = 32;        // V rows per workgroup
constexpr int GDN_THREADS = 256;  // 32 v-rows x 8 k-slices
constexpr float GDN_SOFTPLUS_THRESHOLD = 20.0f;
constexpr float GDN_L2NORM_EPS = 1e-6f;

__device__ __forceinline__ float gdn_ksum(float x) {
  // Reduce across the 8 k-slice lanes sharing one v-row. Lane layout is
  // v = lane >> 3, ks = lane & 7, so masks 1/2/4 stay inside the group.
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

__device__ __forceinline__ float gdn_to_f32(float x) { return x; }
__device__ __forceinline__ float gdn_to_f32(__half x) {
  return __half2float(x);
}
template <typename ST>
__device__ __forceinline__ ST gdn_to_st(float x);
template <>
__device__ __forceinline__ float gdn_to_st<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half gdn_to_st<__half>(float x) {
  return __float2half(x);
}

template <typename AT, typename DT, typename ST>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_decode_packed_rdna2_kernel(
        const __half* __restrict__ mixed_qkv,  // [B, 2*H*K + HV*V]
        const __half* __restrict__ a,          // [B, HV]
        const __half* __restrict__ b,          // [B, HV]
        const AT* __restrict__ A_log,          // [HV]
        const DT* __restrict__ dt_bias,        // [HV]
        __half* __restrict__ out,              // [B, HV, V]
        const ST* __restrict__ state,          // [blocks, HV, V, K] in-place
        const int* __restrict__ ssm_state_indices,  // [B]
        float scale, long num_blocks_g, long stride_qkv_tok, long stride_a_tok,
        long stride_b_tok, long stride_state_block, long stride_indices_seq,
        int H, int HV, int V) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / HV;
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / (HV / H);

  const int lane = threadIdx.x;
  const int v_local = lane >> 3;
  const int ks = lane & 7;
  const int o_v = i_v * GDN_BV + v_local;
  const bool v_ok = o_v < V;
  const int k0 = ks * 16;

  const long state_idx = (long)ssm_state_indices[i_n * stride_indices_seq];

  if (state_idx <= 0 || state_idx >= num_blocks_g) {
    // NULL_BLOCK_ID=0 (vllm.v1.attention.backends.utils.NULL_BLOCK_ID)
    // is the production sentinel for invalid/empty slots. Match the
    // Triton reference: zero output and return without touching state.
    // Without this guard, slot 0 (which holds real weights in
    // production caches) is used as a valid tile and corrupts the GDN
    // recurrence on the first scheduled request.
    if (v_ok) {
      out[((long)i_n * HV + i_hv) * V + o_v] = __float2half(0.0f);
    }
    return;
  }

  // State tile h[o_v, k0 : k0+16] -> registers.
  ST* p_h = const_cast<ST*>(state) + state_idx * stride_state_block +
            (long)i_hv * V * GDN_K + (long)o_v * GDN_K + k0;
  float h[16];
  if (v_ok) {
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = gdn_to_f32(p_h[j]);
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = 0.0f;
  }

  // q/k slices for this thread's k-range (independent of v; the 4 v-row
  // groups per warp load the same 16 elements, L2-served).
  const __half* p_m = mixed_qkv + (long)i_n * stride_qkv_tok;
  const __half* p_q = p_m + i_h * GDN_K + k0;
  const __half* p_k = p_m + (long)H * GDN_K + i_h * GDN_K + k0;
  float q[16], kk[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    q[j] = __half2float(p_q[j]);
    kk[j] = __half2float(p_k[j]);
  }

  // L2 norms over K (use_qk_l2norm_in_kernel semantics).
  float sq = 0.0f, sk = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    sq += q[j] * q[j];
    sk += kk[j] * kk[j];
  }
  sq = gdn_ksum(sq);
  sk = gdn_ksum(sk);
  const float q_scale = scale / sqrtf(sq + GDN_L2NORM_EPS);
  const float k_scale = 1.0f / sqrtf(sk + GDN_L2NORM_EPS);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    q[j] *= q_scale;
    kk[j] *= k_scale;
  }

  // Scalar per-head sigmoid gating (non-KDA path).
  const float a_val = __half2float(a[(long)i_n * stride_a_tok + i_hv]);
  const float b_val = __half2float(b[(long)i_n * stride_b_tok + i_hv]);
  const float x = a_val + gdn_to_f32(dt_bias[i_hv]);
  const float softplus =
      (x <= GDN_SOFTPLUS_THRESHOLD) ? logf(1.0f + expf(x)) : x;
  const float g = -expf(gdn_to_f32(A_log[i_hv])) * softplus;
  // Match Triton: sigmoid computed in the b dtype (fp16) then widened.
  const float beta = __half2float(__float2half(1.0f / (1.0f + expf(-b_val))));

  // Delta rule: decay -> remove -> scale -> write back -> readout.
  const float decay = expf(g);
#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] *= decay;

  float v_val = 0.0f;
  if (v_ok) {
    v_val = __half2float(p_m[2L * H * GDN_K + (long)i_hv * V + o_v]);
  }
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += h[j] * kk[j];
  acc = gdn_ksum(acc);
  const float v2 = (v_val - acc) * beta;

#pragma unroll
  for (int j = 0; j < 16; ++j) h[j] += v2 * kk[j];

  float o_acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc += h[j] * q[j];
  o_acc = gdn_ksum(o_acc);

  if (v_ok) {
    out[((long)i_n * HV + i_hv) * V + o_v] = __float2half(o_acc);
    // In-place paged state store; each (token, head) tile is owned by
    // exactly one workgroup, so no race.
#pragma unroll
    for (int j = 0; j < 16; ++j) p_h[j] = gdn_to_st<ST>(h[j]);
  }
}

}  // namespace

void gdn_decode_rdna2(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
                      torch::Tensor A_log, torch::Tensor dt_bias,
                      torch::Tensor out, torch::Tensor initial_state,
                      torch::Tensor ssm_state_indices, double scale,
                      bool use_qk_l2norm) {
  TORCH_CHECK(use_qk_l2norm,
              "gdn_decode_rdna2 only supports use_qk_l2norm_in_kernel=True");
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.stride(-1) == 1,
              "mixed_qkv must be [B, qkv_dim], contiguous in last dim");
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kHalf,
              "gdn_decode_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(a.dim() == 2 && a.stride(-1) == 1 && a.scalar_type() == at::kHalf,
              "a must be fp16 [B, HV], contiguous in last dim");
  TORCH_CHECK(b.dim() == 2 && b.stride(-1) == 1 && b.scalar_type() == at::kHalf,
              "b must be fp16 [B, HV], contiguous in last dim");
  const auto a_ty = A_log.scalar_type();
  const auto d_ty = dt_bias.scalar_type();
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  (a_ty == at::kFloat || a_ty == at::kHalf),
              "A_log must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  (d_ty == at::kFloat || d_ty == at::kHalf),
              "dt_bias must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(initial_state.dim() == 4 && initial_state.stride(-1) == 1 &&
                  (initial_state.scalar_type() == at::kFloat ||
                   initial_state.scalar_type() == at::kHalf),
              "initial_state must be fp32/fp16 [blocks, HV, V, K], "
              "contiguous in last dim");
  TORCH_CHECK(ssm_state_indices.dim() == 1 &&
                  ssm_state_indices.scalar_type() == at::kInt,
              "ssm_state_indices must be int32 [B]");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kHalf,
              "out must be contiguous fp16 [B, 1, HV, V]");

  const int B = mixed_qkv.size(0);
  const int HV = initial_state.size(-3);
  const int V = initial_state.size(-2);
  const int K = initial_state.size(-1);
  TORCH_CHECK(K == GDN_K, "gdn_decode_rdna2 requires head_k_dim == 128");
  const int qk_dim = mixed_qkv.size(1) - HV * V;
  TORCH_CHECK(qk_dim > 0 && qk_dim % (2 * K) == 0,
              "invalid packed mixed_qkv last dim");
  const int H = qk_dim / (2 * K);
  TORCH_CHECK(HV % H == 0, "HV must be divisible by H");
  TORCH_CHECK(out.size(0) == B && out.size(2) == HV && out.size(3) == V,
              "out shape mismatch");
  TORCH_CHECK(a.size(0) == B && a.size(1) == HV, "a shape mismatch");
  TORCH_CHECK(b.size(0) == B && b.size(1) == HV, "b shape mismatch");
  TORCH_CHECK(ssm_state_indices.size(0) == B,
              "ssm_state_indices shape mismatch");
  if (B == 0) return;

  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((V + GDN_BV - 1) / GDN_BV, B * HV);
#define GDN_LAUNCH(AT, DT, ST, APTR, DPTR, SPTR)                             \
  gdn_decode_packed_rdna2_kernel<AT, DT, ST>                                 \
      <<<grid, GDN_THREADS, 0, stream>>>(                                    \
          reinterpret_cast<const __half*>(mixed_qkv.data_ptr()),             \
          reinterpret_cast<const __half*>(a.data_ptr()),                     \
          reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,         \
          reinterpret_cast<__half*>(out.data_ptr()), SPTR,                   \
          ssm_state_indices.data_ptr<int>(), (float)scale,                   \
          (long)initial_state.size(0), mixed_qkv.stride(0), a.stride(0),     \
          b.stride(0), initial_state.stride(0), ssm_state_indices.stride(0), \
          H, HV, V)
  if (initial_state.scalar_type() == at::kFloat) {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      GDN_LAUNCH(float, float, float, A_log.data_ptr<float>(),
                 dt_bias.data_ptr<float>(), initial_state.data_ptr<float>());
    } else if (a_ty == at::kFloat) {
      GDN_LAUNCH(float, __half, float, A_log.data_ptr<float>(),
                 reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                 initial_state.data_ptr<float>());
    } else if (d_ty == at::kFloat) {
      GDN_LAUNCH(__half, float, float,
                 reinterpret_cast<const __half*>(A_log.data_ptr()),
                 dt_bias.data_ptr<float>(), initial_state.data_ptr<float>());
    } else {
      GDN_LAUNCH(__half, __half, float,
                 reinterpret_cast<const __half*>(A_log.data_ptr()),
                 reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                 initial_state.data_ptr<float>());
    }
  } else {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      GDN_LAUNCH(float, float, __half, A_log.data_ptr<float>(),
                 dt_bias.data_ptr<float>(),
                 reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else if (a_ty == at::kFloat) {
      GDN_LAUNCH(float, __half, __half, A_log.data_ptr<float>(),
                 reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                 reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else if (d_ty == at::kFloat) {
      GDN_LAUNCH(__half, float, __half,
                 reinterpret_cast<const __half*>(A_log.data_ptr()),
                 dt_bias.data_ptr<float>(),
                 reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else {
      GDN_LAUNCH(__half, __half, __half,
                 reinterpret_cast<const __half*>(A_log.data_ptr()),
                 reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                 reinterpret_cast<const __half*>(initial_state.data_ptr()));
    }
  }
#undef GDN_LAUNCH
}

// =============================================================================
// GatedDeltaNet (GDN) speculative-decoding verify kernel for AMD RDNA2
// (gfx1030). Hand port of fused_sigmoid_gating_delta_rule_update_kernel
// (vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py) for the
// shape the MTP spec-verify path uses: IS_VARLEN + IS_CONTINUOUS_BATCHING +
// IS_SPEC_DECODING + USE_INITIAL_STATE + INPLACE_FINAL_STATE, is_kda=False,
// NK == 1 (head_k_dim <= 128), BV == 32.
//
// One workgroup per (V-tile, sequence x value-head), all tokens of the
// sequence are processed in order in registers: the per-token state write is
// what makes this different from the single-token gdn_decode_rdna2 kernel.
// The slot that is loaded is ssm_state_indices[n, num_accepted_tokens - 1] and
// every token then stores its state to ssm_state_indices[n, t], which is the
// sliding-window bookkeeping the Triton kernel implements.
//
// Numerics: fp32 accumulation over fp16 inputs, identical op order to the
// Triton kernel (decay, delta, state update, readout). The four K-reductions
// are 8-lane __shfl_xor butterflies over 16 contiguous elements per lane,
// which is the Triton layout for a [32, 128] tile on a 256-thread block.
// Transcendentals follow the Triton AMD lowering exactly: exp(y) is
// exp2(y * log2(e)), log is logf (llvm.log.f32) and rsqrt is __frsqrt_rn
// (llvm.amdgcn.rsq.f32).

namespace {

// Triton lowers `tl.exp` to `exp2(x * log2(e))` (`llvm.exp2.f32`) and
// `tl.rsqrt` to the raw `llvm.amdgcn.rsq.f32`. HIP's `__expf`/`rsqrtf` are
// different implementations of the same functions (the raw amdgcn intrinsic
// resp. a precise ocml routine) and differ by an ulp for some arguments.
constexpr float GDN_LOG2E = 0x1.715476p+0f;

__device__ __forceinline__ float gdn_exp(float x) {
  return exp2f(x * GDN_LOG2E);
}

template <typename AT, typename DT, typename ST>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_spec_verify_rdna2_kernel(
        const __half* __restrict__ q,    // [T, H, K]
        const __half* __restrict__ k,    // [T, H, K]
        const __half* __restrict__ v,    // [T, HV, V]
        const __half* __restrict__ a,    // [T, HV]
        const __half* __restrict__ b,    // [T, HV]
        const AT* __restrict__ A_log,    // [HV]
        const DT* __restrict__ dt_bias,  // [HV]
        __half* __restrict__ out,        // [T, HV, V]
        const ST* __restrict__ state,    // [blocks, HV, V, K]
        const int* __restrict__ ssm_state_indices,
        const int* __restrict__ num_accepted_tokens,  // nullptr when not spec
        const int* __restrict__ cu_seqlens,           // [N + 1]
        int H, int HV, int V, long stride_q_tok, long stride_k_tok,
        long stride_v_tok, long stride_a_tok, long stride_b_tok,
        long stride_o_tok, long stride_state_block, int stride_indices_seq,
        int stride_indices_tok, float scale) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / HV;
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / (HV / H);

  const int lane = threadIdx.x;
  const int v_local = lane >> 3;
  const int ks = lane & 7;
  const int o_v = i_v * GDN_BV + v_local;
  const bool v_ok = o_v < V;
  const int k0 = ks * 16;

  const int bos = cu_seqlens[i_n];
  const int eos = cu_seqlens[i_n + 1];
  const int n_tok = eos - bos;
  if (n_tok <= 0) return;

  // IS_SPEC_DECODING: the state to start from is the accepted prefix of the
  // previous step's slots.
  const int i_t0 = num_accepted_tokens != nullptr ? num_accepted_tokens[i_n] - 1 : 0;
  const int load_idx =
      ssm_state_indices[i_n * stride_indices_seq + i_t0 * stride_indices_tok];
  const int* idx_row = ssm_state_indices + i_n * stride_indices_seq;

  if (load_idx <= 0) {
    // NULL_BLOCK_ID (or an invalid slot): the Triton kernel returns before
    // writing `o`; zero it instead of leaving the caller's buffer undefined.
    if (v_ok) {
      for (int t = 0; t < n_tok; ++t) {
        out[(long)(bos + t) * stride_o_tok + (long)i_hv * V + o_v] =
            __float2half(0.0f);
      }
    }
    return;
  }

  float h[16];
  if (v_ok) {
    const ST* p_h0 = state + load_idx * stride_state_block +
                     (long)i_hv * V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = gdn_to_f32(p_h0[j]);
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = 0.0f;
  }

  for (int t = 0; t < n_tok; ++t) {
    const long g = bos + t;
    const __half* p_q = q + g * stride_q_tok + (long)i_h * GDN_K + k0;
    const __half* p_k = k + g * stride_k_tok + (long)i_h * GDN_K + k0;

    float qf[16], kf[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      qf[j] = __half2float(p_q[j]);
      kf[j] = __half2float(p_k[j]);
    }

    // USE_QK_L2NORM_IN_KERNEL, then the query scale (Triton multiplies by the
    // rsqrt factor first and by `scale` afterwards -- keep both steps apart).
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      sq += qf[j] * qf[j];
      sk += kf[j] * kf[j];
    }
    sq = gdn_ksum(sq);
    sk = gdn_ksum(sk);
    const float q_norm = __frsqrt_rn(sq + GDN_L2NORM_EPS);
    const float k_norm = __frsqrt_rn(sk + GDN_L2NORM_EPS);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      qf[j] = qf[j] * q_norm * scale;
      kf[j] = kf[j] * k_norm;
    }

    // Sigmoid gating (non-KDA: scalar per value-head).
    const float a_val = __half2float(a[g * stride_a_tok + i_hv]);
    const float b_val = __half2float(b[g * stride_b_tok + i_hv]);
    const float x = a_val + gdn_to_f32(dt_bias[i_hv]);
    const float softplus =
        (x <= GDN_SOFTPLUS_THRESHOLD) ? logf(1.0f + gdn_exp(x)) : x;
    const float g_val = -gdn_exp(gdn_to_f32(A_log[i_hv])) * softplus;
    const float beta = 1.0f / (1.0f + gdn_exp(-b_val));

    const float decay = gdn_exp(g_val);
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] *= decay;

    const float v_val =
        v_ok ? __half2float(v[g * stride_v_tok + (long)i_hv * V + o_v]) : 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) acc += h[j] * kf[j];
    acc = gdn_ksum(acc);
    const float v2 = (v_val - acc) * beta;
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] += v2 * kf[j];

    float o_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc += h[j] * qf[j];
    o_acc = gdn_ksum(o_acc);
    if (v_ok) {
      out[g * stride_o_tok + (long)i_hv * V + o_v] = __float2half(o_acc);
    }

    // INPLACE_FINAL_STATE: token t of the sequence writes slot t.
    const int store_idx = idx_row[t * stride_indices_tok];
    if (store_idx > 0 && v_ok) {
      ST* p_ht = const_cast<ST*>(state) + store_idx * stride_state_block +
                 (long)i_hv * V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
      for (int j = 0; j < 16; ++j) p_ht[j] = gdn_to_st<ST>(h[j]);
    }
  }
}

}  // namespace

void gdn_spec_verify_rdna2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor a, torch::Tensor b,
                           torch::Tensor A_log, torch::Tensor dt_bias,
                           torch::Tensor out, torch::Tensor initial_state,
                           torch::Tensor cu_seqlens,
                           torch::Tensor ssm_state_indices,
                           torch::Tensor num_accepted_tokens, double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && a.is_cuda() &&
                  b.is_cuda() && out.is_cuda() && initial_state.is_cuda(),
              "gdn_spec_verify_rdna2: tensors must be CUDA");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf && a.scalar_type() == at::kHalf &&
                  b.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
              "gdn_spec_verify_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
              "gdn_spec_verify_rdna2 wants [B, T, H, K] / [B, T, HV, V]");
  // The model passes a/b as the flattened [T, HV] slices produced by
  // index_select on the per-step token buffer (q/k/v are [1, T, H, K]).
  TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && a.dim() == b.dim(),
              "gdn_spec_verify_rdna2 wants a/b as [T, HV] or [B, T, HV]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "gdn_spec_verify_rdna2 requires contiguous q/k/v/a/b/out");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.stride(0) == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be unit-stride int32 [N + 1]");

  const int T = q.size(1);
  const int H = q.size(2);
  const int K = q.size(3);
  const int HV = v.size(2);
  const int V = v.size(3);
  const int N = cu_seqlens.size(0) - 1;
  TORCH_CHECK(K == GDN_K, "gdn_spec_verify_rdna2 requires head_k_dim == 128");
  TORCH_CHECK(V >= GDN_BV && V % GDN_BV == 0,
              "gdn_spec_verify_rdna2 requires V >= 32 and V % 32 == 0");
  TORCH_CHECK(HV % H == 0, "HV must be divisible by H");
  // Token dim: 2-D a/b carry the flattened [T, HV] rows (one per token in the
  // varlen batch); 3-D carries [1, T, HV] over the same token axis as q.
  const int a_tok_dim = a.dim() - 2;
  TORCH_CHECK(v.size(1) == T && out.size(1) == T &&
                  a.size(a_tok_dim) == T && b.size(a_tok_dim) == T,
              "gdn_spec_verify_rdna2: token dim mismatch");
  TORCH_CHECK(a.size(a.dim() - 1) == HV && b.size(b.dim() - 1) == HV,
              "gdn_spec_verify_rdna2: a/b must be [T, HV] or [B, T, HV]");

  const auto a_ty = A_log.scalar_type();
  const auto d_ty = dt_bias.scalar_type();
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  (a_ty == at::kFloat || a_ty == at::kHalf),
              "A_log must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  (d_ty == at::kFloat || d_ty == at::kHalf),
              "dt_bias must be contiguous fp32/fp16 [HV]");
  TORCH_CHECK(initial_state.dim() == 4 && initial_state.stride(-1) == 1 &&
                  (initial_state.scalar_type() == at::kFloat ||
                   initial_state.scalar_type() == at::kHalf),
              "initial_state must be fp32/fp16 [blocks, HV, V, K], K-contiguous");
  TORCH_CHECK(initial_state.size(1) == HV && initial_state.size(2) == V &&
                  initial_state.size(3) == K,
              "initial_state shape [blocks, HV, V, K] mismatch");
  TORCH_CHECK(ssm_state_indices.scalar_type() == at::kInt &&
                  (ssm_state_indices.dim() == 1 || ssm_state_indices.dim() == 2),
              "ssm_state_indices must be int32 [N, S] or [N]");
  const bool is_spec = num_accepted_tokens.defined() &&
                       num_accepted_tokens.numel() > 0;
  if (is_spec) {
    TORCH_CHECK(num_accepted_tokens.scalar_type() == at::kInt &&
                    num_accepted_tokens.stride(0) == 1 &&
                    num_accepted_tokens.numel() >= N,
                "num_accepted_tokens must be unit-stride int32 [N]");
  }
  if (N == 0 || T == 0) return;

  int stride_indices_seq, stride_indices_tok;
  if (ssm_state_indices.dim() == 1) {
    stride_indices_seq = (int)ssm_state_indices.stride(0);
    stride_indices_tok = 1;
  } else {
    stride_indices_seq = (int)ssm_state_indices.stride(0);
    stride_indices_tok = (int)ssm_state_indices.stride(1);
  }

  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((V + GDN_BV - 1) / GDN_BV, N * HV);
  const long stride_state_block = initial_state.stride(0);

#define GDN_SPEC_LAUNCH(AT, DT, ST, APTR, DPTR, SPTR)                          \
  gdn_spec_verify_rdna2_kernel<AT, DT, ST>                                     \
      <<<grid, GDN_THREADS, 0, stream>>>(                                      \
          reinterpret_cast<const __half*>(q.data_ptr()),                       \
          reinterpret_cast<const __half*>(k.data_ptr()),                       \
          reinterpret_cast<const __half*>(v.data_ptr()),                       \
          reinterpret_cast<const __half*>(a.data_ptr()),                       \
          reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,           \
          reinterpret_cast<__half*>(out.data_ptr()), SPTR,                     \
          ssm_state_indices.data_ptr<int>(),                                   \
          is_spec ? num_accepted_tokens.data_ptr<int>() : nullptr,             \
          cu_seqlens.data_ptr<int>(), H, HV, V, q.stride(1), k.stride(1),      \
          v.stride(1), a.stride(a_tok_dim), b.stride(a_tok_dim),             \
          out.stride(1),                                                     \
          stride_state_block, stride_indices_seq, stride_indices_tok,          \
          (float)scale)

  if (initial_state.scalar_type() == at::kFloat) {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(float, float, float, A_log.data_ptr<float>(),
                      dt_bias.data_ptr<float>(), initial_state.data_ptr<float>());
    } else if (a_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(float, __half, float, A_log.data_ptr<float>(),
                      reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                      initial_state.data_ptr<float>());
    } else if (d_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(__half, float, float,
                      reinterpret_cast<const __half*>(A_log.data_ptr()),
                      dt_bias.data_ptr<float>(),
                      initial_state.data_ptr<float>());
    } else {
      GDN_SPEC_LAUNCH(__half, __half, float,
                      reinterpret_cast<const __half*>(A_log.data_ptr()),
                      reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                      initial_state.data_ptr<float>());
    }
  } else {
    if (a_ty == at::kFloat && d_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(float, float, __half, A_log.data_ptr<float>(),
                      dt_bias.data_ptr<float>(),
                      reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else if (a_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(float, __half, __half, A_log.data_ptr<float>(),
                      reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                      reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else if (d_ty == at::kFloat) {
      GDN_SPEC_LAUNCH(__half, float, __half,
                      reinterpret_cast<const __half*>(A_log.data_ptr()),
                      dt_bias.data_ptr<float>(),
                      reinterpret_cast<const __half*>(initial_state.data_ptr()));
    } else {
      GDN_SPEC_LAUNCH(__half, __half, __half,
                      reinterpret_cast<const __half*>(A_log.data_ptr()),
                      reinterpret_cast<const __half*>(dt_bias.data_ptr()),
                      reinterpret_cast<const __half*>(initial_state.data_ptr()));
    }
  }
#undef GDN_SPEC_LAUNCH

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
