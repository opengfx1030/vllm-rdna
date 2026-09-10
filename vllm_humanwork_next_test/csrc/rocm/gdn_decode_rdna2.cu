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

template <typename AT, typename DT>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_decode_packed_rdna2_kernel(
        const __half* __restrict__ mixed_qkv,   // [B, 2*H*K + HV*V]
        const __half* __restrict__ a,           // [B, HV]
        const __half* __restrict__ b,           // [B, HV]
        const AT* __restrict__ A_log,           // [HV]
        const DT* __restrict__ dt_bias,         // [HV]
        __half* __restrict__ out,               // [B, HV, V]
        float* __restrict__ state,              // [blocks, HV, V, K] in-place
        const int* __restrict__ ssm_state_indices,  // [B]
        float scale, long num_blocks_g, long stride_qkv_tok,
        long stride_a_tok, long stride_b_tok, long stride_state_block,
        long stride_indices_seq, int H, int HV, int V) {
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

  if (state_idx < 0 || state_idx >= num_blocks_g) {
    // NULL_BLOCK_ID collision guard: zero the output, leave state untouched.
    if (v_ok) {
      out[((long)i_n * HV + i_hv) * V + o_v] = __float2half(0.0f);
    }
    return;
  }

  // State tile h[o_v, k0 : k0+16] -> registers.
  float* p_h = state + state_idx * stride_state_block +
               (long)i_hv * V * GDN_K + (long)o_v * GDN_K + k0;
  float h[16];
  if (v_ok) {
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = p_h[j];
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
  const float beta =
      __half2float(__float2half(1.0f / (1.0f + expf(-b_val))));

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
    for (int j = 0; j < 16; ++j) p_h[j] = h[j];
  }
}

}  // namespace

void gdn_decode_rdna2(torch::Tensor mixed_qkv, torch::Tensor a,
                      torch::Tensor b, torch::Tensor A_log,
                      torch::Tensor dt_bias, torch::Tensor out,
                      torch::Tensor initial_state,
                      torch::Tensor ssm_state_indices, double scale,
                      bool use_qk_l2norm) {
  TORCH_CHECK(use_qk_l2norm,
              "gdn_decode_rdna2 only supports use_qk_l2norm_in_kernel=True");
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.stride(-1) == 1,
              "mixed_qkv must be [B, qkv_dim], contiguous in last dim");
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kHalf,
              "gdn_decode_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(a.dim() == 2 && a.stride(-1) == 1 &&
                  a.scalar_type() == at::kHalf,
              "a must be fp16 [B, HV], contiguous in last dim");
  TORCH_CHECK(b.dim() == 2 && b.stride(-1) == 1 &&
                  b.scalar_type() == at::kHalf,
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
                  initial_state.scalar_type() == at::kFloat,
              "initial_state must be fp32 [blocks, HV, V, K], contiguous in "
              "last dim");
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
#define GDN_LAUNCH(AT, DT, APTR, DPTR)                                     \
  gdn_decode_packed_rdna2_kernel<AT, DT>                                   \
      <<<grid, GDN_THREADS, 0, stream>>>(                                  \
          reinterpret_cast<const __half*>(mixed_qkv.data_ptr()),           \
          reinterpret_cast<const __half*>(a.data_ptr()),                   \
          reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,       \
          reinterpret_cast<__half*>(out.data_ptr()),                       \
          initial_state.data_ptr<float>(),                                 \
          ssm_state_indices.data_ptr<int>(), (float)scale,                 \
          (long)initial_state.size(0), mixed_qkv.stride(0), a.stride(0),   \
          b.stride(0), initial_state.stride(0), ssm_state_indices.stride(0), \
          H, HV, V)
  if (a_ty == at::kFloat && d_ty == at::kFloat) {
    GDN_LAUNCH(float, float, A_log.data_ptr<float>(), dt_bias.data_ptr<float>());
  } else if (a_ty == at::kFloat) {
    GDN_LAUNCH(float, __half, A_log.data_ptr<float>(),
               reinterpret_cast<const __half*>(dt_bias.data_ptr()));
  } else if (d_ty == at::kFloat) {
    GDN_LAUNCH(__half, float, reinterpret_cast<const __half*>(A_log.data_ptr()),
               dt_bias.data_ptr<float>());
  } else {
    GDN_LAUNCH(__half, __half, reinterpret_cast<const __half*>(A_log.data_ptr()),
               reinterpret_cast<const __half*>(dt_bias.data_ptr()));
  }
#undef GDN_LAUNCH
}
