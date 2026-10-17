#include <cuda_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {
namespace rdna2_mrope {

// M-RoPE (multi-modal rotary) forward for gfx1030. Port of
// vllm/model_executor/layers/rotary_embedding/mrope.py::_triton_mrope_forward.
//
// One program per token. The rotary pairs are applied to the query and key
// in-place. `cos`/`sin` are [3, num_tokens, rd/2] (the T/H/W sections).
//
// is_neox_style: the rotary pairs are (i, i+rd/2) (split-half). Otherwise the
// pairs are (2i, 2i+1) (GPT-J adjacent).
// is_interleaved: the T/H/W sections are interleaved (cos_offsets % 3).
//
// The two values of a rotary pair live in columns `col` and `col + pair`.
// Batching B heads' loads before any of their stores is worth ~13 % at the
// 2048-token prefill shape (rd=256, n_qh=24): the compiler cannot raise the
// next head's load above the current head's store, both of which go through the
// same __restrict__ base pointer, so the serial pair-by-pair loop keeps one
// load per wave in flight while the memory system wants >= 4.
constexpr int kQBatch = 4;
constexpr int kKVBatch = 2;

template <int B>
__device__ __forceinline__ void rotate_pair_heads(__half* __restrict__ p,
                                                  int col, int pair, int hd,
                                                  int nh, float c, float s) {
  int h = 0;
  for (; h + B <= nh; h += B) {
    float a[B], b[B];
#pragma unroll
    for (int u = 0; u < B; ++u) {
      a[u] = __half2float(p[(long)(h + u) * hd + col]);
      b[u] = __half2float(p[(long)(h + u) * hd + col + pair]);
    }
#pragma unroll
    for (int u = 0; u < B; ++u) {
      p[(long)(h + u) * hd + col] = __float2half(a[u] * c - b[u] * s);
      p[(long)(h + u) * hd + col + pair] = __float2half(b[u] * c + a[u] * s);
    }
  }
  for (; h < nh; ++h) {
    const float a = __half2float(p[(long)h * hd + col]);
    const float b = __half2float(p[(long)h * hd + col + pair]);
    p[(long)h * hd + col] = __float2half(a * c - b * s);
    p[(long)h * hd + col + pair] = __float2half(b * c + a * s);
  }
}

__global__ void mrope_forward_rdna2_kernel(
    __half* __restrict__ q,          // [num_tokens, n_qh * hd]
    __half* __restrict__ k,          // [num_tokens, n_kh * hd]
    const __half* __restrict__ cos,  // [3, num_tokens, rd/2]
    const __half* __restrict__ sin,  // [3, num_tokens, rd/2]
    const int num_tokens, const int n_qh, const int n_kh, const int hd,
    const int rd, const int sec_t, const int sec_h, const int sec_w,
    const bool is_interleaved, const bool is_neox_style) {
  const int pid = blockIdx.x;
  if (pid >= num_tokens) return;
  const int half_rd = rd >> 1;

  __half* q_ptr = q + (long)pid * n_qh * hd;
  __half* k_ptr = k + (long)pid * n_kh * hd;

  const __half* t_cos = cos + (long)pid * half_rd;
  const __half* h_cos = t_cos + (long)num_tokens * half_rd;
  const __half* w_cos = h_cos + (long)num_tokens * half_rd;
  const __half* t_sin = sin + (long)pid * half_rd;
  const __half* h_sin = t_sin + (long)num_tokens * half_rd;
  const __half* w_sin = h_sin + (long)num_tokens * half_rd;

  const int t_end = sec_t;
  const int h_end = t_end + sec_h;

  for (int i = threadIdx.x; i < half_rd; i += blockDim.x) {
    bool h_mask = false, w_mask = false;
    if (is_interleaved) {
      h_mask = (i % 3 == 1) && (i <= 3 * sec_h);
      w_mask = (i % 3 == 2) && (i <= 3 * sec_w);
    } else {
      h_mask = (t_end <= i) && (i < h_end);
      w_mask = (h_end <= i) && (i < half_rd);
    }
    const bool t_mask = !(h_mask || w_mask);
    const float c = (t_mask ? __half2float(t_cos[i]) : 0.0f) +
                    (h_mask ? __half2float(h_cos[i]) : 0.0f) +
                    (w_mask ? __half2float(w_cos[i]) : 0.0f);
    const float s = (t_mask ? __half2float(t_sin[i]) : 0.0f) +
                    (h_mask ? __half2float(h_sin[i]) : 0.0f) +
                    (w_mask ? __half2float(w_sin[i]) : 0.0f);

    if (is_neox_style) {
      rotate_pair_heads<kQBatch>(q_ptr, i, half_rd, hd, n_qh, c, s);
      rotate_pair_heads<kKVBatch>(k_ptr, i, half_rd, hd, n_kh, c, s);
    } else {
      rotate_pair_heads<kQBatch>(q_ptr, 2 * i, 1, hd, n_qh, c, s);
      rotate_pair_heads<kKVBatch>(k_ptr, 2 * i, 1, hd, n_kh, c, s);
    }
  }
}

}  // namespace rdna2_mrope
}  // namespace vllm

// Host wrapper stays at global scope so the torch_bindings importer resolves
// ::mrope_forward_rdna2 (same pattern as causal_conv1d_rdna2.cu).
void mrope_forward_rdna2(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                         torch::Tensor sin, int64_t num_tokens, int64_t n_qh,
                         int64_t n_kh, int64_t hd, int64_t rd, int64_t sec_t,
                         int64_t sec_h, int64_t sec_w, bool is_interleaved,
                         bool is_neox_style) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "mrope_forward_rdna2: all tensors must be on HIP");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  cos.scalar_type() == at::kHalf &&
                  sin.scalar_type() == at::kHalf,
              "mrope_forward_rdna2: fp16 only");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2, "q/k must be [tokens, heads*hd]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");

  const int num_tokens_i = (int)num_tokens;
  const int n_qh_i = (int)n_qh;
  const int n_kh_i = (int)n_kh;
  const int hd_i = (int)hd;
  const int rd_i = (int)rd;
  const int half_rd = rd_i >> 1;
  TORCH_CHECK(rd_i % 2 == 0, "mrope_forward_rdna2: rotary_dim must be even");
  TORCH_CHECK(
      cos.size(0) == 3 && cos.size(1) == num_tokens_i && cos.size(2) == half_rd,
      "cos must be [3, num_tokens, rd/2]");
  TORCH_CHECK(
      sin.size(0) == 3 && sin.size(1) == num_tokens_i && sin.size(2) == half_rd,
      "sin must be [3, num_tokens, rd/2]");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_tokens_i);
  // One block per token; the block strides over the rotary index. One thread
  // per token left the decode step (3 tokens) running 3 live threads on a
  // 72-CU part: 495 us per call against 10 us here, same tensors, back-to-back.
  // VLLM_RDNA_MROPE_BLOCK overrides the block size (1 = the old launch shape).
  const char* block_env = std::getenv("VLLM_RDNA_MROPE_BLOCK");
  int nthreads = ((half_rd + 31) / 32) * 32;
  if (nthreads > 128) nthreads = 128;
  if (nthreads < 1) nthreads = 1;
  if (block_env != nullptr) {
    const int b = std::atoi(block_env);
    if (b > 0) nthreads = b;
  }
  using vllm::rdna2_mrope::mrope_forward_rdna2_kernel;
  // hipify needs the kernel name on the same line as <<<.
  mrope_forward_rdna2_kernel<<<grid, nthreads, 0, stream.stream()>>>(
      reinterpret_cast<__half*>(q.data_ptr()),
      reinterpret_cast<__half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(cos.data_ptr()),
      reinterpret_cast<const __half*>(sin.data_ptr()), num_tokens_i, n_qh_i,
      n_kh_i, hd_i, rd_i, (int)sec_t, (int)sec_h, (int)sec_w, is_interleaved,
      is_neox_style);
}
