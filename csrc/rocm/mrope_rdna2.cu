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
__global__ void mrope_forward_rdna2_kernel(
    __half* __restrict__ q,             // [num_tokens, n_qh * hd]
    __half* __restrict__ k,             // [num_tokens, n_kh * hd]
    const __half* __restrict__ cos,     // [3, num_tokens, rd/2]
    const __half* __restrict__ sin,     // [3, num_tokens, rd/2]
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

  for (int i = 0; i < half_rd; ++i) {
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
      for (int h = 0; h < n_qh; ++h) {
        const float q1 = __half2float(q_ptr[(long)h * hd + i]);
        const float q2 = __half2float(q_ptr[(long)h * hd + i + half_rd]);
        q_ptr[(long)h * hd + i] = __float2half(q1 * c - q2 * s);
        q_ptr[(long)h * hd + i + half_rd] = __float2half(q2 * c + q1 * s);
      }
      for (int h = 0; h < n_kh; ++h) {
        const float k1 = __half2float(k_ptr[(long)h * hd + i]);
        const float k2 = __half2float(k_ptr[(long)h * hd + i + half_rd]);
        k_ptr[(long)h * hd + i] = __float2half(k1 * c - k2 * s);
        k_ptr[(long)h * hd + i + half_rd] = __float2half(k2 * c + k1 * s);
      }
    } else {
      for (int h = 0; h < n_qh; ++h) {
        const float q1 = __half2float(q_ptr[(long)h * hd + 2 * i]);
        const float q2 = __half2float(q_ptr[(long)h * hd + 2 * i + 1]);
        q_ptr[(long)h * hd + 2 * i] = __float2half(q1 * c - q2 * s);
        q_ptr[(long)h * hd + 2 * i + 1] = __float2half(q2 * c + q1 * s);
      }
      for (int h = 0; h < n_kh; ++h) {
        const float k1 = __half2float(k_ptr[(long)h * hd + 2 * i]);
        const float k2 = __half2float(k_ptr[(long)h * hd + 2 * i + 1]);
        k_ptr[(long)h * hd + 2 * i] = __float2half(k1 * c - k2 * s);
        k_ptr[(long)h * hd + 2 * i + 1] = __float2half(k2 * c + k1 * s);
      }
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
                  cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf,
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
  TORCH_CHECK(cos.size(0) == 3 && cos.size(1) == num_tokens_i &&
                  cos.size(2) == half_rd,
              "cos must be [3, num_tokens, rd/2]");
  TORCH_CHECK(sin.size(0) == 3 && sin.size(1) == num_tokens_i &&
                  sin.size(2) == half_rd,
              "sin must be [3, num_tokens, rd/2]");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_tokens_i);
  vllm::rdna2_mrope::mrope_forward_rdna2_kernel<<<grid, 1, 0, stream.stream()>>>(
      reinterpret_cast<__half*>(q.data_ptr()),
      reinterpret_cast<__half*>(k.data_ptr()),
      reinterpret_cast<const __half*>(cos.data_ptr()),
      reinterpret_cast<const __half*>(sin.data_ptr()), num_tokens_i, n_qh_i,
      n_kh_i, hd_i, rd_i, (int)sec_t, (int)sec_h, (int)sec_w, is_interleaved,
      is_neox_style);
}
