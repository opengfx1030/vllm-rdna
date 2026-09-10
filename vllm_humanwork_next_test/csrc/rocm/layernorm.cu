// SPDX-License-Identifier: Apache-2.0
// HIP RMSNorm / FusedAddRmsNorm for AMD RDNA (gfx1030/gfx1100).
//
// Replaces the upstream Triton `layer_norm_fwd_kernel` / `rms_norm_kernel`
// path on ROCm so that cudagraph capture on gfx1030 does not have to wait
// for Triton to JIT-compile a per-shape kernel variant during inference
// (the canonical source of the cudagraph NaN bug). AOT-compiled HIP code,
// one launch per row, no per-call dispatch — cudagraph-safe by design.
//
// API matches the CUDA `torch.ops._C.rms_norm` / `fused_add_rms_norm`:
//   rms_norm(out, input, weight, epsilon)
//   fused_add_rms_norm(input, residual, weight, epsilon)
// All tensors fp16, weight fp16, epsilon fp32 (double in C++).
//
// Both ops are row-wise: one CTA per row, BLOCK_DIM threads cooperate on
// the squared-sum reduction via warp shuffles + shared-memory cross-warp.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace vllm {
namespace rocm_layernorm {

constexpr int WARP_SIZE = 32;

// Row-wise reduction primitive: each thread owns `local` (a float), the
// block reduces across all BLOCK_DIM threads, and broadcasts the result
// through shared memory so every thread can use it after the call.
// Caller must use a power-of-two BLOCK_DIM <= 1024 and >= 32.
template <int BLOCK_DIM>
__device__ __forceinline__ float block_reduce_sum(float local) {
  static_assert(BLOCK_DIM % WARP_SIZE == 0,
                "BLOCK_DIM must be a multiple of warp size");
  constexpr int NUM_WARPS = BLOCK_DIM / WARP_SIZE;

  // 1. intra-warp reduce.
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, local, offset);
  }

  // 2. cross-warp reduce via shared memory.
  __shared__ float s_partial[NUM_WARPS];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int wid = threadIdx.x / WARP_SIZE;
  if (lane == 0) s_partial[wid] = local;
  __syncthreads();

  // 3. final reduce inside warp 0.
  if (wid == 0) {
    float v = (threadIdx.x < NUM_WARPS) ? s_partial[threadIdx.x] : 0.0f;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      v += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, v, offset);
    }
    if (threadIdx.x == 0) s_partial[0] = v;
  }
  __syncthreads();
  return s_partial[0];
}

// RMSNorm forward: y = x * rstd * weight, rstd = rsqrt(mean(x^2) + eps).
// One CTA per row. BLOCK_DIM threads cooperate on the squared sum.
template <int BLOCK_DIM>
__global__ void rms_norm_kernel(const __half* __restrict__ input,
                                const __half* __restrict__ weight,
                                __half* __restrict__ output, int N,
                                float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const __half* x_row = input + (size_t)row * N;
  __half* y_row = output + (size_t)row * N;

  // Pass 1: squared sum.
  float ssq = 0.0f;
  for (int i = tid; i < N; i += BLOCK_DIM) {
    float v = __half2float(x_row[i]);
    ssq += v * v;
  }
  ssq = block_reduce_sum<BLOCK_DIM>(ssq);
  const float rstd = rsqrtf(ssq / (float)N + epsilon);

  // Pass 2: y = x * rstd * weight.
  for (int i = tid; i < N; i += BLOCK_DIM) {
    float v = __half2float(x_row[i]);
    float w = __half2float(weight[i]);
    y_row[i] = __float2half(v * rstd * w);
  }
}

// Fused (x + residual) + RMSNorm forward.
//   residual <- x + residual     (in-place)
//   out      <- (x + residual) * rstd * weight
// One CTA per row.
template <int BLOCK_DIM>
__global__ void fused_add_rms_norm_kernel(const __half* __restrict__ input,
                                          __half* __restrict__ residual,
                                          const __half* __restrict__ weight,
                                          __half* __restrict__ output, int N,
                                          float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const __half* x_row = input + (size_t)row * N;
  __half* r_row = residual + (size_t)row * N;
  __half* y_row = output + (size_t)row * N;

  // Pass 1: residual += x, accumulate squared sum.
  float ssq = 0.0f;
  for (int i = tid; i < N; i += BLOCK_DIM) {
    float v = __half2float(x_row[i]);
    float r = __half2float(r_row[i]);
    float s = v + r;
    r_row[i] = __float2half(s);
    ssq += s * s;
  }
  ssq = block_reduce_sum<BLOCK_DIM>(ssq);
  const float rstd = rsqrtf(ssq / (float)N + epsilon);

  // Pass 2: y = updated_residual * rstd * weight.
  for (int i = tid; i < N; i += BLOCK_DIM) {
    float v = __half2float(r_row[i]);
    float w = __half2float(weight[i]);
    y_row[i] = __float2half(v * rstd * w);
  }
}

// Pick BLOCK_DIM based on N. Larger N → larger block for fewer strided
// passes; cap at 1024 (one CTA = one wave of 32 warps on gfx1030).
constexpr int pick_block_dim(int N) {
  if (N <= 512) return 128;
  if (N <= 2048) return 256;
  if (N <= 8192) return 512;
  return 1024;
}

// Dispatch helpers. We instantiate a small fixed set of BLOCK_DIM
// templates and dispatch at host-side. All INSTANTIATED variants are
// emitted at AOT time, so cudagraph capture has a stable, pre-compiled
// kernel binary for every supported (BLOCK_DIM, dtype) pair.
template <int BLOCK_DIM>
static void launch_rms_norm(const __half* in, const __half* w, __half* out,
                            int M, int N, float eps, hipStream_t stream) {
  dim3 grid(M);
  dim3 block(BLOCK_DIM);
  hipLaunchKernelGGL((rms_norm_kernel<BLOCK_DIM>), grid, block, 0, stream, in,
                     w, out, N, eps);
}

template <int BLOCK_DIM>
static void launch_fused_add_rms_norm(const __half* in, __half* res,
                                      const __half* w, __half* out, int M,
                                      int N, float eps, hipStream_t stream) {
  dim3 grid(M);
  dim3 block(BLOCK_DIM);
  hipLaunchKernelGGL((fused_add_rms_norm_kernel<BLOCK_DIM>), grid, block, 0,
                     stream, in, res, w, out, N, eps);
}

}  // namespace rocm_layernorm
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public C++ entry points (called from torch_bindings.cpp). Defined at global
// scope so torch_bindings.cpp can register them unqualified, matching every
// other _rocm_C op (gptq_gemm_rdna2, exl3_gemm_rdna2, fa_rdna2_decode_paged,
// etc.).
// ---------------------------------------------------------------------------

void rms_norm(at::Tensor& out, const at::Tensor& input,
              const at::Tensor& weight, double epsilon) {
  using namespace vllm::rocm_layernorm;
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  weight.scalar_type() == at::kHalf &&
                  out.scalar_type() == at::kHalf,
              "rms_norm: fp16 only");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, N]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [N]");
  TORCH_CHECK(input.size(1) == weight.size(0), "input.size(1) == weight.size(0)");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");

  // Reduce dtype (resid input is a float in the C++ API).
  const float eps = (float)epsilon;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int M = (int)input.size(0);
  const int N = (int)input.size(1);

  // All inputs must be contiguous (one CTA per row assumes row-major).
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  out.is_contiguous(),
              "rms_norm: inputs must be contiguous");

  const __half* in_p = (const __half*)input.data_ptr();
  const __half* w_p = (const __half*)weight.data_ptr();
  __half* out_p = (__half*)out.data_ptr();

  switch (pick_block_dim(N)) {
    case 128:
      launch_rms_norm<128>(in_p, w_p, out_p, M, N, eps, stream);
      break;
    case 256:
      launch_rms_norm<256>(in_p, w_p, out_p, M, N, eps, stream);
      break;
    case 512:
      launch_rms_norm<512>(in_p, w_p, out_p, M, N, eps, stream);
      break;
    default:
      launch_rms_norm<1024>(in_p, w_p, out_p, M, N, eps, stream);
      break;
  }
}

void fused_add_rms_norm(at::Tensor& input, at::Tensor& residual,
                        const at::Tensor& weight, double epsilon) {
  using namespace vllm::rocm_layernorm;
  TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  residual.scalar_type() == at::kHalf &&
                  weight.scalar_type() == at::kHalf,
              "fused_add_rms_norm: fp16 only");
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2 &&
                  weight.dim() == 1,
              "input, residual 2D; weight 1D");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual must "
                                                  "have the same shape");
  TORCH_CHECK(input.size(1) == weight.size(0),
              "input.size(1) == weight.size(0)");

  const float eps = (float)epsilon;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int M = (int)input.size(0);
  const int N = (int)input.size(1);

  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "fused_add_rms_norm: inputs must be contiguous");

  const __half* in_p = (const __half*)input.data_ptr();
  __half* res_p = (__half*)residual.data_ptr();
  const __half* w_p = (const __half*)weight.data_ptr();
  __half* out_p = (__half*)input.data_ptr();  // output reuses input's storage
  // NOTE: vLLM's CUDA signature does NOT take an `out` tensor — fused_add_rms_norm
  // returns the normalized result in-place of `input`. To stay API-compatible we
  // mirror that behavior: input is read once, then overwritten with the
  // normalized output. Caller (RMSNorm.forward) is expected to provide a fresh
  // input tensor it does not need afterwards (matches the CUDA op).
  // If the caller wants residual kept alongside, they must clone `input` first.
  // See vllm/_custom_ops.py:fused_add_rms_norm wrapper for the standard idiom.

  switch (pick_block_dim(N)) {
    case 128:
      launch_fused_add_rms_norm<128>(in_p, res_p, w_p, out_p, M, N, eps,
                                      stream);
      break;
    case 256:
      launch_fused_add_rms_norm<256>(in_p, res_p, w_p, out_p, M, N, eps,
                                      stream);
      break;
    case 512:
      launch_fused_add_rms_norm<512>(in_p, res_p, w_p, out_p, M, N, eps,
                                      stream);
      break;
    default:
      launch_fused_add_rms_norm<1024>(in_p, res_p, w_p, out_p, M, N, eps,
                                      stream);
      break;
  }
}