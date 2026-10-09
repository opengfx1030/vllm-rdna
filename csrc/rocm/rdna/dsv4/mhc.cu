// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DeepSeek-V4 manifold hyper-connection (mHC) pre / post for AMD RDNA.
//
// RDNA has neither TileLang nor AITER, so MHCPreOp / MHCPostOp ran the torch
// reference (mhc_pre_torch / mhc_post_torch): ~130 tiny kernels per mhc_pre
// call (the Sinkhorn loop alone is 20 x 4 ops), 86 calls per decode step.
// These kernels fuse everything after the one GEMM that stays in rocBLAS:
//
//   mhc_pre:  mixes = x @ fn^T, fp32 [T, 2*HC + HC*HC]: passed in by the
//             caller (rocBLAS, prefill) or, when fn is given (decode),
//             computed with sum(x^2) by a split-K partial kernel
//     rms   = rsqrt(sum(x^2) / (HC*H) + rms_eps)
//     pre   = sigmoid(mixes[0:HC]   * s0 + base[0:HC]) + pre_eps
//     post  = sigmoid(mixes[HC:2HC] * s1 + base[HC:2HC]) * post_mult
//     comb  = softmax_j(mixes[2HC:] * s2 + base[2HC:]) + sinkhorn_eps,
//             col-normalized, then (repeat - 1) x (row, col) normalization
//     layer_input[h] = sum_c pre[c] * residual[c, h]
//   mhc_post: out[j, h] = sum_i comb[i, j] * residual[i, h] + post[j] * x[h]
//
// Same math and op order as the torch reference (fp32 throughout). One
// block per token for pre; post is elementwise. Launch on the current
// stream, outputs (and the decode partials) from torch::empty: CUDA-graph
// safe.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace vllm {
namespace rdna {
namespace dsv4 {

constexpr int kHc = 4;
constexpr int kMix = 2 * kHc + kHc * kHc;  // 24
constexpr int kPreThreads = 256;
// Decode: x @ fn^T + sum(x^2) is split over this many blocks per token (one
// block streaming the 1.5 MB fp32 fn was single-CU bandwidth bound, ~37 us).
constexpr int kPreSplits = 32;

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 from_f32<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// partials[t, s, 0] = sum(x^2), partials[t, s, 1 + k] = x . fn[k] over the
// s-th chunk of the token's HC*H values. Grid (splits, T).
template <typename T>
__global__ __launch_bounds__(kPreThreads) void mhc_pre_partial_kernel(
    const T* __restrict__ residual,  // [T, HC*H]
    const float* __restrict__ fn,    // [24, HC*H]
    float* __restrict__ partials,    // [T, splits, 25]
    int n, int chunk) {
  const int s = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const T* res = residual + static_cast<int64_t>(t) * n;
  const int begin = s * chunk;
  const int end = min(begin + chunk, n);
  float part[kMix + 1];
#pragma unroll
  for (int k = 0; k <= kMix; k++) part[k] = 0.0f;
  for (int i = begin + tid; i < end; i += kPreThreads) {
    const float v = to_f32(res[i]);
    part[0] += v * v;
#pragma unroll
    for (int k = 0; k < kMix; k++)
      part[k + 1] += v * fn[static_cast<int64_t>(k) * n + i];
  }
#pragma unroll
  for (int k = 0; k <= kMix; k++) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
      part[k] += __shfl_down(part[k], off);
  }
  __shared__ float s_w[kPreThreads / 32][kMix + 1];
  const int lane = tid % warpSize;
  const int warp = tid / warpSize;
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k <= kMix; k++) s_w[warp][k] = part[k];
  }
  __syncthreads();
  if (tid <= kMix) {
    float acc = 0.0f;
    const int warps = kPreThreads / warpSize;
    for (int w = 0; w < warps; w++) acc += s_w[w][tid];
    partials[(static_cast<int64_t>(t) * gridDim.x + s) * (kMix + 1) + tid] =
        acc;
  }
}

template <typename T>
__global__ __launch_bounds__(kPreThreads) void mhc_pre_kernel(
    const T* __restrict__ residual,      // [T, HC, H]
    const float* __restrict__ mixes,     // [T, 24] = x @ fn^T, or null
    const float* __restrict__ partials,  // [T, splits, 25] when mixes is null
    int num_splits,
    const float* __restrict__ hc_scale,  // [3]
    const float* __restrict__ hc_base,   // [24]
    float* __restrict__ post_out,        // [T, HC]
    float* __restrict__ comb_out,        // [T, HC, HC]
    T* __restrict__ layer_input,         // [T, H]
    int hidden, float rms_eps, float pre_eps, float sinkhorn_eps,
    float post_mult, int sinkhorn_repeat) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const T* res = residual + static_cast<int64_t>(t) * kHc * hidden;

  __shared__ float s_red[kPreThreads];
  __shared__ float s_mix[kMix];
  __shared__ float s_pre[kHc];

  if (mixes == nullptr) {
    // Decode: reduce the per-split partials (sum(x^2) and x @ fn^T).
    if (tid <= kMix) {
      float acc = 0.0f;
      for (int sp = 0; sp < num_splits; sp++)
        acc += partials[(static_cast<int64_t>(t) * num_splits + sp) *
                            (kMix + 1) +
                        tid];
      if (tid == 0) {
        s_red[0] = acc;
      } else {
        s_mix[tid - 1] = acc;
      }
    }
  } else {
    const int n = kHc * hidden;
    float sq = 0.0f;
    for (int i = tid; i < n; i += kPreThreads) {
      const float v = to_f32(res[i]);
      sq += v * v;
    }
    s_red[tid] = sq;
    __syncthreads();
    if (tid == 0) {
      float acc = 0.0f;
      for (int r = 0; r < kPreThreads; r++) acc += s_red[r];
      s_red[0] = acc;
    }
  }
  __syncthreads();

  // 16 lanes of the first wave, lane = (i, j) of the 4x4 comb matrix; row
  // sums reduce over lane bits 0-1, column sums over bits 2-3. Same
  // per-element ops as the torch reference (the old single-lane loop did
  // ~640 serial divides).
  if (tid < kHc * kHc) {
    const float rms =
        rsqrtf(s_red[0] / static_cast<float>(kHc * hidden) + rms_eps);
    auto mix = [&](int k) {
      return (mixes == nullptr ? s_mix[k] : mixes[t * kMix + k]) * rms;
    };
    if (tid < kHc) {
      s_pre[tid] = sigmoidf(mix(tid) * hc_scale[0] + hc_base[tid]) + pre_eps;
      post_out[t * kHc + tid] =
          sigmoidf(mix(kHc + tid) * hc_scale[1] + hc_base[kHc + tid]) *
          post_mult;
    }
    auto row_sum = [](float v) {
      v += __shfl_xor(v, 1);
      return v + __shfl_xor(v, 2);
    };
    auto col_sum = [](float v) {
      v += __shfl_xor(v, 4);
      return v + __shfl_xor(v, 8);
    };
    const int k = 2 * kHc + tid;
    float v = mix(k) * hc_scale[2] + hc_base[k];
    float mx = fmaxf(v, __shfl_xor(v, 1));
    mx = fmaxf(mx, __shfl_xor(mx, 2));
    v = expf(v - mx);
    v = v / row_sum(v) + sinkhorn_eps;
    v /= col_sum(v) + sinkhorn_eps;
    for (int r = 0; r < sinkhorn_repeat - 1; r++) {
      v /= row_sum(v) + sinkhorn_eps;
      v /= col_sum(v) + sinkhorn_eps;
    }
    comb_out[t * kHc * kHc + tid] = v;
  }
  __syncthreads();

  const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
  T* out = layer_input + static_cast<int64_t>(t) * hidden;
  for (int h = tid; h < hidden; h += kPreThreads) {
    const float v = p0 * to_f32(res[h]) + p1 * to_f32(res[hidden + h]) +
                    p2 * to_f32(res[2 * hidden + h]) +
                    p3 * to_f32(res[3 * hidden + h]);
    out[h] = from_f32<T>(v);
  }
}

template <typename T>
__global__ void mhc_post_kernel(const T* __restrict__ x,         // [T, H]
                                const T* __restrict__ residual,  // [T, HC, H]
                                const float* __restrict__ post,  // [T, HC]
                                const float* __restrict__ comb,  // [T, HC, HC]
                                T* __restrict__ out,             // [T, HC, H]
                                int hidden) {
  const int t = blockIdx.y;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  const int64_t base = static_cast<int64_t>(t) * kHc * hidden;
  float r[kHc];
  for (int i = 0; i < kHc; i++) r[i] = to_f32(residual[base + i * hidden + h]);
  const float xv = to_f32(x[static_cast<int64_t>(t) * hidden + h]);
  const float* cm = comb + t * kHc * kHc;
  for (int j = 0; j < kHc; j++) {
    float acc = 0.0f;
    for (int i = 0; i < kHc; i++) acc += cm[i * kHc + j] * r[i];
    acc += post[t * kHc + j] * xv;
    out[base + j * hidden + h] = from_f32<T>(acc);
  }
}

}  // namespace dsv4
}  // namespace rdna
}  // namespace vllm

std::vector<torch::Tensor> dsv4_mhc_pre_rdna(
    torch::Tensor const& residual,  // [T, HC, H] fp16/bf16
    torch::Tensor const& mixes,     // [T, 24] fp32 = x @ fn^T, or [24, HC*H]
                                    // fp32 fn: the kernel does the GEMM
    torch::Tensor const& hc_scale,  // [3] fp32
    torch::Tensor const& hc_base,   // [24] fp32
    double rms_eps, double pre_eps, double sinkhorn_eps, double post_mult,
    int64_t sinkhorn_repeat) {
  namespace d = vllm::rdna::dsv4;
  TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 &&
                  residual.size(1) == d::kHc && residual.is_contiguous(),
              "residual must be contiguous [T, 4, H]");
  TORCH_CHECK(residual.scalar_type() == at::kHalf ||
                  residual.scalar_type() == at::kBFloat16,
              "residual must be fp16 or bf16");
  const bool given_fn = mixes.dim() == 2 && mixes.size(0) == d::kMix &&
                        mixes.size(1) == d::kHc * residual.size(2);
  TORCH_CHECK(mixes.scalar_type() == at::kFloat && mixes.is_contiguous() &&
                  mixes.dim() == 2 &&
                  (given_fn || (mixes.size(1) == d::kMix &&
                                mixes.size(0) == residual.size(0))),
              "mixes must be contiguous fp32 [T, 24] (or fn [24, 4*H])");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3 &&
                  hc_base.scalar_type() == at::kFloat &&
                  hc_base.numel() == d::kMix && hc_scale.is_contiguous() &&
                  hc_base.is_contiguous(),
              "hc_scale [3] / hc_base [24] must be contiguous fp32");
  const int64_t num_tokens = residual.size(0);
  const int hidden = static_cast<int>(residual.size(2));
  auto f32 = residual.options().dtype(at::kFloat);
  auto post = torch::empty({num_tokens, d::kHc, 1}, f32);
  auto comb = torch::empty({num_tokens, d::kHc, d::kHc}, f32);
  auto layer_input = torch::empty({num_tokens, hidden}, residual.options());
  if (num_tokens == 0) return {post, comb, layer_input};

  const at::cuda::OptionalCUDAGuard device_guard(device_of(residual));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = d::kHc * hidden;
  const int chunk = (n + d::kPreSplits - 1) / d::kPreSplits;
  torch::Tensor partials;
  if (given_fn) {
    partials = torch::empty({num_tokens, d::kPreSplits, d::kMix + 1}, f32);
  }
  auto launch = [&](auto tag) {
    using T = decltype(tag);
    if (given_fn) {
      d::mhc_pre_partial_kernel<T>
          <<<dim3(d::kPreSplits, static_cast<int>(num_tokens)), d::kPreThreads,
             0, stream>>>(reinterpret_cast<const T*>(residual.data_ptr()),
                          mixes.data_ptr<float>(), partials.data_ptr<float>(),
                          n, chunk);
    }
    d::mhc_pre_kernel<T>
        <<<static_cast<int>(num_tokens), d::kPreThreads, 0, stream>>>(
            reinterpret_cast<const T*>(residual.data_ptr()),
            given_fn ? nullptr : mixes.data_ptr<float>(),
            given_fn ? partials.data_ptr<float>() : nullptr, d::kPreSplits,
            hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
            post.data_ptr<float>(), comb.data_ptr<float>(),
            reinterpret_cast<T*>(layer_input.data_ptr()), hidden,
            static_cast<float>(rms_eps), static_cast<float>(pre_eps),
            static_cast<float>(sinkhorn_eps), static_cast<float>(post_mult),
            static_cast<int>(sinkhorn_repeat));
  };
  if (residual.scalar_type() == at::kHalf) {
    launch(__half{});
  } else {
    launch(__hip_bfloat16{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {post, comb, layer_input};
}

torch::Tensor dsv4_mhc_post_rdna(torch::Tensor const& x,         // [T, H]
                                 torch::Tensor const& residual,  // [T, HC, H]
                                 torch::Tensor const& post,      // [T, HC, 1]
                                 torch::Tensor const& comb) {    // [T, HC, HC]
  namespace d = vllm::rdna::dsv4;
  TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 &&
                  residual.size(1) == d::kHc && residual.is_contiguous(),
              "residual must be contiguous [T, 4, H]");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() && x.is_contiguous() &&
                  x.dim() == 2 && x.size(0) == residual.size(0) &&
                  x.size(1) == residual.size(2),
              "x must be contiguous [T, H] in the residual dtype");
  TORCH_CHECK(residual.scalar_type() == at::kHalf ||
                  residual.scalar_type() == at::kBFloat16,
              "residual must be fp16 or bf16");
  TORCH_CHECK(post.scalar_type() == at::kFloat && post.is_contiguous() &&
                  post.numel() == residual.size(0) * d::kHc,
              "post must be contiguous fp32 [T, 4(, 1)]");
  TORCH_CHECK(comb.scalar_type() == at::kFloat && comb.is_contiguous() &&
                  comb.numel() == residual.size(0) * d::kHc * d::kHc,
              "comb must be contiguous fp32 [T, 4, 4]");
  auto out = torch::empty_like(residual);
  const int64_t num_tokens = residual.size(0);
  const int hidden = static_cast<int>(residual.size(2));
  if (num_tokens == 0) return out;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(residual));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  dim3 grid((hidden + kThreads - 1) / kThreads, static_cast<int>(num_tokens));
  auto launch = [&](auto tag) {
    using T = decltype(tag);
    d::mhc_post_kernel<T><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(residual.data_ptr()), post.data_ptr<float>(),
        comb.data_ptr<float>(), reinterpret_cast<T*>(out.data_ptr()), hidden);
  };
  if (residual.scalar_type() == at::kHalf) {
    launch(__half{});
  } else {
    launch(__hip_bfloat16{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
