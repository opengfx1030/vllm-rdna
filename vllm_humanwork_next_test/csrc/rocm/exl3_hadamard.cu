// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 Hadamard-128 transform (RDNA2/RDNA3), fp16 + fp32.
//
// Faithful port of exllamav3 quant/hadamard.cu had_r_128 (+ had_hf_r_128 /
// had_ff_r_128 kernels and had_hf_r_128_inner / had_ff_r_128_inner from
// hadamard_inner.cuh). Verified against exllamav3_ext.had_r_128.
//
// Semantics: y = H_128(x) * (scale/sqrt(128)), optionally pre-scaled by
// `pre_scale` (=suh on the A side) or post-scaled by `post_scale` (=svh on
// the C side). H_128 is the 128-wide Sylvester Hadamard (fp32 butterfly
// across one warp via __shfl_xor_sync), 1/sqrt(128) normalization folded
// into r_scale.
//
// NOT in the K-dot (wiki kernels/exl3.md): the EXL3 GEMM computes the raw
// trellis decode xh @ W_hat; suh/svh Hadamards are applied outside.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace exl3_dot2 {

// --- inner butterfly (port of hadamard_inner.cuh) ---

__forceinline__ __device__ void shuffle_had_f4x32(float& h0, float& h1,
                                                  float& h2, float& h3,
                                                  const int lane_id) {
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint32_t i0 = __float_as_uint(h0);
    uint32_t i1 = __float_as_uint(h1);
    uint32_t i2 = __float_as_uint(h2);
    uint32_t i3 = __float_as_uint(h3);
    uint64_t h01 = (uint64_t)i0 | (((uint64_t)i1) << 32);
    uint64_t h23 = (uint64_t)i2 | (((uint64_t)i3) << 32);
    uint64_t ph01 = __shfl_xor_sync(0xffffffffull, h01, i);
    uint64_t ph23 = __shfl_xor_sync(0xffffffffull, h23, i);
    float ph0 = __uint_as_float((uint32_t)(ph01 & 0xffffffff));
    float ph1 = __uint_as_float((uint32_t)(ph01 >> 32));
    float ph2 = __uint_as_float((uint32_t)(ph23 & 0xffffffff));
    float ph3 = __uint_as_float((uint32_t)(ph23 >> 32));
    int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
    i0 ^= sfm & 0x80000000;
    i1 ^= sfm & 0x80000000;
    i2 ^= sfm & 0x80000000;
    i3 ^= sfm & 0x80000000;
    h0 = __uint_as_float(i0) + ph0;
    h1 = __uint_as_float(i1) + ph1;
    h2 = __uint_as_float(i2) + ph2;
    h3 = __uint_as_float(i3) + ph3;
  }
}

__forceinline__ __device__ void shuffle_had_f2x32(float& v, float& w,
                                                  const int lane_id) {
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint64_t vw = ((uint64_t)__float_as_uint(v)) |
                  (((uint64_t)__float_as_uint(w)) << 32);
    uint64_t pvw = __shfl_xor_sync(0xffffffffull, vw, i);
    float pv = __uint_as_float((uint32_t)(pvw & 0xffffffff));
    float pw = __uint_as_float((uint32_t)(pvw >> 32));
    uint32_t vi = __float_as_uint(v);
    uint32_t wi = __float_as_uint(w);
    int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
    vi ^= (sfm & 0x80000000);
    wi ^= (sfm & 0x80000000);
    v = __uint_as_float(vi) + pv;
    w = __uint_as_float(wi) + pw;
  }
}

// Half vector, half scales (port of had_hf_r_128_inner).
template <bool pre_scale, bool post_scale>
__forceinline__ __device__ void had_hf_r_128_inner(
    const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
    const half* __restrict__ scale, const float r_scale) {
  const int t = threadIdx.x & 31;

  // half4 emulation via two half2s: elements (0,1) and (2,3).
  const half2* in2 = reinterpret_cast<const half2*>(input_ptr);
  half2 v0h = in2[2 * t];
  half2 v1h = in2[2 * t + 1];

  if constexpr (pre_scale) {
    const int i = blockIdx.y * 32 + t;  // half4 index = 2 half2 indices
    const half2* s2 = reinterpret_cast<const half2*>(scale);
    v0h = __hmul2(v0h, s2[2 * i]);
    v1h = __hmul2(v1h, s2[2 * i + 1]);
  }

  float v0 = __half2float(__low2half(v0h));
  float v1 = __half2float(__high2half(v0h));
  float v2 = __half2float(__low2half(v1h));
  float v3 = __half2float(__high2half(v1h));
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;

  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v0h = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  v1h = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

  if constexpr (post_scale) {
    const int i = blockIdx.y * 32 + t;
    const half2* s2 = reinterpret_cast<const half2*>(scale);
    v0h = __hmul2(v0h, s2[2 * i]);
    v1h = __hmul2(v1h, s2[2 * i + 1]);
  }

  half2* out2 = reinterpret_cast<half2*>(output_ptr);
  out2[2 * t] = v0h;
  out2[2 * t + 1] = v1h;
}

// Float vector, half scales (port of had_ff_r_128_inner).
template <bool pre_scale, bool post_scale>
__forceinline__ __device__ void had_ff_r_128_inner(
    const float* __restrict__ input_ptr, float* __restrict__ output_ptr,
    const half* __restrict__ scale, const float r_scale) {
  const int t = threadIdx.x & 31;

  float4 v = reinterpret_cast<const float4*>(input_ptr)[t];

  if constexpr (pre_scale) {
    const int i = blockIdx.y * 32 + t;
    const half2* s2 = reinterpret_cast<const half2*>(scale);
    half2 sA = s2[2 * i];
    half2 sB = s2[2 * i + 1];
    v.x *= __low2float(sA);
    v.y *= __high2float(sA);
    v.z *= __low2float(sB);
    v.w *= __high2float(sB);
  }

  float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  v.x = s0 + s1;
  v.y = d0 + d1;
  v.z = s0 - s1;
  v.w = d0 - d1;

  shuffle_had_f2x32(v.x, v.y, t);
  shuffle_had_f2x32(v.z, v.w, t);
  v.x *= r_scale;
  v.y *= r_scale;
  v.z *= r_scale;
  v.w *= r_scale;

  if constexpr (post_scale) {
    const int i = blockIdx.y * 32 + t;
    const half2* s2 = reinterpret_cast<const half2*>(scale);
    half2 sA = s2[2 * i];
    half2 sB = s2[2 * i + 1];
    v.x *= __low2float(sA);
    v.y *= __high2float(sA);
    v.z *= __low2float(sB);
    v.w *= __high2float(sB);
  }

  reinterpret_cast<float4*>(output_ptr)[t] = v;
}

// --- kernels (port of had_hf_r_128_kernel / had_ff_r_128_kernel) ---

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_hf_r_128_kernel(
    const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
    const half* __restrict__ scale, const float r_scale) {
  input_ptr += (size_t)gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
  output_ptr += (size_t)gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
  had_hf_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale,
                                            r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_ff_r_128_kernel(
    const float* __restrict__ input_ptr, float* __restrict__ output_ptr,
    const half* __restrict__ scale, const float r_scale) {
  input_ptr += (size_t)gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
  output_ptr += (size_t)gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
  had_ff_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale,
                                            r_scale);
}

}  // namespace exl3_dot2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point: y = H_128(x) * (scale/sqrt(128)), pre/post scaled.
// input/output [rows, cols] (cols % 128 == 0); in-place allowed.
// pre_scale/post_scale: [rows*cols/128]?? NO - upstream uses (gridDim.y*32+t)
//   = blockIdx.y*32+lane within a row's 128-block => scale length = rows*32
//   half4s = (cols/128)*32 halves per row = cols/4 halves = same length as a
//   row of the (rows, cols/4) view. In practice suh/svh are [cols] (the full
//   k or n dim) for EXL3 layers.
// ---------------------------------------------------------------------------

void exl3_hadamard_128(torch::Tensor input, torch::Tensor output,
                       torch::optional<torch::Tensor> pre_scale,
                       torch::optional<torch::Tensor> post_scale,
                       double scale) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA/HIP tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input/output dtype must match");
  TORCH_CHECK(input.size(1) % 128 == 0,
              "dim-1 (the transformed axis) must be divisible by 128");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int rows = (int)input.size(0);
  const int cols = (int)input.size(1);
  const int blocks = cols / 128;
  if ((rows > 1024 || cols > 16384)
      && std::getenv("VLLM_EXL3_HADAMARD_DBG") != nullptr) {
    fprintf(stderr, "[exl3_dbg] exl3_hadamard_128 rows=%d cols=%d blocks=%d "
            "input.device=%d input.data_ptr=%p post_scale=%p "
            "template=<%d,%d>\n",
            rows, cols, blocks,
            (int)input.is_cuda(), (void*)input.data_ptr(),
            (void*)(post_scale.has_value() ? post_scale->data_ptr() : nullptr),
            pre_scale.has_value() ? 1 : 0,
            post_scale.has_value() ? 1 : 0);
    fflush(stderr);
  }
  const float r_scale = (float)scale * 0.088388347648f;  // scale / sqrt(128)
  dim3 blockDim(32);
  dim3 gridDim(rows, blocks);

  const half* pre_p =
      pre_scale.has_value() ? (const half*)pre_scale->data_ptr() : nullptr;
  const half* post_p =
      post_scale.has_value() ? (const half*)post_scale->data_ptr() : nullptr;
  TORCH_CHECK(!(pre_scale && post_scale),
              "pre/post scale are mutually exclusive (upstream semantics)");

  if (input.scalar_type() == torch::kHalf) {
    TORCH_CHECK(output.scalar_type() == torch::kHalf, "output must be half");
    if (pre_scale.has_value())
      vllm::exl3_dot2::had_hf_r_128_kernel<true, false>
          <<<gridDim, blockDim, 0, stream>>>(
              (const half*)input.data_ptr(), (half*)output.data_ptr(), pre_p,
              r_scale);
    else if (post_scale.has_value())
      vllm::exl3_dot2::had_hf_r_128_kernel<false, true>
          <<<gridDim, blockDim, 0, stream>>>(
              (const half*)input.data_ptr(), (half*)output.data_ptr(), post_p,
              r_scale);
    else
      vllm::exl3_dot2::had_hf_r_128_kernel<false, false>
          <<<gridDim, blockDim, 0, stream>>>(
              (const half*)input.data_ptr(), (half*)output.data_ptr(), nullptr,
              r_scale);
  } else if (input.scalar_type() == torch::kFloat) {
    if (pre_scale.has_value())
      vllm::exl3_dot2::had_ff_r_128_kernel<true, false>
          <<<gridDim, blockDim, 0, stream>>>(
              (const float*)input.data_ptr(), (float*)output.data_ptr(), pre_p,
              r_scale);
    else if (post_scale.has_value())
      vllm::exl3_dot2::had_ff_r_128_kernel<false, true>
          <<<gridDim, blockDim, 0, stream>>>(
              (const float*)input.data_ptr(), (float*)output.data_ptr(), post_p,
              r_scale);
    else
      vllm::exl3_dot2::had_ff_r_128_kernel<false, false>
          <<<gridDim, blockDim, 0, stream>>>(
              (const float*)input.data_ptr(), (float*)output.data_ptr(), nullptr,
              r_scale);
  } else {
    TORCH_CHECK(false, "exl3_hadamard_128: unsupported dtype");
  }
}