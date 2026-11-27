// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "core/cuda_compat.h"

// ---------------------------------------------------------------------------
// MoE decode skinny GEMV (W4A16, symmetric, small M): one wave per output
// row, lanes stride along K (coalesced weight streaming), activations staged
// in LDS, silu*mul fused into the gate_up epilogue, topk-weighted reduction
// inside the down kernel (no atomics).
//
// Ported from leapdragon/vllm-rdna2-recipe patch 0008 (Aron Hsiao).
// Layout matches moe_wna16's prepared buffers (Triton WNA16 sequential
// packing), NOT shuffled RDNA2 fused weights [E, K/8, N]:
//   qweight k-sequential nibbles, value = (nibble - 8) * scale
//   (symmetric uint4b8), scales [E, rows, K/G].
// Do not call this on CompressedTensorsWNA16RDNA2MoEMethod weights.
// ---------------------------------------------------------------------------

// T46: topk ids arrive as the router produced them (int64 or int32) and are
// remapped through the EP expert_map in-kernel, so the Python hook launches no
// index/convert/copy kernels (they were 3 launches per MoE layer).
__device__ __forceinline__ int moe_local_expert(const void* ids, bool ids_i64,
                                                const int32_t* expert_map,
                                                int idx) {
  int e = ids_i64 ? (int)reinterpret_cast<const int64_t*>(ids)[idx]
                  : reinterpret_cast<const int32_t*>(ids)[idx];
  if (expert_map != nullptr && e >= 0) e = expert_map[e];
  return e;
}
__device__ __forceinline__ float moe_topk_w(const void* w, bool w_is_half,
                                            int idx) {
  return w_is_half ? __half2float(reinterpret_cast<const half*>(w)[idx])
                   : reinterpret_cast<const float*>(w)[idx];
}

template <int WAVES>
__global__ void moe_w13_silu_gemv_(
    const half* __restrict__ input, const uint32_t* __restrict__ w13,
    const half* __restrict__ s13, const void* __restrict__ topk_ids,
    const bool ids_i64, const int32_t* __restrict__ expert_map,
    half* __restrict__ act, const int K, const int N, const int topk,
    const int group_size) {
  const int m = blockIdx.z, s = blockIdx.y;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = input[m * K + i];
  __syncthreads();
  if (n >= N) return;
  const int expert =
      moe_local_expert(topk_ids, ids_i64, expert_map, m * topk + s);
  // Expert parallelism: ids already remapped to local indices; < 0 means the
  // expert lives on another rank — emit zero so the down kernel adds nothing.
  if (expert < 0) {
    if (lane == 0) act[((uint64_t)m * topk + s) * N + n] = __float2half(0.f);
    return;
  }
  const int K8 = K / 8, KG = K / group_size;
  const uint64_t base = (uint64_t)expert * 2 * N;
  const uint32_t* wg = w13 + (base + n) * K8;
  const uint32_t* wu = w13 + (base + N + n) * K8;
  const half* sg = s13 + (base + n) * KG;
  const half* su = s13 + (base + N + n) * KG;
  float accg = 0.f, accu = 0.f;
  for (int i = lane; i < K8; i += 32) {
    const uint32_t qg = wg[i], qu = wu[i];
    const int k0 = i * 8;
    float pg = 0.f, pu = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = __half2float(xs[k0 + j]);
      pg += (float)((int)((qg >> (4 * j)) & 0xF) - 8) * xv;
      pu += (float)((int)((qu >> (4 * j)) & 0xF) - 8) * xv;
    }
    const int g = k0 / group_size;
    accg += pg * __half2float(sg[g]);
    accu += pu * __half2float(su[g]);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    accg += __shfl_xor(accg, off);
    accu += __shfl_xor(accu, off);
  }
  if (lane == 0) {
    const float silu = accg / (1.f + __expf(-accg));
    act[((uint64_t)m * topk + s) * N + n] = __float2half(silu * accu);
  }
}

template <int WAVES>
__global__ void moe_w2_gemv_(
    const half* __restrict__ act, const uint32_t* __restrict__ w2,
    const half* __restrict__ s2, const void* __restrict__ topk_ids,
    const bool ids_i64, const int32_t* __restrict__ expert_map,
    const void* __restrict__ topk_w, const bool w_is_half,
    half* __restrict__ out, const int N, const int H, const int topk,
    const int group_size) {
  const int m = blockIdx.z;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = blockIdx.x * WAVES + wave;
  extern __shared__ half as[];
  for (int i = threadIdx.x; i < topk * N; i += blockDim.x)
    as[i] = act[(uint64_t)m * topk * N + i];
  __syncthreads();
  if (h >= H) return;
  const int N8 = N / 8, NG = N / group_size;
  float acc = 0.f;
  for (int s = 0; s < topk; s++) {
    const int expert =
        moe_local_expert(topk_ids, ids_i64, expert_map, m * topk + s);
    if (expert < 0) continue;  // non-local expert under EP
    const uint32_t* wrow = w2 + ((uint64_t)expert * H + h) * N8;
    const half* srow = s2 + ((uint64_t)expert * H + h) * NG;
    const half* xrow = as + s * N;
    float sacc = 0.f;
    for (int i = lane; i < N8; i += 32) {
      const uint32_t q = wrow[i];
      const int k0 = i * 8;
      float p = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
        p += (float)((int)((q >> (4 * j)) & 0xF) - 8) *
             __half2float(xrow[k0 + j]);
      sacc += p * __half2float(srow[k0 / group_size]);
    }
    acc += sacc * moe_topk_w(topk_w, w_is_half, m * topk + s);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
  if (lane == 0) out[(uint64_t)m * H + h] = __float2half(acc);
}

void moe_skinny_int4_decode(const at::Tensor& input, const at::Tensor& w13,
                            const at::Tensor& w13_scale, const at::Tensor& w2,
                            const at::Tensor& w2_scale,
                            const at::Tensor& topk_weights,
                            const at::Tensor& topk_ids, at::Tensor& act_buf,
                            at::Tensor& output, const int64_t group_size,
                            const std::optional<at::Tensor>& expert_map) {
  const int M = input.size(0);
  const int K = input.size(1);
  const int topk = topk_ids.size(1);
  const int N = act_buf.size(2);

  TORCH_CHECK(M >= 1 && M <= 16, "moe_skinny_int4_decode: M must be 1..16");
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be divisible by 8");
  TORCH_CHECK(K % group_size == 0 && N % group_size == 0,
              "group_size must divide K and N");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  w13_scale.scalar_type() == at::kHalf &&
                  w2_scale.scalar_type() == at::kHalf &&
                  act_buf.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf,
              "fp16 activations/scales only");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
      "topk_ids must be int32 or int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat ||
                  topk_weights.scalar_type() == at::kHalf,
              "topk_weights must be float32 or fp16");
  TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              "topk_ids/topk_weights must be contiguous");
  const int32_t* emap = nullptr;
  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->scalar_type() == at::kInt && expert_map->is_contiguous(),
        "expert_map must be contiguous int32");
    emap = reinterpret_cast<const int32_t*>(expert_map->const_data_ptr());
  }
  const bool ids_i64 = topk_ids.scalar_type() == at::kLong;
  const bool w_half = topk_weights.scalar_type() == at::kHalf;
  TORCH_CHECK(input.is_contiguous() && act_buf.is_contiguous() &&
                  output.is_contiguous(),
              "activations must be contiguous");
  const int64_t elem = w13.element_size();
  TORCH_CHECK(w13.numel() * elem == (int64_t)w13_scale.size(0) * 2 * N * K / 2,
              "w13 byte-size mismatch");
  TORCH_CHECK(w2.numel() * elem == (int64_t)w2_scale.size(0) * K * N / 2,
              "w2 byte-size mismatch");
  TORCH_CHECK(output.size(1) == K, "output width must equal hidden size");

  constexpr int WAVES = 8;
  dim3 block(WAVES * 32);
  dim3 grid1((N + WAVES - 1) / WAVES, topk, M);
  dim3 grid2((K + WAVES - 1) / WAVES, 1, M);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  moe_w13_silu_gemv_<WAVES><<<grid1, block, K * 2, stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
      reinterpret_cast<const half*>(w13_scale.const_data_ptr()),
      topk_ids.const_data_ptr(), ids_i64, emap,
      reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, N, topk,
      (int)group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  moe_w2_gemv_<WAVES><<<grid2, block, topk * N * 2, stream>>>(
      reinterpret_cast<const half*>(act_buf.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
      reinterpret_cast<const half*>(w2_scale.const_data_ptr()),
      topk_ids.const_data_ptr(), ids_i64, emap, topk_weights.const_data_ptr(),
      w_half, reinterpret_cast<half*>(output.mutable_data_ptr()), N, K, topk,
      (int)group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// fp16 skinny GEMM for gfx1030 decode (M <= 8 tokens): y[M,N] = x[M,K].w[N,K]^T
// (+ bias). Wave-per-output-row, lanes stride K with 16-byte loads, 4-deep
// unroll for memory-level parallelism, v_dot2_f32_f16 accumulate, wave32
// shuffle reduce. Same design as the int4 MoE GEMV above. vLLM's own skinny
// kernels (wvSplitK/LLMM1) are gated to gfx9/gfx11+ and their RDNA path
// gives wrong results when built for gfx10, so every dense projection had
// been falling to rocBLAS Tensile tiles at ~35% of bandwidth (T43).
// ---------------------------------------------------------------------------
// MT is the row count; with GUARD it is a capacity and rows >= M are skipped
// wave-uniformly (MTP verify batches, 8 < M <= 32).
template <int WAVES, int MT, bool GUARD = false>
__global__ void __launch_bounds__(WAVES * 32)
    gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
                    const half* __restrict__ bias, half* __restrict__ y,
                    const int N, const int K, const int M = MT) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int i = lane; i < K8; i += 32 * 4) {
    uint4 wq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      if (idx < K8) {
        const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
#pragma unroll
        for (int m = 0; m < MT; m++) {
          if (GUARD && m >= M) continue;
          const uint4 xq = xr[(size_t)m * K8 + idx];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
          acc[m] = a;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      y[(size_t)m * N + n] = __float2half(acc[m] + b);
    }
  }
}

template <int MT>
static void gemv_f16_rdna2_guarded_launch(const half* x, const half* w,
                                          const half* bias, half* y, int N,
                                          int K, int M, cudaStream_t s) {
  gemv_f16_rdna2_<8, MT, true>
      <<<(N + 7) / 8, 256, 0, s>>>(x, w, bias, y, N, K, M);
}

template <int MT>
static void gemv_f16_rdna2_launch(const half* x, const half* w,
                                  const half* bias, half* y, int N, int K,
                                  cudaStream_t s) {
  // Fewer waves per block for small N so the grid still covers all CUs.
  if (N >= 1152)
    gemv_f16_rdna2_<8, MT><<<(N + 7) / 8, 256, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 576)
    gemv_f16_rdna2_<4, MT><<<(N + 3) / 4, 128, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 288)
    gemv_f16_rdna2_<2, MT><<<(N + 1) / 2, 64, 0, s>>>(x, w, bias, y, N, K);
  else
    gemv_f16_rdna2_<1, MT><<<N, 32, 0, s>>>(x, w, bias, y, N, K);
}

at::Tensor gemv_f16_rdna2(const at::Tensor& x, const at::Tensor& w,
                          const std::optional<at::Tensor>& bias) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 32, "gemv_f16_rdna2: M must be 1..32");
  TORCH_CHECK(w.size(1) == K, "gemv_f16_rdna2: K mismatch");
  TORCH_CHECK(K % 8 == 0, "gemv_f16_rdna2: K % 8 == 0");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf,
              "gemv_f16_rdna2: fp16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "gemv_f16_rdna2: contiguous operands");
  const half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                    bias->numel() == N,
                "gemv_f16_rdna2: bias must be contiguous fp16 [N]");
    bp = reinterpret_cast<const half*>(bias->const_data_ptr());
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  const cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const half* wp = reinterpret_cast<const half*>(w.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  if (M > 8) {
    if (M <= 16)
      gemv_f16_rdna2_guarded_launch<16>(xp, wp, bp, yp, N, K, M, s);
    else if (M <= 24)
      gemv_f16_rdna2_guarded_launch<24>(xp, wp, bp, yp, N, K, M, s);
    else
      gemv_f16_rdna2_guarded_launch<32>(xp, wp, bp, yp, N, K, M, s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  switch (M) {
    case 1:
      gemv_f16_rdna2_launch<1>(xp, wp, bp, yp, N, K, s);
      break;
    case 2:
      gemv_f16_rdna2_launch<2>(xp, wp, bp, yp, N, K, s);
      break;
    case 3:
      gemv_f16_rdna2_launch<3>(xp, wp, bp, yp, N, K, s);
      break;
    case 4:
      gemv_f16_rdna2_launch<4>(xp, wp, bp, yp, N, K, s);
      break;
    case 5:
      gemv_f16_rdna2_launch<5>(xp, wp, bp, yp, N, K, s);
      break;
    case 6:
      gemv_f16_rdna2_launch<6>(xp, wp, bp, yp, N, K, s);
      break;
    case 7:
      gemv_f16_rdna2_launch<7>(xp, wp, bp, yp, N, K, s);
      break;
    default:
      gemv_f16_rdna2_launch<8>(xp, wp, bp, yp, N, K, s);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// int8 weight-only skinny GEMM for gfx1030 decode (M <= 8): y = x . (w_i8 *
// s)^T (+bias). Same wave-per-row design as gemv_f16_rdna2; a 16-byte load
// carries 16 weights, halving the streamed bytes of the fp16 dense projections
// (T45). Per-output-channel symmetric fp16 scale. lm_head/rank [62080x2560]:
// 640 us vs 1270 us fp16 in the harness.
// ---------------------------------------------------------------------------
__device__ __forceinline__ half2 rdna_i8x2_to_h2(int lo, int hi) {
  return __floats2half2_rn((float)lo, (float)hi);
}

template <int WAVES, int MT, bool GUARD = false>
__global__ void __launch_bounds__(WAVES * 32)
    gemv_i8_rdna2_(const half* __restrict__ x, const int8_t* __restrict__ w,
                   const half* __restrict__ scale,
                   const half* __restrict__ bias, half* __restrict__ y,
                   const int N, const int K, const int M = MT) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K16 = K / 16;
  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  constexpr int UNROLL = (MT > 8) ? 1 : 4;
  for (int i = lane; i < K16; i += 32 * UNROLL) {
    uint4 wq[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      const int idx = i + 32 * u;
      wq[u] = (idx < K16) ? wrow[idx] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      const int idx = i + 32 * u;
      if (idx < K16) {
        const int32_t* q = reinterpret_cast<const int32_t*>(&wq[u]);
        half2 wh[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int32_t v = q[j];
          wh[2 * j] =
              rdna_i8x2_to_h2((int8_t)(v & 0xff), (int8_t)((v >> 8) & 0xff));
          wh[2 * j + 1] = rdna_i8x2_to_h2((int8_t)((v >> 16) & 0xff),
                                          (int8_t)((v >> 24) & 0xff));
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
          if (GUARD && m >= M) continue;
          const uint4 xa = xr[((size_t)m * K16 + idx) * 2];
          const uint4 xb = xr[((size_t)m * K16 + idx) * 2 + 1];
          const half2* xh0 = reinterpret_cast<const half2*>(&xa);
          const half2* xh1 = reinterpret_cast<const half2*>(&xb);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh0[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh0[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh0[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh0[3], a, false);
          a = __builtin_amdgcn_fdot2(wh[4], xh1[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[5], xh1[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[6], xh1[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[7], xh1[3], a, false);
          acc[m] = a;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
    if (GUARD && m >= M) continue;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float sc = __half2float(scale[n]);
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      y[(size_t)m * N + n] = __float2half(acc[m] * sc + b);
    }
  }
}

template <int MT, bool GUARD = false>
static void gemv_i8_rdna2_launch(const half* x, const int8_t* w, const half* s,
                                 const half* bias, half* y, int N, int K, int M,
                                 cudaStream_t st) {
  if (N >= 1152)
    gemv_i8_rdna2_<8, MT, GUARD>
        <<<(N + 7) / 8, 256, 0, st>>>(x, w, s, bias, y, N, K, M);
  else if (N >= 576)
    gemv_i8_rdna2_<4, MT, GUARD>
        <<<(N + 3) / 4, 128, 0, st>>>(x, w, s, bias, y, N, K, M);
  else if (N >= 288)
    gemv_i8_rdna2_<2, MT, GUARD>
        <<<(N + 1) / 2, 64, 0, st>>>(x, w, s, bias, y, N, K, M);
  else
    gemv_i8_rdna2_<1, MT, GUARD><<<N, 32, 0, st>>>(x, w, s, bias, y, N, K, M);
}

at::Tensor gemv_i8_rdna2(const at::Tensor& x, const at::Tensor& w,
                         const at::Tensor& scale,
                         const std::optional<at::Tensor>& bias) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 32, "gemv_i8_rdna2: M must be 1..32");
  TORCH_CHECK(w.size(1) == K && K % 16 == 0,
              "gemv_i8_rdna2: K mismatch / K % 16");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kChar &&
                  scale.scalar_type() == at::kHalf && scale.numel() == N,
              "gemv_i8_rdna2: x fp16, w int8, scale fp16[N]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(),
              "gemv_i8_rdna2: contiguous operands");
  const half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                    bias->numel() == N,
                "gemv_i8_rdna2: bias fp16[N]");
    bp = reinterpret_cast<const half*>(bias->const_data_ptr());
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const int8_t* wp = reinterpret_cast<const int8_t*>(w.const_data_ptr());
  const half* sp = reinterpret_cast<const half*>(scale.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  switch (M) {
    case 1:
      gemv_i8_rdna2_launch<1>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 2:
      gemv_i8_rdna2_launch<2>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 3:
      gemv_i8_rdna2_launch<3>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 4:
      gemv_i8_rdna2_launch<4>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 5:
      gemv_i8_rdna2_launch<5>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 6:
      gemv_i8_rdna2_launch<6>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 7:
      gemv_i8_rdna2_launch<7>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    case 8:
      gemv_i8_rdna2_launch<8>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
    default:
      if (M <= 12)
        gemv_i8_rdna2_launch<12, true>(xp, wp, sp, bp, yp, N, K, M, st);
      else if (M <= 16)
        gemv_i8_rdna2_launch<16, true>(xp, wp, sp, bp, yp, N, K, M, st);
      else if (M <= 24)
        gemv_i8_rdna2_launch<24, true>(xp, wp, sp, bp, yp, N, K, M, st);
      else
        gemv_i8_rdna2_launch<32, true>(xp, wp, sp, bp, yp, N, K, M, st);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
