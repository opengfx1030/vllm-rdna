#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"
#include "qdq_fp8_rdna2.cuh"
#include "ops.h"

namespace vllm {
namespace w8a16_fp8_dense_rdna2 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256
constexpr int LDS_PAD = 8;

// FP8 dense GEMM for RDNA2 (gfx1030), fp16 activations.
//
// Tensor layout: b_q is [K, N] uint8 (row-major), one fp8 weight per byte.
// b_scales is [K/group_size, N] fp16, per-N-col scale per group.
//
// For each output position c[m, n+nn], we accumulate:
//   c[m, n+nn] = sum_k a[m, k] * (fp8(b_q[k, n+nn]) * b_scales[k/gs, n+nn])
//
// Thread geometry: THREADS_X=256 threads per block, each thread computes
// 4 N-cols (n, n+1, n+2, n+3). Block covers BLOCK_KN_SIZE K-positions.
// Each K-iter loads 8 K-positions per N-col (8 bytes from 8 different rows).

template <int M_TILE>
__global__ void gemm_w8a16_fp8_dense_kernel_rdna2(
    const half* __restrict__ a, const uint8_t* __restrict__ b_q,
    const half* __restrict__ b_scales, half* __restrict__ c, const int size_m,
    const int size_n, const int size_k, const int groups,
    const int group_size) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_TILE;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  __shared__ half block_a[M_TILE][BLOCK_KN_SIZE + LDS_PAD];

  if (offset_k + t < end_k) {
    for (int m = 0; m < M_TILE; ++m) {
      half av;
      if (offset_m + m < size_m) {
        av = a[(offset_m + m) * size_k + offset_k + t];
      } else {
        av = __float2half_rn(0.0f);
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  const int groupsize = (group_size > 0) ? group_size : size_k;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Scales carry the 2^8 of the branch-free fp8 decode below.
  const half k256 = __float2half_rn(256.0f);
  half s[4];
  {
    const half* sc_row = b_scales + group * size_n;
    s[0] = __hmul(sc_row[n + 0], k256);
    s[1] = __hmul(sc_row[n + 1], k256);
    s[2] = __hmul(sc_row[n + 2], k256);
    s[3] = __hmul(sc_row[n + 3], k256);
  }

  float block_c[M_TILE][4];
  for (int m = 0; m < M_TILE; ++m) {
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    if (k >= nextgroup) {
      group++;
      nextgroup += groupsize;
      const half* sc_row = b_scales + group * size_n;
      s[0] = __hmul(sc_row[n + 0], k256);
      s[1] = __hmul(sc_row[n + 1], k256);
      s[2] = __hmul(sc_row[n + 2], k256);
      s[3] = __hmul(sc_row[n + 3], k256);
    }

    // Load 8 K-rows as one dword each (the thread's 4 adjacent N-cols),
    // then regroup the bytes into one uint64 of 8 K-positions per N-col
    // (byte j = K-row j). Dword loads instead of 32 byte loads per step.
    uint32_t rows[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int kk = k + j;
      rows[j] = kk < end_k ? *reinterpret_cast<const uint32_t*>(
                                 b_q + (int64_t)kk * size_n + n)
                           : 0u;
    }
    // Dequantize once per step and reuse it for every row of the M tile.
    // Branch-free E4M3 -> fp16: placing the 7 magnitude bits at fp16 bit 7
    // gives exactly value * 2^-8 for normals and subnormals alike (the
    // exponent rebias 7 -> 15 is a power-of-two factor), so the 2^8 is
    // folded into the scale. The E4M3 NaN code never occurs in weights.
    half2 dq[4][4];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      const half2 sh = __halves2half2(s[nn], s[nn]);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t b0 = (rows[2 * p] >> (8 * nn)) & 0xFFu;
        const uint32_t b1 = (rows[2 * p + 1] >> (8 * nn)) & 0xFFu;
        const uint32_t h2 = ((b0 & 0x80u) << 8) | ((b0 & 0x7Fu) << 7) |
                            ((b1 & 0x80u) << 24) | ((b1 & 0x7Fu) << 23);
        dq[nn][p] = __hmul2(__builtin_bit_cast(half2, h2), sh);
      }
    }
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      const half* a_ptr = &block_a[m][k - offset_k];
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) {
        block_c[m][nn] += gptq_rdna2::dot22_8_f(dq[nn], a_ptr);
      }
    }
    k += 8;
  }

  for (int m = 0; m < M_TILE; ++m) {
    const int m_row = offset_m + m;
    if (m_row >= size_m) continue;
    half* c_row = c + m_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    gptq_rdna2::atomic_add_pk4_f16(c_row, r01, r23);
  }
}

void gemm_w8a16_fp8_dense(torch::Tensor a, torch::Tensor b_q,
                          torch::Tensor b_scales, torch::Tensor c,
                          int64_t group_size) {
  TORCH_CHECK(a.is_cuda() && b_q.is_cuda() && b_scales.is_cuda() &&
              c.is_cuda());
  TORCH_CHECK(a.dtype() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b_q.dtype() == torch::kUInt8, "b_q must be uint8 (FP8 bytes)");
  TORCH_CHECK(b_scales.dtype() == torch::kHalf, "b_scales must be fp16");
  TORCH_CHECK(c.dtype() == torch::kHalf, "c must be fp16");

  const int size_m = a.size(0);
  const int size_k = a.size(1);
  const int size_n = b_q.size(1);

  TORCH_CHECK(b_q.size(0) == size_k, "b_q first dim must be K");
  TORCH_CHECK(size_n % 4 == 0, "N must be multiple of 4");
  TORCH_CHECK(size_k % 8 == 0, "K must be multiple of 8");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  if (group_size > 0) {
    TORCH_CHECK(size_k % group_size == 0, "K must be multiple of group_size");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int groups = (group_size > 0) ? (size_k / group_size) : 1;
  const int gs = (int)group_size;

  dim3 block(THREADS_X);
  auto launch = [&](auto M_TILE_v) {
    constexpr int M_TILE = decltype(M_TILE_v)::value;
    dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
              (size_m + M_TILE - 1) / M_TILE,
              (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);
    gemm_w8a16_fp8_dense_kernel_rdna2<M_TILE><<<grid, block, 0, stream>>>(
        (const half*)a.data_ptr(), b_q.data_ptr<uint8_t>(),
        (const half*)b_scales.data_ptr(), (half*)c.data_ptr(), size_m, size_n,
        size_k, groups, gs);
  };

  if (size_m == 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (size_m <= 3) {
    launch(std::integral_constant<int, 2>{});
  } else if (size_m <= 7) {
    launch(std::integral_constant<int, 4>{});
  } else {
    launch(std::integral_constant<int, 8>{});
  }
}

}  // namespace w8a16_fp8_dense_rdna2
}  // namespace vllm

void gemm_w8a16_fp8_dense(torch::Tensor a, torch::Tensor b_q_weight,
                          torch::Tensor b_scales, torch::Tensor c,
                          int64_t group_size) {
  vllm::w8a16_fp8_dense_rdna2::gemm_w8a16_fp8_dense(a, b_q_weight, b_scales, c,
                                                    group_size);
}