// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8-FP8 dense GEMM kernel for AMD RDNA2 (gfx1030).
//
// FP8 (E4M3) weights + FP8 (E4M3) activations. FP8->fp16 conversion
// happens during LDS staging (one conversion per element, broadcast
// across the whole 256-lane wave), then v_dot2_f32_f16 reads the staged
// fp16 directly. This eliminates the 256x redundant per-k-window
// activation dequant that would otherwise occur if the conversions were
// in the inner loop (all lanes read the same 8 LDS bytes via broadcast).
//
// gfx1030 (RDNA2) has NO FP8 hardware support. fp8_e4m3_to_fp16_bits
// (qdq_fp8_rdna2.cuh) is an inline bit-trick -- no LUT, no constant
// memory, no cross-TU visibility issues.
//
// Tensor layout:
//   a_q       [M, K]         uint8 (FP8 E4M3 bytes)
//   a_scale   [M, K/gk]      fp16/fp32 per-(token, K-block) scale (v82)
//              [M]            fp16/fp32 per-row scale
//              [1]            fp16/fp32 per-tensor scale (expanded to [M] by
//                             host; numel() == 1 detects this)
//   b_q       [K, N]         uint8 (FP8 E4M3 bytes)
//   b_scales  [K/gs, N]      fp16 per-group weight scale
//   c         [M, N]         fp16 (pre-zeroed; kernel atomic-adds)
//
// gs = weight group_size (multiple of 8); gk = activation K-block (=
// number of K-groups in a_scale's last axis, = 1 for [M]/[M,1]/[1]).
// For DeepSeek V4 Flash attention with `dynamic` FP8 quant + block
// shape [128, 128], a_scale is [M, K/128] and gk = K/128.

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
namespace w8a8_fp8_dense_rdna2 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256
constexpr int LDS_PAD = 8;

template <int M_TILE, bool PER_CHANNEL_SCALE>
__global__ void gemm_w8a8_fp8_dense_kernel_rdna2(
    const uint8_t* __restrict__ a_q,
    const half* __restrict__ a_scale_h,
    const int a_scale_K_groups,
    const uint8_t* __restrict__ b_q,
    const half* __restrict__ b_scales,
    half* __restrict__ c,
    const int size_m, const int size_n, const int size_k,
    const int group_size) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_TILE;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // FP16 activations in LDS (dequant done at staging, not in inner loop).
  __shared__ half block_a[M_TILE][BLOCK_KN_SIZE + LDS_PAD];

  // Stage FP8 activations and convert to scaled FP16 in-register.
  // Per-block-K path (a_scale_K_groups > 1): the activation K-block size
  // matches the weight group_size, so each t belongs to K-block
  // (offset_k + t) / group_size. Index = m * K_groups + k_block.
  // Per-row path (a_scale_K_groups == 1): one scale per m, ignore t.
  if (offset_k + t < end_k) {
    const int k_block = (group_size > 0)
        ? ((offset_k + t) / group_size)
        : 0;
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      uint8_t av;
      if (offset_m + m < size_m) {
        av = a_q[(int64_t)(offset_m + m) * size_k + offset_k + t];
      } else {
        av = 0;
      }
      uint16_t bits = fp8_e4m3_to_fp16_bits(av);
      half h = __ushort_as_half(bits);
      half a_s;
      if (offset_m + m < size_m) {
        a_s = (a_scale_K_groups == 1)
            ? a_scale_h[offset_m + m]
            : a_scale_h[(offset_m + m) * a_scale_K_groups + k_block];
      } else {
        a_s = __float2half_rn(0.0f);
      }
      block_a[m][t] = __hmul(h, a_s);
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  const int groupsize = (group_size > 0) ? group_size : size_k;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  half s[4];
  {
    // Per-channel weight scale (fused-QKV): same scale for all K-groups.
    // Per-group weight scale: one scale per (K-group, n) pair.
    const half* sc_row = PER_CHANNEL_SCALE
        ? b_scales
        : (b_scales + group * size_n);
    s[0] = sc_row[n + 0];
    s[1] = sc_row[n + 1];
    s[2] = sc_row[n + 2];
    s[3] = sc_row[n + 3];
  }

  float block_c[M_TILE][4];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    if (k >= nextgroup) {
      group++;
      nextgroup += groupsize;
      const half* sc_row = b_scales + group * size_n;
      s[0] = sc_row[n + 0];
      s[1] = sc_row[n + 1];
      s[2] = sc_row[n + 2];
      s[3] = sc_row[n + 3];
    }

    // 4 N-cols per thread, 8 K-positions per k-window. Each row's 4
    // N-cols are 4 contiguous bytes; one uint32 load per row pulls them
    // all. 8 uint32 loads per k-window (vs 32 byte loads).
    uint32_t b_w[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int kk = k + j;
      b_w[j] = (kk < end_k) ? *(const uint32_t*)&b_q[(int64_t)kk * size_n + n]
                             : 0;
    }

    // Hoist weight dequant out of m-loop: same weight bytes for all m.
    // 4 N-cols -> 4 dq[] arrays, each holding 4 half2 (= 8 fp16 values).
    half2 dq[4][4];
    half2 zh = __float2half2_rn(0.0f);
    half2 sh[4];
    #pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      sh[nn] = __halves2half2(s[nn], s[nn]);
    }
    #pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      // Pack 8 bytes from rows 2*nn and 2*nn+1 of b_w[] into a uint64.
      // (4 N-cols share a row, so 8 K-positions = 8 different rows.)
      uint64_t packed = 0;
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint8_t byte = (uint8_t)(b_w[j] >> (8 * nn));
        packed |= ((uint64_t)byte) << (j * 8);
      }
      w8a16_fp8_rdna2::dequant_fp8_8_fp16(packed, sh[nn], zh, dq[nn]);
    }

    // Per m-row: dot the 4 N-cols against the 8 K-positions of a_ptr.
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      const half* a_ptr = &block_a[m][k - offset_k];
      block_c[m][0] += gptq_rdna2::dot22_8_f(dq[0], a_ptr);
      block_c[m][1] += gptq_rdna2::dot22_8_f(dq[1], a_ptr);
      block_c[m][2] += gptq_rdna2::dot22_8_f(dq[2], a_ptr);
      block_c[m][3] += gptq_rdna2::dot22_8_f(dq[3], a_ptr);
    }
    k += 8;
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    const int m_row = offset_m + m;
    if (m_row >= size_m) continue;
    half* c_row = c + (int64_t)m_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    gptq_rdna2::atomic_add_pk4_f16(c_row, r01, r23);
  }
}

void gemm_w8a8_fp8_dense(
    torch::Tensor a_q, torch::Tensor a_scale, torch::Tensor b_q,
    torch::Tensor b_scales, torch::Tensor c, int64_t group_size,
    int64_t a_scale_K_groups_param) {
  TORCH_CHECK(a_q.is_cuda() && a_scale.is_cuda() && b_q.is_cuda() &&
              b_scales.is_cuda() && c.is_cuda());
  TORCH_CHECK(a_q.dtype() == torch::kUInt8, "a_q must be uint8 (FP8 bytes)");
  TORCH_CHECK(b_q.dtype() == torch::kUInt8, "b_q must be uint8 (FP8 bytes)");
  TORCH_CHECK(b_scales.dtype() == torch::kHalf, "b_scales must be fp16");
  TORCH_CHECK(c.dtype() == torch::kHalf, "c must be fp16");
  TORCH_CHECK(a_scale.dtype() == torch::kHalf || a_scale.dtype() == torch::kFloat,
              "a_scale must be fp16 or fp32");

  const int size_m = (int)a_q.size(0);
  const int size_k = (int)a_q.size(1);
  const int size_n = (int)b_q.size(1);

  TORCH_CHECK(b_q.size(0) == size_k, "b_q first dim must be K");
  TORCH_CHECK(size_n % 4 == 0, "N must be multiple of 4 (64-bit CAS)");
  TORCH_CHECK(size_k % 8 == 0, "K must be multiple of 8");
  TORCH_CHECK(
      (b_scales.dim() == 1 && b_scales.size(0) == size_n) ||
      (b_scales.dim() == 2 && b_scales.size(1) == size_n),
      "b_scales must be [N] (per-channel, fused-QKV) or [K/gs, N] (per-group), got dim=",
      b_scales.dim(), " shape=", b_scales.sizes());
  if (group_size > 0) {
    TORCH_CHECK(size_k % group_size == 0,
                "K must be multiple of group_size");
    TORCH_CHECK(group_size % 8 == 0,
                "group_size must be a multiple of 8 (k-window stride)");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a_q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int gs = (int)group_size;

  // Normalize a_scale to fp16 contiguous; a_scale_K_groups is supplied
  // by the caller (the Python wrapper computes it from As.shape).
  torch::Tensor a_scale_h = a_scale;
  if (a_scale.dtype() == torch::kFloat) {
    a_scale_h = a_scale.to(torch::kHalf);
  }
  if (a_scale_h.numel() == 1) {
    a_scale_h = a_scale_h.expand({size_m}).contiguous();
  } else {
    a_scale_h = a_scale_h.contiguous();
    TORCH_CHECK(a_scale_h.size(0) == size_m,
                "a_scale first dim must be M (got ", a_scale_h.size(0),
                ", M=", size_m, ")");
    if (a_scale_h.dim() == 2 && a_scale_h.size(1) != 1) {
      // Per-block-K activation scale requires aligned K-block =
      // weight group_size. Skip the check when group_size=0 (degenerate,
      // unused path).
      if (gs > 0) {
        const int K_blocks_expected = size_k / gs;
        TORCH_CHECK(a_scale_K_groups_param == K_blocks_expected,
                    "a_scale [M, K_groups] last-dim must match K/group_size ",
                    "(got K_groups=", a_scale_K_groups_param,
                    ", expected K/gs=", K_blocks_expected, ")");
      }
    }
  }

  dim3 block(THREADS_X);
  // Per-channel weight scale (fused-QKV) detected by 1D b_scales shape [N].
  // Per-group weight scale is 2D [K/gs, N].
  const bool per_channel_scale = (b_scales.dim() == 1);
  auto launch = [&](auto M_TILE_v) {
    constexpr int M_TILE = decltype(M_TILE_v)::value;
    dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
              (size_m + M_TILE - 1) / M_TILE,
              (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);
    if (per_channel_scale) {
      gemm_w8a8_fp8_dense_kernel_rdna2<M_TILE, true><<<grid, block, 0, stream>>>(
          a_q.data_ptr<uint8_t>(), (const half*)a_scale_h.data_ptr(),
          (int)a_scale_K_groups_param,
          b_q.data_ptr<uint8_t>(), (const half*)b_scales.data_ptr(),
          (half*)c.data_ptr(), size_m, size_n, size_k, gs);
    } else {
      gemm_w8a8_fp8_dense_kernel_rdna2<M_TILE, false><<<grid, block, 0, stream>>>(
          a_q.data_ptr<uint8_t>(), (const half*)a_scale_h.data_ptr(),
          (int)a_scale_K_groups_param,
          b_q.data_ptr<uint8_t>(), (const half*)b_scales.data_ptr(),
          (half*)c.data_ptr(), size_m, size_n, size_k, gs);
    }
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

}  // namespace w8a8_fp8_dense_rdna2
}  // namespace vllm

void gemm_w8a8_fp8_dense(
    torch::Tensor a_q, torch::Tensor a_scale, torch::Tensor b_q,
    torch::Tensor b_scales, torch::Tensor c, int64_t group_size,
    int64_t a_scale_K_groups) {
  vllm::w8a8_fp8_dense_rdna2::gemm_w8a8_fp8_dense(
      a_q, a_scale, b_q, b_scales, c, group_size, a_scale_K_groups);
}
