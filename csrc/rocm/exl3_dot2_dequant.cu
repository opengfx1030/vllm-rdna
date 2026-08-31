// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 6bpw dequant kernel for RDNA2/RDNA3 (gfx1030/gfx1100), fp16-only.
// One-time dequant at model load time. Decodes the mul1-codebook 6bpw
// trellis to a raw fp16 weight matrix (no Hadamard folding — caller
// applies suh/svh via PyTorch on GPU).
//
// Tile layout: 16x16 tile = 256 weights = 256*6 bits = 1536 bits
//   = 48 uint32 words = 96 int16 words per tile.
//   trellis: (k/16, n/16, 96) int16
//
// Window extraction uses dq4<6, cb=2> (port of exllamav3 dq4<bits>).
// For each (r, c):
//   p = r * 16 + c             (row-major window index)
//   t_offset = (p / 4) * 4     (dq4 batch start)
//   shift = s2 + 6 * (3 - (p % 4))
//   win = (fshift(tile[i2], tile[i0], shift)) & 0xFFFF
//   decoded = decode_3inst<cb=2>(win)  // mul1 decode

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "exl3_dot2_common.cuh"

#if defined(__HIPCC__) && (defined(__gfx1030__) || defined(__gfx1100__))
  #define __HIP__RDNA__
#endif

#define TILE_I16_6 96

namespace vllm {
namespace exl3_dot2 {

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

__global__ void dequant_bits6_tile_kernel(
    const int16_t* __restrict__ trellis,
    half* __restrict__ out,
    int K_tile, int N_tile, int N) {
  const int r = threadIdx.x;
  const int c = threadIdx.y;
  const int tile_idx = blockIdx.x;
  const int k_tile = tile_idx / N_tile;
  const int n_tile = tile_idx % N_tile;
  if (k_tile >= K_tile || n_tile >= N_tile) return;

  const uint32_t* tile = reinterpret_cast<const uint32_t*>(
      trellis + (k_tile * N_tile + n_tile) * TILE_I16_6);

  const int p = r * 16 + c;
  const int t_offset = (p / 4) * 4;
  const int idx_in_batch = p % 4;

  const int b0 = (t_offset + 257) * 6 - 16;
  const int b1 = b0 + 3 * 6;
  const int b2 = b1 + 16;
  const int i0 = b0 / 32;
  const int i2 = (b2 - 1) / 32;
  const int s2 = (i2 + 1) * 32 - b2;

  const uint32_t a = tile[i0 % 48];
  const uint32_t b = tile[i2 % 48];
  int shift;
  switch (idx_in_batch) {
    case 0: shift = s2 + 18; break;
    case 1: shift = s2 + 12; break;
    case 2: shift = s2 + 6; break;
    default: shift = s2; break;
  }
  const uint32_t win = fshift(b, a, shift) & 0xFFFFu;
  const half decoded = decode_3inst<2>(win);

  out[(k_tile * 16 + r) * N + (n_tile * 16 + c)] = decoded;
}

#else

__global__ void dequant_bits6_tile_kernel(const int16_t*, half*, int, int, int) {}

#endif

void exl3_dequant_bits6_mul1(torch::Tensor trellis, torch::Tensor out) {
  TORCH_CHECK(trellis.is_cuda() && out.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(trellis.dim() == 3 && trellis.size(2) == TILE_I16_6,
              "trellis must be (K/16, N/16, 96)");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(),
              "out must be 2D contiguous");

  const int K_tile = trellis.size(0);
  const int N_tile = trellis.size(1);
  const int K = K_tile * 16;
  const int N = N_tile * 16;
  TORCH_CHECK(out.size(0) == K && out.size(1) == N,
              "out size mismatch");

  const at::cuda::OptionalCUDAGuard dg(device_of(trellis));
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(K_tile * N_tile);
  dim3 block(16, 16);
  dequant_bits6_tile_kernel<<<grid, block, 0, stream>>>(
      (const int16_t*)trellis.data_ptr(), (half*)out.data_ptr(),
      K_tile, N_tile, N);
}

}  // namespace exl3_dot2
}  // namespace vllm
