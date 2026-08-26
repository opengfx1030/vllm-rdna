// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (QTIP bitshift trellis) dense GEMM for RDNA2/RDNA3 (gfx1030/gfx1100),
// fp16-only. REAL tile layout (locked 2026-08-26 against exllamav3_ext
// reconstruct, max-err 0.0 on 112 real + 32 synthetic tiles):
//
//   trellis: (k/16, n/16, 256*bits/16) int16, packed tail-biting stream.
//   Window p (16-bit codebook input) sits at bit (p+1)*bits - 16 mod 256*bits.
//   Weight value = decode_3inst<cb>(window_p).
//   Tile (row r, col c): window position p = (c%8)<<5 | (off(r) mod 32)
//     K=3: off(r) = 8*(r/2) + (r%2) + 2*(r>=8) + 4*(c/8)
//     K=4: off(r) = 8*(r/2) + sel(r) - 4*(c/8), sel = 7,6,5,4 by parity/half
//   suh (k,) / svh (n,) half: caller-side scales + Hadamard flips; NOT in
//   the K-dot (wiki kernels/exl3.md).
//
// Decode-skinny layout (gfx1030 has no WMMA): one thread owns one N-column
// (256 N-cols/block); block iterates k-tiles. Per k-tile: decode 16 windows
// (16 K-rows) in VGPR, then M_COUNT x 8 V_DOT2_F32_F16.
//
// Helpers from exl3_dot2_common.cuh: decode_3inst, exl3_window_pos,
// exl3_window_at.

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

#define THREADS_X 256
#define BLOCK_N_COLS 256
#define K_TILE 16

namespace vllm {
namespace exl3_dot2 {

__forceinline__ __device__ int m_global(int m, int bz, int mc) {
  return bz * mc + m;
}

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

template <int M_COUNT, int bits, int cb>
__global__ void gemm_exl3_tile_kernel_rdna(
    const half* __restrict__ a,          // [size_m, size_k]
    const int16_t* __restrict__ trellis, // [k/16, n/16, 256*bits/16]
    half* __restrict__ c,                // [size_m, size_n]
    const int size_m, const int size_n, const int size_k) {
  constexpr int TILE_WORDS = 8 * bits;     // uint32 per 16x16 tile
  constexpr int TILE_I16 = 2 * TILE_WORDS; // int16 per 16x16 tile
  constexpr int TILES_PER_BLOCK = BLOCK_N_COLS / 16;  // 16
  constexpr int N_K_TILES = 0;  // replaced below (host passes size_k/16)

  const int t = threadIdx.x;
  const int n = blockIdx.x * BLOCK_N_COLS + t;
  const int n_tiles = size_n / 16;
  const int n_tile = t / 16;
  const int c_col = t % 16;
  const int m_blk = blockIdx.z;
  const int n_k_tiles = size_k / K_TILE;

  __shared__ half s_a[M_COUNT][K_TILE];
  __shared__ uint32_t s_tile[TILES_PER_BLOCK][TILE_WORDS];

  float acc[M_COUNT];
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) acc[m] = 0.0f;

  for (int k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
    // Stage trellis for this block's 16 n-tiles at k_tile.
    {
      constexpr int TOTAL_U32 = TILES_PER_BLOCK * TILE_WORDS;
#pragma unroll
      for (int i = t; i < TOTAL_U32; i += THREADS_X) {
        const int tt = i / TILE_WORDS;
        const int wi = i % TILE_WORDS;
        const int16_t* src = trellis +
            ((int64_t)k_tile * n_tiles + blockIdx.x * TILES_PER_BLOCK + tt) *
                TILE_I16;
        s_tile[tt][wi] = reinterpret_cast<const uint32_t*>(src)[wi];
      }
    }

    // Stage A rows [m0..m0+M) x [kt*16, kt*16+16).
    {
#pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        const int mr = m_global(m, m_blk, M_COUNT);
        half av = (mr < size_m)
                      ? a[(int64_t)mr * size_k + k_tile * K_TILE +
                          (t % K_TILE)]
                      : __float2half_rn(0.0f);
        s_a[m][t % K_TILE] = av;
      }
    }
    __syncthreads();

    if (n < size_n) {
      // Decode all 16 windows for this column's (r, c_col).
      half w[K_TILE];
#pragma unroll
      for (int r = 0; r < K_TILE; ++r) {
        const int p = exl3_window_pos<bits>(r, c_col);
        const uint32_t win = exl3_window_at<bits>(s_tile[n_tile], p);
        w[r] = decode_3inst<cb>(win);
      }

      // Accumulate: acc[m] += sum_r A[m, r] * W[r, n] via 8 fdot2.
#pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        const half* a_row = s_a[m];
#pragma unroll
        for (int i = 0; i < K_TILE / 2; ++i) {
          half2 w2 = __halves2half2(w[2 * i], w[2 * i + 1]);
          half2 a2 = __halves2half2(a_row[2 * i], a_row[2 * i + 1]);
          acc[m] = __builtin_amdgcn_fdot2(w2, a2, acc[m], false);
        }
      }
    }
    __syncthreads();  // s_tile reuse next iteration
  }

  if (n >= size_n) return;
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    const int mr = m_global(m, m_blk, M_COUNT);
    if (mr >= size_m) continue;
    c[(int64_t)mr * size_n + n] = __float2half_rn(acc[m]);
  }
}

#else  // non-RDNA: empty stub for symbol parity

template <int M_COUNT, int bits, int cb>
__global__ void gemm_exl3_tile_kernel_rdna(const half*, const int16_t*, half*,
                                           const int, const int, const int) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

__forceinline__ int divide_up(int x, int y) { return (x + y - 1) / y; }

// ---------------------------------------------------------------------------
// Dispatch: M_COUNT -> bits -> cb
// ---------------------------------------------------------------------------

template <int M_COUNT, int bits, int cb>
void launch_tile_mcb(const half* a, const int16_t* trellis, half* c, int sm,
                     int sn, int sk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(divide_up(sn, BLOCK_N_COLS), 1, divide_up(sm, M_COUNT));
  gemm_exl3_tile_kernel_rdna<M_COUNT, bits, cb>
      <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
}

template <int M_COUNT, int bits>
void launch_tile_mb(const half* a, const int16_t* trellis, half* c, int sm,
                    int sn, int sk, int cb, cudaStream_t stream) {
  if (cb == 0)
    launch_tile_mcb<M_COUNT, bits, 0>(a, trellis, c, sm, sn, sk, stream);
  else if (cb == 1)
    launch_tile_mcb<M_COUNT, bits, 1>(a, trellis, c, sm, sn, sk, stream);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_COUNT>
void launch_tile_m(const half* a, const int16_t* trellis, half* c, int sm,
                   int sn, int sk, int bits, int cb, cudaStream_t stream) {
  switch (bits) {
    case 2:
      launch_tile_mb<M_COUNT, 2>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 3:
      launch_tile_mb<M_COUNT, 3>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    case 4:
      launch_tile_mb<M_COUNT, 4>(a, trellis, c, sm, sn, sk, cb, stream);
      break;
    default:
      TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported bits=", bits);
  }
}

void launch_tile(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                 int sk, int bits, int cb, cudaStream_t stream) {
  if (sm == 1)
    launch_tile_m<1>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 3)
    launch_tile_m<2>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 7)
    launch_tile_m<4>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else
    launch_tile_m<8>(a, trellis, c, sm, sn, sk, bits, cb, stream);
}

}  // namespace exl3_dot2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
// Inputs:
//   a        [M, K]       half
//   trellis  [K/16, N/16, 256*bits/16] int16 (real EXL3 tile layout)
//   size_m, size_n, size_k, bits, cb
// Output:
//   c        [M, N]       half

void exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
                     int64_t size_m, int64_t size_n, int64_t size_k,
                     int64_t bits, int64_t cb) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(trellis.is_cuda(), "trellis must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(trellis.dim() == 3,
              "trellis must be 3D [K/16, N/16, 256*bits/16]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "exl3_gemm_rdna2 only supports fp16");
  TORCH_CHECK(size_k % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(size_n % 16 == 0, "N must be a multiple of 16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  vllm::exl3_dot2::launch_tile(
      (const half*)a.data_ptr(), (const int16_t*)trellis.data_ptr(),
      (half*)c.data_ptr(), (int)size_m, (int)size_n, (int)size_k, (int)bits,
      (int)cb, stream);
}