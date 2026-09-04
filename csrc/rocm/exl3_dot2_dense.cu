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
// Kernel geometry follows the RDNA2 W4A16 dense kernel (q_gemm_rdna2.cu):
//   THREADS_X=256, 4 N-columns per thread => 1024 N-cols per block, 8 waves
//   wave32. A is staged once into LDS for BLOCK_KN (256) K-elements per
//   block; the K-loop iterates 32 K at a time (4 x 8-col sub-tiles), and
//   B/decode is register-resident per thread with 4x fdot2 ILP. No barrier
//   inside the K loop.
//
// Helpers from exl3_dot2_common.cuh: decode_3inst, exl3_window_pos,
// exl3_window_at, atomic_add_pk4_f16.

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
#define BLOCK_N 1024           // N-cols per block (4 per thread)
#define BLOCK_K 256            // K-elements staged per block
#define K_TILE 16              // EXL3 tile depth
#define COL_PER_THREAD 4

namespace vllm {
namespace exl3_dot2 {

__forceinline__ __device__ int m_global(int m, int bz, int mc) {
  return bz * mc + m;
}

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

template <int M_PER, int bits, int cb>
__global__ void gemm_exl3_kernel_rdna(
    const half* __restrict__ a,          // [size_m, size_k]
    const int16_t* __restrict__ trellis, // [k/16, n/16, 256*bits/16]
    half* __restrict__ c,                // [size_m, size_n]
    const int size_m, const int size_n, const int size_k) {
  constexpr int TILE_WORDS = 8 * bits;     // uint32 per 16x16 tile
  constexpr int TILE_I16 = 2 * TILE_WORDS;   // int16 per tile
  const int t = threadIdx.x;
  const int n0 = blockIdx.x * BLOCK_N + t * COL_PER_THREAD;
  const int n_tiles_total = size_n / 16;
  const int offset_k = blockIdx.y * BLOCK_K;
  const int end_k = min(offset_k + BLOCK_K, size_k);

  // A staging: [M_PER][BLOCK_K] halves in LDS, PAD for bank conflicts.
  constexpr int LDS_PAD = 8;
  __shared__ half s_a[M_PER][BLOCK_K + LDS_PAD];

  // Each thread loads 4 fp16 per M row (BLOCK_K/THREADS_X = 1 K each).
  static_assert(BLOCK_K == THREADS_X, "one K-elem per thread");
  for (int m = 0; m < M_PER; ++m) {
    const int mr = m_global(m, blockIdx.z, M_PER);
    half av = (mr < size_m) ? a[(int64_t)mr * size_k + offset_k + t]
                            : __float2half_rn(0.0f);
    s_a[m][t] = av;
  }
  __syncthreads();

  if (n0 >= size_n) return;

  // Per-thread accumulators: M_PER x 4 N-cols.
  float acc[M_PER][4];
#pragma unroll
  for (int m = 0; m < M_PER; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  // n-tile(s) touched by this thread's 4 columns (cols n0..n0+3 may span
  // one or two 16-col tiles).
  const int tile_idx0 = (n0) / 16;
  const int tile_idx1 = (n0 + 3) / 16;
  const bool two_tiles = tile_idx1 != tile_idx0;

  // K-loop: each iteration decodes a 16x(4 cols) slice in the codebook
  // domain and does M_PER*4 dot products. No barrier inside.
for (int k_tile = 0; k_tile < (end_k - offset_k) / K_TILE; ++k_tile) {
    const int16_t* tile0 = trellis +
        ((int64_t)(offset_k/K_TILE + k_tile) * n_tiles_total + tile_idx0)
            * TILE_I16;
    const int16_t* tile1 = two_tiles ? tile0 + TILE_I16 : nullptr;

    // Decode 4 cols x 16 K deltas.
    half w0[4][16], w1[4][16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int n_here = n0 + j;
      int nt = (n_here / 16) - tile_idx0;   // 0 or 1
      int ccol = n_here % 16;
      const int16_t* tp = nt ? tile1 : tile0;
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        const int p = exl3_window_pos<bits>(r, ccol);
        const uint32_t win = exl3_window_at<bits>(
            reinterpret_cast<const uint32_t*>(tp), p);
        (nt ? w1[j] : w0[j])[r] = decode_3inst<cb>(win);
      }
    }

    // Accumulate M rows x 4 cols via 8 fdot2 each.
#pragma unroll
    for (int m = 0; m < M_PER; ++m) {
      const half* ak = &s_a[m][k_tile * K_TILE];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const half* wjk = ((n0 + j) / 16 != tile_idx0) ? w1[j] : w0[j];
#pragma unroll
        for (int h = 0; h < K_TILE / 2; ++h) {
          half2 a2 = __halves2half2(ak[2 * h], ak[2 * h + 1]);
          half2 w2 = __halves2half2(wjk[2 * h], wjk[2 * h + 1]);
          acc[m][j] = __builtin_amdgcn_fdot2(w2, a2, acc[m][j], false);
        }
      }
    }
  }

  // Epilogue: atomically accumulate 4 columns per row (multi-K-block adds).
#pragma unroll
  for (int m = 0; m < M_PER; ++m) {
    const int mr = m_global(m, blockIdx.z, M_PER);
    if (mr >= size_m) continue;
    half* out = c + (int64_t)mr * size_n + n0;
    half2 r01 = __halves2half2(__float2half_rn(acc[m][0]),
                               __float2half_rn(acc[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(acc[m][2]),
                               __float2half_rn(acc[m][3]));
    if (gridDim.y > 1) {
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      union {
        unsigned long long u;
        half2 h2[2];
      } v;
      v.h2[0] = r01;
      v.h2[1] = r23;
      *reinterpret_cast<unsigned long long*>(out) = v.u;
    }
  }
}

// Small-M (decode) variant for bits=3: grain-based decode with per-k_tile
// tile-word register staging. A thread owns 16 N-cols = exactly one 16x16
// tile; per k_tile it loads the tile's 24 uint32 words once and derives all
// 32 grains (8 windows each) via fshift from registers. Cuts trellis load
// instructions ~8x vs the per-weight window reads above: bit-identical
// output, 4.2-4.8x faster at M=1 on gfx1030 (microbench 2026-09-04).
// Grain g = j*4+mg covers windows p = 8g..8g+7:
//   i=0,1 -> rows 2mg,2mg+1       col j    (c/8 = 0)
//   i=2,3 -> rows 8+2mg,8+2mg+1   col j    (c/8 = 0)
//   i=4,5 -> rows 2mg,2mg+1       col j+8  (c/8 = 1)
//   i=6,7 -> rows 8+2mg,8+2mg+1   col j+8  (c/8 = 1)
#define V2_THREADS_X 64
#define V2_BLOCK_N 1024          // 16 cols x 64 threads
#define V2_BLOCK_K 128
template <int M_PER, int cb>
__global__ void gemm_exl3_v2_kernel_rdna(
    const half* __restrict__ a, const int16_t* __restrict__ trellis,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k) {
  constexpr int V2_COL = 16;
  constexpr int V2_KTILE = 16;
  constexpr int NW = 24;  // bits=3 tile words
  const int t = threadIdx.x;
  const int n0 = blockIdx.x * V2_BLOCK_N + t * V2_COL;
  const int n_tiles_total = size_n / 16;
  const int offset_k = blockIdx.y * V2_BLOCK_K;
  const int end_k = min(offset_k + V2_BLOCK_K, size_k);
  constexpr int LDS_PAD = 8;
  __shared__ half s_a[M_PER][V2_BLOCK_K + LDS_PAD];
#pragma unroll 1
  for (int m = 0; m < M_PER; ++m) {
    const int mr = blockIdx.z * M_PER + m;
    for (int kk = t; kk < V2_BLOCK_K; kk += V2_THREADS_X) {
      s_a[m][kk] = (mr < size_m)
          ? a[(int64_t)mr * size_k + offset_k + kk]
          : __float2half_rn(0.0f);
    }
  }
  __syncthreads();
  if (n0 >= size_n) return;
  float acc[M_PER][V2_COL];
#pragma unroll
  for (int m = 0; m < M_PER; ++m)
#pragma unroll
    for (int j = 0; j < V2_COL; ++j) acc[m][j] = 0.0f;

  const int tile_idx = n0 / 16;

  for (int kt = 0; kt < (end_k - offset_k) / V2_KTILE; ++kt) {
    const uint32_t* tp = reinterpret_cast<const uint32_t*>(
        trellis + ((int64_t)(offset_k / V2_KTILE + kt) * n_tiles_total + tile_idx)
            * 2 * NW);
    uint32_t tw[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) tw[w] = tp[w];

#pragma unroll
    for (int g = 0; g < 32; ++g) {
      const int j = g >> 2;
      const int mg = g & 3;
      half2 wpair[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // even window position p = 8g + 2q; tail-biting pair read at tpos=p/2
        const int tpos = 4 * g + q;
        const int b0 = tpos * 6 + 755;  // tpos*2*bits + bits - 16 + 256*bits
        const int b2 = b0 + 19;         // b0 + bits + 16
        const int i1_raw = (b2 - 1) >> 5;
        const int i0 = (b0 >> 5) % NW;
        const int i1 = i1_raw % NW;
        // s1 must use the pre-modulo word index (tail-biting wrap): the
        // shift count is only valid in [0,31]; a negative count is UB.
        const int s1 = (i1_raw + 1) * 32 - b2;
        uint32_t w1f = fshift(tw[i1], tw[i0], s1);
        wpair[q] = __halves2half2(
            decode_3inst<cb>((w1f >> 3) & 0xffffu),  // even p -> w0
            decode_3inst<cb>(w1f & 0xffffu));        // odd p  -> w1
      }
      const int r_lo = 2 * mg, r_hi = 8 + 2 * mg;
#pragma unroll
      for (int m = 0; m < M_PER; ++m) {
        const int mr = blockIdx.z * M_PER + m;
        if (mr >= size_m) continue;
        const half* ak = &s_a[m][kt * V2_KTILE];
        half2 a_lo = __halves2half2(ak[r_lo], ak[r_lo + 1]);
        half2 a_hi = __halves2half2(ak[r_hi], ak[r_hi + 1]);
        acc[m][j] = __builtin_amdgcn_fdot2(wpair[0], a_lo, acc[m][j], false);
        acc[m][j] = __builtin_amdgcn_fdot2(wpair[1], a_hi, acc[m][j], false);
        acc[m][j + 8] = __builtin_amdgcn_fdot2(wpair[2], a_lo, acc[m][j + 8], false);
        acc[m][j + 8] = __builtin_amdgcn_fdot2(wpair[3], a_hi, acc[m][j + 8], false);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < M_PER; ++m) {
    const int mr = blockIdx.z * M_PER + m;
    if (mr >= size_m) continue;
    half* out = c + (int64_t)mr * size_n + n0;
#pragma unroll
    for (int jj = 0; jj < V2_COL; jj += 4) {
      half2 r01 = __halves2half2(__float2half_rn(acc[m][jj]),
                                 __float2half_rn(acc[m][jj + 1]));
      half2 r23 = __halves2half2(__float2half_rn(acc[m][jj + 2]),
                                 __float2half_rn(acc[m][jj + 3]));
      if (gridDim.y > 1) {
        atomic_add_pk4_f16(out + jj, r01, r23);
      } else {
        union {
          unsigned long long u;
          half2 h2[2];
        } v;
        v.h2[0] = r01;
        v.h2[1] = r23;
        *reinterpret_cast<unsigned long long*>(out + jj) = v.u;
      }
    }
  }
}

#else  // non-RDNA: empty stub for symbol parity

template <int M_PER, int bits, int cb>
__global__ void gemm_exl3_kernel_rdna(const half*, const int16_t*, half*,
                                      const int, const int, const int) {}
template <int M_PER, int cb>
__global__ void gemm_exl3_v2_kernel_rdna(const half*, const int16_t*, half*,
                                         const int, const int, const int) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

__forceinline__ int divide_up(int x, int y) { return (x + y - 1) / y; }

template <int M_PER, int bits, int cb>
void launch_mcb(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                int sk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(divide_up(sn, BLOCK_N), divide_up(sk, BLOCK_K),
            divide_up(sm, M_PER));
  gemm_exl3_kernel_rdna<M_PER, bits, cb>
      <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
}

template <int M_PER, int bits>
void launch_mb(const half* a, const int16_t* trellis, half* c, int sm, int sn,
               int sk, int cb, cudaStream_t stream) {
  if (cb == 0)
    launch_mcb<M_PER, bits, 0>(a, trellis, c, sm, sn, sk, stream);
  else if (cb == 1)
    launch_mcb<M_PER, bits, 1>(a, trellis, c, sm, sn, sk, stream);
  else if (cb == 2)
    launch_mcb<M_PER, bits, 2>(a, trellis, c, sm, sn, sk, stream);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_PER>
void launch_v2(const half* a, const int16_t* trellis, half* c, int sm, int sn,
               int sk, int cb, cudaStream_t stream) {
  dim3 block(V2_THREADS_X);
  dim3 grid(divide_up(sn, V2_BLOCK_N), divide_up(sk, V2_BLOCK_K),
            divide_up(sm, M_PER));
  if (cb == 0)
    gemm_exl3_v2_kernel_rdna<M_PER, 0>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else if (cb == 1)
    gemm_exl3_v2_kernel_rdna<M_PER, 1>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else if (cb == 2)
    gemm_exl3_v2_kernel_rdna<M_PER, 2>
        <<<grid, block, 0, stream>>>(a, trellis, c, sm, sn, sk);
  else
    TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported cb=", cb);
}

template <int M_PER>
void launch_m(const half* a, const int16_t* trellis, half* c, int sm, int sn,
              int sk, int bits, int cb, cudaStream_t stream) {
  switch (bits) {
    case 2: launch_mb<M_PER, 2>(a, trellis, c, sm, sn, sk, cb, stream); break;
    case 3: launch_mb<M_PER, 3>(a, trellis, c, sm, sn, sk, cb, stream); break;
    case 4: launch_mb<M_PER, 4>(a, trellis, c, sm, sn, sk, cb, stream); break;
    case 6: launch_mb<M_PER, 6>(a, trellis, c, sm, sn, sk, cb, stream); break;
    default: TORCH_CHECK(false, "exl3_gemm_rdna2: unsupported bits=", bits);
  }
}

void launch_tile(const half* a, const int16_t* trellis, half* c, int sm, int sn,
                 int sk, int bits, int cb, cudaStream_t stream) {
  // bits=3 decode batches (sm <= max_num_seqs): the grain-based v2 kernel.
  // M_PER=4 z-split beats M_PER=8 (register pressure); prefill chunks
  // (sm > 8) keep the original kernel.
  if (bits == 3 && sm <= 8) {
    if (sm == 1) launch_v2<1>(a, trellis, c, sm, sn, sk, cb, stream);
    else if (sm == 2) launch_v2<2>(a, trellis, c, sm, sn, sk, cb, stream);
    else launch_v2<4>(a, trellis, c, sm, sn, sk, cb, stream);
    return;
  }
  if (sm == 1)
    launch_m<1>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 3)
    launch_m<2>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else if (sm <= 7)
    launch_m<4>(a, trellis, c, sm, sn, sk, bits, cb, stream);
  else
    launch_m<8>(a, trellis, c, sm, sn, sk, bits, cb, stream);
}

}  // namespace exl3_dot2
}  // namespace vllm

namespace vllm {
namespace exl3_dot2 {

#if defined(__HIP__RDNA__) || !defined(__HIP_DEVICE_COMPILE__)

// One block per 16x16 tile, one output element per thread. The GEMM
// kernel re-runs this exact decode per M-block (M_PER=8 cap); pulling it
// out lets prefill decode each tile once and hand the dot work to rocBLAS.
template <int bits, int cb>
__global__ void decode_trellis_kernel_rdna(const int16_t* __restrict__ trellis,
                                           half* __restrict__ out,
                                           const int size_k,
                                           const int size_n) {
  const int kt = blockIdx.x;
  const int nt = blockIdx.y;
  const int n_tiles = size_n / 16;
  const int16_t* tile =
      trellis + ((int64_t)kt * n_tiles + nt) * (2 * 8 * bits);
  const int t = threadIdx.x;
  const int r = t / 16;
  const int c = t % 16;
  const int p = exl3_window_pos<bits>(r, c);
  const uint32_t win =
      exl3_window_at<bits>(reinterpret_cast<const uint32_t*>(tile), p);
  out[(int64_t)(kt * 16 + r) * size_n + (nt * 16 + c)] =
      decode_3inst<cb>(win);
}

#else  // non-RDNA: empty stub for symbol parity

template <int bits, int cb>
__global__ void decode_trellis_kernel_rdna(const int16_t*, half*, const int,
                                           const int) {}

#endif  // __HIP__RDNA__ || !__HIP_DEVICE_COMPILE__

template <int bits, int cb>
void launch_decode_trellis(const int16_t* trellis, half* out, int sk, int sn,
                           cudaStream_t stream) {
  dim3 grid(sk / 16, sn / 16);
  decode_trellis_kernel_rdna<bits, cb>
      <<<grid, dim3(256), 0, stream>>>(trellis, out, sk, sn);
}

template <int bits>
void launch_decode_cb(const int16_t* trellis, half* out, int sk, int sn,
                      int cb, cudaStream_t stream) {
  if (cb == 0)
    launch_decode_trellis<bits, 0>(trellis, out, sk, sn, stream);
  else if (cb == 1)
    launch_decode_trellis<bits, 1>(trellis, out, sk, sn, stream);
  else if (cb == 2)
    launch_decode_trellis<bits, 2>(trellis, out, sk, sn, stream);
  else
    TORCH_CHECK(false, "exl3_decode_trellis_rdna2: unsupported cb=", cb);
}

}  // namespace exl3_dot2
}  // namespace vllm

void exl3_decode_trellis_rdna2(torch::Tensor trellis, torch::Tensor out,
                               int64_t bits, int64_t cb) {
  const int64_t size_k = trellis.size(0) * 16;
  const int64_t size_n = trellis.size(1) * 16;
  TORCH_CHECK(trellis.is_cuda() && out.is_cuda(), "tensors must be CUDA/HIP");
  TORCH_CHECK(trellis.dim() == 3, "trellis 3D [K/16, N/16, W]");
  TORCH_CHECK(out.scalar_type() == torch::kHalf &&
                  out.size(0) == size_k && out.size(1) == size_n,
              "out must be fp16 [K, N]");
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4,
              "exl3_decode_trellis_rdna2: bits must be 2/3/4 (bits=6 has "
              "exl3_dequant_bits6_mul1)");
  const at::cuda::OptionalCUDAGuard dg(device_of(trellis));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (bits) {
    case 2:
      vllm::exl3_dot2::launch_decode_cb<2>(
          (const int16_t*)trellis.data_ptr(), (half*)out.data_ptr(),
          (int)size_k, (int)size_n, (int)cb, stream);
      break;
    case 3:
      vllm::exl3_dot2::launch_decode_cb<3>(
          (const int16_t*)trellis.data_ptr(), (half*)out.data_ptr(),
          (int)size_k, (int)size_n, (int)cb, stream);
      break;
    default:
      vllm::exl3_dot2::launch_decode_cb<4>(
          (const int16_t*)trellis.data_ptr(), (half*)out.data_ptr(),
          (int)size_k, (int)size_n, (int)cb, stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

void exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
                     int64_t bits, int64_t cb) {
  // Derive sizes from tensors. Taking them as Python ints at the call
  // site would force dynamo to specialize the symbolic
  // input_ids.size()[0] (= a.size(0)) to the trace-time batch (2048),
  // firing ConstraintViolationError against V2's dynamic marker.
  const int64_t size_m = a.size(0);
  const int64_t size_n = c.size(1);
  const int64_t size_k = a.size(1);
  TORCH_CHECK(a.is_cuda() && c.is_cuda() && trellis.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(a.dim() == 2 && trellis.dim() == 3,
              "a 2D, trellis 3D [K/16, N/16, W]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "exl3_gemm_rdna2 fp16 only");
  TORCH_CHECK(size_k % 16 == 0 && size_n % 16 == 0,
              "K and N multiples of 16");
  const at::cuda::OptionalCUDAGuard dg(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Caller must pre-zero c (atomic accumulation, W4A16/mxfp4 convention).
  vllm::exl3_dot2::launch_tile(
      (const half*)a.data_ptr(), (const int16_t*)trellis.data_ptr(),
      (half*)c.data_ptr(), (int)size_m, (int)size_n, (int)size_k, (int)bits,
      (int)cb, stream);
}