// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ prefill kernel for AMD RDNA2 (gfx1030).
//
// Design (single algorithm, multi-tile):
//   * 3-D grid: each block owns one M_TILE x N_TILE output tile over a K-split.
//   * Activations (A) are staged in LDS; weights (B, int4) are read directly
//     from global and dequantized on the fly.
//   * Inner dot uses __builtin_amdgcn_fdot2 (RDNA2's native fp16 dot).
//   * Epilogue uses a 64-bit packed CAS atomic-add on fp16.
//   * K_PER_SPLIT is templated so the LDS array is a true 2-D static shared
//     array (lets the backend emit ds_read_b128 for A).
//   * Dynamic fallback for odd K-per-split values.
//
// Tile configurations (compile-time `Config` template):
//   ConfigV1 (THREADS=512, N=2048, M=8, LDS=8)   — original v1 tile, wins for
//                                                   small M (M <= 64).
//   ConfigA  (THREADS=256, N=1024, M=16, LDS=0)  — general-purpose prefill
//                                                   tile, wins M >= 96 large N.
//   ConfigC  (THREADS=128, N= 512, M=16, LDS=0)  — small-N tile, wins
//                                                   N <= 1024.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

#include "q_gemm_rdna2_common.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2_prefill {

// The shared W4A16 helpers (refresh_group, epilogue, dot22_8_f, etc.) live
// in the sibling gptq_rdna2 namespace via q_gemm_rdna2_common.cuh. Pull
// them in so the call sites below can stay unqualified.
using namespace vllm::gptq_rdna2;

// ---------------------------------------------------------------------------
// Tile configuration template.
//
// All tile dimensions are compile-time constants.  N_PER_THREAD is fixed at 4
// so the pk4 fp16 atomic epilogue is shared across every config.  LDS_PAD=0
// for the wider-M configs keeps the static A tile within the 64 KiB local
// memory limit of gfx1030.
// ---------------------------------------------------------------------------
template <int Threads_, int NPerThread_, int KStep_, int MTile_, int LdsPad_>
struct Config {
  static constexpr int THREADS      = Threads_;
  static constexpr int N_PER_THREAD = NPerThread_;
  static constexpr int N_TILE       = THREADS * N_PER_THREAD;
  static constexpr int K_STEP       = KStep_;
  static constexpr int M_TILE       = MTile_;
  static constexpr int LDS_PAD      = LdsPad_;
};

// Configs exposed to the dispatcher.
using ConfigV1 = Config<512, 4, 32,  8,  8>;   // v1 tile (small M)
using ConfigA  = Config<256, 4, 32, 16,  0>;   // general prefill (large N)
using ConfigC  = Config<128, 4, 32, 16,  0>;   // small N (N_TILE=512)

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Device-side helpers (dot22_8_f, atomic_add_pk4_f16, load4_zeros,
// load4_scales, refresh_group, epilogue) live in q_gemm_rdna2_common.cuh.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Static K-per-split variant: LDS is a 2-D static shared array with a
// constant row stride, so the backend can emit ds_read_b128 for A.
// ---------------------------------------------------------------------------
template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::THREADS) void gemm_static_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm, const int split_k) {
  constexpr int k_per_split = K_PER_SPLIT;
  constexpr int THREADS = Config::THREADS;
  constexpr int N_PER_THREAD = Config::N_PER_THREAD;
  constexpr int N_TILE = Config::N_TILE;
  constexpr int K_STEP = Config::K_STEP;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int t = threadIdx.x;
  const int n = blockIdx.x * N_TILE + t * N_PER_THREAD;
  const int m_tile = blockIdx.y * M_TILE;
  const bool active = (n < size_n);

  const int k_split = blockIdx.z;
  const int k_start = k_split * k_per_split;
  const int k_end = (k_split == split_k - 1) ? size_k : (k_start + k_per_split);

  const int groupsize = size_k / groups;
  int group = k_start / groupsize;
  int nextgroup = (group + 1) * groupsize;

  half2 z1z16_h[N_PER_THREAD][2];
  half2 y1y16_h[N_PER_THREAD][2];

  if (active) {
    refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                          z1z16_h, y1y16_h);
  }

  float block_c[M_TILE][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] = 0.0f;
  }

  __shared__ half block_a[M_TILE][k_per_split + LDS_PAD];
  if (b_q_perm) {
    #pragma unroll 1
    for (int idx = t; idx < M_TILE * k_per_split; idx += THREADS) {
      const int m = idx / k_per_split;
      const int kk = idx % k_per_split;
      const int m_row = m_tile + m;
      const int k = k_start + kk;
      if (m_row < size_m)
        block_a[m][kk] = (a + m_row * size_k)[b_q_perm[k]];
    }
  } else {
    #pragma unroll 1
    for (int idx = t; idx < (M_TILE * k_per_split) / 4; idx += THREADS) {
      const int quad = idx % (k_per_split / 4);
      const int m = idx / (k_per_split / 4);
      const int m_row = m_tile + m;
      const int k = k_start + quad * 4;
      if (m_row < size_m) {
        const float2 a4 = *(const float2*)(a + m_row * size_k + k);
        *(float2*)(&block_a[m][quad * 4]) = a4;
      }
    }
  }
  __syncthreads();

  const uint32_t* b_ptr = b_q_weight + (k_start / 8) * size_n + n;

  int k = k_start;
  if (active) {
    while (k < k_end) {
      if (k == nextgroup) {
        group++;
        nextgroup += groupsize;
        refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                              z1z16_h, y1y16_h);
      }

      int4 b_prefetch[4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        b_prefetch[j] = *(const int4*)(b_ptr + j * size_n);
      }
      b_ptr += 4 * size_n;

      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int a_off = 8 * j;
        half2 dq[N_PER_THREAD][4];
        uint32_t w[N_PER_THREAD];
        w[0] = static_cast<uint32_t>(b_prefetch[j].x);
        w[1] = static_cast<uint32_t>(b_prefetch[j].y);
        w[2] = static_cast<uint32_t>(b_prefetch[j].z);
        w[3] = static_cast<uint32_t>(b_prefetch[j].w);
        #pragma unroll
        for (int col = 0; col < N_PER_THREAD; ++col) {
          vllm::gptq_rdna2::dequant_4bit_8_fp16(
              w[col], dq[col], z1z16_h[col], y1y16_h[col]);
        }

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          const int m_row = m_tile + m;
          if (m_row >= size_m) continue;
          const half* a_window = &block_a[m][(k - k_start) + a_off];
          #pragma unroll
          for (int col = 0; col < N_PER_THREAD; ++col) {
            block_c[m][col] += dot22_8_f(dq[col], a_window);
          }
        }
      }

      k += K_STEP;
    }
    epilogue<Config::M_TILE>(block_c, m_tile, size_m, size_n, n, c);
  }
}

// ---------------------------------------------------------------------------
// Dynamic K-per-split fallback: LDS size and row stride are runtime values.
// Always works regardless of k_per_split, but pays a small address-compute
// cost vs the static variant.
// ---------------------------------------------------------------------------
template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm, const int split_k) {
  constexpr int THREADS = Config::THREADS;
  constexpr int N_PER_THREAD = Config::N_PER_THREAD;
  constexpr int N_TILE = Config::N_TILE;
  constexpr int K_STEP = Config::K_STEP;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int t = threadIdx.x;
  const int n = blockIdx.x * N_TILE + t * N_PER_THREAD;
  const int m_tile = blockIdx.y * M_TILE;
  const bool active = (n < size_n);

  const int k_split = blockIdx.z;
  const int k_per_split = size_k / split_k;
  const int k_start = k_split * k_per_split;
  const int k_end = (k_split == split_k - 1) ? size_k : (k_start + k_per_split);
  // Round the LDS row stride up to a 16-byte (8 half) boundary so 128-bit
  // LDS loads are always aligned.
  const int row_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;

  const int groupsize = size_k / groups;
  int group = k_start / groupsize;
  int nextgroup = (group + 1) * groupsize;

  half2 z1z16_h[N_PER_THREAD][2];
  half2 y1y16_h[N_PER_THREAD][2];

  if (active) {
    refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                          z1z16_h, y1y16_h);
  }

  float block_c[M_TILE][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] = 0.0f;
  }

  extern __shared__ half block_a[];
  if (b_q_perm) {
    #pragma unroll 1
    for (int idx = t; idx < M_TILE * k_per_split; idx += THREADS) {
      const int m = idx / k_per_split;
      const int kk = idx % k_per_split;
      const int m_row = m_tile + m;
      const int k = k_start + kk;
      if (m_row < size_m)
        block_a[m * row_stride + kk] = (a + m_row * size_k)[b_q_perm[k]];
    }
  } else {
    if (k_per_split % 4 == 0) {
      const int half4_count = (M_TILE * k_per_split) / 4;
      #pragma unroll 1
      for (int idx = t; idx < half4_count; idx += THREADS) {
        const int quad = idx % (k_per_split / 4);
        const int m = idx / (k_per_split / 4);
        const int m_row = m_tile + m;
        const int k = k_start + quad * 4;
        if (m_row < size_m) {
          const float2 a4 = *(const float2*)(a + m_row * size_k + k);
          *(float2*)(block_a + m * row_stride + quad * 4) = a4;
        }
      }
    } else {
      const int half2_count = (M_TILE * k_per_split) / 2;
      #pragma unroll 1
      for (int idx = t; idx < half2_count; idx += THREADS) {
        const int pair = idx % (k_per_split / 2);
        const int m = idx / (k_per_split / 2);
        const int m_row = m_tile + m;
        const int k = k_start + pair * 2;
        if (m_row < size_m) {
          const half2 a2 = *(const half2*)(a + m_row * size_k + k);
          *(half2*)(block_a + m * row_stride + pair * 2) = a2;
        }
      }
    }
  }
  __syncthreads();

  const uint32_t* b_ptr = b_q_weight + (k_start / 8) * size_n + n;

  int k = k_start;
  if (active) {
    while (k < k_end) {
      if (k == nextgroup) {
        group++;
        nextgroup += groupsize;
        refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                              z1z16_h, y1y16_h);
      }

      int4 b_prefetch[4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        b_prefetch[j] = *(const int4*)(b_ptr + j * size_n);
      }
      b_ptr += 4 * size_n;

      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int a_off = 8 * j;
        half2 dq[N_PER_THREAD][4];
        uint32_t w[N_PER_THREAD];
        w[0] = static_cast<uint32_t>(b_prefetch[j].x);
        w[1] = static_cast<uint32_t>(b_prefetch[j].y);
        w[2] = static_cast<uint32_t>(b_prefetch[j].z);
        w[3] = static_cast<uint32_t>(b_prefetch[j].w);
        #pragma unroll
        for (int col = 0; col < N_PER_THREAD; ++col) {
          vllm::gptq_rdna2::dequant_4bit_8_fp16(
              w[col], dq[col], z1z16_h[col], y1y16_h[col]);
        }

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          const int m_row = m_tile + m;
          if (m_row >= size_m) continue;
          // Force a 128-bit LDS load; the dynamic row stride is always a
          // multiple of 8 halfs, so the address is 16-byte aligned.
          const int a_base = m * row_stride + (k - k_start) + a_off;
          float4 a8 = *(const float4*)(block_a + a_base);
          const half* a_ptr = reinterpret_cast<const half*>(&a8);
          #pragma unroll
          for (int col = 0; col < N_PER_THREAD; ++col) {
            block_c[m][col] += dot22_8_f(dq[col], a_ptr);
          }
        }
      }

      k += K_STEP;
    }
    epilogue<Config::M_TILE>(block_c, m_tile, size_m, size_n, n, c);
  }
}

#else  // non-RDNA2 device pass

// Stub kernels: same signatures so the launcher compiles on any gfx target.
// Unused at runtime (the dispatch path is gated by on_gfx10x() in Python).
template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::THREADS) void gemm_static_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*, const int,
    const int, const int, const int, const int, const int*, const int) {}

template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*, const int,
    const int, const int, const int, const int, const int*, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Config dispatch and launcher.
//
// select_config(M, N):
//   - M < 64    -> ConfigV1 (small-M corner where v1 wins by a wide margin).
//   - N <= 1024 -> ConfigC  (small-N tile, keeps multiple N blocks per CU).
//   - otherwise -> ConfigA  (general prefill tile, wins M >= 96 large N).
// ---------------------------------------------------------------------------
enum ConfigId : int { ConfigId_V1 = 0, ConfigId_A = 1, ConfigId_C = 3 };

template <typename Config>
inline void launch_for_config(
    const half* a, const uint32_t* b_q_weight, const uint32_t* b_qzeros,
    const half* b_scales, const int* b_q_perm, half* c, int size_m,
    int size_n, int size_k, int groups, int split_k, bool use_v2_format,
    cudaStream_t stream) {
  constexpr int N_TILE = Config::N_TILE;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int zero_offset = use_v2_format ? 0 : 1;
  dim3 block(Config::THREADS);
  dim3 grid((size_n + N_TILE - 1) / N_TILE, (size_m + M_TILE - 1) / M_TILE,
            split_k);
  const int k_per_split = size_k / split_k;
  const int lds_k_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;
  const size_t shmem = M_TILE * lds_k_stride * sizeof(half);

  switch (k_per_split) {
    case 256:
      gemm_static_kernel<Config, 256>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    case 512:
      gemm_static_kernel<Config, 512>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    case 1024:
      gemm_static_kernel<Config, 1024>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    default:
      gemm_dynamic_kernel<Config>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
  }
}

// Split-K search.  Mirrors v1's LDS-budget heuristic and adds a third tier
// (64 KiB) when the problem is medium-sized, matching gfx1030's full local
// memory capacity.
template <typename Config>
int compute_split_k(int size_m, int size_n, int size_k) {
  constexpr int N_TILE = Config::N_TILE;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;
  constexpr int K_STEP = Config::K_STEP;

  const int max_split_k = size_k / K_STEP;
  const int blocks_per_k_split =
      ((size_m + M_TILE - 1) / M_TILE) *
      ((size_n + N_TILE - 1) / N_TILE);

  auto lds_bytes = [&](int split) {
    const int k_per_split = size_k / split;
    const int row_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;
    return M_TILE * row_stride * sizeof(half);
  };
  // 16 KiB when the grid is large (avoid HSA invalid-allocation crashes),
  // 64 KiB at medium scale (use full gfx1030 LDS capacity to cut atomic
  // contention), 32 KiB otherwise.
  const size_t lds_budget =
      (blocks_per_k_split > 1024) ? (16 * 1024)
      : (blocks_per_k_split > 256) ? (64 * 1024)
                                   : (32 * 1024);

  int split_k = 1;
  while (split_k < max_split_k && lds_bytes(split_k) > lds_budget) {
    split_k *= 2;
  }
  while (split_k < 16 && max_split_k >= split_k * 2 &&
         (blocks_per_k_split * split_k < 2048 ||
          (size_k / split_k) > 2048)) {
    const int candidate = split_k * 2;
    if (lds_bytes(candidate) > lds_budget) break;
    split_k = candidate;
  }
  return split_k;
}

// Public dispatcher entry: pick a Config, compute split_k, launch.
// K-aware config selection. Mirrors the empirical 4-branch rule from the
// 3264-cell microbench (M=1..32, K<4096, N=128..16384): ConfigV1 is the safe
// default; ConfigC wins at M=4..8 with high N and K, and at M=12 with mid N
// and any K>=512; ConfigA wins at small M (M<4) with large N. The envelope
// guard (M<=32 AND K<4096) is enforced by the outer dispatcher
// (_rdna2_w4a16_select_kernel) so this function never sees out-of-envelope
// cells; the V1 default covers anything outside the explicit K-gated
// branches.
inline int select_config(int size_m, int size_n, int size_k) {
  if (size_m < 4 && size_n > 4096) return ConfigId_A;
  if (4 <= size_m && size_m <= 8 && size_n >= 2048 && size_k >= 1024)
    return ConfigId_C;
  if (size_m == 12 && size_n >= 2560 && size_n <= 8192 && size_k >= 512)
    return ConfigId_C;
  if (size_m == 32 && size_n >= 5120 && size_n <= 6144 && size_k >= 1536)
    return ConfigId_C;
  return ConfigId_V1;
}

void launch_dispatch(
    const half* a, const uint32_t* b_q_weight, const uint32_t* b_qzeros,
    const half* b_scales, const int* b_q_perm, half* c, int size_m,
    int size_n, int size_k, int groups, bool use_v2_format,
    cudaStream_t stream) {
  switch (select_config(size_m, size_n, size_k)) {
    case ConfigId_V1: {
      const int split_k = compute_split_k<ConfigV1>(size_m, size_n, size_k);
      launch_for_config<ConfigV1>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                  size_m, size_n, size_k, groups, split_k,
                                  use_v2_format, stream);
      break;
    }
    case ConfigId_C: {
      const int split_k = compute_split_k<ConfigC>(size_m, size_n, size_k);
      launch_for_config<ConfigC>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                 size_m, size_n, size_k, groups, split_k,
                                 use_v2_format, stream);
      break;
    }
    case ConfigId_A:
    default: {
      const int split_k = compute_split_k<ConfigA>(size_m, size_n, size_k);
      launch_for_config<ConfigA>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                 size_m, size_n, size_k, groups, split_k,
                                 use_v2_format, stream);
      break;
    }
  }
}

}  // namespace gptq_rdna2_prefill
}  // namespace vllm

// ---------------------------------------------------------------------------
// torch::Tensor public entry point.
//
// `gptq_gemm_rdna2_prefill` is the canonical name going forward.  It calls
// the K-aware `select_config(M, K, N)` to pick V1 / A / C internally; the
// Python dispatcher does not need to know about configs.  An ABI shim with
// the old `..._prefill_direct` name is kept at the bottom of the file so
// existing Python callers continue to link without modification.
// ---------------------------------------------------------------------------
torch::Tensor gptq_gemm_rdna2_prefill(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
    torch::Tensor b_scales, torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "gptq_gemm_rdna2_prefill only supports fp16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const int size_m = a.size(0);
  const int size_k = a.size(1);
  const int size_n = b_q_weight.size(1);
  const int groups = b_qzeros.size(0);
  const int groupsize = size_k / groups;

  TORCH_CHECK(size_k % 32 == 0, "K must be divisible by 32");
  // N divisibility: the smallest config (ConfigC, N_PER_THREAD=4) requires
  // N % 8 == 0 for aligned int4 loads.
  TORCH_CHECK(size_n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(size_k % groups == 0, "K must be divisible by groups");
  TORCH_CHECK(groupsize >= 32, "group_size must be >= 32");

  auto c = torch::zeros({size_m, size_n}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  vllm::gptq_rdna2_prefill::launch_dispatch(
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_qzeros.data_ptr()),
      reinterpret_cast<const half*>(b_scales.data_ptr()),
      b_g_idx.numel() ? b_g_idx.data_ptr<int>() : nullptr,
      reinterpret_cast<half*>(c.data_ptr()), size_m, size_n, size_k, groups,
      use_v2_format, stream.stream());

  return c;
}

// ---------------------------------------------------------------------------
// ABI-compat shim.
//
// Same signature and behaviour as the original v1 public entry.  Kept so
// existing Python callers (RDNA2W4A16LinearKernel.apply_weights in
// rdna2_w4a16.py) that call `ops.gptq_gemm_rdna2_prefill_direct(...)` keep
// working without a Python change.  New code should use
// `ops.gptq_gemm_rdna2_prefill` directly.
// ---------------------------------------------------------------------------
torch::Tensor gptq_gemm_rdna2_prefill_direct(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
    torch::Tensor b_scales, torch::Tensor b_g_idx, bool use_v2_format) {
  return gptq_gemm_rdna2_prefill(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                 use_v2_format);
}
