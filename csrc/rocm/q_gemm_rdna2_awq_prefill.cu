// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 AWQ prefill kernel for AMD RDNA2 (gfx1030).
//
// AWQ-native clone of the exllama prefill structure:
//   * Exllama-like tile: BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, warps=4 (128 threads)
//   * Stage activations (A) in LDS; read weights (B) from global and
//     dequantize on the fly using the same fp16 bit-trick as
//     qdq_4_rdna2.cuh
//   * Inner dot uses __builtin_amdgcn_fdot2 (RDNA2's native fp16 dot)
//   * AWQ-native: zero_offset=0 (no GPTQv1 +1 quirk)
//   * Multi-K-split with fp32 partials + fixed-order reduce (deterministic)
//   * Packed-fp16 atomic-add epilogue for single-split launches
//
// Exllama was used at high M for GPTQ because its tile structure is better
// for high-M prefill (multiple M rows per block, good LDS reuse for
// activations). This kernel replicates that structure but for AWQ format.
//
// Targeting Qwen3.8-27B-AWQ (group_size=32, asymmetric, scalar_types.uint4):
//   - decode:        M=1   -> use gptq_gemm_rdna2 (decode kernel)
//   - prefill:       M=16  -> use gptq_gemm_rdna2_prefill (existing prefill kernel)
//   - chunked-prefill: M=128  -> use THIS kernel (exllama-clone, AWQ-native)
//   - full-prefill:  M=2048 -> use THIS kernel (exllama-clone, AWQ-native)
//
// Exllama cannot be used for AWQ: it always adds +1 to stored zeros
// (a GPTQv1 format quirk), which corrupts AWQ's literal zero values.

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
namespace gptq_rdna2_awq_prefill {

// Pull in shared W4A16 helpers (refresh_group, epilogue, dot22_8_f, etc.)
// from the decode kernel's namespace via the common header.
using namespace vllm::gptq_rdna2;

// ---------------------------------------------------------------------------
// Tile configuration (single, exllama-inspired, optimized for high M).
//
// BLOCK_M=16  (process 16 M rows per block — exllama default)
// BLOCK_N=64  (4 N columns per thread, 16 threads in N dim)
// BLOCK_K=32  (4 int4 weights per thread per K-step; 4-weight dequant covers 8 fp16)
// THREADS=128 (4 warps on RDNA2 wave32, 8 waves per block; 16x8 thread grid)
// LDS_PAD=8  (avoid bank conflicts when M rows read same K offset)
//
// Each thread computes BLOCK_M rows of 4 N columns = 64 fp16 outputs.
// ---------------------------------------------------------------------------
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int THREADS = 128;
constexpr int N_PER_THREAD = 4;  // BLOCK_N / (THREADS / 8) = 64 / 16
constexpr int M_PER_THREAD = 2;  // BLOCK_M / (THREADS / 8) = 16 / 8
constexpr int LDS_PAD = 8;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Main AWQ prefill kernel.
// ---------------------------------------------------------------------------
//
// Grid:  dim3((size_n + BLOCK_N - 1) / BLOCK_N,
//            (size_m + BLOCK_M - 1) / BLOCK_M,
//            split_k)
// Block: dim3(THREADS)
//
// Each block computes a BLOCK_M x BLOCK_N output tile over a K-split.
// Threads are organized as (m_tid=0..7, n_tid=0..15):
//   - m_tid indexes the M row within the tile (each thread covers M_PER_THREAD
//     consecutive rows starting at m_tid * M_PER_THREAD)
//   - n_tid indexes the N column group (each thread covers N_PER_THREAD
//     consecutive columns starting at n_tid * N_PER_THREAD)
//
__global__ __launch_bounds__(THREADS) void gemm_awq_prefill_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm, const int split_k,
    float* __restrict__ partials) {
  const int t = threadIdx.x;
  const int m_tid = t / 16;   // 0..7
  const int n_tid = t % 16;   // 0..15

  const int block_n = blockIdx.x * BLOCK_N;
  const int block_m = blockIdx.y * BLOCK_M;
  const int split_idx = blockIdx.z;

  // K range for this split. Split-K evenly divides size_k (caller enforces).
  const int k_per_split = size_k / split_k;
  const int k_start = split_idx * k_per_split;
  const int k_end = k_start + k_per_split;

  // Per-thread N columns: n_tid * N_PER_THREAD + 0..N_PER_THREAD-1
  const int n0 = block_n + n_tid * N_PER_THREAD;

  // Per-thread M rows: m_tid * M_PER_THREAD + 0..M_PER_THREAD-1
  const int m0 = block_m + m_tid * M_PER_THREAD;

  // Stage activations [BLOCK_M, BLOCK_K] in LDS with LDS_PAD to avoid bank
  // conflicts when different M rows read the same K offset.
  constexpr int lds_k_stride = BLOCK_K + LDS_PAD;
  __shared__ half lds_a[BLOCK_M * lds_k_stride];

  // Per-thread accumulators [M_PER_THREAD][N_PER_THREAD] in fp32.
  // fp32 accumulators avoid the ~3 bits of precision loss from fp16 FMA chain.
  float acc[M_PER_THREAD][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < M_PER_THREAD; ++m) {
    #pragma unroll
    for (int n = 0; n < N_PER_THREAD; ++n) {
      acc[m][n] = 0.0f;
    }
  }

  // Per-column dequant constants (z1z16, y1y16) for N_PER_THREAD columns.
  // Initialized when we cross a group boundary; refreshed at each transition.
  half2 z1z16_h[N_PER_THREAD][2];
  half2 y1y16_h[N_PER_THREAD][2];

  // Group bookkeeping. groupsize = K / groups.
  const int groupsize = size_k / groups;

  // Weight pointer: b_q_weight is [K/8, N] int32, with 8 nibbles per int32.
  // Each thread reads int4 values from b_q_weight[k/8, n] and extracts 4 bits.
  // For BLOCK_K K-elements, we read BLOCK_K/8 = 4 int32 words per N column.
  // With N_PER_THREAD columns, that's 4 * N_PER_THREAD = 16 int32 reads per
  // K-block (per thread).

  // Walk the K range in steps of BLOCK_K.
  for (int k_block = k_start; k_block < k_end; k_block += BLOCK_K) {
    // ---- Stage A into LDS ---------------------------------------------
    // Each thread loads BLOCK_M/8 * BLOCK_K/16 ... actually with 128 threads
    // loading 16x32 = 512 halfs, each thread loads 4 halfs.
    // Layout: thread t loads lds_a[(t/4) * lds_k_stride + (t%4) * 8 + 0..3]
    // (8 threads per M row, 16 M rows / 8 = 2 threads per row of LDS).
    // Simpler: each thread loads lds_a[t * 4 .. t * 4 + 3]? No, 128 threads
    // * 4 = 512 = 16*32. So thread t loads 4 halfs at offset t*4.
    {
      const int load_t = t;
      const int load_m = load_t * 4 / BLOCK_K;  // 0..15
      const int load_k = (load_t * 4) % BLOCK_K;  // 0, 4, 8, ..., 28
      const int g_m = block_m + load_m;
      const int g_k = k_block + load_k;
      half v[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int gk = g_k + i;
        if (g_m < size_m && gk < size_k && gk < k_end) {
          const half* a_row = a + g_m * size_k;
          if (b_q_perm) {
            v[i] = a_row[b_q_perm[gk]];
          } else {
            v[i] = a_row[gk];
          }
        } else {
          v[i] = __float2half_rn(0.0f);
        }
      }
      // Store to LDS at lds_a[load_m, load_k..load_k+3]
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        lds_a[load_m * lds_k_stride + load_k + i] = v[i];
      }
    }
    __syncthreads();

    // ---- Compute group index for this K-block ------------------------
    // Group transitions are checked at K-block granularity; we require
    // groupsize >= BLOCK_K (caller enforces groupsize >= 32).
    const int group = k_block / groupsize;
    // Refresh dequant constants for N_PER_THREAD columns at n0.
    // For threads with n0 + j >= size_n, we'll mask the contribution below.
    refresh_group<N_PER_THREAD>(group, n0, b_qzeros, b_scales, size_n,
                                zero_offset, z1z16_h, y1y16_h);

    // ---- Inner GEMM over BLOCK_K ---------------------------------------
    // For each K-step (8 nibbles = 1 int32 word, 8 K elements):
    //   Load 4 int32 words (covering N_PER_THREAD * 8 K elements)
    //   Dequant each word into 4 half2 pairs
    //   FMA with LDS activations
    //
    // k_within_block: 0..BLOCK_K in steps of 8 (1 int32 word = 8 nibbles)
    // For each of 4 int32 words, we process N_PER_THREAD columns.
    constexpr int K_STEPS = BLOCK_K / 8;  // 4

    #pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      const int k_off = ks * 8;  // K offset within the K-block
      const int qk = (k_block + k_off) / 8;  // weight row index

      // Load N_PER_THREAD int32 words (one per output column).
      uint32_t qw[N_PER_THREAD];
      #pragma unroll
      for (int j = 0; j < N_PER_THREAD; ++j) {
        const int n_col = n0 + j;
        if (n_col < size_n) {
          qw[j] = b_q_weight[qk * size_n + n_col];
        } else {
          qw[j] = 0;
        }
      }

      // Dequant each int32 into 4 half2 pairs and FMA with LDS A.
      #pragma unroll
      for (int j = 0; j < N_PER_THREAD; ++j) {
        half2 dq[4];
        dequant_4bit_8_fp16(qw[j], dq, z1z16_h[j], y1y16_h[j]);
        // dq[0] = (q[0], q[1]), dq[1] = (q[2], q[3]),
        // dq[2] = (q[4], q[5]), dq[3] = (q[6], q[7])
        // 8 K elements starting at k_off within the K-block.
        // FMA into acc[m][j] for each m in M_PER_THREAD.
        #pragma unroll
        for (int m = 0; m < M_PER_THREAD; ++m) {
          const int g_m = m0 + m;
          if (g_m >= size_m) continue;
          // Load 8 halfs from LDS: lds_a[g_m, k_off..k_off+7]
          const half* a_ptr = &lds_a[(m0 - block_m + m) * lds_k_stride + k_off];
          // 4 V_DOT2_F32_F16 calls (each covers 2 half2 = 4 K elements)
          acc[m][j] = __builtin_amdgcn_fdot2(
              dq[0], *reinterpret_cast<const half2*>(a_ptr + 0),
              acc[m][j], false);
          acc[m][j] = __builtin_amdgcn_fdot2(
              dq[1], *reinterpret_cast<const half2*>(a_ptr + 2),
              acc[m][j], false);
          acc[m][j] = __builtin_amdgcn_fdot2(
              dq[2], *reinterpret_cast<const half2*>(a_ptr + 4),
              acc[m][j], false);
          acc[m][j] = __builtin_amdgcn_fdot2(
              dq[3], *reinterpret_cast<const half2*>(a_ptr + 6),
              acc[m][j], false);
        }
      }
    }

    __syncthreads();
  }

  // ---- Epilogue: write to c -----------------------------------------
  // Multi-split: write fp32 partials at [split_idx, m, n] (deterministic).
  // Single-split: use packed-fp16 atomic add (zero-initialized output).
  if (partials != nullptr) {
    #pragma unroll
    for (int m = 0; m < M_PER_THREAD; ++m) {
      const int g_m = m0 + m;
      if (g_m >= size_m) continue;
      #pragma unroll
      for (int j = 0; j < N_PER_THREAD; ++j) {
        const int n_col = n0 + j;
        if (n_col >= size_n) continue;
        float* p = partials + split_idx * size_m * size_n +
                   g_m * size_n + n_col;
        *p = acc[m][j];
      }
    }
  } else {
    // Single-split: zero-initialized output + packed-fp16 atomic add.
    // For N_PER_THREAD=4 columns, we can do one 64-bit atomic per M row.
    // 4 fp16 = 8 bytes = 1 uint64 = atomic_add_pk4_f16.
    #pragma unroll
    for (int m = 0; m < M_PER_THREAD; ++m) {
      const int g_m = m0 + m;
      if (g_m >= size_m) continue;
      const int n_col = n0;
      if (n_col + N_PER_THREAD - 1 >= size_n) {
        // Fall back to scalar atomic adds if not enough N columns.
        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
          const int nc = n_col + j;
          if (nc >= size_n) continue;
          half v = __float2half_rn(acc[m][j]);
          atomicAdd(c + g_m * size_n + nc, v);
        }
      } else {
        half2 v01 = __halves2half2(__float2half_rn(acc[m][0]),
                                   __float2half_rn(acc[m][1]));
        half2 v23 = __halves2half2(__float2half_rn(acc[m][2]),
                                   __float2half_rn(acc[m][3]));
        atomic_add_pk4_f16(c + g_m * size_n + n_col, v01, v23);
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

__global__ void gemm_awq_prefill_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*,
    const int, const int, const int, const int, const int, const int*,
    const int, float*) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Split-K computation.
// ---------------------------------------------------------------------------
// Local reduce kernel for split-K results. Same logic as the common
// reduce_split_partials_kernel in q_gemm_rdna2_common.cuh, but defined
// locally to avoid namespace issues when including from a nested
// namespace.
// ---------------------------------------------------------------------------
__global__ void awq_reduce_split_partials_kernel(
    const float* __restrict__ partials, half* __restrict__ c, int size_m,
    int size_n, int num_splits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = size_m * size_n;
  if (idx >= total) return;
  const int m = idx / size_n;
  const int n = idx % size_n;
  float acc = 0.0f;
  for (int z = 0; z < num_splits; ++z) {
    acc += partials[(z * size_m + m) * size_n + n];
  }
  c[idx] = __float2half_rn(acc);
}

// ---------------------------------------------------------------------------
// Mirrors exllama's heuristic: split until LDS would exceed budget.
// gfx1030 has 64 KiB LDS per block. We use 32 KiB to leave headroom.
//
inline int compute_split_k(int size_m, int size_n, int size_k) {
  auto lds_bytes = [&](int split) {
    const int k_per_split = size_k / split;
    const int row_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;
    return BLOCK_M * row_stride * sizeof(half);
  };

  // 32 KiB LDS budget for stability (gfx1030 limit is 64 KiB).
  constexpr size_t lds_budget = 32 * 1024;

  int split_k = 1;
  const int max_split_k = size_k / BLOCK_K;
  while (split_k < max_split_k && lds_bytes(split_k * 2) <= lds_budget) {
    split_k *= 2;
  }
  return split_k;
}

inline void launch_awq_prefill(
    const half* a, const uint32_t* b_q_weight, const uint32_t* b_qzeros,
    const half* b_scales, const int* b_q_perm, half* c, int size_m,
    int size_n, int size_k, int groups, bool use_v2_format,
    cudaStream_t stream) {
  // AWQ: zero_offset=0. (Kernel is AWQ-native; GPTQv1 +1 quirk not used.)
  const int zero_offset = use_v2_format ? 0 : 0;

  dim3 block(THREADS);
  dim3 grid((size_n + BLOCK_N - 1) / BLOCK_N,
            (size_m + BLOCK_M - 1) / BLOCK_M,
            1);  // split_k filled below

  int split_k = compute_split_k(size_m, size_n, size_k);
  // Align split_k to BLOCK_K (avoid reading past K-end in tail iteration).
  while (split_k > 1 && (size_k / split_k) % BLOCK_K != 0) split_k /= 2;
  grid.z = split_k;

  // Multi-split: deterministic two-phase epilogue. Each split writes
  // disjoint fp32 partials; a fixed-order reduce kernel sums them to fp16.
  torch::Tensor partials_t;
  float* partials = nullptr;
  if (split_k > 1) {
    partials_t = torch::empty(
        {static_cast<long>((size_t)split_k * size_m * size_n)},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(torch::kCUDA, c10::cuda::current_device())));
    partials = partials_t.data_ptr<float>();
  }

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)
  gemm_awq_prefill_kernel<<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm, split_k, partials);
#endif

  if (split_k > 1) {
    const int total = size_m * size_n;
    const int rblock = 256;
    awq_reduce_split_partials_kernel<<<
        (total + rblock - 1) / rblock, rblock, 0, stream>>>(
        partials, c, size_m, size_n, split_k);
  }
}

}  // namespace gptq_rdna2_awq_prefill
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros, AWQ literal values)
//   b_scales  [groups, N]       half
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format                bool   (must be True for AWQ; kept for API parity)
//
// Output:
//   c         [M, N]            half
//
torch::Tensor awq_gemm_rdna2_prefill(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
    torch::Tensor b_scales, torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "awq_gemm_rdna2_prefill only supports fp16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(1);
  const int groups = (int)b_qzeros.size(0);
  const int groupsize = size_k / groups;

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(size_n % 8 == 0,
              "N must be a multiple of 8 (qgemm/4bit nibble packing)");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % vllm::gptq_rdna2_awq_prefill::BLOCK_N == 0,
              "N must be a multiple of BLOCK_N for this kernel");
  TORCH_CHECK(size_m % vllm::gptq_rdna2_awq_prefill::BLOCK_M == 0 ||
                  size_m <= vllm::gptq_rdna2_awq_prefill::BLOCK_M,
              "M must be <= BLOCK_M or a multiple of it (no tail support yet)");
  TORCH_CHECK(size_k % 32 == 0, "K must be divisible by 32");
  TORCH_CHECK(groupsize >= 32, "group_size must be >= 32");
  TORCH_CHECK(use_v2_format,
              "awq_gemm_rdna2_prefill is AWQ-only (use_v2_format must be True)");

  auto c = torch::zeros({size_m, size_n}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  vllm::gptq_rdna2_awq_prefill::launch_awq_prefill(
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_qzeros.data_ptr()),
      reinterpret_cast<const half*>(b_scales.data_ptr()),
      g_idx_ptr,
      reinterpret_cast<half*>(c.data_ptr()), size_m, size_n, size_k, groups,
      use_v2_format, stream.stream());

  return c;
}
