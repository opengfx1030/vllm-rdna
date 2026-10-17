// SPDX-License-Identifier: Apache-2.0
// HIP unquantized (fp16 weight / fp16 activation) MoE kernels for RDNA2
// (gfx1030).
//
// Replacements for the two `fused_moe_kernel` launches (plus the separate
// SiluAndMul and moe_sum) of
// vllm/model_executor/layers/fused_moe/fused_moe.py on the RDNA2 path. Used by
// the MTP drafter (Qwen4ExpMTP, E=128 local / 512 global, N=640, K=2560,
// top_k=10, EP4, fp16).
//
// Layout (the checkpoint layout, no conversion):
//   x        [M, K]            fp16
//   w13      [E, 2*N, K]       fp16, gate = rows [0, N), up = rows [N, 2N)
//   w2       [E, K, N]         fp16
//   topk_ids [M * top_k]       int32/int64, global expert ids
//   expert_map [global_E]      int32, global -> local slot, -1 = not local
//
// Both kernels are *pair* blocked: one block per (token, top-k slot), so no
// token sorting / `moe_align_block_size` and no intermediate cache1 is needed.
// At the served geometry the workload is a batch of GEMVs (M = 8..24 tokens,
// top_k 10, 128 local experts) that is bound by the weight bytes each routed
// expert must be read for -- ~1 pair per touched expert -- so pair blocking
// reads essentially the same bytes as an expert-grouped GEMM would.
//
// Kernel 1 (gate/up + SiLU) writes the activation for one pair directly;
// kernel 2 (down) writes the weighted per-pair down result into a
// `[M*topk, K]` slot cache (the runner's `workspace2`, the same buffer the
// Triton path calls `intermediate_cache3`), each slot rounded once to fp16.
// The top-k reduction is then the shared fp32 `moe_sum` op, so both stages
// round exactly where the Triton path rounds. Accumulating the routed
// contributions in the destination with a fp16 atomic (what this file used to
// do) is wrong twice over: it changes the summation to fp16, and the modular
// runner's destination buffer is a view of the very `workspace13` buffer that
// holds the activations the same kernel is still reading.
//
// Both stages reduce K (resp. N) with 16-byte vector loads and
// `__builtin_amdgcn_fdot2` (v_dot2_f32_f16, fp32 accumulator), i.e. 8 halves
// per lane per iteration; consecutive lanes cover consecutive 16B chunks, so
// every load is fully coalesced.
//
// Constraints enforced at the host boundary: fp16, contiguous, K % 8 == 0,
// N % 8 == 0, K % 4 == 0 (packed 4-half slot store).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>

#include "q_gemm_rdna2_common.cuh"

namespace vllm {
namespace moe_unquantized_rdna2 {

constexpr int kWarpSize = 32;
constexpr int kWarpThreads = 8 * kWarpSize;
// Output columns (resp. rows) per warp per pass.
constexpr int kColsPerWarp = 8;
constexpr int kRowsPerWarp = 4;

__device__ __forceinline__ int local_expert(int32_t global_id,
                                            const int32_t* expert_map) {
  return expert_map != nullptr ? expert_map[global_id] : global_id;
}

// ---------------------------------------------------------------------------
// Stage 1: act[p, j] = silu(w13[e, j] . x[m]) * (w13[e, N + j] . x[m])
//   grid = (num_pairs, ceil(N / (8 * kColsPerWarp))), block = 256
// ---------------------------------------------------------------------------
template <typename IdxT>
__global__ void gate_up_silu_kernel(const half* __restrict__ x,
                                    const half* __restrict__ w13,
                                    const IdxT* __restrict__ topk_ids,
                                    const int32_t* __restrict__ expert_map,
                                    half* __restrict__ act,
                                    const int num_pairs, const int topk,
                                    const int N, const int K) {
  const int p = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;

  const int local = local_expert(topk_ids[p], expert_map);
  if (local < 0) {
    return;
  }
  const int m = p / topk;
  const half* __restrict__ xe = x + (int64_t)m * K;
  const half* __restrict__ we = w13 + (int64_t)local * 2 * (int64_t)N * K;

  const int j0 = blockIdx.y * (kWarpThreads / kWarpSize) * kColsPerWarp +
                 warp * kColsPerWarp;
#pragma unroll 1
  for (int i = 0; i < kColsPerWarp; ++i) {
    const int j = j0 + i;
    if (j >= N) {
      break;
    }
    const half* __restrict__ wg = we + (int64_t)j * K;
    const half* __restrict__ wu = we + (int64_t)(N + j) * K;
    float g = 0.0f;
    float u = 0.0f;
    for (int k = lane * 8; k < K; k += kWarpSize * 8) {
      half2 xv[4], gv[4], uv[4];
      __builtin_memcpy(xv, xe + k, 16);
      __builtin_memcpy(gv, wg + k, 16);
      __builtin_memcpy(uv, wu + k, 16);
      const half* xs = reinterpret_cast<const half*>(xv);
      g += vllm::gptq_rdna2::dot22_8_f(gv, xs);
      u += vllm::gptq_rdna2::dot22_8_f(uv, xs);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      g += __shfl_xor(g, off, kWarpSize);
      u += __shfl_xor(u, off, kWarpSize);
    }
    if (lane == 0) {
      const float s = g / (1.0f + expf(-g));
      act[(int64_t)p * N + j] = __float2half_rn(s * u);
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2: slot_out[p, k] = weight * sum_j act[p, j] * w2[e, k, j]
//   grid = (num_pairs, ceil(K / (8 * kRowsPerWarp))), block = 256
// `slot_out` is the per-(token, top-k) down result, one row per pair; the
// caller reduces the top-k rows with the same fp32 `moe_sum` the Triton path
// uses. It must NOT alias `act` (the runner's `output` does: the modular
// runner hands back the first M*K elements of the very workspace13 buffer
// that holds `act`, and writing it here would corrupt activations that other
// pair blocks are still reading).
// ---------------------------------------------------------------------------
template <typename IdxT>
__global__ void down_kernel(const half* __restrict__ act,
                            const half* __restrict__ w2,
                            const IdxT* __restrict__ topk_ids,
                            const int32_t* __restrict__ expert_map,
                            const float* __restrict__ topk_weights,
                            const bool mul_topk_weight,
                            half* __restrict__ slot_out, const int num_pairs,
                            const int topk, const int N, const int K) {
  const int p = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;

  const int local = local_expert(topk_ids[p], expert_map);
  if (local < 0) {
    return;
  }
  const int k0 = blockIdx.y * (kWarpThreads / kWarpSize) * kRowsPerWarp +
                 warp * kRowsPerWarp;
  if (k0 >= K) {
    return;
  }
  const half* __restrict__ actp = act + (int64_t)p * N;
  const half* __restrict__ we = w2 + (int64_t)local * K * N;

  float acc[kRowsPerWarp];
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    acc[i] = 0.0f;
  }
  for (int j = lane * 8; j < N; j += kWarpSize * 8) {
    half2 av[4];
    __builtin_memcpy(av, actp + j, 16);
    const half* as = reinterpret_cast<const half*>(av);
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      half2 bv[4];
      __builtin_memcpy(bv, we + (int64_t)(k0 + i) * N + j, 16);
      acc[i] += vllm::gptq_rdna2::dot22_8_f(bv, as);
    }
  }
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[i] += __shfl_xor(acc[i], off, kWarpSize);
    }
  }
  if (lane == 0) {
    const float w = mul_topk_weight ? topk_weights[p] : 1.0f;
    half r[kRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      r[i] = __float2half_rn(acc[i] * w);
    }
    __builtin_memcpy(slot_out + (int64_t)p * K + k0, r, 8);
  }
}

template <typename IdxT>
void launch_gate_up_silu(const half* x, const half* w13, const IdxT* topk_ids,
                         const int32_t* expert_map, half* act, int64_t num_pairs,
                         int64_t topk, int64_t N, int64_t K,
                         hipStream_t stream) {
  const int j_tile = kWarpThreads / kWarpSize * kColsPerWarp;
  const dim3 grid((unsigned)num_pairs, (unsigned)((N + j_tile - 1) / j_tile));
  gate_up_silu_kernel<IdxT><<<grid, kWarpThreads, 0, stream>>>(
      x, w13, topk_ids, expert_map, act, (int)num_pairs, (int)topk, (int)N,
      (int)K);
}

template <typename IdxT>
void launch_down(const half* act, const half* w2, const IdxT* topk_ids,
                 const int32_t* expert_map, const float* topk_weights,
                 bool mul_topk_weight, half* slot_out, int64_t num_pairs,
                 int64_t topk, int64_t N, int64_t K, hipStream_t stream) {
  const int k_tile = kWarpThreads / kWarpSize * kRowsPerWarp;
  const dim3 grid((unsigned)num_pairs, (unsigned)((K + k_tile - 1) / k_tile));
  down_kernel<IdxT><<<grid, kWarpThreads, 0, stream>>>(
      act, w2, topk_ids, expert_map, topk_weights, mul_topk_weight, slot_out,
      (int)num_pairs, (int)topk, (int)N, (int)K);
}

const int32_t* expert_map_ptr(const std::optional<torch::Tensor>& expert_map) {
  return expert_map.has_value() && expert_map->defined()
             ? reinterpret_cast<const int32_t*>(expert_map->data_ptr())
             : nullptr;
}

void check_common(const torch::Tensor& x, int64_t N, int64_t K) {
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8");
}

}  // namespace moe_unquantized_rdna2
}  // namespace vllm

void moe_gate_up_silu_unquantized_rdna2(
    torch::Tensor x, torch::Tensor w13, torch::Tensor topk_ids,
    const std::optional<torch::Tensor>& expert_map, torch::Tensor act,
    int64_t topk, int64_t N, int64_t K) {
  using namespace vllm::moe_unquantized_rdna2;
  check_common(x, N, K);
  TORCH_CHECK(w13.scalar_type() == at::kHalf && w13.is_contiguous(),
              "w13 must be fp16 and contiguous");
  TORCH_CHECK(w13.size(1) == 2 * N && w13.size(2) == K, "w13 must be [E,2N,K]");
  TORCH_CHECK(act.scalar_type() == at::kHalf && act.is_contiguous() &&
                  act.size(0) == topk_ids.numel() && act.size(1) == N,
              "act must be fp16 [M*topk, N]");
  TORCH_CHECK(topk_ids.dim() == 1, "topk_ids must be flattened");
  TORCH_CHECK(!expert_map.has_value() || expert_map->scalar_type() == at::kInt,
              "expert_map must be int32");

  const int64_t num_pairs = topk_ids.numel();
  if (num_pairs == 0 || N == 0 || K == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int32_t* emap = expert_map_ptr(expert_map);
  if (topk_ids.scalar_type() == at::kInt) {
    launch_gate_up_silu<int32_t>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(w13.data_ptr()),
        reinterpret_cast<const int32_t*>(topk_ids.data_ptr()), emap,
        reinterpret_cast<half*>(act.data_ptr()), num_pairs, topk, N, K, stream);
  } else {
    TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "topk_ids must be int32/int64");
    launch_gate_up_silu<int64_t>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(w13.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()), emap,
        reinterpret_cast<half*>(act.data_ptr()), num_pairs, topk, N, K, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_down_unquantized_rdna2(
    torch::Tensor act, torch::Tensor w2, torch::Tensor topk_ids,
    const std::optional<torch::Tensor>& expert_map, torch::Tensor topk_weights,
    bool mul_topk_weight, torch::Tensor slot_out, int64_t topk, int64_t N,
    int64_t K) {
  using namespace vllm::moe_unquantized_rdna2;
  TORCH_CHECK(act.scalar_type() == at::kHalf && act.is_contiguous() &&
                  act.size(0) == topk_ids.numel() && act.size(1) == N,
              "act must be fp16 [M*topk, N]");
  TORCH_CHECK(w2.scalar_type() == at::kHalf && w2.is_contiguous(),
              "w2 must be fp16 and contiguous");
  TORCH_CHECK(w2.size(1) == K && w2.size(2) == N, "w2 must be [E,K,N]");
  TORCH_CHECK(slot_out.scalar_type() == at::kHalf && slot_out.is_contiguous() &&
                  slot_out.size(-1) == K &&
                  slot_out.numel() == topk_ids.numel() * K,
              "slot_out must be fp16 [M*topk, K]");
  TORCH_CHECK(K % 4 == 0,
              "K must be a multiple of 4 (packed fp16 slot store)");
  TORCH_CHECK(topk_ids.dim() == 1, "topk_ids must be flattened");
  TORCH_CHECK(!expert_map.has_value() || expert_map->scalar_type() == at::kInt,
              "expert_map must be int32");

  const int64_t num_pairs = topk_ids.numel();
  if (num_pairs == 0 || N == 0 || K == 0) {
    return;
  }
  const float* tw = nullptr;
  if (mul_topk_weight) {
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                    topk_weights.is_contiguous() &&
                    topk_weights.numel() == num_pairs,
                "topk_weights must be fp32 [M*topk]");
    tw = reinterpret_cast<const float*>(topk_weights.data_ptr());
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(slot_out));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int32_t* emap = expert_map_ptr(expert_map);
  if (topk_ids.scalar_type() == at::kInt) {
    launch_down<int32_t>(reinterpret_cast<const half*>(act.data_ptr()),
                         reinterpret_cast<const half*>(w2.data_ptr()),
                         reinterpret_cast<const int32_t*>(topk_ids.data_ptr()),
                         emap, tw, mul_topk_weight,
                         reinterpret_cast<half*>(slot_out.data_ptr()), num_pairs,
                         topk, N, K, stream);
  } else {
    TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "topk_ids must be int32/int64");
    launch_down<int64_t>(reinterpret_cast<const half*>(act.data_ptr()),
                         reinterpret_cast<const half*>(w2.data_ptr()),
                         reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
                         emap, tw, mul_topk_weight,
                         reinterpret_cast<half*>(slot_out.data_ptr()), num_pairs,
                         topk, N, K, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
