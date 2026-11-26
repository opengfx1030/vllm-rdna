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
                                    half* __restrict__ act, const int num_pairs,
                                    const int topk, const int N, const int K) {
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
                         const int32_t* expert_map, half* act,
                         int64_t num_pairs, int64_t topk, int64_t N, int64_t K,
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

// ---------------------------------------------------------------------------
// Grouped FP16 prefill GEMM (BLOCK_M = 16, BN = 64, BK = 32, TM = 4, TN = 4)
// 64 threads (2 Wave32s, TY=4, TX=16), double-buffered LDS (11.5 KiB),
// v_dot2c_f32_f16, zero atomics, zero Triton.
//   A: [M_in, K] fp16 (row = slot / input_div_topk)
//   W: [E, N_out, K] fp16
//   C: [num_pairs, N_out] fp16 (row = slot)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(64) moe_gemm_unquantized_prefill_kernel(
    const half* __restrict__ A, const half* __restrict__ W,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, half* __restrict__ C, int num_pairs,
    int input_div_topk, int N_out, int K, bool mul_topk_weight) {
  constexpr int BM = 16, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int TX = 16, TY = 4, T = 64;
  constexpr int LD = BK + 4;  // 36 halves = 72 B (conflict-free)

  const int tb = blockIdx.x;
  if (tb * BM >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[tb];
  if (expert_id < 0) return;

  const int n0 = blockIdx.y * BN;
  if (n0 >= N_out) return;

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;

  __shared__ __attribute__((aligned(16))) half As[2][BM][LD];
  __shared__ __attribute__((aligned(16))) half Bs[2][BN][LD];
  __shared__ int sh_slot[BM];
  __shared__ int sh_arow[BM];

  if (tid < BM) {
    const int p = sorted_token_ids[tb * BM + tid];
    sh_slot[tid] = p;
    sh_arow[tid] = (p >= 0 && p < num_pairs) ? (p / input_div_topk) : 0;
  }
  __syncthreads();

  const half* we = W + (int64_t)expert_id * (int64_t)N_out * K;
  const int nk = K / BK;

  // A: 16 rows x 4 uint4 chunks = 64 chunks -> 1 load/thread
  const int a_r = tid >> 2;
  const int a_c = (tid & 3) << 3;
  const half* a_ptr = A + (int64_t)sh_arow[a_r] * K + a_c;
  const bool a_valid = (sh_slot[a_r] >= 0 && sh_slot[a_r] < num_pairs);

  // B: 64 rows x 4 uint4 chunks = 256 chunks -> 4 loads/thread (scalar vars
  // keep rb0..rb3 in VGPRs with zero private scratch).
  const int b_c = (tid & 3) << 3;
  const int br0 = tid >> 2;
  const int br1 = (tid + T) >> 2;
  const int br2 = (tid + 2 * T) >> 2;
  const int br3 = (tid + 3 * T) >> 2;
  const half* bp0 = we + (int64_t)min(n0 + br0, N_out - 1) * K + b_c;
  const half* bp1 = we + (int64_t)min(n0 + br1, N_out - 1) * K + b_c;
  const half* bp2 = we + (int64_t)min(n0 + br2, N_out - 1) * K + b_c;
  const half* bp3 = we + (int64_t)min(n0 + br3, N_out - 1) * K + b_c;

  uint4 ra, rb0, rb1, rb2, rb3;
  auto gload = [&](int kt) {
    const int k0 = kt * BK;
    ra = a_valid ? *reinterpret_cast<const uint4*>(a_ptr + k0)
                 : make_uint4(0u, 0u, 0u, 0u);
    rb0 = *reinterpret_cast<const uint4*>(bp0 + k0);
    rb1 = *reinterpret_cast<const uint4*>(bp1 + k0);
    rb2 = *reinterpret_cast<const uint4*>(bp2 + k0);
    rb3 = *reinterpret_cast<const uint4*>(bp3 + k0);
  };
  auto lstore = [&](int buf) {
    *reinterpret_cast<uint4*>(&As[buf][a_r][a_c]) = ra;
    *reinterpret_cast<uint4*>(&Bs[buf][br0][b_c]) = rb0;
    *reinterpret_cast<uint4*>(&Bs[buf][br1][b_c]) = rb1;
    *reinterpret_cast<uint4*>(&Bs[buf][br2][b_c]) = rb2;
    *reinterpret_cast<uint4*>(&Bs[buf][br3][b_c]) = rb3;
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  auto compute = [&](int buf) {
    using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
#pragma unroll 1
    for (int kk = 0; kk < BK; kk += 4) {
      uint2 af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const uint2*>(&As[buf][ty + TY * i][kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const uint2*>(&Bs[buf][tx + TX * j][kk]);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const f16x2 a0 = __builtin_bit_cast(f16x2, af[i].x);
        const f16x2 a1 = __builtin_bit_cast(f16x2, af[i].y);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const f16x2 b0 = __builtin_bit_cast(f16x2, bf[j].x);
          const f16x2 b1 = __builtin_bit_cast(f16x2, bf[j].y);
          acc[i][j] = __builtin_amdgcn_fdot2(a0, b0, acc[i][j], false);
          acc[i][j] = __builtin_amdgcn_fdot2(a1, b1, acc[i][j], false);
        }
      }
    }
  };

  gload(0);
  lstore(0);
  __syncthreads();
  int buf = 0;
  for (int kt = 1; kt < nk; ++kt) {
    gload(kt);
    compute(buf);
    buf ^= 1;
    lstore(buf);
    __syncthreads();
  }
  compute(buf);

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = ty + TY * i;
    const int p = sh_slot[r];
    if (p < 0 || p >= num_pairs) continue;
    const float w = mul_topk_weight ? topk_weights[p] : 1.0f;
    half* out_r = C + (int64_t)p * N_out;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + TX * j;
      if (col < N_out) out_r[col] = __float2half_rn(acc[i][j] * w);
    }
  }
}

}  // namespace moe_unquantized_rdna2
}  // namespace vllm

void moe_gemm_unquantized_prefill_rdna2(
    torch::Tensor x, torch::Tensor w, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    torch::Tensor out, int64_t num_pairs, int64_t input_div_topk,
    bool mul_topk_weight, torch::Tensor topk_weights) {
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(), "x fp16");
  TORCH_CHECK(w.scalar_type() == at::kHalf && w.is_contiguous(), "w fp16");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(),
              "out fp16");
  TORCH_CHECK(sorted_token_ids.is_contiguous() && expert_ids.is_contiguous() &&
                  num_tokens_post_padded.is_contiguous(),
              "routing metadata must be contiguous");
  TORCH_CHECK(!mul_topk_weight || (topk_weights.scalar_type() == at::kFloat &&
                                   topk_weights.is_contiguous()),
              "topk_weights must be contiguous float32");
  const int64_t N_out = w.size(1);
  const int64_t K = w.size(2);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  const int64_t num_blocks = expert_ids.size(0);
  if (num_pairs == 0 || num_blocks == 0 || N_out == 0 || K == 0) return;

  using namespace vllm::moe_unquantized_rdna2;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid((unsigned)num_blocks, (unsigned)((N_out + 63) / 64));
  const float* tw =
      mul_topk_weight ? reinterpret_cast<const float*>(topk_weights.data_ptr())
                      : nullptr;
  moe_gemm_unquantized_prefill_kernel<<<grid, 64, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()),
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), tw,
      reinterpret_cast<half*>(out.data_ptr()), (int)num_pairs,
      (int)input_div_topk, (int)N_out, (int)K, mul_topk_weight);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    TORCH_CHECK(topk_ids.scalar_type() == at::kLong,
                "topk_ids must be int32/int64");
    launch_gate_up_silu<int64_t>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(w13.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()), emap,
        reinterpret_cast<half*>(act.data_ptr()), num_pairs, topk, N, K, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_down_unquantized_rdna2(torch::Tensor act, torch::Tensor w2,
                                torch::Tensor topk_ids,
                                const std::optional<torch::Tensor>& expert_map,
                                torch::Tensor topk_weights,
                                bool mul_topk_weight, torch::Tensor slot_out,
                                int64_t topk, int64_t N, int64_t K) {
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
  TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 (packed fp16 slot store)");
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
                         reinterpret_cast<half*>(slot_out.data_ptr()),
                         num_pairs, topk, N, K, stream);
  } else {
    TORCH_CHECK(topk_ids.scalar_type() == at::kLong,
                "topk_ids must be int32/int64");
    launch_down<int64_t>(reinterpret_cast<const half*>(act.data_ptr()),
                         reinterpret_cast<const half*>(w2.data_ptr()),
                         reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
                         emap, tw, mul_topk_weight,
                         reinterpret_cast<half*>(slot_out.data_ptr()),
                         num_pairs, topk, N, K, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
