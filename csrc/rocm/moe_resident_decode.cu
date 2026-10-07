// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from the donor skinny MoE gate/activation and down-reduction kernels.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float resident_topk_weight(const void* w,
                                                      bool half_w, int idx) {
  return half_w ? __half2float(reinterpret_cast<const half*>(w)[idx])
                : reinterpret_cast<const float*>(w)[idx];
}

// Native resident-layout MoE decode. Packed weights are [E, K/8, N] int32
// words after the RDNA2 shuffle. Four waves compute 32 adjacent output columns
// with coalesced weight loads. The waves split K and reduce their FP32 partial
// sums in LDS.
__device__ __forceinline__ int moe_resident_local_expert(
    const void* ids, bool ids_i64, const int32_t* expert_map,
    const int map_size, const int local_experts, const int idx) {
  const int64_t global =
      ids_i64
          ? reinterpret_cast<const int64_t*>(ids)[idx]
          : static_cast<int64_t>(reinterpret_cast<const int32_t*>(ids)[idx]);
  if (global < 0) return -1;
  int expert;
  if (expert_map != nullptr) {
    if (global >= map_size) return -1;
    expert = expert_map[global];
  } else {
    if (global >= local_experts) return -1;
    expert = static_cast<int>(global);
  }
  return (expert >= 0 && expert < local_experts) ? expert : -1;
}

// Same exact half2 nibble conversion as the donor skinny kernel.
__device__ __forceinline__ half2 moe_resident_dequant_pair(uint32_t q, int pair,
                                                           half scale0,
                                                           half scale1) {
  const uint32_t bits = ((q >> (pair * 4)) & 0x000F000Fu) | 0x64006400u;
  const uint32_t bias = 0x64086408u;
  return __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits),
                         *reinterpret_cast<const half2*>(&bias)),
                 __halves2half2(scale0, scale1));
}

__device__ __forceinline__ float moe_resident_fdot2(half2 a, half2 b,
                                                    float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, false);
}

template <int GroupSize>
__global__ void moe_resident_w13_silu_gemv_(
    const half* __restrict__ input, const uint32_t* __restrict__ w13,
    const half* __restrict__ s13, const void* __restrict__ topk_ids,
    const bool ids_i64, const int32_t* __restrict__ expert_map,
    const int map_size, const int local_experts, half* __restrict__ act,
    const int K, const int intermediate, const int topk,
    const int runtime_group_size) {
  const int group_size = GroupSize ? GroupSize : runtime_group_size;
  const int lane = threadIdx.x & 31;
  const int m = blockIdx.z;
  const int route = blockIdx.y;
  const int n = blockIdx.x * 32 + lane;
  if (n >= intermediate) return;

  const int expert = moe_resident_local_expert(
      topk_ids, ids_i64, expert_map, map_size, local_experts, m * topk + route);
  half* const act_row =
      act + (static_cast<uint64_t>(m) * topk + route) * intermediate;
  if (expert < 0) {
    if (threadIdx.x < 32) act_row[n] = __float2half(0.f);
    return;
  }

  const int K8 = K / 8;
  const int groups = K / group_size;
  const int split = threadIdx.x / 32;
  const half* const x = input + static_cast<uint64_t>(m) * K;
  const uint32_t* const expert_w =
      w13 + static_cast<uint64_t>(expert) * K8 * (2 * intermediate);
  const half* const expert_s =
      s13 + static_cast<uint64_t>(expert) * groups * (2 * intermediate);
  float gate = 0.f;
  float up = 0.f;
  for (int kw = split; kw < K8; kw += 4) {
    const uint64_t row = static_cast<uint64_t>(kw) * (2 * intermediate);
    const uint32_t q_gate = expert_w[row + n];
    const uint32_t q_up = expert_w[row + intermediate + n];
    const int k0 = kw * 8;
    for (int j = 0; j < 8; j += 2) {
      const uint64_t scale_row0 =
          static_cast<uint64_t>((k0 + j) / group_size) * (2 * intermediate);
      const uint64_t scale_row1 =
          static_cast<uint64_t>((k0 + j + 1) / group_size) * (2 * intermediate);
      const half2 gate_pair = moe_resident_dequant_pair(
          q_gate, j / 2, expert_s[scale_row0 + n], expert_s[scale_row1 + n]);
      const half2 up_pair = moe_resident_dequant_pair(
          q_up, j / 2, expert_s[scale_row0 + intermediate + n],
          expert_s[scale_row1 + intermediate + n]);
      const half2 x_pair = __halves2half2(x[k0 + j], x[k0 + j + 1]);
      gate = moe_resident_fdot2(gate_pair, x_pair, gate);
      up = moe_resident_fdot2(up_pair, x_pair, up);
    }
  }

  __shared__ float gates[4][32], ups[4][32];
  gates[split][lane] = gate;
  ups[split][lane] = up;
  __syncthreads();
  if (threadIdx.x < 32) {
    gate = gates[0][lane] + gates[1][lane] + gates[2][lane] + gates[3][lane];
    up = ups[0][lane] + ups[1][lane] + ups[2][lane] + ups[3][lane];
    // Round both projections before SiLU and retain the routed activation in
    // the FP16 workspace consumed by the down projection.
    const float gate_f = __half2float(__float2half(gate));
    const float up_f = __half2float(__float2half(up));
    act_row[n] = __float2half(gate_f / (1.f + __expf(-gate_f)) * up_f);
  }
}

template <int GroupSize>
__global__ void moe_resident_w2_gemv_(
    const half* __restrict__ act, const uint32_t* __restrict__ w2,
    const half* __restrict__ s2, const void* __restrict__ topk_ids,
    const bool ids_i64, const int32_t* __restrict__ expert_map,
    const int map_size, const int local_experts,
    const void* __restrict__ topk_w, const bool w_is_half,
    half* __restrict__ output, const int intermediate, const int hidden,
    const int topk, const int runtime_group_size) {
  const int group_size = GroupSize ? GroupSize : runtime_group_size;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x / 32;
  __shared__ float routed_sums[4][32];
  const int m = blockIdx.z;
  const int h = blockIdx.x * 32 + lane;
  if (h >= hidden) return;

  const int N8 = intermediate / 8;
  const int groups = intermediate / group_size;
  const int split = threadIdx.x / 32;
  float total = 0.f;
  for (int route = 0; route < topk; ++route) {
    const int expert =
        moe_resident_local_expert(topk_ids, ids_i64, expert_map, map_size,
                                  local_experts, m * topk + route);
    if (expert < 0) continue;

    const uint32_t* const expert_w =
        w2 + static_cast<uint64_t>(expert) * N8 * hidden;
    const half* const expert_s =
        s2 + static_cast<uint64_t>(expert) * groups * hidden;
    const half* const x =
        act + (static_cast<uint64_t>(m) * topk + route) * intermediate;
    float acc = 0.f;
    for (int kw = split; kw < N8; kw += 4) {
      const uint32_t q = expert_w[static_cast<uint64_t>(kw) * hidden + h];
      const int k0 = kw * 8;
      for (int j = 0; j < 8; j += 2) {
        const half scale0 =
            expert_s[static_cast<uint64_t>((k0 + j) / group_size) * hidden + h];
        const half scale1 =
            expert_s[static_cast<uint64_t>((k0 + j + 1) / group_size) * hidden +
                     h];
        const half2 w = moe_resident_dequant_pair(q, j / 2, scale0, scale1);
        acc = moe_resident_fdot2(w, __halves2half2(x[k0 + j], x[k0 + j + 1]),
                                 acc);
      }
    }
    total += acc * resident_topk_weight(topk_w, w_is_half, m * topk + route);
  }

  // Keep the full K and route sum in FP32; the public output contract is FP16.
  routed_sums[wave][lane] = total;
  __syncthreads();
  if (wave == 0)
    output[static_cast<uint64_t>(m) * hidden + h] =
        __float2half(routed_sums[0][lane] + routed_sums[1][lane] +
                     routed_sums[2][lane] + routed_sums[3][lane]);
}

void moe_resident_int4_decode(const at::Tensor& input, const at::Tensor& w13,
                              const at::Tensor& w13_scale, const at::Tensor& w2,
                              const at::Tensor& w2_scale,
                              const at::Tensor& topk_weights,
                              const at::Tensor& topk_ids, at::Tensor& act_buf,
                              at::Tensor& output, const int64_t group_size,
                              const std::optional<at::Tensor>& expert_map) {
  const auto device = input.device();
  auto check_tensor = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), "moe_resident_int4_decode: ", name,
                " must be CUDA");
    TORCH_CHECK(tensor.device() == device,
                "moe_resident_int4_decode: all tensors must share a device");
    TORCH_CHECK(tensor.is_contiguous(), "moe_resident_int4_decode: ", name,
                " must be contiguous");
  };
  TORCH_CHECK(input.is_cuda(), "moe_resident_int4_decode: input must be CUDA");
  check_tensor(input, "input");
  check_tensor(w13, "w13");
  check_tensor(w13_scale, "w13_scale");
  check_tensor(w2, "w2");
  check_tensor(w2_scale, "w2_scale");
  check_tensor(topk_weights, "topk_weights");
  check_tensor(topk_ids, "topk_ids");
  check_tensor(act_buf, "act_buf");
  check_tensor(output, "output");
  if (expert_map.has_value()) check_tensor(*expert_map, "expert_map");

  TORCH_CHECK(input.dim() == 2 && w13.dim() == 3 && w13_scale.dim() == 3 &&
                  w2.dim() == 3 && w2_scale.dim() == 3 &&
                  topk_weights.dim() == 2 && topk_ids.dim() == 2 &&
                  act_buf.dim() == 3 && output.dim() == 2,
              "moe_resident_int4_decode: invalid tensor rank");
  TORCH_CHECK(group_size > 0 && group_size <= INT_MAX,
              "moe_resident_int4_decode: group_size must be in 1..INT_MAX");

  const int64_t M64 = input.size(0), K64 = input.size(1), E64 = w13.size(0);
  const int64_t intermediate64 = w13.size(2) / 2;
  const int64_t hidden64 = output.size(1), topk64 = topk_ids.size(1);
  TORCH_CHECK(M64 >= 1 && M64 <= 4, "moe_resident_int4_decode: M must be 1..4");
  TORCH_CHECK(E64 > 0 && E64 <= INT_MAX && K64 > 0 && K64 <= INT_MAX &&
                  intermediate64 > 0 && intermediate64 <= INT_MAX &&
                  hidden64 > 0 && hidden64 <= INT_MAX && topk64 > 0 &&
                  topk64 <= INT_MAX,
              "moe_resident_int4_decode: dimensions out of range");
  const int K = static_cast<int>(K64), E = static_cast<int>(E64);
  const int intermediate = static_cast<int>(intermediate64);
  const int hidden = static_cast<int>(hidden64),
            topk = static_cast<int>(topk64);
  const int group = static_cast<int>(group_size);
  TORCH_CHECK(K % 8 == 0 && intermediate % 32 == 0 && hidden % 32 == 0,
              "moe_resident_int4_decode: K must be divisible by 8; "
              "intermediate and hidden "
              "must be divisible by 32");
  TORCH_CHECK(K % group == 0 && intermediate % group == 0,
              "moe_resident_int4_decode: group_size must divide K and "
              "intermediate");
  TORCH_CHECK(w13.size(1) == K / 8 && w13.size(2) == 2 * intermediate &&
                  w2.size(0) == E && w2.size(1) == intermediate / 8 &&
                  w2.size(2) == hidden,
              "moe_resident_int4_decode: resident weight shape mismatch");
  TORCH_CHECK(w13_scale.size(0) == E && w13_scale.size(1) == K / group &&
                  w13_scale.size(2) == 2 * intermediate &&
                  w2_scale.size(0) == E &&
                  w2_scale.size(1) == intermediate / group &&
                  w2_scale.size(2) == hidden,
              "moe_resident_int4_decode: scale shape mismatch");
  TORCH_CHECK(topk_ids.size(0) == M64 && topk_weights.size(0) == M64 &&
                  topk_weights.size(1) == topk64,
              "moe_resident_int4_decode: routing shape mismatch");
  TORCH_CHECK(act_buf.size(0) == M64 && act_buf.size(1) == topk64 &&
                  act_buf.size(2) == intermediate && output.size(0) == M64 &&
                  output.size(1) == hidden,
              "moe_resident_int4_decode: workspace/output shape mismatch");
  TORCH_CHECK(w13.scalar_type() == at::kInt && w2.scalar_type() == at::kInt,
              "moe_resident_int4_decode: weights must be int32");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  w13_scale.scalar_type() == at::kHalf &&
                  w2_scale.scalar_type() == at::kHalf &&
                  act_buf.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf,
              "moe_resident_int4_decode: input, scales, workspace, and output "
              "must be fp16");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
      "moe_resident_int4_decode: topk_ids must be int32 or int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat ||
                  topk_weights.scalar_type() == at::kHalf,
              "moe_resident_int4_decode: topk_weights must be float32 or "
              "fp16");

  const int32_t* emap = nullptr;
  int map_size = 0;
  if (expert_map.has_value()) {
    TORCH_CHECK(expert_map->dim() == 1 && expert_map->numel() > 0 &&
                    expert_map->numel() <= INT_MAX &&
                    expert_map->scalar_type() == at::kInt,
                "moe_resident_int4_decode: expert_map must be non-empty "
                "contiguous int32");
    emap = reinterpret_cast<const int32_t*>(expert_map->const_data_ptr());
    map_size = static_cast<int>(expert_map->numel());
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 block(128);
  const dim3 grid13((intermediate + 31) / 32, topk, static_cast<int>(M64));
  const dim3 grid2((hidden + 31) / 32, 1, static_cast<int>(M64));
  const bool ids_i64 = topk_ids.scalar_type() == at::kLong;
  const bool weights_half = topk_weights.scalar_type() == at::kHalf;
  if (group == 128) {
    moe_resident_w13_silu_gemv_<128><<<grid13, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.const_data_ptr()),
        reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
        reinterpret_cast<const half*>(w13_scale.const_data_ptr()),
        topk_ids.const_data_ptr(), ids_i64, emap, map_size, E,
        reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, intermediate,
        topk, group);
    moe_resident_w2_gemv_<128><<<grid2, dim3(128), 0, stream>>>(
        reinterpret_cast<const half*>(act_buf.const_data_ptr()),
        reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
        reinterpret_cast<const half*>(w2_scale.const_data_ptr()),
        topk_ids.const_data_ptr(), ids_i64, emap, map_size, E,
        topk_weights.const_data_ptr(), weights_half,
        reinterpret_cast<half*>(output.mutable_data_ptr()), intermediate,
        hidden, topk, group);
  } else {
    moe_resident_w13_silu_gemv_<0><<<grid13, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.const_data_ptr()),
        reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
        reinterpret_cast<const half*>(w13_scale.const_data_ptr()),
        topk_ids.const_data_ptr(), ids_i64, emap, map_size, E,
        reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, intermediate,
        topk, group);
    moe_resident_w2_gemv_<0><<<grid2, dim3(128), 0, stream>>>(
        reinterpret_cast<const half*>(act_buf.const_data_ptr()),
        reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
        reinterpret_cast<const half*>(w2_scale.const_data_ptr()),
        topk_ids.const_data_ptr(), ids_i64, emap, map_size, E,
        topk_weights.const_data_ptr(), weights_half,
        reinterpret_cast<half*>(output.mutable_data_ptr()), intermediate,
        hidden, topk, group);
  }
}
