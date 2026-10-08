// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DeepSeek-V4 q-norm + RoPE + fp8_ds_mla KV insert for AMD RDNA.
//
// RDNA counterpart of upstream's
// `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
// (csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu),
// which is written for bf16 activations. gfx1030 has no bf16 math and the
// RDNA MoE / attention kernels run in fp16, so this kernel accepts fp16 or
// bf16 q/kv and writes the *same* V4 cache row as upstream:
//
//   per paged-cache block (block_size tokens, stride k_cache.stride(0)):
//     [0,      bs*576):         token data: 448 fp8 e4m3fn NoPE + 64 bf16 RoPE
//     [bs*576, bs*576 + bs*8):  UE8M0 scales, 7 real (one per 64 NoPE dims)
//                               + 1 zero pad per token
//
// The RoPE dims are stored as bf16 regardless of the input dtype (that is
// what sparse_mla_decode_rdna2 and the upstream readers expect).
//
// Math (mirrors upstream for the variant DeepSeek-V4 calls: V4 row,
// head-major q, apply_q_norm / apply_q_rope runtime flags):
//   Q slot  h < num_heads_q:          optional weightless RMSNorm over 512
//                                     dims, optional GPT-J RoPE on dims
//                                     [448, 512), stored in the input dtype
//   Q slot  num_heads_q <= h < Hpad:  zero-filled (padded heads)
//   KV slot:                          GPT-J RoPE on dims [448, 512); NoPE
//                                     rounded to the input dtype, UE8M0
//                                     block-quantized to fp8 (scale =
//                                     2^ceil(log2(max(absmax,1e-4)/448)));
//                                     RoPE written as bf16 (RNE from fp32)
// Two deliberate exactness choices, so the cache row is byte-identical to a
// torch reference (and to upstream for bf16 inputs):
//   - RoPE uses non-contracted fp32 mul/add (__fmul_rn/__fadd_rn), no FMA.
//   - ceil(log2(.)) is computed from the fp32 bit pattern, not log2f.
//
// Layout: one wave32 per (token, slot), 16 consecutive dims per lane; 8
// waves per 256-thread block. KV slots of DP-padding tokens
// (token >= slot_mapping.size(0)) and slot_mapping == -1 are skipped.
// Launches on the current stream, allocates only the q output
// (torch::empty), no host syncs: safe under cudagraph capture.

#include <cstdint>
#include <type_traits>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>

#include "../common/arch.cuh"
#include "../common/convert.cuh"

namespace vllm {
namespace rdna {
namespace dsv4 {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  // 448
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;   // 7
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  // 8
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;  // 576
constexpr int kLanes = RdnaArch::kWaveSize;               // 32
constexpr int kElemsPerLane = kHeadDim / kLanes;          // 16
constexpr int kBlockThreads = 256;
constexpr int kWavesPerBlock = kBlockThreads / kLanes;  // 8
constexpr float kAbsmaxFloor = 1e-4f;

static_assert(kLanes == 32, "DSV4 KV insert assumes wave32");
static_assert(kElemsPerLane == 16, "two uint4 (8x16-bit) loads per lane");
static_assert(kQuantBlock / kElemsPerLane == 4, "4 lanes per quant block");

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int mask = kLanes / 2; mask > 0; mask >>= 1) {
    v += __shfl_xor(v, mask, kLanes);
  }
  return v;
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1, kLanes));
  v = fmaxf(v, __shfl_xor(v, 2, kLanes));
  return v;
}

template <typename In>
__device__ __forceinline__ void unpack16(uint4 const& a, uint4 const& b,
                                         float* out) {
  uint16_t const* pa = reinterpret_cast<uint16_t const*>(&a);
  uint16_t const* pb = reinterpret_cast<uint16_t const*>(&b);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i] = In::to_float(pa[i]);
    out[8 + i] = In::to_float(pb[i]);
  }
}

template <typename Out>
__device__ __forceinline__ void pack16_store(float const* in, uint8_t* dst) {
  uint4 a, b;
  uint16_t* pa = reinterpret_cast<uint16_t*>(&a);
  uint16_t* pb = reinterpret_cast<uint16_t*>(&b);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    pa[i] = Out::from_float(in[i]);
    pb[i] = Out::from_float(in[8 + i]);
  }
  reinterpret_cast<uint4*>(dst)[0] = a;
  reinterpret_cast<uint4*>(dst)[1] = b;
}

template <typename In, bool kApplyQNorm>
__global__ __launch_bounds__(kBlockThreads) void qnorm_rope_kv_insert_kernel(
    uint16_t const* __restrict__ q_in,         // [N, Hq, 512]
    uint16_t* __restrict__ q_out,              // [N, Hpad, 512]
    uint16_t const* __restrict__ kv_in,        // [N, 512]
    uint8_t* __restrict__ k_cache,             // [num_blocks, block_stride]
    int64_t const* __restrict__ slot_mapping,  // [num_tokens_insert]
    int64_t const* __restrict__ position_ids,  // [N]
    float const* __restrict__ cos_sin_cache,   // [max_pos, 64] cos || sin
    float const eps, int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const num_heads_q_padded,
    int const cache_block_size, int64_t const kv_block_stride,
    bool const apply_q_rope) {
  int const lane = threadIdx.x % kLanes;
  int const wave = threadIdx.x / kLanes;
  int64_t const global_wave =
      static_cast<int64_t>(blockIdx.x) * kWavesPerBlock + wave;
  int const slots_per_token = num_heads_q_padded + 1;
  int const token = static_cast<int>(global_wave / slots_per_token);
  int const slot = static_cast<int>(global_wave % slots_per_token);
  // Every branch below is uniform across the wave (one wave = one slot),
  // so early returns never split a cross-lane reduction.
  if (token >= num_tokens_full) return;

  bool const is_kv = slot == num_heads_q_padded;
  if (is_kv && token >= num_tokens_insert) return;  // DP padding row
  int const dim_base = lane * kElemsPerLane;

  if (!is_kv && slot >= num_heads_q) {  // padded Q head: zero-fill
    uint4 const zero = {0u, 0u, 0u, 0u};
    uint4* dst = reinterpret_cast<uint4*>(
        q_out +
        (static_cast<int64_t>(token) * num_heads_q_padded + slot) * kHeadDim +
        dim_base);
    dst[0] = zero;
    dst[1] = zero;
    return;
  }

  int64_t slot_id = -1;
  if (is_kv) {
    slot_id = slot_mapping[token];
    if (slot_id < 0) return;
  }

  uint16_t const* src =
      is_kv
          ? kv_in + static_cast<int64_t>(token) * kHeadDim + dim_base
          : q_in +
                (static_cast<int64_t>(token) * num_heads_q + slot) * kHeadDim +
                dim_base;
  float x[kElemsPerLane];
  unpack16<In>(reinterpret_cast<uint4 const*>(src)[0],
               reinterpret_cast<uint4 const*>(src)[1], x);

  if constexpr (kApplyQNorm) {
    if (!is_kv) {
      float sum_sq = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) sum_sq += x[i] * x[i];
      sum_sq = wave_sum(sum_sq);
      float const rms_rcp = rsqrtf(sum_sq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) x[i] *= rms_rcp;
    }
  }

  // GPT-J RoPE (interleaved pairs) on dims [448, 512). Lanes 28..31.
  bool const is_rope_lane = dim_base >= kNopeDim;
  if (is_rope_lane && (is_kv || apply_q_rope)) {
    int64_t const pos = position_ids[token];
    float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
    float const* sin_ptr = cos_ptr + kRopeDim / 2;
    int const half_base = (dim_base - kNopeDim) >> 1;
#pragma unroll
    for (int p = 0; p < kElemsPerLane / 2; p++) {
      float const c = cos_ptr[half_base + p];
      float const s = sin_ptr[half_base + p];
      float const xe = x[2 * p];
      float const xo = x[2 * p + 1];
      x[2 * p] = __fsub_rn(__fmul_rn(xe, c), __fmul_rn(xo, s));
      x[2 * p + 1] = __fadd_rn(__fmul_rn(xe, s), __fmul_rn(xo, c));
    }
  }

  if (!is_kv) {
    uint16_t* dst =
        q_out +
        (static_cast<int64_t>(token) * num_heads_q_padded + slot) * kHeadDim +
        dim_base;
    pack16_store<In>(x, reinterpret_cast<uint8_t*>(dst));
    return;
  }

  // ── KV: UE8M0 fp8 NoPE + bf16 RoPE into the paged V4 row ────────────────
  int64_t const block_idx = slot_id / cache_block_size;
  int64_t const pos_in_block = slot_id % cache_block_size;
  uint8_t* block_base = k_cache + block_idx * kv_block_stride;
  uint8_t* token_ptr = block_base + pos_in_block * kTokenDataBytes;
  uint8_t* scale_ptr =
      block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
      pos_in_block * kScaleBytesPerToken;

  // NoPE dims are rounded to the activation dtype before quantization
  // (upstream does the same with its bf16 activations).
  float local_absmax = 0.0f;
  if (!is_rope_lane) {
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      x[i] = In::round(x[i]);
      local_absmax = fmaxf(local_absmax, fabsf(x[i]));
    }
  }
  // All 32 lanes join the 4-lane reduction; the rope quad's result is unused.
  float const absmax = fmaxf(quad_max(local_absmax), kAbsmaxFloor);

  if (!is_rope_lane) {
    int const exponent = ceil_log2_exact(__fdiv_rn(absmax, kFp8E4m3Max));
    // 2^-exponent; exponent is in [-22, 128) for finite fp16/bf16 inputs.
    float const inv_scale =
        u32_as_float(static_cast<uint32_t>(127 - exponent) << 23);
    uint4 packed;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      bytes[i] = float_to_fp8_e4m3fn(x[i] * inv_scale);
    }
    *reinterpret_cast<uint4*>(token_ptr + dim_base) = packed;
    if ((lane & 3) == 0) scale_ptr[lane >> 2] = e8m0_from_exponent(exponent);
    if (lane == 0) scale_ptr[kNumQuantBlocks] = 0;
  } else {
    pack16_store<Bf16>(x, token_ptr + kNopeDim + (dim_base - kNopeDim) * 2);
  }
}

}  // namespace dsv4
}  // namespace rdna
}  // namespace vllm

// q_head_padded == 0 runs the KV insert alone and returns an empty tensor.
torch::Tensor dsv4_qnorm_rope_kv_insert_rdna(
    torch::Tensor const& q_in,           // [N, num_heads_q, 512] fp16/bf16
    torch::Tensor const& kv,             // [N, 512] same dtype as q_in
    torch::Tensor& k_cache,              // uint8, dim0 = paged block
    torch::Tensor const& slot_mapping,   // [num_tokens_insert] int64
    torch::Tensor const& position_ids,   // [N] int64
    torch::Tensor const& cos_sin_cache,  // [max_pos, 64] fp32
    int64_t q_head_padded, double eps, int64_t cache_block_size,
    bool apply_q_norm, bool apply_q_rope) {
  namespace d = vllm::rdna::dsv4;
  TORCH_CHECK(q_in.is_cuda() && q_in.is_contiguous() && q_in.dim() == 3 &&
                  q_in.size(2) == d::kHeadDim,
              "q_in must be contiguous [N, num_heads_q, 512] on the GPU");
  TORCH_CHECK(kv.is_cuda() && kv.is_contiguous() && kv.dim() == 2 &&
                  kv.size(1) == d::kHeadDim,
              "kv must be contiguous [N, 512] on the GPU");
  TORCH_CHECK(q_in.scalar_type() == kv.scalar_type(),
              "q_in and kv dtype must match");
  TORCH_CHECK(
      q_in.scalar_type() == at::kHalf || q_in.scalar_type() == at::kBFloat16,
      "q_in/kv must be float16 or bfloat16, got ", q_in.scalar_type());
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == at::kByte &&
                  k_cache.dim() >= 2,
              "k_cache must be a uint8 paged cache [num_blocks, ...]");
  TORCH_CHECK(slot_mapping.is_cuda() &&
                  slot_mapping.scalar_type() == at::kLong &&
                  slot_mapping.dim() == 1 && slot_mapping.is_contiguous(),
              "slot_mapping must be contiguous int64 [num_tokens]");
  TORCH_CHECK(position_ids.is_cuda() &&
                  position_ids.scalar_type() == at::kLong &&
                  position_ids.dim() == 1 && position_ids.is_contiguous(),
              "position_ids must be contiguous int64 [N]");
  TORCH_CHECK(
      cos_sin_cache.is_cuda() && cos_sin_cache.scalar_type() == at::kFloat &&
          cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == d::kRopeDim &&
          cos_sin_cache.is_contiguous(),
      "cos_sin_cache must be contiguous float32 [max_pos, 64]");

  int64_t const num_tokens_full = q_in.size(0);
  int64_t const num_tokens_insert = slot_mapping.size(0);
  TORCH_CHECK(
      kv.size(0) == num_tokens_full && position_ids.size(0) == num_tokens_full,
      "q/kv/position_ids row counts must match");
  TORCH_CHECK(num_tokens_insert <= num_tokens_full,
              "slot_mapping must not exceed q row count");
  int64_t const num_heads_q = q_in.size(1);
  TORCH_CHECK(q_head_padded == 0 || q_head_padded >= num_heads_q,
              "q_head_padded must be 0 or >= num_heads_q");
  TORCH_CHECK(cache_block_size > 0, "cache_block_size must be positive");

  int64_t const kv_block_stride = k_cache.stride(0);
  TORCH_CHECK(kv_block_stride >= cache_block_size * (d::kTokenDataBytes +
                                                     d::kScaleBytesPerToken),
              "k_cache block stride ", kv_block_stride,
              " is too small for block_size ", cache_block_size, " x 584 B");
  TORCH_CHECK(kv_block_stride % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % 16 == 0,
              "k_cache blocks must be 16-byte aligned");

  auto q_out = q_head_padded == 0
                   ? torch::empty({0}, q_in.options())
                   : torch::empty({num_tokens_full, q_head_padded, d::kHeadDim},
                                  q_in.options());
  if (num_tokens_full == 0) return q_out;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_in));
  auto stream = at::cuda::getCurrentCUDAStream();

  int64_t const total_waves = num_tokens_full * (q_head_padded + 1);
  int64_t const grid =
      (total_waves + d::kWavesPerBlock - 1) / d::kWavesPerBlock;
  TORCH_CHECK(grid <= INT32_MAX, "too many tokens for one launch");

  auto launch = [&](auto in_tag, auto norm_tag) {
    using In = decltype(in_tag);
    constexpr bool kNorm = decltype(norm_tag)::value;
    d::qnorm_rope_kv_insert_kernel<In, kNorm>
        <<<static_cast<int>(grid), d::kBlockThreads, 0, stream>>>(
            reinterpret_cast<uint16_t const*>(q_in.data_ptr()),
            q_head_padded == 0 ? nullptr
                               : reinterpret_cast<uint16_t*>(q_out.data_ptr()),
            reinterpret_cast<uint16_t const*>(kv.data_ptr()),
            k_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
            position_ids.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(),
            static_cast<float>(eps), static_cast<int>(num_tokens_full),
            static_cast<int>(num_tokens_insert), static_cast<int>(num_heads_q),
            static_cast<int>(q_head_padded), static_cast<int>(cache_block_size),
            kv_block_stride, apply_q_rope);
  };
  auto launch_norm = [&](auto in_tag) {
    if (apply_q_norm) {
      launch(in_tag, std::true_type{});
    } else {
      launch(in_tag, std::false_type{});
    }
  };
  if (q_in.scalar_type() == at::kHalf) {
    launch_norm(vllm::rdna::Fp16{});
  } else {
    launch_norm(vllm::rdna::Bf16{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q_out;
}
