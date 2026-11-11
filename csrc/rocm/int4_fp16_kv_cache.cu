// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kPrefillThreads = 128;
constexpr int kPrefillGroupLanes = 16;

__device__ __forceinline__ float scale_from_bits(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ int round_scaled(float x, float scale, float inv) {
  const float magnitude = fabsf(x);
  int code = static_cast<int>(floorf(magnitude * inv + 0.5f));
  // FP16 scale has six mantissa bits. These midpoint products are exact in
  // FP32; correcting an approximate reciprocal preserves half-away ties.
  if (magnitude < (static_cast<float>(code) - 0.5f) * scale) --code;
  if (magnitude >= (static_cast<float>(code) + 0.5f) * scale) ++code;
  return x < 0.0f ? -code : code;
}

__device__ __forceinline__ bool covers_range(float scale, float range,
                                             float residual) {
  const float capacity = scale * 15.0f;
  return scale >= 1.0e-6f &&
         (capacity > range || (capacity == range && residual <= 0.0f));
}

template <int Width = 32>
__device__ __forceinline__ int quantize_group(float x, uint16_t& scale_bits,
                                              int& zp, float second,
                                              float& scale, float& inv) {
  float lo = fminf(fminf(x, second), 0.0f);
  float hi = fmaxf(fmaxf(x, second), 0.0f);
#pragma unroll
  for (int distance = Width / 2; distance; distance >>= 1) {
    lo = fminf(lo, __shfl_down(lo, distance, Width));
    hi = fmaxf(hi, __shfl_down(hi, distance, Width));
  }
  lo = __shfl(lo, 0, Width);
  hi = __shfl(hi, 0, Width);
  const float larger = fmaxf(hi, -lo), smaller = fminf(hi, -lo);
  const float range = larger + smaller;
  // Two-sum retains a tiny negative endpoint even when FP32 hi-lo rounds to
  // an exact scale-grid boundary (for example hi=15, lo=-2^-24).
  const float residual = smaller - (range - larger);
  const float approx = fmaxf(range * (1.0f / 15.0f), 1.0e-6f);
  scale_bits = __half_as_ushort(__float2half_rn(approx)) & 0xfff0;
  scale = scale_from_bits(scale_bits);
  if (!covers_range(scale, range, residual)) {
    scale_bits += 16;
  } else if (scale_bits >= 16 &&
             covers_range(scale_from_bits(scale_bits - 16), range, residual)) {
    scale_bits -= 16;
  }
  scale = scale_from_bits(scale_bits);
  inv = __builtin_amdgcn_rcpf(scale);
  zp = min(15, max(0, round_scaled(-lo, scale, inv)));
  // A covered width can span 16 rounded codes only at opposite half ties
  // exactly 15 scale steps apart. The midpoint products are exact in FP32.
  if (range == 15.0f * scale && residual == 0.0f &&
      -lo == (static_cast<float>(zp) - 0.5f) * scale) {
    scale_bits += 16;
    scale = scale_from_bits(scale_bits);
    inv = __builtin_amdgcn_rcpf(scale);
    zp = min(15, max(0, round_scaled(-lo, scale, inv)));
  }
  return min(15, max(0, round_scaled(x, scale, inv) + zp));
}

// Four independent Wave32 groups per CTA; no LDS, atomics or grid barriers.
__global__ __launch_bounds__(128) void write_int4_fp16_block_kernel(
    const half* __restrict__ key, const half* __restrict__ value,
    uint8_t* __restrict__ cache, const int64_t* __restrict__ slots,
    int64_t tokens, int64_t heads, int64_t dim, int64_t num_blocks,
    int64_t block_size, int64_t key_s0, int64_t key_s1, int64_t key_s2,
    int64_t value_s0, int64_t value_s1, int64_t value_s2, int64_t cache_s0,
    int64_t cache_s1, int64_t cache_s2, int64_t cache_s3) {
  const int lane = threadIdx.x & 31;
  const int64_t wave =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int64_t groups = dim / 32;
  const int64_t groups_per_kind = tokens * heads * groups;
  if (wave >= 2 * groups_per_kind) return;
  const int64_t kind = wave / groups_per_kind;
  const int64_t local = wave % groups_per_kind;
  const int64_t group = local % groups;
  const int64_t head = (local / groups) % heads;
  const int64_t token = local / (groups * heads);
  const int64_t slot = slots[token];
  if (slot < 0 || slot / block_size >= num_blocks) return;
  const half* source = kind == 0 ? key : value;
  const int64_t s0 = kind == 0 ? key_s0 : value_s0;
  const int64_t s1 = kind == 0 ? key_s1 : value_s1;
  const int64_t s2 = kind == 0 ? key_s2 : value_s2;
  const float x =
      __half2float(source[token * s0 + head * s1 + (group * 32 + lane) * s2]);
  uint16_t scale_bits;
  int zp;
  float scale, inv;
  const int code = quantize_group(x, scale_bits, zp, 0.0f, scale, inv);
  const int even = __shfl(code, lane * 2, 32);
  const int odd = __shfl(code, lane * 2 + 1, 32);
  uint8_t* row = cache + kind * cache_s0 + (slot / block_size) * cache_s1 +
                 (slot % block_size) * cache_s2 + head * cache_s3;
  if (lane < 16) row[group * 16 + lane] = even | (odd << 4);
  if (lane == 0) {
    // Byte stores avoid alignment assumptions for padded cache page strides.
    row[dim / 2 + group * 2] = (scale_bits | zp) & 0xff;
    row[dim / 2 + group * 2 + 1] = (scale_bits | zp) >> 8;
  }
}

// Large writes amortize CTA scheduling and page/slot calculations over all
// heads of one token. Head dimension specialization removes integer division
// from the group-to-head mapping; four waves walk independent groups.
template <int Dim, int Threads, int Width>
__global__ __launch_bounds__(Threads) void write_int4_fp16_prefill_kernel(
    const half* __restrict__ key, const half* __restrict__ value,
    uint8_t* __restrict__ cache, const int64_t* __restrict__ slots,
    int64_t heads, int64_t num_blocks, int64_t block_size, int block_shift,
    int64_t key_s0, int64_t key_s1, int64_t key_s2, int64_t value_s0,
    int64_t value_s1, int64_t value_s2, int64_t cache_s0, int64_t cache_s1,
    int64_t cache_s2, int64_t cache_s3) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slots[token];
  if (slot < 0) return;
  const int64_t page =
      block_shift >= 0 ? slot >> block_shift : slot / block_size;
  const int64_t page_slot =
      block_shift >= 0 ? slot & (block_size - 1) : slot % block_size;
  if (page >= num_blocks) return;
  const int lane = threadIdx.x % Width, wave = threadIdx.x / Width;
  constexpr int groups = Dim / 32;
  for (int kind = 0; kind < 2; ++kind) {
    const half* source = kind == 0 ? key : value;
    const int64_t s0 = kind == 0 ? key_s0 : value_s0;
    const int64_t s1 = kind == 0 ? key_s1 : value_s1;
    const int64_t s2 = kind == 0 ? key_s2 : value_s2;
    uint8_t* token_cache =
        cache + kind * cache_s0 + page * cache_s1 + page_slot * cache_s2;
    for (int64_t item = wave; item < heads * groups; item += Threads / Width) {
      const int64_t head = item / groups;
      const int group = item % groups;
      const int64_t offset =
          token * s0 + head * s1 + (group * 32 + lane * (32 / Width)) * s2;
      const float x = __half2float(source[offset]);
      float second = 0.0f;
      if constexpr (Width == 16) second = __half2float(source[offset + s2]);
      uint16_t scale_bits;
      int zp;
      float scale, inv;
      const int code =
          quantize_group<Width>(x, scale_bits, zp, second, scale, inv);
      uint8_t* row = token_cache + head * cache_s3;
      if constexpr (Width == 16) {
        const int odd = min(15, max(0, round_scaled(second, scale, inv) + zp));
        row[group * 16 + lane] = code | (odd << 4);
      } else {
        const int even = __shfl(code, lane * 2, 32);
        const int odd = __shfl(code, lane * 2 + 1, 32);
        if (lane < 16) row[group * 16 + lane] = even | (odd << 4);
      }
      if (lane == 0) {
        row[Dim / 2 + group * 2] = (scale_bits | zp) & 0xff;
        row[Dim / 2 + group * 2 + 1] = (scale_bits | zp) >> 8;
      }
    }
  }
}

}  // namespace

// Cache record: [D/2 payload bytes, D/32 little-endian uint16 scale/zp words].
// Invalid slots are skipped; callers supply unique slots and finite FP16 input.
void reshape_and_cache_int4_fp16_rdna2(torch::Tensor key, torch::Tensor value,
                                       torch::Tensor cache,
                                       torch::Tensor slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && cache.is_cuda() &&
                  slot_mapping.is_cuda(),
              "INT4 FP16 KV writer requires HIP tensors");
  TORCH_CHECK(key.device() == value.device() &&
                  key.device() == cache.device() &&
                  key.device() == slot_mapping.device(),
              "KV tensors must share a device");
  TORCH_CHECK(key.scalar_type() == at::kHalf &&
                  value.scalar_type() == at::kHalf &&
                  cache.scalar_type() == at::kByte &&
                  slot_mapping.scalar_type() == at::kLong,
              "invalid INT4 FP16 KV dtypes");
  TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(),
              "key/value must have equal [tokens, heads, dim] shapes");
  const int64_t tokens = key.size(0), heads = key.size(1), dim = key.size(2);
  TORCH_CHECK(heads > 0 && dim > 0 && dim % 32 == 0,
              "head dimension must be a positive multiple of 32");
  const int64_t record_bytes = dim / 2 + 2 * (dim / 32);
  TORCH_CHECK(cache.dim() == 5 && cache.size(0) == 2 && cache.size(2) > 0 &&
                  cache.size(3) == heads && cache.size(4) == record_bytes,
              "cache must be [2, pages, block_size, heads, record_bytes]");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.numel() == tokens &&
                  slot_mapping.is_contiguous(),
              "slots must be contiguous int64[tokens]");
  const int64_t page_span = cache.size(2) * cache.stride(2);
  const bool side_outer = cache.stride(1) >= page_span &&
                          cache.stride(0) >= cache.size(1) * cache.stride(1);
  const bool page_outer =
      cache.stride(0) >= page_span && cache.stride(1) >= 2 * cache.stride(0);
  TORCH_CHECK(cache.stride(4) == 1 && cache.stride(3) >= record_bytes &&
                  cache.stride(2) >= heads * cache.stride(3) &&
                  (side_outer || page_outer),
              "cache records must not overlap");
  const at::cuda::OptionalCUDAGuard guard(device_of(key));
  const auto* props = at::cuda::getDeviceProperties(key.get_device());
  TORCH_CHECK(props->warpSize == 32, "INT4 FP16 KV writer requires Wave32");
  if (tokens == 0) return;
  const int64_t waves = 2 * tokens * heads * (dim / 32);
  const int64_t blocks = (waves + 3) / 4;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(),
              "KV writer grid is too large");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (tokens >= 32 && (dim == 128 || dim == 256)) {
    TORCH_CHECK(tokens <= std::numeric_limits<int>::max(),
                "KV writer grid is too large");
    const int64_t page_size = cache.size(2);
    const int block_shift =
        (page_size & (page_size - 1)) == 0
            ? __builtin_ctzll(static_cast<unsigned long long>(page_size))
            : -1;
#define LAUNCH_PREFILL(Dim)                                                   \
  write_int4_fp16_prefill_kernel<Dim, kPrefillThreads, kPrefillGroupLanes>    \
      <<<static_cast<unsigned>(tokens), kPrefillThreads, 0, stream>>>(        \
          reinterpret_cast<const half*>(key.data_ptr()),                      \
          reinterpret_cast<const half*>(value.data_ptr()),                    \
          cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(), heads, \
          cache.size(1), cache.size(2), block_shift, key.stride(0),           \
          key.stride(1), key.stride(2), value.stride(0), value.stride(1),     \
          value.stride(2), cache.stride(0), cache.stride(1), cache.stride(2), \
          cache.stride(3))
    if (dim == 128) {
      LAUNCH_PREFILL(128);
    } else {
      LAUNCH_PREFILL(256);
    }
#undef LAUNCH_PREFILL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  write_int4_fp16_block_kernel<<<static_cast<unsigned>(blocks), 128, 0,
                                 stream>>>(
      reinterpret_cast<const half*>(key.data_ptr()),
      reinterpret_cast<const half*>(value.data_ptr()),
      cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(), tokens,
      heads, dim, cache.size(1), cache.size(2), key.stride(0), key.stride(1),
      key.stride(2), value.stride(0), value.stride(1), value.stride(2),
      cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
