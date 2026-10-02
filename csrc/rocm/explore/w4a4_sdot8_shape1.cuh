// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Explore-only W4A4 shape-(1) device sketch (gfx1030).
//   W i4 × A i4 (both packed signed nibbles) → __builtin_amdgcn_sdot8
//   (v_dot8_i32_i4). Not dest. Not wired. Not W4A8 / sdot4. Not E2M1.
//
// Packing: one dword holds 8 consecutive K signed i4 for one row/col.
// One sdot8 per K-dword. Do not unpack to i8 and call sdot4.
// clamp=false. i32 through K; scales in the epilogue (host / caller).
//
// A is runtime-quantized (symmetric signed i4 {−8…7} seed) before
// launch. This header consumes packed A. No integer-W4A4 checkpoint.
//
// ConfigH lesson: K_STEP is the K actually consumed. Allowed
// values are 16 and 32 only.

#pragma once

#include <cstdint>

#if defined(__HIPCC__) && defined(__gfx1030__)
#define VLLM_EXPLORE_W4A4_SDOT8_GFX1030 1
#endif

namespace vllm {
namespace explore_w4a4_sdot8 {

// ConfigA-class: THREADS=256, N_PER_THREAD=4 → N_TILE=1024, M_TILE=16,
// LDS=0. K_STEP is a template (16 or 32).
template <int KStep_>
struct Shape1Config {
  static constexpr int THREADS = 256;
  static constexpr int N_PER_THREAD = 4;
  static constexpr int N_TILE = THREADS * N_PER_THREAD;
  static constexpr int M_TILE = 16;
  static constexpr int LDS = 0;
  static constexpr int K_STEP = KStep_;
  static_assert(K_STEP == 16 || K_STEP == 32,
                "shape (1): K_STEP must be 16 or 32");
  static_assert(K_STEP % 8 == 0, "shape (1): K_STEP must be a multiple of 8");
  static constexpr int DWORDS_PER_STEP = K_STEP / 8;
};

__host__ __device__ __forceinline__ int32_t sign_extend_nibble(
    uint32_t nibble) {
  return static_cast<int32_t>(nibble << 28) >> 28;
}

// LSB-first K-contiguous signed i4. q in [−8, 7].
__device__ __forceinline__ uint32_t pack_i4x8(const int32_t q[8]) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    packed |= (static_cast<uint32_t>(q[i]) & 0xF) << (4 * i);
  }
  return packed;
}

__device__ __forceinline__ int32_t sdot8_i4x8(uint32_t a_pack, uint32_t b_pack,
                                              int32_t acc) {
#if defined(VLLM_EXPLORE_W4A4_SDOT8_GFX1030)
  return __builtin_amdgcn_sdot8(static_cast<int32_t>(a_pack),
                                static_cast<int32_t>(b_pack), acc,
                                /*clamp=*/false);
#else
  (void)a_pack;
  (void)b_pack;
  return acc;
#endif
}

// CPU / device scalar oracle (TESTPLAN §3). Not the dest path.
__host__ __device__ __forceinline__ int32_t sdot8_ref(uint32_t a_pack,
                                                      uint32_t b_pack,
                                                      int32_t acc) {
  for (int i = 0; i < 8; ++i) {
    const int32_t av = sign_extend_nibble((a_pack >> (4 * i)) & 0xF);
    const int32_t bv = sign_extend_nibble((b_pack >> (4 * i)) & 0xF);
    acc += av * bv;
  }
  return acc;
}

// One sdot8 per K-dword. Both args are packed signed i4 (not i8).
__device__ __forceinline__ int32_t sdot8_from_i4_dwords(uint32_t a_dword,
                                                        uint32_t w_dword,
                                                        int32_t acc) {
  return sdot8_i4x8(a_dword, w_dword, acc);
}

}  // namespace explore_w4a4_sdot8
}  // namespace vllm
