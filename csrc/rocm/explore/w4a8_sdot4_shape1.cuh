// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Explore-only W4A8 shape-(1) device sketch (gfx1030).
//   W i4 (K-contiguous nibbles) × A i8 → __builtin_amdgcn_sdot4
//   (v_dot4c_i32_i8). Not dest. Not wired. Not sdot8 / W4A4.
//
// Packing: one W dword holds 8 consecutive K nibbles for one N.
// Sign-extend each nibble to signed i8 in VGPR (not uint4−8).
// Two sdot4 per W dword. clamp=false. i32 through K; scales in
// the epilogue (host / caller).
//
// ConfigH lesson: K_STEP is the K actually consumed. Allowed
// values are 16 and 32 only.

#pragma once

#include <cstdint>

#if defined(__HIPCC__) && defined(__gfx1030__)
#define VLLM_EXPLORE_W4A8_SDOT4_GFX1030 1
#endif

namespace vllm {
namespace explore_w4a8_sdot4 {

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
  static constexpr int W_DWORDS_PER_STEP = K_STEP / 8;
};

__device__ __forceinline__ int32_t sign_extend_nibble(uint32_t nibble) {
  return static_cast<int32_t>(nibble << 28) >> 28;
}

// LSB-first K-contiguous nibbles → two packed i8 dwords (K0..K3, K4..K7).
__device__ __forceinline__ void nibble_dword_to_two_i8(
    uint32_t w_dword, int32_t& w_lo, int32_t& w_hi) {
  int32_t s0 = sign_extend_nibble(w_dword & 0xF);
  int32_t s1 = sign_extend_nibble((w_dword >> 4) & 0xF);
  int32_t s2 = sign_extend_nibble((w_dword >> 8) & 0xF);
  int32_t s3 = sign_extend_nibble((w_dword >> 12) & 0xF);
  int32_t s4 = sign_extend_nibble((w_dword >> 16) & 0xF);
  int32_t s5 = sign_extend_nibble((w_dword >> 20) & 0xF);
  int32_t s6 = sign_extend_nibble((w_dword >> 24) & 0xF);
  int32_t s7 = sign_extend_nibble((w_dword >> 28) & 0xF);
  w_lo = (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) |
         ((s3 & 0xFF) << 24);
  w_hi = (s4 & 0xFF) | ((s5 & 0xFF) << 8) | ((s6 & 0xFF) << 16) |
         ((s7 & 0xFF) << 24);
}

__device__ __forceinline__ int32_t sdot4_i8x4(int32_t a_pack, int32_t b_pack,
                                              int32_t acc) {
#if defined(VLLM_EXPLORE_W4A8_SDOT4_GFX1030)
  return __builtin_amdgcn_sdot4(a_pack, b_pack, acc, /*clamp=*/false);
#else
  (void)a_pack;
  (void)b_pack;
  return acc;
#endif
}

// Two sdot4 per W dword. a_lo/a_hi are A i8 packs for the same 8 K.
__device__ __forceinline__ int32_t sdot4_two_from_w_dword(uint32_t w_dword,
                                                          int32_t a_lo,
                                                          int32_t a_hi,
                                                          int32_t acc) {
  int32_t w_lo, w_hi;
  nibble_dword_to_two_i8(w_dword, w_lo, w_hi);
  acc = sdot4_i8x4(a_lo, w_lo, acc);
  acc = sdot4_i8x4(a_hi, w_hi, acc);
  return acc;
}

}  // namespace explore_w4a8_sdot4
}  // namespace vllm
