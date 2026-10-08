// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Compile-time RDNA architecture traits shared by the csrc/rocm/rdna/**
// kernels.
//
// The device pass of each offload target defines exactly one `__gfxNNNN__`
// macro; `RdnaArch` folds it to a set of constexpr feature flags so a kernel
// can branch on capabilities with `if constexpr` instead of scattering
// `#if defined(__gfx...)` blocks. The host pass (and any non-RDNA target in a
// multi-arch build) sees `RdnaGen::kNone` with conservative defaults; host
// code that needs the runtime device generation uses `rdna_gen_from_arch()`.
//
//   gen      targets               wave  dot2_f16  wmma  fp8_wmma
//   kRdna2   gfx1030..gfx1036      32    yes       no    no
//   kRdna3   gfx1100..gfx1103      32    yes       yes   no
//   kRdna35  gfx1150..gfx1153      32    yes       yes   no
//   kRdna4   gfx1200, gfx1201      32    yes       yes   yes
//
// All RDNA kernels in this tree are written for wave32 (hipcc's default on
// gfx10+). `kWaveSize` is a static fact, not a tuning knob.

#pragma once

#include <cstring>

namespace vllm {
namespace rdna {

enum class RdnaGen : int { kNone = 0, kRdna2, kRdna3, kRdna35, kRdna4 };

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__)
  #define VLLM_RDNA_DEVICE_GEN 2
#elif defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
    defined(__gfx1153__)
  #define VLLM_RDNA_DEVICE_GEN 35
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__)
  #define VLLM_RDNA_DEVICE_GEN 3
#elif defined(__gfx1200__) || defined(__gfx1201__)
  #define VLLM_RDNA_DEVICE_GEN 4
#else
  #define VLLM_RDNA_DEVICE_GEN 0
#endif

template <RdnaGen G>
struct RdnaArchTraits {
  static constexpr RdnaGen kGen = G;
  static constexpr bool kIsRdna = G != RdnaGen::kNone;
  static constexpr int kWaveSize = 32;
  // V_DOT2_F32_F16 (gfx1030 and newer RDNA).
  static constexpr bool kHasDot2F16 = kIsRdna;
  // V_DOT2_F32_BF16 (RDNA3+; gfx1030 lacks it).
  static constexpr bool kHasDot2Bf16 = kIsRdna && G != RdnaGen::kRdna2;
  // V_WMMA_* f16/bf16/iu8 (RDNA3, RDNA3.5, RDNA4).
  static constexpr bool kHasWmma = kIsRdna && G != RdnaGen::kRdna2;
  // V_WMMA_*_FP8 / BF8 and hardware fp8 conversions (RDNA4).
  static constexpr bool kHasFp8Wmma = G == RdnaGen::kRdna4;
  // LDS per workgroup available to a kernel (WGP mode on RDNA).
  static constexpr int kLdsBytes = 64 * 1024;
};

#if VLLM_RDNA_DEVICE_GEN == 2
using RdnaArch = RdnaArchTraits<RdnaGen::kRdna2>;
#elif VLLM_RDNA_DEVICE_GEN == 3
using RdnaArch = RdnaArchTraits<RdnaGen::kRdna3>;
#elif VLLM_RDNA_DEVICE_GEN == 35
using RdnaArch = RdnaArchTraits<RdnaGen::kRdna35>;
#elif VLLM_RDNA_DEVICE_GEN == 4
using RdnaArch = RdnaArchTraits<RdnaGen::kRdna4>;
#else
using RdnaArch = RdnaArchTraits<RdnaGen::kNone>;
#endif

// Host-side mapping from a `gcnArchName` string (e.g. "gfx1030" or
// "gfx1100:sramecc-:xnack-") to the RDNA generation.
inline RdnaGen rdna_gen_from_arch(const char* gcn_arch_name) {
  if (gcn_arch_name == nullptr) return RdnaGen::kNone;
  auto starts = [&](const char* p) {
    return std::strncmp(gcn_arch_name, p, std::strlen(p)) == 0;
  };
  if (starts("gfx103")) return RdnaGen::kRdna2;
  if (starts("gfx115")) return RdnaGen::kRdna35;
  if (starts("gfx110")) return RdnaGen::kRdna3;
  if (starts("gfx120")) return RdnaGen::kRdna4;
  return RdnaGen::kNone;
}

}  // namespace rdna
}  // namespace vllm
