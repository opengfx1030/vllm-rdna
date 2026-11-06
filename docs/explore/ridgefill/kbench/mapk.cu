// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RidgeFill map GEMM variants for the gfx1030 micro-bench (libmapk.so):
//
//   hipcc --offload-arch=gfx1030 -O3 -fPIC -shared -o libmapk.so mapk.cu
//
// The kernels are the serve op's templates (csrc/rocm/ridgefill_map_rdna2.cuh);
// this file only instantiates tile configurations and exposes a C API that
// bench_map.py drives through ctypes, so a tiling change is measured without
// rebuilding vLLM.

#include <cstdio>

#include "../../../../csrc/rocm/ridgefill_map_rdna2.cuh"

namespace mapk = vllm::ridgefill_map_rdna2;

// Variant ids are listed by mapk_count() / mapk_info().
//
// X(name, dot, TM, TN, LM, WM, WN, BKB, RW, PL, ROT)
#define MAPK_VARIANTS(X)                                           \
  X(f16_8x8_w4x2_b128_p0_r1, kF16, 8, 8, 4, 4, 2, 64, 4, 0, 1)     \
  X(f16_8x8_w4x2_b128_p0_r0, kF16, 8, 8, 4, 4, 2, 64, 4, 0, 0)     \
  X(f16_8x8_w4x2_b128_p1_r1, kF16, 8, 8, 4, 4, 2, 64, 4, 1, 1)     \
  X(f16_8x8_w4x2_b64_p1_r1, kF16, 8, 8, 4, 4, 2, 64, 2, 1, 1)      \
  X(f16_8x8_w4x2_b64_p1_r0, kF16, 8, 8, 4, 4, 2, 64, 2, 1, 0)      \
  X(f16_8x8_l8_w2x4_b128_p0_r1, kF16, 8, 8, 8, 2, 4, 64, 4, 0, 1)  \
  X(f16_8x16_w4x2_b64_p0_r1, kF16, 8, 16, 4, 4, 2, 64, 2, 0, 1)    \
  X(f16_16x8_l8_w2x4_b64_p0_r1, kF16, 16, 8, 8, 2, 4, 64, 2, 0, 1) \
  X(f16_8x8_w2x2_b128_p0_r1, kF16, 8, 8, 4, 2, 2, 64, 4, 0, 1)     \
  X(i8_8x8_w4x2_b128_p0_r1, kI8, 8, 8, 4, 4, 2, 64, 4, 0, 1)       \
  X(i8_8x8_w4x2_b128_p0_r0, kI8, 8, 8, 4, 4, 2, 64, 4, 0, 0)       \
  X(i8_8x8_w4x2_b128_p1_r1, kI8, 8, 8, 4, 4, 2, 64, 4, 1, 1)       \
  X(i8_8x8_w4x2_b64_p1_r1, kI8, 8, 8, 4, 4, 2, 64, 2, 1, 1)        \
  X(i8_8x8_l8_w2x4_b128_p0_r1, kI8, 8, 8, 8, 2, 4, 64, 4, 0, 1)    \
  X(i8_8x16_w4x2_b64_p0_r1, kI8, 8, 16, 4, 4, 2, 64, 2, 0, 1)      \
  X(i8_16x8_l8_w2x4_b64_p0_r1, kI8, 16, 8, 8, 2, 4, 64, 2, 0, 1)   \
  X(i8_8x8_w2x2_b128_p0_r1, kI8, 8, 8, 4, 2, 2, 64, 4, 0, 1)

namespace variants {
#define MAPK_CFG(name, d, tm, tn, lm, wm, wn, bkb, rw, pl, r) \
  using name = mapk::Cfg<mapk::d, tm, tn, lm, wm, wn, bkb, rw, pl, r>;
MAPK_VARIANTS(MAPK_CFG)
#undef MAPK_CFG
}  // namespace variants

extern "C" {

int mapk_count() {
  int n = 0;
#define MAPK_COUNT(name, ...) ++n;
  MAPK_VARIANTS(MAPK_COUNT)
#undef MAPK_COUNT
  return n;
}

// Writes "name dot bm bn bkb threads" for variant id into buf.
int mapk_info(int id, char* buf, int len) {
  int n = 0;
#define MAPK_INFO(name, ...)                                                  \
  if (n++ == id) {                                                            \
    using C = variants::name;                                                 \
    return snprintf(buf, len, "%s %d %d %d %d %d", #name, C::D, C::BM, C::BN, \
                    C::BKB, C::THREADS);                                      \
  }
  MAPK_VARIANTS(MAPK_INFO)
#undef MAPK_INFO
  return -1;
}

// x/w: byte pointers; kb: bytes of K per row (k * elem size); n = rows of w.
int mapk_run(int id, const void* x, int64_t ldx, const void* w, int64_t ldw,
             int m, int n, int kb, int dout, const float* sx, const float* sw,
             const void* bias, void* out, hipStream_t stream) {
  int i = 0;
#define MAPK_RUN(name, ...)                                                 \
  if (i++ == id) {                                                          \
    using C = variants::name;                                               \
    if (kb % C::BKB || n % C::BN || dout % C::BN) return -2;                \
    return mapk::launch_map_gemm<C>(x, ldx, w, ldw, m, n, kb, dout, sx, sw, \
                                    static_cast<const __half*>(bias),       \
                                    static_cast<__half*>(out),              \
                                    stream) == hipSuccess                   \
               ? 0                                                          \
               : -3;                                                        \
  }
  MAPK_VARIANTS(MAPK_RUN)
#undef MAPK_RUN
  return -1;
}

// x [m][ldx] fp16, ldx in elements (mapk_run's ldx / ldw are bytes).
// had: 0 (plain per-row int8) or 128 (Hadamard-128 rotation first).
int mapk_quant_rows(const void* x, int64_t ldx, int m, int k, void* q,
                    float* scale, int had, hipStream_t stream) {
  if (k % 128 || k > mapk::kQuantMaxK) return -2;
  if (had != 0 && had != 128) return -2;
  return mapk::launch_quant_rows(static_cast<const __half*>(x), ldx, m, k,
                                 static_cast<int8_t*>(q), scale, had != 0,
                                 stream) == hipSuccess
             ? 0
             : -3;
}

}  // extern "C"
