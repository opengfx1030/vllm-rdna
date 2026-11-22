// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Weight-only quantized decode GEMV for the dense linears of Inkling-2-Mini on
// RDNA2 (gfx1030: wave32, v_dot2_f32_f16, no fp8 / int8 matrix hardware):
// int8 or int4 weights, one fp16 scale per group of K, fp16 activations, fp32
// accumulation. Up to 4 activation rows (decode, DSpark verification); the
// fp16 `inkling2_linear_*` ops keep the larger batches and the prefill.
//
//   inkling2_qlinear_grouped   S segments (own X, rows of the packed weight,
//   out,
//                              epilogue) in ONE launch: the attention input
//                              pack and o_proj.
//   inkling2_qlinear_lm_head   the same with the final softcap, the vocab mask
//                              and the fused per-block argmax partial.
//
// Layout (offline, qlinear_pack.py): a TILE is R weight rows; a lane of the
// wave owns row (lane % R) and the K quarter (lane / R) of KQ = 32 / R; its
// 16-byte weight chunk c sits at ((tile * nc + c) * 32 + lane) * 16, so the 32
// lanes of a load read 512 contiguous bytes and no lane ever needs a cross-lane
// reduction except the final KQ-way one. A chunk holds CK = 16 (int8) or 32
// (int4) consecutive K elements of the lane's row. Scales are laid out
// [tile][iteration][lane][SV] (SV groups per iteration).
//
// Dequantisation is exact and integer-only: an integer u in an fp16 mantissa
// below 1024 (0x6400 | u) is the number 1024 + u; subtracting 1024 + zero point
// leaves the signed weight exactly as an fp16 value (int4: the "high" nibbles
// are taken in place, 16 * u, and corrected by 1/16 after the dot product).
// All accumulation is fp32 (v_dot2_f32_f16), the group scale multiplies the
// fp32 partial sum.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

// Which kernel instances this translation unit builds: I2Q_ENABLED(BITS, GC, R,
// KQ, MT, CPI, WV, ARG) over the cross product of inst_at(). The
// micro-benchmark and the mutant runs define their own filter before including
// this file.
#ifndef I2Q_ENABLED
  #define I2Q_ENABLED(BITS, GC, R, KQ, MT, CPI, WV, ARG) \
    inkling2_qlinear::prod_instance(BITS, GC, R, MT, CPI, WV, ARG)
#endif

namespace inkling2_qlinear {

constexpr int kMaxSeg = 8;
[[maybe_unused]] constexpr int kActNone = 0;
[[maybe_unused]] constexpr int kActSigmoid = 1;
[[maybe_unused]] constexpr int kActTanh = 2;
[[maybe_unused]] constexpr int kActSoftcap = 3;
constexpr int kHeadWaves = 8;  // waves per block of the lm_head kernels

// The instances of the serving build (bits, scale group in chunks, rows per
// tile, activation rows, chunks per iteration, waves per block, argmax
// partials), the fastest of the sweep in
// docs/rdna2/inkling2-qlinear-kernels.md: R = 32 is the lm_head (with and
// without argmax partials), R = 8 the int8 attention input pack and o_proj, R =
// 8 / 4 the int4 attention input pack / o_proj.
constexpr bool prod_instance(int bits, int gc, int R, int mt, int cpi, int wv,
                             bool arg) {
  const bool fmt =
      (bits == 8 && gc == 4) || (bits == 4 && (gc == 1 || gc == 2));
  if (!fmt || wv != kHeadWaves || (mt != 1 && mt != 2 && mt != 4)) return false;
  if (R == 32) return cpi == 8;
  if (arg) return false;
  if (bits == 8) return R == 8 && cpi == 8;
  return cpi == 4 && (R == 8 || R == 4);
}

struct QSeg {
  const void* x;  // [M, K] fp16, row stride ldx elements
  void* out;      // [M, n] row stride ldo elements (null: no output)
  int ldx;
  int ldo;
  int n;        // weight rows of the segment
  int n_valid;  // rows >= n_valid are written as -inf
  int tile0;    // first packed tile of the segment
  int ntiles;
  int blk0;  // first block of the segment
  int nblk;  // blocks of the segment
  int act;
  int out_f32;
  float p;  // softcap
};

struct QParams {
  QSeg seg[kMaxSeg];
  const uint8_t* w;
  const _Float16* sc;
  int nseg;
  int M;
  int K;
  int nc;     // 16-byte chunks per lane
  int niter;  // iterations per lane
  int write_out;
  int nparts;
  int col_base;
  float* part_val;  // [M, nparts]
  int* part_idx;    // [M, nparts]
};

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using u32x4 = unsigned __attribute__((ext_vector_type(4)));

// fp32 a.x*b.x + a.y*b.y + c (v_dot2_f32_f16)
__device__ __forceinline__ float dot2u(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
}

// packed fp16 a - b (v_pk_add_f16)
__device__ __forceinline__ unsigned pk_sub(unsigned a, unsigned b) {
  return __builtin_bit_cast(
      unsigned, __builtin_bit_cast(f16x2, a) - __builtin_bit_cast(f16x2, b));
}

// fp16 pair (1024 + u0, 1024 + u1) from two bytes of w
__device__ __forceinline__ unsigned bytes_to_h2(unsigned w, unsigned sel) {
  return __builtin_amdgcn_perm(0x64646464u, w, sel);
}

__device__ __forceinline__ float epilogue(float v, int act, float p) {
  switch (act) {
    case kActSigmoid:
      return 1.0f / (1.0f + expf(-v));
    case kActTanh:
      return tanhf(v);
    case kActSoftcap:
      return tanhf(v / p) * p;
    default:
      return v;
  }
}

// ---- one 16-byte chunk against MT activation rows --------------------------
// int4: 4 words of 8 nibbles = 32 consecutive K elements; nibble i of word j
// holds element 8j + {0, 2, 4, 6, 1, 3, 5, 7}[i] (see qlinear_pack.py), so that
//   (w & 0x000F000F)        -> elements (8j, 8j+1)     unscaled
//   (w & 0x00F000F0)        -> elements (8j+2, 8j+3)   times 16
//   (w >> 8 & 0x000F000F)   -> elements (8j+4, 8j+5)   unscaled
//   (w >> 8 & 0x00F000F0)   -> elements (8j+6, 8j+7)   times 16
// and every half2 is one dword of the LDS activation vector.
template <int MT>
__device__ __forceinline__ void chunk_i4(const u32x4 wv, const char* xl,
                                         int mstride, float (&plo)[MT],
                                         float (&phi)[MT]) {
  const unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
  unsigned h0[4], h1[4], h2[4], h3[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned w = wd[j], w8 = wd[j] >> 8;
    h0[j] = pk_sub((w & 0x000F000Fu) | 0x64006400u, 0x64086408u);  // - 1032
    h1[j] = pk_sub((w & 0x00F000F0u) | 0x64006400u, 0x64806480u);  // - 1152
    h2[j] = pk_sub((w8 & 0x000F000Fu) | 0x64006400u, 0x64086408u);
    h3[j] = pk_sub((w8 & 0x00F000F0u) | 0x64006400u, 0x64806480u);
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 x =
          *reinterpret_cast<const uint4*>(xl + m * mstride + j * 16);
      plo[m] = dot2u(h0[j], x.x, plo[m]);
      phi[m] = dot2u(h1[j], x.y, phi[m]);
      plo[m] = dot2u(h2[j], x.z, plo[m]);
      phi[m] = dot2u(h3[j], x.w, phi[m]);
    }
  }
}

// int8: 4 words of 4 bytes = 16 consecutive K elements, offset binary (q + 128)
template <int MT>
__device__ __forceinline__ void chunk_i8(const u32x4 wv, const char* xl,
                                         int mstride, float (&p)[MT]) {
  const unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
  unsigned ha[4], hb[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    ha[j] = pk_sub(bytes_to_h2(wd[j], 0x04010400u), 0x64806480u);  // - 1152
    hb[j] = pk_sub(bytes_to_h2(wd[j], 0x04030402u), 0x64806480u);
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    const uint4 x0 = *reinterpret_cast<const uint4*>(xl + m * mstride);
    const uint4 x1 = *reinterpret_cast<const uint4*>(xl + m * mstride + 16);
    p[m] = dot2u(ha[0], x0.x, p[m]);
    p[m] = dot2u(hb[0], x0.y, p[m]);
    p[m] = dot2u(ha[1], x0.z, p[m]);
    p[m] = dot2u(hb[1], x0.w, p[m]);
    p[m] = dot2u(ha[2], x1.x, p[m]);
    p[m] = dot2u(hb[2], x1.y, p[m]);
    p[m] = dot2u(ha[3], x1.z, p[m]);
    p[m] = dot2u(hb[3], x1.w, p[m]);
  }
}

__device__ __forceinline__ float h2f(unsigned v, int hi) {
  const f16x2 t = __builtin_bit_cast(f16x2, v);
  return static_cast<float>(hi ? t.y : t.x);
}

__device__ __forceinline__ u32x4 load_w(const uint8_t* p) {
  #ifdef I2Q_NONTEMPORAL
  return __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
  #else
  return *reinterpret_cast<const u32x4*>(p);
  #endif
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// BITS: 4 / 8. GC: chunks per scale group (group = GC * CK elements). R rows x
// KQ quarters per wave. MT activation rows. CPI chunks per iteration (loads in
// flight, CPI % GC == 0). WV waves per block. ARG: lm_head argmax partials
// (R = 32, WV = 8).
// ---------------------------------------------------------------------------
template <int BITS, int GC, int R, int KQ, int MT, int CPI, int WV, bool ARG>
__global__ void __launch_bounds__(WV * 32) qgemv_kernel(const QParams p) {
#ifdef __HIP__RDNA2__
  static_assert(R * KQ == 32 && (R & (R - 1)) == 0, "tile shape");
  static_assert(BITS == 4 || BITS == 8, "bits");
  static_assert(CPI % GC == 0, "CPI % GC");
  static_assert(!ARG || (R == 32 && KQ == 1), "argmax needs R = 32");
  constexpr int CK = BITS == 4 ? 32 : 16;  // K elements per chunk
  constexpr int SV = CPI / GC;             // scale groups per iteration
  static_assert(SV == 1 || SV == 2 || SV == 4 || SV == 8, "SV");
  constexpr int NT = WV * 32;  // threads per block

  extern __shared__ __attribute__((aligned(16))) char smem[];
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int r = lane & (R - 1);
  const int kq = lane / R;

  int s = 0;
  #pragma unroll
  for (int i = 1; i < kMaxSeg; ++i)
    if (i < p.nseg && static_cast<int>(blockIdx.x) >= p.seg[i].blk0) s = i;
  const QSeg& sg = p.seg[s];
  const int lb = blockIdx.x - sg.blk0;

  const int K = p.K, M = p.M, nc = p.nc, niter = p.niter;
  const int qB = nc * CK * 2;  // activation bytes of one K quarter
  // quarters are padded 16 bytes apart so that the KQ distinct 16-byte reads of
  // a wave land on distinct LDS banks
  const int sqB = qB + (KQ > 1 ? 16 : 0);
  const int rowB = sqB * KQ;

  // ---- tiles of this wave: (lb + i * nblk) * WV + wave
  const int stride_t = sg.nblk * WV;
  const int t0 = lb * WV + wave;
  const int ntl =
      t0 < sg.ntiles ? (sg.ntiles - t0 + stride_t - 1) / stride_t : 0;

  const size_t wlane = static_cast<size_t>(lane) * 16;
  auto wptr = [&](int ti, int it) {
    return p.w + ((static_cast<size_t>(sg.tile0 + t0 + ti * stride_t) * nc +
                   static_cast<size_t>(it) * CPI) *
                      512 +
                  wlane);
  };
  auto sptr = [&](int ti, int it) {
    return p.sc +
           ((static_cast<size_t>(sg.tile0 + t0 + ti * stride_t) * niter + it) *
                32 +
            lane) *
               SV;
  };

  u32x4 wcur[CPI], wnxt[CPI];
  unsigned scur[(SV + 1) / 2], snxt[(SV + 1) / 2];
  auto load_item = [&](int ti, int it, u32x4(&wv)[CPI],
                       unsigned (&sv)[(SV + 1) / 2]) {
    const uint8_t* wp = wptr(ti, it);
  #pragma unroll
    for (int u = 0; u < CPI; ++u) {
      u32x4 t = {0u, 0u, 0u, 0u};
      if (it * CPI + u < nc) t = load_w(wp + u * 512);
      wv[u] = t;
    }
    const _Float16* sp = sptr(ti, it);
    if constexpr (SV == 1) {
      sv[0] = *reinterpret_cast<const unsigned short*>(sp);
    } else if constexpr (SV == 2) {
      sv[0] = *reinterpret_cast<const unsigned*>(sp);
    } else if constexpr (SV == 4) {
      const uint2 t = *reinterpret_cast<const uint2*>(sp);
      sv[0] = t.x;
      sv[1] = t.y;
    } else {
      const uint4 t = *reinterpret_cast<const uint4*>(sp);
      sv[0] = t.x;
      sv[1] = t.y;
      sv[2] = t.z;
      sv[3] = t.w;
    }
  };

  if (ntl > 0) load_item(0, 0, wcur, scur);

  // ---- stage the activations (fp16, MT rows, zero rows beyond M) in LDS
  {
    const uint4* xg = reinterpret_cast<const uint4*>(sg.x);
    const size_t ldx8 = static_cast<size_t>(sg.ldx) >> 3;
    const int nv = K >> 3, vq = qB >> 4;
  #pragma unroll
    for (int m = 0; m < MT; ++m)
      for (int v = tid; v < nv; v += NT) {
        uint4 t = make_uint4(0u, 0u, 0u, 0u);
        if (m < M) t = xg[static_cast<size_t>(m) * ldx8 + v];
        const int q = v / vq;
        *reinterpret_cast<uint4*>(smem + m * rowB + q * sqB +
                                  (v - q * vq) * 16) = t;
      }
  }
  __syncthreads();

  const char* xlane = smem + kq * sqB;
  float Y[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) Y[m] = 0.0f;
  [[maybe_unused]] float bv[MT];
  [[maybe_unused]] int bi[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    bv[m] = -__builtin_inff();
    bi[m] = 0;
  }

  int ti = 0, it = 0;
  while (ti < ntl) {
    int ti2 = ti, it2 = it + 1;
    if (it2 == niter) {
      it2 = 0;
      ti2 = ti + 1;
    }
    if (ti2 < ntl) load_item(ti2, it2, wnxt, snxt);

    // ---- compute iteration `it`
    float plo[MT], phi[MT];
  #pragma unroll
    for (int m = 0; m < MT; ++m) plo[m] = phi[m] = 0.0f;
  #pragma unroll
    for (int u = 0; u < CPI; ++u) {
      const int c = it * CPI + u;
      if (c < nc) {
        const char* xl = xlane + c * (CK * 2);
        if constexpr (BITS == 4)
          chunk_i4<MT>(wcur[u], xl, rowB, plo, phi);
        else
          chunk_i8<MT>(wcur[u], xl, rowB, plo);
      }
      if ((u % GC) == GC - 1) {
        const unsigned sw = scur[(u / GC) / 2];
        const float sc = h2f(sw, (u / GC) & 1);
  #pragma unroll
        for (int m = 0; m < MT; ++m) {
          if constexpr (BITS == 4)
            Y[m] = fmaf(sc, fmaf(phi[m], 0.0625f, plo[m]), Y[m]);
          else
            Y[m] = fmaf(sc, plo[m], Y[m]);
          plo[m] = 0.0f;
          phi[m] = 0.0f;
        }
      }
    }

    if (it + 1 == niter) {
      // ---- tile done: KQ-way reduction, epilogue, store / argmax
      if constexpr (KQ > 1) {
  #pragma unroll
        for (int m = 0; m < MT; ++m)
  #pragma unroll
          for (int off = R; off < 32; off <<= 1) Y[m] += __shfl_xor(Y[m], off);
      }
      const int row = (t0 + ti * stride_t) * R + r;
      if (kq == 0 && row < sg.n) {
  #pragma unroll
        for (int m = 0; m < MT; ++m) {
          float val = epilogue(Y[m], sg.act, sg.p);
          if (row >= sg.n_valid) val = -__builtin_inff();
          if (m < M && p.write_out && sg.out != nullptr) {
            if (sg.out_f32)
              reinterpret_cast<float*>(
                  sg.out)[static_cast<size_t>(m) * sg.ldo + row] = val;
            else
              reinterpret_cast<_Float16*>(
                  sg.out)[static_cast<size_t>(m) * sg.ldo + row] =
                  static_cast<_Float16>(val);
          }
          if constexpr (ARG) {
            if (val == val && val > bv[m]) {
              bv[m] = val;
              bi[m] = p.col_base + row;
            }
          }
        }
      }
  #pragma unroll
      for (int m = 0; m < MT; ++m) Y[m] = 0.0f;
    }

  #pragma unroll
    for (int u = 0; u < CPI; ++u) wcur[u] = wnxt[u];
  #pragma unroll
    for (int i = 0; i < (SV + 1) / 2; ++i) scur[i] = snxt[i];
    ti = ti2;
    it = it2;
  }

  if constexpr (ARG) {
    // block argmax per row m: waves reduce by shuffles, then WV waves in LDS
    float* rv = reinterpret_cast<float*>(smem + MT * rowB);
    int* ri = reinterpret_cast<int*>(rv + WV * MT);
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = bv[m];
      int ix = bi[m];
  #pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        const float ov = __shfl_xor(v, off);
        const int oi = __shfl_xor(ix, off);
        if (ov > v || (ov == v && oi < ix)) {
          v = ov;
          ix = oi;
        }
      }
      if (lane == 0) {
        rv[wave * MT + m] = v;
        ri[wave * MT + m] = ix;
      }
    }
    __syncthreads();
    if (tid < MT && tid < M) {
      float best_v = -__builtin_inff();
      int best_i = 0;
      for (int w = 0; w < WV; ++w) {
        const float v = rv[w * MT + tid];
        const int ix = ri[w * MT + tid];
        if (v > best_v || (v == best_v && ix < best_i)) {
          best_v = v;
          best_i = ix;
        }
      }
      p.part_val[tid * p.nparts + blockIdx.x] = best_v;
      p.part_idx[tid * p.nparts + blockIdx.x] = best_i;
    }
  }
#endif  // __HIP__RDNA2__
}

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct Cfg {
  int bits;
  int group;  // K elements per scale
  int R;      // rows per tile (KQ = 32 / R)
  int cpi;    // chunks per iteration
  int wv;     // waves per block
};

inline int chunk_elems(int bits) { return bits == 4 ? 32 : 16; }

// ---- instance table: the cross product, filtered by I2Q_ENABLED at compile
// time
struct Inst {
  int bits, gc, R, mt, cpi, wv;
  bool arg;
};

constexpr int kNInst = 2 * 4 * 4 * 3 * 2 * 3 * 2;

constexpr Inst inst_at(int i) {
  Inst k{};
  k.arg = (i % 2) != 0;
  i /= 2;
  constexpr int wvs[3] = {2, 4, 8};
  k.wv = wvs[i % 3];
  i /= 3;
  constexpr int cpis[2] = {4, 8};
  k.cpi = cpis[i % 2];
  i /= 2;
  constexpr int mts[3] = {1, 2, 4};
  k.mt = mts[i % 3];
  i /= 3;
  constexpr int rs[4] = {4, 8, 16, 32};
  k.R = rs[i % 4];
  i /= 4;
  constexpr int gcs[4] = {1, 2, 4, 8};
  k.gc = gcs[i % 4];
  i /= 4;
  k.bits = i == 0 ? 4 : 8;
  return k;
}

// the kernels' static_asserts; int8 groups are 64 / 128, int4 groups 32 / 64 /
// 128
constexpr bool inst_valid(const Inst& k) {
  const bool fmt = k.bits == 8 ? (k.gc == 4 || k.gc == 8) : (k.gc <= 4);
  return fmt && (k.cpi % k.gc == 0) && (!k.arg || (k.R == 32 && k.wv == 8));
}

template <int I>
bool try_inst(const Cfg& c, int gc, int mt, bool arg, const QParams& pr,
              int grid, size_t lds, hipStream_t st) {
  constexpr Inst k = inst_at(I);
  if constexpr (inst_valid(k) && I2Q_ENABLED(k.bits, k.gc, k.R, 32 / k.R, k.mt,
                                             k.cpi, k.wv, k.arg)) {
    if (c.bits == k.bits && gc == k.gc && c.R == k.R && mt == k.mt &&
        c.cpi == k.cpi && c.wv == k.wv && arg == k.arg) {
      qgemv_kernel<k.bits, k.gc, k.R, 32 / k.R, k.mt, k.cpi, k.wv, k.arg>
          <<<grid, k.wv * 32, lds, st>>>(pr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return true;
    }
  }
  return false;
}

template <int... I>
bool try_all(std::integer_sequence<int, I...>, const Cfg& c, int gc, int mt,
             bool arg, const QParams& pr, int grid, size_t lds,
             hipStream_t st) {
  bool done = false;
  [[maybe_unused]] const int order[] = {
      (done = done || try_inst<I>(c, gc, mt, arg, pr, grid, lds, st), 0)...};
  return done;
}

void launch(const Cfg& c, int mt, bool arg, const QParams& pr, int grid,
            size_t lds, hipStream_t st) {
  const int gc = c.group / chunk_elems(c.bits);
  const bool ok = try_all(std::make_integer_sequence<int, kNInst>{}, c, gc, mt,
                          arg, pr, grid, lds, st);
  TORCH_CHECK(ok, "inkling2_qlinear: instance not built (bits ", c.bits,
              " group ", c.group, " R ", c.R, " mt ", mt, " cpi ", c.cpi,
              " waves ", c.wv, arg ? " argmax" : "", ")");
}

void check_x(const torch::Tensor& x, int64_t M, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "inkling2_qlinear: activations must be a 2-D fp16 CUDA tensor");
  TORCH_CHECK(x.size(0) == M && x.size(1) == K,
              "inkling2_qlinear: all activations must be [", M, ", ", K, "]");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.stride(0) >= K,
              "inkling2_qlinear: activation rows need unit column stride and "
              "a row stride that is a multiple of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "inkling2_qlinear: activations must be 16-byte aligned");
}

struct Prepared {
  QParams pr;
  Cfg cfg;
  int mt;
  int grid;
  size_t lds;
};

// seg_n[i] rows of segment i are tiles of R rows (the packer padded every
// segment to a whole number of tiles).
Prepared prepare(const std::vector<torch::Tensor>& xs, const torch::Tensor& w,
                 const torch::Tensor& sc,
                 const std::vector<torch::Tensor>& outs,
                 const std::vector<int64_t>& seg_x,
                 const std::vector<int64_t>& seg_n,
                 const std::vector<int64_t>& seg_act,
                 const std::vector<double>& seg_p,
                 const std::vector<int64_t>& seg_valid, int64_t bits,
                 int64_t group, int64_t rows_per_tile, int64_t cpi,
                 int64_t waves, bool no_out, int nparts) {
  TORCH_CHECK(!xs.empty(), "inkling2_qlinear: no activations");
  const int64_t M = xs[0].size(0), K = xs[0].size(1);
  TORCH_CHECK(M >= 1 && M <= 4, "inkling2_qlinear: M must be 1..4, got ", M);
  for (const auto& x : xs) check_x(x, M, K);
  Prepared out{};
  Cfg& c = out.cfg;
  c.bits = static_cast<int>(bits);
  c.group = static_cast<int>(group);
  c.R = static_cast<int>(rows_per_tile);
  c.cpi = static_cast<int>(cpi);
  c.wv = static_cast<int>(waves);
  TORCH_CHECK(bits == 4 || bits == 8, "inkling2_qlinear: bits must be 4 or 8");
  const int ck = chunk_elems(c.bits);
  TORCH_CHECK(c.R == 32 || c.R == 16 || c.R == 8 || c.R == 4,
              "inkling2_qlinear: rows per tile must be 4, 8, 16 or 32");
  TORCH_CHECK(c.wv == 2 || c.wv == 4 || c.wv == 8,
              "inkling2_qlinear: waves per block must be 2, 4 or 8");
  const int kq = 32 / c.R;
  TORCH_CHECK(K % (ck * kq) == 0, "inkling2_qlinear: K = ", K,
              " must be a multiple of ", ck * kq);
  TORCH_CHECK(c.group % ck == 0,
              "inkling2_qlinear: group must be a multiple of ", ck);
  const int gc = c.group / ck;
  const int nc = static_cast<int>(K) / (ck * kq);
  TORCH_CHECK(nc % gc == 0, "inkling2_qlinear: chunks per lane (", nc,
              ") must be a multiple of the chunks per group (", gc, ")");
  TORCH_CHECK(c.cpi % gc == 0, "inkling2_qlinear: cpi must be a multiple of ",
              gc);
  const int niter = ceil_div(nc, c.cpi);
  const int sv = c.cpi / gc;
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kByte &&
                  w.is_contiguous() && w.dim() == 4 && w.size(1) == nc &&
                  w.size(2) == 32 && w.size(3) == 16,
              "inkling2_qlinear: weights must be uint8 [tiles, ", nc,
              ", 32, 16] (see qlinear_pack.py)");
  TORCH_CHECK(sc.is_cuda() && sc.scalar_type() == at::kHalf &&
                  sc.is_contiguous() && sc.dim() == 4 &&
                  sc.size(0) == w.size(0) && sc.size(1) == niter &&
                  sc.size(2) == 32 && sc.size(3) == sv,
              "inkling2_qlinear: scales must be fp16 [tiles, ", niter, ", 32, ",
              sv, "]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(sc.data_ptr()) % 16 == 0,
              "inkling2_qlinear: weights and scales must be 16-byte aligned");
  const size_t nseg = seg_n.size();
  TORCH_CHECK(nseg >= 1 && nseg <= kMaxSeg &&
                  (no_out ? outs.empty() : outs.size() == nseg) &&
                  seg_x.size() == nseg,
              "inkling2_qlinear: 1..", kMaxSeg,
              " segments, one output and one activation index each");
  TORCH_CHECK(seg_act.empty() || seg_act.size() == nseg,
              "inkling2_qlinear: seg_act must be empty or one per segment");
  TORCH_CHECK(seg_p.empty() || seg_p.size() == nseg,
              "inkling2_qlinear: seg_p must be empty or one per segment");
  TORCH_CHECK(seg_valid.empty() || seg_valid.size() == nseg,
              "inkling2_qlinear: seg_valid must be empty or one per segment");

  QParams& pr = out.pr;
  pr.M = static_cast<int>(M);
  pr.K = static_cast<int>(K);
  pr.nc = nc;
  pr.niter = niter;
  pr.w = w.data_ptr<uint8_t>();
  pr.sc = reinterpret_cast<const _Float16*>(sc.data_ptr());
  pr.write_out = 1;
  int tiles = 0, blocks = 0, ns = 0;
  for (size_t i = 0; i < nseg; ++i) {
    const int64_t n = seg_n[i];
    TORCH_CHECK(n >= 0, "inkling2_qlinear: negative segment size");
    const int act = seg_act.empty() ? 0 : static_cast<int>(seg_act[i]);
    TORCH_CHECK(act >= 0 && act <= kActSoftcap, "inkling2_qlinear: bad act ",
                act);
    if (n == 0) continue;
    TORCH_CHECK(seg_x[i] >= 0 && seg_x[i] < static_cast<int64_t>(xs.size()),
                "inkling2_qlinear: bad activation index");
    QSeg& s = pr.seg[ns++];
    s.x = xs[seg_x[i]].data_ptr();
    s.ldx = static_cast<int>(xs[seg_x[i]].stride(0));
    if (no_out) {
      s.out = nullptr;
      s.ldo = 0;
      s.out_f32 = 1;
    } else {
      const auto& o = outs[i];
      TORCH_CHECK(
          o.is_cuda() && o.dim() == 2 && o.size(0) == M && o.size(1) == n,
          "inkling2_qlinear: output ", i, " must be [", M, ", ", n, "]");
      TORCH_CHECK(o.stride(1) == 1 && o.stride(0) >= n,
                  "inkling2_qlinear: output rows need unit column stride");
      const bool f32 = o.scalar_type() == at::kFloat;
      TORCH_CHECK(f32 || o.scalar_type() == at::kHalf,
                  "inkling2_qlinear: output dtype must be fp16 or fp32");
      TORCH_CHECK(
          reinterpret_cast<uintptr_t>(o.data_ptr()) % (f32 ? 4 : 2) == 0,
          "inkling2_qlinear: misaligned output");
      s.out = o.data_ptr();
      s.ldo = static_cast<int>(o.stride(0));
      s.out_f32 = f32 ? 1 : 0;
    }
    const int64_t nv = seg_valid.empty() ? n : seg_valid[i];
    TORCH_CHECK(nv >= 0 && nv <= n, "inkling2_qlinear: bad seg_valid");
    s.n = static_cast<int>(n);
    s.n_valid = static_cast<int>(nv);
    s.act = act;
    s.p = seg_p.empty() ? 1.0f : static_cast<float>(seg_p[i]);
    s.tile0 = tiles;
    s.ntiles = ceil_div(static_cast<int>(n), c.R);
    s.blk0 = blocks;
    s.nblk = (nparts > 0) ? nparts : ceil_div(s.ntiles, c.wv);
    tiles += s.ntiles;
    blocks += s.nblk;
  }
  TORCH_CHECK(ns >= 1, "inkling2_qlinear: all segments are empty");
  TORCH_CHECK(tiles <= w.size(0), "inkling2_qlinear: segments need ", tiles,
              " tiles, got ", w.size(0));
  pr.nseg = ns;
  out.grid = blocks;
  out.mt = M <= 1 ? 1 : (M == 2 ? 2 : 4);
  const int sqB = nc * ck * 2 + (kq > 1 ? 16 : 0);
  out.lds = static_cast<size_t>(out.mt) * sqB * kq;
  return out;
}

}  // namespace
}  // namespace inkling2_qlinear

using namespace inkling2_qlinear;

void inkling2_qlinear_grouped(
    const std::vector<torch::Tensor>& xs, torch::Tensor w, torch::Tensor sc,
    const std::vector<torch::Tensor>& outs, const std::vector<int64_t>& seg_x,
    const std::vector<int64_t>& seg_n, const std::vector<int64_t>& seg_act,
    const std::vector<double>& seg_p, const std::vector<int64_t>& seg_valid,
    int64_t bits, int64_t group, int64_t rows_per_tile, int64_t cpi,
    int64_t waves) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  Prepared pp =
      prepare(xs, w, sc, outs, seg_x, seg_n, seg_act, seg_p, seg_valid, bits,
              group, rows_per_tile, cpi, waves, false, 0);
  launch(pp.cfg, pp.mt, false, pp.pr, pp.grid, pp.lds, stream);
}

void inkling2_qlinear_lm_head(torch::Tensor x, torch::Tensor w,
                              torch::Tensor sc,
                              const std::optional<torch::Tensor>& logits,
                              const std::optional<torch::Tensor>& part_val,
                              const std::optional<torch::Tensor>& part_idx,
                              double softcap, int64_t n_rows, int64_t n_valid,
                              int64_t col_base, int64_t bits, int64_t group,
                              int64_t cpi) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(part_val.has_value() == part_idx.has_value(),
              "inkling2_qlinear_lm_head: part_val and part_idx go together");
  TORCH_CHECK(logits.has_value() || part_val.has_value(),
              "inkling2_qlinear_lm_head: nothing to write");
  const int64_t M = x.size(0);
  const int64_t nv = n_valid < 0 ? n_rows : n_valid;
  const std::vector<int64_t> seg_n{n_rows};
  const std::vector<int64_t> seg_act{softcap > 0.0 ? kActSoftcap : kActNone};
  const std::vector<double> seg_p{softcap > 0.0 ? softcap : 1.0};
  const std::vector<int64_t> seg_valid{nv};
  const std::vector<int64_t> seg_x{0};
  const std::vector<torch::Tensor> xs{x};
  std::vector<torch::Tensor> outs;
  if (logits.has_value()) {
    TORCH_CHECK(logits->scalar_type() == at::kFloat,
                "inkling2_qlinear_lm_head: logits must be fp32");
    outs.push_back(*logits);
  }
  int nparts = 0;
  if (part_val.has_value()) {
    TORCH_CHECK(
        part_val->scalar_type() == at::kFloat &&
            part_idx->scalar_type() == at::kInt && part_val->dim() == 2 &&
            part_val->size(0) == M && part_val->size(1) >= 1 &&
            part_val->is_contiguous() && part_idx->is_contiguous() &&
            part_idx->sizes() == part_val->sizes(),
        "inkling2_qlinear_lm_head: part_val fp32 / part_idx int32 [M, P]");
    nparts = static_cast<int>(part_val->size(1));
  }
  // no partials: one block per kHeadWaves tiles; with partials: exactly nparts
  // persistent blocks
  Prepared pp =
      prepare(xs, w, sc, outs, seg_x, seg_n, seg_act, seg_p, seg_valid, bits,
              group, 32, cpi, kHeadWaves, !logits.has_value(), nparts);
  pp.pr.write_out = logits.has_value() ? 1 : 0;
  if (nparts > 0) {
    pp.pr.nparts = nparts;
    pp.pr.col_base = static_cast<int>(col_base);
    pp.pr.part_val = part_val->data_ptr<float>();
    pp.pr.part_idx = part_idx->data_ptr<int>();
    pp.lds += static_cast<size_t>(kHeadWaves) * pp.mt * 8;
    launch(pp.cfg, pp.mt, true, pp.pr, pp.grid, pp.lds, stream);
  } else {
    launch(pp.cfg, pp.mt, false, pp.pr, pp.grid, pp.lds, stream);
  }
}
