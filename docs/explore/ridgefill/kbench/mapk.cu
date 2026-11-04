// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RidgeFill map GEMM variants for the gfx1030 micro-bench (libmapk.so):
//
//   hipcc --offload-arch=gfx1030 -O3 -fPIC -shared -o libmapk.so mapk.cu
//
//   out[l][m][d] = sum_k x[m][k] * w[l * dout + d][k] (+ bias)
//
// Both operands are K-contiguous ("NT"), which suits the packed dot
// instructions: one 32-bit register holds two fp16 K values for
// v_dot2_f32_f16 or four int8 K values for v_dot4_i32_i8. The kernel is one
// template over 4-byte "packs" of K, so the fp16 and int8 variants share the
// tiling and differ only in the dot and the epilogue.
//
// * Block tile BM x BN, K slab BKB bytes per row, LDS double buffer, one
//   barrier per slab. The next slab's global reads (128-bit) are issued
//   before the current slab's math and land in LDS after it.
// * LDS rows are padded to an odd number of 16 B chunks, so the 8 lanes of
//   a ds_read_b128 pass hit 8 different bank groups (B) or broadcast (A).
// * Lane (lm, ln) owns rows lm + LM * i and columns ln + LN * j of its wave
//   tile (interleaved, which is what keeps the LDS reads conflict-free).
// * ROT rotates the four packs of every staged B chunk, so pack p of A and
//   pack p of B sit in registers of different VGPR banks when the compiler
//   allocates both 128-bit tuples at the same alignment.
// * int8: x is quantized per row (quant_rows below, optionally after a
//   Hadamard-128 rotation of K that the weights also carry), w per output
//   row; the epilogue is float(acc) * sx[m] * sw[n] + bias[n].

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace mapk {

typedef __fp16 half2_t __attribute__((ext_vector_type(2)));
// Native vectors, not HIP's uint4 struct: struct copies lower to memcpy and
// SROA left the staging registers in scratch.
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

enum DotKind { kF16 = 0, kI8 = 1 };

template <int D>
struct DotOp;

template <>
struct DotOp<kF16> {
  using Acc = float;
  static __device__ __forceinline__ float dot(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, a),
                                  __builtin_bit_cast(half2_t, b), c, false);
  }
};

template <>
struct DotOp<kI8> {
  using Acc = int32_t;
  static __device__ __forceinline__ int32_t dot(uint32_t a, uint32_t b,
                                                int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), c,
                                  false);
  }
};

__device__ __forceinline__ uint32_t comp(const u32x4& v, int q) { return v[q]; }
__device__ __forceinline__ uint32_t comp(const u32x2& v, int q) { return v[q]; }

// Operand read of RW packs (4 B each) along K: b128 or b64.
template <int RW>
struct Vec;
template <>
struct Vec<4> {
  using T = u32x4;
};
template <>
struct Vec<2> {
  using T = u32x2;
};

// B chunks are staged with their packs permuted so that pack p of A and of B
// land in registers of different VGPR banks (same-aligned tuples). RW=4
// rotates the chunk by one; RW=2 swaps the packs inside each 8 B half.
template <int RW, int ROT>
__device__ __forceinline__ u32x4 stage_perm(const u32x4& v) {
  if constexpr (ROT == 0) return v;
  if constexpr (RW == 4) return v.yzwx;
  return v.yxwz;
}
// Register component of a B read that holds pack p of the read.
template <int RW, int ROT>
__device__ __forceinline__ constexpr int b_comp(int p) {
  return ROT == 0 ? p : RW == 4 ? (p + 3) % 4 : p ^ 1;
}

template <int D_, int TM_, int TN_, int LM_, int WM_, int WN_, int BKB_,
          int RW_, int PL_, int ROT_>
struct Cfg {
  static constexpr int D = D_;
  static constexpr int TM = TM_, TN = TN_;
  static constexpr int LM = LM_, LN = 32 / LM_;
  static constexpr int WM = WM_, WN = WN_;
  static constexpr int THREADS = 32 * WM * WN;
  static constexpr int BM = WM * LM * TM;
  static constexpr int BN = WN * LN * TN;
  static constexpr int BKB = BKB_;
  static constexpr int CH = BKB / 16;   // 16 B chunks per row per slab
  static constexpr int ROWU4 = CH + 1;  // odd: conflict-free b128 rows
  static constexpr int A_PER_T = BM * CH / THREADS;
  static constexpr int B_PER_T = BN * CH / THREADS;
  static constexpr int RW = RW_;  // packs per operand read (4: b128, 2: b64)
  static constexpr int PL = PL_;  // 1: prefetch the next step's operands
  static constexpr int ROT = ROT_;
  static constexpr int STEPS = CH * 4 / RW;  // operand reads per slab
  static_assert(CH % 2 == 0, "padded LDS row must be an odd chunk count");
  static_assert(BM * CH % THREADS == 0 && BN * CH % THREADS == 0,
                "staging must split evenly over the threads");
  static_assert(LM * (32 / LM) == 32, "lane grid is one wave");
  static_assert(RW == 2 || RW == 4, "b64 or b128 operand reads");
};

template <class C>
__global__ __launch_bounds__(C::THREADS) void map_gemm_kernel(
    const uint8_t* __restrict__ x, int64_t ldx,  // [M][kb] bytes
    const uint8_t* __restrict__ w, int64_t ldw,  // [N][kb] bytes
    int size_m, int kb, int dout,
    const float* __restrict__ sx,     // [M] (int8) or null
    const float* __restrict__ sw,     // [N] (int8) or null
    const __half* __restrict__ bias,  // [N] or null
    __half* __restrict__ out) {       // [N / dout][M][dout]
  using Op = DotOp<C::D>;
  using Acc = typename Op::Acc;
  __shared__ u32x4 s_a[2][C::BM * C::ROWU4];
  __shared__ u32x4 s_b[2][C::BN * C::ROWU4];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave % C::WM;
  const int wn = wave / C::WM;
  const int lm = lane / C::LN;
  const int ln = lane % C::LN;
  const int m0 = blockIdx.x * C::BM;
  const int n0 = blockIdx.y * C::BN;

  // Staging: thread chunk r covers row id / CH, chunk id % CH of the tile.
  const uint8_t* ga[C::A_PER_T];
  int la[C::A_PER_T];
#pragma unroll
  for (int r = 0; r < C::A_PER_T; ++r) {
    const int id = tid + r * C::THREADS;
    const int row = id / C::CH;
    const int ch = id % C::CH;
    // Rows past M read row M-1 (in bounds); the epilogue drops them.
    const int gm = min(m0 + row, size_m - 1);
    ga[r] = x + gm * ldx + ch * 16;
    la[r] = row * C::ROWU4 + ch;
  }
  const uint8_t* gb[C::B_PER_T];
  int lb[C::B_PER_T];
#pragma unroll
  for (int r = 0; r < C::B_PER_T; ++r) {
    const int id = tid + r * C::THREADS;
    const int row = id / C::CH;
    const int ch = id % C::CH;
    gb[r] = w + (int64_t)(n0 + row) * ldw + ch * 16;
    lb[r] = row * C::ROWU4 + ch;
  }

  Acc acc[C::TM][C::TN];
#pragma unroll
  for (int i = 0; i < C::TM; ++i)
#pragma unroll
    for (int j = 0; j < C::TN; ++j) acc[i][j] = 0;

  u32x4 ra[C::A_PER_T];
  u32x4 rb[C::B_PER_T];
  const int slabs = kb / C::BKB;

  auto load = [&](int s) {
    const int off = s * C::BKB;
#pragma unroll
    for (int r = 0; r < C::A_PER_T; ++r)
      ra[r] = *reinterpret_cast<const u32x4*>(ga[r] + off);
#pragma unroll
    for (int r = 0; r < C::B_PER_T; ++r)
      rb[r] = *reinterpret_cast<const u32x4*>(gb[r] + off);
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int r = 0; r < C::A_PER_T; ++r) s_a[buf][la[r]] = ra[r];
#pragma unroll
    for (int r = 0; r < C::B_PER_T; ++r)
      s_b[buf][lb[r]] = stage_perm<C::RW, C::ROT>(rb[r]);
  };

  load(0);
  store(0);
  __syncthreads();

  using V = typename Vec<C::RW>::T;
  // uint32 offsets of the lane's first A / B row in a buffer.
  const int a_row0 = (wm * C::LM * C::TM + lm) * C::ROWU4 * 4;
  const int b_row0 = (wn * C::LN * C::TN + ln) * C::ROWU4 * 4;
  for (int s = 0; s < slabs; ++s) {
    const int cur = s & 1;
    // The last slab re-reads itself (in bounds) so the load and the LDS
    // store stay unconditional: a conditional pair made the compiler park
    // the staging registers in scratch.
    load(min(s + 1, slabs - 1));
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(s_a[cur]) + a_row0;
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(s_b[cur]) + b_row0;
    auto ld = [&](V* av, V* bv, int t) {
      const int off = (t / (4 / C::RW)) * 4 + (t % (4 / C::RW)) * C::RW;
#pragma unroll
      for (int i = 0; i < C::TM; ++i)
        av[i] =
            *reinterpret_cast<const V*>(pa + i * C::LM * C::ROWU4 * 4 + off);
#pragma unroll
      for (int j = 0; j < C::TN; ++j)
        bv[j] =
            *reinterpret_cast<const V*>(pb + j * C::LN * C::ROWU4 * 4 + off);
    };
    auto mac = [&](const V* av, const V* bv) {
#pragma unroll
      for (int p = 0; p < C::RW; ++p)
#pragma unroll
        for (int i = 0; i < C::TM; ++i)
#pragma unroll
          for (int j = 0; j < C::TN; ++j)
            acc[i][j] =
                Op::dot(comp(av[i], p), comp(bv[j], b_comp<C::RW, C::ROT>(p)),
                        acc[i][j]);
    };
    // The dots are pure, so the IR optimizer sinks them past a
    // sched_barrier and keeps every step's operands live at once (256 VGPRs
    // and scratch spills). An empty asm that "rewrites" each accumulator
    // pins step t's dots before step t + 1's.
    auto fence = [&]() {
#pragma unroll
      for (int i = 0; i < C::TM; ++i)
#pragma unroll
        for (int j = 0; j < C::TN; ++j) asm volatile("" : "+v"(acc[i][j]));
      __builtin_amdgcn_sched_barrier(0);
    };
    if constexpr (C::PL == 0) {
#pragma unroll
      for (int t = 0; t < C::STEPS; ++t) {
        V av[C::TM];
        V bv[C::TN];
        ld(av, bv, t);
        mac(av, bv);
        fence();
      }
    } else {
      V av[2][C::TM];
      V bv[2][C::TN];
      ld(av[0], bv[0], 0);
#pragma unroll
      for (int t = 0; t < C::STEPS; ++t) {
        if (t + 1 < C::STEPS) ld(av[(t + 1) & 1], bv[(t + 1) & 1], t + 1);
        mac(av[t & 1], bv[t & 1]);
        fence();
      }
    }
    store(cur ^ 1);
    __syncthreads();
  }

  // Epilogue: BN divides dout, so the block's layer is n0 / dout.
  const int layer = n0 / dout;
  const int d0 = n0 - layer * dout + wn * C::LN * C::TN + ln;
  const int nb = n0 + wn * C::LN * C::TN + ln;
  float colscale[C::TN];
  float colbias[C::TN];
#pragma unroll
  for (int j = 0; j < C::TN; ++j) {
    colscale[j] = sw ? sw[nb + j * C::LN] : 1.f;
    colbias[j] = bias ? __half2float(bias[nb + j * C::LN]) : 0.f;
  }
  __half* obase = out + (int64_t)layer * size_m * dout + d0;
#pragma unroll
  for (int i = 0; i < C::TM; ++i) {
    const int m = m0 + wm * C::LM * C::TM + lm + C::LM * i;
    if (m >= size_m) continue;
    const float rs = sx ? sx[m] : 1.f;
    __half* orow = obase + (int64_t)m * dout;
#pragma unroll
    for (int j = 0; j < C::TN; ++j) {
      const float v =
          static_cast<float>(acc[i][j]) * (rs * colscale[j]) + colbias[j];
      orow[j * C::LN] = __float2half(v);
    }
  }
}

// Per-row symmetric int8: q = round(y / s), s = max|y| / 127. y = x, or with
// HAD the orthonormal Sylvester Hadamard rotation of every 128-wide K block
// of x; the weights carry the same rotation, so x . w is unchanged while an
// outlier column is spread over its block. One block of 256 threads per row;
// k a multiple of 128 and at most 256 * 8 * kQuantMaxIt. A thread holds 8
// consecutive values, so a 128-block is 16 lanes: three butterfly stages run
// in registers and four across lanes.
constexpr int kQuantMaxIt = 8;

template <bool HAD>
__global__ __launch_bounds__(256) void quant_rows_kernel(
    const __half* __restrict__ x, int64_t ldx, int k, int8_t* __restrict__ q,
    float* __restrict__ scale) {
  const int row = blockIdx.x;
  const __half* xr = x + (int64_t)row * ldx;
  int8_t* qr = q + (int64_t)row * k;
  float y[kQuantMaxIt][8];
  float mx = 0.f;
#pragma unroll
  for (int it = 0; it < kQuantMaxIt; ++it) {
    // k is uniform and a multiple of 128: a 16-lane group is all in or all
    // out, so the shuffles below never mix live and dead lanes.
    if (it * 256 * 8 >= k) break;
    const int i = (it * 256 + threadIdx.x) * 8;
    u32x4 v = {0, 0, 0, 0};
    if (i < k) v = *reinterpret_cast<const u32x4*>(xr + i);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const half2_t h = __builtin_bit_cast(half2_t, v[e]);
      y[it][2 * e] = static_cast<float>(h.x);
      y[it][2 * e + 1] = static_cast<float>(h.y);
    }
    if constexpr (HAD) {
#pragma unroll
      for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (e & h) continue;
          const float a = y[it][e], b = y[it][e + h];
          y[it][e] = a + b;
          y[it][e + h] = a - b;
        }
      }
#pragma unroll
      for (int m = 1; m < 16; m <<= 1) {
        const bool hi = threadIdx.x & m;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float o = __shfl_xor(y[it][e], m, 32);
          y[it][e] = hi ? o - y[it][e] : y[it][e] + o;
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) y[it][e] *= 0.08838834764831845f;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) mx = fmaxf(mx, fabsf(y[it][e]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
  __shared__ float wmax[8];
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = wmax[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) mx = fmaxf(mx, wmax[i]);
  const float s = mx > 0.f ? mx / 127.f : 1.f;
  const float inv = 1.f / s;
  if (threadIdx.x == 0) scale[row] = s;
#pragma unroll
  for (int it = 0; it < kQuantMaxIt; ++it) {
    const int i = (it * 256 + threadIdx.x) * 8;
    if (it * 256 * 8 >= k) break;
    if (i >= k) continue;
    u32x2 packed = {0, 0};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int t = __float2int_rn(y[it][e] * inv);
      t = max(-127, min(127, t));
      packed[e / 4] |= static_cast<uint32_t>(t & 0xff) << (8 * (e % 4));
    }
    *reinterpret_cast<u32x2*>(qr + i) = packed;
  }
}

}  // namespace mapk

// ---------------------------------------------------------------------------
// C launchers. Variant ids are listed by mapk_count() / mapk_info().
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

namespace mapk {
#define MAPK_CFG(name, d, tm, tn, lm, wm, wn, bkb, rw, pl, r) \
  using name = Cfg<d, tm, tn, lm, wm, wn, bkb, rw, pl, r>;
MAPK_VARIANTS(MAPK_CFG)
#undef MAPK_CFG
}  // namespace mapk

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
    using C = mapk::name;                                                     \
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
#define MAPK_RUN(name, ...)                                                  \
  if (i++ == id) {                                                           \
    using C = mapk::name;                                                    \
    if (kb % C::BKB || n % C::BN || dout % C::BN) return -2;                 \
    dim3 grid((m + C::BM - 1) / C::BM, n / C::BN);                           \
    hipLaunchKernelGGL(mapk::map_gemm_kernel<C>, grid, dim3(C::THREADS), 0,  \
                       stream, static_cast<const uint8_t*>(x), ldx,          \
                       static_cast<const uint8_t*>(w), ldw, m, kb, dout, sx, \
                       sw, static_cast<const __half*>(bias),                 \
                       static_cast<__half*>(out));                           \
    return hipGetLastError() == hipSuccess ? 0 : -3;                         \
  }
  MAPK_VARIANTS(MAPK_RUN)
#undef MAPK_RUN
  return -1;
}

// had: 0 (plain per-row int8) or 128 (Hadamard-128 rotation first).
int mapk_quant_rows(const void* x, int64_t ldx, int m, int k, void* q,
                    float* scale, int had, hipStream_t stream) {
  if (k % 128 || k > 256 * 8 * mapk::kQuantMaxIt) return -2;
  if (had != 0 && had != 128) return -2;
  auto kern =
      had ? mapk::quant_rows_kernel<true> : mapk::quant_rows_kernel<false>;
  hipLaunchKernelGGL(kern, dim3(m), dim3(256), 0, stream,
                     static_cast<const __half*>(x), ldx, k,
                     static_cast<int8_t*>(q), scale);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

}  // extern "C"
