// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// One parametrized dense fp16 GEMM variant for gfx1030 (wave32, v_dot2_f32_f16):
//
//   C[M, N] (fp16) = A[M, K] (fp16, row-major) x W[N, K]^T (fp16, row-major)
//
// with fp32 accumulation. It is the HI2 prefill GEMM after the dequant pass
// (A = transformed activation, W = dequantized weights; C = A W^T like
// F.linear). Compiled once per variant with -DCFG_* and loaded by sweep.py
// through ctypes (extern "C" entry points at the bottom), so a sweep is a
// list of compiles, not a list of C++ edits.
//
// Design axes (all compile-time):
//   BM x BN      block tile              TM x TN   register micro-tile per thread
//   BK           K elements per stage    NBUF      LDS stages (1 = two barriers
//                                                  per step, 2 = one barrier)
//   PAD          halves of LDS row pad   GLW       global load width, bytes
//   RAST         block order (0 = N fastest, 1 = M fastest, >= 2 = grouped)
//   BAR          0 = __syncthreads, 1 = s_waitcnt lgkmcnt(0) + s_barrier
//   WPE          amdgpu_waves_per_eu minimum (0 = compiler's choice)
//   SPLITK       K splits (> 1: fp32 partials in a workspace + a reduce kernel)
//   KU           k-step unroll inside a stage (0 = fully unrolled by the compiler)
//   LW           LDS fragment read width, bytes (16 = ds_read_b128, 8 = ds_read_b64)
//   SWZ          1 = XOR-swizzle the 16-byte chunks of an LDS row instead of padding it
//                (needs PAD 0, GLW 16, BK 16/32/64); conflict-free like PAD 8, with
//                the pad's LDS bytes back (32 KiB for the 128x128 / BK 32 / 2-stage tile)
//   MODE         0 = the GEMM; 1..4 isolate one pipe for the roofline study:
//                1 compute only (LDS reads + dot2, no global, no barrier)
//                2 LDS reads only (no dot2), 3 dot2 only (operands in
//                registers), 4 memory only (global + LDS stores + barriers);
//                5 = the GEMM plus a per-block (start, end) record in ws,
//                for the measured-occupancy tool (sweep.py occ)
//   BROT         k-pair rotation of the B operand inside a fragment: dword t of
//                A pairs with dword (t + BROT) % ND of B (the loader rotates
//                the dwords of every B chunk on its way into LDS). With BROT 0
//                both sources of every v_dot2c sit in the same VGPR bank
//                (ds_read_b128 results are 4-aligned tuples); BROT 1 puts them
//                in different banks
//   ORD          order of the ND dependent dot2s of one accumulator: 0 = as the
//                compiler schedules them (it interleaves other accumulators in
//                between), 1 = one inline-asm block (they issue back to back),
//                2 = two blocks of two (ND 4 only)
//
// Thread (ty, tx) owns rows ty + TY*i and columns tx + TX*j of the tile
// (i < TM, j < TN, TX = BN / TN, TY = BM / TM): lanes of a row group read the
// same LDS row (broadcast), the other operand is one distinct row per lane.
//
// Tails: M and N may be anything (loads clamp to the last row, stores are
// guarded); K must be a multiple of BK (the loader guarantees 64 for HI2).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef CFG_BM
#define CFG_BM 128
#endif
#ifndef CFG_BN
#define CFG_BN 128
#endif
#ifndef CFG_BK
#define CFG_BK 32
#endif
#ifndef CFG_TM
#define CFG_TM 8
#endif
#ifndef CFG_TN
#define CFG_TN 8
#endif
#ifndef CFG_NBUF
#define CFG_NBUF 2
#endif
#ifndef CFG_PAD
#define CFG_PAD 8
#endif
#ifndef CFG_RAST
#define CFG_RAST 0
#endif
#ifndef CFG_GLW
#define CFG_GLW 16
#endif
#ifndef CFG_BAR
#define CFG_BAR 0
#endif
#ifndef CFG_WPE
#define CFG_WPE 0
#endif
#ifndef CFG_SPLITK
#define CFG_SPLITK 1
#endif
#ifndef CFG_MODE
#define CFG_MODE 0
#endif
#ifndef CFG_KU
#define CFG_KU 0
#endif
#ifndef CFG_LW
#define CFG_LW 16
#endif
#ifndef CFG_SWZ
#define CFG_SWZ 0
#endif
#ifndef CFG_BROT
#define CFG_BROT 0
#endif
#ifndef CFG_ORD
#define CFG_ORD 0
#endif
#ifndef CFG_NAME
#define CFG_NAME "unnamed"
#endif

#if CFG_WPE > 0
#define WPE_ATTR __attribute__((amdgpu_waves_per_eu(CFG_WPE)))
#else
#define WPE_ATTR
#endif

#define STR_(x) #x
#define XSTR_(x) STR_(x)
#if CFG_KU == 0
#define KU_PRAGMA _Pragma("unroll")
#else
#define KU_PRAGMA _Pragma(XSTR_(unroll CFG_KU))
#endif

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using u32x2 = unsigned int __attribute__((ext_vector_type(2)));

// One dword = two fp16 values: fp32 a.x*b.x + a.y*b.y + c (v_dot2c_f32_f16).
// The operands are taken by value on purpose: __builtin_bit_cast applied to a
// vector-element lvalue (v[t]) reads the vector's first element for every t
// (clang 23, gfx1030). This file shipped with that bug until the delta test
// caught it; the generated code then read one dword per 16-byte fragment.
__device__ __forceinline__ float dot2u(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
}

template <int BYTES>
struct ChunkT;
template <>
struct ChunkT<16> {
  using type = u32x4;
};
template <>
struct ChunkT<8> {
  using type = u32x2;
};
template <>
struct ChunkT<4> {
  using type = unsigned int;
};

struct P {
  static constexpr int BM = CFG_BM, BN = CFG_BN, BK = CFG_BK;
  static constexpr int TM = CFG_TM, TN = CFG_TN;
  static constexpr int NBUF = CFG_NBUF, PAD = CFG_PAD, RAST = CFG_RAST;
  static constexpr int GLW = CFG_GLW, BAR = CFG_BAR, MODE = CFG_MODE, LW = CFG_LW;
  static constexpr int SPLITK = CFG_SPLITK;
  static constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;
  static constexpr int LD = BK + PAD;     // halves per LDS row
  static constexpr int CH = GLW / 2;      // halves per global chunk
  static constexpr int CPR = BK / CH;     // chunks per tile row
  static constexpr int A_CHUNKS = BM * CPR, B_CHUNKS = BN * CPR;
  static constexpr int A_PER_T = (A_CHUNKS + T - 1) / T;
  static constexpr int B_PER_T = (B_CHUNKS + T - 1) / T;
  static constexpr int A_LDS = BM * LD, B_LDS = BN * LD;  // halves
  static constexpr int STAGE = A_LDS + B_LDS;             // halves per buffer
  static constexpr int KS = LW / 2;  // halves per LDS fragment step
  static constexpr int ND = LW / 4;  // dwords per fragment
  static constexpr int SWZ = CFG_SWZ;
  static constexpr int BROT = CFG_BROT;
  static constexpr int ORD = CFG_ORD;
  static constexpr int CPRL = BK / 8;                    // 16-byte chunks per LDS row
  static constexpr int SRP = CPRL <= 8 ? 8 / CPRL : 1;  // rows per swizzle step
  static_assert(SWZ == 0 || (GLW == 16 && PAD == 0 &&
                             (CPRL == 2 || CPRL == 4 || CPRL == 8) &&
                             TX % 8 == 0 && TY % 8 == 0),
                "SWZ: 16-byte global chunks, PAD 0, BK 16/32/64, TX and TY multiples of 8");
  static_assert(BROT == 0 || (GLW == 16 && BROT < ND), "BROT: 16-byte global chunks, BROT < dwords per fragment");
  static_assert(ORD == 0 || ORD == 1 || (ORD == 2 && ND == 4), "ORD: 0, 1, or 2 with 16-byte fragments");
  static_assert(LW == 8 || LW == 16, "LDS fragment width: 8 or 16 bytes");
  static_assert(BK % KS == 0, "BK must be a whole number of fragment steps");
  static_assert(BK % 8 == 0 && LD % 8 == 0, "LDS rows must stay 16-byte aligned");
  static_assert(BK % CH == 0, "BK must be a whole number of global chunks");
  static_assert(BM % TM == 0 && BN % TN == 0, "micro-tile must divide the tile");
  static_assert(T >= 32 && T <= 1024 && T % 32 == 0, "block size");
  static_assert(NBUF == 1 || NBUF == 2, "NBUF");
  static_assert(NBUF * STAGE * 2 <= 65536, "LDS over 64 KiB");
};

using Chunk = ChunkT<P::GLW>::type;
using Frag = ChunkT<P::LW>::type;

// B chunk with its dwords rotated inside every fragment-sized group: dword t of
// the A fragment then meets dword (t + BROT) % ND of the B fragment (BROT 0:
// the identity).
__device__ __forceinline__ Chunk brot_chunk(Chunk c) {
  if constexpr (P::BROT == 0) {
    return c;
  } else {
    Chunk r;
#pragma unroll
    for (int p = 0; p < 4; ++p)
      r[p] = c[(p & ~(P::ND - 1)) + ((p - P::BROT) & (P::ND - 1))];
    return r;
  }
}

// One accumulator, ND dot2s (A dword t with B dword (t + BROT) % ND); see ORD.
__device__ __forceinline__ float dot_chain(const Frag& a, const Frag& b, float c) {
  constexpr int M = P::ND - 1;
  if constexpr (P::ORD == 0) {
#pragma unroll
    for (int t = 0; t < P::ND; ++t) c = dot2u(a[t], b[(t + P::BROT) & M], c);
  } else if constexpr (P::ORD == 1 && P::ND == 4) {
    asm("v_dot2c_f32_f16 %0, %1, %2\n\t"
        "v_dot2c_f32_f16 %0, %3, %4\n\t"
        "v_dot2c_f32_f16 %0, %5, %6\n\t"
        "v_dot2c_f32_f16 %0, %7, %8"
        : "+v"(c)
        : "v"(a[0]), "v"(b[(0 + P::BROT) & M]), "v"(a[1]), "v"(b[(1 + P::BROT) & M]), "v"(a[2]),
          "v"(b[(2 + P::BROT) & M]), "v"(a[3]), "v"(b[(3 + P::BROT) & M]));
  } else {  // chains of two (ND 2, or ORD 2)
#pragma unroll
    for (int t = 0; t < P::ND; t += 2)
      asm("v_dot2c_f32_f16 %0, %1, %2\n\t"
          "v_dot2c_f32_f16 %0, %3, %4"
          : "+v"(c)
          : "v"(a[t]), "v"(b[(t + P::BROT) & M]), "v"(a[t + 1]), "v"(b[(t + 1 + P::BROT) & M]));
  }
  return c;
}

template <int BAR>
__device__ __forceinline__ void block_sync() {
  if constexpr (BAR == 0) {
    __syncthreads();
  } else {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
}

__global__ void __launch_bounds__(P::T) WPE_ATTR
    gemm_kernel(const char* __restrict__ A, const char* __restrict__ W,
                _Float16* __restrict__ Cout, float* __restrict__ ws, int M,
                int N, int K, int lda, int ldw, int ldc, int klen) {
  constexpr int BM = P::BM, BN = P::BN, BK = P::BK, TM = P::TM, TN = P::TN;
  constexpr int TX = P::TX, TY = P::TY, T = P::T, LD = P::LD, CH = P::CH;
  constexpr int CPR = P::CPR, NBUF = P::NBUF, MODE = P::MODE;
  constexpr int A_PER_T = P::A_PER_T, B_PER_T = P::B_PER_T;

  __shared__ __attribute__((aligned(16))) char lds[NBUF * P::STAGE * 2];

  const int tid = threadIdx.x;
  [[maybe_unused]] uint64_t t_begin = 0;
  if constexpr (MODE == 5) {
    if (tid == 0) t_begin = __builtin_amdgcn_s_memrealtime();
  }
  const int tx = tid % TX, ty = tid / TX;
  // XOR swizzle of the 16-byte chunk index inside an LDS row (SWZ = 1): the
  // rows of one 8-lane read phase land in 8 different bank groups without the
  // PAD halves, so a 128x128 BK 32 two-stage tile needs 32 KiB instead of 40.
  // TX and TY are multiples of 8, so every row an operand thread reads has the
  // same swizzle value.
  const int swa = P::SWZ ? (ty / P::SRP) & (P::CPRL - 1) : 0;
  const int swb = P::SWZ ? (tx / P::SRP) & (P::CPRL - 1) : 0;
  auto store_off = [](int r, int kcc) {  // byte offset of chunk kcc of LDS row r
    if constexpr (P::SWZ)
      return r * LD * 2 + ((kcc ^ ((r / P::SRP) & (P::CPRL - 1))) << 4);
    else
      return (r * LD + kcc * CH) * 2;
  };
  auto frag_off = [](int kk, int sw) {  // byte offset of halves kk.. in a row
    if constexpr (P::SWZ)
      return (((kk >> 3) ^ sw) << 4) + ((kk & 7) << 1);
    else
      return kk * 2;
  };

  // ---- block -> tile (rasterization)
  const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
  int tm, tn;
  const int bid = blockIdx.x;
  if constexpr (P::RAST == 0) {
    tn = bid % ntn;
    tm = bid / ntn;
  } else if constexpr (P::RAST == 1) {
    tm = bid % ntm;
    tn = bid / ntm;
  } else {
    constexpr int GM = P::RAST;
    const int per_group = GM * ntn;
    const int group = bid / per_group;
    const int first_m = group * GM;
    const int gsz = min(ntm - first_m, GM);
    tm = first_m + (bid % per_group) % gsz;
    tn = (bid % per_group) / gsz;
  }
  const int m0 = tm * BM, n0 = tn * BN;
  const int kbeg = blockIdx.y * klen;
  const int kstop = min(K, kbeg + klen);
  const int nk = (kstop - kbeg) / BK;

  // ---- global -> register chunk bookkeeping (byte offsets, 32-bit)
  uint32_t aoff[A_PER_T], boff[B_PER_T];
#pragma unroll
  for (int v = 0; v < A_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BM - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(m0 + row, M - 1);
    aoff[v] = static_cast<uint32_t>((grow * lda + kbeg + kc) * 2);
  }
#pragma unroll
  for (int v = 0; v < B_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BN - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(n0 + row, N - 1);
    boff[v] = static_cast<uint32_t>((grow * ldw + kbeg + kc) * 2);
  }

  Chunk ra[A_PER_T], rb[B_PER_T];
  auto gload = [&](int kt) {
    const uint32_t kb = static_cast<uint32_t>(kt * BK * 2);
#pragma unroll
    for (int v = 0; v < A_PER_T; ++v)
      if (A_PER_T * T == P::A_CHUNKS || tid + v * T < P::A_CHUNKS)
        ra[v] = *reinterpret_cast<const Chunk*>(A + aoff[v] + kb);
#pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      if (B_PER_T * T == P::B_CHUNKS || tid + v * T < P::B_CHUNKS)
        rb[v] = *reinterpret_cast<const Chunk*>(W + boff[v] + kb);
  };
  auto lstore = [&](int buf) {
    char* as = lds + buf * P::STAGE * 2;
    char* bs = as + P::A_LDS * 2;
#pragma unroll
    for (int v = 0; v < A_PER_T; ++v) {
      const int c = tid + v * T;
      if (A_PER_T * T == P::A_CHUNKS || c < P::A_CHUNKS)
        *reinterpret_cast<Chunk*>(as + store_off(c / CPR, c % CPR)) = ra[v];
    }
#pragma unroll
    for (int v = 0; v < B_PER_T; ++v) {
      const int c = tid + v * T;
      if (B_PER_T * T == P::B_CHUNKS || c < P::B_CHUNKS)
        *reinterpret_cast<Chunk*>(bs + store_off(c / CPR, c % CPR)) = brot_chunk(rb[v]);
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  unsigned sink = 0;

  auto compute = [&](int buf) {
    const char* as = lds + buf * P::STAGE * 2;
    const char* bs = as + P::A_LDS * 2;
  KU_PRAGMA
    for (int kk = 0; kk < BK; kk += P::KS) {
      Frag af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const Frag*>(as + (ty + TY * i) * LD * 2 +
                                               frag_off(kk, swa));
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const Frag*>(bs + (tx + TX * j) * LD * 2 +
                                               frag_off(kk, swb));
      if constexpr (MODE == 2) {
#pragma unroll
        for (int i = 0; i < TM; ++i) sink ^= af[i][0] ^ af[i][P::ND - 1];
#pragma unroll
        for (int j = 0; j < TN; ++j) sink ^= bf[j][0] ^ bf[j][P::ND - 1];
      } else {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = dot_chain(af[i], bf[j], acc[i][j]);
          }
      }
    }
  };

  if constexpr (MODE == 3) {
    // dot2 only: operands loaded once, the same sequence nk * BK / 8 times
    Frag af[TM], bf[TN];
    gload(0);
    lstore(0);
    block_sync<P::BAR>();
    const char* as = lds;
    const char* bs = as + P::A_LDS * 2;
#pragma unroll
    for (int i = 0; i < TM; ++i)
      af[i] = *reinterpret_cast<const Frag*>(as + (ty + TY * i) * LD * 2 + frag_off(0, swa));
#pragma unroll
    for (int j = 0; j < TN; ++j)
      bf[j] = *reinterpret_cast<const Frag*>(bs + (tx + TX * j) * LD * 2 + frag_off(0, swb));
    for (int kt = 0; kt < nk; ++kt) {
#pragma unroll
      for (int kk = 0; kk < BK; kk += P::KS) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = dot_chain(af[i], bf[j], acc[i][j]);
          }
      }
    }
  } else if constexpr (MODE == 1 || MODE == 2) {
    // compute only: LDS filled once, then reads (+ dot2) with no barrier
    gload(0);
    lstore(0);
    block_sync<P::BAR>();
    for (int kt = 0; kt < nk; ++kt) {
      asm volatile("" ::: "memory");
      compute(0);
    }
  } else if constexpr (MODE == 4) {
    // memory only: global loads, LDS stores, barriers; no reads, no dot2
    gload(0);
    for (int kt = 0; kt < nk; ++kt) {
      lstore(kt % NBUF);
      block_sync<P::BAR>();
      if (kt + 1 < nk) gload(kt + 1);
    }
    sink = *reinterpret_cast<const unsigned*>(lds + tid * 4);
  } else if constexpr (NBUF == 2) {
    gload(0);
    lstore(0);
    block_sync<P::BAR>();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      gload(kt + 1);
      compute(0);
      lstore(1);
      block_sync<P::BAR>();
      if (kt + 2 < nk) gload(kt + 2);
      compute(1);
      if (kt + 2 < nk) lstore(0);
      block_sync<P::BAR>();
    }
    if (kt < nk) compute(0);
  } else {
    gload(0);
    for (int kt = 0; kt < nk; ++kt) {
      block_sync<P::BAR>();
      lstore(0);
      block_sync<P::BAR>();
      if (kt + 1 < nk) gload(kt + 1);
      compute(0);
    }
  }
  if constexpr (MODE == 2 || MODE == 4) acc[0][0] += static_cast<float>(sink);

  // ---- epilogue
  if constexpr (P::SPLITK == 1) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = m0 + ty + TY * i;
      if (row < M) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int col = n0 + tx + TX * j;
          if (col < N)
            Cout[static_cast<size_t>(row) * ldc + col] =
                static_cast<_Float16>(acc[i][j]);
        }
      }
    }
  } else {
    float* wsp = ws + static_cast<size_t>(blockIdx.y) * M * N;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = m0 + ty + TY * i;
      if (row < M) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int col = n0 + tx + TX * j;
          if (col < N) wsp[static_cast<size_t>(row) * N + col] = acc[i][j];
        }
      }
    }
  }
  if constexpr (MODE == 5) {
    // occupancy probe: one record per block (start, end in 10 ns ticks, 0, block)
    if (tid == 0) {
      uint64_t* rec = reinterpret_cast<uint64_t*>(ws) +
                      4ull * (blockIdx.y * gridDim.x + blockIdx.x);
      rec[0] = t_begin;
      rec[1] = __builtin_amdgcn_s_memrealtime();
      rec[2] = 0;
      rec[3] = blockIdx.x;
    }
  }
}

// Sum of the split-K partials, rounded once to fp16.
__global__ void reduce_kernel(const float* __restrict__ ws,
                              _Float16* __restrict__ C, int M, int N, int ldc,
                              int splits) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(M) * N) return;
  float s = 0.0f;
  for (int z = 0; z < splits; ++z) s += ws[static_cast<size_t>(z) * M * N + i];
  const size_t row = i / N, col = i % N;
  C[row * ldc + col] = static_cast<_Float16>(s);
}

extern "C" {

const char* variant_name() { return CFG_NAME; }

// out[0..7]: VGPRs, static LDS bytes, scratch bytes, threads, blocks per
// WGP (hip occupancy), tile rows, tile cols, split-K.
int variant_info(int* out) {
  hipFuncAttributes at;
  if (hipFuncGetAttributes(&at, reinterpret_cast<const void*>(gemm_kernel)) !=
      hipSuccess)
    return -1;
  int nb = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &nb, reinterpret_cast<const void*>(gemm_kernel), P::T, 0) !=
      hipSuccess)
    nb = -1;
  out[0] = at.numRegs;
  out[1] = static_cast<int>(at.sharedSizeBytes);
  out[2] = static_cast<int>(at.localSizeBytes);
  out[3] = P::T;
  out[4] = nb;
  out[5] = P::BM;
  out[6] = P::BN;
  out[7] = P::SPLITK;
  return 0;
}

// Workspace bytes needed for split-K (0 otherwise).
size_t variant_ws_bytes(int M, int N) {
  if (P::MODE == 5)  // 4 x u64 per block
    return static_cast<size_t>((M + P::BM - 1) / P::BM) * ((N + P::BN - 1) / P::BN) * 32;
  return P::SPLITK > 1 ? static_cast<size_t>(P::SPLITK) * M * N * 4 : 0;
}

int variant_launch(const void* A, const void* W, void* C, int M, int N, int K,
                   int lda, int ldw, int ldc, void* ws, void* stream) {
  if (K % P::BK != 0) return -2;
  const int ntm = (M + P::BM - 1) / P::BM, ntn = (N + P::BN - 1) / P::BN;
  int splits = P::SPLITK;
  const int ksteps = K / P::BK;
  int klen = ((ksteps + splits - 1) / splits) * P::BK;
  splits = (K + klen - 1) / klen;
  dim3 grid(ntm * ntn, splits);
  hipLaunchKernelGGL(gemm_kernel, grid, dim3(P::T), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const char*>(A), static_cast<const char*>(W),
                     static_cast<_Float16*>(C), static_cast<float*>(ws), M, N,
                     K, lda, ldw, ldc, klen);
  if (P::SPLITK > 1) {
    const size_t n = static_cast<size_t>(M) * N;
    hipLaunchKernelGGL(reduce_kernel, dim3((n + 255) / 256), dim3(256), 0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const float*>(ws), static_cast<_Float16*>(C),
                       M, N, ldc, splits);
  }
  return static_cast<int>(hipGetLastError());
}

}  // extern "C"
