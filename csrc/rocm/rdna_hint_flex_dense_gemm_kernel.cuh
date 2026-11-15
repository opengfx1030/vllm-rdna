// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Device code of the dense fp16 GEMM for gfx1030 (wave32, v_dot2_f32_f16):
//
//   C[M, N] (fp16) = A[M, K] (fp16, row-major) x W[N, K]^T (fp16, row-major)
//
// fp32 accumulation, one fp16 rounding at the store. No torch headers: the
// production TU (rdna_hint_flex_dense_gemm_rdna2.cu) and the benchmark harness
// (tools/rdna/rdna_hint_flex/gemm_sweep/dense_dev.hip) compile the same
// kernel template, so a design measured in the harness is the code that ships.
//
// Design axes (compile time): BM x BN block tile, TM x TN accumulators per
// thread (thread (ty, tx) owns rows ty + TY i and columns tx + TX j of the
// tile, so the lanes of a row group read one LDS row and the other operand is
// one distinct row per lane), BK K elements per stage, NBUF LDS stages (2 = one
// barrier per tile, double buffered through registers; 1 = two barriers), PAD
// halves of LDS row padding or SWZ = XOR-swizzled 16-byte chunks without
// padding (conflict-free like PAD 8, with the pad's bytes back: the 128x128 BK
// 32 two-stage tile fits four blocks in a WGP's 128 KiB), RAST block order (0 =
// N fastest, >= 2 = groups of RAST M tiles walk the N tiles, so concurrently
// resident blocks share the weight tiles), LW LDS fragment read width (8 bytes
// = ds_read_b64), KU k-step unroll inside a stage (0 = fully unrolled).
//
// Knobs beyond the sweep grammar (all default off, which leaves the code of the
// swept designs unchanged):
//   BROT 1  the dwords of every B fragment are stored rotated by one inside the
//           LDS image, so the two sources of every v_dot2 sit in different VGPR
//           banks (the compiler pairs A and B dword t in the same bank
//           otherwise for half of the fragment pairs: dot2 with same-bank
//           sources issues 13-15 % slower,
//           docs/rdna2/hi2-prefill-gemm-2026-11-15.md). The sum order is
//           unchanged, the result bit-identical. BROT 2 writes the rotated
//           chunk as four separate 32-bit LDS stores (no VALU moves for the
//           shuffle; BROT 1 pays two v_mov per stored chunk).
//   SKIP 1  a wave whose rows (or columns) all lie beyond M (N) skips its dot2
//           work: the tail tile of an M just above a multiple of BM.
//   PRIO 1 / 2  s_setprio 1 around the compute phase / around the global-load
//           and LDS-store phase of each stage.
//   DMA 1   the tiles go global -> LDS with buffer_load_dword ... lds (no
//           staging registers, no ds_write, no address VGPRs per chunk: one
//           lane-offset VGPR per matrix; rows past M / N read as zero through
//           the buffer bounds). The XOR swizzle is applied to the global source
//           offsets of the lanes (each 32-lane instruction fills two whole LDS
//           rows of 64 bytes). Needs SWZ, NBUF 2, BK 32; BROT 0 or 1 (the dword
//           rotation is one more index change in the lanes' source offsets).
//   BUF 1   the global loads are buffer_load_dwordx4 through a buffer
//   descriptor
//           (the K advance is a scalar soffset, the per-thread voffset stays
//           put) instead of global_load_dwordx4 with a 64-bit VGPR address that
//           takes two VALU adds per load and stage. Not with DMA.
//
// Tails: M and N may be anything (row loads clamp to the last row, stores are
// guarded); K must be a multiple of BK; operands need 16-byte aligned base
// pointers and row strides.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace rdna_hint_flex {
namespace gemm {

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using u32x2 = unsigned int __attribute__((ext_vector_type(2)));

// One dword = two fp16 values: fp32 a.x*b.x + a.y*b.y + c (v_dot2c_f32_f16).
// The operands are taken by value on purpose: __builtin_bit_cast applied to a
// vector-element lvalue (v[t]) reads the vector's first element for every t
// (clang 23, gfx1030).
__device__ __forceinline__ float dot2u(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
}

template <int BYTES>
struct FragT;
template <>
struct FragT<16> {
  using type = u32x4;
};
template <>
struct FragT<8> {
  using type = u32x2;
};

// One compile-time design.
template <int BM_, int BN_, int BK_, int TM_, int TN_, int NBUF_, int PAD_,
          int RAST_, int LW_, bool SWZ_, int KU_, int BROT_ = 0, int SKIP_ = 0,
          int PRIO_ = 0, int DMA_ = 0, int BUF_ = 0>
struct Design {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int NBUF = NBUF_, PAD = PAD_, RAST = RAST_, LW = LW_;
  static constexpr bool SWZ = SWZ_;
  static constexpr int KU = KU_;  // 0 = fully unrolled k steps
  static constexpr int BROT = BROT_, SKIP = SKIP_, PRIO = PRIO_, DMA = DMA_;
  static constexpr int BUF = BUF_;
  static constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;
  static constexpr int LD = BK + PAD;  // halves per LDS row
  static constexpr int CH = 8;         // halves per 16-byte global chunk
  static constexpr int CPR = BK / CH;  // chunks per tile row
  static constexpr int A_CHUNKS = BM * CPR, B_CHUNKS = BN * CPR;
  static constexpr int A_PER_T = (A_CHUNKS + T - 1) / T;
  static constexpr int B_PER_T = (B_CHUNKS + T - 1) / T;
  static constexpr int A_LDS = BM * LD, B_LDS = BN * LD;  // halves
  static constexpr int STAGE = A_LDS + B_LDS;             // halves per buffer
  static constexpr int KS = LW / 2;                       // halves per k step
  static constexpr int ND = LW / 4;                       // dwords per fragment
  static constexpr int SRP = CPR <= 8 ? 8 / CPR : 1;  // rows per swizzle step
  static constexpr int LDS_BYTES = NBUF * STAGE * 2;
  static_assert(!SWZ || (PAD == 0 && (CPR == 2 || CPR == 4 || CPR == 8) &&
                         TX % 8 == 0 && TY % 8 == 0),
                "SWZ: PAD 0, BK 16/32/64, TX and TY multiples of 8");
  static_assert(LW == 8 || LW == 16, "LDS fragment width: 8 or 16 bytes");
  static_assert(
      BK % KS == 0 && BK % CH == 0 && LD % 8 == 0,
      "BK: whole fragment steps and chunks; LDS rows 16-byte aligned");
  static_assert(BM % TM == 0 && BN % TN == 0,
                "micro-tile must divide the tile");
  static_assert(T >= 32 && T <= 1024 && T % 32 == 0, "block size");
  static_assert(NBUF == 1 || NBUF == 2, "NBUF");
  static_assert(LDS_BYTES <= 65536, "LDS over 64 KiB per block");
  static_assert(BROT >= 0 && BROT <= 2 && SKIP >= 0 && SKIP <= 1 && PRIO >= 0 &&
                    PRIO <= 2,
                "BROT 0-2, SKIP 0-1, PRIO 0-2");
  static_assert(DMA == 0 ||
                    (DMA == 1 && SWZ && NBUF == 2 && BROT <= 1 && BK == 32 &&
                     (T / 32) % 4 == 0 && (BM / 2) % (T / 32) == 0 &&
                     (BN / 2) % (T / 32) == 0),
                "DMA 1: SWZ, NBUF 2, BK 32, BROT 0 or 1, whole waves per slab");
  static_assert(BUF == 0 || (BUF == 1 && DMA == 0), "BUF 1 excludes DMA");
};

using i32x4 = int __attribute__((ext_vector_type(4)));

// Raw buffer resource (stride 0) over `bytes` bytes at p. Loads at offsets
// >= bytes return zero without faulting.
__device__ __forceinline__ i32x4 buffer_rsrc(const void* p, uint32_t bytes) {
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  return i32x4{
      __builtin_amdgcn_readfirstlane(static_cast<int>(a)),
      __builtin_amdgcn_readfirstlane(static_cast<int>(a >> 32) & 0xffff),
      __builtin_amdgcn_readfirstlane(static_cast<int>(bytes)),
      0x31014000};  // gfx10 raw buffer: dst_sel xyzw, format 32, no OOB-select
}

// buffer_load_dword ... lds: lane l's dword at voff + soff goes to LDS at
// lds_addr + 4 l (lds_addr and soff wave-uniform). The compiler does not track
// these loads: the caller waits with vm_wait() before the barrier.
__device__ __forceinline__ void dma_dword(i32x4 rsrc, uint32_t voff,
                                          uint32_t soff, uint32_t lds_addr) {
  asm volatile(
      "s_mov_b32 m0, %3\n\t"
      "buffer_load_dword %0, %1, %2 offen lds"
      :
      : "v"(voff), "s"(rsrc), "s"(soff), "s"(lds_addr)
      : "memory", "m0");
}

__device__ __forceinline__ void vm_wait() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

template <class P>
__global__ void __launch_bounds__(P::T)
    gemm_kernel(const char* __restrict__ A, const char* __restrict__ W,
                _Float16* __restrict__ Cout, int M, int N, int K, int lda,
                int ldw, int ldc) {
  using Chunk = u32x4;  // 16-byte global chunk
  using Frag = typename FragT<P::LW>::type;
  constexpr int BM = P::BM, BN = P::BN, BK = P::BK, TM = P::TM, TN = P::TN;
  constexpr int TX = P::TX, TY = P::TY, T = P::T, LD = P::LD, CH = P::CH;
  constexpr int CPR = P::CPR, NBUF = P::NBUF;
  constexpr int A_PER_T = P::A_PER_T, B_PER_T = P::B_PER_T;

  __shared__ __attribute__((aligned(16))) char lds[P::LDS_BYTES];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  // XOR swizzle of the 16-byte chunk index inside an LDS row (SWZ): the rows of
  // one 8-lane read phase land in 8 different bank groups without the PAD
  // halves. TX and TY are multiples of 8, so every row an operand thread reads
  // has the same swizzle value.
  const int swa = P::SWZ ? (ty / P::SRP) & (CPR - 1) : 0;
  const int swb = P::SWZ ? (tx / P::SRP) & (CPR - 1) : 0;
  auto store_off = [](int r,
                      int kcc) {  // byte offset of chunk kcc of LDS row r
    if constexpr (P::SWZ)
      return r * LD * 2 + ((kcc ^ ((r / P::SRP) & (CPR - 1))) << 4);
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
  const int nk = K / BK;

  // SKIP: this wave has no valid row or column (uniform per wave: the first
  // thread of a wave owns its smallest row ty and smallest column tx).
  bool wave_live = true;
  if constexpr (P::SKIP) {
    const int w0 = __builtin_amdgcn_readfirstlane(tid) & ~31;
    wave_live = (m0 + w0 / TX < M) && (n0 + (TX >= 32 ? w0 % TX : 0) < N);
  }

  // ---- global -> register chunk bookkeeping (byte offsets, 32-bit)
  uint32_t aoff[A_PER_T], boff[B_PER_T];
  #pragma unroll
  for (int v = 0; v < A_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BM - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(m0 + row, M - 1);
    aoff[v] = static_cast<uint32_t>((grow * lda + kc) * 2);
  }
  #pragma unroll
  for (int v = 0; v < B_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BN - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(n0 + row, N - 1);
    boff[v] = static_cast<uint32_t>((grow * ldw + kc) * 2);
  }

  Chunk ra[A_PER_T], rb[B_PER_T];
  // BUF: the descriptors cover the operands; rows are clamped above, so no load
  // goes out of range.
  const __amdgpu_buffer_rsrc_t rsa = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(A), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(M) * static_cast<uint32_t>(lda) *
                       2u),
      0x31014000);
  const __amdgpu_buffer_rsrc_t rsw = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(W), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(N) * static_cast<uint32_t>(ldw) *
                       2u),
      0x31014000);
  auto gload = [&](int kt) {
    const uint32_t kb = static_cast<uint32_t>(kt * BK * 2);
  #pragma unroll
    for (int v = 0; v < A_PER_T; ++v)
      if (A_PER_T * T == P::A_CHUNKS || tid + v * T < P::A_CHUNKS) {
        if constexpr (P::BUF)
          ra[v] = __builtin_amdgcn_raw_buffer_load_b128(rsa, aoff[v], kb, 0);
        else
          ra[v] = *reinterpret_cast<const Chunk*>(A + aoff[v] + kb);
      }
  #pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      if (B_PER_T * T == P::B_CHUNKS || tid + v * T < P::B_CHUNKS) {
        if constexpr (P::BUF)
          rb[v] = __builtin_amdgcn_raw_buffer_load_b128(rsw, boff[v], kb, 0);
        else
          rb[v] = *reinterpret_cast<const Chunk*>(W + boff[v] + kb);
      }
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
      if (B_PER_T * T == P::B_CHUNKS || c < P::B_CHUNKS) {
        char* dst = bs + store_off(c / CPR, c % CPR);
        if constexpr (P::BROT == 0) {
          *reinterpret_cast<Chunk*>(dst) = rb[v];
        } else {
          // dword d of the chunk lands at position (d + 1) mod ND of its
          // ND-dword fragment: {y, x, w, z} for 8-byte, {w, x, y, z} for
          // 16-byte fragments
          const Chunk q = rb[v];
          if constexpr (P::BROT == 1) {
            if constexpr (P::LW == 8)
              *reinterpret_cast<Chunk*>(dst) = Chunk{q.y, q.x, q.w, q.z};
            else
              *reinterpret_cast<Chunk*>(dst) = Chunk{q.w, q.x, q.y, q.z};
          } else {
            // volatile keeps the four stores apart: merged into one 16-byte
            // store they would need a shuffled register tuple (16 v_mov per
            // loop iteration)
            volatile uint32_t* d32 = reinterpret_cast<volatile uint32_t*>(dst);
            if constexpr (P::LW == 8) {
              d32[0] = q.y;
              d32[1] = q.x;
              d32[2] = q.w;
              d32[3] = q.z;
            } else {
              d32[0] = q.w;
              d32[1] = q.x;
              d32[2] = q.y;
              d32[3] = q.z;
            }
          }
        }
      }
    }
  };

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  auto compute = [&](int buf) {
    const char* as = lds + buf * P::STAGE * 2;
    const char* bs = as + P::A_LDS * 2;
    auto kstep = [&](int kk) {
      Frag af[TM], bf[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const Frag*>(as + (ty + TY * i) * LD * 2 +
                                               frag_off(kk, swa));
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const Frag*>(bs + (tx + TX * j) * LD * 2 +
                                               frag_off(kk, swb));
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j)
  #pragma unroll
          for (int t = 0; t < P::ND; ++t)
            acc[i][j] = dot2u(af[i][t], bf[j][P::BROT ? (t + 1) % P::ND : t],
                              acc[i][j]);
    };
    if constexpr (P::KU == 0) {
  #pragma unroll
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    } else {
  #pragma unroll P::KU
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    }
  };
  auto run = [&](int buf) {
    if constexpr (P::SKIP) {
      if (!wave_live) return;
    }
    if constexpr (P::PRIO == 1) __builtin_amdgcn_s_setprio(1);
    compute(buf);
    if constexpr (P::PRIO == 1) __builtin_amdgcn_s_setprio(0);
  };
  auto mem_begin = [&]() {
    if constexpr (P::PRIO == 2) __builtin_amdgcn_s_setprio(1);
  };
  auto mem_end = [&]() {
    if constexpr (P::PRIO == 2) __builtin_amdgcn_s_setprio(0);
  };

  if constexpr (P::DMA) {
    // Wave w fills the LDS slabs s = j NW + w (tile rows 2 s and 2 s + 1, 128
    // bytes: one instruction), so (s / SRP) & 3 = w & 3: one swizzle value, one
    // lane-offset VGPR per matrix. Lane l's dword lands at slab + 4 l, which is
    // row 2 s + (l >> 4), stored chunk (l & 15) >> 2, dword l & 3; the stored
    // chunk holds logical chunk ((l & 15) >> 2) ^ (w & 3).
    constexpr int NW = T / 32;
    constexpr int A_J = BM / 2 / NW, B_J = BN / 2 / NW;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const uint32_t lda_b = static_cast<uint32_t>(lda) * 2;
    const uint32_t ldw_b = static_cast<uint32_t>(ldw) * 2;
    const i32x4 rsa = buffer_rsrc(A, static_cast<uint32_t>(M) * lda_b);
    const i32x4 rsw = buffer_rsrc(W, static_cast<uint32_t>(N) * ldw_b);
    const uint32_t lane_k =
        static_cast<uint32_t>((((lane & 15) >> 2) ^ (wave & 3)) << 4) +
        static_cast<uint32_t>((lane & 3) << 2);
    const uint32_t row_l = static_cast<uint32_t>(2 * wave + (lane >> 4));
    // BROT 1: stored dword p of an ND-dword weight fragment holds source dword
    // (p + ND - 1) % ND (the compute side reads dword t at (t + 1) % ND)
    const uint32_t dw = static_cast<uint32_t>(lane & 3);
    const uint32_t dws =
        P::BROT ? (dw / P::ND) * P::ND + (dw % P::ND + P::ND - 1) % P::ND : dw;
    const uint32_t lane_kb = (lane_k & ~12u) | (dws << 2);
    const uint32_t va = row_l * lda_b + lane_k, vb = row_l * ldw_b + lane_kb;
    const uint32_t sa = static_cast<uint32_t>(m0) * lda_b;
    const uint32_t sb = static_cast<uint32_t>(n0) * ldw_b;
    const uint32_t lds0 =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
        static_cast<uint32_t>(wave * 128);
    auto dma_stage = [&](int kt, int buf) {
      const uint32_t kb = static_cast<uint32_t>(kt * (BK * 2));
      const uint32_t base = lds0 + static_cast<uint32_t>(buf * P::STAGE * 2);
  #pragma unroll
      for (int j = 0; j < A_J; ++j)
        dma_dword(rsa, va, sa + kb + static_cast<uint32_t>(j * 2 * NW) * lda_b,
                  base + static_cast<uint32_t>(j * NW * 128));
  #pragma unroll
      for (int j = 0; j < B_J; ++j)
        dma_dword(rsw, vb, sb + kb + static_cast<uint32_t>(j * 2 * NW) * ldw_b,
                  base + static_cast<uint32_t>(P::A_LDS * 2 + j * NW * 128));
    };
    dma_stage(0, 0);
    vm_wait();
    __syncthreads();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      dma_stage(kt + 1, 1);
      run(0);
      vm_wait();
      __syncthreads();
      if (kt + 2 < nk) dma_stage(kt + 2, 0);
      run(1);
      vm_wait();
      __syncthreads();
    }
    if (kt < nk) run(0);
  } else if constexpr (NBUF == 2) {
    mem_begin();
    gload(0);
    lstore(0);
    mem_end();
    __syncthreads();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      mem_begin();
      gload(kt + 1);
      mem_end();
      run(0);
      mem_begin();
      lstore(1);
      mem_end();
      __syncthreads();
      mem_begin();
      if (kt + 2 < nk) gload(kt + 2);
      mem_end();
      run(1);
      mem_begin();
      if (kt + 2 < nk) lstore(0);
      mem_end();
      __syncthreads();
    }
    if (kt < nk) run(0);
  } else {
    mem_begin();
    gload(0);
    mem_end();
    for (int kt = 0; kt < nk; ++kt) {
      __syncthreads();
      mem_begin();
      lstore(0);
      mem_end();
      __syncthreads();
      mem_begin();
      if (kt + 1 < nk) gload(kt + 1);
      mem_end();
      run(0);
    }
  }

  // ---- epilogue: fp32 accumulators -> fp16
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
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gemm
}  // namespace rdna_hint_flex
}  // namespace vllm
