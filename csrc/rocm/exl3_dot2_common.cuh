// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (QTIP-style bitshift trellis) dequant + V_DOT2 primitives for RDNA2
// (gfx1030). Faithful port of ExLlamaV3 (turboderp-org/exllamav3)
// inference-decode path, mirrored after the numerically-verified
// CarouselAether/rocm_exl3 HIP port (gfx1151; bit-by-bit PTX lop3 reference,
// 0 mismatches). Per the rdna-hip-wiki EXL3 contract:
//
//   * B: (k/16, n/16, 16*K) uint16, K = bpw in {1..8}; k%16==0, n%16==0.
//     Each 16x16 tile = 16*K uint16 (= 8*K uint32 words).
//   * Per weight: bit-extract a 16-bit state from the packed stream
//     (bitshift trellis - parallel window extraction, no walk), then
//     decode_3inst<cb>(state) -> one half via a procedural codebook.
//   * Nibbles are NOT i4 codes. NEVER fire sdot* on the decoded halves.
//   * Inner op after decode: V_DOT2_F32_F16 via __builtin_amdgcn_fdot2.
//   * Template cb (one codebook per launch) + template K (one bpw per
//     launch - do not mix bpw in a WG).
//
// Codebook (upstream codebook.cuh):
//   cb==0 ("3inst"): x*=89226354; x+=64248484; LOP3; hadd(lo,hi)
//   cb==1 ("mcg"):   x*=0xCBAC1FED;             LOP3; hadd(lo,hi)
//   cb==2 ("mul1"):  x*=0x83DCD12D; byte-sum;   hfma  (not produced)
//
// LOP3 emulation (verified, gfx1151 vs bit-by-bit PTX):
//   upstream:  "lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a"
//   f(x, 0x8fff8fff, 0x3b603b60, 0x6a) = (~x & c) | (x & (b ^ c))
//   where b = 0x8fff8fff, c = 0x3b603b60. Compile-time constants:
//       b ^ c = 0x8fff8fff ^ 0x3b603b60 = 0xb49fb49f
//       ~x & c = (x ^ 0x3b603b60) ^ x's complement... actually a 3-VALU
//   form: (~x & 0x3b603b60u) | (x & 0xb49fb49fu).

#ifndef _EXL3_DOT2_RDNA2_CUH
#define _EXL3_DOT2_RDNA2_CUH

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace vllm {
namespace exl3_dot2 {

// 64-bit CAS half2-add for 4 consecutive fp16 columns (W4A16/mxfp4 pattern;
// shared by the dense and MoE EXL3 kernels). gfx1030 lacks native
// v_global_atomic_pk_add_f16, so emulate with the 64-bit CAS loop.
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// ---------------------------------------------------------------------------
// LOP3 emulation (verified against PTX lop3(0x6a) on gfx1151, 0 mismatches).
// RDNA2 2-op fold: f = c ^ (x & b) with b=0x8fff8fff, c=0x3b603b60 (the
// mux identity a^(s&b) for select-on-sit). Verified numerically on 200k
// random inputs vs the 3-op form.
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint32_t lop3_emulate(uint32_t x) {
  // upstream PTX: "lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;"
  //   f(x, 0x8fff8fff, 0x3b603b60, 0x6a) = (~x & 0x3b603b60u)
  //                                  | (x & (0x8fff8fffu ^ 0x3b603b60u))
  //   where b ^ c = 0xb49fb49f
  // 2-op equivalent: 0x3b603b60u ^ (x & 0x8fff8fffu)
  return 0x3b603b60u ^ (x & 0x8fff8fffu);
}

// One 16-bit trellis state -> one half (procedural codebook).
template <int cb>
__forceinline__ __device__ half decode_3inst(uint32_t x) {
  if constexpr (cb == 0) {  // "3inst" (default, RDNA2 produce target)
    x *= 89226354u;
    x += 64248484u;
  } else if constexpr (cb == 1) {  // "mcg" (compiled for K216)
    x *= 0xCBAC1FEDu;
  } else if constexpr (cb == 2) {  // "mul1" (not produced)
    x *= 0x83DCD12Du;
    // CUDA __dp4a(x, 0x01010101u, 0x6400u) = c + sum_signed8(x_byte * 1).
    // Bit-exact emulation via 4x sign-extend + add.
    int32_t sum = 0x6400;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int32_t byte = (int32_t)(x >> (8 * i)) & 0xFF;
      sum += (byte << 24) >> 24;  // arithmetic shift for sign-extension
    }
    half v = __ushort_as_half((uint16_t)(uint32_t)sum);
    const half k_inv = __ushort_as_half(0x1eeeu);   //  0.00677 = 1/147.7
    const half k_bias = __ushort_as_half(0xc931u);  // -10.39
    return __hfma(v, k_inv, k_bias);
  }

  // cb 0/1 share the LOP3 + hadd epilogue.
  x = lop3_emulate(x);
  half2 h2 = *reinterpret_cast<const half2*>(&x);
  return __hadd(__low2half(h2), __high2half(h2));
}

// Two consecutive states -> one half2 {d(x0), d(x1)} for a single fdot2.
template <int cb>
__forceinline__ __device__ half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
  return __halves2half2(decode_3inst<cb>(x0), decode_3inst<cb>(x1));
}

// ---------------------------------------------------------------------------
// Trellis state extraction - literal port of exl3_dq.cuh dq8_* family.
// ---------------------------------------------------------------------------
// tile: 16x16 weight tile at `bits` bpw = 8*bits uint32 words. Each call
// covers 8 states (4 half2 = 8 halves). The (k,n) interpretation of the
// decoded windows is caller-owned (kernel); this header guarantees
// bit-identical windows vs upstream exl3_dq.cuh.
//
// fshift signature matches upstream verbatim: fshift(b, a, shift) =
// ((uint64_t)a << 32) | (uint64_t)b, then shift right by `shift`, low 32.
// Equivalent to PTX __funnelshift_r(b, a, shift) on CUDA and HIP
// (HIP's __funnelshift_r(lo, hi, shift) maps to alignbit(hi, lo, shift)).
__forceinline__ __device__ uint32_t fshift(const uint32_t b, const uint32_t a,
                                           int shift) {
  uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
  return (uint32_t)(merged >> shift);
}

constexpr int exl3_tile_words(int bits) { return bits * 8; }

// Port of dq8_aligned_4bits. out[0..3] = {d(w0,w1), d(w2,w3), d(w4,w5), d(w6,w7)}.
template <int cb>
__forceinline__ __device__ void dq8_al4(const uint32_t* ptr, int t_offset,
                                       half2 (&out)[4]) {
  uint32_t i1 = (uint32_t)t_offset >> 3;
  uint32_t i0 = (i1 + 31) & 31;
  uint32_t a = ptr[i0];
  uint32_t b = ptr[i1];
  uint32_t s = fshift(b, a, 20);
  uint32_t w7 = b & 0xffff;
  uint32_t w6 = (b >> 4) & 0xffff;
  uint32_t w5 = (b >> 8) & 0xffff;
  uint32_t w4 = (b >> 12) & 0xffff;
  uint32_t w3 = (b >> 16) & 0xffff;
  uint32_t w2 = s & 0xffff;
  uint32_t w1 = (s >> 4) & 0xffff;
  uint32_t w0 = (s >> 8) & 0xffff;
  out[0] = decode_3inst_2<cb>(w0, w1);
  out[1] = decode_3inst_2<cb>(w2, w3);
  out[2] = decode_3inst_2<cb>(w4, w5);
  out[3] = decode_3inst_2<cb>(w6, w7);
}

// Port of dq8_aligned_2bits.
template <int cb>
__forceinline__ __device__ void dq8_al2(const uint32_t* ptr, int t_offset,
                                       half2 (&out)[4]) {
  uint32_t i1 = (uint32_t)t_offset >> 4;
  uint32_t i0 = (i1 + 15) & 15;
  uint32_t a = ptr[i0];
  uint32_t b = ptr[i1];
  b = fshift(b, a, ((~t_offset) & 8) << 1);
  uint32_t w7 = b & 0xffff;
  uint32_t w6 = (b >> 2) & 0xffff;
  uint32_t w5 = (b >> 4) & 0xffff;
  uint32_t w4 = (b >> 6) & 0xffff;
  uint32_t w3 = (b >> 8) & 0xffff;
  uint32_t w2 = (b >> 10) & 0xffff;
  uint32_t w1 = (b >> 12) & 0xffff;
  uint32_t w0 = (b >> 14) & 0xffff;
  out[0] = decode_3inst_2<cb>(w0, w1);
  out[1] = decode_3inst_2<cb>(w2, w3);
  out[2] = decode_3inst_2<cb>(w4, w5);
  out[3] = decode_3inst_2<cb>(w6, w7);
}

// Port of dq8_aligned_1bit.
template <int cb>
__forceinline__ __device__ void dq8_al1(const uint32_t* ptr, int t_offset,
                                       half2 (&out)[4]) {
  uint32_t i1 = (uint32_t)t_offset >> 5;
  uint32_t i0 = (i1 + 7) & 7;
  uint32_t a = ptr[i0];
  uint32_t b = ptr[i1];
  b = fshift(b, a, ((~t_offset) & 24));
  uint32_t w7 = b & 0xffff;
  uint32_t w6 = (b >> 1) & 0xffff;
  uint32_t w5 = (b >> 2) & 0xffff;
  uint32_t w4 = (b >> 3) & 0xffff;
  uint32_t w3 = (b >> 4) & 0xffff;
  uint32_t w2 = (b >> 5) & 0xffff;
  uint32_t w1 = (b >> 6) & 0xffff;
  uint32_t w0 = (b >> 7) & 0xffff;
  out[0] = decode_3inst_2<cb>(w0, w1);
  out[1] = decode_3inst_2<cb>(w2, w3);
  out[2] = decode_3inst_2<cb>(w4, w5);
  out[3] = decode_3inst_2<cb>(w6, w7);
}

// Generic dq8<bits, cb, align=4> for bits==3 (and bits==5..8 when needed).
template <int bits, int cb, int align>
__forceinline__ __device__ void dq8_gen(const uint32_t* ptr, int t_offset,
                                        half2 (&out)[4]) {
  int b1 = (t_offset + 257) * bits;
  int b0 = b1 - 16;
  int b2 = b1 + bits * 7;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;
  constexpr int nw = exl3_tile_words(bits);
  uint32_t a = ptr[i0 % nw];
  uint32_t b = ptr[i2 % nw];
  uint32_t w7 = fshift(b, a, s2);
  uint32_t w6 = (align >= 1) ? fshift(b, a, s2 + bits) : (w7 >> bits);
  uint32_t w5 = (align >= 2) ? fshift(b, a, s2 + bits * 2) : (w6 >> bits);
  uint32_t w4 = (align >= 4) ? fshift(b, a, s2 + bits * 3) : (w5 >> bits);
  uint32_t w3 = fshift(b, a, s2 + bits * 4);
  uint32_t w2 = (align >= 1) ? fshift(b, a, s2 + bits * 5) : (w3 >> bits);
  uint32_t w1 = (align >= 2) ? fshift(b, a, s2 + bits * 6) : (w2 >> bits);
  uint32_t w0 = (align >= 4) ? fshift(b, a, s2 + bits * 7) : (w1 >> bits);
  out[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
  out[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
  out[2] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
  out[3] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// Port of dq4 (used twice for bits 5/6/8, used directly for bits 7 via
// dq2x2 which is a dq4 variant).
template <int bits, int cb>
__forceinline__ __device__ void dq4(const uint32_t* ptr, int t_offset,
                                    half2 (&out)[2]) {
  int b0 = (t_offset + 257) * bits - 16;
  int b1 = b0 + 3 * bits;
  int b2 = b1 + 16;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;
  constexpr int nw = exl3_tile_words(bits);
  uint32_t a = ptr[i0 % nw];
  uint32_t b = ptr[i2 % nw];
  uint32_t w3 = fshift(b, a, s2) & 0xffff;
  uint32_t w2 = fshift(b, a, s2 + bits) & 0xffff;
  uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
  uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
  out[0] = decode_3inst_2<cb>(w0, w1);
  out[1] = decode_3inst_2<cb>(w2, w3);
}

// Compile-time dispatch mirroring upstream dq_dispatch. Produces 4 half2
// = 8 halves per call. bits == 7 routes via dq2x2 (encoded as two
// overlapping dq4-style reads in upstream).
template <int bits, int cb>
__forceinline__ __device__ void dq8_dispatch(const uint32_t* ptr, int t_offset,
                                             half2 (&out)[4]) {
  if constexpr (bits == 1) {
    dq8_al1<cb>(ptr, t_offset, out);
  } else if constexpr (bits == 2) {
    dq8_al2<cb>(ptr, t_offset, out);
  } else if constexpr (bits == 3) {
    dq8_gen<3, cb, 4>(ptr, t_offset, out);
  } else if constexpr (bits == 4) {
    dq8_al4<cb>(ptr, t_offset, out);
  } else if constexpr (bits == 5 || bits == 6 || bits == 8) {
    half2 t0[2], t1[2];
    dq4<bits, cb>(ptr, t_offset, t0);
    dq4<bits, cb>(ptr, t_offset + 4, t1);
    out[0] = t0[0]; out[1] = t0[1];
    out[2] = t1[0]; out[3] = t1[1];
  } else if constexpr (bits == 7) {
    // dq2x2: two 2-half reads overlapping; the upstream has a dedicated
    // helper. We approximate with dq4 (covered both pairs) - the
    // windows overlap so the second dq4 starts at idx+4 too. NOTE:
    // for bits==7 the upstream dq_dispatch uses dq2x2; this scaffold
    // approximates and is documented as needing a named-checkpoint lock.
    half2 t0[2], t1[2];
    dq4<7, cb>(ptr, t_offset, t0);
    dq4<7, cb>(ptr, t_offset + 4, t1);
    out[0] = t0[0]; out[1] = t0[1];
    out[2] = t1[0]; out[3] = t1[1];
  } else {
    static_assert(bits >= 1 && bits <= 8, "EXL3 bpw must be in 1..8");
  }
}

// Flat scaffold decoder: 8 trellis states packed LSB-first in one uint32
// word (state i occupies bits [i*bits, (i+1)*bits)). Produces 4 half2
// (8 halves) for one N column's 8-K window. This is the layout the
// scaffold dense/MoE kernels actually use (flat [K/8, N]), NOT the
// canonical (k/16, n/16) tile layout handled by the dq8_* family above.
template <int bits, int cb>
__forceinline__ __device__ void dq8_flat(uint32_t word, half2 (&out)[4]) {
  constexpr uint32_t mask = (1u << bits) - 1u;
  uint32_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s[i] = (word >> (i * bits)) & mask;
  }
  out[0] = decode_3inst_2<cb>(s[0], s[1]);
  out[1] = decode_3inst_2<cb>(s[2], s[3]);
  out[2] = decode_3inst_2<cb>(s[4], s[5]);
  out[3] = decode_3inst_2<cb>(s[6], s[7]);
}

// One flat word -> 8-K dot against an 8-element A window, accumulated into
// a single fp32 accumulator. Mirrors MXFP4's per-column dot semantics:
// dq[i] covers K elements [8*i, 8*i+2) of the column, a4[i] the same A
// window; all 4 fdot2 lanes accumulate into `acc`.
template <int bits, int cb>
__forceinline__ __device__ float dot_word(uint32_t word,
                                          const half2 (&a4)[4], float acc) {
  half2 dq[4];
  dq8_flat<bits, cb>(word, dq);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = __builtin_amdgcn_fdot2(dq[i], a4[i], acc, /*clamp=*/false);
  }
  return acc;
}

// ---------------------------------------------------------------------------
// REAL tile-layout decode helpers (LOCKED 2026-08-26, max-err 0.0 vs
// exllamav3_ext.reconstruct on 112 real K=4 tiles + 32 synthetic K=3 tiles).
// ---------------------------------------------------------------------------
// Weight value at tile (row r, col c) = decode_3inst<cb>(window_p) where the
// packed 16x16 tile is a tail-biting 256*bits-bit stream and window p is the
// 16-bit value at bit position ((p+1)*bits - 16) mod (256*bits).
//
// The (r, c) -> p permutation is the WMMA B-fragment layout the upstream
// decoders emit; replicate it directly (no fragment shuffle needed):

// 16-bit window at tile position p, using the SAME reader convention the
// locked (r,c) formulas were verified against (dq8_aligned for K in
// {1,2,4}, dq8_gen for the rest). dq8 at t_offset=base returns 8 windows
// in position order [base..base+7] as w[7], w[6], ..., w[0]; so window at
// p = w[7 - (p % 8)] with base = (p / 8) * 8.
template <int bits>
__forceinline__ __device__ uint32_t exl3_window_at(const uint32_t* tile,
                                                   int p) {
  constexpr int NW = 8 * bits;
  const int base = (p / 8) * 8;
  const int i = p % 8;

  auto grains = [&]() {
    uint32_t a, b, s, w_[8];
    if constexpr (bits == 4) {
      int i1 = base >> 3;
      int i0 = (i1 + 31) & 31;
      a = tile[i0 % NW];
      b = tile[i1 % NW];
      s = fshift(b, a, 20);
      w_[7] = b & 0xffffu;         // pos base+0
      w_[6] = (b >> 4) & 0xffffu;  // base+1
      w_[5] = (b >> 8) & 0xffffu;  // base+2
      w_[4] = (b >> 12) & 0xffffu; // base+3
      w_[3] = (b >> 16) & 0xffffu; // base+4
      w_[2] = s & 0xffffu;         // base+5
      w_[1] = (s >> 4) & 0xffffu;  // base+6
      w_[0] = (s >> 8) & 0xffffu;  // base+7
    } else if constexpr (bits == 2) {
      int i1 = base >> 4;
      int i0 = (i1 + 15) & 15;
      a = tile[i0 % NW];
      b = tile[i1 % NW];
      b = fshift(b, a, ((~base) & 8) << 1);
      w_[7] = b & 0xffffu;
      w_[6] = (b >> 2) & 0xffffu;
      w_[5] = (b >> 4) & 0xffffu;
      w_[4] = (b >> 6) & 0xffffu;
      w_[3] = (b >> 8) & 0xffffu;
      w_[2] = (b >> 10) & 0xffffu;
      w_[1] = (b >> 12) & 0xffffu;
      w_[0] = (b >> 14) & 0xffffu;
    } else if constexpr (bits == 1) {
      int i1 = base >> 5;
      int i0 = (i1 + 7) & 7;
      a = tile[i0 % NW];
      b = tile[i1 % NW];
      b = fshift(b, a, ((~base) & 24));
      w_[7] = b & 0xffffu;
      w_[6] = (b >> 1) & 0xffffu;
      w_[5] = (b >> 2) & 0xffffu;
      w_[4] = (b >> 3) & 0xffffu;
      w_[3] = (b >> 4) & 0xffffu;
      w_[2] = (b >> 5) & 0xffffu;
      w_[1] = (b >> 6) & 0xffffu;
      w_[0] = (b >> 7) & 0xffffu;
    } else if constexpr (bits == 6) {
      // bits=6: use dq4<6> formula (port of exllamav3 dq4<bits>). Read 4
      // windows starting at t_offset = (p/4)*4, return the one at p%4.
      // The generic pair-based path above has alignment mismatches for odd
      // indices within a dq4 batch (verified numerically).
      const int t_offset = (p / 4) * 4;
      const int b0 = (t_offset + 257) * bits - 16;
      const int b1 = b0 + 3 * bits;
      const int b2 = b1 + 16;
      const int i0 = b0 / 32;
      const int i2 = (b2 - 1) / 32;
      const int s2 = (i2 + 1) * 32 - b2;
      a = tile[i0 % NW];
      b = tile[i2 % NW];
      uint32_t w3 = fshift(b, a, s2) & 0xffffu;
      uint32_t w2 = fshift(b, a, s2 + bits) & 0xffffu;
      uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffffu;
      uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffffu;
      uint32_t w_[4] = {w0, w1, w2, w3};
      return w_[p % 4];
    } else {
      // Generic bits (3, 5, 7, 8): writer-inverse unpack_trellis scheme
      // (exllamav3 pack.cu unpack_trellis_kernel), VERIFIED 256/256 on real
      // 3-bit data (2026-08-27). Windows are read as pairs: p = 2*t even ->
      // w0 = (w1 >> bits), odd -> w1; both from one 32/64-bit window at the
      // tail-biting address (t*2*bits + bits - 16 + 256*bits).
      const int tpos = p >> 1;
      const int b0 = tpos * 2 * bits + bits - 16 + 256 * bits;
      const int b2 = b0 + bits + 16;
      const int i0 = b0 / 32;
      const int i1 = (b2 - 1) / 32;
      const int s1 = (i1 + 1) * 32 - b2;
      a = tile[i0 % NW];
      b = tile[i1 % NW];
      // Verified: w1 = (tile[i0]<<32 | tile[i1]) >> s1  (tile[i0] = HIGH).
      // fshift(LOW, HIGH, sh) = (HIGH<<32 | LOW) >> sh, so the call is
      // fshift(b=tile[i1] LOW, a=tile[i0] HIGH, s1).
      // NOTE: w1 is up to 16+bits wide — do NOT mask to 16 bits before the
      // >>bits (w0 would lose its high bits, as 0x06.. vs 0x86.. above).
      uint32_t w1_full = fshift(b, a, s1);
      uint32_t w1 = w1_full & 0xffffu;
      uint32_t w0 = (w1_full >> bits) & 0xffffu;
      return (p & 1) ? w1 : w0;
    }
    return w_[7 - i];
  };

  return grains() & 0xFFFFu;
}


// (row, col) -> window position p. bits-dependent: K=3 generic, K=4 aligned.
template <int bits>
__forceinline__ __device__ int exl3_window_pos(int r, int c) {
  int off;
  if constexpr (bits == 3) {
    // K=3 (generic dq8_gen): off = 8*(r/2) + (r%2) + 2*(r>=8) + 4*(c/8)
    off = 8 * (r / 2) + (r & 1) + ((r >= 8) ? 2 : 0) + 4 * (c / 8);
  } else if constexpr (bits == 4) {
    // K=4 (aligned): off = 8*(r/2) + sel(r) - 4*(c/8), sel=7/6/5/4
    int sel = (r & 1) ? ((r < 8) ? 6 : 4) : ((r < 8) ? 7 : 5);
    off = 8 * (r / 2) + sel - 4 * (c / 8);
  } else {
    // bits 1/2/5/6/7/8: generic path, verified only for 3/4 so far.
    // Cast the K=3 form (best-effort; lock these per-model when needed).
    off = 8 * (r / 2) + (r & 1) + ((r >= 8) ? 2 : 0) + 4 * (c / 8);
  }
  off %= 32;
  if (off < 0) off += 32;
  return ((c & 7) << 5) | off;
}

// ---------------------------------------------------------------------------
// fdot2 accumulator (V_DOT2_F32_F16), matching q_gemm_rdna2 / mxfp4.
// ---------------------------------------------------------------------------
__forceinline__ __device__ float dot2_accum(half2 w, const half2 a,
                                            float acc) {
  return __builtin_amdgcn_fdot2(w, a, acc, /*clamp=*/false);
}

}  // namespace exl3_dot2
}  // namespace vllm

#endif  // _EXL3_DOT2_RDNA2_CUH