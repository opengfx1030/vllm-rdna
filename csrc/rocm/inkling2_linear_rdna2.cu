// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dense linears of Inkling-2-Mini for RDNA2 (gfx1030, wave32, v_dot2_f32_f16),
// every projection of the model as a native HIP op (docs/rdna2/
// inkling2-linear-kernels.md):
//
//   inkling2_linear_grouped   fp16 W[N,K] x fp16 X[M,K] -> fp16/fp32, S
//                             segments (own X, rows of W, out, epilogue) in ONE
//                             launch: q|gate|rel|sink, k|dynk, v|dynv, o_proj,
//                             down_proj. M <= 16: weight-streaming GEMV; M >
//                             16: register-tiled dot2 GEMM.
//   inkling2_linear_lm_head   the same with softcap, vocab mask and an optional
//                             per-block argmax partial (logits stay fp32
//                             [M, N]).
//   inkling2_linear_f32w      fp32 W (router, controller gates) x fp16 X with
//                             fp32 FMA.
//   inkling2_linear_argmax_partial / _reduce
//                             argmax over logits / over the partials.
//
// All ops write caller-owned outputs, allocate nothing, never synchronise and
// launch on the current stream with static shapes, so they are
// graph-capturable.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace inkling2_linear {

constexpr int kMaxSeg = 8;
constexpr int kNumWaves = 8;           // GEMV block = 8 waves of 32
constexpr int kLdsBudget = 48 * 1024;  // GEMV activation chunk, bytes
[[maybe_unused]] constexpr int kActNone = 0;
[[maybe_unused]] constexpr int kActSigmoid = 1;
[[maybe_unused]] constexpr int kActTanh = 2;
[[maybe_unused]] constexpr int kActSoftcap = 3;

// One segment: rows [0, n) of the weights at `w` against the activations `x`.
struct Seg {
  const void* x;  // [M, K] fp16, row stride ldx elements
  const void* w;  // [n, K] row-major (fp16 or fp32 per op)
  void* out;      // [M, n] row stride ldo elements
  int ldx;
  int ldo;
  int n;
  int n_valid;  // columns >= n_valid are written as -inf
  int g_begin;  // first GEMV group / GEMM tile of the segment
  int act;
  int out_f32;
  float p;  // softcap
};

struct Params {
  Seg seg[kMaxSeg];
  int nseg;
  int M;
  int K;
  int groups;       // GEMV: total row groups
  int gpb;          // GEMV: groups per block
  int kc8;          // GEMV: activation chunk, in 8-element vectors
  int nc;           // GEMV: chunks per row
  int write_out;    // GEMV: store the outputs
  int nparts;       // argmax partials per row (grid size)
  int col_base;     // added to the argmax indices (vocab shard start)
  float* part_val;  // [M, nparts]
  int* part_idx;    // [M, nparts]
};

// ---------------------------------------------------------------------------
// Device helpers (gfx1030 only; the host pass sees them for the launch code).
// ---------------------------------------------------------------------------
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using u32x2 = unsigned int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));

// fp32 a.x*b.x + a.y*b.y + c (v_dot2_f32_f16). Operands by value: a
// __builtin_bit_cast of a vector-element lvalue reads element 0 (clang 23).
__device__ __forceinline__ float dot2u(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
}

__device__ __forceinline__ float dot8(uint4 w, uint4 x, float acc) {
  acc = dot2u(w.x, x.x, acc);
  acc = dot2u(w.y, x.y, acc);
  acc = dot2u(w.z, x.z, acc);
  return dot2u(w.w, x.w, acc);
}

__device__ __forceinline__ float half_lo(unsigned v) {
  return static_cast<float>(__builtin_bit_cast(f16x2, v).x);
}
__device__ __forceinline__ float half_hi(unsigned v) {
  return static_cast<float>(__builtin_bit_cast(f16x2, v).y);
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

__device__ __forceinline__ void set_zero(uint4& v) {
  v = make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ void set_zero(float4& v) {
  v = make_float4(0.f, 0.f, 0.f, 0.f);
}

// One weight vector against 8 (fp16 W) or 4 (fp32 W) activations of an LDS row.
__device__ __forceinline__ float vec_dot(const uint4& w, const char* xrow,
                                         int e, float acc) {
  return dot8(w, *reinterpret_cast<const uint4*>(xrow + e * 2), acc);
}
__device__ __forceinline__ float vec_dot(const float4& w, const char* xrow,
                                         int e, float acc) {
  const uint2 x = *reinterpret_cast<const uint2*>(xrow + e * 2);
  acc = fmaf(w.x, half_lo(x.x), acc);
  acc = fmaf(w.y, half_hi(x.x), acc);
  acc = fmaf(w.z, half_lo(x.y), acc);
  return fmaf(w.w, half_hi(x.y), acc);
}

// Sum of N per-lane values over the wave, scattered: after log2(N) exchange
// steps lane l holds the total of value (l >> (5 - log2 N)), then the lanes
// sharing that value butterfly the rest. N shuffles per value instead of 5.
template <int N, int OFF>
__device__ __forceinline__ float scatter_sum(float (&a)[N], int lane) {
  if constexpr (N == 1) {
    float v = a[0];
  #pragma unroll
    for (int o = OFF; o >= 1; o >>= 1) v += __shfl_xor(v, o);
    return v;
  } else {
    const bool hi = (lane & OFF) != 0;
    float b[N / 2];
  #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      const float keep = hi ? a[N / 2 + i] : a[i];
      const float send = hi ? a[i] : a[N / 2 + i];
      b[i] = keep + __shfl_xor(send, OFF);
    }
    return scatter_sum<N / 2, OFF / 2>(b, lane);
  }
}

template <typename T>
struct WTraits;
template <>
struct WTraits<_Float16> {
  using Vec = uint4;
  static constexpr int EPV = 8;  // K elements per vector
};
template <>
struct WTraits<float> {
  using Vec = float4;
  static constexpr int EPV = 4;
};

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Weight-streaming GEMV, M <= 32 (MT = 1, 2, 4, 8, 16).
//
// A group is RB = (8 / KS) R rows. Wave w of a block takes R rows of the group
// and the K slice (w % KS) of them: its 32 lanes stride the slice in 16-byte
// weight vectors, U loads in flight per row, and every vector is applied to
// all MT activation rows held in LDS (so each weight byte is read from DRAM
// once whatever M). A block owns gpb consecutive groups; the activations are
// staged in LDS in nc chunks of kc8 vectors (nc = 1 for M <= 8, K <= 3072:
// staged once per block and segment, not per group).
// ---------------------------------------------------------------------------
template <typename WT, int MT, int R, int U, int KS, bool ARG>
__global__ void __launch_bounds__(256) gemv_kernel(const Params p) {
#ifdef __HIP__RDNA2__
  constexpr int NW = kNumWaves;
  constexpr int RB = (NW / KS) * R;
  constexpr int RM = R * MT;
  constexpr int STRIDE = 32 / RM;  // lane distance between the finished values
  static_assert(RM <= 32 && (RM & (RM - 1)) == 0 && NW % KS == 0, "gemv shape");
  using WV = typename WTraits<WT>::Vec;
  constexpr int EPV = WTraits<WT>::EPV;

  extern __shared__ __attribute__((aligned(16))) uint4 smem[];
  uint4* xs4 = smem;  // [MT][kc8] 16-byte activation vectors
  [[maybe_unused]] float* red =
      reinterpret_cast<float*>(smem + MT * p.kc8);  // KS > 1: [NW][RM]
  [[maybe_unused]] float* bres_v = red + (KS > 1 ? NW * RM : 0);  // ARG
  [[maybe_unused]] int* bres_i =
      reinterpret_cast<int*>(bres_v + (ARG ? NW * RM : 0));

  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 31;
  const int rslot = wave / KS, kslice = wave % KS;
  const int K = p.K;
  const int M = p.M;
  const int kc_el = p.kc8 * 8;  // activation chunk, elements

  const int g_lo = blockIdx.x * p.gpb;
  const int g_hi = min(g_lo + p.gpb, p.groups);
  const void* cur_x = nullptr;
  int cur_ldx = 0;
  [[maybe_unused]] float bv = -__builtin_inff();
  [[maybe_unused]] int bi = 0;

  for (int g = g_lo; g < g_hi; ++g) {
    int s = 0;
  #pragma unroll
    for (int i = 1; i < kMaxSeg; ++i)
      if (i < p.nseg && g >= p.seg[i].g_begin) s = i;
    const Seg& sg = p.seg[s];
    const int row0 = (g - sg.g_begin) * RB + rslot * R;

    const WT* wr[R];
  #pragma unroll
    for (int r = 0; r < R; ++r)
      wr[r] = reinterpret_cast<const WT*>(sg.w) +
              static_cast<size_t>(min(row0 + r, sg.n - 1)) * K;

    float acc[R][MT];
  #pragma unroll
    for (int r = 0; r < R; ++r)
  #pragma unroll
      for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;

    for (int c = 0; c < p.nc; ++c) {
      const int eb = c * kc_el;
      const int ee = min(K, eb + kc_el);
      if (p.nc > 1 || sg.x != cur_x || sg.ldx != cur_ldx) {
        __syncthreads();
        const int nch = (ee - eb) >> 3;
        const uint4* xg = reinterpret_cast<const uint4*>(sg.x);
        const size_t ldx8 = static_cast<size_t>(sg.ldx) >> 3;
  #pragma unroll
        for (int m = 0; m < MT; ++m)
          for (int i = tid; i < nch; i += 256) {
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (m < M) v = xg[static_cast<size_t>(m) * ldx8 + (eb >> 3) + i];
            xs4[m * p.kc8 + i] = v;
          }
        __syncthreads();
        cur_x = sg.x;
        cur_ldx = sg.ldx;
      }

      const int vb = eb / EPV, ve = ee / EPV;
      const int per = (ve - vb + KS - 1) / KS;
      const int sb = vb + kslice * per;
      const int se = min(ve, sb + per);
      const char* xl = reinterpret_cast<const char*>(xs4);
      for (int i0 = sb + lane; i0 < se; i0 += 32 * U) {
        WV wq[R][U];
  #pragma unroll
        for (int r = 0; r < R; ++r)
  #pragma unroll
          for (int u = 0; u < U; ++u) {
            const int idx = i0 + 32 * u;
            WV t;
            set_zero(t);
            if (idx < se) t = *reinterpret_cast<const WV*>(wr[r] + idx * EPV);
            wq[r][u] = t;
          }
  #pragma unroll
        for (int u = 0; u < U; ++u) {
          const int idx = i0 + 32 * u;
          if (idx < se) {
            const int e = idx * EPV - eb;
  #pragma unroll
            for (int m = 0; m < MT; ++m) {
              const char* xr = xl + m * p.kc8 * 16;
  #pragma unroll
              for (int r = 0; r < R; ++r)
                acc[r][m] = vec_dot(wq[r][u], xr, e, acc[r][m]);
            }
          }
        }
      }
    }

    // ---- reduce the 32 lanes (and the KS slices), epilogue, store
    float a[RM];
  #pragma unroll
    for (int r = 0; r < R; ++r)
  #pragma unroll
      for (int m = 0; m < MT; ++m) a[r * MT + m] = acc[r][m];
    float mine = scatter_sum<RM, 16>(a, lane);
    const bool writer = (lane & (STRIDE - 1)) == 0;
    const int j = lane / STRIDE;
    if constexpr (KS > 1) {
      if (writer) red[wave * RM + j] = mine;
      __syncthreads();
      mine = 0.0f;
  #pragma unroll
      for (int k = 0; k < KS; ++k) mine += red[(rslot * KS + k) * RM + j];
      __syncthreads();
    }
    if (kslice == 0 && writer) {
      const int r = j / MT, m = j % MT;
      const int row = row0 + r;
      float val = -__builtin_inff();
      if (row < sg.n) {
        val = epilogue(mine, sg.act, sg.p);
        if (row >= sg.n_valid) val = -__builtin_inff();
      }
      if (m < M && row < sg.n && p.write_out) {
        if (sg.out_f32)
          reinterpret_cast<float*>(
              sg.out)[static_cast<size_t>(m) * sg.ldo + row] = val;
        else
          reinterpret_cast<_Float16*>(
              sg.out)[static_cast<size_t>(m) * sg.ldo + row] =
              static_cast<_Float16>(val);
      }
      if constexpr (ARG) {
        if (val == val && val > bv) {
          bv = val;
          bi = p.col_base + row;
        }
      }
    }
  }

  if constexpr (ARG) {
    if (kslice == 0 && (lane & (STRIDE - 1)) == 0) {
      bres_v[wave * RM + lane / STRIDE] = bv;
      bres_i[wave * RM + lane / STRIDE] = bi;
    }
    __syncthreads();
    if (tid < MT && tid < M) {
      float best_v = -__builtin_inff();
      int best_i = 0;
      for (int w = 0; w < NW; w += KS)
  #pragma unroll
        for (int r = 0; r < R; ++r) {
          const float v = bres_v[w * RM + r * MT + tid];
          const int ix = bres_i[w * RM + r * MT + tid];
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

// ---------------------------------------------------------------------------
// Register-tiled dot2 GEMM, C[M, n] = A[M, K] W[n, K]^T per segment. The tile
// designs are those of rdna_hint_flex_dense_gemm_kernel.cuh (docs/rdna2/
// hi2-prefill-gemm-2026-11-15.md); here with K tails (K % 8 == 0 only), ldc,
// an epilogue (sigmoid / tanh / softcap, -inf mask) and fp16 or fp32 output.
// A tile owns one segment, so every block works on one (A, W, out).
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int BK_, int TM_, int TN_, int NBUF_, int PAD_,
          int RAST_, bool SWZ_, int BROT_>
struct Design {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int NBUF = NBUF_, PAD = PAD_, RAST = RAST_, BROT = BROT_;
  static constexpr bool SWZ = SWZ_;
  static constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;
  static constexpr int LD = BK + PAD;  // halves per LDS row
  static constexpr int CH = 8;         // halves per 16-byte global chunk
  static constexpr int CPR = BK / CH;  // chunks per tile row
  static constexpr int A_CHUNKS = BM * CPR, B_CHUNKS = BN * CPR;
  static constexpr int A_PER_T = (A_CHUNKS + T - 1) / T;
  static constexpr int B_PER_T = (B_CHUNKS + T - 1) / T;
  static constexpr int A_LDS = BM * LD, B_LDS = BN * LD;  // halves
  static constexpr int STAGE = A_LDS + B_LDS;
  static constexpr int KSTEP = 4;  // halves per fragment (ds_read_b64)
  static constexpr int ND = 2;     // dwords per fragment
  static constexpr int SRP = CPR <= 8 ? 8 / CPR : 1;
  static constexpr int LDS_BYTES = NBUF * STAGE * 2;
  static_assert(!SWZ || (PAD == 0 && (CPR == 2 || CPR == 4 || CPR == 8) &&
                         TX % 8 == 0 && TY % 8 == 0),
                "SWZ: PAD 0, BK 16/32/64, TX and TY multiples of 8");
  static_assert(BK % KSTEP == 0 && BK % CH == 0 && LD % 8 == 0, "BK");
  static_assert(BM % TM == 0 && BN % TN == 0, "micro-tile");
  static_assert(T >= 32 && T <= 1024 && T % 32 == 0, "block size");
  static_assert(NBUF == 1 || NBUF == 2, "NBUF");
  static_assert(LDS_BYTES <= 65536, "LDS over 64 KiB per block");
};

using D128x128 = Design<128, 128, 32, 8, 16, 2, 0, 8, true, 1>;
using D256x128 = Design<256, 128, 32, 8, 16, 2, 0, 4, true, 1>;
using D128x64 = Design<128, 64, 64, 8, 8, 1, 8, 4, false, 0>;
using D64x128 = Design<64, 128, 64, 8, 8, 2, 0, 0, true, 0>;
using D32x128 = Design<32, 128, 64, 4, 8, 2, 0, 0, true, 0>;
using D64x64 = Design<64, 64, 64, 4, 8, 2, 0, 0, true, 0>;

template <class P, bool KTAIL>
__global__ void __launch_bounds__(P::T) gemm_kernel(const Params p) {
#ifdef __HIP__RDNA2__
  using Chunk = u32x4;
  using Frag = u32x2;
  constexpr int BM = P::BM, BN = P::BN, BK = P::BK, TM = P::TM, TN = P::TN;
  constexpr int TX = P::TX, TY = P::TY, T = P::T, LD = P::LD, CH = P::CH;
  constexpr int CPR = P::CPR, NBUF = P::NBUF;
  constexpr int A_PER_T = P::A_PER_T, B_PER_T = P::B_PER_T;

  __shared__ __attribute__((aligned(16))) char lds[P::LDS_BYTES];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int swa = P::SWZ ? (ty / P::SRP) & (CPR - 1) : 0;
  const int swb = P::SWZ ? (tx / P::SRP) & (CPR - 1) : 0;
  auto store_off = [](int r, int kcc) {
    if constexpr (P::SWZ)
      return r * LD * 2 + ((kcc ^ ((r / P::SRP) & (CPR - 1))) << 4);
    else
      return (r * LD + kcc * CH) * 2;
  };
  auto frag_off = [](int kk, int sw) {
    if constexpr (P::SWZ)
      return (((kk >> 3) ^ sw) << 4) + ((kk & 7) << 1);
    else
      return kk * 2;
  };

  // ---- block -> (segment, tile)
  const int bid = blockIdx.x;
  int s = 0;
  #pragma unroll
  for (int i = 1; i < kMaxSeg; ++i)
    if (i < p.nseg && bid >= p.seg[i].g_begin) s = i;
  const Seg& sg = p.seg[s];
  const int M = p.M, N = sg.n, K = p.K, lda = sg.ldx, ldw = K;
  const char* A = reinterpret_cast<const char*>(sg.x);
  const char* W = reinterpret_cast<const char*>(sg.w);
  const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
  const int lb = bid - sg.g_begin;
  int tm, tn;
  if constexpr (P::RAST == 0) {
    tn = lb % ntn;
    tm = lb / ntn;
  } else {
    constexpr int GM = P::RAST;
    const int per_group = GM * ntn;
    const int group = lb / per_group;
    const int first_m = group * GM;
    const int gsz = min(ntm - first_m, GM);
    tm = first_m + (lb % per_group) % gsz;
    tn = (lb % per_group) / gsz;
  }
  const int m0 = tm * BM, n0 = tn * BN;
  const int nk = (K + BK - 1) / BK;

  const int w0 = __builtin_amdgcn_readfirstlane(tid) & ~31;
  const bool wave_live =
      (m0 + w0 / TX < M) && (n0 + (TX >= 32 ? w0 % TX : 0) < N);

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
        const Chunk q =
            __builtin_amdgcn_raw_buffer_load_b128(rsa, aoff[v], kb, 0);
        if constexpr (KTAIL) {
          const int kc = ((tid + v * T) % CPR) * CH;
          ra[v] = (kt * BK + kc < K) ? q : Chunk{0u, 0u, 0u, 0u};
        } else {
          ra[v] = q;
        }
      }
  #pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      if (B_PER_T * T == P::B_CHUNKS || tid + v * T < P::B_CHUNKS) {
        const Chunk q =
            __builtin_amdgcn_raw_buffer_load_b128(rsw, boff[v], kb, 0);
        if constexpr (KTAIL) {
          const int kc = ((tid + v * T) % CPR) * CH;
          rb[v] = (kt * BK + kc < K) ? q : Chunk{0u, 0u, 0u, 0u};
        } else {
          rb[v] = q;
        }
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
          // dword d of the chunk lands at (d + 1) mod 2 of its fragment, so the
          // two sources of a dot2 sit in different VGPR banks
          const Chunk q = rb[v];
          *reinterpret_cast<Chunk*>(dst) = Chunk{q.y, q.x, q.w, q.z};
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
  #pragma unroll 1
    for (int kk = 0; kk < BK; kk += P::KSTEP) {
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
    }
  };
  auto run = [&](int buf) {
    if (!wave_live) return;
    compute(buf);
  };

  if constexpr (NBUF == 2) {
    gload(0);
    lstore(0);
    __syncthreads();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      gload(kt + 1);
      run(0);
      lstore(1);
      __syncthreads();
      if (kt + 2 < nk) gload(kt + 2);
      run(1);
      if (kt + 2 < nk) lstore(0);
      __syncthreads();
    }
    if (kt < nk) run(0);
  } else {
    gload(0);
    for (int kt = 0; kt < nk; ++kt) {
      __syncthreads();
      lstore(0);
      __syncthreads();
      if (kt + 1 < nk) gload(kt + 1);
      run(0);
    }
  }

  // ---- epilogue: fp32 accumulators -> act -> fp16 / fp32
  const int ldo = sg.ldo, act = sg.act, nv = sg.n_valid;
  const float cap = sg.p;
  auto emit = [&](auto* C, auto generic) __attribute__((always_inline)) {
  #pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = m0 + ty + TY * i;
      if (row < M) {
  #pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int col = n0 + tx + TX * j;
          if (col < N) {
            float v = acc[i][j];
            if constexpr (decltype(generic)::value) {
              v = epilogue(v, act, cap);
              if (col >= nv) v = -__builtin_inff();
            }
            C[static_cast<size_t>(row) * ldo + col] =
                static_cast<std::remove_pointer_t<decltype(C)>>(v);
          }
        }
      }
    }
  };
  if (act == kActNone && nv >= N) {
    if (sg.out_f32)
      emit(reinterpret_cast<float*>(sg.out), std::false_type{});
    else
      emit(reinterpret_cast<_Float16*>(sg.out), std::false_type{});
  } else {
    if (sg.out_f32)
      emit(reinterpret_cast<float*>(sg.out), std::true_type{});
    else
      emit(reinterpret_cast<_Float16*>(sg.out), std::true_type{});
  }
#endif  // __HIP__RDNA2__
}

// ---------------------------------------------------------------------------
// fp32-weight GEMM (router, controller gates): C[M, n] fp32 = X[M, K] fp16 x
// W[n, K]^T fp32, fp32 FMA, 64 x 128 tile, 256 threads, 4 x 8 per thread.
// ---------------------------------------------------------------------------
template <bool KTAIL>
__global__ void __launch_bounds__(256) gemm_f32w_kernel(const Params p) {
#ifdef __HIP__RDNA2__
  constexpr int BM = 64, BN = 128, BK = 16, TM = 4, TN = 8;
  constexpr int TX = BN / TN, TY = BM / TM;  // 16 x 16
  constexpr int LDA = BK + 1, LDB = BK + 1;  // fp32 elements, odd stride
  __shared__ float as[BM * LDA];
  __shared__ float bs[BN * LDB];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int bid = blockIdx.x;
  int s = 0;
  #pragma unroll
  for (int i = 1; i < kMaxSeg; ++i)
    if (i < p.nseg && bid >= p.seg[i].g_begin) s = i;
  const Seg& sg = p.seg[s];
  const int M = p.M, N = sg.n, K = p.K, lda = sg.ldx;
  const _Float16* A = reinterpret_cast<const _Float16*>(sg.x);
  const float* W = reinterpret_cast<const float*>(sg.w);
  const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
  const int lb = bid - sg.g_begin;
  const int tm = lb % ntm, tn = lb / ntm;  // M fastest: W tile shared in L2
  const int m0 = tm * BM, n0 = tn * BN;
  (void)ntn;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile: 64 rows x 16 halves = 128 chunks of 8; B tile: 128 x 16 fp32
    if (tid < 128) {
      const int row = tid >> 1, kc = (tid & 1) * 8;
      const int grow = min(m0 + row, M - 1);
      uint4 q = make_uint4(0u, 0u, 0u, 0u);
      if (!KTAIL || k0 + kc < K)
        q = *reinterpret_cast<const uint4*>(
            A + static_cast<size_t>(grow) * lda + k0 + kc);
      float* d = as + row * LDA + kc;
      d[0] = half_lo(q.x);
      d[1] = half_hi(q.x);
      d[2] = half_lo(q.y);
      d[3] = half_hi(q.y);
      d[4] = half_lo(q.z);
      d[5] = half_hi(q.z);
      d[6] = half_lo(q.w);
      d[7] = half_hi(q.w);
    }
  #pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int c = tid + v * 256;  // 512 chunks of 4 fp32
      const int row = c >> 2, kc = (c & 3) * 4;
      const int grow = min(n0 + row, N - 1);
      float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
      if (!KTAIL || k0 + kc < K)
        q = *reinterpret_cast<const float4*>(W + static_cast<size_t>(grow) * K +
                                             k0 + kc);
      float* d = bs + row * LDB + kc;
      d[0] = q.x;
      d[1] = q.y;
      d[2] = q.z;
      d[3] = q.w;
    }
    __syncthreads();
  #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float af[TM], bf[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = as[(ty + TY * i) * LDA + kk];
  #pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = bs[(tx + TX * j) * LDB + kk];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  float* C = reinterpret_cast<float*>(sg.out);
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + TY * i;
    if (row < M) {
  #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + tx + TX * j;
        if (col < N) {
          float v = epilogue(acc[i][j], sg.act, sg.p);
          if (col >= sg.n_valid) v = -__builtin_inff();
          C[static_cast<size_t>(row) * sg.ldo + col] = v;
        }
      }
    }
  }
#endif  // __HIP__RDNA2__
}

// ---------------------------------------------------------------------------
// Argmax: partial over logits (one block = one row x one column slice), and
// the reduce over the partials. NaN counts as -inf, ties take the lowest index.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    argmax_partial_kernel(const float* __restrict__ logits, int64_t ld, int n,
                          int chunk, int nparts, int col_base,
                          float* __restrict__ part_val,
                          int* __restrict__ part_idx) {
#ifdef __HIP__RDNA2__
  const int row = blockIdx.y, part = blockIdx.x;
  const int c0 = part * chunk, c1 = min(n, c0 + chunk);
  float bv = -__builtin_inff();
  int bi = c0;
  for (int c = c0 + threadIdx.x; c < c1; c += 256) {
    const float v = logits[row * ld + c];
    if (v == v && (v > bv || (v == bv && c < bi))) {
      bv = v;
      bi = c;
    }
  }
  __shared__ float sv[256];
  __shared__ int si[256];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int off = 128; off >= 1; off >>= 1) {
    if (threadIdx.x < off) {
      const float v = sv[threadIdx.x + off];
      const int ix = si[threadIdx.x + off];
      if (v > sv[threadIdx.x] ||
          (v == sv[threadIdx.x] && ix < si[threadIdx.x])) {
        sv[threadIdx.x] = v;
        si[threadIdx.x] = ix;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_val[row * nparts + part] = sv[0];
    part_idx[row * nparts + part] = si[0] + col_base;
  }
#endif  // __HIP__RDNA2__
}

__global__ void __launch_bounds__(256)
    argmax_reduce_kernel(const float* __restrict__ part_val,
                         const int* __restrict__ part_idx, int nparts,
                         int64_t* __restrict__ out) {
#ifdef __HIP__RDNA2__
  const int row = blockIdx.x;
  float bv = -__builtin_inff();
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < nparts; i += 256) {
    const float v = part_val[row * nparts + i];
    const int ix = part_idx[row * nparts + i];
    if (v > bv || (v == bv && ix < bi)) {
      bv = v;
      bi = ix;
    }
  }
  __shared__ float sv[256];
  __shared__ int si[256];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int off = 128; off >= 1; off >>= 1) {
    if (threadIdx.x < off) {
      const float v = sv[threadIdx.x + off];
      const int ix = si[threadIdx.x + off];
      if (v > sv[threadIdx.x] ||
          (v == sv[threadIdx.x] && ix < si[threadIdx.x])) {
        sv[threadIdx.x] = v;
        si[threadIdx.x] = ix;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = static_cast<int64_t>(si[0]);
#endif  // __HIP__RDNA2__
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

struct GemvCfg {
  int R, U, KS, bpc;
};

// Variant ids: 0.. GEMV configs, kGemmBase.. GEMM designs, -1 auto.
constexpr GemvCfg kGemvCfgs[] = {
    {1, 4, 1, 2},  // 0
    {2, 3, 1, 2},  // 1
    {4, 2, 1, 2},  // 2
    {1, 4, 2, 2},  // 3
    {1, 4, 4, 2},  // 4
    {2, 2, 4, 2},  // 5
};
constexpr int kNumGemvCfgs = sizeof(kGemvCfgs) / sizeof(kGemvCfgs[0]);
// GEMV variant id = cfg + 10 * s: s > 0 replaces the persistent blocks per CU
// of the cfg by kBpcSel[s] (tuning knob, no extra kernel instantiations).
constexpr int kBpcSel[] = {0, 1, 4, 8, 16, 32};
constexpr int kNumBpcSel = sizeof(kBpcSel) / sizeof(kBpcSel[0]);
constexpr int kGemmBase = 100;
constexpr int kNumDesigns = 6;
constexpr int kDesignBM[kNumDesigns] = {128, 256, 128, 64, 32, 64};
constexpr int kDesignBN[kNumDesigns] = {128, 128, 64, 128, 128, 64};
constexpr int kGemmMinM = 17;  // M <= 16 runs the GEMV

int ceil_div(int a, int b) { return (a + b - 1) / b; }

int num_mp() {
  static const int v =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return v;
}

int mt_for(int M) {
  return M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16;
}

template <typename WT, int MT, int R, int U, int KS, bool ARG>
void launch_gemv_one(const Params& pr, int grid, size_t lds,
                     hipStream_t stream) {
  if constexpr (R * MT <= 32) {
    gemv_kernel<WT, MT, R, U, KS, ARG><<<grid, 256, lds, stream>>>(pr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    TORCH_CHECK(false,
                "inkling2_linear: gemv variant needs rows per wave * "
                "activation rows <= 32");
  }
}

template <typename WT, int MT, bool ARG>
void launch_gemv_cfg(int cfg, const Params& pr, int grid, size_t lds,
                     hipStream_t st) {
  switch (cfg) {
    case 0:
      launch_gemv_one<WT, MT, 1, 4, 1, ARG>(pr, grid, lds, st);
      break;
    case 1:
      launch_gemv_one<WT, MT, 2, 3, 1, ARG>(pr, grid, lds, st);
      break;
    case 2:
      launch_gemv_one<WT, MT, 4, 2, 1, ARG>(pr, grid, lds, st);
      break;
    case 3:
      launch_gemv_one<WT, MT, 1, 4, 2, ARG>(pr, grid, lds, st);
      break;
    case 4:
      launch_gemv_one<WT, MT, 1, 4, 4, ARG>(pr, grid, lds, st);
      break;
    case 5:
      launch_gemv_one<WT, MT, 2, 2, 4, ARG>(pr, grid, lds, st);
      break;
    default:
      TORCH_CHECK(false, "inkling2_linear: bad gemv variant ", cfg);
  }
}

template <typename WT, bool ARG>
void launch_gemv_mt(int mt, int cfg, const Params& pr, int grid, size_t lds,
                    hipStream_t st) {
  switch (mt) {
    case 1:
      launch_gemv_cfg<WT, 1, ARG>(cfg, pr, grid, lds, st);
      break;
    case 2:
      launch_gemv_cfg<WT, 2, ARG>(cfg, pr, grid, lds, st);
      break;
    case 4:
      launch_gemv_cfg<WT, 4, ARG>(cfg, pr, grid, lds, st);
      break;
    case 8:
      launch_gemv_cfg<WT, 8, ARG>(cfg, pr, grid, lds, st);
      break;
    default:
      launch_gemv_cfg<WT, 16, ARG>(cfg, pr, grid, lds, st);
      break;
  }
}

template <class P>
void launch_gemm_design(const Params& pr, int tiles, bool ktail,
                        hipStream_t st) {
  if (ktail)
    gemm_kernel<P, true><<<tiles, P::T, 0, st>>>(pr);
  else
    gemm_kernel<P, false><<<tiles, P::T, 0, st>>>(pr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemm(int design, const Params& pr, int tiles, bool ktail,
                 hipStream_t st) {
  switch (design) {
    case 0:
      launch_gemm_design<D128x128>(pr, tiles, ktail, st);
      break;
    case 1:
      launch_gemm_design<D256x128>(pr, tiles, ktail, st);
      break;
    case 2:
      launch_gemm_design<D128x64>(pr, tiles, ktail, st);
      break;
    case 3:
      launch_gemm_design<D64x128>(pr, tiles, ktail, st);
      break;
    case 4:
      launch_gemm_design<D32x128>(pr, tiles, ktail, st);
      break;
    case 5:
      launch_gemm_design<D64x64>(pr, tiles, ktail, st);
      break;
    default:
      TORCH_CHECK(false, "inkling2_linear: bad gemm variant ", design);
  }
}

// Auto choices; benchmarked in docs/rdna2/inkling2-linear-kernels.md.
struct GemvPick {
  int cfg;
  int bsel;  // blocks-per-CU selector, 0 = the cfg's own
};

// Tiny K (the controller gates): many short groups, so whole rows per wave and
// the most resident blocks. K < 2048 (o_proj at TP 2 / 4): the K split costs a
// sync per group for a few KB of weights, so one wave per row. Otherwise M <= 8
// splits K over 4 waves (small ops keep every CU busy). For MT = 16 the
// activations take two LDS chunks per group: whole rows per wave with many
// resident blocks hide the restaging; few rows keep the K split.
GemvPick pick_gemv(int mt, int rows, int K) {
  if (K <= 256) return {1, 4};
  if (K < 2048) return {0, 0};
  if (mt <= 8 || rows < 512) return {4, 0};
  return {1, 4};
}

// rows: weight rows of all segments. Up to 4096 rows (o_proj) the 128-wide
// tiles leave half the CUs idle for M < 1024, so the 128 x 64 tile wins.
int pick_gemm_design(int M, int rows) {
  if (M <= 32) return 4;
  if (M <= 64) return 3;
  if (M <= 192) return 2;
  if (M < 1024) return rows <= 4096 ? 2 : 0;
  return 1;
}

struct Prepared {
  Params pr;
};

void check_x(const torch::Tensor& x, int64_t M, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "inkling2_linear: activations must be a 2-D fp16 CUDA tensor");
  TORCH_CHECK(x.size(0) == M && x.size(1) == K,
              "inkling2_linear: all activations must be [", M, ", ", K, "]");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.stride(0) >= K,
              "inkling2_linear: activation rows need unit column stride and a "
              "row stride that is a multiple of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "inkling2_linear: activations must be 16-byte aligned");
}

template <typename WT>
Prepared prepare(const std::vector<torch::Tensor>& xs, const torch::Tensor& w,
                 const std::vector<torch::Tensor>& outs,
                 const std::vector<int64_t>& seg_x,
                 const std::vector<int64_t>& seg_n,
                 const std::vector<int64_t>& seg_act,
                 const std::vector<double>& seg_p,
                 const std::vector<int64_t>& seg_valid, bool no_out = false) {
  constexpr bool kF32 = std::is_same_v<WT, float>;
  TORCH_CHECK(!xs.empty(), "inkling2_linear: no activations");
  const int64_t M = xs[0].size(0), K = xs[0].size(1);
  TORCH_CHECK(M >= 1 && K >= 8 && K % 8 == 0,
              "inkling2_linear: need M >= 1 and K a multiple of 8");
  for (const auto& x : xs) check_x(x, M, K);
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.size(1) == K &&
                  w.is_contiguous() &&
                  w.scalar_type() == (kF32 ? at::kFloat : at::kHalf),
              "inkling2_linear: weights must be a contiguous [N, K] ",
              kF32 ? "fp32" : "fp16", " CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "inkling2_linear: weights must be 16-byte aligned");
  const size_t nseg = seg_n.size();
  TORCH_CHECK(nseg >= 1 && nseg <= kMaxSeg &&
                  (no_out ? outs.empty() : outs.size() == nseg) &&
                  seg_x.size() == nseg,
              "inkling2_linear: 1..", kMaxSeg,
              " segments, one output and one activation index each");
  TORCH_CHECK(seg_act.empty() || seg_act.size() == nseg,
              "inkling2_linear: seg_act must be empty or one per segment");
  TORCH_CHECK(seg_p.empty() || seg_p.size() == nseg,
              "inkling2_linear: seg_p must be empty or one per segment");
  TORCH_CHECK(seg_valid.empty() || seg_valid.size() == nseg,
              "inkling2_linear: seg_valid must be empty or one per segment");

  Prepared out{};
  Params& pr = out.pr;
  pr.M = static_cast<int>(M);
  pr.K = static_cast<int>(K);
  pr.write_out = 1;
  int64_t row_off = 0;
  int ns = 0;
  for (size_t i = 0; i < nseg; ++i) {
    const int64_t n = seg_n[i];
    TORCH_CHECK(n >= 0, "inkling2_linear: negative segment size");
    const int act = seg_act.empty() ? 0 : static_cast<int>(seg_act[i]);
    TORCH_CHECK(act >= 0 && act <= kActSoftcap, "inkling2_linear: bad act ",
                act);
    if (n == 0) continue;
    TORCH_CHECK(seg_x[i] >= 0 && seg_x[i] < static_cast<int64_t>(xs.size()),
                "inkling2_linear: bad activation index");
    Seg& s = pr.seg[ns++];
    s.x = xs[seg_x[i]].data_ptr();
    s.ldx = static_cast<int>(xs[seg_x[i]].stride(0));
    TORCH_CHECK(n * K < (int64_t{1} << 30),
                "inkling2_linear: segment too large (n * K >= 2^30)");
    if (no_out) {
      s.out = nullptr;
      s.ldo = 0;
      s.out_f32 = 1;
    } else {
      const auto& o = outs[i];
      TORCH_CHECK(
          o.is_cuda() && o.dim() == 2 && o.size(0) == M && o.size(1) == n,
          "inkling2_linear: output ", i, " must be [", M, ", ", n, "]");
      TORCH_CHECK(o.stride(1) == 1 && o.stride(0) >= n,
                  "inkling2_linear: output rows need unit column stride");
      const bool f32 = o.scalar_type() == at::kFloat;
      TORCH_CHECK(f32 || (!kF32 && o.scalar_type() == at::kHalf),
                  "inkling2_linear: output dtype ",
                  kF32 ? "must be fp32" : "must be fp16 or fp32");
      TORCH_CHECK(
          reinterpret_cast<uintptr_t>(o.data_ptr()) % (f32 ? 4 : 2) == 0,
          "inkling2_linear: misaligned output");
      s.out = o.data_ptr();
      s.ldo = static_cast<int>(o.stride(0));
      s.out_f32 = f32 ? 1 : 0;
    }
    s.w = reinterpret_cast<const char*>(w.data_ptr()) +
          static_cast<size_t>(row_off) * K * sizeof(WT);
    const int64_t nv = seg_valid.empty() ? n : seg_valid[i];
    TORCH_CHECK(nv >= 0 && nv <= n, "inkling2_linear: bad seg_valid");
    s.n = static_cast<int>(n);
    s.n_valid = static_cast<int>(nv);
    s.act = act;
    s.p = seg_p.empty() ? 1.0f : static_cast<float>(seg_p[i]);
    row_off += n;
  }
  TORCH_CHECK(ns >= 1, "inkling2_linear: all segments are empty");
  TORCH_CHECK(row_off <= w.size(0), "inkling2_linear: segments need ", row_off,
              " weight rows, got ", w.size(0));
  pr.nseg = ns;
  return out;
}

// GEMV launch. nparts > 0: exactly nparts blocks (argmax partials).
template <typename WT, bool ARG>
void run_gemv(Params& pr, int variant, int nparts, hipStream_t stream) {
  const int mt = mt_for(pr.M);
  int rows = 0;
  for (int i = 0; i < pr.nseg; ++i) rows += pr.seg[i].n;
  const GemvPick pk = variant >= 0 ? GemvPick{variant % 10, variant / 10}
                                   : pick_gemv(mt, rows, pr.K);
  const int cfg = pk.cfg;
  TORCH_CHECK(cfg < kNumGemvCfgs && pk.bsel < kNumBpcSel,
              "inkling2_linear: bad gemv variant ", variant);
  const GemvCfg c = kGemvCfgs[cfg];
  const int bpc = pk.bsel == 0 ? c.bpc : kBpcSel[pk.bsel];
  const int rb = (kNumWaves / c.KS) * c.R;
  const int rm = c.R * mt;
  int g = 0;
  for (int i = 0; i < pr.nseg; ++i) {
    pr.seg[i].g_begin = g;
    g += ceil_div(pr.seg[i].n, rb);
  }
  pr.groups = g;
  const int k8 = pr.K / 8;
  const int kc8_max = (kLdsBudget / 16) / mt;
  pr.nc = ceil_div(k8, kc8_max);
  pr.kc8 = ceil_div(k8, pr.nc);
  const int cap = ARG ? nparts : bpc * num_mp();
  pr.gpb = ceil_div(g, std::min(g, cap));
  const int grid = ARG ? nparts : ceil_div(g, pr.gpb);
  size_t lds = static_cast<size_t>(mt) * pr.kc8 * 16;
  if (c.KS > 1) lds += static_cast<size_t>(kNumWaves) * rm * 4;
  if (ARG) lds += static_cast<size_t>(kNumWaves) * rm * 8;
  launch_gemv_mt<WT, ARG>(mt, cfg, pr, grid, lds, stream);
}

void run_gemm(Params& pr, int variant, hipStream_t stream) {
  int rows = 0;
  for (int i = 0; i < pr.nseg; ++i) rows += pr.seg[i].n;
  const int design =
      variant >= 0 ? variant - kGemmBase : pick_gemm_design(pr.M, rows);
  TORCH_CHECK(design >= 0 && design < kNumDesigns,
              "inkling2_linear: bad gemm variant ", design + kGemmBase);
  const int bm = kDesignBM[design], bn = kDesignBN[design];
  const int bk = (design == 0 || design == 1) ? 32 : 64;
  TORCH_CHECK(
      static_cast<int64_t>(pr.M) * pr.seg[0].ldx * 2 < (int64_t{1} << 32),
      "inkling2_linear: activations too large");
  int tiles = 0;
  const int ntm = ceil_div(pr.M, bm);
  for (int i = 0; i < pr.nseg; ++i) {
    pr.seg[i].g_begin = tiles;
    tiles += ntm * ceil_div(pr.seg[i].n, bn);
  }
  launch_gemm(design, pr, tiles, pr.K % bk != 0, stream);
}

// fp16 weights: GEMV for M <= 16, GEMM above.
void run_f16(Params& pr, int64_t variant, int nparts, hipStream_t stream) {
  const bool gemv = variant >= 0 ? variant < kGemmBase : pr.M < kGemmMinM;
  if (gemv) {
    TORCH_CHECK(pr.M <= 16, "inkling2_linear: the GEMV variants take M <= 16");
    if (nparts > 0)
      run_gemv<_Float16, true>(pr, static_cast<int>(variant), nparts, stream);
    else
      run_gemv<_Float16, false>(pr, static_cast<int>(variant), 0, stream);
  } else {
    run_gemm(pr, static_cast<int>(variant), stream);
  }
}

void run_f32w(Params& pr, int64_t variant, hipStream_t stream) {
  const bool gemv = variant >= 0 ? variant < kGemmBase : pr.M < kGemmMinM;
  if (gemv) {
    TORCH_CHECK(pr.M <= 16, "inkling2_linear: the GEMV variants take M <= 16");
    run_gemv<float, false>(pr, static_cast<int>(variant), 0, stream);
  } else {
    constexpr int BM = 64, BN = 128;
    int tiles = 0;
    const int ntm = ceil_div(pr.M, BM);
    for (int i = 0; i < pr.nseg; ++i) {
      pr.seg[i].g_begin = tiles;
      tiles += ntm * ceil_div(pr.seg[i].n, BN);
    }
    if (pr.K % 16 != 0)
      gemm_f32w_kernel<true><<<tiles, 256, 0, stream>>>(pr);
    else
      gemm_f32w_kernel<false><<<tiles, 256, 0, stream>>>(pr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace
}  // namespace inkling2_linear

using namespace inkling2_linear;

void inkling2_linear_grouped(
    const std::vector<torch::Tensor>& xs, torch::Tensor w,
    const std::vector<torch::Tensor>& outs, const std::vector<int64_t>& seg_x,
    const std::vector<int64_t>& seg_n, const std::vector<int64_t>& seg_act,
    const std::vector<double>& seg_p, const std::vector<int64_t>& seg_valid,
    int64_t variant) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  Prepared pp =
      prepare<_Float16>(xs, w, outs, seg_x, seg_n, seg_act, seg_p, seg_valid);
  run_f16(pp.pr, variant, 0, stream);
}

void inkling2_linear_f32w(const std::vector<torch::Tensor>& xs, torch::Tensor w,
                          const std::vector<torch::Tensor>& outs,
                          const std::vector<int64_t>& seg_x,
                          const std::vector<int64_t>& seg_n,
                          const std::vector<int64_t>& seg_act,
                          int64_t variant) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  Prepared pp = prepare<float>(xs, w, outs, seg_x, seg_n, seg_act, {}, {});
  run_f32w(pp.pr, variant, stream);
}

void inkling2_linear_argmax_partial(torch::Tensor logits,
                                    torch::Tensor part_val,
                                    torch::Tensor part_idx, int64_t n,
                                    int64_t col_base) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(
      logits.is_cuda() && logits.dim() == 2 &&
          logits.scalar_type() == at::kFloat && logits.stride(1) == 1,
      "inkling2_linear_argmax_partial: fp32 [M, N] logits, unit stride");
  const int64_t M = logits.size(0);
  TORCH_CHECK(n >= 1 && n <= logits.size(1),
              "inkling2_linear_argmax_partial: bad n");
  TORCH_CHECK(
      part_val.is_cuda() && part_val.dim() == 2 && part_val.size(0) == M &&
          part_val.scalar_type() == at::kFloat && part_val.is_contiguous(),
      "inkling2_linear_argmax_partial: part_val must be fp32 [M, P]");
  const int64_t nparts = part_val.size(1);
  TORCH_CHECK(part_idx.is_cuda() && part_idx.sizes() == part_val.sizes() &&
                  part_idx.scalar_type() == at::kInt &&
                  part_idx.is_contiguous(),
              "inkling2_linear_argmax_partial: part_idx must be int32 [M, P]");
  const int chunk = ceil_div(static_cast<int>(n), static_cast<int>(nparts));
  dim3 grid(static_cast<unsigned>(nparts), static_cast<unsigned>(M));
  argmax_partial_kernel<<<grid, 256, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), static_cast<int>(n), chunk,
      static_cast<int>(nparts), static_cast<int>(col_base),
      part_val.data_ptr<float>(), part_idx.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_linear_argmax_reduce(torch::Tensor part_val,
                                   torch::Tensor part_idx, torch::Tensor out) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(part_val));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(part_val.is_cuda() && part_val.dim() == 2 &&
                  part_val.scalar_type() == at::kFloat &&
                  part_val.is_contiguous(),
              "inkling2_linear_argmax_reduce: part_val must be fp32 [M, P]");
  TORCH_CHECK(part_idx.is_cuda() && part_idx.sizes() == part_val.sizes() &&
                  part_idx.scalar_type() == at::kInt &&
                  part_idx.is_contiguous(),
              "inkling2_linear_argmax_reduce: part_idx must be int32 [M, P]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong &&
                  out.dim() == 1 && out.size(0) == part_val.size(0) &&
                  out.is_contiguous(),
              "inkling2_linear_argmax_reduce: out must be int64 [M]");
  argmax_reduce_kernel<<<static_cast<unsigned>(part_val.size(0)), 256, 0,
                         stream>>>(
      part_val.data_ptr<float>(), part_idx.data_ptr<int>(),
      static_cast<int>(part_val.size(1)), out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_linear_lm_head(torch::Tensor x, torch::Tensor w,
                             const std::optional<torch::Tensor>& logits,
                             const std::optional<torch::Tensor>& part_val,
                             const std::optional<torch::Tensor>& part_idx,
                             double softcap, int64_t n_valid, int64_t col_base,
                             int64_t variant) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(part_val.has_value() == part_idx.has_value(),
              "inkling2_linear_lm_head: part_val and part_idx go together");
  TORCH_CHECK(logits.has_value() || part_val.has_value(),
              "inkling2_linear_lm_head: nothing to write");
  const int64_t N = w.size(0);
  const int64_t M = x.size(0);
  const int64_t nv = n_valid < 0 ? N : n_valid;
  const std::vector<int64_t> seg_n{N};
  const std::vector<int64_t> seg_act{softcap > 0.0 ? kActSoftcap : kActNone};
  const std::vector<double> seg_p{softcap > 0.0 ? softcap : 1.0};
  const std::vector<int64_t> seg_valid{nv};
  const std::vector<int64_t> seg_x{0};
  const std::vector<torch::Tensor> xs{x};
  std::vector<torch::Tensor> outs;
  if (logits.has_value()) {
    TORCH_CHECK(logits->scalar_type() == at::kFloat,
                "inkling2_linear_lm_head: logits must be fp32");
    outs.push_back(*logits);
  }
  Prepared pp = prepare<_Float16>(xs, w, outs, seg_x, seg_n, seg_act, seg_p,
                                  seg_valid, !logits.has_value());
  int nparts = 0;
  if (part_val.has_value()) {
    TORCH_CHECK(
        part_val->scalar_type() == at::kFloat &&
            part_idx->scalar_type() == at::kInt && part_val->dim() == 2 &&
            part_val->size(0) == M && part_val->size(1) >= 1 &&
            part_val->is_contiguous() && part_idx->is_contiguous() &&
            part_idx->sizes() == part_val->sizes(),
        "inkling2_linear_lm_head: part_val fp32 / part_idx int32 [M, P]");
    nparts = static_cast<int>(part_val->size(1));
  }
  const bool fused = nparts > 0 && M <= 16 && variant < kGemmBase;
  pp.pr.write_out = logits.has_value() ? 1 : 0;
  if (fused) {
    pp.pr.nparts = nparts;
    pp.pr.col_base = static_cast<int>(col_base);
    pp.pr.part_val = part_val->data_ptr<float>();
    pp.pr.part_idx = part_idx->data_ptr<int>();
    run_f16(pp.pr, variant, nparts, stream);
    return;
  }
  TORCH_CHECK(logits.has_value(),
              "inkling2_linear_lm_head: partials without logits need M <= 16 "
              "and a GEMV variant");
  run_f16(pp.pr, variant, 0, stream);
  if (nparts > 0)
    inkling2_linear_argmax_partial(*logits, *part_val, *part_idx, nv, col_base);
}
