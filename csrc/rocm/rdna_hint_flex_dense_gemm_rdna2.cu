// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dense fp16 prefill GEMM for gfx1030 (wave32, v_dot2_f32_f16):
//
//   C[M, N] (fp16) = A[M, K] (fp16, row-major) x W[N, K]^T (fp16, row-major)
//
// fp32 accumulation, one fp16 rounding at the store (what rocBLAS does for
// these operands). It is the GEMM after the rdna_hint_flex dequant pass of the
// HI2 prefill path (A = transformed activation, W = dequantized scratch).
// Four designs of one kernel template, all measured against rocBLAS (heuristic
// and TunableOp-tuned) on the Qwen3.8-27B HI2 TP2 shapes; see
// docs/rdna2/hi2-prefill-gemm-2026-11-15.md and
// docs/rdna2/hi2-prefill-gemm2-2026-11-15.md. The sweep sources they come from
// are tools/rdna/rdna_hint_flex/gemm_sweep/gemm_variant.hip (the design name
// grammar of that tool is quoted next to every design below).
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
// = ds_read_b64), KU k-step unroll inside a stage.
//
// Tails: M and N may be anything (row loads clamp to the last row, stores are
// guarded); K must be a multiple of the design's BK; operands need 16-byte
// aligned base pointers and row strides.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.

#include <cstdint>
#include <cstring>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rdna_hint_flex_dense_gemm_rdna2.cuh"
#include "rdna_hint_flex_dense_gemm_table.cuh"

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
          int RAST_, int LW_, bool SWZ_, int KU_>
struct Design {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int NBUF = NBUF_, PAD = PAD_, RAST = RAST_, LW = LW_;
  static constexpr bool SWZ = SWZ_;
  static constexpr int KU = KU_;  // 0 = fully unrolled k steps
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
};

// 128x128 BK32, 8x16 per thread (128 accumulators), swizzled LDS, 8-byte
// fragments. 255 VGPR, 4 waves/SIMD, 4 blocks per WGP (sweep name
// 128x128k32t8x16n2p0r8g16b0w0s1m0u1l8z).
using D1 = Design<128, 128, 32, 8, 16, 2, 0, 8, 8, true, 1>;
// D1 with blocks grouped by 4 (…r4…z).
using D2 = Design<128, 128, 32, 8, 16, 2, 0, 4, 8, true, 1>;
// 128x64 BK64, 8x8 per thread, one stage, pad 8
// (128x64k64t8x8n1p8r4g16b0w0s1m0u1l8).
using D3 = Design<128, 64, 64, 8, 8, 1, 8, 4, 8, false, 1>;
// 256x128 BK32, 8x16 per thread, padded LDS (60 KiB), grouped by 4
// (256x128k32t8x16n2p8r4g16b0w0s1m0u1l8).
using D4 = Design<256, 128, 32, 8, 16, 2, 8, 4, 8, false, 1>;

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
        *reinterpret_cast<Chunk*>(bs + store_off(c / CPR, c % CPR)) = rb[v];
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
            acc[i][j] = dot2u(af[i][t], bf[j][t], acc[i][j]);
    };
    if constexpr (P::KU == 0) {
  #pragma unroll
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    } else {
  #pragma unroll P::KU
      for (int kk = 0; kk < BK; kk += P::KS) kstep(kk);
    }
  };

  if constexpr (NBUF == 2) {
    gload(0);
    lstore(0);
    __syncthreads();
    int kt = 0;
    for (; kt + 1 < nk; kt += 2) {
      gload(kt + 1);
      compute(0);
      lstore(1);
      __syncthreads();
      if (kt + 2 < nk) gload(kt + 2);
      compute(1);
      if (kt + 2 < nk) lstore(0);
      __syncthreads();
    }
    if (kt < nk) compute(0);
  } else {
    gload(0);
    for (int kt = 0; kt < nk; ++kt) {
      __syncthreads();
      lstore(0);
      __syncthreads();
      if (kt + 1 < nk) gload(kt + 1);
      compute(0);
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

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
namespace vllm {
namespace rdna_hint_flex {
namespace gemm {

namespace {

struct DesignInfo {
  const char* name;  // sweep grammar (tools/rdna/rdna_hint_flex/gemm_sweep)
  int bm, bn, bk;
};

constexpr DesignInfo kDesigns[] = {
    {"128x128k32t8x16n2p0r8g16b0w0s1m0u1l8z", 128, 128, 32},
    {"128x128k32t8x16n2p0r4g16b0w0s1m0u1l8z", 128, 128, 32},
    {"128x64k64t8x8n1p8r4g16b0w0s1m0u1l8", 128, 64, 64},
    {"256x128k32t8x16n2p8r4g16b0w0s1m0u1l8", 256, 128, 32},
};
constexpr int kNumDesigns = sizeof(kDesigns) / sizeof(kDesigns[0]);

bool on_gfx1030() {
  const hipDeviceProp_t* p = at::cuda::getCurrentDeviceProperties();
  return std::strncmp(p->gcnArchName, "gfx1030", 7) == 0;
}

template <class P>
void launch_design(const void* A, const void* W, void* C, int64_t M, int64_t N,
                   int64_t K, int64_t lda, int64_t ldw, int64_t ldc,
                   hipStream_t stream) {
  const int64_t ntm = (M + P::BM - 1) / P::BM, ntn = (N + P::BN - 1) / P::BN;
  gemm_kernel<P>
      <<<dim3(static_cast<unsigned>(ntm * ntn)), dim3(P::T), 0, stream>>>(
          static_cast<const char*>(A), static_cast<const char*>(W),
          static_cast<_Float16*>(C), static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K), static_cast<int>(lda), static_cast<int>(ldw),
          static_cast<int>(ldc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

}  // namespace

int pick(int64_t M, int64_t N, int64_t K) {
  for (const Slot& s : kSlots) {
    if (s.n != N || s.k != K) continue;
    if (M < s.m_lo || M > s.m_hi) return kRocblas;
    // nearest measured M (ties go to the smaller one)
    int best = s.anchors[0].design;
    int64_t dist = M > s.anchors[0].m ? M - s.anchors[0].m : s.anchors[0].m - M;
    for (int i = 1; i < s.count; ++i) {
      const int64_t d =
          M > s.anchors[i].m ? M - s.anchors[i].m : s.anchors[i].m - M;
      if (d < dist) {
        dist = d;
        best = s.anchors[i].design;
      }
    }
    return best;
  }
  return kRocblas;
}

bool launch(int design, const void* A, const void* W, void* C, int64_t M,
            int64_t N, int64_t K, int64_t lda, int64_t ldw, int64_t ldc,
            hipStream_t stream) {
  if (design < 0 || design >= kNumDesigns) return false;
  if (M == 0 || N == 0) return true;
  const DesignInfo& d = kDesigns[design];
  // 32-bit byte offsets in the kernel; 16-byte global chunks.
  constexpr int64_t kMaxElems = int64_t{1} << 30;
  if (M < 0 || N < 0 || K <= 0 || K % d.bk != 0 || lda % 8 != 0 ||
      ldw % 8 != 0 || lda < K || ldw < K || ldc < N || M * lda >= kMaxElems ||
      N * ldw >= kMaxElems || ldc >= (int64_t{1} << 31) || !aligned16(A) ||
      !aligned16(W) || !on_gfx1030())
    return false;
  switch (design) {
    case 0:
      launch_design<D1>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 1:
      launch_design<D2>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    case 2:
      launch_design<D3>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
    default:
      launch_design<D4>(A, W, C, M, N, K, lda, ldw, ldc, stream);
      break;
  }
  return true;
}

}  // namespace gemm
}  // namespace rdna_hint_flex
}  // namespace vllm

namespace {
namespace hg = vllm::rdna_hint_flex::gemm;

void check_dense_operands(const torch::Tensor& x, const torch::Tensor& w,
                          const torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "rdna_hint_flex_dense_gemm: tensors must be CUDA/HIP tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                  w.scalar_type() == torch::kHalf &&
                  out.scalar_type() == torch::kHalf,
              "rdna_hint_flex_dense_gemm: x, w and out must be fp16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2,
              "rdna_hint_flex_dense_gemm: x [M, K], w [N, K], out [M, N]");
  TORCH_CHECK(x.size(1) == w.size(1) && out.size(0) == x.size(0) &&
                  out.size(1) == w.size(0),
              "rdna_hint_flex_dense_gemm: shape mismatch x ", x.sizes(), " w ",
              w.sizes(), " out ", out.sizes());
  TORCH_CHECK(
      x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
      "rdna_hint_flex_dense_gemm: row-major operands (unit column stride)");
  TORCH_CHECK(x.device() == w.device() && x.device() == out.device(),
              "rdna_hint_flex_dense_gemm: operands on different devices");
}

}  // namespace

// out = x . w^T. design < 0: the dispatch table picks (rocBLAS when it says so
// or the shape is outside the validated set); design >= 0: that design, an
// error if it cannot run the problem (tests and benchmarks).
void rdna_hint_flex_dense_gemm_rdna2(torch::Tensor x, torch::Tensor w,
                                     torch::Tensor out, int64_t design) {
  check_dense_operands(x, w, out);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t M = x.size(0), N = w.size(0), K = x.size(1);
  const bool forced = design >= 0;
  const int d = forced ? static_cast<int>(design) : hg::pick(M, N, K);
  if (d >= 0 && hg::launch(d, x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N,
                           K, x.stride(0), w.stride(0), out.stride(0), stream))
    return;
  TORCH_CHECK(!forced, "rdna_hint_flex_dense_gemm: design ", design,
              " cannot run M=", M, " N=", N, " K=", K,
              " (needs gfx1030, K % BK == 0, 16-byte aligned operands and row "
              "strides)");
  at::mm_out(out, x, w.t());
}

// The design the table picks for (M, N, K); -1 = rocBLAS.
int64_t rdna_hint_flex_dense_gemm_pick_rdna2(int64_t M, int64_t N, int64_t K) {
  return hg::pick(M, N, K);
}
