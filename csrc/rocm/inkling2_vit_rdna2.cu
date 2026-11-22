// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Inkling-2-Mini vision encoder kernels for gfx1030 (RDNA2, wave32,
// v_dot2_f32_f16 as the matrix primitive).  Docs:
// docs/rdna2/inkling2-vision.md.
//
// The encoder is 23 pre-LN ViT blocks of width 1024 (16 heads of 64) on 20x20
// pixel patches, a 2x2 patch merger and a two-layer projector.  Its tokens are
// packed over all images of a request batch: per image one CLS token followed
// by the native patches in cell-major order (the four patches of every 40x40
// cell, top-left, top-right, bottom-left, bottom-right, cells row-major).  All
// ops take the same per-image table
//
//   meta int32 [n_img, 4] = {rows, cols, tok0, patch0}
//
// rows x cols cells, tok0 = first token (the CLS) of the image in the packed
// token axis, patch0 = first patch row of the image in the patch matrix; an
// image has 4 rows cols + 1 tokens and 4 rows cols patches.
//
// Numerics: the residual stream is fp32; every GEMM operand is fp16 with fp32
// accumulation (v_dot2); LayerNorm / softmax statistics / GELU are fp32.
//
//   inkling2_vit_gather_patches  raw cells -> normalised fp16 patch rows
//   inkling2_vit_gemm            C = A W^T with fused epilogue
//   inkling2_vit_gemm_residual   res += gamma * (A W^T + bias)  (fp32 residual)
//   inkling2_vit_embed           patch GEMM + bilinear position embedding + CLS
//                                -> LayerNorm (the ln_pre) -> fp32 hidden
//   inkling2_vit_layernorm       fp32 rows -> LayerNorm -> fp16 rows
//   inkling2_vit_rope_tables     interleaved axial 2-D RoPE cos / sin per token
//   inkling2_vit_rope_qk         rotate q and k of the fused qkv rows in place
//   inkling2_vit_attention       bidirectional flash attention, per image
//   inkling2_vit_pool            non-CLS tokens -> fp16 rows (merger input)
//
// The GEMM kernel is the "D1" design of rdna_hint_flex_dense_gemm_kernel.cuh
// (128x128 tile, BK 32, 8x16 accumulators per thread, XOR-swizzled LDS,
// B-fragment dword rotation, buffer-descriptor loads) with the epilogues of
// this encoder fused in.  Device code is compiled for gfx1030 only: other
// device passes get empty kernels, so their code objects do not change.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.

#include <cstdint>
#include <cstring>
#include <optional>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace inkling2_vit {

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using u32x2 = unsigned int __attribute__((ext_vector_type(2)));
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));

constexpr int kHeadDim = 64;
constexpr int kPatchPixels = 20;                          // native patch
constexpr int kPatchK = kPatchPixels * kPatchPixels * 3;  // 1200
constexpr int kCellPixels = 2 * kPatchPixels;             // 40
// 40 x 40 x 3 = 4800; like the GEMM and attention tile constants below it is
// read by device code only (the host pass of every target sees empty kernels)
[[maybe_unused]] constexpr int kCellElems = kCellPixels * kCellPixels * 3;
constexpr int kMaxHidden = 2048;  // LayerNorm / embed rows held in registers

// One dword = two fp16 values: fp32 a.x*b.x + a.y*b.y + c (v_dot2c_f32_f16).
__device__ __forceinline__ float dot2u(unsigned a, unsigned b, float c) {
#if defined(__HIP__RDNA2__)
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a),
                                __builtin_bit_cast(f16x2, b), c, false);
#else
  return c;
#endif
}

// (a, b) -> one dword of two fp16 values, round to nearest even.
__device__ __forceinline__ unsigned pack_h2(float a, float b) {
  const f16x2 v = {static_cast<_Float16>(a), static_cast<_Float16>(b)};
  return __builtin_bit_cast(unsigned, v);
}

__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// Image of packed token t: the last image whose first token is <= t.
__device__ __forceinline__ int find_image(const int* __restrict__ meta,
                                          int n_img, int t) {
  int lo = 0, hi = n_img - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (meta[mid * 4 + 2] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// (y, x) of native patch jp (0-based, cell-major) of an image with `cols`
// cells per row.
__device__ __forceinline__ void patch_yx(int jp, int cols, int& y, int& x) {
  const int cell = jp >> 2, sub = jp & 3;
  y = 2 * (cell / cols) + (sub >> 1);
  x = 2 * (cell % cols) + (sub & 1);
}

// ---------------------------------------------------------------------------
// gather_patches
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_f32(_Float16 v) {
  return static_cast<float>(v);
}
__device__ __forceinline__ float to_f32(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

// Round to bf16 precision (nearest even); the value stays an fp32.  Finite
// inputs only.
__device__ __forceinline__ float round_bf16(float x) {
  unsigned u = __builtin_bit_cast(unsigned, x);
  u += 0x7fffu + ((u >> 16) & 1u);
  return __builtin_bit_cast(float, u & 0xffff0000u);
}

// CLIP mean / std per channel.  The image processor's std spelling (0.2613026,
// 0.2757771) and the encoder's (0.26130258, 0.27577711) round to the same
// fp32 values.
__device__ __forceinline__ float clip_mean(int ch) {
  return ch == 0 ? 0.48145466f : (ch == 1 ? 0.4578275f : 0.40821073f);
}
__device__ __forceinline__ float clip_std(int ch) {
  return ch == 0 ? 0.26862954f : (ch == 1 ? 0.26130258f : 0.27577711f);
}

// Pixel value of the patch GEMM for raw 0..255 (or the -1 padding) channel ch.
//   kPixelExact: 2 raw / 255 - 1, the transformers image processor
//   kPixelClipBf16: 2 (bf16((raw / 255 - mean) / std) std + mean) - 1 rounded
//     to bf16, each operation in fp32 without fusion: the CLIP-normalised bf16
//     value of the model's own pipeline, mapped back to [-1, 1] (the
//     reference of the NVIDIA vLLM implementation)
constexpr int kPixelExact = 0;
constexpr int kPixelClipBf16 = 1;

template <int MODE>
__device__ __forceinline__ _Float16 pixel_value(float raw, int ch) {
  if constexpr (MODE == kPixelExact) {
    return static_cast<_Float16>(
        __fdiv_rn(__fsub_rn(__fmul_rn(2.0f, raw), 255.0f), 255.0f));
  } else {
    const float mean = clip_mean(ch), sd = clip_std(ch);
    const float clip =
        round_bf16(__fdiv_rn(__fsub_rn(__fdiv_rn(raw, 255.0f), mean), sd));
    const float pe =
        __fsub_rn(__fmul_rn(2.0f, __fadd_rn(__fmul_rn(clip, sd), mean)), 1.0f);
    return static_cast<_Float16>(round_bf16(pe));
  }
}

// out[4 cell + 2 iy + ix, k] (k = (r 20 + c) 3 + ch for pixel (20 iy + r,
// 20 ix + c) of the cell, k >= 1200 zero) = pixel_value(raw).  Four
// consecutive k stay inside one 60-element pixel row run.
template <typename Tin, int MODE>
__global__ void __launch_bounds__(256)
    gather_patches_kernel(const Tin* __restrict__ raw,
                          _Float16* __restrict__ out, int64_t n_vec, int kpad) {
#if defined(__HIP__RDNA2__)
  const int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= n_vec) return;
  const int vpr = kpad >> 2;
  const int64_t row = v / vpr;
  const int k = static_cast<int>(v - row * vpr) * 4;
  f16x2 o01 = {0, 0}, o23 = {0, 0};
  if (k < kPatchK) {
    const int64_t cell = row >> 2;
    const int sub = static_cast<int>(row & 3);
    const int r = k / 60, kk = k - r * 60;
    const Tin* src = raw + cell * kCellElems +
                     ((sub >> 1) * kPatchPixels + r) * (kCellPixels * 3) +
                     (sub & 1) * (kPatchPixels * 3) + kk;
    const int ch = k % 3;  // 60 and 1200 are multiples of 3
    o01 = {pixel_value<MODE>(to_f32(src[0]), ch),
           pixel_value<MODE>(to_f32(src[1]), (ch + 1) % 3)};
    o23 = {pixel_value<MODE>(to_f32(src[2]), (ch + 2) % 3),
           pixel_value<MODE>(to_f32(src[3]), ch)};
  }
  const u32x2 packed = {__builtin_bit_cast(unsigned, o01),
                        __builtin_bit_cast(unsigned, o23)};
  *reinterpret_cast<u32x2*>(out + row * kpad + k) = packed;
#endif
}

// ---------------------------------------------------------------------------
// GEMM: C[M, N] = A[M, K] W[N, K]^T with fused epilogues
// ---------------------------------------------------------------------------

namespace gemm {

enum Epilogue : int {
  kF16 = 0,          // fp16 C = acc
  kF16Bias = 1,      // fp16 C = acc + bias
  kF16BiasGelu = 2,  // fp16 C = gelu(acc + bias)
  kF16Gelu = 3,      // fp16 C = gelu(acc)
  kF32 = 4,          // fp32 C = acc
  kResidual = 5,     // fp32 res += gamma (acc + bias)
};

constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 16;
constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;  // 8, 16, 128
constexpr int LD = BK;                                  // halves per LDS row
constexpr int CH = 8, CPR = BK / CH;                    // chunk halves, per row
constexpr int A_PER_T = BM * CPR / T, B_PER_T = BN * CPR / T;
[[maybe_unused]] constexpr int A_LDS = BM * LD;  // halves
[[maybe_unused]] constexpr int STAGE = (BM + BN) * LD;
[[maybe_unused]] constexpr int LDS_BYTES = 2 * STAGE * 2;
[[maybe_unused]] constexpr int SRP = 8 / CPR;  // rows per XOR-swizzle step
[[maybe_unused]] constexpr int RAST = 8;       // M tiles per N-tile walk
[[maybe_unused]] constexpr int KS = 4;         // halves per k step
[[maybe_unused]] constexpr int ND = 2;         // dwords per fragment
static_assert(A_PER_T * T == BM * CPR && B_PER_T * T == BN * CPR, "chunks");

template <int EPI>
__global__ void __launch_bounds__(T)
    gemm_kernel(const char* __restrict__ A, const char* __restrict__ W,
                void* __restrict__ Cout, const float* __restrict__ bias,
                const float* __restrict__ gamma, float* __restrict__ res, int M,
                int N, int K, int lda, int ldw, int ldc) {
#if defined(__HIP__RDNA2__)
  using Chunk = u32x4;
  using Frag = u32x2;
  __shared__ __attribute__((aligned(16))) char lds[LDS_BYTES];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  // XOR swizzle of the 16-byte chunk index inside an LDS row: the rows of one
  // read phase land in different bank groups without padding.
  const int swa = (ty / SRP) & (CPR - 1);
  const int swb = (tx / SRP) & (CPR - 1);
  auto store_off = [](int r, int kcc) {
    return r * LD * 2 + ((kcc ^ ((r / SRP) & (CPR - 1))) << 4);
  };
  auto frag_off = [](int kk, int sw) {
    return (((kk >> 3) ^ sw) << 4) + ((kk & 7) << 1);
  };

  // block -> tile: groups of RAST M tiles walk the N tiles
  const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
  const int bid = blockIdx.x;
  const int per_group = RAST * ntn;
  const int group = bid / per_group;
  const int first_m = group * RAST;
  const int gsz = min(ntm - first_m, RAST);
  const int tm = first_m + (bid % per_group) % gsz;
  const int tn = (bid % per_group) / gsz;
  const int m0 = tm * BM, n0 = tn * BN;
  const int nk = K / BK;

  // a wave with no valid row or column skips its dot2 work
  const int w0 = __builtin_amdgcn_readfirstlane(tid) & ~31;
  const bool wave_live = (m0 + w0 / TX < M) && (n0 < N);

  uint32_t aoff[A_PER_T], boff[B_PER_T];
  #pragma unroll
  for (int v = 0; v < A_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = c / CPR;
    const int kc = (c % CPR) * CH;
    const int grow = min(m0 + row, M - 1);
    aoff[v] = static_cast<uint32_t>((grow * lda + kc) * 2);
  }
  #pragma unroll
  for (int v = 0; v < B_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = c / CPR;
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
      ra[v] = __builtin_amdgcn_raw_buffer_load_b128(rsa, aoff[v], kb, 0);
  #pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      rb[v] = __builtin_amdgcn_raw_buffer_load_b128(rsw, boff[v], kb, 0);
  };
  auto lstore = [&](int buf) {
    char* as = lds + buf * STAGE * 2;
    char* bs = as + A_LDS * 2;
  #pragma unroll
    for (int v = 0; v < A_PER_T; ++v) {
      const int c = tid + v * T;
      *reinterpret_cast<Chunk*>(as + store_off(c / CPR, c % CPR)) = ra[v];
    }
  #pragma unroll
    for (int v = 0; v < B_PER_T; ++v) {
      const int c = tid + v * T;
      // the two dwords of every 8-byte weight fragment are stored swapped, so
      // the two sources of a v_dot2 sit in different VGPR banks
      const Chunk q = rb[v];
      *reinterpret_cast<Chunk*>(bs + store_off(c / CPR, c % CPR)) =
          Chunk{q.y, q.x, q.w, q.z};
    }
  };

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  auto compute = [&](int buf) {
    const char* as = lds + buf * STAGE * 2;
    const char* bs = as + A_LDS * 2;
  #pragma unroll 1
    for (int kk = 0; kk < BK; kk += KS) {
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
          for (int t = 0; t < ND; ++t)
            acc[i][j] = dot2u(af[i][t], bf[j][(t + 1) % ND], acc[i][j]);
    }
  };
  auto run = [&](int buf) {
    if (!wave_live) return;
    compute(buf);
  };

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

  // ---- epilogue
  constexpr bool kBias =
      EPI == kF16Bias || EPI == kF16BiasGelu || EPI == kResidual;
  float bj[TN], gj[TN];
  #pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int col = n0 + tx + TX * j;
    bj[j] = (kBias && col < N) ? bias[col] : 0.0f;
    gj[j] = (EPI == kResidual && col < N) ? gamma[col] : 0.0f;
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + TY * i;
    if (row < M) {
  #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + tx + TX * j;
        if (col < N) {
          float v = acc[i][j];
          const size_t at = static_cast<size_t>(row) * ldc + col;
          if constexpr (EPI == kF16) {
            static_cast<_Float16*>(Cout)[at] = static_cast<_Float16>(v);
          } else if constexpr (EPI == kF16Bias) {
            static_cast<_Float16*>(Cout)[at] = static_cast<_Float16>(v + bj[j]);
          } else if constexpr (EPI == kF16BiasGelu) {
            static_cast<_Float16*>(Cout)[at] =
                static_cast<_Float16>(gelu_erf(v + bj[j]));
          } else if constexpr (EPI == kF16Gelu) {
            static_cast<_Float16*>(Cout)[at] =
                static_cast<_Float16>(gelu_erf(v));
          } else if constexpr (EPI == kF32) {
            static_cast<float*>(Cout)[at] = v;
          } else {
            res[at] += gj[j] * (v + bj[j]);
          }
        }
      }
    }
  }
#endif
}

}  // namespace gemm

// ---------------------------------------------------------------------------
// LayerNorm (fp32 rows -> fp16 rows), one wave per row
// ---------------------------------------------------------------------------

// Mean / biased variance over D (a multiple of 128) in two register passes.
template <int NV>
__device__ __forceinline__ void layer_norm_row(float4 (&x)[NV], int D,
                                               const float* __restrict__ w,
                                               const float* __restrict__ b,
                                               float eps, int lane,
                                               float4 (&y)[NV]) {
  const int nv = D >> 7;
  float s = 0.0f;
#pragma unroll
  for (int v = 0; v < NV; ++v)
    if (v < nv) s += (x[v].x + x[v].y) + (x[v].z + x[v].w);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o);
  const float mean = s / static_cast<float>(D);
  float q = 0.0f;
#pragma unroll
  for (int v = 0; v < NV; ++v)
    if (v < nv) {
      const float a = x[v].x - mean, c = x[v].y - mean, d = x[v].z - mean,
                  e = x[v].w - mean;
      q += (a * a + c * c) + (d * d + e * e);
    }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q += __shfl_xor(q, o);
  const float rstd = 1.0f / sqrtf(q / static_cast<float>(D) + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v)
    if (v < nv) {
      const int c = lane * 4 + v * 128;
      const float4 wv = *reinterpret_cast<const float4*>(w + c);
      const float4 bv = *reinterpret_cast<const float4*>(b + c);
      y[v].x = (x[v].x - mean) * rstd * wv.x + bv.x;
      y[v].y = (x[v].y - mean) * rstd * wv.y + bv.y;
      y[v].z = (x[v].z - mean) * rstd * wv.z + bv.z;
      y[v].w = (x[v].w - mean) * rstd * wv.w + bv.w;
    }
}

[[maybe_unused]] constexpr int kNV = kMaxHidden / 128;

__global__ void __launch_bounds__(256)
    layernorm_kernel(const float* __restrict__ h, const float* __restrict__ w,
                     const float* __restrict__ b, _Float16* __restrict__ out,
                     int rows, int D, float eps) {
#if defined(__HIP__RDNA2__)
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int nv = D >> 7;
  float4 x[kNV], y[kNV];
  #pragma unroll
  for (int v = 0; v < kNV; ++v)
    if (v < nv)
      x[v] = *reinterpret_cast<const float4*>(h + static_cast<size_t>(row) * D +
                                              lane * 4 + v * 128);
  layer_norm_row<kNV>(x, D, w, b, eps, lane, y);
  #pragma unroll
  for (int v = 0; v < kNV; ++v)
    if (v < nv) {
      const u32x2 p = {pack_h2(y[v].x, y[v].y), pack_h2(y[v].z, y[v].w)};
      *reinterpret_cast<u32x2*>(out + static_cast<size_t>(row) * D + lane * 4 +
                                v * 128) = p;
    }
#endif
}

// ---------------------------------------------------------------------------
// embed: patch GEMM output + bilinear position embedding (or CLS) -> ln_pre
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
    embed_kernel(const float* __restrict__ emb, const float* __restrict__ pos,
                 const float* __restrict__ cls, const float* __restrict__ ln_w,
                 const float* __restrict__ ln_b, const int* __restrict__ meta,
                 int n_img, float* __restrict__ h, int T_tokens, int D, int G,
                 float eps) {
#if defined(__HIP__RDNA2__)
  const int t = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= T_tokens) return;
  const int nv = D >> 7;
  const int img = find_image(meta, n_img, t);
  const int rows = meta[img * 4], cols = meta[img * 4 + 1];
  const int tok0 = meta[img * 4 + 2], patch0 = meta[img * 4 + 3];
  const int j = t - tok0;
  float4 x[kNV], y[kNV];
  if (j == 0) {
  #pragma unroll
    for (int v = 0; v < kNV; ++v)
      if (v < nv)
        x[v] = *reinterpret_cast<const float4*>(cls + lane * 4 + v * 128);
  } else {
    const int jp = j - 1;
    int py, px;
    patch_yx(jp, cols, py, px);
    // F.interpolate(mode="bilinear", align_corners=False) of the G x G grid to
    // (2 rows) x (2 cols); the source coordinate is clamped at 0
    const float sy =
        fmaxf((py + 0.5f) * (static_cast<float>(G) / (2 * rows)) - 0.5f, 0.0f);
    const float sx =
        fmaxf((px + 0.5f) * (static_cast<float>(G) / (2 * cols)) - 0.5f, 0.0f);
    const int y0 = min(static_cast<int>(sy), G - 1);
    const int x0 = min(static_cast<int>(sx), G - 1);
    const int y1 = min(y0 + 1, G - 1), x1 = min(x0 + 1, G - 1);
    const float ly = sy - static_cast<float>(y0),
                lx = sx - static_cast<float>(x0);
    const float hy = 1.0f - ly, hx = 1.0f - lx;
    const float* p00 = pos + static_cast<size_t>(y0 * G + x0) * D;
    const float* p01 = pos + static_cast<size_t>(y0 * G + x1) * D;
    const float* p10 = pos + static_cast<size_t>(y1 * G + x0) * D;
    const float* p11 = pos + static_cast<size_t>(y1 * G + x1) * D;
    const float* e = emb + static_cast<size_t>(patch0 + jp) * D;
  #pragma unroll
    for (int v = 0; v < kNV; ++v)
      if (v < nv) {
        const int c = lane * 4 + v * 128;
        const float4 a = *reinterpret_cast<const float4*>(p00 + c);
        const float4 b = *reinterpret_cast<const float4*>(p01 + c);
        const float4 cc = *reinterpret_cast<const float4*>(p10 + c);
        const float4 d = *reinterpret_cast<const float4*>(p11 + c);
        const float4 ev = *reinterpret_cast<const float4*>(e + c);
        x[v].x =
            ev.x + (hy * (hx * a.x + lx * b.x) + ly * (hx * cc.x + lx * d.x));
        x[v].y =
            ev.y + (hy * (hx * a.y + lx * b.y) + ly * (hx * cc.y + lx * d.y));
        x[v].z =
            ev.z + (hy * (hx * a.z + lx * b.z) + ly * (hx * cc.z + lx * d.z));
        x[v].w =
            ev.w + (hy * (hx * a.w + lx * b.w) + ly * (hx * cc.w + lx * d.w));
      }
  }
  layer_norm_row<kNV>(x, D, ln_w, ln_b, eps, lane, y);
  #pragma unroll
  for (int v = 0; v < kNV; ++v)
    if (v < nv)
      *reinterpret_cast<float4*>(h + static_cast<size_t>(t) * D + lane * 4 +
                                 v * 128) = y[v];
#endif
}

// ---------------------------------------------------------------------------
// RoPE: interleaved pairs, axial: pair p < hd/4 rotates by x inv[p], the others
// by y inv[p - hd/4]; the CLS token has position (0, 0), patch (py, px) has
// position (py + 1, px + 1).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
    rope_tables_kernel(const int* __restrict__ meta, int n_img,
                       const float* __restrict__ inv_freq,
                       float2* __restrict__ cs, int T_tokens) {
#if defined(__HIP__RDNA2__)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T_tokens * 32) return;
  const int t = idx >> 5, p = idx & 31;
  const int img = find_image(meta, n_img, t);
  const int cols = meta[img * 4 + 1];
  const int j = t - meta[img * 4 + 2];
  float yv = 0.0f, xv = 0.0f;
  if (j > 0) {
    int py, px;
    patch_yx(j - 1, cols, py, px);
    yv = static_cast<float>(py + 1);
    xv = static_cast<float>(px + 1);
  }
  const float ang = (p < 16 ? xv : yv) * inv_freq[p & 15];
  float s, c;
  sincosf(ang, &s, &c);
  cs[idx] = make_float2(c, s);
#endif
}

// qkv [T, 3 D] fp16: rotate q (columns [0, D)) and k ([D, 2 D)) in place.
__global__ void __launch_bounds__(256)
    rope_qk_kernel(_Float16* __restrict__ qkv, const float2* __restrict__ cs,
                   int T_tokens, int D) {
#if defined(__HIP__RDNA2__)
  const int half_d = D >> 1;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T_tokens) * half_d) return;
  const int t = static_cast<int>(idx / half_d);
  const int c = static_cast<int>(idx - static_cast<int64_t>(t) * half_d);
  const float2 r = cs[t * 32 + (c & 31)];
  unsigned* q =
      reinterpret_cast<unsigned*>(qkv + static_cast<size_t>(t) * 3 * D) + c;
  unsigned* k = q + half_d;
  const f16x2 qv = __builtin_bit_cast(f16x2, *q),
              kv = __builtin_bit_cast(f16x2, *k);
  const float qe = qv.x, qo = qv.y, ke = kv.x, ko = kv.y;
  *q = pack_h2(qe * r.x - qo * r.y, qo * r.x + qe * r.y);
  *k = pack_h2(ke * r.x - ko * r.y, ko * r.x + ke * r.y);
#endif
}

// ---------------------------------------------------------------------------
// pool: non-CLS rows of the fp32 hidden -> fp16 rows [P, D]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
    pool_kernel(const float* __restrict__ h, const int* __restrict__ meta,
                int n_img, _Float16* __restrict__ out, int T_tokens, int D) {
#if defined(__HIP__RDNA2__)
  const int t = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= T_tokens) return;
  const int img = find_image(meta, n_img, t);
  if (t == meta[img * 4 + 2]) return;  // CLS
  const size_t o = static_cast<size_t>(t - img - 1) * D;
  for (int c = lane * 4; c < D; c += 128) {
    const float4 v =
        *reinterpret_cast<const float4*>(h + static_cast<size_t>(t) * D + c);
    *reinterpret_cast<u32x2*>(out + o + c) =
        u32x2{pack_h2(v.x, v.y), pack_h2(v.z, v.w)};
  }
#endif
}

// ---------------------------------------------------------------------------
// Bidirectional flash attention over packed images (head_dim 64)
//
// A block owns 64 query rows of one head of one image and walks that image's
// keys in tiles of 64.  Thread (ty = tid / 8, tx = tid % 8) owns the S / O
// micro-tile rows ty + 16 i (i < 4) and, per 16-wide column group u < 4, the
// two adjacent columns 2 tx + 16 u + e (e < 2): the two columns of a pair are
// then packed into one dword of P (staged in LDS) and one dword of the
// key-pair-interleaved V (V^T pairs), so both products are v_dot2 over dwords.
// All P rows of a wave are produced and consumed by that wave.
// ---------------------------------------------------------------------------

namespace attn {
[[maybe_unused]] constexpr int BQ = 64, BKV = 64, THREADS = 128;
// dword pitches of the LDS tiles: sQ rows (bank phase 8 per row), sK rows (b64
// reads of 2 tx rows), sP rows, sV rows (pairs of keys)
[[maybe_unused]] constexpr int QP = 40;
[[maybe_unused]] constexpr int KP = 34;
[[maybe_unused]] constexpr int PP = 40;
[[maybe_unused]] constexpr int VP = 64;
constexpr float kLog2e = 1.4426950408889634f;
}  // namespace attn

__global__ void __launch_bounds__(attn::THREADS)
    attention_kernel(const _Float16* __restrict__ qkv,
                     _Float16* __restrict__ out, const int* __restrict__ meta,
                     const int* __restrict__ blk_prefix, int n_img,
                     int nq_total, int D, float cs) {
#if defined(__HIP__RDNA2__)
  using namespace attn;
  __shared__ __attribute__((aligned(16))) unsigned sQ[BQ * QP];
  __shared__ __attribute__((aligned(16))) unsigned sK[BKV * KP];
  __shared__ __attribute__((aligned(16))) unsigned sV[32 * VP];
  __shared__ __attribute__((aligned(16))) unsigned sP[BQ * PP];

  const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
  const int head = blockIdx.x / nq_total;
  const int qbg = blockIdx.x - head * nq_total;
  int lo = 0, hi = n_img - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (blk_prefix[mid] <= qbg)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int img = lo;
  const int q0 = (qbg - blk_prefix[img]) * BQ;
  const int tok0 = meta[img * 4 + 2];
  const int L = 4 * meta[img * 4] * meta[img * 4 + 1] + 1;
  const int ld = 3 * D;
  const _Float16* qbase =
      qkv + static_cast<size_t>(tok0) * ld + head * kHeadDim;
  const _Float16* kbase = qbase + D;
  const _Float16* vbase = qbase + 2 * D;

  #pragma unroll
  for (int v = 0; v < 4; ++v) {
    const int c = tid + v * THREADS, row = c >> 3, cc = c & 7;
    const int src = min(q0 + row, L - 1);
    *reinterpret_cast<u32x4*>(&sQ[row * QP + cc * 4]) =
        *reinterpret_cast<const u32x4*>(qbase + static_cast<size_t>(src) * ld +
                                        cc * 8);
  }

  u32x4 kr[4], va[2], vb[2];
  auto gload = [&](int jt) {
  #pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = tid + v * THREADS, row = c >> 3, cc = c & 7;
      const int src = min(jt * BKV + row, L - 1);
      kr[v] = *reinterpret_cast<const u32x4*>(
          kbase + static_cast<size_t>(src) * ld + cc * 8);
    }
  #pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int item = tid + v * THREADS, tp = item >> 3, cc = item & 7;
      const int k0 = min(jt * BKV + 2 * tp, L - 1),
                k1 = min(jt * BKV + 2 * tp + 1, L - 1);
      va[v] = *reinterpret_cast<const u32x4*>(
          vbase + static_cast<size_t>(k0) * ld + cc * 8);
      vb[v] = *reinterpret_cast<const u32x4*>(
          vbase + static_cast<size_t>(k1) * ld + cc * 8);
    }
  };
  auto lstore = [&]() {
  #pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = tid + v * THREADS, row = c >> 3, cc = c & 7;
      const u32x4 x = kr[v];
      *reinterpret_cast<u32x2*>(&sK[row * KP + cc * 4]) = u32x2{x.x, x.y};
      *reinterpret_cast<u32x2*>(&sK[row * KP + cc * 4 + 2]) = u32x2{x.z, x.w};
    }
  #pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int item = tid + v * THREADS, tp = item >> 3, cc = item & 7;
      const u32x4 a = va[v], b = vb[v];
      // dword d = (V[2 tp][d], V[2 tp + 1][d])
      const u32x4 lo4 = {__builtin_amdgcn_perm(b.x, a.x, 0x05040100u),
                         __builtin_amdgcn_perm(b.x, a.x, 0x07060302u),
                         __builtin_amdgcn_perm(b.y, a.y, 0x05040100u),
                         __builtin_amdgcn_perm(b.y, a.y, 0x07060302u)};
      const u32x4 hi4 = {__builtin_amdgcn_perm(b.z, a.z, 0x05040100u),
                         __builtin_amdgcn_perm(b.z, a.z, 0x07060302u),
                         __builtin_amdgcn_perm(b.w, a.w, 0x05040100u),
                         __builtin_amdgcn_perm(b.w, a.w, 0x07060302u)};
      *reinterpret_cast<u32x4*>(&sV[tp * VP + cc * 8]) = lo4;
      *reinterpret_cast<u32x4*>(&sV[tp * VP + cc * 8 + 4]) = hi4;
    }
  };

  float o[4][8], m[4], l[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f;
  #pragma unroll
    for (int j = 0; j < 8; ++j) o[i][j] = 0.0f;
  }

  const int ntiles = (L + BKV - 1) / BKV;
  gload(0);
  for (int jt = 0; jt < ntiles; ++jt) {
    __syncthreads();
    lstore();
    if (jt + 1 < ntiles) gload(jt + 1);
    __syncthreads();

    float s[4][8];
  #pragma unroll
    for (int i = 0; i < 4; ++i)
  #pragma unroll
      for (int j = 0; j < 8; ++j) s[i][j] = 0.0f;
  #pragma unroll 2
    for (int kk = 0; kk < 16; ++kk) {
      u32x2 qa[4], kb[8];
  #pragma unroll
      for (int i = 0; i < 4; ++i)
        qa[i] =
            *reinterpret_cast<const u32x2*>(&sQ[(ty + 16 * i) * QP + 2 * kk]);
  #pragma unroll
      for (int j = 0; j < 8; ++j)
        kb[j] = *reinterpret_cast<const u32x2*>(
            &sK[(2 * tx + 16 * (j >> 1) + (j & 1)) * KP + 2 * kk]);
  #pragma unroll
      for (int i = 0; i < 4; ++i)
  #pragma unroll
        for (int j = 0; j < 8; ++j)
          s[i][j] = dot2u(qa[i].y, kb[j].y, dot2u(qa[i].x, kb[j].x, s[i][j]));
    }

    // key columns of this tile are jt 64 + 2 tx + 16 u + e; columns >= L get
    // p = 0
    const int kcol0 = jt * BKV + 2 * tx;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float mx = -INFINITY;
  #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const bool valid = kcol0 + 16 * (j >> 1) + (j & 1) < L;
        s[i][j] = valid ? s[i][j] * cs : -INFINITY;
        mx = fmaxf(mx, s[i][j]);
      }
      mx = fmaxf(mx, __shfl_xor(mx, 1));
      mx = fmaxf(mx, __shfl_xor(mx, 2));
      mx = fmaxf(mx, __shfl_xor(mx, 4));
      const float m_new = fmaxf(m[i], mx);
      const float alpha = __builtin_amdgcn_exp2f(m[i] - m_new);
      float rs = 0.0f;
  #pragma unroll
      for (int j = 0; j < 8; ++j) {
        s[i][j] = __builtin_amdgcn_exp2f(s[i][j] - m_new);
        rs += s[i][j];
      }
      l[i] = l[i] * alpha + rs;
      m[i] = m_new;
  #pragma unroll
      for (int j = 0; j < 8; ++j) o[i][j] *= alpha;
  #pragma unroll
      for (int u = 0; u < 4; ++u)
        sP[(ty + 16 * i) * PP + tx + 8 * u] =
            pack_h2(s[i][2 * u], s[i][2 * u + 1]);
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

  #pragma unroll 2
    for (int t2 = 0; t2 < 16; ++t2) {
      u32x2 pa[4];
  #pragma unroll
      for (int i = 0; i < 4; ++i)
        pa[i] =
            *reinterpret_cast<const u32x2*>(&sP[(ty + 16 * i) * PP + 2 * t2]);
  #pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
  #pragma unroll
        for (int u = 0; u < 4; ++u) {
          const u32x2 vv = *reinterpret_cast<const u32x2*>(
              &sV[(2 * t2 + tt) * VP + 2 * tx + 16 * u]);
  #pragma unroll
          for (int i = 0; i < 4; ++i) {
            const unsigned pk = tt ? pa[i].y : pa[i].x;
            o[i][2 * u] = dot2u(pk, vv.x, o[i][2 * u]);
            o[i][2 * u + 1] = dot2u(pk, vv.y, o[i][2 * u + 1]);
          }
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float lt = l[i];
    lt += __shfl_xor(lt, 1);
    lt += __shfl_xor(lt, 2);
    lt += __shfl_xor(lt, 4);
    const float inv = 1.0f / lt;
    const int q = q0 + ty + 16 * i;
    if (q < L) {
      unsigned* dst = reinterpret_cast<unsigned*>(
          out + static_cast<size_t>(tok0 + q) * D + head * kHeadDim + 2 * tx);
  #pragma unroll
      for (int u = 0; u < 4; ++u)
        dst[8 * u] = pack_h2(o[i][2 * u] * inv, o[i][2 * u + 1] * inv);
    }
  }
#endif
}

}  // namespace inkling2_vit
}  // namespace vllm

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {
namespace iv = vllm::inkling2_vit;

bool on_gfx1030() {
  const hipDeviceProp_t* p = at::cuda::getCurrentDeviceProperties();
  return std::strncmp(p->gcnArchName, "gfx1030", 7) == 0;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

void check_dev(const torch::Tensor& t, const char* what, at::ScalarType dtype,
               int dim) {
  TORCH_CHECK(t.is_cuda(), "inkling2_vit: ", what,
              " must be a CUDA/HIP tensor");
  TORCH_CHECK(t.scalar_type() == dtype, "inkling2_vit: ", what, " must be ",
              dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.dim() == dim && t.is_contiguous(), "inkling2_vit: ", what,
              " must be a contiguous ", dim, "-D tensor");
}

unsigned blocks_for(int64_t n, int64_t per_block) {
  const int64_t b = (n + per_block - 1) / per_block;
  TORCH_CHECK(b < (int64_t{1} << 31), "inkling2_vit: grid too large");
  return static_cast<unsigned>(b);
}

template <int EPI>
void launch_gemm(const torch::Tensor& a, const torch::Tensor& w, void* c,
                 const float* bias, const float* gamma, float* res, int64_t ldc,
                 hipStream_t stream) {
  const int64_t M = a.size(0), N = w.size(0), K = a.size(1);
  const int64_t ntm = (M + iv::gemm::BM - 1) / iv::gemm::BM;
  const int64_t ntn = (N + iv::gemm::BN - 1) / iv::gemm::BN;
  iv::gemm::gemm_kernel<EPI>
      <<<dim3(static_cast<unsigned>(ntm * ntn)), dim3(iv::gemm::T), 0,
         stream>>>(static_cast<const char*>(a.data_ptr()),
                   static_cast<const char*>(w.data_ptr()), c, bias, gamma, res,
                   static_cast<int>(M), static_cast<int>(N),
                   static_cast<int>(K), static_cast<int>(a.stride(0)),
                   static_cast<int>(w.stride(0)), static_cast<int>(ldc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_gemm_operands(const torch::Tensor& a, const torch::Tensor& w) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  TORCH_CHECK(a.is_cuda() && w.is_cuda(), "inkling2_vit_gemm: CUDA tensors");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf,
      "inkling2_vit_gemm: a and w must be fp16");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1),
              "inkling2_vit_gemm: a [M, K], w [N, K]");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1,
              "inkling2_vit_gemm: unit column stride");
  const int64_t K = a.size(1);
  constexpr int64_t kMaxElems = int64_t{1} << 30;
  TORCH_CHECK(K > 0 && K % iv::gemm::BK == 0, "inkling2_vit_gemm: K ", K,
              " must be a multiple of ", iv::gemm::BK);
  TORCH_CHECK(a.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 &&
                  aligned16(a.data_ptr()) && aligned16(w.data_ptr()),
              "inkling2_vit_gemm: 16-byte aligned operands and row strides");
  TORCH_CHECK(a.size(0) * a.stride(0) < kMaxElems &&
                  w.size(0) * w.stride(0) < kMaxElems,
              "inkling2_vit_gemm: operand too large (32-bit byte offsets)");
}

template <typename Tin, int MODE, typename Stream>
void launch_gather(const torch::Tensor& raw, const torch::Tensor& out,
                   int64_t n_vec, Stream stream) {
  iv::gather_patches_kernel<Tin, MODE>
      <<<blocks_for(n_vec, 256), 256, 0, stream>>>(
          static_cast<const Tin*>(raw.data_ptr()),
          static_cast<_Float16*>(out.data_ptr()), n_vec,
          static_cast<int>(out.size(1)));
}

}  // namespace

// out = fp16 patch rows of the raw cells; mode 0: 2 raw / 255 - 1 (the
// transformers image processor), 1: CLIP normalisation rounded to bf16 and
// mapped back to [-1, 1] (the NVIDIA vLLM reference)
void inkling2_vit_gather_patches(torch::Tensor raw, torch::Tensor out,
                                 int64_t mode) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  TORCH_CHECK(raw.is_cuda() && out.is_cuda(), "gather_patches: CUDA tensors");
  TORCH_CHECK(raw.dim() == 4 && raw.size(1) == iv::kCellPixels &&
                  raw.size(2) == iv::kCellPixels && raw.size(3) == 3 &&
                  raw.is_contiguous(),
              "gather_patches: raw must be contiguous [cells, 40, 40, 3]");
  TORCH_CHECK(raw.scalar_type() == torch::kHalf ||
                  raw.scalar_type() == torch::kBFloat16,
              "gather_patches: raw must be fp16 or bf16");
  TORCH_CHECK(mode == iv::kPixelExact || mode == iv::kPixelClipBf16,
              "gather_patches: mode must be 0 (exact) or 1 (clip_bf16)");
  check_dev(out, "gather_patches out", torch::kHalf, 2);
  const int64_t cells = raw.size(0), kpad = out.size(1);
  TORCH_CHECK(
      out.size(0) == 4 * cells && kpad >= iv::kPatchK && kpad % 4 == 0,
      "gather_patches: out must be [4 cells, kpad >= 1200, kpad % 4 == 0]");
  const at::cuda::OptionalCUDAGuard guard(device_of(raw));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n_vec = out.size(0) * (kpad / 4);
  if (n_vec == 0) return;
  const bool half = raw.scalar_type() == torch::kHalf;
  if (mode == iv::kPixelExact) {
    if (half)
      launch_gather<_Float16, iv::kPixelExact>(raw, out, n_vec, stream);
    else
      launch_gather<__hip_bfloat16, iv::kPixelExact>(raw, out, n_vec, stream);
  } else {
    if (half)
      launch_gather<_Float16, iv::kPixelClipBf16>(raw, out, n_vec, stream);
    else
      launch_gather<__hip_bfloat16, iv::kPixelClipBf16>(raw, out, n_vec,
                                                        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = epilogue(a . w^T); 0 fp16, 1 + bias, 2 gelu(+ bias), 3 gelu, 4 fp32
void inkling2_vit_gemm(torch::Tensor a, torch::Tensor w, torch::Tensor out,
                       const std::optional<torch::Tensor>& bias,
                       int64_t epilogue) {
  check_gemm_operands(a, w);
  const bool f32 = epilogue == iv::gemm::kF32;
  TORCH_CHECK(epilogue >= iv::gemm::kF16 && epilogue <= iv::gemm::kF32,
              "inkling2_vit_gemm: epilogue ", epilogue, " not in [0, 4]");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == a.size(0) &&
                  out.size(1) == w.size(0) && out.stride(1) == 1 &&
                  out.scalar_type() == (f32 ? torch::kFloat : torch::kHalf),
              "inkling2_vit_gemm: out [M, N] ", f32 ? "fp32" : "fp16");
  const bool need_bias =
      epilogue == iv::gemm::kF16Bias || epilogue == iv::gemm::kF16BiasGelu;
  const float* bptr = nullptr;
  if (need_bias) {
    TORCH_CHECK(bias.has_value() && bias->is_cuda() &&
                    bias->scalar_type() == torch::kFloat && bias->dim() == 1 &&
                    bias->size(0) == w.size(0) && bias->is_contiguous(),
                "inkling2_vit_gemm: bias must be fp32 [N]");
    bptr = bias->data_ptr<float>();
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (a.size(0) == 0 || w.size(0) == 0) return;
  void* c = out.data_ptr();
  switch (epilogue) {
    case iv::gemm::kF16:
      launch_gemm<iv::gemm::kF16>(a, w, c, nullptr, nullptr, nullptr,
                                  out.stride(0), stream);
      break;
    case iv::gemm::kF16Bias:
      launch_gemm<iv::gemm::kF16Bias>(a, w, c, bptr, nullptr, nullptr,
                                      out.stride(0), stream);
      break;
    case iv::gemm::kF16BiasGelu:
      launch_gemm<iv::gemm::kF16BiasGelu>(a, w, c, bptr, nullptr, nullptr,
                                          out.stride(0), stream);
      break;
    case iv::gemm::kF16Gelu:
      launch_gemm<iv::gemm::kF16Gelu>(a, w, c, nullptr, nullptr, nullptr,
                                      out.stride(0), stream);
      break;
    default:
      launch_gemm<iv::gemm::kF32>(a, w, c, nullptr, nullptr, nullptr,
                                  out.stride(0), stream);
      break;
  }
}

// res += gamma * (a . w^T + bias), res fp32 [M, N] updated in place
void inkling2_vit_gemm_residual(torch::Tensor a, torch::Tensor w,
                                torch::Tensor bias, torch::Tensor gamma,
                                torch::Tensor res) {
  check_gemm_operands(a, w);
  check_dev(bias, "bias", torch::kFloat, 1);
  check_dev(gamma, "gamma", torch::kFloat, 1);
  check_dev(res, "res", torch::kFloat, 2);
  TORCH_CHECK(bias.size(0) == w.size(0) && gamma.size(0) == w.size(0) &&
                  res.size(0) == a.size(0) && res.size(1) == w.size(0),
              "inkling2_vit_gemm_residual: bias / gamma [N], res [M, N]");
  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (a.size(0) == 0 || w.size(0) == 0) return;
  launch_gemm<iv::gemm::kResidual>(
      a, w, nullptr, bias.data_ptr<float>(), gamma.data_ptr<float>(),
      res.data_ptr<float>(), res.stride(0), stream);
}

void inkling2_vit_embed(torch::Tensor emb, torch::Tensor pos, torch::Tensor cls,
                        torch::Tensor ln_w, torch::Tensor ln_b,
                        torch::Tensor meta, torch::Tensor h, double eps) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(emb, "emb", torch::kFloat, 2);
  check_dev(pos, "pos", torch::kFloat, 2);
  check_dev(cls, "cls", torch::kFloat, 1);
  check_dev(ln_w, "ln_w", torch::kFloat, 1);
  check_dev(ln_b, "ln_b", torch::kFloat, 1);
  check_dev(meta, "meta", torch::kInt, 2);
  check_dev(h, "h", torch::kFloat, 2);
  const int64_t D = h.size(1), T = h.size(0), n_img = meta.size(0);
  const int64_t G = static_cast<int64_t>(
      std::llround(std::sqrt(static_cast<double>(pos.size(0)))));
  TORCH_CHECK(
      D % 128 == 0 && D <= iv::kMaxHidden && emb.size(1) == D &&
          pos.size(1) == D && cls.size(0) == D && ln_w.size(0) == D &&
          ln_b.size(0) == D && meta.size(1) == 4 && G * G == pos.size(0),
      "inkling2_vit_embed: shape mismatch (hidden % 128 == 0, <= 2048)");
  TORCH_CHECK(T == emb.size(0) + n_img,
              "inkling2_vit_embed: T must be patches + images");
  const at::cuda::OptionalCUDAGuard guard(device_of(h));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  iv::embed_kernel<<<blocks_for(T, 8), 256, 0, stream>>>(
      emb.data_ptr<float>(), pos.data_ptr<float>(), cls.data_ptr<float>(),
      ln_w.data_ptr<float>(), ln_b.data_ptr<float>(), meta.data_ptr<int>(),
      static_cast<int>(n_img), h.data_ptr<float>(), static_cast<int>(T),
      static_cast<int>(D), static_cast<int>(G), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_vit_layernorm(torch::Tensor h, torch::Tensor w, torch::Tensor b,
                            torch::Tensor out, double eps) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(h, "h", torch::kFloat, 2);
  check_dev(w, "w", torch::kFloat, 1);
  check_dev(b, "b", torch::kFloat, 1);
  check_dev(out, "out", torch::kHalf, 2);
  const int64_t rows = h.size(0), D = h.size(1);
  TORCH_CHECK(
      D % 128 == 0 && D <= iv::kMaxHidden && out.size(0) == rows &&
          out.size(1) == D && w.size(0) == D && b.size(0) == D,
      "inkling2_vit_layernorm: shape mismatch (hidden % 128 == 0, <= 2048)");
  const at::cuda::OptionalCUDAGuard guard(device_of(h));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (rows == 0) return;
  iv::layernorm_kernel<<<blocks_for(rows, 8), 256, 0, stream>>>(
      h.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
      static_cast<_Float16*>(out.data_ptr()), static_cast<int>(rows),
      static_cast<int>(D), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_vit_rope_tables(torch::Tensor meta, torch::Tensor inv_freq,
                              torch::Tensor cs) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(meta, "meta", torch::kInt, 2);
  check_dev(inv_freq, "inv_freq", torch::kFloat, 1);
  check_dev(cs, "cs", torch::kFloat, 3);
  TORCH_CHECK(
      inv_freq.size(0) == 16 && cs.size(1) == 32 && cs.size(2) == 2 &&
          meta.size(1) == 4,
      "inkling2_vit_rope_tables: inv_freq [16], cs [T, 32, 2], meta [n, 4]");
  const at::cuda::OptionalCUDAGuard guard(device_of(cs));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t T = cs.size(0);
  if (T == 0) return;
  iv::rope_tables_kernel<<<blocks_for(T * 32, 256), 256, 0, stream>>>(
      meta.data_ptr<int>(), static_cast<int>(meta.size(0)),
      inv_freq.data_ptr<float>(),
      reinterpret_cast<float2*>(cs.data_ptr<float>()), static_cast<int>(T));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_vit_rope_qk(torch::Tensor qkv, torch::Tensor cs) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(qkv, "qkv", torch::kHalf, 2);
  check_dev(cs, "cs", torch::kFloat, 3);
  const int64_t T = qkv.size(0), D = qkv.size(1) / 3;
  TORCH_CHECK(
      qkv.size(1) == 3 * D && D % (2 * 32) == 0 && cs.size(0) == T &&
          cs.size(1) == 32 && cs.size(2) == 2 && aligned16(qkv.data_ptr()),
      "inkling2_vit_rope_qk: qkv [T, 3 D] (D a multiple of 64), cs [T, 32, 2]");
  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  iv::rope_qk_kernel<<<blocks_for(T * (D / 2), 256), 256, 0, stream>>>(
      static_cast<_Float16*>(qkv.data_ptr()),
      reinterpret_cast<const float2*>(cs.data_ptr<float>()),
      static_cast<int>(T), static_cast<int>(D));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[T, D] = softmax(q k^T / 8) v per image and head; qkv [T, 3 D] fp16
// (q | k | v, head h at columns 64 h); blk_prefix [n_img + 1] = prefix sums
// of the number of 64-row query blocks of every image; num_qblocks =
// blk_prefix[n_img].
void inkling2_vit_attention(torch::Tensor qkv, torch::Tensor out,
                            torch::Tensor meta, torch::Tensor blk_prefix,
                            int64_t num_qblocks) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(qkv, "qkv", torch::kHalf, 2);
  check_dev(out, "out", torch::kHalf, 2);
  check_dev(meta, "meta", torch::kInt, 2);
  check_dev(blk_prefix, "blk_prefix", torch::kInt, 1);
  const int64_t T = qkv.size(0), D = qkv.size(1) / 3, n_img = meta.size(0);
  TORCH_CHECK(qkv.size(1) == 3 * D && D % iv::kHeadDim == 0 &&
                  out.size(0) == T && out.size(1) == D && meta.size(1) == 4 &&
                  blk_prefix.size(0) == n_img + 1 &&
                  aligned16(qkv.data_ptr()) && (3 * D) % 8 == 0,
              "inkling2_vit_attention: qkv [T, 3 D] with D a multiple of 64, "
              "out [T, D]");
  const int64_t heads = D / iv::kHeadDim;
  TORCH_CHECK(heads * num_qblocks < (int64_t{1} << 31),
              "inkling2_vit_attention: grid too large");
  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 0 || num_qblocks == 0) return;
  iv::attention_kernel<<<static_cast<unsigned>(heads * num_qblocks),
                         iv::attn::THREADS, 0, stream>>>(
      static_cast<const _Float16*>(qkv.data_ptr()),
      static_cast<_Float16*>(out.data_ptr()), meta.data_ptr<int>(),
      blk_prefix.data_ptr<int>(), static_cast<int>(n_img),
      static_cast<int>(num_qblocks), static_cast<int>(D),
      0.125f * iv::attn::kLog2e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inkling2_vit_pool(torch::Tensor h, torch::Tensor meta, torch::Tensor out) {
  TORCH_CHECK(on_gfx1030(), "inkling2_vit: needs a gfx1030 device");
  check_dev(h, "h", torch::kFloat, 2);
  check_dev(meta, "meta", torch::kInt, 2);
  check_dev(out, "out", torch::kHalf, 2);
  const int64_t T = h.size(0), D = h.size(1), n_img = meta.size(0);
  TORCH_CHECK(D % 4 == 0 && out.size(1) == D && out.size(0) == T - n_img &&
                  meta.size(1) == 4,
              "inkling2_vit_pool: out must be [T - n_img, D]");
  const at::cuda::OptionalCUDAGuard guard(device_of(h));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 0) return;
  iv::pool_kernel<<<blocks_for(T, 8), 256, 0, stream>>>(
      h.data_ptr<float>(), meta.data_ptr<int>(), static_cast<int>(n_img),
      static_cast<_Float16*>(out.data_ptr()), static_cast<int>(T),
      static_cast<int>(D));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
