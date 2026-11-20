// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// HI2 v3.3 CPTC: block-local tail-biting streams, fp16 table and block scales.
// The v3.2 ladder kernels and dispatch remain unchanged.

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ops.h"
#include "rdna_hint_flex_dense_gemm_rdna2.cuh"

namespace {
constexpr int kWave = 32;
constexpr int kThreads = 256;
constexpr int kRows = kThreads / kWave;
constexpr int kTableSize = 127;

struct CptcWeight {
  const uint32_t* codes;
  const int32_t* code_off;
  const half* scales;
  const int32_t* param_off;
  const uint8_t* nbits;
  const uint8_t* blocks;
  const int32_t* coefficients;
  const half* tables;
};

__device__ __forceinline__ half decode_value(uint32_t word, uint32_t next,
                                             int shift, uint32_t a0,
                                             uint32_t a1, const half* table,
                                             half scale) {
  uint32_t state = word >> shift;
  if (shift > 16) state |= next << (32 - shift);
  const int index =
      (((state * a0) & 65535u) >> 10) + (((state * a1) & 65535u) >> 10);
  return __hmul(scale, table[index]);
}

__device__ __forceinline__ uint32_t next_word(const uint32_t* words, int word,
                                              int block_words) {
  const int block_start = word & ~(block_words - 1);
  return words[block_start + ((word + 1) & (block_words - 1))];
}

template <int NB, int MT>
__device__ __forceinline__ void row_dot(const half* x, int64_t ldx, int m0,
                                        int M, const uint32_t* words,
                                        const half* scales, int block, int lane,
                                        uint32_t a0, uint32_t a1,
                                        const half* table, int wi_begin,
                                        int wi_end, float (&acc)[MT]) {
  constexpr int kCodes = 32 / NB;
  const int block_words = block * NB / 32;
  for (int wi = wi_begin + lane; wi < wi_end; wi += kWave) {
    const uint32_t word = words[wi];
    const uint32_t next = next_word(words, wi, block_words);
    const int col = wi * kCodes;
    const half scale = scales[col / block];
    half2 weight[kCodes / 2];
#pragma unroll
    for (int j = 0; j < kCodes / 2; ++j) {
      weight[j] = __halves2half2(
          decode_value(word, next, 2 * j * NB, a0, a1, table, scale),
          decode_value(word, next, (2 * j + 1) * NB, a0, a1, table, scale));
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m < M) {
        const half* values = x + static_cast<int64_t>(m0 + m) * ldx + col;
#pragma unroll
        for (int j = 0; j < kCodes / 2; ++j) {
          const half2 input = __halves2half2(values[2 * j], values[2 * j + 1]);
#if defined(__gfx1030__)
          acc[m] = __builtin_amdgcn_fdot2(weight[j], input, acc[m], false);
#else
          const float2 a = __half22float2(weight[j]);
          const float2 b = __half22float2(input);
          acc[m] = fmaf(a.x, b.x, acc[m]);
          acc[m] = fmaf(a.y, b.y, acc[m]);
#endif
        }
      }
    }
  }
}

template <int MT>
__global__ void cptc_gemv(const half* x, int64_t ldx, CptcWeight w, half* out,
                          int64_t ldo, int M, int R, int K) {
  __shared__ half tables[2 * kTableSize];
  if (threadIdx.x < 2 * kTableSize) tables[threadIdx.x] = w.tables[threadIdx.x];
  __syncthreads();
  const int row = blockIdx.x * kRows + threadIdx.x / kWave;
  if (row >= R) return;
  const int lane = threadIdx.x % kWave;
  const int m0 = blockIdx.y * MT;
  const int bits = w.nbits[row];
  const int slot = bits == 2 ? 0 : 1;
  const uint32_t a0 = w.coefficients[2 * slot];
  const uint32_t a1 = w.coefficients[2 * slot + 1];
  float acc[MT] = {};
  if (bits == 2) {
    row_dot<2, MT>(x, ldx, m0, M, w.codes + w.code_off[row],
                   w.scales + w.param_off[row], w.blocks[row], lane, a0, a1,
                   tables + slot * kTableSize, 0, K / 16, acc);
  } else {
    row_dot<4, MT>(x, ldx, m0, M, w.codes + w.code_off[row],
                   w.scales + w.param_off[row], w.blocks[row], lane, a0, a1,
                   tables + slot * kTableSize, 0, K / 8, acc);
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2)
      acc[m] += __shfl_down(acc[m], offset, kWave);
    if (lane == 0 && m0 + m < M)
      out[static_cast<int64_t>(m0 + m) * ldo + row] = __float2half_rn(acc[m]);
  }
}

// Cooperative split-K decode GEMV (RDNA_HINT_FLEX_GEMV=v2). SPLIT waves of a
// block own the same output row and disjoint word ranges; lane 0 of each wave
// publishes its fp32 sums in shared memory and wave 0 adds them in wave order.
// No global scratch and no atomics. A wave whose row is out of range still
// reaches the barrier with zero accumulators.
template <int MT, int SPLIT>
__global__ void cptc_gemv_split(const half* x, int64_t ldx, CptcWeight w,
                                half* out, int64_t ldo, int M, int R, int K) {
  constexpr int kSplitRows = kRows / SPLIT;
  static_assert(kRows % SPLIT == 0, "split must divide the waves per block");
  __shared__ half tables[2 * kTableSize];
  __shared__ float red[kRows][MT];
  if (threadIdx.x < 2 * kTableSize) tables[threadIdx.x] = w.tables[threadIdx.x];
  __syncthreads();
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int part = wave % SPLIT;
  const int row = blockIdx.x * kSplitRows + wave / SPLIT;
  const int m0 = blockIdx.y * MT;
  const bool active = row < R;
  float acc[MT] = {};
  if (active) {
    const int bits = w.nbits[row];
    const int slot = bits == 2 ? 0 : 1;
    const uint32_t a0 = w.coefficients[2 * slot];
    const uint32_t a1 = w.coefficients[2 * slot + 1];
    const uint32_t* words = w.codes + w.code_off[row];
    const half* scales = w.scales + w.param_off[row];
    const half* table = tables + slot * kTableSize;
    const int block = w.blocks[row];
    const int nwords = bits == 2 ? K / 16 : K / 8;
    const int wi_begin =
        static_cast<int>((static_cast<int64_t>(nwords) * part) / SPLIT);
    const int wi_end =
        static_cast<int>((static_cast<int64_t>(nwords) * (part + 1)) / SPLIT);
    if (bits == 2) {
      row_dot<2, MT>(x, ldx, m0, M, words, scales, block, lane, a0, a1, table,
                     wi_begin, wi_end, acc);
    } else {
      row_dot<4, MT>(x, ldx, m0, M, words, scales, block, lane, a0, a1, table,
                     wi_begin, wi_end, acc);
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2)
      acc[m] += __shfl_down(acc[m], offset, kWave);
    if (lane == 0) red[wave][m] = acc[m];
  }
  __syncthreads();
  if (active && part == 0 && lane == 0) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m >= M) continue;
      float sum = red[wave][m];
#pragma unroll
      for (int p = 1; p < SPLIT; ++p) sum += red[wave + p][m];
      out[static_cast<int64_t>(m0 + m) * ldo + row] = __float2half_rn(sum);
    }
  }
}

__global__ void cptc_dequant(CptcWeight w, half2* out, int R, int K) {
  __shared__ half tables[2 * kTableSize];
  if (threadIdx.x < 2 * kTableSize) tables[threadIdx.x] = w.tables[threadIdx.x];
  __syncthreads();
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= static_cast<int64_t>(R) * (K / 2)) return;
  const int row = pair / (K / 2);
  const int col = 2 * (pair % (K / 2));
  const int bits = w.nbits[row];
  const int slot = bits == 2 ? 0 : 1;
  const int block = w.blocks[row];
  const int wi = col * bits / 32;
  const int shift = (col * bits) % 32;
  const uint32_t* words = w.codes + w.code_off[row];
  const uint32_t word = words[wi];
  const uint32_t next = next_word(words, wi, block * bits / 32);
  const half scale = w.scales[w.param_off[row] + col / block];
  const uint32_t a0 = w.coefficients[2 * slot];
  const uint32_t a1 = w.coefficients[2 * slot + 1];
  const half* table = tables + slot * kTableSize;
  out[pair] = __halves2half2(
      decode_value(word, next, shift, a0, a1, table, scale),
      decode_value(word, next, shift + bits, a0, a1, table, scale));
}

// ---------------------------------------------------------------------------
// CPTC prefill (prefill_gemm == 5, "cptc-fused"): C[M, R] = X[M, K] W^T with
// the CPTC decode inside the GEMM tile. A 256-thread block (eight Wave32)
// computes a 64 x 64 tile. The weights of the tile are decoded once per
// 64-column K step into LDS and reused by all 64 activation rows; no [R, K]
// fp16 dense tensor is allocated. Each thread owns one 16-column group of one
// weight row during the decode and a 4 x 4 micro-tile during the dot, using
// v_dot2_f32_f16 on LDS fragments. M and R tails are masked; K is 64-aligned
// (checked by the caller) and the last tile row/column is not written.
// ---------------------------------------------------------------------------
constexpr int kFusedM = 64;
constexpr int kFusedN = 64;
constexpr int kFusedK = 64;
constexpr int kFusedPad = 8;
constexpr int kFusedLd = kFusedK + kFusedPad;

struct CptcRowHdr {
  int32_t code_off;
  int32_t param_off;
  uint8_t nbits;
  uint8_t block;
};

template <int NB>
__device__ __forceinline__ void decode_group(const uint32_t* words, int block,
                                             int col, uint32_t a0, uint32_t a1,
                                             const half* table, half scale,
                                             half2 (&out)[8]) {
  const int block_words = block * NB / 32;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const int c = col + 2 * p;
    const int64_t bit = static_cast<int64_t>(c) * NB;
    const int wi = static_cast<int>(bit / 32);
    const int shift = static_cast<int>(bit % 32);
    const uint32_t word = words[wi];
    const uint32_t next = next_word(words, wi, block_words);
    out[p] = __halves2half2(
        decode_value(word, next, shift, a0, a1, table, scale),
        decode_value(word, next, shift + NB, a0, a1, table, scale));
  }
}

template <bool ALIGN16>
__global__ void __launch_bounds__(kThreads)
    cptc_prefill_fused(const half* __restrict__ x, int64_t ldx, CptcWeight w,
                       half* __restrict__ out, int64_t ldo, int M, int R,
                       int K) {
  __shared__ __align__(16) half xs[kFusedM * kFusedLd];
  __shared__ __align__(16) half ws[kFusedN * kFusedLd];
  __shared__ CptcRowHdr hdr[kFusedN];
  __shared__ half tables[2 * kTableSize];
  const int tid = threadIdx.x;
  const int m_base = blockIdx.y * kFusedM;
  const int r_base = blockIdx.x * kFusedN;
  if (tid < 2 * kTableSize) tables[tid] = w.tables[tid];
  for (int i = tid; i < kFusedN; i += kThreads) {
    const int r = r_base + i;
    CptcRowHdr h = {};
    if (r < R) {
      h.code_off = w.code_off[r];
      h.param_off = w.param_off[r];
      h.nbits = w.nbits[r];
      h.block = w.blocks[r];
    }
    hdr[i] = h;
  }
  __syncthreads();

  constexpr int kChunks = kFusedK / 8;
  constexpr int kXPerThread = kFusedM * kChunks / kThreads;
  constexpr int kGroups = kFusedK / 16;
  constexpr int kWPerThread = kFusedN * kGroups / kThreads;
  static_assert(kFusedM * kChunks % kThreads == 0 && kXPerThread >= 1);
  static_assert(kFusedN * kGroups % kThreads == 0 && kWPerThread >= 1);
  uint4 xr[kXPerThread];

  auto load_x = [&](int k0) {
#pragma unroll
    for (int v = 0; v < kXPerThread; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / kChunks;
      const int c = (i % kChunks) * 8;
      const int xm = m_base + row;
      xr[v] = make_uint4(0, 0, 0, 0);
      if (xm < M) {
        const half* src = x + static_cast<int64_t>(xm) * ldx + k0 + c;
        if constexpr (ALIGN16) {
          xr[v] = *reinterpret_cast<const uint4*>(src);
        } else {
          union {
            half h[8];
            uint4 u;
          } t;
#pragma unroll
          for (int j = 0; j < 8; ++j) t.h[j] = src[j];
          xr[v] = t.u;
        }
      }
    }
  };
  auto store_x = [&]() {
#pragma unroll
    for (int v = 0; v < kXPerThread; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / kChunks;
      const int c = (i % kChunks) * 8;
      *reinterpret_cast<uint4*>(xs + row * kFusedLd + c) = xr[v];
    }
  };
  auto decode_w = [&](int k0) {
#pragma unroll
    for (int v = 0; v < kWPerThread; ++v) {
      const int i = tid + v * kThreads;
      const int row = i / kGroups;
      const int gl = i % kGroups;
      const CptcRowHdr h = hdr[row];
      union {
        half2 h2[8];
        uint4 u[2];
      } o;
      if (h.nbits != 0) {
        const int col = k0 + gl * 16;
        const uint32_t* words = w.codes + h.code_off;
        const half scale = w.scales[h.param_off + col / h.block];
        const uint32_t a0 = w.coefficients[h.nbits == 2 ? 0 : 2];
        const uint32_t a1 = w.coefficients[h.nbits == 2 ? 1 : 3];
        const half* table = tables + (h.nbits == 2 ? 0 : kTableSize);
        if (h.nbits == 2)
          decode_group<2>(words, h.block, col, a0, a1, table, scale, o.h2);
        else
          decode_group<4>(words, h.block, col, a0, a1, table, scale, o.h2);
      } else {
        o.u[0] = make_uint4(0, 0, 0, 0);
        o.u[1] = o.u[0];
      }
      half* dst = ws + row * kFusedLd + gl * 16;
      *reinterpret_cast<uint4*>(dst) = o.u[0];
      *reinterpret_cast<uint4*>(dst + 8) = o.u[1];
    }
  };

  constexpr int TM = kFusedM / 16;
  constexpr int TN = kFusedN / 16;
  const int tx = tid % 16, ty = tid / 16;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  load_x(0);
  for (int k0 = 0; k0 < K; k0 += kFusedK) {
    __syncthreads();
    store_x();
    decode_w(k0);
    __syncthreads();
    if (k0 + kFusedK < K) load_x(k0 + kFusedK);
#pragma unroll
    for (int kk = 0; kk < kFusedK; kk += 8) {
      half2 a[TM][4], b[TN][4];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(xs + (ty + 16 * i) * kFusedLd + kk);
        const half2* hv = reinterpret_cast<const half2*>(&v);
#pragma unroll
        for (int t = 0; t < 4; ++t) a[i][t] = hv[t];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(ws + (tx + 16 * j) * kFusedLd + kk);
        const half2* hv = reinterpret_cast<const half2*>(&v);
#pragma unroll
        for (int t = 0; t < 4; ++t) b[j][t] = hv[t];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          float c = acc[i][j];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
#if defined(__gfx1030__)
            c = __builtin_amdgcn_fdot2(a[i][t], b[j][t], c, false);
#else
            const float2 av = __half22float2(a[i][t]);
            const float2 bv = __half22float2(b[j][t]);
            c = fmaf(av.x, bv.x, c);
            c = fmaf(av.y, bv.y, c);
#endif
          }
          acc[i][j] = c;
        }
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m_base + ty + 16 * i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = r_base + tx + 16 * j;
      if (r < R)
        out[static_cast<int64_t>(m) * ldo + r] = __float2half_rn(acc[i][j]);
    }
  }
}

void check_tensor(const torch::Tensor& t, const torch::Device& device,
                  at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype &&
                  t.is_contiguous(),
              "hi2 CPTC: ", name,
              " must be contiguous and have the expected dtype/device");
}

CptcWeight check_weight(const torch::Tensor& codes,
                        const torch::Tensor& code_off,
                        const torch::Tensor& scale,
                        const torch::Tensor& param_off,
                        const torch::Tensor& nbits, const torch::Tensor& blocks,
                        const torch::Tensor& coefficients,
                        const torch::Tensor& tables) {
  const auto device = scale.device();
  check_tensor(codes, device, torch::kInt32, "codes");
  check_tensor(code_off, device, torch::kInt32, "code_off");
  check_tensor(scale, device, torch::kHalf, "scale");
  check_tensor(param_off, device, torch::kInt32, "param_off");
  check_tensor(nbits, device, torch::kUInt8, "row_nbits");
  check_tensor(blocks, device, torch::kUInt8, "row_block");
  check_tensor(coefficients, device, torch::kInt32, "coefficients");
  check_tensor(tables, device, torch::kHalf, "tables");
  const auto R = nbits.numel();
  TORCH_CHECK(nbits.dim() == 1 && blocks.numel() == R &&
                  code_off.numel() == R + 1 && param_off.numel() == R + 1 &&
                  coefficients.sizes() == torch::IntArrayRef({2, 2}) &&
                  tables.sizes() == torch::IntArrayRef({2, kTableSize}) &&
                  R < INT_MAX / kRows,
              "hi2 CPTC: invalid weight metadata shapes");
  // Offset contents/row classes are validated on CPU at checkpoint load, as
  // for v3.2. Reading them here would synchronize every graph replay.
  return {reinterpret_cast<const uint32_t*>(codes.data_ptr<int32_t>()),
          code_off.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(scale.data_ptr()),
          param_off.data_ptr<int32_t>(),
          nbits.data_ptr<uint8_t>(),
          blocks.data_ptr<uint8_t>(),
          coefficients.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(tables.data_ptr())};
}

void launch_dequant(CptcWeight w, torch::Tensor& out, hipStream_t stream) {
  const auto pairs = out.numel() / 2;
  if (pairs == 0) return;
  cptc_dequant<<<(pairs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      w, reinterpret_cast<half2*>(out.data_ptr()), out.size(0), out.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Decode GEMV selection, mirroring the dense rdna_hint_flex_rdna2.cu
// convention: unset/v1 = the existing one-wave-per-row kernel, v2/2 = the
// cooperative split-K kernel, auto = the existing kernel (the dense auto mode
// changes layout and stays v1 for CPTC). Read once per process, like the dense
// switch, so the kernel a process launches does not depend on later env edits.
enum CptcGemvVariant { kCptcGemvV1 = 1, kCptcGemvV2 = 2 };

int cptc_gemv_variant() {
  static const int v = [] {
    const char* e = std::getenv("RDNA_HINT_FLEX_GEMV");
    if (e != nullptr && (std::strcmp(e, "v2") == 0 || std::strcmp(e, "2") == 0))
      return static_cast<int>(kCptcGemvV2);
    return static_cast<int>(kCptcGemvV1);
  }();
  return v;
}

// Same row-count thresholds as the dense GEMV: split the K range over four
// waves when rows do not fill the GPU. K is 64-aligned and every row has at
// least four 2-bit words, so each part takes at least one word.
int cptc_gemv_split_count(int64_t R) {
  return R >= 8192 ? 1 : (R >= 4096 ? 2 : 4);
}

bool on_gfx1030() {
  const hipDeviceProp_t* p = at::cuda::getCurrentDeviceProperties();
  return std::strncmp(p->gcnArchName, "gfx1030", 7) == 0;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

void launch_gemv(const torch::Tensor& x, CptcWeight w, torch::Tensor& out,
                 hipStream_t stream) {
  const int M = x.size(0), R = out.size(1), K = x.size(1);
  const int mt = M == 1 ? 1 : (M <= 4 ? 4 : 8);
  const int requested_split =
      cptc_gemv_variant() == kCptcGemvV2 ? cptc_gemv_split_count(R) : 1;
  // Without a K partition, reuse v1 instead of paying for an LDS reduction.
  const int split = requested_split > 1 ? requested_split : 0;
  const int rows = split > 0 ? kRows / split : kRows;
  const dim3 grid((R + rows - 1) / rows, (M + mt - 1) / mt);
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  half* op = reinterpret_cast<half*>(out.data_ptr());
#define CPTC_GEMV_SPLIT(MT, SPLIT)                           \
  cptc_gemv_split<MT, SPLIT><<<grid, kThreads, 0, stream>>>( \
      xp, x.stride(0), w, op, out.stride(0), M, R, K)
#define CPTC_GEMV_V2(MT)      \
  do {                        \
    if (split == 2)           \
      CPTC_GEMV_SPLIT(MT, 2); \
    else                      \
      CPTC_GEMV_SPLIT(MT, 4); \
  } while (0)
#define CPTC_GEMV(MT)                                                  \
  cptc_gemv<MT><<<grid, kThreads, 0, stream>>>(xp, x.stride(0), w, op, \
                                               out.stride(0), M, R, K)
  if (split > 0) {
    if (mt == 1) {
      CPTC_GEMV_V2(1);
    } else if (mt == 4) {
      CPTC_GEMV_V2(4);
    } else {
      CPTC_GEMV_V2(8);
    }
  } else if (mt == 1) {
    CPTC_GEMV(1);
  } else if (mt == 4) {
    CPTC_GEMV(4);
  } else {
    CPTC_GEMV(8);
  }
#undef CPTC_GEMV
#undef CPTC_GEMV_V2
#undef CPTC_GEMV_SPLIT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prefill_fused(const torch::Tensor& x, CptcWeight w,
                          torch::Tensor& out, int64_t R, hipStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(on_gfx1030(),
              "hi2 CPTC: cptc-fused prefill requires the gfx1030 backend");
  TORCH_CHECK(
      M > 0 && M <= INT_MAX && R <= INT_MAX && K <= INT_MAX && K % kFusedK == 0,
      "hi2 CPTC: cptc-fused prefill needs 32-bit M/R/K and K % 64 == 0");
  const dim3 grid(static_cast<unsigned>((R + kFusedN - 1) / kFusedN),
                  static_cast<unsigned>((M + kFusedM - 1) / kFusedM));
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  half* op = reinterpret_cast<half*>(out.data_ptr());
  if (aligned16(xp) && x.stride(0) % 8 == 0) {
    cptc_prefill_fused<true><<<grid, kThreads, 0, stream>>>(
        xp, x.stride(0), w, op, out.stride(0), static_cast<int>(M),
        static_cast<int>(R), static_cast<int>(K));
  } else {
    cptc_prefill_fused<false><<<grid, kThreads, 0, stream>>>(
        xp, x.stride(0), w, op, out.stride(0), static_cast<int>(M),
        static_cast<int>(R), static_cast<int>(K));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

torch::Tensor rdna_hint_flex_cptc_dequant_rdna2(
    torch::Tensor codes, torch::Tensor code_off, torch::Tensor scale,
    torch::Tensor param_off, torch::Tensor row_nbits, torch::Tensor row_block,
    torch::Tensor coefficients, torch::Tensor tables, int64_t K) {
  auto w = check_weight(codes, code_off, scale, param_off, row_nbits, row_block,
                        coefficients, tables);
  TORCH_CHECK(K > 0 && K % 64 == 0 && K <= INT_MAX,
              "hi2 CPTC: K must be 64-aligned");
  const c10::cuda::CUDAGuard guard(scale.device());
  auto out = torch::empty({row_nbits.numel(), K}, scale.options());
  launch_dequant(w, out, at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor rdna_hint_flex_cptc_linear_rdna2(
    torch::Tensor x, torch::Tensor col_scale, torch::Tensor had,
    torch::Tensor codes, torch::Tensor code_off, torch::Tensor scale,
    torch::Tensor param_off, torch::Tensor row_nbits, torch::Tensor row_block,
    torch::Tensor coefficients, torch::Tensor tables, bool transform,
    int64_t decode_max_m, bool silu_mul, int64_t prefill_gemm) {
  auto w = check_weight(codes, code_off, scale, param_off, row_nbits, row_block,
                        coefficients, tables);
  const auto M = x.size(0);
  const auto R = row_nbits.numel();
  const auto K = silu_mul ? x.size(1) / 2 : x.size(1);
  TORCH_CHECK(x.is_cuda() && x.device() == scale.device() &&
                  x.scalar_type() == torch::kHalf && x.dim() == 2 &&
                  x.stride(1) == 1 && K > 0 && K % 64 == 0 && K <= INT_MAX &&
                  M <= INT_MAX,
              "hi2 CPTC: expected row-major fp16 x[M,K] on the weight device");
  TORCH_CHECK(!silu_mul || (transform && x.size(1) == 2 * K),
              "hi2 CPTC: silu_mul requires transformed x[M,2K]");
  TORCH_CHECK(decode_max_m >= 0 && prefill_gemm >= 0 && prefill_gemm <= 5,
              "hi2 CPTC: invalid decode/prefill dispatch mode");
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({M, R}, x.options());
  if (M == 0 || R == 0) return out;
  auto xt = x;
  if (transform) {
    xt = torch::empty({M, K}, x.options());
    rdna_hint_flex_act_transform_rdna2(x, col_scale, had, xt, false, silu_mul);
  }
  if (M <= decode_max_m) {
    launch_gemv(xt, w, out, stream);
    return out;
  }
  if (prefill_gemm == 5) {
    // Explicit cptc-fused: the real tiled decode-in-GEMM kernel. It never
    // falls back to the dense scratch + rocBLAS path.
    launch_prefill_fused(xt, w, out, R, stream);
    return out;
  }
  auto wd = torch::empty({R, K}, x.options());
  launch_dequant(w, wd, stream);
  namespace hg = vllm::rdna_hint_flex::gemm;
  int64_t main_rows = M;
  int design = hg::kRocblas;
  hg::TailKind tail = hg::TailKind::kNone;
  if (prefill_gemm > 0) {
    design = hg::pick(M, R, K);
    const int64_t remainder = M % hg::kTileRows;
    const auto want = hg::tail_kind(prefill_gemm, M, remainder, decode_max_m);
    if (want != hg::TailKind::kNone) {
      const int main_design = hg::pick(M - remainder, R, K);
      if (main_design >= 0) {
        design = main_design;
        main_rows = M - remainder;
        tail = want;
      }
    }
  }
  if (design < 0 || !hg::launch(design, xt.data_ptr(), wd.data_ptr(),
                                out.data_ptr(), main_rows, R, K, xt.stride(0),
                                wd.stride(0), out.stride(0), stream)) {
    at::mm_out(out, xt, wd.t());
    return out;
  }
  if (main_rows < M) {
    auto xr = xt.narrow(0, main_rows, M - main_rows);
    auto yr = out.narrow(0, main_rows, M - main_rows);
    if (tail == hg::TailKind::kDense) {
      if (!hg::launch(hg::tail_design(M - main_rows), xr.data_ptr(),
                      wd.data_ptr(), yr.data_ptr(), M - main_rows, R, K,
                      xr.stride(0), wd.stride(0), yr.stride(0), stream))
        at::mm_out(yr, xr, wd.t());
    } else {
      launch_gemv(xr, w, yr, stream);
    }
  }
  return out;
}
