// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// HI2 v3.3 CPTC: block-local tail-biting streams, fp16 table and block scales.
// The v3.2 ladder kernels and dispatch remain unchanged.

#include <climits>
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
__device__ __forceinline__ void row_dot(const half* x, int64_t ldx, int K,
                                        int m0, int M, const uint32_t* words,
                                        const half* scales, int block, int lane,
                                        uint32_t a0, uint32_t a1,
                                        const half* table, float (&acc)[MT]) {
  constexpr int kCodes = 32 / NB;
  const int block_words = block * NB / 32;
  for (int wi = lane; wi < K / kCodes; wi += kWave) {
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
    row_dot<2, MT>(x, ldx, K, m0, M, w.codes + w.code_off[row],
                   w.scales + w.param_off[row], w.blocks[row], lane, a0, a1,
                   tables + slot * kTableSize, acc);
  } else {
    row_dot<4, MT>(x, ldx, K, m0, M, w.codes + w.code_off[row],
                   w.scales + w.param_off[row], w.blocks[row], lane, a0, a1,
                   tables + slot * kTableSize, acc);
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

void launch_gemv(const torch::Tensor& x, CptcWeight w, torch::Tensor& out,
                 hipStream_t stream) {
  const int M = x.size(0), R = out.size(1), K = x.size(1);
  const int mt = M == 1 ? 1 : (M <= 4 ? 4 : 8);
  const dim3 grid((R + kRows - 1) / kRows, (M + mt - 1) / mt);
#define CPTC_GEMV(MT)                                              \
  cptc_gemv<MT><<<grid, kThreads, 0, stream>>>(                    \
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), w, \
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0), M, R, K)
  if (mt == 1) {
    CPTC_GEMV(1);
  } else if (mt == 4) {
    CPTC_GEMV(4);
  } else {
    CPTC_GEMV(8);
  }
#undef CPTC_GEMV
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
  TORCH_CHECK(decode_max_m >= 0 && prefill_gemm >= 0 && prefill_gemm <= 4,
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
