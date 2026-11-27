// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native W8A8 (INT8 weights x dynamic INT8 activations, v_dot4_i32_i8) dense
// prefill GEMM for AMD RDNA2 (gfx1030), fp16 I/O.
//
//   C[M, N] (fp16) = A[M, K] (fp16) @ W[N, K]^T (int8 * w_scale[N]) + bias[N]
//
// Designed for models with per-output-channel symmetric int8 dense weights
// (VLLM_RDNA_DENSE_INT8_ONLY=1), replacing on-the-fly fp16 dequant + rocBLAS
// fp16 GEMM (v_dot2_f32_f16, ~20.6 TMAC/s ceiling) with native v_dot4_i32_i8
// (~39.2 TMAC/s ceiling, zero group flushes in K, single epilogue scale).
//
// Architecture follows csrc/rocm/rdna_hint_flex_dense_gemm_kernel.cuh:
// Wave32 register-tiled BMxBNxBK (BK=64 int8 = 64 B/LDS row), raw buffer
// 16-byte global loads (BUF=1), XOR-swizzled conflict-free LDS (SWZ=1),
// double-buffered LDS stages (NBUF=2), rotated B dwords in LDS (BROT=1) so
// v_dot4 sources sit in distinct VGPR banks, and tail wave skip (SKIP=1).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ops.h"

#if defined(__HIPCC__) && \
    (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__))
  #define __HIP__RDNA2_W8A8__
#endif

namespace vllm {
namespace gemm_w8a8_i8_rdna2 {

template <int BM_, int BN_, int BK_, int TM_, int TN_, int RAST_>
struct DesignI8 {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int RAST = RAST_;
  static constexpr int NBUF = 2;
  static constexpr int LW = 8;   // 8-byte LDS fragment (ds_read_b64, 2 dwords)
  static constexpr int ND = 2;   // dwords per fragment
  static constexpr int KS = 8;   // int8 elements per k step
  static constexpr int CH = 16;  // int8 elements per 16-byte global chunk
  static constexpr int CPR = BK / CH;  // chunks per tile row (4 for BK=64)
  static constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;
  static constexpr int A_CHUNKS = BM * CPR, B_CHUNKS = BN * CPR;
  static constexpr int A_PER_T = (A_CHUNKS + T - 1) / T;
  static constexpr int B_PER_T = (B_CHUNKS + T - 1) / T;
  static constexpr int A_LDS = BM * BK, B_LDS = BN * BK;  // bytes
  static constexpr int STAGE = A_LDS + B_LDS;             // bytes per buffer
  static constexpr int SRP = CPR <= 8 ? 8 / CPR : 1;      // rows/swizzle step
  static constexpr int LDS_BYTES = NBUF * STAGE;

  static_assert(BK == 64 && CPR == 4, "BK=64 int8 (64 B/row, 4 chunks)");
  static_assert(TX % 8 == 0 && TY % 8 == 0, "SWZ requires TX, TY % 8 == 0");
  static_assert(BM % TM == 0 && BN % TN == 0, "tile divisibility");
  static_assert(T >= 32 && T <= 1024 && T % 32 == 0, "block size");
  static_assert(LDS_BYTES <= 65536, "LDS <= 64 KiB");
};

using D1_I8 = DesignI8<128, 128, 64, 8, 16, 8>;  // 128 thr, 32 KiB LDS
using D4_I8 = DesignI8<256, 128, 64, 8, 16, 4>;  // 256 thr, 48 KiB LDS
using D6_I8 = DesignI8<64, 128, 64, 8, 8, 0>;    // 128 thr, 24 KiB LDS
using D7_I8 = DesignI8<64, 64, 64, 8, 8, 4>;  //  64 thr, 16 KiB LDS (skinny N)

#if defined(__HIP__RDNA2_W8A8__) || !defined(__HIP_DEVICE_COMPILE__)

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using u32x2 = unsigned int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int32_t sdot4u(uint32_t a, uint32_t b, int32_t c) {
  return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), c,
                                /*clamp=*/false);
}

// ---------------------------------------------------------------------------
// Kernel 1: Dynamic per-token symmetric INT8 quantization (fp16 -> int8 + f32)
// 1 block (256 threads) per row m. K must be a multiple of 8 (guaranteed: %64).
// Fast single-pass path when K <= 4096 (<= 2 uint4 loads/thread kept in regs).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    w8a8_act_quant_per_token_k(const _Float16* __restrict__ x,
                               int8_t* __restrict__ a_i8,
                               float* __restrict__ a_scale, int M, int K) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tid = threadIdx.x;
  const int nvec = K >> 3;  // 8 fp16 = 16 bytes per vector
  const uint4* x_row = reinterpret_cast<const uint4*>(x + (int64_t)m * K);
  uint2* out_row = reinterpret_cast<uint2*>(a_i8 + (int64_t)m * K);

  float local_max = 0.0f;
  uint4 reg_v[2];
  const bool single_pass = (nvec <= 512);
  if (single_pass) {
    if (tid < nvec) {
      reg_v[0] = x_row[tid];
      const __half2* h2 = reinterpret_cast<const __half2*>(&reg_v[0]);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    if (tid + 256 < nvec) {
      reg_v[1] = x_row[tid + 256];
      const __half2* h2 = reinterpret_cast<const __half2*>(&reg_v[1]);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  } else {
    for (int i = tid; i < nvec; i += 256) {
      uint4 v = x_row[i];
      const __half2* h2 = reinterpret_cast<const __half2*>(&v);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  }

  // Wave32 reduction
  const int lane = tid & 31;
  const int wave = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor(local_max, off, 32));
  }

  __shared__ float smax[8];
  if (lane == 0) smax[wave] = local_max;
  __syncthreads();

  float row_max = (tid < 8) ? smax[tid] : 0.0f;
  if (wave == 0) {
  #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      row_max = fmaxf(row_max, __shfl_xor(row_max, off, 32));
    }
    if (lane == 0) smax[0] = row_max;
  }
  __syncthreads();

  row_max = smax[0];
  const float scale = (row_max > 0.0f) ? (row_max * (1.0f / 127.0f)) : 0.0f;
  const float inv_scale = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
  if (tid == 0) a_scale[m] = scale;

  auto quant_store = [&](int idx, const uint4& v) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&v);
    uint32_t p[2] = {0u, 0u};
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h2[j]);
      uint32_t q0 = static_cast<uint8_t>(static_cast<int8_t>(
          max(-127, min(127, __float2int_rn(f.x * inv_scale)))));
      uint32_t q1 = static_cast<uint8_t>(static_cast<int8_t>(
          max(-127, min(127, __float2int_rn(f.y * inv_scale)))));
      p[j >> 1] |= (q0 | (q1 << 8)) << ((j & 1) * 16);
    }
    out_row[idx] = make_uint2(p[0], p[1]);
  };

  if (single_pass) {
    if (tid < nvec) quant_store(tid, reg_v[0]);
    if (tid + 256 < nvec) quant_store(tid + 256, reg_v[1]);
  } else {
    for (int i = tid; i < nvec; i += 256) {
      quant_store(i, x_row[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: Register-tiled Wave32 W8A8 sdot4 GEMM
//   A: [M, K] int8 (row-major, 16-byte aligned)
//   W: [N, K] int8 (row-major, 16-byte aligned)
//   C: [M, N] fp16 (row-major)
// ---------------------------------------------------------------------------
template <class P>
__global__ void __launch_bounds__(P::T)
    gemm_w8a8_i8_kernel(const char* __restrict__ A, const char* __restrict__ W,
                        const float* __restrict__ a_scale,
                        const _Float16* __restrict__ w_scale,
                        const _Float16* __restrict__ bias,
                        _Float16* __restrict__ Cout, int M, int N, int K) {
  using Chunk = u32x4;  // 16-byte global chunk (16 int8)
  using Frag = u32x2;   // 8-byte LDS fragment (8 int8 = 2 dwords)
  constexpr int BM = P::BM, BN = P::BN, BK = P::BK, TM = P::TM, TN = P::TN;
  constexpr int TX = P::TX, TY = P::TY, T = P::T, CH = P::CH, CPR = P::CPR;
  constexpr int A_PER_T = P::A_PER_T, B_PER_T = P::B_PER_T;

  __shared__ __attribute__((aligned(16))) char lds[P::LDS_BYTES];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int swa = (ty / P::SRP) & (CPR - 1);
  const int swb = (tx / P::SRP) & (CPR - 1);

  auto store_off = [](int r, int kcc) {
    return r * BK + ((kcc ^ ((r / P::SRP) & (CPR - 1))) << 4);
  };
  auto frag_off = [](int kk, int sw) {
    return (((kk >> 4) ^ sw) << 4) + (kk & 15);
  };

  // Block rasterization (L2/Infinity-Cache weight reuse)
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

  // Tail wave skip
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
    aoff[v] = static_cast<uint32_t>(grow * K + kc);
  }
  #pragma unroll
  for (int v = 0; v < B_PER_T; ++v) {
    const int c = tid + v * T;
    const int row = min(c / CPR, BN - 1);
    const int kc = (c % CPR) * CH;
    const int grow = min(n0 + row, N - 1);
    boff[v] = static_cast<uint32_t>(grow * K + kc);
  }

  Chunk ra[A_PER_T], rb[B_PER_T];
  const __amdgpu_buffer_rsrc_t rsa = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(A), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(M) * static_cast<uint32_t>(K)),
      0x31014000);
  const __amdgpu_buffer_rsrc_t rsw = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(W), static_cast<short>(0),
      static_cast<int>(static_cast<uint32_t>(N) * static_cast<uint32_t>(K)),
      0x31014000);

  auto gload = [&](int kt) {
    const uint32_t kb = static_cast<uint32_t>(kt * BK);
  #pragma unroll
    for (int v = 0; v < A_PER_T; ++v)
      if (A_PER_T * T == P::A_CHUNKS || tid + v * T < P::A_CHUNKS)
        ra[v] = __builtin_amdgcn_raw_buffer_load_b128(rsa, aoff[v], kb, 0);
  #pragma unroll
    for (int v = 0; v < B_PER_T; ++v)
      if (B_PER_T * T == P::B_CHUNKS || tid + v * T < P::B_CHUNKS)
        rb[v] = __builtin_amdgcn_raw_buffer_load_b128(rsw, boff[v], kb, 0);
  };

  auto lstore = [&](int buf) {
    char* as = lds + buf * P::STAGE;
    char* bs = as + P::A_LDS;
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
        // BROT=1: rotate dwords within each 8-byte fragment so A and B sit in
        // distinct VGPR banks for v_dot4c_i32_i8.
        const Chunk q = rb[v];
        *reinterpret_cast<Chunk*>(bs + store_off(c / CPR, c % CPR)) =
            Chunk{q.y, q.x, q.w, q.z};
      }
    }
  };

  int32_t acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  auto compute = [&](int buf) {
    if (!wave_live) return;
    const char* as = lds + buf * P::STAGE;
    const char* bs = as + P::A_LDS;
  #pragma unroll 1
    for (int kk = 0; kk < BK; kk += P::KS) {
      Frag af[TM], bf[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const Frag*>(as + (ty + TY * i) * BK +
                                               frag_off(kk, swa));
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const Frag*>(bs + (tx + TX * j) * BK +
                                               frag_off(kk, swb));
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = sdot4u(af[i][0], bf[j][1], acc[i][j]);
          acc[i][j] = sdot4u(af[i][1], bf[j][0], acc[i][j]);
        }
    }
  };

  // Double-buffered K loop (1 barrier per tile)
  gload(0);
  lstore(0);
  __syncthreads();
  int buf = 0;
  for (int kt = 1; kt < nk; ++kt) {
    gload(kt);
    compute(buf);
    buf ^= 1;
    lstore(buf);
    __syncthreads();
  }
  compute(buf);

  // Epilogue: single float scale multiply + optional bias -> fp16
  if (!wave_live) return;
  float asc[TM], wsc[TN], bi[TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + TY * i;
    asc[i] = (row < M) ? a_scale[row] : 0.0f;
  }
  #pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int col = n0 + tx + TX * j;
    wsc[j] = (col < N) ? static_cast<float>(w_scale[col]) : 0.0f;
    bi[j] = (bias && col < N) ? static_cast<float>(bias[col]) : 0.0f;
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + TY * i;
    if (row >= M) continue;
    _Float16* out_r = Cout + (int64_t)row * N;
  #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + TX * j;
      if (col < N) {
        float val = static_cast<float>(acc[i][j]) * asc[i] * wsc[j] + bi[j];
        out_r[col] = static_cast<_Float16>(val);
      }
    }
  }
}

#else  // non-RDNA2 device pass: signature parity, empty body.

__global__ void __launch_bounds__(256)
    w8a8_act_quant_per_token_k(const _Float16*, int8_t*, float*, int, int) {}

template <class P>
__global__ void __launch_bounds__(P::T)
    gemm_w8a8_i8_kernel(const char*, const char*, const float*, const _Float16*,
                        const _Float16*, _Float16*, int, int, int) {}

#endif  // __HIP__RDNA2_W8A8__ || !__HIP_DEVICE_COMPILE__

namespace {

struct Scratch {
  torch::Tensor a_i8;
  torch::Tensor a_scale;
  int64_t cap_elems = 0;
  int64_t cap_rows = 0;
};

std::mutex g_scratch_mu;
std::map<int, Scratch> g_scratch;

void get_scratch(int dev, int64_t M, int64_t K, torch::Tensor& a_i8,
                 torch::Tensor& a_scale, const torch::Device& device) {
  std::lock_guard<std::mutex> lk(g_scratch_mu);
  Scratch& s = g_scratch[dev];
  const int64_t need_elems = M * K;
  if (s.cap_elems < need_elems) {
    s.a_i8 =
        torch::empty({need_elems},
                     torch::TensorOptions().dtype(torch::kInt8).device(device));
    s.cap_elems = need_elems;
  }
  if (s.cap_rows < M) {
    s.a_scale = torch::empty(
        {M}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    s.cap_rows = M;
  }
  a_i8 = s.a_i8;
  a_scale = s.a_scale;
}

template <class P>
void launch_i8(const int8_t* A, const int8_t* W, const float* a_scale,
               const _Float16* w_scale, const _Float16* bias, _Float16* C,
               int64_t M, int64_t N, int64_t K, hipStream_t stream) {
  const int64_t ntm = (M + P::BM - 1) / P::BM;
  const int64_t ntn = (N + P::BN - 1) / P::BN;
  gemm_w8a8_i8_kernel<P>
      <<<dim3(static_cast<unsigned>(ntm * ntn)), dim3(P::T), 0, stream>>>(
          reinterpret_cast<const char*>(A), reinterpret_cast<const char*>(W),
          a_scale, w_scale, bias, C, static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

}  // namespace gemm_w8a8_i8_rdna2
}  // namespace vllm

at::Tensor gemm_w8a8_i8_rdna2(const at::Tensor& x, const at::Tensor& w_i8,
                              const at::Tensor& w_scale,
                              const std::optional<at::Tensor>& bias,
                              int64_t design) {
  TORCH_CHECK(x.is_cuda() && w_i8.is_cuda() && w_scale.is_cuda(),
              "gemm_w8a8_i8_rdna2: tensors must be on CUDA/HIP");
  TORCH_CHECK(x.dtype() == torch::kFloat16,
              "gemm_w8a8_i8_rdna2: x must be fp16");
  TORCH_CHECK(w_i8.dtype() == torch::kInt8,
              "gemm_w8a8_i8_rdna2: w_i8 must be int8");
  TORCH_CHECK(w_scale.dtype() == torch::kFloat16,
              "gemm_w8a8_i8_rdna2: w_scale must be fp16");
  TORCH_CHECK(x.dim() == 2 && w_i8.dim() == 2,
              "gemm_w8a8_i8_rdna2: 2D tensors expected");

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = w_i8.size(0);
  TORCH_CHECK(w_i8.size(1) == K && w_scale.numel() == N,
              "gemm_w8a8_i8_rdna2: shape mismatch");
  TORCH_CHECK(K % 64 == 0, "gemm_w8a8_i8_rdna2: K must be a multiple of 64");
  TORCH_CHECK(M * K <= INT32_MAX && N * K <= INT32_MAX,
              "gemm_w8a8_i8_rdna2: tensor element count exceeds int32 buffer");

  const at::Tensor x_c = x.contiguous();
  const at::Tensor w_c = w_i8.contiguous();
  const at::Tensor s_c = w_scale.contiguous();
  const _Float16* bias_ptr = nullptr;
  at::Tensor b_c;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->dtype() == torch::kFloat16 && bias->numel() == N,
                "gemm_w8a8_i8_rdna2: bias must be fp16 [N]");
    b_c = bias->contiguous();
    bias_ptr = reinterpret_cast<const _Float16*>(b_c.data_ptr());
  }

  c10::cuda::CUDAGuard device_guard(x_c.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor out = at::empty({M, N}, x_c.options());
  if (M == 0 || N == 0 || K == 0) return out;

  torch::Tensor a_i8, a_scale;
  vllm::gemm_w8a8_i8_rdna2::get_scratch(x_c.device().index(), M, K, a_i8,
                                        a_scale, x_c.device());

  vllm::gemm_w8a8_i8_rdna2::w8a8_act_quant_per_token_k<<<
      dim3(static_cast<unsigned>(M)), dim3(256), 0, stream>>>(
      reinterpret_cast<const _Float16*>(x_c.data_ptr()),
      a_i8.data_ptr<int8_t>(), a_scale.data_ptr<float>(), static_cast<int>(M),
      static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int pick = static_cast<int>(design);
  if (pick < 0) {
    if (N <= 384 && M >= 512)
      pick = 3;  // D7_I8 (64x64x64, 64 thr, 192 blks on 72 CUs for hc_down)
    else if (M <= 256 || N <= 384)
      pick = 2;  // D6_I8 (64x128x64)
    else
      pick = 0;  // D1_I8 (128x128x64, 32 KiB LDS, 4 blocks/WGP)
  }

  const int8_t* a_ptr = a_i8.data_ptr<int8_t>();
  const int8_t* w_ptr = w_c.data_ptr<int8_t>();
  const float* asc_ptr = a_scale.data_ptr<float>();
  const _Float16* wsc_ptr = reinterpret_cast<const _Float16*>(s_c.data_ptr());
  _Float16* out_ptr = reinterpret_cast<_Float16*>(out.data_ptr());

  if (pick == 1) {
    vllm::gemm_w8a8_i8_rdna2::launch_i8<vllm::gemm_w8a8_i8_rdna2::D4_I8>(
        a_ptr, w_ptr, asc_ptr, wsc_ptr, bias_ptr, out_ptr, M, N, K, stream);
  } else if (pick == 2) {
    vllm::gemm_w8a8_i8_rdna2::launch_i8<vllm::gemm_w8a8_i8_rdna2::D6_I8>(
        a_ptr, w_ptr, asc_ptr, wsc_ptr, bias_ptr, out_ptr, M, N, K, stream);
  } else if (pick == 3) {
    vllm::gemm_w8a8_i8_rdna2::launch_i8<vllm::gemm_w8a8_i8_rdna2::D7_I8>(
        a_ptr, w_ptr, asc_ptr, wsc_ptr, bias_ptr, out_ptr, M, N, K, stream);
  } else {
    vllm::gemm_w8a8_i8_rdna2::launch_i8<vllm::gemm_w8a8_i8_rdna2::D1_I8>(
        a_ptr, w_ptr, asc_ptr, wsc_ptr, bias_ptr, out_ptr, M, N, K, stream);
  }
  return out;
}

// ===========================================================================
// Phase 1.2.3: Native INT8 v_dot4 HC up-projection + sigmoid + gated mean
//   lora[m, r] = silu(dai[m, r] / 4) -> quantize to int8 + f32 scale
//   out[m, h]  = (1/4) * sum_{c=0..3} sigmoid(lora[m] . w_up[c*H+h]) * xn[m,
//   c*H+h]
// ===========================================================================
namespace vllm {
namespace gemm_w8a8_i8_rdna2 {

#if defined(__HIP__RDNA2_W8A8__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ float hc_sigmoid_f(float x) {
  return __builtin_amdgcn_rcpf(1.0f + __expf(-x));
}

// Fused silu(dai / HC) + per-token symmetric int8 quantization.
// 1 block of 128 threads per row m. R must be a multiple of 64.
// Fast single-pass path when R <= 1024 (<= 1 uint4/thread kept in regs).
__global__ void __launch_bounds__(128)
    hc_silu_quant_k(const _Float16* __restrict__ dai, int ld_dai,
                    int8_t* __restrict__ lora_i8,
                    float* __restrict__ lora_scale, int M, int R) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tid = threadIdx.x;
  const int nvec = R >> 3;  // 8 fp16 = 16 bytes
  const _Float16* row_in = dai + (int64_t)m * ld_dai;
  uint2* row_out = reinterpret_cast<uint2*>(lora_i8 + (int64_t)m * R);

  float local_max = 0.0f;
  float reg_y[8];
  const bool single_pass = (nvec <= 128);
  if (single_pass) {
    if (tid < nvec) {
      uint4 v = reinterpret_cast<const uint4*>(row_in)[tid];
      const __half2* h2 = reinterpret_cast<const __half2*>(&v);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        float x0 = f.x * 0.25f;
        float x1 = f.y * 0.25f;
        float y0 = x0 * hc_sigmoid_f(x0);
        float y1 = x1 * hc_sigmoid_f(x1);
        reg_y[2 * j + 0] = y0;
        reg_y[2 * j + 1] = y1;
        local_max = fmaxf(local_max, fmaxf(fabsf(y0), fabsf(y1)));
      }
    }
  } else {
    for (int i = tid; i < nvec; i += 128) {
      uint4 v = reinterpret_cast<const uint4*>(row_in)[i];
      const __half2* h2 = reinterpret_cast<const __half2*>(&v);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        float x0 = f.x * 0.25f;
        float x1 = f.y * 0.25f;
        float y0 = x0 * hc_sigmoid_f(x0);
        float y1 = x1 * hc_sigmoid_f(x1);
        local_max = fmaxf(local_max, fmaxf(fabsf(y0), fabsf(y1)));
      }
    }
  }

  const int lane = tid & 31;
  const int wave = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local_max = fmaxf(local_max, __shfl_xor(local_max, off, 32));
  __shared__ float smax[4];
  if (lane == 0) smax[wave] = local_max;
  __syncthreads();
  float row_max = (tid < 4) ? smax[tid] : 0.0f;
  if (wave == 0) {
    row_max = fmaxf(row_max, __shfl_xor(row_max, 2, 32));
    row_max = fmaxf(row_max, __shfl_xor(row_max, 1, 32));
    if (lane == 0) smax[0] = row_max;
  }
  __syncthreads();

  row_max = smax[0];
  const float scale = (row_max > 0.0f) ? (row_max * (1.0f / 127.0f)) : 0.0f;
  const float inv_scale = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
  if (tid == 0) lora_scale[m] = scale;

  if (single_pass) {
    if (tid < nvec) {
      uint32_t p[2] = {0u, 0u};
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t q0 = static_cast<uint8_t>(static_cast<int8_t>(
            max(-127, min(127, __float2int_rn(reg_y[2 * j + 0] * inv_scale)))));
        uint32_t q1 = static_cast<uint8_t>(static_cast<int8_t>(
            max(-127, min(127, __float2int_rn(reg_y[2 * j + 1] * inv_scale)))));
        p[j >> 1] |= (q0 | (q1 << 8)) << ((j & 1) * 16);
      }
      row_out[tid] = make_uint2(p[0], p[1]);
    }
  } else {
    for (int i = tid; i < nvec; i += 128) {
      uint4 v = reinterpret_cast<const uint4*>(row_in)[i];
      const __half2* h2 = reinterpret_cast<const __half2*>(&v);
      uint32_t p[2] = {0u, 0u};
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h2[j]);
        float x0 = f.x * 0.25f;
        float x1 = f.y * 0.25f;
        float y0 = x0 * hc_sigmoid_f(x0);
        float y1 = x1 * hc_sigmoid_f(x1);
        uint32_t q0 = static_cast<uint8_t>(static_cast<int8_t>(
            max(-127, min(127, __float2int_rn(y0 * inv_scale)))));
        uint32_t q1 = static_cast<uint8_t>(static_cast<int8_t>(
            max(-127, min(127, __float2int_rn(y1 * inv_scale)))));
        p[j >> 1] |= (q0 | (q1 << 8)) << ((j & 1) * 16);
      }
      row_out[i] = make_uint2(p[0], p[1]);
    }
  }
}

// Fused INT8 v_dot4 HC up-GEMM + sigmoid + gated mean.
// TY=16, TM=8 -> BM=128; TX=16, HH=2 -> BH=32, BN=128; kKT=64 int8 (16 dwords).
__global__ void __launch_bounds__(256) hc_up_mix_prefill_i8_k(
    const int8_t* __restrict__ lora_i8, const float* __restrict__ lora_scale,
    const int8_t* __restrict__ w_up_i8, const _Float16* __restrict__ w_up_scale,
    const half* __restrict__ xn, half* __restrict__ out, int M, int H, int R) {
  constexpr int kHC = 4;
  constexpr int kThreads = 256;
  constexpr int TY = 16, TM = 8, HH = 2;
  constexpr int BM = TY * TM;   // 128
  constexpr int BH = 16 * HH;   // 32
  constexpr int BN = kHC * BH;  // 128
  constexpr int kKT = 64;       // 64 int8 elements per slab
  constexpr int kKP = kKT / 4;  // 16 dwords per slab
  constexpr int SA = BM + 4;    // 132 dwords (conflict-free b128)
  constexpr int SB = BN + 2;    // 130 dwords (conflict-free)
  constexpr int A_LD = (BM * (kKT / 16)) / kThreads;  // 2 uint4 loads/thread
  constexpr int B_LD = (BN * (kKT / 16)) / kThreads;  // 2 uint4 loads/thread

  __shared__ uint32_t As[kKP][SA];
  __shared__ uint32_t Bs[kKP][SB];

  const int t = threadIdx.x;
  const int tx = t & 15, ty = t >> 4;
  const int h0 = blockIdx.x * BH;
  const int m0 = blockIdx.y * BM;
  const int HCH = kHC * H;

  const int8_t* a_src[A_LD];
  bool a_ok[A_LD];
  int a_m[A_LD], a_j[A_LD];
  #pragma unroll
  for (int i = 0; i < A_LD; ++i) {
    const int idx = t + kThreads * i;
    a_m[i] = idx >> 2;
    a_j[i] = idx & 3;
    const int row = m0 + a_m[i];
    a_ok[i] = row < M;
    a_src[i] = lora_i8 + (size_t)(a_ok[i] ? row : 0) * R + a_j[i] * 16;
  }
  const int8_t* b_src[B_LD];
  int b_n[B_LD], b_j[B_LD];
  #pragma unroll
  for (int i = 0; i < B_LD; ++i) {
    const int idx = t + kThreads * i;
    b_n[i] = idx >> 2;
    b_j[i] = idx & 3;
    const int c = b_n[i] / BH, hh = b_n[i] % BH;
    b_src[i] = w_up_i8 + (size_t)(c * H + h0 + hh) * R + b_j[i] * 16;
  }

  uint4 ra[A_LD], rb[B_LD];
  auto fetch = [&](int k0) {
  #pragma unroll
    for (int i = 0; i < A_LD; ++i)
      ra[i] = a_ok[i] ? *reinterpret_cast<const uint4*>(a_src[i] + k0)
                      : make_uint4(0, 0, 0, 0);
  #pragma unroll
    for (int i = 0; i < B_LD; ++i)
      rb[i] = *reinterpret_cast<const uint4*>(b_src[i] + k0);
  };

  auto stage = [&]() {
  #pragma unroll
    for (int i = 0; i < A_LD; ++i) {
      const uint32_t* av = reinterpret_cast<const uint32_t*>(&ra[i]);
  #pragma unroll
      for (int q = 0; q < 4; ++q) As[a_j[i] * 4 + q][a_m[i]] = av[q];
    }
  #pragma unroll
    for (int i = 0; i < B_LD; ++i) {
      const uint32_t* bv = reinterpret_cast<const uint32_t*>(&rb[i]);
  #pragma unroll
      for (int q = 0; q < 4; ++q) Bs[b_j[i] * 4 + q][b_n[i]] = bv[q];
    }
  };

  int32_t acc[TM][kHC][HH];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int c = 0; c < kHC; ++c)
  #pragma unroll
      for (int e = 0; e < HH; ++e) acc[i][c][e] = 0;

  fetch(0);
  for (int k0 = 0; k0 < R; k0 += kKT) {
    __syncthreads();
    stage();
    __syncthreads();
    if (k0 + kKT < R) fetch(k0 + kKT);
  #pragma unroll
    for (int kp = 0; kp < kKP; ++kp) {
      uint32_t a[TM];
  #pragma unroll
      for (int i = 0; i < TM; i += 4)
        *reinterpret_cast<uint4*>(&a[i]) =
            *reinterpret_cast<const uint4*>(&As[kp][ty * TM + i]);
      uint32_t b[kHC][HH];
  #pragma unroll
      for (int c = 0; c < kHC; ++c)
        *reinterpret_cast<uint2*>(&b[c][0]) =
            *reinterpret_cast<const uint2*>(&Bs[kp][c * BH + tx * 2]);
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int c = 0; c < kHC; ++c)
  #pragma unroll
          for (int e = 0; e < HH; ++e)
            acc[i][c][e] = sdot4u(a[i], b[c][e], acc[i][c][e]);
    }
  }

  // Epilogue: dequant -> sigmoid -> gated mean over 4 HC streams
  const int h = h0 + tx * HH;
  float wsc[kHC][HH];
  #pragma unroll
  for (int c = 0; c < kHC; ++c) {
    wsc[c][0] = static_cast<float>(w_up_scale[c * H + h + 0]);
    wsc[c][1] = static_cast<float>(w_up_scale[c * H + h + 1]);
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty * TM + i;
    if (row >= M) continue;
    const float asc = lora_scale[row];
    const half* xr = xn + (size_t)row * HCH + h;
    float mix0 = 0.0f, mix1 = 0.0f;
  #pragma unroll
    for (int c = 0; c < kHC; ++c) {
      const float2 xv =
          __half22float2(*reinterpret_cast<const half2*>(xr + c * H));
      const float g0 = static_cast<float>(acc[i][c][0]) * asc * wsc[c][0];
      const float g1 = static_cast<float>(acc[i][c][1]) * asc * wsc[c][1];
      mix0 += hc_sigmoid_f(g0) * xv.x;
      mix1 += hc_sigmoid_f(g1) * xv.y;
    }
    *reinterpret_cast<half2*>(out + (size_t)row * H + h) =
        __floats2half2_rn(mix0 * 0.25f, mix1 * 0.25f);
  }
}

#else  // non-RDNA2 device pass: signature parity, empty body.

__global__ void __launch_bounds__(128)
    hc_silu_quant_k(const _Float16*, int, int8_t*, float*, int, int) {}

__global__ void __launch_bounds__(256)
    hc_up_mix_prefill_i8_k(const int8_t*, const float*, const int8_t*,
                           const _Float16*, const half*, half*, int, int, int) {
}

#endif  // __HIP__RDNA2_W8A8__ || !__HIP_DEVICE_COMPILE__

}  // namespace gemm_w8a8_i8_rdna2
}  // namespace vllm

at::Tensor rdna_hc_up_gate_mix_prefill_i8(const at::Tensor& dai,
                                          const at::Tensor& w_up_i8,
                                          const at::Tensor& w_up_scale,
                                          const at::Tensor& xn,
                                          int64_t lora_rank, int64_t hc_count) {
  TORCH_CHECK(hc_count == 4, "rdna_hc_up_gate_mix_prefill_i8: hc_count == 4");
  TORCH_CHECK(dai.dtype() == torch::kFloat16 && xn.dtype() == torch::kFloat16,
              "rdna_hc_up_gate_mix_prefill_i8: dai and xn must be fp16");
  TORCH_CHECK(
      w_up_i8.dtype() == torch::kInt8 && w_up_scale.dtype() == torch::kFloat16,
      "rdna_hc_up_gate_mix_prefill_i8: w_up_i8 int8, scale fp16");
  const int64_t M = dai.size(0);
  const int64_t ld_dai = dai.stride(0);
  const int64_t R = lora_rank;
  const int64_t H = xn.size(1) / hc_count;
  TORCH_CHECK(R % 64 == 0 && H % 32 == 0,
              "rdna_hc_up_gate_mix_prefill_i8: R%64==0, H%32==0");
  TORCH_CHECK(dai.dim() == 2 && dai.stride(1) == 1 && ld_dai % 8 == 0,
              "rdna_hc_up_gate_mix_prefill_i8: dai must be contiguous with "
              "stride(0)%8==0");
  TORCH_CHECK(xn.dim() == 2 && xn.is_contiguous() && w_up_i8.is_contiguous() &&
                  w_up_scale.is_contiguous(),
              "rdna_hc_up_gate_mix_prefill_i8: xn, w_up_i8, w_up_scale must be "
              "contiguous");

  using namespace vllm::gemm_w8a8_i8_rdna2;
  c10::cuda::CUDAGuard device_guard(dai.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor out = at::empty({M, H}, xn.options());
  if (M == 0) return out;

  torch::Tensor lora_i8, lora_scale;
  get_scratch(dai.device().index(), M, R, lora_i8, lora_scale, dai.device());

  hc_silu_quant_k<<<dim3(static_cast<unsigned>(M)), dim3(128), 0, stream>>>(
      reinterpret_cast<const _Float16*>(dai.data_ptr()),
      static_cast<int>(ld_dai), lora_i8.data_ptr<int8_t>(),
      lora_scale.data_ptr<float>(), static_cast<int>(M), static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 grid(static_cast<unsigned>(H / 32),
                  static_cast<unsigned>((M + 127) / 128));
  hc_up_mix_prefill_i8_k<<<grid, dim3(256), 0, stream>>>(
      lora_i8.data_ptr<int8_t>(), lora_scale.data_ptr<float>(),
      w_up_i8.data_ptr<int8_t>(),
      reinterpret_cast<const _Float16*>(w_up_scale.data_ptr()),
      reinterpret_cast<const half*>(xn.data_ptr()),
      reinterpret_cast<half*>(out.data_ptr()), static_cast<int>(M),
      static_cast<int>(H), static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
