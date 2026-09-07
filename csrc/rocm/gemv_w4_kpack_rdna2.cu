// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// M=1 W4A16 dequant-GEMV for RDNA2 (gfx1030), hand-written HIP.
// Adapted from sebastianmechno-sys/vllm-rocm-windows-rdna2 (Apache-2.0),
// kernels/src/gemv_w4.cu — K-packed layout variant:
//   wq [K//8, N] uint32 packed along K (value for global row k of column n
//   is (wq[k/8, n] >> (k%8)*4) & 0xF); s [K//G, N] half;
//   z [K//G, N] uint8 (asymmetric) or constant bias (symmetric).
//   c[n] = sum_k a[k] * (q(k,n) - z(g,n)) * s(g,n).
//
// Bandwidth strategy (per the original author's research):
//   lane -> N, each thread owns COLS=4 consecutive columns = one 16-byte
//   global dwordx4 per k-row (32 lanes x 16B = 512B coalesced);
//   __builtin_nontemporal_load (glc/dlc streaming); UNROLL k-row loads in
//   flight to cover DRAM latency; split-K via gridDim.y + atomicAdd fp32
//   partials; direct fp16 store when gridDim.y == 1.

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/hip/HIPStream.h>

namespace {
using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
constexpr int COLS = 4;

__device__ __forceinline__ u32x4
nt_load(const u32x4* p, bool full) {
  if (full) {
#if defined(__HIP_PLATFORM_AMD__) && __has_builtin(__builtin_nontemporal_load)
    return __builtin_nontemporal_load(p);
#else
    return *p;
#endif
  }
  return *p;
}
}  // namespace

__global__ void __launch_bounds__(256) gemv_w4_kpack_splitk(
    const __half* __restrict__ a, const unsigned int* __restrict__ wq,
    const __half* __restrict__ s, const unsigned char* __restrict__ z,
    float* __restrict__ c_acc, int K, int N, int Gsz, int split_groups) {
  const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * COLS;
  if (n0 >= N) return;
  const int num_groups = K / Gsz;
  const int g0 = blockIdx.y * split_groups;
  int g1 = g0 + split_groups;
  if (g1 > num_groups) g1 = num_groups;
  const int ROWS = Gsz / 8;  // int32 k-rows per group
  const bool full = (n0 + COLS <= N);

  float acc[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) acc[c] = 0.f;

  for (int g = g0; g < g1; ++g) {
    const int k0 = g * Gsz;
    float sc[COLS], zr[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      int n = n0 + c;
      if (n >= N) n = N - 1;
      sc[c] = __half2float(s[(size_t)g * N + n]);
      zr[c] = (float)z[(size_t)g * N + n];
    }
    float gacc[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) gacc[c] = 0.f;
    float asum = 0.f;

#pragma unroll 4
    for (int rb = 0; rb < ROWS; rb += 4) {
      u32x4 buf[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        int r = rb + u;
        const u32x4* wptr =
            (const u32x4*)(wq + (size_t)((k0 >> 3) + r) * N + n0);
        buf[u] = nt_load(wptr, full);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        int r = rb + u;
        const __half* arow = a + k0 + r * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float av = __half2float(arow[j]);
          asum += av;
#pragma unroll
          for (int c = 0; c < COLS; ++c)
            gacc[c] += av * (float)((buf[u][c] >> (j * 4)) & 0xF);
        }
      }
    }
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[c] += (gacc[c] - zr[c] * asum) * sc[c];
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c)
    if (n0 + c < N) atomicAdd(&c_acc[n0 + c], acc[c]);
}

template <int U>
__global__ void __launch_bounds__(256) gemv_w4_kpack_sym_splitk(
    const __half* __restrict__ a, const unsigned int* __restrict__ wq,
    const __half* __restrict__ s, float* __restrict__ c_acc,
    __half* __restrict__ c_out, int direct, int K, int N, int Gsz,
    int split_groups, const float bias) {
  const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * COLS;
  if (n0 >= N) return;
  const int num_groups = K / Gsz;
  const int g0 = blockIdx.y * split_groups;
  int g1 = g0 + split_groups;
  if (g1 > num_groups) g1 = num_groups;
  const int ROWS = Gsz / 8;
  const bool full = (n0 + COLS <= N);

  float acc[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) acc[c] = 0.f;

  for (int g = g0; g < g1; ++g) {
    const int k0 = g * Gsz;
    float sc[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      int n = n0 + c;
      if (n >= N) n = N - 1;
      sc[c] = __half2float(s[(size_t)g * N + n]);
    }
    float gacc[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) gacc[c] = 0.f;
    float asum = 0.f;

    for (int rb = 0; rb < ROWS; rb += U) {
      u32x4 buf[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        int r = rb + u;
        const u32x4* wptr =
            (const u32x4*)(wq + (size_t)((k0 >> 3) + r) * N + n0);
        buf[u] = nt_load(wptr, full);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        int r = rb + u;
        const __half* arow = a + k0 + r * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float av = __half2float(arow[j]);
          asum += av;
#pragma unroll
          for (int c = 0; c < COLS; ++c)
            gacc[c] += av * (float)((buf[u][c] >> (j * 4)) & 0xF);
        }
      }
    }
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[c] += (gacc[c] - bias * asum) * sc[c];
  }

  if (direct) {
#pragma unroll
    for (int c = 0; c < COLS; ++c)
      if (n0 + c < N) c_out[n0 + c] = (__half)acc[c];
  } else {
#pragma unroll
    for (int c = 0; c < COLS; ++c)
      if (n0 + c < N) atomicAdd(&c_acc[n0 + c], acc[c]);
  }
}

torch::Tensor gemv_w4_kpack(torch::Tensor a, torch::Tensor wq, torch::Tensor s,
                            torch::Tensor z, int64_t group_size,
                            int64_t split_groups) {
  const int K = a.size(-1);
  const int N = s.size(1);
  auto c_acc = torch::zeros(
      {N}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
  const int num_groups = K / (int)group_size;
  const int BT = 256;
  dim3 block(BT);
  dim3 grid((N + BT * COLS - 1) / (BT * COLS),
            (num_groups + split_groups - 1) / split_groups);
  auto stream = c10::hip::getCurrentHIPStream();
  gemv_w4_kpack_splitk<<<grid, block, 0, stream>>>(
      (const __half*)a.data_ptr<at::Half>(),
      (const unsigned int*)wq.data_ptr<int>(),
      (const __half*)s.data_ptr<at::Half>(),
      (const unsigned char*)z.data_ptr<uint8_t>(), c_acc.data_ptr<float>(), K,
      N, (int)group_size, (int)split_groups);
  return c_acc.to(a.dtype()).view({1, N});
}

torch::Tensor gemv_w4_kpack_sym(torch::Tensor a, torch::Tensor wq,
                                torch::Tensor s, int64_t group_size,
                                int64_t split_groups, double bias) {
  const int K = a.size(-1);
  const int N = s.size(1);
  const int num_groups = K / (int)group_size;
  const int BT = 256;
  dim3 block(BT);
  dim3 grid((N + BT * COLS - 1) / (BT * COLS),
            (num_groups + split_groups - 1) / split_groups);
  // No split-K needed (single k-range) -> write fp16 directly.
  const int direct = grid.y <= 1;
  torch::Tensor c_acc, c_out_t;
  float* cp = nullptr;
  __half* op = nullptr;
  if (!direct) {
    c_acc = torch::zeros(
        {N}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
    cp = c_acc.data_ptr<float>();
  } else {
    c_out_t = torch::empty({N}, a.options());
    op = (__half*)c_out_t.data_ptr<at::Half>();
  }
  auto stream = c10::hip::getCurrentHIPStream();
  const int rows = (int)group_size / 8;
  if (rows % 8 == 0)
    gemv_w4_kpack_sym_splitk<8><<<grid, block, 0, stream>>>(
        (const __half*)a.data_ptr<at::Half>(),
        (const unsigned int*)wq.data_ptr<int>(),
        (const __half*)s.data_ptr<at::Half>(), cp, op, direct, K, N,
        (int)group_size, (int)split_groups, (float)bias);
  else if (rows % 4 == 0)
    gemv_w4_kpack_sym_splitk<4><<<grid, block, 0, stream>>>(
        (const __half*)a.data_ptr<at::Half>(),
        (const unsigned int*)wq.data_ptr<int>(),
        (const __half*)s.data_ptr<at::Half>(), cp, op, direct, K, N,
        (int)group_size, (int)split_groups, (float)bias);
  else if (rows % 2 == 0)
    gemv_w4_kpack_sym_splitk<2><<<grid, block, 0, stream>>>(
        (const __half*)a.data_ptr<at::Half>(),
        (const unsigned int*)wq.data_ptr<int>(),
        (const __half*)s.data_ptr<at::Half>(), cp, op, direct, K, N,
        (int)group_size, (int)split_groups, (float)bias);
  else
    gemv_w4_kpack_sym_splitk<1><<<grid, block, 0, stream>>>(
        (const __half*)a.data_ptr<at::Half>(),
        (const unsigned int*)wq.data_ptr<int>(),
        (const __half*)s.data_ptr<at::Half>(), cp, op, direct, K, N,
        (int)group_size, (int)split_groups, (float)bias);
  if (!direct) c_out_t = c_acc.to(a.dtype());
  return c_out_t.view({1, N});
}
