// SPDX-License-Identifier: Apache-2.0
// Isolated gfx1030 BF16 GEMV probe. No vLLM registration or production
// dispatch.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float value) {
  uint32_t bits = __float_as_uint(value);
  // Preserve NaN rather than rounding a small payload into infinity.
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

template <int MT, int WAVES>
__global__ __launch_bounds__(WAVES * 32) void bf16_skinny_gemv_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ y, int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row_start = blockIdx.y * MT;
  const int n = blockIdx.x * WAVES + (threadIdx.x >> 5);
  if (n >= N) return;

  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * K);
  const uint4* __restrict__ xrows = reinterpret_cast<const uint4*>(x);
  const int K8 = K / 8;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

  for (int k8 = lane; k8 < K8; k8 += 32) {
    const uint4 packed_w = wrow[k8];
    const uint16_t* wb = reinterpret_cast<const uint16_t*>(&packed_w);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wf[i] = bf16_to_f32(wb[i]);

#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (row_start + m < M) {
        const uint4 packed_x =
            xrows[static_cast<size_t>(row_start + m) * K8 + k8];
        const uint16_t* xb = reinterpret_cast<const uint16_t*>(&packed_x);
        float sum = acc[m];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          sum = __fmaf_rn(wf[i], bf16_to_f32(xb[i]), sum);
        }
        acc[m] = sum;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (row_start + m < M) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[m] += __shfl_xor(acc[m], offset);
      }
      if (lane == 0) {
        y[static_cast<size_t>(row_start + m) * N + n] = f32_to_bf16_rn(acc[m]);
      }
    }
  }
}

template <int MT>
void launch_m(const uint16_t* x, const uint16_t* w, uint16_t* y, int M, int N,
              int K, hipStream_t stream) {
  // Keep broad CTA coverage for small N, including the N=256 selector.
  const int row_tiles = (M + MT - 1) / MT;
  if (N >= 2304) {
    hipLaunchKernelGGL((bf16_skinny_gemv_kernel<MT, 8>),
                       dim3((N + 7) / 8, row_tiles), dim3(256), 0, stream, x, w,
                       y, M, N, K);
  } else if (N >= 1152) {
    hipLaunchKernelGGL((bf16_skinny_gemv_kernel<MT, 4>),
                       dim3((N + 3) / 4, row_tiles), dim3(128), 0, stream, x, w,
                       y, M, N, K);
  } else if (N >= 576) {
    hipLaunchKernelGGL((bf16_skinny_gemv_kernel<MT, 2>),
                       dim3((N + 1) / 2, row_tiles), dim3(64), 0, stream, x, w,
                       y, M, N, K);
  } else {
    hipLaunchKernelGGL((bf16_skinny_gemv_kernel<MT, 1>), dim3(N, row_tiles),
                       dim3(32), 0, stream, x, w, y, M, N, K);
  }
}

// Isolated shared-X variant. Each CTA stages one 256-element K tile for all
// MT rows, so its output waves can reuse X while streaming different W rows.
// The per-lane k8 order and FP32 FMA/reduction order match the global kernel.
template <int MT, int WAVES>
__global__ __launch_bounds__(WAVES * 32) void bf16_skinny_gemv_shared_x_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ y, int M, int N, int K) {
  __shared__ uint4 x_tile[MT * 32];  // MT x 256 BF16, at most 12 KiB.
  const int lane = threadIdx.x & 31;
  const int row_start = blockIdx.y * MT;
  const int n = blockIdx.x * WAVES + (threadIdx.x >> 5);
  const int K8 = K / 8;
  const uint4* __restrict__ xrows = reinterpret_cast<const uint4*>(x);
  const uint4* __restrict__ wrow =
      n < N ? reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * K)
            : nullptr;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

  for (int tile_k8 = 0; tile_k8 < K8; tile_k8 += 32) {
    for (int idx = threadIdx.x; idx < MT * 32; idx += blockDim.x) {
      const int m = idx / 32;
      const int k8 = tile_k8 + (idx & 31);
      uint4 packed = {0, 0, 0, 0};
      if (row_start + m < M && k8 < K8) {
        packed = xrows[static_cast<size_t>(row_start + m) * K8 + k8];
      }
      x_tile[idx] = packed;
    }
    __syncthreads();

    const int k8 = tile_k8 + lane;
    if (n < N && k8 < K8) {
      const uint4 packed_w = wrow[k8];
      const uint16_t* wb = reinterpret_cast<const uint16_t*>(&packed_w);
      float wf[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) wf[i] = bf16_to_f32(wb[i]);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (row_start + m < M) {
          const uint4 packed_x = x_tile[m * 32 + lane];
          const uint16_t* xb = reinterpret_cast<const uint16_t*>(&packed_x);
          float sum = acc[m];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            sum = __fmaf_rn(wf[i], bf16_to_f32(xb[i]), sum);
          }
          acc[m] = sum;
        }
      }
    }
    __syncthreads();  // all waves finish reading before the next X tile
  }

  // A partial output CTA still participated in every barrier above.
  if (n >= N) return;
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (row_start + m < M) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[m] += __shfl_xor(acc[m], offset);
      }
      if (lane == 0) {
        y[static_cast<size_t>(row_start + m) * N + n] = f32_to_bf16_rn(acc[m]);
      }
    }
  }
}

template <int MT>
void launch_m_shared_x(const uint16_t* x, const uint16_t* w, uint16_t* y, int M,
                       int N, int K, hipStream_t stream) {
  const int row_tiles = (M + MT - 1) / MT;
  if (N >= 2304) {
    hipLaunchKernelGGL((bf16_skinny_gemv_shared_x_kernel<MT, 8>),
                       dim3((N + 7) / 8, row_tiles), dim3(256), 0, stream, x, w,
                       y, M, N, K);
  } else if (N >= 1152) {
    hipLaunchKernelGGL((bf16_skinny_gemv_shared_x_kernel<MT, 4>),
                       dim3((N + 3) / 4, row_tiles), dim3(128), 0, stream, x, w,
                       y, M, N, K);
  } else if (N >= 576) {
    hipLaunchKernelGGL((bf16_skinny_gemv_shared_x_kernel<MT, 2>),
                       dim3((N + 1) / 2, row_tiles), dim3(64), 0, stream, x, w,
                       y, M, N, K);
  } else {
    hipLaunchKernelGGL((bf16_skinny_gemv_shared_x_kernel<MT, 1>),
                       dim3(N, row_tiles), dim3(32), 0, stream, x, w, y, M, N,
                       K);
  }
}

int launch_checked(const void* x, const void* w, void* y, int M, int N, int K,
                   int row_tile, hipStream_t stream, bool shared_x) {
  if (x == nullptr || w == nullptr || y == nullptr || M < 1 || M > 24 ||
      N < 1 || K < 1 || K % 8 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(w) % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
#define LAUNCH_TILE(TILE)                                                      \
  if (shared_x)                                                                \
    launch_m_shared_x<TILE>(static_cast<const uint16_t*>(x),                   \
                            static_cast<const uint16_t*>(w),                   \
                            static_cast<uint16_t*>(y), M, N, K, stream);       \
  else                                                                         \
    launch_m<TILE>(static_cast<const uint16_t*>(x),                            \
                   static_cast<const uint16_t*>(w), static_cast<uint16_t*>(y), \
                   M, N, K, stream)
  if (row_tile == 4) {
    LAUNCH_TILE(4);
  } else if (row_tile == 8) {
    LAUNCH_TILE(8);
  } else if (row_tile == 12) {
    LAUNCH_TILE(12);
  } else if (row_tile != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  } else if (M <= 3) {
    LAUNCH_TILE(3);
  } else if (M == 4) {
    LAUNCH_TILE(4);
  } else if (M <= 8) {
    LAUNCH_TILE(8);
  } else if (M <= 16) {
    LAUNCH_TILE(16);
  } else {
    LAUNCH_TILE(24);
  }
#undef LAUNCH_TILE
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int bf16_skinny_gemv_launch(const void* x, const void* w, void* y,
                                       int M, int N, int K, int row_tile,
                                       hipStream_t stream) {
  return launch_checked(x, w, y, M, N, K, row_tile, stream, false);
}

extern "C" int bf16_skinny_gemv_shared_x_launch(const void* x, const void* w,
                                                void* y, int M, int N, int K,
                                                int row_tile,
                                                hipStream_t stream) {
  return launch_checked(x, w, y, M, N, K, row_tile, stream, true);
}
