// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Contract (T = ceil(M / M_TILE) row tiles):
//   w       [K/8, N]         uint32  GPTQ K-packed, then gptq_shuffle: the
//                                    buffer the RDNA2 W4A16 kernels read.
//                                    uint4 nibbles, zero-extended.
//   qzeros  [K/G, N/8]       uint32  stored zeros; z = nibble + zero_offset
//                                    (0 for AWQ uint4, 1 for GPTQv1 uint4b8).
//   scales  [K/G, N]         fp16
//   a       [T][K/8][MT][8]  int8    per-token symmetric quant, tile-
//                                    interleaved; bytes of each 8-K chunk in
//                                    kAPerm order; rows >= M are zero.
//   a_scale [M]              f32     absmax / 127 per token, or
//           [T][K/G][MT]     f32     per (token, group) for A_GROUP configs
//   asum    [T][K/G][MT]     int32   per-group sums of a
//   out     [M, N]           fp16    (pk4 CAS atomic add when split_k > 1;
//                                    the caller zero-fills) or f32
//                                    (split_k == 1, plain stores).
//
//   out[m,n] = a_scale[m] * sum_g scales[g,n] *
//              (sum_{k in g} a[m,k] * q[k,n] - z[g,n] * asum[m,g])
//
// i32 accumulates within one group; each group is flushed to f32 with its
// scale, because every existing W4 pack is group-quantized.

#pragma once

#include <cstddef>
#include <cstdint>

namespace vllm {
namespace explore_w4a8 {

typedef _Float16 f16_t;
typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));
typedef _Float16 f16x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

// SWAR unpack of a shuffled dword: w & kLoMask holds K offsets {0,4,1,5},
// (w >> 4) & kLoMask holds {2,6,3,7}. A is stored in the matching order.
constexpr uint32_t kLoMask = 0x0F0F0F0Fu;
constexpr int kAPerm[8] = {0, 4, 1, 5, 2, 6, 3, 7};

enum class ASrc : int {
  kLds = 0,   // ConfigA-faithful: stage the int8 A split in LDS.
  kSmem = 1,  // "LDS=0": wave-uniform A loads through the scalar cache.
};

template <int Threads_, int NPerThread_, int KStep_, int MTile_, int Group_,
          ASrc ASrc_, bool AGroup_ = false>
struct Cfg {
  static constexpr int THREADS = Threads_;
  static constexpr int NPT = NPerThread_;
  static constexpr int N_TILE = THREADS * NPT;
  static constexpr int K_STEP = KStep_;
  static constexpr int M_TILE = MTile_;
  static constexpr int GROUP = Group_;
  static constexpr ASrc A_SRC = ASrc_;
  // One activation scale per (token, weight group) instead of per token:
  // outlier channels only coarsen their own group (llama.cpp's Q8_1 idea),
  // for one extra multiply per output per group in the flush.
  static constexpr bool A_GROUP = AGroup_;
  // ConfigH lesson: each unrolled step consumes exactly K_STEP / 8 W dwords
  // per column and then advances k by K_STEP.
  static constexpr int DW_PER_STEP = K_STEP / 8;
  static constexpr int STEPS_PER_GROUP = GROUP / K_STEP;
  // Fence the scheduler between K steps: kSmem must keep one step of A in
  // SGPRs, and NPT=2 must keep one step of W row addresses live, or the
  // hoisted loads spill SGPRs (see isa_check.py). The rest schedule best
  // unconstrained.
  static constexpr bool STEP_BARRIER = A_SRC == ASrc::kSmem || NPT == 2;
  static_assert(K_STEP == 16 || K_STEP == 32, "K_STEP must be 16 or 32");
  static_assert(GROUP == 32 || GROUP == 64 || GROUP == 128,
                "group size must be 32, 64 or 128");
  static_assert(GROUP % K_STEP == 0, "a K step never straddles a group");
  static_assert(NPT == 2 || NPT == 4, "64- or 128-bit W loads");
  static_assert(M_TILE == 8 || M_TILE == 16 || M_TILE == 32, "M tile");
  static_assert(THREADS % 32 == 0 && THREADS <= 1024, "whole wave32s");
};

// LDS bytes of the kLds variant: int8 A tile, int32 group sums, and f32
// group scales for A_GROUP configs.
template <class C>
constexpr int lds_bytes(int k_per_split) {
  return C::A_SRC == ASrc::kLds
             ? C::M_TILE * k_per_split +
                   C::M_TILE * (k_per_split / C::GROUP) * (C::A_GROUP ? 8 : 4)
             : 0;
}

#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ int thread_x() {
  return __builtin_amdgcn_workitem_id_x();
}
__device__ __forceinline__ int block_x() {
  return __builtin_amdgcn_workgroup_id_x();
}
__device__ __forceinline__ int block_y() {
  return __builtin_amdgcn_workgroup_id_y();
}
__device__ __forceinline__ int block_z() {
  return __builtin_amdgcn_workgroup_id_z();
}

__device__ __forceinline__ void block_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

__device__ __forceinline__ int32_t sdot4(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), acc,
                                /*clamp=*/false);
}

// Sign-extend from 24 bits so the zero fold selects v_mul_i32_i24 instead of
// the quarter-rate v_mul_lo_u32. Exact for |asum| < 2^23 (group <= 65535).
__device__ __forceinline__ int32_t as_i24(int32_t x) {
  return static_cast<int32_t>(static_cast<uint32_t>(x) << 8) >> 8;
}

template <int NPT>
__device__ __forceinline__ void load_w(const uint32_t* p, uint32_t (&w)[NPT]) {
  if constexpr (NPT == 4) {
    const u32x4_t v = *reinterpret_cast<const u32x4_t*>(p);
    w[0] = v.x;
    w[1] = v.y;
    w[2] = v.z;
    w[3] = v.w;
  } else {
    const u32x2_t v = *reinterpret_cast<const u32x2_t*>(p);
    w[0] = v.x;
    w[1] = v.y;
  }
}

// Negated zeros and f32 scales of NPT columns in group g. The zeros stay
// int16 across the loop so the multiply provably fits v_mul_i32_i24.
template <int NPT>
__device__ __forceinline__ void load_group(const uint32_t* __restrict__ qzeros,
                                           const f16_t* __restrict__ scales,
                                           int g, int n, int size_n,
                                           int zero_offset, int16_t (&nz)[NPT],
                                           float (&s)[NPT]) {
  const uint32_t zw =
      qzeros[static_cast<size_t>(g) * (size_n / 8) + n / 8] >> ((n & 7) * 4);
  const f16_t* srow = scales + static_cast<size_t>(g) * size_n + n;
  f16_t sh[NPT];
  if constexpr (NPT == 4) {
    const f16x4_t v = *reinterpret_cast<const f16x4_t*>(srow);
    sh[0] = v.x;
    sh[1] = v.y;
    sh[2] = v.z;
    sh[3] = v.w;
  } else {
    const f16x2_t v = *reinterpret_cast<const f16x2_t*>(srow);
    sh[0] = v.x;
    sh[1] = v.y;
  }
  #pragma unroll
  for (int c = 0; c < NPT; ++c) {
    nz[c] = static_cast<int16_t>(
        -static_cast<int32_t>(((zw >> (4 * c)) & 0xFu) + zero_offset));
    s[c] = static_cast<float>(sh[c]);
  }
}

__device__ __forceinline__ void atomic_add_f16x2(f16_t* addr, f16x2_t v) {
  uint32_t* p = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *p;
  while (true) {
    uint32_t desired =
        __builtin_bit_cast(uint32_t, __builtin_bit_cast(f16x2_t, old) + v);
    if (__hip_atomic_compare_exchange_strong(p, &old, desired, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) {
      return;
    }
  }
}

__device__ __forceinline__ void atomic_add_f16x4(f16_t* addr, f16x4_t v) {
  uint64_t* p = reinterpret_cast<uint64_t*>(addr);
  uint64_t old = *p;
  while (true) {
    uint64_t desired =
        __builtin_bit_cast(uint64_t, __builtin_bit_cast(f16x4_t, old) + v);
    if (__hip_atomic_compare_exchange_strong(p, &old, desired, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) {
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// GEMM: grid (ceil(N / N_TILE), ceil(M / M_TILE), split_k), THREADS threads.
// Every K split covers whole groups: k_per_split % GROUP == 0.
// ---------------------------------------------------------------------------
template <class C>
__global__ __launch_bounds__(C::THREADS) void w4a8_gemm_kernel(
    const int8_t* __restrict__ a, const uint32_t* __restrict__ w,
    const uint32_t* __restrict__ qzeros, const f16_t* __restrict__ scales,
    const float* __restrict__ a_scale, const int32_t* __restrict__ asum,
    void* __restrict__ out, int size_m, int size_n, int size_k, int zero_offset,
    int k_per_split, int split_k, int out_f32) {
  constexpr int MT = C::M_TILE;
  constexpr int NPT = C::NPT;
  constexpr int G = C::GROUP;
  constexpr int DW = C::DW_PER_STEP;
  constexpr bool kLds = C::A_SRC == ASrc::kLds;

  const int t = thread_x();
  const int n = block_x() * C::N_TILE + t * NPT;
  const int m0 = block_y() * MT;
  const int k_begin = block_z() * k_per_split;
  const int g_begin = k_begin / G;
  const int groups_in_split = k_per_split / G;
  const int groups_total = size_k / G;
  const bool active = n < size_n;

  // This split of this row tile: (k_per_split / 8) chunks of MT x 8 bytes,
  // then groups_in_split x MT group sums (and group A scales), all
  // contiguous in global memory.
  const int8_t* a_split = a + static_cast<size_t>(block_y()) * size_k * MT +
                          static_cast<size_t>(k_begin) * MT;
  const int32_t* asum_split =
      asum + static_cast<size_t>(block_y()) * groups_total * MT +
      static_cast<size_t>(g_begin) * MT;
  const float* ascale_split = a_scale + (asum_split - asum);

  extern __shared__ uint8_t lds[];
  uint8_t* lds_a = lds;
  int32_t* lds_asum = reinterpret_cast<int32_t*>(lds + MT * k_per_split);
  float* lds_ascale = reinterpret_cast<float*>(lds_asum + MT * groups_in_split);
  if constexpr (kLds) {
    const u32x4_t* src = reinterpret_cast<const u32x4_t*>(a_split);
    u32x4_t* dst = reinterpret_cast<u32x4_t*>(lds_a);
    for (int i = t; i < MT * k_per_split / 16; i += C::THREADS) {
      dst[i] = src[i];
    }
    for (int i = t; i < MT * groups_in_split; i += C::THREADS) {
      lds_asum[i] = asum_split[i];
      if constexpr (C::A_GROUP) {
        lds_ascale[i] = ascale_split[i];
      }
    }
    block_sync();
  }

  if (!active) {
    return;
  }

  const uint8_t* a_base =
      kLds ? lds_a : reinterpret_cast<const uint8_t*>(a_split);
  const int32_t* asum_base = kLds ? lds_asum : asum_split;
  const float* ascale_base = kLds ? lds_ascale : ascale_split;

  float cf[MT][NPT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
  #pragma unroll
    for (int c = 0; c < NPT; ++c) {
      cf[m][c] = 0.0f;
    }
  }

  // Group params are fetched one group ahead so the accumulator init below
  // does not wait on a global load.
  int16_t nz[NPT];
  float s[NPT];
  load_group<NPT>(qzeros, scales, g_begin, n, size_n, zero_offset, nz, s);

  const uint32_t* wp = w + static_cast<size_t>(k_begin / 8) * size_n + n;
  for (int gi = 0; gi < groups_in_split; ++gi) {
    // Zero fold as the accumulator init: acc = -z * asum. v_mul_i32_i24
    // writes the register, so the tied v_dot4c needs no v_mov 0 per group.
    int32_t acc[MT][NPT];
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int32_t as = as_i24(asum_base[gi * MT + m]);
  #pragma unroll
      for (int c = 0; c < NPT; ++c) {
        acc[m][c] = static_cast<int32_t>(nz[c]) * as;
      }
    }

    int16_t nz_next[NPT];
    float s_next[NPT];
    const int g_next = gi + 1 < groups_in_split ? gi + 1 : gi;
    load_group<NPT>(qzeros, scales, g_begin + g_next, n, size_n, zero_offset,
                    nz_next, s_next);

  #pragma unroll
    for (int st = 0; st < C::STEPS_PER_GROUP; ++st) {
      if constexpr (C::STEP_BARRIER) {
        if (st > 0) {
          __builtin_amdgcn_sched_barrier(0);
        }
      }
      uint32_t wv[DW][NPT];
  #pragma unroll
      for (int j = 0; j < DW; ++j) {
        load_w<NPT>(wp + static_cast<size_t>(j) * size_n, wv[j]);
      }
      wp += static_cast<size_t>(DW) * size_n;
      // Chunk-major tile: every A read of the step is base + immediate.
      const uint8_t* a_step = a_base + (gi * G + st * C::K_STEP) * MT;

  #pragma unroll
      for (int j = 0; j < DW; ++j) {
        uint32_t lo[NPT], hi[NPT];
  #pragma unroll
        for (int c = 0; c < NPT; ++c) {
          lo[c] = wv[j][c] & kLoMask;
          hi[c] = (wv[j][c] >> 4) & kLoMask;
        }
  #pragma unroll
        for (int m = 0; m < MT; ++m) {
          const u32x2_t a8 =
              *reinterpret_cast<const u32x2_t*>(a_step + (j * MT + m) * 8);
  #pragma unroll
          for (int c = 0; c < NPT; ++c) {
            acc[m][c] = sdot4(a8.x, lo[c], acc[m][c]);
            acc[m][c] = sdot4(a8.y, hi[c], acc[m][c]);
          }
        }
      }
    }

    // Group flush: cvt_f32_i32 + fma with the group scale (A_GROUP adds a
    // multiply by the row's group A scale).
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      float sa = 1.0f;
      if constexpr (C::A_GROUP) {
        sa = ascale_base[gi * MT + m];
      }
  #pragma unroll
      for (int c = 0; c < NPT; ++c) {
        float v = static_cast<float>(acc[m][c]);
        if constexpr (C::A_GROUP) {
          v *= sa;
        }
        cf[m][c] = __builtin_fmaf(v, s[c], cf[m][c]);
      }
    }
  #pragma unroll
    for (int c = 0; c < NPT; ++c) {
      nz[c] = nz_next[c];
      s[c] = s_next[c];
    }
  }

  // Epilogue: per-token activation scale (already folded for A_GROUP), then
  // store or pk atomic add.
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int row = m0 + m;
    if (row >= size_m) {
      continue;
    }
    float sa = 1.0f;
    if constexpr (!C::A_GROUP) {
      sa = a_scale[row];
    }
    const size_t off = static_cast<size_t>(row) * size_n + n;
    if (out_f32) {
      float* o = static_cast<float*>(out) + off;
  #pragma unroll
      for (int c = 0; c < NPT; ++c) {
        o[c] = cf[m][c] * sa;
      }
      continue;
    }
    f16_t* o = static_cast<f16_t*>(out) + off;
    if constexpr (NPT == 4) {
      const f16x4_t v = {
          static_cast<f16_t>(cf[m][0] * sa), static_cast<f16_t>(cf[m][1] * sa),
          static_cast<f16_t>(cf[m][2] * sa), static_cast<f16_t>(cf[m][3] * sa)};
      if (split_k == 1) {
        *reinterpret_cast<f16x4_t*>(o) = v;
      } else {
        atomic_add_f16x4(o, v);
      }
    } else {
      const f16x2_t v = {static_cast<f16_t>(cf[m][0] * sa),
                         static_cast<f16_t>(cf[m][1] * sa)};
      if (split_k == 1) {
        *reinterpret_cast<f16x2_t*>(o) = v;
      } else {
        atomic_add_f16x2(o, v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Act quant: one block per row tile. Matches vLLM dynamic_scaled_int8_quant
// (absmax / 127, rint(x * (127 / absmax)), saturate), and writes the
// tile-interleaved kAPerm layout plus per-group sums. Thread t always owns
// row t % MT and walks that row's groups, so group sums need no atomics and
// neighbouring threads write neighbouring 8-byte slots.
//
// PER_GROUP writes one scale per (token, group) into a [T][K/G][MT] tile
// instead of one per token. A group's absmax is then local to the thread that
// quantizes it, so that variant needs no block reduction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float chunk_absmax(const f16_t* p, float amax) {
  const u32x4_t raw = *reinterpret_cast<const u32x4_t*>(p);
  const f16x2_t* h = reinterpret_cast<const f16x2_t*>(&raw);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    amax = __builtin_fmaxf(amax, __builtin_fabsf(static_cast<float>(h[i].x)));
    amax = __builtin_fmaxf(amax, __builtin_fabsf(static_cast<float>(h[i].y)));
  }
  return amax;
}

template <int THREADS, int MT, bool PER_GROUP>
__global__ __launch_bounds__(THREADS) void w4a8_act_quant_kernel(
    const f16_t* __restrict__ x, int64_t x_row_stride, int8_t* __restrict__ a,
    float* __restrict__ a_scale, int32_t* __restrict__ asum, int size_m,
    int size_k, int group_size) {
  static_assert((THREADS & (THREADS - 1)) == 0 && THREADS % MT == 0,
                "power-of-two block, whole rows per thread stride");
  __shared__ float red[THREADS];

  const int tile = block_x();
  const int t = thread_x();
  const int m = t % MT;
  const int row = tile * MT + m;
  const bool valid = row < size_m;
  const int groups = size_k / group_size;
  const int chunks_per_group = group_size / 8;
  const f16_t* xr = x + (valid ? row : 0) * x_row_stride;
  int8_t* a_tile = a + static_cast<size_t>(tile) * size_k * MT;
  int32_t* asum_tile = asum + static_cast<size_t>(tile) * groups * MT;

  float inv = 0.0f;
  if constexpr (!PER_GROUP) {
    float amax = 0.0f;
    if (valid) {
      for (int g = t / MT; g < groups; g += THREADS / MT) {
        for (int i = 0; i < chunks_per_group; ++i) {
          amax = chunk_absmax(xr + 8 * (g * chunks_per_group + i), amax);
        }
      }
    }
    red[t] = amax;
    block_sync();
    for (int s = THREADS / 2; s >= MT; s >>= 1) {
      if (t < s) {
        red[t] = __builtin_fmaxf(red[t], red[t + s]);
      }
      block_sync();
    }
    const float absmax = red[m];
    inv = absmax == 0.0f ? 0.0f : 127.0f / absmax;
    if (t < MT && valid) {
      a_scale[row] = absmax / 127.0f;
    }
  }

  for (int g = t / MT; g < groups; g += THREADS / MT) {
    if constexpr (PER_GROUP) {
      float gmax = 0.0f;
      if (valid) {
        for (int i = 0; i < chunks_per_group; ++i) {
          gmax = chunk_absmax(xr + 8 * (g * chunks_per_group + i), gmax);
        }
      }
      inv = gmax == 0.0f ? 0.0f : 127.0f / gmax;
      a_scale[(static_cast<size_t>(tile) * groups + g) * MT + m] =
          gmax / 127.0f;
    }
    int32_t sum = 0;
    for (int i = 0; i < chunks_per_group; ++i) {
      const int c = g * chunks_per_group + i;
      u32x2_t packed = {0u, 0u};
      if (valid) {
        const u32x4_t raw = *reinterpret_cast<const u32x4_t*>(xr + 8 * c);
        const f16x2_t* h = reinterpret_cast<const f16x2_t*>(&raw);
        int32_t q[8];
  #pragma unroll
        for (int p = 0; p < 4; ++p) {
  #pragma unroll
          for (int e = 0; e < 2; ++e) {
            float v = __builtin_rintf(static_cast<float>(h[p][e]) * inv);
            v = __builtin_fminf(__builtin_fmaxf(v, -128.0f), 127.0f);
            q[2 * p + e] = static_cast<int32_t>(v);
            sum += q[2 * p + e];
          }
        }
        packed.x = (static_cast<uint32_t>(q[kAPerm[0]]) & 0xFFu) |
                   (static_cast<uint32_t>(q[kAPerm[1]]) & 0xFFu) << 8 |
                   (static_cast<uint32_t>(q[kAPerm[2]]) & 0xFFu) << 16 |
                   (static_cast<uint32_t>(q[kAPerm[3]]) & 0xFFu) << 24;
        packed.y = (static_cast<uint32_t>(q[kAPerm[4]]) & 0xFFu) |
                   (static_cast<uint32_t>(q[kAPerm[5]]) & 0xFFu) << 8 |
                   (static_cast<uint32_t>(q[kAPerm[6]]) & 0xFFu) << 16 |
                   (static_cast<uint32_t>(q[kAPerm[7]]) & 0xFFu) << 24;
      }
      *reinterpret_cast<u32x2_t*>(a_tile + (static_cast<size_t>(c) * MT + m) *
                                               8) = packed;
    }
    asum_tile[g * MT + m] = sum;
  }
}

#else  // non-gfx1030 device pass: same signatures, no body.

template <class C>
__global__ __launch_bounds__(C::THREADS) void w4a8_gemm_kernel(
    const int8_t*, const uint32_t*, const uint32_t*, const f16_t*, const float*,
    const int32_t*, void*, int, int, int, int, int, int, int) {}

template <int THREADS, int MT, bool PER_GROUP>
__global__ __launch_bounds__(THREADS) void w4a8_act_quant_kernel(
    const f16_t*, int64_t, int8_t*, float*, int32_t*, int, int, int) {}

#endif  // __gfx1030__ || !__HIP_DEVICE_COMPILE__

}  // namespace explore_w4a8
}  // namespace vllm
