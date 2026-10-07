// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A8 (int4 weights x int8 activations) kernel for AMD RDNA2
// (gfx1030), fp16 only.
//
// The MoE half of the W4A8 sdot4 path: the dense kernel in
// ``w4a8_sdot4_rdna2.cu`` is unreachable for models whose quant is entirely
// ``mlp.experts.*`` (Flash-Next), so this kernel makes W4A8 able to fire at
// all. It is a drop-in for ``moe_gptq_gemm_rdna2`` (same output contract,
// same arg list plus ``use_v2_format``) and falls back to it internally on
// any ineligibility.
//
// Design
// ------
// * Activations: the dense ``w4a8_act_quant_kernel`` (per-(token, group)
//   scales, ``a8_lds_k32_ag`` layout) runs ONCE per call over the whole token
//   batch, expert-agnostic. The int8 A, per-(token, group) fp32 scales and
//   int32 group sums live in a per-(M, K, G) cached workspace so their device
//   pointers are stable across calls (a per-call ``at::empty`` inside a
//   captured graph is the suspected `duct` corruption source on this stack).
// * GEMM: grid ``(num_token_blocks, ceil(N/1024), ceil(K/256))``, 256 threads,
//   4 N columns per thread, K_STEP = 32. Each block owns BLOCK_M sorted
//   tokens of one expert and 256 K positions (whole weight groups).
// * A is read direct from global (no LDS cap). Each row is addressed through
//   its ``[T][K/8][8][8]`` tile as (tile_r = row/8, m_r = row%8); padding rows
//   clamp to tile 0/row 0 and are skipped by the epilogue.
// * Epilogue is byte-identical to ``moe_gptq_gemm_rdna2`` (shared helper in
//   ``moe_accum_rdna2.cuh``): router-weight multiply in fp32, then either the
//   default fp32 accumulation (native fp32 atomics into a cached scratch +
//   one fp32 -> fp16 cast) or the opt-in packed fp16 CAS-64 atomic add,
//   with ``output_topk`` row reduce fused.
//
// Weight format matches the W4A16 MoE kernel: [E, K/8, N] uint32 shuffled,
// [E, groups, N] scales, [E, groups, N/8] packed zeros. ``use_v2_format``
// selects ``zero_offset`` (0 for AWQ uint4, 1 for GPTQv1 uint4b8); the
// RDNA2 MoE layout currently produced by the two wired methods is GPTQv1.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "moe_accum_rdna2.cuh"
#include "ops.h"
#include "q_gemm_rdna2_common.cuh"  // gptq_rdna2::atomic_add_pk4_f16
#include "w4a8_sdot4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2_MOE__
#endif

namespace ex = vllm::explore_w4a8;

namespace vllm {
namespace moe_w4a8_rdna2 {

constexpr int kThreads = 256;
constexpr int kNPerThread = 4;
constexpr int kNTile = kThreads * kNPerThread;  // 1024 output columns
constexpr int kKBlock = 256;                    // K positions per block
constexpr int kKStep = 32;                      // K positions per inner step
constexpr int kDW = kKStep / 8;                 // W dwords per step
constexpr int kMTile = 8;                       // act_quant tile (matches dense)
constexpr int kChunkBytes = kMTile * 8;         // bytes per 8-K chunk per row

#if defined(__HIP__RDNA2_MOE__) || !defined(__HIP_DEVICE_COMPILE__)

template <int BLOCK_M, int GROUP, typename C_T>
__global__ __launch_bounds__(kThreads) void moe_w4a8_gemm_kernel(
    const int8_t* __restrict__ a,          // [T][K/8][8][8] int8 (MT=8)
    const float* __restrict__ a_scale,     // [T][G][8] per-(token, group)
    const int32_t* __restrict__ a_sum,     // [T][G][8]
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] shuffled
    const uint32_t* __restrict__ b_qzeros,    // [E, G, N/8] packed
    const ex::f16_t* __restrict__ b_scales,   // [E, G, N]
    C_T* __restrict__ c,                   // [M*topk or M, N] accumulator
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int size_m, int size_n, int size_k, int top_k, int zero_offset,
    bool mul_topk_weight, int output_topk) {
  constexpr int NPT = kNPerThread;
  constexpr int DW = kDW;
  constexpr int STEPS_PER_GROUP = GROUP / kKStep;
  static_assert(GROUP == 32 || GROUP == 64 || GROUP == 128, "supported group");
  static_assert(GROUP % kKStep == 0, "a K step never straddles a group");

  const int t = ex::thread_x();
  const int token_block = ex::block_x();
  const int n = ex::block_y() * kNTile + t * NPT;
  const int offset_k = ex::block_z() * kKBlock;
  const int end_k = min(offset_k + kKBlock, size_k);
  const int groups_in_block = (end_k - offset_k) / GROUP;
  const int g_begin = offset_k / GROUP;

  // Early exit for padding blocks / invalid experts (EA expert_map = -1).
  if (token_block * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const int groups = size_k / GROUP;
  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * ((size_k / 8) * size_n);
  const ex::f16_t* expert_scales =
      b_scales + (int64_t)expert_id * (groups * size_n);
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * (groups * (size_n / 8));

  // Stage the block's whole A window (BLOCK_M x 256 int8) into LDS in one
  // cooperative sweep. Direct-global per-row A addressing kept BLOCK_M live
  // uniform pointers, pushing SGPRs to 107 and spilling; staging mirrors the
  // dense a8_lds config (46 SGPR, 0 spills) at 2 KiB of LDS. Padding rows
  // (token_row >= size_m) clamp to tile 0 / row 0 so reads stay in bounds;
  // the epilogue skips their write.
  constexpr int CHUNKS = kKBlock / 8;
  static_assert(BLOCK_M * CHUNKS <= kThreads, "staging needs <= THREADS chunks");
  const int offset_m_base = token_block * BLOCK_M;
  const int chunk0 = offset_k / 8;
  const int valid_chunks = (end_k - offset_k) / 8;
  __shared__ int sh_s_off[BLOCK_M];
  __shared__ int8_t lds_a[BLOCK_M * kKBlock];
  if (t < BLOCK_M) {
    const int token_id = sorted_token_ids[offset_m_base + t];
    const int token_row = token_id / top_k;
    const int tr = token_row < size_m ? (token_row >> 3) : 0;
    const int mr = token_row < size_m ? (token_row & 7) : 0;
    sh_s_off[t] = tr * groups * kMTile + mr;
  }
  if (t < BLOCK_M * CHUNKS) {
    const int m = t / CHUNKS;
    const int c = t % CHUNKS;
    ex::u32x2_t v = {0u, 0u};
    if (c < valid_chunks) {
      const int token_id = sorted_token_ids[offset_m_base + m];
      const int token_row = token_id / top_k;
      const int tr = token_row < size_m ? (token_row >> 3) : 0;
      const int mr = token_row < size_m ? (token_row & 7) : 0;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(a) +
                           (size_t)tr * (size_k / 8) * kChunkBytes + mr * 8 +
                           (size_t)(chunk0 + c) * kChunkBytes;
      v = *reinterpret_cast<const ex::u32x2_t*>(src);
    }
    *reinterpret_cast<ex::u32x2_t*>(lds_a + m * kKBlock + c * 8) = v;
  }
  __syncthreads();

  // Every thread owns 4 contiguous N columns; with N % 8 == 0 an in-range n
  // keeps all four columns in range.
  if (n >= size_n) return;

  float cf[BLOCK_M][NPT];
  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
  #pragma unroll
    for (int cc = 0; cc < NPT; ++cc) {
      cf[m][cc] = 0.0f;
    }
  }

  int16_t nz[NPT];
  float s[NPT];
  ex::load_group<NPT>(expert_qzeros, expert_scales, g_begin, n, size_n,
                      zero_offset, nz, s);

  const uint32_t* wp = expert_weights + (size_t)chunk0 * size_n + n;

  for (int gi = 0; gi < groups_in_block; ++gi) {
    const int g = g_begin + gi;

    // Zero fold as the accumulator init (mirrors the dense kernel): each
    // row's group sum is subtracted through the negated zero.
    int32_t acc[BLOCK_M][NPT];
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
      const int32_t as = ex::as_i24(a_sum[sh_s_off[m] + g * kMTile]);
      #pragma unroll
      for (int cc = 0; cc < NPT; ++cc) {
        acc[m][cc] = static_cast<int32_t>(nz[cc]) * as;
      }
    }

    int16_t nz_next[NPT];
    float s_next[NPT];
    const int g_next = gi + 1 < groups_in_block ? gi + 1 : gi;
    ex::load_group<NPT>(expert_qzeros, expert_scales, g_begin + g_next, n,
                        size_n, zero_offset, nz_next, s_next);

    #pragma unroll
    for (int st = 0; st < STEPS_PER_GROUP; ++st) {
      uint32_t wv[DW][NPT];
      #pragma unroll
      for (int j = 0; j < DW; ++j) {
        ex::load_w<NPT>(wp + (size_t)j * size_n, wv[j]);
      }
      wp += (size_t)DW * size_n;
      const int8_t* a_step = lds_a + gi * GROUP + st * kKStep;
      #pragma unroll
      for (int j = 0; j < DW; ++j) {
        uint32_t lo[NPT];
        uint32_t hi[NPT];
        #pragma unroll
        for (int cc = 0; cc < NPT; ++cc) {
          lo[cc] = wv[j][cc] & ex::kLoMask;
          hi[cc] = (wv[j][cc] >> 4) & ex::kLoMask;
        }
        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
          const ex::u32x2_t a8 = *reinterpret_cast<const ex::u32x2_t*>(
              a_step + m * kKBlock + j * 8);
          #pragma unroll
          for (int cc = 0; cc < NPT; ++cc) {
            acc[m][cc] = ex::sdot4(a8.x, lo[cc], acc[m][cc]);
            acc[m][cc] = ex::sdot4(a8.y, hi[cc], acc[m][cc]);
          }
        }
      }
    }

    // Group flush: fp32 convert, per-(token, group) A scale, weight scale.
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
      const float sa = a_scale[sh_s_off[m] + g * kMTile];
      #pragma unroll
      for (int cc = 0; cc < NPT; ++cc) {
        const float v = static_cast<float>(acc[m][cc]) * sa;
        cf[m][cc] = __builtin_fmaf(v, s[cc], cf[m][cc]);
      }
    }
    #pragma unroll
    for (int cc = 0; cc < NPT; ++cc) {
      nz[cc] = nz_next[cc];
      s[cc] = s_next[cc];
    }
  }

  // Epilogue: identical shape to moe_gptq_gemm_rdna2 — router weight in fp32,
  // then the shared accumulator (fp32 atomics by default, fp16 CAS opt-in).
  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
      #pragma unroll
      for (int cc = 0; cc < NPT; ++cc) {
        cf[m][cc] *= tw;
      }
    }

    const int64_t out_row = (output_topk > 0)
                                ? (int64_t)(token_id / output_topk)
                                : (int64_t)token_id;
    C_T* out = c + out_row * size_n + n;
    vllm::gptq_rdna2::moe_accum_row<C_T>(cf[m], out);
  }
}

#else  // non-RDNA2 device pass: signature parity, empty body.

template <int BLOCK_M, int GROUP, typename C_T>
__global__ __launch_bounds__(kThreads) void moe_w4a8_gemm_kernel(
    const int8_t*, const float*, const int32_t*, const uint32_t*,
    const uint32_t*, const ex::f16_t*, C_T*, const float*, const int32_t*,
    const int32_t*, const int32_t*, int, int, int, int, int, bool, int) {}

#endif  // __HIP__RDNA2_MOE__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_w4a8_rdna2
}  // namespace vllm

namespace mw = vllm::moe_w4a8_rdna2;

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

namespace {

enum MoeErr : int {
  kOk = 0,
  kBadBlockM = -1,
  kBadGroup = -2,
  kBadLaunch = -4,
};

int group_index(int64_t group_size) {
  switch (group_size) {
    case 32:
      return 0;
    case 64:
      return 1;
    case 128:
      return 2;
    default:
      return -1;
  }
}

bool on_gfx1030() {
  thread_local int cached_dev = -1;
  thread_local bool cached_ok = false;
  const int dev = at::cuda::current_device();
  if (dev != cached_dev) {
    hipDeviceProp_t prop;
    cached_ok = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx1030", 7) == 0;
    cached_dev = dev;
  }
  return cached_ok;
}

// Stable per-(m, k, groups, device) activation workspace. Held for the life of
// the process so the device pointers captured into a graph never move; the
// dense path's per-call allocation is the suspected graph-aliasing source.
struct Workspace {
  at::Tensor a_i8;     // [T, K/8, 8, 8] int8
  at::Tensor a_scale;  // [T, G, 8] fp32
  at::Tensor a_sum;    // [T, G, 8] int32
};

const Workspace& get_workspace(int64_t m, int64_t k, int64_t g, int64_t device) {
  static std::mutex mu;
  static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, Workspace>
      cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(m, k, g, device);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  const at::Device dev(at::kCUDA, device);
  Workspace ws;
  ws.a_i8 = at::empty({(m + 7) / 8, k / 8, 8, 8},
                      at::TensorOptions().device(dev).dtype(at::kChar));
  ws.a_scale = at::empty({(m + 7) / 8, g, 8},
                         at::TensorOptions().device(dev).dtype(at::kFloat));
  ws.a_sum = at::empty({(m + 7) / 8, g, 8},
                       at::TensorOptions().device(dev).dtype(at::kInt));
  return cache.emplace(key, std::move(ws)).first->second;
}

struct MoeArgs {
  const int8_t* a_i8;
  const float* a_scale;
  const int32_t* a_sum;
  const uint32_t* w;
  const uint32_t* qzeros;
  const ex::f16_t* scales;
  void* c;  // accumulator: fp16 (CAS) or fp32 (scratch) by C_T
  const float* topk_weights;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_post_padded;
  int num_token_blocks;
  int size_m;
  int size_n;
  int size_k;
  int top_k;
  int zero_offset;
  bool mul_topk_weight;
  int output_topk;
};

template <int BLOCK_M, int GROUP, typename C_T>
int launch_kernel(const MoeArgs& p, hipStream_t stream) {
  dim3 grid(p.num_token_blocks, (p.size_n + mw::kNTile - 1) / mw::kNTile,
            (p.size_k + mw::kKBlock - 1) / mw::kKBlock);
  mw::moe_w4a8_gemm_kernel<BLOCK_M, GROUP, C_T>
      <<<grid, dim3(mw::kThreads), 0, stream>>>(
          p.a_i8, p.a_scale, p.a_sum, p.w, p.qzeros, p.scales,
          static_cast<C_T*>(p.c), p.topk_weights, p.sorted_token_ids,
          p.expert_ids, p.num_tokens_post_padded, p.size_m, p.size_n, p.size_k,
          p.top_k, p.zero_offset, p.mul_topk_weight, p.output_topk);
  const hipError_t e = hipGetLastError();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e == hipSuccess ? kOk : kBadLaunch;
}

template <int BLOCK_M, typename C_T>
int dispatch_group(int64_t group_size, const MoeArgs& p, hipStream_t stream) {
  switch (group_size) {
    case 32:
      return launch_kernel<BLOCK_M, 32, C_T>(p, stream);
    case 64:
      return launch_kernel<BLOCK_M, 64, C_T>(p, stream);
    case 128:
      return launch_kernel<BLOCK_M, 128, C_T>(p, stream);
    default:
      return kBadGroup;
  }
}

template <typename C_T>
int dispatch_block_m(int64_t block_size_m, int64_t group_size,
                     const MoeArgs& p, hipStream_t stream) {
  switch (block_size_m) {
    case 1:
      return dispatch_group<1, C_T>(group_size, p, stream);
    case 2:
      return dispatch_group<2, C_T>(group_size, p, stream);
    case 4:
      return dispatch_group<4, C_T>(group_size, p, stream);
    case 8:
      return dispatch_group<8, C_T>(group_size, p, stream);
    default:
      return kBadBlockM;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point. Same argument list as ``moe_gptq_gemm_rdna2`` plus the
// trailing zero-offset selector and the fp32-accumulation selector. Writes
// into ``c`` and falls back to ``moe_gptq_gemm_rdna2`` on any ineligibility.
// ---------------------------------------------------------------------------
void moe_w4a8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, bool use_v2_format,
                         bool fp32_accum) {
  const auto fallback = [&] {
    moe_gptq_gemm_rdna2(a, c, b_q_weight, b_scales, b_qzeros, topk_weights,
                        sorted_token_ids, expert_ids, num_tokens_post_padded,
                        top_k, block_size_m, mul_topk_weight, output_topk,
                        fp32_accum);
  };

  if (!a.is_cuda() || !c.is_cuda() || !b_q_weight.is_cuda() ||
      a.scalar_type() != at::kHalf || a.dim() != 2 || c.dim() != 2 ||
      b_q_weight.dim() != 3 || b_scales.dim() != 3 || b_qzeros.dim() != 3) {
    fallback();
    return;
  }

  const int64_t size_m = a.size(0);
  const int64_t size_k = a.size(1);
  const int64_t size_n = b_q_weight.size(2);
  const int64_t groups = b_scales.size(1);
  const int64_t group_size = groups > 0 ? size_k / groups : 0;

  const int64_t row_off_bytes =
      (int64_t)((size_m + 7) / 8) * (size_k / 8) * mw::kChunkBytes;
  const int64_t sum_off_elems = (int64_t)((size_m + 7) / 8) * groups * 8;
  const bool eligible =
      on_gfx1030() && size_k % 32 == 0 && groups > 0 &&
      size_k % group_size == 0 && group_index(group_size) >= 0 &&
      b_q_weight.size(1) * 8 == size_k && size_n % 8 == 0 &&
      a.scalar_type() == at::kHalf && b_scales.scalar_type() == at::kHalf &&
      b_q_weight.scalar_type() == at::kInt &&
      b_qzeros.scalar_type() == at::kInt &&
      sorted_token_ids.scalar_type() == at::kInt &&
      expert_ids.scalar_type() == at::kInt &&
      num_tokens_post_padded.scalar_type() == at::kInt &&
      (block_size_m == 1 || block_size_m == 2 || block_size_m == 4 ||
       block_size_m == 8) &&
      a.stride(1) == 1 && c.stride(1) == 1 &&
      row_off_bytes <= INT32_MAX && sum_off_elems <= INT32_MAX;
  if (!eligible) {
    fallback();
    return;
  }

  const at::cuda::OptionalCUDAGuard guard(a.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t device = a.get_device();

  const Workspace& ws = get_workspace(size_m, size_k, groups, device);

  // int8 A: per-(token, group) quant in the a8_lds_k32_ag layout.
  {
    const dim3 grid(static_cast<unsigned>((size_m + mw::kMTile - 1) /
                                          mw::kMTile));
    ex::w4a8_act_quant_kernel<mw::kThreads, mw::kMTile, /*PER_GROUP=*/true>
        <<<grid, dim3(mw::kThreads), 0, stream>>>(
            reinterpret_cast<const ex::f16_t*>(a.data_ptr()),
            a.stride(0), reinterpret_cast<int8_t*>(ws.a_i8.data_ptr()),
            reinterpret_cast<float*>(ws.a_scale.data_ptr()),
            reinterpret_cast<int32_t*>(ws.a_sum.data_ptr()),
            static_cast<int>(size_m), static_cast<int>(size_k),
            static_cast<int>(group_size));
    const hipError_t e = hipGetLastError();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (e != hipSuccess) {
      fallback();
      return;
    }
  }

  const MoeArgs p{reinterpret_cast<const int8_t*>(ws.a_i8.data_ptr()),
                  reinterpret_cast<const float*>(ws.a_scale.data_ptr()),
                  reinterpret_cast<const int32_t*>(ws.a_sum.data_ptr()),
                  reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
                  reinterpret_cast<const uint32_t*>(b_qzeros.data_ptr()),
                  reinterpret_cast<const ex::f16_t*>(b_scales.data_ptr()),
                  c.data_ptr(),
                  topk_weights.numel() > 0 ? topk_weights.data_ptr<float>()
                                           : nullptr,
                  sorted_token_ids.data_ptr<int32_t>(),
                  expert_ids.data_ptr<int32_t>(),
                  num_tokens_post_padded.data_ptr<int32_t>(),
                  static_cast<int>(sorted_token_ids.size(0) / block_size_m),
                  static_cast<int>(size_m),
                  static_cast<int>(size_n),
                  static_cast<int>(size_k),
                  static_cast<int>(top_k),
                  use_v2_format ? 0 : 1,
                  mul_topk_weight,
                  static_cast<int>(output_topk)};

  if (fp32_accum) {
    // fp32 accumulation: partials land in the cached fp32 scratch, then one
    // elementwise cast rounds to fp16. Shared with the W4A16 MoE kernel.
    if (!c.is_contiguous()) {
      fallback();
      return;
    }
    MoeArgs p32 = p;
    float* scratch = moe_fp32_scratch(c.size(0), c.size(1), c.get_device());
    const hipError_t memset_err = hipMemsetAsync(
        scratch, 0, static_cast<size_t>(c.numel()) * sizeof(float), stream);
    if (memset_err != hipSuccess) {
      fallback();
      return;
    }
    p32.c = scratch;
    if (dispatch_block_m<float>(block_size_m, group_size, p32, stream) != kOk) {
      fallback();
      return;
    }
    moe_cast_f32_to_f16(scratch, reinterpret_cast<half*>(c.data_ptr()),
                        c.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (dispatch_block_m<ex::f16_t>(block_size_m, group_size, p, stream) !=
             kOk) {
    fallback();
    return;
  }

  TORCH_WARN_ONCE(
      "RDNA2 W4A8 sdot4 MoE path active (config moe_a8_k32_ag)");
}
