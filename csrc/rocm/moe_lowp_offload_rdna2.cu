// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Hybrid VRAM-Slot + Mapped Pinned-Host Zero-Copy MoE W4A16 / W8A16 kernels
// for RDNA2 (gfx1030), fp16 activations.
//
// Dispatches per logical expert e = expert_ids[token_block]:
//   slot = resident_map[e]
//   - slot >= 0 && slot < slot_count : reads cached weights from dev_*[slot]
//   - otherwise                      : streams weights directly from pinned
//                                      mapped host memory host_*[e] using
//                                      non-temporal vector loads (zero CPU
//                                      wait, zero destructive VRAM slot
//                                      evictions).

#include <cstdlib>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "mxfp4_dot2_common.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

#define OFFLOAD_BLOCK_KN_SIZE 256
#define OFFLOAD_THREADS_X 256

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

namespace lowp_offload {

using vllm::mxfp4_dot2::atomic_add_pk4_f16;
using vllm::mxfp4_dot2::dot22_8_f;
using vllm::mxfp4_dot2::tzero;

__device__ __forceinline__ float e4m3_scale_to_f32(uint32_t b) {
  const uint32_t mag = b & 0x7Fu;
  const float normal = __uint_as_float((mag << 20) + 0x3C000000u);
  const float sub = static_cast<float>(mag) * 0.001953125f;
  const uint32_t bits = __float_as_uint(mag < 8u ? sub : normal);
  return __uint_as_float(bits | ((b & 0x80u) << 24));
}

__device__ __forceinline__ float ue8m0_to_f32(uint32_t b) {
  return __uint_as_float(b << 23);
}

__device__ __forceinline__ half2 e4m3_block_factor(uint32_t ue8m0) {
  return __half2half2(__float2half_rn(ue8m0_to_f32(ue8m0) * 256.0f));
}

template <uint32_t SEL>
__device__ __forceinline__ half2 e4m3x2_bits(uint32_t w) {
  const uint32_t lanes = __builtin_amdgcn_perm(0u, w, SEL);
  const uint32_t h =
      ((lanes << 8) & 0x80008000u) | ((lanes << 7) & 0x3F803F80u);
  return __builtin_bit_cast(half2, h);
}

__device__ __forceinline__ void deq_e4m3_4(uint32_t w, half2 factor,
                                           half2* dq) {
  dq[0] = __hmul2(e4m3x2_bits<0x0C010C00u>(w), factor);
  dq[1] = __hmul2(e4m3x2_bits<0x0C030C02u>(w), factor);
}

__device__ __forceinline__ void dequant_e2m1_8_fp16_fast(uint32_t qa,
                                                         half2 scale2,
                                                         half2 (&dq)[4]) {
  const uint32_t ev = qa & 0x0F0F0F0Fu;
  const uint32_t od = (qa >> 4) & 0x0F0F0F0Fu;
  const uint32_t hb_e = ((ev << 4) & 0x80808080u) | ((ev << 1) & 0x0E0E0E0Eu);
  const uint32_t hb_o = ((od << 4) & 0x80808080u) | ((od << 1) & 0x0E0E0E0Eu);
  dq[0] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x040C000Cu)),
      scale2);
  dq[1] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x050C010Cu)),
      scale2);
  dq[2] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x060C020Cu)),
      scale2);
  dq[3] = __hmul2(
      __builtin_bit_cast(half2, __builtin_amdgcn_perm(hb_o, hb_e, 0x070C030Cu)),
      scale2);
}

__device__ __forceinline__ float dot2_8_regs(const half2 (&dq)[4],
                                             const uint4& a) {
  float r = 0.0f;
  r = __builtin_amdgcn_fdot2(dq[0], __builtin_bit_cast(half2, a.x), r, false);
  r = __builtin_amdgcn_fdot2(dq[1], __builtin_bit_cast(half2, a.y), r, false);
  r = __builtin_amdgcn_fdot2(dq[2], __builtin_bit_cast(half2, a.z), r, false);
  r = __builtin_amdgcn_fdot2(dq[3], __builtin_bit_cast(half2, a.w), r, false);
  return r;
}

using kf_i32x4 = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int4 load_int4_vec(const int4* ptr, bool use_vram) {
  if (use_vram) return *ptr;
  const kf_i32x4 v =
      __builtin_nontemporal_load(reinterpret_cast<const kf_i32x4*>(ptr));
  return make_int4(v[0], v[1], v[2], v[3]);
}

template <typename T, int BLOCK_SIZE_M>
__device__ __forceinline__ void write_out(
    T* __restrict__ c, const float (&block_c)[BLOCK_SIZE_M][4],
    const int32_t* __restrict__ sorted_token_ids,
    const float* __restrict__ topk_weights, int offset_m_base, int size_m,
    int top_k, int size_n, int n, bool mul_topk_weight, int output_topk) {
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;
    float r[4] = {block_c[m][0], block_c[m][1], block_c[m][2], block_c[m][3]};
    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) r[j] *= tw;
    }
    const int64_t out_row = (output_topk > 0)
                                ? (int64_t)(token_id / output_topk)
                                : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    const half2 r01 =
        __halves2half2(__float2half_rn(r[0]), __float2half_rn(r[1]));
    const half2 r23 =
        __halves2half2(__float2half_rn(r[2]), __float2half_rn(r[3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

__device__ __forceinline__ float swiglu_act(float g, float u, float gate_cap,
                                            float up_cap) {
  const float sig = 1.0f / (1.0f + expf(-g));
  return ((tanhf(g / gate_cap) * gate_cap) * sig) *
         (tanhf(u / up_cap) * up_cap);
}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
__device__ __forceinline__ void load_activations(
    const T* __restrict__ a, T (&block_a)[BLOCK_SIZE_M][BK + 8],
    const int32_t* __restrict__ sorted_token_ids, int offset_m_base,
    int offset_k, int end_k, int t, int size_m, int size_k, int top_k,
    float gate_cap, float up_cap) {
  #pragma unroll
  for (int tk = t; tk < BK; tk += TX) {
    if (offset_k + tk < end_k) {
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const int32_t token_id = sorted_token_ids[offset_m_base + m];
        const int token_row = token_id / top_k;
        T v = tzero<T>();
        if (token_row < size_m) {
          if (gate_cap > 0.0f) {
            const T* row = a + (int64_t)token_row * 2 * size_k + offset_k + tk;
            v = __float2half_rn(swiglu_act(__half2float(row[0]),
                                           __half2float(row[size_k]), gate_cap,
                                           up_cap));
          } else {
            v = a[(int64_t)token_row * size_k + offset_k + tk];
          }
        }
        block_a[m][tk] = v;
      }
    }
  }
}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
__global__ void moe_gemm_nvfp4_hybrid_kernel_rdna2(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ dev_q_weight,
    const uint8_t* __restrict__ dev_scales,
    const float* __restrict__ dev_globals,
    const uint32_t* __restrict__ host_q_weight,
    const uint8_t* __restrict__ host_scales,
    const float* __restrict__ host_globals,
    const int32_t* __restrict__ resident_map, const int slot_count,
    const int num_experts, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk, const float gate_cap,
    const float up_cap) {
  static_assert(BK % 32 == 0, "BK must be a multiple of 32");
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * TX * 4;
  const int offset_k = blockIdx.z * BK;
  const int end_k = min(offset_k + BK, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int slot = resident_map[expert_id];
  const bool use_vram = (slot >= 0 && slot < slot_count);
  const int src_id = use_vram ? slot : expert_id;

  const uint32_t* expert_weights = (use_vram ? dev_q_weight : host_q_weight) +
                                   (int64_t)src_id * expert_weight_stride;
  const uint8_t* expert_scales = (use_vram ? dev_scales : host_scales) +
                                 (int64_t)src_id * expert_scales_stride;
  const float global = (use_vram ? dev_globals : host_globals)[src_id];

  __shared__ T block_a[BLOCK_SIZE_M][BK + 8];
  const int offset_m_base = token_block * BLOCK_SIZE_M;
  load_activations<T, BLOCK_SIZE_M, TX, BK>(
      a, block_a, sorted_token_ids, offset_m_base, offset_k, end_k, t, size_m,
      size_k, top_k, gate_cap, up_cap);
  __syncthreads();
  if (n >= size_n) return;

  const uint32_t* b_ptr = expert_weights + (int64_t)(offset_k / 8) * size_n + n;
  const uint8_t* s_ptr = expert_scales + (int64_t)(offset_k / 16) * size_n + n;

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    const uint32_t sw0 = *(const uint32_t*)s_ptr;
    const uint32_t sw1 =
        *(const uint32_t*)((k + 16 < size_k) ? s_ptr + size_n : s_ptr);
    half2 sc[4][2];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      sc[j][0] = __float2half2_rn(e4m3_scale_to_f32((sw0 >> (8 * j)) & 0xFFu) *
                                  global * 16384.0f);
      sc[j][1] = __float2half2_rn(e4m3_scale_to_f32((sw1 >> (8 * j)) & 0xFFu) *
                                  global * 16384.0f);
    }
    s_ptr += 2 * size_n;

    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = load_int4_vec((const int4*)(b_ptr + j * size_n), use_vram);
    }
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      const half2 s_col0 = sc[0][j / 2], s_col1 = sc[1][j / 2];
      const half2 s_col2 = sc[2][j / 2], s_col3 = sc[3][j / 2];
      half2 dq[4];
      dequant_e2m1_8_fp16_fast((uint32_t)b_w[j].x, s_col0, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][0] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16_fast((uint32_t)b_w[j].y, s_col1, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][1] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16_fast((uint32_t)b_w[j].z, s_col2, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][2] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
      dequant_e2m1_8_fp16_fast((uint32_t)b_w[j].w, s_col3, dq);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        block_c[m][3] +=
            dot22_8_f(dq, reinterpret_cast<const half*>(&block_a[m][a_off]));
    }
    k += 32;
  }

  write_out<T, BLOCK_SIZE_M>(c, block_c, sorted_token_ids, topk_weights,
                             offset_m_base, size_m, top_k, size_n, n,
                             mul_topk_weight, output_topk);
}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
__global__ void moe_gemm_mxfp8_hybrid_kernel_rdna2(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ dev_q_weight,
    const uint8_t* __restrict__ dev_scales,
    const uint32_t* __restrict__ host_q_weight,
    const uint8_t* __restrict__ host_scales,
    const int32_t* __restrict__ resident_map, const int slot_count,
    const int num_experts, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk, const float gate_cap,
    const float up_cap) {
  static_assert(BK % 32 == 0, "BK must be a multiple of 32");
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * TX * 4;
  const int offset_k = blockIdx.z * BK;
  const int end_k = min(offset_k + BK, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int slot = resident_map[expert_id];
  const bool use_vram = (slot >= 0 && slot < slot_count);
  const int src_id = use_vram ? slot : expert_id;

  const uint32_t* expert_weights = (use_vram ? dev_q_weight : host_q_weight) +
                                   (int64_t)src_id * expert_weight_stride;
  const uint8_t* expert_scales = (use_vram ? dev_scales : host_scales) +
                                 (int64_t)src_id * expert_scales_stride;

  __shared__ __attribute__((aligned(16))) T block_a[BLOCK_SIZE_M][BK + 8];
  const int offset_m_base = token_block * BLOCK_SIZE_M;
  load_activations<T, BLOCK_SIZE_M, TX, BK>(
      a, block_a, sorted_token_ids, offset_m_base, offset_k, end_k, t, size_m,
      size_k, top_k, gate_cap, up_cap);
  __syncthreads();
  if (n >= size_n) return;

  const uint32_t* b_ptr = expert_weights + (int64_t)(offset_k / 4) * size_n + n;
  const uint8_t* s_ptr = expert_scales + (int64_t)(offset_k / 32) * size_n + n;

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    const uint32_t sw = *(const uint32_t*)s_ptr;
    half2 factor[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      factor[j] = e4m3_block_factor((sw >> (8 * j)) & 0xFFu);
    }
    s_ptr += size_n;

    uint32_t w[8][4];
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int4 v = load_int4_vec((const int4*)(b_ptr + i * size_n), use_vram);
      w[i][0] = (uint32_t)v.x;
      w[i][1] = (uint32_t)v.y;
      w[i][2] = (uint32_t)v.z;
      w[i][3] = (uint32_t)v.w;
    }
    b_ptr += 8 * size_n;

  #pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int a_off = (k - offset_k) + 8 * g;
      uint4 av[BLOCK_SIZE_M];
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m)
        av[m] = *reinterpret_cast<const uint4*>(&block_a[m][a_off]);
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        half2 dq[4];
        deq_e4m3_4(w[2 * g + 0][j], factor[j], dq + 0);
        deq_e4m3_4(w[2 * g + 1][j], factor[j], dq + 2);
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m)
          block_c[m][j] += dot2_8_regs(dq, av[m]);
      }
    }
    k += 32;
  }

  write_out<T, BLOCK_SIZE_M>(c, block_c, sorted_token_ids, topk_weights,
                             offset_m_base, size_m, top_k, size_n, n,
                             mul_topk_weight, output_topk);
}

}  // namespace lowp_offload

#else  // non-RDNA2 stubs for host/archdiff parity

namespace lowp_offload {

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
__global__ void moe_gemm_nvfp4_hybrid_kernel_rdna2(
    const T*, T*, const uint32_t*, const uint8_t*, const float*,
    const uint32_t*, const uint8_t*, const float*, const int32_t*, const int,
    const int, const float*, const int32_t*, const int32_t*, const int32_t*,
    const int, const int, const int, const int, const int, const int,
    const bool, const int, const float, const float) {}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
__global__ void moe_gemm_mxfp8_hybrid_kernel_rdna2(
    const T*, T*, const uint32_t*, const uint8_t*, const uint32_t*,
    const uint8_t*, const int32_t*, const int, const int, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const bool, const int,
    const float, const float) {}

}  // namespace lowp_offload

#endif

template <typename PtrT>
static PtrT pinned_host_dev_ptr(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(!t.is_cuda(), name, " must be a CPU pinned tensor");
  TORCH_CHECK(t.is_pinned(), name, " must be pinned host memory");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  void* dev_ptr = nullptr;
  hipError_t rc =
      hipHostGetDevicePointer(&dev_ptr, const_cast<void*>(t.data_ptr()), 0);
  TORCH_CHECK(rc == hipSuccess && dev_ptr != nullptr, name,
              " hipHostGetDevicePointer failed: ", hipGetErrorString(rc));
  return reinterpret_cast<PtrT>(dev_ptr);
}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
static void launch_moe_nvfp4_hybrid_cfg(
    const T* a, T* c, const uint32_t* dev_q_weight, const uint8_t* dev_scales,
    const float* dev_globals, const uint32_t* host_q_weight,
    const uint8_t* host_scales, const float* host_globals,
    const int32_t* resident_map, int slot_count, int num_experts,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, float gate_cap, float up_cap, hipStream_t stream) {
  dim3 block(TX);
  dim3 grid(num_token_blocks, (size_n + TX * 4 - 1) / (TX * 4),
            (size_k + BK - 1) / BK);
  lowp_offload::moe_gemm_nvfp4_hybrid_kernel_rdna2<T, BLOCK_SIZE_M, TX, BK>
      <<<grid, block, 0, stream>>>(
          a, c, dev_q_weight, dev_scales, dev_globals, host_q_weight,
          host_scales, host_globals, resident_map, slot_count, num_experts,
          topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
          size_m, size_n, size_k, top_k, expert_weight_stride,
          expert_scales_stride, mul_topk_weight, output_topk, gate_cap, up_cap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int BLOCK_SIZE_M>
static void launch_moe_nvfp4_hybrid(
    const T* a, T* c, const uint32_t* dev_q_weight, const uint8_t* dev_scales,
    const float* dev_globals, const uint32_t* host_q_weight,
    const uint8_t* host_scales, const float* host_globals,
    const int32_t* resident_map, int slot_count, int num_experts,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, float gate_cap, float up_cap, hipStream_t stream) {
  if constexpr (BLOCK_SIZE_M <= 4) {
    if constexpr (BLOCK_SIZE_M == 1) {
      if (num_token_blocks <= 8 && (size_k % 192 == 0) && (size_k % 256 != 0)) {
        launch_moe_nvfp4_hybrid_cfg<T, 1, 256, 192>(
            a, c, dev_q_weight, dev_scales, dev_globals, host_q_weight,
            host_scales, host_globals, resident_map, slot_count, num_experts,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            num_token_blocks, size_m, size_n, size_k, top_k,
            expert_weight_stride, expert_scales_stride, mul_topk_weight,
            output_topk, gate_cap, up_cap, stream);
        return;
      }
    }
    const int n_rem = size_n & 1023;
    if (gate_cap == 0.0f &&
        (n_rem == 128 ||
         (n_rem == 256 && (BLOCK_SIZE_M >= 2 || num_token_blocks >= 24)))) {
      launch_moe_nvfp4_hybrid_cfg<T, BLOCK_SIZE_M, 64, 256>(
          a, c, dev_q_weight, dev_scales, dev_globals, host_q_weight,
          host_scales, host_globals, resident_map, slot_count, num_experts,
          topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, top_k, expert_weight_stride,
          expert_scales_stride, mul_topk_weight, output_topk, gate_cap, up_cap,
          stream);
      return;
    }
  }
  launch_moe_nvfp4_hybrid_cfg<T, BLOCK_SIZE_M, OFFLOAD_THREADS_X,
                              OFFLOAD_BLOCK_KN_SIZE>(
      a, c, dev_q_weight, dev_scales, dev_globals, host_q_weight, host_scales,
      host_globals, resident_map, slot_count, num_experts, topk_weights,
      sorted_token_ids, expert_ids, num_tokens_post_padded, num_token_blocks,
      size_m, size_n, size_k, top_k, expert_weight_stride, expert_scales_stride,
      mul_topk_weight, output_topk, gate_cap, up_cap, stream);
}

static void moe_nvfp4_gemm_hybrid_impl(
    torch::Tensor a, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor dev_globals,
    torch::Tensor host_q_weight, torch::Tensor host_scales,
    torch::Tensor host_globals, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, float gate_cap, float up_cap) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(resident_map.is_cuda() && resident_map.is_contiguous(),
              "resident_map must be contiguous GPU tensor");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(dev_q_weight.is_cuda() && dev_q_weight.is_contiguous() &&
                  dev_q_weight.dim() == 3,
              "dev_q_weight must be contiguous 3D GPU tensor");
  TORCH_CHECK(dev_scales.is_cuda() && dev_scales.dim() == 3,
              "dev_scales must be 3D GPU tensor");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1) / (gate_cap > 0.0f ? 2 : 1);
  TORCH_CHECK(gate_cap == 0.0f ||
                  (a.size(1) % 2 == 0 && a.is_contiguous() && up_cap > 0.0f),
              "swiglu: a must be a contiguous [rows, 2 * K] tensor, caps > 0");
  const int size_n = (int)dev_q_weight.size(2);
  const int slot_count = (int)dev_q_weight.size(0);
  const int num_experts = (int)host_q_weight.size(0);
  TORCH_CHECK((int)resident_map.numel() == num_experts,
              "resident_map length must match num_experts");
  TORCH_CHECK((int)dev_q_weight.size(1) * 8 == size_k &&
                  (int)host_q_weight.size(1) * 8 == size_k &&
                  (int)host_q_weight.size(2) == size_n,
              "q_weight shape mismatch");
  TORCH_CHECK((int)dev_scales.size(1) * 16 == size_k &&
                  (int)dev_scales.size(2) == size_n &&
                  (int)host_scales.size(1) * 16 == size_k &&
                  (int)host_scales.size(2) == size_n,
              "scales shape mismatch");
  TORCH_CHECK(
      dev_globals.numel() == slot_count && host_globals.numel() == num_experts,
      "globals length mismatch");
  TORCH_CHECK(size_n % 8 == 0 && size_k % 32 == 0, "N % 8 and K % 32 required");

  const uint32_t* host_qw_ptr =
      pinned_host_dev_ptr<const uint32_t*>(host_q_weight, "host_q_weight");
  const uint8_t* host_sc_ptr =
      pinned_host_dev_ptr<const uint8_t*>(host_scales, "host_scales");
  const float* host_gl_ptr =
      pinned_host_dev_ptr<const float*>(host_globals, "host_globals");

  const int expert_weight_stride = (int)(dev_q_weight.size(1) * size_n);
  const int expert_scales_stride = (int)(dev_scales.size(1) * size_n);
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

#define LAUNCH_NV_HYB(BM)                                                      \
  launch_moe_nvfp4_hybrid<half, BM>(                                           \
      (const half*)a.data_ptr(), (half*)c.data_ptr(),                          \
      reinterpret_cast<const uint32_t*>(dev_q_weight.data_ptr()),              \
      (const uint8_t*)dev_scales.data_ptr(), dev_globals.data_ptr<float>(),    \
      host_qw_ptr, host_sc_ptr, host_gl_ptr, resident_map.data_ptr<int32_t>(), \
      slot_count, num_experts, topk_w_ptr,                                     \
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),    \
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,    \
      size_n, size_k, (int)top_k, expert_weight_stride, expert_scales_stride,  \
      mul_topk_weight, (int)output_topk, gate_cap, up_cap, stream)
  switch (block_size_m) {
    case 1:
      LAUNCH_NV_HYB(1);
      break;
    case 2:
      LAUNCH_NV_HYB(2);
      break;
    case 4:
      LAUNCH_NV_HYB(4);
      break;
    case 8:
      LAUNCH_NV_HYB(8);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_nvfp4_gemm_hybrid_rdna2: block_size_m must be 1,2,4,8");
  }
#undef LAUNCH_NV_HYB
}

void moe_nvfp4_gemm_hybrid_rdna2(
    torch::Tensor a, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor dev_globals,
    torch::Tensor host_q_weight, torch::Tensor host_scales,
    torch::Tensor host_globals, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk) {
  moe_nvfp4_gemm_hybrid_impl(
      a, c, dev_q_weight, dev_scales, dev_globals, host_q_weight, host_scales,
      host_globals, resident_map, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_padded, top_k, block_size_m, mul_topk_weight, output_topk,
      0.0f, 0.0f);
}

void moe_nvfp4_gemm_swiglu_hybrid_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor dev_globals,
    torch::Tensor host_q_weight, torch::Tensor host_scales,
    torch::Tensor host_globals, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, double gate_cap, double up_cap) {
  TORCH_CHECK(gate_cap > 0.0 && up_cap > 0.0, "softcaps must be > 0");
  moe_nvfp4_gemm_hybrid_impl(
      gate_up, c, dev_q_weight, dev_scales, dev_globals, host_q_weight,
      host_scales, host_globals, resident_map, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, top_k, block_size_m, mul_topk_weight,
      output_topk, (float)gate_cap, (float)up_cap);
}

template <typename T, int BLOCK_SIZE_M, int TX = OFFLOAD_THREADS_X,
          int BK = OFFLOAD_BLOCK_KN_SIZE>
static void launch_moe_mxfp8_hybrid_cfg(
    const T* a, T* c, const uint32_t* dev_q_weight, const uint8_t* dev_scales,
    const uint32_t* host_q_weight, const uint8_t* host_scales,
    const int32_t* resident_map, int slot_count, int num_experts,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, float gate_cap, float up_cap, hipStream_t stream) {
  dim3 block(TX);
  dim3 grid(num_token_blocks, (size_n + TX * 4 - 1) / (TX * 4),
            (size_k + BK - 1) / BK);
  lowp_offload::moe_gemm_mxfp8_hybrid_kernel_rdna2<T, BLOCK_SIZE_M, TX, BK>
      <<<grid, block, 0, stream>>>(
          a, c, dev_q_weight, dev_scales, host_q_weight, host_scales,
          resident_map, slot_count, num_experts, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, top_k,
          expert_weight_stride, expert_scales_stride, mul_topk_weight,
          output_topk, gate_cap, up_cap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int BLOCK_SIZE_M>
static void launch_moe_mxfp8_hybrid(
    const T* a, T* c, const uint32_t* dev_q_weight, const uint8_t* dev_scales,
    const uint32_t* host_q_weight, const uint8_t* host_scales,
    const int32_t* resident_map, int slot_count, int num_experts,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, float gate_cap, float up_cap, hipStream_t stream) {
  if constexpr (BLOCK_SIZE_M == 1) {
    if (num_token_blocks <= 8 && (size_k % 192 == 0) && (size_k % 256 != 0)) {
      launch_moe_mxfp8_hybrid_cfg<T, 1, 256, 192>(
          a, c, dev_q_weight, dev_scales, host_q_weight, host_scales,
          resident_map, slot_count, num_experts, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, gate_cap, up_cap, stream);
      return;
    }
  }
  launch_moe_mxfp8_hybrid_cfg<T, BLOCK_SIZE_M, OFFLOAD_THREADS_X,
                              OFFLOAD_BLOCK_KN_SIZE>(
      a, c, dev_q_weight, dev_scales, host_q_weight, host_scales, resident_map,
      slot_count, num_experts, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_padded, num_token_blocks, size_m, size_n, size_k, top_k,
      expert_weight_stride, expert_scales_stride, mul_topk_weight, output_topk,
      gate_cap, up_cap, stream);
}

static void moe_mxfp8_gemm_hybrid_impl(
    torch::Tensor a, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor host_q_weight,
    torch::Tensor host_scales, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, float gate_cap, float up_cap) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda(), "a and c must be GPU tensors");
  TORCH_CHECK(resident_map.is_cuda() && resident_map.is_contiguous(),
              "resident_map must be contiguous GPU tensor");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(dev_q_weight.is_cuda() && dev_q_weight.is_contiguous() &&
                  dev_q_weight.dim() == 3,
              "dev_q_weight must be contiguous 3D GPU tensor");
  TORCH_CHECK(dev_scales.is_cuda() && dev_scales.dim() == 3,
              "dev_scales must be 3D GPU tensor");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_m = (int)a.size(0);
  const int size_k = (int)a.size(1) / (gate_cap > 0.0f ? 2 : 1);
  TORCH_CHECK(gate_cap == 0.0f ||
                  (a.size(1) % 2 == 0 && a.is_contiguous() && up_cap > 0.0f),
              "swiglu: a must be a contiguous [rows, 2 * K] tensor, caps > 0");
  const int size_n = (int)dev_q_weight.size(2);
  const int slot_count = (int)dev_q_weight.size(0);
  const int num_experts = (int)host_q_weight.size(0);
  TORCH_CHECK((int)resident_map.numel() == num_experts,
              "resident_map length must match num_experts");
  TORCH_CHECK((int)dev_q_weight.size(1) * 4 == size_k &&
                  (int)host_q_weight.size(1) * 4 == size_k &&
                  (int)host_q_weight.size(2) == size_n,
              "q_weight shape mismatch");
  TORCH_CHECK((int)dev_scales.size(1) * 32 == size_k &&
                  (int)dev_scales.size(2) == size_n &&
                  (int)host_scales.size(1) * 32 == size_k &&
                  (int)host_scales.size(2) == size_n,
              "scales shape mismatch");
  TORCH_CHECK(size_n % 8 == 0 && size_k % 32 == 0, "N % 8 and K % 32 required");

  const uint32_t* host_qw_ptr =
      pinned_host_dev_ptr<const uint32_t*>(host_q_weight, "host_q_weight");
  const uint8_t* host_sc_ptr =
      pinned_host_dev_ptr<const uint8_t*>(host_scales, "host_scales");

  const int expert_weight_stride = (int)(dev_q_weight.size(1) * size_n);
  const int expert_scales_stride = (int)(dev_scales.size(1) * size_n);
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

#define LAUNCH_MX8_HYB(BM)                                                    \
  launch_moe_mxfp8_hybrid<half, BM>(                                          \
      (const half*)a.data_ptr(), (half*)c.data_ptr(),                         \
      reinterpret_cast<const uint32_t*>(dev_q_weight.data_ptr()),             \
      (const uint8_t*)dev_scales.data_ptr(), host_qw_ptr, host_sc_ptr,        \
      resident_map.data_ptr<int32_t>(), slot_count, num_experts, topk_w_ptr,  \
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),   \
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,   \
      size_n, size_k, (int)top_k, expert_weight_stride, expert_scales_stride, \
      mul_topk_weight, (int)output_topk, gate_cap, up_cap, stream)
  switch (block_size_m) {
    case 1:
      LAUNCH_MX8_HYB(1);
      break;
    case 2:
      LAUNCH_MX8_HYB(2);
      break;
    case 4:
      LAUNCH_MX8_HYB(4);
      break;
    case 8:
      LAUNCH_MX8_HYB(8);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_mxfp8_gemm_hybrid_rdna2: block_size_m must be 1,2,4,8");
  }
#undef LAUNCH_MX8_HYB
}

void moe_mxfp8_gemm_hybrid_rdna2(
    torch::Tensor a, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor host_q_weight,
    torch::Tensor host_scales, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk) {
  moe_mxfp8_gemm_hybrid_impl(
      a, c, dev_q_weight, dev_scales, host_q_weight, host_scales, resident_map,
      topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k,
      block_size_m, mul_topk_weight, output_topk, 0.0f, 0.0f);
}

void moe_mxfp8_gemm_swiglu_hybrid_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor dev_q_weight,
    torch::Tensor dev_scales, torch::Tensor host_q_weight,
    torch::Tensor host_scales, torch::Tensor resident_map,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, double gate_cap, double up_cap) {
  TORCH_CHECK(gate_cap > 0.0 && up_cap > 0.0, "softcaps must be > 0");
  moe_mxfp8_gemm_hybrid_impl(
      gate_up, c, dev_q_weight, dev_scales, host_q_weight, host_scales,
      resident_map, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_padded, top_k, block_size_m, mul_topk_weight, output_topk,
      (float)gate_cap, (float)up_cap);
}
