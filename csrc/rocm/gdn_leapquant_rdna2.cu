// LeapQuant INT8 recurrent-state storage for GatedDeltaNet on AMD RDNA2
// (gfx1030). arXiv 2609.38166; layout and FP32 oracle:
// vllm/model_executor/layers/mamba/ops/leapquant_gdn.py.
//
// One uint8 blob per (cache block, value head) holds an INT8 residual with
// per-value-row scales and per-key-column smoothing, r FP16 compensator
// tokens, a ring of FP16 per-token records (k_j, u_j, g_j) and an int32
// header [cursor, pending]. The FP32 state is
//   S_c = replay(records[0:c], Z*B*C + U~^T K~),
//   replay step j: S <- exp(g_j) * S + u_j k_j^T.
//
// Kernels (all launched on the current stream, cudagraph-safe):
//   gdn_lq_normalize_rdna2  commit `num_accepted` pending records (one for a
//                           slot-0 reader: num_accepted = None), fold full
//                           windows into a new boundary (one WG per head).
//                           With layer_offsets it covers every layer of a
//                           group in one launch (grid.y = layer).
//   gdn_lq_decode_rdna2     decode / MTP verify on normalized blobs: replay,
//                           run T tokens, append T records, pending = T.
//   gdn_lq_materialize_rdna2  normalized blob -> FP32 [N, HV, V, K].
//   gdn_lq_store_rdna2      FP32 [N, HV, V, K] -> blob (fold, cursor 0).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "gdn_leapquant_layout.cuh"

namespace {

using namespace rdna2_lq;

constexpr int LQ_BV = 32;        // V rows per decode workgroup
constexpr int LQ_THREADS = 256;  // decode: 32 v-rows x 8 k-slices
constexpr int LQ_R = 4;          // compensator tokens (paper: r = 4 at 8 bits)
constexpr float LQ_INT8_MAX = 127.0f;
constexpr float LQ_SMOOTH_FLOOR = 1e-6f;
constexpr float LQ_ORTH_EPS = 1e-12f;
constexpr float LQ_FP16_MAX = 65504.0f;
constexpr float LQ_L2NORM_EPS = 1e-6f;
constexpr float LQ_SOFTPLUS_THRESHOLD = 20.0f;
constexpr float LQ_LOG2E = 0x1.715476p+0f;

__device__ __forceinline__ float lq_exp(float x) { return exp2f(x * LQ_LOG2E); }

__device__ __forceinline__ float lq_ksum(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

__device__ __forceinline__ float lq_kmax(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffffffffffULL, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffffffffffULL, x, 2));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffffffffffULL, x, 4));
  return x;
}

__device__ __forceinline__ float lq_wsum(float x) {
#pragma unroll
  for (int m = 1; m < 32; m <<= 1)
    x += __shfl_xor_sync(0xffffffffffffffffULL, x, m);
  return x;
}

__device__ __forceinline__ float lq_f32(float x) { return x; }
__device__ __forceinline__ float lq_f32(__half x) { return __half2float(x); }

// ---------------------------------------------------------------------------
// Fold: one workgroup (256 threads) per (request, head). Thread t owns rows
// v = vg + 32 * i (i < RPT = V / 32, vg = t >> 3) and columns
// k0 = (t & 7) * 16 .. k0 + 15. Every reduction is a fixed-order tree, so a
// fold is bitwise reproducible run to run.
// ---------------------------------------------------------------------------

template <int RPT>
struct LqFoldSmem {
  float red[8][LQ_K * LQ_R];  // per-wave partial sums over v
  float kr[LQ_K * LQ_R];      // [k][r] (X0, then S^T Q)
  float vr[RPT * 32 * LQ_R];  // [v][r] (Y, then Q)
  float csm[LQ_K];
  int flag;
};

// Sum over all v of part[j][c] (j < 16 local k, c < NCOL) into
// out[k * NCOL + c]; part is per thread, already summed over its RPT rows.
template <int RPT, int NCOL>
__device__ void lq_reduce_over_v(LqFoldSmem<RPT>& sm,
                                 const float (&part)[16][NCOL], float* out) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int k0 = (tid & 7) * 16;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
      float x = part[j][c];
      x += __shfl_xor_sync(0xffffffffffffffffULL, x, 8);
      x += __shfl_xor_sync(0xffffffffffffffffULL, x, 16);
      if (lane < 8) sm.red[wave][(k0 + j) * NCOL + c] = x;
    }
  }
  __syncthreads();
  for (int e = tid; e < LQ_K * NCOL; e += LQ_THREADS) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += sm.red[w][e];
    out[e] = s;
  }
  __syncthreads();
}

// Modified Gram-Schmidt over the LQ_R columns of vr ([V][r]) by wave 0.
template <int RPT>
__device__ void lq_orthonormalize(LqFoldSmem<RPT>& sm) {
  const int tid = threadIdx.x;
  if (tid < 32) {
#pragma unroll
    for (int c = 0; c < LQ_R; ++c) {
#pragma unroll
      for (int p = 0; p < c; ++p) {
        float d = 0.0f;
#pragma unroll
        for (int e = 0; e < RPT; ++e) {
          const int v = tid + 32 * e;
          d += sm.vr[v * LQ_R + c] * sm.vr[v * LQ_R + p];
        }
        d = lq_wsum(d);
#pragma unroll
        for (int e = 0; e < RPT; ++e) {
          const int v = tid + 32 * e;
          sm.vr[v * LQ_R + c] -= d * sm.vr[v * LQ_R + p];
        }
      }
      float n2 = 0.0f;
#pragma unroll
      for (int e = 0; e < RPT; ++e) {
        const float y = sm.vr[(tid + 32 * e) * LQ_R + c];
        n2 += y * y;
      }
      const float n = sqrtf(lq_wsum(n2));
      const float inv = n > LQ_ORTH_EPS ? 1.0f / n : 0.0f;
#pragma unroll
      for (int e = 0; e < RPT; ++e) sm.vr[(tid + 32 * e) * LQ_R + c] *= inv;
    }
  }
  __syncthreads();
}

// vr[v][c] = sum_k S[v][k] * kr[k][c]
template <int RPT>
__device__ void lq_apply_s(LqFoldSmem<RPT>& sm, const float (&S)[RPT][16]) {
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  const int k0 = (tid & 7) * 16;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
#pragma unroll
    for (int c = 0; c < LQ_R; ++c) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) acc += S[i][j] * sm.kr[(k0 + j) * LQ_R + c];
      acc = lq_ksum(acc);
      if ((tid & 7) == 0) sm.vr[(vg + 32 * i) * LQ_R + c] = acc;
    }
  }
  __syncthreads();
}

// kr[k][c] = sum_v S[v][k] * vr[v][c]
template <int RPT>
__device__ void lq_apply_st(LqFoldSmem<RPT>& sm, const float (&S)[RPT][16]) {
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  float part[16][LQ_R];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
#pragma unroll
    for (int c = 0; c < LQ_R; ++c) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < RPT; ++i)
        acc += S[i][j] * sm.vr[(vg + 32 * i) * LQ_R + c];
      part[j][c] = acc;
    }
  }
  lq_reduce_over_v<RPT, LQ_R>(sm, part, sm.kr);
}

// Fold the FP32 state held in registers into the blob (cursor = pending = 0).
// warm: start the subspace iteration from the blob's previous compensator
// keys (a live window boundary); otherwise (a freshly allocated block whose
// bytes are stale) from the fixed integer block.
template <int RPT>
__device__ void lq_fold(LqFoldSmem<RPT>& sm, float (&S)[RPT][16], uint8_t* blob,
                        const LqLayout& L, bool warm) {
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  const int ks = tid & 7;
  const int k0 = ks * 16;
  const int V = L.V;

  const __half* pck = lq_ptr<__half>(blob, L.comp_k);
  int nz = 0;
  if (warm) {
    for (int e = tid; e < LQ_R * LQ_K; e += LQ_THREADS) {
      const float x = __half2float(pck[e]);
      sm.kr[(e % LQ_K) * LQ_R + e / LQ_K] = x;
      nz |= (x != 0.0f);
    }
  }
  nz = __syncthreads_or(nz);
  if (!nz) {
    for (int e = tid; e < LQ_R * LQ_K; e += LQ_THREADS) {
      const int k = e % LQ_K, c = e / LQ_K;
      sm.kr[k * LQ_R + c] = (float)(((k * (2 * c + 3) + 7 * c + 1) % 17) - 8);
    }
    __syncthreads();
  }

  lq_apply_s<RPT>(sm, S);
  lq_orthonormalize<RPT>(sm);
  for (int it = 0; it < L.power_iters; ++it) {
    lq_apply_st<RPT>(sm, S);
    lq_apply_s<RPT>(sm, S);
    lq_orthonormalize<RPT>(sm);
  }
  lq_apply_st<RPT>(sm, S);  // kr = K~ (S^T Q), vr = Q = U~

  // Round compensators to fp16, store them, keep the rounded values.
  __half* ck = lq_ptr<__half>(blob, L.comp_k);
  __half* cu = lq_ptr<__half>(blob, L.comp_u);
  for (int e = tid; e < LQ_R * LQ_K; e += LQ_THREADS) {
    const int k = e % LQ_K, c = e / LQ_K;
    const __half h = __float2half(
        fminf(fmaxf(sm.kr[k * LQ_R + c], -LQ_FP16_MAX), LQ_FP16_MAX));
    ck[c * LQ_K + k] = h;
    sm.kr[k * LQ_R + c] = __half2float(h);
  }
  for (int e = tid; e < LQ_R * V; e += LQ_THREADS) {
    const int v = e % V, c = e / V;
    const __half h = __float2half(
        fminf(fmaxf(sm.vr[v * LQ_R + c], -LQ_FP16_MAX), LQ_FP16_MAX));
    cu[c * V + v] = h;
    sm.vr[v * LQ_R + c] = __half2float(h);
  }
  __syncthreads();

  // Residual R = S - U~ K~^T (in place) and per-key smoothing.
  float part[16][1];
#pragma unroll
  for (int j = 0; j < 16; ++j) part[j][0] = 0.0f;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int v = vg + 32 * i;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float c_sum = 0.0f;
#pragma unroll
      for (int c = 0; c < LQ_R; ++c)
        c_sum += sm.vr[v * LQ_R + c] * sm.kr[(k0 + j) * LQ_R + c];
      S[i][j] -= c_sum;
      part[j][0] += fabsf(S[i][j]);
    }
  }
  lq_reduce_over_v<RPT, 1>(sm, part, sm.csm);
  float* csm_out = lq_ptr<float>(blob, L.csmooth);
  for (int k = tid; k < LQ_K; k += LQ_THREADS) {
    const float c = fmaxf(sqrtf(sm.csm[k] / (float)V), LQ_SMOOTH_FLOOR);
    sm.csm[k] = c;
    csm_out[k] = c;
  }
  __syncthreads();

  // Per-value-row scale and INT8 payload.
  float* bsc = lq_ptr<float>(blob, L.bscale);
  int8_t* z = lq_ptr<int8_t>(blob, L.z);
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int v = vg + 32 * i;
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      S[i][j] /= sm.csm[k0 + j];
      m = fmaxf(m, fabsf(S[i][j]));
    }
    const float bscale = lq_kmax(m) / LQ_INT8_MAX;
    if (ks == 0) bsc[v] = bscale;
    const float inv = bscale > 0.0f ? 1.0f / bscale : 0.0f;
    union {
      int8_t c[16];
      uint4 u;
    } pk;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float q =
          fminf(fmaxf(rintf(S[i][j] * inv), -LQ_INT8_MAX), LQ_INT8_MAX);
      pk.c[j] = (int8_t)q;
    }
    *reinterpret_cast<uint4*>(z + (long)v * LQ_K + k0) = pk.u;
  }
  if (tid == 0) {
    int* hdr = lq_ptr<int>(blob, L.header);
    hdr[0] = 0;
    hdr[1] = 0;
  }
}

// Boundary state + replay of c records into this thread's [RPT][16] tile.
template <int RPT>
__device__ void lq_load_full(float (&S)[RPT][16], uint8_t* blob,
                             const LqLayout& L, int c) {
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  const int k0 = (tid & 7) * 16;
  const int8_t* z = lq_ptr<int8_t>(blob, L.z);
  const float* bsc = lq_ptr<float>(blob, L.bscale);
  const float* csm = lq_ptr<float>(blob, L.csmooth);
  const __half* ck = lq_ptr<__half>(blob, L.comp_k);
  const __half* cu = lq_ptr<__half>(blob, L.comp_u);
  float cs[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) cs[j] = csm[k0 + j];
  for (int i = 0; i < RPT; ++i) {
    const int v = vg + 32 * i;
    const float bs = bsc[v];
    union {
      int8_t c[16];
      uint4 u;
    } pk;
    pk.u = *reinterpret_cast<const uint4*>(z + (long)v * LQ_K + k0);
#pragma unroll
    for (int j = 0; j < 16; ++j) S[i][j] = (float)pk.c[j] * bs * cs[j];
    for (int cc = 0; cc < L.r; ++cc) {
      const float u = __half2float(cu[cc * L.V + v]);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        S[i][j] += u * __half2float(ck[cc * LQ_K + k0 + j]);
    }
  }
  const __half* rk = lq_ptr<__half>(blob, L.rec_k);
  const __half* ru = lq_ptr<__half>(blob, L.rec_u);
  const float* rg = lq_ptr<float>(blob, L.rec_g);
#pragma unroll 2
  for (int t = 0; t < c; ++t) {
    const float decay = lq_exp(rg[t]);
    float kk[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) kk[j] = __half2float(rk[t * LQ_K + k0 + j]);
    for (int i = 0; i < RPT; ++i) {
      const float u = __half2float(ru[t * L.V + vg + 32 * i]);
#pragma unroll
      for (int j = 0; j < 16; ++j) S[i][j] = S[i][j] * decay + u * kk[j];
    }
  }
}

template <int RPT>
__global__ void __launch_bounds__(LQ_THREADS)
    gdn_lq_normalize_kernel(uint8_t* __restrict__ blobs,
                            const int64_t* __restrict__ layer_offsets,
                            const int* __restrict__ state_indices,
                            long idx_stride,
                            const int* __restrict__ num_accepted,
                            long stride_block, long stride_head,
                            long num_blocks, int HV, LqLayout L) {
  __shared__ LqFoldSmem<RPT> sm;
  const int i_n = blockIdx.x / HV;
  const int i_hv = blockIdx.x % HV;
  const long idx = state_indices[(long)i_n * idx_stride];
  if (idx <= 0 || idx >= num_blocks) return;
  if (layer_offsets != nullptr) blobs += layer_offsets[blockIdx.y];
  uint8_t* blob = blobs + idx * stride_block + (long)i_hv * stride_head;
  int* hdr = lq_ptr<int>(blob, L.header);
  if (threadIdx.x == 0) {
    int cursor = hdr[0];
    int pending = hdr[1];
    if (!lq_header_ok(cursor, pending, L)) {
      cursor = 0;
      pending = 0;
    }
    int acc = 0;
    if (pending > 0) {
      acc = min(num_accepted != nullptr ? num_accepted[i_n] : 1, pending);
      acc = max(acc, 0);
    }
    sm.flag = cursor + acc;
  }
  __syncthreads();
  const int c = sm.flag;
  if (c < L.window) {
    if (threadIdx.x == 0) {
      hdr[0] = c;
      hdr[1] = 0;
    }
    return;
  }
  float S[RPT][16];
  lq_load_full<RPT>(S, blob, L, c);
  __syncthreads();
  lq_fold<RPT>(sm, S, blob, L, /*warm=*/true);
}

template <int RPT>
__global__ void __launch_bounds__(LQ_THREADS)
    gdn_lq_store_kernel(const float* __restrict__ state,  // [N, HV, V, K]
                        uint8_t* __restrict__ blobs,
                        const int* __restrict__ state_indices, long idx_stride,
                        long stride_state_n, long stride_state_h,
                        long stride_block, long stride_head, long num_blocks,
                        int HV, LqLayout L) {
  __shared__ LqFoldSmem<RPT> sm;
  const int i_n = blockIdx.x / HV;
  const int i_hv = blockIdx.x % HV;
  const long idx = state_indices[(long)i_n * idx_stride];
  if (idx <= 0 || idx >= num_blocks) return;
  uint8_t* blob = blobs + idx * stride_block + (long)i_hv * stride_head;
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  const int k0 = (tid & 7) * 16;
  const float* src = state + i_n * stride_state_n + i_hv * stride_state_h;
  float S[RPT][16];
  for (int i = 0; i < RPT; ++i) {
    const float4* p =
        reinterpret_cast<const float4*>(src + (long)(vg + 32 * i) * LQ_K + k0);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 x = p[j];
      S[i][4 * j] = x.x;
      S[i][4 * j + 1] = x.y;
      S[i][4 * j + 2] = x.z;
      S[i][4 * j + 3] = x.w;
    }
  }
  lq_fold<RPT>(sm, S, blob, L, /*warm=*/false);
}

template <int RPT>
__global__ void __launch_bounds__(LQ_THREADS)
    gdn_lq_materialize_kernel(const uint8_t* __restrict__ blobs,
                              float* __restrict__ state,
                              const int* __restrict__ state_indices,
                              long idx_stride, long stride_state_n,
                              long stride_state_h, long stride_block,
                              long stride_head, long num_blocks, int HV,
                              LqLayout L) {
  const int i_n = blockIdx.x / HV;
  const int i_hv = blockIdx.x % HV;
  const long idx = state_indices[(long)i_n * idx_stride];
  const int tid = threadIdx.x;
  const int vg = tid >> 3;
  const int k0 = (tid & 7) * 16;
  float* dst = state + i_n * stride_state_n + i_hv * stride_state_h;
  float S[RPT][16];
  if (idx <= 0 || idx >= num_blocks) {
    for (int i = 0; i < RPT; ++i)
#pragma unroll
      for (int j = 0; j < 16; ++j) S[i][j] = 0.0f;
  } else {
    uint8_t* blob = const_cast<uint8_t*>(blobs) + idx * stride_block +
                    (long)i_hv * stride_head;
    const int* hdr = lq_ptr<int>(blob, L.header);
    lq_load_full<RPT>(S, blob, L, lq_header_ok(hdr[0], hdr[1], L) ? hdr[0] : 0);
  }
  for (int i = 0; i < RPT; ++i) {
    float4* p =
        reinterpret_cast<float4*>(dst + (long)(vg + 32 * i) * LQ_K + k0);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      p[j] = make_float4(S[i][4 * j], S[i][4 * j + 1], S[i][4 * j + 2],
                         S[i][4 * j + 3]);
  }
}

// ---------------------------------------------------------------------------
// Decode / verify: one workgroup per (V-tile of 32 rows, request x head),
// lane = v_local * 8 + k-slice, 16 fp32 state registers per thread, the same
// tiling and op order as gdn_spec_verify_rdna2.
// ---------------------------------------------------------------------------

template <typename AT, typename DT>
__global__ void __launch_bounds__(LQ_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_lq_decode_kernel(
        const __half* __restrict__ q, const __half* __restrict__ k,
        const __half* __restrict__ v, const __half* __restrict__ a,
        const __half* __restrict__ b, const AT* __restrict__ A_log,
        const DT* __restrict__ dt_bias, __half* __restrict__ out,
        uint8_t* __restrict__ blobs, const int* __restrict__ state_indices,
        long idx_stride, const int* __restrict__ cu_seqlens, int H, int HV,
        long stride_q_tok, long stride_k_tok, long stride_v_tok,
        long stride_a_tok, long stride_b_tok, long stride_o_tok,
        long stride_block, long stride_head, long num_blocks, float scale,
        LqLayout L) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / HV;
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / (HV / H);
  const int V = L.V;

  const int lane = threadIdx.x;
  const int v_local = lane >> 3;
  const int ks = lane & 7;
  const int o_v = i_v * LQ_BV + v_local;
  const int k0 = ks * 16;

  const int bos = cu_seqlens[i_n];
  const int n_tok = cu_seqlens[i_n + 1] - bos;
  if (n_tok <= 0) return;

  const long idx = state_indices[(long)i_n * idx_stride];
  if (idx <= 0 || idx >= num_blocks) {
    for (int t = 0; t < n_tok; ++t)
      out[(long)(bos + t) * stride_o_tok + (long)i_hv * V + o_v] =
          __float2half(0.0f);
    return;
  }
  uint8_t* blob = blobs + idx * stride_block + (long)i_hv * stride_head;
  int* hdr = lq_ptr<int>(blob, L.header);
  const int c = lq_header_ok(hdr[0], hdr[1], L) ? hdr[0] : 0;

  // Boundary state tile.
  float h[16];
  {
    const float bs = lq_ptr<float>(blob, L.bscale)[o_v];
    const float* csm = lq_ptr<float>(blob, L.csmooth);
    union {
      int8_t c8[16];
      uint4 u;
    } pk;
    pk.u = *reinterpret_cast<const uint4*>(lq_ptr<int8_t>(blob, L.z) +
                                           (long)o_v * LQ_K + k0);
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = (float)pk.c8[j] * bs * csm[k0 + j];
    const __half* ck = lq_ptr<__half>(blob, L.comp_k);
    const __half* cu = lq_ptr<__half>(blob, L.comp_u);
    for (int cc = 0; cc < L.r; ++cc) {
      const float u = __half2float(cu[cc * V + o_v]);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        h[j] += u * __half2float(ck[cc * LQ_K + k0 + j]);
    }
  }
  __half* rk = lq_ptr<__half>(blob, L.rec_k);
  __half* ru = lq_ptr<__half>(blob, L.rec_u);
  float* rg = lq_ptr<float>(blob, L.rec_g);
#pragma unroll 2
  for (int t = 0; t < c; ++t) {
    const float decay = lq_exp(rg[t]);
    const float u = __half2float(ru[t * V + o_v]);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      h[j] = h[j] * decay + u * __half2float(rk[t * LQ_K + k0 + j]);
  }

  const int n_rec = min(n_tok, L.cap - c);
  for (int t = 0; t < n_tok; ++t) {
    const long g = bos + t;
    const __half* p_q = q + g * stride_q_tok + (long)i_h * LQ_K + k0;
    const __half* p_k = k + g * stride_k_tok + (long)i_h * LQ_K + k0;
    float qf[16], kf[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      qf[j] = __half2float(p_q[j]);
      kf[j] = __half2float(p_k[j]);
    }
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      sq += qf[j] * qf[j];
      sk += kf[j] * kf[j];
    }
    sq = lq_ksum(sq);
    sk = lq_ksum(sk);
    const float q_norm = __frsqrt_rn(sq + LQ_L2NORM_EPS);
    const float k_norm = __frsqrt_rn(sk + LQ_L2NORM_EPS);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      qf[j] = qf[j] * q_norm * scale;
      kf[j] = kf[j] * k_norm;
    }
    const float a_val = __half2float(a[g * stride_a_tok + i_hv]);
    const float b_val = __half2float(b[g * stride_b_tok + i_hv]);
    const float x = a_val + lq_f32(dt_bias[i_hv]);
    const float softplus =
        (x <= LQ_SOFTPLUS_THRESHOLD) ? logf(1.0f + lq_exp(x)) : x;
    const float g_val = -lq_exp(lq_f32(A_log[i_hv])) * softplus;
    const float beta = 1.0f / (1.0f + lq_exp(-b_val));
    const float decay = lq_exp(g_val);
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] *= decay;
    const float v_val =
        __half2float(v[g * stride_v_tok + (long)i_hv * V + o_v]);
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) acc += h[j] * kf[j];
    acc = lq_ksum(acc);
    const float u = (v_val - acc) * beta;
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] += u * kf[j];
    float o_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc += h[j] * qf[j];
    o_acc = lq_ksum(o_acc);
    out[g * stride_o_tok + (long)i_hv * V + o_v] = __float2half(o_acc);

    if (t < n_rec) {
      const int pos = c + t;
      ru[pos * V + o_v] = __float2half(u);
      if (i_v == 0 && v_local == 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          rk[pos * LQ_K + k0 + j] = __float2half(kf[j]);
        if (ks == 0) rg[pos] = g_val;
      }
    }
  }
  if (i_v == 0 && lane == 0) hdr[1] = n_rec;
}

LqLayout make_layout(int V, int64_t num_comp, int64_t capacity, int64_t window,
                     int64_t power_iters) {
  TORCH_CHECK(num_comp == LQ_R,
              "LeapQuant GDN is built for num_comp == ", LQ_R);
  TORCH_CHECK(window >= 1 && capacity >= window,
              "LeapQuant needs capacity >= window >= 1");
  TORCH_CHECK(power_iters >= 0, "power_iters must be >= 0");
  LqLayout L;
  L.init(V, (int)num_comp, (int)capacity, (int)window, (int)power_iters);
  return L;
}

void check_blob(const torch::Tensor& blob, const LqLayout& L, int64_t HV) {
  TORCH_CHECK(
      blob.is_cuda() && blob.scalar_type() == at::kByte && blob.dim() == 3,
      "LeapQuant blob must be CUDA uint8 [blocks, HV, nbytes]");
  TORCH_CHECK(
      blob.size(1) == HV && blob.size(2) == L.total && blob.stride(2) == 1,
      "LeapQuant blob shape mismatch: expected [blocks, ", HV, ", ", L.total,
      "] with a dense last dim");
  TORCH_CHECK(blob.stride(0) % LQ_ALIGN == 0 && blob.stride(1) % LQ_ALIGN == 0,
              "LeapQuant blob strides must be 16-byte multiples");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(blob.data_ptr()) % LQ_ALIGN == 0,
              "LeapQuant blob must be 16-byte aligned");
}

// Index vectors may be strided column views (spec_state_indices[:, 0]).
void check_indices(const torch::Tensor& idx, int64_t N) {
  TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == at::kInt &&
                  idx.dim() == 1 && idx.size(0) >= N,
              "state_indices must be CUDA int32 [N]");
}

#define LQ_DISPATCH_RPT(V, ...)                                       \
  switch ((V) / 32) {                                                 \
    case 4: {                                                         \
      constexpr int RPT = 4;                                          \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      TORCH_CHECK(false, "LeapQuant GDN supports head_v_dim == 128"); \
  }

}  // namespace

void gdn_lq_normalize_rdna2(torch::Tensor blob, torch::Tensor state_indices,
                            const std::optional<torch::Tensor>& num_accepted,
                            int64_t head_v_dim, int64_t num_comp,
                            int64_t capacity, int64_t window,
                            int64_t power_iters,
                            const std::optional<torch::Tensor>& layer_offsets) {
  const LqLayout L =
      make_layout((int)head_v_dim, num_comp, capacity, window, power_iters);
  const int64_t HV = blob.size(1);
  check_blob(blob, L, HV);
  const int64_t N = state_indices.size(0);
  check_indices(state_indices, N);
  const bool has_acc = num_accepted.has_value();
  if (has_acc) {
    TORCH_CHECK(num_accepted->is_cuda() &&
                    num_accepted->scalar_type() == at::kInt &&
                    num_accepted->dim() == 1 && num_accepted->stride(0) == 1 &&
                    num_accepted->numel() >= N,
                "num_accepted must be unit-stride CUDA int32 [N]");
  }
  // layer_offsets[l]: byte offset of layer l's blob tensor from `blob`; every
  // layer's blob has blob's shape and strides (the caller guarantees it).
  int64_t num_layers = 1;
  if (layer_offsets.has_value()) {
    TORCH_CHECK(
        layer_offsets->is_cuda() && layer_offsets->scalar_type() == at::kLong &&
            layer_offsets->dim() == 1 && layer_offsets->stride(0) == 1 &&
            layer_offsets->numel() >= 1,
        "layer_offsets must be unit-stride CUDA int64 [layers]");
    num_layers = layer_offsets->numel();
  }
  if (N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(blob.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)(N * HV), (unsigned)num_layers);
  LQ_DISPATCH_RPT(
      head_v_dim,
      gdn_lq_normalize_kernel<RPT><<<grid, LQ_THREADS, 0, stream>>>(
          blob.data_ptr<uint8_t>(),
          layer_offsets.has_value() ? layer_offsets->data_ptr<int64_t>()
                                    : nullptr,
          state_indices.data_ptr<int>(), state_indices.stride(0),
          has_acc ? num_accepted->data_ptr<int>() : nullptr, blob.stride(0),
          blob.stride(1), blob.size(0), (int)HV, L));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_lq_store_rdna2(torch::Tensor state, torch::Tensor blob,
                        torch::Tensor state_indices, int64_t num_comp,
                        int64_t capacity, int64_t window, int64_t power_iters) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
                  state.dim() == 4 && state.size(3) == LQ_K &&
                  state.stride(3) == 1 && state.stride(2) == LQ_K,
              "state must be CUDA fp32 [N, HV, V, 128] with dense [V, K]");
  const int64_t V = state.size(2);
  const LqLayout L =
      make_layout((int)V, num_comp, capacity, window, power_iters);
  const int64_t HV = blob.size(1);
  check_blob(blob, L, HV);
  TORCH_CHECK(state.size(1) == HV, "state / blob head count mismatch");
  const int64_t N = state.size(0);
  check_indices(state_indices, N);
  if (N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(blob.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  LQ_DISPATCH_RPT(
      V, gdn_lq_store_kernel<RPT><<<N * HV, LQ_THREADS, 0, stream>>>(
             state.data_ptr<float>(), blob.data_ptr<uint8_t>(),
             state_indices.data_ptr<int>(), state_indices.stride(0),
             state.stride(0), state.stride(1), blob.stride(0), blob.stride(1),
             blob.size(0), (int)HV, L));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_lq_materialize_rdna2(torch::Tensor blob, torch::Tensor state,
                              torch::Tensor state_indices, int64_t num_comp,
                              int64_t capacity, int64_t window) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
                  state.dim() == 4 && state.size(3) == LQ_K &&
                  state.stride(3) == 1 && state.stride(2) == LQ_K,
              "state must be CUDA fp32 [N, HV, V, 128] with dense [V, K]");
  const int64_t V = state.size(2);
  const LqLayout L = make_layout((int)V, num_comp, capacity, window, 0);
  const int64_t HV = blob.size(1);
  check_blob(blob, L, HV);
  TORCH_CHECK(state.size(1) == HV, "state / blob head count mismatch");
  const int64_t N = state.size(0);
  check_indices(state_indices, N);
  if (N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(blob.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  LQ_DISPATCH_RPT(
      V, gdn_lq_materialize_kernel<RPT><<<N * HV, LQ_THREADS, 0, stream>>>(
             blob.data_ptr<uint8_t>(), state.data_ptr<float>(),
             state_indices.data_ptr<int>(), state_indices.stride(0),
             state.stride(0), state.stride(1), blob.stride(0), blob.stride(1),
             blob.size(0), (int)HV, L));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_lq_decode_rdna2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
                         torch::Tensor dt_bias, torch::Tensor out,
                         torch::Tensor blob, torch::Tensor cu_seqlens,
                         torch::Tensor state_indices, double scale,
                         int64_t num_comp, int64_t capacity, int64_t window) {
  TORCH_CHECK(
      q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
          v.scalar_type() == at::kHalf && a.scalar_type() == at::kHalf &&
          b.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
      "gdn_lq_decode_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
              "gdn_lq_decode_rdna2 wants [1, T, H, K] / [1, T, HV, V]");
  TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && a.dim() == b.dim(),
              "gdn_lq_decode_rdna2 wants a/b as [T, HV] or [1, T, HV]");
  TORCH_CHECK(q.stride(3) == 1 && q.stride(2) == q.size(3) &&
                  k.stride(3) == 1 && k.stride(2) == k.size(3) &&
                  v.stride(3) == 1 && v.stride(2) == v.size(3) &&
                  a.stride(-1) == 1 && b.stride(-1) == 1 && out.is_contiguous(),
              "gdn_lq_decode_rdna2 requires dense per-token q/k/v heads, "
              "unit-stride a/b and contiguous out");
  TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.dim() == 1 &&
                  cu_seqlens.stride(0) == 1 &&
                  cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be unit-stride int32 [N + 1]");
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t HV = v.size(2);
  const int64_t V = v.size(3);
  const int64_t N = cu_seqlens.size(0) - 1;
  TORCH_CHECK(q.size(3) == LQ_K, "gdn_lq_decode_rdna2 requires head_k_dim 128");
  TORCH_CHECK(HV % H == 0, "HV must be divisible by H");
  const int a_tok_dim = a.dim() - 2;
  TORCH_CHECK(v.size(1) == T && out.size(1) == T && a.size(a_tok_dim) == T &&
                  b.size(a_tok_dim) == T && a.size(-1) == HV &&
                  b.size(-1) == HV,
              "gdn_lq_decode_rdna2: token / head dim mismatch");
  const auto a_ty = A_log.scalar_type();
  const auto d_ty = dt_bias.scalar_type();
  TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() &&
                  (a_ty == at::kFloat || a_ty == at::kHalf) &&
                  dt_bias.dim() == 1 && dt_bias.is_contiguous() &&
                  (d_ty == at::kFloat || d_ty == at::kHalf),
              "A_log / dt_bias must be contiguous fp32/fp16 [HV]");
  const LqLayout L = make_layout((int)V, num_comp, capacity, window, 0);
  check_blob(blob, L, HV);
  check_indices(state_indices, N);
  TORCH_CHECK(V == 128, "LeapQuant GDN supports head_v_dim == 128");
  if (N == 0 || T == 0) return;

  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(V / LQ_BV, N * HV);
#define LQ_DECODE_LAUNCH(AT, DT, APTR, DPTR)                                 \
  gdn_lq_decode_kernel<AT, DT><<<grid, LQ_THREADS, 0, stream>>>(             \
      reinterpret_cast<const __half*>(q.data_ptr()),                         \
      reinterpret_cast<const __half*>(k.data_ptr()),                         \
      reinterpret_cast<const __half*>(v.data_ptr()),                         \
      reinterpret_cast<const __half*>(a.data_ptr()),                         \
      reinterpret_cast<const __half*>(b.data_ptr()), APTR, DPTR,             \
      reinterpret_cast<__half*>(out.data_ptr()), blob.data_ptr<uint8_t>(),   \
      state_indices.data_ptr<int>(), state_indices.stride(0),                \
      cu_seqlens.data_ptr<int>(), (int)H, (int)HV, q.stride(1), k.stride(1), \
      v.stride(1), a.stride(a_tok_dim), b.stride(a_tok_dim), out.stride(1),  \
      blob.stride(0), blob.stride(1), blob.size(0), (float)scale, L)
  if (a_ty == at::kFloat && d_ty == at::kFloat) {
    LQ_DECODE_LAUNCH(float, float, A_log.data_ptr<float>(),
                     dt_bias.data_ptr<float>());
  } else if (a_ty == at::kFloat) {
    LQ_DECODE_LAUNCH(float, __half, A_log.data_ptr<float>(),
                     reinterpret_cast<const __half*>(dt_bias.data_ptr()));
  } else if (d_ty == at::kFloat) {
    LQ_DECODE_LAUNCH(__half, float,
                     reinterpret_cast<const __half*>(A_log.data_ptr()),
                     dt_bias.data_ptr<float>());
  } else {
    LQ_DECODE_LAUNCH(__half, __half,
                     reinterpret_cast<const __half*>(A_log.data_ptr()),
                     reinterpret_cast<const __half*>(dt_bias.data_ptr()));
  }
#undef LQ_DECODE_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
