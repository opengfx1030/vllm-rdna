// T46 — fused decode glue for Qwen3.8-Flash-Next on gfx1030 (M <= 8 tokens; the
// shared-expert pair also serves MTP verify batches up to M = 32).
//
// The decode step is dispatch-bound (~2,900 kernels, ~4 us of bubble each).
// These kernels fold the elementwise glue around the dense projections into the
// projection itself:
//
//   rdna_gemv_act        y = x.W^T over the first act_cols + inj_cols weight
//   rows: silu(v * act_scale) on the first act_cols outputs -> lora [M,
//   act_cols],
//                        the next inj_cols outputs raw -> injection [M,
//                        inj_cols] (hyper-connection down+inject GEMV followed
//                        by hc_silu; no strided slice or contiguous copy is
//                        left to the caller)
//   rdna_hc_up_gate_mix  out[m,h] = mean_c sigmoid(W_up[c*H+h,:] . lora[m]) *
//   xn[m,c*H+h]
//                        (hyper-connection up GEMV + sigmoid + gated mean, one
//                        launch)
//   rdna_se_gate_up_silu act[m,i] = silu(W[i,:].x[m]) * (W[I+i,:].x[m]) and
//                        gate[m] = sigmoid(w_gate . x[m]) (fp32), the expert
//                        gate computed once by an extra block
//                        (shared expert gate_up GEMV + silu_and_mul)
//   rdna_se_down_gated   out[m,h] = gate[m] * (W_down[h,:] . act[m])
//                        (shared expert down GEMV + expert-gate mul)
//   rdna_moe_route_zero  moe_align_block_size + count_and_sort + both MoE
//                        accumulator zero fills, one launch (tokens * top_k
//                        <= 1024 routed entries); with the EPLB counter
//                        variant (rdna_moe_route_zero_stats, opt-in) one
//                        extra CTA of the same launch counts the call's
//                        expert load
//
// Weights are fp16 [N,K] or int8 [N,K] with a per-row fp16 scale (the T45
// shadows). Same wave-per-row streaming design as gemv_f16_rdna2 /
// gemv_i8_rdna2.
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int kMaxM = 8;     // exact-MT kernels (HC, int8 shadows)
constexpr int kMaxSeM = 32;  // shared expert: guarded MT = 16 / 24 / 32 above 8

// ---- one weight row (fp16 or int8) dotted with MT activation rows, lanes
// striding K ----
struct RowF16 {
  const half* w;
  __device__ __forceinline__ RowF16(const void* base, int n, int K, const half*)
      : w(reinterpret_cast<const half*>(base) + (size_t)n * K) {}
  // GUARD: MT is a capacity and rows m >= M are skipped (wave-uniform), for
  // MTP verify batches with 8 < M <= MT.
  template <int MT, bool GUARD = false>
  __device__ __forceinline__ void dot(const half* __restrict__ x, int K,
                                      int lane, float (&acc)[MT],
                                      int M = MT) const {
    const int K8 = K / 8;
    const uint4* wr = reinterpret_cast<const uint4*>(w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K8; i += 32 * 4) {
      uint4 wq[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        wq[u] = (idx < K8) ? wr[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        if (idx < K8) {
          const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
#pragma unroll
          for (int m = 0; m < MT; m++) {
            if (GUARD && m >= M) continue;
            const uint4 xq = xr[(size_t)m * K8 + idx];
            const half2* xh = reinterpret_cast<const half2*>(&xq);
            float a = acc[m];
            a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
            a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
            a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
            a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
            acc[m] = a;
          }
        }
      }
    }
  }
  // Two rows against the same activations: every x load feeds both. Each
  // accumulator sums its own row in the same order as dot().
  template <int MT, bool GUARD = false>
  __device__ __forceinline__ void dot_pair(const RowF16& o,
                                           const half* __restrict__ x, int K,
                                           int lane, float (&a0)[MT],
                                           float (&a1)[MT], int M = MT) const {
    const int K8 = K / 8;
    const uint4* w0 = reinterpret_cast<const uint4*>(w);
    const uint4* w1 = reinterpret_cast<const uint4*>(o.w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K8; i += 32 * 4) {
      uint4 q0[4], q1[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        q0[u] = (idx < K8) ? w0[idx] : make_uint4(0, 0, 0, 0);
        q1[u] = (idx < K8) ? w1[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        if (idx < K8) {
          const half2* h0 = reinterpret_cast<const half2*>(&q0[u]);
          const half2* h1 = reinterpret_cast<const half2*>(&q1[u]);
#pragma unroll
          for (int m = 0; m < MT; m++) {
            if (GUARD && m >= M) continue;
            const uint4 xq = xr[(size_t)m * K8 + idx];
            const half2* xh = reinterpret_cast<const half2*>(&xq);
            float p = a0[m], q = a1[m];
            p = __builtin_amdgcn_fdot2(h0[0], xh[0], p, false);
            q = __builtin_amdgcn_fdot2(h1[0], xh[0], q, false);
            p = __builtin_amdgcn_fdot2(h0[1], xh[1], p, false);
            q = __builtin_amdgcn_fdot2(h1[1], xh[1], q, false);
            p = __builtin_amdgcn_fdot2(h0[2], xh[2], p, false);
            q = __builtin_amdgcn_fdot2(h1[2], xh[2], q, false);
            p = __builtin_amdgcn_fdot2(h0[3], xh[3], p, false);
            q = __builtin_amdgcn_fdot2(h1[3], xh[3], q, false);
            a0[m] = p;
            a1[m] = q;
          }
        }
      }
    }
  }
  __device__ __forceinline__ float scale() const { return 1.f; }
};

struct RowI8 {
  const int8_t* w;
  float s;
  __device__ __forceinline__ RowI8(const void* base, int n, int K,
                                   const half* sc)
      : w(reinterpret_cast<const int8_t*>(base) + (size_t)n * K),
        s(__half2float(sc[n])) {}
  template <int MT, bool GUARD = false>
  __device__ __forceinline__ void dot(const half* __restrict__ x, int K,
                                      int lane, float (&acc)[MT],
                                      int M = MT) const {
    const int K16 = K / 16;
    const uint4* wr = reinterpret_cast<const uint4*>(w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K16; i += 32 * 4) {
      uint4 wq[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        wq[u] = (idx < K16) ? wr[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 32 * u;
        if (idx < K16) {
          const int32_t* q = reinterpret_cast<const int32_t*>(&wq[u]);
          half2 wh[8];
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int32_t v = q[j];
            wh[2 * j] = __floats2half2_rn((float)(int8_t)(v & 0xff),
                                          (float)(int8_t)((v >> 8) & 0xff));
            wh[2 * j + 1] =
                __floats2half2_rn((float)(int8_t)((v >> 16) & 0xff),
                                  (float)(int8_t)((v >> 24) & 0xff));
          }
#pragma unroll
          for (int m = 0; m < MT; m++) {
            if (GUARD && m >= M) continue;
            const uint4 xa = xr[((size_t)m * K16 + idx) * 2];
            const uint4 xb = xr[((size_t)m * K16 + idx) * 2 + 1];
            const half2* h0 = reinterpret_cast<const half2*>(&xa);
            const half2* h1 = reinterpret_cast<const half2*>(&xb);
            float a = acc[m];
            a = __builtin_amdgcn_fdot2(wh[0], h0[0], a, false);
            a = __builtin_amdgcn_fdot2(wh[1], h0[1], a, false);
            a = __builtin_amdgcn_fdot2(wh[2], h0[2], a, false);
            a = __builtin_amdgcn_fdot2(wh[3], h0[3], a, false);
            a = __builtin_amdgcn_fdot2(wh[4], h1[0], a, false);
            a = __builtin_amdgcn_fdot2(wh[5], h1[1], a, false);
            a = __builtin_amdgcn_fdot2(wh[6], h1[2], a, false);
            a = __builtin_amdgcn_fdot2(wh[7], h1[3], a, false);
            acc[m] = a;
          }
        }
      }
    }
  }
  template <int MT, bool GUARD = false>
  __device__ __forceinline__ void dot_pair(const RowI8& o,
                                           const half* __restrict__ x, int K,
                                           int lane, float (&a0)[MT],
                                           float (&a1)[MT], int M = MT) const {
    if constexpr (MT > 8) {
      dot<MT, GUARD>(x, K, lane, a0, M);
      o.template dot<MT, GUARD>(x, K, lane, a1, M);
      return;
    }
    const int K16 = K / 16;
    const uint4* w0 = reinterpret_cast<const uint4*>(w);
    const uint4* w1 = reinterpret_cast<const uint4*>(o.w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K16; i += 32 * 2) {
      uint4 q0[2], q1[2];
#pragma unroll
      for (int u = 0; u < 2; u++) {
        const int idx = i + 32 * u;
        q0[u] = (idx < K16) ? w0[idx] : make_uint4(0, 0, 0, 0);
        q1[u] = (idx < K16) ? w1[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 2; u++) {
        const int idx = i + 32 * u;
        if (idx < K16) {
          const int32_t* p0 = reinterpret_cast<const int32_t*>(&q0[u]);
          const int32_t* p1 = reinterpret_cast<const int32_t*>(&q1[u]);
          half2 wh0[8], wh1[8];
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int32_t v0 = p0[j], v1 = p1[j];
            wh0[2 * j] = __floats2half2_rn((float)(int8_t)(v0 & 0xff),
                                           (float)(int8_t)((v0 >> 8) & 0xff));
            wh0[2 * j + 1] =
                __floats2half2_rn((float)(int8_t)((v0 >> 16) & 0xff),
                                  (float)(int8_t)((v0 >> 24) & 0xff));
            wh1[2 * j] = __floats2half2_rn((float)(int8_t)(v1 & 0xff),
                                           (float)(int8_t)((v1 >> 8) & 0xff));
            wh1[2 * j + 1] =
                __floats2half2_rn((float)(int8_t)((v1 >> 16) & 0xff),
                                  (float)(int8_t)((v1 >> 24) & 0xff));
          }
#pragma unroll
          for (int m = 0; m < MT; m++) {
            if (GUARD && m >= M) continue;
            const uint4 xa = xr[((size_t)m * K16 + idx) * 2];
            const uint4 xb = xr[((size_t)m * K16 + idx) * 2 + 1];
            const half2* h0 = reinterpret_cast<const half2*>(&xa);
            const half2* h1 = reinterpret_cast<const half2*>(&xb);
            float p = a0[m], q = a1[m];
            p = __builtin_amdgcn_fdot2(wh0[0], h0[0], p, false);
            q = __builtin_amdgcn_fdot2(wh1[0], h0[0], q, false);
            p = __builtin_amdgcn_fdot2(wh0[1], h0[1], p, false);
            q = __builtin_amdgcn_fdot2(wh1[1], h0[1], q, false);
            p = __builtin_amdgcn_fdot2(wh0[2], h0[2], p, false);
            q = __builtin_amdgcn_fdot2(wh1[2], h0[2], q, false);
            p = __builtin_amdgcn_fdot2(wh0[3], h0[3], p, false);
            q = __builtin_amdgcn_fdot2(wh1[3], h0[3], q, false);
            p = __builtin_amdgcn_fdot2(wh0[4], h1[0], p, false);
            q = __builtin_amdgcn_fdot2(wh1[4], h1[0], q, false);
            p = __builtin_amdgcn_fdot2(wh0[5], h1[1], p, false);
            q = __builtin_amdgcn_fdot2(wh1[5], h1[1], q, false);
            p = __builtin_amdgcn_fdot2(wh0[6], h1[2], p, false);
            q = __builtin_amdgcn_fdot2(wh1[6], h1[2], q, false);
            p = __builtin_amdgcn_fdot2(wh0[7], h1[3], p, false);
            q = __builtin_amdgcn_fdot2(wh1[7], h1[3], q, false);
            a0[m] = p;
            a1[m] = q;
          }
        }
      }
    }
  }
  __device__ __forceinline__ float scale() const { return s; }
};

template <int MT>
__device__ __forceinline__ void wave_reduce(float (&acc)[MT]) {
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
}
__device__ __forceinline__ float silu_f(float v) {
  return v / (1.f + __expf(-v));
}
__device__ __forceinline__ float sigmoid_f(float v) {
  return 1.f / (1.f + __expf(-v));
}

// ---- 1. GEMV with silu(v * act_scale) on the first act_cols outputs, the next
// inj_cols outputs stored raw into a separate tensor
// ----------------------------------
template <typename Row, int WAVES, int MT, bool GUARD = false>
__global__ void __launch_bounds__(WAVES * 32)
    gemv_act_k(const half* __restrict__ x, const void* __restrict__ w,
               const half* __restrict__ sc, half* __restrict__ y,
               half* __restrict__ inj, const int N, const int K,
               const int act_cols, const int inj_cols, const float act_scale,
               const int M = MT) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  Row row(w, n, K, sc);
  row.template dot<MT, GUARD>(x, K, lane, acc, M);
  wave_reduce<MT>(acc);
  if (lane == 0) {
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      const float v = acc[m] * s;
      if (n < act_cols)
        y[(size_t)m * act_cols + n] = __float2half(silu_f(v * act_scale));
      else
        inj[(size_t)m * inj_cols + (n - act_cols)] = __float2half(v);
    }
  }
}

// ---- 2. hyper-connection up GEMV + sigmoid + gated mean
// -------------------------------- out[m,h] = (1/HC) * sum_c sigmoid(W[c*H+h,:]
// . lora[m]) * xn[m, c*H+h]
template <typename Row, int WAVES, int MT, int HC, bool GUARD = false>
__global__ void __launch_bounds__(WAVES * 32)
    hc_up_gate_mix_k(const half* __restrict__ lora, const void* __restrict__ w,
                     const half* __restrict__ sc, const half* __restrict__ xn,
                     half* __restrict__ out, const int H, const int R,
                     const int M = MT) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = blockIdx.x * WAVES + wave;
  if (h >= H) return;
  float mix[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) mix[m] = 0.f;
#pragma unroll
  for (int c = 0; c < HC; c++) {
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.f;
    Row row(w, c * H + h, R, sc);
    row.template dot<MT, GUARD>(lora, R, lane, acc, M);
    wave_reduce<MT>(acc);
    if (lane == 0) {
      const float s = row.scale();
#pragma unroll
      for (int m = 0; m < MT; m++) {
        if (GUARD && m >= M) continue;
        const float g = sigmoid_f(acc[m] * s);
        mix[m] += g * __half2float(xn[(size_t)m * (HC * H) + c * H + h]);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      out[(size_t)m * H + h] = __float2half(mix[m] / HC);
    }
  }
}

// ---- 3. shared expert gate_up GEMV + silu*mul (+ the expert gate)
// act[m,i] = silu(W[i,:].x[m]) * (W[I+i,:].x[m]);  W is [2I, K] (gate rows then
// up rows); both rows of a pair share every x load. The block past the last
// weight row computes gate[m] = sigmoid(wg . x[m]) once per token for the down
// kernel, which used to repeat those K-long dots in every one of its blocks.
template <typename Row, int WAVES, int MT, bool GUARD>
__global__ void __launch_bounds__(WAVES * 32)
    se_gate_up_silu_k(const half* __restrict__ x, const void* __restrict__ w,
                      const half* __restrict__ sc, const half* __restrict__ wg,
                      half* __restrict__ act, float* __restrict__ gate,
                      const int I, const int K, const int M) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (blockIdx.x * WAVES >= I) {  // the gate block
    if (wave == 0) {
      float gacc[MT];
#pragma unroll
      for (int m = 0; m < MT; m++) gacc[m] = 0.f;
      RowF16 rg(wg, 0, K, nullptr);
      rg.template dot<MT, GUARD>(x, K, lane, gacc, M);
      wave_reduce<MT>(gacc);
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
          if (GUARD && m >= M) continue;
          gate[m] = sigmoid_f(gacc[m]);
        }
      }
    }
    return;
  }
  const int i = blockIdx.x * WAVES + wave;
  if (i >= I) return;
  float g[MT], u[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    g[m] = 0.f;
    u[m] = 0.f;
  }
  Row rg(w, i, K, sc), ru(w, I + i, K, sc);
  rg.template dot_pair<MT, GUARD>(ru, x, K, lane, g, u, M);
  wave_reduce<MT>(g);
  wave_reduce<MT>(u);
  if (lane == 0) {
    const float sg = rg.scale(), su = ru.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      act[(size_t)m * I + i] = __float2half(silu_f(g[m] * sg) * (u[m] * su));
    }
  }
}

// ---- 4. shared expert down GEMV scaled by the expert gate
// ------------------------- out[m,h] = gate[m] * (W_down[h,:] . act[m]);
// gate is the fp32 sigmoid(w_gate . x[m]) from the gate_up kernel
template <typename Row, int WAVES, int MT, bool GUARD>
__global__ void __launch_bounds__(WAVES * 32)
    se_down_gated_k(const half* __restrict__ act, const void* __restrict__ w,
                    const half* __restrict__ sc, const float* __restrict__ gate,
                    half* __restrict__ out, const int H, const int I,
                    const int M) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = blockIdx.x * WAVES + wave;
  if (h >= H) return;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  Row row(w, h, I, sc);
  row.template dot<MT, GUARD>(act, I, lane, acc, M);
  wave_reduce<MT>(acc);
  if (lane == 0) {
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (GUARD && m >= M) continue;
      out[(size_t)m * H + h] = __float2half(acc[m] * s * gate[m]);
    }
  }
}

// ---- host helpers
// ------------------------------------------------------------------------
// (kernel launches live in plain template functions: hipify mangles <<<>>>
// inside macros)
inline int pick_waves(int N) {
  return N >= 1152 ? 8 : (N >= 576 ? 4 : (N >= 288 ? 2 : 1));
}

template <typename Row, int WV, int MT, bool G = false>
void launch_gemv_act(int blocks, cudaStream_t st, const half* x, const void* w,
                     const half* sc, half* y, half* inj, int N, int K,
                     int act_cols, int inj_cols, float act_scale, int M = MT) {
  gemv_act_k<Row, WV, MT, G><<<blocks, WV * 32, 0, st>>>(
      x, w, sc, y, inj, N, K, act_cols, inj_cols, act_scale, M);
}
template <typename Row, int WV, int MT, bool G = false>
void launch_hc_mix(int blocks, cudaStream_t st, const half* lora, const void* w,
                   const half* sc, const half* xn, half* out, int H, int R,
                   int M = MT) {
  hc_up_gate_mix_k<Row, WV, MT, 4, G>
      <<<blocks, WV * 32, 0, st>>>(lora, w, sc, xn, out, H, R, M);
}
template <typename Row, int WV, int MT, bool G>
void launch_se_gu(int blocks, cudaStream_t st, const half* x, const void* w,
                  const half* sc, const half* wg, half* act, float* gate, int I,
                  int K, int M) {
  se_gate_up_silu_k<Row, WV, MT, G>
      <<<blocks, WV * 32, 0, st>>>(x, w, sc, wg, act, gate, I, K, M);
}
template <typename Row, int WV, int MT, bool G>
void launch_se_dn(int blocks, cudaStream_t st, const half* act, const void* w,
                  const half* sc, const float* gate, half* out, int H, int I,
                  int M) {
  se_down_gated_k<Row, WV, MT, G>
      <<<blocks, WV * 32, 0, st>>>(act, w, sc, gate, out, H, I, M);
}

// dispatch (M, waves) -> template instance; F is a functor templated on <Row,
// WV, MT>
template <typename F>
void dispatch(int M, int wv, bool i8, F&& f) {
#define RDNA_WV(MT)                                                          \
  switch (wv) {                                                              \
    case 8:                                                                  \
      i8 ? f.template run<RowI8, 8, MT>() : f.template run<RowF16, 8, MT>(); \
      break;                                                                 \
    case 4:                                                                  \
      i8 ? f.template run<RowI8, 4, MT>() : f.template run<RowF16, 4, MT>(); \
      break;                                                                 \
    case 2:                                                                  \
      i8 ? f.template run<RowI8, 2, MT>() : f.template run<RowF16, 2, MT>(); \
      break;                                                                 \
    default:                                                                 \
      i8 ? f.template run<RowI8, 1, MT>() : f.template run<RowF16, 1, MT>(); \
      break;                                                                 \
  }
  switch (M) {
    case 1:
      RDNA_WV(1) break;
    case 2:
      RDNA_WV(2) break;
    case 3:
      RDNA_WV(3) break;
    case 4:
      RDNA_WV(4) break;
    case 5:
      RDNA_WV(5) break;
    case 6:
      RDNA_WV(6) break;
    case 7:
      RDNA_WV(7) break;
    default:
      RDNA_WV(8) break;
  }
#undef RDNA_WV
}

// MTP verify batches, 8 < M <= 32: supports both fp16 and int8 weights, MT is
// a capacity and rows >= M are skipped inside the kernel.
template <typename F>
void dispatch_guarded(int M, int wv, bool i8, F&& f) {
#define RDNA_WVG(MT)                                                       \
  switch (wv) {                                                            \
    case 8:                                                                \
      i8 ? f.template run<RowI8, 8, MT, true>()                            \
         : f.template run<RowF16, 8, MT, true>();                          \
      break;                                                               \
    case 4:                                                                \
      i8 ? f.template run<RowI8, 4, MT, true>()                            \
         : f.template run<RowF16, 4, MT, true>();                          \
      break;                                                               \
    case 2:                                                                \
      i8 ? f.template run<RowI8, 2, MT, true>()                            \
         : f.template run<RowF16, 2, MT, true>();                          \
      break;                                                               \
    default:                                                               \
      i8 ? f.template run<RowI8, 1, MT, true>()                            \
         : f.template run<RowF16, 1, MT, true>();                          \
      break;                                                               \
  }
  if (M <= 12) {
    RDNA_WVG(12)
  } else if (M <= 16) {
    RDNA_WVG(16)
  } else if (M <= 24) {
    RDNA_WVG(24)
  } else {
    RDNA_WVG(32)
  }
#undef RDNA_WVG
}

void check_weight(const at::Tensor& w, const std::optional<at::Tensor>& sc,
                  int64_t N, int64_t K, const char* who) {
  TORCH_CHECK(
      w.dim() == 2 && w.size(0) == N && w.size(1) == K && w.is_contiguous(),
      who, ": weight must be contiguous [", N, ", ", K, "]");
  if (w.scalar_type() == at::kChar) {
    TORCH_CHECK(sc.has_value() && sc->scalar_type() == at::kHalf &&
                    sc->numel() == N && sc->is_contiguous(),
                who, ": int8 weight needs fp16 scale[N]");
    TORCH_CHECK(K % 16 == 0, who, ": int8 needs K % 16 == 0");
  } else {
    TORCH_CHECK(w.scalar_type() == at::kHalf, who,
                ": weight must be fp16 or int8");
    TORCH_CHECK(K % 8 == 0, who, ": K % 8 == 0");
  }
}
inline const half* scale_ptr(const std::optional<at::Tensor>& sc) {
  return sc.has_value() ? reinterpret_cast<const half*>(sc->const_data_ptr())
                        : nullptr;
}
struct GemvActF {
  int blocks;
  cudaStream_t st;
  const half* x;
  const void* w;
  const half* sc;
  half* y;
  half* inj;
  int N, K, act_cols, inj_cols;
  float act_scale;
  int M;
  template <typename Row, int WV, int MT, bool G = false>
  void run() {
    launch_gemv_act<Row, WV, MT, G>(blocks, st, x, w, sc, y, inj, N, K,
                                    act_cols, inj_cols, act_scale, M);
  }
};

struct HcMixF {
  int blocks;
  cudaStream_t st;
  const half* lora;
  const void* w;
  const half* sc;
  const half* xn;
  half* out;
  int H, R, M;
  template <typename Row, int WV, int MT, bool G = false>
  void run() {
    launch_hc_mix<Row, WV, MT, G>(blocks, st, lora, w, sc, xn, out, H, R, M);
  }
};

struct SeGuF {
  int blocks;
  cudaStream_t st;
  const half* x;
  const void* w;
  const half* sc;
  const half* wg;
  half* act;
  float* gate;
  int I, K, M;
  template <typename Row, int WV, int MT, bool G = false>
  void run() {
    launch_se_gu<Row, WV, MT, G>(blocks, st, x, w, sc, wg, act, gate, I, K, M);
  }
};

struct SeDnF {
  int blocks;
  cudaStream_t st;
  const half* act;
  const void* w;
  const half* sc;
  const float* gate;
  half* out;
  int H, I, M;
  template <typename Row, int WV, int MT, bool G = false>
  void run() {
    launch_se_dn<Row, WV, MT, G>(blocks, st, act, w, sc, gate, out, H, I, M);
  }
};

// ---- MoE routing glue: align + sort + the two zero fills in ONE launch
// -------
//
// The fused MoE (moe_gptq_gemm_rdna2) needs, per layer, moe_align_block_size's
// two kernels (align, count_and_sort) and two fills (the gate_up scratch and
// the output accumulator both take fp16 atomic adds, so they start at zero):
// four launches and ~19 us of launch bubbles for a few microseconds of work at
// 9-32 tokens. CTA 0 here does the routing for numel = tokens * top_k <= 1024
// entries: tokens sorted by expert, each expert padded to block_size, expert id
// per block, padded total. It matches moe_align_block_size, except that the
// order inside an expert is the flat index (stable) instead of the atomic
// arrival order. The other CTAs zero the two buffers meanwhile.
constexpr int kRouteThreads = 256;
constexpr int kRouteMaxEntries = 1024;
constexpr int kRouteMaxExperts = 1024;

struct RouteArgs {
  const int32_t* topk_ids;
  const int32_t* expert_map;  // nullable: local id per global id, -1 = not ours
  int32_t* sorted_ids;
  int32_t* expert_ids;
  int32_t* post_pad;
  half* z1;  // [rows1, cols1], row stride ld1
  half* z2;  // [rows2, cols2], row stride ld2
  int numel, E, BM, max_padded, max_blocks;
  int rows1, cols1, ld1, rows2, cols2, ld2;
  int vec;  // 1: 16-byte stores (cols % 8 == 0, ld % 8 == 0, 16-byte aligned)
};

// ---- EPLB load counters (opt-in, VLLM_RDNA_EPLB_STATS_DIR) ----------------
//
// Expert-parallel load balancing needs, per MoE layer, how often every GLOBAL
// expert is routed to and what that means per rank: touched experts T_r (the
// decode kernel reads each touched expert's weights once) and routed pairs P_r
// (the prefill kernel is compute bound). One extra CTA of the routing launch
// (or the standalone eplb_stats_k for calls the routing CTA does not take)
// builds the call's histogram while CTA 0 routes, so there is no extra launch
// and nothing on the routing critical path. Every state word has ONE writer
// (this CTA) and the kernels of a stream run in order, so the counters are
// plain read-modify-write, no atomics in global memory. Single stream only.
//
// The layout is the contract with
// vllm/model_executor/layers/fused_moe/rdna_eplb_stats.py (a test compares the
// constants). Per-layer `state` (int64 words):
//   pairs[2][E]  routed pairs per global expert, phase 0 decode / 1 prefill
//   touch[nb][E] decode calls that touched the expert, per band of valid rows
//   calls[nb+1]  calls per band, last = prefill calls
//   rows[2]      valid rows summed per phase
//   ord          calls of this layer so far (sampled or not)
//   hcalls       prefill histograms written
//   hrows[hist_n] valid rows of each histogram slot
// One ring record = kEplbWords int32: 0 layer position, 1 valid rows, 2 rows
// incl. padding, 3 id hash (additive, order independent: ranks compare it),
// 4/5 wall_clock64 lo/hi, 6 phase, 7 layer call ordinal, 8.. T[rank],
// 16.. P[rank].
constexpr int kEplbWords = 32;
constexpr int kEplbMaxRanks = 8;
constexpr int kEplbMaxBands = 32;

struct StatsArgs {
  int64_t* state;
  int32_t* ring;           // [ring_n, kEplbWords] or null
  int64_t* gseq;           // [1] ring records written so far (with ring)
  int16_t* hist;           // [hist_n, E] prefill histograms or null
  const uint8_t* owner;    // [E] rank of every global expert
  const uint8_t* padding;  // [rows] 1 = padded row, or null
  const int32_t* edges;    // [nb] ascending decode row bands
  int nb, ring_n, hist_n, layer_pos, sample_mask, ranks, top_k, E;
};

__device__ __forceinline__ unsigned eplb_mix(unsigned i, int e) {
  unsigned x = (i * 0x9E3779B1u) ^ (((unsigned)e + 1u) * 0x85EBCA6Bu);
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return x;
}

// One CTA of kRouteThreads threads; every __syncthreads() is reached by all
// threads (thread 0 publishes the call ordinal first, so the early return for
// an unsampled call is uniform and no thread can read the bumped ordinal).
__device__ void eplb_stats_cta(const StatsArgs& s, const int32_t* ids,
                               int numel) {
  __shared__ int s_h[kRouteMaxExperts];
  __shared__ int s_T[kEplbMaxRanks];
  __shared__ int s_P[kEplbMaxRanks];
  __shared__ unsigned s_hash;
  __shared__ int s_valid;
  __shared__ unsigned long long s_ts;
  __shared__ int64_t s_ord;
  const int tid = threadIdx.x;
  const int E = s.E, K = s.top_k, nb = s.nb;
  const int64_t off_touch = 2LL * E;
  const int64_t off_calls = off_touch + (int64_t)nb * E;
  const int64_t off_rows = off_calls + nb + 1;
  const int64_t off_ord = off_rows + 2;
  const int64_t off_hcalls = off_ord + 1;
  const int64_t off_hrows = off_hcalls + 1;
  if (tid == 0) s_ord = s.state[off_ord];
  __syncthreads();
  const int64_t ord = s_ord;
  if ((ord & (int64_t)s.sample_mask) != 0) {  // not a sampled call
    if (tid == 0) s.state[off_ord] = ord + 1;
    return;
  }
  for (int e = tid; e < E; e += kRouteThreads) s_h[e] = 0;
  if (tid < kEplbMaxRanks) {
    s_T[tid] = 0;
    s_P[tid] = 0;
  }
  if (tid == 0) {
    s_hash = 0;
    s_valid = 0;
  }
  __syncthreads();
  unsigned h = 0;
  int valid = 0;
  for (int i = tid; i < numel; i += kRouteThreads) {
    const int row = i / K;
    if (s.padding && s.padding[row]) continue;
    const int e = ids[i];
    if (e >= 0 && e < E) atomicAdd(&s_h[e], 1);
    h += eplb_mix((unsigned)i, e);
    valid += (i == row * K);  // once per valid row
  }
  atomicAdd(&s_hash, h);
  atomicAdd(&s_valid, valid);
  __syncthreads();
  const int rows_valid = s_valid;
  const int phase = rows_valid <= s.edges[nb - 1] ? 0 : 1;
  int band = 0;
  while (band < nb - 1 && s.edges[band] < rows_valid) band++;
  const bool count = rows_valid > 0;
  const int64_t hc = s.hist ? s.state[off_hcalls] : 0;
  const bool want_hist = s.hist != nullptr && phase == 1 && count;
  int64_t* pairs = s.state + (int64_t)phase * E;
  int64_t* touch = s.state + off_touch + (int64_t)band * E;
  int16_t* hslot = s.hist ? s.hist + (size_t)(hc % s.hist_n) * E : nullptr;
  for (int e = tid; e < E; e += kRouteThreads) {
    const int c = s_h[e];
    if (want_hist) hslot[e] = (int16_t)min(c, 32767);
    if (c == 0) continue;
    pairs[e] += c;
    if (phase == 0) touch[e] += 1;
    const int r = min((int)s.owner[e], kEplbMaxRanks - 1);
    atomicAdd(&s_T[r], 1);
    atomicAdd(&s_P[r], c);
  }
  if (tid == 0) s_ts = wall_clock64();
  __syncthreads();
  const int64_t seq = s.ring ? *s.gseq : 0;
  if (s.ring && tid < kEplbWords) {
    int32_t w = 0;
    switch (tid) {
      case 0:
        w = s.layer_pos;
        break;
      case 1:
        w = rows_valid;
        break;
      case 2:
        w = numel / K;
        break;
      case 3:
        w = (int32_t)s_hash;
        break;
      case 4:
        w = (int32_t)(unsigned)(s_ts & 0xffffffffull);
        break;
      case 5:
        w = (int32_t)(unsigned)(s_ts >> 32);
        break;
      case 6:
        w = phase;
        break;
      case 7:
        w = (int32_t)ord;
        break;
      default:
        if (tid >= 8 && tid < 8 + kEplbMaxRanks)
          w = s_T[tid - 8];
        else if (tid >= 16 && tid < 16 + kEplbMaxRanks)
          w = s_P[tid - 16];
    }
    s.ring[(size_t)(seq % s.ring_n) * kEplbWords + tid] = w;
  }
  __syncthreads();
  if (tid == 0) {
    if (count) {
      s.state[off_calls + (phase == 0 ? band : nb)] += 1;
      s.state[off_rows + phase] += rows_valid;
      if (want_hist) {
        s.state[off_hrows + hc % s.hist_n] = rows_valid;
        s.state[off_hcalls] = hc + 1;
      }
    }
    if (s.ring) *s.gseq = seq + 1;
    s.state[off_ord] = ord + 1;
  }
}

// Calls the routing CTA does not take (more than 1024 routed entries, the
// deterministic accumulator path): a launch of its own, a few microseconds
// next to a prefill step of milliseconds.
__global__ void __launch_bounds__(kRouteThreads)
    eplb_stats_k(const StatsArgs s, const int32_t* ids, int numel) {
  eplb_stats_cta(s, ids, numel);
}

struct RouteArgsStats : RouteArgs {
  StatsArgs st;
};
// The counter-free instantiation keeps the exact RouteArgs kernel signature.
template <bool kStats>
using RouteKernelArgs = std::conditional_t<kStats, RouteArgsStats, RouteArgs>;

template <bool kStats>
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_zero_k(const RouteKernelArgs<kStats> a) {
  const int tid = threadIdx.x;
  if (blockIdx.x != 0) {  // ---- zero the two accumulators
    if constexpr (kStats) {
      if (blockIdx.x == gridDim.x - 1) {  // ---- count the expert load
        eplb_stats_cta(a.st, a.topk_ids, a.numel);
        return;
      }
    }
    const int cpr1 = a.vec ? a.cols1 >> 3 : a.cols1;
    const int cpr2 = a.vec ? a.cols2 >> 3 : a.cols2;
    const int n1 = a.rows1 * cpr1, total = n1 + a.rows2 * cpr2;
    const int step = (gridDim.x - 1 - (kStats ? 1 : 0)) * kRouteThreads;
    for (int it = (blockIdx.x - 1) * kRouteThreads + tid; it < total;
         it += step) {
      const bool first = it < n1;
      const int j = first ? it : it - n1;
      const int cpr = first ? cpr1 : cpr2;
      half* base = (first ? a.z1 : a.z2) +
                   (size_t)(j / cpr) * (first ? a.ld1 : a.ld2) +
                   (size_t)(j % cpr) * (a.vec ? 8 : 1);
      if (a.vec)
        *reinterpret_cast<uint4*>(base) = make_uint4(0, 0, 0, 0);
      else
        *reinterpret_cast<uint16_t*>(base) = 0;
    }
    return;
  }

  __shared__ __attribute__((aligned(16))) int s_e[kRouteMaxEntries];  // -1 skip
  __shared__ int s_cnt[kRouteMaxExperts];  // entries per expert
  __shared__ int s_off[kRouteMaxExperts];  // first block of each expert
  __shared__ int s_wsum[kRouteThreads / 32 + 1];

  for (int e = tid; e < a.E; e += kRouteThreads) s_cnt[e] = 0;
  for (int p = tid; p < a.max_padded; p += kRouteThreads)
    a.sorted_ids[p] = a.numel;  // padding id: row numel / top_k = M, skipped
  for (int i = tid; i < a.numel; i += kRouteThreads) {
    int e = a.topk_ids[i];
    if (e < 0 || e >= a.E)
      e = -1;
    else if (a.expert_map)
      e = a.expert_map[e];
    s_e[i] = (e >= 0 && e < a.E) ? e : -1;
  }
  __syncthreads();
  for (int i = tid; i < a.numel; i += kRouteThreads)
    if (s_e[i] >= 0) atomicAdd(&s_cnt[s_e[i]], 1);
  __syncthreads();

  // blocks per expert, exclusive scan over E <= 1024 experts: each thread owns
  // PER consecutive experts, then a wave scan plus the wave totals.
  const int per = (a.E + kRouteThreads - 1) / kRouteThreads;
  int nb[4];
  int sum = 0;
  for (int j = 0; j < per; j++) {
    const int e = tid * per + j;
    nb[j] = e < a.E ? (s_cnt[e] + a.BM - 1) / a.BM : 0;
    sum += nb[j];
  }
  const int lane = tid % warpSize, wave = tid / warpSize;
  int v = sum;
  for (int d = 1; d < warpSize; d <<= 1) {
    const int n = __shfl_up(v, d);
    if (lane >= d) v += n;
  }
  if (lane == warpSize - 1) s_wsum[wave] = v;
  __syncthreads();
  int base = 0, total_blocks = 0;
  for (int w = 0; w < kRouteThreads / warpSize; w++) {
    if (w < wave) base += s_wsum[w];
    total_blocks += s_wsum[w];
  }
  int first_block = base + v - sum;
  for (int j = 0; j < per; j++) {
    const int e = tid * per + j;
    if (e < a.E) {
      s_off[e] = first_block;
      for (int b = 0; b < nb[j]; b++) a.expert_ids[first_block + b] = e;
    }
    first_block += nb[j];
  }
  for (int b = total_blocks + tid; b < a.max_blocks; b += kRouteThreads)
    a.expert_ids[b] = -1;  // inactive blocks, like moe_align_block_size
  if (tid == 0) a.post_pad[0] = total_blocks * a.BM;
  __syncthreads();

  for (int i = tid; i < a.numel; i += kRouteThreads) {
    const int e = s_e[i];
    if (e < 0) continue;
    // stable rank: entries of the same expert before this one, four per read
    int rank = 0;
    const int4* s_e4 = reinterpret_cast<const int4*>(s_e);
#pragma unroll 4
    for (int j = 0; j < (i >> 2); j++) {
      const int4 v = s_e4[j];
      rank += (v.x == e) + (v.y == e) + (v.z == e) + (v.w == e);
    }
    for (int j = i & ~3; j < i; j++) rank += s_e[j] == e;
    a.sorted_ids[s_off[e] * a.BM + rank] = i;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> rdna_gemv_act(
    const at::Tensor& x, const at::Tensor& w,
    const std::optional<at::Tensor>& scale, int64_t act_cols, double act_scale,
    int64_t inj_cols) {
  const int M = x.size(0), K = x.size(1);
  const int64_t rows = act_cols + inj_cols;
  const bool i8 = w.scalar_type() == at::kChar;
  TORCH_CHECK(
      M >= 1 && M <= kMaxSeM && x.scalar_type() == at::kHalf && x.is_contiguous(),
      "rdna_gemv_act: x must be contiguous fp16 [1..32, K]");
  TORCH_CHECK(act_cols >= 1 && inj_cols >= 0 && rows <= w.size(0),
              "rdna_gemv_act: need 1 <= act_cols and act_cols + inj_cols <= N");
  check_weight(w, scale, w.size(0), K, "rdna_gemv_act");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto lora = at::empty({M, act_cols}, x.options());
  auto inj = at::empty({M, inj_cols}, x.options());
  const int N = (int)rows;
  const int wv = pick_waves(N);
  GemvActF f{(N + wv - 1) / wv,
             at::cuda::getCurrentCUDAStream(),
             reinterpret_cast<const half*>(x.const_data_ptr()),
             w.const_data_ptr(),
             scale_ptr(scale),
             reinterpret_cast<half*>(lora.mutable_data_ptr()),
             reinterpret_cast<half*>(inj.mutable_data_ptr()),
             N,
             K,
             (int)act_cols,
             (int)inj_cols,
             (float)act_scale,
             M};
  if (M <= kMaxM)
    dispatch(M, wv, i8, f);
  else
    dispatch_guarded(M, wv, i8, f);
  return {lora, inj};
}

at::Tensor rdna_hc_up_gate_mix(const at::Tensor& lora, const at::Tensor& w,
                               const std::optional<at::Tensor>& scale,
                               const at::Tensor& xn, int64_t hc_count) {
  const int M = lora.size(0), R = lora.size(1);
  TORCH_CHECK(hc_count == 4, "rdna_hc_up_gate_mix: hc_count 4 only");
  const int64_t HCH = w.size(0);
  const int H = (int)(HCH / hc_count);
  const bool i8 = w.scalar_type() == at::kChar;
  TORCH_CHECK(M >= 1 && M <= kMaxSeM && lora.scalar_type() == at::kHalf &&
                  lora.is_contiguous(),
              "rdna_hc_up_gate_mix: lora must be contiguous fp16 [1..32, R]");
  TORCH_CHECK(xn.scalar_type() == at::kHalf && xn.is_contiguous() &&
                  xn.size(0) == M && xn.size(1) == HCH,
              "rdna_hc_up_gate_mix: xn must be fp16 [M, HC*H]");
  check_weight(w, scale, HCH, R, "rdna_hc_up_gate_mix");
  const at::cuda::OptionalCUDAGuard guard(lora.device());
  auto out = at::empty({M, H}, lora.options());
  const int wv = pick_waves(H);
  HcMixF f{(H + wv - 1) / wv,
           at::cuda::getCurrentCUDAStream(),
           reinterpret_cast<const half*>(lora.const_data_ptr()),
           w.const_data_ptr(),
           scale_ptr(scale),
           reinterpret_cast<const half*>(xn.const_data_ptr()),
           reinterpret_cast<half*>(out.mutable_data_ptr()),
           H,
           R,
           M};
  if (M <= kMaxM)
    dispatch(M, wv, i8, f);
  else
    dispatch_guarded(M, wv, i8, f);
  return out;
}

std::tuple<at::Tensor, at::Tensor> rdna_se_gate_up_silu(
    const at::Tensor& x, const at::Tensor& w,
    const std::optional<at::Tensor>& scale, const at::Tensor& w_gate) {
  const int M = x.size(0), K = x.size(1);
  const int64_t N2 = w.size(0);
  TORCH_CHECK(N2 % 2 == 0, "rdna_se_gate_up_silu: weight rows must be 2*I");
  const int I = (int)(N2 / 2);
  const bool i8 = w.scalar_type() == at::kChar;
  TORCH_CHECK(M >= 1 && M <= kMaxSeM && x.scalar_type() == at::kHalf &&
                  x.is_contiguous(),
              "rdna_se_gate_up_silu: x must be contiguous fp16 [1..32, K]");
  TORCH_CHECK(w_gate.scalar_type() == at::kHalf && w_gate.is_contiguous() &&
                  w_gate.numel() == K && K % 8 == 0,
              "rdna_se_gate_up_silu: w_gate must be fp16 [K], K % 8 == 0");
  check_weight(w, scale, N2, K, "rdna_se_gate_up_silu");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto act = at::empty({M, I}, x.options());
  auto gate = at::empty({M}, x.options().dtype(at::kFloat));
  const int wv = pick_waves(I);
  SeGuF f{(I + wv - 1) / wv + 1,  // + the block that computes the expert gate
          at::cuda::getCurrentCUDAStream(),
          reinterpret_cast<const half*>(x.const_data_ptr()),
          w.const_data_ptr(),
          scale_ptr(scale),
          reinterpret_cast<const half*>(w_gate.const_data_ptr()),
          reinterpret_cast<half*>(act.mutable_data_ptr()),
          reinterpret_cast<float*>(gate.mutable_data_ptr()),
          I,
          K,
          M};
  if (M <= kMaxM)
    dispatch(M, wv, i8, f);
  else
    dispatch_guarded(M, wv, i8, f);
  return {act, gate};
}

at::Tensor rdna_se_down_gated(const at::Tensor& act, const at::Tensor& w,
                              const std::optional<at::Tensor>& scale,
                              const at::Tensor& gate) {
  const int M = act.size(0), I = act.size(1), H = w.size(0);
  const bool i8 = w.scalar_type() == at::kChar;
  TORCH_CHECK(M >= 1 && M <= kMaxSeM && act.scalar_type() == at::kHalf &&
                  act.is_contiguous(),
              "rdna_se_down_gated: act must be contiguous fp16 [1..32, I]");
  TORCH_CHECK(gate.scalar_type() == at::kFloat && gate.is_contiguous() &&
                  gate.numel() == M,
              "rdna_se_down_gated: gate must be fp32 [M]");
  check_weight(w, scale, H, I, "rdna_se_down_gated");
  const at::cuda::OptionalCUDAGuard guard(act.device());
  auto out = at::empty({M, H}, act.options());
  const int wv = pick_waves(H);
  SeDnF f{(H + wv - 1) / wv,
          at::cuda::getCurrentCUDAStream(),
          reinterpret_cast<const half*>(act.const_data_ptr()),
          w.const_data_ptr(),
          scale_ptr(scale),
          reinterpret_cast<const float*>(gate.const_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()),
          H,
          I,
          M};
  if (M <= kMaxM)
    dispatch(M, wv, i8, f);
  else
    dispatch_guarded(M, wv, i8, f);
  return out;
}

// Routing for the fused MoE plus both accumulator zero fills, one launch.
// Returns (sorted_ids, expert_ids, num_tokens_post_padded) laid out as
// moe_align_block_size does for the same arguments; zero1 / zero2 are fp16
// 2-D buffers (unit column stride) that are set to zero. `stats` (nullable)
// adds the EPLB counter CTA to the same launch.
static std::tuple<at::Tensor, at::Tensor, at::Tensor> route_zero_impl(
    const at::Tensor& topk_ids, int64_t block_size, int64_t num_experts,
    const std::optional<at::Tensor>& expert_map, at::Tensor& zero1,
    at::Tensor& zero2, const StatsArgs* stats) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.is_contiguous(),
              "rdna_moe_route_zero: topk_ids must be contiguous int32 on GPU");
  const int64_t numel = topk_ids.numel();
  TORCH_CHECK(numel >= 1 && numel <= kRouteMaxEntries,
              "rdna_moe_route_zero: 1..", kRouteMaxEntries,
              " routed entries, got ", numel);
  TORCH_CHECK(num_experts >= 1 && num_experts <= kRouteMaxExperts,
              "rdna_moe_route_zero: 1..", kRouteMaxExperts, " experts, got ",
              num_experts);
  TORCH_CHECK(block_size >= 1 && block_size <= 64,
              "rdna_moe_route_zero: block_size must be 1..64, got ",
              block_size);
  const int32_t* map_ptr = nullptr;
  if (expert_map.has_value()) {
    const at::Tensor& em = *expert_map;
    TORCH_CHECK(em.is_cuda() && em.scalar_type() == at::kInt &&
                    em.is_contiguous() && em.numel() >= num_experts,
                "rdna_moe_route_zero: expert_map must be contiguous int32 "
                "[num_experts] on GPU");
    map_ptr = em.const_data_ptr<int32_t>();
  }
  for (const at::Tensor* z : {&zero1, &zero2})
    TORCH_CHECK(z->is_cuda() && z->scalar_type() == at::kHalf &&
                    z->dim() == 2 && z->stride(1) == 1 &&
                    z->stride(0) >= z->size(1),
                "rdna_moe_route_zero: zero buffers must be fp16 2-D with unit "
                "column stride");
  const at::cuda::OptionalCUDAGuard guard(topk_ids.device());

  int64_t max_padded = numel + num_experts * (block_size - 1);
  if (numel < num_experts)
    max_padded = std::min(max_padded, numel * block_size);
  const int64_t max_blocks = (max_padded + block_size - 1) / block_size;
  auto sorted_ids = at::empty({max_padded}, topk_ids.options());
  auto expert_ids = at::empty({max_blocks}, topk_ids.options());
  auto post_pad = at::empty({1}, topk_ids.options());

  RouteArgs a;
  a.topk_ids = topk_ids.const_data_ptr<int32_t>();
  a.expert_map = map_ptr;
  a.sorted_ids = sorted_ids.mutable_data_ptr<int32_t>();
  a.expert_ids = expert_ids.mutable_data_ptr<int32_t>();
  a.post_pad = post_pad.mutable_data_ptr<int32_t>();
  a.z1 = reinterpret_cast<half*>(zero1.mutable_data_ptr());
  a.z2 = reinterpret_cast<half*>(zero2.mutable_data_ptr());
  a.numel = (int)numel;
  a.E = (int)num_experts;
  a.BM = (int)block_size;
  a.max_padded = (int)max_padded;
  a.max_blocks = (int)max_blocks;
  a.rows1 = (int)zero1.size(0);
  a.cols1 = (int)zero1.size(1);
  a.ld1 = (int)zero1.stride(0);
  a.rows2 = (int)zero2.size(0);
  a.cols2 = (int)zero2.size(1);
  a.ld2 = (int)zero2.stride(0);
  a.vec = a.cols1 % 8 == 0 && a.cols2 % 8 == 0 && a.ld1 % 8 == 0 &&
          a.ld2 % 8 == 0 && reinterpret_cast<uintptr_t>(a.z1) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(a.z2) % 16 == 0;
  const int64_t halves =
      (int64_t)a.rows1 * a.cols1 + (int64_t)a.rows2 * a.cols2;
  const int64_t items = a.vec ? halves / 8 : halves;
  // two items per thread, at most 64 zeroing CTAs next to the routing CTA
  const int zero_ctas =
      (int)std::min<int64_t>(64, std::max<int64_t>(1, (items + 511) / 512));
  if (stats != nullptr) {
    RouteArgsStats sa;
    static_cast<RouteArgs&>(sa) = a;
    sa.st = *stats;
    moe_route_zero_k<true>
        <<<2 + zero_ctas, kRouteThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sa);
  } else {
    moe_route_zero_k<false>
        <<<1 + zero_ctas, kRouteThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted_ids, expert_ids, post_pad};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> rdna_moe_route_zero(
    const at::Tensor& topk_ids, int64_t block_size, int64_t num_experts,
    const std::optional<at::Tensor>& expert_map, at::Tensor& zero1,
    at::Tensor& zero2) {
  return route_zero_impl(topk_ids, block_size, num_experts, expert_map, zero1,
                         zero2, nullptr);
}

// ---- EPLB counters: argument checks shared by the two entry points ---------
static StatsArgs make_stats_args(
    const at::Tensor& topk_ids, const std::optional<at::Tensor>& is_padding,
    at::Tensor& state, const std::optional<at::Tensor>& ring,
    const std::optional<at::Tensor>& gseq,
    const std::optional<at::Tensor>& hist, const at::Tensor& owner,
    const at::Tensor& edges, int64_t layer_pos, int64_t sample_mask,
    int64_t ranks, int64_t num_experts, const char* who) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.is_contiguous() && topk_ids.dim() == 2 &&
                  topk_ids.size(1) >= 1,
              who, ": topk_ids must be contiguous int32 [rows, top_k] on GPU");
  const int64_t rows = topk_ids.size(0);
  TORCH_CHECK(num_experts >= 1 && num_experts <= kRouteMaxExperts, who, ": 1..",
              kRouteMaxExperts, " experts, got ", num_experts);
  TORCH_CHECK(ranks >= 1 && ranks <= kEplbMaxRanks, who, ": 1..", kEplbMaxRanks,
              " ranks, got ", ranks);
  TORCH_CHECK(sample_mask >= 0 && layer_pos >= 0, who,
              ": sample_mask and layer_pos must be >= 0");
  TORCH_CHECK(owner.is_cuda() && owner.scalar_type() == at::kByte &&
                  owner.is_contiguous() && owner.numel() == num_experts,
              who, ": owner must be uint8 [num_experts] on GPU");
  TORCH_CHECK(edges.is_cuda() && edges.scalar_type() == at::kInt &&
                  edges.is_contiguous() && edges.numel() >= 1 &&
                  edges.numel() <= kEplbMaxBands,
              who, ": edges must be int32 [1..", kEplbMaxBands, "] on GPU");
  TORCH_CHECK(ring.has_value() == gseq.has_value(), who,
              ": ring and gseq go together");
  const int64_t nb = edges.numel();
  const int64_t hist_n = hist.has_value() ? hist->size(0) : 0;
  const int64_t need =
      2 * num_experts + nb * num_experts + nb + 1 + 2 + 1 + 1 + hist_n;
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kLong &&
                  state.is_contiguous() && state.numel() == need,
              who, ": state must be int64 [", need, "] on GPU");
  StatsArgs s{};
  if (is_padding.has_value()) {
    const at::Tensor& p = *is_padding;
    TORCH_CHECK(
        (p.scalar_type() == at::kBool || p.scalar_type() == at::kByte) &&
            p.is_cuda() && p.is_contiguous() && p.numel() >= rows,
        who, ": is_padding must be bool [>= rows] on GPU");
    s.padding = reinterpret_cast<const uint8_t*>(p.const_data_ptr());
  }
  if (ring.has_value()) {
    TORCH_CHECK(ring->is_cuda() && ring->scalar_type() == at::kInt &&
                    ring->is_contiguous() && ring->dim() == 2 &&
                    ring->size(1) == kEplbWords && ring->size(0) >= 1,
                who, ": ring must be int32 [N, ", kEplbWords, "] on GPU");
    TORCH_CHECK(gseq->is_cuda() && gseq->scalar_type() == at::kLong &&
                    gseq->numel() == 1,
                who, ": gseq must be int64 [1] on GPU");
    s.ring = ring->mutable_data_ptr<int32_t>();
    s.gseq = gseq->mutable_data_ptr<int64_t>();
    s.ring_n = (int)ring->size(0);
  }
  if (hist.has_value()) {
    TORCH_CHECK(hist->is_cuda() && hist->scalar_type() == at::kShort &&
                    hist->is_contiguous() && hist->dim() == 2 &&
                    hist->size(1) == num_experts && hist->size(0) >= 1,
                who, ": hist must be int16 [H, num_experts] on GPU");
    s.hist = hist->mutable_data_ptr<int16_t>();
    s.hist_n = (int)hist_n;
  }
  s.state = state.mutable_data_ptr<int64_t>();
  s.owner = owner.const_data_ptr<uint8_t>();
  s.edges = edges.const_data_ptr<int32_t>();
  s.nb = (int)nb;
  s.layer_pos = (int)layer_pos;
  s.sample_mask = (int)sample_mask;
  s.ranks = (int)ranks;
  s.top_k = (int)topk_ids.size(1);
  s.E = (int)num_experts;
  return s;
}

// rdna_moe_route_zero plus the EPLB counters in the same launch (one extra
// CTA). Same outputs and, for the routing itself, the same kernel code.
std::tuple<at::Tensor, at::Tensor, at::Tensor> rdna_moe_route_zero_stats(
    const at::Tensor& topk_ids, int64_t block_size, int64_t num_experts,
    const std::optional<at::Tensor>& expert_map, at::Tensor& zero1,
    at::Tensor& zero2, const std::optional<at::Tensor>& is_padding,
    at::Tensor& state, const std::optional<at::Tensor>& ring,
    const std::optional<at::Tensor>& gseq,
    const std::optional<at::Tensor>& hist, const at::Tensor& owner,
    const at::Tensor& edges, int64_t layer_pos, int64_t sample_mask,
    int64_t ranks) {
  const StatsArgs st = make_stats_args(
      topk_ids, is_padding, state, ring, gseq, hist, owner, edges, layer_pos,
      sample_mask, ranks, num_experts, "rdna_moe_route_zero_stats");
  return route_zero_impl(topk_ids, block_size, num_experts, expert_map, zero1,
                         zero2, &st);
}

// The counters alone, for the calls the routing CTA does not take (more than
// 1024 routed entries, the deterministic accumulator path).
void rdna_eplb_stats(const at::Tensor& topk_ids,
                     const std::optional<at::Tensor>& is_padding,
                     at::Tensor& state, const std::optional<at::Tensor>& ring,
                     const std::optional<at::Tensor>& gseq,
                     const std::optional<at::Tensor>& hist,
                     const at::Tensor& owner, const at::Tensor& edges,
                     int64_t layer_pos, int64_t sample_mask, int64_t ranks,
                     int64_t num_experts) {
  const StatsArgs st = make_stats_args(
      topk_ids, is_padding, state, ring, gseq, hist, owner, edges, layer_pos,
      sample_mask, ranks, num_experts, "rdna_eplb_stats");
  const at::cuda::OptionalCUDAGuard guard(topk_ids.device());
  eplb_stats_k<<<1, kRouteThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      st, topk_ids.const_data_ptr<int32_t>(), (int)topk_ids.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
