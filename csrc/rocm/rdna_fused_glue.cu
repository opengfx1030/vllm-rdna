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
//
// Weights are fp16 [N,K] or int8 [N,K] with a per-row fp16 scale (the T45
// shadows). Same wave-per-row streaming design as gemv_f16_rdna2 /
// gemv_i8_rdna2.
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
    static_assert(!GUARD, "int8 weights: exact MT <= 8 only");
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
    dot<MT, GUARD>(x, K, lane, a0, M);
    o.template dot<MT, GUARD>(x, K, lane, a1, M);
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
template <typename Row, int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
    gemv_act_k(const half* __restrict__ x, const void* __restrict__ w,
               const half* __restrict__ sc, half* __restrict__ y,
               half* __restrict__ inj, const int N, const int K,
               const int act_cols, const int inj_cols, const float act_scale) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  Row row(w, n, K, sc);
  row.template dot<MT>(x, K, lane, acc);
  wave_reduce<MT>(acc);
  if (lane == 0) {
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
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
template <typename Row, int WAVES, int MT, int HC>
__global__ void __launch_bounds__(WAVES * 32)
    hc_up_gate_mix_k(const half* __restrict__ lora, const void* __restrict__ w,
                     const half* __restrict__ sc, const half* __restrict__ xn,
                     half* __restrict__ out, const int H, const int R) {
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
    row.template dot<MT>(lora, R, lane, acc);
    wave_reduce<MT>(acc);
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const float g = sigmoid_f(acc[m] * s);
      mix[m] += g * __half2float(xn[(size_t)m * (HC * H) + c * H + h]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MT; m++)
      out[(size_t)m * H + h] = __float2half(mix[m] / HC);
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

template <typename Row, int WV, int MT>
void launch_gemv_act(int blocks, cudaStream_t st, const half* x, const void* w,
                     const half* sc, half* y, half* inj, int N, int K,
                     int act_cols, int inj_cols, float act_scale) {
  gemv_act_k<Row, WV, MT><<<blocks, WV * 32, 0, st>>>(
      x, w, sc, y, inj, N, K, act_cols, inj_cols, act_scale);
}
template <typename Row, int WV, int MT>
void launch_hc_mix(int blocks, cudaStream_t st, const half* lora, const void* w,
                   const half* sc, const half* xn, half* out, int H, int R) {
  hc_up_gate_mix_k<Row, WV, MT, 4>
      <<<blocks, WV * 32, 0, st>>>(lora, w, sc, xn, out, H, R);
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

// MTP verify batches, 8 < M <= 32: fp16 weights only, MT is a capacity and
// rows >= M are skipped inside the kernel. F::run takes a trailing bool G.
template <typename F>
void dispatch_guarded(int M, int wv, F&& f) {
#define RDNA_WVG(MT)                         \
  switch (wv) {                              \
    case 8:                                  \
      f.template run<RowF16, 8, MT, true>(); \
      break;                                 \
    case 4:                                  \
      f.template run<RowF16, 4, MT, true>(); \
      break;                                 \
    case 2:                                  \
      f.template run<RowF16, 2, MT, true>(); \
      break;                                 \
    default:                                 \
      f.template run<RowF16, 1, MT, true>(); \
      break;                                 \
  }
  if (M <= 16) {
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
  template <typename Row, int WV, int MT>
  void run() {
    launch_gemv_act<Row, WV, MT>(blocks, st, x, w, sc, y, inj, N, K, act_cols,
                                 inj_cols, act_scale);
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
  int H, R;
  template <typename Row, int WV, int MT>
  void run() {
    launch_hc_mix<Row, WV, MT>(blocks, st, lora, w, sc, xn, out, H, R);
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

}  // namespace

std::tuple<at::Tensor, at::Tensor> rdna_gemv_act(
    const at::Tensor& x, const at::Tensor& w,
    const std::optional<at::Tensor>& scale, int64_t act_cols, double act_scale,
    int64_t inj_cols) {
  const int M = x.size(0), K = x.size(1);
  const int64_t rows = act_cols + inj_cols;
  TORCH_CHECK(
      M >= 1 && M <= kMaxM && x.scalar_type() == at::kHalf && x.is_contiguous(),
      "rdna_gemv_act: x must be contiguous fp16 [1..8, K]");
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
             (float)act_scale};
  dispatch(M, wv, w.scalar_type() == at::kChar, f);
  return {lora, inj};
}

at::Tensor rdna_hc_up_gate_mix(const at::Tensor& lora, const at::Tensor& w,
                               const std::optional<at::Tensor>& scale,
                               const at::Tensor& xn, int64_t hc_count) {
  const int M = lora.size(0), R = lora.size(1);
  TORCH_CHECK(hc_count == 4, "rdna_hc_up_gate_mix: hc_count 4 only");
  const int64_t HCH = w.size(0);
  const int H = (int)(HCH / hc_count);
  TORCH_CHECK(M >= 1 && M <= kMaxM && lora.scalar_type() == at::kHalf &&
                  lora.is_contiguous(),
              "rdna_hc_up_gate_mix: lora must be contiguous fp16 [1..8, R]");
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
           R};
  dispatch(M, wv, w.scalar_type() == at::kChar, f);
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
  TORCH_CHECK(M >= 1 && M <= (i8 ? kMaxM : kMaxSeM) &&
                  x.scalar_type() == at::kHalf && x.is_contiguous(),
              "rdna_se_gate_up_silu: x must be contiguous fp16 [1..32, K] "
              "(1..8 with int8 weights)");
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
    dispatch_guarded(M, wv, f);
  return {act, gate};
}

at::Tensor rdna_se_down_gated(const at::Tensor& act, const at::Tensor& w,
                              const std::optional<at::Tensor>& scale,
                              const at::Tensor& gate) {
  const int M = act.size(0), I = act.size(1), H = w.size(0);
  const bool i8 = w.scalar_type() == at::kChar;
  TORCH_CHECK(M >= 1 && M <= (i8 ? kMaxM : kMaxSeM) &&
                  act.scalar_type() == at::kHalf && act.is_contiguous(),
              "rdna_se_down_gated: act must be contiguous fp16 [1..32, I] "
              "(1..8 with int8 weights)");
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
    dispatch_guarded(M, wv, f);
  return out;
}
