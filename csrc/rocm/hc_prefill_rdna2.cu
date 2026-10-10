// Fused hyper-connection (HC) mix for prefill (M > 8 tokens) on gfx1030.
//
// The HC mix of Qwen3.8-Flash-Next (Qwen4Exp) is, per token row m:
//
//   dai[m, :]  = xn[m, :] . W_down^T                  [HC*H] -> [R + HC + pad]
//   lora[m, k] = silu(dai[m, k] / HC)                 k < R
//   out[m, h]  = (1/HC) * sum_c sigmoid(lora[m] . W_up[c*H + h]) * xn[m, c*H + h]
//
// The torch prefill path runs rocBLAS (down), a Triton silu, rocBLAS (up)
// writing the [M, HC*H] gate to memory, and a Triton gate-mix kernel reading it
// back. Two kernels here:
//
//   hc_up_mix_prefill_k  silu + up GEMM + sigmoid + gated mean in one kernel.
//       Each block owns BM tokens x BH hidden columns and computes the 4*BH
//       gate columns feeding them (rows c*H + h of W_up for the four streams),
//       so the gate exists only in registers. silu(x/HC) is applied while
//       staging dai into LDS: neither the silu output nor a contiguous lora
//       copy is materialized. Reads dai as fp16 (rocBLAS output) or fp32 (the
//       split-K accumulator below) and can write the fp16 dai the caller needs
//       for the HC injection logits.
//   gemm_tn_splitk_k     the down projection (N = 336, K = 10240) as a split-K
//       TN GEMM with an fp32 atomic epilogue; its output grid is too small to
//       fill 72 CUs without splitting K. ~1.1-1.2x rocBLAS (TunableOp) at
//       M = 512..2048, ~1.9x at M <= 128. The atomic accumulation order varies
//       run to run (fp32 rounding only).
//
// Both: v_dot2_f32_f16 (fp32 accumulate), wave32, a 16 (hidden/col) x TY
// (token) thread grid, LDS holding one K slab (KT = 32) in k-pair-major half2
// layout (tokens read with ds_read_b128, broadcast across the 16 column lanes),
// next slab prefetched into registers while the current one is consumed.
// Current stream, fresh outputs, no persistent state: CUDA-graph safe.
//
// Rounding matches the torch path where it stores fp16: the silu output and the
// gate (rocBLAS writes an fp16 gate) are rounded to fp16 before use.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <type_traits>

namespace {

constexpr int kHC = 4;
constexpr int kKT = 32;       // K slab (elements)
constexpr int kKP = kKT / 2;  // k-pairs per slab

__device__ __forceinline__ float sigmoid_f(float x) {
  return __builtin_amdgcn_rcpf(1.f + __expf(-x));
}

// Benchmark override: VLLM_RDNA_HC_PREFILL_<name>=<int>, read once.
int env_int(const char* name) {
  const char* e = std::getenv(name);
  return e ? std::atoi(e) : -1;
}

// ---------------------------------------------------------------------------
// silu + up GEMM + sigmoid + gated mean
// ---------------------------------------------------------------------------
template <typename TA, int TY, int TM, int HH>
__global__ void __launch_bounds__(TY * 16)
    hc_up_mix_prefill_k(const TA* __restrict__ dai, const int ld_dai,
                        const half* __restrict__ w, const half* __restrict__ xn,
                        half* __restrict__ out, half* __restrict__ dai16,
                        const int n_dai, const int M, const int H,
                        const int R) {
  constexpr int kThreads = TY * 16;
  constexpr int BM = TY * TM;
  constexpr int BH = 16 * HH;
  constexpr int BN = kHC * BH;
  constexpr int SA = BM + 4;  // half2 per k-pair row; multiple of 4 (b128)
  constexpr int SB = BN + 2;  // 4*SB = 8 (mod 32): conflict-free staging
  constexpr int A_LD = BM * 4 / kThreads;  // 8-element chunks per thread
  constexpr int B_LD = BN * 4 / kThreads;
  constexpr int A_U4 = sizeof(TA) / 2;  // uint4 per 8-element chunk
  static_assert(A_LD >= 1 && B_LD >= 1, "tile too small");

  __shared__ half2 As[kKP][SA];
  __shared__ half2 Bs[kKP][SB];

  const int t = threadIdx.x;
  const int tx = t % 16, ty = t / 16;
  const int h0 = blockIdx.x * BH;
  const int m0 = blockIdx.y * BM;
  const int HCH = kHC * H;

  const TA* a_src[A_LD];
  bool a_ok[A_LD];
  int a_m[A_LD], a_j[A_LD];
#pragma unroll
  for (int i = 0; i < A_LD; i++) {
    const int idx = t + kThreads * i;
    a_m[i] = idx / 4;
    a_j[i] = idx % 4;
    const int row = m0 + a_m[i];
    a_ok[i] = row < M;
    a_src[i] = dai + (size_t)(a_ok[i] ? row : 0) * ld_dai + a_j[i] * 8;
  }
  const half* b_src[B_LD];
  int b_n[B_LD], b_j[B_LD];
#pragma unroll
  for (int i = 0; i < B_LD; i++) {
    const int idx = t + kThreads * i;
    b_n[i] = idx / 4;
    b_j[i] = idx % 4;
    const int c = b_n[i] / BH, hh = b_n[i] % BH;
    b_src[i] = w + (size_t)(c * H + h0 + hh) * R + b_j[i] * 8;
  }

  uint4 ra[A_LD][A_U4], rb[B_LD];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < A_LD; i++)
#pragma unroll
      for (int u = 0; u < A_U4; u++)
        ra[i][u] =
            a_ok[i] ? reinterpret_cast<const uint4*>(a_src[i] + k0)[u]
                    : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      rb[i] = *reinterpret_cast<const uint4*>(b_src[i] + k0);
  };
  auto stage = [&]() {
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float2 f;
        if constexpr (std::is_same_v<TA, half>) {
          f = __half22float2(reinterpret_cast<const half2*>(ra[i])[q]);
        } else {
          f = reinterpret_cast<const float2*>(ra[i])[q];
        }
        f.x *= (1.f / kHC);
        f.y *= (1.f / kHC);
        f.x = f.x * sigmoid_f(f.x);
        f.y = f.y * sigmoid_f(f.y);
        As[a_j[i] * 4 + q][a_m[i]] = __floats2half2_rn(f.x, f.y);
      }
    }
#pragma unroll
    for (int i = 0; i < B_LD; i++) {
      const half2* hv = reinterpret_cast<const half2*>(&rb[i]);
#pragma unroll
      for (int q = 0; q < 4; q++) Bs[b_j[i] * 4 + q][b_n[i]] = hv[q];
    }
  };

  // fp32 split-K accumulator -> the fp16 dai the caller slices the injection
  // logits from (one column of blocks writes it).
  if constexpr (!std::is_same_v<TA, half>) {
    if (dai16 != nullptr && blockIdx.x == 0) {
      for (int e = t; e < BM * n_dai; e += kThreads) {
        const int r = m0 + e / n_dai, col = e % n_dai;
        if (r < M)
          dai16[(size_t)r * n_dai + col] =
              __float2half(dai[(size_t)r * ld_dai + col]);
      }
    }
  }

  float acc[TM][kHC][HH];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int c = 0; c < kHC; c++)
#pragma unroll
      for (int e = 0; e < HH; e++) acc[i][c][e] = 0.f;

  fetch(0);
  for (int k0 = 0; k0 < R; k0 += kKT) {
    __syncthreads();  // previous slab fully consumed
    stage();
    __syncthreads();
    if (k0 + kKT < R) fetch(k0 + kKT);  // overlaps with the dot loop
#pragma unroll
    for (int kp = 0; kp < kKP; kp++) {
      half2 a[TM];
#pragma unroll
      for (int i = 0; i < TM; i += 4)
        *reinterpret_cast<uint4*>(&a[i]) =
            *reinterpret_cast<const uint4*>(&As[kp][ty * TM + i]);
      half2 b[kHC][HH];
#pragma unroll
      for (int c = 0; c < kHC; c++) {
        if constexpr (HH == 2) {
          *reinterpret_cast<uint2*>(&b[c][0]) =
              *reinterpret_cast<const uint2*>(&Bs[kp][c * BH + tx * 2]);
        } else {
          b[c][0] = Bs[kp][c * BH + tx];
        }
      }
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int c = 0; c < kHC; c++)
#pragma unroll
          for (int e = 0; e < HH; e++)
            acc[i][c][e] =
                __builtin_amdgcn_fdot2(a[i], b[c][e], acc[i][c][e], false);
    }
  }

  // epilogue: sigmoid(gate) * xn, mean over the HC streams
  const int h = h0 + tx * HH;
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + ty * TM + i;
    if (row >= M) continue;
    const half* xr = xn + (size_t)row * HCH + h;
    float mix[HH];
#pragma unroll
    for (int e = 0; e < HH; e++) mix[e] = 0.f;
#pragma unroll
    for (int c = 0; c < kHC; c++) {
      float xv[HH];
      if constexpr (HH == 2) {
        const float2 f =
            __half22float2(*reinterpret_cast<const half2*>(xr + c * H));
        xv[0] = f.x;
        xv[1] = f.y;
      } else {
        xv[0] = __half2float(xr[c * H]);
      }
#pragma unroll
      for (int e = 0; e < HH; e++) {
        const float g = __half2float(__float2half(acc[i][c][e]));
        mix[e] += sigmoid_f(g) * xv[e];
      }
    }
    half* orow = out + (size_t)row * H + h;
    if constexpr (HH == 2) {
      *reinterpret_cast<half2*>(orow) =
          __floats2half2_rn(mix[0] * (1.f / kHC), mix[1] * (1.f / kHC));
    } else {
      orow[0] = __float2half(mix[0] * (1.f / kHC));
    }
  }
}

// ---------------------------------------------------------------------------
// split-K TN GEMM, fp32 atomic epilogue: C[m, n] += sum_k A[m, k] * B[n, k]
// ---------------------------------------------------------------------------
template <int TY, int TM, int NC>
__global__ void __launch_bounds__(TY * 16)
    gemm_tn_splitk_k(const half* __restrict__ A, const int lda,
                     const half* __restrict__ B, const int ldb,
                     float* __restrict__ C, const int M, const int N,
                     const int K, const int k_per_split) {
  constexpr int kThreads = TY * 16;
  constexpr int BM = TY * TM;
  constexpr int BN = 16 * NC;
  constexpr int SA = BM + 4;
  constexpr int SB = BN + ((2 - BN % 8) + 8) % 8;  // SB = 2 (mod 8)
  constexpr int A_LD = BM * 4 / kThreads;
  constexpr int B_LD = (BN * 4 + kThreads - 1) / kThreads;
  static_assert(A_LD >= 1 && BM * 4 % kThreads == 0, "A tile");

  __shared__ half2 As[kKP][SA];
  __shared__ half2 Bs[kKP][SB];

  const int t = threadIdx.x;
  const int tx = t % 16, ty = t / 16;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int kb = blockIdx.z * k_per_split;
  const int ke = min(K, kb + k_per_split);
  if (kb >= ke) return;

  const half* a_src[A_LD];
  bool a_ok[A_LD];
  int a_m[A_LD], a_j[A_LD];
#pragma unroll
  for (int i = 0; i < A_LD; i++) {
    const int idx = t + kThreads * i;
    a_m[i] = idx / 4;
    a_j[i] = idx % 4;
    const int row = m0 + a_m[i];
    a_ok[i] = row < M;
    a_src[i] = A + (size_t)(a_ok[i] ? row : 0) * lda + a_j[i] * 8;
  }
  const half* b_src[B_LD];
  bool b_ok[B_LD], b_in[B_LD];
  int b_n[B_LD], b_j[B_LD];
#pragma unroll
  for (int i = 0; i < B_LD; i++) {
    const int idx = t + kThreads * i;
    b_in[i] = idx < BN * 4;
    b_n[i] = idx / 4;
    b_j[i] = idx % 4;
    const int col = n0 + b_n[i];
    b_ok[i] = b_in[i] && col < N;
    b_src[i] = B + (size_t)(b_ok[i] ? col : 0) * ldb + b_j[i] * 8;
  }

  uint4 ra[A_LD], rb[B_LD];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < A_LD; i++)
      ra[i] = a_ok[i] ? *reinterpret_cast<const uint4*>(a_src[i] + k0)
                      : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      rb[i] = b_ok[i] ? *reinterpret_cast<const uint4*>(b_src[i] + k0)
                      : make_uint4(0, 0, 0, 0);
  };
  auto stage = [&]() {
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
      const half2* hv = reinterpret_cast<const half2*>(&ra[i]);
#pragma unroll
      for (int q = 0; q < 4; q++) As[a_j[i] * 4 + q][a_m[i]] = hv[q];
    }
#pragma unroll
    for (int i = 0; i < B_LD; i++) {
      if (!b_in[i]) continue;
      const half2* hv = reinterpret_cast<const half2*>(&rb[i]);
#pragma unroll
      for (int q = 0; q < 4; q++) Bs[b_j[i] * 4 + q][b_n[i]] = hv[q];
    }
  };

  float acc[TM][NC];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < NC; j++) acc[i][j] = 0.f;

  fetch(kb);
  for (int k0 = kb; k0 < ke; k0 += kKT) {
    __syncthreads();
    stage();
    __syncthreads();
    if (k0 + kKT < ke) fetch(k0 + kKT);
#pragma unroll
    for (int kp = 0; kp < kKP; kp++) {
      half2 a[TM];
#pragma unroll
      for (int i = 0; i < TM; i += 4)
        *reinterpret_cast<uint4*>(&a[i]) =
            *reinterpret_cast<const uint4*>(&As[kp][ty * TM + i]);
      half2 b[NC];
#pragma unroll
      for (int j = 0; j < NC; j++) b[j] = Bs[kp][tx * NC + j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < NC; j++)
          acc[i][j] = __builtin_amdgcn_fdot2(a[i], b[j], acc[i][j], false);
    }
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + ty * TM + i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int col = n0 + tx * NC + j;
      if (col < N) atomicAdd(C + (size_t)row * N + col, acc[i][j]);
    }
  }
}

// kernel launches live in plain template functions (hipify mangles <<<>>>
// inside macros)
template <typename TA, int TY, int TM, int HH>
void launch_up_mix(cudaStream_t st, const TA* dai, int ld, const half* w,
                   const half* xn, half* out, half* dai16, int n_dai, int M,
                   int H, int R) {
  constexpr int BM = TY * TM, BH = 16 * HH;
  const dim3 grid(H / BH, (M + BM - 1) / BM);
  hc_up_mix_prefill_k<TA, TY, TM, HH>
      <<<grid, TY * 16, 0, st>>>(dai, ld, w, xn, out, dai16, n_dai, M, H, R);
}

template <typename TA>
void dispatch_up_mix(cudaStream_t st, const TA* dai, int ld, const half* w,
                     const half* xn, half* out, half* dai16, int n_dai, int M,
                     int H, int R) {
  // Tile: enough blocks to fill 72 CUs at small M, more W_up reuse at large M.
  static const int forced = env_int("VLLM_RDNA_HC_PREFILL_TILE");
  const int sel = forced >= 0 ? forced : (M <= 64 ? 0 : (M <= 384 ? 1 : 2));
  switch (sel) {
    case 0:
      launch_up_mix<TA, 8, 4, 2>(st, dai, ld, w, xn, out, dai16, n_dai, M, H,
                                 R);
      break;
    case 1:
      launch_up_mix<TA, 8, 8, 2>(st, dai, ld, w, xn, out, dai16, n_dai, M, H,
                                 R);
      break;
    default:
      launch_up_mix<TA, 16, 8, 2>(st, dai, ld, w, xn, out, dai16, n_dai, M, H,
                                  R);
      break;
  }
}

template <int TY, int TM, int NC>
void launch_down(cudaStream_t st, const half* A, int lda, const half* B,
                 int ldb, float* C, int M, int N, int K, int target_blocks) {
  constexpr int BM = TY * TM, BN = 16 * NC;
  const int mn = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
  static const int forced = env_int("VLLM_RDNA_HC_PREFILL_SPLIT");
  int splits = forced > 0 ? forced : (target_blocks + mn / 2) / mn;
  splits = std::max(1, std::min(splits, 16));
  const int kps = ((K + splits - 1) / splits + kKT - 1) / kKT * kKT;
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, (K + kps - 1) / kps);
  gemm_tn_splitk_k<TY, TM, NC>
      <<<grid, TY * 16, 0, st>>>(A, lda, B, ldb, C, M, N, K, kps);
}

void check_up_mix_args(const at::Tensor& w_up, const at::Tensor& xn, int64_t M,
                       int64_t R, int64_t hc_count) {
  TORCH_CHECK(hc_count == kHC, "rdna_hc prefill: hc_count 4 only");
  TORCH_CHECK(w_up.dim() == 2 && w_up.scalar_type() == at::kHalf &&
                  w_up.is_contiguous() && w_up.size(1) == R && R % kKT == 0 &&
                  w_up.size(0) % (kHC * 32) == 0,
              "rdna_hc prefill: w_up must be contiguous fp16 [HC*H, R], "
              "R % 32 == 0, H % 32 == 0");
  TORCH_CHECK(xn.dim() == 2 && xn.scalar_type() == at::kHalf &&
                  xn.is_contiguous() && xn.size(0) == M &&
                  xn.size(1) == w_up.size(0),
              "rdna_hc prefill: xn must be contiguous fp16 [M, HC*H]");
}

}  // namespace

// silu + up GEMM + sigmoid + gated mean on a precomputed fp16 dai
// ([M, >= R], unit inner stride, row stride % 8 == 0, 16-byte aligned).
at::Tensor rdna_hc_up_gate_mix_prefill(const at::Tensor& dai,
                                       const at::Tensor& w_up,
                                       const at::Tensor& xn,
                                       int64_t lora_rank, int64_t hc_count) {
  const int64_t M = dai.size(0), R = lora_rank;
  check_up_mix_args(w_up, xn, M, R, hc_count);
  TORCH_CHECK(dai.dim() == 2 && dai.scalar_type() == at::kHalf &&
                  dai.stride(1) == 1 && dai.stride(0) % 8 == 0 &&
                  dai.size(1) >= R &&
                  reinterpret_cast<uintptr_t>(dai.const_data_ptr()) % 16 == 0,
              "rdna_hc_up_gate_mix_prefill: dai must be fp16 [M, >=R], unit "
              "inner stride, 16-byte aligned rows");
  const int H = (int)(w_up.size(0) / kHC);
  const at::cuda::OptionalCUDAGuard guard(dai.device());
  auto out = at::empty({M, H}, xn.options());
  if (M == 0) return out;
  dispatch_up_mix<half>(
      at::cuda::getCurrentCUDAStream(),
      reinterpret_cast<const half*>(dai.const_data_ptr()), (int)dai.stride(0),
      reinterpret_cast<const half*>(w_up.const_data_ptr()),
      reinterpret_cast<const half*>(xn.const_data_ptr()),
      reinterpret_cast<half*>(out.mutable_data_ptr()), nullptr, 0, (int)M, H,
      (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Whole HC prefill mix: split-K down GEMM into an fp32 accumulator, then the
// fused up/mix kernel. Returns (block_input [M, H], dai [M, N_down] fp16).
std::tuple<at::Tensor, at::Tensor> rdna_hc_mix_prefill(
    const at::Tensor& xn, const at::Tensor& w_down, const at::Tensor& w_up,
    int64_t lora_rank, int64_t hc_count) {
  const int64_t M = xn.size(0), K = xn.size(1), R = lora_rank;
  check_up_mix_args(w_up, xn, M, R, hc_count);
  TORCH_CHECK(w_down.dim() == 2 && w_down.scalar_type() == at::kHalf &&
                  w_down.is_contiguous() && w_down.size(1) == K &&
                  K % kKT == 0 && w_down.size(0) >= R &&
                  w_down.size(0) % 4 == 0,
              "rdna_hc_mix_prefill: w_down must be contiguous fp16 [N, HC*H], "
              "N >= R, N % 4 == 0");
  const int N = (int)w_down.size(0), H = (int)(w_up.size(0) / kHC);
  const at::cuda::OptionalCUDAGuard guard(xn.device());
  auto out = at::empty({M, H}, xn.options());
  auto dai16 = at::empty({M, N}, xn.options());
  if (M == 0) return {out, dai16};
  auto acc = at::zeros({M, N}, xn.options().dtype(at::kFloat));
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const half* px = reinterpret_cast<const half*>(xn.const_data_ptr());
  const half* pwd = reinterpret_cast<const half*>(w_down.const_data_ptr());
  float* pacc = acc.mutable_data_ptr<float>();
  // Split K to ~72 blocks (one per CU) for large tiles; small M uses narrow
  // tiles and more splits (latency-bound otherwise).
  if (M < 256) {
    launch_down<8, 8, 3>(st, px, (int)K, pwd, (int)K, pacc, (int)M, N, (int)K,
                         144);
  } else {
    launch_down<16, 8, 7>(st, px, (int)K, pwd, (int)K, pacc, (int)M, N,
                          (int)K, 72);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dispatch_up_mix<float>(st, pacc, N,
                         reinterpret_cast<const half*>(w_up.const_data_ptr()),
                         px, reinterpret_cast<half*>(out.mutable_data_ptr()),
                         reinterpret_cast<half*>(dai16.mutable_data_ptr()), N,
                         (int)M, H, (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, dai16};
}
