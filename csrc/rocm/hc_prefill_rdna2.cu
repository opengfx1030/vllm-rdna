// Fused hyper-connection (HC) mix for prefill (M > 8 tokens) on gfx1030.
//
// The HC mix of Qwen3.8-Flash-Next (Qwen4Exp) is, per token row m:
//
//   dai[m, :]  = xn[m, :] . W_down^T                  [HC*H] -> [R + HC + pad]
//   lora[m, k] = silu(dai[m, k] / HC)                 k < R
//   out[m, h]  = (1/HC) * sum_c sigmoid(lora[m] . W_up[c*H + h]) * xn[m, c*H +
//   h]
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
//       split-K accumulator below) and, for the fp32 accumulator, writes the
//       contiguous fp16 HC injection logits (columns R..R+HC of dai) the
//       caller needs, so no strided slice has to be copied afterwards.
//   gemm_tn_splitk_k     the down projection (N = 336, K = 10240) as a split-K
//       TN GEMM; its output grid is too small to fill 72 CUs without splitting
//       K. ~1.1-1.2x rocBLAS (TunableOp) at M = 512..2048, ~1.9x at M <= 128.
//       Every K slice stores its fp32 partial tile into a per-call workspace;
//       the last slice to finish a tile (arrival ticket) sums the slices in
//       ascending order and stores the final fp32 value. No atomics, no
//       zero-filled accumulator (the per-call at::zeros fill was one kernel
//       and one graph-node gap per HC mix, ~100 per decode step), and the
//       result is run-to-run deterministic.
//
// Both: v_dot2_f32_f16 (fp32 accumulate), wave32, a 16 (hidden/col) x TY
// (token) thread grid, LDS holding one K slab (KT = 32) in k-pair-major half2
// layout (tokens read with ds_read_b128, broadcast across the 16 column lanes),
// next slab prefetched into registers while the current one is consumed.
// Current stream, fresh outputs: CUDA-graph safe. The only persistent state is
// the arrival-ticket pool (hc_tickets below): zero between launches, one slot
// per stream, so concurrent streams and captured graphs never share tickets.
//
// Rounding matches the torch path where it stores fp16: the silu output and the
// gate (rocBLAS writes an fp16 gate) are rounded to fp16 before use.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace {

constexpr int kHC = 4;
constexpr int kKT = 32;       // K slab (elements)
constexpr int kKP = kKT / 2;  // k-pairs per slab

__device__ __forceinline__ float sigmoid_f(float x) {
  return __builtin_amdgcn_rcpf(1.f + __expf(-x));
}

// ---------------------------------------------------------------------------
// silu + up GEMM + sigmoid + gated mean
// ---------------------------------------------------------------------------
template <typename TA, int TY, int TM, int HH>
__global__ void __launch_bounds__(TY * 16)
    hc_up_mix_prefill_k(const TA* __restrict__ dai, const int ld_dai,
                        const half* __restrict__ w, const half* __restrict__ xn,
                        half* __restrict__ out, half* __restrict__ inj,
                        const int n_inj, const int M, const int H,
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
        ra[i][u] = a_ok[i] ? reinterpret_cast<const uint4*>(a_src[i] + k0)[u]
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

  // fp32 split-K accumulator -> the contiguous fp16 injection logits (one
  // column of blocks writes them).
  if constexpr (!std::is_same_v<TA, half>) {
    if (inj != nullptr && blockIdx.x == 0) {
      for (int e = t; e < BM * n_inj; e += kThreads) {
        const int r = m0 + e / n_inj, col = e % n_inj;
        if (r < M)
          inj[(size_t)r * n_inj + col] =
              __float2half(dai[(size_t)r * ld_dai + R + col]);
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
// split-K TN GEMM, ticketed fixed-order reduction:
//   C[m, n] = sum_k A[m, k] * B[n, k]
// grid = (N tiles, M tiles, K slices). With one slice the block stores C
// directly. With several, every slice stores its fp32 partial tile to
// ws[z][m][n]; the last slice to take a tile's ticket sums the slices in
// ascending z order into C and re-arms the ticket (tickets are zero between
// launches, see hc_tickets). C and ws need no initialization.
// ---------------------------------------------------------------------------
typedef float f4v __attribute__((ext_vector_type(4)));

template <int TY, int TM, int NC>
__global__ void __launch_bounds__(TY * 16)
    gemm_tn_splitk_k(const half* __restrict__ A, const int lda,
                     const half* __restrict__ B, const int ldb,
                     float* __restrict__ C, float* ws, int* tickets,
                     const int M, const int N, const int K,
                     const int k_per_split) {
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
  __shared__ int s_last;

  const int t = threadIdx.x;
  const int tx = t % 16, ty = t / 16;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int kb = blockIdx.z * k_per_split;
  const int ke = min(K, kb + k_per_split);

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

  if (kb < ke) fetch(kb);
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

  if (gridDim.z == 1) {
    // One K slice: this block owns its outputs outright.
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int row = m0 + ty * TM + i;
      if (row >= M) continue;
#pragma unroll
      for (int j = 0; j < NC; j++) {
        const int col = n0 + tx * NC + j;
        if (col < N) C[(size_t)row * N + col] = acc[i][j];
      }
    }
    return;
  }

  float* wz = ws + (size_t)blockIdx.z * M * N;
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + ty * TM + i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int col = n0 + tx * NC + j;
      if (col < N) wz[(size_t)row * N + col] = acc[i][j];
    }
  }
  // Memory ordering: each thread's partial stores precede its __threadfence()
  // (release, device scope); the fence precedes the block barrier, so thread
  // 0's ticket increment happens after every store of the block is
  // device-visible. The last block's second __threadfence() is the acquire
  // side: the other slices' partials are visible before they are read. Every
  // launched block reaches the ticket (there is no early exit), so the count
  // is exact and the last block re-arms the ticket for the next launch.
  __threadfence();
  __syncthreads();
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  if (t == 0) s_last = atomicAdd(tickets + tile, 1) == (int)gridDim.z - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // Ascending-z sum, 16 bytes per thread-step. N % 4 == 0 and n0 % 16 == 0,
  // so every row segment of the tile is a whole number of aligned f4v.
  const int gz = gridDim.z;
  const int rows = min(BM, M - m0);
  const int c4n = min(BN, N - n0) / 4;
  const size_t zs = (size_t)M * N / 4;  // slice stride in f4v
  for (int e = t; e < rows * c4n; e += kThreads) {
    const int r = e / c4n, c4 = e - r * c4n;
    const f4v* p =
        reinterpret_cast<const f4v*>(ws) + ((size_t)(m0 + r) * N + n0) / 4 + c4;
    f4v sum = {0.f, 0.f, 0.f, 0.f};
    int z = 0;
    for (; z + 8 <= gz; z += 8) {
      f4v v[8];
#pragma unroll
      for (int u = 0; u < 8; u++)
        v[u] = __builtin_nontemporal_load(p + (size_t)(z + u) * zs);
#pragma unroll
      for (int u = 0; u < 8; u++) sum += v[u];
    }
    for (; z < gz; z++) sum += __builtin_nontemporal_load(p + (size_t)z * zs);
    *reinterpret_cast<f4v*>(C + (size_t)(m0 + r) * N + n0 + 4 * c4) = sum;
  }
  if (t == 0) tickets[tile] = 0;
}

// kernel launches live in plain template functions (hipify mangles <<<>>>
// inside macros)
template <typename TA, int TY, int TM, int HH>
void launch_up_mix(cudaStream_t st, const TA* dai, int ld, const half* w,
                   const half* xn, half* out, half* inj, int n_inj, int M,
                   int H, int R) {
  constexpr int BM = TY * TM, BH = 16 * HH;
  const dim3 grid(H / BH, (M + BM - 1) / BM);
  hc_up_mix_prefill_k<TA, TY, TM, HH>
      <<<grid, TY * 16, 0, st>>>(dai, ld, w, xn, out, inj, n_inj, M, H, R);
}

template <typename TA>
void dispatch_up_mix(cudaStream_t st, const TA* dai, int ld, const half* w,
                     const half* xn, half* out, half* inj, int n_inj, int M,
                     int H, int R) {
  // Tile: enough blocks to fill 72 CUs at small M, more W_up reuse at large M.
  const int sel = M <= 64 ? 0 : (M <= 384 ? 1 : 2);
  switch (sel) {
    case 0:
      launch_up_mix<TA, 8, 4, 2>(st, dai, ld, w, xn, out, inj, n_inj, M, H, R);
      break;
    case 1:
      launch_up_mix<TA, 8, 8, 2>(st, dai, ld, w, xn, out, inj, n_inj, M, H, R);
      break;
    default:
      launch_up_mix<TA, 16, 8, 2>(st, dai, ld, w, xn, out, inj, n_inj, M, H, R);
      break;
  }
}

// Grid and K-slice length of the down GEMM: split K until about target_blocks
// blocks are in flight (at most 16 slices).
struct DownPlan {
  dim3 grid;
  int k_per_split;
};

template <int TY, int TM, int NC>
DownPlan plan_down(int M, int N, int K, int target_blocks) {
  constexpr int BM = TY * TM, BN = 16 * NC;
  const int mn = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
  int splits = (target_blocks + mn / 2) / mn;
  splits = std::max(1, std::min(splits, 16));
  const int kps = ((K + splits - 1) / splits + kKT - 1) / kKT * kKT;
  return {dim3((N + BN - 1) / BN, (M + BM - 1) / BM, (K + kps - 1) / kps), kps};
}

template <int TY, int TM, int NC>
void launch_down(cudaStream_t st, const DownPlan& p, const half* A, int lda,
                 const half* B, int ldb, float* C, float* ws, int* tickets,
                 int M, int N, int K) {
  gemm_tn_splitk_k<TY, TM, NC><<<p.grid, TY * 16, 0, st>>>(
      A, lda, B, ldb, C, ws, tickets, M, N, K, p.k_per_split);
}

// ---------------------------------------------------------------------------
// Arrival tickets of the split-K reduction
// ---------------------------------------------------------------------------
// One int per output tile; a split-K launch has at most 96 tiles and a slot
// holds kHcTickets. A slot is zero when a launch starts: the last block of a
// tile re-arms its ticket, so it stays zero between launches and no per-call
// fill is needed. A slot is never shared by two streams (two launches in
// flight would corrupt each other's counts): the pool is allocated and zeroed
// once per device, a stream takes a slot on first use and keeps it. A stream
// that is capturing a graph can neither allocate nor enqueue the fill (it
// would be recorded into the graph), so it takes a slot from the reserved
// capture range; graphs captured on one stream share its slot, which is safe
// because at most one of them replays at a time. With no slot available (the
// pool does not exist yet while capturing, or more streams than slots) the
// call falls back to a zeroed per-call allocation: correct, one extra fill.
// Same scheme as wvSplitKrc (skinny_gemms.cu).
constexpr int kHcTickets = 128;
constexpr int kHcSlots = 8;
constexpr int kHcCaptureSlots = 8;

struct HcTicketPool {
  at::Tensor tickets;  // [(kHcSlots + kHcCaptureSlots) * kHcTickets] int32
  int next_slot = 0;
  int next_capture_slot = 0;
  std::map<cudaStream_t, int> slot_of;
  std::map<cudaStream_t, int> capture_slot_of;
};

// Zeroed tickets for a launch on `st`. `spill` keeps a per-call fallback
// allocation alive for the caller.
int* hc_tickets(const at::Device& dev, cudaStream_t st, at::Tensor& spill) {
  static std::mutex mu;
  static std::map<int, HcTicketPool> pools;
  std::lock_guard<std::mutex> lock(mu);
  HcTicketPool& P = pools[dev.index()];

  // A stream whose capture was already invalidated reports an error; treat
  // that as not capturing, since the capture is lost either way.
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  if (cudaStreamGetCaptureInfo(st, &status, &capture_id) != cudaSuccess) {
    status = cudaStreamCaptureStatusNone;
  }
  const bool capturing = status != cudaStreamCaptureStatusNone;

  const auto int_opts = at::TensorOptions().dtype(at::kInt).device(dev);
  if (!P.tickets.defined() && !capturing) {
    P.tickets = at::zeros({(int64_t)(kHcSlots + kHcCaptureSlots) * kHcTickets},
                          int_opts);
    // The fill runs on `st`; no other stream may be handed a slot before it
    // has finished.
    const cudaError_t err = cudaStreamSynchronize(st);
    TORCH_CHECK(err == cudaSuccess,
                "rdna_hc_mix_prefill: failed to zero the ticket pool: ",
                cudaGetErrorString(err));
  }
  int slot = -1;
  if (P.tickets.defined()) {
    auto& slots = capturing ? P.capture_slot_of : P.slot_of;
    int& next = capturing ? P.next_capture_slot : P.next_slot;
    const int limit = capturing ? kHcCaptureSlots : kHcSlots;
    auto it = slots.find(st);
    if (it == slots.end() && next < limit) {
      it = slots.emplace(st, (capturing ? kHcSlots : 0) + next++).first;
    }
    if (it != slots.end()) slot = it->second;
  }
  if (slot >= 0) {
    return P.tickets.data_ptr<int>() + (size_t)slot * kHcTickets;
  }
  spill = at::zeros({kHcTickets}, int_opts);
  return spill.data_ptr<int>();
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
                                       const at::Tensor& xn, int64_t lora_rank,
                                       int64_t hc_count) {
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
  dispatch_up_mix<half>(at::cuda::getCurrentCUDAStream(),
                        reinterpret_cast<const half*>(dai.const_data_ptr()),
                        (int)dai.stride(0),
                        reinterpret_cast<const half*>(w_up.const_data_ptr()),
                        reinterpret_cast<const half*>(xn.const_data_ptr()),
                        reinterpret_cast<half*>(out.mutable_data_ptr()),
                        nullptr, 0, (int)M, H, (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Whole HC prefill mix: split-K down GEMM into an fp32 buffer, then the fused
// up/mix kernel. Returns (block_input [M, H], injection [M, min(HC, N - R)]
// fp16 contiguous: columns R.. of the down projection).
std::tuple<at::Tensor, at::Tensor> rdna_hc_mix_prefill(const at::Tensor& xn,
                                                       const at::Tensor& w_down,
                                                       const at::Tensor& w_up,
                                                       int64_t lora_rank,
                                                       int64_t hc_count) {
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
  // HC injection logits: columns R..R+HC of the down projection (none when
  // the weight has no injection rows).
  const int n_inj = std::min(kHC, N - (int)R);
  auto inj = at::empty({M, n_inj}, xn.options());
  if (M == 0) return {out, inj};
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const half* px = reinterpret_cast<const half*>(xn.const_data_ptr());
  const half* pwd = reinterpret_cast<const half*>(w_down.const_data_ptr());
  // Split K to ~72 blocks (one per CU) for large tiles; small M uses narrow
  // tiles and more splits (latency-bound otherwise).
  const bool narrow = M < 256;
  const DownPlan plan = narrow ? plan_down<8, 8, 3>((int)M, N, (int)K, 144)
                               : plan_down<16, 8, 7>((int)M, N, (int)K, 72);
  // The down GEMM writes every element of acc (and of its workspace slices):
  // no fill, only the ticket pool is persistent.
  auto acc = at::empty({M, N}, xn.options().dtype(at::kFloat));
  at::Tensor ws, spill;
  float* pws = nullptr;
  int* ptickets = nullptr;
  const int64_t splits = plan.grid.z;
  if (splits > 1) {
    TORCH_CHECK((int64_t)plan.grid.x * plan.grid.y <= kHcTickets,
                "rdna_hc_mix_prefill: too many split-K tiles");
    ws = at::empty({splits, M, N}, xn.options().dtype(at::kFloat));
    pws = ws.mutable_data_ptr<float>();
    ptickets = hc_tickets(xn.device(), st, spill);
  }
  float* pacc = acc.mutable_data_ptr<float>();
  if (narrow) {
    launch_down<8, 8, 3>(st, plan, px, (int)K, pwd, (int)K, pacc, pws, ptickets,
                         (int)M, N, (int)K);
  } else {
    launch_down<16, 8, 7>(st, plan, px, (int)K, pwd, (int)K, pacc, pws,
                          ptickets, (int)M, N, (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dispatch_up_mix<float>(st, pacc, N,
                         reinterpret_cast<const half*>(w_up.const_data_ptr()),
                         px, reinterpret_cast<half*>(out.mutable_data_ptr()),
                         reinterpret_cast<half*>(inj.mutable_data_ptr()), n_inj,
                         (int)M, H, (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, inj};
}
