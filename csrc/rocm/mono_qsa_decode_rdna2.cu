// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MonoKernel QSA (gfx1030, mono_qsa_decode_rdna2; part of E3): QSA sparse
// paged GQA attention for decode / MTP verify rows
// in one launch (today: _qsa_sparse_paged_gqa_splitk_kernel +
// _qsa_merge_splitk_kernel). Each row attends to its selected tokens (a
// valid prefix of `indices`, -1 padded) through the paged K/V cache.
//
// grid (rows * kv_heads, splits), 256 threads = 8 waves. Lane l owns head dims
// 8l..8l+7 (head_dim 256); a wave walks every 8th selected token of its split
// with an online softmax per query head of the group (<= 8 heads), so each
// K/V row is read once for the whole group. The 8 waves combine in LDS in a
// fixed order; the last split of a (row, kv head) to finish merges the
// splits' (output, log2-sum-exp) partials and writes fp16 out. Like the
// Triton kernel, probabilities are rounded to fp16 before P @ V.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>

typedef _Float16 qsa_h2v __attribute__((ext_vector_type(2)));

namespace qsa_dec {

using h2v = ::qsa_h2v;

constexpr int kHD = 256, kThreads = 256, kWaves = 8, kMaxG = 8;
#ifndef QSA_DEC_STEP
  #define QSA_DEC_STEP 2
#endif
constexpr int kStep = QSA_DEC_STEP;  // tokens per wave step
#ifndef QSA_DEC_CHUNK
  #define QSA_DEC_CHUNK 8
#endif
constexpr int kChunk = QSA_DEC_CHUNK;  // tokens a wave resolves at once (<= 32)
constexpr int kMaxS = 64;              // splits the merge holds

struct Args {
  const half* q;  // [M, QH, HD]
  long sq_row, sq_head;
  const half* k;  // [pages, page, KVH, HD]
  const half* v;
  long sk_blk, sk_tok, sk_head, sv_blk, sv_tok, sv_head;
  const int* idx;  // [M, W]
  long sidx;
  const int* btab;  // [R, PW]
  long sbt;
  const int* t2r;  // [M]
  half* out;       // [M, QH, HD]
  long so_row, so_head;
  float* po;    // [S, M, QH, HD] fp32 partial outputs
  float* plse;  // [S, M, QH] log2-sum-exp
  int* cnt;     // [M * KVH] split arrivals (zero between calls)
  int M, QH, KVH, G, W, PAGE, PW, R, NB, S;
  float scale_log2;
};

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

template <int G>
__global__ void __launch_bounds__(kThreads) qsa_dec_k(const Args a) {
  const int row = blockIdx.x / a.KVH, kvh = blockIdx.x % a.KVH,
            split = blockIdx.y;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, d0 = lane * 8;
  const int req = a.t2r[row];
  const bool req_ok = req >= 0 && req < a.R;
  const int h0 = kvh * G;

  // q kept packed (fp16 pairs) for v_dot2_f32_f16: fp32-exact products, half
  // the registers of an fp32 copy (8 waves x G <= 8 heads must not spill)
  h2v q[G][4];
#pragma unroll
  for (int h = 0; h < G; h++) {
    const half* qp = a.q + row * a.sq_row + (long)(h0 + h) * a.sq_head + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(qp);
    const h2v* hv = reinterpret_cast<const h2v*>(&raw);
#pragma unroll
    for (int e = 0; e < 4; e++) q[h][e] = hv[e];
  }
  float m[G], l[G], acc[G][8];
#pragma unroll
  for (int h = 0; h < G; h++) {
    m[h] = -1.0e20f;
    l[h] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[h][e] = 0.f;
  }

  const int j0 = (int)((long)split * a.W / a.S),
            j1 = (int)((long)(split + 1) * a.W / a.S);
  const int* irow = a.idx + row * a.sidx;
  const int* brow = a.btab + (long)(req_ok ? req : 0) * a.sbt;
  // the selection is a valid prefix + -1 padding: a split starting on padding
  // has nothing to attend to (at short contexts about half of them)
  __shared__ int s_empty;
  if (tid == 0) s_empty = a.S > 1 && (j0 >= j1 || !req_ok || irow[j0] < 0);
  __syncthreads();
  const long base = (long)split * a.M + row;
  __shared__ int s_last;
  __shared__ float s_L[kMaxG], s_M[kMaxG];  // split partial, then merge weights
  if (s_empty) {
    if (tid < G) a.plse[base * a.QH + h0 + tid] = -INFINITY;
  } else {
    // a wave takes chunks of 32 selected tokens: lane i resolves token i's page
    // (index + block-table loads in parallel), then the wave walks the chunk
    // with the page / offset broadcast from the owning lane, K / V loads kept 2
    // ahead
#ifndef QSA_DEC_NOLOOP
    for (int c0 = j0 + wave * kChunk; c0 < j1; c0 += kWaves * kChunk) {
      long my_off = -1;
      {
        const int j = c0 + (lane < kChunk ? lane : kChunk);
        const int tok = (j < j1 && req_ok) ? irow[j] : -1;
        const int lp = tok >= 0 ? tok / a.PAGE : 0;
        const int pg = (tok >= 0 && lp < a.PW) ? brow[lp] : -1;
        if (pg >= 0 && pg < a.NB)
          my_off = (long)pg * a.PAGE + (tok - lp * a.PAGE);
      }
      const int n = min(kChunk, j1 - c0);
      auto fetch = [&](int t, uint4& kr, uint4& vr) -> bool {
        const long so = __shfl(my_off, t < n ? t : 0);
        if (t >= n || so < 0) return false;
        const long pg = so / a.PAGE, off = so - pg * a.PAGE;
        kr = *reinterpret_cast<const uint4*>(
            a.k + pg * a.sk_blk + off * a.sk_tok + kvh * a.sk_head + d0);
        vr = *reinterpret_cast<const uint4*>(
            a.v + pg * a.sv_blk + off * a.sv_tok + kvh * a.sv_head + d0);
        return true;
      };
      // 4 tokens per step: their K / V loads are in flight together, and the
      // online softmax is updated once per step (tile max, like the Triton
      // tile)
      for (int t = 0; t < n; t += kStep) {
        uint4 kr[kStep], vr[kStep];
        bool ok[kStep];
        bool any = false;
  #pragma unroll
        for (int u = 0; u < kStep; u++) {
          kr[u] = vr[u] = make_uint4(0, 0, 0, 0);
          ok[u] = fetch(t + u, kr[u], vr[u]);
          any |= ok[u];
        }
        if (!any) continue;
        float sc[kStep][G];
  #pragma unroll
        for (int u = 0; u < kStep; u++) {
          const ::qsa_h2v* kh = reinterpret_cast<const ::qsa_h2v*>(&kr[u]);
  #pragma unroll
          for (int h = 0; h < G; h++) {
            float x = 0.f;
  #pragma unroll
            for (int e = 0; e < 4; e++)
              x = __builtin_amdgcn_fdot2(q[h][e], kh[e], x, false);
            sc[u][h] = x;
          }
        }
  #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
  #pragma unroll
          for (int u = 0; u < kStep; u++)
  #pragma unroll
            for (int h = 0; h < G; h++) sc[u][h] += __shfl_xor(sc[u][h], o);
  #pragma unroll
        for (int h = 0; h < G; h++) {
          float mt = m[h];
  #pragma unroll
          for (int u = 0; u < kStep; u++) {
            sc[u][h] = ok[u] ? sc[u][h] * a.scale_log2 : -1.0e20f;
            mt = fmaxf(mt, sc[u][h]);
          }
          const float al = exp2f(m[h] - mt);
          float ps = 0.f;
  #pragma unroll
          for (int e = 0; e < 8; e++) acc[h][e] *= al;
  #pragma unroll
          for (int u = 0; u < kStep; u++) {
            const float pu = ok[u] ? exp2f(sc[u][h] - mt) : 0.f;
            ps += pu;
            const float p16 = __half2float(__float2half(pu));
            const half* vh = reinterpret_cast<const half*>(&vr[u]);
  #pragma unroll
            for (int e = 0; e < 8; e++) acc[h][e] += p16 * __half2float(vh[e]);
          }
          l[h] = l[h] * al + ps;
          m[h] = mt;
        }
      }
    }
#endif
    // combine the 8 waves in a fixed order
    __shared__ float s_m[kWaves][kMaxG], s_l[kWaves][kMaxG];
    __shared__ float s_acc[kMaxG][kHD];
    if (lane == 0)
#pragma unroll
      for (int h = 0; h < G; h++) {
        s_m[wave][h] = m[h];
        s_l[wave][h] = l[h];
      }
    for (int i = tid; i < G * kHD; i += kThreads) s_acc[i / kHD][i % kHD] = 0.f;
    __syncthreads();
    float mx[G], wl[G];
#pragma unroll
    for (int h = 0; h < G; h++) {
      float mm = -1.0e20f;
      for (int w = 0; w < kWaves; w++) mm = fmaxf(mm, s_m[w][h]);
      mx[h] = mm;
      wl[h] = exp2f(m[h] - mm);
    }
    for (int w = 0; w < kWaves; w++) {
      if (wave == w)
#pragma unroll
        for (int h = 0; h < G; h++)
#pragma unroll
          for (int e = 0; e < 8; e++) s_acc[h][d0 + e] += acc[h][e] * wl[h];
      __syncthreads();
    }
    // this split's partial: o = acc / L, lse = M + log2 L  (L = 0: empty split)
    if (tid < G) {
      float mm = -1.0e20f;
      for (int w = 0; w < kWaves; w++) mm = fmaxf(mm, s_m[w][tid]);
      float L = 0.f;
      for (int w = 0; w < kWaves; w++)
        L += s_l[w][tid] * exp2f(s_m[w][tid] - mm);
      s_L[tid] = L;
      s_M[tid] = mm;
    }
    __syncthreads();
    for (int i = tid; i < G * kHD; i += kThreads) {
      const int h = i / kHD, d = i % kHD;
      const float L = s_L[h];
      const float o = L > 0.f ? s_acc[h][d] / L : 0.f;
      if (a.S == 1) {
        a.out[row * a.so_row + (long)(h0 + h) * a.so_head + d] =
            __float2half(o);
      } else {
        a.po[(base * a.QH + h0 + h) * kHD + d] = o;
        if (d == 0)
          a.plse[base * a.QH + h0 + h] =
              L > 0.f ? s_M[h] + log2f(L) : -INFINITY;
      }
    }
    if (a.S == 1) return;
  }  // non-empty split
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(&a.cnt[blockIdx.x], 1) == a.S - 1;
  __syncthreads();
  if (!s_last) return;
  // acquire at agent scope invalidates this CU's L0/L1, so the other splits'
  // partials can be read with plain (pipelined) loads
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  // merge the splits (last split of this (row, kv head))
  __shared__ float s_w[kMaxG][kMaxS];
  {
    // weights: thread (h, s) for every split of every head of the group
    for (int i = tid; i < G * a.S; i += kThreads) {
      const int hl = i / a.S, s2 = i % a.S;
      s_w[hl][s2] = a.plse[((long)s2 * a.M + row) * a.QH + h0 + hl];
    }
    __syncthreads();
    if (tid < G) {
      float mm = -INFINITY;
      for (int s2 = 0; s2 < a.S; s2++) mm = fmaxf(mm, s_w[tid][s2]);
      float den = 0.f;
      for (int s2 = 0; s2 < a.S; s2++) {
        const float wgt = mm > -INFINITY ? exp2f(s_w[tid][s2] - mm) : 0.f;
        s_w[tid][s2] = wgt;
        den += wgt;
      }
      s_L[tid] = den;
    }
    __syncthreads();
  }
  for (int i = tid; i < G * kHD; i += kThreads) {
    const int hl = i / kHD, d = i % kHD, h = h0 + hl;
    const float* src = a.po + ((long)row * a.QH + h) * kHD + d;
    const long sstride = (long)a.M * a.QH * kHD;
    float num = 0.f;
#pragma unroll 8
    for (int s2 = 0; s2 < a.S; s2++)  // empty splits wrote no output
      if (s_w[hl][s2] != 0.f) num += s_w[hl][s2] * src[s2 * sstride];
    const float den = s_L[hl];
    a.out[row * a.so_row + (long)h * a.so_head + d] =
        __float2half(den > 0.f ? num / den : 0.f);
  }
  if (tid == 0) a.cnt[blockIdx.x] = 0;
}

}  // namespace qsa_dec

int64_t mono_qsa_decode_rdna2_splits(int64_t M, int64_t KVH, int64_t W) {
  int64_t s = 288 / std::max<int64_t>(1, M * KVH);
  s = std::min<int64_t>(std::max<int64_t>(s, 1), qsa_dec::kMaxS);
  // >= one chunk per wave per split
  const int64_t per = (int64_t)qsa_dec::kWaves * qsa_dec::kChunk;
  return std::min<int64_t>(s, std::max<int64_t>(1, (W + per - 1) / per));
}

// out[M, QH, HD] = sparse paged GQA attention of q over the selected tokens.
// scratch: fp32 >= splits * M * QH * (HD + 1); counters: int32 [>= M * KVH],
// zero-initialized once (the kernel leaves them zero).
void mono_qsa_decode_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                           const torch::Tensor& v_cache,
                           const torch::Tensor& indices,
                           const torch::Tensor& block_table,
                           const torch::Tensor& token_to_req,
                           torch::Tensor& out, torch::Tensor& scratch,
                           torch::Tensor& counters, int64_t splits) {
  using namespace qsa_dec;
  const int M = q.size(0), QH = q.size(1), KVH = k_cache.size(2);
  TORCH_CHECK(
      q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf &&
          v_cache.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
      "mono_qsa_decode_rdna2: fp16 q / k / v / out");
  TORCH_CHECK(q.size(2) == kHD && k_cache.size(3) == kHD && q.stride(2) == 1 &&
                  k_cache.stride(3) == 1 && v_cache.stride(3) == 1 &&
                  out.stride(2) == 1,
              "mono_qsa_decode_rdna2: head dim 256, unit last stride");
  TORCH_CHECK(QH % KVH == 0 && QH / KVH <= kMaxG,
              "mono_qsa_decode_rdna2: group <= 8");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.size(0) == M &&
                  indices.stride(1) == 1,
              "mono_qsa_decode_rdna2: indices int32 [M, W]");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt && block_table.stride(1) == 1,
      "mono_qsa_decode_rdna2: block table int32");
  TORCH_CHECK(
      token_to_req.scalar_type() == at::kInt && token_to_req.numel() >= M,
      "mono_qsa_decode_rdna2: token_to_req int32 [M]");
  const int S = (int)splits;
  TORCH_CHECK(S >= 1 && S <= kMaxS, "mono_qsa_decode_rdna2: 1 <= splits <= 64");
  TORCH_CHECK(scratch.scalar_type() == at::kFloat &&
                  scratch.numel() >= (int64_t)S * M * QH * (kHD + 1),
              "mono_qsa_decode_rdna2: scratch too small");
  TORCH_CHECK(counters.scalar_type() == at::kInt &&
                  counters.numel() >= (int64_t)M * KVH,
              "mono_qsa_decode_rdna2: counters too small");
  // 16-byte K/V/Q rows
  TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                  k_cache.stride(0) % 8 == 0 && v_cache.stride(0) % 8 == 0 &&
                  v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0 &&
                  q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "mono_qsa_decode_rdna2: 16-byte aligned rows");
  if (M == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.sq_row = q.stride(0);
  a.sq_head = q.stride(1);
  a.k = reinterpret_cast<const half*>(k_cache.data_ptr());
  a.v = reinterpret_cast<const half*>(v_cache.data_ptr());
  a.sk_blk = k_cache.stride(0);
  a.sk_tok = k_cache.stride(1);
  a.sk_head = k_cache.stride(2);
  a.sv_blk = v_cache.stride(0);
  a.sv_tok = v_cache.stride(1);
  a.sv_head = v_cache.stride(2);
  a.idx = indices.data_ptr<int>();
  a.sidx = indices.stride(0);
  a.btab = block_table.data_ptr<int>();
  a.sbt = block_table.stride(0);
  a.t2r = token_to_req.data_ptr<int>();
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.so_row = out.stride(0);
  a.so_head = out.stride(1);
  a.po = scratch.data_ptr<float>();
  a.plse = a.po + (int64_t)S * M * QH * kHD;
  a.cnt = counters.data_ptr<int>();
  a.M = M;
  a.QH = QH;
  a.KVH = KVH;
  a.G = QH / KVH;
  a.W = indices.size(1);
  a.PAGE = k_cache.size(1);
  a.PW = block_table.size(1);
  a.R = block_table.size(0);
  a.NB = k_cache.size(0);
  a.S = S;
  a.scale_log2 = (1.0f / sqrtf((float)kHD)) * 1.4426950408889634f;
  const dim3 grid(M * KVH, S);
  // clang-format off
  switch (a.G) {
#define QSA_DEC_CASE(g)                                                   \
  case g:                                                                 \
    hipLaunchKernelGGL(qsa_dec_k<g>, grid, dim3(kThreads), 0, stream, a); \
    break;
    QSA_DEC_CASE(1) QSA_DEC_CASE(2) QSA_DEC_CASE(3) QSA_DEC_CASE(4)
    QSA_DEC_CASE(5) QSA_DEC_CASE(6) QSA_DEC_CASE(7) QSA_DEC_CASE(8)
#undef QSA_DEC_CASE
    default:
      TORCH_CHECK(false, "mono_qsa_decode_rdna2: unsupported group");
  }
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_store3_rdna2: the QSA indexer's three per-token cache stores
// (compressed keys at the compressed slots, raw keys and the int64 RoPE
// positions at the raw slots) in one launch. HIP replacement for the Triton
// _store_rows3_kernel in mono/rdna2/qsa.py; same masking (a slot is stored
// only when 0 <= slot < num_blocks * page, so PAD = -1 is skipped).
// Row mapping follows qsa_store_cache_rows_kernel (Waldecir Santos): every
// lane moves one 16-byte vector of a dense row, instead of one row per block.
// One lane group per token: the low half stores the compressed row (+ the
// RoPE row), the high half the raw row. Decode widths (<= 64 tokens) use
// 64-lane groups (more lanes per token, ~3.1 us at 1-24 tokens); wider
// batches use 32-lane groups (16 lanes per 16-vector row, 8 tokens per
// block).
// ---------------------------------------------------------------------------
namespace qsa_store3 {

constexpr int kThreads = 256;

struct Store {
  char* cache;
  const int64_t* slots;
  const char* rows;
  int64_t s_block, s_token, s_dim;  // cache strides, bytes
  int64_t r_row, r_dim;             // row strides, bytes
  int64_t limit;                    // num_blocks * page
  int page, width, esz;
  bool dense;  // both last dims contiguous and width*esz a multiple of 16
};

__device__ __forceinline__ void store_row(const Store& s, int64_t row,
                                          int lane, int lanes) {
  const int64_t slot = s.slots[row];
  if (slot < 0 || slot >= s.limit) return;
  char* dst = s.cache + (slot / s.page) * s.s_block + (slot % s.page) * s.s_token;
  const char* src = s.rows + row * s.r_row;
  if (s.dense) {
    const int vecs = s.width * s.esz / 16;
    for (int v = lane; v < vecs; v += lanes)
      *reinterpret_cast<uint4*>(dst + (int64_t)v * 16) =
          *reinterpret_cast<const uint4*>(src + (int64_t)v * 16);
    return;
  }
  for (int i = lane; i < s.width; i += lanes) {
    const char* e = src + (int64_t)i * s.r_dim;
    char* o = dst + (int64_t)i * s.s_dim;
    if (s.esz == 8)
      *reinterpret_cast<int64_t*>(o) = *reinterpret_cast<const int64_t*>(e);
    else if (s.esz == 4)
      *reinterpret_cast<int32_t*>(o) = *reinterpret_cast<const int32_t*>(e);
    else
      *reinterpret_cast<int16_t*>(o) = *reinterpret_cast<const int16_t*>(e);
  }
}

template <int GROUP>
__global__ void __launch_bounds__(kThreads)
    store3_k(Store c, Store r, Store p, int64_t num_rows) {
  constexpr int kHalf = GROUP / 2;
  const int64_t row =
      (int64_t)blockIdx.x * (kThreads / GROUP) + threadIdx.x / GROUP;
  if (row >= num_rows) return;
  const int lane = threadIdx.x % GROUP;
  if (lane < kHalf) {
    store_row(c, row, lane, kHalf);
    store_row(p, row, lane, kHalf);
  } else {
    store_row(r, row, lane - kHalf, kHalf);
  }
}

Store make(const torch::Tensor& cache, const torch::Tensor& slots,
           const torch::Tensor& rows, const char* what) {
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1,
              "mono_qsa_store3_rdna2: ", what, " cache must be [blocks, page, 1, W]");
  TORCH_CHECK(rows.dim() == 2 && rows.size(1) == cache.size(3),
              "mono_qsa_store3_rdna2: ", what, " rows must be [n, W]");
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.size(0) == rows.size(0),
              "mono_qsa_store3_rdna2: ", what, " slots must be int64 [n]");
  TORCH_CHECK(rows.scalar_type() == cache.scalar_type(),
              "mono_qsa_store3_rdna2: ", what, " dtype mismatch");
  Store s;
  s.esz = (int)cache.element_size();
  TORCH_CHECK(s.esz == 2 || s.esz == 4 || s.esz == 8,
              "mono_qsa_store3_rdna2: unsupported element size");
  s.cache = reinterpret_cast<char*>(cache.data_ptr());
  s.slots = slots.data_ptr<int64_t>();
  s.rows = reinterpret_cast<const char*>(rows.data_ptr());
  s.s_block = cache.stride(0) * s.esz;
  s.s_token = cache.stride(1) * s.esz;
  s.s_dim = cache.stride(3) * s.esz;
  s.r_row = rows.stride(0) * s.esz;
  s.r_dim = rows.stride(1) * s.esz;
  s.page = (int)cache.size(1);
  s.width = (int)cache.size(3);
  s.limit = cache.size(0) * cache.size(1);
  const auto aligned = [](const void* ptr, int64_t stride) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) && (stride % 16 == 0);
  };
  s.dense = s.s_dim == s.esz && s.r_dim == s.esz &&
            (s.width * s.esz) % 16 == 0 && aligned(s.rows, s.r_row) &&
            aligned(s.cache, s.s_token) && s.s_block % 16 == 0;
  return s;
}

}  // namespace qsa_store3

void mono_qsa_store3_rdna2(torch::Tensor& c_cache, const torch::Tensor& c_slots,
                           const torch::Tensor& c_rows, torch::Tensor& r_cache,
                           const torch::Tensor& r_slots,
                           const torch::Tensor& r_rows, torch::Tensor& p_cache,
                           const torch::Tensor& p_rows) {
  const int64_t n = c_rows.size(0);
  TORCH_CHECK(r_rows.size(0) == n && p_rows.size(0) == n,
              "mono_qsa_store3_rdna2: the three stores need the same row count");
  if (n == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(c_cache));
  const auto c = qsa_store3::make(c_cache, c_slots, c_rows, "compressed");
  const auto r = qsa_store3::make(r_cache, r_slots, r_rows, "raw");
  const auto p = qsa_store3::make(p_cache, r_slots, p_rows, "rope");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto launch = [&](auto kernel, int group) {
    const int per_block = qsa_store3::kThreads / group;
    const dim3 grid((unsigned)((n + per_block - 1) / per_block));
    hipLaunchKernelGGL(kernel, grid, dim3(qsa_store3::kThreads), 0, stream, c,
                       r, p, n);
  };
  if (n <= 64)
    launch(qsa_store3::store3_k<64>, 64);
  else
    launch(qsa_store3::store3_k<32>, 32);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_k_store_rdna2: the QSA indexer's compressed-key epilogue fused with
// its three cache stores. GemmaRMSNorm (weight + norm_beta) + NeoX RoPE or
// interleaved 3-section MRoPE on the pooled compressed keys (same math as the
// key side of fused_qk_rmsnorm_rope_gate: the normed value is rounded to the
// input dtype before the rotation), written straight into the compressed key
// cache, plus the raw-key and RoPE-position rows at the raw slots. Replaces
// norm_rope + store_rows3 (two launches and an intermediate tensor) on decode.
// Tokens that close no compressed group (slot PAD = -1) skip the norm/RoPE.
// One 64-lane group per token: lanes 0-31 (one wave32) do the compressed key,
// lanes 32-63 the raw row and the RoPE row. head_dim 128, one head, fp16.
// ---------------------------------------------------------------------------
namespace qsa_kstore {

constexpr int kThreads = 256;
constexpr int kGroup = 64;
constexpr int kHD = 128;

struct Args {
  // compressed-key epilogue
  const half* pooled;  // [n, 128]
  int64_t pooled_row;
  const half* weight;  // [128]
  const half* cos_sin; // [max_pos, rotary_dim]
  int64_t cache_row;
  const char* pos;     // [n] or [3, n]
  int64_t pos_m, pos_t;
  int pos_esz;
  int has_mrope, sec_h, sec_w;
  int rotary_dim;
  float eps, beta;
  qsa_store3::Store c;  // destination: compressed key cache (rows unused)
  qsa_store3::Store r;  // raw keys
  qsa_store3::Store p;  // RoPE positions
  int64_t n;
};

__device__ __forceinline__ int64_t load_pos(const Args& a, int m, int64_t t) {
  const char* ptr = a.pos + m * a.pos_m + t * a.pos_t;
  return a.pos_esz == 8 ? *reinterpret_cast<const int64_t*>(ptr)
                        : (int64_t)*reinterpret_cast<const int32_t*>(ptr);
}

__global__ void __launch_bounds__(kThreads) kstore_k(Args a) {
  __shared__ float s_x[kThreads / kGroup][kHD];
  const int g = threadIdx.x / kGroup;
  const int64_t row = (int64_t)blockIdx.x * (kThreads / kGroup) + g;
  if (row >= a.n) return;
  const int lane = threadIdx.x % kGroup;
  if (lane >= 32) {
    qsa_store3::store_row(a.r, row, lane - 32, 32);
    qsa_store3::store_row(a.p, row, lane - 32, 32);
    return;
  }
  const int64_t slot = a.c.slots[row];
  if (slot < 0 || slot >= a.c.limit) return;  // whole wave: same row
  // RMSNorm: each lane holds elements lane + 32 * j.
  const half* x = a.pooled + row * a.pooled_row;
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __half2float(x[lane + 32 * j]);
    ss += v[j] * v[j];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
  const float inv_rms = rsqrtf(ss / kHD + a.eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    const float w = __half2float(a.weight[i]) + a.beta;
    // round to the input dtype before the rotation, like the Triton kernel
    s_x[g][i] = __half2float(__float2half_rn(v[j] * inv_rms * w));
  }
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_s_waitcnt(0);
  half* dst = reinterpret_cast<half*>(a.c.cache + (slot / a.c.page) * a.c.s_block +
                                      (slot % a.c.page) * a.c.s_token);
  const int half_rot = a.rotary_dim / 2;
  const int64_t pt = load_pos(a, 0, row);
  const int64_t ph = a.has_mrope ? load_pos(a, 1, row) : pt;
  const int64_t pw = a.has_mrope ? load_pos(a, 2, row) : pt;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    float out;
    if (i >= a.rotary_dim) {
      out = s_x[g][i];  // pass-through tail (partial rotary)
    } else {
      const int r = i < half_rot ? i : i - half_rot;
      int64_t pos = pt;
      if (a.has_mrope) {
        if ((r % 3 == 1) && (r < 3 * a.sec_h))
          pos = ph;
        else if ((r % 3 == 2) && (r < 3 * a.sec_w))
          pos = pw;
      }
      const half* cs = a.cos_sin + pos * a.cache_row;
      const float c = __half2float(cs[r]);
      const float s = __half2float(cs[half_rot + r]);
      const float x1 = s_x[g][r];
      const float x2 = s_x[g][half_rot + r];
      out = i < half_rot ? x1 * c - x2 * s : x2 * c + x1 * s;
    }
    dst[(int64_t)i * (a.c.s_dim / 2)] = __float2half_rn(out);
  }
}

}  // namespace qsa_kstore

void mono_qsa_k_store_rdna2(
    const torch::Tensor& pooled, const torch::Tensor& weight,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    double eps, double norm_beta, int64_t rotary_dim, int64_t mrope_h,
    int64_t mrope_w, torch::Tensor& c_cache, const torch::Tensor& c_slots,
    torch::Tensor& r_cache, const torch::Tensor& r_slots,
    const torch::Tensor& r_rows, torch::Tensor& p_cache,
    const torch::Tensor& p_rows) {
  const int64_t n = pooled.size(0);
  TORCH_CHECK(pooled.dim() == 2 && pooled.size(1) == qsa_kstore::kHD &&
                  pooled.scalar_type() == torch::kHalf && pooled.stride(1) == 1,
              "mono_qsa_k_store_rdna2: pooled must be fp16 [n, 128], dense rows");
  TORCH_CHECK(weight.numel() == qsa_kstore::kHD &&
                  weight.scalar_type() == torch::kHalf && weight.is_contiguous(),
              "mono_qsa_k_store_rdna2: weight must be fp16 [128]");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kHalf &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.stride(1) == 1 &&
                  cos_sin_cache.size(1) == rotary_dim,
              "mono_qsa_k_store_rdna2: cos_sin_cache must be fp16 [max_pos, rotary_dim]");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= qsa_kstore::kHD && rotary_dim % 2 == 0,
              "mono_qsa_k_store_rdna2: rotary_dim must be even and <= 128");
  TORCH_CHECK(c_cache.scalar_type() == torch::kHalf,
              "mono_qsa_k_store_rdna2: compressed cache must be fp16");
  TORCH_CHECK(r_rows.size(0) == n && p_rows.size(0) == n && c_slots.numel() == n,
              "mono_qsa_k_store_rdna2: row counts must match");
  const bool mrope = positions.dim() == 2;
  TORCH_CHECK((mrope && positions.size(0) == 3 && positions.size(1) == n) ||
                  (!mrope && positions.dim() == 1 && positions.size(0) == n),
              "mono_qsa_k_store_rdna2: positions must be [n] or [3, n]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 ||
                  positions.scalar_type() == torch::kInt32,
              "mono_qsa_k_store_rdna2: positions must be int32/int64");
  if (n == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(pooled));
  qsa_kstore::Args a;
  a.pooled = reinterpret_cast<const half*>(pooled.data_ptr());
  a.pooled_row = pooled.stride(0);
  a.weight = reinterpret_cast<const half*>(weight.data_ptr());
  a.cos_sin = reinterpret_cast<const half*>(cos_sin_cache.data_ptr());
  a.cache_row = cos_sin_cache.stride(0);
  a.pos = reinterpret_cast<const char*>(positions.data_ptr());
  a.pos_esz = (int)positions.element_size();
  a.pos_m = mrope ? positions.stride(0) * a.pos_esz : 0;
  a.pos_t = positions.stride(mrope ? 1 : 0) * a.pos_esz;
  a.has_mrope = mrope ? 1 : 0;
  a.sec_h = (int)mrope_h;
  a.sec_w = (int)mrope_w;
  a.rotary_dim = (int)rotary_dim;
  a.eps = (float)eps;
  a.beta = (float)norm_beta;
  // The compressed Store only needs its slots/cache geometry; rows = pooled.
  a.c = qsa_store3::make(c_cache, c_slots, pooled, "compressed");
  a.r = qsa_store3::make(r_cache, r_slots, r_rows, "raw");
  a.p = qsa_store3::make(p_cache, r_slots, p_rows, "rope");
  a.n = n;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int per_block = qsa_kstore::kThreads / qsa_kstore::kGroup;
  const dim3 grid((unsigned)((n + per_block - 1) / per_block));
  hipLaunchKernelGGL(qsa_kstore::kstore_k, grid, dim3(qsa_kstore::kThreads), 0,
                     stream, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_mqa_rdna2: the QSA indexer scoring (today: Triton
// _qsa_mqa_paged_kernel with EARLY_EXIT) for the MonoKernel.
//
//   logits[r, c] = sum_h relu(q[r, h] . k[c]) / score_divisor,  c < visible[r]
//
// with k[c] read from the paged compressed-key cache. Same outputs as the
// Triton kernel on [0, visible) and the same visible_blocks; tiles that start
// past every row's live context neither score nor store (E3, valid inside
// CUDA graphs where num_columns is the capture-time worst case).
//
// grid (rows, column tiles), WAVES waves per block. Eight lanes own one key
// row (dims 8s..8s+7 and 64+8s..64+8s+7 of lane s: each 16-byte load
// instruction reads 128 contiguous bytes of every key row), so a wave reads 4
// consecutive key rows per step (see mqa_k for the lanes-per-key / unroll
// variants); it issues all steps' loads before computing.
// The rows of one request that are adjacent in the batch (the decode width,
// e.g. 3 for MTP2) are scored together by one block (a group of <= G rows,
// aligned on row % G and split where the request changes): each key row is
// read once for the group, and q stays in registers. Partial dots reduce over
// the eight lanes with xor shuffles.
namespace qsa_mqa {

constexpr int kHD = 128;
constexpr int kHeads = 4;
struct Args {
  const half* q;
  const half* k;
  const int32_t* table;
  const int32_t* t2r;
  const char* pos;
  const char* seq;
  int32_t* vis_out;
  float* logits;
  int64_t q_row, q_head;
  int64_t k_block, k_token;
  int64_t t_req, t_page;
  int64_t l_row;
  int pos_esz, seq_esz;
  int rows, columns, pages, requests, table_width, page_size, ratio;
  int chunked;
  int stream_k;
  float inv_div;
};

typedef unsigned int k16 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int64_t load_int(const char* p, int64_t i, int esz) {
  return esz == 8 ? reinterpret_cast<const int64_t*>(p)[i]
                  : (int64_t)reinterpret_cast<const int32_t*>(p)[i];
}

template <int NL>
__device__ __forceinline__ float dotn(const uint4 (&q)[NL], const uint4 (&k)[NL]) {
  float acc[NL];
#pragma unroll
  for (int j = 0; j < NL; ++j) {
    const qsa_h2v* x = reinterpret_cast<const qsa_h2v*>(&q[j]);
    const qsa_h2v* y = reinterpret_cast<const qsa_h2v*>(&k[j]);
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) t = __builtin_amdgcn_fdot2(x[i], y[i], t, false);
    acc[j] = t;
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NL; ++j) s += acc[j];
  return s;
}

// lane ^ M within a row of 16 lanes, M < 16: a DPP16 row_xmask move (one
// VALU modifier; __shfl_xor lowers to an LDS ds_bpermute).
template <int M>
__device__ __forceinline__ float xor_dpp(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_int(v), 0x160 | M, 0xF, 0xF, false));
}

// sum over the LPK lanes of a key (rows of 16 lanes, LPK <= 16)
template <int LPK>
__device__ __forceinline__ float key_sum(float v) {
  if constexpr (LPK >= 2) v += xor_dpp<1>(v);
  if constexpr (LPK >= 4) v += xor_dpp<2>(v);
  if constexpr (LPK >= 8) v += xor_dpp<4>(v);
  if constexpr (LPK >= 16) v += xor_dpp<8>(v);
  return v;
}

// G rows per group, LPK lanes per key row (NL = 16 / LPK 16-byte loads per
// lane, load j of lane s at dims 8 LPK j + 8 s: every load instruction reads
// 16 LPK contiguous bytes of each key row), U steps of 32 / LPK keys per wave.
// QL: the group's q rows live in LDS (broadcast reads per key) instead of
// G * 4 * NL 16-byte registers per lane, so wide groups keep occupancy.
template <int G, int LPK, int U, bool QL = false>
__global__ void __launch_bounds__(256) mqa_k(const Args a) {
  constexpr int NL = kHD / (8 * LPK);
  __shared__ uint4 s_q[QL ? G : 1][kHeads][kHD / 8];
  constexpr int KPS = 32 / LPK;
  constexpr int KPW = KPS * U;
  static_assert(G <= LPK, "one storing lane per group row");
  const int row0 = blockIdx.x;
  const int32_t req = a.t2r[row0];
  // Group leader: aligned on row % G, or where the request changes.
  if (G > 1 && row0 % G != 0 && a.t2r[row0 - 1] == req) return;
  int gn = 1;
#pragma unroll
  for (int k = 1; k < G; ++k) {
    const int r = row0 + k;
    if (gn == k && r < a.rows && r % G != 0 && a.t2r[r] == req) gn = k + 1;
  }
  const bool req_ok = req >= 0 && req < a.requests;
  const int sreq = min(max((int)req, 0), a.requests - 1);
  const int64_t seq = req_ok ? load_int(a.seq, sreq, a.seq_esz) : 0;
  int64_t vis[G];
  int64_t maxvis = 0;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    vis[g] = 0;
    if (g < gn) {
      const int64_t p = load_int(a.pos, row0 + g, a.pos_esz);
      vis[g] = min((p + 1) / a.ratio, seq / a.ratio);
      maxvis = max(maxvis, vis[g]);
    }
  }
  if (blockIdx.y == 0 && threadIdx.x < (unsigned)gn) {
#pragma unroll
    for (int g = 0; g < G; ++g)
      if (g == (int)threadIdx.x) a.vis_out[row0 + g] = (int32_t)vis[g];
  }
  const int block_n = (int)(blockDim.x / 32) * KPW;
  // E3: column tiles past every row's live context are never visited. The
  // grid holds a bounded number of tile slots per row (a capture-time worst
  // case of MAXLEN / ratio columns would otherwise dispatch thousands of
  // blocks per row that only exit); each block strides over the live tiles.
  if ((int64_t)blockIdx.y * block_n >= maxvis) return;

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int sub = lane % LPK;
  const int kc = lane / LPK;

  uint4 qv[QL ? 1 : G][kHeads][NL];
  if constexpr (QL) {
    // block-uniform: every early return above is taken by the whole block
    for (int i = threadIdx.x; i < G * kHeads * (kHD / 8); i += blockDim.x) {
      const int g = i / (kHeads * (kHD / 8));
      const int h = (i / (kHD / 8)) % kHeads;
      const int c = i % (kHD / 8);
      s_q[g][h][c] =
          g < gn ? *reinterpret_cast<const uint4*>(
                       a.q + (int64_t)(row0 + g) * a.q_row +
                       (int64_t)h * a.q_head + c * 8)
                 : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        const half* qp = a.q + (int64_t)(row0 + g) * a.q_row +
                         (int64_t)h * a.q_head + sub * 8;
#pragma unroll
        for (int j = 0; j < NL; ++j)
          qv[g][h][j] = g < gn ? *reinterpret_cast<const uint4*>(qp + 8 * LPK * j)
                               : make_uint4(0, 0, 0, 0);
      }
  }
  const int32_t* trow = a.table + (int64_t)sreq * a.t_req;
  const float neg_inf = __int_as_float(0xff800000);
  const int64_t live_end = min(maxvis, (int64_t)a.columns);

  // Tile order: strided (tile y, y + slots, ...) or, with a.chunked, one
  // contiguous run of the live tiles per block (DRAM page locality).
  int t_begin = (int)blockIdx.y, t_step = (int)gridDim.y, t_end;
  const int live_tiles = (int)((live_end + block_n - 1) / block_n);
  if (a.chunked) {
    const int per = (live_tiles + (int)gridDim.y - 1) / (int)gridDim.y;
    t_begin = (int)blockIdx.y * per;
    t_end = min(live_tiles, t_begin + per);
    t_step = 1;
  } else {
    t_end = live_tiles;
  }
  for (int t = t_begin; t < t_end; t += t_step) {
    const int tile0 = t * block_n;
    const int wcol0 = tile0 + wave * KPW;
    uint4 kv[U][NL];
    bool ok[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int col = wcol0 + u * KPS + kc;
      const int lp = col / a.page_size;
      bool v = req_ok && col < live_end && lp < a.table_width;
      int32_t pp = -1;
      if (v) {
        pp = trow[(int64_t)lp * a.t_page];
        v = pp >= 0 && pp < a.pages;
      }
      ok[u] = v;
      const half* kp = a.k + (int64_t)max(pp, 0) * a.k_block +
                       (int64_t)(col % a.page_size) * a.k_token + sub * 8;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        kv[u][j] = make_uint4(0, 0, 0, 0);
        if (v) {
          if (a.stream_k) {  // keys are read once: stream them past L2
            const k16 w = __builtin_nontemporal_load(
                reinterpret_cast<const k16*>(kp + 8 * LPK * j));
            kv[u][j] = make_uint4(w.x, w.y, w.z, w.w);
          } else {
            kv[u][j] = *reinterpret_cast<const uint4*>(kp + 8 * LPK * j);
          }
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float d[G][kHeads];
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int h = 0; h < kHeads; ++h)
          if constexpr (QL) {
            uint4 qq[NL];
#pragma unroll
            for (int j = 0; j < NL; ++j) qq[j] = s_q[g][h][LPK * j + sub];
            d[g][h] = key_sum<LPK>(dotn<NL>(qq, kv[u]));
          } else {
            d[g][h] = key_sum<LPK>(dotn<NL>(qv[g][h], kv[u]));
          }
      const int col = wcol0 + u * KPS + kc;
      if (col >= a.columns) continue;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        // a tile at or past this row's live context is not stored (E3)
        if (sub != g || g >= gn || tile0 >= vis[g]) continue;
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < kHeads; ++h) s += fmaxf(d[g][h], 0.f);
        const bool valid = ok[u] && col < vis[g];
        a.logits[(int64_t)(row0 + g) * a.l_row + col] =
            valid ? s * a.inv_div : neg_inf;
      }
    }
  }
}

}  // namespace qsa_mqa

void mono_qsa_mqa_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                        const torch::Tensor& page_table,
                        const torch::Tensor& token_to_req,
                        const torch::Tensor& query_positions,
                        const torch::Tensor& sequence_lengths,
                        torch::Tensor& logits, torch::Tensor& visible_blocks,
                        int64_t compress_ratio, double score_divisor,
                        int64_t group, int64_t waves_arg, int64_t tiles,
                        int64_t lanes_per_key, int64_t unroll, bool q_lds) {
  using namespace qsa_mqa;
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kHD &&
                  q.scalar_type() == torch::kHalf && q.stride(2) == 1 &&
                  q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "mono_qsa_mqa_rdna2: q must be fp16 [rows, 4, 128], dense rows");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1 && k_cache.size(3) == kHD &&
                  k_cache.scalar_type() == torch::kHalf && k_cache.stride(3) == 1 &&
                  k_cache.stride(0) % 8 == 0 && k_cache.stride(1) % 8 == 0,
              "mono_qsa_mqa_rdna2: k_cache must be fp16 [pages, page, 1, 128]");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32 &&
                  token_to_req.scalar_type() == torch::kInt32 &&
                  token_to_req.is_contiguous(),
              "mono_qsa_mqa_rdna2: page_table / token_to_req must be int32");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.stride(1) == 1 &&
                  visible_blocks.scalar_type() == torch::kInt32 &&
                  visible_blocks.is_contiguous(),
              "mono_qsa_mqa_rdna2: logits fp32 [rows, columns], visible int32");
  for (const auto* t : {&query_positions, &sequence_lengths})
    TORCH_CHECK((t->scalar_type() == torch::kInt64 ||
                 t->scalar_type() == torch::kInt32) && t->is_contiguous(),
                "mono_qsa_mqa_rdna2: positions / lengths must be int32/int64");
  TORCH_CHECK(group >= 1 && group <= 4, "mono_qsa_mqa_rdna2: group in 1..4");
  // waves < 0: the same with non-temporal key loads
  const bool stream_k = waves_arg < 0;
  const int64_t waves = std::abs(waves_arg);
  TORCH_CHECK(waves == 2 || waves == 4 || waves == 8,
              "mono_qsa_mqa_rdna2: waves in {2, 4, 8}");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(q.data_ptr()) |
               reinterpret_cast<uintptr_t>(k_cache.data_ptr())) % 16 == 0,
              "mono_qsa_mqa_rdna2: q / k_cache must be 16-byte aligned");
  const int rows = (int)q.size(0);
  const int columns = (int)logits.size(1);
  if (rows == 0 || columns == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  Args a;
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.k = reinterpret_cast<const half*>(k_cache.data_ptr());
  a.table = page_table.data_ptr<int32_t>();
  a.t2r = token_to_req.data_ptr<int32_t>();
  a.pos = reinterpret_cast<const char*>(query_positions.data_ptr());
  a.seq = reinterpret_cast<const char*>(sequence_lengths.data_ptr());
  a.vis_out = visible_blocks.data_ptr<int32_t>();
  a.logits = logits.data_ptr<float>();
  a.q_row = q.stride(0);
  a.q_head = q.stride(1);
  a.k_block = k_cache.stride(0);
  a.k_token = k_cache.stride(1);
  a.t_req = page_table.stride(0);
  a.t_page = page_table.stride(1);
  a.l_row = logits.stride(0);
  a.pos_esz = (int)query_positions.element_size();
  a.seq_esz = (int)sequence_lengths.element_size();
  a.rows = rows;
  a.columns = columns;
  a.pages = (int)k_cache.size(0);
  a.requests = (int)page_table.size(0);
  a.table_width = (int)page_table.size(1);
  a.page_size = (int)k_cache.size(1);
  a.ratio = (int)compress_ratio;
  a.inv_div = (float)(1.0 / score_divisor);
  TORCH_CHECK((lanes_per_key == 2 || lanes_per_key == 4 || lanes_per_key == 8 ||
               lanes_per_key == 16) &&
                  (unroll == 1 || unroll == 2 || unroll == 4 || unroll == 8) &&
                  group <= lanes_per_key,
              "mono_qsa_mqa_rdna2: lanes_per_key in {2, 4, 8, 16} (>= group), "
              "unroll in {1, 2, 4, 8}");
  const int block_n = (int)(waves * (32 / lanes_per_key) * unroll);
  // tile slots per row: |tiles| (0 = every tile of the column bound);
  // tiles < 0 walks a contiguous run of tiles per block instead of a stride
  int ny = (columns + block_n - 1) / block_n;
  if (tiles != 0) ny = std::min<int>(ny, (int)std::abs(tiles));
  a.chunked = tiles < 0 ? 1 : 0;
  a.stream_k = stream_k ? 1 : 0;
  const dim3 grid((unsigned)rows, (unsigned)ny);
  const dim3 block((unsigned)(waves * 32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  bool done = false;
  const auto go = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, grid, block, 0, stream, a);
    done = true;
  };
#define MQA_L(G_, L_, U_)                                                  \
  if (!done && group == G_ && lanes_per_key == L_ && unroll == U_ && !q_lds) \
    go(mqa_k<G_, L_, U_>);
#define MQA_Q(G_, L_, U_)                                                 \
  if (!done && group == G_ && lanes_per_key == L_ && unroll == U_ && q_lds) \
    go(mqa_k<G_, L_, U_, true>);
#define MQA_U(G_, L_) \
  MQA_L(G_, L_, 1) MQA_L(G_, L_, 2) MQA_L(G_, L_, 4) MQA_L(G_, L_, 8)
#define MQA_QU(G_, L_) MQA_Q(G_, L_, 1) MQA_Q(G_, L_, 2) MQA_Q(G_, L_, 4)
  MQA_U(1, 2) MQA_U(2, 2)
  MQA_U(1, 4) MQA_U(2, 4) MQA_U(3, 4) MQA_U(4, 4)
  MQA_U(1, 8) MQA_U(2, 8) MQA_U(3, 8) MQA_U(4, 8)
  MQA_U(1, 16) MQA_U(2, 16) MQA_U(3, 16) MQA_U(4, 16)
  MQA_QU(2, 4) MQA_QU(3, 4) MQA_QU(4, 4)
  MQA_QU(2, 8) MQA_QU(3, 8) MQA_QU(4, 8)
  MQA_QU(2, 16) MQA_QU(3, 16) MQA_QU(4, 16)
#undef MQA_QU
#undef MQA_U
#undef MQA_Q
#undef MQA_L
  TORCH_CHECK(done, "mono_qsa_mqa_rdna2: no kernel for this configuration");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_epilogue_rdna2: the QSA indexer's whole per-token epilogue after
// the qk projection in one launch (today: norm_rope(q) + qsa_compress_groups +
// norm_rope(k) / mono_qsa_k_store_rdna2, plus the positions glue):
//
//   waves 0-3  q head h: GemmaRMSNorm + RoPE / interleaved MRoPE -> q_out
//   wave 4     pool the token's compression group (fp32 sum of `ratio` raw
//              keys: this step's activations, or the compressor ring for
//              members before the chunk; / ratio, rounded to fp16), its first
//              member's RoPE position (activations or the ring's position
//              tail), then norm + RoPE -> compressed key cache (tokens that
//              close no group have a PAD slot and skip)
//   wave 5     raw key row -> ring, RoPE position row -> ring
//
// one block (6 wave32) per token. Same arithmetic as the Triton kernels
// (pooled bitwise, compressed keys to summation order of the RMS).
// Ring reads (wave 4, positions before the chunk) and ring writes (wave 5,
// the chunk's last slots) touch different slots only while every request
// has at most S - ratio + 1 rows in the step (S = ring size): the caller
// gates on max_query_len (decode, MTP verify).
// ---------------------------------------------------------------------------
namespace qsa_epi {

constexpr int kHD = 128;
constexpr int kHeads = 4;
constexpr int kThreads = 6 * 32;

struct Args {
  const half* qk;  // [n, 5*128]: q heads, then k
  int64_t qk_row;
  half* q_out;     // [n, 4, 128] contiguous
  const half* q_w;
  const half* k_w;
  float q_eps, k_eps, beta;
  const half* cos_sin;
  int64_t cache_row;
  int rotary_dim, has_mrope, sec_h, sec_w;
  const char* pos;  // [n] or [3, n]
  int64_t pos_m, pos_t;
  int pos_esz;
  const int32_t* state_table;  // [reqs, >= 1]
  int64_t st_req;
  const int32_t* t2r;
  const int32_t* qsl;          // [reqs + 1]
  const int64_t* end_pos;      // logical positions [n]
  int requests, ratio;
  qsa_store3::Store c;  // compressed key cache, c.slots
  qsa_store3::Store r;  // raw key ring (r.slots = raw slots)
  qsa_store3::Store p;  // RoPE position ring (int64 x 3)
  int64_t n;
};

__device__ __forceinline__ int64_t ld_pos(const Args& a, int m, int64_t t) {
  const char* ptr = a.pos + m * a.pos_m + t * a.pos_t;
  return a.pos_esz == 8 ? *reinterpret_cast<const int64_t*>(ptr)
                        : (int64_t)*reinterpret_cast<const int32_t*>(ptr);
}

// norm + RoPE of one 128-wide head held as v[j] = x[lane + 32 j] (one wave).
__device__ __forceinline__ void norm_rope_store(const Args& a, float (&v)[4],
                                                const half* w, float eps,
                                                int64_t pt, int64_t ph,
                                                int64_t pw, float* s_x,
                                                half* dst, int64_t dst_dim,
                                                int lane) {
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss += v[j] * v[j];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
  const float inv_rms = rsqrtf(ss / kHD + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    s_x[i] = __half2float(
        __float2half_rn(v[j] * inv_rms * (__half2float(w[i]) + a.beta)));
  }
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_s_waitcnt(0);
  const int half_rot = a.rotary_dim / 2;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    float out;
    if (i >= a.rotary_dim) {
      out = s_x[i];
    } else {
      const int r = i < half_rot ? i : i - half_rot;
      int64_t pos = pt;
      if (a.has_mrope) {
        if ((r % 3 == 1) && (r < 3 * a.sec_h))
          pos = ph;
        else if ((r % 3 == 2) && (r < 3 * a.sec_w))
          pos = pw;
      }
      const half* cs = a.cos_sin + pos * a.cache_row;
      const float c = __half2float(cs[r]);
      const float s = __half2float(cs[half_rot + r]);
      const float x1 = s_x[r];
      const float x2 = s_x[half_rot + r];
      out = i < half_rot ? x1 * c - x2 * s : x2 * c + x1 * s;
    }
    dst[(int64_t)i * dst_dim] = __float2half_rn(out);
  }
}

__device__ __forceinline__ const char* ring_at(const qsa_store3::Store& s,
                                               int64_t slot) {
  return s.cache + (slot / s.page) * s.s_block + (slot % s.page) * s.s_token;
}

__global__ void __launch_bounds__(kThreads) epi_k(Args a) {
  __shared__ float s_x[5][kHD];
  const int64_t row = blockIdx.x;
  if (row >= a.n) return;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const half* qk = a.qk + row * a.qk_row;

  if (wave < kHeads) {  // ---- q head
    const half* x = qk + wave * kHD;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half2float(x[lane + 32 * j]);
    const int64_t pt = ld_pos(a, 0, row);
    const int64_t ph = a.has_mrope ? ld_pos(a, 1, row) : pt;
    const int64_t pw = a.has_mrope ? ld_pos(a, 2, row) : pt;
    norm_rope_store(a, v, a.q_w, a.q_eps, pt, ph, pw, s_x[wave],
                    a.q_out + (row * kHeads + wave) * kHD, 1, lane);
    return;
  }
  if (wave == kHeads + 1) {  // ---- raw key + RoPE position rows -> ring
    const int64_t slot = a.r.slots[row];
    if (slot < 0 || slot >= a.r.limit) return;
    const half* k = qk + kHeads * kHD;
    half* dst = reinterpret_cast<half*>(const_cast<char*>(ring_at(a.r, slot)));
    const int64_t dd = a.r.s_dim / 2;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[(lane + 32 * j) * dd] = k[lane + 32 * j];
    if (lane < 3) {
      int64_t* pd =
          reinterpret_cast<int64_t*>(const_cast<char*>(ring_at(a.p, slot)));
      pd[lane * (a.p.s_dim / 8)] = ld_pos(a, a.has_mrope ? lane : 0, row);
    }
    return;
  }
  // ---- wave 4: compressed key
  const int64_t cslot = a.c.slots[row];
  if (cslot < 0 || cslot >= a.c.limit) return;
  const int32_t req = a.t2r[row];
  const bool req_ok = req >= 0 && req < a.requests;
  const int sreq = min(max((int)req, 0), a.requests - 1);
  const int64_t end = a.end_pos[row];
  const int64_t q0 = req_ok ? a.qsl[sreq] : 0;
  const int64_t q1 = req_ok ? a.qsl[sreq + 1] : 0;
  const int64_t chunk0 = end - (row - q0);
  const int32_t blk = req_ok ? a.state_table[(int64_t)sreq * a.st_req] : -1;
  const int S = a.r.page;
  const bool blk_ok = blk >= 0 && (int64_t)blk * S < a.r.limit;
  const bool valid = row >= q0 && row < q1 && req_ok && end >= a.ratio - 1;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int gi = 0; gi < a.ratio; ++gi) {
    const int64_t pos = end - (a.ratio - 1 - gi);
    const bool use_raw = pos >= chunk0;
    const int64_t rr = q0 + pos - chunk0;
    float x[4] = {0.f, 0.f, 0.f, 0.f};
    if (valid && use_raw && rr >= q0 && rr < q1 && rr < a.n) {
      const half* k = a.qk + rr * a.qk_row + kHeads * kHD;
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] = __half2float(k[lane + 32 * j]);
    } else if (valid && !use_raw && blk_ok) {
      const half* k = reinterpret_cast<const half*>(
          ring_at(a.r, (int64_t)blk * S + (pos % S)));
      const int64_t dd = a.r.s_dim / 2;
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] = __half2float(k[(lane + 32 * j) * dd]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] += x[j];
  }
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    v[j] = __half2float(__float2half_rn(acc[j] / (float)a.ratio));
  // the first member's RoPE position (t, h, w)
  const int64_t first = end - a.ratio + 1;
  int64_t fp[3] = {0, 0, 0};
  if (valid) {
    if (first >= chunk0) {
      const int64_t rr = q0 + first - chunk0;
      if (rr >= q0 && rr < q1 && rr < a.n)
#pragma unroll
        for (int m = 0; m < 3; ++m) fp[m] = ld_pos(a, a.has_mrope ? m : 0, rr);
    } else if (blk_ok) {
      const int64_t* pr = reinterpret_cast<const int64_t*>(
          ring_at(a.p, (int64_t)blk * S + (first % S)));
#pragma unroll
      for (int m = 0; m < 3; ++m) fp[m] = pr[m * (a.p.s_dim / 8)];
    }
  }
  half* dst = reinterpret_cast<half*>(const_cast<char*>(ring_at(a.c, cslot)));
  norm_rope_store(a, v, a.k_w, a.k_eps, fp[0], a.has_mrope ? fp[1] : fp[0],
                  a.has_mrope ? fp[2] : fp[0], s_x[4], dst, a.c.s_dim / 2,
                  lane);
}

// Cache geometry of a Store (no source rows: the kernel writes its own).
qsa_store3::Store geom(const torch::Tensor& cache, const torch::Tensor& slots,
                       int64_t n, const char* what) {
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1,
              "mono_qsa_epilogue_rdna2: ", what, " cache must be [blocks, page, 1, W]");
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.size(0) >= n && slots.is_contiguous(),
              "mono_qsa_epilogue_rdna2: ", what, " slots must be int64 [>= n]");
  qsa_store3::Store s{};
  s.esz = (int)cache.element_size();
  s.cache = reinterpret_cast<char*>(cache.data_ptr());
  s.slots = slots.data_ptr<int64_t>();
  s.s_block = cache.stride(0) * s.esz;
  s.s_token = cache.stride(1) * s.esz;
  s.s_dim = cache.stride(3) * s.esz;
  s.page = (int)cache.size(1);
  s.width = (int)cache.size(3);
  s.limit = cache.size(0) * cache.size(1);
  return s;
}

}  // namespace qsa_epi

void mono_qsa_epilogue_rdna2(
    const torch::Tensor& qk, torch::Tensor& q_out, const torch::Tensor& q_w,
    const torch::Tensor& k_w, double q_eps, double k_eps, double norm_beta,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    int64_t rotary_dim, int64_t mrope_h, int64_t mrope_w,
    const torch::Tensor& state_table, const torch::Tensor& token_to_req,
    const torch::Tensor& query_start_loc, const torch::Tensor& end_positions,
    int64_t compress_ratio, torch::Tensor& c_cache, const torch::Tensor& c_slots,
    torch::Tensor& r_cache, const torch::Tensor& r_slots,
    torch::Tensor& p_cache) {
  using namespace qsa_epi;
  const int64_t n = qk.size(0);
  TORCH_CHECK(qk.dim() == 2 && qk.size(1) == (kHeads + 1) * kHD &&
                  qk.scalar_type() == torch::kHalf && qk.stride(1) == 1,
              "mono_qsa_epilogue_rdna2: qk must be fp16 [n, 640], dense rows");
  TORCH_CHECK(q_out.scalar_type() == torch::kHalf && q_out.is_contiguous() &&
                  q_out.numel() == n * kHeads * kHD,
              "mono_qsa_epilogue_rdna2: q_out must be fp16 [n, 4, 128]");
  for (const auto* w : {&q_w, &k_w})
    TORCH_CHECK(w->numel() == kHD && w->scalar_type() == torch::kHalf &&
                    w->is_contiguous(),
                "mono_qsa_epilogue_rdna2: norm weights must be fp16 [128]");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kHalf &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.stride(1) == 1 &&
                  cos_sin_cache.size(1) == rotary_dim && rotary_dim % 2 == 0 &&
                  rotary_dim > 0 && rotary_dim <= kHD,
              "mono_qsa_epilogue_rdna2: cos_sin_cache fp16 [max_pos, rotary_dim]");
  const bool mrope = positions.dim() == 2;
  TORCH_CHECK((mrope && positions.size(0) == 3 && positions.size(1) == n) ||
                  (!mrope && positions.dim() == 1 && positions.size(0) == n),
              "mono_qsa_epilogue_rdna2: positions must be [n] or [3, n]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 ||
                  positions.scalar_type() == torch::kInt32,
              "mono_qsa_epilogue_rdna2: positions int32/int64");
  TORCH_CHECK(state_table.scalar_type() == torch::kInt32 &&
                  token_to_req.scalar_type() == torch::kInt32 &&
                  query_start_loc.scalar_type() == torch::kInt32 &&
                  end_positions.scalar_type() == torch::kInt64 &&
                  token_to_req.is_contiguous() && query_start_loc.is_contiguous() &&
                  end_positions.is_contiguous() && token_to_req.numel() >= n &&
                  end_positions.numel() >= n,
              "mono_qsa_epilogue_rdna2: metadata dtypes/shapes");
  TORCH_CHECK(c_cache.scalar_type() == torch::kHalf &&
                  r_cache.scalar_type() == torch::kHalf &&
                  p_cache.scalar_type() == torch::kInt64 && p_cache.size(3) == 3 &&
                  c_cache.size(3) == kHD && r_cache.size(3) == kHD &&
                  r_cache.size(1) == p_cache.size(1) &&
                  r_cache.size(1) >= compress_ratio && compress_ratio > 0,
              "mono_qsa_epilogue_rdna2: cache layouts");
  if (n == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qk));
  Args a;
  a.qk = reinterpret_cast<const half*>(qk.data_ptr());
  a.qk_row = qk.stride(0);
  a.q_out = reinterpret_cast<half*>(q_out.data_ptr());
  a.q_w = reinterpret_cast<const half*>(q_w.data_ptr());
  a.k_w = reinterpret_cast<const half*>(k_w.data_ptr());
  a.q_eps = (float)q_eps;
  a.k_eps = (float)k_eps;
  a.beta = (float)norm_beta;
  a.cos_sin = reinterpret_cast<const half*>(cos_sin_cache.data_ptr());
  a.cache_row = cos_sin_cache.stride(0);
  a.rotary_dim = (int)rotary_dim;
  a.has_mrope = mrope ? 1 : 0;
  a.sec_h = (int)mrope_h;
  a.sec_w = (int)mrope_w;
  a.pos = reinterpret_cast<const char*>(positions.data_ptr());
  a.pos_esz = (int)positions.element_size();
  a.pos_m = mrope ? positions.stride(0) * a.pos_esz : 0;
  a.pos_t = positions.stride(mrope ? 1 : 0) * a.pos_esz;
  a.state_table = state_table.data_ptr<int32_t>();
  a.st_req = state_table.stride(0);
  a.t2r = token_to_req.data_ptr<int32_t>();
  a.qsl = query_start_loc.data_ptr<int32_t>();
  a.end_pos = end_positions.data_ptr<int64_t>();
  a.requests = (int)query_start_loc.size(0) - 1;
  a.ratio = (int)compress_ratio;
  a.c = geom(c_cache, c_slots, n, "compressed");
  a.r = geom(r_cache, r_slots, n, "raw");
  a.p = geom(p_cache, r_slots, n, "rope");
  a.n = n;
  TORCH_CHECK(a.requests >= 1, "mono_qsa_epilogue_rdna2: no requests");
  const auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(epi_k, dim3((unsigned)n), dim3(kThreads), 0, stream, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
