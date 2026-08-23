// GatedDeltaNet (GDN) chunked prefill final-output kernel for AMD RDNA2
// (gfx1030). Hand port of chunk_fwd_kernel_o from
// vllm/third_party/flash_linear_attention/ops/chunk_o.py (chunk_fwd_o).
//
// Per (V-tile, chunk, batch*head) workgroup:
//   1. Stage q[BT, K]   into LDS (16 KB fp16)
//   2. Stage k[BT, K]   into LDS (16 KB fp16); accumulate
//      b_A[BT, BT] = q @ k^T (fp32 in registers, V_DOT2_F32_F16)
//   3. Stage h[BV, K]   into LDS (8 KB fp16); accumulate
//      b_o[BT, BV] = (q @ h^T) * scale * exp(g[:,None])
//   4. Apply gating b_A *= exp(g_j - g_i) and INCLUSIVE >= causal mask
//      (zero upper triangle + OOB rows/cols) — mask is INCLUSIVE >= per
//      chunk_o.py:124; chunk_scaled_dot_kkt uses strict > which is a
//      DIFFERENT mask and is not used here.
//   5. Spill the masked/scaled b_A[BT, BT] fp32 to LDS (reuses s_q's
//      16 KB slot, so the LDS budget never grows), then load v[BT, BV]
//      into s_v_T[oc, c] (oc outer, c inner — 4 KB transposed).
//   6. bo += (b_A.to(fp16) @ v) * scale via V_DOT2_F32_F16 using the
//      transposed v so that (v[c], v[c+1]) is a contiguous half2 along
//      the c-axis. Without the transpose, the default [BT, BV] layout
//      pairs v[c, oc] with v[c, oc+1] (wrong axis) and the dot is wrong.
//   7. Store b_o[BT, BV] fp16 to global, zeroing rows >= T_local.
//
// Tile: BT=64, BV=32, K=128 (FLA_CHUNK_SIZE=64 fixed by reference).
//       256 threads; each owns a 4x4 fp32 sub-block of b_A and a 4x2 fp32
//       sub-block of b_o. b_A and b_o live in registers; only the staged
//       tiles (q, k, h, v, s_bA) traverse LDS. V_DOT2_F32_F16 via
//       __builtin_amdgcn_fdot2. h is templated on fp16/fp32.
//
// All arithmetic mirrors the Triton reference's V_DOT2 lowering:
//   * q, k, h, v are fp16 -> V_DOT2_F32_F16 (native RDNA2).
//   * b_A is cast fp32 -> fp16 via __float2half_rn (rtne) before the
//     final b_A @ v dot, matching tl.dot(b_A.to(b_v.dtype), b_v).
//   * g is fp32 (post chunk_local_cumsum); expf only, NEVER __expf.
//   * Causal mask: INCLUSIVE >= (chunk_o.py:124). The companion
//     chunk_scaled_dot_kkt.py uses strict >; do NOT confuse the two.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

// Tile dimensions — match vllm/third_party/flash_linear_attention/ops/utils.py
// FLA_CHUNK_SIZE=64 for the Qwen3.5/3.6 GDN path. K and V default to 128.
constexpr int GDN_BT = 64;
constexpr int GDN_BV = 32;
constexpr int GDN_K = 128;
constexpr int GDN_V = 128;
constexpr int GDN_THREADS = 256;

// Sub-block owned by one thread inside the [BT, BT] fp32 b_A tile and
// the [BT, BV] fp32 b_o tile. 4 rows x 4 cols of A = 16 fp32; 4 rows x 2
// cols of o = 8 fp32 (BV=32 -> col-block size halves vs A's 4).
constexpr int GDN_AROW = 4;
constexpr int GDN_ACOL = 4;
constexpr int GDN_OCOL = 2;
static_assert(GDN_BT / GDN_AROW * GDN_BT / GDN_ACOL == GDN_THREADS,
              "thread count must equal (BT/AROW)*(BT/ACOL)");
static_assert(GDN_BT / GDN_AROW * GDN_BV / GDN_OCOL == GDN_THREADS,
              "thread count must also equal (BT/AROW)*(BV/OCOL)");

// LDS byte sizes (fp16 = 2 bytes, fp32 = 4 bytes).
constexpr int GDN_LDS_Q_BYTES = GDN_BT * GDN_K * 2;     // 16384 (s_q [BT,K] fp16)
constexpr int GDN_LDS_K_BYTES = GDN_BT * GDN_K * 2;     // 16384 (s_k [BT,K] fp16)
constexpr int GDN_LDS_H_BYTES = GDN_BV * GDN_K * 2;     //  8192 (s_h [BV,K] fp16)
constexpr int GDN_LDS_V_BYTES = GDN_BT * GDN_BV * 2;    //  4096 (s_v_T [BV,BT] fp16)
constexpr int GDN_LDS_G_FLOATS = GDN_BT;                // 64 * 4 = 256 bytes
// s_bA [BT,BT] fp32 = 16384 bytes reuses s_q's former 16384-byte slot, so
// the LDS budget never grows. We allocate s_q and s_bA at the same
// offset and use them in sequence (separated by __syncthreads()).
constexpr int GDN_LDS_TOTAL_BYTES =
    GDN_LDS_Q_BYTES + GDN_LDS_K_BYTES + GDN_LDS_H_BYTES +
    GDN_LDS_V_BYTES + GDN_LDS_G_FLOATS * sizeof(float);

// V_DOT2_F32_F16 helper: a.x*b.x + a.y*b.y + acc (native RDNA2).
__device__ __forceinline__ float gdn_fdot2(half2 a, half2 b, float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, false);
}

// Cooperative load of [ROWS, COLS] tile into LDS as fp16. Templated on
// the source dtype: HT = __half (direct copy) or HT = float (rtne cast
// from fp32). For the q/k/v loads HT is always __half; for the h load
// HT matches the user's h tensor dtype.
//
// valid_rows: number of rows in [0, ROWS) that are in-bounds for the
// current chunk. Rows >= valid_rows are zero-filled (so the caller does
// NOT read OOB from global). For q/k (T dimension), valid_rows =
// max(0, T_local - i_t*BT). For h (V dimension), V==128==ROWS and
// is always in-bounds.
template <typename HT, int ROWS, int COLS>
__device__ __forceinline__ void gdn_lds_load(__half* dst,
                                            const HT* src,
                                            long src_stride,
                                            int valid_rows) {
  constexpr int TOTAL = ROWS * COLS;
  constexpr int HALF2_TOTAL = TOTAL / 2;
  constexpr int HALF2_PER_THREAD = HALF2_TOTAL / GDN_THREADS;
  static_assert(HALF2_TOTAL % GDN_THREADS == 0,
                "TOTAL/2 must be divisible by GDN_THREADS");
  static_assert(HALF2_TOTAL * 2 == TOTAL,
                "TOTAL must be even for half2 vectorized loads");
  // Zero-fill so OOB rows are already 0 even if we skip the store.
  // For __half (vectorized) we init all lanes then overwrite valid rows;
  // for float we only write the valid rows in the loop below.
  if (std::is_same<HT, __half>::value) {
#pragma unroll
    for (int s = 0; s < HALF2_PER_THREAD; ++s) {
      const int idx = s * GDN_THREADS + threadIdx.x;
      const int row = idx / (COLS / 2);
      const long loff = (long)row * COLS;
      dst[loff] = __float2half(0.0f);
      dst[loff + 1] = __float2half(0.0f);
    }
  }
#pragma unroll
  for (int s = 0; s < HALF2_PER_THREAD; ++s) {
    const int idx = s * GDN_THREADS + threadIdx.x;
    const int row = idx / (COLS / 2);
    const int col_v = (idx % (COLS / 2)) * 2;
    if (row >= valid_rows) continue;  // OOB rows: leave as zero (or skip store)
    const long goff = (long)row * src_stride + col_v;
    const long loff = (long)row * COLS + col_v;
    if constexpr (std::is_same<HT, float>::value) {
      dst[loff]     = __float2half_rn(src[goff]);
      dst[loff + 1] = __float2half_rn(src[goff + 1]);
    } else {
      *reinterpret_cast<half2*>(&dst[loff]) =
          *reinterpret_cast<const half2*>(&src[goff]);
    }
  }
}

// Boundary-checked loader for the v tile, transposed into s_v_T[oc, c]
// (oc outer, c inner). Loads v[c, oc] from global row-major and stores
// to LDS at the transposed offset oc*BT + c. Rows c >= valid_rows are
// zero-filled so the final b_A @ v dot contributes nothing for OOB c.
//
// Why transpose: the b_A @ v dot pairs bA[r, c] (contiguous in s_bA
// row-major) with v[c, oc] (also contiguous in [BV, BT] row-major
// along the inner c axis). V_DOT2 needs the .x/.y halves to match the
// (c, c+1) axis — which is satisfied iff s_v_T is [BV, BT] row-major
// so that s_v_T[oc, c..c+1] is a single half2. Without the transpose,
// the default [BT, BV] layout pairs v[c, oc] with v[c, oc+1] (wrong
// axis) and the dot produces garbage.
template <int BT, int BV>
__device__ __forceinline__ void gdn_lds_load_v_transposed(
    __half* dst,                  // s_v_T[BV, BT] row-major (oc outer)
    const __half* src,            // v[BT, BV] row-major in global
    long src_stride,              // global row stride for v
    int valid_rows) {             // rows in [0, BT) that are in-bounds
  constexpr int TOTAL = BT * BV;
  constexpr int PER_THREAD = TOTAL / GDN_THREADS;
  static_assert(TOTAL % GDN_THREADS == 0,
                "BT*BV must be divisible by GDN_THREADS");
#pragma unroll
  for (int s = 0; s < PER_THREAD; ++s) {
    const int idx = s * GDN_THREADS + threadIdx.x;
    const int c = idx / BV;
    const int oc = idx % BV;
    const long goff = (long)c * src_stride + oc;
    const long loff = (long)oc * BT + c;
    if (c < valid_rows) {
      dst[loff] = src[goff];
    } else {
      dst[loff] = __float2half(0.0f);
    }
  }
}

// Store the thread's 4x4 fp32 b_A sub-block to LDS at s_bA[r, c] row-
// major. Each thread writes 16 fp32 = 64 bytes; the full grid writes
// the entire [BT, BT] tile (4096 fp32 = 16 KB).
__device__ __forceinline__ void gdn_store_bA_to_lds(
    float* dst,                    // s_bA[BT * BT] fp32 row-major
    const float bA[GDN_AROW][GDN_ACOL],
    int rb, int cb) {
  constexpr int COLS_TOTAL = GDN_BT;
#pragma unroll
  for (int dr = 0; dr < GDN_AROW; ++dr) {
#pragma unroll
    for (int dc = 0; dc < GDN_ACOL; ++dc) {
      const int r = rb * GDN_AROW + dr;
      const int c = cb * GDN_ACOL + dc;
      dst[(long)r * COLS_TOTAL + c] = bA[dr][dc];
    }
  }
}

template <typename HT>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(2, 4))) gdn_prefill_o_rdna2_kernel(
        const __half* __restrict__ q,        // [B, T, Hg, K] fp16
        const __half* __restrict__ k,        // [B, T, Hg, K] fp16
        const __half* __restrict__ v,        // [B, T, H, V]  fp16
        const HT* __restrict__ h,            // 4D varlen or 5D non-varlen
        const float* __restrict__ g_cumsum,  // [B, T, H] fp32
        __half* __restrict__ o,              // [B, T, H, V]  fp16
        const int* __restrict__ cu_seqlens,    // [N+1] int32, unused when !is_varlen
        const int* __restrict__ chunk_offsets, // [N+1] int32, unused when !is_varlen
        int N_seqs,                            // number of sequences in varlen
        float scale,
        long stride_q_tok, long stride_k_tok,
        long stride_v_tok, long stride_o_tok,
        long stride_g_tok, long stride_h_chunk,
        int H, int Hg, int V,
        int T, int NT,
        bool is_varlen) {
  const int i_v = blockIdx.x;
  const int i_t = blockIdx.y;
  const int i_bh = blockIdx.z;
  const int i_b = i_bh / H;
  const int i_h = i_bh % H;
  const int khead = i_h / (H / Hg);

  const int t = threadIdx.x;
  const int rb = t / 16;   // 0..15 -> 4-row blocks within b_A / b_o
  const int cb = t % 16;   // 0..15 -> 4-col blocks of b_A, 2-col of b_o

  // Varlen / non-varlen index resolution — matches chunk_o.py:66-82 but
  // uses FLA's chunk_offsets [N+1] convention (see chunk_delta_h.py:74).
  // chunk_offsets[i_n] = total chunks of all sequences before i_n.
  // For B == 1 (varlen flattened batch), i_n is the sequence this chunk
  // belongs to, found by walking chunk_offsets.
  // In varlen mode i_t is already the global flat chunk index, so
  // i_tg == i_t. The walk below only determines i_n to read bos / T_local.
  int bos, T_local, i_tg;
  if (is_varlen) {
    int i_n = 0;
    while (i_n < N_seqs && chunk_offsets[i_n + 1] <= i_t) {
      i_n++;
    }
    bos = cu_seqlens[i_n];
    const int eos = cu_seqlens[i_n + 1];
    T_local = eos - bos;
    i_tg = i_t;
  } else {
    bos = i_b * T;
    T_local = T;
    i_tg = i_b * NT + i_t;
  }

  // Per-head pointers. All point to sequence start (bos); the chunk offset
  // i_t*BT is added inside the loop for q/k/v/o. g is handled separately
  // (p_g stays at bos, row_local includes the chunk offset).
  __half* p_q = const_cast<__half*>(q + (long)bos * stride_q_tok + (long)khead * GDN_K);
  __half* p_k = const_cast<__half*>(k + (long)bos * stride_k_tok + (long)khead * GDN_K);
  __half* p_v = const_cast<__half*>(v + (long)bos * stride_v_tok + (long)i_h * V);
  __half* p_o = o + (long)bos * stride_o_tok + (long)i_h * V;
  const float* p_g = g_cumsum + (long)bos * stride_g_tok + (long)i_h;
  const HT* p_h = h + (long)i_tg * stride_h_chunk + (long)i_h * V * GDN_K;

  // LDS staging buffers. s_bA aliases s_q's slot; reused after q @ k^T
  // and q @ h^T are done. s_v is the transposed [BV, BT] view of v.
  extern __shared__ __half smem[];
  __half* s_q = smem;                                                // [BT, K]
  __half* s_k = smem + (GDN_LDS_Q_BYTES / 2);                         // [BT, K]
  __half* s_h = smem + (GDN_LDS_Q_BYTES + GDN_LDS_K_BYTES) / 2;      // [BV, K]
  __half* s_v = smem + (GDN_LDS_Q_BYTES + GDN_LDS_K_BYTES +
                        GDN_LDS_H_BYTES) / 2;                         // [BV, BT] transposed
  float* s_bg = reinterpret_cast<float*>(
      smem + (GDN_LDS_Q_BYTES + GDN_LDS_K_BYTES + GDN_LDS_H_BYTES +
              GDN_LDS_V_BYTES) / 2);                                 // [BT] fp32 (per-row)
  float* s_bA = reinterpret_cast<float*>(smem);                      // [BT, BT] fp32 (s_q's slot)

  // Per-thread sub-block accumulators.
  float bA[GDN_AROW][GDN_ACOL];
#pragma unroll
  for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
    for (int dc = 0; dc < GDN_ACOL; ++dc) bA[dr][dc] = 0.0f;
  float bo[GDN_AROW][GDN_OCOL];
#pragma unroll
  for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
    for (int do_ = 0; do_ < GDN_OCOL; ++do_) bo[dr][do_] = 0.0f;

  // Per-row g_cumsum (kept in registers for fast access; the cross-
  // thread b_A gating step uses s_bg to broadcast across the workgroup).
  float bg[GDN_AROW];

  const bool chunk_fully_oob = (i_t * GDN_BT) >= T_local;

  if (!chunk_fully_oob) {
    // Reset bo (b_o) at the START of each chunk — matching reference
    // chunk_o.py:90 "b_o = tl.zeros([BT, BV], dtype=tl.float32)".
    // Reset bA similarly; the reference clears b_A at the start of each
    // i_t iteration (line 91 "b_A = tl.zeros([BT, BT])").
#pragma unroll
    for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
      for (int dc = 0; dc < GDN_ACOL; ++dc) bA[dr][dc] = 0.0f;
#pragma unroll
    for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
      for (int do_ = 0; do_ < GDN_OCOL; ++do_) bo[dr][do_] = 0.0f;

    // ---- Stage 1: load q[BT, K] and k[BT, K] into LDS ----------------
    // Offset pointers by chunk start (i_t*BT) to match reference.
    // q/k/v are [B*T, Hg/H, K/V]; the T-dimension offset is i_t*BT.
    const long chunk_off_qk = (long)i_t * GDN_BT * stride_q_tok;
    const long chunk_off_v  = (long)i_t * GDN_BT * stride_v_tok;
    const int valid_rows = T_local - i_t * GDN_BT;
    gdn_lds_load<__half, GDN_BT, GDN_K>(s_q, p_q + chunk_off_qk, stride_q_tok, valid_rows);
    gdn_lds_load<__half, GDN_BT, GDN_K>(s_k, p_k + chunk_off_qk, stride_k_tok, valid_rows);
    __syncthreads();

    // ---- Dot 1: b_A += q @ k^T (V_DOT2_F32_F16, fp32 accum) ---------
    constexpr int K_PAIRS = GDN_K / 2;
#pragma unroll
    for (int kp = 0; kp < K_PAIRS; ++kp) {
      half2 q_pair[GDN_AROW];
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr) {
        const int row = rb * GDN_AROW + dr;
        q_pair[dr] = *reinterpret_cast<half2*>(
            &s_q[row * GDN_K + kp * 2]);
      }
      half2 k_pair[GDN_ACOL];
#pragma unroll
      for (int dc = 0; dc < GDN_ACOL; ++dc) {
        const int col = cb * GDN_ACOL + dc;
        k_pair[dc] = *reinterpret_cast<half2*>(
            &s_k[col * GDN_K + kp * 2]);
      }
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
        for (int dc = 0; dc < GDN_ACOL; ++dc) {
          bA[dr][dc] = gdn_fdot2(q_pair[dr], k_pair[dc], bA[dr][dc]);
        }
    }
    __syncthreads();

    // ---- Stage 2: load h[BV, K] into LDS (templated on h dtype) -----
    gdn_lds_load<HT, GDN_BV, GDN_K>(
        s_h, p_h + (long)i_v * GDN_BV * GDN_K, GDN_K, GDN_BV);
    __syncthreads();

    // ---- Dot 2: b_o += q @ h^T (V_DOT2_F32_F16) ----------------------
    // q is still resident in s_q; safe to read.
#pragma unroll
    for (int kp = 0; kp < K_PAIRS; ++kp) {
      half2 q_pair[GDN_AROW];
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr) {
        const int row = rb * GDN_AROW + dr;
        q_pair[dr] = *reinterpret_cast<half2*>(
            &s_q[row * GDN_K + kp * 2]);
      }
      half2 h_pair[GDN_OCOL];
#pragma unroll
      for (int do_ = 0; do_ < GDN_OCOL; ++do_) {
        const int oc = cb * GDN_OCOL + do_;
        h_pair[do_] = *reinterpret_cast<half2*>(
            &s_h[oc * GDN_K + kp * 2]);
      }
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
        for (int do_ = 0; do_ < GDN_OCOL; ++do_) {
          bo[dr][do_] = gdn_fdot2(q_pair[dr], h_pair[do_], bo[dr][do_]);
        }
    }
    __syncthreads();

    // ---- Stage 3: load g_cumsum[BT] per row to s_bg ------------------
#pragma unroll
    for (int dr = 0; dr < GDN_AROW; ++dr) {
      const int row_local = i_t * GDN_BT + rb * GDN_AROW + dr;
      bg[dr] = (row_local < T_local)
                   ? p_g[(long)row_local * stride_g_tok]
                   : 0.0f;
      s_bg[rb * GDN_AROW + dr] = bg[dr];
    }
    __syncthreads();

    // ---- Scale b_o (chunk_o.py:137) and apply exp(g) to b_o ---------
    // Mathematically (b_o*scale)*exp(g) == (b_o*exp(g))*scale (fp32 mul
    // is commutative), but the order here mirrors Triton's: scale
    // first, then exp, so the per-element intermediate magnitudes match.
#pragma unroll
    for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
      for (int do_ = 0; do_ < GDN_OCOL; ++do_) bo[dr][do_] *= scale;
#pragma unroll
    for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
      for (int do_ = 0; do_ < GDN_OCOL; ++do_) {
        bo[dr][do_] *= expf(bg[dr]);
      }

    // ---- Gating + INCLUSIVE >= causal mask on b_A --------------------
    // Mirrors chunk_o.py:120 then :124. Gating is applied BEFORE the
    // mask so out-of-bound rows/cols hold a finite (zeroed) value after
    // the where.
    {
      int o_rt[GDN_AROW], o_ct[GDN_ACOL];
      bool m_rt[GDN_AROW], m_ct[GDN_ACOL];
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr) {
        o_rt[dr] = i_t * GDN_BT + rb * GDN_AROW + dr;
        m_rt[dr] = o_rt[dr] < T_local;
      }
#pragma unroll
      for (int dc = 0; dc < GDN_ACOL; ++dc) {
        o_ct[dc] = i_t * GDN_BT + cb * GDN_ACOL + dc;
        m_ct[dc] = o_ct[dc] < T_local;
      }
      // Apply gating first (fp32 multiply) — exp(bg[dr] - bg[col]).
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
        for (int dc = 0; dc < GDN_ACOL; ++dc) {
          const int col_local = cb * GDN_ACOL + dc;
          bA[dr][dc] *= expf(bg[dr] - s_bg[col_local]);
        }
      // Then apply INCLUSIVE >= mask (chunk_o.py:124). Store to s_bA
      // immediately so the final dot can read all 64 columns of bA.
#pragma unroll
      for (int dr = 0; dr < GDN_AROW; ++dr)
#pragma unroll
        for (int dc = 0; dc < GDN_ACOL; ++dc) {
          const bool ok = (o_rt[dr] >= o_ct[dc]) & m_rt[dr] & m_ct[dc];
          if (!ok) bA[dr][dc] = 0.0f;
        }
      gdn_store_bA_to_lds(s_bA, bA, rb, cb);
    }
    __syncthreads();

    // ---- Stage 4: load v[BT, BV] transposed into s_v_T[BV, BT] -------
    // valid_rows counts rows in [0, BT) that are in-bounds for this
    // chunk; OOB rows are zero-filled so the dot's contributions vanish.
    // p_v must include the chunk offset (i_t*BT*H*V) so v[c,oc] maps
    // to v[bos+i_t*BT+c, i_h, i_v*BV+oc].
    int valid_rows_v = T_local - i_t * GDN_BT;
    if (valid_rows_v > GDN_BT) valid_rows_v = GDN_BT;
    if (valid_rows_v < 0) valid_rows_v = 0;
    gdn_lds_load_v_transposed<GDN_BT, GDN_BV>(
        s_v, p_v + chunk_off_v + (long)i_v * GDN_BV,
        stride_v_tok, valid_rows_v);  // stride_v_tok = H*V (row stride in global), NOT V
    __syncthreads();

    // ---- Dot 3: bo += (b_A.to(fp16) @ v) * scale (V_DOT2_F32_F16) ---
    // For each (r, oc) owned by this thread:
    //   bo[r, oc] += sum_{c=0..BT-1} b_A[r, c] * v[c, oc] * scale
    // Because s_v_T[oc, c] is a half2 of (v[c, oc], v[c+1, oc]) along
    // the c-axis, we can do V_DOT2 over c pairs (cp = 0..31).
    {
      constexpr int C_PAIRS = GDN_BT / 2;
#pragma unroll
      for (int cp = 0; cp < C_PAIRS; ++cp) {
        // 4 half2's: bA[r, cp*2..cp*2+1] for each row r in the sub-block.
        half2 bA_pair[GDN_AROW];
#pragma unroll
        for (int dr = 0; dr < GDN_AROW; ++dr) {
          const long bA_off = (long)(rb * GDN_AROW + dr) * GDN_BT + cp * 2;
          bA_pair[dr] = __halves2half2(
              __float2half_rn(s_bA[bA_off]),
              __float2half_rn(s_bA[bA_off + 1]));
        }
#pragma unroll
        for (int dr = 0; dr < GDN_AROW; ++dr) {
#pragma unroll
          for (int do_ = 0; do_ < GDN_OCOL; ++do_) {
            const int oc = cb * GDN_OCOL + do_;
            const long v_off = (long)oc * GDN_BT + cp * 2;
            // s_v_T[oc, cp*2..cp*2+1] = (v[cp*2, oc], v[cp*2+1, oc])
            half2 v_pair = *reinterpret_cast<half2*>(&s_v[v_off]);
            // dot = bA[r, cp*2]*v[cp*2, oc] + bA[r, cp*2+1]*v[cp*2+1, oc]
            float acc = gdn_fdot2(bA_pair[dr], v_pair, 0.0f);
            bo[dr][do_] += acc * scale;
          }
        }
      }
    }
    __syncthreads();
  }  // !chunk_fully_oob

  // ---- Store b_o[BT, BV] fp16 to global, zeroing rows >= T_local ----
#pragma unroll
  for (int dr = 0; dr < GDN_AROW; ++dr) {
    const int row = rb * GDN_AROW + dr;
    const int row_local = i_t * GDN_BT + row;
    if (row_local >= T_local) continue;
#pragma unroll
    for (int do_ = 0; do_ < GDN_OCOL; ++do_) {
      const int oc = cb * GDN_OCOL + do_;
      const int col = i_v * GDN_BV + oc;
      if (col >= V) continue;
      // p_o already includes the i_h*V offset (set at line ~253), so
      // the store offset must NOT re-add i_h*V — that would double-count
      // it and write head>0 output into the next token's head-0 slot.
      const long goff = (long)row_local * stride_o_tok + col;
      p_o[goff] = __float2half(bo[dr][do_]);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host wrapper: shape/dtype/stride checks + launch config. Matches the
// Stage-1 gdn_decode_rdna2 style: regular torch::Tensor args, runtime
// flag for varlen, NULL ptrs for absent index tensors.
// ---------------------------------------------------------------------------

void gdn_prefill_o_rdna2(
    torch::Tensor q,             // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor k,             // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor v,             // [B, T, H, V]  fp16 contiguous in V
    torch::Tensor h,             // 5D [B, NT, H, V, K] (non-varlen) or
                                // 4D [NC, H, V, K] (varlen); fp16/fp32, contiguous in K
    torch::Tensor g,             // [B, T, H] fp32 contiguous
    torch::Tensor o,             // [B, T, H, V]  fp16 contiguous in V
    double scale,
    torch::Tensor cu_seqlens,    // [N+1] int32, empty for non-varlen
    torch::Tensor chunk_offsets) {  // [N+1] int32, empty for non-varlen
  TORCH_CHECK(q.dim() == 4 && q.size(-1) == GDN_K &&
                  q.stride(-1) == 1 && q.scalar_type() == at::kHalf,
              "q must be fp16 [B, T, Hg, ", GDN_K, "], contiguous in last dim");
  TORCH_CHECK(k.dim() == 4 && k.size(-1) == GDN_K &&
                  k.stride(-1) == 1 && k.scalar_type() == at::kHalf,
              "k must be fp16 [B, T, Hg, ", GDN_K, "], contiguous in last dim");
  TORCH_CHECK(v.dim() == 4 && v.size(-1) == GDN_V &&
                  v.stride(-1) == 1 && v.scalar_type() == at::kHalf,
              "v must be fp16 [B, T, H, ", GDN_V, "], contiguous in last dim");
  const auto h_ty = h.scalar_type();
  TORCH_CHECK(h.size(-1) == GDN_K && h.stride(-1) == 1 &&
                  (h_ty == at::kHalf || h_ty == at::kFloat),
              "h must be fp16 or fp32 [..., H, V, ", GDN_K,
              "], contiguous in last dim");
  TORCH_CHECK(o.dim() == 4 && o.size(-1) == GDN_V &&
                  o.stride(-1) == 1 && o.scalar_type() == at::kHalf,
              "o must be fp16 [B, T, H, ", GDN_V, "], contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1 &&
                  g.scalar_type() == at::kFloat,
              "g must be fp32 [B, T, H], contiguous in last dim");

  const int B = q.size(0);
  const int T = q.size(1);
  const int Hg = q.size(2);
  const int H = v.size(2);
  const int V = v.size(3);
  TORCH_CHECK(H % Hg == 0,
              "H (value heads) must be divisible by Hg (key heads) for GQA");
  TORCH_CHECK(V == GDN_V, "V must equal ", GDN_V);
  TORCH_CHECK(B == v.size(0) && B == o.size(0) && B == g.size(0),
              "B mismatch across q/v/o/g");
  TORCH_CHECK(T == k.size(1) && T == v.size(1) && T == o.size(1) &&
                  T == g.size(1),
              "T mismatch across q/k/v/o/g");
  TORCH_CHECK(H == g.size(2),
              "g H dim must match v H dim");

  // h layout:
  //   non-varlen (B >= 1): 5D [B, NT, H, V, K]
  //   varlen (B == 1):     5D [1, NT_total, H, V, K] (same 5D as reference)
  // Both use chunk stride = H*V*K (h.stride(1) for 5D).
  const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() > 0;
  int NT = 0;
  int N_seqs = 0;
  long stride_h_chunk = 0;
  if (is_varlen) {
    TORCH_CHECK(B == 1,
                "varlen mode requires flattened batch B == 1 (FLA convention)");
    TORCH_CHECK(chunk_offsets.defined() && chunk_offsets.numel() > 0,
                "varlen mode requires chunk_offsets");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt &&
                chunk_offsets.scalar_type() == at::kInt &&
                cu_seqlens.dim() == 1 && chunk_offsets.dim() == 1 &&
                cu_seqlens.size(0) == chunk_offsets.size(0),
                "cu_seqlens and chunk_offsets must be int32 [N+1] of equal length");
    // Varlen h is now 5D [1, NT_total, H, V, K] (same as reference).
    TORCH_CHECK(h.dim() == 5 && h.size(0) == 1 && h.size(2) == H &&
                    h.size(3) == V,
                "h must be 5D [1, NT_total, H, V, K] in varlen mode");
    N_seqs = (int)cu_seqlens.size(0) - 1;
    // Total chunk count = h.size(1), the NT dim of the 5D h tensor. Reading
    // chunk_offsets[N_seqs] here would dereference a device pointer on the
    // host and yield garbage, corrupting grid.y. Empty batch -> h.size(1)==0.
    NT = h.size(1);
    stride_h_chunk = h.stride(1);  // 5D stride for NT dimension
  } else {
    TORCH_CHECK(h.dim() == 5 && h.size(0) == B && h.size(2) == H &&
                h.size(3) == V,
                "h must be 5D [B, NT, H, V, K] in non-varlen mode");
    NT = (T + GDN_BT - 1) / GDN_BT;
    TORCH_CHECK(h.size(1) == NT,
                "h NT dim (", h.size(1), ") must equal cdiv(T, BT) = ", NT);
    stride_h_chunk = h.stride(1);
  }

  if (B == 0 || NT == 0) return;

  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((V + GDN_BV - 1) / GDN_BV, NT, B * H);
  const size_t smem_bytes = GDN_LDS_TOTAL_BYTES;

#define GDN_O_LAUNCH(HT)                                              \
  gdn_prefill_o_rdna2_kernel<HT><<<grid, GDN_THREADS, smem_bytes,      \
                                  stream>>>(                          \
      reinterpret_cast<const __half*>(q.data_ptr()),                  \
      reinterpret_cast<const __half*>(k.data_ptr()),                  \
      reinterpret_cast<const __half*>(v.data_ptr()),                  \
      reinterpret_cast<const HT*>(h.data_ptr()),                      \
      g.data_ptr<float>(),                                            \
      reinterpret_cast<__half*>(o.data_ptr()),                        \
      is_varlen ? cu_seqlens.data_ptr<int>() : nullptr,               \
      is_varlen ? chunk_offsets.data_ptr<int>() : nullptr,            \
      N_seqs,                                                         \
      (float)scale,                                                   \
      q.stride(1),                                                    \
      k.stride(1),                                                    \
      v.stride(1),                                                    \
      o.stride(1),                                                    \
      g.stride(1),                                                    \
      stride_h_chunk,                                                 \
      H, Hg, V,                                                       \
      T, NT, is_varlen)

  if (h_ty == at::kFloat) {
    GDN_O_LAUNCH(float);
  } else {
    GDN_O_LAUNCH(__half);
  }
#undef GDN_O_LAUNCH
}
