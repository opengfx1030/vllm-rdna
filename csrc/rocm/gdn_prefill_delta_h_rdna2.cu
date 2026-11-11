// GatedDeltaNet (GDN) chunked-prefill inter-chunk recurrence kernel for
// AMD RDNA2 (gfx1030). Hand port of
// chunk_gated_delta_rule_fwd_kernel_h_blockdim64
// (vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py).
//
// Workgroup = one (V-tile, sequence x value-head). 256 threads = 32 v-rows
// x 8 k-slices, the exact lane layout of gdn_decode_rdna2.cu
// (v = lane >> 3, ks = lane & 7); each thread owns 16 fp32 state
// registers h[v, ks*16 : ks*16+16], register-resident across the entire
// serial chunk loop (no LDS for h, no per-chunk sync). Per chunk:
//   1. store start-of-chunk h (fp32 -> fp16 rtne) for chunk_fwd_o;
//   2. v-correction  v_raw[t] = u[t] - sum_k w[t,k] * fp16(h[bv,k]),
//      reduction over K=128 via the 8-k-slice gdn_ksum shfl. The
//      UNGATED v_raw (fp16 rtne) is written to v_new here, BEFORE
//      gating -- chunk_fwd_o consumes the ungated v_new;
//   3. USE_G scaling: v_corr[t] *= exp(g_last - g[t]); h *= exp(g_last);
//   4. h-update  h[bv,k] += sum_t fp16(k[t,k]) * fp16(v_corr_gated[t]),
//      reduction over BT=64 as a serial t-loop, each thread owning
//      distinct h elements, paired-over-t V_DOT2_F32_F16 inner dots.
//
// Parity-critical fp16 roundings (rtne) of the Triton reference are
// preserved: fp32 h state is rounded to fp16 before the v-correction
// dot, and the gated fp32 v_corr is rounded to fp16 before the h-update
// dot. expf (accurate __ocml_exp) is used throughout, never __expf.
// num_stages=4 pipelining is known-broken for this kernel on the Triton
// AMD backend (chunk_delta_h.py:21); the HIP loop is deliberately
// straightforward (no software pipelining, no double-buffering).
//
// USE_GK (per-key gating, gk) is dead code on the Qwen3.5/3.6 call
// path -- chunk.py:61-71 never passes it -- and is therefore not a
// parameter of the wrapper: it is structurally impossible to silently
// diverge on it. h0 (initial_state) IS supported (default None == zeros).
// The SSM state slot may be fp32 (the default cache) or fp16
// (--mamba-ssm-cache-dtype float16): the recurrence runs in fp32 registers
// either way, so only the h0 load and the final-state store convert.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int GDN_K = 128;        // head_k_dim; fixed by the 8x16 slice layout
constexpr int GDN_V = 128;        // head_v_dim
constexpr int GDN_BT = 64;        // FLA_CHUNK_SIZE
constexpr int GDN_BV = 32;        // V rows per workgroup (V/GDN_BV = 4 tiles)
constexpr int GDN_THREADS = 256;  // 32 v-rows * 8 k-slices

// Reduction across the 8 k-slice lanes sharing one v-row. Lane layout is
// v = lane >> 3, ks = lane & 7, so masks 1/2/4 stay inside the group.
__device__ __forceinline__ float gdn_ksum(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 1);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 2);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 4);
  return x;
}

// SSM state dtype conversions. The state cache may be fp32 (the default) or
// fp16 (--mamba-ssm-cache-dtype float16); the recurrence itself always runs
// in fp32 registers, so only the load and the store convert. This mirrors
// gdn_decode_rdna2.cu, which already accepts an fp16 state.
__device__ __forceinline__ float gdn_to_f32(float x) { return x; }
__device__ __forceinline__ float gdn_to_f32(__half x) {
  return __half2float(x);
}
template <typename ST>
__device__ __forceinline__ ST gdn_to_st(float x);
template <>
__device__ __forceinline__ float gdn_to_st<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half gdn_to_st<__half>(float x) {
  return __float2half_rn(x);
}

// V_DOT2_F32_F16: acc += a.x*b.x + a.y*b.y (fp32 accumulate, no clamp).
__device__ __forceinline__ float gdn_fdot2(__half2 a, __half2 b, float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, /*clamp=*/false);
}

__device__ __forceinline__ __half2 gdn_load_f16x2(const __half* p) {
  return *reinterpret_cast<const __half2*>(p);
}

// waves_per_eu(4, 8) = 4 waves per SIMD32 = 2 workgroups per CU, which is the
// most the 37120 B LDS footprint allows. Without it the compiler sizes the
// register file for 1 WG/CU (193 VGPRs), and then only 12*N workgroups of a
// multi-sequence batch can be resident: +16..17 % vs Triton at N=2 instead of
// -19 %. The unroll factor below is what keeps the allocation inside that
// budget (193 -> 113 VGPRs, no spill).
template <typename ST>
__global__ void __launch_bounds__(GDN_THREADS)
    __attribute__((amdgpu_waves_per_eu(4, 8)))
    gdn_prefill_delta_h_packed_kernel(
        const __half* __restrict__ k,  // [B*T, Hg, K]
        const __half* __restrict__ u,  // [B*T, H, V]
        const __half* __restrict__ w,  // [B*T, H, K]
        const float* __restrict__ g,   // [B*T, H] g_cumsum, fp32
        const ST* __restrict__ h0,     // [N, H, V, K] state dtype, nullable
        __half* __restrict__ h,        // [B, NT, H, V, K] fp16 out (5D always)
        ST* __restrict__ ht,           // [N, H, V, K] state dtype, nullable
        __half* __restrict__ v_new,    // [B*T, H, V] fp16 out
        const int* __restrict__ cu_seqlens,     // [N+1] varlen, or nullptr
        const int* __restrict__ chunk_offsets,  // [N+1] varlen, or nullptr
        int T, int H, int Hg, int N, int B, int NT, int store_final_state) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / H;
  const int i_h = i_nh % H;         // value-head (H == HV)
  const int i_hk = i_h / (H / Hg);  // key-head (GQA grouping)

  const int lane = threadIdx.x;
  const int lane_v = lane >> 3;           // [0, 32)
  const int lane_ks = lane & 7;           // [0, 8)
  const int o_v = i_v * GDN_BV + lane_v;  // V row this thread owns
  const bool v_ok = o_v < GDN_V;
  const int k0 = lane_ks * 16;  // this thread's 16-wide K slice

  // One chunk of this head, staged so the serial loop is not a strided global
  // load per token. k/w are [BT, K], u is this V-tile [BT, BV], g is [BT]:
  // 64*128*2*2 + 64*32*2 + 64*4 = 37120 bytes.
  extern __shared__ __align__(16) char smem_raw[];
  __half* ks = reinterpret_cast<__half*>(smem_raw);
  __half* ws = ks + GDN_BT * GDN_K;
  __half* us = ws + GDN_BT * GDN_K;
  float* gs = reinterpret_cast<float*>(us + GDN_BT * GDN_BV);

  // Sequence bounds. cu_seqlens present => varlen (B == 1, flattened);
  // otherwise uniform-length with per-seq length T.
  int bos, T_seq;
  if (cu_seqlens != nullptr) {
    bos = cu_seqlens[i_n];
    const int eos = cu_seqlens[i_n + 1];
    T_seq = eos - bos;
  } else {
    bos = i_n * T;
    T_seq = T;
  }
  const int NT_seq = (T_seq + GDN_BT - 1) / GDN_BT;
  if (NT_seq <= 0) return;

  // Per-batch base offset into 5D h. The 5D layout is [B, NT, H, V, K].
  // For non-varlen: h[i_b, i_t, i_h, ...] -> flat = (i_b*NT + i_t)*H*V*K + ...
  // For varlen: h[0, flat_i_t, i_h, ...] -> flat = flat_i_t*H*V*K + ...
  long h_base_t;  // flat chunk index added to i_t inside the loop
  if (cu_seqlens != nullptr) {
    // Cumulative chunks before sequence i_n (boh equivalent for 5D)
    h_base_t = (long)chunk_offsets[i_n];
  } else {
    h_base_t = (long)i_n * NT_seq;
  }

  // h tile h[o_v, k0 : k0+16] -> registers (16 fp32). Either from h0 or
  // zeros (the wrapper normalizes h0=None to == zeros up front if it
  // wants, but we also handle it here for convenience).
  float hreg[16];
  if (h0 != nullptr) {
    const ST* p_h0 = h0 + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = v_ok ? gdn_to_f32(p_h0[j]) : 0.0f;
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = 0.0f;
  }

  // Streaming base pointers for this (sequence, head). Triton strides:
  // k/w advance by (head*K), u/v_new by (head*V), k uses Hg not H (GQA).
  const __half* p_k = k + (long)bos * Hg * GDN_K + (long)i_hk * GDN_K + k0;
  const __half* p_w = w + (long)bos * H * GDN_K + (long)i_h * GDN_K + k0;
  const __half* u_base = u + (long)bos * H * GDN_V;
  const float* p_g = g + (long)bos * H + i_h;
  __half* p_vnew = v_new + (long)bos * H * GDN_V + (long)i_h * GDN_V + o_v;
  __half* p_h =
      h + ((h_base_t * H + i_h) * GDN_V * GDN_K + (long)o_v * GDN_K) + k0;
  const long stride_h = (long)H * GDN_V * GDN_K;

  // ---- Serial recurrence over this sequence's chunks -----------------
  for (int i_t = 0; i_t < NT_seq; ++i_t) {
    const int chunk_start = i_t * GDN_BT;
    const int t_len =
        (chunk_start + GDN_BT <= T_seq) ? GDN_BT : (T_seq - chunk_start);
    // The loop body's k/w/u/v_new indexing is chunk-local (t in
    // [0, t_len)); rebase the row pointers by chunk_start so chunk i_t
    // reads/writes its own rows, not chunk 0's.
    const __half* p_w_c = p_w + (long)chunk_start * H * GDN_K;
    __half* p_vnew_c = p_vnew + (long)chunk_start * H * GDN_V;
    const __half* p_k_c = p_k + (long)chunk_start * Hg * GDN_K;

    // Stage this chunk. Token rows are H*K apart in global memory and the
    // inner dots reread them, so one coalesced pass into LDS pays for itself.
    const int n_kw = t_len * GDN_K;
    for (int idx = lane; idx < n_kw; idx += GDN_THREADS) {
      const int t = idx / GDN_K;
      const int kk = idx - t * GDN_K;
      ks[idx] = p_k_c[t * (long)Hg * GDN_K + kk - k0];
      ws[idx] = p_w_c[t * (long)H * GDN_K + kk - k0];
    }
    const int n_u = t_len * GDN_BV;
    for (int idx = lane; idx < n_u; idx += GDN_THREADS) {
      const int t = idx / GDN_BV;
      const int lv = idx - t * GDN_BV;
      us[idx] = u_base[(long)(chunk_start + t) * H * GDN_V + (long)i_h * GDN_V +
                       i_v * GDN_BV + lv];
    }
    if (lane < t_len) {
      gs[lane] = p_g[(long)(chunk_start + lane) * H];
    }
    __syncthreads();

    // 1. Store the pre-update h tile (fp16, rtne) for chunk_fwd_o. The
    //    register state stays fp32; only the persisted copy is rounded.
    if (v_ok) {
#pragma unroll
      for (int j = 0; j < 16; ++j) p_h[j] = __float2half_rn(hreg[j]);
    }

    // 2. v-correction: v_raw[t] = u[t] - sum_k w[t,k] * fp16(h).
    //    h is rounded to fp16 (rtne) once per chunk before the dot;
    //    reduction over K=128 is the 8 k-slice shfl.
    //    The UNGATED v_raw is stored to v_new (fp16 rtne) here, BEFORE
    //    gating -- chunk_fwd_o consumes the ungated v_new.
    __half h_fp16[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h_fp16[j] = __float2half_rn(hreg[j]);

    // 2-4 fused: v-correction, gating and h-update in one pass over t.
    //    v-corr only needs the pre-decay h, already captured in h_fp16, so the
    //    gated v_corr of a token is consumed by the h-update of the same
    //    t-pair and never has to stay live across a loop. Keeping a
    //    float v_corr[GDN_BT] instead and unrolling the h-update loop drove
    //    the kernel to 256 VGPRs with 427 spilled and 1632 B/lane of scratch;
    //    this form needs 55 VGPRs with no spill.
    const float g_last = gs[t_len - 1];
    const float decay = expf(g_last);
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] *= decay;

    //    Pairing over t matches Triton's V_DOT2 lowering of tl.dot(b_k, b_v)
    //    with fp16 operands; each thread owns its 16 h elements, no shfl.
    // 2 WGs/CU caps the VGPR budget at 128/thread; unroll 8 needs 193.
#pragma unroll 2
    for (int tp = 0; tp < GDN_BT / 2; ++tp) {
      const int t0 = 2 * tp;
      const bool valid0 = (t0 < t_len);
      const bool valid1 = (t0 + 1 < t_len);
      float vc0 = 0.0f, vc1 = 0.0f;
      if (valid0) {
        float acc = 0.0f;
        const __half* p_wt = ws + (long)t0 * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half2 a = gdn_load_f16x2(p_wt + 2 * j);
          const __half2 b = __halves2half2(h_fp16[2 * j], h_fp16[2 * j + 1]);
          acc = gdn_fdot2(a, b, acc);
        }
        acc = gdn_ksum(acc);
        const float u_val =
            v_ok ? __half2float(us[(long)t0 * GDN_BV + lane_v]) : 0.0f;
        const float v_raw = u_val - acc;
        // Persist UNGATED v_new (fp16, rtne) -- chunk_fwd_o consumes it
        // ungated. Invalid v-rows skip the store but still need the dot +
        // shfl for warp uniformity.
        if (v_ok) p_vnew_c[(long)t0 * H * GDN_V] = __float2half_rn(v_raw);
        vc0 = v_raw * expf(g_last - gs[t0]);
      }
      if (valid1) {
        float acc = 0.0f;
        const __half* p_wt = ws + (long)(t0 + 1) * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half2 a = gdn_load_f16x2(p_wt + 2 * j);
          const __half2 b = __halves2half2(h_fp16[2 * j], h_fp16[2 * j + 1]);
          acc = gdn_fdot2(a, b, acc);
        }
        acc = gdn_ksum(acc);
        const float u_val =
            v_ok ? __half2float(us[(long)(t0 + 1) * GDN_BV + lane_v]) : 0.0f;
        const float v_raw = u_val - acc;
        if (v_ok) p_vnew_c[(long)(t0 + 1) * H * GDN_V] = __float2half_rn(v_raw);
        vc1 = v_raw * expf(g_last - gs[t0 + 1]);
      }
      // Gated v_corr is rounded to fp16 (rtne) before the dot, as in Triton.
      const __half2 vb =
          __halves2half2(__float2half_rn(vc0), __float2half_rn(vc1));
      __half2 a0[8], a1[8];
      if (valid0) {
        const __half* p_k0 = ks + (long)t0 * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) a0[j] = gdn_load_f16x2(p_k0 + 2 * j);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          a0[j] = __halves2half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
      }
      if (valid1) {
        const __half* p_k1 = ks + (long)(t0 + 1) * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) a1[j] = gdn_load_f16x2(p_k1 + 2 * j);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          a1[j] = __halves2half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        hreg[2 * j] =
            gdn_fdot2(__halves2half2(__low2half(a0[j]), __low2half(a1[j])), vb,
                      hreg[2 * j]);
        hreg[2 * j + 1] =
            gdn_fdot2(__halves2half2(__high2half(a0[j]), __high2half(a1[j])),
                      vb, hreg[2 * j + 1]);
      }
    }

    __syncthreads();
    p_h += stride_h;
  }

  // Epilogue: persist the post-final h register state to the state slot
  // (fp32 keeps the full precision, fp16 rounds to nearest even).
  if (store_final_state && v_ok) {
    ST* p_ht = ht + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) p_ht[j] = gdn_to_st<ST>(hreg[j]);
  }
}

// DPP butterfly over the 8 k-slice lanes (lane & 7): the same x + x^1, then ^2,
// then ^4 association as gdn_ksum, kept in the VALU instead of three dependent
// ds_bpermute round trips through the LDS crossbar. quad_perm 0xB1 = [1,0,3,2]
// and 0x4E = [2,3,0,1]; row_xmask:4 = 0x164 (GFX10 DPP16, stays inside the
// 16-lane row, so inside each group of 8). Every lane ends with the same sum
// because a + b == b + a bitwise, exactly as with the shuffle version.
__device__ __forceinline__ float gdn_ksum_dpp(float x) {
  x += __int_as_float(
      __builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));
  x += __int_as_float(
      __builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));
  x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x164,
                                                  0xF, 0xF, false));
  return x;
}

// Latency-hiding variant of gdn_prefill_delta_h_packed_kernel. Same recurrence,
// same fp16 roundings and operation order (results are bit-identical, NaN
// payloads aside); only the schedule differs:
//  * a chunk is staged with 16-byte loads (a few per thread) instead of ~73
//    two-byte loads issued four at a time, each batch waiting for the previous
//    one (~25 serialized global round trips per chunk);
//  * the next chunk's loads are issued into registers before the recurrence of
//    the current one and stored to LDS after the barrier, so their latency
//    hides behind the compute. The loads are unconditional (rows past the end
//    of a partial chunk re-read the last valid row and are zeroed with a
//    select): conditional struct copies made the compiler keep the buffers in
//    scratch;
//  * rows past the end of a partial chunk are staged as zeros and carry a zero
//    decay, so the token loop has no per-token branches; a padded token adds
//    exactly +0 to h;
//  * the 8-lane k-slice reduction runs on DPP (see gdn_ksum_dpp);
//  * v_new goes into an LDS tile and is stored once per chunk with 16-byte
//    stores instead of one predicated 2-byte global store per token;
//  * the per-token decay exp(g_last - g[t]) is computed once per chunk by 64
//    threads while the chunk is staged, not by every thread for every token;
//  * KPAIR: the k tile is stored in LDS as [token pair][column] half2 words
//    (even token in the low half), so the h update reads ready-made dot2
//    operands instead of packing 16 half2 per token pair with v_perm. The two
//    tokens of a pair sit 16 lanes apart in the same wave and trade halves of
//    their 16-byte pieces with __shfl_xor while the chunk is staged.
template <typename ST, int BV, bool KPAIR>
__global__ void __launch_bounds__(BV * 8)
    __attribute__((amdgpu_waves_per_eu(4, 8))) gdn_prefill_delta_h_pipe_kernel(
        const __half* __restrict__ k,  // [B*T, Hg, K]
        const __half* __restrict__ u,  // [B*T, H, V]
        const __half* __restrict__ w,  // [B*T, H, K]
        const float* __restrict__ g,   // [B*T, H] g_cumsum, fp32
        const ST* __restrict__ h0,     // [N, H, V, K] state dtype, nullable
        __half* __restrict__ h,        // [B, NT, H, V, K] fp16 out (5D always)
        ST* __restrict__ ht,           // [N, H, V, K] state dtype, nullable
        __half* __restrict__ v_new,    // [B*T, H, V] fp16 out
        const int* __restrict__ cu_seqlens,     // [N+1] varlen, or nullptr
        const int* __restrict__ chunk_offsets,  // [N+1] varlen, or nullptr
        int T, int H, int Hg, int N, int B, int NT, int store_final_state) {
  constexpr int THREADS = BV * 8;
  constexpr int KW_TOK = THREADS / 16;        // k/w tokens staged per pass
  constexpr int KW_PASSES = GDN_BT / KW_TOK;  // passes over the 64 tokens
  constexpr int U_CHUNKS = BV / 8;            // 16-byte chunks per u/v_new row
  static_assert(THREADS == GDN_BT * U_CHUNKS, "one u chunk per thread");
  static_assert(GDN_V % BV == 0, "V tiles must cover V exactly");

  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / H;
  const int i_h = i_nh % H;
  const int i_hk = i_h / (H / Hg);

  const int lane = threadIdx.x;
  const int lane_v = lane >> 3;
  const int lane_ks = lane & 7;
  const int o_v = i_v * BV + lane_v;
  const int k0 = lane_ks * 16;

  // Separate static arrays: the compiler can then see that the v_new writes
  // never alias the k/w/u/decay reads and hoist the next tokens' LDS loads
  // above them (with one carved dynamic buffer it had to keep program order).
  __shared__ __align__(16) __half ks[GDN_BT * GDN_K];
  __shared__ __align__(16) __half ws[GDN_BT * GDN_K];
  __shared__ __align__(16) __half us[GDN_BT * BV];
  __shared__ __align__(16) __half vn[GDN_BT * BV];
  __shared__ __align__(16) float gd[GDN_BT];

  int bos, T_seq;
  if (cu_seqlens != nullptr) {
    bos = cu_seqlens[i_n];
    const int eos = cu_seqlens[i_n + 1];
    T_seq = eos - bos;
  } else {
    bos = i_n * T;
    T_seq = T;
  }
  const int NT_seq = (T_seq + GDN_BT - 1) / GDN_BT;
  if (NT_seq <= 0) return;

  long h_base_t;
  if (cu_seqlens != nullptr) {
    h_base_t = (long)chunk_offsets[i_n];
  } else {
    h_base_t = (long)i_n * NT_seq;
  }

  float hreg[16];
  if (h0 != nullptr) {
    const ST* p_h0 = h0 + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = gdn_to_f32(p_h0[j]);
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] = 0.0f;
  }

  const __half* kb = k + (long)bos * Hg * GDN_K + (long)i_hk * GDN_K;
  const __half* wb = w + (long)bos * H * GDN_K + (long)i_h * GDN_K;
  const __half* ub =
      u + (long)bos * H * GDN_V + (long)i_h * GDN_V + (long)i_v * BV;
  const float* p_g = g + (long)bos * H + i_h;
  __half* vb_out =
      v_new + (long)bos * H * GDN_V + (long)i_h * GDN_V + (long)i_v * BV;
  __half* p_h =
      h + ((h_base_t * H + i_h) * GDN_V * GDN_K + (long)o_v * GDN_K) + k0;
  const long stride_h = (long)H * GDN_V * GDN_K;

  const int kw_t = lane >> 4;
  const int kw_c = (lane & 15) * 8;
  const int u_t = lane / U_CHUNKS;
  const int u_c = (lane % U_CHUNKS) * 8;

  uint4 pk[KW_PASSES], pw[KW_PASSES], pu;
  float pg, pgl;
  auto fetch = [&](int chunk_start, int t_len) __attribute__((always_inline)) {
    const int last = t_len - 1;
#pragma unroll
    for (int j = 0; j < KW_PASSES; ++j) {
      const int t = kw_t + KW_TOK * j;
      const bool ok = t < t_len;
      const int tc = ok ? t : last;
      const uint4 a = *reinterpret_cast<const uint4*>(
          kb + (long)(chunk_start + tc) * Hg * GDN_K + kw_c);
      const uint4 b = *reinterpret_cast<const uint4*>(
          wb + (long)(chunk_start + tc) * H * GDN_K + kw_c);
      pk[j] = make_uint4(ok ? a.x : 0u, ok ? a.y : 0u, ok ? a.z : 0u,
                         ok ? a.w : 0u);
      pw[j] = make_uint4(ok ? b.x : 0u, ok ? b.y : 0u, ok ? b.z : 0u,
                         ok ? b.w : 0u);
    }
    const bool u_ok = u_t < t_len;
    const uint4 c = *reinterpret_cast<const uint4*>(
        ub + (long)(chunk_start + (u_ok ? u_t : last)) * H * GDN_V + u_c);
    pu = make_uint4(u_ok ? c.x : 0u, u_ok ? c.y : 0u, u_ok ? c.z : 0u,
                    u_ok ? c.w : 0u);
    const bool g_ok = lane < t_len;
    const float gv = p_g[(long)(chunk_start + (g_ok ? lane : last)) * H];
    pg = g_ok ? gv : 0.0f;
    pgl = p_g[(long)(chunk_start + last) * H];
  };
  fetch(0, T_seq < GDN_BT ? T_seq : GDN_BT);

  for (int i_t = 0; i_t < NT_seq; ++i_t) {
    const int chunk_start = i_t * GDN_BT;
    const int t_len =
        (chunk_start + GDN_BT <= T_seq) ? GDN_BT : (T_seq - chunk_start);
    const float g_last = pgl;

    // Registers -> LDS for this chunk. All 64 token rows are written (zeros
    // and a zero decay past t_len), so nothing of the previous chunk survives.
#pragma unroll
    for (int j = 0; j < KW_PASSES; ++j) {
      if constexpr (KPAIR) {
        // Even lane: tokens t (own) and t + 1 (partner), columns kw_c..+3; odd
        // lane: the same pair, columns kw_c + 4..+7. Each sends the half of its
        // piece the other one writes.
        const bool odd = (lane & 16) != 0;
        const uint4 x = pk[j];
        const unsigned s0 = __shfl_xor(odd ? x.x : x.z, 16);
        const unsigned s1 = __shfl_xor(odd ? x.y : x.w, 16);
        const unsigned e0 = odd ? s0 : x.x;  // even token, columns c, c + 1
        const unsigned e1 = odd ? s1 : x.y;  // even token, columns c + 2, c + 3
        const unsigned o0 = odd ? x.z : s0;  // odd token, columns c, c + 1
        const unsigned o1 = odd ? x.w : s1;  // odd token, columns c + 2, c + 3
        const uint4 pair = make_uint4(
            (e0 & 0xFFFFu) | (o0 << 16), (e0 >> 16) | (o0 & 0xFFFF0000u),
            (e1 & 0xFFFFu) | (o1 << 16), (e1 >> 16) | (o1 & 0xFFFF0000u));
        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned*>(ks) +
                                  ((kw_t >> 1) + (KW_TOK / 2) * j) * GDN_K +
                                  kw_c + (odd ? 4 : 0)) = pair;
      } else {
        *reinterpret_cast<uint4*>(ks + (kw_t + KW_TOK * j) * GDN_K + kw_c) =
            pk[j];
      }
      *reinterpret_cast<uint4*>(ws + (kw_t + KW_TOK * j) * GDN_K + kw_c) =
          pw[j];
    }
    *reinterpret_cast<uint4*>(us + u_t * BV + u_c) = pu;
    if (lane < GDN_BT) gd[lane] = (lane < t_len) ? expf(g_last - pg) : 0.0f;
    __syncthreads();

    if (i_t + 1 < NT_seq) {
      const int next_start = chunk_start + GDN_BT;
      const int rest = T_seq - next_start;
      fetch(next_start, rest < GDN_BT ? rest : GDN_BT);
    }

#pragma unroll
    for (int j = 0; j < 16; ++j) p_h[j] = __float2half_rn(hreg[j]);

    __half h_fp16[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h_fp16[j] = __float2half_rn(hreg[j]);

    const float decay = expf(g_last);
#pragma unroll
    for (int j = 0; j < 16; ++j) hreg[j] *= decay;

#pragma unroll 2
    for (int tp = 0; tp < GDN_BT / 2; ++tp) {
      const int t0 = 2 * tp;
      float vc[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int t = t0 + q;
        float acc = 0.0f;
        const __half* p_wt = ws + (long)t * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half2 a = gdn_load_f16x2(p_wt + 2 * j);
          const __half2 b = __halves2half2(h_fp16[2 * j], h_fp16[2 * j + 1]);
          acc = gdn_fdot2(a, b, acc);
        }
        acc = gdn_ksum_dpp(acc);
        const float v_raw = __half2float(us[(long)t * BV + lane_v]) - acc;
        // All 8 k-slice lanes of a row hold the same value; they write it to
        // the same LDS halfword.
        vn[(long)t * BV + lane_v] = __float2half_rn(v_raw);
        const float gdt = gd[t];  // loaded unconditionally: no branch per token
        vc[q] = (t < t_len) ? v_raw * gdt : 0.0f;
      }
      const __half2 vb =
          __halves2half2(__float2half_rn(vc[0]), __float2half_rn(vc[1]));
      if constexpr (KPAIR) {
        const uint4* p_kp = reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned*>(ks) + (long)tp * GDN_K + k0);
        unsigned kw32[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 q = p_kp[i];
          kw32[4 * i] = q.x;
          kw32[4 * i + 1] = q.y;
          kw32[4 * i + 2] = q.z;
          kw32[4 * i + 3] = q.w;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
          hreg[j] =
              gdn_fdot2(__builtin_bit_cast(__half2, kw32[j]), vb, hreg[j]);
      } else {
        __half2 a0[8], a1[8];
        const __half* p_k0 = ks + (long)t0 * GDN_K + k0;
        const __half* p_k1 = ks + (long)(t0 + 1) * GDN_K + k0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          a0[j] = gdn_load_f16x2(p_k0 + 2 * j);
          a1[j] = gdn_load_f16x2(p_k1 + 2 * j);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          hreg[2 * j] =
              gdn_fdot2(__halves2half2(__low2half(a0[j]), __low2half(a1[j])),
                        vb, hreg[2 * j]);
          hreg[2 * j + 1] =
              gdn_fdot2(__halves2half2(__high2half(a0[j]), __high2half(a1[j])),
                        vb, hreg[2 * j + 1]);
        }
      }
    }

    __syncthreads();
    // This chunk's (ungated) v_new rows, 16 bytes per thread.
    if (u_t < t_len) {
      *reinterpret_cast<uint4*>(vb_out + (long)(chunk_start + u_t) * H * GDN_V +
                                u_c) =
          *reinterpret_cast<const uint4*>(vn + u_t * BV + u_c);
    }
    p_h += stride_h;
  }

  if (store_final_state) {
    ST* p_ht = ht + (long)i_nh * GDN_V * GDN_K + (long)o_v * GDN_K + k0;
#pragma unroll
    for (int j = 0; j < 16; ++j) p_ht[j] = gdn_to_st<ST>(hreg[j]);
  }
}

}  // namespace

void gdn_prefill_delta_h_rdna2(torch::Tensor k, torch::Tensor u,
                               torch::Tensor w, torch::Tensor g,
                               torch::Tensor h, torch::Tensor v_new,
                               c10::optional<torch::Tensor> initial_state,
                               c10::optional<torch::Tensor> final_state,
                               c10::optional<torch::Tensor> cu_seqlens,
                               c10::optional<torch::Tensor> chunk_offsets,
                               int64_t chunk_size) {
  TORCH_CHECK(chunk_size == GDN_BT,
              "gdn_prefill_delta_h_rdna2 requires chunk_size == 64");
  TORCH_CHECK(k.scalar_type() == at::kHalf && u.scalar_type() == at::kHalf &&
                  w.scalar_type() == at::kHalf,
              "gdn_prefill_delta_h_rdna2 is fp16-only (gfx1030)");
  TORCH_CHECK(g.scalar_type() == at::kFloat,
              "g (g_cumsum) must be fp32 [B, T, H]");
  TORCH_CHECK(h.scalar_type() == at::kHalf,
              "h (per-chunk states) must be fp16 [B, NT, H, V, K] (5D)");
  TORCH_CHECK(v_new.scalar_type() == at::kHalf,
              "v_new must be fp16 [B, T, H, V]");
  TORCH_CHECK(k.dim() == 4 && k.stride(-1) == 1,
              "k must be [B, T, Hg, K] contiguous in last dim");
  TORCH_CHECK(u.dim() == 4 && u.stride(-1) == 1,
              "u must be [B, T, H, V] contiguous in last dim");
  TORCH_CHECK(w.dim() == 4 && w.stride(-1) == 1,
              "w must be [B, T, H, K] contiguous in last dim");
  TORCH_CHECK(g.dim() == 3 && g.stride(-1) == 1,
              "g must be [B, T, H] contiguous in last dim");

  const int B = k.size(0);
  const int T = k.size(1);
  const int Hg = k.size(2);
  const int K = k.size(3);
  const int H = u.size(2);
  const int V = u.size(3);
  TORCH_CHECK(K == GDN_K,
              "gdn_prefill_delta_h_rdna2 requires head_k_dim == 128");
  TORCH_CHECK(V == GDN_V,
              "gdn_prefill_delta_h_rdna2 requires head_v_dim == 128");
  TORCH_CHECK(H % Hg == 0, "H must be divisible by Hg (GQA); got H=", H,
              " Hg=", Hg);
  TORCH_CHECK(k.size(1) == u.size(1) && u.size(1) == w.size(1) &&
                  w.size(1) == g.size(1),
              "k/u/w/g sequence length mismatch");

  const bool varlen = cu_seqlens.has_value() && cu_seqlens->defined();
  int N;
  if (varlen) {
    TORCH_CHECK(B == 1, "varlen requires flattened batch B == 1");
    TORCH_CHECK(cu_seqlens->scalar_type() == at::kInt,
                "cu_seqlens must be int32 [N+1]");
    TORCH_CHECK(chunk_offsets.has_value() && chunk_offsets->defined(),
                "varlen requires chunk_offsets");
    TORCH_CHECK(chunk_offsets->scalar_type() == at::kInt,
                "chunk_offsets must be int32 [N+1]");
    N = cu_seqlens->size(0) - 1;
  } else {
    N = B;
  }
  TORCH_CHECK(N >= 1, "need at least one sequence");

  const bool has_h0 = initial_state.has_value() && initial_state->defined();
  if (has_h0) {
    TORCH_CHECK(
        (initial_state->scalar_type() == at::kFloat ||
         initial_state->scalar_type() == at::kHalf) &&
            initial_state->dim() == 4 && initial_state->stride(-1) == 1,
        "initial_state must be fp32/fp16 [N, H, V, K] contiguous in last "
        "dim");
    TORCH_CHECK(initial_state->size(0) == N && initial_state->size(1) == H &&
                    initial_state->size(2) == V && initial_state->size(3) == K,
                "initial_state shape mismatch");
  }

  const bool store_final = final_state.has_value() && final_state->defined();
  if (store_final) {
    TORCH_CHECK((final_state->scalar_type() == at::kFloat ||
                 final_state->scalar_type() == at::kHalf) &&
                    final_state->dim() == 4 && final_state->stride(-1) == 1,
                "final_state must be fp32/fp16 [N, H, V, K] contiguous in last "
                "dim");
    TORCH_CHECK(final_state->size(0) == N && final_state->size(1) == H &&
                    final_state->size(2) == V && final_state->size(3) == K,
                "final_state shape mismatch");
  }
  // The kernel is templated on the state dtype, so the two slots must agree.
  if (has_h0 && store_final) {
    TORCH_CHECK(initial_state->scalar_type() == final_state->scalar_type(),
                "initial_state and final_state must have the same dtype (both "
                "fp32 or both fp16)");
  }

  if (T == 0 || N == 0) return;

  // NT_total = total number of chunks across all sequences in this batch.
  // Non-varlen: NT_total = B * cdiv(T, BT)
  // Varlen: NT_total = chunk_offsets[N] (cumulative total at end of array)
  int NT_total;
  if (varlen) {
    // h.size(1) is the NT dim of the 5D h tensor. Reading chunk_offsets[N]
    // here would dereference a device pointer on the host and yield garbage.
    NT_total = h.size(1);
  } else {
    NT_total = B * ((T + GDN_BT - 1) / GDN_BT);
  }

  const at::cuda::OptionalCUDAGuard guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((V + GDN_BV - 1) / GDN_BV, N * H);
  constexpr int kSmem =
      (2 * GDN_BT * GDN_K + GDN_BT * GDN_BV) * (int)sizeof(__half) +
      GDN_BT * (int)sizeof(float);
  // The state dtype picks the kernel instantiation: fp32 (default) or fp16
  // (--mamba-ssm-cache-dtype float16).
  const auto state_ty =
      has_h0 ? initial_state->scalar_type()
             : (store_final ? final_state->scalar_type() : at::kFloat);
  // VLLM_RDNA_GDN_DH_PIPE selects the kernel: 0 = the existing one, 1-4 = the
  // latency-hiding kernel (see gdn_prefill_delta_h_pipe_kernel) with a 32-wide
  // V tile (1), a 64-wide one (2), and the same two with the pair-interleaved k
  // tile (3, 4). It moves k/u/w/v_new in 16-byte pieces, so it needs 16-byte
  // aligned base pointers; anything else runs the existing kernel. Read on
  // every call so a benchmark can A/B.
  int variant = 0;
  {
    const char* e = ::getenv("VLLM_RDNA_GDN_DH_PIPE");
    if (e != nullptr && e[0] >= '1' && e[0] <= '4' && e[1] == '\0' &&
        (((uintptr_t)k.data_ptr() | (uintptr_t)u.data_ptr() |
          (uintptr_t)w.data_ptr() | (uintptr_t)v_new.data_ptr()) &
         15) == 0) {
      variant = e[0] - '0';
    }
  }
#define GDN_PREFILL_ARGS(H0PTR, HTPTR)                                    \
  reinterpret_cast<const __half*>(k.data_ptr()),                          \
      reinterpret_cast<const __half*>(u.data_ptr()),                      \
      reinterpret_cast<const __half*>(w.data_ptr()), g.data_ptr<float>(), \
      H0PTR, reinterpret_cast<__half*>(h.data_ptr()), HTPTR,              \
      reinterpret_cast<__half*>(v_new.data_ptr()),                        \
      varlen ? cu_seqlens->data_ptr<int>() : nullptr,                     \
      varlen ? chunk_offsets->data_ptr<int>() : nullptr, T, H, Hg, N, B,  \
      NT_total, store_final ? 1 : 0
#define GDN_PREFILL_PIPE(ST, BVN, KP, H0PTR, HTPTR)   \
  gdn_prefill_delta_h_pipe_kernel<ST, BVN, KP>        \
      <<<dim3(V / BVN, N * H), BVN * 8, 0, stream>>>( \
          GDN_PREFILL_ARGS(H0PTR, HTPTR))
#define GDN_PREFILL_LAUNCH(ST, H0PTR, HTPTR)         \
  do {                                               \
    if (variant == 4) {                              \
      GDN_PREFILL_PIPE(ST, 64, true, H0PTR, HTPTR);  \
    } else if (variant == 3) {                       \
      GDN_PREFILL_PIPE(ST, 32, true, H0PTR, HTPTR);  \
    } else if (variant == 2) {                       \
      GDN_PREFILL_PIPE(ST, 64, false, H0PTR, HTPTR); \
    } else if (variant == 1) {                       \
      GDN_PREFILL_PIPE(ST, 32, false, H0PTR, HTPTR); \
    } else {                                         \
      gdn_prefill_delta_h_packed_kernel<ST>          \
          <<<grid, GDN_THREADS, kSmem, stream>>>(    \
              GDN_PREFILL_ARGS(H0PTR, HTPTR));       \
    }                                                \
  } while (0)
  if (state_ty == at::kHalf) {
    const __half* h0p =
        has_h0 ? reinterpret_cast<const __half*>(initial_state->data_ptr())
               : nullptr;
    __half* htp = store_final
                      ? reinterpret_cast<__half*>(final_state->data_ptr())
                      : nullptr;
    GDN_PREFILL_LAUNCH(__half, h0p, htp);
  } else {
    const float* h0p = has_h0 ? initial_state->data_ptr<float>() : nullptr;
    float* htp = store_final ? final_state->data_ptr<float>() : nullptr;
    GDN_PREFILL_LAUNCH(float, h0p, htp);
  }
#undef GDN_PREFILL_LAUNCH
#undef GDN_PREFILL_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
