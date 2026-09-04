// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash (Glm5Next) DSA k-pool indexer kernels for AMD RDNA2
// (gfx1030). UNC-36 §D op 1.
//
// Math source (READ-ONLY references):
//   - /tmp/glm53_refs/modeling_glm5_next.py lines 736-1024
//     (Glm5NextTextIndexer): get_pooled_states (lines 899-972) for the
//     pool construction + compression, forward (lines 823-844) for the
//     scoring / head-weighted sum.
//   - vLLM torch reference this kernel is A/B'd against:
//     vllm/model_executor/layers/attention/glm5_dsa_attention.py
//     glm5_dsa_topk_from_packed (lines 104-235).
//
// Geometry (GLM-5.3-Flash config facts):
//   - indexer: n_heads = 32, head_dim = 128 (index_n_heads/index_head_dim)
//   - index_kpool = 4 (pools of 4 consecutive keys, starting at first_key)
//   - index_topk = 2048 raw tokens => select_k = 512 pools; tail <= 3
//     (output width 2051). Topk selection, causal masked_fill, tail
//     expansion and -1 padding are ALL caller-side (torch.topk); this
//     kernel only produces dense per-pool scores + raw indices + validity.
//   - packed indexer rows: [k(128) | gate_scores(128) | valid(1)] = 257
//     fp16 per token, written by the DSA layer each forward.
//
// What this file computes, per (query token q, pool p):
//   1. first_key[q] = first index in packed[q] with valid-channel != 0
//      (reference: valid_keys.any/argmax; = T when no key is valid).
//      Pools start at first_key, not raw slot 0 (modeling lines 906-947).
//   2. member validity (reference grouped_valid_keys):
//        mv[i] = (a_i < kv_lens[q]) && (a_i < T) && (valid_channel != 0),
//      where a_i = first_key + 4p + i. kv_lens[q] is the visible length
//      of query q (decode: the sequence length; padded slots in packed
//      carry valid=0 by caller contract).
//   3. pool compression (modeling lines 961-967):
//        logits[i][d] = gate[i][d] + ape[i][d]      (fp32; -inf if !mv[i])
//        prob[i][d]   = softmax over the 4 members, per dim d
//                       (all-invalid dim -> 0 per torch.nan_to_num)
//        prob cast to fp16 (reference .to(grouped_keys.dtype))
//        pool_key[d]  = (fp16) sum_i (fp16)prob[i][d] * k[i][d]
//                       (fp32 accumulation, result rounded to fp16 like
//                       the reference fp16 .sum())
//   4. scoring (modeling lines 825-830):
//        score_h = relu( (q_h . pool_key) * 128^-0.5 )        fp32
//        scores_out[q,p] = sum_h weights[q,h] * score_h       fp32
//      (weights arrive pre-scaled by 32^-0.5 from the caller, per
//      DESIGN.md §D: weights = weights_proj(x) * 32^-0.5.)
//   5. pool_indices_out: raw a_i, -1 where !mv[i] (reference
//      pool_indices.masked_fill(~grouped_valid_keys, -1)).
//      pool_valid_out: 1 iff all 4 members valid AND the pool's final
//      token index < kv_lens[q]. This folds the reference's pool-level
//      causal visibility check (pool_end <= q_position, modeling lines
//      833-839) into the validity byte: at decode time kv_lens[q] is the
//      visible length, so pool_end < kv_lens[q] <=> pool_end visible.
//      The scores of invalid pools are still written (the reference only
//      masks them at selection time); the caller masks with
//      finfo.min before torch.topk.
//
// Workgroup geometry:
//   - first-key scan kernel: 1 workgroup of 32 threads (1 wave32) per
//     query; lanes stride the valid channel by 32 rows, __ballot +
//     __ffsll find the first set row. One-shot per layer per step.
//   - pool kernel: 1 workgroup of 32 threads (1 wave32) per (q, pool).
//     Grid = Q * max_pools (flat). Lane l owns dims [4l, 4l+4) of the
//     128-wide vectors. Per pool: 4 packed-row loads (scalar fp16 loads;
//     the 514-byte row stride of packed [.,.,257] is only 2-byte aligned,
//     so vectorized global loads are not legal for arbitrary rows), a
//     4-way softmax over 128 dims, then a serial loop over the 32 heads
//     with an 8-byte vector q load + 2x fdot2 (V_DOT2_F32_F16) + wave
//     butterfly reduce per head. The q row (8 KB per query) is re-read by
//     every pool workgroup of that query and stays L2-resident
//     (Q * 8 KB <= 2 MB for decode-sized Q).
//
// RDNA2 rules honored: wave32; fp16 inputs with fp32 accumulation for all
// reductions / softmax / scoring; fdot2 for the dots; no WMMA/MFMA/AGPR/
// FP8; no __launch_bounds__ / waves_per_eu pins; no D2H or host syncs in
// the launcher; outputs are Tensor! out-args with exclusive per-workgroup
// ownership (no atomics; caller pre-zeroing accepted but not relied on —
// every element is written).
//
// Numerics fidelity: the formulae, dtype promotion points (fp16 prob
// cast, fp16 pool_key storage) and masking exactly follow the reference.
// fp32 accumulation ORDER differs from torch's matmul reduction tree, so
// ulp-level differences vs the torch path are expected; the products
// themselves are exact (fp16*fp16 -> fp32 is exact in both fdot2 and
// torch's fp32 matmul of promoted values).
//
// UNVERIFIED: written blind (no runtime available). Compile, numerics A/B
// vs glm5_dsa_topk_from_packed, and occupancy behavior on gfx1030 must be
// checked by a human later. Q == B (one query token per sequence, decode)
// is enforced by TORCH_CHECK: the GLM5DSAAttn backend advertises
// UNIFORM_SINGLE_TOKEN_DECODE only, and multi-query (MTP verify / chunked
// prefill) must stay on the torch path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

#include <limits>

// Global-scope constants (internal linkage; prefixed to avoid collisions
// with the unprefixed constants in fa_rdna2.cu).
constexpr int GLM5_IDX_N_HEADS = 32;       // index_n_heads
constexpr int GLM5_IDX_HEAD_DIM = 128;     // index_head_dim
constexpr int GLM5_IDX_KPOOL = 4;          // index_kpool (GLM-5.3 fixed)
constexpr int GLM5_IDX_PACKED_WIDTH = 257;  // k[128] | gate[128] | valid[1]
constexpr float GLM5_IDX_QK_SCALE =
    0.08838834764831845f;                  // 128^-0.5 (softmax_scale)
constexpr float GLM5_IDX_NEG_INF =
    -std::numeric_limits<float>::infinity();
constexpr int GLM5_IDX_WAVE = 32;  // wave32; one wave per workgroup

namespace glm5_dsa_idx {

// V_DOT2_F32_F16: 2 fp16 multiplies accumulated into fp32. Same wrapper
// as fa_rdna2.cu, kept namespace-local to avoid a global ODR clash.
__device__ __forceinline__ float fdot2(half2 a, half2 b, float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, false);
}

// Wave32 butterfly sum (all lanes receive the total). Offsets 16..1 cover
// the full 32-lane wave on gfx1030.
__device__ __forceinline__ float wave32_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

// First valid key per query (reference: valid_keys.any(-1) /
// argmax(-1), with T when no key is valid). packed rows are 514 bytes;
// the valid channel is the last fp16 of every row (element 256), so the
// loads are 2-byte scalar and strided by 257 halves. One wave per query.
__global__ void glm5_dsa_first_key_kernel(
    const __half* __restrict__ packed,  // [Q, T, 257]
    int T,
    int* __restrict__ first_key) {      // [Q]
  const int q = blockIdx.x;
  const int lane = threadIdx.x;

  const __half* valid_base =
      packed + (long long)q * T * GLM5_IDX_PACKED_WIDTH +
      (GLM5_IDX_PACKED_WIDTH - 1);

  int fk = T;  // no valid key -> T (reference: torch.full((B,), seq_len))
  for (int t0 = 0; t0 < T; t0 += 32) {
    const int t = t0 + lane;
    const bool v =
        (t < T) &&
        (__half2float(valid_base[(long long)t * GLM5_IDX_PACKED_WIDTH]) !=
         0.0f);
    const unsigned long long mask = __ballot(v);
    if (mask != 0ULL) {
      // Rows increase with lane id, so the lowest set bit is the first
      // valid row of this chunk (and of the whole scan: chunks are
      // visited in order).
      fk = t0 + __ffsll(mask) - 1;
      break;
    }
  }
  if (lane == 0) {
    first_key[q] = fk;
  }
}

// One (query, pool) per workgroup; 32 threads = one wave32. Lane l owns
// dims [4l, 4l+4) of every 128-wide vector.
__global__ void glm5_dsa_indexer_pool_kernel(
    const __half* __restrict__ q_idx,            // [Q, 32, 128] fp16
    const __half* __restrict__ packed,           // [Q, T, 257] fp16
    const float* __restrict__ weights,           // [Q, 32] fp32
    const int* __restrict__ kv_lens,             // [Q] int32
    const float* __restrict__ ape,               // [4, 128] fp32
    const int* __restrict__ first_key,           // [Q] int32 (scratch)
    int* __restrict__ pool_indices_out,          // [Q, max_pools, 4] int32
    unsigned char* __restrict__ pool_valid_out,  // [Q, max_pools] uint8
    float* __restrict__ scores_out,              // [Q, max_pools] fp32
    int T,
    int max_pools) {
  const unsigned int wp = blockIdx.x;
  const int q = (int)(wp / (unsigned int)max_pools);
  const int p = (int)(wp % (unsigned int)max_pools);
  const int lane = (int)threadIdx.x;

  const int fk = first_key[q];
  const int kv_len = kv_lens[q];
  const int base_tok = fk + GLM5_IDX_KPOOL * p;  // first raw token of pool

  // Member validity (reference grouped_valid_keys): within the query's
  // visible range AND the packed valid channel set. The ternaries below
  // short-circuit, so rows past T are never dereferenced.
  bool mv[GLM5_IDX_KPOOL];
#pragma unroll
  for (int i = 0; i < GLM5_IDX_KPOOL; ++i) {
    const long long a = (long long)base_tok + i;
    mv[i] = (fk < T) && (a < T) && (a < kv_len) &&
            (__half2float(packed[((long long)q * T + a) *
                                     GLM5_IDX_PACKED_WIDTH +
                                 (GLM5_IDX_PACKED_WIDTH - 1)]) != 0.0f);
  }

  // Load the pool's 4 packed rows: key + gate fragments for this lane's
  // 4 dims; build softmax logits (gate + ape, -inf for invalid members).
  const int d0 = lane * 4;
  float logits[GLM5_IDX_KPOOL][4];
  float kfrag[GLM5_IDX_KPOOL][4];
#pragma unroll
  for (int i = 0; i < GLM5_IDX_KPOOL; ++i) {
    const long long a = (long long)base_tok + i;
    const __half* row =
        packed + ((long long)q * T + a) * GLM5_IDX_PACKED_WIDTH;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = d0 + j;
      kfrag[i][j] = mv[i] ? __half2float(row[d]) : 0.0f;
      const float gate =
          mv[i] ? __half2float(row[GLM5_IDX_HEAD_DIM + d]) : 0.0f;
      logits[i][j] =
          mv[i] ? (gate + ape[i * GLM5_IDX_HEAD_DIM + d]) : GLM5_IDX_NEG_INF;
    }
  }

  // Softmax over the 4 members, independently per dim, then the weighted
  // key sum. Matches modeling lines 962-967:
  //   probabilities = nan_to_num(logits.softmax(dim=2)).to(fp16)
  //   pool_keys = (probabilities * grouped_keys).sum(dim=2)   (fp16 out)
  float pk_frag[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    pk_frag[j] = 0.0f;
    const float m = fmaxf(fmaxf(logits[0][j], logits[1][j]),
                          fmaxf(logits[2][j], logits[3][j]));
    float prob[GLM5_IDX_KPOOL];
    if (m == GLM5_IDX_NEG_INF) {
      // All 4 members invalid: torch softmax -> NaN -> nan_to_num -> 0.
#pragma unroll
      for (int i = 0; i < GLM5_IDX_KPOOL; ++i) prob[i] = 0.0f;
    } else {
      float denom = 0.0f;
#pragma unroll
      for (int i = 0; i < GLM5_IDX_KPOOL; ++i) {
        // Invalid members carry -inf logits: expf(-inf - m) == 0.
        prob[i] = expf(logits[i][j] - m);
        denom += prob[i];
      }
      const float inv = 1.0f / denom;
#pragma unroll
      for (int i = 0; i < GLM5_IDX_KPOOL; ++i) prob[i] *= inv;
    }
#pragma unroll
    for (int i = 0; i < GLM5_IDX_KPOOL; ++i) {
      // Reference casts probabilities to fp16 BEFORE weighting the keys.
      pk_frag[j] +=
          __half2float(__float2half(prob[i])) * kfrag[i][j];
    }
  }

  // The reference stores pool_keys in fp16 (fp32-accumulated .sum()
  // rounded back to fp16); round to fp16 and keep as half2 pairs for the
  // fdot2 head dots.
  __half2 pk2[2];
  pk2[0] = __halves2half2(__float2half(pk_frag[0]), __float2half(pk_frag[1]));
  pk2[1] = __halves2half2(__float2half(pk_frag[2]), __float2half(pk_frag[3]));

  // Head loop: score_h = relu((q_h . pool_key) * 128^-0.5), then the
  // weighted head sum with weights[q] (already scaled by 32^-0.5 by the
  // caller). q loads are 8-byte vectors (q rows are 256-byte aligned).
  const __half* q_base =
      q_idx + ((long long)q * GLM5_IDX_N_HEADS) * GLM5_IDX_HEAD_DIM;
  const float* w_row = weights + (long long)q * GLM5_IDX_N_HEADS;
  float total = 0.0f;
  for (int h = 0; h < GLM5_IDX_N_HEADS; ++h) {
    const __half* qh = q_base + h * GLM5_IDX_HEAD_DIM + d0;
    __half2 qa, qb;
    __builtin_memcpy(&qa, qh, sizeof(qa));
    __builtin_memcpy(&qb, qh + 2, sizeof(qb));
    float dot = fdot2(qa, pk2[0], 0.0f);
    dot = fdot2(qb, pk2[1], dot);
    dot = wave32_sum(dot);
    total += w_row[h] * fmaxf(dot * GLM5_IDX_QK_SCALE, 0.0f);
  }

  // Outputs (exclusive ownership; every element written).
  const bool pool_full = mv[0] && mv[1] && mv[2] && mv[3];
  const bool pool_visible =
      pool_full &&
      ((long long)base_tok + GLM5_IDX_KPOOL - 1 < kv_len);
  const long long out_slot = (long long)q * max_pools + p;
  if (lane == 0) {
    scores_out[out_slot] = total;
    pool_valid_out[out_slot] = pool_visible ? (unsigned char)1
                                            : (unsigned char)0;
  }
  if (lane < GLM5_IDX_KPOOL) {
    pool_indices_out[out_slot * GLM5_IDX_KPOOL + lane] =
        mv[lane] ? (int)((long long)base_tok + lane) : -1;
  }
}

}  // namespace glm5_dsa_idx

// Host launcher. Signature == ops.h declaration == torch_bindings schema
// == _custom_ops wrapper (DESIGN.md UNC-35/36 §D, op 1).
void glm5_dsa_indexer_rdna2(torch::Tensor q_idx, torch::Tensor packed,
                            torch::Tensor weights, torch::Tensor kv_lens,
                            torch::Tensor ape, torch::Tensor pool_indices_out,
                            torch::Tensor pool_valid_out,
                            torch::Tensor scores_out, int64_t kpool) {
  TORCH_CHECK(kpool == GLM5_IDX_KPOOL,
              "glm5_dsa_indexer_rdna2 only supports index_kpool=4 "
              "(GLM-5.3-Flash); got ", kpool);
  TORCH_CHECK(q_idx.is_cuda() && packed.is_cuda() && weights.is_cuda() &&
                  kv_lens.is_cuda() && ape.is_cuda() &&
                  pool_indices_out.is_cuda() && pool_valid_out.is_cuda() &&
                  scores_out.is_cuda(),
              "glm5_dsa_indexer_rdna2: all tensors must be on a HIP device");

  TORCH_CHECK(q_idx.dim() == 3 && q_idx.is_contiguous() &&
                  q_idx.scalar_type() == at::kHalf,
              "q_idx must be contiguous fp16 [Q, 32, 128]");
  TORCH_CHECK(q_idx.size(1) == GLM5_IDX_N_HEADS &&
                  q_idx.size(2) == GLM5_IDX_HEAD_DIM,
              "q_idx must be [Q, 32, 128]; got [", q_idx.size(0), ", ",
              q_idx.size(1), ", ", q_idx.size(2), "]");
  TORCH_CHECK(packed.dim() == 3 && packed.is_contiguous() &&
                  packed.scalar_type() == at::kHalf,
              "packed must be contiguous fp16 [B, T, 257]");
  TORCH_CHECK(packed.size(2) == GLM5_IDX_PACKED_WIDTH,
              "packed last dim must be 257 (k|gate|valid); got ",
              packed.size(2));
  TORCH_CHECK(weights.dim() == 2 && weights.is_contiguous() &&
                  weights.scalar_type() == at::kFloat,
              "weights must be contiguous fp32 [Q, 32]");
  TORCH_CHECK(kv_lens.dim() == 1 && kv_lens.is_contiguous() &&
                  kv_lens.scalar_type() == at::kInt,
              "kv_lens must be contiguous int32 [Q]");
  TORCH_CHECK(ape.dim() == 2 && ape.is_contiguous() &&
                  ape.scalar_type() == at::kFloat,
              "ape must be contiguous fp32 [4, 128]");
  TORCH_CHECK(ape.size(0) == GLM5_IDX_KPOOL &&
                  ape.size(1) == GLM5_IDX_HEAD_DIM,
              "ape must be [4, 128]");
  TORCH_CHECK(pool_indices_out.dim() == 3 &&
                  pool_indices_out.is_contiguous() &&
                  pool_indices_out.scalar_type() == at::kInt,
              "pool_indices_out must be contiguous int32 [Q, max_pools, 4]");
  TORCH_CHECK(pool_indices_out.size(2) == GLM5_IDX_KPOOL,
              "pool_indices_out last dim must be 4");
  TORCH_CHECK(pool_valid_out.dim() == 2 && pool_valid_out.is_contiguous() &&
                  pool_valid_out.scalar_type() == at::kByte,
              "pool_valid_out must be contiguous uint8 [Q, max_pools]");
  TORCH_CHECK(scores_out.dim() == 2 && scores_out.is_contiguous() &&
                  scores_out.scalar_type() == at::kFloat,
              "scores_out must be contiguous fp32 [Q, max_pools]");

  const int Q = q_idx.size(0);
  const int T = packed.size(1);
  const int max_pools = (T + GLM5_IDX_KPOOL - 1) / GLM5_IDX_KPOOL;

  TORCH_CHECK(packed.size(0) == Q,
              "packed batch ", packed.size(0),
              " must equal Q ", Q,
              " (decode contract: one query token per sequence)");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == GLM5_IDX_N_HEADS,
              "weights must be [Q, 32]");
  TORCH_CHECK(kv_lens.size(0) == Q, "kv_lens must be [Q]");
  TORCH_CHECK(pool_indices_out.size(0) == Q &&
                  pool_indices_out.size(1) == max_pools,
              "pool_indices_out must be [Q, max_pools, 4] with max_pools=",
              max_pools);
  TORCH_CHECK(pool_valid_out.size(0) == Q &&
                  pool_valid_out.size(1) == max_pools,
              "pool_valid_out must be [Q, max_pools]");
  TORCH_CHECK(scores_out.size(0) == Q && scores_out.size(1) == max_pools,
              "scores_out must be [Q, max_pools]");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(q_idx.data_ptr()) & 7) == 0,
              "q_idx base must be 8-byte aligned (half2 loads)");
  TORCH_CHECK((long long)Q * max_pools <= 0x7FFFFFFFLL,
              "glm5_dsa_indexer_rdna2: Q * max_pools exceeds grid limit");

  if (Q == 0 || max_pools == 0) {
    return;  // nothing to score; caller-pre-zeroed outputs stay as-is
  }

  // Device scratch for first_key (no D2H; allocation is capture-safe like
  // fa_rdna2_decode_paged's workspace allocations).
  auto fk_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(q_idx.device());
  auto first_key = torch::empty({Q}, fk_opts);

  hipStream_t stream = at::hip::getCurrentHIPStream();

  dim3 grid1(Q);
  glm5_dsa_idx::glm5_dsa_first_key_kernel<<<grid1, GLM5_IDX_WAVE, 0,
                                            stream>>>(
      reinterpret_cast<const __half*>(packed.data_ptr()), T,
      first_key.data_ptr<int>());

  dim3 grid2((unsigned int)((long long)Q * max_pools));
  glm5_dsa_idx::glm5_dsa_indexer_pool_kernel<<<grid2, GLM5_IDX_WAVE, 0,
                                               stream>>>(
      reinterpret_cast<const __half*>(q_idx.data_ptr()),
      reinterpret_cast<const __half*>(packed.data_ptr()),
      weights.data_ptr<float>(),
      kv_lens.data_ptr<int>(),
      ape.data_ptr<float>(),
      first_key.data_ptr<int>(),
      pool_indices_out.data_ptr<int>(),
      pool_valid_out.data_ptr<uint8_t>(),
      scores_out.data_ptr<float>(),
      T,
      max_pools);
}
