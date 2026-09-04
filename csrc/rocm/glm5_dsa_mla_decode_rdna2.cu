// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash (Glm5Next) sparse MLA-NoPE decode attention kernel for AMD
// RDNA2 (gfx1030). UNC-36 §D op 2.
//
// Math source (READ-ONLY references):
//   - /tmp/glm53_refs/modeling_glm5_next.py lines 1064-1256
//     (Glm5NextTextAttention): q_lora 1536 -> q_b -> [64, 256];
//     kv_a 512 latent -> kv_b_proj -> k_nope/v of 256 each;
//     qk_rope_head_dim = 0 (NO RoPE anywhere in the LM); scaling =
//     qk_head_dim^-0.5 = 256^-0.5 = 0.0625; attention over the indexer's
//     topk-selected tokens with -1 == invalid (lines 1218-1256).
//   - vLLM torch reference this kernel is A/B'd against:
//     vllm/v1/attention/backends/glm5_dsa_attn.py
//     GLM5DSAAttnImpl.forward (scores = einsum(q, k_sel) * scale,
//     invalid -> finfo.min, fp32 softmax, out = probs . v_sel).
//
// Geometry (GLM-5.3-Flash config facts):
//   - 64 attention heads x 256 qk_nope_head_dim, v_head_dim = 256.
//   - S = number of caller-gathered selected slots per query
//     (index_topk 2048 + tail <= 3 -> S <= 2051; enforced).
//   - k_sel/v_sel are caller-gathered contiguous rows of the EXPANDED
//     (kv_b_proj'd) per-head keys/values; invalid slots are zeroed by the
//     caller and flagged via sel_valid. The kernel is therefore
//     cache-layout-agnostic (the DSA layer's cache decision lives
//     elsewhere).
//
// Computation per (batch b, head h), one workgroup each:
//   pass 1: score_s = sel_valid[b,s] ? (q_h . k_sel[b,s,h]) * scale : -inf
//           (256-wide dot = 4x fdot2 pairs per lane + wave butterfly sum),
//           scores cached in LDS (S <= 2051 floats = 8.2 KB), running max m.
//   pass 2: prob_s = exp(score_s - m) (0 for -inf), partial sumexp is
//           accumulated per lane (all lanes compute the identical scalar
//           sequence, so no reduction is needed); probs overwrite the LDS
//           scores. If no slot is valid (m == -inf) every prob is 0 and
//           the output row is zeroed (the torch reference likewise zeroes
//           all-invalid rows after softmax).
//   pass 3: out[b, h, :] = sum_s (prob_s / sumexp) * v_sel[b,s,h,:]
//           accumulated in fp32 registers (8 dims per lane), stored fp16.
//
// Workgroup geometry:
//   - grid = (64 heads, B batches); block = 32 threads = one wave32.
//     Head-tile size is ONE head per workgroup: the 256 fp16 query (512 B)
//     stays register-resident as 4 half2 per lane (8 dims/lane, 4 VGPRs)
//     and the 256-wide fp32 output accumulator is 8 VGPRs/lane, so total
//     pressure is ~30 VGPRs/lane -- comfortably under the <=64 VGPR decode
//     target without any occupancy pins. Tiling multiple heads per
//     workgroup would raise VGPR pressure (multi-head q + per-head
//     accumulators) without reducing the serial S loop, so 1 head/WG was
//     chosen; 64*B wave32 workgroups also give more parallelism across the
//     80 CUs than a coarser tiling for decode-sized B.
//   - The S loop is serial inside the workgroup (two-pass softmax; S <=
//     2051 by contract). Cross-lane traffic is one 5-step butterfly per
//     selected slot (QK dot) plus a single final wave-wide agreement on
//     m/sumexp (redundant per-lane computation, no reduction).
//   - LDS: static float s_scores[2051] (8.2 KB per workgroup). Pass 1
//     writes scores, pass 2 reads+overwrites with probs, pass 3 reads.
//     LDS ops of one wave are processed in order; __syncthreads() between
//     passes keeps the phase boundaries explicit.
//   - Loads: q/k/v rows are 512 B and 16-byte aligned (checked in the
//     launcher), issued as uint4 (16 B) vector loads; sel_valid bytes are
//     uniform scalar loads.
//
// RDNA2 rules honored: wave32; fp16 inputs with fp32 accumulation for the
// dots/softmax/PV; fdot2 for the QK dots; no WMMA/MFMA/AGPR/FP8; no
// __launch_bounds__ / waves_per_eu pins; no D2H or host syncs in the
// launcher; out is a Tensor! out-arg with exclusive per-workgroup
// ownership (no atomics; caller pre-zeroing accepted but not relied on --
// every element of out is written).
//
// Numerics fidelity: fp16 q/k products are exact in fp32 (11-bit x 11-bit
// mantissas fit fp32), so the dots match the reference's fp32 matmul of
// promoted values; accumulation ORDER differs from torch (ulp-level
// differences expected). Masking uses -inf where the torch reference uses
// finfo(fp32).min; after max-subtraction both yield exp() == 0, and
// all-invalid rows produce a zero output in both. The reference fp16
// rounding of q/k/v happens in the caller (.to(torch.float16)), per the
// DESIGN.md fp16-buffer contract.
//
// k_sel/v_sel last-dim layouts accepted (memory-identical views):
//   - [B, S, 256]      shared keys (DESIGN.md literal); head_offset = 0.
//   - [B, S, 64*256]   per-head interleaved, i.e. [B, S, H, 256]
//                      contiguous with the heads flattened into the last
//                      dim -- the layout glm5_dsa_attn.py's HIP hook
//                      produces from its [M, W, H, D] gather; head h reads
//                      the 256-slice at h*256.
//
// UNVERIFIED: written blind (no runtime available). Compile, numerics A/B
// vs the torch scan in GLM5DSAAttnImpl.forward, and occupancy/latency on
// gfx1030 must be checked by a human later. Decode contract: one query
// token per sequence (the GLM5DSAAttn backend advertises
// UNIFORM_SINGLE_TOKEN_DECODE only; q > 1 must stay on the torch path).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

#include <limits>

// Global-scope constants (internal linkage; prefixed to avoid collisions
// with the unprefixed constants in fa_rdna2.cu).
constexpr int GLM5_DSA_MLA_HEADS = 64;     // num_attention_heads
constexpr int GLM5_DSA_MLA_DIM = 256;      // qk_nope_head_dim == v_head_dim
constexpr int GLM5_DSA_MAX_SEL = 2051;     // index_topk 2048 + tail <= 3
constexpr int GLM5_DSA_MLA_WAVE = 32;      // wave32; one wave per workgroup
constexpr float GLM5_DSA_NEG_INF =
    -std::numeric_limits<float>::infinity();

namespace glm5_dsa_mla {

// V_DOT2_F32_F16: 2 fp16 multiplies accumulated into fp32. Same wrapper
// as fa_rdna2.cu, kept namespace-local to avoid a global ODR clash.
__device__ __forceinline__ float fdot2(half2 a, half2 b, float acc) {
  return __builtin_amdgcn_fdot2(a, b, acc, false);
}

// Wave32 butterfly sum (all lanes receive the total).
__device__ __forceinline__ float wave32_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

// One (batch, head) per workgroup; 32 threads = one wave32. Lane l owns
// dims [8l, 8l+8) of the 256-wide vectors.
__global__ void glm5_dsa_mla_decode_kernel(
    const __half* __restrict__ q_nope,            // [B, 64, 256] fp16
    const __half* __restrict__ k_sel,             // [B, S, slot_dim] fp16
    const __half* __restrict__ v_sel,             // [B, S, slot_dim] fp16
    const unsigned char* __restrict__ sel_valid,  // [B, S] uint8
    __half* __restrict__ out,                     // [B, 64*256] fp16
    float scale,
    int S,
    int slot_dim,    // 256 (shared keys) or 64*256 (per-head interleaved)
    int head_stride  // 0 (shared) or 256 (per-head slice within a slot)
) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int d0 = lane * 8;

  __shared__ float s_scores[GLM5_DSA_MAX_SEL];

  // Query fragment: 8 fp16 per lane, register-resident for the whole S
  // loop (4 half2 = 4 VGPRs). q rows are 512 B and 16-byte aligned.
  const __half* q_row =
      q_nope + ((long long)b * GLM5_DSA_MLA_HEADS + h) * GLM5_DSA_MLA_DIM +
      d0;
  __half2 q2[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __builtin_memcpy(&q2[j], q_row + 2 * j, sizeof(__half2));
  }

  const __half* k_base =
      k_sel + (long long)b * S * slot_dim + h * head_stride;
  const unsigned char* sv_row = sel_valid + (long long)b * S;

  // Pass 1: QK dots -> masked scores -> LDS, running max.
  float m = GLM5_DSA_NEG_INF;
  for (int s = 0; s < S; ++s) {
    const __half* k_row = k_base + (long long)s * slot_dim + d0;
    uint4 kraw;
    __builtin_memcpy(&kraw, k_row, sizeof(uint4));
    const __half2* k2 = reinterpret_cast<const __half2*>(&kraw);
    float dot = fdot2(q2[0], k2[0], 0.0f);
    dot = fdot2(q2[1], k2[1], dot);
    dot = fdot2(q2[2], k2[2], dot);
    dot = fdot2(q2[3], k2[3], dot);
    dot = wave32_sum(dot);
    const float score =
        (sv_row[s] != 0) ? (dot * scale) : GLM5_DSA_NEG_INF;
    if ((s & (GLM5_DSA_MLA_WAVE - 1)) == lane) {
      s_scores[s] = score;
    }
    m = fmaxf(m, score);
  }
  __syncthreads();

  // Pass 2: exp(score - m) and sumexp. All lanes run the identical
  // scalar sequence, so every lane ends with the same sumexp and no
  // cross-lane reduction is needed. Probs overwrite the LDS scores for
  // pass 3.
  const bool any_valid = (m > GLM5_DSA_NEG_INF);
  float sumexp = 0.0f;
  for (int s = 0; s < S; ++s) {
    float prob = 0.0f;
    if (any_valid) {
      const float x = s_scores[s];
      if (x > GLM5_DSA_NEG_INF) {
        prob = expf(x - m);
      }
    }
    sumexp += prob;
    if ((s & (GLM5_DSA_MLA_WAVE - 1)) == lane) {
      s_scores[s] = prob;
    }
  }
  __syncthreads();

  const float inv_sum = (sumexp > 0.0f) ? (1.0f / sumexp) : 0.0f;

  // Pass 3: PV accumulate in fp32 registers (8 dims per lane).
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

  const __half* v_base =
      v_sel + (long long)b * S * slot_dim + h * head_stride;
  for (int s = 0; s < S; ++s) {
    const float prob = s_scores[s] * inv_sum;
    if (prob == 0.0f) {
      continue;
    }
    const __half* v_row = v_base + (long long)s * slot_dim + d0;
    uint4 vraw;
    __builtin_memcpy(&vraw, v_row, sizeof(uint4));
    const __half* vh = reinterpret_cast<const __half*>(&vraw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[j] += prob * __half2float(vh[j]);
    }
  }

  // Epilogue: fp32 -> fp16 store, 16 B vector per lane.
  __half* o_row =
      out + ((long long)b * GLM5_DSA_MLA_HEADS + h) * GLM5_DSA_MLA_DIM + d0;
  uint4 oraw;
  __half* oh = reinterpret_cast<__half*>(&oraw);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    oh[j] = __float2half(acc[j]);
  }
  __builtin_memcpy(o_row, &oraw, sizeof(uint4));
}

}  // namespace glm5_dsa_mla

// Host launcher. Signature == ops.h declaration == torch_bindings schema
// == _custom_ops wrapper (DESIGN.md UNC-35/36 §D, op 2).
void glm5_dsa_mla_decode_rdna2(torch::Tensor q_nope, torch::Tensor k_sel,
                               torch::Tensor v_sel, torch::Tensor sel_valid,
                               torch::Tensor out, double scale) {
  TORCH_CHECK(q_nope.is_cuda() && k_sel.is_cuda() && v_sel.is_cuda() &&
                  sel_valid.is_cuda() && out.is_cuda(),
              "glm5_dsa_mla_decode_rdna2: all tensors must be on a HIP "
              "device");

  TORCH_CHECK(q_nope.dim() == 3 && q_nope.is_contiguous() &&
                  q_nope.scalar_type() == at::kHalf,
              "q_nope must be contiguous fp16 [B, 64, 256]");
  TORCH_CHECK(q_nope.size(1) == GLM5_DSA_MLA_HEADS &&
                  q_nope.size(2) == GLM5_DSA_MLA_DIM,
              "q_nope must be [B, 64, 256]; got [", q_nope.size(0), ", ",
              q_nope.size(1), ", ", q_nope.size(2), "]");
  TORCH_CHECK(k_sel.dim() == 3 && k_sel.is_contiguous() &&
                  k_sel.scalar_type() == at::kHalf,
              "k_sel must be contiguous fp16 [B, S, 256] or "
              "[B, S, 64*256]");
  TORCH_CHECK(v_sel.dim() == 3 && v_sel.is_contiguous() &&
                  v_sel.scalar_type() == at::kHalf,
              "v_sel must be contiguous fp16, same shape as k_sel");
  TORCH_CHECK(sel_valid.dim() == 2 && sel_valid.is_contiguous() &&
                  sel_valid.scalar_type() == at::kByte,
              "sel_valid must be contiguous uint8 [B, S]");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() &&
                  out.scalar_type() == at::kHalf,
              "out must be contiguous fp16 [B, 64*256]");

  const int B = q_nope.size(0);
  const int S = k_sel.size(1);
  const int slot_dim = k_sel.size(2);

  TORCH_CHECK(v_sel.size(0) == B && v_sel.size(1) == S &&
                  v_sel.size(2) == slot_dim,
              "v_sel shape must match k_sel");
  TORCH_CHECK(sel_valid.size(0) == B && sel_valid.size(1) == S,
              "sel_valid must be [B, S]");
  TORCH_CHECK(out.size(0) == B &&
                  out.size(1) ==
                      (int64_t)GLM5_DSA_MLA_HEADS * GLM5_DSA_MLA_DIM,
              "out must be [B, 64*256]");
  TORCH_CHECK(slot_dim == GLM5_DSA_MLA_DIM ||
                  slot_dim == GLM5_DSA_MLA_HEADS * GLM5_DSA_MLA_DIM,
              "k_sel/v_sel last dim must be 256 (shared keys) or 64*256 "
              "(per-head interleaved); got ", slot_dim);
  TORCH_CHECK(S <= GLM5_DSA_MAX_SEL,
              "S=", S, " exceeds the supported selection width ",
              GLM5_DSA_MAX_SEL, " (index_topk 2048 + tail 3)");
  TORCH_CHECK(B <= 65535, "B exceeds grid.y limit");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(q_nope.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(k_sel.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(v_sel.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              "q_nope/k_sel/v_sel/out bases must be 16-byte aligned "
              "(uint4 loads)");

  if (B == 0 || S == 0) {
    return;  // nothing to attend; caller-pre-zeroed out stays as-is
  }

  const int head_stride =
      (slot_dim == GLM5_DSA_MLA_HEADS * GLM5_DSA_MLA_DIM)
          ? GLM5_DSA_MLA_DIM
          : 0;

  hipStream_t stream = at::hip::getCurrentHIPStream();

  // grid: one wave32 workgroup per (head, batch).
  dim3 grid(GLM5_DSA_MLA_HEADS, B);
  glm5_dsa_mla::glm5_dsa_mla_decode_kernel<<<grid, GLM5_DSA_MLA_WAVE, 0,
                                             stream>>>(
      reinterpret_cast<const __half*>(q_nope.data_ptr()),
      reinterpret_cast<const __half*>(k_sel.data_ptr()),
      reinterpret_cast<const __half*>(v_sel.data_ptr()),
      sel_valid.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(out.data_ptr()),
      (float)scale,
      S,
      slot_dim,
      head_stride);
}
