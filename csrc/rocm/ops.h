#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna2_prefill(torch::Tensor a, torch::Tensor b_q_weight,
                                      torch::Tensor b_qzeros,
                                      torch::Tensor b_scales,
                                      torch::Tensor b_g_idx,
                                      bool use_v2_format);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);

// FA-RDNA2: Flash-Attention v2 hand-port for AMD RDNA2 (gfx1030).
// Dispatches a fast path inside RocmAttentionImpl.forward() for
// decode (split-K) and prefill (paged varlen). Gated by
// VLLM_USE_RDNA2_FA=1 and on_gfx10x().
//
// Definitions live at global namespace in fa_rdna2.cu. The device
// kernels (fa_decode_paged_splitk_kernel_*, fa_prefill_paged_varlen_kernel_*)
// live inside vllm::fa_rdna2:: because they share storage with the
// RDNA2 GEMM paths; the host launchers above are at global scope
// because they are called from torch registration which expects
// unqualified symbol names.
torch::Tensor fa_rdna2_decode_paged(torch::Tensor Q,
                                   torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor block_table,
                                   torch::Tensor seq_lens,
                                   int64_t block_size, int64_t kv_splits,
                                   int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen(torch::Tensor Q,
                                           torch::Tensor key_cache,
                                           torch::Tensor value_cache,
                                           torch::Tensor block_table,
                                           torch::Tensor cu_query_lens,
                                           torch::Tensor seq_lens,
                                           int64_t block_size,
                                           int64_t causal,
                                           int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen_short(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t kv_splits, int64_t sliding_window);

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);


// W8A16-FP8 dense linear kernel for AMD RDNA2 (gfx1030).
// Per-tile FP8 (E4M3) -> fp16 dequant via 256-entry LUT, then v_dot2_f32_f16.
void gemm_w8a16_fp8_dense(torch::Tensor a, torch::Tensor b_q_weight,
                          torch::Tensor b_scales, torch::Tensor c,
                          int64_t group_size);

// W8A8-FP8 dense linear kernel for AMD RDNA2 (gfx1030). DeepSeek V4 Flash
// attention and shared experts: FP8 weights + FP8 activations, per-tile
// FP8 (E4M3) -> fp16 dequant via inline bit-trick (no LUT, no constant
// memory), then v_dot2_f32_f16. Per-row activation scale, per-group weight
// scale. Atomic-add epilogue into a pre-zeroed fp16 output.
void gemm_w8a8_fp8_dense(torch::Tensor a_q, torch::Tensor a_scale,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor c, int64_t group_size,
                          int64_t a_scale_K_groups);

// W4A4 MXFP4 dense linear kernel for AMD RDNA2 (gfx1030).
// E2M1 nibble -> fp16 via 16-entry constant LUT, UE8M0 scale per 32-elem
// group, then v_dot2_f32_f16. Used for non-MoE MXFP4 layers (attention,
// shared experts). Atomic-add epilogue into a pre-zeroed fp16 output.
void mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                      torch::Tensor b_q_weight, torch::Tensor b_scales,
                      int64_t size_m, int64_t size_n, int64_t size_k);

// EXL3 (QTIP-style bitshift trellis) dense GEMM kernel for AMD RDNA2/RDNA3
// (gfx1030/gfx1100). Real tile layout [K/16, N/16, 256*bits/16] int16,
// procedural codebook decode (cb 0=3inst default, 1=mcg) + v_dot2_f32_f16.
// bits = bpw in {2, 3, 4}.
void exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c, torch::Tensor trellis,
                     int64_t bits, int64_t cb);

// EXL3 (QTIP-style bitshift trellis) fused MoE GEMM kernel for AMD
// RDNA2/RDNA3 (gfx1030/gfx1100). Sorted-token-id grouping, per-expert
// packed trellis [E, K/16, N/16, 256*bits/16], codebook decode +
// v_dot2_f32_f16, topk-weighted atomic add epilogue.
void moe_exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor trellis, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded,
                         int64_t top_k, int64_t block_size_m,
                         bool mul_topk_weight, int64_t output_topk,
                         int64_t bits, int64_t cb);

// EXL3 Hadamard-128 transform for AMD RDNA2/RDNA3 (gfx1030/gfx1100).
// y = H_128(x) * (scale/sqrt(128)), optionally pre-scaled (suh, A-side) or
// post-scaled (svh, C-side). Port of exllamav3_ext.had_r_128; applied
// OUTSIDE the EXL3 K-dot (wiki kernels/exl3.md).
void exl3_hadamard_128(torch::Tensor input, torch::Tensor output,
                       torch::optional<torch::Tensor> pre_scale,
                       torch::optional<torch::Tensor> post_scale,
                       double scale);

// EXL3 6bpw dequant for AMD RDNA2/RDNA3 (gfx1030/gfx1100). Caller applies
// suh/svh Hadamard folding on GPU (PyTorch).
void exl3_dequant_bits6_mul1(torch::Tensor trellis, torch::Tensor out);

// EXL3 raw trellis decode for AMD RDNA2/RDNA3 (gfx1030/gfx1100).
// trellis [K/16, N/16, 256*bits/16] int16 -> out [K, N] fp16 (raw decode,
// no suh/svh — those stay on the activation side). One block per 16x16
// tile; exists because the fused GEMM re-decodes every tile per 8 rows
// (M_PER=8 cap), which is 256x redundant at prefill M.
void exl3_decode_trellis_rdna2(torch::Tensor trellis, torch::Tensor out,
                               int64_t bits, int64_t cb);

// Paged MQA logits for DeepSeek V4 Lightning Indexer on AMD RDNA2
// (gfx1030). AITER is CDNA-only and crashes on gfx1030; this kernel
// replaces `rocm_aiter_sparse_attn_indexer`'s paged MQA logits stage
// with a fused FP8 dequant + dot-product + ReLU + per-head weighted
// sum kernel. Output is logits [B*next_n, max_model_len] fp32 with -inf
// in padded slots. Top-K selection is done by the standard upstream
// `top_k_per_row_decode` kernel (runs on gfx1030).
torch::Tensor paged_mqa_logits_decode_rdna2(
    torch::Tensor q_fp8, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    int64_t max_model_len);

// Sparse MLA decode for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_decode_ragged_kernel` path on
// gfx1030 (the AITER MLA path is CDNA-only and does not run on
// gfx1030). One CTA per query, 32 threads (wave32), 2 heads per thread;
// online softmax with full acc_nope/acc_rope state in registers.
// FP8 (E4M3 OCP) K_nope with E8M0 block scales, bf16 K_rope. Gated
// by VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_decode_rdna2(
    torch::Tensor q,                  // [B, H, D] fp16 or bf16
    torch::Tensor main_cache,         // [num_blocks, block_size, 576] uint8
    torch::Tensor main_indices,       // [nnz] int32
    torch::Tensor main_indptr,        // [B+1] int32
    torch::Tensor extra_cache,        // [num_blocks, block_size, 576] uint8 (may be empty)
    torch::Tensor extra_indices,      // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,       // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size,
    int64_t main_num_rows,
    int64_t extra_block_size,
    int64_t extra_num_rows,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);               // [B, H, D] bf16

// Sparse MLA prefill for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_prefill_ragged_kernel` path on
// gfx1030. Same online-softmax structure as sparse_mla_decode_rdna2,
// but the kv rows are plain fp16/bf16 (no fp8 slots, no E8M0 scales —
// the fp8_ds_mla cache encoding only applies post-encoder). One CTA per
// (query, head-group), 32 threads (wave32). Gated by
// VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_prefill_rdna2(
    torch::Tensor q,                  // [T, H, D] fp16 or bf16
    torch::Tensor kv,                 // [skv, D] fp16/bf16 (contiguous rows)
    torch::Tensor indices,            // [nnz] int32
    torch::Tensor indptr,             // [T + 1] int32
    int64_t num_kv,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);               // [T, H, D] same dtype as q

// INT8 per-(token, head) KV-cache writer for AMD RDNA2 (gfx1030).
// Symmetric signed int8 quantize + write to the interleaved cache
// layout used by RDNA_ATTN backend: [2, num_blocks, H_kv, D+4, block_size]
// int8, with the last 4 int8 bytes per (block, head, slot) being the
// raw fp32 K/V scale. Used by fa_rdna2_decode_paged_int8 to populate
// the per-(token, head) scale tensor the kernel reads inside its
// cooperative load. Per the kv-int8.md wiki contract — fused i8 quant
// + scale computation in a single CTA per (token, head).
void reshape_and_cache_int8_rdna2(
    torch::Tensor key,         // [num_tokens, H_kv, D] fp16
    torch::Tensor value,       // [num_tokens, H_kv, D] fp16
    torch::Tensor kv_cache,    // [2, num_blocks, H_kv, D + 4, block_size] int8
    torch::Tensor slot_mapping // [num_tokens] int32 (-1 = skip)
);

// GatedDeltaNet (GDN) packed single-token decode for AMD RDNA2 (gfx1030).
// Hand port of fused_recurrent_gated_delta_rule_packed_decode_kernel
// (is_kda=False, scalar per-head sigmoid gating, qk-l2norm in kernel).
// Workgroup = one (token, value-head, V-tile); 256 threads hold the
// [32, 128] fp32 state tile in registers; K-reductions are warp-local
// __shfl_xor. head_k_dim must be 128; fp16 in/out, fp32 in-place state.
void gdn_decode_rdna2(
    torch::Tensor mixed_qkv,          // [B, 2*H*K + HV*V] fp16
    torch::Tensor a,                  // [B, HV] fp16
    torch::Tensor b,                  // [B, HV] fp16
    torch::Tensor A_log,              // [HV] fp32
    torch::Tensor dt_bias,            // [HV] fp32
    torch::Tensor out,                // [B, 1, HV, V] fp16
    torch::Tensor initial_state,      // [blocks, HV, V, K] fp32, in-place
    torch::Tensor ssm_state_indices,  // [B] int32
    double scale,
    bool use_qk_l2norm);

// GDN prefill kernels for AMD RDNA2 (gfx1030). Hand ports of the Triton/FLA
// chain `chunk_gated_delta_rule_fwd` (chunk.py:23-86) decomposed into 5 HIP
// kernels; varlen uses cu_seqlens/chunk_indices/chunk_offsets (pass empty
// for non-varlen).

// fused_post_conv_prep + chunk_local_cumsum fused. g is already cumsum'd
// when it leaves this kernel (fold of `chunk_local_cumsum` in chunk.py:37).
void gdn_prefill_prep_rdna2(
    torch::Tensor mixed_qkv,    // [L, qkv_dim] fp16 contiguous in last dim
    torch::Tensor a,            // [L, HV] fp16
    torch::Tensor b,            // [L, HV] fp16
    torch::Tensor A_log,        // [HV] fp32 or fp16 contiguous
    torch::Tensor dt_bias,      // [HV] fp32 or fp16 contiguous
    torch::Tensor q,            // [L, H, K] fp16 (output)
    torch::Tensor k_out,        // [L, H, K] fp16 (output)
    torch::Tensor v,            // [L, HV, V] fp16 (output)
    torch::Tensor g_cumsum,     // [L, HV] fp32 (output)
    torch::Tensor beta,         // [L, HV] fp32 (output)
    torch::Tensor cu_seqlens,   // [N+1] int32
    torch::Tensor chunk_indices);// [NT, 2] int32

// chunk_scaled_dot_kkt. A[i,j] = beta_i * exp(g_i - g_j) * (k_i . k_j)
// for i > j (strict), else 0. The [64,128] @ [128,64] dot is fp32 FMA
// (no V_DOT2: _CAST_DOT_TO_K_DTYPE is False on gfx1030, beta*k stays fp32).
void gdn_prefill_kkt_rdna2(
    torch::Tensor k,            // [B, T, Hg, K] fp16 contiguous in last dim
    torch::Tensor beta,         // [B, T, H] fp32
    torch::Tensor g,            // [B, T, H] fp32 (cumsum'd g, from prep)
    torch::Tensor A,            // [B, T, H, BT] fp32 (output)
    torch::Tensor cu_seqlens,   // [N+1] int32
    torch::Tensor chunk_indices);// [NT, 2] int32

// solve_tril + recompute_w_u FUSED. Same (NT, B*H) grid; A_inv [64,64]
// stays on-chip between the fp32 16x16 forward-substitution + Schur phase
// and the V_DOT2_F32_F16 w/u production phase.
void gdn_prefill_solve_wy_rdna2(
    torch::Tensor A,            // [B, T, H, BT] fp32 (from kkt)
    torch::Tensor k,            // [B, T, Hg, K] fp16
    torch::Tensor v,            // [B, T, H, V] fp16
    torch::Tensor beta,         // [B, T, H] fp32
    torch::Tensor g,            // [B, T, H] fp32 (cumsum'd g, from prep)
    torch::Tensor A_inv,        // [B, T, H, BT] fp16 (output)
    torch::Tensor w,            // [B, T, H, K] fp16 (output)
    torch::Tensor u,            // [B, T, H, V] fp16 (output)
    torch::Tensor cu_seqlens,   // [N+1] int32
    torch::Tensor chunk_indices);// [NT, 2] int32

// chunk_gated_delta_rule_fwd_h (serial inter-chunk recurrence). Reuses
// Stage-1 decode layout (256 threads = 32 v-rows x 8 k-slices, h [32,128]
// fp32 register-resident). initial_state = h0, final_state = last h per seq.
void gdn_prefill_delta_h_rdna2(
    torch::Tensor k,                  // [B, T, Hg, K] fp16
    torch::Tensor u,                  // [B, T, H, V] fp16
    torch::Tensor w,                  // [B, T, H, K] fp16
    torch::Tensor g,                  // [B, T, H] fp32 (cumsum'd g)
    torch::Tensor h,                  // [NT, H, V, K] fp16 (per-chunk h, output)
    torch::Tensor v_new,              // [B, T, H, V] fp16 (output)
    c10::optional<torch::Tensor> initial_state,  // [N, H, V, K] fp32 or undefined
    c10::optional<torch::Tensor> final_state,    // [N, H, V, K] fp32 or undefined
    c10::optional<torch::Tensor> cu_seqlens,     // [N+1] int32 or undefined
    c10::optional<torch::Tensor> chunk_offsets,  // [N+1] int32 or undefined
    int64_t chunk_size);              // must be 64

// chunk_fwd_o. q.A (intra-chunk) + q.h (state) with V_DOT2_F32_F16 and the
// inclusive `>=` causal mask. h: 5D non-varlen or 4D varlen, fp16 or fp32.
void gdn_prefill_o_rdna2(
    torch::Tensor q,             // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor k,             // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor v,             // [B, T, H, V] fp16 contiguous in V
    torch::Tensor h,             // 5D non-varlen or 4D varlen; fp16 or fp32
    torch::Tensor g,             // [B, T, H] fp32 (cumsum'd g)
    torch::Tensor o,             // [B, T, H, V] fp16 (output)
    double scale,
    torch::Tensor cu_seqlens,    // [N+1] int32
    torch::Tensor chunk_offsets);// [N+1] int32

// Paged MQA logits for DeepSeek V4 Lightning Indexer on AMD RDNA2
// (gfx1030). AITER is CDNA-only and crashes on gfx1030; this kernel
// replaces `rocm_aiter_sparse_attn_indexer`'s paged MQA logits stage
// with a fused FP8 dequant + dot-product + ReLU + per-head weighted
// sum kernel. Output is logits [B*next_n, max_model_len] fp32 with -inf
// in padded slots. Top-K selection is done by the standard upstream
// `top_k_per_row_decode` kernel (runs on gfx1030).
torch::Tensor paged_mqa_logits_decode_rdna2(
    torch::Tensor q_fp8, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    int64_t max_model_len);
// ===== RDNA2 declarations backported from rdna2_extras ops.h =====
// (Group1-9 ported the .cu sources + torch_bindings.cpp registrations;
//  these are the matching ops.h declarations needed to make them compile.)

void rms_norm(torch::Tensor& out, const torch::Tensor& input,
              const torch::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        const torch::Tensor& weight, double epsilon);

void moe_w8a16_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor b_qzeros, torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor expert_ids,
                           torch::Tensor num_tokens_post_padded,
                           int64_t top_k, int64_t block_size_m,
                           bool mul_topk_weight, int64_t output_topk);

void moe_mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor expert_ids,
                           torch::Tensor num_tokens_post_padded,
                           int64_t top_k, int64_t block_size_m,
                           bool mul_topk_weight, int64_t output_topk);

void moe_w8a16_fp8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                               torch::Tensor b_q_weight,
                               torch::Tensor b_scales,
                               torch::Tensor b_qzeros,
                               torch::Tensor topk_weights,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_padded,
                               int64_t top_k, int64_t block_size_m,
                               bool mul_topk_weight, int64_t output_topk);

void mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                      torch::Tensor b_q_weight, torch::Tensor b_scales,
                      int64_t size_m, int64_t size_n, int64_t size_k);


void sparse_mla_decode_rdna2(
    torch::Tensor q,                  // [B, H, D] fp16 or bf16
    torch::Tensor main_cache,         // [num_blocks, block_size, 576] uint8
    torch::Tensor main_indices,       // [nnz] int32
    torch::Tensor main_indptr,        // [B+1] int32
    torch::Tensor extra_cache,        // [num_blocks, block_size, 576] uint8 (may be empty)
    torch::Tensor extra_indices,      // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,       // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size,
    int64_t main_num_rows,
    int64_t extra_block_size,
    int64_t extra_num_rows,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);

void sparse_mla_prefill_rdna2(
    torch::Tensor q,                  // [T, H, D] fp16 or bf16
    torch::Tensor kv,                 // [skv, D] fp16/bf16 (contiguous rows)
    torch::Tensor indices,            // [nnz] int32
    torch::Tensor indptr,             // [T + 1] int32
    int64_t num_kv,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);

void reshape_and_cache_int8_rdna2(
    torch::Tensor key,         // [num_tokens, H_kv, D] fp16
    torch::Tensor value,       // [num_tokens, H_kv, D] fp16
    torch::Tensor kv_cache,    // [2, num_blocks, H_kv, D + 4, block_size] int8
    torch::Tensor slot_mapping // [num_tokens] int32 (-1 = skip)
);

void gdn_decode_rdna2(
    torch::Tensor mixed_qkv,          // [B, 2*H*K + HV*V] fp16
    torch::Tensor a,                  // [B, HV] fp16
    torch::Tensor b,                  // [B, HV] fp16
    torch::Tensor A_log,              // [HV] fp32
    torch::Tensor dt_bias,            // [HV] fp32
    torch::Tensor out,                // [B, 1, HV, V] fp16
    torch::Tensor initial_state,      // [blocks, HV, V, K] fp32, in-place
    torch::Tensor ssm_state_indices,  // [B] int32
    double scale,
    bool use_qk_l2norm);


// GLM-5.3-Flash (Glm5Next) DSA kpool indexer for AMD RDNA2 (gfx1030).
// Pools of 4 over the packed indexer cache rows [k|gate|valid], softmax
// compression with the learned APE, q.pool_key logits + relu + weighted
// head-sum. Topk stays caller-side (torch). First-cut kernel; default
// path is the torch scan in the GLM5DSA backend (VLLM_GLM5_DSA_HIP=1).
void glm5_dsa_indexer_rdna2(
    torch::Tensor q_idx,              // [Q, 32, 128] fp16
    torch::Tensor packed,             // [B, T, 257] fp16 (k|gate|valid)
    torch::Tensor weights,            // [Q, 32] fp32 (pre-scaled by 32^-0.5)
    torch::Tensor kv_lens,            // [Q] int32
    torch::Tensor ape,                // [4, 128] fp32
    torch::Tensor pool_indices_out,   // [Q, max_pools, 4] int32, in-place
    torch::Tensor pool_valid_out,     // [Q, max_pools] uint8, in-place
    torch::Tensor scores_out,         // [Q, max_pools] fp32, in-place
    int64_t kpool);

// GLM-5.3-Flash DSA gathered MLA-NoPE decode for AMD RDNA2 (gfx1030).
// q/k/v 256-dim, 64 heads, NO RoPE. Attends only over the caller-
// gathered topk selection (k_sel/v_sel [B, S, 256] or pre-expanded
// [B, S, 64*256]); invalid slots masked via sel_valid. First-cut kernel;
// default path is the torch scan (VLLM_GLM5_DSA_HIP=1).
void glm5_dsa_mla_decode_rdna2(
    torch::Tensor q_nope,             // [B, 64, 256] fp16
    torch::Tensor k_sel,              // [B, S, 256] or [B, S, 64*256] fp16
    torch::Tensor v_sel,              // [B, S, 256] or [B, S, 64*256] fp16
    torch::Tensor sel_valid,          // [B, S] uint8
    torch::Tensor out,                // [B, 64*256] fp16, in-place
    double scale);

// GLM-5.3-Flash (Glm5Next) KDA (Kimi Delta Attention) for AMD RDNA2
// (gfx1030). 64 heads x 128, causal conv4, lower-bound sigmoid gate,
// per-(head, k-dim) log-decay. NOT the Qwen GDN tiles (16/48 heads) and
// NOT the Kimi fused_kda_decode tiles — GLM-specific packing/math per
// modeling_glm5_next.py. First-cut kernels gated by VLLM_GLM5_KDA_HIP=1
// (torch fallback in glm5_kda_linear_attn.py is the default path).
void glm5_kda_decode_rdna2(
    torch::Tensor qkv_raw,            // [B, 3*H*D] fp16 pre-conv q|k|v
    torch::Tensor conv_w,             // [3*H*D, 4] fp16 (q|k|v order)
    torch::Tensor conv_state,         // [slots, 3*H*D, 3] fp16, in-place
    torch::Tensor A_log,              // [H] fp32
    torch::Tensor dt_bias,            // [H*D] fp32
    torch::Tensor f,                  // [B, H*D] fp16 forget pre-activation
    torch::Tensor beta,               // [B, H] fp16 pre-sigmoid
    torch::Tensor out_gate,           // [B, H*D] fp16 pre-sigmoid
    torch::Tensor norm_w,             // [D] fp32
    torch::Tensor state_indices,      // [B] int32
    torch::Tensor ssm_state,          // [slots, H, D, D] fp32, in-place
    torch::Tensor out,                // [B, H*D] fp16, in-place
    double lower_bound,
    double norm_eps);

// GLM-5.3 KDA chunked-prefill chain (chunk 64, varlen via cu_seqlens /
// chunk_indices; pass empty for non-varlen). Math per
// chunk_kimi_delta_attention: prep emits l2norm'd q/k, silu'd v and the
// per-(head, dim) log-decay g with chunk-local inclusive cumsum; kkt
// emits P[i,j] = beta_i * sum_d k_i[d]*k_j[d]*exp(g_i[d]-g_j[d]) (i>j)
// so solve_wy computes (I+P)^-1 == the HF WY transform; delta_h is the
// inter-chunk state pass; o applies the q scale and emits raw o (gated
// norm stays caller-side).
void glm5_kda_prefill_prep_rdna2(
    torch::Tensor mixed_qkv,          // [L, 3*H*D] fp16 varlen tokens
    torch::Tensor conv_w,             // [3*H*D, 4] fp16
    torch::Tensor A_log,              // [H] fp32
    torch::Tensor dt_bias,            // [H*D] fp32
    torch::Tensor f,                  // [L, H*D] fp16
    torch::Tensor beta_raw,           // [L, H] fp16 pre-sigmoid
    torch::Tensor q_out,              // [L, H, D] fp16, in-place (l2norm'd)
    torch::Tensor k_out,              // [L, H, D] fp16, in-place (l2norm'd)
    torch::Tensor v_out,              // [L, H, D] fp16, in-place
    torch::Tensor g_out,              // [L, H, D] fp32, in-place (cumsum'd)
    torch::Tensor beta_out,           // [L, H] fp32, in-place (sigmoid'd)
    torch::Tensor cu_seqlens,         // [N+1] int32
    torch::Tensor chunk_indices,      // [NT, 2] int32
    double lower_bound);

void glm5_kda_prefill_kkt_rdna2(
    torch::Tensor k,                  // [L, H, D] fp16
    torch::Tensor beta,               // [L, H] fp32
    torch::Tensor g,                  // [L, H, D] fp32 (cumsum'd)
    torch::Tensor A,                  // [NT, H, 64, 64] fp32, in-place
    torch::Tensor cu_seqlens,         // [N+1] int32
    torch::Tensor chunk_indices);     // [NT, 2] int32

void glm5_kda_prefill_solve_wy_rdna2(
    torch::Tensor A,                  // [NT, H, 64, 64] fp32
    torch::Tensor k,                  // [L, H, D] fp16
    torch::Tensor v,                  // [L, H, D] fp16
    torch::Tensor beta,               // [L, H] fp32
    torch::Tensor g,                  // [L, H, D] fp32 (cumsum'd)
    torch::Tensor A_inv,              // [NT, H, 64, 64] fp16, in-place
    torch::Tensor w,                  // [L, H, D] fp16, in-place
    torch::Tensor u,                  // [L, H, D] fp16, in-place
    torch::Tensor cu_seqlens,         // [N+1] int32
    torch::Tensor chunk_indices);     // [NT, 2] int32

void glm5_kda_prefill_delta_h_rdna2(
    torch::Tensor k,                  // [L, H, D] fp16
    torch::Tensor u,                  // [L, H, D] fp16
    torch::Tensor w,                  // [L, H, D] fp16
    torch::Tensor g,                  // [L, H, D] fp32 (cumsum'd)
    torch::Tensor h,                  // per-chunk states fp16/fp32, in-place
    torch::Tensor v_new,              // [L, H, D] fp16, in-place
    c10::optional<torch::Tensor> initial_state,
    c10::optional<torch::Tensor> final_state,
    c10::optional<torch::Tensor> cu_seqlens,
    c10::optional<torch::Tensor> chunk_offsets,
    int64_t chunk_size);

void glm5_kda_prefill_o_rdna2(
    torch::Tensor q,                  // [L, H, D] fp16 (l2norm'd)
    torch::Tensor k,                  // [L, H, D] fp16
    torch::Tensor v_new,              // [L, H, D] fp16
    torch::Tensor h,                  // per-chunk states fp16/fp32
    torch::Tensor g,                  // [L, H, D] fp32 (cumsum'd)
    torch::Tensor o,                  // [L, H, D] fp16, in-place (raw o)
    double scale,
    torch::Tensor cu_seqlens,         // [N+1] int32
    torch::Tensor chunk_offsets);     // [N+1] int32
