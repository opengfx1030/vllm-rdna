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
