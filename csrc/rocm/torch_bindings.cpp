#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points for asymmetric quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1030
  // W4A16 GPTQ kernel for AMD RDNA2 (gfx1030).
  rocm_ops.def(
      "gptq_gemm_rdna2(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna2", torch::kCUDA, &gptq_gemm_rdna2);

  rocm_ops.def(
      "gptq_gemm_rdna2_prefill(Tensor a, Tensor b_q_weight, "
      "Tensor b_qzeros, Tensor b_scales, Tensor b_g_idx, "
      "bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna2_prefill", torch::kCUDA,
                &gptq_gemm_rdna2_prefill);

  // FA-RDNA2: Flash-Attention v2 hand-port for AMD RDNA2 (gfx1030).
  // Dispatched via a fast path in RocmAttentionImpl.forward().
  rocm_ops.def(
      "fa_rdna2_decode_paged(Tensor Q, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, int block_size, int kv_splits, "
      "int sliding_window) -> Tensor");
  rocm_ops.impl("fa_rdna2_decode_paged", torch::kCUDA,
                &fa_rdna2_decode_paged);

  // GDN packed single-token decode for AMD RDNA2 (gfx1030). Dispatched
  // from Qwen3NextGatedDeltaNet._forward_core_decode_non_spec on gfx10x.
  rocm_ops.def(
      "gdn_decode_rdna2(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor A_log, Tensor dt_bias, Tensor! out, Tensor! initial_state, "
      "Tensor ssm_state_indices, float scale, bool use_qk_l2norm) -> ()");
  rocm_ops.impl("gdn_decode_rdna2", torch::kCUDA, &gdn_decode_rdna2);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int sliding_window) "
      "-> Tensor");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen_short(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int sliding_window) "
      "-> Tensor");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen_short", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen_short);

  // GDN prefill kernels for AMD RDNA2 (gfx1030). 5-kernel chain matching
  // chunk.py:23-86: prep -> kkt -> solve_wy -> delta_h -> o. Optional
  // cu_seqlens/chunk_indices/chunk_offsets for varlen (B == 1).
  rocm_ops.def(
      "gdn_prefill_prep_rdna2(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor A_log, Tensor dt_bias, Tensor! q, Tensor! k_out, Tensor! v, "
      "Tensor! g_cumsum, Tensor! beta, Tensor cu_seqlens, "
      "Tensor chunk_indices) -> ()");
  rocm_ops.impl("gdn_prefill_prep_rdna2", torch::kCUDA,
                &gdn_prefill_prep_rdna2);

  rocm_ops.def(
      "gdn_prefill_kkt_rdna2(Tensor k, Tensor beta, Tensor g, Tensor! A, "
      "Tensor cu_seqlens, Tensor chunk_indices) -> ()");
  rocm_ops.impl("gdn_prefill_kkt_rdna2", torch::kCUDA,
                &gdn_prefill_kkt_rdna2);

  rocm_ops.def(
      "gdn_prefill_solve_wy_rdna2(Tensor A, Tensor k, Tensor v, Tensor beta, "
      "Tensor g, Tensor! A_inv, Tensor! w, Tensor! u, Tensor cu_seqlens, "
      "Tensor chunk_indices) -> ()");
  rocm_ops.impl("gdn_prefill_solve_wy_rdna2", torch::kCUDA,
                &gdn_prefill_solve_wy_rdna2);

  rocm_ops.def(
      "gdn_prefill_delta_h_rdna2(Tensor k, Tensor u, Tensor w, Tensor g, "
      "Tensor! h, Tensor! v_new, Tensor? initial_state, Tensor? final_state, "
      "Tensor? cu_seqlens, Tensor? chunk_offsets, int chunk_size) -> ()");
  rocm_ops.impl("gdn_prefill_delta_h_rdna2", torch::kCUDA,
                &gdn_prefill_delta_h_rdna2);

  rocm_ops.def(
      "gdn_prefill_o_rdna2(Tensor q, Tensor k, Tensor v, Tensor h, "
      "Tensor g, Tensor! o, float scale, Tensor cu_seqlens, "
      "Tensor chunk_offsets) -> ()");
  rocm_ops.impl("gdn_prefill_o_rdna2", torch::kCUDA,
                &gdn_prefill_o_rdna2);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen_splitk(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int kv_splits, "
      "int sliding_window) -> Tensor");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen_splitk", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen_splitk);

  rocm_ops.def(
      "moe_gptq_gemm_rdna2(Tensor a, Tensor(a!) c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna2", torch::kCUDA, &moe_gptq_gemm_rdna2);

  // W4A4 MXFP4 (DeepSeek V4 native: E2M1 + UE8M0) fused MoE kernel for
  // RDNA2 (gfx1030). Native V_DOT2 path; no Marlin/CUTLASS fallback.
  rocm_ops.def(
      "moe_mxfp4_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_mxfp4_gemm_rdna2", torch::kCUDA, &moe_mxfp4_gemm_rdna2);

  // W4A4 MXFP4 dense (non-MoE) GEMM kernel for RDNA2 (gfx1030).
  // Used for MXFP4 attention and shared experts.
  rocm_ops.def(
      "mxfp4_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, "
      "int size_m, int size_n, int size_k) -> ()");
  rocm_ops.impl("mxfp4_gemm_rdna2", torch::kCUDA, &mxfp4_gemm_rdna2);

  // W8A8-FP8 dense linear kernel for RDNA2 (gfx1030). DeepSeek V4 Flash
  // attention / shared experts: FP8 weights + FP8 activations, per-tile
  // FP8->fp16 dequant (no LUT, inline bit-trick), then v_dot2_f32_f16.
  rocm_ops.def(
      "gemm_w8a8_fp8_dense(Tensor a_q, Tensor a_scale, Tensor b_q_weight, "
      "Tensor b_scales, Tensor(a!) c, int group_size) -> ()");
  rocm_ops.impl("gemm_w8a8_fp8_dense", torch::kCUDA, &gemm_w8a8_fp8_dense);
#endif

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
