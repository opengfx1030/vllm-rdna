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

  // T44: push-based one-shot all-reduce for small TP messages on gfx1030
  rocm_ops.def(
      "rdna_ar_init(int rank, int world, Tensor device_ids, int max_bytes, "
      "str shm_name) -> Tensor");
  rocm_ops.impl("rdna_ar_init", torch::kCPU, &rdna_ar_init);
  rocm_ops.def("rdna_ar_connect(int handle, Tensor handles) -> ()");
  rocm_ops.impl("rdna_ar_connect", torch::kCPU, &rdna_ar_connect);
  rocm_ops.def("rdna_ar_can(int handle, Tensor t) -> bool");
  rocm_ops.impl("rdna_ar_can", torch::kCUDA, &rdna_ar_can);
  rocm_ops.def("rdna_ar_all_reduce(int handle, Tensor t) -> Tensor");
  rocm_ops.impl("rdna_ar_all_reduce", torch::kCUDA, &rdna_ar_all_reduce);
  // no tensor arguments -> no dispatch key; register as catch-all
  rocm_ops.def("rdna_ar_timed_out(int handle) -> bool", &rdna_ar_timed_out);
  rocm_ops.def("rdna_ar_fast_calls(int handle) -> int", &rdna_ar_fast_calls);

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
      "moe_gptq_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna2", torch::kCUDA, &moe_gptq_gemm_rdna2);

  // W8A16 (INT8 weight + fp16 act) fused MoE kernel for RDNA2.
  rocm_ops.def(
      "moe_w8a16_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_w8a16_gemm_rdna2", torch::kCUDA, &moe_w8a16_gemm_rdna2);

  // W4A4 MXFP4 (DeepSeek V4 native: E2M1 + UE8M0) fused MoE kernel for
  // RDNA2 (gfx1030). Native V_DOT2 path; no Marlin/CUTLASS fallback.
  rocm_ops.def(
      "moe_mxfp4_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor topk_weights, "
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

  // W8A16-FP8 (FP8 weight + fp16 act) fused MoE kernel for RDNA2.
  // Disabled 2026-08-05: moe_w8a16_fp8_rdna2.cu excluded from gfx1030 build
  // (namespace parser error). MoE experts fall back to existing
  // CompressedTensorsWNA16MoEMethod (INT4) or scaled_mm dispatch.
  // rocm_ops.def(
  //     "moe_w8a16_fp8_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
  //     "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
  //     "Tensor sorted_token_ids, Tensor expert_ids, "
  //     "Tensor num_tokens_post_padded, "
  //     "int top_k, int block_size_m, bool mul_topk_weight, "
  //     "int output_topk) -> ()");
  // rocm_ops.impl("moe_w8a16_fp8_gemm_rdna2", torch::kCUDA,
  //               &moe_w8a16_fp8_gemm_rdna2);

  // W8A16-FP8 dense linear kernel for RDNA2 (gfx1030). Per-tile FP8->fp16
  // conversion via constant-memory LUT, then v_dot2_f32_f16. Atomic-add
  // epilogue into a pre-zeroed fp16 output.
  rocm_ops.def(
      "gemm_w8a16_fp8_dense(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor(a!) c, int group_size) -> ()");
  rocm_ops.impl("gemm_w8a16_fp8_dense", torch::kCUDA,
                &gemm_w8a16_fp8_dense);

  // Paged MQA logits for DeepSeek V4 Lightning Indexer (gfx1030).
  // Replaces the AITER-only decode path of rocm_aiter_sparse_attn_indexer.
  rocm_ops.def(
      "paged_mqa_logits_decode_rdna2(Tensor q_fp8, Tensor kv_cache, "
      "Tensor weights, Tensor context_lens, Tensor block_tables, "
      "int max_model_len) -> Tensor");
  rocm_ops.impl("paged_mqa_logits_decode_rdna2", torch::kCUDA,
                &paged_mqa_logits_decode_rdna2);

  // W8A8-FP8 dense linear kernel for RDNA2 (gfx1030). DeepSeek V4 Flash
  // attention / shared experts: FP8 weights + FP8 activations, per-tile
  // FP8->fp16 dequant (no LUT, inline bit-trick), then v_dot2_f32_f16.
  // a_scale: [1] / [M] / [M, K/gs] (per-block-K dynamic act quant).
  // a_scale_K_groups: number of K-blocks in a_scale (1 for per-row/tensor).
  rocm_ops.def(
      "gemm_w8a8_fp8_dense(Tensor a_q, Tensor a_scale, Tensor b_q_weight, "
      "Tensor b_scales, Tensor(a!) c, int group_size, int a_scale_K_groups) -> ()");
  rocm_ops.impl("gemm_w8a8_fp8_dense", torch::kCUDA, &gemm_w8a8_fp8_dense);

  // Sparse MLA decode for DeepSeek V4 (gfx1030). Replaces the Triton
  // _sparse_attn_decode_ragged_kernel path on gfx1030 (the AITER MLA
  // path is CDNA-only and does not run on gfx1030). 1 CTA per query,
  // 32 threads (wave32), 2 heads per thread; online softmax with
  // full acc_nope/acc_rope state in registers. FP8 (E4M3 OCP) K_nope
  // with E8M0 block scales, bf16 K_rope. q/out may be fp16 (gfx1030)
  // or bf16 (RDNA3+). Gated by VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
  rocm_ops.def(
      "sparse_mla_decode_rdna2(Tensor q, Tensor main_cache, "
      "Tensor main_indices, Tensor main_indptr, "
      "Tensor extra_cache, Tensor extra_indices, Tensor extra_indptr, "
      "int main_block_size, int main_num_rows, "
      "int extra_block_size, int extra_num_rows, float scale, "
      "Tensor attn_sink, Tensor(a!) out) -> ()");
  rocm_ops.impl("sparse_mla_decode_rdna2", torch::kCUDA,
                &sparse_mla_decode_rdna2);

  // Sparse MLA prefill for DeepSeek V4 (gfx1030). Replaces the Triton
  // `_sparse_attn_prefill_ragged_kernel` path on gfx1030. Same
  // online-softmax structure as sparse_mla_decode_rdna2 but kv rows are
  // plain fp16/bf16 (no fp8 slots, no E8M0 scales). q/out may be fp16
  // (gfx1030) or bf16 (RDNA3+). Gated by VLLM_USE_RDNA2_MLA=1 and
  // on_gfx10x().
  rocm_ops.def(
      "sparse_mla_prefill_rdna2(Tensor q, Tensor kv, "
      "Tensor indices, Tensor indptr, int num_kv, float scale, "
      "Tensor attn_sink, Tensor(a!) out) -> ()");
  rocm_ops.impl("sparse_mla_prefill_rdna2", torch::kCUDA,
                &sparse_mla_prefill_rdna2);

  // INT8 per-(token, head) KV-cache writer for RDNA2 (gfx1030).
  // Quantizes fp16 K/V to int8 with per-(token, head) scales and writes
  // them into the interleaved cache layout the RDNA2 FA decode kernel
  // reads (D bytes data + 4 bytes scale per slot, per kv-int8.md wiki
  // contract). Wired into vllm/v1/attention/backends/rdna_attn.py for
  // the INT8_PER_TOKEN_HEAD kv_cache_dtype path.
  rocm_ops.def(
      "reshape_and_cache_int8_rdna2(Tensor key, Tensor value, "
      "Tensor(a!) kv_cache, Tensor slot_mapping) -> ()");
  rocm_ops.impl("reshape_and_cache_int8_rdna2", torch::kCUDA,
                &reshape_and_cache_int8_rdna2);
#endif

  // EXL3 (QTIP-style bitshift trellis) kernels are RDNA-generic
  // (gfx1030 + gfx1100): registered unconditionally (outside the arch
  // guards). Procedural codebook decode (cb: 0=3inst, 1=mcg), no scale
  // tensor. bits = bpw in {2, 3, 4}.
  rocm_ops.def(
      "moe_exl3_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk, int bits, int cb) -> ()");
  rocm_ops.impl("moe_exl3_gemm_rdna2", torch::kCUDA, &moe_exl3_gemm_rdna2);

  rocm_ops.def(
      "exl3_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "int bits, int cb) -> ()");
  rocm_ops.impl("exl3_gemm_rdna2", torch::kCUDA, &exl3_gemm_rdna2);

  // EXL3 Hadamard-128 (suh/svh): y = H_128(x) * (scale/sqrt(128)), outside
  // the K-dot. Port of exllamav3_ext.had_r_128.
  rocm_ops.def(
      "exl3_hadamard_128(Tensor input, Tensor! output, "
      "Tensor? pre_scale, Tensor? post_scale, float scale) -> ()");
  rocm_ops.impl("exl3_hadamard_128", torch::kCUDA, &exl3_hadamard_128);

  rocm_ops.def(
      "exl3_dequant_bits6_mul1(Tensor trellis, Tensor(a!) out) -> ()");
  rocm_ops.impl("exl3_dequant_bits6_mul1", torch::kCUDA,
                &exl3_dequant_bits6_mul1);

  rocm_ops.def(
      "exl3_decode_trellis_rdna2(Tensor trellis, Tensor! out, int bits, "
      "int cb) -> ()");
  rocm_ops.impl("exl3_decode_trellis_rdna2", torch::kCUDA,
                &exl3_decode_trellis_rdna2);

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

  // HIP RMSNorm / FusedAddRmsNorm — AOT-compiled, cudagraph-safe replacement
  // for the upstream Triton layer_norm_fwd_kernel (which JIT-compiles per
  // shape and breaks cudagraph capture on gfx1030). See csrc/rocm/layernorm.cu.
  rocm_ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) "
      "-> ()");
  rocm_ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  rocm_ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  rocm_ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
