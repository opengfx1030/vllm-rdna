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
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Sequential-layout W4A16 MoE skinny GEMV (Triton moe_wna16 packing).
  rocm_ops.def(
      "moe_skinny_int4_decode(Tensor input, Tensor w13, Tensor w13_scale, "
      "Tensor w2, Tensor w2_scale, Tensor topk_weights, Tensor topk_ids, "
      "Tensor! act_buf, Tensor! output, int group_size, Tensor? expert_map) "
      "-> ()");
  rocm_ops.impl("moe_skinny_int4_decode", torch::kCUDA,
                &moe_skinny_int4_decode);

  // Native resident-layout W4A16 MoE skinny GEMV.
  rocm_ops.def(
      "moe_resident_int4_decode(Tensor input, Tensor w13, Tensor w13_scale, "
      "Tensor w2, Tensor w2_scale, Tensor topk_weights, Tensor topk_ids, "
      "Tensor! act_buf, Tensor! output, int group_size, Tensor? expert_map) "
      "-> ()");
  rocm_ops.impl("moe_resident_int4_decode", torch::kCUDA,
                &moe_resident_int4_decode);

  // gfx1030 fp16/int8 skinny GEMM for decode-sized M (T43/T45).
  rocm_ops.def("gemv_f16_rdna2(Tensor x, Tensor w, Tensor? bias) -> Tensor");
  rocm_ops.impl("gemv_f16_rdna2", torch::kCUDA, &gemv_f16_rdna2);
  rocm_ops.def(
      "gemv_i8_rdna2(Tensor x, Tensor w, Tensor scale, Tensor? bias) -> "
      "Tensor");
  rocm_ops.impl("gemv_i8_rdna2", torch::kCUDA, &gemv_i8_rdna2);

  // T46 fused decode glue (hyper-connection mix, shared expert).
  rocm_ops.def(
      "rdna_gemv_act(Tensor x, Tensor w, Tensor? scale, int act_cols, "
      "float act_scale) -> Tensor");
  rocm_ops.impl("rdna_gemv_act", torch::kCUDA, &rdna_gemv_act);
  rocm_ops.def(
      "rdna_hc_up_gate_mix(Tensor lora, Tensor w, Tensor? scale, Tensor xn, "
      "int hc_count) -> Tensor");
  rocm_ops.impl("rdna_hc_up_gate_mix", torch::kCUDA, &rdna_hc_up_gate_mix);
#ifdef VLLM_ROCM_GFX1030
  // HC prefill (M > 8): fused silu/up/sigmoid/mix, optional split-K down.
  rocm_ops.def(
      "rdna_hc_up_gate_mix_prefill(Tensor dai, Tensor w_up, Tensor xn, "
      "int lora_rank, int hc_count) -> Tensor");
  rocm_ops.impl("rdna_hc_up_gate_mix_prefill", torch::kCUDA,
                &rdna_hc_up_gate_mix_prefill);
  rocm_ops.def(
      "rdna_hc_mix_prefill(Tensor xn, Tensor w_down, Tensor w_up, "
      "int lora_rank, int hc_count) -> (Tensor, Tensor)");
  rocm_ops.impl("rdna_hc_mix_prefill", torch::kCUDA, &rdna_hc_mix_prefill);
  // Qwen4Exp RDNA2 MonoKernel (vllm/models/qwen4_exp/amd/mono/rdna2/): K1 + K2.
  rocm_ops.def(
      "mono_k2_rdna2_workspace_bytes(int H, int R, int ND, int E, int Ise, "
      "int I) -> int");
  rocm_ops.impl("mono_k2_rdna2_workspace_bytes",
                &mono_k2_rdna2_workspace_bytes);
  rocm_ops.def(
      "mono_k2_rdna2(Tensor res, Tensor blk, Tensor inj, Tensor norm_w, "
      "Tensor wd, Tensor wu, Tensor wr, Tensor se_w1, Tensor se_w2, "
      "Tensor se_wg, Tensor w13, Tensor s13, Tensor z13, Tensor w2, Tensor s2, "
      "Tensor z2, Tensor! res_out, Tensor! inj_out, Tensor! out, "
      "Tensor! workspace, int expert_base, int zero_offset, int num_wg, "
      "float eps, float timeout_ms) -> ()");
  rocm_ops.impl("mono_k2_rdna2", torch::kCUDA, &mono_k2_rdna2);
  rocm_ops.def("mono_k2_rdna2_error(Tensor workspace) -> Tensor");
  rocm_ops.impl("mono_k2_rdna2_error", torch::kCUDA, &mono_k2_rdna2_error);
  rocm_ops.def("mono_k2_rdna2_mark(Tensor! stats, int kind, int width) -> ()");
  rocm_ops.impl("mono_k2_rdna2_mark", torch::kCUDA, &mono_k2_rdna2_mark);
  rocm_ops.def(
      "mono_k2_rdna2_workspace_alloc(int nbytes, int device) -> Tensor");
  rocm_ops.impl("mono_k2_rdna2_workspace_alloc",
                &mono_k2_rdna2_workspace_alloc);
  rocm_ops.def("mono_k2_rdna2_mem_flags(Tensor t) -> int");
  rocm_ops.impl("mono_k2_rdna2_mem_flags", &mono_k2_rdna2_mem_flags);
  rocm_ops.def("mono_k2_rdna2_set_ar(int handle, int device) -> int");
  rocm_ops.impl("mono_k2_rdna2_set_ar", &mono_k2_rdna2_set_ar);
  rocm_ops.def(
      "mono_k1_rdna2(Tensor res, Tensor blk, Tensor inj, Tensor norm_w, Tensor "
      "wd, "
      "Tensor wu, Tensor! res_out, Tensor! inj_out, Tensor! out, Tensor! "
      "workspace, "
      "int num_wg, float eps, float timeout_ms) -> ()");
  rocm_ops.impl("mono_k1_rdna2", torch::kCUDA, &mono_k1_rdna2);
  rocm_ops.def(
      "mono_k2_rdna2_e1a(Tensor res, Tensor blk, Tensor inj, Tensor norm_w, "
      "Tensor wd, "
      "Tensor wu, Tensor wq, Tensor wba, Tensor convw, Tensor convb, "
      "Tensor! conv_state, Tensor! ssm_state, Tensor alog, Tensor dtb, Tensor "
      "gnorm_w, "
      "Tensor wo, Tensor sidx, Tensor! res_out, Tensor! inj_out, Tensor! out, "
      "Tensor! workspace, int num_wg, float eps, float gn_eps, int gn_sigmoid, "
      "float scale, float timeout_ms) -> ()");
  rocm_ops.impl("mono_k2_rdna2_e1a", torch::kCUDA, &mono_k2_rdna2_e1a);
  rocm_ops.def(
      "mono_gdn_rdna2(Tensor qkvz, Tensor ba, Tensor conv_w, Tensor conv_b, "
      "Tensor! conv_state, Tensor! ssm_state, Tensor A_log, Tensor dt_bias, "
      "Tensor norm_w, Tensor sidx, Tensor nacc, Tensor qsl, Tensor! y, "
      "Tensor! scratch, int num_k_heads, float scale, float eps, "
      "int gate_sigmoid) -> ()");
  rocm_ops.impl("mono_gdn_rdna2", torch::kCUDA, &mono_gdn_rdna2);
  rocm_ops.def("mono_gdn_rdna2_scratch_bytes(int M, int HV, int H) -> int");
  rocm_ops.impl("mono_gdn_rdna2_scratch_bytes", &mono_gdn_rdna2_scratch_bytes);
  rocm_ops.def(
      "mono_qsa_decode_rdna2(Tensor q, Tensor k_cache, Tensor v_cache, Tensor "
      "indices, "
      "Tensor block_table, Tensor token_to_req, Tensor! out, Tensor! scratch, "
      "Tensor! counters, int splits) -> ()");
  rocm_ops.impl("mono_qsa_decode_rdna2", torch::kCUDA, &mono_qsa_decode_rdna2);
  rocm_ops.def("mono_qsa_decode_rdna2_splits(int M, int KVH, int W) -> int");
  rocm_ops.impl("mono_qsa_decode_rdna2_splits", &mono_qsa_decode_rdna2_splits);
  rocm_ops.def(
      "mono_qsa_store3_rdna2(Tensor! c_cache, Tensor c_slots, Tensor c_rows, "
      "Tensor! r_cache, Tensor r_slots, Tensor r_rows, Tensor! p_cache, "
      "Tensor p_rows) -> ()");
  rocm_ops.impl("mono_qsa_store3_rdna2", torch::kCUDA, &mono_qsa_store3_rdna2);
  rocm_ops.def(
      "mono_qsa_k_store_rdna2(Tensor pooled, Tensor weight, Tensor cos_sin_cache, "
      "Tensor positions, float eps, float norm_beta, int rotary_dim, int mrope_h, "
      "int mrope_w, Tensor! c_cache, Tensor c_slots, Tensor! r_cache, "
      "Tensor r_slots, Tensor r_rows, Tensor! p_cache, Tensor p_rows) -> ()");
  rocm_ops.impl("mono_qsa_k_store_rdna2", torch::kCUDA, &mono_qsa_k_store_rdna2);
  rocm_ops.def(
      "mono_qsa_mqa_rdna2(Tensor q, Tensor k_cache, Tensor page_table, "
      "Tensor token_to_req, Tensor query_positions, Tensor sequence_lengths, "
      "Tensor! logits, Tensor! visible_blocks, int compress_ratio, "
      "float score_divisor, int group, int waves, int tiles, "
      "int lanes_per_key, int unroll, bool q_lds) -> ()");
  rocm_ops.impl("mono_qsa_mqa_rdna2", torch::kCUDA, &mono_qsa_mqa_rdna2);
  rocm_ops.def(
      "mono_qsa_epilogue_rdna2(Tensor qk, Tensor! q_out, Tensor q_w, Tensor k_w, "
      "float q_eps, float k_eps, float norm_beta, Tensor cos_sin_cache, "
      "Tensor positions, int rotary_dim, int mrope_h, int mrope_w, "
      "Tensor state_table, Tensor token_to_req, Tensor query_start_loc, "
      "Tensor end_positions, int compress_ratio, Tensor! c_cache, "
      "Tensor c_slots, Tensor! r_cache, Tensor r_slots, Tensor! p_cache) -> ()");
  rocm_ops.impl("mono_qsa_epilogue_rdna2", torch::kCUDA, &mono_qsa_epilogue_rdna2);
  #endif
  rocm_ops.def(
      "rdna_se_gate_up_silu(Tensor x, Tensor w, Tensor? scale) -> Tensor");
  rocm_ops.impl("rdna_se_gate_up_silu", torch::kCUDA, &rdna_se_gate_up_silu);
  rocm_ops.def(
      "rdna_se_down_gated(Tensor act, Tensor w, Tensor? scale, Tensor x, "
      "Tensor w_gate) -> Tensor");
  rocm_ops.impl("rdna_se_down_gated", torch::kCUDA, &rdna_se_down_gated);

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
  // T44b abort record; leapdragon/vllm-rdna2-qwen (Aron Hsiao).
  rocm_ops.def("rdna_ar_timeout_info(int handle) -> int",
               &rdna_ar_timeout_info);
  rocm_ops.def("rdna_ar_slow_info(int handle) -> int", &rdna_ar_slow_info);
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

  // W4A8 (int4 weights, int8 activations) prefill GEMM for gfx1030. Opt-in
  // drop-in for the dense W4A16 prefill path; the gemm entry returns a
  // populated [M, N] fp16 tensor whether the W4A8 fast path fired or the
  // W4A16 prefill fallback ran, so the dispatcher never branches on
  // shape/LDS at runtime. Fake kernels live in _custom_ops.py (register_fake);
  // there are no C++ meta stubs.
  rocm_ops.def(
      "w4a8_act_quant_rdna2(Tensor x, int group_size, Tensor(a!) a_i8, "
      "Tensor(a!) a_scale, Tensor(a!) a_asum) -> Tensor");
  rocm_ops.impl("w4a8_act_quant_rdna2", torch::kCUDA, &w4a8_act_quant_rdna2);

  rocm_ops.def(
      "w4a8_gemm_rdna2(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("w4a8_gemm_rdna2", torch::kCUDA, &w4a8_gemm_rdna2);

  // FA-RDNA2: Flash-Attention v2 hand-port for AMD RDNA2 (gfx1030), used by
  // the RDNA_ATTN backend. The fa_rdna2_* ops write into `out`.
  rocm_ops.def(
      "fa_rdna2_decode_paged(Tensor Q, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, int block_size, int kv_splits, "
      "int sliding_window, float scale, Tensor(a!) out, "
      "Tensor? cu_query_lens=None) -> ()");
  rocm_ops.impl("fa_rdna2_decode_paged", torch::kCUDA, &fa_rdna2_decode_paged);

  // GDN packed single-token decode for AMD RDNA2 (gfx1030). Dispatched
  // from Qwen3NextGatedDeltaNet._forward_core_decode_non_spec on gfx10x.
  rocm_ops.def(
      "gdn_decode_rdna2(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor A_log, Tensor dt_bias, Tensor! out, Tensor! initial_state, "
      "Tensor ssm_state_indices, float scale, bool use_qk_l2norm) -> ()");
  rocm_ops.impl("gdn_decode_rdna2", torch::kCUDA, &gdn_decode_rdna2);

  // GDN speculative-decoding verify for AMD RDNA2 (gfx1030): the MTP-2
  // spec-verify step of fused_sigmoid_gating_delta_rule_update_kernel.
  rocm_ops.def(
      "gdn_spec_verify_rdna2(Tensor q, Tensor k, Tensor v, Tensor a, "
      "Tensor b, Tensor A_log, Tensor dt_bias, Tensor! out, "
      "Tensor! initial_state, Tensor cu_seqlens, Tensor ssm_state_indices, "
      "Tensor num_accepted_tokens, float scale) -> ()");
  rocm_ops.impl("gdn_spec_verify_rdna2", torch::kCUDA, &gdn_spec_verify_rdna2);

  // causal_conv1d_update single-token decode for AMD RDNA2 (gfx1030).
  // Replaces the Triton kernel that captured scratch buffer pointers and
  // page-faulted on cudagraph replay.
  rocm_ops.def(
      "causal_conv1d_update_rdna2(Tensor x, Tensor conv_state, "
      "Tensor weight, Tensor bias, Tensor out, "
      "Tensor conv_state_indices, bool silu_activation) -> ()");
  rocm_ops.impl("causal_conv1d_update_rdna2", torch::kCUDA,
                &causal_conv1d_update_rdna2);
  // causal_conv1d_update varlen / speculative-decoding for AMD RDNA2
  // (gfx1030). Replaces the IS_VARLEN / IS_SPEC_DECODING cases of the same
  // Triton kernel (MTP-2 passes num_accepted_tokens and query_start_loc).
  rocm_ops.def(
      "causal_conv1d_update_spec_rdna2(Tensor x, Tensor conv_state, "
      "Tensor weight, Tensor bias, Tensor out, Tensor conv_state_indices, "
      "Tensor num_accepted_tokens, Tensor query_start_loc, int state_len, "
      "int max_query_len, int null_block_id, bool silu_activation) -> ()");
  rocm_ops.impl("causal_conv1d_update_spec_rdna2", torch::kCUDA,
                &causal_conv1d_update_spec_rdna2);

  // causal_conv1d_fwd varlen prefill for AMD RDNA2 (gfx1030). Replaces the
  // _causal_conv1d_fwd_kernel Triton kernel that captured scratch buffer
  // pointers and page-faulted on cudagraph replay. cudagraph-safe (per-warp
  // scratch in shared memory, no global allocations, no Triton JIT scratch).
  rocm_ops.def(
      "causal_conv1d_fwd_rdna2(Tensor x, Tensor weight, Tensor bias, "
      "Tensor conv_state, Tensor query_start_loc, Tensor cache_indices, "
      "Tensor has_initial_state, Tensor out, bool silu_activation, "
      "int null_block_id=0) -> ()");
  rocm_ops.impl("causal_conv1d_fwd_rdna2", torch::kCUDA,
                &causal_conv1d_fwd_rdna2);

  rocm_ops.def(
      "mrope_forward_rdna2(Tensor q, Tensor k, Tensor cos, Tensor sin, "
      "int num_tokens, int n_qh, int n_kh, int hd, int rd, "
      "int sec_t, int sec_h, int sec_w, bool is_interleaved, "
      "bool is_neox_style) -> ()");
  rocm_ops.impl("mrope_forward_rdna2", torch::kCUDA, &mrope_forward_rdna2);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int sliding_window, "
      "float scale, Tensor(a!) out) -> ()");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen_short(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int sliding_window, "
      "float scale, Tensor(a!) out) -> ()");
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
  rocm_ops.impl("gdn_prefill_kkt_rdna2", torch::kCUDA, &gdn_prefill_kkt_rdna2);

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
  rocm_ops.impl("gdn_prefill_o_rdna2", torch::kCUDA, &gdn_prefill_o_rdna2);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen_splitk(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, int kv_splits, "
      "int sliding_window, float scale, Tensor(a!) out) -> ()");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen_splitk", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen_splitk);

  rocm_ops.def(
      "fa_rdna2_prefill_paged_varlen_gqa(Tensor Q, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor cu_query_lens, "
      "Tensor seq_lens, int block_size, int causal, "
      "int sliding_window, float scale, Tensor(a!) out) -> ()");
  rocm_ops.impl("fa_rdna2_prefill_paged_varlen_gqa", torch::kCUDA,
                &fa_rdna2_prefill_paged_varlen_gqa);

  rocm_ops.def(
      "moe_gptq_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk, bool fp32_accum=True, "
      "bool use_v2_format=False) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna2", torch::kCUDA, &moe_gptq_gemm_rdna2);

  // W4A8 (int4 weights, int8 activations) fused MoE for gfx1030. Drop-in for
  // moe_gptq_gemm_rdna2 (same args plus the trailing use_v2_format); owns the
  // shape/LDS eligibility and the internal W4A16 fallback, so the Python
  // forward never branches on an op result. Fake kernel lives in
  // _custom_ops.py (register_fake).
  rocm_ops.def(
      "moe_w4a8_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk=0, bool use_v2_format=False, "
      "bool fp32_accum=True) -> ()");
  rocm_ops.impl("moe_w4a8_gemm_rdna2", torch::kCUDA, &moe_w4a8_gemm_rdna2);

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
  rocm_ops.impl("gemm_w8a16_fp8_dense", torch::kCUDA, &gemm_w8a16_fp8_dense);

  // Paged MQA logits for DeepSeek V4 Lightning Indexer (gfx1030).
  // Replaces the AITER-only decode path of rocm_aiter_sparse_attn_indexer.
  rocm_ops.def(
      "paged_mqa_logits_decode_rdna2(Tensor q_fp8, Tensor kv_cache, "
      "Tensor weights, Tensor context_lens, Tensor block_tables, "
      "int max_model_len, bool block_flat=False) -> Tensor");
  rocm_ops.impl("paged_mqa_logits_decode_rdna2", torch::kCUDA,
                &paged_mqa_logits_decode_rdna2);
  // Ragged prefill MQA logits for the same indexer (fused, tiled).
  rocm_ops.def(
      "mqa_logits_prefill_rdna2(Tensor q, Tensor k, Tensor k_scale, "
      "Tensor weights, Tensor ks, Tensor ke) -> Tensor");
  rocm_ops.impl("mqa_logits_prefill_rdna2", torch::kCUDA,
                &mqa_logits_prefill_rdna2);

  // W8A8-FP8 dense linear kernel for RDNA2 (gfx1030). DeepSeek V4 Flash
  // attention / shared experts: FP8 weights + FP8 activations, per-tile
  // FP8->fp16 dequant (no LUT, inline bit-trick), then v_dot2_f32_f16.
  // a_scale: [1] / [M] / [M, K/gs] (per-block-K dynamic act quant).
  // a_scale_K_groups: number of K-blocks in a_scale (1 for per-row/tensor).
  rocm_ops.def(
      "gemm_w8a8_fp8_dense(Tensor a_q, Tensor a_scale, Tensor b_q_weight, "
      "Tensor b_scales, Tensor(a!) c, int group_size, int a_scale_K_groups) -> "
      "()");
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

  // DeepSeek-V4 q-norm + RoPE + fp8_ds_mla KV insert for RDNA
  // (csrc/rocm/rdna/dsv4/kv_insert.cu). RDNA counterpart of
  // _C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert that accepts fp16
  // as well as bf16 q/kv; writes the same V4 cache row (RoPE dims as bf16).
  rocm_ops.def(
      "dsv4_qnorm_rope_kv_insert_rdna(Tensor q_in, Tensor kv, "
      "Tensor(a!) k_cache, Tensor slot_mapping, Tensor position_ids, "
      "Tensor cos_sin_cache, int q_head_padded, float eps, "
      "int cache_block_size, bool apply_q_norm=True, "
      "bool apply_q_rope=True) -> Tensor");
  rocm_ops.impl("dsv4_qnorm_rope_kv_insert_rdna", torch::kCUDA,
                &dsv4_qnorm_rope_kv_insert_rdna);

  // DeepSeek-V4 mHC pre (everything after the x @ fn^T GEMM) and post for
  // RDNA (csrc/rocm/rdna/dsv4/mhc.cu); replaces the torch reference.
  rocm_ops.def(
      "dsv4_mhc_pre_rdna(Tensor residual, Tensor mixes, Tensor hc_scale, "
      "Tensor hc_base, float rms_eps, float pre_eps, float sinkhorn_eps, "
      "float post_mult, int sinkhorn_repeat) -> Tensor[]");
  rocm_ops.impl("dsv4_mhc_pre_rdna", torch::kCUDA, &dsv4_mhc_pre_rdna);
  rocm_ops.def(
      "dsv4_mhc_post_rdna(Tensor x, Tensor residual, Tensor post, "
      "Tensor comb) -> Tensor");
  rocm_ops.impl("dsv4_mhc_post_rdna", torch::kCUDA, &dsv4_mhc_post_rdna);

  // INT8 per-(token, head) KV-cache writer for RDNA2 (gfx1030).
  // Quantizes fp16 K/V to int8 with per-(token, head) scales and writes
  // them into the interleaved cache layout the RDNA2 FA decode kernel
  // reads (D bytes data + 4 bytes scale per slot, per kv-int8.md wiki
  // contract). Not wired up yet: RDNA_ATTN still rejects quantized KV
  // caches, and fa_rdna2_decode_paged_int8 is not registered.
  rocm_ops.def(
      "reshape_and_cache_int8_rdna2(Tensor key, Tensor value, "
      "Tensor(a!) kv_cache, Tensor slot_mapping) -> ()");
  rocm_ops.impl("reshape_and_cache_int8_rdna2", torch::kCUDA,
                &reshape_and_cache_int8_rdna2);

  // fp16 flash KV writer (5D packed K / 4D unpacked V). Stride-aware
  // for hybrid GDN pages. Replaces Triton reshape_and_cache_flash so
  // FA-RDNA2 KV updates can live inside a FULL CUDA graph.
  rocm_ops.def(
      "reshape_and_cache_flash_rdna2(Tensor key, Tensor value, "
      "Tensor(a!) key_cache, Tensor(a!) value_cache, "
      "Tensor slot_mapping) -> ()",
      {at::Tag::needs_exact_strides});
  rocm_ops.impl("reshape_and_cache_flash_rdna2", torch::kCUDA,
                &reshape_and_cache_flash_rdna2);

  // Qwen4Exp HC prefill HIP (opt-in: VLLM_RDNA_HC_PREFILL_HIP=1).
  // Replaces the Triton _grouped_gemma_rmsnorm_kernel / _hc_silu_kernel /
  // _hc_gate_mix_kernel / _hc_combine_kernel / _hc_combine_norm_kernel
  // on gfx1030. Faithful ports: y = x * rsqrt(ss/N + eps) * (1+w), etc.
  rocm_ops.def(
      "hc_grouped_gemma_rmsnorm_rdna2(Tensor x, Tensor weight, Tensor(a!) y, "
      "int num_groups, float eps) -> ()");
  rocm_ops.impl("hc_grouped_gemma_rmsnorm_rdna2", torch::kCUDA,
                &hc_grouped_gemma_rmsnorm_rdna2);

  rocm_ops.def("hc_silu_rdna2(Tensor x, Tensor(a!) y, int hc_count) -> ()");
  rocm_ops.impl("hc_silu_rdna2", torch::kCUDA, &hc_silu_rdna2);

  rocm_ops.def(
      "hc_gate_mix_rdna2(Tensor x, Tensor gate, Tensor(a!) y, "
      "int hc_count) -> ()");
  rocm_ops.impl("hc_gate_mix_rdna2", torch::kCUDA, &hc_gate_mix_rdna2);

  rocm_ops.def(
      "hc_combine_rdna2(Tensor residual, Tensor block_output, "
      "Tensor injection_logits, Tensor(a!) out, int hc_count) -> ()");
  rocm_ops.impl("hc_combine_rdna2", torch::kCUDA, &hc_combine_rdna2);

  rocm_ops.def(
      "hc_combine_norm_rdna2(Tensor residual, Tensor block_output, "
      "Tensor injection_logits, Tensor norm_weight, Tensor(a!) out, "
      "Tensor(a!) y, int hc_count, float eps) -> ()");
  rocm_ops.impl("hc_combine_norm_rdna2", torch::kCUDA, &hc_combine_norm_rdna2);

  // Qwen4Exp QSA decode HIP (opt-in: VLLM_RDNA_QSA_HIP=1).
  rocm_ops.def(
      "qsa_store_cache_rows_rdna2(Tensor rows, Tensor slots, "
      "Tensor(a!) cache, Tensor page_size, Tensor width) -> ()");
  rocm_ops.impl("qsa_store_cache_rows_rdna2", torch::kCUDA,
                &qsa_store_cache_rows_rdna2);

  rocm_ops.def(
      "qsa_compress_groups_rdna2(Tensor raw_keys, Tensor raw_positions, "
      "Tensor compressor_state_cache, Tensor rope_cache, "
      "Tensor compressor_state_table, Tensor token_to_req, "
      "Tensor query_start_loc, Tensor logical_positions, "
      "Tensor compressed_slots, Tensor(a!) pooled, Tensor(a!) first_positions, "
      "Tensor compress_ratio, Tensor compressor_state_size, Tensor head_dim, "
      "bool load_rope_positions) -> ()");
  rocm_ops.impl("qsa_compress_groups_rdna2", torch::kCUDA,
                &qsa_compress_groups_rdna2);

  rocm_ops.def(
      "qsa_mqa_paged_rdna2(Tensor q, Tensor k_cache, Tensor page_table, "
      "Tensor token_to_req, Tensor query_positions, Tensor sequence_lengths, "
      "int compress_ratio, int num_columns, float score_scale) "
      "-> (Tensor, Tensor)");
  rocm_ops.impl("qsa_mqa_paged_rdna2", torch::kCUDA, &qsa_mqa_paged_rdna2);

  rocm_ops.def(
      "qsa_expand_indices_rdna2(Tensor block_indices, Tensor query_positions, "
      "Tensor sequence_lengths, Tensor token_to_req, int compress_ratio, "
      "int token_topk, Tensor? out) -> Tensor");
  rocm_ops.impl("qsa_expand_indices_rdna2", torch::kCUDA,
                &qsa_expand_indices_rdna2);

  rocm_ops.def(
      "qsa_sparse_paged_attention_rdna2(Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor logical_indices, Tensor block_table, "
      "Tensor token_to_req, Tensor? out) -> Tensor");
  rocm_ops.impl("qsa_sparse_paged_attention_rdna2", torch::kCUDA,
                &qsa_sparse_paged_attention_rdna2);

  // QSA metadata builder (HIP port of common/qsa_cache.py's
  // _build_qsa_metadata_kernel); runs on gfx10x instead of the Triton builder.
  rocm_ops.def(
      "build_qsa_metadata_rdna2(Tensor query_start_loc, Tensor seq_lens, "
      "Tensor common_slot_mapping, Tensor block_table, "
      "Tensor(a!) token_to_req, Tensor(b!) logical_positions, "
      "Tensor(c!) visible_blocks, Tensor(d!) slot_mapping, "
      "Tensor? k_work_metadata, int num_reqs, int num_mapped_tokens, "
      "int num_tokens, int max_num_work, int num_search_steps, "
      "int work_search_steps, int request_scan_size, int storage_block_size, "
      "int compress_ratio, int circular_buffer_size) -> ()");
  rocm_ops.impl("build_qsa_metadata_rdna2", torch::kCUDA,
                &build_qsa_metadata_rdna2);

  // Vision-tower bilinear position-embedding interpolation (HIP port of
  // qwen3_vl.py::_bilinear_pos_embed_kernel); runs on gfx10x instead of the
  // Triton kernel.
  rocm_ops.def(
      "bilinear_pos_embed_rdna2(Tensor embed_weight, Tensor(a!) out, int H, "
      "int W, float h_scale, float w_scale, int num_grid, int m_size) "
      "-> Tensor");
  rocm_ops.impl("bilinear_pos_embed_rdna2", torch::kCUDA,
                &bilinear_pos_embed_rdna2);

  // Qwen4Exp PLE dilated short-conv HIP (opt-in: VLLM_RDNA_PLE_CONV_HIP=1).
  rocm_ops.def(
      "ple_short_conv_decode_rdna2(Tensor x, Tensor(a!) conv_state, "
      "Tensor weight, Tensor? bias, Tensor(a!) out, Tensor state_idx, "
      "Tensor? has_init, int dilation, int state_len, bool silu, "
      "int null_block) -> ()");
  rocm_ops.impl("ple_short_conv_decode_rdna2", torch::kCUDA,
                &ple_short_conv_decode_rdna2);

  rocm_ops.def(
      "ple_short_conv_prefill_rdna2(Tensor x_packed, Tensor init_state, "
      "Tensor weight, Tensor? bias, Tensor(a!) out, Tensor lengths, "
      "Tensor? valid_state, int dilation, int state_len, bool silu) -> ()");
  rocm_ops.impl("ple_short_conv_prefill_rdna2", torch::kCUDA,
                &ple_short_conv_prefill_rdna2);

  // v1 worker per-step index/arithmetic kernels (HIP ports of
  // _gather_block_tables_kernel, _compute_slot_mappings_kernel,
  // _apply_write_kernel x2, _zero_kv_blocks_kernel).
  rocm_ops.def(
      "gather_block_tables_rdna2(Tensor batch_idx_to_req_idx, "
      "Tensor src_block_table_ptrs, Tensor dst_block_table_ptrs, "
      "Tensor block_table_strides, Tensor num_blocks_ptr, "
      "int num_blocks_stride, int num_reqs, int num_reqs_padded) -> ()");
  rocm_ops.impl("gather_block_tables_rdna2", torch::kCUDA,
                &gather_block_tables_rdna2);

  rocm_ops.def(
      "compute_slot_mappings_rdna2(int max_num_tokens, Tensor idx_mapping, "
      "Tensor query_start_loc, Tensor pos, Tensor block_table_ptrs, "
      "Tensor block_table_strides, Tensor block_sizes, "
      "Tensor kernel_block_sizes, Tensor slot_mapping_enabled, "
      "Tensor dcp_sharded, Tensor(a!) slot_mappings, "
      "int slot_mappings_stride, int cp_rank, int cp_size, int cp_interleave, "
      "int pad_id) -> ()");
  rocm_ops.impl("compute_slot_mappings_rdna2", torch::kCUDA,
                &compute_slot_mappings_rdna2);

  rocm_ops.def(
      "apply_write_rdna2(Tensor(a!) out, int out_stride, "
      "Tensor write_indices, Tensor write_starts, Tensor write_contents, "
      "Tensor write_cu_lens) -> ()");
  rocm_ops.impl("apply_write_rdna2", torch::kCUDA, &apply_write_rdna2);

  rocm_ops.def(
      "fused_apply_write_rdna2(Tensor output_ptrs, Tensor output_strides, "
      "Tensor write_indices, Tensor write_starts, Tensor write_contents, "
      "Tensor write_cu_lens, Tensor write_group_ids) -> ()");
  rocm_ops.impl("fused_apply_write_rdna2", torch::kCUDA,
                &fused_apply_write_rdna2);

  rocm_ops.def(
      "zero_kv_blocks_rdna2(Tensor seg_addrs, Tensor seg_block_strides, "
      "Tensor seg_page_sizes, Tensor block_ids, int chunk_size, "
      "int max_chunks) -> ()");
  rocm_ops.impl("zero_kv_blocks_rdna2", torch::kCUDA, &zero_kv_blocks_rdna2);

  // Unquantized (fp16) MoE for gfx10x: the two fused_moe_kernel stages of the
  // unquantized path, without token sorting or an activation/moe_sum kernel.
  rocm_ops.def(
      "moe_gate_up_silu_unquantized_rdna2(Tensor x, Tensor w13, "
      "Tensor topk_ids, Tensor? expert_map, Tensor(a!) act, int topk, int N, "
      "int K) -> ()");
  rocm_ops.impl("moe_gate_up_silu_unquantized_rdna2", torch::kCUDA,
                &moe_gate_up_silu_unquantized_rdna2);

  rocm_ops.def(
      "moe_down_unquantized_rdna2(Tensor act, Tensor w2, Tensor topk_ids, "
      "Tensor? expert_map, Tensor topk_weights, bool mul_topk_weight, "
      "Tensor(b!) out, int topk, int N, int K) -> ()");
  rocm_ops.impl("moe_down_unquantized_rdna2", torch::kCUDA,
                &moe_down_unquantized_rdna2);

  // v1 worker input-batch kernels (HIP ports of the input_batch.py Triton
  // kernels).
  rocm_ops.def(
      "prepare_prefill_inputs_rdna2(Tensor(a!) input_ids, "
      "Tensor(b!) next_prefill_tokens, int next_prefill_tokens_stride, "
      "int num_lookahead, Tensor idx_mapping, Tensor query_start_loc, "
      "Tensor all_token_ids, int all_token_ids_stride, Tensor prefill_len, "
      "Tensor num_computed_tokens) -> ()");
  rocm_ops.impl("prepare_prefill_inputs_rdna2", torch::kCUDA,
                &prepare_prefill_inputs_rdna2);

  rocm_ops.def(
      "prepare_pos_seq_lens_rdna2(Tensor(a!) pos, Tensor(b!) seq_lens, "
      "Tensor idx_mapping, Tensor query_start_loc, "
      "Tensor num_computed_tokens, int max_num_reqs) -> ()");
  rocm_ops.impl("prepare_pos_seq_lens_rdna2", torch::kCUDA,
                &prepare_pos_seq_lens_rdna2);

  rocm_ops.def(
      "combine_sampled_and_draft_tokens_rdna2(Tensor(a!) input_ids, "
      "Tensor idx_mapping, Tensor last_sampled_tokens, "
      "Tensor query_start_loc, Tensor seq_lens, Tensor prefill_len, "
      "Tensor draft_tokens, int draft_tokens_stride, Tensor cu_num_logits, "
      "Tensor(b!) logits_indices, int num_new_sampled_tokens) -> ()");
  rocm_ops.impl("combine_sampled_and_draft_tokens_rdna2", torch::kCUDA,
                &combine_sampled_and_draft_tokens_rdna2);

  rocm_ops.def(
      "get_num_sampled_and_rejected_rdna2(Tensor(a!) num_sampled, "
      "Tensor(b!) num_rejected, Tensor seq_lens, Tensor cu_num_logits, "
      "Tensor idx_mapping, Tensor prefill_len) -> ()");
  rocm_ops.impl("get_num_sampled_and_rejected_rdna2", torch::kCUDA,
                &get_num_sampled_and_rejected_rdna2);

  rocm_ops.def(
      "post_update_rdna2(Tensor idx_mapping, Tensor(a!) num_computed_tokens, "
      "Tensor(b!) last_sampled_tokens, Tensor? output_bin_counts, "
      "int output_bin_counts_stride, Tensor sampled_tokens, "
      "int sampled_tokens_stride, Tensor num_sampled, Tensor num_rejected, "
      "Tensor? query_start_loc, Tensor(c!) all_token_ids, "
      "int all_token_ids_stride, Tensor(d!) total_len) -> ()");
  rocm_ops.impl("post_update_rdna2", torch::kCUDA, &post_update_rdna2);

  rocm_ops.def(
      "expand_idx_mapping_rdna2(Tensor idx_mapping, "
      "Tensor(a!) expanded_idx_mapping, Tensor(b!) expanded_local_pos, "
      "Tensor cu_num_logits) -> ()");
  rocm_ops.impl("expand_idx_mapping_rdna2", torch::kCUDA,
                &expand_idx_mapping_rdna2);

  // Speculative-decoding kernels (HIP ports of spec_decode/autoregressive/
  // speculator.py, spec_decode/rejection_sampler.py and
  // spec_decode/rejection_sampler_utils.py Triton kernels).
  rocm_ops.def(
      "spec_prepare_prefill_inputs_rdna2(Tensor(a!) last_token_indices, "
      "Tensor(b!) draft_current_step, Tensor(c!) draft_input_ids, "
      "Tensor(d!) draft_positions, Tensor(e!) draft_query_start_loc, "
      "Tensor(f!) draft_seq_lens, Tensor target_input_ids, "
      "Tensor target_positions, Tensor idx_mapping, Tensor last_sampled, "
      "Tensor next_prefill_tokens, Tensor num_sampled, Tensor num_rejected, "
      "Tensor query_start_loc, Tensor seq_lens, int max_num_reqs) -> ()");
  rocm_ops.impl("spec_prepare_prefill_inputs_rdna2", torch::kCUDA,
                &spec_prepare_prefill_inputs_rdna2);

  rocm_ops.def(
      "spec_prepare_decode_inputs_rdna2(Tensor draft_tokens, "
      "Tensor target_seq_lens, Tensor num_rejected, Tensor(a!) input_ids, "
      "Tensor(b!) positions, Tensor(c!) sample_src_positions, "
      "Tensor(d!) query_start_loc, Tensor(e!) seq_lens, int max_model_len, "
      "int max_num_reqs, bool advance_draft_positions) -> ()");
  rocm_ops.impl("spec_prepare_decode_inputs_rdna2", torch::kCUDA,
                &spec_prepare_decode_inputs_rdna2);

  rocm_ops.def(
      "spec_update_draft_inputs_rdna2(Tensor(a!) output_draft_tokens, "
      "Tensor(b!) next_input_hidden_states, Tensor(c!) input_ids, "
      "Tensor(d!) positions, Tensor(e!) sample_src_positions, "
      "Tensor(f!) seq_lens, Tensor draft_tokens, Tensor current_draft_step, "
      "Tensor hidden_states, int hidden_size, int max_model_len, "
      "int num_speculative_steps, bool advance_draft_positions) -> ()");
  rocm_ops.impl("spec_update_draft_inputs_rdna2", torch::kCUDA,
                &spec_update_draft_inputs_rdna2);

  rocm_ops.def(
      "spec_gather_draft_sampled_rdna2(Tensor(a!) draft_sampled, "
      "Tensor(b!) pos, Tensor input_ids, Tensor positions, "
      "Tensor logits_indices, Tensor expanded_idx_mapping, "
      "Tensor expanded_local_pos, Tensor prefill_len) -> ()");
  rocm_ops.impl("spec_gather_draft_sampled_rdna2", torch::kCUDA,
                &spec_gather_draft_sampled_rdna2);

  rocm_ops.def(
      "spec_flatten_sampled_rdna2(Tensor(a!) flat_sampled, Tensor sampled, "
      "Tensor num_sampled, Tensor cu_num_logits) -> ()");
  rocm_ops.impl("spec_flatten_sampled_rdna2", torch::kCUDA,
                &spec_flatten_sampled_rdna2);

  rocm_ops.def(
      "spec_compute_local_logits_stats_rdna2(Tensor(a!) target_local_argmax, "
      "Tensor(b!) target_local_max, Tensor(c!) target_local_sumexp, "
      "Tensor(d!) draft_local_max, Tensor(e!) draft_local_sumexp, "
      "Tensor target_logits, Tensor? draft_logits, "
      "Tensor expanded_idx_mapping, Tensor expanded_local_pos, Tensor temp, "
      "int vocab_size, int num_speculative_steps, "
      "bool has_draft_logits) -> ()");
  rocm_ops.impl("spec_compute_local_logits_stats_rdna2", torch::kCUDA,
                &spec_compute_local_logits_stats_rdna2);

  rocm_ops.def(
      "spec_rejection_rdna2(Tensor(a!) sampled, Tensor(b!) rejected_steps, "
      "Tensor(c!) target_rejected_logsumexp, "
      "Tensor(d!) draft_rejected_logsumexp, Tensor target_logits, "
      "Tensor target_local_argmax, Tensor target_local_max, "
      "Tensor target_local_sumexp, Tensor draft_sampled, "
      "Tensor? draft_logits, Tensor draft_local_max, "
      "Tensor draft_local_sumexp, Tensor cu_num_logits, Tensor idx_mapping, "
      "Tensor temperature, Tensor seed, Tensor pos, "
      "Tensor? synthetic_conditional_rates, Tensor? cumulative_log_p, "
      "Tensor? local_residual_mass, int vocab_num_blocks, "
      "int padded_vocab_num_blocks, bool has_draft_logits, "
      "bool synthetic_mode, bool use_block_verification) -> ()");
  rocm_ops.impl("spec_rejection_rdna2", torch::kCUDA,
                &spec_rejection_rdna2);

  rocm_ops.def(
      "spec_resample_rdna2(Tensor(a!) resampled_local_argmax, "
      "Tensor(b!) resampled_local_max, Tensor target_logits, "
      "Tensor target_rejected_logsumexp, Tensor? draft_logits, "
      "Tensor draft_rejected_logsumexp, Tensor rejected_step, "
      "Tensor cu_num_logits, Tensor expanded_idx_mapping, "
      "Tensor draft_sampled, Tensor temperature, Tensor seed, Tensor pos, "
      "Tensor? cumulative_log_p, int vocab_size, bool has_draft_logits, "
      "bool use_fp64, bool use_block_verification) -> ()");
  rocm_ops.impl("spec_resample_rdna2", torch::kCUDA, &spec_resample_rdna2);

  rocm_ops.def(
      "spec_insert_resampled_rdna2(Tensor(a!) sampled, "
      "Tensor(b!) num_sampled, Tensor resampled_local_argmax, "
      "Tensor resampled_local_max, int resample_num_blocks, "
      "Tensor cu_num_logits, Tensor expanded_idx_mapping, "
      "Tensor temperature) -> ()");
  rocm_ops.impl("spec_insert_resampled_rdna2", torch::kCUDA,
                &spec_insert_resampled_rdna2);

  // v1 sampler kernels (HIP ports of the vllm/v1/worker/gpu/sample/ Triton
  // kernels: gumbel, penalties, min_p, logit_bias, bad_words, logprob and
  // prompt_logprob).
  rocm_ops.def(
      "temperature_rdna2(Tensor(a!) logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor temperature, int vocab_size) "
      "-> ()");
  rocm_ops.impl("temperature_rdna2", torch::kCUDA, &temperature_rdna2);

  rocm_ops.def(
      "gumbel_sample_rdna2(Tensor(a!) local_argmax, int local_argmax_stride, "
      "Tensor(b!) local_max, int local_max_stride, Tensor? logits_cache, "
      "int logits_cache_stride_0, int logits_cache_stride_1, "
      "Tensor? logits_cache_col, Tensor logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor seeds, Tensor pos, "
      "Tensor temperature, int vocab_size, bool is_drafting, "
      "bool apply_temperature, bool use_fp64, bool per_token_col) -> ()");
  rocm_ops.impl("gumbel_sample_rdna2", torch::kCUDA, &gumbel_sample_rdna2);

  rocm_ops.def(
      "penalties_rdna2(Tensor(a!) logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor token_ids, "
      "Tensor expanded_local_pos, Tensor repetition_penalty, "
      "Tensor frequency_penalty, Tensor presence_penalty, "
      "Tensor prompt_bin_mask, int prompt_bin_mask_stride, "
      "Tensor output_bin_counts, int output_bin_counts_stride, "
      "int vocab_size) -> ()");
  rocm_ops.impl("penalties_rdna2", torch::kCUDA, &penalties_rdna2);

  rocm_ops.def(
      "bincount_rdna2(Tensor expanded_idx_mapping, Tensor all_token_ids, "
      "int all_token_ids_stride, Tensor prompt_len, Tensor prefill_len, "
      "Tensor(a!) prompt_bin_mask, int prompt_bin_mask_stride, "
      "Tensor(b!) output_bin_counts, int output_bin_counts_stride, "
      "int max_prefill_len) -> ()");
  rocm_ops.impl("bincount_rdna2", torch::kCUDA, &bincount_rdna2);

  rocm_ops.def(
      "min_p_rdna2(Tensor(a!) logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor min_p, int vocab_size) -> ()");
  rocm_ops.impl("min_p_rdna2", torch::kCUDA, &min_p_rdna2);

  // Combined top-k/top-p masking (HIP port of the topk_topp_triton kernels).
  rocm_ops.def(
      "topk_topp_rdna2(Tensor(a!) logits, int logits_stride, Tensor? k, "
      "Tensor? p, float mask_value) -> ()");
  rocm_ops.impl("topk_topp_rdna2", torch::kCUDA, &topk_topp_rdna2);

  rocm_ops.def(
      "logit_bias_rdna2(Tensor(a!) logits, int logits_stride, int vocab_size, "
      "Tensor expanded_idx_mapping, Tensor num_allowed, Tensor allowed_ids, "
      "int allowed_ids_stride, Tensor num_bias, Tensor bias_ids, "
      "int bias_ids_stride, Tensor bias, int bias_stride, Tensor pos, "
      "Tensor min_lens, Tensor num_stop, Tensor restore_when_all_masked, "
      "Tensor stop_ids, int stop_ids_stride, int block_size, "
      "bool check_all_masked) -> ()");
  rocm_ops.impl("logit_bias_rdna2", torch::kCUDA, &logit_bias_rdna2);

  rocm_ops.def(
      "bad_words_rdna2(Tensor(a!) logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor bad_word_token_ids, "
      "int bad_word_token_ids_stride, Tensor bad_word_offsets, "
      "int bad_word_offsets_stride, Tensor num_bad_words, "
      "Tensor all_token_ids, int all_token_ids_stride, Tensor prompt_len, "
      "Tensor total_len, Tensor input_ids, Tensor expanded_local_pos, "
      "int max_num_bad_words) -> ()");
  rocm_ops.impl("bad_words_rdna2", torch::kCUDA, &bad_words_rdna2);

  rocm_ops.def(
      "topk_log_softmax_rdna2(Tensor(a!) output, Tensor logits, "
      "int logits_stride, Tensor topk_ids, int topk_ids_stride, int topk, "
      "int vocab_size, int topk_block_size) -> ()");
  rocm_ops.impl("topk_log_softmax_rdna2", torch::kCUDA, &topk_log_softmax_rdna2);

  rocm_ops.def(
      "token_ranks_rdna2(Tensor(a!) token_ranks, Tensor logits, "
      "int logits_stride, Tensor sampled_token_ids, int vocab_size) -> ()");
  rocm_ops.impl("token_ranks_rdna2", torch::kCUDA, &token_ranks_rdna2);

  rocm_ops.def(
      "prompt_logprobs_token_ids_rdna2(Tensor(a!) prompt_logprobs_token_ids, "
      "Tensor query_start_loc, Tensor idx_mapping, "
      "Tensor num_computed_tokens, Tensor all_token_ids, "
      "int all_token_ids_stride) -> ()");
  rocm_ops.impl("prompt_logprobs_token_ids_rdna2", torch::kCUDA,
                &prompt_logprobs_token_ids_rdna2);

  // v1 worker Mamba + M-RoPE + structured-output kernels (HIP ports of
  // model_states/mamba_hybrid.py, mamba_utils.py, gpu/mm/rope.py and
  // gpu/structured_outputs.py Triton kernels).
  rocm_ops.def(
      "preprocess_mamba_align_fused_rdna2(Tensor idx_mapping, "
      "Tensor(a!) state_idx, Tensor num_computed_tokens, "
      "Tensor query_start_loc, Tensor(b!) num_accepted_tokens, "
      "Tensor(c!) src_col, Tensor(d!) src_off, int num_reqs, "
      "int mamba_block_size) -> ()");
  rocm_ops.impl("preprocess_mamba_align_fused_rdna2", torch::kCUDA,
                &preprocess_mamba_align_fused_rdna2);

  rocm_ops.def(
      "scatter_num_accepted_rdna2(Tensor idx_mapping, Tensor num_sampled, "
      "Tensor(a!) num_accepted, int num_reqs) -> ()");
  rocm_ops.impl("scatter_num_accepted_rdna2", torch::kCUDA,
                &scatter_num_accepted_rdna2);

  rocm_ops.def(
      "fill_num_accepted_rdna2(Tensor idx_mapping, Tensor(a!) num_accepted, "
      "int value, int num_reqs) -> ()");
  rocm_ops.impl("fill_num_accepted_rdna2", torch::kCUDA,
                &fill_num_accepted_rdna2);

  rocm_ops.def(
      "precopy_mamba_align_fused_rdna2(Tensor mamba_state_idx, "
      "Tensor src_col, Tensor token_bias, Tensor block_table_ptrs, "
      "int block_table_stride_req, Tensor state_base_addrs, "
      "Tensor state_block_strides, Tensor state_elem_sizes, "
      "Tensor state_inner_sizes, Tensor state_conv_widths, "
      "Tensor state_group_indices, Tensor state_dim_row_count, "
      "Tensor state_dim_row_stride, Tensor? idx_mapping, int num_reqs, "
      "bool conv_state_dim_first, int temporal_tiles) -> ()");
  rocm_ops.impl("precopy_mamba_align_fused_rdna2", torch::kCUDA,
                &precopy_mamba_align_fused_rdna2);

  rocm_ops.def(
      "postprocess_mamba_fused_rdna2(Tensor num_accepted_tokens, "
      "Tensor mamba_state_idx, Tensor? num_scheduled_tokens, "
      "Tensor num_computed_tokens, Tensor? num_draft_tokens, "
      "Tensor block_table_ptrs, int block_table_stride_req, "
      "Tensor state_base_addrs, Tensor state_block_strides, "
      "Tensor state_elem_sizes, Tensor state_inner_sizes, "
      "Tensor state_conv_widths, Tensor state_group_indices, "
      "Tensor state_dim_row_count, Tensor state_dim_row_stride, "
      "Tensor(a!) num_accepted_tokens_out, Tensor? idx_mapping, "
      "int num_reqs, int block_size, bool conv_state_dim_first, "
      "bool precomputed_new_computed, int temporal_tiles) -> ()");
  rocm_ops.impl("postprocess_mamba_fused_rdna2", torch::kCUDA,
                &postprocess_mamba_fused_rdna2);

  rocm_ops.def(
      "batch_memcpy_rdna2(Tensor src_ptrs, Tensor dst_ptrs, Tensor sizes) "
      "-> ()");
  rocm_ops.impl("batch_memcpy_rdna2", torch::kCUDA, &batch_memcpy_rdna2);

  rocm_ops.def(
      "prepare_rope_positions_rdna2(Tensor(a!) positions, "
      "int positions_stride, Tensor prefill_positions, "
      "int prefill_stride0, int prefill_stride1, Tensor prefill_delta, "
      "Tensor idx_mapping, Tensor query_start_loc, Tensor prefill_lens, "
      "Tensor num_computed_tokens, int num_dims) -> ()");
  rocm_ops.impl("prepare_rope_positions_rdna2", torch::kCUDA,
                &prepare_rope_positions_rdna2);

  rocm_ops.def(
      "apply_grammar_bitmask_rdna2(Tensor(a!) logits, int logits_stride, "
      "Tensor logits_indices, Tensor cu_num_logits, Tensor bitmask, "
      "int bitmask_stride, int vocab_size, int mask_stride) -> ()");
  rocm_ops.impl("apply_grammar_bitmask_rdna2", torch::kCUDA,
                &apply_grammar_bitmask_rdna2);
  // Glm5Next post-conv KDA recurrence. Called from the plain decode path
  // when the op is present. fp16 or bf16 activations, fp32 state [V, K].
  rocm_ops.def(
      "glm5_kda_recurrent_rdna2(Tensor q, Tensor k, Tensor v, Tensor g, "
      "Tensor beta, Tensor A_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor! ssm_state, Tensor! out, float lower_bound) -> ()");
  rocm_ops.impl("glm5_kda_recurrent_rdna2", torch::kCUDA,
                &glm5_kda_recurrent_rdna2);

  // Ported kernels. Not on the upstream forward path. See ops.h.
  rocm_ops.def(
      "glm5_dsa_indexer_rdna2(Tensor q_idx, Tensor packed, Tensor weights, "
      "Tensor kv_lens, Tensor ape, Tensor! pool_indices_out, "
      "Tensor! pool_valid_out, Tensor! scores_out, int kpool) -> ()");
  rocm_ops.impl("glm5_dsa_indexer_rdna2", torch::kCUDA,
                &glm5_dsa_indexer_rdna2);
  rocm_ops.def(
      "glm5_dsa_mla_decode_rdna2(Tensor q_nope, Tensor k_sel, Tensor v_sel, "
      "Tensor sel_valid, Tensor! out, float scale) -> ()");
  rocm_ops.impl("glm5_dsa_mla_decode_rdna2", torch::kCUDA,
                &glm5_dsa_mla_decode_rdna2);
  rocm_ops.def(
      "glm5_kda_decode_rdna2(Tensor qkv_raw, Tensor conv_w, "
      "Tensor! conv_state, Tensor A_log, Tensor dt_bias, Tensor f, "
      "Tensor beta, Tensor out_gate, Tensor norm_w, Tensor state_indices, "
      "Tensor! ssm_state, Tensor! out, float lower_bound, "
      "float norm_eps) -> ()");
  rocm_ops.impl("glm5_kda_decode_rdna2", torch::kCUDA, &glm5_kda_decode_rdna2);
  rocm_ops.def(
      "glm5_kda_prefill_prep_rdna2(Tensor mixed_qkv, Tensor conv_w, "
      "Tensor A_log, Tensor dt_bias, Tensor f, Tensor beta_raw, "
      "Tensor! q_out, Tensor! k_out, Tensor! v_out, Tensor! g_out, "
      "Tensor! beta_out, Tensor cu_seqlens, Tensor chunk_indices, "
      "float lower_bound) -> ()");
  rocm_ops.impl("glm5_kda_prefill_prep_rdna2", torch::kCUDA,
                &glm5_kda_prefill_prep_rdna2);
  rocm_ops.def(
      "glm5_kda_prefill_kkt_rdna2(Tensor k, Tensor beta, Tensor g, "
      "Tensor! A, Tensor cu_seqlens, Tensor chunk_indices) -> ()");
  rocm_ops.impl("glm5_kda_prefill_kkt_rdna2", torch::kCUDA,
                &glm5_kda_prefill_kkt_rdna2);
  rocm_ops.def(
      "glm5_kda_prefill_solve_wy_rdna2(Tensor A, Tensor k, Tensor v, "
      "Tensor beta, Tensor g, Tensor! A_inv, Tensor! w, Tensor! u, "
      "Tensor cu_seqlens, Tensor chunk_indices) -> ()");
  rocm_ops.impl("glm5_kda_prefill_solve_wy_rdna2", torch::kCUDA,
                &glm5_kda_prefill_solve_wy_rdna2);
  rocm_ops.def(
      "glm5_kda_prefill_delta_h_rdna2(Tensor k, Tensor u, Tensor w, "
      "Tensor g, Tensor! h, Tensor! v_new, Tensor? initial_state, "
      "Tensor? final_state, Tensor? cu_seqlens, Tensor? chunk_offsets, "
      "int chunk_size) -> ()");
  rocm_ops.impl("glm5_kda_prefill_delta_h_rdna2", torch::kCUDA,
                &glm5_kda_prefill_delta_h_rdna2);
  rocm_ops.def(
      "glm5_kda_prefill_o_rdna2(Tensor q, Tensor k, Tensor v_new, "
      "Tensor h, Tensor g, Tensor! o, float scale, Tensor cu_seqlens, "
      "Tensor chunk_offsets) -> ()");
  rocm_ops.impl("glm5_kda_prefill_o_rdna2", torch::kCUDA,
                &glm5_kda_prefill_o_rdna2);
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

  // Decode linear: K-Hadamard + trellis GEMM + N-Hadamard, one dispatch.
  rocm_ops.def(
      "exl3_project_rdna2(Tensor x, Tensor(a!) xh, Tensor(a!) mid, "
      "Tensor(a!) out, Tensor trellis, Tensor suh, Tensor svh, "
      "int bits, int cb) -> ()");
  rocm_ops.impl("exl3_project_rdna2", torch::kCUDA, &exl3_project_rdna2);

  // EXL3 Hadamard-128 (suh/svh): y = H_128(x) * (scale/sqrt(128)), outside
  // the K-dot. Port of exllamav3_ext.had_r_128.
  rocm_ops.def(
      "exl3_hadamard_128(Tensor input, Tensor! output, "
      "Tensor? pre_scale, Tensor? post_scale, float scale) -> ()");
  rocm_ops.impl("exl3_hadamard_128", torch::kCUDA, &exl3_hadamard_128);

  rocm_ops.def("exl3_dequant_bits6_mul1(Tensor trellis, Tensor(a!) out) -> ()");
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
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
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

  // Gated RMSNorm (norm-before-gate) for Qwen3.x GDN layers:
  // y = x * rstd * weight * act(z); activation: 0 = silu, 1 = sigmoid.
  rocm_ops.def(
      "gated_rms_norm(Tensor! out, Tensor input, Tensor z, Tensor weight, "
      "float epsilon, int activation) -> ()");
  rocm_ops.impl("gated_rms_norm", torch::kCUDA, &gated_rms_norm);

}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
