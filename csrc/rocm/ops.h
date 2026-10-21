#pragma once

#include <string>

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

// Sequential-layout W4A16 MoE skinny GEMV (Triton moe_wna16 packing, not
// shuffled RDNA2 fused weights). Symmetric uint4b8, fp16, small M.
void moe_skinny_int4_decode(const at::Tensor& input, const at::Tensor& w13,
                            const at::Tensor& w13_scale, const at::Tensor& w2,
                            const at::Tensor& w2_scale,
                            const at::Tensor& topk_weights,
                            const at::Tensor& topk_ids, at::Tensor& act_buf,
                            at::Tensor& output, const int64_t group_size,
                            const std::optional<at::Tensor>& expert_map);

// Native resident-layout W4A16 MoE skinny GEMV. Weights are shuffled int32
// [E, K/8, N], scales are fp16 [E, K/group_size, N], and routing ids may be
// int32 or int64. Symmetric uint4b8 is the only supported quantization.
void moe_resident_int4_decode(const at::Tensor& input, const at::Tensor& w13,
                              const at::Tensor& w13_scale, const at::Tensor& w2,
                              const at::Tensor& w2_scale,
                              const at::Tensor& topk_weights,
                              const at::Tensor& topk_ids, at::Tensor& act_buf,
                              at::Tensor& output, const int64_t group_size,
                              const std::optional<at::Tensor>& expert_map);

at::Tensor gemv_f16_rdna2(const at::Tensor& x, const at::Tensor& w,
                          const std::optional<at::Tensor>& bias);
at::Tensor gemv_i8_rdna2(const at::Tensor& x, const at::Tensor& w,
                         const at::Tensor& scale,
                         const std::optional<at::Tensor>& bias);
at::Tensor rdna_gemv_act(const at::Tensor& x, const at::Tensor& w,
                         const std::optional<at::Tensor>& scale,
                         int64_t act_cols, double act_scale);
at::Tensor rdna_hc_up_gate_mix(const at::Tensor& lora, const at::Tensor& w,
                               const std::optional<at::Tensor>& scale,
                               const at::Tensor& xn, int64_t hc_count);
at::Tensor rdna_hc_up_gate_mix_prefill(const at::Tensor& dai,
                                       const at::Tensor& w_up,
                                       const at::Tensor& xn,
                                       int64_t lora_rank, int64_t hc_count);
// Qwen4Exp RDNA2 MonoKernel (vllm/models/qwen4_exp/amd/mono/rdna2/): K2 = the
// decode FFN block in one persistent launch, K1 = the attention-side HC
// preamble, GDN = fused GDN decode core, QSA = sparse decode attention.
int64_t mono_k2_rdna2_workspace_bytes(int64_t H, int64_t R, int64_t ND,
                                      int64_t E, int64_t Ise, int64_t I);
void mono_k2_rdna2(const torch::Tensor& res, const torch::Tensor& blk,
                   const torch::Tensor& inj, const torch::Tensor& norm_w,
                   const torch::Tensor& wd, const torch::Tensor& wu,
                   const torch::Tensor& wr, const torch::Tensor& se_w1,
                   const torch::Tensor& se_w2, const torch::Tensor& se_wg,
                   const torch::Tensor& w13, const torch::Tensor& s13,
                   const torch::Tensor& z13, const torch::Tensor& w2,
                   const torch::Tensor& s2, const torch::Tensor& z2,
                   torch::Tensor& res_out, torch::Tensor& inj_out,
                   torch::Tensor& out, torch::Tensor& workspace,
                   int64_t expert_base, int64_t zero_offset, int64_t num_wg,
                   double eps, double timeout_ms);
torch::Tensor mono_k2_rdna2_error(const torch::Tensor& workspace);
void mono_k2_rdna2_mark(torch::Tensor& stats, int64_t kind, int64_t width);
torch::Tensor mono_k2_rdna2_workspace_alloc(int64_t nbytes, int64_t device);
int64_t mono_k2_rdna2_mem_flags(const torch::Tensor& t);
int64_t mono_k2_rdna2_set_ar(int64_t handle, int64_t device);
void mono_k2_rdna2_e1a(const torch::Tensor& res, const torch::Tensor& blk,
                       const torch::Tensor& inj, const torch::Tensor& norm_w,
                       const torch::Tensor& wd, const torch::Tensor& wu,
                       const torch::Tensor& wq, const torch::Tensor& wba,
                       const torch::Tensor& convw, const torch::Tensor& convb,
                       torch::Tensor& conv_state, torch::Tensor& ssm_state,
                       const torch::Tensor& alog, const torch::Tensor& dtb,
                       const torch::Tensor& gnorm_w, const torch::Tensor& wo,
                       const torch::Tensor& sidx, torch::Tensor& res_out,
                       torch::Tensor& inj_out, torch::Tensor& out,
                       torch::Tensor& workspace, int64_t num_wg, double eps,
                       double gn_eps, int64_t gn_sigmoid, double scale,
                       double timeout_ms);
void mono_gdn_rdna2(const torch::Tensor& qkvz, const torch::Tensor& ba,
                    const torch::Tensor& conv_w, const torch::Tensor& conv_b,
                    torch::Tensor& conv_state, torch::Tensor& ssm_state,
                    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
                    const torch::Tensor& norm_w, const torch::Tensor& sidx,
                    const torch::Tensor& nacc, const torch::Tensor& qsl,
                    torch::Tensor& y, torch::Tensor& scratch,
                    int64_t num_k_heads, double scale, double eps,
                    int64_t gate_sigmoid);
int64_t mono_gdn_rdna2_scratch_bytes(int64_t M, int64_t HV, int64_t H);
void mono_qsa_decode_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                           const torch::Tensor& v_cache,
                           const torch::Tensor& indices,
                           const torch::Tensor& block_table,
                           const torch::Tensor& token_to_req,
                           torch::Tensor& out, torch::Tensor& scratch,
                           torch::Tensor& counters, int64_t splits);
int64_t mono_qsa_decode_rdna2_splits(int64_t M, int64_t KVH, int64_t W);
void mono_qsa_k_store_rdna2(
    const torch::Tensor& pooled, const torch::Tensor& weight,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    double eps, double norm_beta, int64_t rotary_dim, int64_t mrope_h,
    int64_t mrope_w, torch::Tensor& c_cache, const torch::Tensor& c_slots,
    torch::Tensor& r_cache, const torch::Tensor& r_slots,
    const torch::Tensor& r_rows, torch::Tensor& p_cache,
    const torch::Tensor& p_rows);
void mono_qsa_mqa_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                        const torch::Tensor& page_table,
                        const torch::Tensor& token_to_req,
                        const torch::Tensor& query_positions,
                        const torch::Tensor& sequence_lengths,
                        torch::Tensor& logits, torch::Tensor& visible_blocks,
                        int64_t compress_ratio, double score_divisor,
                        int64_t group, int64_t waves, int64_t tiles,
                        int64_t lanes_per_key, int64_t unroll, bool q_lds);
void mono_qsa_epilogue_rdna2(
    const torch::Tensor& qk, torch::Tensor& q_out, const torch::Tensor& q_w,
    const torch::Tensor& k_w, double q_eps, double k_eps, double norm_beta,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    int64_t rotary_dim, int64_t mrope_h, int64_t mrope_w,
    const torch::Tensor& state_table, const torch::Tensor& token_to_req,
    const torch::Tensor& query_start_loc, const torch::Tensor& end_positions,
    int64_t compress_ratio, torch::Tensor& c_cache, const torch::Tensor& c_slots,
    torch::Tensor& r_cache, const torch::Tensor& r_slots,
    torch::Tensor& p_cache);
void mono_qsa_store3_rdna2(torch::Tensor& c_cache, const torch::Tensor& c_slots,
                           const torch::Tensor& c_rows, torch::Tensor& r_cache,
                           const torch::Tensor& r_slots,
                           const torch::Tensor& r_rows, torch::Tensor& p_cache,
                           const torch::Tensor& p_rows);
void mono_k1_rdna2(const torch::Tensor& res, const torch::Tensor& blk,
                   const torch::Tensor& inj, const torch::Tensor& norm_w,
                   const torch::Tensor& wd, const torch::Tensor& wu,
                   torch::Tensor& res_out, torch::Tensor& inj_out,
                   torch::Tensor& out, torch::Tensor& workspace, int64_t num_wg,
                   double eps, double timeout_ms);

std::tuple<at::Tensor, at::Tensor> rdna_hc_mix_prefill(
    const at::Tensor& xn, const at::Tensor& w_down, const at::Tensor& w_up,
    int64_t lora_rank, int64_t hc_count);
at::Tensor rdna_se_gate_up_silu(const at::Tensor& x, const at::Tensor& w,
                                const std::optional<at::Tensor>& scale);
at::Tensor rdna_se_down_gated(const at::Tensor& act, const at::Tensor& w,
                              const std::optional<at::Tensor>& scale,
                              const at::Tensor& x, const at::Tensor& w_gate);

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

// W4A8 (int4 weights, int8 activations) prefill GEMM for gfx1030. Opt-in
// drop-in for the dense W4A16 prefill. The gemm entry is self-contained and
// returns a populated [M, N] fp16 tensor whether the W4A8 fast path fired or
// the W4A16 prefill fallback ran (it owns the shape/LDS eligibility and the
// internal fallback). Definitions in csrc/rocm/w4a8_sdot4_rdna2.cu.

// Per-(token, group) int8 quant of fp16 activations x [M, K] into the
// [T][K/8][MT][8] tile layout the GEMM reads, with per-(token, group) f32
// scales and int32 group sums. MT is fixed at 8 (config id 8's M tile).
// Returns the int8 buffer on success or an empty tensor when ineligible.
at::Tensor w4a8_act_quant_rdna2(const at::Tensor& x, int64_t group_size,
                                at::Tensor& a_i8, at::Tensor& a_scale,
                                at::Tensor& a_asum);

// W4A8 GEMM over the SAME packed W4 buffer RDNA2W4A16LinearKernel leaves
// behind (zero-extended nibbles + gptq_shuffle). Self-contained: allocates
// the int8 A, A scales, A group sums and the output internally, fires the
// a8_lds_k32_ag fast path, and falls back to gptq_gemm_rdna2_prefill when
// the shape/LDS is not eligible. use_v2_format selects zero_offset (0 for
// AWQ uint4, 1 for GPTQv1 uint4b8).
at::Tensor w4a8_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                           torch::Tensor b_qzeros, torch::Tensor b_scales,
                           torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

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

// T44: gfx1030 push-based one-shot all-reduce (opt-in via VLLM_RDNA_AR).
// Protocol from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao).
at::Tensor rdna_ar_init(int64_t rank, int64_t world,
                        const at::Tensor& device_ids, int64_t max_bytes,
                        const std::string& shm_name);
void rdna_ar_connect(int64_t handle, const at::Tensor& handles);
bool rdna_ar_can(int64_t handle, const at::Tensor& t);
at::Tensor rdna_ar_all_reduce(int64_t handle, const at::Tensor& in);
bool rdna_ar_timed_out(int64_t handle);
int64_t rdna_ar_timeout_info(int64_t handle);
int64_t rdna_ar_slow_info(int64_t handle);
int64_t rdna_ar_fast_calls(int64_t handle);

// FA-RDNA2: Flash-Attention v2 hand-port for AMD RDNA2 (gfx1030), used by
// the RDNA_ATTN backend (vllm/v1/attention/backends/rdna_attn.py) for
// decode (split-K) and prefill (paged varlen). Each op writes the attention
// output into `out` ([num_tokens, H_q, D] fp16, contiguous, like Q) and
// scales Q.K^T by `scale`. Definitions live in fa_rdna2.cu.
void fa_rdna2_decode_paged(torch::Tensor Q, torch::Tensor key_cache,
                           torch::Tensor value_cache, torch::Tensor block_table,
                           torch::Tensor seq_lens, int64_t block_size,
                           int64_t kv_splits, int64_t sliding_window,
                           double scale, torch::Tensor out,
                           c10::optional<torch::Tensor> cu_query_lens);

void fa_rdna2_prefill_paged_varlen(torch::Tensor Q, torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor block_table,
                                   torch::Tensor cu_query_lens,
                                   torch::Tensor seq_lens, int64_t block_size,
                                   int64_t causal, int64_t sliding_window,
                                   double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_short(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window, double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t kv_splits, int64_t sliding_window, double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_gqa(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window, double scale, torch::Tensor out);

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, bool fp32_accum,
                         bool use_v2_format);

// W4A8 (int4 weights, int8 activations) fused MoE for gfx1030. Opt-in
// drop-in for moe_gptq_gemm_rdna2 on the same packed weight layout: quants the
// whole token batch once, fires the sdot4 MoE kernel, and falls back internally
// to moe_gptq_gemm_rdna2 on any ineligibility. Same arg list plus the trailing
// zero-offset selector (use_v2_format: 0 for AWQ uint4, 1 for GPTQv1 uint4b8).
// Definitions in csrc/rocm/moe_w4a8_rdna2.cu.
void moe_w4a8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, bool use_v2_format,
                         bool fp32_accum);

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

// Decode projection: K-Hadamard(suh) + trellis GEMM + N-Hadamard(svh)
// on one stream. xh/mid/out are caller workspace. M is small (decode).
void exl3_project_rdna2(torch::Tensor x, torch::Tensor xh, torch::Tensor mid,
                        torch::Tensor out, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t bits,
                        int64_t cb);

// EXL3 (QTIP-style bitshift trellis) fused MoE GEMM kernel for AMD
// RDNA2/RDNA3 (gfx1030/gfx1100). Sorted-token-id grouping, per-expert
// packed trellis [E, K/16, N/16, 256*bits/16], codebook decode +
// v_dot2_f32_f16, topk-weighted atomic add epilogue.
void moe_exl3_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor trellis, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, int64_t bits, int64_t cb);

// EXL3 Hadamard-128 transform for AMD RDNA2/RDNA3 (gfx1030/gfx1100).
// y = H_128(x) * (scale/sqrt(128)), optionally pre-scaled (suh, A-side) or
// post-scaled (svh, C-side). Port of exllamav3_ext.had_r_128; applied
// OUTSIDE the EXL3 K-dot (wiki kernels/exl3.md).
void exl3_hadamard_128(torch::Tensor input, torch::Tensor output,
                       torch::optional<torch::Tensor> pre_scale,
                       torch::optional<torch::Tensor> post_scale, double scale);

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
torch::Tensor paged_mqa_logits_decode_rdna2(torch::Tensor q_fp8,
                                            torch::Tensor kv_cache,
                                            torch::Tensor weights,
                                            torch::Tensor context_lens,
                                            torch::Tensor block_tables,
                                            int64_t max_model_len,
                                            bool block_flat);

// Sparse MLA decode for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_decode_ragged_kernel` path on
// gfx1030 (the AITER MLA path is CDNA-only and does not run on
// gfx1030). One CTA per query, 32 threads (wave32), 2 heads per thread;
// online softmax with full acc_nope/acc_rope state in registers.
// FP8 (E4M3 OCP) K_nope with E8M0 block scales, bf16 K_rope. Gated
// by VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_decode_rdna2(
    torch::Tensor q,             // [B, H, D] fp16 or bf16
    torch::Tensor main_cache,    // [num_blocks, block_size, 576] uint8
    torch::Tensor main_indices,  // [nnz] int32
    torch::Tensor main_indptr,   // [B+1] int32
    torch::Tensor
        extra_cache,  // [num_blocks, block_size, 576] uint8 (may be empty)
    torch::Tensor extra_indices,  // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,   // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size, int64_t main_num_rows, int64_t extra_block_size,
    int64_t extra_num_rows, double scale,
    torch::Tensor attn_sink,  // [H] fp32 or empty
    torch::Tensor out);       // [B, H, D] bf16

// Sparse MLA prefill for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_prefill_ragged_kernel` path on
// gfx1030. Same online-softmax structure as sparse_mla_decode_rdna2,
// but the kv rows are plain fp16/bf16 (no fp8 slots, no E8M0 scales —
// the fp8_ds_mla cache encoding only applies post-encoder). One CTA per
// (query, head-group), 32 threads (wave32). Gated by
// VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_prefill_rdna2(
    torch::Tensor q,        // [T, H, D] fp16 or bf16
    torch::Tensor kv,       // [skv, D] fp16/bf16 (contiguous rows)
    torch::Tensor indices,  // [nnz] int32
    torch::Tensor indptr,   // [T + 1] int32
    int64_t num_kv, double scale,
    torch::Tensor attn_sink,  // [H] fp32 or empty
    torch::Tensor out);       // [T, H, D] same dtype as q

// DeepSeek-V4 q-norm + GPT-J RoPE + fp8_ds_mla KV insert for AMD RDNA
// (csrc/rocm/rdna/dsv4/kv_insert.cu). fp16 or bf16 q/kv in; writes the V4
// cache row (448 fp8 e4m3fn NoPE + 64 bf16 RoPE, 8 UE8M0 scale bytes) and
// returns q zero-padded to q_head_padded heads (empty when 0).
torch::Tensor dsv4_qnorm_rope_kv_insert_rdna(
    torch::Tensor const& q_in, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size, bool apply_q_norm, bool apply_q_rope);

// DeepSeek-V4 mHC pre / post for AMD RDNA (csrc/rocm/rdna/dsv4/mhc.cu).
// mhc_pre takes the precomputed mixes = x @ fn^T and returns
// {post [T, 4, 1] fp32, comb [T, 4, 4] fp32, layer_input [T, H]}.
std::vector<torch::Tensor> dsv4_mhc_pre_rdna(
    torch::Tensor const& residual, torch::Tensor const& mixes,
    torch::Tensor const& hc_scale, torch::Tensor const& hc_base,
    double rms_eps, double pre_eps, double sinkhorn_eps, double post_mult,
    int64_t sinkhorn_repeat);
torch::Tensor dsv4_mhc_post_rdna(torch::Tensor const& x,
                                 torch::Tensor const& residual,
                                 torch::Tensor const& post,
                                 torch::Tensor const& comb);

// INT8 per-(token, head) KV-cache writer for AMD RDNA2 (gfx1030).
// Symmetric signed int8 quantize + write to the interleaved cache
// layout used by RDNA_ATTN backend: [2, num_blocks, H_kv, D+4, block_size]
// int8, with the last 4 int8 bytes per (block, head, slot) being the
// raw fp32 K/V scale. Used by fa_rdna2_decode_paged_int8 to populate
// the per-(token, head) scale tensor the kernel reads inside its
// cooperative load. Per the kv-int8.md wiki contract — fused i8 quant
// + scale computation in a single CTA per (token, head).
void reshape_and_cache_int8_rdna2(
    torch::Tensor key,          // [num_tokens, H_kv, D] fp16
    torch::Tensor value,        // [num_tokens, H_kv, D] fp16
    torch::Tensor kv_cache,     // [2, num_blocks, H_kv, D + 4, block_size] int8
    torch::Tensor slot_mapping  // [num_tokens] int32 (-1 = skip)
);

// fp16 flash KV-cache writer for FA-RDNA2. Stride-aware so hybrid GDN
// pages (block_size=784, padded stride(0)) write correctly. Matches
// triton_reshape_and_cache_flash 5D-K / 4D-V addressing.
//   key/value:     [num_tokens, H_kv, D] fp16
//   key_cache:     [nb, H_kv, D/x, block_size, x] fp16
//   value_cache:   [nb, H_kv, D, block_size] fp16
//   slot_mapping:  [num_tokens] int32 or int64 (-1 = skip)
void reshape_and_cache_flash_rdna2(torch::Tensor key, torch::Tensor value,
                                   torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor slot_mapping);

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
    double scale, bool use_qk_l2norm);

// GatedDeltaNet (GDN) speculative-decoding verify for AMD RDNA2 (gfx1030).
// Hand port of fused_sigmoid_gating_delta_rule_update_kernel for the MTP
// spec-verify shape: varlen (cu_seqlens), continuous batching
// (ssm_state_indices), spec decoding (num_accepted_tokens), in-place final
// state, is_kda=False, head_k_dim 128, BV 32. One workgroup per
// (V-tile, sequence x value-head) walks every token of the sequence in
// registers and stores the state to slot t of the request's index row.
void gdn_spec_verify_rdna2(
    torch::Tensor q,                  // [B, T, H, K] fp16
    torch::Tensor k,                  // [B, T, H, K] fp16
    torch::Tensor v,                  // [B, T, HV, V] fp16
    torch::Tensor a,                  // [B, T, HV] fp16
    torch::Tensor b,                  // [B, T, HV] fp16
    torch::Tensor A_log,              // [HV] fp32/fp16
    torch::Tensor dt_bias,            // [HV] fp32/fp16
    torch::Tensor out,                // [B, T, HV, V] fp16
    torch::Tensor initial_state,      // [blocks, HV, V, K] fp32/fp16, in-place
    torch::Tensor cu_seqlens,         // [N + 1] int32
    torch::Tensor ssm_state_indices,  // [N, S] or [N] int32
    torch::Tensor num_accepted_tokens,  // [N] int32 or undefined
    double scale);

// GDN prefill kernels for AMD RDNA2 (gfx1030). Hand ports of the Triton/FLA
// chain `chunk_gated_delta_rule_fwd` (chunk.py:23-86) decomposed into 5 HIP
// kernels; varlen uses cu_seqlens/chunk_indices/chunk_offsets (pass empty
// for non-varlen).

// fused_post_conv_prep + chunk_local_cumsum fused. g is already cumsum'd
// when it leaves this kernel (fold of `chunk_local_cumsum` in chunk.py:37).
void gdn_prefill_prep_rdna2(
    torch::Tensor mixed_qkv,       // [L, qkv_dim] fp16 contiguous in last dim
    torch::Tensor a,               // [L, HV] fp16
    torch::Tensor b,               // [L, HV] fp16
    torch::Tensor A_log,           // [HV] fp32 or fp16 contiguous
    torch::Tensor dt_bias,         // [HV] fp32 or fp16 contiguous
    torch::Tensor q,               // [L, H, K] fp16 (output)
    torch::Tensor k_out,           // [L, H, K] fp16 (output)
    torch::Tensor v,               // [L, HV, V] fp16 (output)
    torch::Tensor g_cumsum,        // [L, HV] fp32 (output)
    torch::Tensor beta,            // [L, HV] fp32 (output)
    torch::Tensor cu_seqlens,      // [N+1] int32
    torch::Tensor chunk_indices);  // [NT, 2] int32

// chunk_scaled_dot_kkt. A[i,j] = beta_i * exp(g_i - g_j) * (k_i . k_j)
// for i > j (strict), else 0. The [64,128] @ [128,64] dot is fp32 FMA
// (no V_DOT2: _CAST_DOT_TO_K_DTYPE is False on gfx1030, beta*k stays fp32).
void gdn_prefill_kkt_rdna2(
    torch::Tensor k,               // [B, T, Hg, K] fp16 contiguous in last dim
    torch::Tensor beta,            // [B, T, H] fp32
    torch::Tensor g,               // [B, T, H] fp32 (cumsum'd g, from prep)
    torch::Tensor A,               // [B, T, H, BT] fp32 (output)
    torch::Tensor cu_seqlens,      // [N+1] int32
    torch::Tensor chunk_indices);  // [NT, 2] int32

// solve_tril + recompute_w_u FUSED. Same (NT, B*H) grid; A_inv [64,64]
// stays on-chip between the fp32 16x16 forward-substitution + Schur phase
// and the V_DOT2_F32_F16 w/u production phase.
void gdn_prefill_solve_wy_rdna2(
    torch::Tensor A,               // [B, T, H, BT] fp32 (from kkt)
    torch::Tensor k,               // [B, T, Hg, K] fp16
    torch::Tensor v,               // [B, T, H, V] fp16
    torch::Tensor beta,            // [B, T, H] fp32
    torch::Tensor g,               // [B, T, H] fp32 (cumsum'd g, from prep)
    torch::Tensor A_inv,           // [B, T, H, BT] fp16 (output)
    torch::Tensor w,               // [B, T, H, K] fp16 (output)
    torch::Tensor u,               // [B, T, H, V] fp16 (output)
    torch::Tensor cu_seqlens,      // [N+1] int32
    torch::Tensor chunk_indices);  // [NT, 2] int32

// chunk_gated_delta_rule_fwd_h (serial inter-chunk recurrence). Reuses
// Stage-1 decode layout (256 threads = 32 v-rows x 8 k-slices, h [32,128]
// fp32 register-resident). initial_state = h0, final_state = last h per seq.
void gdn_prefill_delta_h_rdna2(
    torch::Tensor k,      // [B, T, Hg, K] fp16
    torch::Tensor u,      // [B, T, H, V] fp16
    torch::Tensor w,      // [B, T, H, K] fp16
    torch::Tensor g,      // [B, T, H] fp32 (cumsum'd g)
    torch::Tensor h,      // [NT, H, V, K] fp16 (per-chunk h, output)
    torch::Tensor v_new,  // [B, T, H, V] fp16 (output)
    c10::optional<torch::Tensor>
        initial_state,                         // [N, H, V, K] fp32 or undefined
    c10::optional<torch::Tensor> final_state,  // [N, H, V, K] fp32 or undefined
    c10::optional<torch::Tensor> cu_seqlens,   // [N+1] int32 or undefined
    c10::optional<torch::Tensor> chunk_offsets,  // [N+1] int32 or undefined
    int64_t chunk_size);                         // must be 64

// chunk_fwd_o. q.A (intra-chunk) + q.h (state) with V_DOT2_F32_F16 and the
// inclusive `>=` causal mask. h: 5D non-varlen or 4D varlen, fp16 or fp32.
void gdn_prefill_o_rdna2(
    torch::Tensor q,  // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor k,  // [B, T, Hg, K] fp16 contiguous in K
    torch::Tensor v,  // [B, T, H, V] fp16 contiguous in V
    torch::Tensor h,  // 5D non-varlen or 4D varlen; fp16 or fp32
    torch::Tensor g,  // [B, T, H] fp32 (cumsum'd g)
    torch::Tensor o,  // [B, T, H, V] fp16 (output)
    double scale,
    torch::Tensor cu_seqlens,      // [N+1] int32
    torch::Tensor chunk_offsets);  // [N+1] int32

// Paged MQA logits for DeepSeek V4 Lightning Indexer on AMD RDNA2
// (gfx1030). AITER is CDNA-only and crashes on gfx1030; this kernel
// replaces `rocm_aiter_sparse_attn_indexer`'s paged MQA logits stage
// with a fused FP8 dequant + dot-product + ReLU + per-head weighted
// sum kernel. Output is logits [B*next_n, max_model_len] fp32 with -inf
// in padded slots. Top-K selection is done by the standard upstream
// `top_k_per_row_decode` kernel (runs on gfx1030).
torch::Tensor paged_mqa_logits_decode_rdna2(torch::Tensor q_fp8,
                                            torch::Tensor kv_cache,
                                            torch::Tensor weights,
                                            torch::Tensor context_lens,
                                            torch::Tensor block_tables,
                                            int64_t max_model_len,
                                            bool block_flat);
// Ragged prefill MQA logits [M, N] (fp32, -inf outside [ks, ke)).
torch::Tensor mqa_logits_prefill_rdna2(torch::Tensor q, torch::Tensor k,
                                       torch::Tensor k_scale,
                                       torch::Tensor weights, torch::Tensor ks,
                                       torch::Tensor ke);
// ===== RDNA2 declarations backported from rdna2_extras ops.h =====
// (Group1-9 ported the .cu sources + torch_bindings.cpp registrations;
//  these are the matching ops.h declarations needed to make them compile.)

void rms_norm(torch::Tensor& out, const torch::Tensor& input,
              const torch::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        const torch::Tensor& weight, double epsilon);

void gated_rms_norm(torch::Tensor& out, const torch::Tensor& input,
                    const torch::Tensor& z, const torch::Tensor& weight,
                    double epsilon, int64_t activation);

// ---------------------------------------------------------------------------
// v1 sampler kernels (gfx1030). HIP ports of the Triton kernels in
// vllm/v1/worker/gpu/sample/{gumbel,penalties,min_p,logit_bias,bad_words,
// logprob,prompt_logprob}.py.
// ---------------------------------------------------------------------------

void temperature_rdna2(torch::Tensor logits, int64_t logits_stride,
                       torch::Tensor expanded_idx_mapping,
                       torch::Tensor temperature, int64_t vocab_size);

void gumbel_sample_rdna2(
    torch::Tensor local_argmax, int64_t local_argmax_stride,
    torch::Tensor local_max, int64_t local_max_stride,
    std::optional<torch::Tensor> logits_cache, int64_t logits_cache_stride_0,
    int64_t logits_cache_stride_1,
    std::optional<torch::Tensor> logits_cache_col, torch::Tensor logits,
    int64_t logits_stride, torch::Tensor expanded_idx_mapping,
    torch::Tensor seeds, torch::Tensor pos, torch::Tensor temperature,
    int64_t vocab_size, bool is_drafting, bool apply_temperature,
    bool use_fp64, bool per_token_col);

void penalties_rdna2(torch::Tensor logits, int64_t logits_stride,
                     torch::Tensor expanded_idx_mapping,
                     torch::Tensor token_ids, torch::Tensor expanded_local_pos,
                     torch::Tensor repetition_penalty,
                     torch::Tensor frequency_penalty,
                     torch::Tensor presence_penalty,
                     torch::Tensor prompt_bin_mask,
                     int64_t prompt_bin_mask_stride,
                     torch::Tensor output_bin_counts,
                     int64_t output_bin_counts_stride, int64_t vocab_size);

void bincount_rdna2(torch::Tensor expanded_idx_mapping,
                    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
                    torch::Tensor prompt_len, torch::Tensor prefill_len,
                    torch::Tensor prompt_bin_mask,
                    int64_t prompt_bin_mask_stride,
                    torch::Tensor output_bin_counts,
                    int64_t output_bin_counts_stride, int64_t max_prefill_len);

void min_p_rdna2(torch::Tensor logits, int64_t logits_stride,
                 torch::Tensor expanded_idx_mapping, torch::Tensor min_p,
                 int64_t vocab_size);

void topk_topp_rdna2(torch::Tensor logits, int64_t logits_stride,
                     const std::optional<torch::Tensor>& k,
                     const std::optional<torch::Tensor>& p, double mask_value);

void logit_bias_rdna2(torch::Tensor logits, int64_t logits_stride,
                      int64_t vocab_size, torch::Tensor expanded_idx_mapping,
                      torch::Tensor num_allowed, torch::Tensor allowed_ids,
                      int64_t allowed_ids_stride, torch::Tensor num_bias,
                      torch::Tensor bias_ids, int64_t bias_ids_stride,
                      torch::Tensor bias, int64_t bias_stride, torch::Tensor pos,
                      torch::Tensor min_lens, torch::Tensor num_stop,
                      torch::Tensor restore_when_all_masked,
                      torch::Tensor stop_ids, int64_t stop_ids_stride,
                      int64_t block_size, bool check_all_masked);

void bad_words_rdna2(torch::Tensor logits, int64_t logits_stride,
                     torch::Tensor expanded_idx_mapping,
                     torch::Tensor bad_word_token_ids,
                     int64_t bad_word_token_ids_stride,
                     torch::Tensor bad_word_offsets,
                     int64_t bad_word_offsets_stride,
                     torch::Tensor num_bad_words, torch::Tensor all_token_ids,
                     int64_t all_token_ids_stride, torch::Tensor prompt_len,
                     torch::Tensor total_len, torch::Tensor input_ids,
                     torch::Tensor expanded_local_pos,
                     int64_t max_num_bad_words);

void topk_log_softmax_rdna2(torch::Tensor output, torch::Tensor logits,
                            int64_t logits_stride, torch::Tensor topk_ids,
                            int64_t topk_ids_stride, int64_t topk,
                            int64_t vocab_size, int64_t topk_block_size);

void token_ranks_rdna2(torch::Tensor token_ranks, torch::Tensor logits,
                       int64_t logits_stride, torch::Tensor sampled_token_ids,
                       int64_t vocab_size);

void prompt_logprobs_token_ids_rdna2(
    torch::Tensor prompt_logprobs_token_ids, torch::Tensor query_start_loc,
    torch::Tensor idx_mapping, torch::Tensor num_computed_tokens,
    torch::Tensor all_token_ids, int64_t all_token_ids_stride);

void moe_w8a16_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_qzeros, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);

void moe_mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);

void moe_w8a16_fp8_gemm_rdna2(
    torch::Tensor a, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor b_qzeros, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_topk_weight, int64_t output_topk);

void mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                      torch::Tensor b_q_weight, torch::Tensor b_scales,
                      int64_t size_m, int64_t size_n, int64_t size_k);

void sparse_mla_decode_rdna2(
    torch::Tensor q,             // [B, H, D] fp16 or bf16
    torch::Tensor main_cache,    // [num_blocks, block_size, 576] uint8
    torch::Tensor main_indices,  // [nnz] int32
    torch::Tensor main_indptr,   // [B+1] int32
    torch::Tensor
        extra_cache,  // [num_blocks, block_size, 576] uint8 (may be empty)
    torch::Tensor extra_indices,  // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,   // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size, int64_t main_num_rows, int64_t extra_block_size,
    int64_t extra_num_rows, double scale,
    torch::Tensor attn_sink,  // [H] fp32 or empty
    torch::Tensor out);

void sparse_mla_prefill_rdna2(
    torch::Tensor q,        // [T, H, D] fp16 or bf16
    torch::Tensor kv,       // [skv, D] fp16/bf16 (contiguous rows)
    torch::Tensor indices,  // [nnz] int32
    torch::Tensor indptr,   // [T + 1] int32
    int64_t num_kv, double scale,
    torch::Tensor attn_sink,  // [H] fp32 or empty
    torch::Tensor out);

void reshape_and_cache_int8_rdna2(
    torch::Tensor key,          // [num_tokens, H_kv, D] fp16
    torch::Tensor value,        // [num_tokens, H_kv, D] fp16
    torch::Tensor kv_cache,     // [2, num_blocks, H_kv, D + 4, block_size] int8
    torch::Tensor slot_mapping  // [num_tokens] int32 (-1 = skip)
);

void reshape_and_cache_flash_rdna2(torch::Tensor key, torch::Tensor value,
                                   torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor slot_mapping);

void gdn_decode_rdna2(
    torch::Tensor mixed_qkv,          // [B, 2*H*K + HV*V] fp16
    torch::Tensor a,                  // [B, HV] fp16
    torch::Tensor b,                  // [B, HV] fp16
    torch::Tensor A_log,              // [HV] fp32
    torch::Tensor dt_bias,            // [HV] fp32
    torch::Tensor out,                // [B, 1, HV, V] fp16
    torch::Tensor initial_state,      // [blocks, HV, V, K] fp32, in-place
    torch::Tensor ssm_state_indices,  // [B] int32
    double scale, bool use_qk_l2norm);

// causal_conv1d_update for AMD RDNA2 (gfx1030). Single-token decode path:
// per-batch depthwise FIR filter with conv_state shift-left + append.
// cudagraph-safe (no global scratch, all state in registers).
void causal_conv1d_update_rdna2(
    torch::Tensor x,                   // [batch, dim, 1] fp16
    torch::Tensor conv_state,          // [num_cache_lines, dim, state_len] fp16
    torch::Tensor weight,              // [dim, width] fp16
    torch::Tensor bias,                // [dim] fp16 or undefined
    torch::Tensor out,                 // [batch, dim, 1] fp16
    torch::Tensor conv_state_indices,  // [batch] int32
    bool silu_activation);

// causal_conv1d_update for AMD RDNA2 (gfx1030), varlen / speculative-decoding
// path (the IS_VARLEN / IS_SPEC_DECODING cases of the Triton kernel). Per
// sequence: num_accepted_tokens - 1 selects the first history column, the
// query_start_loc slice gives the token range, and the whole revised
// conv_state (state_len shortened by max_query_len - seqlen) is written back.
void causal_conv1d_update_spec_rdna2(
    torch::Tensor x,                      // [dim, cu_seqlen] fp16 (varlen) or
                                          // [batch, dim, seqlen] fp16
    torch::Tensor conv_state,             // [num_cache_lines, dim, capacity] fp16
    torch::Tensor weight,                 // [dim, width] fp16
    torch::Tensor bias,                   // [dim] fp16 or undefined
    torch::Tensor out,                    // same shape/layout as x
    torch::Tensor conv_state_indices,     // [batch] int32
    torch::Tensor num_accepted_tokens,    // [batch] int32 or undefined
    torch::Tensor query_start_loc,        // [batch+1] int32 or undefined
    int64_t state_len, int64_t max_query_len, int64_t null_block_id,
    bool silu_activation);

// causal_conv1d_fwd for AMD RDNA2 (gfx1030). Varlen prefill path:
// per-sequence depthwise FIR over [dim, cu_seqlen] with conv_state load
// (initial state) and writeback (last state_len tokens). cudagraph-safe
// (per-warp scratch in shared memory, no global allocations, no Triton
// JIT scratch pointers).
void causal_conv1d_fwd_rdna2(
    torch::Tensor x,                  // [dim, cu_seqlen] fp16
    torch::Tensor weight,             // [dim, width] fp16
    torch::Tensor bias,               // [dim] fp16 or undefined
    torch::Tensor conv_state,         // [num_cache_lines, dim, state_len] fp16
    torch::Tensor query_start_loc,    // [batch+1] int32
    torch::Tensor cache_indices,      // [batch] int32
    torch::Tensor has_initial_state,  // [batch] bool or undefined
    torch::Tensor out,                // [dim, cu_seqlen] fp16
    bool silu_activation, int64_t null_block_id = 0);

void mrope_forward_rdna2(torch::Tensor q,    // [num_tokens, n_qh * hd] fp16
                         torch::Tensor k,    // [num_tokens, n_kh * hd] fp16
                         torch::Tensor cos,  // [3, num_tokens, rd/2] fp16
                         torch::Tensor sin,  // [3, num_tokens, rd/2] fp16
                         int64_t num_tokens, int64_t n_qh, int64_t n_kh,
                         int64_t hd, int64_t rd, int64_t sec_t, int64_t sec_h,
                         int64_t sec_w, bool is_interleaved,
                         bool is_neox_style);

// ---------------------------------------------------------------------------
// HC prefill HIP kernels for Qwen4Exp / Qwen3.8-Flash-Next on gfx1030.
// Opt-in via VLLM_RDNA_HC_PREFILL_HIP=1 (default off; the Triton path
// in vllm/models/qwen4_exp/amd/ops/hc.py remains the default until the
// HIP port is verified end-to-end).
// ---------------------------------------------------------------------------

void hc_grouped_gemma_rmsnorm_rdna2(
    torch::Tensor x,       // [N, DIM] fp16, last-dim contiguous
    torch::Tensor weight,  // [GROUP_DIM] or [DIM] fp16
    torch::Tensor y,       // [N, DIM] fp16
    int64_t num_groups, double eps);

void hc_silu_rdna2(torch::Tensor x,  // [N, DIM] fp16
                   torch::Tensor y,  // [N, DIM] fp16
                   int64_t hc_count);

void hc_gate_mix_rdna2(torch::Tensor x,     // [N, DIM] fp16
                       torch::Tensor gate,  // [N, DIM] fp16
                       torch::Tensor y,     // [N, DIM/HC] fp16
                       int64_t hc_count);

void hc_combine_rdna2(torch::Tensor residual,          // [N, DIM] fp16
                      torch::Tensor block_output,      // [N, DIM/HC] fp16
                      torch::Tensor injection_logits,  // [N, HC] fp16
                      torch::Tensor out,               // [N, DIM] fp16
                      int64_t hc_count);

void hc_combine_norm_rdna2(torch::Tensor residual,          // [N, DIM] fp16
                           torch::Tensor block_output,      // [N, DIM/HC] fp16
                           torch::Tensor injection_logits,  // [N, HC] fp16
                           torch::Tensor norm_weight,  // [DIM/HC] or [DIM] fp16
                           torch::Tensor out,  // [N, DIM] fp16 (combined)
                           torch::Tensor y,    // [N, DIM] fp16 (post-norm)
                           int64_t hc_count, double eps);

// ---------------------------------------------------------------------------
// QSA decode HIP kernels for Qwen4Exp / Qwen3.8-Flash-Next on gfx1030.
// Opt-in via VLLM_RDNA_QSA_HIP=1.
// ---------------------------------------------------------------------------

void qsa_store_cache_rows_rdna2(
    torch::Tensor rows,       // [num_rows, WIDTH] fp16
    torch::Tensor slots,      // [num_rows] int64
    torch::Tensor cache,      // [num_blocks, PAGE_SIZE, 1, WIDTH] fp16/int64
    const at::Tensor& page_size,
    const at::Tensor& width);

void qsa_compress_groups_rdna2(
    torch::Tensor raw_keys,
    torch::Tensor raw_positions,
    torch::Tensor compressor_state_cache,
    torch::Tensor rope_cache,
    torch::Tensor compressor_state_table,
    torch::Tensor token_to_req,
    torch::Tensor query_start_loc,
    torch::Tensor logical_positions,
    torch::Tensor compressed_slots,
    torch::Tensor pooled,
    torch::Tensor first_positions,
    const at::Tensor& compress_ratio,
    const at::Tensor& compressor_state_size,
    const at::Tensor& head_dim,
    bool load_rope_positions);

torch::Tensor qsa_sparse_paged_attention_rdna2(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& logical_indices,
    const at::Tensor& block_table,
    const at::Tensor& token_to_req,
    c10::optional<at::Tensor> out);

torch::Tensor qsa_expand_indices_rdna2(
    const at::Tensor& block_indices,
    const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths,
    const at::Tensor& token_to_req,
    int64_t compress_ratio,
    int64_t token_topk,
    c10::optional<at::Tensor> out);

std::tuple<torch::Tensor, torch::Tensor> qsa_mqa_paged_rdna2(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& page_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths,
    int64_t compress_ratio,
    int64_t num_columns,
    double score_scale);

// Qwen4Exp QSA metadata builder HIP kernel for gfx1030 (RDNA2). Always on for
// gfx10x; the Triton builder stays the fallback elsewhere.
void build_qsa_metadata_rdna2(
    torch::Tensor query_start_loc,    // [num_reqs + 1] int32
    torch::Tensor seq_lens,           // [num_reqs] int32
    torch::Tensor common_slot_mapping,  // [num_tokens] int64
    torch::Tensor block_table,        // [num_reqs, num_columns] int32
    torch::Tensor token_to_req,       // [num_tokens] int32, written
    torch::Tensor logical_positions,  // [num_tokens] int64, written
    torch::Tensor visible_blocks,     // [num_tokens] int32, written
    torch::Tensor slot_mapping,       // [num_tokens] int64, written
    const std::optional<torch::Tensor>& k_work_metadata,  // [work, 2] int32
    int64_t num_reqs, int64_t num_mapped_tokens, int64_t num_tokens,
    int64_t max_num_work, int64_t num_search_steps, int64_t work_search_steps,
    int64_t request_scan_size, int64_t storage_block_size,
    int64_t compress_ratio, int64_t circular_buffer_size);

// ---------------------------------------------------------------------------
// Vision-tower bilinear position-embedding interpolation for gfx1030 (RDNA2).
// HIP port of qwen3_vl.py::_bilinear_pos_embed_kernel. Always on for gfx10x;
// the Triton kernel is the fallback on other platforms.
// ---------------------------------------------------------------------------

torch::Tensor bilinear_pos_embed_rdna2(
    torch::Tensor embed_weight,  // [num_grid * num_grid, hidden_dim]
    torch::Tensor out,           // [t * h * w, hidden_dim], written
    int64_t H, int64_t W, double h_scale, double w_scale, int64_t num_grid,
    int64_t m_size);

// ---------------------------------------------------------------------------
// PLE dilated short-conv HIP kernels for Qwen4Exp / Qwen3.8-Flash-Next on
// gfx1030. Opt-in via VLLM_RDNA_PLE_CONV_HIP=1.
// ---------------------------------------------------------------------------

void ple_short_conv_decode_rdna2(
    torch::Tensor x,           // [B, D] fp16
    torch::Tensor conv_state,  // [num_lines, D, state_len] fp16 (in-place)
    torch::Tensor weight,      // [D, K] fp16
    std::optional<torch::Tensor> bias,      // [D] fp16 or undefined
    torch::Tensor out,                      // [B, D] fp16
    torch::Tensor state_idx,                // [B] int32
    std::optional<torch::Tensor> has_init,  // [B] uint8 or undefined
    int64_t dilation, int64_t state_len, bool silu, int64_t null_block);

// Glm5Next post-conv KDA recurrence (gfx1030). State is [slots, H, V, K]
// fp32, matching fused_recurrent_kda. q/k/v/g/beta/out are fp16 or bf16.
// bf16 is widened to fp32 in software. Does not run conv or RMSNorm.
void glm5_kda_recurrent_rdna2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor g, torch::Tensor beta,
                              torch::Tensor A_log, torch::Tensor dt_bias,
                              torch::Tensor state_indices,
                              torch::Tensor ssm_state, torch::Tensor out,
                              double lower_bound);

// Ported Glm5Next kernels. Registered for gfx1030. Not called by the
// upstream forward: the fused decode redoes conv and uses [K, V] state,
// the prefill chain expects a DS conv layout, and the DSA kernels read
// packed fp16 rows / expanded 256-d NoPE rather than the fp8 kpool cache
// and latent MLA.
void glm5_dsa_indexer_rdna2(torch::Tensor q_idx, torch::Tensor packed,
                            torch::Tensor weights, torch::Tensor kv_lens,
                            torch::Tensor ape, torch::Tensor pool_indices_out,
                            torch::Tensor pool_valid_out,
                            torch::Tensor scores_out, int64_t kpool);
void glm5_dsa_mla_decode_rdna2(torch::Tensor q_nope, torch::Tensor k_sel,
                               torch::Tensor v_sel, torch::Tensor sel_valid,
                               torch::Tensor out, double scale);
void glm5_kda_decode_rdna2(torch::Tensor qkv_raw, torch::Tensor conv_w,
                           torch::Tensor conv_state, torch::Tensor A_log,
                           torch::Tensor dt_bias, torch::Tensor f,
                           torch::Tensor beta, torch::Tensor out_gate,
                           torch::Tensor norm_w, torch::Tensor state_indices,
                           torch::Tensor ssm_state, torch::Tensor out,
                           double lower_bound, double norm_eps);
void glm5_kda_prefill_prep_rdna2(
    torch::Tensor mixed_qkv, torch::Tensor conv_w, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor f, torch::Tensor beta_raw,
    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
    torch::Tensor g_out, torch::Tensor beta_out, torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices, double lower_bound);
void glm5_kda_prefill_kkt_rdna2(torch::Tensor k, torch::Tensor beta,
                                torch::Tensor g, torch::Tensor A,
                                torch::Tensor cu_seqlens,
                                torch::Tensor chunk_indices);
void glm5_kda_prefill_solve_wy_rdna2(
    torch::Tensor A, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
    torch::Tensor g, torch::Tensor A_inv, torch::Tensor w, torch::Tensor u,
    torch::Tensor cu_seqlens, torch::Tensor chunk_indices);
void glm5_kda_prefill_delta_h_rdna2(
    torch::Tensor k, torch::Tensor u, torch::Tensor w, torch::Tensor g,
    torch::Tensor h, torch::Tensor v_new,
    c10::optional<torch::Tensor> initial_state,
    c10::optional<torch::Tensor> final_state,
    c10::optional<torch::Tensor> cu_seqlens,
    c10::optional<torch::Tensor> chunk_offsets, int64_t chunk_size);
void glm5_kda_prefill_o_rdna2(torch::Tensor q, torch::Tensor k,
                              torch::Tensor v_new, torch::Tensor h,
                              torch::Tensor g, torch::Tensor o, double scale,
                              torch::Tensor cu_seqlens,
                              torch::Tensor chunk_offsets);

void ple_short_conv_prefill_rdna2(
    torch::Tensor x_packed,                    // [B, D, max_len] fp16
    torch::Tensor init_state,                  // [B, D, state_len] fp16
    torch::Tensor weight,                      // [D, K] fp16
    std::optional<torch::Tensor> bias,         // [D] fp16 or undefined
    torch::Tensor out,                         // [B, D, max_len] fp16
    torch::Tensor lengths,                     // [B] int32
    std::optional<torch::Tensor> valid_state,  // [B] uint8 or undefined
    int64_t dilation, int64_t state_len, bool silu);

// ---------------------------------------------------------------------------
// v1 worker per-step index/arithmetic kernels (gfx1030).
// HIP ports of _gather_block_tables_kernel, _compute_slot_mappings_kernel,
// _apply_write_kernel (single-group and MULTI_GROUP) and _zero_kv_blocks_kernel.
// ---------------------------------------------------------------------------

void gather_block_tables_rdna2(
    torch::Tensor batch_idx_to_req_idx,   // [num_reqs] int32
    torch::Tensor src_block_table_ptrs,   // [num_kv_cache_groups] uint64
    torch::Tensor dst_block_table_ptrs,   // [num_kv_cache_groups] uint64
    torch::Tensor block_table_strides,    // [num_kv_cache_groups] int64
    torch::Tensor num_blocks_ptr,         // [num_kv_cache_groups, max_num_reqs]
    int64_t num_blocks_stride, int64_t num_reqs, int64_t num_reqs_padded);

void compute_slot_mappings_rdna2(
    int64_t max_num_tokens,
    torch::Tensor idx_mapping,            // [num_reqs] int32
    torch::Tensor query_start_loc,        // [num_reqs + 1] int32
    torch::Tensor pos,                    // [num_tokens] int64
    torch::Tensor block_table_ptrs,       // [num_kv_cache_groups] uint64
    torch::Tensor block_table_strides,    // [num_kv_cache_groups] int64
    torch::Tensor block_sizes,            // [num_kv_cache_groups] int32
    torch::Tensor kernel_block_sizes,     // [num_kv_cache_groups] int32
    torch::Tensor slot_mapping_enabled,   // [num_kv_cache_groups] bool
    torch::Tensor dcp_sharded,            // [num_kv_cache_groups] bool
    torch::Tensor slot_mappings,          // [num_kv_cache_groups, max_num_tokens]
    int64_t slot_mappings_stride, int64_t cp_rank, int64_t cp_size,
    int64_t cp_interleave, int64_t pad_id);

void apply_write_rdna2(torch::Tensor out,  // [rows, row_width] int32/int64/fp32
                       int64_t out_stride, torch::Tensor write_indices,
                       torch::Tensor write_starts, torch::Tensor write_contents,
                       torch::Tensor write_cu_lens);

void fused_apply_write_rdna2(torch::Tensor output_ptrs,    // [num_groups] uint64
                             torch::Tensor output_strides,  // [num_groups] int64
                             torch::Tensor write_indices,
                             torch::Tensor write_starts,
                             torch::Tensor write_contents,
                             torch::Tensor write_cu_lens,
                             torch::Tensor write_group_ids);

void zero_kv_blocks_rdna2(
    torch::Tensor seg_addrs,          // [num_segments] uint64
    torch::Tensor seg_block_strides,  // [num_segments] int64
    torch::Tensor seg_page_sizes,     // [num_segments] int64
    torch::Tensor block_ids,          // [num_blocks] int64
    int64_t chunk_size, int64_t max_chunks);

// ---------------------------------------------------------------------------
// Unquantized (fp16) MoE for gfx1030 (RDNA2). HIP replacement for the
// fused_moe_kernel path in fused_moe.py::fused_experts; selected as the
// "RDNA2 HIP" unquantized MoE backend on gfx10x. See
// csrc/rocm/moe_unquantized_rdna2.cu for the layout contract.
// ---------------------------------------------------------------------------

void moe_gate_up_silu_unquantized_rdna2(
    torch::Tensor x,        // [M, K] fp16
    torch::Tensor w13,      // [E, 2*N, K] fp16
    torch::Tensor topk_ids,  // [M * topk] int32/int64, global expert ids
    const std::optional<torch::Tensor>& expert_map,  // [global_E] int32
    torch::Tensor act,      // [M * topk, N] fp16, written
    int64_t topk, int64_t N, int64_t K);

void moe_down_unquantized_rdna2(
    torch::Tensor act,          // [M * topk, N] fp16
    torch::Tensor w2,           // [E, K, N] fp16
    torch::Tensor topk_ids,     // [M * topk] int32/int64
    const std::optional<torch::Tensor>& expert_map,  // [global_E] int32
    torch::Tensor topk_weights,  // [M * topk] fp32 (empty when unused)
    bool mul_topk_weight,
    torch::Tensor slot_out,  // [M * topk, K] fp16, weighted down result per
                             // (token, top-k) pair; top-k reduced by moe_sum
    int64_t topk, int64_t N, int64_t K);

// ---------------------------------------------------------------------------
// v1 worker input-batch kernels (gfx1030). HIP ports of
// _prepare_prefill_inputs_kernel, _prepare_pos_seq_lens_kernel,
// _combine_sampled_and_draft_tokens_kernel,
// _get_num_sampled_and_rejected_kernel, _post_update_kernel and
// _expand_idx_mapping_kernel in vllm/v1/worker/gpu/input_batch.py.
// ---------------------------------------------------------------------------

void prepare_prefill_inputs_rdna2(
    torch::Tensor input_ids,            // [max_num_tokens] int32
    torch::Tensor next_prefill_tokens,  // [num_lookahead, max_num_reqs] int32
    int64_t next_prefill_tokens_stride, int64_t num_lookahead,
    torch::Tensor idx_mapping,      // [num_reqs] int32
    torch::Tensor query_start_loc,  // [num_reqs + 1] int32
    torch::Tensor all_token_ids,    // [max_num_reqs, max_model_len] int32
    int64_t all_token_ids_stride, torch::Tensor prefill_len,
    torch::Tensor num_computed_tokens);

void prepare_pos_seq_lens_rdna2(torch::Tensor pos,       // [num_tokens] int64
                                torch::Tensor seq_lens,  // [max_num_reqs] int32
                                torch::Tensor idx_mapping,
                                torch::Tensor query_start_loc,
                                torch::Tensor num_computed_tokens,
                                int64_t max_num_reqs);

void combine_sampled_and_draft_tokens_rdna2(
    torch::Tensor input_ids, torch::Tensor idx_mapping,
    torch::Tensor last_sampled_tokens,  // [max_num_reqs, 1] int64
    torch::Tensor query_start_loc, torch::Tensor seq_lens,
    torch::Tensor prefill_len,
    torch::Tensor draft_tokens,  // [max_num_reqs, steps] int64
    int64_t draft_tokens_stride, torch::Tensor cu_num_logits,
    torch::Tensor logits_indices,  // [total_num_logits] int64
    int64_t num_new_sampled_tokens);

void get_num_sampled_and_rejected_rdna2(torch::Tensor num_sampled,
                                        torch::Tensor num_rejected,
                                        torch::Tensor seq_lens,
                                        torch::Tensor cu_num_logits,
                                        torch::Tensor idx_mapping,
                                        torch::Tensor prefill_len);

void post_update_rdna2(torch::Tensor idx_mapping,
                       torch::Tensor num_computed_tokens,
                       torch::Tensor last_sampled_tokens,
                       const std::optional<torch::Tensor>& output_bin_counts,
                       int64_t output_bin_counts_stride,
                       torch::Tensor sampled_tokens,
                       int64_t sampled_tokens_stride, torch::Tensor num_sampled,
                       torch::Tensor num_rejected,
                       const std::optional<torch::Tensor>& query_start_loc,
                       torch::Tensor all_token_ids,
                       int64_t all_token_ids_stride, torch::Tensor total_len);

void expand_idx_mapping_rdna2(torch::Tensor idx_mapping,
                              torch::Tensor expanded_idx_mapping,
                              torch::Tensor expanded_local_pos,
                              torch::Tensor cu_num_logits);

// ---------------------------------------------------------------------------
// v1 worker Mamba + M-RoPE + structured-output kernels (gfx1030). HIP ports of
// preprocess_mamba_align_fused_kernel / _scatter_num_accepted_kernel /
// _fill_num_accepted_kernel in model_states/mamba_hybrid.py,
// precopy_mamba_align_fused_kernel / postprocess_mamba_fused_kernel /
// batch_memcpy_kernel in mamba_utils.py, _prepare_rope_positions_kernel in
// gpu/mm/rope.py and _apply_grammar_bitmask_kernel in gpu/structured_outputs.py.
// ---------------------------------------------------------------------------

void preprocess_mamba_align_fused_rdna2(
    torch::Tensor idx_mapping,  // [num_reqs] int32
    torch::Tensor state_idx,    // [max_num_reqs] int32 (in/out)
    torch::Tensor num_computed_tokens, torch::Tensor query_start_loc,
    torch::Tensor num_accepted_tokens,  // [max_num_reqs] int32 (in/out)
    torch::Tensor src_col,              // [max_num_reqs] int32 (out)
    torch::Tensor src_off,              // [max_num_reqs] int32 (out)
    int64_t num_reqs, int64_t mamba_block_size);

void scatter_num_accepted_rdna2(torch::Tensor idx_mapping,
                                torch::Tensor num_sampled,
                                torch::Tensor num_accepted, int64_t num_reqs);

void fill_num_accepted_rdna2(torch::Tensor idx_mapping,
                             torch::Tensor num_accepted, int64_t value,
                             int64_t num_reqs);

void precopy_mamba_align_fused_rdna2(
    torch::Tensor mamba_state_idx, torch::Tensor src_col,
    torch::Tensor token_bias, torch::Tensor block_table_ptrs,
    int64_t block_table_stride_req, torch::Tensor state_base_addrs,
    torch::Tensor state_block_strides, torch::Tensor state_elem_sizes,
    torch::Tensor state_inner_sizes, torch::Tensor state_conv_widths,
    torch::Tensor state_group_indices, torch::Tensor state_dim_row_count,
    torch::Tensor state_dim_row_stride,
    const std::optional<torch::Tensor>& idx_mapping, int64_t num_reqs,
    bool conv_state_dim_first, int64_t temporal_tiles);

void postprocess_mamba_fused_rdna2(
    torch::Tensor num_accepted_tokens, torch::Tensor mamba_state_idx,
    const std::optional<torch::Tensor>& num_scheduled_tokens,
    torch::Tensor num_computed_tokens,
    const std::optional<torch::Tensor>& num_draft_tokens,
    torch::Tensor block_table_ptrs, int64_t block_table_stride_req,
    torch::Tensor state_base_addrs, torch::Tensor state_block_strides,
    torch::Tensor state_elem_sizes, torch::Tensor state_inner_sizes,
    torch::Tensor state_conv_widths, torch::Tensor state_group_indices,
    torch::Tensor state_dim_row_count, torch::Tensor state_dim_row_stride,
    torch::Tensor num_accepted_tokens_out,
    const std::optional<torch::Tensor>& idx_mapping, int64_t num_reqs,
    int64_t block_size, bool conv_state_dim_first,
    bool precomputed_new_computed, int64_t temporal_tiles);

void batch_memcpy_rdna2(torch::Tensor src_ptrs, torch::Tensor dst_ptrs,
                        torch::Tensor sizes);

void prepare_rope_positions_rdna2(
    torch::Tensor positions, int64_t positions_stride,
    torch::Tensor prefill_positions, int64_t prefill_stride0,
    int64_t prefill_stride1, torch::Tensor prefill_delta,
    torch::Tensor idx_mapping, torch::Tensor query_start_loc,
    torch::Tensor prefill_lens, torch::Tensor num_computed_tokens,
    int64_t num_dims);

void apply_grammar_bitmask_rdna2(torch::Tensor logits, int64_t logits_stride,
                                 torch::Tensor logits_indices,
                                 torch::Tensor cu_num_logits,
                                 torch::Tensor bitmask,
                                 int64_t bitmask_stride, int64_t vocab_size,
                                 int64_t mask_stride);

// ---------------------------------------------------------------------------
// Speculative decoding (gfx1030). HIP ports of the spec-decode Triton kernels:
// _prepare_prefill_inputs_kernel, _prepare_decode_inputs_kernel,
// _update_draft_inputs_kernel, _gather_draft_sampled_kernel,
// _flatten_sampled_kernel, _compute_local_logits_stats_kernel,
// _rejection_kernel, _resample_kernel and _insert_resampled_kernel.
// ---------------------------------------------------------------------------
void spec_prepare_prefill_inputs_rdna2(
    torch::Tensor last_token_indices, torch::Tensor draft_current_step,
    torch::Tensor draft_input_ids, torch::Tensor draft_positions,
    torch::Tensor draft_query_start_loc, torch::Tensor draft_seq_lens,
    torch::Tensor target_input_ids, torch::Tensor target_positions,
    torch::Tensor idx_mapping, torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
    torch::Tensor num_rejected, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, int64_t max_num_reqs);

void spec_prepare_decode_inputs_rdna2(
    torch::Tensor draft_tokens, torch::Tensor target_seq_lens,
    torch::Tensor num_rejected, torch::Tensor input_ids,
    torch::Tensor positions, torch::Tensor sample_src_positions,
    torch::Tensor query_start_loc, torch::Tensor seq_lens,
    int64_t max_model_len, int64_t max_num_reqs, bool advance_draft_positions);

void spec_update_draft_inputs_rdna2(
    torch::Tensor output_draft_tokens, torch::Tensor next_input_hidden_states,
    torch::Tensor input_ids, torch::Tensor positions,
    torch::Tensor sample_src_positions, torch::Tensor seq_lens,
    torch::Tensor draft_tokens, torch::Tensor current_draft_step,
    torch::Tensor hidden_states, int64_t hidden_size, int64_t max_model_len,
    int64_t num_speculative_steps, bool advance_draft_positions);

void spec_gather_draft_sampled_rdna2(
    torch::Tensor draft_sampled, torch::Tensor pos, torch::Tensor input_ids,
    torch::Tensor positions, torch::Tensor logits_indices,
    torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
    torch::Tensor prefill_len);

void spec_flatten_sampled_rdna2(torch::Tensor flat_sampled, torch::Tensor sampled,
                           torch::Tensor num_sampled,
                           torch::Tensor cu_num_logits);

void spec_compute_local_logits_stats_rdna2(
    torch::Tensor target_local_argmax, torch::Tensor target_local_max,
    torch::Tensor target_local_sumexp, torch::Tensor draft_local_max,
    torch::Tensor draft_local_sumexp, torch::Tensor target_logits,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
    torch::Tensor temp, int64_t vocab_size, int64_t num_speculative_steps,
    bool has_draft_logits);

void spec_rejection_rdna2(
    torch::Tensor sampled, torch::Tensor rejected_steps,
    torch::Tensor target_rejected_logsumexp,
    torch::Tensor draft_rejected_logsumexp, torch::Tensor target_logits,
    torch::Tensor target_local_argmax, torch::Tensor target_local_max,
    torch::Tensor target_local_sumexp, torch::Tensor draft_sampled,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor draft_local_max, torch::Tensor draft_local_sumexp,
    torch::Tensor cu_num_logits, torch::Tensor idx_mapping,
    torch::Tensor temperature, torch::Tensor seed, torch::Tensor pos,
    const std::optional<torch::Tensor>& synthetic_conditional_rates,
    const std::optional<torch::Tensor>& cumulative_log_p,
    const std::optional<torch::Tensor>& local_residual_mass,
    int64_t vocab_num_blocks, int64_t padded_vocab_num_blocks,
    bool has_draft_logits, bool synthetic_mode, bool use_block_verification);

void spec_resample_rdna2(
    torch::Tensor resampled_local_argmax, torch::Tensor resampled_local_max,
    torch::Tensor target_logits, torch::Tensor target_rejected_logsumexp,
    const std::optional<torch::Tensor>& draft_logits,
    torch::Tensor draft_rejected_logsumexp, torch::Tensor rejected_step,
    torch::Tensor cu_num_logits, torch::Tensor expanded_idx_mapping,
    torch::Tensor draft_sampled, torch::Tensor temperature, torch::Tensor seed,
    torch::Tensor pos, const std::optional<torch::Tensor>& cumulative_log_p,
    int64_t vocab_size, bool has_draft_logits, bool use_fp64,
    bool use_block_verification);

void spec_insert_resampled_rdna2(
    torch::Tensor sampled, torch::Tensor num_sampled,
    torch::Tensor resampled_local_argmax, torch::Tensor resampled_local_max,
    int64_t resample_num_blocks, torch::Tensor cu_num_logits,
    torch::Tensor expanded_idx_mapping, torch::Tensor temperature);

