# Fork feature inventory before the v0.32.0 move

`v0.31.0..rdna_extras` = 453 non-merge commits: 258 code-bearing, 195 docs/tools/tests/style. Each code-bearing commit is a checklist item; tick it only after its change is present in the new tree (file content, not just the subject).

## csrc/rocm  (32 commits)

- [x] `8303cd26c2` feat(rdna2): let the GDN prefill kernel take an fp16 SSM state
      files: csrc/rocm/gdn_prefill_delta_h_rdna2.cu
- [x] `e9aa709943` perf(rdna): stop wasting 15/16 of the qsa_store_cache_rows block
      files: csrc/rocm/qsa_rdna2.cu
- [x] `ee7effd1ad` perf(rdna2): run 5 bpw EXL3 layers on the decode-once GEMM
      files: csrc/rocm/exl3_dot2_dense.cu
- [x] `b7ce84ff06` monokernel: clang-format the RDNA2 MonoKernel sources and their op declarations
      files: csrc/rocm/mono_gdn_rdna2.cu, csrc/rocm/mono_k2_rdna2.cu, csrc/rocm/mono_qsa_decode_rdna2.cu, csrc/rocm/ops.h …
- [x] `6d4400f8aa` monokernel: fix the HIP QSA decode build after the empty-split skip
      files: csrc/rocm/mono_qsa_decode_rdna2.cu
- [x] `e8549f3c65` monokernel: HIP QSA decode skips splits that start on padding; P2/P3 rows4 switches (bench)
      files: csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/qsa_decode_rdna2.cu
- [x] `1516bb80c3` perf(rdna2): batch four heads per wave in the M-RoPE kernel to expose load parallelism
      files: csrc/rocm/mrope_rdna2.cu
- [x] `d72d903544` perf(rdna2): double the QSA expand_indices kernel with a constexpr ratio and 4 cols/thread
      files: csrc/rocm/qsa_rdna2.cu
- [x] `c0b272cb6d` perf(rdna2): give the GDN delta_h recurrence two workgroups per CU
      files: csrc/rocm/gdn_prefill_delta_h_rdna2.cu
- [x] `ca9c846864` perf(rdna2): make the GDN prefill prep kernel DRAM-bound instead of scheduler-bound
      files: csrc/rocm/gdn_prefill_prep_rdna2.cu
- [x] `2542dd38e0` perf(rdna2): widen the GDN prefill-o V tile and issue every load before one barrier
      files: csrc/rocm/gdn_prefill_o_rdna2.cu
- [x] `c367351368` perf(rdna2): vectorize kkt staging and match the reference dot exactly
      files: csrc/rocm/gdn_prefill_kkt_rdna2.cu
- [x] `2cbbb4df7a` monokernel Stage D: one system fence per WG and a single flag poller (grid barrier releases the rest)
      files: csrc/rocm/mono_ffn_rdna2.cu
- [x] `6d8e35b8fc` perf(rdna2): cut the solve_wy critical path; the GDN chain now beats Triton
      files: csrc/rocm/gdn_prefill_solve_wy_rdna2.cu
- [x] `23aa43f6c4` rdna(monokernel): 16-row tiles for the HC-down / router GEMVs up to 16 rows
      files: csrc/rocm/mono_ffn_rdna2.cu
- [x] `dbf2f0a1ca` rdna(monokernel): drop the unused P5 partial-sum LDS buffer
      files: csrc/rocm/mono_ffn_rdna2.cu
- [x] `ecec5b8b13` perf(rdna): qsa_mqa_paged — vectorize the dense path and break the fdot2 chain
      files: csrc/rocm/qsa_rdna2.cu
- [x] `01cc99a4d2` perf(rdna): store o's b_A tile as fp16 in LDS
      files: csrc/rocm/gdn_prefill_o_rdna2.cu
- [x] `cf1f028fae` perf(rdna): partial unroll for kkt — 253 VGPRs was one workgroup per CU
      files: csrc/rocm/gdn_prefill_kkt_rdna2.cu
- [x] `d2cb5e8b35` perf(rdna): stop wasting 15/16 of the qsa_store_cache_rows block
      files: csrc/rocm/qsa_rdna2.cu
- [x] `d95ac73a69` perf(rdna): fuse gdn_prefill_delta_h v-correction into the h-update loop
      files: csrc/rocm/gdn_prefill_delta_h_rdna2.cu
- [x] `571aee58b0` fix(rdna): gdn_prefill_o LDS alias race — nondeterministic output
      files: csrc/rocm/gdn_prefill_o_rdna2.cu
- [x] `8abf4c9719` fix(rocm): moe_w4a8_rdna2 builds for every RDNA arch
      files: csrc/rocm/moe_w4a8_rdna2.cu
- [x] `f419d8d6c0` perf(rdna2): prefetch the next tile words in the EXL3 v3 GEMM
      files: csrc/rocm/exl3_dot2_dense.cu
- [x] `44a8cc76a2` perf(rdna): tiled v_dot2 paged MQA logits for the DSV4 indexer decode
      files: csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu
- [x] `0f596e6e0e` perf(rdna): split the decode mHC pre GEMM across blocks, parallel Sinkhorn
      files: csrc/rocm/rdna/dsv4/mhc.cu
- [x] `395d283c24` perf(rocm): branch-free E2M1 decode in the RDNA2 MXFP4 GEMMs (10-14x)
      files: csrc/rocm/mxfp4_dot2_common.cuh, csrc/rocm/mxfp4_dot2_dense.cu, csrc/rocm/mxfp4_dot2_moe.cu
- [x] `bd96530999` perf(rocm): 7-38x faster gemm_w8a16_fp8_dense at decode shapes
      files: csrc/rocm/q_gemm_w8a16_fp8_rdna2.cu
- [x] `6f5333e868` fix(rocm): sparse_mla_prefill_rdna2 staged rows out of bounds
      files: csrc/rocm/rdna/dsv4/sparse_mla_decode.cu
- [x] `b08d5dc7bd` fix(rocm): keep mrope kernel name on the launch line for hipify
      files: csrc/rocm/mrope_rdna2.cu
- [x] `523eb846e0` fix(rdna2): MXFP4 dense/MoE kernels: accept 32/8-bit weights and e8m0 scales; fix dequant
      files: csrc/rocm/mxfp4_dot2_common.cuh, csrc/rocm/mxfp4_dot2_dense.cu, csrc/rocm/mxfp4_dot2_moe.cu
- [x] `cd100da8cd` feat(rocm): port Aron Hsiao's kernels-csrc changes (3 commits)
      files: csrc/rocm/rdna_fused_glue.cu, csrc/rocm/skinny_gemms_int4.cu

## vllm/model_executor  (20 commits)

- [x] `d77eede824` cleanup(rdna): drop unread MoE scratch and dead AWQ/RMSNorm paths
      files: vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py, vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py, vllm/model_executor/layers/fused_moe/moe_align_block_size.py, vllm/model_executor/layers/layernorm.py …
- [x] `7ba7a07926` cleanup(quant): delete unimported RDNA2 W8A16-FP8 linear kernel
      files: vllm/model_executor/kernels/linear/scaled_mm/rdna2_w8a16_fp8.py
- [x] `b3cbb64031` perf(rdna2): skip the Triton JIT warmup on gfx10x
      files: vllm/model_executor/warmup/kernel_warmup.py
- [x] `aef30748ae` fix(rdna): the GDN timing probe must not sync per region
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `52bc48d82f` diag(rdna): env-gated per-op CUDA-event timing for the GDN prefill path
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `2e6e7395be` perf(rdna): reuse GDN prefill scratch buffers instead of zeroing per call
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `bce000842d` fix(rdna): build int8 dense shadows in row chunks
      files: vllm/model_executor/layers/rdna_dense_int8.py
- [x] `7e2df85aea` perf(rdna2): int4 W4A16 MoE Triton config for E=512,N=640 on V620
      files: vllm/model_executor/layers/fused_moe/configs/E=512,N=640,device_name=AMD_Radeon_Pro_V620,dtype=int4_w4a16.json
- [x] `98c813b90b` perf(rdna2): W4A8 MoE is prefill-only (>= 33 tokens), like the dense path
      files: vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py, vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py
- [x] `ab72778560` fix(rdna): allocate the GDN opaque-forward output in the graph
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `0095c9d00b` feat(rdna): VLLM_RDNA_GDN_OPAQUE=0 keeps GDN projections in the compiled graph
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `f7d4fa738c` fix(rdna): warm up the GDN decode and MTP-verify kernels at profile time
      files: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `7961afd29b` perf(rdna2): fused clamped SwiGLU in the MXFP4 MoE experts
      files: vllm/model_executor/layers/fused_moe/experts/rdna2_mxfp4_moe.py
- [x] `16e890dcea` perf(rdna): pick the RDNA2 MXFP4 MoE block size from the token count
      files: vllm/model_executor/layers/fused_moe/experts/rdna2_mxfp4_moe.py
- [x] `5eb773b656` fix(rocm): apply swiglu_limit in the RDNA2 MXFP4 MoE activation
      files: vllm/model_executor/layers/fused_moe/experts/rdna2_mxfp4_moe.py
- [x] `dc36062823` fix(mhc): torch mHC pre accepts fp16 residuals
      files: vllm/model_executor/kernels/mhc/torch.py
- [x] `30f72c2291` fix(rocm): gate the BF16x3 router GEMM warmup on CUDA
      files: vllm/model_executor/warmup/kernel_warmup.py
- [x] `3f3776778b` fix(rdna): dense W4A16 dequant reads packed 4-bit zero-points
      files: vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py
- [x] `e5ecbfd20a` fix(rdna2): restore K-first MoE weight layout after upstream #52798
      files: vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py
- [x] `730697976e` fix(models): detect unquantized MTP weights from checkpoint metadata
      files: vllm/model_executor/models/qwen3_5_mtp.py

## csrc/rocm, tests/kernels  (18 commits)

- [x] `e9345a4255` perf(rdna2): HIP vision-tower bilinear pos-embed
      files: csrc/rocm/vit_pos_embed_rdna2.cu, tests/kernels/core/test_vit_pos_embed_rdna2.py, vllm/model_executor/models/qwen3_vl.py
- [x] `c0cc7d4c8b` perf(rdna2): HIP unquantized MoE backend for the MTP drafter
      files: csrc/rocm/moe_unquantized_rdna2.cu, tests/kernels/moe/test_rdna2_unquantized_moe.py, tests/kernels/moe/test_unquantized_backend_selection.py, vllm/model_executor/layers/fused_moe/experts/rdna2_unquantized_moe.py …
- [x] `b9bb79bb5b` perf(rdna2): HIP conv1d spec update + GDN spec-verify
      files: csrc/rocm/causal_conv1d_rdna2.cu, csrc/rocm/gdn_decode_rdna2.cu, tests/kernels/mamba/test_causal_conv1d_update_rdna2.py, tests/kernels/quantization/test_gdn_decode_rdna2.py …
- [x] `6743c40057` fix(rdna2): let the HIP causal conv1d run with an MTP-sized conv state
      files: csrc/rocm/causal_conv1d_rdna2.cu, tests/kernels/mamba/test_causal_conv1d_update_rdna2.py, vllm/model_executor/layers/mamba/ops/causal_conv1d.py
- [x] `eaea4f97fd` perf(rdna2): run the QSA sparse attention HIP kernel at head_dim 256
      files: csrc/rocm/qsa_rdna2.cu, tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py
- [x] `2a33c9bdb7` feat(rdna2): HIP port of the QSA sparse paged attention (all 3 Triton leftovers)
      files: csrc/rocm/ops.h, csrc/rocm/qsa_rdna2.cu, csrc/rocm/torch_bindings.cpp, tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py …
- [x] `221782e69d` feat(rdna2): HIP port of the QSA index-expansion kernel (1 of 3 Triton leftovers)
      files: csrc/rocm/ops.h, csrc/rocm/qsa_rdna2.cu, csrc/rocm/torch_bindings.cpp, tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py …
- [x] `e7e63a0a75` perf(rdna2): gemv_f16_rdna2 for MTP verify batches (M <= 24)
      files: csrc/rocm/skinny_gemms_int4.cu, tests/kernels/quantization/test_gemv_f16_rdna2.py, tools/rdna/port_v031/lm_head_bench.py, vllm/model_executor/layers/quantization/exl3.py …
- [x] `1467ee7dc4` fix(rdna2): index the GDN conv1d update state through its strides
      files: csrc/rocm/causal_conv1d_rdna2.cu, tests/kernels/mamba/test_causal_conv1d_update_rdna2.py, tools/rdna/port_v031/unit.sh, vllm/model_executor/layers/mamba/ops/causal_conv1d.py
- [x] `20ab45181d` perf(rdna): fused tiled MQA logits for the DSV4 indexer prefill
      files: csrc/rocm/ops.h, csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu, csrc/rocm/torch_bindings.cpp, tests/kernels/attention/rdna/dsv4/test_attention_ops.py …
- [x] `0835e3cbf5` perf(rdna2): run the EXL3 6 bpw lm_head on the decode-once GEMM
      files: csrc/rocm/exl3_dot2_dense.cu, tests/kernels/quantization/test_exl3_rdna2.py, tools/rdna/port_v031/exl3_gemm_bench.py
- [x] `c2cbf00477` perf(rdna2): EXL3 decode-once GEMM for decode and MTP verify batches
      files: csrc/rocm/exl3_dot2_dense.cu, tests/kernels/quantization/test_exl3_rdna2.py, tools/rdna/port_v031/exl3_gemm_bench.py
- [x] `5148ba7817` perf(rdna2): branch-free E4M3 decode in qdq_fp8_rdna2.cuh
      files: csrc/rocm/qdq_fp8_rdna2.cuh, tests/kernels/quantization/test_rdna2_w8a8_fp8_dense.py, tools/rdna/port_v031/unit.sh
- [x] `9adea88807` perf(rdna): split-KV sparse_mla_decode_rdna2 + branch-free fp8 decode
      files: csrc/rocm/rdna/dsv4/sparse_mla_decode.cu, tests/kernels/attention/rdna/dsv4/test_kv_insert.py
- [x] `9504d8091c` fix(rocm): RDNA2 fp8 E4M3 decode treated the 256..448 binade as Inf/NaN
      files: csrc/rocm/qdq_fp8_rdna2.cuh, tests/kernels/attention/rdna/dsv4/test_attention_ops.py
- [x] `7df02835a2` feat(rdna): DSV4 attention ops in fp16 on gfx1030 (plan steps 3-4)
      files: csrc/rocm/ops.h, csrc/rocm/qsa_rdna2.cu, csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu, csrc/rocm/torch_bindings.cpp …
- [x] `7b98e3f851` fix(rocm): correct E2M1 fp16 bit patterns in the RDNA2 MXFP4 LUT
      files: csrc/rocm/mxfp4_dot2_common.cuh, tests/kernels/quantization/test_mxfp4_rdna2.py
- [x] `8b06c28dfb` fix(rocm): return fresh output tensors from RDNA2 GEMM/attention ops
      files: csrc/rocm/fa_rdna2.cu, csrc/rocm/q_gemm_rdna2.cu, csrc/rocm/q_gemm_rdna2_prefill.cu, csrc/rocm/skinny_gemms.cu …

## vllm/models  (13 commits)

- [x] `8559a60424` perf(rdna2): the HIP QSA path uses #126's indexer epilogue and HIP scoring
      files: vllm/models/qwen4_exp/amd/indexer_qsa.py, vllm/models/qwen4_exp/amd/ops/qsa.py
- [x] `0465e92351` fix(rdna2): --monokernel keeps its QSA indexer scoring ahead of #99's HIP route
      files: vllm/models/qwen4_exp/amd/ops/qsa.py
- [x] `b233ddf4ed` monokernel: log the QSA part's engagement once per decode width
      files: vllm/models/qwen4_exp/amd/mono/rdna2/qsa.py
- [x] `71c476642f` monokernel E3: zero only the padding rows of the QSA output (rdna_mono_qsa)
      files: vllm/models/qwen4_exp/amd/qsa.py
- [x] `e5bccd81a1` fix(rdna2): keep the Triton QSA metadata builder; the torch one costs 13% decode
      files: vllm/models/qwen4_exp/common/qsa_cache.py
- [x] `c3256d1381` fix(rdna2): build QSA metadata without Triton under the HIP QSA gate
      files: vllm/models/qwen4_exp/common/qsa_cache.py
- [x] `5976dc4b12` fix(qwen4_exp): MTP drafter branches on inputs, not PP rank
      files: vllm/models/qwen4_exp/amd/mtp.py
- [x] `59d8d6b067` perf(rdna): DSV4 C128 compressor decode on the two-stage split
      files: vllm/models/deepseek_v4/compressor.py
- [x] `e685911d58` perf(rdna): DSV4 wo_a on the W8A16 fp8 kernel per group
      files: vllm/models/deepseek_v4/amd/rdna/attention.py
- [x] `c44db43e29` perf(rdna): DSV4 compressor score GEMMs on the RDNA2 fp16 GEMV
      files: vllm/models/deepseek_v4/amd/rocm.py, vllm/models/deepseek_v4/attention.py
- [x] `6093e3a899` feat(rdna): DeepseekV4RDNAAttention for fp16 DeepSeek-V4 on RDNA (plan step 5)
      files: vllm/models/deepseek_v4/amd/model.py, vllm/models/deepseek_v4/amd/rdna/attention.py
- [x] `7b10818d0a` fix(deepseek_v4): load checkpoints with fused routed-expert projections
      files: vllm/models/deepseek_v4/amd/model.py
- [x] `0f592bea03` fix(qwen4_exp): accept compressed-tensors checkpoints for the PLE table
      files: vllm/models/qwen4_exp/common/ngram_embedding.py

## CMakeLists.txt, csrc/rocm  (9 commits)

- [x] `8dec4520e4` feat(rdna): INT4 KV cache with FP16 per-32-channel block scales in FA-RDNA2
      files: CMakeLists.txt, csrc/rocm/fa_rdna2.cu, csrc/rocm/int4_fp16_kv_cache.cu, csrc/rocm/ops.h …
- [x] `cc1ef7c361` monokernel: --monokernel switch and upstream MonoKernel layout for the RDNA2 parts
      files: CMakeLists.txt, csrc/rocm/mono_gdn_rdna2.cu, csrc/rocm/mono_k2_rdna2.cu, csrc/rocm/mono_qsa_decode_rdna2.cu …
- [x] `770183e4de` perf(rdna2): make the HIP kernel set opt-in behind VLLM_RDNA_ALL_HIP
      files: CMakeLists.txt, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, docs/rdna2/tuning.md …
- [x] `4664fd31fa` monokernel E3: QSA decode indexer bound + HIP sparse decode attention (rdna_mono_qsa)
      files: CMakeLists.txt, csrc/rocm/ops.h, csrc/rocm/qsa_decode_rdna2.cu, csrc/rocm/torch_bindings.cpp …
- [x] `709f7d3cb2` monokernel E1': fused GDN decode core (conv + delta rule + gated norm) in one launch
      files: CMakeLists.txt, csrc/rocm/gdn_fused_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp …
- [x] `616072cdc9` rdna(monokernel): 2a M2 integration behind additional_config rdna_mono_ffn
      files: CMakeLists.txt, csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp …
- [x] `ec349e9079` perf(rdna): fused HIP mHC pre/post for DeepSeek-V4 on RDNA
      files: CMakeLists.txt, csrc/rocm/ops.h, csrc/rocm/rdna/dsv4/mhc.cu, csrc/rocm/torch_bindings.cpp …
- [x] `01b439bfd2` feat(rdna): DSV4 q-norm + RoPE + fp8_ds_mla KV insert HIP kernel (fp16/bf16)
      files: CMakeLists.txt, csrc/rocm/ops.h, csrc/rocm/rdna/dsv4/kv_insert.cu, csrc/rocm/torch_bindings.cpp …
- [x] `3e9ca702a1` refactor(rdna): csrc/rocm/rdna layout skeleton, move DSV4 kernels to rdna/dsv4
      files: CMakeLists.txt, csrc/rocm/rdna/common/arch.cuh, csrc/rocm/rdna/common/convert.cuh, csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu …

## csrc/rocm, tools/monokernel-2a  (8 commits)

- [x] `7d0726596d` monokernel Stage D: the FFN all-reduce inside the persistent kernel (rdna_mono_ffn_ar)
      files: csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/qsa_decode_rdna2.cu, csrc/rocm/rdna_allreduce.cu …
- [x] `11d70cda3b` monokernel: compile the E1a GDN mode out of the persistent kernel (it slowed the FFN path ~25%)
      files: csrc/rocm/mono_ffn_rdna2.cu, tools/monokernel-2a/serve_2a.sh
- [x] `bd26e5c1ec` rdna(monokernel): E0 attention profile tools, coarse workspace, E1a GDN mode (reference)
      files: csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, tools/monokernel-2a/attn_breakdown.py …
- [x] `b1a6689041` rdna(monokernel): serve timing adds effective SCLK and per-rank routing load stats
      files: csrc/rocm/mono_ffn_rdna2.cu, tools/monokernel-2a/check_op.py, vllm/models/qwen4_exp/amd/mono_ffn.py
- [x] `191c8448f6` rdna(monokernel): serve timing splits the attention core by layer type (QSA / GDN)
      files: csrc/rocm/mono_ffn_rdna2.cu, tools/monokernel-2a/check_op.py, vllm/models/qwen4_exp/amd/mono_ffn.py
- [x] `b0526a3dd0` rdna(monokernel): K1 attention-side HC preamble on the persistent kernel
      files: csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, tools/monokernel-2a/after_bench.py …
- [x] `febd6b5273` rdna(monokernel): 2a kernel rework, all cudagraph modes, serve timing split
      files: csrc/rocm/mono_ffn_rdna2.cu, tools/monokernel-2a/after_bench.py, tools/monokernel-2a/check_op.py, tools/monokernel-2a/clock_window.py …
- [x] `c90db32e28` rdna(monokernel): 2a kernel at 16 waves/CU, LDS-tiled HC/router GEMVs
      files: csrc/rocm/mono_ffn_rdna2.cu, tools/monokernel-2a/ext_binding.cpp, tools/monokernel-2a/m1_bench.py, tools/monokernel-2a/repeat_check.py …

## vllm/v1  (7 commits)

- [x] `dc619cc327` fix(rdna2): keep INT4 decode rows on the split-K kernel in a mixed batch
      files: vllm/v1/attention/backends/rdna_attn.py
- [x] `cf578e3a95` refactor(rdna2): share the INT4 KV layout helpers with the RDNA_ATTN backend
      files: vllm/v1/attention/backends/rdna_attn.py
- [x] `ad356d1135` fix(rdna): size GDN state arenas by the decode batch, not the capture ladder
      files: vllm/v1/attention/backends/gdn_attn.py
- [x] `36c4164b22` fix(rdna): block-aligned mamba splits only for MTP and Flash-Next on V2
      files: vllm/v1/core/sched/scheduler.py
- [x] `bda1589fc6` fix(rdna): mamba block-aligned splits for all V2-runner hybrids at TP>2
      files: vllm/v1/core/sched/scheduler.py
- [x] `dd85ff28b3` feat(rdna2): adaptive prefill scheduling (dynamic chunk cap + step cadence)
      files: vllm/v1/core/sched/dynamic_prefill.py, vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
- [x] `6a314f5dd0` feat(rdna2): port recipe Triton LDS/softmax, skinny MoE GEMV, MTP-under-PP
      files: vllm/v1/attention/backends/triton_attn.py, vllm/v1/attention/ops/triton_unified_attention.py

## tests/kernels, vllm/model_executor  (7 commits)

- [x] `0a1beaa370` revert(rdna2): keep WNA16 tile8 prefill cutoff at 4096 until paired serving proves it
      files: tests/kernels/quantization/test_rdna2_moe_w4a16.py, vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py
- [x] `15554fa661` fix(rdna2): no shared-experts aux-stream fork inside CUDA-graph capture on gfx10x
      files: tests/kernels/moe/test_shared_experts_capture_fork.py, vllm/model_executor/layers/fused_moe/runner/shared_experts.py
- [x] `2907c75d40` perf(rdna2): W8A16 fp8 linears use dequant + rocBLAS for prefill-sized M
      files: tests/kernels/quantization/test_rdna2_w8a16_fp8_block.py, vllm/model_executor/kernels/linear/scaled_mm/rdna2_w8a16_fp8_block.py, vllm/models/deepseek_v4/amd/rdna/attention.py
- [x] `2cc2fbc7c6` fix(rdna): HIP gated RMSNorm takes the per-head 3D GDN input
      files: tests/kernels/mamba/test_gdn_output_projection.py, vllm/model_executor/layers/layernorm.py
- [x] `6f43b9ef5b` feat(rdna2): port Aron Hsiao's moe changes (2 commits)
      files: tests/kernels/quantization/test_rocm_moe_skinny.py, vllm/model_executor/layers/fused_moe/experts/triton_moe.py, vllm/model_executor/layers/fused_moe/fused_moe.py, vllm/model_executor/layers/fused_moe/rocm_moe_skinny.py
- [x] `a567946e00` feat(rdna2): port Aron Hsiao's quant-linear changes (2 commits)
      files: tests/kernels/quantization/test_rdna_hybrid_w4a16.py, vllm/model_executor/kernels/linear/__init__.py, vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py, vllm/model_executor/layers/linear.py …
- [x] `ccc8ad0884` feat(rdna2): W4A16/W4A8/W8A16/FP8 dense linear paths
      files: tests/kernels/quantization/test_gdn_decode_rdna2.py, tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py, tests/kernels/quantization/test_gdn_prefill_kkt_rdna2.py, tests/kernels/quantization/test_gdn_prefill_o_rdna2.py …

## tests/v1, vllm/v1  (7 commits)

- [x] `c7c984ba7d` [Core] PP: do not lay out other stages layers for an absent uniform group
      files: tests/v1/core/test_kv_cache_utils.py, vllm/v1/core/kv_cache_utils.py
- [x] `67ef7272e9` perf(ple_offload): sidecar fast path for MTP verify steps
      files: tests/v1/ple_offload/test_ple_quant_gather.py, vllm/v1/ple_offload/worker.py
- [x] `f8af9e1b89` fix(rdna): decode-stall cap fits mixed steps only; one bucket may shrink the cap
      files: tests/v1/core/test_decode_stall_bound.py, vllm/v1/core/sched/mixed_step.py
- [x] `5bf02f78ea` fix(mamba): aligned state indices read batch rows, not source slots
      files: tests/v1/worker/test_mamba_utils.py, vllm/v1/worker/gpu/model_states/mamba_hybrid.py, vllm/v1/worker/mamba_utils.py
- [x] `6e56ecb782` fix(rdna): GDN update_block_table keeps the RDNA state-arena layout
      files: tests/v1/attention/test_gdn_metadata_builder.py, vllm/v1/attention/backends/gdn_attn.py
- [x] `d5b33727a2` fix(rocm): port Aron Hsiao's engine changes (2 commits)
      files: tests/v1/spec_decode/test_qwen4_exp.py, vllm/v1/executor/multiproc_executor.py, vllm/v1/executor/uniproc_executor.py, vllm/v1/spec_decode/qwen4_exp.py …
- [x] `f69aaf844f` fix(rocm): port George Muravei-Alkhavoi's engine changes (10 commits)
      files: tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py, tests/v1/core/test_kv_cache_utils.py, tests/v1/core/test_mamba_align_chunk_split.py, tests/v1/core/test_prefix_caching.py …

## tools/rdna  (5 commits)

- [x] `aa1c45887c` fix(rdna2): let the launcher derive the prefill budget for this recipe
      files: tools/rdna/recipes/27b-awq.env
- [x] `aaf48bfdcf` fix(rdna): flashnext-exl3 compiles per rank (no shared AOT cache)
      files: tools/rdna/recipes/flashnext-exl3.env
- [x] `b3c8f7cddd` fix(tools): ENABLE_PREFIX_CACHING=0 passes --no-enable-prefix-caching
      files: tools/rdna/serve_rdna.sh
- [x] `0272b15f7a` Revert "tools(rdna): agentic Flash-Next recipe on the flashnext-mtp0 performance block"
      files: tools/rdna/recipes/flashnext.env
- [x] `36da3f93b8` recipe(rdna): dsv4-flash SEQS=8, KV=6e9
      files: tools/rdna/recipes/dsv4-flash.env

## csrc/rocm, tests/models  (5 commits)

- [x] `02154ac2e7` perf(rdna2): inline persistent MonoKernel barriers
      files: csrc/rocm/mono_k2_rdna2.cu, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py
- [x] `59c139c77a` perf(rdna2): MonoKernel QSA indexer epilogue in one launch + HIP scoring
      files: csrc/rocm/mono_qsa_decode_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py …
- [x] `9f0124970c` perf(rdna2): HIP QSA indexer stores for the MonoKernel, fused with norm+RoPE
      files: csrc/rocm/mono_qsa_decode_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py …
- [x] `1707abe9ea` perf(rdna2): rewrite the HC kernels to use the whole block
      files: csrc/rocm/hc_rdna2.cu, tests/models/qwen4_exp/test_hc_ops_rdna2.py, vllm/models/qwen4_exp/amd/ops/hc_rdna2.py
- [x] `a377f9bb75` perf(rdna2): HIP QSA metadata builder
      files: csrc/rocm/qsa_metadata_rdna2.cu, tests/models/qwen4_exp/test_qsa_metadata_rdna2.py, vllm/models/qwen4_exp/common/qsa_cache.py

## csrc/rocm, docs/rdna2  (5 commits)

- [x] `ff53285a02` Support BF16 noncausal DFlash2 attention in FA-RDNA2
      files: csrc/rocm/fa_rdna2.cu, docs/rdna2/tuning.md, tests/kernels/attention/test_fa_rdna2_writer_layout.py, tools/rdna/recipes/flashnext-dflash2.env …
- [x] `ea5fbdd8c9` perf(rdna2): stride the mrope block over the rotary index
      files: csrc/rocm/mrope_rdna2.cu, docs/rdna2/tuning.md, tests/kernels/test_mrope_rdna2.py
- [x] `34cc3c21bd` perf+fix(rdna): kkt/o LDS bank conflicts; kkt was in the wrong precision
      files: csrc/rocm/gdn_prefill_kkt_rdna2.cu, csrc/rocm/gdn_prefill_o_rdna2.cu, docs/rdna2/qwen4_exp_hip_tracker.md, tools/rdna2/bench_gdn_prefill_chain_rdna2.py
- [x] `3892472c9f` perf(rdna): stage gdn_prefill_delta_h chunks in LDS — HIP now beats Triton
      files: csrc/rocm/gdn_prefill_delta_h_rdna2.cu, docs/rdna2/qwen4_exp_hip_tracker.md
- [x] `3491035be0` perf(rdna): remove the CUDA-graph replay-copy wrapper, use upstream replay
      files: csrc/rocm/fa_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/q_gemm_rdna2.cu, csrc/rocm/rdna2_graph_keepalive.cuh …

## benchmarks/kernels, csrc/rocm  (4 commits)

- [x] `8a09c4190b` refactor(rdna2): move the INT4 KV layout into a shared header
      files: benchmarks/kernels/pr134/build_private_core.sh, csrc/rocm/fa_rdna2.cu, csrc/rocm/int4_kv_layout_common.cuh
- [x] `ddb69e64f4` cleanup(rocm): drop W4A16 prefill experiment hooks
      files: benchmarks/kernels/w4a16_compile_dispatch/test_probe_cpu.py, csrc/rocm/hc_prefill_rdna2.cu, csrc/rocm/q_gemm_rdna2.cu, csrc/rocm/q_gemm_rdna2_prefill.cu …
- [x] `96d30278ef` feat(rdna2): port George Muravei-Alkhavoi's moe changes (4 commits)
      files: benchmarks/kernels/benchmark_rocm_moe_skinny.py, csrc/rocm/moe_resident_decode.cu, tests/quantization/test_moe_wna16.py, tuned-moe/E=128,N=640,device_name=AMD_Radeon_Pro_V620,dtype=int4_w4a16.json …
- [x] `00ac7eec65` feat(rocm): RDNA2 HIP kernel library and op bindings
      files: benchmarks/kernels/benchmark_fa_rdna2_prefill.py, benchmarks/kernels/benchmark_fa_rdna2_split_decode.py, benchmarks/kernels/w4a16_compile_dispatch/__init__.py, benchmarks/kernels/w4a16_compile_dispatch/_layers.py …

## csrc/rocm, tests/distributed  (4 commits)

- [x] `e1e0d21f66` fix(rdna): make the rdna_ar one-shot abort sticky inside the kernel
      files: csrc/rocm/rdna_allreduce.cuh, tests/distributed/test_rdna_ar.py, tests/distributed/test_rdna_ar_gpu.py, vllm/distributed/device_communicators/rdna_all_reduce.py
- [x] `611063c62a` fix(rdna): remove the two-shot all-reduce from rdna_ar
      files: csrc/rocm/rdna_allreduce.cu, csrc/rocm/rdna_allreduce.cuh, tests/distributed/test_rdna_ar.py, vllm/distributed/device_communicators/rdna_all_reduce.py …
- [x] `a43be93b10` fix(rdna): rdna_ar one-shot waits tolerate late peers (wall-clock bound)
      files: csrc/rocm/ops.h, csrc/rocm/rdna_allreduce.cu, csrc/rocm/rdna_allreduce.cuh, csrc/rocm/torch_bindings.cpp …
- [x] `00fcc97ff5` fix(rocm): rdna_ar returns a fresh output tensor per all-reduce
      files: csrc/rocm/rdna_allreduce.cu, tests/distributed/test_rdna_ar_gpu.py

## docs/rdna2, tests/kernels  (4 commits)

- [x] `3e401fdb41` fix(rdna2): PR #99 review fixes, mono-aware QSA precedence, conv1d state opt-in
      files: docs/rdna2/tuning.md, tests/kernels/mamba/test_causal_conv1d_update_rdna2.py, tests/kernels/mamba/test_gdn_rocm_layout_dispatch.py, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py …
- [x] `ebaa70a455` fix: handle dynamic MTP graph shapes and zero-draft state transitions
      files: docs/rdna2/tuning.md, tests/kernels/mamba/test_precopy_mamba_align.py, tests/models/qwen4_exp/test_ple.py, tests/v1/cudagraph/test_cudagraph_manager.py …
- [x] `147329b2fd` perf(rdna2): default the QSA/HC/GDN HIP prefill gates on (S6)
      files: docs/rdna2/qwen4_exp_hip_path.md, docs/rdna2/qwen4_exp_hip_tracker.md, docs/rdna2/tuning.md, tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py …
- [x] `8202cb647b` fix(rdna2): port RDNA2 W4A16 kernels to post-#54809 GPTQ API; adaptive prefill queue
      files: docs/rdna2/v0.31.0-port-notes.md, tests/kernels/quantization/test_rdna2_moe_w4a16.py, tests/kernels/quantization/test_rdna2_moe_w4a8.py, tools/rdna2_028/probe_w4a8_moe.py …

## tools/rdna, vllm/model_executor  (4 commits)

- [x] `e3a4bbef8c` feat(qwen4_exp): load the EXL3 MTP mixer patch; flashnext-exl3 recipe
      files: tools/rdna/recipes/flashnext-exl3.env, tools/rdna/serve_rdna.sh, vllm/model_executor/layers/quantization/exl3_ple.py, vllm/models/qwen4_exp/amd/mtp.py …
- [x] `006bca7317` qwen4_exp: allow pipeline parallelism when PLE layers sit on stage 0
      files: tools/rdna/serve_rdna.sh, vllm/model_executor/models/config.py, vllm/models/qwen4_exp/amd/model_state.py, vllm/models/qwen4_exp/nvidia/model_state.py
- [x] `b074874b54` fix(rdna2): keep the gemv_f16_rdna2 dispatch at M <= 8
      files: tools/rdna/port_v031/gemv_crossover_bench.py, vllm/model_executor/layers/utils.py
- [x] `f41482de7b` feat(rdna2): RDNA2 W4A16 MoE kernel for GPTQ (MoeWNA16) checkpoints
      files: tools/rdna/port_v031/gptq_moe_expert_check.py, vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py, vllm/model_executor/layers/fused_moe/oracle/int_wna16.py, vllm/model_executor/layers/quantization/moe_wna16.py

## tools/monokernel-2a  (4 commits)

- [x] `267249532e` monokernel tools: attn_breakdown --classes (kernel lists for ffn segments too)
      files: tools/monokernel-2a/attn_breakdown.py
- [x] `2173c6e4e3` monokernel tools: wait 90 s after releasing the HIP 6-9 lock before re-acquiring it (lock fairness)
      files: tools/monokernel-2a/rebuild_check.sh, tools/monokernel-2a/run_specs.sh
- [x] `8691bee6cb` monokernel tools: rebuild_check.sh (locked rebuild + standalone FFN / QSA checks between serve stages)
      files: tools/monokernel-2a/rebuild_check.sh
- [x] `8aea29a8a3` monokernel tools: per-serve GPU69.lockdir in run_specs, chain_specs, fetch_results
      files: tools/monokernel-2a/chain_specs.sh, tools/monokernel-2a/fetch_results.sh, tools/monokernel-2a/run_specs.sh

## tools/monokernel-2a, vllm/models  (4 commits)

- [x] `866319ce49` monokernel E3: QSA scoring tiles past the live context exit early (valid in CUDA graphs); HIP sparse decode from 3 rows
      files: tools/monokernel-2a/qsa_bench.py, vllm/models/qwen4_exp/amd/ops/qsa.py
- [x] `660f2f6f2d` monokernel E3: the indexer's three per-token cache stores in one launch (rdna_mono_qsa)
      files: tools/monokernel-2a/qsa_store3_check.py, vllm/models/qwen4_exp/amd/indexer_qsa.py, vllm/models/qwen4_exp/amd/ops/qsa.py
- [x] `2f2dc52a27` monokernel E3: indexer GemmaRMSNorm + MRoPE in one launch (rdna_mono_qsa)
      files: tools/monokernel-2a/qsa_normrope_check.py, tools/monokernel-2a/run_specs.sh, vllm/models/qwen4_exp/amd/indexer_qsa.py
- [x] `f546402ab2` monokernel E3: HIP sparse decode from 8 rows; bench amortizes the graph-launch floor
      files: tools/monokernel-2a/graph_floor.py, tools/monokernel-2a/qsa_bench.py, vllm/models/qwen4_exp/amd/ops/qsa.py

## tunableop/README.md, tunableop/profiles.json  (4 commits)

- [x] `52f7358e2d` tunableop(rdna2): re-measure the whole grid, cover the dyn-prefill 256/512 chunks
      files: tunableop/README.md, tunableop/profiles.json, tunableop/rocm10-rocblas5.6/README.md, tunableop/rocm10-rocblas5.6/provenance.json …
- [x] `040b4afeb1` tunableop: qualify rocm10-rocblas5.6 against the build we actually load
      files: tunableop/README.md, tunableop/profiles.json, tunableop/rocm10-rocblas5.6/README.md, tunableop/rocm10-rocblas5.6/provenance.json …
- [x] `9566ee502e` tunableop: qualify rocm10-rocblas5.6 against the build we actually load
      files: tunableop/README.md, tunableop/profiles.json, tunableop/rocm10-rocblas5.6/README.md, tunableop/rocm10-rocblas5.6/provenance.json …
- [x] `4958287a49` tunableop: rows for 8192-token prefill chunks (Flash-Next MTP2)
      files: tunableop/README.md, tunableop/profiles.json, tunableop/rocm7.14-rocblas5.5/provenance.json, tunableop/rocm7.14-rocblas5.5/tunableop_results0.csv …

## tools/rdna2  (3 commits)

- [x] `5d68d933ab` fix(rdna2): bench the sparse attention at the production 6-q-head geometry
      files: tools/rdna2/bench_qsa_sparse_rdna2.py
- [x] `12482b1de7` bench(rdna2): run the gate A/B on the served chunks and the long-context shapes
      files: tools/rdna2/bench_serve_gates_ab.sh
- [x] `dd444ebdba` bench(rdna2): sweep the QSA indexer ops and the GDN chain at served shapes
      files: tools/rdna2/bench_gdn_prefill_chain_rdna2.py, tools/rdna2/bench_qsa_rdna2.py

## vllm/config  (3 commits)

- [x] `7bb0701f21` monokernel E1': make qwen_gdn_attention_core_mono a splitting op
      files: vllm/config/compilation.py
- [x] `eb503cd3ca` Revert "fix(rdna): keep JIT kernel warmup under --enforce-eager on RDNA"
      files: vllm/config/vllm.py
- [x] `d11df8c312` fix(rdna): keep JIT kernel warmup under --enforce-eager on RDNA
      files: vllm/config/vllm.py

## tests/models, vllm/model_executor  (3 commits)

- [x] `1c1e0ecd9a` fix(rdna2): route M-RoPE to the gfx1030 HIP kernel instead of Triton
      files: tests/models/qwen4_exp/test_qsa_amd.py, vllm/model_executor/layers/rotary_embedding/mrope.py, vllm/models/qwen4_exp/amd/indexer_qsa.py
- [x] `0a9a86564f` feat(qwen4_exp): allow pipeline parallelism with PLE on stage 0 (RDNA)
      files: tests/models/qwen4_exp/test_config.py, tests/models/qwen4_exp/test_ple_pp.py, vllm/model_executor/models/config.py, vllm/models/qwen4_exp/amd/model_state.py …
- [x] `92b2cb1ca1` feat(rdna2): port Aron Hsiao's qwen4exp-dsv4 changes (2 commits)
      files: tests/models/qwen4_exp/test_qsa_amd.py, tests/models/qwen4_exp/test_qsa_reference.py, vllm/model_executor/models/deepseek_mtp.py, vllm/models/qwen4_exp/amd/hyperconnection.py …

## tests/models, vllm/models  (3 commits)

- [x] `e5d74cb595` [Bugfix] Fix Qwen4Exp PLE embedding rejecting INC (AutoRound) checkpoints (#59990)
      files: tests/models/qwen4_exp/test_ple.py, vllm/models/qwen4_exp/common/ngram_embedding.py
- [x] `b637fb5429` fix(qwen4_exp): bounded-memory PLE short-conv prefill
      files: tests/models/qwen4_exp/test_ple_prefill_conv.py, vllm/models/qwen4_exp/amd/ple_layer.py
- [x] `d5de41ec55` feat(qwen4_exp): INT4 group-16 PLE table from VLLM_PLE_QUANT_DIR on the upstream PLE path
      files: tests/models/qwen4_exp/test_ple.py, vllm/models/qwen4_exp/amd/ple_layer.py, vllm/models/qwen4_exp/common/ngram_embedding.py, vllm/models/qwen4_exp/nvidia/ngram_embedding.py

## vllm/model_executor, vllm/models  (3 commits)

- [x] `72e760942b` feat(qwen4_exp): serve the INT4 PLE table from disk through the CPU sidecar
      files: vllm/model_executor/layers/ple_offload_layer.py, vllm/models/qwen4_exp/amd/ple_layer.py, vllm/models/qwen4_exp/common/ngram_embedding.py, vllm/v1/worker/gpu_worker.py
- [x] `1578e258c9` fix(rdna): gate the QSA rope/rmsnorm fast paths on RDNA and device tensors
      files: vllm/model_executor/layers/layernorm.py, vllm/models/qwen4_exp/amd/indexer_qsa.py
- [x] `95aed84378` feat(rdna2): Qwen3.8 Flash-Next (Qwen4Exp) and DeepSeek-V4 on gfx1030
      files: vllm/model_executor/models/qwen3_5.py, vllm/models/deepseek_v4/amd/rocm.py, vllm/models/qwen4_exp/amd/hc.py, vllm/models/qwen4_exp/amd/ops/hc_rdna2.py …

## docs/rdna2, tools/rdna  (2 commits)

- [x] `e272acabe9` perf(rdna2): turn on the W4A8 prefill GEMMs for the 27B AWQ recipe
      files: docs/rdna2/tuning.md, tools/rdna/recipes/27b-awq.env
- [x] `6cd5b6bd93` fix(tools): drop unread launcher exports and chassis GPU pins
      files: docs/rdna2/tuning.md, tools/rdna/port_v031/unit.sh, tools/rdna/rdna_launcher_common.sh, tools/rdna/recipes/27b-exl3.env …

## csrc/rocm, vllm/_custom_ops.py  (2 commits)

- [x] `2e9fe02c79` feat(rdna2): add a packed INT4 + FP16 block-scale fused QSA decode
      files: csrc/rocm/int4_kv_layout_common.cuh, csrc/rocm/ops.h, csrc/rocm/qsa_rdna2.cu, csrc/rocm/torch_bindings.cpp …
- [x] `dd25f4908a` fix(rdna): QSA HIP — dedicated mqa_paged kernel, compress regression
      files: csrc/rocm/ops.h, csrc/rocm/qsa_rdna2.cu, csrc/rocm/torch_bindings.cpp, vllm/_custom_ops.py …

## bench_results/2026-10-24-mtp-mono-efficiency, tests/models  (2 commits)

- [x] `3e83ca79d7` fix(rdna2): configure MonoKernel NaN fail-stop before worker spawn
      files: bench_results/2026-10-24-mtp-mono-efficiency/startup-config-compatibility/README.md, bench_results/2026-10-24-mtp-mono-efficiency/startup-config-compatibility/baseline-cpu-tests.json, bench_results/2026-10-24-mtp-mono-efficiency/startup-config-compatibility/baseline-cpu-tests.log, bench_results/2026-10-24-mtp-mono-efficiency/startup-config-compatibility/baseline-run_cpu_tests.py.txt …
- [x] `9576cf5be2` fix(rdna2): require fail-stop for target MonoKernel timeouts
      files: bench_results/2026-10-24-mtp-mono-efficiency/startup-failstop-validation/README.md, bench_results/2026-10-24-mtp-mono-efficiency/startup-failstop-validation/cpu-tests.json, bench_results/2026-10-24-mtp-mono-efficiency/startup-failstop-validation/cpu-tests.log, tests/models/qwen4_exp/test_amd_mono_rdna2.py …

## csrc/rocm, tools/rdna2  (2 commits)

- [x] `a5d70272bd` Fix QSA kernel stride width
      files: csrc/rocm/qsa_rdna2.cu, tools/rdna2/probe_qsa_indexer.cu
- [x] `94fc6528a5` perf(rdna): vectorize gdn_prefill_prep — it was 1048 scalar 2-byte loads
      files: csrc/rocm/gdn_prefill_prep_rdna2.cu, tools/rdna2/bench_gdn_prefill_chain_rdna2.py

## docs/rdna2, tests/v1  (2 commits)

- [x] `e3a89d8dfa` debug(rdna): profile DFlash2 stages and identify head GEMM bottleneck
      files: docs/rdna2/tuning.md, tests/v1/spec_decode/test_lilicorr.py, vllm/model_executor/layers/logits_processor.py, vllm/model_executor/models/qwen3_dflash.py …
- [x] `e09a8c9509` fix(rdna): decode-stall cap fit mode per model: all steps on dense, mixed steps on MoE
      files: docs/rdna2/mixed-batch-stall-probe.md, tests/v1/core/test_decode_stall_bound.py, vllm/v1/core/sched/mixed_step.py, vllm/v1/core/sched/scheduler.py

## tools/rdna, vllm/envs.py  (2 commits)

- [x] `41eaf599c7` perf(rdna2): #99's HIP kernel set on by default
      files: tools/rdna/recipes/flashnext-mtp0.env, tools/rdna/recipes/flashnext-mtp2.env, tools/rdna/recipes/flashnext-pp3-mtp2.env, tools/rdna/recipes/flashnext-pp3.env …
- [x] `726b479de9` fix(rdna2): #99's HIP kernel set opt-in until the serve matrix
      files: tools/rdna/recipes/flashnext-mtp0.env, tools/rdna/recipes/flashnext-mtp2.env, tools/rdna/recipes/flashnext-pp3-mtp2.env, tools/rdna/recipes/flashnext-pp3.env …

## tests/kernels, tests/v1  (2 commits)

- [x] `5e0964109c` fix(rdna2): EXL3 decode under torch.compile no longer faults
      files: tests/kernels/quantization/test_exl3_moe_fused.py, tests/v1/ple_offload/test_exl3_ngram_ring.py, vllm/model_executor/layers/fused_moe/routed_experts.py, vllm/model_executor/layers/quantization/exl3.py …
- [x] `e8d1701784` feat(rdna2): FA-RDNA2 attention, GDN and Mamba paths
      files: tests/kernels/attention/test_fa_rdna2_shape_sweep.py, tests/kernels/attention/test_fa_rdna2_writer_layout.py, tests/kernels/attention/test_rdna_v1_consume.py, tests/v1/attention/test_gdn_metadata_builder.py …

## tests/kernels, tools/rdna  (2 commits)

- [x] `a7247e9fd9` feat(qwen4_exp): load and serve EXL3 Qwen3.8-Flash-Next (eager, coherent)
      files: tests/kernels/quantization/test_exl3_moe_fused.py, tools/rdna/recipes/flashnext-exl3.env, vllm/model_executor/layers/fused_moe/routed_experts.py, vllm/model_executor/layers/quantization/exl3.py …
- [x] `eec301a25a` perf(rdna): select RDNA2W8A16Fp8BlockLinearKernel for fp8 block linears
      files: tests/kernels/quantization/test_rdna2_w8a16_fp8_block.py, tools/rdna/port_v031/unit.sh, vllm/model_executor/kernels/linear/__init__.py, vllm/model_executor/kernels/linear/scaled_mm/rdna2_w8a16_fp8_block.py

## tests/v1, vllm/model_executor  (2 commits)

- [x] `a3ff25529b` feat(qwen4_exp): serve the exllamav3 EXL3 n-gram table from the PLE sidecar
      files: tests/v1/ple_offload/test_exl3_ngram_ring.py, vllm/model_executor/layers/quantization/exl3_ple.py, vllm/models/qwen4_exp/common/ngram_embedding.py, vllm/v1/ple_offload/worker.py
- [x] `460da29668` feat(rdna2): port Qwen4Exp / Qwen3.8 Flash-Next for gfx1030
      files: tests/v1/worker/test_ple_offload_worker.py, vllm/model_executor/layers/ple_offload_layer.py, vllm/v1/ple_offload/__init__.py, vllm/v1/ple_offload/connector.py …

## tests/models, tools/rdna  (2 commits)

- [x] `33892a9af4` monokernel: accept the Qwen4Exp MTP drafter's config under --monokernel
      files: tests/models/qwen4_exp/test_amd_mono_rdna2.py, tools/rdna/mono/ab_table.py, vllm/models/qwen4_exp/amd/mono_decode_rdna2.py
- [x] `a553d55fac` perf(rdna): fused HC prefill mix on by default on gfx1030
      files: tests/models/qwen4_exp/test_hc_prefill_rdna2.py, tools/rdna/serve_rdna.sh, vllm/model_executor/layers/rdna_ops.py, vllm/models/qwen4_exp/amd/ops/hc_rdna2.py

## tests/models  (2 commits)

- [x] `67c3bd6ef5` monokernel tests: K2 result order; no model resolution in the switch test
      files: tests/models/qwen4_exp/test_amd_mono_rdna2.py, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py
- [x] `83ef316369` Fuse AMD PLE grouped normalization on V620
      files: tests/models/qwen4_exp/test_ple_amd.py

## vllm/config, vllm/model_executor  (2 commits)

- [x] `5749e9bf79` monokernel E1': rdna_mono_gdn_graph records the fused GDN kernel inside FULL graphs
      files: vllm/config/compilation.py, vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `ea4b7840d4` feat(rocm): port Aron Hsiao's platform-config changes (2 commits)
      files: vllm/config/kernel.py, vllm/config/parallel.py, vllm/config/speculative.py, vllm/model_executor/layers/fused_qk_norm_rope.py

## csrc/rocm, tests/v1  (2 commits)

- [x] `15c446413f` perf(rdna2): HIP top-k/top-p logit masking
      files: csrc/rocm/v1_topk_topp_rdna2.cu, tests/v1/sample/test_topk_topp_ops_rdna2.py, vllm/v1/sample/ops/topk_topp_sampler.py
- [x] `ccc3c6db02` perf(rdna2): serve the v1 worker/sampler Triton kernels from HIP
      files: csrc/rocm/v1_input_batch_rdna2.cu, csrc/rocm/v1_mamba_rdna2.cu, csrc/rocm/v1_rope_struct_rdna2.cu, csrc/rocm/v1_sample_rdna2.cu …

## tools/monokernel-2a, vllm/model_executor  (2 commits)

- [x] `0cd6965b70` monokernel E1': rdna_mono_gdn_max_rows width gate; run_specs idle check follows SMI_DEVS (HIP 2-5 runs)
      files: tools/monokernel-2a/run_specs.sh, vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- [x] `fa141f0a21` monokernel E1': wire gdn_fused_rdna2 into the Qwen GDN layer behind rdna_mono_gdn
      files: tools/monokernel-2a/parity_hook.sh, tools/monokernel-2a/serve_2a.sh, vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py

## csrc/rocm, docs/rdna  (2 commits)

- [x] `d09df65eb9` rdna(monokernel): 2a M3 serve A/B, in-serve FFN timing, results
      files: csrc/rocm/mono_ffn_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp, docs/rdna/monokernel/2a-m1-results.md …
- [x] `65b38e6431` rdna(monokernel): 2a M1 persistent FFN kernel for gfx1030 + standalone bench
      files: csrc/rocm/mono_ffn_rdna2.cu, docs/rdna/monokernel/2a-m1-results.md, tools/monokernel-2a/README.md, tools/monokernel-2a/ext_binding.cpp …

## tunableop/profiles.json, tunableop/rocm10-rocblas5.6  (2 commits)

- [x] `c42f1fb02f` tunableop(rdna2): add the decode/MTP grid to the rocm10 profile (44 -> 128 rows)
      files: tunableop/profiles.json, tunableop/rocm10-rocblas5.6/README.md, tunableop/rocm10-rocblas5.6/provenance.json, tunableop/rocm10-rocblas5.6/tunableop_results0.csv …
- [x] `e1eabeaa72` tunableop(rdna2): add the decode/MTP grid to the rocm10 profile (44 -> 128 rows)
      files: tunableop/profiles.json, tunableop/rocm10-rocblas5.6/README.md, tunableop/rocm10-rocblas5.6/provenance.json, tunableop/rocm10-rocblas5.6/tunableop_results0.csv …

## tests/quantization, vllm/model_executor  (2 commits)

- [x] `1a4be32741` fix(rdna2): native MoE path declines asymmetric checkpoints
      files: tests/quantization/test_rdna2_moe_asymmetric.py, vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/rocm_moe_rdna.py
- [x] `bc882a3a45` fix(rdna2): resident MoE layout skips the N-first transpose
      files: tests/quantization/test_moe_wna16.py, vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py, vllm/model_executor/layers/quantization/rdna2_moe_resident.py

## vllm/compilation  (2 commits)

- [x] `dc3065852b` fix(rdna): keep strong refs in breakable-graph eager breaks on RDNA
      files: vllm/compilation/breakable_cudagraph.py
- [x] `dc9e536f13` perf(rdna): share piecewise-graph input buffers across capture sizes
      files: vllm/compilation/cuda_graph.py

## tunableop/profiles.json, tunableop/rocm7.14-rocblas5.5  (2 commits)

- [x] `a25f2e0b28` tunableop(rdna2): coverage rows for Qwen3.6, DSV4, EXL3 MTP=2 and AWQ W4A8
      files: tunableop/profiles.json, tunableop/rocm7.14-rocblas5.5/README.md, tunableop/rocm7.14-rocblas5.5/provenance.json, tunableop/rocm7.14-rocblas5.5/tunableop_results0.csv …
- [x] `8aa453030d` perf(rdna2): pad Flash-Next HC prefill GEMM rows to tuned sizes + mixed-batch TunableOp rows
      files: tunableop/profiles.json, tunableop/rocm7.14-rocblas5.5/README.md, tunableop/rocm7.14-rocblas5.5/provenance.json, tunableop/rocm7.14-rocblas5.5/tunableop_results0.csv …

## benchmarks/kernels, tools/rdna  (1 commits)

- [x] `9522c57097` fix(rdna2): scale the prefill token budget with the mamba state block
      files: benchmarks/kernels/pr134/qualify_serving.py, tools/rdna/serve_rdna.sh, vllm/v1/core/sched/scheduler.py

## vllm/platforms, vllm/v1  (1 commits)

- [x] `946b75e026` fix(rdna2): correct the V620 compute-unit count in the comments
      files: vllm/platforms/rocm.py, vllm/v1/attention/backends/rdna_attn.py, vllm/v1/attention/backends/triton_attn.py

## docs/explore, pyproject.toml  (1 commits)

- [x] `72dded5038` feat(rdna): refuse a quantized KV cache for the MonoKernel
      files: docs/explore/kv-int4-fp16-scales/README.md, pyproject.toml, tests/models/qwen4_exp/test_amd_mono_rdna2.py, tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py …

## .buildkite/test-amd.yaml, bench_results/2026-10-24-mtp-mono-efficiency  (1 commits)

- [x] `f3abb8b57a` fix(rdna2): fail-stop Mono MTP faults across ranks and clean up workers
      files: .buildkite/test-amd.yaml, bench_results/2026-10-24-mtp-mono-efficiency/failstop-engine-cleanup-cpu-review-v1/check_cpu.py.txt, bench_results/2026-10-24-mtp-mono-efficiency/failstop-engine-cleanup-cpu-review-v1/core.py.baseline.txt, bench_results/2026-10-24-mtp-mono-efficiency/failstop-engine-cleanup-cpu-review-v1/core.py.candidate.txt …

## bench_results/2026-10-24-mtp-mono-efficiency, tests/v1  (1 commits)

- [x] `4f694dce76` fix(spec-decode): count NaNs across every verification row
      files: bench_results/2026-10-24-mtp-mono-efficiency/spec-logits-nan-validation/reviewed/README.md, bench_results/2026-10-24-mtp-mono-efficiency/spec-logits-nan-validation/reviewed/cpu-ast-result.json, bench_results/2026-10-24-mtp-mono-efficiency/spec-logits-nan-validation/reviewed/create_test_overlay.py.txt, bench_results/2026-10-24-mtp-mono-efficiency/spec-logits-nan-validation/reviewed/frozen-pytest-v1/boot.after …

## bench_results/2026-10-24-mtp-mono-efficiency, csrc/rocm  (1 commits)

- [x] `e5e3e972fb` fix(rdna2): match native expert routing and dense rounding
      files: bench_results/2026-10-24-mtp-mono-efficiency/correctness-core-reviewed/README.md, bench_results/2026-10-24-mtp-mono-efficiency/correctness-core-reviewed/build/boot.after, bench_results/2026-10-24-mtp-mono-efficiency/correctness-core-reviewed/build/boot.before, bench_results/2026-10-24-mtp-mono-efficiency/correctness-core-reviewed/build/build.log …

## bench_results/2026-10-22-mtp-mono-efficiency, tests/kernels  (1 commits)

- [x] `59f85fbda8` perf(rdna2): tune WNA16 tile8 prefill threshold
      files: bench_results/2026-10-22-mtp-mono-efficiency/micro-wna16/gfx1030-vmul-lo-audit-2026-10-23.md, bench_results/2026-10-22-mtp-mono-efficiency/micro-wna16/tile-threshold-hip25-2026-10-23/01-original/boot-id.after, bench_results/2026-10-22-mtp-mono-efficiency/micro-wna16/tile-threshold-hip25-2026-10-23/01-original/boot-id.before, bench_results/2026-10-22-mtp-mono-efficiency/micro-wna16/tile-threshold-hip25-2026-10-23/01-original/config.env …

## bench_results/2026-10-22_dense-mtp-k2-gmem092, csrc/rocm  (1 commits)

- [x] `ae44444b37` feat(rdna): add dense MTP FFN to MonoKernel K2
      files: bench_results/2026-10-22_dense-mtp-k2-gmem092/README.md, csrc/rocm/mono_k2_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp …

## .buildkite/test_areas, csrc/rocm  (1 commits)

- [x] `8c000a6cc2` feat(rdna): opt MTP drafter into safe MonoKernel routing
      files: .buildkite/test_areas/models_basic.yaml, csrc/rocm/mono_k2_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp …

## bench_results/2026-10-20_dflash2-spec2-tunableop, docs/rdna2  (1 commits)

- [x] `c667dc80b8` perf(rdna): add tuned BF16 DFlash2 spec2 GEMM rows
      files: bench_results/2026-10-20_dflash2-spec2-tunableop/README.md, docs/rdna2/tuning.md, tunableop/README.md, tunableop/profiles.json …

## docs/rdna2, tools/rdna2_028  (1 commits)

- [x] `fe7709e4dc` perf(rdna): tune BF16 DFlash2 draft GEMMs on V620
      files: docs/rdna2/tuning.md, tools/rdna2_028/tune_prod_shapes.py, tools/rdna2_028/verify_tunableop_lookup.py, tunableop/README.md …

## README.md, docs/rdna2  (1 commits)

- [x] `2d2c374eec` feat(rdna): serve Flash-Next with BF16 DFlash2 draft and BLHNC FA
      files: README.md, docs/rdna2/tuning.md, tests/kernels/attention/test_fa_rdna2_writer_layout.py, tools/rdna/recipes/flashnext-dflash2.env …

## tests/models, vllm/v1  (1 commits)

- [x] `7c5fa00a43` fix(spec): #119's zero-draft paths only under dynamic speculation
      files: tests/models/qwen4_exp/test_ple.py, vllm/v1/attention/backends/short_conv_attn.py, vllm/v1/worker/gpu/model_states/mamba_hybrid.py

## docs/rdna2, tests/conftest.py  (1 commits)

- [x] `73e81436ba` perf(rdna2): default VLLM_RDNA_ALL_HIP on
      files: docs/rdna2/qwen4_exp_hip_tracker.md, docs/rdna2/tuning.md, tests/conftest.py, vllm/envs.py …

## docs/rdna2, vllm/envs.py  (1 commits)

- [x] `55dffa2961` cleanup(rdna): drop leftover runtime debug probes
      files: docs/rdna2/tuning.md, vllm/envs.py, vllm/utils/rocm_poolwatch.py, vllm/v1/worker/gpu/block_table.py …

## vllm/config, vllm/engine  (1 commits)

- [x] `c6a718553c` [Kernel] Backport --monokernel / KernelConfig.monokernel (vllm-project/vllm#61082)
      files: vllm/config/kernel.py, vllm/engine/arg_utils.py

## tunableop/README.md, tunableop/rocm10-rocblas5.6  (1 commits)

- [x] `bab1060569` tunableop(rdna2): re-measure the whole grid, cover the dyn-prefill 256/512 chunks
      files: tunableop/README.md, tunableop/rocm10-rocblas5.6/provenance.json, tunableop/rocm10-rocblas5.6/tunableop_results0.csv, tunableop/rocm10-rocblas5.6/tunableop_results1.csv …

## csrc/rocm, tests/quantization  (1 commits)

- [x] `6f5edc41a1` feat(rdna2): native W4A16 MoE for asymmetric compressed-tensors checkpoints
      files: csrc/rocm/moe_q_gemm_rdna2.cu, csrc/rocm/moe_w4a8_rdna2.cu, csrc/rocm/ops.h, csrc/rocm/torch_bindings.cpp …

## tools/rdna, vllm/models  (1 commits)

- [x] `15a3bc86c4` feat(rdna): PP3 MTP2 for Flash-Next on three cards
      files: tools/rdna/recipes/flashnext-pp3-mtp2.env, tools/rdna/recipes/flashnext-pp3.env, tools/rdna/serve_rdna.sh, vllm/models/qwen4_exp/amd/mtp.py …

## tests/tools, tools/rdna  (1 commits)

- [x] `516d52f847` fix(tools): MTP expert quantizer uses the checkpoint's absmax/7.5 scale
      files: tests/tools/test_quant_mtp_experts.py, tools/rdna/quant_mtp_experts.py

## csrc/rocm, tools/rdna  (1 commits)

- [x] `f2cc25ca92` fix(rdna2): W4A8 MoE per-call workspace, 512-token threshold, microbench
      files: csrc/rocm/moe_w4a8_rdna2.cu, tools/rdna/bench_moe_w4a8.py, vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py

## (none)  (1 commits)

- [x] `f87b642473` perf(rdna2): W4A8 MoE prefill on Flash-Next (HIP moe_w4a8_gemm_rdna2)

## benchmarks/kernels, docs/rdna2  (1 commits)

- [x] `2603ff082e` bench(rdna): TP all-reduce microbenchmark at serving sizes + RCCL sweep
      files: benchmarks/kernels/benchmark_rdna_allreduce.py, docs/rdna2/tp4-allreduce-sizes-2026-10-10.md, tools/rdna/port_v031/ar_sweep.sh

## CMakeLists.txt, benchmarks/kernels  (1 commits)

- [x] `9cc6d6c61a` perf(rdna): fused HC prefill mix kernels for Flash-Next (opt-in)
      files: CMakeLists.txt, benchmarks/kernels/benchmark_rdna_hc_prefill.py, csrc/rocm/hc_prefill_rdna2.cu, csrc/rocm/ops.h …

## tools/rdna, vllm/platforms  (1 commits)

- [x] `5d9f76f144` feat(rdna): decode-stall cap off by default, enabled in the dense 27B recipes
      files: tools/rdna/recipes/27b-awq.env, tools/rdna/recipes/27b-exl3.env, tools/rdna/recipes/full.env, tools/rdna/serve_rdna.sh …

## tests/v1, vllm/config  (1 commits)

- [x] `38d2a446ec` feat(sched): bound prefill work in steps that carry decodes (RDNA default)
      files: tests/v1/core/test_decode_stall_bound.py, vllm/config/scheduler.py, vllm/engine/arg_utils.py, vllm/platforms/rocm.py …

## vllm/utils, vllm/v1  (1 commits)

- [x] `a5a456939a` debug(rdna): host time, per-custom-op spans and sync sites in the phase timer
      files: vllm/utils/torch_utils.py, vllm/v1/worker/gpu/model_runner.py, vllm/v1/worker/gpu/phase_timer.py

## csrc/libtorch_stable  (1 commits)

- [x] `cc8beb8cc0` perf(rocm): top_k_per_row_decode uses the radix path on gfx10x
      files: csrc/libtorch_stable/sampler.cu

## tests/kernels, vllm/v1  (1 commits)

- [x] `dcd2388100` fix(rdna): memory-bounded indexer prefill MQA logits on gfx1030
      files: tests/kernels/attention/rdna/dsv4/test_attention_ops.py, vllm/v1/attention/ops/rocm_aiter_mla_sparse.py

## tests/distributed, tests/kernels  (1 commits)

- [x] `ab5a35deed` fix(types): mypy clean for fork-owned files
      files: tests/distributed/test_rdna_ar.py, tests/kernels/attention/test_fa_rdna2_writer_layout.py, vllm/compilation/backends.py, vllm/compilation/breakable_cudagraph.py …

## vllm/config, vllm/models  (1 commits)

- [x] `cd8056fd51` fix(qwen4_exp): RDNA: run the PLE n-gram embedding as a graph-splitting op
      files: vllm/config/compilation.py, vllm/models/qwen4_exp/amd/ple_layer.py

## tests/compile, vllm/compilation  (1 commits)

- [x] `8cb1c50c07` fix(rdna): keep upstream behavior off RDNA; gate fork paths to ROCm RDNA
      files: tests/compile/test_cudagraph_replay_inputs.py, vllm/compilation/backends.py, vllm/compilation/breakable_cudagraph.py, vllm/compilation/cuda_graph.py …

## docs/rdna2  (1 commits)

- [x] `fc64e86da8` feat(rdna2): V620 PIECEWISE launcher and measured report
      files: docs/rdna2/V620-PIECEWISE-CUDAGRAPHS.md

## tests/v1  (1 commits)

- [x] `cd1c5498b1` feat(rocm): PLE CPU sidecar offload (kept, not wired on v0.31)
      files: tests/v1/ple_offload/test_ple_quant_gather.py

## tests/compile, tests/v1  (1 commits)

- [x] `4b8444c248` fix(rocm): engine, cudagraph, scheduler and KV-cache fixes for gfx1030
      files: tests/compile/test_config.py, tests/compile/test_cudagraph_replay_inputs.py, tests/v1/core/test_mixed_batch_policy.py, tests/v1/engine/test_prefill_cadence.py …

## csrc/rocm, tests/compile  (1 commits)

- [x] `01e259c6c5` feat(rocm): rdna_ar one-shot all-reduce for gfx10x PCIe
      files: csrc/rocm/rdna_allreduce.cu, csrc/rocm/rdna_allreduce.cuh, tests/compile/test_tp_all_reduce_compile.py, tests/distributed/test_rdna_ar.py …

## tests/kernels  (1 commits)

- [x] `ba94c4f1fd` [Bugfix] Index mamba spec-decode block tables by request slot, not batch row
      files: tests/kernels/mamba/test_precopy_mamba_align.py

## bench_results/2026-10-01_exl3-27b-mtp2, bench_results/2026-10-01_exl3-27b-mul1-tp4  (1 commits)

- [x] `93df2d1f94` feat(rdna): EXL3 quantization on RDNA
      files: bench_results/2026-10-01_exl3-27b-mtp2/SUMMARY.md, bench_results/2026-10-01_exl3-27b-mtp2/eager/acceptance.txt, bench_results/2026-10-01_exl3-27b-mtp2/eager/cells/c1024_1.acceptance.txt, bench_results/2026-10-01_exl3-27b-mtp2/eager/cells/c1024_1/openai-infqps-concurrency1-Qwen3.8-27B-exl3-3.00bpw-20261001-140130.json …

## bench_results/2026-09-29_w4a8-moe, csrc/rocm  (1 commits)

- [x] `52e79c3256` feat(rdna2): native RDNA2 MoE (W4A16/W4A8/W8A16/MXFP4) and dispatch
      files: bench_results/2026-09-29_w4a8-moe/moe_marker.txt, csrc/rocm/moe_accum_rdna2.cuh, csrc/rocm/moe_q_gemm_rdna2.cu, csrc/rocm/moe_w4a8_rdna2.cu …

## vllm/config, vllm/envs.py  (1 commits)

- [x] `9f3dc48636` feat(rocm): RDNA2 platform detection, config and env knobs
      files: vllm/config/compilation.py, vllm/config/scheduler.py, vllm/config/vllm.py, vllm/envs.py …

## pyproject.toml  (1 commits)

- [x] `0994bce323` Port opt-in served resident skinny MoE decode
      files: pyproject.toml

---

## Verification (2026-11-01)

Every box above was checked by content, not by subject. Method: for each commit,
take its added lines (`git show --unified=0`), normalise whitespace, and test
membership in the pre-merge fork tree (`8015ea485b`) and in the merged tree
(`rebase/v0.32.0`). A commit counts as present when its added lines survive at
the same rate in both trees; a drop between the two trees is a merge loss.

### Whole-tree audit of the fork delta

`git diff v0.31.0 8015ea485b`, added code lines only (docs/tests/tools/bench
excluded): __50,431 added lines across 247 files, 236 missing (0.5 %)__ in the
merged tree. All 236 are accounted for below; none is a lost feature.

Per-commit audit of the 258 checklist items: 247 commits show identical
presence before and after the merge; 8 show a drop, each reviewed line by line:

| commit | fork → merged | verdict |
|---|---|---|
| `5bf02f78ea` | 100/100 → 95/100 | `mamba_utils` `bind_block_tables` / `bind_batch_block_tables` reworked around `_write_block_table_ptrs(key, stride)`; source-vs-batch table semantics preserved |
| `ebaa70a455` | 311/316 → 298/316 | `autoregressive/` → `mtp/` rehome upstream; all 72 fork lines of that delta are present in `v1/worker/gpu/spec_decode/` |
| `92b2cb1ca1` | 486/537 → 392/537 | DeepSeek-V4 MTP: upstream rehomed `get_mtp_target_hidden_states` into `spec_decode/speculator.py` + `spec_decode/utils.py` |
| `b637fb5429` | 143/143 → 133/143 | PLE short conv: upstream replaced the torch/HIP conv with Triton `ple_conv`; `VLLM_RDNA_PLE_CONV_HIP` left unwired (default 0, no behaviour change) |
| `72e760942b` | 104/104 → 100/104 | same PLE rework; offload contract (`forward_impl(hidden_states, input_ids, query_start_loc, ngram_context)`) intact |
| `2d2c374eec` | 207/221 → 202/221 | docs/recipe comments only |
| `cc8beb8cc0` | 7/7 → 0/7 | `csrc/libtorch_stable/sampler.cu` flattened to `csrc/sampler.cu`; the gfx10 `preferRadix` hunk is present at `csrc/sampler.cu:877-884` |
| `8cb1c50c07` | 302/400 → 291/400 | RDNA mamba block-size override superseded by upstream `mamba_block_size`; `custom_all_reduce.py` keeps `on_rdna_family()` |

Commits whose files are absent from *both* trees (`csrc/rocm/mono_ffn_rdna2.cu`,
`tools/monokernel-2a/*`) were already reverted inside the v0.31 fork range —
they are not merge losses.

### Post-merge fixes (landed on `rdna_extras`)

`1ecb476adc` merge, then:

| commit | change |
|---|---|
| `1dc6bd7cdc` | int32 index mappings on RDNA (`idx_mapping`/`expanded_idx_mapping`; upstream #60736 made them int64) and defensive casts at the HIP call sites; `kernel_block_sizes` dropped from `compute_slot_mappings_rdna2` (v0.32 unified kv-block and kernel-block sizes) so the HIP kernel matches upstream's Triton math; restored the fork hunks the merge dropped (`set_graph_pool_id` import, `MambaHybridModelState._ensure_align_ctx`); RDNA FULL→PIECEWISE cudagraph fallback dispatches on the unpadded token count |
| `56429f77cb` | per-step `idx_mapping` built as int32 on RDNA |

`INT4_FP16_BLOCK` renumbered 11 → 12 (upstream's `ULTRAQUANT_4BIT` took 11).

### Build + serve gate (par1-cs25, 4× V620, TP=4, venv-7.14.0_0.32.0)

- Build `build_exit=0`; `_rocm_C` exposes __159 ops (144 rdna, 12 hint_flex, 1 expert_cache)__;
  `vllm 0.32.0+rdna`.
- 27B AWQ-INT4 (`27b-awq`): boot 110 s, greedy 3/3, PPL 4.91, prefix PASS,
  16k/1k c=8 __47.03 agg out tok/s__ (pre-merge 40.37, +16.5 %), 0 faults.
- Flash-Next AWQ-W4A16 (`flashnext-mtp0`, PLE int4): boot 250 s, greedy 3/3,
  PPL 4.80, prefix PASS, 16k/1k c=8 __84.75 agg out tok/s__
  (pre-merge 77.75, +9.0 %), 0 faults. Monokernel: off 85.02/85.14, on 86.16 —
  supported on RDNA2 (`mono_decode_rdna2` k1/k2/gdn), ~+1 % at this cell.
- Flash-Next AWQ-W4A16 (`flashnext-mtp2`, PLE int4): boot 280 s, greedy 3/3,
  PPL 4.83, prefix PASS, 16k/1k c=8 __91.13 agg out tok/s__ (pre-merge 80.52,
  +13.2 %), 1k/512 c=1 __72.86__, c=8 __206.26__ agg out, 0 faults.
- HI2 27B (`rdna_hint_flex`, 3.0 bpw; `RECIPE=full TP=4`): boot 111 s,
  greedy 3/3, PPL 5.73, prefix PASS, 16k/1k c=8 __40.85 agg out tok/s__,
  0 faults. The HI2 serving path is now part of this tree — merge `7c28f696f8`
  brings `feat/hi2-weights-support-rebased` (29 commits) in cleanly: the
  quantization method (`rdna_hint_flex{,_moe,_names}` + utils), the
  `csrc/rocm/rdna_hint_flex_rdna2.cu` kernels (13 ops), the MoE wiring and
  the tools/docs. `_rocm_C` now exposes 156 ops (141 rdna, 12 hint_flex).

## v0.33.0 move (2026-11-01)

Upstream tag `v0.33.0` = `4ba787ec99` (released 2026-10-27, identical to `rc3`).
Merge commit `b1b54608b3` on `rdna_extras`; parents `d2a3c99f7d` (the v0.32.0
line) and `4ba787ec99`. Gap: 943 upstream commits, 2908 files; 115 overlap the
fork's 1635 touched files. 67 conflicted files / 119 hunks, resolved by content.
Pre-move state: tag `backup-rdna_extras-v0.32.0-20261101` = `d2a3c99f7d`.

### Where v0.33 moved the fork's code

| fork site (v0.32) | v0.33 location | adaptation |
|---|---|---|
| `get_kv_cache_config_from_groups` in `vllm/v1/core/kv_cache_utils.py` | `vllm/v1/core/kv_cache_planning.py` | PP ownership filter ported: a uniform group with no layer on this worker keeps the global spec but lays out only `group.layer_names`. Same filter added to `vllm/v1/hisparse/layout.py::_build_hisparse_kv_cache_tensors` |
| `csrc/rocm/attention.cu` | `attention_dispatch.cu` + 12 `attention_paged_attn_*.cu` + a configure-time `ATTN_HEAD_SWITCH` guard | fork sources (`rdna_allreduce.cu`, `rdna_expert_cache.cu`) kept in `VLLM_ROCM_EXT_SRC`; `--offload-compress` adopted (module 211 MB → 89 MB) |
| `Qwen4ExpPLEEmbeddingMethod.lookup(...)` | `lookup_dtype(layer)` + `lookup_from_pinned(layer, ids, output)` | `ngram_embedding.py` rebuilt on upstream's file; the int4 method gained `lookup_dtype`, `lookup_from_pinned` (`_lookup_ple_int4_from_pinned_kernel`) and `lookup_dequantizes = True`; the layer exposes a `lookup_dtype` property |
| `AttentionBackendEnum` in `vllm/v1/attention/backend.py` | `vllm/v1/attention/backends/registry.py` | `RDNA_ATTN` registration carried over |
| `GatedResidual.lora_rank` | `GatedResidual.hc_lowrank` | `4336e23f48`: the fork's `_injection_slice` + two `combine_and_mix` call sites renamed (boot died on `AttributeError` under dynamo otherwise) |

### Fork deltas v0.33 removed (verified upstream-side, not merge losses)

- `_uses_trailing_mtp_layers`, `_groups_partition_layers_exactly`,
  `_annotate_eagle_groups` and their 11 tests (draft-group annotation / trailing
  MTP-layer fallback). Present in the merge base, gone from `v0.33.0`.
- `is_breakable_cudagraph_enabled()` early-return in `eager_break_during_capture`
  (upstream now always wraps; `rdna = on_rdna_family()` kept for the FULL-mode
  break-out).
- `causal_conv1d_fn` / `causal_conv1d_update` `block_idx_*` and
  `initial_state_idx` parameters — the HIP dispatch now gates on `metadata is
  None` instead of `block_idx_last_scheduled_token is None`.
- `is_quantized_kv_cache` in `vllm/utils/torch_utils.py` (moved to
  `vllm/v1/kv_cache_interface.py` and keyed on `get_kv_quant_mode`, which maps
  `int4_fp16_block` to `KVQuantMode.INT4_FP16_BLOCK`, so the answer is unchanged).
- `supports_fp8()` widened by upstream to `on_cdna() or on_rdna()`.

### Renumbering

`KVQuantMode.INT4_FP16_BLOCK` stays __12__; v0.33's Octave modes took 13/14/15
and `CUSTOM` 16. No numeric comparisons exist — the values are only used through
the enum.

### Spec-decode dispatch — every pre-existing case is HIP again

The merge initially routed two per-step kernels to Triton; both were brought
back to HIP (`d6267ee3b0`, `b8efd1bf03`):

| op | native HIP | Triton |
|---|---|---|
| rejection | every case the v0.32 HIP kernel covered: nonempty batch, padded-vocab blocks ≤ 32, `rejection_sample_method` `standard`/`synthetic`, `fly_window_size == 0` | only the genuinely new v0.33 features: block-verification seed mixing (`rejection_sample_method=block`) and fly verification (`fly_window_size > 0`) |
| logits statistics / resampling / insert | unchanged eligibility | existing fallbacks |
| `post_update` | __all cases, including MTP drafts__ — the kernel scatters `draft_tokens[req_id][i]` into `draft_tokens_out[req_state_idx][i]` for live rows | never (Triton only when `_ON_RDNA2` is false) |
| block-table gather | __all cases, including sliding-window masking__ — entries below `first_pos - reach` are nulled | never on RDNA |

Our recipes use the default `rejection_sample_method` and `fly_window_size=0`,
so the whole spec-decode step chain (stats → rejection → resample → insert →
`post_update` → gather) runs on HIP.

### Correctness regressions the merge introduced, now fixed

- __`compute_slot_mappings` lost `fea652bcd0`'s bound__ (`b8efd1bf03`). Recurrent
  /circular cache rows are shorter than the sequence, so a long prefill read the
  next request's block-table row and emitted a bogus slot id instead of
  `PAD_SLOT_ID`. The bound (`block_index < block_table_stride`) is restored in
  both the Triton kernel and the HIP port; the fork's own regression test
  `test_slot_mapping_does_not_read_next_request_for_short_state_tables` failed
  after the merge and passes again.
- __`gather_block_tables` fell back to Triton for SWA masking__; the HIP kernel
  now implements the masking, verified bit-for-bit against Triton by a new test
  (`test_gather_block_tables_sliding_window_masking_hip_matches_triton`).
- __`GatedResidual.lora_rank` → `hc_lowrank`__ (`4336e23f48`): the v0.33 rename
  broke Flash-Next boot under dynamo.

Worker test suite on the final tree (4× V620, `HIP_VISIBLE_DEVICES=2`):
`tests/v1/worker/test_gpu_block_table.py` + `test_input_batch_ops_rdna2.py`
→ __70 passed, 0 failed__.

### Build + serve gate (par1-cs25, 4× V620, TP=4, venv-7.14.0_0.32.0)

- Build `build_exit=0`, `vllm 0.33.0+rdna.rocm714`, `_rocm_C.abi3.so` 89 MB
  (`--offload-compress`); __159 `_rocm_C` ops / 144 rdna / 12 hint_flex /
  1 expert_cache__ — identical counts to v0.32, so no fork op was dropped.
- v0.33 raised the Python requirements; the venv needed `orjson>=3.11.6`
  (hard import failure at boot), `certifi>=2026.10.19`, `multidict>=6.9.1`,
  `aiohttp>=3.14.4`, `pillow>=13.0.0`, `tokenizers>=0.23.2` and
  `transformers>=5.16.1,<5.21.0`. torch stayed at 2.12.0+rocm7.14.0.

| recipe | v0.33 | v0.32 baseline | Δ |
|---|---:|---:|---:|
| `27b-awq` 16k/1k c=8 | __47.60__ | 47.03 | +1.2 % |
| `flashnext-mtp0` (PLE int4) 16k/1k c=8 | __84.99__ | 84.75 | +0.3 % |
| `flashnext-mtp2` (PLE int4) 16k/1k c=8 | __89.96__ | 91.13 | −1.3 % |
| `flashnext-mtp2` 1k/512 c=1 | __71.30__ | 72.86 | −2.1 % |
| `flashnext-mtp2` 1k/512 c=8 | __215.49__ | 206.26 | +4.5 % |
| `full` TP=4 (HI2 27B) 16k/1k c=8 | __41.41__ | 40.85 | +1.4 % |

These are the __final-binary__ numbers: every recipe was re-run after
`d6267ee3b0` (HIP draft scatter) and `b8efd1bf03` (HIP SWA gather masking +
restored slot-mapping bound) landed. Boot: 27B 120 s, Flash-Next MTP0 271 s,
MTP2 300 s, HI2 90 s. Every cell:
greedy 3/3, PPL 4.82–5.73, prefix-cache PASS (19–31×), concurrency PASS,
0 fault markers, 0 new dmesg faults. `RDNA_ATTN`, `RDNA2W4A16LinearKernel`,
`CompressedTensorsWNA16RDNA2MoEMethod`, `PLE int4` and `rdna_ar` one-shot
all report active in the serve log.

Artifacts: `~/w4a8_runs/v033/{27b-awq,fn-mtp0,fn-mtp2,full}.log` on the box.

### Follow-up: expert offload under breakable graphs (2026-11-02)

The gate above ran resident recipes on compiled piecewise graphs. The MoE
expert-offload path forces `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for piecewise
capture, which exposed an unwrapped-PLE-op bug: request 2 returned NaN logits.
Fixed on `rdna_extras` as `4af2aa0c2e` (PLE ops are
`@eager_break_during_capture`); TP2 and TP4 offload are re-qualified with
`FULL_AND_PIECEWISE` and prefix caching. Record, root-cause table and numbers:
[`moe-offload-topology-2026-11-02.md`](./moe-offload-topology-2026-11-02.md).

### Follow-up: second serve pass over the paths the gate did not run (2026-11-02)

Every arm is TP=4 on `FULL_AND_PIECEWISE` (27B EXL3: `PIECEWISE`) with prefix
caching, 3 prompts x 3 rounds at request 2+ (greedy, garbage and NaN detection),
then a ~5.2k-token shared-prefix probe (cold, warm, shared prefix, warm again).

| Path | Result | Prefix cache (hit/queried tokens, cold to warm) |
|---|---|---|
| Flash-Next + breakable graphs, no mono/offload (post PLE fix) | 9/9, 0 NaN | n/a |
| Flash-Next MTP0 + `VLLM_RDNA_QSA_HIP=1` | 9/9, 0 NaN | n/a |
| Flash-Next EXL3 3.05bpw, text-only | 9/9, 0 NaN | 15360/20824, 9.7 s to 2.7 s |
| 27B EXL3 3.00bpw, MTP=2 | 9/9, 0 NaN | 14400/20824, 5.0 s to 1.0 s |
| 27B AWQ, MTP=2, W4A8 sdot4 active | 9/9, 0 NaN | 15360/20824, 4.3 s to 0.46 s |

Three fixes came out of it (`cd154d3fc7`):

- `RdnaAttentionMetadataBuilder` set `supports_draft_decode_metadata_update = True`
  without implementing `update_draft_decode_metadata`, so any MTP >= 2 draft loop on
  RDNA attention raised `NotImplementedError` in warmup. Latent since FA-RDNA2 (same
  in v0.32); the update is a documented no-op because the persistent buffers advance
  in place and the decode path never reads `max_seq_len`.
- `PagedAttention.write_to_paged_cache` gained a required `interleaved_v` argument in
  v0.33; the RDNA call site (block size 16/32, native layout) was not updated. Now
  passes `False`. Unreachable from the 1024-block hybrid recipes, so verified by
  mypy only.
- `flashnext-exl3` now defaults to `language-model-only`: the checkpoint EXL3-quantizes
  the vision tower as separate q/k/v trellises that the fused-qkv ViT cannot load.

A mypy scan of the fork's 186 changed Python files against upstream `v0.33.0` found
no other call-signature breaks (remaining findings are vendored Triton typing noise).

Not servable on the box: DSV4 / MXFP4 (no checkpoint present), so those stay
statically verified only.
