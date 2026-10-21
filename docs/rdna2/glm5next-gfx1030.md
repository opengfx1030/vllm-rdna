# GLM-5.3-Flash (Glm5Next) on gfx1030: path audit and run plan

Branch `cursor/glm5next-rdna-upstream-7624` (PR #51) runs GLM-5.3-Flash
through upstream `Glm5NextForConditionalGeneration` with a gfx1030 layer on
top. This page records what each layer type actually executes on a V620 with
the code as of this branch, the three boot blockers found by reading that
path (all fixed here, none yet run on hardware), the memory budget, and the
checklist for the first run. `tools/rdna/recipes/glm53-flash.env` is the
launcher side of the same plan.

Nothing here has run on a GPU. The agent-side evidence is the CPU tests named
below and a clang device-only compile of the nine HIP TUs (see "ISA check").

## The model, as the config class and PR #2 recorded it

PR #2 read the checkpoint's `config.json`; `vllm/transformers_utils/configs/glm5_next.py`
carries the same numbers as defaults. Confirm against the checkpoint before
sizing anything (HF was not reachable from the agent box).

| Block | Shape |
| --- | --- |
| Layers | 45: 34 KDA (`linear_attention`) + 11 DSA (`deepseek_sparse_attention`, layers 3, 7, ..., 43); MTP head is layer 45 |
| Hidden / vocab | 4096 / 154880 |
| MLP | layers 0-2 dense (12288); layers 3-44 MoE: 288 routed experts, `moe_intermediate_size` 2048, top-8, sigmoid scoring, `routed_scaling_factor` 2.5, 1 shared expert, `swiglu_limit` 10 |
| KDA | 64 heads x 128, conv 4, gate lower bound -5, state `[H, 128, 128]` fp32 |
| DSA | `q_lora_rank` 1536, `kv_lora_rank` 512, 64 heads x 256 NoPE (`qk_rope_head_dim` 0), `v_head_dim` 256 |
| Indexer | 32 heads x 128, `index_topk` 2048, `index_kpool` 4, tail always selected |
| mHC | 4 residual streams, Sinkhorn 20 iterations |
| Checkpoint | `wtdcode/GLM-5.3-Flash-AWQ-W4A16`: compressed-tensors W4A16, group 128, symmetric, targets `mlp.experts.*` only; everything else bf16 |

The checkpoint is compressed-tensors, not AutoAWQ. On gfx1030 the routed
experts therefore go through `CompressedTensorsWNA16RDNA2MoEMethod` (the
path Flash-Next serves on); the `AutoAWQConfig` changes in this PR only
matter for AutoAWQ-format checkpoints with real zero points.

## Size: this is not a 4x V620 model

Routed experts: 288 x 3 x 4096 x 2048 = 7.25B parameters per MoE layer,
x 42 layers = **304B**. At W4A16 with group-128 fp16 scales (0.516 B/param)
that is about **157 GB** before packing overhead. The dense remainder is
about 9B parameters (KDA 4.7B, DSA + indexer 1.4B, shared experts 1.1B, the
three dense MLPs 0.45B, embeddings + `lm_head` 1.3B, routers 0.05B), served
fp16: **18 GB**. Total about 175 GB of weights.

- 4x V620 = 128 GB: does not fit at any KV budget.
- 8x V620 = 256 GB: fits at TP=8 with about 80 GB for KV, activations and
  graphs. TP=8 spans both PLX switches (backlog #67); PP=2 x TP=4 through the
  pipeline-parallel work in #80/#82 is the alternative.
- KDA state is 34 x 64 x 128 x 128 x 4 B = **228 MB per sequence** (fp32),
  plus conv state; `SEQS=4` costs ~0.9 GB. DSA KV is 11 x 512 x 2 B =
  11.3 KB per token for the latent, plus the fp8 indexer pools (about
  0.4 KB per token) and the bf16 tail ring.

Head counts at TP=8: KDA 8, DSA 8 (the AITER helper pads to 16), indexer
projections are replicated, 288 experts shard by 8.

## What runs where on gfx1030

| Component | Path on gfx1030 (this branch) | Evidence |
| --- | --- | --- |
| Dense linears (bf16 checkpoint, served fp16) | rocBLAS / TunableOp, like every other model on the box | existing |
| Routed experts W4A16 | `CompressedTensorsWNA16RDNA2MoEMethod` -> `moe_gptq_gemm_rdna2`; SwiGLU clamp from `moe_config.swiglu_limit` (this PR) | `test_rdna2_swiglu_limit_matches_shared_expert_clamp` |
| Shared experts, `lm_head`, embeddings, router, KDA and indexer projections | dense fp16; the load fails if the checkpoint quantized any of them | `test_dense_leftovers_follow_the_w4a16_checkpoint` |
| KDA conv | Triton `causal_conv1d_*` (`VLLM_CAUSAL_CONV1D_RDNA2_*=0`) | as the Qwen hybrids |
| KDA prefill | Triton `chunk_kda_with_fused_gate` (FLA chunk kernels); FlashKDA is CUDA-only | `_resolve_kda_prefill_backend` |
| KDA decode | Triton `fused_recurrent_kda` by default; HIP `glm5_kda_recurrent_rdna2` with `VLLM_GLM5_KDA_HIP=1` | `tests/models/glm5next/test_rdna_kda_recurrent.py` |
| DSA attention (prefill and decode) | backend `ROCM_AITER_MLA_SPARSE`; its NoPE / non-fp8 shape always takes `_rocm_sparse_attn_prefill_ragged_triton` (Triton ragged kernel, `BLOCK_D` 512) for both prefill and decode; KV latent in the served dtype | `test_sparse_mla_metadata_skips_aiter_for_rope_free_non_fp8_models` |
| DSA indexer, prefill logits | `rocm_fp8_mqa_logits` -> gfx10x branch -> HIP `mqa_logits_prefill_rdna2` (from the DeepSeek-V4 port; head count is generic, D=128) | `csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu` |
| DSA indexer, decode logits | `rocm_fp8_paged_mqa_logits` -> gfx10x branch -> `_rdna_fp8_paged_mqa_logits` returns None (the HIP kernel is H=64 only; the indexer has 32 heads) -> `fp8_paged_mqa_logits_torch` (torch reference) | reading; a 32-head HIP variant is the first perf follow-up |
| DSA indexer, top-k and kpool | `_C.top_k_per_row_prefill/decode` (built for ROCm with `sampler.cu`); kpool compress / tail / expand are Triton (fp8 e4m3fn on gfx1030) | as DeepSeek-V4 on the box |
| mHC | torch (upstream common model) | upstream |
| Vision tower | not loaded (`--language-model-only`) | recipe |

## Boot blockers found by reading the path (fixed in this pass)

1. **`glm5_kda_recurrent_rdna2` rejected every call from the decode path.**
   The C++ entry squeezed only 4-D tensors, so the `[1, T, H]` beta the layer
   always passes failed `beta must be [B, H]` and `VLLM_GLM5_KDA_HIP=1`
   would have raised on the first decode step. The entry now squeezes
   `[1, T, H, D]` to rank 3 and `[1, T, H]` to rank 2; the Python wrapper
   normalizes the same shapes and checks the unit last stride, and the layer
   passes the merged-conv column slices without the three `.contiguous()`
   copies (the kernel indexes with the token and head strides).
   `tests/models/glm5next/test_rdna_kda_recurrent.py` runs the wrapper against
   a stand-in op with the C++ checks on CPU, and against the HIP op and
   Triton `fused_recurrent_kda` on a gfx1030 build.
2. **`ROCMAiterMLASparseMetadataBuilder.__init__` imported `aiter`
   unconditionally** (`from aiter import dtypes, get_mla_metadata_info_v1`)
   to size the persistent AITER decode metadata. `aiter` is not installed
   on the V620 box (`VLLM_ROCM_USE_AITER=0` in every launcher), so the
   engine would have died at backend init for the 11 DSA layers. GLM-5.3's
   NoPE latent with a non-fp8 cache never reaches the AITER decode
   (`_use_rocm_sparse_triton` is true for every batch), so the builder now
   skips that setup when `sparse_mla_needs_aiter_metadata()` is false. CDNA
   with fp8 or RoPE models is unchanged.
3. **`SparseAttnIndexerKpool.forward_hip` raised unless AITER was enabled**,
   although the ROCm logits dispatch underneath already has gfx10x paths
   (DeepSeek-V4 uses them). The gate is now `hip_sparse_indexer_available()`:
   AITER enabled, or RDNA.

## Risks that only the hardware run can settle

- Triton on gfx1030 with the 512-wide latent: `_sparse_attn_prefill_ragged_kernel`
  runs `tl.dot` at `BLOCK_H=16, BLOCK_D=512, BLOCK_K=16` with fp16 inputs and
  no matrix cores. It should compile; its speed decides whether a HIP sparse
  MLA kernel for this layout (the `glm5_dsa_mla_decode_rdna2` port reads
  expanded 64 x 256 K/V, not the latent) is worth wiring.
- fp8 e4m3fn conversions in the Triton kpool kernels on gfx1030 (software
  conversion; the DeepSeek-V4 path exercises the same Triton fp8 reads).
- The indexer tail cache spec is bf16 (`Glm5NextTailCache`); gfx1030 stores
  bf16 fine, the kpool kernels convert on read.
- `fp8_paged_mqa_logits_torch` for decode: `[B, 32 heads]` x pools per step
  in torch; fine for the boot test, not for serving.
- TP=8 across both PLX switches has not been validated on the box.
- `--mamba-cache-mode align` and `BLOCK_SIZE=1024` are taken from the
  hybrid recipes; the kpool indexer needs `block_size % 128 == 0`.

## ISA check of the nine HIP TUs

`benchmarks/kernels/glm5next_rdna/isa_check.py` runs the build's hipify on
`csrc/rocm/rdna/glm5next/*.cu`, compiles each TU for gfx1030 with
`clang++ -x hip --cuda-device-only -nogpulib` and the build's flags
(`-O3 -DNDEBUG -DENABLE_FP8 -U__HIP_NO_HALF_*`), and reads the amdhsa
metadata back. No GPU, no device libraries; the HIP public headers come from
a ROCm install or from checkouts of `ROCm/HIP` and `ROCm/clr`. Numbers below:
clang 22.1.3 with the ROCm 7.2.4 headers. All nine TUs also compile with
clang 18 and the ROCm 6.2.4 headers (which predate `__shfl_xor_sync`; the
harness supplies it). The box's ROCm 7.14 / clang 23 will differ in detail.

| TU | kernel | VGPR | SGPR | LDS B | scratch B | spills V/S | max WG | waves/SIMD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `glm5_dsa_indexer_rdna2` | `glm5_dsa_first_key_kernel` | 42 | 67 | 0 | 0 | 0/0 | 1024 | 16 |
| `glm5_dsa_indexer_rdna2` | `glm5_dsa_indexer_pool_kernel` | 60 | 72 | 0 | 0 | 0/0 | 1024 | 16 |
| `glm5_dsa_mla_decode_rdna2` | `glm5_dsa_mla_decode_kernel` | 44 | 84 | 8204 | 0 | 0/0 | 1024 | 16 |
| `glm5_kda_decode_rdna2` | `glm5_kda_decode_rdna2_kernel` | 82 | 74 | 1956 | 0 | 0/0 | 1024 | 11 |
| `glm5_kda_prefill_delta_h_rdna2` | `glm5_kda_prefill_delta_h_rdna2_kernel` | 80 | 70 | 16896 | 272 | 0/0 | 1024 | 12 |
| `glm5_kda_prefill_kkt_rdna2` | `glm5_kda_prefill_kkt_rdna2_kernel` | 118 | 84 | 49152 | 0 | 0/0 | 1024 | 8 |
| `glm5_kda_prefill_o_rdna2` | `glm5_kda_prefill_o_rdna2_kernel` | 256 | 104 | 0 | 160 | 36/0 | 256 | 4 |
| `glm5_kda_prefill_prep_rdna2` | `glm5_kda_prefill_prep_rdna2_kernel` | 139 | 70 | 17152 | 0 | 0/0 | 128 | 7 |
| `glm5_kda_prefill_solve_wy_rdna2` | `glm5_kda_prefill_solve_wy_rdna2_kernel` | 127 | 74 | 59136 | 0 | 0/0 | 1024 | 8 |
| `glm5_kda_recurrent_rdna2` | `glm5_kda_recurrent_rdna2_kernel<fp16>` | 78 | 70 | 516 | 0 | 0/0 | 1024 | 12 |
| `glm5_kda_recurrent_rdna2` | `glm5_kda_recurrent_rdna2_kernel<bf16>` | 79 | 68 | 516 | 0 | 0/0 | 1024 | 12 |

Waves/SIMD is the VGPR-limited bound (1024 VGPRs per lane per SIMD, blocks
of 8, at most 16 waves); LDS limits `solve_wy` and `kkt` to one workgroup
per WGP on top of that. `prefill_o` uses dynamic LDS (`GLM5_LDS_TOTAL_BYTES`),
which the metadata does not show.

What the check found:

- **The wired kernel is fine.** `glm5_kda_recurrent_rdna2_kernel` compiles
  to 78/79 VGPRs, 516 B of LDS and no spills for both dtypes.
- **`__launch_bounds__` on the 256- and 128-thread prefill kernels.** Without
  it the compiler assumes 1024 threads per workgroup and caps every kernel at
  128 VGPRs. `prefill_o` paid for that with 158 VGPR spills and 640 B of
  scratch per lane, `prep` with 2 spills. Stating the real workgroup size
  (`__launch_bounds__(GLM5_O_THREADS)`, `__launch_bounds__(GLM5_PREP_THREADS)`)
  takes `prep` to 139 VGPRs with no scratch, and `prefill_o` to 256 VGPRs
  with 36 spills and 160 B of scratch at 4 waves/SIMD; the register pressure
  of its first-cut tiling is real and its own PERF NOTE already plans a
  chunk-16 variant. `delta_h` fits in 80 VGPRs either way and was left alone.
  These are occupancy facts, not `amdgpu_waves_per_eu` pins.
- **`-DNDEBUG` is load-bearing.** The HIP 7 `__shfl_xor_sync` validates its
  mask with `__hip_assert`; compiled without `NDEBUG`, that check alone takes
  every KDA kernel to the 128-VGPR cap with spills (`recurrent`: 47/16 and
  63/16 V/S spills, 176-192 B scratch). CMake's Release and RelWithDebInfo
  builds define `NDEBUG`, so the fork's build is fine; a Debug build would
  not be, and neither is any harness that omits the flag.

## Checklist for the first run

1. On the box, with the gfx1030 build:

   ```bash
   .venv/bin/python -m pytest tests/kernels/quantization/test_glm5next_rdna_contract.py \
       tests/models/glm5next/test_rdna_kda_recurrent.py -v
   ```

   The HIP-vs-Triton case runs only there.
2. Confirm the dimensions and the dense leftovers against the checkpoint:

   ```bash
   .venv/bin/python - <<'EOF'
   import json, sys
   cfg = json.load(open(sys.argv[1] + "/config.json"))
   tc = cfg.get("text_config", cfg)
   for k in ("num_hidden_layers", "hidden_size", "moe_intermediate_size",
             "n_routed_experts", "num_experts_per_tok", "kv_lora_rank",
             "qk_rope_head_dim", "index_n_heads", "index_kpool", "swiglu_limit"):
       print(k, tc.get(k))
   q = cfg["quantization_config"]
   print("format", q.get("quant_method"), "ignore", q.get("ignore"))
   idx = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["weight_map"]
   packed = sorted({k.rsplit(".", 2)[0] for k in idx if k.endswith("weight_packed")})
   print(len(packed), "packed modules; non-expert:",
         [p for p in packed if ".experts." not in p][:10])
   EOF
   ```

   Every packed module must be under `mlp.experts`.
3. Boot eagerly on 8 GPUs:

   ```bash
   MODEL=/path/to/GLM-5.3-Flash-AWQ-W4A16 bash tools/rdna/serve_rdna.sh RECIPE=glm53-flash
   ```

   In the log: the attention backend for the DSA layers is
   `ROCM_AITER_MLA_SPARSE`, there is no `aiter` import error, the W4A16 MoE
   method is the RDNA2 one, and the first request returns text.
4. Greedy and prefix-cache probes (the ones the Flash-Next and DeepSeek-V4
   recipes use), then PPL with `tools/rdna/port_v031/ppl_probe.py`.
5. Graphs: `EAGER=0` with the breakable-graph lines in the recipe.
6. `VLLM_GLM5_KDA_HIP=1` A/B against Triton: identical greedy output, decode
   ITL.
7. Perf follow-ups, in order of expected gain: a 32-head HIP paged MQA logits
   kernel for the indexer decode; the Triton sparse MLA timing at 512 wide;
   W4A8 MoE opt-in (`VLLM_RDNA2_W4A8_SDOT4=1`), already validated on
   Flash-Next.
