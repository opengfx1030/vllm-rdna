# Qwen4 readiness and backport map (gfx1030 / RDNA2 fork)

Status: 2026-11-03. Research only: no Qwen4 weights exist, no GPU was used.
Every claim is linked; anything not directly observable is marked
[UNCONFIRMED] (reported by a third party, source not verifiable) or
[INFERENCE] (derived here from public numbers).

Update 2026-11-03: the opt-in dense MLP and four clean upstream backports are
now on `feat/qwen4-prep` (section 7). They are CPU-validated only; GPU
validation is queued behind a free window and is recorded as pending below.

The fork is `opengfx1030/vllm-rdna`, branch `rdna_extras`, vLLM v0.33 line,
serving on 8x Radeon PRO V620 (gfx1030): wave32, no bf16 dot product, no FP8,
no matrix cores; fp16 + `v_dot2_f32_f16` + int8 `sdot4`.

## 1. What is public about Qwen4 today

Qwen4 has not shipped. As of 2026-11-03 there is no `Qwen/Qwen4*` repository
on Hugging Face (the HF API returns its generic 401 for every non-public repo,
including control names such as `Qwen/Qwen5`), no `qwen4` model directory in
transformers, and no `qwen4` model class in vLLM or SGLang. The only public
`qwen4*` artifacts belong to **Qwen3.8-Flash-Next**, which Qwen calls "an early
preview of the architecture used in Qwen4".

| Source | Date | Status | What it implies for Qwen4 |
|---|---|---|---|
| [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next) | 2026-08-26 | Released | Official architecture preview: GDN + QSA hybrid, Gated Residual (4 streams), n-gram/PLE embedding table, 125B MoE with 6B active, 51B n-gram table, 4B MTP head, 262K native context extensible to 1M. Qwen says the Qwen4 family will be built on this. |
| [HF config.json of Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) | 2026-08-26 | Released | `architectures: [Qwen4ExpForConditionalGeneration]`, `model_type: qwen4_exp`; the only `qwen4*` config that exists. Exact geometry in section 2. |
| [transformers PR 48337](https://github.com/huggingface/transformers/pull/48337) "Add Qwen4Exp model" | opened 2026-08-26, merged 2026-08-26 | Merged | Adds `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpTextConfig`, `Qwen4ExpConfig`, `Qwen4ExpVisionConfig`; [docs](https://huggingface.co/docs/transformers/main/model_doc/qwen4_exp). |
| transformers follow-ups: [48456](https://github.com/huggingface/transformers/pull/48456), [49840](https://github.com/huggingface/transformers/pull/49840), [50138](https://github.com/huggingface/transformers/pull/50138) | 2026-09-01 .. 2026-10-29 | Merged | Masking, indexer cache lifecycle, `_no_split_modules` fixes. No architecture change. |
| transformers open: [50169](https://github.com/huggingface/transformers/pull/50169), [49923](https://github.com/huggingface/transformers/pull/49923), [49694](https://github.com/huggingface/transformers/pull/49694) | 2026-10-17 .. 2026-10-30 | Open | Vectorised indexer for prefix masks, left-padding NaN fix, skip block scoring when the budget covers everything. Quality/perf only. |
| [vLLM PR 53896](https://github.com/vllm-project/vllm/pull/53896) "Support Qwen3.8-Flash-Next" | opened 2026-08-26, merged 2026-08-31 | Merged | Adds `vllm/models/qwen4_exp/` with NVIDIA and AMD paths, `Qwen4ExpForCausalLM` / `ForConditionalGeneration` / `Qwen4ExpMTP` registry entries. |
| vLLM QSA/PLE follow-ups [54513](https://github.com/vllm-project/vllm/pull/54513), [54517](https://github.com/vllm-project/vllm/pull/54517), [54873](https://github.com/vllm-project/vllm/pull/54873), [54890](https://github.com/vllm-project/vllm/pull/54890), [54915](https://github.com/vllm-project/vllm/pull/54915) | 2026-09-02 .. 2026-09-07 | Merged | Indexer split prefill/decode, fused PLE kernels, fp8 indexer cache, compact logits workspace. These are the kernels our AMD port is based on. |
| [vLLM PR 55557](https://github.com/vllm-project/vllm/pull/55557) | merged 2026-09-16 | Merged | `fp8_e4m3` main KV on the QSA path (NVIDIA). Irrelevant to RDNA2 (no FP8). |
| [vLLM PR 57387](https://github.com/vllm-project/vllm/pull/57387) | merged 2026-10-02 | Merged | Use upstream transformers Qwen4-Exp configs/processor. |
| [vLLM PR 61529](https://github.com/vllm-project/vllm/pull/61529) | merged 2026-10-29 | Merged | fp8 KV and indexer caches on the **AMD** QSA path. Not usable on gfx1030 (no FP8); a backport hazard only. |
| [vLLM PR 60650](https://github.com/vllm-project/vllm/pull/60650) | opened 2026-10-08 | Open (needs rebase) | Serve the AMD path in fp16. Our fork already does this independently; the PR is the upstream equivalent. |
| [vLLM PR 60670](https://github.com/vllm-project/vllm/pull/60670), [60657](https://github.com/vllm-project/vllm/pull/60657) | opened 2026-10-08 | Draft | NF4 n-gram table for AMD, fused HC combine+mix for gfx1100. The latter targets RDNA3, not RDNA2. |
| [vLLM PR 59279](https://github.com/vllm-project/vllm/pull/59279) | opened 2026-09-29 | Open, needs rebase | Decode context parallelism for Qwen4Exp QSA. Not relevant to 8x V620 TP4. |
| [SGLang cookbook PR 36496](https://github.com/sgl-project/sglang/pull/36496); `python/sglang/srt/models/qwen4_exp.py` on main | 2026-08-26 | Merged / present | A second runtime names the architecture `qwen4_exp`; merged fixes include [42478](https://github.com/sgl-project/sglang/pull/42478) and [46543](https://github.com/sgl-project/sglang/pull/46543). |
| [llama.cpp PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742) "model: add Qwen3.8-Flash-Next (qwen4exp)" | opened 2026-08-26, merged 2026-08-27 | Merged | Third runtime; later merged fixes 27880, 27941, 28770, 29824, 30602. [PR 31185](https://github.com/ggml-org/llama.cpp/pull/31185) (external PLE file) is open. |
| [llama.cpp issue 31072](https://github.com/ggml-org/llama.cpp/issues/31072) "Get ready for qwen4" | 2026-10-24 | Open | Points at an Intel XPU kernel PR for **Qwen4-27B dense**; contains the only public per-feature comparison (see below). |
| [vllm-xpu-kernels PR 22](https://github.com/analytics-zoo/vllm-xpu-kernels/pull/22) "Enable Qwen3.8-flash-next 8 cards fp8 & int4 and optimize qwen4-27b" | 2026-10-19 | Merged per reports; repository now returns 404 | TP2 FP8 and TP1/TP2 INT4 SYCL kernels for a "Qwen4-27B dense" model. No QSA weights were available for end-to-end testing. [UNCONFIRMED] |
| [Huxiu report](https://www.huxiu.com/article/4900440.html) on the Intel PR and the llama.cpp reverse-engineering | 2026-10-26 | News | Summarises the 27B comparison: dense, hidden 5120, four hyper-connection streams, sigmoid output gate, QSA, FFN intermediate likely reduced. [UNCONFIRMED] |
| [Apsara Hangzhou press coverage](https://llmhangar.com/blog/deploy-qwen4) | 2026-09-22 | Official statement reported | Qwen4 "currently in training"; Qwen4.5/Qwen5 roadmap at 5-10T parameters. |
| [Apsara Amsterdam report](https://cellcog.ai/blog/qwen-4-release-date/) | 2026-10-29 | Stage slide + attendee report | Lineup: Qwen4-Max, Qwen4-Flash, Qwen4-Plus, Qwen4-27B "coming soon"; an attendee reported the CTO saying "next week". No weights, card, license or price. [UNCONFIRMED] |
| [Qwen Code PR 15762](https://github.com/QwenLM/qwen-code/pull/15762) | merged 2026-10-28 | Merged | Sizes every `qwen4` model at 1M context. A client-side setting, not a model card. |

Net: the family name, the lineup and the 27B dense leak are public; the
**only** artifact with a downloadable config is the Qwen4Exp preview.

## 2. Features implied by the public evidence

### 2.1 Qwen3.8-Flash-Next (`qwen4_exp`), from its released config

| Feature | Value |
|---|---|
| Text backbone | 48 layers, hidden 2560, vocab 248320, `model_type: qwen4_exp_text` |
| Layer pattern | `12 x (GDN -> MoE, GDN -> MoE, GDN -> MoE, QSA -> MoE)`, `full_attention_interval: 4` |
| Full attention | 24 Q heads / 2 KV heads, `head_dim` 256, `partial_rotary_factor` 0.25, rope theta 1e7, mrope sections [11, 11, 10] |
| QSA indexer | 4 query heads / 1 KV head, `indexer_head_dim` 128, `indexer_budget` 2048 tokens (512 blocks), `indexer_compress_ratio` 4 |
| Linear attention (GDN) | 16 K/Q heads and 48 V heads, key/value head dim 128, conv kernel 4, `output_gate_type: sigmoid`, `mamba_ssm_dtype: float32` |
| Gated Residual | 4 streams (`hc_count`), low-rank 320 (`hc_lowrank`) |
| MoE | 512 experts, top-10 routed + 1 shared, `moe_intermediate_size` 640, shared 640, `norm_topk_prob: true` |
| PLE / n-gram | `ple_layer_ids: [2]`, `ple_embed_dim` 2560, conv kernel 4, `ngram_size` 3, 8 heads per n-gram order, base vocab 20,000,000, 128 checkpoint shards |
| MTP | 1 layer, `full_attention`, `hybrid: true`, rope theta 1e7 |
| Vision | 27-layer ViT, hidden 1152, patch 16, spatial merge 2, out hidden 2560 |
| dtype | bfloat16 (the AMD vLLM path in v0.33 hard-requires bf16; our fork serves fp16) |

### 2.2 Qwen4-27B, from the leaked Intel kernels (no weights)

| Feature | Qwen3.8-27B (`qwen3_5`, public) | Qwen4-27B (leak) |
|---|---|---|
| Type | Dense | Dense (no MoE kernels target it) |
| Hidden | 5120 | 5120 |
| Layers | 64, 3:1 linear/full | Unknown |
| GDN | 48 V / 16 K heads, dim 128 | Same (`QKVZ` 8192, `BA` 48 per rank at TP2) |
| Full attention | 24 Q / 4 KV, head dim 256, output gate swish | 24 Q / 4 KV, head dim 256, **sigmoid** output gate |
| Hyper-connections | None | 4 streams (`[M, 20480]`), rank 320 |
| QSA | None | Kernels present; no QSA weights supplied to the leak |
| FFN | 17408 | ~14336 [INFERENCE from tuning tables] |
| MTP | 1 layer | Present (tested with 4 draft tokens) |
| PLE | None | Unknown |
| Vision | None for the text model | Not indicated |

The Intel PR is gone (404) and the numbers come from a llama.cpp maintainer's
reading of its kernels plus a Chinese news summary; treat the whole row as
[UNCONFIRMED] until a model card exists. What is consistent across all sources:
Qwen4-27B stays dense, adopts the four-stream residual and QSA from Flash-Next,
and **cannot be loaded by existing `qwen3_5` code** because the residual path
changed.

### 2.3 The rest of the family

Qwen4-Max, Qwen4-Flash and Qwen4-Plus have no published geometry. The only
defensible planning assumption is that open weights, if any, arrive first for
27B (as with Qwen3.8-27B, Apache 2.0) and that Flash/Plus/Max reuse the
Flash-Next MoE shape at larger scale. Marked [INFERENCE].

## 3. Diff against this fork

### 3.1 Base situation

vLLM v0.33.0 already ships `vllm/models/qwen4_exp/` with both an NVIDIA and an
AMD path. Our fork is not "missing the model"; it has **modified or added 28
files** in that tree relative to `v0.33.0` (+5610 / -188 lines) plus a fork-only
speculative-decoding proposer. The upstream model tree has since moved: 20 files
changed between `v0.33.0` and `upstream/main` (821 insertions, 389 deletions),
including one rename and two new `common/` modules.

Worktree facts used below: base `v0.33.0`; `upstream/main` at `5d55ef1590`
(2026-11-03); empirical cherry-pick matrix from section 4.

### 3.2 Have (present and exercised in this fork)

| Area | Where | Notes |
|---|---|---|
| QSA sparse attention | `vllm/models/qwen4_exp/amd/qsa.py`, `amd/ops/qsa.py`, `amd/ops/qsa_prepare.py`, `amd/ops/qsa_rdna2.py`, `common/qsa_cache.py` | Triton sparse split-K + HIP store/compress/MQA indexer (`qsa_store_cache_rows_rdna2`, `qsa_compress_groups_rdna2`, `qsa_mqa_paged_rdna2`). Compressed KV ring, indexer side caches. |
| QSA cache formats | `amd/qsa.py`, `amd/ops/qsa.py` | fp16/bf16 main cache plus `int4_fp16_block` and `ROCM_OCTAVE` low-bit KV; upstream has bf16 (+fp8 on the NVIDIA path, +fp8 on AMD since PR 61529). |
| GDN linear attention | `csrc/rocm/gdn_*_rdna2.cu`, `amd/model.py` | HIP decode kernel and 5-kernel prefill chain (`prep`, `kkt`, `solve_wy`, `delta_h`, `o`); fp16 finite-normalisation fix. |
| Gated Residual / HC | `amd/hyperconnection.py`, `amd/ops/hc.py`, `amd/ops/hc_rdna2.py`, `csrc/rocm/hc_prefill_rdna2.cu` | Fused decode for M <= 8 (`rdna_hc_up_gate_mix`, `rdna_se_gate_up_silu`, `rdna_se_down_gated`), shared-expert pair up to M <= 32, HIP prefill for M > 8, HC split-K in MTP verify batches. |
| MoE kernels | `compressed_tensors_moe_wna16_rdna2.py`, `compressed_tensors_moe_w4a8_int8.py`, `rocm_moe_rdna.py`, `tools/rdna/mono/` | W4A16 expert GEMM, W4A8 `sdot4` int8 path, fused shared expert (`VLLM_RDNA_FUSED_SE`), routing capture, K2 expert offload, monokernel `--monokernel`. |
| MTP | `vllm/v1/spec_decode/qwen4_exp.py`, `amd/mtp.py`, `amd/mono/rdna2/k1.py`, `k2.py` | Fork-only `Qwen4ExpMTPProposer` (per-group block tables, multi-stream feedback width), verify glue, K1/K2 drafter paths. Upstream v0.33/main has **no** Qwen4Exp proposer (verified by `git grep`). |
| PLE / n-gram | `common/ngram_embedding.py`, `common/ple_pp.py`, `amd/ple_layer.py`, `amd/ops/ple_conv_rdna2.py` | bf16 table, INT4 `ples_int4` table, EXL3 table via the CPU sidecar, pinned-host/disk offload (`VLLM_PLE_CPU_OFFLOAD`), pipeline-parallel support (`ple_pp.py`), HIP dilated short conv. |
| Quant paths | `layers/quantization/exl3*.py`, `gptq_gemm_rdna2`, `q_gemm_rdna2_awq_prefill`, `compressed_tensors_moe_*` | AWQ W4A16 dense, GPTQ W4A16, EXL3 (dense/MoE/PLE), W4A8 int8 MoE, W8A16 MoE. All exercised on Flash-Next except W8A16. |
| fp16 AMD serving | `amd/qsa.py` (`supported_dtypes` includes fp16), `amd/indexer_qsa.py` | Upstream v0.33 raises `Qwen4Exp QSA currently requires BF16`; our fork serves fp16. The upstream equivalent, PR 60650, is still open. |
| RDNA2 attention/allreduce | `csrc/rocm/fa_rdna2*`, `VLLM_USE_RDNA2_FA`, `VLLM_RDNA_AR` | Not Qwen4-specific but required by every recipe. |

### 3.3 Differ (same functionality, different code shape)

| Area | Upstream v0.33 | Upstream main (2026-11-03) | This fork |
|---|---|---|---|
| QSA module layout | `amd/indexer_qsa.py`, `nvidia/indexer_qsa.py`, `common/qsa_cache.py` | renamed `*_qsa.py` -> `*_qsa_indexer.py`; new `common/qsa.py`; FlashAttention dependency removed (PR 64110) | v0.33 layout plus HIP dispatchers; no `common/qsa.py` |
| AMD QSA dtype | bf16 only | bf16 + fp8 e4m3 KV/indexer (PR 61529) | bf16 + fp16 + int4-fp16-block/Octave |
| PLE offload | CPU offload hooks | `ple_offload_mode` config + disk offload (PR 59958) | own CPU sidecar + int4/exl3 tables + PP shim; no `ple_offload_mode` surface |
| PLE + PP | PP > 1 refused | allowed via model/model_state/mtp changes (PR 60234) | allowed via `common/ple_pp.py` (different mechanism) |
| MTP | EagleProposer | still EagleProposer; no Qwen4Exp subclass | fork-only `Qwen4ExpMTPProposer` |

### 3.4 Missing (upstream has it, we do not)

| Item | Upstream | Why it matters on RDNA2 |
|---|---|---|
| Dense MLP path | none anywhere | `Qwen4ExpDecoderLayer` **always** constructs `Qwen4ExpSparseMoeBlock` (`amd/model.py` line ~265). A dense Qwen4-27B config (`num_experts` absent/0) has no MLP class. This is the single largest gap for the most likely open checkpoint. |
| `common/qsa.py` abstraction + rename (PR 64110) | yes | Needed before any future upstream Qwen merge; touches `amd/qsa.py`, `amd/qsa_indexer.py`, `nvidia/qsa.py` — all heavily modified here. |
| int64 HC/PLE row offsets (PR 64820) | yes | Correctness at large tables; our `common/ops/ple.py` has int64 in the Triton kernel but `amd/ops/hc.py` does not have the upstream fix. |
| QSA indexer workspace reservation during profiling (PR 60490) | yes | Prevents under-sized KV/workspace at long context; NVIDIA path only upstream. |
| Quantized q/k/v + BF16 indexer load fix (PR 62318) | yes | Real checkpoints mix quantized projections with a bf16 indexer; our AWQ/GPTQ Flash-Next loads may hit the same bug. |
| Per-layer fused shared expert selection (PR 61457) | yes | Cross-model change to `fused_moe`; our `VLLM_RDNA_FUSED_SE` is a fork-local equivalent and will need reconciling. |
| fp8 KV/indexer on AMD (PR 61529) | yes | Cannot run on gfx1030 (no FP8). Port only the API/plumbing if it reduces conflicts; never enable. |
| SM100/SM103 skinny-GEMM plans (PRs 64482, 64650) | yes | NVIDIA-only; intentionally absent. |
| Qwen4 itself | no | No `qwen4` model type, class, registry entry or config parser exists in transformers, vLLM or SGLang. |

Four rows above are no longer missing on `feat/qwen4-prep`: int64 offsets
(PR 64820), QSA workspace reservation (PR 60490), quantized q/k/v loading
(PR 62318) and PLE disk offload (PR 59958) are cherry-picked there, one commit
each. They are textually clean and compile, but none has run on a GPU yet.

## 4. Backport map

Method: a detached worktree at our current head (`0fc943cb0f`) was used to run
`git cherry-pick -n` for each upstream commit and record textual conflicts. That
measures merge friction, not behavioural compatibility.

Order is the recommended landing order. "Clean" means the patch applies
textually; tests still have to run.

| # | Upstream commit / PR | Date | Files touched | Textual conflict here | RDNA2 concerns | Order |
|---|---|---|---|---|---|---|
| 1 | `58f245c69d` PR 64820 int64 HC/PLE row offsets | 2026-10-30 | `amd/ops/hc.py`, `common/ops/ple.py`, `nvidia/ops/hc.py` | Clean | Cheap correctness win; verify HC offset math with the fork's M<=8 fused kernels. | 1 |
| 2 | `90309f6b74` PR 60490 QSA indexer workspace reservation | 2026-10-27 | `nvidia/qsa.py` | Clean | NVIDIA-only; port the pattern to `amd/qsa.py` so long-context profiling is honest. | 2 |
| 3 | `a91a1d06b7` PR 62318 quantized q/k/v + BF16 indexer load | 2026-10-27 | `nvidia/model.py`, `nvidia/qsa.py` | Clean | Mirror the loader check into `amd/qsa.py`/`amd/model.py`; our AWQ/GPTQ checkpoints use quantized q/k/v. | 3 |
| 4 | `b44a48cef1` PR 59958 PLE disk offload | 2026-10-30 | `common/ngram_embedding.py`, `nvidia/{model_state,ngram_embedding}.py`, `vllm/config/engram.py`, tests | Clean | Overlaps our int4/exl3 sidecar. Port the `ple_offload_mode` config surface, keep the sidecar; do not run two offload mechanisms. | 4 |
| 5 | `3765000a37` PR 64482 and `900b8b08e2` PR 64650 SM100/SM103 skinny plans | 2026-10-29 / 31 | `nvidia/low_latency_gemm.py` | Clean | No RDNA2 effect; take for parity or skip. | 5 |
| 6 | `af4db2a675` PR 61457 per-layer fused shared expert | 2026-10-28 | `fused_moe/*`, many model files incl. `qwen3_5*.py` | Not measured as a Qwen4 patch (broad) | Reconciles with `VLLM_RDNA_FUSED_SE`; test Flash-Next shared width 640 at TP4 and the M32 shared-expert fused path. | 6 |
| 7 | `5c1eb11cb3` PR 60234 PP for PLE + MTP | 2026-10-27 | `amd/model.py`, `amd/model_state.py`, `amd/mtp.py`, `nvidia/*`, `vllm/model_executor/models/config.py`, `vllm/v1/core/kv_cache_planning.py` | **Conflicts: 6 files** | Our `common/ple_pp.py` already makes PP3 work; adopting upstream would double-implement. Pick one before porting. | 7 |
| 8 | `45fd60d4d5` PR 61529 fp8 KV/indexer on AMD QSA | 2026-10-29 | `amd/qsa.py`, `amd/indexer_qsa.py`, `amd/ops/qsa.py`, `amd/ops/qsa_prepare.py`, tests | **Conflicts: 5 files** | gfx1030 has no FP8. Never enable; porting only buys naming/test parity and reduces the next conflict. | 8 |
| 9 | `d492ee7862` PR 64110 QSA rename + FlashAttention removal | 2026-10-30 | `amd/qsa.py`, `amd/indexer_qsa.py` -> `qsa_indexer.py`, `common/qsa.py` (new), `common/qsa_cache.py` -> `qsa_indexer.py`, `nvidia/qsa.py`, tests | **Conflicts: 5 files** | Manual port, not a cherry-pick: our HIP dispatchers live in the renamed files. Do last, on its own branch, with the full QSA test set. | 9 |

Notes on the matrix:

- The three "clean" Qwen4 fixes are recent (2026-10-27 .. 10-30) and small.
  Landing them now is cheap and shrinks the eventual merge.
- The two refactors (PR 64110, PR 61529) are the real risk. Both touch
  `amd/qsa.py` and the indexer, which the fork has rewritten for RDNA2.
- Upstream also refactored `vllm/config/speculative.py` between v0.33 and main
  (789 lines changed). Our only change there is the 9-line
  `use_qwen4_exp_mtp()` helper, so the conflict is small but the surrounding
  API is not; re-read that file before rebasing the fork onto a newer upstream.
- Do **not** cherry-pick PR 60650 (AMD fp16). The fork already implements fp16
  in the same files; taking both would duplicate the dtype policy.

Landed on `feat/qwen4-prep` on 2026-11-03, in this order, one commit each:

| Order | Upstream | Local commit | CPU check | GPU validation |
|---|---|---|---|---|
| 1 | `58f245c69d` PR 64820 | `22196687e9` | `py_compile` | pending |
| 2 | `90309f6b74` PR 60490 | `d92580ced1` | `py_compile` | pending |
| 3 | `a91a1d06b7` PR 62318 | `b1938a5414` | `py_compile`; its new test needs torch | pending |
| 4 | `b44a48cef1` PR 59958 | `4c217d501b` | `py_compile`; its new tests need torch | pending |

The four applied without conflicts in sequence. GPU validation is deliberately
deferred: it needs a free V620 window and a Flash-Next checkpoint, and the
PLE disk-offload path also needs a host with the int4 table on disk.

## 5. Recipe and launcher readiness

### 5.1 What a `qwen4*.env` recipe needs

The launcher (`tools/rdna/serve_rdna.sh`) already hardcodes `--dtype float16`,
so fp16 forcing is automatic; what changes per model is geometry and budget.

```text
SERVE_ENTRY=cli | api
SERVED_NAME=qwen4-27b
PORT=...
TP=<2|4|8>            # weights per rank vs 29 GiB
PP=1
HOST=...
MTP=0                 # 2 only after acceptance is measured on gfx1030
ATTN=fa               # QSA layers select their own backend; FA covers the rest
MAXLEN=262144
BLOCK_SIZE=1024       # 2048 once MAXBAT accounts for decode tokens
SEQS=...
MAXBAT=...
KV=<bytes>            # or KV=auto with GMEM
GMEM=0.90
CG_MODE=FULL_AND_PIECEWISE
COMPILE_MODE=3
FEATURES="trust-remote-code prefix-caching mamba-align prompt-tokens-details ..."
PLE=none | bf16 | int4 | int4-disk | exl3   # only if the checkpoint ships an n-gram table
```

Plus the existing RDNA env block (`VLLM_RDNA_AR`, `VLLM_GDN_HIP_PREFILL`,
`VLLM_RDNA_FUSED_HC`, `VLLM_RDNA_FUSED_SE`, `HSA_FORCE_FINE_GRAIN_PCIE=1`,
`HSA_ENABLE_SDMA=0`, ...). The Qwen4-27B path additionally needs the HC flags
even without MoE, because four-stream residual is a dense-path feature.

`mamba-align` is mandatory: GDN state and QSA indexer state are per-request
side caches; the block size must be a multiple of the mamba block.

### 5.2 Memory on 29 GiB V620s

Assumptions: 29 GiB usable per card, `GMEM=0.90` -> 26.1 GiB, minus ~1.6 GiB
non-torch/graph overhead -> **24.5 GiB** for weights + KV + activations.
The Qwen4-27B geometry assumes 64 layers / 16 full-attention layers
(Qwen3.8-27B's shape, [INFERENCE]).

KV per token:

- Flash-Next: 12 full layers x 2 KV heads x 256 x 2 (K,V) x 2 B = 24 KiB/token.
- Qwen4-27B (64 layers, 16 full): 16 x 4 x 256 x 2 x 2 B = 64 KiB/token.
- GDN state: 48 V heads x 128 x 128 x 4 B per linear layer = 3.0 MiB/layer;
  ~108 MiB/sequence at 36 layers, ~144 MiB at 48.

| Model / quant | TP | Weights per rank | KV headroom | Max KV tokens | Verdict |
|---|---:|---:|---:|---:|---|
| Flash-Next 125B fp16 | 4 | 58.2 GiB | -33.7 | 0 | does not fit |
| Flash-Next 125B fp16 | 8 | 29.1 GiB | -4.6 | 0 | does not fit |
| Flash-Next 125B AWQ W4A16 | 4 | 16.4 GiB | 8.1 | ~355k | current production shape |
| Flash-Next 125B AWQ W4A16 | 8 | 8.2 GiB | 16.3 | ~713k | more KV, half the replicas |
| Qwen4-27B fp16 | 2 | 25.2 GiB | -0.7 | 0 | does not fit |
| Qwen4-27B fp16 | 4 | 12.6 GiB | 11.9 | ~195k | viable, no MTP headroom |
| Qwen4-27B AWQ W4A16 | 1 | 14.1 GiB | 10.4 | ~170k | viable, one card |
| Qwen4-27B AWQ W4A16 | 2 | 7.1 GiB | 17.4 | ~285k | recommended 2-card cell |
| Qwen4-27B AWQ W4A16 | 4 | 3.5 GiB | 21.0 | ~343k | recommended 4-card cell |
| Qwen4-27B EXL3 3.0 bpw | 2 | 4.7 GiB | 19.8 | ~324k | viable |

(W4A16 estimated at 4.5 bits/weight including scales; EXL3 at 3.0. Real
checkpoint sizes will differ by a few percent.)

Throughput sanity check: a dense 27B at 4.5 bpw is ~15 GB of weights, so one
V620 at 512 GB/s is bounded near 34 tok/s aggregate and realistically half that;
TP2/TP4 mainly buy KV and prefill, not decode bandwidth. This is why the 125B
MoE remains the better V620 shape despite 20x the total parameters.

### 5.3 Concrete recipes to add

| Recipe | When | Shape |
|---|---|---|
| `qwen4-27b-awq.env` | 27B dense, AWQ W4A16 checkpoint | TP4, MTP0, `ATTN=fa`, `MAXLEN=262144`, `KV` sized for ~300k tokens, `PLE=none`, HC flags on |
| `qwen4-27b-exl3.env` | 27B dense, EXL3 | TP2, `KV` ~300k, EXL3 env block |
| `qwen4-27b-tp2.env` | single replica on 2 cards, 2 replicas per node | TP2, `GMEM=0.90` |
| `qwen4-flash.env` | if Qwen4-Flash ships as a Flash-Next-like MoE | copy of `flashnext.env`, TP4, `PLE` per checkpoint |
| `qwen4-max.env` | only if open weights appear | TP8 + expert offload or absent; do not plan around it |

Do not create `qwen4-27b-fp16.env`: at TP2 it does not fit and at TP4 it is
slower than AWQ with no quality reason on a 29 GiB card.

### 5.4 Attention backend

`ATTN=fa` selects the RDNA2 FA kernels for layers that use full attention.
QSA layers do not go through that backend: they use the QSA implementation and
its own sparse kernels. A Qwen4 checkpoint that drops `indexer_*` fields makes
`use_qsa` false and falls back to `Qwen3NextAttention` (dense full attention) —
if that happens on 27B, the 64 KiB/token KV estimate is unchanged but attention
compute grows; measure before trusting the recipe.

### 5.5 MTP

The launcher maps `MTP=N` to `{"method": "mtp", "num_speculative_tokens": N}`.
This requires the speculative config to recognise the *new* model type as an
MTP-capable Qwen4 (`qwen4_exp_mtp` today, a new name later). On V620 the measured
picture is consistent: MTP-2 costs throughput in most production cells
(0.73x-0.85x of MTP0 in the 2026-07/09 sweeps) and wins only on specific
8x16k/1k cells. Default `MTP=0`; add `qwen4-27b-mtp2.env` only after measuring
acceptance on the released checkpoint. Do not assume the Flash-Next MTP head
transfers to 27B — the leak says only "MTP present".

## 6. What can only be done after release

1. Confirm whether Qwen4-27B is dense in the shipped config, its layer count,
   FFN width, whether `indexer_*` fields and QSA/indexer weights exist, and
   whether PLE / vision are present.
2. Confirm the dense MLP on the released config. An opt-in dense MLP is
   already on `feat/qwen4-prep` (section 7.1); it selects on `num_experts` /
   `dense_mlp` and reuses the Qwen2-style `gate_up_proj`/`down_proj`
   projections. After release, check the real `intermediate_size` and weight
   names, then enable it. Everything else (HC, QSA, GDN, MTP) already has an
   RDNA2 path.
3. Register the new `model_type` and MTP method name (`qwen4*_mtp`) in
   `vllm/transformers_utils/model_arch_config_convertor.py` and
   `vllm/config/speculative.py`.
4. Convert/quantize checkpoints: AWQ W4A16, GPTQ, EXL3; PLE int4 if the table
   exists and its geometry differs (20M base vocab / 8 heads / layer ids).
5. Measure MTP acceptance and quality (greedy determinism) on the real head.
6. Tune kernels for the new geometry: QSA block sizes for 24Q/4KV, HC rank 320
   on a dense backbone, dense GEMM tiles for 5120x14336, GDN 48V heads.
7. Numerical parity against transformers on CPU (logits per position), then
   quality gates (GSM8K, tool calling) on V620.
8. Decide whether to port any of Qwen4-Flash/Plus/Max; that depends on weights
   and license, neither of which is announced.

## 7. Opt-in prep code (`feat/qwen4-prep`)

Branch `feat/qwen4-prep`, worktree `../vllm-qwen4-prep`. Nothing here is
imported by the serving path, and Flash-Next behaviour is unchanged.

| Commit | What it does |
|---|---|
| `f53fd9e74d` | `vllm/models/qwen4_prep/config_probe.py`: stdlib-only classifier for a Qwen4-family `config.json`. Reports `supported`, concrete `blockers` (dense MLP, unknown model type, unknown layer types, indexer geometry, unsupported quantization), `warnings` (bf16 on gfx1030, QSA disabled, missing HC/PLE fields) and the derived geometry, including KV bytes/token. CLI: `python -m vllm.models.qwen4_prep.config_probe <config.json or dir>`, exit 0 when supported, 1 when blocked. |
| `962054a6f5` | Opt-in dense MLP for Qwen4 checkpoints without routed experts (section 7.1). |
| `22196687e9` | Backport of PR 64820, int64 HC/PLE row offsets. |
| `d92580ced1` | Backport of PR 60490, QSA indexer workspace reservation. |
| `b1938a5414` | Backport of PR 62318, quantized q/k/v with a BF16 indexer. |
| `4c217d501b` | Backport of PR 59958, PLE disk offload. |
| `c3971d2c89` | TP-group fixture for the dense MLP meta tests (34 passed on the GPU host). |

### 7.1 Dense MLP

The leaked Qwen4-27B is dense, but `Qwen4ExpDecoderLayer` always built
`Qwen4ExpSparseMoeBlock`. The prep branch adds a config-selected alternative:

- **Selection** (`common/mlp_selection.py`, stdlib-only): an explicit
  `dense_mlp` field wins when present; otherwise a config with no routed
  experts (`num_experts` missing or 0) is dense. `mlp_only_layers` is not
  consulted because no Qwen config defines it. The rule is provisional: no
  released checkpoint has either field, and the 27B evidence is kernel-only.
- **Module** (`common/dense_mlp.py`): `Qwen4DenseMLP` is a SwiGLU MLP built
  from `MergedColumnParallelLinear` + `SiluAndMul` + `RowParallelLinear`, the
  same shape as `Qwen2MLP`. It exposes `mlp.gate_up_proj` (two shards) and
  `mlp.down_proj`.
- **Weight names**: HF Qwen3.8-27B ships
  `model.language_model.layers.N.mlp.{gate_proj,up_proj,down_proj}.weight`
  (verified from its safetensors index). The existing
  `hf_to_vllm_mapper` strips `model.language_model.` and
  `packed_modules_mapping` fuses `gate_proj`/`up_proj` into the two
  `gate_up_proj` shards, so the dense variant loads through the existing
  `AutoWeightsLoader` path with no new mapping.
- **Sigmoid output gate**: not a separate module. Qwen4Exp QSA already packs
  a sigmoid output gate (`attn_output_gate = True` in `amd/qsa.py`), and GDN
  consumes `output_gate_type`; the leaked 27B kernel's sigmoid gate is the
  same attention-output gate. Nothing new was added.
- **Guards**: dense + MoE sequence parallelism or HC sequence parallelism
  raises at construction; the monokernel hook is disabled for dense layers
  (`is_moe_layer=False`), so the persistent kernels stay on the MoE path.
- **Tests** (`tools/rdna/qwen4/test_dense_mlp.py`): five stdlib tests for the
  selection rule always run. Three meta-device tests instantiate a tiny
  Flash-Next config and a tiny dense config, assert the Flash-Next module tree
  and parameter names equal a directly constructed `Qwen4ExpSparseMoeBlock`,
  assert the dense tree is `gate_up_proj`/`down_proj`/`act_fn`, and assert the
  HF weight-name mapping. They need torch and vLLM (one visible GPU, because
  vLLM queries device capability at import) and pass on the GPU host.

### 7.2 Validation status

- Mac (no torch): the 31 stdlib tests pass; the 3 meta-device tests skip.
- GPU host (par1-cs25, `HIP_VISIBLE_DEVICES=8`, meta device only):
  `pytest tools/rdna/qwen4/` gives **34 passed in 7.9 s**. The TP group is
  initialized with `init_distributed_environment` +
  `ensure_model_parallel_initialized(1, 1)` under `set_current_vllm_config`.
- `py_compile` passes on every file touched by the four backports.
- Serving GPU validation: **pending**. The backports and the dense MLP have
  not run on a V620 against a real checkpoint.
  The queued checks are: Flash-Next load + 1k/512 decode with each backport
  (int64 offsets, QSA workspace, quantized q/k/v, PLE disk offload), then a
  dense synthetic checkpoint through the dense MLP on meta/CPU and, once
  weights exist, the real 27B.

## 8. Verification performed for this report

- File inventory and diff: `git diff --stat v0.33.0 HEAD -- vllm/models/qwen4_exp`,
  `git ls-tree upstream/main`, `git log v0.33.0..upstream/main`.
- Conflict matrix: detached worktree at `0fc943cb0f`, `git cherry-pick -n` for
  nine upstream commits, conflicts recorded above.
- Public sources fetched 2026-11-03: HF config/model card, transformers docs and
  PR API, vLLM PR API, SGLang PR API and raw file check, llama.cpp PR/issue API,
  Qwen Code PR API, HF API for `Qwen/Qwen4*` (absent).
- No GPU run, no weight load, no throughput measurement in this report;
  memory numbers are arithmetic from the published geometry.
