# Latent Sparse Attention (LSA) on this fork

**Status: untested explore draft. Not a serving change.** `VLLM_LSA`
defaults off. `VLLM_LSA=1` still runs exact attention. No recipe, ROCm
pin, torch pin, Triton pin, or kernel default changes. W4A16, W4A8,
EXL3, and `rdna_allreduce` are untouched.

This is a handoff for GPU owners, in the same shape as RidgeFill
([#155](https://github.com/opengfx1030/vllm-rdna/pull/155)). Silicon,
the engine hook, and the empty V620 gates are beside this file.

`VLLM_LSA` is a server-process variable. Clients and gateways never
export it. The example env sets `0`.

## Calls

Engine verdict for this fork: **Later / Take**, behind `VLLM_LSA=0`.
The algorithm is worth a guarded experiment. The SGLang image and the
CuTe kernels are not the serve path. Measure at 128k and 262144 before
any speed claim. The paper's own one-layer gain versus an already-sparse
indexer is 2.1× at 512k on an H200 (Section 4.3) and gets large at 1M–10M,
which these cards do not hold.

| Order | Model | Call | Weights |
| --- | --- | --- | --- |
| 1 | GLM-5.3-Flash | **Take** | Load the released map. No training step to start. LSA replaces that model's own sparse indexer (paper Section 4.1). This tree has `Glm5Next` and a kpool indexer (`vllm/models/glm5next/amd/sparse_indexer.py`, indexer head 128). No file in `tools/rdna/recipes/` names it. |
| 2 | Qwen3.8-27B (`27b-awq`, `27b-exl3`) | **Take** | Load the released map. 16 softmax layers. 48 GDN layers stay exact. Recipes already exist. |
| — | Qwen3.8-Flash-Next | **Later** | No map. A distill covers its QSA / softmax layers only (12 layers). GDN is untouched. |
| — | Qwen3.6-27B | **Later** | Same 16/64 rhythm as 3.8-27B. No map. Same distill as Flash-Next, after the two released maps have been measured. |
| — | DeepSeek-V4-Flash | **Leave** | Own indexer (`index_topk` 512, compress 0/4/128). Recipe pool holds 107892 tokens. |
| — | DeepSeek-V4.1-Flash | **Leave** | `kv_source_layer_ids` ends at 20, `candidate_source_layer_id` is 20. A second indexer is redundant. |
| — | N-LSA | **Leave** | Native training from scratch. Paper Section 4.4: about 0.5B parameters, 128k. The repo directory is `configs/training/native/qwen35_0_8b_gpt2/`. That folder name is not the paper's parameter count. |

Open risk, unverified: the released maps were distilled on the bf16
teachers (paper Appendix F, centroids and KV in bf16). Nobody has checked
whether those projections still pick the right keys when Q and K come
from W4A16, EXL3, or HI2. HI2 shows up in `tools/rdna/rdna_hint_flex/`
as a host-side quant path. Its Q/K layout versus the map was not
re-derived here. If a loaded map misses on a quantized checkpoint, a
refit is the fallback. It is not the first step for GLM or the 27B.

## What was checked, and against which page

Authors, paper Section heading and the blog citation: Ales Kubicek,
Andrea Caciolai, Liangzu Peng, Sukrit Sumant, Benjamin Hoover, Fraser
Greenlee, Ruairidh M. Battleday, Radim Urban, Torsten Hoefler, James
C. R. Whittington. Prefrontal AI and ETH Zurich. arXiv
[2610.42147](https://arxiv.org/abs/2610.42147). The arXiv HTML and the
blog citation block do not print a DOI. Unverified whether one was
registered later.

Announcement: James Whittington,
<https://x.com/jcrwhittington/status/2117317159777177602>
(2026-11-02). Blog: <https://prefrontal.ai/news/lsa>. Code:
<https://github.com/Prefrontal-AI/latent-sparse-attention>, Apache-2.0,
commit `906be195c1052aaa74f95852ad1917ce1926ba64` (2026-11-02, "Initial
release of Latent Sparse Attention"). No file from that tree is copied.

The 10M-token runs, the 72.4× prefill figure, and the token budget are
cited from the paper, not from the third-party summary that prompted
this note.

| Claim in that summary | Paper |
| --- | --- |
| D-LSA trains only indexer projections, KL, frozen model, about 350M tokens | Section 3.3 and 4.1. Only `W_Q`, `W_K` and a temperature `τ` per layer. Base model frozen. Forward KL, Equation 2. **352M** tokens, 48k steps, 128 queries per step, learning rate `3×10^-4`. They do not follow the distillation with full-model sparse training. "About 350M" is that 352M, rounded. |
| N-LSA is native training | Section 3.4 and 4.4. Next-token loss only. No index projections, no teacher. Hybrid of about 0.5B, 100B tokens at 4k then 12B at 128k. |
| Keys are pooled and Ward-clustered into hierarchical buckets | Section 3.1. Pool by summing `n_pool` keys (4 in the main D-LSA settings). Ward inside a block of `n_cluster` keys. Hierarchy is the Tree variant. Flat is one level. |
| Selection is flat or beam search | Section 3.2. Flat scores every centroid and keeps top-`p`. Tree descends with beam search (Algorithm 2). |
| The attention kernel is unchanged | Section 3.3 and the repo README. The model's attention reads `J_i`. Appendix F.2 adds an attention kernel for context parallelism. "Unchanged" is true for the non-CP path they reuse (FA3 and others). |
| Official stack is SGLang plus CuTe/Triton on H200/B300 | Section 4.1 and Appendix F. Repo README: SGLang fork tag `lsa-v0.1.0`, CuTe DSL for H200 and B300, Triton fallback. |
| Distilled maps and training recipes are open | Repo README and release `d-lsa-weights-v1` (2026-11-02): maps for Qwen3.8-27B and GLM-5.3-Flash. Distillation commands are in the README. Native checkpoints are release `n-lsa-weights-v1`. The README says those native tensors were trained with a trainer that is not in the repository. |

## Layers, from the Hub configs fetched 2026-11-03

Softmax layers are the only LSA carriers. `linear_attention` is GDN.

`Qwen/Qwen3.8-Flash-Next` `text_config.layer_types`: 48 layers,
`full_attention` at indices 3, 7, …, 47 (12 layers), `linear_attention`
elsewhere. `indexer_n_heads` 4, `indexer_budget` 2048,
`indexer_compress_ratio` 4, `indexer_head_dim` 128, `head_dim` 256,
`num_key_value_heads` 2, `partial_rotary_factor` 0.25,
`max_position_embeddings` 262144.

This fork builds QSA only when the layer string is
`qwen_sparse_attention` or `indexed_attention`
(`vllm/models/qwen4_exp/amd/model.py`). The Hub file spells those 12
layers `full_attention`. A string this decoder does not know raises
`ValueError`. No rewrite of that string was found in this tree.
Whatever the running process actually stores, the 12 positions are the
softmax layers and the other 36 are GDN. Confirm the string on the
checkpoint you boot. The stub's forward hook runs only on the QSA
strings. `Qwen3Next` hooks `full_attention`, which is the 27B spelling.

`Qwen/Qwen3.8-27B` and `Qwen/Qwen3.6-27B`: 64 layers, `full_attention`
at 3, 7, …, 63 (16 layers), no indexer fields, `num_key_value_heads` 4,
`head_dim` 256, `partial_rotary_factor` 0.25, max position 262144.
The paper calls Qwen-3.8-27B a full-attention model and also says only
the global attention layers get faster (Section 4.3). The Hub file is
a hybrid. The 16 `full_attention` layers are the ones that match the
paper's "global" layers. RoPE on a quarter of the dims matches Section
3.3.

`deepseek-ai/DeepSeek-V4-Flash`: 43 layers, `index_n_heads` 64,
`index_head_dim` 128, `index_topk` 512, `sliding_window` 128,
`compress_ratios` a length-44 list of 0, 4, and 128 (21 fours, 20
one-twenty-eights, 3 zeros), max position 1048576.

`deepseek-ai/DeepSeek-V4.1-Flash` `text_config`: 40 layers,
`kv_source_layer_ids` `[2, 8, 14, 20]`, `index_source_layer_ids`
`[2, 8, 14, 20, 24, 28, 32, 36]`, `candidate_source_layer_id` 20,
`index_topk` 512, compress ratios 0, 1, and 2.

`tests/model_executor/test_lsa.py` locks the 12-of-48 and 16-of-64
rhythm. It does not download the Hub.

## License

Nothing in this PR is a redistributed upstream file.

| Artifact | License | Use here |
| --- | --- | --- |
| `latent-sparse-attention` @ `906be195` | Apache-2.0, copyright 2026 Prefrontal AI | Read. Not copied. |
| NSA kernels inside that repo (`src/lsa/training/native/kernels/nsa`) | MIT, from flash-linear-attention | Not used. |
| PaperTrail kit `data/papertrail` | CC BY 4.0, Sukrit Sumant | Not redistributed. |
| LongDocs sources | Per source, listed in that repo's `THIRD_PARTY_NOTICES.md` (public domain, CC BY, CC BY-SA, ODC-By, Open Parliament Licence, The Stack terms). In-house traces are 13.1% of the distillation tokens (Appendix C) and ship as release `traces-v1`. | Not downloaded. |
| Release `d-lsa-weights-v1` (2026-11-02) | Apache-2.0 of the parent repo. The release body is one line and ships no extra license file. Unverified whether a notice is pickled inside each `.pt`. | Do not vendor. Pin below. Load on a machine that already has the matching checkpoint. |
| Release `n-lsa-weights-v1` | Same: no separate notice in the release body. Paper: about 0.5B hybrid. | Leave. Not a serving checkpoint. |
| SGLang fork tag `lsa-v0.1.0`, commit `042d955ba31b962911f63b114c2fb245effa86f2` | Apache-2.0. `LICENSE` at that tag is the Apache 2.0 text. | Read the serve flags. Do not boot the CUDA image on gfx1030. |
| RULER haystack files in `ruler-sources-v1` | Essays have no stated license. SQuAD and HotpotQA are CC BY-SA 4.0, as that repo's notice says. | Not redistributed. |

## Released maps

Tag `d-lsa-weights-v1`. Checksums are the release's `SHA256SUMS`, and
they match the GitHub asset digests read on 2026-11-03. The trained
files are the ones to load. The `pca-step0` files are the paper's
training-free ablation (Section 4.1: close on Qwen, behind on GLM).

| File | sha256 | Role |
| --- | --- | --- |
| `glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt` | `1e2fef41e76b0740132beddc28465fab8fd1d92a59de9e7cc4870cacb41e2fb9` | GLM trained map. Load this first. |
| `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt` | `24cee934f969756db228a8df7f2bc7613979b63ac3d9aa42e4339bb9b` | Qwen3.8-27B trained map. Load second. |
| `glm53_flash__long-16k-1M-heavy-key-and-query-pca-step0.pt` | `29a88a4108e40b8914b6889b98e7cebcf41cda6450321a52c748eb5751473298` | GLM PCA ablation. |
| `qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.pt` | `baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65` | Qwen PCA ablation. |
| `glm53_flash__long-16k-1M-heavy-x-dense-on-native.pt` | `ebe6a3fc9e4362448dd9764935e6cc86ecb8e417881edb7dc4d78eedc9fdfeb4` | In the release (23 MB). What it contains was not opened. Unverified. |

The filename `d2048` and `lr3e-4` match the paper's top-2048 teacher
mass and learning rate `3×10^-4` (Section 4.1). Layer count and
`d_idx` inside the tensors were not opened. Check those on CPU before
a GPU load.

First measurements are 128k and 262144 tokens, on the bf16 teacher if
you have one, then on the quantized checkpoint the recipe serves.
Record whether the selected keys change. That comparison is the open
quant risk. Do not quote 10M or 72.4× as a V620 result.

## What it takes to make it work

### Load, then distill

GLM-5.3-Flash and Qwen3.8-27B already have maps. The first path is to
load the two trained files above. No new training is required to start
those two. D-LSA only replaces key selection (Section 3.3). The
attention kernel, apart from the context-parallel kernel in Appendix
F.2, and the KV length stay the model's own.

Distillation is for a model with no map. On this board that is
Flash-Next's QSA layers, and later Qwen3.6-27B if those measurements
justify it. GDN layers are not in the loss. DeepSeek stays Leave, so
it is not a distill target.

The paper's recipe, if that distill is run: frozen model, train only
`W_Q`, `W_K`, and `τ` per layer, forward KL (Equation 2), 352M LongDocs
tokens from 16k to 1M, 48k steps, 128 queries per step, learning rate
`3×10^-4`. Init from principal directions. They skip the later
full-model sparse finetune. Their trainer is CUDA (`src/lsa/training`,
`causal-conv1d` wants `nvcc`). GPU-hours are not stated. Rent NVIDIA.
A V620 cluster is the wrong machine for the 512k–1M teacher forwards
in that mix.

Why a V620 is the wrong distill machine:

- The teacher forward is the frozen model's attention over `R_i`, which
  at the long end of the mix is a 1M-token sequence. Qwen3.8-27B in
  bf16 is on the order of 54 GB of weights (27×10^9 parameters × 2
  bytes), which does not fit one 32 GB GPU. Four V620s can shard a
  quantized 27B for serving. The recipe KV pool does not hold a 1M
  context even for inference (`SILICON.md`: 16 GiB of attention KV at
  1M on the 27B, and the `KV=8e9` cap is under 0.5M tokens before GDN
  state). Distillation needs that forward, not a short decode.
- The published training extra builds `causal-conv1d` and wants `nvcc`.
  The documented cluster path is a CUDA SGLang image on Slurm. The
  paper does not state the GPU count used for the 48k distillation
  steps. Serving measurements are H200 and B300.
- A 16k-only subset might fit on four V620s. That is not the mix they
  used. Appendix C says the long-context bias was worth +0.036 RULER
  at 1M in their ablation. Unverified here, and not a reason to pretend
  a V620 can reproduce the map.
- N-LSA is Leave. Its pretrain is 100B tokens at 4k plus 12B at 128k
  (Section 4.4), for a model we do not serve.

Sweep, only if a Flash-Next (or other unmapped) distill is actually
run: `k` in {2048, 4096}, flat vs tree, trained projections vs PCA,
`d_idx` 64, `n_pool` 4, `n_cluster` 512, mean bucket 8, max bucket 16,
`p` 1024, sink 4, window 512. Those are the paper's settings. GLM's
released run uses the smaller `k` below 1M (Table 2). At 128k–262k,
start from that smaller budget and write down what you sweep.

### Eval

`TESTPLAN.md`. RULER, the RULER needles, LongBench v2, and streaming
TTFT plus ITL at several lengths. The multi-turn prefix-cache trace is
required, including the stub row that must match flag-off. Paper
numbers are not results for this card.

## How to pick this up on a V620

1. Read this file, `SILICON.md`, `ENGINE.md`, and `TESTPLAN.md`.
2. Do not boot the CUDA SGLang image, and do not bump torch or Triton.
3. Run the CPU checks in `TESTPLAN.md`. They do not need a device.
4. Serve with the example env. `VLLM_LSA=1` must match flag-off logits.
   Clients set nothing. The stub does not load a map.
5. The selector, when written, lives in `vllm/v1/lsa/` and is called
   only for softmax layers. CuTe is Leave. The Triton indexer is the
   port candidate, with the gfx1030 breaks in `SILICON.md`. A HIP
   `v_dot2_f32_f16` score is the fallback sketch.
6. First real run: load the GLM map, then the Qwen3.8-27B map. Measure
   128k and 262144. Compare bf16 Q/K with W4A16, EXL3, and HI2 Q/K
   before trusting a quantized boot. Prefill graphs stay off once
   selection is real. Decode graphs can stay. APC is off or salted.
7. Distil only if you move on to Flash-Next's QSA layers, or another
   model that has no file in `d-lsa-weights-v1`. That job is rented
   NVIDIA. Then fill `TESTPLAN.md`. Turn 2 loses prefix hits if APC is
   off or the hash is salted.

## Still needs a GPU

- Compile the HIP selector and time `v_dot2_f32_f16` centroid scoring.
  The 8 µs floor in `SILICON.md` is arithmetic, not a profile.
- Try the upstream Triton files on Triton 3.7.1 ROCm only as a negative
  test. Do not upgrade Triton. `num_warps` 8 and 16, bf16 dots, and
  `libdevice` are the expected breaks.
- Sparse gather from an index list into paged fp16 KV, including the
  int4 KV layout if that dtype is on.
- HIP graph capture: eager prefill, one decode graph per length bucket.
  Confirm a host readback is not inside the capture.
- Load the two trained maps and compare bf16 Q/K against W4A16, EXL3,
  and HI2 Q/K at 128k and 262144. The CPU open of the `.pt` (layer
  count, `d_idx`) does not need a GPU. The quality and speed cells do.
- A Flash-Next QSA-only distill, rented NVIDIA, 352M LongDocs tokens,
  only after those two measurements say the method is worth it.
- RULER, LongBench v2, needles, streaming TTFT/ITL.
- Multi-turn prefix-cache trace, boots A and B now, C and D after a
  real selector.
- MTP behaviour once a selector exists. The stub must not change it.

## Files

| Path | Role |
| --- | --- |
| `README.md` | This note. |
| `SILICON.md` | gfx1030 cost, KV limits, kernels. |
| `ENGINE.md` | Flag, hooks, graphs, APC, MTP. |
| `TESTPLAN.md` | Empty GPU cells, CPU cells filled. |
| `reference_lsa.py` | Pool, Ward, flat, tree, toy KL. |
| `example_distill.py` | Prints the toy KL. |
| `example_qwen38_27b.env` | `VLLM_LSA=0`. |
| `sketches/selector_fdot2.cu.txt` | Uncompiled HIP outline. |
| `vllm/model_executor/layers/lsa/` | The guard. |
| `tests/model_executor/test_lsa.py` | CPU tests. |
