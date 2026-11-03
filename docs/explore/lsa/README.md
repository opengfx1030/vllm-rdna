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

| Model | Call | Why |
| --- | --- | --- |
| Qwen3.8-27B (`27b-awq`, `27b-exl3`) | **Take** | The paper's D-LSA model. 16 softmax layers can carry it. GDN cannot. A map is published for this name. Refit on our checkpoint before trusting it. Context that fits is about 128k–256k in the recipe pool, not 10M. |
| Qwen3.6-27B | **Later** | Same Hub layer rhythm and the same KV math. No published map. Needs its own distillation. |
| Qwen3.8-Flash-Next | **Later** | 12 softmax layers can carry it. 36 GDN layers cannot. Those softmax layers already have a QSA indexer (budget 2048, compress 4). No published map. The checkpoint window is 262144, which is below where the paper's indexer speedup gets large. |
| DeepSeek-V4-Flash | **Leave** | Its own indexer (`index_topk` 512, compress 0/4/128). The recipe pool holds 107892 tokens. LSA does not shrink that cache. |
| DeepSeek-V4.1-Flash | **Leave** | Trained KV-source split: `kv_source_layer_ids` ends at 20, `candidate_source_layer_id` is 20, plus later index sources. A second indexer is redundant. Same call RidgeFill made for this model. |

N-LSA is native training of a new hybrid, about 0.5B parameters in the
paper. It is not a way to patch the checkpoints above.

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
| Release `d-lsa-weights-v1` | The release notes say "D-LSA index maps for Qwen3.8-27B and GLM-5.3-Flash" and do not name a separate license. The files sit on the Apache-2.0 repo. Unverified whether each `.pt` carries its own notice. | Do not vendor. Open the Qwen file on a CPU and check shapes before any load. GLM-5.3-Flash is not a model this fork serves. |
| Release `n-lsa-weights-v1` | Same gap in the release notes. 0.5B hybrid checkpoints. | Not our models. |
| SGLang fork `lsa-v0.1.0` | Not re-checked line by line. The image is CUDA 13. | Do not boot it on gfx1030. |
| RULER haystack files in `ruler-sources-v1` | Essays have no stated license. SQuAD and HotpotQA are CC BY-SA 4.0, as that repo's notice says. | Not redistributed. |

## What it takes to make it work

### Distillation

D-LSA on our checkpoints means a new map. The published Qwen3.8-27B map
was trained against that model's frozen attention (Section 4.1). Our
27B recipes serve AWQ and EXL3. Whether those K and Q match the map is
unverified. Flash-Next, Qwen3.6, and both DeepSeek checkpoints have no
published map.

The paper's run: 352M tokens of LongDocs, documents from 16k to 1M,
with 512k and 1M up-weighted. Sequences of 1M tokens are about a third
of the mix (Appendix C). Each step uses 128 queries from one document.
The target keeps the 2048 heaviest attention keys, renormalised. `τ`
is calibrated per layer before step 0. Projections start at the leading
principal directions. They found `d_idx = 64` enough, and `k = 4096`
held accuracy past 1M better than smaller budgets (Table 2). GLM uses
`k = 2048` below 1M and 4096 after. Tree search lost accuracy past 1M
on GLM. A training-free PCA map is the ablation they call D-LSA
(Training-Free). It stayed close on Qwen and trailed on GLM.

Why this is a rented-GPU job, not a V620 cluster:

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
- N-LSA's pretrain is 100B tokens. That is a different project.

Sweep, if a map is fit: `k` in {2048, 4096}, flat vs tree, trained
projections vs PCA, `d_idx` fixed at 64 unless a wider run is cheap,
`n_pool` 4, `n_cluster` 512, mean bucket 8, max bucket 16, `p` 1024,
sink 4, window 512. Those are the paper's settings, not tuned values
for gfx1030.

### Eval

`TESTPLAN.md`. RULER, the RULER needles, LongBench v2, and streaming
TTFT plus ITL at several lengths. The multi-turn prefix-cache trace is
required, including the stub row that must match flag-off. Paper
numbers are not results for this card.

## How to pick this up on a V620

1. Read this file, `SILICON.md`, `ENGINE.md`, and `TESTPLAN.md`.
2. Do not boot the CUDA SGLang image, and do not bump torch or Triton.
3. Run the CPU checks in `TESTPLAN.md`. They do not need a device.
4. Serve with the example env on top of `RECIPE=27b-awq` or
   `RECIPE=flashnext`. Leave `VLLM_LSA` at 0 until you want the guard.
   `VLLM_LSA=1` must match flag-off logits. Clients set nothing.
5. The first GPU work is a HIP selector (`sketches/selector_fdot2.cu.txt`)
   and a sparse gather into `RDNA_ATTN`, still behind the flag, still
   fail-closed to exact attention if the gather is absent.
6. Distillation of our checkpoints happens on rented CUDA GPUs. Bring
   back a map whose shapes match the softmax layers of that checkpoint.
   Then fill `TESTPLAN.md`, including the multi-turn trace. Expect turn
   2 to lose prefix hits if APC is off or the hash is salted.

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
- Refit D-LSA on the exact AWQ or EXL3 checkpoint (rented GPUs for the
  352M-token mix). PCA-only probe is the cheap local alternative and
  is still unwritten.
- Open `d-lsa-weights-v1` for Qwen3.8-27B on CPU and check layer count
  and `d_idx` before any GPU load. That open itself does not need a GPU.
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
