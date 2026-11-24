# Inkling-2-Mini DSpark draft on gfx1030 (V620)

Native-HIP port of the Inkling-2-Mini draft (`mtp.safetensors`, architecture
`Inkling2DSparkModel`) for the V2 model runner of `opengfx1030/vllm-rdna`.  Everything
that runs at serving time is a HIP kernel in `csrc/rocm/inkling2_dspark_rdna2.cu` (plus the
repo's existing dense RDNA2 GEMM designs); PyTorch / CPU code exists only as the offline
fp64 oracle in tests and tools.  Numbers below are copied from runs on the box
(`par1-cs25`, b4 HIP 2, gfx1030) unless marked `[INFERENCE]`.

## Run DSpark

One command on one card (the checkpoint directory has to hold `mtp.safetensors`, the draft):

```bash
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-dspark MODEL=/path/to/inkling2-mini \
  VENV=/path/to/venv HIP_VISIBLE_DEVICES=3
```

The recipe is `inkling2-mini-tp1` plus `SEQS=2`, `MAXLEN=125000` and
`SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,2]]}'`:
the model requires `num_speculative_tokens` 7 (the draft block is 8 rows) and the schedule is how many of
them the engine verifies per step (2 for one or two requests, no draft above). Any key can be overridden
after `RECIPE=`. Smoke test of the running server (text, image, prefix cache, stream, concurrent requests):
`python tools/rdna/inkling2/serve_smoke.py --url http://127.0.0.1:18221`.

| recipe | cards | what it is | one request decodes (mean of six kinds of text) |
| --- | --- | --- | --- |
| `inkling2-mini-tp1-dspark` | 1 | `inkling2-mini-tp1` with DSpark, K = 2 (`[[1,2,2],[3,16,0]]`), `SEQS=16`, `MAXLEN=115000`; `ADAPTIVE=1` (`MAXLEN=95000`) + optional `DRAFT_VOCAB` | 1.31x K=2; 1.51x `ADAPTIVE=1` + 32k vocab |
| `inkling2-mini-tp2` | 2 | DSpark ON by default, K = 2 (`[[1,2,2],[3,8,0]]`), `SEQS=8`, `MAXBAT=1024`, `MAXLEN=720000`; `ADAPTIVE=1` (`MAXLEN=660000`), `DSPARK=0` for 1M ctx | 1.39x K=2 (113 t/s); 1.65x `ADAPTIVE=1` + 32k vocab (135 t/s) |
| `inkling2-mini-tp4` | 4 | DSpark ON by default, K = 2 (`[[1,2,2],[3,8,0]]`), `SEQS=8`, `MAXLEN=1048576`; `ADAPTIVE=1` (1M ctx still fits), `DSPARK=0` for plain | 1.36x K=2 (143 t/s); 1.50x+ `ADAPTIVE=1` + 32k vocab (158+ t/s) |

The adaptive recipe takes a draft vocabulary: an ids file made once, on the CPU (27 s on the box, no GPU). Any code and
prose that is not what you will serve will do, here the venv's python packages and this repository's docs:

```bash
python tools/rdna/inkling2/draft_vocab.py --ckpt /path/to/inkling2-mini --size 32768 \
  --out ~/.cache/rdna/inkling2-mini-draft-vocab-32768.npy \
  --code '/path/to/venv/lib/python3.12/site-packages/**/*.py' --prose 'docs/**/*.md'
```

and is started like the first one with `RECIPE=inkling2-mini-tp1-dspark-adaptive` (`DRAFT_VOCAB=/other/ids.npy`, as an
argument or in the environment, names another file). Its `SPEC_CONFIG` adds
`"speculator_options":{"adaptive_k":{},"draft_vocab":"<ids.npy>"}` and the schedule `[[1,2,7]]`: the scheduler picks the
number of verified drafts (0 to 7) at every step from the acceptance it has seen, and the draft ranks only the listed ids.
Neither changes the output (the target verifies every drafted token whatever the count; a token outside the list is never
drafted, which costs acceptance only). TP=1 and greedy decoding only: the draft vocabulary is refused at TP > 1.

### Measured on the final head

One V620 alone (`par1-cs25` HIP 3), the `0fd3e424dc` build, one request at a time, temperature 0, 160 tokens; ms per token
from the first streamed chunk to the last, mean over the prompts of a kind of text. "No draft" is the recipe with
`SPEC_CONFIG=` (the same engine settings), and all three configurations pass `serve_smoke.py` (text, image, prefix cache,
stream, two concurrent requests). Boot with warm caches: 42 s no draft, 50 s K = 2, 71 s adaptive with the vocabulary.

Tuned prompts (the six kinds of text of the speculation profile, 3 prompts each: the ones the estimator defaults were
tuned on):

| text | no draft | K = 2 | adaptive + vocabulary | speed-up K = 2 | speed-up adaptive |
| --- | --- | --- | --- | --- | --- |
| chat | 17.53 | 15.36 | 14.58 | 1.14x | 1.20x |
| code_gen | 17.55 | 14.27 | 12.97 | 1.23x | 1.35x |
| code_edit | 18.10 | 11.99 | 10.00 | 1.51x | 1.81x |
| prose | 17.59 | 16.08 | 15.11 | 1.09x | 1.16x |
| structured | 17.61 | 12.00 | 10.25 | 1.47x | 1.72x |
| rag | 18.04 | 11.35 | 7.42 | 1.59x | 2.43x |
| mean of the six | 17.74 | 13.51 | 11.72 | 1.31x | 1.51x |

Held-out prompts (new prompts of the same six kinds, 5 - 6 each: none was used to tune the estimator or to rank the
vocabulary, and the quoted excerpts come from code and recipe files, not from the docs the vocabulary was ranked on):

| text | no draft | K = 2 | adaptive + vocabulary | speed-up K = 2 | speed-up adaptive |
| --- | --- | --- | --- | --- | --- |
| chat | 17.61 | 16.59 | 15.98 | 1.06x | 1.10x |
| code_gen | 17.64 | 15.12 | 14.62 | 1.17x | 1.21x |
| code_edit | 18.16 | 11.59 | 9.28 | 1.57x | 1.96x |
| prose | 17.65 | 15.85 | 15.38 | 1.11x | 1.15x |
| structured | 17.66 | 13.16 | 11.87 | 1.34x | 1.49x |
| rag | 18.00 | 11.63 | 8.10 | 1.55x | 2.22x |
| mean of the six | 17.79 | 13.99 | 12.54 | 1.27x | 1.42x |

The held-out means are 3 - 6 % slower than the tuned ones and every kind of text is still faster than no draft with both
recipes; the adaptive recipe is ahead of K = 2 on every kind (the least on chat and prose, the most on rag). The in-process
bench of the speculation profile gave 14.01 and 12.09 for the tuned means at K = 2 and adaptive + vocabulary.

What the draft costs in context (the engine's own numbers at `SEQS=2`, 125,000 requested): the KV cache holds 185,568 tokens
without the draft, 125,597 with K = 2 (weights +0.89 GiB, graphs 0.87 GiB against 0.29) and 106,267 with adaptive K and the
vocabulary (weights +1.05 GiB, graphs 1.38 GiB: every width is captured, and the gathered head rows are 0.16 GiB). The
recipes' `MAXLEN` (125000 and 105000) is the largest round number that boots, 0.5 % and 1 % under the cache: lower it if the
engine refuses to start on another card (it reports the longest length that fits).

### Lossless: the output stays inside the no-draft run-to-run noise

The target verifies every drafted token, so the output is the target's own greedy path; what can differ is the fp16 rounding of a
row computed in a verify step against the same row computed alone, and the no-draft engine does not reproduce itself bit for bit
either (two captures of the same prompts differ, below). The floor is therefore the no-draft engine against itself, and the
question is whether the draft engines stay inside it. Two checks on the final head.

**Greedy outputs of the served engines** (the prompts of the tables above, 160 tokens, the no-draft engine captured twice):

| comparison | identical, tuned (of 18) | median first divergence | identical, held-out (of 33) | median first divergence |
| --- | --- | --- | --- | --- |
| K = 2 against no draft | 5 | token 75 | 6 | token 67 |
| adaptive + vocabulary against no draft | 3 | token 60 | 7 | token 69 |
| no draft against no draft (floor) | 3 | token 100 | 9 | token 67 |

Even two no-draft captures part on most prompts, at a near-tie of the model's own distribution. Before the first divergence the
log-probability of the chosen token differs by (mean, p99, max) 0.0082, 0.097, 0.262 (K = 2) and 0.0082, 0.096, 0.221 (adaptive)
against 0.0064, 0.072, 0.226 for the floor on the tuned prompts, and 0.0089, 0.091, 0.474 / 0.0091, 0.101, 0.465 against
0.0073, 0.076, 0.407 on the held-out ones. At the first divergence the two tokens differ by 0.42 nats or less in the
reference run's log-probability in every comparison but one (K = 2: 0.38 tuned, 0.41 held-out; adaptive 0.20, 0.42; floor 0.19,
0.39). The exception is one position: against the second no-draft capture the K = 2 run parts on chat prompt 2 at token 119 with
a gap of 1.45 nats (the two no-draft captures part on that prompt at token 33, and the K = 2 run followed the second one);
against the first capture it has no such position. A single free-running position selected for being a near-tie cannot say
whether this is more than the tail of the noise, which is what the second check measures.

**Teacher-forced NLL** (`tools/rdna/inkling2/dspark_nll.py`): the tokens of a reference capture are forced through the
in-process TP=1 engine (the recipe's settings, the draft on) and the raw log-probability the engine gives the forced token is
recorded at every position, so every run sees the same context at every position and the differences are the engines'
numerical differences alone, with no free-running selection. The first no-draft capture of the 18 tuned prompts, 2,880 tokens:

| engine | NLL | mean dlogp | mean abs dlogp | p99 abs | max abs |
| --- | --- | --- | --- | --- | --- |
| no draft, run 1 (the reference) | 0.2727 | | | | |
| no draft, run 2 (the floor) | 0.2717 | +0.0010 | 0.0123 | 0.128 | 0.488 |
| K = 2 (1316 drafts, 1.19 accepted each) | 0.2720 | +0.0007 | 0.0114 | 0.125 | 0.316 |
| adaptive + vocabulary (1102 drafts, 1.62 accepted each) | 0.2719 | +0.0007 | 0.0111 | 0.120 | 0.271 |

The pass rule is mean abs dlogp <= 2 x floor + 0.01 = 0.0346, and both draft engines pass it with a mean, a p99 and a worst
position at or below the floor's. Per kind of text the mean NLL of the three engines differs by 0.002 or less (chat 0.3863 /
0.3865 / 0.3866 for run 1 / K = 2 / adaptive, code_gen 0.3226 / 0.3223 / 0.3217, code_edit 0.1281 / 0.1281 / 0.1267, prose
0.3902 / 0.3884 / 0.3887, structured 0.2085 / 0.2073 / 0.2077, rag 0.2003 / 0.1995 / 0.2003).

The same on 8,160 tokens: the 18 tuned prompts forced with the second no-draft capture (the one whose path the 1.45-nat position
above lies on) and the 33 held-out prompts forced with their no-draft capture:

| engine | NLL | mean dlogp | mean abs dlogp | p99 abs | max abs |
| --- | --- | --- | --- | --- | --- |
| no draft, run 1 (the reference) | 0.2642 | | | | |
| no draft, run 2 (the floor) | 0.2644 | -0.0002 | 0.0115 | 0.127 | 0.900 |
| K = 2 (3813 drafts, 1.14 accepted each) | 0.2640 | +0.0002 | 0.0113 | 0.128 | 0.410 |
| adaptive + vocabulary (3306 drafts, 1.47 accepted each) | 0.2639 | +0.0003 | 0.0112 | 0.120 | 0.371 |

The rule is 0.0330 here and both pass. Two no-draft runs differ by 0.90 nats at their worst position of the 8,160, so the
tail of the noise reaches the order of the 1.45 nats seen at one free-running position, and the draft engines' worst
positions (0.41, 0.37) are below it. Per kind of text the mean NLL of the draft engines is within 0.003 of the first no-draft
run (chat 0.3722 / 0.3718 / 0.3704 for run 1 / K = 2 / adaptive, held-out rag 0.0782 / 0.0779 / 0.0781).

`python tools/rdna/inkling2/dspark_nll.py run --refs <refs.json> --out <o.json> [--spec-config '<json>']` makes a run (a JSON
list of `{name, prompt, ids}` references; no draft twice with `--reps 2` for the floor) and `compare` prints the table above.

### TP=2

`inkling2-mini-tp2-dspark` is the same line with the pair (`HIP_VISIBLE_DEVICES=2,3` on the box). On the final head it is up
after 57 s and passes `serve_smoke.py` (the no-draft recipe, `SPEC_CONFIG=`, boots in 60 s and passes too). The draft costs
0.77 GiB of weights (10.38 against 9.61 GiB per rank) and 0.57 GiB of graphs per rank, and the KV cache holds 790,241 tokens at
`MAXLEN=780000` (1.01x; 1,072,761 tokens without the draft). Decode of one request on the tuned prompts (3 of each kind, 160
tokens, ms per token; the adaptive column is `speculator_options {"adaptive_k":{}}` with the schedule `[[1,2,7]]` and
`MAXLEN=500000`, no recipe):

| text | no draft | K = 2 | adaptive, default cost table | speed-up K = 2 | speed-up adaptive |
| --- | --- | --- | --- | --- | --- |
| chat | 12.41 | 10.79 | 11.04 | 1.15x | 1.12x |
| code_gen | 12.41 | 9.68 | 9.69 | 1.28x | 1.28x |
| code_edit | 12.78 | 8.32 | 7.35 | 1.54x | 1.74x |
| prose | 12.39 | 10.63 | 11.67 | 1.17x | 1.06x |
| structured | 12.42 | 8.56 | 7.81 | 1.45x | 1.59x |
| rag | 12.75 | 7.85 | 5.62 | 1.62x | 2.27x |
| mean of the six | 12.53 | 9.31 | 8.86 | 1.35x | 1.41x |

Adaptive K runs at TP=2 (all widths captured, 1.07 GiB of graphs per rank, 758,884 tokens of KV at `MAXLEN=500000`, smoke test
passes) and is ahead of K = 2 on code_edit, structured and rag, level on code_gen and behind on chat and prose (+2 % and
+10 %, with 3 prompts each). The draft vocabulary is refused at TP > 1: the engine stops at load with `NotImplementedError: the
draft_vocab option is not supported under tensor parallelism`.

The policy's default step-cost table is the V620's at TP=1. The step time at TP=2 was measured from static-K runs (ms per token
times tokens per step, mean of rag and chat; K = 4 - 6 interpolated between the K = 3 and K = 7 runs):

| K | rows | TP=2 step ms | TP=1 table, ms |
| --- | --- | --- | --- |
| 0 | 1 | 12.58 | 18.13 |
| 1 | 2 | 19.27 | 26.36 |
| 2 | 3 | 20.93 | 30.03 |
| 3 | 4 | 22.14 | 32.10 |
| 4 | 5 | 23.53 (interpolated) | 34.17 |
| 5 | 6 | 24.93 (interpolated) | 36.24 |
| 6 | 7 | 26.32 (interpolated) | 38.31 |
| 7 | 8 | 27.72 | 40.36 |

Relative to K = 0 the two tables are close (step(K) / step(0) at K = 1, 2, 3, 7: 1.53, 1.66, 1.76, 2.20 at TP=2 against 1.45,
1.66, 1.77, 2.23 at TP=1), which is probably why the default table still gains on average here **[INFERENCE]**. The
`adaptive_k` fields take another table (`forward_ms` per K with the other fields at 0 uses the step times above as they are);
that override was derived from these runs but not run end to end, so whether it removes the loss on chat and prose is not
measured.

Greedy outputs at TP=2 (18 prompts): identical for 5 (K = 2) and 5 (adaptive; 3 against the second capture) against 6 for no
draft against itself, median first divergence token 35 - 55 in all pairs, mean abs log-probability difference before the
divergence 0.0063 - 0.0073 (K = 2) and 0.0074 - 0.0080 (adaptive) against 0.0062 for the floor, largest near-tie gap 0.38, 0.33
and 0.33. `dspark_nll.py` is TP=1 only, so there is no teacher-forced run at TP=2.

## Status

| Item | State |
| --- | --- |
| Draft model, loader, speculator (`amd/dspark.py`) | written; HIP-vs-fp64-oracle run on the REAL `mtp.safetensors` + real target head passes (numbers below) |
| 15 native ops (`inkling2_dspark_*`) plus one Python helper | written, registered (ops.h, torch_bindings.cpp, CMake whitelist, `_custom_ops.py`), reference-tested on b4 |
| Exact-probability gate of the rejection path | CPU oracle + negative controls pass; the production HIP rejection sampler passes on b4 (4 GPU cases) |
| Mutation checks of the kernels | 10 of 10 mutants killed (table in Results); `argmax_partial` 6 of 6 ("Gates on the merged tree") |
| Official `build_lq.sh` build of the new TU | built (exit 0 after 804 s, official `-O2` flags); all 15 ops register in the official `_rocm_C`; on that build (plain pytest on the in-tree `_rocm_C.abi3.so`, b4 HIP 2) `test_inkling2_dspark_rdna2.py` gives 130 passed (48.19 s) and `test_dspark_model.py` + `test_dspark_verifier.py` give 31 passed (37.93 s) with the same draft-vs-oracle errors as the dev `-O3` build |
| Target tap hook in `amd/model.py` | applied and run on the real target through the runner's EAGLE3 interface (part B): `supports_eagle3` was false because the class lacked the `SupportsEagleBase` flags, and the draft config listed the final-output id 48 as a layer to tap (the hook raised); both fixed, CPU tests added |
| Runner-owned Triton (`prepare_dflash_inputs`) | HIP op written and tested against a Python port of the kernel; the dispatch is applied in `dflash/speculator.py` and gated by `DFlashSpeculator.uses_hip_prepare_inputs` (True only for `Inkling2DSparkSpeculator`), so every other DFlash / DSpark draft keeps the Triton kernel |
| V2 runner end-to-end (model load + `propose` through `LLM(...)`) | **run** on the real checkpoint, `FULL_AND_PIECEWISE`, `num_speculative_tokens=7`, TP=1: acceptance, speed and greedy equality in "Part B: the draft on the V2 runner" of `inkling2-port-spec.md` |
| FULL HIP-graph capture inside the runner | done: `DFlashCudaGraphManager` captures the draft step (graph memory 1.95 GiB with the target graphs) |
| TP > 1 | **run through the engine at TP = 2** (`FULL_AND_PIECEWISE`, the collectives inside the graphs): the draft is replicated on every rank, the embedding of the anchors is the target's vocabulary-shard lookup plus one all-reduce, the head's argmax is a per-rank partial plus an all-gather (greedy drafting only); K = 3 gives 1.30x at batch 1 and 1.09x at batch 2 over no draft and the greedy output stays inside the no-draft floor; the draft's replicated K/V does not fit the 1,048,576-token recipe (20.18 GiB needed, 16.14 GiB available: 836,672 tokens fit); TP = 4 was not run through the engine (see "Merged tree") |

## What the draft is

`mtp_config` of the real checkpoint: 3 draft layers (hidden 2304, 24 q / 8 kv heads, head
dim 128, MLP 9216, RoPE theta 1e6, eps 1e-6), layers 0 and 2 sliding-window (1024), layer 1
full attention, block of 8 rows (row 0 = the last committed token, rows 1..7 = mask),
Markov rank 256, taps `target_layer_ids [5, 23, 41, 48]`.

* **Taps.** Three taps are the fp32 attention-residual partial block at the *entry* of
  target layer visits 5, 23 and 41 (global visit index over the 48 visits of the looped
  stack); the fourth is the final normalized output.  The speculator stores them as bf16
  bits (`inkling2_dspark_store_taps`, round-to-nearest-even) in `[max_num_tokens, 4*2304]`.
* **Context path.** For every draft layer: DFlare mix
  `rmsnorm(sum_t softmax(fuse_weight[l])[t] * tap_t) * w[l]` (`mix_rms`), K/V projection
  (dense fp16 GEMM), per-head RMSNorm + RoPE + paged KV write (`qkv_prep`).  The draft owns
  three KV caches (`model.layers.{0,1,2}.self_attn.attn`), page layout `[nb, H, bs, 256]`.
* **Block path.** Anchor/mask embedding with the target's embedding + `embed_norm`
  (`embed_block`), then per layer RMSNorm -> QKV GEMM -> `qkv_prep` -> paged split-KV
  attention (`attention`, bidirectional inside the block, window 1024 on layers 0 and 2) ->
  `o_proj` (fp32 split-K partials) -> `add_rmsnorm` -> gate/up GEMM -> `silu_mul` -> down
  (split-K partials) -> `add_rmsnorm`.
* **Markov head.** `f_rnn` from the last committed row (alpha row 3 of `fuse_weight`),
  base logits from the **target's `lm_head`** (shared), the Markov `hidden_out` folded into
  the head (`fold_w = lm_head . hidden_out_t`, `fold_b`, recomputed by
  `refresh_derived_weights`), then 7 sequential steps `markov_step` -> `head_logits`
  (softcap 30) -> `argmax` (greedy) or the runner's sampler (probabilistic, which also
  fills the draft-logits cache the rejection sampler reads).

## Kernels (`inkling2_dspark_*`, namespace `inkling2_dspark`)

`store_taps`, `mix_rms`, `embed_block`, `add_rmsnorm`, `silu_mul`, `gather_rows`, `gemv`
(skinny fp16, up to 32 rows, optional gathered rows and split-K fp32 partials), `gemm`
(the existing dense RDNA2 designs 0..5, fp16 in/out, fp32 accumulate), `qkv_prep`,
`attention` (split-KV, wave32, `V_DOT2_F32_F16`, pages of 16 or 64), `markov_step`,
`head_logits`, `fold_bias`, `argmax`, and `prepare_inputs` (HIP port of the runner's Triton
`_prepare_dflash_inputs_kernel` + `_pad_dflash_inputs` for context-parallel size 1).  All
launch on the current HIP stream, none synchronise the host, all shapes are static per
captured graph.  The TU is built at the official flags (`-O2`); the dev JIT builds used for
the tests above use `-O3` `[the A/B is below]`.

## Numerics and deviations from the checkpoint

| Quantity | Reference (NVIDIA / transformers) | Here | Why |
| --- | --- | --- | --- |
| Draft weights | bf16 | fp16 (`bf16 -> fp16` is exact in the normal range) | gfx1030 has no bf16 dot; overflow raises at load, flushed-to-zero values are counted |
| Residual stream, reductions, softmax, accumulators | bf16 storage / fp32 math | fp32 | match the reference accumulation |
| GEMM / attention inputs and outputs | bf16 | fp16 | `V_DOT2_F32_F16` |
| Taps | bf16 | bf16 bits (from the target's fp32 blocks and the fp32 final output) | exactly the reference's input precision |
| RoPE angle | fp32 | float64 `pos * inv_freq`, then fp32 sin/cos | long contexts (up to 1,048,576) |
| Markov state | bf16/fp32 | fp32 | recurrence over 7 steps |
| Draft-logits cache (probabilistic drafting) | bf16 | fp16 (`draft_logits_spec -> (float16, 0.0)`) | the sampler samples from the very values it caches, which is what keeps rejection sampling lossless |

fp16 weight report of the real `mtp.safetensors` (`fp16_weight_report`):
`max_abs 38.5`, `overflow 0`, `flushed_to_zero 446`, `max_rel 0.0`.  The fp16 headroom of
the draft *activations* on real target taps is **not measured** (only synthetic taps were
available); the fp32 residual stream makes an overflow unlikely `[INFERENCE]`.

## Results (what was run)

Commands run on the box with the dev-built library at `-O3` and the official kernel flags.

### Draft vs the fp64 oracle on the REAL checkpoint

`dev_run.py tools/rdna/inkling2/dspark_check.py --hip --context 300 1100` (b4 HIP 2): real
`mtp.safetensors` (rounded to the served fp16), the real target embedding / `embed_norm` /
unembedding, synthetic taps (no target forward exists yet), two requests of 300 and 1100
context rows (the second is past the 1024-key window), 7 draft steps each.  The oracle is
teacher-forced with the HIP draft's own tokens, so every step is compared on identical
Markov inputs.

| Quantity | Measured |
| --- | --- |
| K cache, max abs error (largest oracle magnitude 29.58) | 9.44e-3 |
| V cache, max abs error | 1.28e-2 |
| draft hidden, max abs / relative L2 | 5.53e-2 / 1.43e-3 |
| `f_rnn`, max abs | 1.03e-2 |
| softcapped logits (`abs(z) <= 30`), max abs / mean abs | 3.42e-2 / 4.39e-3 |
| greedy tokens differing from the oracle's argmax | 0 of 14 (`token_gap` 0.0) |

The draft's greedy chains were `[62, 2689, 82, 25, 196997, 58, 6]` and
`[25, 196997, 50, 12, 1539, 63, 150220]`.  Taps are synthetic, so these numbers bound the
draft's own arithmetic, not its acceptance length (that needs the target).

### Tests

| Command (box `par1-cs25`) | Result |
| --- | --- |
| `pytest tests/models/inkling2_loop/test_dspark_oracle.py` (CPU) | 15 passed, 16.33 s |
| `pytest tests/models/inkling2_loop/test_dspark_model.py` (CPU, GPU tests skipped) | 14 passed, 8 skipped, 7.51 s |
| `pytest tests/models/inkling2_loop/test_dspark_verifier.py` (CPU, GPU leg skipped) | 5 passed, 4 skipped, 4.54 s |
| `pytest tests/models/inkling2_loop/test_dspark_target_taps.py` (CPU, tiny1 + tiny8) | 4 passed, 3.14 s |
| same file with `visits_seen += 2` (hand mutation of the tap counter) | 4 failed (restored, 1 occurrence of `+= 1` verified) |
| lead's `test_paged_model.py -k "single_step or without_metadata or greedy"` with the hook in place | 10 passed |
| `test_inkling2_dspark_rdna2.py` on b4, earlier build (before mixed-dtype `store_taps`, `prepare_inputs`) | 111 passed, 63.01 s |
| same file, first run with `prepare_inputs` | 122 passed, 8 failed (the new `prepare_inputs` tests: my test layout used a query block of `steps + 1` rows with `sample_from_anchor=True`, which the Triton kernel itself writes out of bounds; the test now uses the runner's layout and the op rejects the other one) |
| same file, clean build (baseline of the mutation run) | 130 passed, 53.65 s; after restoring: 130 passed, 47.91 s |

### Hand mutations of the kernels

Each mutant patches the kernel TU in the box tree copy, rebuilds the dev library, runs
`tests/kernels/rocm/test_inkling2_dspark_rdna2.py -x` and must fail a test that passes on
the clean build (`mutate.py`; the clean source is restored and re-verified at the end).

| Mutant | Killed by |
| --- | --- |
| bf16 round-to-nearest-even -> truncation (`store_taps`) | `test_store_taps_rounds_to_nearest_even` |
| RoPE angle `pos * inv_freq` formed in fp32 instead of float64 | `test_qkv_prep_block_matches_fp64[16]` |
| window start `p - window` -> `p - window + 1` (attention) | `test_attention_matches_fp64[seq_lens2-1024-1]` |
| wrong softmax row of `fuse_weight` (`l ^ 1`) | `test_mix_rms_three_layers_matches_fp64` |
| `gemv` ignores `x_index` | `test_gemv_gathers_rows_through_x_index` |
| split-K consumer drops the last partial | `test_add_rmsnorm_split_partials_sum_in_order` |
| argmax ties resolve to the higher index (4 sites) | `test_argmax_matches_reference[1-200058-200064]` |
| `prepare_inputs` writes slots of the null block | `test_prepare_inputs_matches_the_triton_kernel[3-8-idx_dtype0-False]` |
| softcap without the `cap` scale | `test_head_logits_matches_fp64[1]` |
| Markov gate roles swapped | `test_markov_chain_matches_fp64[prev_dtype0]` |

All ten were killed.  The mutation of the verifier gate is its own negative control (four
wrong rules, run on the CPU).

### Model tests, verifier gate, dispatch and timings (b4 HIP 2, one window)

`pytest tests/models/inkling2_loop/test_dspark_model.py
tests/models/inkling2_loop/test_dspark_verifier.py
tests/kernels/rocm/test_inkling2_dspark_rdna2.py`: **161 passed in 90.34 s**
(22 + 9 + 130).  HIP draft vs the fp64 oracle in the three model-level comparisons:

| Case | hidden rel. L2 | logit max abs | `f_rnn` max abs | K / V cache max abs (largest magnitude) | tokens differing |
| --- | --- | --- | --- | --- | --- |
| random weights of the real shapes, 3 requests (37, 1100, 2100 rows) | 5.56e-4 | 2.19e-2 | 7.80e-3 | 3.88e-3 / 3.17e-3 (7.55) | 0 of 21 |
| the same weights, 40 requests (two head chunks, dense GEMM designs) | 6.33e-4 | 2.43e-2 | 1.44e-2 | 3.95e-3 / 3.06e-3 (7.34) | 1 of 280 (logit gap 1.1e-3, a near tie) |
| real `mtp.safetensors`, 3 requests (300, 1500, 90 rows) | 1.25e-3 | 2.33e-2 | 1.53e-2 | 9.44e-3 / 1.38e-2 (29.58) | 0 of 21 |

Also passing in that window: the page-size test (16 vs 64: same keys, only the split-KV
summation order may differ), padded requests, `f_rnn` after rejections (int32 and int64
`num_rejected`), whole-draft-step HIP graph capture + replay bit-exact against eager, and
the four production-rejection-sampler cases of the exact-probability gate (draft logits
cached as fp32 and fp16, two seeds).  The tolerances in `test_dspark_model.py` are about
10x the observed errors (`f_rnn` 1.3x), i.e. they are loose; the kernel mutants above are
the measured evidence that the tests can fail.

`prepare_inputs`: with the dispatch patch applied to the box tree copy,
`tests/v1/spec_decode/test_dflash_prepare_inputs.py` (the runner's own test of the Triton
kernel; `cp_size == 1` cases go through the HIP op) gave 6 passed in 4.26 s.

Draft timings (`bench_run.py`, mean of 30 runs, HIP events, `-O3` dev build; `context` is
the context pass over *all* listed rows, i.e. prefill-like for the long cases, `block` is
the per-proposal block pass + Markov head at that KV length):

| Requests x context rows | context pass | block pass |
| --- | --- | --- |
| 1 x 8 (steady state of one request) | 0.136 ms | 4.85 ms |
| 8 x 8 | 0.258 ms | 7.42 ms |
| 1 x 300 | 0.542 ms | 4.99 ms |
| 1 x 2100 | 2.14 ms | 5.45 ms |
| 1 x 4096 | 4.36 ms | 5.77 ms |
| 8 x 300 | 2.51 ms | 8.01 ms |
| 4 x 2100 | 9.86 ms | 13.16 ms |

At one request the block pass is ~5 ms for any context up to 4096 rows (layer 1 reads
the whole context; the other two read at most 1024 keys), so a proposal costs about 5 ms
at the steady state `[INFERENCE: the per-kernel split is not profiled]`.  The `-O2` vs
`-O3` A/B of this TU was **not run**; the TU is built with the repo default (`-O2`) and
is not in the `-O3` list of `CMakeLists.txt`.

## Verifier gates

1. **fp64 oracle** (`tools/rdna/inkling2/dspark_check.py`, `tests/models/inkling2_loop/test_dspark_oracle.py`): a
   plain-PyTorch float64 implementation of the whole draft (mix, K/V, window attention,
   MLP, Markov head with teacher forcing) that is also run against the real checkpoint.
2. **Kernel reference tests** (`tests/kernels/rocm/test_inkling2_dspark_rdna2.py`): every
   op against fp32/fp64 PyTorch (plus a Python port of the Triton input-preparation kernel).
3. **Model tests** (`tests/models/inkling2_loop/test_dspark_model.py`): the HIP draft
   against the oracle on random weights (contexts shorter than, just past and far past the
   window; 40 requests = two Markov-head chunks; padded requests; 16 vs 64 page size; `f_rnn`
   after rejections; HIP graph replay bit-exact) and on the real checkpoint.
4. **Exact-probability enumeration gate of the rejection path**
   (`tests/models/inkling2_loop/test_dspark_verifier.py`).  On a 4-token vocabulary with
   Markov target `p` and draft `q` the output law of one verification round (K = 3) is
   enumerated with `fractions.Fraction` (every draft block, every accept/reject path, every
   resample) and shown equal to the closed form
   `P(e_0..e_{i-1}, n >= i) = prod_{j<i-1} min(q_j, p_j)(e_j) * p_{i-1}(e_{i-1})`.  The
   production rejection sampler (HIP ports of the V2 runner) is then run on 400,000 random
   blocks (draft logits cached as fp32 and as fp16, as served) and every level's prefix
   counts are chi-square tested against the exact probabilities (threshold
   `df + 6 sqrt(2 df)`).  The same statistic must reject four deliberately wrong rules
   (always accept, resample from `p` instead of the residual, accept with `sqrt(p/q)`, emit
   the draft token on rejection); this negative control passes on the CPU.
   It covers token-wise verification; block verification is not covered.
5. **Hand mutations** of the kernels (table in Results).

## Wiring outside the model files (all applied in the repo)

### 1. Registry (`vllm/model_executor/models/registry.py`, `_SPECULATIVE_DECODING_MODELS`)

```diff
     "Gemma4DSparkModel": ("gemma4_dspark", "Gemma4DSparkForCausalLM"),
+    "Inkling2DSparkModel": (
+        "vllm.models.inkling2_loop.amd.dspark",
+        "Inkling2DSparkModel",
+    ),
```

### 2. `prepare_dflash_inputs` HIP dispatch (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`)

Applied in part B and gated: the condition in the repo is `if hip_kernel and _ON_RDNA2 and
cp_size == 1:`, `prepare_dflash_inputs(..., hip_kernel=self.uses_hip_prepare_inputs)` and
`Inkling2DSparkSpeculator` is the only class that sets `uses_hip_prepare_inputs = True`.
The diff below is the first, ungated version.  The runner's own
`tests/v1/spec_decode/test_dflash_prepare_inputs.py` therefore runs the Triton kernel again.

```diff
@@ -7,12 +7,14 @@
 import torch
 import torch.nn as nn
 
+from vllm import _custom_ops as ops
 from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
 from vllm.config.compilation import CUDAGraphMode
 from vllm.forward_context import BatchDescriptor, set_forward_context
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
 from vllm.platforms import current_platform
+from vllm.platforms.rdna import on_rdna2_hip
 from vllm.triton_utils import tl, triton
 from vllm.v1.attention.backend import AttentionCGSupport
 from vllm.v1.attention.backends.utils import PAD_SLOT_ID
@@ -37,6 +39,9 @@
 from vllm.v1.worker.utils import AttentionGroup
 
 logger = init_logger(__name__)
+
+# Read once at import: Dynamo-traced hot paths must not re-evaluate the gate.
+_ON_RDNA2 = on_rdna2_hip()
 
 
 class DFlashSpeculator(DraftModelSpeculator):
@@ -992,6 +997,44 @@
     # per-request query length, not the total token count across the batch.
     max_target_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
     max_tokens_per_req = max_target_query_len + num_query_per_req
+    if _ON_RDNA2 and cp_size == 1:
+        # native HIP port of the kernel below (csrc/rocm/inkling2_dspark_rdna2.cu)
+        ops.inkling2_dspark_prepare_inputs(
+            input_buffers.input_ids,
+            input_buffers.positions,
+            input_buffers.is_padding,
+            input_buffers.query_start_loc,
+            input_buffers.seq_lens,
+            query_slot_mapping,
+            context_positions,
+            context_slot_mapping,
+            sample_indices,
+            sample_pos,
+            sample_idx_mapping,
+            temperature,
+            seeds,
+            input_batch.positions,
+            input_batch.query_start_loc,
+            input_batch.idx_mapping,
+            last_sampled,
+            next_prefill_tokens,
+            num_sampled,
+            num_rejected,
+            input_temperature,
+            input_seeds,
+            block_table,
+            block_size,
+            parallel_drafting_token_id,
+            num_query_per_req,
+            num_speculative_steps,
+            max_num_reqs,
+            max_num_tokens,
+            max_model_len,
+            num_reqs,
+            max_tokens_per_req,
+            sample_from_anchor,
+        )
+        return
     BLOCK_SIZE = 128
     num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
     _prepare_dflash_inputs_kernel[(num_reqs + _NUM_PAD_PROGRAMS, num_blocks)](
```

### 3. Target tap hook (`vllm/models/inkling2_loop/amd/model.py`) - APPLIED

`Inkling2Model.aux_hidden_state_layers` (empty by default); `forward` counts layer visits
across the prelude, every pass of the loop and the coda, appends the fp32 block state at
the entry of each selected visit and returns `(out, [*taps, out])` (the final normalized
output is the last tap) when the set is non-empty.  `Inkling2LoopForConditionalGeneration`
gets `supports_eagle3 = True`, `set_aux_hidden_state_layers`,
`get_eagle3_default_aux_hidden_state_layers() -> (5, 23, 41)` and returns
`(hidden, taps)` from `forward` when taps were requested.  The structural
`SupportsEagle3` check of the runner needs `has_own_lm_head` / `has_own_embed_tokens` as
well (the protocol is runtime-checkable over every member; added in part B), and
`Inkling2DSparkConfig.eagle_aux_hidden_state_layer_ids` is `target_layer_ids[:-1]` =
(5, 23, 41): the target appends the final output itself and the id 48 is not a layer.

### 4. Test tethering (`tools/pre_commit/test_tethering_allowlist.txt`) - APPLIED

Four fork-only entries (`test_dspark_model.py`, `test_dspark_oracle.py`,
`test_dspark_verifier.py`, `test_dspark_target_taps.py`).

### 5. Draft quantization (`vllm/v1/worker/gpu/spec_decode/dspark/config.py`) - APPLIED

`DSparkSpeculatorConfig.default_model` copied the target's `modelopt_fp4` to the draft
(DeepSeek DSpark ships its weights inside the quantized target checkpoint).
`mtp.safetensors` is bf16, so `get_draft_quant_config` tried to parse a ModelOpt config
for it and failed (`Missing 'quant_algo'`).  Like Mistral Large 4, an `inkling2_loop` target
leaves the draft unquantized; the draft dtype needs nothing (fp16 like the target, the
bf16 weights are converted at load).

## What remains / risks

* Measured on the V2 runner (`inkling2-port-spec.md`, part B): the draft is correct (the
  acceptance is far above chance and the greedy output of a speculative run stays inside
  the run-to-run floor of the target), but with all 7 draft tokens verified the step costs
  2.4x a plain decode step (48.2 ms against 20 ms at batch 1: every verified row runs the
  MoE kernels, 8 rows pay the 8-row instance), so `num_speculative_tokens=7` alone gives
  0.98x at batch 1 over eight prompts and loses from batch 4.  The scheduler's dynamic
  speculative decoding verifies fewer tokens while the draft still proposes 7 (the model
  requires `num_speculative_tokens = 7`, the draft block is 8 rows):
  `num_speculative_tokens_per_batch_size=[[1, 1, 3], [2, 8, 0]]` measured 1.21x at batch 1
  (16.50 ms per token against 20.03 ms; K = 1, 2, 4, 7: 1.11x, 1.11x, 1.03x, 0.98x) and the
  no-draft speed from batch 2 up.  Part B2 (`inkling2-port-spec.md`) changed the two causes the
  profile found for the fifth-row cost (the row policy picked the 8-row MoE instance from 5
  tokens, and the old mxfp8 kernel spilled registers; per-launch numbers there): the K table
  above was measured before it; the re-measurement on the merged tree is in "Merged tree"
  below (K = 2 is the best verified width there, 1.24x at batch 1).  Which K is best depends on the
  content (K = 3 on chat, 7 on retrieval-style text): `speculator_options={"adaptive_k": {}}` lets the
  scheduler pick it every step from the running acceptance, measurements in
  `inkling2-speculation-profile.md` section 9.
* `draft_logits.py` (`_assign_rows_kernel` / `_stale_drafts_kernel`) is runner-owned Triton
  used only for probabilistic drafting; greedy drafting (default) does not touch it.
* Prefix-cache hits: the draft K/V blocks are cached with the target's, so a hit reads
  K/V written when the block was produced; the acceptance of a warm run equals that of the
  cold run (part B).  Blocks produced by a run without DSpark would be stale (quality
  only, never correctness).
* The sliding-window manager may free the block holding key `P-1024`; the kernel then reads
  the null block (finite zeros), a one-key mismatch for that single key.
* The dense GEMM design table (`_gemm_design`) is not tuned for the draft's shapes.
* TP > 1: the draft's full-attention layer keeps its K/V replicated on every rank, 4 KiB per
  token and rank (`[INFERENCE: 8 kv heads x 128 x 2 B x 2]`; measured through the engine: the
  TP = 2 recipe engine at 1,048,576 tokens needs 20.18 GiB of KV against 16.13 GiB without the
  draft, 4.05 GiB more, with 16.14 GiB available, so the longest context with the draft is
  836,672 tokens).  Sharding that layer by KV head would halve the cost (about 18.2 GiB,
  still above the 16.14 GiB the 1M engine has); only a window on that layer (the draft then
  sees the last W tokens of the context, a change of the draft's numerics that needs its own
  acceptance measurement) reaches the full context, and it is not built.  The draft's attention
  reads the whole context at every pass (about 1.1 ms at 130k tokens and 7 ms at 800k from the
  4 KiB per token at 450 GB/s `[INFERENCE]`); the long-context draft cost was not measured.

## Merged tree (conv ring + TP + graphs): DSpark on `feat/inkling2-merge` 0722fa5075

What was run: the real checkpoint on a box tree equal to `0722fa5075`, `FULL_AND_PIECEWISE`,
prefix caching on, `speculative_config={"method": "dspark", "num_speculative_tokens": 7,
"num_speculative_tokens_per_batch_size": [[1, 8, K]]}` (the model requires
`num_speculative_tokens == 7`, so every configuration drafts 7 and the schedule picks how
many are verified), 24 chat prompts x 128 tokens, 2 repetitions
(`tools/rdna/inkling2/dspark_bench.py --bench --spec-k 7 --schedule '[[1,8,K]]'`).  The box
was shared with other agents, so the times are indicative (the same engine on two cards
differed by about 1%).  The draft boots in 30 - 40 s at TP = 1 (60 s at TP = 2), captures
its FULL graphs for 1..`max_num_seqs` requests, and the conv ring has `3 + 7 = 10` rows.

### Speed at TP = 1

`max_model_len` 4096, `max_num_batched_tokens` 1024, `max_num_seqs` 8.  Milliseconds per
generated token of one request (speed-up over no draft); K is the number of drafted tokens
verified per step:

| | b1 | b2 | b3 | b4 | b8 |
| --- | --- | --- | --- | --- | --- |
| no draft | 19.11 | 23.40 | 27.38 | 31.11 | 43.22 |
| K = 1 | 17.19 (1.11x) | 22.73 (1.03x) | 30.38 (0.90x) | 34.39 (0.90x) | 42.52 (1.02x) |
| K = 2 | **15.43 (1.24x)** | **21.09 (1.11x)** | 28.69 (0.95x) | 32.81 (0.95x) | 41.16 (1.05x) |
| K = 3 | 15.63 (1.22x) | 21.43 (1.09x) | 29.14 (0.94x) | 32.32 (0.96x) | 42.72 (1.01x) |
| K = 4 | 15.52 (1.23x) | 22.82 (1.03x) | 28.91 (0.95x) | 34.15 (0.91x) | 46.77 (0.92x) |
| K = 6 | 17.08 (1.12x) | 25.06 (0.93x) | 32.94 (0.83x) | 38.41 (0.81x) | 54.99 (0.79x) |
| K = 5, earlier run (no cap) | 16.31 (1.17x) | 24.08 (0.97x) | - | 35.83 (0.87x) | 49.00 (0.88x) |
| K = 7, earlier run | 17.75 (1.07x) | 25.42 (0.92x) | - | 39.27 (0.79x) | 55.92 (0.77x) |

The two earlier rows have their own no-draft run (19.04 / 23.40 / - / 31.06 / 43.09) and
were measured before the Markov cap and before the second ring fix below.  Accepted drafted
tokens per step at batch 1 (the other batch sizes are within 0.03): K = 1 0.636; K = 2 1.026,
cumulative per position `[0.640, 0.386]`; K = 3 1.235, `[0.630, 0.379, 0.225]`; K = 5 1.390;
K = 7 1.474.  The step of the bench (ms/token x (1 + accepted)) is 28.1 ms (K = 1), 31.3 ms
(K = 2) and 34.9 ms (K = 3) against 19.1 ms for a plain step: each verified row costs 3.2 -
3.6 ms and buys 0.39 (second token) and 0.21 (third token) accepted tokens per step.

Policy these numbers support: **K = 2 for one or two requests, no draft above**
(`[[1, 2, 2], [3, 8, 0]]`: 1.24x and 1.11x; drafting three or four requests anyway loses
5%).  K = 3 is within 1.5% of K = 2 at batch 1 and 2 (the policy of the earlier tree,
`[[1, 2, 3], [3, 8, 0]]`); K = 4 ties at batch 1 and loses from batch 2 on (1.03x, 0.95x,
0.91x).  K = 2 at batch 8 measures 1.05x and K = 3
1.01x, inside the spread of two repetitions, and batch 5 - 7 were not run, so the schedule
stops at batch 2.

The Markov cap (the chain stops at the schedule's largest verified K,
`scheduled_draft_tokens`) against the uncapped chain, K = 3: 15.63 / 21.43 / 32.32 / 42.72 ms
at b1 / b2 / b4 / b8 against 15.83 / 21.87 / 32.60 / 44.01 (0.2 - 1.3 ms, 1 - 3%; the
profile predicts 0.5 ms per step, 0.2 ms per token at b1).

### Speed at TP = 2

`FULL_AND_PIECEWISE` with the two all-reduces per layer visit inside the graphs
(`VLLM_RDNA_AR=1`), K = 3, `max_model_len` 4096, `max_num_seqs` 8, HIP 4, 5, one session with
its own no-draft baseline (the code of this tree before the merge; the merge changed the graph
guards and the KV planner, not the draft):

| | b1 | b2 | b4 | b8 |
| --- | --- | --- | --- | --- |
| no draft | 14.12 | 16.32 | 20.63 | 28.54 |
| K = 3 | **10.90 (1.30x)** | **14.98 (1.09x)** | 23.03 (0.90x) | 29.19 (0.98x) |

Accepted per step 1.201 / 1.212 / 1.235 / 1.221 at b1 / b2 / b4 / b8, cumulative
`[0.62, 0.364, 0.216]`.  The first TP = 2 boot with the draft stopped in the KV planner (the
sliding-window bucket mixed the target's 4-head pages with the draft's replicated 8-head pages);
the fix is in `kv_cache_planning.py` of the merge (`inkling2-port-spec.md`, defect 4).

On the merged tree, the opt-in recipe engine (HIP 2, 3; `max_num_seqs` 2, the vision tower, 184
tokens per step; no draft and K = 3 at 800,000 tokens, K = 2 at 780,000), same session and cards:

| | b1 | b2 |
| --- | --- | --- |
| no draft | 14.03 | 16.17 |
| K = 2 | **10.90 (1.29x)** | **14.15 (1.14x)** |
| K = 3 | 10.68 (1.31x) | 14.45 (1.12x) |

Accepted per step at b1 / b2: K = 2 1.001 / 1.022, K = 3 1.235 / 1.224.  K = 2 and K = 3 are
within 2% of each other at TP = 2 (K = 3 ahead at b1, K = 2 at b2), so the recipe takes K = 2 as
at TP = 1; batch 3 and above were not run at TP = 2 on the merged tree (the 8-slot session above:
0.90x at b4 with K = 3).

### Equality with no draft

The expert GEMMs accumulate in fp16 with atomics, so two runs of the same engine already
diverge at near-ties.  The floor is no draft against no draft (A vs A2); a speculative run is
judged against it with `dspark_bench.py --gen`, `--compare` (identical / diverged / diverged
at a near-tie, the gap being the no-draft engine's top-1 logprob minus the other token's) and
`--teacher` (every generated token scored by the no-draft engine).  TP = 1, 96 generations x
96 tokens, K = 3 as B:

| pair | identical | diverged | at a near-tie | not near-tie | largest gap there |
| --- | --- | --- | --- | --- | --- |
| A vs A2 (no draft twice) | 55 | 41 | 31 | 10 | 0.245 |
| A vs B | 51 | 45 | 31 | 14 | 0.276 |
| A2 vs B | 56 | 40 | 28 | 12 | 0.345 |

| teacher pass, 4608 positions | not top-1 | mean gap | max gap | gap > 0.1 | gap > 0.3 |
| --- | --- | --- | --- | --- | --- |
| A | 34 | 4.9e-4 | 0.594 | 5 | 2 |
| A2 | 29 | 4.0e-4 | 0.427 | 6 | 1 |
| B (K = 3) | 26 | 3.3e-4 | 0.236 | 6 | 0 |

TP = 2 (HIP 4, 5), 96 generations x 64 tokens, K = 3 as B:

| pair | identical | diverged | at a near-tie | not near-tie | largest gap there |
| --- | --- | --- | --- | --- | --- |
| A vs A2 (no draft twice) | 66 | 30 | 22 | 8 | 0.302 |
| A vs B | 68 | 28 | 18 | 10 | 0.193 |
| A2 vs B | 70 | 26 | 16 | 10 | 0.378 |

| teacher pass, 3072 positions | not top-1 | mean gap | max gap | gap > 0.1 |
| --- | --- | --- | --- | --- |
| A | 21 | 4.8e-4 | 0.383 | 4 |
| A2 | 19 | 2.5e-4 | 0.122 | 1 |
| B (K = 3) | 16 | 3.1e-4 | 0.137 | 3 |

The speculative output is not distinguishable from the no-draft output beyond the floor, at
TP = 1 and TP = 2 (the verifier's accept rule keeps the target's argmax; what differs is the
fp16-atomic noise of a different row count per step).

### Prefix cache (`dspark_bench.py --prefix-check`, TP = 1, K = 3)

A 942-token prompt pair (a, b) run cold, again (hit) and four requests (a, a', b, b') in one
step.  Cached tokens: cold 0, hit 896, same step `[0, 896, 896, 896]` (the same counts without
the draft).  Accepted per step: cold a 1.474, hit a 1.474 (and the hit's generation is
identical to the cold one), cold b 1.350, hit b 1.400, same step 1.098.  The other hits and
duplicates diverge from their cold run at near-ties or at gaps of 0.05 - 0.20 nats, and the
no-draft control does the same (gaps up to 0.165), so a hit does not add a difference beyond
the floor; the draft K/V of a cached block is the one written when the block was produced and
the acceptance of a hit does not drop.

### What the draft costs in memory

The engine's own KV planner (`ValueError: ... estimated maximum model length`), the recipes'
flags (vision tower loaded, `max_num_batched_tokens` 1024 at TP = 1 and 184 at TP = 2, K = 3
schedule unless a row says K = 2; the graphs of a K = 2 schedule take 0.19 GiB more than K = 3):

| | weights per rank | graphs | longest context |
| --- | --- | --- | --- |
| TP = 1, no draft (`inkling2-serving.md`) | 18.68 GiB | 0.70 GiB (SEQS 16) | 180,986 (SEQS 16), 194,368 (SEQS 2) |
| TP = 1, SEQS 2 | 19.57 GiB | 0.68 GiB | 131,392 (130,000 booted: 131,202 tokens) |
| TP = 1, SEQS 2, K = 2 | 19.57 GiB | 0.87 GiB | 125,824 (125,000 booted: 125,706 tokens; 15.65 / 21.28 ms per token at b1 / b2, 1.22x / 1.10x) |
| TP = 1, SEQS 4 | 19.58 GiB | 1.08 GiB | 119,808 |
| TP = 1, SEQS 8 | 19.58 GiB | 1.88 GiB | 96,512 |
| TP = 1, SEQS 16 | 19.59 GiB | 3.44 GiB | 51,072 |
| TP = 2, no draft (`inkling2-serving.md`) | 9.6 GiB | 0.19 GiB | 1,062,736 (1,048,576 needs 16.1 GiB) |
| TP = 2, SEQS 8, 1,048,576 asked | 10.38 GiB | - | refused before the graphs: 20.18 GiB needed, 16.14 GiB available, 836,672 would fit |
| TP = 2, SEQS 8, 800,000 asked | 10.38 GiB | 1.59 GiB | refused: 15.44 GiB needed, 14.24 GiB available, 737,088 fit |
| TP = 2, SEQS 2, 800,000 asked | 10.37 GiB | 0.38 GiB | 800,632 tokens (booted); a K = 2 boot of the same settings estimated 790,592 |
| TP = 2, SEQS 2, K = 2, 780,000 asked | 10.37 GiB | - | 790,494 tokens (booted; 10.90 / 14.15 ms per token at b1 / b2) |

Three costs: 0.8 - 0.9 GiB of draft weights per rank; 4 KiB of KV per token and rank for the
draft's full-attention layer (replicated at TP > 1: 20.18 - 16.13 = 4.05 GiB at 1,048,576
tokens); and graph memory that grows with `max_num_seqs` (the draft's graphs and the
verification graphs; `[INFERENCE]` the engine captures verification graphs for the full block
width of 8 rows per request count, and a list limited to the widths the schedule produces would
take less; not built).  Sharding the draft's attention by KV head would halve the second cost
(about 18.2 GiB at 1,048,576 tokens, against 16.14 GiB available) and still not reach the full
context; a window on that layer would, and it changes the draft's numerics (acceptance at long
context not measured), so it is not built.  The 1,048,576-token recipes therefore keep DSpark
off; the recipes carry opt-in lines with the longest context that was booted.  TP = 4 was not run
through the engine (`[INFERENCE]` the cache needs 12 KiB per token and rank there, 12.3 GiB at
1,048,576 tokens, against a pool of 17.2 GiB without the draft).

### Where a verification step goes

`profile_decode.py --spec-k 7`, async scheduling off, random prompts, so the numbers are of a
K = 7 step, 8 rows per request: batch 1 clean step 37.7 ms (36.8 ms in a second run), GPU
kernels 33.2 / 36.4 ms; the draft is 3.6 ms of the profiler's per-step mean and 4.7 ms in the
traced step (68 kernels, 5.2 ms from the first to the last).  Within a traced draft pass: the
base-logits sweep of the LM head 2.15 ms (one 922 MB read, 429 GB/s), the three layers 1.7 ms
(qkv 71 - 78 us, o_proj 54 - 61 us, gate/up 212 - 216 us, down 128 - 129 us per layer), seven
Markov steps about 0.13 ms each (`markov_step` 16 us, `head_logits` 62 us after a 249 us cold
first one, `argmax` 21 us, launch gaps), the context path 0.2 ms.  The target's verification of
8 rows is 29.7 ms (MoE NVFP4 13.3 ms, dense GEMV 8.4 ms, its own LM head 2.1 ms).  At batch 4
the step is 69.5 ms and the draft's kernels 11.0 ms (the 28-row base GEMV and the backbone on
`gemv_kernel<32>`: 8.3 ms), which only matters where the policy drafts.  The bench's effective
step at K = 7 and batch 1 (43.9 ms) is 6 - 7 ms above the profile's clean step; where that goes
is open (`dspark_bench.py --async-scheduling off` switches the engine's async scheduling to
compare).  The profiler traces show 5 - 7 ms gaps inside a step that clean steps do not have
(busy / wall 98.9% in the second run) `[INFERENCE: profiler artifact]`.

Levers, by what the trace shows.  Done: the Markov chain stops at the schedule's largest
verified K (`scheduled_draft_tokens`, `_markov_steps`; the tokens after it are never read: the
runner passes `draft_tokens[:, :num_spec_tokens]` on), 0.13 ms per skipped step (measured
above); its GPU oracle test (`test_the_markov_chain_stops_at_the_largest_verified_token`)
passes.  Done since (`inkling2-speculation-profile.md` section 9): a frequency-pruned draft
vocabulary as an explicit opt-in file (`speculator_options={"draft_vocab": "ids.npy"}`, made offline
by `tools/rdna/inkling2/draft_vocab.py`; TP = 1, greedy), the whole draft's graph replay at K = 2
falls from 4.47 to 2.37 ms with 32768 ids and the acceptance was measured on real outputs; and a
content-adaptive number of verified drafts (`speculator_options={"adaptive_k": {}}`, scheduler side;
`enable_adaptive_verification` is refused at config time: the Markov head keeps no per-position draft
distribution for it).  Not done: an 8-bit copy of the LM head for the base sweep (about -1 ms per
step at the full vocabulary, 0.17 ms with the pruned one; costs 0.46 GB and acceptance risk
`[INFERENCE]`), one draft graph per verified K (the unused Markov steps, 0.2 - 0.45 ms), graph
capture limited to the widths the schedule produces (memory only; the adaptive mode captures them
all, +0.5 GiB).  The sweep is at the memory roofline, the layers are at 250 - 395 GB/s: the draft is
about 14% of a K = 7 step at the full vocabulary and the K = 2 / 3 gain is bounded by the
verification rows, not the draft.

Bugs found on the merged tree: FULL capture of the draft (verify batches of 8 rows with the
capture's dummy metadata, `seq_lens = 1`, zero block tables) made the conv ring write a negative
row (`Memory Fault ... i2_state_write_kernel`): `i2_ring_keeps` accepted `p < 0`, and padded
PIECEWISE tokens read as request 0 and wrote its ring.  The merge fixes both
(`csrc/rocm/inkling2_attn_rdna2.cu`: `0 <= p < p1`, no write to block 0; `amd/state.py`:
`seq_idx = -1` for padded tokens); the K = 1, 2, 3 rows above ran on the merged tree, the
earlier K = 5 / 7 rows with the first fix only.

DSpark at TP > 1 (`amd/dspark.py`, `tests/models/inkling2_loop/test_dspark_tp.py`): the draft
weights, its K/V and the taps are replicated, so no draft collective sits inside the layers.
The anchors' embedding rows come from the target's vocabulary shard
(`inkling2_glue_embed_lookup`) and one all-reduce of the `[rows, hidden]` block (at most
64 KiB); the folded head runs over this rank's vocabulary rows and
`inkling2_dspark_argmax_partial` (value and global id of the best row of the shard, padding rows
excluded) feeds an all-gather and `inkling2_linear_argmax_reduce` (ties to the lowest id, as at
TP = 1).  Greedy drafting only: the probabilistic draft raises `NotImplementedError`.

### Gates on the merged tree

Box (HIP 2): `test_inkling2_dspark_rdna2.py` + `test_dspark_model.py` + `test_dspark_tp.py`
together 178 passed in 99.5 s (139 kernel tests including 9 for `argmax_partial`; 39 model /
TP tests, thread-emulated TP = 2 / 4 drafts equal to the single-rank draft, the Markov-cap
oracle test).  `test_dspark_model.py -k markov_steps` on the CPU 7 passed.  Six-arch compile
of the TU (gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1200): exit 0 and no errors on all six
(34 kernels on gfx1030, none elsewhere by design).  No-draft `FULL_AND_PIECEWISE` against eager
(6 prompts, 64 tokens): 13 of 24 identical, the divergences at index >= 10 on near-ties of at
most 0.17 nats.

Mutation check of `inkling2_dspark_argmax_partial` (scratch tree, dev library built with the
torch JIT at `-O3`, tests through the card lock): three workers, each with its own baseline
(6 failed, 133 passed in 49 - 51 s: the six cross-shard cases
`test_argmax_partials_of_vocabulary_shards_reduce_to_the_global_argmax[{1,5,32}-{2,4}]`, whose
reduce op is not in the dev library `[INFERENCE]`), then one mutant at a time with those six
deselected.  All six mutants are **killed**: `partial_drops_first_id` (first run);
`partial_value_row_zero` and `partial_first_id_off_by_one` (1 failed, 112 passed, first failing
`test_argmax_partial_is_the_value_and_the_global_id[1-100026-100032-100032]`);
`partial_id_row_overlaps_values` and `partial_scans_the_shard_padding` (1 failed, 113 passed,
`test_argmax_partial_is_the_value_and_the_global_id[7-4099-4112-0]`);
`partial_ties_to_higher_index` (4 sites, 1 failed, 109 passed,
`test_argmax_matches_reference[1-200058-200064]`).

## 2026-11-24 optimization pass: draft attention, concurrency, and TP=1/2/4 recipes

Four changes remove the long-context attention bottleneck and concurrency collapse when serving
DSpark at `SEQS > 2`:

1. **Vectorized split-KV draft attention (`csrc/rocm/inkling2_dspark_rdna2.cu`)**:
   * Adaptive `num_splits` targeting high CU occupancy at long context (64k) with two-stage online
     softmax reduction and coalesced `V_DOT2` K/V loads.
   * Microbench on V620 (`par1-cs25` HIP 2, `base -> cand`):
     * `b=1 ctx=1024 win=1024`: `139.5 -> 36.4 us` (**3.83x**)
     * `b=1 ctx=1024 win=0`: `139.0 -> 31.2 us` (**4.46x**)
     * `b=1 ctx=16384 win=0`: `1,063 -> 218.2 us` (**4.87x**, 307.7 GB/s)
     * `b=1 ctx=65536 win=0`: `2,961 -> 932.6 us` (**3.18x**, **-2.03 ms/draft step**)
     * `b=2 ctx=65536 win=0`: `3,973 -> 1,951 us` (**2.04x**)
   * Gates: 29/29 attention oracle tests pass; 5/5 hand mutants killed (`survived=0`); 6-arch
     `archdiff` clean (0 diffs on `gfx90a/942/950/1100/1200`); built-op `test_inkling2_dspark_rdna2.py`,
     `test_dspark_verifier.py` and `test_dspark_target_taps.py` 148 passed, `test_dspark_tp.py`
     5 passed, `test_dspark_model.py` 37 passed.
2. **Schedule-aware graph capture (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` and
   `vllm/config/vllm.py`)**:
   * `DFlashCudaGraphManager` caps draft FULL graph capture to the largest batch size where the
     dynamic schedule drafts (`k > 0`, i.e., `1..2` for `[[1,2,2],[3,N,0]]`), so raising `SEQS`
     adds zero dead draft graphs.
   * `VllmConfig._set_cudagraph_sizes` computes the default `max_cudagraph_capture_size` from
     `max(q * tier_max_reqs)` across `target_query_lens` instead of `max_num_seqs * uniform_decode_query_len`.
     On TP=1 DSpark (`SEQS=16`, `[[1,2,2],[3,16,0]]`), this cuts target PIECEWISE capture from
     37 graphs (up to 256 tokens, `3.60 GiB`) to 9 graphs (up to 32 tokens, `0.90 GiB`),
     reclaiming **2.70 GiB** and lifting the KV pool from 45,824 to **122,992 tokens**.
3. **TP `draft_vocab` support (`vllm/models/inkling2_loop/amd/dspark.py`)**:
   * Each TP rank filters the requested draft vocabulary ids to its local vocabulary shard,
     gathers local head weights, runs `argmax_partial` with global token ids, and all-gathers
     across ranks (handles empty local shards via `-inf` sentinel).
4. **Updated TP=1/2/4 DSpark recipes**:
   * `inkling2-mini-tp1-dspark`: `SEQS=16`, `MAXLEN=115000`, `[[1,2,2],[3,16,0]]` (122,992 KV tokens, 0.90 GiB graphs).
   * `inkling2-mini-tp1-dspark-adaptive`: `SEQS=16`, `MAXLEN=95000`, `[[1,2,7],[3,16,0]]` (103,659 KV tokens, 1.40 GiB graphs).
   * `inkling2-mini-tp2-dspark`: `SEQS=8`, `MAXBAT=1024`, `MAXLEN=720000`, `[[1,2,2],[3,8,0]]` (734,936 KV tokens).
   * `inkling2-mini-tp4-dspark` (new): `SEQS=8`, `MAXLEN=1048576`, `[[1,2,2],[3,8,0]]` (1,083,012 KV tokens, full 1M context fits).

### Measured E2E (`par1-cs25` b4, 2026-11-24, `serve_validate.sh`, `--temperature 0`)

| Recipe | KV tokens | 1k/512 c=1 prefill | 1k/512 c=1 decode | 1k/512 c=8 agg out | 1k/512 c=8 TTFT | 64k/256 c=1 prefill | 64k/256 c=1 ITL | Faults |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inkling2-mini-tp1-dspark` | 122,992 | 2,084.7 tok/s | 93.76 tok/s/req | 184.66 tok/s | 1.86 s | 1,519.9 tok/s | 40.3 ms | 0 |
| `inkling2-mini-tp2-dspark` | 734,936 | 3,043.9 tok/s | 135.67 tok/s/req | 283.81 tok/s | 1.21 s | 2,403.9 tok/s | 27.7 ms | 0 |
| `inkling2-mini-tp4-dspark` | 1,083,012 | 3,866.7 tok/s | 163.76 tok/s/req | 386.18 tok/s | 1.29 s | 3,475.7 tok/s | 22.6 ms | 0 |

Real-text TP=1 `dspark_bench.py` (`SEQS=16`, `[[1,2,2],[3,16,0]]`, 24 chat prompts, greedy):

* `b=1`: `14.36 ms/tok/req` (**69.6 tok/s**, 1.014 accepted/step)
* `b=2`: `19.03 ms/tok/req` (**105.1 tok/s**, 1.016 accepted/step)
* `b=4`: `28.41 ms/tok/req` (**140.8 tok/s**, draft automatically dropped)
* `b=8`: `38.68 ms/tok/req` (**206.8 tok/s**, draft automatically dropped)

### TP-aware `StepCostModel`, `ADAPTIVE=1` + `draft_vocab` at TP=2/4, and 4-recipe consolidation

`Inkling2DSparkSpeculatorConfig.make_num_tokens_policy` now accepts `tensor_parallel_size`
(passed from `Scheduler`) and selects measured V620 step-cost models for TP=1, TP=2 and TP=4
while scaling `draft_vocab` LM-head savings inversely with TP (`_FULL_VOCAB_DRAFT_MS / tp`,
matching measured `-2.1 / -1.12 / -0.5 ms/step` at TP=1/2/4).

Measured single-stream step times (`par1-cs25` b4, 160 tokens, greedy, `ba9b43bcd4`):

| TP | K=0 | K=1 full | K=2 full | K=3 full | K=7 full | K=2 32k vocab | K=7 32k vocab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=2 | 12.22 ms | 18.47 ms | 20.14 ms | 21.47 ms | 25.78 ms | 19.02 ms | 25.05 ms |
| TP=4 | 9.50 ms | 14.87 ms | 16.06 ms | 17.02 ms | 20.11 ms | 15.61 ms | 19.50 ms |

Measured 6-category real-text decode (`chat`, `code_gen`, `code_edit`, `prose`, `structured`,
`rag`; 3 prompts each, 160 tokens, greedy):

| Config | KV pool | Mean (6 cats) | `chat` | `code_gen` | `code_edit` | `prose` | `structured` | `rag` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=2 no draft (`K=0`) | 973,541 | 81.8 tok/s | 82.1 | 82.0 | 81.5 | 82.1 | 81.9 | 81.5 |
| TP=2 `K=2` full | 723,444 | 113.3 tok/s | 95.4 | 118.6 | 119.9 | 92.6 | 140.1 | 128.3 |
| TP=2 `K=2` + 32k vocab | 719,494 | 121.1 tok/s | 109.4 | 129.8 | 128.1 | 98.2 | 134.9 | 136.8 |
| TP=2 `ADAPTIVE=1` + 32k vocab (TP=2 cost) | 693,467 | **139.0 tok/s** (1.70x) | 100.5 | 130.2 | 150.1 | 116.2 | 185.8 | **204.5** |
| TP=4 no draft (`K=0`) | 2,247,076 | 105.3 tok/s | 105.3 | 105.6 | 104.5 | 105.8 | 105.4 | 105.0 |
| TP=4 `K=2` full | 1,177,580 | 143.3 tok/s | 117.8 | 150.7 | 143.4 | 128.4 | 168.9 | 165.4 |
| TP=4 `K=2` + 32k vocab | 1,173,297 | 145.9 tok/s | 119.9 | 140.7 | 154.3 | 138.2 | 164.4 | 170.0 |
| TP=4 `ADAPTIVE=1` + 32k vocab (TP=4 cost, 1M ctx) | 1,125,661 | **154.4 tok/s** (1.47x) | 121.8 | 131.8 | **179.0** | 109.3 | **226.9** | **254.7** |

Recipes were consolidated from 7 to 4 (`inkling2-mini-tp1`, `inkling2-mini-tp1-dspark`,
`inkling2-mini-tp2`, `inkling2-mini-tp4`), all with `prompt-tokens-details` enabled
(verified `usage.prompt_tokens_details.cached_tokens` 0 cold / 64 warm on TP=1) and
`ADAPTIVE=1`, `DRAFT_VOCAB=...` and `DSPARK=0` switches built in.
