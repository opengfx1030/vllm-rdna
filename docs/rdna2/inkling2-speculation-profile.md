# Inkling-2-Mini b=1 speculation profile on gfx1030 (V620), TP=1

Measured on `par1-cs25`, HIP device 3, real checkpoint (`~/hfcache/inkling2-mini`), tree = worktree HEAD
`f459e45e0d` (csrc and library = `e505d4d436`: DecodeGlue's SwiGLU->w2 fusion, two-row verify blocks, `moe_fast` dequant,
`VLLM_INKLING2_FUSE_GLUE` on). Engine: `FULL_AND_PIECEWISE`, `max_num_seqs 2`, prefix caching off, one request at a time (b=1),
160 generated tokens per prompt, temperature 0, `ignore_eos`. Draft = the shipped HIP DSpark draft (3 layers, 8-row block,
Markov head). Sections 1-8 used scratch tools only (no repo file was changed); section 9 implements what they found.
**[INFERENCE]** marks arithmetic or an estimate, not a measurement.
Sample sizes are small (6-12 prompts per type, one run per configuration; run-to-run noise: section 9 measures 0.1 % without a
draft and +-1-5 % with one).

## 1. Summary

1. **No draft at HEAD: 18.0-18.7 ms/token (55 tok/s).** Step = forward 15.7-16.1 + lm_head 2.11 + 0.1 ms (phase timer).
2. **DSpark K=2 (the default schedule): 11.5-16.3 ms/token depending on the content**, 1.11x (prose) to 1.62x (retrieval-style
   text with quoting). Chat: **15.21 ms/token = 65.7 tok/s (1.18x)**. The step is 29.1 ms (forward 22.5, lm_head 2.1,
   draft 4.3, other 0.3); the host adds 0.4 ms per step (1.4 %): the step is GPU-bound.
3. **The best K depends on the content, and the cost model says how much it is worth.** Marginal forward cost per verify row
   is 3.55 ms for rows 2-3 and 1.94 ms for rows 4-8 (expert union saturates), so long drafts are cheap per row and only pay
   where acceptance is high. Per-type best static K (model, validated at K=2 and K=7 within 0.1-0.4 ms): chat K=3 (-2.1 % vs
   K=2), code_gen K=3 (-4.3 %), prose K=3 (-0.5 %), structured K=5 (-9.6 %), code_edit K=6 (-17.8 %), rag K=7 (-33.8 %).
   Measured e2e at K=7: rag 8.01 ms/token (2.32x over no draft), code_edit 10.70 (1.74x), but chat 17.18 and prose 19.40
   (worse than K=2 and, for prose, worse than no draft).
4. **A content-adaptive K is the best exact combination, and it is built (section 9)**: the scheduler picks K every step from the
   running acceptance (`speculator_options={"adaptive_k": {}}`, opt-in, lossless, every width graph-captured). Measured on these
   prompts, the mean of the six types goes 14.01 (K=2) -> 13.04 ms/token (-6.9 %); the oracle of section 4 that knows the type is
   12.75. The HIP DSpark still does not support `enable_adaptive_verification` (it is refused at config time; it was an
   `AssertionError` at `vllm/models/inkling2_loop/amd/dspark.py:970`).
5. **Prompt lookup (n-gram) does not beat the DSpark draft.** The engine's `ngram` method verifies a fixed K+1 rows every step:
   measured 14.5-26.5 ms/token at K=6 (worse than no draft except rag, 14.49 vs 18.59). A variable-width replay of the exact
   greedy streams (rows = 1 + proposal length, no draft pass) gives 11.3-17.4 ms/token: equal to DSpark K=2 on rag (11.3-11.5),
   worse everywhere else. The hybrid (lookup + DSpark) was replayed offline in section 9.6 (no gain over DSpark alone on 5 of 6
   types); `spec_trace.py` (joint per-step trace) is written but was not run.
6. **A cheaper draft is worth 6-7 % per token and is built as an opt-in file (section 9.5)**: with a restricted draft vocabulary
   of 32768 ids (made offline) the draft step falls from 4.47 to 2.37 ms at K=2 (graph replay), the first-position acceptance
   stays within +-3 %, and the mean of the six types goes 13.04 -> 12.28 ms/token (12.09 with the tuned policy). An int8 head or
   W4 draft layers are not built (-0.17 ms on top of the pruned list **[INFERENCE]**). It only changes acceptance, never the
   output.
7. **<= 10 ms/token at b=1 is reached only on copy-heavy content** (section 9: rag 7.69 ms/token = 130 tok/s, code_edit 9.95 =
   100.5 tok/s; static K=7 above: 8.01 / 10.70). Chat is at 14.30 ms/token (69.9 tok/s, 1.26x), prose 15.64 (63.9 tok/s); the
   mean of the six types is 12.09 ms/token (82.7 tok/s, 1.51x over no draft).

## 2. Method and commands

```bash
P=~/w4a8_runs/inkling2-work/Inkling2PrefillProfile        # tree2 = worktree HEAD + DecodeGlue's built libraries
export TREE=$P/tree2 I2_JIT_DIR=$P/jit ROCM_HOME=/opt/rocm/core-7.14 PATH=/opt/rocm/core-7.14/bin:$PATH VLLM_RDNA_PHASE_TIMING=100000
G="bash ~/w4a8_runs/inkling2-work/Lead/gpu_sub.sh 3 900 $P/scripts/spec_types_bench_v2.py --json $P/out/spec/TAG.json --tokens 160"
$G --spec-k 0 --save-ids $P/out/spec/ids_k0.json     # no draft (k0)
$G --spec-k 2                                         # DSpark verifying 2 (k2)
$G --spec-k 7                                         # DSpark verifying all 7 (k7)
$G --spec-k 7 --adaptive                              # ad7: AssertionError at dspark.py:970 (boot)
$G --method ngram --spec-k 6 --ngram-min 2 --ngram-max 3    # engine n-gram (ng6)
python3 $P/scripts/spec_report.py                     # tables below (parses the phase timer + json + replays n-gram)
```

`spec_types_bench.py` boots the engine once and for every content type runs `dspark_bench.bench` (ms per generated token of
one request, (t_N - t_1)/(N - 1); accepted tokens per step and cumulative acceptance per draft position from the spec
counters) and flushes the phase timer (`VLLM_RDNA_PHASE_TIMING`) per type, so forward / lm_head / sample / draft / gap per
verify width are per content type. Prompt sets (`spec_prompts.py`, all chat-templated): chat = the 12 first
`dspark_bench.CHAT_PROMPTS` (default reasoning effort), code_gen 8, code_edit 6 (real repo files, 1148-1312 tokens), prose 8,
structured 6, rag 6 (465-1531 tokens, excerpts of `docs/rdna2/inkling2-dspark.md`); all except chat use
`reasoning_effort="none"`. The earlier first attempt of the no-draft job was SIGTERMed from outside (log
`spec_k0.log` of the first attempt overwritten); it was rerun unchanged.

## 3. Measured

### 3.1 ms/token, tokens/s and accepted tokens per step (b=1)

| type | no draft | K=2: ms (tok/s) acc/step, vs none | K=7: ms (tok/s) acc/step, vs none | engine n-gram K=6 ms, acc/step |
|---|---|---|---|---|
| chat | 18.01 (55.5) | 15.21 (65.7) 0.941, 1.184x | 17.18 (58.2) 1.343, 1.048x | 25.26, 0.156 |
| code_gen | 18.07 (55.3) | 14.36 (69.6) 1.068, 1.258x | 15.91 (62.9) 1.535, 1.136x | 25.25, 0.173 |
| code_edit | 18.66 (53.6) | 12.68 (78.9) 1.500, 1.471x | 10.70 (93.5) 3.089, 1.744x | 20.58, 0.557 |
| prose | 18.08 (55.3) | 16.26 (61.5) 0.825, 1.112x | 19.40 (51.5) 1.045, 0.932x | 26.52, 0.093 |
| structured | 18.09 (55.3) | 13.57 (73.7) 1.203, 1.333x | 13.34 (75.0) 2.067, 1.356x | 24.26, 0.208 |
| rag | 18.59 (53.8) | 11.49 (87.0) 1.791, 1.618x | 8.01 (124.8) 4.603, 2.321x | 14.49, 1.312 |

### 3.2 DSpark cumulative acceptance per draft position (K=7 run)

| type | a1 | a2 | a3 | a4 | a5 | a6 | a7 | tokens/step with 7 verified |
|---|---|---|---|---|---|---|---|---|
| chat | 0.586 | 0.324 | 0.178 | 0.111 | 0.072 | 0.044 | 0.029 | 2.34 |
| code_gen | 0.625 | 0.377 | 0.231 | 0.134 | 0.087 | 0.049 | 0.032 | 2.53 |
| code_edit | 0.843 | 0.668 | 0.494 | 0.366 | 0.302 | 0.243 | 0.174 | 4.09 |
| prose | 0.535 | 0.240 | 0.128 | 0.072 | 0.035 | 0.019 | 0.014 | 2.04 |
| structured | 0.681 | 0.473 | 0.326 | 0.230 | 0.173 | 0.115 | 0.070 | 3.07 |
| rag | 0.908 | 0.833 | 0.747 | 0.667 | 0.592 | 0.466 | 0.391 | 5.60 |

a2 at K=2 predicts the content class: chat 0.34, prose 0.26, code_gen 0.40 (best K 3), structured 0.49 (K 5), code_edit 0.65
(K 6), rag 0.86 (K 7).

### 3.3 Step anatomy (phase timer, GPU ms per step, means over the steps of the type)

| run / width | forward | lm_head | draft | step total | steps |
|---|---|---|---|---|---|
| no draft, 1 row (6 types) | 15.72-16.07 | 2.11 | 0.02 | 17.97-18.27 | 14,629 |
| K=2, 3 rows: chat / code_gen / prose / structured | 22.49 / 22.59 / 22.69 / 22.85 | 2.12 | 4.33-4.36 | 29.10 / 29.21 / 29.29 / 29.48 | 404-1064 |
| K=2, 3 rows: code_edit / rag (1.2-1.5k-token contexts) | 24.12 / 24.49 | 2.12 | 4.77 / 4.72 | 31.17 / 31.49 | 454 / 404 |
| K=7, 8 rows: 4 short types | 32.05-32.66 | 2.16-2.17 | 4.96-4.99 | 39.36-39.98 | 368-884 |
| K=7, 8 rows: code_edit / rag | 34.47 / 35.02 | 2.16 | 5.41 / 5.31 | 42.20 / 42.66 | 286 / 207 |

Forward(rows) means over all steps: 1 row 15.90, 3 rows 22.99, 8 rows 32.67 ms: **+3.55 ms per row for rows 2-3, +1.94 ms per
row for rows 4-8**. The n-gram run's 7-row steps (26.6-30.2 ms) are not comparable (the padded rows repeat one token and
route to few experts). Draft cost grows 4.43 -> 5.04 ms from K=2 to K=7 (the Markov chain is capped at the scheduled K).
Phase totals reproduce the measured ms/token within 1.4 % (K=2 chat: 29.10 ms x 1/1.941 tokens = 14.99 vs 15.21 measured).

## 4. Cost model and best K

Step cost = forward(1+K) + lm_head + draft(K) + 0.11 (sample, gap) + 0.4 (host exposure, K > 0); tokens per step = 1 +
sum of the cumulative acceptances (greedy verification is exact); forward(rows) piecewise-linear through the measured 1, 3, 8
rows with a per-type offset for the context length. ms/token (`static_k_model_corrected.txt`; model vs measured at K=2 /
K=7 in the last two columns):

| type | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | best K | best | vs K=2 | K=2 model / meas | K=7 model / meas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 18.04 | 16.42 | 15.47 | **15.15** | 15.34 | 15.77 | 16.38 | 17.06 | 3 | 15.15 | 1.021x | 15.47 / 15.21 | 17.06 / 17.18 |
| code_gen | 18.09 | 16.07 | 14.81 | **14.21** | 14.27 | 14.61 | 15.15 | 15.78 | 3 | 14.21 | 1.043x | 14.81 / 14.36 | 15.78 / 15.91 |
| code_edit | 18.37 | 14.66 | 12.42 | 11.11 | 10.56 | 10.29 | **10.21** | 10.31 | 6 | 10.21 | 1.216x | 12.42 / 12.68 | 10.31 / 10.70 |
| prose | 18.34 | 17.13 | 16.76 | **16.69** | 17.09 | 17.79 | 18.61 | 19.46 | 3 | 16.69 | 1.005x | 16.76 / 16.26 | 19.46 / 19.40 |
| structured | 18.12 | 15.62 | 13.89 | 12.91 | 12.58 | **12.55** | 12.77 | 13.16 | 5 | 12.55 | 1.106x | 13.89 / 13.57 | 13.16 / 13.34 |
| rag | 18.39 | 14.26 | 11.51 | 9.69 | 8.67 | 8.06 | 7.77 | **7.62** | 7 | 7.62 | 1.510x | 11.51 / 11.49 | 7.62 / 8.01 |

Mixes (ms/token): equal weights none 18.25, K=2 13.93, K=7 14.09, per-type best K 12.74 (-8.6 % vs K=2, 71.8 -> 78.5 tok/s);
chat-heavy (chat .5, code_gen .15, prose .15, others .067 each) none 18.12, K=2 14.71, K=7 16.02, per-type best K 14.24
(-3.2 %, 68.0 -> 70.2 tok/s). Intermediate K values rest on the linear forward interpolation between 3 and 8 rows
(validated only at K=2 and K=7; see 4.1 for K=4).

Cheaper draft on top of the best K (acceptance unchanged **[INFERENCE: an upper bound]**): chat K=3 15.15 -> 14.65 (int8 draft
head), 14.27 (+ W4 draft layers), 14.32 (vocab-pruned head, calibration, opt-in); rag K=7 7.62 -> 7.44 / 7.29 / 7.31.

### 4.1 K=4 validation run

Measured with the same harness (`logs/spec_k4.log`, `out/spec/k4.json`, phase timer at 5 rows):

| type | K=4 measured ms/token | acc/step | cum a1..a4 | forward at 5 rows | step total | model K=4 |
|---|---|---|---|---|---|---|
| chat | 15.45 | 1.226 | 0.584 0.335 0.194 0.114 | 26.93 | 33.79 | 15.34 |
| code_gen | 13.87 | 1.492 | 0.665 0.411 0.263 0.154 | 26.96 | 33.85 | 14.27 |
| code_edit | 11.07 | 2.375 | 0.818 0.684 0.495 0.379 | 28.85 | 36.15 | 10.56 |
| prose | 17.22 | 0.994 | 0.547 0.250 0.126 0.071 | 27.03 | 33.91 | 17.09 |
| structured | 12.46 | 1.809 | 0.684 0.490 0.368 0.267 | 27.30 | 34.20 | 12.58 |
| rag | 9.14 | 3.124 | 0.910 0.825 0.739 0.650 | 29.21 | 36.49 | 8.67 |

The model is within 0.1-0.5 ms/token of the measurement at K=4 (the linear forward interpolation is 0.3-0.6 ms low at 5
rows, so the model is up to 5 % optimistic on rag and code_edit at K=4-6). **Best measured K among {2, 4, 7} per type**: chat
K=2 15.21, code_gen K=4 13.87 (-3.4 % vs K=2), code_edit K=7 10.70 (-15.6 %), prose K=2 16.26, structured K=4 12.46 (-8.2 %),
rag K=7 8.01 (-30.3 %). Mixes from these measurements: equal weights 12.75 ms/token vs 13.93 at K=2 (**-8.5 %**, 78.4 tok/s),
chat-heavy 14.20 vs 14.71 (**-3.5 %**, 70.4 tok/s).

## 5. Ranked levers for b=1 tokens/s (all exact for the output unless noted)

| # | lever | evidence | gain | risk / status |
|---|---|---|---|---|
| 1 | per-request adaptive K (K=3-4 on chat-like text, up to 6-7 when the recent acceptance is high) | measured K=2/4/7 per type, model validated within 0.1-0.5 ms; rag 11.49 -> 8.01 (-30 %), code_edit 12.68 -> 10.70 (-16 %), structured 13.57 -> 12.46 (-8 %), code_gen 14.36 -> 13.87 (-3 %); chat and prose stay at K=2 | measured mixes: equal -8.5 %, chat-heavy -3.5 %; copy-heavy content up to 1.4x over K=2 | **built, section 9.4**: scheduler policy `adaptive_k` (opt-in), every width captured (+0.5 GiB graphs, -9 % KV); mean of the six types 14.01 -> 13.04 ms/token; engine adaptive verification stays unsupported (refused at config time) |
| 2 | cheaper draft: restricted draft vocabulary (built), int8 head, W4 draft layers (not built) | draft = 4.47 ms at K=2 (graph replay): head sweep 2.15 ms (922 MB, 428 GB/s), 3 layers 1.48, Markov 0.42, gaps 0.34 (section 9.2) | **measured**: 32768 ids 4.47 -> 2.37 ms, mean of six 13.04 -> 12.28 ms/token (-5.8 %), first-position acceptance within +-3 %; int8 head -0.17 ms on top [INFERENCE] | opt-in offline ids file, TP=1 greedy, draft-only (output unchanged); section 9.5 |
| 3 | verify-row MoE efficiency (DecodeGlue) | 3-row real-routing MoE 208 us/visit vs 164 us stream floor (Phase 1) | <= 1.3 ms/step = 0.65 ms/token on chat (4 %) | owned by DecodeGlue |
| 4 | prompt lookup / hybrid | engine n-gram fixed width 14.5-26.5 measured; replay 11.3-17.4, no better than DSpark K=2 | hybrid replayed offline (section 9.6): worse than DSpark alone on 5 of 6 types, -1.1 % on chat | not built; needs variable-width verification |
| 5 | skip experts on draft rows / restrict routing | changes the verify logits | not exact: rejected | - |

What breaks the expert-growth penalty (task item 2): nothing exact removes it, each extra row needs its own top-8. The
measured marginal cost falls from 3.55 to 1.94 ms per row after row 3, so the penalty is paid once; it is the *acceptance
curve* that decides how many rows pay, which is why lever 1 beats every kernel lever on copy-heavy text. Two-row expert blocks
(`_BLOCK_M_BY_TOKENS_SPEC`) are already in HEAD. Distinct experts per verify step on real text (15.9-17.6 at 3 rows, 19-20 at 4,
about 29 at 8, against 22.6 / 29.2 for independent requests) and the 2/3 expert bytes + 1/3 per-row overhead split: section 9.3.

## 6. Design of lever 1 (first design; built in section 9.4 with a cost-model estimator instead of these thresholds)

* Controller (host): EMA of the accepted tokens and of a2 over the last about 8 steps of the request (committed counts are
  known one step late under async scheduling, which is fine for an EMA). Map: a2 < 0.45 -> K=3; 0.45-0.6 -> K=5; 0.6-0.8 ->
  K=6; > 0.8 -> K=7 (thresholds read from section 3.2, **[INFERENCE]** from six content types).
* The draft already produces 7 tokens per step, so K can change per step without recapturing the draft; draft cost rises only
  4.43 -> 5.04 ms. The verify side needs a graph per width 1+K at b=1 (7 FULL graphs; memory small **[INFERENCE]**).
* Files **[INFERENCE: not read in depth]**: scheduler K selection (today keyed by batch size, `vllm/config/speculative.py`
  `num_speculative_tokens_per_batch_size`), the runner's draft-token trim, `inkling2_loop/amd/dspark.py` (the Markov cap).

## 7. Not done / open risks

1. The adaptive-K policy was built in section 9 (opt-in `speculator_options`); its open risks are in 9.10.
2. Greedy equality against the no-draft output was re-run for the adaptive and draft-vocabulary paths in 9.8 (46 prompts, same
   floor as two no-draft runs).
3. Small samples and one run per configuration; copy-heavy types depend on the prompt design (quotes of repo files).
4. The joint DSpark/lookup trace (`spec_trace.py`) was replaced by an offline replay of the real streams (9.6); the cheaper-draft
   acceptance was measured for the restricted vocabulary (9.5), not for int8 / W4 draft weights.

## 8. Files and raw data

Box `$P/out/spec/{k0,k2,k7,ng6,k4}.json`, `ids_k0.json`; logs `$P/logs/spec_{k0,k2,k7,ad7,ng6,k4}.log` (phase timer lines after
`[PHASE-FLUSH after <type>]`); local copies `Inkling2PrefillProfile/data/spec/`. Scratch tools `spec_prompts.py`,
`spec_types_bench.py`, `spec_report.py`, `spec_model.py`, `ngram_sim_types.py`, `spec_trace.py` (unrun). Section 9's data and
scripts are listed in 9.9.

## 9. Draft cost, content-adaptive K and a restricted draft vocabulary (b=1, TP=1)

Same box, card, engine settings and prompts as sections 2-3 (`PROMPT_TREE` = the profile's tree, so the quoted files are the
same), tree = `44fb86a01b` + the patches of this section, csrc unchanged (no rebuild), one run per configuration. Run-to-run
noise measured on repeated identical runs: the two no-draft runs agree within 0.1 % (17.97/17.99, 18.06/18.05, 18.63/18.62,
18.05/18.06, 18.06/18.08, 18.57/18.58), but three K=7 runs of one configuration spread chat +-1.2 %, code_gen +-0.9 %, prose
+-0.9 %, rag +-1.6 %, structured +-3.3 % and code_edit +-5.5 % (9.85-10.99: 6 prompts, a few of them dominate) because the
accepted tokens per step differ between runs (fp16 noise changes the trajectory). Differences below that are not results.
**[INFERENCE]** marks arithmetic or an estimate. Everything the target verifies stays exact; only the draft is approximated,
and only where an option says so.

### 9.1 Results

ms per generated token (tokens/s) and the speed-up over no draft, 160 tokens, b=1, same prompts as sections 3-4 (chat 12,
code_gen 8, code_edit 6, prose 8, structured 6, rag 6, 46 prompts; one run per cell):

| type | no draft | static K=2 (the default schedule) | static K=7 | adaptive K, full-vocabulary draft | **adaptive K + 32768-id draft vocabulary** |
|---|---|---|---|---|---|
| chat | 17.97 (55.6) | 15.23 (65.6) 1.18x | 17.61 (56.8) 1.02x | 15.35 (65.2) 1.17x | **14.30 (69.9) 1.26x** |
| code_gen | 18.06 (55.4) | 14.65 (68.3) 1.23x | 16.18 (61.8) 1.12x | 14.21 (70.4) 1.27x | **13.40 (74.6) 1.35x** |
| code_edit | 18.63 (53.7) | 12.70 (78.7) 1.47x | 10.99 (91.0) 1.70x | 10.80 (92.6) 1.72x | **9.95 (100.5) 1.87x** |
| prose | 18.05 (55.4) | 16.17 (61.8) 1.12x | 19.58 (51.1) 0.92x | 16.67 (60.0) 1.08x | **15.64 (63.9) 1.15x** |
| structured | 18.06 (55.4) | 13.57 (73.7) 1.33x | 13.89 (72.0) 1.30x | 13.11 (76.3) 1.38x | **11.58 (86.4) 1.56x** |
| rag | 18.57 (53.8) | 11.74 (85.2) 1.58x | 8.21 (121.8) 2.26x | 8.13 (123.0) 2.28x | **7.69 (130.1) 2.42x** |
| mean of the 6 types | 18.22 (54.9) | 14.01 (71.4) 1.30x | 14.41 (69.4) 1.26x | 13.04 (76.7) 1.40x | **12.09 (82.7) 1.51x** |

* The last column is `speculator_options={"adaptive_k": {}, "draft_vocab": "vocab3_32768.npy"}` with the shipped estimator
  defaults (`final`); the fourth is `{"adaptive_k": {}}` alone, with the first estimator defaults (decay .93, prior weight 8,
  margin 5 %, no tolerance, no `min_steps`; the policy-only gain, measured before the tuning of 9.7). The static K=7 column is
  the adaptive run with K forced to 7 (two such runs: 9.85-10.99 on code_edit, see the noise above). The profile's own static
  runs (sections 3-4, another session) agree with these within 0.3 ms at K=2 and within 0.6 ms at K=7 (code_edit K=7: 10.70
  there, 9.85-10.99 here).
* **100 tok/s at b=1 is reached on code_edit (100.5) and rag (130.1)**; chat reaches 69.9 (1.26x) and prose 63.9 (1.15x),
  where the draft accepts 0.8-1.0 token per step and a 28-29 ms step cannot be amortized further. The mean over the six
  types is 12.09 ms/token (82.7 tok/s): -33.6 % against no draft, -13.7 % against the K=2 default, -5.2 % against the best
  static K per type of section 4 (12.75 with the full-vocabulary draft, an oracle that knows the type; the policy alone,
  fourth column, is 13.04, +2.3 % above that oracle).
* Greedy output equality with the no-draft output: 9.8.

Per type, the `final` run (the K histogram is over the verified steps; the draft always produced 7 tokens):

| type | mean K | steps at K = 1 / 2 / 3 / 4 / 5 / 6 / 7 (%) | accepted per step | first-position acceptance | tokens per step |
|---|---|---|---|---|---|
| chat | 2.77 | 23 / 17 / 37 / 14 / 5 / 2 / 2 | 1.04 | 0.569 | 2.04 |
| code_gen | 3.21 | 6 / 18 / 48 / 15 / 7 / 3 / 3 | 1.27 | 0.625 | 2.27 |
| code_edit | 4.60 | 0 / 0 / 8 / 47 / 30 / 9 / 7 | 2.67 | 0.837 | 3.67 |
| prose | 2.49 | 30 / 22 / 29 / 11 / 5 / 1 / 1 | 0.80 | 0.515 | 1.80 |
| structured | 3.87 | 1 / 7 / 39 / 31 / 12 / 4 / 7 | 1.78 | 0.701 | 2.78 |
| rag | 5.86 | 0 / 0 / 0 / 16 / 22 / 20 / 41 | 4.21 | 0.893 | 5.21 |

The mean K is 2.5-3.2 on chat / code_gen / prose (the model's best static K is 3), 3.9 on structured (5), 4.6 on code_edit (6)
and 5.9 on rag (7): lower than the static optimum because the policy takes the smallest K within 1 % of the best rate and the
early steps of a request run at the engine prior; inside a request K moves from 1-2 where the draft is rarely right to 6-7
inside copied passages. `K = 0` was never chosen on these six types with the shipped defaults (a first-position acceptance of
0.5 still pays: 24.6 ms for 1.52 tokens at K = 1 against 18.5 ms for 1 token **[INFERENCE: cost table of 9.4]**; the rule
needs the best K to be 8 % slower than no draft). The first defaults without `min_steps` let the engine-wide prior stop
requests that had not been tried: 5 % (chat) and 7 % (prose) of the steps at K = 0 with the full vocabulary, 17 % and 9 % in
the first run with a draft vocabulary (`adk32`), none since.

### 9.2 What the draft step costs

`scripts/draft_micro.py` replays the captured draft graph (context + 8-row block + Markov chain) on the real checkpoint; the
engine's phase timer of 3.3 has the draft at 4.3-4.8 ms per K=2 step depending on the type, the replay 4.47 ms.

| | K=2 full vocab | K=7 full vocab | K=2, 32768 ids | K=7, 32768 ids | K=2, 16384 ids |
|---|---|---|---|---|---|
| graph replay (ms) | 4.470 | 5.079 | 2.374 | 2.649 | 2.148 |
| kernels (busy / idle gaps, ms) | 49 (4.179 / 0.337) | 64 (4.672 / 0.481) | 53 (2.114 / 0.354) | 78 (2.302 / 0.544) | 53 (1.916 / 0.347) |

Where the 4.470 ms (K=2, full vocabulary) go:

| part | ms | share | notes |
|---|---|---|---|
| base LM-head sweep (`gemv_kernel<8,4,false>`, 7 block rows x 200058 ids, 922 MB) | 2.154 | 48 % | 428 GB/s, the stream rate |
| 3 draft layers: GEMVs (context K/V 3 x 28 us; per layer qkv 71-76, o 55-58, gate_up 209-214, down 128-130 us) | 1.48 | 33 % | 250-395 GB/s |
| Markov chain (2 steps: fold GEMVs, `markov_step`, `head_logits` 248 / 62 us, argmax) | 0.42 | 9 % | 0.12 ms per further step at K=7 |
| 14 small kernels (mix/rms, qkv_prep, attention split/combine, add_rmsnorm, silu_mul, gather) | 0.09 | 2 % | |
| gaps between the 49 kernels (graph dispatch: 5 us after a large kernel, 9 us after a small one) | 0.34 | 8 % | |

DRAM bytes of one replay are about 1548 MB **[INFERENCE: weights read once]**, a 3.51 ms floor at 440 GB/s: the full-vocabulary
draft is at 79-85 % of it, so no schedule of the same bytes reaches the 2.5 ms target; only fewer bytes do. Exact reductions
that keep the draft's numbers bit for bit, and what each is worth **[INFERENCE]**:

* the 14 small kernels and their gaps (0.43 ms together at K=2): fusing pairs saves one 5-9 us gap and about 2 us of launch
  work each, at best 0.15 ms; needs csrc (the draft library), not done;
* Markov steps the schedule does not verify: the draft graph runs `scheduled_draft_tokens` steps (7 under the adaptive cap)
  whatever K the step verifies; with a mean K of 3-4 that is 3-4 unused steps, 0.2 ms with 32768 ids, 0.45 ms with the full
  vocabulary. It needs one draft graph per K (the draft graphs are keyed by request count only); not done;
* sampling and the head: the 7 block rows already share one weight read and the Markov head needs the previous token, so
  there is no redundant lm_head work left to skip.

What does shrink the 48 % is reading fewer head rows: the draft vocabulary of 9.5 (-2.1 ms at K=2 with 32768 ids, 2.37 ms in
total, the 2.5 ms target; -2.3 ms with 16384). An int8 copy of the head would save about 1.1 ms at the full vocabulary and
about 0.17 ms on top of the 32768-id list, and needs a new kernel plus an offline-derived weight file: not done.

### 9.3 Cheaper verification: distinct experts per verify step

`scripts/moe_union.py` wraps `inkling2_glue_moe_route` in an eager engine with the adaptive cap 7 (so verify steps of 1-8 rows
occur), 3 prompts per type, 96 tokens, and records the top-8 of every MoE visit with at most 8 rows (128 experts per layer).
Mean distinct experts per visit (slots = 8 T):

| rows T | slots | chat | code_gen | code_edit | prose | structured | rag | independent requests (decode-moe doc 2, b = T) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 8.0 | - | - | - | - | - | 8.0 |
| 3 | 24 | 17.6 | - | - | 15.9 | - | - | 22.6 |
| 4 | 32 | 19.6 | 19.0 | 20.1 | 19.6 | 20.3 | - | 29.2 |
| 5 | 40 | 22.9 | 21.9 | 22.2 | 22.1 | 21.7 | 22.6 | |
| 6 | 48 | 25.2 | 24.7 | 24.5 | 25.2 | 23.7 | 24.9 | |
| 7 | 56 | 26.6 | 27.3 | 25.8 | 27.3 | 23.7 | 28.2 | |
| 8 | 64 | 30.3 | 28.0 | - | 27.9 | 28.5 | 31.3 | |

Adjacent verify rows share 3.0-3.6 of their 8 experts (about 40 %; independent routing would share 0.5). The 22.6 / 29.2 of
the decode-moe doc are for b = T independent requests and overstate a speculative step by a third (T = 3: 15.9-17.6, T = 4:
19-20); the union saturates near 30 at T = 8 (independent routing: 52 **[INFERENCE: 128 x (1 - (1 - 8/128)^8)]**).

What that explains **[INFERENCE: arithmetic on measured numbers]**. An NVFP4 expert is 4.5 MB (35.8 MB per 8) and an MXFP8
one 8.2 MB, so a distinct expert costs 44 x 10.2 us + 4 x 18.6 us = 0.52 ms per forward at the 440.7 GB/s stream rate. Forward
growth over 1 row (profile 3.3: 15.9 -> 23.0 / 26.9 / 32.7 ms at 3 / 5 / 8 rows) against the extra distinct experts: T = 3:
+7.1 ms for 8.7 extra experts (4.5 ms of expert bytes, 63 %); T = 5: +11.0 for 14.3 (7.4 ms, 67 %); T = 8: +16.8 for 21.2
(11.0 ms, 66 %). The other third is about 0.7 ms per extra row (the MoE kernels' distance from the stream floor, 6-13 % in
decode-moe 3, plus 0.2-0.4 ms of dense / glue / attention per row). The fall from 3.55 to 1.94 ms per row at T > 3 is the
union saturating (2.3 new experts per row from T = 5 to 8: 1.2 ms + 0.7 ms = 1.9 ms).

Exact levers, and why none of them removes the penalty:

1. Reading each distinct expert once per visit: done (the two-row expert blocks of `_BLOCK_M_BY_TOKENS_SPEC` are in HEAD); the
   rows' union is the number of different experts they choose.
2. Fewer bytes per expert: the checkpoint's NVFP4 / MXFP8 are the numerics; another format changes the logits.
3. Reusing experts across visits: about 1 GB of other traffic lies between two visits of a layer, 8x the 128 MB Infinity Cache
   (decode-moe 4).
4. Prefetching the next layer's experts: layer l+1's routing is known only after layer l; fetching the previous step's experts
   early reads bytes the step needs only about 40 % of the time, and the stream is the bottleneck, so it adds traffic
   **[INFERENCE]**.
5. The kernel third (about 0.7 ms per row): the MoE kernels' distance from the stream floor (6-13 %, decode-moe 3) and the
   per-row dense / glue work, at most about 0.5 ms per step **[INFERENCE]**: owned by the spec-off work.

Not allowed (changes the logits, so the output): skipping or restricting experts of draft or verify rows.

### 9.4 Content-adaptive K

`AdaptiveSpecTokens` (`vllm/v1/spec_decode/dynamic/adaptive.py`) decides, at every `schedule()`, how many of the proposed drafts
the next step verifies. K is a pure performance knob: verification accepts a draft only if it equals what the target samples at
that position, so a different K cannot change the output beyond the fp16 floor of 9.8, and nothing about the draft or the
rejection sampler is touched.

* **Estimator.** Per request, decayed counts (`decay` 0.90 per step) of the acceptance of the first draft position (`p1`) and of
  the later ones pooled (`p`), shrunk toward the engine-wide estimate (`prior_weight` 6); a new request starts from what the
  engine saw lately (content comes in runs). A per-position estimate is too noisy for the 50-150 steps of a request.
* **Choice.** `K = argmax (1 + p1 + p1 p + ...) / step_ms(K)` with `step_ms(K) = forward(rows) + lm_head and sampling + draft(K)
    * host`, the V620 table of 3.3 (forward at 1/3/5/8 rows = 15.9/23.0/26.9/32.7 ms; draft 4.15 + 0.13 K ms, scaled down by
  the vocabulary share when a draft vocabulary is set); the smallest K within 1 % of the best rate is taken (`k_tolerance`: the
  rate is flat near its optimum and a smaller K costs less when the estimate is too high).`K = 0` only when the best K is slower
  than not drafting by `stop_margin` (8 %) and the request has had `min_steps` (4) verified steps (an engine-wide dip must not
  stop a request that has not been tried: without that rule up to 17 % of the chat steps ran at K = 0, none with it); while K
  is 0, every 32nd step probes one draft.
* **Wiring.** `Inkling2DSparkSpeculatorConfig` accepts `speculator_options={"adaptive_k": true | {...}, "draft_vocab": path}`
  (dict keys: the `StepCostModel` fields for another GPU, and `decay`, `prior_accept`, `prior_weight`, `engine_decay`,
  `engine_weight`, `stop_margin`, `probe_interval`, `min_steps`, `k_tolerance`). The scheduler (`spec_k_policy`) calls `choose`
  instead of the batch-size lookup, `observe` with every verified step (not stale ones under async scheduling) and `forget` on
  free; the batch-size schedule stays the cap (`[[1, 2, 7], [3, 8, 0]]`: up to 7 below 3 requests, none above). Nothing
  changes without the option: the default path is the shipped one.
* **Graphs.** With the option, `get_decode_cudagraph_query_lens` returns every `K' + 1 <= cap + 1` for each batch size: at
  `max_num_seqs 2` the boot captures 16 FULL graphs (query lengths 1-8 x 2 batch sizes) + 14 PIECEWISE sizes + 2 draft graphs:
  1.38 GiB of graphs (static K=2: 0.87 GiB) and a KV cache of 23,539 tokens instead of 25,831 (-8.9 %); no fault at any K
  (the runs of 9.1 used every K from 1 to 7).
* **`--adaptive` (`enable_adaptive_verification`)** is the engine's other mechanism (it trims verified rows inside the attention
  backends and needs a per-position draft distribution the Markov head does not keep). It is now refused at config time with a
  pointer to `adaptive_k` (`ValueError`) instead of failing the boot with a bare `AssertionError` (`dspark.py:970`).

Estimator defaults A/B at the same draft vocabulary (`adk32v3` = first defaults, `final` = shipped): chat 14.29 -> 14.30,
code_gen 13.25 -> 13.40, code_edit 10.25 -> 9.95, prose 15.72 -> 15.64, structured 12.26 -> 11.58, rag 7.89 -> 7.69, mean 12.28
-> 12.09 (-1.5 %), the replay of 9.7 had predicted -0.8 %. Both runs saw the prompts the defaults were tuned on (9.7), so this
is not a held-out test of them.

Several requests (`b4`, `max_num_seqs 8`, `[[1, 2, 7], [3, 8, 0]]`, chat and code_edit prompts, 64 tokens, batches of 1, 2 and 4;
functional check, one run): b=1 12.68 / 10.33 ms/token (chat / code_edit), b=2 17.94 / 14.49 ms/token/request = 111.5 / 138.0
tok/s in total with 319 / 121 drafts verified and K varying per step, b=4 28.52 / 28.68 ms/token/request (140.3 / 139.4 tok/s in
total, no drafts: the cap is 0 from 3 requests on). No fault, no engine error; the per-request estimator and the batch cap
interact as designed. Throughput beyond b=1 was not tuned.

Content a draft cannot guess (stress tests, 5 prompts x 128 tokens each, one run per cell, ms/token):

| content | no draft | adaptive K, full vocabulary | adaptive K + 32768-id list |
|---|---|---|---|
| "random-looking" hexadecimal identifiers, one per line | 17.95 | - | 12.39 (accepted 1.55 per step, first position 0.64, mean K 3.92) |
| 150-word essays in Japanese, Chinese, Russian, Korean, Arabic | 17.91 | 15.44 (1.13, 0.58, mean K 2.69; 13 % of the steps at K = 0) | 14.73 (0.98, 0.54, mean K 2.74) |

The identifiers are not random to a greedy model (its output repeats patterns), so that case does not stress the policy; the
non-English essays stress the vocabulary (a list ranked on English, code and chat text) and the policy together: the draft
still pays (-18 % against no draft with the list, -14 % without it), K falls to 2.7 on average and 13 % of the full-vocabulary
steps run without a draft. A natural prompt whose acceptance is near zero was not found; the i.i.d. bound of 9.7 stays an
estimate.

### 9.5 Restricted draft vocabulary (opt-in)

`speculator_options={"draft_vocab": "ids.npy"}`: the greedy draft ranks only the listed token ids (the LM-head and folded-head
rows are gathered once at load, 168 MB for 32768 ids; TP=1, greedy only). Ids are strictly ascending so the draft's
lowest-id tie rule is the full vocabulary's. A token outside the list is never drafted: the cost is acceptance only, the
target still samples from its whole vocabulary. The file is made offline by `tools/rdna/inkling2/draft_vocab.py` (token
frequency over code, prose and chat you provide, equal group weights; the tokenizer's special tokens are always kept: the chat
template's structure tokens open every reply and no corpus contains them). Nothing is quantized or derived at load in the
default path; without the option the draft is untouched.

Lists made for this section (`vocab3_{16384,32768,49152}.npy`): ranked on 13.1M tokens of python (site-packages), 0.84M of prose
(docs) and 11.0M of chat / tool / agentic / research text (the calibration corpora of the quantization runs), equal group
weights, plus the tokenizer's special tokens; held-out python coverage 97.75 / 99.56 / 99.88 %. Coverage on the target's own
greedy tokens (`ids_k0.json`: 7,360 tokens of the six prompt sets, none of it in the ranking corpora); a position whose target
token is outside the list cannot be accepted, so `1 - coverage` bounds the loss per position:

| type | 16384 ids | 32768 ids | 49152 ids |
|---|---|---|---|
| chat | 89.8 % | 95.9 % | 97.9 % |
| code_gen | 93.6 % | 98.2 % | 99.4 % |
| code_edit | 99.6 % | 99.9 % | 100.0 % |
| prose | 89.5 % | 95.3 % | 98.3 % |
| structured | 94.2 % | 98.4 % | 99.8 % |
| rag | 97.4 % | 99.9 % | 100.0 % |
| all | 93.3 % | 97.6 % | 99.0 % |

What the first lists taught: a list ranked on code and prose alone (`vocab_32768`) covers 98.6-99.9 % of held-out python but only
94.5 % of the real outputs (chat 93.0, prose 88.1; `vocab2_32768`, with some chat text: 96.1 %), because no corpus held the chat
template's structure tokens (`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`, `<|end_message|>` open and close
every reply) or enough English chat prose; the tokenizer's special tokens are now always kept and the chat calibration text is in
the ranking. What is left are names and foreign words of the prompts, which no static list holds.

End to end with adaptive K (ms/token, accepted tokens per step; `adk` = full vocabulary, `adk32` = the code + prose list,
`adk32v3` = the list above):

| type | full vocab: ms, acc/step | 32768 code + prose list | 32768 chat-calibrated list (first defaults) |
|---|---|---|---|
| chat | 15.35, 1.166 | 14.67, 1.135 | 14.29, 1.108 |
| code_gen | 14.21, 1.429 | 13.28, 1.387 | 13.25, 1.365 |
| code_edit | 10.80, 2.737 | 10.03, 2.755 | 10.25, 2.603 |
| prose | 16.67, 0.935 | 16.16, 0.818 | 15.72, 0.851 |
| structured | 13.11, 1.712 | 12.29, 1.607 | 12.26, 1.653 |
| rag | 8.13, 4.337 | 7.84, 4.121 | 7.89, 4.099 |
| mean of 6 | 13.04 | 12.38 | 12.28 |

The draft is 2.1-2.4 ms cheaper per step (7 % of a 30 ms chat step) and the ms/token fall by 3-7 % (-6 % on the mean of the
six types; code_edit and structured are inside the run-to-run noise). `acc/step` depends on the K the policy picks (it changed
between these runs), so the acceptance cost is read off the first draft position: with the 32768-id list `p1` is chat 0.579,
code_gen 0.638, code_edit 0.824, prose 0.525, structured 0.669, rag 0.885, against 0.583 / 0.621 / 0.849 / 0.533 / 0.663 /
0.910 (mean of three full-vocabulary K=7 runs: the profile's and the two forced-K=7 trace runs): within +-3 %, the size of the
run-to-run spread. The 1.6-4.7 % of the target's tokens outside the list in chat, prose, code_gen and structured are the rare
tokens (names, foreign words) the draft would seldom have guessed. 49152 ids would cost about 0.25 ms more per step for +1.4 %
coverage **[INFERENCE: not run]**.

### 9.6 Prompt lookup and a sibling candidate

Prompt lookup (n-gram) alone does not beat DSpark (section 1, item 5: engine `ngram` fixed width 14.5-26.5 ms/token,
variable-width replay 11.3-17.4). The joint question is whether the two combine: verify a long lookup proposal on its own
(exact width, no draft pass: `forward(1 + len) + 2.23 + 0.4`) and fall back to DSpark otherwise. `sim/hybrid_sim.py` replays
that on the real streams (`ids_k0.json` for the lookup, n = 2..3, up to 7 tokens, most recent occurrence; the DSpark outcome at
a stream position is the accepted run of the step of the forced K=7 trace (`trace7_events.json`) that covers it, extended to
its end), full-vocabulary draft cost 4.15 + 0.13 K, forward table of 3.3, each type at its best static K:

| type | DSpark alone (best static K) | hybrid, lookup of at least 4 tokens | steps that verify a lookup |
|---|---|---|---|
| chat | 15.29 (K 3) | 15.12 | 10 % |
| code_gen | 14.49 (K 3) | 14.91 | 12 % |
| code_edit | 9.10 (K 7) | 10.10 | 27 % |
| prose | 17.06 (K 3) | 17.38 | 9 % |
| structured | 13.22 (K 3) | 13.41 | 16 % |
| rag | 7.62 (K 7) | 8.29 | 53 % |

(L = 3, 5, 6 give the same within 0.05 ms.) The simple rule loses on five of six types (-1.1 % on chat, +1.4 to +11 % elsewhere):
where the lookup is long the DSpark draft is also long and accepts more tokens per step than the copy does (code_edit and rag
accept 3.0-4.4 per step at K=7). The environment is approximate (a run-extension model of DSpark off its own trajectory), so
this is a screen, not a measurement; a combination that keeps DSpark's draft and appends the lookup's tokens after a
disagreement would need a variable-width verify with the lookup on the host before the step (not available under async
scheduling) and was not built.

A sibling candidate (the draft's second choice for position 1 as an extra verify row): the row is the 4th at K=2 and costs
1.94 ms (7 % of a 28 ms step), so it must add at least 7 % of the 2.04 tokens a chat step commits, 0.14 tokens. Position 1 is
rejected in 43 % of the chat steps, so the second choice must be right in more than a third of those. That was not measured;
it is at best break-even before the cost of a second draft chain **[INFERENCE]**, and was not built.

### 9.7 Policy tuning on real traces

The trace is a forced K=7 run (`m2.sh trace7`: every step verifies 7, `observe` logs `(type, request, 7, accepted)`): 3,015 steps
of 46 + 6 warm-up requests, the acceptance bursts of real text. `sim/replay.py` replays a request as its step outcomes in
order, one per step, `accepted = min(outcome, K)`, wrapping until 160 tokens are committed (the burstiness is kept, the token
alignment is not), with a policy lag of two steps (async scheduling) and the cost table of 9.4 with the 32768-id draft
(2.19 + 0.063 K ms). Mean over the six types (ms/token; `sim/replay_pick.py`, 5 shuffles):

| policy | mean | chat | code_gen | code_edit | prose | structured | rag |
|---|---|---|---|---|---|---|---|
| no draft | 18.13 | 18.13 | 18.13 | 18.13 | 18.13 | 18.13 | 18.13 |
| static K=2 | 12.86 | 14.02 | 13.78 | 10.98 | 15.24 | 12.87 | 10.25 |
| static K=3 | 12.05 | 13.62 | 13.03 | 9.81 | 15.22 | 11.75 | 8.89 |
| static K=4 | 11.99 | 13.99 | 13.08 | 9.22 | 15.75 | 11.69 | 8.20 |
| static K=7 | 12.99 | 15.99 | 14.90 | 8.79 | 18.35 | 12.89 | 7.04 |
| best static K per type (oracle) | 11.56 | 13.62 | 13.03 | 8.79 | 15.22 | 11.69 | 7.04 |
| adaptive, first defaults (decay .93, prior 8, margin .05) | 11.51 | 13.55 | 12.91 | 8.93 | 15.01 | 11.61 | 7.04 |
| adaptive, shipped defaults (decay .90, prior 6, margin .08, tolerance .01; probe 16 here, 32 shipped) | 11.41 | 13.36 | 12.91 | 8.91 | 14.87 | 11.41 | 7.01 |

The 72-point grid of decay x prior weight x margin x probe interval x engine weight spans 11.41-11.62 (the worst point is 1.8 %
above the best) and the larger grid that adds the tolerance and more decays 11.40-11.71 (+2.7 %), so the policy is not
sensitive; faster adaptation (decay .85-.90, prior weight 4-6) is better on real bursty text than on the i.i.d. simulation
(decay .93 was best there) and the tolerance toward a smaller K takes the same gain more stably. The i.i.d. low-acceptance
check (acceptance 0.05-0.55 per position, 160 tokens): at most +2.3 % ms/token against no draft (probes and the forced first
steps), -10 % at 0.55. The replay is an estimate (6 types, 46 requests, wrapped step sequences): the e2e rows of 9.1 are the
measurement.

### 9.8 Equality with the no-draft output

Verification is exact by construction (a draft is accepted only if it equals the target's greedy token at its row), and
neither K nor the draft vocabulary touches the target's weights or its sampler: they change how many rows a step verifies
and which token the draft proposes. What can differ is numerical: the fp16 atomics of the expert GEMMs make two no-draft runs
diverge (the floor of `inkling2-dspark.md`, "Equality with no draft"), and a step of another row count uses other tile shapes.
`ids_compare.py` / `eq_table.py` compare the 160 greedy ids of all 46 prompts; with `--save-logprobs` the runs also record the
top-2 logprobs per token, so the margin at the **first** divergence (the target's own score gap between the two tokens, the
larger of the two runs' gaps, nats) shows whether a divergence is a near-tie:

| pair | identical prompts | first divergence index: median / min / max | margin at it: median / p90 / max nats; share <= 0.1 / <= 0.25 |
|---|---|---|---|
| no draft vs no draft (`k0` vs `k0b`) | 11/46 | 66 / 7 / 156 | - |
| no draft vs no draft (`k0` vs `k0lp`) | 12/46 | 70 / 18 / 139 | - |
| **no draft vs no draft (`k0lp` vs `k0lp2`), the floor** | 15/46 | 54 / 7 / 130 | 0.066 / 0.465 / 0.794; 65 % / 84 % |
| no draft vs static K=2 | 9/46 | 75 / 7 / 159 | - |
| no draft vs adaptive K, full vocabulary (`adk`) | 15/46 | 52 / 8 / 99 | - |
| `k0lp` vs adaptive K + draft vocabulary, first defaults | 13/46 | 60 / 8 / 139 | 0.099 / 0.296 / 0.813; 50 % / 78 % (+1) |
| `k0lp2` vs adaptive K + draft vocabulary, first defaults | 14/46 | 74 / 7 / 139 | 0.092 / 0.264 / 0.544; 56 % / 84 % |
| **`k0lp` vs `final` (shipped)** | 15/46 | 60 / 15 / 144 | 0.087 / 0.172 / 0.728; 62 % / 90 % (+2) |
| **`k0lp2` vs `final` (shipped)** | 12/46 | 52 / 7 / 152 | 0.092 / 0.240 / 0.699; 67 % / 88 % (+1) |

The speculative runs against no draft show what two no-draft runs show against each other: 9-15 of 46 prompts identical in
both families, the first divergence at about token 50-75 (earliest 7-15), and at the divergence the target's own gap between
the two tokens is a near-tie (median 0.07-0.10 nats, at most 0.81 against 0.79 for the floor). Before the first divergence the
46 outputs are token for token equal. A "(+n)" is a divergence where the other run's token is outside the recorded top 3 of
this one (`logprobs=2` records two tokens and the sampled one): in all of these cases the recorded candidates of both runs lie
within 0.11 nats of each other (a three-way tie: chat prompt 10 index 144, prose 1 index 144, code_gen 7 index 78, code_gen 6
index 74). What is not shown: a teacher-forced NLL comparison (not applicable to a lossless method), and a
rejection-sampling (temperature > 0) check: the adaptive path was run greedy only.

### 9.9 Commands and files

Box (`par1-cs25`, HIP 3 only through `gpu_sub.sh`), work dir `$M = ~/w4a8_runs/inkling2-work/Inkling2SpecProto`, tree
`$M/tree` (= `44fb86a01b` python + the patches; `_rocm_C` and the JIT dir of `Inkling2PrefillProfile`, no csrc change),
`PROMPT_TREE = Inkling2PrefillProfile/tree2` (the prompts quote its files):

```bash
M=~/w4a8_runs/inkling2-work/Inkling2SpecProto; P=~/w4a8_runs/inkling2-work/Inkling2PrefillProfile
export TREE=$M/tree PROMPT_TREE=$P/tree2 I2_JIT_DIR=$P/jit ROCM_HOME=/opt/rocm/core-7.14
G="bash ~/w4a8_runs/inkling2-work/Lead/gpu_sub.sh 3 900 $M/scripts/spec_types_bench_v3.py --tokens 160"
$G --spec-k 0 --json $M/out/k0.json --save-ids $M/out/ids_k0.json                     # no draft (also k0b, k0lp, k0lp2)
$G --spec-k 2 --json $M/out/k2.json                                                    # shipped default schedule
$G --spec-k 7 --adaptive-k --json $M/out/adk.json                                      # adaptive K, full-vocabulary draft
$G --spec-k 7 --adaptive-k --draft-vocab $M/out/vocab3_32768.npy --json $M/out/final.json \
   --save-ids $M/out/ids_final.json --save-logprobs                                    # adaptive K + 32768-id draft
python $M/scripts/draft_micro.py --markov-steps 2 --context 8 --iters 200 --profile [--draft-vocab ids.npy]   # draft step
python $M/scripts/moe_union.py --per-type 3 --tokens 96                                # experts per visit, eager
python tools/rdna/inkling2/draft_vocab.py --ckpt ~/hfcache/inkling2-mini --size 16384 32768 49152 \
   --out vocab3_{size}.npy --code 'site-packages/**/*.py' --prose 'docs/**/*.md' --chat calib.jsonl --heldout 'repo/**/*.py'
```

Serving: `RECIPE=inkling2-mini-tp1-dspark-adaptive` (`tools/rdna/recipes/`; the header has the one command that makes the ids
file). Its `SPEC_CONFIG` is `{"method":"dspark","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,7]],
"speculator_options":{"adaptive_k":{},"draft_vocab":"/path/ids.npy"}}`; with more slots keep the tail `[3,8,0]` of the schedule,
and drop `draft_vocab` for the exact-draft default. Lossless check: `tools/rdna/inkling2/dspark_nll.py` forces a reference text
through the decode path with and without the draft and compares the logprob of every forced token against the run-to-run floor
(`docs/rdna2/inkling2-dspark.md`, "Run DSpark").

Raw data (box `$M/out`, `$M/logs`; local copies `Inkling2SpecProto/data/`): `{k0,k0b,k2,adk,adk32,adk32v3,final,hex0,hexad,lang0,
langad,langv,b4}.json` and logs, `ids_*.json` (greedy ids, `top` = best logprobs for the `lp` runs), `trace7_events.json`,
`moe_union.json`, `dm_*.json` (draft microbench), `vocab3_*.npy`, `replay*.log`, `hybrid.log`. Scratch tools: `scripts/`
(`spec_types_bench_v3.py`, `report_table.py`, `ids_compare.py`, `eq_table.py`, `ids_margins.py`, `vocab_eval.py`,
`uncovered.py`) and `sim/` (`adaptive_sim.py`, `prod_sim.py`, `replay.py`, `replay_variants.py`, `replay_pick.py`,
`hybrid_sim.py`).

### 9.10 Not done / open risks

* **Statistics.** 6-12 prompts per type, one run per configuration; identical runs differ by 0.1 % (no draft) to +-5 %
  (code_edit K=7), so per-type differences under 3-5 % are not results; the mean of the six types is steadier (about +-1 %).
  The estimator defaults were tuned on a trace of these same prompts (9.7); 33 held-out prompts (5-6 of each kind, none used
  for the tuning or for ranking the vocabulary) were run later through the served recipes: their means are 4 % (K = 2) and
  7 % (adaptive + vocabulary) slower than the tuned prompts', and every kind is still faster than no draft
  (`inkling2-dspark.md`, "Run DSpark").
* **Draft vocabulary.** An opt-in artifact: ids are the tokenizer's, ranked on code, prose and chat text that are not the
  benchmark's outputs, 95-96 % coverage on chat / prose at 32768 ids, so text unlike the corpora (other languages, rare names)
  would lose more acceptance. First-position acceptance stayed within +-3 % here. TP=1 and greedy only (anything else raises
  at load). No quantization at load, no change to the default path.
* **Adaptive K is opt-in.** Validated at b=1 for speed and at b=2 / 4 only functionally (64 tokens, two types). The step cost
  table is the V620's (other GPUs: override through the `adaptive_k` dict, or the policy picks by a wrong curve; it is lossless
  either way). Memory: all widths are captured, +0.51 GiB of graphs and -8.9 % KV cache at `max_num_seqs 2` and cap 7
  (25,831 -> 23,539 tokens; 22,841 with the draft vocabulary's 0.15 GiB of gathered rows).
* **Low-acceptance content.** The i.i.d. replay bounds the loss against no draft at +2.3 % ms/token (probes and forced first
  steps) for acceptance 0.05-0.35; the stress runs of 9.4 (identifiers, non-English essays) are 14-31 % faster than no draft.
* **`enable_adaptive_verification`** (the engine's in-backend trimming) remains unsupported for this draft; it is refused at
  config time instead of asserting at boot.
* **Not done**, with the estimated value **[INFERENCE]**: a draft graph per verified K (0.2 ms at 32768 ids, 0.45 ms full
  vocabulary per step); fusing the 14 small draft kernels (0.15 ms, csrc); an int8 LM-head copy for the draft (-1.1 ms at the
  full vocabulary, -0.17 ms with the pruned list; offline file + kernel); the lookup hybrid and a sibling row (9.6: no gain
  expected); a draft-confidence driven K (needs device-side trimming of verify rows). Verify-side: nothing exact is left except
  the kernel gap (9.3).
* **Tests.** 12 CPU tests of the policy, the config / refusal / vocabulary tests of `test_dspark_model.py` and the GPU
  vocabulary tests passed (HIP 3); 13 other CPU tests (`tests/compile/test_config.py`: 12; `test_serving_path_cpu.py::
  test_moe_block_rows_policy[nvfp4]`) fail identically on the pristine base tree on the box (`RuntimeError`, environment), not
  related to these patches.
