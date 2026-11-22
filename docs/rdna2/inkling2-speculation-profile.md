# Inkling-2-Mini b=1 speculation profile on gfx1030 (V620), TP=1

Measured on `par1-cs25`, HIP device 3, real checkpoint (`~/hfcache/inkling2-mini`), tree = worktree HEAD
`f459e45e0d` (csrc and library = `e505d4d436`: DecodeGlue's SwiGLU->w2 fusion, two-row verify blocks, `moe_fast` dequant,
`VLLM_INKLING2_FUSE_GLUE` on). Engine: `FULL_AND_PIECEWISE`, `max_num_seqs 2`, prefix caching off, one request at a time (b=1),
160 generated tokens per prompt, temperature 0, `ignore_eos`. Draft = the shipped HIP DSpark draft (3 layers, 8-row block,
Markov head). Scratch tools only, no repo file was changed. **[INFERENCE]** marks arithmetic or an estimate, not a measurement.
Sample sizes are small (6-12 prompts per type, one run per configuration; run-to-run noise not measured, expect +-3 %).

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
4. **A content-adaptive K is the best exact combination**: equal mix of the six types 13.93 -> 12.74 ms/token (-8.6 %),
   chat-heavy mix 14.71 -> 14.24 (-3.2 %). It is not available today: the HIP DSpark asserts
   `not self.enable_adaptive_verification` (`vllm/models/inkling2_loop/amd/dspark.py:970`, measured: boot fails), and K is keyed
   by batch size only (`num_speculative_tokens_per_batch_size`).
5. **Prompt lookup (n-gram) does not beat the DSpark draft.** The engine's `ngram` method verifies a fixed K+1 rows every step:
   measured 14.5-26.5 ms/token at K=6 (worse than no draft except rag, 14.49 vs 18.59). A variable-width replay of the exact
   greedy streams (rows = 1 + proposal length, no draft pass) gives 11.3-17.4 ms/token: equal to DSpark K=2 on rag (11.3-11.5),
   worse everywhere else. The hybrid (lookup + DSpark) is **not measured**; `spec_trace.py` (joint per-step trace) is written
   but was not run.
6. **A cheaper draft is worth 3-6 % on chat** (draft 4.4-5.0 ms/step = 15 % of the step): int8 draft head -1.05 ms/step,
   with W4 draft layers about -1.85, vocab-pruned head -1.75 (needs a calibration corpus: opt-in) **[INFERENCE: acceptance
   under the cheaper draft is unmeasured]**. It only changes acceptance, never the output.
7. **<= 10 ms/token at b=1 is reachable only on copy-heavy content** (rag K=7 8.0 measured, 7.6 model; code_edit 10.2-10.7).
   For chat the best exact configuration is about 14.3-15.2 ms/token (66-70 tok/s).

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
| 1 | per-request adaptive K (K=3-4 on chat-like text, up to 6-7 when the recent acceptance is high) | measured K=2/4/7 per type, model validated within 0.1-0.5 ms; rag 11.49 -> 8.01 (-30 %), code_edit 12.68 -> 10.70 (-16 %), structured 13.57 -> 12.46 (-8 %), code_gen 14.36 -> 13.87 (-3 %); chat and prose stay at K=2 | measured mixes: equal -8.5 %, chat-heavy -3.5 %; copy-heavy content up to 1.4x over K=2 | needs per-request K in the scheduler and verify graphs per width; engine adaptive verification is unsupported for this draft (assert, measured). **Not built** |
| 2 | cheaper draft: int8 head (RTN, calibration-free), W4 draft layers | draft = 4.3-5.0 ms of a 29 ms step; head sweep 2.15 ms (922 MB) [INFERENCE from Phase 1 stream rate] | chat -0.5 ms/token (3.3 %), both -0.9 (5.9 %) [INFERENCE: acceptance unmeasured] | draft-only (output unchanged); needs an acceptance A/B (>= -1 % relative) |
| 3 | verify-row MoE efficiency (DecodeGlue) | 3-row real-routing MoE 208 us/visit vs 164 us stream floor (Phase 1) | <= 1.3 ms/step = 0.65 ms/token on chat (4 %) | owned by DecodeGlue |
| 4 | prompt lookup / hybrid | engine n-gram fixed width 14.5-26.5 measured; replay 11.3-17.4, no better than DSpark K=2 | none standalone; hybrid headroom unmeasured | needs variable-width verification; `spec_trace.py` not run |
| 5 | skip experts on draft rows / restrict routing | changes the verify logits | not exact: rejected | - |

What breaks the expert-growth penalty (task item 2): nothing exact removes it, each extra row needs its own top-8. The
measured marginal cost falls from 3.55 to 1.94 ms per row after row 3, so the penalty is paid once; it is the *acceptance
curve* that decides how many rows pay, which is why lever 1 beats every kernel lever on copy-heavy text. Two-row expert blocks
(`_BLOCK_M_BY_TOKENS_SPEC`) are already in HEAD.

## 6. Design of lever 1 (not built)

* Controller (host): EMA of the accepted tokens and of a2 over the last about 8 steps of the request (committed counts are
  known one step late under async scheduling, which is fine for an EMA). Map: a2 < 0.45 -> K=3; 0.45-0.6 -> K=5; 0.6-0.8 ->
  K=6; > 0.8 -> K=7 (thresholds read from section 3.2, **[INFERENCE]** from six content types).
* The draft already produces 7 tokens per step, so K can change per step without recapturing the draft; draft cost rises only
  4.43 -> 5.04 ms. The verify side needs a graph per width 1+K at b=1 (7 FULL graphs; memory small **[INFERENCE]**).
* Files **[INFERENCE: not read in depth]**: scheduler K selection (today keyed by batch size, `vllm/config/speculative.py`
  `num_speculative_tokens_per_batch_size`), the runner's draft-token trim, `inkling2_loop/amd/dspark.py` (the Markov cap).

## 7. Not done / open risks

1. The prototype was not built (request budget of the run reached). The measured K=2/K=7 per-type results are the e2e
   evidence for the policy bounds; the controller itself is untested.
2. Greedy equality against the no-draft output was not re-run in this phase (spec verification is exact by construction;
   the earlier floor in `inkling2-dspark.md` is the reference).
3. Small samples and one run per configuration; copy-heavy types depend on the prompt design (quotes of repo files).
4. The joint DSpark/lookup trace (`spec_trace.py`) and the cheaper-draft acceptance were not measured.

## 8. Files and raw data

Box `$P/out/spec/{k0,k2,k7,ng6,k4}.json`, `ids_k0.json`; logs `$P/logs/spec_{k0,k2,k7,ad7,ng6,k4}.log` (phase timer lines after
`[PHASE-FLUSH after <type>]`); local copies `Inkling2PrefillProfile/data/spec/`. Scratch tools `spec_prompts.py`,
`spec_types_bench.py`, `spec_report.py`, `spec_model.py`, `ngram_sim_types.py`, `spec_trace.py` (unrun).
