# Concurrency-probe FAILs on production Flash-Next MTP2: noise, not corruption (2026-11-12)

Question: boot B1 of the contrib-contract smoke (`flashnext-mtp2`, tree
`1136bb952b`, default MoE epilogue) printed two FAILs from
`tools/rdna/port_v031/concurrency_probe.py`, `FAIL together 0: token 6/64 not
in solo top-5` and `FAIL churn (3,1) prompt 1: token 25/64 margin 2.125`. Boot
A1 of the same tree passed, and the deterministic boots passed 4 of 4. Is a
production boot supposed to pass the probe, and is a FAIL a serving bug?

## Summary

- **Verdict: probe misclassification, no serving bug found.** The default
  MoE epilogue is not run-to-run deterministic (packed-fp16 CAS atomics,
  [determinism-2026-11-04.md](determinism-2026-11-04.md)), so two identical
  requests already differ in logprobs and sometimes in a greedy token. A boot
  is supposed to pass the probe. The probe judged noise as corruption by three
  rules: it counted flips after the solo run's stop token, it failed any token
  outside the solo top-5, and it failed on the solo run's margin alone.
- **The logprobs are valid.** The completion path under MTP2 returns the
  target model's logprobs at the right position: the sampled token is the
  argmax of the returned top-20 at all 512 positions of the validity run, and
  of the saved top-5 at all 646,511 positions of the saved Flash-Next probe
  outputs (541,935 on MTP2); the difference from a teacher-forced prefill of
  the same tokens is the difference between two prefills (table below). MTP
  off gives the same numbers.
- **Failure rate of the old probe on production boots:** 1 failing run of 20
  on MTP2 (5 %) and 0 of 20 on MTP off (emulated from the saved outputs, cut
  to the top-5 the old probe saw); 15 of 206 saved noisy Flash-Next runs
  (7.3 %). The fixed probe: 0 of 20 on MTP2, 0 of 20 on MTP off, 0 of 20 on MTP2 with the deterministic epilogue.
- **Teacher-forced margins of every FAIL** (39 rows from 22 saved runs, on a
  clean prefill of the solo prefix): 16 rows are flips after the solo run's
  stop token (median margin 1.38, min -1.00, max 3.74, and the same prefix
  re-run gives margins up to 3.1 nats apart); 3 are flips inside the answer
  (0.25 to 0.82); 7 are the bit-identical `Lisbon` arms (0.26 to 0.97); 13 are
  the known-bad two-lane tree (4.3 to 17.1). B1's two FAILs are 0.25 (a flat
  distribution) and a post-stop flip (1.58 to 2.64 across three prefills).
- **Fix** (branch `fix/concurrency-probe-triage`): top-20 logprobs; on a noisy
  server a flip fails only if both runs are sure of their own token; flips
  after the solo stop token are reported, not judged; a bit-identical server
  keeps the strict solo-margin rule. Re-judged offline, 14 of the 28 archived
  runs that failed under the old rule still fail: the known-bad two-lane tree
  (2), the deterministic Flash-Next arms (7) and deterministic runs of other
  models (5). The known-bad two-lane tree still fails on a live boot,
  5 of 5 runs FAIL (3 or 4 failing checks each) on its head `twolane-B-adffce5765`, while its base `twolane-A-3896928d18` passes 5 of 5.

## What B1 printed, and what it was

Both rows are in `~/w4a8_runs/contrib-contract/smoke3-ab-20261112-000620/pr-receipt-run/boots/B1/concurrency.{txt,json}`.

| row | what the probe said | what the numbers are |
|---|---|---|
| `together 0`, token 6 | context " Paris. Paris is a city", solo `.`, other run `of`: "not in solo top-5" | The solo top-5 is `.` at -1.62, a second token at -1.65, then -2.21, -2.29 and -2.51: a flat distribution, and `of` (-1.82 in the other run) is just below the cut. A teacher-forced prefill of the same prefix gives margins 0.08, 0.26 and 0.41 for `.` over `of`. The other run's text is fluent and factual (" of great historical and cultural significance, known for ... the Eiffel Tower"); the probe's own continuation check passed it (mean lp -0.35 vs -0.15, drop limit 1.5). |
| `churn (3,1)`, token 25 | margin 2.125, solo `<\|im_start\|>`, other run `user` | The solo haiku has `<\|endoftext\|>` at token 23 and a normal request for this prompt stops there (checked on a production boot: `finish_reason` stop). Token 25 is in the continuation `--ignore-eos` forces after it, which serving never shows. Teacher-forced margins of three prefills of the same prefix: 1.58, 2.64 and 1.97, so identical inputs disagree by a nat. The other run was on a near-tie (0.24). |

Re-judged from the saved B1 output, the fixed probe passes the run: the two
solo runs differ (largest logprob difference 2.81 nats, so the server is
noisy), `together 0` is soft (margins solo >0.89, other >0.59, both bounded by
the 5th entry of the saved top-5) and `churn (3,1)` is soft (2.125 solo, 0.242
other).

## Are the returned logprobs the target model's at the right position?

The V2 model runner (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`)
verifies the draft tokens with one target forward and takes the top-k and the
sampled token's logprob from each target logits row
(`vllm/v1/worker/gpu/sample/logprob.py`, `compute_topk_scores`). A shifted
position would show as a wrong argmax and a low top-5 overlap, so the check
compares a completion with a teacher-forced prefill of the same tokens
(`prompt_logprobs`, same server) and two prefills with each other, 8 prompts x
64 tokens, tree `1136bb952b`, production boots:

| measure | MTP2 (`flashnext-mtp2`) | MTP off (`flashnext-mtp0`) | MTP2, `VLLM_RDNA2_MOE_DETERMINISTIC=1` |
|---|---:|---:|---:|
| positions | 512 | 512 | 512 |
| sampled token is not the completion's top-1 | 0 | 0 | 0 |
| sampled token is not the prefill's top-1 | 6 | 6 | 4 |
| top-5 overlap, completion vs prefill (mean of 5) | 4.84 | 4.83 | 4.81 |
| sampled token's \|dlp\|, completion vs prefill: median / p99 / max | 0.0017 / 0.19 / 1.08 | 0.0012 / 0.20 / 0.68 | 0.0007 / 0.13 / 1.22 |
| sampled token's \|dlp\|, prefill vs prefill: median / p99 / max | 0.0018 / 0.18 / 0.34 | 0.0013 / 0.16 / 0.52 | 0 / 0 / 0 |
| shared top-k token's \|dlp\|, completion vs prefill: median / p99 / p99.9 / max | 0.070 / 1.10 / 1.84 / 2.42 | 0.078 / 1.35 / 3.15 / 4.52 | 0.070 / 1.02 / 2.39 / 3.99 |
| shared top-k token's \|dlp\|, prefill vs prefill: median / p99 / p99.9 / max | 0.078 / 0.99 / 1.73 / 2.34 | 0.078 / 1.21 / 2.47 / 3.46 | 0 / 0 / 0 / 0 |
| shared top-k token's \|dlp\|, completion vs completion: median / p99 / p99.9 / max | 0.070 / 0.99 / 2.69 / 4.40 | 0.069 / 0.72 / 1.67 / 1.92 | 0 / 0 / 0 / 0 |
| log-odds of the sampled token over another shared top-k token, prefill vs prefill: p99 / max | 1.04 / 2.48 | 1.28 / 3.46 | 0 / 0 |
| same, completion vs prefill: p99 / max | 1.18 / 2.45 | 1.38 / 4.52 | 1.11 / 3.99 |

The prefill's argmax misses (4 to 6 of 512) are near-ties. The completion path
and the prefill path differ by what two prefills differ by on the default
epilogue, and MTP changes nothing: the MTP-off boot looks the same. With the
deterministic epilogue two prefills and two completions are bit-identical, and
a completion still differs from a prefill by the same amount (p99 of the shared
top-k tokens 1.0 nat, max 4.0). [INFERENCE] The decode/verify step and the
prefill run different shapes, which round differently. That is the size of
difference a solo run and a batched run can have, and a top-5 cut that
excludes a token at -1.8 next to a top entry at -1.6 is not evidence of
anything.

## How noisy is a production boot?

- 20 production runs of the fixed probe on MTP2: in all 20 the two solo runs
  of the same prompts differ in logprobs (largest shared-token difference per
  run: median 3.0 nats, range 1.4 to 5.8). 265 of 560 concurrent outputs
  diverge from their solo output before token 64. MTP off: 20 of 20 noisy
  (median 3.3, range 1.6 to 5.0), 239 of 560 diverge.
- Two sequential, identical requests (no concurrency) flip a greedy token
  with margin above 1 nat in 5 of 602 saved probe outputs. The old probe
  labelled each `noise` (RESULT PASS); all five are the haiku prompt after its
  stop token:

| run | token / n | stop at | solo / other token | probe margin solo / got | teacher-forced margin |
|---|---|---|---|---|---|
| `lanes/bisect/abba-b1` | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.74 / 0.05 | 1.18 [0.98, 1.43] |
| `lanes/bisect/abba-b2` | 29/64 | 21 | `assistant` / `user` | 2.88 / 0.27 | 0.91 [0.86, 0.97] |
| `port-v031/serve-pr149-off-A1` | 28/64 | 23 | `<\|im_start\|>` / `<\|im_end\|>` | 1.02 / 0.05 | 0.55 [0.39, 0.72] |
| `port-v031/serve-pr149-off-B1` | 48/64 | 23 | `<\|im_start\|>` / `<\|im_end\|>` | 1.67 / 0.06 | 0.99 [0.78, 1.28] |
| `topo_smoke_2026-11-05/slotA.run1-benchclient-nogpu` | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.91 / 0.07 | 1.78 [1.26, 2.38] |

- The deterministic MoE epilogue (`VLLM_RDNA2_MOE_DETERMINISTIC=1`) removes
  it: in all 20 runs on the deterministic boot (D1) the two solo runs are bit-identical, so the probe runs in strict mode, and 0 of 20 fail.

## FAIL rate per mode

Tree `1136bb952b` unless named, TP4 on HIP 2-5, one boot per row, 20 runs of
the fixed probe per boot (5 on the two-lane trees). "Boot-time run" is the
single run `serve_validate.sh` made with the old probe on the same boot.

| boot | mode | fixed probe: FAIL runs | run-to-run noise (largest shared-token difference per run) | boot-time run, old probe |
|---|---|---:|---|---|
| P1 | `flashnext-mtp2`, default epilogue | 0 of 20 | noisy in 20 of 20, 1.42 to 5.84 nats | PASS |
| M0 | `flashnext-mtp0`, default epilogue | 0 of 20 | noisy in 20 of 20, 1.63 to 5.01 nats | PASS |
| D1 | `flashnext-mtp2`, `VLLM_RDNA2_MOE_DETERMINISTIC=1` | 0 of 20 | bit-identical in 20 of 20 (strict mode) | PASS |
| TLA | `flashnext-mtp2-twolane`, base `twolane-A-3896928d18` | 0 of 5 | noisy in 5 of 5, 3.27 to 4.91 nats | PASS |
| TLB | `flashnext-mtp2-twolane`, known-bad head `twolane-B-adffce5765` | **5 of 5** (3 or 4 failing checks each) | noisy in 5 of 5, 2.45 to 7.01 nats | FAIL, 8 failing |

The old rule on the same saved outputs: 1 FAIL run of 20 on P1 (the
post-stop haiku flip), 0 of 20 on M0.

## Teacher-forced margin of every FAIL

Each row below is a flip the old probe failed. The margin is the logprob of
the solo run's token minus the other run's token after the solo prefix, from a
clean prefill on the production server (mean of three prefills of the same
tokens, `conc_triage.py adjudicate`). A corrupted run emits a token the model
finds improbable; a noise flip emits a token it finds about as probable.

| group | rows | runs | teacher-forced margin: min / median / max | new probe |
|---|---:|---:|---|---|
| noisy server, flip after the solo stop token | 16 | 11 | -1.00 / 1.38 / 3.74 | 12 soft, 4 near-tie (not judged) |
| noisy server, flip inside the answer | 3 | 3 | 0.25 / 0.50 / 0.82 | 2 near-tie, 1 soft |
| bit-identical server (deterministic arms, `Lisbon` at token 28) | 7 | 7 | 0.26 / 0.83 / 0.97 | 7 FAIL (strict rule, unchanged) |
| known-bad two-lane tree (`abba-B1`, `abba-B2`) | 13 | 2 | 4.33 / 14.92 / 17.10 | still FAIL |

Repeats of the same prefill differ: the three margins of one row in the
post-stop group are a median 0.8 nats apart and up to 3.1. The clean
continuation scores of the two outputs (got minus solo, summed from the flip)
lie between -12.6 and +7.0 nats for the noisy rows and between -39 and -15 for
the known-bad tree, where the other run's continuation is the worse one. The
full list is in the appendix.

## The probe's three flaws, and the fix

1. **Flips after the solo stop token.** `--ignore-eos` (the default, so the
   run is long enough to compare) forces tokens after `<|im_end|>` or
   `<|endoftext|>` that serving never shows. The base model's continuation
   there (`<|im_start|>`, `user`, `assistant`) is unstable: 16 of the 46 old
   FAIL rows of the archive are in it. The probe now makes one normal-stop
   request per prompt, only when a would-be FAIL appears on a noisy server,
   takes the stop token it ends on, and reports (soft) a flip after that
   token's first occurrence in the solo run.
2. **"Not in solo top-5".** It was an automatic FAIL. The probe now asks for
   20 logprobs and reads the margin of both tokens; a token still outside the
   top-20 is bounded by the 20th entry (printed as `>1.234`), so a confident
   solo run with a far-off token still fails.
3. **One-sided threshold.** The solo margin alone was compared with 1 nat,
   below the noise tail (log-odds between two identical prefills: p99 1.0 to
   1.3, max 2.5 to 3.5). On a server whose two solo runs differ (a noisy
   server) a flip fails only if `min(solo margin, other margin)` is above
   `--fail-nats`: both runs have to be sure of a token the other rejected. On
   a server whose solo runs are bit-identical the solo margin alone still
   decides, so the deterministic arms keep the signal the
   [2026-11-06 bisect](tp8-pp2-flashnext-2026-11-06.md) used. The mode is
   printed in the RESULT line.

The garbage and mean-logprob-drop FAILs are unchanged, and so is the leading
`RESULT (PASS|FAIL): N failing, ...` text that `serve_validate.sh` and
`conctally.py` parse. The [2026-11-06 notes](tp8-pp2-flashnext-2026-11-06.md)
had already seen the haiku flips past the stop token and inferred that they
are noise (no run failed with `--no-ignore-eos`, 18 of 18); the probe was
never changed, and this run measured the margins.

## Does it still catch corruption?

Yes. The known-bad two-lane head (`twolane-B-adffce5765`, the tree the
2026-11-06 `abba-B1`/`abba-B2` arms flagged) was booted live with the fixed
probe: 5 of 5 runs FAIL, with 3 or 4 failing checks each, while its base
(`twolane-A-3896928d18`, same recipe, same boot procedure) passes 5 of 5. Its
failures are what corruption looks like, for example from `run00`:

```text
FAIL together 1   prompt 1: token 2/64: margin solo >16.539 got >6.023
FAIL together 2   prompt 2: token 4/64: margin solo >15.508 got >5.828
FAIL long     3   prompt 3: token 1/64: margin solo >10.703 got >6.406
```

Both runs are sure of their own token (both margins far above 1 nat, the
other run's token is outside the 20 returned logprobs), which the noisy rule
still fails. Offline, the archived `abba-B1`/`abba-B2` runs still fail (12 of
their 13 rows), and so do the deterministic `Lisbon` arms under the strict
rule.

## Not verified

- **Scope of the live runs.** One model (Qwen3.8-Flash-Next AWQ W4A16), one
  tree (`1136bb952b`), TP4+EP on four V620, three boots of it (MTP2, MTP off,
  MTP2 with the deterministic epilogue) and two twolane trees. TP8 and PP2
  recipes, the 27B and the dense models are covered only by saved outputs.
- **Rates are small samples.** 20 runs per mode: 0 of 20 has a 95 % upper
  bound of 17 %, 1 of 20 an interval of 0.1 to 25 %. Beyond them the new rule
  has 0 false FAILs in the 204 saved noisy Flash-Next runs that are not the
  known-bad tree. A false FAIL needs the log-odds of the two tokens to swing
  by more than 2 nats between two runs; for two identical prefills that
  exceeds 1.7 nats (MTP2) or 2.5 nats (MTP off) for 0.1 % of the
  (position, alternative token) pairs measured (table above), so one is
  possible but rare; none was seen.
- **What the noisy rule cannot see.** A corruption that shows only after the
  solo stop token of the two stopping prompts is not judged on a noisy server
  (`--no-ignore-eos`, or the deterministic epilogue, judge it). A corruption
  that moves one run onto a near-tie is a WARN, as before. The garbage and
  mean-logprob checks and the failing-solo checks are unchanged.
- **The stop token.** The probe takes it from one normal-stop request per
  prompt and assumes the request ends on it (`finish_reason` stop, last token
  the stop token), which holds on this server for both stopping prompts.
- **Strict mode.** The 7 `Lisbon` arms still FAIL by design (the 2026-11-06
  bisect used them), although their margins on the production server, 0.26 to
  0.97, are near the 1-nat line; they were not re-run on their own TP8/PP2
  servers.
- **Offline re-judgement** of the 333 distinct saved runs uses their top-5
  logprobs. Five more FAIL runs (`tp4bis`, `tp4sw`; 18 or 19 of 48 checks) are
  HTTP 400 errors on the concurrent requests, not flips: the probe still fails
  a request error and this change does not touch it.
- **Not run, because the verdict is noise:** `VLLM_RDNA2_MOE_FP32_ACCUM=1`,
  `rdna_ar` off (RCCL), eager and `FULL_AND_PIECEWISE` bisects. The noise
  source is attributed to the packed-fp16 CAS epilogue by
  [determinism-2026-11-04.md](determinism-2026-11-04.md) and by the
  deterministic boot here, not by a new trace.
- **Not investigated:** the `PREFIX FAIL` that `serve_validate.sh` printed on
  all three boots here. The probe passes when warm TTFT is at most 0.5 x cold
  TTFT and the warm hit rate is above 50 %; the boots measured 0.38 s against
  0.74 s (MTP2), 0.45 s against 0.89 s (MTP off) and 0.39 s against 0.75 s
  (deterministic), each ratio 1 to 4 % over the line, with a 53.1 % warm hit
  rate (hit granularity 2048 tokens). [INFERENCE] a threshold at the edge of
  what a half-hit prompt can reach, not a cache defect.

## Reproduce

```bash
# on a booted server, CPU only
python3 tools/rdna/port_v031/conc_triage.py validity --url http://127.0.0.1:8000 --model m --out OUT
python3 tools/rdna/port_v031/conc_triage.py runs --url http://127.0.0.1:8000 --model m --out OUT/new --runs 20
python3 tools/rdna/port_v031/conc_triage.py adjudicate --only-fail --url http://127.0.0.1:8000 --model m --out OUT/new
# unit tests, no GPU
python3 -m pytest tests/tools/test_concurrency_probe.py
```

## Appendix: every FAIL row of the archive, teacher-forced

One row per flip the old probe failed (RESULT FAIL), from the 22 saved
Flash-Next runs that failed since 2026-10-25 and kept their JSON (run: a path
fragment under `~/w4a8_runs`, file `concurrency.json`). "Stop at" is the
first stop token of the solo run (`<|endoftext|>` or `<|im_end|>`),
"teacher-forced margin" the mean of three prefills with the min and max, "new
probe" the status the fixed probe gives the row from the same saved output,
with the stop cut where it applies (top-5 only: a token outside it is only
bounded by the 5th entry, so the offline verdict on these old files is the
least sharp the new rule gives).

| group | run | check | token / n | stop at | solo / other token | probe margin solo / got | teacher-forced margin | new probe |
|---|---|---|---|---|---|---|---|---|
| noisy, after stop | `2026-10-28-moe-offload` | churn (3, 1) p1 | 26/32 | 21 | `<\|im_start\|>` / `<\|im_end\|>` | 1.27 / 1.14 | -1.00 [-1.27, -0.73] | soft |
| noisy, after stop | `2026-10-28-moe-offload` | churn (11, 1) p1 | 26/32 | 21 | `<\|im_start\|>` / `<\|im_end\|>` | 1.27 / 0.61 | -0.91 [-1.19, -0.51] | soft |
| noisy, after stop | `expert-stats-t10b/arms/G` | churn (3, 1) p1 | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.17 / 0.41 | 1.69 [0.63, 2.50] | soft |
| noisy, after stop | `lanes/fix/proof/w1A1` | churn (11, 1) p1 | 26/64 | 21 | `<\|im_end\|>` / `<\|im_start\|>` | 1.23 / 0.20 | 0.90 [0.62, 1.22] | soft |
| noisy, after stop | `lanes/fix/proof/w1AB1a` | churn (11, 1) p1 | 27/64 | 22 | `<\|im_start\|>` / `<\|im_end\|>` | 1.02 / 0.34 | 1.43 [1.35, 1.58] | soft |
| noisy, after stop | `lanes/ridgefill/abba-fix2` | together 1 p1 | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.86 / 0.14 | 1.33 [0.95, 1.80] | tie |
| noisy, after stop | `lanes/ridgefill/abba-fix2` | churn (11, 1) p1 | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.86 / 0.28 | 2.01 [0.28, 3.12] | soft |
| noisy, after stop | `lanes/ridgefill/abba-head1` | churn (3, 1) p1 | 26/64 | 21 | `<\|im_end\|>` / `<\|im_start\|>` | 1.30 / 0.38 | 0.58 [0.45, 0.72] | soft |
| noisy, after stop | `lanes/ridgefill/abba-head1` | together 1 p1 | 29/64 | 21 | `assistant` / `<\|im_start\|>` | 4.33 / 0.05 | 2.74 [2.56, 3.05] | tie |
| noisy, after stop | `lanes/ridgefill/conc-noise` | together 1 p1 | 29/64 | 21 | `assistant` / `<\|im_start\|>` | 3.86 / 0.33 | 3.74 [2.03, 4.60] | soft |
| noisy, after stop | `lanes/ridgefill/parity-head` | churn (3, 1) p1 | 25/64 | 21 | `<\|im_start\|>` / `<\|im_end\|>` | 1.52 / 0.88 | 0.62 [0.32, 0.88] | soft |
| noisy, after stop | `lanes/ridgefill/parity-head` | together 1 p1 | 29/64 | 21 | `assistant` / `<\|im_start\|>` | 2.91 / 0.12 | 2.77 [2.03, 3.69] | tie |
| noisy, after stop | `smoke3-ab/B1` | churn (3, 1) p1 | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.12 / 0.24 | 2.06 [1.58, 2.64] | soft |
| noisy, after stop | `tp8pp2_2026-11-06/arms/m-pp2-b1` | together 1 p1 | 25/64 | 21 | `<\|im_start\|>` / `<\|im_end\|>` | 1.50 / 0.09 | 1.14 [0.95, 1.30] | tie |
| noisy, after stop | `tp8pp2_2026-11-06/arms/m-pp2-b1` | churn (11, 1) p1 | 25/64 | 21 | `<\|im_start\|>` / `<\|im_end\|>` | 1.50 / 0.90 | 0.91 [0.21, 1.29] | soft |
| noisy, after stop | `tp8pp2_2026-11-06/arms/tp8-c` | together 1 p1 | 25/64 | 23 | `<\|im_start\|>` / `user` | 2.37 / 0.34 | 2.38 [0.59, 3.66] | soft |
| noisy, before stop | `expert-stats-t10/arms/B2` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.38 / 0.17 | 0.82 [0.30, 1.57] | tie |
| noisy, before stop | `smoke3-ab/B1` | together 0 p0 | 6/64 | 17 | `.` / `of` | - / - | 0.25 [0.08, 0.41] | soft |
| noisy, before stop | `tp8pp2/arms/h1v-int-compiled` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.41 / 0.02 | 0.50 [0.14, 0.95] | tie |
| bit-identical | `tp8pp2/arms/h1v-int-concdet` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.03 / 0.11 | 0.26 [0.15, 0.37] | FAIL |
| bit-identical | `tp8pp2/arms/h1v-int-concrepl` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.03 / 0.11 | 0.97 [0.79, 1.24] | FAIL |
| bit-identical | `tp8pp2/arms/h1v-int-verifydig1` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.03 / 0.11 | 0.35 [0.08, 0.77] | FAIL |
| bit-identical | `tp8pp2/arms/h1v-int-verifynodig1` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.03 / 0.11 | 0.88 [0.49, 1.17] | FAIL |
| bit-identical | `tp8pp2_2026-11-06/arms/bis-x2` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.23 / 0.00 | 0.83 [0.50, 1.29] | FAIL |
| bit-identical | `tp8pp2_2026-11-06/arms/h-s2d-flat` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.23 / 0.00 | 0.32 [-0.16, 0.73] | FAIL |
| bit-identical | `tp8pp2_2026-11-06/arms/tp8-s2d` | together 0 p0 | 28/64 | - | `Lisbon` / `\n\n` | 1.23 / 0.00 | 0.84 [0.42, 1.37] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B1` | together 2 p2 | 0/64 | - | ` ` / `\n\n` | 4.42 / 2.27 | 4.33 [4.26, 4.42] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B1` | together 4 p4 | 1/64 | - | `<think>` / `\n\n` | - / - | 16.42 [16.32, 16.48] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B1` | together 3 p3 | 1/64 | - | `n` / `if` | - / - | 14.72 [14.72, 14.72] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B1` | together 1 p1 | 1/64 | 21 | `End` / `\n\n` | - / - | 15.29 [15.29, 15.30] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B1` | long 3 p3 | 1/64 | - | `<think>` / `\n\n` | - / - | 14.77 [14.63, 15.03] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 3 p3 | 1/64 | - | `n` / `if` | - / - | 14.72 [14.71, 14.75] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 2 p2 | 1/64 | - | `2` / `\n\n` | - / - | 14.20 [14.15, 14.24] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 6 p6 | 1/64 | - | `<think>` / `\n\n` | - / - | 17.10 [16.96, 17.21] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 4 p4 | 1/64 | - | `<think>` / `\n\n` | - / - | 16.46 [16.38, 16.53] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 5 p5 | 1/64 | - | `principles` / `the` | - / - | 8.55 [8.51, 8.64] | soft |
| twolane-B | `lanes/feat/abba/abba-B2` | together 7 p7 | 1/64 | - | `lived` / `there` | - / 1.12 | 14.92 [14.86, 15.02] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | together 1 p1 | 1/64 | 22 | `End` / `\n\n` | - / - | 15.36 [15.32, 15.39] | FAIL |
| twolane-B | `lanes/feat/abba/abba-B2` | long 3 p3 | 1/64 | - | `<think>` / `\n\n` | - / - | 15.29 [15.07, 15.40] | FAIL |
