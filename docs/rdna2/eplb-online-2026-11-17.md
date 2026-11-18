# Online EPLB for the RDNA2 W4A16 / W4A8 MoE (2026-11-17)

`VLLM_RDNA_EPLB=1` moves experts between expert-parallel ranks while the
server runs. Default off; with the flag off nothing below exists (one `None`
check per step).

**Status (2026-11-17): opt-in, not merged; no speedup demonstrated.**
The four-hour planner soak completed 71 benchmark cells and 233 expert swaps
without a reported KFD eviction. Its final cell was deliberately interrupted.
The flag-off receipts pass token-exact and overlap checks, but their speed
verdicts still contain regressions; they are not a passing speed gate.
Mapped K2 passes kernel tests and six-architecture disassembly comparison; a
Mono-enabled server completed 192 live swaps. The paired hard-injection gates
exercise the modules path at 21–24 rows, not K2 (limited to eight rows).
The mixed-size soak does not establish stationary-traffic quiescence.
A first run ended in a PCIe fault on the box whose cause remains open.

Background and the load study: [`eplb-lean-design-2026-11-16.md`](./eplb-lean-design-2026-11-16.md).
Upstream `--enable-eplb` stays refused for this MoE method: it routes through a
Triton map-and-record kernel per layer and transfers on a worker thread with
its own stream, neither of which this stack takes on the decode path.

## What it does

A decode step is as slow as its slowest rank. With expert parallelism each
rank holds a fixed quarter (TP4) of every layer's experts, so a rank whose
experts are activated more often is the slow one and the others wait at the
all-reduce. The balancer swaps experts between ranks so the activated experts
spread evenly.

A swap exchanges two experts of one layer between two ranks: each rank keeps
its expert count, its slots and its tensors; only the rows of two slots and
two entries of `expert_map` change. Nothing is reallocated, so captured CUDA
graphs keep working (they hold the same pointers).

## How it stays out of the way of serving

| Rule | Mechanism |
|---|---|
| No collective can hang | Every collective is scheduled by the step counter and by the plan all ranks received, never by rank-local state. A rank with a local problem keeps the schedule, sends zeros and raises an error word; all ranks switch the balancer off at the same step. |
| No thread in the worker | The policy runs in a helper process of rank 0 (numpy only), fed through shared memory one layer per step with non-blocking copies. |
| Bounded work per step | At most one collective per step: a mailbox (every 256 steps, 196 int32 words by default), or one transfer chunk (64 KiB per direction), or one vote (8 words). |
| No lost or duplicated expert | A swap commits only after a vote that carries both owners' checksums of what they sent and received. A failed vote changes nothing and switches the balancer off. |
| Same placement on all ranks | Rank 0 writes the plan into the mailbox all-reduce; the other ranks write zeros. Every rank reads the same plan at the same step and commits at the same step. |

One swap of Flash-Next experts is 2 x 2.55 MB: 39 chunk steps and one vote
step with the default chunk.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_EPLB` | 0 | 1 = on (also switches the load counters on) |
| `VLLM_RDNA_EPLB_MAILBOX_STEPS` | 256 | steps between mailbox collectives |
| `VLLM_RDNA_EPLB_CHUNK_KB` | 64 | KiB per direction and per step |
| `VLLM_RDNA_EPLB_CHUNK_EVERY` | 1 | steps between two chunks |
| `VLLM_RDNA_EPLB_DECISION_STEPS` | 4096 | steps between two decisions |
| `VLLM_RDNA_EPLB_MAX_SWAPS` | 32 | swaps of one plan |
| `VLLM_RDNA_EPLB_OBJECTIVE` | touch | `touch` (decode: distinct experts per rank) or `pairs` (prefill) |
| `VLLM_RDNA_EPLB_CHAOS` | 0 | N > 0: no policy, N random swaps per decision (qualification) |

Mailbox steps take priority over transfer steps. A cadence with
`CHUNK_EVERY % MAILBOX_STEPS == 0` is rejected because every transfer would be
starved by a mailbox (including `MAILBOX_STEPS=1`). The mailbox's deliberate
host/device handshake is covered by the sync-check allow-region, including
its host-to-device copy, so strict debug checking does not reject that copy.

`tools/rdna/serve_rdna.sh` passes them through, e.g.
`bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 MODEL=... VLLM_RDNA_EPLB=1`.

## Scope

- RDNA2 W4A16 MoE, resident and K-first layouts, and the W4A8 activation
  path (same six per-expert tensors: packed weights, scales, zero points of
  `w13` and `w2`).
- Expert parallelism over the TP group, 2 to 8 ranks, one pipeline stage,
  linear placement strategy (a static `VLLM_RDNA_MOE_PLACEMENT` file is the
  starting point if set).
- Not moved: the MTP draft model's BF16 experts, the expert-offload cache.
- The Flash-Next mono K2 kernel (`--monokernel`) runs in map mode while the
  balancer is on: it resolves routed ids through the layer's live
  `expert_map` and reports them for the counters. Kernel tests (HIP 1) 387
  passed / 48 skipped and archdiff over six arches shows only the four new
  Stage-4 functions per arch. A served `--monokernel` + `VLLM_RDNA_EPLB=1`
  run completed 192 swaps and passed its post-load greedy probe; see below.
  A separate cross-build check on HIP 1 gives bit-identical default-K2
  output digests for every batch size 1–24, with zero sticky fault words.
- A swap changes which rank adds which partial sum, so outputs are equal in
  value, not bit for bit (fp16 all-reduce association), as with any change of
  placement.

## Policy

`rdna_eplb_policy.decide_swaps` per layer (moved from `tools/rdna/eplb/`): the
swap sequence that lowers the heaviest rank, judged on the loads the ranks
really had in the window, with a holdout on the previous window and a minimum
relative gain. Balanced stationary samples should produce no plan; real
repeated traffic can still trigger placement changes (see the soak below).

## Tests

`tests/model_executor/layers/fused_moe/test_rdna_eplb_online.py`: N ranks run
as threads and meet in a rendezvous all-reduce, so a rank that schedules a
collective the others do not breaks the test.

- random swaps keep every expert on exactly one rank, bit for bit;
- an idle balancer costs one small collective per mailbox and nothing else;
- one rank's setup error, a damaged transfer and a forged plan each switch
  all ranks off at the same step with the placement untouched;
- a planner answer travels through the mailbox and is executed; invalid
  answers are never posted;
- gfx1030: the real resident kernels (skinny decode and tiled) match an fp32
  reference after swaps, in eager mode and in CUDA graphs captured before the
  swaps; the K-first W4A16 and W4A8 kernels give the same output before and
  after.
- the real planner process, fed by skewed traffic through the counters and
  shared memory, moves hot experts off the overloaded rank in every layer.

## Measurements

`flashnext-mtp2` (AWQ), TP=4 on four V620 (HIP 2-5), FULL_AND_PIECEWISE, MTP2,
2026-11-17, one boot per arm, so differences of 1 % are inside the spread.

Historical zero fault-marker counts used a scanner that missed `GPU Hang` and
`HW Exception`. Kernel-log access was unavailable: the old `0` followed by
`n/a` is unknown, not a measured zero. Those summaries alone do not establish
absence of faults; final qualification uses corrected recognition and
before/after PCIe AER evidence.

| | flag off | chaos swaps | planner (default) |
|---|---|---|---|
| ITL median 1k/512 c=1 | 22.4 ms | 22.7 ms | 22.6 ms |
| ITL median 1k/512 c=8 | 46.3 ms | 46.2 ms | 46.5 ms |
| ITL median 16k/1k c=1 | 23.0 ms | not run | 23.2 ms |
| ITL median 16k/1k c=8 | 48.1 ms | not run | 48.3 ms |
| stall probe max gap, inject 4096 | 425 ms | 424 ms | 425 ms |
| stall probe max gap, inject 16384 | 438 ms | 445 ms | 437 ms |
| PPL probe | 4.81 | pass | 4.83 |
| KFD evictions | 0 | 0 | 0 |
| plans executed | 0 | 50 (8 swaps each) | 4 (118 swaps) |

- 425-445 ms is one prefill step of the injected prompt; the swaps add no gap
  on top of it.
- Earlier counter measurements cost +0.8–1.0 % of ITL at c=1 and +0.3–0.6 %
  at c=8, not +0.9 % in every cell.
- Chaos mode (random swaps every 512 steps, no planner) is the qualification
  of the transfer: 50 plans, each 329 steps (16-30 s wall) for 8 swaps, no
  reported fault.
- In the initial short runs, the planner did nothing during the bench cells.
  During the stall probe's repetitive prompts it made 4 plans
  (32, 32, 32, 22 swaps; 44–73 s wall each, 1317 steps for 32 swaps).
  The longer soak below supersedes that initial observation about
  stationary traffic.
- **No speedup is measured.** The study predicts at most about 0.4 % of ITL
  for a permutation at EP4, below this noise [INFERENCE]; a gain needs replicas
  (not built) or traffic with a persistent skew.

### Four-hour planner soak

`soak1`, 15:18:59–19:19:52 CET, TP4 HIP 2–5, planner enabled, MonoKernel off.
The run was stopped deliberately after four hours: 71 cells completed,
2,272 requests; the 72nd cell was interrupted, not counted as a pass.
The original 110-cell loop was too long for the intended four-hour window.

| Cell | Completed cells (32 requests each) | Median of cell ITL medians | Range of cell ITL medians |
|---|---:|---:|---:|
| 1k/512 c=8 | 36 | 46.0 ms | 45.3–47.0 ms |
| 16k/1k c=8 | 35 | 47.8 ms | 46.9–48.6 ms |

- All 71 completed cells report zero KFD eviction deltas. The kernel-log
  check was unavailable, so this run has no kernel-log no-fault receipt.
- PPL 4.82; the prefix probe failed its ratio threshold (cold 0.74 s,
  warm 0.40 s, 1.9×), not a claim that prefix caching passed.
- Fourteen plans completed, 233 swaps. Plan starts span step 13,056 to
  103,936; the first five each moved 32 experts, later plans moved 3–18.
  This exceeds one plan per 10,000 steps over the observed start interval,
  but alternating prompt sizes is not a fixed stationary workload at the
  decision-window scale. A true stationary-traffic gate remains unproven.
- CPU compilation/archdiff and a 149-second HIP 1 kernel suite overlapped
  this run. This is stability evidence, not a quiet speed measurement.
  Only cell medians are retained for every repetition; no whole-soak
  per-request maximum-gap or p99 guarantee is inferred from those medians.

Logs: `~/w4a8_runs/eplb-online/soak1.log` and
`~/w4a8_runs/port-v031/serve-eplb-soak1/serve.log`.

### Served MonoKernel, W4A8 and live swaps

`monochaos`, 19:20:08–19:54:20 CET, combined tree before the two Python
review fixes, TP4, MTP2, FULL_AND_PIECEWISE and `--monokernel`.
All four ranks logged mapped K2 execution and the W4A8 sdot4 prefill kernel.
Six plans completed, 192 swaps, with zero reported KFD eviction deltas,
zero serve-log fault markers and zero reported new GPU faults.

- Greedy, PPL 4.78, prefix (2.1×), concurrency and post-load greedy probes
  pass. Concurrency: 0 failing of 48 checks, with 17 near-tie and 1 soft.
- ITL medians at 1k c=1/c=8: 24.1/47.6 ms; 16k c=1/c=8: 24.9/49.8 ms.
  These are one-arm observations, not a speed comparison.
- `CHAOS=512` was capped by `MAX_SWAPS=32`; it did **not** mean a 512-step
  decision interval. Plans ran at the default 4,096-step decision cadence.
- The stall probe returned **NO-GATES**, not PASS. Eight decoders filled
  `max_num_seqs=8`; 4k injections waited about 135 s for admission. The
  reverse scenario had a 5.57 s maximum gap, and the 16k injection reported
  four starved decoders. None of this is a clean no-stall qualification.
  The paired rerun below reserves one request slot for the injected prefill.

### Paired injection hard gates after review fixes

`monogate-off` and `monogate-chaos`, 19:57:52–20:16:05 CET, source
`dde11baa4f`, TP4, MTP2, FULL_AND_PIECEWISE, MonoKernel on in both arms.
Seven decoders leave one slot free under `max_num_seqs=8`. Three repetitions
per prompt length, `--max-starved 0 --min-decode-tokens-during-prefill 1
--gate-stat worst`, matching the receipt's existing injection gates.

| Observation | EPLB off | Chaos (8 swaps / 512 decision steps) |
|---|---:|---:|
| Hard injection gates | PASS | PASS |
| Baseline ITL p50 | 45.7 ms | 46.4 ms |
| 4k injection mean maximum gap | 419 ms | 422 ms |
| 16k injection mean maximum gap | 432 ms | 435 ms |
| Reverse 16k mean maximum gap (not gated) | 5,212 ms | 5,215 ms |
| Completed plans / swaps | 0 / 0 | 5 / 40 |
| Reported KFD eviction delta | 0 ms | 0 ms |
| PPL probe | 4.80 | 4.83 |

Greedy, prefix, concurrency and post-load greedy probes pass in both arms;
zero serve-log fault markers. The gates require actual decoder progress and
zero starved decoders during injection, not zero scheduling delay. Reverse
arrival is reported but is not hard-gated. The +0.7 ms ITL and +3 ms gaps
are one paired observation, not an ABBA result or a demonstrated speedup.
No baseline scheduler/admission change was made to obtain this result.

Receipts: `~/w4a8_runs/port-v031/serve-eplb-monogate-{off,chaos}/stall/`
and `~/w4a8_runs/eplb-online/gated_mono.log`. The b4 lease was released.

### Flag-off receipts

Base `b56907c433`, head `7421070327`: the AB receipt passes token-exact
(26/26), overlap, and the stall hard gate. Subsequent ABBA runs pass
token-exact and overlap; stall was disabled for those repeats.

The first ABBA reports regressions in 3/4 cells. The two-cell repeat
(`20261117-123921-7421070327-vs-b56907c433`) still reports regressions in
both cells: aggregate output −2.4% at c=4 and −3.7% at c=8.
Its ITL medians improve 1.0%/0.6%; c=4 p99 changes direction, with
154.2/384.9 ms on base versus 61.9/60.6 ms on head. This demonstrates tail
variability, not that every regression label is noise or that the speed
gate passed. These receipts predate the Mono K2 integration.

### The first run, a PCIe fault on the box

The first chaos run ended with all four workers dead. The recorded kernel
event was a fatal PCIe error on the V620 switch's root port at 01:28:44
(completion timeout, DPC containment), affecting cards outside the run too.
Application logs contain probe completions from 01:28:38 through 01:28:48,
so the previous claim that the server was idle is unsupported. A standalone
reproduction of the balancer's int32 all-reduces (196 and 32768 words, 9000
each, RCCL, four ranks) passed with exact sums, and later runs completed
without reproducing the fault. Cause remains open; neither the balancer
collective nor an unrelated hardware cause has been established.

### Final qualification protocol

`tools/rdna/eplb/qualify_final.sh` runs two bounded campaigns:

- `MODE=r1`: the native eight-cell flag-off ABBA receipt, token-exact checks,
  overlap and default seven-decoder stall probes, using independently built
  base/head extension sets. Speed verdicts remain reported, not relabelled.
- `MODE=r2`: Mono enabled throughout; fresh off/chaos/chaos/off boots at
  1k/512 and 16k/1k, c=1/8, with native request counts and the existing ITL
  envelope. Every on-arm measured cell must overlap a committed plan.

Native benchmark JSON records `benchmark_start_time_monotonic`; controller
start/commit logs record the same host's monotonic clock. Warm-up or an
unfinished plan alone cannot satisfy measured-plan coverage. Guards reject
incomplete results, failed probes, unknown kernel telemetry, KFD eviction,
changed/reset PCIe AER counters, thermal alarms and application fault markers.
A live fault latches an abort; no automatic retry or four-hour soak.

These campaigns do not establish stationarity or resolve the historical
PCIe incident. The base build log was not located; the reused head extensions
come from the recorded Mono build. Record all seven extension hashes rather
than describing this as a newly rebuilt production baseline.

### Original final run (2026-11-18): R1 passes, R2 incomplete

Runtime source `2ea7a364848c`, base `b56907c433`, independently built extension
sets as recorded above. R1's six boots pass token-exact, overlap and stall
gates; the full quality gate was **NOT RUN** (short PPL probes are separate).
The native speed result is **3 wins, 3 regression cells, 2 no measurable
difference**, not an all-green performance receipt. The initial wrapper's
printed count of five matched Markdown headings/footnotes too; its counter
has been corrected to use native JSON cell verdicts.

R2 completed only A1/off and B1/chaos. Both reused 1024 of 1927 prompt tokens
(53.1%). The legacy prefix probe failed B1 because warm TTFT 0.40 s exceeded
half of cold TTFT 0.74 s; A1 was 0.39/0.84 s (rounded logs). B1's prefix
failure at 08:24:35 precedes its first rebalance plan at 08:24:55. All four
B1 measured cells overlap committed plans; mapped 3-row FULL K2 and 24-row
module witnesses are present on all four ranks. Guards report no hardware
faults or KFD eviction; b4 was released.

The original abort and failed prefix result are retained unchanged. This is
**not completed ABBA or EPLB qualification**. The repaired probe separates
verified reuse from an explicitly requested latency target; the original
ITL envelope is unchanged. A revised four-arm campaign must be identified
separately; no failed performance label is being relabelled.

Raw original receipts (463 files, caches excluded) and all seven base/head
extension hashes: [`bench_results/2026-11-18_eplb-final/original-summary.json`](../../bench_results/2026-11-18_eplb-final/original-summary.json)
and its linked `original-receipts.tar.gz`.

### Corrected prefix-contract ABBA: performance fails (2026-11-18)

Source `06e01fee103a`, same native/runtime code and seven reused head extensions.
All four boots complete with valid functional probes and hardware guards.
Prefix reuse is 1024/1927 tokens (53.1%) in every arm; short PPL is 4.82–4.84
over 261 tokens, **not a full quality qualification**. Both chaos boots have
committed-plan overlap in every measured cell. No new faults or KFD evictions;
b4 was released. The original aborted campaign remains unchanged.

| Cell | Off p50 ITL (ms) | On p50 ITL (ms) | Change | Off p99 ITL (ms) | On p99 ITL (ms) | Change |
|---|---:|---:|---:|---:|---:|---:|
| 1k/512 c1 | 23.37 | 24.15 | +3.33% | 23.95 | 27.64 | +15.39% |
| 1k/512 c8 | 46.92 | 47.51 | +1.26% | 380.88 | 383.74 | +0.75% |
| 16k/1k c1 | 24.10 | 24.72 | +2.59% | 24.60 | 28.04 | +14.01% |
| 16k/1k c8 | 49.38 | 49.63 | +0.51% | 429.39 | 430.65 | +0.29% |

Seven of eight ITL rows fail the unchanged `max(0.5%, off-boot spread)` parity
screen. **NOT QUALIFIED**; no threshold relaxation or performance reroll.
Chaos is a transfer stress test, not a demonstration of useful placement.
TTFT/aggregate output remain context metrics, not proof of placement benefit.
Raw four-arm evidence, native gate, aggregate failure report and extension
manifest: [`corrected-summary.json`](../../bench_results/2026-11-18_eplb-final/corrected-summary.json)
and its linked `corrected-receipts.tar.gz`.

### No-plan diagnostic controls

`tools/rdna/eplb/counter_control.sh` runs two additional fresh boots, **C1/C2,
not replacement ABBA arms**. It keeps counters, mapped K2, module routing and
the scheduled mailbox enabled, but uses `CHAOS=1 DECISION_STEPS=1000000000`:
no planner, and no plan may occur. The same four cells, MTP2, F&P, Mono,
functional probes and thermal/AER/KFD guards apply. Take b4's lease before
running with `VENV=... MODEL=... OUT=<fresh path>`; `DRY_RUN=1` prints plans.

`qualify_boot.py boot --arm C1|C2` verifies this diagnostic contract, including
zero plans and witnesses on all four ranks. Counter-only receipts cannot
qualify R2, even if renamed as an on arm. Compare these controls with the saved
off arms only as a later diagnostic, not a fresh ABBA or proof of causality.
Their purpose is to separate always-on counter/map/mailbox cost from
movement/checksum/placement cost before choosing a runtime optimization.
