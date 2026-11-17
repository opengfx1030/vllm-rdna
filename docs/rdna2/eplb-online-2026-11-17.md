# Online EPLB for the RDNA2 W4A16 / W4A8 MoE (2026-11-17)

`VLLM_RDNA_EPLB=1` moves experts between expert-parallel ranks while the
server runs. Default off; with the flag off nothing below exists (one `None`
check per step).

**Status (2026-11-17): opt-in, not merged; no speedup demonstrated.**
The four-hour planner soak completed 71 benchmark cells and 233 expert swaps
without a reported KFD eviction. Its final cell was deliberately interrupted.
The flag-off receipts pass token-exact and overlap checks, but their speed
verdicts still contain regressions; they are not a passing speed gate.
Mono K2 passes kernel tests and six-architecture disassembly comparison;
served MonoKernel + EPLB qualification is running. The stationary-traffic
plan-rate gate is not met over the whole soak. A first run ended in a PCIe
fault on the box whose cause remains open.

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
  Stage-4 functions per arch; a served `--monokernel` + `VLLM_RDNA_EPLB=1`
  run is not qualified yet.
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
- The counters cost about +0.2 ms (+0.9 %) of ITL at c=1 and c=8 in every
  cell (earlier measurement: +0.6 to +0.9 %).
- Chaos mode (random swaps every 512 steps, no planner) is the qualification
  of the transfer: 50 plans, each 329 steps (16-30 s wall) for 8 swaps, no
  fault.
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

- All 71 completed cells report zero KFD eviction deltas. No PCIe
  completion-timeout/DPC/device-lost signature was found in the kernel log.
- PPL 4.82; the prefix probe failed its ratio threshold (cold 0.74 s,
  warm 0.40 s, 1.9×), not a claim that prefix caching passed.
- Fourteen plans completed, 233 swaps. Plan starts span step 13,056 to
  103,936; the first five each moved 32 experts, later plans moved 3–18.
  The whole-run gate of at most one plan per 10,000 steps is not met.
  Convergence is not the same as proving stationary-traffic quiescence.
- CPU compilation/archdiff and a 149-second HIP 1 kernel suite overlapped
  this run. This is stability evidence, not a quiet speed measurement.
  Only cell medians are retained for every repetition; no whole-soak
  per-request maximum-gap or p99 guarantee is inferred from those medians.

Logs: `~/w4a8_runs/eplb-online/soak1.log` and
`~/w4a8_runs/port-v031/serve-eplb-soak1/serve.log`.

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

The first chaos run (same flags) ended with all four workers dead and, in the
box's kernel log, a fatal PCIe error on the root port of the V620 switch at
01:28:44 (completion timeout, DPC containment) that froze every card below it,
including two that were not in the run. The server was idle: no decode step,
so no balancer collective, was executing. Nothing else was found; a single-
process reproduction of the balancer's int32 all-reduces (196 and 32768 words,
9000 each, RCCL, four ranks) passed with exact sums, and the same run later
completed twice without a fault. Cause open; it may be unrelated to this
change. An earlier statement that the balancer's collective was the likely
cause was wrong.
