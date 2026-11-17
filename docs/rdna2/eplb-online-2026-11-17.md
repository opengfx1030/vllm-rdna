# Online EPLB for the RDNA2 W4A16 / W4A8 MoE (2026-11-17)

`VLLM_RDNA_EPLB=1` moves experts between expert-parallel ranks while the
server runs. Default off; with the flag off nothing below exists (one `None`
check per step).

**Status (2026-11-17): works and does not stall; no speedup measured; not
merged.** Two server runs on one slot (24 and 34 minutes) with the flag on
showed no fault, no eviction and no added stall (see [Measurements](#measurements)).
Still owed before it can be called qualified: the flag-off receipt, a soak of
at least 4 hours, stationary-traffic behaviour of the planner, and a measured
gain. A first run ended in a PCIe fault on the box whose cause is open.

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
- A swap changes which rank adds which partial sum, so outputs are equal in
  value, not bit for bit (fp16 all-reduce association), as with any change of
  placement.

## Policy

`rdna_eplb_policy.decide_swaps` per layer (moved from `tools/rdna/eplb/`): the
swap sequence that lowers the heaviest rank, judged on the loads the ranks
really had in the window, with a holdout on the previous window and a minimum
relative gain. A stationary workload produces no plan.

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
- The planner did nothing during the bench cells (uniform random prompts, no
  imbalance to fix). During the stall probe's repetitive prompts it made 4
  plans (32, 32, 32, 22 swaps; 44-73 s wall each, 1317 steps for 32 swaps).
  That is one plan per about 6.5k steps on non-stationary traffic; the gate
  for stationary traffic (at most one per 10,000 steps) is not tested.
- **No speedup is measured.** The study predicts at most about 0.4 % of ITL
  for a permutation at EP4, below this noise [INFERENCE]; a gain needs replicas
  (not built) or traffic with a persistent skew.

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
