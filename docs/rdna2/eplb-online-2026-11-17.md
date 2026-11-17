# Online EPLB for the RDNA2 W4A16 / W4A8 MoE (2026-11-17)

`VLLM_RDNA_EPLB=1` moves experts between expert-parallel ranks while the
server runs. Default off; with the flag off nothing below exists (one `None`
check per step).

**Status: not qualified, do not enable on a shared box.** The first server run
with the flag on ended in a GPU hang (see [Measurements](#measurements)); the
cause is open.

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
- The Flash-Next mono K2 kernel stands down while the balancer is on (it
  needs the contiguous expert block); see "Mono K2" below.
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

No ITL, TTFT or throughput number exists for this path yet.

**First server run, 2026-11-17, FAILED.** `flashnext-mtp2`, TP=4 on four V620,
`VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8 VLLM_RDNA_EPLB_DECISION_STEPS=512`:

- boot and bind normal (`Online EPLB: 48 layers, 4 ranks, 2.44 MiB per expert,
  39 chunks per swap`), three probe requests answered;
- about one minute after ready, all four workers died with `HW Exception by
  GPU node-1 ... reason: GPU Hang`; no plan had been logged;
- the kernel then reported `device lost from bus` on the four serving cards
  and, four minutes later, on the two other V620 of the same PCIe switch.

Cause not established. The balancer's collectives are int32 all-reduces issued
at the step boundary; `rdna_ar` takes float tensors only, so they run on RCCL.
Whether the first mailbox collective (step 256) had been reached when the
hang occurred is not known [INFERENCE: it is the first suspect]. Before any
further server run: reproduce on a single collective outside the server.
