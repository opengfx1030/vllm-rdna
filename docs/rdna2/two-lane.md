# Two-lane engine: acceptance verdicts

Status: **M1 green; M2 correctness green; M4 measured no-go (keep opt-in);
M3 gate running**. Branch `feat/two-lane`; contract in
`~/w4a8_runs/lanes/async/two-lane-contract.md` (untracked copy in the tree).

The engine is opt-in (`TWO_LANE=1`); the flag-off path stays the rdna_extras
default. Flag-on recipes are `tools/rdna/recipes/flashnext-mtp2-twolane.env`
(TP4) and `tools/rdna/recipes/27b-awq-twolane.env` (TP2).

## Runbook: booting a freshly rsynced tree

The rsync excludes `*.abi3.so`, so a new tree has no built custom ops. Copy the
seven op files (`_C.abi3.so`, `_C_stable_libtorch.abi3.so`,
`cumem_allocator.abi3.so`, `fs_io_C.abi3.so`,
`_moe_C_stable_libtorch.abi3.so`, `_rocm_C.abi3.so`, `spinloop.abi3.so`)
from a built tree of the same source before the first boot, otherwise model init
dies with `RuntimeError: input and z must be 2D [M, N]` in
`qwen_gdn_full_forward`. Verified 2026-11-03 on `serve-m2fix-off`/`-on`.

## M1 TP1 worker driver

Tool: `tools/rdna/prof/lane_driver.py`, Qwen3.5-0.8B fp16, one V620 (HIP 9),
TP1, kv 24 GiB, `RDNA`/`ROCM` attention as the serving path. Reports:
`~/w4a8_runs/lanes/async/accept/m1*/lane_driver.json`.

| gate | result | numbers |
|---|---|---|
| (a) handoff L in {1, 5, 1023, 1024, 1025, 2047, 2048, 4097, 16384, 16385} | **PASS** (eager and compiled) | all lengths byte-equal in both modes, self-handoff too, after `1f5707c298` + `f63fb584ed`; 16384/16385 included with `max_model_len=32768` |
| (b) E-GIL D p99 / D p50 solo | **PASS** | 1.408 (gate <=1.5): D concurrent p99 13.35 ms, D solo p50 9.48 ms (`fix4comp`); earlier `fix4eager` 1.529, P eager Python launch is the residual GIL cost |
| (c) P chunk rate >= 45% of solo | **PASS** | 98.0% |
| (d) 10 min, no hang/error, strict clean | **PASS** | 309 soak iterations, 0 mismatches, 0 errors, `d_reqs=0`, `p_reqs=0` |
| (e) KFD census 7 | **PASS** | after `526f1332ea` the driver censuses 7 queues with D full graphs (`[0..6]`, `handoff-compiled5`); the real two-lane worker uses 6 idle and 7 strict, under a short bench |

Findings:

- The L<=16 mismatch was the driver handing out **block id 0**, vLLM's null
  block, not an S5/S6 bug. The driver now reserves block 0 (real
  `KVCacheManager` never allocates it); all lengths match.
- The `post_update_kernel` GPU fault at 16k was the driver's default
  `max_model_len=16384`: a 16385-token request needs 1025 block-table slots,
  the buffer had 1024. Default is now 32768.
- **Corrected 2026-11-03 (M3).** The earlier "compiled handoff green at all
  ten lengths" was a false positive: the driver's allocator handed out block 0,
  vLLM's null/padding block, so both sides read the same zero page. With real
  blocks (`f63afb53f9`), handoff diverged at every length. Two engine bugs are
  now fixed (`9068cef14f`, pushed):
  1. `TwoLaneScheduler` pinned the handoff row to the prompt boundary
     (`num_computed_tokens = num_prompt_tokens`, `prefill_len = prompt_len`),
     so D's first step is a uniform decode, not a 1-token prefill
     continuation with GDN chunked/conv initial state.
  2. `_get_new_block_ids_to_zero` filters the globally drained new-block ids
     to the requests scheduled in this lane. Otherwise one lane's zeroer could
     zero a block the other lane had just written (the wrong token at the
     second KV block, token 17).
  Concurrency hook on HIP 5: **8/8 PASS**, 48/48 checks identical, no error;
  before the fixes, 2/8 failed at token 17 or the server's MTP0 cold probe
  crashed in `_causal_conv1d_update_kernel` with a memory-aperture fault.
  The compiled lane driver at all ten lengths is now green: `handoff-compiled4`
  (and `-5` after the KFD fix) ran the driver with `--compile-mode 3` on
  Qwen3.5-0.8B, `max_num_seqs=1`, kv 12.5e9, and got 10/10 byte-equal, 0
  mismatches, engine `cudagraph`; the diagnostic was `draft_tokens: -1`. The
  pre-fix workaround that forced the whole D batch to eager on a handoff step
  (`cg_mode=NONE`) is gone: `git grep handoff_eager_step` is empty, and the
  full decode graph now replays the handoff step. The driver's
  `--skip-handoff` off path is the regression gate.
- The prefill-tail FULL replay is real at MTP0 (a 1-token tail dispatches
  `mode=FULL has_prefill=True`, `fix-mtp0` at TP1): at MTP0 the plain engine's
  output is still equal because its prefill metadata is correct, but the
  two-lane handoff row's metadata was not. With MTP2 the scheduler pads every
  tail to the 3-token spec shape (`sched=1`, 2 placeholder drafts), so the
  target tail is `PIECEWISE 3` and production never exercises the bad
  classification (`fix-mtp2`; `mtp2` is also PIECEWISE 3).
- The over-budget queues were not the lane streams: the census was 10 with D
  full graphs because `sample_tokens` created two structured-outputs copy
  streams and `graph_capture` a third. `526f1332ea` runs the grammar-bitmask
  copies on the lane stream and captures on `D_full` (a plain stream with the
  whole-device mask, which can host a capture), so the census is
  `[0..6]` = 7 with D full graphs (`handoff-compiled5`); the production
  recipes keep `GPU_MAX_HW_QUEUES=2`, and the real worker reaches 6 idle /
  7 strict (gate 7).
- The 2026-11-03 19:59 `m1-compiled-fix` eleven-length audit was the same
  block-0 false positive; it is not evidence for real blocks. The KFD 7 and
  EGIL numbers from it remain valid.
  D solo max 9.62 ms; P eager 16 chunks of 341 ms, chunk share 0. The
  driver also logs `fence PREFILL seq N recorded twice; overwriting` for
  the P batch done hook (results unaffected; open).

## Boot provenance (2026-11-03)

| arm | tree | recipe copy md5 | `_rocm_C.abi3.so` md5 |
|---|---|---|---|
| TP4 A (unfixed, boot 17:31) | `~/vllm-rdna-0.32.0-fix` at its pre-269314ac56 HEAD, booted before the W4A8-v2 recipe default landed 17:44 | `91b2819026939da8c62635f81f212a13` | `ab4678d7d8b738b12d5fa4da0c51e484` |
| TP4 B (fix, boot 18:06) | `~/vllm-fix-tail-tree`, `3ae43e4b25` | `91b2819026939da8c62635f81f212a13` | `ab4678d7d8b738b12d5fa4da0c51e484` |
| two-lane M1/M2 | `~/vllm-two-lane-head` | `91b2819026939da8c62635f81f212a13` | `cd324ca18bf2c2f23c7c4c8e66035cf0` |

The fix branch was rebased as `fix/prefill-tail-full-graph-v2` =
269314ac56 + `8072fb28a4` (same 21-line change); the production W4A8-v2 recipe
default copy is `29d260e2a431e9cc9db8027a5012acf6`. On that new base the
worker unit file passes 13/13 (`tests/v1/worker/test_gpu_batch_ordering.py`).

## Production MTP2 check (TP4 Flash-Next, no two-lane)

Plain `rdna_extras` tree 269314ac56, compiled F&P, MTP2, TP4. A (unfixed,
booted 17:31 before the tree's W4A8-v2 default landed 17:44) and C (same code +
`3ae43e4b25`) run the same recipe copy (`flashnext-mtp2.env` md5
`91b2819026939da8c62635f81f212a13`) and the same `_rocm_C.abi3.so` (md5
`ab4678d7d8b738b12d5fa4da0c51e484`). Greedy outputs are equal at all ten
lengths, chunked (`MAXBAT=2048`) vs an unchunked boot; probes PASS, PPL 4.81
both, prefix 6.05/0.31 s vs 5.97/0.29 s, concurrency PASS both. Bench (A vs
C): c1 prefill 2061.9 / 2073.5 tok/s, decode 90.24 / 102.87, TTFT 0.50 /
0.49 s, ITL p50 22.7 / 22.8 ms; c6 prefill 1017.8 / 949.9 (agg 2090.4 /
2093.6), decode 39.99 / 43.13 (agg 209.6 / 227.4), TTFT 1.01 / 1.08 s,
ITL p50 42.9 / 43.2 ms. Verdict: **the production MTP2 stack is safe**; its
scheduler pads the prefill tail to 3 tokens, so the FULL-eligibility
invariant is not exercised, and no measured target prefill dispatch replayed
a FULL graph (no production-stack BATCHDESC instrumentation was run; the
PIECEWISE evidence is the TP1 `fix-mtp2` diag: target tail `PIECEWISE 3`,
`fix-mtp0` target tail `PIECEWISE 1`). The fix still removes the latent
violation for any speculative configuration.

## M2 engine at TP1

Real server `serve_validate.sh`, `~/vllm-two-lane-m2fix` (branch
`feat/two-lane` at `022ee2d31c`), Qwen3.5-0.8B + `MTP=2`, TP1, HIP 2,
`MAXLEN=32768 KV=4e9 SEQS=8 MAXBAT=2048`, compiled F&P, greedy, cells
`1024:512:1 1024:512:8 16384:1024:8:32 16384:128:1`, D/P split 20/16,
flag off and flag on in one sequential pair. Verdict: **G3 decode-only
FAIL (flag on slower); strict max-gap FAIL at P activation** (numbers
below; no gate loosened).

Two engine fixes ride here (`022ee2d31c`, pushed):
(1) `_lane_activity` reads *schedulable* work, not lane residency or a
release-only batch. A decoded request keeps the P bit in `lane_resident`,
and the P slot release rides a zero-token batch; counting either as "P
active" kept D on the 20-WGP masked stream after the prompt (the
`c1/c8/16k:128` decode-only regression). The fix covers the scheduler's
queued next batch and the fake scheduler's `lane_activity`.
(2) Adaptive split: a mixed iteration with a small D batch (<=32 scheduled
tokens, drafts included) gives P `P_full` (all WGPs) and leaves D on A
(20 WGPs). It is stamped per lane, not on both outputs; D stays on A and
P takes the full mask. With a big D batch both lanes keep the partitioned
flag. This recovers the 16k:1024 c8 prefill without a full-GPU decode.

| gate | off | on |
|---|---|---|
| correctness / PPL / prefix | PASS / 14.73 / 2.21-0.13 s | PASS / 14.73 / 2.28-0.15 s |
| decode-only c1 (`1024:512:1`) | 301.80 tok/s, ITL 9.8 ms | **284.52 tok/s, ITL 10.4 ms** (slower, G3 FAIL) |
| decode-only c8 (`1024:512:8`) | 114.73 tok/s, ITL 23.5 ms | **109.45 tok/s, ITL 24.1 ms** (slower, G3 FAIL) |
| 16k:128 c1 (small D) | 4711.9 prefill, 215.73 decode, TTFT 3.48 s | 4644.2 prefill, 202.39 decode, TTFT 3.53 s |
| 16k:1024 c8 prefill aggregate | 9124.9 tok/s (`2420.7`/req) | **8548.7 tok/s** (`1872.7`/req), -6.3% (mixed: reported) |
| 16k:1024 c8 decode aggregate | 210.65 tok/s (`32.11`/req), ITL 43.3/92.0 ms | 205.79 tok/s (`33.62`/req), ITL 43.8/87.3 ms |
| 16k:1024 c8 TTFT | 6.77 s | 8.75 s (cross-cell; c1 TTFT 3.48 -> 3.53 s) |
| stall probe 16k inject p99 | 532.97 ms (max 533-535) | **210.7 ms** (max 211), -60% |
| stall probe periodic 4k p99 | 194.8-196.3 ms | 98.1-100.3 ms |
| stall probe busy gap p99 | 129.7 ms | 88.1 ms |
| stall probe min decode tokens (periodic 4k) | n/a | 27-33 (gate wants 100; the 16k minimum applies to 4k windows) |

- Config audit of every M2 arm: model `Qwen3.5-0.8B`, `tensor_parallel_size=1`,
  `num_spec_tokens=2`, `enforce_eager=False`, `FULL_AND_PIECEWISE`; all c1/c8
  pairs below are the same TP1/MTP2/F&P config. `serve-m2fix2`: 9.82/10.40 c1,
  25.43/26.66 c8; `serve-m2spread`: 9.79/10.39 c1, 25.16/27.45 c8;
  `serve-m2-tp1-off-handoff`/`-on`: 9.69/10.29 c1, 23.83/25.66 c8.
- P lane activity bug (fixed in `548d90cd40`): `_lane_schedule_work(P)`
  counted *any* unfinished request in `waiting`/`kv_holding_waiting` as P
  work. A decoded/handoff request can stay there while its D batch runs, so
  `partitioned` stayed true: the 16k:1024 c1 decode measured 44%
  partitioned and 147.10 -> 124.47 tok/s (-15%). The filter now counts only
  a P-lane request with prompt tokens left (`num_computed_tokens <
  num_prompt_tokens`). Exact-SHA pair `vllm-exact-548` (`548d90cd40`),
  HIP 2, TP1/MTP2/F&P, 16k:1024 c1: off 224.56 tok/s, ITL 12.5 ms; on
  220.36 tok/s, ITL 13.1 ms (-1.87% throughput, +4.8% ITL); lane stats show
  `partitioned` down from 44% to 4-10% (the residual is the in-flight
  prefill batch), unit test `test_lane_activity_ignores_residency_and_release_only_batches`
  extended with a D-lane waiting request. Throughput now passes 2% for that
  cell; the remaining +0.6 ms/step ITL was attributed to the lane-thread
  handoff (superseded: it was on the GPU, see "TP1 decode-only +0.6 ms").
- Current tree (HEAD `e5ac50cf59`), HIP 2 single GPU, same TP1/MTP2/F&P,
  1024:512 c1/c8: off 9.8/23.9 ms ITL (`301.97`/`110.42` decode tok/s),
  on 10.4/24.5 ms (`284.75`/`107.55`). The +0.6 ms is the same in c1 and
  c8. It was read as per-step lane machinery; it is a GPU dispatch delay
  after long kernels, which c1 and c8 steps have in equal number (see "TP1
  decode-only +0.6 ms").
- Decode-only: the flag-on arms are **slower** than flag off (c1 -5.7%,
  c8 -4.6%; TwoLaneM4's latest profile reports c8 -8.5%), so **G3 FAILs**
  (the gate is 2%); the miss is 3.7-6.5 points. Only same-day boot pairs
  are comparable.
- Mixed prefill: -6.3% aggregate; this cell has **no pass/fail**
  (the contract asks to report output tok/s, ITL p50/p99 and TTFT against
  flag off; expected: ITL p99 collapses, TTFT rises). The small-D c1/c8 cells
  are reported too (-6.2% and +4.7% decode throughput, the mixed TTFT
  +29%). The adaptive split is what closes the 2701 -> 8548 tok/s gap.
- Prefill-only (16k:128 c1): TTFT 3.48 -> 3.53 s (+1.4%, G3 gate 5%
  PASS).
- Stall max gap: the p99 drops from 533 to 211 ms (-60%), which is the
  product gate; the strict `< 100 ms` max gap misses at P activation
  (211 ms), and the periodic windows sit at 98.1-100.5 ms (0.3-0.5 ms
  over). The first reconcile at P activation is the documented transient
  (the contract bounds it at 110 ms; this boot measured 211 ms, so the
  transient bound should be rechecked). The 4k `min_decode_tokens` FAILs are
  the 16k minimum applied to 4k windows (contract 4k minimum is 30), not
  acceptance misses.
- The PPL 14.73, prefix-cache PASS, 3/3 probes PASS, and warm TTFT 0.18 s
  vs cold 2.38 s agree with the flag-off arm. The `_rocm_C.abi3.so` in the
  fresh tree must be copied from the built tree (rsync excludes it), else
  model init fails with `input and z must be 2D`.
- The 16k:1024 c8 TTFT grows 6.77 -> 8.75 s while its throughput recovers
  9124.9 -> 8548.7 tok/s; the c1 TTFT stays flat (3.48 -> 3.53 s).
- The stall probe logs `STALL PROBE FAIL (rc=1)` because of the strict max
  gap / min-decode gates above; it is not a crash or a hang.
- Decode-only overhead: the D_full stream is already a plain
  `torch.cuda.Stream`, so the remaining 4.6-5.7% was read as lane machinery
  (superseded, see "TP1 decode-only +0.6 ms"). A
  baseline-`step` passthrough for P-idle iterations was tried and rejected:
  the lane requests' `execute_model`/`sample_tokens` futures are lane
  futures, the baseline `step` handed a `Future` to
  `scheduler.update_from_output`, and the server died
  (`'Future' object has no attribute 'sampled_token_ids'`) at the first c1
  decode. A safe zero-overhead path needs the lane futures integrated into
  the baseline step; that is not a one-line change, so the shipped
  configuration is `022ee2d31c` with the honest decode-only miss above.
- py-spy profile (c1 decode, flag off/on, 20 s, engine + lane threads):
  on `lane-decode-output` spends 319 samples (`=1.6 s`) in
  `Event.synchronize` (`torch/cuda/streams.py:254`) versus 41 samples
  (`=0.2 s`) on the baseline engine. Per 2,480 decode batches that is
  0.65 ms/step, which matches the c1 9.8 -> 10.4 ms gap. `lane-decode` is
  idle 3,135 of 3,998 samples (wait on `queue.get`), `lane-prefill` is idle
  3,888/3,998 (no spin per step), and the engine main thread is active
  5.0 s on versus 6.1 s off. The source is therefore the lane output thread's
  per-batch `copy_event.synchronize()` in `AsyncModelRunnerOutput.get_output`,
  not lane polling or an idle P thread.
- The first profile attempt started before the bench client's ~10 s warm-up
  and sampled only idle threads; the useful window is the 20 s one that
  overlaps the decode phase.
- A second zero-overhead attempt (a lean `scheduler.schedule` +
  `execute_model`/`sample_tokens` without lane queues) also fails: the
  lane-scheduled D request is not in the baseline worker's request-state
  table, so `req_states.add_request` asserts `No free indices` at the first
  c1 decode. Both attempts are reverted; removing the decode-only overhead
  needs the lane request states/futures re-keyed into the baseline step,
  not a local fast path.
- The focused inject after the large-batch adaptive split (D_full/P_B)
  measured 217 ms p99, worse than the 211 ms with both lanes partitioned
  (D_A/P_B), so that variant was reverted; inject stays on the 20/16 split.

### Cudagraph-mode check for the +0.6 ms (2026-11-04)

The compiled lane driver's BATCHDESC diagnostic answers whether the
c1/c8 `+0.6 ms/step` could be a decode batch falling back from FULL to
PIECEWISE. Two identical compiled runs on `~/vllm-two-lane`
(`accept/diag1025f`, 17:19, and `accept/diag1025g`, 17:27; both
`--lengths 1025 --decode-steps 2 --compile`, `VLLM_RDNA_LANE_DIAG=1`,
HIP 9) dispatch decode-only batches as **FULL in both** (f 10 FULL, g 8
FULL; g's two extra PIECEWISE batches are prefill rows). The 1-token
prefill tail is `cg FULL` in f and `cg PIECEWISE` in g, i.e. the fix
(2) change is exactly what moved it. So the decode-only regression is
not a FULL->PIECEWISE fallback: decode stays on the FULL replay path in
both arms. The source this check assumed, the lane output thread's
per-batch `copy_event.synchronize()` (py-spy, above), was the symptom of a
longer GPU step, not its cause; see the next section. Server
`serve.log`s carry no BATCHDESC (the diagnostic is lane-driver only),
so server-side mode is inferred from the engine's default FULL decode
dispatch; the production-stack MTP2 check likewise measured no target
prefill FULL replay (target tail `PIECEWISE 3`).

### TP1 decode-only +0.6 ms: root cause (fixed in `cea25a084c`)

The cost was on the GPU. Flag on ran the same 770 kernels per decode step
with the same busy time. About 150 inter-kernel gaps per step doubled, each
right after a long kernel. Flag off keeps a cross-queue wait pending during
every step; lane mode had none.

- Thread census on `5bc92eeaf1` (c1 decode, `top -H` + py-spy) found no
  GIL spinner. The lane output thread runs at 99% CPU busy-waiting inside
  HSA on `copy_event`, without the GIL. A native thread at 100% exists in
  both arms. The engine main thread idles in `_wait_for_lane_wake`.
- Kernel trace (rocprofv3 launch mode, `prof/proflane-off|on`, a tree
  before `5bc92eeaf1`, 60 decode steps):
    - Off: 12.02 ms per step, 8.46 ms busy, 3.56 ms idle.
    - On: 12.85 ms per step, 8.41 ms busy, 4.44 ms idle.
    - About 151 same-queue gaps per step grew from ~4.5 to ~9 us, for
      +0.86 ms per step under the tracer. Each one follows a long kernel
      such as `gemv_f16_rdna2_<8,3>` or `fa_decode_paged_splitk`.
    - The host had launched each step's graph 10+ ms ahead, so no launch
      was late.
- Microbenchmarks (`accept/tp1/qbench3.py`-`qbench5.py` on the box; HIP 2,
  `GPU_MAX_HW_QUEUES=2`): one graph of 100 x (reduction over a distinct
  12 MB weight + 3 tiny kernels), replayed back to back.
    - Alone: 5654-5664 us per replay.
    - With a second stream waiting on an event recorded after each replay:
      5234-5248 us, i.e. ~4.2 us per long kernel.
    - An event record alone, or an unrelated kernel on another stream:
      no change.
    - With 48 us spin kernels: 5707 vs 5278 us.
    - The waiting stream can be a pool stream, a full-mask stream or a
      `[0,20)` CU-masked stream. A `[20,36)` masked waiter made the replay
      slower (5938-6381 us), so the waiting queue matters.
- Flag off waits like this every step: `AsyncOutput` puts the D2H copies on
  a helper stream behind `copy_stream.wait_stream(main)`. In lane mode, the
  copies ran on the lane stream itself (I5), so no wait was pending on
  another queue.
- Fix: while D runs on `D_full`, its output copies go on `A`, behind a wait
  on `D_full`. A is D's own masked stream and idle then. Partitioned D and
  both P streams keep their copies on the current stream. No stream or KFD
  queue is added.
- Same-boot A/B (`5bc92eeaf1` + a runtime knob, HIP 2, 1024:512 c1):
    - Copies on A: 300.95 / 300.52 tok/s, ITL 9.8 ms.
    - Copies on `D_full`: 284.45-285.32 tok/s, ITL 10.4 ms.
    - A fresh pool stream as the copy stream: 284.74 / 285.02 tok/s, no
      gain. Why it does not help was not investigated.

Exact-SHA pairs, HIP 2, TP1, Qwen3.5-0.8B MTP2, compiled F&P, greedy, one
sequential off/on pair per tree (`accept/tp1/serve-p0-*`, `serve-p1-*`).
Decode tok/s is per request; c8 also shows the aggregate output.

| SHA | cell | off tok/s, ITL med/mean | on tok/s, ITL med/mean | on vs off |
|---|---|---|---|---|
| `5bc92eeaf1` | 1024:512 c1 | 301.97, 9.8 ms | 285.11, 10.4 ms | -5.6% |
| `5bc92eeaf1` | 1024:512 c8 | 116.29 (agg 825.45), 23.3/25.1 ms | 109.25 (agg 779.79), 23.8/26.4 ms | -6.1% |
| `5bc92eeaf1` | 16384:1024 c1 | 232.58, 12.4 ms | 222.11, 13.0 ms | -4.5% |
| `cea25a084c` | 1024:512 c1 | 302.01, 9.8 ms | 301.18, 9.8 ms | -0.3% |
| `cea25a084c` | 1024:512 c8 | 115.66 (agg 825.46), 23.2/25.3 ms | 112.55 (agg 803.55), 23.3/25.9 ms | -2.7% |
| `cea25a084c` | 16384:1024 c1 | 232.61, 12.4 ms | 231.86, 12.4 ms | -0.3% |

Both `cea25a084c` arms pass the probes, PPL 14.73 and the prefix check
(17.2x off, 14.8x on). CONCURRENCY passes in both: 45/48 identical off,
48/48 on. The KFD census stays at 6 queues in the process. The c8 cell is
not pure decode: 32 prompts of 1024 tokens arrive while 8 decode. Its split
follows.

#### c8 split: pure decode vs prefill phases (`a9a4d60895`)

One more sequential pair on HIP 2 (same engine code as `cea25a084c`;
`--save-detailed` bench results; lane trace on the flag-on arm;
`accept/tp1/serve-p2-*`):

| cell | off tok/s/req, ITL med/mean | on tok/s/req, ITL med/mean | on vs off |
|---|---|---|---|
| 1024:512 c8, n=32 | 115.67, 23.2/25.3 ms | 113.80, 23.2/25.6 ms | -1.6% |
| 1024:4096 c8, n=8 (pure decode) | 116.92, 25.5/25.4 ms | 116.88, 25.5/25.4 ms | -0.03% |

- Pure decode at c8 matches flag off. In the n=8 cell the 8 requests start
  together, so their prefills cover about 1% of the decode time.
- In the mixed cell the gap comes from the prefill phases. Each request's
  chunk times (start + TTFT + ITLs, `accept/tp1/itl_split.py`) split its
  ITLs into those that overlap another request's prefill window and the
  rest. TPOT goes from 8.645 to 8.788 ms (+0.142 ms):
    - ITLs that overlap a prefill: 413 with mean 53.3 ms off, 538 with
      mean 48.2 ms on, +0.090 ms of TPOT. Each such interval is shorter
      with the flag on, but the prefill phases last longer: TTFT mean 434
      -> 492 ms, p90 618 -> 776 ms.
    - ITLs with no prefill window open: mean 23.04 -> 23.21 ms, +0.052 ms
      of TPOT. The lane trace times steady full-GPU 8-request D batches at
      23.23 ms, the same as the off median ITL (23.20 ms). So this part is
      likely the tail of the prefill phases, not decode speed.
- Lane trace of the flag-on c8 window (20.1 s):
    - 9 prefill phases put D on A for 4.59 s: 96 of 791 D batches, 22.9% of
      D time.
    - Those batches average 48.0 ms with 5.0 requests, against 22.3 ms for
      full-GPU batches.
    - All 52 P batches ran on P_full, the adaptive split's small-D case, so
      P shares A's 20 WGPs with D.
    - P batches (mean 841 tokens) take a median 64 ms of host launch,
      62% of the 104 ms median between P submissions. They run the compiled
      model without graphs, as flag off does at that size (next section).
- The `cea25a084c` pair measured -2.7% on the same mixed cell. Its flag-on
  result moves between boots (112.55 and 113.80 tok/s/req); flag off holds
  at 115.66-115.67.

### What lane P runs (graphs for small batches fixed in `585f1ef7eb`)

A probe build recorded, for each forward, the cudagraph mode, the compiled
callable it entered and its host and GPU time. The build was `ced95c58ce`
plus an uncommitted probe, with one boot per arm, 16 prompts of 841 tokens
at c1 and the 1024:512 c8 cell (`accept/tp1/diag-p3-*`, `serve-p3-*`).

- Both arms run an 841-token prompt as chunks of 544, 296 and 1 tokens.
- The 544- and 296-token chunks take the same path in both arms. They are
  above the largest capture size (24), so they run with cudagraph mode
  NONE inside the compiled `Qwen3_5Model` callable. They are not eager.
    - c1: host forward 28.8/28.3 ms off and 29.8/29.4 ms on; GPU forward
      56.6/48.6 ms off and 55.8/49.2 ms on.
    - c8, all prefill chunks: host median 33.4 ms off and 58.5 ms on (p90
      67.7 ms); GPU 65 ms off and 87.5 ms on. The compile path is the same,
      so the extra time is load (D's batches on the other partition, the
      GIL), not the compile mode.
- The bug: lane P never captured graphs. Its batches of 1-24 tokens (chunk
  tails, short prompts) ran NONE where flag off replays PIECEWISE. At c1
  the 1-token tail took 18.7 ms of host time on P, against 9.3 ms host and
  5.3 ms GPU off. At c8 it took 33.8 ms.

The fix gives P the graphs that flag off has for its shapes:

- P's model uses the mixed routine of D's resolved mode, PIECEWISE under
  F&P. A P batch always has a prefill row, so FULL never dispatches for it.
  P's speculator uses D's mode, as it does with flag off.
- P captures before D, because D's capture locks the workspace.
    - It runs on the main thread under lane P's scope.
    - It captures on `P_capture`, a plain stream of its own. A CU-masked
      stream cannot host a capture. Until `4ff396340a` it captured on
      `D_full`, D's capture stream, which corrupted TP4 Flash-Next (see
      below).
    - The graphs go into P's own graph pool.
- The shared piecewise wrappers key P's entries by lane. A P batch refuses
  to replay a graph outside lane P's scope.
- P gets no graphs with the breakable runner or the sleep-mode cuMem graph
  pool: there is only one of each per process.
- At TP1, P's capture takes 1.2-1.3 s and 0.34 GiB. The KFD census stays at
  6 queues. For TP4, see "Flash-Next TP4 corruption from lane P's graphs".

The same probe on `da62507058` (`diag-p4-on`, `serve-p4d-on`):

- All 72 P batches of 1-24 tokens dispatch PIECEWISE.
- The 1-token tail now takes 10.5 ms of host time at c1 (was 18.7 ms; off
  9.3 ms) and 18.2 ms at c8 (was 33.8 ms).
- The 841-token c1 TTFT under the probe is 155.3 ms. It was 160.1 ms
  before the fix; off is 138.0 ms. The rest of the gap is not the compile
  mode.

Exact-SHA boots on HIP 2 used the same TP1 recipe as above, with
`--save-detailed` and the cells in this order. They ran one after another,
with the probe boot fourth (`accept/tp1/serve-p4-*`, `serve-p5-*`). The
pre-fix row is `a9a4d60895`, the same engine code as `ced95c58ce`.

| boot | SHA | 1024:512 c8 tok/s/req (agg) | c8 TTFT mean/med/p99 | 16k:128 c1 TTFT mean | 16k:128 c1 tok/s |
|---|---|---|---|---|---|
| 1 off | `da62507058` | 116.42 (822.11) | 446/471/752 ms | 3458 ms | 216.41 |
| 2 on | `da62507058` | 112.60 (802.45) | 421/361/851 ms | 3474 ms | 211.01 |
| 3 on | `a9a4d60895` | 112.28 (794.57) | 465/435/877 ms | 3522 ms | 211.60 |
| 5 on | `da62507058` | 112.74 (796.21) | 454/399/894 ms | 3488 ms | 210.13 |
| 6 off | `da62507058` | 116.70 (821.58) | 450/482/746 ms | 3452 ms | 216.27 |

- 16k:128 c1 TTFT: on is +0.5% and +1.0% over the off boots (3474 and 3488
  ms against 3458 and 3452 ms). The pre-fix boot was +1.9% (3522 ms).
- c8 TTFT: with the flag on, the median is lower (361 and 399 ms against
  471 and 482 ms) and p99 is higher (851 and 894 ms against 752 and 746 ms).
  The pre-fix boot was 465/435/877 ms.
- c8 decode is -3.3% and -3.4% per request, with the pre-fix boot in the
  same range (112.28). The fix does not move it; the c8 split above puts
  that gap in the prefill phases.
- In the probe boots c8 ran after the 841-token c1 cell, and both arms
  measured higher there: off 118.67, on 115.40 before the fix and 116.35
  after. Compare only boots that run the same cells in the same order.
- All arms pass the probes, PPL and prefix checks.

### M2 rerun with real blocks and the handoff/zero fixes (2026-11-03 22:15-22:41)

HIP 3, Qwen3.5-0.8B + MTP2, TP1, 32k, KV 4e9, SEQS 8, compiled F&P,
cells `1024:512:1 1024:512:8 16384:1024:1 16384:1024:8`. Flag on and flag
off are separate boots. The table above predates the real-block handoff fix;
these numbers replace its speed cells.

| cell | flag off | flag on |
|---|---|---|
| correctness / probe | PASS | PASS |
| PPL | 14.73 PASS | 14.73 PASS |
| prefix | PASS | PASS (TTFT 2.37/0.25 s) |
| decode-only c1 (`1024:512:1`) | 157.05 tok/s, ITL 9.8 ms | 156.89 tok/s, ITL 10.4 ms |
| decode-only c8 (`1024:512:8`) | 65.27 tok/s, ITL 23.9/25.2 ms | 64.34 tok/s, ITL 24.6/27.5 ms |
| 16k:1024 c1 | 147.10 tok/s, TTFT 3.54 s, ITL 12.5 ms | 124.47 tok/s, TTFT 3.58 s, ITL 13.1 ms |
| 16k:1024 c8 | 22.87 tok/s (`157.20`/req), TTFT 5.85 s, ITL 44.7/78.8 ms | 24.35 tok/s (`168.38`/req), TTFT 5.04 s, ITL 45.2/78.8 ms |

Verdict: decode-only is **not** fixed. The ITL delta is +0.6 ms/step
(c1 10.4 vs 9.8; c8 24.6 vs 23.9), the same +0.6 ms/step TwoLaneM4 measured;
per-request tok/s hides it because c1/c8 aggregate differently. The 16k c1
arm is **-15% flag-on** (124.47 vs 147.10), also open. TwoLaneM4 owns the
decode only perf fix. PPL repeated-span WARNs appear in **both** arms, so
they are baseline degeneration, not a two-lane correctness bug.
`tools/rdna/port_v031/correctness_diff.py` compares `serve-m2fix2-off` and
`-on` directly: probe 3/3 outputs identical, PPL continuation identical, PPL
14.73 identical, prefix verdicts identical (the prefix text is a time-based
nonce and is skipped). Concurrency was not run in that pair, so it is not
measured; the earlier HIP 5 hook was 8/8. M2 correctness parity **passes**
for every text output the pair produced.

Stall probe (16k inject, 4 decoders, `serve-m2fix2`): flag off max 196.79 ms,
p99 196.29 ms; flag on max 104.41 ms, p99 98.11 ms. The on arm is 2x better
but still misses the strict 100 ms max and starves decode to 27-28.5 tokens
per prefill window (gate 100), so M2 G2 **fails**. M3 TP2 27B was queued
behind TwoLaneFeat's PP2 gate on HIP 4+5 and died in worker init because the
GPU pair was shared; it has no verdict yet.

## M3 TP2 27B / M4 TP4 Flash-Next

Serialized sweep `accept/tp_sweep.sh` (one boot at a time, HIP 6-9; logs
`accept/m3-<arm>.log`, `accept/m4-<arm>.log`; done marker
`accept/tp-sweep.done`). Both arms of each milestone use the same recipe on
the same tree (`~/vllm-two-lane-head`).

M3 TP2 Qwen3.8-27B AWQ-INT4, HIP 6,7, compiled F&P, `MAXLEN=32768 KV=6e9`,
D/P split 20/16, greedy. Flag off (measured): correctness PASS, PPL 4.90 PASS,
prefix PASS (TTFT 5.63 s cold / 27.82 s at 16k, solo); 16k inject max stall
2210 ms (p99 2210, 19.5 decode tokens), periodic 4k 1513 ms; bench c1
745.3 tok/s prefill / 30.66 decode / TTFT 1.37 s / ITL 32.6 ms.
The S7 gate off arm (`accept/m3gate`, `two_lane_gate.sh`) confirms it:
PPL 4.90, probe/prefix PASS (cold 19.77 s, warm 0.81 s), concurrency 44/48
identical, 0 failing / 4 near-tie; 16k inject 2201.5 ms, periodic 4k
1515.6 ms, min decode tokens 18-28; bench c1 738.4 prefill / 30.61 decode /
ITL 32.7 ms. Flag on and the G2 4k inject follow in this sweep; numbers are
appended here when the arms land.

TP2 flag-on crash (root cause, fixed): the engine died right after READY in
`_lane_activity` -> `_lane_schedule_work`, which iterated the lane queue
(items are `_LaneBatch`) and read `total_num_scheduled_tokens` on the batch
instead of `batch.scheduler_output.total_num_scheduled_tokens`
(AttributeError). The getter runs before scheduling, so any queued pending
batch hit it. Fix: read through `batch.scheduler_output` (core.py). A second
TP2-only failure follows it: the D lane's `sample_tokens` future resolves
`None` in the engine while both worker ranks return `AsyncOutput` /
`ModelRunnerOutput` (lane-diag, 20:04); the lane RPC response order is the
suspect (execute and sample futures are matched FIFO). Open; M3 flag-on is
not green until it is fixed.

M4 TP4 Flash-Next AWQ-W4A16 MTP2, HIP 6-9, compiled F&P, capturables
[3,6,12,24], `MAXBAT=2048`, greedy. Arms: flag off, then
`LANE_DECODE_WGPS` 16, 18, 20, 24 (so both 16/20 and 20/16). Cells
`1024:512:1 1024:512:8 16384:128:1 16384:1024:8:32` (G3: decode-only,
prefill-only, mixed), `STALL_ARGS` 16k inject with 6 decoders. The
feature-OFF split-16 arm also runs `probe_aux_output.py --capture-ref`
(`LOAD_HOOK`); a queued aux arm (`EXTRA_ARGS=--enable-return-routed-experts`,
`serve-m4-aux16`) compares against it after the sweep.

### M4 TP4 decode slowdown (root cause, fixed in `5bc92eeaf1`)

On `8c3297203e` at split 20, flag-on TP4 c1 decode ran 34.39 tok/s/req with
ITL 53.8 ms (`serve-m4clean-tr20`); flag off ran 81.45 tok/s/req with ITL
24.5 ms. A lane-trace boot (`VLLM_RDNA_LANE_TRACE=1`, per-rank worker
traces) showed the cause. In slow phases, one rank at a time started each D
batch 30-75 ms after submit. Its `execute_model` launch took 20-38 ms, against
9-11 ms on the other ranks. The TP collectives held every rank to that pace,
so the c1 step period doubled (49-65 ms against 24.5-25 ms). The lagging rank
changed over time.

The worker main thread only hands lane RPCs to the lane threads. Its
`rpc_broadcast_mq` reader still busy-polled in Python for 1 s after every
read: a `sched_yield` loop at 79% CPU on rank 0 (`top -H`) that takes the GIL
on every spin, so the lane thread lost the GIL handoff. The engine's
`LaneOutputReader` threads spun the same way (97% CPU).

Fix: with the flag on, those readers block on the writer's notification
(`MessageQueue.set_reader_busy_loop(0)`). A boot after the fix gave:

- c1 decode gap median 24.6-24.9 ms in every phase (before: 49.0, 24.9,
  58.9, 25.0 and 53.5 ms).
- Every rank starts a D batch 0.6-0.9 ms after submit, with 5-9 ms launches.
- The worker main thread runs at 1.7% CPU.

Bench on `5bc92eeaf1` in one session, flag on (split 20) vs flag off:

- c1: 84.57 vs 81.57 tok/s/req, ITL 24.7 vs 24.2 ms.
- c8: aggregate output 246.07 vs 251.77, ITL median/mean 48.2/56.1 vs
  47.1/57.4 ms.

Lane-wake poll: `_wait_for_lane_wake` waits on an event with a 50 ms timeout.
Every hot-path wake source sets the event: lane future done-callbacks, input
arrivals in `process_input_sockets`, add/abort and executor failure. The
timeout only bounds wakes that reach the scheduler without an event, such as
a grammar compile. On the same boot, across 1995 lane submits, submit after
the latest completion was p50 0.64 ms and p99 7.2 ms, with none at 40-60 ms.
No hot-path wait is released by the timeout.

### Flash-Next MTP hidden buffer on lane P (fixed in `700f1b527d`)

`Qwen4ExpModel` keeps one MTP hidden buffer per lane. Its forward writes the
pre-final-mixer hidden state there, and the lane's drafter reads it. Before
`700f1b527d`, the forward picked the buffer with `current_lane()` inside the
compiled region. vLLM drops Dynamo's guards, so that read was traced once, on
lane D. Lane P's forward wrote lane D's buffer, and lane P's buffer was never
written.

A compiled repro on HIP 2 called one `support_torch_compile` module under lane
D, then under lane P:

- vLLM compile, cudagraph NONE and PIECEWISE: one graph. Lane P's rows landed
  in buffer 0, and buffer 1 stayed unwritten.
- Plain `torch.compile`, which keeps the guards: two graphs, and each lane
  wrote its own buffer.

So with the flag on, Flash-Next's MTP drafter on lane P read an unwritten
buffer for every prefill chunk, and lane P's writes raced lane D's drafter
reads. Greedy outputs stay correct because the target verifies every draft.
The MTP acceptance and decode speed of the flag-on M4 runs before
`700f1b527d` are affected by an amount not yet measured. Models without this
buffer (Qwen3.5 at TP1) feed the drafter the forward's returned hidden
states, which are per lane already.

Fix: the uncompiled callers (`Qwen4ExpForCausalLM.forward` and the VL
wrapper) pick the calling lane's buffer and pass it to forward as
`mtp_hidden_buffer`, a graph input. Flag off writes its one buffer with the
same ops. The other `current_lane()` reads on the Flash-Next path (PLE
offload, pinned PLE prefetch, routed-experts capture, GDN) sit in custom-op
bodies or in the runner, which run eagerly.

### Flash-Next TP4 corruption from lane P's graphs (fixed in `4ff396340a`)

From `585f1ef7eb` on, where lane P captures graphs, TP4 Flash-Next broke with
the flag on. The concurrency probe failed with HTTP 400s (NaN logprobs), and
MTP acceptance fell to zero. `cea25a084c`, before P's graphs, was clean. The
arms below ran the `flashnext-mtp2-twolane.env` recipe on HIP 6-9, flag on,
split 20 with adaptive on, one boot each (`serve-tp4bis-*`). Each diag arm is
`00cccc8dfb` plus one uncommitted change.

| arm | concurrency (48 checks) | HTTP 400 | intervals with 0 accepted | c8 acceptance |
|---|---|---|---|---|
| `cea25a084c` | PASS, 0 failing | 0 | 0 of 32 | 0.594 |
| no P model graphs (`noPmodelCG`) | PASS, 0 failing | 0 | 0 of 28 | 0.651 |
| no P draft graphs (`noPspecCG`) | FAIL, 18 failing | 18 | 14 of 34 | 0.0004 |
| P skips the MoE layer scratch (`PnoMoEscratch`) | FAIL, 19 failing | 19 | 16 of 36 | 0.0003 |
| that plus P's own QSA top-k rows (`PnoMoEnoQSAbuf`) | FAIL, 19 failing | 19 | 15 of 34 | 0.0004 |
| P captures on a fresh plain stream (`PcapStream`) | PASS, 0 failing | 0 | 0 of 28 | 0.626 |

P's model graphs carried the fault, not its draft graphs. A
`VLLM_DEBUG_WORKSPACE=1` boot ruled out the shared workspace: every lane P
resize happened before P's capture, and the lock held at 200 MB.

The cause is the arrival tickets of the RDNA2 split-K GEMMs. These are the
`rdna_hc_mix_prefill` down GEMM, `rdna_gemm_smallm` (M 9-32 under the
recipe's `VLLM_RDNA_SMALLM_GEMM_MAX_BYTES`) and the fused shared-expert tiles
(`VLLM_RDNA_FUSED_SE=1`). `wvSplitKrc` uses the same scheme.

- Each launch takes a slot of zeroed arrival counters (`hc_tickets` in
  `csrc/rocm/hc_prefill_rdna2.cu`).
- A captured launch takes the slot of its capture stream, so every graph
  captured on one stream shares one slot. That is safe only while those
  graphs never replay at the same time.
- Lane P captured on `D_full`, as lane D does, and the two lanes replay at
  the same time. Their K slices then counted on the same counters, so a
  reduction could run before all of its slices had landed.
- P's draft graphs captured on a fresh stream each (`graph_capture()` with no
  stream), so they had slots of their own. Eager P, before `585f1ef7eb`, used
  its lane stream's runtime slot.

Fix: each lane captures all of its graphs, model and drafts, on a plain
stream of its own. Lane D uses `D_full`, and lane P uses the new `P_capture`
(`lane_capture_stream`). A plain capture stream adds no KFD queue: the
`PcapStream` arm's census shows 6 queues, as the other arms do.

The same audit found two buffers that lane P wrote although they were lane
D's (S6). `82ecd60b9d` makes them per lane:

- The RDNA2 MoE methods' per-layer scratch. Lane P allocates per call, as
  the large-batch path already does.
- QSA's `topk_indices_buffer`, which the indexer writes and the sparse
  attention and the MTP row compaction read. Lane mode adds a lane P buffer of
  the same shape. At TP4 Flash-Next with `MAXBAT=2048` that is 12 attention
  layers plus the MTP layer, about 208 MiB per rank. Model loading took 20.81
  GiB per rank on `82ecd60b9d`, against 20.6 GiB on the `PcapStream` arm. The
  recipe fixes the KV budget (`KV=4026531840`), so the KV cache stays at
  185,908 tokens.

Neither buffer moved this fault, as the table shows. The drafter's
`indexer.skip_topk`, set by `set_skip_topk` when the draft config has
`index_share_for_mtp_iteration`, is now one flag per lane
(`test_mtp_skip_topk_is_per_lane` in `tests/models/test_lane_model_isolation.py`).
Flash-Next does not set that key, so the flag stays off for both lanes here.

Exact-SHA check on `82ecd60b9d`, same recipe and arm
(`serve-tp4sw-82ecd60b9d-a1-20`):

- Probes, PPL (4.82), prefix and the after-load probe PASS.
- Concurrency PASS with 0 of 48 failing. No HTTP 400, and 0 of 85 intervals
  with 0 accepted.
- Acceptance c1 0.602 and c8 0.630. Flag off on `00cccc8dfb` gave 0.638 and
  0.628.
- The census shows 6 KFD queues. Lane P's capture took 0.29 GiB on three
  ranks and 0.44 GiB on one.

### Lane env aliases on the `vllm serve` path (fixed in `2c62da9966`)

Before `2c62da9966`, `vllm serve` ignored every `VLLM_RDNA_TWO_LANE` /
`VLLM_RDNA_LANE_*` variable that also has a CLI flag: `two_lane`,
`decode_wgps`, `hwq`, `prefill_budget`, both depths, `solo_full` and
`adaptive`. The parser took each flag's default from `SchedulerConfig`, which
overrode the env-derived `EngineArgs` default. Now the flags default to the
env value, and an explicit flag still wins. With no env set the defaults are
unchanged, so flag off and the recipes are unaffected.

Audit of the runs on the box:

- `serve_rdna.sh` passes `TWO_LANE`, `LANE_DECODE_WGPS`, `LANE_HWQ`,
  `LANE_PREFILL_BUDGET` and `LANE_PREFILL_DEPTH` as flags, so those launcher
  knobs were applied. The stream masks in each `serve.log` confirm the split
  for the old M4 16/18/20/24 sweep (`A` = `[0,16)` to `[0,24)`) and for every
  other split run on the box. No split number needs a "split not applied"
  mark.
- Not applied: values set only through the env.
    - `VLLM_RDNA_LANE_ADAPTIVE=0` in the first TP4 sweep: the
    `serve-tp4sw-5bc92eeaf1-a0-*` arms ran `adaptive=True`, so their numbers
    are adaptive-on at that split.
    - `VLLM_RDNA_LANE_PREFILL_BUDGET` exported by `accept/m4_cap.sh`: no cap
    was applied. No serve directory from it remains.
- No log under `~/w4a8_runs` shows a non-default `two_lane_hwq`, or the
  monokernel's `GPU_MAX_HW_QUEUES is pinned to two_lane_hwq` line. No
  monokernel-plus-lane GPU run exists to re-mark.

### TP2 35B lane D conv1d fault (root cause, fixed in `54702c7460`)

Qwen3.6-35B-A3B GPTQ-Int4 (hybrid GDN + MoE), TP2, MTP0, `MAXLEN=4096`,
mamba `align`, compiled F&P (`flashnext-mtp2-twolane` with `TP=2 MTP=0
MAXLEN=4096 KV=2000000000`): flag on died at the first probe with a memory
fault in `_causal_conv1d_update_kernel`. Flag off was clean.

Root cause: lane mode keeps RDNA decodes on the paged GDN state (no arenas),
so `GDNAttentionMetadataBuilder.build()` gave lane D's FULL decode graphs
`block_table_tensor[:, 0]`. In align mode `block_table_tensor` is the output
of `mamba_get_block_table_tensor()`, a new tensor every step, so each graph
replayed the address it saw at capture. An instrumented boot
(`accept/tl1/dbg-unfixed`: `8ed120d9b6` plus a debug print patch) shows it on
one rank:

- The size-1 FULL capture bound the three GDN groups' indices at `…0a00`,
  `…0c00` and `…0800`. The builder's static buffer was not used.
- The first D decode built them at `…0800`, `…0a00` and `…0c00` (values 1,
  2, 3). The graph read each group's index from another group's tensor:
  wrong state, no fault.
- The second D decode built them at `…2e00`, `…0000` and `…0200`. The graph
  read freed memory and faulted at `0x7ef2fe605000`, 104 MiB past the end of
  the rank's KV pool (slot ~147 of 95).

MTP runs never hit this, because spec decodes were already staged in static
buffers.

Fix: in lane mode the builder copies the decode state indices and query start
locations into its static buffers, as the arena path does. Flag off runs the
same ops as before. The unit test
`test_gdn_metadata_builder.py::test_lane_mode_decode_indices_live_in_static_buffer`
fails on `8ed120d9b6` and passes on `54702c7460`. The instrumented fixed boot
(`accept/tl1/dbg-fixed`) captured only the six static buffers (3 groups x 2
ranks), and all 3654 D builds used them.

GPU proof on `54702c7460`, HIP 6,7, same config, greedy bench
(`accept/tl1/proof-54702c7460-{on,off}`; repeats without bench cells in
`accept/tl1/rep-54702c7460-{on,off}`):

| | flag on | flag off |
| --- | --- | --- |
| probes, after-load probe | PASS, PASS | PASS, PASS |
| PPL (repeat) | 14.19 (14.18) | 14.24 (14.25) |
| concurrency, failing of 48 (repeat) | 0 (0) | 0 (0) |
| 1024:512 c1 decode, ITL | 68.68 tok/s/req, 14.5 ms | 61.30 tok/s/req, 16.3 ms |
| 1024:512 c8 decode (agg), ITL | 45.77 (335.3), 20.4 ms | 40.28 (298.6), 23.0 ms |
| 3072:512 c4 decode (agg), TTFT | 48.35 (170.6), 1.28 s | 59.52 (54.8), 25.29 s |
| fault markers, dmesg faults, eviction ms | 0, 0, 0 | 0, 0, 0 |

The flag-off 3072:512 cell is not a lane comparison. It ran one request at a
time (`Running: 1 reqs, Waiting: 3 reqs` at 5.3% KV). The engine warns that
`max_num_batched_tokens=2048 cannot cover two mamba state blocks of 2048
tokens`. The stock prefix probe (12000 words) exceeds `MAXLEN=4096`, so it
gets HTTP 400 in both arms. At 3600 words (`LOAD_HOOK`) both arms hit the
cache alike: cold 0.96 / 0.99 s, warm 0.50 / 0.52 s (off / on). That is just
above the probe's 0.5 ratio gate.

### M4 split sweep: static split vs adaptive split (2026-11-04)

Serialized sweep, one boot per arm, on HIP 6-9 (`accept/tp4/sweep.sh`).
Driver logs are `accept/tp4/sweep-<sha>.log` and serve dirs are
`serve-tp4sw-<sha10>-<arm>`. Recipe `flashnext-mtp2-twolane`, compiled F&P,
MTP2, `MAXBAT=2048`, greedy. Each arm runs the full `serve_validate.sh` set:

- probe, PPL and prefix;
- the stall probe: 6 decoders, 16k inject, periodic 4k, busy and reverse
  phases. Gates: p99 < 100 ms, worst < 150 ms, at least 100 decode tokens
  per prefill window;
- the concurrency probe;
- the cells `1024:512:1 1024:512:8 16384:128:1 16384:1024:8:32`.

Arm `aN-S` means `VLLM_RDNA_LANE_ADAPTIVE=N` with `LANE_DECODE_WGPS=S`. The
`A` mask and the `adaptive` value in the tables are read from each arm's
`serve.log`. Every arm passed probe, PPL (4.80-4.87), prefix, concurrency and
the after-load probe, with no faults.

`2c62da9966`, adaptive off, in one session:

| arm | `A`, adaptive | inject max/p99 ms | periodic 4k p99 ms | c1 ITL ms | c8 agg out tok/s | 16k:1024 c8 prefill agg tok/s, TTFT s | stall gate |
|---|---|---|---|---|---|---|---|
| off | flag off | 540/540 | 526-530 | 23.6 | 256.26 | 3936.7, 11.20 | FAIL |
| a0-20 | `[0,20)`, off | 86/72 | 67-80 | 24.7 | 251.23 | 1403.7 (-64%), 70.78 | PASS |
| a0-16 | `[0,16)`, off | 81/80 | 77-82 | 24.8 | 261.90 | 1424.3 (-64%), 66.74 | FAIL: 88-97.5 decode tokens per window |
| a0-12 | `[0,12)`, off | 102/100 | 95-96 | 24.7 | 242.23 | 1717.4 (-56%), 53.61 | FAIL |
| a0-8 | `[0,8)`, off | 133/132 | 125-128 | 24.7 | 233.35 | 3156.7 (-20%), 24.70 | FAIL |
| a0-6 | `[0,6)`, off | 206/205 | 198-203 | 24.7 | 168.60 | 3346.5 (-15%), 15.80 | FAIL |

`5bc92eeaf1`, adaptive on, in one session. This sweep's `a0-*` dirs ran with
adaptive on (see the env-alias section above), so they are adaptive-on arms
at their split:

| arm (dir) | `A`, adaptive | inject max/p99 ms | periodic 4k p99 ms | c1 ITL ms | c8 agg out tok/s | 16k:1024 c8 prefill agg tok/s, TTFT s | stall gate |
|---|---|---|---|---|---|---|---|
| off | flag off | 539/539 | 529 | 24.2 | 251.77 | 3967.8, 12.62 | FAIL |
| a1-20 | `[0,20)`, on | 205/205 | 188-205 | 24.7 | 246.07 | 3572.1 (-10.0%), 13.34 | FAIL |
| a1-20 (`a0-20`) | `[0,20)`, on | 201/200 | 191-205 | 24.7 | 249.11 | 3603.1 (-9.2%), 16.84 | FAIL |
| a1-16 (`a0-16`) | `[0,16)`, on | 224/224 | 202-208 | 24.7 | 253.96 | 3617.1 (-8.8%), 19.13 | FAIL |
| a1-12 (`a0-12`) | `[0,12)`, on | 257/257 | 236-255 | 24.8 | 249.74 | 3662.9 (-7.7%), 16.66 | FAIL |
| a1-8 (`a0-8`) | `[0,8)`, on | 312/312 | 280-289 | 24.6 | 246.91 | not valid (see below) | FAIL |

What the tables show:

- At TP4, only the static split passes the stall gate: `a0-20` has an inject
  p99 of 72 ms. The prefill lane then keeps `B` through the whole mixed
  phase, and the 16k:1024 c8 prefill aggregate drops 64% (TTFT 11.2 s to
  70.8 s).
- With adaptive off, the prefill rate does not scale with P's WGPs. P got
  16 WGPs at split 20 and 30 at split 6, and the rate went 1404, 1424, 1717,
  3157, 3347 tok/s.
- Adaptive on keeps the prefill aggregate within 10%, but a D batch next to
  a P_full chunk waits about one chunk. The inject gaps are 205, 224, 257 and
  312 ms at splits 20, 16, 12 and 8.
- c1 ITL is 24.7 ms with the flag on. Flag off gave 23.6 ms on `2c62da9966`
  (+4.7%) and 24.2 ms on `5bc92eeaf1` (+2.1%). The two flag-off runs of the
  same flag-off code differ by 0.6 ms, so the ITL gate needs same-session
  pairs.

Verdict: no split passes every gate. On TP4 Flash-Next, static CU
partitioning cannot meet both the stall gate and the prefill gate. The arms
with an inject p99 at or under the 100 ms gate (splits 20, 16 and 12 with
adaptive off) lose 56-64% of the long-prompt prefill throughput. The arms that
stay within 20% of the prefill throughput have an inject p99 of 132-312 ms.
The recipe keeps split 20 with adaptive on as the throughput default.
`VLLM_RDNA_LANE_ADAPTIVE=0` is the latency option: it passes the stall gate at
split 20 and costs 64% of the 16k:1024 c8 prefill throughput.

Stream priority does not replace the partition (measured no-go). The idea was
to keep adaptive on but give D's streams a higher HIP priority than P's, so
that D's waves dispatch as soon as P's in-flight waves retire. A one-GPU
microbench (`accept/tp4/prio_bench.py`, HIP 9, `GPU_MAX_HW_QUEUES=2`) ran a
decode-like step of 81 small kernels on a stream H. Meanwhile a stream L kept
a backlog of about 4 s of 2048x4096x8192 fp16 GEMMs. HIP accepts priorities:
the range is least 1, greatest -1, and `torch.cuda.Stream.priority_range()`
gives `(0, -1)`. gfx1030's dispatch ignores them:

| H stream | L stream | H step alone ms | H step during L backlog p50/p95 ms |
|---|---|---|---|
| mask `[0,20)`, normal | mask `[20,36)`, normal | 7.0 | 8.5/8.6 |
| mask `[0,20)`, normal | mask `[0,36)`, normal | 7.0 | 33.5/33.8 (rerun 33.3) |
| mask `[0,20)`, normal | plain, least priority | 7.0 | 33.8/34.1 |
| plain, normal | mask `[0,36)`, normal | 4.1 | 28.9/29.1 |
| plain, greatest priority | mask `[0,36)`, normal | 4.1 | 29.2/29.4 (first run 30.2) |
| plain, greatest priority | plain, least priority | 4.1 | 30.5/30.8 |

Neither the high nor the low priority moves H's latency by more than 1%.
L's GEMM took 5.7-5.9 ms in every overlapped setup. Each of H's kernels waits
for L's in-flight waves on its CUs, so the cost scales with D's kernel count.
Only disjoint masks protect D's step, so the `a1-20` plus priority arm was not
run.

Runs excluded:

- `serve-tp4sw-2c62da9966-a1-20`: a second sweep driver started while the
  arm was running. It saw that server as READY and ran its own probes and
  benches against it from 06:56 to 07:07.
- `serve-tp4sw-da62507058-off`: run by that second driver, so it is invalid
  for the same reason.
- The 16k:1024 c8 cell of `serve-tp4sw-5bc92eeaf1-a0-8`: the second
  driver's VRAM check polled the GPUs during that cell.

The sweep scripts now take a lock and refuse to start on a busy port.

All of these runs predate `cea25a084c`, `585f1ef7eb` and `700f1b527d`. They
also predate KFD queue-eviction sampling (`evict_watch.py`, per-process
`evicted_ms`), so a window may overlap a neighbor's boot, which evicts this
server's queues for up to tens of seconds. The two adaptive-on split-20 arms
of `5bc92eeaf1` agree (inject 205 and 201 ms, prefill 3572 and 3603 tok/s).

### M4 post-fix check on `82ecd60b9d` (2026-11-04)

`82ecd60b9d` contains `4ff396340a`, which makes lane P capture its graphs on a
stream of its own. It also gives lane P its own MoE scratch and QSA top-k
rows. The setup matches the sweep: same recipe, the same `serve_validate.sh`
set, and the exact tree `~/vllm-tl-82ec`. Serve dirs are
`serve-tp4sw-82ecd60b9d-<arm>`, plus `serve-tp4cells-82ecd60b9d-offr`, a
cells-only flag-off rerun.

A sampler read each worker's KFD `evicted_ms` every 0.5 s
(`accept/tp4/evict_watch.py`). `accept/tp4/evict_windows.py` charged each
eviction to the window it ended in. A window with any eviction after READY
is left out below. Neighbor boots and host-memory reclaim evicted this
server's queues for up to 57 s at a time.

Quality passed in every full arm: probe, PPL 4.82-4.89, prefix, concurrency
(0 failing checks, 0 HTTP 400s) and the after-load probe. The cells-only
rerun passed its two probes. The fixed greedy MTP acceptance uses `/metrics`
deltas over 1024:512 x16 with seed 4242:

- off: 0.557 (c1) and 0.606 (c8)
- `a1-20r`: 0.639 and 0.645
- `a0-20`: 0.591 and 0.631

| arm (dirs) | inject max/p99 ms | periodic 4k p99 ms | c1 ITL ms | c8 agg out tok/s, ITL med/mean | 16k:128 TTFT s | 16k:1024 c8 prefill agg tok/s, TTFT s |
|---|---|---|---|---|---|---|
| off (`off`, `offr`) | 523/523 | 515-518 | 24.5, 24.1 | 249.87, 47.9/57.8 | 7.64 | not measured |
| a1-20 (`a1-20r`) | 194/194 | 189-200 | 22.5 | 251.96, 44.9/53.2 | 7.72 | 3591.6, 15.60 |
| a0-20 (`a0-20`, `a0-20r`) | not measured | not measured | 22.4, 22.4 | 260.75, 46.2/49.2 | 7.60, 7.73 | not measured |

"Not measured" means that window was hit by evictions in every run of the
arm. The flag-off 16k:1024 cell was hit in `off` (+14 s) and `offr` (+43 s).
The static split's stall probe and 16k:1024 cell were hit in `a0-20` (+19 s,
+30 s) and `a0-20r` (+21 s, +62 ms). The first `a1-20` run (10:05-10:20)
predates the sampler, so only its quality results count.

What changed after the fix:

- c1 ITL with the flag on is now 22.4-22.5 ms, against 24.1-24.5 ms with
  the flag off, so the c1 ITL gate passes.
- Adaptive on still fails the stall gate (inject p99 194 ms).
- The `a1-20` prefill aggregate, 3591.6 tok/s, falls inside the pre-fix
  adaptive-on range (3572-3663).

The static-split stall and prefill numbers still come from the `2c62da9966`
sweep, and the verdict above stands.

## Feature matrix

Every feature that works on a plain vLLM instance on this fork, and whether it
works with two-lane on. "Unit" means the CPU test named in the row is green;
"GPU" means a real-prompt run with the flag on versus off. A row is only
"supported" once both are green, so the rows marked *GPU pending* are
implemented and unit-proven but not yet accepted.

| feature | status | proof | notes |
|---|---|---|---|
| PLE CPU offload, bf16 / int4 / int4-disk sidecar | verified, TP4 | unit: `tests/v1/worker/test_ple_offload_worker.py` | one connector per lane; the sidecar keys targets, staging, pinned results, completion pages and sequence counters by `(dp_rank, lane)`; the decode lane's doorbell is polled first GPU: `accept/feat/ple-int4-tp4-awq/gate/verdict.txt` at `77cb138fb7` (Flash-Next AWQ-W4A16 + PLE int4-disk, 4 cards 6-9): both arms boot, probes PASS, PPL 4.85/4.84 (mean NLL 1.579/1.577 over 261 tokens), prefix PASS, concurrency PASS (off 0 failing/32 identical of 48; on 0 failing/34 identical/11 near-tie/3 soft), lane streams created at 6 KFD queues, and the PLE connector reports host waits over 3000 launches. The strict stall probe failed its own thresholds and the 16k G3 cells did not run; both are independent of PLE. |
| structured outputs (grammar / bitmask) | verified, TP1 | unit: `tests/v1/core/test_scheduler_two_lane.py`, `tests/v1/engine/test_engine_core_two_lane.py` | per-lane bitmask; the prefill lane constrains the first sampled token, the decode lane every step after the handoff; the bitmask is deferred until the previous batch of the same lane is applied. Proof (`accept/feat/tp1probe/off.json` vs `on.json`): a 28-token prompt + 48-token JSON grammar request is byte-identical off/on and yields `{"city": "Paris", "country": "France"}` |
| multimodal encoder (generate model with a vision tower) | verified, TP1 | unit: `tests/v1/worker/test_lane_runner.py` | the encoder runs on the prefill lane; both lanes share one `EncoderCache`. Proof (`accept/feat/tp1probe/off.json` vs `on.json`): a 64x64 image prompt is byte-identical off/on (`red`) |
| torch profiler | verified, TP1 (trace per rank; lane ranges pending) | unit: `tests/v1/worker/test_lane_runtime.py` | one trace per rank; each lane RPC is a `lane_<lane>_<method>` range and only the decode lane advances the iteration counter. Proof (`accept/feat/tp1probe/rank0.*.pt.trace.json.gz` off/on): a profile run writes a trace per rank and the sampled text matches. Limitation: `record_function` labels are not emitted for lane threads that started before `start_profile` (the offline `LLM.start_profile` path); lane timing trace (`VLLM_RDNA_LANE_TRACE`) is still to be captured on a server boot |
| `--mm-encoder-only` | refused | impossible | that instance runs the encoder and no decoder, so there is no decode lane |
| pooling runners | refused | impossible | a pooling runner has no decode lane |
| encoder-cache transfer | refused | impossible | it moves encoder outputs between instances; lane mode runs both lanes in one process with a shared cache |
| expert cache (`expert_cache_slots`) | future improvement (refused) | design | per-lane `PackedExpertCache` leases (D and P each own their `_active` slot set, sized from one shared budget) or a P-only device staging stream per layer while D keeps the resident cache; the prefill lane's module path reads the same per-layer cache, and the mono K1/K2 workspace is single-lane, so it needs per-lane caches + `mono_on_decode_lane` before it can be enabled. A prototype is parked in `.scratch/parked/expert-cache-prototype.patch` |
| monokernel | refused (measured blocker), TP4 | unit: `tests/config/test_two_lane_config.py`, `tests/models/qwen4_exp/test_amd_mono_rdna2.py` | The preflight refusal is restored after the GPU gate; the queue census fits: mono on with `LANE_HWQ=1` reports 5 KFD queues, `LANE_HWQ=2` 6 (target 7), so 4 pool + 4 dedicated streams is the only over-budget shape. The TP4 server boots and captures compiled PIECEWISE/FULL graphs, then dies on the first four-token P prefill in `qsa_kstore::kstore_k` and, with the fused store disabled, in `_fused_qk_rmsnorm_rope_gate_kernel` with an HSA memory-aperture fault: lane P's mono QSA cache is not the shared KV cache, so shared-pool block ids address outside it. Flag-off mono TP4 passes the same PPL/concurrency gate (PPL 4.78, concurrency PASS). Fix belongs to lane P's model runner (share the mono QSA cache by pointer); logs `accept/feat/mono-fix-hwq2i/gate` and `mono-tp4-off/gate`. The refusal is restored in `vllm/config/vllm.py`; lane P's mono QSA cache must alias the shared KV tensors (the per-lane workspace path is not enough). Design note: bind lane P's mono QSA c/r/p cache tensors to lane D's after KV init and before P graph capture; the fused stores key `c_slots`/`r_slots` (not the per-lane workspace lane), so a workspace lane alone is insufficient. |
| LoRA | future improvement (refused) | design | `lora_model_runner_mixin._set_active_loras` writes one process-global `lora_manager.set_active_adapters`; the lane design is per-lane active mappings (a lane-scoped manager view), not a forward lock |
| spec methods other than `mtp`, adaptive verification, dynamic speculation | future improvement (refused) | design | the lane split assumes a fixed 1 + k MTP decode row; generalise the D decode-row width first |
| MTP on DeepSeek V4 / V4.1 (`DeepseekV4ForCausalLM`, `DeepseekV4ForConditionalGeneration`, `DeepseekV41ForCausalLM`) | refused | unit: `tests/config/test_two_lane_config.py` | the target writes one MTP hidden-state buffer (`_mtp_hidden_buffer`) from every forward and its drafter reads it after the step, so both lanes would write the rows the other lane's drafter reads. Qwen4Exp keeps one buffer per lane, picked outside its compiled forward (`700f1b527d`). Without MTP these targets are not refused, but they are not audited for two lanes either |
| `--enable-return-routed-experts` | supported, GPU pending | unit: `tests/model_executor/test_routed_experts_capture.py`; proof: `tools/rdna/port_v031/probe_aux_output.py` (`--capture-ref` on the feature-off arm, `--compare` on the feature-on arm) | the aux-output data plane stays on the decode lane, and so does the capturer: only the decode lane builds the connector, buffer and store, and the capture callback bound on the shared model returns immediately when `current_lane()` is the prefill lane, so the two lanes never race in one device buffer (R-OFFLOAD). The prefill lane's routed rows are not request outputs: the aux output is attached to sampled tokens TP2 35B gate: the two-lane-on `_causal_conv1d_update_kernel` fault that blocked it is fixed in `54702c7460` (two-lane on and off both pass without the aux flag, see "TP2 35B lane D conv1d fault"); Flash-Next TP2 OOMs (needs TP4). No aux on/off verdict yet. |
| sleep mode, weight transfer, `reload_weights` | verified, TP1 | unit: `tests/v1/worker/test_gpu_worker.py`, `tests/config/test_two_lane_config.py` | both lanes hold the same weights and KV by pointer, so the backend suspends the one copy. Proof (`accept/feat/tp1probe/off-lanetrace.json` vs `on-lanetrace.json`): `sleep(level=1, mode="wait")` + `wake_up()` then a generate is byte-identical off/on. `Worker.sleep` and `Worker.reload_weights` call `lane_runtime.quiesce()` first: no lane thread is inside a forward (or holding a buffer the backend discards) while the weights move. Level 2 keeps the saved buffers on the primary runner only, and both lanes read the same `model`, so `wake_up` restores them for both |
| EPLB | future improvement (refused) | design | the balancer is process-global; pause the lanes around a rebalance |
| PP > 1 | supported, TP1/PP2 verified (no hang) | unit: `tests/config/test_two_lane_config.py`, `tests/distributed/test_pp_lane_groups.py`; GPU: `accept/feat/pp2gate2/gate/verdict.txt` | every PP stage has its own worker with its own lane runtime. The PP intermediate-tensor p2p now uses lane P's own PP device and CPU groups (`device_communicator.device_group`/`.cpu_group`); lane D keeps the production ones, so the two lanes never share one communicator. The earlier failure (27B AWQ TP1/PP2, HIP 4,5: boot, lane mode, then the first decode hung) was the communicator wrapper passed as a `ProcessGroup`; it made torch reject the send as `CudaCommunicator ... not registered`. Verified 2026-11-03: off and on arms booted, probes PASS, concurrency off/on PASS (46/48 identical, 0 failing, no hang). The gate G3 bench numbers were not produced by that CELLS override (stall/G3 cells pending) |
| DP > 1 | measured incompatible (DP2 + compiled lane) | unit: `tests/config/test_two_lane_config.py`; measured: `accept/feat/dp2/run_noep.log` at `117b739fd5` | a DP2 worker reaches lane start and then fails the KFD census with 9 queues per process (3 baseline + 4 lane streams + the DP communicator set) against the hard limit of 8. `--no-enable-expert-parallel` drops the EP set but still leaves 9. Flag-off DP is rejected upstream for dense models; a MoE DP2 loads but hits the lane census. Needs a budget-aware 3-stream lane mode or a shared DP/lane communicator; until then this is the measured limit. |
| KV connector | supported, GPU pending | unit: `tests/v1/worker/test_kv_connector_lane.py`, `tests/v1/core/test_scheduler_two_lane.py` | the prefill lane is the only admission point, so it is where an external hit is adopted: the connector sizes it after the local prefix hit, the request's num_computed_tokens includes it, and `--num-gpu-blocks-override`-free admission is unchanged. Each lane's `SchedulerOutput` carries its own `kv_connector_metadata`. The worker has one connector instance but each lane thread gets a metadata view that rebinds before every connector call under one lock, because the attention layers read the metadata the connector bound globally. Lane mode starts the load synchronously (`has_sync_kv_loads`): the async path parks the request in `WAITING_FOR_REMOTE_KVS` on the single global step sequence the lanes do not share, so an async load is dropped and the request loads before its forward. The planned disaggregation connector (P/D in two processes) is a separate project |
| DCP / PCP > 1, DBO | future improvement (refused) | design | both re-split the batch the lanes already split; needs a combined partition design |
| LPTH, dynamic prefill, decode stall / time share | refused by design | they compete with the lane split | these are prefill-rationing knobs: they decide how much prefill to run against decode, which is exactly what the two lanes decide. Enabling both would give two controllers for one budget |

Flag-off parity: every change above is behind `lane_mode_enabled()` or
`scheduler_config.two_lane`, so the flag-off path is unchanged.

## Final acceptance (2026-11-04)

Branch `feat/two-lane`; measured on the clean `~/vllm-tl-8c32` tree (8c329)
for M1/M2/M3 and on the exact pushed trees named in each M4 table.

| milestone | verdict | measured |
|---|---|---|
| M1 compiled handoff | **green** | 10/10 lengths byte-equal at `--compile-mode 3`, `COMPILE_MODE=3 FULL_AND_PIECEWISE`, engine cudagraph, `max_num_seqs=1`; E-GIL 1.408; P share 98%; 309-iteration soak, 0 errors; KFD 7 with D full graphs |
| M2 TP1 0.8B MTP2 | **correctness green, G2/G3 open** | probe/PPL/prefix outputs and PPL 14.73 byte-identical off/on; decode c1 ITL +0.6 ms/step (10.4 vs 9.8); 16k c1 -15% (124.47 vs 147.10 tok/s); 16k inject max 104.41 ms / p99 98.11 ms on (off 196.79/196.29) but 27-28.5 decode tokens per window (gate 100) |
| M3 TP2 27B | gate verdict in `accept/m3gate` | flag off: PPL 4.90, probe/prefix PASS; stall and bench in the gate verdict |
| M4 TP4 Flash-Next MTP2 | **no-go (keep opt-in)** | no split passes G2 and G3 together; see the split table below |

M4 split table, flag on, compiled F&P, MTP2, HIP 6-9 (`2c62da9966` static,
`5bc92eeaf1` adaptive; full table and cells above):

| arm | inject max/p99 ms | 16k:1024 c8 prefill agg tok/s, TTFT s | stall gate |
|---|---|---|---|
| static 20/16 (`a0-20`) | 86/72 | 1403.7 (-64%), 70.78 | PASS |
| static 16/20 (`a0-16`) | 81/80 | 1424.3 (-64%), 66.74 | FAIL: 88-97.5 decode tokens |
| adaptive 20/16 (`a1-20`) | 205/205 | 3572.1 (-10.0%), 13.34 | FAIL |
| adaptive 16/20 (`a1-16`) | 224/224 | 3617.1 (-8.8%), 19.13 | FAIL |

Splits 18 and 24 sit between the measured 16/20 and 20/16 arms; they can
move the stall and prefill numbers along the same trade-off, not remove it:
static 20 is already the best static arm and gives up 64% of the prefill
aggregate, while adaptive 20 misses the 100 ms gate by 2x.

**Recommendation: keep the two-lane engine opt-in only and do not enable it
by default on this hardware.** The measured reason is not a missing tuning
knob: with D's full graphs, static CU partitioning meets the stall gate only
when P is starved of WGPs, and then loses 56-64% of long-prompt prefill
throughput; adaptive partitioning keeps prefill within 10% but lets a D batch
wait one P chunk (205-312 ms inject). The recipes therefore stay at split 20
with adaptive on (throughput) and `VLLM_RDNA_LANE_ADAPTIVE=0` is the latency
option; both remain explicit opt-ins. The code and flag-off path are kept:
M1 is green and its KFD/compile fixes are independent of the M4 verdict.

### TP2 feature checks on 35B (2026-11-05)

The TP2 35B runs use `flashnext-mtp2-twolane`, MTP0, `MAXLEN=4096`,
`KV=2e9`, HIP 6,7, one boot per arm.

- Routed experts: flag off passes (probe 3/3, PPL 14.28, prefix PASS).
  Flag on boots and the request reaches the scheduler, then dies in
  `aux_output_connector.take_output` with
  `auxiliary output worker output is missing <request>`. The decode lane
  owns the capture, so a P-lane (prefill) update must not take the aux
  output; `954807c366` gates `take_output` by `request.lane`, after
  `b5013e166c` gated it by the batch lane (the P `SchedulerOutput` still
  carries the default DECODE lane). GPU verdict for the fixed build is
  pending.
- KV connector: the GPU check is blocked before lane mode starts. With
  `ExampleConnector` on Qwen3.5-0.8B, boot fails in
  `_promote_local_kv_cache_specs` with
  `Failed to promote local KV cache specs to one unified type`; the
  same failure happens flag off, so it is the hybrid (mamba + attention)
  cache layout against this connector, not a lane-mode bug. A
  hybrid-aware connector is needed before the GPU check.
- Mono QSA caches: `b50e0a7ece` binds lane P's
  `.indexer.raw_key_cache`,
  `.indexer.compressed_key_cache` and main `.self_attn.attn` caches to
  lane D's by prefix, after lane P's KV init and before its capture.
  CPU unit test: `tests/v1/worker/test_lane_qsa_cache_bind.py`. TP4 gate
  waits for HIP 8.
- Scheduler warning: the 35B TP2 flag-off 3072:512 cell hits
  `max_num_batched_tokens=2048 cannot cover two mamba state blocks of
  2048 tokens` (the 35B state block is 2048, unlike the Flash-Next 1024
  block). It then runs one request at a time; the M4 sweep measured the
  flag-off 3072:512 c4 cell at TTFT 25.29 s versus 1.28 s flag on (the
  flag-on split also differs). `flashnext-mtp2` and `27b-awq` do not
  hit it (block 1024); the 35B TP2 two-lane recipe does, so a 35B TP2
  run needs `MAXBAT=4096`.
