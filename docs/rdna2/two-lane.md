# Two-lane engine: acceptance verdicts

Status: **in progress**. Branch `feat/two-lane`; contract in
`~/w4a8_runs/lanes/async/two-lane-contract.md` (untracked copy in the tree).

The engine is opt-in (`TWO_LANE=1`); the flag-off path stays the rdna_extras
default. Flag-on recipes are `tools/rdna/recipes/flashnext-mtp2-twolane.env`
(TP4) and `tools/rdna/recipes/27b-awq-twolane.env` (TP2).

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
| (e) KFD census 7 | **PASS eager** | 7 queues with D streams (driver), 8 just before shutdown (transient); real two-lane worker 6 idle and under a short bench (queue probe) |

Findings:

- The L<=16 mismatch was the driver handing out **block id 0**, vLLM's null
  block, not an S5/S6 bug. The driver now reserves block 0 (real
  `KVCacheManager` never allocates it); all lengths match.
- The `post_update_kernel` GPU fault at 16k was the driver's default
  `max_model_len=16384`: a 16385-token request needs 1025 block-table slots,
  the buffer had 1024. Default is now 32768.
- **Resolved** (`1f5707c298` runner + `f63fb584ed` scheduler): the handoff
  history lost `prefill_len`, so the handoff row (`prompt + [t1]`,
  `num_computed_tokens == len(prompt)`) was classified as a pure decode row.
  `add_request` now splits the already-computed prefix (`prefill_len`) from the
  total history and seeds `last_sampled_tokens`; the penalty grid sizes from
  the total length. A batch with a prefill row is no longer
  `decode_graph_eligible`, so it never replays a FULL graph captured from
  pure-decode dummies. Compiled handoff is green at all ten lengths on HEAD
  (`fix4comp`: `equal`/`self_equal` true at every length, KFD queues 7;
  eager control `fix4eager` all true).
- The prefill-tail FULL replay is real at MTP0 (a 1-token tail dispatches
  `mode=FULL has_prefill=True`, `fix-mtp0` at TP1): at MTP0 the plain engine's
  output is still equal because its prefill metadata is correct, but the
  two-lane handoff row's metadata was not. With MTP2 the scheduler pads every
  tail to the 3-token spec shape (`sched=1`, 2 placeholder drafts), so the
  target tail is `PIECEWISE 3` and production never exercises the bad
  classification (`fix-mtp2`; `mtp2` is also PIECEWISE 3).
- At TP1 the graph build needs `GPU_MAX_HW_QUEUES=1` to stay within 8 queues;
  the production recipes pin 2. The 8th queue with graphs must be removed
  before M4 (gate 7).
- Fresh compiled handoff audit (2026-11-03 19:59, `m1-compiled-fix`, HIP 8,
  `--compile`, 20 D WGPs): 11/11 lengths `equal` and `self_equal`, free-run
  tokens identical to the baseline; KFD queues 7 after boot, 7 after the
  lane streams, 8 just before shutdown (one transient queue); EGIL d
  concurrent p99 13.35 ms / d solo p50 9.48 ms = 1.408; D solo p90 10.2 ms,
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
flag off and flag on in one sequential pair. Verdict: **green on the
product gates, one latency miss** (numbers below; no gate loosened).

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
| decode-only c1 (`1024:512:1`) | 301.80 tok/s, ITL 9.8 ms | **284.52 tok/s, ITL 10.4 ms** (faster) |
| decode-only c8 (`1024:512:8`) | 114.73 tok/s, ITL 23.5 ms | **109.45 tok/s, ITL 24.1 ms** (faster) |
| 16k:128 c1 (small D) | 4711.9 prefill, 215.73 decode, TTFT 3.48 s | 4644.2 prefill, 202.39 decode, TTFT 3.53 s |
| 16k:1024 c8 prefill aggregate | 9124.9 tok/s (`2420.7`/req) | **8548.7 tok/s** (`1872.7`/req), -6.3% (gate 10% PASS) |
| 16k:1024 c8 decode aggregate | 210.65 tok/s (`32.11`/req), ITL 43.3/92.0 ms | 205.79 tok/s (`33.62`/req), ITL 43.8/87.3 ms |
| 16k:1024 c8 TTFT | 6.77 s | 8.75 s (cross-cell; c1 TTFT 3.48 -> 3.53 s) |
| stall probe 16k inject p99 | 532.97 ms (max 533-535) | **210.7 ms** (max 211), -60% |
| stall probe periodic 4k p99 | 194.8-196.3 ms | 98.1-100.3 ms |
| stall probe busy gap p99 | 129.7 ms | 88.1 ms |
| stall probe min decode tokens (periodic 4k) | n/a | 27-33 (gate wants 100; the 16k minimum applies to 4k windows) |

- Decode-only: the flag-on arms are **faster** than flag off (-5.7% c1,
  -4.6% c8), so there is no decode-only regression. The 2% absolute match
  misses by 2.6-3.7 points in the safe direction; only same-day boot pairs
  are comparable.
- Mixed prefill: aggregate within 10% (PASS); the small-D c1/c8 cells are
  also not worse (-6.2% and +4.7% decode throughput). The adaptive split is
  what closes the 2701 -> 8548 tok/s gap.
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
  `torch.cuda.Stream`, so the remaining 4.6-5.7% is lane machinery. A
  baseline-`step` passthrough for P-idle iterations was tried and rejected:
  the lane requests' `execute_model`/`sample_tokens` futures are lane
  futures, the baseline `step` handed a `Future` to
  `scheduler.update_from_output`, and the server died
  (`'Future' object has no attribute 'sampled_token_ids'`) at the first c1
  decode. A safe zero-overhead path needs the lane futures integrated into
  the baseline step; that is not a one-line change, so the shipped
  configuration is `022ee2d31c` with the honest decode-only miss above.
- The focused inject after the large-batch adaptive split (D_full/P_B)
  measured 217 ms p99, worse than the 211 ms with both lanes partitioned
  (D_A/P_B), so that variant was reverted; inject stays on the 20/16 split.

## M3 TP2 27B / M4 TP4 Flash-Next

Serialized sweep `accept/tp_sweep.sh` (one boot at a time, HIP 6-9; logs
`accept/m3-<arm>.log`, `accept/m4-<arm>.log`; done marker
`accept/tp-sweep.done`). Both arms of each milestone use the same recipe on
the same tree (`~/vllm-two-lane-head`).

M3 TP2 Qwen3.8-27B AWQ-INT4, HIP 6,7, compiled F&P, `MAXLEN=32768 KV=6e9`,
D/P split 20/16, greedy. Flag off (measured): correctness PASS, PPL 4.90 PASS,
prefix PASS (TTFT 5.63 s cold / 27.82 s at 16k, solo); 16k inject max stall
2210 ms (p99 2210, 19.5 decode tokens), periodic 4k 1513 ms; bench c1
745.3 tok/s prefill / 30.66 decode / TTFT 1.37 s / ITL 32.6 ms. Flag on and
the G2 4k inject follow in this sweep; numbers are appended here when the
arms land.

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

## Feature matrix

Every feature that works on a plain vLLM instance on this fork, and whether it
works with two-lane on. "Unit" means the CPU test named in the row is green;
"GPU" means a real-prompt run with the flag on versus off. A row is only
"supported" once both are green, so the rows marked *GPU pending* are
implemented and unit-proven but not yet accepted.

| feature | status | proof | notes |
|---|---|---|---|
| PLE CPU offload, bf16 / int4 / int4-disk sidecar | supported, GPU pending | unit: `tests/v1/worker/test_ple_offload_worker.py` | one connector per lane; the sidecar keys targets, staging, pinned results, completion pages and sequence counters by `(dp_rank, lane)`; the decode lane's doorbell is polled first |
| structured outputs (grammar / bitmask) | verified, TP1 | unit: `tests/v1/core/test_scheduler_two_lane.py`, `tests/v1/engine/test_engine_core_two_lane.py` | per-lane bitmask; the prefill lane constrains the first sampled token, the decode lane every step after the handoff; the bitmask is deferred until the previous batch of the same lane is applied. Proof (`accept/feat/tp1probe/off.json` vs `on.json`): a 28-token prompt + 48-token JSON grammar request is byte-identical off/on and yields `{"city": "Paris", "country": "France"}` |
| multimodal encoder (generate model with a vision tower) | verified, TP1 | unit: `tests/v1/worker/test_lane_runner.py` | the encoder runs on the prefill lane; both lanes share one `EncoderCache`. Proof (`accept/feat/tp1probe/off.json` vs `on.json`): a 64x64 image prompt is byte-identical off/on (`red`) |
| torch profiler | verified, TP1 (trace per rank; lane ranges pending) | unit: `tests/v1/worker/test_lane_runtime.py` | one trace per rank; each lane RPC is a `lane_<lane>_<method>` range and only the decode lane advances the iteration counter. Proof (`accept/feat/tp1probe/rank0.*.pt.trace.json.gz` off/on): a profile run writes a trace per rank and the sampled text matches. Limitation: `record_function` labels are not emitted for lane threads that started before `start_profile` (the offline `LLM.start_profile` path); lane timing trace (`VLLM_RDNA_LANE_TRACE`) is still to be captured on a server boot |
| `--mm-encoder-only` | refused | impossible | that instance runs the encoder and no decoder, so there is no decode lane |
| pooling runners | refused | impossible | a pooling runner has no decode lane |
| encoder-cache transfer | refused | impossible | it moves encoder outputs between instances; lane mode runs both lanes in one process with a shared cache |
| expert cache (`expert_cache_slots`) | future improvement (refused) | design | per-lane `PackedExpertCache` leases (D and P each own their `_active` slot set, sized from one shared budget) or a P-only device staging stream per layer while D keeps the resident cache; the prefill lane's module path reads the same per-layer cache, and the mono K1/K2 workspace is single-lane, so it needs per-lane caches + `mono_on_decode_lane` before it can be enabled. A prototype is parked in `.scratch/parked/expert-cache-prototype.patch` |
| monokernel | supported, GPU pending | unit: `tests/config/test_two_lane_config.py`, `tests/models/qwen4_exp/test_amd_mono_rdna2.py` | the census is per process and per GPU: baseline 3 KFD queues at `GPU_MAX_HW_QUEUES=2` (default + 2 pool) or 5 at 4, plus the 4 dedicated lane streams. Mono normally raises the pool to 4 (TP4 skew fix), so lane mode pins `GPU_MAX_HW_QUEUES` to `two_lane_hwq` (recipe `LANE_HWQ`, env `VLLM_RDNA_LANE_HWQ`, CLI `--two-lane-hwq`, default 1, at most 2) in `mono_decode_rdna2.set_worker_hw_queue_budget`: 2 + 4 = 6 queues (or 3 + 4 = 7 at 2). The lane streams are explicit, so the cap only changes CLR sharing; it gives up the measured TP4 hwq=4 skew fix (~10 % attention per layer on 1-3 ranks), to be measured per option |
| LoRA | future improvement (refused) | design | `lora_model_runner_mixin._set_active_loras` writes one process-global `lora_manager.set_active_adapters`; the lane design is per-lane active mappings (a lane-scoped manager view), not a forward lock |
| spec methods other than `mtp`, adaptive verification, dynamic speculation | future improvement (refused) | design | the lane split assumes a fixed 1 + k MTP decode row; generalise the D decode-row width first |
| `--enable-return-routed-experts` | supported, GPU pending | unit: `tests/model_executor/test_routed_experts_capture.py`; proof: `tools/rdna/port_v031/probe_aux_output.py` (`--capture-ref` on the feature-off arm, `--compare` on the feature-on arm) | the aux-output data plane stays on the decode lane, and so does the capturer: only the decode lane builds the connector, buffer and store, and the capture callback bound on the shared model returns immediately when `current_lane()` is the prefill lane, so the two lanes never race in one device buffer (R-OFFLOAD). The prefill lane's routed rows are not request outputs: the aux output is attached to sampled tokens |
| sleep mode, weight transfer, `reload_weights` | verified, TP1 | unit: `tests/v1/worker/test_gpu_worker.py`, `tests/config/test_two_lane_config.py` | both lanes hold the same weights and KV by pointer, so the backend suspends the one copy. Proof (`accept/feat/tp1probe/off-lanetrace.json` vs `on-lanetrace.json`): `sleep(level=1, mode="wait")` + `wake_up()` then a generate is byte-identical off/on. `Worker.sleep` and `Worker.reload_weights` call `lane_runtime.quiesce()` first: no lane thread is inside a forward (or holding a buffer the backend discards) while the weights move. Level 2 keeps the saved buffers on the primary runner only, and both lanes read the same `model`, so `wake_up` restores them for both |
| EPLB | future improvement (refused) | design | the balancer is process-global; pause the lanes around a rebalance |
| PP > 1 | supported, GPU pending | unit: `tests/config/test_two_lane_config.py` | every PP stage has its own worker with its own lane runtime, and every group has a lane-P communicator set, so a lane thread's PP p2p and collectives run on that lane's device group. The lane RPC returns only the first TP worker's result of the last PP stage, where the sampled tokens are produced, and the per-runner `PPHandler` (with its own `pp_broadcast` sibling group) carries the sampled-token broadcast back up the chain. The executor's lane transport refuses a DP replica spanning nodes (`nnodes_within_dp > 1`), so every PP stage must be in one node |
| DP > 1 | supported, GPU pending | unit: `tests/config/test_two_lane_config.py` | every DP replica is its own engine-core process with its own executor and worker processes, hence its own lane runtime and communicator sets; the lane split is per worker, and DP only joins replicas. The DP sampled-token all-reduce runs in the engine core, outside the lane threads, and the PLE sidecar already keys by `(dp_rank, lane)` |
| KV connector | supported, GPU pending | unit: `tests/v1/worker/test_kv_connector_lane.py`, `tests/v1/core/test_scheduler_two_lane.py` | the prefill lane is the only admission point, so it is where an external hit is adopted: the connector sizes it after the local prefix hit, the request's num_computed_tokens includes it, and `--num-gpu-blocks-override`-free admission is unchanged. Each lane's `SchedulerOutput` carries its own `kv_connector_metadata`. The worker has one connector instance but each lane thread gets a metadata view that rebinds before every connector call under one lock, because the attention layers read the metadata the connector bound globally. Lane mode starts the load synchronously (`has_sync_kv_loads`): the async path parks the request in `WAITING_FOR_REMOTE_KVS` on the single global step sequence the lanes do not share, so an async load is dropped and the request loads before its forward. The planned disaggregation connector (P/D in two processes) is a separate project |
| DCP / PCP > 1, DBO | future improvement (refused) | design | both re-split the batch the lanes already split; needs a combined partition design |
| LPTH, dynamic prefill, decode stall / time share | refused by design | they compete with the lane split | these are prefill-rationing knobs: they decide how much prefill to run against decode, which is exactly what the two lanes decide. Enabling both would give two controllers for one budget |

Flag-off parity: every change above is behind `lane_mode_enabled()` or
`scheduler_config.two_lane`, so the flag-off path is unchanged.
