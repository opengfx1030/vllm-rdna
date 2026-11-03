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
| (b) E-GIL D p99 / D p50 solo | **FAIL (marginal)** | 1.529 (gate <=1.5); D solo p50 11.17 ms, D concurrent p99 17.1 ms. Earlier run 1.489. P eager Python launch is the residual GIL cost |
| (c) P chunk rate >= 45% of solo | **PASS** | 98.0% |
| (d) 10 min, no hang/error, strict clean | **PASS** | 309 soak iterations, 0 mismatches, 0 errors, `d_reqs=0`, `p_reqs=0` |
| (e) KFD census 7 | **PASS eager** | 7 queues; 8 with D graphs (`GPU_MAX_HW_QUEUES=1`), at the hard limit and not the gate |

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

Real server `serve_validate.sh`, `~/vllm-two-lane-head`, Qwen3.5-0.8B +
`MTP=2`, TP1, HIP 9, `MAXLEN=32768 KV=4e9 SEQS=8 MAXBAT=2048`, compiled
F&P, greedy, cells `1024:512:1 1024:512:8 16384:1024:8:32 16384:128:1`,
D/P split 20/16. Verdict: **not green** (numbers below; no gate loosened).

Boot-to-boot spread on this node is large: 16k inject max stall measured
191 ms and 514.7 ms (flag off), 41.3 ms and 107.2 ms (flag on); decode-only
c=1 160.1 and 305.6 tok/s. Only same-boot A/B pairs are comparable; the
handoff probe and the greedy rows are the cross-boot evidence.

| gate | off | on |
|---|---|---|
| correctness / PPL / prefix | PASS / 14.73 / 2.21-0.13 s | PASS / 14.73 / 2.28-0.15 s |
| handoff probe compare (`handoff_probe.py`, greedy) | 11/11 ok, 0 failing | 11/11 byte-equal to off at 1..16385 |
| stall probe, 16k inject | max 191 and 514.7 ms, p99 514.6 | max 41.3 and 107.2+-3 ms, p99 85.5, dec tok 627, TTFT 2.27x solo |
| stall probe, periodic 4k | 190-193 ms | 39-41 ms |
| reverse (16k prefill at arrival) | 7.2 s | 0.46 s |
| concurrency (48) | PASS, 0 failing | FAIL, 1/48 (`long 1`, token 0 not solo top-5) on the 18:03 sync; the gate reruns both arms on HEAD |
| KFD at lane start | n/a (no lane streams) | 6 own lane queues at idle (queue probe); target 7, hard 8. The older `8` report is the capture path to re-check (gate) |

- Handoff (PASS): `handoff_probe.py compare` reads only the greedy stream
  (temperature 0) — 11/11 byte-equal at 1..16385, 0 failing — so the handoff
  decode, KV, GDN state and MTP draft/verify agree. Temperature 0.8 sampling
  is not invariant under a different batch composition (RNG and reduction
  order), so the compare skips it unless `--all-kinds` (e5efb5b1b2); the
  measured non-greedy mismatches first differ at token 1+ (token 0 exact) and
  are informational, not a gate.
- Stall: p99 85 ms passes; the max is 7 % over 100 in one boot (41.3 ms in
  the other), only at P activation; the flag-off arm is 515 ms. D5-open.
  The `min_decode_tokens_during_prefill` FAILs on periodic 4k windows
  (96/90/87/84 vs 100) apply the 16k minimum to 4k windows (contract 4k
  minimum is 30) and are not acceptance misses.
- Bench, latest same-boot rerun (flag off): c1 7069.6 / 160.1 / 0.14 s /
  9.7 ms; c8 3457.4 / 70.8 / 0.30 / 22.3; 16k:1024 c1 4907.9 / 164.0 /
  3.34 / 12.1; 16k:1024 c8 3408.5 / 24.6 / 4.81 / 42.2 mean 73.4.
  Flag on: c1 5630.9 / 287.4 / 0.18 / 10.3; c8 1587.0 (agg 7876) / 112.6 /
  0.65 / 23.8; 16k:1024 c8 481.3 (agg 2700.9) / 77.4 / 34.0 / 45.1 mean
  38.0; 16k:128 c1 4707.5 / 204.0 / 3.48 / 12.7. The mixed 16k:1024 c8
  mean ITL is 2.5x better behind a 7x slower prefill; the p50 is 45.1 vs
  42.4 ms. G3 needs a same-boot A/B pair (cross-boot spread above is too
  large); `two_lane_gate.sh` runs both arms and applies G3.

### S7 gate (`two_lane_gate.sh`)

`tools/rdna/port_v031/two_lane_gate.sh` runs the full acceptance for one
recipe and split: one arm flag off then one arm flag on (same recipe, same
cells, one boot at a time), then per-arm G1/G2/G5 checks and the G3 compare.
It writes `~/w4a8_runs/port-v031/serve-<tag>-gate/gate/{off,on}/` plus
`gate/verdict.txt`. `bench_compare.py` (new helper) applies G3: 16k:1024 c8
mean ITL `on <= 0.7 x off`, p50 `on <= 1.1 x off`; prefill-only 16k:128 c1
TTFT `on <= 1.05 x off`. G4 (split sweep) and G7 (soak) are separate tools.

- KFD: lane start census 7; the extra queue with D FULL graphs (8) is open;
  the capture path is the suspect (I6 target 7, hard 8).

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
| structured outputs (grammar / bitmask) | supported, GPU pending | unit: `tests/v1/core/test_scheduler_two_lane.py`, `tests/v1/engine/test_engine_core_two_lane.py` | per-lane bitmask; the prefill lane constrains the first sampled token, the decode lane every step after the handoff; the bitmask is deferred until the previous batch of the same lane is applied |
| multimodal encoder (generate model with a vision tower) | supported, GPU pending | unit: `tests/v1/worker/test_lane_runner.py` | the encoder runs on the prefill lane; both lanes share one `EncoderCache` |
| torch profiler | supported, GPU pending | unit: `tests/v1/worker/test_lane_runtime.py` | one trace per rank; each lane RPC is a `lane_<lane>_<method>` range and only the decode lane advances the iteration counter |
| `--mm-encoder-only` | refused | impossible | that instance runs the encoder and no decoder, so there is no decode lane |
| pooling runners | refused | impossible | a pooling runner has no decode lane |
| encoder-cache transfer | refused | impossible | it moves encoder outputs between instances; lane mode runs both lanes in one process with a shared cache |
| expert cache (`expert_cache_slots`) | refused | blocked by the monokernel | `cpu_weight_context` asserts `--monokernel` and supports Qwen4Exp only, so the cache cannot be enabled without the monokernel, which lane mode refuses on the queue budget. The cache itself is per-layer state, not process-global |
| monokernel | supported, GPU pending | unit: `tests/config/test_two_lane_config.py`, `tests/models/qwen4_exp/test_amd_mono_rdna2.py` | the census is per process and per GPU: baseline 3 KFD queues at `GPU_MAX_HW_QUEUES=2` (default + 2 pool) or 5 at 4, plus the 4 dedicated lane streams. Mono normally raises the pool to 4 (TP4 skew fix), so lane mode pins `GPU_MAX_HW_QUEUES` to `two_lane_hwq` (recipe `LANE_HWQ`, env `VLLM_RDNA_LANE_HWQ`, CLI `--two-lane-hwq`, default 1, at most 2) in `mono_decode_rdna2.set_worker_hw_queue_budget`: 2 + 4 = 6 queues (or 3 + 4 = 7 at 2). The lane streams are explicit, so the cap only changes CLR sharing; it gives up the measured TP4 hwq=4 skew fix (~10 % attention per layer on 1-3 ranks), to be measured per option |
| expert cache (`expert_cache_slots`) | refused | audit: prefill lane | `cpu_weight_context` requires `--monokernel` and Qwen4Exp, and leaves the only expert copy packed on the host (`requires_device_loading=False`). `PackedExpertCache` also keeps one per-layer lease (`_active`) with no lock. The prefill lane's module path would read CPU expert weights on the device (and could race the lease), so the cache is refused until it is per lane or the prefill lane reloads device experts |
| LoRA | refused | not yet | adapter state is process-global |
| spec methods other than `mtp`, adaptive verification, dynamic speculation | refused | not yet | the lane split assumes a fixed 1 + k decode row |
| `--enable-return-routed-experts` | supported, GPU pending | unit: `tests/model_executor/test_routed_experts_capture.py`; proof: `tools/rdna/port_v031/probe_aux_output.py` (`--capture-ref` on the feature-off arm, `--compare` on the feature-on arm) | the aux-output data plane stays on the decode lane, and so does the capturer: only the decode lane builds the connector, buffer and store, and the capture callback bound on the shared model returns immediately when `current_lane()` is the prefill lane, so the two lanes never race in one device buffer (R-OFFLOAD). The prefill lane's routed rows are not request outputs: the aux output is attached to sampled tokens |
| sleep mode, weight transfer, `reload_weights` | supported, GPU pending | unit: `tests/v1/worker/test_gpu_worker.py`, `tests/config/test_two_lane_config.py` | both lanes hold the same weights and KV by pointer, so the backend suspends the one copy. `Worker.sleep` and `Worker.reload_weights` call `lane_runtime.quiesce()` first: no lane thread is inside a forward (or holding a buffer the backend discards) while the weights move. Level 2 keeps the saved buffers on the primary runner only, and both lanes read the same `model`, so `wake_up` restores them for both |
| EPLB | refused | not yet | the balancer is process-global |
| PP > 1 | supported, GPU pending | unit: `tests/config/test_two_lane_config.py` | every PP stage has its own worker with its own lane runtime, and every group has a lane-P communicator set, so a lane thread's PP p2p and collectives run on that lane's device group. The lane RPC returns only the first TP worker's result of the last PP stage, where the sampled tokens are produced, and the per-runner `PPHandler` (with its own `pp_broadcast` sibling group) carries the sampled-token broadcast back up the chain. The executor's lane transport refuses a DP replica spanning nodes (`nnodes_within_dp > 1`), so every PP stage must be in one node |
| DP > 1 | supported, GPU pending | unit: `tests/config/test_two_lane_config.py` | every DP replica is its own engine-core process with its own executor and worker processes, hence its own lane runtime and communicator sets; the lane split is per worker, and DP only joins replicas. The DP sampled-token all-reduce runs in the engine core, outside the lane threads, and the PLE sidecar already keys by `(dp_rank, lane)` |
| KV connector | supported, GPU pending | unit: `tests/v1/worker/test_kv_connector_lane.py`, `tests/v1/core/test_scheduler_two_lane.py` | the prefill lane is the only admission point, so it is where an external hit is adopted: the connector sizes it after the local prefix hit, the request's num_computed_tokens includes it, and `--num-gpu-blocks-override`-free admission is unchanged. Each lane's `SchedulerOutput` carries its own `kv_connector_metadata`. The worker has one connector instance but each lane thread gets a metadata view that rebinds before every connector call under one lock, because the attention layers read the metadata the connector bound globally. Lane mode starts the load synchronously (`has_sync_kv_loads`): the async path parks the request in `WAITING_FOR_REMOTE_KVS` on the single global step sequence the lanes do not share, so an async load is dropped and the request loads before its forward. The planned disaggregation connector (P/D in two processes) is a separate project |
| DCP / PCP > 1, DBO | refused | not yet | both re-split the batch the lanes already split |
| LPTH, dynamic prefill, decode stall / time share | refused by design | they compete with the lane split | these are prefill-rationing knobs: they decide how much prefill to run against decode, which is exactly what the two lanes decide. Enabling both would give two controllers for one budget |

Flag-off parity: every change above is behind `lane_mode_enabled()` or
`scheduler_config.two_lane`, so the flag-off path is unchanged.
