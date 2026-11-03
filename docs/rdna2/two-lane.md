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
| (a) handoff L in {1, 5, 1023, 1024, 1025, 2047, 2048, 4097, 16384, 16385} | **PASS** (eager) | all lengths byte-equal; 16384/16385 included with `max_model_len=32768` |
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
- **Open:** with D compiled (`--compile`), P's first token matches the baseline
  but D's first decode diverges for every handoff length; eager D is exact.
  GDN state arenas are now disabled in lane mode and the divergence remains, so
  the captured handout path (input/state buffer of a NEW request) is the
  suspect. This blocks M2/M4 handoff gates under the production compile mode.
- At TP1 the graph build needs `GPU_MAX_HW_QUEUES=1` to stay within 8 queues;
  the production recipes pin 2. The 8th queue with graphs must be removed
  before M4 (gate 7).

## M2 engine at TP1

Pending the compiled handoff fix. `two_lane_gate.sh` is in place; the real
server smoke is running with the 27B recipe overridden to TP1/Qwen3.5-0.8B.

## M3 TP2 27B / M4 TP4 Flash-Next

Not started; G0-G7, G4 split table (16/18/20/24, including 16/20 and 20/16)
and the keep/drop recommendation follow after M2.

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
| monokernel | refused | queue budget, measured | the monokernel raises `GPU_MAX_HW_QUEUES` to 4 (`mono/rdna2/__init__.py:38`), and that is per stream. Lane mode has four lane streams, so the census would be 3 baseline + 4 x 4 = 19 against a hard limit of 8. Even a shared-queue design that runs the monokernel on the decode lane's full-graph stream only needs 3 + 4 + 3 = 10. It fits only if the two partitioned streams are dropped (3 + 4 + 1 = 8, exactly at the limit), and those streams are what a mixed batch uses |
| LoRA | refused | not yet | adapter state is process-global |
| spec methods other than `mtp`, adaptive verification, dynamic speculation | refused | not yet | the lane split assumes a fixed 1 + k decode row |
| `--enable-return-routed-experts` | refused | not yet | the aux-output connector is process-global |
| sleep mode, weight transfer, `reload_weights` | refused | not yet | both lanes hold the same weights and KV by pointer |
| EPLB | refused | not yet | the balancer is process-global |
| PP > 1, DP > 1 | refused | not yet | the lane split is per worker |
| KV connector | supported, GPU pending | unit: `tests/v1/worker/test_kv_connector_lane.py`, `tests/v1/core/test_scheduler_two_lane.py` | the prefill lane is the only admission point, so it is where an external hit is adopted: the connector sizes it after the local prefix hit, the request's num_computed_tokens includes it, and `--num-gpu-blocks-override`-free admission is unchanged. Each lane's `SchedulerOutput` carries its own `kv_connector_metadata`. The worker has one connector instance but each lane thread gets a metadata view that rebinds before every connector call under one lock, because the attention layers read the metadata the connector bound globally. Lane mode starts the load synchronously (`has_sync_kv_loads`): the async path parks the request in `WAITING_FOR_REMOTE_KVS` on the single global step sequence the lanes do not share, so an async load is dropped and the request loads before its forward. The planned disaggregation connector (P/D in two processes) is a separate project |
| DCP / PCP > 1, DBO | refused | not yet | both re-split the batch the lanes already split |
| LPTH, dynamic prefill, decode stall / time share | refused by design | they compete with the lane split | these are prefill-rationing knobs: they decide how much prefill to run against decode, which is exactly what the two lanes decide. Enabling both would give two controllers for one budget |

Flag-off parity: every change above is behind `lane_mode_enabled()` or
`scheduler_config.two_lane`, so the flag-off path is unchanged.
