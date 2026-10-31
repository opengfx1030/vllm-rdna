# RDNA2 MonoKernel tools (`--monokernel`)

The model code is `vllm/models/qwen4_exp/amd/mono_decode_rdna2.py` +
`mono/rdna2/`; the pytest suite is `tests/models/qwen4_exp/test_amd_mono_rdna2*.py`.
History and results: [`docs/rdna/monokernel/`](../../../docs/rdna/monokernel/).

Serve drivers (HIP 6-9 lock, `serve_validate.sh` gates):

    ARM=on|off|env GPUS=6,7,8,9 PORT=18790 TAG=x RECIPE=flashnext-mtp0 \
        bash tools/rdna/mono/serve_2a.sh          # one serve; ARM=on adds --monokernel
    SPECS='mtp0|off|a|;mtp0|on|a|' CELLS="1024:512:1 1024:512:8" GPUS=6,7,8,9 \
        bash tools/rdna/mono/run_specs.sh         # a queue of serves, one lock each

`MONO_CFG='"k2_max_rows":8'` adds `rocm_mono_decode` knobs; `ARM=on-t|off-t`
turn on the in-serve timing (mono vs today's modules under the same marks).

Standalone checks and benches (`PYTHONPATH=<tree>`):

* `check_op.py` — registered `mono_k2_rdna2` vs today's ops, widths 1..24,
  graph replay, fail-stop NaN, stats marks.
* `k1_bench.py`, `m1_bench.py`, `gdn_fused_bench.py`, `qsa_bench.py` —
  correctness and graph timing of K1, K2, GDN, QSA against today's op
  sequences (JIT builds of the `csrc/rocm/mono_*_rdna2.cu` sources).
* `qsa_normrope_check.py`, `qsa_store3_check.py` — QSA fused norm+RoPE and the
  one-launch indexer stores.
* `m1_final.sh`, `run_m1.sh`, `sweep.sh` — the M1 standalone runs;
  `MONO_FFN_PREFETCH=1`, `MONO_FFN_VERBOSE=1`, `MONO_MIN_BLOCKS=2` build variants.

## Expert-cache copy qualification

`serve_offload.sh` keeps the existing tensor copy backend by default. After
rebuilding the ROCm extension, opt in to host-planned native copies with:

    MONO_CFG='"expert_cache_copy":"native"' bash tools/rdna/mono/serve_offload.sh

This RDNA2-only backend submits all six persistent SoA payloads for a cache
pass together using `hipMemcpyBatchAsync` on HIP 7.14+ when available. Older
headers/runtimes use a native `hipMemcpyAsync` loop. It preserves the pinned
CPU bank, GPU pool addresses, cache policy and consumer-stream fencing.
Copy planning stays outside captured graphs. A failed batch is drained and
reported rather than retried, because it may already have submitted a prefix.

`KV_OFFLOAD_GB=0` disables the optional CPU KV tier by omitting its connector
flags. Positive budgets retain the native KV backend; the default is 16 GiB.

For isolated, localhost qualification, load
`--worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker`
and enable `VLLM_SERVER_DEV_MODE=1`. Between workloads only, POST to
`/collective_rpc` with method `rdna_expert_cache_control` and optional kwargs
`{"copy_mode":"tensor"}` or `{"copy_mode":"native"}`. Omitting kwargs reads
per-rank counters without changing the backend. Switching preserves residency
and captured compute graphs; it requires released leases and synchronizes the
device first. Counters measure distinct expert demands per committed bounded
pass, not token-level cache hits. Native calls and successful HIP batch calls
are reported separately, so fallback cannot be mistaken for batching.
The copy A/B clears both GPU and external CPU prefix caches before each arm
and rejects warm or missing prefix-hit counters. Expert residency remains
order-dependent; the results do not isolate a causal model-level speedup.
The optional `finite_checks="on"` / `"off"` RPC argument toggles the existing
Mono debug mode while quiescent. Runs with this debugging are diagnostic and
must not be used as clean performance comparisons.
The diagnostic `post_step_checks="collective" | "wait" | "off"` control
separates TP collective pacing from the post-step completion wait. `collective`
is the normal guard; the other modes are solely for reproducing failures.

Native-copy kernel/cache tests pass on HIP 7.14 gfx1030. An earlier long
native-copy MTP2 qualification hit NaN logits. The current gate of 8 passed
TP2/MTP2 graph-mode quality and repeated 16k/1024 c8 workloads; this does not
establish the earlier failure's root cause. The wider gate remains unqualified.
Native copy remains opt-in. Receipts and scope are in
`bench_results/2026-10-29-tiered-offload/`.

### Tiered-offload experiments

The next implementation stages and acceptance gates are tracked in
`docs/rdna2/tiered-offload-plan.md`. An experimental hybrid KV spill/resume implementation is present behind
`KV_LIVE_OFFLOAD=1`; graph/eager spill/resume, graph abort cleanup and combined
eager overlap tests pass for the qualified TP2/MTP2 recipe. The default CPU KV
tier continues to serve prefix reuse. Do not treat passing prefix-cache tests
as qualification of live paging.

Live paging requires a positive `KV_OFFLOAD_GB`, the RDNA2 Qwen4Exp MonoKernel,
Model Runner V2, aligned recurrent state, FCFS scheduling and PP=1.
`KV_LIVE_WATERMARK=0.9` selects the GPU KV occupancy threshold. The scheduler
selects a previously scheduled, fully prefilled request by LRU when another
request is waiting. It transfers complete packed physical pages into protected
leases in the native CPU pool, waits for all ranks before releasing GPU pages,
and restores into newly reserved GPU pages before publishing replacement tables.
The V2 request slot and numerical selectors remain resident and count against
`max_num_seqs`; vision/encoder, LoRA and resumable streaming-input requests are
excluded.
`KV_LIVE_PAUSE_TARGET_S=10` sets a soft suspension target, measured from the
quiesce request. An overdue host snapshot gets restore priority over fresh
admissions. If its full GPU reservation cannot fit, the scheduler suspends an
eligible resident request to make room, even when all V2 slots are occupied.
Another eviction waits for pending snapshot transfers. Newly resumed requests
receive at least this much resident time before becoming eviction candidates
again. New admission spills wait until previous snapshots have resumed.
Prefill compute and fresh admission are deferred while snapshot fences drain
or copies are pending; resident decode can continue. An overdue host restore
also blocks fresh admission. If no transfer or eligible eviction can start,
existing prefills may continue so they can finish, release pages, or become
eligible victims; freezing them would deadlock restoration.
Retained decode resumes may replay several accepted MTP tail tokens; that work
is allowed through the prefill gate. Logs distinguish restore-ready time from
the first resumed scheduling step.
Async prefix-load completion and failure cleanup run before live admission
gates. Existing GPU-page holders can progress; a load rewound to zero computed
tokens releases its old tables before a fresh lookup. New V2 slots still respect
the cap.
This target is not a wall-clock guarantee: work already in flight, source
readers, multiple pages needed for a restore, and the resident-time protection
can extend it. Phase logs separate quiesce, store, host wait and load durations;
they include scheduler/rank acknowledgement delays, not just DMA service time.
This is whole-request spill/resume, not per-layer KV streaming. Reset is refused
while a live snapshot or unpublished resume owns state. Insufficient CPU/GPU
capacity leaves the ordinary scheduler and recompute path available.
Keep this control off for ordinary launches until the workload demonstrates a
latency/admission benefit; correctness qualification does not establish speedup.
The normal 90% watermark test improved long c8 TTFT but produced a 126-second
victim streaming gap. Whole-request paging is an explicit capacity/batch tradeoff,
not an interactive default. See the matched budget-matrix receipts.

For per-layer transfer diagnostics, use the worker extension above and call
`rdna_expert_cache_control` with `{"transfer_timing":"on"}` between workloads.
The `layers` receipts report capacities, exact payload/map bytes, demands,
misses, copy bytes, and separated timing intervals. Turn timing off before
clean performance runs. GPU intervals include scheduling/enqueue gaps and are
not pure DMA service time. Timing records are bounded; inspect dropped samples.

`VLLM_RDNA_EXPERT_ROUTE_DIR=/persistent/path` enables lightweight route receipts
from routing IDs already on the CPU. `VLLM_RDNA_EXPERT_ROUTE_MAX_CALLS=128`
bounds invocations per rank/layer/phase. Prefill and decode have separate streams;
metadata-free profiling and graph recording are excluded. Collection writes
files and is diagnostic: do not use it for final latency claims.

For workload-only traces, also set `VLLM_RDNA_EXPERT_ROUTE_ARM_FILE` to a
nonexistent marker path, then create that file after startup/warmup and remove
it when collection ends. Directory-only tracing can include startup warmups
that carry dummy attention metadata and can consume the invocation limit before
real requests arrive. An `is_profile` marker is not a production warmup fence.

`tools/rdna/mono/sweep_offload_policy.py` accepts these receipts, with
`--trace-phase decode` or `--trace-phase prefill` when needed. `--plan-profile profile.json`
and `--plan-output plan.json` enable deterministic nonuniform slot planning.
The profile contains `rank_budgets` in GPU bytes and `layers` with rank, layer,
`expert_bytes`, optional `map_bytes`, min/max capacities and candidate capacities.
Every traced layer needs a profile; fixed map bytes count against each rank's
budget. JSONL plans cross-check payload and fixed map costs against measured
trace headers; legacy/manual inputs remain explicitly unverified modeled costs.
This is an empty-cache replay heuristic, not a claim of global optimality
or a prediction of end-to-end latency. `objective_weight` may represent measured
transfer cost, but an arbitrary weight is not a measured speedup.

Launch with `SLOT_PLAN=/path/plan.json` to consume its `layer_slots` map. Every
loaded expert layer must have a capacity for its rank, including any offloaded
MTP layers; missing entries fail startup. Allocation occurs once before capture.
Without a plan, `SLOTS=96` retains uniform allocation.

`MONO_CFG='"expert_cache_overlap":true'` opts into resident-hit-first execution
while copying known misses to disjoint slots in the same GPU pool. Reservations
are unpublished until their transfer fence is established. No router prediction
or extra expert pool is involved. Extra passes/map uploads may outweigh overlap;
keep the default off until matched workloads establish a benefit. Prefetch,
adoption, unused-byte and failure counters are included in per-layer receipts.

`rdna_kv_snapshot_layout` reports native physical page/chunk counts, exact pool
bytes and configured paging capability without synchronizing the device. CPU KV
usage metrics measure non-evictable pins, including protected snapshots; idle
ready entries still occupy the allocated RAM pool even when the gauge is zero.
