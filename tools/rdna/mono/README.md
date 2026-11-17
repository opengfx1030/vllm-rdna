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

Topology is explicit: `TP` (2 or 4; 256 or 128 local experts per rank) and
`GPUS` must agree in device count, and the `GPUS` set must be listed in
`ALLOWED_GPU_SETS` (space-separated set list; default `6,7`, the maintainer
host's allocation restriction). `SLOTS` is capped at the per-rank local expert
count (default 96 at TP2, 48 at TP4 — the same three-eighths-of-local budget).
A four-card host runs the same plan with:

    GPUS=0,1,2,3 TP=4 ALLOWED_GPU_SETS="0,1,2,3" PRINT=1 bash tools/rdna/mono/serve_offload.sh

(`PRINT=1` prints the resolved plan and exits; drop it to launch.) The
qualification drivers accept the matching `TP`; `sweep_offload_policy.py`
takes `--local-experts 128 --global-experts 512` for TP4 traces and keeps the
TP2 defaults, and `qualify_expert_copy.py` validates any rank count via
`--expected-caches` (48 caches per rank on Qwen4Exp). The split-K2 fused
all-reduce checker `benchmarks/kernels/check_rdna2_offload_tp2_ar.py` takes
`--tp 2|4` (default 2): each rank owns 512/tp experts and a 640/tp-wide
shared slice, the AR probe must sum to tp(tp+1)/2, and at TP4 the independent
oracle is the resident 128-expert `mono_k2_rdna2` entry with AR detached.

`KV_OFFLOAD_GB=0` disables the optional CPU KV tier by omitting its connector
flags. Positive budgets retain the native KV backend; the default is 16 GiB.

For symmetric INT4 tensor copies, opt in to initializing constant zero points
once rather than copying them on every expert miss:

    MONO_CFG='"expert_cache_constant_qzeros":true' KV_OFFLOAD_GB=0 \
        bash tools/rdna/mono/serve_offload.sh

The host zero points are validated, the dummy slot remains zero, and all six
payload views retain their layout. Only weight/scale H2D bytes change; host/VRAM
usage and expert selection do not. The option defaults off and rejects native
copies, including a later RPC backend switch.

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

`rdna_expert_cache_trace` accepts `{"action":"start","max_calls":1024,"layers":[0,1]}`
and later `{"action":"stop"}` through the same endpoint. It returns bounded
CPU-visible demands, including prefill/padding rows, initial LFU/LRU state,
and miss counters for exact replay. Call only between workloads; traced windows
add Python work and are never clean performance evidence.
Retention stops at 16 MiB of raw IDs per worker (the JSON response is larger).
If this cuts a forward short, replay scores only the complete shared prefix
and reports discarded tail records.

    python tools/rdna/mono/ram10_prefetch.py TRACE_STOP_RPC.json --out prefetch.json

Ordinary replay must exactly reproduce observed cache misses before causal
previous-step and physically adjacent-layer predictions are compared at equal
slot count. This counts speculative traffic and useful loads, not runtime
overlap or ITL improvement; it never replaces the model's router.

`rdna_cache_memory` reports exact V2 KV storage bytes, deduplicating shared
tensor views by backing storage. It works on resident and offloaded workers
without synchronizing the device; TP capacity uses the minimum rank budget.

Native-copy kernel/cache tests pass on HIP 7.14 gfx1030. A long native-copy
MTP2 qualification hit NaN logits and remains under investigation; this backend
and the wider K2 gate are not enabled by default.

## TP4 approximately 10% expert-RAM study

`ram10_study.sh` compares four arms without changing the learned router:

| Arm | Real expert slots/rank | Change |
|---|---:|---|
| `R` / `resident` | 128, resident model | Normal compiled MonoKernel; W4A8 prefill |
| `F` / `fullcache` | 128, cache backend | Full-capacity cache control; W4A16 prefill |
| `D` / `demand` | 115, cache backend | Demand-loaded experts |
| `Z` / `zeros` | 115, cache backend | As D, initializing constant zero points once |

D/Z leave 13 of 128 experts nonresident, but their dummy slot means **116
physical rows and 12 rows saved**. The immutable CPU bank still contains every
expert. This is expert-weight offload, not a CPU KV-cache tier.

The defaults hold TP4, MTP2, FULL_AND_PIECEWISE, MonoKernel, PLE int4, eight
sequences, 1024-token blocks, 2048 batched tokens, `GMEM=0.92` and `KV=auto`
fixed. Cached arms need breakable graphs and W4A16 prefill; F separates this
backend cost from reducing the cache to 115 slots. R-to-D is therefore a
deployment comparison, **not an isolated PCIe-copy benchmark**.

Run only under a lease with a checked HIP-to-PCI map and the host's thermal
watchdog. The caller owns those safeguards; the driver owns boot/quiet-window
coordination, prefix resets, warm-up, KFD checks and benchmark health.

    ARM=D GPUS=2,3,4,5 MODEL=<checkpoint> PLE_QUANT_DIR=<ples_int4> \
        bash tools/rdna/mono/ram10_study.sh --dryrun

`CELLS` entries are `input:output:concurrency:requests`; the default matrix
uses 1k/512, 16k/1k and 32k/2k at concurrency 1 and 8. Each cell warms its
exact shape, resets prefix caches, and saves detailed request timings.
Preemptions are cumulative-metric deltas and remain visible: a capacity-limited
32k cell must not be disguised by changing scheduling between arms.

    python tools/rdna/mono/ram10_report.py --run-root <receipts> --baseline R

The report retains per-request CSV, exact TP-minimum/summed KV storage bytes,
token capacity, expert-copy counters and baseline-relative timing. Prefill and
decode span rates are wall-span estimates that can overlap other phases, not
isolated GPU kernel rates; streamed ITL measures delivery chunks under MTP.
Missing observations remain missing, never zero.

`AFTER_MATRIX_HOOK=$TREE/tools/rdna/mono/ram10_trace.sh` collects separate
diagnostic traces for D/Z after clean timing windows. Equal-capacity causal
replay screens bounded prefetch traffic; it cannot prove hidden copy time or
an ITL speedup. The functional/capacity pilot receipts are in
`bench_results/2026-11-17_tp4-ram10/`; timing claims require the matched matrix.

For coding-task rather than random-input diagnostics, `TRACE_DATASET_DIR` may
point to pre-rendered chat JSONL files named `1024.jsonl` and `16384.jsonl`,
with `prompt` and `output_tokens` fields. Their hashes are saved. Override
`TRACE_CELLS="16384:512:1:4 1024:512:8:8"` to include sequential task changes;
nominal file names do not imply exact token lengths.

## Mono next to the W4A8 MoE prefill

`--monokernel` serves decode (K1 up to 24 rows, K2 up to 8, GDN up to 64); the
W4A8 MoE prefill (`VLLM_RDNA2_W4A8_SDOT4=1`) starts at 512 tokens and reads the
same W4A16 expert tensors, so one serve runs both (the expert cache refuses
W4A8). `w4a8/` is the GPU plan that measures it on TP4 Flash-Next MTP2 (recipe
`flashnext-mtp2`, one exact-SHA tree, one boot per arm):

| arm | serve |
|---|---|
| `a` | production: W4A8 prefill, no `--monokernel`, the recipe's HW queues (2 until 2026-11-06, 4 since: `a` then equals `e`) |
| `b` | `--monokernel` with `VLLM_RDNA2_W4A8_SDOT4=0` (mono raises the queues to 4) |
| `c` | `--monokernel` with W4A8, 4 HW queues |
| `d` | as `c` with `rocm_mono_decode.hw_queues=0` (the recipe's queues: 2 until 2026-11-06, 4 since) |
| `e` | production at `GPU_MAX_HW_QUEUES=4` (queue-attribution control, opt-in) |
| `a0` | production with the recipe's PLE (only after the int4 PLE fallback) |

A trailing `2` repeats an arm (`c2`, `a2`) for the boot-to-boot noise floor. The
first run (2026-11-05, tree `5b8d1e5570`, arms `c a c2 a2 d e b`) is written up in
`docs/rdna2/mono-w4a8-2026-11-05.md`; its raw outputs are in
`bench_results/2026-11-05_mono-w4a8/`.

    EXPECT_SHA=<sha> RUN=~/w4a8_runs/mono-w4a8 setsid nohup \
        bash tools/rdna/mono/w4a8/slot.sh > $RUN/slot.log 2>&1 < /dev/null &
    touch $RUN/GO     # open the window (`STOP` aborts; GO may hold KEY=VALUE overrides)
    DRY=1 EXPECT_SHA=<sha> ARMS="a b c d e" bash tools/rdna/mono/w4a8/slot.sh

`slot.sh` touches no GPU before `GO`, then takes the lease, boots each arm
through `port_v031/serve_validate.sh` (`CELLS=none`, `CONC_PROBE=1`) and runs
`w4a8/hook.sh` as its load hook. `w4a8/arm_driver.py` warms each prompt shape
(unmeasured), then measures solo TTFT 4k/16k, the 16k/1k c8 aggregate prefill
and decode at c1/c6/c8 through `vllm bench serve`, each cell between KFD
eviction snapshots (`port_v031/kfd_evictions.py`; a window with a counter jump
is invalid and runs again), then dumps decode logprobs (`logprob_parity.py`)
and scores teacher-forced windows (`port_v031/topk_kl.py`). `w4a8/analyze.py
check-arm` proves from the logs that an arm ran what it was asked to (mono
lines, W4A8 part and sdot4 warning, K2 FULL graphs, the engine's raise of the
queue budget (only the mono arms with 4 queues have it), and the mlock of the
pinned PLE table: the log line and `VmLck` of every worker); `analyze.py
report` writes `results/tables.md`, with the host around each arm (load, AER,
memory lock, late peers, driver evictions, KFD compute queues per worker,
junction temperatures).

Start `slot.sh` from a fresh login shell: the workers mlock their 23.84 GiB
pinned PLE table, and the slot refuses an `ulimit -l` below 32 GiB
(`MEMLOCK_MIN_KB`). While it holds the lease, `w4a8/gpu_temps.py` logs the
junction temperature of each leased GPU every 15 s to `$RUN/temps.csv` and
`w4a8/queue_probe.py` the KFD queues of the running arm's workers every 20 s
to `$RUN/queues.jsonl` (log only, nothing gates on them). The queue budget
cannot be read from a worker (vLLM retitles its processes, which overwrites
`/proc/<pid>/environ`), so the compute queues CLR actually created are the
evidence. CPU tests: `tests/tools/rdna/test_mono_w4a8_plan.py` (it also runs
`slot.sh` end to end against a stub `serve_validate.sh`: the lease, retries,
the config-mismatch stop, the memory-lock gate, the int4 PLE fallback and
`STOP`).
