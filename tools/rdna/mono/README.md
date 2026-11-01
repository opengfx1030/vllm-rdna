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

Native-copy kernel/cache tests pass on HIP 7.14 gfx1030. A long native-copy
MTP2 qualification hit NaN logits and remains under investigation; this backend
and the wider K2 gate are not enabled by default.
