# Expert-copy fixed-cost pass

Primary workload: **16,384 input tokens and 1,024 output tokens**, concurrency
1 and 8. The same-server comparison also submits eight such requests at
0.125 requests/s to exercise mixed prefill/decode. Only HIP GPUs 6–7 are used;
the copy-only probe and regression suite use HIP GPU 7.

Flash-Next AWQ INT4 uses TP2/EP2, MTP2 and FULL_AND_PIECEWISE graphs, with
96 expert slots/layer, native expert copies, overlap off, active KV paging off
and the stall controller off. GPU KV is 4,026,531,840 bytes/rank and native CPU
KV is 16 GiB total. The candidate retains all six expert host/device tensors and
the same memory budgets, routing, cache policy and kernel arguments.

## Unchanged baseline and diagnostic

`baseline-profile-raw.tar.gz` retains the complete four-cell workload, server
log and drivers. `baseline-manifest.json` records source revision, native binary
hash and every archive member. `baseline-profile-summary.json` is a compact
counter/timing reduction. The unchanged c8 cell delivers 32.31 output tokens/s,
mean TTFT 95.20 s and mean ITL 302.0 ms.

Timing is off in baseline cells. Separate instrumented cells measure per-rank
planning, copy submission, map-source waits and GPU copy intervals. Their
throughput is not performance evidence. At c8, the two ranks copy 1.086/1.142 TB
and report 71.49/73.31 s in copy intervals, with 51.28/52.95 s exposed copy waits.
CPU map-source waits total 8.82/8.99 s per rank. Samples are complete, with zero
dropped or pending events. These rank-local sums must not be added and called
wall time, and CPU/GPU intervals can overlap.

## Candidate

The symmetric CT loader synthesizes both packed zero-point banks as int32
`0x77777777`, meaning logical zero 8 in GPTQv1 format. Validate them once, fill
their GPU slots once and fence initialization before cross-stream use. Omit
these two immutable payloads on subsequent misses. Generic caches retain the
six-payload default. Native copies accept four or six arrays through the same
existing schema, with full validation before writes.

`expert_bytes` remains the six-payload slot cost. `copy_expert_bytes` measures
actual per-miss transfers; copy, prefetch, adoption and cancellation counters use
that value. A quiescent development RPC toggles the redundant zero copies for
same-server A/B without changing pools, residency policy or routing.

The candidate passes 866 GPU 7 regressions, including native four/six-payload
validation, static-zero churn, cross-stream first use and reuse, and prefetch
adoption/cancellation. The client probe's 29 CPU tests pass separately. Native
registration retains 159 ROCm ops (144 RDNA-named), including all-reduce.

## Build and copy-only evidence

`build.py.txt` rebuilds only the host copy-planning object using the existing
ROCm 7.14 compile/link commands. It checks the cached build's host code against
the installed baseline before reusing objects, preserves a remote baseline
backup, and verifies that the native GPU fatbinary section is unchanged.
`build-receipt.json` retains exact commands and hashes. No binary is committed.

`copy-probe.py.txt` exercises the actual cache copy method with identical random
bindings per ABBA cycle, poisons mutable destinations and verifies exact payload
bytes. `copy-probe.json` retains 30 samples/mode/cardinality after warmup. Copy
interval medians improve about 9.8%, 4.3%, 2.0%, 1.3% and 1.1% for
1/4/16/48/96 misses. Actual bytes fall from 2,553,600 to 2,534,400 per miss
(0.752%); descriptors fall from six to four. Each pass still uses one native
batch API call. These are copy-only observations, not model throughput gains.

## End-to-end acceptance

`ab.py.txt` drives c1/c8 burst ABBA and a staggered c8 pair, all with 16k/1k
requests. Prefixes reset externally before each cell; expert history is retained
and is not identical between arms. Timing diagnostics stay off. Record actual
misses, transferred bytes, MTP acceptance, throughput, TTFT and ITL before
claiming a serving gain. Run semantic/PPL and prefix-reload gates separately.

End-to-end measurement and model-quality qualification are still in progress.
