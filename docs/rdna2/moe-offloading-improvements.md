# RDNA2 MoE offloading: first optimization pass

This first pass follows #146. The default copy backend and pool
budgets are unchanged; native batching remains explicitly selectable.
The existing post-step fault check now also monitors the model-offload workspace.
The qualified baseline and receipts live in `moe-offloading-target.md` and
`bench_results/2026-10-28-moe-offload/`.

Changing active expert sets now updates the existing pinned int32 slot map
through a persistent zero-copy NumPy view. This removes scalar Tensor dispatch
without changing the map address or the previous-upload completion fence.
On the build server, matched CPU `PackedExpertCache.ensure/release` hit-only
passes with changing active sets took 42.23 → 34.62 µs for one expert,
83.42 → 37.73 µs for ten, 178.81 → 43.31 µs for 32, and
317.51 → 51.18 µs for 64 (ABBA, 30 samples per arm). These are CPU control-path
measurements, excluding GPU event and DMA time; they do not predict model speedup.
All 147 focused GPU tests passed, including real CT overflow and cross-stream
parity for both copy backends. The reproducible benchmark and receipts live in
`bench_results/2026-10-29-moe-offload-first-pass/`.

## First-pass serving results

The final graph run used TP2/MTP2 Flash-Next AWQ text serving on GPUs 6–7,
FULL_AND_PIECEWISE segmented graphs, 96 expert slots per layer, and CPU KV
offloading enabled. Both backends passed greedy, PPL (native 4.82 / tensor 4.83),
prefix reuse and 48 normal-EOS concurrency checks each with zero hard failures.
The 16-arm ABBA matrix completed 72 forced-length requests and 55,296 output
tokens; every request reached its exact length. Post-load greedy checks passed
with zero numeric or new GPU fault markers.

Eager mode (`enforce_eager=True`, graph mode NONE) also passed greedy, PPL 4.84,
prefix reuse, 48 normal-EOS concurrency checks with zero hard failures, the
1k/512 and 16k/1k c1 native-copy workloads, and post-load greedy checks. Both
workloads reached their exact output lengths; no numeric or GPU fault markers
were observed. All new qualification used only GPUs 6–7.

Both GPU and external CPU prefixes are now reset before each measured arm.
The client rejects warm or unavailable hit counters; all measured arms had
zero prefix hits. Expert residency remains order-dependent and MTP acceptance
varies, so these are model observations rather than a causal backend speedup.
Full prefill/decode/TTFT/ITL ranges and transfer counters are retained in
`graph-performance-summary.txt`, `graph-performance-summary.json`, and
`graphs-cold/copy-abba/qualification.json` under the first-pass receipts.

At 16k/c1, native TTFT was 16.06–16.12 s versus tensor 17.10–17.66 s.
At 16k/c8, aggregate output was 31.91–32.63 tok/s native versus
24.17–25.10 tensor. Short c1 decode was mixed, with substantially different MTP
acceptance. The default remains tensor; there is no across-the-board speedup
claim. Pool budgets and the K2 gate stay unchanged.

## Earlier diagnostic history

An earlier long native-copy MTP2 test failed with NaN logits at 810/1024 output
tokens. Its cause has not been causally attributed to native copies or the GDN
defect below. The failing receipt remains preserved alongside subsequent passes.

With CPU KV disabled and per-layer finite checks enabled, the subsequent tensor
and native 16k/c1 arms both completed all 1024 tokens without NaNs. These two
changed conditions prevent attributing the original failure to either one.
The joint CPU-KV-enabled test also completed both arms with finite checks on
and off, while the new post-step completion monitor remained enabled. This
does not exclude a pacing-sensitive failure.

A separate GDN defect is reproducible: its Q32 squared-sum accumulator overflows
for large finite FP16 core values, producing Inf or NaN during RMS normalization.
FP32 partial sums in fixed tile order cover the finite FP16 range without an
additional kernel launch. GPU 7 validation passed 17 diagnostic tests and 49
default-path tests (four diagnostics-only skips), including existing recurrence,
state and repeatability parity. The receipts are in the `gdn-range/` subdirectory.
The earlier serving NaN has not yet been tied to this defect. Full TP2/MTP2
testing on GPUs 6–7 completed 58 forced-length requests and 40,960 output tokens
with the fix: short c1/c8 alternating arms, long c1 tensor/native pairs with
post-step checks off, completion-wait-only and collective, then long c8 arms
under the normal collective guard. Every request reached its exact output
length. CPU KV was enabled throughout; counters confirm external-prefix hits
and restores. No GDN nonfinite, K2 fault, logit NaN or new amdgpu fault markers
were observed. Greedy, PPL (4.86), prefix reuse (11.85 s cold / 2.31 s warm) and
post-load greedy checks passed. Receipts and launch scripts are in
`bench_results/2026-10-28-moe-offload-improvements/gdn-serving-diagnostic/`.
This bounded passing replay does not establish the cause of the original NaN
or qualify a causal model-level copy-backend speedup.

## Implementation status

`expert_cache_copy="native"` submits a bounded pass of missed experts through
one C++ call. HIP 7.14+ uses `hipMemcpyBatchAsync` when the loaded runtime exports
it; the fallback submits individual copies inside C++. It preserves all six SoA
payload pools, graph addresses, the pinned CPU bank and consumer-stream fencing.
The default remains `"tensor"`; native batching is an opt-in tested on HIP 7.14.
Older-HIP fallback has not received runtime qualification.

Focused validation: 73 local tests passed (25 GPU-dependent skips), and 29
integrated cache/native-copy tests passed on GPU8 gfx1030/HIP 7.14 without skips.
An additional 16 eviction/graph tests passed for widths through 24 and TP2/TP4
partition geometries on one GPU. These are not full TP4 tests. Receipts are in
`bench_results/2026-10-28-moe-offload-improvements/`.

The optional development worker extension reports per-rank pass demands, misses,
bytes and native/batch submission counts. Its quiescent backend switch permits
same-process A/B with retained expert residency and captured compute graphs.
`python -m tools.rdna.mono.qualify_expert_copy` alternates arms and records these
counters alongside latency and throughput. Both prefix tiers reset between cells;
expert-cache histories remain order-dependent, so alternating repeats are needed.
No causal model-level native-copy speedup is claimed.

The integrated extension subsequently passed all 119 focused GPU tests without
skips. Full TP2 native-copy greedy, PPL (4.78), prefix-cache and concurrency
(48 checks, zero hard failures) gates passed with zero GPU faults. The first
performance hook stopped on an unexported KV-load metric, and its client gate
was fixed without engine changes. The final cold-prefix matrix is described above.

The matched gate-8/gate-24 serving experiment completed all four cells and
quality gates without GPU faults. Gate 24 changed aggregate output throughput
by +19.33% at 1k/c1, -14.04% at 1k/c8, -1.57% at 16k/c1 and -5.16% at 16k/c8.
The default stays 8. These are one serial run per arm with different expert-cache
histories and MTP acceptance; they do not isolate kernel execution time. Raw
results and forced-output warnings are retained in the improvements receipts.

## Baseline and constraints

TP2/EP2 uses 96 GPU expert slots per layer, LFU/LRU with decay 128,
segmented FULL_AND_PIECEWISE graphs, MTP2, prefix caching, and an optional
16 GiB aggregate CPU KV pool. Cold expert weights remain in pinned RAM.
Active experts use the same GPU slots; there is no separate active-weight pool.

Per GPU, model-load allocation is 21.39 GiB, including 10.959 GiB of expert
slots. The explicit GPU KV pool is 3.75 GiB. Capture-era allocation grows by
2.12 GiB; this is not a pure graph-memory census or a peak-headroom guarantee.
Across TP2, host expert banks occupy 58.447 GiB, pinned INT4 PLE tables about
29.802 GiB, plus the configured 16 GiB CPU KV pool. Budgets are fixed at startup.

Each GPU owns 256 whole experts out of 512. Dense projections are tensor
parallel; routed experts are expert parallel, with partial outputs reduced.
The default K2 row gate is 8, so MTP2 concurrency 8 verification at 24 rows
uses the cached modular MoE fallback. Native K2 support alone does not enable it.

Measured 16k/1k aggregate output throughput is 19.767 tok/s at c=1 and
26.718 tok/s at c=8, including prefill. Mean TTFT rises from 17.307 s to
106.260 s, and mean TPOT from 33.721 ms to 148.441 ms. These observations
identify a scaling limitation, not its causal attribution. Hosted CI was
skipped; focused local and dedicated GPU validation passed for #146.

## Later passes

- [ ] Capture CPU/HIP timelines and per-rank expert hits, misses, transfer bytes,
  readback waits, collective time, and peak memory. Distinguish queue time from
  prefill execution; do not infer bottleneck percentages from wall time alone.
- [x] A/B the 24-row K2 gate with the current 8-row default. Keep 8 after the
  observed c8 regressions; further wide-K2 work belongs with #110.
- [x] Batch the six payload copy submissions per missed expert and measure
  submission overhead versus transfer time, preserving pinned-buffer lifetime.
- [ ] Reduce blocking route readback and host planning overhead without changing
  routing, eviction protection, overflow behavior, or graph replay correctness.
- [ ] Evaluate route-informed prefetch and overlapping transfers with computation.
  Bound wasted bytes and protect demand transfers from speculative traffic.
- [ ] Measure per-layer demand and rebalance expert slots within a fixed budget.
  One additional slot across all 48 layers costs 116.895 MiB per GPU; adding
  32 slots costs about 3.653 GiB per GPU before other transient allocations.
- [ ] Optimize deterministic dense accumulation while retaining established
  parity. Isolated 4096-row QKVZ cost rose from 8.844 to 14.173 ms; a whole-model
  A/B is required before claiming an end-to-end improvement.
- [ ] Measure collective crossover and rank imbalance, coordinating with #111
  and #129. Current custom all-reduce eligibility ends at 64 KiB; a 24-row
  FP16 hidden-state tensor of width 2560 exceeds this bound.
- [ ] Evaluate coordinated expert/KV transfer priorities and memory budgets.
  Current native KV offload restores eligible prefix/hybrid state; it is not
  arbitrary active-context paging. Keep raw QSA-ring restore restrictions.
- [ ] Extend correctness and regression coverage to affected platform paths and
  reproducible mixed workloads, including prefix reuse and cache churn.

## Acceptance gates

Every runtime optimization needs an isolated commit, eager and graph parity,
TP2 cache-churn coverage, MTP2 and prefix-cache checks, and normal-EOS semantic
probes. Keep forced-length throughput probes separate from semantic evaluation.
Report prefill tok/s, decode tok/s per request, TTFT, ITL and TPOT for 16k/1k
and 1k/512 at c=1 and c=8, plus peak GPU/host memory and expert transfer counters.
Use identical seeds, launch configuration and warmed tuning rows for A/B runs.

Only physical GPUs 6 and 7 are allocated. Full-resident TP2 needs about
39.655 GiB per GPU before KV and graphs, so it is not a feasible control on
these V620 devices. Use a feasible matched control and state its limits.
