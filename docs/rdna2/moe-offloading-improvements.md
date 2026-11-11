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

## K2 offload: split demand with a zero slot

The offloaded K2 runs three launches per MoE layer: stage 1 (prefix, router,
route ids), one expert pass, stage 3 (finish). It used to read the route ids
back (`route_ids.cpu()`), plan the expert loads on the host, copy the missed
experts, upload a slot map and only then launch the pass. A rocprofv3 profile
of TP2 c=1 decode with `tools/rdna/prof/offload_gaps.py` (about 100 steps, 4806
K2 layers) showed the GPU idle for 451 µs of every 1036 µs layer (p50 381, p90
682): stage 1 end to pass start took 774 µs, 67% of the layers loaded at least
one expert, and in the median 86.2 ms step the GPU was busy 67.2% of the time.

A demand that fits in the cache (`rows × top-k ≤ slot_count`) is now split over
two launches. `resident_map` sends every expert resident before this layer to
its slot and every other expert to one extra, all-zero slot; `miss_map` does the
opposite for the experts this layer loads. The resident launch is queued right
behind the route-id readback, so the GPU works while the host waits for the
ids, plans and issues the copies. The miss launch follows the copies and one
upload of both maps, and is skipped when nothing loaded. Layers that only hit
run one launch and never wait for the host. Padded rows are masked out of the
route ids on the device, and the padding flags are read back once per forward
pass instead of once per layer.

The result is bit-identical to one full pass. Each launch counts the whole
demand, so both use the single pass's K-slice width and tile schedule; K2 sums
fixed-point integers, and the experts a launch does not own read the zero slot,
which adds exact zeros. Two cheaper variants were measured and dropped.
Counting only each launch's own experts changed the K-slice: c1 ITL p50 fell to
51.8–54.1 ms (from 62.4–64.0 ms in the same runs, which were not yet kept clear
of boots), but mean logprob drift was 0.007–0.017 and the greedy text differed
between repeated runs. A device-side gate (seven small kernels deciding whether
the resident pass may run) was exact but gained only 4% ITL: it cost about 70 µs
of GPU time per replay and miss layers replayed the pass twice.

Serving A/B on TP2/MTP2 (96 slots, graphs, copy backend `tensor`, 1024/512
random prompts, seed 777, 10 requests at c=1 and 12 at c=4). Each run measured
the old and the new tree at the same time on two GPU pairs, with no server
booting during the measurement; the second run swapped the pairs. A boot pins
about 90 GB of host memory and can stall a serving rank for seconds, long
enough to trip the mono kernel's 2 s barrier and poison the logits, so boots are
kept out of the measured windows. Cells show run 1, run 2.

| 1024/512 | BEFORE | AFTER | mean change |
|---|---|---|---|
| c1 decode tok/s per request | 31.86, 31.49 | 36.51, 36.81 | +15.7% |
| c1 ITL p50 / p99 ms | 64.9 / 107, 65.8 / 107 | 56.6 / 94, 56.3 / 93 | -13.6% / -12.6% |
| c1 step period ms (acceptance length / decode rate) | 67.7, 68.5 | 59.1, 58.6 | -13.6% |
| c4 decode tok/s per request | 12.12, 11.96 | 12.89, 12.84 | +6.9% |
| c4 ITL p50 / p99 ms | 173.5 / 314, 176.5 / 302 | 168.3 / 308, 162.7 / 311 | -5.4% / +0.5% |

MTP acceptance length was 2.16 at c1 in all four arms. c4 runs 12 rows, above
the K2 gate, so it never reaches the split; its gain was not isolated (the
cached modular path shares the cheaper policy planning and the single copy
stream). Greedy output and per-token logprobs of six probe prompts, sequential
and concurrent (702 tokens each), are identical to BEFORE in every graph-mode
arm: max |Δlogprob| 0.0000, sha256 5527d8b3bd52d7d5.

The profile shows where the time went. Its two windows differ in cache phase
(67% of the BEFORE layers and 77% of the AFTER layers loaded an expert), so the
layers are compared by class, mean µs per layer:

| K2 layer | BEFORE | AFTER |
|---|---|---|
| only hits: wall (share of layers) | 536 (33%) | 282 (23%) |
| loads experts: wall (share of layers) | 1284 (67%) | 1172 (77%) |
| stage 1 end to first expert pass, all layers | 774 | 38 |
| GPU idle inside the layer, all layers (p50) | 451 (381) | 232 (207) |
| wall at the BEFORE class mix | 1037 | 878 |

Tests: `test_rdna_expert_cache.py` covers the dummy slot, both maps and the
copy-only `by_residency` call. `test_amd_mono_rdna2_offload.py` runs the
orchestrator against a fake boundary (hit, miss, no-dummy, over-capacity and
failed-load cases) and, on GPU, checks with `torch.equal` that a demand split
over two or three launches that each count it all equals one full pass (TP2, 1,
3 and 8 rows). The three offload test files pass 144 tests on gfx1030 (109 on
CPU, the rest GPU-only), apart from three existing `ple` failures that also
fail on the base tree. A temporary 400-step random stress run (both expert
shards, 96 and 24 slots; eager, graph replay and modular passes interleaved)
matched one full pass bit for bit on all 1004 single-pass steps, with the
payload equal to the host bank and the dummy row still zero.

Eager mode (graph mode none) on the same two trees gives greedy sha256
07dd38b5327a6947 in both, all six probe prompts identical with max |Δlogprob|
0.0000; c1 decode 26.77 → 30.62 tok/s per request and ITL p50 76.1 → 67.2 ms
(4 requests). The graph validation gate on the new tree passed: greedy probes
3/3, PPL 4.84 (mean NLL 1.577), prefix reuse (cold 12.63 s, warm 2.36 s), 48
concurrency checks with 0 failing (34 identical, 9 near-tie, 5 soft) and no GPU
faults. A second run on another GPU pair (HIP 2,3) gave the same verdicts:
probes 3/3, PPL 4.84, prefix 13.01 s cold / 2.43 s warm, concurrency 0 failing
(33 identical, 12 near-tie, 3 soft), no faults. The identical count of that gate
ranges from 30 to 48 in earlier receipts; 0 failing is the criterion. Receipts,
scripts and source hashes: `bench_results/2026-11-02-moe-offload-k2-zero-slot/`.

Limits: the split needs `rows × top-k ≤ slot_count` and a cache with fewer slots
than experts. A demand that may outgrow the cache (TP4's 48 slots from five
rows, say) and a cache that holds every expert keep the old drain-then-pass
order. The cached modular fallback (MTP2 c4 and c8, above the K2 row gate of 8)
still reads back. The extra slot costs 116.895 MiB per GPU. The dummy launch
still occupies the GPU: the miss pass takes about 54 µs where a pass over only
the misses takes 29 µs. Layers that load experts are now dominated by the
copies, about 11 per layer: 566 µs of DMA plus fixed 8 µs gaps after the large
copies and 24 µs after the small ones. Only fewer misses or prefetch will
shorten that. The default copy backend is unchanged. 16k/1k, c8 and TP4 serving
were not measured.

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
- [x] Reduce blocking route readback and host planning overhead without changing
  routing, eviction protection, overflow behavior, or graph replay correctness.
  Done for the K2 path (see the zero-slot split above); the cached modular
  fallback still reads back.
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
