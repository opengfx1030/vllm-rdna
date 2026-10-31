# Tiered offload qualification receipts — 2026-10-29

Target: Flash-Next AWQ INT4, gfx1030, TP=2/EP=2, MTP2,
FULL_AND_PIECEWISE graphs. Only HIP GPUs 6–7 were used for serving;
physical-page and regression tests used GPU 7.

## Verified so far

- `bounded-scheduler-regression-tests.txt`: 819 tests pass on GPU 7, including
  the retained MTP tail scheduler regression and incomplete-prefill progress.
  `prefill-starvation-red-green.txt` verifies failure before the progress fix.
- `restore-fence-regression-tests.txt`: earlier 801-test suite passes after the
  admission-order/readiness fixes and restore-destination prefix-reader fence.
- `live-pre-serve-tests.txt`: 792 regression tests passed, including typed hybrid
  snapshots, CPU leases, connector transactions, cancellation, prefix caching and
  allocator-backed GPU → RAM → GPU physical page copies.
- `graphs-initial`: uniform 96-slot quality qualification; greedy, PPL 4.83,
  prefix reuse and 48 normal-EOS concurrency checks passed.
- `overlap-quality`: known-demand overlap PPL 4.80 and concurrency quality passed.
- `overlap-abba/results.json`: off/on/on/off aggregate output throughput was
  51.25 / 52.93 / 47.85 / 51.83 tokens/s. Expert history was retained and MTP
  trajectories differed. This does not establish a consistent overlap benefit;
  overlap stays disabled by default.

## Diagnostic and modeled results

`profile` contains instrumented measurements, not clean throughput benchmarks.
Transfer intervals include enqueue/scheduling gaps, and a large rank-0 layer-16
wait outlier must not be interpreted as isolated DMA service time.
Optional KV transfer counters absent from the server are recorded as unavailable.
The kernel ring could not be read; log scans do not establish hardware-ring health.
`profile/server-info.json` is sanitized configuration only.

`policy-replay.json.gz` replays 61 decode calls per rank/layer from empty caches.
Frequency decay at 128 calls was not exercised. Original route files/manifests
are hash-verified in `decode-route-traces.tar.gz`. Prefill and previous residency
are excluded. Slot plans are candidates, not measured speedups or global optima.

- `slot-plan.json`: same expert byte budget, 48–160 slots per layer; about 7.2%
  fewer modeled decode copy bytes than uniform 96.
- `slot-plan-0.88.json`: releases 1.33180618 GiB/GPU for about 2.6% more modeled
  decode copy bytes. The launch recipe explicitly caps GPU KV at 3.75 GiB;
  using those freed bytes requires `KV=5456547840` (about 5.08 GiB).

The initial live-paging model run passed PPL 4.87, 48 normal concurrency checks
and 56 checks with eight long contexts (zero hard failures). Logs show repeated
spill/resume at identical nonaligned frontiers between 25.9k and 29k tokens.
There were 31 paired spill/resumes, all at equal token frontiers; the only
unpaired spill belongs to the intentionally disconnected request. Eight surviving
requests finished after that disconnect, post-abort PPL passed, and native cache
reset succeeded. The 12k-word prefix probe measured 11.67 s cold versus 2.30 s
warm TTFT. The 3k-word probe showed no benefit (3.25 s versus 3.16 s) and is
retained as a failed timing gate. See `live-qualification/`.
The matched expert/KV matrix is complete in `budget-matrix/`: the warmed 88%
plan changes long c8 throughput 31.20 → 31.13 output tok/s, mean TTFT
96.18 → 79.40 s, and mean ITL 307.6 → 400.9 ms. Shorter cells regress;
uniform96 remains the general default. `eager-live-qualification/` adds PPL 4.80,
56 checks with zero hard failures and 28 matched-frontier spill/resumes. Combined
eager overlap/live paging passed PPL 4.86 and 48 checks; all 802,067,884,800
known-demand prefetch bytes were adopted, with zero unused bytes or failures.
These are correctness receipts, not an overlap performance claim.
Live paging uses a forced zero watermark during correctness qualification only;
that setting is not a recommended serving default.

The earlier normal-threshold paging comparison is complete in `budget-matrix/`.
Uniform96 plus active RAM paging measures 32.44 versus 31.20 output tok/s and
76.93 versus 96.18 s mean TTFT in warmed long c8, but the longest streaming gap
is 125.99 versus 2.33 s. Keep the general default at uniform96 with ordinary
CPU prefix reuse and active paging off. The final source also passes PPL 4.78,
48 checks, and real CPU-prefix reload with 32 identical tokens and zero maximum
token-logprob delta. Both prefix tiers reset successfully and owned servers stop.

## Soft suspension target — source `90b38f9ad9`

The opt-in pager now prioritizes overdue restores, protects recently resumed
requests, and defers fresh admission and prefill work during actual transfers.
Existing prefills can progress when no transfer can start, avoiding a deadlock
with an overdue snapshot. Retained MTP tail replay passes the prefill gate.
Logs measure both restore readiness and the first resumed scheduling step.

The final-source targeted 16k/1024 c8 first/repeat measured 31.09 / 30.59 output
tok/s and 17.62 / 18.21-second worst streaming gaps. The repeated cell measured
90.13 s mean TTFT, 227.25 s p99 TTFT and 242.76 s mean completion time.
Prefix-only baseline was 31.20 tok/s, 2.33 s worst gap, 96.18 s mean TTFT,
215.84 s p99 TTFT and 224.33 s mean completion. The previous live policy had
a 125.99 s gap and 32.44 tok/s. The target reduces large suspension outliers,
but does not establish a general efficiency gain; keep active paging off by
default. These final cells use a separate launch and expert-history sequence
from the earlier full matrices. Both prefix tiers reset before each cell.

All 29 benchmark spill/resume pairs preserved their token frontiers, with zero
recompute preemptions. The repeated cell copied 19.35 GB of private snapshots
in each direction across both ranks, within the same configured RAM/GPU pools.
See `budget-matrix/summary.json` for exact counters and phase durations.

Final-source quality passed PPL 4.78 and 56 normal-EOS checks: zero hard failures,
13 near-tie, four soft and 39 identical results. Prefix TTFT improved
14.32 → 2.71 s. Native CPU reload preserved 32 greedy tokens, with maximum
token-logprob delta 0.03638 within the enforced 0.05 tolerance; this probe
compares available cache boundaries. Byte-identical physical-page transport
has separate regression coverage. Final cache reset passed, all owned processes
exited, and HIP 6–7 returned to idle VRAM usage. Receipts and runtime/source
hashes are in `bounded-final-quality/`.
