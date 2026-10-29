# Expert/KV byte-budget comparison

Source: `5dccb11e7a`; Flash-Next AWQ INT4, HIP 6–7, TP2/EP2, MTP2,
FULL_AND_PIECEWISE, native expert copy and K2 gate 8. Live KV paging, route
recording and transfer timing were disabled. CPU KV budget: 16 GiB total.
Both GPU and CPU prefix caches reset per cell. Expert history carries between
cells. Fixed workload seeds/order match across launches. Random performance
outputs force EOS continuation and are not semantic quality evidence.

`baseline` and `budget88` are first passes, including startup shape effects.
`*-repeat` are warmed repeats. Each comparison is one sequential configuration pair plus its repeats;
MTP/routing trajectories and expert history can differ. Use the observations
as workload-specific evidence, not an isolated causal or universal speedup.

## Controlled memory

Per rank, uniform96 uses 11,767,037,952 expert/map GPU bytes plus
4,026,531,840 configured GPU KV bytes. The 88% plan uses 10,337,021,952
expert/map GPU bytes plus 5,456,547,840 configured GPU KV bytes.
Both equal **15,793,569,792 bytes**. Other model/graph/runtime allocations
are outside this controlled sum. Page rounding leaves small unused KV slack.
GPU physical pages increase 139 → 188; CPU chunks stay 296/rank.
Expert RAM stays 31,378,685,952 bytes/rank. The plan uses 48–160 expert slots
per layer; uniform96 uses 96 everywhere.

The owned HIP 6–7 workers map to SMI 0–1 / PCI 0000:07:00.0 and
0000:0a:00.0 on this boot. The sampler reads only these PCI sysfs paths.
Sampled device-wide VRAM peaks are 30,750,601,216 and 30,717,046,784 bytes
out of 32,195,477,504 each, leaving roughly 1.45 GB; this includes allocator
reservations and contexts and does not prove all remaining bytes are safe KV
headroom. Sampling covers part of the candidate warmed pass, not every stage.

## Warmed results

| Cell | Output tok/s, uniform → 88% | Mean TTFT s | Mean ITL ms |
| --- | --- | --- | --- |
| 1k/512 c1 | 44.41 → 30.27 | 1.16 → 1.23 | 58.9 → 73.3 |
| 1k/512 c8 | 69.17 → 62.87 | 5.51 → 5.70 | 190.8 → 216.4 |
| 16k/1024 c1 | 23.45 → 19.03 | 16.17 → 21.45 | 69.3 → 79.5 |
| 16k/1024 c8 | 31.20 → 31.13 | 96.18 → 79.40 | 307.6 → 400.9 |

Long c8 admits eight requests together in the candidate versus six running
and two waiting in the baseline. Candidate mean/p99 TTFT improve 17.4%/34.3%,
and sampled waiting request-seconds fall about 24.8%. Throughput is unchanged
(−0.2%), mean ITL worsens 30.3%, and expert-copy bytes rise 22.6%.
All cells completed with zero recompute preemptions.

Keep uniform96 as the general default. The 88% plan is an optional admission/TTFT
tradeoff, not a throughput improvement. Initial-pass comparisons reach the same
conclusion: long c8 throughput 31.67 → 31.36, TTFT 94.96 → 81.24 s, ITL
298.9 → 421.9 ms. Candidate quality after both matrices passed PPL 4.83 and
48 concurrency checks with zero hard failures.

## Bottlenecks and next route

Reduced-capacity layers add more transfers than enlarged layers save. In the
repeated short c8 cell, excess misses are largest in layers 31 (48 slots),
27, 33 and 25 (64 slots). In repeated long c8, layers 11, 10, 27 and 39
lead the excess traffic. These are observed counter differences, not isolated
layer latency measurements.

The repeated short c1 outlier also changes MTP mean acceptance from 2.909 to
2.388 and increases misses in some enlarged layers. Its entire slowdown cannot
be assigned to capacity alone. The 61-call decode-only planning trace did not
exercise frequency decay and is insufficient evidence for permanent allocation.
Retain the expert budget for the next nonuniform-plan experiment; gather longer
representative c1/c8 prefill and decode traces first. Evaluate live request KV
spilling separately: these no-preemption results do not establish its speedup.

`summary.json` contains exact metrics, byte deltas, MTP acceptance and sampled
admission totals; `summarize-tiered-budget.py.txt` reproduces it. The raw archive retains
each arm's benchmark results, per-layer before/after receipts and one-second metrics.
Raw matrices are in `raw-matrices.tar.gz`; every member is hash-verified in
`raw-matrices-manifest.json`. To reproduce from the repository root, extract it
into this directory, then run `python3` on `summarize-tiered-budget.py.txt`.
The archived inputs reproduce every published summary value exactly.
Server/client logs are gzip-compressed; `log-scan.json` records server hashes
and numeric/fault markers. The baseline has one output-handler EngineDeadError
traceback during deliberate shutdown after all cells and the idle layout RPC;
no serving numeric/GPU-fault marker was found. Kernel-ring access was unavailable.

## Uniform96 with normal-threshold RAM paging

`live90` / `live90-repeat` use source `292e0b0f70`, the same expert and KV
budgets as baseline, and a 90% live-paging watermark. Overlap remains off.
The initial churn diagnostic is separately archived in
`admission-churn-before-fix/`; the corrected comparison does not include it.

| Warmed 16k/1024 c8 | Prefix-only baseline | Active RAM paging |
| --- | --- | --- |
| Output tok/s | 31.20 | 32.44 |
| Mean TTFT | 96.18 s | 76.93 s |
| p99 TTFT | 215.84 s | 137.35 s |
| Mean reported ITL | 307.6 ms | 371.1 ms |
| Longest streaming gap | 2.33 s | 125.99 s |
| Mean request completion | 224.33 s | 226.82 s |
| Longest request completion | 262.58 s | 252.55 s |

The first paging pass measured 32.06 output tok/s, 76.53 s mean TTFT and
353.6 ms mean ITL. Each pressure cell uses two snapshots; all four stores and
restores preserve nonaligned token frontiers. Across both cells, private
snapshot D2H and H2D are each 5,098,176,512 bytes (ordinary prefix traffic is
additional). The earlier policy did eleven snapshots in one cell. Fixed
admission ordering avoids immediately consuming pages just freed for newcomers.

This is whole-request suspension: the victim waits in RAM until enough GPU
pages are available. Earlier admission improves TTFT but can create a long gap
in an existing stream. Keep live paging off for ordinary interactive serving;
consider it only as an explicit capacity/batch/TTFT tradeoff. The approximately
4% aggregate throughput difference is an observation from this pair, not a
universal speedup. MTP trajectories differ, and cells below the watermark do
not page at all; their throughput variation cannot be credited to paging.

Final post-load quality: PPL 4.78 and 48 checks with zero hard failures. A real
CPU-prefix reload after GPU-only reset passed: 32 identical output tokens,
zero maximum token-logprob delta versus the GPU-cached reference, positive
347,602,944-byte stores and 289,669,120-byte loads. External CPU state was
preserved. Final GPU+CPU prefix reset succeeded; the API and TP workers stopped.
`live90/qualification-summary.json` and `runtime-manifest.json` retain details.

MTP streaming emits token chunks; the reported ITL follows the benchmark's
chunk timing. The longest streaming gap is included explicitly because rare
whole-request suspension is not adequately described by a mean or p99 alone.

## Soft suspension target

`fair10-final` / `fair10-final-repeat` use source `90b38f9ad9`, a 10 s soft
suspension target and the same uniform96/GPU-KV/CPU budgets. They repeat only
the fixed-seed 16k/1024 c8 cell on a separate launch; their expert-history
sequence differs from the earlier full matrices. Both prefix tiers reset per
cell. These are sequential observations, not an isolated causal speedup.

First/repeated output throughput is 31.09 / 30.59 tok/s; worst streaming gaps
are 17.62 / 18.21 s. The repeated cell improves the older live policy's
125.99 s gap, while prefix-only serving remains at 2.33 s. Compared with the
prefix-only repeated baseline, throughput changes −1.9%, mean TTFT −6.3%,
p99 TTFT +5.3%, mean reported ITL +24.3% and mean completion time about +8.2%.
Keep active paging off for the general default; the smaller mean TTFT does
not justify claiming an overall efficiency gain.

There are 14 / 15 matched spill/resume pairs, with all frontiers preserved
and zero recompute preemptions. Private snapshot copies in the repeated cell
total 19,349,897,216 bytes in each direction across both ranks, compared with
2,549,088,256 bytes under the older two-spill policy. Expert copy bytes also
rise about 2.8% versus prefix-only; routing/MTP trajectories differ.
The pool allocation stays unchanged. Cumulative copied bytes exceed the
allocated CPU tier because chunks are reused across cycles.

Repeated phase maxima are quiesce 2.598 s, store 0.873 s, host wait 14.085 s
and load 2.833 s; these include scheduling and rank acknowledgement delays.
The maximum quiesce-to-first-resumed-scheduling interval is 16.004 s, and
the measured stream gap includes surrounding output/compute intervals.
The target is soft; transfer duration, protected residents and memory
availability can extend it.

`fair10` is the failed diagnostic before the retained-MTP-tail exemption:
restore-ready time alone missed a 98.78 s streaming gap. `fair10-fixed*` are
intermediate full matrices after that exemption, before the independent review
found the incomplete-prefill starvation case. They are preserved separately
from final-source qualification. The final guard allows existing prefills to
progress when an overdue restore cannot reserve pages or select a victim,
while fresh admissions remain blocked. Its regression fails before the fix
and passes afterward.

Final-source PPL, normal-EOS quality, prefix reuse, CPU reload and cleanup
receipts are in `../bounded-final-quality/`. To reproduce every summary value,
extract `raw-matrices.tar.gz`, `bounded-intermediate-raw.tar.gz` and
`bounded-final-raw.tar.gz` into this directory, then run
`summarize-tiered-budget.py.txt` from the repository root. Each archive has a
hash manifest; the script also verifies benchmark spill/restore frontiers from
the retained compressed server logs. Original six-arm summary values are
unchanged.

The final server log retains startup diagnostics for an optional gfx1250 Triton
import and Qwen VL processor documentation. The qualified recipe uses native
AWQ HIP and language-only serving; those other paths are unqualified here.
No matched serving numeric/GPU fault marker was found in the final log.
The intermediate server has an output-handler EngineDeadError traceback during
deliberate shutdown after its matrices completed. `log-scan.json` preserves and
classifies the markers; kernel-ring health was not audited.
