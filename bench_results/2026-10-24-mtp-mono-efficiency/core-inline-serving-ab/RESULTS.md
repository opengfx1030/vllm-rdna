# MonoKernel barrier inlining qualification

## Patch

Add `__forceinline__` to the existing gfx1030 K2 `grid_barrier` device helper.
The arithmetic, synchronization, bounded wait, sticky error, poison writes,
workspace, M dispatch and TP all-reduce logic are unchanged. UNR stays4.
Three existing synthetic test objects now explicitly set `dense_mtp=False`
to match the current K2 interface; they previously failed before launching.

## Measured isolated results

Same-boot installed / unchanged JIT-O3 / inline JIT-O3 triples,50 samples per
M,32 captured graph replays per sample, all6 order permutations cycled.
The50 triples have counts9/9/8/8/8/8. Every M1..8,12,16,24 passed eager
parity, exact top-k, two changed-input graph replays, each graph versus fresh
eager, final timed replay parity, finite outputs and clear sticky records.

Quantized target:36.9–42.1% lower kernel latency versus matched JIT control.
Dense MTP:17.0–55.4% lower. Within-run paired median bootstrap95% intervals
exclude no gain for every row. These are warm-cache rank-local kernel
measurements, excluding TP collectives; they are not serving gains.

Actual G2 zero-budget barrier timeouts at M3/M24 passed complete output
poison, sticky retry, explicit-reset recovery bit-identical to fresh workspace,
and two changed-input graph checks. Observed timeout phase1; this does not
qualify every timeout phase or prove immunity to external GPU faults.

## Production-flags candidate

Exact original Ninja compile/link commands, retaining `-O2 -g -DNDEBUG` and
51 unchanged other objects. Source/header/object post-build verification PASS.
Generated HIP normalizes exactly to the tracked baseline and differs only by
the force-inline annotation. Installed source/core were not replaced.

- Original core SHA256: `6e8e486e548213d159236298a24b0f9ffbc18e49ccc234dcace7e8c1d59f4df3`
- Candidate core: `25f3fc556ad5b77aabb0faa561d19a7fc95addfd8692bbda0ea26441d8d9913f`
- Original source: `101e623a65c1cbb0057845b6accd35d20a04b0f30d6faa00c3684418ef6d5fd1`
- Candidate source: `aec873933fde435846249935243945a6fe9a53572c3f6435d4e46bdafca3a5bf`

All four production kernel buckets drop from128 to109 VGPR, private segment
856–872 bytes to0, LDS25296→25280 bytes. SGPR spills20–22 remain (to VGPR);
this is specifically elimination of private-memory/VGPR spilling. Wave32 PASS.
The candidate core passed 26 native K2 tests: eager and captured-graph widths
1–24, captured MTP control reset, and NaN fail-stop. The test log reports
`26 passed, 306 deselected` in 10.02 s. A separate shared-barrier K1/GDN suite
adds 61 passed tests, for 87 completed native tests total across the two
focused suites. The first K2 attempt found old `SimpleNamespace` test fixtures
missing the current `dense_mtp` field; the field was added in a disposable test
overlay and the suite rerun. The original failed-before-kernel log is retained.

Build and test provenance:

| Evidence | File |
| --- | --- |
| Disposable exact-flags build procedure (byte-preserving snapshot) | [`build_disposable_core.py.txt`](script-snapshots/build_disposable_core.py.txt) |
| Build inputs, commands, other-object hashes, and output hash | [`build_manifest.json`](../large-m-mono-core-inline/build_manifest.json) |
| Generated HIP to tracked CU normalization | [`hipify_normalization.json`](../large-m-mono-core-inline/hipify_normalization.json) |
| Post-build source formatting/token identity check | [`postbuild-formatting.json`](../large-m-mono-core-inline/postbuild-formatting.json) |
| Disposable serving-tree allowed-file and hash map | [`overlay_manifest.json`](../large-m-mono-core-inline/overlay_manifest.json) |
| Build verifier and remaining executed script snapshots | [`verify_disposable_core.py.txt`](script-snapshots/verify_disposable_core.py.txt), [`SHA256SUMS.txt`](script-snapshots/SHA256SUMS.txt) |
| Full binary/source/header/other-object verification | [`core-inline-postbuild-verification.json`](../large-m-mono-run/remote-evidence/core-inline-postbuild-verification.json) |
| Focused 26-test runner and completed log | [`run_production_inline_k2_tests_v2.sh.txt`](script-snapshots/run_production_inline_k2_tests_v2.sh.txt), [`production-inline-k2-tests-v2.log`](../large-m-mono-run/remote-evidence/production-inline-k2-tests-v2.log) |
| Shared-barrier K1/GDN suite | [`run_production_inline_k1_gdn_tests.sh.txt`](script-snapshots/run_production_inline_k1_gdn_tests.sh.txt), [`production-inline-k1-gdn-tests.log`](../large-m-mono-run/remote-evidence/production-inline-k1-gdn-tests.log) |

The ABBA driver, validation gate, analyzers, evidence exporter and build/test
launchers are preserved byte-for-byte as `.py.txt`/`.sh.txt` under
[`script-snapshots/`](script-snapshots/). Restore the original suffix before
running a snapshot. `SHA256SUMS.txt` records every snapshot digest.

The K2 test command loaded the disposable core by its explicit SHA256 and
selected `test_k2_matches_today`,
`test_mtp_control_reset_is_captured_and_clears_sticky_error`, and
`test_k2_fail_stop_writes_nan`. It ran on HIP6 with the reserved-GPU check,
and verified installed source/core hashes and boot ID did not change.

## Serving qualification

Fresh serial TP4/EP4 arms, HIP6–9, same immutable AWQ/PLE snapshots, native
MTP2 drafting,0.92 memory utilization,4,026,531,840-byte KV budget, LPTH1024,
prefill interval2, greedy temperature0, benchmark/corpus seed12345, cold prefix reset and
detailed requests for1k/512 and16k/1k at c1/c8. All-rank TunableOp lookup-only
profile is hash-matched to frozen rocBLAS `f30bb442e9b5`. No timed profiling.

The fresh original-core TP4 serving control started 2026-10-24 06:48:56 CEST
on boot `7a8851c9-87c4-4dc0-8f5b-3e172aa84773`. Its completed first arm is
captured under [`remote-evidence/01-control/`](remote-evidence/01-control/),
with detailed metrics summarized in
[`control-first-arm-metrics.json`](control-first-arm-metrics.json). The control
arm passed startup, PPL, prefix, concurrency, after-load and fault gates and
completed all four timed cells. The first candidate arm also completed all80
requests and passed startup/PPL/prefix/concurrency/after-load and fault gates,
including0 hard failures among48 normal-EOS/churn checks. Worker mappings
verified all four ranks on physicalHIP6–9 with the candidate core.

All four arms completed and passed the serving gates. The curated evidence and
per-request/per-cell analysis are in
[`../core-inline-serving-reviewed/`](../core-inline-serving-reviewed/).
`abba-repeat-comparison.csv` there records both paired comparisons, within-arm
repeat spreads, and descriptive two-repeat means. The two repeats are only two
serving-run observations per variant; no significance claim is made.

Across the two-run means, whole-run input/output throughput increased in all
four cells by +29.6%, +6.6%, +4.2%, and +1.9% (short c1, short c8, long c1,
long c8). Mean per-request decode rate increased +30.2%, +6.7%, +11.3%, and
+9.3%; median stream ITL fell 17.5%, 11.5%, 17.4%, and 9.9%. The TTFT-derived
prefill wave proxy is not measured prefill throughput: mean per-request
prefill rate changed +0.2%, +3.1%, +0.1%, and −5.4%. Mean TTFT changed −0.2%,
−2.4%, −0.1%, and +4.2%; thus long c8 TTFT regressed by about 0.77 s.

Acceptance length changed +0.170, −0.093, −0.287, and −0.065 tokens. The
candidate therefore improves decode/ITL and whole-run throughput, while draft
acceptance drops in three cells, most sharply at long c1. Fixed-prompt
cross-arm log-probability checks also show generated-token differences, with
within-variant repeat drift present. Treat this as a mixed result, not a
universal quality/performance win; the two-run means are descriptive and do
not isolate a pure kernel effect.

The API server logs engine seed0, unchanged in both arms; the benchmark command
uses corpus/request seed12345. The engine seed was not overridden. Timing uses
greedy sampling, and the exact engine/request settings are retained in logs.

The initial sequence stopped before launching arm03 when a cleanup descendant
briefly retained arm02's GPU lock. A direct audit found no lock owners, KFD
owners or listeners. `resume_validation_sequence_v2.sh` preserved the
original failure log, required completed01/02 provenance, waited for lock
release, and resumed only03-candidate/04-control. No completed cells were
rerun.

The test totals are 26 K2 tests plus 61 shared-barrier K1/GDN tests. Both
completed logs are now available in the linked evidence paths above.

## Correctness scope and follow-up diagnostics

Serving quality gates passed for the tested workloads; that does not prove
exact native-MTP equivalence at every MonoKernel row count. A separate strict
quantized-target rank-local check against the module path failed at M3, M7 and
M24 route-ID comparison, and at M24 output comparison (maximum absolute error
0.19696044921875, tolerance 0.005). It excludes serving collectives and blocks
a blanket full-MTP correctness claim. See
[`production25f3-vs-modules-v2.json`](../large-m-mono-run/remote-evidence/production25f3-vs-modules-v2.json).
