# DFlash2 head and attention optimization (2026-10-20)

This campaign starts from the already-qualified 1860-row rocBLAS profile and
commit `49880b2fc8`. Target weights, native MTP2 and all TunableOp rows stay
unchanged. Hardware: HIP 6–9, four Radeon PRO V620 (gfx1030), TP4/EP4,
ROCm 7.14.60850 and the installed v0.31.0 venv on `par1-cs25`.

The serving objective is a measurable DFlash2 advantage over native MTP2
at **both 16k/1k c=1 and c=8**. The old/new DFlash2 gains below are
intermediate results and do not meet that objective. PR #132 merged the
correctness and initial optimization work; the performance follow-up is
being tracked separately and remains unqualified.

## Controlled serving improvement

Both arms used the same boot, target snapshot, 1860-row profile, spec2,
TP4/EP4, HIP 6–9, greedy seed 12345, profiling off and detailed streaming
records. Each cell starts with a verified prefix-cache reset. Each arm
completed eight requests per c=1 cell and 32 per c=8 cell (80/80 total).

| Workload | Path | Input tok/s full run | Prefill tok/s/req (wave proxy) | Decode tok/s/req (aggregate output) | Mean TTFT | ITL median / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024/512 c=1 | Old head/FA | 132.4 | 2177.7 (2098.3) | 70.34 (66.19) | 0.470 s | 31.1 / 31.2 ms |
| 1,024/512 c=1 | FP16 head + S16 FA | 155.4 | 2175.4 (2140.8) | 83.51 (77.69) | 0.471 s | 26.7 / 26.8 ms |
| 1,024/512 c=8 | Old head/FA | 428.9 | 971.4 (2169.1) | 29.43 (214.45) | 1.054 s | 63.3 / 74.3 ms |
| 1,024/512 c=8 | FP16 head + S16 FA | 449.6 | 1001.0 (2181.6) | 31.11 (224.80) | 1.023 s | 57.4 / 67.9 ms |
| 16,384/1,024 c=1 | Old head/FA | 723.2 | 2050.1 (2037.0) | 69.77 (45.20) | 7.992 s | 32.7 / 32.6 ms |
| 16,384/1,024 c=1 | FP16 head + S16 FA | 805.0 | 2072.6 (2062.7) | 82.18 (50.31) | 7.905 s | 27.2 / 27.2 ms |
| 16,384/1,024 c=8 | Old head/FA | 1366.7 | 960.0 (2017.0) | 13.34 (85.42) | 17.067 s | 67.8 / 162.9 ms |
| 16,384/1,024 c=8 | FP16 head + S16 FA | 1419.2 | 917.8 (2044.2) | 14.13 (88.70) | 17.852 s | 58.6 / 149.9 ms |

Aggregate output gains were **17.37%, 4.82%, 11.31% and 3.84%**.
Median ITL fell 14.1%, 9.3%, 16.7% and 13.6%. Long c=8 mean TTFT
**regressed 17.067 → 17.852 s (+4.60%)**; median improved 8.902 → 8.768 s
and p99 64.982 → 64.118 s. This is a distribution change, not evidence
of uniformly slower prefill. Acceptance length changed +1.8%, −3.3%,
−1.8% and −2.5%, so output gains combine lower iteration latency with
different speculative acceptance and generation trajectories. These are
single paired runs, not repeated confidence estimates.

Input throughput is total input tokens / full run time, including decode
and queueing. Per-request prefill is mean input / mean TTFT; the wave proxy
is latency-derived, not isolated GPU prefill throughput. Per-request decode
is 1000 / mean TPOT; aggregate output is output tokens / full run time.
ITL measures streamed chunk intervals and differs from token-normalized TPOT.

[Full metrics](headopt_sameboot_pair_rollup.csv),
[all 160 individual requests](headopt_sameboot_pair_per_request_detailed.csv),
[deltas](headopt_sameboot_pair_deltas.csv),
[independent audit](headopt_sameboot_pair_audit.json) and
[raw artifact hashes](headopt_sameboot_pair_raw_sha256.csv).

Both arms passed greedy, PPL (4.85 old / 4.80 new), prefix, normal-EOS
(0 failures/48 per arm), after-load and all eight reset gates, with no
recorded GPU faults. Saved-token comparisons found no matching-prefix
premature stops or first divergences outside reference top five;
[old/new](headopt_candidate_vs_sameboot_old_eos.json) and
[target-only](headopt_candidate_vs_targetonly_eos.json) audits retain
near-tie/soft divergences and later stopping differences. This supports
the sampled checks, not bit identity.

KV capacity stayed fixed at 281,804 tokens (3.75 GiB/GPU), with no logged
preemptions/recompute. Candidate graph capture used 0.28–0.35 GiB/rank
versus 0.55–0.62 GiB/rank, saving about **0.27 GiB per GPU**.
Candidate before/after source, profile and boot hashes matched. The old
driver did not write its final hash manifest after a teardown check raced
a lingering child; its verified before-manifest and subsequent safe
process cleanup are documented in the audit.

The fresh same-boot native MTP2 control completed all 80 requests and all
four reset/quality/fault gates with unchanged source, profile and boot
manifests. Its 16k/1k c=1 cell reports 57.79 aggregate output tok/s and 102.93 decode
tok/s/request, versus 50.31 and 82.18 for the candidate. Acceptance length
is 2.516 versus 2.237; full-run verification rates are 22.964 versus
22.490 steps/s. Acceptance therefore explains most of this throughput gap,
although these full-run rates also include prefill and queueing. This is
not a pure verification-kernel timing comparison.

At 16k/1k c=8, native MTP2 reports 92.47 aggregate output tok/s and 15.31
decode tok/s/request, versus 88.70 and 14.13 for DFlash2. DFlash2's
aggregate output is **12.95% lower at c=1 and 4.08% lower at c=8**.
At c=8 it has a 5.55% higher full-run verification rate but a 9.15% lower
acceptance length, so the acceptance deficit reverses the throughput
advantage. Its c=8 mean TTFT is lower (17.85 versus 19.36 s).
The inherent KV capacities differ: 281,804 DFlash2 versus 266,945 native
tokens at the same fixed memory allocation. Neither run logged preemption.
See [all eight cells](native_control/normalized_rollup.csv),
[all 160 requests](native_control/per_request.csv) and the
[independent native-control audit](native_control/audit.json).

Production phase and proposal-quality diagnostics at 16k/1k c=1 and c=8
are complete; results appear below. Short depth-three qualification is
deferred while the long-context objective is investigated.

## Remaining costs after GEMM tuning

Fresh 16k/1k production-shape diagnostics after the head/FA changes measured
outer draft replay at 5.57 ms (c=1) and 7.60 ms (c=8), versus target forward
at 19.33 and 47.00 ms. A separate run with retained external event nodes
inside the graph places **77–79% of draft replay in the five-layer body**,
15–16% in candidate generation and about 0.19 ms in the selector. HIP
capture/replay support was checked independently before enabling these
events. Instrumented runs are not serving throughput comparisons. See
[long-context phase records and collection rules](long_phase/NOTES.md).

An isolated BF16 GEMV prototype retains BF16 operands/output and FP32
accumulation, avoiding the draft residual overflow of a full FP16 cast.
All 24 tested M=3/4/24 shape cases and the full-range check passed. Cached
kernel latency ratios against lookup-only rocBLAS were 0.076–0.207 at M=3,
0.089–0.273 at M=4 and 0.355–0.891 at M=24. Row tiling additionally helped
M=24 N=640 (tile eight) and N=256 (tile four). The guarded integration was
built into `_rocm_C.abi3.so` c390; BF16/Inductor graph-replay and targeted
postbuild suites passed. Same-boot serving and native-MTP2 results are recorded
in the final section below. They do not establish a DFlash2 win beyond run
variance, so the microbenchmarks remain evidence for kernel potential only.

Before the head/FA changes, the FULL graph profile measured draft replay at 9.923 ms for c=1
and 13.791 ms for c=8. Fine eager diagnostics ranked the head at
3.550/3.568 ms and attention at about 1.66/4.56 ms across five layers.
Eager totals are not production throughput. See
[profiling_baseline.csv](profiling_baseline.csv) and its
[collection notes](profiling_baseline.notes.md); graph replay hides inner
Python marks, so warmup marks are excluded from steady graph measurements.

## Proposal-quality diagnosis

Bounded eager diagnostics paired 128 proposal records with the next target
verification in each cell. All recorded positions matched. In the c=1
sample, the target's first greedy token appeared in the candidate top 16 in
124/128 records, raw head top one in 91/128 and the selected draft in 94/128.
The c=8 counts were 109/128, 73/128 and 73/128. The c=1 records are from one
sequence; the c=8 cap covers only four early request slots, so neither is an
independent or balanced accuracy estimate. Eager arithmetic and concurrent
target trajectories also differ from production graph execution.

An exploratory selector-weight sweep found no convincing improvement;
the default remains unchanged. These records separate candidate recall
from ranking losses, rather than establishing a new calibration. See
[proposal-quality evidence](proposal_quality/).

## Shared-head dtype mismatch

The FP16 target's dense head is shared with DFlash2. DFlash2's BF16 model
configuration nevertheless initialized its candidate `LogitsProcessor` to
BF16. `compute_candidates` first casts hidden states to the shared FP16
weight dtype, then `_apply_head` casts both operands back to BF16. This
materializes about 318 MB of converted head weights per GPU on every draft
step before the BF16 GEMM.

The selected RDNA2 path uses the existing shared FP16 head directly; it does
not cast or modify target weights. Draft residuals and other draft weights
stay BF16. Explicit draft head-dtype overrides remain respected.

Actual-weight probes on all four ranks used seeded synthetic normalized
inputs at M=2/3/4/6/8/12/16/24, five warmups, 30 GPU-event repetitions and
lookup-only TunableOp. Mean rank medians (current recast + BF16 GEMM → direct
FP16 dispatch) were 3.512 → 0.757 ms at M=2, 3.729 → 0.773 at M=8,
3.721 → 1.116 at M=16 and 3.915 → 1.298 at M=24. M≤8 uses the existing
RDNA2 FP16 GEMV; larger shapes use rocBLAS. These isolated reductions do not
establish serving gains. Synthetic local top-one IDs match in 292/300 rows;
this changes draft arithmetic and is not candidate-exact.

[head_dtype_probe.csv](head_dtype_probe.csv), the
[derivation notes](head_dtype_probe.notes.md) and four `head-rank*.json`
files retain range, numerical, timing and source details. All weights were
finite with zero FP16 overflow. A conservative RMSNorm/Cauchy range estimate
with a pragmatic 2% rounding allowance is ≤8,205.3 for the largest shard,
below FP16 max 65,504; this assumes finite normalization inputs and does not
replace runtime correctness gates.

## Attention candidate

The BF16 noncausal GQA path has only two active CTAs per request for this
checkpoint's short query block. The candidate adds aligned vector K/V loads
and optional split-K over the clipped sliding-window tile range, with FP32
O/max/sum partials merged by the existing combine algorithm. One split
retains the unsplit path.

The same-boot graph sweep measured all 12 c=1/c=8, q=3/q=4,
KV=1024/2048/16384 combinations with a physical five-layer block-major
cache. Vector-only S1 improved every cell by 1.79–4.14× versus the old
scalar kernel. S16 improved every cell by 10.35–18.29× versus old scalar
and 3.33–9.92× versus vector S1. S32 had small mixed wins/losses, so the
production policy selects S16 for BF16 noncausal D128 Hq8/Hkv2,
page1024/window2048, max query length 1–4, 1–8 sequences and max KV≥1024.
Other geometries and larger prefills retain S1. Selection uses only CPU
metadata and requires no device synchronization. See
[all split timings](attention_micro_sameboot.csv) and the
[raw graph timing logs](attention_micro/attention-old-sameboot.log).
These are isolated kernel gains; the controlled serving results above
measure the combined head and attention change.

The split reducer reads the live-query limit from `cu_query_lens` on device
and zeros trailing graph padding before reading partials. This matters
because FULL graph Q capacity can exceed the last cumulative query offset.
The regression test captures four rows and replays with one, two, then one
live row, checking both live output and zero padding. Workspace follows the
existing serialized worker-stream contract; distinct concurrent streams
must not share it.

## Fresh controlled baseline and qualification status

The old-head/old-kernel baseline completed all 80 standard benchmark
requests with detailed streaming records. Aggregate output was
68.73/212.29 tok/s at 1k/512 c=1/c=8 and 43.70/85.61 at 16k/1k.
All four prefix resets, greedy, PPL (4.78), prefix, normal-EOS
(0 failures/48) and after-load gates passed, with no recorded GPU faults.
Source, binary and all four 1860-row profile hashes stayed unchanged.
See [full baseline metrics](headopt_baseline_rollup.csv),
[80 individual request records](headopt_baseline_per_request_detailed.csv)
and [audit/provenance](headopt_baseline_audit.json).

The saved normal-EOS comparison against target-only inference found no
same-prefix premature stops or first divergences outside the target's
top five. Six different stop tuples followed earlier near-tie/soft token
divergences; see [the token/stopping audit](headopt_baseline_vs_targetonly_eos.json).

Candidate compilation was interrupted at 72/99 targets by an unplanned
reboot. On recovery, the old installed binary was unchanged and no build
process survived. The build resumed using its cache and frozen dependencies
on boot `9064ffa5-1d25-4059-a521-5e5d69d03a36`, completing in 355 seconds.
The new binary SHA256 is
`53a8cb48a69e0cd115e53bc56c155d380b5919fdbd38a1099c4cd410c203855a`;
141 `_rocm_C` ops (126 RDNA) registered, the GQA schema has
`int kv_splits=1`, and `ldd` found no missing dependencies.

The targeted GPU suites passed: writer/layout 99, DFlash2 19 (10 CUDA-only
skips), phase-profile 9 and lilicorr 30: 157 passed, 10 skipped, no failures.
This includes vector/scalar parity, split/window/reference parity, block-major
cache strides, graph replay with changing live queries, and head guard cases.

The 12 new dispatch-boundary regressions also passed, for 169 targeted tests
passed and 10 CUDA-only skips overall. Logs and source/binary hashes are in
[validation/](validation/test-gqa-split-dispatch.log).

The old runtime baseline repeated on the new boot: 80/80 requests and all
serving gates passed, PPL 4.85, normal-EOS 0 failures/48, four cold resets
and no recorded faults. Aggregate output was 66.19/214.45/45.20/85.42 tok/s
in the standard four cells. See [metrics](headopt_sameboot_old_rollup.csv),
[individual requests](headopt_sameboot_old_per_request_detailed.csv) and
[audit](headopt_sameboot_old_audit.json).

The driver exited nonzero after all serving gates because its immediate
process-group cleanup check saw a lingering child. No PID/group members or
listening port remained when manually checked before restoring the candidate.
The original final live-runtime hash manifest was not written; the audit
records this limitation rather than claiming an after-manifest match.

The combined candidate serving pair completed; current results are at the
top of this file. The earlier pre-reboot baseline is historical only.

## Post-merge BF16-core follow-up and native MTP2 surface audit

The BF16 GEMV core was built in the frozen ROCm 7.14 venv as binary
`c39006c814d32c95aac0388f12d0d26b7c6dc4201f071586a9ef4c8a69df8bd4`.
The targeted postbuild checks passed: 105 BF16/Inductor graph tests, 161 FA,
DFlash2 and profile tests, and 30 LiLiCorr tests. Actual Inductor graph capture
and changed-input replay passed for M=3/4/24. The binary, registration and
Wave32 evidence are in [core_bf16/](core_bf16/).

On the same boot, DFlash2 c390 produced 52.25 aggregate output tok/s at
16k/1k c=1 (8/8 requests; acceptance length 2.057) and 90.55 at c=8 (32/32;
acceptance 2.186). The c=1 native MTP2 runs produced 51.68 and 49.82
aggregate tok/s on two attempts; c=8 native runs produced 90.85 and 92.55.
These measurements do not show DFlash2 beating native MTP2 beyond variance
at both concurrency levels. The repeated native c=1 runs themselves varied
in acceptance and saved output text despite the same seed and token lengths;
raw prompt hashes were not retained. The c=8 DFlash2 result is tied to the
native range. Do not treat this as a qualified performance win.

The bounded native kernel trace found zero `gemv_bf16_rdna2` launches and
24,383 `gemv_f16_rdna2` launches on the captured native workers. The trace
covered TP0/TP2/TP3, not TP1, and included startup/capture plus tiny diagnostic
requests. It supports that the new BF16 GEMV was not selected on those
observed MTP2 workers; it does not prove all shared changes are performance
neutral. Native MTP2 functional gates passed (normal-EOS, greedy, PPL, prefix,
after-load, request-success and fault checks). Other shared surfaces changed
by this branch include FA-RDNA2 BF16/noncausal dispatch, appended BF16
TunableOp rows and the Qwen4 auxiliary-state hook. The tested native
FP16-causal configuration stays on its prior attention path, and the Qwen4
hook is conditional on requested DFlash2 auxiliary layers. See the
[surface audit](core_bf16/MTP2_SURFACE_AUDIT.md) for trace coverage and limits.
