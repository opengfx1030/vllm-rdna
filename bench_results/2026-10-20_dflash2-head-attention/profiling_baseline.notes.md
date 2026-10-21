# Tuned DFlash2 phase baseline

Source: TP0 `serve.log` from `serve-phase-tuned1860-dflash2-spec2` (FULL
graph) and `serve-phase-tuned1860-dflash2-spec2-eager` (eager), both run on
2026-10-20 with the 1860-row profile, two draft tokens, HIP 6–9 and the same
recipe. The benchmark windows come from each run's `summary.txt`.
Persistent remote originals are
`/home/chenco_adm/w4a8_runs/port-v031/serve-phase-tuned1860-dflash2-spec2/serve.log`
and
`/home/chenco_adm/w4a8_runs/port-v031/serve-phase-tuned1860-dflash2-spec2-eager/serve.log`;
their `summary.txt` and benchmark JSON files are alongside the logs.

[`profiling_baseline.csv`](profiling_baseline.csv) keeps pure decode shapes only:
`tokens=3` for c=1 and `tokens=24` for c=8. The first TP0 PhaseTimer flush
second in each cell is excluded because it can contain the preceding warmup
or cell. Remaining sample sizes are 200 (FULL c=1), 174 (FULL c=8), 150
(eager c=1), and 176 (eager c=8) iterations across 4, 4, 3, and 4 flushes.
`weighted_mean_ms = sum(n * reported_phase_ms) / sum(n)`; `flush_median_ms`
is the median of the remaining flush means, not a per-iteration median.
Phases may overlap or omit work, so their values must not be summed into an
exact wall time.

FULL graph replay exposes `draft` as a single GPU event: 9.923 ms c=1 and
13.791 ms c=8, with target `forward` at 18.895 and 45.203 ms. Eager runs
expose internal draft phases but have materially different `forward` and
`total` timings; use them for component ranking, not serving latency. Their
`dflash_head_gemm` is 3.550 ms c=1 and 3.568 ms c=8, including the current
FP16-to-BF16 shared-head recast. BF16 input/post-attention RMSNorm marks are
about 0.08–0.09 ms each. The eager terminal `draft` mark (~0.02 ms) is **not**
the whole draft phase and should not be compared with graph replay `draft`.
