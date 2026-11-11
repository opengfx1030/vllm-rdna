# Combined O2 core serving A/B

This bundle records one same-boot serving pair comparing the old inline K2
core (`01-old`) with the combined quantized-router and dense-rounding O2 core
(`02-new`) under ordinary native MTP2. The target used MonoKernel with
`k2_max_rows=8`; `mtp_drafter=false` and `dense_calls=0` in both arms. This
therefore checks whether the target-side core changes affect the regular native
MTP2 route. It does not measure dense MTP drafting, and one pair cannot
estimate run variance or statistical confidence.

Both arms ran on boot `1e8b5da1-759f-4dbb-ae87-e58c2faaff48`, with the same
model/PLE snapshots, TP4/EP4 on HIP6–9, frozen venv, recipe, tuned rocBLAS rows,
KV and memory settings, temperature 0, seed 12345, prefix reset before each
cell, and the four standard request cells. Strict provenance validation
passed: all four worker core mappings and GPU owners matched, and all
non-core source/config hashes matched. Per-arm hashes and filtered gate logs
are under `arms/`; full raw logs and generated response bodies are omitted.

## Serving metrics

The table reports old → new. Aggregate whole-run input/output rates are total
timed-cell tokens divided by the measured timed duration. Per-request prefill
and decode rates are arithmetic means of request rates. TTFT and ITL are shown
as median / mean milliseconds. Acceptance is accepted-draft rate and mean
accepted length from the exact timed-cell window.

| Cell | Whole-run input / output tok/s | Prefill tok/s per request | Decode tok/s per request | Median TTFT ms | Median / mean ITL ms | Acceptance % / length |
| --- | --- | --- | --- | --- | --- | --- |
| 1k/512 c1 | 148.37→172.43 / 74.18→86.22 (+16.22%) | 2302.10→2274.60 (-1.19%) | 86.40→96.45 (+11.63%) | 442.10→452.89 (+2.44%) | 24.30/24.36→24.23/24.31 | 46.58/1.932→63.29/2.266 |
| 1k/512 c8 | 448.10→481.16 / 224.05→240.58 (+7.38%) | 1460.36→1444.96 (-1.05%) | 34.01→35.26 (+3.69%) | 623.23→623.41 (+0.03%) | 53.56/63.02→54.04/64.48 | 48.38/1.968→56.12/2.122 |
| 16k/1k c1 | 932.44→834.82 / 58.28→52.18 (-10.47%) | 2190.29→2186.26 (-0.18%) | 101.89→91.48 (-10.21%) | 7484.46→7480.78 (-0.05%) | 24.83/24.85→24.78/24.79 | 76.09/2.522→54.58/2.092 |
| 16k/1k c8 | 1473.20→1495.25 / 92.08→93.45 (+1.50%) | 1353.63→1237.45 (-8.58%) | 17.64→17.40 (-1.36%) | 9545.84→12911.40 (+35.26%) | 55.49/143.79→56.91/151.77 | 55.19/2.104→66.45/2.329 |

Across all four timed cells, whole-run input and output throughput each rose
0.247%. The long c1 cell regressed in whole-run throughput, per-request decode
rate, and draft acceptance. The long c8 cell had a small whole-run throughput
gain and better acceptance, alongside worse prefill rate, TTFT, and mean ITL.
These results are mixed and do not establish an overall serving win. The full
per-request and per-cell data are in `analysis/`.

The TTFT-derived prefill wave proxy is not true throughput. ITL is the gap
between streamed token chunks, not TPOT. Detailed definitions and the raw
per-request values are retained in the analysis CSVs.

## Correctness and health gates

Both arms passed startup probes (3/3), PPL probes (old 4.84, new 4.90), prefix
checks (8.9× and 8.8× warm speedup), the concurrency gate (zero failures in
both), post-load probes, all four timed cells, and fault checks. The old
concurrency run recorded 19 near-ties and 29/48 identical checks; the new run
recorded 16 near-ties, one soft case, and 31/48 identical checks. Cross-arm
fixed-prompt greedy/log-probability comparison matched 568/772 tokens on the
shared prefix, with maximum absolute log-probability difference 0.3548. The
response text is not identical throughout; the parity file contains token and
log-probability summaries without generated response bodies.

An independent same-boot kernel-journal audit read 831 lines and found zero
matches for the fault patterns. The bounded audit summary is
`kernel-window-audit.json`; the raw journal is omitted. This result is one
paired serving observation with the recorded gates, not a long-run reliability
claim.

## Reproducibility and scope

`analysis/provenance-validation.json` is the strict pair validation result.
`analysis/` contains per-request and aggregate metrics, acceptance windows, and
cross-arm parity. `source-input-hashes.json` pins omitted raw inputs, including
full server/driver logs and detailed request files; these may contain generated
text. Exact analyzer and validator source snapshots are under
`tools/source-snapshots/`. `evidence-manifest.json` hashes every retained file.

The cores compared were `25f3fc556ad5…` / CU `aec873933fde…` and
`3f688006eec3…` / CU `bc8e35dd429f…`. There was one run per arm and no repeated
ABBA sequence, so the small whole-run change cannot be separated from run
variation. No dense MTP correctness or performance claim is made here.
