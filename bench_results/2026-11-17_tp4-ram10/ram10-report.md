# RAM10 TP4 expert-offload study

Generated `2026-11-18T00:17:43.314772+00:00` from `vllm-ram10-study/bench_results/2026-11-17_tp4-ram10/matrix-r1`; baseline arm `R`.

## Capacity and startup

| Arm/rep | Run | Slots | W4A8 pre/sd | Compile / graph | KV ranks | KV TP-min GiB (B) | KV sum GiB | KV tokens | PPL | Startup c/a/g GiB | Graph pool a/e GiB | Expert dev/host GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R/1 | resident-r1 | 0 | 1/1 | 3 / FULL_AND_PIECEWISE | 4 | 4.1680 (4475387904) | 16.672 | 279318 | 4.79 | 21.30-21.55 / 1.85 / 0.32 | 0.32 / 0.62 | n/a |
| F/1 | fullcache-r1 | 128 | 0/0 | 0 / FULL_AND_PIECEWISE | 4 | 2.4819 (2664955904) | 9.928 | 166325 | 4.80 | 21.21-21.62 / 3.48 / 1.57 | 1.57 / 2.25 | 58.45 / 58.45 |
| D/1 | demand-r1 | 115 | 0/0 | 0 / FULL_AND_PIECEWISE | 4 | 3.6015 (3867082752) | 14.406 | 241353 | 4.80 | 19.92-20.31 / 3.66 / 1.57 | 1.57 / 2.44 | 52.97 / 58.45 |
| Z/1 | zeros-r1 | 115 | 0/0 | 0 / FULL_AND_PIECEWISE | 4 | 3.7094 (3982950400) | 14.838 | 248584 | 4.80 | 19.91-20.21 / 3.66 / 1.57 | 1.57 / 2.44 | 52.97 / 58.45 |

## Cells

| Cell | Arm/rep | ok/req | Prompt | TTFT p50/p99 ms | Prefill req/agg/span tok/s | Decode req/wtd/span tok/s | Out e2e tok/s | ITL p50/p99 ms | MTP acc | Cache miss% (m/d) / copy | Preempt | Fault | Evict | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench-1024x512-c1 | R/1 | 4/4 | 1024 | 410.6 / 419.5 | 2484.4 / 2441.1 / 200.4 | 85.77 / 85.77 / 81.52 | 80.37 | 23.4 / 23.9 | 50.4% (1027/2038) | n/a | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c8 | R/1 | 8/8 | 1024 | 2178.6 / 3246.9 | 488.8 / 2523.0 / 2518.6 | 40.65 / 40.65 / 252.91 | 247.02 | 45.9 / 377.6 | 48.2% (2008/4168) | n/a | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c1 | R/1 | 2/2 | 16384 | 6275.5 / 6304.8 | 2610.8 / 2598.7 / 1102.7 | 77.32 / 77.32 / 62.44 | 52.49 | 23.8 / 24.5 | 42.1% (936/2224) | n/a | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c8 | R/1 | 8/8 | 16384 | 30771.9 / 54740.8 | 529.4 / 2394.4 / 2373.5 | 23.90 / 23.90 / 113.32 | 102.91 | 49.7 / 451.7 | 63.7% (4588/7202) | n/a | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c1 | R/1 | 2/2 | 32768 | 12915.5 / 12970.9 | 2537.1 / 2526.3 / 1336.2 | 96.84 / 96.84 / 74.10 | 60.14 | 23.9 / 24.6 | 65.7% (2326/3540) | n/a | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c8 | R/1 | 8/8 | 32768 | 63283.0 / 141530.9 | 464.0 / 1852.2 / 1839.1 | 27.99 / 27.99 / 108.22 | 99.11 | 46.2 / 466.4 | 82.7% (10210/12342) | n/a | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c1 | F/1 | 4/4 | 1024 | 631.3 / 642.7 | 1624.3 / 1593.2 / 113.0 | 49.39 / 49.39 / 47.22 | 46.64 | 37.1 / 39.0 | 41.8% (931/2226) | 0.0% (25/1189575) / 60.88 MiB | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c8 | F/1 | 8/8 | 1024 | 3367.5 / 5009.8 | 315.9 / 1635.2 / 1633.0 | 28.77 / 28.77 / 161.13 | 157.52 | 71.2 / 590.6 | 56.8% (2176/3832) | 0.0% (47/1399798) / 114.46 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c1 | F/1 | 2/2 | 16384 | 9225.1 / 9267.6 | 1776.0 / 1767.9 / 1045.4 | 62.94 / 62.94 / 49.07 | 40.19 | 36.6 / 38.8 | 65.5% (1162/1774) | 0.0% (3/1128244) / 7.31 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c8 | F/1 | 8/8 | 16384 | 44400.7 / 120340.3 | 278.2 / 1089.2 / 1083.1 | 16.39 / 16.39 / 61.98 | 57.27 | 65.0 / 648.6 | 72.2% (4839/6702) | 0.0% (2/3817486) / 4.87 MiB | 11 | 0 | ok | yes (warn) |
| bench-32768x2048-c1 | F/1 | 2/2 | 32768 | 18776.2 / 18812.0 | 1745.2 / 1741.9 / 878.3 | 64.95 / 64.95 / 50.07 | 40.72 | 36.6 / 39.3 | 69.5% (2381/3428) | 0.0% (6/2168901) / 14.61 MiB | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c8 | F/1 | 8/8 | 32768 | 146674.5 / 276532.9 | 232.2 / 948.0 / 942.7 | 23.21 / 23.21 / 56.30 | 52.63 | 59.7 / 639.9 | 76.2% (9893/12984) | 0.0% (6/8625802) / 14.61 MiB | 4 | 0 | ok | yes (warn) |
| bench-1024x512-c1 | D/1 | 4/4 | 1024 | 651.0 / 671.3 | 1569.9 / 1525.3 / 117.8 | 51.67 / 51.67 / 49.21 | 48.56 | 35.4 / 38.9 | 41.8% (931/2226) | 0.3% (3360/1189575) / 8182.62 MiB | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c8 | D/1 | 8/8 | 1024 | 3381.7 / 4969.3 | 317.4 / 1648.5 / 1646.1 | 24.45 / 24.45 / 137.43 | 134.84 | 70.6 / 108.3 | 39.1% (1796/4592) | 0.2% (3059/1440595) / 7449.59 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c1 | D/1 | 2/2 | 16384 | 9288.6 / 9295.3 | 1763.9 / 1762.6 / 989.1 | 61.13 / 61.13 / 47.86 | 39.35 | 34.8 / 37.8 | 56.6% (1087/1920) | 0.2% (2687/1273963) / 6543.66 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c8 | D/1 | 8/8 | 16384 | 44628.4 / 79013.5 | 365.0 / 1658.9 / 1644.7 | 15.85 / 15.85 / 73.25 | 66.71 | 69.7 / 648.1 | 51.8% (4167/8046) | 0.3% (8892/3362538) / 21654.71 MiB | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c1 | D/1 | 2/2 | 32768 | 18864.1 / 18886.4 | 1737.1 / 1735.0 / 857.7 | 59.57 / 59.57 / 46.73 | 38.48 | 34.7 / 38.8 | 54.2% (2130/3928) | 0.3% (6683/2395947) / 16275.13 MiB | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c8 | D/1 | 8/8 | 32768 | 89679.5 / 206250.0 | 304.5 / 1271.0 / 1262.4 | 18.82 / 18.82 / 68.36 | 63.02 | 67.2 / 647.4 | 59.2% (8882/14996) | 0.4% (30587/7730714) / 74488.60 MiB | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c1 | Z/1 | 4/4 | 1024 | 651.6 / 669.1 | 1573.3 / 1530.3 / 119.2 | 52.13 / 52.13 / 49.63 | 48.98 | 34.8 / 38.8 | 41.8% (931/2226) | 0.3% (3329/1189575) / 8046.17 MiB | 0 | 0 | ok | yes (warn) |
| bench-1024x512-c8 | Z/1 | 8/8 | 1024 | 3435.3 / 5054.0 | 310.7 / 1620.9 / 1618.4 | 26.55 / 26.55 / 152.75 | 149.50 | 71.3 / 590.3 | 49.1% (2026/4128) | 0.2% (3356/1375459) / 8111.43 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c1 | Z/1 | 2/2 | 16384 | 9331.0 / 9367.7 | 1755.9 / 1749.0 / 902.9 | 55.74 / 55.74 / 44.48 | 36.99 | 34.9 / 38.6 | 47.9% (1002/2092) | 0.2% (2846/1252311) / 6878.76 MiB | 0 | 0 | ok | yes (warn) |
| bench-16384x1024-c8 | Z/1 | 8/8 | 16384 | 44697.7 / 79228.9 | 364.1 / 1654.3 / 1640.1 | 16.18 / 16.18 / 73.57 | 66.95 | 73.3 / 658.3 | 53.3% (4228/7930) | 0.3% (12119/3660745) / 29291.53 MiB | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c1 | Z/1 | 2/2 | 32768 | 18857.2 / 18877.0 | 1737.7 / 1735.9 / 916.9 | 68.45 / 68.45 / 52.03 | 42.00 | 35.9 / 39.6 | 72.7% (2426/3338) | 0.3% (5506/2133232) / 13307.96 MiB | 0 | 0 | ok | yes (warn) |
| bench-32768x2048-c8 | Z/1 | 8/8 | 32768 | 89706.3 / 206471.9 | 305.0 / 1269.6 / 1261.1 | 19.82 / 19.82 / 60.88 | 56.60 | 61.0 / 643.3 | 50.0% (8187/16384) | 0.3% (27844/8257665) / 67298.73 MiB | 0 | 0 | ok | yes (warn) |

## Baseline-relative deltas

| Cell | Arm/rep | Baseline | Δ prefill req/agg/span % | Δ TTFT p50/p99 % | Δ decode req/wtd/span % | Δ ITL p50/p99 % | Δ out e2e % | Δ KV TP-min/tokens % | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench-1024x512-c1 | F/1 | resident-r1 r1 | -34.6% / -34.7% / -43.6% | +53.7% / +53.2% | -42.4% / -42.4% / -42.1% | +58.5% / +63.2% | -42.0% | -40.5% / -40.5% | ok |
| bench-1024x512-c8 | F/1 | resident-r1 r1 | -35.4% / -35.2% / -35.2% | +54.6% / +54.3% | -29.2% / -29.2% / -36.3% | +55.2% / +56.4% | -36.2% | -40.5% / -40.5% | ok |
| bench-16384x1024-c1 | F/1 | resident-r1 r1 | -32.0% / -32.0% / -5.2% | +47.0% / +47.0% | -18.6% / -18.6% / -21.4% | +54.0% / +58.7% | -23.4% | -40.5% / -40.5% | ok |
| bench-16384x1024-c8 | F/1 | resident-r1 r1 | -47.5% / -54.5% / -54.4% | +44.3% / +119.8% | -31.4% / -31.4% / -45.3% | +31.0% / +43.6% | -44.3% | -40.5% / -40.5% | ok |
| bench-32768x2048-c1 | F/1 | resident-r1 r1 | -31.2% / -31.1% / -34.3% | +45.4% / +45.0% | -32.9% / -32.9% / -32.4% | +52.8% / +59.6% | -32.3% | -40.5% / -40.5% | ok |
| bench-32768x2048-c8 | F/1 | resident-r1 r1 | -50.0% / -48.8% / -48.7% | +131.8% / +95.4% | -17.1% / -17.1% / -48.0% | +29.1% / +37.2% | -46.9% | -40.5% / -40.5% | ok |
| bench-1024x512-c1 | D/1 | resident-r1 r1 | -36.8% / -37.5% / -41.2% | +58.5% / +60.0% | -39.8% / -39.8% / -39.6% | +51.6% / +62.8% | -39.6% | -13.6% / -13.6% | ok |
| bench-1024x512-c8 | D/1 | resident-r1 r1 | -35.1% / -34.7% / -34.6% | +55.2% / +53.0% | -39.8% / -39.8% / -45.7% | +53.8% / -71.3% | -45.4% | -13.6% / -13.6% | ok |
| bench-16384x1024-c1 | D/1 | resident-r1 r1 | -32.4% / -32.2% / -10.3% | +48.0% / +47.4% | -20.9% / -20.9% / -23.4% | +46.5% / +54.7% | -25.0% | -13.6% / -13.6% | ok |
| bench-16384x1024-c8 | D/1 | resident-r1 r1 | -31.1% / -30.7% / -30.7% | +45.0% / +44.3% | -33.7% / -33.7% / -35.4% | +40.3% / +43.5% | -35.2% | -13.6% / -13.6% | ok |
| bench-32768x2048-c1 | D/1 | resident-r1 r1 | -31.5% / -31.3% / -35.8% | +46.1% / +45.6% | -38.5% / -38.5% / -36.9% | +45.0% / +57.4% | -36.0% | -13.6% / -13.6% | ok |
| bench-32768x2048-c8 | D/1 | resident-r1 r1 | -34.4% / -31.4% / -31.4% | +41.7% / +45.7% | -32.8% / -32.8% / -36.8% | +45.4% / +38.8% | -36.4% | -13.6% / -13.6% | ok |
| bench-1024x512-c1 | Z/1 | resident-r1 r1 | -36.7% / -37.3% / -40.5% | +58.7% / +59.5% | -39.2% / -39.2% / -39.1% | +48.9% / +62.2% | -39.1% | -11.0% / -11.0% | ok |
| bench-1024x512-c8 | Z/1 | resident-r1 r1 | -36.4% / -35.8% / -35.7% | +57.7% / +55.7% | -34.7% / -34.7% / -39.6% | +55.4% / +56.3% | -39.5% | -11.0% / -11.0% | ok |
| bench-16384x1024-c1 | Z/1 | resident-r1 r1 | -32.7% / -32.7% / -18.1% | +48.7% / +48.6% | -27.9% / -27.9% / -28.8% | +47.0% / +57.9% | -29.5% | -11.0% / -11.0% | ok |
| bench-16384x1024-c8 | Z/1 | resident-r1 r1 | -31.2% / -30.9% / -30.9% | +45.3% / +44.7% | -32.3% / -32.3% / -35.1% | +47.6% / +45.8% | -34.9% | -11.0% / -11.0% | ok |
| bench-32768x2048-c1 | Z/1 | resident-r1 r1 | -31.5% / -31.3% / -31.4% | +46.0% / +45.5% | -29.3% / -29.3% / -29.8% | +49.8% / +60.8% | -30.2% | -11.0% / -11.0% | ok |
| bench-32768x2048-c8 | Z/1 | resident-r1 r1 | -34.3% / -31.5% / -31.4% | +41.8% / +45.9% | -29.2% / -29.2% / -43.7% | +32.0% / +37.9% | -42.9% | -11.0% / -11.0% | ok |

## Health and missing data

- R/1/bench-1024x512-c1: warnings: itl_partial
- R/1/bench-1024x512-c8: warnings: itl_partial
- R/1/bench-16384x1024-c1: warnings: itl_partial
- R/1/bench-16384x1024-c8: warnings: itl_partial
- R/1/bench-32768x2048-c1: warnings: itl_partial
- R/1/bench-32768x2048-c8: warnings: itl_partial
- F/1/bench-1024x512-c1: warnings: itl_partial
- F/1/bench-1024x512-c8: warnings: itl_partial
- F/1/bench-16384x1024-c1: warnings: itl_partial
- F/1/bench-16384x1024-c8: warnings: preemptions_observed, itl_partial
- F/1/bench-32768x2048-c1: warnings: itl_partial
- F/1/bench-32768x2048-c8: warnings: preemptions_observed, itl_partial
- D/1/bench-1024x512-c1: warnings: itl_partial
- D/1/bench-1024x512-c8: warnings: itl_partial
- D/1/bench-16384x1024-c1: warnings: itl_partial
- D/1/bench-16384x1024-c8: warnings: itl_partial
- D/1/bench-32768x2048-c1: warnings: itl_partial
- D/1/bench-32768x2048-c8: warnings: itl_partial
- Z/1/bench-1024x512-c1: warnings: itl_partial
- Z/1/bench-1024x512-c8: warnings: itl_partial
- Z/1/bench-16384x1024-c1: warnings: itl_partial
- Z/1/bench-16384x1024-c8: warnings: itl_partial
- Z/1/bench-32768x2048-c1: warnings: itl_partial
- Z/1/bench-32768x2048-c8: warnings: itl_partial

## Notes

- D/Z have 115 real slots plus one dummy: 116 physical rows, saving 12, not 13. F is the full-capacity cache control. The CPU bank mirrors all 128 local experts, not only the 13 nonresident ones.
- R uses W4A8 prefill and compiled F&P; F/D/Z use W4A16 prefill and breakable F&P, compile NONE. R-to-offload measures deployment cost, not PCIe alone. No resident W4A16 control was measured.
- KV bytes are exact deduplicated rdna_cache_memory storage: TP-min limits capacity; sum covers all ranks. Token capacity is the engine's GPU KV cache size. Missing RPC data stays null, never inferred from weight savings. Expert host/device bytes are separate.
- Logged peak activation includes estimated graph reserve: it is not isolated transient memory. Graph-pool actual/estimated values are reported separately; startup memory fields must not be summed blindly.
- Output e2e is total output/duration. Decode wtd is sum(out-1)/sum(latency-TTFT); decode span is sum(out-1)/(last completion-first first token), not pure GPU decode. Prefill req is mean prompt/mean TTFT; agg is bench_metrics' one-wave estimate. Span rates include queueing and overlapping other phases.
- CSV rates use each request's prompt/TTFT and (out-1)/(latency-TTFT), never 1/median ITL. ITL measures delivery chunks under MTP; missing token intervals are not fabricated as zeros.
- Preemptions are cumulative metric deltas, excluding _created. Positive counts flag capacity limits; faults/KFD evictions invalidate. Baseline comparisons match cell, replicate, request and completed counts. Missing observations stay null/n/a/blank.
