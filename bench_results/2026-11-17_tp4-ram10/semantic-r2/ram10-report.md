# RAM10 TP4 expert-offload study

Generated `2026-11-18T00:22:56.569971+00:00` from `/home/chenco_adm/w4a8_runs/ram10/semantic-r2`; baseline arm `R`.

## Capacity and startup

| Arm/rep | Run | Slots | W4A8 pre/sd | Compile / graph | KV ranks | KV TP-min GiB (B) | KV sum GiB | KV tokens | PPL | Startup c/a/g GiB | Graph pool a/e GiB | Expert dev/host GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D/2 | demand-r2 | 115 | 0/0 | 0 / FULL_AND_PIECEWISE | 4 | 3.6689 (3939500032) | 14.676 | 245872 | 4.80 | 20.04-20.25 / 3.66 / 1.57 | 1.57 / 2.44 | 52.97 / 58.45 |

## Cells

| Cell | Arm/rep | ok/req | Prompt | TTFT p50/p99 ms | Prefill req/agg/span tok/s | Decode req/wtd/span tok/s | Out e2e tok/s | ITL p50/p99 ms | MTP acc | Cache miss% (m/d) / copy | Preempt | Fault | Evict | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench-1024x512-c1 | D/2 | 1/1 | 1024 | 635.1 / 635.1 | 1612.4 / 1612.4 / 1612.4 | 32.70 / 32.70 / 32.70 | 31.48 | 35.3 / 42.9 | 8.3% (73/876) | 0.1% (284/467268) / 691.63 MiB | 0 | 0 | ok | yes (warn) |

## Baseline-relative deltas

| Cell | Arm/rep | Baseline | Δ prefill req/agg/span % | Δ TTFT p50/p99 % | Δ decode req/wtd/span % | Δ ITL p50/p99 % | Δ out e2e % | Δ KV TP-min/tokens % | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench-1024x512-c1 | D/2 | n/a | n/a / n/a / n/a | n/a / n/a | n/a / n/a / n/a | n/a / n/a | n/a | n/a / n/a | missing_baseline |

## Health and missing data

- D/2/bench-1024x512-c1: warnings: itl_partial

## Notes

- D/Z have 115 real slots plus one dummy: 116 physical rows, saving 12, not 13. F is the full-capacity cache control. The CPU bank mirrors all 128 local experts, not only the 13 nonresident ones.
- R uses W4A8 prefill and compiled F&P; F/D/Z use W4A16 prefill and breakable F&P, compile NONE. R-to-offload measures deployment cost, not PCIe alone. No resident W4A16 control was measured.
- KV bytes are exact deduplicated rdna_cache_memory storage: TP-min limits capacity; sum covers all ranks. Token capacity is the engine's GPU KV cache size. Missing RPC data stays null, never inferred from weight savings. Expert host/device bytes are separate.
- Logged peak activation includes estimated graph reserve: it is not isolated transient memory. Graph-pool actual/estimated values are reported separately; startup memory fields must not be summed blindly.
- Output e2e is total output/duration. Decode wtd is sum(out-1)/sum(latency-TTFT); decode span is sum(out-1)/(last completion-first first token), not pure GPU decode. Prefill req is mean prompt/mean TTFT; agg is bench_metrics' one-wave estimate. Span rates include queueing and overlapping other phases.
- CSV rates use each request's prompt/TTFT and (out-1)/(latency-TTFT), never 1/median ITL. ITL measures delivery chunks under MTP; missing token intervals are not fabricated as zeros.
- Preemptions are cumulative metric deltas, excluding _created. Positive counts flag capacity limits; faults/KFD evictions invalidate. Baseline comparisons match cell, replicate, request and completed counts. Missing observations stay null/n/a/blank.
