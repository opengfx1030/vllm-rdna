# Scoreboard: arm `gemv`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto`, GPUs 2,3 (PCI d9 dc), tree sha `f7edb8b1e7`
- started 2026-11-15 04:25:59, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.51 / 1.53 | 677 | 36.5 | 27.4 / 27.7 | 33.0 | 99 | 33.0 |
| 1024x512-c8 | 16 | 7.23 / 12.33 | 200 | 18.8 | 43.4 / 45.1 | 117.8 | 353 | 0.0 |
| 16384x1024-c1 | 8 | 30.55 / 30.72 | 536 | 29.6 | 33.8 / 34.1 | 15.7 | 267 | 0.0 |
| 16384x1024-c8 | 16 | 97.16 / 259.70 | 264 | 5.2 | 71.6 / 2458.3 | 24.3 | 414 | 0.0 |
| 32768x2048-c1 | 4 | 75.83 / 76.23 | 432 | 24.7 | 40.5 / 40.9 | 12.9 | 219 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | within floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL within floor | dNLL within floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 6.180e-06 | 5.657e-05 | 512.0 | 19/32 | 0 | 12871 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 180 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.74 s, warm TTFT 0.42 s, speedup 11.4x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 3 near-tie, 0 soft, 0 solo-noise, 45 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 83 C, worst 5 s mean 81.8 C (limit 90), VRAM max 27145 MiB
- bus dc: junction max 81 C, worst 5 s mean 79.2 C (limit 90), VRAM max 27153 MiB
