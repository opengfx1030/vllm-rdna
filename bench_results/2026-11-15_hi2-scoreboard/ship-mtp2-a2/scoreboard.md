# Scoreboard: arm `ship-mtp2-a2`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RECIPE=27b-hi2-tp2-mtp2`, GPUs 0,1 (PCI d1 d4), tree sha `15a4bc6f49`
- started 2026-11-15 22:07:57, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.39 / 1.41 | 735 | 85.7 | 33.3 / 33.7 | 69.0 | 207 | 69.0 |
| 1024x512-c8 | 16 | 7.12 / 12.41 | 211 | 25.4 | 92.7 / 1581.2 | 136.9 | 411 | 8.6 |
| 16384x1024-c1 | 8 | 22.20 / 22.27 | 738 | 69.1 | 41.6 / 42.1 | 27.6 | 470 | 0.0 |
| 16384x1024-c8 | 16 | 76.48 / 210.25 | 354 | 6.2 | 195.9 / 1855.3 | 29.5 | 501 | 0.0 |
| 32768x2048-c1 | 4 | 48.49 / 48.66 | 676 | 58.8 | 50.2 / 51.5 | 24.6 | 418 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6635 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6635 | +0.007% | 0.00007 | 0.00217 | 6/6 | 96.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | +0.00001 [-0.00002, +0.00003] | 0.9986 [0.9981, 0.9989] | 0.0158 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 7.861e-06 | 6.869e-05 | 512.0 | 18/32 | 0 | 11558 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 191 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.21 s, warm TTFT 0.42 s, speedup 10.1x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d1: junction max 85 C, worst 5 s mean 82.4 C (limit 90), VRAM max 27591 MiB
- bus d4: junction max 84 C, worst 5 s mean 81.8 C (limit 90), VRAM max 27599 MiB
