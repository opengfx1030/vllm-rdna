# Scoreboard: arm `base2`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides ``, GPUs 2,3 (PCI d9 dc), tree sha `ef0b99ea9b+scoreboard`
- started 2026-11-15 02:32:15, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.52 / 1.53 | 676 | 36.5 | 27.4 / 29.2 | 33.0 | 99 | 33.0 |
| 1024x512-c8 | 16 | 7.26 / 12.37 | 198 | 13.9 | 62.0 / 66.0 | 92.5 | 277 | 0.0 |
| 16384x1024-c1 | 8 | 30.64 / 30.79 | 535 | 29.6 | 33.8 / 34.9 | 15.7 | 267 | 0.0 |
| 16384x1024-c8 | 16 | 97.28 / 259.98 | 263 | 4.6 | 90.0 / 2459.0 | 23.1 | 393 | 0.0 |
| 32768x2048-c1 | 4 | 75.83 / 76.18 | 432 | 24.7 | 40.5 / 40.9 | 12.9 | 220 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `None`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |

## 4. Validity

- READY after 180 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.76 s, warm TTFT 0.41 s, speedup 11.6x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 0 near-tie, 0 soft, 0 solo-noise, 48 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 84 C, worst 5 s mean 82.8 C (limit 90), VRAM max 27143 MiB
- bus dc: junction max 81 C, worst 5 s mean 79.8 C (limit 90), VRAM max 27151 MiB
