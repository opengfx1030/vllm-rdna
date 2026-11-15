# Scoreboard: arm `residual`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1`, GPUs 2,3 (PCI d9 dc), tree sha `dd1d104148`
- started 2026-11-15 07:15:12, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.51 / 1.52 | 680 | 39.3 | 25.5 / 25.8 | 35.3 | 106 | 35.3 |
| 1024x512-c8 | 16 | 7.23 / 12.30 | 199 | 14.3 | 60.4 / 61.7 | 94.4 | 283 | 0.0 |
| 16384x1024-c1 | 8 | 30.50 / 30.71 | 537 | 31.3 | 32.0 / 32.3 | 16.2 | 276 | 0.0 |
| 16384x1024-c8 | 16 | 97.06 / 259.41 | 264 | 4.7 | 88.2 / 2455.9 | 23.3 | 396 | 0.0 |
| 32768x2048-c1 | 4 | 75.71 / 76.04 | 433 | 25.9 | 38.6 / 39.1 | 13.2 | 225 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 5/6 | 80.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | within floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL within floor | dNLL within floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 1.977e-06 | 3.041e-05 | 512.0 | 28/32 | 0 | 15562 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | within floor | | | | | |

## 4. Validity

- READY after 180 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.73 s, warm TTFT 0.41 s, speedup 11.4x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 0 near-tie, 0 soft, 0 solo-noise, 48 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 84 C, worst 5 s mean 82.4 C (limit 90), VRAM max 27152 MiB
- bus dc: junction max 81 C, worst 5 s mean 80.0 C (limit 90), VRAM max 27160 MiB
