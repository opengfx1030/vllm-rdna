# Scoreboard: arm `attn`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `VLLM_FA_RDNA2_UNFUSED_PREFILL=1`, GPUs 2,3 (PCI d9 dc), tree sha `ef0b99ea9b+scoreboard`
- started 2026-11-15 03:31:09, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.50 / 1.51 | 682 | 36.5 | 27.4 / 27.8 | 33.0 | 99 | 33.0 |
| 1024x512-c8 | 16 | 7.20 / 12.25 | 200 | 13.9 | 62.0 / 63.1 | 92.8 | 278 | 0.0 |
| 16384x1024-c1 | 8 | 24.64 / 24.83 | 665 | 29.6 | 33.8 / 34.1 | 17.3 | 294 | 0.0 |
| 16384x1024-c8 | 16 | 79.33 / 212.65 | 324 | 5.1 | 90.0 / 1814.9 | 26.8 | 455 | 0.0 |
| 32768x2048-c1 | 4 | 53.22 / 53.53 | 616 | 24.7 | 40.5 / 41.1 | 15.0 | 256 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6635 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6635 | +0.005% | 0.00005 | 0.00210 | 6/6 | 96.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | -0.00002 [-0.00005, +0.00000] | 0.9985 [0.9981, 0.9989] | 0.0157 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 6.972e-06 | 5.741e-05 | 512.0 | 19/32 | 0 | 12574 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 181 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.55 s, warm TTFT 0.40 s, speedup 11.5x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 81 C, worst 5 s mean 80.4 C (limit 90), VRAM max 27708 MiB
- bus dc: junction max 80 C, worst 5 s mean 77.0 C (limit 90), VRAM max 27714 MiB
