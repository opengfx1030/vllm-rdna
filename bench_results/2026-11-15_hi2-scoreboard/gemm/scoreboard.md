# Scoreboard: arm `gemm`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_PREFILL_GEMM=hip TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`, GPUs 2,3 (PCI d9 dc), tree sha `b47eb6c384`
- started 2026-11-15 17:28:38, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.33 / 1.34 | 769 | 36.5 | 27.4 / 30.5 | 33.4 | 100 | 33.4 |
| 1024x512-c8 | 16 | 6.00 / 10.30 | 236 | 14.3 | 62.0 / 68.0 | 97.5 | 292 | 0.0 |
| 16384x1024-c1 | 8 | 26.87 / 27.10 | 610 | 29.6 | 33.8 / 34.1 | 16.7 | 283 | 0.0 |
| 16384x1024-c8 | 16 | 83.21 / 221.86 | 307 | 5.0 | 90.0 / 2142.7 | 26.0 | 442 | 0.0 |
| 32768x2048-c1 | 4 | 68.54 / 68.81 | 478 | 24.7 | 40.5 / 41.0 | 13.5 | 230 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6631 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6631 | -0.004% | -0.00004 | 0.00200 | 6/6 | 96.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | -0.00001 [-0.00004, +0.00002] | 0.9986 [0.9983, 0.9989] | 0.0157 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 9.482e-06 | 6.504e-05 | 512.0 | 20/32 | 0 | 12220 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 181 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.21 s, warm TTFT 0.42 s, speedup 10.1x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 2 near-tie, 0 soft, 0 solo-noise, 46 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 85 C, worst 5 s mean 83.2 C (limit 90), VRAM max 27147 MiB
- bus dc: junction max 82 C, worst 5 s mean 79.8 C (limit 90), VRAM max 27155 MiB
