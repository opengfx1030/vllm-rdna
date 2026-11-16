# Scoreboard: arm `tunable`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`, GPUs 2,3 (PCI d9 dc), tree sha `b2d569c402`
- started 2026-11-15 14:06:21, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.41 / 1.43 | 724 | 36.6 | 27.3 / 27.7 | 33.3 | 100 | 33.3 |
| 1024x512-c8 | 16 | 7.08 / 11.99 | 204 | 14.0 | 61.9 / 63.6 | 93.1 | 279 | 0.0 |
| 16384x1024-c1 | 8 | 28.68 / 28.84 | 571 | 29.6 | 33.8 / 34.3 | 16.2 | 275 | 0.0 |
| 16384x1024-c8 | 16 | 96.00 / 257.28 | 267 | 4.6 | 90.0 / 2455.0 | 23.2 | 395 | 0.0 |
| 32768x2048-c1 | 4 | 72.09 / 72.44 | 455 | 24.7 | 40.5 / 41.0 | 13.2 | 225 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6636 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6636 | +0.009% | 0.00009 | 0.00207 | 6/6 | 96.0 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | -0.00000 [-0.00003, +0.00002] | 0.9986 [0.9983, 0.9989] | 0.0156 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 6.436e-06 | 5.117e-05 | 512.0 | 22/32 | 0 | 12685 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 180 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.41 s, warm TTFT 0.41 s, speedup 10.7x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 2 near-tie, 0 soft, 0 solo-noise, 46 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 85 C, worst 5 s mean 82.8 C (limit 90), VRAM max 27146 MiB
- bus dc: junction max 81 C, worst 5 s mean 79.4 C (limit 90), VRAM max 27154 MiB
