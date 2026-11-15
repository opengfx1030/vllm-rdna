# Scoreboard: arm `gdnglue`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `VLLM_RDNA_GDN_DECODE_INPLACE=1`, GPUs 2,3 (PCI d9 dc), tree sha `dd1d104148`
- started 2026-11-15 08:13:05, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.51 / 1.52 | 680 | 41.6 | 24.1 / 24.3 | 37.1 | 111 | 37.1 |
| 1024x512-c8 | 16 | 7.23 / 12.31 | 199 | 15.3 | 55.7 / 57.2 | 99.8 | 300 | 0.0 |
| 16384x1024-c1 | 8 | 30.54 / 30.68 | 537 | 32.8 | 30.5 / 30.8 | 16.6 | 282 | 0.0 |
| 16384x1024-c8 | 16 | 97.10 / 259.53 | 264 | 4.8 | 83.5 / 2456.7 | 23.6 | 401 | 0.0 |
| 32768x2048-c1 | 4 | 75.75 / 76.08 | 433 | 26.9 | 37.2 / 37.7 | 13.5 | 229 | 0.0 |

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
| arm | 2.400e-06 | 3.062e-05 | 512.0 | 30/32 | 0 | 16053 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | within floor | | | | | |

## 4. Validity

- READY after 181 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.74 s, warm TTFT 0.41 s, speedup 11.5x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 0 near-tie, 0 soft, 0 solo-noise, 48 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 83 C, worst 5 s mean 82.6 C (limit 90), VRAM max 26449 MiB
- bus dc: junction max 80 C, worst 5 s mean 79.6 C (limit 90), VRAM max 26457 MiB
