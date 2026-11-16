# Scoreboard: arm `stack`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 RDNA_HINT_FLEX_FUSE_SILU=1`, GPUs 2,3 (PCI d9 dc), tree sha `dd1d104148`
- started 2026-11-15 09:09:57, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.48 / 1.50 | 690 | 45.2 | 22.1 / 22.4 | 40.0 | 120 | 40.0 |
| 1024x512-c8 | 16 | 7.12 / 12.13 | 203 | 22.3 | 35.2 / 36.4 | 134.6 | 404 | 0.0 |
| 16384x1024-c1 | 8 | 24.52 / 24.69 | 668 | 34.9 | 28.6 / 29.0 | 19.0 | 324 | 0.0 |
| 16384x1024-c8 | 16 | 79.83 / 211.90 | 317 | 6.1 | 63.3 / 1811.0 | 29.3 | 498 | 0.0 |
| 32768x2048-c1 | 4 | 53.06 / 53.39 | 618 | 28.2 | 35.4 / 35.9 | 16.3 | 277 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6633 | +0.002% | 0.00002 | 0.00210 | 5/6 | 87.5 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | +0.00001 [-0.00002, +0.00003] | 0.9986 [0.9982, 0.9989] | 0.0159 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 8.317e-06 | 6.953e-05 | 498.0 | 16/32 | 0 | 10664 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 160 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.50 s, warm TTFT 0.39 s, speedup 11.4x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 3 near-tie, 0 soft, 0 solo-noise, 45 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 82 C, worst 5 s mean 80.4 C (limit 90), VRAM max 27020 MiB
- bus dc: junction max 80 C, worst 5 s mean 76.8 C (limit 90), VRAM max 27028 MiB
