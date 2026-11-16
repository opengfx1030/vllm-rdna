# Scoreboard: arm `final`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 RDNA_HINT_FLEX_FUSE_SILU=1 TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`, GPUs 2,3 (PCI d9 dc), tree sha `b2d569c402`
- started 2026-11-15 15:54:04, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.39 / 1.40 | 737 | 45.2 | 22.1 / 25.0 | 40.3 | 121 | 40.3 |
| 1024x512-c8 | 16 | 6.97 / 11.80 | 209 | 22.4 | 35.2 / 39.6 | 135.4 | 406 | 0.0 |
| 16384x1024-c1 | 8 | 22.75 / 22.90 | 720 | 34.9 | 28.6 / 31.8 | 19.7 | 335 | 0.0 |
| 16384x1024-c8 | 16 | 78.63 / 209.54 | 322 | 6.1 | 63.5 / 1808.4 | 29.4 | 500 | 0.0 |
| 32768x2048-c1 | 4 | 49.40 / 49.69 | 663 | 28.3 | 35.4 / 37.5 | 16.8 | 286 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6635 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6635 | +0.007% | 0.00007 | 0.00217 | 5/6 | 87.5 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | +0.00001 [-0.00002, +0.00003] | 0.9986 [0.9981, 0.9989] | 0.0158 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 7.011e-06 | 6.174e-05 | 512.0 | 21/32 | 0 | 12606 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 160 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.22 s, warm TTFT 0.39 s, speedup 10.7x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 84 C, worst 5 s mean 81.4 C (limit 90), VRAM max 27022 MiB
- bus dc: junction max 80 C, worst 5 s mean 78.0 C (limit 90), VRAM max 27030 MiB
