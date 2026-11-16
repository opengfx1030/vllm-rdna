# Scoreboard: arm `final-a2`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 RDNA_HINT_FLEX_FUSE_SILU=1 TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`, GPUs 0,1 (PCI d1 d4), tree sha `unknown`
- started 2026-11-15 18:18:47, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.35 / 1.36 | 757 | 45.5 | 22.0 / 22.6 | 40.7 | 122 | 40.7 |
| 1024x512-c8 | 16 | 6.76 / 11.48 | 215 | 22.7 | 34.9 / 36.4 | 137.7 | 413 | 0.0 |
| 16384x1024-c1 | 8 | 21.81 / 21.88 | 751 | 35.2 | 28.5 / 28.9 | 20.1 | 342 | 0.0 |
| 16384x1024-c8 | 16 | 75.95 / 202.26 | 333 | 6.2 | 63.0 / 1743.9 | 30.3 | 515 | 0.0 |
| 32768x2048-c1 | 4 | 47.43 / 47.58 | 691 | 28.5 | 35.2 / 35.8 | 17.2 | 292 | 0.0 |

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
| arm | 8.039e-06 | 6.397e-05 | 512.0 | 21/32 | 0 | 12606 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 160 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.12 s, warm TTFT 0.39 s, speedup 10.6x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 3 near-tie, 0 soft, 0 solo-noise, 45 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d1: junction max 85 C, worst 5 s mean 82.0 C (limit 90), VRAM max 27020 MiB
- bus d4: junction max 85 C, worst 5 s mean 80.6 C (limit 90), VRAM max 27028 MiB
