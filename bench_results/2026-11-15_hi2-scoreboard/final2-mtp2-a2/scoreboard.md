# Scoreboard: arm `final2-mtp2-a2`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 RDNA_HINT_FLEX_FUSE_SILU=1 TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2 RDNA_HINT_FLEX_PREFILL_GEMM=hip MTP=2`, GPUs 0,1 (PCI d1 d4), tree sha `unknown`
- started 2026-11-15 20:31:35, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.30 / 1.32 | 787 | 86.2 | 33.3 / 33.7 | 70.2 | 211 | 70.2 |
| 1024x512-c8 | 16 | 5.51 / 10.44 | 276 | 25.6 | 92.6 / 1315.1 | 146.6 | 440 | 9.2 |
| 16384x1024-c1 | 8 | 20.88 / 20.96 | 785 | 69.5 | 41.6 / 42.2 | 28.7 | 488 | 0.0 |
| 16384x1024-c8 | 16 | 68.12 / 179.37 | 374 | 6.8 | 196.0 / 1641.3 | 32.9 | 560 | 0.0 |
| 32768x2048-c1 | 4 | 45.91 / 46.02 | 714 | 58.7 | 50.3 / 51.6 | 25.4 | 431 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6631 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6631 | -0.006% | -0.00006 | 0.00210 | 5/6 | 87.5 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | ABOVE floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00001 [0.00001, 0.00001] | -0.00001 [-0.00003, +0.00002] | 0.9987 [0.9982, 0.9990] | 0.0159 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL ABOVE floor | dNLL ABOVE floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 8.076e-06 | 7.182e-05 | 512.0 | 21/32 | 0 | 12425 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 190 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 3.98 s, warm TTFT 0.42 s, speedup 9.6x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d1: junction max 85 C, worst 5 s mean 83.4 C (limit 90), VRAM max 27593 MiB
- bus d4: junction max 84 C, worst 5 s mean 82.4 C (limit 90), VRAM max 27599 MiB
- speculative acceptance:

```text
spec_accept: conc=1 requests=24 completion_tokens=6092 in 95.5 s (63.8 tok/s aggregate, not a benchmark)
spec_accept: drafts=2638 draft_tokens=5276 accepted=3448
spec_accept: mean acceptance length 2.307, draft acceptance rate 65.4%, per position [0.763, 0.544]
spec_accept: PASS
spec_accept: conc=4 requests=24 completion_tokens=6092 in 48.4 s (126.0 tok/s aggregate, not a benchmark)
spec_accept: drafts=2630 draft_tokens=5260 accepted=3458
spec_accept: mean acceptance length 2.315, draft acceptance rate 65.7%, per position [0.768, 0.547]
spec_accept: PASS
```
