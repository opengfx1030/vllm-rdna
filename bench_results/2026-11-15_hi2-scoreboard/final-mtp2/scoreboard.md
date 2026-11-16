# Scoreboard: arm `final-mtp2`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 RDNA_HINT_FLEX_FUSE_SILU=1 TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2 MTP=2`, GPUs 2,3 (PCI d9 dc), tree sha `b2d569c402`
- started 2026-11-15 16:42:13, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.43 / 1.44 | 717 | 85.5 | 33.4 / 33.9 | 68.6 | 206 | 68.6 |
| 1024x512-c8 | 16 | 7.41 / 12.72 | 204 | 24.9 | 94.1 / 1624.5 | 134.0 | 402 | 8.4 |
| 16384x1024-c1 | 8 | 23.14 / 23.33 | 708 | 68.8 | 41.9 / 43.5 | 26.9 | 457 | 0.0 |
| 16384x1024-c8 | 16 | 81.06 / 218.00 | 327 | 6.1 | 197.1 / 1920.5 | 28.6 | 486 | 0.0 |
| 32768x2048-c1 | 4 | 50.48 / 50.79 | 649 | 58.4 | 50.6 / 51.8 | 23.9 | 407 | 0.0 |

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
| arm | 8.363e-06 | 6.717e-05 | 326.0 | 13/32 | 0 | 10235 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 191 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.28 s, warm TTFT 0.42 s, speedup 10.1x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 85 C, worst 5 s mean 81.8 C (limit 90), VRAM max 27595 MiB
- bus dc: junction max 80 C, worst 5 s mean 78.6 C (limit 90), VRAM max 27601 MiB
- speculative acceptance:

```text
spec_accept: conc=1 requests=24 completion_tokens=6092 in 95.8 s (63.6 tok/s aggregate, not a benchmark)
spec_accept: drafts=2638 draft_tokens=5276 accepted=3448
spec_accept: mean acceptance length 2.307, draft acceptance rate 65.4%, per position [0.763, 0.544]
spec_accept: PASS
spec_accept: conc=4 requests=24 completion_tokens=6092 in 49.3 s (123.6 tok/s aggregate, not a benchmark)
spec_accept: drafts=2630 draft_tokens=5260 accepted=3458
spec_accept: mean acceptance length 2.315, draft acceptance rate 65.7%, per position [0.768, 0.547]
spec_accept: PASS
```
