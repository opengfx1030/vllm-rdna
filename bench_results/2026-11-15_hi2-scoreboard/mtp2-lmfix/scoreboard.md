# Scoreboard: arm `mtp2-lmfix`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `MTP=2 VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32`, GPUs 2,3 (PCI d9 dc), tree sha `f7edb8b1e7`
- started 2026-11-15 06:45:25, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 8 | 1.55 / 1.58 | 659 | 67.4 | 42.2 / 42.8 | 55.2 | 166 | 55.2 |
| 1024x512-c2 | 8 | 2.42 / 3.35 | 463 | 49.1 | 55.1 / 55.7 | 75.2 | 226 | 56.4 |
| 1024x512-c4 | 8 | 5.30 / 6.45 | 241 | 30.2 | 90.9 / 92.6 | 86.2 | 258 | 0.0 |
| 1024x512-c8 | 16 | 6.71 / 13.05 | 213 | 16.6 | 150.0 / 1658.2 | 100.4 | 301 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp | greedy identical | mean first divergence |
|---|---|---:|---:|---:|---|---:|
| arm | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 5/6 | 87.5 |
| floor | 3.6633 -> 3.6633 | +0.000% | 0.00000 | 0.00000 | 6/6 | 96.0 |
| judged | | | | within floor | | |

| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] | top-1 [95% CI] | abs dlogp p99 |
|---|---|---|---|---|---:|
| arm | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| floor | docs | 0.00000 [0.00000, 0.00000] | +0.00000 [+0.00000, +0.00000] | 1.0000 [1.0000, 1.0000] | 0.0000 |
| judged | docs | KL within floor | dNLL within floor | | |

| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) | identical | confident flips | positions |
|---|---:|---:|---:|---|---:|---:|
| arm | 7.103e-06 | 6.032e-05 | 512.0 | 22/32 | 0 | 13985 |
| floor | 2.368e-06 | 3.062e-05 | 512.0 | 28/32 | 0 | 15473 |
| judged | ABOVE floor | | | | | |

## 4. Validity

- READY after 210 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.84 s, warm TTFT 0.45 s, speedup 10.7x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 0 near-tie, 0 soft, 0 solo-noise, 48 identical of 48 checks
- KFD evictions: all zero (5 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 83 C, worst 5 s mean 81.0 C (limit 90), VRAM max 28856 MiB
- bus dc: junction max 79 C, worst 5 s mean 78.0 C (limit 90), VRAM max 28864 MiB
- speculative acceptance:

```text
spec_accept: conc=1 requests=24 completion_tokens=6092 in 119.2 s (51.1 tok/s aggregate, not a benchmark)
spec_accept: drafts=2632 draft_tokens=5264 accepted=3454
spec_accept: mean acceptance length 2.312, draft acceptance rate 65.6%, per position [0.766, 0.546]
spec_accept: PASS
spec_accept: conc=4 requests=24 completion_tokens=6074 in 67.0 s (90.7 tok/s aggregate, not a benchmark)
spec_accept: drafts=2601 draft_tokens=5202 accepted=3473
spec_accept: mean acceptance length 2.335, draft acceptance rate 66.8%, per position [0.777, 0.559]
spec_accept: PASS
```
