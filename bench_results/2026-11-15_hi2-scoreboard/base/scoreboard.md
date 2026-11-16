# Scoreboard: arm `base`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides ``, GPUs 2,3 (PCI d9 dc), tree sha `ef0b99ea9b+scoreboard`
- started 2026-11-15 01:33:23, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|
| 1024x512-c1 | 16 | 1.51 / 1.53 | 678 | 36.5 | 27.4 / 30.5 | 33.0 | 99 | 33.0 |
| 1024x512-c8 | 16 | 7.00 / 12.54 | 172 | 13.8 | 62.0 / 63.4 | 92.5 | 278 | 0.0 |
| 16384x1024-c1 | 8 | 30.54 / 30.69 | 536 | 29.6 | 33.8 / 34.2 | 15.7 | 267 | 0.0 |
| 16384x1024-c8 | 16 | 97.14 / 259.63 | 264 | 4.6 | 90.1 / 2456.4 | 23.2 | 394 | 0.0 |
| 32768x2048-c1 | 4 | 75.75 / 76.08 | 433 | 24.7 | 40.5 / 41.1 | 12.9 | 220 | 0.0 |

## 3. Quality

- short PPL 5.730; long-context PPL 3.6633 over passages of [17081, 14850, 16697, 21982, 21805, 22636] tokens

## 4. Validity

- READY after 180 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.72 s, warm TTFT 0.41 s, speedup 11.4x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 1 near-tie, 0 soft, 0 solo-noise, 47 identical of 48 checks
- KFD evictions: all zero (6 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 84 C, worst 5 s mean 82.4 C (limit 90), VRAM max 27143 MiB
- bus dc: junction max 81 C, worst 5 s mean 79.8 C (limit 90), VRAM max 27151 MiB
