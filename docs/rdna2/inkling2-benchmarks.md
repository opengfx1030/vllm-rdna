# Inkling-2-Mini on gfx1030: serving benchmarks (TP=1, one Radeon PRO V620)

Build for sections 1-4: `rdna_extras` after PR #195 (code of `0fd3e424dc`), native NVFP4/MXFP8 weights, no
draft model unless stated. Recipe `inkling2-mini-tp1`: max model len 180,000, KV pool
180,756 tokens, CUDA graphs `FULL_AND_PIECEWISE` (0.70 GiB), prefix cache on. One
V620 (gfx1030, 512 GB/s). All numbers are over HTTP (`vllm serve`) at c=1 unless the cell
says otherwise. KFD `evicted_ms` was sampled in every window: 0 jumps. Max GPU
temperature 86 C.

## 1. Prefill and decode (c=1)

Fresh random-token prompts, a unique `cache_salt` per request (no prefix-cache hit),
8K warm-up, 3 runs per cell, TTFT from the HTTP stream with `max_tokens=1`.
Mean (min-max) tok/s.

| Prompt tokens | Prefill tok/s |
|---|---|
| 2,048 | 1939 (1923-1960) |
| 8,192 | 1813 (1799-1834) |
| 16,384 | 1704 (1694-1721) |
| 32,768 | 1529 (1520-1541) |
| 65,536 | 1279 (1266-1297) |

Real-text prompts (unique corpus slices, 3 runs) are 2-3% lower: 1892 / 1767 / 1664 /
1496 tok/s at 2K / 8K / 16K / 32K.

| Decode | tok/s |
|---|---|
| Short agent requests (20 chat requests x 3 rounds, 303-1444 prompt tokens, 128-512 max tokens, natural stop, default thinking on, n=60) | 56.4 (54.9-58.5), 17.7 ms/token |
| 128 output tokens after a 16,384-token prompt, random tokens (3 runs) | 51.7 (51.6-51.8) |
| The same, real-text prompt | 52.2 (52.1-52.2) |

## 2. Load: TTFT, ITL, c=1 and c=4

`vllm bench serve`, random prompts, 2 runs per cell (4 requests at c=1, 8 at c=4),
mean of the 2 runs. "BM on" is the default (dot2 32-row prefill expert kernels);
"BM off" is `VLLM_INKLING2_PREFILL_BM=0` (the pre-PR prefill kernels, same weights).

| Cell | Arm | TTFT p50 / p99 (s) | Prefill tok/s/req | ITL p50 / p99 (ms) | Decode tok/s/req | Output tok/s (agg) |
|---|---|---|---|---|---|---|
| 1k/512 c=1 | BM on | 0.53 / 0.54 | 1951 | 18.2 / 21.8 | 54.9 | 52.0 |
| | BM off | 0.78 / 0.79 | 1321 | 18.2 / 22.4 | 54.8 | 50.7 |
| 1k/512 c=4 | BM on | 1.60 / 2.10 | 706 | 27.6 / 33.5 | 34.8 | 126.3 |
| | BM off | 2.36 / 3.09 | 459 | 27.4 / 33.8 | 34.5 | 119.3 |
| 16k/1k c=1 | BM on | 9.68 / 9.80 | 1694 | 19.5 / 25.1 | 51.2 | 34.5 |
| | BM off | 13.79 / 13.92 | 1190 | 19.6 / 31.4 | 51.1 | 30.3 |
| 16k/1k c=4 | BM on | 12.04 / 38.20 | 910 | 31.7 / 635.6 | 20.5 | 58.0 |
| | BM off | 17.14 / 54.29 | 640 | 31.6 / 897.9 | 18.1 | 47.0 |

Prefill TTFT is 31% (1k) and 30% (16k) lower with BM on; decode is unchanged, as
designed. The c=4 16k ITL p99 (636 / 898 ms) is decode steps waiting behind other
requests' 16k prefill chunks; the p50 is 31.7 ms. Two runs per cell: the p99 columns
are loose.

## 3. A conversation filled to 80% of max context, prefix cache on

One conversation grown turn by turn (about 8k, 12k or 16k new tokens of real text and
code per turn plus the previous reply), 128 output tokens per turn, prefix cache on.
`cached` is the server's prefix-cache hit (rounded to 64-token blocks). "New tok/s" is
prefill speed over the uncached tokens only.

| Turn | Context | Prompt | Cached | New tokens | TTFT (s) | New tok/s | Decode tok/s | ITL p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8,399 | 8,271 | 0 | 8,271 | 4.48 | 1845 | 54.5 | 18.6 |
| 2 | 20,840 | 20,712 | 8,256 | 12,456 | 7.76 | 1606 | 51.9 | 19.6 |
| 3 | 37,376 | 37,248 | 20,672 | 16,576 | 12.23 | 1355 | 49.4 | 21.0 |
| 4 | 49,815 | 49,687 | 37,184 | 12,503 | 10.75 | 1163 | 47.5 | 21.7 |
| 5 | 58,160 | 58,032 | 49,664 | 8,368 | 8.01 | 1045 | 45.7 | 22.4 |
| 6 | 70,601 | 70,473 | 57,984 | 12,489 | 12.83 | 973 | 44.0 | 23.3 |
| 7 | 87,138 | 87,010 | 70,464 | 16,546 | 18.93 | 874 | 42.1 | 24.5 |
| 8 | 99,578 | 99,450 | 86,976 | 12,474 | 15.79 | 790 | 40.7 | 25.4 |
| 9 | 107,923 | 107,795 | 99,392 | 8,403 | 11.49 | 731 | 40.7 | 26.0 |
| 10 | 120,361 | 120,233 | 107,776 | 12,457 | 18.15 | 686 | 38.1 | 26.9 |
| 11 | 136,898 | 136,770 | 120,192 | 16,578 | 25.74 | 644 | 36.9 | 28.0 |
| 12 | 143,945 | 143,817 | 136,768 | 7,049 | 11.74 | 600 | 36.1 | 28.6 |

Every turn after the first reuses the whole earlier conversation from the cache. The
cost that remains is the new tokens attending to the long cached prefix: new-token
prefill falls from 1845 to 600 tok/s over the fill, and decode from 54.5 to 36.1 tok/s.

Same conversation, prefix cache **off** (fresh server, last 3 turns only, so each turn
prefills its whole prompt):

| Turn (prompt tokens) | TTFT, cache off (s) | Prefill tok/s | TTFT, cache on (s) |
|---|---|---|---|
| 10 (120,233) | 123.4 | 974 | 18.1 |
| 11 (136,770) | 150.7 | 907 | 25.7 |
| 12 (143,817) | 162.9 | 883 | 11.7 |

## 4. With the optional DSpark draft (adaptive K)

Opt-in; not in the default recipe. Tree `0fd3e424dc` plus the adaptive-K python of the
stacked branch; `num_speculative_tokens` 7 with per-batch-size `[[1,2,7],[3,8,0]]`,
32,768-id draft vocabulary. **Max model len 100,000** (the draft needs 2 sequences and
its own memory: KV pool 105,340 tokens, graphs 1.38 GiB, weights 19.74 GiB including the
draft, boot 75 s). Same 20 short requests as section 1, 3 rounds.

| Decode | no draft | DSpark |
|---|---|---|
| Short agent requests | 56.4 | 101.1 (median 101.2, min 67.3, max 146.1), 9.9 ms/token |
| 128 tokens after a 16,384-token prompt, random tokens | 51.7 | 71.9 (70.0-75.0) |
| The same, real-text prompt | 52.2 | 75.8 (70.3-79.7) |

By request kind (tok/s): chat 86.4, code_gen 106.9, code_edit 103.6, structured 102.9,
rag 109.9. Accepted tokens per step: 2.30-2.39. Prefill is unchanged (TTFT after 16k:
9.66 / 9.97 s). Thinking is on by default and the reasoning text is repetitive, which
helps the draft; a six-prompt-type mix at b=1 averages 82.7 tok/s.
Greedy output is coherent on the 3 coherence prompts; outputs differ in length from the
no-draft run, which is the fp16 floor.

## 5. After the pipelined prefill attention tile (this PR)

Before = sections 1 and 3 (PR #195 head). After = this PR's head (`8ce0d65559`, `_rocm_C` a6af06d3), recipe defaults (max model len 180,000, KV pool 180,756 tokens, graphs 0.70 GiB, prefix cache on, TP=1, no draft, boot 57 s). Prefill rows: 3 runs, random-token prompts, unique `cache_salt`, 8K warm-up. The fill replays the same saved conversation, so each turn has the same messages and the same cached tokens as before (before = one run, after = one run; a single turn varies by about 1-2%). 0 KFD `evicted_ms` jumps.

| prompt tokens | before tok/s | after tok/s | change | TTFT before -> after (s) |
|---|---|---|---|---|
| 2,048 | 1939 (1923-1960) | **1977 (1961-1996)** | +1.9% | 1.06 -> 1.04 |
| 8,192 | 1813 (1799-1834) | **1879 (1864-1899)** | +3.6% | 4.52 -> 4.36 |
| 16,384 | 1704 (1694-1721) | **1798 (1785-1815)** | +5.5% | 9.61 -> 9.11 |
| 32,768 | 1529 (1520-1541) | **1662 (1653-1675)** | +8.7% | 21.44 -> 19.72 |
| 65,536 | 1279 (1266-1297) | **1443 (1437-1451)** | +12.8% | 51.23 -> 45.41 |

128 output tokens after a 16,384-token prompt (decode tok/s): random-token prompt 51.66 (51.61-51.75) -> **51.48 (51.45-51.52)**; real-text prompt 52.15 (52.12-52.19) -> **52.15 (52.11-52.19)** (decode untouched, as expected). TTFT of that 16K prompt: 9.63 -> 9.15 s.

### Conversation grown to 144k, prefix cache on, per turn

| turn | context after turn | new tokens | prefill tok/s on new tokens: before -> after | TTFT s: before -> after | decode tok/s: before -> after | ITL p50 ms: before -> after |
|---|---|---|---|---|---|---|
| 1 | 8,399 | 8,271 | 1845 -> **1808** (-2%) | 4.48 -> **4.58** | 54.51 -> 54.14 | 18.6 -> 18.8 |
| 2 | 20,840 | 12,456 | 1606 -> **1632** (+2%) | 7.76 -> **7.63** | 51.85 -> 51.37 | 19.6 -> 19.8 |
| 3 | 37,376 | 16,576 | 1355 -> **1444** (+7%) | 12.23 -> **11.48** | 49.38 -> 48.20 | 21.0 -> 21.2 |
| 4 | 49,815 | 12,503 | 1163 -> **1284** (+10%) | 10.75 -> **9.73** | 47.51 -> 46.42 | 21.7 -> 21.9 |
| 5 | 58,160 | 8,368 | 1045 -> **1178** (+13%) | 8.01 -> **7.10** | 45.66 -> 45.31 | 22.4 -> 22.6 |
| 6 | 70,601 | 12,489 | 973 -> **1109** (+14%) | 12.83 -> **11.26** | 44.02 -> 43.48 | 23.3 -> 23.6 |
| 7 | 87,138 | 16,546 | 874 -> **1017** (+16%) | 18.93 -> **16.27** | 42.08 -> 41.59 | 24.5 -> 24.7 |
| 8 | 99,578 | 12,474 | 790 -> **933** (+18%) | 15.79 -> **13.37** | 40.69 -> 40.22 | 25.4 -> 25.6 |
| 9 | 107,923 | 8,403 | 731 -> **872** (+19%) | 11.49 -> **9.63** | 40.72 -> 39.86 | 26.0 -> 26.3 |
| 10 | 120,361 | 12,457 | 686 -> **836** (+22%) | 18.15 -> **14.90** | 38.08 -> 38.15 | 26.9 -> 27.2 |
| 11 | 136,898 | 16,578 | 644 -> **787** (+22%) | 25.74 -> **21.07** | 36.89 -> 36.61 | 28.0 -> 28.3 |
| 12 | 143,945 | 7,049 | 600 -> **732** (+22%) | 11.74 -> **9.63** | 36.09 -> 35.95 | 28.6 -> 28.8 |

Decode is untouched by the kernel. The after-run decode reads 0.2-1.6% lower in most turns; GPU temperature was 4-8 C higher in that run (83-87 C against 64-81 C), so [INFERENCE] this is thermal noise; it was not investigated further.

## 6. TP=2 (two V620)

Recipe `inkling2-mini-tp2`: max model len 1,048,576, KV pool 1,062,482 tokens, graphs 0.21 GiB per GPU, prefix cache on, no draft, weights 9.61 GiB per GPU, boot 66 s. Same protocol as section 1 on the PR #195 build; TP=1 column is section 1.

| Benchmark | TP=2 | TP=1 (T1) | PolyStrata post |
|---|---|---|---|
| Prefill: 2,048 tokens | **1671 (1663-1685)** tok/s | 1939 (1923-1960) | 992 |
| Prefill: 8,192 tokens | **1648 (1638-1663)** tok/s | 1813 (1799-1834) | 937 |
| Prefill: 16,384 tokens | **1602 (1595-1616)** tok/s | 1704 (1694-1721) | 883 |
| Prefill: 32,768 tokens | **1524 (1519-1533)** tok/s | 1529 (1520-1541) | 780 |
| Prefill: 65,536 tokens | **1385 (1381-1392)** tok/s | 1279 (1266-1297) | 628 |
| Decode: short agent requests | **80.5** (min 78.3, max 83.3, n=60) | 56.4 | 46.5-49.8 |
| Decode: 128 tokens after 16,384 (random / real text) | **75.3 (75.3-75.4)** / 75.8 (75.8-75.9) | 51.7 (51.6-51.7) / 52.2 (52.1-52.2) | 41.0 |

Real-text prefill TP=2: 2,048: 1552 (1534-1571); 8,192: 1530 (1520-1546); 16,384: 1487 (1472-1495); 32,768: 1416 (1399-1427) tok/s. One of the three real-text 32,768 requests had a KFD evicted_ms jump (value 1399-1427 inside the normal band, kept). TP=2 prefill is lower than TP=1 at 2K-16K ([INFERENCE] all-reduce cost), equal at 32K, higher at 64K; decode is +43% over TP=1.

TP=2 prefill is lower than TP=1 at 2K-16K ([INFERENCE] all-reduce cost), equal at 32K and higher at 64K; decode is +43%.

### TP=2 conversation grown to 262,144 tokens, prefix cache on

Same conversation and protocol as the TP=1 fill (single run). The fill stops at turn 22 (about 262k tokens), which is 25% of the 1,048,576 maximum, not 80%.

| turn | context after turn (tok) | prompt tokens | cached tokens | new tokens | TTFT s | prefill tok/s (new tokens) | decode tok/s | ITL p50 / p99 ms |
|---|---|---|---|---|---|---|---|---|
| 1 | 8,399 | 8,271 | 0 | 8,271 | 5.21 | 1586 | 77.84 | 13.1 / 13.3 |
| 2 | 20,840 | 20,712 | 8,256 | 12,456 | 8.50 | 1465 | 75.08 | 13.6 / 13.9 |
| 3 | 37,376 | 37,248 | 20,672 | 16,576 | 12.32 | 1345 | 71.72 | 14.3 / 17.3 |
| 4 | 49,817 | 49,689 | 37,184 | 12,505 | 10.10 | 1239 | 69.37 | 14.8 / 16.8 |
| 5 | 58,162 | 58,034 | 49,664 | 8,370 | 7.18 | 1165 | 67.93 | 15.0 / 16.3 |
| 6 | 70,603 | 70,475 | 57,984 | 12,491 | 11.23 | 1112 | 66.11 | 15.6 / 18.5 |
| 7 | 87,140 | 87,012 | 70,464 | 16,548 | 15.84 | 1045 | 64.57 | 16.2 / 16.6 |
| 8 | 99,580 | 99,452 | 86,976 | 12,476 | 12.77 | 977 | 63.40 | 16.6 / 17.0 |
| 9 | 107,925 | 107,797 | 99,392 | 8,405 | 9.06 | 928 | 61.98 | 16.9 / 17.4 |
| 10 | 120,363 | 120,235 | 107,776 | 12,459 | 13.86 | 899 | 60.62 | 17.4 / 17.9 |
| 11 | 136,900 | 136,772 | 120,192 | 16,580 | 19.31 | 859 | 58.21 | 18.0 / 18.4 |
| 12 | 149,341 | 149,213 | 136,768 | 12,445 | 15.35 | 811 | 57.16 | 18.5 / 19.1 |
| 13 | 157,686 | 157,558 | 149,184 | 8,374 | 10.87 | 770 | 56.47 | 18.8 / 20.4 |
| 14 | 170,127 | 169,999 | 157,504 | 12,495 | 16.55 | 755 | 55.48 | 19.2 / 19.8 |
| 15 | 186,663 | 186,535 | 169,984 | 16,551 | 22.71 | 729 | 53.89 | 19.8 / 20.4 |
| 16 | 199,104 | 198,976 | 186,496 | 12,480 | 17.40 | 717 | 52.13 | 20.2 / 20.5 |
| 17 | 207,449 | 207,321 | 198,912 | 8,409 | 12.41 | 678 | 51.47 | 20.6 / 20.9 |
| 18 | 219,889 | 219,761 | 207,296 | 12,465 | 18.82 | 662 | 50.95 | 21.1 / 21.6 |
| 19 | 236,426 | 236,298 | 219,712 | 16,586 | 26.01 | 638 | 49.38 | 21.7 / 22.6 |
| 20 | 248,867 | 248,739 | 236,288 | 12,451 | 20.50 | 607 | 48.31 | 22.2 / 22.6 |
| 21 | 257,211 | 257,083 | 248,704 | 8,379 | 14.05 | 596 | 48.87 | 22.4 / 24.1 |
| 22 | 262,086 | 261,958 | 257,024 | 4,934 | 8.83 | 559 | 47.56 | 22.7 / 23.9 |

## Not comparable

Other published numbers for this model (for example RX 6700 XT, 4-bit GGUF) use a
different card (384 GB/s), different weights (requantized) and an unstated context.
