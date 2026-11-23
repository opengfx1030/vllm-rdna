# Inkling-2-Mini on gfx1030: serving benchmarks (TP=1, one Radeon PRO V620)

Build: `rdna_extras` after PR #195 (code of `0fd3e424dc`), native NVFP4/MXFP8 weights, no
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

## Not comparable

Other published numbers for this model (for example RX 6700 XT, 4-bit GGUF) use a
different card (384 GB/s), different weights (requantized) and an unstated context.
