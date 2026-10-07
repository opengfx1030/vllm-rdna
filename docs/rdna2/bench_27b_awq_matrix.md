# 27B AWQ bench matrix (gfx1030, TP=4)

Config: Qwen3.8-27B-AWQ-INT4, TP=4, FA-RDNA2 + RDNA2 W4A16
(VLLM_USE_RDNA2_FA=1, default RDNA2W4A16LinearKernel), FULL_AND_PIECEWISE,
breakable cudagraphs, PYNCCL, prefix caching, gpu-mem-util 0.90, KV 12e9,
max-model-len 32768. Date: 2026-09-14.

| Workload | concurrency | out tok/s | total tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---:|---:|---:|---:|
| 1k/512 | 1 | 22.66 | 66.92 | 1,776 | 40.7 |
| 1k/512 | 4 | 54.43 | 160.75 | 9,842 | 54.3 |
| 1k/512 | 8 | 133.83 | 395.21 | 5,382 | 49.2 |
| 16k/1k | 1 | 16.22 | 275.76 | 24,996 | 36.7 |
| 16k/1k | 4 | 27.46 | 466.74 | 50,639 | 94.9 |
| 16k/1k | 8 | 45.35 | 770.98 | 71,126 | 104.8 |

Notes:
- 0 failed requests, prefix caching on, deterministic sampling.
- Aggregate throughput rises with concurrency; TTFT balloons because the
  system becomes prefill-dominated at high c (same profile as the earlier
  16k/1k c=8 = 28.28 tok/s cell).
- The Triton-FA / Triton-W4A16 cells are not included: their JIT warmup on
  gfx1030 exceeds even the raised 1800s distributed timeout (see
  flash-next-ple-ipc.md §"Triton warmup"), so the HIP paths are the practical
  config. They should be re-benched once the Flash-Next HIP path is finalised
  and compared against these numbers.
