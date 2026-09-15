# Qwen3.8-Flash-Next on gfx1030 — Triton path status (2026-09-15)

## Working: Triton path in `--enforce-eager` (coherent baseline)

The `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` checkpoint (Qwen4Exp, 512-expert
W4A16 MoE, PLE offload) now **serves coherenently at TP=4** in `--enforce-eager`
mode. The earlier SIGABRT was the **cudagraph capture** (a capture-unsafe op in
the QSA forward), not the model — eagerly it runs correctly.

Coherence (all coherent):
- "The capital of France is" -> "Paris is a city. The capital of France is Paris"
- "1+1=" -> "2=3，3+1=4，4+..."
- "What is the capital of Japan?" -> coherent continuation

### What fixed the eager path
- `do_kv_cache_update` override in `Qwen4ExpQSAFlashAttentionImpl`
  (`vllm/models/qwen4_exp/amd/qsa.py`, from PR #5/GeorgeMA-Strong): use the
  native `ops.reshape_and_cache_flash` (ROCm HIP cache writer) instead of the
  FlashAttention base which requires flash_attn.
- The whole PLE/offload/cache stack (see flash-next-ple-ipc.md +
  qwen4_exp_hip_tracker.md).

## Bench (TP=4, eager, 0 failed for 1k/512)

| Workload | c | out tok/s | total tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---:|---:|---:|---:|
| 1k/512 | 1 | 9.27 | 27.37 | 6,511 | 95.4 |
| 1k/512 | 4 | 12.11 | 35.76 | 64,884 | 203.9 |
| 1k/512 | 8 | 40.69 | 120.16 | 8,424 | 180.1 |

16k/1k cells fail in eager: the long prefill's activation memory exceeds the
GPU (no cudagraph to bound it) and the server crashes. Eager is also slow
(Triton MoE).

## Remaining for production
1. ~~**Capture replay page fault**~~ **FIXED** (commit `8dc656b61`): the UVA
   staging buffers in `StagedWriteTensor` recorded host VAs into the captured
   graph. Fixed by adding GPU mirror tensors and routing staging writes through
   them when `torch.cuda.is_current_stream_capturing()` is True. Also fixed
   `gpu_model_runner.py:1168` calling the singular `get_mamba_state_copy_func()`
   instead of the plural `get_mamba_state_copy_funcs(mamba_types)`.
2. ~~**Scheduler KeyError at chunked prefill**~~ **FIXED** (commit `4224ce202`):
   the same singular/plural bug also affected two per-step call sites
   (`gpu_model_runner.py:1707`, `:4652`) that only triggered under
   eager + prefix-caching + mamba-hybrid + chunked-prefill. Cached the
   computed dict on `self._mamba_state_copy_funcs` in `_get_mamba_bufs` and
   reused it at both per-step call sites.

## Comparison vs the 27B baseline (bench_27b_awq_matrix.md)

Full bench matrix, TP=4 on 4× Radeon PRO V620, V1 + FULL_AND_PIECEWISE +
breakable cudagraphs:

| Workload | concurrency | Flash-Next out tok/s | 27B out tok/s | ratio |
|---|---|---:|---:|---:|
| 1k/512 | 1 | **24.79** | 22.66 | **1.09×** |
| 1k/512 | 4 | **78.56** | 54.43 | **1.44×** |
| 1k/512 | 8 | 35.26 | **133.83** | 0.26× |
| 16k/1k | 1 | **20.06** | 16.22 | **1.24×** |
| 16k/1k | 4 | **36.01** | 27.46 | **1.31×** |
| 16k/1k | 8 | 44.68 | **45.35** | 0.99× |

Key findings:
- **Flash-Next wins at c=1 and c=4 for both workloads** (1.09×–1.44× the 27B)
- **1k/512 c=8 regression** (0.26×): the Flash-Next becomes prefill-dominated
  at high concurrency — TTFT balloons to 46.4s and TPOT to 136ms. The 27B's
  dense decode path handles c=8 better. Likely fixable by tuning chunked-prefill
  parameters (the 27B sweep showed `--max-num-batched-tokens=4096` regressed
  TTFT by 3.4×; the Flash-Next may need a different sweet spot).
- **16k/1k c=8 essentially ties** (0.99×): both models are prefill-bound at
  this point; the gap closes as the bottleneck shifts to prefill throughput
  rather than decode.

## Full Flash-Next bench data

| Workload | concurrency | out tok/s | total tok/s | TTFT ms | TPOT ms | duration s |
|---|---|---:|---:|---:|---:|---:|
| 1k/512 | 1 | 24.79 | 73.20 | 733 | 39.0 | 82.6 |
| 1k/512 | 4 | 78.56 | 232.00 | 1,569 | 47.9 | 104.3 |
| 1k/512 | 8 | 35.26 | 104.14 | 46,435 | 136.4 | 464.6 |
| 16k/1k | 1 | 20.06 | 340.96 | 9,418 | 40.5 | 49.9 |
| 16k/1k | 4 | 36.01 | 612.21 | 26,003 | 84.9 | 222.2 |
| 16k/1k | 8 | 44.68 | 759.53 | 35,536 | 143.1 | 358.1 |

All cells: 0 failed requests, prefix caching ON, deterministic sampling.
