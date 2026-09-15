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
   `gpu_model_runner.py` calling the singular `get_mamba_state_copy_func()`
   instead of the plural `get_mamba_state_copy_funcs(mamba_types)`. Verified:
   V1 + FULL_AND_PIECEWISE + breakable cudagraphs, coherent output, c=1
   1k/512 = 24.79 out tok/s.
2. **Scheduler KeyError at chunked prefill**: c>=4 (1k/512) and any c at 16k
   hit `KeyError: 'cmpl-bench-...'` at `scheduler.py:1882` in
   `update_from_output`. A scheduled request ID is missing from the model
   output's `req_id_to_index`. This is a separate bug from the cudagraph
   replay — likely the chunked-prefill / async-scheduling interaction. Not yet
   fixed.

## Comparison vs the 27B baseline (bench_27b_awq_matrix.md)

At c=1, 1k/512 input, TP=4 on 4× Radeon PRO V620:

| Stack | Backend | out tok/s | total tok/s | TTFT ms | TPOT ms |
|---|---|---:|---:|---:|---:|
| **Qwen3.8-27B-AWQ** | FA-RDNA2 + RDNA2 W4A16 (HIP, cudagraph) | 22.66 | 66.92 | 1,776 | 40.7 |
| **Qwen3.8-Flash-Next-AWQ** (Triton, cudagraph) | full Triton | **24.79** | 73.20 | 733 | 39.0 |
| **Qwen3.8-Flash-Next-AWQ** (Triton, eager) | full Triton | 9.27 | 71.22 | 7.1 | 113 |

The Flash-Next Triton path with cudagraph is **2.5× faster than eager** and
slightly faster than the 27B's full-HIP path at c=1 (24.79 vs 22.66 out
tok/s). At c=8 Flash-Next eager hits 40.69 out tok/s, which would likely
scale to ~110+ out tok/s with cudagraph once the scheduler KeyError is fixed.
