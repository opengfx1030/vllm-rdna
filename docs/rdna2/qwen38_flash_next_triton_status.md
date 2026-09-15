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
1. **Capture replay page fault**: the FULL_AND_PIECEWISE capture succeeds (7s)
   but the first replay faults with a **GPU page fault** (dmesg: `client 0x1b
   (UTCL2)` at a host VA `0x00007fac...`) -> the captured graph references a
   buffer that moved/unmapped after capture. The QSA is an eager break point.
   A different kernel reads a stale host address on replay (likely a registered
   host buffer - PLE/offload - or a captured pointer not held by the graph-pool
   allocator). Needs a first-replay torch.profiler or a HIP-graph-memory audit.
2. Compare vs the 27B baseline (docs/rdna2/bench_27b_awq_matrix.md).
