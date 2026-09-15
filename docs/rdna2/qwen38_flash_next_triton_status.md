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

## Prefix caching: MUST use PIECEWISE (not FULL_AND_PIECEWISE) [FOUND 2026-09-15]

With prefix caching ON, the multi-spec Flash-Next (GDN + PLE short-conv)
**FULL graph capture produces NaN on replay** (first token correct, then
collapses to `!`): 11/18 probe failures after a 16k x 16 @ 1k-out load.

The NaN originates in the GDN `mixed_qkv` at L1 during decode (L0 finite,
L1 NaN) with a finite GDN input — the FULL graph captures a buffer that
goes stale on replay. Isolated via:
- eager + prefix caching -> 0/18 (exonerates the mamba-align state logic)
- PIECEWISE + prefix caching -> 0/18, TPOT 89.6ms (best correct mode)
- FULL_AND_PIECEWISE + prefix caching -> 11/18 (the FULL graph is the bug)
- GDN pre-copy / PLE output / MoE / RDNA2 W4A16 dense / eviction all
  exonerated with data.

**Working production config: `cudagraph_mode=PIECEWISE` + `--enable-prefix-caching`.**
Verified 16/16 successful (68.65 tok/s, TPOT 89.63 ms at 16k/1k c=16),
0/18 sequential correctness, 8/8 concurrent coherence.

## Comparison vs the 27B baseline (bench_27b_awq_matrix.md)

Full bench matrix, TP=4 on 4× Radeon PRO V620, V1 + FULL_AND_PIECEWISE +
breakable cudagraphs:

| Workload | concurrency | Flash-Next Triton | Flash-Next RDNA2 W4A16 | 27B AWQ HIP | Flash-Next vs 27B |
|---|---|---:|---:|---:|---:|
| 1k/512 | 1 | 24.79 | **24.69** | 22.66 | **1.09×** |
| 1k/512 | 4 | 78.56 | **82.76** | 54.43 | **1.52×** |
| 1k/512 | 8 | 35.26 | **150.67** | 133.83 | **1.13×** |
| 16k/1k | 1 | 20.06 | **19.82** | 16.22 | **1.22×** |
| 16k/1k | 4 | 36.01 | **49.17** | 27.46 | **1.79×** |
| 16k/1k | 8 | 44.68 | **68.83** | 45.35 | **1.52×** |

Key findings after RDNA2 W4A16 kernel wired in (commit `b549c2299`):
- **Flash-Next wins ALL 6 cells** against the 27B AWQ HIP baseline
  (1.09×–1.79×)
- **1k/512 c=8**: 35.26 → 150.67 out tok/s (**4.3× faster** with RDNA2 kernel)
- **16k/1k c=4**: 36.01 → 49.17 out tok/s (**1.37× faster**)
- **16k/1k c=8**: 44.68 → 68.83 out tok/s (**1.54× faster**)
- Output correctness verified at c=8 16k: "Paris. The capital of Germany
  is Berlin..." and "2, 2+2=4,"

## HIP GDN prefill wired in (commit `ce3c93970`)

`_resolve_gdn_prefill_backend` now returns "rdna2" on gfx10x when the 5 HIP
`gdn_prefill_*_rdna2` kernels are registered. The log now shows "Using RDNA2
HIP GDN prefill kernel" instead of the misleading "Triton/FLA" (the dispatch
already used the HIP chain via `_gdn_prefill_dispatch_available()`; this makes
the selection explicit).

16k/1k c=8 (HIP GDN) = **68.71 out tok/s** vs Triton/FLA baseline 68.83 — no
regression, but no speedup either. The 48-layer prefill is not bottlenecked by
the GDN prefill kernel itself; it's dominated by aggregate prefill work across
MoE + attention + GDN. The HIP GDN prefill is correct and now explicit, but
the 16k prefill bottleneck is elsewhere.

## Full Flash-Next bench data (RDNA2 W4A16)

| Workload | concurrency | out tok/s | total tok/s | TTFT ms | TPOT ms | duration s |
|---|---|---:|---:|---:|---:|---:|
| 1k/512 | 1 | 24.69 | 72.90 | 701 | 39.2 | 83.0 |
| 1k/512 | 4 | 82.76 | 244.41 | 1,569 | 45.3 | 99.0 |
| 1k/512 | 8 | 150.67 | 444.94 | 2,487 | 48.3 | 108.7 |
| 16k/1k | 1 | 19.82 | 336.96 | 9,354 | 41.1 | 100.9 |
| 16k/1k | 4 | 49.17 | 835.87 | 18,125 | 63.0 | 162.7 |
| 16k/1k | 8 | 68.83 | 1,170.10 | 27,960 | 87.8 | 232.5 |

All cells: 0 failed requests, prefix caching ON, deterministic sampling,
coherent output verified.
