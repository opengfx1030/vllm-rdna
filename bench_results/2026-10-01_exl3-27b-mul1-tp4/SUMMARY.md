# EXL3 27B mul1 — TP=4, full HIP stack (2026-10-01)

**Model**: `Qwen3.8-27B-exl3-3.00bpw` (`turboderp`), codebook `mul1`, bits 3.0, head_bits 6, out_scales always.
**Box**: `par1-cs25`, venv-7.14.0_0.28.0, tree `/home/chenco_adm/vllm-rdna-0.28.0`, GPUs 4-7 (4x Radeon PRO V620, gfx1030).
**Fork**: `vllm-rdna-0.28.0`, branch `rdna_extras` (EXL3 loader backport `a50520a3d`/`aaa374ad4`/etc.).

## Serve config (Rung 1 = eager, Rung 2 = FULL_AND_PIECEWISE)

- `--tensor-parallel-size 4 --attention-backend RDNA_ATTN` + `VLLM_USE_RDNA2_FA=1` (FA-RDNA2, never Triton)
- `VLLM_ROCM_USE_AITER=0 VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0 VLLM_USE_V2_MODEL_RUNNER=1 GPU_MAX_HW_QUEUES=2`
- `VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_AOT_COMPILE=0` (27B TP=4 AOT abort without these)
- `VLLM_EXL3_DEBUG=1 --dtype float16 --language-model-only --skip-mm-profiling`
- `--gpu-memory-utilization 0.90 --kv-cache-memory-bytes 6000000000` (KV cap 6e9)
- `--enable-prefix-caching --mamba-cache-mode align` (hybrid GDN model requires align)
- TunableOp via `tools/rdna2_028/tunableop_env.sh` (lookup-only)
- RCCL `NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0` (TP=4 allreduce safety)

Model loaded at 7.94 GiB (eager) / 8.57 GiB (F&P) per worker. KV cache 289,823 tokens (14x concurrency at 20,480).

## Rung 1 (eager) — coherence PASS

Probes via `/v1/chat/completions`, temperature 0:
- "The capital of France is" → `...Paris` ✓
- "2 + 2 =" → `...4` ✓
- "1+1=" → `...2` ✓
- "Once upon a time" → coherent story opening ✓
- "Write one sentence about the moon." → coherent sentence ✓

Note: the model is a Qwen3.8 reasoning model; it emits a short reasoning preamble before the final answer (`reasoning: null` means no parser extracts it). All final answers are correct.

## Rung 2 (FULL_AND_PIECEWISE) — coherence PASS

Same probes all correct. Graph capture: FULL cudagraphs for sizes [1,2,4,8,16], took ~1.9 GiB, 9s. init engine 78.76s (compilation 63.29s).

## Performance table (FULL_AND_PIECEWISE, TP=4)

`vllm bench serve` (`/v1/completions`, random dataset, ignore-eos, seed 12345). Decode per-req = 1000/TPOT. Prefill derived from TTFT at c=1.

| Cell | c | Prefill tok/s | Decode per-req (tok/s) | Decode agg (tok/s) | TTFT (s) | ITL median (ms) | TPOT (ms) |
|------|---|--------------|------------------------|--------------------|----------|-----------------|-----------|
| 1k/512 | 1 | 908 | 19.7 | 18.91 | 1.13 | 50.77 | 50.77 |
| 1k/512 | 8 | (prefill-dominated) | 11.6 | 81.77 | 5.88 | 81.23 | 86.26 |
| 16k/1k | 1 | 830 | 18.6 | 13.68 | 19.75 | 53.92 | 53.89 |
| 16k/1k | 8 | (prefill-dominated) | 5.9 | 30.58 | 92.27 | 104.15 | 169.52 |

Observations: decode per-request ~19-20 tok/s at c=1; aggregate output peaks at 81.77 tok/s (1k/512 c=8). The 16k cells are prefill-dominated — TTFT balloons to 92s at c=8 (8x16k input queued), and per-request decode collapses to 5.9 tok/s because GPU time is split between prefill and decode. This matches the known RDNA batching economics (prefill is the lever, not decode).

## Stall diagnosis + fix

1. **Prior TP=4 `expanded size (2560) vs (10240)`** (pre-backport log): resolved by the loader rewrite — the current loader TP-slices correctly (no longer reproduces).
2. **F&P `gemma_rms_norm` fake-kernel stride mismatch** — the real blocker. Under FULL_AND_PIECEWISE, inductor's piecewise compile asserted `torch.ops.vllm.gemma_rms_norm` output stride `(256, 524288, 1)`, but the real op (`ir.ops.rms_norm`) preserves the (contiguous at runtime) input stride `(1536, 256, 1)` for the `(2048,6,256)` GDN rms_norm. Root cause: `torch.empty_like(x)` on a non-contiguous **traced** fake returns a transposed stride, so inductor asserts a stride the runtime output lacks.
   - **Fix** (`vllm/model_executor/layers/layernorm.py`): `gemma_rms_norm_fake` and `gemma_fused_add_rms_norm_fake` now return contiguous `torch.empty(x.size(), ...)` instead of `torch.empty_like(x)`, matching the real op's runtime output.
   - Also required: clear `TORCHINDUCTOR_CACHE_DIR` (`cache/inductor`) — `VLLM_DISABLE_COMPILE_CACHE=1` does **not** cover it, so the stale compiled artifact (with the old stride assert) was being reused.
3. **APIServer orphan**: `pkill -f "vllm-rdna-0.28.0"` misses the API server (cmdline is `python -m vllm.entrypoints.openai.api_server`, no vllm path). The EngineCore workers are `VLLM::EngineCore`/`VLLM::Worker_TP*`. Kill those explicitly.

## Files

- `exl3_27b_tp4_serve.sh` — serve launcher (eager/F&P switch)
- `exl3_probe_coherence.py` — coherence probe
- `exl3_27b_eager.log` — Rung 1 serve log
- `exl3_27b_fp.log` — Rung 2 serve log
- `cell_16k1k_c8.log` — 16k/1k c=8 bench result
