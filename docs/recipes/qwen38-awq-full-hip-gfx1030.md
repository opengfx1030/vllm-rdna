# Production: Qwen3.8-27B-AWQ-INT4 FULL HIP graphs on gfx1030

**Status**: greedy PASS 3/3 (2026-09-09, FPP18/FPP19).  
**Branch**: `rdna_extras` (`9f0fcbe9e` and later).  
**Do not use `--enforce-eager`.** Do not enable `VLLM_RDNA_AR`.

Decode runs a skip_compiled HIP CUDA graph of FA-RDNA2 + RDNA2 W4A16 + HIP
KV write + HIP GDN. Prefill/mixed still use piecewise graphs at capture
sizes `[1,2,4,8]` (larger prefills are eager until piecewise capture
sizes grow).

## Why this replaced FPP13

FPP13 dispatched FULL but *executed* piecewise graphs so greedy stayed
coherent. That was a stand-in, not production FULL.

TRUE FULL with **custom all-reduce inside the HIP graph** garbles tokens
after the first (FPP17: `Paris` then `duct`). The same FULL graph with
**PYNCCL** is correct (FPP18/FPP19). Code now auto-disables custom AR
when TRUE FULL is on, even if `VLLM_FORCE_CUSTOM_ALL_REDUCE=1`.

Opt out of TRUE FULL (old FPP13 piecewise execute + custom AR) with
`VLLM_ROCM_TRUE_FULL=0`.

## Correctness gate

```bash
python tools/probe_greedy_correctness.py \
  --url http://127.0.0.1:PORT/v1/completions \
  --model "$MODEL"
```

PASS requires full-completion coherence, not first-token-only:

| Prompt | Must contain | Must not |
|---|---|---|
| `The capital of France is` | `Paris` in first 32 chars | `duct`, `\ufffd`, bang storms |
| `1+1=` | starts with `2` | same |
| `The capital of Germany is` | `Berlin` in first 40 chars | same |

## Stack

| Piece | Implementation |
|---|---|
| Self-attention | FA-RDNA2 (`RDNA_ATTN`, `VLLM_USE_RDNA2_FA=1`) |
| AWQ W4A16 | `RDNA2W4A16LinearKernel` |
| KV write (hybrid `block_size=784`) | `_rocm_C.reshape_and_cache_flash_rdna2` |
| GDN decode | `_rocm_C.gdn_decode_rdna2` |
| Decode CUDA graph | skip_compiled FULL HIP graph (`VLLM_ROCM_TRUE_FULL` default on) |
| TP all-reduce | **PYNCCL** (PIX + Simple). Custom AR is gated off. |
| Prefill graphs | piecewise, capture `[1,2,4,8]` |

## Env + CLI

See `scripts/serve_gfx1030_full.sh`. Required:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RDNA2_FA=1
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export VLLM_BATCH_INVARIANT=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=pxb
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export VLLM_FORCE_CUSTOM_ALL_REDUCE=1
# TRUE FULL is default-on. HIP custom AR copies into init-time IPC buffers.
unset VLLM_RDNA_AR
# VLLM_ROCM_TRUE_FULL=0  # only to restore FPP13 piecewise execute
```

`--max-model-len` is **200000** in production (Qwen3.5/3.8 hybrid context). **32768 is the floor** — do not ship 4096; that was FPP isolation only.

```bash
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --dtype float16 \
  --max-model-len 200000 \
  --max-num-seqs 16 \
  --kv-cache-memory-bytes 10000000000 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[],"max_cudagraph_capture_size":16,"cudagraph_capture_sizes":[1,2,4,8,16],"inductor_compile_config":{"combo_kernels":false}}' \
  --block-size 16 \
  --enable-prefix-caching \
  --language-model-only \
  --skip-mm-profiling \
  --trust-remote-code
```

Startup log must show:

- `Using RDNA2W4A16LinearKernel`
- `Overriding with RDNA_ATTN`
- `GDN decode using HIP gdn_decode_rdna2`
- `Captured FULL HIP cudagraph ... (skip_compiled FA+W4A16+GDN)`
- `Custom allreduce force-enabled by VLLM_FORCE_CUSTOM_ALL_REDUCE`
- `Using ['CUSTOM', 'PYNCCL'] all-reduce backends`

## Hardware / venv

Same as other gfx1030 recipes: `.176`, venv-7.14.0, PyTorch 2.12.0+rocm7.14.0,
HIP 7.14.60850, 4× Radeon PRO V620. `cd /tmp` before `python -m vllm...`.
