#!/bin/bash
# Fork recipe: btbtyler09/Qwen3.8-27B-GPTQ-4bit on 2x gfx1030 (perf/parity).
#
# Two modes (2026-09-05 measurements, TP2 cards 2,3):
#   MTP=2 GDN=0 (default): interactive/multi-user — c1 87 t/s, c8 180 t/s
#   MTP=0 GDN=1: long-context single-stream — decode @3.5k 49 t/s
# (gdn_decode_rdna2 is off by default since it drops the MTP drafter to
# ~80ms/step; enable per-mode with GDN=1.)
MTP="${MTP:-2}"
GDN="${GDN:-0}"
[ "$MTP" = "2" ] && [ "$GDN" = "0" ] || true

export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export FD_RDNA2=1
export FD_CHUNK=512
export FD_TILE=32
export FD_WARPS=8
export FD_MAXQ=4
export AR_RDNA2="${AR_RDNA2:-0}"
export GPU_MAX_HW_QUEUES=4
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_GDN_DECODE_RDNA2="$GDN"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2,3}"

SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP}}"

exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve btbtyler09/Qwen3.8-27B-GPTQ-4bit \
    --dtype float16 \
    --kv-cache-dtype int8_per_token_head \
    --linear-backend exllama \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    ${SPEC_ARG} \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --language-model-only \
    --skip-mm-profiling \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --served-model-name Qwen3.8-27B-GPTQ \
    "$@"
