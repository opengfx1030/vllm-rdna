#!/bin/bash
# Fork TP1 variant of serve_q38_27b.sh (AWQ) for the 2026-09-05 serve race.
# Deltas vs vllm-upstream/serve_q38_27b.sh: TP1 single card, max-model-len
# 32768 (131k+ won't fit one 32GB card), --language-model-only +
# --skip-mm-profiling (VL encoder workspace OOM otherwise on this fork).
export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2}"
exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
    --dtype float16 \
    --kv-cache-dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --skip-mm-profiling \
    --gpu-memory-utilization 0.98 \
    --served-model-name Qwen3.8-27B-AWQ \
    "$@"
