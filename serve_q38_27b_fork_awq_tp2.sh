#!/bin/bash
# Fork TP2 variant of serve_q38_27b.sh (AWQ) for the 2026-09-05 fix race.
# Deltas vs vllm-upstream/serve_q38_27b.sh: --language-model-only +
# --skip-mm-profiling (VL encoder workspace OOM otherwise on this fork).
#
# MODE=single (default): int8_per_token_head KV + FD_RDNA2 flash-decode.
#   Decode @3.5k: 6.4 -> 22.3 t/s (the stock int8 batch>1 path is slow;
#   FD plugin only fast-paths q==1, so use MODE=multi for concurrency).
# MODE=multi: original fp16 KV, no FD plugin. c8: 48 t/s.
MODE="${MODE:-single}"
FD_ARGS=""
KV_ARGS="--kv-cache-dtype float16"
if [ "$MODE" = "single" ]; then
    export FD_RDNA2=1
    export FD_CHUNK=512
    export FD_TILE=32
    export FD_WARPS=8
    export FD_MAXQ=4
    KV_ARGS="--kv-cache-dtype int8_per_token_head"
fi
export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2,3}"
exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
    --dtype float16 \
    $KV_ARGS \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --skip-mm-profiling \
    --gpu-memory-utilization 0.98 \
    --served-model-name Qwen3.8-27B-AWQ \
    "$@"
