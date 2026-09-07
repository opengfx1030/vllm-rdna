#!/bin/bash
# Fork recipe: btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit (qwen4_exp) on 4x
# Radeon PRO V620 (gfx1030). Measured 2026-09-06 TP4 cards 0-3 (150 W caps):
#   MTP=3 PIECEWISE: decode @3.7k 24.7 t/s, c1 79.9, c8 152.5
#   (eager MTP=3: 10.5 t/s; eager no-MTP: 6.1)
#   EP tested and rejected: --enable-expert-parallel loses on this PCIe
#   host (19.4/-21% decode, 68.8/-14% c1, 129.2/-15% c8, greedy-clean) —
#   matches the BM35 finding; keep experts TP-sharded.
#   Parameter sweep 2026-09-06 — this recipe is the optimum of:
#   - MTP k=2/k=4 lose (k=3: c1 79.9 vs 70.1/69.1; c8 152.5 vs 144.6/138.0;
#     k=4 decode-only +3.7% not worth it)
#   - 0.95 util + max-num-seqs 16 collapses c8 to 63.2 (-59%)
#   - max-num-batched-tokens 8192 is a wash (-3.8% c1, -8% TTFT)
#   - --linear-backend exllama is REQUIRED: auto-select picks
#     RDNA2TritonW4A16 which wants a 95 GiB repack buffer (OOM on 32 GB)
#
# Context length (2026-09-06): 64k (pool 205k tok, 3.1x), 128k (224k, 1.7x),
# and the native 262144 all work at full speed (decode 23.7-26.6 = baseline
# noise). 256k needs --gpu-memory-utilization 0.95. CAVEAT: do NOT combine
# small --max-num-seqs (<=2) with this model — dynamo 0/1-dim specialization
# on query_start_loc crashes with ConstraintViolationError.
#
# Multimodality (2026-09-06): works. Drop --language-model-only and
# --skip-mm-profiling (optionally add --limit-mm-per-prompt image=1);
# image captioning is correct, works with MTP=3 and at 17k+ token contexts.
# Quirk: in MM mode --served-model-name is not honored (use the full repo
# id in requests). Text-only quality is unchanged.
# Modes (env overrides):
#   MTP=N   speculative tokens (default 3; 0 disables)
#   CTX=N   max-model-len (default 262144, the native max, at 0.98 util;
#           32768/65536/131072 also validated. Do NOT combine with small
#           --max-num-seqs <=2 — dynamo query_start_loc crash.)
#   MM=0    disable vision (default: multimodal ON, unlimited images per
#           prompt). In MM mode --served-model-name is not honored; use
#           the full repo id in requests.
#   UTIL=N  override --gpu-memory-utilization (default 0.98)
MTP="${MTP:-3}"
CTX="${CTX:-262144}"
MM="${MM:-1}"
UTIL="${UTIL:-0.98}"
export VLLM_PLE_MMAP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export GPU_MAX_HW_QUEUES=4
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"

MM_ARGS="--language-model-only --skip-mm-profiling"
if [ "${MM}" != "0" ]; then
    MM_ARGS=""
fi

exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit \
    --dtype float16 \
    --kv-cache-dtype auto \
    --linear-backend exllama \
    --tensor-parallel-size 4 \
    --max-model-len ${CTX} \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    ${SPEC_ARG} \
    --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
    ${MM_ARGS} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization ${UTIL} \
    --served-model-name Qwen3.8-Flash-Next \
    "$@"
