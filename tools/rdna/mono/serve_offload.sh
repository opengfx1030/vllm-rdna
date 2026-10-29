#!/usr/bin/env bash
# TP2 MonoKernel expert cache, using the standard recipe and native KV tier.
# PRINT=1 bash tools/rdna/mono/serve_offload.sh prints the resolved command.
# VALIDATE=1 MTP=0 EAGER=1 CELLS=none runs the existing qualification driver.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
export GPUS=${GPUS:-6,7}
case "$GPUS" in
    6,7|7,6) ;;
    *) echo 'This work allocation permits only HIP GPUs 6 and 7.' >&2; exit 2 ;;
esac
export MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
export PLE_QUANT_DIR=${PLE_QUANT_DIR:-$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
export VENV=${VENV:-$TREE/.venv}
export PATH=$VENV/bin:$PATH
export RECIPE=${RECIPE:-flashnext-mtp2}
export PORT=${PORT:-18766} TAG=${TAG:-moe-offload-tp2}
SLOTS=${SLOTS:-96}
POLICY=${POLICY:-lfu_lru}
DECAY=${DECAY:-128}
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-16}
KV_LIVE_OFFLOAD=${KV_LIVE_OFFLOAD:-0}
KV_LIVE_WATERMARK=${KV_LIVE_WATERMARK:-0.9}
KV_LIVE_PAUSE_TARGET_S=${KV_LIVE_PAUSE_TARGET_S:-10}
[[ $KV_LIVE_OFFLOAD == 0 || $KV_LIVE_OFFLOAD == 1 ]] || exit 2
[[ $KV_LIVE_WATERMARK =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || exit 2
[[ $KV_LIVE_PAUSE_TARGET_S =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
LIVE_CFG=
if [[ $KV_LIVE_OFFLOAD == 1 ]]; then
    [[ ! $KV_OFFLOAD_GB =~ ^0+([.]0+)?$ ]] || exit 2
    LIVE_CFG=",\"kv_cache_live_offload\":true,\"kv_cache_live_offload_watermark\":$KV_LIVE_WATERMARK,\"kv_cache_live_offload_pause_target_s\":$KV_LIVE_PAUSE_TARGET_S"
fi
[[ $SLOTS =~ ^[0-9]+$ && $SLOTS -ge 1 && $SLOTS -le 256 ]] || exit 2
[[ $DECAY =~ ^[0-9]+$ && $DECAY -ge 1 ]] || exit 2
[[ $POLICY == lfu_lru || $POLICY == lru ]] || exit 2
[[ $KV_OFFLOAD_GB =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
PLAN_CFG=
if [[ -n ${SLOT_PLAN:-} ]]; then
    # -S avoids loading the runtime's HIP-initializing sitecustomize.
    PLAN_CFG=$("$VENV/bin/python" -S -c '
import json, sys
with open(sys.argv[1]) as source:
    plan = json.load(source)["layer_slots"]
if not isinstance(plan, dict) or not plan:
    raise ValueError("SLOT_PLAN requires nonempty layer_slots")
print(",\"expert_cache_slot_plan\":" + json.dumps(plan, separators=(",", ":")))
' "$SLOT_PLAN")
fi
KNOBS="{\"rocm_mono_decode\":{\"expert_cache_slots\":$SLOTS,\"expert_cache_policy\":\"$POLICY\",\"expert_cache_decay\":$DECAY$PLAN_CFG$LIVE_CFG${MONO_CFG:+,$MONO_CFG}}}"
export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--monokernel --additional-config $KNOBS"
if [[ ! $KV_OFFLOAD_GB =~ ^0+([.]0+)?$ ]]; then
    export EXTRA_ARGS="$EXTRA_ARGS --kv-offloading-size $KV_OFFLOAD_GB --kv-offloading-backend native"
fi
export PYTHONPATH=$TREE
# Avoid split-K FP16 atomic rounding varying with cache/replay scheduling.
# This opt-in is scoped to the isolated offloading launch, not resident recipes.
export VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=${VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE:-1}
export HIP_VISIBLE_DEVICES=$GPUS
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [[ ${VALIDATE:-0} == 1 && ${PRINT:-0} != 1 ]]; then
    exec bash "$TREE/tools/rdna/port_v031/serve_validate.sh" \
        TP=2 PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" \
        GMEM="${GMEM:-0.92}" MAXLEN="${MAXLEN:-32768}" \
        MAXBAT="${MAXBAT:-4096}" "$@"
fi
exec bash "$TREE/tools/rdna/serve_rdna.sh" RECIPE="$RECIPE" \
    MODEL="$MODEL" VENV="$VENV" VLLM_TREE="$TREE" \
    TP=2 PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" \
    GMEM="${GMEM:-0.92}" PORT="$PORT" MAXLEN="${MAXLEN:-32768}" \
    MAXBAT="${MAXBAT:-4096}" "$@"
