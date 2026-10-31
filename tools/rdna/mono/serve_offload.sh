#!/usr/bin/env bash
# Expert-cache serve for TP1/2/4 expert-parallel topologies, using the
# standard recipe and the native KV tier.
#
# The maintainer host pins this work to HIP GPUs 6,7; that allocation-safety
# restriction is preserved as an ALLOWED_GPU_SETS allowlist (default "6,7").
# Portability is explicit, not implicit: a portable host opts in with, e.g.,
#   GPUS=0,1,2,3 TP=4 ALLOWED_GPU_SETS="0,1,2,3" SLOTS=48 \
#     PRINT=1 bash tools/rdna/mono/serve_offload.sh
# PRINT=1 prints the resolved plan without launching.
# VALIDATE=1 MTP=0 EAGER=1 CELLS=none runs the existing qualification driver.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
export MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
export PLE_QUANT_DIR=${PLE_QUANT_DIR:-$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
export VENV=${VENV:-$TREE/.venv}
export PATH=$VENV/bin:$PATH
export RECIPE=${RECIPE:-flashnext-mtp2}
export PORT=${PORT:-18766} TAG=${TAG:-moe-offload-tp2}

# --- topology: TP x device set, validated together -------------------------
# Qualified expert-parallel geometries (K2 kernel stage capacities):
#   TP2 -> 256 local experts/rank, TP4 -> 128 local experts/rank.
TP=${TP:-2}
GLOBAL_EXPERTS=${GLOBAL_EXPERTS:-512}
case "$TP" in 1|2|4) ;; *) echo "TP must be 1, 2 or 4 (K2-qualified expert counts)." >&2; exit 2 ;; esac
(( GLOBAL_EXPERTS % TP == 0 )) || exit 2
LOCAL_EXPERTS=$(( GLOBAL_EXPERTS / TP ))
case "$LOCAL_EXPERTS" in 128|256) ;; *) echo "TP=$TP gives $LOCAL_EXPERTS local experts/rank; only 128 or 256 are qualified." >&2; exit 2 ;; esac

# Allocation-safety allowlist: the GPUS set must appear verbatim (same order)
# in ALLOWED_GPU_SETS. Default keeps the maintainer host's 6,7-only policy.
ALLOWED_GPU_SETS=${ALLOWED_GPU_SETS:-6,7}
# Exported so delegated drivers (port_v031/serve_validate.sh hard-requires
# GPUS) inherit the resolved set even when it comes from the default.
export GPUS=${GPUS:-6,7}
# Canonical HIP IDs: unique, non-negative integers.
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPUS must be a comma-separated list of HIP indices." >&2; exit 2; }
IFS=',' read -r -a _gpu_arr <<< "$GPUS"
declare -A _seen=()
for g in "${_gpu_arr[@]}"; do
    [[ -z "${_seen[$g]:-}" ]] || { echo "GPUS lists device $g twice." >&2; exit 2; }
    _seen[$g]=1
done
if [[ " $ALLOWED_GPU_SETS " != *" $GPUS "* ]]; then
    echo "GPUS=$GPUS is not in ALLOWED_GPU_SETS=[$ALLOWED_GPU_SETS]; this allocation refuses it." >&2
    exit 2
fi
if (( ${#_gpu_arr[@]} != TP )); then
    echo "GPUS=$GPUS has ${#_gpu_arr[@]} devices but TP=$TP; rank count must equal device count." >&2
    exit 2
fi

# --- expert cache knobs ------------------------------------------------------
# Slots are per rank and may not exceed the rank's local expert count.
SLOTS=${SLOTS:-$(( TP == 2 ? 96 : 48 ))}
POLICY=${POLICY:-lfu_lru}
DECAY=${DECAY:-128}
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-16}
[[ $SLOTS =~ ^[0-9]+$ && $SLOTS -ge 1 && $SLOTS -le $LOCAL_EXPERTS ]] || {
    echo "SLOTS=$SLOTS must be in [1, $LOCAL_EXPERTS] for TP=$TP." >&2; exit 2; }
[[ $DECAY =~ ^[0-9]+$ && $DECAY -ge 1 ]] || exit 2
[[ $POLICY == lfu_lru || $POLICY == lru ]] || exit 2
[[ $KV_OFFLOAD_GB =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2

KNOBS="{\"rocm_mono_decode\":{\"expert_cache_slots\":$SLOTS,\"expert_cache_policy\":\"$POLICY\",\"expert_cache_decay\":$DECAY${MONO_CFG:+,$MONO_CFG}}}"
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
if [[ ${PRINT:-0} == 1 ]]; then
    printf 'OFFLOAD-PLAN tp=%s devices=%s local_experts=%s slots=%s validate=%s\n' \
        "$TP" "$GPUS" "$LOCAL_EXPERTS" "$SLOTS" "${VALIDATE:-0}"
    printf 'OFFLOAD-ENV HIP_VISIBLE_DEVICES=%s deterministic_dense=%s\n' \
        "$HIP_VISIBLE_DEVICES" "$VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE"
    printf 'OFFLOAD-EXTRA %s\n' "$EXTRA_ARGS"
    exit 0
fi
if [[ ${VALIDATE:-0} == 1 ]]; then
    exec bash "$TREE/tools/rdna/port_v031/serve_validate.sh" \
        TP="$TP" PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" \
        GMEM="${GMEM:-0.92}" MAXLEN="${MAXLEN:-32768}" \
        MAXBAT="${MAXBAT:-4096}" "$@"
fi
exec bash "$TREE/tools/rdna/serve_rdna.sh" RECIPE="$RECIPE" \
    MODEL="$MODEL" VENV="$VENV" VLLM_TREE="$TREE" \
    TP="$TP" PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" \
    GMEM="${GMEM:-0.92}" PORT="$PORT" MAXLEN="${MAXLEN:-32768}" \
    MAXBAT="${MAXBAT:-4096}" "$@"
