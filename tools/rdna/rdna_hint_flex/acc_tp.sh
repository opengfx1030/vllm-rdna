#!/usr/bin/env bash
# Held-out log-probs of one checkpoint with full cudagraphs at a given TP,
# for TP=N vs TP=1 comparisons (e2e_check.py --compare).
#   GPU=4,5 TP=2 TAG=q4b_f16_tp2_graph MODEL=/path \
#     bash tools/rdna/rdna_hint_flex/acc_tp.sh [extra e2e_check args]
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
# shellcheck source=tools/rdna/rdna_hint_flex/env.sh
source "$TREE/tools/rdna/rdna_hint_flex/env.sh" "${GPU:?GPU}" > /dev/null
export NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0 RCCL_P2P_NET_DISABLE=1
R=${ROOT:-$HOME/w4a8_runs/hi2-support}
export TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache_${TAG:?TAG}
export VLLM_CACHE_ROOT=$R/vllm_cache_$TAG
L=$R/acc_long
mkdir -p "$L"
cd "$R" || exit 1
timeout 2400 "$PY" "$TREE/tools/rdna/rdna_hint_flex/e2e_check.py" --model "${MODEL:?MODEL}" \
  --out "$L/$TAG.json" --gpu-mem 0.85 --tp "${TP:-1}" \
  --text "${TEXT:-$HOME/w4a8_runs/hadamard-int2/study_text_all.txt}" "$@" \
  > "$L/$TAG.log" 2>&1
echo "$TAG rc=$? $(grep teacher "$L/$TAG.log")"
