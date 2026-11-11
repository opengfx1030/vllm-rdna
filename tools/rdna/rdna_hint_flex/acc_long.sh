#!/usr/bin/env bash
# Held-out perplexity (8 x 2048 tokens after 800000, the tools/hi2 eval
# protocol) of each MODEL=path:extra-args arm, eager, one GPU:
#   GPU=3 bash tools/rdna/rdna_hint_flex/acc_long.sh tag=path[:args] ...
# MODE=f16|i8|i8_all sets VLLM_RDNA_HINT_FLEX_COMPUTE_MODE.
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
# shellcheck source=tools/rdna/rdna_hint_flex/env.sh
source "$TREE/tools/rdna/rdna_hint_flex/env.sh" "${GPU:?GPU}" > /dev/null
L=${OUT:-$HOME/w4a8_runs/hi2-support/acc_long}
T=${TEXT:-$HOME/w4a8_runs/hadamard-int2/study_text_all.txt}
mkdir -p "$L"
cd "$L" || exit 1
for arm in "$@"; do
  tag=${arm%%=*}
  rest=${arm#*=}
  model=${rest%%:*}
  extra=${rest#*:}
  [[ $extra == "$rest" ]] && extra=""
  # shellcheck disable=SC2086
  VLLM_RDNA_HINT_FLEX_COMPUTE_MODE=${MODE:-f16} timeout 3000 "$PY" \
    "$TREE/tools/rdna/rdna_hint_flex/e2e_check.py" --model "$model" \
    --out "$L/$tag.json" --eager --gpu-mem 0.7 --text "$T" \
    --skip-tokens 800000 --samples 8 --seq-len 2048 $extra > "$L/$tag.log" 2>&1
  echo "$tag rc=$? $(grep teacher "$L/$tag.log")"
done
