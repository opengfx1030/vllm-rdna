#!/usr/bin/env bash
# One actq_study.py job on one GPU:
#   GPU=2 TAG=q4b_fq64 MODEL=dir bash tools/rdna/rdna_hint_flex/actq_run.sh --mode fq --group 64
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
# shellcheck source=tools/rdna/rdna_hint_flex/env.sh
source "$TREE/tools/rdna/rdna_hint_flex/env.sh" "${GPU:?GPU}" > /dev/null
OUT=${OUT:-$HOME/w4a8_runs/hi2-support/actq}
mkdir -p "$OUT"
cd "$OUT" || exit 1
"$PY" "$TREE/tools/rdna/rdna_hint_flex/actq_study.py" --model "${MODEL:?MODEL}" \
  --out "$OUT/${TAG:?TAG}.json" "$@" > "$OUT/$TAG.log" 2>&1
echo "$TAG rc=$? $(tail -n 1 "$OUT/$TAG.log" | cut -c1-400)"
