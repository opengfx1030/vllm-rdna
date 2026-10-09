#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# A/B the mixed-batch stall probe across server-knob arms, one boot per arm.
#
#   RECIPE=flashnext-mtp0 MODEL=/path GPUS=6,7,8,9 TAG=f-fn \
#     ARMS="base PREFILL_INTERVAL=4 LPTH=256 PREFILL_INTERVAL=4,LPTH=256" \
#     LOCK=~/w4a8_runs/port-v031/GPU69_LOCK \
#     STALL_ARGS="--decoders 6 --prefill-lens 4096,16384" \
#     bash tools/rdna/port_v031/stall_ab.sh
#
# ARMS: space-separated; each arm is "base" or comma-separated recipe
# overrides (KEY=value). Every arm runs serve_validate.sh with STALL_PROBE=1,
# CELLS=none, PPL_PROBE=0, PREFIX_PROBE=0 (greedy probes stay on) and writes
# ~/w4a8_runs/port-v031/serve-$TAG-<arm>/. All arms share one compile cache
# (CACHE, default cache-$TAG) so only the first boot is cold. Other env
# (SERVE_TREE, SERVE_SCRIPT, VENV, PORT, ...) passes through to
# serve_validate.sh. LOCK, if set, is created with noclobber (exit 3 when it
# already exists) and removed on exit. The combined summary goes to
# ~/w4a8_runs/port-v031/stall-ab-$TAG.txt.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${RECIPE:?RECIPE required}" "${MODEL:?MODEL required}" "${GPUS:?GPUS required}"
TAG=${TAG:-stall-$RECIPE}
ARMS=${ARMS:-base}
ROOT=${ROOT:-$HOME/w4a8_runs/port-v031}
export CACHE=${CACHE:-$ROOT/cache-$TAG}
COMBINED=$ROOT/stall-ab-$TAG.txt

if [[ -n ${LOCK:-} ]]; then
    if ! (set -o noclobber; echo "$$ stall_ab $TAG $(date -Is)" > "$LOCK") 2>/dev/null; then
        echo "lock $LOCK held: $(cat "$LOCK" 2>/dev/null)" >&2
        exit 3
    fi
    trap 'rm -f "$LOCK"' EXIT
fi

: > "$COMBINED"
for arm in $ARMS; do
    name=${arm//=/}
    name=${name//,/-}
    overrides=()
    [[ $arm != base ]] && IFS=, read -r -a overrides <<< "$arm"
    arm_tag=$TAG-$name
    arm_out=$ROOT/serve-$arm_tag
    echo "[$(date +%H:%M:%S)] arm $arm -> $arm_out" | tee -a "$COMBINED"
    STALL_PROBE=1 CELLS=none PPL_PROBE=${PPL_PROBE:-0} PREFIX_PROBE=${PREFIX_PROBE:-0} \
        TAG="$arm_tag" OUT="$arm_out" bash "$HERE/serve_validate.sh" "${overrides[@]}"
    grep -E "READY|BOOT FAILED|PROBE|STALL|SERVER DIED|fault markers" \
        "$arm_out/summary.txt" | sed 's/^/  /' | tee -a "$COMBINED"
    sed 's/^/  /' "$arm_out/stall/summary.txt" 2>/dev/null | tee -a "$COMBINED"
done
