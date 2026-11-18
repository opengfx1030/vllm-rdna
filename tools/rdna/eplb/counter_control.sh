#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Two diagnostic boots: counters/map mode/mailbox on, no planner or moves.
# Take the b4 lease first. These C1/C2 results never enter the R2 ABBA gate.
# VENV=/venv MODEL=/model OUT=/fresh/controls [DRY_RUN=1] bash this-script
# Per-arm exports intentionally stay inside their subshell.
# shellcheck disable=SC2030,SC2031
set -euo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${VENV:?VENV required}" "${MODEL:?MODEL required}" "${OUT:?OUT required}"
PY=${QUALIFY_PYTHON:-/usr/bin/python3}
CELLS="1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"
DRY_RUN=${DRY_RUN:-0}
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || exit 2
[[ -f $TREE/.sha ]] || { echo "counter control needs a committed snapshot .sha" >&2; exit 2; }
if [[ -e $OUT ]]; then
    [[ -d $OUT && -z $(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || {
        echo "counter control OUT must be fresh" >&2; exit 2;
    }
fi
mkdir -p "$OUT/logs" "$OUT/cache/triton" "$OUT/cache/inductor"
OUT=$(cd "$OUT" && pwd)
{
    echo "diagnostic only; not an ABBA qualification or a replacement for failed R2"
    echo "head_sha=$(cat "$TREE/.sha")"
    echo "C1 C2: counters/map mode/mailbox on; CHAOS=1 DECISION_STEPS=1000000000"
    echo "no planner, no expert moves; all other recipe defaults unchanged"
    echo "cells=$CELLS"
} > "$OUT/plan.txt"
for arm in C1 C2; do
    [[ ! -e $OUT/ABORT ]] || exit 1
    (
        for name in ${!VLLM_RDNA_EPLB@}; do unset "$name"; done
        unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH EAGER
        unset QUALITY_WINDOWS QUALITY_LIMIT QUALITY_TOP QUALITY_PASSAGES
        unset LOAD_HOOK STALL_ARGS STALL_ITER_LOG
        export VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=1
        export VLLM_RDNA_EPLB_DECISION_STEPS=1000000000
        export SERVE_TREE=$TREE RECIPE=flashnext-mtp2 GPUS=2,3,4,5
        export REFUSED_GPUS=0,1 PORT=18248 TAG="counter-control-$arm"
        export CELLS BENCH_EXTRA='--temperature 0' CONC_PROBE=1
        export STALL_PROBE=0 RESET_PREFIX_CACHE=0 READY_TIMEOUT=2400
        export EXTRA_ARGS=--monokernel QUIET_HOLD=1 QUIET_OWN=EplbCounter
        export QUIET_WIN_MAX=240 STOP_GRACE=120
        export QUIET_DIR="$HOME/w4a8_runs/lanes/quiet-b4"
        export BOOT_LOCK="$HOME/w4a8_runs/boot.lock"
        export TRITON_CACHE_DIR=$OUT/cache/triton
        export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor
        export QUALIFY_PYTHON=$PY QUALIFY_CAMPAIGN_OUT=$OUT
        export QUALIFY_ABORT=$OUT/ABORT OUT=$OUT/boots/$arm
        if ((DRY_RUN)); then
            HIP_VISIBLE_DEVICES='' PRINT=1 bash "$TREE/tools/rdna/serve_rdna.sh" \
                RECIPE="$RECIPE" MODEL="$MODEL" VENV="$VENV" \
                VLLM_TREE="$TREE" PORT="$PORT" HOST=127.0.0.1 \
                VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=1 \
                VLLM_RDNA_EPLB_DECISION_STEPS=1000000000 EXTRA_ARGS=--monokernel
        else
            HIP_VISIBLE_DEVICES=2,3,4,5 QUALIFY_WRAPPER=1 \
                bash "$TREE/tools/rdna/eplb/qualify_final.sh" \
                VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=1 \
                VLLM_RDNA_EPLB_DECISION_STEPS=1000000000 EXTRA_ARGS=--monokernel
        fi
    ) > "$OUT/logs/$arm.log" 2>&1
    if ((DRY_RUN == 0)); then
        if ! env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES= PYTHONPATH="$TREE" "$PY" \
            "$TREE/tools/rdna/eplb/qualify_boot.py" boot --dir "$OUT/boots/$arm" \
            --arm "$arm" --cells "$CELLS" --head-sha "$(cat "$TREE/.sha")"; then
            printf '%s: counter-only diagnostic evidence failed; no retry\n' "$arm" > "$OUT/ABORT"
            exit 1
        fi
    fi
done
cat "$OUT/plan.txt"
