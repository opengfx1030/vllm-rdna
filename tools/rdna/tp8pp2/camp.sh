#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# TunableOp campaign for the TP8 / TP4xPP2 GEMM shapes on slots A+B.
#
#   export RUN_DIR=~/w4a8_runs/tp8pp2 BASE_TREE=<tree with the shipped tunableop/ rows>
#   CENSUS_GLOB='$RUN_DIR/arms/tp8-census/untuned*.csv ...' bash camp.sh census
#   bash camp.sh tune      # one shard per GPU on slots A+B (both slots idle and leased)
#   bash camp.sh check     # numerics gate (rowcheck.py) on the tuned rows -> $W/reject.txt
#   bash camp.sh harvest   # copy of the shipped rows + ranks 4..7 + shard rows -> $W/rows
#   bash camp.sh hit [ROWS_ROOT]   # per-ordinal lookup-hit proof of the census (0..7)
#
# TREE_TOOLS (default: the checkout holding this script) provides
# tune_missing_gemms.sh and tunableop_missing.py (extend, REFUSED_GPUS); BASE_TREE
# provides the shipped rows (tunableop/). Rows are never written into BASE_TREE.
set -uo pipefail
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
W=$R/tunable
TREE_TOOLS=${TREE_TOOLS:-$(cd "$H/../../.." && pwd)}
BASE_TREE=${BASE_TREE:?BASE_TREE=<tree with the shipped tunableop/ rows>}
PROFILE=${PROFILE:-rocm7.14-rocblas5.5}
MISSING=$TREE_TOOLS/tools/rdna/prof/tunableop_missing.py
TUNE=$TREE_TOOLS/tools/rdna/prof/tune_missing_gemms.sh
mkdir -p "$W"
phase=${1:?census|tune|check|harvest|hit}
case $phase in
    census)
        # shellcheck disable=SC2086
        cat ${CENSUS_GLOB:-$R/arms/tp8-census/untuned*.csv} | grep '^Gemm' | sort -u > "$W/census.csv"
        echo "census: $(wc -l < "$W/census.csv") unique keys"
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
            python3 "$MISSING" missing --census "$W/census.csv" --profile "$PROFILE" \
            --rows-dir "$BASE_TREE/tunableop" --dry-run 2>&1 | grep -E '^(census|table|missing)'
        ;;
    tune)
        verify_slot AB || exit 1
        IFS=, read -r -a gpus <<< "$(slot_hip AB)"
        rm -rf "$W"/shard*
        pids=()
        for i in "${!gpus[@]}"; do
            REFUSED_GPUS=0,5 TUNABLEOP_DIR=$BASE_TREE/tunableop \
                bash "$TUNE" tune --census "$W/census.csv" --gpu "${gpus[$i]}" \
                --shard "$i/${#gpus[@]}" --out "$W/shard$i" --profile "$PROFILE" \
                > "$W/shard$i.log" 2>&1 &
            pids+=($!)
        done
        rc=0
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        for i in "${!gpus[@]}"; do
            echo "shard $i (HIP ${gpus[$i]}): $(grep -c '^Gemm' "$W/shard$i/tunableop_results0.csv" 2> /dev/null) rows; $(tail -1 "$W/shard$i.log" | cut -c1-120)"
        done
        exit $rc
        ;;
    check)
        # rowcheck.py on the freshly tuned rows (one V620 of slot A), two processes: the
        # rocBLAS default (TunableOp never enabled) and the rows. The keys it rejects land
        # in $W/reject.txt, which `harvest` then leaves out
        verify_slot A || exit 1
        rm -rf "$W/checkview" "$W/reject.txt" "$W"/rowcheck-*.json
        mkdir -p "$W/checkview"
        {
            grep -h '^Validator' "$W/shard0/tunableop_results0.csv"
            grep -h '^Gemm' "$W"/shard*/tunableop_results0.csv
        } > "$W/checkview/tunableop_results0.csv"
        grep -h '^Gemm' "$W"/shard*/tunableop_results0.csv | cut -d, -f1,2 > "$W/checkview/keys.csv"
        IFS=, read -r -a gpus <<< "$(slot_hip A)"
        HIP_VISIBLE_DEVICES=${gpus[0]} "$VENV/bin/python" "$H/rowcheck.py" run default \
            "$W/census.csv" "$W/checkview/keys.csv" --out "$W/rowcheck-default.json" \
            > "$W/rowcheck.txt" 2>&1 || exit 1
        HIP_VISIBLE_DEVICES=${gpus[0]} PYTORCH_TUNABLEOP_FILENAME=$W/checkview/tunableop_results.csv \
            "$VENV/bin/python" "$H/rowcheck.py" run tuned "$W/census.csv" \
            "$W/checkview/keys.csv" --out "$W/rowcheck-tuned.json" \
            >> "$W/rowcheck.txt" 2>&1 || exit 1
        "$VENV/bin/python" "$H/rowcheck.py" compare "$W/rowcheck-default.json" \
            "$W/rowcheck-tuned.json" --reject-out "$W/reject.txt" >> "$W/rowcheck.txt" 2>&1
        rc=$?
        tail -3 "$W/rowcheck.txt"
        exit $rc
        ;;
    harvest)
        rm -rf "$W/rows"
        mkdir -p "$W/rows"
        cp -a "$BASE_TREE/tunableop/." "$W/rows/"
        # shellcheck disable=SC2016
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= bash -c '
            set -e
            python3 "$0" extend --ranks 8 --profile "$1" --rows-dir "$2"
            args=()
            for f in "$3"/shard*/tunableop_results0.csv; do args+=(--tuned "$f"); done
            if [[ -s $3/reject.txt ]]; then args+=(--reject "$3/reject.txt"); fi
            python3 "$0" harvest "${args[@]}" --profile "$1" --rows-dir "$2"
            python3 "$0" count --profile "$1" --rows-dir "$2"' "$MISSING" "$PROFILE" "$W/rows" "$W"
        ;;
    hit)
        verify_slot AB || exit 1
        ROOT=${2:-$W/rows}
        HIP_VISIBLE_DEVICES=$(slot_hip AB)
        export HIP_VISIBLE_DEVICES
        export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
        export TORCH_BLAS_PREFER_HIPBLASLT=0
        export PYTORCH_TUNABLEOP_FILENAME=$ROOT/$PROFILE/tunableop_results.csv
        unset PYTORCH_TUNABLEOP_RECORD_UNTUNED PYTORCH_TUNABLEOP_UNTUNED_FILENAME
        rc=0
        for d in 0 1 2 3 4 5 6 7; do
            "$VENV/bin/python" "$H/tunehit.py" "$d" "$W/census.csv" 2>&1 | grep -E '^ordinal|MISSING' || rc=1
        done
        exit $rc
        ;;
    *)
        echo "unknown phase $phase" >&2
        exit 2
        ;;
esac
