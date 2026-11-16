#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# TunableOp campaign for the Qwen3.8-27B HI2 TP=2 rocBLAS shapes (recipe 27b-hi2-tp2):
# shape census -> tune in short GPU holds -> numerics gate -> timing -> curated rows.
# Follows tools/rdna/tp8pp2/camp.sh (rowcheck gate, harvest) and
# docs/rdna2/hi2-27b-tunableop-2026-11-15.md.
#
#   export W=~/w4a8_runs/hi2-tunable/run MODEL=<27B HI2 checkpoint> VENV=<venv> GPU=<hip>
#   bash tunableop_hi2.sh shapes          # census files per group (CPU)
#   bash tunableop_hi2.sh record          # real-op keys of the census M values (GPU)
#   bash tunableop_hi2.sh tune GROUP      # one flock hold per shard (SHARDS, default 4)
#   bash tunableop_hi2.sh gather          # all shard rows -> $W/scratch/tunableop_results0.csv
#   bash tunableop_hi2.sh measure heur|new   # cold-weight graph timing JSON per condition
#   bash tunableop_hi2.sh check           # rowcheck.py gate -> $W/reject.txt
#   bash tunableop_hi2.sh curate          # keep rows >= ADOPT (3 %) faster -> $W/curated
#   bash tunableop_hi2.sh emul untuned|tuned|record   # TP=2 GEMMs emulated at TP=1 (tp2_emul_check.py)
#   bash tunableop_hi2.sh time heur|new   # real-op timing (dequant + GEMM, dense dispatch)
#   bash tunableop_hi2.sh hit             # every census key finds its row (lookup only)
#
# GPU phases run under `flock $LOCK` (default ~/w4a8_runs/hi2-gpu4.lock): one shard,
# one measurement or one check per hold, so other jobs on that GPU wait minutes at most.
# GPU must be a HIP index that resolves to a V620; REFUSED_GPUS (default "0 1 2 3 5")
# are never used. Tuning env: ENABLED=1 TUNING=1 HIPBLASLT off, MAX_TUNING_DURATION_MS=30.
# Rows land in $W only; the repo profile is built from $W/curated (see the doc).
set -euo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
H=$TREE/tools/rdna/rdna_hint_flex
W=${W:?W required (persistent, never /tmp)}
MODEL=${MODEL:?MODEL required (27B HI2 checkpoint)}
VENV=${VENV:?VENV required}
GPU=${GPU:?GPU required (HIP index)}
LOCK=${LOCK:-$HOME/w4a8_runs/hi2-gpu4.lock}
PROFILE=${PROFILE:-rocm7.14-rocblas5.5}
SHARDS=${SHARDS:-4}
ADOPT=${ADOPT:-0.03}
PREFILL_M=${PREFILL_M:-1024,2048}
SMALL_M=${SMALL_M:-9,12,15,18,21,24}
PL_M=${PL_M:-1024,2048}
SCHED_SLOTS=${SCHED_SLOTS:-gate_up,qkvz,ba,out,qkv}
SCHED0_M=${SCHED0_M:-1023,1029,1030,1031,2046}
SCHED1_M=${SCHED1_M:-1025,1026,1027,1028,2047}
SCHED2_M=${SCHED2_M:-1033,1036,1039,1042,1045}
PY=$VENV/bin/python
SP=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib/core/lib:$SP/_rocm_sdk_core/lib:$SP/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TREE${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=${PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS:-30}
export REFUSED_GPUS=${REFUSED_GPUS:-0 1 2 3 5}
mkdir -p "$W"

# group name -> `tunable_hi2.py shapes` arguments (the census M lists are in the doc):
#   prefill       the hi2 linears + the MTP fc at the chunk sizes (target and drafter pass)
#   sched0..2     SCHED_SLOTS (the slots whose 1024 / 2048 rows were adopted) at the M of
#                 the scheduler census: 1023 (last aligned chunk of a block-multiple
#                 prompt), 1025..1031 (a chunk plus 2..8 decode rows), 2046 / 2047 (two
#                 prompts in one step); sched2 = the MTP=2 mixed-step values 1033..1045
#   small         the MTP layer's dense GEMMs at the verify / draft M values
#   lmhead_small  the dense head at those M (only below the GEMV cap, see the doc)
#   lmhead_pl     the dense head at chunk sizes (prompt_logprobs passes)
#   heuronly      down / fc at the sched M: never tuned (no gain at 1024 / 2048), only
#                 timed so the expected-gain model has the whole forward pass
group_args() {
    case $1 in
        prefill) echo "--slots gate_up,down,qkvz,ba,out,qkv,o,fc --m $PREFILL_M --dense-m $PREFILL_M" ;;
        sched0) echo "--slots $SCHED_SLOTS --m $SCHED0_M" ;;
        sched1) echo "--slots $SCHED_SLOTS --m $SCHED1_M" ;;
        sched2) echo "--slots $SCHED_SLOTS --m $SCHED2_M" ;;
        heuronly) echo "--slots down,fc --m $SCHED0_M,$SCHED1_M,$SCHED2_M --dense-m $SCHED0_M,$SCHED1_M,$SCHED2_M" ;;
        small) echo "--slots d_qkv,d_o,d_gate_up,d_down,fc --dense-m $SMALL_M" ;;
        lmhead_small) echo "--slots lm_head --dense-m $SMALL_M" ;;
        lmhead_pl) echo "--slots lm_head --dense-m $PL_M" ;;
        *) echo "unknown group $1 (prefill sched0 sched1 sched2 small lmhead_small lmhead_pl heuronly)" >&2; return 1 ;;
    esac
}
GROUPS_ALL=(prefill sched0 sched1 sched2 small lmhead_small lmhead_pl heuronly)

# GPU phases run one command per flock hold. A rows file is looked up as
# <dir>/tunableop_results.csv: torch inserts the device ordinal, which is 0 under
# HIP_VISIBLE_DEVICES=<n>, so the file is <dir>/tunableop_results0.csv.
gpu_run() {
    flock "$LOCK" "$@"
}

phase=${1:?phase}
shift
case $phase in
    shapes)
        mkdir -p "$W/shapes"
        for g in "${GROUPS_ALL[@]}"; do
            # shellcheck disable=SC2046
            python3 "$H/tunable_hi2.py" shapes "$MODEL" --tp 2 $(group_args "$g") \
                > "$W/shapes/$g.csv"
            echo "$g: $(wc -l < "$W/shapes/$g.csv") keys"
        done
        sort -u "$W"/shapes/*.csv > "$W/census.csv"
        echo "census: $(wc -l < "$W/census.csv") unique keys"
        ;;
    record)
        mkdir -p "$W/record"
        rm -f "$W"/record/untuned*.csv
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
            PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
            PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$W/record/untuned.csv \
            gpu_run "$PY" "$H/tunable_hi2.py" record "$MODEL" --tp 2 \
            --m "${RECORD_M:-49,$PREFILL_M}" --dense-m "${RECORD_DENSE_M:-8,$SMALL_M,33,$PREFILL_M}"
        sort -u "$W/record/untuned0.csv" > "$W/record/recorded.csv"
        echo "recorded: $(wc -l < "$W/record/recorded.csv") keys"
        ;;
    tune)
        g=${1:?group}
        shift
        for ((i = 0; i < SHARDS; i++)); do
            HIP_VISIBLE_DEVICES=$GPU
            export HIP_VISIBLE_DEVICES
            gpu_run env TUNABLEOP_DIR="$TREE/tunableop" VENV="$VENV" \
                bash "$TREE/tools/rdna/prof/tune_missing_gemms.sh" tune \
                --census "$W/shapes/$g.csv" --gpu "$GPU" --shard "$i/$SHARDS" \
                --profile "$PROFILE" --out "$W/tune/$g/shard$i" \
                > "$W/tune/$g.shard$i.log" 2>&1 || echo "shard $i failed (see log)"
            echo "$g shard $i: $(grep -c '^Gemm' "$W/tune/$g/shard$i/tunableop_results0.csv" 2> /dev/null || echo 0) rows"
        done
        ;;
    gather)
        mkdir -p "$W/scratch"
        shards=("$W"/tune/*/shard*/tunableop_results0.csv)
        {
            grep -h '^Validator' "${shards[0]}"
            cat "${shards[@]}" | grep '^Gemm' | sort -t, -k2,2 -u
        } > "$W/scratch/tunableop_results0.csv"
        grep '^Gemm' "$W/scratch/tunableop_results0.csv" | cut -d, -f1,2 > "$W/scratch/keys.csv"
        echo "scratch rows: $(wc -l < "$W/scratch/keys.csv")"
        ;;
    measure)
        # F.linear of every census key on cold rotating weights, CUDA-graph timed
        # (tunable_hi2.py gemm): heuristic = TunableOp off, new = rows looked up
        cond=${1:?heur|new}
        mkdir -p "$W/measure"
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        if [[ $cond == new ]]; then
            export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
            export PYTORCH_TUNABLEOP_FILENAME=${ROWS:-$W/scratch}/tunableop_results.csv
        else
            export PYTORCH_TUNABLEOP_ENABLED=0
        fi
        # every tuned key plus the heuristic-only keys of the expected-gain model
        if [[ -z ${CENSUS:-} ]]; then
            sort -u "$W/scratch/keys.csv" "$W/shapes/heuronly.csv" > "$W/measure/census.csv"
        fi
        gpu_run "$PY" "$H/tunable_hi2.py" gemm "${CENSUS:-$W/measure/census.csv}" \
            --out "$W/measure/$cond.json" > "$W/measure/$cond.log" 2>&1
        echo "measured $cond"
        ;;
    check)
        # rowcheck.py, two processes: the rocBLAS default and the rows (lookup only)
        mkdir -p "$W/check"
        rm -f "$W/reject.txt" "$W"/check/*.json
        cut -d, -f1,2 "$W/scratch/tunableop_results0.csv" | grep '^Gemm' > "$W/check/keys.csv"
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        gpu_run "$PY" "$TREE/tools/rdna/tp8pp2/rowcheck.py" run default "$W/census.csv" \
            "$W/check/keys.csv" --out "$W/check/default.json" > "$W/check/rowcheck.txt" 2>&1
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        PYTORCH_TUNABLEOP_FILENAME=$W/scratch/tunableop_results.csv \
            gpu_run "$PY" "$TREE/tools/rdna/tp8pp2/rowcheck.py" run tuned "$W/census.csv" \
            "$W/check/keys.csv" --out "$W/check/tuned.json" >> "$W/check/rowcheck.txt" 2>&1
        "$PY" "$TREE/tools/rdna/tp8pp2/rowcheck.py" compare "$W/check/default.json" \
            "$W/check/tuned.json" --verbose --reject-out "$W/reject.txt" >> "$W/check/rowcheck.txt" 2>&1 || true
        tail -3 "$W/check/rowcheck.txt"
        ;;
    curate)
        "$PY" "$TREE/tools/rdna2_028/curate_tunableop_rows.py" \
            --current-rows "$TREE/tunableop/$PROFILE" --scratch-rows "$W/scratch" \
            --heur "$W/measure/heur.json" --cur "$W/measure/heur.json" --new "$W/measure/new.json" \
            --out "$W/curated" --adopt-margin "$ADOPT" --drop-margin "$ADOPT" --ranks 0
        grep '^Gemm' "$W/curated/tunableop_results0.csv" | cut -d, -f1,2 > "$W/curated/keys.csv"
        echo "curated rows: $(wc -l < "$W/curated/keys.csv")"
        ;;
    time)
        cond=${1:?heur|new}
        mkdir -p "$W/time"
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        if [[ $cond == new ]]; then
            export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
            export PYTORCH_TUNABLEOP_FILENAME=${ROWS:-$W/curated}/tunableop_results.csv
        else
            export PYTORCH_TUNABLEOP_ENABLED=0
        fi
        gpu_run "$PY" "$H/tunable_hi2.py" time "$MODEL" --tp 2 --m "${TIME_M:-$PREFILL_M}" \
            --dense-m "${TIME_DENSE_M:-$SMALL_M}" ${TIME_SLOTS:+--slots "$TIME_SLOTS"} \
            --out "$W/time/$cond${TIME_TAG:-}.json" > "$W/time/$cond${TIME_TAG:-}.log" 2>&1
        echo "timed $cond"
        ;;
    hit)
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
            PYTORCH_TUNABLEOP_FILENAME=${ROWS:-$W/curated}/tunableop_results.csv \
            gpu_run "$PY" "$TREE/tools/rdna/tp8pp2/tunehit.py" 0 "${CENSUS:-$W/curated/keys.csv}"
        ;;
    emul)
        # model-level check at TP=1 with the two TP=2 ranks' GEMMs emulated
        # (tp2_emul_check.py); one process per arm: torch 2.12 cannot toggle rows
        # in-process. untuned = TunableOp off; tuned = the rows of ROWS (a profile dir);
        # record = RECORD_UNTUNED, lists every rocBLAS key the model forward issues.
        arm=${1:?untuned|tuned|record}
        mkdir -p "$W/emul"
        HIP_VISIBLE_DEVICES=$GPU
        export HIP_VISIBLE_DEVICES
        # the launcher env of recipe 27b-hi2-tp2 that touches numerics or the engine path
        export VLLM_USE_RDNA2_FA=1 VLLM_FA_RDNA2_GQA_DECODE=1 VLLM_RDNA_HINT_FLEX_COMPUTE_MODE=f16
        export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
        export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1
        export VLLM_RDNA_DECODE_STALL_MS=0 VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32 VLLM_ROCM_USE_AITER=0
        export VLLM_BATCH_INVARIANT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export VLLM_CACHE_ROOT="$W/emul/cache/vllm" TRITON_CACHE_DIR="$W/emul/cache/triton"
        rows_file=""
        case $arm in
            untuned) export PYTORCH_TUNABLEOP_ENABLED=0 ;;
            tuned)
                rows_file=${ROWS:?ROWS (a dir holding tunableop_results0.csv) required}/tunableop_results0.csv
                export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
                export PYTORCH_TUNABLEOP_FILENAME=${ROWS}/tunableop_results.csv
                ;;
            record)
                mkdir -p "$W/emul/record"
                export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
                export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
                export PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$W/emul/record/untuned.csv
                ;;
            *) echo "emul arm must be untuned, tuned or record" >&2; exit 2 ;;
        esac
        # the rows file to count hits against is always the tuned arm's
        rows_file=${rows_file:-${ROWS:+$ROWS/tunableop_results0.csv}}
        # EMUL_SEQ=1: one request at a time (the chunks are the rows' aligned M);
        # EMUL_TAG names the output files ($W/emul/<arm><tag>.json)
        gpu_run "$PY" "$H/tp2_emul_check.py" run "$MODEL" --arm "$arm" \
            ${rows_file:+--rows "$rows_file"} --out "$W/emul/$arm${EMUL_TAG:-}.json" \
            ${EMUL_STEPS:+--steps "$EMUL_STEPS"} ${EMUL_LENGTHS:+--lengths "$EMUL_LENGTHS"} \
            ${EMUL_SEQ:+--sequential} \
            > "$W/emul/$arm${EMUL_TAG:-}.log" 2>&1
        tail -2 "$W/emul/$arm${EMUL_TAG:-}.log"
        ;;
    *)
        echo "unknown phase $phase" >&2
        exit 2
        ;;
esac
