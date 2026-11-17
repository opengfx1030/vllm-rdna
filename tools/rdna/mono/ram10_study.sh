#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reproducible TP4 expert-cache study runner (R/F/D/Z arms).
#
#   ARM=matrix bash tools/rdna/mono/ram10_study.sh --dryrun
#   ARM=demand REPLICATE=1 bash tools/rdna/mono/ram10_study.sh
#
# One server boot per arm.  The existing launcher runs the correctness probes
# (CELLS=none), then this script is invoked as its LOAD_HOOK to run each
# benchmark cell: reset prefix cache, one exact-length warm-up request, reset
# again, then the measured `vllm bench serve --save-detailed` run.  Measured
# and warm-up JSON are kept in separate files, and every run writes
# metadata.json with arm knobs, launch argv/env, source and binary hashes.
#
# Arms (all other settings identical):
#   R resident         slots 0,   recipe W4A8 prefill, SDOT4=1
#   F fullcache        slots 128, offload backend control
#   D demand           slots 115 (~89.84% resident, +1 dummy slot)
#   Z zeros            slots 115 + constant symmetric qzeros
#
# Offload arms must override VLLM_RDNA2_W4A8_SDOT4=0: the MonoKernel config
# refuses expert_cache_slots with the W4A8 prefill, and _prepare_expert_cache
# also forces _w4a8=False.  The cache auto-enables breakable FULL_AND_PIECEWISE
# graphs, which sets compilation mode NONE; resident keeps compile mode 3.
# This is an inherent precision/graph selection difference, not PCIe cost.
#
# Environment (defaults follow the existing launchers):
#   TREE VENV MODEL PLE PLE_QUANT_DIR GPUS TP PORT TAG REPLICATE OUT_ROOT
#   ARM CELLS BENCH_EXTRA RESET_PREFIX_CACHE WARMUP SEED RESIDENT_SDOT4
#   REPORT FORCE TREE_SHA AFTER_MATRIX_HOOK
# AFTER_MATRIX_HOOK is optional: after a valid timing matrix and the quiet
# window closes, this script runs it once with OUT URL PORT SERVED MODEL VENV
# GPUS TREE ARM exported. It is not timed; use it for post-matrix diagnostics.
# `--dryrun` / `PRINT=1` prints the exact CPU-only launch and bench commands.
# This script never takes a GPU lease; GPUS is the caller's allocation.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
REPORT_PY=$TREE/tools/rdna/mono/ram10_report.py
BENCH_METRICS=$TREE/tools/rdna/port_v031/bench_metrics.py
VALIDATE=$TREE/tools/rdna/port_v031/serve_validate.sh
OFFLOAD=$TREE/tools/rdna/mono/serve_offload.sh
KFD=$TREE/tools/rdna/port_v031/kfd_evictions.py

DEFAULT_MODEL=$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
DEFAULT_PLE=$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4

die() { printf 'ram10_study: %s\n' "$*" >&2; exit 2; }

usage() {
    cat >&2 <<'EOF'
usage: ram10_study.sh [ARM=...] [CELLS=...] [--dryrun]

  ARM=resident|fullcache|demand|zeros|matrix|R,F,D,Z  (default matrix)
  CELLS='in:out:conc[:requests] ...' (default six cells; `none` = probes only)
  REPLICATE=<id>   included in every run tag and output path (default 1)
  OUT_ROOT=<dir>   default $HOME/w4a8_runs/ram10-study
  FORCE=1          allow an existing non-empty output directory
  REPORT=0         skip the automatic report after the arms
  --dryrun, PRINT=1, -n   print commands, launch nothing
EOF
}

quote() { printf '%q' "$1"; }
print_cmd() {
    local arg
    for arg in "$@"; do printf '%q ' "$arg"; done
    printf '\n'
}

# ---------------------------------------------------------------------------
# Load-hook mode: serve_validate calls this with URL/PORT/SERVED/MODEL/OUT.
# ---------------------------------------------------------------------------
preempt_counter() {
    local dest=$1
    curl -fsS --max-time 30 "$BASE_URL/metrics" > "$dest" || return 1
    awk '
        /^vllm:num_preemptions(_total)?(\{|[ \t])/ { sum += $2; found = 1 }
        END { if (!found) exit 1; printf "%.0f\n", sum }
    ' "$dest"
}

server_alive() {
    local pid
    [[ -r $OUT/serve.pid ]] || return 1
    pid=$(cat "$OUT/serve.pid" 2>/dev/null || true)
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

snapshot_cache() {
    local dest=$1
    [[ $ARM == R ]] && return 0
    curl -sf --max-time 120 -X POST -H 'Content-Type: application/json' \
        -d '{"method":"rdna_expert_cache_control","kwargs":{}}' \
        "$BASE_URL/collective_rpc" -o "$dest" 2> "$dest.err"
}

snapshot_cache_memory() {
    local dest=$1
    curl -sf --max-time 120 -X POST -H 'Content-Type: application/json' \
        -d '{"method":"rdna_cache_memory","kwargs":{}}' \
        "$BASE_URL/collective_rpc" -o "$dest" 2> "$dest.err"
}

reset_prefix() {
    local dest=$1 _attempt
    for _attempt in 1 2 3 4 5; do
        if curl -sf --max-time 30 -X POST "$BASE_URL/reset_prefix_cache" \
            -o "$dest" 2>/dev/null; then
            if env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
                "$VENV/bin/python" - "$dest" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if data.get("success") is True else 1)
PY
            then
                return 0
            fi
        fi
        sleep 1
    done
    return 1
}

build_bench_argv() {
    local input_len=$1 output_len=$2 requests=$3 conc=$4
    local cell_dir=$5 result_file=$6 role=$7 served=$8
    local -a extra=()
    read -r -a extra <<< "${BENCH_EXTRA:-}"
    BENCH_ARGS=(
        "$VENV/bin/python" -m vllm.entrypoints.cli.main bench serve
        --backend openai --endpoint /v1/completions
        --base-url "$BASE_URL" --model "$served" --tokenizer "$MODEL"
        --dataset-name random --random-input-len "$input_len"
        --random-output-len "$output_len" --num-prompts "$requests"
        --max-concurrency "$conc" --ignore-eos --request-rate inf
        --seed "$SEED" --save-result --result-dir "$cell_dir"
        --result-filename "$result_file"
    )
    BENCH_ARGS+=("${extra[@]}")
    BENCH_ARGS+=(
        --metadata
        "ram10_arm=$ARM"
        "ram10_slots=$SLOTS"
        "ram10_replicate=$REPLICATE"
        "ram10_cell=$CELL"
        "ram10_tag=$TAG"
        "ram10_role=$role"
    )
}

run_bench() {
    local input_len=$1 output_len=$2 requests=$3 conc=$4
    local cell_dir=$5 result_file=$6 role=$7
    build_bench_argv "$input_len" "$output_len" "$requests" "$conc" \
        "$cell_dir" "$result_file" "$role" "$SERVED"
    printf '%q ' "${BENCH_ARGS[@]}" > "$cell_dir/$role.cmd"
    printf '\n' >> "$cell_dir/$role.cmd"
    HIP_VISIBLE_DEVICES="$GPUS" "${BENCH_ARGS[@]}" > "$cell_dir/$role.log" 2>&1
}

cell_dir_for() {
    local spec=$1 input_len output_len conc requests
    IFS=: read -r input_len output_len conc requests <<< "$spec"
    CELL_NAME="bench-${input_len}x${output_len}-c${conc}"
    CELL=$CELL_NAME
    CELL_REQUESTS=$(( conc * 4 > 8 ? conc * 4 : 8 ))
    (( conc == 1 )) && CELL_REQUESTS=4
    [[ -n ${requests:-} ]] && CELL_REQUESTS=$requests
    CELL_INPUT=$input_len
    CELL_OUTPUT=$output_len
    CELL_CONC=$conc
}

write_cell_json() {
    local cell_dir=$1 bench_rc=$2 warmup_rc=$3 reset_before=$4
    local reset_after=$5 evictions=$6 preemptions=$7 preempt_total=$8
    local faults=$9 alive=${10} result_present=${11} warmup_present=${12}
    local line_start=${13} line_end=${14}
    local invalid_json='[]' warning_json='[]'
    if ((${#CELL_INVALID[@]})); then
        invalid_json=$(
            printf '"%s",' "${CELL_INVALID[@]}" | sed 's/,$//'
        )
        invalid_json="[$invalid_json]"
    fi
    if ((${#CELL_WARNINGS[@]})); then
        warning_json=$(
            printf '"%s",' "${CELL_WARNINGS[@]}" | sed 's/,$//'
        )
        warning_json="[$warning_json]"
    fi
    cat > "$cell_dir/cell.json" <<EOF
{
  "arm": "$ARM",
  "replicate": "$REPLICATE",
  "tag": "$TAG",
  "cell": "$CELL_NAME",
  "in": $CELL_INPUT,
  "out": $CELL_OUTPUT,
  "conc": $CELL_CONC,
  "requests": $CELL_REQUESTS,
  "bench_rc": $bench_rc,
  "warmup_rc": $warmup_rc,
  "warmup_enabled": $([[ ${WARMUP:-1} == 1 ]] && echo true || echo false),
  "reset_before_warmup_ok": $reset_before,
  "reset_after_warmup_ok": $reset_after,
  "evictions": "$evictions",
  "preemptions_in_window": $preemptions,
  "preemptions_total_after": $preempt_total,
  "faults_in_window": $faults,
  "server_alive_after": $alive,
  "result_present": $result_present,
  "warmup_present": $warmup_present,
  "serve_log_line_start": $line_start,
  "serve_log_line_end": $line_end,
  "measured_json": "cells/$CELL_NAME/bench.json",
  "warmup_json": "cells/$CELL_NAME/warmup.json",
  "valid": false,
  "invalid_reasons": $invalid_json,
  "warnings": $warning_json,
  "measured_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

finalize_cell() {
    local cell_dir=$1
    if [[ ! -x $VENV/bin/python || ! -r $REPORT_PY ]]; then
        echo "RAM10 CELL $CELL_NAME FUZZY finalizer unavailable"
        return 2
    fi
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    "$VENV/bin/python" "$REPORT_PY" --finalize-cell "$cell_dir" \
        > "$cell_dir/finalize.json" 2>&1
}

run_cell() {
    local spec=$1
    cell_dir_for "$spec"
    local cell_dir="$OUT/cells/$CELL_NAME"
    mkdir -p "$cell_dir"
    LAST_CELL_VALID=0
    CELL_INVALID=()
    CELL_WARNINGS=()
    local reset_before=1 reset_after=1 bench_rc=0 warmup_rc=0
    local preempt_before preempt_after preemptions=0 faults=0
    local line_start line_end alive=0 result_present=0 warmup_present=0
    local evictions="unavailable" spid=""

    echo "RAM10 CELL $CELL_NAME in=$CELL_INPUT out=$CELL_OUTPUT conc=$CELL_CONC requests=$CELL_REQUESTS"
    if (( RESET_PREFIX )); then
        if ! reset_prefix "$cell_dir/reset-before-warmup.json"; then
            reset_before=0
            CELL_INVALID+=("prefix_reset_failed")
        fi
    fi
    if (( WARMUP )) && (( reset_before )); then
        run_bench "$CELL_INPUT" "$CELL_OUTPUT" 1 1 "$cell_dir" \
            warmup.json warmup || warmup_rc=$?
        [[ -r $cell_dir/warmup.json ]] && warmup_present=1
        (( warmup_rc == 0 )) || CELL_INVALID+=("warmup_failed")
    fi
    if (( RESET_PREFIX )); then
        if ! reset_prefix "$cell_dir/reset-after-warmup.json"; then
            reset_after=0
            CELL_INVALID+=("prefix_reset_failed")
        fi
    fi
    if (( ! reset_before || ! reset_after )); then
        write_cell_json "$cell_dir" "$bench_rc" "$warmup_rc" \
            "$([[ $reset_before == 1 ]] && echo true || echo false)" \
            "$([[ $reset_after == 1 ]] && echo true || echo false)" \
            "$evictions" 0 0 0 1 0 "$warmup_present" 0 0
        echo "RAM10 CELL $CELL_NAME FAIL prefix reset"
        return 1
    fi

    if ! snapshot_cache "$cell_dir/cache-before.json"; then
        CELL_WARNINGS+=("cache_counters_unavailable")
    fi
    if [[ -r $OUT/serve.pid ]]; then
        spid=$(cat "$OUT/serve.pid" 2>/dev/null || true)
    fi
    if [[ -n $spid ]] && kill -0 "$spid" 2>/dev/null && [[ -x $VENV/bin/python ]]; then
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        "$VENV/bin/python" "$KFD" "$spid" --save \
            "$cell_dir/evicted-before.json" > /dev/null 2>&1 || true
    fi
    line_start=$(wc -l < "$OUT/serve.log" 2>/dev/null || echo 0)
    preempt_before=$(preempt_counter "$cell_dir/metrics-before.txt") || return 1

    run_bench "$CELL_INPUT" "$CELL_OUTPUT" "$CELL_REQUESTS" "$CELL_CONC" \
        "$cell_dir" bench.json measured || bench_rc=$?
    [[ -r $cell_dir/bench.json ]] && result_present=1
    line_end=$(wc -l < "$OUT/serve.log" 2>/dev/null || echo 0)
    preempt_after=$(preempt_counter "$cell_dir/metrics-after.txt") || return 1
    if (( preempt_after < preempt_before )); then
        echo "RAM10 preemption counter reset during measurement"
        return 1
    fi
    preemptions=$(( preempt_after - preempt_before ))
    if ! snapshot_cache "$cell_dir/cache-after.json"; then
        CELL_WARNINGS+=("cache_counters_unavailable")
    fi
    : > "$cell_dir/serve-window.log"
    if (( line_end >= line_start )); then
        tail -n +$((line_start + 1)) "$OUT/serve.log" 2>/dev/null |
            head -n $((line_end - line_start)) > "$cell_dir/serve-window.log" ||
            true
    fi
    if [[ -s $cell_dir/serve-window.log ]]; then
        faults=$(
            grep -acE 'Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped' \
                "$cell_dir/serve-window.log" || true
        )
        [[ -n $faults ]] || faults=0
    fi
    if [[ -n $spid ]] && [[ -r $cell_dir/evicted-before.json ]]; then
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        "$VENV/bin/python" "$KFD" "$spid" --since \
            "$cell_dir/evicted-before.json" > "$cell_dir/evictions.txt" 2>&1 || true
        if grep -q 'INVALID' "$cell_dir/evictions.txt" 2>/dev/null; then
            evictions="INVALID"
        elif grep -q -- '-> ok' "$cell_dir/evictions.txt" 2>/dev/null; then
            evictions="ok"
        fi
    else
        echo "evictions unavailable" > "$cell_dir/evictions.txt"
    fi
    server_alive && alive=1

    (( bench_rc == 0 )) || CELL_INVALID+=("bench_failed")
    (( warmup_rc == 0 )) || CELL_INVALID+=("warmup_failed")
    (( result_present == 1 )) || CELL_INVALID+=("no_result")
    (( alive == 1 )) || CELL_INVALID+=("server_died")
    [[ $evictions == "INVALID" ]] && CELL_INVALID+=("kfd_evictions")
    [[ $evictions == "unavailable" ]] && CELL_WARNINGS+=("kfd_evictions_unavailable")
    (( preemptions == 0 )) || CELL_WARNINGS+=("preemptions_observed")
    (( faults == 0 )) || CELL_INVALID+=("faults")
    if (( WARMUP )) && (( warmup_present == 0 )); then
        CELL_INVALID+=("warmup_missing")
    fi
    if [[ -r $cell_dir/bench.json ]]; then
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        "$VENV/bin/python" "$BENCH_METRICS" "$cell_dir/bench.json" 2>&1 |
            tee -a "$cell_dir/metrics.txt" || true
    fi
    write_cell_json "$cell_dir" "$bench_rc" "$warmup_rc" \
        "$([[ $reset_before == 1 ]] && echo true || echo false)" \
        "$([[ $reset_after == 1 ]] && echo true || echo false)" \
        "$evictions" "$preemptions" "$preempt_after" "$faults" "$alive" \
        "$result_present" "$warmup_present" "$line_start" "$line_end"
    finalize_cell "$cell_dir"
    local frc=$?
    if (( frc == 1 )); then
        LAST_CELL_VALID=0
        echo "RAM10 CELL $CELL_NAME INVALID"
        return 0
    elif (( frc != 0 )); then
        LAST_CELL_VALID=0
        echo "RAM10 CELL $CELL_NAME FINALIZE_FAIL"
        return 1
    fi
    LAST_CELL_VALID=1
    echo "RAM10 CELL $CELL_NAME PASS"
    if (( alive == 0 )); then
        return 1
    fi
    return 0
}


quiet_lock_wait() {
    [[ -n ${BOOT_LOCK:-} ]] || return 0
    local start waited=0
    start=$(date +%s)
    while ! flock -n "$BOOT_LOCK" true 2>/dev/null; do
        if (( $(date +%s) - start > ${QUIET_LOCK_WAIT:-600} )); then
            echo "RAM10 quiet: $BOOT_LOCK still held; refusing measurement"
            return 1
        fi
        echo "RAM10 quiet: boot lock held; waiting"
        sleep "${QUIET_POLL:-30}"
        waited=1
    done
    (( waited )) && sleep "${QUIET_SETTLE:-30}"
    return 0
}

quiet_begin() {
    [[ -n ${QUIET_DIR:-} && -d ${QUIET_DIR:-} ]] || return 0
    local shell=$$
    while pgrep -f "${BUILDX_PAT:-^(/usr/sbin/|/usr/bin/)?docker buildx build}" \
        > /dev/null 2>&1; do
        echo "RAM10 quiet: docker buildx build running; waiting"
        sleep "${QUIET_POLL:-30}"
    done
    QUIET_FILE=$QUIET_DIR/${QUIET_OWN:-${QUIET_NAME:-ram10-$TAG}}
    touch "$QUIET_FILE"
    QUIET_T0=$(date +%s)
    (
        stop=$(( $(date +%s) + ${QUIET_CAP:-900} ))
        while sleep "${QUIET_TOUCH:-60}" && (( $(date +%s) < stop )) &&
            kill -0 "$shell" 2>/dev/null; do
            touch "$QUIET_FILE"
        done
    ) &
    QUIET_PID=$!
    echo "RAM10 quiet: window open $QUIET_FILE"
    quiet_lock_wait
}

quiet_end() {
    if [[ -n ${QUIET_PID:-} ]]; then
        kill "$QUIET_PID" 2>/dev/null || true
        wait "$QUIET_PID" 2>/dev/null || true
        QUIET_PID=""
    fi
    if [[ -n ${QUIET_FILE:-} ]]; then
        rm -f "$QUIET_FILE"
        QUIET_FILE=""
        echo "RAM10 quiet: window closed"
    fi
}

quiet_renew() {
    [[ -n ${QUIET_FILE:-} ]] && quiet_lock_wait
    if [[ -n ${QUIET_FILE:-} && -n ${QUIET_WIN_MAX:-} ]] &&
        (( $(date +%s) - QUIET_T0 > QUIET_WIN_MAX )); then
        quiet_end
        echo "RAM10 quiet: gap ${QUIET_GAP:-120}s"
        sleep "${QUIET_GAP:-120}"
        quiet_begin
    fi
}

load_hook_main() {
    : "${OUT:?LOAD_HOOK requires OUT}" "${PORT:?LOAD_HOOK requires PORT}"
    : "${SERVED:?LOAD_HOOK requires SERVED}" "${MODEL:?LOAD_HOOK requires MODEL}"
    VENV=${RAM10_VENV:-${VENV:-$TREE/.venv}}
    ARM=${RAM10_ARM:?LOAD_HOOK requires RAM10_ARM}
    SLOTS=${RAM10_SLOTS:-0}
    REPLICATE=${RAM10_REPLICATE:-1}
    TAG=${RAM10_TAG:-ram10}
    SEED=${RAM10_SEED:-12345}
    WARMUP=${RAM10_WARMUP:-1}
    RESET_PREFIX=${RAM10_RESET_PREFIX:-1}
    BENCH_EXTRA=${RAM10_BENCH_EXTRA:-}
    BASE_URL=${RAM10_BASE_URL:-${URL:-}}
    BASE_URL=${BASE_URL%/v1/completions}
    [[ -n $BASE_URL ]] || BASE_URL=http://127.0.0.1:$PORT
    local specs=${RAM10_CELLS:-}
    if [[ -z $specs ]]; then
        echo "RAM10 STUDY SKIP no cells"
        echo "RAM10 STUDY PASS"
        return 0
    fi
    if [[ ! -x $VENV/bin/python || ! -r $REPORT_PY ]]; then
        echo "RAM10 STUDY FAIL python/report unavailable"
        return 1
    fi
    QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet}
    QUIET_OWN=${QUIET_OWN:-${QUIET_NAME:-ram10-$TAG}}
    QUIET_FILE=""
    QUIET_PID=""
    trap quiet_end EXIT
    quiet_begin
    rm -f "$OUT/cache-memory.json" "$OUT/cache-memory.json.err"
    if ! snapshot_cache_memory "$OUT/cache-memory.json"; then
        echo "RAM10 CACHE_MEMORY FAIL exact KV bytes unavailable"
        return 1
    fi
    local hard=0 invalid=0 spec
    for spec in $specs; do
        quiet_renew
        if ! run_cell "$spec"; then
            hard=1
            echo "RAM10 CELL $CELL_NAME HARD_FAIL"
            break
        fi
        (( LAST_CELL_VALID )) || invalid=1
    done
    quiet_end
    if [[ -n ${AFTER_MATRIX_HOOK:-} ]]; then
        if (( hard == 0 && invalid == 0 )); then
            local after_rc=0
            URL=${URL:-$BASE_URL/v1/completions}
            GPUS=${GPUS:-}
            export OUT URL PORT SERVED MODEL VENV GPUS TREE ARM
            echo "RAM10 AFTER_MATRIX $AFTER_MATRIX_HOOK"
            if [[ -r $AFTER_MATRIX_HOOK ]]; then
                bash "$AFTER_MATRIX_HOOK" > "$OUT/after_matrix.log" 2>&1 ||
                    after_rc=$?
            else
                echo "RAM10 AFTER_MATRIX hook unreadable: $AFTER_MATRIX_HOOK" |
                    tee -a "$OUT/after_matrix.log"
                after_rc=1
            fi
            if (( after_rc != 0 )); then
                echo "RAM10 AFTER_MATRIX FAIL rc=$after_rc"
                hard=1
            else
                echo "RAM10 AFTER_MATRIX PASS"
            fi
        else
            echo "RAM10 AFTER_MATRIX SKIP invalid matrix"
        fi
    fi
    if (( hard )); then
        echo "RAM10 STUDY FAIL hard cell failure"
        return 1
    fi
    if (( invalid )); then
        echo "RAM10 STUDY FAIL invalid cells"
        return 1
    fi
    echo "RAM10 STUDY PASS"
    return 0
}

# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------
write_metadata() {
    local meta_path=$1 status=$2
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    "$VENV/bin/python" - "$meta_path" "$RUN_OUT/launch.argv" \
        "$RUN_OUT/launch.env" "$status" <<'PY'
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

meta_path, argv_path, env_path, status = sys.argv[1:5]


def text(name):
    value = os.environ.get("RAM10_META_" + name, "")
    return value if value != "" else None


def as_int(name):
    value = text(name)
    return int(value) if value is not None else None


def as_float(name):
    value = text(name)
    return float(value) if value is not None else None


def as_bool(name):
    value = text(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes")


def sha256_file(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_argv(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in data.split(b"\0") if part]


def read_env(path):
    out = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


tree = text("TREE") or "."
venv = text("VENV") or ""
source_files = [
    "vllm/model_executor/layers/fused_moe/rdna_expert_cache.py",
    "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py",
    "vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py",
    "vllm/models/qwen4_exp/amd/mono_decode_rdna2.py",
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py",
    "vllm/config/vllm.py",
    "tools/rdna/serve_rdna.sh",
    "tools/rdna/mono/serve_offload.sh",
    "tools/rdna/mono/expert_cache_worker.py",
    "tools/rdna/port_v031/serve_validate.sh",
    "tools/rdna/port_v031/bench_metrics.py",
    "tools/rdna/mono/ram10_study.sh",
    "tools/rdna/mono/ram10_report.py",
]
file_hashes = {
    rel: sha256_file(Path(tree) / rel) for rel in source_files
}
after_matrix_hook = text("AFTER_MATRIX_HOOK")
if after_matrix_hook:
    hook_path = Path(after_matrix_hook)
    if not hook_path.is_absolute():
        hook_path = Path(tree) / hook_path
    file_hashes[str(hook_path)] = sha256_file(hook_path)
binary_paths = []
for pattern in (
    "lib/python*/site-packages/vllm/_rocm_C*.so",
    "lib/python*/site-packages/vllm/_C*.so",
    "vllm/_rocm_C*.so",
    "vllm/_C*.so",
):
    binary_paths.extend(glob.glob(str(Path(venv) / pattern)))
seen = set()
binary_hashes = []
for path in sorted(binary_paths):
    real = os.path.realpath(path)
    if real in seen:
        continue
    seen.add(real)
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    binary_hashes.append(
        {"path": path, "size": size, "sha256": sha256_file(path)}
    )
git_head = None
git_dirty = None
if text("TREE_SHA") is None:
    try:
        git_head = subprocess.run(
            ["git", "-C", tree, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", tree, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        git_dirty = bool(dirty.strip())
    except (OSError, subprocess.SubprocessError):
        git_head = None
        git_dirty = None

mono_knobs = text("MONO_KNOBS")
if mono_knobs:
    try:
        mono_knobs = json.loads(mono_knobs)
    except ValueError:
        pass
cells = (text("CELLS") or "").split()
metadata = {
    "study": "ram10",
    "created_utc": text("CREATED_UTC")
    or datetime.now(timezone.utc).isoformat(),
    "status": status,
    "arm": text("ARM"),
    "arm_name": text("ARM_NAME"),
    "slots": as_int("SLOTS"),
    "local_experts": as_int("LOCAL_EXPERTS"),
    "dummy_slot": as_bool("DUMMY_SLOT"),
    "device_cache_rows": as_int("DEVICE_CACHE_ROWS"),
    "rows_saved_vs_128": as_int("ROWS_SAVED_VS_128"),
    "resident_fraction": as_float("RESIDENT_FRACTION"),
    "tag": text("TAG"),
    "replicate": text("REPLICATE"),
    "out": text("OUT"),
    "host": socket.gethostname(),
    "tree": text("TREE"),
    "venv": text("VENV"),
    "model": text("MODEL"),
    "ple": text("PLE"),
    "ple_quant_dir": text("PLE_QUANT_DIR"),
    "gpus": text("GPUS"),
    "tp": as_int("TP"),
    "port": as_int("PORT"),
    "recipe": text("RECIPE"),
    "gmem": as_float("GMEM"),
    "kv": text("KV"),
    "maxlen": as_int("MAXLEN"),
    "maxbat": as_int("MAXBAT"),
    "block_size": as_int("BLOCK_SIZE"),
    "seqs": as_int("SEQS"),
    "policy": text("POLICY"),
    "decay": as_int("DECAY"),
    "copy": text("COPY"),
    "mono_cfg": text("MONO_CFG"),
    "mono_knobs": mono_knobs,
    "deterministic_dense": as_bool("DETERMINISTIC_DENSE"),
    "w4a8_prefill": as_bool("W4A8_PREFILL"),
    "w4a8_sdot4": as_int("W4A8_SDOT4"),
    "compile_mode": as_int("COMPILE_MODE"),
    "cudagraph_mode": text("CUDAGRAPH_MODE"),
    "breakable_cudagraph": as_bool("BREAKABLE_CUDAGRAPH"),
    "kv_offload_gb": as_int("KV_OFFLOAD_GB"),
    "cells": cells,
    "functional_only": as_bool("FUNCTIONAL_ONLY"),
    "bench_extra": text("BENCH_EXTRA"),
    "bench_seed": as_int("BENCH_SEED"),
    "warmup_requests_per_cell": as_int("WARMUP"),
    "reset_prefix_cache": as_bool("RESET_PREFIX"),
    "quiet_dir": text("QUIET_DIR"),
    "boot_lock": text("BOOT_LOCK"),
    "after_matrix_hook": text("AFTER_MATRIX_HOOK"),
    "tree_sha": text("TREE_SHA") or git_head,
    "git_head": git_head,
    "git_dirty": git_dirty,
    "source_hashes": file_hashes,
    "binary_hashes": binary_hashes,
    "launch_plan": {
        "argv": read_argv(argv_path),
        "env": read_env(env_path),
        "cells": cells,
    },
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
PY
}

update_metadata() {
    local meta_path=$1 rc=$2
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    "$VENV/bin/python" - "$meta_path" "$rc" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, rc = sys.argv[1:3]
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except (OSError, ValueError):
    sys.exit(0)
data["exit_code"] = int(rc)
data["status"] = "done" if int(rc) == 0 else "failed"
data["finished_utc"] = datetime.now(timezone.utc).isoformat()
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
PY
}

# ---------------------------------------------------------------------------
# Arm resolution and launch plan.
# ---------------------------------------------------------------------------
normalize_arm() {
    case ${1,,} in
        r | resident) echo R ;;
        f | fullcache | full) echo F ;;
        d | demand) echo D ;;
        z | zeros | zero) echo Z ;;
        *) return 1 ;;
    esac
}

arm_name() {
    case $1 in
        R) echo resident ;;
        F) echo fullcache ;;
        D) echo demand ;;
        Z) echo zeros ;;
    esac
}

arm_slots() {
    case $1 in
        R) echo 0 ;;
        F) echo 128 ;;
        D | Z) echo 115 ;;
    esac
}

arm_mono_cfg() {
    case $1 in
        Z) echo '"expert_cache_constant_qzeros":true' ;;
        *) echo "" ;;
    esac
}

resolve_arms() {
    local raw=${1:-matrix} item arm
    ARMS=()
    if [[ ${raw,,} == matrix || ${raw,,} == all ]]; then
        ARMS=(R F D Z)
        return
    fi
    IFS=',' read -r -a _items <<< "$raw"
    for item in "${_items[@]}"; do
        arm=$(normalize_arm "$item") || die "unknown ARM=$item"
        local seen=0
        for existing in "${ARMS[@]:-}"; do
            [[ $existing == "$arm" ]] && seen=1
        done
        (( seen )) || ARMS+=("$arm")
    done
    ((${#ARMS[@]})) || die "no arms selected"
}

build_launch_plan() {
    local arm=$1
    ARM=$arm
    ARM_NAME=$(arm_name "$arm")
    SLOTS=$(arm_slots "$arm")
    MONO_CFG=$(arm_mono_cfg "$arm")
    LOCAL_EXPERTS=128
    DUMMY_SLOT=false
    DEVICE_CACHE_ROWS=""
    ROWS_SAVED_VS_128=0
    RESIDENT_FRACTION=""
    if [[ $arm == R ]]; then
        W4A8_PREFILL=$([[ $RESIDENT_SDOT4 == 1 ]] && echo true || echo false)
        W4A8_SDOT4=$RESIDENT_SDOT4
        if [[ ${VLLM_USE_BREAKABLE_CUDAGRAPH:-} == 1 ]]; then
            BREAKABLE_CUDAGRAPH=true
            COMPILE_MODE=0
        else
            BREAKABLE_CUDAGRAPH=false
            COMPILE_MODE=3
        fi
    else
        W4A8_PREFILL=false
        W4A8_SDOT4=0
        BREAKABLE_CUDAGRAPH=true
        COMPILE_MODE=0
        if [[ $SLOTS -lt $LOCAL_EXPERTS ]]; then
            DUMMY_SLOT=true
        fi
        DEVICE_CACHE_ROWS=$(( SLOTS + (SLOTS < LOCAL_EXPERTS ? 1 : 0) ))
        ROWS_SAVED_VS_128=$(( LOCAL_EXPERTS - DEVICE_CACHE_ROWS ))
        RESIDENT_FRACTION=$(awk -v s="$SLOTS" -v l="$LOCAL_EXPERTS" \
            'BEGIN { printf "%.9f", s / l }')
    fi
    CUDAGRAPH_MODE=FULL_AND_PIECEWISE
    TAG="${BASE_TAG}-${ARM_NAME}-r${REPLICATE}"
    if [[ -n ${OUT_OVERRIDE:-} ]]; then
        RUN_OUT=$OUT_OVERRIDE
        TAG=$(basename "$RUN_OUT")
    else
        RUN_OUT=$OUT_ROOT/$TAG
    fi
    MONO_KNOBS=""
    if [[ $arm != R ]]; then
        MONO_KNOBS="{\"expert_cache_slots\":$SLOTS,\"expert_cache_policy\":\"$POLICY\",\"expert_cache_decay\":$DECAY${MONO_CFG:+,$MONO_CFG}}"
    fi

    LAUNCH_ENV=()
    LAUNCH_ARGS=()
    LAUNCH_ENV+=(
        "MODEL=$MODEL"
        "VENV=$VENV"
        "PLE=$PLE"
        "PLE_QUANT_DIR=$PLE_QUANT_DIR"
        "RECIPE=$RECIPE"
        "GPUS=$GPUS"
        "PORT=$PORT"
        "TAG=$TAG"
        "OUT=$RUN_OUT"
        "CELLS=none"
        "LOAD_HOOK=$SELF"
        "BENCH_EXTRA=$BENCH_EXTRA"
        "VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=1"
        "VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-1}"
        "RAM10_LOAD_HOOK=1"
        "RAM10_ARM=$ARM"
        "RAM10_SLOTS=$SLOTS"
        "RAM10_REPLICATE=$REPLICATE"
        "RAM10_TAG=$TAG"
        "RAM10_CELLS=$CELLS"
        "RAM10_BENCH_EXTRA=$BENCH_EXTRA"
        "RAM10_SEED=$SEED"
        "RAM10_WARMUP=$WARMUP"
        "RAM10_RESET_PREFIX=$RESET_PREFIX_CACHE"
        "RAM10_TREE=$TREE"
        "RAM10_VENV=$VENV"
        "RAM10_BASE_URL=http://127.0.0.1:$PORT"
    )
    if [[ -n ${AFTER_MATRIX_HOOK:-} ]]; then
        LAUNCH_ENV+=("AFTER_MATRIX_HOOK=$AFTER_MATRIX_HOOK")
    fi
    if [[ $arm == R ]]; then
        LAUNCH_ENV+=(
            "EXTRA_ARGS=--monokernel --worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker"
        )
        LAUNCH_ARGS+=(
            "RECIPE=$RECIPE"
            "GPUS=$GPUS"
            "TP=$TP"
            "GMEM=$GMEM"
            "KV=$KV"
            "MAXLEN=$MAXLEN"
            "MAXBAT=$MAXBAT"
            "BLOCK_SIZE=$BLOCK_SIZE"
            "SEQS=$SEQS"
            "VLLM_RDNA2_W4A8_SDOT4=$RESIDENT_SDOT4"
        )
        LAUNCH_LAUNCHER=$VALIDATE
    else
        LAUNCH_ENV+=(
            "VALIDATE=1"
            "RECIPE=$RECIPE"
            "TP=$TP"
            "GPUS=$GPUS"
            "ALLOWED_GPU_SETS=$GPUS"
            "SLOTS=$SLOTS"
            "KV_OFFLOAD_GB=0"
            "MONO_CFG=$MONO_CFG"
            "POLICY=$POLICY"
            "DECAY=$DECAY"
            "GMEM=$GMEM"
            "MAXLEN=$MAXLEN"
            "MAXBAT=$MAXBAT"
            "EXTRA_ARGS=--worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker"
        )
        LAUNCH_ARGS+=(
            "KV=$KV"
            "BLOCK_SIZE=$BLOCK_SIZE"
            "SEQS=$SEQS"
            "VLLM_RDNA2_W4A8_SDOT4=0"
        )
        LAUNCH_LAUNCHER=$OFFLOAD
    fi
    LAUNCH_CMD=(env "${LAUNCH_ENV[@]}" bash "$LAUNCH_LAUNCHER" "${LAUNCH_ARGS[@]}")
}

export_meta_env() {
    export RAM10_META_ARM="$ARM"
    export RAM10_META_ARM_NAME="$ARM_NAME"
    export RAM10_META_SLOTS="$SLOTS"
    export RAM10_META_LOCAL_EXPERTS="$LOCAL_EXPERTS"
    export RAM10_META_DUMMY_SLOT="$DUMMY_SLOT"
    export RAM10_META_DEVICE_CACHE_ROWS="$DEVICE_CACHE_ROWS"
    export RAM10_META_ROWS_SAVED_VS_128="$ROWS_SAVED_VS_128"
    export RAM10_META_RESIDENT_FRACTION="$RESIDENT_FRACTION"
    export RAM10_META_TAG="$TAG"
    export RAM10_META_REPLICATE="$REPLICATE"
    export RAM10_META_OUT="$RUN_OUT"
    export RAM10_META_TREE="$TREE"
    export RAM10_META_VENV="$VENV"
    export RAM10_META_MODEL="$MODEL"
    export RAM10_META_PLE="$PLE"
    export RAM10_META_PLE_QUANT_DIR="$PLE_QUANT_DIR"
    export RAM10_META_GPUS="$GPUS"
    export RAM10_META_TP="$TP"
    export RAM10_META_PORT="$PORT"
    export RAM10_META_RECIPE="$RECIPE"
    export RAM10_META_GMEM="$GMEM"
    export RAM10_META_KV="$KV"
    export RAM10_META_MAXLEN="$MAXLEN"
    export RAM10_META_MAXBAT="$MAXBAT"
    export RAM10_META_BLOCK_SIZE="$BLOCK_SIZE"
    export RAM10_META_SEQS="$SEQS"
    export RAM10_META_POLICY="$POLICY"
    export RAM10_META_DECAY="$DECAY"
    export RAM10_META_COPY="$COPY"
    export RAM10_META_MONO_CFG="$MONO_CFG"
    export RAM10_META_MONO_KNOBS="$MONO_KNOBS"
    export RAM10_META_DETERMINISTIC_DENSE=1
    export RAM10_META_W4A8_PREFILL="$W4A8_PREFILL"
    export RAM10_META_W4A8_SDOT4="$W4A8_SDOT4"
    export RAM10_META_COMPILE_MODE="$COMPILE_MODE"
    export RAM10_META_CUDAGRAPH_MODE="$CUDAGRAPH_MODE"
    export RAM10_META_BREAKABLE_CUDAGRAPH="$BREAKABLE_CUDAGRAPH"
    export RAM10_META_KV_OFFLOAD_GB=0
    export RAM10_META_CELLS="$CELLS"
    export RAM10_META_FUNCTIONAL_ONLY="$([[ -z $CELLS ]] && echo true || echo false)"
    export RAM10_META_TREE_SHA="${TREE_SHA:-}"
    export RAM10_META_BENCH_EXTRA="$BENCH_EXTRA"
    export RAM10_META_BENCH_SEED="$SEED"
    export RAM10_META_WARMUP="$WARMUP"
    export RAM10_META_RESET_PREFIX="$RESET_PREFIX_CACHE"
    export RAM10_META_QUIET_DIR="${QUIET_DIR:-}"
    export RAM10_META_BOOT_LOCK="${BOOT_LOCK:-}"
    export RAM10_META_AFTER_MATRIX_HOOK="${AFTER_MATRIX_HOOK:-}"
    export RAM10_META_CREATED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

print_plan() {
    echo "RAM10 STUDY PLAN"
    echo "  TREE=$TREE"
    echo "  VENV=$VENV"
    echo "  MODEL=$MODEL"
    echo "  PLE=$PLE ($PLE_QUANT_DIR)"
    echo "  GPUS=$GPUS TP=$TP PORT=$PORT"
    echo "  GMEM=$GMEM KV=$KV MAXLEN=$MAXLEN MAXBAT=$MAXBAT BLOCK_SIZE=$BLOCK_SIZE SEQS=$SEQS"
    echo "  BENCH_EXTRA=$BENCH_EXTRA"
    echo "  CELLS=${CELLS:-<none: probes only>}"
    echo "  RUN_ROOT=$OUT_ROOT REPLICATE=$REPLICATE"
    local arm spec input_len output_len conc requests name
    BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
    for arm in "${ARMS[@]}"; do
        build_launch_plan "$arm"
        echo
        echo "ARM $ARM ($ARM_NAME) slots=$SLOTS dummy=$DUMMY_SLOT device_cache_rows=${DEVICE_CACHE_ROWS:-resident} rows_saved=${ROWS_SAVED_VS_128} resident_fraction=${RESIDENT_FRACTION:-1}"
        echo "  W4A8_PREFILL=$W4A8_PREFILL W4A8_SDOT4=$W4A8_SDOT4 BREAKABLE=$BREAKABLE_CUDAGRAPH COMPILE_MODE=$COMPILE_MODE CUDAGRAPH=$CUDAGRAPH_MODE"
        echo "  run=$RUN_OUT"
        echo -n "  LAUNCH "
        print_cmd "${LAUNCH_CMD[@]}"
        for spec in ${CELLS:-}; do
            cell_dir_for "$spec"
            name=$CELL_NAME
            echo "  CELL $spec requests=$CELL_REQUESTS"
            echo -n "    RESET  curl -sf --max-time 30 -X POST "
            quote "http://127.0.0.1:$PORT/reset_prefix_cache"
            printf ' -o %q\n' "$RUN_OUT/cells/$name/reset-before-warmup.json"
            if (( WARMUP )); then
                build_bench_argv "$CELL_INPUT" "$CELL_OUTPUT" 1 1 \
                    "$RUN_OUT/cells/$name" warmup.json warmup \
                    "${SERVED_NAME:-flash-next}"
                echo -n "    WARMUP "
                print_cmd "${BENCH_ARGS[@]}"
            fi
            build_bench_argv "$CELL_INPUT" "$CELL_OUTPUT" "$CELL_REQUESTS" \
                "$CELL_CONC" "$RUN_OUT/cells/$name" bench.json measured \
                "${SERVED_NAME:-flash-next}"
            echo -n "    MEASURED "
            print_cmd "${BENCH_ARGS[@]}"
        done
    done
    echo
    echo "REPORT $VENV/bin/python $REPORT_PY --run-root $OUT_ROOT --baseline R"
}

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
if [[ ${RAM10_LOAD_HOOK:-0} == 1 ]]; then
    load_hook_main
    exit $?
fi

DRY=0
OUT_OVERRIDE=""
for arg in "$@"; do
    case $arg in
        --dryrun | --dry-run | -n | PRINT=1) DRY=1 ;;
        -h | --help) usage; exit 0 ;;
        *=*)
            key=${arg%%=*}
            value=${arg#*=}
            case $key in
                ARM | CELLS | REPLICATE | TAG | OUT_ROOT | OUT | PORT | TP | GPUS | GPU | VENV | MODEL | PLE | PLE_QUANT_DIR | GMEM | KV | MAXLEN | MAXBAT | BLOCK_SIZE | SEQS | BENCH_EXTRA | RESET_PREFIX_CACHE | WARMUP | SEED | RESIDENT_SDOT4 | FORCE | REPORT | TREE_SHA)
                    printf -v "$key" '%s' "$value"
                    ;;
                *) die "unknown override $key" ;;
            esac
            ;;
        *) die "unexpected argument '$arg'" ;;
    esac
done
if [[ ${PRINT:-0} == 1 ]]; then DRY=1; fi

ARM=${ARM:-matrix}
CELLS=${CELLS:-'1024:512:1 1024:512:8 16384:1024:1 16384:1024:8 32768:2048:1 32768:2048:8'}
[[ $CELLS == none ]] && CELLS=""
REPLICATE=${REPLICATE:-1}
BASE_TAG=${TAG:-ram10}
OUT_ROOT=${OUT_ROOT:-$HOME/w4a8_runs/ram10-study}
PORT=${PORT:-18766}
TP=${TP:-4}
GPUS=${GPUS:-${GPU:-2,3,4,5}}
VENV=${VENV:-$TREE/.venv}
MODEL=${MODEL:-$DEFAULT_MODEL}
PLE=${PLE:-int4}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-$DEFAULT_PLE}
RECIPE=${RECIPE:-flashnext-mtp2}
GMEM=${GMEM:-0.92}
KV=${KV:-auto}
MAXLEN=${MAXLEN:-131072}
MAXBAT=${MAXBAT:-2048}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
SEQS=${SEQS:-8}
BENCH_EXTRA=${BENCH_EXTRA:-'--temperature 0 --save-detailed'}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-1}
WARMUP=${WARMUP:-1}
SEED=${SEED:-12345}
RESIDENT_SDOT4=${RESIDENT_SDOT4:-1}
POLICY=${POLICY:-lfu_lru}
DECAY=${DECAY:-128}
COPY=${COPY:-tensor}
FORCE=${FORCE:-0}
REPORT=${REPORT:-1}
if [[ -n ${OUT:-} ]]; then OUT_OVERRIDE=$OUT; fi

resolve_arms "$ARM"
[[ $TP == 4 ]] || die "this study is TP4-only (128 local experts/rank)"
[[ $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPUS must be a comma-separated HIP list"
IFS=',' read -r -a _gpus <<< "$GPUS"
((${#_gpus[@]} == TP)) || die "GPUS lists ${#_gpus[@]} cards but TP=$TP"
if [[ ! $PORT =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    die "PORT must be 1..65535"
fi
[[ $REPLICATE =~ ^[A-Za-z0-9._-]+$ ]] || die "REPLICATE has invalid characters"
[[ $BASE_TAG =~ ^[A-Za-z0-9._-]+$ ]] || die "TAG has invalid characters"
[[ $GMEM =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GMEM must be numeric"
for _var in MAXLEN MAXBAT BLOCK_SIZE SEQS SEED; do
    [[ ${!_var} =~ ^[0-9]+$ ]] || die "$_var must be an integer"
done
[[ $KV == auto || $KV =~ ^[0-9]+$ ]] || die "KV must be auto or bytes"
[[ $POLICY == lfu_lru || $POLICY == lru ]] || die "POLICY must be lfu_lru or lru"
[[ $COPY == tensor ]] || die "this study uses the tensor copy backend only"
[[ $RESIDENT_SDOT4 == 0 || $RESIDENT_SDOT4 == 1 ]] ||
    die "RESIDENT_SDOT4 must be 0 or 1"
[[ $BENCH_EXTRA == *--save-detailed* ]] ||
    die "BENCH_EXTRA must include --save-detailed for per-request data"
[[ $BENCH_EXTRA == *--temperature* ]] ||
    die "BENCH_EXTRA must include --temperature 0"
if [[ $RESET_PREFIX_CACHE == 1 && ${VLLM_SERVER_DEV_MODE:-1} == 0 ]]; then
    die "RESET_PREFIX_CACHE=1 needs VLLM_SERVER_DEV_MODE=1"
fi
if [[ ${VLLM_USE_BREAKABLE_CUDAGRAPH:-} == 0 ]]; then
    die "offload arms need breakable graphs; unset VLLM_USE_BREAKABLE_CUDAGRAPH"
fi

declare -A _seen_cells=()
CELL_SPECS=()
for _spec in $CELLS; do
    IFS=: read -r _in _out _conc _nreq <<< "$_spec"
    [[ $_in =~ ^[0-9]+$ && $_out =~ ^[0-9]+$ && $_conc =~ ^[0-9]+$ ]] ||
        die "bad CELLS entry '$_spec'"
    [[ -z ${_nreq:-} || $_nreq =~ ^[0-9]+$ ]] ||
        die "bad request count in '$_spec'"
    (( _in > 0 && _out > 0 && _conc > 0 )) || die "bad CELLS entry '$_spec'"
    (( _conc <= SEQS )) || die "cell concurrency $_conc exceeds SEQS=$SEQS"
    (( _in + _out <= MAXLEN )) ||
        die "cell $_in+$_out exceeds MAXLEN=$MAXLEN"
    _name="bench-${_in}x${_out}-c${_conc}"
    [[ -z ${_seen_cells[$_name]:-} ]] || die "duplicate cell '$_name'"
    _seen_cells[$_name]=1
    CELL_SPECS+=("$_spec")
done
CELLS="${CELL_SPECS[*]:-}"

if ((DRY)); then
    print_plan
    exit 0
fi

[[ -r $VALIDATE && -r $OFFLOAD && -r $KFD && -r $BENCH_METRICS && -r $REPORT_PY ]] ||
    die "one or more harness/study files are missing under $TREE"
[[ -x $VENV/bin/python ]] || die "VENV=$VENV has no bin/python"

overall=0
for _arm in "${ARMS[@]}"; do
    build_launch_plan "$_arm"
    if [[ -e $RUN_OUT && -n $(ls -A "$RUN_OUT" 2>/dev/null) && $FORCE != 1 ]]; then
        die "$RUN_OUT is not empty; set REPLICATE=<new> or FORCE=1"
    fi
    mkdir -p "$RUN_OUT"
    printf '%s\0' "${LAUNCH_CMD[@]}" > "$RUN_OUT/launch.argv"
    printf '%s\n' "${LAUNCH_ENV[@]}" > "$RUN_OUT/launch.env"
    export_meta_env
    write_metadata "$RUN_OUT/metadata.json" running
    echo "RAM10 ARM $ARM ($ARM_NAME) slots=$SLOTS -> $RUN_OUT"
    set +e
    "${LAUNCH_CMD[@]}" 2>&1 | tee -a "$RUN_OUT/runner.log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc == 0 )) && {
        ! grep -qx 'RAM10 STUDY PASS' "$RUN_OUT/load_hook.txt" ||
        ! grep -q 'PROBE-AFTER-LOAD PASS' "$RUN_OUT/summary.txt" ||
        grep -qE 'PROBE FAIL|PPL PROBE FAIL|CONCURRENCY FAIL' "$RUN_OUT/summary.txt"
    }; then
        echo "RAM10 qualification or load hook failed despite harness exit 0"
        rc=1
    fi
    update_metadata "$RUN_OUT/metadata.json" "$rc"
    if (( rc != 0 )); then
        echo "RAM10 ARM $ARM FAIL rc=$rc"
        overall=$rc
        break
    fi
    echo "RAM10 ARM $ARM DONE"
done

if (( overall == 0 )) && [[ $REPORT == 1 ]]; then
    echo "RAM10 REPORT $OUT_ROOT"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    "$VENV/bin/python" "$REPORT_PY" --run-root "$OUT_ROOT" --baseline R ||
        overall=$?
fi
exit "$overall"
