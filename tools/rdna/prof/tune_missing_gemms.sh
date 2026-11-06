#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# tune_missing_gemms.sh -- offline TunableOp harness for prefill GEMM shapes
# that are absent from the shipped profile tables.
#
# Usage:
#   tune_missing_gemms.sh missing --census <csv> [--profile <name>]
#                                  [--rows-dir <dir>] [--dry-run]
#   tune_missing_gemms.sh tune --census <csv> --gpu <hip idx> --out <dir>
#                               [--profile <name>] [--rows-dir <dir>]
#                               [--include-strided] [--shard I/N] [--dry-run]
#   tune_missing_gemms.sh harvest --tuned <csv> [--tuned <csv> ...]
#                                  --profile <name> [--rows-dir <dir>] [--dry-run]
#                                  [--reject <file> ...]   (op,key lines to leave out)
#   tune_missing_gemms.sh verify --rows-dir <dir> --census <csv> --gpu <idx>
#                                 [--profile <name>] [--dry-run]
#
# Environment:
#   VENV          Python venv with torch+rocm (for tune/verify)
#   TUNABLEOP_DIR default tunableop dir (default: <tree>/tunableop)
#   PROFILE       default profile name (default: rocm7.14-rocblas5.5)
#   REFUSED_GPUS HIP indices the script never touches, space or comma separated
#                 (default "0 1"; par1-cs25 has its W7800s on HIP 0 and 5: "0 5")
#   TUNE_ATOL, TUNE_RTOL
#                 tolerances of the tuner's numerical check (default 2e-3 each, about
#                 four fp16 half-ulps): a candidate solver whose result differs from the
#                 rocBLAS default's by more is never selected. It rejects inaccurate
#                 solvers only: ones that round differently but as accurately pass (392
#                 of 557 tuned TN rows did). tune_prod_shapes.py uses 1e-2.
#
# The script refuses to run "tune" or "verify" without an explicit --gpu and
# refuses the GPUs in REFUSED_GPUS. It checks that the target GPU is idle (< 1 GiB used,
# measured through torch on that HIP device, so the HIP index needs no
# rocm-smi card mapping) before tuning. "missing" and "harvest --dry-run" are
# CPU-only. "tune --shard I/N" tunes every N-th missing shape, so N GPUs can
# tune disjoint slices in parallel; pass every shard's results file to
# "harvest" with one --tuned each.

set -euo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROWS_DIR=${TUNABLEOP_DIR:-$TREE/tunableop}
PROFILE=${TUNABLEOP_PROFILE:-rocm7.14-rocblas5.5}
PY_HELPER=$TREE/tools/rdna/prof/tunableop_missing.py
VENV=${VENV:-}

usage() {
    sed -n '5,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
}

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

die() { echo "ERROR: $*" >&2; exit 1; }

check_gpu_index() {
    local gpu=$1 refused list=${REFUSED_GPUS:-0 1}
    [[ -n $gpu ]] || die "--gpu is required"
    for refused in ${list//,/ }; do
        if [[ $gpu -eq $refused ]]; then
            die "refusing GPU $gpu: listed in REFUSED_GPUS ($list)"
        fi
    done
}

check_gpu_idle() {
    local gpu=$1
    local used
    set_rocm_libs
    used=$(HIP_VISIBLE_DEVICES="$gpu" "$VENV/bin/python" -c \
        'import torch; free, total = torch.cuda.mem_get_info(); print(total - free)') \
        || die "could not read VRAM of HIP device $gpu through torch"
    local limit=$((1024 * 1024 * 1024))
    if (( used >= limit )); then
        die "GPU $gpu is busy: ${used} bytes used (limit ${limit})"
    fi
    log "GPU $gpu idle: ${used} bytes used (includes this probe's own context)"
}

set_rocm_libs() {
    [[ -n $VENV ]] || die "set VENV to the venv with torch+rocm"
    local sdk=$VENV/lib/python3.12/site-packages
    export LD_LIBRARY_PATH="$sdk/_rocm_sdk_libraries/lib:$sdk/_rocm_sdk_core/lib/host-math/lib:$sdk/_rocm_sdk_core/lib/rocm_sysdeps/lib:$sdk/_rocm_sdk_core/lib:$sdk/torch/lib:${LD_LIBRARY_PATH:-}"
}

cmd_missing() {
    local census="" dry_run=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --census) census=$2; shift 2 ;;
            --rows-dir) ROWS_DIR=$2; shift 2 ;;
            --profile) PROFILE=$2; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            *) die "unknown missing option: $1" ;;
        esac
    done
    [[ -n $census ]] || die "--census is required"
    local args=(missing --census "$census" --profile "$PROFILE"
                --rows-dir "$ROWS_DIR")
    [[ $dry_run -eq 1 ]] && args+=(--dry-run)
    python3 "$PY_HELPER" "${args[@]}"
}

cmd_tune() {
    local census="" gpu="" out="" dry_run=0 include_strided=0 shard=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --census) census=$2; shift 2 ;;
            --gpu) gpu=$2; shift 2 ;;
            --out) out=$2; shift 2 ;;
            --profile) PROFILE=$2; shift 2 ;;
            --rows-dir) ROWS_DIR=$2; shift 2 ;;
            --include-strided) include_strided=1; shift ;;
            --shard) shard=$2; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            *) die "unknown tune option: $1" ;;
        esac
    done
    [[ -n $census ]] || die "--census is required"
    [[ -n $out ]] || die "--out is required"
    check_gpu_index "$gpu"
    local untuned=$out/untuned.csv
    local results=$out/tunableop_results0.csv
    local tune_script=$out/tune_run.py
    local args=(untuned --census "$census" --profile "$PROFILE"
                --rows-dir "$ROWS_DIR" --out "$untuned")
    [[ $include_strided -eq 1 ]] && args+=(--include-strided)
    [[ -n $shard ]] && args+=(--shard "$shard")
    [[ $dry_run -eq 1 ]] && args+=(--dry-run)
    python3 "$PY_HELPER" "${args[@]}"
    log "untuned file: $untuned"
    log "results file: $results"
    log "tuning script: $tune_script"
    if [[ $dry_run -eq 1 ]]; then
        log "dry-run: would write tuning script $tune_script"
        log "dry-run: would run the offline tuner on GPU $gpu"
        return 0
    fi
    check_gpu_idle "$gpu"
    mkdir -p "$out"
    cat >"$tune_script" <<EOF
import torch.cuda.tunable as tunable
tunable.enable(True)
tunable.tuning_enable(True)
tunable.set_numerical_check_tolerances(True, ${TUNE_ATOL:-2e-3}, ${TUNE_RTOL:-2e-3})
tunable.tune_gemm_in_file("$untuned")
tunable.enable(False)
EOF
    set_rocm_libs
    export HIP_VISIBLE_DEVICES="$gpu"
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=1
    export PYTORCH_TUNABLEOP_FILENAME="$out/tunableop_results%d.csv"
    export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    export TORCH_BLAS_PREFER_HIPBLASLT=0
    log "running offline tuner on GPU $gpu"
    "$VENV/bin/python" "$tune_script"
    log "tuned rows: $results"
}

cmd_harvest() {
    local tuned=() rejects=() dry_run=0 profile_set=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tuned) tuned+=("$2"); shift 2 ;;
            --reject) rejects+=("$2"); shift 2 ;;
            --rows-dir) ROWS_DIR=$2; shift 2 ;;
            --profile) PROFILE=$2; profile_set=1; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            *) die "unknown harvest option: $1" ;;
        esac
    done
    [[ ${#tuned[@]} -gt 0 ]] || die "--tuned is required"
    [[ $profile_set -eq 1 ]] || die "--profile is required for harvest"
    local args=(harvest --profile "$PROFILE" --rows-dir "$ROWS_DIR")
    local t
    for t in "${tuned[@]}"; do args+=(--tuned "$t"); done
    if [[ ${#rejects[@]} -gt 0 ]]; then
        for t in "${rejects[@]}"; do args+=(--reject "$t"); done
    fi
    [[ $dry_run -eq 1 ]] && args+=(--dry-run)
    python3 "$PY_HELPER" "${args[@]}"
}

cmd_verify() {
    local census="" gpu="" dry_run=0 rows_dir_set=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --census) census=$2; shift 2 ;;
            --gpu) gpu=$2; shift 2 ;;
            --rows-dir) ROWS_DIR=$2; rows_dir_set=1; shift 2 ;;
            --profile) PROFILE=$2; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            *) die "unknown verify option: $1" ;;
        esac
    done
    [[ -n $census ]] || die "--census is required"
    [[ $rows_dir_set -eq 1 ]] || die "--rows-dir is required for verify"
    check_gpu_index "$gpu"
    local scratch=$ROWS_DIR/verify_scratch
    local args=(verify --census "$census" --gpu "$gpu"
                --rows-dir "$ROWS_DIR" --profile "$PROFILE"
                --out "$scratch")
    [[ $dry_run -eq 1 ]] && args+=(--dry-run)
    python3 "$PY_HELPER" "${args[@]}"
    if [[ $dry_run -eq 1 ]]; then
        log "dry-run: would run the verify script"
        return 0
    fi
    check_gpu_idle "$gpu"
    set_rocm_libs
    export HIP_VISIBLE_DEVICES="$gpu"
    log "running lookup-only verification on GPU $gpu"
    "$VENV/bin/python" "$scratch/verify_lookup.py"
}

main() {
    [[ $# -ge 1 ]] || usage
    local cmd=$1
    shift
    case $cmd in
        missing) cmd_missing "$@" ;;
        tune) cmd_tune "$@" ;;
        harvest) cmd_harvest "$@" ;;
        verify) cmd_verify "$@" ;;
        *) usage ;;
    esac
}

main "$@"
