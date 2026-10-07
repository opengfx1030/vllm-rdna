#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Offline tune / measure / curate the fork's production FP16 TunableOp rows.
#
#   PHASE=tune    tune every production shape on each of OURS GPUs 0..3 into
#                 <WORK>/scratch/tunableop_results{0..3}.csv
#   PHASE=measure time every shape under heuristic / current-rows / new-rows
#   PHASE=all     tune then measure
#
# No /tmp: everything under $WORK. The repo rows are never written here.
#
#   VENV=/path/to/venv   (or activate one so $VIRTUAL_ENV is set)
#   VLLM_TREE=/path/to/tree   (default: the tree this script lives in)
#   WORK=/path/to/scratch     (default: <tree>/cache/tunableop-rows)
#   PROFILE=rocm7.14-rocblas5.5   baseline profile to measure against
set -uo pipefail

T=${VLLM_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
V=${VENV:-${VIRTUAL_ENV:-}}
: "${V:?set VENV=/path/to/python-venv (or activate one so \$VIRTUAL_ENV is set)}"
WORK=${WORK:-$T/cache/tunableop-rows}
PROFILE=${TUNABLEOP_PROFILE:-${PROFILE:-rocm7.14-rocblas5.5}}
ROWS=$T/tunableop/$PROFILE
SHAPE_SRC=${SHAPE_SRC:-$ROWS/tunableop_results0.csv}
SCRATCH=$WORK/scratch
MEAS=$WORK/measure
PHASE=${1:-all}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0

mkdir -p "$SCRATCH" "$MEAS"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/driver.log"; }

sel_count() { sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
SERR0=$(sel_count)

log "=== tunableop rows pipeline PHASE=$PHASE work=$WORK ==="
log "uptime=$(uptime -p) PCI-SERR=$SERR0 tree=$T rows=$ROWS"

if [ "$PHASE" = "tune" ] || [ "$PHASE" = "all" ]; then
  # One tuning process per GPU (each pins its own device). Parallel is safe
  # because the curated set is chosen from fresh same-process measurements that
  # run after; a pick that only looked fast under contention is rejected there.
  pids=()
  for gpu in 0 1 2 3; do
    log "tune gpu=$gpu -> $SCRATCH/tunableop_results$gpu.csv (parallel)"
    "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$SHAPE_SRC" --mode tune --device "$gpu" \
      --scratch "$SCRATCH" --rank "$gpu" \
      --iterations 10 --max-ms 25 \
      >"$WORK/tune_gpu$gpu.json" 2>"$WORK/tune_gpu$gpu.err" &
    pids+=($!)
  done
  rc=0
  for i in 0 1 2 3; do
    wait "${pids[$i]}" || { log "TUNE FAIL gpu=$i"; tail -20 "$WORK/tune_gpu$i.err"; rc=1; }
  done
  [ "$rc" -ne 0 ] && exit 1
  for gpu in 0 1 2 3; do
    log "tune gpu=$gpu done rows=$(($(wc -l <"$SCRATCH/tunableop_results$gpu.csv") - 5))"
  done
fi

if [ "$PHASE" = "measure" ] || [ "$PHASE" = "all" ]; then
  for cond in heuristic current new; do
    case $cond in
      heuristic) src=(--heuristic) ;;
      current)   src=(--rows "$ROWS/tunableop_results0.csv") ;;
      new)       src=(--rows "$SCRATCH/tunableop_results0.csv") ;;
    esac
    log "measure $cond"
    "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$SHAPE_SRC" --mode measure --device 0 "${src[@]}" \
      --warmup 5 --reps 15 --out "$MEAS/$cond.json" \
      >"$WORK/measure_$cond.out" 2>"$WORK/measure_$cond.err" \
      || { log "MEASURE FAIL $cond"; tail -20 "$WORK/measure_$cond.err"; exit 1; }
  done
fi

SERR1=$(sel_count)
log "done PCI-SERR=$SERR1 (base $SERR0) uptime=$(uptime -p)"
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$WORK/STOP"; exit 2; }
exit 0
