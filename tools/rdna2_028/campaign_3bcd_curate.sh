#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP 3 post-capture orchestration:
#   1. Diff per-arm scratch rows vs the frozen source rows to identify NEW shapes.
#   2. If deltas exist: tune them (PHASE=tune), then curate (PHASE=measure + curate).
#   3. If no deltas: skip the tune phase; the existing rows already cover the
#      captured configs.
#   4. Freeze the merged/curated rows into tunableop/rocblas-f30bb442e9b5/
#      (4 ranks) and update provenance.json.
#
# Caller is responsible for committing + pushing after this script writes
# the rows. No /tmp; everything under $WORK.
set -uo pipefail

V=${VENV:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.28.0}
WORK=${WORK:-/home/chenco_adm/w4a8_runs/tunableop-campaign}
ROWS=$T/tunableop/rocblas-f30bb442e9b5
CAP_ROOT=${CAP_ROOT:-/home/chenco_adm/w4a8_runs/_captures}
ARMS=${ARMS:-"w4a16-fa-m2 w4a16-triton-m2 w4a8-fa-m2 w4a8-triton-m2"}
PHASE=${PHASE:-all}  # diff|tune|curate|freeze|all

mkdir -p "$WORK"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/driver.log"; }

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0

# --- 1. DIFF: collect all captured keys per arm and union them ------------
diff_phase() {
  log "=== PHASE=diff ==="
  local union_keys=$WORK/captured_keys.txt
  : >"$union_keys"
  local per_arm=$WORK/per_arm_keys.txt
  : >"$per_arm"
  local total_captured=0

  for arm in $ARMS; do
    local f=$CAP_ROOT/$arm/scratch/tunableop_results0.csv
    if [ ! -f "$f" ]; then
      log "$arm: no scratch rows, skipping"
      continue
    fi
    local cur=$(grep -c "^GemmTunableOp" "$f" || echo 0)
    log "$arm: scratch rows=$cur"
    awk -F, '$1 ~ /^GemmTunableOp/ {print $2}' "$f" | sort -u >>"$union_keys"
    echo "$arm $(grep -c "^GemmTunableOp" "$f")" >>"$per_arm"
  done

  sort -u "$union_keys" -o "$union_keys"
  local n_captured=$(wc -l <"$union_keys")

  awk -F, '$1 ~ /^GemmTunableOp/ {print $2}' "$ROWS/tunableop_results0.csv" | \
    sort -u >"$WORK/source_keys.txt"
  local n_source=$(wc -l <"$WORK/source_keys.txt")

  comm -23 "$union_keys" "$WORK/source_keys.txt" >"$WORK/new_keys.txt"
  local n_new=$(wc -l <"$WORK/new_keys.txt")
  log "source rows: $n_source; captured (union over $ARMS): $n_captured; new: $n_new"

  if [ "$n_new" -gt 0 ]; then
    log "NEW shapes detected:"
    head -20 "$WORK/new_keys.txt" | tee -a "$WORK/driver.log"
    [ "$n_new" -gt 20 ] && log "... ($((n_new - 20)) more)"
  fi
}

# --- 2. TUNE: offline tune the new shapes (only if diff found deltas) ----
tune_phase() {
  local n_new=$(wc -l <"$WORK/new_keys.txt" 2>/dev/null || echo 0)
  if [ "$n_new" -eq 0 ]; then
    log "=== PHASE=tune SKIPPED (no new shapes) ==="
    return 0
  fi
  log "=== PHASE=tune ($n_new new shapes) ==="

  # Build a shape-source file containing source rows + new keys
  local SHAPE_SRC=$WORK/shape_source.csv
  cat "$ROWS/tunableop_results0.csv" >"$SHAPE_SRC"
  while read -r key; do
    [ -z "$key" ] && continue
    echo "GemmTunableOp_Half_TN,$key,Default,0.001" >>"$SHAPE_SRC"
  done <"$WORK/new_keys.txt"

  local TUNE_SCRATCH=$WORK/scratch
  mkdir -p "$TUNE_SCRATCH"
  local pids=()
  for gpu in 0 1 2 3; do
    log "tune gpu=$gpu -> $TUNE_SCRATCH/tunableop_results$gpu.csv (parallel)"
    "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$SHAPE_SRC" --mode tune --device "$gpu" \
      --scratch "$TUNE_SCRATCH" --rank "$gpu" --no-small-m \
      --iterations 10 --max-ms 25 \
      >"$WORK/tune_gpu$gpu.json" 2>"$WORK/tune_gpu$gpu.err" &
    pids+=($!)
  done
  local rc=0
  for i in 0 1 2 3; do
    wait "${pids[$i]}" || { log "TUNE FAIL gpu=$i"; tail -20 "$WORK/tune_gpu$i.err"; rc=1; }
  done
  [ "$rc" -ne 0 ] && exit 1
  for gpu in 0 1 2 3; do
    log "tune gpu=$gpu done rows=$(grep -c "^GemmTunableOp" "$TUNE_SCRATCH/tunableop_results$gpu.csv" 2>/dev/null || echo 0)"
  done
}

# --- 3. CURATE: heuristic / current / new A/B per shape, then decide -----
curate_phase() {
  local n_new=$(wc -l <"$WORK/new_keys.txt" 2>/dev/null || echo 0)
  if [ "$n_new" -eq 0 ]; then
    log "=== PHASE=curate SKIPPED (no new shapes; existing rows unchanged) ==="
    echo "no_new_shapes" >"$WORK/curate_status.txt"
    return 0
  fi

  log "=== PHASE=curate ==="

  # Union shape source: existing rows + new keys (so the A/B measures each one)
  local SHAPE_SRC=$WORK/shape_source.csv
  cat "$ROWS/tunableop_results0.csv" >"$SHAPE_SRC"
  while read -r key; do
    [ -z "$key" ] && continue
    echo "GemmTunableOp_Half_TN,$key,Default,0.001" >>"$SHAPE_SRC"
  done <"$WORK/new_keys.txt"

  local MEAS=$WORK/measure
  mkdir -p "$MEAS"
  for cond in heuristic current new; do
    local src=()
    case $cond in
      heuristic) src=(--heuristic) ;;
      current)   src=(--rows "$ROWS/tunableop_results0.csv") ;;
      new)       src=(--rows "$WORK/scratch/tunableop_results0.csv") ;;
    esac
    log "measure $cond"
    "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$SHAPE_SRC" --mode measure --device 0 "${src[@]}" \
      --warmup 5 --reps 15 --out "$MEAS/$cond.json" \
      >"$WORK/measure_$cond.out" 2>"$WORK/measure_$cond.err" \
      || { log "MEASURE FAIL $cond"; tail -20 "$WORK/measure_$cond.err"; exit 1; }
  done

  "$V/bin/python" "$T/tools/rdna2_028/curate_tunableop_rows.py" \
    --current-rows "$ROWS" --scratch-rows "$WORK/scratch" \
    --heur "$MEAS/heuristic.json" --cur "$MEAS/current.json" --new "$MEAS/new.json" \
    --adopt-margin 0.03 --drop-margin 0.03 --ranks 0,1,2,3 \
    --out "$WORK/curated" \
    >"$WORK/curate.log" 2>&1 \
    || { log "CURATE FAIL"; cat "$WORK/curate.log"; exit 1; }
  grep -E "^decisions" "$WORK/curate.log" | tee -a "$WORK/driver.log"
  echo "curated" >"$WORK/curate_status.txt"
}

# --- 4. FREEZE: copy curated rows into tunableop/, update provenance ------
freeze_phase() {
  log "=== PHASE=freeze ==="
  local status=$(cat "$WORK/curate_status.txt" 2>/dev/null || echo "no_new_shapes")

  if [ "$status" = "no_new_shapes" ]; then
    log "freeze SKIPPED: no new shapes; existing rows unchanged"
    echo "no_freeze_needed" >"$WORK/freeze_status.txt"
    return 0
  fi

  for r in 0 1 2 3; do
    cp "$WORK/curated/tunableop_results${r}.csv" "$ROWS/tunableop_results${r}.csv"
    local n=$(($(wc -l <"$ROWS/tunableop_results${r}.csv") - 5))
    log "freeze rank $r: $ROWS/tunableop_results${r}.csv ($n data rows)"
  done

  # --- Update provenance.json ---
  local prov=$ROWS/provenance.json
  local config_list
  config_list=$(echo "$ARMS" | tr ' ' ',' | sed 's/-m2//')
  python3 - "$prov" "$ARMS" "$WORK" <<'PY'
import json, sys, datetime
from pathlib import Path
prov_path, arms, work = sys.argv[1], sys.argv[2].split(), sys.argv[3]
p = json.loads(Path(prov_path).read_text())
new_keys = (Path(work) / "new_keys.txt").read_text().splitlines() if (Path(work) / "new_keys.txt").exists() else []
curated_log = (Path(work) / "curate.log").read_text() if (Path(work) / "curate.log").exists() else ""
new_count = sum(1 for line in curated_log.splitlines() if "new " in line.lower() or "decisions:" in line.lower())
p["campaign"] = {
    "date": datetime.date.today().isoformat(),
    "arms": arms,
    "cells_per_arm": "c=1/c=8 x 1k/512 + 16k/1k",
    "method": "record_untuned_enable during live boot + offline tune + curate",
    "new_shapes_captured": len(new_keys),
}
p["validated"] = (
    datetime.date.today().isoformat()
    + ": campaign expansion to " + " ".join(arms)
    + " with the existing 719-row set as the source of truth; "
    + str(len(new_keys)) + " new shape(s) captured, "
    + "re-tuned at >=10 iters / 25 ms budget, curated against fresh same-process"
    + " heuristic / current / new measurements (tools/rdna2_028/{tune_prod_shapes,"
    + "curate_tunableop_rows}.py), and merged into tunableop/rocblas-f30bb442e9b5/."
    + " Solution IDs are build-specific."
)
Path(prov_path).write_text(json.dumps(p, indent=2) + "\n")
print("provenance updated", prov_path)
PY
  log "provenance.json updated"

  echo "frozen" >"$WORK/freeze_status.txt"
}

# --- Driver loop -----------------------------------------------------------
sel_count() { timeout 8 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
SERR0=$(sel_count)
log "=== campaign 3b/c/d start: PHASE=$PHASE ARMS=$ARMS PCI-SERR=$SERR0 ==="
log "uptime=$(uptime -p) tree=$T rows=$ROWS"

case "$PHASE" in
  diff)   diff_phase ;;
  tune)   diff_phase; tune_phase ;;
  curate) diff_phase; tune_phase; curate_phase ;;
  freeze) freeze_phase ;;
  all)
    diff_phase
    tune_phase
    curate_phase
    freeze_phase
    ;;
  *) log "unknown PHASE=$PHASE"; exit 1 ;;
esac

SERR1=$(sel_count)
log "=== done PHASE=$PHASE PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$WORK/STOP"; log "STOP: new PCI SERR"; exit 2; }
exit 0
