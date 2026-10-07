#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EXL3 27B TunableOp campaign: turn the record-untuned census into a curated
# union row set and freeze it.
#
#   PHASE=diff     union the captured shapes, diff against the frozen rows,
#                  write novel GemmTunableOp_Half keys
#   PHASE=tune     offline-tune ONLY the novel keys (4 GPUs in parallel)
#   PHASE=measure  heuristic / current / new per shape over the union set
#   PHASE=curate   adopt >=3% / drop >3% / fold Default (curate_tunableop_rows.py)
#   PHASE=freeze   backup the prior rows + install the curated union + provenance
#   PHASE=verify   lookup-hit proof over the frozen rows
#   PHASE=all      diff -> tune -> measure -> curate
#
# GPUs: default 4-7 (HIP_VISIBLE_DEVICES=4,5,6,7 -> in-process 0..3). The
# Flash-Next co-tenant owns 0-3 and is never touched. No /tmp.
set -uo pipefail

V=${VENV:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.28.0}
WORK=${WORK:-/home/chenco_adm/w4a8_runs/exl3-tunableop}
ROWS=${ROWS:-$T/tunableop/rocblas-f30bb442e9b5}
CAP_ROOT=${CAP_ROOT:-/home/chenco_adm/w4a8_runs/_captures-exl3}
ARMS=${ARMS:-"exl3-m0 exl3-m2"}
GPUS=${GPUS:-4,5,6,7}
PHASE=${PHASE:-all}

mkdir -p "$WORK"
ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/driver.log"; }
sel_count() { timeout 8 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

diff_phase() {
  log "=== PHASE=diff ==="
  : >"$WORK/cap_keys.txt"
  for arm in $ARMS; do
    f=$CAP_ROOT/$arm/shapes_$arm.txt
    [ -f "$f" ] || { log "$arm: no shapes file ($f)"; continue; }
    grep -E "^GemmTunableOp_Half_(TN|NN)," "$f" >>"$WORK/cap_keys.txt"
    log "$arm: $(grep -cE '^GemmTunableOp_Half' "$f") GemmTunableOp_Half keys"
  done
  sort -u "$WORK/cap_keys.txt" -o "$WORK/cap_keys.txt"
  awk -F, '$1 ~ /^GemmTunableOp/ {print $1","$2}' "$ROWS/tunableop_results0.csv" | sort -u >"$WORK/src_keys.txt"
  comm -23 "$WORK/cap_keys.txt" "$WORK/src_keys.txt" >"$WORK/new_keys.txt"
  comm -12 "$WORK/cap_keys.txt" "$WORK/src_keys.txt" >"$WORK/collide_keys.txt"
  local n_cap n_src n_new n_col
  n_cap=$(wc -l <"$WORK/cap_keys.txt"); n_src=$(wc -l <"$WORK/src_keys.txt")
  n_new=$(wc -l <"$WORK/new_keys.txt"); n_col=$(wc -l <"$WORK/collide_keys.txt")
  log "captured GemmTunableOp_Half=$n_cap current=$n_src colliding=$n_col NOVEL=$n_new"
  [ "$n_new" -gt 0 ] && { log "novel keys:"; head -30 "$WORK/new_keys.txt" | tee -a "$WORK/driver.log"; [ "$n_new" -gt 30 ] && log "... ($((n_new - 30)) more)"; }
}

build_tune_src() {
  echo "Validator,PT_VERSION,2.12.0" >"$WORK/tune_source.csv"
  echo "Validator,HIP_VERSION,714" >>"$WORK/tune_source.csv"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    printf '%s,Default,0.001\n' "$line" >>"$WORK/tune_source.csv"
  done <"$WORK/new_keys.txt"
}

build_meas_src() {
  cat "$ROWS/tunableop_results0.csv" >"$WORK/measure_source.csv"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    printf '%s,Default,0.001\n' "$line" >>"$WORK/measure_source.csv"
  done <"$WORK/new_keys.txt"
}

tune_phase() {
  local n_new; n_new=$(wc -l <"$WORK/new_keys.txt" 2>/dev/null || echo 0)
  if [ "$n_new" -eq 0 ]; then log "=== PHASE=tune SKIPPED (no novel keys) ==="; return 0; fi
  log "=== PHASE=tune ($n_new novel keys) ==="
  build_tune_src
  mkdir -p "$WORK/scratch"
  local pids=() gpu
  for gpu in 0 1 2 3; do
    HIP_VISIBLE_DEVICES="$GPUS" "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$WORK/tune_source.csv" --mode tune --device "$gpu" \
      --scratch "$WORK/scratch" --rank "$gpu" \
      --iterations 10 --max-ms 25 \
      >"$WORK/tune_inproc$gpu.json" 2>"$WORK/tune_inproc$gpu.err" &
    pids+=($!)
  done
  local rc=0 i
  for i in 0 1 2 3; do
    wait "${pids[$i]}" || { log "TUNE FAIL inproc=$i"; tail -15 "$WORK/tune_inproc$i.err" | tee -a "$WORK/driver.log"; rc=1; }
  done
  [ "$rc" -ne 0 ] && return 1
  for gpu in 0 1 2 3; do
    log "tune rank $gpu rows=$(grep -c '^GemmTunableOp' "$WORK/scratch/tunableop_results$gpu.csv" 2>/dev/null || echo 0)"
  done
}

measure_phase() {
  local n_new; n_new=$(wc -l <"$WORK/new_keys.txt" 2>/dev/null || echo 0)
  if [ "$n_new" -eq 0 ]; then log "=== PHASE=measure SKIPPED (no novel keys) ==="; return 0; fi
  log "=== PHASE=measure (union $(grep -c '^GemmTunableOp' "$WORK/measure_source.csv" 2>/dev/null || echo 0) shapes) ==="
  build_meas_src
  mkdir -p "$WORK/measure"
  local cond
  for cond in heuristic current new; do
    local src=()
    case $cond in
      heuristic) src=(--heuristic) ;;
      current)   src=(--rows "$ROWS/tunableop_results0.csv") ;;
      new)       src=(--rows "$WORK/scratch/tunableop_results0.csv") ;;
    esac
    log "measure $cond"
    HIP_VISIBLE_DEVICES="$GPUS" "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$WORK/measure_source.csv" --mode measure --device 0 "${src[@]}" \
      --warmup 5 --reps 15 --out "$WORK/measure/$cond.json" \
      >"$WORK/measure_$cond.out" 2>"$WORK/measure_$cond.err" \
      || { log "MEASURE FAIL $cond"; tail -15 "$WORK/measure_$cond.err" | tee -a "$WORK/driver.log"; return 1; }
  done
}

curate_phase() {
  local n_new; n_new=$(wc -l <"$WORK/new_keys.txt" 2>/dev/null || echo 0)
  if [ "$n_new" -eq 0 ]; then log "=== PHASE=curate SKIPPED (no novel keys) ==="; return 0; fi
  log "=== PHASE=curate ==="
  "$V/bin/python" "$T/tools/rdna2_028/curate_tunableop_rows.py" \
    --current-rows "$ROWS" --scratch-rows "$WORK/scratch" \
    --heur "$WORK/measure/heuristic.json" --cur "$WORK/measure/current.json" --new "$WORK/measure/new.json" \
    --adopt-margin 0.03 --drop-margin 0.03 --ranks 0,1,2,3 \
    --out "$WORK/curated" >"$WORK/curate.log" 2>&1 \
    || { log "CURATE FAIL"; cat "$WORK/curate.log" | tee -a "$WORK/driver.log"; return 1; }
  grep -E "^decisions" "$WORK/curate.log" | tee -a "$WORK/driver.log"
  # summary counts
  python3 - "$WORK/curated/decisions.json" "$WORK" <<'PY' | tee -a "$WORK/driver.log"
import json, sys
from pathlib import Path
d = json.load(open(sys.argv[1]))
w = Path(sys.argv[2])
rows0 = [l for l in (w/"curated/tunableop_results0.csv").read_text().splitlines() if l.startswith("GemmTunableOp")]
new = sorted(k for k, v in d.items() if v == "new")
dropped = sorted(k for k, v in d.items() if v == "dropped")
(w/"adopted_new_keys.txt").write_text("\n".join(new) + ("\n" if new else ""))
(w/"dropped_keys.txt").write_text("\n".join(dropped) + ("\n" if dropped else ""))
print(f"curated rows/rank={len(rows0)} adopted_new={len(new)} dropped={len(dropped)}")
PY
}

freeze_phase() {
  log "=== PHASE=freeze ==="
  local pre=$WORK/repo-rows-premerge
  mkdir -p "$pre"
  cp "$ROWS"/tunableop_results*.csv "$ROWS/provenance.json" "$pre/" 2>/dev/null
  local pre_n; pre_n=$(grep -c '^GemmTunableOp' "$ROWS/tunableop_results0.csv")
  local r
  for r in 0 1 2 3; do
    cp "$WORK/curated/tunableop_results$r.csv" "$ROWS/tunableop_results$r.csv"
    log "rank $r: $(grep -c '^GemmTunableOp' "$ROWS/tunableop_results$r.csv") rows (was $pre_n)"
  done
  local post_n; post_n=$(grep -c '^GemmTunableOp' "$ROWS/tunableop_results0.csv")
  log "FROZEN $pre_n -> $post_n rows/rank (backup $pre)"
}

verify_phase() {
  log "=== PHASE=verify ==="
  HIP_VISIBLE_DEVICES="$GPUS" "$V/bin/python" "$T/tools/rdna2_028/verify_tunableop_lookup.py" \
    --rows "$ROWS/tunableop_results0.csv" --out "$WORK/lookup_hits.json" --device 0 \
    >"$WORK/lookup_hit.out" 2>"$WORK/lookup_hit.err" \
    || { log "VERIFY FAIL"; tail -20 "$WORK/lookup_hit.err" | tee -a "$WORK/driver.log"; return 1; }
  grep -E "^lookup hits|MISS" "$WORK/lookup_hit.out" | tee -a "$WORK/driver.log"
}

SERR0=$(sel_count)
log "=== exl3 tunableop campaign PHASE=$PHASE ARMS=[$ARMS] GPUs=$GPUS PCI-SERR=$SERR0 ==="
log "tree=$T rows=$ROWS work=$WORK uptime=$(uptime -p)"

case "$PHASE" in
  diff)    diff_phase ;;
  tune)    diff_phase; tune_phase ;;
  measure) diff_phase; tune_phase; measure_phase; curate_phase ;;
  curate)  diff_phase; tune_phase; measure_phase; curate_phase ;;
  freeze)  freeze_phase ;;
  verify)  verify_phase ;;
  all)     diff_phase; tune_phase; measure_phase; curate_phase ;;
  *) log "unknown PHASE=$PHASE"; exit 1 ;;
esac

SERR1=$(sel_count)
log "=== done PHASE=$PHASE PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$WORK/STOP"; log "STOP: new PCI SERR"; exit 2; }
exit 0
