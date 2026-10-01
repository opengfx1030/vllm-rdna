#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tune + measure + curate the new shapes captured by the parallel
# `tools/rdna2_028/campaign_matrix.sh` capture phase. Uses a single GPU
# (default cuda:0) so the parallel campaign's TP=4 boot on GPUs 0-3 stays
# healthy; tune is still serial on the device, so 1 GPU is enough.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
WORK=/home/chenco_adm/w4a8_runs/tunableop-campaign
CAP_DIR=${CAP_DIR:-/home/chenco_adm/w4a8_runs/_captures/w4a16-fa-m0}
ARM_NAME=${ARM_NAME:-w4a16-fa-m0}
ROWS=$T/tunableop/rocblas-f30bb442e9b5
DEVICE=${DEVICE:-0}

mkdir -p "$WORK"
ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export HIP_VISIBLE_DEVICES=$DEVICE

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/driver_${ARM_NAME}.log"; }

log "=== tune+curate ARM=$ARM_NAME HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES (device=$DEVICE in scope) work=$WORK ==="

awk -F, '$1 ~ /^GemmTunableOp_Half/' "$CAP_DIR/shapes_${ARM_NAME}.txt" 2>/dev/null \
  | awk -F, '{print $2}' | sort -u >"$WORK/cap_keys_${ARM_NAME}.txt"
awk -F, '$1 ~ /^GemmTunableOp/ {print $2}' "$ROWS/tunableop_results0.csv" \
  | sort -u >"$WORK/src_keys_${ARM_NAME}.txt"
comm -23 "$WORK/cap_keys_${ARM_NAME}.txt" "$WORK/src_keys_${ARM_NAME}.txt" \
  >"$WORK/new_keys_${ARM_NAME}.txt"
N_NEW=$(wc -l <"$WORK/new_keys_${ARM_NAME}.txt")
log "source rows: $(wc -l <$WORK/src_keys_${ARM_NAME}.txt); cap keys: $(wc -l <$WORK/cap_keys_${ARM_NAME}.txt); new: $N_NEW"

if [ "$N_NEW" -eq 0 ]; then
  log "no new shapes -> skip tune/curate"
  exit 0
fi

TUNE_SRC=$WORK/tune_source_${ARM_NAME}.csv
echo "Validator,PT_VERSION,2.12.0" >"$TUNE_SRC"
echo "Validator,HIP_VERSION,714" >>"$TUNE_SRC"
while read -r key; do
  [ -z "$key" ] && continue
  echo "GemmTunableOp_Half_TN,$key,Default,0.001" >>"$TUNE_SRC"
done <"$WORK/new_keys_${ARM_NAME}.txt"
log "tune source: $TUNE_SRC ($N_NEW keys)"

TUNE_SCRATCH=$WORK/scratch_${ARM_NAME}
mkdir -p "$TUNE_SCRATCH"
log "tune device=$DEVICE -> $TUNE_SCRATCH/tunableop_results0.csv"
"$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
  --shape-source "$TUNE_SRC" --mode tune --device "$DEVICE" \
  --scratch "$TUNE_SCRATCH" --rank 0 --no-small-m \
  --iterations 10 --max-ms 25 \
  >"$WORK/tune_${ARM_NAME}.out" 2>"$WORK/tune_${ARM_NAME}.err" \
  || { log "TUNE FAIL"; tail -20 "$WORK/tune_${ARM_NAME}.err"; exit 1; }
log "tune done rows=$(grep -c '^GemmTunableOp' $TUNE_SCRATCH/tunableop_results0.csv 2>/dev/null || echo 0)"

MEAS_SRC=$WORK/measure_source_${ARM_NAME}.csv
cp "$ROWS/tunableop_results0.csv" "$MEAS_SRC"
while read -r key; do
  [ -z "$key" ] && continue
  echo "GemmTunableOp_Half_TN,$key,Default,0.001" >>"$MEAS_SRC"
done <"$WORK/new_keys_${ARM_NAME}.txt"
log "measure source: $MEAS_SRC ($(grep -c "^GemmTunableOp" $MEAS_SRC) total)"

# 3. Measure: heuristic / current / new
MEAS=$WORK/measure_${ARM_NAME}
mkdir -p "$MEAS"
for cond in heuristic current new; do
  case $cond in
    heuristic) src=(--heuristic) ;;
    current)   src=(--rows "$ROWS/tunableop_results0.csv") ;;
    new)       src=(--rows "$TUNE_SCRATCH/tunableop_results0.csv") ;;
  esac
  log "measure $cond"
  "$V/bin/python" "$T/tools/rdna2_028/tune_prod_shapes.py" \
    --shape-source "$MEAS_SRC" --mode measure --device "$DEVICE" "${src[@]}" \
    --warmup 5 --reps 15 --out "$MEAS/$cond.json" \
    >"$WORK/measure_${ARM_NAME}_$cond.out" 2>"$WORK/measure_${ARM_NAME}_$cond.err" \
    || { log "MEASURE FAIL $cond"; tail -20 "$WORK/measure_${ARM_NAME}_$cond.err"; exit 1; }
done

# 4. Curate
log "curate"
"$V/bin/python" "$T/tools/rdna2_028/curate_tunableop_rows.py" \
  --current-rows "$ROWS" --scratch-rows "$TUNE_SCRATCH" \
  --heur "$MEAS/heuristic.json" --cur "$MEAS/current.json" --new "$MEAS/new.json" \
  --adopt-margin 0.03 --drop-margin 0.03 --ranks 0,1,2,3 \
  --out "$WORK/curated_${ARM_NAME}" \
  >"$WORK/curate_${ARM_NAME}.log" 2>&1 \
  || { log "CURATE FAIL"; cat "$WORK/curate_${ARM_NAME}.log"; exit 1; }
grep -E "^decisions" "$WORK/curate_${ARM_NAME}.log" | tee -a "$WORK/driver_${ARM_NAME}.log"

log "tune+curate ARM=$ARM_NAME done"
