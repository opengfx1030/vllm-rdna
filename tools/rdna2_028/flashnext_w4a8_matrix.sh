#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Qwen3.8-Flash-Next W4A8/W4A16 matrix orchestrator (TP=4, FA-RDNA2, RDNA AR=1,
# FULL_AND_PIECEWISE, prefix caching). Four sequential one-engine boots:
#
#   1) W4A8=1 MTP=0   2) W4A8=1 MTP=2   (primary: the HIP W4A8 arm)
#   3) W4A8=0 MTP=0   4) W4A8=0 MTP=2   (same-session W4A16 control)
#
# One global PCI-SERR baseline; a NEW 'PCI SERR' after any phase writes STOP and
# aborts the remaining arms. Each arm is a full tools/rdna2_028/flashnext_w4a8_arm.sh
# run (own tagged cache, warmth, 4 cells). Logs under /home/chenco_adm/w4a8_runs/.
#
# Usage: setsid nohup bash flashnext_w4a8_matrix.sh > /home/chenco_adm/w4a8_runs/matrix.log 2>&1 &
set -uo pipefail

T=/home/chenco_adm/vllm-rdna-0.28.0
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
STAMP=${STAMP:-2026-09-29}
RUN=$OUT/${STAMP}_matrix
mkdir -p "$RUN"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/matrix.log"; }
sel_count() { sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

log "=== Flash-Next W4A8 matrix start host=$(hostname) uptime=$(uptime -p) ==="
SERR0=$(sel_count)
log "global PCI-SERR baseline=$SERR0"
echo "$SERR0" >"$RUN/serr_baseline.txt"

run_arm() {
  local arm=$1 w4a8=$2 mtp=$3 port=$4
  local tag="${STAMP}_fn-${arm}-m${mtp}"
  log "--- ARM $tag (W4A8=$w4a8 MTP=$mtp port=$port) start ---"
  env ARM="$arm" W4A8="$w4a8" MTP="$mtp" PORT="$port" TAG="$tag" OUT="$OUT" \
    SERR_BASE="$SERR0" WARM_TRITON=1 DO_WARMUP=1 \
    bash "$T/tools/rdna2_028/flashnext_w4a8_arm.sh" >>"$RUN/$tag.driver" 2>&1
  local rc=$?
  local c; c=$(sel_count)
  log "--- ARM $tag end rc=$rc PCI-SERR=$c ---"
  if [ "$c" -gt "$SERR0" ]; then
    echo "STOP new PCI SERR in $tag ($c > $SERR0)" >"$RUN/STOP"
    log "STOP: new PCI SERR in $tag; aborting remaining arms"
    return 1
  fi
  [ "$rc" -ne 0 ] && { log "arm $tag rc=$rc (see $RUN/$tag.driver)"; }
  return 0
}

run_arm w4a8on  1 0 18260 || exit 1
run_arm w4a8on  1 2 18261 || exit 1
run_arm w4a8off 0 0 18262 || exit 1
run_arm w4a8off 0 2 18263 || exit 1

log "all arms done; final PCI-SERR=$(( $(sel_count) - SERR0 )) new"
echo "MATRIX_DONE" >"$RUN/matrix.status"
log "=== Flash-Next W4A8 matrix end ==="
