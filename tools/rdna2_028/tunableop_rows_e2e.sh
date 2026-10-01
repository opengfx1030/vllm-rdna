#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One production Flash-Next arm (W4A16 or W4A8, MTP=0) at the four cells with
# whatever repo rows are currently installed. Thin wrapper over
# flashnext_w4a8_arm.sh that records the box state around the run.
#
#   ARM=w4a8off W4A8=0 TAG=post-w4a16 PORT=18270 bash tunableop_rows_e2e.sh
set -uo pipefail

T=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.28.0}
W=${WORK:-/home/chenco_adm/w4a8_runs/tunableop-rows}
ARM=${ARM:?set ARM (w4a8on|w4a8off)}
W4A8=${W4A8:?set W4A8 (0|1)}
TAG=${TAG:?set TAG}
PORT=${PORT:-18270}
OUT=${OUT:-$W/e2e}
CELLS=${CELLS:-"1 1024 512 221|8 1024 512 222|1 16384 1024 111|8 16384 1024 112"}
mkdir -p "$OUT"

sel() { sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/$TAG.wrapper.log"; }

log "=== e2e $TAG ARM=$ARM W4A8=$W4A8 PORT=$PORT uptime=$(uptime -p) PCI-SERR=$(sel) ==="

ARM=$ARM W4A8=$W4A8 MTP=0 PORT=$PORT OUT=$OUT TAG=$TAG CELLS=$CELLS \
  bash "$T/tools/rdna2_028/flashnext_w4a8_arm.sh"

log "--- $TAG cells.csv ---"
cat "$OUT/$TAG/cells.csv" 2>/dev/null | tee -a "$OUT/$TAG.wrapper.log"
log "--- $TAG markers (W4A8 arm) ---"
grep -E "W4A8 sdot4 path active|rdna_ar:" "$OUT/$TAG/markers.txt" 2>/dev/null | head -8 | tee -a "$OUT/$TAG.wrapper.log"
log "=== e2e $TAG done status=$(cat "$OUT/$TAG/status.txt" 2>/dev/null) PCI-SERR=$(sel) uptime=$(uptime -p) ==="
