#!/usr/bin/env bash
# PR34 full matrix orchestrator: 2 arms x {MTP0,MTP2} x 4 cells + routing-off A/B.
set -uo pipefail
D=/home/chenco_adm/w4a8_runs/pr34_validate
cd "$D"
export CELLS="1 1024 512 701|8 1024 512 708|1 16384 1024 711|8 16384 1024 718"
rm -f "$D/IM_STOP" "$D/matrix.status"

ARM=baseline RECIPE=flashnext-mtp0 TAG=base_mtp0 PORT=18130 bash pr34_ab.sh || echo "STEP_FAIL base_mtp0"
[ -f "$D/IM_STOP" ] && { echo "STOP:PCI_SERR" > "$D/matrix.status"; exit 2; }
ARM=patched  RECIPE=flashnext-mtp0 TAG=pat_mtp0  PORT=18131 WARM_FROM=base_mtp0 bash pr34_ab.sh || echo "STEP_FAIL pat_mtp0"
[ -f "$D/IM_STOP" ] && { echo "STOP:PCI_SERR" > "$D/matrix.status"; exit 2; }
ARM=baseline RECIPE=flashnext-mtp2 TAG=base_mtp2 PORT=18132 bash pr34_ab.sh || echo "STEP_FAIL base_mtp2"
[ -f "$D/IM_STOP" ] && { echo "STOP:PCI_SERR" > "$D/matrix.status"; exit 2; }
ARM=patched  RECIPE=flashnext-mtp2 TAG=pat_mtp2  PORT=18133 WARM_FROM=base_mtp2 bash pr34_ab.sh || echo "STEP_FAIL pat_mtp2"
[ -f "$D/IM_STOP" ] && { echo "STOP:PCI_SERR" > "$D/matrix.status"; exit 2; }
ARM=patched  RECIPE=flashnext-mtp0 MODE=off TAG=pat_mtp0_off PORT=18134 WARM_FROM=pat_mtp0 bash pr34_ab.sh || echo "STEP_FAIL pat_mtp0_off"

echo DONE > "$D/matrix.status"
