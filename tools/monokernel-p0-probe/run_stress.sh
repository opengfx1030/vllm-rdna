#!/usr/bin/env bash
# Extra P0(a) cases that run.sh's defaults do not reach (README "Stress cases"):
#  1. grids above the 72-WG target (plain and coop): where co-residency and the
#     cooperative-launch limit really are (device timestamps, not the occupancy API).
#  2. bounded wait + sticky record: --inject-stall makes one WG stop publishing;
#     plain must report the exact phase/wg/peer and exit; coop is expected to hang
#     in the unbounded grid.sync until the host backstop (exit 3).
#  3. forward progress with a busy kernel that fills every wave slot for longer
#     than the flag-wait bound.
set -uo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs; LOG=logs/p0-stress-$TS.log
exec > >(tee "$LOG") 2>&1
echo "# MonoKernel P0 stress  $(date -Is)  host=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
run() { echo; echo "### $*"; timeout --signal=KILL 300 "$@"; echo "exit=$?"; }
for v in wgp cumode; do
  echo; echo "################ variant: $v"
  P=build/$v/probe
  run $P --only graph --replays 3 --wait-ms 200 --host-ms 5000
  for g in 73 108 144; do
    run $P --only graph --grid $g --replays 3 --wait-ms 200 --host-ms 5000
  done
  for l in 32768 16384; do
    run $P --only graph --grid 144 --lds $l --replays 3 --wait-ms 200 --host-ms 5000
  done
  run $P --only graph --inject-stall 5:17 --replays 3 --wait-ms 100 --host-ms 5000
  for o in busy-first flag-first; do
    run $P --only progress --order $o --busy-per-cu 16 --busy-ms 3000 --wait-ms 500 --host-ms 10000
  done
done
echo; echo "log: $LOG"
