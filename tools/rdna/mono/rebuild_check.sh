#!/usr/bin/env bash
# Under the shared HIP 6-9 lock and once the GPUs are idle: rebuild this tree's
# extensions, then the standalone checks (FFN kernel widths, QSA decode).
# Logs under $OUT. Used between serve stages (no server may be running: the
# rebuild replaces vllm/_rocm_C.abi3.so in place).
#   OUT=~/w4a8_runs/monokernel-2a/rb1 bash tools/rdna/mono/rebuild_check.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:?}
LOCKDIR=${LOCKDIR:-$HOME/w4a8_runs/port-v031/GPU69.lockdir}
mkdir -p "$OUT"
sleep "${LOCK_YIELD_S:-90}"  # let a waiting agent in first
until mkdir "$LOCKDIR" 2>/dev/null; do sleep 15; done
echo "monokernel-2a: rebuild + standalone checks" > "$LOCKDIR/owner"
trap 'rm -rf "$LOCKDIR"' EXIT
until [ "$(rocm-smi -d 0 1 2 3 --showmemuse 2>/dev/null | awk '/VRAM%/{s+=$NF} END{print s+0}')" = 0 ]; do
    sleep 10
done
cd "$TREE"
bash "$HERE/build_tree.sh" > "$OUT/build.log" 2>&1 || { echo "BUILD FAILED"; exit 1; }
tail -1 "$OUT/build.log"
GPU=6 OUT=$OUT SCRIPT=$HERE/m1_bench.py HSA_FORCE_FINE_GRAIN_PCIE=0 timeout 900 \
    bash "$HERE/run_m1.sh" --widths 1,3,8 > "$OUT/m1.log" 2>&1
GPU=6 OUT=$OUT SCRIPT=$HERE/qsa_bench.py HSA_FORCE_FINE_GRAIN_PCIE=0 timeout 900 \
    bash "$HERE/run_m1.sh" --iters 10 > "$OUT/qsa.log" 2>&1
echo "checks done"
