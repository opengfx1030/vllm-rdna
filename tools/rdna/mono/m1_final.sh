#!/usr/bin/env bash
# M1 final matrix: widths 3/6/12/24, 16 distinct layers, prefetch off and on.
#   GPU=2 bash tools/rdna/mono/m1_final.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/w4a8_runs/monokernel-2a/m1final}
mkdir -p "$OUT"
for pf in 0 1; do
    MONO_FFN_PREFETCH=$pf OUT=$OUT timeout 3000 bash "$HERE/run_m1.sh" \
        --widths 3,6,12,24 --layers 16 --iters 20 --wg 72 \
        --json "$OUT/m1_final_pf$pf.json" > "$OUT/m1_final_pf$pf.log" 2>&1
    echo "prefetch=$pf rc=$?"
    python3 - "$OUT/m1_final_pf$pf.json" "$pf" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for M, v in r["widths"].items():
    t = v["timing"]
    print(sys.argv[2], M, round(t["today_block"]["median_us_per_layer"], 1),
          round(t["mono_kernel"]["median_us_per_layer"], 1), round(t["speedup"], 3),
          t["mono_phase_us"], t["sticky_err_after"])
PY
done
