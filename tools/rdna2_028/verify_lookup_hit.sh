#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP 4 lookup-hit proof: prove every row in the frozen CSV is actually
# consumed by TunableOp (not falling back to Default). Iterates every
# GemmTunableOp_Half key, runs it once with the rows enabled, and records
# the solver that the lookup actually returned. A hit = the row's solver;
# a miss = "Default". Misses would mean a row is dead code in the file.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
ROWS=$T/tunableop/rocblas-f30bb442e9b5
OUT=/home/chenco_adm/w4a8_runs/tunableop-campaign
DEVICE=${DEVICE:-3}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=$DEVICE

mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/lookup_hit.log"; }

log "=== lookup-hit proof: rows=$ROWS device=$DEVICE ==="

"$V/bin/python" "$T/tools/rdna2_028/verify_tunableop_lookup.py" \
  --rows "$ROWS/tunableop_results0.csv" \
  --out "$OUT/lookup_hits.json" \
  --device 0 \
  >"$OUT/lookup_hit.out" 2>"$OUT/lookup_hit.err" \
  || { log "VERIFY FAIL"; tail -20 "$OUT/lookup_hit.err"; exit 1; }

python3 - <<'PY' "$OUT/lookup_hits.json" "$OUT/lookup_hit_summary.txt"
import json, sys
from pathlib import Path
hits_path, summary_path = sys.argv[1], sys.argv[2]
p = json.load(open(hits_path))
results = p["results"]
expected_path = "/home/chenco_adm/vllm-rdna-0.28.0/tunableop/rocblas-f30bb442e9b5/tunableop_results0.csv"
import csv
expected = {}
for row in csv.reader(open(expected_path)):
    if len(row) >= 4 and row[0] == "GemmTunableOp_Half_TN":
        expected[row[1]] = row[2]
hit = miss = total = 0
misses = []
for key, exp_solver in expected.items():
    total += 1
    rec = results.get(key)
    if rec is None:
        miss += 1
        misses.append((key, exp_solver, "no-record"))
        continue
    solver = rec.get("solver")
    if solver == exp_solver:
        hit += 1
    else:
        miss += 1
        misses.append((key, exp_solver, solver))
with open(summary_path, "w") as f:
    f.write(f"total={total} hit={hit} miss={miss}\n")
    if misses:
        f.write(f"first 10 misses:\n")
        for k, e, a in misses[:10]:
            f.write(f"  {k}: expected={e} actual={a}\n")
print(open(summary_path).read())
PY

cat "$OUT/lookup_hit_summary.txt" | tee -a "$OUT/lookup_hit.log"
log "lookup-hit proof done"
