#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Lookup-hit proof: prove every row in the frozen CSV is actually consumed by
# TunableOp (not falling back to Default). Iterates every GemmTunableOp_Half
# key, runs it once with the rows enabled, and records the solver that the
# lookup actually returned. A hit = the row's solver; a miss = "Default".
# Misses would mean a row is dead code in the file.
#
#   VENV=/path/to/venv            (or activate one so $VIRTUAL_ENV is set)
#   VLLM_TREE=/path/to/tree       (default: the tree this script lives in)
#   PROFILE=<profile name>        (default: rocm7.14-rocblas5.5)
#   ROWS=<profile dir>            (overrides PROFILE)
#   OUT=<dir>                     (default: <tree>/cache/tunableop-lookup)
#   DEVICE=<gpu index>            (default: 3)
set -uo pipefail

T=${VLLM_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
V=${VENV:-${VIRTUAL_ENV:-}}
: "${V:?set VENV=/path/to/python-venv (or activate one so \$VIRTUAL_ENV is set)}"
PROFILE=${TUNABLEOP_PROFILE:-${PROFILE:-rocm7.14-rocblas5.5}}
ROWS=${ROWS:-$T/tunableop/$PROFILE}
OUT=${OUT:-$T/cache/tunableop-lookup}
DEVICE=${DEVICE:-3}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=$DEVICE
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0

mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/lookup_hit.log"; }

log "=== lookup-hit proof: rows=$ROWS device=$DEVICE ==="

"$V/bin/python" "$T/tools/rdna2_028/verify_tunableop_lookup.py" \
  --rows "$ROWS/tunableop_results0.csv" \
  --out "$OUT/lookup_hits.json" \
  --device 0 \
  >"$OUT/lookup_hit.out" 2>"$OUT/lookup_hit.err" \
  || { log "VERIFY FAIL"; tail -20 "$OUT/lookup_hit.err"; exit 1; }

python3 - <<'PY' "$OUT/lookup_hits.json" "$OUT/lookup_hit_summary.txt" "$ROWS/tunableop_results0.csv"
import csv, json, sys
hits_path, summary_path, rows_path = sys.argv[1], sys.argv[2], sys.argv[3]
p = json.load(open(hits_path))
solvers = p["solvers"]
expected = {}
for row in csv.reader(open(rows_path)):
    if len(row) >= 4 and row[0].startswith("GemmTunableOp"):
        expected[row[1]] = row[2]
misses = [(k, exp, solvers.get(k)) for k, exp in expected.items() if solvers.get(k) != exp]
with open(summary_path, "w") as f:
    f.write(f"total={len(expected)} hit={len(expected) - len(misses)} miss={len(misses)}\n")
    if misses:
        f.write("first 10 misses:\n")
        for k, e, a in misses[:10]:
            f.write(f"  {k}: expected={e} actual={a}\n")
print(open(summary_path).read())
PY

cat "$OUT/lookup_hit_summary.txt" | tee -a "$OUT/lookup_hit.log"
log "lookup-hit proof done"
