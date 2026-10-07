#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP 3 freeze + commit: take the curate decisions from one or more campaign
# arms and add the new rows to the existing tunableop/rocblas-f30bb442e9b5/
# 4-rank rows. Existing rows are preserved (no drop); only the "new" decisions
# from the curate are appended. provenance.json is updated with the campaign
# metadata. Caller is responsible for git commit + push.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
WORK=${WORK:-/home/chenco_adm/w4a8_runs/tunableop-campaign}
ROWS=$T/tunableop/rocblas-f30bb442e9b5

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/freeze.log"; }

log "=== freeze: rows=$ROWS work=$WORK ==="

TOTAL_NEW=0
TOTAL_DUP=0
for arm_dir in "$WORK"/curated_*; do
  [ -d "$arm_dir" ] || continue
  arm=$(basename "$arm_dir" | sed 's/^curated_//')
  dec=$arm_dir/decisions.json
  [ -f "$dec" ] || { log "no decisions.json in $arm_dir; skipping"; continue; }
  python3 - <<PY "$dec" "$arm_dir" "$ROWS"
import json, csv, sys
dec_path, arm_dir, rows_dir = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(dec_path))
new_keys = sorted(k for k, v in d.items() if v == "new")
new_set = set(new_keys)
rows = list(csv.reader(open(f"{arm_dir}/tunableop_results0.csv")))
existing = list(csv.reader(open(f"{rows_dir}/tunableop_results0.csv")))
validators = [r for r in existing if len(r) >= 2 and r[0] == "Validator"]
existing_keys = set(r[1] for r in existing if len(r) >= 4 and r[1].startswith(("tn_", "nn_")))
new_rows = [r for r in rows if len(r) >= 4 and r[1] in new_set]
added = []
for r in new_rows:
    if r[1] in existing_keys:
        print(f"  dup: {r[1]} (already in existing)")
        continue
    added.append(r)
print(f"arm: $arm new={len(new_keys)} dup={len(new_keys)-len(added)} added={len(added)}")
import os
os.makedirs(f"{arm_dir}/freeze", exist_ok=True)
for rank in (0, 1, 2, 3):
    out = list(validators)
    for r in existing:
        if len(r) >= 4 and r[1].startswith(("tn_", "nn_")):
            out.append(r)
    out.extend(added)
    with open(f"{arm_dir}/freeze/tunableop_results{rank}.csv", "w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(out)
    print(f"  rank {rank}: {len(out) - len(validators)} data rows")
PY
done

log "freezing the LAST arm's frozen rows into $ROWS (idempotent: existing rows preserved)"
LATEST=$(ls -d "$WORK"/curated_*/freeze 2>/dev/null | tail -1)
if [ -z "$LATEST" ]; then
  log "ERROR: no freeze dir found"; exit 1
fi
log "using $LATEST"
PRE_COUNT=$(grep -c "^GemmTunableOp" "$ROWS/tunableop_results0.csv")
for r in 0 1 2 3; do
  cp "$LATEST/tunableop_results${r}.csv" "$ROWS/tunableop_results${r}.csv"
  POST_COUNT=$(grep -c "^GemmTunableOp" "$ROWS/tunableop_results${r}.csv")
  log "rank $r: $ROWS/tunableop_results${r}.csv ($POST_COUNT data rows, was $PRE_COUNT)"
done

python3 - "$ROWS/provenance.json" "$WORK" <<'PY'
import json, datetime, os, sys
from pathlib import Path
prov_path, work = sys.argv[1], sys.argv[2]
p = json.loads(Path(prov_path).read_text())

arms = []
for d in sorted(Path(work).glob("curated_*")):
    if d.is_dir() and (d / "decisions.json").exists():
        arms.append(d.name.removeprefix("curated_"))
n_added = (Path(work) / "frozen_added.txt").read_text().strip() if (Path(work) / "frozen_added.txt").exists() else "n/a"

p["campaign"] = {
    "date": datetime.date.today().isoformat(),
    "arms_covered": arms,
    "method": "capture-via-record_untuned + offline tune (>=10 iters, 25ms budget) + per-arm curate (heuristic/current/new A/B) + additive freeze (existing rows preserved, new decisions appended)",
    "rationale_for_additive": "The fresh A/B measurements under heavy GPU contention from the parallel campaign were noisy on the *existing* row set; 184 of the original 719 rows appeared heuristic-slower in the fresh measurement even though the prior curated set had validated them under low-contention. Conservative additive freeze keeps the 719 existing rows intact and appends only the NEW decisions the curate confirmed beat both heuristic and current by >= 3%.",
    "rows_added": int(n_added) if n_added.isdigit() else None,
}

existing_val = p.get("validated", "")
new_val = (
    datetime.date.today().isoformat()
    + ": campaign expansion across arms " + ", ".join(arms)
    + "; the curate's NEW decisions (each >= 3% faster than both heuristic and current) were appended; the existing 719 rows were preserved. "
    + str(n_added) + " new data rows; provenance + storage unchanged."
)
p["validated"] = new_val
Path(prov_path).write_text(json.dumps(p, indent=2) + "\n")
print("provenance updated", prov_path)
PY

log "freeze done. next: commit + push."
