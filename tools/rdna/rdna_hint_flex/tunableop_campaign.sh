#!/usr/bin/env bash
# TunableOp census -> tune -> measure -> curate -> merge for the
# rdna_hint_flex / dense Qwen3.5-4B serving shapes, following
# tunableop/rocm7.14-rocblas5.5/README.md (one profile folder per rocBLAS
# build; never /tmp or the run CWD).
#
#   WORK=$HOME/w4a8_runs/hi2-support/tunable GPU=<hip> \
#     bash tools/rdna/rdna_hint_flex/tunableop_campaign.sh census|tune|measure|curate|merge
#
# census  : union of the RECORD_UNTUNED files under $CENSUS (bench_serve.sh
#           CENSUS_DIR; CENSUS_GLOB narrows it) -> $WORK/census_all.csv
# tune    : tune every census shape (plus the M=1..8 decode extension) on GPU
#           -> $WORK/scratch/tunableop_results0.csv
# tune-part / gather : the same split over several GPUs (PART/PARTS), merged
# measure : time each shape under heuristic / current rows / new rows
# curate  : keep a new row only when >= 3 % faster than both (adopt margin)
# merge   : add the adopted rows to all four rank files of the profile
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
WORK=${WORK:?WORK required (persistent, never /tmp)}
CENSUS=${CENSUS:-$HOME/w4a8_runs/hi2-support/census}
PROFILE=${PROFILE:-rocm7.14-rocblas5.5}
ROWS=$TREE/tunableop/$PROFILE
PY=$VENV/bin/python
SP=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib/core/lib:$SP/torch/lib
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0
mkdir -p "$WORK"
case ${1:?phase} in
  census)
    # shellcheck disable=SC2086
    cat ${CENSUS_GLOB:-$CENSUS/*/untuned*.csv} | grep '^GemmTunableOp_' | sort -u \
      > "$WORK/census_all.csv"
    wc -l "$WORK/census_all.csv"
    ;;
  tune)
    HIP_VISIBLE_DEVICES=${GPU:?GPU} "$PY" "$TREE/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$WORK/census_all.csv" --mode tune --device 0 \
      --scratch "$WORK/scratch" --rank 0 --iterations 10 --max-ms 25 \
      > "$WORK/tune.json" 2> "$WORK/tune.err"
    echo "rows: $(grep -c '^GemmTunableOp_' "$WORK/scratch/tunableop_results0.csv")"
    ;;
  tune-part)
    # Parallel tuning: shapes not yet in $WORK/scratch, every PARTS-th from
    # PART, into $WORK/scratch-part$PART (gather merges them back).
    part=${PART:?PART}
    "$PY" - "$TREE" "$WORK" "$part" "${PARTS:?PARTS}" <<'PYEOF'
import csv, sys
from pathlib import Path
tree, work, part, parts = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, f"{tree}/tools/rdna2_028")
from tune_prod_shapes import extend_with_small_m, load_shapes
shapes = load_shapes(work / "census_all.csv")
extend_with_small_m(shapes)
done = set()
f = work / "scratch" / "tunableop_results0.csv"
if f.exists():
    done = {r[1] for r in csv.reader(f.open()) if len(r) > 1 and r[0] != "Validator"}
todo = sorted((k for k in shapes if k not in done),
              key=lambda k: (shapes[k][1], shapes[k][2], shapes[k][3]))
mine = todo[part::parts]
with (work / f"census_part{part}.csv").open("w") as fh:
    for k in mine:
        fh.write(f"GemmTunableOp_Half_{shapes[k][0].upper()},{k}\n")
print(f"part {part}: {len(mine)} of {len(todo)} remaining shapes")
PYEOF
    HIP_VISIBLE_DEVICES=${GPU:?GPU} "$PY" "$TREE/tools/rdna2_028/tune_prod_shapes.py" \
      --shape-source "$WORK/census_part$part.csv" --no-small-m --mode tune \
      --device 0 --scratch "$WORK/scratch-part$part" --rank 0 --iterations 10 \
      --max-ms 25 > "$WORK/tune_part$part.json" 2> "$WORK/tune_part$part.err"
    ;;
  gather)
    "$PY" - "$WORK" <<'PYEOF'
import csv, sys
from pathlib import Path
work = Path(sys.argv[1])
main = work / "scratch" / "tunableop_results0.csv"
main.parent.mkdir(parents=True, exist_ok=True)
rows = list(csv.reader(main.open())) if main.exists() else []
have = {r[1] for r in rows if len(r) > 1 and r[0] != "Validator"}
for f in sorted(work.glob("scratch-part*/tunableop_results0.csv")):
    for r in csv.reader(f.open()):
        if r and r[0] == "Validator":
            if not any(x[:2] == r[:2] for x in rows):
                rows.insert(0, r)
        elif len(r) > 1 and r[1] not in have:
            rows.append(r)
            have.add(r[1])
with main.open("w") as fh:
    csv.writer(fh, lineterminator="\n").writerows(rows)
print(f"scratch rows: {len(have)}")
PYEOF
    ;;
  measure)
    mkdir -p "$WORK/measure"
    for cond in heuristic current new; do
      case $cond in
        heuristic) src=(--heuristic) ;;
        current) src=(--rows "$ROWS/tunableop_results0.csv") ;;
        new) src=(--rows "$WORK/scratch/tunableop_results0.csv") ;;
      esac
      HIP_VISIBLE_DEVICES=${GPU:?GPU} "$PY" "$TREE/tools/rdna2_028/tune_prod_shapes.py" \
        --shape-source "$WORK/census_all.csv" --mode measure --device 0 "${src[@]}" \
        --warmup 5 --reps 15 --out "$WORK/measure/$cond.json" \
        > "$WORK/measure_$cond.out" 2> "$WORK/measure_$cond.err"
      echo "measured $cond"
    done
    ;;
  curate)
    "$PY" "$TREE/tools/rdna2_028/curate_tunableop_rows.py" \
      --current-rows "$ROWS" --scratch-rows "$WORK/scratch" \
      --heur "$WORK/measure/heuristic.json" --cur "$WORK/measure/current.json" \
      --new "$WORK/measure/new.json" --out "$WORK/curated" \
      --adopt-margin 0.03 --drop-margin 0.03 --ranks 0
    ;;
  merge)
    "$PY" - "$ROWS" "$WORK/curated/tunableop_results0.csv" "$WORK/curated/decisions.json" <<'PYEOF'
import csv, json, sys
from pathlib import Path
rows_dir, curated, decisions = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dec = json.loads(decisions.read_text())
new_rows = [r for r in csv.reader(curated.open())
            if len(r) >= 4 and r[0] != "Validator" and dec.get(r[1]) == "new"]
for rank in range(4):
    f = rows_dir / f"tunableop_results{rank}.csv"
    rows = list(csv.reader(f.open()))
    have = {r[1] for r in rows if len(r) >= 2 and r[0] != "Validator"}
    add = [r for r in new_rows if r[1] not in have]
    with f.open("w") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows + add)
    n = sum(1 for r in rows + add if r and r[0] != "Validator")
    print(f"rank {rank}: +{len(add)} rows -> {n}")
PYEOF
    ;;
  *) echo "unknown phase" >&2; exit 2 ;;
esac
