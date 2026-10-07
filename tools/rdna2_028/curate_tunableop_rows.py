# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Curate a TunableOp row set from heuristic / current / freshly-tuned timings.

Inputs are the three per-shape measurement JSONs produced by
`tune_prod_shapes.py --mode measure` (heuristic, current lookup, new lookup) and
the two row sets (current repo rows, freshly tuned scratch rows).

Per shape, keep whichever condition is fastest:

  * `new`     -- fresh offline solver beats both the current row and the
                 heuristic by `--adopt-margin`; adopt for every rank.
  * `current` -- current row is within `--drop-margin` of the heuristic; keep.
  * `dropped` -- current row is slower than the heuristic by more than
                 `--drop-margin`: the row misleads, omit it so the shape falls
                 back to the default.

Rows whose solver is the heuristic (`Default`) are folded into `dropped`: an
absent row and a `Default` row are behaviourally identical, and omitting them
keeps the shipped file small and production-focused.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path):
    """Return (validators, {key: (op, key, solver, ms)})."""
    validators, rows = [], {}
    if not path.exists():
        return validators, rows
    for row in csv.reader(path.open()):
        if len(row) >= 2 and row[0] == "Validator":
            validators.append(row)
        elif len(row) >= 4:
            rows[row[1]] = tuple(row)
    return validators, rows


def write_rows(path: Path, validators, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerows(validators)
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-rows", required=True, type=Path)
    p.add_argument("--scratch-rows", required=True, type=Path)
    p.add_argument("--heur", required=True, type=Path)
    p.add_argument("--cur", required=True, type=Path)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--adopt-margin", type=float, default=0.03)
    p.add_argument("--drop-margin", type=float, default=0.03)
    p.add_argument("--ranks", default="0,1,2,3")
    args = p.parse_args()

    heur = json.loads(args.heur.read_text())["results"]
    cur_t = json.loads(args.cur.read_text())["results"]
    new_t = json.loads(args.new.read_text())["results"]

    ranks = [int(r) for r in args.ranks.split(",")]
    cur_files = {r: args.current_rows / f"tunableop_results{r}.csv" for r in ranks}
    scratch_files = {r: args.scratch_rows / f"tunableop_results{r}.csv" for r in ranks}
    cur = {r: read_rows(f) for r, f in cur_files.items()}
    new = {r: read_rows(f) for r, f in scratch_files.items()}

    shapes = sorted(set(heur) | set(cur_t) | set(new_t))
    decisions, table = {}, []
    for key in shapes:
        h, c, n = heur.get(key), cur_t.get(key), new_t.get(key)
        if h is None or c is None:
            continue
        if n is not None and n < min(c, h) * (1.0 - args.adopt_margin):
            dec = "new"
        elif c <= h * (1.0 + args.drop_margin):
            dec = "current"
        else:
            dec = "dropped"
        # Fold explicit Default solvers into "dropped" (behaviourally identical).
        if dec in ("new", "current"):
            solver = (new[ranks[0]][1].get(key) if dec == "new" else cur[ranks[0]][1].get(key))
            if solver is None or solver[2] == "Default":
                dec = "dropped"
        decisions[key] = dec
        chosen = {"new": n, "current": c}.get(dec)
        table.append(
            dict(shape=key, heur_ms=h, current_ms=c, new_ms=n,
                 decision=dec, chosen_ms=chosen, gain_vs_current=(c / n if (dec == "new" and n) else 1.0))
        )

    counts = {d: sum(1 for v in decisions.values() if v == d) for d in ("new", "current", "dropped")}
    print(f"decisions: {counts} of {len(decisions)} shapes")

    # Build per-rank curated files.
    args.out.mkdir(parents=True, exist_ok=True)
    for r in ranks:
        validators = cur[r][0] or new[r][0]
        out_rows = []
        for key in shapes:
            dec = decisions.get(key)
            if dec == "new":
                src = (new[r][1].get(key) or new[ranks[0]][1].get(key)
                       or cur[r][1].get(key))
            elif dec == "current":
                src = (cur[r][1].get(key) or new[ranks[0]][1].get(key)
                       or new[r][1].get(key))
            else:
                src = None
            if src is not None:
                out_rows.append(src)
        write_rows(args.out / f"tunableop_results{r}.csv", validators, out_rows)
        if r == ranks[0]:
            base_keys = {row[1] for row in out_rows}
        elif {row[1] for row in out_rows} != base_keys:
            print(f"WARNING: rank {r} key set differs from rank {ranks[0]}")

    with (args.out / "before_after.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shape", "heur_ms", "current_ms", "new_ms", "decision",
                         "chosen_ms", "gain_vs_current", "current_solver", "new_solver"])
        for row in table:
            key = row["shape"]
            writer.writerow([
                key, f"{row['heur_ms']:.6f}",
                f"{row['current_ms']:.6f}" if row["current_ms"] is not None else "",
                f"{row['new_ms']:.6f}" if row["new_ms"] is not None else "",
                row["decision"],
                f"{row['chosen_ms']:.6f}" if row["chosen_ms"] is not None else "",
                f"{row['gain_vs_current']:.4f}",
                (cur[ranks[0]][1].get(key) or ["", "", "", ""])[2],
                (new[ranks[0]][1].get(key) or ["", "", "", ""])[2],
            ])
    (args.out / "decisions.json").write_text(json.dumps(decisions, indent=1))
    print(f"wrote {args.out}/tunableop_results{{{' ,'.join(map(str, ranks))}}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
