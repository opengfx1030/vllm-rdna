# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare captured TunableOp shape traces against the frozen row sets.

record_untuned (PYTORCH_TUNABLEOP_RECORD_UNTUNED=1) records EVERY GemmTunableOp
half-precision call the process makes, regardless of whether a row exists, so
the trace is a full observed-shape census. Cross-referencing it with the
shipped rows tells us:

  * covered      -- key already in the curated row file (lookup will hit),
  * dropped       -- observed, present in an earlier (harvested) set but not in
                     the curated set: deliberately dropped because it reproduced
                     the heuristic or measured slower. Not a gap.
  * novel         -- observed in no row file we have ever shipped: a genuinely
                     uncaptured shape family and a tuning candidate.

    analyze_captures.py --captures <dir> --rows <curated0.csv> \
        [--backup <harvested0.csv>] [--out novel.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def keys_in_file(path: Path):
    keys = set()
    if not path or not path.exists():
        return keys
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and (parts[1].startswith("tn_") or parts[1].startswith("nn_")):
            keys.add(parts[1])
    return keys


def load_capture(p: Path):
    """shapes_*.txt are one bare key per line; untuned*.csv have a header col."""
    keys = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split(",")[-1]
        if tok.startswith(("tn_", "nn_")):
            keys.add(tok)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", required=True, type=Path,
                    help="dir holding shapes_*.txt (and/or untuned*.csv)")
    ap.add_argument("--rows", required=True, type=Path, help="curated rows0.csv")
    ap.add_argument("--backup", type=Path, default=None,
                    help="previously-shipped (harvested) rows0.csv")
    ap.add_argument("--out", type=Path, default=None, help="write novel keys")
    args = ap.parse_args()

    curated = keys_in_file(args.rows)
    harvested = keys_in_file(args.backup) if args.backup else set()
    print(f"curated rows : {len(curated)}")
    if args.backup:
        print(f"harvested rows: {len(harvested)}")

    per_arm = {}
    def skip(f):
        return any(part.startswith("_") for part in f.relative_to(args.captures).parts)

    for f in sorted(args.captures.rglob("shapes_*.txt")):
        if skip(f):
            continue
        arm = f.parent.name if f.parent != args.captures else f.stem[len("shapes_"):]
        per_arm.setdefault(arm, set()).update(load_capture(f))
    for f in sorted(args.captures.rglob("untuned[0-9].csv")):
        if skip(f):
            continue
        arm = f.parent.name
        per_arm.setdefault(arm, set()).update(load_capture(f))

    if not per_arm:
        print("no capture files found under", args.captures)
        return 1

    union = set().union(*per_arm.values())
    print(f"\n{'arm':28s} {'observed':>9s} {'covered':>8s} {'dropped':>8s} {'novel':>7s}")
    for arm in sorted(per_arm):
        obs = per_arm[arm]
        cov = obs & curated
        drp = (obs & harvested) - curated
        nov = obs - curated - harvested
        print(f"{arm:28s} {len(obs):9d} {len(cov):8d} {len(drp):8d} {len(nov):7d}")

    print("\npairwise set equality (by MTP):")
    import itertools
    for grp in ("m0", "m2"):
        arms = sorted(a for a in per_arm if a.endswith(grp))
        for a, b in itertools.combinations(arms, 2):
            same = per_arm[a] == per_arm[b]
            only = len(per_arm[a] ^ per_arm[b])
            print(f"  {a} == {b}: {same} (symdiff {only})")

    covered = union & curated
    dropped = (union & harvested) - curated
    novel = sorted(union - curated - harvested)
    print(f"\n{'UNION':28s} {len(union):9d} {len(covered):8d} {len(dropped):8d} {len(novel):7d}")
    print("\nnovel (not in curated and not in harvested):")
    for k in novel:
        print("  ", k)
    if args.out:
        args.out.write_text("\n".join(novel) + ("\n" if novel else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
