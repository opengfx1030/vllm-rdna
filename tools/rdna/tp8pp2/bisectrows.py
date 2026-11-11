#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Row-subset trees for bisecting a TunableOp rows commit (python3, no dependencies).

    bisectrows.py ROWS_DIR NEW_KEYS OUT_ROOT NAME=SPEC[+SPEC...] [NAME=SPEC ...]

ROWS_DIR: a rows tree (profiles.json, rocm*/tunableop_results<rank>.csv) that holds
the shipped rows plus the new ones. NEW_KEYS: the `op,key` lines of the new rows. For
each NAME the script writes OUT_ROOT/NAME, a copy of ROWS_DIR whose rank files keep
the Validator lines, every row that is not new and the new rows the SPECs select (the
union). Point TUNABLEOP_ROOT of an arm at OUT_ROOT/NAME.

SPEC: none | all | nn | tn_s (n <= 64) | tn_m (65 <= n <= 256) | tn_b (n > 256)
      | bit:B:GROUP      rows of GROUP sorted by key whose index has bit B set
      | idx:GROUP:LO:HI  rows of GROUP sorted by key, index range [LO, HI)
GROUP is one of the names above except none. n is the third field of the key
(`tn_<m>_<n>_<k>_ld_...`), the token count of the GEMM.

Single-defective group testing: with the rows of one group sorted, the arms
bit:0 .. bit:ceil(log2(len)) - 1 fail exactly where the culprit's index has the bit
set, so the failing pattern is the culprit's index in binary.
"""

import shutil
import sys
from pathlib import Path

GROUPS = ("all", "nn", "tn_s", "tn_m", "tn_b")


def in_group(row: tuple[str, str], name: str) -> bool:
    op, key = row
    n = int(key.split("_")[2])
    return {
        "all": True,
        "nn": op.endswith("_NN"),
        "tn_s": op.endswith("_TN") and n <= 64,
        "tn_m": op.endswith("_TN") and 64 < n <= 256,
        "tn_b": op.endswith("_TN") and n > 256,
    }[name]


def select(new: list[tuple[str, str]], spec: str) -> set[tuple[str, str]]:
    if spec == "none":
        return set()
    if spec in GROUPS:
        return {r for r in new if in_group(r, spec)}
    kind, _, rest = spec.partition(":")
    if kind == "bit":
        bit, _, name = rest.partition(":")
        rows = sorted(r for r in new if in_group(r, name))
        return {r for i, r in enumerate(rows) if i >> int(bit) & 1}
    if kind == "idx":
        name, lo, hi = rest.split(":")
        rows = sorted(r for r in new if in_group(r, name))
        return set(rows[int(lo) : int(hi)])
    raise SystemExit(f"unknown spec {spec!r}")


def main() -> None:
    rows_dir, new_keys, out_root = (Path(a) for a in sys.argv[1:4])
    new = [
        (op, key)
        for op, key in (
            line.split(",", 1) for line in new_keys.read_text().splitlines() if line
        )
    ]
    all_new = set(new)
    for arg in sys.argv[4:]:
        name, _, specs = arg.partition("=")
        keep: set[tuple[str, str]] = set()
        for spec in specs.split("+"):
            keep |= select(new, spec)
        out = out_root / name
        shutil.copytree(rows_dir, out)
        for csv in sorted(out.rglob("tunableop_results*.csv")):
            kept = []
            for line in csv.read_text().splitlines(keepends=True):
                parts = line.split(",", 2)
                row = (parts[0], parts[1]) if len(parts) > 2 else None
                if row in all_new and row not in keep:
                    continue
                kept.append(line)
            csv.write_text("".join(kept))
        print(f"{name}: {len(keep)} of {len(new)} new rows -> {out}")


if __name__ == "__main__":
    main()
