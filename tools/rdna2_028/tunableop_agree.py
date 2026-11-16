#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make the per-rank TunableOp row files of a profile agree on the solver of every key.

A TP worker on device ordinal ``r`` reads ``tunableop_results<r>.csv``. A replicated
GEMM (the MoE router, the hyper-connection and PLE projections) multiplies the same
operands on every rank, so every rank has to run the same rocBLAS solution or the
accumulation orders differ and the logits differ in the last bit. The rows were tuned
rank by rank, and for 531 of the 3330 keys of ``rocm7.14-rocblas5.5`` the files named
different solvers (the router GEMM at 1023 rows: ``tn_512_1023_2560_ld_2560_2560_512``
was ``...046`` on ranks 0, 1 and 3 and ``...044`` on rank 2, so rank 2 flipped near-tie
top-k decisions in every pure 1023-row prefill pass; see
docs/rdna2/tunableop-rank-agreement-2026-11-16.md).

    tunableop_agree.py check tunableop/rocm7.14-rocblas5.5 [more profile dirs]
    tunableop_agree.py write tunableop/rocm7.14-rocblas5.5 [--out DIR]

``check`` lists the keys whose solver differs between rank files (exit 1 if any, also
for a key that is missing from some file). ``write`` gives every rank the solver that
was fastest where it was measured (the 4th column; ties go to the lower rank): only the
lines whose solver loses are replaced, by the winning line verbatim, so the files keep
their order and the rows that already agree keep their own timing column. No key is
added or dropped, so a row that is left out on purpose stays out.
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

RANK_FILE = "tunableop_results{}.csv"


@dataclass
class Row:
    line: int  # index into the file's lines
    text: str
    solver: str
    ms: float


def rank_files(profile: Path) -> dict[int, Path]:
    found = {}
    for path in glob.glob(str(profile / "tunableop_results*.csv")):
        stem = Path(path).stem
        digits = stem[len("tunableop_results") :]
        if digits.isdigit():
            found[int(digits)] = Path(path)
    return dict(sorted(found.items()))


def parse(lines: list[str]) -> tuple[list[str], dict[tuple[str, str], Row]]:
    """Validator lines and the rows of one file, keyed by (op, shape)."""
    validators: list[str] = []
    rows: dict[tuple[str, str], Row] = {}
    for i, text in enumerate(lines):
        if text.startswith("Validator,"):
            validators.append(text)
            continue
        fields = text.split(",")
        if len(fields) < 3:
            continue
        try:
            ms = float(fields[3]) if len(fields) > 3 else float("inf")
        except ValueError:
            ms = float("inf")
        rows[(fields[0], fields[1])] = Row(i, text, fields[2], ms)
    return validators, rows


def load(profile: Path) -> tuple[dict[int, list[str]], dict[int, dict]]:
    texts = {r: p.read_text().splitlines() for r, p in rank_files(profile).items()}
    return texts, {r: parse(t)[1] for r, t in texts.items()}


def disagreements(rows: dict[int, dict]) -> tuple[dict, dict]:
    """(keys whose solver differs between ranks, keys missing from some rank)."""
    keys = set().union(*[set(r) for r in rows.values()]) if rows else set()
    differ, missing = {}, {}
    for key in sorted(keys):
        have = {r: rows[r][key] for r in rows if key in rows[r]}
        if len(have) != len(rows):
            missing[key] = sorted(set(rows) - set(have))
        elif len({row.solver for row in have.values()}) > 1:
            differ[key] = have
    return differ, missing


def winner(have: dict[int, Row]) -> int:
    """The rank whose row to keep: lowest measured time, then the lowest rank."""
    return min(have, key=lambda r: (have[r].ms, r))


def agree(profile: Path, out: Path | None = None) -> dict[str, int]:
    """Rewrite the losing lines of every disagreeing key; returns the counts."""
    texts, rows = load(profile)
    differ, missing = disagreements(rows)
    if missing:
        raise SystemExit(
            f"{profile}: {len(missing)} keys are missing from some rank file"
        )
    rewritten = 0
    for key, have in differ.items():
        best = have[winner(have)]
        for r, row in have.items():
            if row.solver != best.solver:
                texts[r][row.line] = best.text
                rewritten += 1
    target = out or profile
    target.mkdir(parents=True, exist_ok=True)
    for r, lines in texts.items():
        (target / RANK_FILE.format(r)).write_text("\n".join(lines) + "\n")
    return {"keys": len(differ), "lines": rewritten, "files": len(texts)}


def check(profile: Path) -> int:
    texts, rows = load(profile)
    if len(rows) < 2:
        print(f"{profile}: {len(rows)} rank file(s), nothing to compare")
        return 0
    validators = {tuple(parse(t)[0]) for t in texts.values()}
    differ, missing = disagreements(rows)
    n = len(next(iter(rows.values())))
    print(
        f"{profile}: ranks {sorted(rows)}, {n} rows in rank {min(rows)}, "
        f"{len(differ)} keys with different solvers, {len(missing)} missing from some "
        f"rank, {len(validators)} distinct validator header(s)"
    )
    for key, have in list(differ.items())[:20]:
        print(
            "   ", key[1], {r: row.solver.rsplit("_", 1)[-1] for r, row in have.items()}
        )
    return int(bool(differ or missing or len(validators) > 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("profile", nargs="+", type=Path)
    w = sub.add_parser("write")
    w.add_argument("profile", type=Path)
    w.add_argument(
        "--out", type=Path, help="write the agreed files here (default: in place)"
    )
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return max(check(p) for p in args.profile)
    done = agree(args.profile, args.out)
    print(
        f"{args.profile}: {done['keys']} keys, {done['lines']} lines rewritten in "
        f"{done['files']} files -> {args.out or args.profile}"
    )
    if args.out is not None:
        for name in ("provenance.json", "README.md"):
            src = args.profile / name
            if src.is_file() and not (args.out / name).exists():
                (args.out / name).write_bytes(src.read_bytes())
    return 0


if __name__ == "__main__":
    os.umask(0o022)
    sys.exit(main())
