#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline TunableOp missing-GEMM set logic and harvest helper.

The vLLM serve stack resolves GEMM solutions from hash-keyed row sets under
``tunableop/<profile>/tunableop_results{0..3}.csv``.  A census produced with
``PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`` lists every GEMM signature the model
actually called.  This module compares the two, extracts the missing keys,
merges freshly tuned rows back into a profile, and emits lookup-only
verification snippets.  It is stdlib-only and never imports torch, so it runs
on a laptop without the ROCm stack.

Sub-commands:
  missing    print census keys absent from a profile's rank-0 table
  untuned    write the missing keys to an untuned file for the offline tuner
  harvest    merge a tuned file into all four per-rank tables of a profile
  verify     emit a lookup-only verification script
  count      print row counts per rank file
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

TREE = Path(__file__).resolve().parents[3]
DEFAULT_ROWS_DIR = TREE / "tunableop"
DEFAULT_PROFILE = "rocm7.14-rocblas5.5"
RANKS = (0, 1, 2, 3)

VERIFY_TEMPLATE = """\
import os
import sys

os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "1"
os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = {rows_file!r}
os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = {untuned_file!r}

import torch
import torch.cuda.tunable as tunable
from torch.cuda.tunable import _process_single_offline_gemm

torch.accelerator.set_device_index({device})
tunable.set_filename({rows_file!r}, insert_device_ordinal=False)
tunable.enable(True)
tunable.tuning_enable(False)

with open({census_file!r}) as fh:
    lines = [
        line.strip()
        for line in fh
        if line.strip().startswith(("Gemm", "ScaledGemm"))
    ]

for line in lines:
    _process_single_offline_gemm(line, {device})

tunable.enable(False)

untuned_path = {untuned_file!r}
if os.path.exists(untuned_path):
    with open(untuned_path) as fh:
        gemm_lines = [line for line in fh if line.startswith("Gemm")]
    if gemm_lines:
        print(
            "FAIL: untuned record is not empty: %d lines" % len(gemm_lines),
            file=sys.stderr,
        )
        sys.exit(1)
print("PASS: untuned record is empty", file=sys.stderr)
"""


def read_lines(path):
    """Return the raw lines of a file, preserving order."""
    with open(path, encoding="utf-8", newline="") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def split_rows(lines):
    """Split raw lines into (validators, rows)."""
    validators, rows = [], []
    for line in lines:
        if line.startswith("Validator,"):
            validators.append(line)
        else:
            rows.append(line)
    return validators, rows


def key_of(line):
    """Return the (op_signature, params_signature) tuple of a CSV line."""
    parts = line.split(",")
    if len(parts) < 2:
        return None
    return (parts[0], parts[1])


def is_noise(key):
    """16x16x16 warm-up shapes are census noise."""
    return "_16_16_16_" in key[1]


def is_strided(key):
    """Strided-batched fp32 keys."""
    return key[0].startswith("GemmStridedBatched")


def load_profile(rows_dir, profile):
    """Load profiles.json and return the profile entry."""
    registry_path = rows_dir / "profiles.json"
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)
    profiles = registry.get("profiles", {})
    if profile not in profiles:
        raise SystemExit(f"profile {profile!r} not found in {registry_path}")
    return profiles[profile]


def profile_dir(rows_dir, profile):
    """Return the on-disk directory of a profile."""
    entry = load_profile(rows_dir, profile)
    return rows_dir / entry["dir"]


def table_path(rows_dir, profile, rank):
    """Return the per-rank results file path."""
    return profile_dir(rows_dir, profile) / f"tunableop_results{rank}.csv"


def find_missing(census_path, table_path_):
    """Return (supported, noise, strided) sets of missing keys."""
    census_keys = set()
    for line in read_lines(census_path):
        key = key_of(line)
        if key:
            census_keys.add(key)
    table_keys = set()
    for line in read_lines(table_path_):
        key = key_of(line)
        if key:
            table_keys.add(key)
    missing = census_keys - table_keys
    noise = {k for k in missing if is_noise(k)}
    strided = {k for k in missing if is_strided(k) and not is_noise(k)}
    supported = missing - noise - strided
    return supported, noise, strided


def cmd_missing(args):
    """Print missing keys and classify them."""
    supported, noise, strided = find_missing(
        args.census, table_path(args.rows_dir, args.profile, 0)
    )
    total = len(supported) + len(noise) + len(strided)
    print(f"census: {args.census}", file=sys.stderr)
    print(
        f"table:  {table_path(args.rows_dir, args.profile, 0)}",
        file=sys.stderr,
    )
    print(
        f"missing: {total} keys "
        f"({len(supported)} tunable, {len(noise)} noise, "
        f"{len(strided)} strided-batched)",
        file=sys.stderr,
    )
    for op, key in sorted(supported):
        print(f"{op},{key}")
    for op, key in sorted(noise):
        print(f"{op},{key}")
    for op, key in sorted(strided):
        print(f"{op},{key}")
    if noise:
        print("noise keys (excluded from tuning):", file=sys.stderr)
        for op, key in sorted(noise):
            print(f"  {op},{key}", file=sys.stderr)
    if strided:
        print(
            "strided-batched keys (excluded from tuning by default):",
            file=sys.stderr,
        )
        for op, key in sorted(strided):
            print(f"  {op},{key}", file=sys.stderr)
    if args.dry_run:
        print("dry-run: no files written", file=sys.stderr)
    return 0


def cmd_untuned(args):
    """Write the missing keys to an untuned file."""
    supported, noise, strided = find_missing(
        args.census, table_path(args.rows_dir, args.profile, 0)
    )
    keys = set(supported)
    if args.include_strided:
        keys |= strided
    if args.include_noise:
        keys |= noise
    keys = sorted(keys)
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if count < 1 or not 0 <= index < count:
            print(
                f"bad --shard {args.shard!r}: want I/N with 0 <= I < N", file=sys.stderr
            )
            return 1
        keys = keys[index::count]
    print(f"untuned: {len(keys)} keys -> {args.out}", file=sys.stderr)
    if args.dry_run:
        print("dry-run: not writing", file=sys.stderr)
        return 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for op, key in keys:
            fh.write(f"{op},{key}\n")
    print(f"wrote {len(keys)} keys to {out_path}", file=sys.stderr)
    return 0


def _update_metadata(rows_dir, profile, new_rows):
    """Update provenance.json and profiles.json after a harvest."""
    target_dir = profile_dir(rows_dir, profile)
    prov_path = target_dir / "provenance.json"
    with open(prov_path, encoding="utf-8") as fh:
        prov = json.load(fh)
    old_count = prov.get("rows_per_rank", 0)
    new_keys = []
    for row in new_rows:
        key = key_of(row)
        if key:
            new_keys.append(f"{key[0]},{key[1]}")
    rank0_path = target_dir / "tunableop_results0.csv"
    _, rows = split_rows(read_lines(rank0_path))
    new_count = len(rows)
    prov["missing_gemms"] = {
        "date": date.today().isoformat(),
        "tool": "tools/rdna/prof/tune_missing_gemms.sh",
        "shape_list": new_keys,
    }
    prov["rows_per_rank"] = new_count
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")

    registry_path = rows_dir / "profiles.json"
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)
    registry["profiles"][profile]["rows_per_rank"] = new_count
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
        fh.write("\n")

    print(
        f"README updates needed: **{old_count}** -> **{new_count}**",
        file=sys.stderr,
    )
    print(f"  {rows_dir / 'README.md'}", file=sys.stderr)
    print(f"  {target_dir / 'README.md'}", file=sys.stderr)
    return new_count


def cmd_harvest(args):
    """Merge tuned rows into all four per-rank tables."""
    tuned_validators = None
    tuned_rows = []
    for tuned_path in args.tuned:
        validators, rows = split_rows(read_lines(tuned_path))
        if tuned_validators is None:
            tuned_validators = validators
        elif validators != tuned_validators:
            print(
                f"Validator mismatch between tuned files: {tuned_path}", file=sys.stderr
            )
            return 1
        tuned_rows.extend(rows)
    target_dir = profile_dir(args.rows_dir, args.profile)
    rank0_path = target_dir / "tunableop_results0.csv"
    rank0_validators, _ = split_rows(read_lines(rank0_path))
    if tuned_validators != rank0_validators:
        print("Validator mismatch: tuned file validators differ from", file=sys.stderr)
        print(f"  tuned:  {tuned_validators}", file=sys.stderr)
        print(f"  rank 0: {rank0_validators}", file=sys.stderr)
        print("Refusing to merge.", file=sys.stderr)
        return 1

    tuned_keys = set()
    new_rows = []
    for row in tuned_rows:
        key = key_of(row)
        if key is None or key in tuned_keys:
            continue
        tuned_keys.add(key)
        new_rows.append(row)

    total_new = 0
    total_skipped = 0
    for rank in RANKS:
        rank_path = target_dir / f"tunableop_results{rank}.csv"
        validators, rows = split_rows(read_lines(rank_path))
        existing_keys = {key_of(row) for row in rows if key_of(row)}
        rank_new = []
        rank_skipped = 0
        for row in new_rows:
            key = key_of(row)
            if key in existing_keys:
                rank_skipped += 1
                continue
            rank_new.append(row)
        total_new += len(rank_new)
        total_skipped += rank_skipped
        if not args.dry_run:
            with open(rank_path, "w", encoding="utf-8", newline="") as fh:
                for v in validators:
                    fh.write(f"{v}\n")
                for row in rows:
                    fh.write(f"{row}\n")
                for row in rank_new:
                    fh.write(f"{row}\n")
        print(
            f"rank {rank}: +{len(rank_new)} rows, {rank_skipped} skipped",
            file=sys.stderr,
        )

    unique_skipped = len(tuned_rows) - len(new_rows)
    unique_word = "duplicate" if unique_skipped == 1 else "duplicates"
    total_word = "duplicate" if total_skipped == 1 else "duplicates"
    print(
        f"unique: {len(new_rows)} new rows, {unique_skipped} skipped {unique_word}",
        file=sys.stderr,
    )
    print(
        f"total: {total_new} new rows, "
        f"{total_skipped} skipped {total_word} across {len(RANKS)} ranks",
        file=sys.stderr,
    )
    if args.dry_run:
        print("dry-run: no files written", file=sys.stderr)
        return 0

    _update_metadata(args.rows_dir, args.profile, new_rows)
    return 0


def cmd_verify(args):
    """Emit a lookup-only verification script."""
    target_dir = profile_dir(args.rows_dir, args.profile)
    rows_file = target_dir / "tunableop_results0.csv"
    untuned_file = args.out / "untuned_verify.csv"
    script_path = args.out / "verify_lookup.py"
    script = VERIFY_TEMPLATE.format(
        rows_file=str(rows_file),
        census_file=str(args.census),
        gpu=args.gpu,
        device=0,
        untuned_file=str(untuned_file),
    )
    print(f"verify script: {script_path}", file=sys.stderr)
    print(f"  rows: {rows_file}", file=sys.stderr)
    print(f"  census: {args.census}", file=sys.stderr)
    print(f"  gpu: {args.gpu}", file=sys.stderr)
    print(f"  untuned record: {untuned_file}", file=sys.stderr)
    if args.dry_run:
        print("dry-run: not writing", file=sys.stderr)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    print(f"wrote {script_path}", file=sys.stderr)
    return 0


def cmd_count(args):
    """Print row counts per rank file."""
    target_dir = profile_dir(args.rows_dir, args.profile)
    for rank in RANKS:
        path = target_dir / f"tunableop_results{rank}.csv"
        _, rows = split_rows(read_lines(path))
        print(f"rank {rank}: {len(rows)} rows")
    return 0


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--profile", default=DEFAULT_PROFILE)
        p.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS_DIR)
        p.add_argument("--dry-run", action="store_true")

    p_missing = sub.add_parser("missing")
    p_missing.add_argument("--census", required=True, type=Path)
    add_common(p_missing)
    p_missing.set_defaults(func=cmd_missing)

    p_untuned = sub.add_parser("untuned")
    p_untuned.add_argument("--census", required=True, type=Path)
    p_untuned.add_argument("--out", required=True, type=Path)
    p_untuned.add_argument("--include-strided", action="store_true")
    p_untuned.add_argument(
        "--shard", default="", help="I/N: keep every N-th key starting at I"
    )
    p_untuned.add_argument("--include-noise", action="store_true")
    add_common(p_untuned)
    p_untuned.set_defaults(func=cmd_untuned)

    p_harvest = sub.add_parser("harvest")
    p_harvest.add_argument("--tuned", required=True, type=Path, action="append")
    p_harvest.add_argument("--profile", required=True)
    p_harvest.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS_DIR)
    p_harvest.add_argument("--dry-run", action="store_true")
    p_harvest.set_defaults(func=cmd_harvest)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--census", required=True, type=Path)
    p_verify.add_argument("--gpu", type=int, required=True)
    p_verify.add_argument("--out", required=True, type=Path)
    add_common(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_count = sub.add_parser("count")
    add_common(p_count)
    p_count.set_defaults(func=cmd_count)

    return parser


def main(argv=None):
    """Run the requested sub-command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
