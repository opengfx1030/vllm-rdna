# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare two FP16 reference arms of the same recipe (e.g. across a rebase).

The FP16 reference is the baseline every INT4 delta is measured against, so a
rebase that quietly moved it would move the headline numbers with it. This
tool compares two FP16 arms cell by cell and reports the relative change of
every metric, so a rebase can be shown not to have shifted the baseline.

Usage: python compare_baselines.py <baseline-dir> <candidate-dir> [-o out.json]
"""

import argparse
import json
from pathlib import Path

from compare_arms import CELLS, KEYS


def load(arm: Path):
    metrics = json.loads((arm / "metrics.json").read_text())
    run = json.loads((arm / "run.json").read_text())
    status = json.loads((arm / "status.json").read_text())
    return {
        "metrics": metrics,
        "run": run,
        "status": status,
        "core_sha256": core_sha256(arm),
        "path": str(arm),
    }


def core_sha256(arm: Path) -> str | None:
    """The SHA256 of the _rocm_C that served the arm, from its own manifest."""
    manifest_file = arm / "manifest.before.json"
    if not manifest_file.exists():
        return None
    manifest = json.loads(manifest_file.read_text())
    return manifest.get("vllm/_rocm_C.abi3.so")


def build(baseline, candidate):
    rows = []
    for cell in CELLS:
        if cell not in baseline["metrics"] or cell not in candidate["metrics"]:
            continue
        base = baseline["metrics"][cell]
        cand = candidate["metrics"][cell]
        row = {"cell": cell}
        for key, label in KEYS:
            base_value, cand_value = base[key], cand[key]
            row[label] = {
                "baseline": base_value,
                "candidate": cand_value,
                "ratio": (cand_value / base_value) if base_value else None,
                "delta_pct": ((cand_value - base_value) / base_value * 100.0)
                if base_value
                else None,
            }
        rows.append(row)
    return {
        "baseline": {
            "dir": baseline["path"],
            "boot_id": baseline["run"]["boot_id"],
            "raw_status": baseline["status"],
            "core_sha256": baseline["core_sha256"],
        },
        "candidate": {
            "dir": candidate["path"],
            "boot_id": candidate["run"]["boot_id"],
            "raw_status": candidate["status"],
            "core_sha256": candidate["core_sha256"],
        },
        "same_boot": baseline["run"]["boot_id"] == candidate["run"]["boot_id"],
        "cells": rows,
    }


def markdown(report):
    lines = [
        "| cell | metric | baseline | candidate | ratio | delta % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["cells"]:
        for _, label in KEYS:
            entry = row[label]
            ratio = f"{entry['ratio']:.4f}" if entry["ratio"] is not None else "n/a"
            delta = (
                f"{entry['delta_pct']:+.2f}%"
                if entry["delta_pct"] is not None
                else "n/a"
            )
            lines.append(
                f"| {row['cell']} | {label} | {entry['baseline']:.3f} "
                f"| {entry['candidate']:.3f} | {ratio} | {delta} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    report = build(load(args.baseline), load(args.candidate))
    # The widest relative change over every metric of every cell: one number
    # that says whether the baseline moved.
    deltas = [
        abs(entry["delta_pct"])
        for row in report["cells"]
        for _, label in KEYS
        if (entry := row[label])["delta_pct"] is not None
    ]
    report["max_abs_delta_pct"] = max(deltas) if deltas else None
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print()
    print(markdown(report))


if __name__ == "__main__":
    main()
