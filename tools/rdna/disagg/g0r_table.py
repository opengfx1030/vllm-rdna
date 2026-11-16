#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markdown tables of the DisaggRework comparison files (stdlib only).

For every decode arm ``arms/<arm>/vs_base_ds.json`` (against the baseline's cold run)
and ``vs_base_ds_hit.json`` (against the baseline's own prefix-hit run, the
like-for-like tail recompute) one table with a row per prompt; the baseline controls
come from ``arms/base_ds/aa.json`` (cold repeated after a cache reset; prefix hit
against cold).

  g0r_table.py --root R [--arms consumer_mono,consumer]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HEADER = (
    "| prompt | N | tokens | first-token top-20 bitwise | first-token max d "
    "| logprobs bitwise (else max d) | top-2 gap at divergence |\n"
    "|---|---|---|---|---|---|---|"
)


def load(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cell(r: dict[str, Any]) -> str:
    if "error" in r:
        return "ERROR " + str(r["error"])[:40]
    first = r["first_top_max_d"]
    bitwise = (
        "yes" if r["bitwise_logprobs"] else format(r["max_dlogprob_on_prefix"], ".4f")
    )
    gap = r.get("divergence_top2_gap_ref")
    gap_text = ""
    if "ref_token" in r:
        gap_text = "gap -" if gap is None else "gap " + format(gap, ".4f")
    return " | ".join(
        [
            f"{r['identical_prefix']}/64",
            "yes" if r["first_top_bitwise"] else "no",
            "-" if first is None else format(first, ".4f"),
            bitwise,
            gap_text,
        ]
    )


def table(title: str, rows: list[dict[str, Any]] | None) -> str:
    if not rows:
        return title + ": not available\n"
    lines = ["**" + title + "**", "", HEADER]
    for r in rows:
        lines.append(f"| {r['id']} | {r.get('length', '-')} | {cell(r)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="driver result dir (R)")
    ap.add_argument("--arms", default="consumer_mono,consumer")
    ap.add_argument("--base", default="base_ds")
    args = ap.parse_args()
    arms = os.path.join(args.root, "arms")
    aa = load(os.path.join(arms, args.base, "aa.json")) or {}
    controls = {
        "A/A'": "baseline control: cold run repeated after a cache reset vs cold",
        "prefix-hit": "baseline control: prefix-cache hit (tail recompute) vs cold",
    }
    for label, title in controls.items():
        print(table(title, aa.get(label)))
    for arm in args.arms.split(","):
        for name, what in (
            ("vs_base_ds.json", "against the baseline's cold run"),
            ("vs_base_ds_hit.json", "against the baseline's prefix-hit run"),
        ):
            print(table(f"{arm} {what}", load(os.path.join(arms, arm, name))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
