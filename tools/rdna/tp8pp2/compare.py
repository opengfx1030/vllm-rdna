#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Side-by-side and ABBA tables for harness arms.

  compare.py TREE ARMS_ROOT label=arm_dir_name ... [--abba A=a1,a2 B=b1,b2]

TREE supplies tools/rdna/port_v031/spec_acceptance.py. ARMS_ROOT holds one directory
per arm with cells.tsv, cells/*.json and serve.log (hook.sh). Prints a markdown table
per cell (arms as columns) and, with --abba, the paired means with the spread of each
pair. A difference counts as real only when it is larger than the spread inside both
pairs (|a1-a2| and |b1-b2|); smaller ones are reported as "within spread".
"""

import json
import os
import sys
from pathlib import Path

tree, root = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
sys.path.insert(0, os.path.join(tree, "tools/rdna/port_v031"))
import spec_acceptance as sa  # noqa: E402

abba = {}
labels = {}
if "--abba" in args:
    i = args.index("--abba")
    for spec in args[i + 1 :]:
        k, _, v = spec.partition("=")
        abba[k] = v.split(",")
    args = args[:i]
for spec in args:
    k, _, v = spec.partition("=")
    labels[k] = v

CELLS = ["1024x512-c1", "1024x512-c8", "16384x1024-c1", "16384x1024-c8"]
METRICS = [
    ("ttft_mean", "TTFT mean s", 1e-3, False),
    ("ttft_med", "TTFT p50 s", 1e-3, False),
    ("ttft_p99", "TTFT p99 s", 1e-3, False),
    ("itl_med", "ITL p50 ms", 1.0, False),
    ("itl_p99", "ITL p99 ms", 1.0, False),
    ("tps", "out tok/s", 1.0, True),
    ("pf_req", "prefill tok/s per req", 1.0, True),
    ("pf_agg", "prefill agg tok/s", 1.0, True),
    ("acc", "MTP acc len", 1.0, True),
]


def cell(arm: str, name: str):
    d = Path(root) / arm
    tsv = d / "cells.tsv"
    if not tsv.is_file():
        return None
    rows = [r.split("\t") for r in tsv.read_text().splitlines() if r.strip()]
    rows = [r for r in rows if r[1] == f"bench-{name}" and "INVALID" not in r[6]]
    if not rows:
        return None
    cname, _, _try, _seed, t0, t1, _kfd = rows[-1][:7]
    j = json.loads((d / "cells" / f"{cname}.json").read_text())
    done = max(int(j.get("completed", 0)), 1)
    in_len = j["total_input_tokens"] / done
    wave = min(int(j.get("max_concurrency") or 1), done)
    p99 = j["p99_ttft_ms"] / 1000.0
    ttft = j["mean_ttft_ms"] / 1000.0
    with open(d / "serve.log", errors="replace") as f:
        acc = sa.acceptance(sa.parse_log(f), t0, t1)
    return {
        "ttft_mean": j["mean_ttft_ms"],
        "ttft_med": j["median_ttft_ms"],
        "ttft_p99": j["p99_ttft_ms"],
        "itl_med": j["median_itl_ms"],
        "itl_p99": j["p99_itl_ms"],
        "tps": j["output_throughput"],
        "pf_req": in_len / ttft if ttft > 0 else float("nan"),
        "pf_agg": wave * in_len / p99 if p99 > 0 else float("nan"),
        "acc": acc["length"],
    }


data = {k: {c: cell(v, c) for c in CELLS} for k, v in labels.items()}
for grp in abba.values():
    for arm in grp:
        data.setdefault(arm, {c: cell(arm, c) for c in CELLS})


def fmt(v, scale):
    return "-" if v is None else f"{v * scale:.2f}"


for c in CELLS:
    print(f"### {c}\n")
    keys = list(labels)
    print("| metric | " + " | ".join(keys) + " |")
    print("|---|" + "---:|" * len(keys))
    for m, label, scale, _hib in METRICS:
        vals = [data[k][c][m] if data[k][c] else None for k in keys]
        print(f"| {label} | " + " | ".join(fmt(v, scale) for v in vals) + " |")
    print()

if abba:
    names = list(abba)
    a, b = names[0], names[1]
    print(f"## ABBA: {a} = {abba[a]}, {b} = {abba[b]}\n")
    print(
        f"| cell | metric | {a} mean | {b} mean | {b} vs {a} "
        f"| spread {a} | spread {b} | verdict |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for c in CELLS:
        for m, label, scale, hib in METRICS:
            av = [data[x][c][m] if data[x][c] else None for x in abba[a]]
            bv = [data[x][c][m] if data[x][c] else None for x in abba[b]]
            if None in av or None in bv:
                continue
            am, bm = sum(av) / len(av), sum(bv) / len(bv)
            sa_ = abs(av[0] - av[-1]) / am * 100 if len(av) > 1 and am else 0.0
            sb_ = abs(bv[0] - bv[-1]) / bm * 100 if len(bv) > 1 and bm else 0.0
            delta = (bm - am) / am * 100 if am else float("nan")
            better = delta > 0 if hib else delta < 0
            real = abs(delta) > max(sa_, sb_)
            verdict = ("better" if better else "worse") if real else "within spread"
            print(
                f"| {c} | {label} | {am * scale:.2f} | {bm * scale:.2f} | "
                f"{delta:+.1f} % | {sa_:.1f} % | {sb_:.1f} % | {verdict} |"
            )
    print()
