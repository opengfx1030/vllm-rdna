#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarise the measured cells of arm directories (stdlib + spec_acceptance).

  cellsum.py TREE ARM_DIR [ARM_DIR ...]

Per arm: cells.tsv (name, base, try, seed, start, end, kfd) written by hook.sh, the
bench JSON in cells/<name>.json and serve.log. A cell whose KFD line says INVALID is
skipped (the rerun row of the same base name counts). Prints one markdown table per
arm: TTFT mean/p50/p99, ITL p50/p99, output tok/s, prefill tok/s per request and
aggregate, MTP acceptance length (SpecDecoding metrics inside the cell window).
"""

import json
import os
import sys
from pathlib import Path

tree = sys.argv[1]
sys.path.insert(0, os.path.join(tree, "tools/rdna/port_v031"))
import spec_acceptance as sa  # noqa: E402


def cells(arm_dir: str):
    rows = {}
    tsv = Path(arm_dir) / "cells.tsv"
    if not tsv.exists():
        return []
    for line in tsv.read_text().splitlines():
        p = line.split("\t")
        if len(p) < 7 or "INVALID" in p[6]:
            continue
        rows[p[1]] = p
    return list(rows.values())


def row(arm_dir: str, p, records):
    cname, base, _try, _seed, t0, t1, kfd = p[:7]
    try:
        d = json.loads((Path(arm_dir) / "cells" / f"{cname}.json").read_text())
    except (OSError, ValueError):
        return f"| {base} | unreadable | | | | | | | |"
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    ttft = d.get("mean_ttft_ms", 0.0) / 1000.0
    p50t = d.get("median_ttft_ms", 0.0) / 1000.0
    p99t = d.get("p99_ttft_ms", 0.0) / 1000.0
    wave = min(int(d.get("max_concurrency") or 1), done)
    pre = in_len / ttft if ttft > 0 else 0.0
    pre_agg = wave * in_len / p99t if p99t > 0 else 0.0
    acc = sa.acceptance(records, t0, t1)
    return (
        f"| {base} | {done} | {ttft:.2f} / {p50t:.2f} / {p99t:.2f} "
        f"| {d.get('median_itl_ms', 0):.1f} / {d.get('p99_itl_ms', 0):.1f} "
        f"| {d.get('output_throughput', 0):.1f} | {pre:.0f} | {pre_agg:.0f} "
        f"| {acc['length']:.2f} | {'ok' if 'INVALID' not in kfd else 'INVALID'} |"
    )


for arm_dir in sys.argv[2:]:
    log = os.path.join(arm_dir, "serve.log")
    with open(log, errors="replace") as f:
        records = sa.parse_log(f)
    print(f"### {os.path.basename(arm_dir.rstrip('/'))}")
    print()
    print(
        "| cell | ok | TTFT mean/p50/p99 s | ITL p50/p99 ms | out tok/s "
        "| prefill tok/s/req | prefill agg tok/s | MTP acc len | KFD |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for p in cells(arm_dir):
        print(row(arm_dir, p, records))
    print()
