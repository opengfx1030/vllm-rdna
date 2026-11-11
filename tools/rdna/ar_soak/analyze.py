#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM115
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarise one rdna_ar soak run (stdlib only).

  analyze.py RUN_DIR      # RUN_DIR = the driver's run/ directory

Prints markdown: cells and failures, ITL drift per (shape, concurrency) group
(median per cell vs time: slope per hour, first vs last third), probe and PPL
checks, event counts, KFD eviction jumps with the neighbour windows they fall
in, junction temperatures, host memory low point.
"""

import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import regex as re


def secs(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def rows(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def slope_per_hour(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den * 3600


def main(run: str) -> None:
    soak = os.path.join(run, "soak")
    cells = rows(os.path.join(soak, "cells.tsv"))
    t0 = secs(cells[0]["start"]) if cells else 0

    def rel(hms: str) -> int:
        t = secs(hms) - t0
        return t + 86400 if t < -3600 else t

    print("## Cells\n")
    bad = [
        c
        for c in cells
        if c["rc"] != "0" or c["failed"] != "0" or c["completed"] == "0"
    ]
    walls = [int(c["wall_s"]) for c in cells]
    print(f"- cells: {len(cells)} ({sum(c['ref'] == 'ref' for c in cells)} reference)")
    print(f"- bad cells (rc!=0, failed>0 or none completed): {len(bad)}")
    for c in bad:
        print(
            f"  - {c['cell']} rc={c['rc']} completed={c['completed']} "
            f"failed={c['failed']}"
        )
    if walls:
        print(
            f"- wall per cell: median {statistics.median(walls)} s, max {max(walls)} s"
        )
        print(f"- span: {cells[0]['start']} - {cells[-1]['end']}")
    total_ok = sum(int(c["completed"]) for c in cells)
    total_fail = sum(int(c["failed"]) for c in cells)
    print(f"- requests completed {total_ok}, failed {total_fail}")

    print("\n## ITL drift (median of per-cell median ITL, ms)\n")
    print("| group | cells | first third | last third | change | slope %/h | cv % |")
    print("|---|---|---|---|---|---|---|")
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for c in cells:
        if c["completed"] == "0":
            continue
        groups[(c["shape"], c["conc"])].append((rel(c["start"]), float(c["itl_med"])))
    for key in sorted(groups, key=lambda k: (int(k[0]), int(k[1]))):
        pts = sorted(groups[key])
        if len(pts) < 3:
            continue
        ys = [y for _, y in pts]
        k = max(1, len(ys) // 3)
        a, b = statistics.median(ys[:k]), statistics.median(ys[-k:])
        sl = slope_per_hour([p[0] for p in pts], ys)
        cv = statistics.pstdev(ys) / statistics.mean(ys) * 100
        print(
            f"| {key[0]} c{key[1]} | {len(pts)} | {a:.2f} | {b:.2f} | "
            f"{(b - a) / a * 100:+.1f} % | "
            f"{sl / statistics.mean(ys) * 100:+.2f} | {cv:.1f} |"
        )

    print("\n## Reference cells (ITL median ms by time)\n")
    ref: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for c in cells:
        if c["ref"] == "ref" and c["completed"] != "0":
            ref[re.sub(r"-\d+$", "", c["cell"])].append(
                (rel(c["start"]), float(c["itl_med"]), float(c["ttft_mean"]))
            )
    for name, pts in sorted(ref.items()):
        print(
            f"- {name}: " + ", ".join(f"{t // 60}m:{y:.1f}" for t, y, _ in sorted(pts))
        )

    print("\n## Checks (probe, PPL)\n")
    ck = rows(os.path.join(soak, "checks.tsv"))
    ppls = [float(c["ppl"]) for c in ck if re.fullmatch(r"[0-9.]+", c["ppl"])]
    for c in ck:
        print(
            f"- {c['check']} {c['time']}: probe {c['probe']} ppl {c['ppl']} "
            f"(rc {c['ppl_rc']})"
        )
    if ppls:
        print(
            f"\nPPL min {min(ppls):.2f} max {max(ppls):.2f} "
            f"mean {statistics.mean(ppls):.3f} over {len(ppls)} checks"
        )

    print("\n## Events\n")
    ev = open(os.path.join(soak, "events.log")).read().splitlines()
    kinds: dict[str, int] = defaultdict(int)
    for ln in ev:
        parts = ln.split(" ", 3)
        if len(parts) >= 3:
            kinds[parts[2]] += 1
    for k, n in sorted(kinds.items()):
        print(f"- {k}: {n}")
    fails = sorted(glob.glob(os.path.join(soak, "FAIL.*")))
    print(f"- FAIL markers: {[os.path.basename(f) for f in fails]}")
    try:
        cnt = json.load(open(os.path.join(soak, "counts.json")))
        print(f"- KFD evicted_ms totals per worker: {cnt['evict_total_ms']}")
        print(f"- max single jump per worker (ms): {cnt['evict_max_jump_ms']}")
    except (OSError, ValueError):
        pass
    print("\nEviction jumps:\n")
    for ln in ev:
        if " kfd_evict " in ln:
            print(f"- {ln}")
    print("\nNeighbour and boot.lock timeline:\n")
    for ln in ev:
        if re.search(r" (neighbour_up|neighbour_gone|boot_lock|lease) ", ln):
            print(f"- {ln[:230]}")

    print("\n## Junction temperatures (1 s samples, C)\n")
    mx: dict[str, float] = defaultdict(float)
    n = 0
    for ln in open(os.path.join(run, "temps1s.log")):
        n += 1
        for tok in ln.split()[1:]:
            b, v = tok.split("=")
            mx[b] = max(mx[b], float(v))
    print(f"- {n} samples; max per GPU: {dict(mx)}")

    print("\n## Host memory\n")
    lows = []
    for r in csv.DictReader(open(os.path.join(soak, "meminfo.csv"))):
        lows.append((int(r["memfree_kb"]), r["time"]))
    if lows:
        lo = min(lows)
        print(f"- lowest MemFree {lo[0] / 1048576:.1f} GiB at {lo[1]}")


if __name__ == "__main__":
    main(sys.argv[1])
