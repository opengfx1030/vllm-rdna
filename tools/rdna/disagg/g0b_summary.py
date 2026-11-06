#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markdown tables of the G0b runs (output of g0b_driver.sh / g0b_run.sh).

  g0b_summary.py ROOT [RUN ...] [--json out.json]

Per cell: initiator-side post -> complete time and bandwidth (the WRITE initiator is
the src process, the READ initiator the dst process), the receiver-side one-way time
of a WRITE (post -> notification), the receiver's byte-exact verification count, the
KFD evicted_ms jump, and the KFD queue counts. Bandwidth is GB/s = 1e9 B/s; GiB/s is
in the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HEADER = (
    "| cell | MiB | win | GB/s median | GB/s mean | GB/s @p95 dur | median ms | p95 ms "
    "| min..max ms | one-way ms (rx) | verified | evict ms | valid | queues src/dst "
    "| backend |"
)
RULE = "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|"


def load(path: Path) -> tuple[dict, dict[str, dict]]:
    info: dict = {}
    cells: dict[str, dict] = {}
    if not path.exists():
        return info, cells
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if "info" in rec:
            info = rec["info"]
        else:
            cells[rec["cell"]] = rec
    return info, cells


def text(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def cell_row(cell: str, s: dict, d: dict) -> dict:
    row: dict = {"cell": cell}
    problem = s.get("error") or d.get("error") or s.get("skipped") or d.get("skipped")
    if problem:
        row["error"] = problem
        return row
    if s.get("op") == "NOTIF":
        row["oneway_us"] = d.get("oneway_stats_us")
        row["rtt_us"] = s.get("rtt_stats_us")
        return row
    init = s if s.get("initiator") else d
    row.update(
        op=s.get("op"),
        layout=s.get("layout"),
        mib=round(s.get("bytes", 0) / 2**20, 1),
        windows=s.get("windows"),
        iters=s.get("iters"),
        gbps_median=init.get("gbps_median"),
        gbps_mean=init.get("gbps_mean"),
        gbps_at_p95_dur=init.get("gbps_at_p95_dur"),
        gib_s_median=init.get("gib_s_median"),
        dur_ms=init.get("dur_stats_ms"),
        oneway_ms=d.get("oneway_stats_ms"),
        verify_ok=d.get("verify_ok"),
        verify_bad=d.get("verify_bad"),
        evicted_jump_ms=[s.get("evicted_jump_ms"), d.get("evicted_jump_ms")],
        valid=bool(s.get("valid")) and bool(d.get("valid")),
        queues_src=(s.get("evidence_after") or {}).get("queues"),
        queues_dst=(d.get("evidence_after") or {}).get("queues"),
        backend=init.get("backend"),
        telemetry=(init.get("telemetry") or [None])[0],
        prep_ms=init.get("prep_ms"),
    )
    return row


def summarize_run(root: Path, name: str) -> dict:
    run = root / name
    src_info, src = load(run / "src.jsonl")
    dst_info, dst = load(run / "dst.jsonl")
    rows = [
        cell_row(c, src.get(c, {}), dst.get(c, {})) for c in (list(src) or list(dst))
    ]
    return {
        "run": name,
        "config": text(run / "config.txt"),
        "src_info": src_info,
        "dst_info": dst_info,
        "rc": "\n".join(text(run / "rc.txt")),
        "dmesg": text(run / "dmesg_summary.txt"),
        "rows": rows,
    }


def queues(q: dict | None) -> str:
    return "-" if not q else f"{q['compute']}c/{q['sdma']}s"


def render_row(r: dict) -> str:
    if "error" in r:
        return f"| {r['cell']} | | | | | | | | | | **{r['error']}** | | | | |"
    if "oneway_us" in r:
        o, t = r["oneway_us"] or {}, r["rtt_us"] or {}
        return (
            f"| {r['cell']} | | | | | | | | | one-way us median {fmt(o.get('median'))} "
            f"p99 {fmt(o.get('p99'))} (min {fmt(o.get('min'))}) | rtt us median "
            f"{fmt(t.get('median'))} p99 {fmt(t.get('p99'))} | | | | |"
        )
    d, ow = r["dur_ms"] or {}, r["oneway_ms"] or {}
    return (
        f"| {r['cell']} | {r['mib']} | {r['windows']} | {fmt(r['gbps_median'])} "
        f"| {fmt(r['gbps_mean'])} | {fmt(r['gbps_at_p95_dur'])} "
        f"| {fmt(d.get('median'), 3)} | {fmt(d.get('p95'), 3)} "
        f"| {fmt(d.get('min'), 3)}..{fmt(d.get('max'), 3)} "
        f"| {fmt(ow.get('median'), 3)} (p95 {fmt(ow.get('p95'), 3)}) "
        f"| {r['verify_ok']}/{r['iters']} bad {r['verify_bad']} "
        f"| {r['evicted_jump_ms']} | {r['valid']} "
        f"| {queues(r['queues_src'])} / {queues(r['queues_dst'])} "
        f"| {r['backend'] or '-'} |"
    )


def render(res: dict) -> str:
    out = [f"### {res['run']}", ""]
    out += ["```", *res["config"][:3], f"rc: {res['rc']}", *res["dmesg"], "```", ""]
    out += [HEADER, RULE]
    out += [render_row(r) for r in res["rows"]]
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args()
    root = Path(a.root)
    names = a.runs or sorted(
        n for n in os.listdir(root) if (root / n / "src.jsonl").exists()
    )
    results = [summarize_run(root, n) for n in names]
    print("\n".join(render(r) for r in results))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
