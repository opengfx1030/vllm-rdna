#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speculative-decoding acceptance length over a bench cell, from the server log.

    python3 spec_acceptance.py SERVE_LOG START END [--bench BENCH.json]

START and END are the cell's ``HH:MM:SS`` on the day of the log (the proof's
``summary.txt`` has them: ``bench bench-1024x512-c1`` and the next line). The
server logs ``SpecDecoding metrics`` every 10 s: the tokens accepted and
drafted in that interval and one acceptance rate per draft position. An
interval counts for the cell when its middle lies inside it. With k draft
positions a step drafts k tokens, so

    acceptance length = 1 + k * accepted / drafted

and, given the bench's JSON, ``output_throughput / acceptance length`` is the
rate of decode steps. A cell of a few requests has an output throughput that
moves with the acceptance length (the tokens sampled), not with the speed of
the engine: the step rate is what a change of the host or the server can
alter.
"""

import argparse
import json
import sys
from collections.abc import Iterable

import regex as re

INTERVAL_S = 10.0
LINE = re.compile(
    r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[\S+\] SpecDecoding metrics: .*?"
    r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
    r"Per-position acceptance rate: ([\d., ]+?), Avg"
)


def seconds(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def parse_log(lines: Iterable[str]) -> list[tuple[int, int, int, int]]:
    """``(second of the day, accepted, drafted, draft positions)`` per interval."""
    out = []
    for line in lines:
        m = LINE.search(line)
        if m:
            k = len([x for x in m.group(4).split(",") if x.strip()])
            out.append((seconds(m.group(1)), int(m.group(2)), int(m.group(3)), k))
    return out


def acceptance(
    records: list[tuple[int, int, int, int]], start: str, end: str
) -> dict[str, float]:
    """Accepted and drafted tokens and the acceptance length over [start, end)."""
    lo, hi = seconds(start), seconds(end)
    mine = [r for r in records if lo <= r[0] - INTERVAL_S / 2 < hi]
    accepted = sum(r[1] for r in mine)
    drafted = sum(r[2] for r in mine)
    k = max((r[3] for r in mine), default=0)
    length = 1 + k * accepted / drafted if drafted else float("nan")
    return {
        "intervals": len(mine),
        "accepted": accepted,
        "drafted": drafted,
        "length": length,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--bench", help="the cell's bench JSON (vllm bench serve)")
    args = ap.parse_args(argv)
    with open(args.log, errors="replace") as f:
        result = acceptance(parse_log(f), args.start, args.end)
    text = (
        f"acceptance length {result['length']:.3f} ({result['accepted']} accepted, "
        f"{result['drafted']} drafted, {result['intervals']} log intervals)"
    )
    if args.bench:
        with open(args.bench) as f:
            bench = json.load(f)
        out = bench["output_throughput"]
        text += (
            f"; output {out:.1f} tok/s; steps/s {out / result['length']:.2f}; "
            f"mean ITL {bench['mean_itl_ms']:.2f} ms; "
            f"mean TPOT {bench['mean_tpot_ms']:.2f} ms"
        )
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
