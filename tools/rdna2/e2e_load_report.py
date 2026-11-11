#!/usr/bin/env python
"""Report TTFT / ITL / TPOT / throughput per (scenario, concurrency) for a sweep.

    .venv/bin/python tools/rdna2/e2e_load_report.py <label> [<label2> ...]

Reads .pi/sweep_<label>/*.json (written by e2e_load_sweep.sh with --save-result)
and prints one table per scenario. With two labels it prints both side by side
plus the delta, which is what the Triton-vs-HIP arm comparison needs.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (column header, how to read it). The JSON stores percentiles as
# percentiles_<metric>_ms = [[50, v], [95, v], [99, v]], not as flat p99_* keys.
PCTS = (50, 95, 99)
METRICS = [
    ("TTFT p50", ("median", "ttft")),
    ("TTFT p95", ("pct", "ttft", 95)),
    ("TTFT p99", ("pct", "ttft", 99)),
    ("ITL p50", ("median", "itl")),
    ("ITL p95", ("pct", "itl", 95)),
    ("ITL p99", ("pct", "itl", 99)),
    ("TPOT p50", ("median", "tpot")),
    ("E2EL p50", ("median", "e2el")),
    ("out tok/s", ("flat", "output_throughput")),
    ("tot tok/s", ("flat", "total_token_throughput")),
    ("req/s", ("flat", "request_throughput")),
]


def value(row, spec):
    kind = spec[0]
    if kind == "median":
        return row.get(f"median_{spec[1]}_ms")
    if kind == "flat":
        return row.get(spec[1])
    if kind == "pct":
        metric, p = spec[1], spec[2]
        for entry in row.get(f"percentiles_{metric}_ms") or []:
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and int(entry[0]) == p:
                return entry[1]
        return row.get(f"p{p}_{metric}_ms")
    return None


def load(label):
    d = {}
    for p in sorted((REPO / ".pi" / f"sweep_{label}").glob("*.json")):
        scenario, _, conc = p.stem.rpartition("_c")
        try:
            d[(scenario, int(conc))] = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
    return d


def fmt(v):
    if v is None:
        return "      -"
    if v >= 1000:
        return f"{v:7.0f}"
    if v >= 10:
        return f"{v:7.1f}"
    return f"{v:7.2f}"


def table(scenarios, arms, concs):
    for sc in scenarios:
        head = f"{'conc':>4}"
        for label in arms:
            head += "  " + " ".join(f"{h:>7}" for h, _ in METRICS)
        print(f"\n=== {sc} ===")
        print(head)
        for c in concs:
            rows = [arms[label].get((sc, c)) for label in arms]
            if not any(rows):
                continue
            line = f"{c:>4}"
            for r in rows:
                line += "  " + " ".join(
                    fmt(None if r is None else value(r, spec)) for _, spec in METRICS
                )
            print(line)
        if len(arms) == 2:
            a, b = arms.values()
            print(f"{'delta':>4} (B-A)/A, negative = B better")
            for c in concs:
                ra, rb = a.get((sc, c)), b.get((sc, c))
                if not ra or not rb:
                    continue
                cells = []
                for _, spec in METRICS:
                    va, vb = value(ra, spec), value(rb, spec)
                    if va and vb:
                        pct = (vb - va) / va * 100
                        cells.append(f"{pct:+7.1f}")
                    else:
                        cells.append("      -")
                print(f"{c:>4}  " + " ".join(cells))


def main():
    labels = sys.argv[1:]
    if not labels:
        sys.exit("usage: e2e_load_report.py <label> [<label2>]")
    arms = {label: load(label) for label in labels}
    for label in labels:
        if not arms[label]:
            sys.exit(f"no results in .pi/sweep_{label}/")
    scenarios, concs = [], []
    for label in labels:
        for sc, c in arms[label]:
            if sc not in scenarios:
                scenarios.append(sc)
            if c not in concs:
                concs.append(c)
    order = ["prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "mixed_2k", "decode"]
    scenarios.sort(key=lambda s: order.index(s) if s in order else len(order))
    concs.sort()
    print(f"arms: {', '.join(labels)}")
    table(scenarios, arms, concs)


if __name__ == "__main__":
    main()
