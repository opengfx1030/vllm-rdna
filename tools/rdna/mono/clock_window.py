#!/usr/bin/env python3
"""Mean SCLK / MCLK / power / GPU use per bench window from run_specs.sh's
clocks.log, windows taken from the serve's summary.txt bench start lines.

  python3 clock_window.py SERVE_DIR [SERVE_DIR ...]
"""

import datetime
import os
import re
import statistics
import sys


def samples(path):
    out, t, cur = [], None, {}
    for line in open(path):
        if line.startswith("## "):
            if t is not None:
                out.append((t, cur))
            t, cur = float(line.split()[1]), {}
            continue
        m = re.match(r"GPU\[(\d+)\]\s*: (.*)", line)
        if not m:
            continue
        g, rest = int(m.group(1)), m.group(2)
        v = re.search(r"\((\d+)Mhz\)", rest)
        if "sclk" in rest and v:
            cur.setdefault(g, {})["sclk"] = int(v.group(1))
        elif "mclk" in rest and v:
            cur.setdefault(g, {})["mclk"] = int(v.group(1))
        elif "Power" in rest:
            cur.setdefault(g, {})["power"] = float(rest.split(":")[-1])
        elif "GPU use" in rest:
            cur.setdefault(g, {})["use"] = float(rest.split(":")[-1])
    if t is not None:
        out.append((t, cur))
    return out


def windows(summary, day_ts):
    starts = []
    for line in open(summary):
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] bench (\S+)", line)
        if m:
            h, mi, se, name = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            starts.append((name, h * 3600 + mi * 60 + se))
        elif starts and re.match(r"\[\d\d:\d\d:\d\d\] ", line):
            m2 = re.match(r"\[(\d\d):(\d\d):(\d\d)\]", line)
            starts.append((None, int(m2.group(1)) * 3600 + int(m2.group(2)) * 60 + int(m2.group(3))))
    wins = []
    for i, (name, s) in enumerate(starts):
        if name is None:
            continue
        e = starts[i + 1][1] if i + 1 < len(starts) else s + 600
        wins.append((name, day_ts + s, day_ts + e))
    return wins


def main():
    for d in sys.argv[1:]:
        smp = samples(os.path.join(d, "clocks.log"))
        if not smp:
            continue
        t0 = datetime.datetime.fromtimestamp(smp[0][0])
        day = datetime.datetime(t0.year, t0.month, t0.day).timestamp()
        print(f"== {os.path.basename(d.rstrip('/'))}")
        for name, s, e in windows(os.path.join(d, "summary.txt"), day):
            sel = [c for t, c in smp if s + 5 <= t <= e - 2]
            if not sel:
                continue
            row = []
            for k in ("sclk", "mclk", "power", "use"):
                vals = [c[g][k] for c in sel for g in c if k in c[g]]
                row.append(f"{k} {statistics.mean(vals):7.1f}" if vals else f"{k} -")
            print(f"  {name:22s} n={len(sel):3d} " + " | ".join(row))


if __name__ == "__main__":
    main()
