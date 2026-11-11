#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MTP acceptance and step time per perf cell of a determinism arm.

  dspec.py ARM_DIR

Cell windows come from load_hook.txt ("perf start", then one "<cell> kfd:"
line per cell). Sums the 10 s SpecDecoding counters of serve.log inside each
window: mean acceptance length = 1 + 2 * accepted / drafted (2 draft tokens
per step). From the bench JSON: output tokens, median/mean ITL and TPOT; the
step time estimate is mean TPOT x acceptance length.
"""

import json
import os
import sys

import regex as re

LINE = re.compile(
    r"(\d\d:\d\d:\d\d) \[metrics.py:\d+\] SpecDecoding metrics: .*"
    r"Accepted: (\d+) tokens, Drafted: (\d+) tokens"
)


def sec(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def main():
    arm = sys.argv[1]
    with open(os.path.join(arm, "load_hook.txt")) as f:
        hook = f.read().splitlines()
    start = None
    cells = []
    for line in hook:
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] (perf start|(perf-\S+) kfd: (.*))", line)
        if not m:
            continue
        if m.group(2) == "perf start":
            start = sec(m.group(1))
        else:
            end = sec(m.group(1))
            cells.append((m.group(3), start, end, m.group(4).split("-> ")[-1]))
            start = end
    samples = []
    with open(os.path.join(arm, "serve.log"), errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                samples.append((sec(m.group(1)), int(m.group(2)), int(m.group(3))))
    for name, t0, t1, kfd in cells:
        # A counter line at time t covers the 10 s before it.
        acc = sum(a for t, a, d in samples if t0 < t <= t1 + 10)
        dra = sum(d for t, a, d in samples if t0 < t <= t1 + 10)
        alen = 1 + 2 * acc / dra if dra else float("nan")
        with open(os.path.join(arm, f"{name}.json")) as f:
            res = json.load(f)
        tpot = res["mean_tpot_ms"]
        print(
            f"{os.path.basename(arm)} {name}: kfd {kfd}; "
            f"acceptance length {alen:.3f} "
            f"({acc}/{dra}); output tokens {res['total_output_tokens']}; "
            f"TTFT mean {res['mean_ttft_ms'] / 1000:.2f} s; "
            f"ITL med {res['median_itl_ms']:.1f} mean {res['mean_itl_ms']:.1f} ms; "
            f"TPOT mean {tpot:.2f} ms; step estimate {tpot * alen:.1f} ms"
        )


if __name__ == "__main__":
    main()
