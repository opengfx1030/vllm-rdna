#!/usr/bin/env python3
"""P0(b): kernel nodes per captured hipGraph from graphcount_shim output
(serve_p0b.sh PROF=graphs), bucketed like analyze_trace.py.

  python3 graph_counts.py OUT/graphs/graphs-<pid>.jsonl
"""

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_trace import NAMES, bucket  # noqa: E402


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path)]
        print(f"== {path}: {len(rows)} captured graphs")
        piecewise = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("seq", 0) < 0:
                print("  error:", r.get("error"))
                continue
            caller = "/".join(r.get("caller", [])[1:4])
            if "execution_fn" in caller:  # piecewise backend
                m = re.search(r"num_tokens=(\d+)", r.get("batch", ""))
                w = m.group(1) if m else "?"
                piecewise[w][0] += 1
                piecewise[w][1] += r["kernels"]
                continue
            b = collections.Counter()
            for name, c in r["names"].items():
                b[bucket(name)] += c
            moe = sum(c for n, c in r["names"].items() if "moe_gemm_q4" in n or "fused_moe" in n)
            print(f"  FULL seq {r['seq']:3d}: {r['kernels']:5d} kernels (MoE GEMMs {moe})  "
                  + " ".join(f"{x}={b[x]}" for x in NAMES if b[x]))
        for w, (g, k) in piecewise.items():
            print(f"  PIECEWISE width {w}: {g} graphs, {k} kernels")


if __name__ == "__main__":
    main()
