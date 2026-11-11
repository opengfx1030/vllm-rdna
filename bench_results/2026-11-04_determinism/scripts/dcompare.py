#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare two dtrace forward records (same rank): first divergent calls.

  dcompare.py DIR --a 3 --b 5 [--rank 0] [--show 40]

For every call in execution order, an output whose per-row hashes differ is a
divergence. A call whose outputs differ while all its hashed inputs match is a
SOURCE (the op itself produced a different result from identical inputs).
"""

import argparse
import collections
import os

import numpy as np
import regex as re
import torch


def load(d, rank, fwd):
    return torch.load(os.path.join(d, f"r{rank}_f{fwd:03d}.pt"), weights_only=False)


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name + ".")
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--rank", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    for rank in args.rank:
        try:
            ra, rb = load(args.dir, rank, args.a), load(args.dir, rank, args.b)
        except FileNotFoundError as e:
            print(f"rank {rank}: missing {e}")
            continue
        A, B = ra["rec"], rb["rec"]
        print(
            f"=== rank {rank}: fwd {args.a} ({len(A)} entries) "
            f"vs fwd {args.b} ({len(B)} entries)"
        )
        if len(A) != len(B):
            print("  WARNING: different entry counts; comparing common prefix by key")
        ib = {(c, n, k, i): h for c, n, k, i, s, dt, h in B}
        calls = collections.OrderedDict()
        for c, n, k, i, s, dt, h in A:
            calls.setdefault((c, n), []).append((k, i, s, dt, h))
        shown = 0
        n_div_calls = 0
        first_src = None
        per_name = collections.Counter()
        div_names = set()
        for (c, n), ents in calls.items():
            in_same = True
            out_div = []
            for k, i, s, dt, h in ents:
                hb = ib.get((c, n, k, i))
                if hb is None:
                    continue
                diff = h != hb
                if k == "in" and diff.any():
                    in_same = False
                if k == "out" and diff.any():
                    rows = np.nonzero(diff)[0]
                    out_div.append((i, s, dt, int(diff.sum()), int(rows[0]), len(h)))
            if n.startswith("root.") and not in_same:
                print(f"  ROOT INPUT DIFFERS at {n}")
            if out_div:
                n_div_calls += 1
                per_name[re.sub(r"layers\.\d+\.", "layers.N.", n)] += 1
                tag = "SOURCE" if in_same else "prop"
                if in_same and (
                    n == "root"
                    and div_names
                    or any(d.startswith(n + ".") for d in div_names)
                ):
                    tag = "parent"  # a child call diverged first
                div_names.add(n)
                if tag == "SOURCE" and first_src is None:
                    first_src = (c, n)
                if shown < args.show or tag == "SOURCE":
                    desc = "; ".join(
                        f"out{i} {tuple(s)} {dt}: {nd}/{nr} rows differ "
                        f"(first row {r0})"
                        for i, s, dt, nd, r0, nr in out_div
                    )
                    print(f"  [{c:5d}] {tag:6s} L{layer_of(n):2d} {n}: {desc}")
                    shown += 1
        print(
            f"  divergent calls: {n_div_calls} / {len(calls)}; "
            f"first SOURCE: {first_src}"
        )
        print("  divergent call count by module kind (top 15):")
        for name, cnt in per_name.most_common(15):
            print(f"    {cnt:5d}  {name}")


if __name__ == "__main__":
    main()
