#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""List traced forwards grouped by identical root inputs (same window).

  dgroups.py TRACE_DIR [--rank 0]

Prints one line per forward: index, group id (forwards with bit-identical
input_ids and positions share it), entry count, elapsed seconds and the
perturbations applied. Then, per group, the forwards to pair with dcompare.py.
"""

import argparse
import glob
import hashlib
import os

import regex as re
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, f"r{args.rank}_f*.pt")))
    groups = {}
    for path in files:
        fwd = int(re.search(r"_f(\d+)\.pt$", path).group(1))
        r = torch.load(path, weights_only=False)
        key = hashlib.sha1()
        for c, n, k, i, s, dt, h in r["rec"]:
            if n in ("root.input_ids", "root.positions"):
                key.update(n.encode())
                key.update(h.tobytes())
        gid = groups.setdefault(key.hexdigest(), len(groups))
        pert = r.get("perturb") or []
        desc = ", ".join(f"{p[0]} {p[1]} {p[2:]}" for p in pert)[:120]
        print(
            f"fwd {fwd:3d}  group {gid}  entries {len(r['rec']):6d}  "
            f"{r['elapsed']:6.1f}s  {desc}"
        )
    print(f"{len(groups)} groups")


if __name__ == "__main__":
    main()
