#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Element-level diff of the tensors dumped for two traced forwards.

  ddump_diff.py TRACE_DIR --a 3 --b 5 [--rank 0] [--match mlp]

For every dumped tensor (same name, kind, index) of both forwards: elements
that differ, rows that differ, fp16 ulp distance histogram (same-sign values),
max |d| and relative L2 of the difference.
"""

import argparse
import collections
import glob
import os

import regex as re
import torch

PAT = re.compile(r"r(\d+)_f(\d+)_c(\d+)_(.+)_(in|out)(\d+)\.pt$")


def index(d, rank, fwd):
    out = {}
    for p in glob.glob(os.path.join(d, "dump", f"r{rank}_f{fwd:03d}_c*.pt")):
        m = PAT.search(os.path.basename(p))
        if m:
            out[(m.group(4), m.group(5), int(m.group(6)))] = (int(m.group(3)), p)
    return out


def ulp_hist(a, b):
    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    same_sign = (ia < 0) == (ib < 0)
    d = (ia - ib).abs()
    d = torch.where(same_sign, d, torch.full_like(d, 1 << 20))
    hist = collections.Counter()
    for k in (1, 2, 3, 4):
        hist[str(k)] = int((d == k).sum())
    hist["5-16"] = int(((d >= 5) & (d <= 16)).sum())
    hist[">16"] = int(((d > 16) & (d < (1 << 20))).sum())
    hist["sign"] = int((d == (1 << 20)).sum())
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--match", default="")
    args = ap.parse_args()
    ia, ib = index(args.dir, args.rank, args.a), index(args.dir, args.rank, args.b)
    for key in sorted(ia, key=lambda k: ia[k][0]):
        if key not in ib or args.match not in key[0]:
            continue
        x = torch.load(ia[key][1])
        y = torch.load(ib[key][1])
        name = f"{key[0]} {key[1]}{key[2]} {tuple(x.shape)} {x.dtype}"
        if x.shape != y.shape:
            print(f"{name}: shape mismatch {tuple(y.shape)}")
            continue
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            ne = int((x != y).sum())
            print(f"{name}: {ne} elements differ")
            continue
        diff = x != y
        ne = int(diff.sum())
        rows = int(diff.reshape(x.shape[0], -1).any(1).sum()) if x.dim() > 1 else ne
        xf, yf = x.float(), y.float()
        rel = ((xf - yf).norm() / xf.norm().clamp_min(1e-30)).item()
        mx = (xf - yf).abs().max().item()
        line = (
            f"{name}: {ne}/{x.numel()} elems ({ne / max(1, x.numel()):.2e}) "
            f"rows {rows}/{x.shape[0]} max|d| {mx:.3e} relL2 {rel:.3e}"
        )
        if x.dtype == torch.float16 and ne:
            line += f" ulps {dict(ulp_hist(x[diff], y[diff]))}"
        print(line)


if __name__ == "__main__":
    main()
