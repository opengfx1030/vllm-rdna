#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layer-by-layer growth of a difference between two traced forwards.

  dgrowth.py TRACE_DIR --a 5 --b 6 [--rank 0] [--rows 1990,2009]
             [--lp noise_mix.npz:0,noise_mix.npz:1] [--top 6]

Needs full dumps (rank 0) of every decoder layer output and of every
``mlp.gate`` output for both forwards. Prints, per layer, how many rows differ
and the row-wise relative L2 difference of the layer's first output (median,
p99, max), plus how many rows changed their router top-k set. Then, for the
selected rows (``--rows``, or the ``--top`` positions whose logprob moved most
between the two ``--lp`` dclient entries), the per-layer relative difference
and the layers whose top-k set changed with the 10th/11th logit margin.
"""

import argparse
import glob
import os

import numpy as np
import regex as re
import torch

PAT = re.compile(r"r(\d+)_f(\d+)_c(\d+)_(.+)_(out)(\d+)\.pt$")
TOPK = 10


def dumps(d, rank, fwd):
    out = {}
    for p in glob.glob(os.path.join(d, "dump", f"r{rank}_f{fwd:03d}_c*_out*.pt")):
        m = PAT.search(os.path.basename(p))
        if m:
            out[(m.group(4), int(m.group(6)))] = p
    return out


def layer_names(keys, suffix):
    found = {}
    for name, idx in keys:
        m = re.search(r"layers\.(\d+)" + re.escape(suffix) + r"$", name)
        if m and idx == 0:
            found[int(m.group(1))] = name
    return [found[k] for k in sorted(found)]


def rowrel(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--rows", default="")
    ap.add_argument("--lp", default="")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="window position of row 0 of these forwards (chunked prefill)",
    )
    args = ap.parse_args()

    da, db = dumps(args.dir, args.rank, args.a), dumps(args.dir, args.rank, args.b)
    layers = layer_names(da, "")
    gates = layer_names(da, ".mlp.gate")
    rows = [int(r) - args.offset for r in args.rows.split(",") if r.strip()]
    nrows = torch.load(da[(layers[0], 0)]).shape[0]
    if args.lp:
        (fa, ka), (fb, kb) = [
            (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in args.lp.split(",")
        ]
        lpa = np.load(fa)["act_lp"][ka]
        lpb = np.load(fb)["act_lp"][kb]
        dd = np.abs(lpa - lpb)
        dd[: args.offset] = -1
        dd[args.offset + nrows :] = -1
        order = np.argsort(-dd)[: args.top]
        print(
            "largest logprob moves in this chunk (position: a -> b):",
            ", ".join(f"{int(r)}: {lpa[r]:.2f} -> {lpb[r]:.2f}" for r in order),
        )
        rows = rows or [int(r) - args.offset for r in order]

    per_row = {r: [] for r in rows}
    print(
        f"{'layer':>5} {'rows diff':>9} {'med rel':>9} {'p99 rel':>9} {'max rel':>9}"
        f" {'topk flips':>10}"
    )
    for li, name in enumerate(layers):
        a = torch.load(da[(name, 0)])
        b = torch.load(db[(name, 0)])
        rel = rowrel(a, b)
        flips = ""
        flip_rows = {}
        if li < len(gates) and (gates[li], 0) in da and (gates[li], 0) in db:
            ga = torch.load(da[(gates[li], 0)]).float()
            gb = torch.load(db[(gates[li], 0)]).float()
            ta = ga.topk(TOPK, dim=1).indices.sort(dim=1).values
            tb = gb.topk(TOPK, dim=1).indices.sort(dim=1).values
            changed = (ta != tb).any(dim=1)
            flips = f"{int(changed.sum()):10d}"
            srt = ga.sort(dim=1, descending=True).values
            margin = (srt[:, TOPK - 1] - srt[:, TOPK]).numpy()
            for r in rows:
                if bool(changed[r]):
                    flip_rows[r] = float(margin[r])
        print(
            f"{li:5d} {int((rel > 0).sum()):9d} {np.median(rel):9.2e} "
            f"{np.quantile(rel, 0.99):9.2e} {rel.max():9.2e} {flips}"
        )
        for r in rows:
            per_row[r].append((rel[r], flip_rows.get(r)))
    for r in rows:
        curve = " ".join(f"{v:.0e}" if v > 0 else "0" for v, _ in per_row[r])
        flips = [(li, m) for li, (_, m) in enumerate(per_row[r]) if m is not None]
        print(f"position {r + args.offset}: rel per layer: {curve}")
        if flips:
            print(
                f"position {r + args.offset}: top-{TOPK} set changed at layers"
                " (layer, 10th-11th margin in a):",
                ", ".join(f"({li}, {m:.4f})" for li, m in flips),
            )


if __name__ == "__main__":
    main()
