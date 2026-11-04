#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Repeat-identity statistics of dclient.py runs.

  dident.py run.npz [--ref other.npz]

Without --ref every repeat r > 0 of a window is compared with repeat 0 of the
same window in the same file; with --ref, every entry of run.npz is compared
with repeat 0 of the same window in other.npz. Per pair: bit-identical
positions, position-0 identity, top-1 agreement, mean/max |d logp|, moves of
at least 1 and 5 nats. Then the totals.
"""

import argparse

import numpy as np


def load(path):
    z = np.load(path)
    return [
        (int(r), str(d), int(i), z["act_lp"][k], z["top1"][k])
        for k, (r, d, i) in enumerate(zip(z["rep"], z["dom"], z["idx"]))
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref")
    args = ap.parse_args()
    cur = load(args.run)
    ref = {(d, i): (lp, t1) for r, d, i, lp, t1 in load(args.ref or args.run) if r == 0}
    tot = dict(pairs=0, pos=0, ident=0, pos0=0, win_ident=0, agree=0, ge1=0, ge5=0)
    sum_d = 0.0
    max_d = (0.0, None)
    for r, d, i, lp, t1 in cur:
        if args.ref is None and r == 0:
            continue
        lp0, t10 = ref[(d, i)]
        dd = np.abs(lp - lp0)
        same = lp == lp0
        tot["pairs"] += 1
        tot["pos"] += lp.size
        tot["ident"] += int(same.sum())
        tot["pos0"] += int(same[0])
        tot["win_ident"] += int(same.all())
        tot["agree"] += int((t1 == t10).sum())
        tot["ge1"] += int((dd >= 1).sum())
        tot["ge5"] += int((dd >= 5).sum())
        sum_d += float(dd.sum())
        if dd.max() > max_d[0]:
            max_d = (
                float(dd.max()),
                (
                    d,
                    i,
                    r,
                    int(dd.argmax()),
                    float(lp0[dd.argmax()]),
                    float(lp[dd.argmax()]),
                ),
            )
        first = np.nonzero(~same)[0]
        print(
            f"{d}:{i:3d} rep {r}: ident {same.mean() * 100:6.2f}%  "
            f"pos0 {int(same[0])}  "
            f"first diff {int(first[0]) if first.size else -1:5d}  "
            f"mean|d| {dd.mean():.4f}  "
            f"max|d| {dd.max():7.3f} @{int(dd.argmax())}  "
            f">=1 {int((dd >= 1).sum())}  "
            f">=5 {int((dd >= 5).sum())}"
        )
    pos = max(1, tot["pos"])
    print(
        f"TOTAL {tot['pairs']} pairs: "
        f"positions identical {tot['ident'] / pos * 100:.2f}%  "
        f"pos0 identical {tot['pos0']}/{tot['pairs']}  windows identical "
        f"{tot['win_ident']}/{tot['pairs']}  "
        f"top1 agree {tot['agree'] / pos * 100:.2f}%  "
        f"mean|d| {sum_d / pos:.5f}  >=1 nat {tot['ge1']}  >=5 nat {tot['ge5']}  "
        f"max|d| {max_d[0]:.3f} {max_d[1]}"
    )


if __name__ == "__main__":
    main()
