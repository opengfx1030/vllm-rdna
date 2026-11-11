#!/usr/bin/env python3
"""Run the mono kernel N times on the same inputs per width and compare each
output with today's op sequence (catches races and first-launch effects).

  python3 repeat_check.py [--widths 3,12,24] [--n 8]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_bench as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="3,6,12,24")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ext = B.build_ext()
    gen = torch.Generator(device=dev).manual_seed(7)
    L, _ = B.make_layer(gen, dev)
    ws = torch.zeros(ext.workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I),
                     dtype=torch.uint8, device=dev)
    ref = B.Reference(0, dev)
    bad = 0
    for M in [int(w) for w in a.widths.split(",")]:
        res = torch.randn(M, B.HC * B.H, generator=gen, device=dev).half()
        blk = (torch.randn(M, B.H, generator=gen, device=dev) * 0.5).half()
        inj = torch.randn(M, B.HC, generator=gen, device=dev).half()
        o_ref = ref(L, res, blk, inj)[0].float()
        diffs = []
        for _ in range(a.n):
            out = torch.empty(M, B.H, dtype=torch.half, device=dev)
            ro = torch.empty_like(res)
            io = torch.empty(M, B.HC, dtype=torch.half, device=dev)
            ext.mono_ffn(res, blk, inj, L["norm_w"], L["wd"], L["wu"], L["wr"],
                         L["se_w1"], L["se_w2"], L["se_wg"], L["w13"], L["s13"],
                         L["z13"], L["w2"], L["s2"], L["z2"], ro, io, out, ws, 0, 1,
                         0, B.EPS, 2000.0)
            torch.cuda.synchronize()
            d = (out.float() - o_ref).abs()
            diffs.append(round(d.max().item(), 5))
            if d.max().item() > 0.01:
                bad += 1
                rows = (d.max(1).values > 0.01).nonzero().flatten().tolist()
                cols = (d.max(0).values > 0.01).nonzero().flatten()
                print(f"  M={M} bad rows {rows} cols {cols[:8].tolist()}.. n={len(cols)}")
        print(f"M={M} max|mono-ref| per run: {diffs} err={ext.error(ws).tolist()}", flush=True)
    print("BAD" if bad else "OK", bad)


if __name__ == "__main__":
    main()
