#!/usr/bin/env python3
"""Per-phase work imbalance of the registered mono_k2_rdna2 op: for each phase,
when each WG finished its work (wg_done) relative to the phase start (WG0's
release of the previous barrier), over 16 distinct layers (cold IC) in a graph.

  PYTHONPATH=<tree> python3 wg_spread.py [--widths 1,3,8,24]
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_bench as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,8,24")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--op", action="store_true", help="registered op instead of the JIT ext")
    a = ap.parse_args()
    import vllm._custom_ops  # noqa: F401
    if a.op:
        wsbytes, run_op = torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes, torch.ops._rocm_C.mono_k2_rdna2
        err_of = torch.ops._rocm_C.mono_k2_rdna2_error
    else:
        ext = B.build_ext()
        wsbytes, run_op, err_of = ext.workspace_bytes, ext.mono_ffn, ext.error
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5)
    layers = [B.make_layer(gen, dev, keep_golden=False)[0] for _ in range(a.layers)]
    nb = int(wsbytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    for M in [int(x) for x in a.widths.split(",")]:
        res = torch.randn(M, B.HC * B.H, generator=gen, device=dev).half()
        blk = (torch.randn(M, B.H, generator=gen, device=dev) * 0.5).half()
        inj = torch.randn(M, B.HC, generator=gen, device=dev).half()
        out = torch.empty(M, B.H, dtype=torch.half, device=dev)
        ro = torch.empty_like(res)
        io = torch.empty(M, B.HC, dtype=torch.half, device=dev)
        spans = {ph: [] for ph in range(1, 7)}
        works = {ph: [] for ph in range(1, 7)}
        for _ in range(a.reps):
            for L in layers:
                run_op(
                    res, blk, inj, L["norm_w"], L["wd"], L["wu"], L["wr"], L["se_w1"],
                    L["se_w2"], L["se_wg"], L["w13"], L["s13"], L["z13"], L["w2"], L["s2"],
                    L["z2"], ro, io, out, ws, 0, 1, 0, B.EPS, 2000.0)
            torch.cuda.synchronize()
            t = ws[32 : 32 + 8 * (8 + 10 * 256)].view(torch.int64).cpu()
            ts, done = t[:8], t[8:].view(10, 256)
            G = int(err_of(ws)[3])
            for ph in range(1, 7):
                d = (done[ph - 1, :G] - ts[ph - 1]).float() / 100.0  # us
                spans[ph].append(float((ts[ph] - ts[ph - 1]) / 100.0))
                works[ph].append((float(d.min()), float(d.median()), float(d.max())))
            for key, row in (("5topk", 8), ("5reset", 9), ("5a", 7), ("5b", 6)):  # P5 sub-steps from ts[4]
                d = (done[row, :G] - ts[4]).float() / 100.0
                works.setdefault(key, []).append((float(d.min()), float(d.median()), float(d.max())))
        print(f"M={M} (grid {G}): phase span | WG work end min / median / max (us)")
        for ph in list(range(1, 7)) + ["5topk", "5reset", "5a", "5b"]:
            if ph not in spans:
                spans[ph] = [0.0]
            mn = statistics.median(w[0] for w in works[ph])
            md = statistics.median(w[1] for w in works[ph])
            mx = statistics.median(w[2] for w in works[ph])
            print(f"  P{ph}: {statistics.median(spans[ph]):6.1f} | {mn:6.1f} / {md:6.1f} / {mx:6.1f}")


if __name__ == "__main__":
    main()
