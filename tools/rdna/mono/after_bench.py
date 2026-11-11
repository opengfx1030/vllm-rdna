#!/usr/bin/env python3
"""Does the mono kernel slow down the work that follows it? Per layer, a graph
runs the FFN block (mono kernel or today's op sequence) and then "next" work
shaped like the attention block that follows in the serve (cold fp16 GEMMs
plus elementwise ops on distinct weights). Reports the per-layer time of
block + next, the block alone, and the difference (= next after that block).

  PYTHONPATH=<tree> python3 after_bench.py [--widths 1,3,8] [--num-wg 0]
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_bench as B  # noqa: E402


def graph_time(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,8")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--num-wg", type=int, default=0)
    ap.add_argument("--next-gemms", type=int, default=6)
    ap.add_argument("--next", default="gemm", choices=["gemm", "small"],
                    help="small: ~30 distinct small kernels (GDN/attention-like glue)")
    a = ap.parse_args()
    import vllm._custom_ops  # noqa: F401
    ext = B.build_ext()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    layers = [B.make_layer(gen, dev, keep_golden=False)[0] for _ in range(a.layers)]
    # attention-like next work per layer: cold [2560 -> 1280] fp16 GEMMs + norm-ish ops
    nxt = [[(torch.randn(1280, B.H, generator=gen, device=dev) * 0.02).half()
            for _ in range(a.next_gemms)] for _ in layers]
    nb = int(ext.workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    ref = B.Reference(0, dev)
    for M in [int(x) for x in a.widths.split(",")]:
        gt = torch.Generator(device=dev).manual_seed(99 + M)
        ins = [((torch.randn(M, B.HC * B.H, generator=gt, device=dev)).half(),
                (torch.randn(M, B.H, generator=gt, device=dev) * 0.5).half(),
                torch.randn(M, B.HC, generator=gt, device=dev).half()) for _ in layers]
        outs = [torch.empty(M, B.H, dtype=torch.half, device=dev) for _ in layers]
        ros = [torch.empty(M, B.HC * B.H, dtype=torch.half, device=dev) for _ in layers]
        ios = [torch.empty(M, B.HC, dtype=torch.half, device=dev) for _ in layers]
        xs = [torch.randn(M, B.H, generator=gt, device=dev).half() for _ in layers]

        def mono(i):
            L = layers[i]
            r, b, j = ins[i]
            ext.mono_ffn(r, b, j, L["norm_w"], L["wd"], L["wu"], L["wr"], L["se_w1"],
                         L["se_w2"], L["se_wg"], L["w13"], L["s13"], L["z13"], L["w2"],
                         L["s2"], L["z2"], ros[i], ios[i], outs[i], ws, 0, 1, a.num_wg,
                         B.EPS, 2000.0)

        def today(i):
            r, b, j = ins[i]
            ref(layers[i], r, b, j)

        def nextw(i):
            x = xs[i]
            if a.next == "small":  # many distinct tiny kernels, no bandwidth
                y = x.float()
                for _ in range(3):
                    y = torch.nn.functional.softplus(y) * 0.5
                    y = torch.sigmoid(y) + y.mean(dim=1, keepdim=True)
                    y = torch.nn.functional.layer_norm(y, y.shape[1:])
                    y = torch.tanh(y).cumsum(dim=1) * 1e-3
                    y = torch.nn.functional.gelu(y).flip(1)
                    y = torch.exp(-y.abs()).softmax(dim=1)
                    y = (y - y.amax(dim=1, keepdim=True)).clamp(-4, 4)
                xs[i].copy_(y.half())
                return
            for w in nxt[i]:
                y = torch.nn.functional.linear(x, w)
                x = torch.nn.functional.silu(torch.cat([y, y], dim=1))

        res = {}
        for name, blk in (("today", today), ("mono", mono)):
            t_blk = graph_time(lambda: [blk(i) for i in range(len(layers))], a.iters)
            t_all = graph_time(lambda: [(blk(i), nextw(i)) for i in range(len(layers))], a.iters)
            res[name] = (t_blk / len(layers), t_all / len(layers))
        t_next = graph_time(lambda: [nextw(i) for i in range(len(layers))], a.iters) / len(layers)
        print(f"M={M}: next alone {t_next:6.1f} us/layer")
        for name, (tb, ta) in res.items():
            print(f"  {name:5s} block {tb:6.1f} | block+next {ta:6.1f} | next after block {ta - tb:6.1f} us/layer")
        err = ext.error(ws).tolist()
        print("  sticky", err[:3])


if __name__ == "__main__":
    main()
