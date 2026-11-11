#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph timing of the Flash-Next decode-step glue ops per token count M.

Each case captures REPS calls that cycle through NW distinct weight sets, so
the weights stream from DRAM as they do across layers (one set would sit in the
128 MB Infinity Cache), and prints the graph-replay time per call, launch gaps
included. ``--kernels`` also lists the kernels one eager call launches.

    HIP_VISIBLE_DEVICES=2 python tools/rdna/port_v031/decode_ops_bench.py hc \
        --m 1,6,8,9,18,24,64
    HIP_VISIBLE_DEVICES=2 python tools/rdna/port_v031/decode_ops_bench.py se \
        --m 1,6,8,9,18,24,32 --kernels

The hc case copes with both op contracts: when the op returns the full down
projection (old contract) the caller's injection slice copy is timed too.

The se case times the fused op, which does not look at VLLM_RDNA3_HIP_OPS (the
model-level gate does); ``--torch`` times the plain torch composition instead,
the route the model takes when the se family is off.
"""

import argparse
import collections

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.rdna_ops  # noqa: F401  (registers the ops)

R, HC, H = 320, 4, 2560  # HC lora rank, streams, hidden
I_SE = 160  # shared-expert intermediate per rank (TP4)


def graph_us_per_call(calls, replays=30):
    for f in calls[:4]:
        f()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    for _ in range(3):
        g.replay()
    torch.accelerator.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(replays):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / (replays * len(calls))


def kernel_table(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    rows = collections.OrderedDict()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            t = getattr(ev, "device_time", None)
            if t is None:
                t = ev.cuda_time
            n, tot = rows.get(ev.name, (0, 0.0))
            rows[ev.name] = (n + 1, tot + t)
    return rows


def run(name, make_calls, ms, kernels):
    print(f"{'M':>4} {'us/call':>9}  kernels")
    for m in ms:
        calls = make_calls(m)
        us = graph_us_per_call(calls)
        nk, desc = "", ""
        if kernels:
            tab = kernel_table(calls[0])
            nk = sum(n for n, _ in tab.values())
            desc = "; ".join(
                f"{k[:44]} x{n} {t / n:.1f}us" for k, (n, t) in tab.items()
            )
        print(f"{m:>4} {us:>9.1f}  {nk} {desc}")


def bench_hc(ms, nw, reps, kernels):
    wd = [
        (torch.randn(R + HC + 12, HC * H, device="cuda") * 0.02).half()
        for _ in range(nw)
    ]
    wu = [(torch.randn(HC * H, R, device="cuda") * 0.05).half() for _ in range(nw)]

    def make(m):
        xn = torch.randn(m, HC * H, device="cuda").half()

        def mk(i):
            def f():
                b, second = torch.ops.vllm.rdna_hc_mix(
                    xn, wd[i % nw], None, None, wu[i % nw], None, None, R, HC
                )
                if second.shape[1] != HC:  # old contract: the full down output
                    second = second[:, R : R + HC].contiguous()
                return b, second

            return f

        return [mk(i) for i in range(reps)]

    run("hc", make, ms, kernels)


def bench_se(ms, nw, reps, kernels, torch_route=False):
    w1 = [(torch.randn(2 * I_SE, H, device="cuda") * 0.02).half() for _ in range(nw)]
    w2 = [(torch.randn(H, I_SE, device="cuda") * 0.05).half() for _ in range(nw)]
    wg = [(torch.randn(H, device="cuda") * 0.05).half() for _ in range(nw)]

    def make(m):
        x = torch.randn(m, H, device="cuda").half()

        def mk(i):
            def f():
                if torch_route:
                    gu = F.linear(x, w1[i % nw])
                    act = F.silu(gu[..., :I_SE]) * gu[..., I_SE:]
                    out = F.linear(act, w2[i % nw])
                    gate = F.linear(x, wg[i % nw].reshape(1, -1))
                    return torch.sigmoid(gate) * out
                return torch.ops.vllm.rdna_shared_expert(
                    x, w1[i % nw], None, None, w2[i % nw], None, None, wg[i % nw]
                )

            return f

        return [mk(i) for i in range(reps)]

    run("se-torch" if torch_route else "se", make, ms, kernels)


def main():
    global I_SE
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["hc", "se"])
    ap.add_argument("--m", default="1,3,6,8,9,12,16,18,24,32")
    ap.add_argument(
        "--nw", type=int, default=0, help="weight sets (default: 16 hc, 64 se)"
    )
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument(
        "--torch",
        action="store_true",
        help="se only: time the torch composition instead of the fused op",
    )
    ap.add_argument(
        "--i-se",
        type=int,
        default=I_SE,
        help="shared-expert intermediate per rank (TP4: 160, TP2: 320)",
    )
    args = ap.parse_args()
    I_SE = args.i_se
    ms = [int(v) for v in args.m.split(",")]
    if args.case == "hc":
        bench_hc(ms, args.nw or 16, args.reps, args.kernels)
    else:
        bench_se(ms, args.nw or 64, args.reps, args.kernels, args.torch)


if __name__ == "__main__":
    main()
