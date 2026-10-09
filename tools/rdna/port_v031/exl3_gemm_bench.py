#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark ``exl3_gemm_rdna2`` at the Qwen3.8-27B TP=4 shapes.

Times every EXL3 linear of one decoder step (per GPU) at the batch sizes a
verify step produces (M = 1, 3, 6, 8, 12, 24, ...) and prints the per-shape
time and the per-step total (shape time x layer count).

    HIP_VISIBLE_DEVICES=6 python tools/rdna/port_v031/exl3_gemm_bench.py \
        [--m 1,3,6,8,12,24] [--bits 3] [--cb 2] [--check]

``--check`` compares each shape against a decode-once fp32 reference
(``exl3_decode_trellis_rdna2`` + matmul), so a kernel change can be checked
at the real shapes, not only at the unit-test sizes.
"""

import argparse

import torch

# (name, K, N, layers per step) for Qwen3.8-27B (hidden 5120, inter 17408,
# 48 GDN + 16 attention layers, 64 MLPs) at TP=4. Fused partitions with
# different suh run as separate GEMMs; that is how exl3_project groups them.
SHAPES_27B_TP4 = [
    ("gdn.in_proj_qkv", 5120, 2560, 48),
    ("gdn.in_proj_z", 5120, 1536, 48),
    ("gdn.out_proj", 1536, 5120, 48),
    ("attn.q_proj", 5120, 3072, 16),
    ("attn.k_proj", 5120, 256, 16),
    ("attn.v_proj", 5120, 256, 16),
    ("attn.o_proj", 1536, 5120, 16),
    ("mlp.gate", 5120, 4352, 64),
    ("mlp.up", 5120, 4352, 64),
    ("mlp.down", 4352, 5120, 64),
]


def time_loop(step, iters: int, graph: bool) -> float:
    """Mean microseconds per step, optionally replayed from one graph."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    if graph:
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
            with torch.cuda.graph(g, stream=stream):
                for _ in range(iters):
                    step()
        torch.cuda.current_stream().wait_stream(stream)
        g.replay()
        torch.cuda.synchronize()
        s.record()
        g.replay()
        e.record()
    else:
        s.record()
        for _ in range(iters):
            step()
        e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,3,6,8,12,16,24")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--cb", type=int, default=2)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--head", action="store_true", help="only the lm_head (62080 x 5120)"
    )
    ap.add_argument(
        "--project",
        action="store_true",
        help="time the whole exl3_project_rdna2 (memset + 2 Hadamards + GEMM)",
    )
    ap.add_argument(
        "--graph", action="store_true", help="replay the timed loop as a graph"
    )
    args = ap.parse_args()
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

    ops = torch.ops._rocm_C
    dev = "cuda"
    ms = [int(m) for m in args.m.split(",")]
    torch.manual_seed(0)
    words = 16 * args.bits
    shapes = [("lm_head", 5120, 62080, 1)] if args.head else SHAPES_27B_TP4
    trellises = {}
    for name, k, n, _ in shapes:
        trellises[name] = torch.randint(
            -32768, 32767, (k // 16, n // 16, words), dtype=torch.int16, device=dev
        )
    totals = {}
    print(f"bits={args.bits} cb={args.cb}  (us per call; step = sum x layers)")
    hdr = "shape".ljust(18) + "".join(f"M={m:<9d}" for m in ms)
    print(hdr)
    for name, k, n, count in shapes:
        row = name.ljust(18)
        t = trellises[name]
        w_ref = None
        if args.check:
            w_ref = torch.zeros(k, n, dtype=torch.half, device=dev)
            ops.exl3_decode_trellis_rdna2(t, w_ref, args.bits, args.cb)
            w_ref = w_ref.float()
        for m in ms:
            a = torch.randn(m, k, dtype=torch.half, device=dev) * 0.05
            c = torch.zeros(m, n, dtype=torch.half, device=dev)
            for _ in range(3):
                c.zero_()
                ops.exl3_gemm_rdna2(a, c, t, args.bits, args.cb)
            torch.cuda.synchronize()
            if w_ref is not None:
                ref = a.float() @ w_ref
                err = (c.float() - ref).abs().max().item()
                scale = ref.abs().max().item() + 1e-6
                if err > 0.02 * scale + 0.05:
                    row += f"BAD({err:.2g})".ljust(11)
                    continue
            if args.project:
                xh = torch.zeros_like(a)
                out = torch.zeros_like(c)
                suh = torch.ones(k, dtype=torch.half, device=dev)
                svh = torch.ones(n, dtype=torch.half, device=dev)

                def step(a=a, xh=xh, c=c, out=out, t=t, suh=suh, svh=svh):
                    ops.exl3_project_rdna2(
                        a, xh, c, out, t, suh, svh, args.bits, args.cb
                    )
            else:

                def step(a=a, c=c, t=t):
                    c.zero_()
                    ops.exl3_gemm_rdna2(a, c, t, args.bits, args.cb)

            us = time_loop(step, args.iters, args.graph)
            totals[m] = totals.get(m, 0.0) + us * count
            row += f"{us:<11.1f}"
        print(row)
    print(
        "step(ms)".ljust(18) + "".join(f"{totals.get(m, 0) / 1000:<11.2f}" for m in ms)
    )


if __name__ == "__main__":
    main()
