#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gemv_f16_rdna2 vs F.linear (rocBLAS + TunableOp rows) per (N, K, M).

Run with the serving TunableOp env so F.linear uses the shipped rows
(ROWS=<tree>/tunableop/rocm7.14-rocblas5.5):

    PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
    PYTORCH_TUNABLEOP_FILENAME=$ROWS/tunableop_results.csv \
    HIP_VISIBLE_DEVICES=5 python tools/rdna/port_v031/gemv_crossover_bench.py \
        [--shapes 10240x2560,...] [--m 1,3,8,9,12,16,24]
"""

import argparse
import functools

import torch
import torch.nn.functional as F

# Flash-Next / 27B AWQ fp16 dense shapes (N x K per rank) from the shipped rows.
DEFAULT = (
    "10240x2560,10240x320,24x2560,2560x1536,2560x2560,320x10240,320x2560,"
    "336x10240,3584x2560,4096x2560,512x2560,62080x2560,640x2560,62080x5120"
)


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def timeit_cold(calls, replays=20):
    """Graph-replay time per call over distinct weight sets (one set would sit
    in the 128 MB Infinity Cache), launch gaps included."""
    for f in calls[:3]:
        f()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    g.replay()
    torch.accelerator.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(replays):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / (replays * len(calls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT)
    ap.add_argument("--m", default="1,3,8,9,12,16,24")
    ap.add_argument(
        "--cold",
        action="store_true",
        help="graph replay over distinct weight sets (about 160 MB) instead of "
        "eager calls on one cache-resident weight",
    )
    args = ap.parse_args()
    import vllm._custom_ops  # noqa: F401

    rc = torch.ops._rocm_C
    ms = [int(v) for v in args.m.split(",")]
    has_smallm = hasattr(rc, "rdna_gemm_smallm")
    print("N x K".ljust(14) + "".join(f"M={m:<22d}" for m in ms))
    print("".ljust(14) + "".join("blas/gemv/smallm us   " for _ in ms))
    for shp in args.shapes.split(","):
        n, k = (int(v) for v in shp.split("x"))
        w = torch.randn(n, k, dtype=torch.half, device="cuda") * 0.02
        ws = [w]
        if args.cold:
            nw = max(1, min(1024, -(-160_000_000 // (n * k * 2))))
            ws += [torch.randn_like(w) * 0.02 for _ in range(nw - 1)]
        row = shp.ljust(14)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.half, device="cuda")
            cands = [
                ("b", lambda v, x=x: F.linear(x, v)),
                ("g", lambda v, x=x: rc.gemv_f16_rdna2(x, v, None)),
            ]
            if has_smallm and m <= 32 and k % 32 == 0 and n % 4 == 0 and n <= 6144:
                cands.append(("s", lambda v, x=x: rc.rdna_gemm_smallm(x, v)))
            times = {}
            for tag, fn in cands:
                if args.cold:
                    times[tag] = timeit_cold([functools.partial(fn, v) for v in ws])
                else:
                    times[tag] = timeit(functools.partial(fn, w))
            best = min(times, key=times.get)
            cell = "/".join(
                f"{times[t]:5.1f}" if t in times else "  -  " for t in "bgs"
            )
            row += f"{cell}{best}  "
        print(row)
    print(
        "last letter = fastest: b rocBLAS (TunableOp), g gemv_f16_rdna2, "
        "s rdna_gemm_smallm"
    )


if __name__ == "__main__":
    main()
