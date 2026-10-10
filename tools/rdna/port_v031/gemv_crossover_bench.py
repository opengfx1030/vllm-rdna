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
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT)
    ap.add_argument("--m", default="1,3,8,9,12,16,24")
    args = ap.parse_args()
    import vllm._custom_ops  # noqa: F401

    rc = torch.ops._rocm_C
    ms = [int(v) for v in args.m.split(",")]
    print("N x K".ljust(14) + "".join(f"M={m:<16d}" for m in ms))
    print("".ljust(14) + "".join("blas/gemv us    " for _ in ms))
    for shp in args.shapes.split(","):
        n, k = (int(v) for v in shp.split("x"))
        w = torch.randn(n, k, dtype=torch.half, device="cuda") * 0.02
        row = shp.ljust(14)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.half, device="cuda")
            tb = timeit(functools.partial(F.linear, x, w))
            tg = timeit(functools.partial(rc.gemv_f16_rdna2, x, w, None))
            mark = "*" if tg < tb else " "
            row += f"{tb:6.1f}/{tg:6.1f}{mark}  "
        print(row)
    print("* = gemv faster")


if __name__ == "__main__":
    main()
