#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the fp16 lm_head GEMM at decode / verify batch sizes.

    HIP_VISIBLE_DEVICES=6 python tools/rdna/port_v031/lm_head_bench.py \
        [--n 62080] [--k 5120] [--m 1,3,8,24]

Default shape: Qwen3.8 vocab 248320 / TP=4 x hidden 5120 (the EXL3 6bpw
head is folded to a dense fp16 weight at load).
"""

import argparse
import functools

import torch
import torch.nn.functional as F


def timeit(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=62080)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--m", default="1,3,6,8,12,24")
    args = ap.parse_args()
    import vllm._custom_ops  # noqa: F401

    w = torch.randn(args.n, args.k, dtype=torch.half, device="cuda") * 0.02
    gb = w.numel() * 2 / 1e9
    print(f"W [{args.n}, {args.k}] fp16 = {gb:.3f} GB")
    rc = torch.ops._rocm_C
    for m in [int(v) for v in args.m.split(",")]:
        x = torch.randn(m, args.k, dtype=torch.half, device="cuda")
        ref = F.linear(x, w)
        ms = timeit(functools.partial(F.linear, x, w))
        line = f"M={m:<3d} F.linear {ms:7.3f} ms ({gb / ms * 1e3:6.1f} GB/s)"
        if m <= 32 and hasattr(rc, "gemv_f16_rdna2"):
            out = rc.gemv_f16_rdna2(x, w, None)
            err = (out.float() - ref.float()).abs().max().item()
            ms = timeit(functools.partial(rc.gemv_f16_rdna2, x, w, None))
            line += f" | gemv_f16_rdna2 {ms:7.3f} ms ({gb / ms * 1e3:6.1f} GB/s"
            line += f", err {err:.3g})"
        print(line)


if __name__ == "__main__":
    main()
