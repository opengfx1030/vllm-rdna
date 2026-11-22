#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-launch time of the Inkling-2 MoE expert GEMMs (``moe_nvfp4_gemm_rdna2`` and
``moe_mxfp8_gemm_rdna2``) by token count and routed rows per expert block.

Real layer shapes (hidden 2304, expert width 1152, 128 experts, top-8), random
weights, synthetic routing (each token draws 8 distinct experts, optionally skewed
towards low expert ids), the engine's ``moe_align_block_size`` layout.  For each
(format, stage, tokens, block_size_m) it prints the mean microseconds of one launch
(the pre-zeroed output costs are subtracted) and the number of active expert blocks.
Run on one GPU: ``python tools/rdna/inkling2/moe_rows_bench.py [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json

import torch

HIDDEN, INTER, EXPERTS, TOP_K = 2304, 1152, 128, 8


def _routing(tokens: int, skew: float, gen: torch.Generator) -> torch.Tensor:
    pop = -skew * torch.linspace(0, 1, EXPERTS, device="cuda")
    logits = torch.randn(tokens, EXPERTS, device="cuda", generator=gen) + pop
    return logits.topk(TOP_K, dim=-1).indices.to(torch.int32).contiguous()


def _weights(fmt: str, k: int, n: int, gen: torch.Generator):
    def rnd_bytes(shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.int32, device="cuda", generator=gen
        )

    if fmt == "nvfp4":
        q = rnd_bytes((EXPERTS, k // 8, n), -(2**31), 2**31 - 1).to(torch.int32)
        scales = rnd_bytes((EXPERTS, k // 16, n), 0x20, 0x50).to(torch.uint8)
        glob = torch.full((EXPERTS,), 1e-3, dtype=torch.float32, device="cuda")
        return q, scales, glob
    # e4m3fn: 0x7F / 0xFF are NaN, keep the finite range
    raw = rnd_bytes((EXPERTS, k // 4, n, 4), 0, 0x7F) | (
        rnd_bytes((EXPERTS, k // 4, n, 4), 0, 2) << 7
    )
    q = (
        (raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16) | (raw[..., 3] << 24))
        .to(torch.int64)
        .to(torch.int32)
    )
    scales = rnd_bytes((EXPERTS, k // 32, n), 120, 128).to(torch.uint8)
    return q, scales, None


def _time(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.accelerator.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return 1e3 * start.elapsed_time(end) / iters


def run(args) -> dict:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    if args.jit:
        import pathlib
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from rdna.inkling2 import moe_jit

        ops = moe_jit.build("i2_moe_bench", args.jit)
    else:
        from vllm import _custom_ops as ops

    gen = torch.Generator(device="cuda").manual_seed(0)
    out: dict = {}
    stages = {"w13": (HIDDEN, 2 * INTER), "w2": (INTER, HIDDEN)}
    for fmt in ("nvfp4", "mxfp8"):
        for stage, (k, n) in stages.items():
            q, scales, glob = _weights(fmt, k, n, gen)
            fn = (
                ops.moe_nvfp4_gemm_rdna2 if fmt == "nvfp4" else ops.moe_mxfp8_gemm_rdna2
            )
            for tokens in args.tokens:
                ids = _routing(tokens, args.skew, gen)
                rows = tokens * TOP_K
                if stage == "w13":
                    a = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
                    c = torch.zeros(rows, n, device="cuda", dtype=torch.float16)
                    top_k, mul, otk = TOP_K, False, 0
                    tw = torch.empty(0, dtype=torch.float32, device="cuda")
                else:
                    a = torch.randn(rows, k, device="cuda", dtype=torch.float16)
                    c = torch.zeros(tokens, n, device="cuda", dtype=torch.float16)
                    top_k, mul, otk = 1, True, TOP_K
                    tw = torch.rand(rows, device="cuda", dtype=torch.float32)
                for bm in args.block_ms:
                    sorted_ids, expert_ids, ntpp = moe_align_block_size(
                        ids, bm, EXPERTS
                    )
                    active = int(ntpp.item()) // bm
                    extra = (glob,) if fmt == "nvfp4" else ()

                    call = (
                        a,
                        c,
                        q,
                        scales,
                        *extra,
                        tw,
                        sorted_ids,
                        expert_ids,
                        ntpp,
                        top_k,
                        bm,
                        mul,
                        otk,
                    )

                    def launch(fn=fn, call=call, c=c):
                        c.zero_()
                        fn(*call)

                    def zero(c=c):
                        c.zero_()

                    us = _time(launch, args.iters) - _time(zero, args.iters)
                    key = f"{fmt}/{stage}/T{tokens}/M{bm}"
                    out[key] = {"us": us, "active_blocks": active}
                    print(
                        f"{key:24s} {us:9.1f} us  active blocks {active:4d}", flush=True
                    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokens",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16],
    )
    ap.add_argument(
        "--block-ms",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 8],
    )
    ap.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="bias of the synthetic router towards low expert ids",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--jit", default="", help="bench a JIT build of this .cu file")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    result = run(args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
