#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level latency of the dense W4A16 prefill GEMM per production shape.

Times `_rocm_C.gptq_gemm_rdna2_prefill` (the W4A16 arm the W4A8 entry falls
back to) and reports the output norm/NaN flag. Run once per kernel build; with
VLLM_RDNA2_PREFILL_DEBUG=1 the kernel prints its chosen split on stdout
(`[rdna2_prefill_split] m=... n=... k=... split=...`), which is joined with the
latencies afterwards to compare the split choice across builds.

The per-shape buffers are byte-identical to what RDNA2W4A16LinearKernel leaves
behind (same packing as probe_w4a8_op.py). Single GPU.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_w4a8_op import (  # noqa: E402
    _pack_weights,
    _quantize_random,
)

PROD_SHAPES = [
    (m, k, n, 32)
    for m in (225, 2001, 2048)
    for k in (1536, 4352, 5120, 6144, 8704)
    for n in (3584, 4096, 5120, 8704)
]


def _bench(fn, warmup=3, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_shape(m, k, n, group, use_v2, iters):
    q_int4, scales_gn, zeros_gn = _quantize_random(k, n, group, seed=0)
    w_q, w_zp, w_s = _pack_weights(q_int4, scales_gn, zeros_gn, group)
    x = (0.25 * torch.randn((m, k), device="cuda", dtype=torch.float32)).to(
        torch.float16
    )
    w_q_t = torch.from_numpy(w_q).to("cuda")
    w_zp_t = torch.from_numpy(w_zp).to("cuda")
    w_s_t = torch.from_numpy(w_s).to("cuda")
    g_idx = torch.empty(0, dtype=torch.int32, device="cuda")

    def call():
        return torch.ops._rocm_C.gptq_gemm_rdna2_prefill(
            x, w_q_t, w_zp_t, w_s_t, g_idx, use_v2
        )

    ms = _bench(call, iters=iters)
    out = call()
    norm = float(out.to(torch.float32).norm())
    has_nan = bool(torch.isnan(out).any())
    return ms, norm, has_nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--shapes", default=None, help="m,k,n,group;... override")
    ap.add_argument("--variants", default="awq,gptq")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    else:
        shapes = PROD_SHAPES

    variants = {
        "awq": (True, False),   # uint4: use_v2_format=True
        "gptq": (False, True),  # uint4b8: use_v2_format=False
    }
    chosen = [v for v in args.variants.split(",") if v in variants]

    with open(args.out, "w") as f:
        f.write("m,k,n,group,variant,ms,norm,nan\n")
        for (m, k, n, group) in shapes:
            for v in chosen:
                use_v2, _gptq = variants[v]
                try:
                    ms, norm, has_nan = run_shape(m, k, n, group, use_v2, args.iters)
                except Exception as e:  # noqa: BLE001
                    f.write(f"{m},{k},{n},{group},{v},ERROR,0,1  # {e!r}\n")
                    f.flush()
                    continue
                f.write(f"{m},{k},{n},{group},{v},{ms:.4f},{norm:.4f},{int(has_nan)}\n")
                f.flush()
                print(f"({m},{k},{n},{group}) {v} {ms:.4f}ms norm={norm:.4f} nan={has_nan}")


if __name__ == "__main__":
    main()
