#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time gptq_gemm_rdna2_prefill for one shape with a forced split_k.

VLLM_RDNA2_PREFILL_FORCE_SPLIT_K is read once per process, so one process per
(m,n,k,force) tuple; the caller loops. Emits one CSV line. Single GPU.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_w4a8_op import _pack_weights, _quantize_random  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--force", type=int, required=True)
    ap.add_argument("--use-v2", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    os.environ["VLLM_RDNA2_PREFILL_FORCE_SPLIT_K"] = str(args.force)

    q_int4, scales_gn, zeros_gn = _quantize_random(args.k, args.n, args.group, seed=0)
    w_q, w_zp, w_s = _pack_weights(q_int4, scales_gn, zeros_gn, args.group)
    x = (0.25 * torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)).to(
        torch.float16
    )
    w_q_t = torch.from_numpy(w_q).to("cuda")
    w_zp_t = torch.from_numpy(w_zp).to("cuda")
    w_s_t = torch.from_numpy(w_s).to("cuda")
    g_idx = torch.empty(0, dtype=torch.int32, device="cuda")
    use_v2 = bool(args.use_v2)

    def call():
        return torch.ops._rocm_C.gptq_gemm_rdna2_prefill(
            x, w_q_t, w_zp_t, w_s_t, g_idx, use_v2
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    ms = times[len(times) // 2]
    print(f"{args.m},{args.k},{args.n},{args.group},{args.force},{ms:.4f}")


if __name__ == "__main__":
    main()
