#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert micro-benchmark: RDNA2 fused MoE W4A8 (sdot4) vs W4A16.

Runs the two ``torch.ops._rocm_C`` MoE entries on the Qwen3.8-Flash-Next AWQ
expert geometry (E=128 local, top_k=10, group_size=128) over a token sweep and
reports ms/call plus the W4A8/W4A16 speedup. A rel-L2 sanity check against the
W4A16 output guards against a fast-but-wrong kernel.

Usage:
  python probe_w4a8_moe.py [--experts 128] [--group 128] [--top-k 10]
                           [--iters 50] [--tokens 1,64,256,1024,2048]
"""

import argparse
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

device = "cuda"


def _packed_weights(e, k, n):
    q = torch.randint(0, 16, (e, k, n), dtype=torch.int32, device=device)
    packed = torch.zeros(e, k // 8, n, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q[:, i::8, :] & 0xF) << (i * 4)
    g_idx = torch.empty(0, dtype=torch.int32, device=device)
    for expert in range(e):
        we = packed[expert].contiguous()
        ops.gptq_shuffle(we, g_idx, 4)
        packed[expert] = we
    return packed


def _scales(e, groups, n):
    return (0.05 * torch.rand((e, groups, n), device=device) + 0.01).to(
        torch.float16
    )


def _qzeros(e, groups, n):
    zeros = torch.full(
        (groups, n),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    packed = pack_quantized_values_into_int32(
        zeros, scalar_types.uint4b8, packed_dim=1
    )
    return packed.unsqueeze(0).expand(e, -1, -1).contiguous()


def _make_case(e, k, n, group, tokens, top_k, block_size_m, seed):
    torch.manual_seed(seed)
    a = torch.randn(tokens, k, dtype=torch.float16, device=device)
    w = _packed_weights(e, k, n)
    s = _scales(e, k // group, n)
    z = _qzeros(e, k // group, n)
    ids = torch.randint(0, e, (tokens, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(ids, block_size_m, e)
    return a, w, s, z, si, ei, ntp


def _time(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _run(kind, a, c, w, s, z, si, ei, ntp, top_k, block_size_m, mul_topk,
         output_topk):
    if kind == "w4a8":
        ops.moe_w4a8_gemm_rdna2(
            a, c, w, s, z, torch.empty(0, device=device), si, ei, ntp, top_k,
            block_size_m, mul_topk, output_topk, False,
        )
    else:
        ops.moe_gptq_gemm_rdna2(
            a, c, w, s, z, torch.empty(0, device=device), si, ei, ntp, top_k,
            block_size_m, mul_topk, output_topk,
        )


def _rel_l2(got, ref):
    g, r = got.float(), ref.float()
    return ((g - r).norm() / r.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--block-size-m", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tokens", default="1,64,256,1024,2048")
    args = ap.parse_args()

    tokens_list = [int(t) for t in args.tokens.split(",")]
    # Flash-Next w1 (gate+up) and w2 (down) expert shapes.
    passes = [
        ("w13", 2560, 1280),
        ("w2", 640, 2560),
    ]
    print(
        f"E={args.experts} group={args.group} top_k={args.top_k} "
        f"bsm={args.block_size_m} iters={args.iters}"
    )
    for name, k, n in passes:
        print(f"\n== {name}: K={k} N={n} ==")
        print(f"{'tokens':>8} {'w4a16 ms':>10} {'w4a8 ms':>10} "
              f"{'speedup':>8} {'relL2':>9}")
        for tokens in tokens_list:
            params = _make_case(
                args.experts, k, n, args.group, tokens, args.top_k,
                args.block_size_m, seed=1000 + tokens,
            )
            a, w, s, z, si, ei, ntp = params
            flat = tokens * args.top_k
            c16 = torch.zeros(flat, n, dtype=torch.float16, device=device)
            c8 = torch.zeros(flat, n, dtype=torch.float16, device=device)

            _run("w4a16", a, c16, w, s, z, si, ei, ntp, args.top_k,
                 args.block_size_m, False, 0)
            torch.cuda.synchronize()
            ref = c16.clone()

            def run16():
                c16.zero_()
                _run("w4a16", a, c16, w, s, z, si, ei, ntp, args.top_k,
                     args.block_size_m, False, 0)

            def run8():
                c8.zero_()
                _run("w4a8", a, c8, w, s, z, si, ei, ntp, args.top_k,
                     args.block_size_m, False, 0)

            ms16 = _time(run16, args.iters)
            ms8 = _time(run8, args.iters)

            c8.zero_()
            _run("w4a8", a, c8, w, s, z, si, ei, ntp, args.top_k,
                 args.block_size_m, False, 0)
            torch.cuda.synchronize()
            rel = _rel_l2(c8, ref)

            print(f"{tokens:>8} {ms16:>10.3f} {ms8:>10.3f} "
                  f"{ms16 / ms8:>7.2f}x {rel:>9.5f}")
    print(f"\n[done {time.strftime('%H:%M:%S')}]")


if __name__ == "__main__":
    main()
