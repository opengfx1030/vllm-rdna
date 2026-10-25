# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench + accuracy check: RDNA2 W8A8 vs W8A16 int8 MoE GEMMs (one GPU).

Random int8 experts (symmetric per-channel scales) in both layouts: the natural
[E, K, N] int8 the W8A16 kernel reads and the packed [E, K/4, N] int32 the W8A8
kernel reads (rdna2_w8a8.pack_moe_weight), Flash-Next per-rank geometry by
default (EP4: 512 global / 128 local experts, top-10, hidden 2560, intermediate
640). For each token count it times gate_up (w13) and down (w2, fused top-k
reduce) on both kernels, reports the first-call wall time and the W8A8 relative
error against W8A16.

  python tools/rdna/bench_moe_w8a8.py [--tokens 1,4,64,256,1024,2048,4096]
      [--w8a8-bm 8] [--json out.json]
"""

import argparse
import functools
import json
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.rdna2_w8a8 import (
    act_group,
    pack_moe_weight,
)


def make_experts(e, k, n, dev):
    w = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, device=dev)
    scales = (torch.rand(e, 1, n, device=dev) * 0.02 + 0.005).half()
    w8a16 = w.transpose(1, 2).contiguous()  # [E, K, N]
    return dict(w8a8=pack_moe_weight(w), w8a16=w8a16, scales=scales)


def run(kind, bm, a, act, w13, w2, topk_ids, topk_w, local, emap, top_k):
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, bm, local, emap, ignore_invalid_experts=True
    )
    m = a.size(0)
    c1 = torch.zeros(
        m * top_k, w13["scales"].size(2), dtype=torch.half, device=a.device
    )
    c2 = torch.zeros(m, w2["scales"].size(2), dtype=torch.half, device=a.device)
    empty = topk_w.new_empty(0)
    if kind == "w8a8":
        ops.moe_w8a8_gemm_rdna2(
            a,
            c1,
            w13["w8a8"],
            w13["scales"],
            empty,
            sorted_ids,
            expert_ids,
            ntp,
            top_k,
            bm,
            False,
            0,
            act_group(a.size(1)),
        )
        ops.moe_w8a8_gemm_rdna2(
            act,
            c2,
            w2["w8a8"],
            w2["scales"],
            topk_w,
            sorted_ids,
            expert_ids,
            ntp,
            1,
            bm,
            True,
            top_k,
            act_group(act.size(1)),
        )
    else:
        no_zeros = torch.empty(0, dtype=torch.half, device=a.device)
        ops.moe_w8a16_gemm_rdna2(
            a,
            c1,
            w13["w8a16"],
            w13["scales"],
            no_zeros,
            empty,
            sorted_ids,
            expert_ids,
            ntp,
            top_k,
            bm,
            False,
        )
        ops.moe_w8a16_gemm_rdna2(
            act,
            c2,
            w2["w8a16"],
            w2["scales"],
            no_zeros,
            topk_w,
            sorted_ids,
            expert_ids,
            ntp,
            1,
            bm,
            True,
            output_topk=top_k,
        )
    return c1, c2


def timed(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,4,64,256,512,1024,2048,4096")
    p.add_argument("--w8a8-bm", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=640)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--local", type=int, default=128)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--json", default="")
    args = p.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    w13 = make_experts(args.local, args.hidden, 2 * args.inter, dev)
    w2 = make_experts(args.local, args.inter, args.hidden, dev)
    emap = torch.full((args.experts,), -1, dtype=torch.int32, device=dev)
    emap[: args.local] = torch.arange(args.local, dtype=torch.int32, device=dev)

    rows = []
    for m in [int(x) for x in args.tokens.split(",")]:
        a = torch.randn(m, args.hidden, dtype=torch.half, device=dev)
        act = torch.randn(m * args.top_k, args.inter, dtype=torch.half, device=dev)
        logits = torch.randn(m, args.experts, device=dev)
        topk_w, topk_ids = torch.topk(logits.softmax(-1), args.top_k, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        topk_w = topk_w.reshape(-1).float().contiguous()
        bm16 = 1 if m <= 4 else (8 if m >= 512 else 4)
        bm8 = 1 if m <= 4 else (args.w8a8_bm if m >= 512 else 4)
        common = (a, act, w13, w2, topk_ids, topk_w, args.local, emap, args.top_k)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r8 = run("w8a8", bm8, *common)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3
        r16 = run("w8a16", bm16, *common)
        torch.cuda.synchronize()

        def rel(x, y):
            return (
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-9)
            ).item()

        err1, err2 = rel(r8[0], r16[0]), rel(r8[1], r16[1])
        t16 = timed(functools.partial(run, "w8a16", bm16, *common), args.iters)
        t8 = timed(functools.partial(run, "w8a8", bm8, *common), args.iters)
        row = dict(
            tokens=m,
            bm_w8a16=bm16,
            bm_w8a8=bm8,
            w8a16_ms=round(t16, 3),
            w8a8_ms=round(t8, 3),
            speedup=round(t16 / t8, 3),
            w8a8_first_call_ms=round(first_ms, 1),
            rel_err_w13=round(err1, 4),
            rel_err_w2=round(err2, 4),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
