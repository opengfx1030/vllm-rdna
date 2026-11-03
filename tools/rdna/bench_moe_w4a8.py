# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench + accuracy check: RDNA2 W4A8 vs W4A16 MoE GEMMs (one GPU).

Random int4 experts in the RDNA2 MoE layout ([E, K/8, N] shuffled-equivalent
random dwords, symmetric zeros), Flash-Next per-rank geometry by default
(EP4: 512 global / 128 local experts, top-10, hidden 2560, intermediate 640,
group 128). For each token count it times gate_up (w13) and down (w2, fused
top-k reduce) on both kernels, reports the first-call wall time (catches
per-shape one-off stalls) and the W4A8 relative error against W4A16.

  python tools/rdna/bench_moe_w4a8.py [--tokens 64,256,1023,2048,4096]
      [--w4a8-bm 8] [--json out.json]

  python tools/rdna/bench_moe_w4a8.py --tokens 2048 --variants 8:0,8:640,16:0
      times the W4A8 launch variants (block_size_m:k_block, the
      VLLM_RDNA_W4A8_KBLOCK knob; 0 = original kernel, -1 = whole K) per GEMM
      and checks each against the first.
"""

import argparse
import functools
import json
import os
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def resolve_device(device_str):
    """Resolve the torch device index and guard physical HIP devices 0 and 1.

    The torch index is mapped through ``HIP_VISIBLE_DEVICES`` when it is set
    (comma list; unset = identity). Only the physical index is refused.
    """
    if device_str.isdigit():
        torch_idx = int(device_str)
    else:
        d = torch.device(device_str)
        torch_idx = d.index if d.index is not None else 0
    hip = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip:
        vis = [int(x.strip()) for x in hip.split(",") if x.strip()]
        if torch_idx >= len(vis):
            print(
                f"torch index {torch_idx} is out of range for HIP_VISIBLE_DEVICES={hip}"
            )
            sys.exit(1)
        phys = vis[torch_idx]
    else:
        phys = torch_idx
    if phys in (0, 1):
        print(
            f"Refusing physical device {phys} (torch index {torch_idx}): "
            "devices 0 and 1 are reserved."
        )
        sys.exit(1)
    name = torch.cuda.get_device_name(torch_idx)
    print(f"Running on torch device {torch_idx} (physical HIP {phys}): {name}")
    torch.accelerator.set_device_index(torch_idx)
    return torch_idx


def make_experts(e, k, n, group, dev):
    w = torch.randint(
        -(2**31), 2**31 - 1, (e, k // 8, n), dtype=torch.int32, device=dev
    )
    scales = (torch.rand(e, k // group, n, device=dev) * 0.02 + 0.005).half()
    # GPTQv1 symmetric: stored nibble 7 -> zero point 8 with zero_offset 1.
    qzeros = torch.full(
        (e, k // group, n // 8), 0x77777777, dtype=torch.int32, device=dev
    )
    return w, scales, qzeros


def run(kind, bm, a, act, w13, w2, topk_ids, topk_w, local, emap, top_k):
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, bm, local, emap, ignore_invalid_experts=True
    )
    m = a.size(0)
    c1 = torch.zeros(m * top_k, w13[0].size(2), dtype=torch.half, device=a.device)
    c2 = torch.zeros(m, w2[0].size(2), dtype=torch.half, device=a.device)
    empty = topk_w.new_empty(0)
    gemm = ops.moe_w4a8_gemm_rdna2 if kind == "w4a8" else ops.moe_gptq_gemm_rdna2
    gemm(
        a,
        c1,
        w13[0],
        w13[1],
        w13[2],
        empty,
        sorted_ids,
        expert_ids,
        ntp,
        top_k,
        bm,
        False,
        0,
    )
    gemm(
        act,
        c2,
        w2[0],
        w2[1],
        w2[2],
        topk_w,
        sorted_ids,
        expert_ids,
        ntp,
        1,
        bm,
        True,
        top_k,
    )
    return c1, c2


def timed(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def make_inputs(args, dev, m):
    """Activations, routing ids and router weights for ``m`` tokens."""
    a = torch.randn(m, args.hidden, dtype=torch.half, device=dev)
    act = torch.randn(m * args.top_k, args.inter, dtype=torch.half, device=dev)
    if args.hist:
        with open(args.hist) as f:
            lines = [json.loads(line) for line in f]
        rec = None
        for line in lines:
            if "counts" in line and line.get("tokens", 0) >= 512:
                rec = line
                break
        if rec is None:
            raise ValueError(f"No valid histogram record found in {args.hist}")
        counts = rec["counts"]
        rank = args.hist_rank
        local_counts = counts[rank * 128 : (rank + 1) * 128]
        total = sum(local_counts)
        probs = torch.tensor(
            [c / total for c in local_counts], dtype=torch.float32, device=dev
        )
        topk_ids = torch.multinomial(
            probs.expand(m, -1), args.top_k, replacement=False
        ).to(torch.int32)
        topk_w = torch.rand(m, args.top_k, device=dev).float().contiguous().reshape(-1)
    elif args.routing == "zipf":
        s = 1.0
        probs = 1.0 / (torch.arange(1, args.local + 1, device=dev).float() ** s)
        probs = probs / probs.sum()
        topk_ids = torch.multinomial(
            probs.expand(m, -1), args.top_k, replacement=False
        ).to(torch.int32)
        topk_w = torch.rand(m, args.top_k, device=dev).float().contiguous().reshape(-1)
    else:
        logits = torch.randn(m, args.experts, device=dev)
        topk_w, topk_ids = torch.topk(logits.softmax(-1), args.top_k, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        topk_w = topk_w.reshape(-1).float().contiguous()
    return a, act, topk_ids, topk_w


def set_variant(kb, nt=0):
    """Select the W4A8 MoE launch variant (``kb`` / ``nt`` 0 = shipped kernel)."""
    for name, value in (
        ("VLLM_RDNA_W4A8_KBLOCK", kb),
        ("VLLM_RDNA_W4A8_NTILE", nt),
    ):
        if value == 0:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def rel_diff(x, y):
    return ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-9)).item()


def timed_for(fn, target_ms):
    """Per-call ms of ``fn`` over a loop sized to about ``target_ms`` of GPU time."""
    est = timed(fn, 3)
    return timed(fn, max(5, int(target_ms / max(est, 1e-3))))


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def bench_variants(args, dev, w13, w2, emap):
    """W4A8 launch variants, timed per GEMM with the routing fixed.

    ``--variants 8:0,8:640,16:0`` is a list of ``block_size_m:k_block``
    (k_block 0 = the original kernel, -1 = whole K). Timings are the GEMMs
    alone (no output zero-fill, no align kernel); outputs are compared with
    the first variant on freshly zeroed buffers.

    The card's clocks follow the load, so a short timing loop right after
    idle host work reads slow: every sample is preceded by a warm-up on the
    same variant, sized in milliseconds rather than iterations, and the
    variants are visited round-robin (forward, then reversed) for
    ``--rounds`` rounds. The reported time is the median round.
    """
    variants = []
    for item in args.variants.split(","):
        bm, kb, *nt = item.split(":")
        variants.append((int(bm), int(kb), int(nt[0]) if nt else 0))
    top_k = args.top_k
    rows = []
    for m in [int(x) for x in args.tokens.split(",")]:
        a, act, topk_ids, topk_w = make_inputs(args, dev, m)
        empty = topk_w.new_empty(0)
        setups = []
        for bm, kb, nt in variants:
            set_variant(kb, nt)
            route = moe_align_block_size(
                topk_ids, bm, args.local, emap, ignore_invalid_experts=True
            )
            c1 = torch.zeros(m * top_k, w13[0].size(2), dtype=torch.half, device=dev)
            c2 = torch.zeros(m, w2[0].size(2), dtype=torch.half, device=dev)
            f13 = functools.partial(
                ops.moe_w4a8_gemm_rdna2, a, c1, *w13, empty, *route, top_k, bm, False, 0
            )
            f2 = functools.partial(
                ops.moe_w4a8_gemm_rdna2,
                act,
                c2,
                *w2,
                topk_w,
                *route,
                1,
                bm,
                True,
                top_k,
            )
            f13()
            f2()
            torch.accelerator.synchronize()
            setups.append(
                dict(bm=bm, kb=kb, nt=nt, f13=f13, f2=f2, out=(c1.clone(), c2.clone()))
            )
        first = setups[0]["out"]
        samples = [[] for _ in setups]
        for rnd in range(args.rounds):
            order = list(range(len(setups)))
            if rnd % 2:
                order.reverse()
            for i in order:
                s = setups[i]
                set_variant(s["kb"], s["nt"])
                timed_for(s["f13"], args.warm_ms)
                t13 = timed_for(s["f13"], args.sample_ms)
                timed_for(s["f2"], args.warm_ms / 2)
                t2 = timed_for(s["f2"], args.sample_ms)
                samples[i].append((t13, t2))
        base_total = None
        for s, smp in zip(setups, samples):
            sums = [t13 + t2 for t13, t2 in smp]
            total = median(sums)
            if base_total is None:
                base_total = total
            row = dict(
                tokens=m,
                bm=s["bm"],
                k_block=s["kb"],
                n_tile=s["nt"],
                w13_ms=round(median([t[0] for t in smp]), 3),
                w2_ms=round(median([t[1] for t in smp]), 3),
                sum_ms=round(total, 3),
                sum_min_ms=round(min(sums), 3),
                spread_pct=round(100 * (max(sums) - min(sums)) / total, 1),
                vs_first=round(base_total / total, 3),
                rel_err_w13=round(rel_diff(s["out"][0], first[0]), 5),
                rel_err_w2=round(rel_diff(s["out"][1], first[1]), 5),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    set_variant(0)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="64,256,512,1023,1024,2048,4096")
    p.add_argument("--w4a8-bm", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=640)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--local", type=int, default=128)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="--variants rounds")
    p.add_argument("--warm-ms", type=float, default=150.0, help="--variants warm-up")
    p.add_argument("--sample-ms", type=float, default=100.0, help="--variants sample")
    p.add_argument("--w4a16-bm", default="")
    p.add_argument("--hist", default="")
    p.add_argument("--hist-rank", type=int, default=0)
    p.add_argument("--routing", default="random", choices=["random", "zipf"])
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--json", default="")
    p.add_argument(
        "--variants",
        default="",
        help="W4A8 launch variants block_size_m:k_block,... (k_block 0 = original "
        "kernel, -1 = whole K); times w13 and w2 separately and skips the W4A16 "
        "comparison.",
    )
    args = p.parse_args()

    torch_idx = resolve_device(args.device)
    dev = torch.device(f"cuda:{torch_idx}")
    torch.manual_seed(0)
    w13 = make_experts(args.local, args.hidden, 2 * args.inter, args.group, dev)
    w2 = make_experts(args.local, args.inter, args.hidden, args.group, dev)
    emap = torch.full((args.experts,), -1, dtype=torch.int32, device=dev)
    emap[: args.local] = torch.arange(args.local, dtype=torch.int32, device=dev)

    if args.variants:
        rows = bench_variants(args, dev, w13, w2, emap)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return

    rows = []
    for m in [int(x) for x in args.tokens.split(",")]:
        a, act, topk_ids, topk_w = make_inputs(args, dev, m)
        if args.w4a16_bm:
            bm16_list = [int(x) for x in args.w4a16_bm.split(",")]
        else:
            bm16_list = [1 if m <= 4 else (8 if m >= 512 else 4)]
        bm8 = args.w4a8_bm
        common = (a, act, w13, w2, topk_ids, topk_w, args.local, emap, args.top_k)

        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        r8 = run("w4a8", bm8, *common)
        torch.accelerator.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3

        def rel(x, y):
            return (
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-9)
            ).item()

        r16_list = []
        for bm16 in bm16_list:
            r16 = run("w4a16", bm16, *common)
            torch.accelerator.synchronize()
            r16_list.append(r16)

        err1 = rel(r8[0], r16_list[0][0])
        err2 = rel(r8[1], r16_list[0][1])
        t8 = timed(functools.partial(run, "w4a8", bm8, *common), args.iters)
        t16_list = []
        for bm16 in bm16_list:
            t16 = timed(functools.partial(run, "w4a16", bm16, *common), args.iters)
            t16_list.append(t16)
        row = dict(
            tokens=m,
            bm_w4a16=bm16_list[0],
            bm_w4a8=bm8,
            w4a16_ms=round(t16_list[0], 3),
            w4a8_ms=round(t8, 3),
            speedup=round(t16_list[0] / t8, 3),
            w4a8_first_call_ms=round(first_ms, 1),
            rel_err_w13=round(err1, 4),
            rel_err_w2=round(err2, 4),
        )
        if len(bm16_list) > 1:
            row["w4a16_bm_list"] = bm16_list
            row["w4a16_ms_list"] = [round(t, 3) for t in t16_list]
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.w4a16_bm:
            print(
                f"TABLE {m:>6} "
                + " ".join(f"tile{bm}={t:.3f}" for bm, t in zip(bm16_list, t16_list))
                + f" w4a8={t8:.3f}",
                file=sys.stderr,
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
