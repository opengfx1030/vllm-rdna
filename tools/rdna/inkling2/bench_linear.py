#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark of the inkling2_linear_* kernels against rocBLAS (F.linear).

Every dense projection of Inkling-2-Mini at TP = 1 / 2 / 4, the native op next to
the torch composition the model runs today (``rocBLAS+torch``) and, for M <= 32,
the in-repo ``gemv_f16_rdna2`` skinny kernel (fp16 output, no epilogue, one launch
per weight). Weights rotate through --rot-mb of
distinct copies (the V620 has a 128 MB Infinity Cache, so a single hot weight
would measure the cache, not DRAM), the calls of one round are captured into one
HIP graph and timed by events, the implementations alternate round by round and
the median of the rounds is reported.

    python tools/rdna/inkling2/bench_linear.py --ops attn_in,o_proj,lm_head \\
        --m 1,4,16,1024 --tp 1,2,4

GB/s counts the weight bytes (decode is DRAM-bound), TFLOP/s counts 2 M N K
(prefill). The V620's v_dot2 peak is 18.4 TFLOP/s per GHz of shader clock.
``--variants`` additionally times forced kernel variants (sweep for tuning).
"""

import argparse
import contextlib
import csv
import statistics
import sys

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.models.inkling2_loop.amd import linear_pack as lp

H = 2304
HEAD_DIM = 128
D_REL = 16
CONV_K = 4
VOCAB = 200058
SOFTCAP = 30.0
OPS = (
    "attn_in",
    "q_proj",
    "kv_proj",
    "o_proj",
    "down",
    "router",
    "controller",
    "lm_head",
    "lm_head_argmax",
)
VARIANT = [-1]  # read by the closures when a graph is captured


def skinny(x, w):
    """The in-repo fp16 GEMV (gemv_f16_rdna2): fp16 output, no epilogue, M <= 32."""
    return ops.gemv_f16_rdna2(x, w, None)


def rand(*shape, dtype=torch.float16, scale=None):
    scale = scale if scale is not None else shape[-1] ** -0.5
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def build_attn_in(tp, m, dyn, nsets):
    nh, nkv = 24 // tp, 8 // tp
    rows = [nh * HEAD_DIM, nh * HEAD_DIM, nh * D_REL, nh, nkv * HEAD_DIM]
    base = [rand(r, H) for r in rows] + [rand(nkv * HEAD_DIM, H)]  # q g rel sink k v
    if dyn:
        base += [rand(nkv * CONV_K, H), rand(nkv * CONV_K, H)]  # dyn_k dyn_v
    xs = [rand(m, H, scale=1.0) for _ in range(3)]
    mine, ref, sk = [], [], []
    for i in range(nsets):
        ws = base if i == 0 else [w.clone() for w in base]
        pack = lp.inkling2_linear_pack_attn_in(*ws[:6], *(ws[6:] or (None, None)))
        outs = pack.alloc_outputs(m)
        order = ws[:5] + ([ws[6]] if dyn else []) + [ws[5]] + ([ws[7]] if dyn else [])
        sx = pack.seg_x
        mine.append(lambda p=pack, o=outs: p(xs, o, VARIANT[0]))
        ref.append(lambda ws=order, sx=sx: [F.linear(xs[s], w) for s, w in zip(sx, ws)])
        sk.append(lambda ws=order, sx=sx: [skinny(xs[s], w) for s, w in zip(sx, ws)])
    n = sum(w.shape[0] for w in base)
    return mine, ref, 2.0 * m * n * H, sum(w.numel() * 2 for w in base), sk


def build_single(k, n, m, nsets, out_dtype=torch.float16):
    base = rand(n, k)
    x = rand(m, k, scale=1.0)
    mine, ref, sk = [], [], []
    for i in range(nsets):
        w = base if i == 0 else base.clone()
        out = torch.empty(m, n, dtype=out_dtype, device="cuda")
        mine.append(
            lambda w=w, out=out: ops.inkling2_linear_grouped(
                [x], w, [out], [0], [n], variant=VARIANT[0]
            )
        )
        ref.append(lambda w=w: F.linear(x, w).to(out_dtype))
        if out_dtype == torch.float16:
            sk.append(lambda w=w: skinny(x, w))
    return mine, ref, 2.0 * m * n * k, base.numel() * 2, sk


def build_router(m, nsets):
    e = 128
    base = rand(e, H, dtype=torch.float32)
    x = rand(m, H, scale=1.0)
    mine, ref = [], []
    for i in range(nsets):
        pack = lp.inkling2_linear_pack_router(base if i == 0 else base.clone())
        out = pack.alloc_outputs(m)
        mine.append(lambda p=pack, o=out: p([x], o, VARIANT[0]))
        ref.append(lambda w=pack.weight: F.linear(x.float(), w))
    return mine, ref, 2.0 * m * e * H, base.numel() * 4, None


def build_controller(m, nsets):
    rank = 64
    wf, wg = rand(H, rank, dtype=torch.float32), rand(H, rank, dtype=torch.float32)
    gh = rand(m, rank, scale=1.0)
    mine, ref = [], []
    for i in range(nsets):
        a, b = (wf, wg) if i == 0 else (wf.clone(), wg.clone())
        pack = lp.inkling2_linear_pack_controller(a, b)
        outs = pack.alloc_outputs(m)
        mine.append(lambda p=pack, o=outs: p([gh], o, VARIANT[0]))
        ref.append(
            lambda a=a, b=b: (
                torch.tanh(F.linear(gh.float(), a)),
                torch.tanh(F.linear(gh.float(), b)),
            )
        )
    return mine, ref, 2.0 * m * 2 * H * rank, (wf.numel() + wg.numel()) * 4, None


def build_lm_head(tp, m, nsets):
    rows = -(-VOCAB // 64) * 64 // tp if tp > 1 else VOCAB
    base = rand(rows, H)
    x = rand(m, H, scale=1.0)
    mine, ref, sk = [], [], []
    for i in range(nsets):
        w = base if i == 0 else base.clone()
        logits = torch.empty(m, rows, dtype=torch.float32, device="cuda")
        mine.append(
            lambda w=w, lg=logits: ops.inkling2_linear_lm_head(
                x, w, lg, None, None, SOFTCAP, rows, 0, VARIANT[0]
            )
        )
        ref.append(lambda w=w: torch.tanh(F.linear(x, w).float() / SOFTCAP) * SOFTCAP)
        sk.append(lambda w=w: skinny(x, w))
    return mine, ref, 2.0 * m * rows * H, base.numel() * 2, sk


def build_lm_head_argmax(tp, m, nsets, nparts=64):
    """Greedy decode: lm_head + softcap + argmax -> token ids (int64 [m])."""
    rows = -(-VOCAB // 64) * 64 // tp if tp > 1 else VOCAB
    base = rand(rows, H)
    x = rand(m, H, scale=1.0)
    pv = torch.empty(m, nparts, dtype=torch.float32, device="cuda")
    pi = torch.empty(m, nparts, dtype=torch.int32, device="cuda")
    tok = torch.empty(m, dtype=torch.int64, device="cuda")
    mine, ref = [], []
    for i in range(nsets):
        w = base if i == 0 else base.clone()
        logits = None
        if m > 16:
            logits = torch.empty(m, rows, dtype=torch.float32, device="cuda")

        def fused(w=w, lg=logits):
            ops.inkling2_linear_lm_head(x, w, lg, pv, pi, SOFTCAP, rows, 0, VARIANT[0])
            ops.inkling2_linear_argmax_reduce(pv, pi, tok)

        mine.append(fused)
        ref.append(
            lambda w=w: (torch.tanh(F.linear(x, w).float() / SOFTCAP) * SOFTCAP).argmax(
                -1
            )
        )
    return mine, ref, 2.0 * m * rows * H, base.numel() * 2, None


def capture(fns):
    for f in fns[:2]:
        f()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay()
    torch.accelerator.synchronize()
    return g


def replay_us(g, calls, replays):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * calls)


def warm_clocks(seconds=1.5):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(20):
            a @ a
        end.record()
        end.synchronize()
        if start.elapsed_time(end) > seconds * 1e3:
            break


def run_case(op, tp, m, args, writer):
    builders = {
        "attn_in": lambda ns: build_attn_in(tp, m, args.dyn, ns),
        "q_proj": lambda ns: build_single(H, 3072 // tp, m, ns),
        "kv_proj": lambda ns: build_single(H, 1024 // tp, m, ns),
        "o_proj": lambda ns: build_single(
            3072 // tp, H, m, ns, torch.float32 if args.out_fp32 else torch.float16
        ),
        "down": lambda ns: build_single(2 * H, 64, m, ns),
        "router": lambda ns: build_router(m, ns),
        "controller": lambda ns: build_controller(m, ns),
        "lm_head": lambda ns: build_lm_head(tp, m, ns),
        "lm_head_argmax": lambda ns: build_lm_head_argmax(tp, m, ns, args.nparts),
    }
    nbytes = builders[op](1)[3]
    nsets = max(2, min(args.max_sets, int(args.rot_mb * 2**20 // nbytes) + 1))
    mine_fns, ref_fns, flop, nbytes, sk_fns = builders[op](nsets)
    graphs = {"rocBLAS+torch": capture(ref_fns)}
    if sk_fns and m <= 32:
        graphs["skinny gemv_f16"] = capture(sk_fns)
    for v in [-1] + [int(v) for v in args.variants.split(",") if v]:
        VARIANT[0] = v
        try:
            graphs[f"native v={v}"] = capture(mine_fns)
        except RuntimeError as e:  # a forced variant that does not fit this M
            if "M <= 16" not in str(e) and "<= 32" not in str(e):
                raise
    VARIANT[0] = -1
    times = {name: [] for name in graphs}
    names = list(graphs)
    for r in range(args.rounds):
        k = r % len(names)  # rotate the start: no implementation always runs first
        for name in names[k:] + names[:k]:
            replay_us(graphs[name], nsets, 1)  # untimed: let the clocks settle
            times[name].append(replay_us(graphs[name], nsets, args.replays))
    ref_us = statistics.median(times["rocBLAS+torch"])
    for name, ts in times.items():
        us = statistics.median(ts)
        row = {
            "op": op, "tp": tp, "M": m, "impl": name, "us": round(us, 2),
            "gbps": round(nbytes / us / 1e3, 1),
            "tflops": round(flop / us / 1e6, 2),
            "vs_ref": round(ref_us / us, 2), "sets": nsets,
        }  # fmt: skip
        print(
            f"{op:10s} tp{tp} M={m:5d} {name:16s} {us:9.2f} us {row['gbps']:7.1f} GB/s"
            f" {row['tflops']:6.2f} TFLOP/s  x{row['vs_ref']:.2f}",
            flush=True,
        )
        if writer:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--m", default="1,2,4,8,16,32,64,128,256,1024,4096")
    ap.add_argument("--tp", default="1")
    ap.add_argument("--dyn", action="store_true", help="attn_in with the dyn taps")
    ap.add_argument("--out-fp32", action="store_true", help="o_proj fp32 partials")
    ap.add_argument("--variants", default="", help="extra forced variants to time")
    ap.add_argument("--rot-mb", type=float, default=700)
    ap.add_argument("--max-sets", type=int, default=1500)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--nparts", type=int, default=64, help="lm_head_argmax blocks")
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(
        f"device: {props.name} {props.gcnArchName} "
        f"multi_processor_count={props.multi_processor_count}",
        flush=True,
    )
    warm_clocks()
    with contextlib.ExitStack() as stack:
        writer = None
        if args.csv:
            out = stack.enter_context(open(args.csv, "w", newline=""))
            writer = csv.DictWriter(
                out,
                fieldnames=[
                    "op", "tp", "M", "impl", "us", "gbps", "tflops", "vs_ref", "sets",
                ],
            )  # fmt: skip
            writer.writeheader()
        for op in args.ops.split(","):
            if op not in OPS:
                sys.exit(f"unknown op {op}")
            for tp in [int(t) for t in args.tp.split(",")]:
                if op in ("router", "controller", "down") and tp != 1:
                    continue
                for m in [int(v) for v in args.m.split(",")]:
                    run_case(op, tp, m, args, writer)
                    torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
