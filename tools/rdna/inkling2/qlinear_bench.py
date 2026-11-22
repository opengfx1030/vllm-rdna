#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark of the weight-only quantized decode GEMV against the fp16 kernels.

For every weight family of the decode step (attention input pack of a normal / a full
attention layer, o_proj, lm_head with the fused argmax partials) it times
``inkling2_qlinear_*`` (int8 group 64, int4 group 32 / 64) and the fp16
``inkling2_linear_*`` it replaces with the method of ``docs/rdna2/hi2-prefill-gemm-2026-11-15.md``:
COLD weights (a rotating set of copies > ``--cold-mb`` MB, so every call reads DRAM), ONE
HIP graph over all copies, microseconds per call including the ~4 us graph gap of a
launch.  Every quantized case is also checked against the fp64 matmul with the dequantized
weights (``err`` is relative to the output rms).

    python qlinear_bench.py                    # installed ops, the production configs
    python qlinear_bench.py --sweep            # installed ops, the (R, cpi, waves) sweep
    python qlinear_bench.py --jit --formats 8:64,4:64 --rs 8,16 --cpis 4,8   # JIT instances
                                               # (qlinear_jit.py), formats of the sweep
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlinear_check import JitBackend, OpBackend  # noqa: E402

FAMILIES = {
    # name: (segment rows, K, input index per segment)
    "attn_in": ([3072, 3072, 384, 24, 1024, 1024], 2304, [0, 0, 0, 0, 1, 2]),
    "attn_in_full": (
        [3072, 3072, 384, 24, 1024, 32, 1024, 32],
        2304,
        [0, 0, 0, 0, 1, 1, 2, 2],
    ),
    "o_proj": ([2304], 3072, [0]),
    "lm_head": ([200058], 2304, [0]),
}


def time_graph(fns, replays=4):
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(3):
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        e1.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / (replays * len(fns)))
    return best


def make_weights(rows, K, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    ws = []
    for n in rows:
        w = torch.empty(n, K, dtype=torch.float16, device=dev)
        for lo in range(0, n, 16384):
            hi = min(n, lo + 16384)
            base = torch.randn(hi - lo, K, device=dev, generator=g)
            rs = 0.01 + 0.04 * torch.rand(hi - lo, 1, device=dev, generator=g)
            w[lo:hi] = (base * rs).half()  # row-dependent scale, like real weights
        ws.append(w)
    return ws


def copies(t, ncopy):
    return [t] + [t.clone() for _ in range(ncopy - 1)]


def run_case(qp, be, fam, bits, group, R, cpi, wv, M, args):
    rows, K, seg_x = FAMILIES[fam]
    dev = "cuda"
    ck = qp.CK[bits]
    gc = group // ck
    if K % (ck * (32 // R)) or (K // (ck * (32 // R))) % gc or cpi % gc:
        return None
    ws = make_weights(rows, K, dev)
    packs = [
        qp.pack_weight(w, bits, group, R, cpi, clip=bits == 4 and args.clip) for w in ws
    ]
    wq = torch.cat([p[0] for p in packs], 0).contiguous()
    sc = torch.cat([p[1] for p in packs], 0).contiguous()
    nbytes = wq.numel() + 2 * sc.numel()
    g = torch.Generator(device=dev).manual_seed(1)
    xs = [
        torch.randn(M, K, device=dev, generator=g).half() for _ in range(max(seg_x) + 1)
    ]
    outs32 = [torch.empty(M, n, dtype=torch.float32, device=dev) for n in rows]
    be.grouped(xs, wq, sc, outs32, seg_x, rows, [], [], [], bits, group, R, cpi, wv)
    torch.cuda.synchronize()
    err = rms = 0.0
    for o, w, sx in zip(outs32, ws, seg_x):
        q, s = qp.quantize_rows(w, bits, group, bits == 4 and args.clip)
        ref = xs[sx].double() @ qp.dequantize_rows(q, s, group).double().t()
        err = max(err, float((o.double() - ref).abs().max()))
        rms = max(rms, math.sqrt(float((ref**2).mean())))
    ncopy = max(2, math.ceil(args.cold_mb * 1e6 / nbytes))
    outs16 = [torch.empty(M, n, dtype=torch.float16, device=dev) for n in rows]
    fns = [
        (
            lambda w=w, s=s: be.grouped(
                xs, w, s, outs16, seg_x, rows, [], [], [], bits, group, R, cpi, wv
            )
        )
        for w, s in zip(copies(wq, ncopy), copies(sc, ncopy))
    ]
    us = time_graph(fns)
    return dict(us=us, nbytes=nbytes, err=err / rms, ncopy=ncopy)


def run_lm_head(qp, be, bits, group, cpi, M, args):
    rows, K, _ = FAMILIES["lm_head"]
    N, dev = rows[0], "cuda"
    ck = qp.CK[bits]
    gc = group // ck
    if K % ck or (K // ck) % gc or cpi % gc:
        return None
    clip = bits == 4 and args.clip
    w = make_weights(rows, K, dev)[0]
    g = torch.Generator(device=dev).manual_seed(1)
    x = torch.randn(M, K, device=dev, generator=g).half()
    wq, sc = qp.pack_weight(w, bits, group, 32, cpi, clip=clip)
    nbytes = wq.numel() + 2 * sc.numel()
    ref = torch.empty(M, N, dtype=torch.float64, device=dev)
    for lo in range(0, N, 16384):
        q, s = qp.quantize_rows(w[lo : lo + 16384], bits, group, clip)
        ref[:, lo : lo + 16384] = (
            x.double() @ qp.dequantize_rows(q, s, group).double().t()
        )
    ref = torch.tanh(ref / 30.0) * 30.0
    del w
    logits = torch.empty(M, N, dtype=torch.float32, device=dev)
    be.lm_head(x, wq, sc, logits, None, None, 30.0, N, N, 0, bits, group, cpi)
    torch.cuda.synchronize()
    err = float((logits.double() - ref).abs().max()) / math.sqrt(float((ref**2).mean()))
    # the fused argmax partials of the greedy path against the argmax of the logits
    pv = torch.empty(M, args.nparts, dtype=torch.float32, device=dev)
    pi = torch.empty(M, args.nparts, dtype=torch.int32, device=dev)
    be.lm_head(x, wq, sc, None, pv, pi, 30.0, N, N, 0, bits, group, cpi)
    torch.cuda.synchronize()
    ok = True
    for m in range(M):
        best = float(pv[m].max())
        cand = int(pi[m][pv[m] == best].min())
        mx = logits[m].max()
        ok &= cand == int((logits[m] == mx).nonzero().min()) and best == float(mx)
    ncopy = max(2, math.ceil(args.cold_mb * 1e6 / nbytes))
    fns = [
        (
            lambda w=w, s=s: be.lm_head(
                x, w, s, None, pv, pi, 30.0, N, N, 0, bits, group, cpi
            )
        )
        for w, s in zip(copies(wq, ncopy), copies(sc, ncopy))
    ]
    us = time_graph(fns)
    return dict(us=us, nbytes=nbytes, err=err, ncopy=ncopy, argmax_ok=ok)


def run_fp16(fam, M, args):
    from vllm import _custom_ops as ops

    rows, K, seg_x = FAMILIES[fam]
    dev = "cuda"
    w = torch.cat(make_weights(rows, K, dev), 0).contiguous()
    g = torch.Generator(device=dev).manual_seed(1)
    xs = [
        torch.randn(M, K, device=dev, generator=g).half() for _ in range(max(seg_x) + 1)
    ]
    outs = [torch.empty(M, n, dtype=torch.float16, device=dev) for n in rows]
    ncopy = max(2, math.ceil(args.cold_mb * 1e6 / (w.numel() * 2)))
    if fam == "lm_head":
        pv = torch.empty(M, args.nparts, dtype=torch.float32, device=dev)
        pi = torch.empty(M, args.nparts, dtype=torch.int32, device=dev)
        fns = [
            (
                lambda w=w: ops.inkling2_linear_lm_head(
                    xs[0], w, None, pv, pi, 30.0, rows[0], 0, -1
                )
            )
            for w in copies(w, ncopy)
        ]
    else:
        fns = [
            (
                lambda w=w: ops.inkling2_linear_grouped(
                    xs, w, outs, seg_x, rows, [], [], [], -1
                )
            )
            for w in copies(w, ncopy)
        ]
    return dict(us=time_graph(fns), nbytes=w.numel() * 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default="attn_in,attn_in_full,o_proj,lm_head")
    ap.add_argument("--jit", action="store_true", help="JIT build (qlinear_jit.py)")
    ap.add_argument("--source", help="kernel source for --jit (default: the repo's)")
    ap.add_argument("--name", default="i2q_bench")
    ap.add_argument("--filter", default="1", help="C++ instance filter for --jit")
    ap.add_argument("--defs", default="", help="C++ text before the kernel for --jit")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="sweep (R, cpi, waves) instead of production",
    )
    ap.add_argument("--formats", default="8:64,4:32,4:64")
    ap.add_argument("--rs", default="32,16,8,4")
    ap.add_argument("--cpis", default="4,8")
    ap.add_argument("--wvs", default="8")
    ap.add_argument("--ms", default="1,2,3,4")
    ap.add_argument("--nparts", type=int, default=64)
    ap.add_argument("--cold-mb", type=float, default=700.0)
    ap.add_argument("--clip", action="store_true", help="int4 with the clipping search")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    torch.cuda.init()
    from vllm.models.inkling2_loop.amd import qlinear_pack as qp

    fmts = [tuple(int(v) for v in f.split(":")) for f in args.formats.split(",")]
    if args.jit:
        import qlinear_jit

        t0 = time.time()
        be = JitBackend(
            qlinear_jit.build(
                args.name,
                args.source,
                fmts,
                extra_filter=args.filter,
                extra_defs=args.defs,
            )
        )
        print(f"JIT build {time.time() - t0:.0f}s", flush=True)
    else:
        be = OpBackend()
    fams = args.families.split(",")
    ms = [int(m) for m in args.ms.split(",")]
    fp16 = {}
    if not args.no_fp16:
        for fam in fams:
            for M in ms:
                r = fp16[(fam, M)] = run_fp16(fam, M, args)
                print(
                    f"FP16 {fam} M={M}: {r['us']:.1f} us {r['nbytes'] / r['us'] / 1e3:.0f} GB/s "
                    f"({r['nbytes'] / 1e6:.1f} MB)",
                    flush=True,
                )
    table = []
    for fam in fams:
        for bits, group in fmts:
            if args.sweep or args.jit:
                cfgs = [
                    (int(R), int(cpi), int(wv))
                    for R in args.rs.split(",")
                    for cpi in args.cpis.split(",")
                    for wv in args.wvs.split(",")
                ]
            elif fam == "lm_head":
                cfgs = [(32, qp.LM_HEAD_CPI, 8)]
            elif fam == "o_proj":
                cfgs = [qp.O_PROJ_CONFIG[bits]]
            else:
                cfgs = [qp.ATTN_IN_CONFIG[bits]]
            if fam == "lm_head":
                cfgs = sorted({(32, cpi, 8) for _, cpi, _ in cfgs})
            for R, cpi, wv in cfgs:
                for M in ms:
                    base = fp16.get((fam, M))
                    try:
                        if fam == "lm_head":
                            r = run_lm_head(qp, be, bits, group, cpi, M, args)
                        else:
                            r = run_case(qp, be, fam, bits, group, R, cpi, wv, M, args)
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"FAIL {fam} w{bits}g{group} R={R} cpi={cpi} wv={wv} M={M}: {str(e)[:300]}",
                            flush=True,
                        )
                        continue
                    if r is None:
                        continue
                    sp = f"{base['us'] / r['us']:.2f}x" if base else "-"
                    extra = f" argmax_ok={r['argmax_ok']}" if fam == "lm_head" else ""
                    print(
                        f"{fam} w{bits}g{group} R={R} cpi={cpi} wv={wv} M={M}: {r['us']:.1f} us "
                        f"{r['nbytes'] / r['us'] / 1e3:.0f} GB/s ({r['nbytes'] / 1e6:.1f} MB, x{r['ncopy']}) "
                        f"err/rms {r['err']:.1e}{extra}  vs fp16 {sp}",
                        flush=True,
                    )
                    table.append((fam, f"w{bits}g{group}", R, cpi, wv, M, r, base))
    print(
        "\n| family | format | R/cpi/waves | M | MB | us | GB/s | fp16 us | speedup |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for fam, fmt, R, cpi, wv, M, r, base in table:
        print(
            f"| {fam} | {fmt} | {R}/{cpi}/{wv} | {M} | {r['nbytes'] / 1e6:.1f} | {r['us']:.1f} | "
            f"{r['nbytes'] / r['us'] / 1e3:.0f} | {base['us'] if base else float('nan'):.1f} | "
            f"{base['us'] / r['us'] if base else float('nan'):.2f}x |"
        )


if __name__ == "__main__":
    main()
