#!/usr/bin/env python3
"""QSA indexer scoring (qsa_mqa_paged) for one Flash-Next TP rank, timed in a
CUDA graph at the capture-time column bound (MAXLEN / ratio columns, as under
F&P decode):

  Triton      the Triton kernel without the early exit (no --monokernel)
  Triton+E3   the Triton kernel with the early exit (--monokernel today)
  HIP99+E3    #99's qsa_mqa_paged_rdna2 with the early exit (trees that have
              it built)
  HIP g<G>w<W> mono_qsa_mqa_rdna2, G rows per request scored together,
              W waves per block

Layouts: ``own`` (every row its own request: MTP0 decode) and ``mtp`` (rows in
runs of 3 per request at positions seq-3..seq-1: MTP2 verify) and ``pre``
(one request, rows at consecutive positions: a prefill chunk). Checks that
every variant gives the same visible_blocks and the same logits on
[0, visible) as the Triton kernel.

  PYTHONPATH=<tree> python3 qsa_mqa_bench.py [--rows 1,3,8,24] [--ctx 1024,16384]
"""

import argparse
import math
import re

import torch

MAXLEN = 262144
RATIO = 4
IH, ID = 4, 128  # indexer heads / dim per rank
PAGE = 64


def build(gen, dev, rows, ctx, width):
    if width == 0:  # prefill: one request, rows at consecutive positions
        width = rows
    cols_cap = MAXLEN // RATIO
    cpages_per_req = cols_cap // PAGE
    live_c = (ctx // RATIO + PAGE - 1) // PAGE
    reqs = (rows + width - 1) // width
    nc = reqs * live_c + 1
    ccache = (torch.randn(nc, PAGE, 1, ID, generator=gen, device=dev) * 0.5).half()
    table = torch.full((reqs, cpages_per_req), -1, dtype=torch.int32, device=dev)
    for r in range(reqs):
        table[r, :live_c] = torch.arange(
            1 + r * live_c, 1 + (r + 1) * live_c, device=dev
        )
    q = (torch.randn(rows, IH, ID, generator=gen, device=dev) * 0.5).half()
    token_to_req = (torch.arange(rows, device=dev) // width).int()
    seq = torch.full((reqs,), ctx, dtype=torch.int32, device=dev)
    off = width - 1 - (torch.arange(rows, device=dev) % width)
    pos = (ctx - 1 - off).clamp(min=0).long()
    return q, ccache, table, token_to_req, pos, seq, cols_cap


FLUSH = None  # --cold: a buffer larger than the 128 MB Infinity Cache


def flush_cache():
    torch.sum(FLUSH, dtype=torch.float32)


def graph_us(fn, calls=50, reps=10):
    if FLUSH is not None:  # cold: flush before every call, minus the flush
        return graph_us_raw(lambda: (flush_cache(), fn()), calls, reps) - (
            graph_us_raw(flush_cache, calls, reps)
        )
    return graph_us_raw(fn, calls, reps)


def graph_us_raw(fn, calls=50, reps=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / calls)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,4,6,8,12,16,24")
    ap.add_argument("--ctx", default="1024,4096,16384,65536")
    ap.add_argument("--layouts", default="own,mtp")
    ap.add_argument("--variants", default="g1w4,g1w8,g3w4,g4w4,g1w4t0,g1w4t64,g1w4t512")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--bounded",
        action="store_true",
        help="score width bounded by the live context as the --monokernel "
        "decode passes it (max(512, ceil(ctx / ratio / 64) * 64)); default: "
        "the capture-time worst case MAXLEN / ratio",
    )
    ap.add_argument(
        "--cold",
        action="store_true",
        help="flush the Infinity Cache before every call (the serve's long-"
        "context case: each QSA layer reads its own cache)",
    )
    a = ap.parse_args()
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as MQ
    from vllm.models.qwen4_exp.amd.ops import qsa as Q

    dev = torch.device("cuda")
    global FLUSH
    if a.cold:
        FLUSH = torch.ones(96 << 20, dtype=torch.float16, device=dev)  # 192 MB

    no_hip = type("E3", (), {"mqa_paged": staticmethod(lambda *x: False)})
    # #99's op (other trees carry an unrelated op of the same name)
    has99 = hasattr(torch.ops._rocm_C, "qsa_mqa_paged_rdna2") and "early_exit" in str(
        torch.ops._rocm_C.qsa_mqa_paged_rdna2.default._schema
    )

    def run(variant, args, cols):
        q, cc, table, t2r, pos, seq = args
        if variant == "tri":
            Q.MONO_QSA = None
        elif variant == "hip99":
            return torch.ops._rocm_C.qsa_mqa_paged_rdna2(
                q, cc, table, t2r, pos, seq, RATIO, cols, math.sqrt(ID), True
            )
        elif variant is None:  # Triton + E3
            Q.MONO_QSA = no_hip
        else:
            MQ.FORCE = variant
            Q.MONO_QSA = MQ
        try:
            return Q.qsa_mqa_paged(
                q,
                cc,
                table,
                t2r,
                pos,
                seq,
                RATIO,
                num_columns=cols,
                score_scale=math.sqrt(ID),
            )
        finally:
            Q.MONO_QSA = None
            MQ.FORCE = None

    variants = {"Triton": "tri", "Triton+E3": None}
    if has99:
        variants["HIP99+E3"] = "hip99"
    for v in filter(None, a.variants.split(",")):
        # g<G>w<W>[l<lanes per key>][u<unroll>][t<tile slots>]: t0 = one slot
        # per tile (no stride); defaults l8 u4 t auto
        f = dict(re.findall(r"([gwlutc])(\d+)", v))
        t = f.get("t") or f.get("c")  # c<N>: N contiguous-run slots
        tiles = (1 << 30 if t == "0" else int(t)) if t else 0
        if "c" in f:
            tiles = -tiles
        waves = -int(f["w"]) if v.endswith("n") else int(f["w"])  # n: streamed K
        q_lds = "q" in v  # q: the group's q rows in LDS
        variants[f"HIP {v}"] = (
            int(f["g"]),
            waves,
            tiles,
            int(f.get("l", 8)),
            int(f.get("u", 4)),
            q_lds,
        )
    print(
        f"{'layout':>6} {'ctx':>6} {'rows':>4} "
        + " ".join(f"{k:>10}" for k in variants)
        + "   (us/call, graph, columns = MAXLEN/ratio)"
    )
    ok = True
    for layout in a.layouts.split(","):
        width = {"own": 1, "mtp": 3, "pre": 0}[layout]
        for ctx in [int(x) for x in a.ctx.split(",")]:
            for rows in [int(x) for x in a.rows.split(",")]:
                gen = torch.Generator(device=dev).manual_seed(rows * 7 + ctx)
                q, cc, table, t2r, pos, seq, cols = build(gen, dev, rows, ctx, width)
                if a.bounded:
                    cols = min(cols, max(512, -(-ctx // RATIO // 64) * 64))
                args = (q, cc, table, t2r, pos, seq)
                ref_l, ref_v = run(None, args, cols)
                torch.cuda.synchronize()
                live = torch.arange(cols, device=dev)[None, :] < ref_v[:, None].long()
                for k, v in variants.items():
                    lg, vb = run(v, args, cols)
                    torch.cuda.synchronize()
                    same_v = torch.equal(vb, ref_v)
                    d = torch.where(live, (lg - ref_l).abs(), 0.0)
                    err = d.max().item() if d.numel() else 0.0
                    rel = err / max(ref_l[live].abs().max().item(), 1)
                    if not same_v or rel > 1e-3:
                        ok = False
                        print(
                            f"  MISMATCH {k} {layout} ctx={ctx} rows={rows} "
                            f"vis_eq={same_v} err={err:.3g}"
                        )
                calls = (10 if a.cold else 50) if rows <= 64 else 5
                times = {
                    k: graph_us(
                        lambda v=v, args=args, cols=cols: run(v, args, cols), calls
                    )
                    for k, v in variants.items()
                }
                hip = {k: t for k, t in times.items() if k.startswith("HIP g")}
                other = {k: t for k, t in times.items() if not k.startswith("HIP g")}
                best = min(hip, key=hip.get) if hip else ""
                rival = min(other, key=other.get)
                verdict = (
                    f"  best {best} {other[rival] / hip[best]:.2f}x vs {rival}"
                    if hip
                    else ""
                )
                print(
                    f"{layout:>6} {ctx:6d} {rows:4d} "
                    + " ".join(f"{times[k]:10.1f}" for k in variants)
                    + verdict
                )
    print("correctness:", "ALL OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
