#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chain-of-N graph benchmark of the fused attention-chain kernels vs the launches
they replace (``csrc/rocm/inkling2_fuse_attn_rdna2.cu``).

    python tools/rdna/inkling2/bench_fuse_attn.py [--tokens 1] [--units 48] [--trace]

Every variant is one HIP graph of ``--units`` visits.  Between two visits one flush
kernel streams ``--flush-mb`` MB (more than the 4 MB L2), so the small tensors of a
visit (conv state / weights, block states, metadata) start cold like they do between
the 80+ MB weight GEMVs of a real decode step; the per-visit weights and conv state are
distinct for every visit (rotating).  The cost of a variant is
``(chain - flush-only chain) / units`` microseconds per visit; the saving of a fusion
is the difference of two variants.  ``--trace`` adds the kernel durations of a profiler
run (the rest of the difference is dispatch gap).
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import statistics

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]


def load_test_module():
    spec = importlib.util.spec_from_file_location(
        "fuse_attn_test",
        REPO / "tests" / "kernels" / "core" / "test_inkling2_fuse_attn_rdna2.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_graph(fn, reps: int, warm: int = 3) -> float:
    """Median replay time (us) of the graph captured from ``fn``."""
    fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.accelerator.synchronize()
    ts = []
    for _ in range(reps):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def time_graphs(fns, reps: int, warm: int = 3) -> list[float]:
    """Median replay time (us) of the graph of every ``fns``; the replays alternate, so
    slow drifts (clocks, temperature) hit all of them alike."""
    gs = []
    for fn in fns:
        fn()
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        gs.append(g)
    for g in gs:
        for _ in range(warm):
            g.replay()
    torch.accelerator.synchronize()
    ts: list[list[float]] = [[] for _ in gs]
    for _ in range(reps):
        for k, g in enumerate(gs):
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            ts[k].append(s.elapsed_time(e) * 1e3)
    return [statistics.median(t) for t in ts]


def kernel_sums(fn, steps: int = 4) -> dict[str, tuple[int, float]]:
    """Kernel name -> (count per replay, mean duration us) over ``steps`` replays."""
    from torch.profiler import ProfilerActivity, profile

    g = torch.cuda.CUDAGraph()
    fn()
    torch.accelerator.synchronize()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            g.replay()
        torch.accelerator.synchronize()
    acc: dict[str, list[float]] = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            acc.setdefault(ev.name, []).append(ev.device_time)
    return {k: (len(v) // steps, sum(v) / len(v)) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--units", type=int, default=48)
    ap.add_argument("--flush-mb", type=float, default=16.0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ring", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    torch.cuda.init()

    tm = load_test_module()
    ref, ops = tm.ref_ops, tm.ops
    H, NKV, DEV = tm.H, tm.NKV, tm.DEV
    T, U, nb = args.tokens, args.units, args.n_blocks
    world = tm.World(T, 0, 0, args.bs, args.ring, seed=1, seq_lens=[200] * T)
    table, seq_idx, qsl, seq_lens = world.meta()

    # shared like the persistent workspace of a model (one buffer per role)
    x16 = world.x.to(DEV)  # moe_out / o_out
    blocks = tm.make_states(T, nb, False, 3)
    hid = tm.rand_f32((T, H), 4)
    res = torch.zeros(T, H, device=DEV)
    new_x0 = tm.rand_f32((T, H), 5)
    mlp_c = torch.zeros(T, H, device=DEV)
    attn_c = torch.zeros(T, H, device=DEV)
    out1 = torch.zeros(T, 1, H, dtype=torch.float16, device=DEV)
    out3 = torch.zeros(T, 3, H, dtype=torch.float16, device=DEV)
    raw1 = torch.zeros(T, 1, H, device=DEV)
    # rotating per-visit tensors
    caches = [world.cache0.clone().to(DEV) for _ in range(U)]
    cws = [(world.w + 0.01 * i).to(DEV) for i in range(U)]
    q1 = [tm.rand_f32((1, H), 10 + i, 0.5) for i in range(U)]
    q3 = [tm.rand_f32((3, H), 70 + i, 0.5) for i in range(U)]
    nw1 = [(tm.rand_f32((1, H), 130 + i, 0.3) + 1.0).half() for i in range(U)]
    nw3 = [(tm.rand_f32((3, H), 190 + i, 0.3) + 1.0).half() for i in range(U)]
    flush = torch.zeros(max(int(args.flush_mb * (1 << 18)), 1), device=DEV)

    off_a, off_m = tm.OFFSET, 544

    def flush_k():
        if args.flush_mb > 0:
            flush.add_(1.0)

    def conv_ref(i, off, out):
        ref.inkling2_sconv_static(
            x16, cws[i], out, caches[i], table, seq_idx, qsl, seq_lens, off, NKV,
            args.bs, True,
        )  # fmt: skip

    def attn_res(i, a0, a1, q, out, **kw):
        ref.inkling2_glue_attn_res(
            blocks, nb, a0, q, out, x1=a1, eps=tm.EPS, norm_eps=tm.EPS, **kw
        )

    def fused(i, off, x0, q, out, **kw):
        ops.attn_res_conv(
            blocks, nb, x0, x16, cws[i], caches[i], table, seq_idx, qsl, seq_lens, off,
            NKV, args.bs, kw.get("x1_out"), kw.get("partial_out"), kw.get("append"),
            q, kw.get("norm_w"), out, tm.EPS, tm.EPS,
        )  # fmt: skip

    # name -> (unfused chain, fused chain); one visit per call
    cases = {
        # post-MLP conv + attention-input attn_res (sliding, no messages)
        "A_norm_q1": (
            lambda i: (
                conv_ref(i, off_m, mlp_c),
                attn_res(i, new_x0, mlp_c, q1[i], out1, partial_out=hid, norm_w=nw1[i]),
            ),
            lambda i: fused(
                i, off_m, new_x0, q1[i], out1, partial_out=hid, norm_w=nw1[i]
            ),
        ),
        # ... full-attention layer: three layer norms
        "A_norm_q3": (
            lambda i: (
                conv_ref(i, off_m, mlp_c),
                attn_res(i, new_x0, mlp_c, q3[i], out3, partial_out=hid, norm_w=nw3[i]),
            ),
            lambda i: fused(
                i, off_m, new_x0, q3[i], out3, partial_out=hid, norm_w=nw3[i]
            ),
        ),
        # ... message visit: raw output (the gates follow)
        "A_raw_q1": (
            lambda i: (
                conv_ref(i, off_m, mlp_c),
                attn_res(i, new_x0, mlp_c, q1[i], raw1, partial_out=hid),
            ),
            lambda i: fused(i, off_m, new_x0, q1[i], raw1, partial_out=hid),
        ),
        # post-attention conv + MLP-input attn_res (not a block boundary)
        "M_norm": (
            lambda i: (
                conv_ref(i, off_a, attn_c),
                attn_res(i, hid, attn_c, q1[i], out1, partial_out=res, norm_w=nw1[i]),
            ),
            lambda i: fused(i, off_a, hid, q1[i], out1, partial_out=res, norm_w=nw1[i]),
        ),
        # ... block boundary: conv alone is the partial state, hid is appended
        "M_norm_boundary": (
            lambda i: (
                conv_ref(i, off_a, attn_c),
                attn_res(i, attn_c, None, q1[i], out1, append=hid, norm_w=nw1[i]),
            ),
            lambda i: fused(
                i, off_a, None, q1[i], out1, append=hid, norm_w=nw1[i], x1_out=attn_c
            ),
        ),
        "M_raw": (
            lambda i: (
                conv_ref(i, off_a, attn_c),
                attn_res(i, hid, attn_c, q1[i], raw1, partial_out=res),
            ),
            lambda i: fused(i, off_a, hid, q1[i], raw1, partial_out=res),
        ),
    }
    only = [s for s in args.only.split(",") if s]

    def chain(visit):
        def run():
            for i in range(U):
                flush_k()
                visit(i)

        return run

    # without a flush kernel the visits rotate through more bytes than the caches hold
    base_fn = chain(lambda i: None) if args.flush_mb > 0 else None
    base = time_graph(base_fn, args.reps) / U if base_fn is not None else 0.0
    print(
        f"T={T} units={U} flush={args.flush_mb} MB n_blocks={nb}: "
        f"flush-only chain {base:.2f} us/unit"
    )
    rows = []
    for name, (unf, fus) in cases.items():
        if only and name not in only:
            continue
        t_u, t_f = (
            t / U - base for t in time_graphs([chain(unf), chain(fus)], args.reps)
        )
        rows.append((name, t_u, t_f))
        print(
            f"{name:18s} unfused {t_u:7.2f} us   fused {t_f:7.2f} us   "
            f"saved {t_u - t_f:6.2f} us/visit"
        )
        if args.trace:
            for tag, fn in (("unfused", chain(unf)), ("fused", chain(fus))):
                ks = kernel_sums(fn)
                for k, (n, d) in sorted(ks.items()):
                    if "add" in k or "elementwise" in k:
                        continue
                    print(f"      {tag:8s} {n:3d}x {d:7.2f} us  {k[:90]}")

    # messages of one loop pass: L per-layer launches vs one batched launch
    if not only or "fill" in only:
        L = 12
        queries = tm.rand_f32((L, 4, H), 300, 0.5)
        msgs = torch.zeros(16, L, 4, H, device=DEV)
        one = torch.zeros(T, 4, H, device=DEV)
        n_prev = 3

        def fill_unf():
            for j in range(L):
                flush_k()
                ref.inkling2_glue_attn_res(
                    blocks, n_prev, hid, queries[j], one, x1=res, eps=tm.EPS
                )

        def fill_fus():
            flush_k()
            ops.msgs_fill(blocks, n_prev, hid, res, queries, msgs[:T], tm.EPS)

        # per pass: 12 flushes in the first, 1 in the second: subtract their own cost
        bf = time_graph(chain(lambda i: None), args.reps) / U
        t_u = time_graph(fill_unf, args.reps) - L * bf
        t_f = time_graph(fill_fus, args.reps) - bf
        print(
            f"{'msgs_fill (12 L)':18s} unfused {t_u:7.2f} us   fused {t_f:7.2f} us   "
            f"saved {t_u - t_f:6.2f} us/pass  ({(t_u - t_f) / L:.2f} us/visit)"
        )

    if rows:
        print("\nper-step estimates (x visits of each kind of a 48-visit step):")
        for name, t_u, t_f in rows:
            print(f"  {name:18s} {t_u - t_f:6.2f} us/visit")


if __name__ == "__main__":
    main()
