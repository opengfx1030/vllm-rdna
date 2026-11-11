#!/usr/bin/env python3
"""QSA indexer stores: the three per-token cache stores (compressed keys,
raw keys, int64 RoPE positions) four ways, each timed in a CUDA graph.

  A  3x Triton _store_qsa_rows (no --monokernel, HIP QSA off)
  B  mono Triton store_rows3 (one launch)
  C  3x HIP qsa_store_cache_rows_rdna2 (HIP QSA on)
  D  mono HIP mono_qsa_store3_rdna2 (one launch)

B, C and D must leave the caches bit-identical to A (PAD = -1 slots included).

  PYTHONPATH=<tree> python3 qsa_store3_bench.py [--rows 1,3,8,24,512,2048]
"""

import argparse

import torch


def layouts(n, gen, dev, pages=256, page=64, width=128):
    comp = torch.randn(pages, page, 1, width, generator=gen, device=dev).half()
    raw = torch.randn(pages, page, 1, width, generator=gen, device=dev).half()
    # rope positions: int64 x3 in a padded fp16 page, like the serve
    rope_store = torch.zeros(pages, page, 1, 12, device=dev, dtype=torch.float16)
    c_slots = torch.randperm(pages * page, generator=gen, device=dev)[:n].long()
    c_slots[::3] = -1  # most tokens close no group
    r_slots = torch.randperm(pages * page, generator=gen, device=dev)[:n].long()
    crow = torch.randn(n, width, generator=gen, device=dev).half()
    rrow = torch.randn(n, width, generator=gen, device=dev).half()
    prow = torch.randint(0, 1 << 40, (3, n), generator=gen, device=dev).t()
    return [comp, raw, rope_store], c_slots, r_slots, crow, rrow, prow


def graph_us(fn, calls=100, reps=20):
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
    ap.add_argument("--rows", default="1,3,8,24,512,2048")
    a = ap.parse_args()
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as MQ
    from vllm.models.qwen4_exp.amd.ops import qsa as Q
    from vllm.models.qwen4_exp.amd.ops import qsa_rdna2 as QR

    dev = torch.device("cuda")
    has_d = MQ._has_hip_store3()
    print(
        f"{'rows':>6} {'A 3xTriton':>11} {'B monoTriton':>13} {'C 3xHIP':>9} "
        f"{'D monoHIP':>10}   (us/call, graph)"
    )
    ok = True
    for n in [int(x) for x in a.rows.split(",")]:
        gen = torch.Generator(device=dev).manual_seed(n)
        caches, cs, rs, cr, rr, pr = layouts(n, gen, dev)

        args = (cs, rs, cr, rr, pr)

        def views(c):
            return c[0], c[1], c[2].view(torch.int64)[..., :3]

        def run_a(c, cs, rs, cr, rr, pr):
            cc, rc, pc = views(c)
            for cache, slots, rows in ((cc, cs, cr), (rc, rs, rr), (pc, rs, pr)):
                Q.qsa_store_cache_rows(cache, slots, rows)

        def run_b(c, cs, rs, cr, rr, pr):
            cc, rc, pc = views(c)
            orig = MQ._has_hip_store3
            MQ._has_hip_store3 = lambda: False
            try:
                assert MQ.store_rows3(cc, cs, cr, rc, rs, rr, pc, pr)
            finally:
                MQ._has_hip_store3 = orig

        def run_c(c, cs, rs, cr, rr, pr):
            cc, rc, pc = views(c)
            for cache, slots, rows in ((cc, cs, cr), (rc, rs, rr), (pc, rs, pr)):
                QR.qsa_store_cache_rows_compat(cache, slots, rows)

        def run_d(c, cs, rs, cr, rr, pr):
            cc, rc, pc = views(c)
            torch.ops._rocm_C.mono_qsa_store3_rdna2(cc, cs, cr, rc, rs, rr, pc, pr)

        runs = {"A": run_a, "B": run_b, "C": run_c}
        if has_d:
            runs["D"] = run_d
        outs = {}
        for k, fn in runs.items():
            c = [t.clone() for t in caches]
            fn(c, *args)
            torch.cuda.synchronize()
            outs[k] = c
        for k in runs:
            if k == "A":
                continue
            same = all(
                torch.equal(x.view(torch.int16), y.view(torch.int16))
                for x, y in zip(outs["A"], outs[k])
            )
            if not same:
                ok = False
                print(f"  MISMATCH {k} vs A at rows={n}")
        times = {}
        for k, fn in runs.items():
            c = [t.clone() for t in caches]
            times[k] = graph_us(lambda fn=fn, c=c, args=args: fn(c, *args))
        print(
            f"{n:6d} {times['A']:11.2f} {times['B']:13.2f} {times['C']:9.2f} "
            f"{times.get('D', float('nan')):10.2f}"
        )
    print("bitwise vs A:", "ALL OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
