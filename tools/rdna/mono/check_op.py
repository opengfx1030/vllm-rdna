#!/usr/bin/env python3
"""The registered torch.ops._rocm_C.mono_k2_rdna2 of this tree's build against
today's op sequence (same harness as m1_bench.py), widths 1..24, plus a graph
capture/replay and the fail-stop NaN path.

  PYTHONPATH=<tree> python3 check_op.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_bench as B  # noqa: E402


def main():
    import vllm._custom_ops  # noqa: F401
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)
    L, G = B.make_layer(gen, dev, keep_golden=True)
    nb = int(torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    ref = B.Reference(0, dev)
    ok = True

    def run(res, blk, inj):
        M = res.shape[0]
        out = torch.empty(M, B.H, dtype=torch.half, device=dev)
        ro = torch.empty_like(res)
        io = torch.empty(M, B.HC, dtype=torch.half, device=dev)
        torch.ops._rocm_C.mono_k2_rdna2(res, blk, inj, L["norm_w"], L["wd"], L["wu"], L["wr"],
                                          L["se_w1"], L["se_w2"], L["se_wg"], L["w13"], L["s13"],
                                          L["z13"], L["w2"], L["s2"], L["z2"], ro, io, out, ws,
                                          0, 1, 0, B.EPS, 2000.0)
        return out, ro, io

    for M in (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 23, 24):
        res = torch.randn(M, B.HC * B.H, generator=gen, device=dev).half()
        blk = (torch.randn(M, B.H, generator=gen, device=dev) * 0.5).half()
        inj = torch.randn(M, B.HC, generator=gen, device=dev).half()
        o_ref, ro_ref, io_ref, _ = ref(L, res, blk, inj)
        out, ro, io = run(res, blk, inj)
        torch.cuda.synchronize()
        rowd = (out.float() - o_ref.float()).abs().max(1).values
        d = rowd.max().item()
        r = (ro.float() - ro_ref.float()).abs().max().item()
        B.golden(G, res, blk, inj, 0)
        gaps = B.golden.last_gap
        ties = [m for m in range(M) if rowd[m] > 5e-3 and gaps[m] < 1e-3]
        good = r == 0 and all(rowd[m] < 5e-3 or m in ties for m in range(M))
        if ties:
            print(f"   rows {ties} differ at fp32 routing near-ties (top10/11 gaps "
                  f"{[round(gaps[m], 6) for m in ties]})")
        ok &= good
        print(f"M={M:2d} max|out-ref| {d:.2e} max|res_out-ref| {r:.1e} {'ok' if good else 'BAD'}")
    # graph capture + replay with new inputs copied in
    M = 3
    res = torch.randn(M, B.HC * B.H, device=dev).half()
    blk = torch.randn(M, B.H, device=dev).half()
    inj = torch.randn(M, B.HC, device=dev).half()
    run(res, blk, inj)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gout, _, _ = run(res, blk, inj)
    res.copy_(torch.randn_like(res))
    g.replay()
    torch.cuda.synchronize()
    d = (gout.float() - ref(L, res, blk, inj)[0].float()).abs().max().item()
    print(f"graph replay max|out-ref| {d:.2e}")
    ok &= d < 5e-3
    err = torch.ops._rocm_C.mono_k2_rdna2_error(ws).tolist()
    print("sticky record", err[:3], "grid", err[3])
    ok &= err[0] == 0
    # fail-stop: set the sticky code by hand, the next launch must write NaN
    ws[8:12].copy_(torch.tensor([1], dtype=torch.int32).view(torch.uint8))
    out, _, _ = run(res, blk, inj)
    torch.cuda.synchronize()
    nan = bool(torch.isnan(out).all())
    print("fail-stop launch output all NaN:", nan)
    ok &= nan
    # serve timing stats (workspace tail) + marks
    ws.zero_()
    st = ws[nb : nb + 26 * 36 * 8].view(torch.int64)
    st[25 * 36] = -1
    st[25 * 36 + 3] = 1
    for _ in range(3):
        torch.ops._rocm_C.mono_k2_rdna2_mark(st, 0, 3)
        run(res, blk, inj)
        torch.ops._rocm_C.mono_k2_rdna2_mark(st, 1, 3)
    torch.cuda.synchronize()
    r = st.view(26, 36)[3].tolist()
    print("stats w3: launches", r[0], "phases us", [round(v / r[0] / 100, 1) for v in r[1:8]],
          "spread us", round(r[8] / r[0] / 100, 1), "total us", round(r[9] / r[0] / 100, 1),
          "| marks", r[12], "region us", round(r[13] / max(r[12], 1) / 100, 1),
          "mark->first/last us", round(r[14] / r[0] / 100, 1), round(r[15] / r[0] / 100, 1),
          "| sclk over region MHz", round(100.0 * r[30] / max(r[31], 1)))
    ok &= r[0] == 3 and r[12] == 3 and r[13] > r[9] > 0
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
