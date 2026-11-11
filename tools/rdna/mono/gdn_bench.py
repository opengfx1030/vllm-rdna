#!/usr/bin/env python3
"""E1 standalone: mono_k2_rdna2_e1a (K1 + GDN decode in one persistent launch)
against today's op sequence for a Qwen4Exp linear-attention layer on one TP
rank: hc_combine_norm -> rdna_hc_mix -> in_proj_qkvz / in_proj_ba GEMVs ->
causal_conv1d_update -> gdn_decode_rdna2 -> gated_rms_norm -> out_proj.
Compares the output, the HC outputs and the updated conv / ssm states, then
times both paths in a graph of 16 distinct layers (cold Infinity Cache).

  PYTHONPATH=<tree> python3 gdn_bench.py [--widths 1,3,6,12,24]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402
import m1_bench as B  # noqa: E402

GH, GHV, GK, GV = 4, 12, 128, 128
CD = 2 * GH * GK + GHV * GV  # 2560 conv channels
NQ = CD + GHV * GV           # 4096
NB = 2 * GHV                 # 24
VD = GHV * GV                # 1536
SLOTS = 64


def make_gdn(gen, dev):
    def rn(*s, std):
        return (torch.randn(*s, generator=gen, device=dev) * std).half()
    L = B.make_layer(gen, dev, keep_golden=False)[0]  # HC weights (attention side)
    L.update(
        wq=rn(NQ, B.H, std=0.02), wba=rn(NB, B.H, std=0.02), convw=rn(CD, 4, std=0.3),
        convb=rn(CD, std=0.1), gnw=(1.0 + rn(GV, std=0.1).float()).half(),
        wo=rn(B.H, VD, std=0.02),
        alog=(torch.rand(GHV, generator=gen, device=dev) * 2.0 - 1.0).float(),
        dtb=(torch.randn(GHV, generator=gen, device=dev) * 0.5).float(),
    )
    return L


def states(gen, dev, layout):
    conv = (torch.randn(SLOTS, CD, 3, generator=gen, device=dev) * 0.5).half()
    if layout == "SD":  # (slot, 3, dim) storage, (slot, dim, 3) view
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    ssm = (torch.randn(SLOTS, GHV, GV, GK, generator=gen, device=dev) * 0.1).float()
    return conv, ssm


def reference(L, res, blk, inj, conv, ssm, sidx, eps_gn, sig):
    from vllm import _custom_ops  # noqa: F401
    from vllm.model_executor.layers import rdna_ops  # noqa: F401
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    M = res.shape[0]
    res_out, xn = hc_combine_norm(res, blk, inj, L["norm_w"], B.EPS, B.HC)
    b, dai = torch.ops.vllm.rdna_hc_mix(xn, L["wd"], None, None, L["wu"], None, None, B.R, B.HC)
    inj_out = dai[:, B.R:B.R + B.HC]
    qkvz = rocm_unquantized_gemm_impl(b, L["wq"], None)
    ba = rocm_unquantized_gemm_impl(b, L["wba"], None)
    mixed = qkvz[:, :CD]
    z = qkvz[:, CD:].reshape(M, GHV, GV)
    bb, aa = ba.chunk(2, dim=-1)
    mixed = causal_conv1d_update(mixed, conv, L["convw"], L["convb"], "silu",
                                 conv_state_indices=sidx, validate_data=False)
    out = torch.empty(M, 1, GHV, GV, dtype=torch.half, device=res.device)
    torch.ops._rocm_C.gdn_decode_rdna2(mixed.contiguous(), aa.contiguous(), bb.contiguous(),
                                       L["alog"], L["dtb"], out, ssm, sidx, GK ** -0.5, True)
    y = torch.empty(M * GHV, GV, dtype=torch.half, device=res.device)
    torch.ops._rocm_C.gated_rms_norm(y, out.reshape(-1, GV), z.reshape(-1, GV).contiguous(),
                                     L["gnw"], eps_gn, sig)
    o = rocm_unquantized_gemm_impl(y.view(M, VD), L["wo"], None)
    return o, res_out, inj_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,6,12,24")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--layout", default="DS,SD")
    a = ap.parse_args()
    ext = B.build_ext()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(21)
    nb = int(ext.workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    eps_gn, sig = 1e-6, 0
    ok = True
    L0 = make_gdn(gen, dev)

    def mono(L, res, blk, inj, conv, ssm, sidx, out, ro, io):
        ext.mono_gdn(res, blk, inj, L["norm_w"], L["wd"], L["wu"], L["wq"], L["wba"],
                     L["convw"], L["convb"], conv, ssm, L["alog"], L["dtb"], L["gnw"],
                     L["wo"], sidx, ro, io, out, ws, 0, B.EPS, eps_gn, sig, GK ** -0.5, 2000.0)

    for layout in a.layout.split(","):
        for M in [int(x) for x in a.widths.split(",")]:
            gt = torch.Generator(device=dev).manual_seed(100 + M)
            res = torch.randn(M, B.HC * B.H, generator=gt, device=dev).half()
            blk = (torch.randn(M, B.H, generator=gt, device=dev) * 0.5).half()
            inj = torch.randn(M, B.HC, generator=gt, device=dev).half()
            sidx = (torch.randperm(SLOTS - 1, generator=gt, device=dev)[:M] + 1).int()
            if M >= 3:
                sidx[1] = 0  # a null / padded row
            conv, ssm = states(gt, dev, layout)
            conv_r, ssm_r = conv.clone(), ssm.clone()
            o_r, ro_r, io_r = reference(L0, res, blk, inj, conv_r, ssm_r, sidx, eps_gn, sig)
            out = torch.empty(M, B.H, dtype=torch.half, device=dev)
            ro = torch.empty_like(res)
            io = torch.empty(M, B.HC, dtype=torch.half, device=dev)
            mono(L0, res, blk, inj, conv, ssm, sidx, out, ro, io)
            torch.cuda.synchronize()
            valid = sidx > 0
            d_o = (out[valid].float() - o_r[valid].float()).abs().max().item()
            rel = d_o / max(o_r[valid].float().abs().max().item(), 1e-6)
            d_ro = (ro.float() - ro_r.float()).abs().max().item()
            d_io = (io.float() - io_r.float()).abs().max().item()
            # slot 0 is the null block: today's conv update writes it for padded
            # rows, the kernel leaves it alone; its content is never read
            d_cs = (conv[1:].float() - conv_r[1:].float()).abs().max().item()
            d_ss = (ssm - ssm_r).abs().max().item()
            if d_cs >= 2e-3 and os.environ.get("GDN_DEBUG"):
                dif = (conv.float() - conv_r.float()).abs()
                bad_slots = (dif.amax(dim=(1, 2)) > 2e-3).nonzero().flatten().tolist()
                print("   conv_state differs in slots", bad_slots, "rows' slots", sidx.tolist())
                for sl in bad_slots[:2]:
                    ch = (dif[sl].amax(dim=1) > 2e-3).nonzero().flatten()
                    print("   slot", sl, "channels", ch.min().item(), "..", ch.max().item(),
                          "count", ch.numel())
            good = rel < 2e-2 and d_ro == 0 and d_io < 5e-3 and d_cs < 2e-3 and d_ss < 1e-3
            ok &= good
            ts = ext.phase_ts(ws)[:8].tolist()
            ph = "/".join(f"{(ts[i] - ts[i - 1]) / 100:.1f}" for i in range(1, 8))
            print(f"   phases P1..P7 us {ph}  (P4 in_proj, P5 conv+gdn+norm, P6 out_proj)")
            print(f"{layout} M={M:2d} out max {d_o:.2e} (rel {rel:.1e}) res_out {d_ro:.1e} "
                  f"inj {d_io:.1e} conv_state {d_cs:.1e} ssm_state {d_ss:.1e} "
                  f"{'ok' if good else 'BAD'}")
    print("sticky", ext.error(ws).tolist()[:3])

    # timing: 16 distinct layers (weights + states), graph-captured
    layers = [make_gdn(gen, dev) for _ in range(a.layers)]
    for M in [int(x) for x in a.widths.split(",")]:
        gt = torch.Generator(device=dev).manual_seed(200 + M)
        ins = [((torch.randn(M, B.HC * B.H, generator=gt, device=dev)).half(),
                (torch.randn(M, B.H, generator=gt, device=dev) * 0.5).half(),
                torch.randn(M, B.HC, generator=gt, device=dev).half()) for _ in layers]
        sts = [states(gt, dev, "DS") for _ in layers]
        sidx = (torch.randperm(SLOTS - 1, generator=gt, device=dev)[:M] + 1).int()
        outs = [torch.empty(M, B.H, dtype=torch.half, device=dev) for _ in layers]
        ros = [torch.empty(M, B.HC * B.H, dtype=torch.half, device=dev) for _ in layers]
        ios = [torch.empty(M, B.HC, dtype=torch.half, device=dev) for _ in layers]
        st = ws[nb : nb + 26 * 36 * 8].view(torch.int64)
        st.zero_()
        st[25 * 36] = -1
        st[25 * 36 + 3] = 1  # kernel phase stats over the timed replays
        t_m = A.graph_time(lambda: [mono(layers[i], *ins[i], *sts[i], sidx, outs[i], ros[i],
                                          ios[i]) for i in range(len(layers))],
                           a.iters) / len(layers)
        r = st.view(26, 36)[M].tolist()
        st[25 * 36 + 3] = 0
        if r[0]:
            ph = "/".join(f"{r[i] / r[0] / 100:.1f}" for i in range(1, 8))
            print(f"   graph phases P1..P7 us {ph} | kernel {r[9] / r[0] / 100:.1f} us | "
                  f"WG entry spread {r[8] / r[0] / 100:.1f} us")
        t_r = A.graph_time(lambda: [reference(layers[i], *ins[i], *sts[i], sidx, eps_gn, sig)
                                    for i in range(len(layers))], a.iters) / len(layers)
        print(f"M={M:2d} today {t_r:6.1f} us  mono {t_m:6.1f} us  x{t_r / t_m:.3f}")
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
