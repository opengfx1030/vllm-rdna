#!/usr/bin/env python3
"""K1 (mono_k1_rdna2, the attention-side HC preamble on the persistent kernel)
against today's op sequence (hc_combine_norm -> rdna_hc_mix): outputs and
per-layer time in a graph of 16 distinct layers (cold Infinity Cache).

  PYTHONPATH=<tree> python3 k1_bench.py [--widths 1,3,8,24]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402
import m1_bench as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,8,24")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers import rdna_ops  # noqa: F401  (registers rdna_hc_mix)
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    ext = B.build_ext()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    layers = [B.make_layer(gen, dev, keep_golden=False)[0] for _ in range(a.layers)]
    nb = int(ext.workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    ok = True
    for M in [int(x) for x in a.widths.split(",")]:
        gt = torch.Generator(device=dev).manual_seed(50 + M)
        ins = [((torch.randn(M, B.HC * B.H, generator=gt, device=dev)).half(),
                (torch.randn(M, B.H, generator=gt, device=dev) * 0.5).half(),
                torch.randn(M, B.HC, generator=gt, device=dev).half()) for _ in layers]
        outs = [torch.empty(M, B.H, dtype=torch.half, device=dev) for _ in layers]
        ros = [torch.empty(M, B.HC * B.H, dtype=torch.half, device=dev) for _ in layers]
        ios = [torch.empty(M, B.HC, dtype=torch.half, device=dev) for _ in layers]

        def k1(i):
            L = layers[i]
            r, b, j = ins[i]
            ext.mono_hc(r, b, j, L["norm_w"], L["wd"], L["wu"], ros[i], ios[i], outs[i], ws,
                        0, B.EPS, 2000.0)

        def today(i):
            L = layers[i]
            r, b, j = ins[i]
            res_out, xn = hc_combine_norm(r, b, j, L["norm_w"], B.EPS, B.HC)
            blk, dai = torch.ops.vllm.rdna_hc_mix(xn, L["wd"], None, None, L["wu"], None,
                                                  None, B.R, B.HC)
            return res_out, blk, dai[:, B.R:B.R + B.HC]

        # correctness on layer 0
        k1(0)
        ro, bl, io = today(0)
        torch.cuda.synchronize()
        d_out = (outs[0].float() - bl.float()).abs().max().item()
        d_res = (ros[0].float() - ro.float()).abs().max().item()
        d_inj = (ios[0].float() - io.float()).abs().max().item()
        good = d_out < 5e-3 and d_res == 0 and d_inj < 5e-3
        ok &= good
        t_k1 = A.graph_time(lambda: [k1(i) for i in range(len(layers))], a.iters) / len(layers)
        t_td = A.graph_time(lambda: [today(i) for i in range(len(layers))], a.iters) / len(layers)
        print(f"M={M:2d} today {t_td:6.1f} us  K1 {t_k1:6.1f} us  x{t_td / t_k1:.3f} | "
              f"max|out| {d_out:.1e} |res_out| {d_res:.1e} |inj| {d_inj:.1e} "
              f"{'ok' if good else 'BAD'}")
    print("sticky", ext.error(ws).tolist()[:3])
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
