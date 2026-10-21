#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 (mono_k1_rdna2, the attention-side HC preamble on the persistent kernel)
against today's op sequence (hc_combine_norm -> rdna_hc_mix): outputs and
per-layer time in a graph of 16 distinct layers (a weight working set larger
than the Infinity Cache).

  PYTHONPATH=<tree> python k1_bench.py [--widths 1,3,8,24]

For an installed-kernel M32 split probe, use --installed --chunk-rows 16
--widths 32 --rows <rank-0 lookup-only TunableOp CSV>. This does not build an
extension or change production dispatch. The two M16 calls share one coarse-
grained workspace and run on the current stream inside the same HIP graph.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.cuda.tunable as tunable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402
import m1_bench as B  # noqa: E402


def make_hc_layer(gen: torch.Generator, dev: torch.device) -> dict[str, torch.Tensor]:
    """Only the three K1 weights; 16 layers total about 205 MiB."""
    return {
        "norm_w": (torch.randn(B.HC * B.H, generator=gen, device=dev) * 0.1).half(),
        "wd": (
            torch.randn(B.ND, B.HC * B.H, generator=gen, device=dev)
            * (B.HC * B.H) ** -0.5
        ).half(),
        "wu": (
            torch.randn(B.HC * B.H, B.R, generator=gen, device=dev) * (B.R**-0.5 * 2)
        ).half(),
    }


def installed_ops(rows: Path):
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") != "1":
        raise RuntimeError("Set PYTORCH_TUNABLEOP_ENABLED=1 for the serve baseline")
    if os.environ.get("PYTORCH_TUNABLEOP_TUNING") != "0":
        raise RuntimeError("Set PYTORCH_TUNABLEOP_TUNING=0; this is lookup-only")
    if not os.environ.get("PYTORCH_TUNABLEOP_FILENAME"):
        raise RuntimeError("Set PYTORCH_TUNABLEOP_FILENAME to tuned rank-0 rows")
    rows = rows.resolve(strict=True)
    tunable.set_filename(str(rows), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)
    rows_hash = hashlib.sha256(rows.read_bytes()).hexdigest()
    print(f"# TunableOp rows={rows} sha256={rows_hash}")
    ops = torch.ops._rocm_C
    required = (
        "mono_k1_rdna2",
        "mono_k2_rdna2_workspace_bytes",
        "mono_k2_rdna2_workspace_alloc",
        "mono_k2_rdna2_error",
    )
    missing = [name for name in required if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(f"Installed _rocm_C lacks K1 ops: {missing}")
    return SimpleNamespace(
        mono_hc=ops.mono_k1_rdna2,
        workspace_bytes=ops.mono_k2_rdna2_workspace_bytes,
        workspace_alloc=ops.mono_k2_rdna2_workspace_alloc,
        error=ops.mono_k2_rdna2_error,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,8,24")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--installed", action="store_true", help="Use installed _rocm_C K1")
    ap.add_argument("--chunk-rows", type=int, default=0, choices=(0, 16))
    ap.add_argument("--rows", type=Path, help="Lookup-only rank-0 TunableOp CSV")
    a = ap.parse_args()
    widths = [int(x) for x in a.widths.split(",")]
    if a.layers < 1 or a.iters < 1 or any(m < 1 or m > 32 for m in widths):
        ap.error("layers/iters and widths must be positive; maximum width is 32")
    if any(m > 24 for m in widths) and a.chunk_rows != 16:
        ap.error("widths above 24 require --chunk-rows 16")
    if a.chunk_rows and not a.installed:
        ap.error("--chunk-rows requires --installed")
    if a.installed and a.rows is None:
        ap.error("--installed requires --rows with rank-0 lookup-only results")
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers import (
        rdna_ops,  # noqa: F401  (registers rdna_hc_mix)
    )
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    ext = installed_ops(a.rows) if a.installed else B.build_ext()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    layers = [
        make_hc_layer(gen, dev) if a.installed else B.make_layer(gen, dev)[0]
        for _ in range(a.layers)
    ]
    if a.installed:
        weight_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layer in layers
            for tensor in layer.values()
        )
        print(
            f"# distinct K1 weights={weight_bytes} B ({weight_bytes / 2**20:.1f} MiB)"
        )
    nb = int(ext.workspace_bytes(B.H, B.R, B.ND, B.E, B.ISE, B.I))
    ws = (
        ext.workspace_alloc(nb + 8192, torch.accelerator.current_device_index())
        if a.installed
        else torch.zeros(nb + 8192, dtype=torch.uint8, device=dev)
    )
    ok = True
    for M in widths:
        gt = torch.Generator(device=dev).manual_seed(50 + M)
        ins = [
            (
                (torch.randn(M, B.HC * B.H, generator=gt, device=dev)).half(),
                (torch.randn(M, B.H, generator=gt, device=dev) * 0.5).half(),
                torch.randn(M, B.HC, generator=gt, device=dev).half(),
            )
            for _ in layers
        ]
        outs = [torch.empty(M, B.H, dtype=torch.half, device=dev) for _ in layers]
        ros = [torch.empty(M, B.HC * B.H, dtype=torch.half, device=dev) for _ in layers]
        ios = [torch.empty(M, B.HC, dtype=torch.half, device=dev) for _ in layers]

        def k1(i, ins=ins, ros=ros, ios=ios, outs=outs, m=M):
            L = layers[i]
            r, b, j = ins[i]
            chunk = a.chunk_rows if m > 24 else m
            for lo in range(0, m, chunk):
                hi = min(lo + chunk, m)
                ext.mono_hc(
                    r[lo:hi],
                    b[lo:hi],
                    j[lo:hi],
                    L["norm_w"],
                    L["wd"],
                    L["wu"],
                    ros[i][lo:hi],
                    ios[i][lo:hi],
                    outs[i][lo:hi],
                    ws,
                    0,
                    B.EPS,
                    2000.0,
                )

        def today(i, ins=ins):
            L = layers[i]
            r, b, j = ins[i]
            res_out, xn = hc_combine_norm(r, b, j, L["norm_w"], B.EPS, B.HC)
            blk, dai = torch.ops.vllm.rdna_hc_mix(
                xn, L["wd"], None, None, L["wu"], None, None, B.R, B.HC
            )
            return res_out, blk, dai[:, B.R : B.R + B.HC]

        # correctness on layer 0
        k1(0)
        ro, bl, io = today(0)
        torch.accelerator.synchronize()
        d_out = (outs[0].float() - bl.float()).abs().max().item()
        d_res = (ros[0].float() - ro.float()).abs().max().item()
        d_inj = (ios[0].float() - io.float()).abs().max().item()
        good = d_out < 5e-3 and d_res == 0 and d_inj < 5e-3
        ok &= good
        if a.installed:
            # Capture the whole layer sequence, then replay with changed data
            # at the same addresses. This checks ordered reuse of the workspace.
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for i in range(len(layers)):
                    k1(i)
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(len(layers)):
                    k1(i)
            for i, (r, b, j) in enumerate(ins):
                r.copy_(torch.randn(r.shape, generator=gt, device=dev).half())
                b.copy_((torch.randn(b.shape, generator=gt, device=dev) * 0.5).half())
                j.copy_(torch.randn(j.shape, generator=gt, device=dev).half())
            graph.replay()
            torch.accelerator.synchronize()
            replay_good = True
            for i in range(len(layers)):
                ref_res, ref_out, ref_inj = today(i)
                replay_good &= bool(torch.equal(ros[i], ref_res))
                replay_good &= bool(
                    (outs[i].float() - ref_out.float()).abs().max() < 5e-3
                )
                replay_good &= bool(
                    (ios[i].float() - ref_inj.float()).abs().max() < 5e-3
                )
            ok &= replay_good
            print(
                f"M={M:2d} changed-input graph replay {'ok' if replay_good else 'BAD'}"
            )
        t_k1 = A.graph_time(lambda: [k1(i) for i in range(len(layers))], a.iters) / len(
            layers
        )
        t_td = A.graph_time(
            lambda: [today(i) for i in range(len(layers))], a.iters
        ) / len(layers)
        print(
            f"M={M:2d} today {t_td:6.1f} us  K1 {t_k1:6.1f} us  x{t_td / t_k1:.3f} | "
            f"max|out| {d_out:.1e} |res_out| {d_res:.1e} |inj| {d_inj:.1e} "
            f"{'ok' if good else 'BAD'}"
        )
    sticky = ext.error(ws).tolist()[:3]
    ok &= sticky[0] == 0
    print("sticky", sticky)
    print("OK" if ok else "BAD")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
