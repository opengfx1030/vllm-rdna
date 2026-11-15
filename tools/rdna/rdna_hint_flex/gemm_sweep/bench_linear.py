# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The whole prefill linear (activation transform + dequant + GEMM [+ GEMV tail]) at the
HI2 27B TP2 shapes, rocBLAS against the dense HIP GEMM, interleaved on cold weights.

``torch.ops._rocm_C.rdna_hint_flex_linear_rdna2`` on synthetic HI2 tensors (random
2/4-bit rows, ``--int4-frac`` 0.5 = 3.0 bits per weight, the checkpoint's average):

  rocblas_heur    prefill_gemm 0, TunableOp off (what serving runs without rows)
  rocblas_tuned   prefill_gemm 0, TunableOp on with ``--rows`` (when the cell has a row)
  hip             prefill_gemm 1: the dense kernel on all M rows
  hip_tail        prefill_gemm 2: the dense kernel on the 128-multiple rows, the decode
                  GEMV on the M % 128 <= 48 remaining rows

    HIP_VISIBLE_DEVICES=5 python bench_linear.py --shapes all --m 1024,1031,1045,2048 \\
        --rows ../../../../tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2/tunableop_results0.csv \\
        --csv lin.csv --label lin

The timing, clock and CSV code is sweep.py's. ``vs_rb`` = better rocBLAS row / this row.
"""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sweep = importlib.import_module("sweep")
bp = importlib.import_module("bench_prod")
bk = importlib.import_module("bench_kernel")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--m", default="1024,1031,1045,2048")
    ap.add_argument(
        "--modes", default="1,2", help="prefill_gemm values to time next to rocBLAS"
    )
    ap.add_argument("--int4-frac", type=float, default=0.5)
    ap.add_argument("--rows", default="")
    ap.add_argument("--tune-missing", action="store_true")
    ap.add_argument("--tune-ms", type=float, default=300)
    ap.add_argument("--tune-iters", type=int, default=30)
    ap.add_argument("--rot-mb", type=int, default=640)
    ap.add_argument(
        "--ws-mb", type=float, default=640, help="packed weight bytes rotated through"
    )
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--win-ms", type=float, default=60)
    ap.add_argument("--warm-s", type=float, default=2.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--label", default="lin")
    args = ap.parse_args()

    import torch
    import torch.cuda.tunable as tunable

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu

    print("device:", sweep.device_check(torch), flush=True)
    merged = (
        Path(os.environ.get("SWEEP_OUT", str(sweep.OUT))) / "tunableop_lin_rows.csv"
    )
    paths = [Path(p) for p in args.rows.split(",") if p]
    nrows = bp.merge_rows(paths, merged) if paths else 0
    if not paths:
        merged.parent.mkdir(parents=True, exist_ok=True)
        merged.write_text("")
    tunable.set_filename(str(merged), insert_device_ordinal=False)
    if nrows:
        tunable.read_file(str(merged))
    tunable.enable(False)

    modes = [int(v) for v in args.modes.split(",") if v]
    wr = sweep.Writer(args.csv)
    shapes = {
        k: s
        for k, s in sweep.SHAPES.items()
        if args.shapes == "all" or k in args.shapes.split(",")
    }
    ms = [int(m) for m in args.m.split(",")]
    for name, (r, k) in shapes.items():
        t = bk.make(r, k, args.int4_frac)
        hu.validate(t)
        base = {f: v.cuda() for f, v in t.items()}
        nbytes = sum(v.numel() * v.element_size() for v in base.values())
        nsets = max(2, int(args.ws_mb * 2**20 // nbytes) + 1)
        sets = [base] + [
            {f: v.clone() for f, v in base.items()} for _ in range(nsets - 1)
        ]
        print(
            f"{name}: {nbytes / 2**20:.1f} MB packed ({nbytes * 8 / (r * k):.2f} bits/weight), {nsets} sets",
            flush=True,
        )
        for m in ms:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            flop = 2.0 * m * r * k
            common = dict(shape=name, N=r, K=k, M=m)

            def call(d, mode, x=x):
                return ops.rdna_hint_flex_linear_rdna2(
                    x, d["col_scale"], d["had"], d["codes"], d["code_off"],
                    d["blk_params"], d["param_off"], d["row_consts"],
                    d["row_nbits"], d["row_block"], True, 48, 0, False, mode,
                )  # fmt: skip

            tunable.enable(False)
            ref = call(base, 0).float()

            def vs_ref(out, ref=ref):
                return float((out.float() - ref).norm() / ref.norm())

            cons = []
            heur = sweep.Contender(
                "rocblas_heur", "default",
                [lambda d=d: call(d, 0) for d in sets], flop, tun=False,
            )  # fmt: skip
            heur.err = (None, None, 0.0)
            cons.append(heur)
            sol = bp.has_row(tunable, r, m, k)
            if sol is None and args.tune_missing:
                xt = torch.randn(m, k, dtype=torch.float16, device="cuda")
                wd = torch.randn(r, k, dtype=torch.float16, device="cuda")
                sweep.tune_rocblas(
                    torch, xt, wd, tunable, args.tune_ms, args.tune_iters, args.rot_mb
                )
                sweep.save_rows(tunable, merged)
                sol = bp.has_row(tunable, r, m, k)
                del xt, wd
            if sol is not None:
                tuned = sweep.Contender(
                    "rocblas_tuned", "tunableop",
                    [lambda d=d: call(d, 0) for d in sets], flop, tun=True, note=sol,
                )  # fmt: skip
                tuned.err = (None, None, 0.0)
                cons.append(tuned)
            for mode in modes:
                out = call(base, mode)
                torch.accelerator.synchronize()
                con = sweep.Contender(
                    "hip" if mode == 1 else "hip_tail", f"prefill_gemm={mode}",
                    [lambda d=d, mode=mode: call(d, mode) for d in sets], flop,
                    tun=False, note=f"mode {mode}",
                )  # fmt: skip
                con.err = (None, None, vs_ref(out))
                cons.append(con)
            clk = sweep.measure(
                torch, tunable, cons, args.rounds, args.warm_s, args.win_ms
            )
            best_us = min(
                statistics.median(c.times)
                for c in cons
                if c.kind in ("rocblas_heur", "rocblas_tuned")
            )
            heur_us = statistics.median(heur.times)
            for con in cons:
                med = statistics.median(con.times)
                row = dict(
                    phase=args.label, kind=con.kind, variant=con.variant, us=round(med, 1),
                    us_min=round(min(con.times), 1),
                    tflops=round(con.flop / med / 1e6, 2),
                    vs_rb=round(best_us / med, 3),
                    gfx_mhz=con.clk[0] or clk[0], power_w=con.clk[1] or clk[1], **common,
                )  # fmt: skip
                row["rel_fp32"] = f"{con.err[2]:.2e}" if con.err else None
                row.update(con.extra)
                row["note"] = f"{con.extra.get('note', '')} vs_heur {heur_us / med:.3f}"
                wr.add(**row)
            x = ref = cons = None
            torch.cuda.memory.empty_cache()
        sets = base = None
        torch.cuda.memory.empty_cache()


if __name__ == "__main__":
    main()
