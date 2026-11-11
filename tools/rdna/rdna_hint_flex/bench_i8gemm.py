# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I8 prefill GEMM (activation scale per 64 columns) against dense fp16
F.linear (rocBLAS, TunableOp lookup from --rows) on Qwen3.5-4B projection
shapes. Times the pre-pass + GEMM and the GEMM alone; TF-equivalent counts
2*M*R*K. --keys writes the dense shapes as a TunableOp shape file to tune.

  python bench_i8gemm.py --rows rows.csv [--tile 0|1]
"""

import argparse
import functools

import torch
import torch.nn.functional as F
from bench_kernel import SHAPES, make, timeit

from vllm import _custom_ops as ops

MS = (16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None, help="TunableOp rows for dense")
    ap.add_argument("--keys", default=None, help="write dense shape keys and exit")
    ap.add_argument("--m", default=",".join(map(str, MS)))
    ap.add_argument("--tiles", default="0,1")
    args = ap.parse_args()
    ms = [int(v) for v in args.m.split(",")]
    if args.keys:
        with open(args.keys, "w") as f:
            for r, k in SHAPES.values():
                for m in ms:
                    f.write(f"GemmTunableOp_Half_TN,tn_{r}_{m}_{k}_ld_{k}_{k}_{r}\n")
        return
    if args.rows:
        import torch.cuda.tunable as tunable

        tunable.set_filename(args.rows, insert_device_ordinal=False)
        tunable.enable(True)
        tunable.tuning_enable(False)
    tiles = [int(t) for t in args.tiles.split(",")]
    for name, (r, k) in SHAPES.items():
        d = {f: v.cuda() for f, v in make(r, k).items()}
        w = torch.randn(r, k, dtype=torch.float16, device="cuda")
        for m in ms:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            xq = torch.empty(m, k, dtype=torch.int8, device="cuda")
            sx = torch.empty(m, k // 64, dtype=torch.float32, device="cuda")
            sq = torch.empty(m, k // 16, dtype=torch.float32, device="cuda")
            out = torch.empty(m, r, dtype=torch.float16, device="cuda")
            pre = functools.partial(
                ops.rdna_hint_flex_act_quant64_rdna2,
                x, d["col_scale"], d["had"], True, xq, sx, sq,
            )  # fmt: skip
            it = 20 if m <= 512 else 5
            t_d = timeit(functools.partial(F.linear, x, w), iters=it, reps=3)
            t_p = timeit(pre, iters=it, reps=3)
            tf = 2 * m * r * k / 1e6
            row = (
                f"{name:13s} M={m:5d} | dense {t_d:8.1f} us {tf / t_d:5.1f} TF"
                f" | prepass {t_p:6.1f}"
            )
            for tile in tiles:
                g = functools.partial(
                    ops.rdna_hint_flex_i8gemm_rdna2, xq, sx, sq, d, out, tile
                )
                t_g = timeit(g, iters=it, reps=3)
                tot = t_g + t_p
                row += (
                    f" | t{tile} gemm {t_g:8.1f} ({tf / t_g:5.1f} TF)"
                    f" +pre {tot:8.1f} x{t_d / tot:4.2f}"
                )
            print(row, flush=True)


if __name__ == "__main__":
    main()
