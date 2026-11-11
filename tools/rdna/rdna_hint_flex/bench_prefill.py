# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill (M > 48) timing of rdna_hint_flex_linear_rdna2 against a dense fp16
F.linear (rocBLAS) of the same shape. The prefill variant comes from
RDNA_HINT_FLEX_PREFILL (0 dequant + GEMM, 1/2/3 fused decode-in-GEMM tiles);
run one process per variant. TunableOp follows the environment (lookup rows
via tools/rdna2_028/tunableop_env.sh, or off).

  RDNA_HINT_FLEX_PREFILL=3 python tools/rdna/rdna_hint_flex/bench_prefill.py
"""

import argparse
import functools
import os

import torch
import torch.nn.functional as F
from bench_kernel import SHAPES, make, timeit

from vllm import _custom_ops as ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="64,128,512,2048")
    ap.add_argument("--mode", type=int, default=0, help="0 F16, 2 I8 for every M")
    args = ap.parse_args()
    variant = os.environ.get("RDNA_HINT_FLEX_PREFILL", "0")
    for name, (r, k) in SHAPES.items():
        d = {f: v.cuda() for f, v in make(r, k).items()}
        w = torch.randn(r, k, dtype=torch.float16, device="cuda")
        for m in (int(v) for v in args.m.split(",")):
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            lin = (
                x, d["col_scale"], d["had"], d["codes"], d["code_off"],
                d["blk_params"], d["param_off"], d["row_consts"],
                d["row_nbits"], d["row_block"], True, 48,
            )  # fmt: skip
            f = functools.partial(ops.rdna_hint_flex_linear_rdna2, *lin, args.mode)
            t_q = timeit(f, iters=10, reps=3)
            t_d = timeit(functools.partial(F.linear, x, w), iters=10, reps=3)
            tf = 2 * m * r * k / 1e6
            print(
                f"v{variant}m{args.mode} {name:13s} R={r:6d} K={k:5d} M={m:5d} | ours "
                f"{t_q:9.1f} us ({tf / t_q:5.1f} TF/s) | dense {t_d:9.1f} us "
                f"({tf / t_d:5.1f} TF/s) | ratio {t_q / t_d:5.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
