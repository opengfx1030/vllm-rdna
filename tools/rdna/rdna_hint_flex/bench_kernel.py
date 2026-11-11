# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level timing of the rdna_hint_flex linear paths on Qwen3.5-4B
projection shapes (random tensors, ~30% INT4 rows, mixed blocks), against a
dense fp16 F.linear of the same shape. A kernel microbenchmark, not a serving
benchmark.

  python tools/rdna/rdna_hint_flex/bench_kernel.py [--variants]
"""

import argparse
import functools

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu

SHAPES = {  # Qwen3.5-4B projections (R, K)
    "gate_up": (2 * 9216, 2560),
    "down": (2560, 9216),
    "in_proj_qkvz": (12288, 2560),
    "out_proj": (2560, 4096),
}


def make(r: int, k: int, int4_frac: float = 0.3, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    nb = torch.where(torch.rand(r, generator=g) < int4_frac, 4, 2)
    bl = torch.tensor([16, 32, 64])[torch.randint(3, (r,), generator=g)]
    words = k * nb // 32
    blocks = k // bl
    co = torch.zeros(r + 1, dtype=torch.int64)
    co[1:] = torch.cumsum(words, 0)
    po = torch.zeros(r + 1, dtype=torch.int64)
    po[1:] = torch.cumsum(blocks, 0)
    return {
        "codes": torch.randint(-(2**31), 2**31 - 1, (int(co[-1]),), dtype=torch.int32),
        "code_off": co.to(torch.int32),
        "blk_params": torch.randint(0, 256, (2 * int(po[-1]),), dtype=torch.uint8),
        "param_off": po.to(torch.int32),
        "row_consts": (torch.rand(r, 3) * 1e-2).to(torch.float16),
        "row_nbits": nb.to(torch.uint8),
        "row_block": bl.to(torch.uint8),
        "had": torch.ones(k // 16, dtype=torch.uint8),
        "col_scale": (0.5 + torch.rand(k)).to(torch.float16),
    }


def timeit(fn, iters: int = 50, reps: int = 4) -> float:
    """GPU time per call, from a CUDA graph of ``iters`` calls (no Python
    launch overhead in the measurement)."""
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / (iters * reps) * 1e3  # us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true", help="time GEMV variants")
    ap.add_argument("--m", default="1,2,3,4,8,16,32,64")
    args = ap.parse_args()
    ms = [int(v) for v in args.m.split(",")]
    for name, (r, k) in SHAPES.items():
        t = make(r, k)
        hu.validate(t)
        d = {f: v.cuda() for f, v in t.items()}
        qbytes = sum(v.numel() * v.element_size() for v in d.values())
        w = torch.randn(r, k, dtype=torch.float16, device="cuda")
        for m in ms:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            lin = (
                x, d["col_scale"], d["had"], d["codes"], d["code_off"],
                d["blk_params"], d["param_off"], d["row_consts"],
                d["row_nbits"], d["row_block"], True,
            )  # fmt: skip
            row = f"{name:13s} R={r:6d} K={k:5d} M={m:3d}"
            t_q = timeit(functools.partial(ops.rdna_hint_flex_linear_rdna2, *lin, 48))
            row += f" | f16 {t_q:7.1f} us ({qbytes / t_q / 1e3:5.1f} GB/s)"
            t_8 = timeit(
                functools.partial(ops.rdna_hint_flex_linear_rdna2, *lin, 48, 1)
            )
            row += f" | i8 {t_8:7.1f} us"
            if args.variants:
                xt = torch.empty_like(x)
                t_t = timeit(
                    functools.partial(
                        ops.rdna_hint_flex_act_transform_rdna2,
                        x,
                        d["col_scale"],
                        d["had"],
                        xt,
                    )  # fmt: skip
                )
                row += f" | transform {t_t:6.1f}"
                t_g = timeit(
                    functools.partial(ops.rdna_hint_flex_linear_rdna2, *lin, 0)
                )
                row += f" | dq+gemm {t_g:7.1f}"
            t_d = timeit(functools.partial(F.linear, x, w))
            row += f" | dense fp16 {t_d:7.1f} us"
            print(row, flush=True)


if __name__ == "__main__":
    main()
