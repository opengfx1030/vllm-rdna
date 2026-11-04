# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time ``top_k_per_row_decode`` at the QSA block-selection shapes (gfx1030).

The Flash-Next QSA indexer picks k = 512 compressed blocks per query row with
``ops.top_k_per_row_decode(logits, 1, visible_blocks, ...)`` (block = 4
tokens, so 16384 columns is a 64K-token context). Rows run from decode with
MTP 2 (1, 3, 24) to a 2048-token prefill chunk. Below 200,000 columns every
row is one block's radix select.

Score distributions:
  relu4   sum of four ReLU'd Gaussians / 4, like QSA's sum of relu(q_h . k):
          about 6 % exact zeros, the cut among distinct positive scores
  zeros   300 positive scores per row over exact zeros: the cut falls in the
          zeros, more of them than the 2048-slot final sort holds
  coarse  scores on a 1/16 grid: ties straddle the cut in the sorted bin

Each sample replays a CUDA graph of --inner back-to-back calls between two
events; the median over --samples is reported in microseconds per call.
Compare builds by running this from each tree, interleaved.

VLLM_RDNA2_MOE_DETERMINISTIC=1 times the kernels that keep the lowest columns
among tied logits at the cut. The launcher reads it once per process, so time
each mode in its own process; the records carry it as "knob".

  python benchmarks/kernels/benchmark_qsa_top_k_rdna2.py --tag base --json a.jsonl
"""

import argparse
import functools
import json
import statistics

import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
from vllm import envs

TOP_K = 512
DISTRIBUTIONS = ("relu4", "zeros", "coarse")


def make_logits(rows: int, cols: int, dist: str, gen: torch.Generator):
    if dist == "relu4":
        x = torch.randn(4, rows, cols, device="cuda", generator=gen)
        return x.relu_().sum(0) / 4
    if dist == "zeros":
        x = torch.zeros(rows, cols, device="cuda")
        noise = torch.rand(rows, cols, device="cuda", generator=gen)
        cols_pos = noise.argsort(dim=1)[:, :300]
        values = 1.0 + torch.rand(rows, 300, device="cuda", generator=gen)
        return x.scatter_(1, cols_pos, values)
    grid = torch.randint(0, 64, (rows, cols), device="cuda", generator=gen)
    return grid.float() / 16


def run(logits, seq_lens, out):
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        out,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        TOP_K,
    )


def check(logits, out):
    picked = logits.gather(1, out.long()).sort(dim=1, descending=True).values
    torch.testing.assert_close(picked, logits.topk(TOP_K, dim=1).values, atol=0, rtol=0)


def time_us(fn, inner: int, samples: int) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3, 24, 256, 2048])
    parser.add_argument(
        "--cols",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384],
        help="Visible compressed blocks per row (the logits width)",
    )
    parser.add_argument(
        "--dists", nargs="+", choices=DISTRIBUTIONS, default=list(DISTRIBUTIONS)
    )
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pad-mb",
        type=int,
        default=0,
        help="Allocate this many MiB first, which moves every later buffer: a "
        "control for run differences that come from where the logits land",
    )
    parser.add_argument("--tag", default="", help="Label copied into each record")
    parser.add_argument("--json", help="Append one JSON record per case here")
    args = parser.parse_args()
    _pad = torch.empty(args.pad_mb << 20, dtype=torch.uint8, device="cuda")  # kept

    for dist in args.dists:
        for rows in args.rows:
            for cols in args.cols:
                gen = torch.Generator(device="cuda").manual_seed(args.seed)
                logits = make_logits(rows, cols, dist, gen)
                seq_lens = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
                out = torch.empty(rows, TOP_K, dtype=torch.int32, device="cuda")
                run(logits, seq_lens, out)
                torch.accelerator.synchronize()
                check(logits, out)
                us = time_us(
                    functools.partial(run, logits, seq_lens, out),
                    args.inner,
                    args.samples,
                )
                record = dict(
                    tag=args.tag,
                    knob=int(envs.VLLM_RDNA2_MOE_DETERMINISTIC),
                    pad_mb=args.pad_mb,
                    dist=dist,
                    rows=rows,
                    cols=cols,
                    us=us,
                )
                print(json.dumps(record), flush=True)
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
