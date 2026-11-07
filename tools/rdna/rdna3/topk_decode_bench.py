#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G8: time ``top_k_per_row_decode`` with the radix and the insertion select.

``csrc/sampler.cu`` reads ``VLLM_TOPK_DECODE_ALGO`` (``radix``, ``insertion`` or
unset = the arch default) on every host call, so one process times all three by
capturing a CUDA graph per setting. The selected indices of the three runs must
agree row by row (the data is continuous random, ties do not occur).

    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/topk_decode_bench.py \
        [--top-k 2048] [--batch 1,8,16] [--next-n 3] [--cols 1024,4096,8192]

``rows = batch * next_n`` (the MTP-2 verify shape has next_n = 3). The last column
of the table is the faster explicit algorithm; keep the arch default (``auto``)
only when it is that algorithm.
"""

import argparse
import os

import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
from vllm.platforms import current_platform  # noqa: F401  (loads the extensions)

ALGOS = ("auto", "radix", "insertion")


def graph_us(fn, iters: int = 50) -> float:
    """Mean us per call, replayed from a graph (no host launch cost)."""
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iters):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def set_algo(algo: str) -> None:
    if algo == "auto":
        os.environ.pop("VLLM_TOPK_DECODE_ALGO", None)
    else:
        os.environ["VLLM_TOPK_DECODE_ALGO"] = algo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--batch", default="1,8,16")
    parser.add_argument("--next-n", type=int, default=3)
    parser.add_argument("--cols", default="1024,4096,8192")
    args = parser.parse_args()

    print(
        f"device {torch.cuda.get_device_properties(0).gcnArchName}, top_k {args.top_k}"
    )
    print(
        f"{'cols':>6} {'rows':>5} " + " ".join(f"{a:>10}" for a in ALGOS) + "  faster"
    )
    failures = 0
    for cols in [int(v) for v in args.cols.split(",")]:
        for batch in [int(v) for v in args.batch.split(",")]:
            rows = batch * args.next_n
            gen = torch.Generator(device="cuda").manual_seed(cols + batch)
            logits = torch.randn(
                rows, cols, device="cuda", dtype=torch.float32, generator=gen
            )
            seq_lens = torch.full((batch,), cols, device="cuda", dtype=torch.int32)
            indices = torch.full(
                (rows, args.top_k), -1, device="cuda", dtype=torch.int32
            )

            def call(logits=logits, seq_lens=seq_lens, indices=indices, rows=rows):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    args.next_n,
                    seq_lens,
                    indices,
                    rows,
                    logits.stride(0),
                    logits.stride(1),
                    args.top_k,
                )

            times, picked = {}, {}
            for algo in ALGOS:
                set_algo(algo)
                indices.fill_(-1)
                times[algo] = graph_us(call)
                call()
                torch.accelerator.synchronize()
                picked[algo] = indices.sort(dim=1).values.clone()
            set_algo("auto")
            agree = all(torch.equal(picked["auto"], picked[a]) for a in ALGOS[1:])
            failures += 0 if agree else 1
            faster = min(("radix", "insertion"), key=lambda a: times[a])
            print(
                f"{cols:>6} {rows:>5} "
                + " ".join(f"{times[a]:>8.1f}us" for a in ALGOS)
                + f"  {faster}{'' if agree else '  INDICES DIFFER'}"
            )
    if failures:
        raise SystemExit(f"{failures} cell(s) where the algorithms disagree")


if __name__ == "__main__":
    main()
