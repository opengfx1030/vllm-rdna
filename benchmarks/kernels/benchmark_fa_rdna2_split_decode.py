# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark RDNA_ATTN on spec-decode verify and mixed batches (gfx1030).

Times RdnaAttentionImpl.forward with the decode/prefill split off (every row
through a prefill kernel) and on (decode-like rows through the split-K decode
kernel), and prints the max output difference between the two.

Usage:
    python benchmarks/kernels/benchmark_fa_rdna2_split_decode.py
    python benchmarks/kernels/benchmark_fa_rdna2_split_decode.py \
        --heads 12/2 --contexts 16384 --batches 8 --num-spec 3
"""

import argparse

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.rdna_attn import (
    RdnaAttentionImpl,
    RdnaAttentionMetadataBuilder,
)


def make_batch(q_lens, seq_lens, H_kv, D, block_size, layout):
    blocks = [(s + block_size - 1) // block_size for s in seq_lens]
    nb = sum(blocks)
    if layout == "interleaved":
        kv_cache = torch.randn(
            nb, 2, H_kv, D, block_size, dtype=torch.float16, device="cuda"
        ).transpose(0, 1)
    else:
        kv_cache = torch.randn(
            2, nb, H_kv, D, block_size, dtype=torch.float16, device="cuda"
        )
    perm = torch.randperm(nb, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(
        len(seq_lens), max(blocks), dtype=torch.int32, device="cuda"
    )
    start = 0
    for i, n in enumerate(blocks):
        block_table[i, :n] = perm[start : start + n]
        start += n
    cu = [0]
    for n in q_lens:
        cu.append(cu[-1] + n)
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=torch.tensor(cu, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        num_reqs=len(q_lens),
        num_actual_tokens=cu[-1],
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(cu[-1], dtype=torch.int64, device="cuda"),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )
    return kv_cache.transpose(0, 1), cm


def build(cm, reorder_batch_threshold):
    builder = RdnaAttentionMetadataBuilder.__new__(RdnaAttentionMetadataBuilder)
    builder.reorder_batch_threshold = reorder_batch_threshold
    return builder.build(0, cm)


def bench_case(name, q_lens, seq_lens, H_q, H_kv, args):
    D = args.head_size
    kv_cache, cm = make_batch(q_lens, seq_lens, H_kv, D, args.block_size, args.layout)
    impl = RdnaAttentionImpl(H_q, D, D**-0.5, H_kv, None, None, "auto")
    q = torch.randn(cm.num_actual_tokens, H_q, D, dtype=torch.float16, device="cuda")
    outs, times = [], []
    for threshold in (None, 1 + args.num_spec):
        meta = build(cm, threshold)
        out = torch.empty_like(q)

        def run(meta=meta, out=out):
            impl.forward(None, q, None, None, kv_cache, meta, out)

        run()
        torch.accelerator.synchronize()
        bench = (
            triton.testing.do_bench
            if args.no_graph
            else triton.testing.do_bench_cudagraph
        )
        times.append(bench(run, return_mode="median") * 1000)
        outs.append(out)
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    print(
        f"{name:<34} H={H_q}/{H_kv} {times[0]:10.1f} {times[1]:10.1f} "
        f"{times[0] / times[1]:8.2f}x {diff:10.2e}"
    )


def main(args):
    print(
        f"{'case':<34} {'heads':<6} {'split off':>10} {'split on':>10} "
        f"{'speedup':>9} {'max|diff|':>10}   (us)"
    )
    q_verify = 1 + args.num_spec
    for heads in args.heads:
        H_q, H_kv = (int(h) for h in heads.split("/"))
        for ctx in args.contexts:
            for b in args.batches:
                bench_case(
                    f"verify q={q_verify} B={b} ctx={ctx}",
                    [q_verify] * b,
                    [ctx] * b,
                    H_q,
                    H_kv,
                    args,
                )
            # Seven decoding requests next to a fresh 100-token prompt, and
            # next to a 2k chunk behind a prefix (chunked prefill).
            chunk_ctx = max(ctx, 2048)
            bench_case(
                f"mixed 7x1 + prompt 100 ctx={ctx}",
                [1] * 7 + [100],
                [ctx] * 7 + [100],
                H_q,
                H_kv,
                args,
            )
            bench_case(
                f"mixed 7x1 + chunk 2048 ctx={ctx}",
                [1] * 7 + [2048],
                [ctx] * 7 + [chunk_ctx],
                H_q,
                H_kv,
                args,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--heads", nargs="+", default=["6/1", "12/2"])
    parser.add_argument("--head-size", type=int, default=256, choices=[128, 256])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--num-spec", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=784)
    parser.add_argument(
        "--layout", choices=["dense", "interleaved"], default="interleaved"
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="time without CUDA graphs"
    )
    main(parser.parse_args())
