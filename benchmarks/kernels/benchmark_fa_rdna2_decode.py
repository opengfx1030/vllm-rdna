# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the FA-RDNA2 split-K decode kernel (gfx1030).

Times fa_rdna2_decode_paged over batch size x context length for the given
head geometries. Run it on two builds of _rocm_C to compare kernel versions.

Usage:
    python benchmarks/kernels/benchmark_fa_rdna2_decode.py
    python benchmarks/kernels/benchmark_fa_rdna2_decode.py \
        --head-size 128 --heads 32/8 --block-size 16 --layout dense
"""

import argparse

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.rdna_attn import _reinterpret_v_to_5d
from vllm.v1.attention.ops import fa_rdna2_backend as fa
from vllm.v1.attention.ops.paged_attn import PagedAttention


def make_cache(batch, ctx, H_kv, D, block_size, layout):
    per_seq = (ctx + block_size - 1) // block_size
    nb = batch * per_seq
    if layout == "interleaved":
        kv_cache = torch.randn(
            nb, 2, H_kv, D, block_size, dtype=torch.float16, device="cuda"
        ).transpose(0, 1)
    else:
        kv_cache = torch.randn(
            2, nb, H_kv, D, block_size, dtype=torch.float16, device="cuda"
        )
    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, H_kv, D)
    value_cache = _reinterpret_v_to_5d(key_cache, value_cache, D)
    block_table = torch.randperm(nb, dtype=torch.int32, device="cuda").view(
        batch, per_seq
    )
    return key_cache, value_cache, block_table


def bench_case(H_q, H_kv, batch, ctx, args):
    D = args.head_size
    kc, vc, bt = make_cache(batch, ctx, H_kv, D, args.block_size, args.layout)
    q = torch.randn(batch, H_q, D, dtype=torch.float16, device="cuda")
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)

    def run():
        fa.fa_rdna2_decode_paged(
            q,
            kc,
            vc,
            bt,
            seq_lens,
            args.block_size,
            kv_splits=args.kv_splits,
            out=out,
        )

    run()
    torch.accelerator.synchronize()
    bench = (
        triton.testing.do_bench if args.no_graph else triton.testing.do_bench_cudagraph
    )
    return bench(run, return_mode="median") * 1000


def main(args):
    print(f"{'heads':<7} {'batch':>5} {'ctx':>6} {'us':>9}")
    for heads in args.heads:
        H_q, H_kv = (int(h) for h in heads.split("/"))
        for ctx in args.contexts:
            for batch in args.batches:
                us = bench_case(H_q, H_kv, batch, ctx, args)
                print(f"{heads:<7} {batch:>5} {ctx:>6} {us:>9.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--heads", nargs="+", default=["6/1", "12/2", "16/4"])
    parser.add_argument("--head-size", type=int, default=256, choices=[128, 256])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--kv-splits", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=784)
    parser.add_argument(
        "--layout", choices=["dense", "interleaved"], default="interleaved"
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="time without CUDA graphs"
    )
    main(parser.parse_args())
