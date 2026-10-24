# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the FA-RDNA2 prefill kernels (gfx1030).

Times every prefill kernel that supports the head size (gqa, short, varlen,
splitk) on fresh prompts, chunks behind a prefix and small batches, and
prints the max output difference of each kernel against the gqa kernel.

Usage:
    python benchmarks/kernels/benchmark_fa_rdna2_prefill.py
    python benchmarks/kernels/benchmark_fa_rdna2_prefill.py \
        --head-size 256 --heads 24/4 12/4 --block-size 784
    python benchmarks/kernels/benchmark_fa_rdna2_prefill.py \
        --dflash --heads 8/2 --block-size 1024 --block-major-layers 5
"""

import argparse

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.rdna_attn import _reinterpret_v_to_5d
from vllm.v1.attention.ops import fa_rdna2_backend as fa
from vllm.v1.attention.ops.paged_attn import PagedAttention

# (name, [(query_len, kv_len), ...]) per batch.
CASES = [
    ("prompt 512", [(512, 512)]),
    ("prompt 1k", [(1024, 1024)]),
    ("prompt 2k", [(2048, 2048)]),
    ("prompt 4k", [(4096, 4096)]),
    ("prompt 8k", [(8192, 8192)]),
    ("4 x prompt 1k", [(1024, 1024)] * 4),
    ("chunk 1k @ 8k", [(1024, 8192)]),
    ("chunk 2k @ 16k", [(2048, 16384)]),
]


def make_batch(seqs, H_q, H_kv, D, block_size, dtype, block_major_layers):
    blocks = [(kv + block_size - 1) // block_size for _, kv in seqs]
    nb = sum(blocks)
    if block_major_layers > 1:
        kv_cache = torch.randn(
            nb,
            block_major_layers,
            2,
            H_kv,
            D,
            block_size,
            dtype=dtype,
            device="cuda",
        )[:, 0].permute(1, 0, 2, 3, 4)
    else:
        kv_cache = torch.randn(2, nb, H_kv, D, block_size, dtype=dtype, device="cuda")
    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, H_kv, D)
    value_cache = _reinterpret_v_to_5d(key_cache, value_cache, D)
    perm = torch.randperm(nb, dtype=torch.int32, device="cuda")
    block_table = torch.zeros(len(seqs), max(blocks), dtype=torch.int32, device="cuda")
    start = 0
    for i, n in enumerate(blocks):
        block_table[i, :n] = perm[start : start + n]
        start += n
    cu = [0]
    for q_len, _ in seqs:
        cu.append(cu[-1] + q_len)
    q = torch.randn(cu[-1], H_q, D, dtype=dtype, device="cuda")
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([kv for _, kv in seqs], dtype=torch.int32, device="cuda")
    return q, key_cache, value_cache, block_table, cu_t, seq_lens


def kernels(D, max_kv):
    kv_splits = max(2, min(8, (max_kv + 1023) // 1024))
    ks = {"gqa": fa.fa_rdna2_prefill_paged_varlen_gqa}
    if D == 128:
        ks["short"] = fa.fa_rdna2_prefill_paged_varlen_short
    ks["varlen"] = fa.fa_rdna2_prefill_paged_varlen

    def splitk(*a, **kw):
        return fa.fa_rdna2_prefill_paged_varlen_splitk(*a, kv_splits=kv_splits, **kw)

    ks["splitk"] = splitk
    return ks


def bench_case(seqs, H_q, H_kv, args):
    D = args.head_size
    dtype = torch.bfloat16 if args.dflash else torch.float16
    q, kc, vc, bt, cu, seq_lens = make_batch(
        seqs, H_q, H_kv, D, args.block_size, dtype, args.block_major_layers
    )
    times, outs = {}, {}
    selected = (
        {"gqa": fa.fa_rdna2_prefill_paged_varlen_gqa}
        if args.dflash
        else kernels(D, max(kv for _, kv in seqs))
    )
    for name, kernel in selected.items():
        out = torch.empty_like(q)

        def run(kernel=kernel, out=out):
            extra = {"causal": False, "sliding_window": 2048} if args.dflash else {}
            if args.dflash and args.kv_splits > 1:
                extra["kv_splits"] = args.kv_splits
            kernel(q, kc, vc, bt, cu, seq_lens, args.block_size, out=out, **extra)

        run()
        torch.accelerator.synchronize()
        bench = (
            triton.testing.do_bench
            if args.no_graph
            else triton.testing.do_bench_cudagraph
        )
        times[name] = bench(run, return_mode="median") * 1000
        outs[name] = out
    diffs = {
        name: (out.float() - outs["gqa"].float()).abs().max().item()
        for name, out in outs.items()
        if name != "gqa"
    }
    return times, diffs


def main(args):
    torch.manual_seed(12345)
    cases = (
        [
            (f"c{c} q{q} kv{kv}", [(q, kv)] * c)
            for c in (1, 8)
            for q in (3, 4)
            for kv in (1024, 2048, 16384)
        ]
        if args.dflash
        else CASES
    )
    for heads in args.heads:
        H_q, H_kv = (int(h) for h in heads.split("/"))
        print(
            f"\nD={args.head_size} heads {heads} bs={args.block_size} "
            f"splits={args.kv_splits} (us)"
        )
        for label, seqs in cases:
            times, diffs = bench_case(seqs, H_q, H_kv, args)
            cols = "  ".join(f"{k}={v:9.1f}" for k, v in times.items())
            comparison = (
                f"  max|diff| vs gqa={max(diffs.values()):.1e}" if diffs else ""
            )
            print(f"{label:<16} {cols}{comparison}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--heads", nargs="+", default=["32/8", "28/4", "16/4"])
    parser.add_argument("--head-size", type=int, default=128, choices=[128, 256])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dflash", action="store_true", help="BF16 noncausal GQA window2048"
    )
    parser.add_argument("--block-major-layers", type=int, default=1)
    parser.add_argument("--kv-splits", type=int, default=1, choices=[1, 4, 8, 16, 32])
    parser.add_argument(
        "--no-graph", action="store_true", help="time without CUDA graphs"
    )
    main(parser.parse_args())
