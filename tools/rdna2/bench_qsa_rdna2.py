#!/usr/bin/env python
"""QSA HIP kernels vs the Triton reference, at the production call sites.

Reports two numbers per op:

  call_us   wall time per call, CUDA events around a batched loop. This is what
            a non-cudagraph decode step pays, and on gfx1030 it is dominated by
            Python + Triton launch overhead (~20 us) rather than by the kernel.
  kernel_us  summed GPU kernel time from torch.profiler. This is what a
            cudagraph-captured decode step pays, and is the number to compare
            kernels with.

The arms differ only in VLLM_RDNA_QSA_HIP, and inputs come from the parity
test's own builders, so the bench runs shapes the tests already verify.

    HIP_VISIBLE_DEVICES=1 .venv/bin/python tools/rdna2/bench_qsa_rdna2.py
"""

import importlib
import os
import sys

import torch

sys.path.insert(0, "tests/kernels/rocm/qwen4_exp")

import test_qsa_rdna2 as T  # noqa: E402

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.amd.ops import qsa_rdna2  # noqa: E402

DEV = T.device
WARMUP, ITERS = 10, 50


def call_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS * 1000.0


def kernel_us(fn):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in prof.key_averages()) / ITERS


def set_gate(gate: bool) -> None:
    os.environ["VLLM_RDNA_QSA_HIP"] = "1" if gate else "0"
    reloaded = importlib.reload(qsa_rdna2)
    qsa_ops.qsa_rdna2 = reloaded


def bench_store(rows=512, width=128, page_size=256, num_blocks=64):
    """Tracker S5 production shape: (rows=512, WIDTH=128, PAGE_SIZE=256)."""
    torch.manual_seed(1)
    rowsv = torch.randn(rows, width, dtype=torch.float16, device=DEV)
    cap = num_blocks * page_size
    slots = torch.randperm(cap, device=DEV)[:rows].to(torch.int64)
    cache = torch.zeros(num_blocks, page_size, 1, width, dtype=torch.float16, device=DEV)
    return lambda: qsa_ops.qsa_store_cache_rows(cache, slots, rowsv)


def bench_compress(rows=512, head_dim=128, ratio=4, n_req=64, num_blocks=64):
    """Tracker S5 production shape: (rows=512, head_dim=128, ratio=4)."""
    a = T._compress_inputs(rows, head_dim, ratio, n_req, num_blocks, True)
    return lambda: qsa_ops.qsa_compress_groups_with_ratio(
        a[0], a[1], a[2], a[4], a[5], a[6], a[7], a[8], ratio, a[3]
    )


def bench_mqa(rows=512, heads=4, head_dim=128, num_pages=64, page_size=256,
              table_width=32, num_req=2, ratio=4):
    """Model geometry: indexer_n_heads=4, indexer_head_dim=128, ratio=4."""
    q, cache, table, req, pos, seq = T._mqa_inputs(
        rows, heads, head_dim, num_pages, page_size, table_width, num_req, ratio
    )
    return lambda: qsa_ops.qsa_mqa_paged(
        q, cache, table, req, pos, seq, ratio, table_width * page_size
    )


def bench_expand(rows=512, block_topk=512, ratio=4, num_requests=8):
    """Model geometry: indexer_budget 2048 / compress_ratio 4 -> block_topk 512."""
    a = T._expand_inputs(rows, block_topk, ratio, num_requests, seed=rows)
    return lambda: qsa_ops.expand_qsa_block_indices_cuda(
        a[0], a[1], a[2], a[3], ratio, block_topk * ratio
    )


# Prefill chunks of the served batch (--max-num-batched-tokens 2048) plus the
# MTP-verify sizes the decode step pays for.
ROWS = (3, 6, 12, 24, 64, 256, 512, 1024, 2048)


def main():
    print(f"{'op':<44}{'call_us':>18}{'kernel_us':>20}")
    for name, mk in (
        ("store_cache_rows w=128", bench_store),
        ("compress_groups d=128 r=4", bench_compress),
        ("mqa_paged cols=8192", bench_mqa),
        ("expand_indices block_topk=512", bench_expand),
    ):
        for rows in ROWS:
            row = {}
            for gate in (False, True):
                set_gate(gate)
                fn = mk(rows=rows)
                fn()
                torch.cuda.synchronize()
                row[gate] = (call_us(fn), kernel_us(fn))
            (c0, k0), (c1, k1) = row[False], row[True]
            print(
                f"{name + f' rows={rows}':<44}"
                f"{c0:8.2f} -> {c1:7.2f}  {100 * (c0 - c1) / c0:+5.1f}%"
                f"{k0:8.2f} -> {k1:7.2f}  {100 * (k0 - k1) / k0:+5.1f}%"
            )


if __name__ == "__main__":
    main()
