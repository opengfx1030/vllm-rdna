#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the QSA decode core of one Flash-Next layer at the hub shapes.

Per layer, graph-replayed over ``--banks`` distinct inputs (cold KV caches), for
``--seqs`` sequences of ``--ctx`` tokens with ``1 + --spec`` rows each:

* ``store``     ``qsa_store_cache_rows`` (the new indexer key rows)
* ``compress``  ``qsa_compress_groups_with_ratio`` (the compressor ring)
* ``select``    ``qsa_select_paged_tokens``: paged MQA scoring of the compressed
                keys (4 indexer heads x 128), top-k 512 blocks, index expansion
* ``sparse``    ``qsa_sparse_paged_attention`` over the 2048 selected tokens

The product dispatch decides HIP versus Triton, so run it twice for the A/B:
``VLLM_RDNA3_HIP_OPS=`` (default route) and ``VLLM_RDNA3_HIP_OPS=qsa``. The dense
projections around it (qkv, o_proj) are in gemv_crossover_bench.py.

``ctx`` is the KV length of every sequence. The scoring sees ``ctx // 4`` live
compressed columns inside a page table ``--capacity`` tokens wide (a captured decode
graph has the maximum model length there, not the live context; the W7800 recipes use
262144); the sparse attention reads ``min(2048, ctx)`` tokens per row and the rest of
the index row is the -1 sentinel. 1k and 2k are the contexts of the TP4 anchors (1k
prompts, 512 generated tokens).

    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/qsa_decode_bench.py \
        [--geometry flashnext-tp1] [--seqs 1,8,16] [--ctx 1024,2048,8192] [--spec 2]
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "kernels" / "rocm" / "qwen4_exp"))

# (q heads, kv heads, head_dim, QSA layers) per rank
GEOMETRIES = {
    "flashnext-tp1": (24, 2, 256, 12),
    "flashnext-tp2": (12, 1, 256, 12),
}
INDEXER_HEADS, INDEXER_DIM, RATIO, TOKEN_TOPK = 4, 128, 4, 2048
MAIN_PAGE, INDEXER_PAGE = 1024, 256


def graph_us(calls, replays=20, warm_s=0.3, repeats=3):
    """(best us per call, spread) of one graph over ``calls``.

    One capture; ``warm_s`` seconds of replays first so the clocks have ramped, then
    ``repeats`` batches of ``replays`` replays: the best batch is the figure, and
    spread is the worst batch over the best minus 1.
    """
    for call in calls[:2]:
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.accelerator.synchronize()
    deadline = time.perf_counter() + warm_s
    while time.perf_counter() < deadline:
        graph.replay()
        torch.accelerator.synchronize()
    batches = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        batches.append(start.elapsed_time(end) * 1000 / (replays * len(calls)))
    return min(batches), max(batches) / min(batches) - 1


def select_call(qsa_ops, seqs, q_len, ctx, capacity, seed):
    """Paged MQA scoring + top-k + expansion over compressed indexer keys."""
    rows = seqs * q_len
    width = -(-capacity // RATIO // INDEXER_PAGE)  # page-table width in pages
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        rows, INDEXER_HEADS, INDEXER_DIM, device="cuda", generator=gen
    ).half()
    k_cache = torch.randn(
        seqs * width, INDEXER_PAGE, 1, INDEXER_DIM, device="cuda", generator=gen
    ).half()
    table = torch.arange(seqs * width, dtype=torch.int32, device="cuda")
    table = table.reshape(seqs, width)
    row = torch.arange(rows, device="cuda")
    token_to_req = (row // q_len).to(torch.int32)
    positions = (ctx - q_len + row % q_len).to(torch.int64)
    lengths = torch.full((seqs,), ctx, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, TOKEN_TOPK + RATIO - 1, dtype=torch.int32, device="cuda")
    return functools.partial(
        qsa_ops.qsa_select_paged_tokens,
        q,
        k_cache,
        table,
        token_to_req,
        positions,
        lengths,
        TOKEN_TOPK,
        RATIO,
        out,
    )


def sparse_call(qsa_ops, helper, geometry, seqs, q_len, ctx, seed):
    """Sparse paged attention over min(2048, ctx) tokens of each row's context."""
    heads, kv_heads, head_dim, _ = geometry
    rows = seqs * q_len
    pages_per_seq = max(1, ctx // MAIN_PAGE)
    q, k_cache, v_cache, logical, table, _ = helper._sparse_inputs(
        rows,
        heads,
        kv_heads,
        head_dim,
        TOKEN_TOPK,
        MAIN_PAGE,
        seqs * pages_per_seq,
        seqs,
        seed=seed,
    )
    # Every row is live (the builder masks row 0), sequences own q_len rows, and the
    # index row holds min(2048, ctx) tokens followed by the -1 sentinel: the whole
    # context when it is shorter than the selection, a random draw otherwise.
    token_to_req = (torch.arange(rows, device="cuda") // q_len).to(torch.int32)
    valid = min(TOKEN_TOPK, ctx)
    logical.fill_(-1)
    if ctx <= TOKEN_TOPK:
        logical[:, :valid] = torch.arange(valid, dtype=torch.int32, device="cuda")
    else:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        logical[:, :valid] = torch.randint(
            0, ctx, (rows, valid), dtype=torch.int32, device="cuda", generator=gen
        )
    return functools.partial(
        qsa_ops.qsa_sparse_paged_attention,
        q,
        k_cache,
        v_cache,
        logical,
        table,
        token_to_req,
    )


def store_call(qsa_ops, rows):
    """The indexer key rows of the step scattered into the paged cache."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    values = torch.randn(rows, INDEXER_DIM, device="cuda", generator=gen).half()
    blocks = 64
    cache = torch.zeros(
        blocks, INDEXER_PAGE, 1, INDEXER_DIM, dtype=torch.float16, device="cuda"
    )
    slots = torch.randperm(blocks * INDEXER_PAGE, device="cuda", generator=gen)
    return functools.partial(
        qsa_ops.qsa_store_cache_rows, cache, slots[:rows].to(torch.int64), values
    )


def compress_call(qsa_ops, helper, rows, seqs):
    a = helper._compress_inputs(rows, INDEXER_DIM, RATIO, seqs, 64, True)
    return functools.partial(
        qsa_ops.qsa_compress_groups_with_ratio,
        a[0],
        a[1],
        a[2],
        a[4],
        a[5],
        a[6],
        a[7],
        a[8],
        RATIO,
        a[3],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="flashnext-tp1")
    ap.add_argument("--seqs", default="1,8,16")
    ap.add_argument("--ctx", default="1024,2048,8192,32768")
    ap.add_argument("--spec", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=0, help="page-table tokens")
    ap.add_argument("--banks", type=int, default=3)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--ops", default="store,compress,select,sparse")
    args = ap.parse_args()
    import test_qsa_rdna2 as helper  # the parity test's input builders

    from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
    from vllm.models.qwen4_exp.amd.ops import qsa_rdna2

    geometry = GEOMETRIES[args.geometry]
    layers = geometry[3]
    ops = args.ops.split(",")
    q_len = 1 + args.spec
    print(
        f"{args.geometry}: {geometry[0]} q / {geometry[1]} kv heads x {geometry[2]}, "
        f"indexer {INDEXER_HEADS} x {INDEXER_DIM}, ratio {RATIO}, top-k {TOKEN_TOPK}; "
        f"HIP qsa route: {qsa_rdna2.qsa_use_rdna2()}; ms/step = {layers} layers"
    )
    print(
        f"{'seqs':>5s}{'rows':>5s}{'kv_len':>8s}{'live':>7s}{'cap':>8s}  {'op':9s}"
        f"{'us/call':>9s}{'ms/step':>9s}{'spread':>8s}"
    )

    def calls_for(name, seqs, ctx):
        rows = seqs * q_len
        if name == "store":
            return [store_call(qsa_ops, rows)] * args.banks
        if name == "compress":
            return [compress_call(qsa_ops, helper, rows, seqs)] * args.banks
        capacity = max(args.capacity, ctx)
        if name == "select":
            return [
                select_call(qsa_ops, seqs, q_len, ctx, capacity, 7 + b)
                for b in range(args.banks)
            ]
        return [
            sparse_call(qsa_ops, helper, geometry, seqs, q_len, ctx, 7 + b)
            for b in range(args.banks)
        ]

    contexts = [int(v) for v in args.ctx.split(",")]
    for seqs in [int(v) for v in args.seqs.split(",")]:
        for name in ops:
            # the store and the compressor do not depend on the context length
            for ctx in [0] if name in ("store", "compress") else contexts:
                calls = calls_for(name, seqs, ctx)
                us, spread = graph_us(calls, args.replays)
                if ctx:
                    live, cap = ctx // RATIO, max(args.capacity, ctx) // RATIO
                    where = f"{ctx:8d}{live:7d}{cap:8d}"
                else:
                    where = f"{'-':>8s}{'-':>7s}{'-':>8s}"
                print(
                    f"{seqs:5d}{seqs * q_len:5d}{where}  {name:9s}{us:9.1f}"
                    f"{us * layers / 1000:9.3f}{spread * 100:7.1f}%"
                )
                del calls
                torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
