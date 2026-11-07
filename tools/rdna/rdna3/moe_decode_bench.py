#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the Flash-Next router and W4A16 routed experts at decode and prefill batches.

Both are graph-replayed over several distinct weight banks, so every call reads
cold weights (one bank would sit in the Infinity Cache) and no host launch cost is
in the number:

* ``router``: the gate GEMM ([M, 2560] x [512, 2560]^T) and ``topk_softmax``;
* ``experts``: the shipping ``Rdna3WNA16Experts.apply`` (moe_align_block_size, GEMM 1,
  silu*mul, GEMM 2 with the fused top-k reduction) in the default CAS mode, in Q32 mode
  (``VLLM_RDNA3_MOE_DETERMINISTIC=1``) and in ``wmma`` mode (``VLLM_RDNA3_MOE_WMMA=1``:
  the S1b WMMA kernel from 256 tokens up, the scalar kernel below).

Each bank holds ``--slots`` resident experts of random int4 weights (2560 -> 1280 ->
640 -> 2560, group 128 by default). With ``--global-experts`` equal to ``--slots``
every routed pair is resident (the cost per distinct expert); with a larger count the
router draws over that many experts and the resident ones are a random subset, the
rest is dropped through the expert map as on the hub.

    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/moe_decode_bench.py \
        [--m 1,3,8,24,48] [--slots 192] [--global-experts 192] [--banks 8] \
        [--block-m 0] [--modes default,q32,wmma]

``--block-m`` lists ``VLLM_RDNA3_MOE_BLOCK_M`` values (0: the automatic choice, 1 for
up to 4 tokens and 4 above), or ``VLLM_RDNA3_MOE_WMMA_BLOCK_M`` (0, 16, 32, 64) in
``wmma`` mode. The TFLOPS column counts the resident pairs only (9.83 MFLOP each).
"""

import argparse
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "kernels" / "quantization"))

HIDDEN, INTER, TOP_K, LAYERS = 2560, 640, 10, 48


def graph_us(calls, replays=20):
    """Graph-replay time per call over ``calls`` (one capture, ``replays`` replays)."""
    for call in calls[:3]:
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * len(calls))


def make_bank(slots, group, process_weights):
    """Random int4 weights of ``slots`` experts, processed into the kernel layout."""

    def packed(n, k):
        return torch.randint(
            -(2**31), 2**31 - 1, (slots, n, k // 8), dtype=torch.int32, device="cuda"
        )

    def scales(n, k):
        s = torch.rand(slots, n, k // group, dtype=torch.float16, device="cuda")
        return s * 0.1 + 0.01

    w13, w2 = packed(2 * INTER, HIDDEN), packed(HIDDEN, INTER)
    s13, s2 = scales(2 * INTER, HIDDEN), scales(HIDDEN, INTER)
    w13, w2, s13, s2, z13, z2, *_ = process_weights(w13, w2, s13, s2, group)
    return SimpleNamespace(w13=w13, w2=w2, w13_s=s13, w2_s=s2, w13_z=z13, w2_z=z2)


def bytes_per_expert(bank, slots):
    tensors = (bank.w13, bank.w2, bank.w13_s, bank.w2_s, bank.w13_z, bank.w2_z)
    return sum(t.numel() * t.element_size() for t in tensors) / slots


def router_us(m, banks, replays):
    """Gate GEMM + topk_softmax, one gate weight per bank (cold)."""
    from vllm import _custom_ops as ops

    gates = [
        (torch.randn(512, HIDDEN, device="cuda") * 0.02).half() for _ in range(banks)
    ]
    x = torch.randn(m, HIDDEN, device="cuda").half()
    weights = torch.empty(m, TOP_K, dtype=torch.float32, device="cuda")
    ids = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
    token_expert = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")

    def run(gate, upcast):
        logits = F.linear(x, gate)
        logits = logits.float() if upcast else logits
        ops.topk_softmax(weights, ids, token_expert, logits, True)

    upcast = False
    try:
        run(gates[0], upcast)
        torch.accelerator.synchronize()
    except RuntimeError as err:  # an op build without fp16 gating logits
        print(f"  router: fp16 logits rejected by topk_softmax ({err}); fp32")
        upcast = True
    return graph_us([functools.partial(run, g, upcast) for g in gates], replays)


def experts_us(args, helper, banks, mode, block_m, m, slot_map):
    """(us per apply, mean distinct resident experts, mean resident pairs)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    glob = args.global_experts
    wmma = mode == "wmma"
    patch = pytest.MonkeyPatch()
    try:
        patch.setenv("VLLM_RDNA3_MOE_WMMA", "1" if wmma else "0")
        patch.setenv("VLLM_RDNA3_MOE_WMMA_BLOCK_M", str(block_m if wmma else 0))
        experts = [
            helper._hub_experts(
                patch,
                bank,
                glob,
                q32=mode == "q32",
                block_m=None if wmma else block_m or None,
            )
            for bank in banks
        ]
    finally:
        patch.undo()
    shapes = experts[0].workspace_shapes(
        m, INTER, HIDDEN, TOP_K, glob, args.slots, None, MoEActivation.SILU
    )
    output = torch.empty(shapes[2], dtype=torch.float16, device="cuda")
    ws13 = torch.empty(shapes[0], dtype=torch.float16, device="cuda")
    ws2 = torch.empty(shapes[1], dtype=torch.float16, device="cuda")
    # The router draws over the global experts; an all-resident bank takes no map.
    routing_map = slot_map
    if routing_map is None:
        routing_map = helper._slot_map(
            args.slots, torch.arange(args.slots, device="cuda")
        )
    calls, distinct, pairs = [], [], []
    for i, (exp, bank) in enumerate(zip(experts, banks)):
        x, ids, topk_w = helper._routing(m, glob, routing_map, seed=1000 * m + i)
        resident = ids.long()[routing_map[ids.long()] >= 0]
        distinct.append(resident.unique().numel())
        pairs.append(resident.numel())
        calls.append(
            functools.partial(
                exp.apply,
                output,
                x,
                bank.w13,
                bank.w2,
                topk_w,
                ids,
                MoEActivation.SILU,
                glob,
                slot_map,
                None,
                None,
                ws13,
                ws2,
                None,
                False,
            )
        )
    us = graph_us(calls, args.replays)
    return us, sum(distinct) / len(distinct), sum(pairs) / len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,3,8,24,48")
    ap.add_argument("--slots", type=int, default=192)
    ap.add_argument("--global-experts", type=int, default=None)
    ap.add_argument("--banks", type=int, default=8)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--block-m", default="0")
    ap.add_argument("--modes", default="default,q32")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--skip-router", action="store_true")
    args = ap.parse_args()
    if args.global_experts is None:
        args.global_experts = args.slots
    if args.global_experts < args.slots:
        ap.error("--global-experts must be at least --slots")
    import vllm._custom_ops  # noqa: F401

    ms = [int(v) for v in args.m.split(",")]
    print(
        f"experts: {args.slots} resident of {args.global_experts}, top-{TOP_K}, "
        f"group {args.group}, {args.banks} cold banks; ms/step = {LAYERS} layers"
    )
    if not args.skip_router:
        print("router (gate GEMM + topk_softmax):")
        for m in ms:
            us = router_us(m, args.banks, args.replays)
            print(f"  M={m:<3d} {us:7.1f} us/call  ({us * LAYERS / 1000:5.2f} ms/step)")

    import test_rdna3_moe_w4a16 as helper  # the GPU test module's builders

    if not helper.on_gfx1100() or not hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
        sys.exit("needs a gfx1100 build with moe_gptq_gemm_rdna3")
    helper.HUB_SLOTS = args.slots
    helper.HUB_GROUP = args.group
    torch.manual_seed(7)
    banks = [
        make_bank(args.slots, args.group, helper._process_weights_rdna3)
        for _ in range(args.banks)
    ]
    per_expert = bytes_per_expert(banks[0], args.slots)
    print(f"weights: {per_expert / 1e6:.3f} MB per expert")
    slot_map = (
        None
        if args.global_experts == args.slots
        else helper._slot_map(args.global_experts)
    )
    print(
        f"{'mode':8s}{'block_m':>8s}{'M':>5s}{'distinct':>10s}{'us/call':>10s}"
        f"{'us/expert':>11s}{'GB/s':>8s}{'TFLOPS':>8s}{'ms/step':>9s}"
    )
    for mode in args.modes.split(","):
        for block_m in [int(v) for v in args.block_m.split(",")]:
            for m in ms:
                us, distinct, pairs = experts_us(
                    args, helper, banks, mode, block_m, m, slot_map
                )
                gbs = distinct * per_expert / us / 1e3
                tflops = pairs * 6 * HIDDEN * INTER / us / 1e6
                print(
                    f"{mode:8s}{block_m:8d}{m:5d}{distinct:10.1f}{us:10.1f}"
                    f"{us / distinct:11.2f}{gbs:8.1f}{tflops:8.2f}"
                    f"{us * LAYERS / 1000:9.2f}"
                )


if __name__ == "__main__":
    main()
