#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-layer replay of the RDNA2 routed MoE: run-to-run identity and cost.

Flash-Next geometry per TP4/EP4 rank: hidden 2560, intermediate 640, 128 of
512 experts local, top-10, group 128, fp16. Synthetic int4 weights, random
routing with distinct experts per token. For each epilogue mode the same
inputs run REPEATS times; every run is compared bitwise with the first. Then
the median time per call (CUDA events, 3 x 20 calls).

  HIP_VISIBLE_DEVICES=2 PYTHONPATH=~/vllm-det-d638 python repro_moe.py
"""

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    compressed_tensors_moe_wna16_rdna2 as wiring,
)

sys.path.insert(0, os.path.expanduser("~/vllm-det-d638/tests/kernels/quantization"))
import test_rdna2_moe_deterministic as t  # noqa: E402

REPEATS = int(os.environ.get("REPEATS", "20"))


def _packed_weights_per_expert(E, K, N):
    # Same packing as the test helper, one expert at a time (128 experts).
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device="cuda")
    for e in range(E):
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device="cuda")
        we = torch.zeros(K // 8, N, dtype=torch.int32, device="cuda")
        for i in range(8):
            we |= (q[i::8, :] & 0xF) << (i * 4)
        ops.gptq_shuffle(we, 4)
        packed[e] = we
    return packed


t._packed_weights = _packed_weights_per_expert
H, INTER, TOPK, LOCAL, GLOBAL, GROUP = 2560, 640, 10, 128, 512, 128


def run_mode(mode, M, w4a8, problem):
    layer, x, ids, tw, emap = problem
    wiring._DETERMINISTIC = mode == "det"
    ops._RDNA2_MOE_FP32_ACCUM = mode == "fp32"
    os.environ["VLLM_RDNA_W4A8_KBLOCK"] = "-1"
    wiring._W4A8_BLOCK_M = 8

    def call():
        return wiring._rdna2_fused_moe(
            x,
            tw,
            ids,
            layer=layer,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=GLOBAL,
            expert_map=emap,
            w4a8=w4a8,
        )

    first = call().clone()
    runs_diff = 0
    elems = []
    max_ulp = 0
    for _ in range(REPEATS - 1):
        y = call()
        d = y != first
        if bool(d.any()):
            runs_diff += 1
            elems.append(int(d.sum()))
            ia = first[d].view(torch.int16).to(torch.int32)
            ib = y[d].view(torch.int16).to(torch.int32)
            max_ulp = max(max_ulp, int((ia - ib).abs().max()))
    times = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.accelerator.synchronize()
        start.record()
        for _ in range(20):
            call()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end) / 20)
    med = statistics.median(times)
    # GPU-only time: one call captured in a CUDA graph, replayed 100 times
    # (decode runs as a full CUDA graph in production).
    graph_ms = float("nan")
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        gtimes = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.accelerator.synchronize()
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            torch.accelerator.synchronize()
            gtimes.append(start.elapsed_time(end) / 100)
        graph_ms = statistics.median(gtimes)
        del graph
    except Exception as exc:  # report, keep going
        print(f"    graph capture failed: {type(exc).__name__}: {exc}", flush=True)
    frac = (statistics.mean(elems) / first.numel()) if elems else 0.0
    print(
        f"M={M:5d} {'W4A8 ' if w4a8 else 'W4A16'} {mode:4s}: "
        f"runs differing from run 0: "
        f"{runs_diff}/{REPEATS - 1}  mean differing elems {frac * 100:.3f}%  "
        f"max ulp {max_ulp}  eager {med:.3f} ms/call  graph {graph_ms:.3f} ms/call",
        flush=True,
    )
    return med, graph_ms


def main():
    print(f"device {torch.cuda.get_device_name(0)}  repeats {REPEATS}")
    for M, w4a8 in ((2048, True), (512, True), (24, False), (3, False), (1, False)):
        problem = t._problem(M, H, INTER, TOPK, LOCAL, GLOBAL, GROUP, seed=17)
        base = None
        for mode in ("cas", "fp32", "det"):
            ms = run_mode(mode, M, w4a8, problem)
            if mode == "cas":
                base = ms
            else:
                print(
                    f"    {mode} vs cas: eager {(ms[0] / base[0] - 1) * 100:+.1f} %"
                    f"  graph {(ms[1] / base[1] - 1) * 100:+.1f} %"
                    f" ({(ms[1] - base[1]) * 1000:+.1f} us/call)"
                )


if __name__ == "__main__":
    main()
