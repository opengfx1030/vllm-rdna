# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare sequential W4A16 skinny HIP and tile Triton at MTP decode sizes.

Requires an idle gfx1030 GPU and matching native extension. Uses the Intel
Flash-Next EP4 dimensions by default, not checkpoint weights. Does not change
dispatch, model configuration, or services. Prints JSON results after validation.
"""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
from vllm.platforms.rocm import on_gfx10x


def graph_time(graph, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def dequant(weight, scale, group):
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    return (codes.float() - 8) * scale.float().repeat_interleave(group, dim=-1)


def run(args):
    torch.accelerator.set_device_index(args.device)
    assert on_gfx10x(), "Requires gfx10x"
    torch.manual_seed(20260927)
    device = torch.device("cuda", args.device)
    e, k, n, topk, group = 128, 2560, 640, 10, 128
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device=device, dtype=torch.uint8)
    s1 = torch.full((e, 2 * n, k // group), 0.02, device=device, dtype=torch.float16)
    s2 = torch.full((e, k, n // group), 0.02, device=device, dtype=torch.float16)
    expert_map = torch.full((e * 4,), -1, device=device, dtype=torch.int32)
    expert_map[::4] = torch.arange(e, device=device, dtype=torch.int32)
    for m in args.rows:
        for routing in ("ep4", "all-local"):
            x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
            ids = torch.randint(0, e * 4, (m, topk), device=device, dtype=torch.int64)
            if routing == "all-local":
                ids.div_(4, rounding_mode="floor").mul_(4)
            else:
                ids[-1].fill_(1)  # Entire padded/nonlocal row must become zero.
            weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1)
            act = torch.empty(m, topk, n, device=device, dtype=torch.float16)
            out = torch.empty_like(x)

            def skinny(x=x, weights=weights, ids=ids, act=act, out=out):
                ops.moe_skinny_int4_decode(
                    x, w1, s1, w2, s2, weights, ids, act, out, group, expert_map
                )
                return out

            def tile(x=x, weights=weights, ids=ids):
                return fused_experts_impl(
                    x,
                    w1,
                    w2,
                    weights,
                    ids,
                    use_int4_w4a16=True,
                    global_num_experts=e * 4,
                    expert_map=expert_map,
                    w1_scale=s1,
                    w2_scale=s2,
                    block_shape=[0, group],
                )

            reference = torch.zeros_like(x, dtype=torch.float32)
            for row in range(m):
                for slot in range(topk):
                    expert = int(expert_map[ids[row, slot]])
                    if expert < 0:
                        continue
                    gate, up = (
                        dequant(w1[expert], s1[expert], group) @ x[row].float()
                    ).chunk(2)
                    hidden = (torch.nn.functional.silu(gate) * up).half().float()
                    reference[row] += weights[row, slot] * (
                        dequant(w2[expert], s2[expert], group) @ hidden
                    )
            reference = reference.half()
            graphs, outputs = [], []
            for fn in (skinny, tile):
                result = fn()
                torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    fn()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = fn()
                graph.replay()
                torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
                graphs.append(graph)
                outputs.append(result)

            # Same arithmetic must agree exactly across eager/captured changing
            # inputs; this is separate from the FP32-reference tolerance above.
            for factor in (0.5, -1.0):
                x.mul_(factor)
                for fn, graph, result in zip((skinny, tile), graphs, outputs):
                    eager = fn().clone()
                    graph.replay()
                    torch.testing.assert_close(result, eager, atol=0, rtol=0)

            timings = [[], []]
            for sample in range(6):
                for idx in (0, 1) if sample % 2 == 0 else (1, 0):
                    timings[idx].append(graph_time(graphs[idx], args.repeats))
            medians = [statistics.median(t) for t in timings]
            print(
                json.dumps(
                    {
                        "device": args.device,
                        "rows": m,
                        "routing": routing,
                        "skinny_us": medians[0],
                        "triton_us": medians[1],
                        "speedup": medians[1] / medians[0],
                        "samples_us": timings,
                        "correct": True,
                    }
                ),
                flush=True,
            )
            for graph in graphs:
                graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 3, 6, 9, 12, 16])
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0 or any(not 1 <= m <= 16 for m in args.rows):
        parser.error("positive repeats and rows between 1 and 16 required")
    run(args)
