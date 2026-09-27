# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify RDNA all-reduce and compare graph latency with RCCL on four GPUs.

Run with the matching ROCm environment via torchrun --standalone --nproc-per-node=4.
Requires exclusive GPU access. Does not start or stop services.
"""

import argparse
import json
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.rdna_all_reduce import RdnaOneShotAllReduce


def agree(condition, label):
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, None if condition else label)
    if any(failure is not None for failure in failures):
        raise AssertionError(f"{label}: rank failures {failures}")


def graph_latency(graph, repeats):
    samples = []
    for _ in range(5):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, statistics.median(samples))
    return max(ranks)


def run(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    assert dist.get_world_size() == 4, "This qualification requires four GPUs"
    os.environ["VLLM_RDNA_AR_MAX_KB"] = "64"
    backend = RdnaOneShotAllReduce(dist.group.WORLD, device)
    agree(not backend.disabled, "native startup self-test")
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=90))
    pynccl = PyNcclCommunicator(dist.group.WORLD, device)
    agree(not pynccl.disabled, "vLLM RCCL communicator startup")
    rank_sum = 10
    try:
        for dtype in (torch.float16, torch.float32):
            width = torch.empty((), dtype=dtype).element_size()
            for n in (2560, 7680, 15360, 30720, 65536 // width):
                if n * width > 65536:
                    continue
                x = torch.full((n,), rank + 1, dtype=dtype, device=device)
                z = torch.full_like(x, 2 * (rank + 1))
                agree(backend.should_use(x), "eligible decode payload")
                first = backend.all_reduce(x)
                second = backend.all_reduce(z)
                torch.accelerator.synchronize()
                backend.check()
                agree(bool((second == 2 * rank_sum).all()), "second eager sum")
                agree(bool((first == rank_sum).all()), "retained eager output")

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    backend.all_reduce(x)
                torch.cuda.current_stream().wait_stream(stream)
                torch.accelerator.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    first = backend.all_reduce(x)
                    second = backend.all_reduce(z)
                for step in range(1, 17):
                    x.fill_((rank + 1) * step)
                    z.fill_((rank + 1) * (step + 1))
                    graph.replay()
                    torch.accelerator.synchronize()
                    backend.check()
                    agree(bool((first == rank_sum * step).all()), "graph first sum")
                    agree(
                        bool((second == rank_sum * (step + 1)).all()),
                        "graph second sum",
                    )
                    # Eager work must not overwrite outputs retained by a graph.
                    backend.all_reduce(x)
                    torch.accelerator.synchronize()
                    agree(
                        bool((second == rank_sum * (step + 1)).all()),
                        "graph output survives eager work",
                    )

                row = {"dtype": str(dtype), "bytes": n * width, "correct": True}
                if not args.correctness_only:
                    # Both timing graphs contain two out-of-place reductions.
                    scratch1 = torch.empty_like(x)
                    scratch2 = torch.empty_like(z)
                    with torch.cuda.stream(stream):
                        scratch1.copy_(x)
                        dist.all_reduce(scratch1, group=rccl)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    rccl_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(rccl_graph, stream=stream):
                        scratch1.copy_(x)
                        dist.all_reduce(scratch1, group=rccl)
                        scratch2.copy_(z)
                        dist.all_reduce(scratch2, group=rccl)
                    rccl_graph.replay()
                    torch.accelerator.synchronize()
                    agree(bool(torch.equal(first, scratch1)), "RCCL first reference")
                    agree(bool(torch.equal(second, scratch2)), "RCCL second reference")
                    row["rdna_us_per_collective"] = (
                        graph_latency(graph, args.repeats) / 2
                    )
                    backend.check()
                    row["rccl_us_per_collective"] = (
                        graph_latency(rccl_graph, args.repeats) / 2
                    )
                    # RCCL graph references must be released before destroying
                    # their process group, otherwise teardown can wait forever.
                    rccl_graph.reset()
                    with torch.cuda.stream(stream):
                        pynccl.all_reduce(x, stream=stream)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    pynccl_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pynccl_graph, stream=stream):
                        direct1 = pynccl.all_reduce(x, stream=stream)
                        direct2 = pynccl.all_reduce(z, stream=stream)
                    pynccl_graph.replay()
                    torch.accelerator.synchronize()
                    agree(
                        bool(torch.equal(first, direct1)), "vLLM RCCL first reference"
                    )
                    agree(
                        bool(torch.equal(second, direct2)), "vLLM RCCL second reference"
                    )
                    row["vllm_rccl_us_per_collective"] = (
                        graph_latency(pynccl_graph, args.repeats) / 2
                    )
                    pynccl_graph.reset()
                graph.reset()
                if rank == 0:
                    print(json.dumps(row), flush=True)
            oversized = torch.empty(65536 // width + 1, dtype=dtype, device=device)
            agree(not backend.should_use(oversized), "oversized payload fallback")
    finally:
        pynccl.destroy()
        dist.destroy_process_group(rccl)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    run(args)
