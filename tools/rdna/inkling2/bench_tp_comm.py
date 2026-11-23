#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost of the tensor-parallel messages of Inkling-2 (TP = 2, 4), per backend.

One process per GPU with vLLM's own distributed init, so the backends are the ones
the model calls through ``TensorParallel`` (``parallel.py``):

* ``all_reduce`` of the fp32 ``o_proj`` partial ``[T, 2304]`` (9.2 KB per token) and of
  the fp16 MoE partial ``[T, 2304]`` (4.6 KB per token): 2 per layer visit, 96 per
  decode step;
* ``all_gather`` of the greedy argmax partials ``[2 M, P]`` (fp32; P = 64 for M <= 4,
  else 8 x CUs) and of a vocabulary shard of logits ``[M, padded_vocab / tp]`` (fp32).

Backends: ``pynccl`` (RCCL), ``rdna_ar`` (the one-shot HIP push all-reduce, only up to
its size gate: ``--gate-kb``, sets ``VLLM_RDNA_AR_MAX_KB`` and ``ONESHOT_KB``) and
``auto`` (``tensor_model_parallel_all_reduce`` / ``..._all_gather``: exactly what the
model calls, including the custom-op wrapper).  Times are per call, the maximum over
ranks of the median over repeats; ``graph`` replays the calls from a captured HIP
graph (the decode / FULL path), ``eager`` launches them one by one.  Every result is
checked against the exact sum (small integers: exact in fp16 / fp32).

    HIP_VISIBLE_DEVICES=4,5 python bench_tp_comm.py --world 2 --out comm_tp2.json
    HIP_VISIBLE_DEVICES=2,3,4,5 python bench_tp_comm.py --world 4 --out comm_tp4.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2304


def _bench(fn, x, iters: int, repeats: int, use_graph: bool) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(5):
        fn(x)
    torch.accelerator.synchronize()
    graph = None
    if use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(stream)
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(x)
        stream.wait_stream(side)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn(x)
        torch.accelerator.synchronize()
    times = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iters):
                fn(x)
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(times)


def _worker(rank: int, args: argparse.Namespace, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if args.ar_backend == "rdna":
        os.environ["VLLM_RDNA_AR"] = "1"
        os.environ["VLLM_RDNA_AR_MAX_KB"] = str(args.gate_kb)
        os.environ["VLLM_RDNA_AR_ONESHOT_KB"] = str(args.gate_kb)
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import get_tp_group, set_custom_all_reduce
    from vllm.platforms import current_platform

    set_custom_all_reduce(current_platform.use_custom_allreduce())
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=args.world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=args.world)
    comm = get_tp_group().device_communicator
    pynccl = comm.pynccl_comm
    rdna = comm.rdna_ar_comm
    cpu_group = get_tp_group().cpu_group
    cus = torch.cuda.get_device_properties(0).multi_processor_count

    reduce_backends = {"pynccl": lambda t: pynccl.all_reduce(t)}
    if rdna is not None and not rdna.disabled:
        reduce_backends["rdna_ar"] = lambda t: rdna.all_reduce(t)
    reduce_backends["auto"] = tensor_model_parallel_all_reduce

    rows: list[dict] = []

    def run(kind: str, name: str, fn, x: torch.Tensor, want: torch.Tensor, shape: str):
        got = fn(x)
        ok = bool(torch.equal(got, want))
        for mode in args.modes:
            try:
                us = _bench(fn, x, args.iters, args.repeats, mode == "graph")
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"{kind} {name} {mode} {shape} failed: {e}")
                us = float("nan")
            t = torch.tensor([us])
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
            okt = torch.tensor([int(ok)])
            dist.all_reduce(okt, op=dist.ReduceOp.MIN, group=cpu_group)
            row = {
                "kind": kind,
                "backend": name,
                "mode": mode,
                "shape": shape,
                "dtype": str(x.dtype).removeprefix("torch."),
                "bytes": x.numel() * x.element_size(),
                "us": round(float(t.item()), 1),
                "correct": bool(okt.item()),
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"{kind:10s} {name:8s} {mode:5s} {shape:>14s} {row['dtype']:8s} "
                    f"{row['bytes'] / 1024:9.1f} KiB {row['us']:9.1f} us "
                    f"ok={row['correct']}",
                    flush=True,
                )

    for dtype in (torch.float32, torch.float16):
        for tokens in args.tokens:
            g = torch.Generator().manual_seed(tokens * 7 + (dtype == torch.float16))
            full = torch.randint(-8, 8, (args.world, tokens, HIDDEN), generator=g)
            want = full.sum(0).to(dtype).cuda()
            x = full[rank].to(dtype).cuda()
            for name, fn in reduce_backends.items():
                if name == "rdna_ar" and not rdna.should_use(x):
                    continue
                run("all_reduce", name, fn, x, want, f"{tokens}x{HIDDEN}")

    for m in args.gather_rows:
        parts = 64 if m <= 4 else 8 * cus
        for shape, name in (
            ((2 * m, parts), "argmax"),
            ((m, args.logits_shard), "logits"),
        ):
            g = torch.Generator().manual_seed(m * 13 + shape[1])
            full = torch.randint(-100, 100, (args.world, *shape), generator=g)
            want = torch.cat(list(full.float()), dim=-1).cuda()
            x = full[rank].float().cuda()
            run(
                "all_gather",
                "auto",
                lambda t: tensor_model_parallel_all_gather(t, -1),
                x,
                want,
                f"{name} {shape[0]}x{shape[1]}",
            )
    if rank == 0 and args.out:
        env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "RCCL_", "VLLM_RDNA_AR", "HSA_", "GPU_MAX"))
        }
        with open(args.out, "w") as f:
            json.dump({"env": env, "world": args.world, "rows": rows}, f, indent=1)
    dist.barrier()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,256,1024")
    p.add_argument("--gather-rows", default="1,4,16")
    p.add_argument(
        "--logits-shard",
        type=int,
        default=0,
        help="vocabulary shard of the logits gather (default: 200064 / world)",
    )
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--ar-backend", choices=["rccl", "rdna"], default="rdna")
    p.add_argument("--gate-kb", type=int, default=256, help="rdna_ar one-shot gate")
    p.add_argument("--port", type=int, default=29613)
    p.add_argument("--out", default="")
    args = p.parse_args()
    args.tokens = [int(t) for t in args.tokens.split(",")]
    args.gather_rows = [int(t) for t in args.gather_rows.split(",")]
    args.modes = args.modes.split(",")
    args.logits_shard = args.logits_shard or 200064 // args.world
    mp.spawn(_worker, args=(args, args.port), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
