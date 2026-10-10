# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP all-reduce latency at serving sizes: rdna_ar one-shot vs PYNCCL (RCCL).

One process per GPU, vLLM's own distributed init, so the backends are exactly
the ones serving uses. Sizes are tokens x hidden fp16 (decode 1-24 tokens,
mixed 256-2048, prefill 2048). Every result is checked against the exact sum.

    HIP_VISIBLE_DEVICES=2,3,4,5 python benchmarks/kernels/benchmark_rdna_allreduce.py \
        --world 4 --hidden 2048,5120 --tokens 1,8,16,24,64,256,512,1024,2048

RCCL is tuned through its environment (NCCL_*, RCCL_*), so sweep settings by
running the script once per setting. Times are per call, max over ranks of
the median over repeats; `graph` replays the calls from a captured CUDA graph
(the decode/FULL path), `eager` launches them one by one.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import get_tp_group

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

    backends = {"pynccl": lambda t: pynccl.all_reduce(t)}
    if rdna is not None and not rdna.disabled:
        backends["rdna_ar"] = lambda t: rdna.all_reduce(t)
    backends["auto"] = lambda t: comm.all_reduce(t)
    if args.rs_ag:

        def rs_ag(t):
            n = t.shape[0] // args.world
            part = torch.empty((n,) + t.shape[1:], dtype=t.dtype, device=t.device)
            pynccl.reduce_scatter(part, t)
            out = torch.empty_like(t)
            pynccl.all_gather(out, part)
            return out

        backends["rs+ag"] = rs_ag

    rows = []
    for hidden in args.hidden:
        for tokens in args.tokens:
            g = torch.Generator().manual_seed(tokens * 7 + hidden)
            # Small integers: the fp16 sum is exact, so any mismatch is a bug.
            full = torch.randint(-8, 8, (args.world, tokens, hidden), generator=g)
            ref = full.sum(0).half().cuda()
            x = full[rank].half().cuda()
            nbytes = x.numel() * x.element_size()
            for name, fn in backends.items():
                if name == "rdna_ar" and not rdna.should_use(x):
                    continue
                if name == "rs+ag" and tokens % args.world:
                    continue
                ok = bool(torch.equal(fn(x), ref))
                for mode in args.modes:
                    try:
                        us = _bench(fn, x, args.iters, args.repeats, mode == "graph")
                    except Exception as e:  # noqa: BLE001
                        if rank == 0:
                            print(f"{name} {mode} {tokens}x{hidden} failed: {e}")
                        us = float("nan")
                    t = torch.tensor([us])
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
                    okt = torch.tensor([int(ok)])
                    dist.all_reduce(okt, op=dist.ReduceOp.MIN, group=cpu_group)
                    row = dict(
                        backend=name,
                        mode=mode,
                        tokens=tokens,
                        hidden=hidden,
                        bytes=nbytes,
                        us=round(float(t.item()), 1),
                        busbw_GBs=round(
                            2
                            * (args.world - 1)
                            / args.world
                            * nbytes
                            / (float(t.item()) * 1e3),
                            2,
                        ),
                        correct=bool(okt.item()),
                    )
                    rows.append(row)
                    if rank == 0:
                        print(
                            f"{name:8s} {mode:5s} {tokens:5d}x{hidden:<5d} "
                            f"{nbytes / 1024:9.1f} KiB {row['us']:9.1f} us "
                            f"busbw {row['busbw_GBs']:6.2f} GB/s ok={row['correct']}",
                            flush=True,
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
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--hidden", default="2048,5120")
    p.add_argument("--tokens", default="1,8,16,24,64,256,512,1024,2048")
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--rs-ag", action="store_true", help="also time RS + AG")
    p.add_argument("--port", type=int, default=29611)
    p.add_argument("--out", default="")
    args = p.parse_args()
    args.hidden = [int(h) for h in args.hidden.split(",")]
    args.tokens = [int(t) for t in args.tokens.split(",")]
    args.modes = args.modes.split(",")
    mp.spawn(_worker, args=(args, args.port), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
