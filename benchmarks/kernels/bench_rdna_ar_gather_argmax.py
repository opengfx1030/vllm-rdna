# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latency of the rdna_ar HIP all-gather / vocab argmax against the RCCL path.

Baselines are what the decode step runs today:
  * gather: ``all_gather_into_tensor`` on the last dim (RCCL)
  * argmax: ``get_top_tokens``'s pair gather + argmax + gather + cast
Variants alternate inside each round so a noisy neighbour on a shared host hits
both. Run: ``python bench_rdna_ar_gather_argmax.py --world 4`` with
``HIP_VISIBLE_DEVICES`` set to idle GPUs. Reports microseconds per call
(pipelined, no per-call sync) as median over rounds.
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _timed(fn, iters: int) -> float:
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _worker(rank: int, world: int, port: int, args: argparse.Namespace) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
    )

    dev = torch.device("cuda", rank)
    comm = RdnaOneShotAllReduce(dist.group.WORLD, dev)
    assert not comm.disabled
    results: list[tuple[str, float, float]] = []

    def run_pair(name: str, base, hip) -> None:
        base()
        hip()
        torch.accelerator.synchronize()
        dist.barrier()
        b: list[float] = []
        h: list[float] = []
        for _ in range(args.rounds):
            b.append(_timed(base, args.iters))
            dist.barrier()
            h.append(_timed(hip, args.iters))
            dist.barrier()
        results.append((name, statistics.median(b), statistics.median(h)))

    for rows, cols, dtype in ((1, 640, torch.float16), (24, 640, torch.float16)):
        x = torch.randn(rows, cols, device=dev).to(dtype)

        def base_gather(x=x):
            out = torch.empty(
                (world * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device
            )
            dist.all_gather_into_tensor(out, x.contiguous())
            return (
                out.reshape(world, *x.shape)
                .movedim(0, -2)
                .reshape(x.shape[0], world * x.shape[1])
            )

        run_pair(
            f"all_gather [{rows},{cols}] {str(dtype)[6:]}",
            base_gather,
            lambda x=x: comm.all_gather(x),
        )

    for rows in (1, 8, 24):
        vals = torch.rand(rows, device=dev).half()
        idx = (torch.randint(0, 1024, (rows,), device=dev) + rank * 1024).long()

        def base_argmax(vals=vals, idx=idx, rows=rows):
            pair = torch.stack([vals.float(), idx.float()], dim=-1)
            out = torch.empty((world * rows, 2), dtype=pair.dtype, device=dev)
            dist.all_gather_into_tensor(out, pair)
            g = out.reshape(world, rows, 2).movedim(0, -2).reshape(rows, 2 * world)
            g = g.view(rows, world, 2)
            w = g[:, :, 0].argmax(dim=-1, keepdim=True)
            return g[:, :, 1].gather(dim=-1, index=w).squeeze(-1).to(torch.int64)

        run_pair(
            f"argmax pair  rows={rows}",
            base_argmax,
            lambda vals=vals, idx=idx: comm.argmax(vals, idx),
        )

    gathered: list = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        print(f"world={world}  us/call, median of {args.rounds} rounds x {args.iters}")
        print(f"{'case':34s} {'RCCL path':>10s} {'HIP':>10s} {'speedup':>8s}")
        for name, base, hip in gathered[0]:
            print(f"{name:34s} {base:10.1f} {hip:10.1f} {base / hip:7.2f}x")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--port", type=int, default=29541)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world, args.port, args), nprocs=args.world)


if __name__ == "__main__":
    main()
