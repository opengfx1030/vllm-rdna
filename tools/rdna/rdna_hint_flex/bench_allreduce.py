#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RCCL vs rdna_ar all-reduce, and GEMM + all-reduce overlap, on two GPUs (TP=2).

    HIP_VISIBLE_DEVICES=0,1 python3 tools/rdna/rdna_hint_flex/bench_allreduce.py sweep
    HIP_VISIBLE_DEVICES=0,1 python3 tools/rdna/rdna_hint_flex/bench_allreduce.py overlap

``sweep``: fp16 all-reduce of 16 KiB .. 32 MiB (the TP2 prefill all-reduce is
2048 x 5120 x 2 B = 20 MiB, 10 MiB for a 1024-token chunk), us per call and GB/s
of payload, back to back on one stream, RCCL (torch.distributed nccl backend,
the recipe's NCCL_* environment) against the rdna_ar push kernel with its gate
raised past the size, at the default launch width and at forced widths. Every
size also compares the two results bit for bit.

``overlap``: the row-parallel GEMM of the 27B at the scheduler's chunk (down_proj
K=8704 and o_proj / out_proj K=3072 per rank, N=5120, M=2048 or 1024) followed by
its all-reduce, once in sequence and once split over M into k pieces whose
all-reduces run on RCCL's stream while the next piece multiplies. The GEMM here
is rocBLAS fp16, a stand-in for the dequant + GEMM of the model: what it shows is
whether the PCIe transfer and the GEMM can overlap on this card, and what the
RCCL kernel costs the GEMM while both run.

Rank 0 prints a table and, with --csv, writes it. The sweep prints the GPU sclk /
power / junction temperature at the start and the end.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import tempfile
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_KIB = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 8192, 10240, 16384]
SIZES_KIB += [20480, 32768]
WIDTHS = (8, 16, 32, 64)


def gpu_state(rank: int) -> str:
    """Sclk / power / junction of this rank's GPU from sysfs (empty when absent)."""
    p = torch.cuda.get_device_properties(rank)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    hw = glob.glob(f"/sys/bus/pci/devices/{bdf}/hwmon/hwmon*")
    if not hw:
        return ""

    def rd(name: str) -> int:
        try:
            with open(f"{hw[0]}/{name}") as fh:
                return int(fh.read())
        except (OSError, ValueError):
            return 0

    return (
        f"bus {p.pci_bus_id:02x} sclk {rd('freq1_input') / 1e6:.0f} MHz "
        f"power {rd('power1_average') / 1e6:.0f} W "
        f"junction {rd('temp2_input') / 1e3:.0f} C"
    )


def timed(fn, iters: int, warmup: int = 3) -> float:
    """Microseconds per call of fn() over iters back-to-back calls."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def worst(rank: int, value: float, world: int) -> float:
    """The slowest rank's value (the collective is as slow as its last rank)."""
    got = [None] * world
    dist.all_gather_object(got, value)
    return max(got)


def setup(rank: int, world: int, args):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(args.port),
        VLLM_CACHE_ROOT=args.cache,
        VLLM_RDNA_AR_MAX_KB=str(max(SIZES_KIB)),
        VLLM_RDNA_AR_ONESHOT_KB=str(max(SIZES_KIB)),
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    pg = dist.new_group(list(range(world)), backend="nccl")
    t = torch.ones(8, device="cuda")
    dist.all_reduce(t, group=pg)
    torch.accelerator.synchronize()
    assert float(t[0]) == world
    return pg


def sweep(rank: int, world: int, args) -> None:
    pg = setup(rank, world, args)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
    )

    # VLLM_RDNA_AR_VPUSH is read when an instance is created: the first is the
    # shipped scalar push, the optional second one pushes 16 bytes per store.
    os.environ["VLLM_RDNA_AR_VPUSH"] = "0"
    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    comm_v = None
    if args.vpush:
        os.environ["VLLM_RDNA_AR_VPUSH"] = "1"
        comm_v = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    ops = comm._ops
    assert not comm.disabled, "rdna_ar did not start"
    assert comm_v is None or not comm_v.disabled, "rdna_ar (vpush) did not start"
    if rank == 0:
        print(f"start: {gpu_state(rank)}", flush=True)
    rows = []
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    for kib in SIZES_KIB:
        n = kib * 1024 // 2
        iters = 100 if kib <= 256 else 40 if kib <= 4096 else 15
        bufs = [
            (torch.randn(n, device="cuda", generator=gen) * 0.05).half()
            for _ in range(3)
        ]
        i = [0]

        def call_rccl(bufs=bufs, i=i):
            i[0] += 1
            dist.all_reduce(bufs[i[0] % 3], group=pg)

        def call_ours(c, w, bufs=bufs, i=i):
            i[0] += 1
            ops.rdna_ar_all_reduce_algo(c.handle, bufs[i[0] % 3], -1, w)

        # exact: same inputs through both, compared bit for bit
        x = bufs[0].clone()
        theirs = x.clone()
        dist.all_reduce(theirs, group=pg)
        mine = ops.rdna_ar_all_reduce_algo(comm.handle, x, -1, 0)
        torch.accelerator.synchronize()
        same = bool(torch.equal(mine, theirs))
        maxdiff = float((mine.float() - theirs.float()).abs().max())
        t_rccl = worst(rank, timed(call_rccl, iters), world)
        t_auto = worst(rank, timed(partial(call_ours, comm, 0), iters), world)
        widths = {
            w: worst(rank, timed(partial(call_ours, comm, w), iters), world)
            for w in WIDTHS
        }
        best_w = min(widths, key=widths.get)
        row = {
            "kib": kib,
            "rccl_us": t_rccl,
            "rccl_GBps": kib * 1024 / t_rccl / 1e3,
            "ar_us": t_auto,
            "ar_GBps": kib * 1024 / t_auto / 1e3,
            "ar_best_blocks": best_w,
            "ar_best_us": widths[best_w],
            "ar_best_GBps": kib * 1024 / widths[best_w] / 1e3,
            "bit_equal": same,
            "max_abs_diff": maxdiff,
        }
        tail = ""
        if comm_v is not None:
            pushed = ops.rdna_ar_all_reduce_algo(comm_v.handle, x, -1, 0)
            torch.accelerator.synchronize()
            same_push = bool(torch.equal(mine, pushed))
            v_widths = {
                w: worst(rank, timed(partial(call_ours, comm_v, w), iters), world)
                for w in (0, *WIDTHS)
            }
            v_best_w = min(v_widths, key=v_widths.get)
            row.update(
                vpush_us=v_widths[0],
                vpush_best_blocks=v_best_w,
                vpush_best_us=v_widths[v_best_w],
                vpush_best_GBps=kib * 1024 / v_widths[v_best_w] / 1e3,
                vpush_equal_scalar=same_push,
            )
            tail = (
                f" | 16B push best {v_best_w:2d} blocks {v_widths[v_best_w]:9.1f} us "
                f"{row['vpush_best_GBps']:5.2f} GB/s, equal to scalar {same_push}"
            )
        rows.append(row)
        if rank == 0:
            print(
                f"{kib:6d} KiB  rccl {t_rccl:9.1f} us {row['rccl_GBps']:5.2f} GB/s | "
                f"rdna_ar {t_auto:9.1f} us {row['ar_GBps']:5.2f} GB/s | best "
                f"{best_w:2d} blocks {widths[best_w]:9.1f} us "
                f"{row['ar_best_GBps']:5.2f} GB/s | equal to RCCL {same} "
                f"(max |d| {maxdiff:.2e}){tail}",
                flush=True,
            )
    if rank == 0:
        print(f"end:   {gpu_state(rank)}", flush=True)
        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    dist.barrier()
    dist.destroy_process_group()


def overlap(rank: int, world: int, args) -> None:
    pg = setup(rank, world, args)
    n_out = 5120
    cases = [("down_proj", 8704), ("o_proj/out_proj", 3072)]
    rows = []
    gen = torch.Generator(device="cuda").manual_seed(7 + rank)
    if rank == 0:
        print(f"start: {gpu_state(rank)}", flush=True)
    for name, k_dim in cases:
        w = (torch.randn(k_dim, n_out, device="cuda", generator=gen) * 0.02).half()
        for m in args.rows:
            x = (torch.randn(m, k_dim, device="cuda", generator=gen) * 0.5).half()
            y = torch.empty(m, n_out, device="cuda", dtype=torch.half)
            iters = 10

            def gemm_only(x=x, w=w, y=y):
                torch.matmul(x, w, out=y)

            def ar_only(y=y):
                dist.all_reduce(y, group=pg)
                y.mul_(0.5)  # keep the values bounded, same work every call

            def sequential(x=x, w=w, y=y):
                torch.matmul(x, w, out=y)
                dist.all_reduce(y, group=pg)
                y.mul_(0.5)

            def pipelined(k, x=x, w=w, y=y, m=m):
                step = (m // k + 255) // 256 * 256
                works = []
                for r0 in range(0, m, step):
                    r1 = min(m, r0 + step)
                    torch.matmul(x[r0:r1], w, out=y[r0:r1])
                    works.append(dist.all_reduce(y[r0:r1], group=pg, async_op=True))
                for wk in works:
                    wk.wait()
                y.mul_(0.5)

            t_g = worst(rank, timed(gemm_only, iters), world)
            t_a = worst(rank, timed(ar_only, iters), world)
            t_s = worst(rank, timed(sequential, iters), world)
            row = {
                "layer": name,
                "m": m,
                "gemm_us": t_g,
                "ar_us": t_a,
                "sequential_us": t_s,
            }
            for k in args.splits:
                row[f"pipe{k}_us"] = worst(
                    rank, timed(partial(pipelined, k), iters), world
                )
            rows.append(row)
            if rank == 0:
                pipes = " ".join(
                    f"k={k} {row[f'pipe{k}_us']:8.0f} us "
                    f"({100 * (1 - row[f'pipe{k}_us'] / t_s):+5.1f}%)"
                    for k in args.splits
                )
                print(
                    f"{name:16s} M={m:5d}  gemm {t_g:8.0f}  all-reduce {t_a:8.0f}  "
                    f"sequential {t_s:8.0f} us | {pipes}",
                    flush=True,
                )
    if rank == 0:
        print(f"end:   {gpu_state(rank)}", flush=True)
        if args.csv:
            fields = sorted({f for r in rows for f in r}, key=lambda f: f != "layer")
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        med = statistics.median(r["sequential_us"] for r in rows)
        print(f"median sequential {med:.0f} us")
    dist.barrier()
    dist.destroy_process_group()


def check_buses(spec: str) -> bool:
    """Do the visible devices sit on the PCI buses in `spec` (hex, in order)?"""
    want = [int(b, 16) for b in spec.split(",") if b]
    got = [torch.cuda.get_device_properties(i).pci_bus_id for i in range(len(want))]
    print(f"devices on buses {[f'{b:02x}' for b in got]}, expected {spec}")
    return torch.accelerator.device_count() == len(want) and got == want


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=("sweep", "overlap"))
    ap.add_argument("--port", type=int, default=29671)
    ap.add_argument("--csv", default="")
    ap.add_argument("--rows", type=int, nargs="+", default=[2048, 1024])
    ap.add_argument("--splits", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "ar_bench"))
    ap.add_argument(
        "--buses", default="", help="expected PCI buses of the visible GPUs, e.g. d1,d4"
    )
    ap.add_argument(
        "--vpush",
        action="store_true",
        help="sweep: also run an instance with VLLM_RDNA_AR_VPUSH=1 (needs the 16-byte"
        " push in _rocm_C)",
    )
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    world = torch.accelerator.device_count()
    if world != 2:
        print(f"needs exactly 2 visible GPUs, found {world}", file=sys.stderr)
        return 2
    if args.buses and not check_buses(args.buses):
        print("refusing to run: the visible GPUs are not the expected buses")
        return 2
    mp.spawn(sweep if args.mode == "sweep" else overlap, args=(world, args), nprocs=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
