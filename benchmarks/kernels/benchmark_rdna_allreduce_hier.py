# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 all-reduce latency across two PCIe islands: flat rdna_ar vs hier vs RCCL.

One process per GPU, vLLM's own distributed init, so the backends are the
ones serving uses (PyNccl = RCCL; rdna_ar with VLLM_RDNA_AR=1). Configs:

  rccl           PyNccl all_reduce (set NCCL_P2P_LEVEL / NCCL_PROTO in the env)
  flat           rdna_ar flat exchange, rank-order fp32 sum (VLLM_RDNA_AR_HIER=0)
  flat-isl       flat exchange with the island-grouped arithmetic of hier
  hier[:xN][:bM] reduce-scatter / cross-root exchange / all-gather, N owners
                 per island (default: every rank), M blocks (default: by size)
  auto           the size dispatch of the instance as the env configures it: the
                 flat one-shot below VLLM_RDNA_AR_HIER_MIN_KB, hier from there

Time per call = slope of the graph replay time over the number of calls in the
graph (K_HI vs K_LO calls back to back), which cancels the start skew between
ranks and the graph launch cost; reported as the median over --replays replays
of the slowest rank, then median / min / max over --repeats. Every config's
result is checked against the exact sum (small integers) before it is timed.

    HIP_VISIBLE_DEVICES=1,2,3,4,6,7,8,9 NCCL_P2P_LEVEL=pxb NCCL_PROTO=Simple \\
      python benchmarks/kernels/benchmark_rdna_allreduce_hier.py --world 8 \\
      --hidden 2560 --tokens 1,3,6,12,24,64,256,1024 --out hier_bench.jsonl
"""

import argparse
import contextlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

K_LO, K_HI = 4, 20


def _parse_config(spec: str) -> dict:
    name, *opts = spec.split(":")
    cfg: dict = {"spec": spec, "name": name, "owners": 0, "blocks": 0}
    for opt in opts:
        if opt[:1] == "x" and opt[1:].isdigit():
            cfg["owners"] = int(opt[1:])
        elif opt[:1] == "b" and opt[1:].isdigit():
            cfg["blocks"] = int(opt[1:])
        else:
            raise SystemExit(f"bad config option {opt!r} in {spec!r}")
    if name not in ("rccl", "flat", "flat-isl", "hier", "auto"):
        raise SystemExit(f"bad config {spec!r}")
    return cfg


def _graph_time_us(fn, x, calls: int, replays: int, cpu_group) -> float:
    """Median over replays of the elapsed time of `calls` calls in one graph."""
    stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x)
    stream.wait_stream(side)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn(x)
    torch.accelerator.synchronize()
    for _ in range(3):
        dist.barrier(group=cpu_group)
        graph.replay()
    torch.accelerator.synchronize()
    times = []
    for _ in range(replays):
        dist.barrier(group=cpu_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        graph.replay()
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    del graph
    return statistics.median(times)


def _worker(rank: int, args: argparse.Namespace, port: int, q) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    max_kb = max(args.sizes_kb)
    # The gate also sizes the staging (2 x world x gate bytes per GPU).
    os.environ["VLLM_RDNA_AR"] = "1"
    os.environ["VLLM_RDNA_AR_MAX_KB"] = str(max_kb)
    os.environ["VLLM_RDNA_AR_ONESHOT_KB"] = str(max_kb)
    os.environ["VLLM_RDNA_AR_HIER_MAX_KB"] = str(max_kb)
    os.environ["VLLM_RDNA_AR_HIER"] = args.hier_mode
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import get_tp_group, set_custom_all_reduce
    from vllm.platforms import current_platform

    # ParallelConfig does this from current_platform.use_custom_allreduce(); the
    # bare init path would otherwise build CustomAllreduce, which needs amdsmi.
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
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    have_rdna = rdna is not None and not rdna.disabled
    islands = rdna.hier_islands if have_rdna else None
    if rank == 0:
        buses = [
            hex(torch.cuda.get_device_properties(i).pci_bus_id)
            for i in range(args.world)
        ]
        print(
            f"rdna_ar active: {have_rdna}; hier islands: {islands}; bus ids: {buses}",
            flush=True,
        )
    configs = [_parse_config(s) for s in args.configs.split(",")]
    for cfg in configs:
        cfg["skip"] = (
            "rdna_ar disabled"
            if cfg["name"] != "rccl" and not have_rdna
            else "no hier islands"
            if cfg["name"] in ("flat-isl", "hier", "auto") and islands is None
            else None
        )

    def reconfigure(cfg: dict) -> None:
        """Owners / blocks of the hier launches: same call on every rank."""
        dist.barrier(group=cpu_group)
        if cfg["name"] in ("hier", "auto") and islands is not None:
            # "auto" is the dispatch as configured by the environment
            rdna._ops.rdna_ar_hier_init(
                rdna.handle,
                torch.tensor(islands, dtype=torch.int64),
                cfg["owners"],
                rdna._hier_cfg.min_kb if cfg["name"] == "auto" else 0,
                -1,
                cfg["blocks"],
                max_kb,
            )
        dist.barrier(group=cpu_group)

    def make_fn(cfg: dict):
        algo = {"flat": 0, "flat-isl": 1, "hier": 2}.get(cfg["name"])
        if cfg["name"] == "rccl":
            return lambda t: pynccl.all_reduce(t)
        if cfg["name"] == "auto":
            return lambda t: rdna._ops.rdna_ar_all_reduce(rdna.handle, t)
        return lambda t: rdna._ops.rdna_ar_all_reduce_algo(rdna.handle, t, algo, 0)

    for size_kb in args.sizes_kb:
        n = size_kb * 1024 // torch.empty((), dtype=dtype).element_size()
        g = torch.Generator().manual_seed(size_kb)
        # Small integers: every partial sum is exact in every dtype, so any
        # mismatch is a bug, not rounding.
        full = torch.randint(-8, 8, (args.world, n), generator=g)
        ref = full.sum(0).to(dtype).cuda()
        x = full[rank].to(dtype).cuda()
        for cfg in configs:
            row = {
                "config": cfg["spec"],
                "size_kb": size_kb,
                "dtype": args.dtype,
                "world": args.world,
            }
            if cfg["skip"]:
                row["skipped"] = cfg["skip"]
            elif cfg["name"] != "rccl" and not rdna.should_use(x):
                row["skipped"] = "above the rdna_ar gate"
            else:
                reconfigure(cfg)
                fn = make_fn(cfg)
                exact = bool(torch.equal(fn(x), ref))
                torch.accelerator.synchronize()
                reps = []
                for _ in range(args.repeats):
                    # fewer, longer graphs for big payloads
                    replays = args.replays if size_kb <= 1024 else args.replays // 4
                    k_hi = K_HI if size_kb <= 1024 else K_LO + 4
                    t_lo = _graph_time_us(fn, x, K_LO, replays, cpu_group)
                    t_hi = _graph_time_us(fn, x, k_hi, replays, cpu_group)
                    reps.append(((t_hi - t_lo) / (k_hi - K_LO), t_hi / k_hi))
                code = (
                    int(rdna._ops.rdna_ar_timeout_info(rdna.handle)) if have_rdna else 0
                )
                row.update(
                    exact=exact,
                    us=[round(r[0], 2) for r in reps],
                    us_naive=[round(r[1], 2) for r in reps],
                    abort=code,
                )
            gathered: list = [None] * args.world
            dist.all_gather_object(gathered, row, group=cpu_group)
            if rank == 0:
                merged = dict(row)
                if "us" in row:
                    # slowest rank per repeat
                    merged["us_worst_rank"] = [
                        round(max(g_["us"][i] for g_ in gathered), 2)
                        for i in range(len(row["us"]))
                    ]
                    merged["exact"] = all(g_["exact"] for g_ in gathered)
                    merged["abort"] = max(g_["abort"] for g_ in gathered)
                    w = merged["us_worst_rank"]
                    merged["us_median"] = round(statistics.median(w), 2)
                    merged["us_min"], merged["us_max"] = min(w), max(w)
                q.put(merged)
    dist.barrier(group=cpu_group)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--tokens", default="", help="sizes as tokens x hidden")
    ap.add_argument("--sizes-kb", default="", help="sizes in KiB (instead)")
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    ap.add_argument("--configs", default="rccl,flat,flat-isl,hier:x4,hier:x2,hier:x1")
    ap.add_argument("--hier-mode", default="1", help="VLLM_RDNA_AR_HIER of the run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    elt = {"fp16": 2, "bf16": 2, "fp32": 4}[args.dtype]
    if args.sizes_kb:
        args.sizes_kb = [int(s) for s in args.sizes_kb.split(",")]
    else:
        toks = args.tokens or "1,3,6,12,24,64,256,1024,4096"
        args.sizes_kb = [
            max(1, int(t) * args.hidden * elt // 1024) for t in toks.split(",")
        ]
    port = 29600 + os.getpid() % 300
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, args, port, q)) for r in range(args.world)
    ]
    t0 = time.time()
    for p in procs:
        p.start()
    expected = len(args.sizes_kb) * len(args.configs.split(","))
    got = 0
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(open(args.out, "a")) if args.out else None
        while got < expected:
            try:
                row = q.get(timeout=1800)
            except Exception:
                print("timeout waiting for results", flush=True)
                break
            got += 1
            row["host_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            row["visible"] = os.environ.get("HIP_VISIBLE_DEVICES", "")
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
            if "us_median" in row:
                print(
                    f"{row['size_kb']:>6} KiB {row['config']:<12} "
                    f"{row['us_median']:>8.2f} us (min {row['us_min']:.2f} max "
                    f"{row['us_max']:.2f}) exact={row['exact']} "
                    f"abort={row['abort']}",
                    flush=True,
                )
            else:
                print(f"{row['size_kb']:>6} KiB {row['config']:<12} {row['skipped']}")
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.kill()
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
