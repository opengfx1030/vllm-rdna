# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stress the rdna_ar one-shot all-reduce: wedges per size, topology and load.

One process per visible GPU (HIP_VISIBLE_DEVICES picks the set) runs the
serving path, RdnaOneShotAllReduce, from a CUDA graph of --graph-k collectives
replayed back to back. Every collective's input changes on every replay (an
in-graph integer bump), so a stale staging slot of the same parity shows up as
a wrong sum, and every result is compared in-graph with the exact sum of all
ranks' inputs (each rank mirrors every rank's payload). After each batch of
replays the ranks read the kernel's abort record (a wedge) and late-peer
record, and agree over gloo whether to go on.

Background PCIe load (--traffic), from separate processes, one per GPU:
  p2p   64 MiB copies to the next GPU of the set (a ring: every GPU sends and
        receives peer-to-peer writes over the switch paths the one-shot uses)
  host  alternating 64 MiB host-to-device and device-to-host copies (the
        GPU <-> root-port uplink)
  mix   both

A wedge or a wrong sum ends the attempt; the cell restarts on fresh processes
until it has --collectives collectives or --max-wedges wedges. PCIe AER
counters (/sys/bus/pci/devices/*/aer_dev_*) are diffed around every cell.

    HIP_VISIBLE_DEVICES=2,3,4,5 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2 \\
      python tools/rdna/rdna_ar_stress.py --sizes-kb 64,128,256 \\
      --traffic none,mix --collectives 2000000 --wait-ms 20000 \\
      --label tp4 --out stress.jsonl

--stall-rank/--stall-ms/--stall-every make one rank sleep on the host before
every Nth batch, so its peers wait --stall-ms for it at the next collective
(the late-peer path). --wait-ms 0 keeps the kernel's default wait bound.
"""

import argparse
import glob
import json
import os
import queue
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MIB = 1 << 20
_MARKER = "rdna_ar_wedged"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _aer_snapshot() -> dict[str, dict[str, int]]:
    """Non-zero AER counters of every PCI function: {bdf: {kind:name: n}}."""
    snap: dict[str, dict[str, int]] = {}
    for path in glob.glob("/sys/bus/pci/devices/*/aer_dev_*"):
        bdf = os.path.basename(os.path.dirname(path))
        kind = os.path.basename(path)[len("aer_dev_") :]
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in lines:
            name, _, val = line.strip().rpartition(" ")
            if name and val.isdigit() and int(val):
                snap.setdefault(bdf, {})[f"{kind}:{name}"] = int(val)
    return snap


def _aer_delta(before: dict, after: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for bdf, counters in after.items():
        for name, val in counters.items():
            d = val - before.get(bdf, {}).get(name, 0)
            if d:
                out.setdefault(bdf, {})[name] = d
    return out


def _traffic_main(dev: int, world: int, mode: str, stop, out_q) -> None:
    torch.accelerator.set_device_index(dev)
    nbytes = 64 * MIB
    src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dev}")
    peer_dst = None
    host = None
    if mode in ("p2p", "mix") and world > 1:
        peer = (dev + 1) % world
        peer_dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{peer}")
    if mode in ("host", "mix"):
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()
    moved = 0
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        while not stop.is_set():
            for _ in range(4):
                if peer_dst is not None:
                    peer_dst.copy_(src, non_blocking=True)
                    moved += nbytes
                if host is not None:
                    src.copy_(host, non_blocking=True)
                    host.copy_(src, non_blocking=True)
                    moved += 2 * nbytes
            stream.synchronize()
    out_q.put(("traffic", dev, moved / (time.perf_counter() - t0) / 1e9))


def _rank_main(rank: int, world: int, port: int, cfg: dict, out_q) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_CACHE_ROOT=cfg["cache_root"],
        VLLM_RDNA_AR_MAX_KB=str(cfg["size_kb"]),
        VLLM_RDNA_AR_ONESHOT_KB=str(cfg["size_kb"]),
    )
    if cfg["wait_ms"] > 0:
        os.environ["VLLM_RDNA_AR_WAIT_MS"] = str(cfg["wait_ms"])
    else:
        os.environ.pop("VLLM_RDNA_AR_WAIT_MS", None)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm import _custom_ops as ops
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
        describe_abort,
        describe_late,
    )

    res: dict = {
        "rank": rank,
        "bus": hex(torch.cuda.get_device_properties(rank).pci_bus_id),
    }
    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    if comm.disabled:
        res["error"] = "rdna_ar disabled (init or boot self-test failed)"
        out_q.put(("result", rank, res))
        return

    n = cfg["size_kb"] * 1024 // 2  # fp16 elements
    k_graph = cfg["graph_k"]
    # Integer payloads below 2048 / world: every partial and full sum is an
    # exact fp16 integer, so the check is exact for any summation order.
    mask = 2048 // world - 1
    idx = torch.arange(n, dtype=torch.int32, device="cuda")

    def payload(r: int, k: int) -> torch.Tensor:
        return ((idx + r * 131 + k * 17) & mask).to(torch.int16)

    cnt = [payload(rank, k) for k in range(k_graph)]
    mirror = [
        torch.stack([payload(r, k) for r in range(world)]) for k in range(k_graph)
    ]
    inp = [c.to(torch.float16) for c in cnt]
    bad = torch.zeros(1, dtype=torch.int64, device="cuda")

    def body() -> None:
        for k in range(k_graph):
            cnt[k].add_(1).bitwise_and_(mask)
            mirror[k].add_(1).bitwise_and_(mask)
            inp[k].copy_(cnt[k])
            out = comm.all_reduce(inp[k])
            want = mirror[k].sum(0, dtype=torch.int32)
            bad.add_((out.to(torch.int32) != want).sum())

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
        body()
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.accelerator.synchronize()
    warm_bad = int(bad.item())
    bad.zero_()

    per_batch = cfg["batch_replays"] * k_graph
    stall_every = cfg["stall_every"]
    done = 0
    batches = 0
    late = 0
    max_late_ms = 0
    late_examples: list[str] = []
    last_slow = int(ops.rdna_ar_slow_info(comm.handle))
    wedge = None
    flags = torch.zeros(2, dtype=torch.int64)
    last_hb = 0.0
    t_start = time.perf_counter()
    while done < cfg["collectives"]:
        if (
            stall_every
            and rank == cfg["stall_rank"]
            and batches % stall_every == stall_every - 1
        ):
            time.sleep(cfg["stall_ms"] / 1000.0)
        for _ in range(cfg["batch_replays"]):
            graph.replay()
        torch.accelerator.synchronize()
        done += per_batch
        batches += 1
        slow = int(ops.rdna_ar_slow_info(comm.handle))
        if slow and slow != last_slow:
            last_slow = slow
            late += 1
            max_late_ms = max(max_late_ms, ((slow >> 16) & 0xFFFF) * 16)
            if len(late_examples) < 5:
                late_examples.append(describe_late(slow, rank))
        code = int(ops.rdna_ar_timeout_info(comm.handle))
        if code and wedge is None:
            wedge = describe_abort(code, rank)
        flags[0] = 1 if code else 0
        flags[1] = int(bad.item())
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        now = time.perf_counter()
        if now - last_hb > 1.0:
            out_q.put(("hb", rank, done))
            last_hb = now
        if int(flags[0]) or int(flags[1]):
            break
    elapsed = time.perf_counter() - t_start
    res.update(
        done=done,
        elapsed_s=round(elapsed, 3),
        us_per_collective=round(elapsed * 1e6 / max(done, 1), 2),
        late=late,
        max_late_ms=max_late_ms,
        late_examples=late_examples,
        wedge=wedge,
        mismatches=int(bad.item()),
        warmup_mismatches=warm_bad,
    )
    out_q.put(("result", rank, res))


def _kill(procs) -> None:
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=30)


def _attempt(args, world: int, size_kb: int, traffic: str, todo: int) -> dict:
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    stop = ctx.Event()
    marker = os.path.join(args.cache_root, _MARKER)
    if os.path.exists(marker):
        os.remove(marker)
    cfg = {
        "cache_root": args.cache_root,
        "size_kb": size_kb,
        "wait_ms": args.wait_ms,
        "graph_k": args.graph_k,
        "batch_replays": args.batch_replays,
        "collectives": todo,
        "stall_rank": args.stall_rank,
        "stall_ms": args.stall_ms,
        "stall_every": args.stall_every,
    }
    movers = []
    if traffic != "none":
        for dev in range(world):
            p = ctx.Process(
                target=_traffic_main, args=(dev, world, traffic, stop, out_q)
            )
            p.start()
            movers.append(p)
    port = _free_port()
    ranks = [
        ctx.Process(target=_rank_main, args=(r, world, port, cfg, out_q))
        for r in range(world)
    ]
    for p in ranks:
        p.start()
    results: dict[int, dict] = {}
    rates: dict[int, float] = {}
    hang = None
    last_progress = time.monotonic()
    while len(results) < world:
        try:
            item = out_q.get(timeout=5)
        except queue.Empty:
            dead = [r for r, p in enumerate(ranks) if not p.is_alive()]
            if any(r not in results for r in dead):
                hang = f"rank(s) {dead} exited without a result"
                break
            if time.monotonic() - last_progress > args.hang_s:
                hang = f"no progress for {args.hang_s} s"
                break
            continue
        kind, who, payload = item
        if kind == "hb":
            last_progress = time.monotonic()
        elif kind == "result":
            results[who] = payload
            last_progress = time.monotonic()
        elif kind == "traffic":
            rates[who] = payload
    stop.set()
    if hang is not None:
        _kill(ranks + movers)
    else:
        for p in ranks:
            p.join(timeout=120)
        deadline = time.monotonic() + 60
        while len(rates) < len(movers) and time.monotonic() < deadline:
            try:
                kind, who, payload = out_q.get(timeout=5)
            except queue.Empty:
                continue
            if kind == "traffic":
                rates[who] = payload
        _kill(ranks + movers)
    wedge_marker = None
    if os.path.exists(marker):
        with open(marker) as f:
            wedge_marker = f.read().strip()
        os.remove(marker)
    done = min((r.get("done", 0) for r in results.values()), default=0)
    return {
        "done": done,
        "hang": hang,
        "ranks": [results.get(r, {"rank": r, "missing": True}) for r in range(world)],
        "traffic_gbps": [round(rates[d], 2) for d in sorted(rates)],
        "marker": wedge_marker,
    }


def _run_cell(args, world: int, size_kb: int, traffic: str) -> dict:
    aer0 = _aer_snapshot()
    t0 = time.time()
    attempts = []
    total = 0
    wedges = 0
    while total < args.collectives and wedges < args.max_wedges:
        att = _attempt(args, world, size_kb, traffic, args.collectives - total)
        attempts.append(att)
        total += att["done"]
        ranks = att["ranks"]
        wedged = any(r.get("wedge") for r in ranks)
        wedges += int(wedged)
        if att["hang"] or any(r.get("error") for r in ranks):
            break
        if any(r.get("mismatches") for r in ranks):
            break
        if not wedged and att["done"] == 0:
            break
    ranks_all = [r for a in attempts for r in a["ranks"]]
    busy_s = sum(r.get("elapsed_s", 0.0) for r in ranks_all if r.get("rank") == 0)
    return {
        "label": args.label,
        "devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "bus": [r.get("bus") for r in attempts[0]["ranks"]] if attempts else [],
        "world": world,
        "size_kb": size_kb,
        "traffic": traffic,
        "collectives": total,
        "wedges": wedges,
        "hangs": sum(1 for a in attempts if a["hang"]),
        "mismatches": sum(r.get("mismatches", 0) for r in ranks_all),
        "late": sum(r.get("late", 0) for r in ranks_all),
        "max_late_ms": max((r.get("max_late_ms", 0) for r in ranks_all), default=0),
        "us_per_collective": round(busy_s * 1e6 / total, 2) if total else None,
        "wedge_msgs": [r["wedge"] for r in ranks_all if r.get("wedge")],
        "errors": [r["error"] for r in ranks_all if r.get("error")]
        + [a["hang"] for a in attempts if a["hang"]],
        "aer_delta": _aer_delta(aer0, _aer_snapshot()),
        "wall_s": round(time.time() - t0, 1),
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "wait_ms": args.wait_ms,
        "stall": [args.stall_rank, args.stall_ms, args.stall_every]
        if args.stall_every
        else None,
        "env": {
            k: os.environ.get(k)
            for k in ("HSA_ENABLE_SDMA", "GPU_MAX_HW_QUEUES", "VLLM_RDNA_AR_VEC")
        },
        "attempts": attempts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes-kb", default="64,128,256")
    ap.add_argument("--traffic", default="none,mix", help="none,p2p,host,mix")
    ap.add_argument("--collectives", type=int, default=2_000_000, help="per cell")
    ap.add_argument("--wait-ms", type=int, default=20000, help="0 = build default")
    ap.add_argument("--graph-k", type=int, default=16)
    ap.add_argument("--batch-replays", type=int, default=64)
    ap.add_argument("--max-wedges", type=int, default=3)
    ap.add_argument("--hang-s", type=int, default=300)
    ap.add_argument("--stall-rank", type=int, default=0)
    ap.add_argument("--stall-ms", type=int, default=0)
    ap.add_argument("--stall-every", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True, help="JSONL, one line per cell")
    ap.add_argument("--cache-root", default="", help="VLLM_CACHE_ROOT for ranks")
    args = ap.parse_args()
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not visible:
        raise SystemExit("set HIP_VISIBLE_DEVICES to the GPUs to stress")
    world = len(visible.split(","))
    if not args.cache_root:
        args.cache_root = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "vllm-cache"
        )
    os.makedirs(args.cache_root, exist_ok=True)
    sizes = [int(s) for s in args.sizes_kb.split(",")]
    modes = args.traffic.split(",")
    rows = []
    for traffic in modes:
        for size_kb in sizes:
            cell = _run_cell(args, world, size_kb, traffic)
            with open(args.out, "a") as f:
                f.write(json.dumps(cell) + "\n")
            rows.append(cell)
            print(
                f"{cell['label']} world={world} {size_kb} KiB traffic={traffic}: "
                f"{cell['collectives']} collectives, {cell['wedges']} wedges, "
                f"{cell['hangs']} hangs, {cell['mismatches']} mismatches, "
                f"{cell['late']} late (max {cell['max_late_ms']} ms), "
                f"{cell['us_per_collective']} us/coll, aer {cell['aer_delta']}"
                + (f", errors {cell['errors']}" if cell["errors"] else ""),
                flush=True,
            )


if __name__ == "__main__":
    main()
