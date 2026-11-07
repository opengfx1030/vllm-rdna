# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU probe of the rdna_ar push all-reduce on gfx1100 (2 x W7800, cross-root).

NOT TO BE RUN WITHOUT GPU APPROVAL. It needs both W7800 idle (no server, no KFD
client, the box lease and boot.lock rules of AGENTS.md) and the wedge marker
clear: rdna_ar refuses to start while <VLLM_CACHE_ROOT>/rdna_ar_wedged_gfx1100
exists. The probe uses its own cache root (--cache-root, default
<out dir>/ar_probe_cache), so a wedge here never writes the production marker;
--clear-marker deletes a stale one of that root.

    HIP_VISIBLE_DEVICES=0,5 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2 \\
      torchrun --standalone --nproc_per_node=2 tools/rdna/rdna3/ar_probe.py \\
      --out ar_probe_w7800.json --timeout-test

HIP 0 and 5 are the W7800 behind the two root ports c0 and 00
(docs/rdna2/pcie-iommu-audit-2026-11-06.md), so the pair is cross-root, the
topology of the TP2 slice. The probe builds RdnaOneShotAllReduce directly on a
gloo group (as tools/rdna/rdna_ar_stress.py does), so the group-wide agreement
of its constructor (one architecture on every rank, the "ar" family of
VLLM_RDNA3_HIP_OPS, no wedge marker) is part of what runs. It sets
VLLM_RDNA3_HIP_OPS=ar and a bounded VLLM_RDNA_AR_WAIT_MS (--wait-ms, 30 s) unless
the caller set them: a peer that never arrives ends the kernel, not the machine.

Phases (--phases, default exact,latency,soak,gather,rccl; timeout is added by
--timeout-test). Every rank runs the same collectives, and every step ends with
a gloo agreement on the abort record, so a wedge ends the run on all ranks:

  exact    rdna_ar against a fp32 reference summed in rank order and rounded to
           the dtype, bit for bit, for --sizes-kb x --dtypes x two payloads:
           small signed integers (exact in every dtype; a misplaced element
           shows) and non-zero normals (the sum rounds; no +-0 ambiguity). With
           2 ranks a+b is exact and commutative, so the order is no freedom.
           Sizes above --gate-kb run on a second instance that covers them
           (the 10 MiB case, fp16).
  latency  microseconds per collective, p50/p99/max: a CUDA graph of --graph-k
           all-reduces replayed back to back (--iters collectives, the decode
           path) and eager single calls timed with events (--eager-iters).
  soak     --soak-s seconds of graph replays whose payload changes on every
           replay (in-graph integer bump), each result checked in-graph against
           the exact sum: a stale staging slot of the same parity shows as a
           wrong sum (the pattern of tools/rdna/rdna_ar_stress.py).
  gather   the last-dim all-gather and the vocab argmax, bit-exact against
           their documented layout and tie rule.
  rccl     the same sizes through RCCL (torch.distributed nccl group): rdna_ar
           vs RCCL vs the reference (RCCL vs reference is information, not a
           gate) and RCCL eager latency as the A/B figure.
  timeout  with --timeout-test only: a second instance with a short
           VLLM_RDNA_AR_WAIT_MS (--timeout-wait-ms); rank 1 skips one collective
           on purpose, rank 0's kernel must give up at the bound and return, its
           next collective must return at once (sticky abort), and check() must
           raise "never reached" without writing a marker. The wait the record
           measured on the device, against the host clock, is the first
           measurement of the gfx1100 wall-clock rate.

Gates (exit status 0 only if every executed gate holds):
  bit_exact   zero mismatching elements in exact, soak and gather, all ranks
  p50         <= --p50-limit-us (20) per collective at --p50-size-kb (8 KiB)
              fp16, graph mode, worst rank
  no_wedge    no abort record and no wedge marker (late peers are listed only)
  timeout     record phase 2 with the announce bit clear and a device-measured
              wait in [0.8, 1.5] x the bound; rank 0's call returned within the
              bound + --timeout-slack-ms; the next call took < 200 ms; check()
              raised; no marker

--dry-run runs the same code on CPU stand-ins (gloo arithmetic for the kernel,
no CUDA graphs or events), blocks amdsmi and hides every GPU, to check the
control flow, the reference arithmetic and the verdict without a GPU:

    torchrun --standalone --nproc_per_node=2 tools/rdna/rdna3/ar_probe.py \\
      --dry-run --out ar_probe_dry.json

Raw push/pull bandwidth and flag latency without rdna_ar: tools/rdna/p2p_probe
(its build line takes --offload-arch=gfx1100 for the W7800). Read
`sudo -n dmesg` before and after the run; the probe records the AER counter
delta and the KFD eviction time itself.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import json
import math
import os
import sys
import threading
import time
import types
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

KIB = 1 << 10
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
PHASES = ("exact", "latency", "soak", "gather", "rccl", "timeout")
_ENV_PREFIXES = (
    "HIP",
    "HSA",
    "ROCR",
    "GPU_",
    "NCCL",
    "RCCL",
    "VLLM_RDNA",
    "VLLM_CACHE",
    "CUDA_VISIBLE",
    "TORCH",
)
_ROWS = 64  # rows of the all-gather / argmax checks
_COLS = 256


class Stop(Exception):
    """A phase found a reason to end the whole run (every rank raises it)."""


# --- pure helpers (unit-tested on CPU) ---------------------------------------


def percentile(ascending: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of an ascending list."""
    if not ascending:
        raise ValueError("percentile of nothing")
    k = math.ceil(q / 100.0 * len(ascending)) - 1
    return ascending[max(0, min(len(ascending) - 1, k))]


def summarize(us: list[float]) -> dict:
    s = sorted(us)
    return {
        "n": len(s),
        "min_us": round(s[0], 3),
        "p50_us": round(percentile(s, 50), 3),
        "p99_us": round(percentile(s, 99), 3),
        "p999_us": round(percentile(s, 99.9), 3),
        "max_us": round(s[-1], 3),
        "mean_us": round(sum(s) / len(s), 3),
    }


def make_payload(
    kind: str, rank: int, numel: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    """CPU payload of one rank; every rank regenerates every rank's from this.

    "int": small signed integers, exact in every dtype and in any sum of up to 8
    ranks, so a misplaced or stale element changes the result visibly.
    "rand": normals with the magnitude lifted to >= 1e-3, never zero in any
    dtype, so the rounding of the sum is exercised and +0 / -0 cannot differ.
    """
    if kind == "int":
        idx = torch.arange(numel, dtype=torch.int64)
        return (((idx * 5 + rank * 3 + seed) % 17) - 8).to(dtype)
    gen = torch.Generator().manual_seed(seed * 1_000_003 + rank)
    x = torch.randn(numel, generator=gen, dtype=torch.float32)
    return torch.copysign(x.abs().clamp_min(1e-3), x).to(dtype)


def reference_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """The all-reduce result: fp32 sum in rank order, rounded to the dtype."""
    acc = parts[0].to(torch.float32).clone()
    for p in parts[1:]:
        acc += p.to(torch.float32)
    return acc.to(parts[0].dtype)


def bit_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bit patterns differ (16- or 32-bit dtypes)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    view = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum())


def gather_block(rank: int, dtype: torch.dtype) -> torch.Tensor:
    """[_ROWS, _COLS] block of one rank for the all-gather check.

    The gather is a copy, so the values only have to differ per rank and
    position; both sides convert them to the dtype the same way.
    """
    row = torch.arange(_ROWS).unsqueeze(1)
    col = torch.arange(_COLS).unsqueeze(0)
    return ((col + row * 37 + rank * 1000) % 1500).to(dtype)


def argmax_case(rows: int, world: int, rank: int) -> tuple[list[float], list[int]]:
    """(values, global indices) of one rank; exact in fp16, bf16 and fp32.

    Every fourth row is a tie across ranks, the rest have distinct values.
    """
    vals, idx = [], []
    for row in range(rows):
        base = (row * 31 + 7) % 100
        tie = row % 4 == 0
        vals.append(float(base + (0 if tie else 3 * ((row + rank) % world))))
        idx.append(rank * 1_000_003 + row)
    return vals, idx


def argmax_expected(rows: int, world: int) -> list[int]:
    """Per row the index of the largest value; a tie goes to the lowest rank."""
    per_rank = [argmax_case(rows, world, r) for r in range(world)]
    want = []
    for row in range(rows):
        best = max(range(world), key=lambda r: (per_rank[r][0][row], -r))
        want.append(per_rank[best][1][row])
    return want


def preflight_env_problems(env: dict, world: int, dry_run: bool) -> list[str]:
    """What is wrong with the launch environment, before any GPU is touched."""
    if dry_run:
        return []
    problems = []
    hip = env.get("HIP_VISIBLE_DEVICES", "")
    listed = [d for d in hip.split(",") if d.strip()]
    if not listed:
        problems.append(
            "HIP_VISIBLE_DEVICES is not set: name the devices explicitly "
            "(HIP_VISIBLE_DEVICES=0,5 for the two W7800)"
        )
    elif len(listed) != world:
        problems.append(
            f"HIP_VISIBLE_DEVICES={hip} lists {len(listed)} devices for "
            f"--nproc_per_node={world}"
        )
    if env.get("ROCR_VISIBLE_DEVICES"):
        problems.append("unset ROCR_VISIBLE_DEVICES (HIP_VISIBLE_DEVICES selects)")
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    if cuda and cuda != hip:
        problems.append(f"CUDA_VISIBLE_DEVICES={cuda} differs from HIP={hip}")
    return problems


def verdict(per_rank: list[dict], args: argparse.Namespace) -> dict:
    """The gates from the per-rank results; a requested phase must have finished."""
    requested = set(per_rank[0].get("phases", []))
    done = {p for p in requested if all(p in r for r in per_rank)}
    bad = 0
    for r in per_rank:
        bad += sum(c["bad"] for c in r.get("exact", []))
        bad += sum(c["bad"] for c in r.get("gather", []))
        bad += int(r.get("soak", {}).get("bad", 0))
    gates: dict[str, bool] = {}
    counted = {"exact", "soak", "gather"} & requested
    if counted:
        gates["bit_exact"] = counted <= done and bad == 0
    if "latency" in requested:
        p50s = [
            c["graph"]["p50_us"]
            for r in per_rank
            for c in r.get("latency", [])
            if c["size_kb"] == args.p50_size_kb and c["dtype"] == "float16"
        ]
        # A dry run times CPU stand-ins: it checks that the data is there, not
        # the threshold.
        limit = float("inf") if args.dry_run else args.p50_limit_us
        gates["p50"] = "latency" in done and bool(p50s) and max(p50s) <= limit
    gates["no_wedge"] = not any(r.get("wedge") or r.get("marker") for r in per_rank)
    if "timeout" in requested:
        gates["timeout"] = "timeout" in done and all(
            r["timeout"].get("ok", False) for r in per_rank if "timeout" in r
        )
    return {"gates": gates, "pass": all(gates.values()), "mismatches": bad}


# --- environment, sysfs and process plumbing ---------------------------------


def env_snapshot() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PREFIXES)}


def evicted_ms() -> int:
    """KFD's running total of this process's queue eviction time; -1 if absent."""
    paths = glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/stats_*/evicted_ms")
    total = -1
    for path in paths:
        try:
            with open(path) as f:
                total = max(total, 0) + int(f.read().strip())
        except (OSError, ValueError):
            continue
    return total


def aer_snapshot() -> dict[str, dict[str, int]]:
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


def aer_delta(before: dict, after: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for bdf, counters in after.items():
        for name, val in counters.items():
            d = val - before.get(bdf, {}).get(name, 0)
            if d:
                out.setdefault(bdf, {})[name] = d
    return out


def start_watchdog(limit_s: float, state: dict, rank: int) -> None:
    """End this process if the run outlives --hard-timeout-s (a stuck sync)."""

    def run() -> None:
        t0 = time.monotonic()
        while True:
            time.sleep(5.0)
            if time.monotonic() - t0 > limit_s:
                print(
                    f"ar_probe[rank {rank}]: watchdog: still in phase "
                    f"{state['phase']!r} after {limit_s:.0f} s; exiting",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(3)

    threading.Thread(target=run, name="ar_probe_watchdog", daemon=True).start()


# --- GPU services and the CPU stand-ins of --dry-run -------------------------


class Gpu:
    """CUDA graphs, events and sync; CPU stand-ins in a dry run."""

    def __init__(self, local_rank: int, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.device = (
            torch.device("cpu") if dry_run else torch.device("cuda", local_rank)
        )

    def sync(self) -> None:
        if not self.dry_run:
            torch.accelerator.synchronize()

    def capture(self, body):
        """A callable that replays body (a CUDA graph; body itself in a dry run).

        Two eager passes first (lazy code-object load, allocator), on a side
        stream as the capture rules want; every rank runs the same sequence.
        """
        if self.dry_run:
            return body
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
        return graph.replay

    def time_calls(self, replay, per_replay: int, reps: int) -> list[float]:
        """Microseconds per collective of reps back-to-back calls of replay."""
        if self.dry_run:
            out = []
            for _ in range(reps):
                t0 = time.perf_counter()
                replay()
                out.append((time.perf_counter() - t0) * 1e6 / per_replay)
            return out
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            replay()
            stops[i].record()
        torch.accelerator.synchronize()
        return [a.elapsed_time(b) * 1e3 / per_replay for a, b in zip(starts, stops)]


class DryComm:
    """RdnaOneShotAllReduce stand-in on CPU: the same API through gloo."""

    disabled = False
    handle = 0
    arch_class = "dry-run"

    def __init__(self, group, world: int) -> None:
        self.group = group
        self.world = world
        self._ops = types.SimpleNamespace(
            rdna_ar_timeout_info=lambda h: 0, rdna_ar_slow_info=lambda h: 0
        )

    def _gather(self, x: torch.Tensor) -> list[torch.Tensor]:
        raw = x.contiguous().view(-1).view(torch.uint8)
        got = [torch.empty_like(raw) for _ in range(self.world)]
        dist.all_gather(got, raw, group=self.group)
        return [g.view(x.dtype).reshape(x.shape) for g in got]

    def should_use(self, x) -> bool:
        return True

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return reference_sum(self._gather(x))

    def should_all_gather(self, x) -> bool:
        return True

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(self._gather(x), dim=-1)

    def should_argmax(self, vals, idx) -> bool:
        return True

    def argmax(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        v = torch.stack(self._gather(vals)).to(torch.float32)
        i = torch.stack(self._gather(idx))
        winner = v.argmax(dim=0)  # first maximum: the lowest rank
        return i.gather(0, winner.unsqueeze(0)).squeeze(0)


def chain(comm, x: torch.Tensor, k: int):
    """A body of k back-to-back all-reduces of x (what a graph captures)."""

    def body() -> None:
        for _ in range(k):
            comm.all_reduce(x)

    return body


@dataclass
class Ctx:
    args: argparse.Namespace
    rank: int
    world: int
    gpu: Gpu
    gloo: object
    rdna_ar: object  # vllm.distributed.device_communicators.rdna_all_reduce
    state: dict
    wedge: str | None = None
    late: list[str] = field(default_factory=list)
    comms: dict = field(default_factory=dict)
    _last_slow: dict = field(default_factory=dict)

    def say(self, msg: str) -> None:
        if self.rank == 0:
            print(f"ar_probe: {msg}", flush=True)

    def barrier(self) -> None:
        dist.barrier(group=self.gloo)

    def agree_any(self, flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.gloo)
        return bool(t.item())

    def poll_records(self, comm) -> bool:
        """Read comm's abort and late-peer records (host words, no sync)."""
        ops = comm._ops
        code = int(ops.rdna_ar_timeout_info(comm.handle))
        if code and self.wedge is None:
            self.wedge = self.rdna_ar.describe_abort(code, self.rank)
        slow = int(ops.rdna_ar_slow_info(comm.handle))
        fresh = slow != self._last_slow.get(comm.handle, 0)
        if slow and not slow & self.rdna_ar._REC_LIVE and fresh:
            self._last_slow[comm.handle] = slow
            self.late.append(self.rdna_ar.describe_late(slow, self.rank))
        return self.wedge is not None

    def checkpoint(self, comm, what: str) -> None:
        """End of a step: a wedge on any rank ends the run on every rank."""
        if self.agree_any(self.poll_records(comm)):
            raise Stop(f"wedge after {what}: {self.wedge or 'on a peer rank'}")

    def comm_for(self, size_kb: int):
        """The instance whose gate covers size_kb (built on first use)."""
        gate = self.args.gate_kb if size_kb <= self.args.gate_kb else size_kb
        if gate not in self.comms:
            self.comms[gate] = self.make_comm(gate)
        return self.comms[gate]

    def make_comm(self, gate_kb: int, wait_ms: int | None = None):
        self.state["phase"] = f"construct gate={gate_kb}KiB"
        if self.args.dry_run:
            return DryComm(self.gloo, self.world)
        env = {
            "VLLM_RDNA_AR_MAX_KB": str(gate_kb),
            "VLLM_RDNA_AR_ONESHOT_KB": str(gate_kb),
        }
        if wait_ms is not None:
            env["VLLM_RDNA_AR_WAIT_MS"] = str(wait_ms)
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            t0 = time.perf_counter()
            comm = self.rdna_ar.RdnaOneShotAllReduce(self.gloo, self.gpu.device)
            self.say(
                f"instance gate {gate_kb} KiB: disabled={comm.disabled} "
                f"arch={comm.arch_class} built in {time.perf_counter() - t0:.2f} s"
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if self.agree_any(comm.disabled):
            raise Stop(
                f"rdna_ar is disabled (gate {gate_kb} KiB): see the rdna_ar log "
                "lines above (group agreement, wedge marker, init, self-test)"
            )
        return comm


# --- phases ------------------------------------------------------------------


def _numel(size_kb: int, dtype: torch.dtype) -> int:
    return size_kb * KIB // torch.empty(0, dtype=dtype).element_size()


def _dtypes_for(ctx: Ctx, size_kb: int) -> list[str]:
    # The big size is fp16 only (the slice's dtype); the rest sweep --dtypes.
    return ["float16"] if size_kb > ctx.args.gate_kb else ctx.args.dtypes


def phase_exact(ctx: Ctx) -> list[dict]:
    ctx.state["phase"] = "exact"
    rows: list[dict] = []
    for size_kb in ctx.args.sizes_kb:
        comm = ctx.comm_for(size_kb)
        reps = ctx.args.exact_reps if size_kb <= ctx.args.gate_kb else 1
        for name in _dtypes_for(ctx, size_kb):
            dtype = DTYPES[name]
            numel = _numel(size_kb, dtype)
            for kind in ("int", "rand"):
                bad, eligible = 0, True
                for rep in range(reps):
                    parts = [
                        make_payload(kind, r, numel, dtype, ctx.args.seed + rep)
                        for r in range(ctx.world)
                    ]
                    x = parts[ctx.rank].to(ctx.gpu.device)
                    if not comm.should_use(x):
                        eligible = False
                        break
                    out = comm.all_reduce(x)
                    ctx.gpu.sync()
                    bad += bit_mismatches(out.cpu(), reference_sum(parts))
                rows.append(
                    {
                        "size_kb": size_kb,
                        "dtype": name,
                        "kind": kind,
                        "reps": reps,
                        "eligible": eligible,
                        "bad": bad if eligible else numel,
                    }
                )
                state = "bit-exact" if eligible and not bad else "MISMATCH"
                ctx.say(
                    f"exact {size_kb:>6} KiB {name:<8} {kind:<4} {state}"
                    f"{'' if eligible else ' (not eligible)'}"
                    f"{f' {bad} elements' if bad else ''}"
                )
                ctx.checkpoint(comm, f"exact {size_kb} KiB {name} {kind}")
    return rows


def phase_latency(ctx: Ctx) -> list[dict]:
    ctx.state["phase"] = "latency"
    rows: list[dict] = []
    for size_kb in ctx.args.sizes_kb:
        comm = ctx.comm_for(size_kb)
        small = size_kb <= ctx.args.gate_kb
        for name in _dtypes_for(ctx, size_kb):
            if name != "float16" and size_kb != ctx.args.p50_size_kb:
                continue  # the other dtypes' latency at the gate size only
            dtype = DTYPES[name]
            x = make_payload("int", ctx.rank, _numel(size_kb, dtype), dtype, 1)
            x = x.to(ctx.gpu.device)
            k = ctx.args.graph_k if small else 4
            iters = ctx.args.iters if small else ctx.args.big_iters
            replay = ctx.gpu.capture(chain(comm, x, k))
            ctx.barrier()
            for _ in range(20):
                replay()
            ctx.gpu.sync()
            ctx.barrier()
            graph = summarize(ctx.gpu.time_calls(replay, k, max(iters // k, 50)))
            ctx.checkpoint(comm, f"latency graph {size_kb} KiB {name}")
            one = chain(comm, x, 1)
            n = ctx.args.eager_iters if small else min(ctx.args.eager_iters, 200)
            ctx.barrier()
            for _ in range(20):
                one()
            ctx.gpu.sync()
            ctx.barrier()
            eager = summarize(ctx.gpu.time_calls(one, 1, n))
            ctx.checkpoint(comm, f"latency eager {size_kb} KiB {name}")
            rows.append(
                {
                    "size_kb": size_kb,
                    "dtype": name,
                    "graph_k": k,
                    "graph": graph,
                    "eager": eager,
                }
            )
            ctx.say(
                f"latency {size_kb:>6} KiB {name:<8} graph p50 "
                f"{graph['p50_us']:.1f} p99 {graph['p99_us']:.1f} max "
                f"{graph['max_us']:.1f} us/collective; eager p50 "
                f"{eager['p50_us']:.1f} p99 {eager['p99_us']:.1f} us"
            )
    return rows


def phase_soak(ctx: Ctx) -> dict:
    """Rotating-pattern soak: in-graph bump and check, as rdna_ar_stress.py."""
    ctx.state["phase"] = "soak"
    a = ctx.args
    comm = ctx.comm_for(a.soak_kb)
    dev, world, rank, k = ctx.gpu.device, ctx.world, ctx.rank, a.graph_k
    n = a.soak_kb * KIB // 2  # fp16 elements
    # Integers below 2048 / world: every partial and full sum is an exact fp16
    # integer, so the check holds for any summation order.
    mask = 2048 // world - 1
    idx = torch.arange(n, dtype=torch.int32, device=dev)

    def pattern(r: int, kk: int) -> torch.Tensor:
        return ((idx + r * 131 + kk * 17) & mask).to(torch.int16)

    cnt = [pattern(rank, kk) for kk in range(k)]
    mirror = [torch.stack([pattern(r, kk) for r in range(world)]) for kk in range(k)]
    inp = [c.to(torch.float16) for c in cnt]
    bad = torch.zeros(1, dtype=torch.int64, device=dev)

    def body() -> None:
        for kk in range(k):
            cnt[kk].add_(1).bitwise_and_(mask)
            mirror[kk].add_(1).bitwise_and_(mask)
            inp[kk].copy_(cnt[kk])
            out = comm.all_reduce(inp[kk])
            want = mirror[kk].sum(0, dtype=torch.int32)
            bad.add_((out.to(torch.int32) != want).sum())

    replay = ctx.gpu.capture(body)
    ctx.gpu.sync()
    warm_bad = int(bad.item())  # the two eager warm-up passes of capture
    bad.zero_()
    deadline = time.monotonic() + a.soak_s
    done = 0
    t_start = time.perf_counter()
    while True:
        for _ in range(a.soak_batch):
            replay()
        ctx.gpu.sync()
        done += a.soak_batch * k
        stop = (
            ctx.poll_records(comm)
            or int(bad.item()) != 0
            or time.monotonic() > deadline
        )
        if ctx.agree_any(stop):
            break
    elapsed = time.perf_counter() - t_start
    total_bad = warm_bad + int(bad.item())
    res = {
        "size_kb": a.soak_kb,
        "collectives": done,
        "elapsed_s": round(elapsed, 2),
        "us_per_collective": round(elapsed * 1e6 / max(done, 1), 2),
        "bad": total_bad,
    }
    ctx.say(
        f"soak {a.soak_kb} KiB: {done} collectives in {elapsed:.0f} s, "
        f"{total_bad} wrong sums"
    )
    ctx.checkpoint(comm, "soak")
    return res


def phase_gather(ctx: Ctx) -> list[dict]:
    ctx.state["phase"] = "gather"
    comm = ctx.comm_for(ctx.args.p50_size_kb)
    dev, world = ctx.gpu.device, ctx.world
    rows: list[dict] = []

    def record(op: str, eligible: bool, bad: int) -> None:
        rows.append({"op": op, "eligible": eligible, "bad": bad if eligible else 1})
        ctx.say(f"gather {op}: {'exact' if eligible and not bad else 'FAIL'}")
        ctx.checkpoint(comm, op)

    for name in ("float16", "bfloat16", "float32"):
        dtype = DTYPES[name]
        x = gather_block(ctx.rank, dtype).to(dev)
        eligible, bad = comm.should_all_gather(x), 0
        if eligible:
            out = comm.all_gather(x)
            ctx.gpu.sync()
            want = torch.cat([gather_block(r, dtype) for r in range(world)], dim=1)
            bad = bit_mismatches(out.cpu(), want)
        record(f"all_gather {name}", eligible, bad)
    want = torch.tensor(argmax_expected(_ROWS, world), dtype=torch.int64)
    for name in ("float16", "bfloat16", "float32"):
        vals, idx = argmax_case(_ROWS, world, ctx.rank)
        v = torch.tensor(vals, dtype=DTYPES[name], device=dev)
        i = torch.tensor(idx, dtype=torch.int64, device=dev)
        eligible, bad = comm.should_argmax(v, i), 0
        if eligible:
            out = comm.argmax(v, i)
            ctx.gpu.sync()
            bad = int((out.cpu() != want).sum())
        record(f"argmax {name}", eligible, bad)
    return rows


def phase_rccl(ctx: Ctx) -> list[dict]:
    ctx.state["phase"] = "rccl"
    pg = None if ctx.args.dry_run else dist.new_group(backend="nccl")
    rows: list[dict] = []
    for size_kb in ctx.args.sizes_kb:
        comm = ctx.comm_for(size_kb)
        dtype = torch.float16
        numel = _numel(size_kb, dtype)
        row: dict = {"size_kb": size_kb}
        for kind in ("int", "rand"):
            parts = [make_payload(kind, r, numel, dtype, 1) for r in range(ctx.world)]
            ref = reference_sum(parts)
            x = parts[ctx.rank].to(ctx.gpu.device)
            ours = comm.all_reduce(x)
            theirs = reference_sum(parts) if pg is None else x.clone()
            if pg is not None:
                dist.all_reduce(theirs, group=pg)
            ctx.gpu.sync()
            row[kind] = {
                "ar_vs_ref": bit_mismatches(ours.cpu(), ref),
                "rccl_vs_ref": bit_mismatches(theirs.cpu(), ref),
                "ar_vs_rccl": bit_mismatches(ours.cpu(), theirs.cpu()),
            }
        if pg is not None and ctx.args.rccl_iters > 0:
            t = torch.zeros(numel, dtype=dtype, device=ctx.gpu.device)  # sums stay 0
            n = ctx.args.rccl_iters if size_kb <= ctx.args.gate_kb else 100
            ctx.barrier()
            for _ in range(20):
                dist.all_reduce(t, group=pg)
            ctx.gpu.sync()
            ctx.barrier()
            row["rccl_eager"] = summarize(
                ctx.gpu.time_calls(lambda t=t: dist.all_reduce(t, group=pg), 1, n)
            )
        rows.append(row)
        p50 = row.get("rccl_eager", {}).get("p50_us")
        ctx.say(
            f"rccl {size_kb:>6} KiB: ar-vs-rccl mismatches int "
            f"{row['int']['ar_vs_rccl']} rand {row['rand']['ar_vs_rccl']}; "
            f"rccl-vs-reference rand {row['rand']['rccl_vs_ref']}; rccl eager p50 "
            f"{'n/a' if p50 is None else f'{p50:.1f} us'}"
        )
        ctx.checkpoint(comm, f"rccl {size_kb} KiB")
    return rows


def phase_timeout(ctx: Ctx) -> dict:
    """Peer skip: rank 1 does not call the collective rank 0 waits in."""
    ctx.state["phase"] = "timeout"
    a = ctx.args
    comm = ctx.make_comm(a.p50_size_kb, wait_ms=a.timeout_wait_ms)
    dtype = torch.float16
    x = make_payload("int", ctx.rank, _numel(a.p50_size_kb, dtype), dtype, 1)
    x = x.to(ctx.gpu.device)
    comm.all_reduce(x)  # collective #1 on both ranks: the instance works
    ctx.gpu.sync()
    ctx.barrier()
    res: dict = {"wait_ms": a.timeout_wait_ms, "ok": True}
    if ctx.rank == 0:
        mod = ctx.rdna_ar
        t0 = time.perf_counter()
        comm.all_reduce(x)  # collective #2: rank 1 never calls it
        ctx.gpu.sync()
        first_ms = (time.perf_counter() - t0) * 1e3
        code = int(comm._ops.rdna_ar_timeout_info(comm.handle))
        phase, peer, device_ms, seq = mod._fields(code)
        seen = bool(code & mod._REC_SEEN)
        t1 = time.perf_counter()
        again = comm.all_reduce(x)  # sticky abort: must not wait again
        ctx.gpu.sync()
        second_ms = (time.perf_counter() - t1) * 1e3
        marker = mod.marker_path(comm.arch_class)
        had_marker = os.path.exists(marker)
        raised = ""
        try:
            comm.check()
        except RuntimeError as e:
            raised = str(e)
        res.update(
            record=hex(code),
            phase=phase,
            peer=peer,
            seq=seq,
            announce_seen=seen,
            device_ms=device_ms,
            first_call_ms=round(first_ms, 1),
            second_call_ms=round(second_ms, 1),
            sticky_output_is_local_input=bool(torch.equal(again.cpu(), x.cpu())),
            check_raised=raised[:300],
            marker_written=os.path.exists(marker) and not had_marker,
        )
        res["ok"] = bool(
            code
            and phase == 2
            and not seen
            and 0.8 * a.timeout_wait_ms <= device_ms <= 1.5 * a.timeout_wait_ms
            and first_ms <= a.timeout_wait_ms + a.timeout_slack_ms
            and second_ms < 200
            and "never reached" in raised
            and not res["marker_written"]
        )
        ctx.say(f"timeout: {json.dumps(res)}")
    else:
        time.sleep((a.timeout_wait_ms + a.timeout_slack_ms) / 1e3 + 1.0)
    ctx.barrier()
    box = [res]
    dist.broadcast_object_list(box, src=0, group=ctx.gloo)
    return box[0]


# --- run ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Needs both W7800 idle and the wedge marker clear; see the header.",
    )
    p.add_argument("--out", default="ar_probe.json", help="JSON result path")
    p.add_argument("--sizes-kb", default="8,64,120,256,10240")
    p.add_argument("--dtypes", default="float16,bfloat16,float32")
    p.add_argument("--gate-kb", type=int, default=256, help="gate of the main instance")
    p.add_argument(
        "--phases",
        default="exact,latency,soak,gather,rccl",
        help="comma list of " + ",".join(PHASES),
    )
    p.add_argument("--exact-reps", type=int, default=3)
    p.add_argument("--iters", type=int, default=100_000, help="collectives per size")
    p.add_argument("--big-iters", type=int, default=1_000, help="above --gate-kb")
    p.add_argument("--eager-iters", type=int, default=20_000)
    p.add_argument("--rccl-iters", type=int, default=2_000, help="0: no RCCL timing")
    p.add_argument("--graph-k", type=int, default=32, help="collectives per graph")
    p.add_argument("--soak-s", type=float, default=60.0)
    p.add_argument("--soak-kb", type=int, default=8)
    p.add_argument("--soak-batch", type=int, default=50, help="replays per check")
    p.add_argument("--p50-limit-us", type=float, default=20.0)
    p.add_argument("--p50-size-kb", type=int, default=8)
    p.add_argument("--wait-ms", type=int, default=30_000)
    p.add_argument("--timeout-test", action="store_true", help="add the timeout phase")
    p.add_argument("--timeout-wait-ms", type=int, default=1_500)
    p.add_argument("--timeout-slack-ms", type=int, default=2_000)
    p.add_argument("--cache-root", default=None, help="isolated VLLM_CACHE_ROOT")
    p.add_argument("--clear-marker", action="store_true")
    p.add_argument("--expect-arch", default="gfx1100", help="gcnArchName or 'any'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hard-timeout-s", type=float, default=1200.0)
    p.add_argument("--dry-run", action="store_true", help="CPU stand-ins, no GPU")
    a = p.parse_args(argv)
    a.sizes_kb = [int(s) for s in a.sizes_kb.split(",") if s.strip()]
    a.dtypes = [d.strip() for d in a.dtypes.split(",") if d.strip()]
    a.phases = [s.strip() for s in a.phases.split(",") if s.strip()]
    if a.timeout_test and "timeout" not in a.phases:
        a.phases.append("timeout")
    unknown = (set(a.phases) - set(PHASES)) | (set(a.dtypes) - set(DTYPES))
    if unknown:
        p.error(f"unknown phases/dtypes: {sorted(unknown)}")
    if a.p50_size_kb > a.gate_kb:
        p.error("--p50-size-kb must not exceed --gate-kb")
    if "latency" in a.phases and a.p50_size_kb not in a.sizes_kb:
        p.error("--p50-size-kb must be one of --sizes-kb (the p50 gate measures it)")
    if a.dry_run and "timeout" in a.phases:
        p.error("the timeout phase needs the kernel: not available with --dry-run")
    return a


def hide_gpus() -> None:
    """--dry-run: nothing may enumerate or wake a GPU (amdsmi lists them all)."""
    sys.modules["amdsmi"] = None  # type: ignore[assignment]
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def preflight(ctx: Ctx, res: dict) -> None:
    """Checks that end the run before the first collective of rdna_ar."""
    a = ctx.args
    mod = ctx.rdna_ar
    problems: list[str] = []
    if not a.dry_run:
        idx = ctx.gpu.device.index
        props = torch.cuda.get_device_properties(idx)
        arch = props.gcnArchName.split(":")[0]
        res["device"] = {
            "index": idx,
            "name": props.name,
            "arch": props.gcnArchName,
            "pci_bus_id": hex(props.pci_bus_id),
            "mem_gib": round(props.total_memory / (1 << 30), 1),
        }
        if a.expect_arch != "any" and arch != a.expect_arch:
            problems.append(f"device {idx} is {arch}, not {a.expect_arch}")
        peers = {
            j: bool(torch.cuda.can_device_access_peer(idx, j))
            for j in range(ctx.world)
            if j != idx
        }
        res["can_access_peer"] = peers
        if not all(peers.values()):
            problems.append(f"no peer access from device {idx}: {peers}")
        marker = mod.marker_path(mod.arch_class_of(arch))
        res["marker_path"] = marker
        if os.path.exists(marker):
            if a.clear_marker:
                os.remove(marker)
            else:
                problems.append(f"wedge marker {marker} exists (--clear-marker)")
    if ctx.agree_any(bool(problems)):
        raise Stop("preflight: " + "; ".join(problems or ["a peer rank"]))


def run(ctx: Ctx, res: dict) -> None:
    a = ctx.args
    cache_root = a.cache_root or os.path.join(
        os.path.dirname(os.path.abspath(a.out)), "ar_probe_cache"
    )
    os.environ["VLLM_CACHE_ROOT"] = cache_root
    os.environ.setdefault("VLLM_RDNA3_HIP_OPS", "ar")
    os.environ.setdefault("VLLM_RDNA_AR_WAIT_MS", str(a.wait_ms))
    res["env"] = env_snapshot()
    problems = preflight_env_problems(dict(os.environ), ctx.world, a.dry_run)
    if ctx.agree_any(bool(problems)):
        raise Stop("launch environment: " + "; ".join(problems or ["a peer rank"]))
    if not a.dry_run:
        # Before vllm.platforms is imported, so every rank's platform probe
        # refers to its own GPU rather than the first visible one.
        torch.accelerator.set_device_index(ctx.gpu.device.index)
    from vllm.distributed.device_communicators import rdna_all_reduce

    ctx.rdna_ar = rdna_all_reduce
    preflight(ctx, res)
    ctx.say(
        f"{ctx.world} ranks, cache root {cache_root}, phases {a.phases}, sizes "
        f"{a.sizes_kb} KiB{' (DRY RUN: CPU stand-ins)' if a.dry_run else ''}"
    )
    aer0, evict0 = aer_snapshot(), evicted_ms()
    res["phases"] = list(a.phases)
    ctx.comm_for(a.p50_size_kb)  # the main instance, before the first phase
    runners = {
        "exact": phase_exact,
        "latency": phase_latency,
        "soak": phase_soak,
        "gather": phase_gather,
        "rccl": phase_rccl,
        "timeout": phase_timeout,
    }
    for name in PHASES:
        if name in a.phases:
            res[name] = runners[name](ctx)
    res["evicted_ms"] = evicted_ms() - evict0 if evict0 >= 0 else None
    if ctx.rank == 0:
        res["aer_delta"] = aer_delta(aer0, aer_snapshot())


def report(gathered: list[dict], args: argparse.Namespace, error: str | None) -> bool:
    """Rank 0: write the JSON, print the gates; True when the run passed."""
    out: dict = {"args": vars(args), "ranks": gathered}
    errors = [g["error"] for g in gathered if g.get("error")]
    started = all("phases" in g for g in gathered)
    out["verdict"] = verdict(gathered, args) if started else None
    out["errors"] = errors
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"ar_probe: wrote {args.out}", flush=True)
    for g in gathered:
        for late in g.get("late_peers", []):
            print(f"ar_probe: WARN late peer: {late}", flush=True)
    for err in errors:
        print(f"ar_probe: ERROR {err}", flush=True)
    if out["verdict"] is None:
        print("ar_probe: FAIL: stopped before the phases", flush=True)
        return False
    for gate, held in out["verdict"]["gates"].items():
        print(f"ar_probe: {'PASS' if held else 'FAIL'} {gate}", flush=True)
    if args.dry_run:
        print("ar_probe: dry run: plumbing only, not a measurement", flush=True)
    return out["verdict"]["pass"] and not errors


def main() -> None:
    args = parse_args()
    if args.dry_run:
        hide_gpus()
    try:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    except KeyError:
        sys.exit("ar_probe: launch it with torchrun (RANK / WORLD_SIZE unset); see -h")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    state = {"phase": "start"}
    start_watchdog(args.hard_timeout_s, state, rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=10))
    ctx = Ctx(
        args=args,
        rank=rank,
        world=world,
        gpu=Gpu(local_rank, args.dry_run),
        gloo=dist.group.WORLD,
        rdna_ar=None,
        state=state,
    )
    res: dict = {"rank": rank, "dry_run": args.dry_run}
    error = None
    try:
        run(ctx, res)
    except Stop as e:
        error = str(e)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    res.update(wedge=ctx.wedge, late_peers=ctx.late, error=error)
    if ctx.rdna_ar is not None and ctx.comms:
        comm = next(iter(ctx.comms.values()))
        res["marker"] = os.path.exists(ctx.rdna_ar.marker_path(comm.arch_class))
    gathered: list = [None] * world
    dist.all_gather_object(gathered, res, group=ctx.gloo)
    box = [report(gathered, args, error) if rank == 0 else False]
    dist.broadcast_object_list(box, src=0, group=ctx.gloo)
    with contextlib.suppress(Exception):
        dist.destroy_process_group()
    sys.exit(0 if box[0] else 1)


if __name__ == "__main__":
    main()
