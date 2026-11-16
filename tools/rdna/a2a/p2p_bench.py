#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe P2P and all-to-all microbenchmark for the RDNA2 EP/DBO feasibility study.

Target: six Radeon PRO V620 (gfx1030) under one Broadcom PEX880xx switch tree
(root port c0:03.1 -> switch c5:00.0 -> two child switches; HIP 0,1 under one,
HIP 2-5 under the other). Every number of the old two-root topology
(docs/rdna2/pcie-iommu-audit-2026-11-06.md) has to be re-measured. This tool
records the sysfs link width and speed next to each measurement and never
assumes x8 or x16.

What it measures
  * per-pair copy bandwidth (4 KiB .. 64 MiB) and completion latency
  * 1..4 concurrent flows inside a child switch (intra_*), across child
    switches (cross*), fan-in and fan-out, with a COLLAPSE verdict when the
    aggregate drops by more than 25 % as a flow is added (the old cross-root
    signature: 25.2, 27.6, 18.7, 17.0 GB/s for 1..4 pairs)
  * flat all-to-all among 4 and 6 GPUs plus its intra-island and cross-island
    parts (the two phases of an island-aware all-to-all)

Backends
  torch  Tensor.copy_ between peer devices; no new kernels. Torch runs a
         cross-device copy on the SOURCE device's current stream (hipMemcpyAsync,
         SDMA or blit kernel per HSA_ENABLE_SDMA) between a two-way event
         barrier. This is the write direction of a synchronised copy: small
         sizes are launch and barrier bound, and torch cannot issue peer loads
         or flag stores. Flows that share a destination are lock-stepped by
         that barrier, and a small-chunk all-to-all is bound by the host
         enqueue loop, not by the link.
  probe  drives the kernel-level tools/rdna/p2p_probe/p2p_probe in batch mode:
         push stores, pull loads, sdma, flag ping-pong and the one-shot
         all-to-all step that rdna_ar uses. Build it on the box:
           hipcc -x hip -O3 --offload-arch=gfx1030 -o p2p_probe \\
             tools/rdna/p2p_probe/p2p_probe.cpp -pthread

CPU only (any machine, no torch needed)
    python3 tools/rdna/a2a/p2p_bench.py selftest
    python3 tools/rdna/a2a/p2p_bench.py plan --islands "0,1;2,3,4,5"

GPU (only inside a lease window; AGENTS.md quiet protocol)
    HIP_VISIBLE_DEVICES=2,3,4,5 GPU_MAX_HW_QUEUES=6 HSA_ENABLE_SDMA=0 \\
      python3 tools/rdna/a2a/p2p_bench.py all --owner <Agent> --out b4.json
    python3 tools/rdna/a2a/p2p_bench.py all --owner <Agent> --dry-run

Sub-commands: selftest plan topo pair flows a2a probe all. The GPU commands
refuse to run unless <owner>.<slot> lease files cover every visible GPU bus,
no docker build runs, and they touch ~/w4a8_runs/lanes/quiet/<owner> for the
duration (at most --max-minutes). HSA_ENABLE_SDMA, GPU_MAX_HW_QUEUES and
HIP_VISIBLE_DEVICES are recorded in the JSON; run the suite once per
HSA_ENABLE_SDMA value, and with GPU_MAX_HW_QUEUES at least the largest number
of flows one GPU sends (5 for a flat 6-GPU all-to-all) or flows share a queue.
"""

import argparse
import contextlib
import importlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Self

SCHEMA = "p2p_bench/1"
SYSFS = "/sys/bus/pci/devices"
LEASE_DIR = "~/w4a8_runs/lanes/lease"
QUIET_DIR = "~/w4a8_runs/lanes/quiet"
GPU_CLASSES = ("0x0300", "0x0302", "0x0380")  # VGA, 3D, display controller
# Interim layout (lease README, 2026-11-16): slot -> PCI buses of its GPUs.
SLOT_BUSES: dict[str, tuple[str, ...]] = {
    "a2": ("d1", "d4"),
    "b4": ("d9", "dc", "df", "e2"),
}
KIB = 1 << 10
MIB = 1 << 20
PAIR_SIZES = (4 * KIB, 16 * KIB, 64 * KIB, 256 * KIB, MIB, 4 * MIB, 16 * MIB, 64 * MIB)
LAT_SIZES = (4 * KIB, 16 * KIB, 64 * KIB, 256 * KIB)
FLOW_SIZES = (MIB, 4 * MIB, 16 * MIB)
A2A_CHUNKS = (4 * KIB, 16 * KIB, 64 * KIB, 256 * KIB, MIB)
PROBE_PAYLOADS = (4 * KIB, 8 * KIB, 16 * KIB, 32 * KIB, 64 * KIB)  # audit sizes
OLD_CROSS_ROOT_GBPS = (25.2, 27.6, 18.7, 17.0)  # 1..4 cross-root pairs (audit)
MAX_LADDER = 4
DROP_FRACTION = 0.75  # COLLAPSE: aggregate below 75 % of the best so far
SATURATED_EFFICIENCY = 0.6
# Estimated seconds per cell, only used by `plan` (rough, GPU speed dependent).
EST_S = {"pair": 0.35, "lat": 0.25, "flows": 0.5, "a2a": 0.6, "probe": 1.5}

Flow = tuple[int, int]  # (source, destination) logical GPU index


# --------------------------------------------------------------------------- #
# Pure helpers (CPU only)
# --------------------------------------------------------------------------- #
def parse_size(text: str) -> int:
    """Parse 4k, 64MiB, 1m, 4096 into bytes."""
    t = text.strip().lower().replace("ib", "").replace("b", "")
    mult = 1
    if t and t[-1] in "kmg":
        mult = {"k": KIB, "m": MIB, "g": 1 << 30}[t[-1]]
        t = t[:-1]
    return int(float(t) * mult)


def fmt_size(n: int) -> str:
    """Format bytes as KiB or MiB when exact."""
    for unit, div in (("MiB", MIB), ("KiB", KIB)):
        if n >= div and n % div == 0:
            return f"{n // div} {unit}"
    return f"{n} B"


def pct(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sample (as p2p_probe)."""
    if not sorted_vals:
        return float("nan")
    rank = max(1, math.ceil(q / 100.0 * len(sorted_vals)))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def summarize(samples: Sequence[float]) -> dict[str, float]:
    """Min, p50, p99, max and mean of a sample."""
    s = sorted(samples)
    return {
        "n": float(len(s)),
        "min": s[0],
        "p50": pct(s, 50),
        "p99": pct(s, 99),
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def gbps(nbytes: float, usec: float) -> float:
    """GB/s (1e9 bytes) for nbytes moved in usec microseconds."""
    return nbytes / usec / 1e3 if usec > 0 else 0.0


def copies_per_rep(size: int) -> int:
    """Back-to-back copies per timed repetition: about 30 ms of traffic."""
    est_s = size / 20e9 + 20e-6
    return max(4, min(256, int(0.03 / est_s)))


def link_raw_gbs(speed: str, width: str) -> float:
    """Raw per-direction GB/s of a PCIe link from its sysfs strings."""
    try:
        gts = float(speed.split()[0])
        lanes = int(width)
    except (ValueError, IndexError):
        return 0.0
    enc = 128 / 130 if gts >= 8.0 else 8 / 10
    return gts * enc * lanes / 8


def link_hint(peak_gbps: float) -> str:
    """Link class suggested by a measured single-flow peak (a hint only)."""
    if peak_gbps <= 0:
        return "unknown"
    if peak_gbps <= 18.0:
        return "x8-class (about 14 GB/s practical)"
    return "x16-class (23-26 GB/s practical)"


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def is_bdf(name: str) -> bool:
    """True for a PCI address such as 0000:d9:00.0."""
    return len(name) == 12 and name[4] == ":" and name[7] == ":" and name[10] == "."


def bdf_chain(realpath: str) -> list[str]:
    """Bridges above an endpoint from the root port down; endpoint excluded."""
    parts = [p for p in realpath.split("/") if is_bdf(p)]
    return parts[:-1]


def island_keys(chains: dict[str, list[str]]) -> dict[str, str]:
    """Island of each GPU: first ancestor below the one all GPUs share.

    Old two-root box: the root port. Single switch tree: the child switch.
    """
    if not chains:
        return {}
    lists = list(chains.values())
    shortest = min(len(c) for c in lists)
    lcp = 0
    while lcp < shortest and all(c[lcp] == lists[0][lcp] for c in lists):
        lcp += 1
    out: dict[str, str] = {}
    for bdf, chain in chains.items():
        if lcp < len(chain):
            out[bdf] = chain[lcp]
        else:
            out[bdf] = chain[-1] if chain else "root"
    return out


def scan_gpus(sysfs: str = SYSFS) -> list[str]:
    """AMD GPU functions (vendor 0x1002, display class) found in sysfs."""
    try:
        names = sorted(os.listdir(sysfs))
    except OSError:
        return []
    out = []
    for name in names:
        base = os.path.join(sysfs, name)
        if _read(os.path.join(base, "vendor")) != "0x1002":
            continue
        if _read(os.path.join(base, "class"))[:6] in GPU_CLASSES:
            out.append(name)
    return out


def link_info(bdf: str, sysfs: str = SYSFS) -> dict[str, str]:
    """Current and maximum link speed and width of one PCI function."""
    base = os.path.join(sysfs, bdf)
    keys = (
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
    )
    return {k: _read(os.path.join(base, k)) for k in keys}


def narrowest_link(path: Sequence[str], sysfs: str = SYSFS) -> dict[str, Any]:
    """Slowest link on a root-to-GPU path; the on-card bridge hides the slot link."""
    best: dict[str, Any] = {}
    for bdf in path:
        link = link_info(bdf, sysfs)
        raw = link_raw_gbs(link["current_link_speed"], link["current_link_width"])
        if raw > 0 and (not best or raw < best["raw_gbs"]):
            best = {
                "bdf": bdf,
                "speed": link["current_link_speed"],
                "width": link["current_link_width"],
                "max_width": link["max_link_width"],
                "raw_gbs": round(raw, 3),
            }
    return best


def discover_topology(sysfs: str = SYSFS) -> dict[str, Any]:
    """GPUs, their bridge chains, island keys and link state from sysfs."""
    gpus = scan_gpus(sysfs)
    chains = {b: bdf_chain(os.path.realpath(os.path.join(sysfs, b))) for b in gpus}
    keys = island_keys(chains)
    entries = []
    for b in gpus:
        chain = chains[b]
        parent = chain[-1] if chain else ""
        link = link_info(b, sysfs)
        entries.append(
            {
                "bdf": b,
                "bus": b.split(":")[1],
                "island": keys[b],
                "chain": chain,
                "link": link,
                "link_raw_gbs": link_raw_gbs(
                    link["current_link_speed"], link["current_link_width"]
                ),
                "narrowest": narrowest_link([*chain, b], sysfs),
                "parent": parent,
                "parent_link": link_info(parent, sysfs) if parent else {},
            }
        )
    return {"gpus": entries, "islands": sorted(set(keys.values()))}


def aer_snapshot(bdfs: Sequence[str], sysfs: str = SYSFS) -> dict[str, int]:
    """Sum of the AER counters per (function, kind) for the given functions."""
    snap: dict[str, int] = {}
    for bdf in bdfs:
        for kind in ("correctable", "nonfatal", "fatal"):
            total = 0
            for line in _read(os.path.join(sysfs, bdf, f"aer_dev_{kind}")).splitlines():
                name, _, val = line.strip().rpartition(" ")
                if name and not name.startswith("TOTAL") and val.isdigit():
                    total += int(val)
            snap[f"{bdf}:{kind}"] = total
    return snap


def aer_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Non-zero AER counter growth between two snapshots."""
    return {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}


def members_from_keys(idx_to_island: dict[int, str]) -> list[list[int]]:
    """Logical GPU indices grouped by island, lowest index first."""
    groups: dict[str, list[int]] = {}
    for idx in sorted(idx_to_island):
        groups.setdefault(idx_to_island[idx], []).append(idx)
    return sorted(groups.values(), key=lambda g: g[0])


def parse_islands(text: str) -> list[list[int]]:
    """Parse '0,1;2,3,4,5' into [[0, 1], [2, 3, 4, 5]]."""
    return [[int(x) for x in grp.split(",") if x] for grp in text.split(";") if grp]


def ring(members: Sequence[int]) -> list[Flow]:
    """Ring flows inside one island (a pair gives both directions)."""
    if len(members) < 2:
        return []
    if len(members) == 2:
        return [(members[0], members[1]), (members[1], members[0])]
    return [(m, members[(i + 1) % len(members)]) for i, m in enumerate(members)]


def interleave(lists: Sequence[Sequence[Flow]]) -> list[Flow]:
    """Round-robin merge, so the k-th prefix touches every list."""
    out: list[Flow] = []
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x))
    return out


def flow_patterns(members: Sequence[Sequence[int]]) -> dict[str, list[Flow]]:
    """Named flow lists; the k-th prefix of a list is the k-concurrent case."""
    pats: dict[str, list[Flow]] = {}
    letters = "ABCDEFGH"
    rings = []
    for i, grp in enumerate(members):
        r = ring(grp)
        if r:
            pats[f"intra_{letters[i]}"] = r
            rings.append(r)
    if len(rings) > 1:
        pats["intra_all"] = interleave(rings)
    if len(members) > 1 and members[0] and members[1]:
        a, b = members[0], members[1]
        fwd = [(a[i % len(a)], b[i % len(b)]) for i in range(MAX_LADDER)]
        pats["cross"] = fwd
        pats["cross_bidir"] = [
            (s, d) if i % 2 == 0 else (d, s) for i, (s, d) in enumerate(fwd)
        ]
    everyone = [g for grp in members for g in grp]
    if len(everyone) >= 3:
        big = max(members, key=len)
        sink = big[0]
        others = [g for g in everyone if g != sink]
        pats["fanin"] = [(g, sink) for g in others[:MAX_LADDER]]
        pats["fanout"] = [(sink, g) for g in others[:MAX_LADDER]]
    return {k: v[:MAX_LADDER] for k, v in pats.items() if v}


def representative_pairs(members: Sequence[Sequence[int]]) -> list[Flow]:
    """Ordered pairs for the per-pair sweep: inside each island and across."""
    pairs: list[Flow] = []
    for grp in members:
        if len(grp) >= 2:
            pairs += [(grp[0], grp[1]), (grp[1], grp[0])]
        if len(grp) >= 3:
            pairs += [(grp[0], grp[-1]), (grp[-1], grp[0])]
    if len(members) > 1 and members[0] and members[1]:
        a, b = members[0], members[1]
        pairs += [(a[0], b[0]), (b[0], a[0]), (a[-1], b[-1]), (b[-1], a[-1])]
    return list(dict.fromkeys(pairs))


def a2a_groups(members: Sequence[Sequence[int]], n: int) -> dict[str, list[list[int]]]:
    """Named a2a group sets (a set runs its groups concurrently)."""
    everyone = [g for grp in members for g in grp]
    out: dict[str, list[list[int]]] = {}
    for i, grp in enumerate(members):
        if len(grp) >= 4:
            out[f"island{i}_4"] = [list(grp[:4])]
    if len(members) > 1 and len(members[0]) >= 2 and len(members[1]) >= 2:
        out["mixed_4"] = [[*members[0][:2], *members[1][:2]]]
        out["intra_concurrent"] = [list(g) for g in members if len(g) >= 2]
    if n >= 6 and len(everyone) >= 6:
        out["flat_6"] = [everyone[:6]]
    if len(everyone) >= 2:
        out["pair_2"] = [list(everyone[:2])]
    return out


def ladder_verdict(agg: Sequence[float]) -> dict[str, Any]:
    """Classify an aggregate-GB/s ladder: agg[k-1] is k concurrent flows."""
    if not agg or agg[0] <= 0:
        return {"verdict": "NO_DATA", "scaling": []}
    scaling = [round(v / (k * agg[0]), 3) for k, v in enumerate(agg, 1)]
    best = 0.0
    for k, v in enumerate(agg, 1):
        if k > 1 and v < DROP_FRACTION * best:
            detail = f"{v:.1f} GB/s < {DROP_FRACTION:.2f} x best {best:.1f} GB/s"
            return {
                "verdict": "COLLAPSE",
                "at_k": k,
                "detail": detail,
                "scaling": scaling,
            }
        best = max(best, v)
    if len(agg) > 1 and agg[-1] < SATURATED_EFFICIENCY * len(agg) * agg[0]:
        return {"verdict": "SATURATED", "scaling": scaling}
    return {"verdict": "LINEAR", "scaling": scaling}


def parse_value(text: str) -> Any:
    """Parse an int or float token, else keep the text."""
    for conv in (int, float):
        with contextlib.suppress(ValueError):
            return conv(text)
    return text


def parse_probe_line(line: str) -> dict[str, Any] | None:
    """Parse 'KIND label k=v ...' into a record; None for comments."""
    parts = line.split()
    if not parts or parts[0].startswith(("#", "###")):
        return None
    rec: dict[str, Any] = {"kind": parts[0]}
    labels = []
    for tok in parts[1:]:
        key, eq, val = tok.partition("=")
        if eq:
            rec[key] = parse_value(val)
        else:
            labels.append(tok)
    if labels:
        rec["labels"] = labels
    return rec


def probe_plan(members: Sequence[Sequence[int]], n: int) -> list[tuple[str, str]]:
    """(name, batch command) list for the kernel-level probe."""
    cmds: list[tuple[str, str]] = []
    pairs = representative_pairs(members)
    plist = " ".join(f"{a}:{b}" for a, b in pairs)
    sizes = ",".join(str(s) for s in PAIR_SIZES)
    for payload in (0, 4 * KIB, 16 * KIB, 64 * KIB):
        cmds.append((f"lat payload={payload}", f"lat 2000 {payload} {plist}"))
    for mode in ("push", "pull", "sdma"):
        cmds.append((f"curve {mode}", f"curve {mode} 2000 0 {sizes} {plist}"))
    for name, flows in flow_patterns(members).items():
        for k in range(1, len(flows) + 1):
            xs = " ".join(f"p:{a}>{b}" for a, b in flows[:k])
            for size in FLOW_SIZES:
                cmds.append(
                    (f"conc {name} k={k} size={size}", f"bw 0.4 {size} 128 {xs}")
                )
    for gname, groups in a2a_groups(members, n).items():
        gl = " ".join(",".join(str(i) for i in g) for g in groups)
        for payload in PROBE_PAYLOADS:
            cmds.append((f"a2a {gname} payload={payload}", f"a2a 500 {payload} {gl}"))
    return cmds


def plan_cases(members: Sequence[Sequence[int]], n: int) -> dict[str, Any]:
    """Cell counts and a rough runtime for each stage."""
    pairs = representative_pairs(members)
    pats = flow_patterns(members)
    groups = a2a_groups(members, n)
    cells = {
        "pair": len(pairs) * len(PAIR_SIZES),
        "lat": len(pairs) * len(LAT_SIZES),
        "flows": sum(len(v) for v in pats.values()) * len(FLOW_SIZES),
        "a2a": len(groups) * len(A2A_CHUNKS),
        "probe": len(probe_plan(members, n)),
    }
    est = {k: round(v * EST_S[k]) for k, v in cells.items()}
    return {
        "members": [list(m) for m in members],
        "pairs": pairs,
        "patterns": pats,
        "a2a_groups": groups,
        "cells": cells,
        "est_s": est,
        "est_total_s": sum(est.values()),
    }


# --------------------------------------------------------------------------- #
# Lease, quiet window, build lock
# --------------------------------------------------------------------------- #
def held_slots(owner: str, lease_dir: str) -> list[str]:
    """Slots held by owner: files <owner>.<slot> in the lease directory."""
    try:
        names = sorted(os.listdir(os.path.expanduser(lease_dir)))
    except OSError:
        return []
    return [n.rpartition(".")[2] for n in names if n.rpartition(".")[0] == owner]


def lease_problems(
    owner: str,
    lease_dir: str,
    slot_buses: dict[str, tuple[str, ...]],
    buses: dict[int, str],
) -> list[str]:
    """Reasons the visible GPUs are not all covered by the owner's slots."""
    slots = held_slots(owner, lease_dir)
    if not slots:
        return [f"no lease file {owner}.<slot> in {lease_dir}"]
    allowed: set[str] = set()
    problems = []
    for slot in slots:
        if slot in slot_buses:
            allowed.update(slot_buses[slot])
        else:
            problems.append(f"slot {slot!r} has no known buses (use --slot)")
    for idx, bus in sorted(buses.items()):
        if bus not in allowed:
            problems.append(f"HIP {idx} is bus {bus}, outside held slots {slots}")
    return problems


def docker_build_running() -> bool:
    """True while a docker buildx build runs (no measured window then)."""
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return False
    res = subprocess.run(
        [pgrep, "-f", "docker.*buildx.*build"], capture_output=True, text=True
    )
    return res.returncode == 0 and bool(res.stdout.strip())


@dataclass
class Guard:
    """Quiet-window discipline: touch file, runtime cap, release on exit."""

    owner: str
    quiet_dir: str = QUIET_DIR
    max_minutes: float = 14.0
    clock: Callable[[], float] = time.monotonic
    enabled: bool = True
    _t0: float = field(default=0.0, init=False)
    _touched: float = field(default=0.0, init=False)

    def _path(self) -> str:
        return os.path.join(os.path.expanduser(self.quiet_dir), self.owner)

    def __enter__(self) -> Self:
        self._t0 = self._touched = self.clock()
        if self.enabled and self.owner:
            os.makedirs(os.path.dirname(self._path()), exist_ok=True)
            with open(self._path(), "a", encoding="utf-8"):
                os.utime(self._path())
        return self

    def tick(self) -> bool:
        """Re-touch the quiet file every 4 minutes; False once past the cap."""
        now = self.clock()
        if self.enabled and self.owner and now - self._touched > 240:
            with contextlib.suppress(OSError):
                os.utime(self._path())
            self._touched = now
        return now - self._t0 <= self.max_minutes * 60

    def __exit__(self, *exc: object) -> None:
        if self.enabled and self.owner:
            with contextlib.suppress(OSError):
                os.remove(self._path())


# --------------------------------------------------------------------------- #
# Probe backend (kernel-level p2p_probe through its batch mode)
# --------------------------------------------------------------------------- #
class ProbeSession:
    """One `p2p_probe batch` process: READY once, then one command per call."""

    def __init__(self, binary: str, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [binary, "batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.ready = self._read_until("READY", 120.0)[-1]

    def _read_until(self, prefix: str, timeout: float) -> list[dict[str, Any]]:
        assert self.proc.stdout is not None
        timer = threading.Timer(timeout, self.proc.kill)
        timer.start()
        recs: list[dict[str, Any]] = []
        try:
            for line in self.proc.stdout:
                rec = parse_probe_line(line)
                if rec is None:
                    continue
                recs.append(rec)
                if rec["kind"] == prefix:
                    return recs
        finally:
            timer.cancel()
        err = self.proc.stderr.read() if self.proc.stderr else ""
        raise RuntimeError(f"p2p_probe ended before {prefix}: {err.strip()[:400]}")

    def run(self, command: str, timeout: float = 300.0) -> list[dict[str, Any]]:
        """Run one batch command; records up to and including its END line."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        return self._read_until("END", timeout)

    def close(self) -> None:
        if self.proc.stdin:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5)
        if self.proc.poll() is None:
            self.proc.kill()


# --------------------------------------------------------------------------- #
# Torch backend (Tensor.copy_ between peer devices; no new kernels)
# --------------------------------------------------------------------------- #
def device_count(torch: Any) -> int:
    """Visible accelerator count without the device-wide torch.cuda calls."""
    acc = getattr(torch, "accelerator", None)
    if acc is not None and hasattr(acc, "device_count"):
        return int(acc.device_count())
    count = 0
    while True:
        try:
            torch.empty(1, device=f"cuda:{count}")
        except (RuntimeError, AssertionError):
            return count
        count += 1


class TorchBench:
    """Peer copies between the visible GPUs through Tensor.copy_."""

    def __init__(self, want: int | None) -> None:
        try:
            torch: Any = importlib.import_module("torch")
        except ImportError as exc:
            raise SystemExit(f"p2p_bench: torch is not importable ({exc})") from exc
        if not torch.cuda.is_available():
            raise SystemExit("p2p_bench: torch sees no GPU (HIP_VISIBLE_DEVICES?)")
        self.t = torch
        total = device_count(torch)
        self.n = total if want is None else min(want, total)
        if self.n < 2:
            raise SystemExit("p2p_bench: need at least two visible GPUs")
        self._streams: dict[tuple[Any, ...], Any] = {}

    def props(self) -> list[dict[str, Any]]:
        """Name, PCI address and arch of each visible device."""
        out = []
        for i in range(self.n):
            p = self.t.cuda.get_device_properties(i)
            bus = getattr(p, "pci_bus_id", None)
            dom = getattr(p, "pci_domain_id", 0)
            dev = getattr(p, "pci_device_id", 0)
            bdf = f"{dom:04x}:{bus:02x}:{dev:02x}.0" if bus is not None else ""
            out.append(
                {
                    "idx": i,
                    "name": p.name,
                    "arch": getattr(p, "gcnArchName", ""),
                    "bdf": bdf,
                    "bus": f"{bus:02x}" if bus is not None else "",
                    "mem_gib": round(p.total_memory / (1 << 30), 1),
                }
            )
        return out

    def peer_matrix(self) -> list[list[bool]]:
        """can_device_access_peer for every ordered pair."""
        return [
            [
                i != j and bool(self.t.cuda.can_device_access_peer(i, j))
                for j in range(self.n)
            ]
            for i in range(self.n)
        ]

    def env(self) -> dict[str, Any]:
        """Torch and HIP versions."""
        return {
            "torch": self.t.__version__,
            "hip": getattr(self.t.version, "hip", None),
        }

    # -- plumbing ---------------------------------------------------------- #
    def _stream(self, dev: int, key: tuple[Any, ...]) -> Any:
        s = self._streams.get(key)
        if s is None:
            s = self.t.cuda.Stream(device=dev)
            self._streams[key] = s
        return s

    def _sync(self, *devs: int) -> None:
        for d in dict.fromkeys(devs):
            self.t.cuda.current_stream(d).synchronize()

    def _pattern(self, dev: int, size: int, seed: int) -> Any:
        t = self.t
        chunk = min(size, MIB)
        base = (t.arange(chunk, dtype=t.int32, device=f"cuda:{dev}") * 7 + seed) % 251
        reps = -(-size // chunk)
        return base.to(t.uint8).repeat(reps)[:size].contiguous()

    def _copy(self, dst: Any, src: Any, stream: Any) -> None:
        with self.t.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)

    def _event(self) -> Any:
        return self.t.cuda.Event(enable_timing=True)

    # -- measurements ------------------------------------------------------ #
    def pair(self, src: int, dst: int, size: int, reps: int) -> dict[str, Any]:
        """Sustained copy bandwidth of one ordered pair at one size."""
        stream = self._stream(src, ("pair", src, dst))
        a = self._pattern(src, size, 1)
        b = self._pattern(dst, size, 2)  # stale content: a skipped copy shows
        self._sync(src, dst)
        n_iter = copies_per_rep(size)
        for _ in range(3):
            self._copy(b, a, stream)
        stream.synchronize()
        usec = []
        for _ in range(reps):
            e0, e1 = self._event(), self._event()
            e0.record(stream)
            for _ in range(n_iter):
                self._copy(b, a, stream)
            e1.record(stream)
            e1.synchronize()
            usec.append(e0.elapsed_time(e1) * 1e3 / n_iter)
        self._sync(dst)
        ok = bool(self.t.equal(b, self._pattern(dst, size, 1)))
        stats = summarize(usec)
        return {
            "src": src,
            "dst": dst,
            "size": size,
            "copies_per_rep": n_iter,
            "us_per_copy": stats,
            "gbps_p50": gbps(size, stats["p50"]),
            "gbps_best": gbps(size, stats["min"]),
            "verify": "ok" if ok else "FAIL",
        }

    def latency(self, src: int, dst: int, size: int, samples: int) -> dict[str, Any]:
        """Completion latency of a single synchronised copy."""
        stream = self._stream(src, ("lat", src, dst))
        a = self._pattern(src, size, 3)
        b = self._pattern(dst, size, 4)
        self._sync(src, dst)
        e0, e1 = self._event(), self._event()
        for _ in range(20):
            self._copy(b, a, stream)
        stream.synchronize()
        gpu_us, host_us = [], []
        for _ in range(samples):
            h0 = time.perf_counter()
            e0.record(stream)
            self._copy(b, a, stream)
            e1.record(stream)
            e1.synchronize()
            host_us.append((time.perf_counter() - h0) * 1e6)
            gpu_us.append(e0.elapsed_time(e1) * 1e3)
        return {
            "src": src,
            "dst": dst,
            "size": size,
            "gpu_us": summarize(gpu_us),
            "host_us": summarize(host_us),
        }

    def _flow_state(self, flows: Sequence[Flow], size: int, tag: str) -> list[Any]:
        state = []
        for i, (s, d) in enumerate(flows):
            stream = self._stream(s, (tag, i, s, d))
            state.append(
                (
                    stream,
                    self._pattern(s, size, 10 + i),
                    self._pattern(d, size, 100 + i),
                )
            )
        self._sync(*[x for f in flows for x in f])
        return state

    def flows(self, flows: Sequence[Flow], size: int, reps: int) -> dict[str, Any]:
        """Concurrent flows, one stream each: aggregate and per-flow GB/s."""
        state = self._flow_state(flows, size, "flow")
        n_iter = copies_per_rep(size)
        for _ in range(2):
            for stream, a, b in state:
                self._copy(b, a, stream)
        for stream, _, _ in state:
            stream.synchronize()
        agg_runs: list[float] = []
        per_runs: list[list[float]] = [[] for _ in flows]
        for _ in range(reps):
            ev = [(self._event(), self._event()) for _ in flows]
            for (stream, _, _), (e0, _) in zip(state, ev):
                e0.record(stream)
            for _ in range(n_iter):
                for stream, a, b in state:
                    self._copy(b, a, stream)
            for (stream, _, _), (_, e1) in zip(state, ev):
                e1.record(stream)
            for _, e1 in ev:
                e1.synchronize()
            ms = [e0.elapsed_time(e1) for e0, e1 in ev]
            agg_runs.append(gbps(size * n_iter * len(flows), max(ms) * 1e3))
            for j, m in enumerate(ms):
                per_runs[j].append(gbps(size * n_iter, m * 1e3))
        ok = True
        for i, (_, a, b) in enumerate(state):
            d = flows[i][1]
            self._sync(d)
            ok = ok and bool(self.t.equal(b, self._pattern(d, size, 10 + i)))
        return {
            "flows": [list(f) for f in flows],
            "size": size,
            "agg_gbps": statistics.median(agg_runs),
            "per_flow_gbps": [round(statistics.median(r), 2) for r in per_runs],
            "verify": "ok" if ok else "FAIL",
        }

    def a2a(
        self, groups: Sequence[Sequence[int]], chunk: int, reps: int
    ) -> dict[str, Any]:
        """Flat all-to-all step (groups run concurrently), host wall clock."""
        flows: list[Flow] = []
        for grp in groups:
            for shift in range(1, len(grp)):
                flows += [(s, grp[(i + shift) % len(grp)]) for i, s in enumerate(grp)]
        state = self._flow_state(flows, chunk, "a2a")
        gmax = max(len(g) for g in groups)
        est_step = chunk * (gmax - 1) / 12e9 + 20e-6 * len(flows)
        n_iter = max(2, min(200, int(0.02 / est_step)))
        for _ in range(2):
            for stream, a, b in state:
                self._copy(b, a, stream)
        for stream, _, _ in state:
            stream.synchronize()
        step_us = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                for stream, a, b in state:
                    self._copy(b, a, stream)
            for stream, _, _ in state:
                stream.synchronize()
            step_us.append((time.perf_counter() - t0) * 1e6 / n_iter)
        ok = True
        for i, (_, a, b) in enumerate(state):
            d = flows[i][1]
            self._sync(d)
            ok = ok and bool(self.t.equal(b, self._pattern(d, chunk, 10 + i)))
        stats = summarize(step_us)
        return {
            "groups": [list(g) for g in groups],
            "chunk": chunk,
            "flows": len(flows),
            "steps_per_rep": n_iter,
            "step_us": stats,
            "algbw_gbps_per_rank": gbps((gmax - 1) * chunk, stats["p50"]),
            "verify": "ok" if ok else "FAIL",
        }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def env_record() -> dict[str, Any]:
    """Environment that changes the result."""
    keys = (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_ENABLE_SDMA",
        "GPU_MAX_HW_QUEUES",
        "HSA_FORCE_FINE_GRAIN_PCIE",
    )
    return {
        "host": os.uname().nodename,
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "env": {k: os.environ.get(k) for k in keys},
    }


def print_table(rows: Sequence[Sequence[str]]) -> None:
    """Left-aligned fixed-width table on stdout."""
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def stage_topology(
    devices: Sequence[dict[str, Any]],
    tb: TorchBench | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Sysfs topology joined with the visible device list."""
    topo = discover_topology()
    bus_island = {g["bus"]: g["island"] for g in topo["gpus"]}
    idx_island = {d["idx"]: bus_island.get(d["bus"], "unknown") for d in devices}
    if args.islands:
        members = parse_islands(args.islands)
    else:
        members = members_from_keys(idx_island) if idx_island else []
    topo["devices"] = list(devices)
    topo["members"] = members
    topo["peer_access"] = tb.peer_matrix() if tb else []
    by_bus = {g["bus"]: g for g in topo["gpus"]}
    print("topology (logical index -> bus, island, link):")
    for d in devices:
        g = by_bus.get(d["bus"], {})
        link = g.get("link", {})
        narrow = g.get("narrowest", {})
        print(
            f"  {d['idx']}: bus {d['bus']} {d.get('name', '')} "
            f"island={g.get('island', '?')} endpoint "
            f"{link.get('current_link_speed', '?')} "
            f"x{link.get('current_link_width', '?')}; narrowest on path "
            f"x{narrow.get('width', '?')} {narrow.get('speed', '?')} "
            f"raw {narrow.get('raw_gbs', 0):.1f} GB/s at {narrow.get('bdf', '?')}"
        )
    print(f"  islands (logical): {members}")
    return topo


def stage_pair(
    tb: TorchBench,
    members: Sequence[Sequence[int]],
    args: argparse.Namespace,
    guard: Guard,
) -> list[dict[str, Any]]:
    """Per-pair bandwidth sweep and latency percentiles."""
    pairs = args.pair_list or representative_pairs(members)
    sizes = args.sizes or PAIR_SIZES
    out: list[dict[str, Any]] = []
    rows = [["pair", *(fmt_size(s) for s in sizes)]]
    for src, dst in pairs:
        row = [f"{src}>{dst}"]
        for size in sizes:
            if not guard.tick():
                print("p2p_bench: runtime cap reached, stopping pair stage")
                return out
            res = tb.pair(src, dst, size, args.reps)
            out.append(res)
            row.append(
                f"{res['gbps_p50']:.1f}" + ("" if res["verify"] == "ok" else "!")
            )
        rows.append(row)
    print("pair copy bandwidth, GB/s p50 (! = verify FAIL):")
    print_table(rows)
    for src, dst in pairs:
        for size in LAT_SIZES:
            if not guard.tick():
                return out
            res = tb.latency(src, dst, size, args.lat_samples)
            res["stage"] = "latency"
            out.append(res)
    lat = [r for r in out if r.get("stage") == "latency"]
    print("single-copy completion latency, us (gpu events p50/p99, host p50):")
    print_table(
        [["pair", "size", "gpu_p50", "gpu_p99", "host_p50"]]
        + [
            [
                f"{r['src']}>{r['dst']}",
                fmt_size(r["size"]),
                f"{r['gpu_us']['p50']:.1f}",
                f"{r['gpu_us']['p99']:.1f}",
                f"{r['host_us']['p50']:.1f}",
            ]
            for r in lat
        ]
    )
    return out


def stage_flows(
    tb: TorchBench,
    members: Sequence[Sequence[int]],
    args: argparse.Namespace,
    guard: Guard,
) -> list[dict[str, Any]]:
    """Concurrent-flow ladders per pattern and size, with collapse verdicts."""
    out: list[dict[str, Any]] = []
    sizes = args.flow_sizes or FLOW_SIZES
    for name, flows in flow_patterns(members).items():
        for size in sizes:
            aggs: list[float] = []
            cells = []
            for k in range(1, len(flows) + 1):
                if not guard.tick():
                    print("p2p_bench: runtime cap reached, stopping flows stage")
                    return out
                res = tb.flows(flows[:k], size, args.reps)
                aggs.append(res["agg_gbps"])
                cells.append(res)
            verdict = ladder_verdict(aggs)
            out.append(
                {
                    "pattern": name,
                    "size": size,
                    "flows": [list(f) for f in flows],
                    "agg_gbps": aggs,
                    "cells": cells,
                    **verdict,
                }
            )
            line = " ".join(f"{a:.1f}" for a in aggs)
            print(
                f"flows {name:12s} {fmt_size(size):8s} agg[1..k]= {line} GB/s "
                f"-> {verdict['verdict']}"
                + (f" at k={verdict['at_k']}" if "at_k" in verdict else "")
            )
    return out


def stage_a2a(
    tb: TorchBench,
    members: Sequence[Sequence[int]],
    args: argparse.Namespace,
    guard: Guard,
) -> list[dict[str, Any]]:
    """Flat all-to-all steps and their island-local parts."""
    out: list[dict[str, Any]] = []
    cap = int(os.environ.get("GPU_MAX_HW_QUEUES", "4"))
    for name, groups in a2a_groups(members, tb.n).items():
        gmax = max(len(g) for g in groups)
        if gmax - 1 > cap:
            print(
                f"p2p_bench: warning: {name} sends {gmax - 1} flows per GPU but "
                f"GPU_MAX_HW_QUEUES={cap}; flows will share queues"
            )
        for chunk in args.a2a_chunks or A2A_CHUNKS:
            if not guard.tick():
                print("p2p_bench: runtime cap reached, stopping a2a stage")
                return out
            res = tb.a2a(groups, chunk, args.reps)
            res["name"] = name
            out.append(res)
            step = res["step_us"]["p50"]
            print(
                f"a2a {name:17s} chunk={fmt_size(chunk):8s} step p50={step:.1f} us "
                f"algbw/rank={res['algbw_gbps_per_rank']:.1f} GB/s "
                f"verify={res['verify']}"
            )
    return out


def probe_devices(binary: str, env: dict[str, str]) -> list[dict[str, Any]]:
    """Logical index and PCI bus from the DEV lines of `p2p_probe info`."""
    if not os.path.isfile(binary):
        return []
    res = subprocess.run(
        [binary, "info"], capture_output=True, text=True, env=env, timeout=120
    )
    devices = []
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0] == "DEV" and parts[1].isdigit():
            fields = parts[-1].partition("=")[2].split(":")
            if len(fields) >= 2:
                bus = fields[-2].lower()
                devices.append({"idx": int(parts[1]), "bus": bus, "name": ""})
    return devices


def curve_table(records: Sequence[dict[str, Any]]) -> list[list[str]]:
    """GB/s (size over p50 latency) per mode, pair and size of CURVE records."""
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for r in records:
        if r.get("kind") == "CURVE" and r.get("p50_us", 0) > 0:
            key = (str(r["mode"]), f"{r['src']}>{r['dst']}")
            cells.setdefault(key, {})[int(r["size"])] = gbps(r["size"], r["p50_us"])
    sizes = sorted({s for v in cells.values() for s in v})
    rows = [["mode pair", *(fmt_size(s) for s in sizes)]]
    for (mode, pair), v in sorted(cells.items()):
        vals = [f"{v[s]:.1f}" if s in v else "-" for s in sizes]
        rows.append([f"{mode} {pair}", *vals])
    return rows


def stage_probe(
    members: Sequence[Sequence[int]], n: int, args: argparse.Namespace, guard: Guard
) -> dict[str, Any]:
    """Kernel-level push/pull/sdma, flag latency, concurrency, one-shot a2a."""
    binary = args.probe_bin
    if not os.path.isfile(binary):
        return {"skipped": f"{binary} not found; build it (see the module docstring)"}
    session = ProbeSession(binary, dict(os.environ))
    records: list[dict[str, Any]] = []
    try:
        for name, cmd in probe_plan(members, n):
            if not guard.tick():
                print("p2p_bench: runtime cap reached, stopping probe stage")
                break
            recs = session.run(cmd)
            for r in recs:
                r["cell"] = name
            records.extend(recs)
    finally:
        session.close()
    rows = curve_table(records)
    if len(rows) > 1:
        print("probe curve, GB/s at p50 latency (push = stores, pull = loads):")
        print_table(rows)
    for r in records:
        if r["kind"] in ("LAT", "A2A", "AGG"):
            body = " ".join(f"{k}={v}" for k, v in r.items() if k != "kind")
            print(f"probe {r['kind']} {body}")
    return {"ready": session.ready, "records": records}


def findings(result: dict[str, Any]) -> dict[str, Any]:
    """Headline verdicts derived from the measured cells."""
    big = [
        r
        for r in result.get("pair", [])
        if r.get("size", 0) >= 4 * MIB and "gbps_p50" in r
    ]
    peak = max((r["gbps_p50"] for r in big), default=0.0)
    ladders = result.get("flows", [])
    collapsed = [
        f"{x['pattern']}@{fmt_size(x['size'])}"
        for x in ladders
        if x["verdict"] == "COLLAPSE"
    ]
    cells = [c for x in ladders for c in x["cells"]]
    failed = [
        r
        for r in result.get("pair", []) + result.get("a2a", []) + cells
        if r.get("verify") == "FAIL"
    ]
    keys = ("src", "dst", "size", "chunk", "name", "flows", "groups")
    return {
        "peak_pair_gbps_ge_4MiB": round(peak, 2),
        "link_hint": link_hint(peak),
        "collapse_detected": bool(collapsed),
        "collapsed": collapsed,
        "verify_failures": len(failed),
        "verify_failed_first": [{k: r[k] for k in keys if k in r} for r in failed[:5]],
    }


def default_members(args: argparse.Namespace) -> list[list[int]]:
    """Islands from --islands, else the target six-GPU layout or one island."""
    if args.islands:
        return parse_islands(args.islands)
    if not args.gpus or args.gpus == 6:
        return [[0, 1], [2, 3, 4, 5]]
    return [list(range(args.gpus))]


def slot_map(extra: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """SLOT_BUSES plus --slot NAME=BUS,BUS overrides."""
    out = dict(SLOT_BUSES)
    for item in extra:
        name, _, buses = item.partition("=")
        out[name] = tuple(b.strip().lower() for b in buses.split(",") if b.strip())
    return out


def run_gpu(args: argparse.Namespace) -> int:
    """Run the selected GPU stages under the lease, quiet and build guards."""
    stages = {"all": ("topo", "pair", "flows", "a2a", "probe")}.get(
        args.cmd, (args.cmd,)
    )
    if args.dry_run:
        members = default_members(args)
        plan = plan_cases(members, args.gpus or sum(len(m) for m in members))
        print(json.dumps({"dry_run": True, "stages": stages, **plan}, indent=1))
        return 0
    if not args.owner and not args.no_lease_check:
        print("p2p_bench: --owner <holder name> is required (lease and quiet file)")
        return 2
    needs_torch = any(s in stages for s in ("topo", "pair", "flows", "a2a"))
    tb = TorchBench(args.gpus) if needs_torch else None
    devices = tb.props() if tb else probe_devices(args.probe_bin, dict(os.environ))
    result: dict[str, Any] = {"schema": SCHEMA, **env_record(), "stages": list(stages)}
    if tb:
        result["torch"] = tb.env()
    topo = stage_topology(devices, tb, args)
    result["topology"] = topo
    if not args.no_lease_check:
        buses = {d["idx"]: d["bus"] for d in devices}
        problems = lease_problems(
            args.owner, args.lease_dir, slot_map(args.slot), buses
        )
        if problems:
            print("p2p_bench: lease check failed:\n  " + "\n  ".join(problems))
            print(
                "  hold the slot files first, or pass --no-lease-check (never on a "
                "box someone else measures on)"
            )
            return 2
    if not args.ignore_build and docker_build_running():
        print(
            "p2p_bench: a docker buildx build is running; no measured window now "
            "(--ignore-build overrides)"
        )
        return 2
    n = len(devices) or args.gpus or 2
    members = topo["members"] or [list(range(n))]
    bdfs = [g["bdf"] for g in topo["gpus"]] + [
        g["parent"] for g in topo["gpus"] if g["parent"]
    ]
    aer0 = aer_snapshot(bdfs)
    with Guard(
        args.owner, args.quiet_dir, args.max_minutes, enabled=not args.no_quiet
    ) as guard:
        if tb and "pair" in stages:
            result["pair"] = stage_pair(tb, members, args, guard)
        if tb and "flows" in stages:
            result["flows"] = stage_flows(tb, members, args, guard)
        if tb and "a2a" in stages:
            result["a2a"] = stage_a2a(tb, members, args, guard)
        if "probe" in stages:
            result["probe"] = stage_probe(members, n, args, guard)
    result["aer_delta"] = aer_delta(aer0, aer_snapshot(bdfs))
    result["findings"] = findings(result)
    print("findings:", json.dumps(result["findings"], default=str))
    if result["aer_delta"]:
        print("AER counters moved during the run:", result["aer_delta"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=str)
        print(f"wrote {args.out}")
    return 0


def run_plan(args: argparse.Namespace) -> int:
    """Print the cell plan and the commands to run it (no GPU, no torch)."""
    members = default_members(args)
    n = args.gpus or sum(len(m) for m in members)
    plan = plan_cases(members, n)
    print(f"islands (logical indices): {plan['members']}")
    print("pairs:", " ".join(f"{a}>{b}" for a, b in plan["pairs"]))
    for name, flows in plan["patterns"].items():
        print(f"pattern {name}: " + " ".join(f"{a}>{b}" for a, b in flows))
    for name, groups in plan["a2a_groups"].items():
        print(f"a2a {name}: " + " ; ".join(",".join(map(str, g)) for g in groups))
    print("cells:", plan["cells"])
    print(f"estimated seconds (rough): {plan['est_s']} total {plan['est_total_s']}")
    print(
        "run once per HSA_ENABLE_SDMA in 0 1; b4 (HIP 2-5) needs the b4 lease, "
        "the six-GPU run needs a2 and b4:"
    )
    print(
        "  HIP_VISIBLE_DEVICES=2,3,4,5 GPU_MAX_HW_QUEUES=6 HSA_ENABLE_SDMA=0 "
        "python3 tools/rdna/a2a/p2p_bench.py all --owner <Agent> --out b4_sdma0.json"
    )
    return 0


FAKE_PROBE = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "if sys.argv[1] == 'info':\n"
    "    print('DEV 0 gfx1030 pci=0000:d9:00.0')\n"
    "    print('DEV 1 gfx1030 pci=0000:dc:00.0')\n"
    "elif sys.argv[1] == 'batch':\n"
    "    print('READY devices=2 tick_mhz=100.0,100.0 version=fake', flush=True)\n"
    "    for line in sys.stdin:\n"
    "        tok = line.split()\n"
    "        print('###', *tok)\n"
    "        if tok[0] == 'curve':\n"
    "            print('CURVE mode=' + tok[1] + ' src=0 dst=1 size=4096 p50_us=8.0')\n"
    "        print('END rc=0 ms=1.0', flush=True)\n"
)


def selftest() -> int:
    """CPU-only checks of the pure logic."""
    assert parse_size("64MiB") == 64 * MIB and parse_size("4k") == 4 * KIB
    assert parse_size("4096") == 4096 and fmt_size(4 * MIB) == "4 MiB"
    assert fmt_size(5000) == "5000 B"
    vals = [float(x) for x in range(1, 101)]
    assert pct(vals, 50) == 50.0 and pct(vals, 99) == 99.0 and pct(vals, 100) == 100.0
    assert summarize([3.0, 1.0, 2.0])["p50"] == 2.0
    assert abs(link_raw_gbs("16.0 GT/s PCIe", "8") - 15.75) < 0.01
    assert abs(link_raw_gbs("16.0 GT/s PCIe", "16") - 31.5) < 0.01
    assert link_raw_gbs("", "") == 0.0
    assert link_hint(13.9).startswith("x8") and link_hint(24.0).startswith("x16")
    assert 4 <= copies_per_rep(4 * KIB) <= 256 and copies_per_rep(64 * MIB) >= 4
    # island keys: single switch tree (new) and two roots (old)
    new = {
        "0000:d1:00.0": [
            "0000:c0:03.1",
            "0000:c5:00.0",
            "0000:c6:04.0",
            "0000:cb:00.0",
            "0000:cc:10.0",
        ],
        "0000:d4:00.0": [
            "0000:c0:03.1",
            "0000:c5:00.0",
            "0000:c6:04.0",
            "0000:cb:00.0",
            "0000:cc:18.0",
        ],
        "0000:d9:00.0": [
            "0000:c0:03.1",
            "0000:c5:00.0",
            "0000:c6:08.0",
            "0000:d5:00.0",
            "0000:d6:00.0",
        ],
        "0000:dc:00.0": [
            "0000:c0:03.1",
            "0000:c5:00.0",
            "0000:c6:08.0",
            "0000:d5:00.0",
            "0000:d6:08.0",
        ],
    }
    keys = island_keys(new)
    assert keys["0000:d1:00.0"] == keys["0000:d4:00.0"] == "0000:c6:04.0"
    assert keys["0000:d9:00.0"] == keys["0000:dc:00.0"] == "0000:c6:08.0"
    old = {
        "0000:ce:00.0": ["0000:c0:01.1", "0000:c1:00.0"],
        "0000:0a:00.0": ["0000:00:01.1", "0000:01:00.0"],
    }
    assert set(island_keys(old).values()) == {"0000:c0:01.1", "0000:00:01.1"}
    assert island_keys({"0000:d9:00.0": new["0000:d9:00.0"]}) == {
        "0000:d9:00.0": "0000:d6:00.0"
    }
    assert bdf_chain(
        "/sys/devices/pci0000:c0/0000:c0:03.1/0000:c5:00.0/0000:d1:00.0"
    ) == ["0000:c0:03.1", "0000:c5:00.0"]
    assert members_from_keys({0: "x", 1: "x", 2: "y", 3: "y", 4: "y"}) == [
        [0, 1],
        [2, 3, 4],
    ]
    assert parse_islands("0,1;2,3,4,5") == [[0, 1], [2, 3, 4, 5]]
    # patterns on the target layout
    members = parse_islands("0,1;2,3,4,5")
    pats = flow_patterns(members)
    assert pats["intra_A"] == [(0, 1), (1, 0)]
    assert pats["intra_B"] == [(2, 3), (3, 4), (4, 5), (5, 2)]
    assert pats["intra_all"][:2] == [(0, 1), (2, 3)]
    assert pats["cross"] == [(0, 2), (1, 3), (0, 4), (1, 5)]
    assert pats["cross_bidir"][1] == (3, 1)
    assert pats["fanin"] == [(g, 2) for g in (0, 1, 3, 4)]
    assert all(len(v) <= MAX_LADDER for v in pats.values())
    assert set(flow_patterns([[2, 3, 4, 5]])) == {"intra_A", "fanin", "fanout"}
    assert (0, 2) in representative_pairs(members) and (2, 0) in representative_pairs(
        members
    )
    groups = a2a_groups(members, 6)
    assert groups["flat_6"] == [[0, 1, 2, 3, 4, 5]] and groups["mixed_4"] == [
        [0, 1, 2, 3]
    ]
    assert groups["intra_concurrent"] == [[0, 1], [2, 3, 4, 5]] and groups[
        "island1_4"
    ] == [[2, 3, 4, 5]]
    # collapse verdicts: the old audit must read as a collapse at k=3
    old_v = ladder_verdict(OLD_CROSS_ROOT_GBPS)
    assert old_v["verdict"] == "COLLAPSE" and old_v["at_k"] == 3
    assert ladder_verdict([14.0, 28.0, 42.0, 56.0])["verdict"] == "LINEAR"
    assert ladder_verdict([14.0, 14.5, 14.2, 14.1])["verdict"] == "SATURATED"
    assert ladder_verdict([14.0])["verdict"] == "LINEAR"
    # probe line parsing
    rec = parse_probe_line(
        "CURVE mode=push src=0 dst=1 size=4096 blocks=1 n=500 p50_us=7.5 p99_us=9.1"
    )
    assert (
        rec and rec["kind"] == "CURVE" and rec["p50_us"] == 7.5 and rec["size"] == 4096
    )
    rec = parse_probe_line("A2A payload=4096 group=0,1,2,3 step_us=10.4")
    assert rec and rec["group"] == "0,1,2,3" and rec["step_us"] == 10.4
    rec = parse_probe_line("LAT 0<->1 payload=0 one_way_us=6.9")
    assert rec and rec["labels"] == ["0<->1"]
    assert parse_probe_line("### lat 10 0 0:1") is None and parse_probe_line("") is None
    cmds = probe_plan(members, 6)
    assert cmds[0][1].startswith("lat 2000 0 ") and any(
        c.startswith("a2a 500 4096 0,1,2,3,4,5") for _, c in cmds
    )
    assert any(c.startswith("a2a 500 4096 0,1 2,3,4,5") for _, c in cmds)
    plan = plan_cases(members, 6)
    assert plan["cells"]["pair"] > 0 and plan["est_total_s"] > 0
    # lease logic and quiet window on temp directories
    with tempfile.TemporaryDirectory() as tmp:
        lease, quiet = os.path.join(tmp, "lease"), os.path.join(tmp, "quiet")
        os.makedirs(lease)
        for name in ("Me.b4", "Other.a2", "Me.zz"):
            open(os.path.join(lease, name), "w", encoding="utf-8").close()
        assert (
            held_slots("Me", lease) == ["b4", "zz"]
            and held_slots("Nobody", lease) == []
        )
        ok = lease_problems(
            "Me", lease, SLOT_BUSES, {0: "d9", 1: "dc", 2: "df", 3: "e2"}
        )
        assert ok == ["slot 'zz' has no known buses (use --slot)"], ok
        bad = lease_problems(
            "Me", lease, {"b4": SLOT_BUSES["b4"], "zz": ()}, {0: "d9", 1: "d1"}
        )
        assert bad == ["HIP 1 is bus d1, outside held slots ['b4', 'zz']"], bad
        assert (
            "no lease file" in lease_problems("Nobody", lease, SLOT_BUSES, {0: "d9"})[0]
        )
        now = [0.0]
        with Guard("Me", quiet, max_minutes=1.0, clock=lambda: now[0]) as g:
            assert os.path.exists(os.path.join(quiet, "Me"))
            now[0] = 30.0
            assert g.tick()
            now[0] = 61.0
            assert not g.tick()
        assert not os.path.exists(os.path.join(quiet, "Me"))
        # fake sysfs: realpath chain, vendor/class scan, link and AER files
        root = os.path.join(tmp, "sys")
        real = os.path.join(
            root,
            "devices",
            "pci0000:c0",
            "0000:c0:03.1",
            "0000:c5:00.0",
            "0000:d1:00.0",
        )
        os.makedirs(real)
        flat = os.path.join(root, "bus", "pci", "devices")
        os.makedirs(flat)
        os.symlink(real, os.path.join(flat, "0000:d1:00.0"))
        for bridge, width in (("0000:c0:03.1", "16"), ("0000:c5:00.0", "8")):
            adir = real.split(bridge)[0] + bridge
            os.symlink(adir, os.path.join(flat, bridge))
            for fname, val in (
                ("current_link_speed", "16.0 GT/s PCIe"),
                ("current_link_width", width),
                ("max_link_width", "16"),
            ):
                with open(os.path.join(adir, fname), "w", encoding="utf-8") as f:
                    f.write(val + "\n")
        for fname, val in (
            ("vendor", "0x1002"),
            ("class", "0x038000"),
            ("current_link_speed", "16.0 GT/s PCIe"),
            ("current_link_width", "16"),
            ("max_link_width", "16"),
            ("aer_dev_correctable", "RxErr 2\nBadTLP 1\nTOTAL_ERR_COR 3\n"),
        ):
            with open(os.path.join(real, fname), "w", encoding="utf-8") as f:
                f.write(val + "\n")
        topo = discover_topology(flat)
        assert [g["bdf"] for g in topo["gpus"]] == ["0000:d1:00.0"]
        assert topo["gpus"][0]["chain"] == ["0000:c0:03.1", "0000:c5:00.0"]
        assert abs(topo["gpus"][0]["link_raw_gbs"] - 31.5) < 0.01
        narrow = topo["gpus"][0]["narrowest"]
        assert narrow["bdf"] == "0000:c5:00.0" and narrow["width"] == "8", narrow
        assert abs(narrow["raw_gbs"] - 15.754) < 0.01
        snap = aer_snapshot(["0000:d1:00.0"], flat)
        assert snap["0000:d1:00.0:correctable"] == 3 and aer_delta(snap, snap) == {}
        assert aer_delta({}, snap) == {"0000:d1:00.0:correctable": 3}
        # fake p2p_probe: probe_devices, ProbeSession, curve table, stage_probe
        fake = os.path.join(tmp, "p2p_probe")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE_PROBE)
        os.chmod(fake, 0o755)
        devs = probe_devices(fake, dict(os.environ))
        assert [(d["idx"], d["bus"]) for d in devs] == [(0, "d9"), (1, "dc")], devs
        session = ProbeSession(fake, dict(os.environ))
        assert session.ready["devices"] == 2
        recs = session.run("curve push 100 0 4096 0:1")
        session.close()
        assert [r["kind"] for r in recs] == ["CURVE", "END"], recs
        rows = curve_table(recs)
        assert rows[0] == ["mode pair", "4 KiB"] and rows[1][0] == "push 0>1"
        assert rows[1][1] == f"{gbps(4096, 8.0):.1f}"
        with Guard("", quiet, enabled=False) as g:
            out = stage_probe([[0, 1]], 2, argparse.Namespace(probe_bin=fake), g)
        assert out["ready"]["kind"] == "READY" and any(
            r["kind"] == "CURVE" for r in out["records"]
        )
        miss = argparse.Namespace(probe_bin=os.path.join(tmp, "absent"))
        with Guard("", quiet, enabled=False) as g:
            assert "skipped" in stage_probe([[0, 1]], 2, miss, g)
        # argument helpers
        assert slot_map(["x=AA, bb"])["x"] == ("aa", "bb") and "b4" in slot_map([])
        ns = argparse.Namespace(islands="", gpus=None)
        assert default_members(ns) == [[0, 1], [2, 3, 4, 5]]
        assert default_members(argparse.Namespace(islands="", gpus=4)) == [[0, 1, 2, 3]]
        assert default_members(argparse.Namespace(islands="0;1,2", gpus=None)) == [
            [0],
            [1, 2],
        ]
    print("selftest ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="CPU-only checks of the pure logic")
    for name in ("plan", "topo", "pair", "flows", "a2a", "probe", "all"):
        s = sub.add_parser(
            name, help=f"{name} stage" if name != "all" else "every stage"
        )
        s.add_argument(
            "--gpus", type=int, default=None, help="use the first N visible GPUs"
        )
        s.add_argument(
            "--islands", default="", help="override islands, e.g. '0,1;2,3,4,5'"
        )
        if name == "plan":
            continue
        s.add_argument("--owner", default="", help="lease/quiet holder name")
        s.add_argument("--lease-dir", default=LEASE_DIR)
        s.add_argument("--quiet-dir", default=QUIET_DIR)
        s.add_argument(
            "--slot",
            action="append",
            default=[],
            metavar="NAME=BUS,BUS",
            help="extra or overriding slot -> PCI bus mapping",
        )
        s.add_argument("--no-lease-check", action="store_true")
        s.add_argument(
            "--no-quiet", action="store_true", help="do not touch the quiet file"
        )
        s.add_argument(
            "--ignore-build", action="store_true", help="ignore a running docker build"
        )
        s.add_argument("--max-minutes", type=float, default=14.0)
        s.add_argument(
            "--dry-run", action="store_true", help="print the plan, touch nothing"
        )
        s.add_argument("--out", default="", help="write the JSON result here")
        s.add_argument("--reps", type=int, default=7)
        s.add_argument("--lat-samples", type=int, default=200)
        s.add_argument(
            "--sizes",
            type=lambda t: [parse_size(x) for x in t.split(",")],
            default=None,
        )
        s.add_argument(
            "--flow-sizes",
            type=lambda t: [parse_size(x) for x in t.split(",")],
            default=None,
        )
        s.add_argument(
            "--a2a-chunks",
            type=lambda t: [parse_size(x) for x in t.split(",")],
            default=None,
        )
        s.add_argument(
            "--pairs",
            dest="pair_list",
            default=None,
            type=lambda t: [
                (int(a), int(b)) for a, b in (x.split(">") for x in t.split(","))
            ],
            help="ordered pairs such as 0>1,2>3 (default: representative set)",
        )
        here = os.path.dirname(os.path.abspath(__file__))
        s.add_argument(
            "--probe-bin", default=os.path.join(here, "..", "p2p_probe", "p2p_probe")
        )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "plan":
        return run_plan(args)
    return run_gpu(args)


if __name__ == "__main__":
    sys.exit(main())
