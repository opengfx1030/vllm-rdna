# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is a switch idle?  The gate before its traffic counters are enabled.

The counter writes land in the registers of live GPU links, so they are only
made while nothing uses the GPUs below the switch (user approval, 2026-11-07).
``check_idle`` refuses unless, for every GPU below the switch:

* no process has a KFD queue on it (``/sys/class/kfd/kfd/proc/<pid>/queues/*/gpuid``
  mapped through the GPU's ``kfd_gpu_id``; every KFD process lists a ``stats_*``
  directory for *all* GPUs, so that is no evidence of use),
* it holds no GPU memory worth speaking of and reports 0 % busy (read from the
  amdgpu sysfs files; a runtime-suspended GPU is not woken, it counts as idle),

and the lease directory shows the caller as holder of the lease of every V620 below
the switch and nobody else (``pciescope.probe.lease``: both ``a2`` and ``b4`` on the
SlimSAS card, which carries all six GPUs), and no holder at all for a W7800 of the
switch.  Everything here is read-only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pciescope.model import Gpu, Topology, full_bdf
from pciescope.probe import lease
from pciescope.probe.hw import LEASE_KEYS, SLOT_KEYS
from pciescope.sysfs import Sysfs, is_awake

#: GPU memory above this many MiB means somebody uses the GPU (idle cards: < 200)
VRAM_IDLE_MIB = 1024.0
_KFD_PROCS = "class/kfd/kfd/proc"


@dataclass(frozen=True, slots=True)
class GpuIdle:
    """What was seen on one GPU."""

    gpu: str
    bus: str
    hip: int | None
    asleep: bool
    vram_used_mib: float | None
    busy_pct: tuple[int, ...]
    pids: tuple[tuple[int, str], ...]
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "bus": self.bus,
            "hip": self.hip,
            "asleep": self.asleep,
            "vram_used_mib": self.vram_used_mib,
            "busy_pct": list(self.busy_pct),
            "pids": [{"pid": p, "cmd": c} for p, c in self.pids],
            "problems": list(self.problems),
        }


@dataclass(frozen=True, slots=True)
class IdleReport:
    """Verdict of ``check_idle`` for one switch."""

    switch: str
    owner: str | None
    ok: bool
    gpus: tuple[GpuIdle, ...]
    problems: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "switch": self.switch,
            "owner": self.owner,
            "ok": self.ok,
            "gpus": [g.to_dict() for g in self.gpus],
            "problems": list(self.problems),
        }

    def text(self) -> str:
        lines = [
            f"switch {self.switch} idle check (owner {self.owner or '-'}): "
            + ("IDLE" if self.ok else "NOT IDLE")
        ]
        for g in self.gpus:
            if g.asleep:
                state = "asleep"
            elif g.vram_used_mib is None:
                state = "unreadable"
            else:
                busy = max(g.busy_pct, default=0)
                state = f"vram {g.vram_used_mib:.0f} MiB busy {busy}%"
            who = ", ".join(f"{p} {c}" for p, c in g.pids) or "no KFD queues"
            lines.append(f"  {g.gpu} hip {g.hip}: {state}; {who}")
        lines += [f"  PROBLEM {p}" for p in self.problems]
        return "\n".join(lines)


def kfd_queue_owners(sysfs: Sysfs) -> dict[int, list[int]]:
    """``{kfd gpu_id: [pid, ...]}`` of the processes with a queue on that GPU."""
    owners: dict[int, set[int]] = {}
    for pid in sysfs.listdir(_KFD_PROCS):
        if not pid.isdigit():
            continue
        base = f"{_KFD_PROCS}/{pid}/queues"
        for queue in sysfs.listdir(base):
            gid = sysfs.read_int(f"{base}/{queue}/gpuid")
            if gid is not None:
                owners.setdefault(gid, set()).add(int(pid))
    return {gid: sorted(pids) for gid, pids in owners.items()}


def _cmdline(proc_root: Path, pid: int) -> str:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return "?"
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:100] or "?"


def _lease_problems(
    gpus: Sequence[Gpu], owner: str | None, directory: Path | None
) -> list[str]:
    """Lease refusals of the V620 groups and holders of the switch's W7800 keys."""
    key_of = {b: k for k, buses in LEASE_KEYS.items() for b in buses}
    slot_buses = [g.bus for g in gpus if key_of.get(g.bus) in SLOT_KEYS]
    problems: list[str] = []
    if slot_buses:
        chk = lease.check_leases(slot_buses, owner, directory)
        problems += [line for line in chk.reason.splitlines() if line]
    holders = lease.list_holders(directory)
    for g in gpus:
        key = key_of.get(g.bus)
        if key is None or key in SLOT_KEYS:
            continue
        who = [h.owner for h in holders if h.key == key]
        if who:
            problems.append(f"{key} (bus {g.bus}) is held by {', '.join(who)}")
    return problems


def check_idle(
    topo: Topology,
    switch: str,
    owner: str | None,
    *,
    sysfs: Sysfs | None = None,
    proc_root: Path = Path("/proc"),
    lease_directory: Path | None = None,
    busy_samples: int = 3,
    busy_interval_s: float = 0.2,
    vram_idle_mib: float = VRAM_IDLE_MIB,
    sleep: Callable[[float], None] = time.sleep,
) -> IdleReport:
    """Check that nothing runs below ``switch`` and that ``owner`` holds its lease."""
    sysfs = sysfs or Sysfs()
    gpus = sorted(
        (g for g in topo.gpus.values() if g.switch == switch),
        key=lambda g: (g.hip is None, g.hip, g.bus),
    )
    problems: list[str] = []
    if not gpus:
        problems.append(f"no GPU below switch {switch} in the topology")
    problems += _lease_problems(gpus, owner, lease_directory)
    owners = kfd_queue_owners(sysfs)
    rows: list[GpuIdle] = []
    for g in gpus:
        bdf = full_bdf(g.bdf)
        found: list[str] = []
        asleep = not is_awake(sysfs.runtime_status(bdf))
        vram: float | None = None
        busy: list[int] = []
        if not asleep:
            used = sysfs.read_int(f"bus/pci/devices/{bdf}/mem_info_vram_used")
            if used is None:
                found.append("memory use unreadable")
            else:
                vram = used / (1 << 20)
                if vram > vram_idle_mib:
                    found.append(f"{vram:.0f} MiB of GPU memory in use")
            for i in range(max(busy_samples, 1)):
                if i:
                    sleep(busy_interval_s)
                pct = sysfs.read_int(f"bus/pci/devices/{bdf}/gpu_busy_percent")
                if pct is not None:
                    busy.append(pct)
            if not busy:
                found.append("gpu_busy_percent unreadable")
            elif any(busy):
                found.append(f"busy {max(busy)}%")
        pids: list[tuple[int, str]] = []
        if g.kfd_gpu_id is None:
            found.append("no KFD gpu_id: cannot map processes to it")
        else:
            pids = [(p, _cmdline(proc_root, p)) for p in owners.get(g.kfd_gpu_id, [])]
            if pids:
                found.append("KFD queues of " + ", ".join(f"pid {p}" for p, _ in pids))
        rows.append(
            GpuIdle(
                g.id, g.bus, g.hip, asleep, vram, tuple(busy), tuple(pids), tuple(found)
            )
        )
        problems += [f"{g.id} (hip {g.hip}): {f}" for f in found]
    return IdleReport(switch, owner, not problems, tuple(rows), tuple(problems))
