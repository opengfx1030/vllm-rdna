# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KFD per-process queue evictions (``/sys/class/kfd/kfd/proc``), sysfs only.

KFD keeps ``proc/<pid>/stats_<gpu_id>/evicted_ms`` (cumulative milliseconds the
process's queues on that GPU were evicted) and ``proc/<pid>/queues/<n>/gpuid``
for every queue.  The box rules invalidate a measured window when an eviction
counter jumps, so the collector reports ``kfd.evicted_ms`` for every process that
has at least one queue on a GPU (tags ``pid`` and ``comm``).  ``kfd_pids`` and
``evicted_ms`` are public helpers for the active probe's guard.
"""

from __future__ import annotations

from collections.abc import Iterable

from pciescope.collectors import BaseCollector
from pciescope.model import Gpu, Sample, Topology
from pciescope.model import now_ns as _now_ns
from pciescope.sysfs import Sysfs

NAME = "kfd"
KFD_PROC = "class/kfd/kfd/proc"


def _pids(sysfs: Sysfs) -> list[int]:
    return sorted(int(p) for p in sysfs.listdir(KFD_PROC) if p.isdigit())


def queue_gpu_ids(sysfs: Sysfs, pid: int) -> dict[int, int]:
    """``{gpu_id: number of queues}`` of one KFD process."""
    out: dict[int, int] = {}
    base = f"{KFD_PROC}/{pid}/queues"
    for q in sysfs.listdir(base):
        gid = sysfs.read_int(f"{base}/{q}/gpuid")
        if gid is not None:
            out[gid] = out.get(gid, 0) + 1
    return out


def kfd_pids(sysfs: Sysfs, gpu: Gpu, *, active_only: bool = True) -> list[int]:
    """PIDs with a KFD context on ``gpu``.

    ``active_only`` keeps those with at least one queue on that GPU; otherwise
    every process that has a ``stats_<gpu_id>`` directory (all KFD processes get
    one per GPU, active or not).
    """
    if gpu.kfd_gpu_id is None:
        return []
    out: list[int] = []
    for pid in _pids(sysfs):
        if active_only:
            if gpu.kfd_gpu_id in queue_gpu_ids(sysfs, pid):
                out.append(pid)
        elif sysfs.isdir(f"{KFD_PROC}/{pid}/stats_{gpu.kfd_gpu_id}"):
            out.append(pid)
    return out


def evicted_ms(
    sysfs: Sysfs, gpu: Gpu, pids: Iterable[int] | None = None
) -> dict[int, int]:
    """``{pid: cumulative evicted_ms}`` on ``gpu`` (all KFD processes by default)."""
    if gpu.kfd_gpu_id is None:
        return {}
    out: dict[int, int] = {}
    for pid in _pids(sysfs) if pids is None else pids:
        val = sysfs.read_int(f"{KFD_PROC}/{pid}/stats_{gpu.kfd_gpu_id}/evicted_ms")
        if val is not None:
            out[pid] = val
    return out


def process_name(pid: int, procfs: Sysfs | None = None) -> str:
    """``/proc/<pid>/comm`` (empty when the process is gone)."""
    return (procfs or Sysfs("/proc")).read(f"{pid}/comm") or ""


class KfdCollector(BaseCollector):
    """``kfd.evicted_ms`` per (GPU, process with a queue on it)."""

    name = NAME

    def __init__(
        self,
        topo: Topology,
        sysfs: Sysfs | None = None,
        *,
        procfs: Sysfs | None = None,
        all_processes: bool = False,
    ) -> None:
        super().__init__(topo, sysfs)
        self.procfs = procfs or Sysfs("/proc")
        self.all_processes = all_processes
        self._by_gpu_id = {
            g.kfd_gpu_id: g for g in topo.gpu_list("bus") if g.kfd_gpu_id is not None
        }
        self._comm: dict[int, str] = {}

    def available(self) -> tuple[bool, str]:
        if not self._by_gpu_id:
            return False, "no GPU has a KFD gpu_id (KFD topology not readable)"
        if not self.sysfs.isdir(KFD_PROC):
            return False, f"{KFD_PROC} missing (no amdkfd)"
        return True, ""

    def _name(self, pid: int) -> str:
        name = self._comm.get(pid)
        if name is None:
            name = process_name(pid, self.procfs)
            if name:
                self._comm[pid] = name
        return name

    def collect(self) -> list[Sample]:
        ts = _now_ns()
        out: list[Sample] = []
        for pid in _pids(self.sysfs):
            gids = (
                set(self._by_gpu_id)
                if self.all_processes
                else set(queue_gpu_ids(self.sysfs, pid)) & set(self._by_gpu_id)
            )
            for gid in sorted(gids):
                g = self._by_gpu_id[gid]
                val = self.sysfs.read_int(f"{KFD_PROC}/{pid}/stats_{gid}/evicted_ms")
                if val is None:
                    continue
                tags = {"pid": str(pid), "comm": self._name(pid)}
                out.append(Sample(ts, g.id, "kfd.evicted_ms", float(val), "ms", tags))
        return out


def make(
    topo: Topology,
    *,
    sysfs: Sysfs | None = None,
    procfs: Sysfs | None = None,
    all_processes: bool = False,
    **_opts: object,
) -> KfdCollector:
    return KfdCollector(topo, sysfs, procfs=procfs, all_processes=all_processes)
