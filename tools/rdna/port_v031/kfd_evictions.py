#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KFD queue-eviction time of each vLLM worker under a server process tree.

While the kernel driver has a process's GPU queues evicted (a KFD userptr
invalidation, memory pressure, a neighbour's boot), that process runs
nothing. The TP peers then wait for it, so every latency measured over that
time says something about the host, not about the code under test. KFD
accumulates the evicted time per process in
``/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms``, which reads
the same in every ``stats_*`` of a process. A measurement window that
overlaps an eviction is INVALID and has to be rerun.

  kfd_evictions.py ROOT_PID --save before.json   # snapshot before a window
  kfd_evictions.py ROOT_PID --since before.json  # one line: per-worker
                                                 # jump, then ok or INVALID

Workers are the descendants of ROOT_PID whose process title is
``VLLM::Worker_...``; each is labelled by that title (``Worker_TP0`` ...).
``Sampler`` records samples in the background, for windows that are only
known afterwards (stall_probe.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from collections.abc import Callable

PROC = "/proc"
KFD_PROC = "/sys/class/kfd/kfd/proc"
TITLE = "VLLM::"


def worker_pids(root_pid: int, proc: str = PROC) -> dict[str, int]:
    """{title without ``VLLM::``: pid} of the vLLM workers under root_pid."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            with open(f"{proc}/{entry}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        # The command name may contain spaces and parens; ppid follows the last ")".
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    out: dict[str, int] = {}
    todo = list(children.get(root_pid, []))
    while todo:
        pid = todo.pop()
        todo.extend(children.get(pid, []))
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                title = f.read().split(b"\0")[0].decode(errors="replace").strip()
        except OSError:
            continue
        if title.startswith(TITLE + "Worker"):
            out[title[len(TITLE) :]] = pid
    return out


def evicted_ms(pid: int, kfd_proc: str = KFD_PROC) -> int | None:
    """Total ms KFD kept this process's queues evicted; None without a KFD context."""
    values = []
    for path in glob.glob(f"{kfd_proc}/{pid}/stats_*/evicted_ms"):
        try:
            with open(path) as f:
                values.append(int(f.read().split()[0]))
        except (OSError, ValueError, IndexError):
            continue
    return max(values) if values else None


def snapshot(
    root_pid: int, proc: str = PROC, kfd_proc: str = KFD_PROC
) -> dict[str, int]:
    out = {}
    for label, pid in worker_pids(root_pid, proc).items():
        ms = evicted_ms(pid, kfd_proc)
        if ms is not None:
            out[label] = ms
    return out


def jumps(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Evicted ms gained per worker present in both snapshots."""
    return {w: after[w] - before[w] for w in sorted(after) if w in before}


def verdict_line(delta: dict[str, int]) -> str:
    if not delta:
        return "evicted_ms: no vLLM workers with a KFD context found"
    per = " ".join(f"{w}=+{ms}" for w, ms in delta.items())
    if any(ms > 0 for ms in delta.values()):
        return (
            f"evicted_ms {per} -> INVALID (KFD evicted a worker's GPU queues "
            "during this window; rerun it)"
        )
    return f"evicted_ms {per} -> ok"


class Sampler:
    """Samples every worker's evicted_ms each ``period_s`` in a thread.

    ``now`` is the caller's clock, so windows and samples share a time base.
    """

    def __init__(
        self,
        root_pid: int,
        now: Callable[[], float],
        period_s: float = 0.5,
        read: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self.now = now
        self.period_s = period_s
        self.read = read or (lambda: snapshot(root_pid))
        self.samples: list[tuple[float, dict[str, int]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        self.samples.append((self.now(), self.read()))

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def totals(self) -> dict[str, int]:
        if not self.samples:
            return {}
        return jumps(self.samples[0][1], self.samples[-1][1])

    def overlap_ms(self, t0: float, t1: float) -> dict[str, int]:
        """Per worker, the evicted ms that may overlap [t0, t1].

        A jump of D ms between samples at ta < tb is an eviction that ended
        in (ta, tb] after lasting D ms, so it lies within [ta - D, tb].
        """
        out: dict[str, int] = {}
        for (ta, a), (tb, b) in zip(self.samples, self.samples[1:]):
            for w, ms in jumps(a, b).items():
                if ms > 0 and ta - ms / 1000.0 <= t1 and tb >= t0:
                    out[w] = out.get(w, 0) + ms
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root_pid", type=int, help="server (or serve script) pid")
    ap.add_argument("--save", help="write the snapshot to this JSON file")
    ap.add_argument("--since", help="report the jump since this saved snapshot")
    args = ap.parse_args(argv)
    now = snapshot(args.root_pid)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(now, f)
    if args.since:
        with open(args.since) as f:
            print(verdict_line(jumps(json.load(f), now)))
    elif not args.save:
        print(json.dumps(now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
