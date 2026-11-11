#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Is every KFD host-memory buffer of a worker locked? (read-only /proc)

    python3 pinned_fdinfo_check.py PID [PID ...] [--max-unlocked-kib N]
                                   [--every SECONDS [--count N]]

The amdgpu driver reports, in the fdinfo of each render-node file descriptor, the
size of the system-memory buffer objects of the process in that GPU's VM
(``drm-total-cpu``): userptr buffers (paged ``hipHostMalloc`` memory and the
exact-size PLE table) are the ones that matter, they show up there and nowhere
else. This tool sets, for each worker:

  * ``drm-total-cpu`` of its own GPU (the largest of the GPUs it has open),
  * the sum of its ``rw-s /dev/zero (deleted)`` mappings (``/proc/PID/maps``),
  * ``VmLck`` (``/proc/PID/status``).

On par1-cs25 the first two are equal to the KiB (4 of 4 workers of a TP4
server, 659-663 mappings), and NCCL's ``/dev/shm`` segments are in neither. So
``total - VmLck`` is the host memory that KFD can evict the worker for and that
nothing locks. With ``--max-unlocked-kib`` the exit status is 1 when it is
larger for any worker (blocks that appeared in the last 2 s are not locked yet:
leave some room). With ``--every`` it prints one line per worker and tick
instead of the table, until a worker exits or ``--count`` ticks are done.

It reads only ``/proc/PID/{fdinfo,maps,status}`` and costs a few milliseconds.
"""

import argparse
import glob
import os
import sys
import time
from typing import Any

PINNED_PERMS = "rw-s"
PINNED_PATH = "/dev/zero (deleted)"
_UNITS = {"": 1, "KiB": 1024, "MiB": 1024 * 1024}


def size_bytes(value: str) -> int:
    """``25387360 KiB`` -> bytes. The kernel prints bytes, KiB or MiB."""
    fields = value.split()
    unit = fields[1] if len(fields) > 1 else ""
    return int(fields[0]) * _UNITS[unit]


def parse_fdinfo(text: str) -> tuple[str, int] | None:
    """``(PCI address, drm-total-cpu in bytes)`` of a DRM fdinfo, else None."""
    pdev = total = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "drm-pdev":
            pdev = value
        elif key == "drm-total-cpu":
            total = size_bytes(value)
    if pdev is None or total is None:
        return None
    return pdev, total


def pinned_mappings(maps_text: str) -> tuple[int, int]:
    """``(count, bytes)`` of the ``rw-s /dev/zero (deleted)`` mappings."""
    count = size = 0
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or fields[1] != PINNED_PERMS:
            continue
        if fields[5].strip() != PINNED_PATH:
            continue
        start, end = (int(x, 16) for x in fields[0].split("-"))
        count += 1
        size += end - start
    return count, size


def vmlck_bytes(status_text: str) -> int:
    for line in status_text.splitlines():
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024
    raise ValueError("no VmLck line")


def check(pid: int, proc: str = "/proc") -> dict[str, Any]:
    if not os.path.isdir(f"{proc}/{pid}"):
        raise FileNotFoundError(f"pid {pid} is gone")
    per_gpu: dict[str, int] = {}
    for path in glob.glob(f"{proc}/{pid}/fdinfo/*"):
        try:
            with open(path) as f:
                parsed = parse_fdinfo(f.read())
        except OSError:
            continue
        if parsed:
            per_gpu[parsed[0]] = parsed[1]
    if not per_gpu:
        raise ValueError(f"pid {pid}: no DRM fdinfo with drm-total-cpu")
    own = max(per_gpu, key=lambda k: per_gpu[k])
    with open(f"{proc}/{pid}/maps") as f:
        count, mapped = pinned_mappings(f.read())
    with open(f"{proc}/{pid}/status") as f:
        locked = vmlck_bytes(f.read())
    return {
        "pid": pid,
        "own_gpu": own,
        "total_cpu": per_gpu[own],
        "peer_total_cpu": sorted(v for k, v in per_gpu.items() if k != own),
        "mappings": count,
        "mapped": mapped,
        "vmlck": locked,
        "unlocked": per_gpu[own] - locked,
        "unmapped_diff": per_gpu[own] - mapped,
    }


def render(rows: list[dict[str, Any]]) -> str:
    kib = 1024
    header = (
        "| pid | own GPU | drm-total-cpu KiB | rw-s zero mappings (KiB) | "
        "total - mappings KiB | VmLck KiB | total - VmLck KiB |"
    )
    out = [header, "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| {r['pid']} | {r['own_gpu']} | {r['total_cpu'] // kib} | "
            f"{r['mappings']} ({r['mapped'] // kib}) | {r['unmapped_diff'] // kib} | "
            f"{r['vmlck'] // kib} | {r['unlocked'] // kib} |"
        )
    return "\n".join(out) + "\n"


def series_line(now: float, r: dict[str, Any]) -> str:
    stamp = time.strftime("%H:%M:%S", time.localtime(now))
    return (
        f"{stamp} pid={r['pid']} total_kib={r['total_cpu'] // 1024} "
        f"mappings={r['mappings']} vmlck_kib={r['vmlck'] // 1024} "
        f"unlocked_kib={r['unlocked'] // 1024}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pid", type=int, nargs="+")
    ap.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    ap.add_argument(
        "--max-unlocked-kib",
        type=int,
        help="exit 1 when total - VmLck exceeds this for any worker",
    )
    ap.add_argument(
        "--every",
        type=float,
        metavar="SECONDS",
        help="repeat, one line per worker and tick, until a worker exits",
    )
    ap.add_argument(
        "--count", type=int, default=0, help="with --every: stop after N ticks"
    )
    args = ap.parse_args(argv)
    limit = None if args.max_unlocked_kib is None else args.max_unlocked_kib * 1024
    exceeded = False
    ticks = 0
    while True:
        try:
            rows = [check(pid, args.proc) for pid in args.pid]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            if ticks == 0:
                raise
            break  # a worker exited while the series ran: it is over
        ticks += 1
        if args.every is None:
            sys.stdout.write(render(rows))
        else:
            now = time.time()
            for r in rows:
                sys.stdout.write(series_line(now, r) + "\n")
            sys.stdout.flush()
        exceeded = exceeded or (
            limit is not None and any(r["unlocked"] > limit for r in rows)
        )
        if args.every is None or (args.count and ticks >= args.count):
            break
        time.sleep(args.every)
    return int(exceeded)


if __name__ == "__main__":
    sys.exit(main())
