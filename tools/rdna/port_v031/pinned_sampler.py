#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What reclaims or moves the pinned host memory of vLLM workers (run as root).

    sudo python3 pinned_sampler.py --out DIR [--hours 1] [--period 1]
        [--root-pid SERVER_PID]

Needs numpy (the vLLM venv's python has it) and root: pagemap shows PFNs, and
so shows moved pages, only to CAP_SYS_ADMIN.

Every ``--period`` seconds, for each ``VLLM::Worker`` process, it appends to
``DIR/samples.jsonl``:

  * ``/proc/<pid>/status``: VmLck, VmSwap, RssAnon, RssFile, RssShmem (no
    page-table walk) and the KFD ``evicted_ms``;
  * host reclaim: the ``/proc/vmstat`` counters that matter (kswapd and direct
    scan and steal, swap-out and -in, compaction, migration), ``/proc/meminfo``,
    memory pressure (PSI), zram, KFD's system-memory total, and the CPU time of
    kswapd and kcompactd;
  * the worker's mappings by class: ``pinned`` is what hipHostMalloc leaves
    (shared anonymous, ``rw-s /dev/zero (deleted)``), ``anon`` is a large
    private anonymous mapping (where hipHostRegister'd memory would be),
    ``shm`` is a ``/dev/shm`` or memfd file. Count, GiB, and for the pinned
    ones how much is locked.

What moved comes from ``/proc/<pid>/pagemap``: each mapping's entries are read
and compared with the previous read, page by page. ``DIR/pagemap.jsonl`` gets
one record for every mapping where a page moved (present with another PFN:
migrated, or swapped out and back between two reads), was swapped out or was
dropped. Pinned mappings under ``--big-gib`` are read every tick; the PLE
table and the other classes every ``--slow-period`` seconds. Every
``--lock-period`` seconds a few pages of each mapping are looked up in
``/proc/kpageflags`` to say whether it is mlocked/unevictable or on the LRU.

Read-only: it opens files under /proc, /sys/class/kfd and /sys/kernel/debug/kfd
(``mem_limit`` only) and writes nothing but its own output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
from typing import Any

import numpy as np

PAGE = os.sysconf("SC_PAGE_SIZE")
PM_PRESENT = 1 << 63
PM_SWAPPED = 1 << 62
PM_PFN_MASK = (1 << 55) - 1
KPF_LRU = 5
KPF_UNEVICTABLE = 18
KPF_MLOCKED = 33
# pagemap entries read at a time: 512k pages = 2 GiB of address space.
CHUNK_PAGES = 1 << 19

PROC = "/proc"
KFD_PROC = "/sys/class/kfd/kfd/proc"
KFD_MEM_LIMIT = "/sys/kernel/debug/kfd/mem_limit"
TITLE = "VLLM::Worker"
GIB = 1 << 30

VMSTAT_KEYS = frozenset(
    (
        "pgscan_kswapd",
        "pgscan_direct",
        "pgscan_khugepaged",
        "pgsteal_kswapd",
        "pgsteal_direct",
        "pgsteal_khugepaged",
        "pswpin",
        "pswpout",
        "pgmigrate_success",
        "pgmigrate_fail",
        "compact_migrate_scanned",
        "compact_free_scanned",
        "compact_isolated",
        "compact_stall",
        "compact_fail",
        "compact_success",
        "compact_daemon_wake",
        "compact_daemon_migrate_scanned",
        "allocstall_normal",
        "allocstall_movable",
        "allocstall_dma32",
        "pageoutrun",
        "kswapd_low_wmark_hit_quickly",
        "kswapd_high_wmark_hit_quickly",
        "pgrefill",
        "pgdeactivate",
        "pgactivate",
        "pgpgin",
        "pgpgout",
        "pgmajfault",
        "workingset_refault_anon",
        "workingset_refault_file",
        "workingset_activate_anon",
        "workingset_activate_file",
        "unevictable_pgs_mlocked",
        "unevictable_pgs_munlocked",
        "unevictable_pgs_culled",
        "unevictable_pgs_rescued",
        "thp_collapse_alloc",
        "thp_split_page",
        "numa_pages_migrated",
        "oom_kill",
        "nr_free_pages",
        "nr_shmem",
        "nr_mlock",
        "nr_unevictable",
        "nr_active_anon",
        "nr_inactive_anon",
        "nr_active_file",
        "nr_inactive_file",
    )
)
MEMINFO_KEYS = (
    "MemFree",
    "MemAvailable",
    "Cached",
    "Shmem",
    "Mlocked",
    "Unevictable",
    "SwapTotal",
    "SwapFree",
    "AnonPages",
    "Dirty",
    "Writeback",
)
STATUS_KEYS = {
    "VmLck": "vmlck_kb",
    "VmPin": "vmpin_kb",
    "VmSwap": "vmswap_kb",
    "VmRSS": "rss_kb",
    "RssAnon": "rss_anon_kb",
    "RssFile": "rss_file_kb",
    "RssShmem": "rss_shmem_kb",
}
SYSCTLS = (
    "vm/swappiness",
    "vm/min_free_kbytes",
    "vm/watermark_scale_factor",
    "vm/watermark_boost_factor",
    "vm/compact_unevictable_allowed",
    "vm/compaction_proactiveness",
    "vm/vfs_cache_pressure",
    "vm/page-cluster",
    "vm/zone_reclaim_mode",
    "kernel/numa_balancing",
)


def read_text(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def parse_kv(text: str) -> dict[str, int]:
    """``Key:   123 kB`` lines to {Key: 123}."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            with contextlib.suppress(ValueError):
                out[key.strip()] = int(parts[0])
    return out


def vmstat() -> dict[str, int]:
    out = {}
    for line in (read_text(f"{PROC}/vmstat") or "").splitlines():
        key, _, val = line.partition(" ")
        if key in VMSTAT_KEYS:
            out[key] = int(val)
    return out


def meminfo() -> dict[str, int]:
    kv = parse_kv(read_text(f"{PROC}/meminfo") or "")
    return {k: kv[k] for k in MEMINFO_KEYS if k in kv}


def psi_memory() -> dict[str, int]:
    """Memory stall time in us since boot: {some, full}."""
    out = {}
    for line in (read_text(f"{PROC}/pressure/memory") or "").splitlines():
        kind, *fields = line.split()
        for f in fields:
            if f.startswith("total="):
                out[kind] = int(f[6:])
    return out


def find_kthreads(names: tuple[str, ...] = ("kswapd0", "kcompactd0")) -> dict[str, int]:
    """{name: pid} of the reclaim kernel threads."""
    out = {}
    for entry in os.listdir(PROC):
        if entry.isdigit():
            comm = read_text(f"{PROC}/{entry}/comm")
            if comm is not None and comm.strip() in names:
                out[comm.strip()] = int(entry)
    return out


def cpu_ticks(pid: int) -> int | None:
    """CPU time (utime + stime) of a process in clock ticks."""
    stat = read_text(f"{PROC}/{pid}/stat")
    if not stat:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def zram_bytes() -> dict[str, int]:
    f = (read_text("/sys/block/zram0/mm_stat") or "").split()
    return {"orig": int(f[0]), "compr": int(f[1]), "used": int(f[2])} if f else {}


def kfd_system_mem_mib() -> int | None:
    text = read_text(KFD_MEM_LIMIT)
    if text and text.startswith("System mem used"):
        return int(text.split()[3].rstrip("M"))
    return None


def sysctls() -> dict[str, str]:
    out = {}
    for name in SYSCTLS:
        text = read_text(f"{PROC}/sys/{name}")
        if text is not None:
            out[name.replace("/", ".")] = text.strip()
    for path in (
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/sys/kernel/mm/transparent_hugepage/defrag",
        "/sys/kernel/mm/lru_gen/enabled",
        "/sys/kernel/mm/lru_gen/min_ttl_ms",
    ):
        text = read_text(path)
        if text is not None:
            out[path.removeprefix("/sys/kernel/mm/")] = text.strip()
    return out


def worker_pids(proc: str = PROC) -> dict[int, str]:
    """{pid: title} of every VLLM::Worker process."""
    out = {}
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            with open(f"{proc}/{entry}/cmdline", "rb") as f:
                cmd = f.read(80).split(b"\0")[0].decode(errors="replace").strip()
        except OSError:
            continue
        if cmd.startswith(TITLE):
            out[int(entry)] = cmd
    return out


def parent_pid(pid: int) -> int | None:
    stat = read_text(f"{PROC}/{pid}/stat")
    return int(stat.rsplit(")", 1)[1].split()[1]) if stat else None


def process_tree(roots: set[int]) -> set[int]:
    """The roots and every process below them."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir(PROC):
        if entry.isdigit():
            ppid = parent_pid(int(entry))
            if ppid is not None:
                children.setdefault(ppid, []).append(int(entry))
    out = set(roots)
    todo = list(roots)
    while todo:
        for child in children.get(todo.pop(), ()):
            if child not in out:
                out.add(child)
                todo.append(child)
    return out


def evicted_ms(pid: int, kfd_proc: str = KFD_PROC) -> int | None:
    """KFD queue-eviction ms of the process (the same in every stats_*)."""
    base = f"{kfd_proc}/{pid}"
    try:
        dirs = [d for d in os.listdir(base) if d.startswith("stats_")]
    except OSError:
        return None
    vals = []
    for d in dirs:
        text = read_text(f"{base}/{d}/evicted_ms")
        if text and text.strip().isdigit():
            vals.append(int(text))
    return max(vals) if vals else None


def process_status(pid: int) -> dict[str, int]:
    kv = parse_kv(read_text(f"{PROC}/{pid}/status") or "")
    return {out: kv[key] for key, out in STATUS_KEYS.items() if key in kv}


def classify(perms: str, path: str, size: int) -> str | None:
    """pinned: what hipHostMalloc leaves. anon: a private anonymous mapping
    big enough to hold hipHostRegister'd memory. shm: a shared /dev/shm or
    memfd file. None: libraries, checkpoint, heap, devices, reservations."""
    if perms == "rw-s" and path == "/dev/zero (deleted)":
        return "pinned"
    if perms == "rw-s" and path.startswith(("/dev/shm/", "/memfd:")):
        return "shm" if size >= 65536 else None
    if perms == "rw-p" and not path and size >= (1 << 20):
        return "anon"
    return None


class Vma:
    """A mapping from /proc/<pid>/maps that the sampler follows."""

    __slots__ = ("start", "end", "path", "inode", "cls")

    def __init__(self, start: int, end: int, path: str, inode: int, cls: str) -> None:
        self.start = start
        self.end = end
        self.path = path
        self.inode = inode
        self.cls = cls

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def key(self) -> tuple[int, int]:
        return (self.start, self.end)


def read_maps(pid: int) -> list[Vma] | None:
    """The mappings of ``pid`` that fall in a sampler class."""
    text = read_text(f"{PROC}/{pid}/maps")
    if text is None:
        return None
    out = []
    for line in text.splitlines():
        f = line.split(None, 5)
        if len(f) < 5:
            continue
        a, _, b = f[0].partition("-")
        start, end = int(a, 16), int(b, 16)
        path = f[5].strip() if len(f) > 5 else ""
        cls = classify(f[1], path, end - start)
        if cls:
            out.append(Vma(start, end, path, int(f[4]), cls))
    return out


def read_pagemap(fd: int, start: int, end: int) -> list[bytes] | None:
    """The pagemap entries of [start, end) as chunks of little-endian u64."""
    npages = (end - start) // PAGE
    off = start // PAGE * 8
    chunks = []
    done = 0
    while done < npages:
        n = min(CHUNK_PAGES, npages - done)
        try:
            b = os.pread(fd, n * 8, off + done * 8)
        except OSError:
            return None
        if len(b) != n * 8:
            return None
        chunks.append(b)
        done += n
    return chunks


def diff_pagemap(prev: list[bytes], cur: list[bytes]) -> dict[str, int]:
    """Per-page changes between two reads of the same mapping."""
    out = {"moved": 0, "swapped_out": 0, "dropped": 0, "swapped_in": 0, "populated": 0}
    for a, b in zip(prev, cur, strict=True):
        if a == b:
            continue
        x = np.frombuffer(a, dtype="<u8")
        y = np.frombuffer(b, dtype="<u8")
        xp = (x & PM_PRESENT) != 0
        yp = (y & PM_PRESENT) != 0
        xs = (x & PM_SWAPPED) != 0
        ys = (y & PM_SWAPPED) != 0
        moved = xp & yp & ((x & PM_PFN_MASK) != (y & PM_PFN_MASK))
        out["moved"] += int(np.count_nonzero(moved))
        out["swapped_out"] += int(np.count_nonzero(xp & ys))
        out["dropped"] += int(np.count_nonzero(xp & ~yp & ~ys))
        out["swapped_in"] += int(np.count_nonzero(xs & yp))
        out["populated"] += int(np.count_nonzero(~xp & ~xs & yp))
    return out


def count_state(chunks: list[bytes]) -> tuple[int, int]:
    """(present pages, swapped pages) of a pagemap read."""
    present = swapped = 0
    for c in chunks:
        x = np.frombuffer(c, dtype="<u8")
        present += int(np.count_nonzero(x & PM_PRESENT))
        swapped += int(np.count_nonzero(x & PM_SWAPPED))
    return present, swapped


def sample_lock_state(
    kpf: int, chunks: list[bytes], present: int, want: int = 16
) -> dict[str, int] | None:
    """Look up up to ``want`` evenly spaced present pages in kpageflags.

    Returns how many of the ``n`` sampled are mlocked, unevictable, on the
    LRU; None when there is nothing to sample or PFNs are hidden (no
    CAP_SYS_ADMIN).
    """
    if present == 0:
        return None
    step = max(present // want, 1)
    seen = n = mlocked = unevictable = lru = 0
    for c in chunks:
        x = np.frombuffer(c, dtype="<u8")
        idx = np.nonzero(x & PM_PRESENT)[0]
        if idx.size == 0:
            continue
        for i in idx[(-seen) % step :: step]:
            pfn = int(x[i] & PM_PFN_MASK)
            if pfn == 0:
                return None
            try:
                flags = int.from_bytes(os.pread(kpf, 8, pfn * 8), "little")
            except OSError:
                return None
            n += 1
            mlocked += (flags >> KPF_MLOCKED) & 1
            unevictable += (flags >> KPF_UNEVICTABLE) & 1
            lru += (flags >> KPF_LRU) & 1
            if n >= want:
                return {
                    "n": n,
                    "mlocked": mlocked,
                    "unevictable": unevictable,
                    "lru": lru,
                }
        seen += idx.size
    return {"n": n, "mlocked": mlocked, "unevictable": unevictable, "lru": lru}


def lock_class(state: dict[str, int] | None) -> str:
    """Classify a sample_lock_state result: locked, unlocked or unknown."""
    if not state or not state["n"]:
        return "unknown"
    return "locked" if state["mlocked"] * 2 >= state["n"] else "unlocked"


class Snapshot:
    """The last pagemap read of one mapping."""

    __slots__ = ("chunks", "present", "swapped", "t")

    def __init__(self, chunks: list[bytes], t: float) -> None:
        self.chunks = chunks
        self.present, self.swapped = count_state(chunks)
        self.t = t


class WorkerState:
    """What the sampler remembers about one worker between ticks."""

    def __init__(self, pid: int, title: str) -> None:
        self.pid = pid
        self.title = title
        self.ppid = parent_pid(pid)
        self.pagemap = os.open(f"{PROC}/{pid}/pagemap", os.O_RDONLY)
        self.snap: dict[tuple[int, int], Snapshot] = {}
        self.known: set[tuple[int, int]] = set()
        self.lock: dict[tuple[int, int], dict[str, int] | None] = {}
        self.last_slow = 0.0
        self.last_lock = 0.0

    def close(self) -> None:
        os.close(self.pagemap)


def scan_worker(
    w: WorkerState, now: float, args: argparse.Namespace, kpf: int | None, out: Any
) -> dict[str, Any]:
    """Read the worker's mappings, diff them with the last read, log changes."""
    vmas = read_maps(w.pid)
    if vmas is None:
        return {"err": "maps"}
    keys = {v.key for v in vmas}
    summary: dict[str, Any] = {}
    for cls in ("pinned", "anon", "shm"):
        sel = [v for v in vmas if v.cls == cls]
        summary[cls] = {"n": len(sel), "gib": round(sum(v.size for v in sel) / GIB, 3)}
    new = [v for v in vmas if v.key not in w.known and v.cls == "pinned"]
    gone = w.known - keys
    if w.known and (new or gone):
        summary["pinned_new"] = [[hex(v.start), v.size] for v in new[:20]]
        summary["gone"] = len(gone)
    for k in gone:
        w.snap.pop(k, None)
        w.lock.pop(k, None)
    w.known = keys

    do_slow = now - w.last_slow >= args.slow_period
    do_lock = kpf is not None and now - w.last_lock >= args.lock_period
    moved_total = swapped_total = 0
    for v in vmas:
        fast = v.cls == "pinned" and v.size < args.big_gib * GIB
        if not fast and not do_slow and v.key in w.snap:
            continue
        chunks = read_pagemap(w.pagemap, v.start, v.end)
        if chunks is None:
            w.snap.pop(v.key, None)
            continue
        prev = w.snap.get(v.key)
        if (
            prev is not None
            and len(prev.chunks) == len(chunks)
            and prev.chunks == chunks
        ):
            prev.t = now
            cur = prev
            diff = None
        else:
            cur = Snapshot(chunks, now)
            diff = (
                diff_pagemap(prev.chunks, chunks)
                if prev is not None and len(prev.chunks) == len(chunks)
                else None
            )
        if kpf is not None and (do_lock or v.key not in w.lock):
            w.lock[v.key] = sample_lock_state(kpf, cur.chunks, cur.present)
        if diff and (diff["moved"] or diff["swapped_out"] or diff["dropped"]):
            moved_total += diff["moved"]
            swapped_total += diff["swapped_out"]
            out.write(
                json.dumps(
                    {
                        **diff,
                        "t": round(now, 3),
                        "dt": round(now - prev.t, 3),
                        "pid": w.pid,
                        "cls": v.cls,
                        "start": hex(v.start),
                        "size": v.size,
                        "inode": v.inode,
                        "path": v.path if v.cls == "shm" else "",
                        "present": cur.present,
                        "swapped": cur.swapped,
                        "lock": w.lock.get(v.key),
                    }
                )
                + "\n"
            )
        w.snap[v.key] = cur
    if do_slow:
        w.last_slow = now
    if do_lock:
        w.last_lock = now
    gib: dict[str, float] = {"locked": 0.0, "unlocked": 0.0, "unknown": 0.0}
    for v in vmas:
        if v.cls == "pinned":
            gib[lock_class(w.lock.get(v.key))] += v.size / GIB
    summary["pinned_gib_by_lock"] = {k: round(g, 3) for k, g in gib.items()}
    summary["moved"] = moved_total
    summary["swapped_out"] = swapped_total
    return summary


def own_output(path: str) -> None:
    """Hand the file to the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        with contextlib.suppress(OSError):
            os.chown(path, int(uid), int(gid))


def sync_workers(workers: dict[int, WorkerState], live: dict[int, str]) -> None:
    """Drop the workers that left, follow the new ones, keep every title current.

    A worker's title gains its rank (``_TP0_EP0``) after the fork, so the one
    read when it was first seen may lack it.
    """
    for pid in [p for p in workers if p not in live]:
        workers.pop(pid).close()
    for pid, title in live.items():
        if pid in workers:
            workers[pid].title = title
        else:
            with contextlib.suppress(OSError):
                workers[pid] = WorkerState(pid, title)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--period", type=float, default=1.0, help="seconds per tick")
    ap.add_argument("--big-gib", type=float, default=1.0)
    ap.add_argument("--slow-period", type=float, default=5.0)
    ap.add_argument("--lock-period", type=float, default=30.0)
    ap.add_argument("--discover-period", type=float, default=5.0)
    ap.add_argument("--pid", type=int, action="append", help="follow this pid, too")
    ap.add_argument(
        "--root-pid",
        type=int,
        action="append",
        help="follow only the workers below this pid (a server's), not every "
        "VLLM::Worker on the host",
    )
    ap.add_argument("--no-pagemap", action="store_true", help="status/vmstat only")
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    own_output(args.out)
    kpf = None
    with contextlib.suppress(OSError):
        kpf = os.open(f"{PROC}/kpageflags", os.O_RDONLY)
    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    workers: dict[int, WorkerState] = {}
    kthreads = find_kthreads()
    last_discover = 0.0
    deadline = time.time() + args.hours * 3600
    with (
        open(f"{args.out}/samples.jsonl", "a", buffering=1) as samples,
        open(f"{args.out}/pagemap.jsonl", "a", buffering=1) as pagemap,
    ):
        own_output(f"{args.out}/samples.jsonl")
        own_output(f"{args.out}/pagemap.jsonl")
        meta = {
            "start": time.time(),
            "args": vars(args),
            "kernel": os.uname().release,
            "sysctl": sysctls(),
        }
        samples.write(json.dumps({"meta": meta}) + "\n")
        next_t = time.time()
        while not stop and time.time() < deadline:
            t0 = time.time()
            if t0 - last_discover >= args.discover_period:
                last_discover = t0
                live = worker_pids()
                if args.root_pid:
                    below = process_tree(set(args.root_pid))
                    live = {p: t for p, t in live.items() if p in below}
                for pid in args.pid or ():
                    live[pid] = f"pid{pid}"
                sync_workers(workers, live)
                if len(kthreads) < 2:
                    kthreads = find_kthreads()
            rec: dict[str, Any] = {
                "t": round(t0, 3),
                "vm": vmstat(),
                "mem": meminfo(),
                "psi": psi_memory(),
                "kcpu": {n: cpu_ticks(p) for n, p in kthreads.items()},
                "zram": zram_bytes(),
                "kfd_sys_mib": kfd_system_mem_mib(),
                "w": {},
            }
            for pid, w in workers.items():
                entry: dict[str, Any] = {
                    "title": w.title,
                    "ppid": w.ppid,
                    "evicted_ms": evicted_ms(pid),
                    **process_status(pid),
                }
                if not args.no_pagemap:
                    try:
                        entry["maps"] = scan_worker(w, t0, args, kpf, pagemap)
                    except Exception as exc:  # a vanished worker must not stop us
                        entry["maps"] = {"err": repr(exc)}
                rec["w"][str(pid)] = entry
            rec["tick_s"] = round(time.time() - t0, 3)
            samples.write(json.dumps(rec) + "\n")
            if args.once:
                break
            next_t += args.period
            time.sleep(max(next_t - time.time(), 0))
            if time.time() - next_t > 10 * args.period:
                next_t = time.time()  # fell behind: resynchronise
    return 0


if __name__ == "__main__":
    sys.exit(main())
