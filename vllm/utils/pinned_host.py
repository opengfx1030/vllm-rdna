# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mlock() every pinned host mapping a ROCm worker owns.

On ROCm, page-locked host memory (hipHostMalloc: torch's ``pin_memory=True``,
ROCr's own staging pools, the exact-size PLE table) is a KFD userptr buffer
object over ordinary shmem pages. The driver does not pin those pages. When
page reclaim (swap-out to zram) or compaction moves one of them, the MMU
notifier invalidates the whole buffer object and KFD evicts every GPU queue of
the process until the buffer is faulted back in, which stalls the worker's
collectives.

``mlock()`` keeps reclaim away from a mapping and, with
``vm.compact_unevictable_allowed=0``, keeps compaction away too. The PLE
table is locked when it is allocated; this module locks the rest: every block
that exists when warm-up ends, and every one that appears afterwards (the
first requests add a couple of hundred small ones), within seconds.

Every hipHostMalloc block shows up in ``/proc/<pid>/maps`` as a shared
anonymous mapping, ``rw-s`` named ``/dev/zero (deleted)``. That is the only
kind of mapping locked here. File mappings (the memory-mapped checkpoint,
libraries), the heap, ``/dev/shm`` files and device nodes are never touched,
and neither is the address space of the process as a whole (no ``mlockall``).
"""

import ctypes
import os
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# What hipHostMalloc leaves in /proc/<pid>/maps: a shared anonymous mapping.
_PINNED_PERMS = "rw-s"
_PINNED_PATH = "/dev/zero (deleted)"

_GIB = 1 << 30
_MAPS = "/proc/self/maps"
_STATUS = "/proc/self/status"

# Blocks that appear after warm-up are found by rescanning this often...
_RESCAN_INTERVAL_S = 2.0
# ...and the rescans that locked something are logged together this often.
_RESCAN_LOG_INTERVAL_S = 30.0


class ProcMapping(NamedTuple):
    """One line of ``/proc/<pid>/maps``."""

    start: int
    end: int
    perms: str
    offset: int
    dev: str
    inode: int
    path: str

    @property
    def size(self) -> int:
        return self.end - self.start


def parse_proc_maps(text: str) -> list[ProcMapping]:
    """Parse ``/proc/<pid>/maps``; lines that do not parse are skipped.

    A line is ``start-end perms offset dev inode [path]`` and the path may
    contain spaces (``/dev/zero (deleted)``).
    """
    mappings = []
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            continue
        try:
            start, end = (int(part, 16) for part in fields[0].split("-"))
            mappings.append(
                ProcMapping(
                    start=start,
                    end=end,
                    perms=fields[1],
                    offset=int(fields[2], 16),
                    dev=fields[3],
                    inode=int(fields[4]),
                    path=fields[5].strip() if len(fields) == 6 else "",
                )
            )
        except ValueError:
            continue
    return mappings


def is_pinned_host_mapping(mapping: ProcMapping) -> bool:
    """True for a hipHostMalloc block: ``rw-s /dev/zero (deleted)``."""
    return (
        mapping.perms == _PINNED_PERMS
        and mapping.path == _PINNED_PATH
        and mapping.size > 0
    )


def select_pinned_host_mappings(mappings: list[ProcMapping]) -> list[ProcMapping]:
    """The pinned host mappings among ``mappings``, smallest first.

    Smallest first, so that when RLIMIT_MEMLOCK runs out it is the biggest
    block (the PLE table, which is locked when it is allocated anyway) that
    stays unlocked, not the dozens of small staging buffers.
    """
    return sorted(
        (m for m in mappings if is_pinned_host_mapping(m)),
        key=lambda m: (m.size, m.start),
    )


class _MlockFn(Protocol):
    def __call__(self, addr: int, nbytes: int) -> int: ...


_libc: ctypes.CDLL | None = None


def mlock_range(addr: int, nbytes: int) -> int:
    """mlock() ``[addr, addr + nbytes)``; returns 0 or the errno.

    Fails with ENOMEM/EPERM when the range would take the process over
    RLIMIT_MEMLOCK, and with ENOMEM when part of it is not mapped.
    """
    global _libc
    if nbytes <= 0:
        return 0
    if _libc is None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mlock.restype = ctypes.c_int
        _libc = libc
    if _libc.mlock(addr, nbytes) == 0:
        return 0
    return ctypes.get_errno()


def vm_locked_bytes(status_path: str = _STATUS) -> int | None:
    """VmLck of this process in bytes, or None when it cannot be read."""
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmLck:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


@dataclass
class PinnedLockReport:
    """Outcome of one ``PinnedHostLocker.scan`` pass."""

    candidates: int = 0
    locked: int = 0
    locked_bytes: int = 0
    failed: int = 0
    failed_bytes: int = 0
    # errno -> number of mappings that failed with it
    errors: dict[int, int] = field(default_factory=dict)
    vm_locked_before: int | None = None
    vm_locked_after: int | None = None

    def add(self, later: "PinnedLockReport") -> None:
        """Fold a later pass that locked or failed something into this one."""
        self.candidates = later.candidates
        self.locked += later.locked
        self.locked_bytes += later.locked_bytes
        self.failed += later.failed
        self.failed_bytes += later.failed_bytes
        for err, count in later.errors.items():
            self.errors[err] = self.errors.get(err, 0) + count
        if self.vm_locked_before is None:
            self.vm_locked_before = later.vm_locked_before
        self.vm_locked_after = later.vm_locked_after


_MappingKey = tuple[int, int, int]


def _key(mapping: ProcMapping) -> _MappingKey:
    # The inode tells a new block at a reused address from the one before it.
    return (mapping.start, mapping.end, mapping.inode)


class PinnedHostLocker:
    """Locks the pinned host mappings of this process, now and as they appear.

    ``scan()`` locks every mapping that no earlier scan locked, so it can be
    called again and again. Most blocks exist when warm-up ends, but a couple
    of hundred small ones appear while the first requests run: ``start()``
    rescans in a daemon thread, so those are locked within seconds too.
    """

    def __init__(
        self,
        maps_path: str = _MAPS,
        *,
        mlock: _MlockFn = mlock_range,
        status_path: str = _STATUS,
    ) -> None:
        self._maps_path = maps_path
        self._mlock = mlock
        self._status_path = status_path
        self._scan_lock = threading.Lock()
        self._locked: set[_MappingKey] = set()
        self._failed: set[_MappingKey] = set()
        self._last_maps: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def locked_count(self) -> int:
        """The pinned mappings this locker holds locked right now."""
        return len(self._locked)

    def _read_maps(self) -> str:
        with open(self._maps_path) as f:
            return f.read()

    def scan(self) -> PinnedLockReport:
        """mlock() the pinned host mappings that are new since the last scan.

        Never raises on a mapping that cannot be locked: the worker keeps
        serving with that mapping unlocked, and the report says how many and
        why. Raises OSError when the maps file cannot be read.
        """
        with self._scan_lock:
            return self._scan()

    def _scan(self) -> PinnedLockReport:
        text = self._read_maps()
        if text == self._last_maps:
            return PinnedLockReport(candidates=len(self._locked) + len(self._failed))
        pinned = select_pinned_host_mappings(parse_proc_maps(text))
        present = {_key(m) for m in pinned}
        # Unmapping a block drops its lock: forget it, a new block at the same
        # address is locked again.
        self._locked &= present
        self._failed &= present
        report = PinnedLockReport(
            candidates=len(pinned),
            vm_locked_before=vm_locked_bytes(self._status_path),
        )
        refused: list[tuple[ProcMapping, int]] = []
        for mapping in pinned:
            if _key(mapping) in self._locked or _key(mapping) in self._failed:
                continue
            err = self._mlock(mapping.start, mapping.size)
            if err == 0:
                self._locked.add(_key(mapping))
                report.locked += 1
                report.locked_bytes += mapping.size
            else:
                refused.append((mapping, err))
        if refused:
            # A block freed since the maps were read fails with ENOMEM too:
            # only the ones still mapped are blamed on the memlock limit.
            still = {
                _key(m)
                for m in select_pinned_host_mappings(parse_proc_maps(self._read_maps()))
            }
            for mapping, err in refused:
                if _key(mapping) not in still:
                    continue
                self._failed.add(_key(mapping))
                report.failed += 1
                report.failed_bytes += mapping.size
                report.errors[err] = report.errors.get(err, 0) + 1
        report.vm_locked_after = vm_locked_bytes(self._status_path)
        self._last_maps = text
        return report

    def start(self, interval_s: float = _RESCAN_INTERVAL_S) -> None:
        """Rescan every ``interval_s`` seconds in a daemon thread."""
        if self._thread is not None or interval_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._rescan_forever,
            args=(interval_s,),
            name="pinned-host-mlock",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _rescan_forever(self, interval_s: float) -> None:
        pending = PinnedLockReport()
        last_log = time.monotonic()
        while not self._stop.wait(interval_s):
            try:
                report = self.scan()
            except OSError as exc:
                logger.warning(
                    "No longer locking new pinned host mappings: cannot read %s: %s",
                    self._maps_path,
                    exc,
                )
                return
            if report.locked or report.failed:
                pending.add(report)
            if (
                pending.locked or pending.failed
            ) and time.monotonic() - last_log >= _RESCAN_LOG_INTERVAL_S:
                log_pinned_rescan_report(pending, self.locked_count)
                pending = PinnedLockReport()
                last_log = time.monotonic()


def lock_pinned_host_mappings(
    maps_path: str = _MAPS,
    *,
    mlock: _MlockFn = mlock_range,
    status_path: str = _STATUS,
) -> PinnedLockReport:
    """One scan: mlock() every pinned host mapping listed in ``maps_path``."""
    return PinnedHostLocker(maps_path, mlock=mlock, status_path=status_path).scan()


def _gib(nbytes: int | None) -> str:
    return "?" if nbytes is None else f"{nbytes / _GIB:.2f}"


def _warn_failed(report: PinnedLockReport) -> None:
    logger.warning(
        "mlock failed for %d pinned host mappings (%s GiB): %s. Host page "
        "reclaim and compaction can still evict this worker's GPU queues. "
        "Raise RLIMIT_MEMLOCK (ulimit -l).",
        report.failed,
        _gib(report.failed_bytes),
        ", ".join(
            f"{os.strerror(err)} x{count}"
            for err, count in sorted(report.errors.items())
        ),
    )


def log_pinned_lock_report(report: PinnedLockReport) -> None:
    logger.info(
        "Locked %d of %d pinned host mappings (%s GiB); VmLck %s -> %s GiB.",
        report.locked,
        report.candidates,
        _gib(report.locked_bytes),
        _gib(report.vm_locked_before),
        _gib(report.vm_locked_after),
    )
    if report.failed:
        _warn_failed(report)


def log_pinned_rescan_report(report: PinnedLockReport, locked_total: int) -> None:
    """The rescans since the last line: blocks that appeared after warm-up.

    Worded so that it is not mistaken for the warm-up line above, which the
    GPU proof counts (one per worker).
    """
    logger.info(
        "Pinned host mappings that appeared after warm-up: %d now mlocked "
        "(%.1f MiB), %d in all; VmLck %s -> %s GiB.",
        report.locked,
        report.locked_bytes / (1 << 20),
        locked_total,
        _gib(report.vm_locked_before),
        _gib(report.vm_locked_after),
    )
    if report.failed:
        _warn_failed(report)


def pinned_host_lock_enabled() -> bool:
    """Whether this worker locks its pinned host mappings.

    VLLM_ROCM_MLOCK_PINNED_HOST: "auto" (default) is on for ROCm gfx10x only,
    "1" is on for any ROCm GPU, "0" is off.
    """
    mode = envs.VLLM_ROCM_MLOCK_PINNED_HOST
    if mode == "0" or not current_platform.is_rocm():
        return False
    if mode == "1":
        return True
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


_locker: PinnedHostLocker | None = None


def lock_pinned_host_after_warmup() -> PinnedLockReport | None:
    """Worker hook, called once after model load and CUDA graph capture.

    Locks every pinned host mapping the worker owns at that point, logs the
    total once, and keeps locking the ones that appear later. Returns None
    when disabled.
    """
    global _locker
    if not pinned_host_lock_enabled():
        return None
    locker = _locker or PinnedHostLocker()
    try:
        report = locker.scan()
    except OSError as exc:
        logger.warning("Could not read %s to lock pinned host memory: %s", _MAPS, exc)
        return None
    log_pinned_lock_report(report)
    locker.start()
    _locker = locker
    return report
