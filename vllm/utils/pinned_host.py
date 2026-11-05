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
table is locked when it is allocated; this module locks the rest.

Every hipHostMalloc block shows up in ``/proc/<pid>/maps`` as a shared
anonymous mapping, ``rw-s`` named ``/dev/zero (deleted)``. That is the only
kind of mapping locked here. File mappings (the memory-mapped checkpoint,
libraries), the heap, ``/dev/shm`` files and device nodes are never touched,
and neither is the address space of the process as a whole (no ``mlockall``).
"""

import ctypes
import os
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Env switch: "auto" (default) locks on ROCm gfx10x, "1" on any ROCm GPU, "0" never.
ENV_MLOCK_PINNED_HOST = "VLLM_ROCM_MLOCK_PINNED_HOST"

# What hipHostMalloc leaves in /proc/<pid>/maps: a shared anonymous mapping.
_PINNED_PERMS = "rw-s"
_PINNED_PATH = "/dev/zero (deleted)"

_GIB = 1 << 30
_MAPS = "/proc/self/maps"
_STATUS = "/proc/self/status"


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
    """Outcome of one ``lock_pinned_host_mappings`` pass."""

    candidates: int = 0
    locked: int = 0
    locked_bytes: int = 0
    failed: int = 0
    failed_bytes: int = 0
    # errno -> number of mappings that failed with it
    errors: dict[int, int] = field(default_factory=dict)
    vm_locked_before: int | None = None
    vm_locked_after: int | None = None


def lock_pinned_host_mappings(
    maps_path: str = _MAPS,
    *,
    mlock: _MlockFn = mlock_range,
    status_path: str = _STATUS,
) -> PinnedLockReport:
    """mlock() every pinned host mapping listed in ``maps_path``.

    Never raises on a mapping that cannot be locked: the worker keeps serving
    with that mapping unlocked, and the report says how many and why.
    """
    with open(maps_path) as f:
        candidates = select_pinned_host_mappings(parse_proc_maps(f.read()))
    report = PinnedLockReport(
        candidates=len(candidates), vm_locked_before=vm_locked_bytes(status_path)
    )
    for mapping in candidates:
        err = mlock(mapping.start, mapping.size)
        if err == 0:
            report.locked += 1
            report.locked_bytes += mapping.size
        else:
            report.failed += 1
            report.failed_bytes += mapping.size
            report.errors[err] = report.errors.get(err, 0) + 1
    report.vm_locked_after = vm_locked_bytes(status_path)
    return report


def _gib(nbytes: int | None) -> str:
    return "?" if nbytes is None else f"{nbytes / _GIB:.2f}"


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


def pinned_host_lock_enabled() -> bool:
    """Whether this worker locks its pinned host mappings.

    VLLM_ROCM_MLOCK_PINNED_HOST: "auto" (default) is on for ROCm gfx10x only,
    "1" is on for any ROCm GPU, "0" is off.
    """
    mode = os.environ.get(ENV_MLOCK_PINNED_HOST, "auto").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return False
    if not current_platform.is_rocm():
        return False
    if mode in ("1", "true", "on", "yes"):
        return True
    if mode != "auto":
        logger.warning(
            "Ignoring %s=%r (expected auto, 1 or 0); using auto.",
            ENV_MLOCK_PINNED_HOST,
            mode,
        )
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


def lock_pinned_host_after_warmup() -> PinnedLockReport | None:
    """Worker hook, called once after model load and CUDA graph capture.

    Locks every pinned host mapping the worker owns at that point and logs
    the total once. Returns None when disabled.
    """
    if not pinned_host_lock_enabled():
        return None
    try:
        report = lock_pinned_host_mappings()
    except OSError as exc:
        logger.warning("Could not read %s to lock pinned host memory: %s", _MAPS, exc)
        return None
    log_pinned_lock_report(report)
    return report
