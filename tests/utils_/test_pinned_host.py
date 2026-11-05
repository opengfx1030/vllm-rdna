# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locking a ROCm worker's pinned host mappings: map walk, selection, lock."""

import errno
import mmap
import os
import resource
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import vllm.utils.pinned_host as ph
from vllm.utils.pinned_host import (
    ProcMapping,
    is_pinned_host_mapping,
    lock_pinned_host_after_warmup,
    lock_pinned_host_mappings,
    parse_proc_maps,
    pinned_host_lock_enabled,
    select_pinned_host_mappings,
    vm_locked_bytes,
)

PINNED = "/dev/zero (deleted)"
ENV = "VLLM_ROCM_MLOCK_PINNED_HOST"

# A trimmed /proc/<pid>/maps of a gfx1030 worker: a 32 GiB and two small
# hipHostMalloc blocks, the memory-mapped checkpoint, the heap, KFD/DRI device
# nodes, NCCL's /dev/shm segment, a PROT_NONE reservation and an anonymous
# private mapping.
MAPS = f"""\
5599c3a00000-5599c3a21000 rw-p 00000000 00:00 0  [heap]
7f3134c00000-7f490c4be000 rw-p 00000000 00:2b 123456  /home/u/blobs/0262c8
7f657b000000-7f6d7b000000 rw-s 00000000 00:01 4097  {PINNED}
7f6ea6c00000-7f6fa6c00000 ---p 00000000 00:00 0
7f7000000000-7f7000200000 rw-s 00000000 00:01 4100  {PINNED}
7f7000400000-7f7000401000 rw-s 00000000 00:01 4101  {PINNED}
7f7100000000-7f7100100000 rw-s 00000000 00:05 9  /dev/shm/nccl-iHDPv0 (deleted)
7f7200000000-7f7200002000 rw-s 00000000 00:06 3  /dev/kfd
7f7200100000-7f7200200000 rw-s 00000000 00:06 4  /dev/dri/renderD128
7f7300000000-7f7300100000 r--s 00000000 00:01 4102  {PINNED}
7f7400000000-7f7400100000 rw-p 00000000 00:00 0
7ffd8a2f0000-7ffd8a311000 rw-p 00000000 00:00 0  [stack]
"""

GIB = 1 << 30


def _pinned(start: int, size: int, perms: str = "rw-s", path: str = PINNED):
    return ProcMapping(start, start + size, perms, 0, "00:01", 1, path)


def test_parse_proc_maps_fields_and_paths_with_spaces():
    mappings = parse_proc_maps(MAPS)
    assert len(mappings) == 12
    big = mappings[2]
    assert big == ProcMapping(
        start=0x7F657B000000,
        end=0x7F6D7B000000,
        perms="rw-s",
        offset=0,
        dev="00:01",
        inode=4097,
        path=PINNED,
    )
    assert big.size == 32 * GIB
    assert mappings[0].path == "[heap]"
    assert mappings[3].path == ""  # no path column
    assert mappings[6].path == "/dev/shm/nccl-iHDPv0 (deleted)"


def test_parse_proc_maps_skips_malformed_lines():
    text = "garbage\n\n7f00-zz rw-s 0 00:01 1 /x\n7f0000-7f1000 rw-s 00000000 00:01 7\n"
    assert parse_proc_maps(text) == [
        ProcMapping(0x7F0000, 0x7F1000, "rw-s", 0, "00:01", 7, "")
    ]


def test_select_takes_only_hip_host_blocks_smallest_first():
    selected = select_pinned_host_mappings(parse_proc_maps(MAPS))
    assert [m.start for m in selected] == [
        0x7F7000400000,  # 4 KiB
        0x7F7000000000,  # 2 MiB
        0x7F657B000000,  # 32 GiB, the PLE table's block
    ]
    assert all(m.path == PINNED and m.perms == "rw-s" for m in selected)


@pytest.mark.parametrize(
    "perms, path, expected",
    [
        ("rw-s", PINNED, True),
        ("rw-s", "/dev/zero", False),  # the device file itself, not shmem
        ("rw-s", "/dev/shm/nccl-iHDPv0 (deleted)", False),
        ("rw-s", "/dev/kfd", False),
        ("rw-s", "/dev/dri/renderD128", False),
        ("rw-s", "/memfd:pytorch_shm (deleted)", False),
        ("rw-p", PINNED, False),  # private: not a hipHostMalloc block
        ("r--s", PINNED, False),
        ("---s", PINNED, False),
        ("rw-p", "", False),
        ("rw-p", "[heap]", False),
        ("rw-p", "/home/u/.cache/huggingface/hub/blobs/02/0262c8", False),
        ("r-xp", "/lib64/libc.so.6", False),
    ],
)
def test_is_pinned_host_mapping(perms: str, path: str, expected: bool):
    assert is_pinned_host_mapping(_pinned(0x1000, 0x1000, perms, path)) is expected


def test_empty_mapping_is_not_selected():
    assert not is_pinned_host_mapping(_pinned(0x1000, 0))


def _write_status(path: Path, vm_lck_kib: int) -> None:
    path.write_text(f"Name:\tpython\nVmLck:\t{vm_lck_kib:>8} kB\nVmPin:\t       0 kB\n")


def test_vm_locked_bytes(tmp_path: Path):
    status = tmp_path / "status"
    _write_status(status, 24 * 1024 * 1024)
    assert vm_locked_bytes(str(status)) == 24 * GIB
    assert vm_locked_bytes(str(tmp_path / "missing")) is None
    status.write_text("Name:\tpython\n")
    assert vm_locked_bytes(str(status)) is None


def test_lock_pinned_host_mappings_locks_every_block_and_counts_gib(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(MAPS)
    status = tmp_path / "status"
    _write_status(status, 100)
    calls: list[tuple[int, int]] = []

    def fake_mlock(addr: int, nbytes: int) -> int:
        calls.append((addr, nbytes))
        _write_status(status, 100 + sum(n for _, n in calls) // 1024)
        return 0

    report = lock_pinned_host_mappings(
        str(maps), mlock=fake_mlock, status_path=str(status)
    )
    assert calls == [
        (0x7F7000400000, 4096),
        (0x7F7000000000, 2 << 20),
        (0x7F657B000000, 32 * GIB),
    ]
    assert report.candidates == 3
    assert report.locked == 3
    assert report.locked_bytes == 4096 + (2 << 20) + 32 * GIB
    assert report.failed == 0 and report.failed_bytes == 0 and report.errors == {}
    assert report.vm_locked_before == 100 * 1024
    assert report.vm_locked_after == 100 * 1024 + report.locked_bytes


def test_lock_pinned_host_mappings_never_touches_other_mappings(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(MAPS)
    locked_starts = set()

    def fake_mlock(addr: int, nbytes: int) -> int:
        locked_starts.add(addr)
        return 0

    lock_pinned_host_mappings(str(maps), mlock=fake_mlock, status_path="/nonexistent")
    checkpoint = 0x7F3134C00000
    heap = 0x5599C3A00000
    assert checkpoint not in locked_starts and heap not in locked_starts
    assert locked_starts == {0x7F657B000000, 0x7F7000000000, 0x7F7000400000}


def test_lock_pinned_host_mappings_reports_failures_and_keeps_going(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(MAPS)

    def fake_mlock(addr: int, nbytes: int) -> int:
        # Out of RLIMIT_MEMLOCK for the big block only.
        return errno.ENOMEM if nbytes == 32 * GIB else 0

    report = lock_pinned_host_mappings(
        str(maps), mlock=fake_mlock, status_path="/nonexistent"
    )
    assert report.candidates == 3
    assert report.locked == 2 and report.locked_bytes == 4096 + (2 << 20)
    assert report.failed == 1 and report.failed_bytes == 32 * GIB
    assert report.errors == {errno.ENOMEM: 1}
    assert report.vm_locked_before is None and report.vm_locked_after is None


def test_lock_pinned_host_mappings_without_pinned_blocks(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text("5599c3a00000-5599c3a21000 rw-p 00000000 00:00 0   [heap]\n")
    report = lock_pinned_host_mappings(
        str(maps), mlock=lambda a, n: pytest.fail("locked"), status_path="/x"
    )
    assert report.candidates == 0 and report.locked == 0 and report.failed == 0


def _maps_with(*blocks: tuple[int, int, int]) -> str:
    """A maps file: a heap and these pinned blocks, as (start, size, inode)."""
    lines = ["5599c3a00000-5599c3a21000 rw-p 00000000 00:00 0  [heap]"]
    for start, size, inode in sorted(blocks):
        lines.append(
            f"{start:x}-{start + size:x} rw-s 00000000 00:01 {inode}  {PINNED}"
        )
    return "\n".join(lines) + "\n"


A = (0x7F7000000000, 2 << 20, 10)
B = (0x7F7000400000, 4096, 11)
C = (0x7F7000800000, 3 << 20, 12)


def _locker(maps: Path, calls: list[int], *, refuse=lambda addr, nbytes: 0):
    def fake_mlock(addr: int, nbytes: int) -> int:
        calls.append(addr)
        return refuse(addr, nbytes)

    return ph.PinnedHostLocker(str(maps), mlock=fake_mlock, status_path="/nonexistent")


def test_locker_scan_locks_only_what_is_new(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A, B))
    calls: list[int] = []
    locker = _locker(maps, calls)

    first = locker.scan()
    assert first.candidates == 2 and first.locked == 2
    assert calls == [B[0], A[0]]

    # Nothing changed: no work, and above all no second pass over a big block.
    assert locker.scan() == ph.PinnedLockReport(candidates=2)
    assert calls == [B[0], A[0]]

    maps.write_text(_maps_with(A, B, C))
    third = locker.scan()
    assert third.candidates == 3 and third.locked == 1
    assert third.locked_bytes == C[1]
    assert calls == [B[0], A[0], C[0]]
    assert locker.locked_count == 3


def test_locker_does_not_reparse_unchanged_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A))
    parses: list[int] = []
    real_parse = ph.parse_proc_maps

    def counting_parse(text: str) -> list[ProcMapping]:
        parses.append(1)
        return real_parse(text)

    monkeypatch.setattr(ph, "parse_proc_maps", counting_parse)
    locker = _locker(maps, [])
    locker.scan()
    locker.scan()
    locker.scan()
    assert len(parses) == 1


def test_locker_locks_a_new_block_at_the_address_of_an_unmapped_one(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A))
    calls: list[int] = []
    locker = _locker(maps, calls)
    locker.scan()

    # Freed and allocated again between two scans: same address and size, but
    # a new shmem file. Unmapping dropped the lock, so it is locked again.
    reused = (A[0], A[1], 99)
    maps.write_text(_maps_with(reused))
    assert locker.scan().locked == 1

    # Freed in one scan, back (same inode) in the next one: locked again.
    maps.write_text(_maps_with(B))
    locker.scan()
    maps.write_text(_maps_with(B, reused))
    assert locker.scan().locked == 1
    assert calls == [A[0], A[0], B[0], A[0]]
    assert locker.locked_count == 2


def test_locker_does_not_blame_the_limit_for_a_block_freed_meanwhile(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A, B))

    def freed_while_locking(addr: int, nbytes: int) -> int:
        if addr == B[0]:
            maps.write_text(_maps_with(A))
            return errno.ENOMEM  # what mlock says about an unmapped range
        return 0

    report = _locker(maps, [], refuse=freed_while_locking).scan()
    assert report.locked == 1
    assert report.failed == 0 and report.errors == {}


def test_locker_reports_a_refused_block_once_and_never_retries_it(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A, B))
    calls: list[int] = []
    locker = _locker(
        maps, calls, refuse=lambda addr, nbytes: errno.ENOMEM if addr == B[0] else 0
    )
    first = locker.scan()
    assert first.locked == 1 and first.failed == 1
    assert first.failed_bytes == B[1] and first.errors == {errno.ENOMEM: 1}

    maps.write_text(_maps_with(A, B, C))
    second = locker.scan()
    assert second.locked == 1 and second.failed == 0
    assert calls.count(B[0]) == 1
    assert locker.locked_count == 2


def _wait_for(condition, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_rescan_thread_locks_blocks_that_appear_later(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A))
    calls: list[int] = []
    locker = _locker(maps, calls)
    locker.scan()
    locker.start(0.01)
    try:
        maps.write_text(_maps_with(A, C))
        assert _wait_for(lambda: C[0] in calls)
    finally:
        locker.stop()
    assert calls == [A[0], C[0]]
    assert "pinned-host-mlock" not in [t.name for t in threading.enumerate()]


def test_rescan_thread_logs_what_it_locked_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    infos: list[str] = []
    monkeypatch.setattr(ph.logger, "info", lambda msg, *a: infos.append(msg % a))
    monkeypatch.setattr(ph, "_RESCAN_LOG_INTERVAL_S", 0.0)
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A))
    locker = _locker(maps, [])
    locker.scan()
    locker.start(0.01)
    try:
        maps.write_text(_maps_with(A, C))
        assert _wait_for(lambda: infos)
        time.sleep(0.1)  # further scans find nothing and log nothing
    finally:
        locker.stop()
    assert infos == [
        (
            "Pinned host mappings that appeared after warm-up: 1 now mlocked "
            "(3.0 MiB), 2 in all; VmLck ? -> ? GiB."
        )
    ]
    assert "Locked " not in infos[0]  # the GPU proof counts the warm-up line only


def test_rescan_thread_stops_when_maps_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    warnings: list[str] = []
    monkeypatch.setattr(ph.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    maps = tmp_path / "maps"
    maps.write_text(_maps_with(A))
    locker = _locker(maps, [])
    locker.scan()
    locker.start(0.01)
    maps.unlink()
    assert _wait_for(lambda: warnings)
    assert locker._thread is not None
    locker._thread.join(timeout=20)
    assert not locker._thread.is_alive()
    assert "No longer locking new pinned host mappings" in warnings[0]
    locker.stop()


def test_report_add_folds_later_passes_in():
    total = ph.PinnedLockReport()
    total.add(
        ph.PinnedLockReport(
            candidates=5,
            locked=2,
            locked_bytes=100,
            failed=1,
            failed_bytes=7,
            errors={errno.ENOMEM: 1},
            vm_locked_before=10,
            vm_locked_after=110,
        )
    )
    total.add(
        ph.PinnedLockReport(
            candidates=6,
            locked=1,
            locked_bytes=50,
            errors={},
            vm_locked_before=110,
            vm_locked_after=160,
        )
    )
    assert total == ph.PinnedLockReport(
        candidates=6,
        locked=3,
        locked_bytes=150,
        failed=1,
        failed_bytes=7,
        errors={errno.ENOMEM: 1},
        vm_locked_before=10,
        vm_locked_after=160,
    )


def test_log_reports_the_total_once_and_warns_on_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(ph.logger, "info", lambda msg, *a: infos.append(msg % a))
    monkeypatch.setattr(ph.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    report = ph.PinnedLockReport(
        candidates=4,
        locked=3,
        locked_bytes=3 * GIB,
        failed=1,
        failed_bytes=GIB,
        errors={errno.ENOMEM: 1},
        vm_locked_before=24 * GIB,
        vm_locked_after=27 * GIB,
    )
    ph.log_pinned_lock_report(report)
    assert infos == [
        "Locked 3 of 4 pinned host mappings (3.00 GiB); VmLck 24.00 -> 27.00 GiB."
    ]
    assert len(warnings) == 1
    assert "1 pinned host mappings (1.00 GiB)" in warnings[0]
    assert os.strerror(errno.ENOMEM) in warnings[0]

    infos.clear()
    warnings.clear()
    ph.log_pinned_lock_report(ph.PinnedLockReport(candidates=2, locked=2))
    assert len(infos) == 1 and "VmLck ? -> ? GiB" in infos[0]
    assert warnings == []


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch):
    """Fake current_platform and on_gfx10x; returns the knobs."""
    knobs = types.SimpleNamespace(rocm=True, gfx10x=True)
    monkeypatch.setattr(
        ph, "current_platform", types.SimpleNamespace(is_rocm=lambda: knobs.rocm)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        types.SimpleNamespace(on_gfx10x=lambda: knobs.gfx10x),
    )
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(ph, "_locker", None)
    return knobs


def test_enabled_by_default_on_rocm_gfx10x_only(platform):
    assert pinned_host_lock_enabled() is True
    platform.gfx10x = False
    assert pinned_host_lock_enabled() is False
    platform.rocm = False
    platform.gfx10x = True
    assert pinned_host_lock_enabled() is False


def test_env_switch(platform, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV, "0")
    assert pinned_host_lock_enabled() is False

    # "1" extends it to every ROCm GPU, but never to other platforms.
    monkeypatch.setenv(ENV, "1")
    platform.gfx10x = False
    assert pinned_host_lock_enabled() is True
    platform.rocm = False
    assert pinned_host_lock_enabled() is False


def test_unknown_env_value_is_rejected(platform, monkeypatch: pytest.MonkeyPatch):
    # a typo must not silently pick a mode: envs validates the choices
    for value in ("maybe", "true", "AUTO"):
        monkeypatch.setenv(ENV, value)
        with pytest.raises(ValueError, match=ENV):
            pinned_host_lock_enabled()
    monkeypatch.setenv(ENV, "auto")
    assert pinned_host_lock_enabled() is True


def test_hook_does_nothing_when_disabled(platform, monkeypatch: pytest.MonkeyPatch):
    platform.rocm = False
    monkeypatch.setattr(
        ph, "PinnedHostLocker", lambda *a, **k: pytest.fail("walked maps")
    )
    assert lock_pinned_host_after_warmup() is None


def _fake_locker(monkeypatch: pytest.MonkeyPatch, scans: list):
    """PinnedHostLocker stand-in: scan() returns or raises what ``scans`` says."""
    made: list = []

    class FakeLocker:
        def __init__(self) -> None:
            self.started = 0
            made.append(self)

        def scan(self):
            item = scans.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        def start(self) -> None:
            self.started += 1

    monkeypatch.setattr(ph, "PinnedHostLocker", FakeLocker)
    return made


def test_hook_scans_logs_once_starts_the_rescan_and_survives_unreadable_maps(
    platform, monkeypatch: pytest.MonkeyPatch
):
    report = ph.PinnedLockReport(candidates=1, locked=1, locked_bytes=4096)
    logged: list[ph.PinnedLockReport] = []
    monkeypatch.setattr(ph, "log_pinned_lock_report", logged.append)
    denied = PermissionError(errno.EACCES, "denied")
    made = _fake_locker(monkeypatch, [report, denied])
    assert lock_pinned_host_after_warmup() is report
    assert logged == [report]
    assert made[0].started == 1

    # Called again (a wake-up, say): the same locker, not a second thread.
    assert lock_pinned_host_after_warmup() is None
    assert len(made) == 1 and made[0].started == 1
    assert logged == [report]


def test_hook_starts_nothing_when_maps_cannot_be_read(
    platform, monkeypatch: pytest.MonkeyPatch
):
    made = _fake_locker(monkeypatch, [PermissionError(errno.EACCES, "denied")])
    assert lock_pinned_host_after_warmup() is None
    assert made[0].started == 0
    assert ph._locker is None


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc/self/maps and mlock"
)
def test_locks_real_shared_anonymous_mappings_as_they_appear():
    """Real /proc/self/maps and mlock: a shared anonymous mmap is walked,
    selected and locked, and the process's VmLck grows by its size; a second
    one created after the first scan is locked by the next scan, and the
    first is not locked again."""
    import ctypes

    size = 256 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    needed = (vm_locked_bytes() or 0) + 3 * size
    if soft != resource.RLIM_INFINITY and soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            pytest.skip("RLIMIT_MEMLOCK too small to lock a test mapping")
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (needed, hard))
    regions: list[tuple[mmap.mmap, object, int]] = []
    mine: set[int] = set()
    attempts: list[int] = []

    def new_region() -> None:
        region = mmap.mmap(-1, size)  # MAP_SHARED | MAP_ANONYMOUS
        view = ctypes.c_char.from_buffer(region)
        addr = ctypes.addressof(view)
        regions.append((region, view, addr))
        region.write(b"\x01" * size)
        with open("/proc/self/maps") as f:
            ours = [m for m in parse_proc_maps(f.read()) if m.start <= addr < m.end]
        if not ours or not is_pinned_host_mapping(ours[0]):
            pytest.skip(f"this kernel names shared anonymous memory {ours!r}")
        mine.add(ours[0].start)

    def lock_only_mine(start: int, nbytes: int) -> int:
        # The real mlock, for the test's mappings only: leave the rest of the
        # pytest process (torch's own shared memory) alone.
        if start not in mine:
            return 0
        attempts.append(start)
        return ph.mlock_range(start, nbytes)

    try:
        new_region()
        locker = ph.PinnedHostLocker(mlock=lock_only_mine)
        before = vm_locked_bytes()
        report = locker.scan()
        after_first = vm_locked_bytes()
        assert report.failed == 0, report
        assert report.locked >= 1 and report.locked_bytes >= size
        assert before is not None and after_first is not None
        assert after_first - before >= size
        assert report.vm_locked_after == after_first

        new_region()
        report = locker.scan()
        after_second = vm_locked_bytes()
        assert report.failed == 0, report
        assert report.locked >= 1 and report.locked_bytes >= size
        assert after_second is not None and after_second - after_first >= size
        assert attempts == sorted(mine, key=attempts.index) and len(attempts) == 2
    finally:
        while regions:
            # Popped, so that the ctypes view is the last reference to the buffer.
            region, view, addr = regions.pop()
            ctypes.CDLL(None).munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            del view
            region.close()
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
