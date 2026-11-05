# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locking a ROCm worker's pinned host mappings: map walk, selection, lock."""

import errno
import mmap
import os
import resource
import sys
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
    monkeypatch.delenv(ph.ENV_MLOCK_PINNED_HOST, raising=False)
    return knobs


def test_enabled_by_default_on_rocm_gfx10x_only(platform):
    assert pinned_host_lock_enabled() is True
    platform.gfx10x = False
    assert pinned_host_lock_enabled() is False
    platform.rocm = False
    platform.gfx10x = True
    assert pinned_host_lock_enabled() is False


def test_env_switch(platform, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ph.ENV_MLOCK_PINNED_HOST, "0")
    assert pinned_host_lock_enabled() is False

    # "1" extends it to every ROCm GPU, but never to other platforms.
    monkeypatch.setenv(ph.ENV_MLOCK_PINNED_HOST, "1")
    platform.gfx10x = False
    assert pinned_host_lock_enabled() is True
    platform.rocm = False
    assert pinned_host_lock_enabled() is False


def test_unknown_env_value_falls_back_to_auto(
    platform, monkeypatch: pytest.MonkeyPatch
):
    warnings: list[str] = []
    monkeypatch.setattr(ph.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setenv(ph.ENV_MLOCK_PINNED_HOST, "maybe")
    assert pinned_host_lock_enabled() is True  # auto, on gfx10x
    platform.gfx10x = False
    assert pinned_host_lock_enabled() is False
    assert warnings and "maybe" in warnings[0]


def test_hook_does_nothing_when_disabled(platform, monkeypatch: pytest.MonkeyPatch):
    platform.rocm = False
    monkeypatch.setattr(
        ph, "lock_pinned_host_mappings", lambda *a, **k: pytest.fail("walked maps")
    )
    assert lock_pinned_host_after_warmup() is None


def test_hook_walks_maps_logs_once_and_survives_unreadable_maps(
    platform, monkeypatch: pytest.MonkeyPatch
):
    report = ph.PinnedLockReport(candidates=1, locked=1, locked_bytes=4096)
    logged: list[ph.PinnedLockReport] = []
    monkeypatch.setattr(ph, "lock_pinned_host_mappings", lambda: report)
    monkeypatch.setattr(ph, "log_pinned_lock_report", logged.append)
    assert lock_pinned_host_after_warmup() is report
    assert logged == [report]

    def unreadable():
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(ph, "lock_pinned_host_mappings", unreadable)
    assert lock_pinned_host_after_warmup() is None
    assert logged == [report]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc/self/maps and mlock"
)
def test_locks_a_real_shared_anonymous_mapping():
    """Real /proc/self/maps and mlock: a shared anonymous mmap is walked,
    selected and locked, and the process's VmLck grows by its size."""
    import ctypes

    size = 256 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    needed = (vm_locked_bytes() or 0) + 2 * size
    if soft != resource.RLIM_INFINITY and soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            pytest.skip("RLIMIT_MEMLOCK too small to lock a test mapping")
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (needed, hard))
    region = mmap.mmap(-1, size)  # MAP_SHARED | MAP_ANONYMOUS
    view = ctypes.c_char.from_buffer(region)
    addr = ctypes.addressof(view)
    try:
        region.write(b"\x01" * size)
        with open("/proc/self/maps") as f:
            ours = [m for m in parse_proc_maps(f.read()) if m.start <= addr < m.end]
        if not ours or not is_pinned_host_mapping(ours[0]):
            pytest.skip(f"this kernel names shared anonymous memory {ours!r}")

        def lock_only_ours(start: int, nbytes: int) -> int:
            # The real mlock, for this mapping only: leave the rest of the
            # pytest process (torch's own shared memory) alone.
            return ph.mlock_range(start, nbytes) if start == ours[0].start else 0

        before = vm_locked_bytes()
        report = lock_pinned_host_mappings(mlock=lock_only_ours)
        after = vm_locked_bytes()
        assert report.failed == 0, report
        assert report.locked >= 1 and report.locked_bytes >= size
        assert before is not None and after is not None
        assert after - before >= size
        assert report.vm_locked_after == after
    finally:
        ctypes.CDLL(None).munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        del view
        region.close()
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
