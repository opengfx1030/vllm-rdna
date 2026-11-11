# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KFD-buffer-versus-VmLck check, on a fake /proc tree."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "pinned_fdinfo_check", ROOT / "tools/rdna/port_v031/pinned_fdinfo_check.py"
)
assert _spec is not None and _spec.loader is not None
fc = importlib.util.module_from_spec(_spec)
sys.modules["pinned_fdinfo_check"] = fc
_spec.loader.exec_module(fc)

# What amdgpu prints per render-node fd: bytes, KiB or MiB, whichever is exact.
FDINFO_OWN = """\
pos:\t0
drm-driver:\tamdgpu
drm-pdev:\t0000:ce:00.0
drm-client-id:\t9854
drm-total-cpu:\t25547328 KiB
drm-shared-cpu:\t0
drm-total-vram:\t27257156 KiB
"""
FDINFO_PEER = FDINFO_OWN.replace("ce:00.0", "d3:00.0").replace(
    "25547328 KiB", "25387360 KiB"
)
FDINFO_MIB = FDINFO_OWN.replace("ce:00.0", "07:00.0").replace("25547328 KiB", "324 MiB")
FDINFO_OTHER = "pos:\t0\nflags:\t02000002\nmnt_id:\t23\n"

TABLE_KIB = 25000120
SMALL_KIB = 547208
MAPS = f"""\
7f0000000000-7f0000001000 r--p 00000000 00:00 0
7f1000000000-{0x7F1000000000 + TABLE_KIB * 1024:x} rw-s 00000000 00:01 7 \
 /dev/zero (deleted)
7f9000000000-{0x7F9000000000 + SMALL_KIB * 1024:x} rw-s 00000000 00:01 8 \
 /dev/zero (deleted)
7fa000000000-7fa001300000 rw-s 00000000 00:2f 9  /dev/shm/nccl-qYzA51 (deleted)
7fb000000000-7fb010000000 rw-s 00000000 00:2f 10  /dev/shm/psm_00d45b71
7fc000000000-7fc000100000 rw-p 00000000 00:00 0
"""


def _proc(root: Path, pid: int, vmlck_kib: int, fdinfos: dict[str, str]) -> None:
    (root / str(pid) / "fdinfo").mkdir(parents=True)
    for fd, text in fdinfos.items():
        (root / str(pid) / "fdinfo" / fd).write_text(text)
    (root / str(pid) / "maps").write_text(MAPS)
    (root / str(pid) / "status").write_text(
        f"Name:\tVLLM::Worker\nVmPin:\t0 kB\nVmLck:\t{vmlck_kib} kB\nVmRSS:\t1 kB\n"
    )


def test_fdinfo_units_and_other_fds():
    assert fc.size_bytes("25547328 KiB") == 25547328 * 1024
    assert fc.size_bytes("324 MiB") == 324 * 1024 * 1024
    assert fc.size_bytes("4096") == 4096
    assert fc.parse_fdinfo(FDINFO_OWN) == ("0000:ce:00.0", 25547328 * 1024)
    assert fc.parse_fdinfo(FDINFO_MIB) == ("0000:07:00.0", 324 * 1024 * 1024)
    assert fc.parse_fdinfo(FDINFO_OTHER) is None  # not a DRM fd
    assert fc.parse_fdinfo("drm-pdev:\t0000:ce:00.0\n") is None  # no totals


def test_pinned_mappings_counts_only_the_zero_shmem_blocks():
    count, size = fc.pinned_mappings(MAPS)
    assert (count, size) == (2, (TABLE_KIB + SMALL_KIB) * 1024)
    assert fc.pinned_mappings("") == (0, 0)


def test_the_unlocked_part_is_the_total_minus_vmlck(tmp_path: Path):
    _proc(
        tmp_path,
        42,
        TABLE_KIB,  # the base arm: only the table is locked
        {"3": FDINFO_OTHER, "10": FDINFO_OWN, "11": FDINFO_PEER, "12": FDINFO_MIB},
    )
    row = fc.check(42, str(tmp_path))
    assert row["own_gpu"] == "0000:ce:00.0"
    assert row["total_cpu"] == (TABLE_KIB + SMALL_KIB) * 1024
    assert row["unmapped_diff"] == 0  # the buffers are exactly the mapped blocks
    assert row["unlocked"] == SMALL_KIB * 1024
    assert row["peer_total_cpu"] == [324 * 1024 * 1024, 25387360 * 1024]


def test_exit_status_follows_the_unlocked_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _proc(tmp_path, 7, TABLE_KIB, {"10": FDINFO_OWN})
    _proc(tmp_path, 8, TABLE_KIB + SMALL_KIB - 100, {"10": FDINFO_OWN})
    argv = ["--proc", str(tmp_path)]
    assert fc.main([*argv, "7"]) == 0  # no limit given: report only
    assert fc.main([*argv, "7", "--max-unlocked-kib", "1024"]) == 1  # base arm
    capsys.readouterr()
    assert fc.main([*argv, "8", "--max-unlocked-kib", "1024"]) == 0  # fix arm
    out = capsys.readouterr().out
    assert "| 8 | 0000:ce:00.0 | 25547328 | 2 (25547328) | 0 | 25547228 | 100 |" in out


def test_a_process_without_drm_fds_is_an_error(tmp_path: Path):
    _proc(tmp_path, 9, TABLE_KIB, {"3": FDINFO_OTHER})
    with pytest.raises(ValueError, match="no DRM fdinfo"):
        fc.check(9, str(tmp_path))


def test_series_prints_a_line_per_worker_and_tick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _proc(tmp_path, 7, TABLE_KIB, {"10": FDINFO_OWN})
    _proc(tmp_path, 8, TABLE_KIB + SMALL_KIB, {"10": FDINFO_OWN})
    argv = ["--proc", str(tmp_path), "7", "8", "--every", "0", "--count", "3"]
    assert fc.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert lines[0].endswith(
        "pid=7 total_kib=25547328 mappings=2 vmlck_kib=25000120 unlocked_kib=547208"
    )
    assert lines[1].endswith(
        "pid=8 total_kib=25547328 mappings=2 vmlck_kib=25547328 unlocked_kib=0"
    )


def test_series_ends_when_a_worker_exits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    _proc(tmp_path, 7, TABLE_KIB, {"10": FDINFO_OWN})
    sleeps: list[float] = []

    def exit_worker(seconds: float) -> None:
        sleeps.append(seconds)
        shutil.rmtree(tmp_path / "7")

    monkeypatch.setattr(fc.time, "sleep", exit_worker)
    argv = ["--proc", str(tmp_path), "7", "--every", "1", "--max-unlocked-kib", "1"]
    assert fc.main(argv) == 1  # the first tick was over the limit
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert sleeps == [1.0]


def test_series_of_a_stale_pid_fails_at_once(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        fc.main(["--proc", str(tmp_path), "7", "--every", "0", "--count", "2"])
