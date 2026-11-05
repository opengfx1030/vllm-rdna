# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-memory eviction sampler and its report, on fake /proc trees."""

import argparse
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"tools/rdna/port_v031/{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load("pinned_sampler")
pr = _load("pinned_report")

PAGE = ps.PAGE
PRESENT = ps.PM_PRESENT
SWAPPED = ps.PM_SWAPPED
DEVZERO = "/dev/zero (deleted)"


def entries(*vals: int) -> bytes:
    return struct.pack(f"<{len(vals)}Q", *vals)


def present(pfn: int) -> int:
    return PRESENT | pfn


def swap(offset: int = 1) -> int:
    return SWAPPED | (offset << 5)


@pytest.mark.parametrize(
    "perms, path, size, expected",
    [
        ("rw-s", DEVZERO, 4096, "pinned"),
        ("rw-s", "/dev/shm/nccl-x (deleted)", 1 << 20, "shm"),
        ("rw-s", "/dev/shm/nccl-x (deleted)", 4096, None),
        ("rw-s", "/memfd:torch (deleted)", 1 << 20, "shm"),
        ("rw-p", "", 2 << 20, "anon"),
        ("rw-p", "", 4096, None),
        ("rw-p", "[heap]", 64 << 20, None),
        ("rw-p", "/home/u/blobs/x.safetensors", 1 << 30, None),
        ("rw-s", "/dev/kfd", 8192, None),
        ("r--s", DEVZERO, 4096, None),
        ("---p", "", 4 << 20, None),
    ],
)
def test_classify(perms: str, path: str, size: int, expected: str | None):
    assert ps.classify(perms, path, size) == expected


def test_proc_files_of_a_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    (proc / "101").mkdir()
    (proc / "102").mkdir()
    (proc / "100" / "cmdline").write_bytes(b"VLLM::Worker_TP1_EP1\0")
    (proc / "101" / "cmdline").write_bytes(b"/usr/bin/python\0serve\0")
    (proc / "102" / "cmdline").write_bytes(b"VLLM::EngineCore\0")
    (proc / "100" / "stat").write_text(
        "100 (VLLM::Worker_TP) S 77 100 100 0 -1 0 0 0 0 0 5 6\n"
    )
    (proc / "100" / "status").write_text(
        "Name:\tVLLM::Worker_TP\nVmLck:\t  24576 kB\nVmSwap:\t      0 kB\n"
        "RssAnon:\t  100 kB\nRssFile:\t  200 kB\nRssShmem:\t 300 kB\n"
    )
    (proc / "100" / "maps").write_text(
        "7f0000000000-7f0000100000 rw-s 00000000 00:01 11 /dev/zero (deleted)\n"
        "7f0000200000-7f0000201000 rw-p 00000000 00:00 0\n"
        "7f0000300000-7f0000500000 rw-p 00000000 00:00 0\n"
        "7f0000600000-7f0000700000 r-xp 00000000 08:01 12 /lib64/libc.so.6\n"
        "7f0000800000-7f0000900000 rw-s 00000000 00:05 13 /dev/shm/nccl-a (deleted)\n"
    )
    monkeypatch.setattr(ps, "PROC", str(proc))
    assert ps.worker_pids(str(proc)) == {100: "VLLM::Worker_TP1_EP1"}
    assert ps.parent_pid(100) == 77
    assert ps.cpu_ticks(100) == 11
    assert ps.process_status(100) == {
        "vmlck_kb": 24576,
        "vmswap_kb": 0,
        "rss_anon_kb": 100,
        "rss_file_kb": 200,
        "rss_shmem_kb": 300,
    }
    vmas = ps.read_maps(100)
    assert vmas is not None
    assert [(hex(v.start), v.cls, v.inode) for v in vmas] == [
        ("0x7f0000000000", "pinned", 11),
        ("0x7f0000300000", "anon", 0),
        ("0x7f0000800000", "shm", 13),
    ]
    assert ps.read_maps(999) is None


def test_process_tree_follows_one_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    proc = tmp_path / "proc"
    for pid, ppid in ((10, 1), (11, 10), (12, 11), (13, 11), (20, 1), (21, 20)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(f"{pid} (a b) S {ppid} 0 0\n")
    (proc / "self").mkdir()
    monkeypatch.setattr(ps, "PROC", str(proc))
    assert ps.process_tree({10}) == {10, 11, 12, 13}
    assert ps.process_tree({10, 20}) == {10, 11, 12, 13, 20, 21}
    assert ps.process_tree({99}) == {99}


def test_evicted_ms_is_the_largest_stat(tmp_path: Path):
    for gpu, ms in (("1", 40), ("2", 7000), ("3", 7000)):
        d = tmp_path / "55" / f"stats_{gpu}"
        d.mkdir(parents=True)
        (d / "evicted_ms").write_text(f"{ms}\n")
    (tmp_path / "55" / "queues").mkdir()
    assert ps.evicted_ms(55, str(tmp_path)) == 7000
    assert ps.evicted_ms(56, str(tmp_path)) is None


def test_read_pagemap_chunks(tmp_path: Path):
    start = 0x100000  # page 256: entry offset 2048
    body = entries(*[present(10 + i) for i in range(6)])
    f = tmp_path / "pagemap"
    f.write_bytes(b"\0" * (start // PAGE * 8) + body)
    fd = ps.os.open(str(f), ps.os.O_RDONLY)
    try:
        assert b"".join(ps.read_pagemap(fd, start, start + 6 * PAGE)) == body
        assert ps.read_pagemap(fd, start, start + 8 * PAGE) is None  # past EOF
    finally:
        ps.os.close(fd)


def test_diff_pagemap_classifies_every_change():
    before = entries(
        present(1),  # stays
        present(2),  # moves to pfn 20
        present(3),  # swapped out
        present(4),  # dropped
        swap(),  # swapped in
        0,  # populated
        present(7),  # stays
    )
    after = entries(
        present(1),
        present(20),
        swap(),
        0,
        present(5),
        present(6),
        present(7),
    )
    assert ps.diff_pagemap([before], [after]) == {
        "moved": 1,
        "swapped_out": 1,
        "dropped": 1,
        "swapped_in": 1,
        "populated": 1,
    }
    assert ps.diff_pagemap([before], [before]) == dict.fromkeys(
        ("moved", "swapped_out", "dropped", "swapped_in", "populated"), 0
    )
    assert ps.count_state([after]) == (5, 1)


def _kpageflags(tmp_path: Path, flags: dict[int, int]) -> int:
    f = tmp_path / "kpageflags"
    size = (max(flags) + 1) * 8
    buf = bytearray(size)
    for pfn, val in flags.items():
        buf[pfn * 8 : pfn * 8 + 8] = struct.pack("<Q", val)
    f.write_bytes(bytes(buf))
    return ps.os.open(str(f), ps.os.O_RDONLY)


def test_sample_lock_state_reads_kpageflags(tmp_path: Path):
    locked = (1 << ps.KPF_MLOCKED) | (1 << ps.KPF_UNEVICTABLE)
    lru = 1 << ps.KPF_LRU
    kpf = _kpageflags(tmp_path, {1: locked, 2: locked, 3: lru, 4: lru})
    try:
        chunk = entries(*[present(p) for p in (1, 2, 3, 4)])
        state = ps.sample_lock_state(kpf, [chunk], 4, want=4)
        assert state == {"n": 4, "mlocked": 2, "unevictable": 2, "lru": 2}
        # fewer samples than pages: evenly spaced, still bounded by `want`
        assert ps.sample_lock_state(kpf, [chunk], 4, want=2)["n"] == 2
        assert ps.sample_lock_state(kpf, [chunk], 0) is None
        # hidden PFNs (no CAP_SYS_ADMIN) read as 0
        assert ps.sample_lock_state(kpf, [entries(PRESENT, PRESENT)], 2) is None
    finally:
        ps.os.close(kpf)
    assert (
        ps.lock_class({"n": 16, "mlocked": 16, "unevictable": 16, "lru": 0}) == "locked"
    )
    assert (
        ps.lock_class({"n": 16, "mlocked": 0, "unevictable": 0, "lru": 16})
        == "unlocked"
    )
    assert (
        ps.lock_class({"n": 0, "mlocked": 0, "unevictable": 0, "lru": 0}) == "unknown"
    )
    assert ps.lock_class(None) == "unknown"


def test_sync_workers_follows_new_ones_drops_gone_ones_and_refreshes_titles(
    monkeypatch: pytest.MonkeyPatch,
):
    closed: list[int] = []

    class Fake:
        def __init__(self, pid: int, title: str) -> None:
            self.pid, self.title = pid, title

        def close(self) -> None:
            closed.append(self.pid)

    monkeypatch.setattr(ps, "WorkerState", Fake)
    workers: dict = {}
    ps.sync_workers(workers, {10: "VLLM::Worker", 11: "VLLM::Worker"})  # at the fork
    ps.sync_workers(workers, {10: "VLLM::Worker_TP0_EP0", 12: "VLLM::Worker"})
    assert {p: w.title for p, w in workers.items()} == {
        10: "VLLM::Worker_TP0_EP0",  # the rank arrived
        12: "VLLM::Worker",
    }
    assert closed == [11]  # 11 left

    def refuse(pid: int, title: str) -> None:
        raise OSError("process gone")

    monkeypatch.setattr(ps, "WorkerState", refuse)
    ps.sync_workers(workers, {10: "VLLM::Worker_TP0_EP0", 13: "VLLM::Worker"})
    assert 13 not in workers and closed == [11, 12]  # a vanished pid is skipped


def test_scan_worker_logs_the_mapping_whose_pages_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    proc = tmp_path / "proc"
    (proc / "300").mkdir(parents=True)
    small, big = 0x100000, 0x200000  # two pinned mappings, 4 and 2 pages
    (proc / "300" / "maps").write_text(
        f"{small:x}-{small + 4 * PAGE:x} rw-s 00000000 00:01 21 {DEVZERO}\n"
        f"{big:x}-{big + 2 * PAGE:x} rw-s 00000000 00:01 22 {DEVZERO}\n"
    )
    (proc / "300" / "stat").write_text("300 (w) S 1 0 0 0 -1 0\n")
    pagemap = proc / "300" / "pagemap"

    def write(small_pfns: list[int], big_pfns: list[int]) -> None:
        buf = bytearray((big // PAGE + 2) * 8)
        for base, pfns in ((small, small_pfns), (big, big_pfns)):
            for i, pfn in enumerate(pfns):
                at = (base // PAGE + i) * 8
                buf[at : at + 8] = struct.pack("<Q", present(pfn))
        pagemap.write_bytes(bytes(buf))

    monkeypatch.setattr(ps, "PROC", str(proc))
    args = argparse.Namespace(
        big_gib=1.0, slow_period=5.0, lock_period=30.0, discover_period=5.0
    )
    kpf = _kpageflags(
        tmp_path,
        dict.fromkeys(range(1, 100), (1 << ps.KPF_MLOCKED) | (1 << ps.KPF_UNEVICTABLE)),
    )
    out = io.StringIO()
    write([1, 2, 3, 4], [50, 51])
    w = ps.WorkerState(300, "VLLM::Worker_TP0_EP0")
    try:
        first = ps.scan_worker(w, 100.0, args, kpf, out)
        assert out.getvalue() == ""  # nothing to compare with yet
        assert first["pinned"]["n"] == 2
        for key in ((small, small + 4 * PAGE), (big, big + 2 * PAGE)):
            assert ps.lock_class(w.lock[key]) == "locked"
        assert first["pinned_gib_by_lock"]["unlocked"] == 0.0
        write([1, 2, 3, 4], [50, 51])
        assert ps.scan_worker(w, 101.0, args, kpf, out)["moved"] == 0
        assert out.getvalue() == ""
        write([1, 77, 3, 4], [50, 51])  # one page of the small mapping moved
        second = ps.scan_worker(w, 102.0, args, kpf, out)
        assert second["moved"] == 1 and second["swapped_out"] == 0
    finally:
        w.close()
        ps.os.close(kpf)
    (rec,) = [json.loads(line) for line in out.getvalue().splitlines()]
    assert rec["pid"] == 300 and rec["cls"] == "pinned"
    assert rec["start"] == hex(small) and rec["size"] == 4 * PAGE
    assert rec["moved"] == 1 and rec["inode"] == 21
    assert rec["dt"] == 1.0 and rec["t"] == 102.0
    assert rec["lock"]["mlocked"] == rec["lock"]["n"] == 4


KFD_LOG = """\
# start 2026-11-05T13:00:00.000000 ['0000:47:00.0']
13:00:10.250000 0000:47:00.0 9 5a1 -700 20e9 1
13:00:10.300000 0000:47:00.0 9 5a2 -701 20e9 1
13:00:12.250000 0000:47:00.0 a 5a9 -700 20e9 R
13:00:12.500000 0000:47:00.0 a 5aa -700 20e9
13:00:13.000000 0000:47:00.0 9 5b0 -700 20e9 2
13:00:14.000000 0000:47:00.0 b 5b1 -700 @7f00(4096) 20e9 0
garbage line
"""


def test_parse_and_pair_kfd_events(tmp_path: Path):
    log = tmp_path / "kfd.log"
    log.write_text(KFD_LOG)
    events = pr.parse_kfd_events(str(log))
    assert [(e.kind, e.pid, e.trigger) for e in events] == [
        ("evict", 700, "1"),
        ("evict", 701, "1"),
        ("restore", 700, "R"),
        ("restore", 700, ""),
        ("evict", 700, "2"),
    ]
    evictions = pr.pair_events(events)
    # a restore with R is a rescheduled attempt: the first real restore ends the
    # eviction, and a never-restored one stays open
    assert [
        (v.start.pid, v.start.trigger, v.end is not None, v.resched, v.again)
        for v in evictions
    ] == [
        (700, "1", True, 1, 0),
        (701, "1", False, 0, 0),
        (700, "2", False, 0, 0),
    ]
    first = evictions[0]
    assert first.end is not None and first.end.t - first.start.t == pytest.approx(2.25)


def _tick(t: float, ms: int, scan: int = 0, pid: str = "700") -> dict:
    return {
        "t": t,
        "vm": {"pgscan_kswapd": scan, "pgscan_direct": 0, "nr_free_pages": 1000},
        "psi": {"some": int(t * 1000), "full": 0},
        "w": {
            pid: {
                "title": "VLLM::Worker_TP2_EP2",
                "ppid": 5,
                "evicted_ms": ms,
                "vmlck_kb": 1 << 20,
                "maps": {
                    "pinned": {"n": 3},
                    "pinned_gib_by_lock": {"locked": 1.0, "unlocked": 0.5},
                },
            }
        },
    }


def test_jumps_and_reclaim_bursts():
    ticks = [
        _tick(100, 0, 0),
        _tick(101, 0, 10),
        _tick(102, 0, 60000),
        _tick(103, 900, 160000),
        _tick(104, 900, 160010),
    ]
    (jump,) = pr.eviction_jumps(ticks)
    assert (jump.label, jump.ms, jump.ta, jump.tb) == ("TP2", 900, 102, 103)
    (burst,) = pr.reclaim_bursts(ticks)
    assert burst.kind == "scan" and burst.t0 == 101 and burst.t1 == 103
    assert burst.total == 160000 - 10 and burst.peak_per_s == 100000


def test_report_names_the_mapping_that_moved_in_an_eviction_burst():
    ticks = [
        _tick(100 + i, 0 if i < 5 else 900, 0 if i < 3 else 80000 * (i - 2))
        for i in range(8)
    ]
    pagemap = [
        # moved in the second before the jump: related to the burst
        {
            "t": 104.5,
            "dt": 1.0,
            "pid": "700",
            "cls": "pinned",
            "start": "0x7f1000",
            "size": 4 << 20,
            "moved": 12,
            "swapped_out": 3,
            "dropped": 0,
            "lock": {"n": 16, "mlocked": 0, "unevictable": 0, "lru": 16},
        },
        # long before: not related
        {
            "t": 100.5,
            "dt": 1.0,
            "pid": "700",
            "cls": "pinned",
            "start": "0x7f9000",
            "size": 1 << 20,
            "moved": 99,
            "swapped_out": 0,
            "dropped": 0,
            "lock": {"n": 16, "mlocked": 16, "unevictable": 16, "lru": 0},
        },
    ]
    events = [
        pr.KfdEvent(103.6, "evict", 700, "20e9", "1"),
        pr.KfdEvent(104.5, "restore", 700, "20e9", "R"),
    ]
    text = pr.build_report({"kernel": "7.1"}, ticks, pagemap, events, slack=0.5)
    assert "## 1. Evictions per worker" in text
    assert "| 700 | TP2 | 5 | 1 | 900 | 900 |" in text
    assert "### Burst 1" in text and "reclaim near: ['scan']" in text
    burst_part = text.split("### Burst 1")[1].split("## 4.")[0]
    assert "0x7f1000" in burst_part and "| 12 | 3 |" in burst_part
    assert "0x7f9000" not in burst_part
    # the restore was rescheduled (R): the queues are still evicted, no end
    assert "| 700 | 20e9 | +3.6s | 1 |  | 1 | 0 |" in burst_part
    assert (
        "1 QUEUE_EVICTION (one per process and GPU) by trigger "
        "(0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND) {'1': 1}; 0 restored, 1 not restored "
        "(0 of them at a worker exit); 1 restores were rescheduled" in text
    )
    totals = text.split("## 4.")[1]
    assert "| pinned | unlocked | 12 | 3 |" in totals
    assert "| pinned | locked | 99 | 0 |" in totals
    assert pr.build_report({}, [], [], []) == "no samples\n"


def test_report_counts_the_seconds_with_a_pinned_move_against_those_under_eviction():
    def rec(t: float, dt: float, cls: str, moved: int = 0, dropped: int = 0) -> dict:
        return {
            "t": t,
            "dt": dt,
            "pid": "700",
            "cls": cls,
            "start": f"0x{int(t * 1000):x}",
            "size": 1 << 20,
            "moved": moved,
            "swapped_out": 0,
            "dropped": dropped,
            "lock": {"n": 16, "mlocked": 0, "unevictable": 0, "lru": 16},
        }

    # seconds 100..109; the counter jumps by 900 ms between 104 and 105: an
    # eviction that ended in (104, 105] and began after 103.1, so the bins
    # 103, 104 and 105 are under eviction
    ticks = [_tick(100 + i, 0 if i < 5 else 900) for i in range(10)]
    pagemap = [
        rec(104.5, 1.0, "pinned", moved=5),  # bins 103, 104
        rec(108.5, 1.0, "pinned", moved=2),  # bins 107, 108: nobody evicted
        rec(101.5, 1.0, "anon", moved=7),  # bins 100, 101
        rec(106.5, 0.5, "pinned", dropped=40),  # an unmap is not a move
    ]
    jumps = pr.eviction_jumps(ticks)
    assert pr.move_seconds(ticks, jumps, pagemap) == [
        ["pinned pages moved or swapped out", 4, 2],
        ["only other pages moved or swapped out", 2, 0],
        ["nothing moved", 4, 1],
    ]
    text = pr.build_report({}, ticks, pagemap, [])
    assert "| pinned pages moved or swapped out | 4 | 2 (50 %) |" in text
    assert "| nothing moved | 4 | 1 (25 %) |" in text
    # without pagemap data "nothing moved" would be a claim about nothing
    assert "Seconds of the run by what moved" not in pr.build_report({}, ticks, [], [])


def test_report_summarises_each_burst_in_one_row_before_the_details():
    def tick(t: float, ms: int, scan: int) -> dict:
        row = _tick(t, ms, scan)
        row["vm"].update(
            pgmigrate_success=scan // 10,
            compact_stall=scan // 40000,
            pswpout=scan // 64000,
            compact_migrate_scanned=scan * 2,
            compact_daemon_migrate_scanned=scan * 3 // 2,
            compact_daemon_wake=scan // 160000,
        )
        return row

    ticks = [
        tick(100 + i, 0 if i < 5 else 900, 0 if i < 3 else 80000 * (i - 2))
        for i in range(8)
    ]

    def rec(cls: str, moved: int, swapped: int, mlocked: int) -> dict:
        return {
            "t": 104.5,
            "dt": 1.0,
            "pid": "700",
            "cls": cls,
            "start": "0x7f1000",
            "size": 4 << 20,
            "moved": moved,
            "swapped_out": swapped,
            "dropped": 0,
            "lock": {"n": 16, "mlocked": mlocked, "unevictable": mlocked, "lru": 16},
        }

    pagemap = [
        rec("pinned", 12, 3, 0),
        rec("pinned", 4, 0, 16),
        rec("shm", 7, 0, 0),
        rec("anon", 100, 1, 0),
    ]
    events = [
        pr.KfdEvent(103.6, "evict", 700, "20e9", "1"),
        pr.KfdEvent(104.5, "restore", 700, "20e9", "R"),
    ]
    text = pr.build_report({"kernel": "7.1"}, ticks, pagemap, events, slack=0.5)
    section = text.split("## 3.")[1]
    summary = section.split("### Burst 1")[0]
    assert (
        "| burst | window | evicted (counter) | reclaim near | host counters |"
        in summary
    )
    (row,) = [ln for ln in summary.splitlines() if ln.startswith("| 1 |")]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "TP2 900 ms" and cells[3] == "compact, scan, stall"
    assert cells[4] == (
        "scan 320k/0, migrated 32k, compaction scan 640k (kcompactd 480k, 2 wakes), "
        "stalls 8, swap-out 5"
    )
    # pinned unlocked 12 / 3, pinned locked 4 / 0, shm 7 / 0, anon 100 / 1
    assert cells[5:9] == ["12 / 3", "4 / 0", "7 / 0", "100 / 1"]
    assert cells[9] == "none"
    assert pr.human(2713924) == "2.71M" and pr.human(44007) == "44k"
    assert pr.human(999) == "999" and pr.human(0) == "0"


def test_bursts_are_labelled_with_the_proof_window_they_start_in(tmp_path: Path):
    windows = [
        {"name": "idle1", "t0": 90.0, "t1": 100.0},
        {"name": "load1", "t0": 100.0, "t_end": 103.0, "t1": 106.0},
    ]
    assert pr.window_label(95.0, windows) == "idle1 (idle)"
    assert pr.window_label(103.0, windows) == "load1 (neighbour running)"
    assert pr.window_label(104.0, windows) == "load1 (settling)"
    assert pr.window_label(120.0, windows) == "gap"

    rows = [{"meta": {"start": 100.0, "kernel": "7.1", "sysctl": {}}}]
    rows += [_tick(100 + i, 0 if i < 5 else 900) for i in range(8)]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "windows.json").write_text(json.dumps({"windows": windows}))
    plain = tmp_path / "plain.md"
    labelled = tmp_path / "labelled.md"
    assert pr.main([str(tmp_path), "--md", str(plain)]) == 0
    assert (
        pr.main(
            [
                str(tmp_path),
                "--windows",
                str(tmp_path / "windows.json"),
                "--md",
                str(labelled),
            ]
        )
        == 0
    )
    assert "proof window" not in plain.read_text()
    text = labelled.read_text()
    assert "| weight load | proof window |" in text
    # the burst starts at +3.1 s = 103.1: after the neighbour left, in the settle
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| 1 |")]
    assert row.rstrip(" |").endswith("| load1 (settling)")


def _vtick(t: float, **vm: int) -> dict:
    tick = _tick(t, 0)
    tick["vm"].update(vm)
    return tick


def test_kcompactd_passes_say_whether_kswapd_ran_before_them():
    ticks = [
        _vtick(100, pageoutrun=0, compact_daemon_wake=0, pgscan_kswapd=0),
        _vtick(101, pageoutrun=3, compact_daemon_wake=2, pgscan_kswapd=860_000),
        _vtick(102, pageoutrun=6, compact_daemon_wake=4, pgscan_kswapd=1_700_000),
        _vtick(103, pageoutrun=6, compact_daemon_wake=4, pgscan_kswapd=1_700_000),
        _vtick(
            118,
            pageoutrun=6,
            compact_daemon_wake=4,
            pgscan_kswapd=1_700_000,
            compact_daemon_migrate_scanned=22_000_000,
            pgmigrate_success=4_000_000,
        ),
        # 200 s later, kswapd idle all along: the proactive path
        _vtick(318, pageoutrun=6, compact_daemon_wake=4, pgscan_kswapd=1_700_000)
        | {
            "vm": {
                "pageoutrun": 6,
                "compact_daemon_wake": 4,
                "pgscan_kswapd": 1_700_000,
                "compact_daemon_migrate_scanned": 30_000_000,
            }
        },
    ]
    follows, proactive = pr.kcompactd_passes(ticks)
    assert follows["t"] == 118 and follows["scanned"] == 22_000_000
    assert follows["migrated"] == 4_000_000
    assert (follows["wakes"], follows["kswapd_passes"]) == (4, 6)
    assert follows["kswapd_scan"] == 1_700_000 and follows["since_kswapd"] == 16
    assert proactive["kswapd_passes"] == 0 and proactive["wakes"] == 0
    assert proactive["since_kswapd"] == 216
    # a tick under the threshold is not a pass
    assert pr.kcompactd_passes(ticks[:4]) == []


def test_report_has_a_row_per_proof_window_and_lists_the_kcompactd_passes():
    windows = [
        {"name": "idle1", "t0": 100.0, "t1": 102.0},
        {
            "name": "load1",
            "t0": 102.0,
            "t_end": 108.0,
            "t1": 110.0,
            "evicted_ms": {"Worker_TP1_EP1": 1500, "Worker_TP0_EP0": 400},
        },
        {"name": "idle9", "t0": 500.0, "t1": 530.0},  # the samples stop at 110
    ]
    ticks = [
        _vtick(
            100 + i,
            pageoutrun=i,
            pgscan_kswapd=1000 * i,
            pgscan_direct=10 * i,
            compact_stall=i // 2,
            pgmigrate_success=100 * i,
            pswpout=5 * i,
            compact_daemon_migrate_scanned=2_000_000 if i == 6 else 0,
            nr_free_pages=(10 - i) << 18,
        )
        for i in range(11)
    ]
    for i, tick in enumerate(ticks):  # the counters are cumulative
        if i > 6:
            tick["vm"]["compact_daemon_migrate_scanned"] = 2_000_000
    text = pr.build_report({}, ticks, [], [], windows=windows)
    rows = [ln for ln in text.splitlines() if ln.startswith(("| idle", "| load"))]
    assert [r.split(" | ")[0] for r in rows] == ["| idle1", "| load1"]
    load = rows[1].split(" | ")
    # TP0 before TP1; kswapd 8k, direct 80, 4 stalls, 800 migrated, 40 swapped out,
    # the daemon's 2M scan, lowest MemFree 0 pages' worth at tick 110: 0.0 GiB
    assert load[2] == "TP0 400, TP1 1500"
    assert load[3:9] == ["8k", "80", "4", "800", "40", "2M"]
    assert load[9].rstrip(" |") == "0.0"
    assert "| +6.0s | 2M |" in text  # relative to the first tick
    assert "1 of 1 passes follow kswapd passes within 60 s; 0 came with" in text


def _wtick(t: float, ms: int, rss_kb: dict[str, int], ppid: int = 9) -> dict:
    """Victim 700 (server 5, evicted_ms ``ms``) and loaders at the given VmRSS."""
    workers: dict[str, dict] = {
        "700": {
            "title": "VLLM::Worker_TP0_EP0",
            "ppid": 5,
            "evicted_ms": ms,
            "rss_kb": 30 << 20,
            "rss_shmem_kb": 25 << 20,
        }
    }
    for pid, rss in rss_kb.items():
        workers[pid] = {
            "title": "VLLM::Worker_TP1_EP1",
            "ppid": ppid,
            "evicted_ms": 0,
            "rss_kb": rss,
            "rss_shmem_kb": rss // 2,
            "rss_anon_kb": rss // 4,
            "rss_file_kb": rss // 4,
        }
    vm = {"pgscan_kswapd": int(t) * 1000, "pgscan_direct": int(t) * 500}
    return {
        "t": t,
        "vm": vm,
        "psi": {"some": 0, "full": 0},
        "w": workers,
        "tick_s": 0.25 if t == 130.0 else 0.01,
    }


def _boot_ticks(ppid: int = 9) -> list[dict]:
    """800 loads 40 GiB at 2 GiB/s from t=110 to t=130; 700 is evicted at t=125."""
    ticks = []
    for i in range(60):
        grown = min(max(i - 10, 0), 20) * (2 << 20)  # kB
        ms = 0 if i < 25 else 1500
        ticks.append(_wtick(100.0 + i, ms, {"800": (1 << 20) + grown}, ppid))
    return ticks


def test_weight_loads_find_a_boot_and_ignore_steady_workers():
    (load,) = pr.weight_loads(_boot_ticks())
    assert (load.pid, load.label, load.ppid) == ("800", "TP1", 9)
    assert (load.t0, load.t1) == (110.0, 130.0)
    assert load.gib == pytest.approx(40.0) and load.peak_gib_s == pytest.approx(2.0)
    assert load.shmem_gib == pytest.approx(20.0)
    steady = [_wtick(100.0 + i, 0, {"800": 1 << 20}) for i in range(30)]
    assert pr.weight_loads(steady) == []
    # a 3 GiB growth is not a weight load, a 12 GiB one in two bursts is one
    small = [
        _wtick(100.0 + i, 0, {"800": (1 << 20) * (1 + min(i, 3))}) for i in range(9)
    ]
    assert pr.weight_loads(small) == []
    gap = [
        _wtick(100.0 + i, 0, {"800": (1 << 20) * (1 + 6 * (i > 2) + 6 * (i > 12))})
        for i in range(30)
    ]
    (joined,) = pr.weight_loads(gap)
    assert joined.gib == pytest.approx(12.0) and (joined.t0, joined.t1) == (102, 113)


def test_report_correlates_an_eviction_with_a_neighbour_load():
    text = pr.build_report({"kernel": "7.1"}, _boot_ticks(), [], [], slack=0.5)
    assert "Sampler cost per tick: mean 14 ms, max 250 ms." in text
    loads = text.split("### Weight loads of the followed workers")[1].split("## 3.")[0]
    assert "| TP1 | 800 | 9 | +10.0s..+30.0s | 40.0 | 20.0/10.0/10.0 | 2.00 |" in loads
    # 20 s at 500 direct and 1000 kswapd pages/s while it ran; 700 lost 1500 ms
    assert "| 10000/20000 | 0 | 0 | 1500 | 1 |" in loads
    assert "1 of 1 weight loads overlap an eviction of a worker of another" in loads
    burst = text.split("### Burst 1")[1].split("## 4.")[0]
    assert "Weight loads in progress: TP1 pid 800 (server 9, neighbour)" in burst
    # the same server booting its own worker is labelled as such
    own = pr.build_report({}, _boot_ticks(ppid=5), [], [], slack=0.5)
    assert "(server 5, the same server)" in own
    assert "0 of 1 weight loads overlap" in own


def test_report_says_when_no_worker_loaded_weights(tmp_path: Path):
    quiet = [_wtick(100.0 + i, 0 if i < 5 else 900, {}) for i in range(8)]
    text = pr.build_report({}, quiet, [], [])
    assert "No worker grew by 4 GiB or more at 0.05 GiB/s or more." in text
    assert "No weight load of a followed worker in progress." in text


def test_report_cli_reads_the_sampler_output(tmp_path: Path):
    rows = [{"meta": {"start": 100.0, "kernel": "7.1", "sysctl": {}}}]
    rows += [_tick(100 + i, 0 if i < 5 else 900) for i in range(8)]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "report.md"
    assert pr.main([str(tmp_path), "--md", str(out)]) == 0  # no pagemap.jsonl
    text = out.read_text()
    assert "### Burst 1" in text
    assert "No page of a followed mapping moved" in text


# Lines of the real 11-04 raw log (par1-cs25, kfd_smi_raw.py): two evictions that
# were restored at once, one of 29 s with four rescheduled restores before the
# real one, and one that was evicted again after a reschedule and a restore.
REAL_KFD_LOG = """\
# start 2026-11-04T10:41:17.454332 ['0000:07:00.0', '0000:47:00.0']
10:48:41.580881 0000:47:00.0 9 224366324019700 -1704018 2c59 1
10:48:41.589700 0000:47:00.0 a 224366332833176 -1704018 2c59 0
10:48:41.590782 0000:47:00.0 9 224366333936636 -1704018 2c59 1
10:48:41.594954 0000:47:00.0 a 224366338112287 -1704018 2c59 0
10:52:05.200824 0000:47:00.0 9 224569943940724 -1704020 2c59 1
10:52:05.896715 0000:47:00.0 a 224570639874649 -1704020 2c59 R
10:52:06.258511 0000:47:00.0 a 224571001662694 -1704020 2c59 R
10:52:06.604265 0000:47:00.0 a 224571347382767 -1704020 2c59 R
10:52:07.250895 0000:47:00.0 a 224571994063209 -1704020 2c59 R
10:52:34.411959 0000:47:00.0 a 224599155112760 -1704020 2c59 0
13:16:53.256885 0000:47:00.0 9 233258000028485 -1983799 2c59 1
13:16:53.426266 0000:47:00.0 a 233258169384338 -1983799 2c59 R
13:17:06.534702 0000:47:00.0 a 233271277862883 -1983799 2c59 0
13:17:26.059111 0000:47:00.0 9 233290802278048 -1983799 2c59 1
13:17:26.122513 0000:47:00.0 a 233290865647761 -1983799 2c59 R
"""


def test_rescheduled_restores_do_not_end_an_eviction_in_the_real_log(tmp_path: Path):
    log = tmp_path / "kfd_raw.log"
    log.write_text(REAL_KFD_LOG)
    evictions = pr.pair_events(pr.parse_kfd_events(str(log)))
    rows = [
        (
            v.start.pid,
            v.start.bus,
            None if v.end is None else round(v.end.t - v.start.t, 3),
            v.resched,
        )
        for v in evictions
    ]
    assert rows == [
        (1704018, "47", 0.009, 0),
        (1704018, "47", 0.004, 0),
        # the first restore came 0.7 s in but was rescheduled: it lasted 29.2 s
        (1704020, "47", 29.211, 4),
        (1983799, "47", 13.278, 1),
        (1983799, "47", None, 1),  # still evicted when the log ends
    ]


def test_an_eviction_while_evicted_is_counted_not_paired():
    events = [
        pr.KfdEvent(1.0, "evict", 5, "g", "1"),
        pr.KfdEvent(2.0, "evict", 5, "g", "1"),
        pr.KfdEvent(3.0, "restore", 5, "g", "0"),
    ]
    (v,) = pr.pair_events(events)
    assert (v.again, v.resched, v.end is not None) == (1, 0, True)


def _gone_tick(t: float) -> dict:
    return {**_tick(t, 0), "w": {}}


def test_worker_exits_name_the_interval_a_worker_vanished_in():
    ticks = [_tick(100, 0), _tick(101, 0), _gone_tick(102), _gone_tick(103)]
    assert pr.worker_exits(ticks) == {"700": (101, 102)}
    exits = pr.worker_exits(ticks)
    gone = pr.Eviction(pr.KfdEvent(101.4, "evict", 700, "g", "1"))
    assert pr.at_exit(gone, exits)
    # evicted 42 s before it left, and never restored: a hang, not a teardown
    assert not pr.at_exit(pr.Eviction(pr.KfdEvent(60.0, "evict", 700, "g", "1")), exits)
    restored = pr.Eviction(gone.start, pr.KfdEvent(101.6, "restore", 700, "g", "0"))
    assert not pr.at_exit(restored, exits)  # it came back: not a teardown


def test_report_calls_the_evictions_of_an_exiting_worker_a_teardown():
    ticks = [_tick(100, 0), _tick(101, 0), _tick(102, 0), _gone_tick(103)]
    ticks += [_gone_tick(104)]
    events = [
        pr.KfdEvent(102.4, "evict", 700, "20e9", "1", "0000:ce:00.0"),
        pr.KfdEvent(102.4, "evict", 700, "21e9", "1", "0000:d3:00.0"),
        # the server process, not a followed worker, goes down with it
        pr.KfdEvent(102.5, "evict", 5, "20e9", "1", "0000:ce:00.0"),
    ]
    text = pr.build_report({"kernel": "7.1"}, ticks, [], events)
    assert "3 not restored (2 of them at a worker exit)" in text
    assert (
        "### Burst 1: +2.4s to +2.5s, teardown: 1 worker(s) evicted as they exited, "
        "1 other process(es) with them, reclaim near: none"
    ) in text
    burst = text.split("### Burst 1")[1].split("## 4.")[0]
    assert "| 700 | TP2 | 5 | " in burst and "| 2 on 2 GPUs | 1 |" in burst
    assert "| 5 | server of the workers |" in burst
    assert "| 700 | ce | +2.4s | 1 |" in burst
    # the same eviction of a worker that kept running is a burst, not a teardown
    alive = [_tick(100 + i, 0) for i in range(6)]
    text = pr.build_report({"kernel": "7.1"}, alive, [], events)
    assert (
        "teardown" not in text and "### Burst 1: +2.4s to +2.5s, 2 process(es)" in text
    )


def test_counter_jumps_and_raw_evictions_merge_into_one_burst():
    jump = pr.Jump("700", "TP2", 102.0, 103.0, 900)  # lies within [101.1, 103.0]
    near = pr.Eviction(pr.KfdEvent(105.5, "evict", 700, "g", "1"))
    far = pr.Eviction(pr.KfdEvent(130.0, "evict", 700, "g", "1"))
    one, two = pr.eviction_bursts([jump], [near, far])
    assert (one.t0, one.t1) == (101.1, 105.5)
    assert one.jumps == [jump] and one.evictions == [near]
    assert two.jumps == [] and two.evictions == [far]
    assert pr.eviction_bursts([], []) == []


def test_serve_logs_name_the_workers_whose_title_has_no_rank(tmp_path: Path):
    log = tmp_path / "serve.log"
    log.write_text(
        "(Worker_TP0_EP0 pid=700) INFO 11-05 16:11:23 [x.py:1] hello\n"
        "(EngineCore pid=9) INFO nothing\n"
        "(Worker_TP1_EP1 pid=701) INFO 11-05 16:11:23 [x.py:1] hello\n"
        "(Worker pid=702) INFO no rank\n"
    )
    labels = pr.parse_worker_labels([str(log), str(tmp_path / "missing.log")])
    assert labels == {"700": "TP0", "701": "TP1"}
    ticks = [_tick(100, 0)]
    ticks[0]["w"]["700"]["title"] = "VLLM::Worker"  # before it set its title
    ticks[0]["w"]["701"] = {**ticks[0]["w"]["700"], "title": "VLLM::Worker_TP3_EP3"}
    pr.apply_labels(ticks, labels)
    assert ticks[0]["w"]["700"]["title"] == "VLLM::Worker_TP0"
    assert ticks[0]["w"]["701"]["title"] == "VLLM::Worker_TP3_EP3"  # kept


def test_worker_table_uses_the_last_record_that_could_read_the_process():
    full = _tick(100, 0)
    dying = _tick(101, 0)
    del dying["w"]["700"]["vmlck_kb"]  # /proc/<pid>/status is gone
    dying["w"]["700"]["maps"] = {"pinned": {"n": 0}, "pinned_gib_by_lock": {}}
    text = pr.build_report({"kernel": "7.1"}, [full, dying, _gone_tick(102)], [], [])
    assert "| 700 | TP2 | 5 | 0 | 0 | 0 | 1.00 | 1.00/0.50/0.00 | 3 |" in text


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc, GNU stat")
def test_evidence_launcher_starts_stops_and_reports(tmp_path: Path):
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "sudo").write_text('#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n')
    (shim / "sudo").chmod(0o755)
    raw = tmp_path / "raw.py"
    raw.write_text(
        "import sys, time\n"
        "open(sys.argv[1], 'w').write('# start 2026-11-05T12:00:00\\n')\n"
        "time.sleep(60)\n"
    )
    out = tmp_path / "ev"
    env = {
        **os.environ,
        "PATH": f"{shim}:{os.environ['PATH']}",
        "PY": sys.executable,
        "KFD_RAW": f"{sys.executable} {raw}",
        "HOURS": "0.01",
    }
    script = str(ROOT / "tools/rdna/port_v031/pinned_evidence.sh")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", script, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    res = run("start", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    status = run("status", str(out)).stdout
    assert "sampler: running" in status and "kfd_raw: running" in status
    time.sleep(2)
    res = run("stop", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "vm.swappiness=" in (out / "host_start.txt").read_text()
    assert "## zone watermarks" in (out / "host_end.txt").read_text()
    assert (out / "report.md").read_text().startswith("# Pinned host memory evictions")
    assert "sampler: not running" in run("status", str(out)).stdout
    assert run("start", str(out)).returncode == 2  # the same directory twice
    # SERVE_LOGS reaches the report as one --serve-log each
    wrapper = tmp_path / "py.sh"
    wrapper.write_text(
        f'#!/bin/sh\necho "$@" >> {tmp_path}/py.args\nexec {sys.executable} "$@"\n'
    )
    wrapper.chmod(0o755)
    env.update(PY=str(wrapper), SERVE_LOGS=f"{tmp_path}/a.log {tmp_path}/b.log")
    assert run("report", str(out)).returncode == 0
    assert (
        f"--serve-log {tmp_path}/a.log --serve-log {tmp_path}/b.log"
        in (tmp_path / "py.args").read_text()
    )
