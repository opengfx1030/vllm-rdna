# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-memory eviction sampler and its report, on fake /proc trees."""

import argparse
import importlib.util
import io
import json
import struct
import sys
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
    pairs = pr.pair_events(events)
    assert [(e.pid, e.trigger, r and r.trigger) for e, r in pairs] == [
        (700, "1", "R"),
        (701, "1", None),  # never restored
        (700, "2", None),
    ]
    assert pairs[0][1].t - pairs[0][0].t == pytest.approx(2.0)


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
    assert "| 700 | 20e9 | +3.6s | 1 | 900 | yes |" in burst_part
    assert (
        "1 QUEUE_EVICTION by trigger (0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND) {'1': 1}"
        in text
    )
    totals = text.split("## 4.")[1]
    assert "| pinned | unlocked | 12 | 3 |" in totals
    assert "| pinned | locked | 99 | 0 |" in totals
    assert pr.build_report({}, [], [], []) == "no samples\n"


def test_report_cli_reads_the_sampler_output(tmp_path: Path):
    rows = [{"meta": {"start": 100.0, "kernel": "7.1", "sysctl": {}}}]
    rows += [_tick(100 + i, 0 if i < 5 else 900) for i in range(8)]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "report.md"
    assert pr.main([str(tmp_path), "--md", str(out)]) == 0  # no pagemap.jsonl
    text = out.read_text()
    assert "### Burst 1" in text
    assert "No page of a followed mapping moved" in text
