# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-host mlock proof's helpers: neighbour replay, steady load,
quiet-window lock, and the verdict over the arms. No GPU, no server."""

import importlib.util
import json
import mmap
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "pinned_proof", ROOT / "tools/rdna/port_v031/pinned_proof.py"
)
assert _spec is not None and _spec.loader is not None
pp = importlib.util.module_from_spec(_spec)
sys.modules["pinned_proof"] = pp
_spec.loader.exec_module(pp)

GIB_KIB = 1 << 20


def ns(**kw: Any):
    import argparse

    return argparse.Namespace(**kw)


# ---- neighbour ----------------------------------------------------------------


def test_plan_sizes_the_demand_above_the_free_memory():
    # 10 GiB free, 400 available: the anonymous part brings the total to
    # free + 16 GiB = 26 GiB over 2 ranks, minus the 23.84 shm each -> 0.
    plan = pp.plan_demand(10 * GIB_KIB, 400 * GIB_KIB, 2, 23.84, None, 16.0, 60.0)
    assert plan["anon_gib"] == 0.0 and plan["affordable"]
    # 100 GiB free: 116 GiB total, 58 per rank, 34.16 of it anonymous.
    plan = pp.plan_demand(100 * GIB_KIB, 450 * GIB_KIB, 2, 23.84, None, 16.0, 60.0)
    assert plan["anon_gib"] == pytest.approx(34.16)
    assert plan["total_gib"] == pytest.approx(116.0)
    # capped at MemAvailable - margin: 200 free, 220 available, margin 60.
    plan = pp.plan_demand(200 * GIB_KIB, 220 * GIB_KIB, 2, 23.84, None, 16.0, 60.0)
    assert plan["total_gib"] == pytest.approx(160.0)
    assert plan["affordable"]


def test_plan_refuses_a_demand_that_does_not_fit():
    plan = pp.plan_demand(5 * GIB_KIB, 90 * GIB_KIB, 2, 23.84, None, 16.0, 60.0)
    assert not plan["affordable"]  # 2 x 23.84 > 90 - 60
    plan = pp.plan_demand(5 * GIB_KIB, 400 * GIB_KIB, 2, 23.84, 200.0, 16.0, 60.0)
    assert not plan["affordable"]


def test_neighbour_replay_allocates_reads_and_releases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "a.safetensors").write_bytes(os.urandom(1 << 20))
    (ckpt / "b.safetensors").write_bytes(os.urandom(1 << 20))
    fake = {"MemFree": 8 * GIB_KIB, "MemAvailable": 64 * GIB_KIB}
    monkeypatch.setattr(pp, "meminfo_kib", lambda *_: dict(fake))
    args = ns(
        workers=2,
        shm_gib=4 / 1024,
        anon_gib=str(2 / 1024),
        overshoot_gib=0.0,
        margin_gib=0.0,
        hold=0.3,
        lock=False,
        read=[str(ckpt)],
    )
    assert pp.run_neighbour(args) == 0
    line = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("NEIGHBOUR ")
    ]
    summary = json.loads(line[0].removeprefix("NEIGHBOUR "))
    assert summary["workers"] == 2 and not summary["aborted"]
    assert summary["read_gib"] == pytest.approx(2 / 1024, abs=1e-3)  # 2 MiB
    assert summary["t1"] - summary["t_alloc"] >= 0.3


def test_neighbour_refuses_when_the_host_is_too_small(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        pp, "meminfo_kib", lambda *_: {"MemFree": GIB_KIB, "MemAvailable": 10 * GIB_KIB}
    )
    args = ns(
        workers=2, shm_gib=23.84, anon_gib="auto", overshoot_gib=16.0,
        margin_gib=60.0, hold=0.0, lock=False, read=None,
    )  # fmt: skip
    assert pp.run_neighbour(args) == 3
    assert "ABORT" in capsys.readouterr().out


def test_neighbour_stops_when_memavailable_falls_below_the_floor(
    monkeypatch: pytest.MonkeyPatch,
):
    stop = threading.Event()
    floor = pp.Floor(floor_kib=100, stop=stop)
    monkeypatch.setattr(pp, "meminfo_kib", lambda *_: {"MemAvailable": 99})
    mm = pp.touch_shm(4 * mmap.PAGESIZE, False, floor)
    assert stop.is_set()
    assert isinstance(mm, mmap.mmap)
    assert pp.touch_anon(mmap.PAGESIZE, floor) == []  # nothing once stopped


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mlock")
def test_mlock_buffer_locks_a_small_mapping():
    mm = mmap.mmap(-1, mmap.PAGESIZE)
    assert pp.mlock_buffer(mm, mmap.PAGESIZE) == 0


# ---- steady -------------------------------------------------------------------


def test_stream_records_log_slow_gaps_and_a_summary():
    t_req = 100.0
    stamps = [100.5, 100.55, 100.60, 101.10, 101.15]
    recs = pp.stream_records(stamps, t_req, 5000.0, 2, gap_ms=250.0)
    gaps = [r for r in recs if "gap_ms" in r]
    assert len(gaps) == 1 and gaps[0]["gap_ms"] == pytest.approx(500.0, abs=1)
    assert gaps[0]["t"] == pytest.approx(5001.1, abs=1e-3)
    summary = recs[-1]
    assert summary["ttft_ms"] == pytest.approx(500.0, abs=1)
    assert summary["chunks"] == 5 and summary["gap_max_ms"] == pytest.approx(
        500.0, abs=1
    )


class StallingServer(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(6):
            time.sleep(0.45 if i == 3 else 0.02)
            self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def test_steady_load_writes_gaps_and_requests(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), StallingServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "steady.jsonl"
    args = ns(
        url=f"http://127.0.0.1:{server.server_address[1]}", model="m", out=str(out),
        duration=2.0, streams=2, max_tokens=8, gap_ms=250.0,
    )  # fmt: skip
    try:
        assert pp.run_steady(args) == 0
    finally:
        server.shutdown()
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert any("gap_ms" in r and r["gap_ms"] >= 400 for r in recs)
    summaries = [r for r in recs if "ttft_ms" in r]
    assert len(summaries) >= 2 and all(r["chunks"] == 6 for r in summaries)
    assert {r["stream"] for r in summaries} == {0, 1}


# ---- quiet windows and the boot lock -------------------------------------------


def _touch(path: Path, age_s: float) -> None:
    path.write_text("")
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_recent_quiet_ignores_old_and_own_files(tmp_path: Path):
    _touch(tmp_path / "TwoLaneFeat.proof", 30)
    _touch(tmp_path / "Old.proof", 3600)
    _touch(tmp_path / "EvictFix.proof", 1)
    now = time.time()
    assert pp.recent_quiet(str(tmp_path), 360, "EvictFix", now) == ["TwoLaneFeat.proof"]
    assert pp.recent_quiet(str(tmp_path), 360, "", now) == [
        "EvictFix.proof",
        "TwoLaneFeat.proof",
    ]
    assert pp.recent_quiet(str(tmp_path / "missing"), 360, "", now) == []


def test_boot_lock_waits_for_quiet_and_for_another_boot(tmp_path: Path):
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    lock = tmp_path / "boot.lock"
    _touch(quiet / "TwoLaneFeat.proof", 10)
    holder = os.open(lock, os.O_WRONLY | os.O_CREAT)
    pp.fcntl.flock(holder, pp.fcntl.LOCK_EX)
    sleeps: list[float] = []
    messages: list[str] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) == 1:  # the measuring server finishes
            (quiet / "TwoLaneFeat.proof").unlink()
        elif len(sleeps) == 2:  # the other boot ends
            pp.fcntl.flock(holder, pp.fcntl.LOCK_UN)

    fd = pp.take_boot_lock(
        str(lock), str(quiet), "EvictFix", messages.append, sleep, time.time, poll_s=5
    )
    try:
        assert sleeps == [5, 5]
        assert "quiet window (TwoLaneFeat.proof)" in messages[0]
        assert "is held" in messages[1]
        other = os.open(lock, os.O_WRONLY)
        with pytest.raises(BlockingIOError):
            pp.fcntl.flock(other, pp.fcntl.LOCK_EX | pp.fcntl.LOCK_NB)
        os.close(other)
    finally:
        os.close(fd)
        os.close(holder)


def test_boot_lock_gives_way_when_a_quiet_file_appears_while_locking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    lock = tmp_path / "boot.lock"
    calls = {"n": 0}
    real = pp.recent_quiet

    def racing(qd: str, age: float, own: str, now: float) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 2:  # the re-check after taking the lock
            _touch(quiet / "Late.proof", 0)
        return real(qd, age, own, now)

    monkeypatch.setattr(pp, "recent_quiet", racing)
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        (quiet / "Late.proof").unlink()

    fd = pp.take_boot_lock(
        str(lock), str(quiet), "EvictFix", lambda _m: None, sleep, time.time, poll_s=1
    )
    os.close(fd)
    assert sleeps == [1] and calls["n"] == 5


# ---- the verdict ----------------------------------------------------------------


def test_parse_lock_line_reads_what_the_worker_logs():
    from vllm.utils import pinned_host

    class Capture:
        lines: list[str] = []

        def info(self, msg: str, *a: Any) -> None:
            self.lines.append(msg % a)

        def warning(self, *_: Any) -> None:
            pass

    cap = Capture()
    original = pinned_host.logger
    pinned_host.logger = cap  # type: ignore[assignment]
    try:
        rep = pinned_host.PinnedLockReport(
            candidates=330,
            locked=329,
            locked_bytes=int(1.06 * (1 << 30)),
            vm_locked_before=int(23.84 * (1 << 30)),
            vm_locked_after=int(24.9 * (1 << 30)),
        )
        pinned_host.log_pinned_lock_report(rep)
    finally:
        pinned_host.logger = original
    line = (
        "(Worker_TP1 pid=77) INFO 11-05 13:00:00 [pinned_host.py:200] " + cap.lines[0]
    )
    parsed = pp.parse_lock_line(line)
    assert parsed == {
        "locked": 329,
        "candidates": 330,
        "gib": 1.06,
        "vmlck_before": 23.84,
        "vmlck_after": 24.9,
    }
    assert pp.parse_lock_line("INFO nothing here") is None
    assert pp.parse_lock_line("Locked many of few") is None


def arm(
    name: str,
    loads_ms: dict[str, int],
    vmlck: float = 23.84,
    lock: tuple[int, int, float] | None = None,
    ttft: float = 300.0,
    itl: float = 22.0,
    idle_ms: int = 0,
) -> dict[str, Any]:
    lock_lines = []
    if lock:
        lock_lines = [
            {
                "locked": lock[0],
                "candidates": lock[1],
                "gib": lock[2],
                "vmlck_before": 23.84,
                "vmlck_after": 23.84 + lock[2],
            }
        ]
    workers = ("Worker_TP0", "Worker_TP1")
    windows = [
        {
            "name": "idle",
            "evicted_ms": dict.fromkeys(workers, idle_ms),
            "vmlck": dict.fromkeys(workers, vmlck),
        },
        {
            "name": "load1",
            "evicted_ms": loads_ms,
            "vmlck": dict.fromkeys(workers, vmlck),
        },
    ]
    cell = {m: 1.0 for m in pp.METRICS}
    cell.update(median_ttft_ms=ttft, median_itl_ms=itl)
    return {
        "name": name,
        "kind": pp.arm_kind(name),
        "proof": {"windows": windows, "vmlck_end": dict.fromkeys(workers, vmlck)},
        "lock_lines": lock_lines,
        "cells": {"bench-1024x512-c1": cell},
        "invalid_cells": [],
    }


def fix_arm(name: str, **kw: Any) -> dict[str, Any]:
    kw.setdefault("lock", (329, 329, 1.0))
    kw.setdefault("vmlck", 24.84)
    return arm(name, {"Worker_TP0": 0, "Worker_TP1": 12}, **kw)


def base_arm(name: str, **kw: Any) -> dict[str, Any]:
    return arm(name, {"Worker_TP0": 7100, "Worker_TP1": 1300}, **kw)


def test_arm_kind():
    assert [pp.arm_kind(n) for n in ("base", "base2", "fix", "fix2")] == [
        "base", "base", "fix", "fix",
    ]  # fmt: skip


def test_verdict_proven_when_the_baseline_evicts_and_the_fix_does_not():
    res = pp.compare(
        [base_arm("base"), fix_arm("fix"), fix_arm("fix2"), base_arm("base2")]
    )
    assert res["verdict"] == "PROVEN", res["reasons"]
    assert res["reproduced"] == ["base", "base2"]
    (row,) = [r for r in res["vmlck"] if r["worker"] == "Worker_TP0"]
    assert row["delta"] == pytest.approx(1.0) and row["logged_mean"] == pytest.approx(
        1.0
    )
    assert all(c["ok"] for c in res["cells"])


def test_verdict_inconclusive_when_the_baseline_did_not_evict():
    quiet_base = arm("base", {"Worker_TP0": 40, "Worker_TP1": 0})
    res = pp.compare([quiet_base, fix_arm("fix", vmlck=24.84)])
    assert res["verdict"] == "INCONCLUSIVE"
    assert "nothing to remove" in res["reasons"][0]


def test_verdict_failed_when_a_fix_arm_is_evicted():
    bad = arm(
        "fix", {"Worker_TP0": 0, "Worker_TP1": 1800}, lock=(329, 329, 1.0), vmlck=24.84
    )
    res = pp.compare([base_arm("base"), bad])
    assert res["verdict"] == "FAILED"
    assert "evicted 1800 ms" in res["reasons"][0]


def test_verdict_failed_when_a_lock_did_not_hold_or_did_not_run():
    short = fix_arm("fix", lock=(300, 329, 0.9))
    res = pp.compare([base_arm("base"), short])
    assert res["verdict"] == "FAILED" and "locked 300 of 329" in res["reasons"][0]
    silent = fix_arm("fix", lock=None)
    res = pp.compare([base_arm("base"), silent])
    assert res["verdict"] == "FAILED" and "no 'Locked N of M" in res["reasons"][0]
    leaky = base_arm("base", lock=(5, 5, 0.1))
    res = pp.compare([leaky, fix_arm("fix")])
    assert res["verdict"] == "FAILED" and "env not applied" in res["reasons"][0]


def test_verdict_failed_on_side_effects():
    slow = fix_arm("fix", ttft=330.0)  # +10 % TTFT, no base spread
    res = pp.compare([base_arm("base"), slow])
    assert res["verdict"] == "FAILED" and "side effects" in res["reasons"][0]
    # a base spread of 12 % is the noise floor: the same +10 % is inside it
    res = pp.compare([base_arm("base"), base_arm("base2", ttft=336.0), slow])
    assert res["verdict"] == "PROVEN", res["reasons"]
    # VmLck rising by more than the fix logged
    odd = fix_arm("fix", vmlck=27.0)
    res = pp.compare([base_arm("base"), odd])
    assert res["verdict"] == "FAILED" and "side effects" in res["reasons"][0]


def test_verdict_failed_when_vmlck_moves_inside_an_arm_or_a_hook_is_missing():
    moving = fix_arm("fix")
    moving["proof"]["windows"][1]["vmlck"] = {"Worker_TP0": 30.0, "Worker_TP1": 24.84}
    res = pp.compare([base_arm("base"), moving])
    assert res["verdict"] == "FAILED" and "VmLck" in res["reasons"][0]
    missing = fix_arm("fix")
    missing["proof"] = None
    res = pp.compare([base_arm("base"), missing])
    assert res["verdict"] == "FAILED" and "did not finish" in res["reasons"][0]


def write_arm(run: Path, name: str, a: dict[str, Any], gib: float | None) -> None:
    d = run / "arms" / name
    (d / "proof").mkdir(parents=True)
    (d / "proof" / "windows.json").write_text(json.dumps(a["proof"]))
    lines = ["INFO unrelated line"]
    if gib is not None:
        lines.append(
            "(Worker_TP0 pid=9) INFO 11-05 13:00:00 [pinned_host.py:200] "
            f"Locked 329 of 329 pinned host mappings ({gib:.2f} GiB); "
            f"VmLck 23.84 -> {23.84 + gib:.2f} GiB."
        )
    (d / "serve.log").write_text("\n".join(lines) + "\n")
    cell = a["cells"]["bench-1024x512-c1"]
    (d / "bench-1024x512-c1.json").write_text(json.dumps(cell))
    (d / "summary.txt").write_text(
        "[13:00:00] bench 1024x512-c1 (n=4)\n    evicted_ms Worker_TP0=+0 -> ok\n"
    )


def test_verdict_cli_reads_the_arm_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    for name, a, gib in (
        ("base", base_arm("base"), None),
        ("fix", fix_arm("fix"), 1.0),
        ("fix2", fix_arm("fix2"), 1.0),
        ("base2", base_arm("base2"), None),
    ):
        write_arm(tmp_path, name, a, gib)
    rc = pp.main(["verdict", "--run", str(tmp_path), "--arms", "base fix fix2 base2"])
    out = capsys.readouterr().out
    assert rc == 0 and "PROVEN" in out.splitlines()[0]
    assert (tmp_path / "results" / "verdict.md").read_text() == out
    assert (
        json.loads((tmp_path / "results" / "verdict.json").read_text())["verdict"]
        == "PROVEN"
    )
    # an INVALID bench cell is left out of the TTFT/ITL comparison
    (tmp_path / "arms" / "fix" / "summary.txt").write_text(
        "[13:00:00] bench 1024x512-c1 (n=4)\n"
        "    evicted_ms Worker_TP0=+900 -> INVALID (KFD evicted ...)\n"
    )
    assert pp.load_arm(str(tmp_path), "fix")["invalid_cells"] == ["bench-1024x512-c1"]


def test_hook_needs_a_neighbour_command(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        pp.main(["hook", "--boot-lock", "x", "--quiet-dir", "y"])
    assert "neighbour load after --" in capsys.readouterr().err


# ---- one quiet window -------------------------------------------------------------


def make_window(tmp_path: Path, **kw: Any) -> tuple[Path, Any]:
    kw.setdefault("say", lambda _m: None)
    kw.setdefault("building", lambda: False)
    quiet = tmp_path / "quiet"
    win = pp.QuietWindow(str(quiet), "EvictFix", str(tmp_path / "boot.lock"), **kw)
    return quiet, win


def test_quiet_window_touches_refreshes_and_removes_its_file(tmp_path: Path):
    quiet, win = make_window(tmp_path, touch_s=0.05)
    path = quiet / "EvictFix"
    win.open()
    try:
        assert path.exists() and win.opened_at is not None and not win.expired
        first = path.stat().st_mtime
        time.sleep(0.4)
        assert path.stat().st_mtime > first  # kept fresh
    finally:
        win.close()
    assert not path.exists() and win.closed_at is not None
    time.sleep(0.2)
    assert not path.exists()  # no refresher left to recreate it


def test_quiet_window_stops_refreshing_after_its_cap(tmp_path: Path):
    quiet, win = make_window(tmp_path, touch_s=0.05, cap_s=0.3)
    path = quiet / "EvictFix"
    win.open()
    try:
        time.sleep(0.8)
        mtime = path.stat().st_mtime
        time.sleep(0.3)
        assert path.stat().st_mtime == mtime  # a stuck measurer goes stale
        assert win.expired
    finally:
        win.close()


def test_quiet_window_waits_for_the_gap_the_build_and_a_boot(tmp_path: Path):
    clock = {"t": 1030.0}
    events: list[str] = []
    holder = os.open(tmp_path / "boot.lock", os.O_WRONLY | os.O_CREAT)
    pp.fcntl.flock(holder, pp.fcntl.LOCK_EX)
    builds = iter([True, True, False])

    def sleep(s: float) -> None:
        events.append(f"sleep {s:g}")
        clock["t"] += s
        if s == 5:  # the boot that was in flight ends
            pp.fcntl.flock(holder, pp.fcntl.LOCK_UN)

    quiet, win = make_window(
        tmp_path, now=lambda: clock["t"], sleep=sleep, building=lambda: next(builds)
    )
    win.closed_at = 1000.0  # the previous window ended 30 s ago
    win.open()
    try:
        # 120 s gap - 30 s, two 60 s polls of the build, one 5 s poll of the lock
        assert events == ["sleep 90", "sleep 60", "sleep 60", "sleep 5"]
        assert win.waited_lock_s == 5 and (quiet / "EvictFix").exists()
    finally:
        win.close()
        os.close(holder)
    assert not (quiet / "EvictFix").exists()


def test_quiet_window_measures_anyway_when_a_boot_never_ends(tmp_path: Path):
    clock = {"t": 0.0}
    messages: list[str] = []
    holder = os.open(tmp_path / "boot.lock", os.O_WRONLY | os.O_CREAT)
    pp.fcntl.flock(holder, pp.fcntl.LOCK_EX)

    def sleep(s: float) -> None:
        clock["t"] += s

    _, win = make_window(
        tmp_path, now=lambda: clock["t"], sleep=sleep, say=messages.append, cap_s=100
    )
    win.open()
    try:
        assert win.waited_lock_s > 70  # 70 % of the cap
        assert any("measuring anyway" in m for m in messages)
    finally:
        win.close()
        os.close(holder)


def test_build_running_is_false_without_pgrep_or_a_match(
    monkeypatch: pytest.MonkeyPatch,
):
    assert pp.build_running("no such process pattern " + "x" * 12) is False

    def missing(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("pgrep")

    monkeypatch.setattr(pp.subprocess, "run", missing)
    assert pp.build_running() is False


@pytest.mark.skipif(shutil.which("pgrep") is None, reason="pgrep")
def test_build_running_matches_a_build_and_not_a_waiter_that_names_it():
    # Another agent's `bash -c 'while pgrep -f "docker buildx build"; ...'` waiter
    # carries the words in its own command line; it must not count as a build.
    # The words get a suffix so that a real build on the host cannot interfere.
    words = f"docker buildx build-{os.getpid()}"
    pattern = pp.BUILD_PATTERN.replace("docker buildx build", words)
    assert pattern.startswith("^") and pattern != words
    waiter = subprocess.Popen(["bash", "-c", f'sleep 30; true # pgrep -f "{words}"'])
    build = None
    try:
        time.sleep(0.3)
        assert pp.build_running(pattern) is False
        build = subprocess.Popen(["bash", "-c", f'exec -a "{words}" sleep 30'])
        time.sleep(0.3)
        assert pp.build_running(pattern) is True
    finally:
        for proc in (waiter, build):
            if proc is not None:
                proc.kill()
                proc.wait()


# ---- evictions outside the windows ------------------------------------------------


def test_outside_ms_counts_only_the_jumps_that_overlap_no_window():
    kfd = pp.sibling("kfd_evictions")
    samples: list[tuple[float, dict[str, int]]] = [
        (0.0, {"W0": 0}),
        (1.0, {"W0": 0}),
        (2.0, {"W0": 400}),
        (3.0, {"W0": 400}),
        (4.0, {"W0": 900}),
    ]
    # 400 ms lies in [1.6, 2.0], 500 ms in [3.5, 4.0]
    assert pp.outside_ms(kfd, samples, [{"t0": 1.5, "t1": 2.2}]) == {"W0": 500}
    assert pp.outside_ms(kfd, samples, [{"t0": 1.5, "t1": 4.5}]) == {}
    assert pp.outside_ms(kfd, samples, []) == {"W0": 900}


def test_verdict_lists_the_gaps_and_the_windows_that_outlasted_their_cap():
    a = base_arm("base")
    a["proof"]["evicted_outside_ms"] = {"Worker_TP0": 800}
    a["proof"]["windows"][1]["quiet"] = {
        "opened": 1.0,
        "waited_lock_s": 0.0,
        "expired": True,
    }
    res = pp.compare([a, fix_arm("fix")])
    row = res["rows"][0]
    assert row["outside_ms"] == {"Worker_TP0": 800} and row["expired"] == ["load1"]
    assert "outlasted its 15 minute cap" in res["caveats"][0]
    text = pp.verdict_text(res)
    assert "outside the windows" in text and "Caveats:" in text
    assert pp.compare([base_arm("base"), fix_arm("fix")])["caveats"] == []


# ---- the lease README's check of a slot -------------------------------------------


class FakeHip:
    """The two HIP calls of hip_bus_ids; device 0 is on bus 0a, device 1 on CE."""

    buses = ("0000:0a:00.0", "0000:CE:00.0")
    fail = False

    def hipGetDeviceCount(self, ref: Any) -> int:  # noqa: N802
        ref._obj.value = len(self.buses)
        return 1 if self.fail else 0

    def hipDeviceGetPCIBusId(self, buf: Any, _n: int, dev: int) -> int:  # noqa: N802
        buf.value = self.buses[dev].encode()
        return 0


def test_hip_bus_ids_maps_every_device_by_its_pci_bus():
    assert pp.hip_bus_ids(FakeHip()) == {"0a": 0, "ce": 1}
    broken = FakeHip()
    broken.fail = True
    with pytest.raises(RuntimeError, match="hipGetDeviceCount"):
        pp.hip_bus_ids(broken)


def test_buses_cli_prints_the_indices_in_the_order_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    assert pp.main(["buses", "--venv", str(tmp_path), "ce"]) == 1
    assert "no libamdhip64.so" in capsys.readouterr().err
    lib = tmp_path / "lib/python3.12/site-packages/_rocm_sdk_core/lib/core/lib"
    lib.mkdir(parents=True)
    (lib / "libamdhip64.so.7").write_text("")
    monkeypatch.setattr(pp.ctypes, "CDLL", lambda _path: FakeHip())
    assert pp.main(["buses", "--venv", str(tmp_path), "CE", "0a"]) == 0
    assert capsys.readouterr().out.strip() == "1,0"
    assert pp.main(["buses", "--venv", str(tmp_path), "ce", "ff"]) == 1
    assert "no HIP device on bus ['ff']" in capsys.readouterr().err


# ---- the hook: one window per load ------------------------------------------------


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc and the flock binary"
)
def test_hook_measures_each_load_in_a_quiet_window_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    kfd = pp.sibling("kfd_evictions")
    flag = tmp_path / "flag"
    state = {"ms": 0}

    def snapshot(_pid: int) -> dict[str, int]:
        if flag.exists():  # the second load evicts the worker for 300 ms
            flag.unlink()
            state["ms"] += 300
        return {"Worker_TP0": state["ms"]}

    real_sibling = pp.sibling
    monkeypatch.setattr(kfd, "snapshot", snapshot)
    monkeypatch.setattr(kfd, "worker_pids", lambda _spid: {"Worker_TP0": os.getpid()})
    monkeypatch.setattr(
        pp, "sibling", lambda n: kfd if n == "kfd_evictions" else real_sibling(n)
    )
    monkeypatch.setattr(pp, "spawn", lambda _cmd, _out: None)
    monkeypatch.setattr(pp, "build_running", lambda *_a, **_k: False)
    out, quiet, lock, obs = (
        tmp_path / "out",
        tmp_path / "quiet",
        tmp_path / "boot.lock",
        tmp_path / "obs.log",
    )
    out.mkdir()
    quiet.mkdir()
    (out / "serve.pid").write_text(f"{os.getpid()}\n")
    for k, v in (("OUT", out), ("PORT", 1), ("SERVED", "m")):
        monkeypatch.setenv(k, str(v))
    # what one load sees: our own window file, and the boot lock held by the hook
    load = [
        "bash",
        "-c",
        (
            f'echo "quiet=$([ -e {quiet}/EvictFix ] && echo yes || echo no)'
            f' lock=$(flock -n {lock} true && echo free || echo held)" >> {obs};'
            f'[ "$(wc -l < {obs})" = 2 ] && touch {flag}; sleep 0.4'
        ),
    ]
    rc = pp.main(
        [
            *("hook", "--loads", "3", "--idle", "0.2", "--gap", "0.1"),
            *("--settle", "0.4", "--streams", "1", "--max-minutes", "1"),
            *("--boot-lock", str(lock), "--quiet-dir", str(quiet)),
            *("--poll", "0.1", "--quiet-gap", "1.2", "--quiet-touch", "0.1"),
            "--",
            *load,
        ]
    )
    assert rc == 0
    assert obs.read_text().split("\n")[:-1] == ["quiet=yes lock=held"] * 3
    assert list(quiet.iterdir()) == []  # every window closed
    data = json.loads((out / "proof/windows.json").read_text())
    wins = {w["name"]: w for w in data["windows"]}
    assert list(wins) == ["idle1", "load1", "idle2", "load2", "idle3", "load3"]
    assert wins["load2"]["evicted_ms"] == {"Worker_TP0": 300}
    assert wins["load1"]["evicted_ms"] == {} and wins["load3"]["evicted_ms"] == {}
    assert data["evicted_outside_ms"] == {}
    opened = [wins[f"load{i}"]["quiet"]["opened"] for i in (1, 2, 3)]
    ended = [wins[f"load{i}"]["t1"] for i in (1, 2, 3)]
    assert opened[1] - ended[0] >= 1.2 and opened[2] - ended[1] >= 1.2
    assert not any(wins[f"load{i}"]["quiet"]["expired"] for i in (1, 2, 3))
