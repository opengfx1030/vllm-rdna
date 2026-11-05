# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-host mlock proof's helpers: neighbour replay, steady load,
quiet-window lock, and the verdict over the arms. No GPU, no server."""

import importlib.util
import json
import mmap
import os
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
