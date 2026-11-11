# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hx host rendezvous (``rdv.py``): atomic json files, waits,
pid liveness and the pidfd peer watcher (real subprocesses)."""

import os
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("torch")

from vllm.distributed.hetero_moe import rdv  # noqa: E402

LINUX_PROC = sys.platform == "linux" and os.path.isdir("/proc/self")
needs_proc = pytest.mark.skipif(not LINUX_PROC, reason="needs /proc")
needs_pidfd = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"), reason="os.pidfd_open is missing"
)

SLEEPER = "import time; time.sleep(600)"


def spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", SLEEPER])


def spawn_exited() -> subprocess.Popen:
    """A child that has exited but is not reaped (a zombie). ``waitid`` with
    ``WNOWAIT`` waits for the exit without collecting the status."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)  # type: ignore[attr-defined]
    return proc


def reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(30)


@pytest.fixture
def procs():
    spawned: list[subprocess.Popen] = []
    yield spawned
    for p in spawned:
        reap(p)


# ---- hx_dir ------------------------------------------------------------------------


def test_hx_dir_creates_a_private_directory_once(tmp_path):
    root = tmp_path / "shm" / "deeper"
    d = rdv.hx_dir("00000000000000a1", root=str(root))
    assert d == root / "hx_00000000000000a1"
    assert d.is_dir()
    assert (d.stat().st_mode & 0o777) == 0o700
    (d / "marker").write_text("x")
    assert rdv.hx_dir("00000000000000a1", root=str(root)) == d  # second caller
    assert (d / "marker").read_text() == "x"


# ---- write_json / read_json --------------------------------------------------------


def test_json_roundtrip_leaves_no_temp_file(tmp_path):
    path = tmp_path / "doc.json"
    doc = {"b": [1, 2, 3], "a": {"x": "y"}, "pid": os.getpid()}
    rdv.write_json(path, doc)
    assert rdv.read_json(path) == doc
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.json"]
    assert path.read_text().index('"a"') < path.read_text().index('"b"')  # sorted


def test_read_json_of_missing_empty_or_partial_files_is_none(tmp_path):
    assert rdv.read_json(tmp_path / "missing.json") is None
    for name, text in [("empty", ""), ("cut", '{"a": [1, 2'), ("junk", "not json")]:
        path = tmp_path / f"{name}.json"
        path.write_text(text)
        assert rdv.read_json(path) is None, name


def test_write_json_replaces_the_old_document_whole(tmp_path):
    path = tmp_path / "doc.json"
    rdv.write_json(path, {"v": 1, "big": "z" * 100_000})
    rdv.write_json(path, {"v": 2})
    assert rdv.read_json(path) == {"v": 2}


def test_failed_write_keeps_the_old_document(tmp_path):
    path = tmp_path / "doc.json"
    rdv.write_json(path, {"v": 1})
    with pytest.raises(TypeError):
        rdv.write_json(path, {"v": object()})  # not serialisable, fails mid-write
    assert rdv.read_json(path) == {"v": 1}


def test_write_json_is_atomic_for_a_hammering_reader(tmp_path):
    """After the first publish a reader never sees a missing or partial file,
    whatever the writer is doing (sizes change on every rewrite)."""
    path = tmp_path / "doc.json"
    rdv.write_json(path, {"n": 0, "pad": ""})
    stop = threading.Event()
    seen: list[int] = []
    bad: list[str] = []

    def reader():
        while not stop.is_set():
            obj = rdv.read_json(path)
            if obj is None:
                bad.append("None")
            elif obj["pad"] != "p" * (obj["n"] * 37 % 4001):
                bad.append(f"torn n={obj['n']}")
            else:
                seen.append(obj["n"])

    t = threading.Thread(target=reader)
    t.start()
    try:
        for n in range(1, 301):
            rdv.write_json(path, {"n": n, "pad": "p" * (n * 37 % 4001)})
    finally:
        stop.set()
        t.join(30)
    assert not t.is_alive()
    assert not bad, bad[:5]
    assert len(seen) > 10
    assert seen == sorted(seen)  # versions only move forward
    assert rdv.read_json(path)["n"] == 300


# ---- wait_json ---------------------------------------------------------------------


def test_wait_json_returns_an_existing_file_without_asking_alive(tmp_path):
    path = tmp_path / "doc.json"
    rdv.write_json(path, {"v": 1})

    def alive():
        raise AssertionError("alive must not be consulted when the file exists")

    assert rdv.wait_json(path, 5.0, what="thing", alive=alive) == {"v": 1}


def test_wait_json_sees_a_file_that_appears_later(tmp_path):
    path = tmp_path / "doc.json"
    t = threading.Timer(0.15, rdv.write_json, args=(path, {"v": 7}))
    t.start()
    try:
        assert rdv.wait_json(path, 30.0, what="late", poll_s=0.01) == {"v": 7}
    finally:
        t.join(30)


def test_wait_json_times_out_with_what_and_path(tmp_path):
    path = tmp_path / "never.json"
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        rdv.wait_json(path, 0.3, what="server exp7 (plan x)", poll_s=0.01)
    assert time.monotonic() - t0 >= 0.29
    text = str(exc.value)
    assert "server exp7 (plan x)" in text and str(path) in text
    assert "did not appear" in text and "0.3 s" in text


def test_wait_json_alive_callback_failure_ends_the_wait_at_once(tmp_path):
    path = tmp_path / "never.json"
    calls = []

    def alive():
        calls.append(1)
        return None if len(calls) < 3 else "peer exp0 (pid 123) died"

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"hub: peer exp0 \(pid 123\) died"):
        rdv.wait_json(path, 60.0, what="hub", alive=alive, poll_s=0.01)
    assert len(calls) == 3
    assert time.monotonic() - t0 < 30.0  # nowhere near the 60 s timeout


def test_wait_json_alive_returning_none_keeps_waiting_until_the_file_lands(tmp_path):
    path = tmp_path / "doc.json"
    t = threading.Timer(0.15, rdv.write_json, args=(path, {"ok": True}))
    t.start()
    try:
        got = rdv.wait_json(path, 30.0, what="x", alive=lambda: None, poll_s=0.01)
    finally:
        t.join(30)
    assert got == {"ok": True}


def test_remove_quiet(tmp_path):
    path = tmp_path / "f"
    path.write_text("x")
    rdv.remove_quiet(path)
    assert not path.exists()
    rdv.remove_quiet(path)  # already gone: no error
    rdv.remove_quiet(tmp_path / "never-existed")


# ---- pid_alive ---------------------------------------------------------------------


def test_pid_alive_for_this_process_and_invalid_pids():
    assert rdv.pid_alive(os.getpid())
    assert not rdv.pid_alive(0)
    assert not rdv.pid_alive(-1)
    assert not rdv.pid_alive(-os.getpid())


def test_pid_alive_for_a_running_and_a_reaped_child(procs):
    proc = spawn_sleeper()
    procs.append(proc)
    assert rdv.pid_alive(proc.pid)
    proc.kill()
    proc.wait(30)
    assert not rdv.pid_alive(proc.pid)


@needs_proc
def test_pid_alive_is_false_for_a_zombie_child():
    proc = spawn_exited()  # exited, not reaped: kill(pid, 0) still succeeds
    try:
        os.kill(proc.pid, 0)
        assert not rdv.pid_alive(proc.pid)
    finally:
        reap(proc)
    assert not rdv.pid_alive(proc.pid)


@needs_proc
def test_pid_alive_is_true_for_a_process_of_another_user():
    # pid 1 exists on every Linux box; it is not ours unless we are root
    assert rdv.pid_alive(1)


# ---- PeerWatcher -------------------------------------------------------------------


class Deaths:
    def __init__(self):
        self.items: list[tuple[int, str]] = []
        self.threads: list[str] = []
        self.event = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, pid: int, name: str) -> None:
        with self._lock:
            self.items.append((pid, name))
            self.threads.append(threading.current_thread().name)
        self.event.set()

    def wait(self, count: int, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.items) >= count:
                    return True
            time.sleep(0.01)
        return False


@needs_pidfd
def test_peer_watcher_reports_a_death_exactly_once(procs):
    deaths = Deaths()
    watcher = rdv.PeerWatcher(deaths)
    watcher.start()
    proc = spawn_sleeper()
    procs.append(proc)
    try:
        watcher.add(proc.pid, "exp0")
        assert not deaths.event.wait(0.3)  # alive: nothing reported
        proc.kill()
        assert deaths.wait(1)
        time.sleep(0.3)  # a second report would land by now
        assert deaths.items == [(proc.pid, "exp0")]
        assert watcher.dead == [(proc.pid, "exp0")]
        assert deaths.threads == ["hx-peer-watcher"]  # reported on the watcher thread
    finally:
        watcher.stop()
        watcher.join(30)
    assert not watcher.is_alive()


@needs_pidfd
def test_peer_watcher_reports_a_peer_that_is_already_gone_at_once():
    deaths = Deaths()
    watcher = rdv.PeerWatcher(deaths)  # not even started: add() reports directly
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(30)  # reaped: pidfd_open finds no such process
    watcher.add(proc.pid, "gone")
    assert deaths.items == [(proc.pid, "gone")]  # before add() returned
    assert watcher.dead == [(proc.pid, "gone")]


@needs_pidfd
def test_peer_watcher_reports_a_zombie_added_late(procs):
    deaths = Deaths()
    watcher = rdv.PeerWatcher(deaths)
    watcher.start()
    proc = spawn_exited()
    procs.append(proc)
    try:
        watcher.add(proc.pid, "zombie")  # pidfd is readable: the thread reports
        assert deaths.wait(1)
        assert deaths.items == [(proc.pid, "zombie")]
    finally:
        watcher.stop()
        watcher.join(30)
    assert not watcher.is_alive()


@needs_pidfd
def test_peer_watcher_tells_the_dead_peer_from_the_living(procs):
    deaths = Deaths()
    watcher = rdv.PeerWatcher(deaths)
    watcher.start()
    peers = {name: spawn_sleeper() for name in ("a", "b", "c")}
    procs.extend(peers.values())
    try:
        for name, proc in peers.items():
            watcher.add(proc.pid, name)
        peers["b"].kill()
        assert deaths.wait(1)
        peers["a"].kill()
        assert deaths.wait(2)
        time.sleep(0.2)
        assert sorted(deaths.items) == sorted(
            [(peers["a"].pid, "a"), (peers["b"].pid, "b")]
        )
        assert deaths.items[0] == (peers["b"].pid, "b")  # first dead, first reported
        peers["c"].kill()
        assert deaths.wait(3)
        assert len(deaths.items) == 3
    finally:
        watcher.stop()
        watcher.join(30)


@needs_pidfd
def test_peer_watcher_stop_ends_the_thread_without_any_peer():
    watcher = rdv.PeerWatcher(lambda pid, name: None)
    watcher.start()
    assert watcher.is_alive() and watcher.daemon
    t0 = time.monotonic()
    watcher.stop()
    watcher.join(30)
    assert not watcher.is_alive()
    assert time.monotonic() - t0 < 10.0


@needs_pidfd
def test_peer_watcher_stop_with_a_living_peer_closes_cleanly_and_stays_silent(procs):
    deaths = Deaths()
    watcher = rdv.PeerWatcher(deaths)
    watcher.start()
    proc = spawn_sleeper()
    procs.append(proc)
    watcher.add(proc.pid, "exp0")
    time.sleep(0.1)
    watcher.stop()
    watcher.join(30)
    assert not watcher.is_alive()
    proc.kill()
    proc.wait(30)
    time.sleep(0.1)
    assert deaths.items == []  # stopped watchers report nothing


@needs_pidfd
def test_peer_watcher_stop_is_safe_to_call_twice():
    """``ExpertServer.close`` has no closed guard: a second close calls ``stop()``
    again after the thread exited and its pipe is closed; that must not raise."""
    watcher = rdv.PeerWatcher(lambda pid, name: None)
    watcher.start()
    watcher.stop()
    watcher.join(30)
    watcher.stop()
