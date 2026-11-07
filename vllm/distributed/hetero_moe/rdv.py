# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host rendezvous of the ``hx`` processes (hub and expert servers).

Everything lives in one directory per plan, ``/dev/shm/hx_<plan_id>/``; the names
carry the plan's device ids (``exp0``, ``hub0``; see ``hx.py`` ``*_path``):

* ``srv_<id>.json``       server ``<id>`` is ready: IPC handle of its inbox,
                          geometry, pid, gcnArchName, allocation kind (written
                          last, after load, self-test and graph capture, so a hub
                          that sees it can run).
* ``hub_<id>.json``       hub ``<id>``: IPC handle of its reply region, pid,
                          geometry.
* ``ack_<srv>_<hub>.json``  server ``<srv>`` opened hub ``<hub>`` and is serving it.
* ``ring_<hub>.bin``      the step ring of hub ``<hub>`` (``ring.py``).

Files are written atomically (temp file + ``os.replace``). Liveness of a peer is
a pidfd: a process that died is noticed within a poll interval, whatever the GPU
side is doing; the caller then pokes the abort word of its own control page
(``ops.hx_host_abort``) so kernels spinning on a flag leave.
"""

import contextlib
import json
import os
import select
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SHM_ROOT = "/dev/shm"


def hx_dir(plan_id: str, root: str = SHM_ROOT) -> Path:
    """The rendezvous directory of a plan (created by whoever comes first)."""
    path = Path(root) / f"hx_{plan_id}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Atomic publish: readers see the old file or the whole new one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_json(
    path: Path,
    timeout_s: float,
    *,
    what: str,
    alive: Callable[[], str | None] | None = None,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    """Block until ``path`` exists. ``alive`` returns an error text when the
    awaited peer is gone (the wait then fails at once instead of timing out)."""
    deadline = time.monotonic() + timeout_s
    while True:
        obj = read_json(path)
        if obj is not None:
            return obj
        if alive is not None:
            err = alive()
            if err:
                raise RuntimeError(f"hx rendezvous: {what}: {err}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"hx rendezvous: {what}: {path} did not appear within {timeout_s:g} s"
            )
        time.sleep(poll_s)


def remove_quiet(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); /proc tells.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def _pidfd_open(pid: int) -> int:
    # os.pidfd_open exists on Linux only (Python 3.9+): looked up when called
    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        raise RuntimeError("hx peer watching needs os.pidfd_open (Linux)")
    return int(opener(pid))


class PeerWatcher(threading.Thread):
    """Calls ``on_death(pid, name)`` once for each watched peer that exits.

    ``pidfd_open`` + ``poll``: no polling interval, no pid-reuse race once the fd
    is open. Peers that are already gone when added are reported at once.
    """

    def __init__(self, on_death: Callable[[int, str], None]):
        super().__init__(name="hx-peer-watcher", daemon=True)
        self._on_death = on_death
        self._lock = threading.Lock()
        self._fds: dict[int, tuple[int, str]] = {}  # fd -> (pid, name)
        self._wake_r, self._wake_w = os.pipe()
        self._stop_flag = False
        self.dead: list[tuple[int, str]] = []

    def add(self, pid: int, name: str) -> None:
        try:
            fd = _pidfd_open(pid)
        except ProcessLookupError:
            self._report(pid, name)
            return
        with self._lock:
            if self._stop_flag:  # stopped: the descriptors are closed or closing
                os.close(fd)
                return
            self._fds[fd] = (pid, name)
            os.write(self._wake_w, b"x")

    def stop(self) -> None:
        """Idempotent. The thread closes its descriptors when it leaves; a
        watcher that was never started closes them here."""
        with self._lock:
            if self._stop_flag:
                return
            self._stop_flag = True
            if self.ident is None:  # never started: nobody else will clean up
                self._close_fds()
            else:
                os.write(self._wake_w, b"x")

    def _close_fds(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        os.close(self._wake_r)
        os.close(self._wake_w)

    def _report(self, pid: int, name: str) -> None:
        self.dead.append((pid, name))
        self._on_death(pid, name)

    def run(self) -> None:
        poller = select.poll()
        poller.register(self._wake_r, select.POLLIN)
        registered: set[int] = set()
        while not self._stop_flag:
            with self._lock:
                for fd in self._fds:
                    if fd not in registered:
                        poller.register(fd, select.POLLIN)
                        registered.add(fd)
            for fd, _ in poller.poll():
                if fd == self._wake_r:
                    os.read(self._wake_r, 64)
                    continue
                with self._lock:
                    pid, name = self._fds.pop(fd)
                poller.unregister(fd)
                registered.discard(fd)
                os.close(fd)
                self._report(pid, name)
        with self._lock:
            self._close_fds()
