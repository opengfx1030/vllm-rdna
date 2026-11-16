# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clients of the BAR0 helper: subprocess (sudo), in-process (root) and fixture.

All of them expose the same small interface (``PlxClient``) so the collector, the
CLI and the tests do not care where the registers come from.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pciescope.plx import helper, regs
from pciescope.plx.helper import AccessError

__all__ = [
    "AccessError",
    "DictMmio",
    "LocalClient",
    "PlxClient",
    "SubprocessClient",
    "connect",
    "fixture_client",
]

_PKG_ROOT = Path(__file__).resolve().parents[2]  # directory that holds ``pciescope``


class PlxClient(Protocol):
    """What the collector needs from a register source."""

    def hello(self) -> dict[str, Any]: ...

    def read(self, bdf: str, items: Sequence[tuple[int, int]]) -> list[int]: ...

    def dump(self, bdf: str, ports: Sequence[int]) -> dict[int, dict[int, int]]: ...

    def perf_plan(
        self, bdf: str, ports: Sequence[int], stop: bool = False
    ) -> dict[str, Any]: ...

    def perf_start(self, bdf: str, ports: Sequence[int]) -> dict[str, Any]: ...

    def perf_stop(self, bdf: str) -> dict[str, Any]: ...

    def perf_info(self, bdf: str) -> dict[str, Any]: ...

    def perf_totals(self, bdf: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _windows(resp: Mapping[str, Any]) -> dict[int, dict[int, int]]:
    return {
        int(p): {int(o): int(v) for o, v in w.items()}
        for p, w in resp["windows"].items()
    }


class LocalClient:
    """In-process access (pciescope runs as root, or a fake register file)."""

    def __init__(self, chips: Mapping[str, helper.ChipAccess]) -> None:
        self.chips = dict(chips)

    def _chip(self, bdf: str) -> helper.ChipAccess:
        try:
            return self.chips[bdf]
        except KeyError:
            raise AccessError(f"unknown chip {bdf!r}") from None

    def hello(self) -> dict[str, Any]:
        return {"chips": {b: c.hello() for b, c in self.chips.items()}}

    def read(self, bdf: str, items: Sequence[tuple[int, int]]) -> list[int]:
        return self._chip(bdf).read(items)

    def dump(self, bdf: str, ports: Sequence[int]) -> dict[int, dict[int, int]]:
        return self._chip(bdf).dump(ports)

    def perf_plan(
        self, bdf: str, ports: Sequence[int], stop: bool = False
    ) -> dict[str, Any]:
        return self._chip(bdf).perf_plan(ports, stop=stop)

    def perf_start(self, bdf: str, ports: Sequence[int]) -> dict[str, Any]:
        return self._chip(bdf).perf_start(ports)

    def perf_stop(self, bdf: str) -> dict[str, Any]:
        return self._chip(bdf).perf_stop()

    def perf_info(self, bdf: str) -> dict[str, Any]:
        return self._chip(bdf).perf_info()

    def perf_totals(self, bdf: str) -> dict[str, Any]:
        return self._chip(bdf).perf_totals()

    def close(self) -> None:
        for c in self.chips.values():
            c.close()


class DictMmio:
    """Read-only BAR0 backed by recorded windows ``{port: {offset: dword}}``."""

    def __init__(self, windows: Mapping[int, Mapping[int, int]]) -> None:
        self.windows = {p: dict(w) for p, w in windows.items()}

    def read32(self, off: int) -> int:
        port, reg = divmod(off - regs.PORT_WINDOW_BASE, regs.PORT_WINDOW_SIZE)
        win = self.windows.get(port)
        if win is None or off < regs.PORT_WINDOW_BASE:
            raise AccessError(f"fixture has no window at BAR0 offset {off:#x}")
        return win.get(reg, 0)

    def write32(self, off: int, val: int) -> None:
        raise AccessError("fixture is read-only")


def fixture_client(
    chips: Mapping[str, tuple[int, Mapping[int, Mapping[int, int]]]],
) -> LocalClient:
    """A read-only client over recorded windows: ``{bdf: (usp_port, windows)}``."""
    return LocalClient(
        {
            bdf: helper.ChipAccess(bdf, usp_port, DictMmio(windows))
            for bdf, (usp_port, windows) in chips.items()
        }
    )


class SubprocessClient:
    """Talks JSON lines to ``sudo -n python -m pciescope.plx.helper``."""

    def __init__(
        self,
        chips: Mapping[str, int],
        *,
        allow_writes: bool | Collection[str] = False,
        sudo: bool = True,
        sysfs: str | None = None,
        timeout_s: float = 10.0,
        python: str | None = None,
    ) -> None:
        """``allow_writes``: ``True`` maps every chip read-write, a collection of
        chip BDFs only those (the others stay ``PROT_READ``), ``False`` none."""
        self.chips = dict(chips)
        self.allow_writes = allow_writes
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._buf = b""
        self._next_id = 0
        cmd: list[str] = []
        if sudo:
            if shutil.which("sudo") is None:
                raise AccessError("not root and sudo is not installed")
            cmd += ["sudo", "-n"]
        cmd += [python or sys.executable, "-m", "pciescope.plx.helper"]
        for bdf, port in self.chips.items():
            cmd += ["--chip", f"{bdf}:{port}"]
        if isinstance(allow_writes, bool):
            if allow_writes:
                cmd.append("--allow-writes")
        else:
            for bdf in sorted(allow_writes):
                cmd += ["--write-chip", bdf]
        if sysfs:
            cmd += ["--sysfs", sysfs]
        self._proc = subprocess.Popen(
            cmd,
            cwd=_PKG_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )
        try:
            self.info = self._call("hello")
        except AccessError:
            self.close()
            raise

    # -- transport ------------------------------------------------------------
    def _read_line(self, deadline: float) -> bytes:
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AccessError(f"helper did not answer within {self.timeout_s:g} s")
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise AccessError(self._died())
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _died(self) -> str:
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        err = b""
        if self._proc.stderr is not None:
            with contextlib.suppress(OSError):
                err = self._proc.stderr.read(4096) or b""
        text = err.decode("utf-8", "replace").strip().splitlines()
        detail = f": {text[-1]}" if text else ""
        return f"helper exited ({self._proc.returncode}){detail}"

    def _call(self, op: str, **kw: Any) -> dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                raise AccessError(self._died())
            self._next_id += 1
            req = {"id": self._next_id, "op": op, **kw}
            data = json.dumps(req, separators=(",", ":")).encode() + b"\n"
            try:
                self._proc.stdin.write(data)  # type: ignore[union-attr]
            except OSError:
                raise AccessError(self._died()) from None
            deadline = time.monotonic() + self.timeout_s
            while True:
                resp = json.loads(self._read_line(deadline))
                if resp.get("id") in (self._next_id, None):
                    break
            if not resp.get("ok"):
                raise AccessError(resp.get("error", "helper error"))
            return resp

    # -- PlxClient -------------------------------------------------------------
    def hello(self) -> dict[str, Any]:
        return self.info

    def read(self, bdf: str, items: Sequence[tuple[int, int]]) -> list[int]:
        return self._call("read", chip=bdf, regs=[list(i) for i in items])["vals"]

    def dump(self, bdf: str, ports: Sequence[int]) -> dict[int, dict[int, int]]:
        return _windows(self._call("dump", chip=bdf, ports=list(ports)))

    def perf_plan(
        self, bdf: str, ports: Sequence[int], stop: bool = False
    ) -> dict[str, Any]:
        return self._call("perf_plan", chip=bdf, ports=list(ports), stop=stop)

    def perf_start(self, bdf: str, ports: Sequence[int]) -> dict[str, Any]:
        return self._call("perf_start", chip=bdf, ports=list(ports))

    def perf_stop(self, bdf: str) -> dict[str, Any]:
        return self._call("perf_stop", chip=bdf)

    def perf_info(self, bdf: str) -> dict[str, Any]:
        return self._call("perf_info", chip=bdf)

    def perf_totals(self, bdf: str) -> dict[str, Any]:
        return self._call("perf_totals", chip=bdf)

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()  # EOF: the helper stops its counters, exits
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError):
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def connect(
    chips: Mapping[str, int],
    *,
    allow_writes: bool | Collection[str] = False,
    sysfs: str | None = None,
) -> PlxClient:
    """A client for the chips ``{usp_bdf (domain form): usp_port}``.

    Root opens BAR0 in-process; anybody else starts the helper through
    ``sudo -n`` (raises ``AccessError`` when that is not possible).
    ``allow_writes`` is ``True`` (every chip), a collection of chip BDFs (those
    only) or ``False``.
    """
    if os.geteuid() == 0:
        opened: dict[str, helper.ChipAccess] = {}
        try:
            for bdf, port in chips.items():
                opened[bdf] = helper.open_chip(
                    bdf,
                    port,
                    allow_writes=helper.chip_may_write(bdf, allow_writes),
                    sysfs=sysfs or helper.SYSFS_DEVICES,
                )
        except Exception:
            for c in opened.values():
                c.close()
            raise
        return LocalClient(opened)
    return SubprocessClient(chips, allow_writes=allow_writes, sysfs=sysfs)
