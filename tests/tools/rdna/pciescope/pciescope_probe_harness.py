# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness of the probe runner/API tests: a writable fake box.

``make_box`` copies the captured sysfs/procfs of par1-cs25 (``fx_root``) into a
temporary directory, so a test can change a junction temperature or a KFD
``evicted_ms`` counter, and puts an executable ``p2p_probe`` double
(``fixtures/probe/fake_p2p_probe.py``, SYNTHETIC numbers) next to it. Nothing
here touches the real box.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pciescope.model import Topology
from pciescope.probe.guard import EvictionWatcher
from pciescope.probe.runner import GuardEnv, RunLimits
from pciescope.sysfs import Sysfs

FAKE_PROBE = Path(__file__).parent / "fixtures" / "probe" / "fake_p2p_probe.py"


def wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@dataclass
class FakeBox:
    root: Path
    topo: Topology
    binary: Path
    spec_path: Path
    log: Path
    quiet_dir: Path
    boot_lock: Path
    lease_dir: Path
    proc_root: Path
    spec: dict[str, Any] = field(default_factory=dict)

    # -- the trees the guards read ------------------------------------------

    @property
    def sysfs(self) -> Sysfs:
        return Sysfs(self.root / "sys")

    @property
    def procfs(self) -> Sysfs:
        return Sysfs(self.root / "proc")

    @property
    def env(self) -> GuardEnv:
        return GuardEnv(
            sysfs=self.sysfs,
            procfs=self.procfs,
            proc_root=self.proc_root,
            quiet_dir=self.quiet_dir,
            boot_lock=self.boot_lock,
            lease_dir=self.lease_dir,
        )

    def limits(self, **over: Any) -> RunLimits:
        """Limits short enough for tests (override any field)."""
        base: dict[str, Any] = {
            "max_seconds": 60.0,
            "step_floor_s": 5.0,
            "start_s": 10.0,
            "noisy_wait_s": 2.0,
            "temp_period_s": 0.05,
            "kill_grace_s": 1.0,
            "close_grace_s": 5.0,
            "poll_s": 0.05,
        }
        base.update(over)
        return RunLimits(**base)

    # -- the fake probe --------------------------------------------------------

    def set_spec(self, **kw: Any) -> None:
        self.spec.update(kw)
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")

    def add_event(self, **ev: Any) -> None:
        self.set_spec(events=[*self.spec.get("events", []), ev])

    def commands(self) -> list[str]:
        """Commands the probe double received, in order."""
        if not self.log.is_file():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [ln.removeprefix("cmd ") for ln in lines if ln.startswith("cmd ")]

    def log_lines(self) -> list[str]:
        if not self.log.is_file():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()

    # -- files the guards read -------------------------------------------------

    def temp_path(self, bus: str) -> Path:
        gpu = self.topo.gpu(bus)
        assert gpu.hwmon
        return self.root / "sys" / gpu.hwmon / "temp2_input"

    def set_temp(self, bus: str, celsius: float) -> None:
        self.temp_path(bus).write_text(f"{int(celsius * 1000)}\n", encoding="utf-8")

    def quiet(self, name: str, age_s: float = 0.0) -> Path:
        path = self.quiet_dir / name
        path.write_text("", encoding="utf-8")
        when = time.time() - age_s
        os.utime(path, (when, when))
        return path

    def lease(self, owner: str, key: str, text: str | None = None) -> Path:
        path = self.lease_dir / f"{owner}.{key}"
        path.write_text(
            text or f"{owner} 2026-11-07T15:00:00+01:00 test\n", encoding="utf-8"
        )
        return path

    def kfd_procs(self, buses: list[str]) -> dict[int, dict[str, int]]:
        """Pid -> bus -> evicted_ms of the KFD processes on ``buses`` (as captured)."""
        gpus = [self.topo.gpu(b) for b in buses]
        watcher = EvictionWatcher(gpus, sysfs=self.sysfs, procfs=self.procfs)
        return {pid: dict(p.evicted_ms) for pid, p in watcher.snapshot().items()}

    def evicted_path(self, pid: int, bus: str) -> Path:
        gpu = self.topo.gpu(bus)
        return (
            self.root
            / "sys"
            / "class/kfd/kfd/proc"
            / str(pid)
            / f"stats_{gpu.kfd_gpu_id}"
            / "evicted_ms"
        )

    def set_evicted(self, pid: int, bus: str, ms: int) -> None:
        path = self.evicted_path(pid, bus)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{ms}\n", encoding="utf-8")

    def new_kfd_process(self, pid: int, comm: str, buses: list[str]) -> None:
        for bus in buses:
            self.set_evicted(pid, bus, 0)
        proc = self.root / "proc" / str(pid)
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "comm").write_text(comm + "\n", encoding="utf-8")


def write_launcher(path: Path) -> Path:
    """An executable that runs the probe double (``$FAKE_PROBE_SPEC`` configures it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_PROBE}" "$@"\n', encoding="utf-8"
    )
    path.chmod(0o755)
    return path


def make_box(tmp_path: Path, fx_root: Path, topo: Topology) -> FakeBox:
    root = tmp_path / "box"
    shutil.copytree(fx_root, root, symlinks=True)
    for name in ("quiet", "lease", "procroot"):
        (tmp_path / name).mkdir()
    boot_lock = tmp_path / "boot.lock"
    boot_lock.write_text("", encoding="utf-8")
    binary = write_launcher(tmp_path / "bin" / "p2p_probe")
    box = FakeBox(
        root=root,
        topo=topo,
        binary=binary,
        spec_path=tmp_path / "spec.json",
        log=tmp_path / "probe.log",
        quiet_dir=tmp_path / "quiet",
        boot_lock=boot_lock,
        lease_dir=tmp_path / "lease",
        proc_root=tmp_path / "procroot",
    )
    box.set_spec(log=str(box.log))
    return box
