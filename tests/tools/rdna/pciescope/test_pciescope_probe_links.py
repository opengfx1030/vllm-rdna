# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Link watcher: the best PCIe link state seen on the probed paths.

Real data: the captured sysfs of par1-cs25 (every link at 16 GT/s x16). A
writable copy lets a test lower or restore ``current_link_speed`` /
``current_link_width`` the way a GPU's idle power management does.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope.model import LINK_HOST_UPLINK  # noqa: E402
from pciescope.probe.links import LinkWatcher  # noqa: E402
from pciescope.sysfs import Sysfs  # noqa: E402

SLOT_B = ["0a", "0f", "14", "17"]


@pytest.fixture
def sys_copy(tmp_path: Path, fx_root: Path) -> Path:
    shutil.copytree(fx_root / "sys", tmp_path / "sys", symlinks=True)
    return tmp_path / "sys"


def device(sys_root: Path, bdf: str) -> Path:
    return sys_root / "bus/pci/devices" / f"0000:{bdf}"


def watcher(
    topo, sys_root: Path, buses: list[str] = SLOT_B, **kw: float
) -> LinkWatcher:
    gpus = [topo.gpus[f"gpu:{b}"] for b in buses]
    return LinkWatcher(topo, gpus, sysfs=Sysfs(sys_root), **kw)


def test_it_watches_the_gpu_links_and_the_host_uplink_of_the_switch(
    fx_topo, sys_copy
) -> None:
    w = watcher(fx_topo, sys_copy)
    links = [fx_topo.links[lid] for lid in w.links]
    assert all(lk.physical for lk in links)
    assert {lk.gpu for lk in links if lk.gpu} == {f"gpu:{b}" for b in SLOT_B}
    uplinks = [lk for lk in links if lk.kind == LINK_HOST_UPLINK]
    assert [lk.switch for lk in uplinks] == ["B"]
    assert "link:04:10.0-08:00.0" in w.links  # the switch port of GPU 0a
    other = watcher(fx_topo, sys_copy, ["ce"])
    assert [
        lk.switch
        for lk in map(fx_topo.links.get, other.links)
        if lk.kind == LINK_HOST_UPLINK
    ] == ["A"]


def test_the_captured_box_has_no_degraded_link(fx_topo, sys_copy) -> None:
    w = watcher(fx_topo, sys_copy)
    w.sample_once()
    assert w.n_samples == 1 and set(w.best) == set(w.links)
    assert all(state == (16.0, 16) for state in w.best.values())
    assert w.degraded() == {}


def test_a_downtrained_link_is_reported_with_its_state_and_its_maximum(
    fx_topo, sys_copy
) -> None:
    (device(sys_copy, "08:00.0") / "current_link_speed").write_text("8.0 GT/s PCIe\n")
    (device(sys_copy, "0f:00.0") / "current_link_width").write_text("8\n")
    w = watcher(fx_topo, sys_copy)
    w.sample_once()
    bad = w.degraded()
    assert bad["link:04:10.0-08:00.0"] == "8 GT/s x16 (max 16 GT/s x16)"
    assert bad["link:0e:00.0-0f:00.0"] == "16 GT/s x8 (max 16 GT/s x16)"
    assert set(bad) == {"link:04:10.0-08:00.0", "link:0e:00.0-0f:00.0"}


def test_the_best_state_seen_wins_because_idle_gpus_lower_their_speed(
    fx_topo, sys_copy
) -> None:
    speed = device(sys_copy, "08:00.0") / "current_link_speed"
    speed.write_text("2.5 GT/s PCIe\n")  # idle
    w = watcher(fx_topo, sys_copy)
    w.sample_once()
    assert "link:04:10.0-08:00.0" in w.degraded()
    speed.write_text("16.0 GT/s PCIe\n")  # under load
    w.sample_once()
    assert w.degraded() == {}
    speed.write_text("2.5 GT/s PCIe\n")  # idle again: the best state is kept
    w.sample_once()
    assert w.degraded() == {} and w.n_samples == 3


def test_unreadable_link_files_leave_no_verdict(fx_topo, sys_copy) -> None:
    (device(sys_copy, "08:00.0") / "current_link_speed").unlink()
    w = watcher(fx_topo, sys_copy)
    w.sample_once()
    assert "link:04:10.0-08:00.0" not in w.best
    assert w.degraded() == {}


def test_the_thread_samples_until_stopped(fx_topo, sys_copy) -> None:
    w = watcher(fx_topo, sys_copy, period_s=0.02)
    w.start()
    w.start()  # a second start does nothing
    deadline = time.monotonic() + 5
    while w.n_samples < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    w.stop()
    assert time.monotonic() - started < 2
    assert w.n_samples >= 3
    count = w.n_samples
    time.sleep(0.1)
    assert w.n_samples == count, "no sample after stop"
    w.stop()  # idempotent


def test_stop_is_prompt_with_a_long_period(fx_topo, sys_copy) -> None:
    w = watcher(fx_topo, sys_copy, period_s=600.0)
    w.start()
    deadline = time.monotonic() + 5
    while w.n_samples < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    w.stop()
    assert time.monotonic() - started < 2


def test_stop_without_start(fx_topo, sys_copy) -> None:
    watcher(fx_topo, sys_copy).stop()
