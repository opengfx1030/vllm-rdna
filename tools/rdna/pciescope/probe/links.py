# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best PCIe link state seen on the probed paths while a run is in progress.

A GPU lowers its PCIe link speed when idle (and a runtime-suspended GPU reads
2.5 GT/s x1), so one reading before or after a run says little. The watcher
samples Core's ``sysfs_link`` collector (passive sysfs reads) every
``period_s`` while the probe runs and keeps, per link, the sample with the
highest capacity. A link whose best sample stays below the link maximum is
reported as degraded: the measured bandwidth on it is limited by the link,
not by the fabric (finding ``link-downtrained`` of ``effectiveness``).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from pciescope.collectors import sysfs_link
from pciescope.model import LINK_HOST_UPLINK, Gpu, Topology, raw_gbs
from pciescope.sysfs import Sysfs

_JOIN_TIMEOUT_S = 5.0


class LinkWatcher:
    """Daemon thread that keeps the best link state per physical link."""

    def __init__(
        self,
        topo: Topology,
        gpus: Sequence[Gpu],
        *,
        sysfs: Sysfs | None = None,
        period_s: float = 0.5,
    ) -> None:
        self.topo = topo
        self.period_s = period_s
        self._collector = sysfs_link.make(topo, sysfs=sysfs)
        entities = {g.id for g in gpus}
        switches = {g.switch for g in gpus if g.switch}
        self.links = sorted(
            lk.id
            for lk in topo.links.values()
            if lk.physical
            and (
                lk.gpu in entities
                or (lk.kind == LINK_HOST_UPLINK and lk.switch in switches)
            )
        )
        self.best: dict[str, tuple[float, int]] = {}  # link -> (GT/s, lanes)
        self.n_samples = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        """Read the link state once and keep the best one per link."""
        wanted = set(self.links)
        values: dict[str, dict[str, float]] = {}
        for s in self._collector.collect():
            if s.entity in wanted:
                values.setdefault(s.entity, {})[s.metric] = s.value
        with self._lock:
            self.n_samples += 1
            for link, vals in values.items():
                self._keep(link, vals)

    def _keep(self, link: str, vals: dict[str, float]) -> None:
        speed, width = vals.get("link.speed_gts"), vals.get("link.width")
        cap = vals.get("link.capacity_gbs")
        if speed is None or width is None or cap is None:
            return
        old = self.best.get(link)
        if old is None or cap > _capacity(old):
            self.best[link] = (speed, int(width))

    def degraded(self) -> dict[str, str]:
        """Links whose best state is below their maximum, with a text each."""
        out: dict[str, str] = {}
        with self._lock:
            for link, (speed, width) in sorted(self.best.items()):
                info = self.topo.links[link]
                max_speed, max_width = info.max_speed_gts, info.max_width
                if max_speed is None or max_width is None:
                    continue
                if speed < max_speed or width < max_width:
                    out[link] = (
                        f"{speed:g} GT/s x{width} (max {max_speed:g} GT/s x{max_width})"
                    )
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            if self._stop.wait(self.period_s):
                return

    def start(self) -> None:
        """Start the sampling thread (no-op when it already runs)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="pciescope-link-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT_S)


def _capacity(state: tuple[float, int]) -> float:
    return raw_gbs(state[0], state[1]) if state[1] > 0 else 0.0
