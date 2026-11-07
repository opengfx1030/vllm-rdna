# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Link speed and width of every port and link of the fabric (sysfs only).

Reads ``current_link_speed`` / ``current_link_width`` (the device's own LnkSta)
of each node of the topology and derives per-link values from the two ends of
every physical link:

* ``pcie.link_speed_gts`` / ``pcie.link_width`` for ``port:<BDF>`` (``gpu:<bus>``
  for the GPU function itself, i.e. the on-card link from the GPU side);
* ``link.speed_gts`` / ``link.width`` (minimum of both ends),
  ``link.capacity_gbs`` (raw payload rate at that speed and width) and
  ``link.degraded`` (below the link maximum, or the ends disagree) for every
  physical link (host uplink, GPU links, on-card links).

Virtual links inside the switch (fabric, management) carry no lanes and are not
reported at the link level.
"""

from __future__ import annotations

from pciescope.collectors import BaseCollector
from pciescope.model import (
    ROLE_GPU_AUDIO,
    Link,
    PciNode,
    Sample,
    Topology,
    parse_speed_gts,
    raw_gbs,
)
from pciescope.model import now_ns as _now_ns
from pciescope.sysfs import Sysfs

NAME = "sysfs_link"

_SPEED_CACHE: dict[str, float | None] = {}


def _speed(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return _SPEED_CACHE[text]
    except KeyError:
        return _SPEED_CACHE.setdefault(text, parse_speed_gts(text))


def _sysname(bdf: str) -> str:
    return bdf if bdf.count(":") == 2 else f"0000:{bdf}"


class SysfsLinkCollector(BaseCollector):
    """Per-port and per-link PCIe link state from sysfs."""

    name = NAME

    def __init__(self, topo: Topology, sysfs: Sysfs | None = None) -> None:
        super().__init__(topo, sysfs)
        self._ports: list[tuple[PciNode, str, str, str]] = []
        for n in topo.nodes.values():
            if n.role == ROLE_GPU_AUDIO:
                continue
            base = f"bus/pci/devices/{_sysname(n.bdf)}"
            self._ports.append(
                (
                    n,
                    n.entity,
                    f"{base}/current_link_speed",
                    f"{base}/current_link_width",
                )
            )
        self._links: list[Link] = [lk for lk in topo.links.values() if lk.physical]

    def available(self) -> tuple[bool, str]:
        if not self._ports:
            return False, "topology has no PCI nodes"
        n, _, sp, _ = self._ports[0]
        if self.sysfs.read(sp) is None:
            return False, f"cannot read {sp}"
        return True, ""

    def collect(self) -> list[Sample]:
        ts = _now_ns()
        read, read_int = self.sysfs.read, self.sysfs.read_int
        out: list[Sample] = []
        state: dict[str, tuple[float | None, int | None]] = {}
        for n, ent, sp, wp in self._ports:
            speed = _speed(read(sp))
            width = read_int(wp)
            state[n.bdf] = (speed, width)
            if speed is not None:
                out.append(Sample(ts, ent, "pcie.link_speed_gts", speed, "GT/s"))
            if width is not None:
                out.append(Sample(ts, ent, "pcie.link_width", float(width), "lanes"))
        for lk in self._links:
            su, wu = state.get(lk.up, (None, None))
            sd, wd = state.get(lk.down, (None, None))
            if su is None or sd is None or wu is None or wd is None:
                continue
            speed, width = min(su, sd), min(wu, wd)
            out.append(Sample(ts, lk.id, "link.speed_gts", speed, "GT/s"))
            out.append(Sample(ts, lk.id, "link.width", float(width), "lanes"))
            cap = raw_gbs(speed, width) if width > 0 else 0.0
            out.append(Sample(ts, lk.id, "link.capacity_gbs", round(cap, 3), "GB/s"))
            degraded = (
                su != sd
                or wu != wd
                or (lk.max_speed_gts is not None and speed < lk.max_speed_gts)
                or (lk.max_width is not None and width < lk.max_width)
            )
            out.append(Sample(ts, lk.id, "link.degraded", float(degraded), "bool"))
        return out


def make(
    topo: Topology, *, sysfs: Sysfs | None = None, **_opts: object
) -> SysfsLinkCollector:
    return SysfsLinkCollector(topo, sysfs)
