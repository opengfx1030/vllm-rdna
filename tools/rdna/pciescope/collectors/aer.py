# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe Advanced Error Reporting counters (sysfs ``aer_dev_*``).

With ``pcie_ports=native`` the kernel owns AER and keeps per-device counters in
``aer_dev_correctable`` / ``aer_dev_nonfatal`` / ``aer_dev_fatal`` (one
``Name count`` line per error type, ending in ``TOTAL_ERR_*``) and per-root-port
totals in ``aer_rootport_total_err_{cor,nonfatal,fatal}``.  Per device and class
the total is always reported; the per-type breakdown (``aer.error``, tags
``class`` and ``type``) only for types with a non-zero count, so a healthy fabric
costs one short read per file.
"""

from __future__ import annotations

from pciescope.collectors import BaseCollector
from pciescope.model import ROLE_GPU_AUDIO, ROLE_ROOT_PORT, PciNode, Sample, Topology
from pciescope.model import now_ns as _now_ns
from pciescope.sysfs import Sysfs

NAME = "aer"

_CLASSES = (
    ("aer_dev_correctable", "correctable", "aer.correctable"),
    ("aer_dev_nonfatal", "nonfatal", "aer.nonfatal"),
    ("aer_dev_fatal", "fatal", "aer.fatal"),
)
_ROOT_TOTALS = (
    ("aer_rootport_total_err_cor", "aer.rootport_correctable"),
    ("aer_rootport_total_err_nonfatal", "aer.rootport_nonfatal"),
    ("aer_rootport_total_err_fatal", "aer.rootport_fatal"),
)


def parse_aer(text: str) -> dict[str, int]:
    """Parse an ``aer_dev_*`` file into ``{name: count}`` (incl. ``TOTAL_ERR_*``)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        name, _, val = line.rpartition(" ")
        if not name:
            continue
        try:
            out[name] = int(val)
        except ValueError:
            continue
    return out


def _total(text: str) -> int | None:
    """The ``TOTAL_ERR_*`` count: it is the last line of the file."""
    last = text.rstrip().rsplit("\n", 1)[-1]
    name, _, val = last.rpartition(" ")
    if not name.startswith("TOTAL_ERR_"):
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _sysname(bdf: str) -> str:
    return bdf if bdf.count(":") == 2 else f"0000:{bdf}"


class AerCollector(BaseCollector):
    """AER error counters of every port, GPU and root port of the fabric."""

    name = NAME

    def __init__(self, topo: Topology, sysfs: Sysfs | None = None) -> None:
        super().__init__(topo, sysfs)
        self._devs: list[tuple[PciNode, str, str]] = []
        for n in topo.nodes.values():
            if n.role == ROLE_GPU_AUDIO or not n.has_aer:
                continue
            self._devs.append((n, n.entity, f"bus/pci/devices/{_sysname(n.bdf)}"))

    def available(self) -> tuple[bool, str]:
        if not self._devs:
            return False, "no device of the fabric exposes aer_dev_correctable"
        _, _, base = self._devs[0]
        if self.sysfs.read(f"{base}/aer_dev_correctable") is None:
            return False, "aer_dev_correctable not readable"
        return True, ""

    def collect(self) -> list[Sample]:
        ts = _now_ns()
        read = self.sysfs.read
        out: list[Sample] = []
        for n, ent, base in self._devs:
            for fname, cls, metric in _CLASSES:
                text = read(f"{base}/{fname}")
                if text is None:
                    continue
                total = _total(text)
                if total is None:
                    continue
                out.append(Sample(ts, ent, metric, float(total), "count"))
                if total:
                    for name, count in parse_aer(text).items():
                        if count and not name.startswith("TOTAL_ERR_"):
                            out.append(
                                Sample(
                                    ts,
                                    ent,
                                    "aer.error",
                                    float(count),
                                    "count",
                                    {"class": cls, "type": name},
                                )
                            )
            if n.role == ROLE_ROOT_PORT:
                for fname, metric in _ROOT_TOTALS:
                    val = self.sysfs.read_int(f"{base}/{fname}")
                    if val is not None:
                        out.append(Sample(ts, ent, metric, float(val), "count"))
        return out


def make(
    topo: Topology, *, sysfs: Sysfs | None = None, **_opts: object
) -> AerCollector:
    return AerCollector(topo, sysfs)
