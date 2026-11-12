# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collector registry: build the passive collectors available on this host.

A collector module defines ``NAME`` and ``make(topo, *, sysfs=None, **opts)``
returning an object that satisfies ``pciescope.model.Collector``
(``name``, ``available() -> (bool, reason)``, ``collect() -> list[Sample]``;
optionally ``drain_events() -> list[Event]``).  Modules are imported lazily, so a
missing optional module or dependency only skips that collector.

Registered names: ``sysfs_link`` ``aer`` ``amdgpu`` ``kfd`` (Core) and ``plx``
(the PEX88096 register reader, PlxSwitch).  ``DEFAULT_PASSIVE`` is what ``snap``
and ``watch`` use when no ``--collectors`` is given; a collector that cannot work
here (``plx`` without root or passwordless sudo, or on a captured tree) is
skipped with its reason.  Nothing in the default set writes to hardware.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from pciescope.model import Collector, Sample, Topology
from pciescope.sysfs import Sysfs

__all__ = [
    "DEFAULT_PASSIVE",
    "REGISTRY",
    "BaseCollector",
    "BuildResult",
    "Collector",
    "build",
    "names",
]

#: collector name -> module that provides ``make``
REGISTRY: dict[str, str] = {
    "sysfs_link": "pciescope.collectors.sysfs_link",
    "aer": "pciescope.collectors.aer",
    "amdgpu": "pciescope.collectors.amdgpu",
    "kfd": "pciescope.collectors.kfd",
    "plx": "pciescope.collectors.plx",
}

#: collectors enabled by default: sysfs reads and the read-only switch register
#: reader (switch register writes need ``allow_writes`` and are never default)
DEFAULT_PASSIVE: tuple[str, ...] = ("sysfs_link", "aer", "amdgpu", "kfd", "plx")


class BaseCollector:
    """Convenience base: holds the topology and the sysfs view."""

    name = "base"

    def __init__(self, topo: Topology, sysfs: Sysfs | None = None) -> None:
        self.topo = topo
        self.sysfs = sysfs or Sysfs()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def collect(self) -> list[Sample]:
        raise NotImplementedError


@dataclass(slots=True)
class BuildResult:
    """Outcome of ``build``: usable collectors and why the others were skipped."""

    collectors: list[Collector] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def names() -> list[str]:
    """All registered collector names."""
    return list(REGISTRY)


def build(
    topo: Topology,
    selected: list[str] | tuple[str, ...] | None = None,
    *,
    sysfs: Sysfs | None = None,
    options: dict[str, dict[str, Any]] | None = None,
) -> BuildResult:
    """Instantiate ``selected`` (default ``DEFAULT_PASSIVE``) for ``topo``.

    A collector whose module cannot be imported or whose ``available()`` is false
    lands in ``skipped`` with the reason; nothing raises for a missing source.
    """
    res = BuildResult()
    for name in selected if selected is not None else DEFAULT_PASSIVE:
        modname = REGISTRY.get(name)
        if modname is None:
            res.skipped.append(
                (name, f"unknown collector (known: {', '.join(REGISTRY)})")
            )
            continue
        try:
            mod = importlib.import_module(modname)
        except ImportError as exc:
            res.skipped.append((name, f"module not importable: {exc}"))
            continue
        opts = dict((options or {}).get(name, {}))
        try:
            coll = mod.make(topo, sysfs=sysfs, **opts)
            ok, why = coll.available()
        except Exception as exc:  # a broken collector must not take the others down
            res.skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        if ok:
            res.collectors.append(coll)
        else:
            res.skipped.append((name, why or "unavailable"))
    return res
