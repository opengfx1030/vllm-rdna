# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where to read from: sysfs/procfs roots, host name, store root.

One ``Env`` is shared by the CLI verbs and the MCP server so both discover the
topology and build the collectors identically (``--sysfs-root`` points the whole
tool at a captured tree for offline replays and tests).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pciescope import collectors, store, topology
from pciescope.collectors import BuildResult
from pciescope.model import Topology
from pciescope.sysfs import Sysfs


def parse_names(spec: str | Iterable[str] | None) -> list[str] | None:
    """``"aer,kfd"`` / ``["aer", "kfd"]`` -> names; ``None``/``default`` -> None.

    ``all`` expands to every registered collector.
    """
    if spec is None:
        return None
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    names = [n.strip() for chunk in items for n in chunk.split(",") if n.strip()]
    if not names or names == ["default"]:
        return None
    if names == ["all"]:
        return collectors.names()
    return names


@dataclass(slots=True)
class Env:
    """Roots and defaults of one tool invocation."""

    sysfs_root: str = "/sys"
    proc_root: str = "/proc"
    host: str | None = None
    store_root: str | None = None

    def sysfs(self) -> Sysfs:
        return Sysfs(self.sysfs_root)

    def procfs(self) -> Sysfs:
        return Sysfs(self.proc_root)

    def discover(self) -> Topology:
        return topology.discover(self.sysfs(), host=self.host)

    def build(
        self,
        topo: Topology,
        names: list[str] | None = None,
        *,
        allow_switch_writes: bool = False,
    ) -> BuildResult:
        """Instantiate collectors (see ``collectors.build``)."""
        options: dict[str, dict[str, Any]] = {"kfd": {"procfs": self.procfs()}}
        if allow_switch_writes:
            options["plx"] = {"allow_writes": True}
        return collectors.build(topo, names, sysfs=self.sysfs(), options=options)

    def session_root(self) -> Path:
        return Path(self.store_root) if self.store_root else store.default_root()


def load_topology(ref: str | os.PathLike[str], root: str | None = None) -> Topology:
    """Topology from a ``topo --json`` file or from a session reference."""
    p = Path(ref).expanduser()
    if p.is_file():
        doc = json.loads(p.read_text())
        return Topology.from_dict(doc.get("topology", doc))
    topo = store.Session.open(ref, root).topology()
    if topo is None:
        raise FileNotFoundError(f"session {ref} has no topology.json")
    return topo


def close_collectors(colls: Iterable[Any]) -> None:
    """Call ``close()`` on collectors that hold resources (the switch reader)."""
    for c in colls:
        closer = getattr(c, "close", None)
        if callable(closer):
            closer()
