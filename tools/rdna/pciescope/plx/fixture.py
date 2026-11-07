# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recorded PEX88096 register windows: capture on the box, replay anywhere.

A fixture is one JSON document holding the topology (``Topology.to_dict()``) and,
per switch, the non-zero dwords of the BAR0 window of every port (read-only reads,
read-sensitive registers excluded).  ``python -m pciescope.plx capture`` writes
one on the box (gzip-compressed when the name ends in ``.gz``); ``--fixture FILE``
replays it for ``ports`` / ``status`` / ``dump`` / ``perf plan`` without hardware,
and the tests run the collector over it.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pciescope.model import Topology, now_ns
from pciescope.plx import access
from pciescope.plx.ports import ChipMap

SCHEMA = 1


def capture(
    topo: Topology, client: access.PlxClient, maps: list[ChipMap]
) -> dict[str, Any]:
    """Read every port window of every chip through ``client`` (read-only)."""
    chips: dict[str, Any] = {}
    for m in maps:
        bdf = m.usp_bdf if m.usp_bdf.count(":") == 2 else f"0000:{m.usp_bdf}"
        wins = client.dump(bdf, [p.port for p in m.ports])
        chips[bdf] = {
            "switch": m.switch,
            "usp_port": m.usp_port,
            "windows": {
                str(port): {f"{off:03x}": f"{v:08x}" for off, v in sorted(w.items())}
                for port, w in sorted(wins.items())
            },
        }
    return {
        "schema": SCHEMA,
        "captured_ns": now_ns(),
        "topology": topo.to_dict(),
        "chips": chips,
    }


def save(doc: Mapping[str, Any], path: str | Path) -> None:
    data = (json.dumps(doc, indent=None, separators=(",", ":")) + "\n").encode()
    path = Path(path)
    if path.suffix == ".gz":
        data = gzip.compress(data, 9, mtime=0)  # reproducible bytes
    path.write_bytes(data)


def read_doc(path: str | Path) -> dict[str, Any]:
    """The JSON document of a fixture file (plain or ``.gz``)."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load(path: str | Path) -> tuple[Topology, access.LocalClient]:
    """(topology, read-only client) of a fixture file."""
    return from_doc(read_doc(path))


def from_doc(doc: Mapping[str, Any]) -> tuple[Topology, access.LocalClient]:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unsupported fixture schema {doc.get('schema')!r}")
    topo = Topology.from_dict(doc["topology"])
    chips = {
        bdf: (
            int(c["usp_port"]),
            {
                int(port): {int(off, 16): int(v, 16) for off, v in win.items()}
                for port, win in c["windows"].items()
            },
        )
        for bdf, c in doc["chips"].items()
    }
    return topo, access.fixture_client(chips)
