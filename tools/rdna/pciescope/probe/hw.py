# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static description of the GPUs of par1-cs25 (pure data, no I/O).

Two arrangements of the box are described.  Their buses are disjoint, so one table
serves both and the topology found at run time says which one is installed:

* interim, since the 2026-11-08 case move and until the replacement switch
  arrives (box rules, lease README of 2026-11-10, ``python -m pciescope topo`` on
  the box): ONE PEX88096 on a SlimSAS card behind root port c0:03.1 carries six
  V620, every link Gen4 x8, no W7800.  Leaf A (``d1 d4`` = HIP 0, 1) is lease
  ``a2``, leaf B (``d9 dc df e2`` = HIP 2-5) is lease ``b4``.
* pre-move, 2026-11-07 (docs/rdna2/pcie-iommu-audit-2026-11-06.md "Topology" and
  the lease README of 2026-11-05), kept for the recorded fixtures and for the day
  the two original cards come back: switch A behind root complex ``c0`` (W7800
  ``cb`` = HIP 0 and V620 ``ce d3 d8 db`` = HIP 1-4, "slot A") and switch B behind
  root complex ``00`` (W7800 ``07`` = HIP 5 and V620 ``0a 0f 14 17`` = HIP 6-9,
  "slot B").  The bus numbers of a rebuilt box will differ again: re-derive then.

HIP indices change with the enumeration; they are only the expectation the probe
verifies at run time (``p2p_probe buses``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    """One GPU of the box."""

    bus: str  # 2-hex-digit PCI bus of function 0
    name: str  # marketing name
    arch: str  # gfx target
    root: str  # root complex id ('00' or 'c0')
    switch: str  # 'A' or 'B', the letter the topology gives the chip
    lease_key: str  # holder-file suffix: a2, b4, slotA, slotB, w7800cb, w7800-07
    hip_hint: int  # HIP index in that arrangement (expectation, not an authority)


KNOWN_GPUS: dict[str, GpuSpec] = {
    s.bus: s
    for s in (
        # interim arrangement (2026-11-10): one switch, two leaves
        GpuSpec("d1", "Radeon Pro V620", "gfx1030", "c0", "A", "a2", 0),
        GpuSpec("d4", "Radeon Pro V620", "gfx1030", "c0", "A", "a2", 1),
        GpuSpec("d9", "Radeon Pro V620", "gfx1030", "c0", "A", "b4", 2),
        GpuSpec("dc", "Radeon Pro V620", "gfx1030", "c0", "A", "b4", 3),
        GpuSpec("df", "Radeon Pro V620", "gfx1030", "c0", "A", "b4", 4),
        GpuSpec("e2", "Radeon Pro V620", "gfx1030", "c0", "A", "b4", 5),
        # pre-move arrangement (2026-11-07): two switches on two root complexes
        GpuSpec("cb", "Radeon PRO W7800", "gfx1100", "c0", "A", "w7800cb", 0),
        GpuSpec("ce", "Radeon Pro V620", "gfx1030", "c0", "A", "slotA", 1),
        GpuSpec("d3", "Radeon Pro V620", "gfx1030", "c0", "A", "slotA", 2),
        GpuSpec("d8", "Radeon Pro V620", "gfx1030", "c0", "A", "slotA", 3),
        GpuSpec("db", "Radeon Pro V620", "gfx1030", "c0", "A", "slotA", 4),
        GpuSpec("07", "Radeon PRO W7800", "gfx1100", "00", "B", "w7800-07", 5),
        GpuSpec("0a", "Radeon Pro V620", "gfx1030", "00", "B", "slotB", 6),
        GpuSpec("0f", "Radeon Pro V620", "gfx1030", "00", "B", "slotB", 7),
        GpuSpec("14", "Radeon Pro V620", "gfx1030", "00", "B", "slotB", 8),
        GpuSpec("17", "Radeon Pro V620", "gfx1030", "00", "B", "slotB", 9),
    )
}

# Lease key -> buses covered by that holder file.
LEASE_KEYS: dict[str, tuple[str, ...]] = {
    "a2": ("d1", "d4"),
    "b4": ("d9", "dc", "df", "e2"),
    "slotA": ("ce", "d3", "d8", "db"),
    "slotB": ("0a", "0f", "14", "17"),
    "w7800cb": ("cb",),
    "w7800-07": ("07",),
}

#: keys of groups of V620 (the lease check of ``lease.py`` applies to them); the
#: other keys name one W7800
SLOT_KEYS = frozenset(s.lease_key for s in KNOWN_GPUS.values() if s.arch == "gfx1030")

# Never probed unless the lead creates '<owner>.w7800-07' (box rules).  No W7800 is
# installed in the interim arrangement.
LEAD_ONLY_KEYS = frozenset({"w7800-07"})

ROOTS = ("c0", "00")  # root complex ids; interim: all GPUs behind c0
