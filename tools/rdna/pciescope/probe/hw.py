# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static description of the GPUs of par1-cs25 (pure data, no I/O).

Source: docs/rdna2/pcie-iommu-audit-2026-11-06.md "Topology" and the lease
README of 2026-11-05 (verify on the box with ``topo``; this table is the
fallback when sysfs cannot be read and the key of the lease rules).

* switch A = PEX88096 behind root complex ``c0`` (root port c0:03.1):
  W7800 ``cb`` (HIP 0) and V620 ``ce d3 d8 db`` (HIP 1-4, "slot A").
* switch B = PEX88096 behind root complex ``00`` (root port 00:01.1):
  W7800 ``07`` (HIP 5) and V620 ``0a 0f 14 17`` (HIP 6-9, "slot B").

HIP indices change with the enumeration; they are only the expectation the
probe verifies at run time (``p2p_probe buses``).
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
    switch: str  # 'A' or 'B'
    lease_key: str  # holder-file suffix: slotA, slotB, w7800cb, w7800-07
    hip_hint: int  # HIP index on 2026-11-07 (expectation, not an authority)


KNOWN_GPUS: dict[str, GpuSpec] = {
    s.bus: s
    for s in (
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
    "slotA": ("ce", "d3", "d8", "db"),
    "slotB": ("0a", "0f", "14", "17"),
    "w7800cb": ("cb",),
    "w7800-07": ("07",),
}

# Never probed unless the lead creates '<owner>.w7800-07' (box rules).
LEAD_ONLY_KEYS = frozenset({"w7800-07"})

ROOTS = ("c0", "00")  # root complex ids; switch A hangs off c0, B off 00
PCIE_LINK = "16 GT/s x16"  # every GPU and switch port on the box (audit)
