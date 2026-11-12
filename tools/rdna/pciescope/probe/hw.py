# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease layouts of par1-cs25: which holder-file key covers which GPUs (pure data).

Source: the box rules, ``~/w4a8_runs/lanes/lease/README.md``. A lease key is a
policy of those rules, not a property of the wiring, so it is taken from the
layout whose GPU buses equal the buses of the live topology, never from a bus
alone. ``pciescope.topology.DOCUMENTED_LAYOUTS`` (Core) lists the same layouts
by bus, HIP order and slot letter; a test keeps the two tables in step. When
the hardware changes again (the replacement switch) add the layout to both, in
the same commit that follows the README of the box.

* ``INTERIM`` (since the case move of 2026-11-08, README of 2026-11-10): six
  V620 behind ONE PEX88096, every GPU link Gen4 x8. ``b4`` = HIP 2-5 = ``d9 dc
  df e2`` (leaf B), ``a2`` = HIP 0-1 = ``d1 d4`` (leaf A). A job on HIP 1 plus
  HIP 2-5 needs both holder files. No W7800 is installed.
* ``TEN_GPU`` (2026-11-05 until the case move, ``README.pre-casemove-2026-11-08.md``,
  invalid on the box now): ``slotA`` = V620 ``ce d3 d8 db`` (switch A, HIP 1-4),
  ``slotB`` = V620 ``0a 0f 14 17`` (switch B, HIP 6-9), ``w7800cb`` = W7800 ``cb``
  (HIP 0), ``w7800-07`` = W7800 ``07`` (HIP 5, never without the lead).

HIP indices change with the enumeration; they are only the expectation the
probe verifies at run time (``p2p_probe buses``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """One GPU layout of the box with the lease key of every GPU."""

    name: str
    keys: tuple[tuple[str, tuple[str, ...]], ...]  # lease key -> buses (README order)
    slot_keys: tuple[tuple[str, str], ...]  # ``Topology.slot_gpus`` letter -> key
    lead_only: tuple[tuple[str, str], ...] = ()  # key -> why it is never granted

    @property
    def key_names(self) -> tuple[str, ...]:
        """The lease keys, in README order."""
        return tuple(key for key, _ in self.keys)

    @property
    def buses(self) -> tuple[str, ...]:
        """Every GPU bus of the layout."""
        return tuple(bus for _, buses in self.keys for bus in buses)

    def key_of(self, bus: str) -> str | None:
        """Return the lease key that covers ``bus``, None when no key does."""
        for key, buses in self.keys:
            if bus in buses:
                return key
        return None

    def key_buses(self, key: str) -> tuple[str, ...]:
        """Return the buses ``key`` covers (key matched case-insensitively)."""
        for name, buses in self.keys:
            if name.lower() == key.lower():
                return buses
        return ()

    def lead_only_reason(self, key: str) -> str | None:
        """Return why ``key`` is never granted without the lead, else None."""
        return dict(self.lead_only).get(key)


INTERIM = Layout(
    "six V620 behind one switch (interim, since 2026-11-08)",
    keys=(("a2", ("d1", "d4")), ("b4", ("d9", "dc", "df", "e2"))),
    slot_keys=(("A", "a2"), ("B", "b4")),
)

TEN_GPU = Layout(
    "ten GPUs behind two switches (2026-11-05)",
    keys=(
        ("slotA", ("ce", "d3", "d8", "db")),
        ("slotB", ("0a", "0f", "14", "17")),
        ("w7800cb", ("cb",)),
        ("w7800-07", ("07",)),
    ),
    slot_keys=(("A", "slotA"), ("B", "slotB")),
    lead_only=(("w7800-07", "W7800 bus 07 (HIP 5): NEVER without the lead"),),
)

LAYOUTS: tuple[Layout, ...] = (INTERIM, TEN_GPU)  # newest first

# Nothing is covered: the layout of a box whose GPU buses match no documented one.
NO_LAYOUT = Layout("no documented layout matches the GPUs of this box", (), ())


def all_keys() -> tuple[str, ...]:
    """Return every lease key of every documented layout (no duplicates)."""
    return tuple(dict.fromkeys(k for layout in LAYOUTS for k in layout.key_names))


def layouts_knowing(buses: Iterable[str]) -> tuple[Layout, ...]:
    """Return the documented layouts that know at least one of ``buses``."""
    wanted = set(buses)
    return tuple(layout for layout in LAYOUTS if wanted & set(layout.buses))


def layout_of_buses(buses: Iterable[str]) -> Layout | None:
    """Return the one documented layout that knows some of ``buses``, else None.

    None when no layout knows any of them or when they belong to several (the
    lease check then refuses the lot). It is the fallback for callers without a
    topology; the topology's own buses decide through :func:`layout_of_box`.
    """
    found = layouts_knowing(buses)
    return found[0] if len(found) == 1 else None


def layout_of_box(box_buses: Iterable[str]) -> Layout:
    """Return the documented layout whose GPU buses are exactly ``box_buses``.

    ``NO_LAYOUT`` when none is: the lease check then refuses every GPU, which
    is the box-rules instruction to re-derive this table before any probe.
    """
    wanted = set(box_buses)
    for layout in LAYOUTS:
        if wanted == set(layout.buses):
            return layout
    return NO_LAYOUT
