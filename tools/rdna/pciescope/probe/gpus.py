# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn the user's ``--gpus`` value into buses of the discovered topology.

Accepted tokens (comma or space separated, case-insensitive):

* a PCI bus (``0a``, ``gpu:0a``, ``0000:0a:00.0``): two hex digits, so ``14``
  is bus 0x14, never HIP 14; ``gpu:<bus>`` always means the bus;
* one digit ``0``..``9`` or ``hip:N`` / ``hipN``: a HIP index (the HIP order of
  ``Topology.gpus[..].hip``);
* ``A``/``B``/``slotA``/``slotB``: the V620 of that lease slot letter
  (``Topology.slot_gpus``: a switch until 2026-11-08, a leaf of the single
  switch since);
* a lease key of the layout of the box (``hw.py``): ``b4`` (HIP 2-5) and ``a2``
  (HIP 0-1) on the interim box, ``w7800cb`` before. ``a2`` and ``b4`` are also
  two hex digits: a box with a GPU on such a bus refuses the bare token as
  ambiguous, write ``gpu:a2`` for the bus;
* ``all-v620``: both slots.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pciescope.model import Gpu, Topology

from .hw import layout_of_box
from .ids import normalize_bus


def split_tokens(value: str | Iterable[str]) -> list[str]:
    """Split a ``--gpus`` value (string or already split) into tokens."""
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        out += [t for t in item.replace(",", " ").split() if t]
    return out


def _slot_buses(topo: Topology, slot: str) -> list[str]:
    return [g.bus for g in topo.slot_gpus(slot)]


def resolve_gpus(value: str | Iterable[str], topo: Topology) -> list[str]:
    """Return the buses (order kept, no duplicates) named by ``value``.

    Raises:
        ValueError: for an empty selection, an unknown token or a bus without a
            GPU in ``topo``.

    """
    buses: list[str] = []

    def add(bus: str) -> None:
        if bus not in buses:
            buses.append(bus)

    layout = layout_of_box(g.bus for g in topo.gpus.values())
    for tok in split_tokens(value):
        low = tok.lower()
        if low in ("a", "b", "slota", "slotb"):
            found = _slot_buses(topo, low[-1])
            if not found:
                raise ValueError(f"slot {low[-1].upper()} has no GPU in the topology")
            for bus in found:
                add(bus)
        elif low == "all-v620":
            for slot in ("A", "B"):
                for bus in _slot_buses(topo, slot):
                    add(bus)
        elif not low.startswith("gpu:") and layout.key_buses(low):
            if any(g.bus == low for g in topo.gpus.values()):
                raise ValueError(
                    f"{tok!r} is both a lease key and the bus of a GPU: write "
                    f"gpu:{low} for the bus"
                )
            for bus in layout.key_buses(low):
                add(_gpu_bus(topo, bus))
        elif low.startswith("hip") or (len(low) == 1 and low.isdigit()):
            digits = low.removeprefix("hip").removeprefix(":")
            if not digits.isdigit():
                raise ValueError(f"bad HIP index {tok!r}")
            add(_hip_bus(topo, int(digits)))
        else:
            add(_gpu_bus(topo, normalize_bus(low)))
    if not buses:
        raise ValueError("no GPU selected (--gpus is empty)")
    return buses


def _gpu_bus(topo: Topology, bus: str) -> str:
    try:
        return topo.gpu(bus).bus
    except KeyError:
        known = " ".join(sorted(g.bus for g in topo.gpus.values()))
        raise ValueError(f"no GPU on bus {bus} (known: {known})") from None


def _hip_bus(topo: Topology, index: int) -> str:
    try:
        return topo.gpu(index).bus
    except KeyError:
        raise ValueError(f"no GPU with HIP index {index}") from None


def gpu_records(buses: Sequence[str], topo: Topology) -> list[Gpu]:
    """Topology GPUs for ``buses`` in HIP order (the probe's logical order)."""
    gpus = [topo.gpu(b) for b in buses]
    missing = [g.bus for g in gpus if g.hip is None]
    if missing:
        raise ValueError(
            f"HIP index unknown for bus {' '.join(missing)} "
            f"(topology hip_source={topo.hip_source})"
        )
    return sorted(gpus, key=lambda g: g.hip if g.hip is not None else -1)
