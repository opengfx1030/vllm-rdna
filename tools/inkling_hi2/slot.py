#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot / HIP -> PCI bus gate for the Inkling-2 HI2 kit.

The box has six V620s in two leases (AGENTS.md, "Two-slot box operation
2026-11-15"):

    a2  HIP 0,1      PCI d1 d4
    b4  HIP 2,3,4,5  PCI d9 dc df e2

A HIP index is enumeration order and can move; the PCI bus does not.  The launchers
resolve the requested HIP indices to their live PCI buses before every launch and
refuse a mismatch (AGENTS.md rule 1).  The HTTP prediction kit (``kl_probe.py``,
``bench.py``) cannot see the GPUs and only checks that the declared
``SLOT``/``HIP_VISIBLE_DEVICES``/``EXPECT_PCI_BUSES`` agree with the documented
table, so a moved enumeration is caught by the launcher, not by the client.

No import-time torch: the declared check is CPU-only, and :func:`physical_table`
imports torch lazily after clearing the visibility variables so it always sees the
physical enumeration.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

SLOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "a2": {"hip": ("0", "1"), "buses": ("d1", "d4")},
    "b4": {"hip": ("2", "3", "4", "5"), "buses": ("d9", "dc", "df", "e2")},
}
DEFAULT_SLOT = "a2"
ENV_SLOT = "SLOT"
ENV_HIP = "HIP_VISIBLE_DEVICES"
ENV_BUSES = "EXPECT_PCI_BUSES"
VISIBILITY_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


class SlotError(RuntimeError):
    """The declared or observed device set is not the slot's V620s."""


def normalise_bus(value: str | int) -> str:
    """``d1`` / ``0xd1`` / ``0000:d1:00.0`` / ``0xD1`` -> ``d1``.

    A string is hex (the form operators and ``lspci`` print); an ``int`` is the
    value torch returns (``pci_bus_id`` is an integer), so it is used as-is.
    """
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip().lower()
        if ":" in text:
            parts = text.split(":")
            text = parts[-2] if len(parts) >= 2 and "." in parts[-1] else parts[-1]
        text = text.removeprefix("0x")
        try:
            number = int(text, 16)
        except ValueError as exc:
            raise SlotError(f"not a PCI bus number: {value!r}") from exc
    if not 0 <= number <= 0xFF:
        raise SlotError(f"PCI bus out of range: {value!r}")
    return f"{number:02x}"


def parse_hip_list(text: str) -> tuple[int, ...]:
    items = [item for item in text.replace(",", " ").split() if item]
    if not items:
        raise SlotError("HIP_VISIBLE_DEVICES is empty")
    try:
        hips = tuple(int(item) for item in items)
    except ValueError as exc:
        raise SlotError(
            f"HIP_VISIBLE_DEVICES={text!r} is not a list of indexes"
        ) from exc
    if any(hip < 0 for hip in hips):
        raise SlotError(f"HIP_VISIBLE_DEVICES={text!r} has a negative index")
    if len(set(hips)) != len(hips):
        raise SlotError(f"HIP_VISIBLE_DEVICES={text!r} repeats an index")
    return hips


def parse_bus_list(text: str) -> tuple[str, ...]:
    buses = tuple(
        normalise_bus(item) for item in text.replace(",", " ").split() if item
    )
    if not buses:
        raise SlotError("EXPECT_PCI_BUSES is empty")
    if len(set(buses)) != len(buses):
        raise SlotError(f"EXPECT_PCI_BUSES={text!r} repeats a bus")
    return buses


def slot_spec(slot: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    try:
        spec = SLOTS[slot]
    except KeyError as exc:
        raise SlotError(
            f"unknown SLOT {slot!r} (known: {', '.join(sorted(SLOTS))})"
        ) from exc
    return tuple(int(hip) for hip in spec["hip"]), tuple(spec["buses"])


def default_pair(slot: str) -> tuple[str, str]:
    """``(HIP_VISIBLE_DEVICES, EXPECT_PCI_BUSES)`` of a slot, as env strings."""
    hips, buses = slot_spec(slot)
    return ",".join(str(hip) for hip in hips), ",".join(buses)


def env_pair(slot: str) -> tuple[str, str]:
    """The declared pair: env when set, else the slot's documented pair."""
    default_hip, default_buses = default_pair(slot)
    return (
        os.environ.get(ENV_HIP) or default_hip,
        os.environ.get(ENV_BUSES) or default_buses,
    )


def check_declared(slot: str, hip: str, buses: str) -> None:
    """Refuse a declared device set that is not the slot's documented table entry."""
    want_hips, want_buses = slot_spec(slot)
    got_hips = parse_hip_list(hip)
    got_buses = parse_bus_list(buses)
    if got_hips != want_hips:
        raise SlotError(
            f"SLOT={slot} is HIP {','.join(map(str, want_hips))}; declared HIP {hip}"
        )
    if got_buses != want_buses:
        raise SlotError(
            f"SLOT={slot} is PCI {','.join(want_buses)}; declared PCI {buses}"
        )


def check_physical(
    hip: str,
    buses: str,
    table: Mapping[int, tuple[str, str]],
) -> None:
    """Refuse unless every declared HIP index is the expected bus and a V620.

    ``table`` maps a physical HIP index to ``(pci_bus, device_name)``; a CPU test
    passes a fake table, :func:`physical_table` builds the live one.
    """
    hips = parse_hip_list(hip)
    expected = parse_bus_list(buses)
    if len(hips) != len(expected):
        raise SlotError(
            f"HIP_VISIBLE_DEVICES lists {len(hips)} GPUs but EXPECT_PCI_BUSES "
            f"lists {len(expected)} buses"
        )
    missing = [index for index in hips if index not in table]
    if missing:
        raise SlotError(
            f"no GPU at HIP {missing}; present HIP indexes: "
            f"{', '.join(str(index) for index in sorted(table))}"
        )
    observed = tuple(normalise_bus(table[index][0]) for index in hips)
    if observed != expected:
        raise SlotError(
            f"HIP {','.join(map(str, hips))} sits on PCI {','.join(observed)}, "
            f"not the expected {','.join(expected)} (AGENTS two-slot table)"
        )
    foreign = [
        (index, table[index][1]) for index in hips if "V620" not in table[index][1]
    ]
    if foreign:
        raise SlotError(f"not V620 devices: {foreign}")


def physical_table() -> dict[int, tuple[str, str]]:
    """``{physical HIP: (pci_bus, name)}`` with the visibility vars cleared."""
    saved = {
        name: os.environ.pop(name) for name in VISIBILITY_VARS if name in os.environ
    }
    try:
        import torch
        import torch.accelerator

        torch.cuda.init()
        return {
            index: (
                normalise_bus(torch.cuda.get_device_properties(index).pci_bus_id),
                torch.cuda.get_device_properties(index).name,
            )
            for index in range(torch.accelerator.device_count())
        }
    except SlotError:
        raise
    except Exception as exc:  # torch missing, no GPU, driver error
        raise SlotError(f"cannot enumerate the GPUs: {exc}") from exc
    finally:
        os.environ.update(saved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", default=os.environ.get(ENV_SLOT) or DEFAULT_SLOT)
    parser.add_argument(
        "--hip", default=None, help=f"default: ${ENV_HIP} or the slot's"
    )
    parser.add_argument(
        "--expect-buses", default=None, help=f"default: ${ENV_BUSES} or the slot's"
    )
    parser.add_argument(
        "--print", action="store_true", dest="print_env", help="print the env pair only"
    )
    parser.add_argument(
        "--declared-only",
        action="store_true",
        help="skip the live torch enumeration (HTTP clients, CPU hosts)",
    )
    args = parser.parse_args(argv)

    try:
        default_hip, default_buses = default_pair(args.slot)
        hip = args.hip or os.environ.get(ENV_HIP) or default_hip
        buses = args.expect_buses or os.environ.get(ENV_BUSES) or default_buses
        check_declared(args.slot, hip, buses)
        if args.print_env:
            print(f"{ENV_HIP}={hip}")
            print(f"{ENV_BUSES}={buses}")
            return 0
        if args.declared_only:
            print(f"slot.py: SLOT={args.slot} HIP {hip} -> PCI {buses} (declared)")
            return 0
        table = physical_table()
        check_physical(hip, buses, table)
        names = {table[index][1] for index in parse_hip_list(hip)}
        print(
            f"slot.py: SLOT={args.slot} HIP {hip} -> PCI {buses} "
            f"({', '.join(sorted(names))})"
        )
        return 0
    except SlotError as exc:
        print(f"slot.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
