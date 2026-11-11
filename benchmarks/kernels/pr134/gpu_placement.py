# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve which physical GPUs a serving arm actually used, and assert it.

The campaign pins HIP_VISIBLE_DEVICES, but rocm-smi's card index does NOT equal
the HIP ordinal: rocm-smi sorts by PCI bus address, while HIP follows the KFD
node order. On this box the two W7800s sit at KFD nodes 1 and 2, so they are
HIP 0 and 1 while rocm-smi calls them card8 and card9. Reading the wrong index
would file the placement evidence on the wrong GPUs -- and HIP 0/1 belong to
another workload.

This resolves, for every card: KFD node -> HIP ordinal -> PCI bus -> PCIe root
port, plus the VRAM actually held, and asserts the served set.

Usage: python gpu_placement.py [--expect 6,7,8,9] [-o placement.json]
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys

ROCM_SMI = "rocm-smi"
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"

# A serving worker holds tens of GiB per rank. Anything above this is a
# workload on the card; the ~30 MiB an idle card shows is driver overhead.
TOUCHED_BYTES = 1 << 30


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def smi_csv(*args: str) -> list[dict[str, str]]:
    out = run([ROCM_SMI, *args, "--csv"])
    if not out.strip():
        return []
    return list(csv.DictReader(io.StringIO(out)))


def resolve_hip_ordinals(cards: list[dict[str, str]]) -> dict[str, int]:
    """Map a rocm-smi GUID to its HIP ordinal via the KFD node order.

    KFD node 0 is the CPU; the GPU nodes follow in the order HIP assigns
    device ordinals. The GUID is the join key between the two views.
    """
    guid_to_node: dict[str, int] = {}
    if os.path.isdir(KFD_NODES):
        for name in os.listdir(KFD_NODES):
            try:
                with open(os.path.join(KFD_NODES, name, "gpu_id")) as handle:
                    gpu_id = handle.read().strip()
            except OSError:
                continue
            if gpu_id and gpu_id != "0":
                guid_to_node[gpu_id] = int(name) - 1  # node 0 is the CPU
    ordinals: dict[str, int] = {}
    for card in cards:
        guid = (card.get("GUID") or "").strip()
        if guid in guid_to_node:
            ordinals[guid] = guid_to_node[guid]
    return ordinals


def is_pci_address(name: str) -> bool:
    """True for a sysfs PCI address like ``0000:0a:00.0``."""
    parts = name.split(":")
    if len(parts) != 3:
        return False
    bus, device, function = parts
    # bus is 4 hex digits, device 2, function is "<2 hex>.<1 hex>"
    if len(bus) != 4 or len(device) != 2 or len(function) != 4:
        return False
    if function[2] != ".":
        return False
    digits = bus + device + function[0:2] + function[3]
    return all(char in "0123456789abcdef" for char in digits)


def pci_root_for(slot: str) -> str | None:
    """Find the PCIe root port a GPU hangs off, by walking sysfs upwards.

    The chain ends at the root complex (``pci0000:NN``); the element just below
    it is the root port, e.g. ``0000:00:01.1`` for the serving group and
    ``0000:40:03.1`` for the group that shows corrected-link PCIe events.
    """
    path = f"/sys/bus/pci/devices/0000:{slot}"
    if not os.path.exists(path):
        return None
    chain: list[str] = []
    current = os.path.realpath(path)
    for _ in range(12):
        parent = os.path.realpath(os.path.join(current, ".."))
        if not parent or parent == current:
            break
        chain.append(os.path.basename(parent))
        current = parent
    devices = [name for name in chain if is_pci_address(name)]
    return devices[-1].split(":", 1)[1] if devices else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        default="6,7,8,9",
        help="comma-separated HIP ordinals the serving arm must have used",
    )
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    expected = {int(x) for x in args.expect.split(",") if x.strip()}

    cards = smi_csv("--showproductname", "--showbus")
    vram = {
        (row.get("device") or "").strip(): row
        for row in smi_csv("--showmeminfo", "vram")
    }
    ordinals = resolve_hip_ordinals(cards)
    resolved = []
    for card in cards:
        dev = (card.get("device") or "").strip()
        bus = (card.get("PCI Bus") or "").strip()
        guid = (card.get("GUID") or "").strip()
        # "0000:07:00.0" -> "07:00.0" (the sysfs address under the root bus).
        # rocm-smi prints uppercase hex ("0000:0A:00.0"); sysfs is lowercase.
        parts = bus.lower().split(":")
        slot = f"{parts[1]}:{parts[2]}" if len(parts) == 3 else ""
        used = 0
        row = vram.get(dev, {})
        try:
            used = int(row.get("VRAM Total Used Memory (B)") or 0)
        except ValueError:
            used = 0
        resolved.append(
            {
                "smi_card": dev,
                "pci_bus": bus,
                "pci_slot": slot,
                "product": (card.get("Card Series") or "").strip(),
                "gfx": (card.get("GFX Version") or "").strip(),
                "guid": guid,
                "hip_ordinal": ordinals.get(guid),
                "pcie_root": pci_root_for(slot) if slot else None,
                "vram_used_bytes": used,
                "vram_used_gib": round(used / 2**30, 2),
            }
        )

    served = [r for r in resolved if r["vram_used_bytes"] > TOUCHED_BYTES]
    served_ordinals = sorted(
        r["hip_ordinal"] for r in served if r["hip_ordinal"] is not None
    )
    touched_ordinals = sorted(
        r["hip_ordinal"]
        for r in resolved
        if r["hip_ordinal"] is not None and r["vram_used_bytes"] > TOUCHED_BYTES
    )
    roots = sorted({r["pcie_root"] for r in served if r["pcie_root"]})

    report = {
        "expected_hip_ordinals": sorted(expected),
        "served_hip_ordinals": served_ordinals,
        "placement_matches": served_ordinals == sorted(expected),
        "touched_hip_ordinals": touched_ordinals,
        "foreign_workload_untouched": all(
            r["vram_used_bytes"] <= TOUCHED_BYTES
            for r in resolved
            if r["product"] != "AMD Radeon Pro V620"
        ),
        "served_pcie_roots": roots,
        "cards": resolved,
    }

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)

    if not report["placement_matches"]:
        print(
            f"FAIL: served HIP ordinals {served_ordinals} != expected "
            f"{sorted(expected)}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: serving confined to HIP {served_ordinals} on root(s) {roots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
