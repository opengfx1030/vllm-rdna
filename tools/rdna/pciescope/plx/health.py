# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health of a switch before and after a counter run, and what changed.

``capture`` records, for every port function of one switch, the decoded link state
(speed, width, training, bandwidth-management flags, equalization), the sticky
error registers (DevSta, AER uncorrectable/correctable, lane errors, DPC) and the
port settings, plus the kernel's AER counters (``aer_dev_*`` sysfs files) of every
function below the switch (ports, on-card bridges, GPUs).  ``diff`` lists every
difference; a run that changed anything is an anomaly.  Read-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pciescope.model import Topology, full_bdf
from pciescope.sysfs import Sysfs

AER_FILES = ("aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal")


def _counters(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (text or "").splitlines():
        name, _, value = line.rpartition(" ")
        if name and value.isdigit():
            out[name] = int(value)
    return out


def kernel_aer(
    topo: Topology, switch: str, sysfs: Sysfs | None = None
) -> dict[str, Any]:
    """``{bdf: {file: {counter: n}}}`` of every function below ``switch``."""
    sysfs = sysfs or Sysfs()
    gpus = {g.id for g in topo.gpus.values() if g.switch == switch}
    out: dict[str, Any] = {}
    for node in topo.nodes.values():
        if node.switch != switch and node.gpu not in gpus:
            continue
        files = {}
        for name in AER_FILES:
            text = sysfs.read(f"bus/pci/devices/{full_bdf(node.bdf)}/{name}")
            if text is not None:
                files[name] = _counters(text)
        if files:
            out[node.bdf] = files
    return out


def capture(
    snapshot: Mapping[str, Any],
    topo: Topology,
    switch: str,
    sysfs: Sysfs | None = None,
) -> dict[str, Any]:
    """State of ``switch`` from a collector ``snapshot()`` plus kernel counters."""
    ports = {
        p["bdf"]: dict(p["state"])
        for p in snapshot["ports"]
        if p["switch"] == switch and p["state"] is not None
    }
    return {"ports": ports, "kernel": kernel_aer(topo, switch, sysfs)}


#: state fields that are register contents (shown in hex); the rest is decimal
HEX_FIELDS = frozenset(
    {"dev_status", "uncorr_status", "corr_status", "lane_err", "dpc_status", "acs_ctl"}
)


def _fmt(key: str, value: Any) -> str:
    if key in HEX_FIELDS and isinstance(value, int):
        return f"{value:#x}"
    return str(value)


def diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Human-readable differences between two ``capture`` results (empty = same)."""
    out: list[str] = []
    for bdf in sorted(set(before["ports"]) | set(after["ports"])):
        a, b = before["ports"].get(bdf), after["ports"].get(bdf)
        if a is None or b is None:
            out.append(f"port {bdf}: {'appeared' if a is None else 'vanished'}")
            continue
        out += [
            f"port {bdf} {key}: {_fmt(key, a.get(key))} -> {_fmt(key, b.get(key))}"
            for key in sorted(set(a) | set(b))
            if a.get(key) != b.get(key)
        ]
    for bdf in sorted(set(before["kernel"]) | set(after["kernel"])):
        a_files = before["kernel"].get(bdf, {})
        b_files = after["kernel"].get(bdf, {})
        for name in sorted(set(a_files) | set(b_files)):
            a_cnt, b_cnt = a_files.get(name, {}), b_files.get(name, {})
            out += [
                f"kernel {bdf} {name} {key}: {a_cnt.get(key, 0)} -> {b_cnt.get(key, 0)}"
                for key in sorted(set(a_cnt) | set(b_cnt))
                if a_cnt.get(key) != b_cnt.get(key)
            ]
    return out
