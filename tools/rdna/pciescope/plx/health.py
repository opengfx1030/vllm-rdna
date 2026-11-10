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

from collections.abc import Collection, Iterable, Mapping
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

#: link-state fields of a port that move by themselves when its device goes through
#: runtime power management (asleep: 2.5 GT/s x1 and degraded, awake: full speed);
#: the error registers are never among them
PM_LINK_FIELDS = frozenset(
    {
        "speed_gts",
        "width",
        "link_up",
        "training",
        "bw_mgmt_status",
        "autonomous_bw_status",
        "degraded",
        "eq_complete",
    }
)


def _fmt(key: str, value: Any) -> str:
    if key in HEX_FIELDS and isinstance(value, int):
        return f"{value:#x}"
    return str(value)


def _changed(
    bdf: str, a: Mapping[str, Any], b: Mapping[str, Any], keys: Iterable[str]
) -> list[str]:
    return [
        f"port {bdf} {key}: {_fmt(key, a.get(key))} -> {_fmt(key, b.get(key))}"
        for key in sorted(keys)
        if a.get(key) != b.get(key)
    ]


def diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    pm_bdfs: Collection[str] = (),
) -> list[str]:
    """Human-readable differences between two ``capture`` results (empty = same).

    The link-state fields (``PM_LINK_FIELDS``) of the ports in ``pm_bdfs`` are left
    out: a device in runtime power management changes them on its own, ``pm_diff``
    lists them.  Every other field of those ports is still compared.
    """
    out: list[str] = []
    for bdf in sorted(set(before["ports"]) | set(after["ports"])):
        a, b = before["ports"].get(bdf), after["ports"].get(bdf)
        if a is None or b is None:
            out.append(f"port {bdf}: {'appeared' if a is None else 'vanished'}")
            continue
        keys = set(a) | set(b)
        out += _changed(bdf, a, b, keys - PM_LINK_FIELDS if bdf in pm_bdfs else keys)
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


def pm_diff(
    before: Mapping[str, Any], after: Mapping[str, Any], pm_bdfs: Collection[str]
) -> list[str]:
    """The link-state differences of the runtime-PM ports that ``diff`` leaves out."""
    out: list[str] = []
    for bdf in sorted(pm_bdfs):
        a, b = before["ports"].get(bdf), after["ports"].get(bdf)
        if a is not None and b is not None:
            out += _changed(bdf, a, b, PM_LINK_FIELDS)
    return out
