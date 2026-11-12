# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoders for the standard PCIe registers of a PEX port window.

Everything works on a ``get(offset) -> dword`` callable: a live BAR0 window read
through the root helper, or a recorded fixture.  The PEX88096 port windows mirror
the PCI configuration space (header, capability list at 0x34, extended
capabilities from 0x100) so the same walk that ``lspci`` does applies [BOX].
Bit positions follow the PCI Express Base Specification (``[SPEC]``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pciescope.plx import regs

Get = Callable[[int], int]

#: PCIe capability "Device/Port Type" field
PORT_TYPE_NAMES = {
    0: "endpoint",
    1: "legacy_endpoint",
    4: "root_port",
    5: "upstream",
    6: "downstream",
    7: "pcie_to_pci_bridge",
    8: "pci_to_pcie_bridge",
    9: "rc_endpoint",
    10: "rc_event_collector",
}

#: AER uncorrectable error status bits (the kernel's aer_dev_* names, except that
#: two abbreviated names are spelled out)
UNCORR_BITS = {
    4: "DLP",
    5: "SDES",
    12: "TLP",
    13: "FCP",
    14: "CmpltTO",
    15: "CmpltAbrt",
    16: "UnexpectedCmplt",
    17: "RxOF",
    18: "MalfTLP",
    19: "ECRC",
    20: "UnsupReq",
    21: "ACSViol",
    22: "UncorrIntErr",
    23: "BlockedTLP",
    24: "AtomicOpBlocked",
    25: "TLPBlockedErr",
    26: "PoisonTLPBlocked",
    27: "DMWrReqBlocked",
    28: "IDECheck",
    29: "MisroutedIDETLP",
    30: "PCRC_CHECK",
    31: "TLPXlatBlocked",
}
#: AER correctable error status bits
CORR_BITS = {
    0: "RxErr",
    6: "BadTLP",
    7: "BadDLLP",
    8: "Rollover",
    12: "Timeout",
    13: "NonFatalErr",
    14: "CorrIntErr",
    15: "HeaderOF",
}
#: DevSta error flags (bit -> name)
DEVSTA_BITS = {0: "CorrErr", 1: "NonFatalErr", 2: "FatalErr", 3: "UnsupReq"}

#: ACS control bits
ACS_BITS = {
    0: "SrcValid",
    1: "TransBlk",
    2: "ReqRedir",
    3: "CmpltRedir",
    4: "UpstreamFwd",
    5: "EgressCtrl",
    6: "DirectTrans",
}


def bit_names(mask: int, table: dict[int, str]) -> list[str]:
    """Names of the bits of ``mask`` that are set and known to ``table``."""
    return [name for bit, name in table.items() if mask >> bit & 1]


_STD_NAMES = {regs.CAP_PM: "pm", regs.CAP_MSI: "msi", regs.CAP_PCIE: "pcie"}


@dataclass(frozen=True, slots=True)
class CapMap:
    """Capability offsets of one function: standard ids and extended ids."""

    std: dict[int, int] = field(default_factory=dict)
    ext: dict[int, int] = field(default_factory=dict)

    def names(self) -> dict[str, int]:
        """``{name: offset}``; unknown ids appear as ``std_xx`` / ``ext_xxxx``."""
        out: dict[str, int] = {}
        for cid, off in self.std.items():
            out.setdefault(_STD_NAMES.get(cid, f"std_{cid:02x}"), off)
        for cid, off in self.ext.items():
            out.setdefault(regs.ECAP_NAMES.get(cid, f"ext_{cid:04x}"), off)
        return out


def walk_caps(get: Get) -> CapMap:
    """Walk the standard list (0x34) and the extended list (0x100) of a window."""
    std: dict[int, int] = {}
    if (get(0x04) >> 16) & 0x10:  # status register: capabilities list present
        ptr = get(0x34) & 0xFC
        seen: set[int] = set()
        while 0x40 <= ptr < 0x100 and ptr not in seen and len(seen) < 48:
            seen.add(ptr)
            hdr = get(ptr)
            std.setdefault(hdr & 0xFF, ptr)
            ptr = (hdr >> 8) & 0xFC
    ext: dict[int, int] = {}
    off = 0x100
    seen_ext: set[int] = set()
    while 0x100 <= off < 0x1000 and off not in seen_ext and len(seen_ext) < 96:
        seen_ext.add(off)
        hdr = get(off)
        if hdr in (0, 0xFFFFFFFF):
            break
        ext.setdefault(hdr & 0xFFFF, off)
        off = (hdr >> 20) & 0xFFC
    return CapMap(std, ext)


@dataclass(frozen=True, slots=True)
class PortStatic:
    """Facts about a port that do not change while it runs."""

    vendor: int
    device: int
    pci_class: int  # base class << 16 | sub << 8 | prog-if
    header_type: int
    primary_bus: int
    secondary_bus: int
    subordinate_bus: int
    caps: CapMap
    port_type: int | None  # PCIe capability port type, None without the capability
    pcie_version: int
    port_number: int | None  # LnkCap[31:24]: the PEX port number
    max_speed_gts: float
    max_width: int
    dl_active_reporting: bool
    mps_supported: int  # bytes
    aspm_support: int  # LnkCap[11:10]
    serial: str | None  # Device Serial Number capability, "xx-xx-..." like lspci

    @property
    def is_bridge(self) -> bool:
        return self.header_type & 0x7F == 1

    @property
    def port_type_name(self) -> str:
        if self.port_type is None:
            return "none"
        return PORT_TYPE_NAMES.get(self.port_type, "none")


def _serial(get: Get, caps: CapMap) -> str | None:
    """Device Serial Number as lspci prints it (upper dword first)."""
    off = caps.ext.get(regs.ECAP_DSN)
    if off is None:
        return None
    lo, hi = get(off + 4), get(off + 8)
    if not lo and not hi:
        return None
    return "-".join(f"{b:02x}" for b in (hi.to_bytes(4, "big") + lo.to_bytes(4, "big")))


def decode_static(get: Get) -> PortStatic:
    """Decode the unchanging part of a port window."""
    caps = walk_caps(get)
    ids = get(0x00)
    cls_rev = get(0x08)
    hdr_type = (get(0x0C) >> 16) & 0xFF
    bus = get(0x18)
    pcie = caps.std.get(regs.CAP_PCIE)
    port_type = None
    version = 0
    port_number = None
    max_speed = 0.0
    max_width = 0
    dl_rep = False
    mps_sup = 0
    aspm = 0
    if pcie is not None:
        cap_reg = get(pcie) >> 16
        version = cap_reg & 0xF
        port_type = (cap_reg >> 4) & 0xF
        lnk_cap = get(pcie + 0x0C)
        max_speed = regs.SPEED_GTS.get(lnk_cap & 0xF, 0.0)
        max_width = (lnk_cap >> 4) & 0x3F
        aspm = (lnk_cap >> 10) & 0x3
        dl_rep = bool(lnk_cap >> 20 & 1)
        port_number = (lnk_cap >> 24) & 0xFF
        mps_sup = 128 << (get(pcie + 0x04) & 0x7)
    return PortStatic(
        vendor=ids & 0xFFFF,
        device=ids >> 16,
        pci_class=cls_rev >> 8,
        header_type=hdr_type,
        primary_bus=bus & 0xFF,
        secondary_bus=(bus >> 8) & 0xFF,
        subordinate_bus=(bus >> 16) & 0xFF,
        caps=caps,
        port_type=port_type,
        pcie_version=version,
        port_number=port_number,
        max_speed_gts=max_speed,
        max_width=max_width,
        dl_active_reporting=dl_rep,
        mps_supported=mps_sup,
        aspm_support=aspm,
        serial=_serial(get, caps),
    )


# dynamic registers, in the order they are read every sample: name -> (capability,
# offset inside it); the pcie_version gate applies to the "lnk2" dword
_DYN_LAYOUT: tuple[tuple[str, str, int], ...] = (
    ("dev", "pcie", 0x08),  # DevCtl | DevSta << 16
    ("lnk", "pcie", 0x10),  # LnkCtl | LnkSta << 16
    ("lnk2", "pcie", 0x30),  # LnkCtl2 | LnkSta2 << 16
    ("uncorr", "aer", 0x04),  # uncorrectable status (sticky)
    ("corr", "aer", 0x10),  # correctable status (sticky)
    ("dpc", "dpc", 0x04),  # DPC capability | DPC control << 16
    ("dpc_sta", "dpc", 0x08),  # DPC status | error source id << 16
    ("lane_err", "secondary_pcie", 0x08),  # lane error status
    ("phy16", "phy_16gt", 0x0C),  # 16 GT/s status (equalization)
    ("acs", "acs", 0x04),  # ACS capability | ACS control << 16
)


def dynamic_offsets(st: PortStatic) -> tuple[tuple[str, int], ...]:
    """``((name, window offset), ...)`` of the registers sampled every collect."""
    names = st.caps.names()
    out: list[tuple[str, int]] = []
    for key, cap, rel in _DYN_LAYOUT:
        base = names.get(cap)
        if base is None:
            continue
        if key == "lnk2" and st.pcie_version < 2:
            continue
        out.append((key, base + rel))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class LinkState:
    """Decoded dynamic state of one port (all booleans are plain ``bool``)."""

    speed_gts: float
    width: int
    link_up: bool
    training: bool
    bw_mgmt_status: bool  # LBMS (sticky)
    autonomous_bw_status: bool  # LABS (sticky)
    target_speed_gts: float
    max_speed_gts: float
    max_width: int
    degraded: bool
    eq_complete: bool | None  # at the current speed (None below 8 GT/s)
    dev_status: int  # DevSta[3:0]: CE=1 NF=2 F=4 UR=8 (sticky)
    uncorr_status: int | None
    corr_status: int | None
    lane_err: int | None
    dpc_status: int | None  # DPC status register (0 = never triggered)
    mps: int
    mrrs: int
    aspm: int  # LnkCtl[1:0]
    acs_ctl: int | None


def decode_dynamic(st: PortStatic, vals: dict[str, int]) -> LinkState:
    """Decode one sample (``vals`` = ``{name: dword}`` from ``dynamic_offsets``)."""
    dev = vals.get("dev", 0)
    lnk = vals.get("lnk", 0)
    lnk2 = vals.get("lnk2", 0)
    dev_ctl, dev_sta = dev & 0xFFFF, dev >> 16
    lnk_ctl, lnk_sta = lnk & 0xFFFF, lnk >> 16
    speed = regs.SPEED_GTS.get(lnk_sta & 0xF, 0.0)
    width = (lnk_sta >> 4) & 0x3F
    training = bool(lnk_sta >> 11 & 1)
    dl_active = bool(lnk_sta >> 13 & 1)
    # an upstream port or a function that cannot report DL_Active: fall back to the
    # negotiated width (0 = no link)
    link_up = dl_active if st.dl_active_reporting else width > 0
    target = regs.SPEED_GTS.get((lnk2 & 0xFFFF) & 0xF, 0.0)
    eq: bool | None = None
    if speed >= 16.0 and "phy16" in vals:
        eq = bool(vals["phy16"] & 1)
    elif speed >= 8.0 and "lnk2" in vals:
        eq = bool((lnk2 >> 16) >> 1 & 1)
    degraded = link_up and (
        (st.max_speed_gts > 0 and speed < st.max_speed_gts)
        or (st.max_width > 0 and width < st.max_width)
    )
    dpc = vals.get("dpc_sta")
    acs = vals.get("acs")
    return LinkState(
        speed_gts=speed,
        width=width,
        link_up=link_up,
        training=training,
        bw_mgmt_status=bool(lnk_sta >> 14 & 1),
        autonomous_bw_status=bool(lnk_sta >> 15 & 1),
        target_speed_gts=target,
        max_speed_gts=st.max_speed_gts,
        max_width=st.max_width,
        degraded=degraded,
        eq_complete=eq,
        dev_status=dev_sta & 0xF,
        uncorr_status=vals.get("uncorr"),
        corr_status=vals.get("corr"),
        lane_err=vals.get("lane_err"),
        dpc_status=None if dpc is None else dpc & 0xFFFF,
        mps=128 << ((dev_ctl >> 5) & 0x7),
        mrrs=128 << ((dev_ctl >> 12) & 0x7),
        aspm=lnk_ctl & 0x3,
        acs_ctl=None if acs is None else (acs >> 16) & 0x7F,
    )


def dpc_reason(status: int) -> str:
    """Human readable DPC trigger reason from the DPC status register."""
    if not status & 1:
        return "not triggered"
    reason = (status >> 1) & 0x3
    if reason == 3:
        ext = (status >> 5) & 0x3
        return {0: "root port PIO error", 1: "software trigger"}.get(
            ext, f"extended reason {ext}"
        )
    return {0: "unmasked uncorrectable error", 1: "ERR_NONFATAL", 2: "ERR_FATAL"}[
        reason
    ]
