# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEX88096 ("Atlas") register map: offsets, bit fields and their provenance.

Provenance tags used in the comments:

``[SDK]``  Broadcom PLX SDK 9.81 (GitHub ``xiallc/broadcom_pci_pcie_sdk``, GPLv2),
           ``Driver/Source.PlxSvc/ApiFunc.c`` (``PlxPciPerformance*``),
           ``PlxApi/PlxApi.c``, ``Include/Plx.h``; and ``DaiZhiyuan/PlxSdk``
           ``PlxApi/PlxApiDirect.[ch]`` (2025).  Only offsets and bit meanings
           (facts) are used here, no SDK code.
``[BOX]``  read-only probe of par1-cs25 (BAR0 mmap with PROT_READ, 2026-11-07),
           see docs/rdna2/pciescope-2026-11-07.md.
``[SPEC]`` PCI Express Base Specification capability structures.

BAR0 of the upstream port (USP) is 16 MiB.  Port ``N`` has a 4 KiB register
window at ``0x800000 + N * 0x1000`` [SDK ``ATLAS_PEX_REGS_BASE_OFFSET``,
``ATLAS_PORT_REGS_SIZE``]; its first 0x100 bytes and every PCIe extended
capability read exactly like the port's PCI configuration space [BOX, verified
for every non-zero config dword of ports 0, 48 and 116], and the rest of the
window holds vendor registers that config space does not expose (they read 0
through config space).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

VENDOR_ID = 0x1000
DEVICE_ID = 0xC010  # PEX880xx "Atlas" [BOX lspci, SDK PLX_FAMILY_ATLAS C010/C011/C012]
ID_DWORD = (DEVICE_ID << 16) | VENDOR_ID  # dword 0 of every port window

BAR0_SIZE = 0x1000000
PORT_WINDOW_BASE = 0x800000
PORT_WINDOW_SIZE = 0x1000
PORTS_PER_STATION = 16
#: stations 0..5 carry the PerformancePAK counters (96 ports) [SDK]
PERF_STATIONS = 6
#: physical port numbers are < 128; 224.. are the virtual-switch ports below
MAX_PHYSICAL_PORT = 127

# -- special (virtual) ports of the iSSW fan-out [SDK PlxApiDirect.h, BOX] ---------
PORT_INT_DS = (224, 225, 226, 227)  # internal downstream ports, device 0/4/8/12
PORT_INT_UP = (232, 233, 234, 235)  # internal upstream ports of the sub-switches
PORT_MPT0 = 247  # management endpoint (device 0x00b2) [BOX class 0x0180]
PORT_GEP_DS = 255  # downstream port above the management endpoint [BOX device 0x1c]
VIRTUAL_PORTS = frozenset((*PORT_INT_DS, *PORT_INT_UP, PORT_MPT0, PORT_GEP_DS))
#: the host-facing port of the two original cards [BOX 0x360 reads 0x30]; the uplink
#: of the SlimSAS card is port 0 [BOX 0x360 reads 0].  Only the tie-break of
#: ``ports.build_port_maps`` when the topology alone cannot tell which station the
#: uplink uses; every root verb checks the map against the windows
DEFAULT_USP_PORT = 48


# -- what no write may touch ------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KnownChip:
    """A PEX88096 whose arrangement was derived on the box.

    ``usp_bdf`` is the BDF (domain form) of its upstream port where it was derived:
    a chip that moved is refused until the entry is re-derived.  ``protected`` maps
    stations whose port windows no write may ever land in to the reason; the
    per-station registers (0x3E0, 0x3F0, 0x768, ...) live in the window of the
    station's base port, so a protected station is not written at all.
    """

    switch: str  # the letter the topology gave it in that arrangement
    usp_bdf: str
    note: str
    protected: Mapping[int, str] = field(default_factory=dict)


#: the PEX88096 chips of par1-cs25 by Device Serial Number.  The helper maps a chip
#: writable only when its serial is listed here AND it sits at the listed BDF, so a
#: replaced card or a chip that moved (bus numbers change with the hardware) is
#: read-only until someone re-derives its entry.  Being listed approves nothing:
#: ``plx/policy.py`` says which chips may be written.
KNOWN_CHIPS: dict[str, KnownChip] = {
    # 2026-11-08 case move: one SlimSAS PLX card carries all six V620 (no W7800); its
    # x16 uplink is port 0, the GPUs sit on the x8 ports 48, 56 (leaf A) and 64, 72,
    # 80, 88 (leaf B).  Nothing on it is protected: the hazard that protected a
    # station before (below) came with a W7800 that is no longer installed
    "00-80-5e-10-b7-3d-3b-62": KnownChip(
        "A", "0000:c5:00.0", "SlimSAS card, interim layout (2026-11-10)"
    ),
    # the two cards of the pre-move layout (not installed since 2026-11-08)
    "00-80-5e-10-fa-53-6b-2a": KnownChip(
        "A", "0000:c5:00.0", "root complex c0, pre-move layout (2026-11-07)"
    ),
    "00-80-5e-10-f9-57-20-61": KnownChip(
        "B",
        "0000:01:00.0",
        "root complex 00, pre-move layout (2026-11-07)",
        {
            0: "the W7800 of HIP 5 (gpu:07, lease key w7800-07: the lead only) hangs "
            "on port 0"
        },
    ),
}


def port_window(port: int) -> int:
    """BAR0 offset of the 4 KiB register window of PEX port ``port``."""
    return PORT_WINDOW_BASE + port * PORT_WINDOW_SIZE


def station_of(port: int) -> int:
    """Station (16 ports each) of a physical port."""
    return port // PORTS_PER_STATION


def station_base_port(station: int) -> int:
    """Port whose window carries the per-station registers (3E0/3F0/768...)."""
    return station * PORTS_PER_STATION


# -- vendor registers inside a port window (offsets within the 4 KiB) --------------
#: clock-enable bit masks, ports 0-31, 32-63, 64-95 and (bit 0 = port 116) 116-127
#: [SDK PlxPciPerformance* port filter; BOX 0x00010001 x3 + 0x3 on the two original
#: cards (x16 ports 0, 16, 32, 48, 64, 80 and 116, 117); 0x01010001, 0x01010101,
#: 0x01010101, 0x3 on the SlimSAS card (x16 port 0, x8 ports 16, 24 ... 88)].
#: Only the station-base ports (0, 16, 32, 48, 64, 80) return them [BOX].
REG_PORT_ENABLE = (0x30C, 0x310, 0x314, 0x318)
REG_VS0_UPSTREAM = 0x360  # virtual switch 0 upstream port number [BOX 0x30 = 48, 0]
REG_CAPTURED_BUS = 0x97C  # [7:0] bus, [12:8] device captured by the port [BOX]

# PerformancePAK (per-port TLP/DLLP counters) [SDK PlxPciPerformanceMonitorControl]
REG_PERF_PROBE_MODE = 0x350  # bit 8 = probe mode (must be 0 for the counters)
REG_PERF_CTRL = 0x3E0  # station base ports: bit 31 enable, 30 reset, 28/27 mode
REG_PERF_FIFO = 0x3E4  # counter FIFO: every read advances it (never read blindly)
REG_PERF_RAM_CTRL = 0x3F0  # station base ports: RAM control
REG_PERF_RAM_PTRS = (0x3F4, 0x3F8, 0x3FC)  # increment by 1 << 22 per read [BOX]
REG_PERF_INGRESS = 0x768  # station base ports, bit 29: ingress counters enable
#: every port, bit 6 egress counters enable.  Bit 6 reads back 1 on the station base
#: ports (0, 48, 64, 80 of the interim chip) and 0 on the second port of a station
#: (56, 72, 88), whose egress counters count all the same (window of 2026-11-10).
#: No code checks the bit after a write, and stopping writes 0 to it on every port.
REG_PERF_EGRESS = 0xF30

PERF_BIT_ENABLE = 1 << 31
PERF_CTRL_START = 0xD8000000  # enable | reset | infinite sampling | start [SDK]
PERF_CTRL_STOP = 0x40000000  # reset | disable [SDK]
PERF_RAM_CAPTURE = 0x25  # (2 << 4) | reset read pointer | RAM enable [SDK]
PERF_INGRESS_BIT = 1 << 29
PERF_EGRESS_ENABLE_BIT = 1 << 6
#: egress credit enable: reads 0 here but the SDK always keeps it set on every
#: read-modify-write of 0xF30, so we do too [SDK]
PERF_EGRESS_CREDIT_BIT = 1 << 21
PERF_PROBE_MODE_BIT = 1 << 8

#: registers whose *read* has a side effect: never read outside the perf sequence
READ_SENSITIVE = frozenset((REG_PERF_FIFO, *REG_PERF_RAM_PTRS))

# counter RAM layout per station: 224 dwords [SDK PlxPciPerformanceGetCounters]
PERF_COUNTERS_PER_PORT = 14
PERF_DWORDS_PER_STATION = PORTS_PER_STATION * PERF_COUNTERS_PER_PORT  # 224
PERF_TLP_FIELDS = ("ph", "pdw", "nph", "npdw", "cplh", "cpldw")
PERF_INGRESS_BASE = 0  # port p: 6 dwords at 6 * p
PERF_EGRESS_BASE = 6 * PORTS_PER_STATION  # 96
PERF_DLLP_IN_BASE = 12 * PORTS_PER_STATION  # 192, one dword per port
PERF_DLLP_OUT_BASE = 13 * PORTS_PER_STATION  # 208

# -- standard capability ids ------------------------------------------------------
CAP_PM = 0x01
CAP_MSI = 0x05
CAP_PCIE = 0x10

ECAP_AER = 0x0001
ECAP_VC = 0x0002
ECAP_DSN = 0x0003
ECAP_POWER_BUDGET = 0x0004
ECAP_VSEC = 0x000B
ECAP_ACS = 0x000D
ECAP_MULTICAST = 0x0012
ECAP_LTR = 0x0018
ECAP_SECONDARY_PCIE = 0x0019
ECAP_DPC = 0x001D
ECAP_DATA_LINK_FEATURE = 0x0025
ECAP_PHY_16GT = 0x0026
ECAP_LANE_MARGINING = 0x0027

ECAP_NAMES = {
    ECAP_AER: "aer",
    ECAP_VC: "vc",
    ECAP_DSN: "dsn",
    ECAP_POWER_BUDGET: "power_budget",
    ECAP_VSEC: "vsec",
    ECAP_ACS: "acs",
    ECAP_MULTICAST: "multicast",
    ECAP_LTR: "ltr",
    ECAP_SECONDARY_PCIE: "secondary_pcie",
    ECAP_DPC: "dpc",
    ECAP_DATA_LINK_FEATURE: "data_link_feature",
    ECAP_PHY_16GT: "phy_16gt",
    ECAP_LANE_MARGINING: "lane_margining",
}

#: link speed encoding of LnkCap/LnkSta (GT/s) [SPEC]
SPEED_GTS = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}
