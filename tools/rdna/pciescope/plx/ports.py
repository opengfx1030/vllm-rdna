# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEX port number <-> BDF <-> GPU (HIP index, smi index, lease slot) map.

A PEX88096 in standard mode presents a nested virtual hierarchy to the host (the
"iSSW" fan-out of Broadcom's SDK), measured on par1-cs25 [BOX]::

    root port -> USP (port 48 or 0, see below)       BAR0 = all PEX registers
      +- internal DSP, device 0/4/8/12 (224..227)
      |    +- internal USP (232..235) of sub-switch k = 0/4/8/12
      |         +- leaf DSPs (physical x16 or x8 ports, or the empty x1 ports
      |            116/117)
      |              +- on-card AMD bridge pair -> GPU
      +- GEP DSP, device 0x1c (255)
           +- management endpoint MPT0 (247), class 0x0180

A leaf DSP's PEX port number is ``32 * j + device`` (``j`` = index of the
sub-switch: its internal DSP has device ``4 * j``), which reproduces the numbers
read from LnkCap[31:24] of every function (0, 16, 32, 64, 80, 116, 117 on the two
original cards; 16, 24 ... 88, 116, 117 on the SlimSAS card).  A physical port
belongs to station ``port // 16``.  The uplink is a x16 port, so it owns the one
station no leaf port lives in: its number is that station's base port (48 on the
original cards, 0 on the SlimSAS card).  The rules are fallbacks: the collector
verifies them against the bus numbers in the port windows before use.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from pciescope.model import Gpu, PciNode, Topology, short_bdf
from pciescope.plx import regs
from pciescope.probe.hw import layout_of_box

KIND_USP = "usp"
KIND_DSP = "dsp"
KIND_INT_DSP = "int_dsp"
KIND_INT_USP = "int_usp"
KIND_GEP = "gep_dsp"
KIND_MGMT = "mgmt"

#: internal downstream ports by their device number on the USP's secondary bus
_TOP_DSP_BY_DEVICE = {
    0x00: regs.PORT_INT_DS[0],
    0x04: regs.PORT_INT_DS[1],
    0x08: regs.PORT_INT_DS[2],
    0x0C: regs.PORT_INT_DS[3],
    0x1C: regs.PORT_GEP_DS,
}
#: switch letter by root complex bus (brief: root c0 = A, root 00 = B)
SWITCH_BY_ROOT_BUS = {"c0": "A", "00": "B"}


def _device(bdf: str) -> int:
    """PCI device number of a (short) BDF."""
    return int(short_bdf(bdf).rsplit(".", 1)[0].rsplit(":", 1)[-1], 16)


@dataclass(frozen=True, slots=True)
class PortInfo:
    """One function of a switch with its PEX port number."""

    switch: str  # 'A' | 'B'
    port: int  # PEX port number (LnkCap[31:24], window index)
    bdf: str  # short BDF of the function; entity ``port:<bdf>``
    kind: str  # usp | dsp | int_dsp | int_usp | gep_dsp | mgmt
    station: int | None  # port // 16 for physical ports
    station_port: int | None  # port % 16
    physical: bool  # has lanes (a leaf DSP or the USP)
    peer: str | None  # 'host' | 'gpu:<bus>' | None
    peer_bdf: str | None  # root port (usp) or the GPU function
    hip: int | None = None  # HIP index of the GPU on a leaf DSP (box rules table)
    smi: int | None = None  # rocm-smi / amd-smi index
    slot: str | None = None  # slot letter the topology gives a gfx1030 card
    gpu_name: str | None = None  # marketing name from the PCI database
    lease: str | None = None  # holder-file suffix of its lease (a2, b4, slotA, ...)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChipMap:
    """The ports of one switch chip."""

    switch: str
    usp_bdf: str
    usp_port: int
    root_bus: str
    serial: str | None
    ports: tuple[PortInfo, ...]

    def by_port(self) -> dict[int, PortInfo]:
        return {p.port: p for p in self.ports}

    def by_bdf(self) -> dict[str, PortInfo]:
        return {p.bdf: p for p in self.ports}

    def physical_ports(self) -> tuple[PortInfo, ...]:
        return tuple(p for p in self.ports if p.physical)

    def to_dict(self) -> dict[str, object]:
        return {
            "switch": self.switch,
            "usp_bdf": self.usp_bdf,
            "usp_port": self.usp_port,
            "root_bus": self.root_bus,
            "serial": self.serial,
            "ports": [p.to_dict() for p in self.ports],
        }


def is_switch_node(n: PciNode) -> bool:
    """A function of a PEX88096 (every one of them reports 1000:c010)."""
    return n.vendor == regs.VENDOR_ID and n.device == regs.DEVICE_ID


def _root_bus(topo: Topology, usp: PciNode) -> str:
    cur = usp
    while cur.parent is not None:
        cur = topo.nodes[cur.parent]
    return cur.root or short_bdf(cur.bdf).split(":")[-2]


def _children(topo: Topology, node: PciNode) -> list[PciNode]:
    return [topo.nodes[c] for c in node.children if c in topo.nodes]


def uplink_port(leaf_ports: Iterable[int]) -> int:
    """PEX port number of the upstream port: the base port of the free station.

    The uplink is a x16 port and owns a whole station, so it is the station no leaf
    port lives in (48 on the two original cards, 0 on the SlimSAS card).  When
    several stations are free (a board with fewer GPU ports) the original wiring
    wins if it fits, else the lowest; the collector checks the result against the
    port windows, so a wrong guess makes the chip unavailable instead of mislabelled.
    """
    used = {regs.station_of(p) for p in leaf_ports}
    free = [
        regs.station_base_port(s) for s in range(regs.PERF_STATIONS) if s not in used
    ]
    if len(free) == 1:
        return free[0]
    if regs.DEFAULT_USP_PORT in free or not free:
        return regs.DEFAULT_USP_PORT
    return free[0]


def build_port_maps(topo: Topology) -> list[ChipMap]:
    """Port maps of every PEX88096 found in ``topo`` (sorted by switch letter)."""
    peer_by_port: dict[str, str] = {
        short_bdf(g.port): g.id for g in topo.gpus.values() if g.port
    }
    gpu_by_id = {g.id: g for g in topo.gpus.values()}
    layout = layout_of_box(g.bus for g in topo.gpus.values())
    chips: list[ChipMap] = []
    for usp in topo.nodes.values():
        if not is_switch_node(usp):
            continue
        parent = topo.nodes.get(usp.parent) if usp.parent else None
        if parent is not None and is_switch_node(parent):
            continue  # not the top of a chip
        root_bus = _root_bus(topo, usp)
        letter = usp.switch or SWITCH_BY_ROOT_BUS.get(root_bus, "?")
        ports: list[PortInfo] = []
        leaf_ports: list[int] = []

        def add(
            node: PciNode,
            port: int,
            kind: str,
            peer: str | None = None,
            peer_bdf: str | None = None,
            gpu: Gpu | None = None,
            _letter: str = letter,
            _ports: list[PortInfo] = ports,
        ) -> None:
            physical = kind in (KIND_USP, KIND_DSP) and port <= regs.MAX_PHYSICAL_PORT
            _ports.append(
                PortInfo(
                    switch=_letter,
                    port=port,
                    bdf=short_bdf(node.bdf),
                    kind=kind,
                    station=regs.station_of(port) if physical else None,
                    station_port=port % regs.PORTS_PER_STATION if physical else None,
                    physical=physical,
                    peer=peer,
                    peer_bdf=peer_bdf,
                    hip=gpu.hip if gpu else None,
                    smi=gpu.smi if gpu else None,
                    slot=gpu.slot if gpu else None,
                    gpu_name=gpu.name if gpu else None,
                    lease=layout.key_of(gpu.bus) if gpu else None,
                )
            )

        for top in _children(topo, usp):
            if not is_switch_node(top):
                continue
            dev = _device(top.bdf)
            top_port = _TOP_DSP_BY_DEVICE.get(dev)
            if top_port is None:
                continue
            if top_port == regs.PORT_GEP_DS:
                add(top, top_port, KIND_GEP)
                for ep in _children(topo, top):
                    if is_switch_node(ep):
                        add(ep, regs.PORT_MPT0, KIND_MGMT)
                continue
            add(top, top_port, KIND_INT_DSP)
            j = dev // 4
            for sub in _children(topo, top):
                if not is_switch_node(sub):
                    continue
                add(sub, regs.PORT_INT_UP[j], KIND_INT_USP)
                for leaf in _children(topo, sub):
                    if not is_switch_node(leaf):
                        continue
                    port = 32 * j + _device(leaf.bdf)
                    leaf_ports.append(port)
                    peer = peer_by_port.get(short_bdf(leaf.bdf))
                    add(
                        leaf,
                        port,
                        KIND_DSP,
                        peer,
                        gpu_by_id[peer].bdf if peer else None,
                        gpu_by_id[peer] if peer else None,
                    )
        usp_port = uplink_port(leaf_ports)
        add(usp, usp_port, KIND_USP, "host", parent.bdf if parent else None)
        ports.sort(key=lambda p: p.port)
        chips.append(
            ChipMap(
                switch=letter,
                usp_bdf=short_bdf(usp.bdf),
                usp_port=usp_port,
                root_bus=root_bus,
                serial=usp.serial,
                ports=tuple(ports),
            )
        )
    chips.sort(key=lambda c: c.switch)
    return chips


def verify_headers(
    chip: ChipMap, topo: Topology, headers: dict[int, tuple[int, int]]
) -> list[str]:
    """Compare window header dwords against the topology; return mismatch texts.

    ``headers[port] = (class dword at 0x08, bus dword at 0x18)``.  Bridges must
    show the same primary/secondary/subordinate bus numbers as the topology, the
    management endpoint only its class code.
    """
    problems: list[str] = []
    for p in chip.ports:
        if p.port not in headers:
            problems.append(f"port {p.port} ({p.bdf}): no header read")
            continue
        cls_dw, bus_dw = headers[p.port]
        node = topo.node(p.bdf)
        if cls_dw >> 8 != node.pci_class:
            problems.append(
                f"port {p.port} ({p.bdf}): class {cls_dw >> 8:#08x} "
                f"!= topology {node.pci_class:#08x}"
            )
        if p.kind == KIND_MGMT:
            continue
        want = (
            int(short_bdf(p.bdf).rsplit(":", 2)[-2], 16),
            node.secondary_bus,
            node.subordinate_bus,
        )
        got = (bus_dw & 0xFF, (bus_dw >> 8) & 0xFF, (bus_dw >> 16) & 0xFF)
        if want != got:
            problems.append(
                f"port {p.port} ({p.bdf}): buses (pri,sec,sub) {got} != topology {want}"
            )
    return problems
