# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover the PCIe fabric of the box from sysfs (no lspci, no privileges).

``discover()`` walks ``/sys/bus/pci/devices``, keeps every root complex that
carries an AMD GPU, rebuilds the bridge tree from the canonical sysfs paths and
classifies each bridge by its position (the PEX880xx switches present a virtual
hierarchy: USP -> fabric DSP -> sub-switch USP -> port DSP -> the card's own AMD
USP/DSP pair -> GPU), joins the GPUs with the KFD topology (HIP order = KFD node
order; rocm-smi order = ascending PCI address) and derives switch letters, lease
slots and links.  ``render_tree()`` prints it for humans.

What is a link: a PCIe link joins a downstream-type port (root port, switch DSP,
on-card AMD DSP) to the device below it.  The edge from a USP to its DSP children
is the switch's internal virtual bus: it has no link status and no lanes.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from pciescope.model import (
    LINK_FABRIC,
    LINK_GPU,
    LINK_HOST_UPLINK,
    LINK_MGMT,
    LINK_ONBOARD,
    LINK_OTHER,
    PORT_DSP,
    PORT_ENDPOINT,
    PORT_ROOT,
    PORT_USP,
    ROLE_GPU,
    ROLE_GPU_AUDIO,
    ROLE_GPU_DOWN,
    ROLE_GPU_UP,
    ROLE_OTHER,
    ROLE_ROOT_PORT,
    ROLE_SWITCH_DOWN,
    ROLE_SWITCH_FABRIC,
    ROLE_SWITCH_MGMT,
    ROLE_SWITCH_SUB,
    ROLE_SWITCH_UP,
    Gpu,
    Link,
    PciNode,
    RootComplex,
    Switch,
    Topology,
    bdf_bus,
    gpu_entity,
    link_entity,
    now_ns,
    parse_speed_gts,
    raw_gbs,
    root_entity,
    short_bdf,
    switch_entity,
)
from pciescope.sysfs import Sysfs

VENDOR_AMD = 0x1002
SWITCH_VENDORS = (0x1000, 0x10B5)  # Broadcom/LSI (PEX880xx), PLX (older PEX)
AMD_BRIDGE_UP = 0x1478  # Navi 10 XL upstream port of PCI Express switch
AMD_BRIDGE_DOWN = 0x1479  # Navi 10 XL downstream port of PCI Express switch

#: PCI device id -> (marketing name, gfx target) for the cards this fork runs
GPU_PCI_IDS: dict[int, tuple[str, str]] = {
    0x73A1: ("Radeon Pro V620", "gfx1030"),
    0x7449: ("Radeon Pro W7800 48GB", "gfx1100"),
}
#: (vendor, device) -> switch model (lspci names 1000:c010 "PEX880xx PCIe Gen 4")
SWITCH_MODELS: dict[tuple[int, int], str] = {(0x1000, 0xC010): "PEX88096"}


@dataclass(frozen=True, slots=True)
class BoxLayout:
    """One GPU layout of par1-cs25 as the box rules document it.

    ``hip_buses`` is the bus of HIP 0..n-1 (KFD node order); ``slots`` maps a lease
    slot letter to the gfx1030 buses its holder file covers (a W7800 is in no slot).
    The lease slot is a policy of the box rules, not a property of the wiring.
    """

    name: str
    hip_buses: tuple[str, ...]
    slots: tuple[tuple[str, tuple[str, ...]], ...]


#: Layouts the box rules (``~/w4a8_runs/lanes/lease/README.md``) have documented, newest
#: first.  A GPU bus set that equals one of them gets that layout's lease slots; any
#: other set falls back to the switch letter and, on par1-cs25, adds a note.
DOCUMENTED_LAYOUTS: tuple[BoxLayout, ...] = (
    BoxLayout(  # since the case move of 2026-11-08: one switch, leases a2 and b4
        "six V620 behind one switch (interim, since 2026-11-08)",
        ("d1", "d4", "d9", "dc", "df", "e2"),
        (("A", ("d1", "d4")), ("B", ("d9", "dc", "df", "e2"))),
    ),
    BoxLayout(  # until 2026-11-08: two switches, slotA and slotB, W7800 in none
        "ten GPUs behind two switches (2026-11-05)",
        ("cb", "ce", "d3", "d8", "db", "07", "0a", "0f", "14", "17"),
        (("A", ("ce", "d3", "d8", "db")), ("B", ("0a", "0f", "14", "17"))),
    ),
)


_PCI_DEVICES = "bus/pci/devices"
_KFD_NODES = "class/kfd/kfd/topology/nodes"


@dataclass(slots=True)
class _Raw:
    """One PCI function as listed by sysfs, before classification."""

    sysname: str  # 0000:ce:00.0
    bdf: str  # ce:00.0
    vendor: int
    device: int
    pci_class: int
    chain: list[str]  # short BDFs from the root port down to this function
    root_bus: str


@dataclass(slots=True)
class _Kfd:
    node: int
    gpu_id: int
    bdf: str
    gfx_target_version: int
    props: dict[str, int]


def _is_bdf(name: str) -> bool:
    try:
        short_bdf(name)
    except ValueError:
        return False
    return True


def _parse_chain(real: str) -> tuple[str, list[str]] | None:
    """Split ``devices/pci0000:c0/0000:c0:03.1/0000:c5:00.0`` into (root bus, chain)."""
    parts = real.split("/")
    for i, part in enumerate(parts):
        if part.startswith("pci") and ":" in part and not _is_bdf(part):
            bus = part.split(":", 1)[1].lower()
            chain = [short_bdf(p) for p in parts[i + 1 :] if _is_bdf(p)]
            return (bus, chain) if chain else None
    return None


def _gfx_name(target_version: int) -> str:
    """KFD ``gfx_target_version`` (100300) -> ``gfx1030``; 90a00 -> ``gfx90a``."""
    major, minor, step = (
        target_version // 10000,
        (target_version // 100) % 100,
        (target_version % 100),
    )
    return f"gfx{major}{minor:x}{step:x}"


def _parse_props(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (text or "").splitlines():
        key, _, val = line.partition(" ")
        try:
            out[key] = int(val.strip())
        except ValueError:
            continue
    return out


def _read_kfd(sfs: Sysfs) -> list[_Kfd]:
    """GPU nodes of the KFD topology, in node-id order (that is the HIP order)."""
    out: list[_Kfd] = []
    names = [n for n in sfs.listdir(_KFD_NODES) if n.isdigit()]
    for name in sorted(names, key=int):
        base = f"{_KFD_NODES}/{name}"
        props = _parse_props(sfs.read(f"{base}/properties"))
        gpu_id = sfs.read_int(f"{base}/gpu_id") or 0
        if gpu_id == 0 or "location_id" not in props:
            continue  # CPU node
        loc = props["location_id"]
        dom = props.get("domain", 0)
        bdf = short_bdf(
            f"{dom:04x}:{(loc >> 8) & 0xFF:02x}:{(loc >> 3) & 0x1F:02x}.{loc & 7}"
        )
        out.append(
            _Kfd(int(name), gpu_id, bdf, props.get("gfx_target_version", 0), props)
        )
    return out


def _list_raw(sfs: Sysfs) -> list[_Raw]:
    out: list[_Raw] = []
    for name in sfs.listdir(_PCI_DEVICES):
        if not _is_bdf(name):
            continue
        base = f"{_PCI_DEVICES}/{name}"
        vendor = sfs.read_int(f"{base}/vendor", 0)
        device = sfs.read_int(f"{base}/device", 0)
        cls = sfs.read_int(f"{base}/class", 0)
        real = sfs.realpath(base)
        if vendor is None or device is None or cls is None or real is None:
            continue
        parsed = _parse_chain(real)
        if parsed is None:
            continue
        root_bus, chain = parsed
        out.append(_Raw(name, short_bdf(name), vendor, device, cls, chain, root_bus))
    return out


def _driver(sfs: Sysfs, base: str) -> str | None:
    link = sfs.readlink(f"{base}/driver")
    return link.rsplit("/", 1)[-1] if link else None


def _read_node(sfs: Sysfs, raw: _Raw) -> PciNode:
    base = f"{_PCI_DEVICES}/{raw.sysname}"
    return PciNode(
        bdf=raw.bdf,
        vendor=raw.vendor,
        device=raw.device,
        pci_class=raw.pci_class,
        parent=raw.chain[-2] if len(raw.chain) > 1 else None,
        root=raw.root_bus,
        depth=len(raw.chain) - 1,
        driver=_driver(sfs, base),
        secondary_bus=sfs.read_int(f"{base}/secondary_bus_number"),
        subordinate_bus=sfs.read_int(f"{base}/subordinate_bus_number"),
        link_speed_gts=parse_speed_gts(sfs.read(f"{base}/current_link_speed")),
        link_width=sfs.read_int(f"{base}/current_link_width"),
        max_link_speed_gts=parse_speed_gts(sfs.read(f"{base}/max_link_speed")),
        max_link_width=sfs.read_int(f"{base}/max_link_width"),
        numa_node=sfs.read_int(f"{base}/numa_node"),
        has_aer=sfs.exists(f"{base}/aer_dev_correctable"),
    )


def _is_bridge(n: PciNode) -> bool:
    return (n.pci_class >> 8) == 0x0604


def _is_gpu_class(n: PciNode) -> bool:
    return n.vendor == VENDOR_AMD and (n.pci_class >> 16) == 0x03


def _classify(nodes: dict[str, PciNode]) -> None:
    """Fill ``children``, ``port_type`` and ``role`` of every node (in place)."""
    for n in nodes.values():
        n.children = []
    for n in nodes.values():
        if n.parent is not None:
            nodes[n.parent].children.append(n.bdf)
    for n in nodes.values():
        n.children.sort()
    order = sorted(nodes.values(), key=lambda n: (n.depth, n.bdf))
    for n in order:  # parents before children
        if n.parent is None:
            n.port_type = PORT_ROOT if _is_bridge(n) else PORT_ENDPOINT
        elif not _is_bridge(n):
            n.port_type = PORT_ENDPOINT
        else:
            up = nodes[n.parent].port_type
            n.port_type = PORT_USP if up in (PORT_ROOT, PORT_DSP) else PORT_DSP
    for n in order:
        n.role = _role(n, nodes)


def _role(n: PciNode, nodes: dict[str, PciNode]) -> str:
    is_switch_vendor = n.vendor in SWITCH_VENDORS
    if n.parent is None and _is_bridge(n):
        return ROLE_ROOT_PORT
    if is_switch_vendor and _is_bridge(n):
        if n.port_type == PORT_USP:
            parent_role = nodes[n.parent].role if n.parent else ""
            return ROLE_SWITCH_UP if parent_role == ROLE_ROOT_PORT else ROLE_SWITCH_SUB
        if any(
            nodes[c].vendor in SWITCH_VENDORS
            and (_is_bridge(nodes[c]) or (nodes[c].pci_class >> 16) != 0x06)
            for c in n.children
        ):
            return ROLE_SWITCH_FABRIC  # leads to a sub-switch or the mgmt endpoint
        return ROLE_SWITCH_DOWN
    if is_switch_vendor and (n.pci_class >> 16) != 0x06:
        return ROLE_SWITCH_MGMT
    if n.vendor == VENDOR_AMD and _is_bridge(n):
        if n.device == AMD_BRIDGE_UP:
            return ROLE_GPU_UP
        if n.device == AMD_BRIDGE_DOWN:
            return ROLE_GPU_DOWN
    if _is_gpu_class(n):
        return ROLE_GPU
    if n.vendor == VENDOR_AMD and (n.pci_class >> 8) == 0x0403:
        return ROLE_GPU_AUDIO
    return ROLE_OTHER


def _link_kind(up: PciNode, down: PciNode) -> tuple[str, bool]:
    """(kind, physical) of the link between a downstream-type port and its child."""
    pair = (up.role, down.role)
    if pair == (ROLE_ROOT_PORT, ROLE_SWITCH_UP):
        return LINK_HOST_UPLINK, True
    if pair == (ROLE_SWITCH_FABRIC, ROLE_SWITCH_SUB):
        return LINK_FABRIC, False
    if pair == (ROLE_SWITCH_FABRIC, ROLE_SWITCH_MGMT):
        return LINK_MGMT, False
    if pair == (ROLE_SWITCH_DOWN, ROLE_GPU_UP):
        return LINK_GPU, True
    if up.role == ROLE_GPU_DOWN and down.role in (ROLE_GPU, ROLE_GPU_AUDIO):
        return LINK_ONBOARD, True
    if down.role == ROLE_GPU and up.role in (ROLE_ROOT_PORT, ROLE_SWITCH_DOWN):
        return LINK_GPU, True  # a card without the on-card bridge pair
    return LINK_OTHER, True


def _build_links(nodes: dict[str, PciNode]) -> dict[str, Link]:
    links: dict[str, Link] = {}
    seen_devices: set[tuple[str, str]] = set()
    for n in sorted(nodes.values(), key=lambda x: x.bdf):
        if n.parent is None:
            continue
        parent = nodes[n.parent]
        if parent.port_type not in (PORT_ROOT, PORT_DSP):
            continue  # USP -> DSP: internal bus, not a link
        dev = (parent.bdf, n.bdf.rsplit(".", 1)[0])
        if dev in seen_devices:
            continue  # another function of the same device shares the link
        seen_devices.add(dev)
        kind, physical = _link_kind(parent, n)
        speeds = [s for s in (parent.max_link_speed_gts, n.max_link_speed_gts) if s]
        widths = [w for w in (parent.max_link_width, n.max_link_width) if w]
        ms = min(speeds) if speeds else None
        mw = min(widths) if widths else None
        lid = link_entity(parent.bdf, n.bdf)
        links[lid] = Link(
            id=lid,
            up=parent.bdf,
            down=n.bdf,
            kind=kind,
            physical=physical,
            max_speed_gts=ms,
            max_width=mw,
            max_gbs=round(raw_gbs(ms, mw), 3) if ms and mw else None,
            switch=n.switch,
        )
    return links


def _read_gpu(
    sfs: Sysfs, n: PciNode, kfd: _Kfd | None, hip: int | None, smi: int | None
) -> Gpu:
    base = f"{_PCI_DEVICES}/{_sysname(n.bdf)}"
    known = GPU_PCI_IDS.get(n.device)
    product = sfs.read(f"{base}/product_name")
    if kfd is not None and kfd.gfx_target_version:
        arch = _gfx_name(kfd.gfx_target_version)
    else:
        arch = known[1] if known else "unknown"
    name = product or (known[0] if known else f"AMD GPU {n.vendor:04x}:{n.device:04x}")
    drm = sfs.listdir(f"{base}/drm")
    card = next((d for d in drm if d.startswith("card") and d[4:].isdigit()), None)
    render = next((d for d in drm if d.startswith("renderD")), None)
    hw = next((h for h in sfs.listdir(f"{base}/hwmon") if h.startswith("hwmon")), None)
    return Gpu(
        id=gpu_entity(bdf_bus(n.bdf)),
        bus=bdf_bus(n.bdf),
        bdf=n.bdf,
        name=name,
        arch=arch,
        device_id=n.device,
        root=n.root,
        switch=None,
        slot=None,
        port=None,
        link=None,
        hip=hip,
        smi=smi,
        kfd_node=kfd.node if kfd else None,
        kfd_gpu_id=kfd.gpu_id if kfd else None,
        card=card,
        render=render,
        vram_total_bytes=sfs.read_int(f"{base}/mem_info_vram_total"),
        unique_id=sfs.read(f"{base}/unique_id"),
        hwmon=f"{base}/hwmon/{hw}" if hw else None,
    )


def _sysname(bdf: str) -> str:
    return bdf if bdf.count(":") == 2 else f"0000:{bdf}"


def discover(
    sysfs: Sysfs | None = None,
    *,
    host: str | None = None,
) -> Topology:
    """Discover the GPU fabric of this host (or of a captured sysfs tree)."""
    sfs = sysfs or Sysfs()
    raws = _list_raw(sfs)
    gpu_raws = [
        r for r in raws if r.vendor == VENDOR_AMD and (r.pci_class >> 16) == 0x03
    ]
    keep_roots = {r.chain[0] for r in gpu_raws}
    nodes = {
        r.bdf: _read_node(sfs, r) for r in raws if r.chain and r.chain[0] in keep_roots
    }
    topo = Topology(host=host or socket.gethostname(), captured_ns=now_ns())
    if not nodes:
        topo.notes.append("no AMD GPU found in sysfs")
        return topo
    _classify(nodes)
    topo.nodes = dict(sorted(nodes.items()))

    kfd = _read_kfd(sfs)
    kfd_by_bdf = {k.bdf: k for k in kfd}
    gpu_nodes = sorted(
        (n for n in nodes.values() if n.role == ROLE_GPU), key=lambda n: n.bdf
    )
    smi_rank = {n.bdf: i for i, n in enumerate(gpu_nodes)}
    hip_rank: dict[str, int] = {}
    if kfd:
        hip_rank = {k.bdf: i for i, k in enumerate(kfd) if k.bdf in nodes}
        topo.hip_source = "kfd-node-order"
    else:
        topo.hip_source = "unavailable (no KFD topology): hip is None"
        topo.notes.append("KFD topology not readable: HIP indices unknown")
    gpus = {
        g.id: g
        for g in (
            _read_gpu(
                sfs, n, kfd_by_bdf.get(n.bdf), hip_rank.get(n.bdf), smi_rank[n.bdf]
            )
            for n in gpu_nodes
        )
    }
    topo.gpus = gpus
    _assign_switches(topo)
    topo.links = _build_links(nodes)
    _attach_gpus(topo)
    _label(topo)
    _assign_slots(topo)
    return topo


def _gpu_order_key(g: Gpu) -> tuple[int, int, str]:
    return (g.hip is None, g.hip if g.hip is not None else 0, g.bus)


def _assign_switches(topo: Topology) -> None:
    """Group nodes below each switch upstream port; letter switches in HIP order."""
    nodes = topo.nodes
    ups = sorted(n.bdf for n in nodes.values() if n.role == ROLE_SWITCH_UP)
    first_gpu: dict[str, tuple[int, int, str]] = {}
    gpu_up: dict[str, str] = {}
    for g in topo.gpus.values():
        cur = g.bdf
        while cur is not None and nodes[cur].role != ROLE_SWITCH_UP:
            cur = nodes[cur].parent
        if cur is not None:
            gpu_up[g.id] = cur
            key = _gpu_order_key(g)
            if cur not in first_gpu or key < first_gpu[cur]:
                first_gpu[cur] = key
    ordered = sorted(ups, key=lambda b: first_gpu.get(b, (True, 0, b)))
    letters = {bdf: chr(ord("A") + i) for i, bdf in enumerate(ordered)}
    for up_bdf in ordered:
        letter = letters[up_bdf]
        stack = [up_bdf]
        members: list[str] = []
        while stack:
            cur = stack.pop()
            nodes[cur].switch = letter
            members.append(cur)
            stack.extend(nodes[cur].children)
        up = nodes[up_bdf]
        sw = Switch(
            id=switch_entity(letter),
            letter=letter,
            root=up.root,
            root_port=up.parent or "",
            upstream=up_bdf,
            model=SWITCH_MODELS.get(
                (up.vendor, up.device), f"{up.vendor:04x}:{up.device:04x}"
            ),
        )
        sw.ports = sorted(m for m in members if nodes[m].role == ROLE_SWITCH_DOWN)
        mgmt = [m for m in members if nodes[m].role == ROLE_SWITCH_MGMT]
        sw.mgmt = mgmt[0] if mgmt else None
        sw.gpus = [g.id for g in topo.gpus.values() if gpu_up.get(g.id) == up_bdf]
        sw.gpus.sort(key=lambda gid: _gpu_order_key(topo.gpus[gid]))
        topo.switches[sw.id] = sw
    for g in topo.gpus.values():
        up_bdf = gpu_up.get(g.id)
        if up_bdf is not None:
            g.switch = letters[up_bdf]


def _attach_gpus(topo: Topology) -> None:
    """Join every GPU with its downstream port, physical link and bridge nodes."""
    nodes = topo.nodes
    for g in topo.gpus.values():
        chain: list[str] = []
        cur: str | None = g.bdf
        while cur is not None:
            chain.append(cur)
            cur = nodes[cur].parent
        for bdf in chain:
            if nodes[bdf].role in (ROLE_GPU, ROLE_GPU_UP, ROLE_GPU_DOWN):
                nodes[bdf].gpu = g.id
        for sibling in nodes[chain[1]].children if len(chain) > 1 else []:
            if nodes[sibling].role == ROLE_GPU_AUDIO:
                nodes[sibling].gpu = g.id
        for lid, link in topo.links.items():
            if link.down in chain and link.kind in (LINK_GPU, LINK_ONBOARD):
                link.gpu = g.id
                if link.kind == LINK_GPU:
                    g.link = lid
                    g.port = link.up
    rcs: list[RootComplex] = []
    for rc_bus in sorted({n.root for n in nodes.values()}):
        root_ports = [
            n for n in nodes.values() if n.parent is None and n.root == rc_bus
        ]
        sw = next((s for s in topo.switches.values() if s.root == rc_bus), None)
        rcs.append(
            RootComplex(
                id=root_entity(rc_bus),
                bus=rc_bus,
                root_port=root_ports[0].bdf,
                switch=sw.letter if sw else None,
                gpus=[g.id for g in topo.gpu_list() if g.root == rc_bus],
            )
        )
    for rc in sorted(rcs, key=lambda r: (r.switch is None, r.switch or "", r.bus)):
        topo.roots[rc.id] = rc


def _label(topo: Topology) -> None:
    for n in topo.nodes.values():
        sw = f"switch {n.switch} " if n.switch else ""
        gpu = topo.gpus.get(n.gpu) if n.gpu else None
        gname = f" of {gpu.id}" if gpu else ""
        n.label = {
            ROLE_ROOT_PORT: "root port",
            ROLE_SWITCH_UP: f"{sw}upstream port",
            ROLE_SWITCH_FABRIC: f"{sw}fabric port",
            ROLE_SWITCH_SUB: f"{sw}sub-switch",
            ROLE_SWITCH_DOWN: f"{sw}downstream port",
            ROLE_SWITCH_MGMT: f"{sw}management endpoint",
            ROLE_GPU_UP: f"on-card bridge (up){gname}",
            ROLE_GPU_DOWN: f"on-card bridge (down){gname}",
            ROLE_GPU: gpu.name if gpu else "GPU",
            ROLE_GPU_AUDIO: f"GPU audio function{gname}",
        }.get(n.role, "other")
    for link in topo.links.values():
        link.label = {
            LINK_HOST_UPLINK: "host uplink",
            LINK_FABRIC: "switch fabric (virtual)",
            LINK_GPU: "GPU link",
            LINK_ONBOARD: "on-card link",
            LINK_MGMT: "management (virtual)",
        }.get(link.kind, "link")
        if link.gpu:
            link.label += f" {link.gpu}"
        elif link.switch:
            link.label += f" switch {link.switch}"


def _assign_slots(topo: Topology) -> None:
    """Set ``Gpu.slot`` from the documented layout; note what matches none.

    Until the 2026-11-08 case move a lease slot was a switch (A, B); since, it is a
    leaf of the single remaining switch (A = HIP 0-1, B = HIP 2-5), so the switch
    letter no longer says which lease a GPU belongs to.  A GPU set that equals no
    documented layout keeps the old rule (the switch letter for a gfx1030 card).
    """
    buses = {g.bus for g in topo.gpus.values()}
    layout = next((x for x in DOCUMENTED_LAYOUTS if set(x.hip_buses) == buses), None)
    slot_of = {b: s for s, members in (layout.slots if layout else ()) for b in members}
    for g in topo.gpus.values():
        if g.arch != "gfx1030":
            g.slot = None
        else:
            g.slot = slot_of.get(g.bus) if layout else g.switch
    if not topo.host.startswith("par1-cs25") or not buses:
        return
    got = tuple(g.bus for g in topo.gpu_list("hip") if g.hip is not None)
    if layout is None:
        known = "; ".join(
            f"{x.name}: {','.join(x.hip_buses)}" for x in DOCUMENTED_LAYOUTS
        )
        topo.notes.append(
            f"GPU buses {','.join(sorted(buses))} match no layout of the box rules "
            f"table ({known}): lease slots fall back to the switch letter"
        )
    elif got != layout.hip_buses:
        topo.notes.append(
            f"HIP order (KFD node order) differs from the box rules table "
            f"({layout.name}): {','.join(got)} != {','.join(layout.hip_buses)}"
        )


# ---------------------------------------------------------------------------
# human rendering
# ---------------------------------------------------------------------------


def _fmt_link(speed: float | None, width: int | None) -> str:
    if speed is None or width is None:
        return "?"
    if width == 0:
        return "down"
    s = f"{speed:g}"
    return f"{s} GT/s x{width}"


def _node_line(topo: Topology, n: PciNode) -> str:
    link = _fmt_link(n.link_speed_gts, n.link_width)
    return f"{n.bdf} {n.label} [{link}]"


def _gpu_line(topo: Topology, g: Gpu) -> str:
    n = topo.nodes[g.bdf]
    gl = topo.links.get(g.link) if g.link else None
    cap = f" {gl.max_gbs:.1f} GB/s raw" if gl and gl.max_gbs else ""
    port = topo.nodes[g.port] if g.port else None
    state = (
        _fmt_link(port.link_speed_gts, port.link_width)
        if port
        else _fmt_link(n.link_speed_gts, n.link_width)
    )
    slot = f" slot {g.slot}" if g.slot else ""
    ids = f"HIP {g.hip}" if g.hip is not None else "HIP ?"
    smi = f" smi {g.smi}" if g.smi is not None else ""
    kfd = f" kfd-node {g.kfd_node}" if g.kfd_node is not None else ""
    return f"{g.id} {g.name} {g.arch} {ids}{smi}{kfd}{slot} [{state}{cap}]"


def render_tree(topo: Topology, *, full: bool = False) -> str:
    """Readable tree: roots -> switch -> sub-switches -> ports -> GPUs.

    The two on-card AMD bridges between a switch port and its GPU are folded into
    the GPU line unless ``full`` is set (then every bridge is listed).
    """
    out: list[str] = []
    out.append(
        f"{topo.host}: {len(topo.roots)} root complex(es), {len(topo.switches)} "
        f"switch(es), {len(topo.gpus)} GPU(s); HIP order from {topo.hip_source}"
    )

    def walk(bdf: str, prefix: str, last: bool, top: bool = False) -> None:
        n = topo.nodes[bdf]
        branch = "" if top else ("└─ " if last else "├─ ")
        kids = list(n.children)
        if not full and n.role == ROLE_SWITCH_DOWN and n.children:
            gpu_ids = [
                topo.nodes[c].gpu for c in n.children if topo.nodes[c].gpu is not None
            ]
            if gpu_ids and gpu_ids[0] in topo.gpus:
                g = topo.gpus[gpu_ids[0]]
                out.append(f"{prefix}{branch}{n.bdf} port -> {_gpu_line(topo, g)}")
                return
        if not full and n.role in (ROLE_GPU_UP, ROLE_GPU_DOWN, ROLE_GPU_AUDIO):
            return
        if n.role == ROLE_GPU:
            g = topo.gpus[gpu_entity(bdf_bus(bdf))]
            out.append(f"{prefix}{branch}{_gpu_line(topo, g)}")
            return
        if n.role == ROLE_SWITCH_DOWN and not n.children:
            state = _fmt_link(n.link_speed_gts, n.link_width)
            out.append(f"{prefix}{branch}{n.bdf} downstream port [unused: {state}]")
            return
        out.append(f"{prefix}{branch}{_node_line(topo, n)}")
        child_prefix = prefix + ("" if top else ("   " if last else "│  "))
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    for rc in topo.roots.values():
        out.append(f"root complex {rc.bus} (switch {rc.switch})")
        walk(rc.root_port, "  ", True, top=False)
    for note in topo.notes:
        out.append(f"note: {note}")
    return "\n".join(out)


def summary(topo: Topology) -> dict[str, object]:
    """Small machine summary of a topology (counts and the GPU table)."""
    return {
        "host": topo.host,
        "roots": [r.id for r in topo.roots.values()],
        "switches": {
            s.id: {"root": s.root, "upstream": s.upstream, "gpus": s.gpus}
            for s in topo.switches.values()
        },
        "gpus": [
            {
                "id": g.id,
                "hip": g.hip,
                "smi": g.smi,
                "kfd_node": g.kfd_node,
                "arch": g.arch,
                "switch": g.switch,
                "slot": g.slot,
                "name": g.name,
            }
            for g in topo.gpu_list()
        ],
        "hip_source": topo.hip_source,
        "notes": list(topo.notes),
    }
