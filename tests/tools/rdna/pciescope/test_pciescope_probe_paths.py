# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FabricModel of the pciescope probe on a synthetic two-switch topology.

The topology copies the shape of par1-cs25 (root complexes ``c0`` and ``00``,
one switch each, four V620 per switch, every physical link 16 GT/s x16) as
``Topology.path`` needs it: root port, switch upstream port, switch-internal
bridge, switch downstream port, on-card bridge pair, GPU. The expected numbers
are hand computed from 16 GT/s x16 = 31.508 GB/s per direction.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope import model as m  # noqa: E402
from pciescope.probe.paths import FabricModel  # noqa: E402

X16 = m.raw_gbs(16.0, 16)  # 31.508
A_GPUS = ("ce", "d3", "d8", "db")  # switch A, root c0
B_GPUS = ("0a", "0f", "14", "17")  # switch B, root 00

# root bus, root port, switch upstream, switch internal, downstream bus, letter
_LAYOUT = (
    ("c0", "c0:03.1", "c5:00.0", "c6:00.0", "c7", "A", A_GPUS),
    ("00", "00:01.1", "01:00.0", "02:00.0", "03", "B", B_GPUS),
)


def _node(
    bdf: str,
    role: str,
    parent: str | None,
    root: str,
    depth: int,
    switch: str | None,
    gpu: str | None = None,
) -> m.PciNode:
    return m.PciNode(
        bdf=bdf,
        vendor=0x1002,
        device=0,
        pci_class=0x0604,
        role=role,
        parent=parent,
        root=root,
        depth=depth,
        switch=switch,
        gpu=gpu,
    )


def build_topology(
    gpu_links: Mapping[str, tuple[float, int]] | None = None,
) -> m.Topology:
    """Build the synthetic fabric; ``gpu_links`` overrides (GT/s, width) per bus."""
    gpu_links = gpu_links or {}
    topo = m.Topology(host="synthetic", captured_ns=0)

    def add(
        node: m.PciNode, kind: str | None, physical: bool, gt: float, w: int
    ) -> None:
        topo.nodes[node.bdf] = node
        if node.parent is None:
            return
        topo.nodes[node.parent].children.append(node.bdf)
        assert kind is not None
        link = m.Link(
            id=m.link_entity(node.parent, node.bdf),
            up=node.parent,
            down=node.bdf,
            kind=kind,
            physical=physical,
            max_speed_gts=gt if physical else None,
            max_width=w if physical else None,
            max_gbs=m.raw_gbs(gt, w) if physical else None,
            gpu=node.gpu,
            switch=node.switch,
        )
        topo.links[link.id] = link

    for root, port, up, internal, dbus, letter, buses in _LAYOUT:
        add(_node(port, m.ROLE_ROOT_PORT, None, root, 0, None), None, False, 0, 0)
        add(
            _node(up, m.ROLE_SWITCH_UP, port, root, 1, letter),
            m.LINK_HOST_UPLINK,
            True,
            16.0,
            16,
        )
        add(
            _node(internal, m.ROLE_SWITCH_FABRIC, up, root, 2, letter),
            m.LINK_FABRIC,
            False,
            0,
            0,
        )
        gpu_ids = []
        for i, bus in enumerate(buses):
            n = int(bus, 16)
            down = f"{dbus}:{i:02x}.0"
            bridge_up, bridge_down, gpu_bdf = (
                f"{n - 2:02x}:00.0",
                f"{n - 1:02x}:00.0",
                f"{bus}:00.0",
            )
            gt, w = gpu_links.get(bus, (16.0, 16))
            ent = m.gpu_entity(bus)
            add(
                _node(down, m.ROLE_SWITCH_DOWN, internal, root, 3, letter),
                m.LINK_FABRIC,
                False,
                0,
                0,
            )
            add(
                _node(bridge_up, m.ROLE_GPU_UP, down, root, 4, letter, ent),
                m.LINK_GPU,
                True,
                gt,
                w,
            )
            add(
                _node(bridge_down, m.ROLE_GPU_DOWN, bridge_up, root, 5, letter, ent),
                m.LINK_ONBOARD,
                False,
                0,
                0,
            )
            add(
                _node(gpu_bdf, m.ROLE_GPU, bridge_down, root, 6, letter, ent),
                m.LINK_ONBOARD,
                False,
                0,
                0,
            )
            topo.gpus[ent] = m.Gpu(
                id=ent,
                bus=bus,
                bdf=gpu_bdf,
                name="Radeon Pro V620",
                arch="gfx1030",
                device_id=0,
                root=root,
                switch=letter,
                slot=letter,
                port=down,
                link=m.link_entity(down, bridge_up),
            )
            gpu_ids.append(ent)
        topo.switches[m.switch_entity(letter)] = m.Switch(
            id=m.switch_entity(letter),
            letter=letter,
            root=root,
            root_port=port,
            upstream=up,
            gpus=gpu_ids,
        )
        topo.roots[m.root_entity(root)] = m.RootComplex(
            id=m.root_entity(root),
            bus=root,
            root_port=port,
            switch=letter,
            gpus=gpu_ids,
        )
    return topo


def gpu_link(bus: str) -> str:
    return build_topology().gpus[m.gpu_entity(bus)].link or ""


@pytest.fixture
def model() -> FabricModel:
    return FabricModel(build_topology())


# --- the synthetic topology itself --------------------------------------------


def test_helper_reproduces_both_path_classes() -> None:
    topo = build_topology()
    assert topo.path_class("gpu:ce", "gpu:d3") == "intra-switch"
    assert topo.path_class("gpu:0a", "gpu:17") == "intra-switch"
    assert topo.path_class("gpu:ce", "gpu:0a") == "cross-root"
    assert topo.path_class("gpu:0f", "gpu:db") == "cross-root"
    assert topo.path_class("gpu:ce", m.HOST) == "host"
    assert topo.path_class(m.HOST, "gpu:0a") == "host"
    assert len(topo.gpus) == 8
    assert pytest.approx(31.508, abs=1e-3) == X16


# --- resources, capacity, bottleneck -------------------------------------------


def test_intra_switch_flow_uses_the_two_gpu_links(model: FabricModel) -> None:
    assert model.resources("ce", "d3") == (
        f"{gpu_link('ce')}:up",
        f"{gpu_link('d3')}:down",
    )


def test_cross_root_flow_adds_both_switch_uplinks(model: FabricModel) -> None:
    res = model.resources("ce", "0a")
    assert res == (
        f"{gpu_link('ce')}:up",
        "link:c0:03.1-c5:00.0:up",
        "link:00:01.1-01:00.0:down",
        f"{gpu_link('0a')}:down",
    )
    # only physical links: no fabric link inside the switch, no pseudo hop
    assert all(model.capacity(r) == pytest.approx(X16) for r in res)


def test_host_flows_use_the_uplink_of_the_gpu_switch(model: FabricModel) -> None:
    assert model.resources(m.HOST, "0a") == (
        "link:00:01.1-01:00.0:down",
        f"{gpu_link('0a')}:down",
    )
    assert model.resources("gpu:ce", m.HOST) == (
        f"{gpu_link('ce')}:up",
        "link:c0:03.1-c5:00.0:up",
    )


def test_entity_forms_are_equivalent(model: FabricModel) -> None:
    assert model.resources("CE", "gpu:0a") == model.resources("gpu:ce", "0a")


def test_bottleneck_is_the_first_of_equal_resources(model: FabricModel) -> None:
    res, gbps = model.bottleneck("ce", "0a")
    assert res == f"{gpu_link('ce')}:up"
    assert gbps == pytest.approx(31.508, abs=1e-3)


def test_bottleneck_finds_a_narrow_link() -> None:
    topo = build_topology({"0a": (16.0, 8)})
    mod = FabricModel(topo)
    res, gbps = mod.bottleneck("ce", "0a")
    assert res == f"{gpu_link('0a')}:down"
    assert gbps == pytest.approx(m.raw_gbs(16.0, 8))
    assert gbps == pytest.approx(15.754, abs=1e-3)
    # the narrow link limits only its own direction
    assert mod.bottleneck("0a", "ce")[1] == pytest.approx(15.754, abs=1e-3)
    assert mod.bottleneck("ce", "d3")[1] == pytest.approx(X16)


def test_link_without_known_rate_is_not_a_resource() -> None:
    topo = build_topology()
    link = topo.gpus["gpu:ce"].link
    assert link is not None
    topo.links[link].max_gbs = None
    assert FabricModel(topo).resources("ce", "d3") == (f"{gpu_link('d3')}:down",)


def test_errors(model: FabricModel) -> None:
    with pytest.raises(ValueError, match="different"):
        model.resources("ce", "gpu:ce")
    with pytest.raises(ValueError, match="different"):
        model.resources(m.HOST, m.HOST)
    with pytest.raises(KeyError):
        model.resources("ce", "ff")
    with pytest.raises(ValueError, match="unknown resource"):
        model.capacity("link:nope-nope:up")
    with pytest.raises(ValueError, match="unknown resource"):
        model.capacity(f"{gpu_link('ce')}:sideways")
    with pytest.raises(ValueError, match="unlimited"):
        model.capacity("host:read")


# --- hand-computed examples of the contract ------------------------------------


def test_one_cross_root_flow(model: FabricModel) -> None:
    assert model.ideal_rates([("ce", "0a")]) == [pytest.approx(31.508, abs=1e-3)]
    assert model.theoretical_aggregate([("ce", "0a")]) == pytest.approx(
        31.508, abs=1e-3
    )


def test_three_cross_root_flows_share_the_uplinks(model: FabricModel) -> None:
    flows = [("ce", "0a"), ("d3", "0f"), ("d8", "14")]
    rates = model.ideal_rates(flows)
    assert rates == [pytest.approx(10.503, abs=1e-3)] * 3
    assert model.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_four_ring_inside_one_switch(model: FabricModel) -> None:
    flows = [("ce", "d3"), ("d3", "d8"), ("d8", "db"), ("db", "ce")]
    assert model.ideal_rates(flows) == [pytest.approx(31.508, abs=1e-3)] * 4
    assert model.theoretical_aggregate(flows) == pytest.approx(126.03, abs=1e-2)


def test_all_to_all_inside_one_switch(model: FabricModel) -> None:
    flows = [(a, b) for a in A_GPUS for b in A_GPUS if a != b]
    assert len(flows) == 12
    # every GPU sources 3 and sinks 3 flows: X16 / 3 each, 4 * X16 in total
    assert model.ideal_rates(flows) == [pytest.approx(X16 / 3)] * 12
    assert model.theoretical_aggregate(flows) == pytest.approx(4 * X16)


def test_fan_in_is_limited_by_the_ingress_link(model: FabricModel) -> None:
    flows = [("d3", "ce"), ("d8", "ce"), ("db", "ce")]
    assert model.ideal_rates(flows) == [pytest.approx(10.503, abs=1e-3)] * 3
    assert model.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_fan_out_is_limited_by_the_egress_link(model: FabricModel) -> None:
    flows = [("ce", "d3"), ("ce", "d8"), ("ce", "db")]
    assert model.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_h2d_to_four_gpus_of_one_switch(model: FabricModel) -> None:
    flows = [(m.HOST, b) for b in A_GPUS]
    assert model.ideal_rates(flows) == [pytest.approx(7.877, abs=1e-3)] * 4
    assert model.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_h2d_to_one_gpu_on_each_switch(model: FabricModel) -> None:
    flows = [(m.HOST, "ce"), (m.HOST, "0a")]
    assert model.theoretical_aggregate(flows) == pytest.approx(63.016, abs=1e-3)


def test_d2h_to_four_gpus_of_one_switch(model: FabricModel) -> None:
    flows = [(b, m.HOST) for b in B_GPUS]
    assert model.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_bidirectional_pair_does_not_share(model: FabricModel) -> None:
    for a, b in (("ce", "d3"), ("ce", "0a")):
        flows = [(a, b), (b, a)]
        assert model.ideal_rates(flows) == [pytest.approx(31.508, abs=1e-3)] * 2
        assert model.theoretical_aggregate(flows) == pytest.approx(63.016, abs=1e-3)


def test_h2d_and_d2h_of_one_gpu_do_not_share(model: FabricModel) -> None:
    flows = [(m.HOST, "ce"), ("ce", m.HOST)]
    assert model.theoretical_aggregate(flows) == pytest.approx(63.016, abs=1e-3)


# --- progressive filling --------------------------------------------------------


def test_max_min_gives_the_unused_share_to_the_other_flows() -> None:
    # 0a sits on an x8 link (15.754 GB/s): the two flows into it get 7.877
    # each; the flow d3 -> 0f takes the rest of the uplink of switch A
    # (31.508 - 15.754 = 15.754), which is below its own GPU link.
    mod = FabricModel(build_topology({"0a": (16.0, 8)}))
    rates = mod.ideal_rates([("ce", "0a"), ("d8", "0a"), ("d3", "0f")])
    half = m.raw_gbs(16.0, 8)
    assert rates == [
        pytest.approx(half / 2),
        pytest.approx(half / 2),
        pytest.approx(X16 - half),
    ]
    assert sum(rates) == pytest.approx(X16)


def test_rates_do_not_depend_on_the_flow_order(model: FabricModel) -> None:
    flows = [
        ("ce", "0a"),
        ("d3", "0f"),
        ("d8", "ce"),
        ("db", "ce"),
        (m.HOST, "d3"),
        ("0a", m.HOST),
        ("14", "17"),
    ]
    forward = model.ideal_rates(flows)
    backward = model.ideal_rates(flows[::-1])[::-1]
    assert forward == backward  # exactly equal: deterministic arithmetic
    assert model.ideal_rates(flows) == forward


def test_empty_and_repeated_flows(model: FabricModel) -> None:
    assert model.ideal_rates([]) == []
    assert model.theoretical_aggregate([]) == 0.0
    # the same pair twice is two flows that split the links
    assert model.ideal_rates([("ce", "d3")] * 2) == [pytest.approx(X16 / 2)] * 2


def test_many_flows_terminate_with_a_full_allocation(model: FabricModel) -> None:
    everyone = A_GPUS + B_GPUS
    flows = [(a, b) for a in everyone for b in everyone if a != b]
    rates = model.ideal_rates(flows)
    assert len(rates) == 56
    assert all(math.isfinite(r) and r > 0 for r in rates)
    # no resource is over-subscribed
    used: dict[str, float] = {}
    for (s, d), rate in zip(flows, rates, strict=True):
        for res in model.resources(s, d):
            used[res] = used.get(res, 0.0) + rate
    assert all(v <= model.capacity(r) * (1 + 1e-9) for r, v in used.items())
    # each cross-root direction is bounded by one uplink
    assert used["link:c0:03.1-c5:00.0:up"] == pytest.approx(X16)
    assert used["link:00:01.1-01:00.0:down"] == pytest.approx(X16)


# --- host side ------------------------------------------------------------------


def test_host_dram_limit_applies_when_given() -> None:
    mod = FabricModel(build_topology(), host_gbps=10.0)
    assert mod.resources(m.HOST, "ce")[0] == "host:read"
    assert mod.resources("ce", m.HOST)[-1] == "host:write"
    assert mod.capacity("host:read") == 10.0
    assert mod.capacity("host:write") == 10.0
    assert mod.bottleneck(m.HOST, "ce") == ("host:read", 10.0)
    flows = [(m.HOST, b) for b in A_GPUS]
    assert mod.theoretical_aggregate(flows) == pytest.approx(10.0)
    # reads and writes are separate resources
    both = [(m.HOST, "ce"), ("d3", m.HOST)]
    assert mod.ideal_rates(both) == [pytest.approx(10.0)] * 2
    # GPU to GPU flows never touch the host
    assert all(not r.startswith("host:") for r in mod.resources("ce", "0a"))


def test_host_dram_is_unlimited_by_default(model: FabricModel) -> None:
    assert all(not r.startswith("host:") for r in model.resources(m.HOST, "ce"))
    assert all(not r.startswith("host:") for r in model.resources("ce", m.HOST))


# --- ceiling --------------------------------------------------------------------


def test_ceiling_uses_the_core_constants(model: FabricModel) -> None:
    expected = X16 * 256 / (256 + m.TLP_OVERHEAD_BYTES) * (1 - m.DLLP_OVERHEAD)
    assert model.ceiling(X16) == pytest.approx(expected)
    assert model.ceiling(X16) == pytest.approx(m.practical_gbs(16.0, 16))
    assert model.ceiling(X16) == pytest.approx(28.0, abs=0.05)
    assert model.ceiling(X16, mps=512) == pytest.approx(
        m.practical_gbs(16.0, 16, mps=512)
    )
    assert FabricModel.ceiling(0.0) == 0.0


# --- FabricModel on the real topology of par1-cs25 (conftest fixture fx_topo) -------
# Every GPU hangs off its switch through TWO physical x16 links in series: the
# card link (switch downstream port -> on-card bridge) and the on-card link
# (on-card bridge -> GPU). Both carry the GPU's whole traffic, so they limit
# every flow equally; the fabric links inside the switches have no lanes.

UP_A = "link:c0:03.1-c5:00.0"  # root port c0 <-> switch A upstream port
UP_B = "link:00:01.1-01:00.0"  # root port 00 <-> switch B upstream port
CARD = {  # bus -> (card link, on-card link)
    "ce": ("link:c8:10.0-cc:00.0", "link:cd:00.0-ce:00.0"),
    "d3": ("link:d0:00.0-d1:00.0", "link:d2:00.0-d3:00.0"),
    "d8": ("link:d5:00.0-d6:00.0", "link:d7:00.0-d8:00.0"),
    "db": ("link:d5:10.0-d9:00.0", "link:da:00.0-db:00.0"),
    "cb": ("link:c8:00.0-c9:00.0", "link:ca:00.0-cb:00.0"),  # W7800
    "0a": ("link:04:10.0-08:00.0", "link:09:00.0-0a:00.0"),
    "0f": ("link:0c:00.0-0d:00.0", "link:0e:00.0-0f:00.0"),
    "14": ("link:11:00.0-12:00.0", "link:13:00.0-14:00.0"),
    "17": ("link:11:10.0-15:00.0", "link:16:00.0-17:00.0"),
    "07": ("link:04:00.0-05:00.0", "link:06:00.0-07:00.0"),  # W7800
}
REAL_A = ("ce", "d3", "d8", "db")
REAL_B = ("0a", "0f", "14", "17")


@pytest.fixture
def real(fx_topo: m.Topology) -> FabricModel:
    return FabricModel(fx_topo)


def test_real_cards_have_the_pinned_links(fx_topo: m.Topology) -> None:
    for bus, (card, onboard) in CARD.items():
        assert fx_topo.gpus[m.gpu_entity(bus)].link == card
        assert fx_topo.links[onboard].physical
        assert fx_topo.links[onboard].down == f"{bus}:00.0"
    assert fx_topo.gpus["gpu:cb"].slot is None  # W7800: in no lease slot
    assert fx_topo.gpus["gpu:07"].slot is None
    assert fx_topo.gpus["gpu:ce"].slot == "A"
    assert fx_topo.gpus["gpu:0a"].slot == "B"


def test_real_cross_root_flow_resources(real: FabricModel) -> None:
    # ce (switch A) -> 0a (switch B): up out of ce, out of switch A, down into
    # switch B, down into 0a; the pseudo hop between the roots is no resource.
    card_ce, board_ce = CARD["ce"]
    card_0a, board_0a = CARD["0a"]
    assert real.resources("ce", "0a") == (
        f"{board_ce}:up",
        f"{card_ce}:up",
        f"{UP_A}:up",
        f"{UP_B}:down",
        f"{card_0a}:down",
        f"{board_0a}:down",
    )
    assert real.resources("gpu:ce", "gpu:0a") == real.resources("ce", "0a")


def test_real_cross_root_bottleneck_and_ceiling(real: FabricModel) -> None:
    res, gbps = real.bottleneck("ce", "0a")
    assert res == f"{CARD['ce'][1]}:up"  # first of six equal 31.508 resources
    assert gbps == pytest.approx(31.508, abs=1e-3)
    # Core stores Link.max_gbs rounded to 3 decimals: exactly 31.508
    assert all(real.capacity(r) == 31.508 for r in real.resources("ce", "0a"))
    assert real.ideal_rates([("ce", "0a")]) == [pytest.approx(31.508, abs=1e-3)]
    assert real.theoretical_aggregate([("ce", "0a")]) == pytest.approx(31.508, abs=1e-3)
    # practical posted-write ceiling of the 31.508 GB/s link: 28.0 GB/s
    assert real.ceiling(gbps) == pytest.approx(28.0, abs=0.05)


@pytest.mark.parametrize("pairs", [3, 4])
def test_real_concurrent_cross_root_pairs_share_the_uplinks(
    real: FabricModel, pairs: int
) -> None:
    flows = list(zip(REAL_A[:pairs], REAL_B[:pairs], strict=True))
    rates = real.ideal_rates(flows)
    assert rates == [pytest.approx(31.508 / pairs, abs=1e-3)] * pairs
    assert rates[0] == pytest.approx(31.508 / pairs)  # exact: 31.508 is the capacity
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)
    # the shared resources are exactly the two uplinks of the switches
    shared = set.intersection(*(set(real.resources(a, b)) for a, b in flows))
    assert shared == {f"{UP_A}:up", f"{UP_B}:down"}


def test_real_ring_of_four_inside_slot_b(real: FabricModel) -> None:
    ring = list(zip(REAL_B, REAL_B[1:] + REAL_B[:1], strict=True))
    assert ring == [("0a", "0f"), ("0f", "14"), ("14", "17"), ("17", "0a")]
    assert real.ideal_rates(ring) == [pytest.approx(31.508, abs=1e-3)] * 4
    assert real.theoretical_aggregate(ring) == pytest.approx(126.03, abs=1e-2)
    # an intra-switch flow touches the two cards only, never an uplink
    res = real.resources("0a", "0f")
    assert res == (
        f"{CARD['0a'][1]}:up",
        f"{CARD['0a'][0]}:up",
        f"{CARD['0f'][0]}:down",
        f"{CARD['0f'][1]}:down",
    )


def test_real_fan_in_of_three_is_limited_by_one_ingress_link(real: FabricModel) -> None:
    flows = [("0f", "0a"), ("14", "0a"), ("17", "0a")]
    assert real.ideal_rates(flows) == [pytest.approx(10.503, abs=1e-3)] * 3
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)
    shared = set.intersection(*(set(real.resources(a, b)) for a, b in flows))
    assert shared == {f"{CARD['0a'][0]}:down", f"{CARD['0a'][1]}:down"}


def test_real_h2d_to_four_gpus_of_one_switch(real: FabricModel) -> None:
    flows = [(m.HOST, b) for b in REAL_B]
    assert real.ideal_rates(flows) == [pytest.approx(7.877, abs=1e-3)] * 4
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)
    assert real.resources(m.HOST, "0a") == (
        f"{UP_B}:down",
        f"{CARD['0a'][0]}:down",
        f"{CARD['0a'][1]}:down",
    )
    # one GPU on each switch: two uplinks, 63.016 in total
    both = [(m.HOST, "ce"), (m.HOST, "0a")]
    assert real.theoretical_aggregate(both) == pytest.approx(63.016, abs=1e-3)


def test_real_w7800_cb_on_switch_a(real: FabricModel) -> None:
    # cb shares switch A (and its uplink) with the V620 ce d3 d8 db
    assert real.resources("ce", "cb") == (
        f"{CARD['ce'][1]}:up",
        f"{CARD['ce'][0]}:up",
        f"{CARD['cb'][0]}:down",
        f"{CARD['cb'][1]}:down",
    )
    assert real.theoretical_aggregate([("ce", "cb")]) == pytest.approx(31.508, abs=1e-3)
    # H2D to all five GPUs of switch A, cb included: one uplink, 31.508 in total
    flows = [(m.HOST, b) for b in ("cb", *REAL_A)]
    assert real.ideal_rates(flows) == [pytest.approx(31.508 / 5)] * 5
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_real_w7800_07_on_switch_b(real: FabricModel) -> None:
    assert real.resources("07", m.HOST) == (
        f"{CARD['07'][1]}:up",
        f"{CARD['07'][0]}:up",
        f"{UP_B}:up",
    )
    # D2H of 07 and 0a share the uplink of switch B toward the host
    flows = [("07", m.HOST), ("0a", m.HOST)]
    assert real.ideal_rates(flows) == [pytest.approx(31.508 / 2)] * 2
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)


def test_real_w7800_cross_root_pair_and_mixed_concurrency(real: FabricModel) -> None:
    assert real.resources("cb", "07") == (
        f"{CARD['cb'][1]}:up",
        f"{CARD['cb'][0]}:up",
        f"{UP_A}:up",
        f"{UP_B}:down",
        f"{CARD['07'][0]}:down",
        f"{CARD['07'][1]}:down",
    )
    assert real.theoretical_aggregate([("cb", "07")]) == pytest.approx(31.508, abs=1e-3)
    # W7800 and V620 pairs mixed: three concurrent cross-root pairs, 31.508 total
    flows = [("cb", "07"), ("ce", "0a"), ("d3", "0f")]
    assert real.ideal_rates(flows) == [pytest.approx(10.503, abs=1e-3)] * 3
    assert real.theoretical_aggregate(flows) == pytest.approx(31.508, abs=1e-3)
    # the reverse direction does not share a resource with the forward one
    both = [("cb", "07"), ("07", "cb")]
    assert real.theoretical_aggregate(both) == pytest.approx(63.016, abs=1e-3)


# --- interim layout (since 2026-11-08): six V620 behind ONE switch, x8 GPU links ----
# Real capture of 2026-11-10 (``fx_interim_topo``): every GPU link is Gen4 x8
# (15.754 GB/s raw per direction), the host uplink of the switch is x16 and shared.

X8 = m.raw_gbs(16.0, 8)  # 15.754
LEAF_A = ("gpu:d1", "gpu:d4")  # Gpu.slot A
LEAF_B = ("gpu:d9", "gpu:dc", "gpu:df", "gpu:e2")  # Gpu.slot B


@pytest.fixture
def interim(fx_interim_topo: m.Topology) -> FabricModel:
    return FabricModel(fx_interim_topo)


def test_interim_x8_is_half_of_x16() -> None:
    assert pytest.approx(X16 / 2, abs=1e-3) == X8
    assert pytest.approx(15.754, abs=1e-3) == X8


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("gpu:d9", "gpu:dc"),  # same leaf (slot B)
        ("gpu:d1", "gpu:d4"),  # same leaf (slot A)
        ("gpu:d4", "gpu:d9"),  # cross leaf, one switch
        (m.HOST, "gpu:d9"),
        ("gpu:d9", m.HOST),
        (m.HOST, "gpu:d1"),
    ],
)
def test_interim_every_route_is_limited_by_an_x8_link(
    interim: FabricModel, src: str, dst: str
) -> None:
    resource, gbps = interim.bottleneck(src, dst)
    assert gbps == pytest.approx(X8, abs=1e-3)
    assert resource in interim.resources(src, dst)
    assert interim.capacity(resource) == gbps
    assert interim.ideal_rates([(src, dst)]) == [pytest.approx(X8, abs=1e-3)]
    assert interim.theoretical_aggregate([(src, dst)]) == pytest.approx(X8, abs=1e-3)
    # nothing on the route is narrower than the x8 GPU link
    assert min(interim.capacity(r) for r in interim.resources(src, dst)) == gbps


def test_interim_host_uplink_is_one_shared_x16_link(
    interim: FabricModel, fx_interim_topo: m.Topology
) -> None:
    flows = [(m.HOST, g) for g in LEAF_B]
    uplink = interim.resources(m.HOST, "gpu:d9")[0]
    assert interim.capacity(uplink) == pytest.approx(X16, abs=1e-3)
    # every host flow of every GPU starts with that very link
    every = [interim.resources(m.HOST, g.id)[0] for g in fx_interim_topo.gpu_list()]
    assert set(every) == {uplink}
    # four GPUs at once: bounded by the x16 uplink, shared four ways
    rates = interim.ideal_rates(flows)
    assert rates == [pytest.approx(X16 / 4, abs=1e-3)] * 4
    assert all(r < X8 for r in rates)
    agg = interim.theoretical_aggregate(flows)
    assert agg == pytest.approx(X16, abs=1e-3)
    assert agg < 4 * X8 - 1.0  # four x8 links alone would allow 63.016


def test_interim_host_flows_x8_vs_x16_regimes(interim: FabricModel) -> None:
    two = [(m.HOST, "gpu:d1"), (m.HOST, "gpu:d9")]
    assert interim.ideal_rates(two) == [pytest.approx(X8, abs=1e-3)] * 2
    three = [(m.HOST, g) for g in ("gpu:d1", "gpu:d9", "gpu:df")]
    assert interim.ideal_rates(three) == [pytest.approx(X16 / 3, abs=1e-3)] * 3
    # the same holds toward the host
    out = [(g, m.HOST) for g in ("gpu:d1", "gpu:d4", "gpu:d9", "gpu:dc", "gpu:df")]
    assert interim.theoretical_aggregate(out) == pytest.approx(X16, abs=1e-3)


def test_interim_p2p_flows_do_not_share_the_uplink(interim: FabricModel) -> None:
    # d1>d9 and d4>dc cross the leaves yet stay below the uplink: x8 each
    cross = [("gpu:d1", "gpu:d9"), ("gpu:d4", "gpu:dc")]
    assert interim.ideal_rates(cross) == [pytest.approx(X8, abs=1e-3)] * 2
    ring = [("gpu:d9", "gpu:dc"), ("gpu:dc", "gpu:df")]
    ring += [("gpu:df", "gpu:e2"), ("gpu:e2", "gpu:d9")]
    assert interim.ideal_rates(ring) == [pytest.approx(X8, abs=1e-3)] * 4
    assert interim.theoretical_aggregate(ring) == pytest.approx(4 * X8, abs=1e-3)
    # the uplink is not on a GPU to GPU route
    uplink = interim.resources(m.HOST, "gpu:d9")[0]
    assert uplink not in interim.resources("gpu:d4", "gpu:d9")
