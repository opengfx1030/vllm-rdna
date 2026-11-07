# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Topology discovery against the captured par1-cs25 sysfs."""

from __future__ import annotations

import collections
import shutil

import pytest
from pciescope import model, topology
from pciescope.sysfs import Sysfs

KFD_ORDER = ["cb", "ce", "d3", "d8", "db", "07", "0a", "0f", "14", "17"]
SMI_ORDER = ["07", "0a", "0f", "14", "17", "cb", "ce", "d3", "d8", "db"]


def test_counts(fx_topo):
    t = fx_topo
    assert len(t.nodes) == 70
    assert len(t.links) == 32
    assert len(t.gpus) == 10
    assert sorted(t.switches) == ["switch:A", "switch:B"]
    assert sorted(t.roots) == ["root:00", "root:c0"]
    kinds = collections.Counter(lk.kind for lk in t.links.values())
    assert kinds == {
        model.LINK_GPU: 10,
        model.LINK_ONBOARD: 10,
        model.LINK_FABRIC: 8,
        model.LINK_HOST_UPLINK: 2,
        model.LINK_MGMT: 2,
    }
    physical = [lk for lk in t.links.values() if lk.physical]
    assert len(physical) == 22
    assert all(lk.max_speed_gts == 16.0 and lk.max_width == 16 for lk in physical)
    assert t.hip_source == "kfd-node-order"
    assert t.notes == []


def test_hip_and_smi_order(fx_topo):
    assert [g.bus for g in fx_topo.gpu_list("hip")] == KFD_ORDER
    assert [g.hip for g in fx_topo.gpu_list("hip")] == list(range(10))
    assert [g.bus for g in fx_topo.gpu_list("smi")] == SMI_ORDER
    assert [g.kfd_node for g in fx_topo.gpu_list("hip")] == list(range(1, 11))
    assert all(g.kfd_gpu_id for g in fx_topo.gpus.values())


def test_gpu_identity(fx_topo):
    w7800 = [g for g in fx_topo.gpus.values() if g.arch == "gfx1100"]
    v620 = [g for g in fx_topo.gpus.values() if g.arch == "gfx1030"]
    assert sorted(g.bus for g in w7800) == ["07", "cb"]
    assert len(v620) == 8
    assert all(g.slot is None for g in w7800)
    assert {g.slot for g in v620} == {"A", "B"}
    assert all(g.slot == g.switch for g in v620)
    assert fx_topo.gpu("gpu:cb").hip == 0  # box-rules table: HIP 0 = W7800 on switch A
    assert fx_topo.gpu("gpu:07").hip == 5  # HIP 5 = W7800 on switch B
    ce = fx_topo.gpu("gpu:ce")
    assert ce.name == "AMD RADEON PRO V620 Azure"
    assert ce.port == "c8:10.0"
    assert ce.link == "link:c8:10.0-cc:00.0"
    assert ce.vram_total_bytes == 32195477504
    assert ce.device_id == 0x73A1
    assert fx_topo.gpu("gpu:cb").device_id == 0x7449


def test_switch_structure(fx_topo):
    a, b = fx_topo.switches["switch:A"], fx_topo.switches["switch:B"]
    assert (a.root, a.root_port, a.upstream) == ("c0", "c0:03.1", "c5:00.0")
    assert (b.root, b.root_port, b.upstream) == ("00", "00:01.1", "01:00.0")
    assert a.gpus == ["gpu:cb", "gpu:ce", "gpu:d3", "gpu:d8", "gpu:db"]
    assert b.gpus == ["gpu:07", "gpu:0a", "gpu:0f", "gpu:14", "gpu:17"]
    assert a.mgmt == "e0:00.0"
    assert b.mgmt == "1c:00.0"
    # 5 cards + 2 unused downstream ports; the management (GEP) port is fabric
    assert a.ports == [
        "c8:00.0",
        "c8:10.0",
        "d0:00.0",
        "d5:00.0",
        "d5:10.0",
        "dd:14.0",
        "dd:15.0",
    ]
    assert len(b.ports) == 7
    assert fx_topo.nodes["c6:1c.0"].role == model.ROLE_SWITCH_FABRIC
    assert fx_topo.nodes["c6:1c.0"].children == ["e0:00.0"]
    assert "c6:1c.0" not in a.ports
    # the two unused downstream ports of switch A have no link partner
    unused = [
        n
        for n in fx_topo.nodes.values()
        if n.role == model.ROLE_SWITCH_DOWN and not n.children
    ]
    assert sorted(n.bdf for n in unused if n.switch == "A") == ["dd:14.0", "dd:15.0"]


def test_roles_follow_port_types(fx_topo):
    n = fx_topo.nodes
    assert n["c0:03.1"].role == model.ROLE_ROOT_PORT
    assert n["c0:03.1"].port_type == model.PORT_ROOT
    assert n["c5:00.0"].role == model.ROLE_SWITCH_UP
    assert n["c5:00.0"].port_type == model.PORT_USP
    assert n["c6:00.0"].role == model.ROLE_SWITCH_FABRIC
    assert n["c7:00.0"].role == model.ROLE_SWITCH_SUB
    assert n["c8:10.0"].role == model.ROLE_SWITCH_DOWN
    assert n["cc:00.0"].role == model.ROLE_GPU_UP
    assert n["cd:00.0"].role == model.ROLE_GPU_DOWN
    assert n["ce:00.0"].role == model.ROLE_GPU
    assert n["ce:00.0"].entity == "gpu:ce"
    assert n["c8:10.0"].entity == "port:c8:10.0"
    assert n["e0:00.0"].role == model.ROLE_SWITCH_MGMT


def test_links_of_one_card(fx_topo):
    lk = fx_topo.links["link:c8:10.0-cc:00.0"]
    assert lk.kind == model.LINK_GPU
    assert lk.physical
    assert (lk.gpu, lk.switch) == ("gpu:ce", "A")
    assert lk.max_gbs == pytest.approx(31.508, abs=1e-3)
    onboard = fx_topo.links["link:cd:00.0-ce:00.0"]
    assert onboard.kind == model.LINK_ONBOARD
    fabric = fx_topo.links["link:c6:00.0-c7:00.0"]
    assert fabric.kind == model.LINK_FABRIC
    assert not fabric.physical
    up = fx_topo.links["link:c0:03.1-c5:00.0"]
    assert up.kind == model.LINK_HOST_UPLINK
    assert fx_topo.link_above("c5:00.0") is up


def test_render_tree_and_summary(fx_topo):
    text = topology.render_tree(fx_topo)
    assert "par1-cs25: 2 root complex(es), 2 switch(es), 10 GPU(s)" in text
    assert "root complex c0 (switch A)" in text
    assert "root complex 00 (switch B)" in text
    assert (
        "gpu:ce AMD RADEON PRO V620 Azure gfx1030 HIP 1 smi 6 kfd-node 2 slot A" in text
    )
    assert "[unused: down]" in text
    full = topology.render_tree(fx_topo, full=True)
    assert len(full) > len(text)
    s = topology.summary(fx_topo)
    assert s["hip_source"] == "kfd-node-order"
    assert len(s["gpus"]) == 10


def test_discover_empty_tree_reports_note(tmp_path):
    t = topology.discover(Sysfs(tmp_path), host="nowhere")
    assert not t.gpus
    assert t.notes == ["no AMD GPU found in sysfs"]


def test_hip_order_mismatch_is_noted(fx_root, tmp_path):
    """A KFD order that disagrees with the box-rules table is flagged."""
    sys_copy = tmp_path / "sys"
    shutil.copytree(fx_root / "sys", sys_copy, symlinks=True)
    nodes = sys_copy / "class/kfd/kfd/topology/nodes"
    p1, p2 = nodes / "1" / "properties", nodes / "2" / "properties"
    first, second = p1.read_text(), p2.read_text()
    p1.write_text(second)
    p2.write_text(first)
    t = topology.discover(Sysfs(sys_copy), host="par1-cs25")
    assert [g.bus for g in t.gpu_list("hip")][:2] == ["ce", "cb"]
    assert any("differs from the box rules table" in n for n in t.notes)
    # the same tree under another host name is not compared with the table
    other = topology.discover(Sysfs(sys_copy), host="elsewhere")
    assert other.notes == []


def test_no_kfd_means_no_hip(fx_root, tmp_path):
    sys_copy = tmp_path / "sys"
    shutil.copytree(fx_root / "sys", sys_copy, symlinks=True)
    shutil.rmtree(sys_copy / "class/kfd")
    t = topology.discover(Sysfs(sys_copy), host="par1-cs25")
    assert len(t.gpus) == 10
    assert all(g.hip is None for g in t.gpus.values())
    assert t.hip_source.startswith("unavailable")
    assert any("HIP indices unknown" in n for n in t.notes)


def test_links_ordered_follows_the_hip_order(fx_topo):
    phys = [lk.id for lk in fx_topo.links_ordered(physical_only=True)]
    assert len(phys) == 22
    assert phys[:2] == ["link:c0:03.1-c5:00.0", "link:00:01.1-01:00.0"]  # uplinks A, B
    # per GPU in HIP order: the switch-to-card link, then the on-card link
    assert phys[2:6] == [
        "link:c8:00.0-c9:00.0",  # HIP 0 (cb)
        "link:ca:00.0-cb:00.0",
        "link:c8:10.0-cc:00.0",  # HIP 1 (ce)
        "link:cd:00.0-ce:00.0",
    ]
    assert phys[-4:] == [
        "link:11:00.0-12:00.0",  # HIP 8 (14)
        "link:13:00.0-14:00.0",
        "link:11:10.0-15:00.0",  # HIP 9 (17)
        "link:16:00.0-17:00.0",
    ]
    everything = fx_topo.links_ordered()
    assert [lk.id for lk in everything[:22]] == phys
    assert len(everything) == 32
    assert {lk.kind for lk in everything[22:]} == {"fabric", "mgmt"}
    assert [lk.kind for lk in everything[-2:]] == ["mgmt", "mgmt"]
