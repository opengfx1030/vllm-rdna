# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEX port number <-> BDF <-> GPU map of the two PEX88096 switches (par1-cs25)."""

from __future__ import annotations

import plx_fakes
import pytest
from pciescope.model import Topology
from pciescope.plx import ports as plx_ports
from pciescope.plx import regs

#: (port, bdf, kind, peer) read from LnkCap[31:24] of every function on the box
SWITCH_A = [
    (0, "c8:00.0", "dsp", "gpu:cb"),
    (16, "c8:10.0", "dsp", "gpu:ce"),
    (32, "d0:00.0", "dsp", "gpu:d3"),
    (48, "c5:00.0", "usp", "host"),
    (64, "d5:00.0", "dsp", "gpu:d8"),
    (80, "d5:10.0", "dsp", "gpu:db"),
    (116, "dd:14.0", "dsp", None),
    (117, "dd:15.0", "dsp", None),
    (224, "c6:00.0", "int_dsp", None),
    (225, "c6:04.0", "int_dsp", None),
    (226, "c6:08.0", "int_dsp", None),
    (227, "c6:0c.0", "int_dsp", None),
    (232, "c7:00.0", "int_usp", None),
    (233, "cf:00.0", "int_usp", None),
    (234, "d4:00.0", "int_usp", None),
    (235, "dc:00.0", "int_usp", None),
    (247, "e0:00.0", "mgmt", None),
    (255, "c6:1c.0", "gep_dsp", None),
]
SWITCH_B = [
    (0, "04:00.0", "dsp", "gpu:07"),
    (16, "04:10.0", "dsp", "gpu:0a"),
    (32, "0c:00.0", "dsp", "gpu:0f"),
    (48, "01:00.0", "usp", "host"),
    (64, "11:00.0", "dsp", "gpu:14"),
    (80, "11:10.0", "dsp", "gpu:17"),
    (116, "19:14.0", "dsp", None),
    (117, "19:15.0", "dsp", None),
    (224, "02:00.0", "int_dsp", None),
    (225, "02:04.0", "int_dsp", None),
    (226, "02:08.0", "int_dsp", None),
    (227, "02:0c.0", "int_dsp", None),
    (232, "03:00.0", "int_usp", None),
    (233, "0b:00.0", "int_usp", None),
    (234, "10:00.0", "int_usp", None),
    (235, "18:00.0", "int_usp", None),
    (247, "1c:00.0", "mgmt", None),
    (255, "02:1c.0", "gep_dsp", None),
]


@pytest.fixture(scope="module")
def maps():
    topo, chips, _ = plx_fakes.load_capture()
    return topo, chips, plx_ports.build_port_maps(topo)


def test_two_switches_lettered_like_the_brief(maps):
    _, _, cms = maps
    assert [(m.switch, m.root_bus, m.usp_bdf) for m in cms] == [
        ("A", "c0", "c5:00.0"),
        ("B", "00", "01:00.0"),
    ]


@pytest.mark.parametrize(("letter", "expect"), [("A", SWITCH_A), ("B", SWITCH_B)])
def test_port_table_matches_hardware(maps, letter, expect):
    _, _, cms = maps
    cm = next(m for m in cms if m.switch == letter)
    got = [(p.port, p.bdf, p.kind, p.peer) for p in cm.ports]
    assert got == expect


def test_rule_agrees_with_lnkcap_port_numbers(maps):
    """The port numbers derived from the BDF structure equal LnkCap[31:24]."""
    _, chips, cms = maps
    from pciescope.plx import pcie

    for cm in cms:
        wins = chips[f"0000:{cm.usp_bdf}"]
        for p in cm.ports:
            win = wins[p.port]
            st = pcie.decode_static(lambda o, w=win: w.get(o, 0))
            assert st.port_number == p.port, p


def test_station_and_physical_flags(maps):
    _, _, cms = maps
    for cm in cms:
        by = cm.by_port()
        for port in (0, 16, 32, 48, 64, 80):
            p = by[port]
            assert p.physical and p.station == port // 16 and p.station_port == 0
        for port in (116, 117):  # empty x1 ports: station 7, no perf counters
            assert by[port].physical and by[port].station == 7
        for port in (224, 232, 247, 255):
            assert not by[port].physical and by[port].station is None
        assert [p.port for p in cm.physical_ports()] == [
            0,
            16,
            32,
            48,
            64,
            80,
            116,
            117,
        ]


def test_uplink_peer_is_the_root_port(maps):
    _, _, cms = maps
    assert {m.switch: m.by_port()[48].peer_bdf for m in cms} == {
        "A": "c0:03.1",
        "B": "00:01.1",
    }


def test_headers_of_the_capture_verify(maps):
    topo, chips, cms = maps
    for cm in cms:
        wins = chips[f"0000:{cm.usp_bdf}"]
        headers = {
            p.port: (wins[p.port].get(0x08, 0), wins[p.port].get(0x18, 0))
            for p in cm.ports
        }
        assert plx_ports.verify_headers(cm, topo, headers) == []


def test_header_mismatch_is_reported(maps):
    topo, chips, cms = maps
    cm = cms[1]
    wins = chips[f"0000:{cm.usp_bdf}"]
    headers = {
        p.port: (wins[p.port].get(0x08, 0), wins[p.port].get(0x18, 0)) for p in cm.ports
    }
    cls_dw, bus_dw = headers[16]
    headers[16] = (cls_dw, bus_dw ^ 0x0001_0000)  # wrong subordinate bus
    headers[32] = (0x0300_0000, headers[32][1])  # wrong class
    del headers[64]
    problems = plx_ports.verify_headers(cm, topo, headers)
    assert len(problems) == 3
    assert any("port 16" in p and "buses" in p for p in problems)
    assert any("port 32" in p and "class" in p for p in problems)
    assert any("port 64" in p and "no header" in p for p in problems)


def test_no_switch_in_a_topology_without_broadcom():
    assert plx_ports.build_port_maps(Topology(host="x", captured_ns=0)) == []


def test_device_number_rule_for_the_leaf_ports():
    # leaf port = 32 * sub-switch index + device number on its bus
    assert 32 * 0 + 0x00 == 0 and 32 * 0 + 0x10 == 16
    assert 32 * 1 + 0x00 == 32
    assert 32 * 2 + 0x00 == 64 and 32 * 2 + 0x10 == 80
    assert 32 * 3 + 0x14 == 116 and 32 * 3 + 0x15 == 117
    assert regs.station_of(116) == 7
