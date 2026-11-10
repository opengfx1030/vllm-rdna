# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PEX88096 of the interim layout (2026-11-10): one chip, six V620, no W7800.

``plx_par1_cs25_interim.json.gz`` is the read-only capture of the chip on the
SlimSAS card: serial ``00-80-5e-10-b7-3d-3b-62``, uplink ``c5:00.0`` = PEX port 0
(x16), the six V620 on the x8 ports 48 and 56 (leaf A, lease ``a2``, HIP 0-1) and
64, 72, 80, 88 (leaf B, lease ``b4``, HIP 2-5), empty slots on 16, 24, 32, 40.
The tests pin the port map, the chip registry and the policy data of that chip,
the leases of its GPUs, and drive the counters over the stateful fake: what no
test can say is whether the counters of the second port of a station (the x8
ports 56, 72, 88) read as the model assumes; that needs a switch window.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import plx_fakes
import pytest
from pciescope.collectors import plx as plx_collector
from pciescope.model import Topology, full_bdf, raw_gbs
from pciescope.plx import cli, fixture, health, helper, pcie, policy, preflight, regs
from pciescope.plx.access import AccessError, LocalClient
from pciescope.plx.ports import build_port_maps, uplink_port, verify_headers
from pciescope.sysfs import Sysfs

FIX = str(plx_fakes.INTERIM_CAPTURE)
CHIP = plx_fakes.INTERIM_SWITCH
SERIAL = "00-80-5e-10-b7-3d-3b-62"

#: (port, bdf, kind, peer) read from LnkCap[31:24] of every function on the box
PORT_TABLE = [
    (0, "c5:00.0", "usp", "host"),
    (16, "c8:10.0", "dsp", None),
    (24, "c8:18.0", "dsp", None),
    (32, "cc:00.0", "dsp", None),
    (40, "cc:08.0", "dsp", None),
    (48, "cc:10.0", "dsp", "gpu:d1"),
    (56, "cc:18.0", "dsp", "gpu:d4"),
    (64, "d6:00.0", "dsp", "gpu:d9"),
    (72, "d6:08.0", "dsp", "gpu:dc"),
    (80, "d6:10.0", "dsp", "gpu:df"),
    (88, "d6:18.0", "dsp", "gpu:e2"),
    (116, "e4:14.0", "dsp", None),
    (117, "e4:15.0", "dsp", None),
    (224, "c6:00.0", "int_dsp", None),
    (225, "c6:04.0", "int_dsp", None),
    (226, "c6:08.0", "int_dsp", None),
    (227, "c6:0c.0", "int_dsp", None),
    (232, "c7:00.0", "int_usp", None),
    (233, "cb:00.0", "int_usp", None),
    (234, "d5:00.0", "int_usp", None),
    (235, "e3:00.0", "int_usp", None),
    (247, "e7:00.0", "mgmt", None),
    (255, "c6:1c.0", "gep_dsp", None),
]
#: (port, bdf, peer, hip, smi, slot, lease key, station, port in the station); the
#: slot letter is the leaf (A = d1 d4, B = d9 dc df e2), not the chip
GPU_MAP = [
    (48, "cc:10.0", "gpu:d1", 0, 0, "A", "a2", 3, 0),
    (56, "cc:18.0", "gpu:d4", 1, 1, "A", "a2", 3, 8),
    (64, "d6:00.0", "gpu:d9", 2, 2, "B", "b4", 4, 0),
    (72, "d6:08.0", "gpu:dc", 3, 3, "B", "b4", 4, 8),
    (80, "d6:10.0", "gpu:df", 4, 4, "B", "b4", 5, 0),
    (88, "d6:18.0", "gpu:e2", 5, 5, "B", "b4", 5, 8),
]
GPU_PORTS = [row[0] for row in GPU_MAP]
#: the ports the counters would run on: the uplink and the six GPU ports
PERF_PORTS = [0, *GPU_PORTS]


@pytest.fixture(scope="module")
def capture():
    topo, chips, _ = plx_fakes.load_capture(plx_fakes.INTERIM_CAPTURE)
    return topo, chips


@pytest.fixture(scope="module")
def cmap(capture):
    (cm,) = build_port_maps(capture[0])
    return cm


@pytest.fixture(autouse=True)
def _no_env_policy(monkeypatch):
    monkeypatch.delenv(policy.ENV_WRITE_SWITCHES, raising=False)


def run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


# -- the port map -------------------------------------------------------------------


def test_the_capture_carries_no_layout_warning(capture):
    # the topology of the capture was built by the code that knows this layout
    assert list(capture[0].notes) == []


def test_one_chip_with_its_uplink_on_port_0(cmap):
    assert (cmap.switch, cmap.root_bus, cmap.usp_bdf) == ("A", "c0", "c5:00.0")
    assert cmap.usp_port == 0  # the original cards had it on port 48
    assert [(p.port, p.bdf, p.kind, p.peer) for p in cmap.ports] == PORT_TABLE
    usp = cmap.by_port()[0]
    assert (usp.station, usp.station_port, usp.peer_bdf) == (0, 0, "c0:03.1")
    assert cmap.by_port()[48].kind == "dsp"  # and port 48 is a GPU port now


def test_gpu_ports_carry_hip_smi_slot_and_lease(cmap):
    rows = [
        (
            p.port,
            p.bdf,
            p.peer,
            p.hip,
            p.smi,
            p.slot,
            p.lease,
            p.station,
            p.station_port,
        )
        for p in cmap.ports
        if p.peer and p.peer != "host"
    ]
    assert rows == GPU_MAP
    assert {p.gpu_name for p in cmap.ports if p.hip is not None} == {
        "AMD RADEON PRO V620 Azure"
    }
    others = [p for p in cmap.ports if p.hip is None]
    assert all(p.slot is None and p.lease is None for p in others)
    assert all(p.gpu_name is None for p in others)
    assert [p.port for p in cmap.physical_ports() if p.peer is None] == [
        16,
        24,
        32,
        40,
        116,
        117,
    ]  # empty slots have a link-less port, no GPU, no traffic


def test_rule_agrees_with_lnkcap_port_numbers(capture, cmap):
    """Leaf port = 32 * j + device holds for the x8 ports as well."""
    _, chips = capture
    for p in cmap.ports:
        win = chips[CHIP][p.port]
        st = pcie.decode_static(lambda o, w=win: w.get(o, 0))
        assert st.port_number == p.port, p


def test_headers_of_the_capture_verify(capture, cmap):
    topo, chips = capture
    wins = chips[CHIP]
    headers = {
        p.port: (wins[p.port].get(0x08, 0), wins[p.port].get(0x18, 0))
        for p in cmap.ports
    }
    assert verify_headers(cmap, topo, headers) == []
    # the original wiring's guess (uplink on port 48) would be caught: port 48's
    # window belongs to the GPU port cc:10.0, not to the USP c5:00.0
    wrong = {**headers, 0: headers[48]}
    assert any("port 0 (c5:00.0)" in p for p in verify_headers(cmap, topo, wrong))


@pytest.mark.parametrize(
    ("leaves", "want"),
    [
        ([0, 16, 32, 64, 80, 116, 117], 48),  # the original cards: station 3 is free
        ([16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 116, 117], 0),  # SlimSAS card
        ([16, 32], 48),  # several free stations: the original wiring wins
        ([48, 64, 80], 0),  # ... else the lowest
        ([0, 16, 32, 48, 64, 80], 48),  # none free: the default, the windows refuse it
    ],
)
def test_the_uplink_is_the_base_port_of_the_free_station(leaves, want):
    assert uplink_port(leaves) == want


def test_the_two_cards_of_2026_11_07_keep_their_uplink_on_port_48():
    topo, _, _ = plx_fakes.load_capture()
    assert {m.switch: m.usp_port for m in build_port_maps(topo)} == {"A": 48, "B": 48}


def test_capture_refuses_windows_that_disagree_with_the_map(capture):
    doc = fixture.read_doc(plx_fakes.INTERIM_CAPTURE)
    doc["chips"][CHIP]["windows"]["48"]["018"] = "00000000"  # the bus numbers
    topo, client = fixture.from_doc(doc)
    with pytest.raises(AccessError, match="topology and BAR0 windows disagree"):
        fixture.capture(topo, client, build_port_maps(topo))
    ok_topo, ok_client = fixture.load(plx_fakes.INTERIM_CAPTURE)
    redone = fixture.capture(ok_topo, ok_client, build_port_maps(ok_topo))
    assert redone["chips"][CHIP]["usp_port"] == 0
    assert sorted(map(int, redone["chips"][CHIP]["windows"])) == [
        p[0] for p in PORT_TABLE
    ]


# -- the registry of chips and the policy data -----------------------------------------


def _serial(windows, usp_port):
    return pcie.decode_static(lambda o: windows[usp_port].get(o, 0)).serial


def test_the_registry_knows_the_three_chips_the_box_has_shown():
    for path in (plx_fakes.CAPTURE, plx_fakes.INTERIM_CAPTURE):
        _, chips, doc = plx_fakes.load_capture(path)
        for bdf, wins in chips.items():
            c = doc["chips"][bdf]
            entry = regs.KNOWN_CHIPS[_serial(wins, c["usp_port"])]
            assert (entry.usp_bdf, entry.switch) == (bdf, c["switch"])
    assert len(regs.KNOWN_CHIPS) == 3
    assert SERIAL in regs.KNOWN_CHIPS


def test_the_installed_chip_is_known_by_serial_and_bdf_and_nothing_is_protected(
    capture,
):
    _, chips = capture
    chip, _ = plx_fakes.chip_access(chips[CHIP], CHIP, usp_port=0)
    assert chip.serial == SERIAL and chip.switch == "A"
    assert chip.known == regs.KNOWN_CHIPS[SERIAL]
    assert chip.protected_ports == frozenset()
    assert regs.KNOWN_CHIPS[SERIAL].protected == {}
    hello = chip.hello()
    assert (hello["serial"], hello["switch"], hello["usp_port"]) == (SERIAL, "A", 0)
    assert hello["protected_ports"] == []
    assert hello["enabled"] == [0, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 116, 117]


def test_a_chip_that_moved_or_changed_serial_is_read_only(capture):
    _, chips = capture
    win = chips[CHIP]
    plx_fakes.chip_access(win, CHIP, allow_writes=True, usp_port=0)  # known: fine
    moved, _ = plx_fakes.chip_access(win, "0000:c6:00.0", usp_port=0)
    assert moved.known is None and moved.switch is None and moved.serial == SERIAL
    assert moved.hello()["protected_ports"] == [] and not moved.allow_writes
    with pytest.raises(AccessError, match="not a known PEX88096"):
        plx_fakes.chip_access(win, "0000:c6:00.0", allow_writes=True, usp_port=0)
    other = copy.deepcopy(win)
    dsn = pcie.walk_caps(lambda o: win[0].get(o, 0)).ext[regs.ECAP_DSN]
    other[0][dsn + 4] ^= 1  # a different serial at the right place
    with pytest.raises(AccessError, match="serial 00-80-5e-10-b7-3d-3b-63"):
        plx_fakes.chip_access(other, CHIP, allow_writes=True, usp_port=0)


def test_no_switch_is_approved_and_no_gpu_is_excluded_until_the_user_says_so():
    assert policy.DEFAULT_WRITE_SWITCHES == () and policy.NEVER_WRITTEN_GPUS == ()
    assert policy.write_switches(None, {}) == frozenset()
    assert policy.write_switches(None, {policy.ENV_WRITE_SWITCHES: "A"}) == {"A"}


# -- the verbs over the capture ---------------------------------------------------


def test_ports_text_lists_the_map(capsys):
    rc, out, _ = run(capsys, "ports", "--fixture", FIX)
    assert rc == 0
    assert "switch A  root c0  upstream c5:00.0 (port 0)" in out
    assert "   0 0.0 c5:00.0  usp      host       -   -    - -" in out
    assert (
        "  48 3.0 cc:10.0  dsp      gpu:d1     0   0    A a2       "
        "AMD RADEON PRO V620 Azure"
    ) in out
    assert (
        "  88 5.8 d6:18.0  dsp      gpu:e2     5   5    B b4       "
        "AMD RADEON PRO V620 Azure"
    ) in out
    assert " 247   - e7:00.0  mgmt" in out


def test_status_shows_every_gpu_link_at_x8(capsys):
    rc, out, _ = run(capsys, "status", "--fixture", FIX, "--json")
    assert rc == 0
    doc = json.loads(out)
    assert doc["chips"] == [
        {
            "switch": "A",
            "usp_bdf": "c5:00.0",
            "serial": SERIAL,
            "perf_running": False,
            "perf_ports": [],
        }
    ]
    assert doc["events"] == []
    by_port = {p["port"]: p for p in doc["ports"]}
    assert (by_port[0]["state"]["speed_gts"], by_port[0]["state"]["width"]) == (
        16.0,
        16,
    )
    for port in GPU_PORTS:
        st = by_port[port]["state"]
        assert (st["speed_gts"], st["width"], st["link_up"]) == (16.0, 8, True)
        assert st["eq_complete"] is True and st["mps"] == 256
    for port in (16, 24, 32, 40, 116, 117):
        assert by_port[port]["state"]["link_up"] is False


def test_dump_of_the_uplink_names_its_capabilities(capsys):
    rc, out, _ = run(capsys, "dump", "0", "--fixture", FIX, "--decode", "--json")
    assert rc == 0
    doc = json.loads(out)
    assert doc["port"] == 0 and doc["bdf"] == "c5:00.0"
    assert doc["dwords"]["000"] == "c0101000"  # 1000:c010
    assert {"pcie", "aer", "dsn"} <= set(doc["caps"])


def test_perf_plan_covers_the_stations_of_the_gpu_ports_only(capsys):
    rc, out, _ = run(capsys, "perf", "plan", "--fixture", FIX, "--json")
    assert rc == 0  # a fixture client is read-only: any write would have raised
    doc = json.loads(out)["A"]
    assert doc["ports"] == PERF_PORTS  # nothing is planned for the empty slots
    steps = doc["steps"]

    def regs_at(name):
        return sorted(s["port"] for s in steps if s["step"] == name)

    assert regs_at("ingress_enable") == [0, 48, 64, 80]  # stations 0, 3, 4, 5
    assert regs_at("monitor_start") == [0, 48, 64, 80]
    assert regs_at("egress_enable") == PERF_PORTS
    assert {(s["reg"], s["old"], s["new"]) for s in steps} == {
        (0x768, 0x09000000, 0x29000000),
        (0xF30, 0, 0x00200040),
        (0x3E0, 0, 0xD8000000),
    }  # every counter is off on the box
    assert doc["monitor_already_running"] is False


def test_perf_plan_stages_the_ports(capsys):
    rc, out, _ = run(capsys, "perf", "plan", "--fixture", FIX, "--ports", "48,56")
    assert rc == 0
    assert "switch A: counters on ports [48, 56]" in out
    assert "  monitor_start       48 " in out and " 64 " not in out
    with pytest.raises(SystemExit, match="not measurable"):
        cli.main(["perf", "plan", "--fixture", FIX, "--ports", "16"])  # empty slot
    with pytest.raises(SystemExit, match="not measurable"):
        cli.main(["perf", "plan", "--fixture", FIX, "--ports", "224"])


def test_perf_plan_honours_an_exclusion(capsys):
    rc, out, _ = run(capsys, "perf", "plan", "--fixture", FIX, "--exclude", "gpu:e2")
    assert rc == 0 and "counters on ports [0, 48, 56, 64, 72, 80]" in out


# -- the counters over the stateful fake ------------------------------------------


def build(capture, *, allow_writes=False, write_switches=None, **kw):
    topo, windows = capture
    chip, fake = plx_fakes.chip_access(
        windows[CHIP], CHIP, allow_writes=allow_writes, usp_port=0
    )
    coll = plx_collector.PlxCollector(
        topo,
        client=LocalClient({CHIP: chip}),
        allow_writes=allow_writes,
        write_switches=write_switches,
        **kw,
    )
    return coll, fake, chip


def by_key(samples):
    return {(s.entity, s.metric): s for s in samples}


def test_read_only_samples_of_the_x8_links(capture):
    coll, fake, _ = build(capture)
    got = by_key(coll.collect())
    for _, bdf, *_rest in [r for r in GPU_MAP]:
        assert got[f"port:{bdf}", "plx.link_speed_gts"].value == 16.0
        assert got[f"port:{bdf}", "plx.link_width"].value == 8
        assert got[f"port:{bdf}", "plx.link_degraded"].value == 0.0
    assert got["port:c5:00.0", "plx.link_width"].value == 16
    assert got["port:c8:10.0", "plx.link_up"].value == 0.0  # an empty slot
    assert [e for e in coll.drain_events() if e.severity != "info"] == []
    assert fake.writes == []


def test_the_default_policy_writes_nothing_to_the_installed_chip(capture):
    coll, fake, _ = build(capture, allow_writes=True)
    coll.collect()
    (event,) = [e for e in coll.drain_events() if e.name == "plx_perf_skipped"]
    assert event.entity == "switch:A"
    assert "approved for switch none only" in event.message
    assert fake.writes == []
    coll.close()
    assert fake.writes == []


def _station(offset):
    return (offset - regs.PORT_WINDOW_BASE) // regs.PORT_WINDOW_SIZE // 16


def test_counters_touch_only_the_stations_of_the_ports_with_a_device(capture):
    coll, fake, chip = build(capture, allow_writes=True, write_switches=("A",))
    coll.collect()
    (start,) = [e for e in coll.drain_events() if e.name == "plx_perf_start"]
    assert start.data["ports"] == PERF_PORTS
    assert {_station(off) for off, _ in fake.writes} == {0, 3, 4, 5}
    chip.close()  # the injected client is not the collector's: undo through the chip
    assert {_station(off) for off, _ in fake.writes} == {0, 3, 4, 5}
    for st in (0, 3, 4, 5):  # undone: the ingress enable bit is off again
        base = regs.port_window(regs.station_base_port(st))
        assert fake.read32(base + regs.REG_PERF_INGRESS) == 0x09000000
    assert chip.perf_info()["running"] is False


def test_the_counters_of_the_second_port_of_a_station_use_its_own_slot(capture):
    coll, fake, chip = build(capture, allow_writes=True, write_switches=("A",))
    coll.collect()
    coll.drain_events()
    # d4 (port 56, station port 8) sent 10 posted 64 B writes, got 4 completions;
    # its neighbour d1 (port 48, station port 0) stays silent
    fake.add_traffic(
        56,
        ingress={"ph": 10, "pdw": 230, "dllp": 7},
        egress={"cplh": 4, "cpldw": 84, "dllp": 3},
    )
    chip.perf_poll()
    got = by_key(coll.collect())
    ent = "port:cc:18.0"
    assert got[ent, "port.rx_bytes"].value == 230 * 4 + 7 * 8
    assert got[ent, "port.tx_bytes"].value == 84 * 4 + 3 * 8
    assert got[ent, "plx.rx_payload_bytes"].value == 640
    assert got[ent, "plx.rx_tlps"].value == 10 and got[ent, "plx.tx_tlps"].value == 4
    assert got["port:cc:10.0", "port.rx_bytes"].value == 0
    assert got["switch:A", "plx.perf_ports"].value == len(PERF_PORTS)
    assert not any(
        s.entity == "port:c8:10.0" and s.metric == "port.rx_bytes" for s in got.values()
    )
    # a second capture gives rates; the capacity is the x8 link's, not x16
    fake.add_traffic(56, ingress={"ph": 1000, "pdw": 22000}, egress={})
    chip.perf_poll()
    second = by_key(coll.collect())
    dt = (second[ent, "port.rx_bytes"].ts_ns - got[ent, "port.rx_bytes"].ts_ns) / 1e9
    assert dt > 0
    delta = 22000 * 4
    assert second[ent, "port.rx_gbs"].value == pytest.approx(delta / dt / 1e9)
    assert second[ent, "plx.rx_util"].value == pytest.approx(
        min(100.0, 100.0 * delta / dt / 1e9 / raw_gbs(16.0, 8))
    )
    assert raw_gbs(16.0, 8) < raw_gbs(16.0, 16)
    chip.close()


# -- perf run over the fake ------------------------------------------------------------


class FakeBox:
    """The chip with persistent state behind a patched ``connect``."""

    def __init__(self, capture):
        self.topo, windows = capture
        self.fakes = {b: plx_fakes.FakeAtlas(w) for b, w in windows.items()}
        self.connects: list[tuple[dict, object]] = []

    def connect(self, chips, *, allow_writes=False, sysfs=None):
        self.connects.append((dict(chips), allow_writes))
        return LocalClient(
            {
                b: helper.ChipAccess(
                    b,
                    port,
                    self.fakes[b],
                    allow_writes=helper.chip_may_write(b, allow_writes),
                    poll_s=0.02,
                )
                for b, port in chips.items()
            }
        )


@pytest.fixture()
def box(capture, monkeypatch):
    fake_box = FakeBox(capture)
    monkeypatch.setattr(cli, "_topology", lambda args: (fake_box.topo, None))
    monkeypatch.setattr(plx_collector, "connect", fake_box.connect)
    monkeypatch.setattr(health, "kernel_aer", lambda *a, **k: {})  # no real /sys
    return fake_box


RUN = ("perf", "run", "--allow-switch-writes", "--json", "--interval", "0.05")


def test_perf_run_refuses_the_chip_until_it_is_approved(box, capsys):
    rc, out, err = run(capsys, *RUN, "--switch", "A", "--duration", "0.2")
    assert rc == 2 and out == ""
    assert "approved for switch none only, not A" in err
    assert box.connects == []  # no helper was even started
    assert box.fakes[CHIP].writes == []


def test_perf_run_stages_the_ports_once_the_chip_is_approved(box, capsys, monkeypatch):
    monkeypatch.setenv(policy.ENV_WRITE_SWITCHES, "A")
    rc, out, _ = run(
        capsys, *RUN, "--switch", "A", "--ports", "48,56", "--duration", "0.3"
    )
    assert rc == 0
    doc = json.loads(out)
    assert doc["ports"] == [48, 56]
    assert doc["anomaly"] is None and doc["undo_mismatch"] == []
    assert doc["health_changes"] == []
    assert {_station(off) for off, _ in box.fakes[CHIP].writes} == {3}  # leaf A only
    assert [c for c in box.connects if c[1]] == [({CHIP: 0}, frozenset({CHIP}))]
    r = doc["registers"]
    assert r["0x768@48"]["after"] == r["0x768@48"]["before"] == 0x09000000
    assert r["0x768@48"]["during"] == 0x29000000
    assert r["0x3e0@48"]["after"] == regs.PERF_CTRL_STOP


def test_perf_run_covers_the_whole_chip_with_the_default_ports(
    box, capsys, monkeypatch
):
    monkeypatch.setenv(policy.ENV_WRITE_SWITCHES, "A")
    rc, out, _ = run(capsys, *RUN, "--switch", "A", "--duration", "0.3")
    assert rc == 0 and json.loads(out)["ports"] == PERF_PORTS
    assert {_station(off) for off, _ in box.fakes[CHIP].writes} == {0, 3, 4, 5}


# -- the idle gate and the leases of the six GPUs ---------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_sysfs(root: Path, topo: Topology, queues=None) -> Sysfs:
    """The amdgpu and KFD files of the six V620 (idle unless told)."""
    for g in topo.gpus.values():
        dev = root / "bus" / "pci" / "devices" / full_bdf(g.bdf)
        _write(dev / "power" / "runtime_status", "active\n")
        _write(dev / "mem_info_vram_used", f"{40 << 20}\n")
        _write(dev / "gpu_busy_percent", "0\n")
    for pid, gpu_ids in (queues or {}).items():
        for i, gid in enumerate(gpu_ids):
            _write(
                root / "class/kfd/kfd/proc" / str(pid) / "queues" / str(i) / "gpuid",
                f"{gid}\n",
            )
    return Sysfs(root)


def lease_dir(tmp_path: Path, **holders: str) -> Path:
    """A lease directory with ``{key: owner}`` holder files (``<owner>.<key>``)."""
    directory = tmp_path / "lease"
    directory.mkdir(parents=True, exist_ok=True)
    for key, owner in holders.items():
        text = f"{owner} 2026-11-10T09:00:00+01:00 test pid {os.getpid()}\n"
        (directory / f"{owner}.{key}").write_text(text)
    return directory


def check(capture, tmp_path, directory, *, queues=None, owner="PlxSwitch"):
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True, exist_ok=True)
    (proc / "4242" / "cmdline").write_bytes(b"python\0-m\0vllm.entrypoints\0")
    topo = capture[0]
    return preflight.check_idle(
        topo,
        "A",
        owner,
        sysfs=make_sysfs(tmp_path / "sys", topo, queues),
        proc_root=proc,
        lease_directory=directory,
        busy_samples=2,
        sleep=lambda _s: None,
    )


def test_the_idle_gate_wants_both_leases_for_the_whole_chip(capture, tmp_path):
    rep = check(capture, tmp_path, lease_dir(tmp_path, a2="PlxSwitch", b4="PlxSwitch"))
    assert rep.ok and rep.problems == ()
    assert [(g.bus, g.hip) for g in rep.gpus] == [
        ("d1", 0),
        ("d4", 1),
        ("d9", 2),
        ("dc", 3),
        ("df", 4),
        ("e2", 5),
    ]
    only_b4 = check(capture, tmp_path, lease_dir(tmp_path / "x", b4="PlxSwitch"))
    assert not only_b4.ok
    assert any("a2" in p and "you need PlxSwitch.a2" in p for p in only_b4.problems)
    assert not any("b4 (buses" in p for p in only_b4.problems)


def test_a_foreign_holder_of_one_leaf_blocks_the_chip(capture, tmp_path):
    directory = lease_dir(tmp_path, a2="PlxSwitch", b4="PlxSwitch")
    assert check(capture, tmp_path, directory).ok
    _write(directory / "HxServer.b4", "HxServer 2026-11-10T09:00:00+01:00 job\n")
    rep = check(capture, tmp_path, directory)
    assert not rep.ok and any("HxServer" in p and "b4" in p for p in rep.problems)


def test_a_process_with_a_queue_on_one_gpu_blocks(capture, tmp_path):
    topo = capture[0]
    directory = lease_dir(tmp_path, a2="PlxSwitch", b4="PlxSwitch")
    gid = topo.gpus["gpu:dc"].kfd_gpu_id
    rep = check(capture, tmp_path, directory, queues={4242: [gid]})
    assert not rep.ok
    (g,) = [g for g in rep.gpus if g.pids]
    assert g.bus == "dc" and g.hip == 3
