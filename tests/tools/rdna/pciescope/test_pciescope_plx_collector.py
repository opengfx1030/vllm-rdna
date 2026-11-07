# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``plx`` collector over the capture of par1-cs25 (and a stateful fake).

Read-only mode is compared against ``lspci`` output of the same functions; the
traffic-counter mode (``allow_writes``) runs on ``plx_fakes.FakeAtlas`` and
checks the metrics, the events and that the excluded GPU's port is never written.
"""

from __future__ import annotations

import plx_fakes
import pytest
from pciescope import collectors
from pciescope.collectors import plx as plx_collector
from pciescope.model import Topology, full_bdf, raw_gbs
from pciescope.plx import metrics as plx_metrics
from pciescope.plx import pcie, regs
from pciescope.plx.access import AccessError, LocalClient
from pciescope.plx.ports import build_port_maps

SW_A, SW_B = plx_fakes.SWITCH_A, plx_fakes.SWITCH_B
HIP5 = "gpu:07"


@pytest.fixture(scope="module")
def capture():
    topo, chips, _ = plx_fakes.load_capture()
    return topo, chips


def build(capture, *, allow_writes=False, perf=None, **kw):
    """Collector over fresh fakes of both chips; returns (collector, fakes)."""
    topo, windows = capture
    chips, fakes = {}, {}
    for bdf, win in windows.items():
        chips[bdf], fakes[bdf] = plx_fakes.chip_access(
            win, bdf, allow_writes=allow_writes
        )
    coll = plx_collector.PlxCollector(
        topo,
        client=LocalClient(chips),
        allow_writes=allow_writes,
        perf=perf,
        **kw,
    )
    return coll, fakes, chips


def by_key(samples):
    return {(s.entity, s.metric): s for s in samples}


def poke(fake, port, name, fn):
    """Rewrite register ``name`` (see ``pcie.dynamic_offsets``) of a fake window."""
    win = fake.win[port]
    st = pcie.decode_static(lambda o: win.get(o, 0))
    off = dict(pcie.dynamic_offsets(st))[name]
    win[off] = fn(win.get(off, 0))


# -- catalogue ---------------------------------------------------------------------


def test_metric_catalogue_is_consistent():
    names = [r[0] for r in plx_metrics.ROWS]
    assert len(names) == len(set(names))
    assert set(plx_collector.METRICS) == set(names)
    for name, md in plx_collector.METRICS.items():
        domain, _, rest = name.partition(".")
        assert domain in ("plx", "port") and rest and "." not in rest, name
        assert md.kind in ("gauge", "counter"), name
        assert md.entity in ("port", "switch"), name
        assert md.source == "plx" and md.doc, name
    kinds = {n: m.kind for n, m in plx_collector.METRICS.items()}
    for n in ("port.rx_bytes", "port.tx_bytes", "plx.rx_tlps", "plx.tx_dllps"):
        assert kinds[n] == "counter"
    for n in ("port.rx_gbs", "port.tx_gbs", "plx.rx_util", "plx.link_width"):
        assert kinds[n] == "gauge"


# -- read-only collection ---------------------------------------------------------


def test_available_and_note(capture):
    coll, _, _ = build(capture)
    ok, why = coll.available()
    assert ok and "off (need --allow-switch-writes)" in why
    coll2, _, _ = build(capture, allow_writes=True)
    assert "will start" in coll2.available()[1]


def test_collect_matches_lspci(capture):
    coll, _, _ = build(capture)
    got = by_key(coll.collect())
    expect = plx_fakes.load_lspci()
    physical = ("usp", "dsp")
    seen = 0
    for cm in coll._chips:
        for port, p in cm.ports.items():
            ent = f"port:{p.info.bdf}"
            e = expect[p.info.bdf]
            if p.info.kind == "mgmt":
                assert not [k for k in got if k[0] == ent]
                continue
            seen += 1
            assert got[ent, "plx.dev_status"].value == e["devsta"]
            assert got[ent, "plx.mps_bytes"].value == e["devctl_mps"]
            assert got[ent, "plx.mrrs_bytes"].value == e["devctl_mrrs"]
            assert got[ent, "plx.aspm"].value == e["lnkctl_aspm"]
            if p.info.kind in physical:
                assert got[ent, "plx.link_speed_gts"].value == e["lnksta_speed"]
                assert got[ent, "plx.link_width"].value == e["lnksta_width"]
                assert got[ent, "plx.target_speed_gts"].value == e["lnkctl2_target"]
                assert bool(got[ent, "plx.link_training"].value) == e["lnksta_train"]
            else:  # virtual fan-out ports carry no link metrics
                assert (ent, "plx.link_speed_gts") not in got
    assert seen == 34  # 36 PEX functions minus the two management endpoints


def test_link_metrics_of_the_gpu_ports(capture):
    coll, _, _ = build(capture)
    got = by_key(coll.collect())
    for bdf in ("04:10.0", "0c:00.0", "11:00.0", "11:10.0", "c8:00.0", "d5:10.0"):
        ent = f"port:{bdf}"
        assert got[ent, "plx.link_speed_gts"].value == 16.0
        assert got[ent, "plx.link_width"].value == 16
        assert got[ent, "plx.link_up"].value == 1.0
        assert got[ent, "plx.link_degraded"].value == 0.0
        assert got[ent, "plx.eq_complete"].value == 1.0
        assert got[ent, "plx.acs_ctl"].value == 0.0  # P2P goes straight through
    for bdf in ("19:14.0", "19:15.0", "dd:14.0", "dd:15.0"):  # the empty x1 slots
        ent = f"port:{bdf}"
        assert got[ent, "plx.link_up"].value == 0.0
        assert got[ent, "plx.link_degraded"].value == 0.0
    for letter in "AB":
        sw = f"switch:{letter}"
        assert got[sw, "plx.ports_up"].value == 6
        assert got[sw, "plx.perf_enabled"].value == 0.0
        assert got[sw, "plx.perf_ports"].value == 0.0


def test_every_sample_is_catalogued_and_tagged(capture):
    coll, fakes, _ = build(capture)
    samples = coll.collect()
    assert len(samples) == 382  # the same number the live box produced
    for s in samples:
        md = plx_collector.METRICS[s.metric]
        assert s.unit == md.unit, s.metric
        assert s.entity.startswith(md.entity + ":"), s
        assert s.ts_ns == samples[0].ts_ns
        assert s.tags["switch"] in ("A", "B")
        if md.entity == "port":
            assert {"plx_port", "role"} <= set(s.tags)
    leaf = [s for s in samples if s.entity == "port:04:10.0"]
    assert {s.tags["peer"] for s in leaf} == {"gpu:0a"}
    assert {s.tags["peer"] for s in samples if s.entity == "port:01:00.0"} == {"host"}
    assert all("peer" not in s.tags for s in samples if s.entity == "port:02:00.0")
    # one batched read per chip and no register write at all
    assert all(f.writes == [] for f in fakes.values())
    assert all(f.fifo_reads == 0 for f in fakes.values())


def test_first_collect_reports_sticky_flags_at_start_once(capture):
    coll, _, _ = build(capture)
    coll.collect()
    events = coll.drain_events()
    assert {e.name for e in events} == {"plx_error_flags"}
    assert {e.entity for e in events} == {
        f"port:{b}" for b in ("04:00.0", "04:10.0", "0c:00.0", "11:00.0", "11:10.0")
    }
    for e in events:
        assert e.severity == "info" and "set at start" in e.message
        assert "CorrErr" in e.message and "UnsupReq" in e.message
        assert e.data["switch"] == "B" and e.data["register"] == "dev_status"
    coll.collect()
    assert coll.drain_events() == []


def test_snapshot_reports_chips_and_ports(capture):
    coll, _, _ = build(capture)
    coll.collect()
    snap = coll.snapshot()
    assert [(c["switch"], c["usp_bdf"], c["serial"]) for c in snap["chips"]] == [
        ("A", "c5:00.0", "00-80-5e-10-fa-53-6b-2a"),
        ("B", "01:00.0", "00-80-5e-10-f9-57-20-61"),
    ]
    assert len(snap["ports"]) == 36
    gpu = next(d for d in snap["ports"] if d["bdf"] == "04:10.0")
    assert gpu["peer"] == "gpu:0a" and gpu["state"]["speed_gts"] == 16.0
    assert gpu["perf"] is False


# -- events -----------------------------------------------------------------------


def test_link_downgrade_and_recovery_events(capture):
    coll, fakes, _ = build(capture)
    coll.collect()
    coll.drain_events()
    port16 = fakes[SW_B]
    # 16 GT/s -> 8 GT/s (LnkSta[3:0] = 3)
    poke(port16, 16, "lnk", lambda v: (v & ~(0xF << 16)) | 3 << 16)
    got = by_key(coll.collect())
    assert got["port:04:10.0", "plx.link_speed_gts"].value == 8.0
    assert got["port:04:10.0", "plx.link_degraded"].value == 1.0
    ev = {e.name: e for e in coll.drain_events()}
    assert set(ev) == {"link_change", "link_degraded"}
    assert ev["link_change"].severity == "warn"
    assert ev["link_change"].data["old"] == [16.0, 16, True]
    assert ev["link_change"].data["new"] == [8.0, 16, True]
    assert ev["link_change"].entity == "port:04:10.0"
    assert "16 GT/s x16 -> 8 GT/s x16" in ev["link_change"].message
    # back to 16 GT/s
    poke(port16, 16, "lnk", lambda v: (v & ~(0xF << 16)) | 4 << 16)
    coll.collect()
    ev = {e.name: e for e in coll.drain_events()}
    assert set(ev) == {"link_change", "link_restored"}
    assert ev["link_change"].severity == "info"


def test_link_down_event_and_width_change(capture):
    coll, fakes, _ = build(capture)
    coll.collect()
    coll.drain_events()
    # port 32 loses the link: DL_Active clear and width 0
    poke(
        fakes[SW_A],
        32,
        "lnk",
        lambda v: v & ~((1 << 29) | (0x3F << 20)),
    )
    got = by_key(coll.collect())
    assert got["port:d0:00.0", "plx.link_up"].value == 0.0
    assert got["switch:A", "plx.ports_up"].value == 5
    (ev,) = coll.drain_events()
    assert ev.name == "link_change" and ev.severity == "warn"
    assert ev.data["new"][2] is False


def test_dpc_and_new_error_bits_raise_events_once(capture):
    coll, fakes, _ = build(capture)
    coll.collect()
    coll.drain_events()
    fb = fakes[SW_B]
    poke(fb, 64, "dpc_sta", lambda v: (v & ~0xFFFF) | 0b0_10_1)  # ERR_FATAL trigger
    poke(fb, 32, "dev", lambda v: v | 1 << 18)  # DevSta fatal error bit
    poke(fb, 80, "uncorr", lambda v: v | 1 << 12)  # a new uncorrectable status bit
    got = by_key(coll.collect())
    assert got["port:11:00.0", "plx.dpc_status"].value == 5
    ev = {(e.name, e.entity): e for e in coll.drain_events()}
    dpc = ev["dpc_triggered", "port:11:00.0"]
    assert dpc.severity == "error" and "ERR_FATAL" in dpc.message
    fatal = ev["plx_error_flags", "port:0c:00.0"]
    assert fatal.severity == "error" and "FatalErr" in fatal.message
    uncorr = ev["plx_error_flags", "port:11:10.0"]
    assert uncorr.severity == "warn"
    assert uncorr.data["register"] == "uncorr_status"
    assert uncorr.data["new"] == 1 << 12 and "TLP" in uncorr.message
    assert len(ev) == 3
    coll.collect()
    assert coll.drain_events() == []  # sticky: reported when they appear, once


# -- availability -----------------------------------------------------------------


def test_unavailable_without_a_switch_in_the_topology():
    coll = plx_collector.PlxCollector(Topology(host="x", captured_ns=0))
    ok, why = coll.available()
    assert not ok and "no PEX88096" in why
    assert coll.collect() == [] and coll.drain_events() == []


def test_unavailable_when_the_helper_cannot_start(capture):
    topo, _ = capture

    def refuse(chips, **kw):
        raise AccessError("sudo: a password is required")

    coll = plx_collector.PlxCollector(topo, connector=refuse)
    ok, why = coll.available()
    assert not ok and "password is required" in why
    assert coll.collect() == []


def test_unavailable_when_the_windows_disagree_with_the_topology(capture):
    topo, windows = capture
    bad = {b: {p: dict(w) for p, w in ws.items()} for b, ws in windows.items()}
    bad[SW_B][16][0x18] ^= 0x0001_0000  # wrong subordinate bus at port 16
    chips = {b: plx_fakes.chip_access(w, b)[0] for b, w in bad.items()}
    coll = plx_collector.PlxCollector(topo, client=LocalClient(chips))
    ok, why = coll.available()
    assert not ok and "disagree" in why and "port 16" in why


def test_reconnects_after_the_helper_dies(capture):
    topo, windows = capture
    made = []

    def connector(chips, **kw):
        c = {b: plx_fakes.chip_access(windows[b], b)[0] for b in chips}
        client = LocalClient(c)
        made.append(client)
        return client

    coll = plx_collector.PlxCollector(topo, connector=connector)
    assert coll.available()[0] and len(coll.collect()) == 382
    client = made[0]
    real_read = client.read

    def dead(bdf, items):
        client.read = real_read
        raise AccessError("helper exited")

    client.read = dead
    with pytest.raises(AccessError, match="helper exited"):
        coll.collect()
    # the next collect builds a fresh client instead of reusing the dead one
    assert len(coll.collect()) == 382
    assert len(made) == 2
    coll.close()


def test_registry_builds_the_collector_with_an_injected_client(capture):
    topo, windows = capture
    chips = {b: plx_fakes.chip_access(w, b)[0] for b, w in windows.items()}
    res = collectors.build(
        topo, ["plx"], options={"plx": {"client": LocalClient(chips)}}
    )
    assert res.skipped == []
    (coll,) = res.collectors
    assert coll.name == "plx" and isinstance(coll, plx_collector.PlxCollector)
    assert len(coll.collect()) == 382


def test_registry_skips_with_the_reason_when_not_possible(capture):
    topo, _ = capture

    def refuse(chips, **kw):
        raise AccessError("no root")

    res = collectors.build(topo, ["plx"], options={"plx": {"connector": refuse}})
    assert res.collectors == []
    assert res.skipped == [("plx", "AccessError: no root")]


def test_collector_works_on_the_core_fixture_topology(fx_topo, capture):
    """The topology Core builds from the sysfs capture gives the same port maps."""
    _, windows = capture
    chips = {b: plx_fakes.chip_access(w, b)[0] for b, w in windows.items()}
    coll = plx_collector.PlxCollector(fx_topo, client=LocalClient(chips))
    assert coll.available()[0]
    assert len(coll.collect()) == 382


# -- traffic counters (writes gated) ----------------------------------------------


def test_no_counters_without_the_write_flag(capture):
    coll, fakes, _ = build(capture, allow_writes=False, perf=True)
    assert coll.want_perf is False  # perf needs allow_writes
    coll.collect()
    coll.collect()
    assert all(f.writes == [] for f in fakes.values())
    assert not [e for e in coll.drain_events() if e.name.startswith("plx_perf")]


def test_counters_skip_the_excluded_gpu_and_never_write_its_port(capture):
    coll, fakes, _ = build(capture, allow_writes=True)
    coll.collect()
    events = [e for e in coll.drain_events() if e.name == "plx_perf_start"]
    assert {e.entity: e.data["ports"] for e in events} == {
        "switch:A": [0, 16, 32, 48, 64, 80],
        "switch:B": [16, 32, 48, 64, 80],  # port 0 hosts gpu:07 (HIP 5)
    }
    assert all(e.data["owned"] for e in events)
    lo, hi = regs.port_window(0), regs.port_window(1)
    assert [w for w in fakes[SW_B].writes if lo <= w[0] < hi] == []
    # switch A's port 0 is a different GPU: its counters do start
    assert any(lo <= w[0] < hi for w in fakes[SW_A].writes)
    assert coll.snapshot()["chips"][1]["perf_ports"] == [16, 32, 48, 64, 80]


def test_perf_exclusion_is_configurable(capture):
    coll, fakes, _ = build(
        capture, allow_writes=True, perf_exclude=("gpu:07", "gpu:0a")
    )
    coll.collect()
    ports = {
        e.entity: e.data["ports"]
        for e in coll.drain_events()
        if e.name == "plx_perf_start"
    }
    assert ports["switch:B"] == [32, 48, 64, 80]
    none, nf, _ = build(capture, allow_writes=True, perf=False)
    none.collect()
    assert all(f.writes == [] for f in nf.values())


def test_counter_metrics_and_rates(capture):
    coll, fakes, chips = build(capture, allow_writes=True)
    coll.collect()  # starts the counters
    coll.drain_events()
    fb = fakes[SW_B]
    # GPU 0a (port 16): it sent 10 posted 64 B writes and 7 DLLPs, got 4 completions
    fb.add_traffic(
        16,
        ingress={"ph": 10, "pdw": 220, "dllp": 7},
        egress={"cplh": 4, "cpldw": 84, "dllp": 3},
    )
    chips[SW_B].perf_poll()
    first = coll.collect()
    got = by_key(first)
    ent = "port:04:10.0"
    assert got[ent, "port.rx_bytes"].value == 220 * 4 + 7 * 8
    assert got[ent, "port.tx_bytes"].value == 84 * 4 + 3 * 8
    assert got[ent, "plx.rx_payload_bytes"].value == 640
    assert got[ent, "plx.tx_payload_bytes"].value == 256
    assert got[ent, "plx.rx_tlps"].value == 10 and got[ent, "plx.tx_tlps"].value == 4
    assert got[ent, "plx.rx_dllps"].value == 7 and got[ent, "plx.tx_dllps"].value == 3
    assert got[ent, "port.rx_bytes"].unit == "B"
    assert got["switch:B", "plx.perf_enabled"].value == 1.0
    assert got["switch:B", "plx.perf_ports"].value == 5
    assert (ent, "port.rx_gbs") not in got  # needs two captures
    # the helper's capture time is the sample time, not the collect time
    assert got[ent, "port.rx_bytes"].ts_ns == chips[SW_B].perf_info()["ts_ns"]
    assert "port:04:00.0" not in {
        s.entity for s in first if s.metric == "port.rx_bytes"
    }
    # a second capture 1.5 MB later gives the rates over the capture interval
    fb.add_traffic(16, ingress={"ph": 1000, "pdw": 22000}, egress={})
    chips[SW_B].perf_poll()
    second = by_key(coll.collect())
    dt = (second[ent, "port.rx_bytes"].ts_ns - got[ent, "port.rx_bytes"].ts_ns) / 1e9
    assert dt > 0
    delta = 22000 * 4
    assert second[ent, "port.rx_gbs"].value == pytest.approx(delta / dt / 1e9)
    assert second[ent, "plx.rx_payload_gbs"].value == pytest.approx(
        (22000 * 4 - 1000 * 24) / dt / 1e9
    )
    cap = raw_gbs(16.0, 16)
    assert second[ent, "plx.rx_util"].value == pytest.approx(
        min(100.0, 100.0 * delta / dt / 1e9 / cap)
    )
    assert second[ent, "port.tx_gbs"].value == 0.0
    coll.close()


def test_idle_collect_without_a_new_capture_emits_no_counters(capture):
    coll, _, chips = build(capture, allow_writes=True)
    coll.collect()
    chips[SW_A].perf_poll()
    chips[SW_B].perf_poll()
    n = len([s for s in coll.collect() if s.metric == "port.rx_bytes"])
    assert n == 11  # 6 ports of switch A + 5 of switch B
    again = [s for s in coll.collect() if s.metric == "port.rx_bytes"]
    assert again == []  # the sampler has not captured since


def test_a_dead_sampler_is_reported_as_an_event(capture):
    coll, fakes, chips = build(capture, allow_writes=True)
    coll.collect()
    coll.drain_events()
    chips[SW_B]._perf.error = "AccessError: bus error"
    coll.collect()
    ev = [e for e in coll.drain_events() if e.name == "plx_perf_error"]
    assert len(ev) == 1 and ev[0].entity == "switch:B" and "bus error" in ev[0].message


def test_start_failure_is_an_event_not_an_exception(capture):
    coll, fakes, chips = build(capture, allow_writes=True)

    def refuse(ports):
        raise AccessError("another process holds the BAR0 lock (perf owner)")

    chips[SW_A].perf_start = refuse
    coll.collect()
    ev = [e for e in coll.drain_events() if e.name == "plx_perf_error"]
    assert len(ev) == 1 and ev[0].entity == "switch:A" and "BAR0 lock" in ev[0].message
    assert ev[0].severity == "error"
    # the other switch still runs, and the failed one is not retried every second
    assert coll.snapshot()["chips"][1]["perf_running"] is True
    coll.collect()
    assert [e for e in coll.drain_events() if e.name == "plx_perf_error"] == []


def test_close_stops_counters_started_through_the_connector(capture):
    topo, windows = capture
    fakes = {}

    def connector(chips, *, allow_writes):
        c = {}
        for b in chips:
            c[b], fakes[b] = plx_fakes.chip_access(windows[b], b, allow_writes=True)
        return LocalClient(c)

    coll = plx_collector.PlxCollector(topo, allow_writes=True, connector=connector)
    coll.collect()
    assert fakes[SW_B].win[16][0x3E0] == 0xD8000000
    coll.close()
    assert fakes[SW_B].win[16][0x3E0] == 0x40000000  # monitor_stop
    assert fakes[SW_A].win[0][0x3E0] == 0x40000000
    coll.close()  # idempotent


def test_chip_keys_match_the_capture(capture):
    topo, windows = capture
    assert {full_bdf(m.usp_bdf) for m in build_port_maps(topo)} == set(windows)
