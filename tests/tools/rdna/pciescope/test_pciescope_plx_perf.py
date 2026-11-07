# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PerformancePAK: counter layout, arithmetic and the gated write sequence.

The hardware sequence is exercised on ``plx_fakes.FakeAtlas`` (a stateful BAR0
fake); nothing here touches a real switch, so what these tests prove is that the
code issues exactly the documented writes, in order, and unwraps the 32-bit
counters; calibration against real traffic needs the write approval (see the
doc section "Switch registers (PEX88096)").
"""

from __future__ import annotations

import io
import json
import struct
import time

import plx_fakes
import pytest
from pciescope.plx import access, helper, perf, regs
from pciescope.plx.access import AccessError


@pytest.fixture(scope="module")
def windows_b():
    _, chips, _ = plx_fakes.load_capture()
    return chips[plx_fakes.SWITCH_B]


def w(port: int, reg: int) -> int:
    return regs.port_window(port) + reg


# -- layout and arithmetic -----------------------------------------------------------


def test_decode_port_uses_the_documented_layout():
    image = [1000 + i for i in range(regs.PERF_DWORDS_PER_STATION)]
    ing, egr = perf.decode_port(image, 3)
    assert (ing.ph, ing.pdw, ing.nph, ing.npdw, ing.cplh, ing.cpldw) == tuple(
        range(1018, 1024)
    )
    assert ing.dllp == 1000 + 192 + 3
    assert (egr.ph, egr.pdw, egr.nph, egr.npdw, egr.cplh, egr.cpldw) == tuple(
        range(1114, 1120)
    )
    assert egr.dllp == 1000 + 208 + 3
    first, _ = perf.decode_port(image, 0)
    assert first.ph == 1000 and first.dllp == 1192
    _, last = perf.decode_port(image, 15)
    assert last.cpldw == 1000 + 96 + 6 * 15 + 5 and last.dllp == 1000 + 208 + 15


def test_decode_port_rejects_bad_input():
    with pytest.raises(ValueError, match="expected 224"):
        perf.decode_port([0] * 100, 0)
    with pytest.raises(ValueError, match="out of range"):
        perf.decode_port([0] * 224, 16)


def test_bytes_and_payload_model():
    # ten 64-byte posted writes: 4 DW header + 16 DW data + 2 DW overhead each
    c = perf.DirCounters(ph=10, pdw=220, nph=0, npdw=0, cplh=0, cpldw=0, dllp=5)
    assert c.tlps() == 10
    assert c.tlp_bytes() == 880
    assert c.total_bytes() == 880 + 5 * 8
    assert c.payload_bytes() == 640
    # four 64-byte completions: 3 DW header + 16 DW data + 2 DW overhead each
    c = perf.DirCounters(0, 0, 0, 0, cplh=4, cpldw=84, dllp=0)
    assert c.payload_bytes() == 256
    # non-posted requests carry no payload; their DW only count as bytes
    c = perf.DirCounters(0, 0, nph=3, npdw=18, cplh=0, cpldw=0, dllp=0)
    assert (c.payload_bytes(), c.tlp_bytes(), c.tlps()) == (0, 72, 3)


def test_payload_never_goes_negative():
    assert perf.DirCounters(5, 1, 0, 0, 0, 0, 0).payload_bytes() == 0


def test_rates_and_utilisation():
    prev = perf.DirCounters(0, 0, 0, 0, 0, 0, 0)
    cur = perf.DirCounters(
        ph=1_000_000, pdw=22_000_000, nph=0, npdw=0, cplh=0, cpldw=0, dllp=0
    )
    r = perf.rates(prev, cur, 2.0, 26.0)
    assert r.total_gbs == pytest.approx(0.044)
    assert r.payload_gbs == pytest.approx(0.032)
    assert r.util_pct == pytest.approx(100 * 0.044 / 26.0)
    # clamped at 100 like the SDK, and no divide by zero on odd input
    assert perf.rates(prev, cur, 2.0, 0.01).util_pct == 100.0
    assert perf.rates(prev, cur, 2.0, 0.0).util_pct == 0.0
    assert perf.rates(prev, cur, 0.0, 26.0) == perf.DirRates(0.0, 0.0, 0.0)
    assert cur.minus(cur) == prev


# -- START / STOP sequences on the fake ------------------------------------------------


def test_start_issues_the_sdk_writes_in_order(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    info = chip.perf_start([16, 0])
    assert info["running"] and info["owned"] and info["ports"] == [0, 16]
    assert info["stations"] == [0, 1]
    assert fake.writes == [
        # ingress counters on, once per station base port: 0x09000000 | bit 29
        (w(0, 0x768), 0x29000000),
        (w(16, 0x768), 0x29000000),
        # egress counters on, per port: bit 6 and bit 21 (reads 0, must be kept)
        (w(0, 0xF30), 0x00200040),
        (w(16, 0xF30), 0x00200040),
        # monitor: enable | reset | infinite | start
        (w(0, 0x3E0), 0xD8000000),
        (w(16, 0x3E0), 0xD8000000),
        # first capture: reset the RAM read pointer of each station
        (w(0, 0x3F0), 0x25),
        (w(16, 0x3F0), 0x25),
    ]
    assert fake.fifo_reads == 2 * regs.PERF_DWORDS_PER_STATION
    chip.close()


def test_stop_undoes_start_and_keeps_the_credit_bit(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    chip.perf_start([0, 80])
    fake.writes.clear()
    assert chip.perf_stop() == {"running": False, "owned": True}
    assert fake.writes == [
        (w(0, 0x768), 0x09000000),  # bit 29 cleared again
        (w(80, 0x768), 0x09000000),
        (w(0, 0xF30), 0x00200000),  # bit 6 cleared, bit 21 stays set
        (w(80, 0xF30), 0x00200000),
        (w(0, 0x3E0), 0x40000000),  # reset | disable
        (w(80, 0x3E0), 0x40000000),
    ]
    assert chip.perf_info() == {"running": False}


def test_close_stops_the_counters_it_started(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    chip.perf_start([0])
    chip.close()
    assert (w(0, 0x3E0), 0x40000000) in fake.writes
    assert fake.win[0][0x3E0] & regs.PERF_BIT_ENABLE == 0


def test_counters_started_by_someone_else_are_left_alone(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    fake.win[0][0x3E0] = 0xD8000000  # an SDK tool already runs the monitor
    info = chip.perf_start([0, 16])
    assert info["owned"] is False
    assert fake.writes == [(w(0, 0x3F0), 0x25), (w(16, 0x3F0), 0x25)]
    fake.writes.clear()
    assert chip.perf_stop() == {"running": False, "owned": False}
    assert fake.writes == []  # not ours: not switched off


def test_start_twice_and_bad_ports_are_refused(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    with pytest.raises(AccessError, match="no PerformancePAK"):
        chip.perf_start([116])  # station 7: no counter RAM
    with pytest.raises(AccessError, match="neither clock-enabled"):
        chip.perf_start([5])
    with pytest.raises(AccessError, match="no ports"):
        chip.perf_start([])
    assert fake.writes == []  # refused before the first write
    chip.perf_start([0])
    with pytest.raises(AccessError, match="already started"):
        chip.perf_start([16])
    chip.close()


def test_perf_stop_without_start_is_a_noop(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    assert chip.perf_stop() == {"running": False}
    assert fake.writes == []


# -- sampling and unwrapping ----------------------------------------------------


def _totals(chip, station=0):
    return chip.perf_totals()["totals"][str(station)]


def _idx(station_port, name, egress=False):
    base = regs.PERF_EGRESS_BASE if egress else regs.PERF_INGRESS_BASE
    return base + 6 * station_port + regs.PERF_TLP_FIELDS.index(name)


def test_totals_follow_the_traffic_and_decode(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    chip.perf_start([0, 16])
    fake.add_traffic(
        16,
        ingress={"ph": 10, "pdw": 220, "dllp": 7},
        egress={"cplh": 4, "cpldw": 84, "dllp": 3},
    )
    chip.perf_poll()
    ing, egr = perf.decode_port(_totals(chip, 1), 0)  # port 16 = station 1 port 0
    assert (ing.ph, ing.pdw, ing.dllp) == (10, 220, 7)
    assert ing.payload_bytes() == 640
    assert (egr.cplh, egr.cpldw, egr.dllp) == (4, 84, 3)
    quiet, _ = perf.decode_port(_totals(chip, 0), 0)
    assert quiet.total_bytes() == 0
    info = chip.perf_info()
    assert info["polls"] == 2 and info["resets"] == 0 and info["error"] is None
    chip.close()


def test_32_bit_wrap_is_unwrapped_to_64_bit(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    chip.perf_start([0])
    i = _idx(0, "pdw")
    for live in (0x7FFFFFF0, 0xFFFFFFE0, 0x20):  # the last step wraps the counter
        fake.live[0][i] = live
        chip.perf_poll()
    assert _totals(chip)[i] == 0x1_0000_0020
    assert chip.perf_info()["resets"] == 0
    chip.close()


def test_an_external_reset_is_counted_and_does_not_subtract(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    chip.perf_start([0])
    i = _idx(0, "ph")
    fake.live[0][i] = 1000
    chip.perf_poll()
    fake.live[0][i] = 0  # somebody wrote the reset bit
    chip.perf_poll()
    assert _totals(chip)[i] == 1000
    assert chip.perf_info()["resets"] == 1
    fake.live[0][i] = 50
    chip.perf_poll()
    assert _totals(chip)[i] == 1050
    chip.close()


def test_sampler_thread_polls_on_its_own(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True, poll_s=0.01)
    chip.perf_start([0])
    deadline = time.monotonic() + 5
    while chip.perf_info()["polls"] < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert chip.perf_info()["polls"] >= 4
    assert chip.perf_info()["ts_ns"] > 0
    chip.close()
    n = fake.fifo_reads
    time.sleep(0.05)
    assert fake.fifo_reads == n  # no more polling after close


def test_a_failing_read_is_reported_not_raised(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True, poll_s=0.01)
    chip.perf_start([0])

    def boom(off):
        raise AccessError("bus error")

    fake.read32 = boom  # every later read fails
    deadline = time.monotonic() + 5
    while not chip.perf_info()["error"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "bus error" in chip.perf_info()["error"]
    chip._stop.set()


# -- two perf owners on one BAR0 --------------------------------------------------


def test_second_perf_owner_is_refused_by_the_bar0_lock(tmp_path, windows_b):
    dev = tmp_path / plx_fakes.SWITCH_B
    dev.mkdir()
    plx_fakes.write_resource0(dev / "resource0", windows_b)
    kw = {"allow_writes": True, "sysfs": str(tmp_path)}
    a = helper.open_chip(plx_fakes.SWITCH_B, 48, **kw)
    b = helper.open_chip(plx_fakes.SWITCH_B, 48, **kw)
    a.poll_s = b.poll_s = 3600.0  # the sampler thread stays idle
    try:
        a.perf_start([0])
        with pytest.raises(AccessError, match="BAR0 lock"):
            b.perf_start([16])
        a.perf_stop()
    finally:
        a.close()
        b.close()


# -- the real helper process with writes enabled ----------------------------------


def _bar0_dword(path, port, reg):
    with open(path, "rb") as f:
        f.seek(regs.port_window(port) + reg)
        return struct.unpack("<I", f.read(4))[0]


@pytest.fixture()
def fake_bar0(tmp_path, windows_b):
    dev = tmp_path / plx_fakes.SWITCH_B
    dev.mkdir()
    plx_fakes.write_resource0(dev / "resource0", windows_b)
    return tmp_path, dev / "resource0"


def _writing_client(sysfs):
    return access.SubprocessClient(
        {plx_fakes.SWITCH_B: 48},
        allow_writes=True,
        sudo=False,
        sysfs=str(sysfs),
        timeout_s=20,
    )


def _assert_counters_off(res, ports):
    for port in ports:
        assert _bar0_dword(res, port, 0x3E0) == 0x40000000
        assert _bar0_dword(res, port, 0x768) == 0x09000000
        assert _bar0_dword(res, port, 0xF30) == 0x00200000


def test_helper_process_starts_and_stops_the_counters_on_eof(fake_bar0):
    sysfs, res = fake_bar0
    client = _writing_client(sysfs)
    try:
        assert client.hello()["chips"][plx_fakes.SWITCH_B]["writes"] is True
        info = client.perf_start(plx_fakes.SWITCH_B, [16, 0])
        assert info["running"] and info["owned"] and info["ports"] == [0, 16]
        got = client.read(
            plx_fakes.SWITCH_B, [(0, 0x3E0), (16, 0x3E0), (0, 0x768), (16, 0xF30)]
        )
        assert got == [0xD8000000, 0xD8000000, 0x29000000, 0x00200040]
        totals = client.perf_totals(plx_fakes.SWITCH_B)
        assert totals["running"] and sorted(totals["totals"]) == ["0", "1"]
        assert len(totals["totals"]["0"]) == regs.PERF_DWORDS_PER_STATION
    finally:
        client.close()  # EOF on the helper's stdin
    assert client._proc.poll() is not None
    _assert_counters_off(res, [0, 16])
    # ports that were never named were never written
    assert _bar0_dword(res, 32, 0x3E0) == 0
    assert _bar0_dword(res, 32, 0x768) == 0x09000000


def test_helper_process_stops_the_counters_on_sigterm(fake_bar0):
    sysfs, res = fake_bar0
    client = _writing_client(sysfs)
    try:
        client.perf_start(plx_fakes.SWITCH_B, [32])
        assert client.read(plx_fakes.SWITCH_B, [(32, 0x3E0)]) == [0xD8000000]
        client._proc.terminate()
        client._proc.wait(timeout=20)
    finally:
        client.close()
    _assert_counters_off(res, [32])


def test_helper_process_without_the_flag_never_writes(fake_bar0, windows_b):
    sysfs, res = fake_bar0
    before = res.read_bytes()
    client = access.SubprocessClient(
        {plx_fakes.SWITCH_B: 48}, sudo=False, sysfs=str(sysfs), timeout_s=20
    )
    try:
        with pytest.raises(AccessError, match="writes are disabled"):
            client.perf_start(plx_fakes.SWITCH_B, [0])
        client.perf_plan(plx_fakes.SWITCH_B, [0, 16])
        client.dump(plx_fakes.SWITCH_B, [0, 48])
    finally:
        client.close()
    assert res.read_bytes() == before


def test_serve_runs_the_counter_ops_over_json(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=True)
    rfile = io.StringIO(
        "".join(
            json.dumps(r) + "\n"
            for r in (
                {
                    "id": 1,
                    "op": "perf_start",
                    "chip": plx_fakes.SWITCH_B,
                    "ports": [16],
                },
                {"id": 2, "op": "perf_totals", "chip": plx_fakes.SWITCH_B},
                {"id": 3, "op": "perf_info", "chip": plx_fakes.SWITCH_B},
                {"id": 4, "op": "perf_stop", "chip": plx_fakes.SWITCH_B},
                {"id": 5, "op": "perf_info", "chip": plx_fakes.SWITCH_B},
            )
        )
    )
    wfile = io.StringIO()
    helper.serve({plx_fakes.SWITCH_B: chip}, rfile, wfile)
    out = [json.loads(line) for line in wfile.getvalue().splitlines()]
    assert all(r["ok"] for r in out)
    assert out[0]["running"] and out[0]["ports"] == [16]
    assert list(out[1]["totals"]) == ["1"]
    assert out[2]["polls"] == 1
    assert out[3] == {"running": False, "owned": True, "ok": True, "id": 4}
    assert out[4]["running"] is False
    assert fake.win[16][0x3E0] == 0x40000000
    chip.close()
