# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BAR0 helper policy: what may be read, what is refused, and that nothing writes.

``ChipAccess`` runs over recorded windows (``DictMmio``/``FakeAtlas``), a real
``mmap`` of a file that mimics ``resource0``, and the real ``python -m
pciescope.plx.helper`` child process (``sudo`` off, a temp dir as sysfs).
"""

from __future__ import annotations

import hashlib
import io
import json

import plx_fakes
import pytest
from pciescope.plx import access, helper, regs
from pciescope.plx.access import AccessError
from pciescope.plx.ports import build_port_maps


@pytest.fixture(scope="module")
def windows_b():
    _, chips, _ = plx_fakes.load_capture()
    return chips[plx_fakes.SWITCH_B]


@pytest.fixture(scope="module")
def windows_a():
    _, chips, _ = plx_fakes.load_capture()
    return chips[plx_fakes.SWITCH_A]


# -- construction and read policy ----------------------------------------------


def test_wrong_window_id_is_rejected(windows_b):
    bad = {p: dict(w) for p, w in windows_b.items()}
    bad[48][0] = 0x1234_1000
    with pytest.raises(AccessError, match="reads id 0x12341000"):
        plx_fakes.chip_access(bad)


def test_usp_must_be_a_station_base_port(windows_b):
    with pytest.raises(AccessError, match="station base"):
        helper.ChipAccess("x", 49, plx_fakes.FakeAtlas(windows_b))


def test_enabled_ports_come_from_the_clock_masks(windows_b):
    chip, _ = plx_fakes.chip_access(windows_b)
    assert sorted(chip.enabled) == [0, 16, 32, 48, 64, 80, 116, 117]
    assert chip.masks == [0x00010001, 0x00010001, 0x00010001, 0x3]


def test_read_returns_dwords_in_request_order(windows_b):
    chip, _ = plx_fakes.chip_access(windows_b)
    got = chip.read([(48, 0x00), (0, 0x00), (48, 0x30C), (224, 0x00)])
    assert got == [regs.ID_DWORD, regs.ID_DWORD, 0x00010001, regs.ID_DWORD]


@pytest.mark.parametrize("port", [1, 5, 15, 96, 118, 128, 248, 251, 253])
def test_ports_that_are_not_enabled_are_refused(windows_b, port):
    chip, fake = plx_fakes.chip_access(windows_b)
    with pytest.raises(AccessError, match="neither clock-enabled"):
        chip.read([(port, 0)])
    with pytest.raises(AccessError, match="neither clock-enabled"):
        chip.dump([port])


@pytest.mark.parametrize("port", [224, 227, 232, 235, 247, 255])
def test_virtual_ports_are_allowed(windows_b, port):
    chip, _ = plx_fakes.chip_access(windows_b)
    assert chip.read([(port, 0)]) == [regs.ID_DWORD]


@pytest.mark.parametrize("off", sorted(regs.READ_SENSITIVE))
def test_read_sensitive_registers_are_never_read(windows_b, off):
    chip, fake = plx_fakes.chip_access(windows_b)
    with pytest.raises(AccessError, match="side effects"):
        chip.read([(48, off)])
    assert fake.fifo_reads == 0


@pytest.mark.parametrize("off", [1, 2, 3, 0x1000, 0x2000, -4])
def test_bad_offsets_are_refused(windows_b, off):
    chip, _ = plx_fakes.chip_access(windows_b)
    with pytest.raises(AccessError, match="bad register offset"):
        chip.read([(48, off)])


def test_dump_skips_the_sensitive_registers(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b)
    # even a window that holds values at those offsets must not be read there
    for off in regs.READ_SENSITIVE:
        fake.win[0][off] = 0xDEAD_BEEF
    dump = chip.dump([0])
    assert fake.fifo_reads == 0
    assert all(off not in dump[0] for off in regs.READ_SENSITIVE)
    assert dump[0][0] == regs.ID_DWORD


def test_dump_equals_the_recorded_window(windows_b):
    chip, _ = plx_fakes.chip_access(windows_b)
    for port in (0, 48, 116, 224, 247):
        assert chip.dump([port])[port] == windows_b[port]


# -- nothing writes without the opt-in --------------------------------------------


def test_perf_needs_writes_enabled(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, allow_writes=False)
    with pytest.raises(AccessError, match="writes are disabled"):
        chip.perf_start([0, 16])
    with pytest.raises(AccessError, match="writes are disabled"):
        chip.perf_stop()
    chip.close()
    assert fake.writes == []


def test_dict_mmio_cannot_be_written(windows_b):
    mm = access.DictMmio(windows_b)
    with pytest.raises(AccessError, match="read-only"):
        mm.write32(regs.port_window(48) + 0x768, 1)


def test_perf_plan_is_read_only_and_lists_the_documented_writes(windows_a):
    chip, fake = plx_fakes.chip_access(windows_a, plx_fakes.SWITCH_A)
    ports = [0, 16, 32, 48, 64, 80]
    plan = chip.perf_plan(ports)
    assert fake.writes == []  # a plan reads only
    assert plan["monitor_already_running"] is False
    steps = [
        (s["step"], s["port"], s["reg"], s["old"], s["new"]) for s in plan["steps"]
    ]
    ingress = [("ingress_enable", p, 0x768, 0x09000000, 0x29000000) for p in ports]
    egress = [("egress_enable", p, 0xF30, 0, 0x00200040) for p in ports]
    start = [("monitor_start", p, 0x3E0, 0, 0xD8000000) for p in ports]
    # probe mode (0x350 bit 8) is already clear on the box: no write is planned
    assert steps == ingress + egress + start
    assert [(s["port"], s["reg"], s["new"]) for s in plan["per_poll"]] == [
        (p, 0x3F0, 0x25) for p in ports
    ]
    stop = chip.perf_plan(ports, stop=True)
    assert [(s["step"], s["new"]) for s in stop["steps"]][:1] == [
        ("ingress_disable", 0x09000000)
    ]
    assert [s["new"] for s in stop["steps"] if s["step"] == "egress_disable"] == [
        0x00200000
    ] * 6  # bit 21 stays set
    assert fake.writes == []


def test_plan_rejects_ports_without_counters(windows_a):
    chip, _ = plx_fakes.chip_access(windows_a, plx_fakes.SWITCH_A)
    with pytest.raises(AccessError, match="no PerformancePAK"):
        chip.perf_plan([116])
    with pytest.raises(AccessError, match="no ports"):
        chip.perf_plan([])
    with pytest.raises(AccessError, match="neither clock-enabled"):
        chip.perf_plan([7])


# -- JSON-lines protocol ----------------------------------------------------------


def _serve(chip, *requests):
    rfile = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    wfile = io.StringIO()
    helper.serve({plx_fakes.SWITCH_B: chip}, rfile, wfile)
    return [json.loads(line) for line in wfile.getvalue().splitlines()]


def test_serve_answers_read_dump_and_errors(windows_b):
    chip, _ = plx_fakes.chip_access(windows_b)
    out = _serve(
        chip,
        {"id": 1, "op": "hello"},
        {"id": 2, "op": "read", "chip": plx_fakes.SWITCH_B, "regs": [[48, 0]]},
        {"id": 3, "op": "read", "chip": plx_fakes.SWITCH_B, "regs": [[48, 0x3E4]]},
        {"id": 4, "op": "dump", "chip": plx_fakes.SWITCH_B, "ports": [48]},
        {"id": 5, "op": "perf_start", "chip": plx_fakes.SWITCH_B, "ports": [0]},
        {"id": 6, "op": "nope", "chip": plx_fakes.SWITCH_B},
        {"id": 7, "op": "read", "chip": "0000:ff:00.0", "regs": []},
        {"id": 8, "op": "quit"},
        {"id": 9, "op": "hello"},  # never answered: the loop ended at quit
    )
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5, 6, 7]
    assert out[0]["ok"] and out[0]["chips"][plx_fakes.SWITCH_B]["writes"] is False
    assert out[1] == {"vals": [regs.ID_DWORD], "us": out[1]["us"], "ok": True, "id": 2}
    assert not out[2]["ok"] and "side effects" in out[2]["error"]
    assert out[3]["ok"] and out[3]["windows"]["48"]["0"] == regs.ID_DWORD
    assert not out[4]["ok"] and "writes are disabled" in out[4]["error"]
    assert not out[5]["ok"] and "unknown op" in out[5]["error"]
    assert not out[6]["ok"] and "unknown chip" in out[6]["error"]


def test_serve_survives_garbage_lines(windows_b):
    chip, _ = plx_fakes.chip_access(windows_b)
    rfile = io.StringIO('not json\n\n{"id": 1, "op": "hello"}\n')
    wfile = io.StringIO()
    helper.serve({plx_fakes.SWITCH_B: chip}, rfile, wfile)
    lines = [json.loads(x) for x in wfile.getvalue().splitlines()]
    assert lines[0]["ok"] is False and lines[0]["id"] is None
    assert lines[1]["ok"] is True and lines[1]["id"] == 1


# -- the real mmap path ------------------------------------------------------------


@pytest.fixture()
def fake_sysfs(tmp_path, windows_b):
    dev = tmp_path / plx_fakes.SWITCH_B
    dev.mkdir()
    plx_fakes.write_resource0(dev / "resource0", windows_b)
    return tmp_path


def test_bar0_read_only_mapping_cannot_write(fake_sysfs):
    bar = helper.Bar0(str(fake_sysfs / plx_fakes.SWITCH_B / "resource0"))
    try:
        assert bar.read32(regs.port_window(48)) == regs.ID_DWORD
        with pytest.raises(AccessError, match="read-only"):
            bar.write32(regs.port_window(48) + 0x768, 0)
        with pytest.raises(TypeError):  # the memoryview itself is read-only too
            bar._u32[0] = 0
    finally:
        bar.close()


def test_bar0_writable_mapping_round_trips_and_locks(fake_sysfs):
    path = str(fake_sysfs / plx_fakes.SWITCH_B / "resource0")
    a = helper.Bar0(path, writable=True)
    b = helper.Bar0(path, writable=True)
    try:
        off = regs.port_window(48) + 0xF30
        a.write32(off, 0x0020_0040)
        assert b.read32(off) == 0x0020_0040
        assert a.lock() is True
        assert b.lock() is False  # one perf owner at a time
    finally:
        a.close()
        b.close()


def test_open_chip_validates_the_mapping(fake_sysfs):
    chip = helper.open_chip(plx_fakes.SWITCH_B, 48, sysfs=str(fake_sysfs))
    try:
        assert chip.read([(0, 0x68 + 0x10)])[0] >> 16 & 0xF == 4  # LnkSta speed
        assert not chip.allow_writes
    finally:
        chip.close()
    # port 96 is a station base with no window behind it: reads as zero
    with pytest.raises(AccessError, match="reads id 0x00000000"):
        helper.open_chip(plx_fakes.SWITCH_B, 96, sysfs=str(fake_sysfs))


def test_main_reports_a_missing_resource_as_json(tmp_path, capsys):
    rc = helper.main(["--chip", "0000:99:00.0:48", "--sysfs", str(tmp_path)])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "FileNotFoundError" in out["error"]


# -- the real child process ---------------------------------------------------------


def test_subprocess_client_end_to_end(fake_sysfs, windows_b):
    client = access.SubprocessClient(
        {plx_fakes.SWITCH_B: 48}, sudo=False, sysfs=str(fake_sysfs), timeout_s=20
    )
    try:
        info = client.hello()["chips"][plx_fakes.SWITCH_B]
        assert info["writes"] is False and info["usp_port"] == 48
        assert info["enabled"] == [0, 16, 32, 48, 64, 80, 116, 117]
        assert client.read(plx_fakes.SWITCH_B, [(0, 0), (48, 0x30C)]) == [
            regs.ID_DWORD,
            0x00010001,
        ]
        assert client.dump(plx_fakes.SWITCH_B, [16, 247]) == {
            16: windows_b[16],
            247: windows_b[247],
        }
        with pytest.raises(AccessError, match="side effects"):
            client.read(plx_fakes.SWITCH_B, [(0, 0x3E4)])
        with pytest.raises(AccessError, match="writes are disabled"):
            client.perf_start(plx_fakes.SWITCH_B, [0])
        plan = client.perf_plan(plx_fakes.SWITCH_B, [16, 32])
        assert [s["step"] for s in plan["steps"]].count("monitor_start") == 2
    finally:
        client.close()
    assert client._proc.poll() is not None  # EOF on stdin ended the helper


def test_subprocess_client_reports_a_helper_that_cannot_start(tmp_path):
    with pytest.raises(AccessError, match="FileNotFoundError|exited"):
        access.SubprocessClient(
            {"0000:99:00.0": 48}, sudo=False, sysfs=str(tmp_path), timeout_s=20
        )


# -- one chip read-write, the other read-only ------------------------------------

SW_A, SW_B = plx_fakes.SWITCH_A, plx_fakes.SWITCH_B


def test_chip_may_write():
    assert helper.chip_may_write("x", True) is True
    assert helper.chip_may_write("x", False) is False
    assert helper.chip_may_write("x", ["x", "y"]) is True
    assert helper.chip_may_write("z", frozenset({"x"})) is False
    assert helper.chip_may_write("x", frozenset()) is False


@pytest.fixture()
def two_chip_sysfs(tmp_path):
    _, chips, _ = plx_fakes.load_capture()
    for bdf, win in chips.items():
        (tmp_path / bdf).mkdir()
        plx_fakes.write_resource0(tmp_path / bdf / "resource0", win)
    return tmp_path


def _run_main(monkeypatch, capsys, requests, argv):
    monkeypatch.setattr(helper.signal, "signal", lambda *_: None)
    lines = "".join(json.dumps(r) + "\n" for r in requests)
    monkeypatch.setattr(helper.sys, "stdin", io.StringIO(lines))
    rc = helper.main(argv)
    return rc, [json.loads(x) for x in capsys.readouterr().out.splitlines()]


def test_write_chip_maps_only_the_named_chip_read_write(
    two_chip_sysfs, monkeypatch, capsys
):
    rc, (hello, refused) = _run_main(
        monkeypatch,
        capsys,
        [
            {"id": 1, "op": "hello"},
            {"id": 2, "op": "perf_start", "chip": SW_A, "ports": [16]},
        ],
        [
            "--chip",
            f"{SW_A}:48",
            "--chip",
            f"{SW_B}:48",
            "--write-chip",
            SW_B,
            "--sysfs",
            str(two_chip_sysfs),
        ],
    )
    assert rc == 0
    assert {b: c["writes"] for b, c in hello["chips"].items()} == {
        SW_A: False,
        SW_B: True,
    }
    assert refused["ok"] is False and "writes are disabled" in refused["error"]


def test_allow_writes_flag_still_maps_every_chip_read_write(
    two_chip_sysfs, monkeypatch, capsys
):
    rc, (hello,) = _run_main(
        monkeypatch,
        capsys,
        [{"id": 1, "op": "hello"}],
        [
            "--chip",
            f"{SW_A}:48",
            "--chip",
            f"{SW_B}:48",
            "--allow-writes",
            "--sysfs",
            str(two_chip_sysfs),
        ],
    )
    assert rc == 0
    assert all(c["writes"] for c in hello["chips"].values())


def test_write_chip_must_name_a_chip_of_the_run(two_chip_sysfs, monkeypatch, capsys):
    rc, (err,) = _run_main(
        monkeypatch,
        capsys,
        [],
        [
            "--chip",
            f"{SW_B}:48",
            "--write-chip",
            SW_A,
            "--sysfs",
            str(two_chip_sysfs),
        ],
    )
    assert rc == 2
    assert err["ok"] is False and "--write-chip" in err["error"]


def test_subprocess_client_writes_only_the_approved_chip(two_chip_sysfs):
    def digest(bdf):
        return hashlib.sha256((two_chip_sysfs / bdf / "resource0").read_bytes())

    a_before = digest(SW_A).hexdigest()
    b_before = digest(SW_B).hexdigest()
    client = access.SubprocessClient(
        {SW_A: 48, SW_B: 48},
        allow_writes=frozenset({SW_B}),
        sudo=False,
        sysfs=str(two_chip_sysfs),
        timeout_s=20,
    )
    try:
        chips = client.hello()["chips"]
        assert (chips[SW_A]["writes"], chips[SW_B]["writes"]) == (False, True)
        with pytest.raises(AccessError, match="writes are disabled"):
            client.perf_start(SW_A, [16])
        info = client.perf_start(SW_B, [16])
        assert info["owned"] is True
        client.perf_stop(SW_B)
    finally:
        client.close()
    assert digest(SW_A).hexdigest() == a_before  # not one byte of A changed
    assert digest(SW_B).hexdigest() != b_before  # the 0xF30/0x3E0/0x768 undo values


# -- the HIP 5 port is refused by the root helper itself --------------------------
# The W7800 of HIP 5 (gpu:07) hangs on port 0 of switch B.  The collector and the
# CLI leave it out by default; these tests ask the helper directly, without any
# of that, and it must still refuse before one store reaches the chip.


@pytest.mark.parametrize(
    "ports", [[0], [0, 16], [16, 0, 32, 48, 64, 80], [0, 0], [5], [15]]
)
def test_helper_refuses_every_start_touching_the_hip5_station(windows_b, ports):
    chip, fake = plx_fakes.chip_access(windows_b, SW_B, allow_writes=True)
    with pytest.raises(AccessError, match="protected station"):
        chip.perf_start(ports)
    assert fake.writes == []  # refused before the first store
    assert chip.perf_info() == {"running": False}
    chip.close()
    assert fake.writes == []


def test_helper_refuses_plans_and_ops_of_the_hip5_station(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, SW_B, allow_writes=True)
    for call in (chip.perf_plan, chip.start_ops, chip.stop_ops):
        with pytest.raises(AccessError, match="protected station"):
            call([0, 16])
    with pytest.raises(AccessError, match="protected station"):
        chip.perf_plan([0], stop=True)
    assert fake.writes == []


def test_the_single_write_path_refuses_the_hip5_windows(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, SW_B, allow_writes=True)
    for port in (0, 1, 15):  # station 0: the base port's registers are station-wide
        for reg in (regs.REG_PERF_RAM_CTRL, regs.REG_PERF_CTRL, regs.REG_PERF_INGRESS):
            with pytest.raises(AccessError, match="refusing a write to port"):
                chip._write(regs.port_window(port) + reg, 1)
    with pytest.raises(AccessError, match="outside the port windows"):
        chip._write(0x100, 0)
    with pytest.raises(AccessError, match="unaligned write"):
        chip._write(regs.port_window(16) + 0x3F2, 0)
    assert fake.writes == []
    chip._write(regs.port_window(16) + regs.REG_PERF_RAM_CTRL, 0x25)  # a leaf port
    assert fake.writes == [(regs.port_window(16) + regs.REG_PERF_RAM_CTRL, 0x25)]


def test_serve_refuses_a_direct_start_on_the_hip5_port(windows_b):
    chip, fake = plx_fakes.chip_access(windows_b, SW_B, allow_writes=True)
    out = _serve(
        chip,
        {"id": 1, "op": "perf_start", "chip": SW_B, "ports": [0, 16]},
        {"id": 2, "op": "perf_start", "chip": SW_B, "ports": [0]},
        {"id": 3, "op": "perf_plan", "chip": SW_B, "ports": [0]},
        {"id": 4, "op": "perf_info", "chip": SW_B},
    )
    assert [o["ok"] for o in out] == [False, False, False, True]
    assert all("protected station" in o["error"] for o in out[:3])
    assert out[3]["running"] is False
    assert fake.writes == []


def test_the_other_stations_and_the_other_switch_are_not_affected(windows_a, windows_b):
    a, fa = plx_fakes.chip_access(windows_a, SW_A, allow_writes=True)
    info = a.perf_start([0, 16])  # switch A's port 0 is another GPU (HIP 0)
    assert info["running"] is True and fa.writes
    a.close()
    b, fb = plx_fakes.chip_access(windows_b, SW_B, allow_writes=True)
    assert b.perf_start([16, 32, 48, 64, 80])["running"] is True
    lo, hi = regs.port_window(0), regs.port_window(16)
    assert [w for w in fb.writes if lo <= w[0] < hi] == []
    b.close()
    assert [w for w in fb.writes if lo <= w[0] < hi] == []  # nor on the undo


def test_hello_reports_what_the_helper_protects(windows_a, windows_b):
    a, _ = plx_fakes.chip_access(windows_a, SW_A)
    b, _ = plx_fakes.chip_access(windows_b, SW_B)
    assert (a.hello()["switch"], a.hello()["protected_ports"]) == ("A", [])
    assert (b.hello()["switch"], b.hello()["protected_ports"]) == ("B", list(range(16)))


def test_an_unknown_chip_is_never_mapped_writable(windows_b):
    mmio = plx_fakes.FakeAtlas(windows_b)
    with pytest.raises(AccessError, match="not a known PEX88096"):
        helper.ChipAccess("0000:99:00.0", 48, mmio, allow_writes=True)
    chip = helper.ChipAccess("0000:99:00.0", 48, mmio)  # reading is fine
    assert chip.hello()["switch"] is None and chip.protected_ports == frozenset()
    assert mmio.writes == []


def test_the_protected_ports_are_the_ones_the_topology_puts_gpu07_on():
    topo, windows, _ = plx_fakes.load_capture()
    hip5 = {
        m.switch: [p.port for p in m.ports if p.peer == "gpu:07"]
        for m in build_port_maps(topo)
    }
    assert hip5 == {"A": [], "B": [0]}
    for m in build_port_maps(topo):
        bdf = next(b for b in windows if b.endswith(m.usp_bdf))
        chip, _ = plx_fakes.chip_access(windows[bdf], bdf)
        assert set(hip5[m.switch]) <= chip.protected_ports
        # nothing else is protected: every other physical port keeps its counters
        other = {p.port for p in m.physical_ports() if p.peer != "gpu:07"}
        assert not other & chip.protected_ports


def test_subprocess_helper_refuses_the_hip5_port_asked_directly(two_chip_sysfs):
    def digest():
        return hashlib.sha256(
            (two_chip_sysfs / SW_B / "resource0").read_bytes()
        ).hexdigest()

    before = digest()
    client = access.SubprocessClient(
        {SW_A: 48, SW_B: 48},
        allow_writes=frozenset({SW_B}),
        sudo=False,
        sysfs=str(two_chip_sysfs),
        timeout_s=20,
    )
    try:
        info = client.hello()["chips"][SW_B]
        assert info["writes"] is True and info["protected_ports"] == list(range(16))
        for ports in ([0], [0, 16, 32]):
            with pytest.raises(AccessError, match="protected station"):
                client.perf_start(SW_B, ports)
        assert client.perf_info(SW_B)["running"] is False
    finally:
        client.close()
    assert digest() == before  # not one byte of switch B changed
