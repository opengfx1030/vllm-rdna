# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope.plx`` verbs on the replayed capture, and the fixture format."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import plx_fakes
import pytest
from pciescope.model import Topology
from pciescope.plx import access, cli, fixture, regs
from pciescope.plx.ports import build_port_maps

FIX = str(plx_fakes.CAPTURE)
RDNA = Path(__file__).resolve().parents[4] / "tools" / "rdna"


def run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


# -- ports / status / dump --------------------------------------------------------


def test_ports_text_lists_the_map(capsys):
    rc, out, _ = run(capsys, "ports", "--fixture", FIX)
    assert rc == 0
    assert "switch A  root c0  upstream c5:00.0 (port 48)" in out
    assert "switch B  root 00  upstream 01:00.0 (port 48)" in out
    assert "   0 0.0 04:00.0  dsp      gpu:07" in out
    assert "  16 1.0 c8:10.0  dsp      gpu:ce" in out
    assert " 247   - e0:00.0  mgmt" in out


def test_ports_json_and_switch_filter(capsys):
    rc, out, _ = run(capsys, "ports", "--fixture", FIX, "--json", "--switch", "b")
    assert rc == 0
    (chip,) = json.loads(out)
    assert chip["switch"] == "B" and chip["usp_bdf"] == "01:00.0"
    gpu = {p["port"]: p["peer"] for p in chip["ports"] if p["peer"]}
    assert gpu == {
        0: "gpu:07",
        16: "gpu:0a",
        32: "gpu:0f",
        48: "host",
        64: "gpu:14",
        80: "gpu:17",
    }
    with pytest.raises(SystemExit, match="no switch 'C'"):
        cli.main(["ports", "--fixture", FIX, "--switch", "C"])


def test_status_json(capsys):
    rc, out, _ = run(capsys, "status", "--fixture", FIX, "--json")
    assert rc == 0
    doc = json.loads(out)
    assert set(doc) == {"chips", "ports", "events", "collect_ms", "samples"}
    assert doc["samples"] == 382 and len(doc["ports"]) == 36
    assert [c["serial"] for c in doc["chips"]] == [
        "00-80-5e-10-fa-53-6b-2a",
        "00-80-5e-10-f9-57-20-61",
    ]
    assert {e["name"] for e in doc["events"]} == {"plx_error_flags"}
    gpu = next(p for p in doc["ports"] if p["bdf"] == "04:10.0")
    assert gpu["state"]["speed_gts"] == 16.0 and gpu["state"]["eq_complete"] is True


def test_status_text(capsys):
    rc, out, err = run(capsys, "status", "--fixture", FIX)
    assert rc == 0 and err == ""
    assert (
        "switch A  upstream c5:00.0  serial 00-80-5e-10-fa-53-6b-2a  "
        "traffic counters off"
    ) in out
    assert "switch B  upstream 01:00.0  serial 00-80-5e-10-f9-57-20-61" in out
    lines = {tuple(x.split()[:3]): x for x in out.splitlines() if x.startswith("  ")}
    assert "16 GT/s x16" in lines["16", "04:10.0", "dsp"]
    assert "CorrErr,UnsupReq" in lines["16", "04:10.0", "dsp"]
    assert "down" in lines["116", "19:14.0", "dsp"]
    assert "virtual" in lines["224", "02:00.0", "int_dsp"]
    assert "382 samples in " in out
    assert "event plx_error_flags port:04:10.0" in out


def test_status_switch_filter(capsys):
    rc, out, _ = run(capsys, "status", "--fixture", FIX, "--json", "--switch", "b")
    assert rc == 0
    doc = json.loads(out)
    assert [c["switch"] for c in doc["chips"]] == ["B"]
    assert {p["switch"] for p in doc["ports"]} == {"B"} and len(doc["ports"]) == 18
    assert len(doc["events"]) == 5 and {e["data"]["switch"] for e in doc["events"]} == {
        "B"
    }
    rc, out, _ = run(capsys, "status", "--fixture", FIX, "--switch", "A")
    assert "switch A" in out and "switch B" not in out and "event " not in out
    with pytest.raises(SystemExit, match="no switch 'Z'"):
        cli.main(["status", "--fixture", FIX, "--switch", "Z"])


def test_dump_decodes_capabilities(capsys):
    rc, out, _ = run(
        capsys, "dump", "16", "--fixture", FIX, "--switch", "B", "--decode", "--json"
    )
    assert rc == 0
    doc = json.loads(out)
    assert doc["bdf"] == "04:10.0" and doc["port"] == 16
    assert doc["dwords"]["000"] == "c0101000"
    assert doc["caps"]["pcie"] == 0x68 and doc["caps"]["aer"] == 0xFB4
    assert all(len(k) == 3 and len(v) == 8 for k, v in doc["dwords"].items())
    rc, out, _ = run(capsys, "dump", "48", "--fixture", FIX, "--switch", "A")
    assert rc == 0 and "000: c0101000" in out
    with pytest.raises(SystemExit, match="has no port 5"):
        cli.main(["dump", "5", "--fixture", FIX])


# -- perf -------------------------------------------------------------------------


def test_perf_plan_lists_the_writes_without_writing(capsys):
    rc, out, _ = run(capsys, "perf", "plan", "--fixture", FIX, "--json")
    assert rc == 0  # a fixture client is read-only: any write would have raised
    doc = json.loads(out)
    assert doc["B"]["ports"] == [16, 32, 48, 64, 80]  # gpu:07 is always excluded
    assert doc["A"]["ports"] == [0, 16, 32, 48, 64, 80]
    steps = doc["B"]["steps"]
    assert [s["step"] for s in steps].count("ingress_enable") == 5
    assert {(s["reg"], s["old"], s["new"]) for s in steps} == {
        (0x768, 0x09000000, 0x29000000),
        (0xF30, 0, 0x00200040),
        (0x3E0, 0, 0xD8000000),
    }
    assert {s["reg"] for s in doc["B"]["per_poll"]} == {0x3F0}
    assert doc["B"]["monitor_already_running"] is False


def test_perf_plan_text_and_extra_exclusion(capsys):
    rc, out, _ = run(
        capsys, "perf", "plan", "--fixture", FIX, "--switch", "B", "--exclude", "gpu:0a"
    )
    assert rc == 0
    assert "switch B: counters on ports [32, 48, 64, 80]" in out
    assert "ingress_enable" in out and "29000000" in out and "d8000000" in out
    assert "switch A" not in out


def test_perf_run_refuses_without_the_flag(capsys):
    rc, out, err = run(capsys, "perf", "run", "--fixture", FIX)
    assert rc == 2 and out == ""
    assert err.strip() == cli.WRITE_REFUSAL
    rc, _, err = run(capsys, "perf", "run", "--fixture", FIX, "--allow-switch-writes")
    assert rc == 2 and "needs the real hardware" in err


# -- fixture ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["again.json", "again.json.gz"])
def test_capture_from_a_replay_round_trips(tmp_path, capsys, name):
    out = tmp_path / name
    rc, text, _ = run(capsys, "capture", "--fixture", FIX, "--out", str(out))
    assert rc == 0 and "2 switches" in text
    original = fixture.read_doc(FIX)
    again = fixture.read_doc(out)
    assert again["schema"] == fixture.SCHEMA == 1
    assert again["chips"] == original["chips"]
    assert Topology.from_dict(again["topology"]) == Topology.from_dict(
        original["topology"]
    )
    assert (out.read_bytes()[:2] == b"\x1f\x8b") == name.endswith(".gz")


def test_gzip_fixture_bytes_are_reproducible(tmp_path):
    doc = fixture.read_doc(FIX)
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    fixture.save(doc, a)
    fixture.save(doc, b)
    assert a.read_bytes() == b.read_bytes()


def test_captured_windows_hold_no_read_sensitive_register():
    doc = fixture.read_doc(FIX)
    for chip in doc["chips"].values():
        for win in chip["windows"].values():
            assert not {int(o, 16) for o in win} & regs.READ_SENSITIVE


def test_fixture_load_gives_a_read_only_client():
    topo, client = fixture.load(FIX)
    assert [m.switch for m in build_port_maps(topo)] == ["A", "B"]
    assert client.read(plx_fakes.SWITCH_B, [(48, 0)]) == [regs.ID_DWORD]
    with pytest.raises(access.AccessError, match="writes are disabled"):
        client.perf_start(plx_fakes.SWITCH_B, [16])


def test_fixture_with_a_future_schema_is_rejected(tmp_path):
    doc = fixture.read_doc(FIX)
    doc["schema"] = 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unsupported fixture schema 2"):
        fixture.load(bad)


def test_module_entry_point_runs():
    env = {**os.environ, "PYTHONPATH": str(RDNA)}
    proc = subprocess.run(
        [sys.executable, "-m", "pciescope.plx", "ports", "--fixture", FIX, "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert [c["switch"] for c in json.loads(proc.stdout)] == ["A", "B"]
