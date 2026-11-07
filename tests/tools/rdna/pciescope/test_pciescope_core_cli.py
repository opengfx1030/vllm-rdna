# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI verbs on the captured par1-cs25 sysfs (stdout is data, logs go to stderr)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pciescope import capture, cli, store, topology
from pciescope.sysfs import Sysfs


@pytest.fixture()
def run(fx_root, tmp_path, capsys):
    """Run ``cli.main`` against the fixture; returns (exit code, stdout, stderr)."""
    base = [
        "--sysfs-root",
        str(fx_root / "sys"),
        "--proc-root",
        str(fx_root / "proc"),
        "--host",
        "par1-cs25",
        "--root",
        str(tmp_path / "store"),
    ]

    def go(verb: str, *args: str, common: bool = True) -> tuple[int, str, str]:
        capsys.readouterr()
        argv = [verb, *args, *(base if common else [])]
        code = cli.main(argv)
        out = capsys.readouterr()
        return code, out.out, out.err

    return go


def test_no_verb_is_a_usage_error(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "usage" in capsys.readouterr().err.lower()


def test_topo_text_and_json(run):
    code, out, _ = run("topo")
    assert code == 0
    assert "2 root complex(es), 2 switch(es), 10 GPU(s)" in out
    assert "gpu:ce AMD RADEON PRO V620 Azure gfx1030 HIP 1" in out
    code, out, _ = run("topo", "--json")
    doc = json.loads(out)
    assert code == 0
    assert len(doc["gpus"]) == 10
    assert doc["gpus"]["gpu:cb"]["hip"] == 0
    code, full, _ = run("topo", "--full")
    assert len(full) > len(run("topo")[1])


def test_topo_without_gpus_fails(tmp_path, capsys):
    code = cli.main(["topo", "--sysfs-root", str(tmp_path), "--json"])
    cap = capsys.readouterr()
    assert code == cli.EXIT_FAIL
    assert "no AMD GPU found" in cap.err
    assert json.loads(cap.out)["gpus"] == {}  # the document is still printed


def test_topo_from_a_file(run, tmp_path):
    _, out, _ = run("topo", "--json")
    p = tmp_path / "topo.json"
    p.write_text(out)
    code, text, _ = run("topo", "--topology-from", str(p), common=False)
    assert code == 0
    assert "root complex c0 (switch A)" in text


def test_snap_json_summary(run):
    code, out, err = run("snap", "--json")
    assert code == 0, err
    doc = json.loads(out)
    assert doc["host"] == "par1-cs25"
    assert doc["collectors"] == ["sysfs_link", "aer", "amdgpu", "kfd"]
    assert doc["skipped"] == {}
    assert len(doc["gpus"]) == 10
    gpu = {g["id"]: g for g in doc["gpus"]}
    assert gpu["gpu:ce"]["hip"] == 1
    assert gpu["gpu:ce"]["gpu_link"] == {
        "speed_gts": 16,
        "width": 16,
        "dpm_speed_gts": 16,
        "dpm_width": 16,
    }
    assert gpu["gpu:ce"]["vram_total_bytes"] == 32195477504
    assert len(doc["links"]) == 22
    assert all(lk["degraded"] is False for lk in doc["links"])
    assert doc["aer"]["devices_checked"] == 66
    assert any(e["entity"] == "port:c0:03.1" for e in doc["aer"]["nonzero"])
    assert set(doc["cost_ms"]) == {"sysfs_link", "aer", "amdgpu", "kfd"}
    assert "samples" not in doc  # raw samples only on request
    assert doc["events"] == []


def test_snap_raw_and_filters(run):
    code, out, _ = run(
        "snap", "--json", "--metric", "link.speed", "--entity", "link:c8"
    )
    doc = json.loads(out)
    assert code == 0
    assert {s["metric"] for s in doc["samples"]} == {"link.speed_gts"}
    assert all(s["entity"].startswith("link:c8") for s in doc["samples"])
    assert len(doc["samples"]) == 2  # the card links of cb (c8:00.0) and ce (c8:10.0)
    code, out, _ = run("snap", "--raw", "--collectors", "aer")
    assert code == 0
    assert "aer.rootport_correctable" in out


def test_snap_collector_selection_and_failure(run):
    code, out, err = run("snap", "--json", "--collectors", "kfd")
    assert code == 0
    assert json.loads(out)["collectors"] == ["kfd"]
    code, out, err = run("snap", "--json", "--collectors", "nope")
    assert code == cli.EXIT_FAIL
    assert "unknown collector" in err
    assert out == ""


def test_snap_text_table(run):
    code, out, _ = run("snap")
    assert code == 0
    assert "links: 22 physical, 22 at their maximum" in out
    assert "gpu:ce" in out
    assert "AER:" in out


def test_watch_records_a_session(run, tmp_path):
    code, out, err = run("watch", "--duration", "0.4", "--interval", "0.1", "--json")
    assert code == 0, err
    doc = json.loads(out)
    assert doc["status"] == "ok"
    assert doc["ticks"] >= 3
    assert doc["samples_written"] < doc["samples_seen"]  # change filter at work
    assert doc["collectors"] == ["sysfs_link", "aer", "amdgpu", "kfd"]
    assert set(doc["collector_cost_ms"]) == {"sysfs_link", "aer", "amdgpu", "kfd"}
    assert [f["code"] for f in doc["findings"]] == ["aer_nonzero_since_boot"]
    path = tmp_path / "store" / doc["session"]
    assert str(path) == doc["path"]
    sess = store.Session.open(path)
    assert sess.meta["status"] == "ok"
    assert sess.topology() is not None
    names = [e.name for e in sess.events()]
    assert names == ["watch_start", "watch_stop"]
    assert "recording to" in err


def test_watch_variants(run, tmp_path):
    code, out, _ = run("watch", "--ticks", "2", "--interval", "0.05", "--no-record")
    assert code == 0
    assert "watch ok: 2 ticks" in out
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").iterdir())
    code, out, err = run(
        "watch", "--ticks", "2", "--interval", "0.05", "--stream", "--full",
        "--session", "named", "--label", "x",
    )  # fmt: skip
    assert code == 0
    lines = [json.loads(ln) for ln in out.splitlines()]
    assert {ln["type"] for ln in lines} >= {"sample"}
    assert sum(1 for ln in lines if ln.get("metric") == "link.speed_gts") == 22 * 2
    assert (tmp_path / "store" / "named").is_dir()
    assert '"ticks":2' in err  # the summary goes to stderr when streaming


def test_watch_without_collectors_fails(run):
    code, _, err = run("watch", "--ticks", "1", "--collectors", "nope", "--no-record")
    assert code == cli.EXIT_FAIL
    assert "no collector is available" in err


def test_report_trace_sessions_metrics(run, tmp_path):
    _, out, _ = run("watch", "--ticks", "3", "--interval", "0.05", "--json")
    name = json.loads(out)["session"]
    code, out, _ = run("report", name, "--json")
    rep = json.loads(out)
    assert code == 0
    assert rep["session"]["name"] == name
    assert len(rep["links"]) == 22
    assert rep["ceilings"]["gen4_x16_practical_gbs"] == pytest.approx(28.03, abs=0.01)
    code, text, _ = run("report")  # latest
    assert code == 0
    assert text.startswith("pciescope report:")
    code, out, _ = run("report", "--virtual", "--json")
    assert len(json.loads(out)["links"]) == 32
    code, out, _ = run("trace", "export", name, "--json")
    info = json.loads(out)
    assert code == 0
    doc = json.loads(Path(info["path"]).read_text())
    assert any(e["ph"] == "C" for e in doc["traceEvents"])
    code, out, _ = run("sessions", "--json")
    assert [s["name"] for s in json.loads(out)["sessions"]] == [name]
    code, out, _ = run("metrics", "--json")
    names = {m["name"] for m in json.loads(out)}
    assert {"link.speed_gts", "aer.correctable", "kfd.evicted_ms"} <= names
    code, out, _ = run("sessions")
    assert name in out


def test_report_of_a_missing_session_fails(run):
    code, out, err = run("report", "does-not-exist")
    assert code == cli.EXIT_FAIL
    assert "session not found" in err
    assert out == ""


def test_measure_without_probe_module_fails(run, no_probe):
    code, out, err = run("measure")
    assert code == cli.EXIT_FAIL
    assert "the probe module is not available" in err
    assert out == ""


def test_measure_delegates_to_the_probe(run, fake_probe, tmp_path):
    code, out, err = run(
        "measure", "--suite", "standard", "--gpus", "ce", "cb", "--json", "-q",
        "--owner", "me", "--repeats", "2",
    )  # fmt: skip
    assert code == 0, err
    assert json.loads(out) == {"status": "x", "gpus": ["ce", "cb"], "dry_run": False}
    args, session = fake_probe.calls[0]
    # the probe's own --quiet replaced the common one; -q still works
    assert args.quiet and args.json and args.suite == "standard"
    assert args.owner == "me" and args.repeats == 2
    assert session is not None  # a real run records into a session
    assert json.loads((session.path / "session.json").read_text())["status"] == "ok"
    code, out, _ = run("measure", "--gpus", "ce", "--dry-run", "--quiet")
    assert code == 0
    assert fake_probe.calls[1][1] is None  # dry runs write nothing
    code, _, _ = run("measure", "--gpus", "ce", "--owner", "refuse", "--no-record")
    assert code == 2  # the probe's exit code is passed through
    assert fake_probe.calls[2][1] is None
    run("watch", "--ticks", "1", "--interval", "0.05", "--session", "named-ses")
    code, _, _ = run("measure", "--gpus", "ce", "--session", "named-ses")
    assert code == 0
    assert fake_probe.calls[3][1].name == "named-ses"


def test_capture_replays_to_the_same_topology(run, fx_root, tmp_path):
    out_dir = tmp_path / "cap"
    tar = tmp_path / "cap.tar.gz"
    code, out, _ = run("capture", str(out_dir), "--tar", str(tar), "--json")
    info = json.loads(out)
    assert code == 0
    assert info["gpus"] == 10
    assert info["redacted_ids"] == 20
    assert info["skipped"] == {}
    replay = topology.discover(Sysfs(out_dir / "sys"), host="par1-cs25")
    assert len(replay.gpus) == 10
    assert all(g.unique_id and len(g.unique_id) == 16 for g in replay.gpus.values())
    orig = topology.discover(Sysfs(fx_root / "sys"), host="par1-cs25")
    assert {g.unique_id for g in replay.gpus.values()} != {
        g.unique_id for g in orig.gpus.values()
    }  # redacted
    tar2 = tmp_path / "cap2.tar.gz"
    capture.write_tarball(out_dir, tar2)
    assert tar.read_bytes() == tar2.read_bytes()  # deterministic
