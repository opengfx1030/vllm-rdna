# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TUI <-> Core glue: record a live session, replay it, export a trace."""

import json
import sys
import types
from typing import Any

import pciescope_tui_fixtures as fx
import pytest
from pciescope import store
from pciescope.tui import app as tui_app
from pciescope.tui import session
from pciescope.tui.sources import ReplaySource, record_ts
from pciescope_tui_fixtures import FakeCollector, drive, make_live, shown


def write_session(root, records=None, topo=None, **kw):
    topo = topo or fx.box_topology()
    sess = store.Session.create(
        root,
        "tui-test",
        topo=topo,
        collectors=["sysfs_link", "aer", "amdgpu"],
        skipped={"plx": "needs root: BAR0 read"},
        interval_s=1.0,
        **kw,
    )
    sess.append_many(fx.synthetic_records(30) if records is None else records)
    sess.close()
    return sess.path


def open_replay(ref, **kw):
    args = dict(
        replay=ref,
        interval=1.0,
        collectors=None,
        session_dir=None,
        sysfs_root=None,
        owner="",
        record=False,
        dry_run=False,
        speed=1.0,
    )
    args.update(kw)
    return session.open_session(**args)


def test_replay_of_a_recorded_session_roundtrips_through_the_store(tmp_path):
    path = write_session(tmp_path)
    setup = open_replay(str(path))
    src = setup.source
    assert isinstance(src, ReplaySource) and setup.runner is None
    assert src.topology_ok if hasattr(src, "topology_ok") else True
    assert src.topo.host == "par1-cs25" and len(src.topo.gpus) == 10
    assert src.collectors == ["sysfs_link", "aer", "amdgpu"]
    assert src.unavailable == {"plx": "needs root: BAR0 read"}
    assert src.records == sorted(src.records, key=record_ts)
    assert src.duration_s == pytest.approx(29.0, abs=1.0)


def test_replay_by_session_name_and_latest(tmp_path):
    path = write_session(tmp_path)
    assert open_replay(path.name, session_dir=str(tmp_path)).source.path == str(path)
    assert open_replay("latest", session_dir=str(tmp_path)).source.path == str(path)


def test_replay_errors_are_user_facing(tmp_path, capsys):
    with pytest.raises(session.SessionError, match="session not found"):
        open_replay(str(tmp_path / "nope"))
    empty = write_session(tmp_path, records=[])
    with pytest.raises(session.SessionError, match="no records"):
        open_replay(str(empty))
    notopo = write_session(tmp_path, topo=fx.box_topology())
    (notopo / "topology.json").unlink()
    with pytest.raises(session.SessionError, match="no topology.json"):
        open_replay(str(notopo))
    assert tui_app.run(replay=str(tmp_path / "nope")) == 2
    assert "pciescope tui: session not found" in capsys.readouterr().err


def test_replay_app_shows_recorded_session_and_unavailable_reason(tmp_path):
    path = write_session(tmp_path)
    src = open_replay(str(path)).source
    src.paused = True
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        src.seek_to(src.duration_s)
        app._on_tick()
        t = await shown(pilot, "2")
        assert "plx: needs root: BAR0 read" in t
        assert path.name[:12] in t.split("\n")[0]
        t = await shown(pilot, "3")
        assert "intra-switch" in t and "cross-root" in t
        t = await shown(pilot, "4")
        assert "aer.correctable" in t and "link_change" in t

    drive(app, steps)


def test_export_uses_core_trace_exporter(tmp_path, monkeypatch):
    path = write_session(tmp_path)
    calls = []

    def export_chrome(session_path, out_path):
        calls.append((str(session_path), str(out_path)))
        return out_path

    fake: Any = types.ModuleType("pciescope.trace")
    fake.export_chrome = export_chrome
    monkeypatch.setitem(sys.modules, "pciescope.trace", fake)
    monkeypatch.setattr("pciescope.trace", fake, raising=False)
    setup = open_replay(str(path))
    app = tui_app.PcieScopeApp(setup.source, export=setup.export)

    async def steps(pilot):
        t = await shown(pilot, "e")
        assert calls == [(str(path), str(path / "trace.perfetto.json"))]
        assert "exported" in t.split("\n")[0]

    drive(app, steps)


def test_live_recording_writes_a_replayable_session(tmp_path):
    topo = fx.box_topology()
    sess = store.Session.create(tmp_path, "tui", topo=topo, collectors=["fake"])
    src = make_live([FakeCollector()], topo=topo, session=sess)
    src.sample_once()
    src.sample_once()
    assert src.session_path == str(sess.path)
    src.close()  # closes + finalises the session
    meta = json.loads((sess.path / "session.json").read_text())
    assert meta["status"] == "ok"
    replay = open_replay(str(sess.path)).source
    assert [
        r.value for r in replay.records if getattr(r, "metric", "") == "gpu.busy_pct"
    ] == [10, 20]
    assert replay.collectors == ["fake"]


def test_toggle_record_creates_a_new_session_when_started_without_one(tmp_path):
    topo = fx.box_topology()
    made = []

    def factory():
        sess = store.Session.create(tmp_path, "tui-late", topo=topo)
        made.append(sess)
        return sess

    src = make_live([FakeCollector()], topo=topo, session_factory=factory)
    src.sample_once()  # not recorded
    src.toggle_record()
    assert src.recording and len(made) == 1
    src.sample_once()
    src.sample_once()
    src.toggle_record()
    src.sample_once()  # recording paused again
    src.close()
    got = list(store.Session.open(str(made[0].path)).records())
    busy = [r.value for r in got if getattr(r, "metric", "") == "gpu.busy_pct"]
    assert busy == [20, 30]
    # the resumed session starts with a full snapshot (fresh dedup state)
    assert any(getattr(r, "metric", "") == "gpu.temp_junction_c" for r in got)


def real_session(tmp_path):
    """A 34 s passive recording captured by this TUI on par1-cs25 (2026-11-07)."""
    import tarfile

    with tarfile.open(fx.FIXTURES / "tui_real_session_par1-cs25.tgz") as tf:
        tf.extractall(tmp_path / "real", filter="data")
    return tmp_path / "real"


def test_real_box_session_replays_and_matches_golden_screens(tmp_path):
    src = open_replay(str(real_session(tmp_path))).source
    assert src.collectors == ["sysfs_link", "aer", "amdgpu", "kfd"]
    assert src.duration_s == pytest.approx(33.0, abs=1.5)
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        src.seek_to(src.duration_s)
        app._on_tick()
        out = {}
        for key, name in (("1", "topology"), ("2", "live"), ("4", "errors")):
            out[name] = await shown(pilot, key)
        return out

    screens = drive(app, steps)
    for name, text in screens.items():
        fx.check_golden(f"screen_real_{name}_80x24", text)
    assert "H5:07" in screens["topology"] and "W7800" in screens["topology"]
    assert "220 error series watched" in screens["errors"]
    assert "amdgpu pcie_bw not exposed" in screens["live"]
    st = app.state
    assert st.value("gpu:ce", "gpu.vram_total_bytes") == 32195477504
    assert st.value(fx.box_topology().gpus["gpu:0a"].link, "link.speed_gts") == 16.0


def test_real_box_session_rewind_rebuilds_state(tmp_path):
    src = open_replay(str(real_session(tmp_path))).source
    src.paused = True
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        src.seek_to(20)
        app._on_tick()
        late = app.state.n_samples
        src.seek_to(5)
        app._on_tick()
        assert 0 < app.state.n_samples < late
        assert app.state.last_ts <= src.playhead

    drive(app, steps)
