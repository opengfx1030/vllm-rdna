# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the pciescope TUI with textual's pilot against a recorded session."""

import threading
import time
from types import SimpleNamespace

import pciescope_tui_fixtures as fx
import pytest
from pciescope.model import Event, Measurement
from pciescope.tui import app as tui_app
from pciescope.tui import measure as tui_measure
from pciescope.tui.measure import MeasureRunner
from pciescope.tui.sources import ReplaySource
from pciescope_tui_fixtures import (
    Clock,
    FakeCollector,
    ListSession,
    drive,
    make_live,
    shown,
)


def replay(seconds=30, **kw):
    clock = Clock()
    src = ReplaySource(
        fx.box_topology(),
        fx.synthetic_records(seconds, **kw),
        path="/runs/20261107-demo",
        collectors=["sysfs_link", "aer", "amdgpu"],
        unavailable={"plx": "needs root"},
        clock=clock,
        start_paused=True,
    )
    return src, clock


def test_replay_views_at_80x24_match_golden():
    src, _ = replay()
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        src.seek_to(src.duration_s)
        app._on_tick()
        out = {}
        for key, name in (
            ("1", "topology"),
            ("2", "live"),
            ("3", "matrix"),
            ("4", "errors"),
            ("5", "measure"),
        ):
            out[name] = await shown(pilot, key)
        return out

    screens = drive(app, steps)
    for name, text in screens.items():
        lines = text.split("\n")
        assert len(lines) == 24
        assert all(len(x) <= 80 for x in lines)
        assert "REPLAY" in lines[0] and "pciescope" in lines[0]
        fx.check_golden(f"screen_{name}_80x24", text)
    assert "H2:d3" in screens["errors"]
    assert "Measure is disabled" in screens["measure"]
    assert "intra-switch" in screens["matrix"]


def test_replay_pause_seek_speed_and_rewind():
    src, clock = replay()
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        assert src.paused and src.pos_s == 0
        await shown(pilot, "right")
        assert src.pos_s == 10 and app.state.n_samples > 0
        before = app.state.n_samples
        await shown(pilot, "shift+right")
        assert src.pos_s == src.duration_s and app.state.n_samples > before
        await shown(pilot, "end")
        assert src.pos_s == src.duration_s and src.finished
        gen = src.generation
        await shown(pilot, "home")
        assert src.pos_s == 0 and src.generation == gen + 1
        assert app.state.last_ts == fx.T0  # rebuilt from history up to playhead
        assert app.state.value("gpu:d3", "aer.correctable") == 0
        await shown(pilot, "plus")
        await shown(pilot, "plus")
        assert src.speed == 4.0
        text = await shown(pilot, "space")  # resume
        assert not src.paused
        clock.t += 5.0  # 5 s real time at 4x = 20 s of session
        app._on_tick()
        assert src.pos_s == pytest.approx(20.0)
        assert "x4" in app.screen_text().split("\n")[0]
        return text

    drive(app, steps)


def test_keys_switch_views_and_help_and_quit():
    src, _ = replay()
    app = tui_app.PcieScopeApp(src)

    async def steps(pilot):
        t = await shown(pilot, "tab")
        assert app.view == "live" and "GPU" in t
        await shown(pilot, "shift+tab")
        assert app.view == "topology"
        await shown(pilot, "d")
        assert app.detail
        t = await shown(pilot, "question_mark")
        assert "pciescope TUI keys" in t
        t = await shown(pilot, "x")  # any key closes help
        assert "pciescope TUI keys" not in t
        await shown(pilot, "3")
        await shown(pilot, "m")
        assert app.sel.view == "eff"
        await shown(pilot, "v")
        await shown(pilot, "left_square_bracket")
        await pilot.press("q")

    drive(app, steps)


def test_export_key_calls_exporter_and_reports_path():
    src, _ = replay()
    calls = []

    def export():
        calls.append(1)
        return "/runs/x/trace.json"

    app = tui_app.PcieScopeApp(src, export=export)

    async def steps(pilot):
        t = await shown(pilot, "e")
        assert calls == [1]
        assert "exported x/trace.json" in t.split("\n")[0]

    drive(app, steps)


def test_export_failure_is_shown_not_raised():
    src, _ = replay()

    def boom():
        raise RuntimeError("disk full")

    app = tui_app.PcieScopeApp(src, export=boom)

    async def steps(pilot):
        t = await shown(pilot, "e")
        assert "export failed: disk full" in t.split("\n")[0]

    drive(app, steps)


def test_live_source_records_and_header_shows_rec():
    rec = ListSession()
    src = make_live([FakeCollector()], [("plx", "needs root")], session=rec)
    app = tui_app.PcieScopeApp(src, tick=60)

    async def steps(pilot):
        src.sample_once()
        src.sample_once()
        app._on_tick()
        await pilot.pause()
        head = app.screen_text().split("\n")[0]
        assert "LIVE" in head and "● REC 20261107-live" in head
        assert rec.metric("gpu.busy_pct") == [10, 20]
        assert rec.metric("gpu.temp_junction_c") == [50.0]  # deduplicated
        await shown(pilot, "2")
        row = next(x for x in app.screen_text().split("\n") if x.startswith("H1:ce"))
        assert "20%" in row and "50C" in row
        await shown(pilot, "space")  # freeze display, recording continues
        src.sample_once()
        app._on_tick()
        assert rec.metric("gpu.busy_pct") == [10, 20, 30]
        assert app.state.value("gpu:ce", "gpu.busy_pct") == 20.0
        assert "frozen" in app.screen_text().split("\n")[0]
        await shown(pilot, "space")
        app._on_tick()
        assert app.state.value("gpu:ce", "gpu.busy_pct") == 30.0
        await shown(pilot, "r")
        src.sample_once()
        assert rec.metric("gpu.busy_pct") == [10, 20, 30]  # recording off
        assert "not recording" in app.screen_text().split("\n")[0]

    drive(app, steps)
    assert rec.closed


def test_live_collector_failure_becomes_event_not_crash():
    class Bad(FakeCollector):
        def collect(self):
            raise OSError("sysfs gone")

    src = make_live([Bad()])
    ev = [r for r in src.sample_once() if isinstance(r, Event)]
    assert [r.name for r in ev] == ["collector_error"]
    assert "sysfs gone" in ev[0].message and ev[0].severity == "error"


# -- measure panel -----------------------------------------------------------


def fake_probe(held=True):
    suites = [
        SimpleNamespace(name="quick", description="lat + bw per pair", est_seconds=30),
        SimpleNamespace(name="full", description="everything", est_seconds=600),
    ]
    seen = {}

    def check_leases(gpus, owner=None):
        seen["check"] = (list(gpus), owner)
        if held:
            return SimpleNamespace(ok=True, needed=[], reason="", statuses=[])
        return SimpleNamespace(
            ok=False,
            needed=[f"{owner}.slotA"],
            reason="lease slotA not held",
            statuses=[],
        )

    def run_suite(
        suite,
        gpus,
        *,
        owner=None,
        topo=None,
        progress=None,
        cancel=None,
        dry_run=False,
        force_noisy=False,
        sink=None,
    ):
        seen["run"] = (suite, list(gpus), owner, dry_run)
        progress(Event(fx.T0, "host", "plan", "info", "", {"n": 2, "est_s": 3}))
        ms: list = []
        for i in (1, 2):
            progress(
                Event(
                    fx.T0,
                    "host",
                    "step_start",
                    "info",
                    "",
                    {"i": i, "n": 2, "scenario": f"lat {i}"},
                )
            )
            if cancel is not None and cancel.wait(0.05):
                return SimpleNamespace(
                    measurements=ms,
                    status="aborted",
                    refusal=None,
                    valid=True,
                    invalid_reasons=[],
                )
            m = Measurement(
                "lat",
                "gpu:ce",
                "gpu:d3",
                "d2d-push",
                4096 * i,
                6.2,
                "us",
                {},
                "intra-switch",
                None,
                None,
                fx.T0 + i,
                fx.T0 + i + 1,
                ["gpu:ce", "gpu:d3"],
            )
            ms.append(m)
            sink(m)
            progress(
                Event(
                    fx.T0,
                    "host",
                    "step_done",
                    "info",
                    "",
                    {"i": i, "n": 2, "scenario": f"lat {i}"},
                )
            )
        return SimpleNamespace(
            measurements=ms, status="ok", refusal=None, valid=True, invalid_reasons=[]
        )

    mod = SimpleNamespace(
        list_suites=lambda: suites, check_leases=check_leases, run_suite=run_suite
    )
    return mod, seen


def wait_status(runner, want, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        runner.drain()
        if runner.model.status in want:
            return
        time.sleep(0.02)
    raise AssertionError(f"status {runner.model.status}, wanted {want}")


def test_measure_start_refused_without_lease(monkeypatch):
    mod, seen = fake_probe(held=False)
    monkeypatch.setattr(tui_measure, "_probe", lambda: mod)
    topo = fx.box_topology()
    src = make_live([FakeCollector()], topo=topo)
    runner = MeasureRunner(topo, owner="PcieTui", sink=src.publish)
    app = tui_app.PcieScopeApp(src, runner=runner, tick=60)

    async def steps(pilot):
        t = await shown(pilot, "5")
        assert "needs: PcieTui.slotA" in t and "lease slotA not held" in t
        t = await shown(pilot, "s")
        assert "run" not in seen
        assert "refused: lease slotA not held" in t
        assert runner.model.status == "idle"
        fx.check_golden("screen_measure_refused_80x24", t)

    drive(app, steps)


def test_measure_runs_suite_streams_progress_and_feeds_matrix(monkeypatch):
    mod, seen = fake_probe(held=True)
    monkeypatch.setattr(tui_measure, "_probe", lambda: mod)
    topo = fx.box_topology()
    rec = ListSession()
    src = make_live([FakeCollector()], topo=topo, session=rec)
    runner = MeasureRunner(topo, owner="PcieTui", sink=src.publish)
    app = tui_app.PcieScopeApp(src, runner=runner, tick=60)

    async def steps(pilot):
        await shown(pilot, "5")
        await shown(pilot, "down")
        await shown(pilot, "up")
        assert runner.model.sel == 0
        await shown(pilot, "s")
        wait_status(runner, {"done"})
        app._on_tick()
        t = await shown(pilot, "5")
        assert seen["run"] == (
            "quick",
            ["gpu:ce", "gpu:d3", "gpu:d8", "gpu:db"],
            "PcieTui",
            False,
        )
        assert "finished: ok, 2 results" in t and "[1/2] start lat 1" in t
        assert len(app.state.measurements) == 2
        t = await shown(pilot, "3", "m", "m")
        assert "latency us" in t and "6.2" in t

    drive(app, steps)
    assert any(isinstance(r, Measurement) for r in rec.records)


def test_measure_cancel_stops_after_current_cell(monkeypatch):
    mod, _ = fake_probe(held=True)
    monkeypatch.setattr(tui_measure, "_probe", lambda: mod)
    runner = MeasureRunner(fx.box_topology(), owner="PcieTui")
    assert runner.start() is None
    runner.cancel()
    wait_status(runner, {"aborted"})
    assert any("cancel requested" in x for x in runner.model.log)


def test_measure_gpu_presets_cycle_and_refresh_lease(monkeypatch):
    mod, seen = fake_probe(held=True)
    monkeypatch.setattr(tui_measure, "_probe", lambda: mod)
    runner = MeasureRunner(fx.box_topology(), owner="o")
    assert runner.model.gpus[0] == "gpu:ce"  # slot A first
    runner.cycle_gpus()
    assert runner.model.gpus == ["gpu:0a", "gpu:0f", "gpu:14", "gpu:17"]
    assert seen["check"][0] == ["0a", "0f", "14", "17"]
    offered = {g for _, ids in runner.presets for g in ids}
    assert "gpu:07" not in offered and "gpu:cb" in offered  # HIP 5 never


def test_measure_without_probe_module_says_so(monkeypatch):
    def nope():
        raise ImportError("no module pciescope.probe")

    monkeypatch.setattr(tui_measure, "_probe", nope)
    runner = MeasureRunner(fx.box_topology(), owner="o")
    assert not runner.available and "probe unavailable" in runner.model.error
    assert runner.start().startswith("no suites") or "unavailable" in runner.start()
    assert threading.active_count() >= 1
