# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure render functions of the pciescope TUI against recorded-session fixtures."""

import pciescope_tui_fixtures as fx
import pytest
from pciescope.model import Event, Measurement, Sample
from pciescope.tui import fmt, render
from pciescope.tui.state import ViewState, error_change


@pytest.fixture(scope="module")
def topo():
    return fx.box_topology()


def make_state(plx=False, **kw):
    st = ViewState()
    st.ingest(fx.synthetic_records(30, plx=plx, **kw))
    st.collectors = ["sysfs_link", "aer", "amdgpu"]
    st.unavailable = {"plx": "needs root: BAR0 read"}
    return st


@pytest.fixture(scope="module")
def state():
    return make_state()


def plain(text, width=80):
    out = render.to_plain(text, width)
    assert all(len(line) <= width for line in out.split("\n")), out
    return out


def test_sparkline_scale_and_padding():
    assert fmt.sparkline([], 5) == "     "
    assert fmt.sparkline([0, 50, 100], 5, 100) == "  ▁▅█"
    assert fmt.sparkline(list(range(100)), 4)[-1] == "█"
    assert len(fmt.sparkline([1, 2, 3], 10)) == 10


def test_error_change_rules():
    assert error_change("aer.correctable", 3, 3) is None
    assert error_change("aer.correctable", 3, 5) == ("", 2.0)
    assert error_change("plx.err.dev_sta", 1, 1) is None
    text, delta = error_change("plx.err.dev_sta", 1, 3)
    assert "NF" in text and delta == 2.0


def test_state_series_keep_tags_apart():
    st = ViewState()
    for typ, v in (("BadTLP", 2), ("RxErr", 5)):
        st.ingest_sample(Sample(1, "gpu:ce", "aer.error", v, "count", {"type": typ}))
    assert len(st.latest) == 2
    assert [x[1] for x in st.counters_nonzero()] == [
        "aer.error[BadTLP]",
        "aer.error[RxErr]",
    ]


def test_state_rate_and_errors_ordering():
    st = ViewState()
    st.ingest_sample(Sample(0, "port:c8:10.0", "plx.perf.in_bytes", 0, "B"))
    st.ingest_sample(
        Sample(2_000_000_000, "port:c8:10.0", "plx.perf.in_bytes", 8e9, "B")
    )
    assert st.rate("port:c8:10.0", "plx.perf.in_bytes") == pytest.approx(4e9)
    st.ingest_event(Event(5, "host", "thermal", "warn", "hot"))
    st.ingest_event(Event(6, "host", "session_start", "info"))
    assert [e.what for e in st.errors] == ["thermal"]


def test_topology_golden(topo, state):
    out = plain(render.render_topology(topo, state, 80))
    assert "switch A" in out and "switch B" in out
    assert out.count("V620") == 8 and out.count("W7800") == 2
    fx.check_golden("topology_80", out)


def test_topology_detail_and_wide(topo, state):
    fx.check_golden(
        "topology_detail_80", plain(render.render_topology(topo, state, 80, True))
    )
    wide = plain(render.render_topology(topo, state, 120), 120)
    assert "smi" in wide
    fx.check_golden("topology_120", wide)


def test_topology_marks_degraded_link(topo):
    st = make_state(errors=False)
    st.ingest_sample(
        Sample(
            fx.T0 + 99 * fx.NS, topo.gpus["gpu:0f"].link, "link.speed_gts", 8.0, "GT/s"
        )
    )
    st.ingest_sample(
        Sample(fx.T0 + 99 * fx.NS, "gpu:0f", "pcie.link_speed_gts", 8.0, "GT/s")
    )
    out = plain(render.render_topology(topo, st, 80))
    line = next(x for x in out.split("\n") if "H7:0f" in x)
    assert "▲" in line and "8Gx16" in line and "max 16G" in line


def test_live_view_na_reason_and_values(topo, state):
    out = plain(render.render_live(topo, state, 80))
    assert "n/a" in out and "pcie_bw not exposed" in out and "plx: needs root" in out
    assert out.splitlines()[0].startswith("GPU")
    fx.check_golden("live_80", out)


def test_live_view_with_switch_counters(topo):
    st = make_state(plx=True)
    out = plain(render.render_live(topo, st, 80))
    assert "RX/TX n/a" not in out
    assert "switch-port GB/s" in out
    fx.check_golden("live_plx_80", out)


def test_live_view_scales_to_wide_terminal(topo, state):
    out = plain(render.render_live(topo, state, 160), 160)
    first = out.splitlines()[2]
    assert len(first.rstrip()) > 100
    fx.check_golden("live_160", out)


@pytest.mark.parametrize("view,size_idx", [("bw", -1), ("eff", -1), ("lat", 0)])
def test_matrix_views(topo, state, view, size_idx):
    sel = render.MatrixSel(view, size_idx=size_idx)
    out = plain(render.render_matrix(topo, state, 80, sel))
    assert "×" in out and "cross-root" in out and "intra-switch" in out
    fx.check_golden(f"matrix_{view}_80", out)


def test_matrix_pull_direction_shows_cross_root_collapse(topo, state):
    sel = render.MatrixSel("bw", "d2d-pull")
    out = plain(render.render_matrix(topo, state, 80, sel))
    assert " 5.7×" in out or " 5.8×" in out
    summary = [x for x in out.split("\n") if x.startswith("cross-root")][0]
    assert "median" in summary and "eff  19%" in summary


def test_matrix_empty_without_results(topo):
    st = make_state(probes=False)
    out = plain(render.render_matrix(topo, st, 80))
    assert "No bandwidth GB/s results yet" in out


def test_matrix_selection_cycles(topo, state):
    sel = render.MatrixSel()
    dirs, sizes = render.matrix_choices(state, "bw")
    assert dirs == ["d2d-push", "d2d-pull"] and sizes == [64 * 2**20]
    sel.cycle_direction(dirs)
    assert sel.direction == "d2d-pull"
    sel.cycle_view()
    assert sel.view == "eff"


def test_matrix_uses_latest_measurement_per_cell(topo):
    st = ViewState()
    mk = lambda v, t: Measurement(
        "bw",
        "gpu:ce",
        "gpu:d3",
        "d2d-push",
        1 << 20,
        v,
        "GB/s",
        {},
        "intra-switch",
        31.5,
        v / 31.5,
        t,
        t + 1,
    )
    st.ingest_measurement(mk(10.0, 1))
    st.ingest_measurement(mk(20.0, 5))
    st.ingest_measurement(mk(15.0, 3))
    cells = render.latest_cells(st.measurements, "bw")
    assert cells[("gpu:ce", "gpu:d3", "d2d-push", 1 << 20)].value == 20.0


def test_errors_view(topo, state):
    out = plain(render.render_errors(topo, state, 80))
    assert "aer.correctable" in out and "+3" in out and "link_change" in out
    assert "speed 16.0 -> 8.0 GT/s" in out
    fx.check_golden("errors_80", out)


def test_errors_view_quiet(topo):
    st = make_state(errors=False)
    out = plain(render.render_errors(topo, st, 80))
    assert "0 non-zero" in out and "no new errors" in out


def test_header_and_keys():
    h = render.HeaderInfo("live", "par1-cs25", "live", fx.T0, True, "20261107-1333")
    s = plain(render.render_header(h, 80))
    assert "LIVE" in s and "● REC 20261107-1333" in s
    h = render.HeaderInfo(
        "replay", "par1-cs25", "live", fx.T0, False, "sess", True, 2.0, 5.0, 30.0
    )
    s = plain(render.render_header(h, 80))
    assert "REPLAY" in s and "5.0/30.0s x2" in s and "paused" in s
    assert "seek" in plain(render.render_keys(120, "live", "replay"), 120)
    assert "e export" in plain(render.render_keys(80, "live", "live"))


def test_measure_view_lists_lease_requirement(topo):
    suites = [
        type("S", (), {"name": n, "description": d, "est_seconds": s})
        for n, d, s in (("quick", "lat+bw pairs", 30), ("full", "everything", 600))
    ]
    m = render.MeasureModel(
        suites=suites,
        gpus=["gpu:ce", "gpu:d3"],
        owner="me",
        lease_reason="no lease for slotA",
        lease_needed=["me.slotA"],
    )
    out = plain(render.render_measure(topo, m, 80, 21))
    assert "▶ quick" in out and "needs: me.slotA" in out and "H1:ce H2:d3" in out
    fx.check_golden("measure_refused_80", out)
    m.lease_ok, m.status, m.i, m.n = True, "running", 3, 12
    m.log = ["13:33:20 [3/12] start lat 1>2"]
    out = plain(render.render_measure(topo, m, 80, 21, 14.0))
    assert "3/12" in out and "Lease: ok" in out and "14s" in out


def dedup(records, heartbeat_ns=30 * fx.NS):
    """What the session store keeps: a sample only when it changed (or 30 s)."""
    last: dict = {}
    out = []
    for r in records:
        if isinstance(r, Sample):
            k = r.key()
            prev = last.get(k)
            if (
                prev is not None
                and prev[1] == r.value
                and r.ts_ns - prev[0] < heartbeat_ns
            ):
                continue
            last[k] = (r.ts_ns, r.value)
        out.append(r)
    return out


def test_deduplicated_session_renders_like_the_full_one(topo):
    full = fx.synthetic_records(30, plx=True)
    sparse = dedup(full)
    assert len(sparse) < len(full) * 0.7
    a, b = ViewState(), ViewState()
    a.ingest(full)
    b.ingest(sparse)
    end = max(r.ts_ns for r in full if isinstance(r, Sample))
    a.fill_to(end)
    b.fill_to(end)
    for view in (render.render_live, render.render_errors):
        assert plain(view(topo, a, 80)) == plain(view(topo, b, 80))
    key = ("gpu:ce", "gpu.busy_pct", ())
    assert [v for _, v in b.series[key]] == [v for _, v in a.series[key]]


def test_fill_is_bounded_and_idempotent():
    st = ViewState()
    st.ingest_sample(Sample(0, "gpu:ce", "gpu.busy_pct", 5, "%"))
    st.ingest_sample(Sample(10_000 * fx.NS, "gpu:ce", "gpu.busy_pct", 7, "%"))
    dq = st.series[("gpu:ce", "gpu.busy_pct", ())]
    assert len(dq) == 241 or len(dq) == 240  # bounded by HISTORY
    n = len(dq)
    st.fill_to(10_000 * fx.NS)
    assert len(dq) == n and dq[-1][1] == 7


def test_only_fabric_errors_recolour_the_topology():
    st = ViewState()
    st.ingest_event(Event(fx.T0, "gpu:07", "throttle", "warn", "mask 0x10"))
    assert render.recent_error_entities(st) == set()
    st.ingest_event(Event(fx.T0, "gpu:0a", "link_change", "warn", "16x16 -> 8x16"))
    st.ingest_event(Event(fx.T0, "gpu:0f", "thermal", "error", "junction 96C"))
    assert render.recent_error_entities(st) == {"gpu:0a", "gpu:0f"}
