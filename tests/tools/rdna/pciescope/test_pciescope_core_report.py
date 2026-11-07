# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report and Perfetto export of a synthetic session on the real topology."""

from __future__ import annotations

import json

import pytest
from pciescope import model, report, store, trace

SEC = 1_000_000_000
T0 = 1_800_000_000 * SEC
LINK = "link:c8:10.0-cc:00.0"


def S(t, entity, metric, value, unit="x", **tags):
    return model.Sample(T0 + int(t * SEC), entity, metric, float(value), unit, tags)


def _measurement(i, kind, src, dst, value, unit, direction, topo, **kw):
    info = topo.path(src, dst)
    return model.Measurement(
        kind=kind,
        src=src,
        dst=dst,
        direction=direction,
        size_bytes=kw.pop("size_bytes", 1 << 28),
        value=value,
        unit=unit,
        stats={"p50": value, "p99": value * 1.05, "n": 10.0},
        path_class=info.path_class,
        theoretical=info.bottleneck_gbs if unit == "GB/s" else None,
        efficiency=value / info.bottleneck_gbs if unit == "GB/s" else None,
        started_ns=T0 + (20 + i) * SEC,
        ended_ns=T0 + (20 + i) * SEC + SEC // 2,
        gpus_used=[x for x in (src, dst) if x.startswith("gpu:")],
        **kw,
    )


@pytest.fixture()
def session(tmp_path, fx_topo):
    s = store.Session.create(
        tmp_path,
        "synthetic",
        topo=fx_topo,
        collectors=["sysfs_link", "aer", "amdgpu", "kfd"],
        interval_s=1.0,
    )
    recs: list = []
    for t in range(10):
        recs.append(S(t, LINK, "link.speed_gts", 16 if t < 5 else 8, "GT/s"))
        recs.append(S(t, LINK, "link.width", 16, "lanes"))
        recs.append(
            S(t, LINK, "link.capacity_gbs", 31.508 if t < 5 else 15.754, "GB/s")
        )
        recs.append(S(t, LINK, "link.degraded", 0 if t < 5 else 1, "bool"))
        recs.append(S(t, "gpu:ce", "gpu.temp_junction_c", 70 + 3 * t, "C"))
        recs.append(S(t, "gpu:ce", "gpu.busy_pct", 50, "%"))
        recs.append(S(t, "gpu:d3", "gpu.busy_pct", 0, "%"))
        recs.append(S(t, "port:c8:10.0", "aer.correctable", 0 if t < 6 else 2, "count"))
        recs.append(
            S(
                t,
                "gpu:0f",
                "kfd.evicted_ms",
                0 if t < 7 else 300,
                "ms",
                pid="42",
                comm="w",
            )
        )
        recs.append(
            S(t, "host", "collector.cost_ms", 1.0 + 0.1 * t, "ms", collector="aer")
        )
        recs.append(S(t, "host", "collector.samples", 66, "count", collector="aer"))
    recs.append(
        S(
            6,
            "port:c8:10.0",
            "aer.error",
            2,
            "count",
            **{"class": "correctable", "type": "BadTLP"},
        )
    )
    recs.append(S(0, "port:00:01.1", "aer.rootport_correctable", 10, "count"))
    recs.append(model.Event(T0 + 5 * SEC, LINK, "link_change", "warn", "speed 16 -> 8"))
    recs.append(model.Event(T0 + 9 * SEC, "gpu:ce", "thermal", "warn", "junction 97 C"))
    recs.append(model.Event(T0 + 1 * SEC, "host", "watch_start", "info"))
    recs.append(
        _measurement(0, "bw", "gpu:ce", "gpu:d3", 24.5, "GB/s", "d2d-push", fx_topo)
    )
    recs.append(
        _measurement(1, "bw", "gpu:ce", "gpu:d3", 25.5, "GB/s", "d2d-push", fx_topo)
    )
    recs.append(
        _measurement(2, "bw", "gpu:ce", "gpu:0a", 20.0, "GB/s", "d2d-push", fx_topo)
    )
    recs.append(
        _measurement(
            3, "lat", "host", "gpu:ce", 2.1, "us", "h2d", fx_topo, size_bytes=4
        )
    )
    recs.append(
        _measurement(
            4,
            "bw",
            "gpu:ce",
            "gpu:cb",
            0.0,
            "GB/s",
            "d2d-push",
            fx_topo,
            ok=False,
            error="timeout",
        )
    )
    s.append_many(recs)
    s.close()
    return store.Session.open(s.path)


def test_tw_stats_forward_fills():
    pts = [(0, 10.0), (10 * SEC, 20.0)]
    st = report.tw_stats(pts, 30 * SEC)
    assert st["min"] == 10.0
    assert st["max"] == 20.0
    assert st["first"] == 10.0
    assert st["last"] == 20.0
    # 10 s at 10 and 20 s at 20 -> time-weighted mean 16.67, not the plain mean 15
    assert st["mean"] == pytest.approx(50 / 3, abs=1e-6)
    assert report.tw_stats([], 1) == {}


def test_report_sections(session):
    rep = report.build_report(session)
    s = rep["session"]
    assert s["host"] == "par1-cs25"
    assert s["duration_s"] == pytest.approx(24.5, abs=0.01)  # last probe ends at 24.5
    assert s["counts"]["measurements"] == 5
    assert rep["ceilings"]["gen4_x16_raw_gbs"] == pytest.approx(31.508, abs=1e-3)
    assert rep["ceilings"]["gen4_x16_practical_gbs"] == pytest.approx(28.03, abs=0.01)
    cost = rep["collector_cost"]["aer"]
    assert cost["n"] == 10
    assert cost["max_ms"] == pytest.approx(1.9)
    assert cost["samples_per_tick"] == 66.0


def test_report_links_and_gpus(session):
    rep = report.build_report(session)
    links = {lk["id"]: lk for lk in rep["links"]}
    assert all(lk["kind"] != "fabric" for lk in links.values())  # virtual hidden
    lk = links[LINK]
    assert lk["speed_gts"] == {"min": 8.0, "max": 16.0, "last": 8.0}
    assert lk["degraded_s"] == pytest.approx(19.5, abs=0.1)  # t=5 s .. end at 24.5 s
    assert lk["link_changes"] == 1
    assert lk["practical_gbs_at_max"] == pytest.approx(28.03, abs=0.01)
    assert lk["capacity_gbs_last"] == pytest.approx(15.754)
    with_virtual = report.build_report(session, include_virtual=True)
    assert len(with_virtual["links"]) > len(rep["links"])
    gpus = {g["id"]: g for g in rep["gpus"]}
    assert gpus["gpu:ce"]["temp_junction_c"]["max"] == 97.0
    assert gpus["gpu:ce"]["busy_pct"]["mean"] == 50.0
    assert gpus["gpu:0f"]["kfd_evicted"][0]["delta_ms"] == 300.0
    assert gpus["gpu:0f"]["kfd_evicted"][0]["pid"] == "42"


def test_report_errors_and_events(session):
    rep = report.build_report(session)
    inc = rep["errors"]["increased_during_session"]
    assert [(e["entity"], e["class"], e["delta"]) for e in inc] == [
        ("port:c8:10.0", "correctable", 2.0)
    ]
    totals = {
        (e["entity"], e["class"]): e for e in rep["errors"]["nonzero_totals_since_boot"]
    }
    assert totals[("port:00:01.1", "rootport_correctable")]["total"] == 10.0
    assert totals[("port:c8:10.0", "correctable")]["types"] == {"BadTLP": 2.0}
    ev = rep["events"]
    assert ev["by_name"]["link_change"] == 1
    assert [e["name"] for e in ev["notable"]] == ["link_change", "thermal"]


def test_report_findings(session):
    rep = report.build_report(session)
    codes = {(f["code"], f["entity"]) for f in rep["findings"]}
    assert ("link_degraded", LINK) in codes
    assert ("link_retrain", LINK) in codes
    assert ("aer_increase", "port:c8:10.0") in codes
    assert ("thermal", "gpu:ce") in codes
    assert ("evicted_ms", "gpu:0f") in codes
    assert ("probe_failed", "gpu:ce") in codes
    sev = [f["severity"] for f in rep["findings"]]
    assert sev == sorted(sev, key={"error": 0, "warn": 1, "info": 2}.get)
    evicted = next(f for f in rep["findings"] if f["code"] == "evicted_ms")
    assert "INVALID" in evicted["message"]


def test_report_probe_analysis(session):
    pr = report.build_report(session)["probes"]
    assert pr["count"] == 5
    assert {(g["method"], g["pattern"]) for g in pr["summary"]} == {("", "")}
    groups = {(g["kind"], g["direction"], g["path_class"]): g for g in pr["summary"]}
    g = groups[("bw", "d2d-push", "intra-switch")]
    assert (g["n"], g["min"], g["max"]) == (2, 24.5, 25.5)
    assert g["median"] == pytest.approx(25.0)  # the failed run is excluded
    assert groups[("bw", "d2d-push", "cross-root")]["median"] == 20.0
    assert groups[("lat", "h2d", "host")]["unit"] == "us"
    first = pr["measurements"][0]
    assert first["practical_gbs"] == pytest.approx(28.03, abs=0.01)
    assert first["efficiency_practical"] == pytest.approx(24.5 / 28.03, abs=1e-3)
    assert first["efficiency_raw"] == pytest.approx(24.5 / 31.508, abs=1e-3)
    mx = pr["matrix_gbs"]
    assert mx["rows"] == ["gpu:ce"]
    assert mx["cols"] == ["gpu:0a", "gpu:d3"]
    assert mx["values"] == [[20.0, 25.0]]
    util = {(u["link"], u["direction"]): u for u in pr["link_utilization"]}
    peak = util[("link:c8:10.0-cc:00.0", "up")]
    assert peak["peak_gbs"] == 25.5  # best single flow through the card link
    assert peak["util_raw_pct"] == pytest.approx(100 * 25.5 / 31.508, abs=0.1)
    assert peak["util_practical_pct"] == pytest.approx(100 * 25.5 / 28.03, abs=0.1)
    # virtual fabric links carry no lanes and are never charged
    assert not any(k[0].startswith("link:c6:") for k in util)


def test_report_probe_summary_keeps_sizes_methods_and_patterns_apart(tmp_path, fx_topo):
    s = store.Session.create(
        tmp_path, "probes", topo=fx_topo, collectors=[], interval_s=1.0
    )
    recs: list[model.Measurement] = []

    def add(kind, src, dst, value, unit, direction, **kw):
        recs.append(
            _measurement(
                len(recs), kind, src, dst, value, unit, direction, fx_topo, **kw
            )
        )

    push = {"method": "kernel-store"}
    for size, val in ((1 << 12, 0.5), (1 << 20, 9.0), (1 << 26, 24.0), (1 << 26, 26.0)):
        add(
            "bw",
            "gpu:ce",
            "gpu:d3",
            val,
            "GB/s",
            "d2d-push",
            size_bytes=size,
            tags=push,
        )
    add("bw", "gpu:ce", "gpu:d3", 12.0, "GB/s", "d2d-sdma", tags={"method": "sdma"})
    add("lat", "gpu:ce", "gpu:d3", 2.0, "us", "d2d-push", size_bytes=4, tags=push)
    add(
        "lat", "gpu:ce", "gpu:d3", 90.0, "us", "d2d-push", size_bytes=1 << 26, tags=push
    )
    add("bw", "host", "gpu:ce", 20.0, "GB/s", "h2d", tags={"method": "pinned"})
    add("bw", "host", "gpu:ce", 6.0, "GB/s", "h2d", tags={"method": "pageable"})
    # a pattern is not a single flow, whatever its kind says: never in the matrix
    # and never charged to the links of its first src -> dst pair
    ring = {"pattern": "ring-intra", "flows": "4"}
    add("bw", "gpu:ce", "gpu:0a", 99.0, "GB/s", "d2d-push", tags=ring)
    recs.append(
        model.Measurement(
            kind="concurrency",
            src="gpu:ce+gpu:d3",
            dst="gpu:0a+gpu:0f",
            direction="d2d-push",
            size_bytes=1 << 26,
            value=55.0,
            unit="GB/s",
            path_class="cross-root",
            started_ns=T0 + 40 * SEC,
            ended_ns=T0 + 41 * SEC,
            tags={"pattern": "fan-in", "flows": "2"},
        )
    )
    s.append_many(recs)
    s.close()
    rep = report.build_report(store.Session.open(s.path))
    pr = rep["probes"]
    rows = {
        (g["kind"], g["direction"], g["method"], g["pattern"]): g for g in pr["summary"]
    }
    top = rows[("bw", "d2d-push", "kernel-store", "")]
    assert top["size_bytes"] == 1 << 26
    assert top["sizes"] == [1 << 12, 1 << 20, 1 << 26]
    assert (top["n"], top["min"], top["median"], top["max"]) == (2, 24.0, 25.0, 26.0)
    assert rows[("bw", "d2d-sdma", "sdma", "")]["median"] == 12.0
    lat = rows[("lat", "d2d-push", "kernel-store", "")]
    assert (lat["size_bytes"], lat["median"]) == (4, 2.0)  # base latency, not 64 MiB
    assert rows[("concurrency", "d2d-push", "", "fan-in")]["median"] == 55.0
    assert rows[("bw", "d2d-push", "", "ring-intra")]["median"] == 99.0
    mx = pr["matrix_gbs"]
    assert mx["rows"] == ["gpu:ce", "host"]
    assert mx["cols"] == ["gpu:ce", "gpu:d3"]
    assert mx["values"] == [[None, 25.0], [20.0, None]]  # push only, pinned only
    assert mx["size_bytes"][0] == [None, 1 << 26]
    util = {(u["link"], u["direction"]): u for u in pr["link_utilization"]}
    assert util[("link:c8:10.0-cc:00.0", "up")]["peak_gbs"] == 26.0  # not 99
    text = report.render_text(rep)
    for needle in ("kernel-store", "fan-in", "64 MiB", "smallest for latency"):
        assert needle in text, needle


def test_report_text_rendering(session):
    text = report.render_text(report.build_report(session))
    for needle in (
        "pciescope report:",
        "FINDINGS",
        "[warn] link_degraded",
        "LINKS (physical)",
        "GPUS",
        "ERRORS (AER)",
        "COLLECTOR COST",
        "PROBES (5 measurements)",
        "bandwidth matrix GB/s",
        "per-link utilisation attributed from probes",
    ):
        assert needle in text, needle
    assert max(len(line) for line in text.splitlines()) < 220


def test_report_without_probes_or_topology(tmp_path):
    s = store.Session.create(tmp_path, "bare")
    s.append(S(0, "gpu:ce", "gpu.busy_pct", 1))
    s.close()
    rep = report.build_report(store.Session.open(s.path))
    assert rep["probes"] == {"count": 0}
    assert "links" not in rep  # no topology.json in this session
    assert rep["findings"] == []
    assert "none (no degraded link" in report.render_text(rep)


def test_chrome_export(session, tmp_path):
    out = trace.export_chrome(session, tmp_path / "t.json")
    doc = json.loads(out.read_text())
    assert doc["metadata"]["host"] == "par1-cs25"
    ev = doc["traceEvents"]
    phases = {e["ph"] for e in ev}
    assert {"M", "C", "X", "i"} <= phases
    counters = [e for e in ev if e["ph"] == "C" and e["name"] == "gpu.busy_pct"]
    assert counters
    assert counters[0]["pid"] == trace.PID_GPU
    assert counters[0]["args"] == {"gpu:ce": 50.0, "gpu:d3": 0.0}
    tagged = [e for e in ev if e["ph"] == "C" and e["name"] == "kfd.evicted_ms"]
    assert "gpu:0f{comm=w,pid=42}" in tagged[0]["args"]
    slices = [e for e in ev if e["ph"] == "X"]
    assert len(slices) == 5
    assert slices[0]["pid"] == trace.PID_PROBE
    assert slices[0]["dur"] == pytest.approx(500_000.0)  # 0.5 s in microseconds
    assert slices[0]["args"]["path_class"] == "intra-switch"
    instants = [e for e in ev if e["ph"] == "i"]
    assert {e["cat"] for e in instants} == {"warn", "info"}
    # timestamps are relative to the first record
    assert min(e["ts"] for e in ev if "ts" in e) == 0.0
    assert doc["metadata"]["t0_epoch_ns"] == T0


def test_chrome_export_default_path(session):
    out = trace.export_chrome(session.path)
    assert out == session.path / "trace.perfetto.json"
    assert json.loads(out.read_text())["traceEvents"]
