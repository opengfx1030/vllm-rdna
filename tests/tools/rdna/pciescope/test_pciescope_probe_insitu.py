# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-situ mode: link utilisation from passive counters (no probe, no lease).

``analyze`` is tested on counter samples written by hand for the real link of
GPU 0a of the captured par1-cs25 topology; the numbers are SYNTHETIC. ``observe``
runs Core's collectors on the captured sysfs, where no traffic counter exists:
the report must say so instead of inventing a utilisation.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope.model import Event, Sample, port_entity  # noqa: E402
from pciescope.probe import insitu  # noqa: E402
from pciescope.store import Session  # noqa: E402

GB = 1_000_000_000
SEC = 1_000_000_000


def counter(entity: str, metric: str, values: list[float], t0: int = 0) -> list[Sample]:
    """One sample per second of a byte counter."""
    return [Sample(t0 + i * SEC, entity, metric, v, "B") for i, v in enumerate(values)]


def gauge(entity: str, metric: str, values: list[float], unit: str) -> list[Sample]:
    return [Sample(i * SEC, entity, metric, v, unit) for i, v in enumerate(values)]


@pytest.fixture
def link(fx_topo):
    """The physical link of GPU 0a (switch B) and its upstream port entity."""
    lk = fx_topo.links[fx_topo.gpus["gpu:0a"].link]
    return lk, port_entity(lk.up)


def test_port_directions_map_rx_and_tx_of_both_link_ends(fx_topo, link) -> None:
    lk, up_port = link
    mapping = insitu.port_directions(fx_topo)
    # the upstream end of the link receives what the GPU side sends: that is "up"
    assert mapping[(up_port, "rx")] == (lk.id, "up")
    assert mapping[(up_port, "tx")] == (lk.id, "down")
    down_port = port_entity(lk.down)
    assert mapping.get((down_port, "rx")) == (lk.id, "down")
    assert mapping.get((down_port, "tx")) == (lk.id, "up")
    assert all(fx_topo.links[lid].physical for lid, _ in mapping.values())


def test_a_byte_counter_becomes_a_rate_per_link_direction(fx_topo, link) -> None:
    lk, up_port = link
    # 8 GB per second toward the GPU (tx of the switch port), 3 samples
    samples = counter(up_port, "port.tx_bytes", [0, 8 * GB, 16 * GB, 24 * GB])
    report = insitu.analyze(samples, fx_topo)
    assert report.available and report.sources == {"plx": 1}
    (u,) = report.links
    assert (u.link, u.direction, u.source, u.owner) == (lk.id, "down", "plx", "gpu:0a")
    assert u.samples == 3
    assert u.mean_gbs == pytest.approx(8.0) and u.peak_gbs == pytest.approx(8.0)
    assert u.raw_gbs == pytest.approx(31.508, abs=0.01)
    assert u.util_mean_pct == pytest.approx(100 * 8.0 / 31.508, abs=0.05)
    assert u.practical_gbs is not None and u.practical_gbs < u.raw_gbs
    assert u.util_peak_practical_pct == pytest.approx(100 * 8.0 / u.practical_gbs)
    assert report.window_s == pytest.approx(3.0)
    assert report.n_samples == 4


def test_gauges_in_gbs_are_taken_as_they_are(fx_topo, link) -> None:
    lk, up_port = link
    samples = gauge(up_port, "port.rx_gbs", [2.0, 4.0, 6.0, 20.0], "GB/s")
    (u,) = insitu.analyze(samples, fx_topo).links
    assert (u.link, u.direction) == (lk.id, "up")
    assert u.mean_gbs == pytest.approx(8.0) and u.peak_gbs == 20.0
    assert u.p50_gbs == pytest.approx(5.0)


def test_a_counter_reset_is_skipped_not_reported_as_traffic(fx_topo, link) -> None:
    _, up_port = link
    samples = counter(up_port, "port.rx_bytes", [5 * GB, 6 * GB, 1 * GB, 2 * GB])
    (u,) = insitu.analyze(samples, fx_topo).links
    assert u.samples == 2 and u.peak_gbs == pytest.approx(1.0)
    only_reset = counter(up_port, "port.rx_bytes", [9 * GB, 1 * GB])
    assert not insitu.analyze(only_reset, fx_topo).available


def test_switch_ports_win_over_the_gpu_side_in_either_order(fx_topo, link) -> None:
    lk, up_port = link
    plx = counter(up_port, "port.tx_bytes", [0, 4 * GB, 8 * GB])
    drv = gauge("gpu:0a", "gpu.pcie_rx_bytes_per_s", [9 * GB, 9 * GB], "B/s")
    for samples in (plx + drv, drv + plx):
        (u,) = insitu.analyze(samples, fx_topo).links
        assert u.source == "plx" and u.mean_gbs == pytest.approx(4.0)
    only = insitu.analyze(drv, fx_topo)
    (u,) = only.links
    assert (u.link, u.direction, u.source) == (lk.id, "down", "amdgpu")
    assert u.mean_gbs == pytest.approx(9.0) and only.sources == {"amdgpu": 1}
    tx = gauge("gpu:0a", "gpu.pcie_tx_bytes_per_s", [1 * GB], "B/s")
    assert insitu.analyze(tx, fx_topo).links[0].direction == "up"


def test_links_are_listed_busiest_first(fx_topo, link) -> None:
    _, up_port = link
    other = fx_topo.links[fx_topo.gpus["gpu:ce"].link]
    samples = counter(up_port, "port.tx_bytes", [0, 2 * GB, 4 * GB]) + counter(
        port_entity(other.up), "port.tx_bytes", [0, 12 * GB, 24 * GB]
    )
    report = insitu.analyze(samples, fx_topo)
    assert [u.owner for u in report.links] == ["gpu:ce", "gpu:0a"]
    assert report.sources == {"plx": 2}


def test_without_counters_the_report_says_why_and_invents_nothing(fx_topo) -> None:
    busy = gauge("gpu:0a", "gpu.busy_pct", [10.0, 90.0], "%")
    report = insitu.analyze(busy, fx_topo)
    assert not report.available and report.links == [] and report.sources == {}
    assert "no port.rx/tx series reported" in report.reason
    assert "amdgpu exposes no pcie_bw" in report.reason
    assert report.context["gpus"]["gpu:0a"] == {
        "gpu.busy_pct.mean": 50.0,
        "gpu.busy_pct.max": 90.0,
    }


def test_the_reason_names_a_skipped_switch_collector(fx_topo) -> None:
    skipped = [("plx", "switch writes not allowed (no --allow-switch-writes)")]
    report = insitu.analyze([], fx_topo, skipped=skipped)
    assert not report.available
    assert "collector plx unavailable (switch writes not allowed" in report.reason
    assert report.skipped == skipped


def test_the_reason_names_disabled_switch_counters(fx_topo) -> None:
    off = gauge("switch:B", "plx.perf_enabled", [0.0, 0.0], "")
    report = insitu.analyze(off, fx_topo)
    assert "plx.perf_enabled is 0" in report.reason
    assert "needs approval" in report.reason


def test_samples_of_unknown_ports_and_gpus_are_ignored(fx_topo) -> None:
    samples = counter("port:ff:00.0", "port.rx_bytes", [0, GB, 2 * GB]) + counter(
        "gpu:zz", "gpu.pcie_rx_bytes_per_s", [GB]
    )
    assert not insitu.analyze(samples, fx_topo).available


def test_report_is_strict_json_and_renders(fx_topo, link) -> None:
    _, up_port = link
    samples = counter(up_port, "port.tx_bytes", [0, 8 * GB, 16 * GB]) + gauge(
        "gpu:0a", "gpu.busy_pct", [50.0, 60.0, 70.0], "%"
    )
    report = insitu.analyze(samples, fx_topo, window_s=10.0)
    doc = report.to_dict()
    json.dumps(doc, allow_nan=False)
    assert doc["available"] and doc["window_s"] == 10.0 and doc["links"][0]["link"]
    text = insitu.render_text(report)
    assert "link utilisation (GB/s" in text and "gpu:0a" in text and "plx" in text
    assert "GPU load in the window" in text
    none = insitu.render_text(insitu.analyze([], fx_topo, skipped=[("plx", "off")]))
    assert "UNAVAILABLE" in none and "collector plx skipped: off" in none


# -- observe on the captured sysfs ---------------------------------------------------


def test_observe_on_the_captured_box_reports_unavailable_utilisation(
    fx_topo, fx_sysfs
) -> None:
    ticks: list[int] = []
    report = insitu.observe(
        fx_topo,
        duration_s=0.3,
        interval_s=0.1,
        selected=("sysfs_link", "amdgpu"),
        sysfs=fx_sysfs,
        on_tick=lambda n, elapsed: ticks.append(n),
    )
    assert report.n_samples > 0 and ticks
    assert not report.available
    assert "no port.rx/tx series reported" in report.reason
    assert report.window_s > 0
    assert set(report.context["gpus"]) <= set(fx_topo.gpus)


def test_observe_records_samples_and_the_end_event_in_the_session(
    fx_topo, fx_sysfs, tmp_path: Path
) -> None:
    with Session.create(tmp_path, "insitu", topo=fx_topo) as session:
        report = insitu.observe(
            fx_topo,
            duration_s=0.25,
            interval_s=0.1,
            selected=("sysfs_link",),
            sysfs=fx_sysfs,
            session=session,
        )
    reopened = Session.open(session.path)
    assert any(True for _ in reopened.samples())
    (end,) = [e for e in reopened.events() if e.name == "insitu_end"]
    assert isinstance(end, Event) and end.severity == "warn"
    assert end.data == {"available": False, "links": 0}
    assert report.reason in end.message


def test_observe_stops_when_cancelled(fx_topo, fx_sysfs) -> None:
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    report = insitu.observe(
        fx_topo,
        duration_s=30.0,
        interval_s=0.05,
        selected=("sysfs_link",),
        sysfs=fx_sysfs,
        cancel=cancel,
    )
    assert report.window_s < 5.0
