# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling loop: event derivation, dedup, collector failures, watch thread."""

from __future__ import annotations

import threading
import time

from pciescope import model, runner, store


def S(ts, entity, metric, value, **tags):
    return model.Sample(ts, entity, metric, float(value), "x", dict(tags))


def names(events):
    return [(e.name, e.severity) for e in events]


# -- EventDetector ---------------------------------------------------------


def test_link_downgrade_and_recovery():
    d = runner.EventDetector()
    lk = "link:c8:10.0-cc:00.0"
    assert d.feed([S(1, lk, "link.speed_gts", 16), S(1, lk, "link.width", 16)]) == []
    ev = d.feed([S(2, lk, "link.speed_gts", 8), S(2, lk, "link.width", 16)])
    assert names(ev) == [("link_change", "warn")]
    assert ev[0].data == {"metric": "link.speed_gts", "old": 16.0, "new": 8.0}
    ev = d.feed([S(3, lk, "link.speed_gts", 16)])
    assert names(ev) == [("link_change", "info")]
    ev = d.feed([S(4, lk, "link.degraded", 0)])
    assert ev == []
    assert names(d.feed([S(5, lk, "link.degraded", 1)])) == [("link_degraded", "warn")]
    assert names(d.feed([S(6, lk, "link.degraded", 0)])) == [("link_restored", "info")]


def test_aer_increase_carries_error_types():
    d = runner.EventDetector()
    e = "port:c8:10.0"
    assert d.feed([S(1, e, "aer.correctable", 0)]) == []
    ev = d.feed(
        [
            S(2, e, "aer.correctable", 3),
            S(2, e, "aer.error", 2, **{"class": "correctable", "type": "BadTLP"}),
            S(2, e, "aer.error", 1, **{"class": "correctable", "type": "RxErr"}),
        ]
    )
    assert names(ev) == [("aer_delta", "warn")]
    assert ev[0].data["delta"] == 3.0
    assert ev[0].data["types"] == {"BadTLP": 2.0, "RxErr": 1.0}
    assert d.feed([S(3, e, "aer.correctable", 3)]) == []  # unchanged: no event
    ev = d.feed([S(4, "port:c0:03.1", "aer.rootport_fatal", 0)])
    assert ev == []
    ev = d.feed([S(5, "port:c0:03.1", "aer.rootport_fatal", 1)])
    assert names(ev) == [("aer_delta", "error")]
    # a counter that goes down (driver reload) is not an error burst
    assert d.feed([S(6, e, "aer.correctable", 0)]) == []


def test_thermal_thresholds_with_hysteresis():
    d = runner.EventDetector()
    g = "gpu:ce"
    t = iter(range(1, 100))

    def feed(v):
        return d.feed([S(next(t), g, "gpu.temp_junction_c", v)])

    assert feed(60) == []
    assert names(feed(91)) == [("thermal", "warn")]
    assert feed(92) == []  # still hot: one event per crossing
    assert names(feed(96)) == [("thermal", "error")]
    assert feed(89) == []  # inside the hysteresis band: no re-arm
    assert feed(96) == []
    assert feed(70) == []
    assert names(feed(93)) == [("thermal", "warn")]


def test_eviction_jump_and_throttle():
    d = runner.EventDetector()
    g = "gpu:0f"
    assert d.feed([S(1, g, "kfd.evicted_ms", 0, pid="42", comm="w")]) == []
    ev = d.feed([S(2, g, "kfd.evicted_ms", 250, pid="42", comm="w")])
    assert names(ev) == [("evicted_jump", "warn")]
    assert ev[0].data["delta_ms"] == 250.0
    assert ev[0].data["pid"] == "42"
    # another pid is its own series: first sight is not a jump
    assert d.feed([S(3, g, "kfd.evicted_ms", 9, pid="43", comm="w")]) == []
    assert d.feed([S(4, g, "gpu.throttle_status", 0)]) == []
    assert names(d.feed([S(5, g, "gpu.throttle_status", 0x20)])) == [
        ("throttle", "warn")
    ]


# -- Dedup -----------------------------------------------------------------


def test_dedup_keeps_changes_and_heartbeats():
    dd = runner.Dedup(heartbeat_s=10.0)
    e = "gpu:ce"
    sec = 1_000_000_000
    out = dd.filter([S(0, e, "m", 1), S(0, e, "n", 5)])
    assert len(out) == 2  # first sight
    assert dd.filter([S(1 * sec, e, "m", 1), S(1 * sec, e, "n", 6)]) == [
        S(1 * sec, e, "n", 6)
    ]
    assert dd.filter([S(9 * sec, e, "m", 1)]) == []
    assert dd.filter([S(10 * sec, e, "m", 1)]) == [S(10 * sec, e, "m", 1)]
    # tags are part of the series identity
    assert len(dd.filter([S(11 * sec, e, "m", 1, pid="1")])) == 1


# -- Runner ----------------------------------------------------------------


class Fake:
    def __init__(self, name, frames):
        self.name = name
        self.frames = list(frames)
        self.calls = 0

    def available(self):
        return True, ""

    def collect(self):
        i = min(self.calls, len(self.frames) - 1)
        self.calls += 1
        f = self.frames[i]
        if isinstance(f, Exception):
            raise f
        return f(model.now_ns()) if callable(f) else f


def test_tick_records_cost_events_and_dedups(tmp_path, fx_topo):
    ok = Fake("ok", [lambda ts: [S(ts, "gpu:ce", "gpu.busy_pct", 5)]])
    bad = Fake("bad", [RuntimeError("boom")])
    sess = store.Session.create(tmp_path, "tick", topo=fx_topo)
    r = runner.Runner(fx_topo, [ok, bad], session=sess, interval_s=0.01)
    t1 = r.tick()
    t2 = r.tick()
    assert set(t1.costs_ms) == {"ok", "bad"}
    assert names(t1.events) == [("collector_error", "error")]
    assert "RuntimeError: boom" in t1.events[0].message
    assert t2.events == []  # the same failure is reported once
    metrics = {s.metric for s in t1.samples}
    assert {"gpu.busy_pct", "collector.cost_ms", "collector.samples"} <= metrics
    sess.close()
    written = list(store.Session.open(sess.path).samples())
    busy = [s for s in written if s.metric == "gpu.busy_pct"]
    assert len(busy) == 1  # second tick unchanged -> not written again
    assert r.summary.ticks == 2
    assert r.summary.samples_written < r.summary.samples_seen
    assert r.summary.event_names == {"collector_error": 1}


def test_run_honours_limits_and_writes_markers(tmp_path, fx_topo):
    n = iter(range(1000))
    c = Fake("c", [lambda ts: [S(ts, "gpu:ce", "gpu.busy_pct", next(n))]])
    sess = store.Session.create(tmp_path, "run", topo=fx_topo)
    seen: list[runner.Tick] = []
    r = runner.Runner(fx_topo, [c], session=sess, interval_s=0.01, dedup=False)
    summary = r.run(max_ticks=5, on_tick=seen.append)
    sess.close()
    assert summary.ticks == len(seen) == 5
    ev = store.Session.open(sess.path).events()
    assert [e.name for e in ev] == ["watch_start", "watch_stop"]
    assert summary.cost_ms["c"]["n"] == 5
    assert summary.samples_per_tick["c"] == 1.0
    d = summary.to_dict()
    assert d["ticks"] == 5
    assert "collector_cost_ms" in d


def test_run_by_duration_and_stop_event(fx_topo):
    c = Fake("c", [[S(1, "gpu:ce", "gpu.busy_pct", 1)]])
    r = runner.Runner(fx_topo, [c], interval_s=0.02)
    t0 = time.monotonic()
    r.run(duration_s=0.1)
    assert 0.08 <= time.monotonic() - t0 < 1.0
    assert 4 <= r.summary.ticks <= 6
    stop = threading.Event()
    r2 = runner.Runner(fx_topo, [c], interval_s=0.01)
    threading.Timer(0.05, stop.set).start()
    r2.run(stop=stop)
    assert r2.summary.ticks >= 1


def test_start_watch_and_stop(tmp_path, fx_topo):
    c = Fake("c", [lambda ts: [S(ts, "gpu:ce", "gpu.busy_pct", 1)]])
    sess = store.Session.create(tmp_path, "bg", topo=fx_topo)
    h = runner.start_watch(fx_topo, [c], sess, interval_s=0.01)
    deadline = time.monotonic() + 5
    while h.status()["ticks"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert h.running
    out = h.stop()
    assert not h.running
    assert out["ticks"] >= 3
    assert out["session"] == sess.name
    assert store.Session.open(sess.path).meta["status"] == "ok"


def test_start_watch_with_duration_closes_itself(tmp_path, fx_topo):
    c = Fake("c", [[S(1, "gpu:ce", "gpu.busy_pct", 1)]])
    sess = store.Session.create(tmp_path, "dur", topo=fx_topo)
    h = runner.start_watch(fx_topo, [c], sess, interval_s=0.01, duration_s=0.05)
    h.thread.join(5)
    assert not h.running
    assert store.Session.open(sess.path).meta["status"] == "ok"


def test_snapshot_does_not_dedup_or_write(fx_topo):
    c = Fake("c", [[S(1, "gpu:ce", "gpu.busy_pct", 1)]])
    tk = runner.snapshot(fx_topo, [c])
    assert any(s.metric == "gpu.busy_pct" for s in tk.samples)
    assert tk.costs_ms["c"] >= 0
