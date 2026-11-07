# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sampling loop shared by ``snap``, ``watch``, the MCP server and the TUI.

``Runner`` calls every collector once per tick (timing each call), derives
``Event`` records from consecutive samples (link retrain/downgrade, AER counter
increase, junction temperature thresholds, KFD eviction jumps, collector
failures), drops samples that did not change (``Dedup``: a series is written when
its value changes or every ``heartbeat_s`` seconds, readers forward-fill) and
appends the rest to a ``Session``.  ``start_watch`` runs a ``Runner`` on a thread
and returns a ``WatchHandle`` (used by the MCP ``start_watch`` / ``stop_watch``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pciescope.model import (
    HOST,
    Collector,
    Event,
    Sample,
    Topology,
    now_ns,
)
from pciescope.store import Session

THERMAL_WARN_C = 90.0
THERMAL_ERROR_C = 95.0
_THERMAL_RESET_C = 88.0


class EventDetector:
    """Derive ``Event`` records from consecutive samples of the same series."""

    def __init__(
        self,
        *,
        thermal_warn_c: float = THERMAL_WARN_C,
        thermal_error_c: float = THERMAL_ERROR_C,
    ) -> None:
        self.thermal_warn_c = thermal_warn_c
        self.thermal_error_c = thermal_error_c
        self._last: dict[tuple, float] = {}
        self._hot: dict[str, int] = {}  # entity -> 0 ok, 1 warn, 2 error

    def feed(self, samples: Iterable[Sample]) -> list[Event]:
        batch = list(samples)
        types: dict[tuple[str, str], dict[str, float]] = {}
        for s in batch:
            if s.metric == "aer.error":
                cls = s.tags.get("class", "")
                types.setdefault((s.entity, cls), {})[s.tags.get("type", "?")] = s.value
        out: list[Event] = []
        for s in batch:
            key = s.key()
            old = self._last.get(key)
            self._last[key] = s.value
            m = s.metric
            if m == "gpu.temp_junction_c":
                out.extend(self._thermal(s))
            if old is None or old == s.value:
                continue
            if m in ("link.speed_gts", "link.width"):
                out.append(
                    Event(
                        s.ts_ns,
                        s.entity,
                        "link_change",
                        "warn" if s.value < old else "info",
                        f"{m} {old:g} -> {s.value:g}",
                        {"metric": m, "old": old, "new": s.value},
                    )
                )
            elif m == "link.degraded":
                name = "link_degraded" if s.value else "link_restored"
                out.append(
                    Event(s.ts_ns, s.entity, name, "warn" if s.value else "info")
                )
            elif m.startswith("aer.") and m != "aer.error" and s.value > old:
                cls = m.split(".", 1)[1]
                fatal = cls in (
                    "nonfatal",
                    "fatal",
                    "rootport_nonfatal",
                    "rootport_fatal",
                )
                data: dict[str, Any] = {
                    "class": cls,
                    "old": old,
                    "new": s.value,
                    "delta": s.value - old,
                }
                kinds = types.get((s.entity, cls))
                if kinds:
                    data["types"] = dict(kinds)
                out.append(
                    Event(
                        s.ts_ns,
                        s.entity,
                        "aer_delta",
                        "error" if fatal else "warn",
                        f"{m} +{s.value - old:g} (now {s.value:g})",
                        data,
                    )
                )
            elif m == "kfd.evicted_ms" and s.value > old:
                out.append(
                    Event(
                        s.ts_ns,
                        s.entity,
                        "evicted_jump",
                        "warn",
                        f"evicted_ms +{s.value - old:g} (pid {s.tags.get('pid', '?')})",
                        {"delta_ms": s.value - old, **s.tags},
                    )
                )
            elif m == "gpu.throttle_status" and old == 0 and s.value:
                out.append(
                    Event(
                        s.ts_ns, s.entity, "throttle", "warn", f"mask {int(s.value):#x}"
                    )
                )
        return out

    def _thermal(self, s: Sample) -> list[Event]:
        level = (
            2
            if s.value >= self.thermal_error_c
            else 1
            if s.value >= self.thermal_warn_c
            else 0
        )
        prev = self._hot.get(s.entity, 0)
        if level == 0 and s.value > _THERMAL_RESET_C:
            level = prev
        self._hot[s.entity] = level
        if level > prev:
            return [
                Event(
                    s.ts_ns,
                    s.entity,
                    "thermal",
                    "error" if level == 2 else "warn",
                    f"junction {s.value:g} C",
                    {"junction_c": s.value},
                )
            ]
        return []


class Dedup:
    """Keep a sample only if its value changed or the heartbeat elapsed."""

    def __init__(self, heartbeat_s: float = 30.0) -> None:
        self.heartbeat_ns = int(heartbeat_s * 1e9)
        self._last: dict[tuple, tuple[float, int]] = {}

    def filter(self, samples: Iterable[Sample]) -> list[Sample]:
        out: list[Sample] = []
        for s in samples:
            key = s.key()
            prev = self._last.get(key)
            if (
                prev is None
                or prev[0] != s.value
                or s.ts_ns - prev[1] >= self.heartbeat_ns
            ):
                self._last[key] = (s.value, s.ts_ns)
                out.append(s)
        return out


@dataclass(slots=True)
class Tick:
    """One sampling pass: samples (all), derived events and per-collector cost."""

    ts_ns: int
    samples: list[Sample]
    events: list[Event]
    costs_ms: dict[str, float]


def _stats(vals: list[float]) -> dict[str, float]:
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": sum(v) / n,
        "p50": v[n // 2],
        "p95": v[min(n - 1, int(n * 0.95))],
        "max": v[-1],
    }


@dataclass(slots=True)
class RunSummary:
    """What a run did (ticks, records written, collector cost statistics)."""

    ticks: int = 0
    samples_seen: int = 0
    samples_written: int = 0
    events: int = 0
    duration_s: float = 0.0
    overruns: int = 0
    cost_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    samples_per_tick: dict[str, float] = field(default_factory=dict)
    event_names: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "samples_seen": self.samples_seen,
            "samples_written": self.samples_written,
            "events": self.events,
            "duration_s": round(self.duration_s, 3),
            "overruns": self.overruns,
            "collector_cost_ms": self.cost_ms,
            "samples_per_tick": self.samples_per_tick,
            "event_names": self.event_names,
        }


class Runner:
    """Sample a set of collectors, derive events, write to a session."""

    def __init__(
        self,
        topo: Topology,
        collectors: list[Collector],
        *,
        session: Session | None = None,
        interval_s: float = 1.0,
        dedup: bool = True,
        heartbeat_s: float = 30.0,
        detector: EventDetector | None = None,
    ) -> None:
        self.topo = topo
        self.collectors = collectors
        self.session = session
        self.interval_s = interval_s
        self.detector = detector or EventDetector()
        self.dedup = Dedup(heartbeat_s) if dedup else None
        self.summary = RunSummary()
        self._costs: dict[str, list[float]] = {c.name: [] for c in collectors}
        self._counts: dict[str, list[int]] = {c.name: [] for c in collectors}
        self._errors: dict[str, str] = {}

    def tick(self) -> Tick:
        """Run every collector once; write to the session if there is one."""
        ts = now_ns()
        samples: list[Sample] = []
        events: list[Event] = []
        costs: dict[str, float] = {}
        for c in self.collectors:
            t0 = time.perf_counter()
            try:
                got = c.collect()
            except Exception as exc:  # a failing source must not stop the others
                got = []
                msg = f"{type(exc).__name__}: {exc}"
                if self._errors.get(c.name) != msg:
                    self._errors[c.name] = msg
                    events.append(
                        Event(
                            now_ns(),
                            HOST,
                            "collector_error",
                            "error",
                            msg,
                            {"collector": c.name},
                        )
                    )
            ms = (time.perf_counter() - t0) * 1e3
            costs[c.name] = ms
            self._costs[c.name].append(ms)
            self._counts[c.name].append(len(got))
            samples.extend(got)
            drain = getattr(c, "drain_events", None)
            if callable(drain):
                events.extend(drain())
        events.extend(self.detector.feed(samples))
        meta = []
        for name, ms in costs.items():
            meta.append(
                Sample(
                    ts,
                    HOST,
                    "collector.cost_ms",
                    round(ms, 3),
                    "ms",
                    {"collector": name},
                )
            )
            meta.append(
                Sample(
                    ts,
                    HOST,
                    "collector.samples",
                    float(self._counts[name][-1]),
                    "count",
                    {"collector": name},
                )
            )
        all_samples = samples + meta
        to_write = self.dedup.filter(all_samples) if self.dedup else all_samples
        if self.session is not None:
            self.session.append_many([*to_write, *events])
        s = self.summary
        s.ticks += 1
        s.samples_seen += len(all_samples)
        s.samples_written += len(to_write)
        s.events += len(events)
        for ev in events:
            s.event_names[ev.name] = s.event_names.get(ev.name, 0) + 1
        return Tick(ts, all_samples, events, costs)

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_ticks: int | None = None,
        stop: threading.Event | None = None,
        on_tick: Callable[[Tick], None] | None = None,
    ) -> RunSummary:
        """Tick every ``interval_s`` (absolute schedule) until a limit or ``stop``."""
        stop = stop or threading.Event()
        start = time.monotonic()
        k = 0
        self._mark(
            "watch_start",
            {
                "interval_s": self.interval_s,
                "collectors": [c.name for c in self.collectors],
            },
        )
        try:
            while not stop.is_set():
                tk = self.tick()
                if on_tick is not None:
                    on_tick(tk)
                k += 1
                if max_ticks is not None and k >= max_ticks:
                    break
                nxt = start + k * self.interval_s
                now = time.monotonic()
                if duration_s is not None and nxt - start >= duration_s:
                    break
                if nxt <= now:
                    self.summary.overruns += 1
                    nxt = now
                if stop.wait(nxt - now):
                    break
        finally:
            self.summary.duration_s = time.monotonic() - start
            self._finish()
            self._mark("watch_stop", {"ticks": self.summary.ticks})
        return self.summary

    def _mark(self, name: str, data: dict[str, Any]) -> None:
        if self.session is not None:
            self.session.append(Event(now_ns(), HOST, name, "info", "", data))

    def _finish(self) -> None:
        s = self.summary
        s.cost_ms = {n: _stats(v) for n, v in self._costs.items()}
        s.samples_per_tick = {
            n: (sum(v) / len(v) if v else 0.0) for n, v in self._counts.items()
        }


@dataclass(slots=True)
class WatchHandle:
    """A ``Runner`` running on a background thread."""

    runner: Runner
    thread: threading.Thread
    stop_event: threading.Event
    session: Session
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def status(self) -> dict[str, Any]:
        s = self.runner.summary
        return {
            "session": self.session.name,
            "path": str(self.session.path),
            "running": self.running,
            "ticks": s.ticks,
            "samples_written": s.samples_written,
            "events": s.events,
            "interval_s": self.runner.interval_s,
        }

    def stop(self, timeout: float = 30.0) -> dict[str, Any]:
        """Stop the loop, finalise the session and return the run summary."""
        self.stop_event.set()
        self.thread.join(timeout)
        if self.session.writable:
            self.session.close("ok")
        out = self.runner.summary.to_dict()
        out.update({"session": self.session.name, "path": str(self.session.path)})
        return out


def start_watch(
    topo: Topology,
    collectors: list[Collector],
    session: Session,
    *,
    interval_s: float = 1.0,
    duration_s: float | None = None,
    dedup: bool = True,
) -> WatchHandle:
    """Run a ``Runner`` for ``session`` on a daemon thread."""
    runner = Runner(
        topo, collectors, session=session, interval_s=interval_s, dedup=dedup
    )
    stop = threading.Event()

    def _loop() -> None:
        try:
            runner.run(duration_s=duration_s, stop=stop)
        finally:
            if session.writable and duration_s is not None:
                session.close("ok")

    th = threading.Thread(
        target=_loop, name=f"pciescope-watch-{session.name}", daemon=True
    )
    handle = WatchHandle(runner, th, stop, session)
    th.start()
    return handle


def snapshot(topo: Topology, collectors: list[Collector]) -> Tick:
    """Collect once without writing anything (the ``snap`` verb)."""
    return Runner(topo, collectors, dedup=False).tick()
