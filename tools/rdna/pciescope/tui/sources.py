# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data sources of the TUI: live passive collectors, or a recorded session.

Both expose the same small interface to the app: ``poll()`` returns the records
that became visible since the last call, ``generation`` changes when the
visible history was rewound (replay seek backwards) so the app rebuilds its
``ViewState`` from ``history()``.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable
from typing import Protocol

from pciescope.model import Event, Measurement, Record, Sample, Topology, now_ns

NS = 1_000_000_000


def record_ts(r: Record) -> int:
    """Time a record becomes visible (measurements: when they finished)."""
    if isinstance(r, Measurement):
        return r.ended_ns or r.started_ns
    return r.ts_ns


class Source(Protocol):
    mode: str  # 'live' | 'replay'
    topo: Topology
    collectors: list[str]
    unavailable: dict[str, str]
    paused: bool
    generation: int

    def start(self) -> None: ...

    def poll(self) -> list[Record]: ...

    def history(self) -> list[Record]: ...

    def display_ns(self) -> int | None: ...

    def close(self) -> None: ...


class LiveSource:
    """Sample the passive collectors on a worker thread via Core's ``Runner``.

    Collectors are read-only and cheap, but a hung sysfs read must never block
    the UI, so sampling happens off the event loop.  The runner derives the
    events (link changes, AER deltas, thermal, KFD evictions) and, while
    ``recording``, appends deduplicated records to ``session``; the display gets
    every sample.  Recording continues while the display is frozen.
    """

    mode = "live"
    generation = 0

    def __init__(
        self,
        topo: Topology,
        runner,
        skipped: Iterable[tuple[str, str]] = (),
        *,
        interval: float = 1.0,
        session=None,
        session_factory: Callable[[], object] | None = None,
        threaded: bool = True,
    ) -> None:
        self.topo = topo
        self.runner = runner
        self._threaded = threaded
        self.collectors = [c.name for c in runner.collectors]
        self.unavailable = dict(skipped)
        self.interval = max(0.1, interval)
        self.session = session
        self.session_factory = session_factory
        self.recording = session is not None
        self.paused = False
        self._q: queue.Queue[Record] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ns: int | None = None
        self.ticks = 0
        self.errors: list[str] = []

    # lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None and self._threaded:
            self._mark("watch_start")
            self._thread = threading.Thread(
                target=self._run, name="tui-sampler", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._mark("watch_stop", ticks=self.ticks)
        if self.session is not None:
            self.session.close()
            self.session = None
            self.recording = False

    def _mark(self, name: str, **data) -> None:
        if self.session is not None and self.recording:
            self.session.append(
                Event(
                    now_ns(),
                    "host",
                    name,
                    "info",
                    "",
                    {
                        "source": "tui",
                        "interval_s": self.interval,
                        "collectors": self.collectors,
                        **data,
                    },
                )
            )

    def _run(self) -> None:
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # the sampler must outlive any one failure
                self.errors.append(f"{type(exc).__name__}: {exc}")
            nxt += self.interval
            self._stop.wait(max(0.0, nxt - time.monotonic()))

    def sample_once(self) -> list[Record]:
        """One tick of the runner; queue its samples and events for display."""
        self.runner.session = self.session if self.recording else None
        tick = self.runner.tick()
        self.ticks += 1
        out: list[Record] = [*tick.samples, *tick.events]
        for r in out:
            self._q.put(r)
        return out

    def publish(self, r: Record) -> None:
        """Thread-safe entry for records of other producers (probe runs)."""
        if self.session is not None and self.recording:
            try:
                self.session.append(r)
            except (OSError, ValueError) as exc:
                self.errors.append(f"session: {exc}")
                self.recording = False
        self._q.put(r)

    # app side -----------------------------------------------------------------
    def poll(self) -> list[Record]:
        if self.paused:
            return []
        out: list[Record] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        for r in out:
            if isinstance(r, Sample):
                self._last_ns = r.ts_ns
        return out

    def history(self) -> list[Record]:
        return []

    def display_ns(self) -> int | None:
        return self._last_ns

    @property
    def session_path(self) -> str:
        return str(self.session.path) if self.session is not None else ""

    def flush(self) -> None:
        if self.session is not None:
            self.session.flush()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_record(self) -> None:
        """Stop recording, or resume (starting a session when there is none)."""
        if self.recording:
            self._mark("recording_pause")
            self.recording = False
            return
        if self.session is None:
            if self.session_factory is None:
                return
            self.session = self.session_factory()
        self.recording = True
        old = getattr(self.runner, "dedup", None)
        if old is not None:  # a fresh dedup state: re-emit every series once
            from pciescope.runner import Dedup

            self.runner.dedup = Dedup(old.heartbeat_ns / 1e9)
        self._mark("recording_resume")


class ReplaySource:
    """Play back a recorded session (records sorted by visibility time).

    The playhead advances ``speed`` x real time while not paused.  Seeking back
    bumps ``generation``; the app then rebuilds its view state from
    ``history()`` (everything up to the playhead).
    """

    mode = "replay"

    def __init__(
        self,
        topo: Topology,
        records: Iterable[Record],
        *,
        path: str = "",
        collectors: list[str] | None = None,
        unavailable: dict[str, str] | None = None,
        speed: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        start_paused: bool = False,
        interval_s: float = 1.0,
    ) -> None:
        self.topo = topo
        self.path = path
        self.records = sorted(records, key=record_ts)
        self.collectors = list(collectors or [])
        self.unavailable = dict(unavailable or {})
        self.speed = speed
        self.interval = interval_s
        self.paused = start_paused
        self.generation = 0
        self.t0 = record_ts(self.records[0]) if self.records else 0
        self.t1 = record_ts(self.records[-1]) if self.records else 0
        self.playhead = self.t0
        self._idx = 0
        self._clock = clock
        self._last = clock()

    def start(self) -> None:
        self._last = self._clock()

    def close(self) -> None:
        pass

    @property
    def duration_s(self) -> float:
        return (self.t1 - self.t0) / NS

    @property
    def pos_s(self) -> float:
        return (self.playhead - self.t0) / NS

    @property
    def finished(self) -> bool:
        return self._idx >= len(self.records)

    def _advance_clock(self) -> None:
        now = self._clock()
        dt, self._last = now - self._last, now
        if not self.paused and dt > 0:
            self.playhead = min(self.t1, self.playhead + int(dt * self.speed * NS))

    def poll(self) -> list[Record]:
        self._advance_clock()
        out: list[Record] = []
        n = len(self.records)
        while self._idx < n and record_ts(self.records[self._idx]) <= self.playhead:
            out.append(self.records[self._idx])
            self._idx += 1
        if self._idx >= n and self.playhead >= self.t1 and not self.paused:
            self.playhead = self.t1
        return out

    def history(self) -> list[Record]:
        return self.records[: self._idx]

    def display_ns(self) -> int | None:
        return self.playhead if self.records else None

    def toggle_pause(self) -> None:
        self._advance_clock()
        self.paused = not self.paused

    def seek(self, delta_s: float) -> None:
        self.seek_to(self.pos_s + delta_s)

    def seek_to(self, pos_s: float) -> None:
        self._advance_clock()
        pos_s = max(0.0, min(self.duration_s, pos_s))
        new_head = self.t0 + int(pos_s * NS)
        rewound = new_head < self.playhead
        self.playhead = new_head
        if rewound:
            self._idx = 0
            while (
                self._idx < len(self.records)
                and record_ts(self.records[self._idx]) <= new_head
            ):
                self._idx += 1
            self.generation += 1

    def set_speed(self, factor: float) -> None:
        self._advance_clock()
        self.speed = max(0.25, min(64.0, self.speed * factor))
