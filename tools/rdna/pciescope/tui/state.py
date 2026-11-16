# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rolling view state fed by Samples, Measurements and Events.

Pure data (no textual): the same object backs the live app, replay and the
render-function tests.  Series are keyed like ``Sample.key()`` (entity, metric,
sorted tags) so tagged series (``aer.error`` per error type, ``kfd.evicted_ms``
per pid) never mix.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pciescope.model import Event, Measurement, Sample

HISTORY = 240  # samples kept per series
NO_TAGS: tuple = ()


@dataclass(slots=True)
class Latest:
    ts_ns: int
    value: float
    unit: str
    tags: dict


@dataclass(slots=True)
class ErrorRow:
    """One line of the errors panel: a counter delta or a warn/error event."""

    ts_ns: int
    entity: str
    what: str  # metric (+ tags) or event name
    delta: float | None
    value: float | None
    severity: str = "warn"
    text: str = ""


DEV_STA_BITS = ((1, "CE"), (2, "NF"), (4, "F"), (8, "UR"))
# events that restate a counter/mask change this module already derives
DERIVED_EVENTS = ("aer_delta", "plx_error_flags")


def is_plx_error_mask(metric: str) -> bool:
    """Sticky error registers of a switch port (device, AER, DPC status and the
    lane error mask).  ``plx.link_bw_status`` is a speed-change flag, not an
    error."""
    return (
        metric.startswith("plx.")
        and metric.endswith(("_status", "_err"))
        and metric != "plx.link_bw_status"
    )


def is_error_metric(metric: str) -> bool:
    """Metrics that mean something went wrong (AER counters, switch error regs)."""
    return metric.startswith("aer.") or is_plx_error_mask(metric)


def error_change(metric: str, prev: float, cur: float) -> tuple[str, float] | None:
    """Describe a worsening between two reads of an error metric.

    Returns (human text, delta) or None.  AER counters are monotonic (delta > 0);
    the ``plx.*`` error registers are sticky bit masks (newly set bits).
    """
    if is_plx_error_mask(metric):
        new = int(cur) & ~int(prev)
        if not new:
            return None
        if metric == "plx.dev_status":
            names = "+".join(n for b, n in DEV_STA_BITS if new & b)
            return f"new bits {names or hex(new)}", float(new)
        return f"new bits {new:#x}", float(new)
    if cur > prev:
        return "", cur - prev
    return None


def is_nonzero_error(metric: str, value: float) -> bool:
    return is_error_metric(metric) and value > 0


def tag_suffix(tags: dict) -> str:
    return "".join(f"[{v}]" for _, v in sorted(tags.items()))


@dataclass
class ViewState:
    series: dict = field(default_factory=dict)  # Sample.key() -> deque[(ts, v)]
    latest: dict = field(default_factory=dict)  # Sample.key() -> Latest
    errors: deque = field(default_factory=lambda: deque(maxlen=500))
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    measurements: list = field(default_factory=list)
    unavailable: dict = field(default_factory=dict)  # collector -> reason
    collectors: list = field(default_factory=list)  # names of active collectors
    first_ts: int | None = None
    last_ts: int | None = None
    n_samples: int = 0
    baseline_ts: int | None = None
    interval_ns: int = 1_000_000_000  # nominal sampling period (forward-fill grid)
    prev_real: dict = field(default_factory=dict)  # key -> (ts, v) before latest

    # -- ingest -----------------------------------------------------------
    def ingest_sample(self, s: Sample) -> None:
        key = s.key()
        self.n_samples += 1
        if self.first_ts is None or s.ts_ns < self.first_ts:
            self.first_ts = s.ts_ns
        if self.last_ts is None or s.ts_ns > self.last_ts:
            self.last_ts = s.ts_ns
        prev = self.latest.get(key)
        if prev is not None and is_error_metric(s.metric):
            ch = error_change(s.metric, prev.value, s.value)
            if ch is not None:
                text, delta = ch
                self.errors.append(
                    ErrorRow(
                        s.ts_ns,
                        s.entity,
                        s.metric + tag_suffix(s.tags),
                        delta,
                        s.value,
                        "error" if "fatal" in s.metric else "warn",
                        text,
                    )
                )
        dq = self.series.get(key)
        if dq is None:
            dq = self.series[key] = deque(maxlen=HISTORY)
        if prev is not None:
            self._fill(dq, prev.value, s.ts_ns)
        dq.append((s.ts_ns, s.value))
        if prev is not None:
            self.prev_real[key] = (prev.ts_ns, prev.value)
        self.latest[key] = Latest(s.ts_ns, s.value, s.unit, dict(s.tags))

    def _fill(self, dq: deque, value: float, until_ns: int) -> None:
        """Repeat ``value`` on the sampling grid after the series tail.

        Stored sessions are deduplicated (unchanged samples are not written),
        so a series is held at its last value between records.
        """
        step = self.interval_ns
        last = dq[-1][0]
        if until_ns - last < step * 3 // 2:
            return
        n = min((until_ns - last - step // 2) // step, HISTORY)
        t = until_ns - n * step if n == HISTORY else last + step
        for _ in range(n):
            dq.append((t, value))
            t += step

    def fill_to(self, ts_ns: int) -> None:
        """Hold every series at its last value up to ``ts_ns`` (display clock)."""
        for key, lt in self.latest.items():
            self._fill(self.series[key], lt.value, ts_ns)

    def ingest_measurement(self, m: Measurement) -> None:
        self.measurements.append(m)
        if m.ended_ns and (self.last_ts is None or m.ended_ns > self.last_ts):
            self.last_ts = m.ended_ns

    def ingest_event(self, e: Event) -> None:
        self.events.append(e)
        if self.last_ts is None or e.ts_ns > self.last_ts:
            self.last_ts = e.ts_ns
        if e.severity in ("warn", "error") and e.name not in DERIVED_EVENTS:
            self.errors.append(
                ErrorRow(e.ts_ns, e.entity, e.name, None, None, e.severity, e.message)
            )

    def ingest(self, records) -> None:
        for r in records:
            if isinstance(r, Sample):
                self.ingest_sample(r)
            elif isinstance(r, Measurement):
                self.ingest_measurement(r)
            elif isinstance(r, Event):
                self.ingest_event(r)

    # -- queries used by renderers ----------------------------------------
    def get(self, entity: str, metric: str) -> Latest | None:
        return self.latest.get((entity, metric, NO_TAGS))

    def value(self, entity: str, metric: str) -> float | None:
        lt = self.get(entity, metric)
        return None if lt is None else lt.value

    def values(self, entity: str, metric: str) -> list[float]:
        dq = self.series.get((entity, metric, NO_TAGS))
        return [v for _, v in dq] if dq else []

    def rate(self, entity: str, metric: str) -> float | None:
        """Per-second rate of a counter between its last two real samples."""
        key = (entity, metric, NO_TAGS)
        lt, prev = self.latest.get(key), self.prev_real.get(key)
        if lt is None or prev is None:
            return None
        (t0, v0), (t1, v1) = prev, (lt.ts_ns, lt.value)
        if t1 <= t0 or v1 < v0:
            return None
        return (v1 - v0) / ((t1 - t0) / 1e9)

    def rates(self, entity: str, metric: str) -> list[float]:
        dq = list(self.series.get((entity, metric, NO_TAGS)) or ())
        out = []
        for (t0, v0), (t1, v1) in zip(dq, dq[1:]):
            out.append(max(v1 - v0, 0.0) / ((t1 - t0) / 1e9) if t1 > t0 else 0.0)
        return out

    def has_metric(self, metric: str) -> bool:
        return any(k[1] == metric for k in self.latest)

    def tagged(self, metric: str):
        """Yield (entity, tags, Latest) for every series of ``metric``."""
        for (ent, met, _), lt in self.latest.items():
            if met == metric:
                yield ent, lt.tags, lt

    def counters_nonzero(self):
        """Error counters currently above zero: [(entity, label, value)]."""
        out = []
        for (ent, met, _), lt in self.latest.items():
            if is_nonzero_error(met, lt.value):
                out.append((ent, met + tag_suffix(lt.tags), lt.value))
        return sorted(out)

    def n_error_series(self) -> int:
        return sum(1 for (_, m, _) in self.latest if is_error_metric(m))
