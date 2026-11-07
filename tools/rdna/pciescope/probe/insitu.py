# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-situ mode: link utilisation of a running workload from passive counters.

Nothing is injected: ``observe`` runs Core's collectors for a while and
``analyze`` turns the traffic counters they report into per-link, per-direction
utilisation against the link's raw and practical capacity. Counter sources:

* ``port.rx_bytes`` / ``port.tx_bytes`` (counters) or ``port.rx_gbs`` /
  ``port.tx_gbs`` (gauges) of the PEX88096 switch ports (collector ``plx``);
  ``rx`` is traffic into the switch port from its link partner;
* ``gpu.pcie_rx_bytes_per_s`` / ``gpu.pcie_tx_bytes_per_s`` of the amdgpu
  driver (``rx`` is into the GPU).

[INFERENCE: docs/rdna2/pciescope-2026-11-07.md] On par1-cs25 neither source
reports today: the switch counters are not free-running (enabling them writes
switch registers, which needs lead approval) and amdgpu exposes no ``pcie_bw``
on RDNA2/3. Then ``available`` is False, ``reason`` says which source is
missing, and no utilisation is invented; the GPU load seen in the window is
still reported as context.
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pciescope.model import (
    Event,
    Sample,
    Topology,
    now_ns,
    port_entity,
    practical_gbs,
)
from pciescope.sysfs import Sysfs

#: counters (bytes) and gauges that carry per-port traffic of the switch
PORT_COUNTERS = {"port.rx_bytes": "rx", "port.tx_bytes": "tx"}
PORT_GAUGES = {"port.rx_gbs": "rx", "port.tx_gbs": "tx"}
GPU_GAUGES = {"gpu.pcie_rx_bytes_per_s": "rx", "gpu.pcie_tx_bytes_per_s": "tx"}

#: collectors in-situ mode asks for (the switch reader is skipped if unavailable)
DEFAULT_COLLECTORS = ("sysfs_link", "aer", "amdgpu", "kfd", "plx")

CONTEXT_METRICS = ("gpu.busy_pct", "gpu.mem_busy_pct", "gpu.power_w")


@dataclass
class LinkUtil:
    """Utilisation of one direction of one physical link."""

    link: str
    direction: str  # up = toward the root complex, down = toward the GPU
    owner: str  # the GPU or switch the link belongs to
    source: str  # plx | amdgpu
    samples: int
    mean_gbs: float
    p50_gbs: float
    p99_gbs: float
    peak_gbs: float
    raw_gbs: float
    practical_gbs: float | None

    @property
    def util_mean_pct(self) -> float:
        return 100.0 * self.mean_gbs / self.raw_gbs

    @property
    def util_peak_pct(self) -> float:
        return 100.0 * self.peak_gbs / self.raw_gbs

    @property
    def util_peak_practical_pct(self) -> float | None:
        if not self.practical_gbs:
            return None
        return 100.0 * self.peak_gbs / self.practical_gbs

    def to_dict(self) -> dict[str, Any]:
        return {
            "link": self.link,
            "direction": self.direction,
            "owner": self.owner,
            "source": self.source,
            "samples": self.samples,
            "mean_gbs": round(self.mean_gbs, 4),
            "p50_gbs": round(self.p50_gbs, 4),
            "p99_gbs": round(self.p99_gbs, 4),
            "peak_gbs": round(self.peak_gbs, 4),
            "raw_gbs": self.raw_gbs,
            "practical_gbs": self.practical_gbs,
            "util_mean_pct": round(self.util_mean_pct, 2),
            "util_peak_pct": round(self.util_peak_pct, 2),
            "util_peak_practical_pct": None
            if self.util_peak_practical_pct is None
            else round(self.util_peak_practical_pct, 2),
        }


@dataclass
class InsituReport:
    """Result of an in-situ observation."""

    available: bool
    window_s: float
    n_samples: int
    sources: dict[str, int] = field(default_factory=dict)  # source -> series seen
    links: list[LinkUtil] = field(default_factory=list)
    reason: str = ""  # why no utilisation is available
    context: dict[str, Any] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "window_s": round(self.window_s, 2),
            "samples": self.n_samples,
            "sources": dict(self.sources),
            "reason": self.reason,
            "links": [lk.to_dict() for lk in self.links],
            "context": self.context,
            "skipped_collectors": [{"name": n, "why": w} for n, w in self.skipped],
        }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def port_directions(topo: Topology) -> dict[tuple[str, str], tuple[str, str]]:
    """Map ``(port entity, 'rx'|'tx')`` to ``(link id, 'up'|'down')``.

    Only physical links: a link's upstream port sees the child's traffic as
    ``rx`` (up) and its own traffic toward the child as ``tx`` (down); the
    downstream end of the link sees the opposite.
    """
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for link in topo.links.values():
        if not link.physical or not link.max_gbs:
            continue
        out[(port_entity(link.up), "rx")] = (link.id, "up")
        out[(port_entity(link.up), "tx")] = (link.id, "down")
        out.setdefault((port_entity(link.down), "rx"), (link.id, "down"))
        out.setdefault((port_entity(link.down), "tx"), (link.id, "up"))
    return out


def _counter_rates(series: Sequence[Sample]) -> list[float]:
    """GB/s between consecutive samples of a byte counter (resets skipped)."""
    rates: list[float] = []
    ordered = sorted(series, key=lambda s: s.ts_ns)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        dt = (cur.ts_ns - prev.ts_ns) / 1e9
        delta = cur.value - prev.value
        if dt > 0 and delta >= 0:
            rates.append(delta / dt / 1e9)
    return rates


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _make_util(
    topo: Topology,
    link_id: str,
    direction: str,
    source: str,
    rates: Sequence[float],
) -> LinkUtil | None:
    link = topo.links.get(link_id)
    if link is None or not link.max_gbs or not rates:
        return None
    practical = (
        practical_gbs(link.max_speed_gts, link.max_width)
        if link.max_speed_gts and link.max_width
        else None
    )
    return LinkUtil(
        link=link_id,
        direction=direction,
        owner=link.gpu or (f"switch:{link.switch}" if link.switch else ""),
        source=source,
        samples=len(rates),
        mean_gbs=statistics.fmean(rates),
        p50_gbs=statistics.median(rates),
        p99_gbs=_percentile(rates, 0.99),
        peak_gbs=max(rates),
        raw_gbs=link.max_gbs,
        practical_gbs=practical,
    )


def _series(samples: Iterable[Sample]) -> dict[tuple[str, str], list[Sample]]:
    out: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        out.setdefault((s.entity, s.metric), []).append(s)
    return out


def _context(
    topo: Topology, series: dict[tuple[str, str], list[Sample]]
) -> dict[str, Any]:
    gpus: dict[str, dict[str, float]] = {}
    for (entity, metric), ss in series.items():
        if metric in CONTEXT_METRICS and entity in topo.gpus:
            vals = [s.value for s in ss]
            gpus.setdefault(entity, {})[f"{metric}.mean"] = round(
                statistics.fmean(vals), 2
            )
            gpus[entity][f"{metric}.max"] = round(max(vals), 2)
    return {"gpus": gpus}


def _unavailable_reason(
    series: dict[tuple[str, str], list[Sample]], skipped: Sequence[tuple[str, str]]
) -> str:
    parts: list[str] = []
    plx = next((why for name, why in skipped if name == "plx"), None)
    enabled = [
        s.value for (e, m), ss in series.items() if m == "plx.perf_enabled" for s in ss
    ]
    if plx is not None:
        parts.append(f"switch counters: collector plx unavailable ({plx})")
    elif enabled and max(enabled) == 0:
        parts.append(
            "switch counters: plx.perf_enabled is 0 (the PEX88096 traffic counters "
            "are off; enabling them writes switch registers and needs approval)"
        )
    else:
        parts.append("switch counters: no port.rx/tx series reported")
    parts.append(
        "GPU side: no gpu.pcie_rx/tx_bytes_per_s series (amdgpu exposes no "
        "pcie_bw on RDNA2/3)"
    )
    return "; ".join(parts)


def analyze(
    samples: Iterable[Sample],
    topo: Topology,
    *,
    skipped: Sequence[tuple[str, str]] = (),
    window_s: float = 0.0,
) -> InsituReport:
    """Per-link utilisation from the traffic counters in ``samples``.

    The switch ports (``plx``) win over the GPU side (``amdgpu``) when both
    report the same link direction. ``window_s`` is only echoed (0: derived
    from the sample times).
    """
    samples = list(samples)
    series = _series(samples)
    mapping = port_directions(topo)
    rates: dict[
        tuple[str, str], tuple[str, list[float]]
    ] = {}  # (link, dir) -> src, rates
    seen: dict[str, int] = {}
    for (entity, metric), ss in series.items():
        if entity.startswith("port:") and metric in PORT_COUNTERS | PORT_GAUGES:
            side = (PORT_COUNTERS | PORT_GAUGES)[metric]
            target = mapping.get((entity, side))
            if target is None:
                continue
            vals = (
                _counter_rates(ss) if metric in PORT_COUNTERS else [s.value for s in ss]
            )
            if vals:
                seen["plx"] = seen.get("plx", 0) + 1
                rates[target] = ("plx", vals)
        elif entity in topo.gpus and metric in GPU_GAUGES:
            gpu = topo.gpus[entity]
            if gpu.link is None:
                continue
            direction = "down" if GPU_GAUGES[metric] == "rx" else "up"
            vals = [s.value / 1e9 for s in ss]
            if vals:
                seen["amdgpu"] = seen.get("amdgpu", 0) + 1
                rates.setdefault((gpu.link, direction), ("amdgpu", vals))
    links = [
        u
        for (link, direction), (source, vals) in sorted(rates.items())
        if (u := _make_util(topo, link, direction, source, vals)) is not None
    ]
    links.sort(key=lambda u: -u.util_peak_pct)
    times = [s.ts_ns for s in samples]
    span = (max(times) - min(times)) / 1e9 if times else 0.0
    return InsituReport(
        available=bool(links),
        window_s=window_s or span,
        n_samples=len(samples),
        sources=seen,
        links=links,
        reason="" if links else _unavailable_reason(series, skipped),
        context=_context(topo, series),
        skipped=list(skipped),
    )


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------


def observe(
    topo: Topology,
    *,
    duration_s: float,
    interval_s: float = 1.0,
    selected: Sequence[str] = DEFAULT_COLLECTORS,
    sysfs: Sysfs | None = None,
    session: Any = None,
    cancel: threading.Event | None = None,
    on_tick: Callable[[int, float], None] | None = None,
) -> InsituReport:
    """Run the passive collectors for ``duration_s`` and analyse the counters.

    No GPU work is started and no lease is needed: only Core's collectors read
    sysfs (``sysfs``: the view to read, default the live ``/sys``) and the
    switch reader, when it is available. ``session`` is a
    ``pciescope.store.Session`` that receives the samples (so the trace also
    holds them); ``on_tick(n, elapsed_s)`` reports progress.
    """
    from pciescope import collectors
    from pciescope.runner import Runner, Tick

    built = collectors.build(topo, list(selected), sysfs=sysfs)
    runner = Runner(
        topo, built.collectors, session=session, interval_s=interval_s, dedup=False
    )
    taken: list[Sample] = []
    count = 0

    def collect(tick: Tick) -> None:
        nonlocal count
        taken.extend(tick.samples)
        count += 1
        if on_tick is not None:
            on_tick(count, count * interval_s)

    runner.run(duration_s=duration_s, stop=cancel, on_tick=collect)
    report = analyze(
        taken, topo, skipped=built.skipped, window_s=runner.summary.duration_s
    )
    if session is not None:
        session.append(
            Event(
                now_ns(),
                "host",
                "insitu_end",
                "info" if report.available else "warn",
                "in-situ observation: "
                + ("utilisation available" if report.available else report.reason),
                {"available": report.available, "links": len(report.links)},
            )
        )
    return report


def render_text(report: InsituReport) -> str:
    """Human-readable form of an in-situ report."""
    lines = [
        f"in-situ observation: {report.window_s:.0f} s, {report.n_samples} samples, "
        f"sources {report.sources or 'none'}"
    ]
    if not report.available:
        lines.append("link utilisation: UNAVAILABLE")
        lines.append(f"  {report.reason}")
    else:
        lines.append("link utilisation (GB/s, % of raw line rate):")
        lines.append(
            f"  {'link':<28s} {'dir':<5s} {'owner':<9s} {'mean':>7s} {'p99':>7s} "
            f"{'peak':>7s} {'raw':>6s} {'mean%':>6s} {'peak%':>6s} src"
        )
        for u in report.links:
            lines.append(
                f"  {u.link:<28s} {u.direction:<5s} {u.owner:<9s} "
                f"{u.mean_gbs:7.2f} {u.p99_gbs:7.2f} {u.peak_gbs:7.2f} "
                f"{u.raw_gbs:6.1f} {u.util_mean_pct:6.1f} {u.util_peak_pct:6.1f} "
                f"{u.source}"
            )
    gpus = report.context.get("gpus", {})
    if gpus:
        lines.append("GPU load in the window:")
        for entity, vals in sorted(gpus.items()):
            text = "  ".join(f"{k}={v:g}" for k, v in sorted(vals.items()))
            lines.append(f"  {entity}: {text}")
    for name, why in report.skipped:
        lines.append(f"collector {name} skipped: {why}")
    return "\n".join(lines)
