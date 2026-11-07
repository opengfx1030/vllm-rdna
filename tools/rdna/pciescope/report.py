# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarise a session: links, GPUs, errors, events, probe results, findings.

``build_report`` streams the trace once and returns a JSON-ready dict;
``render_text`` prints it for humans.  Samples are forward-filled (the writer
drops unchanged values), so averages are time-weighted over the session window.

Ceilings used (documented in docs/rdna2/pciescope-2026-11-07.md):

* ``theoretical`` / "raw": ``speed x width x 128/130 / 8`` per direction, 31.51
  GB/s for Gen4 x16;
* ``practical``: raw x ``MPS / (MPS + 26)`` x 0.98 with the measured MPS of 256
  bytes: 28.03 GB/s for Gen4 x16 (max-size posted writes, one direction).

Per-link utilisation comes from (a) traffic counters when a collector provides
them (metrics ``*.rx_bytes`` / ``*.tx_bytes`` on ports, or ``*.rx_gbs`` /
``*.tx_gbs``) and (b) the probe: each bandwidth measurement is attributed to the
physical links of its route (``Topology.path``) in the direction it travelled.
Passive sysfs collectors alone expose no traffic counters, and the report says so.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pciescope.model import (
    DEFAULT_MPS,
    Event,
    Gpu,
    Link,
    Measurement,
    Sample,
    Topology,
    entity_kind,
    metric_kind,
    practical_gbs,
    short_bdf,
)
from pciescope.store import Session

Points = list[tuple[int, float]]
SeriesKey = tuple[str, str, tuple[tuple[str, str], ...]]


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def tw_stats(points: Points, t_end: int) -> dict[str, float]:
    """Time-weighted stats of a forward-filled series up to ``t_end``."""
    if not points:
        return {}
    pts = sorted(points)
    vals = [v for _, v in pts]
    total = 0.0
    weighted = 0.0
    for i, (ts, v) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else max(t_end, ts)
        dt = max(nxt - ts, 0)
        total += dt
        weighted += v * dt
    mean = weighted / total if total > 0 else vals[-1]
    return {
        "n": len(pts),
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
    }


def _seconds_where(points: Points, t_end: int, pred: Any) -> float:
    pts = sorted(points)
    acc = 0
    for i, (ts, v) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else max(t_end, ts)
        if pred(v):
            acc += max(nxt - ts, 0)
    return acc / 1e9


def _percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _series_name(key: SeriesKey) -> str:
    ent, metric, tags = key
    t = ",".join(f"{k}={v}" for k, v in tags)
    return f"{ent} {metric}" + (f"{{{t}}}" if t else "")


class _Data:
    """Everything read from the trace in one pass."""

    def __init__(self) -> None:
        self.series: dict[SeriesKey, Points] = {}
        self.events: list[Event] = []
        self.measurements: list[Measurement] = []
        self.t_first: int | None = None
        self.t_last: int | None = None
        self.n_samples = 0

    def add(self, rec: Any) -> None:
        if isinstance(rec, Sample):
            self.series.setdefault(rec.key(), []).append((rec.ts_ns, rec.value))
            self.n_samples += 1
            ts = rec.ts_ns
        elif isinstance(rec, Measurement):
            self.measurements.append(rec)
            ts = rec.started_ns
            self._span(rec.ended_ns)
        else:
            self.events.append(rec)
            ts = rec.ts_ns
        self._span(ts)

    def _span(self, ts: int) -> None:
        if self.t_first is None or ts < self.t_first:
            self.t_first = ts
        if self.t_last is None or ts > self.t_last:
            self.t_last = ts

    def get(self, entity: str, metric: str) -> Points:
        return self.series.get((entity, metric, ()), [])

    def tagged(
        self, entity: str, metric: str
    ) -> dict[tuple[tuple[str, str], ...], Points]:
        return {
            k[2]: v for k, v in self.series.items() if k[0] == entity and k[1] == metric
        }

    def of_metric(self, metric: str) -> dict[SeriesKey, Points]:
        return {k: v for k, v in self.series.items() if k[1] == metric}


def _load(records: Iterable[Any]) -> _Data:
    d = _Data()
    for rec in records:
        d.add(rec)
    return d


# ---------------------------------------------------------------------------


def build_report(
    session: Session,
    topo: Topology | None = None,
    *,
    include_virtual: bool = False,
) -> dict[str, Any]:
    """Build the report document of ``session`` (``topo`` defaults to its own)."""
    topo = topo or session.topology()
    data = _load(session.records())
    meta = session.meta
    t0 = data.t_first if data.t_first is not None else int(meta.get("created_ns") or 0)
    t1 = data.t_last if data.t_last is not None else t0
    if meta.get("ended_ns"):
        t1 = max(t1, int(meta["ended_ns"]))
    rep: dict[str, Any] = {
        "session": {
            "name": session.name,
            "path": str(session.path),
            "host": meta.get("host") or (topo.host if topo else ""),
            "label": meta.get("label"),
            "status": meta.get("status"),
            "start_ns": t0,
            "end_ns": t1,
            "duration_s": _r((t1 - t0) / 1e9, 3),
            "interval_s": meta.get("interval_s"),
            "collectors": meta.get("collectors"),
            "skipped": meta.get("skipped"),
            "tool_version": meta.get("tool_version"),
            "counts": {
                "samples": data.n_samples,
                "series": len(data.series),
                "measurements": len(data.measurements),
                "events": len(data.events),
            },
            "dedup_note": (
                "unchanged samples are not written (heartbeat 30 s); averages "
                "are time-weighted with forward fill"
            ),
        },
        "ceilings": {
            "mps_bytes": DEFAULT_MPS,
            "gen4_x16_raw_gbs": _r(practical_ref(raw=True), 3),
            "gen4_x16_practical_gbs": _r(practical_ref(raw=False), 3),
        },
    }
    rep["collector_cost"] = _collector_cost(data)
    if topo is not None:
        rep["links"] = _links(data, topo, t1, include_virtual)
        rep["gpus"] = _gpus(data, topo, t1)
    rep["errors"] = _errors(data, topo)
    rep["events"] = _events(data, t0)
    rep["probes"] = _probes(data, topo, t0)
    rep["findings"] = _findings(rep, data, topo)
    return rep


def practical_ref(*, raw: bool) -> float:
    """Gen4 x16 reference ceilings (GB/s) quoted in every report."""
    from pciescope.model import raw_gbs

    return raw_gbs(16.0, 16) if raw else practical_gbs(16.0, 16)


def _collector_cost(data: _Data) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pts in data.of_metric("collector.cost_ms").items():
        name = dict(key[2]).get("collector", "?")
        vals = [v for _, v in pts]
        out[name] = {
            "n": len(vals),
            "mean_ms": _r(sum(vals) / len(vals), 3),
            "p50_ms": _r(_median(vals), 3),
            "p95_ms": _r(_percentile(vals, 0.95), 3),
            "max_ms": _r(max(vals), 3),
        }
    for key, pts in data.of_metric("collector.samples").items():
        name = dict(key[2]).get("collector", "?")
        if name in out:
            out[name]["samples_per_tick"] = _r(sum(v for _, v in pts) / len(pts), 1)
    return out


def _rate_stats(
    data: _Data, ent: str, gauge: str, counter: str, t1: int
) -> dict[str, float] | None:
    """Mean and peak GB/s of ``ent``: a rate gauge, else the rate of a byte counter."""
    pts = data.get(ent, gauge)
    if pts:
        st = tw_stats(pts, t1)
        return {"mean": st["mean"], "max": st["max"]}
    ctr = data.get(ent, counter)
    rates = [
        (b[1] - a[1]) / ((b[0] - a[0]) / 1e9) / 1e9
        for a, b in zip(ctr, ctr[1:])
        if b[0] > a[0] and b[1] >= a[1]
    ]
    if not rates:
        return None
    return {"mean": sum(rates) / len(rates), "max": max(rates)}


def _link_traffic(
    data: _Data, topo: Topology, link: Link, t1: int
) -> list[dict[str, Any]] | None:
    """Per-port rates from the traffic counters on the link's ends, if any exist.

    ``port.rx_gbs`` / ``port.tx_gbs`` (or the rate of ``port.*_bytes``) count TLP and
    DLLP bytes on the wire, so they are set against the raw capacity;
    ``plx.*_payload_gbs`` (estimated payload) against the practical ceiling.  ``rx`` is
    what the port received from its link partner, ``tx`` what it sent.
    """
    raw = link.max_gbs
    prac = (
        practical_gbs(link.max_speed_gts, link.max_width)
        if link.max_speed_gts and link.max_width
        else None
    )
    rows: list[dict[str, Any]] = []
    for end, bdf in (("up", link.up), ("down", link.down)):
        ent = f"port:{short_bdf(bdf)}"
        for direction in ("rx", "tx"):
            rate = _rate_stats(
                data, ent, f"port.{direction}_gbs", f"port.{direction}_bytes", t1
            )
            if rate is None:
                continue
            row: dict[str, Any] = {
                "end": end,
                "port": ent,
                "direction": direction,
                "mean_gbs": _r(rate["mean"]),
                "peak_gbs": _r(rate["max"]),
            }
            if raw:
                row["util_raw_mean_pct"] = _r(100.0 * rate["mean"] / raw, 1)
                row["util_raw_pct"] = _r(100.0 * rate["max"] / raw, 1)
            payload = data.get(ent, f"plx.{direction}_payload_gbs")
            if payload:
                peak = tw_stats(payload, t1)["max"]
                row["payload_peak_gbs"] = _r(peak)
                if prac:
                    row["util_practical_pct"] = _r(100.0 * peak / prac, 1)
            rows.append(row)
    return rows or None


def _links(
    data: _Data, topo: Topology, t1: int, include_virtual: bool
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    changes: dict[str, int] = {}
    for ev in data.events:
        if ev.name == "link_change":
            changes[ev.entity] = changes.get(ev.entity, 0) + 1
    for lk in topo.links_ordered(physical_only=not include_virtual):
        speed = tw_stats(data.get(lk.id, "link.speed_gts"), t1)
        width = tw_stats(data.get(lk.id, "link.width"), t1)
        cap = tw_stats(data.get(lk.id, "link.capacity_gbs"), t1)
        degr = data.get(lk.id, "link.degraded")
        item: dict[str, Any] = {
            "id": lk.id,
            "kind": lk.kind,
            "gpu": lk.gpu,
            "switch": lk.switch,
            "max_speed_gts": lk.max_speed_gts,
            "max_width": lk.max_width,
            "raw_gbs_at_max": lk.max_gbs,
            "practical_gbs_at_max": _r(
                practical_gbs(lk.max_speed_gts, lk.max_width)
                if lk.max_speed_gts and lk.max_width
                else None
            ),
            "speed_gts": {k: speed[k] for k in ("min", "max", "last") if k in speed},
            "width": {k: width[k] for k in ("min", "max", "last") if k in width},
            "capacity_gbs_last": cap.get("last"),
            "degraded_s": _r(_seconds_where(degr, t1, lambda v: v >= 0.5), 1)
            if degr
            else None,
            "link_changes": changes.get(lk.id, 0),
        }
        traffic = _link_traffic(data, topo, lk, t1)
        if traffic:
            item["traffic_counters"] = traffic
        out.append(item)
    return out


def _gpus(data: _Data, topo: Topology, t1: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in topo.gpu_list():
        item: dict[str, Any] = {
            "id": g.id,
            "hip": g.hip,
            "arch": g.arch,
            "name": g.name,
            "switch": g.switch,
            "slot": g.slot,
        }
        for metric, label in (
            ("gpu.busy_pct", "busy_pct"),
            ("gpu.mem_busy_pct", "mem_busy_pct"),
            ("gpu.temp_edge_c", "temp_edge_c"),
            ("gpu.temp_junction_c", "temp_junction_c"),
            ("gpu.temp_mem_c", "temp_mem_c"),
            ("gpu.power_w", "power_w"),
            ("gpu.vram_used_bytes", "vram_used_bytes"),
            ("gpu.link_speed_gts", "link_speed_gts"),
            ("gpu.link_width", "link_width"),
            ("gpu.pcie_dpm_speed_gts", "dpm_speed_gts"),
        ):
            st = tw_stats(data.get(g.id, metric), t1)
            if st:
                item[label] = {k: _r(st[k], 2) for k in ("min", "mean", "max", "last")}
        thr = data.get(g.id, "gpu.throttle_status")
        if thr:
            item["throttle_seen"] = any(v for _, v in thr)
        ev = []
        for tags, pts in data.tagged(g.id, "kfd.evicted_ms").items():
            if len(pts) >= 1:
                delta = pts[-1][1] - pts[0][1]
                ev.append({**dict(tags), "delta_ms": delta, "total_ms": pts[-1][1]})
        if ev:
            item["kfd_evicted"] = sorted(ev, key=lambda e: -e["delta_ms"])
        out.append(item)
    return out


def _entity_label(topo: Topology | None, entity: str) -> str:
    if topo is None:
        return ""
    if entity.startswith("port:"):
        n = topo.nodes.get(entity[5:])
        return n.label if n else ""
    if entity.startswith("gpu:"):
        g = topo.gpus.get(entity)
        return g.name if g else ""
    return ""


def _errors(data: _Data, topo: Topology | None) -> dict[str, Any]:
    increased: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for key, pts in sorted(data.series.items()):
        ent, metric, tags = key
        if not metric.startswith("aer.") or metric == "aer.error" or not pts:
            continue
        first, last = pts[0][1], pts[-1][1]
        cls = metric.split(".", 1)[1]
        row = {"entity": ent, "label": _entity_label(topo, ent), "class": cls}
        if last > first:
            increased.append(
                {**row, "first": first, "last": last, "delta": last - first}
            )
        if last:
            kinds = {
                dict(k[2]).get("type", "?"): int(v[-1][1])
                for k, v in data.series.items()
                if k[0] == ent
                and k[1] == "aer.error"
                and dict(k[2]).get("class") == cls
            }
            totals.append({**row, "total": last, "types": kinds or None})
    return {
        "increased_during_session": increased,
        "nonzero_totals_since_boot": totals,
        "note": (
            "device counters (aer_dev_*) count errors the device itself reported; "
            "root-port totals count error messages received from everything below"
        ),
    }


def _events(data: _Data, t0: int) -> dict[str, Any]:
    by_name: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for ev in data.events:
        by_name[ev.name] = by_name.get(ev.name, 0) + 1
        by_sev[ev.severity] = by_sev.get(ev.severity, 0) + 1
    notable = [
        {
            "t_s": _r((ev.ts_ns - t0) / 1e9, 3),
            "severity": ev.severity,
            "entity": ev.entity,
            "name": ev.name,
            "message": ev.message,
        }
        for ev in data.events
        if ev.severity != "info" or ev.name in ("measure_start", "measure_end")
    ]
    return {"by_name": by_name, "by_severity": by_sev, "notable": notable[:200]}


def _bottleneck_practical(topo: Topology | None, m: Measurement) -> float | None:
    """Practical ceiling (GB/s) of the narrowest physical link on ``m``'s route."""
    if topo is None:
        return None
    try:
        info = topo.path(m.src, m.dst)
    except (KeyError, ValueError):
        return None
    best: float | None = None
    for lid in info.physical_links:
        lk = topo.links[lid]
        if lk.max_speed_gts and lk.max_width:
            p = practical_gbs(lk.max_speed_gts, lk.max_width)
            best = p if best is None else min(best, p)
    return best


_GroupKey = tuple[str, str, str, str, str]


def _group_key(m: Measurement) -> _GroupKey:
    """Kind, direction, path class, method and traffic pattern of a measurement."""
    return (
        m.kind,
        m.direction,
        m.path_class,
        m.tags.get("method", ""),
        m.tags.get("pattern", ""),
    )


def _single_flow(m: Measurement) -> bool:
    """One flow on its own: no traffic pattern, no concurrent flows."""
    return not m.tags.get("pattern") and m.tags.get("flows", "1") == "1"


def _summary_row(
    key: _GroupKey, by_size: dict[int, list[float]], unit: str
) -> dict[str, Any]:
    """One summary row per group at its headline size.

    Latency is summarised at the smallest size (the base latency); bandwidth,
    concurrency and all-to-all at the largest (the plateau).  ``sizes`` lists every
    size the group has, so a reader can ask for the others in ``measurements``.
    """
    sizes = sorted(by_size)
    size = sizes[0] if key[0] == "lat" else sizes[-1]
    v = by_size[size]
    return {
        "kind": key[0],
        "direction": key[1],
        "path_class": key[2],
        "method": key[3],
        "pattern": key[4],
        "size_bytes": size,
        "sizes": sizes,
        "n": len(v),
        "min": _r(min(v), 4),
        "median": _r(_median(v), 4),
        "max": _r(max(v), 4),
        "unit": unit,
    }


def _probes(data: _Data, topo: Topology | None, t0: int) -> dict[str, Any]:
    ms = data.measurements
    if not ms:
        return {"count": 0}
    rows: list[dict[str, Any]] = []
    groups: dict[_GroupKey, dict[int, list[float]]] = {}
    units: dict[_GroupKey, str] = {}
    util: dict[tuple[str, str], dict[str, Any]] = {}
    for i, m in enumerate(ms):
        pc = _bottleneck_practical(topo, m) if m.unit == "GB/s" else None
        row: dict[str, Any] = {
            "i": i,
            "t_s": _r((m.started_ns - t0) / 1e9, 3),
            "kind": m.kind,
            "src": m.src,
            "dst": m.dst,
            "direction": m.direction,
            "size_bytes": m.size_bytes,
            "value": _r(m.value, 4),
            "unit": m.unit,
            "p50": _r(m.stats.get("p50"), 4),
            "p99": _r(m.stats.get("p99"), 4),
            "n": m.stats.get("n"),
            "path_class": m.path_class,
            "theoretical": m.theoretical,
            "efficiency_raw": _r(m.efficiency, 4),
            "ok": m.ok,
        }
        if m.tags:
            row["tags"] = dict(m.tags)
        if pc and m.value:
            row["practical_gbs"] = _r(pc, 3)
            row["efficiency_practical"] = _r(m.value / pc, 4)
        if m.error:
            row["error"] = m.error
        rows.append(row)
        if m.ok:
            key = _group_key(m)
            groups.setdefault(key, {}).setdefault(m.size_bytes, []).append(m.value)
            units.setdefault(key, m.unit)
        if (
            topo is not None
            and m.kind == "bw"
            and m.unit == "GB/s"
            and m.ok
            and _single_flow(m)
        ):
            _attribute(topo, m, i, util)
    return {
        "count": len(ms),
        "summary": [
            _summary_row(k, by_size, units[k]) for k, by_size in sorted(groups.items())
        ],
        "matrix_gbs": _matrix(ms),
        "link_utilization": sorted(util.values(), key=lambda u: -u["util_raw_pct"]),
        "measurements": rows[:500],
    }


def _attribute(
    topo: Topology,
    m: Measurement,
    idx: int,
    util: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Charge a single-flow bandwidth result to the links of its route."""
    try:
        info = topo.path(m.src, m.dst)
    except (KeyError, ValueError):
        return
    for hop in info.hops:
        if hop.link is None or hop.link not in info.physical_links:
            continue
        lk = topo.links[hop.link]
        if not lk.max_gbs:
            continue
        key = (hop.link, hop.direction)
        cur = util.get(key)
        if cur is None or m.value > cur["peak_gbs"]:
            prac = practical_gbs(lk.max_speed_gts or 0, lk.max_width or 0)
            util[key] = {
                "link": hop.link,
                "direction": hop.direction,
                "peak_gbs": _r(m.value, 3),
                "raw_gbs": lk.max_gbs,
                "util_raw_pct": _r(100.0 * m.value / lk.max_gbs, 1),
                "util_practical_pct": _r(100.0 * m.value / prac, 1) if prac else None,
                "measurement": idx,
            }


#: data directions of the bandwidth matrix: into a peer's memory, over the host link
_MATRIX_DIRECTIONS = ("d2d-push", "h2d", "d2h")


def _matrix(ms: list[Measurement]) -> dict[str, Any] | None:
    """Bandwidth matrix source -> destination in GB/s (kernel-store GPU pairs, host).

    Each cell is the median over the repeats of the largest size measured for that
    pair, one flow at a time (traffic patterns, concurrent flows, the other copy
    methods and pageable host memory are left out).  ``size_bytes`` has the size
    behind every cell.
    """
    cells: dict[tuple[str, str], tuple[int, list[float]]] = {}
    for m in ms:
        if (
            m.kind != "bw"
            or m.unit != "GB/s"
            or not m.ok
            or m.src == m.dst
            or not m.size_bytes
            or m.direction not in _MATRIX_DIRECTIONS
            or not _single_flow(m)
            or m.tags.get("method") == "pageable"
        ):
            continue
        key = (m.src, m.dst)
        cur = cells.get(key)
        if cur is None or m.size_bytes > cur[0]:
            cells[key] = (m.size_bytes, [m.value])
        elif m.size_bytes == cur[0]:
            cur[1].append(m.value)
    if not cells:
        return None
    rows = sorted({k[0] for k in cells})
    cols = sorted({k[1] for k in cells})
    return {
        "rows": rows,
        "cols": cols,
        "values": [
            [
                _r(_median(cells[(r, c)][1]), 2) if (r, c) in cells else None
                for c in cols
            ]
            for r in rows
        ],
        "size_bytes": [
            [cells[(r, c)][0] if (r, c) in cells else None for c in cols] for r in rows
        ],
    }


def _findings(
    rep: dict[str, Any], data: _Data, topo: Topology | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(sev: str, code: str, entity: str, msg: str) -> None:
        out.append({"severity": sev, "code": code, "entity": entity, "message": msg})

    nonzero = rep["errors"]["nonzero_totals_since_boot"]
    if nonzero:
        classes = ", ".join(sorted({e["class"] for e in nonzero}))
        add(
            "info",
            "aer_nonzero_since_boot",
            "fabric",
            f"{len(nonzero)} AER counter(s) are non-zero since boot ({classes}); "
            "none of them increased during this session",
        )
    for lk in rep.get("links", []):
        if lk.get("degraded_s"):
            add(
                "warn",
                "link_degraded",
                lk["id"],
                f"{lk['id']} ({lk['kind']}) below its maximum for "
                f"{lk['degraded_s']} s: speed {lk['speed_gts']}, width {lk['width']}",
            )
        if lk.get("link_changes"):
            add(
                "warn",
                "link_retrain",
                lk["id"],
                f"{lk['link_changes']} speed/width change(s)",
            )
    for e in rep["errors"]["increased_during_session"]:
        add(
            "error" if "fatal" in e["class"] else "warn",
            "aer_increase",
            e["entity"],
            f"{e['entity']} {e['label']}: aer {e['class']} +{e['delta']:g} "
            "during the session",
        )
    for g in rep.get("gpus", []):
        tj = g.get("temp_junction_c")
        if tj and tj["max"] >= 95:
            add(
                "error",
                "thermal",
                g["id"],
                f"{g['id']} junction reached {tj['max']} C (>= 95)",
            )
        elif tj and tj["max"] >= 90:
            add(
                "warn",
                "thermal",
                g["id"],
                f"{g['id']} junction reached {tj['max']} C (>= 90)",
            )
        for ev in g.get("kfd_evicted", []):
            if ev["delta_ms"] > 0:
                add(
                    "warn",
                    "evicted_ms",
                    g["id"],
                    f"KFD queues of pid {ev.get('pid')} ({ev.get('comm')}) were "
                    f"evicted {ev['delta_ms']:g} ms on {g['id']}: a measured window "
                    "in this interval is INVALID (box rules)",
                )
    for name, c in rep["collector_cost"].items():
        if c["p95_ms"] > 5.0:
            add(
                "info",
                "collector_cost",
                name,
                f"collector {name}: p95 {c['p95_ms']} ms per collect() exceeds "
                "the 5 ms budget",
            )
    for ev in data.events:
        if ev.name == "collector_error":
            add("error", "collector_error", ev.entity, ev.message)
    for p in (
        rep["probes"].get("measurements", []) if rep["probes"].get("count") else []
    ):
        if not p["ok"]:
            add(
                "error",
                "probe_failed",
                p["src"],
                f"{p['kind']} {p['src']}->{p['dst']}: {p.get('error')}",
            )
    order = {"error": 0, "warn": 1, "info": 2}
    out.sort(key=lambda f: (order.get(f["severity"], 3), f["code"], f["entity"]))
    return out


# ---------------------------------------------------------------------------
# text rendering
# ---------------------------------------------------------------------------


def _fmt_size(n: int) -> str:
    """``4 B``, ``64 KiB``, ``256 MiB``: binary units when the size is exact."""
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div and n % div == 0:
            return f"{n // div} {unit}"
    return f"{n} B"


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [
        fmt.format(*header).rstrip(),
        fmt.format(*("-" * w for w in widths)).rstrip(),
    ]
    out += [fmt.format(*r).rstrip() for r in rows]
    return out


def _mm(d: dict[str, Any] | None) -> str:
    if not d:
        return "-"
    if d.get("min") == d.get("max"):
        return f"{d['max']:g}"
    return f"{d['min']:g}..{d['max']:g}"


def render_text(rep: dict[str, Any]) -> str:
    """Human-readable report."""
    s = rep["session"]
    out = [
        f"pciescope report: {s['name']}  host {s['host']}  {s['duration_s']} s  "
        f"status {s['status']}",
        f"collectors: {', '.join(s['collectors'] or [])}"
        + (f"  skipped: {s['skipped']}" if s.get("skipped") else ""),
        f"records: {s['counts']}",
        f"ceilings: Gen4 x16 raw {rep['ceilings']['gen4_x16_raw_gbs']} GB/s, "
        f"practical {rep['ceilings']['gen4_x16_practical_gbs']} GB/s "
        f"(MPS {rep['ceilings']['mps_bytes']} B)",
        "",
    ]
    out.append("FINDINGS")
    if rep["findings"]:
        out += [
            f"  [{f['severity']}] {f['code']}: {f['message']}" for f in rep["findings"]
        ]
    else:
        out.append(
            "  none (no degraded link, no AER increase, no thermal or eviction event)"
        )
    if rep.get("links"):
        out += ["", "LINKS (physical)"]
        rows = [
            [
                lk["id"],
                lk["kind"],
                lk["gpu"] or lk["switch"] or "",
                f"{lk['max_speed_gts']:g}x{lk['max_width']}"
                if lk["max_speed_gts"]
                else "?",
                _mm(lk["speed_gts"]),
                _mm(lk["width"]),
                "-"
                if lk["capacity_gbs_last"] is None
                else f"{lk['capacity_gbs_last']:g}",
                str(lk["link_changes"]),
                "-" if lk["degraded_s"] is None else f"{lk['degraded_s']:g}",
            ]
            for lk in rep["links"]
        ]
        out += _table(
            rows,
            [
                "link",
                "kind",
                "owner",
                "max",
                "GT/s",
                "width",
                "cap GB/s",
                "chg",
                "degr s",
            ],
        )
        traffic = [
            (lk["id"], t)
            for lk in rep["links"]
            for t in lk.get("traffic_counters") or []
        ]
        if traffic:
            traffic.sort(key=lambda x: -(x[1].get("util_raw_pct") or 0.0))

            def pct(row: dict[str, Any], key: str) -> str:
                return "-" if row.get(key) is None else f"{row[key]:g}%"

            out += [
                "",
                "TRAFFIC (switch counters: wire bytes vs raw capacity, "
                "payload estimate vs practical)",
            ]
            out += _table(
                [
                    [
                        lid,
                        t["port"],
                        t["direction"],
                        f"{t['mean_gbs']:g}",
                        f"{t['peak_gbs']:g}",
                        pct(t, "util_raw_pct"),
                        "-"
                        if t.get("payload_peak_gbs") is None
                        else f"{t['payload_peak_gbs']:g}",
                        pct(t, "util_practical_pct"),
                    ]
                    for lid, t in traffic[:30]
                ],
                [
                    "link",
                    "port",
                    "dir",
                    "mean GB/s",
                    "peak GB/s",
                    "of raw",
                    "payload peak",
                    "of practical",
                ],
            )
        else:
            out.append(
                "  (no traffic counters in this session: utilisation below comes from "
                "probe measurements only)"
            )
    if rep.get("gpus"):
        out += ["", "GPUS"]
        rows = []
        for g in rep["gpus"]:

            def mean_max(key: str, g: dict[str, Any] = g) -> str:
                d = g.get(key)
                return "-" if not d else f"{d['mean']:g}/{d['max']:g}"

            rows.append(
                [
                    g["id"],
                    str(g["hip"]),
                    g["arch"],
                    g["slot"] or "-",
                    mean_max("busy_pct"),
                    mean_max("temp_junction_c"),
                    mean_max("power_w"),
                    "-"
                    if "vram_used_bytes" not in g
                    else f"{g['vram_used_bytes']['max'] / 2**30:.2f}",
                    _mm(g.get("link_speed_gts")) + "x" + _mm(g.get("link_width")),
                ]
            )
        out += _table(
            rows,
            [
                "gpu",
                "hip",
                "arch",
                "slot",
                "busy% mean/max",
                "Tj C mean/max",
                "W mean/max",
                "VRAM GiB",
                "GPU-side link",
            ],
        )
    err = rep["errors"]
    out += ["", "ERRORS (AER)"]
    if err["increased_during_session"]:
        out += [
            f"  +{e['delta']:g} {e['class']}  {e['entity']}  {e['label']}"
            for e in err["increased_during_session"]
        ]
    else:
        out.append("  no counter increased during the session")
    if err["nonzero_totals_since_boot"]:
        out.append("  non-zero totals since boot:")
        for e in err["nonzero_totals_since_boot"]:
            kinds = f"  {e['types']}" if e.get("types") else ""
            out.append(
                f"    {e['entity']:<14} {e['class']:<20} {e['total']:g}{kinds}  "
                f"{e['label']}"
            )
    cost = rep["collector_cost"]
    if cost:
        out += ["", "COLLECTOR COST (per collect())"]
        out += _table(
            [
                [
                    n,
                    str(c["n"]),
                    f"{c['samples_per_tick']:g}" if "samples_per_tick" in c else "-",
                    f"{c['mean_ms']}",
                    f"{c['p95_ms']}",
                    f"{c['max_ms']}",
                ]
                for n, c in sorted(cost.items())
            ],
            ["collector", "calls", "samples", "mean ms", "p95 ms", "max ms"],
        )
    ev = rep["events"]
    if ev["notable"]:
        out += ["", f"EVENTS {ev['by_name']}"]
        out += [
            f"  +{e['t_s']:>8.2f}s [{e['severity']}] {e['name']} {e['entity']} "
            f"{e['message']}"
            for e in ev["notable"][:40]
        ]
    pr = rep["probes"]
    if pr.get("count"):
        out += ["", f"PROBES ({pr['count']} measurements)"]
        out += _table(
            [
                [
                    x["kind"],
                    x["direction"],
                    x["path_class"],
                    x["method"] or "-",
                    x["pattern"] or "-",
                    _fmt_size(x["size_bytes"]),
                    str(x["n"]),
                    f"{x['min']:g}",
                    f"{x['median']:g}",
                    f"{x['max']:g}",
                    x["unit"],
                ]
                for x in pr["summary"]
            ],
            [
                "kind",
                "direction",
                "path",
                "method",
                "pattern",
                "size",
                "n",
                "min",
                "median",
                "max",
                "unit",
            ],
        )
        out.append(
            "  (one row per kind/direction/path/method/pattern at its headline size: "
            "smallest for latency, largest otherwise; n = values at that size)"
        )
        if pr.get("matrix_gbs"):
            mx = pr["matrix_gbs"]
            out += [
                "",
                "  bandwidth matrix GB/s (src->dst, one flow, largest size, median; "
                "kernel-store and host links)",
            ]
            out += [
                "  " + ln
                for ln in _table(
                    [
                        [r] + [("-" if v is None else f"{v:g}") for v in row]
                        for r, row in zip(mx["rows"], mx["values"])
                    ],
                    ["src \\ dst", *mx["cols"]],
                )
            ]
        if pr.get("link_utilization"):
            out += [
                "",
                "  per-link utilisation attributed from probes (peak single flow)",
            ]
            out += [
                "  " + ln
                for ln in _table(
                    [
                        [
                            u["link"],
                            u["direction"],
                            f"{u['peak_gbs']:g}",
                            f"{u['raw_gbs']:g}",
                            f"{u['util_raw_pct']:g}%",
                            f"{u['util_practical_pct']:g}%",
                        ]
                        for u in pr["link_utilization"][:20]
                    ],
                    ["link", "dir", "peak GB/s", "raw GB/s", "of raw", "of practical"],
                )
            ]
    return "\n".join(out)


def gpu_of(topo: Topology, entity: str) -> Gpu | None:
    """Resolve a ``gpu:`` entity (helper for callers formatting reports)."""
    return topo.gpus.get(entity) if entity_kind(entity) == "gpu" else None


def is_counter(metric: str) -> bool:
    return metric_kind(metric) == "counter"
