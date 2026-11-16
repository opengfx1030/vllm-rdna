# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export a session to the Chrome / Perfetto trace-event JSON format.

Open the result in https://ui.perfetto.dev or ``chrome://tracing``.  Layout:

* process "GPUs": one counter track per ``gpu.*`` / ``kfd.*`` metric and GPU;
* process "Links": ``link.*`` counters per link (speed, width, capacity, degraded);
* process "Ports": ``pcie.*`` and ``aer.*`` counters per port;
* process "Host": collector cost counters and session/watch markers;
* process "Probe": one slice (``ph: X``) per ``Measurement`` (thread per kind),
  with src, dst, size, value, efficiency and path class in ``args``;
* every ``Event`` is an instant event on the track of its entity's process.

Timestamps are microseconds relative to the first record;
``metadata.t0_epoch_ns`` is the absolute origin.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pciescope.model import Event, Measurement, Record, Sample, Topology, entity_kind
from pciescope.store import Session

PID_HOST, PID_GPU, PID_LINK, PID_PORT, PID_PROBE = 1, 2, 3, 4, 5
_PROCESS_NAMES = {
    PID_HOST: "Host",
    PID_GPU: "GPUs",
    PID_LINK: "Links",
    PID_PORT: "Ports",
    PID_PROBE: "Probe",
}


def _pid_of(entity: str, metric: str = "") -> int:
    kind = entity_kind(entity)
    if kind == "gpu":
        return PID_GPU
    if kind == "link":
        return PID_LINK
    if kind in ("port", "switch", "root"):
        return PID_PORT
    return PID_HOST


def _series(s: Sample) -> str:
    if not s.tags:
        return s.entity
    tags = ",".join(f"{k}={v}" for k, v in sorted(s.tags.items()))
    return f"{s.entity}{{{tags}}}"


def _us(ts_ns: int, t0: int) -> float:
    return round((ts_ns - t0) / 1000.0, 3)


def to_chrome(
    records: Iterable[Record],
    topo: Topology | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the trace-event document for ``records``."""
    recs = list(records)
    stamps = [r.started_ns if isinstance(r, Measurement) else r.ts_ns for r in recs]
    t0 = min(stamps) if stamps else 0
    events: list[dict[str, Any]] = []
    for pid, name in _PROCESS_NAMES.items():
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": name},
            }
        )
        events.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": pid},
            }
        )
    counters: dict[tuple[int, str, int], dict[str, float]] = {}
    probe_threads: dict[str, int] = {}
    for rec in recs:
        if isinstance(rec, Sample):
            key = (_pid_of(rec.entity, rec.metric), rec.metric, rec.ts_ns)
            counters.setdefault(key, {})[_series(rec)] = rec.value
        elif isinstance(rec, Measurement):
            tid = probe_threads.setdefault(rec.kind, len(probe_threads) + 1)
            dur = max(rec.ended_ns - rec.started_ns, 1000)
            events.append(
                {
                    "ph": "X",
                    "cat": "measurement",
                    "name": (
                        f"{rec.kind} {rec.src}->{rec.dst} {rec.value:.3g} {rec.unit}"
                    ),
                    "pid": PID_PROBE,
                    "tid": tid,
                    "ts": _us(rec.started_ns, t0),
                    "dur": round(dur / 1000.0, 3),
                    "args": rec.to_dict(),
                }
            )
        elif isinstance(rec, Event):
            events.append(
                {
                    "ph": "i",
                    "s": "p",
                    "cat": rec.severity,
                    "name": f"{rec.name} {rec.entity}",
                    "pid": _pid_of(rec.entity),
                    "tid": 0,
                    "ts": _us(rec.ts_ns, t0),
                    "args": {
                        "message": rec.message,
                        "severity": rec.severity,
                        **rec.data,
                    },
                }
            )
    for kind, tid in probe_threads.items():
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": PID_PROBE,
                "tid": tid,
                "args": {"name": kind},
            }
        )
    for (pid, metric, ts), args in sorted(counters.items(), key=lambda kv: kv[0][2]):
        events.append(
            {
                "ph": "C",
                "name": metric,
                "pid": pid,
                "tid": 0,
                "ts": _us(ts, t0),
                "args": args,
            }
        )
    md: dict[str, Any] = {"t0_epoch_ns": t0, "tool": "pciescope"}
    if topo is not None:
        md["host"] = topo.host
        md["gpus"] = {
            g.id: {"hip": g.hip, "name": g.name, "switch": g.switch}
            for g in topo.gpu_list()
        }
    if meta:
        md.update(meta)
    return {"traceEvents": events, "displayTimeUnit": "ms", "metadata": md}


def export_chrome(
    session: Session | str | os.PathLike[str], out: str | os.PathLike[str] | None = None
) -> Path:
    """Write ``<session>/trace.perfetto.json`` (or ``out``) and return its path."""
    sess = session if isinstance(session, Session) else Session.open(session)
    doc = to_chrome(
        sess.records(),
        sess.topology(),
        meta={"session": sess.name, "label": sess.meta.get("label")},
    )
    dest = Path(out) if out else sess.path / "trace.perfetto.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"), allow_nan=False)
    return dest
