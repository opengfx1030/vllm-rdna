# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session store: one directory per run with an append-only JSONL trace.

Layout under ``~/w4a8_runs/pciescope/`` (override with ``$PCIESCOPE_HOME``)::

    <YYYYmmdd-HHMMSS>-<label>/
        session.json   metadata (host, interval, collectors, status, counts)
        topology.json  the discovered fabric at session start (Topology.to_dict)
        trace.jsonl    one Sample / Measurement / Event per line (append-only)

``Session.create`` opens a new session for writing, ``Session.open`` reads one
(also while another process is still writing it: a truncated last line is
skipped).  ``append`` is thread-safe and line-buffered per call, so a crash loses
at most the record being written.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pciescope.model import (
    SCHEMA_VERSION,
    TOOL_VERSION,
    Event,
    Measurement,
    Record,
    Sample,
    Topology,
    now_ns,
    record_from_dict,
)

ENV_HOME = "PCIESCOPE_HOME"
TRACE_FILE = "trace.jsonl"
META_FILE = "session.json"
TOPO_FILE = "topology.json"


def default_root() -> Path:
    """``$PCIESCOPE_HOME`` or ``~/w4a8_runs/pciescope``."""
    env = os.environ.get(ENV_HOME)
    return Path(env) if env else Path.home() / "w4a8_runs" / "pciescope"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def _is_finite(rec: Record) -> bool:
    if isinstance(rec, Sample):
        return math.isfinite(rec.value)
    if isinstance(rec, Measurement):
        vals = [rec.value, *rec.stats.values()]
        vals += [v for v in (rec.theoretical, rec.efficiency) if v is not None]
        return all(math.isfinite(v) for v in vals)
    return True


class Session:
    """One recorded run (writer or reader)."""

    def __init__(self, path: Path, meta: dict[str, Any], *, writable: bool) -> None:
        self.path = path
        self.meta = meta
        self.writable = writable
        self.dropped = 0
        self._lock = threading.Lock()
        self._fh: Any = None
        self._counts = {"sample": 0, "measurement": 0, "event": 0}
        if writable:
            self._fh = open(  # noqa: SIM115 - kept open for the session lifetime
                path / TRACE_FILE, "a", encoding="utf-8", buffering=1
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str] | None = None,
        label: str = "session",
        *,
        topo: Topology | None = None,
        collectors: Iterable[str] = (),
        skipped: dict[str, str] | None = None,
        interval_s: float | None = None,
        cmdline: str = "",
        name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session directory and open it for writing."""
        base = Path(root) if root else default_root()
        base.mkdir(parents=True, exist_ok=True)
        stem = name or f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(label)}"
        path, n = base / stem, 1
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError:
                n += 1
                path = base / f"{stem}-{n}"
        meta: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "name": path.name,
            "label": label,
            "host": topo.host if topo else "",
            "created_ns": now_ns(),
            "ended_ns": None,
            "status": "running",
            "tool_version": TOOL_VERSION,
            "pid": os.getpid(),
            "cmdline": cmdline,
            "interval_s": interval_s,
            "collectors": list(collectors),
            "skipped": dict(skipped or {}),
        }
        if extra:
            meta.update(extra)
        sess = cls(path, meta, writable=True)
        sess._write_meta()
        if topo is not None:
            (path / TOPO_FILE).write_text(json.dumps(topo.to_dict(), indent=1) + "\n")
        return sess

    @classmethod
    def open(
        cls, ref: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
    ) -> Session:
        """Open an existing session read-only.

        ``ref`` is a directory, a session name under ``root`` or ``latest``.
        """
        path = resolve(ref, root)
        meta_path = path / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("name", path.name)
        return cls(path, meta, writable=False)

    # -- writing -----------------------------------------------------------

    def append(self, rec: Record) -> None:
        """Append one record (thread-safe, one line)."""
        self.append_many((rec,))

    def append_many(self, recs: Iterable[Record]) -> None:
        """Append records as one batch (one flush)."""
        if self._fh is None:
            raise ValueError("session is not open for writing")
        lines = []
        kinds = {"sample": 0, "measurement": 0, "event": 0}
        for rec in recs:
            if not _is_finite(rec):
                self.dropped += 1
                continue
            d = rec.to_dict()
            lines.append(_dumps(d))
            kinds[d["type"]] += 1
        if not lines:
            return
        blob = "\n".join(lines) + "\n"
        with self._lock:
            self._fh.write(blob)
            for k, v in kinds.items():
                self._counts[k] += v

    def flush(self) -> None:
        if self._fh is not None:
            with self._lock:
                self._fh.flush()

    def close(self, status: str = "ok") -> None:
        """Close the trace and finalise ``session.json`` (idempotent)."""
        if self._fh is None:
            return
        with self._lock:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        self.meta["ended_ns"] = now_ns()
        self.meta["status"] = status
        self.meta["counts"] = dict(self._counts)
        self.meta["dropped"] = self.dropped
        self._write_meta()

    def update_meta(self, **kv: Any) -> None:
        self.meta.update(kv)
        if self.writable:
            self._write_meta()

    def _write_meta(self) -> None:
        tmp = self.path / (META_FILE + ".tmp")
        tmp.write_text(json.dumps(self.meta, indent=1) + "\n")
        tmp.replace(self.path / META_FILE)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close("error" if exc_type else "ok")

    # -- reading -----------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self.meta.get("name", self.path.name))

    def topology(self) -> Topology | None:
        p = self.path / TOPO_FILE
        if not p.exists():
            return None
        return Topology.from_dict(json.loads(p.read_text()))

    def _scan(self, offset: int = 0) -> Iterator[tuple[Record | None, int]]:
        """Yield ``(record or None, offset after the line)`` per complete line."""
        p = self.path / TRACE_FILE
        if not p.exists():
            return
        with open(p, "rb") as fh:
            fh.seek(offset)
            pos = offset
            for raw in fh:
                if not raw.endswith(b"\n"):
                    return  # a line another process is still writing
                pos += len(raw)
                if not raw.strip():
                    yield None, pos
                    continue
                try:
                    rec = record_from_dict(json.loads(raw))
                except (ValueError, KeyError, TypeError):
                    rec = None
                yield rec, pos

    def records(
        self,
        types: Iterable[str] | None = None,
        *,
        since_ns: int | None = None,
        until_ns: int | None = None,
    ) -> Iterator[Record]:
        """Stream records in file order (``types``: sample/measurement/event)."""
        want = set(types) if types is not None else None
        for rec, _pos in self._scan(0):
            if rec is None:
                continue
            if want is not None and _kind(rec) not in want:
                continue
            ts = _ts(rec)
            if since_ns is not None and ts < since_ns:
                continue
            if until_ns is not None and ts > until_ns:
                continue
            yield rec

    def read_from(self, offset: int = 0) -> tuple[list[Record], int]:
        """Records starting at byte ``offset``; returns (records, next offset).

        Only complete lines are consumed, so polling a trace that another process
        is appending to never yields a half-written record.
        """
        out: list[Record] = []
        pos = offset
        for rec, pos in self._scan(offset):
            if rec is not None:
                out.append(rec)
        return out, pos

    def samples(self, **kw: Any) -> Iterator[Sample]:
        for rec in self.records(["sample"], **kw):
            if isinstance(rec, Sample):
                yield rec

    def measurements(self) -> list[Measurement]:
        return [r for r in self.records(["measurement"]) if isinstance(r, Measurement)]

    def events(self) -> list[Event]:
        return [r for r in self.records(["event"]) if isinstance(r, Event)]


def _kind(rec: Record) -> str:
    if isinstance(rec, Sample):
        return "sample"
    if isinstance(rec, Measurement):
        return "measurement"
    return "event"


def _ts(rec: Record) -> int:
    if isinstance(rec, Measurement):
        return rec.started_ns
    return rec.ts_ns


def _slug(label: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "-" for c in label.strip()]
    return "".join(keep).strip("-") or "session"


def resolve(
    ref: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
) -> Path:
    """Resolve a session reference to its directory."""
    base = Path(root) if root else default_root()
    s = os.fspath(ref)
    if s == "latest":
        found = list_sessions(base)
        if not found:
            raise FileNotFoundError(f"no session under {base}")
        return found[-1]
    p = Path(s).expanduser()
    if (p / TRACE_FILE).exists() or (p / META_FILE).exists():
        return p
    if (base / s).is_dir():
        return base / s
    raise FileNotFoundError(f"session not found: {ref} (looked in {base})")


def list_sessions(root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Session directories under ``root``, oldest first (by name)."""
    base = Path(root) if root else default_root()
    if not base.is_dir():
        return []
    return sorted(
        p
        for p in base.iterdir()
        if p.is_dir() and ((p / META_FILE).exists() or (p / TRACE_FILE).exists())
    )


def describe(path: Path) -> dict[str, Any]:
    """One-line description of a session directory (for ``sessions`` listings)."""
    try:
        meta = json.loads((path / META_FILE).read_text())
    except (OSError, ValueError):
        meta = {}
    trace = path / TRACE_FILE
    return {
        "name": path.name,
        "path": str(path),
        "label": meta.get("label"),
        "status": meta.get("status"),
        "created_ns": meta.get("created_ns"),
        "ended_ns": meta.get("ended_ns"),
        "interval_s": meta.get("interval_s"),
        "collectors": meta.get("collectors"),
        "trace_bytes": trace.stat().st_size if trace.exists() else 0,
    }
