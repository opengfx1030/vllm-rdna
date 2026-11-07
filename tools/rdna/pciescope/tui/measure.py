# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure panel controller: run a probe suite on leased GPUs, stream progress.

The TUI never touches a GPU itself.  It asks ``pciescope.probe`` which suites
exist, asks it whether the lease for the selected GPUs is held (the probe
refuses otherwise) and runs ``run_suite`` on a worker thread; progress events
and measurements come back through a queue that the app drains on its tick.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Sequence

from pciescope.model import Event, Measurement, Record, Topology
from pciescope.tui import fmt
from pciescope.tui.render import MeasureModel, entity_label


def _probe():
    """Import the probe lazily (the TUI must start without it).

    ``pciescope.probe.api`` is the public surface: ``list_suites``,
    ``check_leases`` and ``run_suite``.
    """
    from pciescope.probe import api

    return api


def default_owner() -> str:
    return os.environ.get("PCIESCOPE_OWNER", "")


NEVER_OFFERED = frozenset({"gpu:07"})  # W7800 bus 07 (HIP 5): lead only, never here


def gpu_presets(topo: Topology) -> list[tuple[str, list[str]]]:
    """GPU selections the user can cycle through (``g``); never HIP 5."""
    pre: list[tuple[str, list[str]]] = []
    for slot in ("A", "B"):
        ids = [g.id for g in topo.slot_gpus(slot) if g.id not in NEVER_OFFERED]
        if ids:
            pre.append((f"slot {slot}", ids))
    for g in topo.gpu_list():
        if not g.slot and g.id not in NEVER_OFFERED:
            pre.append((f"{entity_label(topo, g.id)} (no slot)", [g.id]))
    return pre


def progress_line(e: Event) -> str:
    """One log line for a probe progress event."""
    kind = e.name or str(e.data.get("kind", ""))
    d = e.data or {}
    i, n = d.get("i"), d.get("n")
    pos = f"[{i}/{n}] " if i is not None and n else ""
    base = {
        "plan": (
            f"plan: {n} steps" + (f", ~{d['est_s']:.0f}s" if d.get("est_s") else "")
        )
        if n
        else "plan",
        "step_start": f"{pos}start {d.get('scenario', d.get('step_id', ''))}",
        "step_done": f"{pos}done  {d.get('scenario', d.get('step_id', ''))}"
        + (f"  FAILED: {d['error']}" if d.get("ok") is False else "")
        + (f"  {t:.0f}C" if isinstance(t := d.get("temp_c"), int | float) else ""),
        "measurement": f"{pos}result {e.message}",
        "guard": f"guard: {e.message or ''} temp={d.get('temp_c', '?')}C",
        "abort": f"ABORT: {e.message}",
        "done": f"done: {e.message}",
    }.get(kind, f"{kind} {e.message}".strip())
    return f"{fmt.clock(e.ts_ns)} {base}"


class MeasureRunner:
    """State machine behind the Measure view."""

    def __init__(
        self,
        topo: Topology,
        *,
        owner: str = "",
        sink: Callable[[Record], None] | None = None,
        dry_run: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.topo = topo
        self.owner = owner or default_owner()
        self.sink = sink
        self.dry_run = dry_run
        self.model = MeasureModel(owner=self.owner)
        self._clock = clock
        self._q: queue.Queue = queue.Queue()
        self._published: set[tuple] = set()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.presets = gpu_presets(topo)
        self.preset_idx = 0
        self.available = True
        try:
            suites = _probe().list_suites()
            self.model.suites = list(suites)
        except Exception as exc:  # probe missing/broken: the panel says so
            self.available = False
            self.model.error = f"probe unavailable: {type(exc).__name__}: {exc}"
        self._select_preset(0)

    # selection -------------------------------------------------------------
    def move(self, step: int) -> None:
        if self.model.suites and self.model.status != "running":
            self.model.sel = (self.model.sel + step) % len(self.model.suites)

    def cycle_gpus(self) -> None:
        if self.presets and self.model.status != "running":
            self._select_preset((self.preset_idx + 1) % len(self.presets))

    def _select_preset(self, idx: int) -> None:
        if not self.presets:
            return
        self.preset_idx = idx
        self.model.gpus = list(self.presets[idx][1])
        self.refresh_lease()

    def refresh_lease(self) -> None:
        """Ask the probe whether the selected GPUs are leased to us."""
        m = self.model
        m.lease_ok, m.lease_reason, m.lease_needed = False, "", []
        if not self.available:
            m.lease_reason = "probe unavailable"
            return
        try:
            chk = _probe().check_leases(
                [g.split(":", 1)[-1] for g in m.gpus], owner=self.owner or None
            )
        except Exception as exc:
            m.lease_reason = f"lease check failed: {type(exc).__name__}: {exc}"
            return
        m.lease_ok = bool(chk.ok)
        m.lease_reason = "" if chk.ok else (chk.reason or "lease not held")
        m.lease_needed = [] if chk.ok else list(chk.needed)

    # run -------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.model.status == "running"

    def start(self) -> str | None:
        """Start the selected suite; returns an error string when refused."""
        m = self.model
        if self.running:
            return "already running"
        if not m.suites:
            return m.error or "no suites"
        self.refresh_lease()
        if not m.lease_ok and not self.dry_run:
            return f"refused: {m.lease_reason}"
        suite = m.suites[m.sel]
        self._cancel.clear()
        m.status, m.i, m.n, m.error = "running", 0, 0, ""
        m.est_s = float(suite.est_seconds)
        m.started = self._clock()
        m.log = [
            f"{fmt.clock(time.time_ns())} starting {suite.name} on "
            + " ".join(entity_label(self.topo, g) for g in m.gpus)
        ]
        self._thread = threading.Thread(
            target=self._work,
            args=(suite.name, list(m.gpus)),
            daemon=True,
            name="tui-measure",
        )
        self._thread.start()
        return None

    def cancel(self) -> None:
        if self.running:
            self._cancel.set()
            self.model.log.append("cancel requested (stops after the current cell)")

    def _work(self, suite: str, gpus: Sequence[str]) -> None:
        def progress(e: Event) -> None:
            self._q.put(("event", e))

        def sink(rec: Record) -> None:
            self._q.put(("record", rec))

        try:
            res = _probe().run_suite(
                suite,
                gpus,
                owner=self.owner or None,
                topo=self.topo,
                progress=progress,
                cancel=self._cancel,
                dry_run=self.dry_run,
                sink=sink,
            )
            self._q.put(("result", res))
        except Exception as exc:
            self._q.put(("error", f"{type(exc).__name__}: {exc}"))

    def _fresh(self, meas: Measurement) -> bool:
        """True the first time a measurement is seen (probe sink + final result)."""
        key = (
            meas.kind,
            meas.src,
            meas.dst,
            meas.direction,
            meas.size_bytes,
            meas.started_ns,
            meas.value,
        )
        if key in self._published:
            return False
        self._published.add(key)
        return True

    def drain(self) -> list[Record]:
        """Apply queued progress to the model; return records for the state."""
        m = self.model
        records: list[Record] = []
        fresh = self._fresh
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == "event":
                e: Event = payload
                m.log.append(progress_line(e))
                d = e.data or {}
                if d.get("n"):
                    m.n = int(d["n"])
                if d.get("i") is not None:
                    m.i = (
                        int(d["i"]) if (e.name or d.get("kind")) == "step_done" else m.i
                    )
                if (e.name or d.get("kind")) == "plan" and d.get("est_s"):
                    m.est_s = float(d["est_s"])
                # UI progress only feeds the log; the probe's sink records the data
            elif kind == "record":
                if not isinstance(payload, Measurement) or fresh(payload):
                    records.append(payload)
            elif kind == "result":
                res = payload
                for meas in res.measurements:
                    if isinstance(meas, Measurement) and fresh(meas):
                        records.append(meas)
                m.status = {"ok": "done", "dry-run": "done", "partial": "done"}.get(
                    res.status, res.status
                )
                why = {
                    "refused": res.refusal,
                    "failed": getattr(res, "error", ""),
                    "aborted": getattr(res, "abort_reason", ""),
                    "partial": "some steps failed",
                }.get(res.status, "")
                if why:
                    m.error = why or res.status
                for line in getattr(res, "blockers", None) or ():
                    m.log.append(f"a real run would be blocked: {line}")
                if getattr(res, "asleep", None):
                    m.log.append(
                        "asleep (dry run did not wake): " + " ".join(res.asleep)
                    )
                if getattr(res, "valid", True) is False:
                    m.error = "INVALID: " + "; ".join(res.invalid_reasons)
                m.log.append(f"finished: {res.status}, {len(res.measurements)} results")
                m.i = m.n
            elif kind == "error":
                m.status, m.error = "failed", payload
                m.log.append(f"error: {payload}")
        if self.sink is not None:
            for r in records:
                self.sink(r)
        return records

    def elapsed(self) -> float | None:
        if self.running and self.model.started is not None:
            return self._clock() - self.model.started
        return None
