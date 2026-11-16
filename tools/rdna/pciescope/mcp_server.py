# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MCP stdio server: the CLI verbs as tools for agents.

Start it with ``python -m pciescope mcp`` (on the box:
``~/Apps/pciescope-venv/bin/python -m pciescope mcp``; from a laptop:
``ssh par1-cs25 ~/Apps/pciescope-venv/bin/python -m pciescope mcp``).  Needs the
``mcp`` package (2.x: ``MCPServer``; 1.x ``FastMCP`` also works) in the
environment.  Tools: ``topology`` ``snapshot`` ``start_watch`` ``stop_watch``
``watch_status`` ``measure`` ``report`` ``query_trace`` ``sessions`` ``metrics``.

``Service`` holds the logic as plain methods returning JSON-able dicts (unit
tested without MCP); ``build_server`` wires them as tools.  Nothing but the MCP
protocol is written to stdout.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import logging
import threading
from typing import Any

from pciescope import model, report, runner, store, topology, trace, views
from pciescope.env import Env, close_collectors, parse_names
from pciescope.report import tw_stats

log = logging.getLogger("pciescope.mcp")

#: the server never starts GPU work unless the caller says ``dry_run=false``
DEFAULT_MEASURE_DRY_RUN = True

#: exit codes of ``pciescope.probe.cli_run`` (``MeasureResult.exit_code``)
MEASURE_EXIT = {
    0: "ok or dry-run",
    1: "failed or partial",
    2: "refused (lease, forbidden GPU, noisy box, VRAM, bad suite)",
    3: "aborted (guard, time limit or cancel); results so far are kept",
    4: "finished but INVALID (noisy window forced or evicted_ms jump)",
}


class Service:
    """Tool implementations (state: cached topology, running watches)."""

    def __init__(self, env: Env | None = None, *, max_watches: int = 4) -> None:
        self.env = env or Env()
        self.max_watches = max_watches
        self._topo: model.Topology | None = None
        self._watches: dict[str, runner.WatchHandle] = {}
        self._closers: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    # -- topology ----------------------------------------------------------

    def topo(self, refresh: bool = False) -> model.Topology:
        with self._lock:
            if self._topo is None or refresh:
                self._topo = self.env.discover()
            return self._topo

    def topology(self, view: str = "summary", refresh: bool = False) -> dict[str, Any]:
        """``summary``: tree text + GPU table; ``full``: the whole Topology document."""
        topo = self.topo(refresh)
        if not topo.gpus:
            raise ValueError(f"no AMD GPU found under {self.env.sysfs_root}")
        if view == "full":
            return {"topology": topo.to_dict()}
        if view != "summary":
            raise ValueError("view must be 'summary' or 'full'")
        return {
            "tree": topology.render_tree(topo),
            "summary": topology.summary(topo),
            "hint": "view='full' returns every node and link; entity ids are "
            "gpu:<bus>, port:<bdf>, link:<up>-<down>, switch:<A|B>",
        }

    # -- snapshot ----------------------------------------------------------

    def snapshot(
        self,
        collectors: str | None = None,
        raw: bool = False,
        metric: list[str] | None = None,
        entity: list[str] | None = None,
        allow_switch_writes: bool = False,
    ) -> dict[str, Any]:
        """One pass of the collectors: per-GPU, per-link and AER tables."""
        topo = self.topo()
        res = self.env.build(
            topo, parse_names(collectors), allow_switch_writes=allow_switch_writes
        )
        if not res.collectors:
            raise ValueError(f"no collector available: {dict(res.skipped)}")
        try:
            tick = runner.snapshot(topo, res.collectors)
        finally:
            close_collectors(res.collectors)
        doc = views.snapshot_summary(
            topo,
            tick.samples,
            costs_ms=tick.costs_ms,
            collectors=[c.name for c in res.collectors],
            skipped=dict(res.skipped),
            ts_ns=tick.ts_ns,
        )
        doc["events"] = [e.to_dict() for e in tick.events]
        if raw or metric or entity:
            want_m, want_e = metric or [], entity or []
            doc["samples"] = [
                s.to_dict()
                for s in tick.samples
                if (not want_m or any(s.metric.startswith(p) for p in want_m))
                and (not want_e or any(s.entity.startswith(p) for p in want_e))
            ]
        return doc

    # -- watches -----------------------------------------------------------

    def start_watch(
        self,
        interval_s: float = 1.0,
        duration_s: float | None = None,
        collectors: str | None = None,
        label: str = "mcp-watch",
        allow_switch_writes: bool = False,
        heartbeat_s: float = 30.0,
    ) -> dict[str, Any]:
        """Record a session in the background (``stop_watch`` ends it)."""
        if interval_s < 0.05:
            raise ValueError("interval_s must be >= 0.05")
        with self._lock:
            live = [w for w in self._watches.values() if w.running]
            if len(live) >= self.max_watches:
                raise ValueError(
                    f"{len(live)} watches already running (max {self.max_watches}); "
                    "stop one with stop_watch"
                )
        topo = self.topo()
        res = self.env.build(
            topo, parse_names(collectors), allow_switch_writes=allow_switch_writes
        )
        if not res.collectors:
            raise ValueError(f"no collector available: {dict(res.skipped)}")
        names = [c.name for c in res.collectors]
        sess = store.Session.create(
            self.env.session_root(),
            label,
            topo=topo,
            collectors=names,
            skipped=dict(res.skipped),
            interval_s=interval_s,
            cmdline="mcp start_watch",
        )
        handle = runner.start_watch(
            topo,
            res.collectors,
            sess,
            interval_s=interval_s,
            duration_s=duration_s,
            dedup=heartbeat_s > 0,
            heartbeat_s=heartbeat_s if heartbeat_s > 0 else 30.0,
        )
        handle.extras.update(collectors=names, skipped=dict(res.skipped))
        with self._lock:
            self._watches[sess.name] = handle
            self._closers[sess.name] = list(res.collectors)
        return {
            "watch_id": sess.name,
            "session": sess.name,
            "path": str(sess.path),
            "interval_s": interval_s,
            "duration_s": duration_s,
            "collectors": names,
            "skipped": dict(res.skipped),
            "next": "call stop_watch(watch_id) then report(session)",
        }

    def watch_status(self) -> dict[str, Any]:
        with self._lock:
            return {"watches": [w.status() for w in self._watches.values()]}

    def stop_watch(self, watch_id: str | None = None) -> dict[str, Any]:
        """Stop a watch (the only running one when ``watch_id`` is omitted)."""
        with self._lock:
            if watch_id is None:
                live = [k for k, w in self._watches.items() if w.running]
                if len(live) != 1:
                    raise ValueError(
                        f"{len(live)} watches running ({live}); pass watch_id"
                    )
                watch_id = live[0]
            handle = self._watches.get(watch_id)
            if handle is None:
                raise ValueError(f"unknown watch_id {watch_id!r}")
            closers = self._closers.pop(watch_id, [])
        out = handle.stop()
        close_collectors(closers)
        with self._lock:
            self._watches.pop(watch_id, None)
        rep = report.build_report(store.Session.open(handle.session.path))
        out["findings"] = rep["findings"]
        return out

    # -- measure -----------------------------------------------------------

    def measure(
        self,
        suite: str = "quick",
        gpus: str | list[str] | None = None,
        *,
        owner: str | None = None,
        dry_run: bool = DEFAULT_MEASURE_DRY_RUN,
        repeats: int | None = None,
        max_seconds: float | None = None,
        force_noisy: bool = False,
        wait_quiet_s: float | None = None,
        in_situ_s: float | None = None,
        interval_s: float = 1.0,
        list_suites: bool = False,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Run the active probe (``pciescope.probe``) like ``pciescope measure --json``.

        Same flags, defaults and exit codes as the CLI; the probe refuses GPUs
        whose lease the caller does not hold.  A dry run records nothing.
        """
        try:
            from pciescope import probe
        except Exception as exc:  # the probe slice is optional
            raise ValueError(f"the probe module is not available: {exc}") from exc
        parser = argparse.ArgumentParser(prog="measure", add_help=False)
        probe.add_cli_arguments(parser)
        argv = ["--suite", suite, "--quiet"]
        tokens = gpus.replace(",", " ").split() if isinstance(gpus, str) else gpus
        if tokens:
            argv += ["--gpus", *tokens]
        for flag, val in (
            ("--owner", owner),
            ("--repeats", repeats),
            ("--max-seconds", max_seconds),
            ("--wait-quiet", wait_quiet_s),
            ("--in-situ", in_situ_s),
        ):
            if val is not None:
                argv += [flag, str(val)]
        if in_situ_s is not None:
            argv += ["--interval", str(interval_s)]
        for flag, on in (
            ("--dry-run", dry_run),
            ("--force-noisy", force_noisy),
            ("--list-suites", list_suites),
        ):
            if on:
                argv.append(flag)
        ns = parser.parse_args(argv)
        ns.json = True
        topo = self.topo()
        records = in_situ_s is not None or not (dry_run or list_suites)
        sess: store.Session | store.LazySession | None = None
        if records:
            root = self.env.session_root()
            label = f"measure-{suite}"
            if session:
                sess = store.Session.attach(session, root)
            elif in_situ_s is not None:  # nothing to be refused
                sess = store.Session.create(
                    root, label, topo=topo, cmdline="mcp measure"
                )
            else:  # created by the first record of a run that really starts
                sess = store.LazySession(root, label, topo=topo, cmdline="mcp measure")
        lines: list[str] = []
        status = "ok"
        code = 1
        try:
            code = int(
                probe.cli_run(
                    ns,
                    topo=topo,
                    session=sess,
                    emit=lambda *a: lines.append(" ".join(map(str, a))),
                )
                or 0
            )
            if code:
                status = f"exit {code}"
        finally:
            if sess is not None:
                sess.close(status)
        text = "\n".join(lines)
        try:
            out = json.loads(text)
            if not isinstance(out, dict):
                out = {"result": out}
        except ValueError:
            out = {"output": text}
        out["exit_code"] = code
        out["exit_meaning"] = MEASURE_EXIT.get(code, "")
        if sess is not None and getattr(sess, "created", True):
            out["session"] = sess.name
            out["path"] = str(sess.path)
        return out

    # -- sessions ----------------------------------------------------------

    def sessions(self, limit: int = 20) -> dict[str, Any]:
        root = self.env.session_root()
        items = [store.describe(p) for p in reversed(store.list_sessions(root))]
        return {"root": str(root), "sessions": items[:limit] if limit else items}

    def _open(self, ref: str) -> store.Session:
        # a running watch writes the same files: reading it is always safe
        return store.Session.open(ref, self.env.session_root())

    def report(
        self,
        session: str = "latest",
        sections: list[str] | None = None,
        format: str = "json",
        include_virtual: bool = False,
    ) -> dict[str, Any]:
        """Summarise a session (links, GPUs, AER, events, probes, findings)."""
        sess = self._open(session)
        rep = report.build_report(sess, include_virtual=include_virtual)
        if format == "text":
            return {"session": sess.name, "text": report.render_text(rep)}
        if sections:
            unknown = [s for s in sections if s not in rep]
            if unknown:
                raise ValueError(f"unknown sections {unknown}; have {sorted(rep)}")
            rep = {k: v for k, v in rep.items() if k in sections or k == "session"}
        return rep

    def query_trace(
        self,
        session: str = "latest",
        kind: str = "samples",
        metric: str | None = None,
        entity: str | None = None,
        since_s: float | None = None,
        until_s: float | None = None,
        severity: str | None = None,
        limit: int = 200,
        tail: bool = False,
    ) -> dict[str, Any]:
        """Query records of a session.

        ``kind``: ``samples`` (raw points), ``stats`` (per series first/last/min/
        max/time-weighted mean), ``series`` (names and counts), ``events``,
        ``measurements``.  ``metric`` and ``entity`` are prefix filters,
        ``since_s``/``until_s`` seconds after the session start.
        """
        if kind not in ("samples", "stats", "series", "events", "measurements"):
            raise ValueError("kind must be samples|stats|series|events|measurements")
        sess = self._open(session)
        origin = int(sess.meta.get("created_ns") or 0)

        def rel(ts: int) -> float:
            return round((ts - origin) / 1e9, 3)

        lo = None if since_s is None else origin + int(since_s * 1e9)
        hi = None if until_s is None else origin + int(until_s * 1e9)
        types = {
            "samples": ["sample"],
            "stats": ["sample"],
            "series": ["sample"],
            "events": ["event"],
            "measurements": ["measurement"],
        }[kind]
        keep: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=limit
            if tail and kind in ("samples", "events", "measurements")
            else None
        )
        n_match = 0
        series: dict[Any, list[tuple[int, float]]] = {}
        last_ts = 0
        for rec in sess.records(types, since_ns=lo, until_ns=hi):
            ts = rec.started_ns if isinstance(rec, model.Measurement) else rec.ts_ns

            last_ts = max(last_ts, ts)
            if isinstance(rec, model.Sample):
                if metric and not rec.metric.startswith(metric):
                    continue
                if entity and not rec.entity.startswith(entity):
                    continue
                if kind in ("stats", "series"):
                    series.setdefault(rec.key(), []).append((rec.ts_ns, rec.value))
                    continue
            elif isinstance(rec, model.Event):
                if severity and rec.severity != severity:
                    continue
                if metric and not rec.name.startswith(metric):
                    continue
                if entity and not rec.entity.startswith(entity):
                    continue
            elif isinstance(rec, model.Measurement):
                if metric and not rec.kind.startswith(metric):
                    continue
                if entity and entity not in (rec.src, rec.dst, *rec.gpus_used):
                    continue
            n_match += 1
            if not tail and len(keep) >= limit:
                continue
            d = rec.to_dict()
            d["t_s"] = rel(ts)
            keep.append(d)
        out: dict[str, Any] = {
            "session": sess.name,
            "kind": kind,
            "origin_ns": origin,
            "span_s": rel(last_ts) if last_ts else 0.0,
        }
        if kind in ("stats", "series"):
            rows = []
            for (ent, met, tags), pts in sorted(series.items()):
                row: dict[str, Any] = {"entity": ent, "metric": met}
                if tags:
                    row["tags"] = dict(tags)
                if kind == "series":
                    row["n"] = len(pts)
                else:
                    st = tw_stats(pts, last_ts)
                    row.update({k: round(v, 4) for k, v in st.items()})
                rows.append(row)
            out["count"] = len(rows)
            out["truncated"] = len(rows) > limit
            out["series"] = rows[:limit]
            return out
        out["count"] = n_match
        out["truncated"] = n_match > len(keep)
        out["records"] = list(keep)
        return out

    def metrics(self) -> dict[str, Any]:
        return {
            "metrics": [
                {
                    "name": m.name,
                    "unit": m.unit,
                    "kind": m.kind,
                    "entity": m.entity,
                    "source": m.source,
                    "doc": m.doc,
                }
                for m in model.METRICS.values()
            ]
        }

    def export_trace(
        self, session: str = "latest", out: str | None = None
    ) -> dict[str, Any]:
        sess = self._open(session)
        path = trace.export_chrome(sess, out)
        return {"session": sess.name, "path": str(path), "bytes": path.stat().st_size}

    # -- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        """Stop every running watch (called when the client disconnects)."""
        with self._lock:
            ids = [k for k, w in self._watches.items() if w.running]
        for wid in ids:
            with contextlib.suppress(Exception):
                self.stop_watch(wid)


INSTRUCTIONS = """\
pciescope reads the PCIe fabric of the box (2 Broadcom PEX88096 switches, 10 GPUs).
Start with topology (entity ids: gpu:<bus>, port:<bdf>, link:<up>-<down>, switch:A|B).
snapshot = one passive sample (GPU load/temperature, link speed/width, AER error
counters, KFD evictions). start_watch/stop_watch record a session; report summarises it
(degraded links, AER increases, thermals, evictions, probe efficiency vs the raw and the
practical ceiling of 28.0 GB/s per direction for Gen4 x16); query_trace reads series.
measure runs the active latency/bandwidth probes: it only touches GPUs whose lease the
caller holds (owner=...) and is a dry run unless dry_run=false."""


class _Registrar:
    """``srv.tool`` that turns the service's expected failures into ``ToolError``
    (the SDK only forwards the message of that exception type to the client).
    """

    def __init__(self, srv: Any, tool_error: type[Exception]) -> None:
        self.srv = srv
        self.tool_error = tool_error

    def tool(self, name: str | None = None) -> Any:
        def deco(fn: Any) -> Any:
            @functools.wraps(fn)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(*args, **kwargs)
                except (ValueError, KeyError, OSError) as exc:
                    raise self.tool_error(f"{type(exc).__name__}: {exc}") from exc

            return self.srv.tool(name=name)(guarded)

        return deco


def _make_server(svc: Service) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp 2.x
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
        from mcp.server.fastmcp.exceptions import ToolError

    srv = Server("pciescope", instructions=INSTRUCTIONS)
    reg = _Registrar(srv, ToolError)

    @reg.tool(name="topology")
    def topology_tool(view: str = "summary", refresh: bool = False) -> dict[str, Any]:
        """PCIe topology: root complexes, the two switches, ports, links, GPUs (HIP/smi
        order, slot). view='summary' (tree text + table) or 'full' (every node/link)."""
        return svc.topology(view, refresh)

    @reg.tool()
    def snapshot(
        collectors: str | None = None,
        raw: bool = False,
        metric: list[str] | None = None,
        entity: list[str] | None = None,
    ) -> dict[str, Any]:
        """One passive sample: per-GPU load/temp/power/link, per-link speed/width/
        degraded, AER error counters, KFD eviction. raw/metric/entity add raw samples
        (prefix filters)."""
        return svc.snapshot(collectors, raw, metric, entity)

    @reg.tool()
    def start_watch(
        interval_s: float = 1.0,
        duration_s: float | None = None,
        collectors: str | None = None,
        label: str = "mcp-watch",
    ) -> dict[str, Any]:
        """Record a monitoring session in the background; returns watch_id."""
        return svc.start_watch(interval_s, duration_s, collectors, label)

    @reg.tool()
    def stop_watch(watch_id: str | None = None) -> dict[str, Any]:
        """Stop a background watch; returns its summary and findings."""
        return svc.stop_watch(watch_id)

    @reg.tool()
    def watch_status() -> dict[str, Any]:
        """Running watches with live tick/sample/event counters."""
        return svc.watch_status()

    @reg.tool()
    def measure(
        suite: str = "quick",
        gpus: str | list[str] | None = None,
        owner: str | None = None,
        dry_run: bool = DEFAULT_MEASURE_DRY_RUN,
        repeats: int | None = None,
        max_seconds: float | None = None,
        force_noisy: bool = False,
        wait_quiet_s: float | None = None,
        in_situ_s: float | None = None,
        interval_s: float = 1.0,
        list_suites: bool = False,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Active latency/bandwidth probes (suite quick|standard|full) on GPUs
        (buses such as '0a', hip:6, slot A/B). Dry run by default: plans, lists the
        blockers and the lease files needed. dry_run=false needs owner=<name> and
        the lease of every GPU (never GPU 5 / bus 07). in_situ_s=N instead observes
        passive counters for N seconds (no lease). list_suites=true lists suites."""
        return svc.measure(
            suite,
            gpus,
            owner=owner,
            dry_run=dry_run,
            repeats=repeats,
            max_seconds=max_seconds,
            force_noisy=force_noisy,
            wait_quiet_s=wait_quiet_s,
            in_situ_s=in_situ_s,
            interval_s=interval_s,
            list_suites=list_suites,
            session=session,
        )

    @reg.tool(name="report")
    def report_tool(
        session: str = "latest",
        sections: list[str] | None = None,
        format: str = "json",
    ) -> dict[str, Any]:
        """Summarise a session: links, GPUs, errors, events, probes, findings.
        format='text' returns the human table."""
        return svc.report(session, sections, format)

    @reg.tool()
    def query_trace(
        session: str = "latest",
        kind: str = "stats",
        metric: str | None = None,
        entity: str | None = None,
        since_s: float | None = None,
        until_s: float | None = None,
        severity: str | None = None,
        limit: int = 200,
        tail: bool = False,
    ) -> dict[str, Any]:
        """Query a session trace: kind=stats|series|samples|events|measurements,
        metric/entity prefix filters, since_s/until_s seconds from session start."""
        return svc.query_trace(
            session, kind, metric, entity, since_s, until_s, severity, limit, tail
        )

    @reg.tool()
    def sessions(limit: int = 20) -> dict[str, Any]:
        """Recorded sessions (newest first)."""
        return svc.sessions(limit)

    @reg.tool()
    def export_trace(session: str = "latest", out: str | None = None) -> dict[str, Any]:
        """Write a Chrome/Perfetto trace of a session (open it at
        https://ui.perfetto.dev); returns the file path on the box."""
        return svc.export_trace(session, out)

    @reg.tool()
    def metrics() -> dict[str, Any]:
        """Metric names, units, kind (gauge/counter) and meaning."""
        return svc.metrics()

    return srv


def build_server(env: Env | None = None) -> tuple[Any, Service]:
    """The MCP server object and its ``Service`` (tools are registered)."""
    svc = Service(env)
    return _make_server(svc), svc


def serve(env: Env | None = None) -> int:
    """Run the stdio server until the client disconnects."""
    srv, svc = build_server(env)
    try:
        srv.run("stdio")
    except KeyboardInterrupt:
        pass
    finally:
        svc.shutdown()
    return 0


__all__ = ["INSTRUCTIONS", "Service", "build_server", "serve"]
