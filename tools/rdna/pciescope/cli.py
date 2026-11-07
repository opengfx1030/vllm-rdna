# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope``: read and monitor the PCIe fabric of par1-cs25.

Verbs: ``topo`` ``snap`` ``watch`` ``measure`` ``report`` ``trace export``
``sessions`` ``metrics`` ``mcp`` ``tui`` ``plx`` ``capture``.  Every verb takes
``--json`` (machine output on stdout, logs and progress on stderr) and exits
non-zero on failure (1 failure, 2 usage; ``measure`` adds 2 refused and
3 aborted, see ``pciescope.probe``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pciescope import model, report, runner, store, topology, trace, views
from pciescope.env import Env, close_collectors, load_topology, parse_names

log = logging.getLogger("pciescope")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def emit_json(obj: Any) -> None:
    """Print ``obj`` as JSON on stdout (indented on a terminal, compact in a pipe)."""
    indent = 2 if sys.stdout.isatty() else None
    sys.stdout.write(
        json.dumps(
            obj, indent=indent, separators=(",", ":") if indent is None else None
        )
        + "\n"
    )


_HANDLER_NAME = "pciescope-cli"


def _setup_logging(args: argparse.Namespace) -> None:
    """Log to the *current* stderr (stdout carries data); safe to call repeatedly."""
    level = logging.WARNING
    if getattr(args, "quiet", False):
        level = logging.ERROR
    elif getattr(args, "verbose", 0) >= 2:
        level = logging.DEBUG
    elif getattr(args, "verbose", 0) == 1:
        level = logging.INFO
    for old in [h for h in log.handlers if h.get_name() == _HANDLER_NAME]:
        log.removeHandler(old)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("pciescope: %(levelname)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)


def _env(args: argparse.Namespace) -> Env:
    return Env(
        sysfs_root=args.sysfs_root,
        proc_root=args.proc_root,
        host=args.host,
        store_root=args.root,
    )


def _live_topology(args: argparse.Namespace) -> model.Topology:
    """Topology from ``--topology-from`` or discovered under ``--sysfs-root``."""
    src = getattr(args, "topology_from", None)
    if src:
        return load_topology(src, args.root)
    topo = _env(args).discover()
    for note in topo.notes:
        log.warning("%s", note)
    return topo


def _need_gpus(topo: model.Topology, args: argparse.Namespace) -> bool:
    if topo.gpus:
        return True
    log.error("no AMD GPU found under %s", args.sysfs_root)
    return False


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------


def cmd_topo(args: argparse.Namespace) -> int:
    topo = _live_topology(args)
    if args.json:
        emit_json(topo.to_dict())
    else:
        print(topology.render_tree(topo, full=args.full))
    return EXIT_OK if _need_gpus(topo, args) else EXIT_FAIL


def _match(value: str, prefixes: Sequence[str]) -> bool:
    return not prefixes or any(value.startswith(p) for p in prefixes)


def _log_skipped(skipped: Sequence[tuple[str, str]], *, explicit: bool) -> None:
    """A collector the user named and cannot have is a warning; one from the
    default set that cannot work here (plx off the box, say) is routine."""
    for name, why in skipped:
        (log.warning if explicit else log.info)("collector %s skipped: %s", name, why)


def cmd_snap(args: argparse.Namespace) -> int:
    env = _env(args)
    topo = _live_topology(args)
    if not _need_gpus(topo, args):
        return EXIT_FAIL
    res = env.build(
        topo, parse_names(args.collectors), allow_switch_writes=args.allow_switch_writes
    )
    _log_skipped(res.skipped, explicit=bool(args.collectors))
    if not res.collectors:
        log.error("no collector is available")
        return EXIT_FAIL
    try:
        tick = runner.snapshot(topo, res.collectors)
    finally:
        close_collectors(res.collectors)
    summary = views.snapshot_summary(
        topo,
        tick.samples,
        costs_ms=tick.costs_ms,
        collectors=[c.name for c in res.collectors],
        skipped=dict(res.skipped),
        ts_ns=tick.ts_ns,
    )
    raw = [
        s
        for s in tick.samples
        if _match(s.metric, args.metric) and _match(s.entity, args.entity)
    ]
    errors = [e for e in tick.events if e.name == "collector_error"]
    for e in errors:
        log.error("%s", e.message)
    if args.json:
        doc = dict(summary)
        doc["events"] = [e.to_dict() for e in tick.events]
        if args.raw or args.metric or args.entity:
            doc["samples"] = [s.to_dict() for s in raw]
        emit_json(doc)
    else:
        print(views.render_snapshot(summary))
        if args.raw or args.metric or args.entity:
            print()
            for s in raw:
                tags = ",".join(f"{k}={v}" for k, v in sorted(s.tags.items()))
                print(f"{s.entity:<24} {s.metric:<26} {s.value:<14g} {s.unit} {tags}")
    return EXIT_FAIL if errors else EXIT_OK


class _Stop:
    """SIGINT/SIGTERM -> graceful stop; a second SIGINT aborts."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._old: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        if self.event.is_set() and signum == signal.SIGINT:
            raise KeyboardInterrupt
        log.info("stopping (signal %d)", signum)
        self.event.set()

    def __enter__(self) -> threading.Event:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handler)
        return self.event

    def __exit__(self, *exc: object) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)


def cmd_watch(args: argparse.Namespace) -> int:
    env = _env(args)
    topo = _live_topology(args)
    if not _need_gpus(topo, args):
        return EXIT_FAIL
    res = env.build(
        topo, parse_names(args.collectors), allow_switch_writes=args.allow_switch_writes
    )
    _log_skipped(res.skipped, explicit=bool(args.collectors))
    if not res.collectors:
        log.error("no collector is available")
        return EXIT_FAIL
    names = [c.name for c in res.collectors]
    sess: store.Session | None = None
    if not args.no_record:
        sess = store.Session.create(
            env.session_root(),
            args.label,
            topo=topo,
            collectors=names,
            skipped=dict(res.skipped),
            interval_s=args.interval,
            cmdline=" ".join(["pciescope", *sys.argv[1:]]),
            name=args.session,
        )
        log.warning("recording to %s", sess.path)
    rn = runner.Runner(
        topo,
        res.collectors,
        session=sess,
        interval_s=args.interval,
        dedup=not args.full,
        heartbeat_s=args.heartbeat,
    )
    last_note = [time.monotonic()]

    def on_tick(tk: runner.Tick) -> None:
        for ev in tk.events:
            if ev.severity != "info":
                log.warning(
                    "[%s] %s %s %s", ev.severity, ev.name, ev.entity, ev.message
                )
        if args.stream:
            sys.stdout.write(
                "".join(
                    json.dumps(r.to_dict(), separators=(",", ":")) + "\n"
                    for r in [*tk.written, *tk.events]
                )
            )
            sys.stdout.flush()
        now = time.monotonic()
        if not args.quiet and now - last_note[0] >= 10.0:
            last_note[0] = now
            log.warning(
                "%d ticks, %d samples written, %d events",
                rn.summary.ticks,
                rn.summary.samples_written,
                rn.summary.events,
            )

    status = "ok"
    try:
        with _Stop() as stop:
            summary = rn.run(
                duration_s=args.duration,
                max_ticks=args.ticks,
                stop=stop,
                on_tick=on_tick,
            )
    except KeyboardInterrupt:
        status = "interrupted"
        summary = rn.summary
    finally:
        close_collectors(res.collectors)
        if sess is not None:
            sess.close(status)
    doc: dict[str, Any] = summary.to_dict()
    doc.update({"collectors": names, "skipped": dict(res.skipped), "status": status})
    if sess is not None:
        doc["session"] = sess.name
        doc["path"] = str(sess.path)
        doc["findings"] = report.build_report(store.Session.open(sess.path))["findings"]
    if args.stream:
        log.warning("%s", json.dumps(doc, separators=(",", ":")))
    elif args.json:
        emit_json(doc)
    else:
        print(_render_watch_summary(doc))
    errors = summary.event_names.get("collector_error", 0)
    return EXIT_FAIL if errors or status != "ok" or summary.ticks == 0 else EXIT_OK


def _render_watch_summary(doc: dict[str, Any]) -> str:
    out = [
        f"watch {doc['status']}: {doc['ticks']} ticks in {doc['duration_s']} s, "
        f"{doc['samples_written']}/{doc['samples_seen']} samples written, "
        f"{doc['events']} events {doc['event_names'] or ''}".rstrip(),
        "collectors: "
        + ", ".join(
            f"{n} {c['mean']:.2f} ms mean / {c['p95']:.2f} p95 / {c['max']:.2f} max"
            for n, c in doc["collector_cost_ms"].items()
            if c.get("n")
        ),
    ]
    if doc.get("path"):
        out.append(f"session: {doc['path']}")
        for f in doc.get("findings", []):
            out.append(f"  [{f['severity']}] {f['code']}: {f['message']}")
        if not doc.get("findings"):
            out.append("  no findings")
        out.append(f"report: python -m pciescope report {doc['session']}")
    return "\n".join(out)


_PROBE: list[Any] = []


def _probe_module() -> tuple[Any, str]:
    """The optional ``pciescope.probe`` package (module, error text)."""
    if not _PROBE:
        try:
            from pciescope import probe

            _PROBE.append((probe, ""))
        except Exception as exc:  # a broken optional slice must not take topo/snap down
            _PROBE.append((None, f"{type(exc).__name__}: {exc}"))
    return _PROBE[0]


def cmd_measure(args: argparse.Namespace) -> int:
    probe, why = _probe_module()
    if probe is None:
        log.error("the probe module is not available: %s", why)
        return EXIT_FAIL
    topo = _live_topology(args)
    if not _need_gpus(topo, args):
        return EXIT_FAIL
    root = _env(args).session_root()
    sess: store.Session | None = None
    if not (
        args.no_record
        or getattr(args, "dry_run", False)
        or getattr(args, "list_suites", False)
    ):
        if args.session:
            sess = store.Session.attach(args.session, root)
        else:
            sess = store.Session.create(
                root,
                f"measure-{getattr(args, 'suite', 'run')}",
                topo=topo,
                cmdline=" ".join(["pciescope", *sys.argv[1:]]),
            )
        log.warning("recording to %s", sess.path)
    status = "ok"
    code = EXIT_FAIL
    try:
        code = int(probe.cli_run(args, topo=topo, session=sess, emit=print) or 0)
        if code:
            status = f"exit {code}"
    except KeyboardInterrupt:
        status, code = "interrupted", 3
    finally:
        if sess is not None:
            sess.close(status)
    return code


def cmd_report(args: argparse.Namespace) -> int:
    sess = store.Session.open(args.session, args.root)
    topo = load_topology(args.topology_from, args.root) if args.topology_from else None
    rep = report.build_report(sess, topo, include_virtual=args.virtual)
    if args.json:
        emit_json(rep)
    else:
        print(report.render_text(rep))
    return EXIT_OK


def cmd_trace_export(args: argparse.Namespace) -> int:
    sess = store.Session.open(args.session, args.root)
    out = trace.export_chrome(sess, args.out)
    info = {
        "session": sess.name,
        "path": str(out),
        "bytes": out.stat().st_size,
        "viewer": "https://ui.perfetto.dev (Open trace file)",
    }
    if args.json:
        emit_json(info)
    else:
        print(f"{out}  ({info['bytes']} bytes)  open in {info['viewer']}")
    return EXIT_OK


def cmd_sessions(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else store.default_root()
    items = [store.describe(p) for p in reversed(store.list_sessions(root))]
    items = items[: args.limit] if args.limit else items
    if args.json:
        emit_json({"root": str(root), "sessions": items})
        return EXIT_OK
    print(f"{root}")
    for it in items:
        print(
            f"  {it['name']:<40} {it['status'] or '?':<8} "
            f"{(it['trace_bytes'] or 0) / 1024:>9.1f} KiB  {it['label'] or ''}"
        )
    return EXIT_OK


def cmd_metrics(args: argparse.Namespace) -> int:
    rows = [
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
    if args.json:
        emit_json(rows)
        return EXIT_OK
    for r in rows:
        print(
            f"{r['name']:<28} {r['unit']:<8} {r['kind']:<8} {r['entity']:<10} "
            f"{r['doc']}"
        )
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from pciescope import mcp_server
    except ImportError as exc:
        log.error("the MCP server needs the 'mcp' package: %s", exc)
        return EXIT_FAIL
    return mcp_server.serve(_env(args))


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        from pciescope.tui import app
    except ImportError as exc:
        log.error("the TUI is not installed (pip install textual): %s", exc)
        return EXIT_FAIL
    sysfs_root = None if args.sysfs_root == "/sys" else args.sysfs_root
    return int(
        app.run(
            replay=args.replay,
            interval=args.interval,
            collectors=parse_names(args.collectors),
            session_dir=args.session_dir or args.root,
            sysfs_root=sysfs_root,
            owner=args.owner,
            record=not args.no_record,
            dry_run=args.dry_run,
            speed=args.speed,
        )
    )


def cmd_plx(args: argparse.Namespace) -> int:
    try:
        from pciescope.plx.cli import main as plx_main
    except ImportError as exc:
        log.error("the switch register reader is not installed: %s", exc)
        return EXIT_FAIL
    return int(plx_main(list(args.plx_args)) or 0)


def cmd_capture(args: argparse.Namespace) -> int:
    from pciescope import capture

    info = capture.capture(
        args.outdir, sys_root=args.sysfs_root, proc_root=args.proc_root, host=args.host
    )
    if args.tar:
        info["tarball"] = args.tar
        info["tarball_bytes"] = capture.write_tarball(args.outdir, args.tar)
    if args.json:
        emit_json(info)
    else:
        for k, v in info.items():
            print(f"{k}: {v}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json", action="store_true", help="JSON on stdout")
    p.add_argument(
        "--sysfs-root",
        default=os.environ.get("PCIESCOPE_SYSFS", "/sys"),
        help="sysfs root (default /sys; a captured tree for replays)",
    )
    p.add_argument(
        "--proc-root",
        default=os.environ.get("PCIESCOPE_PROC", "/proc"),
        help="procfs root",
    )
    p.add_argument("--host", default=None, help="host name recorded in the topology")
    p.add_argument(
        "--root",
        default=None,
        help="session store (default $PCIESCOPE_HOME or ~/w4a8_runs/pciescope)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    p.add_argument("-q", "--quiet", action="store_true", help="errors only")
    return p


def _collectors_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--collectors",
        default=None,
        help="comma list, 'all' or 'default' (sysfs_link,aer,amdgpu,kfd)",
    )
    p.add_argument(
        "--allow-switch-writes",
        action="store_true",
        help="let the plx collector enable the switch performance counters "
        "(register writes; off by default)",
    )


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    ap = argparse.ArgumentParser(
        prog="pciescope",
        description="Read and monitor the PCIe fabric (2 PEX88096 switches, GPU "
        "links) and measure latency, bandwidth and effectiveness.",
    )
    ap.add_argument(
        "--version", action="version", version=f"pciescope {model.TOOL_VERSION}"
    )
    sub = ap.add_subparsers(dest="verb", metavar="VERB")

    def verb(name: str, fn: Callable[[argparse.Namespace], int], help_: str, **kw: Any):
        p = sub.add_parser(name, parents=[common], help=help_, description=help_, **kw)
        p.set_defaults(func=fn)
        return p

    p = verb("topo", cmd_topo, "discover and print the fabric topology")
    p.add_argument("--full", action="store_true", help="list on-card bridges too")
    p.add_argument("--topology-from", help="topo --json file or session, not sysfs")

    p = verb("snap", cmd_snap, "one sample of every collector")
    _collectors_arg(p)
    p.add_argument("--metric", action="append", default=[], help="prefix filter (raw)")
    p.add_argument("--entity", action="append", default=[], help="prefix filter (raw)")
    p.add_argument("--raw", action="store_true", help="include the raw samples")
    p.add_argument("--topology-from", help="topo --json file or session")

    p = verb("watch", cmd_watch, "sample continuously and record a session")
    _collectors_arg(p)
    p.add_argument("--interval", type=float, default=1.0, help="seconds per tick")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    p.add_argument("--session", default=None, help="exact session directory name")
    p.add_argument("--label", default="watch", help="session label (name suffix)")
    p.add_argument("--no-record", action="store_true", help="do not write a session")
    p.add_argument("--full", action="store_true", help="write unchanged samples too")
    p.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="rewrite unchanged samples every N s",
    )
    p.add_argument("--stream", action="store_true", help="print records as JSONL")
    p.add_argument("--topology-from", help="topo --json file or session")

    p = verb(
        "measure",
        cmd_measure,
        "active latency/bandwidth probes (needs GPU leases)",
        conflict_handler="resolve",  # the probe's --quiet replaces the common one
    )
    probe, _why = _probe_module()
    if probe is not None:
        probe.add_cli_arguments(p)
    p.add_argument("--session", default=None, help="record into this existing session")
    p.add_argument("--no-record", action="store_true", help="do not write a session")
    p.add_argument("--topology-from", help="topo --json file or session (off-box plan)")

    p = verb("report", cmd_report, "summarise a recorded session")
    p.add_argument("session", nargs="?", default="latest", help="name, path or latest")
    p.add_argument("--virtual", action="store_true", help="include virtual links")
    p.add_argument("--topology-from", help="topology for sessions without one")

    tp = sub.add_parser("trace", help="trace tools", description="trace tools")
    tsub = tp.add_subparsers(dest="trace_verb", metavar="ACTION", required=True)
    q = tsub.add_parser(
        "export", parents=[common], help="write Chrome/Perfetto trace JSON"
    )
    q.add_argument("session", nargs="?", default="latest")
    q.add_argument("--out", default=None, help="output file (default in the session)")
    q.set_defaults(func=cmd_trace_export, verb="trace")

    p = verb("sessions", cmd_sessions, "list recorded sessions (newest first)")
    p.add_argument("--limit", type=int, default=20)

    verb("metrics", cmd_metrics, "list the metric names, units and meaning")
    verb("mcp", cmd_mcp, "serve the MCP tools over stdio")

    p = verb("tui", cmd_tui, "interactive terminal dashboard")
    p.add_argument("--replay", default=None, help="session to replay instead of live")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--collectors", default=None)
    p.add_argument("--session-dir", default=None, help="where the TUI records")
    p.add_argument("--owner", default=os.environ.get("PCIESCOPE_OWNER", ""))
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="probes plan only")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed")

    px = sub.add_parser(
        "plx",
        add_help=False,
        help="PEX88096 switch register reader (see pciescope.plx)",
    )
    px.add_argument("plx_args", nargs=argparse.REMAINDER)
    px.set_defaults(func=cmd_plx)

    p = verb("capture", cmd_capture, "developer: record sysfs for offline tests")
    p.add_argument("outdir")
    p.add_argument("--tar", default=None, help="also write a deterministic tarball")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["plx"]:  # its own parser: leading options (--help) pass through
        return cmd_plx(argparse.Namespace(plx_args=raw[1:]))
    ap = build_parser()
    args = ap.parse_args(raw)
    _setup_logging(args)
    if getattr(args, "func", None) is None:
        ap.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:
        with contextlib.suppress(OSError):
            sys.stdout.close()
        return EXIT_OK
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError) as exc:
        if getattr(args, "verbose", 0):
            raise
        log.error("%s: %s", type(exc).__name__, exc)
        return EXIT_FAIL
