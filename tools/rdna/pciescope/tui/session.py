# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between the TUI and Core: discovery, collectors, session store, export."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from pciescope.model import Topology
from pciescope.tui.measure import MeasureRunner
from pciescope.tui.sources import LiveSource, ReplaySource, Source


class SessionError(Exception):
    """A user-facing startup problem (bad session path, no topology ...)."""


@dataclass
class SessionSetup:
    source: Source
    runner: MeasureRunner | None
    export: Callable[[], str] | None


def open_session(
    *,
    replay: str | None,
    interval: float,
    collectors: list[str] | None,
    session_dir: str | None,
    sysfs_root: str | None,
    owner: str,
    record: bool,
    dry_run: bool,
    speed: float,
) -> SessionSetup:
    if replay:
        return _open_replay(replay, session_dir, speed)
    return _open_live(
        interval, collectors, session_dir, sysfs_root, owner, record, dry_run
    )


def export_trace(session_path: str, flush: Callable[[], None] | None = None) -> str:
    """Write a Perfetto / chrome trace next to the session; returns its path."""
    from pciescope import trace

    if flush is not None:
        flush()
    out = os.path.join(session_path, "trace.perfetto.json")
    result = trace.export_chrome(session_path, out)
    return str(result or out)


def _open_live(interval, collectors, session_dir, sysfs_root, owner, record, dry_run):
    from pciescope import collectors as cmod
    from pciescope import runner as runner_mod
    from pciescope import store, topology
    from pciescope.sysfs import Sysfs

    sfs = Sysfs(sysfs_root) if sysfs_root else None
    try:
        topo = topology.discover(sfs)
    except Exception as exc:
        msg = f"topology discovery failed: {type(exc).__name__}: {exc}"
        raise SessionError(msg) from exc
    if not topo.gpus:
        raise SessionError("no GPUs found in sysfs (wrong --sysfs-root?)")
    built = cmod.build(topo, collectors, sysfs=sfs)
    names = [c.name for c in built.collectors]
    skipped = dict(built.skipped)

    def new_session():
        return store.Session.create(
            session_dir,
            "tui",
            topo=topo,
            collectors=names,
            skipped=skipped,
            interval_s=interval,
            cmdline="pciescope tui",
        )

    try:
        sess = new_session() if record else None
    except OSError as exc:
        raise SessionError(f"cannot create the session directory: {exc}") from exc
    sampler = runner_mod.Runner(
        topo, built.collectors, session=None, interval_s=interval
    )
    source = LiveSource(
        topo,
        sampler,
        built.skipped,
        interval=interval,
        session=sess,
        session_factory=new_session,
    )
    measure = MeasureRunner(topo, owner=owner, sink=source.publish, dry_run=dry_run)

    def export() -> str:
        if source.session is None:
            raise RuntimeError("not recording: nothing to export (press r)")
        return export_trace(source.session_path, source.flush)

    return SessionSetup(source, measure, export)


def _open_replay(ref: str, session_dir: str | None, speed: float) -> SessionSetup:
    from pciescope import store

    try:
        sess = store.Session.open(ref, session_dir)
    except FileNotFoundError as exc:
        raise SessionError(str(exc)) from exc
    topo: Topology | None = sess.topology()
    if topo is None:
        raise SessionError(f"{sess.path} has no topology.json; cannot replay")
    records = list(sess.records())
    if not records:
        raise SessionError(f"{sess.path} holds no records yet")
    source = ReplaySource(
        topo,
        records,
        path=str(sess.path),
        collectors=list(sess.meta.get("collectors") or []),
        unavailable=dict(sess.meta.get("skipped") or {}),
        speed=speed,
        interval_s=float(sess.meta.get("interval_s") or 1.0),
    )
    return SessionSetup(source, None, lambda: export_trace(str(sess.path)))
