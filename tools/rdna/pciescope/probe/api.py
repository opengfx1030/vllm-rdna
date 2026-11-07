# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public API of the active probe: ``run_suite`` and the ``measure`` CLI helpers.

``run_suite`` is what ``pciescope measure``, the MCP tool and the TUI call. It
never raises for policy or environment problems; it returns a ``MeasureResult``
whose ``status`` is

* ``refused``: a rule forbids the run (lease not held, unknown or forbidden
  GPU, noisy box, too little free VRAM, bad suite). Nothing ran. ``refusal``
  says why and ``needed`` lists the lease holder files to create;
* ``failed``: the environment is broken (binary missing, bus order differs,
  the probe died) or no result was produced; ``partial``: some steps failed;
* ``aborted``: a guard or the user stopped the run (junction temperature,
  time limit, cancel, noisy box), the results so far are kept;
* ``ok`` / ``dry-run``.

Exit codes (``MeasureResult.exit_code``): 0 ok or dry-run, 1 failed or partial,
2 refused, 3 aborted, 4 finished but INVALID (noisy window forced, or an
``evicted_ms`` jump): the numbers must not be used.

Preflight order of a real run: GPUs -> lease -> suite plan -> noise (buildx,
quiet files, boot lock) -> junction sensors and free VRAM -> binary (``version``)
-> bus order of the probe (``buses`` under ``HIP_VISIBLE_DEVICES``) -> run.
Only the lease and VRAM checks cannot be bypassed; ``force_noisy`` skips the
noise refusal and marks the affected cells invalid. A dry run never wakes a
runtime-suspended GPU: it does not read its VRAM counters and lists it in
``MeasureResult.asleep`` (a real run reads them, it is about to use the GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pciescope.model import Event, Measurement, Record, Topology, now_ns

from . import insitu
from .binary import BinaryStatus, binary_status
from .gpus import resolve_gpus, split_tokens
from .guard import gpu_suspended, noisy_reasons, vram_info
from .ids import HOST, gpu_entity
from .lease import LeaseCheck, check_leases, resolve_owner
from .paths import FabricModel
from .plan import SUITES, Plan, SuiteInfo, list_suites, make_plan
from .render import render_text
from .results import aggregate, effectiveness_report
from .runner import (
    GuardEnv,
    RunLimits,
    Runner,
    RunOutcome,
    check_bus_order,
    default_binary,
    new_run_id,
    probe_buses,
)

ProgressFn = Callable[[Event], None]
SinkFn = Callable[[Record], None]

GIB = 1 << 30
#: free VRAM the probe needs per GPU: it allocates 1.5 GiB (peer plans)
MIN_FREE_VRAM = 3 * GIB
MIN_FREE_VRAM_HOST_ONLY = 1 * GIB

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_ABORTED, EXIT_INVALID = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------


@dataclass
class MeasureResult:
    """Everything a measure run produced or why it did not run."""

    status: str
    suite: str
    run_id: str
    owner: str = ""
    gpus: list[str] = field(default_factory=list)
    plan: Plan | None = None
    measurements: list[Measurement] = field(default_factory=list)
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    refusal: str = ""
    needed: list[str] = field(default_factory=list)
    error: str = ""
    abort_reason: str = ""
    blockers: list[str] = field(default_factory=list)  # dry-run: what would stop it
    steps_total: int = 0
    steps_done: int = 0
    steps_failed: list[dict[str, str]] = field(default_factory=list)
    max_temp_c: dict[str, float] = field(default_factory=dict)
    evicted: list[dict[str, Any]] = field(default_factory=list)
    degraded_links: dict[str, str] = field(default_factory=dict)
    noise: list[str] = field(default_factory=list)
    asleep: list[str] = field(default_factory=list)  # suspended GPUs a dry run spared
    windows: int = 0
    tick_mhz: list[float] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)
    binary: BinaryStatus | None = None
    lease: LeaseCheck | None = None
    started_ns: int = 0
    ended_ns: int = 0
    session: str | None = None
    model: FabricModel | None = field(default=None, repr=False)

    @property
    def exit_code(self) -> int:
        if self.status == "refused":
            return EXIT_REFUSED
        if self.status == "aborted":
            return EXIT_ABORTED
        if self.status in ("failed", "partial"):
            return EXIT_FAILED
        if not self.valid:
            return EXIT_INVALID
        return EXIT_OK

    def summary(self) -> list[Measurement]:
        """The measurements with repeats folded into one record per cell."""
        return aggregate(self.measurements)

    def effectiveness(self) -> dict[str, Any]:
        """Measured vs theoretical per path class, audit baseline, findings."""
        return effectiveness_report(
            self.summary(), self.model, degraded_links=self.degraded_links
        )

    def to_dict(self, *, steps: bool | None = None) -> dict[str, Any]:
        """JSON-ready document (``--json`` output and the MCP tool result).

        ``steps`` adds the full list of planned commands to ``plan`` (default:
        only for a dry run).
        """
        want_steps = self.status == "dry-run" if steps is None else steps
        plan = None
        if self.plan is not None:
            plan = self.plan.to_dict()
            if not want_steps:
                plan.pop("steps", None)
                plan["steps_total"] = len(self.plan.steps)
        summary = self.summary()
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "suite": self.suite,
            "run": self.run_id,
            "owner": self.owner,
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "gpus": list(self.gpus),
            "refusal": self.refusal,
            "needed": list(self.needed),
            "error": self.error,
            "abort_reason": self.abort_reason,
            "blockers": list(self.blockers),
            "steps": {
                "total": self.steps_total,
                "done": self.steps_done,
                "failed": list(self.steps_failed),
            },
            "duration_s": round((self.ended_ns - self.started_ns) / 1e9, 2)
            if self.ended_ns
            else 0.0,
            "plan": plan,
            "summary": [m.to_dict() for m in summary],
            "measurements": [m.to_dict() for m in self.measurements],
            "effectiveness": self.effectiveness() if summary else None,
            "guard": {
                "max_temp_c": dict(self.max_temp_c),
                "evicted": list(self.evicted),
                "degraded_links": dict(self.degraded_links),
                "noise_at_start": list(self.noise),
                "asleep": list(self.asleep),
                "windows": self.windows,
            },
            "probe": {
                "tick_mhz": list(self.tick_mhz),
                "stderr_tail": list(self.stderr_tail),
                "binary": self.binary.to_dict() if self.binary else None,
            },
            "lease": None
            if self.lease is None
            else {
                "ok": self.lease.ok,
                "owner": self.lease.owner,
                "needed": list(self.lease.needed),
                "reason": self.lease.reason,
            },
            "session": self.session,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _emit(progress: ProgressFn | None, name: str, message: str, **data: Any) -> None:
    if progress is not None:
        progress(Event(now_ns(), HOST, name, "info", message, {"kind": name, **data}))


def _refused(
    result: MeasureResult,
    reasons: Sequence[str],
    needed: Sequence[str] = (),
    sink: SinkFn | None = None,
) -> MeasureResult:
    result.status = "refused"
    result.refusal = "\n".join(reasons)
    result.needed = list(needed)
    result.ended_ns = now_ns()
    if sink is not None:
        sink(
            Event(
                now_ns(),
                HOST,
                "measure_refused",
                "warn",
                result.refusal,
                {
                    "run": result.run_id,
                    "suite": result.suite,
                    "owner": result.owner,
                    "gpus": result.gpus,
                    "needed": result.needed,
                },
            )
        )
    return result


def _wait_for_quiet(
    me: str,
    env: GuardEnv,
    wait_s: float,
    cancel: threading.Event | None,
) -> list[str]:
    """Noise reasons now; wait up to ``wait_s`` for them to clear."""
    deadline = time.monotonic() + wait_s
    while True:
        reasons = noisy_reasons(
            me,
            proc_root=env.proc_root,
            quiet_dir=env.quiet_dir,
            boot_lock=env.boot_lock,
        )
        if not reasons or time.monotonic() >= deadline:
            return reasons
        (cancel or threading.Event()).wait(min(2.0, deadline - time.monotonic()))
        if cancel is not None and cancel.is_set():
            return reasons


def _vram_problems(
    plan: Plan, env: GuardEnv, *, wake: bool, asleep: list[str]
) -> list[str]:
    """GPUs without enough free VRAM for the probe.

    A runtime-suspended GPU is only read when ``wake`` (a real run, about to use
    it anyway); otherwise it goes to ``asleep`` unread, because reading its
    VRAM counters would resume it and its bridge.
    """
    need = MIN_FREE_VRAM if plan.needs_peer else MIN_FREE_VRAM_HOST_ONLY
    problems: list[str] = []
    for bus in plan.buses:
        if not wake and gpu_suspended(bus, env.sysfs):
            asleep.append(gpu_entity(bus))
            continue
        info = vram_info(bus, env.sysfs)
        if info is None:
            problems.append(f"{gpu_entity(bus)}: cannot read its VRAM usage")
            continue
        free, total = info
        if free < need:
            problems.append(
                f"{gpu_entity(bus)}: {free / GIB:.1f} GiB of {total / GIB:.1f} GiB "
                f"VRAM free, the probe needs {need / GIB:.0f} GiB "
                "(another process, for example a server, holds the rest)"
            )
    return problems


def _binary_problem(status: BinaryStatus, plan: Plan) -> str:
    if status.problem:
        return status.problem
    wanted = {s.kind for s in plan.steps} | {"batch", "buses"}
    missing = sorted(wanted - set(status.modes))
    if missing:
        return (
            f"{status.path} lacks mode(s) {', '.join(missing)}: rebuild it with "
            "python -m pciescope.probe build"
        )
    return ""


# ---------------------------------------------------------------------------
# run_suite
# ---------------------------------------------------------------------------


def run_suite(
    suite: str,
    gpus: str | Iterable[str],
    *,
    owner: str | None = None,
    topo: Topology | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    dry_run: bool = False,
    force_noisy: bool = False,
    sink: SinkFn | None = None,
    repeats: int | None = None,
    max_seconds: float | None = None,
    wait_quiet_s: float = 0.0,
    binary: Path | None = None,
    env: GuardEnv | None = None,
    limits: RunLimits | None = None,
    session_path: str | None = None,
) -> MeasureResult:
    """Plan, check, run and evaluate ``suite`` on ``gpus`` (see the module doc).

    ``gpus``: tokens (or one string) as ``--gpus``: buses (``0a``, ``gpu:0a``),
    HIP indices (``hip:6``), ``A``/``B`` (lease slots), ``w7800cb``. ``owner``
    is the lease holder name (default env ``PCIESCOPE_OWNER``). ``progress``
    receives UI events (plan, step_start, step_done, measurement, guard, abort,
    done), ``sink`` the trace records (measurements and the events
    measure_start, thermal, evicted_jump, measure_end, measure_refused).
    ``cancel`` stops the run after the current step. ``dry_run`` plans and
    checks everything but starts no GPU work and refuses nothing: the blockers a
    real run would hit are in ``blockers``.
    """
    if topo is None:
        from pciescope.topology import discover

        topo = discover()
    guard_env = env or GuardEnv()
    me = resolve_owner(owner) or ""
    result = MeasureResult(
        status="failed",
        suite=suite,
        run_id=new_run_id(),
        owner=me,
        started_ns=now_ns(),
        session=session_path,
    )
    try:
        buses = resolve_gpus(gpus, topo)
    except ValueError as exc:
        return _refused(result, [str(exc)], sink=sink)
    result.gpus = [gpu_entity(b) for b in buses]

    policy: list[str] = []  # rules that forbid the run: refused
    broken: list[str] = []  # environment problems: failed
    check = check_leases(buses, me or None, guard_env.lease_dir)
    result.lease = check
    if not check.ok:
        policy.append(check.message())
    try:
        plan = make_plan(suite, buses, topo, repeats=repeats)
    except ValueError as exc:
        plan = None
        policy.append(str(exc))
    if plan is not None:
        result.plan = plan
        result.gpus = plan.entities
        result.steps_total = len(plan.steps)
    if policy and not dry_run:
        return _refused(result, policy, check.needed, sink)

    if plan is not None:
        policy += _preflight(
            result, plan, guard_env, force_noisy, wait_quiet_s, cancel, wake=not dry_run
        )
        status = binary_status(binary if binary is not None else default_binary())
        result.binary = status
        problem = _binary_problem(status, plan)
        if problem:
            broken.append(problem)
    if dry_run:
        result.status = "dry-run"
        result.blockers = policy + broken
        result.ended_ns = now_ns()
        _emit(progress, "plan", f"{suite}: dry run", n=len(plan.steps) if plan else 0)
        _emit(
            progress, "done", "dry run: " + ("blocked" if result.blockers else "ready")
        )
        return result
    if policy:
        return _refused(result, policy, (), sink)
    assert plan is not None and result.binary is not None
    if broken:
        result.error = "; ".join(broken)
        result.ended_ns = now_ns()
        return result
    devices, err = probe_buses(Path(result.binary.path), plan.hip_visible)
    order = err or check_bus_order(devices, plan)
    if order:
        result.error = f"probe bus order check failed: {order}"
        result.ended_ns = now_ns()
        return result

    model = FabricModel(topo)
    runner = Runner(
        plan,
        topo,
        binary=Path(result.binary.path),
        owner=me,
        run_id=result.run_id,
        env=guard_env,
        limits=limits or RunLimits(max_seconds=max_seconds),
        force_noisy=force_noisy,
        progress=progress,
        sink=sink,
        cancel=cancel,
        model=model,
        extra_tags={"force_noisy": "1"} if force_noisy else {},
    )
    _finish(result, runner.run(), model)
    return result


def _preflight(
    result: MeasureResult,
    plan: Plan,
    env: GuardEnv,
    force_noisy: bool,
    wait_quiet_s: float,
    cancel: threading.Event | None,
    *,
    wake: bool,
) -> list[str]:
    """Noise and VRAM checks: the rules that refuse a run (empty: all clear)."""
    out: list[str] = []
    noise = _wait_for_quiet(result.owner, env, wait_quiet_s, cancel)
    result.noise = list(noise)
    if noise and not force_noisy:
        out.append(
            "the box is not quiet (another measured window or a build is "
            "active); wait, use --wait-quiet SECONDS, or --force-noisy to run "
            "anyway with the affected results marked INVALID:\n  - "
            + "\n  - ".join(noise)
        )
    out += _vram_problems(plan, env, wake=wake, asleep=result.asleep)
    return out


def _finish(result: MeasureResult, out: RunOutcome, model: FabricModel) -> None:
    result.status = out.status
    result.measurements = out.measurements
    result.valid = out.valid
    result.invalid_reasons = out.invalid_reasons
    result.error = out.error
    result.abort_reason = out.abort_reason
    result.steps_total = out.steps_total
    result.steps_done = out.steps_done
    result.steps_failed = out.steps_failed
    result.max_temp_c = out.max_temp_c
    result.evicted = out.evicted
    result.degraded_links = out.degraded_links
    result.windows = out.windows
    result.tick_mhz = out.tick_mhz
    result.stderr_tail = out.stderr_tail
    result.started_ns = out.started_ns or result.started_ns
    result.ended_ns = out.ended_ns or now_ns()
    result.model = model


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``measure`` options to ``parser`` (Core adds ``--json`` etc.)."""
    parser.add_argument(
        "--suite",
        choices=list(SUITES),
        default="quick",
        help="quick (~30 s), standard (~5 min, key cells x3), full (~15 min)",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        metavar="GPU",
        help="GPUs to measure: buses (0a gpu:0f), HIP indices (hip:6), slot A or B, "
        "w7800cb; comma or space separated. Every GPU needs your lease.",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="lease holder name (default: $PCIESCOPE_OWNER); the holder file "
        "~/w4a8_runs/lanes/lease/<owner>.<slot> must exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and check only: no GPU work, nothing is refused, blockers listed",
    )
    parser.add_argument(
        "--force-noisy",
        action="store_true",
        help="run although a build or another agent's quiet window is active; "
        "affected results are marked INVALID (the lease and VRAM checks stay)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="time guard of the whole run (default: 3x the estimate + 60 s)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="runs of every key cell (default: quick 1, standard 3, full 3)",
    )
    parser.add_argument(
        "--wait-quiet",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="wait up to this long for a quiet box before refusing",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="probe binary (default: $PCIESCOPE_PROBE_BIN or "
        "~/Apps/pciescope/bin/p2p_probe)",
    )
    parser.add_argument(
        "--list-suites", action="store_true", help="list the suites and exit"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no progress lines on stderr"
    )
    parser.add_argument(
        "--in-situ",
        type=float,
        default=None,
        metavar="SECONDS",
        help="no suite: observe passive counters for this long and report link "
        "utilisation of the running workload (no lease needed, nothing injected)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="sampling interval of --in-situ in seconds (default: 1)",
    )


def _suite_dict(s: SuiteInfo) -> dict[str, Any]:
    return {"name": s.name, "description": s.description, "est_seconds": s.est_seconds}


def _progress_printer(stream: Any) -> ProgressFn:
    def show(e: Event) -> None:
        d = e.data
        if e.name == "step_done":
            line = f"[{d.get('i')}/{d.get('n')}] {d.get('scenario')}"
            if d.get("error"):
                line += f"  FAILED: {d['error']}"
        elif e.name in ("measurement", "plan", "guard", "abort", "done"):
            line = f"{e.name}: {e.message}"
        else:
            return
        print(line, file=stream, flush=True)

    return show


def cli_run(
    args: argparse.Namespace,
    *,
    topo: Topology,
    session: Any = None,
    emit: Callable[[str], None] = print,
) -> int:
    """Run ``pciescope measure`` for parsed ``args``; return the exit code.

    ``session`` is a ``pciescope.store.Session`` (or None): every measurement
    and event is appended to its trace. Output goes through ``emit``: one JSON
    document with ``--json``, otherwise a text report; progress lines go to
    stderr unless ``--quiet``.
    """
    as_json = bool(getattr(args, "json", False))
    in_situ = getattr(args, "in_situ", None)
    if in_situ is not None:
        report = insitu.observe(
            topo, duration_s=in_situ, interval_s=args.interval, session=session
        )
        if as_json:
            emit(json.dumps({"mode": "in-situ", **report.to_dict()}))
        else:
            emit(insitu.render_text(report))
        return EXIT_OK
    if getattr(args, "list_suites", False):
        if as_json:
            emit(json.dumps({"suites": [_suite_dict(s) for s in list_suites()]}))
        else:
            emit(
                "\n".join(
                    f"{s.name:9s} ~{s.est_seconds:>4d} s  {s.description}"
                    for s in list_suites()
                )
            )
        return EXIT_OK
    if not args.gpus:
        msg = (
            "no GPUs given: pass --gpus (buses such as 0a, slot A or B, hip:6); "
            "every GPU needs your lease"
        )
        if as_json:
            emit(
                json.dumps(
                    {"status": "refused", "exit_code": EXIT_REFUSED, "refusal": msg}
                )
            )
        else:
            emit(msg)
        return EXIT_REFUSED
    sink: SinkFn | None = session.append if session is not None else None
    progress = None if args.quiet else _progress_printer(sys.stderr)
    result = run_suite(
        args.suite,
        split_tokens(args.gpus),
        owner=args.owner,
        topo=topo,
        progress=progress,
        dry_run=args.dry_run,
        force_noisy=args.force_noisy,
        sink=sink,
        repeats=args.repeats,
        max_seconds=args.max_seconds,
        wait_quiet_s=args.wait_quiet,
        binary=Path(args.binary).expanduser() if args.binary else None,
        session_path=str(session.path) if session is not None else None,
    )
    if as_json:
        emit(json.dumps(result.to_dict(), separators=(",", ":"), allow_nan=False))
    else:
        emit(render_text(result))
    return result.exit_code


__all__ = [
    "EXIT_ABORTED",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "EXIT_REFUSED",
    "MeasureResult",
    "add_cli_arguments",
    "check_leases",
    "cli_run",
    "list_suites",
    "run_suite",
]
