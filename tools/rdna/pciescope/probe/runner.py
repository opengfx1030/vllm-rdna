# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a ``Plan`` on one ``p2p_probe batch`` process with the box guards.

The probe process is started once (it allocates and enables peer access once),
fed one command per step and read until the ``END`` line of that command. Before
every step the runner re-checks the noise rules (``docker buildx build``, quiet
files of other agents, ``boot.lock``); after every step it reads the KFD
``evicted_ms`` counters of the probed GPUs. A step that ran while the window
was noisy (forced) or while a counter jumped produces invalid measurements
(``ok=False``). A background thread watches the junction temperature: a breach
of the box rule (5 s mean >= 90 C or any sample >= 95 C), a step that exceeds
its time limit, the run time limit or a cancel request kills the probe process
group and ends the run as ``aborted``.

Events: ``progress`` receives the interface events of the TUI (``plan``,
``step_start``, ``step_done``, ``measurement``, ``guard``, ``abort``, ``done``);
``sink`` receives the records of the trace (every ``Measurement`` and the
events ``measure_start``, ``thermal``, ``evicted_jump``, ``measure_end``).
"""

from __future__ import annotations

import collections
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pciescope.model import HOST, Event, Measurement, Record, Topology, now_ns
from pciescope.sysfs import Sysfs

from . import parse
from .guard import (
    EvictionWatcher,
    KfdProc,
    QuietWindow,
    TempWatcher,
    noisy_reasons,
)
from .ids import gpu_entity
from .links import LinkWatcher
from .paths import FabricModel
from .plan import Plan, Step
from .results import ConvertContext, block_error, convert_block

ProgressFn = Callable[[Event], None]
SinkFn = Callable[[Record], None]

DEFAULT_BINARY = Path("~/Apps/pciescope/bin/p2p_probe").expanduser()
ENV_BINARY = "PCIESCOPE_PROBE_BIN"
_STDERR_KEEP = 40


def default_binary() -> Path:
    """Return the probe binary (env ``PCIESCOPE_PROBE_BIN`` overrides)."""
    env = os.environ.get(ENV_BINARY)
    return Path(env).expanduser() if env else DEFAULT_BINARY


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardEnv:
    """Where the guards read (defaults: the real box; tests pass fake trees)."""

    sysfs: Sysfs | None = None
    procfs: Sysfs | None = None
    proc_root: Path = Path("/proc")
    quiet_dir: Path | None = None
    boot_lock: Path | None = None
    lease_dir: Path | None = None


@dataclass(frozen=True)
class RunLimits:
    """Time limits of a run (seconds)."""

    max_seconds: float | None = None  # None: derived from the plan estimate
    step_floor_s: float = 20.0  # step limit = max(floor, factor * estimate + 10)
    step_factor: float = 3.0
    start_s: float = 180.0  # allocation + calibration of the probe process
    noisy_wait_s: float = 120.0  # how long a cell waits for a quiet box
    window_cap_s: float = 720.0  # rotate the quiet window (rule: 15 min max)
    window_gap_s: float = 125.0  # pause between windows (rule: 2 min gap)
    temp_period_s: float = 1.0
    kill_grace_s: float = 3.0
    close_grace_s: float = 15.0
    poll_s: float = 0.2

    def run_limit(self, plan: Plan) -> float:
        if self.max_seconds is not None:
            return self.max_seconds
        return min(3600.0, max(120.0, 3.0 * plan.est_total_s + 60.0))

    def step_limit(self, step: Step) -> float:
        return max(self.step_floor_s, self.step_factor * step.est_s + 10.0)


# ---------------------------------------------------------------------------
# helpers around the binary
# ---------------------------------------------------------------------------


def cpu_only_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that exposes no GPU (box rule for CPU-only commands)."""
    env = dict(os.environ if base is None else base)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = ""
    return env


def gpu_env(hip_visible: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that exposes exactly the HIP devices ``hip_visible``."""
    env = dict(os.environ if base is None else base)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = hip_visible
    return env


def _feed(text: str) -> list[parse.Item]:
    parser = parse.StreamParser()
    items: list[parse.Item] = []
    for line in text.splitlines():
        items.extend(parser.feed(line))
    return items


def probe_version(binary: Path, *, timeout: float = 30.0) -> parse.Version | None:
    """Run ``p2p_probe version`` (CPU-only) and return the parsed line.

    None when the binary is missing, does not run, or has no ``version`` mode
    (an old build).
    """
    try:
        done = subprocess.run(
            [str(binary), "version"],
            env=cpu_only_env(),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for item in _feed(done.stdout):
        if isinstance(item, parse.Version):
            return item
    return None


def probe_buses(
    binary: Path, hip_visible: str, *, timeout: float = 60.0
) -> tuple[list[parse.Device], str]:
    """Run ``p2p_probe buses`` with ``HIP_VISIBLE_DEVICES=hip_visible``.

    Returns the devices in logical order and an error text (empty when the
    command ran). It allocates nothing and moves no data.
    """
    try:
        done = subprocess.run(
            [str(binary), "buses"],
            env=gpu_env(hip_visible),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    devices = [i for i in _feed(done.stdout) if isinstance(i, parse.Device)]
    if done.returncode != 0 and not devices:
        tail = " | ".join(done.stderr.strip().splitlines()[-3:])
        return devices, f"rc={done.returncode} {tail}".strip()
    return devices, ""


def check_bus_order(devices: Sequence[parse.Device], plan: Plan) -> str:
    """Return why the probe's device order differs from the plan, '' if equal."""
    if len(devices) != len(plan.buses):
        return (
            f"probe sees {len(devices)} devices with HIP_VISIBLE_DEVICES="
            f"{plan.hip_visible}, the plan has {len(plan.buses)}"
        )
    for lid, (dev, bus) in enumerate(zip(devices, plan.buses, strict=True)):
        if dev.bus != bus:
            return (
                f"logical device {lid} is bus {dev.bus} in the probe but bus "
                f"{bus} in the topology (HIP_VISIBLE_DEVICES={plan.hip_visible})"
            )
    return ""


# ---------------------------------------------------------------------------
# the child process
# ---------------------------------------------------------------------------


class ProbeProcess:
    """A ``p2p_probe batch`` child in its own session, with line queues."""

    def __init__(self, argv: Sequence[str], env: Mapping[str, str]) -> None:
        self.argv = list(argv)
        self.env = dict(env)
        self._proc: subprocess.Popen[str] | None = None
        self._out: queue.Queue[str | None] = queue.Queue()
        self._err: collections.deque[str] = collections.deque(maxlen=_STDERR_KEEP)
        self._threads: list[threading.Thread] = []

    @property
    def pid(self) -> int:
        return self._proc.pid if self._proc is not None else -1

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
            close_fds=True,
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        for name, stream, sink in (
            ("out", self._proc.stdout, self._pump_out),
            ("err", self._proc.stderr, self._pump_err),
        ):
            thread = threading.Thread(
                target=sink, args=(stream,), name=f"probe-{name}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _pump_out(self, stream: Iterable[str]) -> None:
        try:
            for line in stream:
                self._out.put(line.rstrip("\r\n"))
        finally:
            self._out.put(None)

    def _pump_err(self, stream: Iterable[str]) -> None:
        for line in stream:
            self._err.append(line.rstrip("\r\n"))

    def send(self, line: str) -> bool:
        """Write one command line; False when the child is gone."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return False
        try:
            proc.stdin.write(line.rstrip("\n") + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            return False
        return True

    def get_line(self, timeout: float) -> tuple[bool, str | None]:
        """Return ``(got, line)``; ``(True, None)`` is the end of the output."""
        try:
            return True, self._out.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def stderr_tail(self) -> list[str]:
        return list(self._err)

    def kill(self, grace_s: float = 3.0) -> None:
        """SIGTERM the process group, SIGKILL it after ``grace_s``."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def close(self, grace_s: float = 15.0) -> int | None:
        """End of input, wait for the child to exit; kill it after ``grace_s``."""
        proc = self._proc
        if proc is None:
            return None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(grace_s)
        except subprocess.TimeoutExpired:
            self.kill()
        for thread in self._threads:
            thread.join(2.0)
        return proc.returncode


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """What a run produced (``api.MeasureResult`` wraps it)."""

    status: str = "ok"  # ok | partial | failed | aborted
    measurements: list[Measurement] = field(default_factory=list)
    steps_total: int = 0
    steps_done: int = 0
    steps_failed: list[dict[str, str]] = field(default_factory=list)
    abort_reason: str = ""
    error: str = ""
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    max_temp_c: dict[str, float] = field(default_factory=dict)
    evicted: list[dict[str, Any]] = field(default_factory=list)
    tick_mhz: list[float] = field(default_factory=list)
    probe_version: str = ""
    stderr_tail: list[str] = field(default_factory=list)
    started_ns: int = 0
    ended_ns: int = 0
    windows: int = 0
    degraded_links: dict[str, str] = field(default_factory=dict)

    def add_invalid(self, reason: str) -> None:
        if reason not in self.invalid_reasons:
            self.invalid_reasons.append(reason)
        self.valid = False


def new_run_id() -> str:
    """A short unique id for tags and event data (``YYYYMMDDTHHMMSS-xxxx``)."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _event(
    name: str,
    message: str = "",
    *,
    entity: str = HOST,
    severity: str = "info",
    **data: Any,
) -> Event:
    return Event(now_ns(), entity, name, severity, message, {"kind": name, **data})


def describe(m: Measurement) -> str:
    """One line for a result (progress and the human report)."""
    eff = "" if m.efficiency is None else f" ({m.efficiency:.0%} of raw)"
    size = f" {m.size_bytes} B" if m.size_bytes else ""
    flag = "" if m.ok else " INVALID"
    return (
        f"{m.kind} {m.direction}{size} {m.src}>{m.dst}: "
        f"{m.value:.2f} {m.unit}{eff}{flag}"
    )


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


class Runner:
    """Run ``plan`` on ``binary`` under the guards (see the module docstring)."""

    def __init__(
        self,
        plan: Plan,
        topo: Topology,
        *,
        binary: Path,
        owner: str,
        run_id: str | None = None,
        env: GuardEnv | None = None,
        limits: RunLimits | None = None,
        force_noisy: bool = False,
        progress: ProgressFn | None = None,
        sink: SinkFn | None = None,
        cancel: threading.Event | None = None,
        model: FabricModel | None = None,
        extra_tags: Mapping[str, str] | None = None,
    ) -> None:
        self.plan = plan
        self.topo = topo
        self.binary = binary
        self.owner = owner
        self.run_id = run_id or new_run_id()
        self.env = env or GuardEnv()
        self.limits = limits or RunLimits()
        self.force_noisy = force_noisy
        self.progress = progress
        self.sink = sink
        self.cancel = cancel or threading.Event()
        self.model = model if model is not None else FabricModel(topo)
        self.tags = {"run": self.run_id, "suite": plan.suite, "owner": owner}
        self.tags.update(extra_tags or {})
        self.out = RunOutcome(steps_total=len(plan.steps))
        self._abort = threading.Event()
        self._abort_text = ""
        self._thermal = ""
        self._t0 = time.monotonic()
        self._window: QuietWindow | None = None
        self._window_n = 0
        self._probe: ProbeProcess | None = None
        self._parser = parse.StreamParser()
        self._gpus = [topo.gpu(b) for b in plan.buses]
        self._links = LinkWatcher(topo, self._gpus, sysfs=self.env.sysfs)
        self._temps = TempWatcher(
            self._gpus,
            sysfs=self.env.sysfs,
            period_s=self.limits.temp_period_s,
            on_breach=self._on_thermal,
        )
        self._evict = EvictionWatcher(
            self._gpus, sysfs=self.env.sysfs, procfs=self.env.procfs
        )
        self._evict_base: dict[int, KfdProc] = {}
        self._evict_first: dict[int, KfdProc] = {}
        self._evicted_ms: dict[int, dict[str, int]] = {}

    # -- events --------------------------------------------------------------

    def _ui(self, name: str, message: str = "", **data: Any) -> None:
        if self.progress is not None:
            data.setdefault("run", self.run_id)
            data.setdefault("elapsed_s", round(time.monotonic() - self._t0, 2))
            self.progress(_event(name, message, **data))

    def _trace(self, event: Event) -> None:
        if self.sink is not None:
            self.sink(event)

    def _abort_run(self, reason: str) -> None:
        """Remember the first reason that ends the run."""
        if not self._abort.is_set():
            self._abort_text = reason
            self._abort.set()

    def _on_thermal(self, reason: str) -> None:
        self._thermal = reason
        self._abort_run(f"junction temperature guard: {reason}")

    # -- waiting -------------------------------------------------------------

    def _nap(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; True when cancel or abort woke us."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return False
            if self._abort.is_set() or self.cancel.is_set():
                return True
            self.cancel.wait(min(left, self.limits.poll_s))

    def _noise(self) -> list[str]:
        return noisy_reasons(
            self.owner,
            proc_root=self.env.proc_root,
            quiet_dir=self.env.quiet_dir,
            boot_lock=self.env.boot_lock,
        )

    def _wait_quiet(self) -> list[str]:
        """Wait for a quiet box; return the reasons that remain after the wait.

        With ``force_noisy`` the reasons are returned at once (the cell runs
        and is marked invalid). Returns early when cancelled or aborted.
        """
        reasons = self._noise()
        if not reasons or self.force_noisy:
            return reasons
        self._ui("guard", "waiting for a quiet box: " + "; ".join(reasons))
        deadline = time.monotonic() + self.limits.noisy_wait_s
        while reasons and time.monotonic() < deadline:
            if self._nap(max(self.limits.poll_s, 1.0)):
                return reasons
            reasons = self._noise()
        return reasons

    # -- quiet window --------------------------------------------------------

    def _open_window(self) -> None:
        self._window = QuietWindow(self.owner, quiet_dir=self.env.quiet_dir)
        self._window.__enter__()
        self._window_n += 1
        self.out.windows = self._window_n

    def _close_window(self) -> None:
        if self._window is not None:
            self._window.__exit__(None, None, None)
            self._window = None

    def _rotate_window(self) -> None:
        """Leave the quiet window after its cap and re-enter it after the gap.

        Box rule: a measured window lasts at most 15 minutes with a gap of at
        least 2 minutes before the next one. The probe process stays idle.
        """
        window = self._window
        if window is None or window.age_s < self.limits.window_cap_s:
            return
        self._close_window()
        self._ui(
            "guard",
            f"quiet window ended after {window.age_s / 60:.0f} min: pausing "
            f"{self.limits.window_gap_s:.0f} s",
        )
        if self._nap(self.limits.window_gap_s):
            return
        reasons = self._wait_quiet()
        if reasons and not self.force_noisy:
            self._abort_run("box not quiet after the window gap: " + "; ".join(reasons))
            return
        self._open_window()

    # -- the process -----------------------------------------------------------

    def _argv(self) -> list[str]:
        argv = [str(self.binary)]
        if not self.plan.needs_peer:
            argv.append("--no-peer")
        argv.append("batch")
        return argv

    def _drain_until(
        self, want: Callable[[parse.Item], bool], timeout_s: float, what: str
    ) -> tuple[parse.Item | None, str]:
        """Feed output lines to the parser until ``want(item)`` matches.

        Returns ``(item, '')`` or ``(None, reason)`` on timeout, abort, cancel,
        a process that exited, or the end of its output.
        """
        probe = self._probe
        assert probe is not None
        deadline = time.monotonic() + timeout_s
        while True:
            if self._temps.missing:
                self._abort_run(
                    "junction temperature sensor unreadable: "
                    + ", ".join(gpu_entity(b) for b in self._temps.missing)
                )
            if self._abort.is_set():
                return None, self._abort_text
            if self.cancel.is_set():
                return None, "cancelled"
            left = deadline - time.monotonic()
            if left <= 0:
                return None, f"timeout after {timeout_s:.0f} s waiting for {what}"
            got, line = probe.get_line(min(left, self.limits.poll_s))
            if not got:
                continue
            if line is None:
                rc = probe.returncode()
                return None, f"probe output ended (rc={rc}) before {what}"
            for item in self._parser.feed(line):
                if want(item):
                    return item, ""

    def _start_probe(self) -> bool:
        env = gpu_env(self.plan.hip_visible)
        self._probe = ProbeProcess(self._argv(), env)
        try:
            self._probe.start()
        except OSError as exc:
            self.out.status = "failed"
            self.out.error = f"cannot start {self.binary}: {exc}"
            return False
        item, why = self._drain_until(
            lambda i: isinstance(i, parse.Ready), self.limits.start_s, "READY"
        )
        if not isinstance(item, parse.Ready):
            self._finish_failure(why)
            return False
        self.out.tick_mhz = list(item.tick_mhz)
        self.out.probe_version = item.version
        return True

    def _finish_failure(self, why: str) -> None:
        """Set the outcome of a run that ended before its plan was done."""
        if self._abort.is_set():
            self.out.status = "aborted"
            self.out.abort_reason = self._abort_text
        elif self.cancel.is_set():
            self.out.status = "aborted"
            self.out.abort_reason = "cancelled"
        else:
            self.out.status = "failed"
            self.out.error = why
        if self._probe is not None:
            self._probe.kill(self.limits.kill_grace_s)
            self.out.stderr_tail = self._probe.stderr_tail()

    # -- evictions -------------------------------------------------------------

    def _evictions(self, probe_pid: int) -> list[str]:
        """Reasons from KFD counters since the previous check (invalidating)."""
        now = self._evict.snapshot()
        reasons: list[str] = []
        before = self._evict_base
        deltas = self._evict.diff(before, now)
        for pid, per_bus in sorted(deltas.items()):
            for bus, delta in sorted(per_bus.items()):
                if delta == 0:
                    continue
                proc = now[pid]
                reasons.append(
                    f"evicted_ms jump {delta:+d} ms of pid {pid} ({proc.comm}) on "
                    f"{gpu_entity(bus)}"
                )
                self._evicted_ms.setdefault(pid, {})
                self._evicted_ms[pid][bus] = self._evicted_ms[pid].get(bus, 0) + delta
        for pid in sorted(set(now) - set(before) - {probe_pid}):
            proc = now[pid]
            reasons.append(
                f"new KFD process on the probed GPUs: pid {pid} ({proc.comm})"
            )
            self._evict_first.setdefault(pid, proc)
        for pid, proc in now.items():
            self._evict_first.setdefault(pid, proc)
        self._evict_base = now
        return reasons

    def _eviction_summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pid, per_bus in sorted(self._evicted_ms.items()):
            proc = self._evict_base.get(pid) or self._evict_first.get(pid)
            out.append(
                {
                    "pid": pid,
                    "comm": proc.comm if proc else "",
                    "cmdline": proc.cmdline if proc else "",
                    "evicted_ms_delta": {
                        gpu_entity(b): ms for b, ms in per_bus.items()
                    },
                }
            )
        return out

    # -- one step --------------------------------------------------------------

    def _convert_context(self, ok: bool, error: str | None) -> ConvertContext:
        return ConvertContext(
            plan=self.plan,
            topo=self.topo,
            model=self.model,
            run_tags=self.tags,
            probe_version=self.out.probe_version,
            ok=ok,
            error=error,
        )

    def _run_step(self, index: int, step: Step) -> bool:
        """Run one step; False when the run cannot continue."""
        probe = self._probe
        assert probe is not None
        noise = self._wait_quiet()
        if self._abort.is_set() or self.cancel.is_set():
            return False
        if noise and not self.force_noisy:
            self._abort_run(
                f"box stayed noisy for {self.limits.noisy_wait_s:.0f} s: "
                + "; ".join(noise)
            )
            return False
        self._rotate_window()
        if self._abort.is_set():
            return False
        self._ui(
            "step_start",
            step.scenario,
            i=index,
            n=len(self.plan.steps),
            step_id=step.id,
            scenario=step.scenario,
            cmd=step.cmd,
            temp_c=self._hot(),
        )
        started = now_ns()
        if not probe.send(step.cmd):
            self._finish_failure(f"probe is gone before {step.id}")
            return False
        item, why = self._drain_until(
            lambda i: isinstance(i, parse.BlockEnd),
            self.limits.step_limit(step),
            f"END of {step.id}",
        )
        ended = now_ns()
        if not isinstance(item, parse.BlockEnd):
            self._finish_failure(why)
            if not self._abort.is_set() and not self.cancel.is_set():
                self.out.error = why
            return False
        block = item.block
        problems = list(noise) if noise else []
        problems += self._evictions(probe.pid)
        error = block_error(block)
        invalid = "; ".join(problems) if problems else None
        ms = convert_block(
            step,
            block,
            self._convert_context(invalid is None, invalid),
            started,
            ended,
        )
        for reason in problems:
            self.out.add_invalid(reason)
        for reason in (r for r in problems if r.startswith("evicted_ms")):
            self._trace(
                _event(
                    "evicted_jump",
                    reason,
                    severity="warn",
                    step_id=step.id,
                    run=self.run_id,
                )
            )
        for m in ms:
            self.out.measurements.append(m)
            if self.sink is not None:
                self.sink(m)
            self._ui(
                "measurement",
                describe(m),
                i=index,
                n=len(self.plan.steps),
                step_id=step.id,
                scenario=step.scenario,
            )
        if error is not None or not ms:
            why_failed = error or "no result rows"
            self.out.steps_failed.append(
                {"step": step.id, "scenario": step.scenario, "error": why_failed}
            )
        self.out.steps_done += 1
        self._ui(
            "step_done",
            step.scenario,
            i=index,
            n=len(self.plan.steps),
            step_id=step.id,
            scenario=step.scenario,
            ok=error is None and bool(ms),
            error=error or "",
            temp_c=self._hot(),
        )
        return True

    def _hot(self) -> float | None:
        """Highest last junction reading of the probed GPUs (None before any)."""
        values = list(self._temps.last_c.values())
        return max(values) if values else None

    # -- the run -----------------------------------------------------------------

    def run(self) -> RunOutcome:
        """Run the whole plan; never raises for probe or guard problems."""
        self.out.started_ns = now_ns()
        self._trace(
            _event(
                "measure_start",
                f"{self.plan.suite} on {' '.join(self.plan.entities)}",
                run=self.run_id,
                suite=self.plan.suite,
                gpus=self.plan.entities,
                steps=len(self.plan.steps),
                est_s=round(self.plan.est_total_s, 1),
                owner=self.owner,
                force_noisy=self.force_noisy,
                binary=str(self.binary),
            )
        )
        self._ui(
            "plan",
            f"{self.plan.suite}: {len(self.plan.steps)} steps",
            n=len(self.plan.steps),
            est_s=round(self.plan.est_total_s, 1),
            gpus=self.plan.entities,
        )
        try:
            self._run()
        except KeyboardInterrupt:
            self._abort_run("interrupted (Ctrl-C)")
            self.out.status = "aborted"
            self.out.abort_reason = self._abort_text
        finally:
            self._cleanup()
        self._finalize()
        return self.out

    def _run(self) -> None:
        self._temps.check_once()
        if self._temps.missing:
            self.out.status = "failed"
            self.out.error = (
                "no readable junction temperature sensor for "
                + ", ".join(gpu_entity(b) for b in self._temps.missing)
                + ": the temperature guard cannot run"
            )
            return
        if self._abort.is_set():
            self.out.status = "aborted"
            self.out.abort_reason = self._abort_text
            return
        self._open_window()
        self._temps.start()
        self._links.start()
        if not self._start_probe():
            return
        assert self._probe is not None
        self._evict_base = self._evict.snapshot()
        self._evict_first = dict(self._evict_base)
        deadline = time.monotonic() + self.limits.run_limit(self.plan)
        for index, step in enumerate(self.plan.steps, 1):
            if self.cancel.is_set():
                self.out.status = "aborted"
                self.out.abort_reason = "cancelled"
                return
            if time.monotonic() > deadline:
                self._abort_run(
                    f"run time limit {self.limits.run_limit(self.plan):.0f} s reached"
                )
            if self._abort.is_set():
                self.out.status = "aborted"
                self.out.abort_reason = self._abort_text
                return
            if not self._run_step(index, step):
                if self.out.status == "ok":
                    self.out.status = "aborted" if self._abort.is_set() else "failed"
                if self._abort.is_set() and not self.out.abort_reason:
                    self.out.abort_reason = self._abort_text
                return

    def _cleanup(self) -> None:
        probe = self._probe
        if probe is not None:
            if self.out.status in ("aborted", "failed") or self._abort.is_set():
                probe.kill(self.limits.kill_grace_s)
            else:
                probe.close(self.limits.close_grace_s)
            self.out.stderr_tail = probe.stderr_tail()
        self._temps.stop()
        self._links.stop()
        self._close_window()

    def _finalize(self) -> None:
        out = self.out
        out.ended_ns = now_ns()
        out.max_temp_c = {gpu_entity(b): t for b, t in self._temps.max_c.items()}
        out.evicted = self._eviction_summary()
        out.degraded_links = self._links.degraded()
        if self._thermal:
            out.add_invalid("run aborted by the junction temperature guard")
            self._trace(
                _event(
                    "thermal",
                    self._thermal,
                    severity="error",
                    run=self.run_id,
                    max_temp_c=out.max_temp_c,
                )
            )
        if out.status == "ok" and out.steps_failed:
            out.status = "partial" if out.measurements else "failed"
        if out.status == "failed" and not out.error and not out.measurements:
            out.error = out.error or "no measurement was produced"
        if out.status == "aborted":
            self._ui("abort", out.abort_reason, reason=out.abort_reason)
        invalid = (
            f", INVALID: {'; '.join(out.invalid_reasons)}" if not out.valid else ""
        )
        text = (
            f"{out.status}: {len(out.measurements)} results, "
            f"{out.steps_done}/{out.steps_total} steps{invalid}"
        )
        self._ui("done", text, status=out.status, valid=out.valid)
        self._trace(
            _event(
                "measure_end",
                text,
                severity="info" if out.status == "ok" and out.valid else "warn",
                run=self.run_id,
                status=out.status,
                valid=out.valid,
                invalid_reasons=out.invalid_reasons,
                steps_done=out.steps_done,
                steps_total=out.steps_total,
                results=len(out.measurements),
                max_temp_c=out.max_temp_c,
                evicted=out.evicted,
                abort_reason=out.abort_reason,
                error=out.error,
                duration_s=round((out.ended_ns - out.started_ns) / 1e9, 2),
            )
        )


__all__ = [
    "DEFAULT_BINARY",
    "ENV_BINARY",
    "GuardEnv",
    "ProbeProcess",
    "RunLimits",
    "RunOutcome",
    "Runner",
    "check_bus_order",
    "cpu_only_env",
    "default_binary",
    "describe",
    "gpu_env",
    "new_run_id",
    "probe_buses",
    "probe_version",
]
