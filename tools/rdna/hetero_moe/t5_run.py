# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T5 driver: golden runs, the hub replay against real servers, fault injection.

No torch at module level (the launcher is imported when servers are started).
Run it from the repository root with ``PYTHONPATH=<tree>``; every GPU process it
starts gets its own ``HIP_VISIBLE_DEVICES``.

``golden``  one ``t5_golden.py`` subprocess per expert server of the plan (on that
            server's HIP device, sequentially unless ``--parallel``):
            ``DIR/golden_<srv>.npz`` and ``DIR/golden_<srv>.log``.
``run``     start the servers (``Launcher``), run ``t5_hub_replay.py`` for the
            requested modes on the hub's HIP device (``--hub-hip`` overrides it:
            V620 stand-in), then check that the hub passed, the servers exited 0
            after the hub closed and ``/dev/shm/hx_<plan_id>`` has no ``srv_*``,
            ``ack_*``, ``hub_*``, ``ring_*`` file left. Bench gates (spec 11.2):
            the byte-exact checks and the fault cases are (a); (b) the server
            chain, real minus synthetic-server layer time <= 1.15 x n_max x 6.67 us,
            is judged only with ``--baseline`` (the ``t5_run.json`` of ``run
            --synthetic --modes bench`` on the same plan and widths); (c) the
            layer time against the plan-specific 1.3 x model is only printed.
``faults``  synthetic servers (no weights), one fresh start per case: ``kill-hub``,
            ``kill-server``, ``freeze-server``, ``freeze-hub``, ``wrong-plan``; the
            wall time from the injection to the hub's and every server's exit must
            be <= ``--limit`` seconds (5.0) with the expected exit codes. The hub of
            ``freeze-hub`` runs eager steps and SIGSTOPs itself between two layers
            when the driver creates its trigger file (see ``run_fault``).

Every check prints ``T5 <tag> PASS|FAIL <details>``; JSON results go to
``<out>/t5_<command>.json``. Exit status 0 only when every check passed.
"""

import argparse
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tools.rdna.hetero_moe import t5_common as C

REPO = Path(__file__).resolve().parents[3]
SHM_ROOT = "/dev/shm"
LEFTOVER_PREFIXES = ("srv_", "ack_", "hub_", "ring_")
FAULT_CASES = ("kill-hub", "kill-server", "freeze-server", "freeze-hub", "wrong-plan")
FAULT_LIMIT_S = 5.0
HUB_MODULE = "tools.rdna.hetero_moe.t5_hub_replay"
GOLDEN_MODULE = "tools.rdna.hetero_moe.t5_golden"
PHASE_COMBINE = 2  # hx_proto.HX_PH_COMBINE
SIGKILL_RC = -int(signal.SIGKILL)


# ---- plan access without torch ------------------------------------------------------


def plan_module():
    """``vllm/distributed/hetero_moe/plan.py`` loaded by path: it needs numpy and
    regex only, importing the vllm package would pull in torch."""
    key = "_t5_hetero_plan"
    if key not in sys.modules:
        path = REPO / "vllm" / "distributed" / "hetero_moe" / "plan.py"
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def wrong_plan_doc(doc: dict, compute_plan_id, how: str = "hip") -> dict:
    """A plan document whose ``plan_id`` differs from ``doc``'s, still valid.

    ``hip``: the last device's ``hip`` index changes; ``owners``: layer 0 swaps the
    owner lists of experts 0 and 1 (same device loads, other placement). The id is
    recomputed with ``compute_plan_id`` (``plan.compute_plan_id``)."""
    out = copy.deepcopy(doc)
    if how == "hip":
        out["devices"][-1]["hip"] = max(d["hip"] for d in doc["devices"]) + 100
    elif how == "owners":
        owners = out["layers"][0]["owners"]
        owners[0], owners[1] = owners[1], owners[0]
    else:
        raise ValueError(f"unknown variant {how!r}: hip or owners")
    out["plan_id"] = compute_plan_id(out)
    if out["plan_id"] == doc["plan_id"]:
        raise ValueError(f"variant {how!r} did not change the plan id")
    return out


# ---- processes and files -----------------------------------------------------


def leftover_files(plan_id: str, root: str | Path = SHM_ROOT) -> list[str]:
    """``srv_*``, ``ack_*``, ``hub_*``, ``ring_*`` files in ``<root>/hx_<plan_id>``."""
    directory = Path(root) / f"hx_{plan_id}"
    if not directory.is_dir():
        return []
    return sorted(
        f.name for f in directory.iterdir() if f.name.startswith(LEFTOVER_PREFIXES)
    )


def await_exits(
    procs: dict[str, subprocess.Popen],
    t0: float,
    timeout_s: float,
    poll_s: float = 0.01,
) -> dict[str, float | None]:
    """Seconds from ``t0`` (a ``time.monotonic()`` value) until each process has
    exited; ``None`` for the ones still running ``timeout_s`` after ``t0``."""
    done: dict[str, float | None] = {}
    pending = dict(procs)
    deadline = t0 + timeout_s
    while pending:
        for name, proc in list(pending.items()):
            if proc.poll() is not None:
                done[name] = time.monotonic() - t0
                del pending[name]
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    done.update(dict.fromkeys(pending))
    return done


def wait_for_text(
    path: Path, needle: str, timeout_s: float, proc: subprocess.Popen | None = None
) -> bool:
    """True once ``needle`` appears in the file; False on timeout or when ``proc``
    exits first."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if needle in path.read_text(errors="replace"):
                return True
        except OSError:
            pass
        if (proc is not None and proc.poll() is not None) or (
            time.monotonic() >= deadline
        ):
            return False
        time.sleep(0.05)


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def child_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + old if old else "")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def spawn(cmd: list[str], log: Path, hip: int) -> subprocess.Popen:
    """A child on its own GPU (``HIP_VISIBLE_DEVICES``), stdout+stderr in ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as out:
        return subprocess.Popen(
            cmd,
            cwd=REPO,
            env=child_env(HIP_VISIBLE_DEVICES=str(hip)),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def kill_quiet(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait()


def parse_abort(text: str) -> tuple[int, int] | None:
    """``(phase, peer)`` of the last ``HX_ABORT ... phase=<n> peer=<k>`` line."""
    found = None
    for line in text.splitlines():
        if not line.startswith("HX_ABORT"):
            continue
        fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
        try:
            found = (int(fields["phase"]), int(fields["peer"]))
        except (KeyError, ValueError):
            found = None
    return found


def parse_waited(text: str) -> float | None:
    """Seconds the hub waited for the servers, from ``T5 connect FAIL ... waited
    <x> s`` (measured inside the hub, after the interpreter start-up)."""
    for line in text.splitlines():
        if line.startswith("T5 connect FAIL") and "waited " in line:
            word = line.split("waited ", 1)[1].split()[0]
            try:
                return float(word)
            except ValueError:
                return None
    return None


# ---- fault verdicts (pure) ---------------------------------------------------


@dataclass
class Observation:
    """What one fault case saw. Times are seconds from the injection (or, for
    ``wrong-plan``, from the start of the hub process)."""

    case: str
    victim: str | None = None  # server device id
    victim_index: int | None = None
    hub_rc: int | None = None
    hub_exit_s: float | None = None
    server_rc: dict[str, int | None] = field(default_factory=dict)
    server_exit_s: dict[str, float | None] = field(default_factory=dict)
    servers_alive_after_hub: dict[str, bool] = field(default_factory=dict)
    hub_log: str = ""
    server_logs: dict[str, str] = field(default_factory=dict)
    leftover_procs: list[str] = field(default_factory=list)
    connect_timeout_s: float = 20.0


def judge(obs: Observation, limit_s: float = FAULT_LIMIT_S) -> list[str]:
    """The reasons a case failed (empty: pass)."""
    why: list[str] = []

    def hub_within(rc_want: int) -> None:
        if obs.hub_rc != rc_want:
            why.append(f"hub exit code {obs.hub_rc}, expected {rc_want}")
        if obs.hub_exit_s is None or obs.hub_exit_s > limit_s:
            why.append(f"hub exit after {obs.hub_exit_s} s, limit {limit_s} s")

    def servers_within(names, codes: tuple[int, ...]) -> None:
        for name in names:
            rc, t = obs.server_rc.get(name), obs.server_exit_s.get(name)
            if rc not in codes:
                why.append(f"server {name} exit code {rc}, expected one of {codes}")
            if t is None or t > limit_s:
                why.append(f"server {name} exit after {t} s, limit {limit_s} s")

    everyone = list(obs.server_rc)
    others = [s for s in everyone if s != obs.victim]
    abort = parse_abort(obs.hub_log)
    if obs.case == "kill-hub":
        servers_within(everyone, (3,))
    elif obs.case == "kill-server":
        hub_within(3)
        if abort is None:
            why.append("hub printed no HX_ABORT line")
        if obs.server_rc.get(obs.victim or "") != SIGKILL_RC:
            why.append(
                f"victim {obs.victim} exit code {obs.server_rc.get(obs.victim or '')}"
            )
        servers_within(others, (0, 3))
    elif obs.case == "freeze-server":
        hub_within(3)
        if abort is None:
            why.append("hub printed no HX_ABORT line")
        elif abort != (PHASE_COMBINE, obs.victim_index):
            why.append(
                f"HX_ABORT phase/peer {abort}, expected "
                f"{(PHASE_COMBINE, obs.victim_index)} (combine wait, the frozen server)"
            )
        servers_within(others, (0, 3))
    elif obs.case == "freeze-hub":
        servers_within(everyone, (3,))
        for name in everyone:
            if "(phase 3)" not in obs.server_logs.get(name, ""):
                why.append(f"server {name} log has no 'phase 3' (request wait) abort")
    elif obs.case == "wrong-plan":
        if obs.hub_rc in (None, 0):
            why.append(f"hub exit code {obs.hub_rc}, expected a failure")
        waited = parse_waited(obs.hub_log)
        if waited is None:
            why.append("hub printed no 'T5 connect FAIL ... waited <s>' line")
        elif waited > obs.connect_timeout_s + 5.0:
            why.append(
                f"hub waited {waited} s, connect timeout {obs.connect_timeout_s}"
            )
        alive = [s for s, up in obs.servers_alive_after_hub.items() if not up]
        if alive:
            why.append(f"servers {alive} exited when the wrong-plan hub failed")
        for name in everyone:
            if obs.server_rc.get(name) != 0:
                why.append(
                    f"server {name} exit code {obs.server_rc.get(name)} after SIGTERM"
                )
    else:
        why.append(f"unknown case {obs.case!r}")
    if obs.leftover_procs:
        why.append(f"processes still alive: {obs.leftover_procs}")
    return why


# ---- argument parsing --------------------------------------------------------


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True, help="directory for logs and results")
    p.add_argument(
        "--hub-hip", type=int, default=None, help="override the plan's hub HIP"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--spill-k", default=None, help='override the plan K, "3:10,24:20"')


def add_servers(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--buses", default=None, help="expected PCI bus per server, hex list"
    )
    p.add_argument("--wait-ms", type=int, default=2000)
    p.add_argument("--ready-timeout", type=float, default=3600.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("golden", help="golden file of every server")
    add_common(g)
    g.add_argument("--model", required=True)
    g.add_argument("--widths", default="3,24,48,2048")
    g.add_argument("--patterns", type=int, default=C.DEFAULT_PATTERNS)
    g.add_argument("--parallel", action="store_true")
    g.add_argument("--timeout", type=float, default=7200.0, help="per server, s")

    r = sub.add_parser("run", help="servers + hub replay")
    add_common(r)
    add_servers(r)
    r.add_argument("--model", default=None)
    r.add_argument(
        "--synthetic",
        action="store_true",
        help="weight-free servers: use it with --modes bench to make the baseline "
        "of gate (b); verify/graph need real servers",
    )
    r.add_argument(
        "--baseline",
        default=None,
        help="t5_run.json of an earlier `run --synthetic --modes bench` of the same "
        "plan and widths: enables bench gate (b), the server chain (real minus "
        "synthetic layer time <= 1.15 x n_max x 6.67 us); without it (b) is "
        "printed as not judged",
    )
    r.add_argument("--golden-dir", default=None, help="default: --out")
    r.add_argument("--modes", default="verify,graph,bench")
    r.add_argument("--widths", default=None)
    r.add_argument("--patterns", type=int, default=C.DEFAULT_PATTERNS)
    r.add_argument("--layers", type=int, default=None)
    r.add_argument("--iters", type=int, default=None)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--seconds", type=float, default=0.0)
    r.add_argument("--buckets", default="3,6,12,24,48")
    r.add_argument("--connect-timeout", type=float, default=300.0)
    r.add_argument("--hub-timeout", type=float, default=7200.0)

    f = sub.add_parser("faults", help="failure injection, synthetic servers")
    add_common(f)
    add_servers(f)
    f.add_argument("--cases", default=",".join(FAULT_CASES))
    f.add_argument("--victim", type=int, default=1, help="server index to kill/freeze")
    f.add_argument("--limit", type=float, default=FAULT_LIMIT_S)
    f.add_argument("--connect-timeout", type=float, default=20.0)
    f.add_argument("--width", type=int, default=3)
    return ap


# ---- plan / launcher helpers -------------------------------------------------


def load_plan(path: str):
    return plan_module().load_plan(path)


def hub_of(plan, hub_id: str | None = None) -> tuple[int, str]:
    dev = plan.device_index(hub_id) if hub_id else plan.hubs(0)[0]
    return dev, plan.device_id(dev)


def start_servers(args, plan, log_dir: Path, *, synthetic: bool, buckets: list[int]):
    """Start every expert server of the plan and wait until all are ready."""
    from vllm.distributed.hetero_moe.launcher import Launcher

    launcher = Launcher(
        plan,
        log_dir=log_dir,
        model_dir=getattr(args, "model", None),
        synthetic=synthetic,
        buses=args.buses.split(",") if args.buses else None,
        buckets=buckets,
        wait_ms=args.wait_ms,
        env=child_env(),
    )
    launcher.start()
    try:
        launcher.wait_ready(args.ready_timeout)
    except BaseException:
        launcher.stop()
        raise
    return launcher


def hub_command(args, plan_path: str, hub_id: str, mode: str, out_json: Path | None):
    cmd = [
        sys.executable,
        "-m",
        HUB_MODULE,
        "--plan",
        plan_path,
        "--hub",
        hub_id,
        "--mode",
        mode,
        "--seed",
        str(args.seed),
        "--wait-ms",
        str(args.wait_ms),
        "--connect-timeout",
        str(args.connect_timeout),
    ]
    if args.spill_k:
        cmd += ["--spill-k", args.spill_k]
    if out_json is not None:
        cmd += ["--out", str(out_json)]
    return cmd


# ---- golden ------------------------------------------------------------------


def cmd_golden(args) -> int:
    plan = load_plan(args.plan)
    _, hub_id = hub_of(plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    servers = [plan.device_id(d) for d in plan.experts(plan.island_of(plan.hubs(0)[0]))]
    jobs = []
    for srv in servers:
        dev = plan.device_index(srv)
        cmd = [
            sys.executable,
            "-m",
            GOLDEN_MODULE,
            "--plan",
            args.plan,
            "--model",
            args.model,
            "--server",
            srv,
            "--hub",
            hub_id,
            "--out",
            str(out),
            "--widths",
            args.widths,
            "--seed",
            str(args.seed),
            "--patterns",
            str(args.patterns),
        ]
        if args.spill_k:
            cmd += ["--spill-k", args.spill_k]
        jobs.append((srv, cmd, int(plan.devices[dev]["hip"])))

    results, ok = [], True

    def finish(srv: str, proc: subprocess.Popen, t0: float) -> None:
        nonlocal ok
        try:
            rc = proc.wait(args.timeout)
        except subprocess.TimeoutExpired:
            kill_quiet(proc)
            rc = None
        path = C.golden_path(out, srv)
        problem = None
        if rc != 0:
            problem = f"exit code {rc}, see {out / f'golden_{srv}.log'}"
        else:
            try:
                C.load_golden(path, plan_id=plan.plan_id, server_id=srv)
            except C.GoldenError as exc:
                problem = str(exc)
        ok &= C.emit(
            f"golden {srv}",
            problem is None,
            problem or f"{path} ({time.monotonic() - t0:.0f} s)",
        )
        results.append({"server": srv, "rc": rc, "file": str(path), "problem": problem})

    if args.parallel:
        started = [
            (srv, spawn(cmd, out / f"golden_{srv}.log", hip), time.monotonic())
            for srv, cmd, hip in jobs
        ]
        for srv, proc, t0 in started:
            finish(srv, proc, t0)
    else:
        for srv, cmd, hip in jobs:
            t0 = time.monotonic()
            finish(srv, spawn(cmd, out / f"golden_{srv}.log", hip), t0)
    (out / "t5_golden.json").write_text(json.dumps(results, indent=1))
    return 0 if ok else 1


# ---- run ---------------------------------------------------------------------


def cmd_run(args) -> int:
    if args.synthetic and args.baseline:
        print(
            "--baseline is for real servers: a synthetic run is the baseline",
            file=sys.stderr,
        )
        return C.EXIT_SETUP
    plan = load_plan(args.plan)
    hub_dev, hub_id = hub_of(plan)
    hub_hip = (
        args.hub_hip if args.hub_hip is not None else int(plan.devices[hub_dev]["hip"])
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    ok = True

    def check(tag: str, good: bool, details: str = "") -> bool:
        nonlocal ok
        ok &= C.emit(tag, good, details)
        checks.append({"tag": tag, "ok": good, "details": details})
        return good

    launcher = hub = None
    baseline = str(Path(args.baseline).resolve()) if args.baseline else None
    hub_json = out / "hub_result.json"
    hub_log = out / "hub.log"
    try:
        try:
            launcher = start_servers(
                args,
                plan,
                out / "servers",
                synthetic=args.synthetic,
                buckets=C.parse_int_list(args.buckets),
            )
        except Exception as exc:  # noqa: BLE001 - reported as a failed check
            check("run servers ready", False, f"{type(exc).__name__}: {exc}")
            return C.EXIT_SETUP
        check(
            "run servers ready", True, ", ".join(h.device_id for h in launcher.handles)
        )
        cmd = hub_command(args, args.plan, hub_id, args.modes, hub_json)
        cmd += [
            "--golden-dir",
            args.golden_dir or args.out,
            "--patterns",
            str(args.patterns),
        ]
        for flag, value in (
            ("--widths", args.widths),
            ("--layers", args.layers),
            ("--iters", args.iters),
            ("--warmup", args.warmup),
            ("--seconds", args.seconds),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        if args.synthetic:
            cmd += ["--synthetic-servers"]
        if baseline:
            cmd += ["--baseline", baseline]
        hub = spawn(cmd, hub_log, hub_hip)
        try:
            hub_rc = hub.wait(args.hub_timeout)
        except subprocess.TimeoutExpired:
            kill_quiet(hub)
            hub_rc = None
        print(read_text(hub_log), end="", flush=True)
        check(
            "run hub",
            hub_rc == 0,
            f"exit code {hub_rc} on HIP {hub_hip}, log {hub_log}",
        )
        exits = await_exits(
            {h.device_id: h.proc for h in launcher.handles}, time.monotonic(), 15.0
        )
        for h in launcher.handles:
            rc = h.proc.poll()
            check(
                f"run server {h.device_id}",
                rc == 0 and exits[h.device_id] is not None,
                f"exit code {rc} after the hub closed ({exits[h.device_id]} s)",
            )
        launcher.stop()
        left = leftover_files(plan.plan_id)
        check(
            "run leftovers", not left, f"/dev/shm/hx_{plan.plan_id}: {left or 'none'}"
        )
        if left:
            launcher.cleanup_hubs()
    finally:
        kill_quiet(hub)
        if launcher is not None:
            launcher.stop()
    (out / "t5_run.json").write_text(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "synthetic": args.synthetic,
                "baseline": baseline,
                "checks": checks,
                "hub": json.loads(read_text(hub_json) or "null"),
            },
            indent=1,
        )
    )
    return 0 if C.emit("run", ok) else 1


# ---- faults ------------------------------------------------------------------


def run_fault(case: str, args, plan, out: Path) -> Observation:
    hub_dev, hub_id = hub_of(plan)
    hub_hip = (
        args.hub_hip if args.hub_hip is not None else int(plan.devices[hub_dev]["hip"])
    )
    case_dir = out / case
    launcher = start_servers(
        args, plan, case_dir / "servers", synthetic=True, buckets=[args.width]
    )
    obs = Observation(case, connect_timeout_s=args.connect_timeout)
    names = [h.device_id for h in launcher.handles]
    procs = {h.device_id: h.proc for h in launcher.handles}
    hub = None
    hub_log = case_dir / "hub.log"
    try:
        plan_path = args.plan
        wrong_id = None
        if case == "wrong-plan":
            doc = wrong_plan_doc(plan.doc, plan_module().compute_plan_id)
            wrong_id = doc["plan_id"]
            plan_path = str(case_dir / "plan_wrong.json")
            Path(plan_path).write_text(json.dumps(doc))
        cmd = hub_command(args, plan_path, hub_id, "serve", None)
        cmd += ["--widths", str(args.width)]
        trigger = case_dir / "freeze.trigger"
        if case == "freeze-hub":
            # An external SIGSTOP cannot leave the servers waiting mid-step: the
            # hub GPU keeps running what was enqueued and an idle server has no
            # timeout. The eager hub therefore stops itself between two layers.
            cmd += ["--serve-eager", "--freeze-trigger", str(trigger)]
        t_hub = time.monotonic()
        hub = spawn(cmd, hub_log, hub_hip)
        if case == "wrong-plan":
            exits = await_exits({"hub": hub}, t_hub, args.connect_timeout + 90.0)
            obs.hub_exit_s = exits["hub"]
            obs.servers_alive_after_hub = {
                n: p.poll() is None for n, p in procs.items()
            }
            obs.hub_rc = hub.poll()
            obs.server_rc = launcher.stop()
        else:
            if not wait_for_text(hub_log, "STEP ", args.ready_timeout, hub):
                raise RuntimeError(
                    f"the hub did not start replaying; log {hub_log}:\n"
                    + read_text(hub_log)[-2000:]
                )
            victim = launcher.handles[args.victim] if args.victim < len(names) else None
            obs.victim = victim.device_id if victim else None
            obs.victim_index = args.victim
            t0 = time.monotonic()
            watch = {"hub": hub, **procs}
            if case == "kill-hub":
                os.kill(hub.pid, signal.SIGKILL)
            elif case == "kill-server":
                os.kill(victim.proc.pid, signal.SIGKILL)
            elif case == "freeze-server":
                os.kill(victim.proc.pid, signal.SIGSTOP)
                watch.pop(obs.victim)
            elif case == "freeze-hub":
                trigger.write_text("freeze\n")  # the hub stops itself mid-step
                watch.pop("hub")
            exits = await_exits(watch, t0, args.limit + 5.0)
            if case == "freeze-server":
                kill_quiet(victim.proc)
            kill_quiet(hub)
            obs.hub_exit_s = exits.get("hub")
            obs.hub_rc = hub.returncode
            obs.server_exit_s = {n: exits.get(n) for n in names}
            obs.server_rc = {n: p.poll() for n, p in procs.items()}
        obs.hub_log = read_text(hub_log)
        obs.server_logs = {h.device_id: read_text(h.log) for h in launcher.handles}
    finally:
        kill_quiet(hub)
        launcher.stop()
        launcher.cleanup_hubs()
        if wrong_id is not None:
            # the wrong-plan hub created its own (empty) rendezvous directory
            wrong_dir = Path(SHM_ROOT) / f"hx_{wrong_id}"
            for f in wrong_dir.glob("*"):
                f.unlink(missing_ok=True)
            if wrong_dir.is_dir():
                wrong_dir.rmdir()
    obs.leftover_procs = [
        f"{n} pid {p.pid}" for n, p in procs.items() if p.poll() is None
    ] + ([f"hub pid {hub.pid}"] if hub is not None and hub.poll() is None else [])
    return obs


def cmd_faults(args) -> int:
    plan = load_plan(args.plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in args.cases.replace(" ", "").split(",") if c]
    unknown = [c for c in cases if c not in FAULT_CASES]
    if unknown:
        print(f"unknown case(s) {unknown}; choose from {FAULT_CASES}", file=sys.stderr)
        return C.EXIT_SETUP
    ok, rows, records = True, [], []
    for case in cases:
        try:
            obs = run_fault(case, args, plan, out)
            why = judge(obs, args.limit)
        except Exception as exc:  # noqa: BLE001 - a case that cannot run fails
            obs, why = Observation(case), [f"{type(exc).__name__}: {exc}"]
        fmt = lambda t: "-" if t is None else f"{t:.2f}"  # noqa: E731
        ok &= C.emit(
            f"fault {case}",
            not why,
            f"hub rc={obs.hub_rc} t={fmt(obs.hub_exit_s)} s; servers "
            + " ".join(
                f"{n}:rc={obs.server_rc.get(n)}@{fmt(obs.server_exit_s.get(n))}"
                for n in obs.server_rc
            )
            + ("; " + "; ".join(why) if why else ""),
        )
        slowest = [t for t in obs.server_exit_s.values() if t is not None]
        rows.append(
            [
                case,
                obs.hub_rc if obs.hub_rc is not None else "-",
                fmt(obs.hub_exit_s),
                fmt(max(slowest)) if slowest else "-",
                "PASS" if not why else "FAIL",
            ]
        )
        records.append({**asdict(obs), "failures": why, "ok": not why})
    print(
        C.format_table(("case", "hub_rc", "hub_s", "servers_s_max", "verdict"), rows),
        flush=True,
    )
    (out / "t5_faults.json").write_text(json.dumps(records, indent=1, default=str))
    return 0 if C.emit("faults", ok) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"golden": cmd_golden, "run": cmd_run, "faults": cmd_faults}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
