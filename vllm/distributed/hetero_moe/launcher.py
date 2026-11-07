# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-driven launcher of the ``hx`` expert servers (spec C.2, C.8 item 4).

One server process per expert device of the plan's islands, each started with
``HIP_VISIBLE_DEVICES=<its HIP index>`` so it sees exactly its own GPU::

    python -m vllm.distributed.hetero_moe.launcher --plan P --model M \
        --log-dir D [--island 0] [--servers exp0,exp1] [--buses ce,d3,d8,db]

The launcher waits for every server's ready file (written after the expert
weights, the known-answer self-test and the graph capture), prints one
``HX_SERVERS_READY`` line and then supervises: SIGTERM/SIGINT or the death of
any server stops all of them (SIGTERM, then SIGKILL after the grace period).
Exit code 0 after a requested stop, 3 when a server died, 4 on a setup failure.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vllm.distributed.hetero_moe import rdv
from vllm.distributed.hetero_moe.hx import ack_path, hub_path, ring_path, srv_path
from vllm.distributed.hetero_moe.plan import load_plan
from vllm.logger import init_logger

logger = init_logger(__name__)

SERVER_MODULE = "vllm.distributed.hetero_moe.server"
FORBIDDEN_HIP = 5  # the second W7800 (bus 07): never started on without approval


class LauncherError(RuntimeError):
    pass


@dataclass
class ServerHandle:
    device_id: str
    hip: int
    island: int
    proc: subprocess.Popen
    log: Path
    ready: Path


def tail(path: Path, lines: int = 20) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "<no log>"
    return "\n".join(text.splitlines()[-lines:])


class Launcher:
    """Starts and supervises the servers of ``plan`` (stage 1: one island)."""

    def __init__(
        self,
        plan,
        *,
        log_dir: str | Path,
        model_dir: str | None = None,
        python: str | None = None,
        islands: Sequence[int] | None = None,
        devices: Sequence[str] | None = None,
        buses: Sequence[str] | None = None,
        buckets: Sequence[int] = (3, 6, 12, 24, 48),
        max_rows: int = 2048,
        nslots: int = 2,
        wait_ms: int = 2000,
        begin_wait_ms: int = 600_000,
        hub_timeout_s: float = 7200.0,
        synthetic: bool = False,
        extra_args: Sequence[str] = (),
        env: dict[str, str] | None = None,
    ):
        self.plan = plan
        self.log_dir = Path(log_dir)
        self.model_dir = model_dir
        self.python = python or sys.executable
        self.buckets = list(buckets)
        self.max_rows = max_rows
        self.nslots = nslots
        self.wait_ms = wait_ms
        self.begin_wait_ms = begin_wait_ms
        self.hub_timeout_s = hub_timeout_s
        self.synthetic = synthetic
        self.extra_args = list(extra_args)
        self.env = dict(os.environ if env is None else env)
        self.handles: list[ServerHandle] = []
        if not synthetic and not model_dir:
            raise LauncherError("a checkpoint directory (--model) is required")
        wanted = list(range(plan.num_islands)) if islands is None else list(islands)
        self.targets: list[tuple[str, int, int]] = []  # (device id, hip, island)
        for isl in wanted:
            for dev in plan.experts(isl):
                self.targets.append(
                    (plan.device_id(dev), int(plan.devices[dev]["hip"]), isl)
                )
        if devices is not None:
            known = {t[0] for t in self.targets}
            missing = [d for d in devices if d not in known]
            if missing:
                raise LauncherError(
                    f"no expert device {missing} in the selected islands"
                )
            self.targets = [t for t in self.targets if t[0] in set(devices)]
        if not self.targets:
            raise LauncherError("the plan has no expert device to launch")
        self.buses = list(buses) if buses else None
        if self.buses is not None and len(self.buses) != len(self.targets):
            raise LauncherError(
                f"{len(self.buses)} PCI buses for {len(self.targets)} servers"
            )
        self.dir = rdv.hx_dir(plan.plan_id)
        self.plan_path = self._plan_file()

    def _plan_file(self) -> Path:
        path = getattr(self.plan, "path", None)
        if path and Path(path).is_file():
            return Path(path).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        copy = self.log_dir / f"plan_{self.plan.plan_id}.json"
        copy.write_text(json.dumps(self.plan.doc))
        return copy

    # ---- rendezvous directory ---------------------------------------------------

    def clean_stale(self) -> None:
        """Remove the files of a previous run; refuse when one of its processes is
        still alive (two runs of one plan would share inboxes)."""
        for path in sorted(self.dir.iterdir()):
            doc = rdv.read_json(path) if path.suffix == ".json" else None
            if doc is not None and rdv.pid_alive(int(doc.get("pid", 0))):
                raise LauncherError(
                    f"{path.name} belongs to the live process {doc['pid']}; "
                    f"another run of plan {self.plan.plan_id} is active in {self.dir}"
                )
            rdv.remove_quiet(path)

    # ---- processes -----------------------------------------------------------------

    def command(self, device_id: str, ready: Path, bus: str | None) -> list[str]:
        cmd = [
            self.python,
            "-m",
            SERVER_MODULE,
            "--plan",
            str(self.plan_path),
            "--device-id",
            device_id,
            "--ready-file",
            str(ready),
            "--buckets",
            ",".join(str(b) for b in self.buckets),
            "--max-rows",
            str(self.max_rows),
            "--nslots",
            str(self.nslots),
            "--wait-ms",
            str(self.wait_ms),
            "--begin-wait-ms",
            str(self.begin_wait_ms),
            "--hub-timeout",
            str(self.hub_timeout_s),
        ]
        if self.model_dir:
            cmd += ["--model", self.model_dir]
        if self.synthetic:
            cmd.append("--synthetic")
        if bus:
            cmd += ["--expect-bus", bus]
        return cmd + self.extra_args

    def start(self) -> None:
        for device_id, hip, _ in self.targets:
            # checked before any process exists
            if not isinstance(hip, int) or isinstance(hip, bool) or hip < 0:
                raise LauncherError(
                    f"{device_id}: HIP index {hip!r} is not a non-negative int"
                )
            if hip == FORBIDDEN_HIP:
                raise LauncherError(f"{device_id}: HIP {hip} is forbidden")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.clean_stale()
        for k, (device_id, hip, island) in enumerate(self.targets):
            env = dict(self.env)
            env.pop("VLLM_TARGET_DEVICE", None)  # a server is a GPU process
            env["HIP_VISIBLE_DEVICES"] = str(hip)
            env["VLLM_HETERO_MOE_ROLE"] = "expert"
            env["VLLM_HETERO_MOE_DEVICE_ID"] = device_id
            env.setdefault("PYTHONUNBUFFERED", "1")
            ready = self.log_dir / f"ready_{device_id}.json"
            ready.unlink(missing_ok=True)
            log = self.log_dir / f"server_{device_id}.log"
            bus = self.buses[k] if self.buses else None
            with open(log, "ab") as out:
                proc = subprocess.Popen(
                    self.command(device_id, ready, bus),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.handles.append(ServerHandle(device_id, hip, island, proc, log, ready))
            logger.info(
                "hx launcher: %s on HIP %d (pid %d), log %s",
                device_id,
                hip,
                proc.pid,
                log,
            )

    def dead(self) -> list[tuple[str, int]]:
        return [
            (h.device_id, h.proc.returncode)
            for h in self.handles
            if h.proc.poll() is not None
        ]

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        pending = {h.device_id: h for h in self.handles}
        while pending:
            for dev, h in list(pending.items()):
                if h.ready.is_file():
                    del pending[dev]
                    continue
                rc = h.proc.poll()
                if rc is not None:
                    raise LauncherError(
                        f"server {dev} exited with {rc} before it was ready:\n"
                        f"{tail(h.log)}"
                    )
            if pending and time.monotonic() >= deadline:
                raise LauncherError(
                    f"servers {sorted(pending)} not ready within {timeout_s:g} s"
                )
            if pending:
                time.sleep(0.5)

    def stop(self, grace_s: float = 15.0) -> dict[str, int | None]:
        """SIGTERM every server, SIGKILL the ones still alive after the grace
        period; removes the rendezvous files of the servers."""
        for h in self.handles:
            if h.proc.poll() is None:
                h.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        for h in self.handles:
            try:
                h.proc.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.error("hx launcher: %s ignored SIGTERM, killing", h.device_id)
                h.proc.kill()
                h.proc.wait()
        for h in self.handles:
            for hub in self.plan.hubs(h.island):
                hub_id = self.plan.device_id(hub)
                rdv.remove_quiet(ack_path(self.dir, h.device_id, hub_id))
            rdv.remove_quiet(srv_path(self.dir, h.device_id))
        return {h.device_id: h.proc.returncode for h in self.handles}

    def cleanup_hubs(self) -> None:
        """Hub-side files of the plan (the hub removes its own on a clean close;
        after a crash they are stale)."""
        for h in self.handles:
            for hub in self.plan.hubs(h.island):
                hub_id = self.plan.device_id(hub)
                rdv.remove_quiet(hub_path(self.dir, hub_id))
                rdv.remove_quiet(ring_path(self.dir, hub_id))

    def __enter__(self) -> "Launcher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m vllm.distributed.hetero_moe.launcher",
        description="Start the hx expert servers of a placement plan.",
    )
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", default=None, help="checkpoint directory")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--python", default=None)
    ap.add_argument("--island", type=int, action="append", default=None)
    ap.add_argument("--servers", default=None, help="comma separated device ids")
    ap.add_argument("--buses", default=None, help="expected PCI bus per server, hex")
    ap.add_argument("--buckets", default="3,6,12,24,48")
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--nslots", type=int, default=2)
    ap.add_argument("--wait-ms", type=int, default=2000)
    ap.add_argument("--begin-wait-ms", type=int, default=600_000)
    ap.add_argument("--hub-timeout", type=float, default=7200.0)
    ap.add_argument("--ready-timeout", type=float, default=3600.0)
    ap.add_argument("--grace", type=float, default=15.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra argument for every server (repeatable)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    stop = {"flag": False}

    def on_signal(signum, _frame):
        logger.info("hx launcher: signal %d, stopping the servers", signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    launcher = None
    try:
        launcher = Launcher(
            load_plan(args.plan),
            log_dir=args.log_dir,
            model_dir=args.model,
            python=args.python,
            islands=args.island,
            devices=args.servers.split(",") if args.servers else None,
            buses=args.buses.split(",") if args.buses else None,
            buckets=[int(b) for b in args.buckets.split(",") if b],
            max_rows=args.max_rows,
            nslots=args.nslots,
            wait_ms=args.wait_ms,
            begin_wait_ms=args.begin_wait_ms,
            hub_timeout_s=args.hub_timeout,
            synthetic=args.synthetic,
            extra_args=args.server_arg,
        )
        launcher.start()
        launcher.wait_ready(args.ready_timeout)
        print(
            "HX_SERVERS_READY "
            + json.dumps(
                {
                    "plan_id": launcher.plan.plan_id,
                    "dir": str(launcher.dir),
                    "servers": [
                        {"id": h.device_id, "hip": h.hip, "pid": h.proc.pid}
                        for h in launcher.handles
                    ],
                }
            ),
            flush=True,
        )
        code = 0
        while not stop["flag"]:
            dead = launcher.dead()
            if dead:
                logger.error("hx launcher: servers exited: %s", dead)
                code = 3
                break
            time.sleep(0.5)
        launcher.stop(args.grace)
        return code
    except (LauncherError, OSError, ValueError) as e:
        logger.error("hx launcher: %s", e)
        if launcher is not None:
            launcher.stop(args.grace)
        return 4


if __name__ == "__main__":
    sys.exit(main())
