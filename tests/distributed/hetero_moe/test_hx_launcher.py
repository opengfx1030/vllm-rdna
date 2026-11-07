# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the plan-driven launcher (``launcher.py``) with a fake server
module: a tiny script on ``PYTHONPATH`` that parses the real server CLI, writes
the ready file after a delay and exits on SIGTERM. Real subprocesses, signals and
a rendezvous directory in ``tmp_path``."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hx_fake_ops import tiny_plan, tiny_plan_doc  # noqa: E402

from vllm.distributed.hetero_moe import hx, rdv  # noqa: E402
from vllm.distributed.hetero_moe import launcher as L  # noqa: E402
from vllm.distributed.hetero_moe import server as S  # noqa: E402
from vllm.distributed.hetero_moe.plan import load_plan, plan_from_doc  # noqa: E402

FAKE_SERVER = textwrap.dedent(
    '''
    """Stand-in of vllm.distributed.hetero_moe.server for the launcher tests."""
    import argparse
    import json
    import os
    import signal
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--buckets", default="3,6,12,24,48")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--nslots", type=int)
    ap.add_argument("--wait-ms", type=int)
    ap.add_argument("--begin-wait-ms", type=int)
    ap.add_argument("--hub-timeout", type=float)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--expect-bus", default=None)
    ap.add_argument("--extra", default=None)
    args = ap.parse_args()

    modes = json.loads(os.environ.get("FAKE_HX_MODES", "{}"))
    mode = modes.get(args.device_id, os.environ.get("FAKE_HX_MODE", "ok"))
    state = {"stop": False}


    def on_term(signum, frame):
        state["stop"] = True


    signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore_term" else on_term)
    with open(args.ready_file + ".started", "w") as f:  # handlers are installed
        f.write(str(os.getpid()))
    if mode == "early_exit":
        print("fatal: injected failure on", args.device_id, file=sys.stderr, flush=True)
        sys.exit(2)
    time.sleep(float(os.environ.get("FAKE_HX_DELAY", "0.2")))
    if mode != "never_ready":
        keys = (
            "HIP_VISIBLE_DEVICES",
            "VLLM_HETERO_MOE_ROLE",
            "VLLM_HETERO_MOE_DEVICE_ID",
            "PYTHONUNBUFFERED",
            "MY_MARK",
        )
        doc = {
            "argv": sys.argv[1:],
            "args": vars(args),
            "env": {k: os.environ.get(k) for k in keys},
            "pid": os.getpid(),
        }
        tmp = args.ready_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, args.ready_file)
    while mode == "ignore_term" or not state["stop"]:
        time.sleep(0.05)
    sys.exit(0)
    '''
)


@pytest.fixture
def rdir(tmp_path, monkeypatch):
    root = tmp_path / "shm"

    def hx_dir(plan_id, root_arg=None):
        path = root / f"hx_{plan_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(rdv, "hx_dir", hx_dir)
    return hx_dir


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    mod_dir = tmp_path / "fakemod"
    mod_dir.mkdir()
    (mod_dir / "fake_hx_server.py").write_text(FAKE_SERVER)
    monkeypatch.setattr(L, "SERVER_MODULE", "fake_hx_server")
    return mod_dir


@pytest.fixture
def make(tmp_path, rdir, fake_module):
    """Factory of launchers over the fake server; every process is gone at the end."""
    made: list[L.Launcher] = []

    def factory(plan=None, env_extra=None, **kw) -> L.Launcher:
        env = {**os.environ, "PYTHONPATH": str(fake_module), "MY_MARK": "kept"}
        env.pop("PYTHONUNBUFFERED", None)  # the launcher only sets a default
        env.update(env_extra or {})
        kw.setdefault("synthetic", True)
        kw.setdefault("buckets", (3, 6))
        launcher = L.Launcher(
            plan or tiny_plan(2), log_dir=tmp_path / "logs", env=env, **kw
        )
        made.append(launcher)
        return launcher

    yield factory
    for launcher in made:
        for h in launcher.handles:
            if h.proc.poll() is None:
                h.proc.kill()
            h.proc.wait(30)


def wait_for(pred, timeout: float = 60.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def wait_started(launcher: L.Launcher) -> None:
    """Every fake server has installed its signal handler (a SIGTERM before that
    would kill a still-starting interpreter with -15)."""
    for h in launcher.handles:
        started = Path(str(h.ready) + ".started")
        wait_for(
            lambda started=started, h=h: started.exists() or h.proc.poll() is not None,
            what=f"{h.device_id} to start",
        )


def ready_doc(handle: L.ServerHandle) -> dict:
    return json.loads(handle.ready.read_text())


# ---- construction -----------------------------------------------------------------


def test_targets_follow_the_plan_order_with_their_hip_indices(make):
    launcher = make()
    assert launcher.targets == [("exp0", 1, 0), ("exp1", 2, 0)]
    assert launcher.buses is None and launcher.handles == []
    assert launcher.dir == rdv.hx_dir(launcher.plan.plan_id)


def test_devices_selects_a_subset_in_plan_order(make):
    assert make(devices=["exp1"]).targets == [("exp1", 2, 0)]
    assert make(devices=["exp1", "exp0"]).targets == [
        ("exp0", 1, 0),
        ("exp1", 2, 0),
    ]


def test_an_unknown_device_or_the_hub_is_an_error(make):
    with pytest.raises(L.LauncherError, match=r"no expert device \['exp9'\]"):
        make(devices=["exp0", "exp9"])
    with pytest.raises(L.LauncherError, match=r"no expert device \['hub0'\]"):
        make(devices=["hub0"])  # a hub is not launched


def test_an_empty_selection_is_an_error(make):
    with pytest.raises(L.LauncherError, match="no expert device to launch"):
        make(devices=[])


def test_a_model_directory_is_required_without_synthetic(make):
    with pytest.raises(L.LauncherError, match="checkpoint directory"):
        make(synthetic=False)
    assert make(synthetic=False, model_dir="/ckpt").model_dir == "/ckpt"


def test_buses_must_match_the_number_of_servers(make):
    with pytest.raises(L.LauncherError, match="1 PCI buses for 2 servers"):
        make(buses=["ce"])
    with pytest.raises(L.LauncherError, match="3 PCI buses for 1 servers"):
        make(devices=["exp0"], buses=["ce", "d3", "d8"])
    assert make(buses=["ce", "d3"]).buses == ["ce", "d3"]
    assert make(buses=[]).buses is None  # empty means "do not check"


def test_the_plan_file_is_a_copy_in_the_log_dir_unless_the_plan_has_a_path(
    make, tmp_path
):
    launcher = make()
    copy = tmp_path / "logs" / f"plan_{launcher.plan.plan_id}.json"
    assert launcher.plan_path == copy
    assert json.loads(copy.read_text()) == launcher.plan.doc
    on_disk = tmp_path / "plan.json"
    on_disk.write_text(json.dumps(tiny_plan_doc(2)))
    loaded = load_plan(on_disk)
    assert make(plan=loaded).plan_path == on_disk.resolve()
    gone = plan_from_doc(tiny_plan_doc(2), path=str(tmp_path / "gone.json"))
    assert make(plan=gone).plan_path.parent == tmp_path / "logs"  # path is no file


# ---- command line ------------------------------------------------------------------


def test_command_line(make, tmp_path):
    launcher = make(
        max_rows=64, nslots=3, wait_ms=7, begin_wait_ms=9, hub_timeout_s=5.5
    )
    ready = tmp_path / "ready.json"
    cmd = launcher.command("exp1", ready, "d3")
    assert cmd == [
        sys.executable,
        "-m",
        "fake_hx_server",
        "--plan",
        str(launcher.plan_path),
        "--device-id",
        "exp1",
        "--ready-file",
        str(ready),
        "--buckets",
        "3,6",
        "--max-rows",
        "64",
        "--nslots",
        "3",
        "--wait-ms",
        "7",
        "--begin-wait-ms",
        "9",
        "--hub-timeout",
        "5.5",
        "--synthetic",
        "--expect-bus",
        "d3",
    ]


def test_command_line_options_that_are_optional(make, tmp_path):
    launcher = make(
        synthetic=False,
        model_dir="/ckpt/flash",
        python="/usr/bin/pythonX",
        extra_args=["--no-graphs", "--flush", "1"],
    )
    cmd = launcher.command("exp0", tmp_path / "r.json", None)
    assert cmd[0] == "/usr/bin/pythonX"
    assert "--synthetic" not in cmd and "--expect-bus" not in cmd
    assert cmd[cmd.index("--model") + 1] == "/ckpt/flash"
    assert cmd[-3:] == ["--no-graphs", "--flush", "1"]  # extras come last


def test_the_command_line_is_accepted_by_the_real_server_cli(
    make, tmp_path, monkeypatch
):
    monkeypatch.setattr(L, "SERVER_MODULE", "vllm.distributed.hetero_moe.server")
    launcher = make(
        synthetic=False,
        model_dir="/ckpt",
        max_rows=64,
        nslots=3,
        wait_ms=7,
        begin_wait_ms=9,
        hub_timeout_s=5.5,
        extra_args=["--no-graphs"],
    )
    ready = tmp_path / "r.json"
    cmd = launcher.command("exp1", ready, "d3")
    assert cmd[1:3] == ["-m", "vllm.distributed.hetero_moe.server"]
    args = S._parse(cmd[3:])  # argparse exits on any flag the server lacks
    assert (args.plan, args.device_id, args.ready_file) == (
        str(launcher.plan_path),
        "exp1",
        str(ready),
    )
    assert (args.buckets, args.max_rows, args.nslots) == ("3,6", 64, 3)
    assert (args.wait_ms, args.begin_wait_ms, args.hub_timeout) == (7, 9, 5.5)
    assert args.model == "/ckpt" and args.expect_bus == "d3" and args.no_graphs
    assert not args.synthetic
    synth = make().command("exp0", ready, None)
    assert S._parse(synth[3:]).synthetic


# ---- start / wait_ready / stop -----------------------------------------------------


def test_happy_path_commands_env_exit_codes_and_cleanup(make, tmp_path):
    launcher = make(buses=["ce", "d3"])
    launcher.start()
    assert [h.device_id for h in launcher.handles] == ["exp0", "exp1"]
    assert all(h.proc.pid > 0 for h in launcher.handles)
    launcher.wait_ready(120.0)
    assert launcher.dead() == []
    docs = {h.device_id: ready_doc(h) for h in launcher.handles}
    for dev, hip, bus in (("exp0", 1, "ce"), ("exp1", 2, "d3")):
        doc = docs[dev]
        assert doc["env"] == {
            "HIP_VISIBLE_DEVICES": str(hip),
            "VLLM_HETERO_MOE_ROLE": "expert",
            "VLLM_HETERO_MOE_DEVICE_ID": dev,
            "PYTHONUNBUFFERED": "1",
            "MY_MARK": "kept",  # the caller's environment is passed on
        }
        args = doc["args"]
        assert args["device_id"] == dev and args["expect_bus"] == bus
        assert args["synthetic"] is True and args["model"] is None
        assert args["buckets"] == "3,6" and args["max_rows"] == 2048
        assert args["plan"] == str(launcher.plan_path)
        assert doc["pid"] == next(
            h.proc.pid for h in launcher.handles if h.device_id == dev
        )
    # a server that published, and a hub that left files behind
    d = launcher.dir
    for sid in ("exp0", "exp1"):
        rdv.write_json(hx.srv_path(d, sid), {"pid": docs[sid]["pid"]})
        rdv.write_json(hx.ack_path(d, sid, "hub0"), {"pid": docs[sid]["pid"]})
    rdv.write_json(hx.hub_path(d, "hub0"), {"pid": os.getpid()})
    hx.ring_path(d, "hub0").write_bytes(b"ring")

    codes = launcher.stop(grace_s=60.0)
    assert codes == {"exp0": 0, "exp1": 0}  # SIGTERM -> exit code 0
    for sid in ("exp0", "exp1"):
        assert not hx.srv_path(d, sid).exists()
        assert not hx.ack_path(d, sid, "hub0").exists()
        assert (tmp_path / "logs" / f"server_{sid}.log").exists()
    assert hx.hub_path(d, "hub0").exists()  # the hub's files are the hub's
    launcher.cleanup_hubs()
    assert not hx.hub_path(d, "hub0").exists() and not hx.ring_path(d, "hub0").exists()
    assert launcher.stop(grace_s=5.0) == codes  # stopping again is harmless


def test_a_server_subset_launches_only_that_server(make):
    launcher = make(devices=["exp1"])
    launcher.start()
    launcher.wait_ready(120.0)
    assert [h.device_id for h in launcher.handles] == ["exp1"]
    assert ready_doc(launcher.handles[0])["env"]["HIP_VISIBLE_DEVICES"] == "2"
    wait_started(launcher)
    assert launcher.stop(grace_s=60.0) == {"exp1": 0}


def test_start_removes_an_old_ready_file_and_appends_to_the_log(make, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "ready_exp0.json").write_text('{"stale": true}')
    (logs / "server_exp0.log").write_text("earlier run\n")
    launcher = make(devices=["exp0"], env_extra={"FAKE_HX_MODE": "never_ready"})
    launcher.start()
    assert not (logs / "ready_exp0.json").exists()  # a stale file would pass for ready
    wait_started(launcher)
    launcher.stop(grace_s=60.0)
    assert (logs / "server_exp0.log").read_text().startswith("earlier run\n")


def test_the_context_manager_starts_and_stops(make):
    launcher = make()
    with launcher as entered:
        assert entered is launcher and len(launcher.handles) == 2
        launcher.wait_ready(120.0)
        wait_started(launcher)
    assert [h.proc.returncode for h in launcher.handles] == [0, 0]


def test_early_exit_before_ready_reports_the_code_and_the_log_tail(make):
    launcher = make(env_extra={"FAKE_HX_MODES": json.dumps({"exp1": "early_exit"})})
    launcher.start()
    with pytest.raises(L.LauncherError) as exc:
        launcher.wait_ready(120.0)
    text = str(exc.value)
    assert "server exp1 exited with 2 before it was ready" in text
    assert "fatal: injected failure on exp1" in text  # the tail of its log
    wait_for(lambda: len(launcher.dead()) == 1, what="the early exit")
    assert launcher.dead() == [("exp1", 2)]
    wait_started(launcher)
    assert launcher.stop(grace_s=60.0) == {"exp0": 0, "exp1": 2}


def test_wait_ready_times_out_and_names_the_pending_servers(make):
    launcher = make(env_extra={"FAKE_HX_MODE": "never_ready"})
    launcher.start()
    t0 = time.monotonic()
    with pytest.raises(L.LauncherError) as exc:
        launcher.wait_ready(1.0)
    assert "servers ['exp0', 'exp1'] not ready within 1 s" in str(exc.value)
    assert time.monotonic() - t0 >= 1.0
    wait_started(launcher)
    assert launcher.stop(grace_s=60.0) == {"exp0": 0, "exp1": 0}


def test_wait_ready_returns_at_once_when_nothing_was_started(make):
    make().wait_ready(0.1)  # no handles: nothing to wait for


def test_stop_escalates_to_sigkill_for_a_server_that_ignores_sigterm(make):
    launcher = make(
        env_extra={"FAKE_HX_MODES": json.dumps({"exp0": "ignore_term"})},
    )
    launcher.start()
    launcher.wait_ready(120.0)
    wait_started(launcher)
    t0 = time.monotonic()
    codes = launcher.stop(grace_s=1.0)
    assert codes == {"exp0": -signal.SIGKILL, "exp1": 0}
    assert time.monotonic() - t0 < 60.0
    assert all(h.proc.poll() is not None for h in launcher.handles)


# ---- clean_stale -------------------------------------------------------------------


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(30)
    return proc.pid


def test_clean_stale_removes_dead_files_of_a_previous_run(make):
    launcher = make()
    d = launcher.dir
    rdv.write_json(d / "srv_exp0.json", {"pid": dead_pid()})
    rdv.write_json(d / "hub_hub0.json", {"pid": dead_pid()})
    rdv.write_json(d / "ack_exp0_hub0.json", {"note": "no pid at all"})
    (d / "ring_hub0.bin").write_bytes(b"x" * 100)  # not json: never inspected
    (d / "broken.json").write_text('{"pid": ')
    launcher.clean_stale()
    assert list(d.iterdir()) == []


def test_clean_stale_refuses_when_a_process_of_the_plan_is_alive(make):
    launcher = make()
    d = launcher.dir
    rdv.write_json(d / "a_dead.json", {"pid": dead_pid()})
    rdv.write_json(d / "srv_exp0.json", {"pid": os.getpid()})  # alive: this process
    with pytest.raises(L.LauncherError) as exc:
        launcher.clean_stale()
    text = str(exc.value)
    assert "srv_exp0.json" in text and f"live process {os.getpid()}" in text
    assert launcher.plan.plan_id in text
    assert (d / "srv_exp0.json").exists()  # a live run's file stays
    assert not (d / "a_dead.json").exists()


def test_start_refuses_before_spawning_when_another_run_is_alive(make):
    launcher = make()
    rdv.write_json(launcher.dir / "hub_hub0.json", {"pid": os.getpid()})
    with pytest.raises(L.LauncherError, match="another run of plan"):
        launcher.start()
    assert launcher.handles == []


# ---- helpers -------------------------------------------------------------------------


def test_tail(tmp_path):
    assert L.tail(tmp_path / "missing.log") == "<no log>"
    log = tmp_path / "x.log"
    log.write_text("".join(f"line {i}\n" for i in range(50)))
    assert L.tail(log).splitlines() == [f"line {i}" for i in range(30, 50)]
    assert L.tail(log, lines=2) == "line 48\nline 49"
    log.write_bytes(b"ok \xff\xfe bytes\n")
    assert "bytes" in L.tail(log)  # undecodable bytes do not hide the log
    log.write_text("")
    assert L.tail(log) == ""


def test_launcher_error_is_a_runtime_error():
    assert issubclass(L.LauncherError, RuntimeError)
