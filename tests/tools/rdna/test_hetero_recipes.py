# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hetero-MoE launcher stanza (tools/rdna/rdna_hetero_launch.sh), print mode.

Two halves, neither touches a GPU:

* ``PRINT=1`` dry runs of flashnext-hetero-h1 (hub W7800 + four V620 servers)
  and of flashnext-hetero-h1-v620 (V620 stand-in: hub bus ce, servers d3 d8 db)
  with a stub venv python and the plan CLI run by this interpreter
  (``HETERO_PYTHON``; numpy + regex only). Plans are real stripe plans made by
  ``python vllm/distributed/hetero_moe/cli.py stripe``.
* The supervisor (``rdna_hetero_launch``) driven by stub servers and stub hubs
  injected as command arrays: ready wait, a server dying before ready or under a
  running hub, the hub's exit code, SIGTERM to the supervisor.
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
PLAN_CLI = REPO / "vllm/distributed/hetero_moe/cli.py"
H1 = "flashnext-hetero-h1"
V620 = "flashnext-hetero-h1-v620"
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "VLLM_RDNA3_HIP_OPS",
    "VLLM_CACHE_ROOT",
    "LAUNCH_WRAPPER",
    "RUN_CWD",
    "TP",
)
STUB_RESOLVER = """#!/bin/sh
[ "$1" = -c ] || exit 1
shift 2
out=
for b in "$@"; do
  case $b in
    cb) i=0 ;; ce) i=1 ;; d3) i=2 ;; d8) i=3 ;; db) i=4 ;;
    *) echo "no GPU on PCI bus $b" >&2; exit 1 ;;
  esac
  out="$out${out:+,}$i"
done
echo "$out"
"""


def _stub_venv(root: Path, body: str) -> Path:
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(body)
    python.chmod(0o755)
    return root / "venv"


@pytest.fixture()
def venv(tmp_path):
    """A venv whose python cannot import torch: buses stay unresolved."""
    return _stub_venv(tmp_path, "#!/bin/sh\nexit 1\n")


@pytest.fixture()
def resolving_venv(tmp_path):
    """A venv whose python resolves cb/ce/d3/d8/db to HIP 0..4."""
    return _stub_venv(tmp_path, STUB_RESOLVER)


def _stripe(path: Path, *args: str) -> Path:
    subprocess.run(
        [sys.executable, str(PLAN_CLI), "stripe", "--out", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    return path


@pytest.fixture(scope="module")
def h1_plan(tmp_path_factory):
    return _stripe(tmp_path_factory.mktemp("plans") / "h1.json")


@pytest.fixture(scope="module")
def v620_plan(tmp_path_factory):
    return _stripe(
        tmp_path_factory.mktemp("plans") / "v620.json",
        *("--hub-hips", "1", "--expert-hips", "2,3,4"),
        *("--stripe", "16", "--hub-arch", "gfx1030"),
    )


def serve(venv, recipe, *args, **env):
    full = {
        k: v
        for k, v in os.environ.items()
        if k not in DROPPED
        and not k.startswith(("HETERO_", "VLLM_HETERO_MOE", "VLLM_RDNA"))
    }
    full.update(
        VENV=str(venv),
        MODEL="/nonexistent/model",
        PRINT="1",
        HOME="/home/someone",
        HETERO_PYTHON=sys.executable,
        **env,
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), f"RECIPE={recipe}", *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=120,
    )


def server_commands(out):
    return [
        line
        for line in out.splitlines()
        if line.startswith("env ") and "hetero_moe.server" in line
    ]


def command_of(out, device_id):
    (line,) = [c for c in server_commands(out) if f"--device-id {device_id} " in c]
    return line


def test_h1_hub_is_tp1_on_its_own_bus(venv, h1_plan):
    result = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='cb' (not resolvable on this host" in out
    assert "'--tensor-parallel-size' '1'" in out
    assert "--enable-expert-parallel" not in out
    assert '"num_speculative_tokens":2' in out
    assert '"cudagraph_capture_sizes":[3,6,12,24]' in out
    assert "'--kv-cache-memory-bytes' '10737418240'" in out
    assert "--attention-backend" not in out
    assert (
        "export VLLM_CACHE_ROOT='/home/someone/.cache/vllm-flashnext-hetero-h1'" in out
    )
    assert "export VLLM_RDNA_AR='0'" in out
    assert "export VLLM_RDNA2_W4A8_SDOT4='0'" in out
    assert "export VLLM_RDNA3_HIP_OPS=''" in out
    assert "export VLLM_RDNA3_MOE_DETERMINISTIC='1'" in out
    assert "'--port' '18099'" in out


def test_h1_hub_env_names_the_hub_and_the_plan(venv, h1_plan):
    out = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}").stdout
    assert "export VLLM_HETERO_MOE='1'" in out
    assert "export VLLM_HETERO_MOE_ROLE='hub'" in out
    assert "export VLLM_HETERO_MOE_DEVICE_ID='hub0'" in out
    assert f"export VLLM_HETERO_MOE_PLAN='{h1_plan}'" in out
    assert "export VLLM_HETERO_MOE_SERVERS='ce,d3,d8,db'" in out
    assert "export VLLM_HETERO_MOE_WAIT_MS='2000'" in out
    assert "export VLLM_HETERO_MOE_STATS='0'" in out


def test_h1_stanza_lists_four_servers_before_the_environment(venv, h1_plan):
    out = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}").stdout
    assert out.index("# hetero:") < out.index("# --- environment ---")
    assert "1 hub(s), 4 expert device(s)" in out
    for dev, bus in zip(("exp0", "exp1", "exp2", "exp3"), ("ce", "d3", "d8", "db")):
        assert re.search(rf"# hetero:\s+server {dev} island 0 bus {bus} ->", out)
        assert f"/hetero/{dev}.log" in out
        assert f"/hetero/{dev}.ready" in out
    assert len(server_commands(out)) == 4


def test_h1_server_command_and_explicit_kernel_env(venv, h1_plan):
    out = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}").stdout
    cmd = command_of(out, "exp0")
    assert (
        f"-m vllm.distributed.hetero_moe.server --plan {h1_plan} --device-id exp0 "
        "--model /nonexistent/model --ready-file "
    ) in cmd
    assert cmd.endswith(
        "/hetero/exp0.ready --buckets 3,6,12,24,48 --max-rows 2048 --wait-ms 2000"
    )
    # The servers' kernels never follow the hub recipe: set explicitly (the hub
    # recipe has W4A8 sdot4 OFF).
    for word in (
        "VLLM_RDNA2_W4A8_SDOT4=1",
        "VLLM_RDNA2_MOE_DETERMINISTIC=1",
        "GPU_MAX_HW_QUEUES=4",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
        "VLLM_HETERO_MOE_ROLE=expert",
        "VLLM_HETERO_MOE_DEVICE_ID=exp0",
        f"VLLM_HETERO_MOE_PLAN={h1_plan}",
        "VLLM_HETERO_MOE_WAIT_MS=2000",
    ):
        assert f" {word} " in cmd
    assert "VLLM_RDNA3" not in cmd
    assert cmd.index("HIP_VISIBLE_DEVICES=") < cmd.index("-m vllm")
    assert "/nonexistent" not in command_of(out, "exp1").split(" --model ")[0]


def test_unresolved_buses_are_printed_as_such(venv, h1_plan):
    out = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}").stdout
    assert out.count("(not resolvable on this host; resolved at launch)") == 1 + 4
    assert "HIP_VISIBLE_DEVICES=@bus=ce " in command_of(out, "exp0")


def test_resolved_buses_give_each_server_its_own_hip(resolving_venv, h1_plan):
    result = serve(resolving_venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='cb' -> HIP_VISIBLE_DEVICES=0" in out
    assert "export HIP_VISIBLE_DEVICES='0'" in out
    hips = []
    for dev, bus, hip in (
        ("exp0", "ce", 1),
        ("exp1", "d3", 2),
        ("exp2", "d8", 3),
        ("exp3", "db", 4),
    ):
        assert re.search(rf"server {dev} island 0 bus {bus} -> HIP {hip} ", out)
        cmd = command_of(out, dev)
        assert f" HIP_VISIBLE_DEVICES={hip} " in cmd
        hips.append(hip)
    assert len(set(hips)) == 4 and 0 not in hips


def test_hub_and_server_must_not_share_a_gpu(resolving_venv, h1_plan):
    result = serve(
        resolving_venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}", HIP_VISIBLE_DEVICES="2"
    )
    assert result.returncode != 0
    assert "disjoint" in result.stderr
    result = serve(
        resolving_venv,
        H1,
        f"VLLM_HETERO_MOE_PLAN={h1_plan}",
        "VLLM_HETERO_MOE_SERVERS=cb,d3,d8,db",
    )
    assert result.returncode != 0
    assert "both a hub bus" in result.stderr


def test_standin_recipe_hub_ce_three_servers(venv, v620_plan):
    result = serve(venv, V620, f"VLLM_HETERO_MOE_PLAN={v620_plan}")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='ce' (not resolvable on this host" in out
    assert "'--tensor-parallel-size' '1'" in out
    assert "'--attention-backend' 'RDNA_ATTN'" in out
    assert "--enable-expert-parallel" not in out
    assert "'--max-model-len' '16384'" in out
    assert "'--kv-cache-memory-bytes' '2147483648'" in out
    assert "'--port' '18098'" in out
    assert "VLLM_ROCM_MONO_DECODE" not in out
    assert "VLLM_TUNED_CONFIG_FOLDER" not in out
    assert "export VLLM_RDNA_AR='0'" in out
    assert "VLLM_RDNA_AR_MAX_KB" not in out
    assert "export VLLM_HETERO_MOE_SPILL_K='3:2,6:4,12:8,24:12,48:16'" in out
    assert "export VLLM_HETERO_MOE_DEVICE_ID='hub0'" in out
    assert (
        "export VLLM_CACHE_ROOT='/home/someone/.cache/vllm-flashnext-hetero-v620'"
        in out
    )
    assert "1 hub(s), 3 expert device(s)" in out
    assert len(server_commands(out)) == 3
    for dev, bus in zip(("exp0", "exp1", "exp2"), ("d3", "d8", "db")):
        assert re.search(rf"server {dev} island 0 bus {bus} ->", out)
    assert "exp3" not in out
    cmd = command_of(out, "exp2")
    assert "--buckets 3,6,12,24,48 " in cmd and "--wait-ms 2000" in cmd


def test_standin_resolved_buses_are_disjoint_from_the_hub(resolving_venv, v620_plan):
    out = serve(resolving_venv, V620, f"VLLM_HETERO_MOE_PLAN={v620_plan}").stdout
    assert "PCI_BUSES='ce' -> HIP_VISIBLE_DEVICES=1" in out
    for dev, hip in (("exp0", 2), ("exp1", 3), ("exp2", 4)):
        assert f" HIP_VISIBLE_DEVICES={hip} " in command_of(out, dev)


@pytest.mark.parametrize("recipe", [H1, V620])
def test_missing_or_empty_plan_names_the_stripe_command(venv, recipe, tmp_path):
    for args in ((), (f"VLLM_HETERO_MOE_PLAN={tmp_path / 'missing.json'}",)):
        result = serve(venv, recipe, *args)
        assert result.returncode != 0
        assert (
            "python vllm/distributed/hetero_moe/cli.py stripe --out <plan>"
            in result.stderr
        )
        assert "# hetero:" not in result.stdout


def test_an_unusable_plan_is_refused(venv, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    result = serve(venv, H1, f"VLLM_HETERO_MOE_PLAN={bad}")
    assert result.returncode != 0
    assert "is not usable" in result.stderr


def test_server_bus_count_mismatch_dies(venv, h1_plan):
    result = serve(
        venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}", "VLLM_HETERO_MOE_SERVERS=ce,d3"
    )
    assert result.returncode != 0
    assert (
        "lists 2 bus(es) (ce,d3) but the plan has 4 expert device(s)" in result.stderr
    )
    result = serve(
        venv,
        H1,
        f"VLLM_HETERO_MOE_PLAN={h1_plan}",
        "VLLM_HETERO_MOE_SERVERS=ce,d3,d8,db;0a,0f,14,17",
    )
    assert result.returncode != 0
    assert "lists 2 island(s) but the plan has 1" in result.stderr


def test_hub_count_must_match_tp_and_pci_buses(venv, h1_plan):
    result = serve(
        venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}", "TP=2", "PCI_BUSES=cb 07"
    )
    assert result.returncode != 0
    assert "plan has 1 hub(s)" in result.stderr


def test_unset_server_buses_fall_back_to_plan_hips_with_a_warning(venv, h1_plan):
    result = serve(
        venv, H1, f"VLLM_HETERO_MOE_PLAN={h1_plan}", "VLLM_HETERO_MOE_SERVERS="
    )
    assert result.returncode == 0, result.stderr
    assert "WARNING: VLLM_HETERO_MOE_SERVERS is unset" in result.stderr
    assert "HIP indices move" in result.stderr
    for dev, hip in (("exp0", 1), ("exp1", 2), ("exp2", 3), ("exp3", 4)):
        assert f" HIP_VISIBLE_DEVICES={hip} " in command_of(result.stdout, dev)


def test_log_dir_and_extra_server_args_knobs(venv, h1_plan):
    out = serve(
        venv,
        H1,
        f"VLLM_HETERO_MOE_PLAN={h1_plan}",
        "HETERO_LOG_DIR=/var/log/hx",
        "HETERO_SERVER_ARGS=--foo 1 --bar",
        "HETERO_READY_TIMEOUT=60",
        "VLLM_HETERO_MOE_WAIT_MS=500",
    ).stdout
    cmd = command_of(out, "exp3")
    assert "--ready-file /var/log/hx/exp3.ready " in cmd
    assert cmd.endswith("--wait-ms 500 --foo 1 --bar")
    assert "ready-timeout 60s" in out


def test_non_hetero_recipe_has_no_hetero_stanza(venv):
    result = serve(venv, "flashnext-w2-mtp2")
    assert result.returncode == 0, result.stderr
    assert "# hetero" not in result.stdout
    assert "VLLM_HETERO" not in result.stdout
    assert "PCI_BUSES='cb 07' (not resolvable on this host" in result.stdout
    # The default path still ends in the one exec.
    last = LAUNCHER.read_text().rstrip("\n").splitlines()[-1]
    assert last == 'exec ${LAUNCH_WRAPPER:-} "${cmd[@]}"'


# --- the supervisor -----------------------------------------------------------

SOURCE = """set -euo pipefail
source tools/rdna/rdna_launcher_common.sh
source tools/rdna/rdna_hetero_launch.sh
HETERO_POLL_INTERVAL=0.1
HETERO_TERM_TIMEOUT=%(term)s
HETERO_READY_TIMEOUT=%(ready)s
rdna_hetero_sids=(%(ids)s)
rdna_hetero_slogs=(%(logs)s)
rdna_hetero_sready=(%(readies)s)
"""
PID = 'echo $$ > "$D/$0.pid"'
READY = 'touch "$D/$0.ready"'


def server_ok(name, delay=0):
    return f"sh -c '{PID}; sleep {delay}; {READY}; exec sleep 60' {name}"


def server_dies_before_ready(name, code):
    return f"sh -c '{PID}; echo boom-$0; exit {code}' {name}"


def server_dies_after_ready(name, code, after):
    return f"sh -c '{PID}; {READY}; sleep {after}; exit {code}' {name}"


def server_never_ready(name):
    return f"sh -c '{PID}; exec sleep 60' {name}"


def hub_exits(code):
    return (
        'sh -c \'echo $$ > "$D/hub.pid"; echo "${HUBWRAP:-no}" > "$D/hub.wrap";'
        ' if [ -e "$D/a.ready" ] && [ -e "$D/b.ready" ]; then'
        ' touch "$D/hub.after_ready"; else touch "$D/hub.early"; fi;'
        f" exit {code}' hub"
    )


HUB_SLEEPS = (
    'sh -c \'echo $$ > "$D/hub.pid"; sleep 60 & echo $! > "$D/hub.child"; wait\' hub'
)
HUB_IGNORES_TERM = (
    'sh -c \'trap "" TERM; echo $$ > "$D/hub.pid"; while :; do sleep 1; done\' hub'
)


def supervisor_script(servers, hub, *, term=3, ready=5, wrapper=""):
    names = "abcdef"[: len(servers)]
    script = SOURCE % {
        "term": term,
        "ready": ready,
        "ids": " ".join(names),
        "logs": " ".join(f'"$D/{n}.log"' for n in names),
        "readies": " ".join(f'"$D/{n}.ready"' for n in names),
    }
    for i, cmd in enumerate(servers):
        script += f"rdna_hetero_server_cmd_{i}=({cmd})\n"
    script += f"rdna_hetero_hub_cmd=({hub})\n"
    if wrapper:
        script += f"LAUNCH_WRAPPER='{wrapper}'\n"
    script += 'rc=0\nrdna_hetero_launch || rc=$?\nexit "$rc"\n'
    return script


def read_pids(tmp_path):
    return {
        p.stem: int(p.read_text())
        for p in tmp_path.glob("*.pid")
        if p.read_text().strip()
    }


def is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.05)
    return not is_alive(pid)


@pytest.fixture()
def sup(tmp_path):
    """Run a supervisor script; leftover stub processes are killed afterwards."""
    env = {**os.environ, "D": str(tmp_path)}
    env.pop("LAUNCH_WRAPPER", None)

    def run(script, **kwargs):
        return subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env={**env, **kwargs.pop("extra_env", {})},
            cwd=REPO,
            timeout=60,
            **kwargs,
        )

    yield run
    for pid in read_pids(tmp_path).values():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    for child in tmp_path.glob("*.child"):
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(child.read_text()), signal.SIGKILL)


def test_ready_wait_then_hub_exit_code_is_passed_through(sup, tmp_path):
    result = sup(
        supervisor_script([server_ok("a"), server_ok("b", delay=0.5)], hub_exits(7))
    )
    assert result.returncode == 7, result.stderr
    assert (tmp_path / "hub.after_ready").exists()
    assert not (tmp_path / "hub.early").exists()
    assert "the hub exited with code 7" in result.stderr
    pids = read_pids(tmp_path)
    assert wait_dead(pids["a"]) and wait_dead(pids["b"])


def test_clean_hub_exit_is_zero(sup, tmp_path):
    result = sup(supervisor_script([server_ok("a"), server_ok("b")], hub_exits(0)))
    assert result.returncode == 0, result.stderr
    assert all(wait_dead(p) for p in read_pids(tmp_path).values())


def test_launch_wrapper_applies_to_the_hub_only(sup, tmp_path):
    result = sup(
        supervisor_script(
            [server_ok("a"), server_ok("b")],
            hub_exits(0),
            wrapper="env HUBWRAP=yes",
        )
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "hub.wrap").read_text().strip() == "yes"
    assert not list(tmp_path.glob("a.wrap"))


def test_server_dying_before_ready_fails_and_never_starts_the_hub(sup, tmp_path):
    result = sup(
        supervisor_script(
            [server_ok("a"), server_dies_before_ready("b", 5)], hub_exits(0)
        )
    )
    assert result.returncode == 1
    assert not (tmp_path / "hub.pid").exists()
    assert "expert server b" in result.stderr
    assert "exited with code 5 before it was ready" in result.stderr
    assert "boom-b" in result.stderr  # the log tail
    assert wait_dead(read_pids(tmp_path)["a"])


def test_ready_timeout_tears_the_servers_down(sup, tmp_path):
    result = sup(
        supervisor_script(
            [server_ok("a"), server_never_ready("b")], hub_exits(0), ready=1
        )
    )
    assert result.returncode == 1
    assert not (tmp_path / "hub.pid").exists()
    assert "expert server b" in result.stderr and "not ready after 1s" in result.stderr
    assert "ERROR: expert server a" not in result.stderr
    assert all(wait_dead(p) for p in read_pids(tmp_path).values())


def test_stale_ready_file_does_not_release_the_hub(sup, tmp_path):
    (tmp_path / "a.ready").write_text("stale")
    (tmp_path / "b.ready").write_text("stale")
    result = sup(
        supervisor_script(
            [server_ok("a", delay=0.4), server_ok("b", delay=0.4)], hub_exits(0)
        )
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "hub.after_ready").exists()


def test_server_dying_under_a_running_hub_exits_3_and_kills_the_hub(sup, tmp_path):
    result = sup(
        supervisor_script(
            [server_ok("a"), server_dies_after_ready("b", 4, 0.6)], HUB_SLEEPS
        )
    )
    assert result.returncode == 3, result.stderr
    assert "expert server b" in result.stderr
    assert "exited with code 4 while the hub was running" in result.stderr
    pids = read_pids(tmp_path)
    assert wait_dead(pids["hub"]) and wait_dead(pids["a"])
    assert wait_dead(int((tmp_path / "hub.child").read_text()))


def test_a_hub_that_ignores_sigterm_is_killed_after_the_term_timeout(sup, tmp_path):
    started = time.monotonic()
    result = sup(
        supervisor_script(
            [server_ok("a"), server_dies_after_ready("b", 4, 0.6)],
            HUB_IGNORES_TERM,
            term=1,
        )
    )
    assert result.returncode == 3, result.stderr
    assert "SIGKILL process group" in result.stderr
    assert wait_dead(read_pids(tmp_path)["hub"])
    assert time.monotonic() - started < 30


def test_sigterm_to_the_supervisor_kills_everything(tmp_path):
    env = {**os.environ, "D": str(tmp_path)}
    env.pop("LAUNCH_WRAPPER", None)
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            supervisor_script([server_ok("a"), server_ok("b")], HUB_SLEEPS),
        ],
        env=env,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "hub.child").exists():
            assert time.monotonic() < deadline, "the hub never started"
            time.sleep(0.05)
        pids = read_pids(tmp_path)
        child = int((tmp_path / "hub.child").read_text())
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 143, err
    for pid in (*pids.values(), child):
        assert wait_dead(pid), pid


def test_setsid_is_used_when_available(sup, tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "setsid").write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"open({str(tmp_path / 'setsid.calls')!r}, 'a').write(sys.argv[1] + '\\n')\n"
        "os.setsid()\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n"
    )
    (shim / "setsid").chmod(0o755)
    result = sup(
        supervisor_script([server_ok("a"), server_ok("b")], hub_exits(2)),
        extra_env={"PATH": f"{shim}:{os.environ['PATH']}"},
    )
    assert result.returncode == 2, result.stderr
    assert (tmp_path / "setsid.calls").read_text().split() == ["sh", "sh", "sh"]
    assert all(wait_dead(p) for p in read_pids(tmp_path).values())


STUB_LAUNCH = STUB_RESOLVER.replace(
    '[ "$1" = -c ] || exit 1\nshift 2\n',
    """case "$1" in
  -c) shift 2 ;;
  -m)
    case "$2" in
      vllm.distributed.hetero_moe.server)
        shift 2
        while [ $# -gt 0 ]; do
          [ "$1" = --ready-file ] && ready=$2
          shift
        done
        echo "$HIP_VISIBLE_DEVICES $VLLM_HETERO_MOE_ROLE" > "$ready.env"
        touch "$ready"
        exec sleep 60 ;;
      *)
        echo "$HIP_VISIBLE_DEVICES $VLLM_HETERO_MOE_ROLE" > "$D/hub.env"
        sleep 0.5
        exit 9 ;;
    esac ;;
  *) exit 1 ;;
esac
""",
)


def test_serve_rdna_runs_servers_first_then_the_hub(tmp_path, h1_plan):
    """serve_rdna.sh without PRINT, stub python for servers and hub."""
    venv = _stub_venv(tmp_path, STUB_LAUNCH)
    run = tmp_path / "run"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in DROPPED and not k.startswith(("HETERO_", "VLLM_HETERO_MOE"))
    }
    env.update(HOME="/home/someone", D=str(tmp_path), HETERO_PYTHON=sys.executable)
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            f"RECIPE={H1}",
            f"VLLM_HETERO_MOE_PLAN={h1_plan}",
            f"VENV={venv}",
            "MODEL=/nonexistent/model",
            f"RUN_CWD={run}",
            "HETERO_POLL_INTERVAL=0.1",
            "HETERO_TERM_TIMEOUT=2",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=60,
    )
    assert result.returncode == 9, result.stderr
    assert (tmp_path / "hub.env").read_text().split() == ["0", "hub"]
    for dev, hip in (("exp0", 1), ("exp1", 2), ("exp2", 3), ("exp3", 4)):
        assert (run / "hetero" / f"{dev}.ready").exists()
        seen = (run / "hetero" / f"{dev}.ready.env").read_text().split()
        assert seen == [str(hip), "expert"]
    assert "the hub exited with code 9" in result.stderr
