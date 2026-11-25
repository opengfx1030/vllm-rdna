# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/inkling_hi2/validate_final_a2.sh (the one-shot card driver).

The driver is pure bash; every external tool is overridable through an env
variable, so these tests build a scratch kit of stubs and exercise the driver
logic itself: the dry-run pass (no lease), the disk-free guard, resumability
(a PASSed step is skipped) and stop-on-first-failure with the summary table.

No GPU, no torch, no /tmp: everything lives under pytest's tmp_path.
"""

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "tools" / "inkling_hi2" / "validate_final_a2.sh"
LEASE_SH = ROOT / "tools" / "inkling_hi2" / "lease.sh"

pytestmark = pytest.mark.skipif(not DRIVER.is_file(), reason="driver missing")

VISION_LINE = "vision tensors bit-equal to stock: 422/422"
MTP_LINE = "mtp.safetensors: 54 tensors bit-equal to stock: True"
DENSE_LINE = "dense text tensors bit-equal to stock: 426"


def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def make_kit(base: Path) -> dict:
    kit = base / "kit"
    kit.mkdir()
    shutil.copy(DRIVER, kit / "validate_final_a2.sh")
    shutil.copy(LEASE_SH, kit / "lease.sh")
    shutil.copy(ROOT / "tools" / "inkling_hi2" / "slot.py", kit / "slot.py")
    _stub(
        kit / "serve.sh",
        'echo "DRY MODEL=$MODEL MAXLEN=$MAXLEN DSPARK=$DSPARK VISION=$VISION"; '
        "exit 0\n",
    )
    _stub(kit / "smoke.sh", "exit 0\n")
    (kit / "help.py").write_text("print('help')\n")
    _stub(kit / "evalpy", "echo eval-help\n")
    _stub(kit / "prep.sh", "exit 0\n")
    (kit / "validate.py").write_text("import sys\nsys.exit(0)\n")
    # check_real prints the three lines the step parses.
    (kit / "check.py").write_text(
        "import os, sys\n"
        f'print("{VISION_LINE}")\n'
        f'print("{MTP_LINE}")\n'
        f'print("{DENSE_LINE}")\n'
        "open(os.path.join(os.environ['CALLS'], 'check'), 'a').write('call\\n')\n"
        "sys.exit(0)\n"
    )
    (kit / "check_fail.py").write_text(
        "import os, sys\n"
        "open(os.path.join(os.environ['CALLS'], 'check'), 'a').write('call\\n')\n"
        "sys.exit(1)\n"
    )
    (kit / "calls").mkdir()
    fork = kit / "fork"
    (fork / "tests" / "models" / "inkling2_loop").mkdir(parents=True)
    (fork / "tests" / "models" / "inkling2_loop" / "test_hi2_kernel_gpu.py").write_text(
        "def test_placeholder():\n    assert True\n"
    )
    model = kit / "model"
    model.mkdir()
    stock = kit / "stock"
    stock.mkdir()
    (stock / "model.safetensors.index.json").write_text("{}")
    evaltree = kit / "eval"
    (evaltree / "tools" / "inkling_eval").mkdir(parents=True)
    (evaltree / "tools" / "inkling_eval" / "cli.py").write_text("")
    venv = kit / "venv"
    (venv / "bin").mkdir(parents=True)
    os.symlink(sys.executable, venv / "bin" / "python")
    for d in ("quantizer", "deps", "rt"):
        (kit / d).mkdir()
    _drm_tree(kit)
    shim = _date_shim(kit)
    return {
        "driver": kit / "validate_final_a2.sh",
        "kit": kit,
        "fork": fork,
        "model": model,
        "stock": stock,
        "evaltree": evaltree,
        "venv": venv,
        "calls": kit / "calls",
        "shim": shim,
    }


def run(k: dict, tag: str, **extra) -> subprocess.CompletedProcess:
    env = {
        "PATH": str(k["shim"]) + os.pathsep + os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/"),
        "V": sys.executable,
        "VENV": str(k["venv"]),
        "TREE": str(k["fork"]),
        "RT": str(k["kit"] / "rt" / tag),
        "MODEL": str(k["model"]),
        "STOCK": str(k["stock"]),
        "EVAL_TREE": str(k["evaltree"]),
        "EVAL_PY": str(k["eval_tree_py"]),
        "QUANTIZER_TREE": str(k["kit"] / "quantizer"),
        "DEPS": str(k["kit"] / "deps"),
        "TINY": str(k["kit"] / "tiny"),
        "SERVE_DIR": str(k["kit"] / "serve"),
        "SERVE": str(k["kit"] / "serve.sh"),
        "SMOKE": str(k["kit"] / "smoke.sh"),
        "KLPROBE": str(k["kit"] / "help.py"),
        "BENCH": str(k["kit"] / "help.py"),
        "MTP_PROBE": str(k["kit"] / "help.py"),
        "VISION_PROBE": str(k["kit"] / "help.py"),
        "PREP": str(k["kit"] / "prep.sh"),
        "VALIDATE": str(k["kit"] / "validate.py"),
        "CHECK_REAL": str(k["kit"] / "check.py"),
        "LEASE_DIR": str(k["kit"] / "lease"),
        "DRM_SYSFS": str(k["kit"] / "drm"),
        "LEASE_POLL_SECONDS": "1",
        "LEASE_WAIT_SECONDS": "5",
        "QUIET_DIR": str(k["kit"] / "quiet"),
        "BOOT_LOCK": str(k["kit"] / "boot.lock"),
        "OWNER": "HI2InklingRuntime",
        "SLOT": "a2",
        "HIP_VISIBLE_DEVICES": "0,1",
        "EXPECT_PCI_BUSES": "d1,d4",
        "RUN": str(k["kit"] / "rt" / tag),
        "TAG": tag,
        "NOTES": "pytest",
        "DETACH": "0",
        "SETSID": "",
        "MIN_FREE_GIB": "0",
        "CALLS": str(k["calls"]),
    }
    for name, value in extra.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    return subprocess.run(
        ["bash", str(k["driver"])],
        env=env,
        cwd=str(k["kit"]),
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture
def kit(tmp_path):
    k = make_kit(tmp_path)
    k["eval_tree_py"] = k["kit"] / "evalpy"
    return k


def rt(k: dict, tag: str) -> Path:
    return k["kit"] / "rt" / tag


def test_dry_run_needs_no_lease(kit):
    r = run(kit, "dry", DRY_RUN=1)
    assert r.returncode == 0, r.stdout + r.stderr
    dry = (rt(kit, "dry") / "dryrun.log").read_text()
    # every serve flavour's argv is dry-run through serve.sh PRINT=1
    assert dry.count("DRY ") >= 5, dry
    assert "MAXLEN=720000" in dry and "DSPARK=1" in dry and "VISION=1" in dry, dry
    lines = dry.splitlines()
    assert lines.count("help") == 4, dry  # the four tool --help dry-runs
    assert "eval-help" in lines
    assert (
        (rt(kit, "dry") / "summary.md")
        .read_text()
        .startswith("# HI2 final validation: DRY-RUN")
    )
    # no lease and no boot lock were taken
    assert not (kit["kit"] / "lease" / "HI2InklingRuntime.a2").exists()
    assert not (kit["kit"] / "boot.lock").exists()


def test_disk_guard_stops_before_anything(kit):
    r = run(kit, "disk", DRY_RUN=1, MIN_FREE_GIB=10**9)
    assert r.returncode == 3, r.stdout + r.stderr
    status = (rt(kit, "disk") / "STATUS.log").read_text()
    assert "disk guard" in status
    assert not (rt(kit, "disk") / "dryrun.log").exists()


def test_passed_step_is_skipped_on_resume(kit):
    first = run(kit, "resume", ONLY="serve_dir", RESUME=1)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (rt(kit, "resume") / "steps" / "serve_dir.pass").exists()
    assert (kit["calls"] / "check").read_text().count("call") == 1
    second = run(kit, "resume", ONLY="serve_dir", RESUME=1)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "SKIP serve_dir" in (rt(kit, "resume") / "STATUS.log").read_text()
    assert (kit["calls"] / "check").read_text().count("call") == 1


def test_first_failure_stops_and_is_recorded(kit):
    r = run(
        kit,
        "fail",
        ONLY="serve_dir",
        RESUME=1,
        CHECK_REAL=str(kit["kit"] / "check_fail.py"),
    )
    assert r.returncode == 1, r.stdout + r.stderr
    text = (rt(kit, "fail") / "summary.md").read_text()
    assert "| serve_dir | FAIL |" in text, text
    assert not (rt(kit, "fail") / "steps" / "kl.pass").exists()
    status = (rt(kit, "fail") / "STATUS.log").read_text()
    assert "FAIL serve_dir" in status


def _drm_tree(base: Path) -> Path:
    """A sysfs-shaped DRM tree with two idle 32 GiB cards (d1, d4)."""
    drm = base / "drm"
    for i, bus in enumerate(("d1", "d4")):
        target = base / f"0000:{bus}:00.0"
        target.mkdir()
        (target / "mem_info_vram_used").write_text("0")
        (target / "mem_info_vram_total").write_text(str(32 * 1024**3))
        (drm / f"card{i}").mkdir(parents=True)
        (drm / f"card{i}" / "device").symlink_to(target, target_is_directory=True)
    return drm


def _date_shim(base: Path) -> Path:
    """A `date` that accepts the Linux `-Is` form the lease uses (macOS does not)."""
    shim = base / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "date").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *-I*) echo 2026-11-25T00:00:00Z ;;\n"
        '  *) exec /bin/date "$@" ;;\n'
        "esac\n"
    )
    (shim / "date").chmod(0o755)
    return shim


def lease_run(tmp_path: Path, script: str, inherit: int = 1):
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir(exist_ok=True)
    shim = _date_shim(tmp_path)
    env = {
        "PATH": str(shim) + os.pathsep + os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/"),
        "OWNER": "HI2InklingRuntime",
        "SLOT": "a2",
        "LEASE_DIR": str(lease_dir),
        "EXPECT_PCI_BUSES": "d1,d4",
        "DRM_SYSFS": str(_drm_tree(tmp_path)),
        "LEASE_FREE_GIB": "28",
        "LEASE_POLL_SECONDS": "1",
        "LEASE_WAIT_SECONDS": "5",
        "LEASE_INHERIT": str(inherit),
        "LEASE_SH": str(LEASE_SH),
    }
    r = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r, lease_dir


def _own_holder(lease_dir: Path, pid: int, gate: str | None = "32768") -> None:
    (lease_dir / "HI2InklingRuntime.a2").write_text(
        f"HI2InklingRuntime pid {pid} since test\n"
    )
    if gate is not None:
        (lease_dir / "HI2InklingRuntime.a2.gate").write_text(gate + "\n")


def test_lease_inherit_own_holder_skips_queue(tmp_path):
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    _own_holder(lease_dir, os.getpid())
    r = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\n. "$LEASE_SH"\nlease_acquire\necho INHERITED\n',
        ],
        env={
            "PATH": str(_date_shim(tmp_path)) + os.pathsep + os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/"),
            "OWNER": "HI2InklingRuntime",
            "SLOT": "a2",
            "LEASE_DIR": str(lease_dir),
            "EXPECT_PCI_BUSES": "d1,d4",
            "DRM_SYSFS": str(_drm_tree(tmp_path)),
            "LEASE_FREE_GIB": "28",
            "LEASE_POLL_SECONDS": "1",
            "LEASE_WAIT_SECONDS": "5",
            "LEASE_INHERIT": "1",
            "LEASE_SH": str(LEASE_SH),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "INHERITED" in r.stdout, r.stderr
    assert "inheriting our HI2InklingRuntime.a2" in r.stdout, r.stdout
    assert not (lease_dir / "a2.request").exists()
    assert (lease_dir / "HI2InklingRuntime.a2").exists()


def test_lease_inherit_foreign_holder_refuses(tmp_path):
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    (lease_dir / "OtherAgent.a2").write_text(
        f"OtherAgent pid {os.getpid()} since test\n"
    )
    (lease_dir / "OtherAgent.a2.gate").write_text("32768\n")
    # The gate that a *bypassed* inherit check would read; it must still not
    # let a foreign holder through.
    (lease_dir / "HI2InklingRuntime.a2.gate").write_text("32768\n")
    r, _ = lease_run(
        tmp_path,
        'mkdir -p "$LEASE_DIR"\n'
        '. "$LEASE_SH"\n'
        "if lease_acquire; then echo ACQ; else echo REFUSED; fi\n",
        inherit=1,
    )
    assert "REFUSED" in r.stdout, r.stdout + r.stderr
    assert "holds the a2 lease" in r.stderr, r.stderr


def test_lease_release_inherit_keeps_owner_holder(tmp_path):
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    _own_holder(lease_dir, os.getpid())
    r, _ = lease_run(
        tmp_path,
        'mkdir -p "$LEASE_DIR"\n'
        '. "$LEASE_SH"\n'
        "lease_acquire\nlease_release\necho DONE\n",
        inherit=1,
    )
    assert "DONE" in r.stdout, r.stderr
    assert (lease_dir / "HI2InklingRuntime.a2").exists()
    assert (lease_dir / "HI2InklingRuntime.a2.gate").exists()


def test_lease_own_request_does_not_self_queue(tmp_path):
    r, lease_dir = lease_run(
        tmp_path,
        'mkdir -p "$LEASE_DIR"\n'
        '. "$LEASE_SH"\n'
        'printf \'%s pid %s since t\\n\' "$OWNER" "$$">"$LEASE_DIR/$SLOT.request"\n'
        "lease_acquire\ntrap - EXIT\necho ACQUIRED\n",
        inherit=0,
    )
    assert "ACQUIRED" in r.stdout, r.stderr
    assert "queueing" not in r.stdout + r.stderr, r.stdout + r.stderr
    assert (lease_dir / "HI2InklingRuntime.a2").exists()
    assert not (lease_dir / "a2.request").exists()


FAKE_HEALTH = """
import http.server, os, sys

port = int(sys.argv[1])
pidfile = sys.argv[2] if len(sys.argv) > 2 else None
if pidfile:
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

FAKE_BENCH = """
import argparse, json, os

ap = argparse.ArgumentParser()
for flag in ("--url", "--server-pid", "--label", "--log", "--baseline", "--out"):
    ap.add_argument(flag)
ap.add_argument("--decode-tokens", type=int, default=32)
a = ap.parse_args()
with open(os.path.join(os.environ["CALLS"], "bench"), "a") as f:
    f.write(json.dumps(vars(a)) + "\\n")
tps, b1, b4 = {
    "stock": (2000.0, 10.0, 12.0),
    "hi2": (1100.0, 18.0, 22.0),
}.get(a.label, (1500.0, 15.0, 18.0))
cells = [
    {"cell": "prefill", "batch": 1, "prompt_tokens_per_s": tps},
    {"cell": "decode", "batch": 1, "ms_per_step": b1},
    {"cell": "decode", "batch": 4, "ms_per_step": b4},
]
with open(a.out, "w") as f:
    json.dump({"label": a.label, "max_model_len": 32768, "cells": cells,
               "decode_tokens": a.decode_tokens}, f)
"""

FAKE_SERVE = """#!/usr/bin/env bash
if [[ ${PRINT:-0} == 1 ]]; then
    echo "DRY MODEL=$MODEL MAXLEN=$MAXLEN DSPARK=$DSPARK VISION=$VISION"
    exit 0
fi
exec {python} "{FAKE_KIT}/fake_health.py" "$PORT" "$SERVER_PID_FILE"
"""


def test_bench_pair_then_solo_arm(kit):
    """The bench step runs the pair arm, then a solo HI2 arm at GMEM 0.92."""
    for port in (18230, 18231):
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            pytest.skip(f"port {port} is busy")
        finally:
            probe.close()
    run_dir = kit["kit"] / "rt" / "bench"
    (run_dir / "pids").mkdir(parents=True)
    (kit["kit"] / "fake_health.py").write_text(FAKE_HEALTH)
    (kit["kit"] / "bench.py").write_text(FAKE_BENCH)
    serve = kit["kit"] / "fake_serve.sh"
    body = FAKE_SERVE.replace("{python}", sys.executable).replace(
        "{FAKE_KIT}", str(kit["kit"])
    )
    _stub(serve, body)
    servers = []
    for name, port in (("stock", 18231), ("hi2", 18230)):
        servers.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(kit["kit"] / "fake_health.py"),
                    str(port),
                    str(run_dir / "pids" / f"{name}.pid"),
                ]
            )
        )
    try:
        deadline = time.time() + 15
        for port in (18230, 18231):
            while True:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    if time.time() > deadline:
                        pytest.skip("fake health server did not start")
                    time.sleep(0.1)
        r = run(
            kit,
            "bench",
            ONLY="bench",
            BENCH=str(kit["kit"] / "bench.py"),
            SERVE=str(serve),
        )
        assert r.returncode == 0, r.stdout + r.stderr
        numbers = json.loads((run_dir / "steps" / "bench.json").read_text())["numbers"]
        assert numbers["decode_tokens"] == "128"
        assert "gmem=0.45" in numbers["pair_flags"]
        assert "gmem=0.92" in numbers["solo_flags"]
        ratios = numbers["ratios"]
        assert "pair_prefill=0.550" in ratios, ratios
        assert "solo_prefill_tps=1500.0" in ratios, ratios
        calls = [
            json.loads(line)
            for line in (kit["calls"] / "bench").read_text().splitlines()
        ]
        assert [c["label"] for c in calls] == ["stock", "hi2", "hi2-solo"], calls
        assert all(c["decode_tokens"] == 128 for c in calls), calls
        assert calls[-1]["url"] == "http://127.0.0.1:18230"
        assert not (run_dir / "pids" / "solo.pid").exists()
    finally:
        for proc in servers:
            proc.kill()
