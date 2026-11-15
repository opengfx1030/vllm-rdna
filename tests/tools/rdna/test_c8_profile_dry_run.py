# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/rdna_hint_flex/c8_profile.py against a fake server: boot wait, the
rank map from the worker process titles, the iteration-log state machine that
opens the prefill / mixed / decode windows, the SIGUSR1/SIGUSR2 pairs, the GPU
refusal. No GPU, no vLLM, no rocprofv3. Linux only (/proc, pgrep -s)."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "tools/rdna/rdna_hint_flex/c8_profile.py"
FAKE = Path(__file__).with_name("c8_fake_server.py")
LINUX = sys.platform.startswith("linux")

SERVE = """#!/usr/bin/env bash
for kv in "$@"; do case $kv in PORT=*) PORT=${kv#PORT=};; esac; done
echo "serve args: $*"
exec python3 %s serve --port "$PORT"
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_venv(root: Path) -> Path:
    """The files c8_profile.py looks for in a venv, none of them real."""
    site = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    core = site / "site-packages/_rocm_sdk_core"
    (core / "bin").mkdir(parents=True)
    (core / "lib").mkdir(parents=True)
    (core / "bin/rocprofv3").write_text("")
    (core / "lib/librocprofiler-sdk-roctx.so.1").write_text("")
    (root / "bin").mkdir()
    (root / "bin/python").symlink_to(sys.executable)
    return root


def _run(tmp: Path, gpus: str = "0,1") -> subprocess.CompletedProcess:
    port = _free_port()
    serve = tmp / "fake_serve.sh"
    serve.write_text(SERVE % FAKE)
    cmd = [
        sys.executable,
        str(DRIVER),
        "--label", "dry",
        "--gpus", gpus,
        "--port", str(port),
        "--model", "m",
        "--venv", str(_fake_venv(tmp / "venv")),
        "--out-root", str(tmp / "out"),
        "--boot-lock", str(tmp / "boot.lock"),
        "--quiet-dir", str(tmp / "quiet"),
        "--serve-script", str(serve),
        "--bench-prefix", f"{sys.executable} {FAKE} bench --port {port}",
        "--prefill-window-s", "1",
        "--mixed-window-s", "1",
        "--decode-window-s", "0.5",
        "--settle-s", "0.2",
        "--state-timeout-s", "40",
        "--ready-timeout-s", "40",
        "--no-analyse",
        "--",
        "RECIPE=27b-hi2-tp2",
        "MAXBAT=2056",
    ]  # fmt: skip
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "", "C8_FAKE_DIR": str(tmp)},
    )


@pytest.mark.skipif(not LINUX, reason="/proc and pgrep -s")
def test_workload_windows_and_signals(tmp_path):
    (tmp_path / "quiet").mkdir()
    # a stale quiet file (older than the 6 minute rule) must not hold the boot
    stale = tmp_path / "quiet/Other"
    stale.write_text("")
    os.utime(stale, (1, 1))
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "out/dry"
    assert "FINISHED rc=0" in r.stdout
    # the recipe pass-through, with the iteration-detail flag appended
    serve_log = (out / "serve.log").read_text()
    assert "RECIPE=27b-hi2-tp2" in serve_log and "MAXBAT=2056" in serve_log
    assert "EXTRA_ARGS=--enable-logging-iteration-details" in serve_log
    ranks = json.loads((out / "ranks.json").read_text())
    assert sorted(ranks) == ["0", "1"] and len(set(ranks.values())) == 2
    win = json.loads((out / "windows.json").read_text())
    assert [w["kind"] for w in win["windows"]] == ["prefill", "mixed", "decode"]
    for w in win["windows"]:
        assert w["t_stop"] - w["t_start"] >= 0.45
        assert sorted(w["workers"]) == ["0", "1"]
    iters = json.loads((out / "iterations.json").read_text())
    kinds = {
        "prefill": lambda i: i["gen_req"] == 0 and i["ctx_tok"] == 2048,
        "mixed": lambda i: i["gen_req"] == 7 and i["ctx_tok"] == 1024,
        "decode": lambda i: i["gen_req"] == 8 and i["ctx_tok"] == 0,
    }
    for name, ok in kinds.items():
        assert any(ok(i) for i in iters), name
    # every window opened in the state it was asked for: the state machine reads
    # the iteration log, not a timer
    by_n = {i["iter"]: i for i in iters}
    for w in win["windows"]:
        assert any(
            kinds[w["kind"]](by_n[n])
            for n in range(w["iter_first"] - 3, w["iter_first"] + 1)
            if n in by_n
        ), w["kind"]
    # the priming window plus the three recorded ones signalled both workers
    events = [
        ln.split()[0] for ln in (tmp_path / "signals.log").read_text().splitlines()
    ]
    assert events.count("resume") == 8 and events.count("pause") == 8


@pytest.mark.skipif(not LINUX, reason="/proc and pgrep -s")
def test_refuses_the_other_slots_gpus(tmp_path):
    (tmp_path / "quiet").mkdir()
    r = _run(tmp_path, gpus="2,3")
    assert r.returncode == 2
    assert "refusing HIP GPU 2" in r.stdout
    assert not (tmp_path / "out/dry/serve.log").exists()
