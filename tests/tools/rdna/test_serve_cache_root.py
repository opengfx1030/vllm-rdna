# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default VLLM_CACHE_ROOT is per GPU set + port; the stale-server kill stays inside it.

Two servers of one tree booted in parallel on different cards used to share
``<tree>/cache/vllm``, so the second launch's ``rdna_kill_stale`` stopped the first.
These tests need no GPU: ``serve_rdna.sh PRINT=1`` shows the resolved root, and
``rdna_kill_stale`` runs against a fake process table (``RDNA_PROC_ROOT``) with
``pgrep``, ``kill`` and ``sleep`` stubbed on PATH.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
COMMON = REPO / "tools/rdna/rdna_launcher_common.sh"
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "VLLM_CACHE_ROOT",
    "PORT",
)


@pytest.fixture()
def venv(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    return tmp_path / "venv"


def printed_root(venv, **env):
    full = {k: v for k, v in os.environ.items() if k not in DROPPED}
    # Every recipe sets PORT; a caller-set PORT wins, so pin it.
    full.update(VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", PORT="18094")
    full.update(env)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "RECIPE=inkling2-mini-tp1"],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("export VLLM_CACHE_ROOT="):
            return line.split("=", 1)[1].strip("'")
    raise AssertionError(result.stdout)


def test_default_root_differs_per_hip_list(venv):
    a = printed_root(venv, HIP_VISIBLE_DEVICES="0,1")
    b = printed_root(venv, HIP_VISIBLE_DEVICES="2,3")
    assert a != b
    assert a == f"{REPO}/cache/vllm-0_1-18094"
    assert b == f"{REPO}/cache/vllm-2_3-18094"


def test_default_root_differs_per_port(venv):
    a = printed_root(venv, HIP_VISIBLE_DEVICES="0", PORT="18001")
    b = printed_root(venv, HIP_VISIBLE_DEVICES="0", PORT="18002")
    assert a == f"{REPO}/cache/vllm-0-18001"
    assert b == f"{REPO}/cache/vllm-0-18002"


def test_explicit_root_is_respected(venv, tmp_path):
    root = str(tmp_path / "mine")
    assert printed_root(venv, HIP_VISIBLE_DEVICES="0,1", VLLM_CACHE_ROOT=root) == root
    assert printed_root(venv, HIP_VISIBLE_DEVICES="2,3", VLLM_CACHE_ROOT=root) == root


def fake_host(tmp_path, servers):
    """A process table (pid -> VLLM_CACHE_ROOT) plus stub pgrep / kill / sleep."""
    proc = tmp_path / "proc"
    for pid, root in servers.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "environ").write_bytes(
            f"PATH=/usr/bin\0VLLM_CACHE_ROOT={root}\0HOME=/root\0".encode()
        )
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "kill.log"
    (stubs / "pgrep").write_text(f"#!/bin/sh\nls {proc}\n")
    (stubs / "kill").write_text(f'#!/bin/sh\necho "$*" >> {log}\n')
    (stubs / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    return proc, stubs, log


def killed(tmp_path, servers, root, hip, port="18094"):
    proc, stubs, log = fake_host(tmp_path, servers)
    # `kill` is a bash builtin, so shadow it with a function that logs instead.
    script = (
        f'source "{COMMON}"; '
        f"kill() {{ echo \"$*\" >> '{log}'; }}; "
        f'source_dir="{REPO}"; PORT={port}; HIP_VISIBLE_DEVICES={hip}; '
        + (f'VLLM_CACHE_ROOT="{root}"; ' if root else "")
        + "rdna_default_cache_root; "
        + "printf '%s\\n' \"$VLLM_CACHE_ROOT\"; rdna_kill_stale"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("VLLM_CACHE_ROOT", "PORT")}
    env.update(PATH=f"{stubs}:{os.environ['PATH']}", RDNA_PROC_ROOT=str(proc))
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 0, result.stderr
    lines = log.read_text().split("\n") if log.exists() else []
    pids = sorted({int(line.split()[-1]) for line in lines if line})
    return result.stdout.strip(), pids, lines


def test_kill_touches_only_its_own_default_root(tmp_path):
    first = f"{REPO}/cache/vllm-0_1-18094"
    second = f"{REPO}/cache/vllm-2_3-18094"
    servers = {101: first, 102: first, 201: second, 301: f"{REPO}/cache/vllm"}
    # Booting cards 2,3 must not touch the server on cards 0,1 (or the legacy root).
    root, pids, _ = killed(tmp_path, servers, None, "2,3")
    assert root == second
    assert pids == [201]
    # A relaunch on cards 0,1 replaces only its own stale processes (TERM then KILL).
    (tmp_path / "again").mkdir()
    root, pids, lines = killed(tmp_path / "again", servers, None, "0,1")
    assert root == first
    assert pids == [101, 102]
    assert [line.split()[0] for line in lines if line] == ["-TERM"] * 2 + ["-KILL"] * 2


def test_kill_is_scoped_by_port(tmp_path):
    servers = {
        11: f"{REPO}/cache/vllm-0-18001",
        12: f"{REPO}/cache/vllm-0-18002",
    }
    root, pids, _ = killed(tmp_path, servers, None, "0", port="18002")
    assert root == f"{REPO}/cache/vllm-0-18002"
    assert pids == [12]


def test_explicit_root_kill_is_unchanged(tmp_path):
    shared = "/data/shared-cache"
    servers = {
        1: shared,
        2: shared,
        3: f"{REPO}/cache/vllm-0_1-18094",
        4: "/data/shared-cache-other",
    }
    root, pids, _ = killed(tmp_path, servers, shared, "0,1")
    assert root == shared
    assert pids == [1, 2]
