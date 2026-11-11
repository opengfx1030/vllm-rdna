# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""serve_validate.sh renews its quiet window between cells (QUIET_WIN_MAX).

The real script runs against a stub tree: a launcher that starts a tiny HTTP
server, a venv whose python fakes `bench serve` and the probes, and a private
quiet directory. Nothing touches a GPU, the shared quiet directory or a lock.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PORT_DIR = REPO / "tools/rdna/port_v031"

SERVER = """
import http.server, json, sys


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self):
        body = json.dumps({"data": [{"id": "stub"}], "success": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply()

    def log_message(self, *args):
        pass


addr = ("127.0.0.1", int(sys.argv[1]))
http.server.ThreadingHTTPServer(addr, Handler).serve_forever()
"""

LAUNCHER = """#!/usr/bin/env bash
for a in "$@"; do [[ $a == PORT=* ]] && PORT=${a#PORT=}; done
exec python3 "$(dirname "$0")/stub_server.py" "$PORT"
"""

# Fakes `bench serve` (sleeps, then writes the result file) and every probe script.
PYTHON = """#!/usr/bin/env bash
if [[ $1 == -m ]]; then
    while (($#)); do
        case $1 in --result-dir) dir=$2 ;; --result-filename) file=$2 ;; esac
        shift
    done
    sleep "${STUB_BENCH_S:-2}"
    cp "$STUB_BENCH_JSON" "$dir/$file"
    exit 0
fi
[[ $1 == *.py ]] && exit 0
exec python3 "$@"
"""

# A host without setsid (macOS) gets the same call through perl.
SETSID = """#!/usr/bin/env bash
exec perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' "$@"
"""

BENCH = {
    "completed": 4,
    "failed": 0,
    "duration": 2.0,
    "total_input_tokens": 64,
    "max_concurrency": 1,
    "mean_ttft_ms": 100.0,
    "median_ttft_ms": 100.0,
    "p99_ttft_ms": 120.0,
    "mean_tpot_ms": 20.0,
    "median_itl_ms": 20.0,
    "mean_itl_ms": 20.0,
    "p99_itl_ms": 30.0,
    "output_throughput": 50.0,
}


def write_exe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def run_cells(tmp_path: Path, **env: str) -> list[str]:
    tree = tmp_path / "tree"
    port_dir = tree / "tools/rdna/port_v031"
    port_dir.mkdir(parents=True)
    for name in ("serve_validate.sh", "bench_metrics.py", "kfd_evictions.py"):
        shutil.copy(PORT_DIR / name, port_dir / name)
    write_exe(tree / "tools/rdna/serve_rdna.sh", LAUNCHER)
    (tree / "tools/rdna/stub_server.py").write_text(SERVER)
    write_exe(tmp_path / "venv/bin/python", PYTHON)
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(BENCH))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if shutil.which("setsid") is None:
        write_exe(bindir / "setsid", SETSID)
    out = tmp_path / "out"
    full = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "VENV": str(tmp_path / "venv"),
        "STUB_BENCH_JSON": str(bench),
        "GPUS": "2",
        "REFUSED_GPUS": "",
        "RECIPE": "unit",
        "MODEL": "/nonexistent-model",
        "PORT": str(18900 + os.getpid() % 90),
        "OUT": str(out),
        "CACHE": str(tmp_path / "cache"),
        "CELLS": "16:8:1 32:8:1 48:8:1",
        "PPL_PROBE": "0",
        "PREFIX_PROBE": "0",
        "WARMUP": "0",
        "CONC_PROBE": "0",
        "STOP_GRACE": "2",
        "QUIET_DIR": str(tmp_path / "quiet"),
        "QUIET_OWN": "Tester",
        "QUIET_HOLD": "1",
        "QUIET_TOUCH": "1",
        "QUIET_POLL": "1",
        **env,
    }
    (tmp_path / "quiet").mkdir()
    res = subprocess.run(
        ["bash", str(port_dir / "serve_validate.sh")],
        capture_output=True,
        text=True,
        env=full,
        timeout=180,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert not (tmp_path / "quiet/Tester").exists()  # the window is closed at the end
    return (out / "summary.txt").read_text().splitlines()


def events(lines: list[str]) -> list[str]:
    keys = (
        "quiet window open",
        "quiet window closed",
        "quiet gap",
        "bench bench-",
        "STALL PROBE",
    )
    return [
        next(k for k in keys if k in line)
        for line in lines
        if any(k in line for k in keys)
    ]


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs setsid or perl")
def test_window_is_renewed_between_cells_once_it_is_too_old(tmp_path):
    lines = run_cells(tmp_path, QUIET_WIN_MAX="1", QUIET_GAP="1")
    assert events(lines) == [
        "quiet window open",
        "bench bench-",
        "quiet window closed",
        "quiet gap",
        "quiet window open",
        "bench bench-",
        "quiet window closed",
        "quiet gap",
        "quiet window open",
        "bench bench-",
        "quiet window closed",
    ]
    assert any("1 s without a window between cells" in line for line in lines)


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs setsid or perl")
def test_window_stays_open_for_every_cell_without_a_limit(tmp_path):
    lines = run_cells(tmp_path)
    assert events(lines) == [
        "quiet window open",
        "bench bench-",
        "bench bench-",
        "bench bench-",
        "quiet window closed",
    ]


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs setsid or perl")
def test_stall_probe_runs_inside_the_window_of_the_last_cell(tmp_path):
    # The stub python exits 0 for every *.py, so the probe's verdict is ERROR:
    # only where it runs is under test.
    lines = run_cells(tmp_path, CELLS="16:8:1", STALL_PROBE="1", STALL_ITER_LOG="0")
    assert events(lines) == [
        "quiet window open",
        "bench bench-",
        "STALL PROBE",
        "quiet window closed",
    ]


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs setsid or perl")
def test_stall_probe_gets_a_fresh_window_when_the_cells_aged_it(tmp_path):
    lines = run_cells(
        tmp_path,
        CELLS="16:8:1",
        STALL_PROBE="1",
        STALL_ITER_LOG="0",
        QUIET_WIN_MAX="1",
        QUIET_GAP="1",
    )
    assert events(lines) == [
        "quiet window open",
        "bench bench-",
        "quiet window closed",
        "quiet gap",
        "quiet window open",
        "STALL PROBE",
        "quiet window closed",
    ]
