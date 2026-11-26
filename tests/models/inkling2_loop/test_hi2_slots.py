# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test of the HI2 kit's slot gate, lease carrier and launcher protocol.

The live GPU is faked: a fake ``torch`` under a fake venv enumerates a table of
PCI buses, a stub ``serve_rdna.sh`` serves ``/health`` and refuses to run with a
foreign visibility filter or a free boot lock, and all paths live under a $HOME
scratch dir (``$HI2_TEST_BASE``, never /tmp).
No GPU, no torch and no network beyond 127.0.0.1 are needed.

``INKLING2_KIT_ROOT`` points the launcher text/logic tests at another kit tree
(the mutants runner uses that).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REAL_PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[3]
KIT = Path(os.environ.get("INKLING2_KIT_ROOT", REPO_ROOT / "tools" / "inkling_hi2"))
sys.path.insert(0, str(KIT))

import slot  # noqa: E402  (the sys.path line above finds the kit module)

GPU_TEMPS = REPO_ROOT / "tools/rdna/mono/w4a8/gpu_temps.py"
V620 = "Radeon PRO V620"
TABLE = {
    0: ("d1", V620),
    1: ("d4", V620),
    2: ("d9", V620),
    3: ("dc", V620),
    4: ("df", V620),
    5: ("e2", V620),
}
GIB = 1024 * 1024 * 1024


def _read(name: str) -> str:
    return (KIT / name).read_text()


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _test_tmp_base() -> Path:
    return Path(
        os.environ.get("HI2_TEST_BASE", Path.home() / ".cache" / "inkling2-hi2-tests")
    )


def _kill_processes_matching(pattern: str) -> None:
    """Kill processes whose full command line contains ``pattern`` (ours only)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "--", pattern], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return
    for token in result.stdout.split():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(token), signal.SIGKILL)


@pytest.fixture
def tmp_path(request):
    """Per-test scratch dir under ``$HOME``, never /tmp (the box keeps logs in $HOME).

    Overrides pytest's builtin so the fake kit, the stub servers and the thermal
    trees live under ``$HI2_TEST_BASE`` (default ``~/.cache/inkling2-hi2-tests``)
    and are removed after the test.
    """
    nodeid = request.node.nodeid
    name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in nodeid)[-100:]
    path = _test_tmp_base() / str(os.getpid()) / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _stub_servers_do_not_outlive_the_session():
    yield
    _kill_processes_matching(str(_test_tmp_base() / str(os.getpid())))


FAKE_INIT = """\
from . import cuda, accelerator
"""

FAKE_CUDA = """\
from ._fake import visible, props


def init():
    visible()


def get_device_properties(i):
    return props(i)
"""

FAKE_ACCELERATOR = """\
from ._fake import visible


def device_count():
    return len(visible())
"""

FAKE_IMPL = """\
import json
import os
import sys
from types import SimpleNamespace

_LOGGED = False


def visible():
    global _LOGGED
    with open(os.environ["FAKE_GPU_TABLE"]) as f:
        table = json.load(f)
    idx = list(range(len(table)))
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if value is None:
            continue
        idx = [idx[int(x)] for x in value.split(",") if x != ""]
    if not _LOGGED:
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(" ".join(f"{table[i][0]:02x}" for i in idx) + "\\n")
        _LOGGED = True
    fail = os.environ.get("FAKE_FAIL_IF_FILE")
    if fail and os.path.exists(fail):
        raise RuntimeError("fake: requested failure")
    return [table[i] for i in idx]


def props(i):
    bus, name = visible()[i]
    return SimpleNamespace(pci_bus_id=bus, name=name)
"""

FAKE_VENV_PY = """\
#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FAKETORCH${{PYTHONPATH:+:$PYTHONPATH}}"
exec "{python}" "$@"
"""

STUB_SERVE = """\
#!/usr/bin/env bash
set -u
: "${STUB_MARKERS:?}" "${STUB_HTTP:?}"
echo "STUB HIP=${HIP_VISIBLE_DEVICES:-<unset>}" \\
    "ROCR=${ROCR_VISIBLE_DEVICES:-<unset>} CUDA=${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'HIP=%s\\nROCR=%s\\nCUDA=%s\\n' \\
    "${HIP_VISIBLE_DEVICES:-}" "${ROCR_VISIBLE_DEVICES:-}" \\
    "${CUDA_VISIBLE_DEVICES:-}" > "$STUB_MARKERS/env.txt"
printf '%s\n' "$@" > "$STUB_MARKERS/args.txt"
if [[ ${STUB_REQUIRE_ENV:-1} == 1 ]]; then
    [[ ${HIP_VISIBLE_DEVICES:-} == 0,1 ]] || { echo "STUB bad HIP"; exit 3; }
    [[ -z ${ROCR_VISIBLE_DEVICES:-} && -z ${CUDA_VISIBLE_DEVICES:-} ]] || {
        echo "STUB inherited a visibility filter"; exit 3; }
fi
if flock -n "$BOOT_LOCK" true 2>/dev/null; then
    echo "STUB boot.lock FREE during boot"; exit 4
fi
echo "STUB boot.lock HELD during boot"
: > "$STUB_MARKERS/booting"
sleep "${STUB_BOOT_DELAY:-0}"
exec "${STUB_PYTHON:?}" "$STUB_HTTP" "$PORT"
"""

STUB_HTTP = """\
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = os.environ.get("STUB_CHAT_MODE", "paris")
CALLS = [0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "fake", "max_model_len": 8192}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        usage = {"prompt_tokens": 10, "completion_tokens": req.get("max_tokens", 1)}
        if self.path == "/v1/chat/completions":
            prompt = (req.get("messages") or [{}])[0].get("content") or ""
            if MODE == "reasoning":
                text = "The question asks for a capital; I will name the city."
            elif "capital of France" in prompt:
                text = "The capital of France is Paris."
            elif MODE == "vary":
                CALLS[0] += 1
                text = f"attempt {CALLS[0]}"
            else:
                text = "Paris"
            self._send({"choices": [{"message": {"content": text}}], "usage": usage})
        elif self.path == "/v1/completions":
            self._send({"choices": [{"text": "x"}], "usage": usage})
        else:
            self.send_error(404)


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

STUB_TEMPS = """\
import argparse
import csv
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--period", type=float, default=5.0)
ap.add_argument("--count", type=int, default=0)
ap.add_argument("--kill-pid", type=int, default=None)
ap.add_argument("--max-c", type=float, default=95.0)
ap.add_argument("--mean-c", type=float, default=90.0)
ap.add_argument("--mean-seconds", type=float, default=5.0)
args = ap.parse_args()
fresh = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
temp = float(os.environ.get("STUB_TEMP", "60"))
with open(args.out, "a", newline="") as f:
    w = csv.writer(f)
    if fresh:
        w.writerow(["time", "epoch", "hip0", "hip1"])
    w.writerow(["t", "1", f"{temp:.1f}", f"{temp:.1f}"])
    f.flush()
    rc = int(os.environ.get("STUB_TEMP_RC", "0"))
    if rc:
        sys.exit(rc)
    while True:
        time.sleep(5)
"""

STUB_KFD = """\
import json
import os
import sys

mode = os.environ.get("STUB_KFD_MODE", "ok")
args = sys.argv[1:]
if "--save" in args:
    path = args[args.index("--save") + 1]
    body = {} if mode == "missing" else {"1": 0}
    with open(path, "w") as f:
        json.dump(body, f)
elif "--since" in args:
    if mode == "missing":
        print("evicted_ms: no vLLM workers with a KFD context found")
    elif mode == "evict":
        print("evicted_ms 1=+5 -> INVALID (KFD evicted a worker's GPU queues)")
    else:
        print("evicted_ms 1=+0 -> ok")
else:
    print(json.dumps({} if mode == "missing" else {"1": 0}))
"""


class DryKit:
    """A fake six-V620 box: fake torch, stub server, sysfs, lease and quiet dirs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.model = root / "model"
        _write(self.model / "model.safetensors.index.json", "{}")
        _write(self.model / "tokenizer.json", "{}")
        self.venv = root / "venv"
        _write(
            self.venv / "bin/python",
            FAKE_VENV_PY.format(python=REAL_PYTHON),
            executable=True,
        )
        fake = root / "faketorch/torch"
        _write(fake / "__init__.py", FAKE_INIT)
        _write(fake / "cuda.py", FAKE_CUDA)
        _write(fake / "accelerator.py", FAKE_ACCELERATOR)
        _write(fake / "_fake.py", FAKE_IMPL)
        self.table = root / "table.json"
        _write(self.table, json.dumps([[0xD1, V620], [0xD4, V620]]))
        self.fake_log = root / "fake.log"
        self.markers = root / "markers"
        self.markers.mkdir()
        self.tree = root / "tree"
        _write(self.tree / "tools/rdna/serve_rdna.sh", STUB_SERVE, executable=True)
        self.http = root / "stub_http.py"
        _write(self.http, STUB_HTTP)
        self.temps_stub = root / "stub_temps.py"
        _write(self.temps_stub, STUB_TEMPS)
        self.kfd_stub = root / "stub_kfd.py"
        _write(self.kfd_stub, STUB_KFD)
        self.lease = root / "lease"
        self.quiet = root / "quiet-a2"
        self.run_dir = root / "run"
        for path in (self.lease, self.quiet, self.run_dir / "logs"):
            path.mkdir(parents=True, exist_ok=True)
        self.boot_lock = root / "boot.lock"
        self.boot_lock.touch()
        self.drm = root / "drm"
        self.set_vram_free(29)
        self.port = _free_port()
        self.procs: list[subprocess.Popen] = []

    def set_vram_free(self, free_gib: int) -> None:
        used = 30 - free_gib
        for index, bus in enumerate(("d1", "d4")):
            device = self.drm / f"pci/0000:{bus}:00.0"
            _write(device / "mem_info_vram_total", str(30 * GIB))
            _write(device / "mem_info_vram_used", str(used * GIB))
            card = self.drm / f"card{index}/device"
            card.parent.mkdir(parents=True, exist_ok=True)
            if card.is_symlink() or card.exists():
                card.unlink()
            os.symlink(f"../pci/0000:{bus}:00.0", card)

    def env(self, **overrides) -> dict[str, str]:
        scrub = (
            "SLOT",
            "OWNER",
            "HIP_VISIBLE_DEVICES",
            "EXPECT_PCI_BUSES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
        env = {k: v for k, v in os.environ.items() if k not in scrub}
        env.update(
            {
                "MODEL": str(self.model),
                "VENV": str(self.venv),
                "TREE": str(self.tree),
                "BOOT_LOCK": str(self.boot_lock),
                "QUIET_DIR": str(self.quiet),
                "LEASE_DIR": str(self.lease),
                "DRM_SYSFS": str(self.drm),
                "LEASE_FREE_GIB": "28",
                "LEASE_POLL_SECONDS": "0.2",
                "LEASE_WAIT_SECONDS": "20",
                "RUN_DIR": str(self.run_dir),
                "PORT": str(self.port),
                "READY_SECONDS": "30",
                "WAIT_SECONDS": "60",
                "BOOT_TIMEOUT_SECONDS": "10",
                "CELL_TIMEOUT": "30",
                "EVICT_TRIES": "1",
                "EVICT_POLL_SECONDS": "0.1",
                "FAKETORCH": str(self.root / "faketorch"),
                "FAKE_GPU_TABLE": str(self.table),
                "FAKE_LOG": str(self.fake_log),
                "STUB_MARKERS": str(self.markers),
                "STUB_HTTP": str(self.http),
                "STUB_PYTHON": REAL_PYTHON,
                "STUB_REQUIRE_ENV": "1",
                "STUB_BOOT_DELAY": "0",
                "STUB_TEMP": "60",
                "STUB_TEMP_RC": "0",
                "STUB_KFD_MODE": "ok",
                "TEMPS_TOOL": str(self.temps_stub),
                "KFD_TOOL": str(self.kfd_stub),
                "QUIET_TOUCH": REAL_PYTHON,
            }
        )
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def start_serve(self, **overrides) -> subprocess.Popen:
        proc = subprocess.Popen(
            ["bash", str(KIT / "serve.sh")],
            env=self.env(**overrides),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.procs.append(proc)
        return proc

    def run_serve(self, timeout: float = 40, **overrides):
        return subprocess.run(
            ["bash", str(KIT / "serve.sh")],
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_smoke(self, timeout: float = 90, **overrides):
        return subprocess.run(
            ["bash", str(KIT / "smoke.sh")],
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def lock_free(self) -> bool:
        return (
            subprocess.run(
                ["flock", "-n", str(self.boot_lock), "true"],
                capture_output=True,
            ).returncode
            == 0
        )

    def wait_for(self, path: Path, timeout: float = 10) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        self.procs.clear()
        # serve.sh backgrounds the stub server; a SIGTERM that races its trap or a
        # timed-out run can orphan it.  Sweep the exact script this kit wrote.
        _kill_processes_matching(str(self.http))


@pytest.fixture
def dry(tmp_path):
    kit = DryKit(tmp_path / "dry")
    try:
        yield kit
    finally:
        kit.stop()


def test_the_two_slot_table_is_the_agents_one():
    assert slot.default_pair("a2") == ("0,1", "d1,d4")
    assert slot.default_pair("b4") == ("2,3,4,5", "d9,dc,df,e2")
    with pytest.raises(slot.SlotError, match="unknown SLOT"):
        slot.slot_spec("a4")


def test_declared_pairs_of_both_slots_pass():
    slot.check_declared("a2", "0,1", "d1,d4")
    slot.check_declared("b4", "2,3,4,5", "d9,dc,df,e2")
    slot.check_declared("a2", "0, 1", "0xd1, 0000:d4:00.0")


def test_declared_mismatches_are_refused():
    with pytest.raises(slot.SlotError, match="SLOT=a2 is HIP 0,1"):
        slot.check_declared("a2", "0,2", "d1,d4")
    with pytest.raises(slot.SlotError, match="SLOT=a2 is PCI d1,d4"):
        slot.check_declared("a2", "0,1", "d1,dc")
    with pytest.raises(slot.SlotError, match="repeats"):
        slot.check_declared("a2", "0,0", "d1,d4")
    with pytest.raises(slot.SlotError, match="not a PCI bus"):
        slot.check_declared("a2", "0,1", "d1,zz")


def test_the_live_check_accepts_the_visible_pair():
    slot.check_visible("0,1", "d1,d4", {0: TABLE[0], 1: TABLE[1]})
    slot.check_visible(
        "2,3,4,5", "d9,dc,df,e2", {0: TABLE[2], 1: TABLE[3], 2: TABLE[4], 3: TABLE[5]}
    )


def test_the_live_check_refuses_a_moved_or_foreign_device():
    swapped = {0: ("d4", V620), 1: ("d1", V620)}
    with pytest.raises(slot.SlotError, match="shows PCI d4,d1"):
        slot.check_visible("0,1", "d1,d4", swapped)
    with pytest.raises(slot.SlotError, match="sees 6 GPUs"):
        slot.check_visible("0,1", "d1,d4", TABLE)
    with pytest.raises(slot.SlotError, match="not V620"):
        slot.check_visible("0,1", "d1,d4", {0: ("d1", "W7800"), 1: ("d4", V620)})
    with pytest.raises(slot.SlotError, match="lists 2 GPUs"):
        slot.check_visible("0,1", "d1", {0: TABLE[0], 1: TABLE[1]})
    with pytest.raises(slot.SlotError):
        slot.check_visible("0,5", "d1,d4", {0: TABLE[0], 1: TABLE[5]})


def test_the_live_check_uses_only_the_server_environment(monkeypatch):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(slot.SlotError, match="set HIP_VISIBLE_DEVICES"):
        slot.visible_table()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3,4,5")
    with pytest.raises(slot.SlotError, match="unset ROCR_VISIBLE_DEVICES"):
        slot.visible_table()
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5")
    with pytest.raises(slot.SlotError, match="unset CUDA_VISIBLE_DEVICES"):
        slot.visible_table()
    source = _read("slot.py")
    assert "os.environ.pop" not in source
    assert "torch.cuda.init()" in source  # the count is 0 until init on 2.12+rocm
    assert "accelerator.device_count()" in source


def test_bus_names_normalise():
    assert slot.normalise_bus("d1") == "d1"
    assert slot.normalise_bus("0xD1") == "d1"
    assert slot.normalise_bus("0000:d1:00.0") == "d1"
    assert slot.normalise_bus(0x0A) == "0a"
    assert slot.normalise_bus(0xD1) == "d1"
    slot.check_visible("0,1", "d1,d4", {0: (0xD1, V620), 1: (0xD4, V620)})


def test_the_cli_checks_without_touching_torch(monkeypatch, capsys):
    for name in ("SLOT", "HIP_VISIBLE_DEVICES", "EXPECT_PCI_BUSES"):
        monkeypatch.delenv(name, raising=False)
    assert slot.main(["--slot", "a2", "--declared-only"]) == 0
    assert "SLOT=a2 HIP 0,1 -> PCI d1,d4" in capsys.readouterr().out
    assert (
        slot.main(
            [
                "--slot",
                "a2",
                "--hip",
                "0,2",
                "--expect-buses",
                "d1,d4",
                "--declared-only",
            ]
        )
        == 2
    )
    assert "SLOT=a2 is HIP 0,1" in capsys.readouterr().err
    assert slot.main(["--slot", "a2", "--print"]) == 0
    assert capsys.readouterr().out.split() == [
        "HIP_VISIBLE_DEVICES=0,1",
        "EXPECT_PCI_BUSES=d1,d4",
    ]


# --- launcher behaviour against the fake box (no GPU, no real torch) ---------


def test_serve_refuses_another_jobs_holder(dry):
    """Default OWNER must not match the live HI2Inkling.a2 holder of a job."""
    (dry.lease / "HI2Inkling.a2").write_text("another job\n")
    result = dry.run_serve()
    assert result.returncode != 0
    assert "HI2Inkling.a2 holds the a2 lease" in result.stdout + result.stderr
    assert not (dry.lease / "HI2InklingRuntime.a2").exists()
    assert not (dry.lease / "a2.request").exists()
    assert not dry.fake_log.exists() or dry.fake_log.read_text() == ""


def test_serve_takes_and_releases_its_own_holder(dry):
    proc = dry.start_serve()
    assert dry.wait_for(dry.markers / "booting")
    assert _wait_port(dry.port)
    assert (dry.lease / "HI2InklingRuntime.a2").exists()
    assert not (dry.lease / "a2.request").exists()
    proc.terminate()
    proc.wait(timeout=15)
    assert not (dry.lease / "HI2InklingRuntime.a2").exists()


def test_serve_waits_for_free_vram(dry):
    dry.set_vram_free(5)
    result = dry.run_serve(LEASE_WAIT_SECONDS="1", LEASE_POLL_SECONDS="0.2")
    assert result.returncode != 0
    assert "MiB free per GPU" in result.stdout + result.stderr
    assert not (dry.lease / "HI2InklingRuntime.a2").exists()
    assert not (dry.lease / "a2.request").exists()


def test_serve_never_overwrites_another_owners_request(dry):
    """The lane README has one <slot>.request; a foreign one is queued behind."""
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        request = dry.lease / "a2.request"
        original = f"OtherAgent pid {sleeper.pid} since 2026-11-25T09:00:00+00:00\n"
        request.write_text(original)
        result = dry.run_serve(LEASE_WAIT_SECONDS="1", LEASE_POLL_SECONDS="0.2")
        out = result.stdout + result.stderr
        assert result.returncode != 0
        assert request.read_text() == original, (
            "another owner's request was overwritten"
        )
        assert "a2.request belongs to OtherAgent" in out
        assert "queueing" in out
        assert not (dry.lease / "HI2InklingRuntime.a2").exists()
        assert not dry.fake_log.exists() or dry.fake_log.read_text() == ""
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_serve_drops_a_stale_own_holder_and_still_waits_for_vram(dry):
    dead = subprocess.Popen(["sleep", "0.01"])
    dead.wait()
    holder = dry.lease / "HI2InklingRuntime.a2"
    holder.write_text(
        f"HI2InklingRuntime pid {dead.pid} since 2026-11-25T09:00:00+00:00\n"
    )
    dry.set_vram_free(5)
    result = dry.run_serve(LEASE_WAIT_SECONDS="1", LEASE_POLL_SECONDS="0.2")
    out = result.stdout + result.stderr
    assert result.returncode != 0
    assert "dropping stale holder HI2InklingRuntime.a2" in out
    assert "MiB free per GPU" in out
    assert "reusing our holder" not in out
    assert not holder.exists()
    assert not (dry.lease / "a2.request").exists()


def test_serve_rechecks_free_vram_before_reusing_a_live_own_holder(dry):
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        holder = dry.lease / "HI2InklingRuntime.a2"
        holder.write_text(
            f"HI2InklingRuntime pid {sleeper.pid} since 2026-11-25T09:00:00+00:00\n"
        )
        dry.set_vram_free(5)
        result = dry.run_serve(LEASE_WAIT_SECONDS="1", LEASE_POLL_SECONDS="0.2")
        out = result.stdout + result.stderr
        assert result.returncode != 0
        assert "MiB free per GPU" in out
        assert "reusing our holder" not in out
        assert holder.exists(), "another live job's holder must not be removed"
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_serve_holds_the_boot_lock_until_ready(dry):
    dry.start_serve(STUB_BOOT_DELAY="3")
    assert dry.wait_for(dry.markers / "booting")
    assert not dry.lock_free(), "boot.lock was released before /health"
    assert _wait_port(dry.port)
    deadline = time.monotonic() + 10
    while not dry.lock_free() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert dry.lock_free(), "boot.lock was still held after /health"
    (dry.markers / "booting").unlink()


def test_serve_lets_the_recipe_set_the_spec_config(dry):
    # DSPARK=1 alone must reach serve_rdna without an empty SPEC_CONFIG
    # override: the recipe builds the dspark block and an empty override would
    # wipe it (speculative_config=None, 0 drafts).
    dry.start_serve(DSPARK="1")
    assert _wait_port(dry.port)
    args = (dry.markers / "args.txt").read_text().splitlines()
    assert "DSPARK=1" in args, args
    assert not any(a.startswith("SPEC_CONFIG=") for a in args), args
    (dry.markers / "booting").unlink()


def test_serve_forwards_an_explicit_spec_config(dry):
    spec = '{"method":"dspark"}'
    dry.start_serve(DSPARK="1", SPEC_CONFIG=spec)
    assert _wait_port(dry.port)
    args = (dry.markers / "args.txt").read_text().splitlines()
    assert "DSPARK=1" in args, args
    assert f"SPEC_CONFIG={spec}" in args, args
    (dry.markers / "booting").unlink()


def test_serve_checks_the_devices_around_the_boot(dry):
    dry.start_serve(ROCR_VISIBLE_DEVICES="2,3,4,5", CUDA_VISIBLE_DEVICES="2,3")
    assert _wait_port(dry.port)
    deadline = time.monotonic() + 15
    while len(dry.fake_log.read_text().splitlines()) < 3:
        assert time.monotonic() < deadline, "the post-/health device check did not run"
        time.sleep(0.1)
    # before the quiet wait, immediately before the launch, after /health
    assert dry.fake_log.read_text().splitlines() == ["d1 d4", "d1 d4", "d1 d4"]
    env = (dry.markers / "env.txt").read_text()
    assert "HIP=0,1" in env
    assert "ROCR=\n" in env
    assert "CUDA=\n" in env
    (dry.markers / "booting").unlink()


def test_serve_rechecks_the_devices_after_health(dry):
    # The pre-launch checks pass (the marker appears when the stub boots); the
    # check after /health sees the marker and must kill the server: a moved device
    # list while the server came up is not a ready boot.
    result = dry.run_serve(FAKE_FAIL_IF_FILE=str(dry.markers / "booting"))
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "fake: requested failure" in out
    assert "the device check after /health failed" in out
    assert "READY" not in result.stdout


def test_smoke_passes_with_cool_temps_and_ok_evictions(dry):
    result = dry.run_smoke()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT greedy PASS" in result.stdout
    assert "RESULT capsule PASS" in result.stdout
    assert "RESULT measured window PASS" in result.stdout
    assert not (dry.lease / "HI2InklingRuntime.a2").exists()
    assert not (dry.quiet / "inkling-hi2-smoke").exists()


def test_smoke_runs_the_same_probe_on_the_stock_server(dry):
    # The stub serves both roles here: the code path prints both outputs and the
    # first-divergence position (identical texts -> divergence at the end).
    result = dry.run_smoke(STOCK_URL=f"http://127.0.0.1:{dry.port}", STOCK_KL="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "greedy: stock" in result.stdout
    assert "first divergence at char" in result.stdout
    assert "RESULT greedy PASS" in result.stdout


def test_smoke_fails_when_the_greedy_probe_fails(dry):
    result = dry.run_smoke(STUB_CHAT_MODE="reasoning")
    out = result.stdout + result.stderr
    assert result.returncode != 0, "smoke.sh exited 0 after a failed assertion"
    assert "does not name Paris" in out
    assert "RESULT greedy PASS" not in result.stdout
    assert "RESULT measured window PASS" not in result.stdout
    assert not (dry.lease / "HI2InklingRuntime.a2").exists()
    assert not (dry.quiet / "inkling-hi2-smoke").exists()


def test_smoke_fails_when_the_capsule_does_not_repeat(dry):
    result = dry.run_smoke(STUB_CHAT_MODE="vary")
    out = result.stdout + result.stderr
    assert result.returncode != 0, "smoke.sh exited 0 after a failed assertion"
    assert "RESULT greedy PASS" in result.stdout
    assert "RESULT capsule PASS" not in result.stdout
    assert "not a deterministic repeat" in out
    assert "RESULT measured window PASS" not in result.stdout


def test_smoke_rechecks_the_devices_before_each_cell(dry):
    result = dry.run_smoke(FAKE_FAIL_IF_FILE=str(dry.markers / "booting"))
    assert result.returncode != 0
    assert "fake: requested failure" in result.stdout + result.stderr


def test_smoke_window_is_invalid_without_kfd_workers(dry):
    result = dry.run_smoke(STUB_KFD_MODE="missing")
    assert result.returncode != 0
    assert "no KFD worker context" in result.stdout + result.stderr
    assert "RESULT measured window INVALID" in result.stdout


def test_smoke_window_is_invalid_when_hot(dry):
    result = dry.run_smoke(STUB_TEMP="96")
    assert result.returncode != 0
    assert "RESULT measured window INVALID" in result.stdout
    assert "max junction 96.0 C" in result.stdout


def test_quiet_touch_exits_on_sigterm(dry):
    quiet = dry.root / "quiet-touch-test"
    proc = subprocess.Popen(
        [
            REAL_PYTHON,
            str(KIT / "quiet_touch.py"),
            "--file",
            str(quiet),
            "--interval",
            "60",
            "--count",
            "3",
        ]
    )
    time.sleep(0.3)
    proc.terminate()
    assert proc.wait(timeout=5) == 0
    assert not quiet.exists()


# --- real gpu_temps watcher against a fake sysfs -----------------------------


def _thermal_tree(root: Path, temp_c: float) -> tuple[Path, Path]:
    nodes = root / "kfd/nodes"
    _write(nodes / "0/properties", "simd_count 64\ndrm_render_minor 128\n")
    drm = root / "drm"
    hwmon = drm / "renderD128/device/hwmon/hwmon0"
    _write(hwmon / "temp1_label", "junction\n")
    _write(hwmon / "temp1_input", f"{int(temp_c * 1000)}\n")
    return nodes, drm


def test_gpu_temps_logs_and_exits_after_count(tmp_path):
    nodes, drm = _thermal_tree(tmp_path, 60)
    out = tmp_path / "temps.csv"
    result = subprocess.run(
        [
            REAL_PYTHON,
            str(GPU_TEMPS),
            "--gpus",
            "0",
            "--out",
            str(out),
            "--period",
            "0.05",
            "--count",
            "2",
            "--kfd-nodes",
            str(nodes),
            "--drm",
            str(drm),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(out.read_text().splitlines()) == 3  # header + 2 rows


def test_gpu_temps_watcher_kills_on_a_hot_sample(tmp_path):
    nodes, drm = _thermal_tree(tmp_path, 96)
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        result = subprocess.run(
            [
                REAL_PYTHON,
                str(GPU_TEMPS),
                "--gpus",
                "0",
                "--out",
                str(tmp_path / "hot.csv"),
                "--period",
                "0.05",
                "--kfd-nodes",
                str(nodes),
                "--drm",
                str(drm),
                "--kill-pid",
                str(sleeper.pid),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 3
        assert "95 C" in result.stderr
        deadline = time.monotonic() + 5
        while sleeper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sleeper.poll() is not None, "the watcher did not kill the server pid"
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()


def test_gpu_temps_watcher_trips_on_the_mean(tmp_path):
    nodes, drm = _thermal_tree(tmp_path, 91)
    result = subprocess.run(
        [
            REAL_PYTHON,
            str(GPU_TEMPS),
            "--gpus",
            "0",
            "--out",
            str(tmp_path / "mean.csv"),
            "--period",
            "5",
            "--max-c",
            "99",
            "--mean-c",
            "90",
            "--kfd-nodes",
            str(nodes),
            "--drm",
            str(drm),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 3
    assert "mean" in result.stderr


# --- HTTP clients: b4 refused, eviction evidence required -------------------


def test_clients_refuse_b4(monkeypatch):
    monkeypatch.setenv("SLOT", "b4")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3,4,5")
    monkeypatch.setenv("EXPECT_PCI_BUSES", "d9,dc,df,e2")
    bench = importlib.import_module("bench")
    kl_probe = importlib.import_module("kl_probe")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--url", "http://127.0.0.1:9"])
    assert bench.main() == 2
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kl_probe.py",
            "--hi2-url",
            "http://127.0.0.1:9",
            "--stock-url",
            "http://127.0.0.1:9",
            "--ids",
            "unused.npy",
        ],
    )
    assert kl_probe.main() == 2


def test_kl_probe_scores_from_position_one():
    # vLLM returns None for the first prompt position (no context).  The old
    # parser indexed it and raised AttributeError; the probe must score from 1.
    kl = importlib.import_module("kl_probe")
    choice = {
        "logprobs": {
            "tokens": ["a", "b", "c"],
            "top_logprobs": [
                None,
                {"a": -0.1, "b": -1.0},
                {"a": -0.2},
            ],
        }
    }
    assert kl.top_distribution(choice, 0) is None
    assert kl.top_distribution(choice, 1) == {"a": -0.1, "b": -1.0}
    assert kl.top_entry_size(choice) == 2
    values, agrees, skipped, considered = kl.score_window(choice, choice, 3)
    assert (skipped, considered, len(values), agrees) == (0, 2, 2, 2)


def test_kl_probe_skips_omitted_positions():
    kl = importlib.import_module("kl_probe")
    gaps = {"logprobs": {"top_logprobs": [None, {"a": -0.1}, None, {"a": -0.2}]}}
    full = {"logprobs": {"top_logprobs": [None, {"a": -0.1}, {"a": -0.2}, {"a": -0.3}]}}
    values, agrees, skipped, considered = kl.score_window(gaps, full, 4)
    assert considered == 3
    assert skipped == 1
    assert len(values) == 2
    assert agrees == 2
    empty = {"logprobs": {"top_logprobs": [None]}}
    assert kl.score_window(empty, empty, 2)[0] == []
    assert kl.top_entry_size(empty) == 0


def test_bench_window_record_requires_eviction_evidence():
    bench = importlib.import_module("bench")
    ok = bench.window_record(60.0, "evicted_ms 1=+0 -> ok", True)
    assert ok["valid"]
    no_workers = bench.window_record(
        60.0, "evicted_ms: no vLLM workers with a KFD context found", True
    )
    assert not no_workers["valid"]
    unsampled = bench.window_record(60.0, None, False)
    assert not unsampled["valid"]
    hot = bench.window_record(96.0, "evicted_ms 1=+0 -> ok", True)
    assert not hot["valid"]
    evicted = bench.window_record(60.0, "evicted_ms 1=+5 -> INVALID", True)
    assert not evicted["valid"]


def test_bench_invalid_without_kfd_workers(dry):
    server = subprocess.Popen(
        [REAL_PYTHON, str(dry.http), str(dry.port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_port(dry.port)
        env = dry.env(
            SLOT="a2",
            HIP_VISIBLE_DEVICES="0,1",
            EXPECT_PCI_BUSES="d1,d4",
            STUB_KFD_MODE="missing",
        )
        result = subprocess.run(
            [
                REAL_PYTHON,
                str(KIT / "bench.py"),
                "--url",
                f"http://127.0.0.1:{dry.port}",
                "--label",
                "test",
                "--batches",
                "1",
                "--prefill-tokens",
                "10",
                "--decode-tokens",
                "1",
                "--server-pid",
                str(server.pid),
                "--run-dir",
                str(dry.run_dir),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "INVALID" in result.stderr
    finally:
        server.terminate()
        server.wait(timeout=10)


# --- launcher text contracts (fast guards for the scripts themselves) --------


def test_the_launchers_unset_the_other_visibility_vars():
    for name in ("serve.sh", "smoke.sh"):
        text = _read(name)
        assert "unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES" in text, name
        assert "env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES" in text, name


def test_the_boot_lock_is_held_until_the_server_is_ready():
    serve = _read("serve.sh")
    assert "exec 9>" in serve
    assert "flock 9" in serve
    assert serve.index("exec 9>&-") > serve.index("READY_SECONDS")
    assert 'EXTRA_ARGS="$EXTRA" "${spec_args[@]}" 9>&-' in serve
    assert 'flock -w 900 "$BOOT_LOCK" true' not in serve
    assert "quiet window opened under the boot lock" in serve


def test_the_launchers_check_the_lease_and_take_an_owner():
    serve = _read("serve.sh")
    smoke = _read("smoke.sh")
    lease = _read("lease.sh")
    assert "OWNER=${OWNER:-HI2InklingRuntime}" in serve
    assert "OWNER=${OWNER:-HI2InklingRuntime}" in smoke
    assert "lease_acquire" in serve
    assert 'lease_refuse_foreign "smoke.sh"' in smoke
    assert "holds the $SLOT lease" in lease
    assert "REQUEST" in lease and "trap lease_release EXIT" in lease
    assert "LEASE_FREE_GIB" in lease and "mem_info_vram_used" in lease


def test_the_launchers_confirm_the_lease_before_touching_the_gpus():
    serve = _read("serve.sh")
    smoke = _read("smoke.sh")
    assert "\ncheck_devices\n" not in serve.split("lease_acquire")[0]
    assert "\ncheck_devices\n" in serve.split("lease_acquire")[-1]
    assert "\ncheck_devices\n" not in smoke.split("lease_refuse_foreign")[0]


def test_smoke_retouches_quiet_and_samples_the_window():
    smoke = _read("smoke.sh")
    assert "quiet_touch.py" in smoke
    assert "--interval 300" in smoke
    assert "sleep 300" not in smoke
    assert "gpu_temps.py" in smoke or "TEMPS_TOOL" in smoke
    assert "kfd_evictions.py" in smoke or "KFD_TOOL" in smoke
    assert "--kill-pid" in smoke
    assert "no vLLM KFD context" in smoke
    assert 'timeout "$CELL_TIMEOUT"' in smoke
    assert "READY_SECONDS" in smoke


def test_bench_samples_the_window():
    bench = _read("bench.py")
    assert "TEMPS_TOOL" in bench
    assert "KFD_TOOL" in bench
    assert "no vLLM worker KFD context" in bench
    assert "--kill-pid" in bench
    assert "temperature watcher exited" in bench


def test_the_hi2_recipe_pins_the_a2_buses():
    recipe = (
        REPO_ROOT / "tools" / "rdna" / "recipes" / "inkling2-hi2-tp2.env"
    ).read_text()
    assert 'PCI_BUSES="d1 d4"' in recipe
    assert "HIP_VISIBLE_DEVICES=0,1" not in recipe
