# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and describe the ``p2p_probe`` binary (one fat binary, two archs).

``build`` compiles ``tools/rdna/p2p_probe/p2p_probe.cpp`` for ``gfx1030`` (V620)
and ``gfx1100`` (W7800) into ``~/Apps/pciescope/bin/p2p_probe``, under the box
build lock (``flock ~/w4a8_runs/audit/build.lock nice -n 19``). The binary is
written next to its final name and renamed into place only after
``p2p_probe version`` ran, so a failed build never replaces a working binary.
A stamp (``p2p_probe.stamp.json``) records the source hash, git commit, archs,
compiler version and the ``version`` line; ``binary_status`` compares it with
the current source.

Command line (on the box, inside the pciescope venv)::

    python -m pciescope.probe build      # compile (waits for the lock)
    python -m pciescope.probe status     # path, version, stamp, stale or not
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runner import default_binary, probe_version

ARCHS = ("gfx1030", "gfx1100")
DEFAULT_LOCK = Path("~/w4a8_runs/audit/build.lock").expanduser()
STAMP_SUFFIX = ".stamp.json"
LOCK_WAIT_S = 3000
HIPCC_FALLBACK = Path("/opt/rocm/bin/hipcc")


def source_path() -> Path:
    """The probe source of this checkout (``tools/rdna/p2p_probe/p2p_probe.cpp``)."""
    return Path(__file__).resolve().parents[2] / "p2p_probe" / "p2p_probe.cpp"


def stamp_path(binary: Path) -> Path:
    return binary.with_name(binary.name + STAMP_SUFFIX)


def sha256_of(path: Path) -> str | None:
    """Hex SHA-256 of a file, None when unreadable."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def git_commit(directory: Path) -> str:
    """Commit hash of the checkout that holds ``directory`` ('' when unknown)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def find_hipcc() -> str | None:
    """Path of ``hipcc`` (PATH first, then the ROCm default)."""
    found = shutil.which("hipcc")
    if found:
        return found
    return str(HIPCC_FALLBACK) if HIPCC_FALLBACK.exists() else None


def read_stamp(binary: Path) -> dict[str, Any] | None:
    """Return the build stamp of ``binary`` (None when missing or invalid)."""
    try:
        data = json.loads(stamp_path(binary).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class BinaryStatus:
    """What is known about the probe binary."""

    path: str
    exists: bool
    runnable: bool = False
    version: str = ""
    modes: list[str] = field(default_factory=list)
    stamp: dict[str, Any] | None = None
    stale: bool | None = None  # built from other source than the checkout's
    problem: str = ""  # why the binary cannot be used ('' when it can)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def binary_status(
    binary: Path | None = None, source: Path | None = None
) -> BinaryStatus:
    """Describe ``binary``: present, runs (``version``), stamp, stale or not."""
    path = binary if binary is not None else default_binary()
    status = BinaryStatus(path=str(path), exists=path.is_file())
    if not status.exists:
        status.problem = (
            f"{path} does not exist; build it with: python -m pciescope.probe build"
        )
        return status
    if not os.access(path, os.X_OK):
        status.problem = f"{path} is not executable"
        return status
    version = probe_version(path)
    if version is None:
        status.problem = (
            f"{path} does not answer `version`: an old build, rebuild it with: "
            "python -m pciescope.probe build"
        )
        return status
    status.runnable = True
    status.version = version.version
    status.modes = list(version.modes)
    status.stamp = read_stamp(path)
    src = source if source is not None else source_path()
    current = sha256_of(src)
    if status.stamp is not None and current is not None:
        status.stale = status.stamp.get("source_sha256") != current
    return status


@dataclass
class BuildResult:
    """Outcome of ``build``."""

    ok: bool
    binary: str
    message: str = ""
    seconds: float = 0.0
    stamp: dict[str, Any] | None = None
    log_tail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RunFn = Callable[..., "subprocess.CompletedProcess[str]"]


def build(
    *,
    source: Path | None = None,
    out: Path | None = None,
    lock: Path | None = None,
    archs: Sequence[str] = ARCHS,
    hipcc: str | None = None,
    lock_wait_s: int = LOCK_WAIT_S,
    run: RunFn = subprocess.run,
) -> BuildResult:
    """Compile the probe under the build lock and install it atomically.

    ``run`` replaces ``subprocess.run`` (tests). The compile uses
    ``flock -w <lock_wait_s> <lock> nice -n 19 hipcc ...``; the result carries
    the tail of the compiler output on failure.
    """
    src = source if source is not None else source_path()
    final = out if out is not None else default_binary()
    compiler = hipcc if hipcc is not None else find_hipcc()
    if not src.is_file():
        return BuildResult(False, str(final), f"source {src} not found")
    if compiler is None:
        return BuildResult(False, str(final), "hipcc not found (PATH, /opt/rocm/bin)")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".new")
    lock_file = lock if lock is not None else DEFAULT_LOCK
    cmd = [
        "flock",
        "-w",
        str(lock_wait_s),
        str(lock_file),
        "nice",
        "-n",
        "19",
        compiler,
        "-x",
        "hip",
        "-O3",
        "-std=c++17",
        *[f"--offload-arch={a}" for a in archs],
        "-o",
        str(tmp),
        str(src),
        "-pthread",
    ]
    began = time.monotonic()
    try:
        done = run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return BuildResult(False, str(final), f"cannot run {cmd[0]}: {exc}")
    seconds = time.monotonic() - began
    tail = (done.stdout + done.stderr).strip().splitlines()[-30:]
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        return BuildResult(
            False,
            str(final),
            f"build failed (rc={done.returncode})",
            seconds,
            None,
            tail,
        )
    version = probe_version(tmp)
    if version is None:
        tmp.unlink(missing_ok=True)
        return BuildResult(
            False,
            str(final),
            "the new binary does not answer `version`; not installed",
            seconds,
            None,
            tail,
        )
    stamp: dict[str, Any] = {
        "source": str(src),
        "source_sha256": sha256_of(src),
        "git_commit": git_commit(src.parent),
        "archs": list(archs),
        "hipcc": compiler,
        "version": version.version,
        "modes": list(version.modes),
        "built_ns": time.time_ns(),
        "build_seconds": round(seconds, 1),
    }
    os.chmod(tmp, 0o755)
    os.replace(tmp, final)
    stamp_path(final).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return BuildResult(
        True, str(final), f"built {version.version}", seconds, stamp, tail
    )


__all__ = [
    "ARCHS",
    "BinaryStatus",
    "BuildResult",
    "binary_status",
    "build",
    "read_stamp",
    "source_path",
]
