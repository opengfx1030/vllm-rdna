# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and status of the probe binary without a compiler or a GPU.

``run`` (the compiler call) is replaced by a function that writes the output
file the way ``hipcc`` would: here an executable that runs the probe double of
``fixtures/probe/fake_p2p_probe.py`` (SYNTHETIC), so ``version`` answers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pciescope.probe import binary as bn  # noqa: E402
from pciescope.probe.__main__ import main as probe_main  # noqa: E402
from pciescope_probe_harness import write_launcher  # noqa: E402

SOURCE_TEXT = "// synthetic p2p_probe source\nint main() { return 0; }\n"


@pytest.fixture(autouse=True)
def _spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FAKE_PROBE_SPEC", str(spec))
    return spec


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "p2p_probe.cpp"
    path.parent.mkdir()
    path.write_text(SOURCE_TEXT, encoding="utf-8")
    return path


class FakeCompiler:
    """Stands in for ``subprocess.run`` of the flock/nice/hipcc command."""

    def __init__(self, rc: int = 0, stderr: str = "", works: bool = True) -> None:
        self.rc = rc
        self.stderr = stderr
        self.works = works
        self.calls: list[list[str]] = []

    def __call__(
        self, cmd: Sequence[str], **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        assert kw.get("capture_output") and kw.get("text")
        if self.rc == 0:
            out = Path(cmd[cmd.index("-o") + 1])
            if self.works:
                write_launcher(out)
            else:  # a binary that cannot answer `version`
                out.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
                out.chmod(0o755)
        return subprocess.CompletedProcess(list(cmd), self.rc, "", self.stderr)


# -- helpers -----------------------------------------------------------------------


def test_sha256_of(tmp_path: Path) -> None:
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert bn.sha256_of(f) == hashlib.sha256(b"abc").hexdigest()
    assert bn.sha256_of(tmp_path / "absent") is None


def test_stamp_path_sits_next_to_the_binary(tmp_path: Path) -> None:
    assert bn.stamp_path(tmp_path / "bin" / "p2p_probe") == (
        tmp_path / "bin" / "p2p_probe.stamp.json"
    )


def test_read_stamp(tmp_path: Path) -> None:
    binary = tmp_path / "p2p_probe"
    assert bn.read_stamp(binary) is None
    bn.stamp_path(binary).write_text("{not json", encoding="utf-8")
    assert bn.read_stamp(binary) is None
    bn.stamp_path(binary).write_text("[1, 2]", encoding="utf-8")
    assert bn.read_stamp(binary) is None
    bn.stamp_path(binary).write_text('{"version": "x"}', encoding="utf-8")
    assert bn.read_stamp(binary) == {"version": "x"}


def test_source_path_is_the_checkout_file() -> None:
    src = bn.source_path()
    assert src.parts[-3:] == ("rdna", "p2p_probe", "p2p_probe.cpp")
    assert src.is_file()


def test_find_hipcc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bn.shutil, "which", lambda name: "/usr/bin/hipcc")
    assert bn.find_hipcc() == "/usr/bin/hipcc"
    monkeypatch.setattr(bn.shutil, "which", lambda name: None)
    monkeypatch.setattr(bn, "HIPCC_FALLBACK", tmp_path / "absent")
    assert bn.find_hipcc() is None
    fallback = tmp_path / "hipcc"
    fallback.write_text("")
    monkeypatch.setattr(bn, "HIPCC_FALLBACK", fallback)
    assert bn.find_hipcc() == str(fallback)


# -- binary_status ---------------------------------------------------------------


def test_status_of_a_missing_binary_points_to_the_build(
    tmp_path: Path, source: Path
) -> None:
    st = bn.binary_status(tmp_path / "absent", source)
    assert not st.exists and not st.runnable
    assert "python -m pciescope.probe build" in st.problem


def test_status_of_a_file_that_is_not_executable(tmp_path: Path, source: Path) -> None:
    f = tmp_path / "p2p_probe"
    f.write_text("x")
    f.chmod(0o644)
    st = bn.binary_status(f, source)
    assert st.exists and not st.runnable and "not executable" in st.problem


def test_status_of_an_old_binary_without_version_mode(
    tmp_path: Path, source: Path
) -> None:
    f = tmp_path / "p2p_probe"
    f.write_text("#!/bin/sh\nexit 1\n")  # the pre-pciescope probe returned 1
    f.chmod(0o755)
    st = bn.binary_status(f, source)
    assert st.exists and not st.runnable
    assert "does not answer `version`" in st.problem and "rebuild" in st.problem


def test_status_of_a_working_binary_without_stamp(tmp_path: Path, source: Path) -> None:
    f = write_launcher(tmp_path / "bin" / "p2p_probe")
    st = bn.binary_status(f, source)
    assert st.runnable and st.problem == "" and st.version == "fake/1"
    assert "batch" in st.modes and "curve" in st.modes
    assert st.stamp is None and st.stale is None
    assert json.dumps(st.to_dict())


def test_status_compares_the_stamp_with_the_source(
    tmp_path: Path, source: Path
) -> None:
    f = write_launcher(tmp_path / "bin" / "p2p_probe")
    digest = hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()
    bn.stamp_path(f).write_text(json.dumps({"source_sha256": digest}), encoding="utf-8")
    assert bn.binary_status(f, source).stale is False
    source.write_text(SOURCE_TEXT + "// edited\n", encoding="utf-8")
    assert bn.binary_status(f, source).stale is True


# -- build ---------------------------------------------------------------------------


def test_build_runs_the_box_command_and_installs_with_a_stamp(
    tmp_path: Path, source: Path
) -> None:
    run = FakeCompiler()
    out = tmp_path / "Apps" / "pciescope" / "bin" / "p2p_probe"
    lock = tmp_path / "build.lock"
    res = bn.build(
        source=source, out=out, lock=lock, hipcc="/opt/rocm/bin/hipcc", run=run
    )
    assert res.ok and res.message == "built fake/1" and res.binary == str(out)
    (cmd,) = run.calls
    # box rules: flock on the shared build lock, nice 19, both architectures, fat binary
    assert cmd[:6] == ["flock", "-w", str(bn.LOCK_WAIT_S), str(lock), "nice", "-n"]
    assert cmd[6:8] == ["19", "/opt/rocm/bin/hipcc"]
    assert "--offload-arch=gfx1030" in cmd and "--offload-arch=gfx1100" in cmd
    assert cmd[cmd.index("-o") + 1] == str(out) + ".new"
    assert cmd[-2:] == [str(source), "-pthread"]
    assert "-O3" in cmd and "-x" in cmd and "hip" in cmd
    # installed atomically: no leftover, executable, answers version, stamp complete
    assert out.is_file() and not Path(str(out) + ".new").exists()
    assert out.stat().st_mode & 0o111
    stamp = bn.read_stamp(out)
    assert stamp is not None
    assert stamp["source_sha256"] == hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()
    assert stamp["archs"] == ["gfx1030", "gfx1100"]
    assert stamp["version"] == "fake/1" and "batch" in stamp["modes"]
    assert stamp["hipcc"] == "/opt/rocm/bin/hipcc" and stamp["built_ns"] > 0
    assert res.stamp == stamp
    assert bn.binary_status(out, source).stale is False


def test_build_for_one_architecture(tmp_path: Path, source: Path) -> None:
    run = FakeCompiler()
    res = bn.build(
        source=source,
        out=tmp_path / "p2p_probe",
        lock=tmp_path / "l",
        hipcc="hipcc",
        archs=("gfx1030",),
        lock_wait_s=7,
        run=run,
    )
    assert res.ok
    cmd = run.calls[0]
    assert [a for a in cmd if a.startswith("--offload-arch")] == [
        "--offload-arch=gfx1030"
    ]
    assert cmd[2] == "7"


def test_a_failed_compile_keeps_the_working_binary(
    tmp_path: Path, source: Path
) -> None:
    out = write_launcher(tmp_path / "bin" / "p2p_probe")
    before = out.read_text()
    run = FakeCompiler(
        rc=1, stderr="p2p_probe.cpp:10:3: error: boom\n1 error generated."
    )
    res = bn.build(source=source, out=out, lock=tmp_path / "l", hipcc="hipcc", run=run)
    assert not res.ok and "build failed (rc=1)" in res.message
    assert any("error: boom" in line for line in res.log_tail)
    assert out.read_text() == before
    assert not Path(str(out) + ".new").exists()
    assert bn.read_stamp(out) is None


def test_a_new_binary_that_does_not_answer_version_is_not_installed(
    tmp_path: Path, source: Path
) -> None:
    out = write_launcher(tmp_path / "bin" / "p2p_probe")
    before = out.read_text()
    res = bn.build(
        source=source,
        out=out,
        lock=tmp_path / "l",
        hipcc="hipcc",
        run=FakeCompiler(works=False),
    )
    assert not res.ok and "does not answer `version`" in res.message
    assert out.read_text() == before and not Path(str(out) + ".new").exists()


def test_build_without_hipcc_or_source_or_flock(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bn, "find_hipcc", lambda: None)
    res = bn.build(source=source, out=tmp_path / "b", run=FakeCompiler())
    assert not res.ok and "hipcc not found" in res.message
    res = bn.build(source=tmp_path / "absent.cpp", out=tmp_path / "b", hipcc="hipcc")
    assert not res.ok and "not found" in res.message

    def no_flock(cmd: Sequence[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("flock")

    res = bn.build(source=source, out=tmp_path / "b", hipcc="hipcc", run=no_flock)
    assert not res.ok and "cannot run flock" in res.message


def test_git_commit_of_a_non_repository_is_empty(tmp_path: Path) -> None:
    assert bn.git_commit(tmp_path) == ""


def test_git_commit_of_this_checkout_is_a_hash() -> None:
    commit = bn.git_commit(bn.source_path().parent)
    assert commit == "" or (
        len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    )


# -- python -m pciescope.probe ---------------------------------------------------


def test_cli_status_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = write_launcher(tmp_path / "bin" / "p2p_probe")
    rc = probe_main(["status", "--binary", str(binary), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc["runnable"] and doc["version"] == "fake/1"
    rc = probe_main(["status", "--binary", str(tmp_path / "absent"), "--json"])
    assert rc == 1 and json.loads(capsys.readouterr().out)["exists"] is False


def test_cli_status_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = write_launcher(tmp_path / "bin" / "p2p_probe")
    assert probe_main(["status", "--binary", str(binary)]) == 0
    text = capsys.readouterr().out
    assert "runnable: True" in text and "version: fake/1" in text
    assert probe_main(["status", "--binary", str(tmp_path / "absent")]) == 1
    assert "problem:" in capsys.readouterr().out


def test_cli_build_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bn, "find_hipcc", lambda: None)
    out = tmp_path / "p2p_probe"
    rc = probe_main(["build", "--out", str(out), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["ok"] is False and "hipcc not found" in doc["message"]
    rc = probe_main(["build", "--out", str(out)])
    assert rc == 1 and "FAILED: hipcc not found" in capsys.readouterr().out
