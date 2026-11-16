# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import spawn_new_process_for_each_test

pytest_plugins = ["pytester"]


@spawn_new_process_for_each_test
def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    assert 1 + 1 == 2


@pytest.mark.xfail(raises=RuntimeError, strict=True)
@spawn_new_process_for_each_test
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def test_spawn_decorator_skip():
    """pytest.skip inside subprocess should propagate correctly."""
    pytest.skip("intentional skip")


def test_spawn_decorator_reports_skip(pytester):
    """A child skip must remain a skip in the parent pytest report."""
    result = pytester.runpytest_subprocess(
        f"{Path(__file__).resolve()}::test_spawn_decorator_skip", "-rs"
    )
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["SKIPPED *intentional skip"])


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected


def test_spawn_decorator_imports_collected_module(pytester, monkeypatch):
    """A fresh child must import modules from pytest's collection paths."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    directory = pytester.path / "collected"
    directory.mkdir()
    output = pytester.path / "child-result.json"
    module = directory / "test_spawn_path_probe.py"
    module.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from tests.utils import spawn_new_process_for_each_test\n"
        "@spawn_new_process_for_each_test\n"
        "def test_child_records_identity():\n"
        f"    Path({str(output)!r}).write_text(json.dumps("
        "{'pid': os.getpid(), 'module': __name__}))\n"
    )
    result = pytester.runpytest_subprocess(str(module), "-v")
    result.assert_outcomes(passed=1)
    observed = json.loads(output.read_text())
    assert observed["pid"] != os.getpid()
    assert observed["module"] == module.stem


def test_spawn_decorator_preserves_native_fatal_trace(pytester, monkeypatch):
    """An explicitly enabled fatal trace must survive test isolation."""
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(Path(__file__).resolve().parents[2]), *sys.path]),
    )
    module = pytester.makepyfile(
        test_native_abort_probe="""
        import ctypes
        import resource
        import threading
        from tests.utils import spawn_new_process_for_each_test

        def native_terminate():
            ctypes.CDLL("libstdc++.so.6")._ZSt9terminatev()

        @spawn_new_process_for_each_test
        def test_native_abort():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            worker = threading.Thread(target=native_terminate)
            worker.start()
            worker.join()
        """
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", "pytest", str(module), "-q"],
        cwd=pytester.path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "killed by SIGABRT" in output, output
    assert "Fatal Python error: Aborted" in output, output
    assert "in native_terminate" in output, output
