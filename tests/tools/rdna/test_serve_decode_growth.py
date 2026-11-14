# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DECODE_GROWTH_ADMISSION launcher key (print mode, stub venv).

``serve_rdna.sh PRINT=1`` needs only a VENV directory, so no GPU is touched.
The key is off for every shipped recipe until the owner enables it per recipe.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
RECIPES = REPO / "tools/rdna/recipes"
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "DECODE_GROWTH_ADMISSION",
)


@pytest.fixture()
def venv(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    return tmp_path / "venv"


def plan(venv, *args, **env):
    full = {k: v for k, v in os.environ.items() if k not in DROPPED}
    full.update(VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", **env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )


@pytest.mark.parametrize("recipe", ["flashnext-mtp2", "flashnext-leapquant-mtp2"])
def test_flag_absent_by_default(venv, recipe):
    result = plan(venv, f"RECIPE={recipe}")
    assert result.returncode == 0, result.stderr
    assert "--decode-growth-admission" not in result.stdout


@pytest.mark.parametrize("recipe", ["flashnext-mtp2", "flashnext-leapquant-mtp2"])
def test_key_appends_flag(venv, recipe):
    result = plan(venv, f"RECIPE={recipe}", "DECODE_GROWTH_ADMISSION=1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("'--decode-growth-admission'") == 1


def test_key_zero_omits_flag(venv):
    result = plan(venv, "RECIPE=flashnext-mtp2", DECODE_GROWTH_ADMISSION="0")
    assert result.returncode == 0, result.stderr
    assert "--decode-growth-admission" not in result.stdout


def test_no_shipped_recipe_enables_it():
    offenders = [
        p.name
        for p in RECIPES.glob("*.env")
        if "DECODE_GROWTH_ADMISSION" in p.read_text()
    ]
    assert offenders == []
