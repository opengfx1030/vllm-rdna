# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""REPLAYSSM_STATE_DTYPE launcher key and the retired VLLM_RDNA_GDN_LEAPQUANT.

``serve_rdna.sh PRINT=1`` needs only a VENV directory. The stub python below cannot
import torch, so no GPU is touched and no server starts.

LeapQuant (arXiv 2609.38166) is selected with ``--replayssm-state-dtype leapquant``.
The key name follows vllm-project/vllm#52962 (askliar); the refusal list behind the
flag follows the RFC vllm-project/vllm#64725 (lihaoyang-amd). No code was taken from
either.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "VLLM_RDNA_GDN_LEAPQUANT",
)
FLAG = "--replayssm-state-dtype"


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


def test_leapquant_recipe_passes_the_flag_after_the_block_size(venv):
    result = plan(venv, "RECIPE=flashnext-leapquant-mtp2")
    assert result.returncode == 0, result.stderr
    assert f"'--block-size' '512' '{FLAG}' 'leapquant'" in result.stdout
    assert "VLLM_RDNA_GDN_LEAPQUANT" not in result.stdout


def test_recipe_without_the_key_passes_no_flag(venv):
    result = plan(venv, "RECIPE=flashnext-mtp2")
    assert result.returncode == 0, result.stderr
    assert FLAG not in result.stdout


def test_key_selects_the_flag_on_any_recipe(venv):
    result = plan(venv, "RECIPE=flashnext-mtp2", "REPLAYSSM_STATE_DTYPE=leapquant")
    assert result.returncode == 0, result.stderr
    assert f"'--block-size' '1024' '{FLAG}' 'leapquant'" in result.stdout


def test_auto_is_passed_through(venv):
    result = plan(venv, "RECIPE=flashnext-leapquant-mtp2", "REPLAYSSM_STATE_DTYPE=auto")
    assert result.returncode == 0, result.stderr
    assert f"'{FLAG}' 'auto'" in result.stdout
    assert "'leapquant'" not in result.stdout


@pytest.mark.parametrize("how", ["environment", "launcher argument"])
def test_the_retired_switch_stops_the_launcher(venv, how):
    # The engine ignores the variable, so a launch would serve FP32 state silently.
    args, env = ["RECIPE=flashnext-leapquant-mtp2"], {}
    if how == "environment":
        env["VLLM_RDNA_GDN_LEAPQUANT"] = "1"
    else:
        args.append("VLLM_RDNA_GDN_LEAPQUANT=1")
    result = plan(venv, *args, **env)
    assert result.returncode != 0
    assert "REPLAYSSM_STATE_DTYPE=leapquant" in result.stdout + result.stderr


def test_the_retired_switch_set_to_zero_is_harmless(venv):
    result = plan(venv, "RECIPE=flashnext-mtp2", VLLM_RDNA_GDN_LEAPQUANT="0")
    assert result.returncode == 0, result.stderr
    assert FLAG not in result.stdout
    assert "VLLM_RDNA_GDN_LEAPQUANT" not in result.stdout
