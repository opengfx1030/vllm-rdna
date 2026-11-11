# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCI_BUSES launcher key and the TP8 / TP4xPP2 Flash-Next recipes (print mode).

``serve_rdna.sh PRINT=1`` needs only a VENV directory. The stub python below cannot
import torch, so PCI_BUSES stays unresolved ("resolved at launch") and no GPU is
touched.
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


def test_tp8_recipe_plan(venv):
    result = plan(venv, "RECIPE=flashnext-tp8-mtp2")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='ce d3 d8 db 0a 0f 14 17' (not resolvable on this host" in out
    assert "'--tensor-parallel-size' '8'" in out
    assert "--pipeline-parallel-size" not in out
    assert "'--kv-cache-memory-bytes' '13958643712'" in out
    assert "export VLLM_RDNA_AR_ONESHOT_KB='128'" in out


def test_tp4pp2_recipe_plan(venv):
    result = plan(venv, "RECIPE=flashnext-tp4pp2-mtp2")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='ce d3 d8 db 0a 0f 14 17' (not resolvable on this host" in out
    assert "'--tensor-parallel-size' '4'" in out
    assert "'--pipeline-parallel-size' '2'" in out
    assert "'--kv-cache-memory-bytes' '13958643712'" in out


def test_bus_count_must_equal_tp_times_pp(venv):
    result = plan(venv, "RECIPE=flashnext-tp8-mtp2", "PCI_BUSES=ce d3")
    assert result.returncode != 0
    assert "PCI_BUSES lists 2 GPUs but TP*PP is 8" in result.stdout + result.stderr


def test_explicit_hip_list_wins_over_buses(venv):
    result = plan(
        venv, "RECIPE=flashnext-tp8-mtp2", HIP_VISIBLE_DEVICES="1,2,3,4,6,7,8,9"
    )
    assert result.returncode == 0, result.stderr
    assert "an explicit device list wins" in result.stdout
