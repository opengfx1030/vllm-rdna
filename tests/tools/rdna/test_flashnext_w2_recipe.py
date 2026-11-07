# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashnext-w2-mtp2 (TP2/EP2 on two gfx1100 W7800), serve_rdna.sh print mode.

Pins the safe-baseline contract of the recipe: no gfx1030-only knob leaks onto
gfx1100 (TunableOp rows, tuned-MoE configs, W4A8 sdot4, RDNA2 FA backend,
rdna_ar) and the gfx1100 opt-in list is empty unless the caller sets it. No GPU
is touched: the stub python cannot import torch, so PCI_BUSES stays unresolved.
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
    "VLLM_RDNA3_HIP_OPS",
    "VLLM_CACHE_ROOT",
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
        ["bash", str(LAUNCHER), "RECIPE=flashnext-w2-mtp2", *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )


def test_recipe_plan_is_two_way_tp_with_expert_parallel(venv):
    result = plan(venv)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PCI_BUSES='cb 07' (not resolvable on this host" in out
    assert "'--tensor-parallel-size' '2'" in out
    assert "'--enable-expert-parallel'" in out
    assert "'--dtype' 'float16'" in out
    assert '"num_speculative_tokens":2' in out
    assert "--pipeline-parallel-size" not in out
    assert '"cudagraph_capture_sizes":[3,6,12,24]' in out


def test_recipe_does_not_pin_the_gfx1030_attention_backend(venv):
    out = plan(venv).stdout
    assert "--attention-backend" not in out
    assert "VLLM_USE_RDNA2_FA" not in out


def test_recipe_keeps_gfx1030_artefacts_off(venv):
    result = plan(venv)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "disabled (TUNABLEOP=0)" in out
    assert "VLLM_TUNED_CONFIG_FOLDER" not in out
    assert "export VLLM_RDNA2_W4A8_SDOT4='0'" in out
    assert "export VLLM_RDNA_AR='0'" in out
    assert "export VLLM_RDNA_SMALLM_GEMM_MAX_BYTES='0'" in out
    assert "export VLLM_RDNA_DENSE_INT8='0'" in out


def test_hip_family_list_is_empty_by_default_and_overridable(venv):
    out = plan(venv).stdout
    assert "export VLLM_RDNA3_HIP_OPS=''" in out
    assert "export VLLM_RDNA3_MOE_DETERMINISTIC='0'" in out
    out = plan(venv, "VLLM_RDNA3_HIP_OPS=hc,gdn").stdout
    assert "export VLLM_RDNA3_HIP_OPS='hc,gdn'" in out


def test_cache_root_is_separate_from_the_gfx1030_tree(venv):
    out = plan(venv, HOME="/home/someone").stdout
    assert "export VLLM_CACHE_ROOT='/home/someone/.cache/vllm-flashnext-w7800'" in out
