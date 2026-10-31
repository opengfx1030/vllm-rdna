# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for tools/rdna/port_v031/serve_validate.sh device pin.

The driver refuses HIP GPUs 0/1 — but that pin is host policy for the
maintainer box (par1-cs25: HIP 0/1 are W7800s owned by another job), not a
vLLM constraint. A four-card host whose V620s enumerate as HIP 0-3 must be
able to clear the list without editing the driver. The real script is
executed in a stub tree whose serve target exits immediately: the refusal
path exits 2 (before any boot), the accepted path proceeds to the bounded
BOOT FAILED exit 1. No GPU is touched in either case.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DRIVER = REPO / "tools/rdna/port_v031/serve_validate.sh"


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "tree"
    (root / "tools/rdna/port_v031").mkdir(parents=True)
    (root / "tools/rdna/serve_rdna.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "STUB-SERVE $*"\n'
    )
    (root / "tools/rdna/port_v031/serve_validate.sh").write_text(DRIVER.read_text())
    home = tmp_path / "home"
    home.mkdir()
    return root, home


def run(tree, home, **env):
    full = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "RECIPE": "unit-test",
        "MODEL": "/nonexistent-model",
        "CELLS": "none",
        "READY_TIMEOUT": "2",
        "PORT": "18999",
        **env,
    }
    return subprocess.run(
        ["bash", str(tree / "tools/rdna/port_v031/serve_validate.sh")],
        capture_output=True,
        text=True,
        env=full,
        timeout=120,
    )


def test_default_still_refuses_hip_0_and_1(tree):
    root, home = tree
    for gpus in ("0,1", "0,1,2,3", "1,2,3,4"):
        result = run(root, home, GPUS=gpus)
        assert result.returncode == 2, gpus
        assert "refusing HIP GPU" in result.stdout, gpus


def test_refusal_list_can_be_cleared_for_four_card_hosts(tree):
    root, home = tree
    result = run(root, home, GPUS="0,1,2,3", REFUSED_GPUS="")
    assert result.returncode == 1  # reached the bounded boot-fail path
    assert "refusing HIP GPU" not in result.stdout
    assert "GPUS=0,1,2,3" in result.stdout  # driver booted with the full set


def test_maintainer_pair_path_unaffected(tree):
    root, home = tree
    result = run(root, home, GPUS="6,7")
    assert result.returncode == 1
    assert "refusing HIP GPU" not in result.stdout


def test_refused_list_repointable(tree):
    root, home = tree
    result = run(root, home, GPUS="2,3", REFUSED_GPUS="2,3")
    assert result.returncode == 2
    assert "refusing HIP GPU" in result.stdout
