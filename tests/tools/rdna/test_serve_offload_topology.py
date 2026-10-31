# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for tools/rdna/mono/serve_offload.sh topology selection.

The wrapper is executed for real in a throwaway tree whose serve targets are
stubs that report their resolved configuration, so these tests observe the
resolved command and environment, not source text.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/rdna/mono/serve_offload.sh"


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "tree"
    (root / "tools/rdna/mono").mkdir(parents=True)
    (root / "tools/rdna/port_v031").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(root / ".venv/bin/python", 0o755)
    (root / "tools/rdna/mono/serve_offload.sh").write_text(SCRIPT.read_text())
    # Stub downstream launch targets: print resolved identity plus what the
    # wrapper exported, then exit without touching a GPU.
    stub = (
        "#!/usr/bin/env bash\n"
        # The real serve_validate.sh hard-requires an exported GPUS; mirror
        # that contract so delegation regressions are visible here.
        ': "${GPUS:?GPUS required}"\n'
        'echo "STUB $(basename "$0") $*"\n'
        'echo "STUB-ENV TP=${TP:-unset} GPUS=${GPUS} '
        'HIP=${HIP_VISIBLE_DEVICES:-unset} '
        'EXTRA=${EXTRA_ARGS:-unset} RECIPE=${RECIPE:-unset} PORT=${PORT:-unset}"\n'
    )
    (root / "tools/rdna/serve_rdna.sh").write_text(stub)
    (root / "tools/rdna/port_v031/serve_validate.sh").write_text(stub)
    return root


def run(tree, *, print_=0, validate=0, **env):
    full = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tree),
        "PRINT": str(print_),
        "VALIDATE": str(validate),
        **env,
    }
    return subprocess.run(
        ["bash", str(tree / "tools/rdna/mono/serve_offload.sh")],
        capture_output=True,
        text=True,
        env=full,
        timeout=30,
    )


def plan(result):
    lines = result.stdout.splitlines()
    head = next(line for line in lines if line.startswith("OFFLOAD-PLAN "))
    fields = dict(pair.split("=", 1) for pair in head.removeprefix("OFFLOAD-PLAN ").split())
    fields["extra"] = next(
        line.removeprefix("OFFLOAD-EXTRA ")
        for line in lines
        if line.startswith("OFFLOAD-EXTRA ")
    )
    return fields


def test_default_resolves_tp2_on_6_7(tree):
    result = run(tree, print_=1)
    assert result.returncode == 0, result.stderr
    fields = plan(result)
    assert (fields["tp"], fields["devices"]) == ("2", "6,7")
    assert fields["local_experts"] == "256"
    assert fields["slots"] == "96"  # historical baseline is byte-for-byte preserved
    assert '"expert_cache_slots":96' in fields["extra"]
    assert "--kv-offloading-size 16" in fields["extra"]  # KV tier, not expert cache


def test_print_does_not_launch(tree):
    result = run(tree, print_=1, TP="4", GPUS="0,1,2,3", ALLOWED_GPU_SETS="0,1,2,3")
    assert result.returncode == 0
    assert "STUB" not in result.stdout


def test_tp4_offload_plan(tree):
    result = run(
        tree,
        print_=1,
        TP="4",
        GPUS="0,1,2,3",
        ALLOWED_GPU_SETS="6,7 0,1,2,3",
    )
    assert result.returncode == 0, result.stderr
    fields = plan(result)
    assert fields == {
        "tp": "4",
        "devices": "0,1,2,3",
        "local_experts": "128",
        "slots": "48",  # one third of 128 mirrors the 96/256 TP2 baseline
        "validate": "0",
        "extra": fields["extra"],
    }
    assert '"expert_cache_slots":48' in fields["extra"]


def test_unauthorized_device_set_refused_even_with_matching_tp(tree):
    result = run(tree, print_=1, TP="4", GPUS="0,1,2,3")
    assert result.returncode == 2
    assert "ALLOWED_GPU_SETS" in result.stderr


def test_device_count_must_equal_tp(tree):
    result = run(tree, print_=1, TP="4", GPUS="6,7")
    assert result.returncode == 2
    assert "rank count must equal device count" in result.stderr


def test_malformed_and_duplicate_devices_refused(tree):
    # Empty GPUS intentionally falls back to the default pair (bash :-), as
    # in the pre-TP4 wrapper; malformed lists must refuse.
    for gpus in ("6,,7", "6, 7", "6,6", "a,7", "6,7,7"):
        result = run(tree, print_=1, GPUS=gpus)
        assert result.returncode == 2, gpus


def test_slots_capped_by_local_experts_per_rank(tree):
    ok = run(
        tree,
        print_=1,
        TP="4",
        GPUS="1,2,3,4",
        ALLOWED_GPU_SETS="1,2,3,4",
        SLOTS="128",
    )
    assert ok.returncode == 0, ok.stderr
    assert plan(ok)["slots"] == "128"
    bad = run(
        tree,
        print_=1,
        TP="4",
        GPUS="1,2,3,4",
        ALLOWED_GPU_SETS="1,2,3,4",
        SLOTS="200",
    )
    assert bad.returncode == 2
    assert "must be in [1, 128]" in bad.stderr


def test_only_qualified_expert_geometries(tree):
    result = run(tree, print_=1, TP="3", GPUS="6,7,5", ALLOWED_GPU_SETS="5,6,7")
    assert result.returncode == 2  # TP3 is not a power of two
    result = run(tree, print_=1, TP="1", GPUS="6", ALLOWED_GPU_SETS="6")
    assert result.returncode == 2  # 512 local experts is not K2-qualified


def test_kv_offload_disabled_switch_absent(tree):
    result = run(tree, print_=1, KV_OFFLOAD_GB="0")
    assert result.returncode == 0
    assert "--kv-offloading" not in plan(result)["extra"]


def test_serve_target_receives_resolved_tp(tree):
    result = run(
        tree,
        TP="4",
        GPUS="0,1,2,3",
        ALLOWED_GPU_SETS="0,1,2,3",
    )
    assert result.returncode == 0, result.stderr
    assert "STUB serve_rdna.sh" in result.stdout
    assert "TP=4" in result.stdout
    assert "HIP=0,1,2,3" in result.stdout


def test_validate_target_receives_resolved_tp(tree):
    result = run(
        tree,
        validate=1,
        CELLS="none",
        TP="4",
        GPUS="0,1,2,3",
        ALLOWED_GPU_SETS="0,1,2,3",
    )
    assert result.returncode == 0, result.stderr
    assert "STUB serve_validate.sh" in result.stdout
    assert "TP=4" in result.stdout


def test_validate_delegation_exports_default_gpus(tree):
    # GPUS is deliberately NOT in the caller env: the wrapper default (6,7)
    # must still reach the validation driver, which hard-requires GPUS.
    result = run(tree, validate=1, CELLS="none")
    assert result.returncode == 0, result.stderr
    assert "STUB serve_validate.sh" in result.stdout
    assert "GPUS=6,7" in result.stdout
    assert "TP=2" in result.stdout
