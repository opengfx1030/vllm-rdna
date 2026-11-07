# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of tools/rdna/rdna3/gpu_g0_g7.sh (print modes only, no GPU).

The runner may only ever touch HIP 0 (the W7800 on bus cb): these tests pin the
visibility of every command it would run, that the test files it names exist,
that dry runs leave no lease or quiet file behind, and that a held lease makes
a real run refuse before it starts anything.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "rdna" / "rdna3" / "gpu_g0_g7.sh"
BASH = shutil.which("bash")


def _bash_major() -> int:
    if BASH is None:
        return 0
    out = subprocess.run(
        [BASH, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True
    ).stdout.strip()
    return int(out or 0)


pytestmark = pytest.mark.skipif(_bash_major() < 4, reason="needs bash >= 4")


def _run(args, home, check_rc=0):
    assert BASH is not None
    env = {"PATH": os.environ["PATH"], "HOME": str(home)}
    proc = subprocess.run(
        [BASH, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    if check_rc is not None:
        assert proc.returncode == check_rc, proc.stdout + proc.stderr
    return proc


def _plan(tmp_path, *extra):
    proc = _run(["--dry-run", "--tree", str(REPO), *extra], tmp_path)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_list_prints_the_steps_in_order(tmp_path):
    proc = _run(["--list"], tmp_path)
    assert proc.stdout.split() == [f"G{i}" for i in range(9)] + ["G16"]
    assert _run(["--list", "--only", "G4"], tmp_path).stdout.split() == ["G4"]
    sub = _run(["--list", "--from", "G2", "--to", "G3"], tmp_path)
    assert sub.stdout.split() == ["G2", "G3"]
    tail = _run(["--list", "--from", "G8"], tmp_path)
    assert tail.stdout.split() == ["G8", "G16"]


def test_unknown_step_and_argument_are_rejected(tmp_path):
    assert _run(["--list", "--only", "G9"], tmp_path, check_rc=2).returncode == 2
    assert _run(["--bogus"], tmp_path, check_rc=2).returncode == 2


def test_every_command_names_only_hip_0_or_no_gpu(tmp_path):
    lines = _plan(tmp_path)
    visibility = [re.findall(r"HIP_VISIBLE_DEVICES=(\S*)", line) for line in lines]
    seen = {v for found in visibility for v in found}
    assert seen == {"0", ""}, seen  # GPU commands: 0; CPU-only commands: empty
    gpu_lines = [
        line for line, found in zip(lines, visibility) if found and found[0] == "0"
    ]
    assert len(gpu_lines) > 20
    text = "\n".join(lines)
    for forbidden in ("HIP_VISIBLE_DEVICES=5", "0,5", "HIP_VISIBLE_DEVICES=1"):
        assert forbidden not in text
    assert "07" not in re.findall(r"--bus (\S+)", text)
    assert set(re.findall(r"--bus (\S+)", text)) == {"cb"}


def test_gpu_commands_never_inherit_family_opt_in_or_other_visibility(tmp_path):
    for line in _plan(tmp_path):
        if "HIP_VISIBLE_DEVICES=0" in line:
            assert "-u VLLM_RDNA3_HIP_OPS" in line
            assert "-u ROCR_VISIBLE_DEVICES" in line
            assert "-u CUDA_VISIBLE_DEVICES" in line


def test_pytest_groups_opt_in_every_family_and_g7_cells_do_not(tmp_path):
    for line in _plan(tmp_path):
        label = line.split()[0] if line.strip() else ""
        if "-m pytest" in line:
            assert "VLLM_RDNA3_HIP_OPS=all" in line, label
        elif label.startswith("G7-") and "VLLM_RDNA3_HIP_OPS=all" in line:
            raise AssertionError(label)


def test_labels_select_individual_commands(tmp_path):
    plan = _plan(tmp_path, "--only", "G5", "--labels", "G5-2")
    labels = [line.split()[0] for line in plan if line.startswith("G5-")]
    assert labels == ["G5-2"]
    none = _plan(tmp_path, "--only", "G5", "--labels", "no-such-label")
    assert [line for line in none if line.startswith("G5-")] == []


def test_g7_cells_pair_default_and_family_routes(tmp_path):
    g7 = _plan(tmp_path, "--only", "G7")
    labels = {line.split()[0] for line in g7}
    for family in ("hc", "gdn-tp2", "gdn-tp1"):
        assert f"G7-{family}-default" in labels and f"G7-{family}-hip" in labels
    for tp in ("tp1", "tp2"):  # the se op ignores the family: fused op vs torch
        assert f"G7-se-{tp}-torch" in labels and f"G7-se-{tp}-op" in labels
    hip = [line for line in g7 if "VLLM_RDNA3_HIP_OPS=" in line]
    assert hip and all(re.search(r"VLLM_RDNA3_HIP_OPS=(hc|gdn)\b", x) for x in hip)
    text = "\n".join(g7)
    assert "--i-se 640" in text and "--i-se 320" in text  # TP1 and TP2 shared expert
    assert "--n 248320" in text and "--n 124160" in text  # TP1 and TP2 lm_head
    assert "flashnext-tp1" in text and "flashnext-tp2" in text
    # The crossover bench times eager launches unless --cold: device time needs it.
    assert [x for x in g7 if "gemv_crossover_bench" in x and "--cold" not in x] == []
    assert "gemv-tp1" in text and "gemv-tp2" in text


def test_named_test_files_exist(tmp_path):
    lines = _plan(tmp_path)
    paths = set(re.findall(r"\b((?:tests|tools)/[\w/.-]+\.py)\b", "\n".join(lines)))
    assert len(paths) > 25
    optional = {"tests/kernels/quantization/test_rdna3_moe_wmma.py"}  # S1b, G4 only
    missing = sorted(p for p in paths - optional if not (REPO / p).exists())
    assert not missing, missing


def test_dry_run_takes_no_lease_and_writes_nothing(tmp_path):
    _plan(tmp_path)
    assert not (tmp_path / "w4a8_runs").exists()


def test_a_held_lease_refuses_before_any_command(tmp_path):
    lease = tmp_path / "w4a8_runs" / "lanes" / "lease"
    lease.mkdir(parents=True)
    (lease / "HxServer.w7800cb").write_text("held\n")
    proc = _run(["--only", "G1", "--tree", str(REPO)], tmp_path, check_rc=3)
    assert "lease held by" in proc.stderr
    assert not list((tmp_path / "w4a8_runs").glob("rdna3_port/gpu/*/*.log"))


def test_mixed_arch_sensitive_steps_are_cpu_or_g0(tmp_path):
    for line in _plan(tmp_path, "--only", "G1"):
        assert "HIP_VISIBLE_DEVICES=0" not in line  # the build check is CPU only
