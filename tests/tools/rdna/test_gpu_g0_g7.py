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


def test_wmma_tests_run_in_three_stages_smallest_first(tmp_path):
    stages = [x for x in _plan(tmp_path, "--only", "G4") if x.startswith("G4-wmma-")]
    assert [x.split()[0] for x in stages] == ["G4-wmma-1", "G4-wmma-2", "G4-wmma-3"]
    assert "k128-n16\\ or\\ valid_call" in stages[0]  # printf %q escapes the spaces
    assert "e8-gate_up-m16\\ or\\ e8-down-m16" in stages[1]
    assert not re.search(r"\s-k\s", stages[2])  # the whole file
    assert all("test_rdna3_moe_wmma.py" in x for x in stages)


def test_g13_is_opt_in_and_starts_three_engines_on_hip_0(tmp_path):
    assert "G13" not in _run(["--list"], tmp_path).stdout.split()
    assert _run(["--list", "--only", "G13"], tmp_path).stdout.split() == ["G13"]
    assert _run(["--list", "--to", "G13"], tmp_path).stdout.split()[-2:] == [
        "G16",
        "G13",
    ]
    plan = [x for x in _plan(tmp_path, "--only", "G13") if x.startswith("G13-")]
    assert [x.split()[0] for x in plan] == [
        "G13-base",
        "G13-all",
        "G13-wmma",
        "G13-compare-all",
        "G13-compare-wmma",
    ]
    base, every, wmma = plan[:3]
    for line in (base, every, wmma):
        assert "HIP_VISIBLE_DEVICES=0" in line and "-u VLLM_RDNA3_HIP_OPS" in line
        assert "trunc_e2e.py run" in line and "--max-model-len 8192" in line
        # the kernel-selecting variables are the recipe's, the cache root is the
        # runner's (one VLLM_CACHE_ROOT, not the recipe's)
        assert "VLLM_USE_V2_MODEL_RUNNER=1" in line and "GPU_MAX_HW_QUEUES=4" in line
        assert line.count("VLLM_CACHE_ROOT=") == 1
    assert "VLLM_RDNA3_HIP_OPS=" not in base and "VLLM_RDNA3_MOE_WMMA" not in base
    assert "VLLM_RDNA3_HIP_OPS=all" in every and "VLLM_RDNA3_MOE_WMMA" not in every
    assert "VLLM_RDNA3_HIP_OPS=all" in wmma and "VLLM_RDNA3_MOE_WMMA=1" in wmma
    for line in plan[3:]:
        assert "HIP_VISIBLE_DEVICES= " in line and "trunc_e2e.py compare" in line


def test_g7_cells_pair_default_and_family_routes(tmp_path):
    g7 = _plan(tmp_path, "--only", "G7")
    labels = {line.split()[0] for line in g7}
    for family in ("hc", "gdn-tp2", "gdn-tp1", "qsa-tp2", "qsa-tp1"):
        assert f"G7-{family}-default" in labels and f"G7-{family}-hip" in labels
    for tp in ("tp1", "tp2"):  # the se op ignores the family: fused op vs torch
        assert f"G7-se-{tp}-torch" in labels and f"G7-se-{tp}-op" in labels
    hip = [line for line in g7 if "VLLM_RDNA3_HIP_OPS=" in line]
    assert hip and all(re.search(r"VLLM_RDNA3_HIP_OPS=(hc|gdn|qsa)\b", x) for x in hip)
    text = "\n".join(g7)
    assert "--i-se 640" in text and "--i-se 320" in text  # TP1 and TP2 shared expert
    assert "--n 248320" in text and "--n 124160" in text  # TP1 and TP2 lm_head
    assert "flashnext-tp1" in text and "flashnext-tp2" in text
    # The crossover bench times eager launches unless --cold: device time needs it.
    assert [x for x in g7 if "gemv_crossover_bench" in x and "--cold" not in x] == []
    assert "gemv-tp1" in text and "gemv-tp2" in text


def _g7_cells(tmp_path, *labels):
    """{label: words} of the G7 cells from one plan (one run directory);
    printf %q escapes the commas, undone here."""
    wanted = [f"G7-{label}" for label in labels]
    plan = _plan(tmp_path, "--only", "G7", "--labels", ",".join(wanted))
    cells = {line.split()[0]: line.replace("\\,", ",").split() for line in plan}
    assert set(wanted) <= set(cells), sorted(cells)
    return {label: cells[f"G7-{label}"] for label in labels}


def test_g7_decode_cells_answer_the_hub_questions(tmp_path):
    cells = _g7_cells(
        tmp_path,
        "se-tp1-op-48",
        "se-tp2-op-48",
        "qsa-tp1-hip",
        "qsa-tp1-hip-tight",
        "qsa-tp1-hip-insertion",
        "moe-decode",
        "moe-decode-hub",
        "moe-decode-h1",
        "moe-decode-k-h1",
        "moe-decode-k-h2",
        "moe-decode-block-m",
        "moe-prefill",
        "moe-prefill-hub",
        "moe-prefill-block-m",
    )
    # The shared-expert op at 48 rows needs its row cap raised to 48.
    for tp, i_se in (("tp1", "640"), ("tp2", "320")):
        words = cells[f"se-{tp}-op-48"]
        assert "VLLM_RDNA_SE_DECODE_MAX=48" in words
        assert words[words.index("--i-se") + 1] == i_se
        assert words[words.index("--m") + 1] == "24,32,48"
    # QSA decode core, MoE router + experts: new benches, one process per route.
    qsa = cells["qsa-tp1-hip"]
    assert "tools/rdna/rdna3/qsa_decode_bench.py" in qsa
    # the TP4 anchors' 1k context, in a page table as wide as the recipe's max length
    assert "1024" in qsa[qsa.index("--ctx") + 1].split(",")
    assert qsa[qsa.index("--capacity") + 1] == "262144"
    tight = cells["qsa-tp1-hip-tight"]  # page table as wide as the live context
    assert "--capacity" not in tight and "VLLM_RDNA3_HIP_OPS=qsa" in tight
    insertion = cells["qsa-tp1-hip-insertion"]  # the select with the other top-k
    assert "VLLM_TOPK_DECODE_ALGO=insertion" in insertion
    assert insertion[insertion.index("--ops") + 1] == "select"
    for label in ("moe-decode", "moe-decode-hub", "moe-decode-block-m"):
        assert "tools/rdna/rdna3/moe_decode_bench.py" in cells[label]
    hub = cells["moe-decode-hub"]
    assert hub[hub.index("--global-experts") + 1] == "512"
    # H1's hub: 130 resident slots; 264 routed experts give the resident-touched counts
    # the T10 routing statistics measured there (14 / 79 / 110 at 3 / 24 / 48 rows).
    h1 = cells["moe-decode-h1"]
    assert h1[h1.index("--slots") + 1] == "130"
    assert h1[h1.index("--global-experts") + 1] == "264"
    # exactly K distinct experts live per width: H1 and H2 hub tables of the design
    for label, expected in (
        ("moe-decode-k-h1", "10,21,37,63,98"),
        ("moe-decode-k-h2", "7,13,22,37,58"),
    ):
        words = cells[label]
        assert words[words.index("--distinct") + 1] == expected
        assert words[words.index("--m") + 1] == "3,6,12,24,48"
    # Prefill sizes: the scalar kernel against the WMMA kernel. The bench sets the
    # WMMA variables per mode, so the cell carries none of them.
    for label in ("moe-prefill", "moe-prefill-hub", "moe-prefill-block-m"):
        words = cells[label]
        assert "tools/rdna/rdna3/moe_decode_bench.py" in words
        assert "wmma" in words[words.index("--modes") + 1].split(",")
        assert min(int(v) for v in words[words.index("--m") + 1].split(",")) >= 256
        assert not [w for w in words if w.startswith("VLLM_RDNA3_MOE_WMMA")]


def test_lm_head_tunable_op_cells_tune_then_read_back(tmp_path):
    labels = [
        f"lm_head-{tp}-{kind}" for tp in ("tp1", "tp2") for kind in ("tune", "tuned")
    ]
    cells = _g7_cells(tmp_path, *labels)
    for tp, vocab in (("tp1", "248320"), ("tp2", "124160")):
        tune, tuned = cells[f"lm_head-{tp}-tune"], cells[f"lm_head-{tp}-tuned"]
        # several variables in one cell reach `env` as separate words
        for words, tuning in ((tune, "1"), (tuned, "0")):
            assert "PYTORCH_TUNABLEOP_ENABLED=1" in words
            assert f"PYTORCH_TUNABLEOP_TUNING={tuning}" in words
            assert words[words.index("--n") + 1] == vocab
            assert "VLLM_RDNA3_HIP_OPS=all" not in words
        csv = [w for w in tune if w.startswith("PYTORCH_TUNABLEOP_FILENAME=")]
        assert len(csv) == 1 and csv[0] in tuned and csv[0].endswith(f"{tp}.csv")
    # the untuned baselines and every other cell stay out of TunableOp
    for line in _plan(tmp_path, "--only", "G7"):
        if "TUNABLEOP" in line:
            assert line.split()[0].startswith("G7-lm_head-tp")
            assert line.split()[0].endswith(("-tune", "-tuned"))


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
