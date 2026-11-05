# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the mono + W4A8 GPU plan (tools/rdna/mono/w4a8/ and
tools/rdna/mono/kfd_evict_sample.py).

The plan boots TP4 serves on GPUs, so what can be proven without one is
proven here: the eviction verdict that decides whether a measured window counts,
the arm checks that tell the logs of the arm that was asked for from another
one, the cell metrics, and the arm configurations ``slot.sh`` hands to the
launcher (its dry run, against a stub launcher). No GPU is touched.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MONO = ROOT / "tools/rdna/mono"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KFD = _load("kfd_evict_sample", MONO / "kfd_evict_sample.py")
DRIVER = _load("mono_w4a8_arm_driver", MONO / "w4a8/arm_driver.py")
ANALYZE = _load("mono_w4a8_analyze", MONO / "w4a8/analyze.py")


# ---- eviction verdict ---------------------------------------------------------


def _snap(procs: dict[int, dict[str, int]], t: float = 0.0) -> dict:
    return {
        "t": t,
        "root": 1,
        "procs": {
            str(pid): {"comm": "w", "evicted_ms": c, "total_ms": sum(c.values())}
            for pid, c in procs.items()
        },
    }


WORKERS = {10: {"2": 100, "3": 0}, 11: {"2": 5, "3": 5}, 12: {"2": 0}, 13: {"2": 9}}


def test_verdict_no_change_is_ok():
    v = KFD.verdict(_snap(WORKERS), _snap(WORKERS, 30.0), min_procs=4)
    assert v["ok"] and v["jumps"] == [] and v["window_s"] == 30.0
    assert KFD.verdict_line(v).startswith("EVICT OK (4 procs")


def test_verdict_counter_jump_is_invalid_and_names_the_worst():
    after = {p: dict(c) for p, c in WORKERS.items()}
    after[11]["3"] += 40
    after[13]["2"] += 7
    v = KFD.verdict(_snap(WORKERS), _snap(after, 5.0), min_procs=4)
    assert not v["ok"]
    assert {(j["pid"], j["gpu"], j["delta_ms"]) for j in v["jumps"]} == {
        (11, "3", 40),
        (13, "2", 7),
    }
    assert v["total_delta_ms"] == 47
    line = KFD.verdict_line(v)
    assert line.startswith("EVICT INVALID") and "pid 11" in line and "+40 ms" in line


def test_verdict_tolerance_and_decrease():
    after = {p: dict(c) for p, c in WORKERS.items()}
    after[10]["2"] += 3
    after[12]["2"] -= 0  # unchanged
    assert not KFD.verdict(_snap(WORKERS), _snap(after))["ok"]
    assert KFD.verdict(_snap(WORKERS), _snap(after), tolerance_ms=3)["ok"]


def test_verdict_lost_worker_and_missing_coverage_are_invalid():
    fewer = {p: c for p, c in WORKERS.items() if p != 12}
    gone = KFD.verdict(_snap(WORKERS), _snap(fewer), min_procs=3)
    assert not gone["ok"] and gone["gone"] == ["12"]
    # no KFD process seen at all is no proof that nothing was evicted
    empty = KFD.verdict(_snap({}), _snap({}), min_procs=4)
    assert not empty["ok"] and "need 4" in empty["reasons"][0]
    assert KFD.verdict(_snap({}), _snap({}), min_procs=0)["ok"]


def test_parse_stat_survives_parentheses_and_spaces_in_the_command_name():
    text = "4242 (VLLM::Worker (TP 3)) S 4000 4242 4242 0 -1 4194560"
    fields = KFD.parse_stat(text)
    assert fields[0] == "S" and int(fields[1]) == 4000


def test_snapshot_reads_the_descendants_only(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    for pid, counters in {101: {"0": 5, "1": 7}, 202: {"0": 9}}.items():
        for gpu, ms in counters.items():
            d = proc / str(pid) / f"stats_{gpu}"
            d.mkdir(parents=True)
            (d / "evicted_ms").write_text(f"{ms}\n")
    (proc / "not-a-pid").mkdir()
    monkeypatch.setattr(KFD, "KFD_PROC", proc)
    monkeypatch.setattr(KFD, "descends_from", lambda pid, root: pid == 101)
    monkeypatch.setattr(KFD, "_comm", lambda pid: "VLLM::Worker_TP0")
    snap = KFD.snapshot(7)
    assert list(snap["procs"]) == ["101"]
    assert snap["procs"]["101"]["evicted_ms"] == {"0": 5, "1": 7}
    assert snap["procs"]["101"]["total_ms"] == 12


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_descends_from_follows_the_real_process_tree():
    child = subprocess.Popen(["sleep", "30"])
    try:
        assert KFD.descends_from(child.pid, os.getpid())
        assert KFD.descends_from(child.pid, child.pid)
        assert not KFD.descends_from(os.getpid(), child.pid)
    finally:
        child.kill()
        child.wait()


# ---- cell metrics ---------------------------------------------------------------


def test_cells_are_whole_waves_with_distinct_seeds():
    names = [c.name for c in DRIVER.CELLS]
    assert len(set(names)) == len(names)
    assert len({c.seed for c in DRIVER.CELLS}) == len(names)
    for c in DRIVER.CELLS:
        assert c.requests % c.concurrency == 0, c.name
    by = {c.name: c for c in DRIVER.CELLS}
    # the MTP2 widths 3, 18 and 24 rows and the two solo TTFT lengths
    assert (by["c1"].concurrency, by["c6"].concurrency) == (1, 6)
    assert by["c8"].concurrency == 8 and by["agg16k"].input_len == 16384
    assert by["ttft4k"].input_len == 4096 and by["ttft16k"].concurrency == 1


def test_summarize_matches_the_bench_metrics_definitions(tmp_path):
    result = tmp_path / "agg16k.json"
    result.write_text(
        json.dumps(
            {
                "completed": 16,
                "failed": 0,
                "total_input_tokens": 16 * 16384,
                "max_concurrency": 8,
                "mean_ttft_ms": 12830.0,
                "median_ttft_ms": 12000.0,
                "p99_ttft_ms": 49000.0,
                "mean_tpot_ms": 47.6,
                "median_itl_ms": 46.9,
                "mean_itl_ms": 60.0,
                "p99_itl_ms": 300.0,
                "output_throughput": 170.0,
                "duration": 120.0,
                "spec_decode_acceptance_length": 2.1,
                "spec_decode_acceptance_rate": 55.0,
                "spec_decode_per_position_acceptance_rates": [0.6, 0.5],
            }
        )
    )
    m = DRIVER.summarize(result, DRIVER.CELLS[-1])
    assert m["in_len"] == 16384
    assert m["prefill_tok_s_req"] == pytest.approx(16384 / 12.83)
    assert m["prefill_tok_s_agg"] == pytest.approx(8 * 16384 / 49.0)
    assert m["decode_tok_s_req"] == pytest.approx(1000 / 47.6)
    assert m["itl_median_ms"] == 46.9 and m["accept_len"] == 2.1


# ---- arm checks -----------------------------------------------------------------

ARMS = {  # arm: (mono, w4a8, hw queues the workers must have)
    "a": (0, 1, 2),
    "b": (1, 0, 4),
    "c": (1, 1, 4),
    "d": (1, 1, 2),
    "e": (0, 1, 4),
}


def _arm_dir(
    tmp_path: Path,
    mono: int,
    w4a8: int,
    hwq: int,
    *,
    k2: bool = True,
    sdot4: bool = True,
    workers_hwq: int | None = None,
    disabled: bool = False,
) -> Path:
    d = tmp_path / "arm"
    (d / "arm").mkdir(parents=True)
    lines = []
    for rank in range(4):
        if mono:
            part = (
                ", W4A8 MoE prefill from 512 tokens (modules, not K2)" if w4a8 else ""
            )
            lines.append(
                f"(Worker_TP{rank}) INFO mono rdna2: --monokernel on TP rank "
                f"{rank}: k1 (<= 24 rows), k2 (<= 8 rows){part}"
            )
        if w4a8 and sdot4:
            lines.append(
                f"[rank{rank}]:[W] Warning: RDNA2 W4A8 sdot4 MoE path active "
                "(config moe_a8_k32_ag)"
            )
    if mono and hwq == 4:
        lines.append("mono rdna2: GPU_MAX_HW_QUEUES 2 -> 4 for the workers (x)")
    if mono and k2:
        lines += [
            f"mono k2_rdna2: {rows} rows (full) -> mono kernel" for rows in (3, 6)
        ]
        lines.append("mono k2_rdna2: 12 rows (full) -> modules")
    if disabled:
        lines.append("mono k2_rdna2 disabled for FlashNextDecoderLayer: no")
    (d / "serve.log").write_text("\n".join(lines) + "\n")
    env = {
        "GPU_MAX_HW_QUEUES": str(workers_hwq if workers_hwq is not None else hwq),
        "VLLM_RDNA2_W4A8_SDOT4": str(w4a8),
    }
    (d / "arm" / "summary.json").write_text(
        json.dumps({"worker_env": {str(p): dict(env) for p in range(4)}})
    )
    return d


def _failed(tmp_path, arm, **kw) -> list[str]:
    mono, w4a8, hwq = ARMS[arm]
    d = _arm_dir(tmp_path, mono, w4a8, hwq, **kw)
    return [n for n, ok, _ in ANALYZE.check_arm(d, mono, w4a8, hwq, 4) if not ok]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_check_arm_accepts_each_arm_as_configured(tmp_path, arm):
    assert _failed(tmp_path, arm) == []


def test_check_arm_catches_a_mono_arm_that_never_ran_k2(tmp_path):
    assert _failed(tmp_path, "c", k2=False) == [
        "K2 serves the 3-row FULL graph",
        "K2 serves the 6-row FULL graph",
    ]
    assert _failed(tmp_path / "x", "c", disabled=True) == [
        "no K1/K2 'disabled for' warning"
    ]


def test_check_arm_catches_w4a8_that_did_not_run(tmp_path):
    # the arm says W4A8, the log has no sdot4 MoE warning: prefill was W4A16
    assert _failed(tmp_path, "a", sdot4=False) == [
        "sdot4 MoE path active (W4A8 prefill ran)"
    ]


def test_check_arm_catches_the_wrong_queue_budget(tmp_path):
    # production arm whose workers were spawned with 4 queues
    assert _failed(tmp_path, "a", workers_hwq=4) == [
        "workers spawned with the arm's knobs"
    ]


def test_check_arm_cli_exit_code(tmp_path):
    d = _arm_dir(tmp_path, 1, 1, 4)
    base = [sys.executable, str(MONO / "w4a8/analyze.py"), "check-arm"]
    ok = subprocess.run(
        [*base, "--arm-dir", str(d), "--mono", "1", "--w4a8", "1", "--hwq", "4"],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0 and "MISMATCH" not in ok.stdout
    bad = subprocess.run(
        [*base, "--arm-dir", str(d), "--mono", "0", "--w4a8", "1", "--hwq", "2"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1 and "MISMATCH mono boot lines" in bad.stdout


# ---- slot.sh dry run --------------------------------------------------------------

STUB_LAUNCHER = """#!/usr/bin/env bash
q=2 w=1 ple=bf16
for a in "$@"; do
    case $a in
        GPU_MAX_HW_QUEUES=*) q=${a#*=} ;;
        VLLM_RDNA2_W4A8_SDOT4=*) w=${a#*=} ;;
        PLE=*) ple=${a#*=} ;;
    esac
done
echo "# ple:       $ple"
echo "export GPU_MAX_HW_QUEUES='$q'"
echo "export VLLM_RDNA2_W4A8_SDOT4='$w'"
echo "'--speculative-config' EXTRA_ARGS=${EXTRA_ARGS:-}"
"""


@pytest.fixture()
def slot_tree(tmp_path):
    tree = tmp_path / "tree"
    slot = tree / "tools/rdna/mono/w4a8"
    slot.mkdir(parents=True)
    shutil.copy(MONO / "w4a8/slot.sh", slot / "slot.sh")
    (tree / "tools/rdna/serve_rdna.sh").write_text(STUB_LAUNCHER)
    (tree / ".sha").write_text("0123456789abcdef\n")
    (tree / "vllm").mkdir()
    for so in ("_C", "_rocm_C"):
        (tree / f"vllm/{so}.abi3.so").write_text("")
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    (tmp_path / "model").mkdir()
    (tmp_path / "windows.json").write_text("{}")
    (tmp_path / "home").mkdir()
    return tree


def _slot(slot_tree, **env):
    base = slot_tree.parent
    full = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(base / "home"),
        "RUN": str(base / "run"),
        "VENV": str(base / "venv"),
        "MODEL": str(base / "model"),
        "MONO_WINDOWS": str(base / "windows.json"),
        "EXPECT_SHA": "0123456789abcdef",
        "DRY": "1",
        **env,
    }
    return subprocess.run(
        ["bash", str(slot_tree / "tools/rdna/mono/w4a8/slot.sh")],
        capture_output=True,
        text=True,
        env=full,
        timeout=60,
    )


def _arm_blocks(stdout: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    cur = ""
    for line in stdout.splitlines():
        if "] arm " in line:
            cur = line.split("] arm ", 1)[1].split(":", 1)[0]
            blocks[cur] = line + "\n"
        elif cur:
            blocks[cur] += line + "\n"
    return blocks


def test_slot_dry_run_hands_each_arm_its_configuration(slot_tree):
    res = _slot(slot_tree, ARMS="a b c d e a0 c2")
    assert res.returncode == 0, res.stdout + res.stderr
    arms = _arm_blocks(res.stdout)
    assert list(arms) == ["a", "b", "c", "d", "e", "a0", "c2"]

    for arm in ("a", "a0", "e"):  # production arms: no extra server arguments
        assert "EXTRA_ARGS=\n" in arms[arm], arm
        assert "expect mono=0 w4a8=1" in arms[arm], arm
    assert "GPU_MAX_HW_QUEUES='2'" in arms["a"]
    assert "GPU_MAX_HW_QUEUES='4'" in arms["e"] and "hwq=4" in arms["e"]

    assert "VLLM_RDNA2_W4A8_SDOT4='0'" in arms["b"]  # mono without W4A8
    assert "EXTRA_ARGS=--monokernel\n" in arms["b"] and "mono=1 w4a8=0" in arms["b"]
    for arm in ("c", "c2"):  # mono + W4A8, the launcher's 2 queues untouched
        assert "VLLM_RDNA2_W4A8_SDOT4='1'" in arms[arm], arm
        assert "EXTRA_ARGS=--monokernel\n" in arms[arm], arm
        assert "hw_queues" not in arms[arm] and "hwq=4" in arms[arm], arm
    # d keeps the recipe's 2 queues through the mono knob
    assert (
        'EXTRA_ARGS=--monokernel --additional-config {"rocm_mono_decode":'
        '{"hw_queues":0}}' in arms["d"]
    )
    assert "hwq=2" in arms["d"] and "mono=1 w4a8=1" in arms["d"]
    assert all("# ple:       bf16" in b for b in arms.values())


def test_slot_dry_run_int4_ple_applies_to_all_but_a0(slot_tree):
    res = _slot(slot_tree, ARMS="a c a0", PLE_MODE="int4")
    assert res.returncode == 0, res.stdout + res.stderr
    arms = _arm_blocks(res.stdout)
    assert "# ple:       int4" in arms["a"] and "# ple:       int4" in arms["c"]
    assert "# ple:       bf16" in arms["a0"]


@pytest.mark.parametrize(
    ("env", "needle"),
    [
        ({"EXPECT_SHA": ""}, "EXPECT_SHA is required"),
        ({"EXPECT_SHA": "feedface"}, "is not feedface"),
        ({"GPUS": "0,1,2,3"}, "GPU 0 is not in 2-9"),
        ({"GPUS": "2,3,4,10"}, "GPU 10 is not in 2-9"),
        ({"ARMS": "zz"}, "unknown arm zz"),
    ],
)
def test_slot_refuses_what_it_must(slot_tree, env, needle):
    res = _slot(slot_tree, **env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert needle in res.stdout


def test_slot_needs_the_built_extensions(slot_tree):
    (slot_tree / "vllm/_rocm_C.abi3.so").unlink()
    res = _slot(slot_tree)
    assert res.returncode == 2 and "_rocm_C.abi3.so is missing" in res.stdout
