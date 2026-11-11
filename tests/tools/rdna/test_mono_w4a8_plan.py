# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the mono + W4A8 GPU plan (tools/rdna/mono/w4a8/).

The plan boots TP4 serves on GPUs, so what can be proven without one is
proven here: the eviction verdict that decides whether a measured window counts,
the arm checks that tell the logs of the arm that was asked for from another
one (queues, W4A8, the memory lock of the pinned PLE table), the host table,
the junction temperature log, the KFD queue probe, the cell metrics, and the
arm configurations ``slot.sh`` hands to the launcher (its dry run and a full run
against a stub ``serve_validate.sh``). No GPU is touched.
"""

import csv
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
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


DRIVER = _load("mono_w4a8_arm_driver", MONO / "w4a8/arm_driver.py")
ANALYZE = _load("mono_w4a8_analyze", MONO / "w4a8/analyze.py")
TEMPS = _load("mono_w4a8_gpu_temps", MONO / "w4a8/gpu_temps.py")
QUEUES = _load("mono_w4a8_queue_probe", MONO / "w4a8/queue_probe.py")


# ---- eviction windows -------------------------------------------------------------


def _take(ms: dict[str, int], t: float = 0.0) -> dict:
    return {"t": t, "ms": ms}


WORKERS = {"Worker_TP0": 100, "Worker_TP1": 5, "Worker_TP2": 0, "Worker_TP3": 9}


def test_window_without_a_change_is_ok():
    v = DRIVER.window(_take(WORKERS), _take(WORKERS, 30.0), workers=4)
    assert v["ok"] and v["reasons"] == [] and v["workers"] == 4
    assert v["window_s"] == 30.0 and v["total_delta_ms"] == 0
    assert DRIVER.window_line(v) == "EVICT OK (4 workers, 30 s)"


def test_window_with_a_counter_jump_is_invalid_and_names_the_workers():
    after = {**WORKERS, "Worker_TP1": 45, "Worker_TP3": 16}
    v = DRIVER.window(_take(WORKERS), _take(after, 5.0), workers=4)
    assert not v["ok"] and v["total_delta_ms"] == 47
    assert v["delta_ms"]["Worker_TP1"] == 40 and v["delta_ms"]["Worker_TP3"] == 7
    line = DRIVER.window_line(v)
    assert line.startswith("EVICT INVALID") and "Worker_TP1=+40ms" in line
    assert "Worker_TP0" not in line  # only the evicted workers are named


def test_window_with_a_lost_worker_or_no_coverage_is_invalid():
    fewer = {k: v for k, v in WORKERS.items() if k != "Worker_TP2"}
    gone = DRIVER.window(_take(WORKERS), _take(fewer), workers=3)
    assert not gone["ok"] and gone["gone"] == ["Worker_TP2"]
    # no KFD worker seen at all is no proof that nothing was evicted
    empty = DRIVER.window(_take({}), _take({}), workers=4)
    assert not empty["ok"] and "need 4" in empty["reasons"][0]
    assert DRIVER.window(_take({}), _take({}), workers=0)["ok"]


def test_sample_appends_one_json_line_per_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(DRIVER.kfd, "snapshot", lambda root: {"Worker_TP0": 7})
    trail = tmp_path / "evict.jsonl"
    assert DRIVER.sample(1, trail) == {"Worker_TP0": 7}
    DRIVER.sample(1, trail)
    rows = [json.loads(line) for line in trail.read_text().splitlines()]
    assert [r["ms"] for r in rows] == [{"Worker_TP0": 7}] * 2


def _fake_proc(tmp_path: Path, *, vmlck_kb: int = 24999936, limit: str = "unlimited"):
    """/proc of a server (pid 100): an engine core and four workers, and the
    worker of another server (pid 999) that must not be counted."""
    proc = tmp_path / "proc"

    def put(pid: int, ppid: int, title: str, **files: str) -> None:
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({title}) S {ppid} {pid} {pid} 0 -1 0\n")
        (d / "cmdline").write_bytes(title.encode() + b"\0")
        for name, text in files.items():
            (d / name).write_text(text)

    put(100, 1, "bash")
    put(200, 100, "VLLM::EngineCore")
    for rank in range(4):
        put(
            300 + rank,
            200,
            f"VLLM::Worker_TP{rank}",
            status=f"Name:\tpython\nVmLck:\t{vmlck_kb} kB\nVmRSS:\t1 kB\n",
            limits="Limit  Soft Limit  Hard Limit  Units\n"
            f"Max locked memory  {limit}  unlimited  bytes\n",
        )
    put(999, 1, "VLLM::Worker_TP0")
    return proc


def test_worker_mem_reads_the_workers_of_this_server(tmp_path):
    proc = str(_fake_proc(tmp_path))
    mem = DRIVER.worker_mem(100, proc)
    assert sorted(mem) == ["300", "301", "302", "303"]
    assert sorted(m["comm"] for m in mem.values()) == [
        f"Worker_TP{r}" for r in range(4)
    ]
    assert {m["vmlck_kb"] for m in mem.values()} == {24999936}
    assert {(m["memlock_soft"], m["memlock_hard"]) for m in mem.values()} == {
        ("unlimited", "unlimited")
    }


def test_worker_mem_shows_a_lock_that_did_not_happen(tmp_path):
    proc = str(_fake_proc(tmp_path, vmlck_kb=0, limit="8388608"))
    mem = DRIVER.worker_mem(100, proc)
    assert {m["vmlck_kb"] for m in mem.values()} == {0}
    assert {m["memlock_soft"] for m in mem.values()} == {"8388608"}


# ---- junction temperatures ----------------------------------------------------------


def _fake_sysfs(tmp_path: Path) -> tuple[Path, Path]:
    """KFD nodes: a CPU, two W7800s (HIP 0 and 1), then gfx1030 cards; node 10 sorts
    before node 2 as text and after it as a number."""
    nodes, drm = tmp_path / "nodes", tmp_path / "drm"
    layout = {0: (0, 0), 1: (140, 128), 2: (140, 129), 3: (144, 130)}
    layout |= {4: (144, 131), 10: (144, 132)}
    for node, (simd, minor) in layout.items():
        d = nodes / str(node)
        d.mkdir(parents=True)
        (d / "properties").write_text(
            f"cpu_cores_count 0\nsimd_count {simd}\ndrm_render_minor {minor}\n"
        )
    for minor, milli in {128: 99000, 130: 61000, 131: 72000, 132: 44500}.items():
        hw = drm / f"renderD{minor}/device/hwmon/hwmon{minor}"
        hw.mkdir(parents=True)
        for i, (label, value) in enumerate(
            [("edge", 50000), ("junction", milli), ("mem", 55000)], start=1
        ):
            (hw / f"temp{i}_label").write_text(label + "\n")
            (hw / f"temp{i}_input").write_text(f"{value}\n")
    return nodes, drm


def test_gpu_temps_maps_hip_ordinals_through_the_kfd_topology(tmp_path):
    nodes, drm = _fake_sysfs(tmp_path)
    assert TEMPS.render_minors(nodes) == [128, 129, 130, 131, 132]
    paths = TEMPS.sensors([2, 3, 4, 9], nodes, drm)
    # HIP 2 is the third GPU node, HIP 4 is node 10 (numeric order), HIP 9 is absent
    assert [TEMPS.read_c(p) for p in paths.values()] == [61.0, 72.0, 44.5, None]
    # the junction sensor, not the edge or memory one
    assert paths[2].name == "temp2_input"


def test_gpu_temps_cli_appends_rows_and_keeps_going_without_a_sensor(tmp_path):
    nodes, drm = _fake_sysfs(tmp_path)
    out = tmp_path / "temps.csv"
    args = ["--gpus", "2,3,9", "--out", str(out), "--period", "0.01"]
    args += ["--kfd-nodes", str(nodes), "--drm", str(drm)]
    assert TEMPS.main([*args, "--count", "2"]) == 0
    assert TEMPS.main([*args, "--count", "1"]) == 0
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["time", "epoch", "hip2", "hip3", "hip9"]  # one header
    assert [r[2:] for r in rows[1:]] == [["61.0", "72.0", ""]] * 3
    assert all(float(r[1]) > 1e9 for r in rows[1:])


def test_gpu_temps_without_a_kfd_topology_logs_blanks(tmp_path, capsys):
    out = tmp_path / "temps.csv"
    args = ["--gpus", "2", "--out", str(out), "--period", "0.01", "--count", "1"]
    assert TEMPS.main([*args, "--kfd-nodes", str(tmp_path / "none")]) == 0
    assert "no KFD topology" in capsys.readouterr().err
    assert [r[2:] for r in list(csv.reader(out.open()))[1:]] == [[""]]


# ---- cell metrics ---------------------------------------------------------------


def test_cells_are_whole_waves_with_distinct_seeds():
    names = [c.name for c in (*DRIVER.CELLS, *DRIVER.WARMUPS)]
    assert len(set(names)) == len(names)
    # a retry runs with seed + 1000 * (attempt - 1): no prompt set repeats, so no
    # attempt, cell or warm-up can hit another's prefix-cache entries
    seeds = [
        c.seed + 1000 * k
        for c in (*DRIVER.CELLS, *DRIVER.WARMUPS)
        for k in range(3)
        if c in DRIVER.CELLS or k == 0
    ]
    assert len(set(seeds)) == len(seeds)
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
    raised: bool | None = None,
    disabled: bool = False,
    pinned: bool = True,
    mlocked: bool = True,
    vmlck_kb: int = 25_000_000,
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
        if pinned:
            lines.append(
                f"(Worker_TP{rank}) INFO Pinned PLE table (1048576, 8192) "
                "torch.bfloat16: 23.84 GiB exact (torch's pinned allocator would "
                f"have taken 32.00 GiB), {'mlocked' if mlocked else 'not mlocked'}."
            )
        if pinned and not mlocked:
            lines.append(
                f"(Worker_TP{rank}) WARNING mlock of the pinned PLE table failed "
                "(Cannot allocate memory): host page reclaim and compaction can "
                "still evict this worker's GPU queues for seconds."
            )
    if (mono and hwq == 4) if raised is None else raised:
        lines.append("mono rdna2: GPU_MAX_HW_QUEUES 2 -> 4 for the workers (x)")
    if mono and k2:
        lines += [
            f"mono k2_rdna2: {rows} rows (full) -> mono kernel" for rows in (3, 6)
        ]
        lines.append("mono k2_rdna2: 12 rows (full) -> modules")
    if disabled:
        lines.append("mono k2_rdna2 disabled for FlashNextDecoderLayer: no")
    (d / "serve.log").write_text("\n".join(lines) + "\n")
    mem = {
        str(p): {
            "comm": f"Worker_TP{p}",
            "vmlck_kb": vmlck_kb,
            "memlock_soft": "unlimited",
            "memlock_hard": "unlimited",
        }
        for p in range(4)
    }
    (d / "arm" / "summary.json").write_text(
        json.dumps(
            {
                "worker_mem": mem,
                "worker_mem_end": mem,
            }
        )
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


def test_check_arm_catches_a_queue_raise_that_should_not_have_happened(tmp_path):
    # arm d keeps the recipe's 2 queues: an engine that raised them is a config bug
    assert _failed(tmp_path, "d", raised=True) == ["mono raised the queue budget"]


def test_check_arm_catches_a_mono_arm_that_did_not_raise_the_queues(tmp_path):
    assert _failed(tmp_path, "c", raised=False) == ["mono raised the queue budget"]


def test_check_arm_catches_a_pinned_table_that_was_not_locked(tmp_path):
    # RLIMIT_MEMLOCK too low: the workers say so, serve anyway, and hold no lock
    failed = _failed(tmp_path, "a", mlocked=False, vmlck_kb=4096)
    assert failed == ["PLE table mlocked", "workers hold the lock (VmLck)"]


def test_check_arm_catches_a_lock_the_kernel_does_not_show(tmp_path):
    # the log says mlocked, VmLck is far below the table
    assert _failed(tmp_path, "c", vmlck_kb=1000) == ["workers hold the lock (VmLck)"]


def test_check_arm_wants_the_pinned_table_for_the_recipe_ple_only(tmp_path):
    mono, w4a8, hwq = ARMS["a"]
    d = _arm_dir(tmp_path, mono, w4a8, hwq, pinned=False)

    def failed(ple: str) -> list[str]:
        checks = ANALYZE.check_arm(d, mono, w4a8, hwq, 4, ple)
        return [n for n, ok, _ in checks if not ok]

    assert failed("recipe") == ["PLE table mlocked"]
    assert failed("int4") == []  # the int4 PLE lives on the GPU: nothing is pinned


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


def test_check_arm_cli_ple_int4_needs_no_pinned_table(tmp_path):
    d = _arm_dir(tmp_path, 0, 1, 2, pinned=False)
    cmd = [sys.executable, str(MONO / "w4a8/analyze.py"), "check-arm"]
    cmd += ["--arm-dir", str(d), "--mono", "0", "--w4a8", "1", "--hwq", "2"]
    recipe = subprocess.run(cmd, capture_output=True, text=True)
    assert recipe.returncode == 1 and "MISMATCH PLE table mlocked" in recipe.stdout
    int4 = subprocess.run([*cmd, "--ple", "int4"], capture_output=True, text=True)
    assert int4.returncode == 0, int4.stdout


# ---- the host around each arm ---------------------------------------------------

START = datetime.fromisoformat("2026-11-05T10:00:00+01:00")
END = datetime.fromisoformat("2026-11-05T10:30:00+01:00")
ARM_TXT = (  # what slot.sh writes around an arm
    "arm=c try=1 desc=mono + W4A8\n"
    f"load=5.53 5.92 7.49 aer=127 date={START.isoformat()}\n"
    "top=92.3 python 12.0 ps\n"
    "memlock=unlimited compact_unevictable=0\n"
    f"rc=0 aer_end=127 end={END.isoformat()}\n"
)


def _host_run(tmp_path: Path) -> Path:
    """A run dir with arm c (4 locked workers, a late peer, an eviction) and the
    temperature log of two GPUs, samples before and after the arm included."""
    run = tmp_path / "run"
    (run / "arms").mkdir(parents=True)
    arm = _arm_dir(tmp_path, 1, 1, 4).rename(run / "arms/c")
    (arm / "arm.txt").write_text(ARM_TXT)
    summary = json.loads((arm / "arm/summary.json").read_text())
    summary["env_end"] = {
        "loadavg": ["6.10", "6.00", "7.00"],
        "top_cpu": ["433 python", "200 ps"],
    }
    (arm / "arm/summary.json").write_text(json.dumps(summary))
    with (arm / "serve.log").open("a") as f:
        f.write("rdna_ar: late peer -- rank 2 arrived 3.1 s after rank 0\n")
        f.write("rank 1 evicted for ~812 ms (KFD queue eviction, 1 times)\n")
    t = START.timestamp()
    samples = [(t - 60, 99.0, 99.0), (t + 60, 60.0, 70.0), (t + 75, 64.0, 72.0)]
    samples.append((t + 1801, 99.0, 99.0))
    (run / "temps.csv").write_text(
        "time,epoch,hip2,hip3\n" + "".join(f"x,{e},{a},{b}\n" for e, a, b in samples)
    )
    queues = [(t - 60, 9), (t + 60, 3), (t + 75, 5), (t + 1801, 9)]
    (run / "queues.jsonl").write_text(
        "".join(
            json.dumps({"t": e, "arm": "c", "workers": {"w0": {"compute": n}}}) + "\n"
            for e, n in queues
        )
    )
    return run


def test_arm_facts_take_the_end_time_not_the_aer_count():
    facts = ANALYZE.arm_facts(ARM_TXT)
    assert facts["start"] == START.isoformat() and facts["end"] == END.isoformat()
    assert (facts["aer_start"], facts["aer_end"]) == ("127", "127")
    assert facts["load"] == "5.53 5.92 7.49"


def test_host_table_reports_the_host_around_each_arm(tmp_path):
    run = _host_run(tmp_path)
    text, data = ANALYZE.host_table(run, {"c": ANALYZE.load_arm(run, "c")})
    row = next(x for x in text.splitlines() if x.startswith("| c |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:] == [
        "5.53 5.92 7.49 / 6.10 6.00 7.00",
        "92.3% python / 433% python",
        "0",
        "23.84 GiB, 4/4 mlocked",
        "23.8 / 23.8",
        "unlimited",
        "1",
        "1",
        "5 (n=2)",
        "hip2:62/64 hip3:71/72",
    ]
    c = data["c"]
    # the samples before and after the arm are outside its window
    assert c["junction_c"] == {"hip2": [62.0, 64.0], "hip3": [71.0, 72.0]}
    assert c["vmlck_gib_min"] == [23.84, 23.84] and c["memlock_soft"] == ["unlimited"]
    assert (c["late_peers"], c["driver_evictions"], c["aer_new"]) == (1, 1, 0)


def test_host_table_without_a_temperature_log_or_worker_memory(tmp_path):
    run = _host_run(tmp_path)
    (run / "temps.csv").unlink()
    arm = run / "arms/c"
    summary = json.loads((arm / "arm/summary.json").read_text())
    del summary["worker_mem"], summary["worker_mem_end"]
    (arm / "arm/summary.json").write_text(json.dumps(summary))
    text, data = ANALYZE.host_table(run, {"c": ANALYZE.load_arm(run, "c")})
    assert data["c"]["junction_c"] == {} and data["c"]["vmlck_gib_min"] == [None, None]
    row = next(x for x in text.splitlines() if x.startswith("| c |"))
    assert row.endswith("| n/a / n/a | n/a | 1 | 1 | 5 (n=2) | n/a |")
    (run / "queues.jsonl").unlink()
    text, data = ANALYZE.host_table(run, {"c": ANALYZE.load_arm(run, "c")})
    assert data["c"]["kfd_compute_queues"] == {}
    row = next(x for x in text.splitlines() if x.startswith("| c |"))
    assert row.endswith("| n/a / n/a | n/a | 1 | 1 | n/a | n/a |")


def test_perf_tables_separate_the_queue_budget_from_the_mono_kernels():
    def arm(itl: float) -> dict:
        return {"cells": {"c1": {"metrics": {"itl_median_ms": itl}}}}

    arms = {"a": arm(24.0), "e": arm(22.0), "c": arm(23.0), "d": arm(24.6)}
    text, data = ANALYZE.perf_tables(arms)
    row = data["c1 ITL median (ms)"]
    assert round(row["mono+W4A8 vs prod"], 1) == -4.2  # the recipe change
    assert round(row["prod 4q vs 2q"], 1) == -8.3  # the queue budget alone
    assert round(row["mono+W4A8 vs prod 4q"], 1) == 4.5  # the kernels, equal budget
    assert round(row["mono+W4A8 2q vs prod"], 1) == 2.5  # the kernels at 2 queues
    assert "| prod 4q vs 2q |" in text and "| mono+W4A8 vs prod 4q |" in text


def test_check_arm_compares_the_kfd_compute_queues_with_the_budget(tmp_path):
    run = _host_run(tmp_path)  # arm c, queue samples of 3 and 5 compute queues
    arm = run / "arms/c"
    name = "KFD compute queues per worker = budget + 1"

    def checks(hwq: int) -> dict[str, bool]:
        return {n: ok for n, ok, _ in ANALYZE.check_arm(arm, 1, 1, hwq, 4)}

    assert checks(4)[name] is True and all(checks(4).values())
    assert checks(2)[name] is False  # 5 queues are no 2-queue budget
    (run / "queues.jsonl").unlink()
    assert name not in checks(4)  # no samples (probe not running): no claim


def test_queue_stats_count_the_compute_queues_inside_the_arm_window():
    t = START.timestamp()
    rows = [
        {"t": t - 5, "arm": "c", "workers": {"w0": {"compute": 9}}},  # before
        {
            "t": t + 10,
            "arm": "c",
            "workers": {"w0": {"compute": 3}, "w1": {"compute": 4}},
        },
        {
            "t": t + 99,
            "arm": "c",
            "workers": {"w0": {"compute": 5}, "w1": {"compute": 5}},
        },
        {"t": t + 50, "arm": "a", "workers": {"w0": {"compute": 2}}},  # another arm
    ]
    facts = ANALYZE.arm_facts(ARM_TXT)
    stats = ANALYZE.queue_stats(rows, "c", facts)
    assert stats == {"samples": 2, "max": 5, "last": {"w0": 5, "w1": 5}}
    assert ANALYZE.queue_stats(rows, "b", facts) == {}
    assert ANALYZE.queue_stats(rows, "c", {"start": None, "end": None}) == {}


def test_queue_samples_skip_unreadable_lines(tmp_path):
    path = tmp_path / "queues.jsonl"
    path.write_text('{"t": 1, "arm": "a", "workers": {}}\nnot json\n\n')
    assert ANALYZE.read_queue_samples(path) == [{"t": 1, "arm": "a", "workers": {}}]
    assert ANALYZE.read_queue_samples(tmp_path / "missing.jsonl") == []


# ---- KFD queues of the running arm -------------------------------------------------


def _fake_kfd_queues(tmp_path: Path, per_worker: dict[int, list[tuple[int, str]]]):
    """/sys/class/kfd/kfd/proc: queues (type, gpuid) of each pid."""
    kfd = tmp_path / "kfd"
    for pid, queues in per_worker.items():
        for i, (kind, gpu) in enumerate(queues):
            d = kfd / str(pid) / "queues" / str(i)
            d.mkdir(parents=True)
            (d / "type").write_text(f"{kind}\n")
            (d / "gpuid").write_text(f"{gpu}\n")
            (d / "size").write_text("4096\n")
    return kfd


def test_queue_probe_counts_the_compute_queues_of_each_worker(tmp_path):
    kfd = _fake_kfd_queues(
        tmp_path,
        {
            300: [(0, "29156")] * 3 + [(1, "29156")],  # 3 compute + 1 SDMA
            301: [(0, "11353")] * 5 + [(2, "11353")],  # a HIQ is not a worker queue
            302: [],
        },
    )
    assert QUEUES.read_queues(300, str(kfd)) == {"compute": 3, "sdma": 1, "gpus": 1}
    assert QUEUES.read_queues(301, str(kfd)) == {"compute": 5, "sdma": 0, "gpus": 1}
    assert QUEUES.read_queues(302, str(kfd)) == {}  # no KFD context
    assert QUEUES.read_queues(999, str(kfd)) == {}


def test_queue_probe_samples_the_server_of_the_newest_live_arm(tmp_path):
    proc = _fake_proc(tmp_path)
    kfd = _fake_kfd_queues(tmp_path, {300 + r: [(0, str(r))] * 3 for r in range(4)})
    run = tmp_path / "run"
    for arm, pid, age in (("a", 100, 0), ("c", 4242, 50), ("b", 100, 100)):
        d = run / "arms" / arm
        d.mkdir(parents=True)
        (d / "serve.pid").write_text(f"{pid} {pid}\n")
        os.utime(d / "serve.pid", (1e9 - age, 1e9 - age))
    # c is the newest but its server is not alive: the newest live one is a
    assert QUEUES.running_arm(run, str(proc)) == ("a", 100)
    row = QUEUES.sample(run, str(proc), str(kfd))
    assert row["arm"] == "a" and row["root"] == 100
    assert {w["compute"] for w in row["workers"].values()} == {3}
    assert sorted(row["workers"]) == [f"Worker_TP{r}" for r in range(4)]
    # no queues yet (the workers have not created any): no row
    assert QUEUES.sample(run, str(proc), str(tmp_path / "empty")) is None
    assert QUEUES.running_arm(tmp_path / "no-run", str(proc)) is None


def test_queue_probe_cli_appends_one_row_per_sample(tmp_path):
    proc = _fake_proc(tmp_path)
    kfd = _fake_kfd_queues(tmp_path, {300 + r: [(0, str(r))] * 5 for r in range(4)})
    run = tmp_path / "run"
    (run / "arms/c").mkdir(parents=True)
    (run / "arms/c/serve.pid").write_text("100 100\n")
    args = ["--run", str(run), "--period", "0.01", "--proc", str(proc)]
    assert QUEUES.main([*args, "--kfd-proc", str(kfd), "--count", "2"]) == 0
    rows = [json.loads(x) for x in (run / "queues.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows] == ["c", "c"]
    assert all(w["compute"] == 5 for r in rows for w in r["workers"].values())


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


def _run_group(
    cmd: list[str], env: dict[str, str], timeout: float
) -> "subprocess.CompletedProcess[str]":
    """``subprocess.run`` in a session of its own. On a timeout the whole group
    dies: slot.sh starts observers (gpu_temps.py, queue_probe.py) that a plain
    ``run`` would leave behind, polling the sysfs of the host for hours."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


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
    return _run_group(
        ["bash", str(slot_tree / "tools/rdna/mono/w4a8/slot.sh")], full, timeout=60
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


# ---- slot.sh full run against a stub serve_validate.sh ---------------------------

# What serve_validate.sh leaves behind for one arm, derived from the arm's launch
# configuration (the env it is called with), so the slot's gating, retries,
# marker checks and fallbacks run for real without a server or a GPU.
STUB_SERVE_VALIDATE = r"""#!/usr/bin/env bash
out=$OUT
mkdir -p "$out/arm"
mono=0
[[ ${EXTRA_ARGS:-} == *--monokernel* ]] && mono=1
w4a8=1 hwq=2 ple=bf16
for a in "$@"; do
    case $a in
        VLLM_RDNA2_W4A8_SDOT4=*) w4a8=${a#*=} ;;
        GPU_MAX_HW_QUEUES=*) hwq=${a#*=} ;;
        PLE=*) ple=${a#*=} ;;
    esac
done
if ((mono)) && [[ ${EXTRA_ARGS} != *'"hw_queues":0'* ]]; then
    hwq=4  # the mono hook raises a lower budget to 4
fi
[[ -n ${STUB_IGNORE_HWQ0:-} && $TAG == *-d ]] && hwq=4
echo "$TAG ple=$ple" >> "$STUB_LOG"
echo "$TAG hip=${HIP_VISIBLE_DEVICES:-}" >> "$STUB_LOG.hip"
n=$(grep -c "^$TAG " "$STUB_LOG")
: > "$out/serve.log"
if ((mono)) && [[ -n ${STUB_OOM:-} && $ple == bf16 ]]; then
    echo "torch.OutOfMemoryError: HIP out of memory" >> "$out/serve.log"
    echo "[0] BOOT FAILED after 100s" > "$out/summary.txt"
    exit 1
fi
lock=mlocked vm=25000000
[[ -n ${STUB_NO_MLOCK:-} ]] && { lock="not mlocked"; vm=4096; }
for r in 0 1 2 3; do
    {
        if ((mono)); then
            part=""
            ((w4a8)) && part=", W4A8 MoE prefill from 512 tokens (modules, not K2)"
            echo "INFO mono rdna2: --monokernel on TP rank $r: k1, k2$part"
        fi
        ((w4a8)) && echo "[rank$r]:[W] Warning: RDNA2 W4A8 sdot4 MoE path active (x)"
        echo "rdna_ar: one-shot all-reduce active (handle 0, rank $r/4)"
        if [[ $ple == bf16 ]]; then
            echo "INFO Pinned PLE table (1048576, 8192) torch.bfloat16: 23.84 GiB" \
                "exact (torch's pinned allocator would have taken 32.00 GiB), $lock."
            [[ $lock == mlocked ]] ||
                echo "WARNING mlock of the pinned PLE table failed (x): see ulimit -l"
        fi
    } >> "$out/serve.log"
done
if ((mono)); then
    if [[ $hwq == 4 ]] &&
        [[ ${EXTRA_ARGS} != *'"hw_queues":0'* || -n ${STUB_IGNORE_HWQ0:-} ]]; then
        echo "mono rdna2: GPU_MAX_HW_QUEUES 2 -> 4 for the workers (x)" \
            >> "$out/serve.log"
    fi
    {
        echo "mono k2_rdna2: 3 rows (full) -> mono kernel"
        echo "mono k2_rdna2: 6 rows (full) -> mono kernel"
    } >> "$out/serve.log"
fi
echo "Model loading took 20.56 GiB memory and 99 seconds" >> "$out/serve.log"
mem=""
for p in 1 2 3 4; do
    mem="$mem${mem:+,}\"$p\":{\"comm\":\"Worker_TP$p\",\"vmlck_kb\":$vm,"
    mem="$mem\"memlock_soft\":\"unlimited\",\"memlock_hard\":\"unlimited\"}"
done
echo "{\"worker_mem\":{$mem},\"worker_mem_end\":{$mem}}" > "$out/arm/summary.json"
echo 1999999 > "$out/serve.pid"
hook=PASS
if [[ -n ${STUB_FAIL_ONCE:-} && $TAG == *"-$STUB_FAIL_ONCE" && $n == 1 ]]; then
    hook=FAIL
fi
{
    echo "[0] READY in 10s served=flash-next"
    echo "[0] PROBE PASS"
    echo "[0] PPL PROBE PASS"
    echo "[0] PREFIX PASS"
    echo "[0] CONCURRENCY PASS"
    if [[ $hook == PASS ]]; then
        echo "[0] LOAD HOOK rc=0 PASS"
    else
        echo "[0] LOAD HOOK rc=1 FAIL"
    fi
} > "$out/summary.txt"
[[ $hook == PASS ]]
"""


class SlotRun:
    """Runs slot.sh of a stub tree whose scratch directories live under ``base``."""

    def __init__(self, tree: Path, base: Path, bindir: Path) -> None:
        self.tree = tree
        self.base = base
        self.bindir = bindir

    def __call__(
        self, *, ulimit_l: int | None = None, **env: str
    ) -> "subprocess.CompletedProcess[str]":
        """``ulimit_l``: soft RLIMIT_MEMLOCK (kB) of the shell slot.sh runs in."""
        base = self.base
        full = {
            "PATH": f"{self.bindir}:/usr/bin:/bin",
            "HOME": str(base / "home"),
            "RUN": str(base / "run"),
            "LEASE": str(base / "lease"),
            "LANES": str(base / "lanes"),
            "BOOT_LOCK": str(base / "boot.lock"),
            "VENV": str(base / "venv"),
            "MODEL": str(base / "model"),
            "MONO_WINDOWS": str(base / "windows.json"),
            "EXPECT_SHA": "0123456789abcdef",
            "REAL_PY": sys.executable,
            "STUB_LOG": str(base / "stub.log"),
            "MONO_PHASES": "perf",
            "MEMLOCK_MIN_KB": "0",
            "TEMPS_PERIOD": "0.2",
            **env,
        }
        slot = str(self.tree / "tools/rdna/mono/w4a8/slot.sh")
        cmd = ["bash", slot]
        if ulimit_l is not None:
            cmd = ["bash", "-c", f'ulimit -S -l {ulimit_l} && exec bash "$0"', slot]
        return _run_group(cmd, full, timeout=600)


@pytest.fixture()
def slot_run(slot_tree, tmp_path):
    """A stub tree that can run slot.sh end to end (no GPU, no server)."""
    tools = slot_tree / "tools/rdna"
    (tools / "port_v031").mkdir(parents=True)
    (tools / "port_v031/serve_validate.sh").write_text(STUB_SERVE_VALIDATE)
    w4a8 = slot_tree / "tools/rdna/mono/w4a8"
    for name in ("analyze.py", "hook.sh", "gpu_temps.py", "queue_probe.py"):
        shutil.copy(MONO / "w4a8" / name, w4a8 / name)
    shutil.copy(ROOT / "tools/rdna/port_v031/kfd_evictions.py", tools / "port_v031/")
    shutil.copy(ROOT / "tools/rdna/port_v031/topk_kl.py", tools / "port_v031/")
    (tools / "mono").mkdir(exist_ok=True)
    shutil.copy(MONO / "logprob_parity.py", tools / "mono/")
    # a python whose -c is the free-memory query (30 GiB on each of four cards)
    py = tmp_path / "venv/bin/python"
    py.unlink()
    py.write_text(
        "#!/usr/bin/env bash\n"
        '[[ ${1:-} == -c ]] && { echo "30.0 30.0 30.0 30.0"; exit 0; }\n'
        'exec "$REAL_PY" "$@"\n'
    )
    py.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sudo").write_text("#!/usr/bin/env bash\nexit 1\n")  # no dmesg: AER n/a
    (bindir / "sudo").chmod(0o755)
    for d in ("lease", "lanes", "run"):
        (tmp_path / d).mkdir()
    (tmp_path / "run/GO").write_text("")

    return SlotRun(slot_tree, tmp_path, bindir)


def test_slot_runs_every_arm_and_releases_the_lease(slot_run):
    res = slot_run(ARMS="c a b d")
    assert res.returncode == 0, res.stdout + res.stderr
    base = slot_run.base
    for arm in ("c", "a", "b", "d"):
        assert (base / f"run/arms/{arm}/DONE").exists(), arm
        assert "MISMATCH" not in (base / f"run/arms/{arm}/check.txt").read_text()
    assert (base / "run/DONE").exists()
    assert (base / "run/results/tables.md").exists()
    # the lease: holder files and the request are gone, the lane marker files are set
    assert list((base / "lease").iterdir()) == []
    assert (base / "lanes/hip25.free").exists()
    assert (base / "lanes/hip25.monow4a8.done").exists()
    assert "lease taken" in res.stdout and "lease released" in res.stdout


def test_slot_marks_the_memory_lock_and_the_host_in_arm_txt(slot_run):
    res = slot_run(ARMS="a")
    assert res.returncode == 0, res.stdout + res.stderr
    text = (slot_run.base / "run/arms/a/arm.txt").read_text()
    assert "memlock=" in text and "compact_unevictable=" in text
    assert (
        "OK       PLE table mlocked"
        in (slot_run.base / "run/arms/a/check.txt").read_text()
    )
    report = (slot_run.base / "run/results/tables.md").read_text()
    assert "Host during the arm" in report and "23.84 GiB, 4/4 mlocked" in report


def test_slot_does_not_accept_a_worker_that_could_not_lock_its_ple_table(slot_run):
    res = slot_run(ARMS="a", STUB_NO_MLOCK="1")
    assert res.returncode == 1, res.stdout + res.stderr
    arm = slot_run.base / "run/arms/a"
    check = (arm / "check.txt").read_text()
    assert "MISMATCH PLE table mlocked" in check
    assert "MISMATCH workers hold the lock (VmLck)" in check
    assert not (arm / "DONE").exists() and not (arm / "attempt1").exists()


def test_slot_refuses_to_start_under_a_low_memlock_limit(slot_run):
    # a shell that did not get the limits.d limit: the workers could not lock
    res = slot_run(ulimit_l=8, MEMLOCK_MIN_KB="33554432")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "RLIMIT_MEMLOCK is 8 kB" in res.stdout and "fresh login" in res.stdout
    assert "waiting for" not in res.stdout  # refused before the wait for GO
    assert not (slot_run.base / "stub.log").exists()
    assert list((slot_run.base / "lease").iterdir()) == []


def test_slot_logs_the_junction_temperatures_of_its_lease(slot_run):
    res = slot_run(ARMS="a")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "junction temperatures of HIP 2,3,4,5 every 0.2 s" in res.stdout
    rows = list(csv.reader((slot_run.base / "run/temps.csv").open()))
    assert rows[0] == ["time", "epoch", "hip2", "hip3", "hip4", "hip5"]
    assert len(rows) >= 2 and all(len(r) == 6 for r in rows)
    assert "KFD queues of the workers every 20 s" in res.stdout


def test_slot_retries_an_arm_whose_hook_failed(slot_run):
    res = slot_run(ARMS="a", STUB_FAIL_ONCE="a")
    assert res.returncode == 0, res.stdout + res.stderr
    arm = slot_run.base / "run/arms/a"
    assert (arm / "DONE").exists()
    assert (arm / "attempt1/summary.txt").exists()  # the failed boot, kept aside
    assert "LOAD HOOK rc=1" in (arm / "attempt1/summary.txt").read_text()
    assert "discarded: boot or hook failed" in res.stdout
    tries = (slot_run.base / "stub.log").read_text().splitlines()
    assert tries == ["mono-w4a8-a ple=bf16"] * 2


def test_slot_does_not_retry_a_config_mismatch(slot_run):
    # d asks for the recipe's 2 queues, the engine raised them to 4: a config bug
    res = slot_run(ARMS="d c", STUB_IGNORE_HWQ0="1")
    assert res.returncode == 1, res.stdout + res.stderr
    d = slot_run.base / "run/arms/d"
    assert not (d / "DONE").exists() and not (d / "attempt1").exists()
    assert "MISMATCH mono raised the queue budget" in (d / "check.txt").read_text()
    assert "markers do not match the arm" in res.stdout
    assert (slot_run.base / "run/arms/c/DONE").exists()  # the next arm still ran
    assert list((slot_run.base / "lease").iterdir()) == []


def test_slot_falls_back_to_the_int4_ple_when_a_mono_arm_ooms(slot_run):
    res = slot_run(ARMS="a c", STUB_OOM="1")
    assert res.returncode == 0, res.stdout + res.stderr
    run = slot_run.base / "run"
    assert "OOM with the recipe PLE" in (run / "FINDINGS.txt").read_text()
    # the arm that ran before the OOM moved aside, then every arm reran with int4
    assert (run / "arms-bf16/a").is_dir() and (run / "arms-bf16/c").is_dir()
    for arm in ("a", "c", "a0"):
        assert (run / f"arms/{arm}/DONE").exists(), arm
    log = (slot_run.base / "stub.log").read_text().splitlines()
    assert log == [
        "mono-w4a8-a ple=bf16",
        "mono-w4a8-c ple=bf16",  # OOM
        "mono-w4a8-a ple=int4",
        "mono-w4a8-c ple=int4",
        "mono-w4a8-a0 ple=bf16",  # production with the recipe's PLE
    ]


def test_slot_stops_on_the_stop_file(slot_run):
    (slot_run.base / "run/STOP").write_text("")
    res = slot_run(ARMS="a")
    assert res.returncode == 3, res.stdout + res.stderr
    assert not (slot_run.base / "run/arms/a/DONE").exists()
    assert list((slot_run.base / "lease").iterdir()) == []


def test_slot_waits_for_the_other_holders_of_its_cards(slot_run):
    # a holder of HIP 4 blocks the lease; STOP then ends the wait without a boot
    (slot_run.base / "lease/Other.hip4").write_text("x")
    (slot_run.base / "run/STOP").write_text("")
    res = slot_run(ARMS="a")
    assert res.returncode == 3 and "STOP while waiting for the lease" in res.stdout
    # the other holder is untouched; our request and holder files are not left behind
    assert sorted(p.name for p in (slot_run.base / "lease").iterdir()) == ["Other.hip4"]


def test_go_file_moves_the_run_to_hip_6_to_9(slot_run):
    base = slot_run.base
    (base / "run/GO").write_text("GPUS=6,7,8,9\nARMS=a\n")
    # a holder of HIP 5 does not block a run on 6-9
    (base / "lease/Other.hip5").write_text("x")
    res = slot_run()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "GO override: GPUS=6,7,8,9" in res.stdout
    lease_line = next(line for line in res.stdout.splitlines() if "lease taken" in line)
    for g in (6, 7, 8, 9):
        assert f"MonoW4A8.hip{g}" in lease_line
    assert "MonoW4A8.hip2" not in lease_line and "MonoW4A8.hip5" not in lease_line
    # the arm saw exactly the leased cards
    assert (base / "stub.log.hip").read_text() == "mono-w4a8-a hip=6,7,8,9\n"
    # everything of ours is gone again; the lane markers name the group
    assert sorted(p.name for p in (base / "lease").iterdir()) == ["Other.hip5"]
    assert (base / "lanes/hip69.free").exists()
    assert (base / "lanes/hip69.monow4a8.done").exists()
    assert not (base / "lanes/hip25.free").exists()


def test_a_holder_of_one_card_of_the_group_blocks_the_lease(slot_run):
    base = slot_run.base
    (base / "run/GO").write_text("GPUS=6,7,8,9\nARMS=a\n")
    (base / "lease/GRANT-Determinism.hip9").write_text("x")
    (base / "run/STOP").write_text("")
    res = slot_run()
    assert res.returncode == 3 and "STOP while waiting for the lease" in res.stdout
    assert not (base / "stub.log.hip").exists()  # nothing booted


def test_go_file_only_overrides_the_whitelisted_keys(slot_run):
    base = slot_run.base
    (base / "run/GO").write_text(
        "ARMS=a\nEXPECT_SHA=deadbeef\nTREE=/nonexistent\nRUN=/nonexistent\n"
    )
    res = slot_run()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "GO override: ARMS=a" in res.stdout
    assert "GO override: EXPECT_SHA" not in res.stdout
    assert "GO override: TREE" not in res.stdout
    assert (base / "run/arms/a/DONE").exists()


def test_go_file_cannot_lease_the_w7800s(slot_run):
    base = slot_run.base
    (base / "run/GO").write_text("GPUS=0,1,2,3\n")
    res = slot_run(ARMS="a")
    assert res.returncode == 2 and "GPU 0 is not in 2-9" in res.stdout
    assert list((base / "lease").iterdir()) == []
    assert not (base / "stub.log.hip").exists()


def _alive(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_a_timed_out_slot_leaves_none_of_its_children_behind(tmp_path):
    pidfile = tmp_path / "child.pid"
    cmd = ["bash", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]
    with pytest.raises(subprocess.TimeoutExpired):
        _run_group(cmd, {"PATH": "/usr/bin:/bin"}, timeout=1)
    pid = int(pidfile.read_text())
    for _ in range(50):
        if not _alive(pid):
            return
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pytest.fail("the child of the timed-out shell is still alive")
