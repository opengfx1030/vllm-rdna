# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-visible CPU checks for tools/rdna/eplb/qualify_boot.py.

No root conftest, no GPU and no torch: run with

    python -m pytest --confcutdir=tests/tools/rdna \
        tests/tools/rdna/test_eplb_qualification.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "rdna" / "eplb" / "qualify_boot.py"
SPEC = importlib.util.spec_from_file_location("eplb_qualify_boot_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qb
SPEC.loader.exec_module(qb)


def _write_device(directory: Path, total: int) -> None:
    for name, label in (
        ("aer_dev_correctable", "CORRECTABLE"),
        ("aer_dev_nonfatal", "NONFATAL"),
        ("aer_dev_fatal", "FATAL"),
    ):
        (directory / name).write_text(f"TOTAL_ERR_{label} {total}\n")


def _write_rootport(directory: Path, total: int) -> None:
    for name in (
        "aer_rootport_total_err_cor",
        "aer_rootport_total_err_nonfatal",
        "aer_rootport_total_err_fatal",
    ):
        (directory / name).write_text(f"{total}\n")


def _fake_sys(tmp_path: Path, total: int = 0) -> tuple[Path, Path, Path]:
    """A fake /sys with one endpoint and one ancestor root port."""
    root = tmp_path / "sys"
    bridge = root / "devices" / "pci0000:00" / "0000:00:01.1"
    endpoint = bridge / "0000:d9:00.0"
    endpoint.mkdir(parents=True)
    link = root / "bus" / "pci" / "devices" / "0000:d9:00.0"
    link.parent.mkdir(parents=True)
    link.symlink_to(endpoint, target_is_directory=True)
    (endpoint / "class").write_text("0x030000\n")
    _write_device(endpoint, total)
    _write_rootport(bridge, total)
    return root, endpoint, bridge


def _snapshot(tmp_path: Path, root: Path, name: str) -> Path:
    out = tmp_path / name
    rc = qb.main(["aer", "--root", str(root), "--buses", "d9", "--out", str(out)])
    assert rc == 0
    return out


def test_aer_compare_rejects_changed_and_reset(tmp_path: Path, capsys) -> None:
    root, endpoint, _bridge = _fake_sys(tmp_path, total=2)
    before = _snapshot(tmp_path, root, "before.json")
    _write_device(endpoint, 1)
    reset = _snapshot(tmp_path, root, "reset.json")
    assert qb.main(["aer-compare", str(before), str(reset)]) == 1
    assert "reset" in capsys.readouterr().err

    _write_device(endpoint, 4)
    changed = _snapshot(tmp_path, root, "changed.json")
    assert qb.main(["aer-compare", str(reset), str(changed)]) == 1
    assert "changed" in capsys.readouterr().err


def test_aer_rejects_missing_telemetry(tmp_path: Path, capsys) -> None:
    root, endpoint, _bridge = _fake_sys(tmp_path)
    (endpoint / "aer_dev_fatal").unlink()
    assert (
        qb.main(
            [
                "aer",
                "--root",
                str(root),
                "--buses",
                "d9",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    assert "partial AER telemetry" in capsys.readouterr().err

    (tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:d9:00.0").unlink()
    assert (
        qb.main(
            [
                "aer",
                "--root",
                str(root),
                "--buses",
                "d9",
                "--out",
                str(tmp_path / "out2.json"),
            ]
        )
        == 1
    )
    assert "no PCI device" in capsys.readouterr().err

    assert (
        qb.main(
            [
                "aer-compare",
                str(tmp_path / "missing.json"),
                str(tmp_path / "missing.json"),
            ]
        )
        == 1
    )
    assert "REFUSED" in capsys.readouterr().err


def test_aer_rejects_partial_rootport_group(tmp_path: Path, capsys) -> None:
    root, _endpoint, bridge = _fake_sys(tmp_path)
    (bridge / "aer_rootport_total_err_cor").unlink()
    assert (
        qb.main(
            [
                "aer",
                "--root",
                str(root),
                "--buses",
                "d9",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    assert "partial AER root-port telemetry" in capsys.readouterr().err


def test_faults_rejects_hang_and_application_errors(tmp_path: Path, capsys) -> None:
    log = tmp_path / "serve.log"
    log.write_text(
        "boot ok\n"
        "amdgpu: GPU Hang detected\n"
        "amdgpu: HW Exception on ring\n"
        "Fatal Python error: Aborted\n"
        "Online EPLB rank 2: OFF at step 9 (vote)\n"
    )
    assert qb.main(["faults", "--log", str(log)]) == 1
    out = capsys.readouterr().out
    assert "GPU Hang" in out
    assert "HW Exception" in out
    assert "Online EPLB rank 2: OFF" in out

    clean = tmp_path / "clean.log"
    clean.write_text("boot ok\nno faults here\n")
    assert qb.main(["faults", "--log", str(clean)]) == 0


def test_faults_missing_log_is_unknown(tmp_path: Path, capsys) -> None:
    assert qb.main(["faults", "--log", str(tmp_path / "absent.log")]) == 3
    assert "unreadable" in capsys.readouterr().err


def test_temps_reject_thermal_abort(tmp_path: Path, capsys) -> None:
    log = tmp_path / "temps-gpus.log"
    log.write_text(
        "10:00:00 d9=40 dc=41 df=42 e2=43\n10:00:05 d9=40 dc=41 df=42 e2=43\n"
    )
    abort = tmp_path / "THERMAL_ABORT"
    abort.write_text("d9=96 C\n")
    assert (
        qb.main(
            [
                "temps",
                "--log",
                str(log),
                "--buses",
                "d9 dc df e2",
                "--abort",
                str(abort),
            ]
        )
        == 1
    )
    assert "thermal abort" in capsys.readouterr().out


def test_gpu_mapping_rejects_wrong_bus_and_name() -> None:
    info = {
        "gpus": [
            {"hip": 2, "bus": "d9", "gfx": "gfx1030", "name": "AMD Radeon Pro V620"},
            {"hip": 3, "bus": "dc", "gfx": "gfx1030", "name": "AMD Radeon Pro V620"},
            {"hip": 4, "bus": "df", "gfx": "gfx1030", "name": "AMD Radeon Pro V620"},
            {"hip": 5, "bus": "e2", "gfx": "gfx1030", "name": "AMD Radeon Pro V620"},
        ]
    }
    assert qb.gpu_problems(info) == []
    info["gpus"][1]["bus"] = "00"
    info["gpus"][2]["name"] = "Unknown Device"
    problems = qb.gpu_problems(info)
    assert any("HIP 3: bus 00 != dc" in item for item in problems)
    assert any("HIP 4: name Unknown Device lacks V620" in item for item in problems)


CELLS = "1024:512:1 1024:512:8"


def _boot(tmp_path: Path, log: str, payloads: dict[str, dict] | None = None) -> Path:
    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "serve.log").write_text(log)
    cells = payloads or {
        "bench-1024x512-c1": {
            "benchmark_start_time_monotonic": 100.0,
            "duration": 10.0,
        },
        "bench-1024x512-c8": {
            "benchmark_start_time_monotonic": 200.0,
            "duration": 20.0,
        },
    }
    for name, payload in cells.items():
        (boot / f"{name}.json").write_text(json.dumps(payload))
    return boot


def _start(seq: int, clock: float, step: int = 512) -> str:
    return (
        f"1999-01-01 00:00:00 INFO x Online EPLB: plan {seq} at step {step}, "
        f"8 swap(s) monotonic={clock:.9f}"
    )


def _done(seq: int, clock: float) -> str:
    return (
        f"1999-01-01 00:00:00 INFO x Online EPLB: plan {seq} done, 8 swap(s) "
        f"in 300 steps (0.1 s) monotonic={clock:.9f}"
    )


def _log(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_plan_overlap_uses_native_monotonic_windows(tmp_path: Path) -> None:
    boot = _boot(
        tmp_path,
        _log(
            _start(1, 99.0),
            _done(1, 105.0),
            _start(2, 195.0, 1024),
            _done(2, 205.0),
        ),
    )
    coverage = qb.check_plan_overlap(boot, CELLS)
    assert coverage.hits == {
        "bench-1024x512-c1": [1],
        "bench-1024x512-c8": [2],
    }
    assert coverage.unfinished == ()


def test_plan_overlap_groups_interleaved_worker_logs(tmp_path: Path) -> None:
    log = _log(
        _done(2, 205.0),
        _start(2, 195.0, 1024),
        _done(1, 104.0),
        _start(1, 98.5),
        _done(1, 105.0),
        _start(1, 99.0),
    )
    spans, unfinished = qb.plan_spans(log)
    assert spans == {1: qb.PlanSpan(98.5, 105.0), 2: qb.PlanSpan(195.0, 205.0)}
    assert unfinished == ()
    coverage = qb.check_plan_overlap(_boot(tmp_path, log), CELLS)
    assert coverage.hits["bench-1024x512-c1"] == [1]
    assert coverage.hits["bench-1024x512-c8"] == [2]


@pytest.mark.parametrize(("start", "end"), [(50.0, 80.0), (300.0, 320.0)])
def test_plan_overlap_rejects_plans_outside_every_cell(
    tmp_path: Path, start: float, end: float
) -> None:
    boot = _boot(tmp_path, _log(_start(1, start), _done(1, end)))
    with pytest.raises(qb.QualificationError, match="bench-1024x512-c1"):
        qb.check_plan_overlap(boot, CELLS)


def test_plan_overlap_excludes_unfinished_tail(tmp_path: Path) -> None:
    boot = _boot(
        tmp_path,
        _log(_start(1, 99.0), _done(1, 105.0), _start(2, 205.0, 1024)),
    )
    coverage = qb.check_plan_overlap(boot, "1024:512:1")
    assert coverage.hits == {"bench-1024x512-c1": [1]}
    assert coverage.unfinished == (2,)
    with pytest.raises(qb.QualificationError, match="bench-1024x512-c8") as err:
        qb.check_plan_overlap(boot, CELLS)
    assert "unfinished plan(s) 2 are excluded" in str(err.value)


def test_plan_overlap_rejects_absent_plan_evidence(tmp_path: Path) -> None:
    boot = _boot(tmp_path, "1999-01-01 00:00:00 INFO x no plan here\n")
    with pytest.raises(
        qb.QualificationError, match="no started-and-committed EPLB plan"
    ):
        qb.check_plan_overlap(boot, CELLS)


def test_plan_overlap_rejects_commit_without_start(tmp_path: Path) -> None:
    boot = _boot(tmp_path, _log(_done(1, 105.0)))
    with pytest.raises(qb.QualificationError, match="committed without a start"):
        qb.check_plan_overlap(boot, CELLS)


def test_plan_overlap_rejects_commit_before_start(tmp_path: Path) -> None:
    boot = _boot(tmp_path, _log(_start(1, 120.0), _done(1, 110.0)))
    with pytest.raises(qb.QualificationError, match="commits before it starts"):
        qb.check_plan_overlap(boot, CELLS)


def test_plan_overlap_rejects_missing_monotonic_clock(tmp_path: Path) -> None:
    line = "1999-01-01 00:00:00 INFO x Online EPLB: plan 1 at step 512, 8 swap(s)"
    boot = _boot(tmp_path, _log(line))
    with pytest.raises(qb.QualificationError, match="no monotonic= clock"):
        qb.check_plan_overlap(boot, CELLS)


@pytest.mark.parametrize("clock", ["nan", "inf", "-inf"])
def test_plan_overlap_rejects_nonfinite_monotonic_clock(
    tmp_path: Path, clock: str
) -> None:
    line = (
        "1999-01-01 00:00:00 INFO x Online EPLB: plan 1 at step 512, 8 "
        f"swap(s) monotonic={clock}"
    )
    boot = _boot(tmp_path, _log(line))
    with pytest.raises(qb.QualificationError, match="not finite"):
        qb.check_plan_overlap(boot, CELLS)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"duration": 10.0}, "benchmark_start_time_monotonic"),
        (
            {"benchmark_start_time_monotonic": float("nan"), "duration": 10.0},
            "not finite",
        ),
        (
            {"benchmark_start_time_monotonic": 100.0},
            "duration: missing or non-numeric",
        ),
        (
            {"benchmark_start_time_monotonic": 100.0, "duration": float("inf")},
            "not finite",
        ),
    ],
)
def test_plan_overlap_rejects_bad_native_clock(
    tmp_path: Path, payload: dict, match: str
) -> None:
    boot = _boot(
        tmp_path,
        _log(_start(1, 99.0), _done(1, 105.0)),
        {
            "bench-1024x512-c1": payload,
            "bench-1024x512-c8": {
                "benchmark_start_time_monotonic": 200.0,
                "duration": 20.0,
            },
        },
    )
    with pytest.raises(qb.QualificationError, match=match):
        qb.check_plan_overlap(boot, "1024:512:1")


def test_gate_requires_eight_passing_rows() -> None:
    pytest.importorskip("regex")
    cells = ("1024x512-c1", "1024x512-c8", "16384x1024-c1", "16384x1024-c8")
    rows = [
        {"cell": cell, "metric": metric, "verdict": "pass"}
        for cell in cells
        for metric in ("median_itl_ms", "p99_itl_ms")
    ]
    assert qb.gate_row_problems(rows, qb.R2_CELLS) == []
    rows[0]["verdict"] = "FAIL"
    problems = qb.gate_row_problems(rows, qb.R2_CELLS)
    assert any("not pass" in item for item in problems)


def test_speed_regressions_use_cell_verdicts_not_metric_labels():
    receipt = {
        "speed": {
            "cells": [
                {
                    "name": "bench-1024x512-c1",
                    "verdict": "win",
                    "metrics": {"out_tps": {"verdict": "regression"}},
                },
                {"name": "bench-1024x512-c8", "verdict": "regression"},
                {"name": "bench-16384x1024-c1", "verdict": "none"},
                {"name": "bench-16384x1024-c8", "verdict": "regression"},
            ]
        }
    }
    assert qb._speed_regressions(receipt) == [
        "bench-1024x512-c8",
        "bench-16384x1024-c8",
    ]


@pytest.mark.parametrize("speed", [None, {}, {"cells": []}, {"cells": [{}]}])
def test_missing_speed_verdicts_cannot_be_reported_as_zero_regressions(speed):
    with pytest.raises(qb.QualificationError):
        qb._speed_regressions({"speed": speed})
