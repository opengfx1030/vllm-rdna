#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native guards and evidence for tools/rdna/eplb/qualify_final.sh.

The campaign entrypoint owns the fixed settings and the boots; this module
turns their artefacts into a fail-closed verdict without a second benchmark
verdict:

* ``gpus`` resolves HIP 2,3,4,5 to PCI buses in one separate torch process
  (``pr_receipt_report.py env``), requires the b4 mapping (d9 dc df e2), the
  V620 name and gfx1030, and prints the bus list.  The guard process itself
  never imports torch.
* ``aer`` snapshots the sysfs AER counters of the four GPU endpoints and every
  ancestor bridge (the parser is ``pciescope.collectors.aer.parse_aer``);
  ``aer-compare`` refuses any change, any reset and any missing device.
* ``faults`` counts the shared serve-log fault phrases (``SERVE_LOG_FAULTS``)
  plus the EPLB/application signals; a missing serve.log is unknown, never
  zero.
* ``temps`` requires numeric tempwatch samples for all four slot buses and no
  thermal-abort file.
* ``boot`` validates one r2 arm: the quality/greedy/concurrency/post-load
  gates, the four native cells, KFD/INVALID windows, guard telemetry,
  application faults, the four Mono ranks, the mapped 3-row FULL K2 evidence
  and, for on arms, one committed EPLB plan interval overlapping every cell.
* ``r1`` validates the pr_receipt.sh abba receipt and its six boots; ``r2``
  assembles the four arm evidence files and the itl_gate rows.
* ``gate`` requires the eight gated itl_gate rows, all ``pass``.

Clocks: every native cell result carries ``benchmark_start_time_monotonic``
and its native ``duration``; EPLB plan start and commit lines carry
``monotonic=`` values from the same host clock.  Plan lines are grouped by
plan number, so worker log ordering is irrelevant.  A start without a commit
is an unfinished tail: it is reported and excluded from coverage, never
treated as committed.  Missing or non-finite clocks are invalid.

This module must run with ``HIP_VISIBLE_DEVICES`` empty (the guard process).
Only ``gpus`` starts a torch process, and it sets HIP_VISIBLE_DEVICES itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import regex as re

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPORT_PATH = TOOLS / "pr_receipt_report.py"
GATE_PATH = HERE / "itl_gate.py"

EXPECTED_HIP = (2, 3, 4, 5)
EXPECTED_BUSES = ("d9", "dc", "df", "e2")
EXPECTED_GFX = "gfx1030"
EXPECTED_NAME = "V620"
R2_CELLS = "1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"
R2_ARMS = ("A1", "B1", "B2", "A2")
R1_BOOTS = (
    ("XA", "base", "exact"),
    ("XB", "head", "exact"),
    ("A1", "base", "speed"),
    ("B1", "head", "speed"),
    ("B2", "head", "speed"),
    ("A2", "base", "speed"),
)
R1_GATES = ("exact", "quality", "overlap", "stall")
GATE_OK = ("PASS", "NOT RUN")
R2_INJECTION_NOTE = (
    "Injection coverage was not run (STALL_PROBE=0). The old paired injection "
    "gate is historical and exercised the modules path only (>=21 rows); no "
    "mapped K2 under injection is claimed here."
)

AER_FILES = (
    ("aer_dev_correctable", "aer.correctable"),
    ("aer_dev_nonfatal", "aer.nonfatal"),
    ("aer_dev_fatal", "aer.fatal"),
)
ROOT_TOTAL_FILES = (
    ("aer_rootport_total_err_cor", "aer.rootport_correctable"),
    ("aer_rootport_total_err_nonfatal", "aer.rootport_nonfatal"),
    ("aer_rootport_total_err_fatal", "aer.rootport_fatal"),
)

PLAN_LINE = re.compile(r"Online EPLB: plan (\d+) ")
PLAN_START = re.compile(r"Online EPLB: plan (\d+) at step (\d+), (\d+) swap\(s\)")
PLAN_DONE = re.compile(r"Online EPLB: plan (\d+) done, (\d+) swap\(s\) in (\d+) steps")
MONOTONIC = re.compile(r"\bmonotonic=(\S+)")
TEMPS_LINE = re.compile(r"^(\d\d:\d\d:\d\d)((?: \S+=\S+)+)$")
MONO_RANK = re.compile(r"mono rdna2: --monokernel on TP rank (\d+)")
K2_ROW = re.compile(r"mono k2_rdna2: (\d+) rows \(([a-z]+)\) -> (mono kernel|modules)")
WORKER_TP = re.compile(r"\(Worker_TP(\d+)_EP\d+ pid=\d+\)")
EPLB_ON = "Online EPLB is ON"
EPLB_COUNTERS = "EPLB load counters are ON"
K2_FULL_ROWS = 3
MODULE_MIN_ROWS = 24


class QualificationError(ValueError):
    """An artefact is missing, contradictory or not joinable; never guessed."""


_MODULES: dict[str, ModuleType] = {}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def report_module() -> ModuleType:
    """``pr_receipt_report.py``: cells, summary parsing and the shared faults."""
    if "receipt" not in _MODULES:
        _MODULES["receipt"] = _load(REPORT_PATH, "pr_receipt_report")
    return _MODULES["receipt"]


def gate_module() -> ModuleType:
    """``itl_gate.py``: the existing gate and its INVALID-cell convention."""
    if "itl_gate" not in _MODULES:
        _MODULES["itl_gate"] = _load(GATE_PATH, "itl_gate")
    return _MODULES["itl_gate"]


def _parse_aer(text: str) -> dict[str, int]:
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    if "aer" not in _MODULES:
        from pciescope.collectors import aer

        _MODULES["aer"] = aer
    return _MODULES["aer"].parse_aer(text)


# --------------------------------------------------------------------- faults


def _fault_pattern() -> re.Pattern[str]:
    shared = report_module().SERVE_LOG_FAULTS.pattern
    return re.compile(
        shared
        + r"|Fatal Python error|rdna_ar: WEDGED|queues evicted"
        + r"|Online EPLB rank \d+: OFF"
    )


def fault_lines(path: Path) -> list[str]:
    """Matching lines of a serve.log; a missing file raises (unknown != zero)."""
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        raise QualificationError(f"{path}: serve.log is unreadable ({exc})") from exc
    pattern = _fault_pattern()
    return [line for line in text.splitlines() if pattern.search(line)]


# --------------------------------------------------------------- GPU mapping


def gpu_problems(info: dict[str, Any]) -> list[str]:
    """Why ``pr_receipt_report.py env`` does not describe b4 HIP 2,3,4,5."""
    problems: list[str] = []
    error = info.get("torch_error")
    if error:
        problems.append(f"the torch probe failed: {error}")
    gpus = info.get("gpus")
    if not isinstance(gpus, list):
        return problems + ["the torch probe reported no gpus list"]
    if len(gpus) != len(EXPECTED_HIP):
        problems.append(
            f"the torch probe saw {len(gpus)} device(s), expected {len(EXPECTED_HIP)}"
        )
    for index, hip in enumerate(EXPECTED_HIP):
        if index >= len(gpus):
            break
        entry = gpus[index]
        if not isinstance(entry, dict):
            problems.append(f"device {index}: not an object")
            continue
        try:
            seen_hip = int(entry.get("hip"))
        except (TypeError, ValueError):
            problems.append(f"device {index}: no HIP index")
            seen_hip = -1
        if seen_hip != hip:
            problems.append(f"device {index}: HIP {seen_hip} != {hip}")
        bus = str(entry.get("bus", "")).lower().removeprefix("0000:")
        expected_bus = EXPECTED_BUSES[index]
        if bus != expected_bus:
            problems.append(f"HIP {hip}: bus {bus or '?'} != {expected_bus}")
        if str(entry.get("gfx", "")) != EXPECTED_GFX:
            problems.append(f"HIP {hip}: gfx {entry.get('gfx')} != {EXPECTED_GFX}")
        if EXPECTED_NAME not in str(entry.get("name", "")):
            problems.append(
                f"HIP {hip}: name {entry.get('name')} lacks {EXPECTED_NAME}"
            )
    return problems


def resolve_gpus(venv: Path, out: Path) -> str:
    """Run the torch probe in its own process with HIP_VISIBLE_DEVICES set."""
    python = venv / "bin" / "python"
    if not python.is_file():
        raise QualificationError(f"{venv}: no bin/python")
    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = ",".join(str(hip) for hip in EXPECTED_HIP)
    site = venv / "lib" / "python3.12" / "site-packages"
    rocm_lib = site / "_rocm_sdk_libraries" / "lib"
    rocm_core = site / "_rocm_sdk_core" / "lib"
    torch_lib = site / "torch" / "lib"
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(rocm_lib),
            str(rocm_core / "host-math" / "lib"),
            str(rocm_core / "rocm_sysdeps" / "lib"),
            str(rocm_core),
            str(torch_lib),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    command = [
        str(python),
        str(REPORT_PATH),
        "env",
        "--gpus",
        ",".join(str(hip) for hip in EXPECTED_HIP),
        "--out",
        str(out),
        "--buses",
    ]
    try:
        proc = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError(f"the torch probe did not run: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise QualificationError(
            "the torch probe failed"
            + (f": {detail[-1]}" if detail else f" (rc {proc.returncode})")
        )
    try:
        info = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        raise QualificationError(
            f"{out}: unreadable torch probe output ({exc})"
        ) from exc
    problems = gpu_problems(info)
    if problems:
        raise QualificationError("; ".join(problems))
    return " ".join(EXPECTED_BUSES)


# ------------------------------------------------------------------------ AER


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _aer_total(path: Path) -> int:
    text = _read_text(path)
    if text is None:
        raise QualificationError(f"{path}: unreadable")
    totals = [
        value
        for name, value in _parse_aer(text).items()
        if name.startswith("TOTAL_ERR_")
    ]
    if len(totals) != 1:
        raise QualificationError(f"{path}: no single TOTAL_ERR_* count")
    return totals[0]


def _device_counters(directory: Path) -> dict[str, dict[str, int]] | None:
    present = [name for name, _ in AER_FILES if (directory / name).is_file()]
    totals = [name for name, _ in ROOT_TOTAL_FILES if (directory / name).is_file()]
    if not present and not totals:
        return None
    if present and len(present) != len(AER_FILES):
        raise QualificationError(
            f"{directory}: partial AER telemetry ({len(present)} of "
            f"{len(AER_FILES)} device classes)"
        )
    if totals and len(totals) != len(ROOT_TOTAL_FILES):
        raise QualificationError(
            f"{directory}: partial AER root-port telemetry ({len(totals)} of "
            f"{len(ROOT_TOTAL_FILES)})"
        )
    entry: dict[str, dict[str, int]] = {"dev": {}, "rootport": {}}
    for name, metric in AER_FILES:
        if name in present:
            entry["dev"][metric] = _aer_total(directory / name)
    for name, metric in ROOT_TOTAL_FILES:
        if name not in totals:
            continue
        text = _read_text(directory / name)
        try:
            entry["rootport"][metric] = int((text or "").strip())
        except ValueError as exc:
            raise QualificationError(
                f"{directory / name}: not an integer ({text!r})"
            ) from exc
    return entry


def _endpoint_chain(root: Path, bus: str) -> list[Path]:
    link = root / "bus" / "pci" / "devices" / f"0000:{bus}:00.0"
    if not link.exists():
        raise QualificationError(f"{link}: no PCI device for HIP bus {bus}")
    real = Path(os.path.realpath(link))
    root_real = Path(os.path.realpath(root))
    try:
        inside = os.path.commonpath([str(real), str(root_real)]) == str(root_real)
    except ValueError:
        inside = False
    if not inside:
        raise QualificationError(f"{link} resolves outside {root_real}")
    if real.name != f"0000:{bus}:00.0":
        raise QualificationError(f"{link} resolves to {real.name}, not the bus")
    chain = []
    current = real
    while True:
        chain.append(current)
        if current == root_real:
            break
        parent = current.parent
        if parent == current or os.path.commonpath(
            [str(parent), str(root_real)]
        ) != str(root_real):
            break
        current = parent
    return chain


def aer_snapshot(root: Path, buses: Sequence[str]) -> dict[str, Any]:
    """Counters of every endpoint and ancestor bridge on the four GPU paths."""
    root_real = Path(os.path.realpath(root))
    endpoints: dict[str, str] = {}
    devices: dict[str, Any] = {}
    for bus in buses:
        chain = _endpoint_chain(root_real, bus)
        endpoint = os.path.relpath(chain[0], root_real)
        for directory in chain:
            counters = _device_counters(directory)
            if counters is None:
                continue
            relative = os.path.relpath(directory, root_real)
            devices[relative] = {"bdf": directory.name, **counters}
        endpoint_entry = devices.get(endpoint)
        if endpoint_entry is None or not endpoint_entry["dev"]:
            raise QualificationError(
                f"0000:{bus}:00.0: no device AER counters on the endpoint"
            )
        endpoints[bus] = endpoint
    if len(set(endpoints.values())) != len(buses):
        raise QualificationError("two HIP buses resolved to the same PCI device")
    snapshot: dict[str, Any] = {
        "root": str(root_real),
        "buses": list(buses),
        "endpoints": endpoints,
        "devices": devices,
    }
    digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()
    snapshot["sha256"] = digest
    return snapshot


def compare_aer(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Refuse a changed, reset or disappeared AER counter."""
    for key in ("root", "buses", "endpoints"):
        if before.get(key) != after.get(key):
            raise QualificationError(f"AER {key} changed across the boot")
    old = before.get("devices") or {}
    new = after.get("devices") or {}
    missing = sorted(set(old) - set(new))
    extra = sorted(set(new) - set(old))
    problems = [f"{name}: endpoint or bridge disappeared" for name in missing]
    problems += [
        f"{name}: endpoint or bridge appeared during the boot" for name in extra
    ]
    for name in sorted(set(old) & set(new)):
        for section in ("dev", "rootport"):
            old_values = old[name].get(section) or {}
            new_values = new[name].get(section) or {}
            gone = sorted(set(old_values) - set(new_values))
            problems += [
                f"{name} {section} {metric}: telemetry disappeared" for metric in gone
            ]
            for metric, value in old_values.items():
                if metric not in new_values:
                    continue
                current = new_values[metric]
                if current == value:
                    continue
                kind = "reset" if current < value else "changed"
                problems.append(f"{name} {metric}: {value} -> {current} ({kind})")
    if problems:
        raise QualificationError("; ".join(problems))


# --------------------------------------------------------------- temperatures


def temps_problems(log: Path, buses: Sequence[str]) -> list[str]:
    """Missing, non-numeric or absent-per-bus tempwatch samples."""
    text = _read_text(log)
    if text is None:
        return [f"{log}: no temperature log"]
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        match = TEMPS_LINE.match(line.strip())
        if not match:
            continue
        values: dict[str, str] = {}
        for field in match.group(2).split():
            bus, _, value = field.partition("=")
            values[bus] = value
        rows.append(values)
    if len(rows) < 2:
        return [f"{log}: {len(rows)} numeric sample line(s)"]
    problems = []
    for bus in buses:
        samples = []
        for row in rows:
            value = row.get(bus)
            if value is None:
                continue
            try:
                samples.append(float(value))
            except ValueError:
                problems.append(f"{bus}: non-numeric sample {value!r}")
        finite = [sample for sample in samples if math.isfinite(sample)]
        if len(finite) != len(samples):
            problems.append(f"{bus}: non-finite sample")
        elif len(samples) < 2:
            problems.append(f"{bus}: {len(samples)} sample(s)")
        elif max(samples) >= 95:
            problems.append(f"{bus}: sample {max(samples):.0f} C >= 95 without abort")
    last = rows[-1]
    problems += [
        f"{bus}: absent from the last sample" for bus in buses if bus not in last
    ]
    return problems


def temps_max(log: Path) -> float | None:
    values = [
        float(value)
        for line in (_read_text(log) or "").splitlines()
        for value in re.findall(r"=(-?\d+(?:\.\d+)?)", line)
    ]
    return max(values) if values else None


# ---------------------------------------------------------- plan intervals


@dataclass(frozen=True)
class PlanSpan:
    """A started-and-committed plan on the shared monotonic clock."""

    start: float
    end: float


@dataclass(frozen=True)
class PlanCoverage:
    """Committed spans covering each cell; unfinished plan numbers are excluded."""

    hits: dict[str, list[int]]
    unfinished: tuple[int, ...]


def _finite(value: Any, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise QualificationError(f"{where}: missing or non-numeric") from None
    if not math.isfinite(number):
        raise QualificationError(f"{where}: not finite ({value!r})")
    return number


def _plan_clock(line: str) -> float:
    match = MONOTONIC.search(line)
    if match is None:
        raise QualificationError(
            f"EPLB plan line has no monotonic= clock: {line.strip()}"
        )
    return _finite(match.group(1), f"EPLB plan line monotonic clock: {line.strip()}")


def plan_spans(text: str) -> tuple[dict[int, PlanSpan], tuple[int, ...]]:
    """Committed plan spans keyed by plan number, plus unfinished plan numbers.

    Plan lines are grouped by number, so interleaved worker logs need no
    cross-rank ordering.  A plan with several starts uses the earliest start;
    several commits use the latest commit.  A start without a commit is an
    unfinished tail: it is excluded from coverage and reported, not an error.
    """
    starts: dict[int, tuple[float, int]] = {}
    dones: dict[int, float] = {}
    for line in text.splitlines():
        if not PLAN_LINE.search(line):
            continue
        stamp = _plan_clock(line)
        start = PLAN_START.search(line)
        done = PLAN_DONE.search(line)
        if start is not None:
            seq = int(start.group(1))
            step = int(start.group(2))
            previous = starts.get(seq)
            if previous is None:
                starts[seq] = (stamp, step)
            elif previous[1] != step:
                raise QualificationError(
                    f"plan {seq} starts at two steps ({previous[1]} and {step})"
                )
            else:
                starts[seq] = (min(stamp, previous[0]), step)
        elif done is not None:
            seq = int(done.group(1))
            dones[seq] = max(dones.get(seq, stamp), stamp)
        else:
            raise QualificationError(f"unrecognised EPLB plan line: {line.strip()}")
    unknown = sorted(set(dones) - set(starts))
    if unknown:
        raise QualificationError(
            "plan(s) committed without a start event: " + ", ".join(map(str, unknown))
        )
    spans = {}
    for seq, (start, _step) in starts.items():
        if seq not in dones:
            continue
        if dones[seq] < start:
            raise QualificationError(f"plan {seq} commits before it starts")
        spans[seq] = PlanSpan(start, dones[seq])
    unfinished = tuple(sorted(set(starts) - set(dones)))
    return spans, unfinished


def measured_window(boot_dir: Path, name: str) -> PlanSpan:
    """The native cell window from its result clock and native duration."""
    path = boot_dir / f"{name}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise QualificationError(f"{path}: unreadable ({exc})") from exc
    start = _finite(
        data.get("benchmark_start_time_monotonic"),
        f"{path}: benchmark_start_time_monotonic",
    )
    duration = _finite(data.get("duration"), f"{path}: duration")
    if not 0 < duration < 86400:
        raise QualificationError(f"{path}: duration {duration} is not a cell")
    end = start + duration
    if not math.isfinite(end):
        raise QualificationError(f"{path}: measured window end is not finite")
    return PlanSpan(start, end)


def check_plan_overlap(boot_dir: Path, cells: str) -> PlanCoverage:
    """Every native cell must overlap a started-and-committed monotonic span."""
    log = _required(boot_dir / "serve.log")
    spans, unfinished = plan_spans(log)
    if not spans:
        tail = (
            " (unfinished plan(s) "
            + ", ".join(map(str, unfinished))
            + " have no commit)"
            if unfinished
            else ""
        )
        raise QualificationError(
            "no started-and-committed EPLB plan in serve.log" + tail
        )
    hits = {}
    for cell in report_module().parse_cells(cells):
        window = measured_window(boot_dir, cell.name)
        covered = sorted(
            seq
            for seq, span in spans.items()
            if span.end > span.start
            and span.start < window.end
            and span.end > window.start
        )
        if not covered:
            tail = (
                "; unfinished plan(s) "
                + ", ".join(map(str, unfinished))
                + " are excluded"
                if unfinished
                else ""
            )
            raise QualificationError(
                f"cell {cell.name}: no started-and-committed EPLB plan "
                f"overlaps measured window [{window.start:.3f}, "
                f"{window.end:.3f}] s{tail}"
            )
        hits[cell.name] = covered
    return PlanCoverage(hits, unfinished)


# ------------------------------------------------------------ summary gates


def _required(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise QualificationError(f"{path}: unreadable ({exc})") from exc


def summary_problems(summary: str, cells: str, require_dmesg: bool = True) -> list[str]:
    """The queryable gates serve_validate.sh writes, including unknowns."""
    report = report_module()
    log = report.parse_summary(summary)
    problems = []
    if log.ready_s is None:
        problems.append("no READY line")
    if log.boot_failed:
        problems.append("BOOT FAILED")
    if log.died_in:
        problems.append(f"server died during {log.died_in}")
    if log.reset_failed:
        problems.append(f"prefix reset failed before {log.reset_failed}")
    if not log.finished:
        problems.append("summary has no done line")
    for name in ("PROBE", "PPL PROBE", "PREFIX", "CONCURRENCY", "PROBE-AFTER-LOAD"):
        verdict = (log.marks or {}).get(name)
        if verdict != "PASS":
            problems.append(f"{name} is {verdict or 'missing'}, not PASS")
    if log.fault_markers is None:
        problems.append("summary fault-marker count is unknown")
    elif log.fault_markers:
        problems.append(f"summary reports {log.fault_markers} fault marker(s)")
    if require_dmesg and log.dmesg_faults != "0":
        problems.append(f"new amdgpu dmesg faults is {log.dmesg_faults}, not 0")
    if log.evicted_since_ready != "ok":
        problems.append(
            "KFD eviction telemetry since READY is "
            f"{log.evicted_since_ready or 'missing'}, not ok"
        )
    expected = {cell.name: cell.n for cell in report.parse_cells(cells)}
    seen = log.cells or {}
    if set(seen) != set(expected):
        problems.append(
            "cells differ: measured "
            + (", ".join(sorted(seen)) or "none")
            + "; expected "
            + ", ".join(sorted(expected))
        )
    for name, number in expected.items():
        entry = seen.get(name)
        if entry is None:
            continue
        if entry.n != number:
            problems.append(f"{name}: {entry.n} requests, expected {number}")
        if entry.evicted != "ok":
            problems.append(f"{name}: KFD eviction is {entry.evicted or 'missing'}")
    return problems


def cell_problems(boot_dir: Path, cells: str) -> list[str]:
    """Raw bench results: native request counts, complete requests, ITL present."""
    problems = []
    expected = {cell.name: cell.n for cell in report_module().parse_cells(cells)}
    found = {path.stem for path in boot_dir.glob("bench-*.json")}
    extra = sorted(found - set(expected))
    if extra:
        problems.append("unexpected bench result(s): " + ", ".join(extra))
    for name, number in expected.items():
        path = boot_dir / f"{name}.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: no readable result ({exc})")
            continue
        try:
            prompts = int(data.get("num_prompts"))
        except (TypeError, ValueError):
            problems.append(f"{name}: no num_prompts")
            prompts = -1
        if prompts != number:
            problems.append(f"{name}: {prompts} prompts, expected {number}")
        try:
            completed = int(data.get("completed"))
            failed = int(data.get("failed") or 0)
        except (TypeError, ValueError):
            problems.append(f"{name}: no completed/failed count")
            continue
        if failed:
            problems.append(f"{name}: {failed} failed request(s)")
        if completed != number:
            problems.append(f"{name}: {completed} of {number} requests completed")
        for metric in ("median_itl_ms", "p99_itl_ms"):
            value = data.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{name}: {metric} is missing")
    return problems


def _worker_rank(line: str) -> int | None:
    match = WORKER_TP.search(line)
    return int(match.group(1)) if match else None


def log_evidence_problems(text: str, on: bool) -> tuple[list[str], dict[str, Any]]:
    """Mono/K2/EPLB markers of one boot, with the rank attribution kept."""
    problems = []
    mono_ranks: set[int] = set()
    k2_ranks: set[int] = set()
    module_ranks: set[int] = set()
    eplb_on = 0
    counters = 0
    configs: list[str] = []
    off_warnings: list[str] = []
    plan_starts = 0
    for line in text.splitlines():
        match = MONO_RANK.search(line)
        if match is not None:
            mono_ranks.add(int(match.group(1)))
        if "Online EPLB: off" in line:
            off_warnings.append(line.strip())
        rows = K2_ROW.search(line)
        if rows is not None:
            count, mode, path = int(rows.group(1)), rows.group(2), rows.group(3)
            rank = _worker_rank(line)
            if rank is None:
                problems.append(
                    f"K2 {count}-row event has no worker TP rank: {line.strip()}"
                )
            elif count == K2_FULL_ROWS and mode == "full" and path == "mono kernel":
                k2_ranks.add(rank)
            elif count >= MODULE_MIN_ROWS and path == "modules":
                module_ranks.add(rank)
        if EPLB_ON in line:
            eplb_on += 1
            configs.append(line.strip())
        if EPLB_COUNTERS in line:
            counters += 1
        if PLAN_START.search(line):
            plan_starts += 1
    if mono_ranks != set(range(4)):
        problems.append(f"Mono ranks are {sorted(mono_ranks)}, expected TP ranks 0-3")
    if k2_ranks != set(range(4)):
        problems.append(
            "mapped 3-row FULL K2 (-> mono kernel) ranks are "
            f"{sorted(k2_ranks)}, expected TP ranks 0-3"
        )
    if module_ranks != set(range(4)):
        problems.append(
            f">= {MODULE_MIN_ROWS}-row modules (c=8 path) ranks are "
            f"{sorted(module_ranks)}, expected TP ranks 0-3"
        )
    if on:
        if eplb_on != 4:
            problems.append(f"'{EPLB_ON}' appears {eplb_on} time(s), expected 4")
        if counters != 4:
            problems.append(f"'{EPLB_COUNTERS}' appears {counters} time(s), expected 4")
        if off_warnings:
            problems.append(f"EPLB turned itself off: {off_warnings[0]}")
        for config in configs:
            if "chaos=8" not in config:
                problems.append(f"EPLB config does not show CHAOS=8: {config}")
                break
        for config in configs:
            if "decision_steps=512" not in config:
                problems.append(
                    f"EPLB config does not show DECISION_STEPS=512: {config}"
                )
                break
    else:
        if eplb_on or counters:
            problems.append("off arm logged EPLB is ON or its load counters")
        if "Online EPLB" in text:
            problems.append("off arm logged an online-EPLB line")
    evidence = {
        "mono_ranks": sorted(mono_ranks),
        "k2_3row_full_ranks": sorted(k2_ranks),
        "modules_24row_ranks": sorted(module_ranks),
        "eplb_on_lines": eplb_on,
        "counter_lines": counters,
        "plan_starts": plan_starts,
    }
    return problems, evidence


# ------------------------------------------------------------- guard files


def guard_problems(boot_dir: Path) -> list[str]:
    """The per-boot guard telemetry: GPU map, AER, temps, tempwatch, faults."""
    problems = []
    try:
        info = json.loads((boot_dir / "gpus.json").read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"gpus.json: unreadable ({exc})")
    else:
        problems += [f"gpus.json: {item}" for item in gpu_problems(info)]
    try:
        guard = json.loads((boot_dir / "guard.json").read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"guard.json: unreadable ({exc})")
    else:
        if guard.get("ok") is not True:
            problems.append(f"guard.json: ok is {guard.get('ok')!r}")
        if guard.get("buses") != " ".join(EXPECTED_BUSES):
            problems.append(f"guard.json: buses are {guard.get('buses')!r}")
        if guard.get("faults") != 0:
            problems.append(f"guard.json: faults are {guard.get('faults')!r}")
        if guard.get("validator_rc") != 0:
            problems.append(
                f"guard.json: validator rc is {guard.get('validator_rc')!r}"
            )
    abort = boot_dir / "THERMAL_ABORT"
    if abort.exists():
        problems.append(f"thermal abort: {_read_text(abort) or 'file present'}")
    problems += temps_problems(boot_dir / "temps-gpus.log", EXPECTED_BUSES)
    try:
        before = json.loads((boot_dir / "aer-before.json").read_text())
        after = json.loads((boot_dir / "aer-after.json").read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"AER snapshots: unreadable ({exc})")
    else:
        try:
            compare_aer(before, after)
        except QualificationError as exc:
            problems.append(f"AER: {exc}")
    return problems


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")


# ------------------------------------------------------------- boot command


def validate_boot(
    boot_dir: Path, cells: str, arm: str
) -> tuple[list[str], dict[str, Any]]:
    """All r2 checks of one arm directory, as a list of consumer failures."""
    problems = []
    summary = _required(boot_dir / "summary.txt")
    serve = _required(boot_dir / "serve.log")
    on = arm.startswith("B")
    problems += summary_problems(summary, cells)
    try:
        invalid = gate_module().invalid_cells(str(boot_dir))
    except gate_module().GateInputError as exc:
        problems.append(f"INVALID window: {exc}")
        invalid = set()
    if invalid:
        problems.append("INVALID cell(s): " + ", ".join(sorted(invalid)))
    problems += cell_problems(boot_dir, cells)
    log_problems, evidence = log_evidence_problems(serve, on)
    problems += log_problems
    hits: dict[str, list[int]] = {}
    unfinished: tuple[int, ...] = ()
    if on:
        try:
            coverage = check_plan_overlap(boot_dir, cells)
        except QualificationError as exc:
            problems.append(str(exc))
        else:
            hits = coverage.hits
            unfinished = coverage.unfinished
    else:
        try:
            spans, tail = plan_spans(serve)
            if spans or tail:
                problems.append("off arm logged an EPLB plan")
        except QualificationError as exc:
            problems.append(f"off arm plan evidence: {exc}")
    problems += guard_problems(boot_dir)
    try:
        faults = fault_lines(boot_dir / "serve.log")
    except QualificationError as exc:
        problems.append(str(exc))
        faults = []
    if faults:
        problems.append(f"{len(faults)} serve.log fault line(s): {faults[0].strip()}")
    evidence["plan_overlap"] = hits
    evidence["unfinished_plans"] = list(unfinished)
    evidence["on"] = on
    evidence["cells"] = cells
    evidence["faults"] = len(faults)
    evidence["ok"] = not problems
    evidence["problems"] = problems
    return problems, evidence


def cmd_boot(args: argparse.Namespace) -> int:
    boot_dir = Path(args.dir)
    try:
        problems, evidence = validate_boot(boot_dir, args.cells, args.arm)
    except QualificationError as exc:
        problems = [str(exc)]
        evidence = {
            "on": args.arm.startswith("B"),
            "cells": args.cells,
            "plan_overlap": {},
            "unfinished_plans": [],
            "faults": None,
            "ok": False,
            "problems": problems,
        }
    evidence["head_sha"] = args.head_sha
    _write_json(Path(args.out) if args.out else boot_dir / "evidence.json", evidence)
    if problems:
        print(f"boot {args.arm}: NOT VALID ({len(problems)} problem(s))")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"boot {args.arm}: VALID ({'on' if evidence['on'] else 'off'}, "
        f"{len(evidence['plan_overlap'])} cell(s) with a live plan, "
        f"{evidence['faults']} fault line(s))"
    )
    return 0


# ---------------------------------------------------------------- r1 receipt


def _cells_spec(spec: str) -> str:
    return " ".join(cell.spec for cell in report_module().parse_cells(spec))


def _speed_regressions(receipt: dict[str, Any]) -> list[str]:
    """Count native cell verdicts, never Markdown headings or explanatory text."""
    speed = receipt.get("speed")
    if (
        not isinstance(speed, dict)
        or not isinstance(speed.get("cells"), list)
        or not speed["cells"]
    ):
        raise QualificationError("receipt.json: speed cell verdicts are missing")
    names = []
    for cell in speed["cells"]:
        if (
            not isinstance(cell, dict)
            or not isinstance(cell.get("name"), str)
            or cell.get("verdict") not in {"win", "regression", "none", "invalid"}
        ):
            raise QualificationError("receipt.json: malformed speed cell verdict")
        if cell["verdict"] == "regression":
            names.append(cell["name"])
    return names


def cmd_r1(args: argparse.Namespace) -> int:
    run = Path(args.run)
    problems = []
    try:
        manifest = json.loads((run / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"r1: cannot read {run / 'manifest.json'}: {exc}")
        return 1
    settings = manifest.get("settings") or {}
    report = report_module()
    expected = {
        "mode": "abba",
        "exact": "1",
        "stall": "1",
        "gpus": "2,3,4,5",
        "recipe": "flashnext-mtp2",
        "cells": _cells_spec(report.DEFAULT_CELLS),
        "exact_cells": _cells_spec(report.DEFAULT_EXACT_CELLS),
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "tools_sha": args.tools_sha,
        "base_tree": args.base_tree,
        "head_tree": args.head_tree,
    }
    for key, value in expected.items():
        if str(settings.get(key)) != value:
            problems.append(f"manifest {key}={settings.get(key)!r}, expected {value!r}")
    if args.receipt_rc != 0:
        problems.append(f"pr_receipt.sh exit {args.receipt_rc}, not 0")
    if settings.get("base_rocm_c") == settings.get("head_rocm_c"):
        problems.append("base and head record the same extension binary")
    try:
        boot_records = {rec["id"]: rec for rec in manifest.get("boots") or []}
    except (KeyError, TypeError) as exc:
        print(f"r1: malformed boots list: {exc}")
        return 1
    if set(boot_records) != {boot[0] for boot in R1_BOOTS}:
        problems.append(
            "manifest boots are "
            + (", ".join(sorted(boot_records)) or "none")
            + "; expected "
            + ", ".join(boot[0] for boot in R1_BOOTS)
        )
    try:
        boots = {boot.id: boot for boot in report.load_boots(run, manifest)}
    except (KeyError, TypeError) as exc:
        print(f"r1: malformed boot record: {exc}")
        return 1
    boot_summary: dict[str, Any] = {}
    for boot_id, role, kind in R1_BOOTS:
        record = boot_records.get(boot_id)
        if record is None:
            continue
        if record.get("role") != role or record.get("kind") != kind:
            problems.append(
                f"boot {boot_id}: {record.get('role')}/{record.get('kind')}, "
                f"expected {role}/{kind}"
            )
        boot = boots.get(boot_id)
        if boot is None:
            problems.append(f"boot {boot_id}: no directory in the manifest")
            continue
        boot_problems = []
        if boot.rc != 0:
            boot_problems.append(f"rc {boot.rc}, not 0")
        try:
            summary_text = _required(boot.path / "summary.txt")
        except QualificationError as exc:
            boot_problems.append(str(exc))
            summary_text = ""
        wanted = report.DEFAULT_EXACT_CELLS if kind == "exact" else report.DEFAULT_CELLS
        boot_problems += summary_problems(summary_text, wanted, require_dmesg=False)
        log = boot.log
        if kind == "speed" and log.stall != "PASS":
            boot_problems.append(f"stall gate is {log.stall or 'missing'}, not PASS")
        try:
            serve = _required(boot.path / "serve.log")
        except QualificationError as exc:
            boot_problems.append(str(exc))
            serve = ""
        if EPLB_ON in serve or EPLB_COUNTERS in serve:
            boot_problems.append("flag-off boot logged online EPLB")
        try:
            faults = fault_lines(boot.path / "serve.log")
        except QualificationError as exc:
            boot_problems.append(str(exc))
            faults = []
        if faults:
            boot_problems.append(
                f"{len(faults)} serve.log fault line(s): {faults[0].strip()}"
            )
        if "INVALID" in summary_text:
            boot_problems.append("summary.txt contains INVALID")
        boot_problems += guard_problems(boot.path)
        seen = log.cells or {}
        problems += [f"boot {boot_id}: {item}" for item in boot_problems]
        boot_summary[boot_id] = {
            "role": role,
            "kind": kind,
            "rc": boot.rc,
            "faults": len(faults),
            "cells": len(seen),
            "temps_max": temps_max(boot.path / "temps-gpus.log"),
            "ok": not boot_problems,
        }
    receipt = {}
    try:
        receipt = json.loads((run / "receipt.json").read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"receipt.json: unreadable ({exc})")
    gates = receipt.get("gates") or {}
    for gate in R1_GATES:
        verdict = gates.get(gate)
        if verdict not in GATE_OK:
            problems.append(f"receipt gate {gate} is {verdict or 'missing'}")
    if receipt and receipt.get("ok") is not True:
        problems.append("receipt.json says ok is not true")
    try:
        labels = _speed_regressions(receipt)
    except QualificationError as exc:
        problems.append(str(exc))
        labels = []
    (run / "speed-labels.txt").write_text(
        "\n".join(labels) + "\n" if labels else "0 REGRESSION labels\n"
    )
    payload = {
        "mode": "r1",
        "result": "not qualified" if problems else "qualified",
        "base_tree": args.base_tree,
        "head_tree": args.head_tree,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "tools_sha": args.tools_sha,
        "receipt_rc": args.receipt_rc,
        "gates": gates,
        "boots": boot_summary,
        "speed_labels": labels,
        "problems": problems,
    }
    _write_json(run / "qualification.json", payload)
    text = [
        "# EPLB final qualification: r1",
        "",
        f"result: {payload['result']}",
        f"base: {args.base_tree} ({args.base_sha})",
        f"head: {args.head_tree} ({args.head_sha})",
        f"tools: {args.tools_sha}",
        f"receipt exit: {args.receipt_rc}",
        "",
        "| boot | role | kind | rc | faults | cells | max C | ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for boot_id, _role, _kind in R1_BOOTS:
        item = boot_summary.get(boot_id)
        if item is None:
            text.append(f"| {boot_id} | - | - | - | - | - | - | missing |")
            continue
        text.append(
            f"| {boot_id} | {item['role']} | {item['kind']} | {item['rc']} | "
            f"{item['faults']} | {item['cells']} | "
            f"{item['temps_max'] if item['temps_max'] is not None else '-'} | "
            f"{'yes' if item['ok'] else 'no'} |"
        )
    text += [
        "",
        "Receipt gates: "
        + ", ".join(f"{name}={gates.get(name, 'missing')}" for name in R1_GATES),
        "",
        "Speed labels are reported, never gated:",
    ]
    text += [f"- {label}" for label in labels] or ["- none"]
    if problems:
        text += ["", "Problems:"]
        text += [f"- {problem}" for problem in problems]
    (run / "qualification.md").write_text("\n".join(text) + "\n")
    if problems:
        print(f"r1: NOT QUALIFIED ({len(problems)} problem(s))")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"r1: QUALIFIED; {len(labels)} REGRESSION label(s) reported, not gated")
    return 0


# ---------------------------------------------------------------- r2 assembly


def gate_row_problems(rows: Sequence[dict[str, Any]], cells: str) -> list[str]:
    """The eight gated rows, exactly once, all pass; context rows are ignored."""
    gated_names = tuple(gate_module().GATED)
    expected = {
        (cell.name.removeprefix("bench-"), metric)
        for cell in report_module().parse_cells(cells)
        for metric in gated_names
    }
    problems = []
    seen = set()
    for row in rows:
        metric = row.get("metric")
        if metric not in gated_names:
            continue
        key = (row.get("cell"), metric)
        if key in seen:
            problems.append(f"duplicate gated row {key[0]} {key[1]}")
        seen.add(key)
        if key not in expected:
            problems.append(f"unexpected gated row {key[0]} {key[1]}")
        if row.get("verdict") != "pass":
            problems.append(
                f"{key[0]} {key[1]}: verdict {row.get('verdict')!r}, not pass"
            )
    missing = sorted(expected - seen)
    problems += [f"missing gated row {cell} {metric}" for cell, metric in missing]
    return problems


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        rows = json.loads(Path(args.json).read_text())
    except (OSError, ValueError) as exc:
        print(f"gate: cannot read {args.json}: {exc}")
        return 1
    if not isinstance(rows, list):
        print("gate: the itl_gate JSON is not a row list")
        return 1
    problems = gate_row_problems(rows, args.cells)
    if problems:
        print(f"gate: REFUSED ({len(problems)} problem(s))")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    gated = sum(1 for row in rows if row.get("metric") in tuple(gate_module().GATED))
    print(f"gate: PASS ({gated}/{gated} gated rows, default 0.5% envelope)")
    return 0


def cmd_r2(args: argparse.Namespace) -> int:
    run = Path(args.run)
    problems = []
    arms: dict[str, Any] = {}
    for arm in args.arms:
        try:
            evidence = json.loads((run / "boots" / arm / "evidence.json").read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"arm {arm}: no readable evidence.json ({exc})")
            arms[arm] = {"ok": False}
            continue
        arms[arm] = {
            key: evidence.get(key)
            for key in (
                "on",
                "ok",
                "cells",
                "mono_ranks",
                "k2_3row_full_ranks",
                "modules_24row_ranks",
                "eplb_on_lines",
                "counter_lines",
                "plan_starts",
                "plan_overlap",
                "unfinished_plans",
                "faults",
                "problems",
            )
        }
        if args.head_sha and evidence.get("head_sha") != args.head_sha:
            problems.append(
                f"arm {arm}: head {evidence.get('head_sha')!r}, "
                f"expected {args.head_sha!r}"
            )
        if evidence.get("ok") is not True:
            problems.append(f"arm {arm}: evidence is not ok")
            for item in evidence.get("problems") or []:
                problems.append(f"arm {arm}: {item}")
    try:
        rows = json.loads(Path(args.gate_json).read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"itl_gate JSON: unreadable ({exc})")
        rows = []
    if isinstance(rows, list):
        problems += gate_row_problems(rows, args.cells)
    else:
        problems.append("itl_gate JSON is not a row list")
    payload = {
        "mode": "r2",
        "result": "not qualified" if problems else "qualified",
        "arms": arms,
        "gate_rows": len(rows) if isinstance(rows, list) else 0,
        "injection": "not run",
        "notes": [R2_INJECTION_NOTE],
        "problems": problems,
    }
    _write_json(run / "qualification.json", payload)
    text = [
        "# EPLB final qualification: r2",
        "",
        f"result: {payload['result']}",
        "design: A1 off, B1 chaos, B2 chaos, A2 off; `--monokernel`; "
        "cells " + args.cells,
        "",
        "| arm | EPLB | mono | K2 3-row FULL | modules >=24 | plan cells | "
        "unfinished | faults | ok |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in args.arms:
        item = arms.get(arm) or {}
        text.append(
            f"| {arm} | {'on' if item.get('on') else 'off'} | "
            f"{len(item.get('mono_ranks') or [])} | "
            f"{len(item.get('k2_3row_full_ranks') or [])} | "
            f"{len(item.get('modules_24row_ranks') or [])} | "
            f"{len(item.get('plan_overlap') or {})} | "
            f"{len(item.get('unfinished_plans') or [])} | "
            f"{item.get('faults', '?')} | "
            f"{'yes' if item.get('ok') else 'no'} |"
        )
    text += [
        "",
        "K2 evidence is the c=1/probe 3-row FULL path; the c=8 module path is "
        "the >=24-row evidence. They are never merged.",
        "",
        R2_INJECTION_NOTE,
    ]
    if problems:
        text += ["", "Problems:"]
        text += [f"- {problem}" for problem in problems]
    (run / "qualification.md").write_text("\n".join(text) + "\n")
    if problems:
        print(f"r2: NOT QUALIFIED ({len(problems)} problem(s))")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("r2: QUALIFIED (four arms, all eight gated ITL rows pass)")
    return 0


# ------------------------------------------------------------------ commands


def cmd_gpus(args: argparse.Namespace) -> int:
    try:
        buses = resolve_gpus(Path(args.venv), Path(args.out))
    except QualificationError as exc:
        print(f"gpus: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(buses)
    return 0


def cmd_aer(args: argparse.Namespace) -> int:
    buses = args.buses.split()
    try:
        snapshot = aer_snapshot(Path(args.root), buses)
    except QualificationError as exc:
        print(f"aer: REFUSED: {exc}", file=sys.stderr)
        return 1
    _write_json(Path(args.out), snapshot)
    print(f"aer: {len(snapshot['devices'])} device(s) snapshotted")
    return 0


def cmd_aer_compare(args: argparse.Namespace) -> int:
    try:
        before = json.loads(Path(args.before).read_text())
        after = json.loads(Path(args.after).read_text())
        compare_aer(before, after)
    except (OSError, ValueError) as exc:
        print(f"aer-compare: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"aer-compare: unchanged ({len(before['devices'])} device(s))")
    return 0


def cmd_faults(args: argparse.Namespace) -> int:
    try:
        lines = fault_lines(Path(args.log))
    except QualificationError as exc:
        print(exc, file=sys.stderr)
        return 3
    if lines:
        print("\n".join(lines))
        return 1
    return 0


def cmd_temps(args: argparse.Namespace) -> int:
    log = Path(args.log)
    buses = args.buses.split()
    problems = temps_problems(log, buses)
    if args.abort and Path(args.abort).exists():
        note = _read_text(Path(args.abort)) or "present"
        problems.insert(0, f"thermal abort: {note}")
    if problems:
        print("temps: REFUSED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"temps: OK (4 slot buses, max {temps_max(log):.0f} C)")
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    payload = {
        "ok": args.ok == "1",
        "buses": args.buses,
        "faults": args.faults,
        "note": args.note,
        "validator_rc": args.validator_rc,
        "thermal": bool(args.abort and Path(args.abort).exists()),
    }
    _write_json(Path(args.out), payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gpus", help="resolve HIP 2,3,4,5 in a separate torch process")
    p.add_argument("--venv", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("aer", help="snapshot endpoint and ancestor AER counters")
    p.add_argument("--root", default="/sys")
    p.add_argument("--buses", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_aer)

    p = sub.add_parser("aer-compare", help="refuse a changed or missing counter")
    p.add_argument("before")
    p.add_argument("after")
    p.set_defaults(func=cmd_aer_compare)

    p = sub.add_parser("faults", help="shared serve.log fault phrases")
    p.add_argument("--log", required=True)
    p.set_defaults(func=cmd_faults)

    p = sub.add_parser("temps", help="slot temperature telemetry")
    p.add_argument("--log", required=True)
    p.add_argument("--buses", required=True)
    p.add_argument("--abort", default="")
    p.set_defaults(func=cmd_temps)

    p = sub.add_parser("guard", help="record the per-boot guard verdict")
    p.add_argument("--out", required=True)
    p.add_argument("--ok", choices=("0", "1"), required=True)
    p.add_argument("--buses", required=True)
    p.add_argument("--faults", type=int, default=0)
    p.add_argument("--validator-rc", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--abort", default="")
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("boot", help="validate one r2 arm directory")
    p.add_argument("--dir", required=True)
    p.add_argument("--arm", required=True, choices=R2_ARMS)
    p.add_argument("--cells", required=True)
    p.add_argument("--head-sha", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_boot)

    p = sub.add_parser("gate", help="require the eight gated itl_gate rows")
    p.add_argument("--json", required=True)
    p.add_argument("--cells", required=True)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("r1", help="validate the pr_receipt.sh abba receipt")
    p.add_argument("--run", required=True)
    p.add_argument("--base-tree", required=True)
    p.add_argument("--head-tree", required=True)
    p.add_argument("--base-sha", required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--tools-sha", required=True)
    p.add_argument("--receipt-rc", type=int, required=True)
    p.set_defaults(func=cmd_r1)

    p = sub.add_parser("r2", help="assemble the four-arm r2 qualification")
    p.add_argument("--run", required=True)
    p.add_argument("--arms", nargs="+", default=list(R2_ARMS))
    p.add_argument("--head-sha", default="")
    p.add_argument("--gate-json", required=True)
    p.add_argument("--cells", required=True)
    p.set_defaults(func=cmd_r2)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except QualificationError as exc:
        print(f"qualify_boot: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
