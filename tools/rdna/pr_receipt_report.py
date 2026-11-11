#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Receipt tables and verdicts for tools/rdna/pr_receipt.sh.

The driver boots the base tree and the head tree through serve_validate.sh and
records what it ran in ``RUN_DIR/manifest.json``; this script turns the boot
directories into ``receipt.md`` and ``receipt.json``.

    pr_receipt_report.py cells [--exact] [SPEC]  default (or a checked) cell list
    pr_receipt_report.py set RUN_DIR KEY=VALUE ...      run-wide settings
    pr_receipt_report.py boot RUN_DIR ID ROLE KIND DIR [KEY=VALUE ...]
    pr_receipt_report.py env --gpus 2,3,4,5 [--out FILE] [--buses]
    pr_receipt_report.py plan-diff BASE_PLAN HEAD_PLAN [--mask FROM=TO ...]
    pr_receipt_report.py plan-field PLAN compilation
    pr_receipt_report.py report RUN_DIR

A boot has a role (``base`` or ``head``) and a kind: ``exact`` (deterministic
MoE, token hashes), ``speed`` (production mode: bench cells, quality
collects) or ``floor`` (a second base boot that only feeds the quality floor).

``report`` exits 0 when every gate that was run passed, 1 when one failed or
could not be judged. Speed verdicts are reported, not gated. Standard library
and ``regex`` only; run it with the python of a built venv.
"""

import argparse
import difflib
import hashlib
import importlib.util
import json
import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import regex as re

HERE = Path(__file__).resolve().parent

DEFAULT_CELLS = (
    "1024:512:1 1024:512:4 1024:512:8 "
    "16384:1024:1 16384:1024:4 16384:1024:8 "
    "32768:2048:1 32768:2048:4"
)
DEFAULT_EXACT_CELLS = "1024:128:1:4 16384:128:1:2"
MAX_PPL_REL = 0.02


def load_sibling(name: str) -> ModuleType:
    """Import tools/rdna/port_v031/<name>.py (a script, not a package)."""
    path = HERE / "port_v031" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- cells


def short_tokens(n: int) -> str:
    """1024 -> 1k, 512 -> 512."""
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


@dataclass(frozen=True)
class Cell:
    """One bench cell, ``in:out:conc[:prompts]`` as serve_validate.sh reads it."""

    in_len: int
    out_len: int
    conc: int
    prompts: int | None = None

    @property
    def name(self) -> str:
        """Result name serve_validate.sh gives the cell (``bench-<name>.json``)."""
        return f"bench-{self.in_len}x{self.out_len}-c{self.conc}"

    @property
    def n(self) -> int:
        """Requests in the cell (serve_validate.sh: 4 at c=1, else max(4c, 8))."""
        if self.prompts is not None:
            return self.prompts
        return 4 if self.conc == 1 else max(self.conc * 4, 8)

    @property
    def label(self) -> str:
        return f"{short_tokens(self.in_len)}/{short_tokens(self.out_len)} c={self.conc}"

    @property
    def spec(self) -> str:
        head = f"{self.in_len}:{self.out_len}:{self.conc}"
        return head if self.prompts is None else f"{head}:{self.prompts}"


def parse_cells(spec: str) -> list[Cell]:
    """Parse a CELLS list; ``none`` and the empty string are no cells."""
    spec = spec.strip()
    if spec in ("", "none"):
        return []
    cells: list[Cell] = []
    seen: set[str] = set()
    for token in spec.split():
        parts = token.split(":")
        if not 3 <= len(parts) <= 4:
            raise ValueError(f"cell {token!r}: want in:out:conc[:prompts]")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"cell {token!r}: fields must be integers") from None
        if min(nums) < 1:
            raise ValueError(f"cell {token!r}: fields must be positive")
        cell = Cell(*nums)
        if cell.name in seen:
            raise ValueError(
                f"cell {token!r}: {cell.name} is already in the list "
                "(serve_validate.sh names a result by shape and concurrency)"
            )
        seen.add(cell.name)
        cells.append(cell)
    return cells


# ------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    key: str  # key of bench_metrics.metrics()
    title: str
    unit: str
    scale: float  # stored value / scale is what the tables print
    digits: int
    lower_is_better: bool


METRICS = (
    Metric("ttft_p50_ms", "TTFT p50", "s", 1000.0, 2, True),
    Metric("ttft_p99_ms", "TTFT p99", "s", 1000.0, 2, True),
    Metric("prefill_req_tps", "prefill per request", "tok/s", 1.0, 1, False),
    Metric("prefill_agg_tps", "prefill aggregate", "tok/s", 1.0, 1, False),
    Metric("itl_p50_ms", "ITL p50", "ms", 1.0, 1, True),
    Metric("itl_p99_ms", "ITL p99", "ms", 1.0, 1, True),
    Metric("decode_req_tps", "decode per request", "tok/s", 1.0, 2, False),
    Metric("out_tps", "output aggregate", "tok/s", 1.0, 2, False),
)


def abba_verdict(
    base: Sequence[float], head: Sequence[float], lower_is_better: bool
) -> str:
    """``regression`` only if every head run is worse than every base run.

    ``win`` only if every head run beats every base run; anything else,
    including a tie, is ``none`` (no measurable difference).
    """
    if not base or not head:
        return "n/a"
    if lower_is_better:
        worse = min(head) > max(base)
        better = max(head) < min(base)
    else:
        worse = max(head) < min(base)
        better = min(head) > max(base)
    if worse:
        return "regression"
    if better:
        return "win"
    return "none"


def ab_direction(base: float, head: float, lower_is_better: bool) -> str:
    """One run each: ``worse``, ``better`` or ``equal``. Not a verdict."""
    if head == base:
        return "equal"
    return "better" if (head < base) == lower_is_better else "worse"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def delta_pct(base: Sequence[float], head: Sequence[float]) -> float | None:
    """Head mean against base mean, in percent (None when the base mean is 0)."""
    if not base or not head or mean(base) == 0:
        return None
    return (mean(head) - mean(base)) / mean(base) * 100.0


def roll_up(per_metric: dict[str, str]) -> str:
    """ABBA verdict of a cell: a metric that regressed makes it a regression."""
    values = set(per_metric.values())
    if "regression" in values:
        return "regression"
    if "win" in values:
        return "win"
    return "none"


def roll_up_ab(per_metric: dict[str, str]) -> str:
    """Single base/head run of a cell: worse, better or equal (not a verdict)."""
    values = set(per_metric.values())
    if "worse" in values:
        return "worse"
    if "better" in values:
        return "better"
    return "equal"


# ------------------------------------------------------------- summary.txt

LOG_LINE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (.*)$")
EVICT_LINE = re.compile(r"^\s+evicted_ms\b.*->\s*(ok|INVALID)\b")
EVICT_NONE = re.compile(r"^\s+evicted_ms: no vLLM workers")
MARK_LINE = re.compile(
    r"^(PROBE|PPL PROBE|PREFIX|CONCURRENCY|PROBE-AFTER-LOAD) (PASS|FAIL)\b"
)
BENCH_LINE = re.compile(r"^bench (\S+) \(n=(\d+)\)")


@dataclass
class CellLog:
    name: str
    n: int
    start: str
    evicted: str | None = None  # ok | INVALID | unchecked | None (no line)


@dataclass
class BootLog:
    """What serve_validate.sh wrote to summary.txt, as far as the receipt needs."""

    ready_s: int | None = None
    boot_failed: bool = False
    finished: bool = False
    died_in: str | None = None
    reset_failed: str | None = None
    marks: dict[str, str] | None = None
    stall: str | None = None
    hook: str | None = None  # "rc=0 PASS ..." of the LOAD_HOOK line
    cells: dict[str, CellLog] | None = None
    evicted_since_ready: str | None = None
    fault_markers: int | None = None
    dmesg_faults: str | None = None
    markers: list[str] | None = None

    def mark(self, key: str) -> str | None:
        return (self.marks or {}).get(key)


def parse_summary(text: str) -> BootLog:
    """Parse summary.txt; unknown lines are ignored."""
    log = BootLog(marks={}, cells={}, markers=[])
    assert log.marks is not None and log.cells is not None
    assert log.markers is not None
    current: CellLog | None = None
    in_markers = False
    for raw in text.splitlines():
        m = LOG_LINE.match(raw)
        if m is None:
            if in_markers and raw.startswith("    "):
                log.markers.append(raw.strip())
            elif current is not None:
                ev = EVICT_LINE.match(raw)
                if ev and current.evicted is None:
                    current.evicted = ev.group(1)
                elif EVICT_NONE.match(raw) and current.evicted is None:
                    current.evicted = "unchecked"
            continue
        stamp, msg = m.groups()
        current = None
        in_markers = False
        mark = MARK_LINE.match(msg)
        bench = BENCH_LINE.match(msg)
        if mark:
            log.marks[mark.group(1)] = mark.group(2)
        elif bench:
            current = CellLog(bench.group(1), int(bench.group(2)), stamp)
            log.cells[current.name] = current
        elif msg.startswith("READY in "):
            log.ready_s = int(msg.split()[2].rstrip("s"))
        elif msg.startswith("BOOT FAILED"):
            log.boot_failed = True
        elif msg.startswith("SERVER DIED during "):
            log.died_in = msg.rsplit(" ", 1)[1]
        elif msg.startswith("PREFIX RESET FAILED before "):
            log.reset_failed = msg.rsplit(" ", 1)[1]
        elif msg.startswith("STALL PROBE "):
            log.stall = msg.split()[2]
        elif msg.startswith("LOAD HOOK rc="):
            log.hook = msg.removeprefix("LOAD HOOK ")
        elif msg.startswith("evicted_ms since READY:"):
            verdict = re.search(r"->\s*(ok|INVALID)\b", msg)
            log.evicted_since_ready = verdict.group(1) if verdict else "unchecked"
        elif msg.startswith("fault markers in serve.log:"):
            log.fault_markers = int(msg.rsplit(" ", 1)[1])
        elif msg.startswith("new amdgpu dmesg faults:"):
            log.dmesg_faults = msg.rsplit(" ", 1)[1]
        elif msg == "markers:":
            in_markers = True
        elif msg == "done":
            log.finished = True
    return log


# ------------------------------------------------------------ boot results


@dataclass
class CellResult:
    cell: Cell
    status: str  # ok | invalid | errors | missing
    detail: str = ""
    values: dict[str, float] | None = None
    accept_len: float | None = None
    accept_source: str = ""


def hms_plus(start: str, seconds: float) -> str:
    stamp = datetime.strptime(start, "%H:%M:%S") + timedelta(seconds=seconds)
    return stamp.strftime("%H:%M:%S")


def acceptance_from_log(serve_log: Path, start: str, duration_s: float) -> float | None:
    """Acceptance length of one cell from the server's SpecDecoding lines."""
    if not serve_log.is_file():
        return None
    spec = load_sibling("spec_acceptance")
    records = spec.parse_log(serve_log.read_text(errors="replace").splitlines())
    got = spec.acceptance(records, start, hms_plus(start, duration_s + 5))
    length = got["length"]
    return None if length != length else float(length)


def load_cell(bootdir: Path, cell: Cell, log: BootLog) -> CellResult:
    """The numbers of one cell of one boot and whether they may be used."""
    entry = (log.cells or {}).get(cell.name)
    path = bootdir / f"{cell.name}.json"
    if not path.is_file():
        why = "server died" if log.died_in == cell.name else "no result file"
        if log.reset_failed == cell.name:
            why = "prefix-cache reset failed"
        return CellResult(cell, "missing", why)
    try:
        data = json.loads(path.read_text())
        values = load_sibling("bench_metrics").metrics(data)
    except (OSError, ValueError) as err:
        return CellResult(cell, "missing", f"unreadable result ({err})")
    result = CellResult(cell, "ok", values=values)
    length = data.get("spec_decode_acceptance_length")
    if length is not None:
        result.accept_len, result.accept_source = float(length), "bench"
    elif entry is not None:
        got = acceptance_from_log(
            bootdir / "serve.log", entry.start, float(data.get("duration") or 0.0)
        )
        if got is not None:
            result.accept_len, result.accept_source = got, "server log"
    if entry is not None and entry.evicted == "INVALID":
        result.status = "invalid"
        result.detail = "KFD evicted a worker's GPU queues"
    elif int(data.get("failed") or 0) or int(data.get("completed") or 0) < cell.n:
        result.status = "errors"
        result.detail = (
            f"{int(data.get('completed') or 0)} of {cell.n} requests completed"
        )
    return result


@dataclass
class Boot:
    id: str
    role: str  # base | head
    kind: str  # exact | speed | floor
    path: Path
    fields: dict[str, str]
    log: BootLog

    @property
    def rc(self) -> int | None:
        rc = self.fields.get("rc")
        return int(rc) if rc not in (None, "") else None


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def load_boots(run: Path, manifest: dict) -> list[Boot]:
    boots = []
    for rec in manifest.get("boots", []):
        path = run / rec["dir"]
        fields = {k: str(v) for k, v in rec.items()}
        boots.append(
            Boot(
                rec["id"],
                rec["role"],
                rec["kind"],
                path,
                fields,
                parse_summary(read_text(path / "summary.txt")),
            )
        )
    return boots


# ----------------------------------------------------------------- speed


def judge_speed(
    boots: Sequence[Boot], cells: Sequence[Cell], mode: str
) -> dict[str, object]:
    """Per-cell, per-metric verdicts of the speed boots (see the module doc)."""
    speed = [b for b in boots if b.kind == "speed"]
    out_cells = []
    counts = {"win": 0, "regression": 0, "none": 0, "invalid": 0}
    counts_ab = {"worse": 0, "better": 0, "equal": 0, "invalid": 0}
    for cell in cells:
        runs = {b.id: load_cell(b.path, cell, b.log) for b in speed}
        roles = {b.id: b.role for b in speed}
        bad = {k: r for k, r in runs.items() if r.status != "ok"}
        entry: dict[str, object] = {
            "name": cell.name,
            "label": cell.label,
            "n": cell.n,
            "runs": {
                k: {
                    "role": roles[k],
                    "status": r.status,
                    "detail": r.detail,
                    "values": r.values,
                    "accept_len": r.accept_len,
                    "accept_source": r.accept_source,
                }
                for k, r in runs.items()
            },
        }
        base_ids = [k for k in runs if roles[k] == "base"]
        head_ids = [k for k in runs if roles[k] == "head"]
        if bad or not base_ids or not head_ids:
            entry["verdict"] = "invalid"
            entry["invalid"] = {k: f"{r.status}: {r.detail}" for k, r in bad.items()}
            counts["invalid"] += 1
            counts_ab["invalid"] += 1
            out_cells.append(entry)
            continue
        metrics: dict[str, dict[str, object]] = {}
        verdicts: dict[str, str] = {}
        directions: dict[str, str] = {}
        for metric in METRICS:
            base = [runs[k].values[metric.key] for k in base_ids]  # type: ignore[index]
            head = [runs[k].values[metric.key] for k in head_ids]  # type: ignore[index]
            verdict = abba_verdict(base, head, metric.lower_is_better)
            direction = ab_direction(mean(base), mean(head), metric.lower_is_better)
            metrics[metric.key] = {
                "base": base,
                "head": head,
                "delta_pct": delta_pct(base, head),
                "verdict": verdict,
                "direction": direction,
            }
            verdicts[metric.key] = verdict
            directions[metric.key] = direction
        entry["metrics"] = metrics
        if mode == "abba":
            entry["verdict"] = roll_up(verdicts)
            entry["flagged"] = [
                k for k, v in verdicts.items() if v in ("win", "regression")
            ]
            counts[str(entry["verdict"])] += 1
        else:
            entry["verdict"] = roll_up_ab(directions)
            entry["flagged"] = [
                k for k, v in directions.items() if v in ("worse", "better")
            ]
            counts_ab[str(entry["verdict"])] += 1
        out_cells.append(entry)
    return {
        "mode": mode,
        "boots": {b.id: b.role for b in speed},
        "cells": out_cells,
        "counts": counts if mode == "abba" else counts_ab,
    }


# --------------------------------------------------------------- exactness

RESULT_LINE = re.compile(
    r"RESULT (PASS|FAIL): (\d+) failing, (\d+) near-tie, (\d+) soft, "
    r"(\d+) solo-noise, (\d+) identical of (\d+) checks"
)
SAVED_KEYS = ("solo", "solo2", "long_solo")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def exact_hashes(bootdir: Path) -> dict[str, str]:
    """Digests of the greedy outputs a deterministic boot saved.

    ``solo/<i>``, ``solo2/<i>``, ``long_solo/<i>`` hash the token ids of the
    concurrency probe's requests; ``<cell>/<i>`` hashes the text of request i
    of a bench cell run with ``--save-detailed``.
    """
    out: dict[str, str] = {}
    saved = bootdir / "concurrency.json"
    if saved.is_file():
        data = json.loads(saved.read_text())
        for key in SAVED_KEYS:
            for idx, rec in sorted(
                data.get(key, {}).items(), key=lambda kv: int(kv[0])
            ):
                out[f"{key}/{idx}"] = digest("\n".join(rec["tokens"]))
    for path in sorted(bootdir.glob("bench-*.json")):
        texts = json.loads(path.read_text()).get("generated_texts") or []
        for idx, text in enumerate(texts):
            out[f"{path.stem}/{idx}"] = digest(text)
    return out


def probe_result(bootdir: Path) -> dict[str, object]:
    """The RESULT line of the concurrency probe of one boot."""
    text = read_text(bootdir / "concurrency.txt")
    m = RESULT_LINE.search(text)
    if m is None:
        if "RESULT FAIL" in text:
            return {"result": "FAIL", "failing": None, "note": "solo requests fail"}
        return {"result": "missing"}
    names = ("failing", "near_tie", "soft", "noise", "identical", "checks")
    got = dict(zip(names, (int(x) for x in m.groups()[1:])))
    return {"result": m.group(1), **got}


def judge_exact(boots: Sequence[Boot]) -> dict[str, object]:
    """Base against head on the deterministic boots: hashes and the probe."""
    base = next((b for b in boots if b.kind == "exact" and b.role == "base"), None)
    head = next((b for b in boots if b.kind == "exact" and b.role == "head"), None)
    if base is None or head is None:
        return {"ran": False, "verdict": "NOT RUN"}
    hb, hh = exact_hashes(base.path), exact_hashes(head.path)
    keys = sorted(set(hb) | set(hh))
    differ = [k for k in keys if k in hb and k in hh and hb[k] != hh[k]]
    missing = [k for k in keys if k not in hb or k not in hh]

    def unstable(hashes: dict[str, str]) -> list[str]:
        return [
            k
            for k in hashes
            if k.startswith("solo/")
            and hashes[k] != hashes.get(k.replace("solo/", "solo2/", 1), hashes[k])
        ]

    flaky = {"base": unstable(hb), "head": unstable(hh)}
    probes = {"base": probe_result(base.path), "head": probe_result(head.path)}
    probe_ok = probes["head"].get("result") == "PASS"
    if not keys or not probe_ok:
        verdict = "FAIL"
    elif flaky["base"] or flaky["head"]:
        verdict = "INCONCLUSIVE"
    elif differ or missing:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {
        "ran": True,
        "verdict": verdict,
        "boots": {"base": base.id, "head": head.id},
        "compared": len(keys) - len(missing),
        "equal": len(keys) - len(missing) - len(differ),
        "differ": differ,
        "missing": missing,
        "unstable": flaky,
        "probe": probes,
    }


# ----------------------------------------------------------------- quality

TOTAL_LINE = re.compile(
    r"TOTAL PPL ([\d.]+) -> ([\d.]+) \(([+-][\d.]+)%\).*?"
    r"greedy exact (\d+)/(\d+).*?(PASS|FAIL)\s*$",
    re.MULTILINE,
)


def parse_ppl_total(text: str) -> dict[str, object] | None:
    """The TOTAL line of ``quality_ab.py compare``."""
    m = TOTAL_LINE.search(text)
    if m is None:
        return None
    return {
        "ppl_ref": float(m.group(1)),
        "ppl_cand": float(m.group(2)),
        "rel_pct": float(m.group(3)),
        "greedy_exact": int(m.group(4)),
        "passages": int(m.group(5)),
        "verdict": m.group(6),
    }


def within_floor(head: dict, floor: dict) -> tuple[bool, bool]:
    """Head KL and top-1 agreement against the base-vs-base floor.

    KL must not exceed the upper end of the floor's bootstrap CI; top-1
    agreement must not fall below the lower end of the floor's CI.
    """
    return head["kl"] <= floor["kl_ci"][1], head["top1"] >= floor["top1_ci"][0]


def judge_quality(run: Path, boots: Sequence[Boot], settings: dict) -> dict:
    """KL floor, PPL rise, probes and greedy checks of the quality boots."""
    if str(settings.get("quality", "0")) != "1":
        return {"ran": False, "verdict": "NOT RUN"}
    checks: list[dict[str, object]] = []
    kl_path = run / "quality" / "kl.json"
    domains: dict[str, dict[str, object]] = {}
    if kl_path.is_file():
        pairs = json.loads(kl_path.read_text())["pairs"]
        floor = pairs.get("floor")
        heads = {k: v for k, v in pairs.items() if k != "floor"}
        for name in (floor or {}).get("domains", {}):
            f = floor["domains"][name]
            row: dict[str, object] = {"floor": f, "heads": {}}
            for label, pair in heads.items():
                h = pair["domains"][name]
                kl_ok, top1_ok = within_floor(h, f)
                row["heads"][label] = {**h, "kl_ok": kl_ok, "top1_ok": top1_ok}  # type: ignore[index]
                checks.append(
                    {
                        "name": f"{name}: {label} KL within floor",
                        "ok": kl_ok,
                        "detail": f"{h['kl']:.5f} <= {f['kl_ci'][1]:.5f}",
                    }
                )
                checks.append(
                    {
                        "name": f"{name}: {label} top-1 within floor",
                        "ok": top1_ok,
                        "detail": f"{h['top1']:.4f} >= {f['top1_ci'][0]:.4f}",
                    }
                )
            domains[name] = row
        if not heads or floor is None:
            checks.append({"name": "KL pairs", "ok": False, "detail": "no floor pair"})
    else:
        checks.append({"name": "KL compare", "ok": False, "detail": "kl.json missing"})
    ppl: dict[str, object] = {}
    for path in sorted((run / "quality").glob("ppl-*.txt")):
        label = path.stem.removeprefix("ppl-")
        total = parse_ppl_total(read_text(path))
        ppl[label] = total
        if label == "floor":
            continue  # shown, not gated: the fixed bound applies to head
        ok = total is not None and total["verdict"] == "PASS"
        limit = float(settings.get("max_ppl_rel") or MAX_PPL_REL) * 100
        detail = (
            "no TOTAL line"
            if total is None
            else f"PPL {total['rel_pct']:+.3f} % (limit {limit:.1f} %)"
        )
        checks.append({"name": f"PPL rise, {label}", "ok": ok, "detail": detail})
    if not any(k != "floor" for k in ppl):
        checks.append({"name": "PPL compare", "ok": False, "detail": "not run"})
    heads_boots = [b for b in boots if b.role == "head" and b.kind == "speed"]
    for boot in heads_boots:
        probes = probe_result(boot.path)
        failing = probes.get("failing")
        checks.append(
            {
                "name": f"concurrency probe, {boot.id}",
                "ok": probes.get("result") == "PASS",
                "detail": f"{probes.get('result')}, {failing} failing",
            }
        )
        for key in ("PROBE", "PROBE-AFTER-LOAD"):
            got = boot.log.mark(key)
            checks.append(
                {
                    "name": f"greedy {key.lower()}, {boot.id}",
                    "ok": got == "PASS",
                    "detail": got or "missing",
                }
            )
    if not heads_boots:
        checks.append({"name": "head boots", "ok": False, "detail": "none"})
    for boot in boots:
        if boot.kind in ("speed", "floor"):
            got = boot.log.hook or ""
            if "rc=0" not in got:
                checks.append(
                    {
                        "name": f"quality collect, {boot.id}",
                        "ok": False,
                        "detail": got or "no load-hook line",
                    }
                )
    verdict = "PASS" if checks and all(c["ok"] for c in checks) else "FAIL"
    return {
        "ran": True,
        "verdict": verdict,
        "checks": checks,
        "domains": domains,
        "ppl": ppl,
    }


# ------------------------------------------------------------------ report


def esc(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(esc(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_value(metric: Metric, value: float | None) -> str:
    return "-" if value is None else f"{value / metric.scale:.{metric.digits}f}"


def fmt_delta(pct: float | None) -> str:
    return "-" if pct is None else f"{pct:+.1f} %"


def mask_numbers(line: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", line)


def speed_markdown(speed: dict) -> str:
    """The speed tables: one row per cell, then every number of every boot."""
    cells = speed["cells"]
    if not cells:
        return "No speed cells were run."
    abba = speed["mode"] == "abba"
    marks = {"win": " (win)", "regression": " (regression)"} if abba else {}
    head_row = ["cell", "requests"] + [m.title for m in METRICS] + ["verdict"]
    rows = []
    for cell in cells:
        row: list[object] = [cell["label"], cell["n"]]
        metrics = cell.get("metrics")
        for metric in METRICS:
            if metrics is None:
                row.append("-")
                continue
            m = metrics[metric.key]
            row.append(fmt_delta(m["delta_pct"]) + marks.get(m["verdict"], ""))
        row.append(verdict_text(cell, abba))
        rows.append(row)
    text = [md_table(head_row, rows)]
    text.append(
        "\nEach figure is the head mean against the base mean "
        "(for TTFT and ITL a negative number is faster)."
        + (
            " `win` and `REGRESSION` apply the ABBA rule: the verdict needs "
            "both head runs on the same side of both base runs."
            if abba
            else " One base run and one head run: this is not a verdict; "
            "rerun with `MODE=abba` to claim a win or to confirm a worse cell."
        )
    )
    ids = list(speed["boots"])
    detail_head = ["cell", "metric"] + [f"{i} {speed['boots'][i]}" for i in ids]
    detail_head += ["head vs base", "result"]
    detail_rows = []
    for cell in cells:
        for metric in METRICS:
            row = [cell["label"], f"{metric.title} ({metric.unit})"]
            for i in ids:
                run = cell["runs"].get(i)
                if run is None or run["status"] != "ok":
                    status = "missing" if run is None else run["status"].upper()
                    row.append(status)
                else:
                    row.append(fmt_value(metric, run["values"][metric.key]))
            m = (cell.get("metrics") or {}).get(metric.key)
            if m is None:
                row += ["-", "-"]
            else:
                shown = m["verdict"] if abba else m["direction"]
                row += [fmt_delta(m["delta_pct"]), shown if shown != "none" else ""]
            detail_rows.append(row)
    text.append(
        "\n<details><summary>Every figure of every boot</summary>\n\n"
        + md_table(detail_head, detail_rows)
        + "\n\n</details>"
    )
    invalid = [c for c in cells if c["verdict"] == "invalid"]
    if invalid:
        lines = [
            f"- {c['label']}: " + "; ".join(f"{k} {v}" for k, v in c["invalid"].items())
            if c.get("invalid")
            else f"- {c['label']}: roles missing"
            for c in invalid
        ]
        again = " ".join(
            f"{c['name'].split('-', 1)[1].replace('x', ':').replace('-c', ':')}"
            for c in invalid
        )
        text.append(
            "\nInvalid cells (no verdict; rerun them, for example with "
            f'`CELLS="{again}"`):\n' + "\n".join(lines)
        )
    return "\n".join(text)


def verdict_text(cell: dict, abba: bool) -> str:
    v = cell["verdict"]
    if v == "invalid":
        return "INVALID"
    if abba:
        return {
            "regression": "REGRESSION",
            "win": "win",
            "none": "no measurable difference",
        }[v]
    return {
        "worse": "worse; ABBA needed",
        "better": "better; ABBA needed to claim a win",
        "equal": "equal",
    }[v]


def accept_markdown(speed: dict) -> str | None:
    ids = list(speed["boots"])
    rows = []
    for cell in speed["cells"]:
        vals = [cell["runs"].get(i, {}).get("accept_len") for i in ids]
        if all(v is None for v in vals):
            continue
        sources = {
            cell["runs"][i]["accept_source"]
            for i in ids
            if cell["runs"].get(i, {}).get("accept_len") is not None
        }
        rows.append(
            [cell["label"]]
            + ["-" if v is None else f"{v:.3f}" for v in vals]
            + [", ".join(sorted(sources))]
        )
    if not rows:
        return None
    head = ["cell"] + [f"{i} {speed['boots'][i]}" for i in ids] + ["source"]
    return md_table(head, rows)


def exact_markdown(exact: dict) -> str:
    if not exact["ran"]:
        return "Not run."
    probe = exact["probe"]

    def probe_cell(p: dict) -> str:
        if p.get("result") == "missing":
            return "no result"
        if p.get("failing") is None:
            return f"{p['result']} ({p.get('note', '')})"
        return (
            f"{p['result']}: {p['failing']} failing, {p['near_tie']} near-tie, "
            f"{p['identical']} identical of {p['checks']}"
        )

    rows = [
        ["greedy outputs hashed", f"{exact['compared']} compared"],
        ["equal base vs head", f"{exact['equal']} of {exact['compared']}"],
        ["not in both boots", ", ".join(exact["missing"][:6]) or "none"],
        ["differing", ", ".join(exact["differ"][:6]) or "none"],
        [
            "repeat of a solo request differs (base, head)",
            f"{len(exact['unstable']['base'])}, {len(exact['unstable']['head'])}",
        ],
        ["concurrency probe, base", probe_cell(probe["base"])],
        ["concurrency probe, head", probe_cell(probe["head"])],
    ]
    return md_table(["check", "result"], rows)


def quality_markdown(quality: dict) -> str:
    if not quality["ran"]:
        return "Not run."
    out = []
    rows = []
    for name, row in quality["domains"].items():
        f = row["floor"]
        rows.append(
            [
                "floor (base vs base)",
                name,
                f["windows"],
                f"{f['kl']:.5f} [{f['kl_ci'][0]:.5f}, {f['kl_ci'][1]:.5f}]",
                f"{f['top1']:.4f} [{f['top1_ci'][0]:.4f}, {f['top1_ci'][1]:.4f}]",
                f"{f['ppl_ref']:.4f} -> {f['ppl_cand']:.4f}",
                "-",
            ]
        )
        for label, h in row["heads"].items():
            ok = "yes" if h["kl_ok"] and h["top1_ok"] else "NO"
            rows.append(
                [
                    f"{label} (base vs head)",
                    name,
                    h["windows"],
                    f"{h['kl']:.5f} [{h['kl_ci'][0]:.5f}, {h['kl_ci'][1]:.5f}]",
                    f"{h['top1']:.4f} [{h['top1_ci'][0]:.4f}, {h['top1_ci'][1]:.4f}]",
                    f"{h['ppl_ref']:.4f} -> {h['ppl_cand']:.4f}",
                    ok,
                ]
            )
    if rows:
        out.append(
            md_table(
                [
                    "pair",
                    "domain",
                    "windows",
                    "KL nats/token [95 % CI]",
                    "top-1 agreement [95 % CI]",
                    "PPL ref -> cand",
                    "within floor",
                ],
                rows,
            )
        )
    ppl_rows = []
    for label, total in quality["ppl"].items():
        if total is None:
            ppl_rows.append([label, "no TOTAL line", "", "", ""])
            continue
        shown = "floor, not gated" if label == "floor" else total["verdict"]
        ppl_rows.append(
            [
                label,
                f"{total['ppl_ref']:.4f} -> {total['ppl_cand']:.4f}",
                f"{total['rel_pct']:+.3f} %",
                f"{total['greedy_exact']}/{total['passages']}",
                shown,
            ]
        )
    if ppl_rows:
        out.append(
            md_table(
                ["pair", "long-context PPL", "rise", "greedy exact passages", "result"],
                ppl_rows,
            )
        )
    out.append(
        md_table(
            ["check", "result", ""],
            [
                [c["name"], "pass" if c["ok"] else "FAIL", c["detail"]]
                for c in quality["checks"]
            ],
        )
    )
    return "\n\n".join(out)


def health_markdown(boots: Sequence[Boot]) -> str:
    rows = []
    for b in boots:
        log = b.log
        seen = {
            k: log.mark(k)
            for k in ("PROBE", "PPL PROBE", "PREFIX", "CONCURRENCY", "PROBE-AFTER-LOAD")
            if log.mark(k) is not None
        }
        failed = [k.lower() for k, v in seen.items() if v != "PASS"]
        probes = f"FAIL: {', '.join(failed)}" if failed else f"all {len(seen)} pass"
        state = (
            "failed to boot"
            if log.boot_failed
            else ("complete" if log.finished else "incomplete")
        )
        rows.append(
            [
                b.id,
                b.role,
                b.kind,
                "-" if log.ready_s is None else f"{log.ready_s} s",
                state,
                probes if seen else "-",
                log.evicted_since_ready or "-",
                "-" if log.fault_markers is None else log.fault_markers,
                log.dmesg_faults or "-",
                "-" if b.rc is None else b.rc,
            ]
        )
    return md_table(
        [
            "boot",
            "role",
            "kind",
            "READY",
            "summary",
            "probes",
            "KFD evictions since READY",
            "fault markers",
            "dmesg faults",
            "exit",
        ],
        rows,
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def environment_markdown(settings: dict, run: Path) -> str:
    env = {r: read_json(run / f"env-{r}.json") for r in ("base", "head")}

    def both(key: str, label: str) -> list[str]:
        return [
            label,
            f"`{settings.get(f'base_{key}', '-')}`",
            f"`{settings.get(f'head_{key}', '-')}`",
        ]

    rows = [
        both("tree", "tree"),
        both("sha", "git SHA"),
        both("rocm_c", "_rocm_C.abi3.so md5"),
        both("venv", "venv"),
        [
            "torch / HIP",
            *(
                f"`{env[r].get('torch', '-')}` / `{env[r].get('hip', '-')}`"
                for r in ("base", "head")
            ),
        ],
    ]
    text = [md_table(["", "base", "head"], rows)]
    gpus = env["head"].get("gpus") or env["base"].get("gpus") or []
    gpu_rows = [
        [
            g.get("hip"),
            g.get("name"),
            g.get("gfx"),
            g.get("bus"),
            f"{g.get('mem_gib', 0):.1f} GiB",
        ]
        for g in gpus
    ]
    shared = [
        ["recipe", f"`{settings.get('recipe', '-')}`"],
        ["recipe overrides", f"`{settings.get('overrides') or 'none'}`"],
        ["model", f"`{settings.get('model', '-')}`"],
        ["model revision", settings.get("model_rev", "-")],
        ["model config.json sha256", settings.get("model_cfg_sha", "-")],
        ["mode", f"`MODE={settings.get('mode', '-')}`"],
        [
            "host / kernel",
            f"{settings.get('host', '-')} / {settings.get('kernel', '-')}",
        ],
        ["date (UTC)", settings.get("date", "-")],
        ["compilation config", f"`{settings.get('compilation') or 'not read'}`"],
        [
            "tools tree",
            f"`{settings.get('tools_tree', '-')}` @ {settings.get('tools_sha', '-')}",
        ],
        ["speed cells", f"`{settings.get('cells') or 'none'}`"],
        ["exactness cells", f"`{settings.get('exact_cells') or 'none'}`"],
    ]
    if str(settings.get("quality", "0")) == "1":
        quality_windows = (
            f"`{settings.get('quality_windows', '-')}`, "
            f"{settings.get('quality_limit', '-')} per domain, "
            f"ids sha256 {settings.get('quality_windows_sha', '-')}"
        )
        shared.append(["quality windows", quality_windows])
    text.append(md_table(["", ""], shared))
    if gpu_rows:
        text.append(md_table(["HIP", "GPU", "gfx", "PCI bus", "memory"], gpu_rows))
    return "\n\n".join(text)


def reach_markdown(settings: dict, boots: Sequence[Boot], run: Path) -> str:
    """Change class, platforms and the paths the servers reported selecting."""
    gpus = read_json(run / "env-head.json").get("gpus", [])
    arches = sorted({g["gfx"] for g in gpus if g.get("gfx")})
    ran = f"{', '.join(arches) or 'unknown'} ({len(gpus)} GPUs)"
    cls = settings.get("change_class") or "not stated"
    not_run = settings.get("not_run") or "not stated"
    reached = settings.get("reached") or "not stated"
    plan = settings.get("plan_diff") or "not compared"
    lines = [
        f"- Change class: {cls}",
        f"- Platforms run: {ran}",
        f"- Platforms not run: {not_run}",
        f"- Paths and callers reached, as the author states them: {reached}",
        f"- Launch plans of the two trees (tree paths masked): {plan}",
    ]

    def marker_set(role: str) -> set[str]:
        seen: set[str] = set()
        for b in boots:
            if b.role == role and b.log.markers:
                seen |= {mask_numbers(m.split(" ", 1)[-1]) for m in b.log.markers}
        return seen

    base_m, head_m = marker_set("base"), marker_set("head")
    if head_m:
        only_b, only_h = sorted(base_m - head_m), sorted(head_m - base_m)
        same = "the same for base and head"
        differ = f"{len(only_b)} only in base, {len(only_h)} only in head"
        lines.append(
            "- Paths the servers selected (marker lines of serve.log, numbers "
            f"masked): {differ if only_b or only_h else same}"
        )
        for tag, only in (("base", only_b), ("head", only_h)):
            lines += [f"  - only in {tag}: `{m}`" for m in only[:8]]
    return "\n".join(lines)


def settings_of(manifest: dict) -> dict:
    return manifest.get("settings", {})


def build_receipt(run: Path) -> dict:
    """Everything the receipt says, as data."""
    manifest = json.loads((run / "manifest.json").read_text())
    settings = settings_of(manifest)
    boots = load_boots(run, manifest)
    cells = parse_cells(settings.get("cells", ""))
    mode = settings.get("mode", "ab")
    speed = judge_speed(boots, cells, mode) if cells else None
    exact = (
        judge_exact(boots)
        if str(settings.get("exact", "1")) == "1"
        else {"ran": False, "verdict": "NOT RUN"}
    )
    quality = judge_quality(run, boots, settings)
    boot_bad = [b.id for b in boots if b.rc not in (None, 0) or b.log.boot_failed]
    gates = {"exact": exact["verdict"], "quality": quality["verdict"]}
    ok = (
        all(v in ("PASS", "NOT RUN") for v in gates.values())
        and not boot_bad
        and (exact["ran"] or str(settings.get("exact", "1")) != "1")
    )
    return {
        "settings": settings,
        "boots": [
            {"id": b.id, "role": b.role, "kind": b.kind, "rc": b.rc} for b in boots
        ],
        "exact": exact,
        "quality": quality,
        "speed": speed,
        "gates": gates,
        "boot_failures": boot_bad,
        "ok": ok,
        "_boots": boots,
    }


EXACT_INTRO = (
    "Greedy output hashes, base against head, with "
    "`VLLM_RDNA2_MOE_DETERMINISTIC=1`; concurrent replies against their solo "
    "runs (`concurrency_probe.py`)."
)
SPEED_INTRO = (
    "Production mode of the recipe, greedy (`--temperature 0`), `--ignore-eos`, "
    "seed 12345, prefix cache reset before every cell, capture-ladder warm-up "
    "after boot; a KFD eviction invalidates a cell."
)


def quality_intro(settings: dict) -> str:
    limit = float(settings.get("max_ppl_rel") or MAX_PPL_REL) * 100
    return (
        "`topk_kl.py`: head against base, with base against a second base boot "
        "as the floor (head KL within the floor's upper CI bound, head top-1 "
        "agreement above the floor's lower CI bound); `quality_ab.py`: PPL rise "
        f"at most {limit:.0f} %; the concurrency probe and the greedy probes "
        "pass."
    )


def speed_counts(speed: dict | None) -> str:
    if speed is None:
        return "not run"
    c = speed["counts"]
    if speed["mode"] == "abba":
        return (
            f"{c['none']} cells no measurable difference, {c['win']} win, "
            f"{c['regression']} REGRESSION, {c['invalid']} invalid"
        )
    return (
        f"{c['equal']} equal, {c['better']} better, {c['worse']} worse, "
        f"{c['invalid']} invalid (one run each; not a verdict)"
    )


def raw_results(settings: dict, run: Path) -> str:
    where = settings.get("out", run)
    files = "`serve.log`, `summary.txt`, `bench-*.json`, `concurrency.json`"
    return (
        f"- Run directory: `{where}` on {settings.get('host', '-')}; per boot "
        f"directory (`boots/<id>/`): {files}, `topk.npz`, `quality.json`.\n"
        f"- Command: `{settings.get('command', '-')}`"
    )


def render(receipt: dict, run: Path) -> str:
    """receipt.md: the tables of the PR template, in its order."""
    settings = receipt["settings"]
    boots = receipt["_boots"]
    speed = receipt["speed"]
    gates = receipt["gates"]
    failed = receipt["boot_failures"]
    speed_body = SPEED_INTRO + "\n\n"
    if speed:
        speed_body += "Boots in order: " + ", ".join(
            f"{k} ({v})" for k, v in speed["boots"].items()
        )
        speed_body += ".\n\n" + speed_markdown(speed)
        accept = accept_markdown(speed)
        if accept:
            speed_body += "\n\n### Speculative acceptance length\n\n" + accept
    else:
        speed_body += "No speed cells were run."
    sections = [
        ("Environment", environment_markdown(settings, run)),
        ("Change class and reach", reach_markdown(settings, boots, run)),
        ("Exactness", EXACT_INTRO + "\n\n" + exact_markdown(receipt["exact"])),
        (
            "Quality",
            quality_intro(settings) + "\n\n" + quality_markdown(receipt["quality"]),
        ),
        ("Speed", speed_body),
        ("Boots", health_markdown(boots)),
        ("Raw results", raw_results(settings, run)),
    ]
    head = str(settings.get("head_sha", "?"))[:10]
    base = str(settings.get("base_sha", "?"))[:10]
    summary = md_table(
        ["gate", "result"],
        [
            ["token-exact", gates["exact"]],
            ["quality", gates["quality"]],
            ["speed", speed_counts(speed)],
            ["boots", "FAILED: " + ", ".join(failed) if failed else "all exited 0"],
        ],
    )
    out = [f"# RDNA receipt: {head} against {base}", "", summary]
    for title, body in sections:
        out += ["", f"## {title}", "", body]
    return "\n".join(out) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    run = Path(args.run)
    try:
        receipt = build_receipt(run)
    except (OSError, ValueError, KeyError) as err:
        print(f"cannot read {run}: {err}", file=sys.stderr)
        return 2
    text = render(receipt, run)
    (run / "receipt.md").write_text(text)
    plain = {k: v for k, v in receipt.items() if k != "_boots"}
    (run / "receipt.json").write_text(json.dumps(plain, indent=1, default=str))
    gates = receipt["gates"]
    speed = receipt["speed"]
    print(f"token-exact: {gates['exact']}")
    print(f"quality: {gates['quality']}")
    if speed:
        print(
            f"speed ({speed['mode']}): "
            + ", ".join(f"{v} {k}" for k, v in speed["counts"].items())
        )
    for b in receipt["boot_failures"]:
        print(f"boot {b}: FAILED")
    print(f"receipt: {run / 'receipt.md'}")
    return 0 if receipt["ok"] else 1


# --------------------------------------------------------------- run files


def manifest_path(run: str) -> Path:
    return Path(run) / "manifest.json"


def load_manifest(run: str) -> dict:
    path = manifest_path(run)
    if path.is_file():
        return json.loads(path.read_text())
    return {"settings": {}, "boots": []}


def split_pairs(pairs: Sequence[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"want KEY=VALUE, got {pair!r}")
        out[key] = value
    return out


def cmd_set(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run)
    manifest["settings"].update(split_pairs(args.pairs))
    Path(args.run).mkdir(parents=True, exist_ok=True)
    manifest_path(args.run).write_text(json.dumps(manifest, indent=1))
    return 0


def cmd_boot(args: argparse.Namespace) -> int:
    if args.role not in ("base", "head") or args.kind not in (
        "exact",
        "speed",
        "floor",
    ):
        raise SystemExit("role is base|head, kind is exact|speed|floor")
    manifest = load_manifest(args.run)
    rec = {
        "id": args.id,
        "role": args.role,
        "kind": args.kind,
        "dir": args.dir,
        **split_pairs(args.pairs),
    }
    manifest["boots"] = [b for b in manifest["boots"] if b["id"] != args.id]
    manifest["boots"].append(rec)
    Path(args.run).mkdir(parents=True, exist_ok=True)
    manifest_path(args.run).write_text(json.dumps(manifest, indent=1))
    return 0


def cmd_cells(args: argparse.Namespace) -> int:
    if args.spec is not None:
        spec = args.spec
    else:
        spec = DEFAULT_EXACT_CELLS if args.exact else DEFAULT_CELLS
    try:
        cells = parse_cells(spec)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    print(" ".join(c.spec for c in cells) or "none")
    return 0


def plan_diff(base: str, head: str, masks: Sequence[tuple[str, str]]) -> list[str]:
    """Lines that differ between two launch plans once tree paths are masked."""
    ordered = sorted((m for m in masks if m[0]), key=lambda m: -len(m[0]))
    sides = []
    for text in (base, head):
        for src, dst in ordered:
            text = text.replace(src, dst)
        sides.append(text.splitlines())
    diff = difflib.unified_diff(sides[0], sides[1], "base", "head", lineterm="", n=0)
    return [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---"))]


def cmd_plan_diff(args: argparse.Namespace) -> int:
    plans = [read_text(Path(p)) for p in (args.base, args.head)]
    if not all(plans):
        print("a launch plan is missing")
        return 0
    masks = [tuple(m.split("=", 1)) for m in args.mask if "=" in m]
    lines = plan_diff(plans[0], plans[1], masks)  # type: ignore[arg-type]
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n" if lines else "")
    print(f"{len(lines)} lines differ" if lines else "identical")
    return 0


def cmd_plan_field(args: argparse.Namespace) -> int:
    text = read_text(Path(args.plan))
    if args.field == "compilation":
        m = re.search(r"'--compilation-config'\s+'([^']*)'", text)
        print(m.group(1) if m else "")
    return 0


def collect_env(gpus: str) -> dict[str, object]:
    """Python, torch, ROCm and GPU facts; GPUs as torch numbers them."""
    info: dict[str, object] = {
        "python": platform.python_version(),
        "host": platform.node(),
        "kernel": platform.release(),
    }
    try:
        import torch
    except ImportError as err:
        info["torch_error"] = str(err)
        return info
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    hip_ids = [g for g in gpus.split(",") if g.strip()]
    devices = []
    for index, hip in enumerate(hip_ids):
        try:
            props = torch.cuda.get_device_properties(index)
        except (RuntimeError, AssertionError, IndexError):
            continue
        bus = str(getattr(props, "pci_bus_id", "") or "")
        devices.append(
            {
                "hip": int(hip),
                "name": props.name,
                "gfx": str(getattr(props, "gcnArchName", "")).split(":")[0],
                "bus": f"{int(bus):02x}" if bus.isdigit() else bus,
                "mem_gib": props.total_memory / 2**30,
            }
        )
    info["gpus"] = devices
    return info


def cmd_env(args: argparse.Namespace) -> int:
    info = collect_env(args.gpus)
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=1))
    if args.buses:
        print(" ".join(str(g["bus"]) for g in info.get("gpus", [])))  # type: ignore[attr-defined]
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells", help="print the default or a checked cell list")
    p.add_argument("spec", nargs="?")
    p.add_argument("--exact", action="store_true", help="the exactness cells")
    p.set_defaults(func=cmd_cells)
    p = sub.add_parser("set", help="record run-wide settings")
    p.add_argument("run")
    p.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    p.set_defaults(func=cmd_set)
    p = sub.add_parser("boot", help="record one boot")
    p.add_argument("run")
    p.add_argument("id")
    p.add_argument("role")
    p.add_argument("kind")
    p.add_argument("dir", help="boot directory relative to RUN_DIR")
    p.add_argument("pairs", nargs="*", metavar="KEY=VALUE")
    p.set_defaults(func=cmd_boot)
    p = sub.add_parser("env", help="GPU, torch and ROCm facts of this python")
    p.add_argument("--gpus", required=True, help="HIP numbers, as in GPUS")
    p.add_argument("--out")
    p.add_argument("--buses", action="store_true", help="print the PCI buses")
    p.set_defaults(func=cmd_env)
    p = sub.add_parser("plan-diff", help="compare two serve_rdna.sh PRINT=1 plans")
    p.add_argument("base")
    p.add_argument("head")
    p.add_argument("--mask", action="append", default=[], metavar="FROM=TO")
    p.add_argument("--out")
    p.set_defaults(func=cmd_plan_diff)
    p = sub.add_parser("plan-field", help="one value of a launch plan")
    p.add_argument("plan")
    p.add_argument("field", choices=["compilation"])
    p.set_defaults(func=cmd_plan_field)
    p = sub.add_parser("report", help="write receipt.md and receipt.json")
    p.add_argument("run")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
