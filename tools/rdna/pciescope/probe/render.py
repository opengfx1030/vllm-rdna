# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Human-readable report of a ``MeasureResult`` (the non-``--json`` output)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pciescope.model import Measurement

from .effectiveness import row_baseline
from .results import eff_row

if TYPE_CHECKING:
    from .api import MeasureResult

_SCENARIO_WIDTH = 64
_MAX_STEP_LINES = 40


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header).rstrip(), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*row).rstrip() for row in rows]
    return ["  " + line for line in lines]


def _clip(text: str, width: int = _SCENARIO_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "~"


def _value(m: Measurement, digits: int = 2) -> str:
    text = f"{m.value:.{digits}f}"
    n = int(m.stats.get("n", 1))
    if m.tags.get("reps") and n > 1 and "min" in m.stats and "max" in m.stats:
        text += f" ({m.stats['min']:.{digits}f}..{m.stats['max']:.{digits}f})"
    return text if m.ok else text + " INVALID"


def _audit(m: Measurement) -> str:
    row = eff_row(m)
    diff = None if row is None else row_baseline(row)
    if diff is None:
        return ""
    return f"{diff.verdict} ({diff.ratio:.2f}x of {diff.baseline:g})"


def _bandwidth_rows(ms: Sequence[Measurement]) -> list[list[str]]:
    rows: list[list[str]] = []
    for m in ms:
        eff = "" if m.efficiency is None else f"{m.efficiency:.0%}"
        theo = "" if not m.theoretical else f"{m.theoretical:.1f}"
        rows.append(
            [
                m.path_class,
                _clip(m.tags.get("scenario", m.id)),
                _value(m),
                theo,
                eff,
                _audit(m),
            ]
        )
    return rows


def _latency_rows(ms: Sequence[Measurement]) -> list[list[str]]:
    rows: list[list[str]] = []
    for m in ms:
        p99 = m.stats.get("p99")
        rows.append(
            [
                m.path_class,
                _clip(m.tags.get("scenario", m.id)),
                f"{m.size_bytes}",
                _value(m),
                "" if p99 is None else f"{p99:.2f}",
                _audit(m),
            ]
        )
    return rows


def _plan_lines(result: MeasureResult) -> list[str]:
    plan = result.plan
    if plan is None:
        return []
    out = [
        f"plan: {len(plan.steps)} steps, ~{plan.est_total_s:.0f} s, "
        f"repeats {plan.repeats}, HIP_VISIBLE_DEVICES={plan.hip_visible}",
    ]
    for bus, sw, arch in zip(plan.buses, plan.switch, plan.arch, strict=True):
        out.append(f"  gpu:{bus}  switch {sw}  {arch}")
    for step in plan.steps[:_MAX_STEP_LINES]:
        out.append(f"  {step.id:<20s} {_clip(step.scenario, 70)}")
        out.append(f"  {'':<20s} $ p2p_probe {step.cmd}")
    if len(plan.steps) > _MAX_STEP_LINES:
        out.append(f"  ... {len(plan.steps) - _MAX_STEP_LINES} more (use --json)")
    for what, why in plan.skipped:
        out.append(f"  skipped: {what}: {why}")
    return out


def render_text(result: MeasureResult) -> str:
    """Return the report of ``result`` as text."""
    gpus = " ".join(result.gpus) or "(no GPU)"
    lines = [
        f"pciescope measure {result.suite} on {gpus}  "
        f"(owner {result.owner or 'none'}, run {result.run_id})"
    ]
    if result.status == "refused":
        lines.append("REFUSED, nothing ran:")
        lines += [f"  {x}" for x in result.refusal.splitlines()]
        return "\n".join(lines)
    lines += _status_lines(result)
    if result.status == "dry-run":
        lines += _plan_lines(result)
        lines.append(
            "ready: a real run is allowed" if not result.blockers else "BLOCKED:"
        )
        lines += [f"  - {b}" for b in result.blockers]
        return "\n".join(lines)
    summary = result.summary()
    bw = [m for m in summary if m.kind in ("bw", "concurrency")]
    lat = [m for m in summary if m.kind in ("lat", "a2a")]
    if bw:
        lines += ["", "BANDWIDTH (GB/s per direction; raw = line rate after coding)"]
        lines += _table(
            ["path", "what", "GB/s", "raw", "eff", "vs audit"], _bandwidth_rows(bw)
        )
    if lat:
        lines += ["", "LATENCY (us; one-way push+flag, copy, kernel read or a2a step)"]
        lines += _table(
            ["path", "what", "bytes", "p50", "p99", "vs audit"], _latency_rows(lat)
        )
    report = result.effectiveness() if summary else None
    if report is not None:
        lines += ["", "FINDINGS"]
        found = report["findings"]
        if found:
            lines += [f"  [{f['severity']}] {f['code']}: {f['text']}" for f in found]
        else:
            lines.append("  none")
    return "\n".join(lines)


def _status_lines(result: MeasureResult) -> list[str]:
    seconds = (result.ended_ns - result.started_ns) / 1e9 if result.ended_ns else 0.0
    temp = ""
    if result.max_temp_c:
        gpu, hot = max(result.max_temp_c.items(), key=lambda kv: kv[1])
        temp = f"  max junction {hot:.0f} C ({gpu})"
    lines = [
        f"status {result.status}  "
        f"{'VALID' if result.valid else 'INVALID'}  "
        f"steps {result.steps_done}/{result.steps_total}  {seconds:.1f} s{temp}"
    ]
    if result.error:
        lines.append(f"error: {result.error}")
    if result.abort_reason:
        lines.append(f"aborted: {result.abort_reason}")
    lines += [f"invalid: {r}" for r in result.invalid_reasons]
    lines += [
        f"step failed: {s['step']}: {s['error']}  ({s['scenario']})"
        for s in result.steps_failed
    ]
    for link, text in result.degraded_links.items():
        lines.append(f"degraded link {link}: {text}")
    for item in result.evicted:
        lines.append(
            f"evicted_ms of pid {item['pid']} ({item['comm']}): "
            f"{item['evicted_ms_delta']}"
        )
    return lines
