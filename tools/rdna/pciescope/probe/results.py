# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert parsed ``p2p_probe`` blocks into pciescope ``Measurement`` records.

One probe command (a ``Step``) produces one ``Block`` of rows. Conversion rules:

* ``bw``/``hbw`` with ONE foreground flow -> ``kind='bw'``, value in GB/s,
  ``theoretical`` = the fabric model's ideal rate of that flow (raw line rate of
  the narrowest physical link of the path), ``efficiency`` = value / theoretical.
* ``bw``/``hbw`` with several flows or background flows -> ONE record
  ``kind='concurrency'``: value = the foreground aggregate, ``theoretical`` = the
  model's max-min fair aggregate of the foreground flows (background flows
  compete for the same links), per-flow numbers in ``stats`` (min/p50/max over
  flows) and in ``tags``; ``src``/``dst`` are the ``+``-joined entity sets.
* ``lat``/``latc``/``curve``/``hlat``/``hflag`` -> ``kind='lat'`` in
  microseconds (``hflag``: host to GPU doorbell round trip, ``size_bytes`` 0,
  not halved); ``a2a`` -> ``kind='a2a'`` (one-shot all-to-all step time).

``direction`` is ``h2d``, ``d2h``, ``d2d-push`` (kernel stores into the peer),
``d2d-pull`` (kernel loads from the peer, data flows src -> dst) or
``d2d-sdma`` (``hipMemcpyPeerAsync``). ``src``/``dst`` follow the data.
Every ``tags`` value is a string (Core's model).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pciescope.model import Measurement, Topology, now_ns

from . import parse
from .effectiveness import (
    BIDIR,
    UNIT_GBPS,
    BaselineDiff,
    EffRow,
    efficiency,
    findings,
    row_baseline,
)
from .ids import HOST, gpu_entity
from .paths import FabricModel
from .plan import Plan, Step

METHOD_BY_D2D_MODE = {"p": "kernel-store", "l": "kernel-load", "s": "sdma"}
DIRECTION_BY_D2D_MODE = {"p": "d2d-push", "l": "d2d-pull", "s": "d2d-sdma"}
DIRECTION_BY_CURVE_MODE = {
    "push": "d2d-push",
    "pull": "d2d-pull",
    "sdma": "d2d-sdma",
}
METHOD_BY_CURVE_MODE = {
    "push": "kernel-store",
    "pull": "kernel-load",
    "sdma": "sdma",
}
#: what the number of a CURVE row means
SEMANTICS_BY_CURVE_MODE = {
    "push": "one-way push+flag (RTT/2)",
    "pull": "kernel read of the payload",
    "sdma": "host-timed hipMemcpyPeerAsync + sync",
}
HOST_METHOD = {
    "h2d": "pinned",
    "d2h": "pinned",
    "h2d_pg": "pageable",
    "d2h_pg": "pageable",
    "h2d_k": "zero-copy",
    "d2h_k": "zero-copy",
}


@dataclass
class ConvertContext:
    """Everything the conversion needs besides the block."""

    plan: Plan
    topo: Topology
    model: FabricModel | None
    run_tags: dict[str, str] = field(default_factory=dict)
    probe_version: str = ""
    ok: bool = True  # False: the whole run is invalid (noisy, evicted ...)
    error: str | None = None

    def entity(self, lid: int) -> str:
        return gpu_entity(self.plan.buses[lid])


def path_class(topo: Topology, src: str, dst: str) -> str:
    """``intra-switch`` | ``cross-root`` | ``host`` (``same`` and unknown pass)."""
    try:
        return topo.path_class(src, dst)
    except (KeyError, ValueError):
        return ""


def _tags(
    step: Step, ctx: ConvertContext, extra: dict[str, str] | None = None
) -> dict[str, str]:
    tags = dict(ctx.run_tags)
    tags["step"] = step.id
    tags["scenario"] = step.scenario
    if step.pattern:
        tags["pattern"] = step.pattern
    tags.update(dict(step.tags))
    if ctx.probe_version:
        tags["probe"] = ctx.probe_version
    if extra:
        tags.update(extra)
    return tags


def _gpus_of(entities: Iterable[str]) -> list[str]:
    out: list[str] = []
    for e in entities:
        if e.startswith("gpu:") and e not in out:
            out.append(e)
    return out


def _efficiency(value: float, theoretical: float | None) -> float | None:
    if not theoretical:
        return None
    return value / theoretical


def _stats_of(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "n": float(len(ordered)),
    }


def _fmt(values: Iterable[float], digits: int = 2) -> str:
    return ",".join(f"{v:.{digits}f}" for v in values)


def _sum_known(values: Sequence[float | None]) -> float | None:
    """Sum of ``values``, None when any is unknown."""
    if any(v is None for v in values):
        return None
    return math.fsum(v for v in values if v is not None)


@dataclass(frozen=True)
class _Flow:
    """One flow of a bandwidth step, resolved to entities."""

    src: str
    dst: str
    direction: str
    method: str
    gbps: float
    bg: bool
    verify_ok: bool
    size: int
    blocks: int


def _d2d_flow(row: parse.XferRow, ctx: ConvertContext) -> _Flow:
    return _Flow(
        ctx.entity(row.src),
        ctx.entity(row.dst),
        DIRECTION_BY_D2D_MODE[row.mode],
        METHOD_BY_D2D_MODE[row.mode],
        row.gbps,
        row.bg,
        row.verify_ok,
        row.size,
        row.blocks,
    )


def _host_flow(row: parse.HostXferRow, ctx: ConvertContext) -> _Flow:
    gpu = ctx.entity(row.dev)
    h2d = row.mode.startswith("h2d")
    return _Flow(
        HOST if h2d else gpu,
        gpu if h2d else HOST,
        "h2d" if h2d else "d2h",
        HOST_METHOD[row.mode],
        row.gbps,
        row.bg,
        row.verify_ok,
        row.size,
        row.blocks,
    )


def _ideal(ctx: ConvertContext, flows: Sequence[_Flow]) -> list[float | None] | None:
    """Ideal fair rates of ``flows`` (foreground first), None without a model.

    A flow the model cannot bound (no limiting link on its path) gets None.
    """
    if ctx.model is None:
        return None
    try:
        rates = ctx.model.ideal_rates([(f.src, f.dst) for f in flows])
    except (KeyError, ValueError):
        return None
    return [r if math.isfinite(r) else None for r in rates]


def _bandwidth(
    step: Step,
    ctx: ConvertContext,
    flows: list[_Flow],
    agg_fg: float,
    agg_bg: float,
    started_ns: int,
    ended_ns: int,
) -> list[Measurement]:
    ordered = [f for f in flows if not f.bg] + [f for f in flows if f.bg]
    fg = [f for f in ordered if not f.bg]
    if not fg:
        return []
    ideal = _ideal(ctx, ordered)
    verify_ok = all(f.verify_ok for f in flows)
    ok = ctx.ok and verify_ok
    error = ctx.error if not ctx.ok else (None if verify_ok else "verify FAIL")
    size = fg[0].size
    if len(flows) == 1:
        f = fg[0]
        theo = ideal[0] if ideal else None
        extra = {"method": f.method, "blocks": str(f.blocks), "flows": "1"}
        return [
            Measurement(
                kind="bw",
                src=f.src,
                dst=f.dst,
                direction=f.direction,
                size_bytes=f.size,
                value=f.gbps,
                unit="GB/s",
                stats={"n": 1.0},
                path_class=path_class(ctx.topo, f.src, f.dst),
                theoretical=theo,
                efficiency=_efficiency(f.gbps, theo),
                started_ns=started_ns,
                ended_ns=ended_ns,
                gpus_used=_gpus_of((f.src, f.dst)),
                id=f"{step.id}",
                tags=_tags(step, ctx, extra),
                ok=ok,
                error=error,
            )
        ]
    classes = sorted(
        {path_class(ctx.topo, f.src, f.dst) for f in ordered if f.src != f.dst}
    )
    theo_fg = _sum_known(ideal[: len(fg)]) if ideal else None
    per_flow = [f.gbps for f in fg]
    extra = {
        "method": ",".join(sorted({f.method for f in fg})),
        "blocks": str(fg[0].blocks),
        "flows": str(len(fg)),
        "flow_list": ",".join(f"{f.src}>{f.dst}" for f in fg),
        "per_flow_gbps": _fmt(per_flow),
        "path_classes": ",".join(classes),
        "bg_flows": str(len(ordered) - len(fg)),
        "bg_gbps": f"{agg_bg:.2f}",
    }
    if ideal:
        extra["ideal_per_flow_gbps"] = ",".join(
            "-" if r is None else f"{r:.2f}" for r in ideal[: len(fg)]
        )
    directions = sorted({f.direction for f in fg})
    sources = sorted({f.src for f in fg})
    dests = sorted({f.dst for f in fg})
    if "host" in sources + dests:
        pclass = "host"
    elif "cross-root" in classes:
        pclass = "cross-root"
    else:
        pclass = classes[0] if classes else ""
    gpus = _gpus_of(sources + dests + [x for f in ordered for x in (f.src, f.dst)])
    return [
        Measurement(
            kind="concurrency",
            src="+".join(sources),
            dst="+".join(dests),
            direction=directions[0] if len(directions) == 1 else "mixed",
            size_bytes=size,
            value=agg_fg,
            unit="GB/s",
            stats=_stats_of(per_flow),
            path_class=pclass,
            theoretical=theo_fg,
            efficiency=_efficiency(agg_fg, theo_fg),
            started_ns=started_ns,
            ended_ns=ended_ns,
            gpus_used=gpus,
            id=f"{step.id}",
            tags=_tags(step, ctx, extra),
            ok=ok,
            error=error,
        )
    ]


def _convert_bw(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    flows = [_d2d_flow(r, ctx) for r in block.rows if isinstance(r, parse.XferRow)]
    agg = next((r for r in block.rows if isinstance(r, parse.AggRow)), None)
    if not flows or agg is None:
        return []
    return _bandwidth(step, ctx, flows, agg.fg_gbps, agg.bg_gbps, t0, t1)


def _convert_hbw(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    flows = [_host_flow(r, ctx) for r in block.rows if isinstance(r, parse.HostXferRow)]
    agg = next((r for r in block.rows if isinstance(r, parse.HostAggRow)), None)
    if not flows or agg is None:
        return []
    return _bandwidth(step, ctx, flows, agg.fg_gbps, agg.bg_gbps, t0, t1)


def _convert_lat(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    iters = 0
    if len(block.argv) > 1 and block.argv[1].isdigit():
        iters = int(block.argv[1])
    out: list[Measurement] = []
    rows = [r for r in block.rows if isinstance(r, parse.LatRow)]
    for r in rows:
        src, dst = ctx.entity(r.a), ctx.entity(r.b)
        extra = {
            "method": "kernel-store",
            "semantics": "one-way push+flag (RTT/2, mean of 2000 iterations)",
        }
        if r.concurrent:
            extra["concurrent_pairs"] = str(len(rows))
        out.append(
            Measurement(
                kind="lat",
                src=src,
                dst=dst,
                direction="d2d-push",
                size_bytes=r.payload,
                value=r.one_way_us,
                unit="us",
                stats={"mean": r.one_way_us, "n": float(iters)},
                path_class=path_class(ctx.topo, src, dst),
                started_ns=t0,
                ended_ns=t1,
                gpus_used=_gpus_of((src, dst)),
                id=f"{step.id}:{r.a}-{r.b}",
                tags=_tags(step, ctx, extra),
                ok=ctx.ok,
                error=None if ctx.ok else ctx.error,
            )
        )
    return out


def _convert_curve(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    out: list[Measurement] = []
    for r in block.rows:
        if not isinstance(r, parse.CurveRow):
            continue
        src, dst = ctx.entity(r.src), ctx.entity(r.dst)
        stats = {
            "min": r.min_us,
            "p50": r.p50_us,
            "p99": r.p99_us,
            "max": r.max_us,
            "mean": r.mean_us,
            "n": float(r.n),
        }
        extra = {
            "method": METHOD_BY_CURVE_MODE[r.mode],
            "semantics": SEMANTICS_BY_CURVE_MODE[r.mode],
            "blocks": str(r.blocks),
        }
        if r.ev_us is not None:
            extra["ev_us"] = f"{r.ev_us:.3f}"
        out.append(
            Measurement(
                kind="lat",
                src=src,
                dst=dst,
                direction=DIRECTION_BY_CURVE_MODE[r.mode],
                size_bytes=r.size,
                value=r.p50_us,
                unit="us",
                stats=stats,
                path_class=path_class(ctx.topo, src, dst),
                started_ns=t0,
                ended_ns=t1,
                gpus_used=_gpus_of((src, dst)),
                id=f"{step.id}:{r.mode}-{r.src}-{r.dst}-{r.size}",
                tags=_tags(step, ctx, extra),
                ok=ctx.ok,
                error=None if ctx.ok else ctx.error,
            )
        )
    return out


def _convert_hlat(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    out: list[Measurement] = []
    for r in block.rows:
        if not isinstance(r, parse.HostLatRow):
            continue
        gpu = ctx.entity(r.dev)
        h2d = r.direction == "h2d"
        src, dst = (HOST, gpu) if h2d else (gpu, HOST)
        extra = {
            "method": r.mem,
            "semantics": "host-timed hipMemcpyAsync + sync",
        }
        out.append(
            Measurement(
                kind="lat",
                src=src,
                dst=dst,
                direction=r.direction,
                size_bytes=r.size,
                value=r.p50_us,
                unit="us",
                stats={
                    "min": r.min_us,
                    "p50": r.p50_us,
                    "p99": r.p99_us,
                    "max": r.max_us,
                    "mean": r.mean_us,
                    "n": float(r.n),
                },
                path_class="host",
                started_ns=t0,
                ended_ns=t1,
                gpus_used=[gpu],
                id=f"{step.id}:{r.direction}-{r.mem}-{r.dev}-{r.size}",
                tags=_tags(step, ctx, extra),
                ok=ctx.ok,
                error=None if ctx.ok else ctx.error,
            )
        )
    return out


def _convert_hflag(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    out: list[Measurement] = []
    for r in block.rows:
        if not isinstance(r, parse.HostFlagRow):
            continue
        gpu = ctx.entity(r.dev)
        extra = {
            "method": "doorbell",
            "semantics": "round trip: CPU store, GPU poll and answer, CPU poll "
            "(not halved)",
        }
        out.append(
            Measurement(
                kind="lat",
                src=HOST,
                dst=gpu,
                direction="h2d",
                size_bytes=0,
                value=r.rtt_p50_us,
                unit="us",
                stats={
                    "min": r.rtt_min_us,
                    "p50": r.rtt_p50_us,
                    "p99": r.rtt_p99_us,
                    "max": r.rtt_max_us,
                    "mean": r.rtt_mean_us,
                    "n": float(r.n),
                },
                path_class="host",
                started_ns=t0,
                ended_ns=t1,
                gpus_used=[gpu],
                id=f"{step.id}:hflag-{r.dev}",
                tags=_tags(step, ctx, extra),
                ok=ctx.ok,
                error=None if ctx.ok else ctx.error,
            )
        )
    return out


def _convert_a2a(
    step: Step, block: parse.Block, ctx: ConvertContext, t0: int, t1: int
) -> list[Measurement]:
    out: list[Measurement] = []
    for r in block.rows:
        if not isinstance(r, parse.A2aRow):
            continue
        members = [ctx.entity(i) for i in r.group]
        switches = {ctx.plan.switch[i] for i in r.group}
        pclass = "cross-root" if len(switches) > 1 else "intra-switch"
        extra = {
            "method": "kernel-store",
            "ranks": str(len(r.group)),
            "semantics": "one-shot all-to-all step: push, fence, flag, wait",
        }
        out.append(
            Measurement(
                kind="a2a",
                src="+".join(members),
                dst="+".join(members),
                direction="d2d-push",
                size_bytes=r.payload,
                value=r.step_us,
                unit="us",
                stats={"n": 1.0},
                path_class=pclass,
                started_ns=t0,
                ended_ns=t1,
                gpus_used=members,
                id=f"{step.id}:{'-'.join(map(str, r.group))}",
                tags=_tags(step, ctx, extra),
                ok=ctx.ok,
                error=None if ctx.ok else ctx.error,
            )
        )
    return out


_CONVERTERS: dict[
    str, Callable[[Step, parse.Block, ConvertContext, int, int], list[Measurement]]
] = {
    "bw": _convert_bw,
    "hbw": _convert_hbw,
    "lat": _convert_lat,
    "latc": _convert_lat,
    "curve": _convert_curve,
    "hlat": _convert_hlat,
    "hflag": _convert_hflag,
    "a2a": _convert_a2a,
}


def convert_block(
    step: Step,
    block: parse.Block,
    ctx: ConvertContext,
    started_ns: int | None = None,
    ended_ns: int | None = None,
) -> list[Measurement]:
    """Return the Measurements of one finished step (empty on a failed block)."""
    if block.rc not in (None, 0) or any(
        isinstance(r, parse.ErrRow) for r in block.rows
    ):
        return []
    t1 = ended_ns if ended_ns is not None else now_ns()
    t0 = started_ns if started_ns is not None else t1
    convert = _CONVERTERS[step.kind]
    return convert(step, block, ctx, t0, t1)


def block_error(block: parse.Block) -> str | None:
    """Text of the first error of a block (``ERR`` row or non-zero rc), if any."""
    for r in block.rows:
        if isinstance(r, parse.ErrRow):
            return r.text
    if block.rc not in (None, 0):
        return f"rc={block.rc}"
    return None


# ---------------------------------------------------------------------------
# repeats
# ---------------------------------------------------------------------------


def _identity(m: Measurement) -> tuple[str, ...]:
    t = m.tags
    return (
        m.kind,
        m.src,
        m.dst,
        m.direction,
        str(m.size_bytes),
        t.get("method", ""),
        t.get("pattern", ""),
        t.get("flow_list", ""),
        t.get("scenario", ""),
    )


def aggregate(measurements: Iterable[Measurement]) -> list[Measurement]:
    """Fold repeats of the same cell into one record (median, spread in stats).

    Records with the same identity (kind, endpoints, direction, size, method,
    pattern, flows, scenario) are combined; the value is the median of the
    values, ``stats`` min/p50/max/mean/n are over the repeats (p99 of a
    latency curve stays the worst p99 seen). Invalid repeats are dropped when
    a valid one exists. Order of first appearance is kept.
    """
    groups: dict[tuple[str, ...], list[Measurement]] = {}
    for m in measurements:
        groups.setdefault(_identity(m), []).append(m)
    out: list[Measurement] = []
    for members in groups.values():
        good = [m for m in members if m.ok] or members
        if len(good) == 1:
            out.append(good[0])
            continue
        values = [m.value for m in good]
        first = good[0]
        stats = _stats_of(values)
        if "p99" in first.stats:
            stats["p99"] = max(m.stats.get("p99", m.value) for m in good)
        tags = dict(first.tags)
        tags["reps"] = str(len(good))
        tags.pop("rep", None)
        out.append(
            Measurement(
                kind=first.kind,
                src=first.src,
                dst=first.dst,
                direction=first.direction,
                size_bytes=first.size_bytes,
                value=stats["p50"],
                unit=first.unit,
                stats=stats,
                path_class=first.path_class,
                theoretical=first.theoretical,
                efficiency=_efficiency(stats["p50"], first.theoretical),
                started_ns=min(m.started_ns for m in good),
                ended_ns=max(m.ended_ns for m in good),
                gpus_used=list(first.gpus_used),
                tool_version=first.tool_version,
                id=first.id,
                tags=tags,
                ok=all(m.ok for m in good),
                error=None if all(m.ok for m in good) else first.error,
            )
        )
    return out


# ---------------------------------------------------------------------------
# effectiveness: measured vs theoretical per path class
# ---------------------------------------------------------------------------

_D2D_KIND_BY_METHOD = {"kernel-store": "push", "kernel-load": "pull", "sdma": "sdma"}
_PATTERN_KIND = {
    "ring-intra": "ring",
    "ring-both": "ring",
    "fan-in": "fan-in",
    "fan-in-cross": "fan-in",
    "fan-out": "fan-out",
    "fan-out-cross": "fan-out",
    "a2a-intra": "a2a",
}


def _eff_row(m: Measurement, kind: str, direction: str, n_flows: int) -> EffRow:
    return EffRow(
        kind=kind,
        direction=direction,
        path_class=m.path_class,
        size=m.size_bytes,
        value=m.value,
        unit=m.unit,
        n_flows=n_flows,
        theoretical=m.theoretical,
        src=m.src,
        dst=m.dst,
        label=m.tags.get("scenario", m.id),
    )


def _eff_bw(m: Measurement) -> EffRow | None:
    method = m.tags.get("method", "")
    if m.path_class == "host":
        host_kind = "host" if method == "pinned" else f"host-{method}"
        return _eff_row(m, host_kind, m.direction, 1)
    kind = _D2D_KIND_BY_METHOD.get(method)
    return None if kind is None else _eff_row(m, kind, "fwd", 1)


def _eff_concurrency(m: Measurement) -> EffRow | None:
    pattern = m.tags.get("pattern", "")
    flows = m.tags.get("flows", "")
    if not flows.isdigit():
        return None
    n = int(flows)
    if m.path_class == "host":
        direction = "bidir" if "bidir" in pattern else m.direction
        return _eff_row(m, "host", direction, n)
    if pattern.startswith("cross-k") and m.path_class == "cross-root":
        bidir = pattern.endswith("-bidir")
        return _eff_row(m, "push", BIDIR if bidir else "fwd", n)
    if pattern == "bidir-pair":
        return _eff_row(m, "push", BIDIR, n)
    kind = _PATTERN_KIND.get(pattern)
    return None if kind is None else _eff_row(m, kind, "fwd", n)


def _eff_lat(m: Measurement) -> EffRow | None:
    semantics = m.tags.get("semantics", "")
    if not semantics.startswith("one-way push+flag") or "concurrent_pairs" in m.tags:
        return None
    return _eff_row(m, "lat", "none", 1)


def _eff_a2a(m: Measurement) -> EffRow | None:
    ranks = m.tags.get("ranks", "")
    return _eff_row(m, "a2a-step", "none", int(ranks)) if ranks.isdigit() else None


def eff_row(m: Measurement) -> EffRow | None:
    """Map one valid measurement to a row of ``effectiveness`` (None: no rule).

    Pageable and zero-copy host transfers get their own kinds (``host-pageable``,
    ``host-zero-copy``): no rule compares them with the raw link rate.
    """
    if not m.ok:
        return None
    if m.kind == "bw":
        return _eff_bw(m)
    if m.kind == "concurrency":
        return _eff_concurrency(m)
    if m.kind == "lat":
        return _eff_lat(m)
    if m.kind == "a2a":
        return _eff_a2a(m)
    return None


def eff_rows(measurements: Iterable[Measurement]) -> list[EffRow]:
    """Rows of ``effectiveness`` for every valid measurement that has one."""
    rows = (eff_row(m) for m in measurements)
    return [r for r in rows if r is not None]


def _baseline_dict(diff: BaselineDiff | None) -> dict[str, Any] | None:
    if diff is None:
        return None
    return {
        "key": diff.key,
        "baseline": diff.baseline,
        "unit": diff.unit,
        "ratio": round(diff.ratio, 3),
        "tolerance": diff.tolerance,
        "verdict": diff.verdict,
        "source": diff.source,
    }


def effectiveness_report(
    measurements: Sequence[Measurement],
    model: FabricModel | None = None,
    *,
    degraded_links: Mapping[str, str] | None = None,
    suspended: Collection[str] = (),
) -> dict[str, Any]:
    """Measured vs theoretical per path class, baseline diffs and findings.

    ``measurements`` should be the aggregated records (``aggregate``). Returns a
    JSON-ready dict: ``rows`` (one per comparable measurement with value,
    theoretical, efficiency, practical ceiling and the audit baseline diff),
    ``by_class`` (best bandwidth and efficiency per path class) and
    ``findings``.
    """
    rows = eff_rows(measurements)
    table: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, Any]] = {}
    for row in rows:
        eff = efficiency(row.value, row.theoretical)
        ceiling = (
            FabricModel.ceiling(row.theoretical)
            if row.theoretical and row.unit == UNIT_GBPS
            else None
        )
        table.append(
            {
                "label": row.label,
                "kind": row.kind,
                "direction": row.direction,
                "path_class": row.path_class,
                "size_bytes": row.size,
                "n_flows": row.n_flows,
                "value": row.value,
                "unit": row.unit,
                "theoretical": row.theoretical,
                "efficiency": eff,
                "practical_ceiling": ceiling,
                "of_practical_ceiling": efficiency(row.value, ceiling),
                "baseline": _baseline_dict(row_baseline(row)),
            }
        )
        if row.unit == UNIT_GBPS and row.theoretical:
            cell = by_class.setdefault(
                row.path_class, {"best_gbps": 0.0, "best_efficiency": 0.0, "rows": 0}
            )
            cell["rows"] += 1
            cell["best_gbps"] = max(cell["best_gbps"], row.value)
            cell["best_efficiency"] = max(cell["best_efficiency"], eff or 0.0)
    found = findings(rows, model, degraded_links=degraded_links, suspended=suspended)
    return {
        "rows": table,
        "by_class": by_class,
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "text": f.text,
                "rows": list(f.rows),
            }
            for f in found
        ],
    }
