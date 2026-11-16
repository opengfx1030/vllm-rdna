# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit baseline and efficiency findings of the probe measurements.

``AUDIT_BASELINE`` transcribes the numbers of
``docs/rdna2/pcie-iommu-audit-2026-11-06.md`` (every entry cites its table and
line in ``source``). Where the audit gives a range, the value is the midpoint
of that range and the tolerance is the larger of 10 % and the half range
relative to the midpoint. Tolerances are fractions (0.10 = 10 %).

``EffRow`` vocabulary (the producer of the rows follows it):

* ``kind``: ``push`` | ``pull`` | ``sdma`` (GPU to GPU bandwidth, GB/s),
  ``host`` (host to or from GPU bandwidth, GB/s), ``ring`` | ``a2a`` |
  ``fan-in`` | ``fan-out`` (aggregate of a traffic pattern, GB/s, ``n_flows``
  flows), ``lat`` (one-way latency with ``size`` bytes of payload, us),
  ``a2a-step`` (one-shot all-to-all step, us, ``n_flows`` GPUs) and
  ``rccl-allreduce`` (us, ``n_flows`` ranks).
* ``direction``: ``h2d`` | ``d2h`` | ``fwd`` | ``bidir`` | ``none``;
  ``bidir`` = both directions at once, ``n_flows`` counts both directions.
* ``path_class``: ``intra-switch`` | ``cross-root`` | ``host`` (the values of
  ``Topology.path_class``; ``intra-root`` is accepted but has no baseline).
* ``theoretical``: raw capacity in GB/s the model assigns to the flows
  (``FabricModel.theoretical_aggregate``), None when unknown.
* ``src``/``dst``: entity ids (``gpu:0a``, ``host``); ``label``: free text
  naming the row in findings.
* ``link_raw``: smallest per-flow bottleneck raw capacity of the step's route in
  GB/s (conversion tag ``link_raw_gbps``), None when unknown; it selects the
  baseline table (15.754 = x8 -> interim table, 31.508 = x16 -> audit table).

Two baseline tables: ``AUDIT_BASELINE`` (x16 GPU links, the 2026-11-05 layout)
and ``INTERIM_X8_BASELINE`` (the quoted box rules of the x8 layout since
2026-11-08: ~13.9 GB/s single flow). ``row_baseline`` picks the table from
``EffRow.theoretical`` (``is_x8_limited``); a case the chosen table lacks has
no baseline, so an x8 route never gets an audit verdict.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from pciescope.model import gpu_entity, raw_gbs

from .paths import FabricModel

KIB = 1024
MIB = 1024 * KIB

BIDIR = "bidir"
UNIT_GBPS = "GB/s"
UNIT_US = "us"

# --- rule thresholds (contract of the probe) ---------------------------------
TOLERANCE = 0.10  # default baseline tolerance
PUSH_MIN_OF_CEILING = 0.70  # single-flow push below this share of the ceiling: warn
PULL_CROSS_MAX_OF_INTRA = 0.50  # cross-root pull below this share of intra: finding
COLLAPSE_MIN_FLOWS = 3  # k >= 3 concurrent cross-root flows
COLLAPSE_MAX_OF_SINGLE = 0.75  # aggregate below this share of one flow: finding
EXCEEDS_THEORETICAL = 1.25  # more than 125 % of the raw capacity: model is wrong
LATENCY_FACTOR = 1.25  # latency above baseline x 1.25: warn
HOST_MIN_OF_RAW = 0.60  # host bandwidth below this share of the raw link: warn

# --- link classes of the baselines ---------------------------------------------
# The audit table was measured with x16 GPU links (2026-11-05 layout). Since
# 2026-11-08 every GPU link is x8 (the box rules quote ~13.9 GB/s for it).
X16_GBPS = raw_gbs(16.0, 16)  # 31.508 GB/s raw, per direction
X8_GBPS = raw_gbs(16.0, 8)  # 15.754 GB/s raw, per direction
LINK_CLASS_TOLERANCE = 0.05  # a theoretical value within 5 % of a class is of it
INTERIM_X8_GBPS = 13.9  # box rules 2026-11-10, single flow on an x8 route
X8_LINK = "x8"
X8_QUALIFIER = f"link={X8_LINK}"

_DOC = "docs/rdna2/pcie-iommu-audit-2026-11-06.md"


_CLASS_SHORT = {
    "intra-switch": "intra",
    "cross-root": "cross",
    "host": "host",
    "intra-root": "intra-root",
    "intra": "intra",
    "cross": "cross",
}

# Kinds whose rows are aggregates of a pattern keyed by the number of flows.
_PATTERN_KINDS = frozenset({"ring", "a2a", "fan-in", "fan-out"})
_SINGLE_KINDS = ("push", "pull", "sdma")


@dataclass(frozen=True)
class Baseline:
    """One audit number (``unit`` ``GB/s`` or ``us``, ``tolerance`` a fraction)."""

    key: str
    value: float
    unit: str
    tolerance: float
    source: str


@dataclass(frozen=True)
class BaselineDiff:
    """Result of ``compare_to_baseline``."""

    key: str
    value: float
    baseline: float
    unit: str
    ratio: float  # value / baseline
    tolerance: float
    verdict: str  # 'ok' | 'slower' | 'faster'
    source: str


@dataclass(frozen=True)
class EffRow:
    """One measured value with the model's theoretical value (see the module)."""

    kind: str
    direction: str
    path_class: str
    size: int | None
    value: float
    unit: str
    n_flows: int
    theoretical: float | None
    src: str
    dst: str
    label: str
    link_raw: float | None = None  # smallest per-flow bottleneck capacity, GB/s


@dataclass(frozen=True)
class Finding:
    """One finding; ``rows`` are the labels of the rows it is about."""

    severity: str  # 'info' | 'warn' | 'bad'
    code: str
    text: str
    rows: tuple[str, ...]


def _size_label(size: int) -> str:
    if size >= MIB and size % MIB == 0:
        return f"{size // MIB}MiB"
    if size >= KIB and size % KIB == 0:
        return f"{size // KIB}KiB"
    return f"{size}B"


def baseline_key(
    kind: str, path_class: str, size_bytes: int | None = None, **qual: object
) -> str:
    """Return the baseline key ``<kind>.<class>[.<size>][.<k>=<v>...]``.

    ``path_class`` is ``intra-switch`` / ``cross-root`` / ``host`` (shortened
    to ``intra`` / ``cross`` / ``host``); the size reads ``64KiB`` or ``1MiB``;
    qualifiers follow sorted by name, for example ``push.cross.1MiB`` and
    ``concurrent-push.cross.pairs=3``.

    Raises:
        ValueError: for an unknown path class.

    """
    short = _CLASS_SHORT.get(path_class)
    if short is None:
        raise ValueError(f"unknown path class {path_class!r}")
    parts = [kind, short]
    if size_bytes is not None:
        parts.append(_size_label(size_bytes))
    parts += [f"{k}={qual[k]}" for k in sorted(qual)]
    return ".".join(parts)


def _build_baseline() -> dict[str, Baseline]:
    table: dict[str, Baseline] = {}

    def point(
        key: str,
        value: float,
        unit: str,
        source: str,
        spread: Sequence[float] = (),
    ) -> None:
        # Tolerance: 10 %, or the largest deviation of the other audit values.
        tol = max([TOLERANCE] + [abs(x - value) / value for x in spread])
        table[key] = Baseline(key, value, unit, tol, f"{_DOC}: {source}")

    def span(key: str, lo: float, hi: float, unit: str, source: str) -> None:
        # Range in the audit: midpoint, tolerance max(10 %, half range).
        mid = (lo + hi) / 2.0
        tol = max(TOLERANCE, (hi - lo) / 2.0 / mid)
        text = f"{source}; range {lo}-{hi}, value = midpoint"
        table[key] = Baseline(key, mid, unit, tol, f"{_DOC}: {text}")

    # "Single pair, GB/s per direction (min / mean / max over ordered pairs)":
    # lines 121-132. Intra range = lowest min and highest max of the columns
    # Intra A and Intra B; cross-root range = min and max of that column.
    single = (
        # kind, size, intra lo, intra hi, cross lo, cross hi, line
        ("push", 64 * KIB, 24.6, 27.0, 25.2, 26.8, 121),
        ("push", 1 * MIB, 24.5, 26.9, 25.1, 26.8, 122),
        ("push", 5 * MIB, 24.0, 26.3, 24.9, 26.2, 123),
        ("push", 64 * MIB, 25.4, 28.0, 25.5, 27.8, 124),
        ("pull", 64 * KIB, 24.4, 27.0, 5.7, 5.8, 125),
        ("pull", 1 * MIB, 24.4, 27.0, 5.6, 5.8, 126),
        ("pull", 5 * MIB, 23.7, 26.2, 5.6, 5.8, 127),
        ("pull", 64 * MIB, 25.4, 28.2, 5.7, 5.9, 128),
        ("sdma", 64 * KIB, 5.9, 6.1, 4.6, 4.8, 129),
        ("sdma", 1 * MIB, 19.9, 21.5, 19.8, 21.6, 130),
        ("sdma", 5 * MIB, 24.2, 26.7, 24.2, 25.3, 131),
        ("sdma", 64 * MIB, 25.4, 28.0, 25.5, 26.4, 132),
    )
    for kind, size, ilo, ihi, clo, chi, line in single:
        label = f"'Single pair' table line {line} ({kind} {_size_label(size)})"
        span(baseline_key(kind, "intra-switch", size), ilo, ihi, UNIT_GBPS, label)
        span(baseline_key(kind, "cross-root", size), clo, chi, UNIT_GBPS, label)

    # "Concurrent flows", cross-root, per direction (lines 187-190), the
    # bidirectional column counts both directions together; line 194: every
    # 3-subset gives 18.3-19.6 GB/s (used as spread of the 3-pair entry).
    pairs_table = (
        # pairs, aggregate, bidirectional aggregate, spread of the aggregate
        (1, 25.2, 53.5, ()),
        (2, 27.6, 55.1, ()),
        (3, 18.7, 39.5, (18.3, 19.6)),
        (4, 17.0, 35.3, ()),
    )
    for pairs, agg, bidir, spread in pairs_table:
        line = 186 + pairs
        point(
            baseline_key("concurrent-push", "cross-root", pairs=pairs),
            agg,
            UNIT_GBPS,
            f"'Concurrent flows' table line {line} ({pairs} concurrent "
            "cross-root pairs, aggregate per direction)",
            spread,
        )
        point(
            baseline_key("concurrent-push-bidir", "cross-root", pairs=pairs),
            bidir,
            UNIT_GBPS,
            f"'Concurrent flows' table line {line} ({pairs} concurrent "
            "cross-root pairs, bidirectional aggregate)",
        )
    point(
        baseline_key("fan-in", "cross-root", flows=4),
        25.6,
        UNIT_GBPS,
        "'Concurrent flows' table line 191 (4 sources to 1 destination)",
    )
    for flows in (3, 4):
        point(
            baseline_key("fan-out", "cross-root", flows=flows),
            27.7,
            UNIT_GBPS,
            "'Concurrent flows' table line 192 (1 source to 3-4 destinations)",
        )

    # "Intra-switch, 4 GPUs concurrently" (lines 200-202).
    point(
        baseline_key("ring", "intra-switch", flows=4),
        105.0,
        UNIT_GBPS,
        "'Intra-switch, 4 GPUs concurrently' table line 200 "
        "(ring ce>d3>d8>db>ce, 4 flows)",
    )
    point(
        baseline_key("ring", "intra-switch", flows=8),
        210.0,
        UNIT_GBPS,
        "'Intra-switch, 4 GPUs concurrently' table line 202 "
        "(both switches, ring each, 8 flows)",
    )
    span(
        baseline_key("a2a", "intra-switch", flows=12),
        93.0,
        107.0,
        UNIT_GBPS,
        "'Intra-switch, 4 GPUs concurrently' table line 201 (all-to-all "
        "inside one switch, 12 flows: 105-107 at 64 KiB, 93 at 1-20 MiB)",
    )

    # Latency, one-way flag ping-pong (us). Line 179: intra-switch 6.1-6.3
    # flag only (7.0 at 16 KiB, 8.4 at 64 KiB), cross-root 6.8-6.9 (7.6, 9.0).
    span(
        baseline_key("lat", "intra-switch", 0),
        6.1,
        6.3,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (flag only)",
    )
    span(
        baseline_key("lat", "cross-root", 0),
        6.8,
        6.9,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (flag only)",
    )
    point(
        baseline_key("lat", "intra-switch", 16 * KIB),
        7.0,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (16 KiB payload)",
    )
    point(
        baseline_key("lat", "intra-switch", 64 * KIB),
        8.4,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (64 KiB payload)",
    )
    point(
        baseline_key("lat", "cross-root", 4 * KIB),
        7.2,
        UNIT_US,
        "'Latency regime' table line 223 (4 KiB, push+flag k=1)",
    )
    span(
        baseline_key("lat", "cross-root", 16 * KIB),
        7.1,
        7.6,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (7.6) and 'Latency regime' "
        "table line 225 (7.1), 16 KiB payload",
    )
    point(
        baseline_key("lat", "cross-root", 64 * KIB),
        9.0,
        UNIT_US,
        "'Latency, one-way flag ping-pong' line 179 (64 KiB payload; line 227 "
        "k=1 agrees)",
    )

    # One-shot all-to-all step (us), 4 KiB (line 233); the 14.0-15.2 us of the
    # 'pcie_ports' states (line 351) are the spread of the flat 8-GPU entry.
    point(
        baseline_key("a2a-step", "cross-root", 4 * KIB, gpus=8),
        15.7,
        UNIT_US,
        "'One-shot all-to-all step' table line 233 (8 GPUs flat, 4 KiB); "
        "spread: 14.0, 14.8, 15.2 us of line 351",
        (14.0, 14.8, 15.2),
    )
    point(
        baseline_key("a2a-step", "intra-switch", 4 * KIB, gpus=4),
        10.4,
        UNIT_US,
        "'One-shot all-to-all step' table line 233 (4 GPUs inside each switch, "
        "both at once, 4 KiB)",
    )

    # RCCL all_reduce TP8, 4 KiB, ring of 8 ranks (2 cross-root hops): 115 us
    # (lines 24 and 248); 113 and 110 (line 331), 111 and 109 (line 352).
    span(
        baseline_key("rccl-allreduce", "cross-root", 4 * KIB, tp=8),
        109.0,
        115.0,
        UNIT_US,
        "RCCL all_reduce TP8 4 KiB, lines 248 (115), 331 (113, 110, 110, 109) "
        "and 352 (113, 110, 111, 109, 110, 111)",
    )
    return table


AUDIT_BASELINE: dict[str, Baseline] = _build_baseline()


_INTERIM_SOURCE = (
    "lease README 2026-11-10 (box rules), quoted, not measured by us: "
    "~13.9 GB/s P2P (same or cross leaf) and host->GPU"
)


def _interim_point(key: str, case: str) -> Baseline:
    return Baseline(
        key, INTERIM_X8_GBPS, UNIT_GBPS, TOLERANCE, f"{_INTERIM_SOURCE}; {case}"
    )


def _build_interim_x8_baseline() -> dict[str, Baseline]:
    table: dict[str, Baseline] = {}
    # The box rules quote one number for every single flow between two GPUs
    # (same or cross leaf) and for host to GPU; we did not measure it.
    cases = [(k, s) for k in _SINGLE_KINDS for s in (1 * MIB, 64 * MIB)]
    for kind, size in cases:
        key = baseline_key(kind, "intra-switch", size, link=X8_LINK)
        table[key] = _interim_point(key, f"single-flow {kind}, {_size_label(size)}")
    for direction in ("h2d", "d2h"):
        key = baseline_key("host", "host", 64 * MIB, dir=direction, link=X8_LINK)
        table[key] = _interim_point(key, f"{direction} pinned, 64MiB")
    return table


INTERIM_X8_BASELINE: dict[str, Baseline] = _build_interim_x8_baseline()


def compare_to_baseline(
    key: str,
    value: float,
    unit: str,
    *,
    table: Mapping[str, Baseline] | None = None,
) -> BaselineDiff | None:
    """Compare ``value`` with the baseline of ``key`` (default: the audit table).

    The ratio is ``value / baseline``; the verdict is ``ok`` within the
    tolerance, otherwise ``slower`` / ``faster`` (for ``us`` a higher value is
    slower, for ``GB/s`` a lower one). None when the key has no baseline.

    Raises:
        ValueError: when ``unit`` differs from the unit of the baseline.

    """
    base = (AUDIT_BASELINE if table is None else table).get(key)
    if base is None:
        return None
    if unit != base.unit:
        raise ValueError(f"{key}: unit {unit!r}, the baseline is in {base.unit!r}")
    ratio = value / base.value
    if abs(ratio - 1.0) <= base.tolerance:
        verdict = "ok"
    elif (ratio < 1.0) == (unit == UNIT_GBPS):
        verdict = "slower"
    else:
        verdict = "faster"
    return BaselineDiff(
        key, value, base.value, unit, ratio, base.tolerance, verdict, base.source
    )


def efficiency(value: float, theoretical: float | None) -> float | None:
    """Return ``value / theoretical``, None when ``theoretical`` is None or 0."""
    if not theoretical:
        return None
    return value / theoretical


def _row_key(row: EffRow) -> str | None:
    """Return the baseline key of a row, None when no table has such a case."""
    pc, kind = row.path_class, row.kind
    if pc not in _CLASS_SHORT:
        return None
    bidir = row.direction == BIDIR
    if kind in _SINGLE_KINDS and row.n_flows == 1 and not bidir:
        return baseline_key(kind, pc, row.size)
    if kind == "host" and row.n_flows == 1 and row.direction in ("h2d", "d2h"):
        return baseline_key(kind, pc, row.size, dir=row.direction)
    if kind == "push" and pc == "cross-root":
        if not bidir:
            return baseline_key("concurrent-push", pc, pairs=row.n_flows)
        if row.n_flows % 2 == 0:
            return baseline_key("concurrent-push-bidir", pc, pairs=row.n_flows // 2)
        return None
    if kind in _PATTERN_KINDS:
        return baseline_key(kind, pc, flows=row.n_flows)
    if kind == "lat":
        return baseline_key(kind, pc, row.size)
    if kind == "a2a-step":
        return baseline_key(kind, pc, row.size, gpus=row.n_flows)
    if kind == "rccl-allreduce":
        return baseline_key(kind, pc, row.size, tp=row.n_flows)
    return None


def _near(value: float, target: float) -> bool:
    return abs(value - target) <= LINK_CLASS_TOLERANCE * target


def is_x8_limited(row: EffRow) -> bool:
    """True when the route of ``row`` has an x8 GPU link (the audit must not judge it).

    ``EffRow.link_raw`` (set at conversion from the fabric model: the smallest
    per-flow bottleneck capacity of the step) decides when present: within 5 %
    of one x8 link. Without it (older sessions, no model) the aggregate
    ``theoretical`` is used: within 5 % of one x16 link is x16 (checked first,
    two x8 flows add up to the same number), one x8 link or ``n_flows``
    independent x8 links is x8. [INFERENCE: that fallback cannot tell an
    aggregate bounded by shared x8 links from another route adding up to a
    multiple of one x8 link; that is why ``link_raw`` exists.]
    """
    if row.link_raw is not None:
        return _near(row.link_raw, X8_GBPS)
    th = row.theoretical
    if not th or _near(th, X16_GBPS):
        return False
    return _near(th, X8_GBPS) or (row.n_flows > 1 and _near(th, row.n_flows * X8_GBPS))


def row_baseline(row: EffRow) -> BaselineDiff | None:
    """Compare ``row`` with the baseline of its case (None: no baseline).

    The table follows the link width of the route: with ``EffRow.link_raw`` an
    x8 route (within 5 % of 15.754) uses ``INTERIM_X8_BASELINE`` only, an x16
    route (within 5 % of 31.508) the audit table, any other width has no
    baseline. Without ``link_raw`` the theoretical value decides (see
    ``is_x8_limited``; no theoretical value: audit table). Cases a table does
    not list give None, never a verdict from the other one.
    """
    key = _row_key(row)
    if key is None:
        return None
    if row.link_raw is not None and not (
        _near(row.link_raw, X8_GBPS) or _near(row.link_raw, X16_GBPS)
    ):
        return None
    if is_x8_limited(row):
        key = f"{key}.{X8_QUALIFIER}"
        table = INTERIM_X8_BASELINE
    else:
        table = AUDIT_BASELINE
    base = table.get(key)
    if base is None or base.unit != row.unit:
        return None
    return compare_to_baseline(key, row.value, row.unit, table=table)


def _is_bandwidth(row: EffRow) -> bool:
    return row.unit == UNIT_GBPS


def _pairs(row: EffRow) -> int:
    return row.n_flows // 2 if row.direction == BIDIR else row.n_flows


# --- rules -------------------------------------------------------------------
# Each rule takes the rows and returns its findings in row order; _RULES lists
# them in the order the findings are reported.


# Below this size a push is bound by latency, not by the link (4 KiB reaches
# ~9 GB/s on par1-cs25); the audit's bandwidth reference size is 1 MiB.
BW_MIN_SIZE = 1 << 20


def _rule_push_below_ceiling(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if not (
            row.kind == "push"
            and _is_bandwidth(row)
            and row.n_flows == 1
            and row.direction != BIDIR
            and row.theoretical
            and row.size is not None
            and row.size >= BW_MIN_SIZE
        ):
            continue
        ceiling = FabricModel.ceiling(row.theoretical)
        eff = efficiency(row.value, ceiling)
        if eff is not None and eff < PUSH_MIN_OF_CEILING:
            out.append(
                Finding(
                    "warn",
                    "push-below-ceiling",
                    f"{row.label}: single-flow push {row.value:.2f} GB/s is "
                    f"{eff:.0%} of the practical ceiling {ceiling:.2f} GB/s "
                    f"(raw {row.theoretical:.2f} GB/s); expected at least "
                    f"{PUSH_MIN_OF_CEILING:.0%}",
                    (row.label,),
                )
            )
    return out


def _rule_cross_root_pull(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if not (
            row.kind == "pull"
            and row.path_class == "cross-root"
            and row.n_flows == 1
            and _is_bandwidth(row)
        ):
            continue
        intra = [
            r.value
            for r in rows
            if r.kind == "pull"
            and r.path_class == "intra-switch"
            and r.n_flows == 1
            and r.size == row.size
            and _is_bandwidth(r)
        ]
        if not intra or row.value >= PULL_CROSS_MAX_OF_INTRA * max(intra):
            continue
        diff = row_baseline(row)
        known = diff is not None and diff.verdict == "ok"
        out.append(
            Finding(
                "info" if known else "bad",
                "cross-root-pull-penalty",
                f"{row.label}: cross-root pull {row.value:.2f} GB/s is "
                f"{row.value / max(intra):.0%} of the intra-switch pull "
                f"{max(intra):.2f} GB/s at the same size"
                + (
                    "; matches the known audit value (never pull across roots)"
                    if known
                    else "; does not match the audit baseline"
                ),
                (row.label,),
            )
        )
    return out


def _rule_cross_root_collapse(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if not (
            row.kind == "push"
            and row.path_class == "cross-root"
            and _is_bandwidth(row)
            and _pairs(row) >= COLLAPSE_MIN_FLOWS
        ):
            continue
        singles = [
            r.value
            for r in rows
            if r.kind == "push"
            and r.path_class == "cross-root"
            and _is_bandwidth(r)
            and r.size == row.size
            and (r.direction == BIDIR) == (row.direction == BIDIR)
            and _pairs(r) == 1
        ]
        if not singles or row.value >= COLLAPSE_MAX_OF_SINGLE * max(singles):
            continue
        diff = row_baseline(row)
        known = diff is not None and diff.verdict == "ok"
        out.append(
            Finding(
                "info" if known else "bad",
                "cross-root-collapse",
                f"{row.label}: {_pairs(row)} concurrent cross-root pairs give "
                f"{row.value:.2f} GB/s, {row.value / max(singles):.0%} of one "
                f"pair ({max(singles):.2f} GB/s)"
                + (
                    "; matches the known audit value (use at most 2 pairs)"
                    if known
                    else "; does not match the audit baseline"
                ),
                (row.label,),
            )
        )
    return out


def _rule_exceeds_theoretical(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if (
            _is_bandwidth(row)
            and row.theoretical
            and row.value > EXCEEDS_THEORETICAL * row.theoretical
        ):
            out.append(
                Finding(
                    "warn",
                    "exceeds-theoretical",
                    f"{row.label}: {row.value:.2f} GB/s is more than "
                    f"{EXCEEDS_THEORETICAL - 1:.0%} above the raw capacity "
                    f"{row.theoretical:.2f} GB/s: the model or the link state "
                    "is wrong",
                    (row.label,),
                )
            )
    return out


def _rule_latency(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if row.unit != UNIT_US:
            continue
        diff = row_baseline(row)
        if diff is not None and row.value > LATENCY_FACTOR * diff.baseline:
            out.append(
                Finding(
                    "warn",
                    "latency-above-baseline",
                    f"{row.label}: {row.value:.2f} us is {diff.ratio:.2f}x the "
                    f"audit baseline {diff.baseline:.2f} us ({diff.key}); "
                    f"limit {LATENCY_FACTOR:.2f}x",
                    (row.label,),
                )
            )
    return out


def _rule_host_underused(rows: Sequence[EffRow]) -> list[Finding]:
    out: list[Finding] = []
    for row in rows:
        if not (row.kind == "host" and _is_bandwidth(row) and row.theoretical):
            continue
        eff = efficiency(row.value, row.theoretical)
        if eff is not None and eff < HOST_MIN_OF_RAW:
            out.append(
                Finding(
                    "warn",
                    "host-link-underused",
                    f"{row.label}: host transfer {row.value:.2f} GB/s is "
                    f"{eff:.0%} of the raw link {row.theoretical:.2f} GB/s; "
                    f"expected at least {HOST_MIN_OF_RAW:.0%}",
                    (row.label,),
                )
            )
    return out


_RULES = (
    _rule_push_below_ceiling,
    _rule_cross_root_pull,
    _rule_cross_root_collapse,
    _rule_exceeds_theoretical,
    _rule_latency,
    _rule_host_underused,
)


def _norm_entity(token: str) -> str:
    t = token.strip().lower()
    return t if ":" in t else gpu_entity(t)


def _link_findings(
    rows: Sequence[EffRow],
    model: FabricModel | None,
    degraded_links: Mapping[str, str],
    suspended: Collection[str],
) -> list[Finding]:
    asleep = {_norm_entity(s) for s in suspended}
    out: list[Finding] = []
    for key in sorted(degraded_links):
        gpu: str | None = None
        if key.startswith("gpu:"):
            gpu = _norm_entity(key)
        elif model is not None and key in model.topo.links:
            owner = model.topo.links[key].gpu
            gpu = None if owner is None else _norm_entity(owner)
        labels = tuple(
            r.label
            for r in rows
            if gpu is not None and gpu in (_norm_entity(r.src), _norm_entity(r.dst))
        )
        text = degraded_links[key]
        if gpu is not None and gpu in asleep:
            out.append(
                Finding(
                    "info",
                    "link-suspended",
                    f"{key}: link {text}; the GPU is runtime-suspended, it "
                    "reports a 2.5 GT/s x1 link until it wakes",
                    labels,
                )
            )
        else:
            out.append(
                Finding(
                    "bad",
                    "link-downtrained",
                    f"{key}: link {text}; measured bandwidth is limited by "
                    "the downtrained link",
                    labels,
                )
            )
    return out


def findings(
    rows: Sequence[EffRow],
    model: FabricModel | None = None,
    *,
    degraded_links: Mapping[str, str] | None = None,
    suspended: Collection[str] = (),
) -> list[Finding]:
    """Return the findings of the rows, rules in the order of ``_RULES``.

    ``degraded_links`` maps a link or GPU entity to a text such as
    ``'8 GT/s x16 (max 16 GT/s x16)'``, supplied by the caller. Each entry is a
    ``link-downtrained`` (bad) finding, except when its GPU is in
    ``suspended`` (entity ids): a runtime-suspended GPU reports a 2.5 GT/s x1
    link until it wakes, that is an info finding ``link-suspended``. A link
    entity is attributed to its GPU through ``model.topo.links``.
    """
    out: list[Finding] = []
    for rule in _RULES:
        out.extend(rule(rows))
    out.extend(_link_findings(rows, model, degraded_links or {}, suspended))
    return out
