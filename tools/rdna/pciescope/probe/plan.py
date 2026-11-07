# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurement plans: which ``p2p_probe`` commands a suite runs on which GPUs.

A plan is a list of ``Step`` objects. Each step is ONE probe command (batch
mode line) plus the metadata needed to turn its rows into ``Measurement``
records. Device ids in the commands are *logical*: the index of the GPU in
``Plan.buses`` (= the order of ``HIP_VISIBLE_DEVICES``, ascending HIP index).

Suites (``SUITES``): ``quick`` (about 30 s: one pair per path class, host
transfers, a few concurrency patterns), ``standard`` (a few minutes: latency
and bandwidth curves, concurrency sweeps, host concurrency, one-shot all-to-all,
key cells repeated), ``full`` (all pairs, everything repeated).

A step that needs GPUs the selection does not contain (no second switch, fewer
than 3 GPUs ...) is not planned; the reason is listed in ``Plan.skipped``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pciescope.model import Topology

from .gpus import gpu_records
from .ids import gpu_entity

KIB = 1 << 10
MIB = 1 << 20

LAT_SIZES = (
    4,
    64,
    256,
    KIB,
    4 * KIB,
    16 * KIB,
    64 * KIB,
    256 * KIB,
    MIB,
    4 * MIB,
    16 * MIB,
    64 * MIB,
)
BW_SIZES = (4 * KIB, 64 * KIB, MIB, 5 * MIB, 64 * MIB)
HOST_BYTES = 64 * MIB
BW_BLOCKS = 128
STARTUP_S = 4.0  # one-time allocation + calibration of the probe process
MAX_A2A_RANKS = 8  # kMaxG of p2p_probe
LAT_PAIRS_PER_STEP = 8  # pairs per latency command: a failure stays local
BATCH_LINE_MAX = 4000  # p2p_probe batch reads lines of at most 4095 characters

#: probe step kinds (see Step.kind)
KINDS = ("bw", "hbw", "lat", "latc", "a2a", "curve", "hlat")


@dataclass(frozen=True)
class SuiteInfo:
    """Suite name, one line of description, typical duration in seconds."""

    name: str
    description: str
    est_seconds: int


SUITES: dict[str, SuiteInfo] = {
    "quick": SuiteInfo(
        "quick",
        "one pair per path class (latency, push/pull/sdma bandwidth), host "
        "transfers, ring/fan-in/cross-root concurrency",
        30,
    ),
    "standard": SuiteInfo(
        "standard",
        "quick plus latency and bandwidth curves (4 B..64 MiB), concurrency "
        "sweeps, host concurrency, one-shot all-to-all; key cells x3",
        300,
    ),
    "full": SuiteInfo(
        "full",
        "standard plus every GPU pair (latency, push/pull/sdma, bidirectional) "
        "and every host link; key cells x3",
        900,
    ),
}


def list_suites() -> list[SuiteInfo]:
    """Return the suites in order of cost."""
    return list(SUITES.values())


@dataclass(frozen=True)
class Step:
    """One probe command and what its rows mean.

    ``kind``: bw | hbw | lat | latc | a2a | curve | hlat (selects the parser
    rows and the Measurement conversion). ``pattern`` names a concurrency
    pattern (ring, fan-in, cross-k3 ...) and is empty for isolated flows.
    ``key`` marks the cells that ``--repeats`` runs several times.
    """

    id: str
    cmd: str
    kind: str
    scenario: str
    est_s: float
    pattern: str = ""
    key: bool = False
    tags: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cmd": self.cmd,
            "kind": self.kind,
            "scenario": self.scenario,
            "est_s": round(self.est_s, 2),
            "pattern": self.pattern,
            "key": self.key,
            "tags": dict(self.tags),
        }


@dataclass
class Plan:
    """The commands of one suite on one GPU selection."""

    suite: str
    buses: tuple[str, ...]  # logical id -> bus
    hip: tuple[int, ...]  # logical id -> HIP index
    arch: tuple[str, ...]
    switch: tuple[str, ...]  # logical id -> 'A' | 'B'
    steps: list[Step] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    repeats: int = 1

    @property
    def entities(self) -> list[str]:
        """``gpu:<bus>`` of every logical id."""
        return [gpu_entity(b) for b in self.buses]

    @property
    def hip_visible(self) -> str:
        """Value of ``HIP_VISIBLE_DEVICES`` for the probe process."""
        return ",".join(str(i) for i in self.hip)

    @property
    def needs_peer(self) -> bool:
        """True when any step uses GPU-to-GPU transfers (not host-only)."""
        return any(s.kind not in ("hbw", "hlat") for s in self.steps)

    @property
    def est_total_s(self) -> float:
        """Estimated wall time including the probe start-up."""
        return STARTUP_S + 0.5 * len(self.buses) + sum(s.est_s for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "buses": list(self.buses),
            "hip_visible_devices": self.hip_visible,
            "arch": list(self.arch),
            "switch": list(self.switch),
            "repeats": self.repeats,
            "needs_peer": self.needs_peer,
            "est_total_s": round(self.est_total_s, 1),
            "steps": [s.to_dict() for s in self.steps],
            "skipped": [{"what": w, "why": y} for w, y in self.skipped],
        }


# ---------------------------------------------------------------------------
# planning context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ctx:
    """The selected GPUs by logical id, grouped by switch."""

    buses: tuple[str, ...]
    switch: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]  # switch letter -> logical ids

    @property
    def n(self) -> int:
        return len(self.buses)

    def label(self, lid: int) -> str:
        return gpu_entity(self.buses[lid])

    def big_groups(self, minimum: int = 2) -> list[tuple[int, ...]]:
        """Switch groups with at least ``minimum`` GPUs (letter order)."""
        return [g for _, g in sorted(self.groups.items()) if len(g) >= minimum]

    def cross_pairs(self, k: int) -> list[tuple[int, int]]:
        """``k`` disjoint pairs from the first GPUs of two different switches."""
        ordered = [g for _, g in sorted(self.groups.items())]
        if len(ordered) < 2:
            return []
        a, b = ordered[0], ordered[1]
        k = min(k, len(a), len(b))
        return [(a[i], b[i]) for i in range(k)]


class _Builder:
    """Collects steps, skips duplicates and records what could not be planned."""

    def __init__(self, ctx: Ctx, suite: str) -> None:
        self.ctx = ctx
        self.suite = suite
        self.steps: list[Step] = []
        self.skipped: list[tuple[str, str]] = []
        self._seen: set[str] = set()

    def skip(self, what: str, why: str) -> None:
        if (what, why) not in self.skipped:
            self.skipped.append((what, why))

    def add(
        self,
        kind: str,
        cmd: str,
        scenario: str,
        est_s: float,
        *,
        pattern: str = "",
        key: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        if cmd in self._seen:
            return
        self._seen.add(cmd)
        sid = f"{self.suite}/{len(self.steps) + 1:03d}-{kind}"
        self.steps.append(
            Step(
                sid,
                cmd,
                kind,
                scenario,
                est_s,
                pattern,
                key,
                tuple(sorted((tags or {}).items())),
            )
        )

    # -- command helpers ----------------------------------------------------

    def bw(
        self,
        size: int,
        xfers: Sequence[str],
        scenario: str,
        *,
        dur: float = 0.4,
        pattern: str = "",
        key: bool = False,
    ) -> None:
        cmd = f"bw {dur:g} {size} {BW_BLOCKS} {' '.join(xfers)}"
        self.add("bw", cmd, scenario, dur + 0.3, pattern=pattern, key=key)

    def hbw(
        self,
        size: int,
        xfers: Sequence[str],
        scenario: str,
        *,
        dur: float = 0.4,
        pattern: str = "",
        key: bool = False,
    ) -> None:
        cmd = f"hbw {dur:g} {size} {BW_BLOCKS} {' '.join(xfers)}"
        self.add("hbw", cmd, scenario, dur + 0.4, pattern=pattern, key=key)

    def lat(
        self,
        payload: int,
        pairs: Sequence[tuple[int, int]],
        scenario: str,
        *,
        concurrent: bool = False,
        key: bool = False,
    ) -> None:
        verb = "latc" if concurrent else "lat"
        arg = " ".join(f"{a}:{b}" for a, b in pairs)
        est = 0.3 if concurrent else 0.2 + 0.15 * len(pairs)
        self.add(
            verb,
            f"{verb} 2000 {payload} {arg}",
            scenario,
            est,
            pattern="concurrent-pairs" if concurrent else "",
            key=key,
        )


def make_ctx(buses: Sequence[str], topo: Topology) -> tuple[Ctx, list[int], list[str]]:
    """Order ``buses`` by HIP index; return the context, HIP ids and archs."""
    gpus = gpu_records(buses, topo)
    groups: dict[str, list[int]] = {}
    for lid, g in enumerate(gpus):
        groups.setdefault(g.switch or "?", []).append(lid)
    ctx = Ctx(
        buses=tuple(g.bus for g in gpus),
        switch=tuple(g.switch or "?" for g in gpus),
        groups={k: tuple(v) for k, v in groups.items()},
    )
    hips = [g.hip for g in gpus if g.hip is not None]
    return ctx, hips, [g.arch for g in gpus]


# ---------------------------------------------------------------------------
# step groups (each takes the builder and a repeat/level flag)
# ---------------------------------------------------------------------------


def _host_gpus(ctx: Ctx, per_switch: int = 1) -> list[int]:
    out: list[int] = []
    for _, g in sorted(ctx.groups.items()):
        out += list(g[:per_switch])
    return out


def _host_basic(b: _Builder) -> None:
    for lid in _host_gpus(b.ctx):
        e = b.ctx.label(lid)
        for mode, text in (
            ("h2d", "H2D pinned"),
            ("d2h", "D2H pinned"),
            ("h2d_pg", "H2D pageable"),
            ("d2h_pg", "D2H pageable"),
        ):
            b.hbw(
                HOST_BYTES,
                [f"{mode}:{lid}"],
                f"{text} {HOST_BYTES // MIB} MiB {e}",
                key=True,
            )
        b.hbw(
            HOST_BYTES,
            [f"h2d:{lid}", f"d2h:{lid}"],
            f"H2D + D2H pinned at once {HOST_BYTES // MIB} MiB {e}",
            pattern="bidir-host",
        )


def _host_concurrency(b: _Builder, sweep: bool) -> None:
    ctx = b.ctx
    groups = ctx.big_groups(2)
    if not groups:
        b.skip("host concurrency", "no switch with 2 or more selected GPUs")
        return
    g0 = groups[0]
    ks = range(2, len(g0) + 1) if sweep else (len(g0),)
    for k in ks:
        for mode, text in (("h2d", "H2D"), ("d2h", "D2H")):
            b.hbw(
                HOST_BYTES,
                [f"{mode}:{lid}" for lid in g0[:k]],
                f"{k} GPUs of switch {ctx.switch[g0[0]]} {text} at once",
                dur=0.5,
                pattern=f"host-k{k}",
            )
    if sweep:
        b.hbw(
            HOST_BYTES,
            [f"{m}:{lid}" for lid in g0 for m in ("h2d", "d2h")],
            f"{len(g0)} GPUs of switch {ctx.switch[g0[0]]} H2D + D2H at once",
            dur=0.5,
            pattern="host-bidir",
        )
    if len(ctx.groups) >= 2:
        firsts = _host_gpus(ctx, per_switch=1)
        seconds = _host_gpus(ctx, per_switch=2)
        for ids, text in ((firsts, "one GPU per switch"), (seconds, "two per switch")):
            if len(ids) < 2 or (ids is seconds and not sweep):
                continue
            b.hbw(
                HOST_BYTES,
                [f"h2d:{lid}" for lid in ids],
                f"H2D {text} at once",
                dur=0.5,
                pattern="host-both-switches",
            )
    else:
        b.skip("host H2D on both switches at once", "only one switch selected")


def _pair_cells(b: _Builder, level: str) -> list[tuple[str, tuple[int, int]]]:
    """(class, pair) cells: the first intra-switch pair and the first cross pair."""
    ctx = b.ctx
    cells: list[tuple[str, tuple[int, int]]] = []
    intra = ctx.big_groups(2)
    if intra:
        cells.append(("intra-switch", (intra[0][0], intra[0][1])))
        if level != "quick" and len(intra) > 1:
            cells.append(("intra-switch", (intra[1][0], intra[1][1])))
    else:
        b.skip("intra-switch pair", "no switch with 2 or more selected GPUs")
    cross = ctx.cross_pairs(1)
    if cross:
        cells.append(("cross-root", cross[0]))
    else:
        b.skip("cross-root pair", "selection has GPUs on one switch only")
    return cells


def _d2d_lat_quick(b: _Builder, cells: list[tuple[str, tuple[int, int]]]) -> None:
    pairs = [p for _, p in cells]
    if not pairs:
        return
    for payload in (0, 4 * KIB):
        what = "flag only" if payload == 0 else f"{payload // KIB} KiB payload"
        b.lat(payload, pairs, f"one-way push+flag latency, {what}", key=True)


def _pairs_text(k: int) -> str:
    return "1 pair" if k == 1 else f"{k} pairs"


def _d2d_bw_cells(
    b: _Builder,
    cells: list[tuple[str, tuple[int, int]]],
    sizes: Sequence[int],
    *,
    plateau: bool = False,
) -> None:
    """Push / pull / sdma per cell and size; ``plateau`` adds sdma at 64 MiB."""
    for cls, (a, c) in cells:
        ea, ec = b.ctx.label(a), b.ctx.label(c)
        for size in sizes:
            sz = _fmt_size(size)
            b.bw(size, [f"p:{a}>{c}"], f"push {sz} {cls} {ea}>{ec}", key=True)
            b.bw(size, [f"l:{a}>{c}"], f"pull {sz} {cls} {ea}>{ec}", key=True)
            b.bw(size, [f"s:{a}>{c}"], f"sdma {sz} {cls} {ea}>{ec}", key=True)
        if plateau and 64 * MIB not in sizes:
            b.bw(
                64 * MIB,
                [f"s:{a}>{c}"],
                f"sdma {_fmt_size(64 * MIB)} {cls} {ea}>{ec}",
                key=True,
            )


def _ring(group: Sequence[int]) -> list[str]:
    n = len(group)
    return [f"p:{group[i]}>{group[(i + 1) % n]}" for i in range(n)]


def _concurrency_quick(b: _Builder) -> None:
    ctx = b.ctx
    intra = ctx.big_groups(3)
    if intra:
        g = intra[0][:4]
        b.bw(
            MIB,
            _ring(g),
            f"ring of {len(g)} inside switch {ctx.switch[g[0]]}",
            dur=0.5,
            pattern="ring-intra",
            key=True,
        )
        b.bw(
            MIB,
            [f"p:{x}>{g[0]}" for x in g[1:]],
            f"fan-in {len(g) - 1} GPUs into {ctx.label(g[0])}",
            dur=0.5,
            pattern="fan-in",
            key=True,
        )
    else:
        b.skip("ring / fan-in", "no switch with 3 or more selected GPUs")
    k = min(3, *(len(g) for g in ctx.groups.values())) if len(ctx.groups) > 1 else 0
    if k >= 3:
        pairs = ctx.cross_pairs(3)
        b.bw(
            MIB,
            [f"p:{a}>{c}" for a, c in pairs],
            "3 concurrent cross-root pairs",
            dur=0.5,
            pattern="cross-k3",
            key=True,
        )
    else:
        b.skip("3 concurrent cross-root pairs", "needs 3 GPUs on each of two switches")


def _concurrency_sweep(b: _Builder, everything: bool) -> None:
    ctx = b.ctx
    groups = ctx.big_groups(2)
    ordered = [g for _, g in sorted(ctx.groups.items())]
    # cross-root flow count scan, both directions and bidirectional
    if len(ordered) >= 2:
        kmax = min(4, len(ordered[0]), len(ordered[1]))
        for size in (64 * KIB, MIB):
            for k in range(1, kmax + 1):
                pairs = ctx.cross_pairs(k)
                ab = [f"p:{a}>{c}" for a, c in pairs]
                ba = [f"p:{c}>{a}" for a, c in pairs]
                sz = _fmt_size(size)
                b.bw(
                    size,
                    ab,
                    f"{_pairs_text(k)} concurrent cross-root {sz}, A to B",
                    dur=0.5,
                    pattern=f"cross-k{k}",
                    key=size == MIB,
                )
                if size == MIB:
                    b.bw(
                        size,
                        ba,
                        f"{_pairs_text(k)} concurrent cross-root {sz}, B to A",
                        dur=0.5,
                        pattern=f"cross-k{k}-rev",
                    )
                    b.bw(
                        size,
                        ab + ba,
                        f"{_pairs_text(k)} cross-root {sz}, both directions",
                        dur=0.5,
                        pattern=f"cross-k{k}-bidir",
                        key=True,
                    )
        # fan-in / fan-out across the root link
        a_grp, b_grp = ordered[0], ordered[1]
        if len(a_grp) >= 2:
            b.bw(
                MIB,
                [f"p:{x}>{b_grp[0]}" for x in a_grp],
                f"fan-in {len(a_grp)} GPUs of one switch into one of the other",
                dur=0.5,
                pattern="fan-in-cross",
            )
        if len(b_grp) >= 2:
            b.bw(
                MIB,
                [f"p:{a_grp[0]}>{x}" for x in b_grp],
                f"fan-out one GPU into {len(b_grp)} GPUs of the other switch",
                dur=0.5,
                pattern="fan-out-cross",
            )
        # does intra-switch background traffic slow a cross-root flow
        fg = ctx.cross_pairs(1)
        rest_a, rest_b = a_grp[1:], b_grp[1:]
        bg: list[str] = []
        for g in (rest_a, rest_b):
            if len(g) == 2:
                bg += [f"bg:p:{g[0]}>{g[1]}", f"bg:p:{g[1]}>{g[0]}"]
            elif len(g) > 2:
                bg += [f"bg:{x}" for x in _ring(g)]
        if fg and bg:
            b.bw(
                MIB,
                [f"p:{a}>{c}" for a, c in fg] + bg,
                "1 cross-root pair with intra-switch rings on the other GPUs",
                dur=0.5,
                pattern="interference",
            )
    else:
        b.skip("cross-root concurrency", "selection has GPUs on one switch only")
    # inside one switch: ring, all-to-all, fan-in, fan-out, bidirectional pair
    for g in groups:
        g = g[:4]
        sw = ctx.switch[g[0]]
        if len(g) >= 3:
            b.bw(
                MIB,
                _ring(g),
                f"ring of {len(g)} inside switch {sw}",
                dur=0.5,
                pattern="ring-intra",
                key=True,
            )
            b.bw(
                MIB,
                [f"p:{x}>{y}" for x in g for y in g if x != y],
                f"all-to-all of {len(g)} GPUs inside switch {sw}",
                dur=0.5,
                pattern="a2a-intra",
            )
            b.bw(
                MIB,
                [f"p:{x}>{g[0]}" for x in g[1:]],
                f"fan-in {len(g) - 1} GPUs into {ctx.label(g[0])}",
                dur=0.5,
                pattern="fan-in",
                key=True,
            )
            b.bw(
                MIB,
                [f"p:{g[0]}>{x}" for x in g[1:]],
                f"fan-out {ctx.label(g[0])} into {len(g) - 1} GPUs",
                dur=0.5,
                pattern="fan-out",
            )
        b.bw(
            MIB,
            [f"p:{g[0]}>{g[1]}", f"p:{g[1]}>{g[0]}"],
            f"bidirectional pair {ctx.label(g[0])}<>{ctx.label(g[1])}",
            dur=0.5,
            pattern="bidir-pair",
            key=True,
        )
        if not everything:
            break
    if len(groups) >= 2:
        both = [x for g in groups for x in _ring(g[:4]) if len(g) >= 3]
        if len(both) >= 6:
            b.bw(
                MIB,
                both,
                "rings of both switches at once",
                dur=0.5,
                pattern="ring-both",
            )


def _lat_concurrency(b: _Builder) -> None:
    ctx = b.ctx
    cross = ctx.cross_pairs(4)
    for k in range(2, len(cross) + 1):
        b.lat(
            4 * KIB,
            cross[:k],
            f"{_pairs_text(k)} cross-root ping-pong at once",
            concurrent=True,
        )
    for g in ctx.big_groups(4):
        b.lat(
            4 * KIB,
            [(g[0], g[1]), (g[2], g[3])],
            f"2 intra-switch pairs ping-pong at once (switch {ctx.switch[g[0]]})",
            concurrent=True,
        )
        break


def _curves(b: _Builder, cells: list[tuple[str, tuple[int, int]]]) -> None:
    csv = ",".join(str(s) for s in LAT_SIZES)
    seen: set[str] = set()
    for cls, (a, c) in cells:
        if cls in seen:
            continue
        seen.add(cls)
        ea, ec = b.ctx.label(a), b.ctx.label(c)
        for mode, text in (
            ("push", "push+flag one-way"),
            ("pull", "kernel read"),
            ("sdma", "hipMemcpyPeerAsync"),
        ):
            b.add(
                "curve",
                f"curve {mode} 1000 0 {csv} {a}:{c}",
                f"{text} latency vs size {cls} {ea}>{ec}",
                0.3 * len(LAT_SIZES) + 0.5,
                key=True,
            )
    for lid in _host_gpus(b.ctx):
        b.add(
            "hlat",
            f"hlat 500 {lid} {' '.join(str(s) for s in LAT_SIZES)}",
            f"host<->device copy latency vs size {b.ctx.label(lid)}",
            4 * 0.25 * len(LAT_SIZES) + 1.0,
            key=True,
        )


def _a2a(b: _Builder) -> None:
    ctx = b.ctx
    for payload in (4 * KIB, 16 * KIB, 64 * KIB):
        for g in ctx.big_groups(2):
            g = g[:MAX_A2A_RANKS]
            b.add(
                "a2a",
                f"a2a 2000 {payload} {','.join(str(x) for x in g)}",
                f"one-shot all-to-all {_fmt_size(payload)}, {len(g)} GPUs of "
                f"switch {ctx.switch[g[0]]}",
                0.5,
                pattern=f"a2a-{len(g)}",
            )
        if len(ctx.groups) >= 2 and 2 < ctx.n <= MAX_A2A_RANKS:
            b.add(
                "a2a",
                f"a2a 2000 {payload} {','.join(str(x) for x in range(ctx.n))}",
                f"one-shot all-to-all {_fmt_size(payload)}, all {ctx.n} GPUs",
                0.5,
                pattern=f"a2a-{ctx.n}-flat",
            )


def _all_pairs(b: _Builder) -> None:
    ctx = b.ctx
    pairs = list(itertools.combinations(range(ctx.n), 2))
    if not pairs:
        return
    for payload in (0, 4 * KIB, 64 * KIB):
        what = "flag only" if payload == 0 else _fmt_size(payload) + " payload"
        for i in range(0, len(pairs), LAT_PAIRS_PER_STEP):
            chunk = pairs[i : i + LAT_PAIRS_PER_STEP]
            b.lat(
                payload,
                chunk,
                f"one-way push+flag latency, {what}, pairs "
                f"{i + 1}-{i + len(chunk)} of {len(pairs)}",
                key=True,
            )
    for a, c in itertools.permutations(range(ctx.n), 2):
        e = f"{ctx.label(a)}>{ctx.label(c)}"
        b.bw(MIB, [f"p:{a}>{c}"], f"push 1 MiB {e}", key=True)
        b.bw(MIB, [f"l:{a}>{c}"], f"pull 1 MiB {e}", key=True)
        b.bw(MIB, [f"s:{a}>{c}"], f"sdma 1 MiB {e}", key=True)
    for a, c in pairs:
        b.bw(
            MIB,
            [f"p:{a}>{c}", f"p:{c}>{a}"],
            f"bidirectional pair {ctx.label(a)}<>{ctx.label(c)}",
            dur=0.5,
            pattern="bidir-pair",
        )
    for lid in range(ctx.n):
        for mode in ("h2d", "d2h"):
            b.hbw(
                HOST_BYTES,
                [f"{mode}:{lid}"],
                f"{mode.upper()} pinned {HOST_BYTES // MIB} MiB {ctx.label(lid)}",
                key=True,
            )


def _fmt_size(n: int) -> str:
    if n >= MIB and n % MIB == 0:
        return f"{n // MIB} MiB"
    if n >= KIB and n % KIB == 0:
        return f"{n // KIB} KiB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------


def _suite_quick(b: _Builder) -> None:
    cells = _pair_cells(b, "quick")
    _host_basic(b)
    _d2d_lat_quick(b, cells)
    sizes = (MIB,)
    _d2d_bw_cells(b, cells, sizes, plateau=True)
    _concurrency_quick(b)
    _host_concurrency(b, sweep=False)


def _suite_standard(b: _Builder) -> None:
    cells = _pair_cells(b, "standard")
    _host_basic(b)
    _d2d_lat_quick(b, cells)
    _d2d_bw_cells(b, cells, BW_SIZES)
    _concurrency_quick(b)
    _concurrency_sweep(b, everything=False)
    _host_concurrency(b, sweep=True)
    _lat_concurrency(b)
    _curves(b, cells)
    _a2a(b)


def _suite_full(b: _Builder) -> None:
    _suite_standard(b)
    _concurrency_sweep(b, everything=True)
    _all_pairs(b)


_BUILDERS: dict[str, Callable[[_Builder], None]] = {
    "quick": _suite_quick,
    "standard": _suite_standard,
    "full": _suite_full,
}

#: repeats used when the caller does not choose
DEFAULT_REPEATS = {"quick": 1, "standard": 3, "full": 3}


def make_plan(
    suite: str,
    buses: Sequence[str],
    topo: Topology,
    *,
    repeats: int | None = None,
) -> Plan:
    """Build the plan of ``suite`` for ``buses`` (any order, HIP order is used).

    Raises:
        ValueError: unknown suite or a GPU without a HIP index.

    """
    if suite not in _BUILDERS:
        raise ValueError(f"unknown suite {suite!r}; choose {', '.join(SUITES)}")
    ctx, hips, arch = make_ctx(buses, topo)
    builder = _Builder(ctx, suite)
    _BUILDERS[suite](builder)
    reps = DEFAULT_REPEATS[suite] if repeats is None else max(1, repeats)
    plan = Plan(
        suite=suite,
        buses=ctx.buses,
        hip=tuple(hips),
        arch=tuple(arch),
        switch=ctx.switch,
        steps=_expand_repeats(builder.steps, reps),
        skipped=builder.skipped,
        repeats=reps,
    )
    if not plan.steps:
        raise ValueError(f"suite {suite} plans no step for {' '.join(ctx.buses)}")
    too_long = [s.id for s in plan.steps if len(s.cmd) >= BATCH_LINE_MAX]
    if too_long:
        raise ValueError(
            f"command of step {too_long[0]} exceeds the {BATCH_LINE_MAX} characters "
            "a batch line may have"
        )
    return plan


def _expand_repeats(steps: Sequence[Step], repeats: int) -> list[Step]:
    """Run every ``key`` step ``repeats`` times (one full pass per repeat).

    Key steps carry ``rep`` and ``reps`` tags when ``repeats`` > 1; the other
    steps run once.
    """
    if repeats <= 1:
        return list(steps)
    out: list[Step] = []
    for rep in range(1, repeats + 1):
        for s in steps:
            if rep == 1 or s.key:
                tags = s.tags
                if s.key:
                    tags += (("rep", str(rep)), ("reps", str(repeats)))
                sid = s.id if rep == 1 else f"{s.id}#{rep}"
                out.append(
                    Step(
                        sid, s.cmd, s.kind, s.scenario, s.est_s, s.pattern, s.key, tags
                    )
                )
    return out
