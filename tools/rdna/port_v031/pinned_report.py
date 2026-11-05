#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Which pinned mappings were moved or swapped when KFD evicted a worker.

    python3 pinned_report.py DIR [--kfd-events FILE] [--md OUT.md]

DIR is a pinned_sampler.py output (``samples.jsonl``, ``pagemap.jsonl``).
``--kfd-events`` is the log of ``kfd_smi_raw.py`` (raw QUEUE_EVICTION and
QUEUE_RESTORE records with their trigger); without it the evictions come from
the jumps of ``evicted_ms`` in the samples alone.

The Markdown report has:

  1. each worker's eviction total, VmLck and how much pinned memory is locked;
  2. the host's reclaim bursts (kswapd/direct scan, swap-out, compaction) and
     the reclaim totals of the run, then the weight loads of the followed
     workers (a worker whose resident memory grows like a checkpoint load) with
     the host reclaim during each and the evictions of other servers' workers
     that overlap it;
  3. per eviction burst, the raw KFD events, the weight loads in progress, the
     host reclaim around it and the mappings that moved or were swapped out
     while the eviction can have started, with their lock state;
  4. over the whole run, moved and swapped-out pages by mapping class and lock
     state, and the largest individual mappings.

An eviction of D ms seen between the samples at ta and tb ended in (ta, tb] and
started in [ta - D, tb]. A pagemap record (t, dt) says pages changed in
(t - dt, t]. They are related when those intervals overlap, within ``--slack``
seconds.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Per-second rate of a vmstat counter group from which a tick counts as part
# of a reclaim burst of that kind.
RECLAIM_RATES = {
    "scan": (("pgscan_kswapd", "pgscan_direct"), 20000),
    "swapout": (("pswpout",), 500),
    "compact": (("compact_migrate_scanned", "pgmigrate_success"), 50000),
    "stall": (("allocstall_normal", "allocstall_movable", "compact_stall"), 1),
}
# vmstat counters reported per burst and for the run.
RECLAIM_KEYS = (
    "pgscan_kswapd",
    "pgscan_direct",
    "pgsteal_kswapd",
    "pgsteal_direct",
    "pswpout",
    "pswpin",
    "compact_stall",
    "compact_migrate_scanned",
    "pgmigrate_success",
    "pgmigrate_fail",
    "allocstall_normal",
    "allocstall_movable",
    "workingset_refault_anon",
    "workingset_refault_file",
    "unevictable_pgs_culled",
    "unevictable_pgs_rescued",
)
GIB = 1 << 30
# A weight load: a worker whose resident memory adds at least LOAD_MIN_GIB, in
# intervals that each grow at LOAD_RATE_GIB_S or more, windows closer than
# LOAD_JOIN_S being one load. LOAD_SLACK_S is how far from a load an eviction
# still counts as concurrent.
LOAD_MIN_GIB = 4.0
LOAD_RATE_GIB_S = 0.05
LOAD_JOIN_S = 20.0
LOAD_SLACK_S = 10.0


@dataclass
class Jump:
    """evicted_ms of one worker grew by ``ms`` between two samples."""

    pid: str
    label: str
    ta: float
    tb: float
    ms: int


@dataclass
class KfdEvent:
    t: float
    kind: str  # evict | restore
    pid: int
    gpu: str
    trigger: str  # evict: the trigger code; restore: "R" if rescheduled


@dataclass
class Burst:
    """A run of consecutive ticks where one reclaim kind is above its rate."""

    kind: str
    t0: float
    t1: float
    total: int = 0
    ticks: int = 0
    peak_per_s: int = 0


@dataclass
class Load:
    """A worker whose resident memory grew like a weight load (GiB)."""

    pid: str
    label: str
    ppid: Any
    t0: float
    t1: float
    gib: float = 0.0
    shmem_gib: float = 0.0
    anon_gib: float = 0.0
    file_gib: float = 0.0
    peak_gib_s: float = 0.0


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list[Any]]) -> None:
        self.add("| " + " | ".join(header) + " |")
        self.add("| " + " | ".join("---" for _ in header) + " |")
        for row in rows:
            self.add("| " + " | ".join(str(c) for c in row) + " |")
        self.add()

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def load_samples(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    ticks = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                meta = row["meta"]
            else:
                ticks.append(row)
    return meta, ticks


def load_jsonl(path: str) -> list[dict[str, Any]]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def worker_label(title: str) -> str:
    """``VLLM::Worker_TP2_EP2`` -> ``TP2``."""
    for part in title.split("_"):
        if part.startswith(("TP", "PP", "DP")) and part[2:].isdigit():
            return part
    return title


def parse_kfd_events(path: str, start: float | None = None) -> list[KfdEvent]:
    """Events from the ``kfd_smi_raw.py`` log.

    Lines are ``HH:MM:SS.ffffff <bdf> <id hex> <ns> -<pid> <gpu hex> <arg>``;
    the first line is ``# start <iso> [...]``, which gives the date.
    """
    out = []
    day = datetime.fromtimestamp(start) if start else None
    with open(path) as f:
        for line in f:
            if line.startswith("# start"):
                with contextlib.suppress(IndexError, ValueError):
                    day = datetime.fromisoformat(line.split()[2])
                continue
            parts = line.split()
            if len(parts) < 6 or day is None:
                continue
            try:
                hh, mm, ss = parts[0].split(":")
                t = day.replace(
                    hour=int(hh), minute=int(mm), second=0, microsecond=0
                ).timestamp() + float(ss)
                ev = int(parts[2], 16)
                pid = abs(int(parts[4]))
            except ValueError:
                continue
            arg = parts[6] if len(parts) > 6 else ""
            if ev == 9 and arg:
                out.append(KfdEvent(t, "evict", pid, parts[5], arg))
            elif ev == 10:
                out.append(KfdEvent(t, "restore", pid, parts[5], arg))
    return out


def pair_events(events: list[KfdEvent]) -> list[tuple[KfdEvent, KfdEvent | None]]:
    """Each eviction with the next restore of the same process and GPU."""
    open_: dict[tuple[int, str], KfdEvent] = {}
    pairs: list[tuple[KfdEvent, KfdEvent | None]] = []
    for e in sorted(events, key=lambda e: e.t):
        key = (e.pid, e.gpu)
        if e.kind == "evict":
            if key in open_:
                pairs.append((open_[key], None))
            open_[key] = e
        elif key in open_:
            pairs.append((open_.pop(key), e))
    pairs.extend((e, None) for e in open_.values())
    return sorted(pairs, key=lambda p: p[0].t)


def eviction_jumps(ticks: list[dict[str, Any]]) -> list[Jump]:
    """Every growth of a worker's evicted_ms between two consecutive samples."""
    out = []
    last: dict[str, tuple[float, int]] = {}
    for tick in ticks:
        for pid, w in tick["w"].items():
            ms = w.get("evicted_ms")
            if ms is None:
                continue
            if pid in last and ms > last[pid][1]:
                label = worker_label(w["title"])
                out.append(Jump(pid, label, last[pid][0], tick["t"], ms - last[pid][1]))
            last[pid] = (tick["t"], ms)
    return out


def reclaim_bursts(ticks: list[dict[str, Any]]) -> list[Burst]:
    """Runs of consecutive ticks above the per-second rate of a reclaim kind."""
    out: list[Burst] = []
    open_: dict[str, Burst] = {}
    for a, b in zip(ticks, ticks[1:], strict=False):
        dt = max(b["t"] - a["t"], 1e-3)
        for kind, (keys, rate) in RECLAIM_RATES.items():
            d = sum(b["vm"].get(k, 0) - a["vm"].get(k, 0) for k in keys)
            per_s = d / dt
            if per_s >= rate:
                burst = open_.get(kind)
                if burst is None or a["t"] > burst.t1 + 1.5:
                    burst = open_[kind] = Burst(kind, a["t"], b["t"])
                    out.append(burst)
                burst.t1 = b["t"]
                burst.total += d
                burst.ticks += 1
                burst.peak_per_s = max(burst.peak_per_s, int(per_s))
    return out


def weight_loads(
    ticks: list[dict[str, Any]],
    min_gib: float = LOAD_MIN_GIB,
    rate_gib_s: float = LOAD_RATE_GIB_S,
    join_s: float = LOAD_JOIN_S,
) -> list[Load]:
    """Windows in which a followed worker's resident memory grew like a load.

    A booting worker reads its checkpoint (RssFile), allocates its pinned
    tables (RssShmem) and its anonymous memory; the sampler follows it from
    the moment its process title is set.
    """
    kib = 1 << 20  # kB per GiB
    series: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for tick in ticks:
        for pid, w in tick["w"].items():
            if w.get("rss_kb") is not None:
                series[pid].append((tick["t"], w))

    def grown(key: str, a: dict[str, Any], b: dict[str, Any]) -> float:
        return (b.get(key, 0) - a.get(key, 0)) / kib

    out: list[Load] = []
    for pid, rows in series.items():
        cur: Load | None = None
        base: dict[str, Any] = {}
        for (ta, a), (tb, b) in zip(rows, rows[1:], strict=False):
            rate = grown("rss_kb", a, b) / max(tb - ta, 1e-3)
            if rate < rate_gib_s:
                continue
            if cur is None or ta > cur.t1 + join_s:
                cur = Load(pid, worker_label(b["title"]), b.get("ppid"), ta, tb)
                base = a
                out.append(cur)
            cur.t1 = tb
            cur.peak_gib_s = max(cur.peak_gib_s, rate)
            cur.gib = grown("rss_kb", base, b)
            cur.shmem_gib = grown("rss_shmem_kb", base, b)
            cur.anon_gib = grown("rss_anon_kb", base, b)
            cur.file_gib = grown("rss_file_kb", base, b)
    return sorted((ld for ld in out if ld.gib >= min_gib), key=lambda ld: ld.t0)


def loads_near(
    loads: list[Load], t0: float, t1: float, slack: float = LOAD_SLACK_S
) -> list[Load]:
    """The weight loads that were running at some time in [t0, t1] (+- slack)."""
    return [ld for ld in loads if ld.t0 - slack <= t1 and ld.t1 + slack >= t0]


def vm_delta(ticks: list[dict[str, Any]], t0: float, t1: float) -> dict[str, int]:
    """Growth of the reclaim counters from the tick before t0 to the one after t1."""
    before = [t for t in ticks if t["t"] <= t0]
    after = [t for t in ticks if t["t"] >= t1]
    if not before or not after:
        return {}
    a, b = before[-1]["vm"], after[0]["vm"]
    return {k: b[k] - a[k] for k in RECLAIM_KEYS if k in a and k in b}


def overlaps(rec: dict[str, Any], t0: float, t1: float, slack: float) -> bool:
    return rec["t"] - rec["dt"] <= t1 + slack and rec["t"] >= t0 - slack


def lock_state(rec: dict[str, Any]) -> str:
    lk = rec.get("lock")
    if not lk or not lk.get("n"):
        return "unknown"
    return "locked" if lk["mlocked"] * 2 >= lk["n"] else "unlocked"


def fmt_t(t: float, base: float) -> str:
    return f"+{t - base:.1f}s"


def nonzero(d: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()) if v) or "all zero"


def section_workers(
    rep: Report, workers: dict[str, dict[str, Any]], jumps: list[Jump]
) -> None:
    rep.add("## 1. Evictions per worker")
    rep.add()
    rows = []
    for pid, info in sorted(
        workers.items(), key=lambda kv: (kv[1]["ppid"] or 0, kv[1]["label"])
    ):
        mine = [j for j in jumps if j.pid == pid]
        last = info["last"]
        maps = last.get("maps", {})
        gib = maps.get("pinned_gib_by_lock", {})
        rows.append(
            [
                pid,
                info["label"],
                info["ppid"],
                len(mine),
                sum(j.ms for j in mine),
                max((j.ms for j in mine), default=0),
                f"{(last.get('vmlck_kb') or 0) / 1048576:.2f}",
                "/".join(
                    f"{gib.get(k, 0):.2f}" for k in ("locked", "unlocked", "unknown")
                ),
                maps.get("pinned", {}).get("n", ""),
            ]
        )
    rep.table(
        [
            "pid",
            "worker",
            "server",
            "jumps",
            "evicted ms",
            "largest ms",
            "VmLck GiB",
            "pinned GiB locked/unlocked/unknown",
            "pinned mappings",
        ],
        rows,
    )


def section_reclaim(
    rep: Report, ticks: list[dict[str, Any]], bursts: list[Burst], t_first: float
) -> None:
    rep.add("## 2. Host reclaim")
    rep.add()
    totals = vm_delta(ticks, ticks[0]["t"], ticks[-1]["t"])
    rep.add("Totals over the run: " + nonzero(totals) + ".")
    rep.add()
    scan_k, scan_d = totals.get("pgscan_kswapd", 0), totals.get("pgscan_direct", 0)
    if scan_k + scan_d:
        rep.add(
            f"Direct reclaim is {100 * scan_d / (scan_k + scan_d):.0f}% of the pages "
            "scanned, kswapd the rest."
        )
        rep.add()
    free = [
        (t["vm"].get("nr_free_pages"), t["t"])
        for t in ticks
        if t["vm"].get("nr_free_pages") is not None
    ]
    if free:
        low = min(free)
        rep.add(
            f"Lowest MemFree {low[0] * 4096 / GIB:.2f} GiB at {fmt_t(low[1], t_first)}."
        )
        rep.add()
    psi = [(t["t"], t["psi"]) for t in ticks if t.get("psi")]
    if len(psi) > 1:
        rep.add(
            "Memory stall (PSI total over the run): "
            + ", ".join(
                f"{k} {(psi[-1][1][k] - psi[0][1][k]) / 1e6:.2f} s"
                for k in ("some", "full")
                if k in psi[0][1] and k in psi[-1][1]
            )
            + "."
        )
        rep.add()
    if bursts:
        rep.table(
            ["reclaim", "start", "end", "pages", "peak/s"],
            [
                [
                    b.kind,
                    fmt_t(b.t0, t_first),
                    fmt_t(b.t1, t_first),
                    b.total,
                    b.peak_per_s,
                ]
                for b in bursts
            ],
        )
    else:
        rates = {k: v[1] for k, v in RECLAIM_RATES.items()}
        rep.add(f"No reclaim burst above the per-second thresholds {rates}.")
        rep.add()


def section_loads(
    rep: Report,
    ticks: list[dict[str, Any]],
    loads: list[Load],
    jumps: list[Jump],
    ppid_of: dict[str, Any],
    t_first: float,
) -> None:
    rep.add("### Weight loads of the followed workers")
    rep.add()
    if not loads:
        rep.add(
            f"No worker grew by {LOAD_MIN_GIB:g} GiB or more at {LOAD_RATE_GIB_S:g} "
            "GiB/s or more."
        )
        rep.add()
        return
    rows = []
    hit = 0
    for ld in loads:
        victims = [
            j
            for j in jumps
            if ppid_of.get(j.pid) != ld.ppid
            and j.ta - j.ms / 1000 <= ld.t1 + LOAD_SLACK_S
            and j.tb >= ld.t0 - LOAD_SLACK_S
        ]
        hit += bool(victims)
        vm = vm_delta(ticks, ld.t0, ld.t1)
        rows.append(
            [
                ld.label,
                ld.pid,
                ld.ppid,
                f"{fmt_t(ld.t0, t_first)}..{fmt_t(ld.t1, t_first)}",
                f"{ld.gib:.1f}",
                f"{ld.shmem_gib:.1f}/{ld.anon_gib:.1f}/{ld.file_gib:.1f}",
                f"{ld.peak_gib_s:.2f}",
                f"{vm.get('pgscan_direct', 0)}/{vm.get('pgscan_kswapd', 0)}",
                vm.get("pswpout", 0),
                vm.get("pgmigrate_success", 0),
                sum(j.ms for j in victims),
                len({j.pid for j in victims}),
            ]
        )
    rep.table(
        [
            "loader",
            "pid",
            "server",
            "window",
            "RSS +GiB",
            "shmem/anon/file +GiB",
            "peak GiB/s",
            "scan direct/kswapd",
            "swap-out",
            "migrated",
            "evicted ms elsewhere",
            "workers",
        ],
        rows,
    )
    rep.add(
        f"{hit} of {len(loads)} weight loads overlap an eviction of a worker of "
        "another server (jumps ending within "
        f"{LOAD_SLACK_S:g} s of the load)."
    )
    rep.add()


def section_bursts(
    rep: Report,
    ticks: list[dict[str, Any]],
    jumps: list[Jump],
    bursts: list[Burst],
    pagemap: list[dict[str, Any]],
    pairs: list[tuple[KfdEvent, KfdEvent | None]],
    t_first: float,
    slack: float,
    loads: list[Load],
    ppid_of: dict[str, Any],
) -> None:
    rep.add("## 3. Eviction bursts and the mappings that moved")
    rep.add()
    merged: list[list[Jump]] = []
    for j in sorted(jumps, key=lambda j: j.tb):
        if merged and j.tb - merged[-1][-1].tb <= 3.0:
            merged[-1].append(j)
        else:
            merged.append([j])
    if not merged:
        rep.add("No jump of evicted_ms in the samples.")
        rep.add()
    for n, group in enumerate(merged, 1):
        g0 = min(j.ta - j.ms / 1000 for j in group)
        g1 = max(j.tb for j in group)
        near = sorted({b.kind for b in bursts if b.t0 - 5 <= g1 and b.t1 + 5 >= g0})
        rep.add(
            f"### Burst {n}: {fmt_t(g0, t_first)} to {fmt_t(g1, t_first)}, "
            f"{len(group)} worker jumps, reclaim near: {near or 'none'}"
        )
        rep.add()
        rep.table(
            ["worker", "pid", "ms", "ended between"],
            [
                [
                    j.label,
                    j.pid,
                    j.ms,
                    f"{fmt_t(j.ta, t_first)}..{fmt_t(j.tb, t_first)}",
                ]
                for j in group
            ],
        )
        servers = {ppid_of.get(j.pid) for j in group}
        running = loads_near(loads, g0, g1)
        if running:
            rep.add(
                "Weight loads in progress: "
                + "; ".join(
                    f"{ld.label} pid {ld.pid} (server {ld.ppid}, "
                    f"{'the same server' if ld.ppid in servers else 'neighbour'}) "
                    f"{fmt_t(ld.t0, t_first)}..{fmt_t(ld.t1, t_first)} "
                    f"+{ld.gib:.1f} GiB"
                    for ld in running
                )
                + "."
            )
        else:
            rep.add("No weight load of a followed worker in progress.")
        rep.add()
        raw = [
            (e, r)
            for e, r in pairs
            if g0 - slack <= e.t <= g1 + slack and str(e.pid) in {j.pid for j in group}
        ]
        if raw:
            rep.table(
                [
                    "pid",
                    "gpu",
                    "evicted at",
                    "trigger",
                    "restored after ms",
                    "rescheduled",
                ],
                [
                    [
                        e.pid,
                        e.gpu,
                        fmt_t(e.t, t_first),
                        e.trigger,
                        "" if r is None else round((r.t - e.t) * 1000),
                        "" if r is None else ("yes" if r.trigger == "R" else "no"),
                    ]
                    for e, r in raw
                ],
            )
        rep.add(
            "Host reclaim over the burst: "
            + nonzero(vm_delta(ticks, g0 - slack, g1 + slack))
            + "."
        )
        rep.add()
        sel: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0, 0])
        for rec in pagemap:
            if overlaps(rec, g0, g1, slack):
                key = (
                    rec["pid"],
                    rec["cls"],
                    lock_state(rec),
                    rec["start"],
                    rec["size"],
                )
                agg = sel[key]
                agg[0] += rec["moved"]
                agg[1] += rec["swapped_out"]
                agg[2] += rec["dropped"]
        if sel:
            rep.table(
                [
                    "pid",
                    "class",
                    "lock",
                    "mapping",
                    "MiB",
                    "moved",
                    "swapped out",
                    "dropped",
                ],
                [
                    [k[0], k[1], k[2], k[3], k[4] >> 20, *v]
                    for k, v in sorted(
                        sel.items(), key=lambda kv: -kv[1][0] - kv[1][1]
                    )[:15]
                ],
            )
        else:
            rep.add(
                "No page of a followed mapping moved or was swapped out in this burst."
            )
            rep.add()


def section_totals(rep: Report, pagemap: list[dict[str, Any]]) -> None:
    rep.add("## 4. Moved and swapped-out pages, whole run")
    rep.add()
    tot: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    by_map: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    for rec in pagemap:
        state = lock_state(rec)
        t = tot[(rec["cls"], state)]
        t[0] += rec["moved"]
        t[1] += rec["swapped_out"]
        t[2] += rec["dropped"]
        t[3] += 1
        m = by_map[(rec["pid"], rec["cls"], state, rec["start"], rec["size"])]
        m[0] += rec["moved"]
        m[1] += rec["swapped_out"]
    rep.table(
        ["class", "lock", "moved pages", "swapped-out pages", "dropped", "records"],
        [[c, ls, *v] for (c, ls), v in sorted(tot.items())],
    )
    if by_map:
        rep.add("Largest individual mappings:")
        rep.add()
        rep.table(
            ["pid", "class", "lock", "mapping", "MiB", "moved", "swapped out"],
            [
                [k[0], k[1], k[2], k[3], k[4] >> 20, *v]
                for k, v in sorted(by_map.items(), key=lambda kv: -kv[1][0] - kv[1][1])[
                    :15
                ]
            ],
        )


def build_report(
    meta: dict[str, Any],
    ticks: list[dict[str, Any]],
    pagemap: list[dict[str, Any]],
    events: list[KfdEvent],
    slack: float = 1.0,
) -> str:
    rep = Report()
    if not ticks:
        return "no samples\n"
    t_first, t_last = ticks[0]["t"], ticks[-1]["t"]
    rep.add(f"# Pinned host memory evictions ({t_last - t_first:.0f} s of samples)")
    rep.add()
    rep.add(
        f"Start {datetime.fromtimestamp(t_first).isoformat(timespec='seconds')}, "
        f"{len(ticks)} ticks, kernel {meta.get('kernel', '?')}, "
        f"sysctl {json.dumps(meta.get('sysctl', {}))}."
    )
    rep.add()
    cost = [t["tick_s"] for t in ticks if "tick_s" in t]
    if cost:
        rep.add(
            f"Sampler cost per tick: mean {1000 * sum(cost) / len(cost):.0f} ms, "
            f"max {1000 * max(cost):.0f} ms."
        )
        rep.add()
    workers: dict[str, dict[str, Any]] = {}
    for tick in ticks:
        for pid, w in tick["w"].items():
            info = workers.setdefault(pid, {"label": worker_label(w["title"])})
            info["ppid"] = w.get("ppid")
            info["last"] = w
    jumps = eviction_jumps(ticks)
    section_workers(rep, workers, jumps)
    pairs = pair_events(events)
    if pairs:
        trig: dict[str, int] = defaultdict(int)
        for e, _ in pairs:
            trig[e.trigger] += 1
        resched = sum(1 for _, r in pairs if r is not None and r.trigger == "R")
        rep.add(
            f"Raw KFD events: {len(pairs)} QUEUE_EVICTION by trigger "
            f"(0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND) {dict(trig)}, "
            f"{sum(1 for _, r in pairs if r is None)} without a restore, "
            f"{resched} rescheduled restores."
        )
        rep.add()
    bursts = reclaim_bursts(ticks)
    loads = weight_loads(ticks)
    ppid_of = {pid: info["ppid"] for pid, info in workers.items()}
    section_reclaim(rep, ticks, bursts, t_first)
    section_loads(rep, ticks, loads, jumps, ppid_of, t_first)
    section_bursts(
        rep, ticks, jumps, bursts, pagemap, pairs, t_first, slack, loads, ppid_of
    )
    section_totals(rep, pagemap)
    return rep.text()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir", help="pinned_sampler.py output directory")
    ap.add_argument("--kfd-events", help="kfd_smi_raw.py log")
    ap.add_argument("--slack", type=float, default=1.0, help="seconds")
    ap.add_argument("--md", help="write the report here instead of stdout")
    args = ap.parse_args(argv)
    meta, ticks = load_samples(f"{args.dir}/samples.jsonl")
    pagemap = load_jsonl(f"{args.dir}/pagemap.jsonl")
    events = (
        parse_kfd_events(args.kfd_events, meta.get("start")) if args.kfd_events else []
    )
    text = build_report(meta, ticks, pagemap, events, args.slack)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
