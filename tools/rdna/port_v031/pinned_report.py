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
     state, the largest individual mappings, and the seconds of the run by what
     moved in them (pinned pages, other pages, nothing) against the seconds a
     worker was under eviction.

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
    "compact_daemon_migrate_scanned",
    "compact_daemon_wake",
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
# Evictions of one burst lie within BURST_GAP_S of each other; RAW_ROWS raw
# evictions are listed per burst.
BURST_GAP_S = 3.0
RAW_ROWS = 12
# A never-restored eviction is a teardown when the worker left the samples within
# this many seconds of it.
EXIT_WITHIN_S = 30.0
# A kcompactd pass: a sampler tick in which the compaction daemon scanned at
# least PASS_SCAN_MIN pages. The wakes and kswapd's work are counted over the
# PASS_LOOKBACK_S before it.
PASS_SCAN_MIN = 1_000_000
PASS_LOOKBACK_S = 60.0


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
    bdf: str = ""  # PCI address of the GPU, 0000:ce:00.0

    @property
    def bus(self) -> str:
        """PCI bus of the GPU (``ce``), else its KFD id."""
        parts = self.bdf.split(":")
        return parts[1] if len(parts) > 2 else self.gpu


@dataclass
class Eviction:
    """The queues of one process on one GPU, evicted until the first real restore."""

    start: KfdEvent
    end: KfdEvent | None = None
    resched: int = 0  # restores that were rescheduled (arg R) in between
    again: int = 0  # evictions that came while the queues were still evicted


@dataclass
class EvBurst:
    """Counter jumps and raw evictions within BURST_GAP_S of each other."""

    t0: float
    t1: float
    jumps: list[Jump] = field(default_factory=list)
    evictions: list[Eviction] = field(default_factory=list)


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


def parse_worker_labels(paths: list[str]) -> dict[str, str]:
    """Map worker pids to ranks (TP0) using ``(Worker_TP0_EP0 pid=1234) ...`` lines."""
    out: dict[str, str] = {}
    for path in paths:
        with contextlib.suppress(OSError), open(path, errors="replace") as f:
            for line in f:
                head, _, rest = line.partition(" ")
                if head.startswith("(Worker_") and rest.startswith("pid="):
                    label = worker_label(head[1:])
                    if label != head[1:]:
                        out[rest[4:].split(")")[0]] = label
    return out


def apply_labels(ticks: list[dict[str, Any]], labels: dict[str, str]) -> None:
    """Give workers whose title has no rank (the sampler saw them before they set
    it) the rank their server logged."""
    for tick in ticks:
        for pid, w in tick["w"].items():
            if worker_label(w["title"]) == w["title"] and pid in labels:
                w["title"] = f"VLLM::Worker_{labels[pid]}"


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
                out.append(KfdEvent(t, "evict", pid, parts[5], arg, parts[1]))
            elif ev == 10:
                out.append(KfdEvent(t, "restore", pid, parts[5], arg, parts[1]))
    return out


def pair_events(events: list[KfdEvent]) -> list[Eviction]:
    """Each eviction of a process on a GPU with the restore that ended it.

    A restore with argument ``R`` was rescheduled and the queues stay evicted;
    the eviction ends with the first restore that is not.
    """
    open_: dict[tuple[int, str], Eviction] = {}
    done: list[Eviction] = []
    for e in sorted(events, key=lambda e: e.t):
        key = (e.pid, e.gpu)
        cur = open_.get(key)
        if e.kind == "evict":
            if cur is None:
                open_[key] = Eviction(e)
            else:
                cur.again += 1
        elif cur is not None:
            if e.trigger == "R":
                cur.resched += 1
            else:
                cur.end = e
                done.append(open_.pop(key))
    done.extend(open_.values())
    return sorted(done, key=lambda v: v.start.t)


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


def worker_exits(ticks: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Per worker that left: the last sample with it and the first without it."""
    out: dict[str, tuple[float, float]] = {}
    for a, b in zip(ticks, ticks[1:], strict=False):
        for pid in a["w"]:
            if pid not in b["w"] and pid not in out:
                out[pid] = (a["t"], b["t"])
    return out


def at_exit(v: Eviction, exits: dict[str, tuple[float, float]]) -> bool:
    """Never restored, and the worker left the samples within ``EXIT_WITHIN_S``.

    The queues of an exiting process are evicted when its memory is unmapped,
    which is a few seconds before its pid disappears from /proc.
    """
    gone = exits.get(str(v.start.pid))
    return (
        v.end is None
        and gone is not None
        and -1.0 <= gone[1] - v.start.t <= EXIT_WITHIN_S
    )


def eviction_bursts(
    jumps: list[Jump], evictions: list[Eviction], gap: float = BURST_GAP_S
) -> list[EvBurst]:
    """Counter jumps and raw evictions merged into bursts.

    A jump of D ms between samples at ta < tb is an eviction that ended in
    (ta, tb] after lasting D ms, so it lies within [ta - D, tb].
    """
    items: list[tuple[float, float, Jump | Eviction]] = [
        (j.ta - j.ms / 1000, j.tb, j) for j in jumps
    ]
    items += [(v.start.t, v.end.t if v.end else v.start.t, v) for v in evictions]
    out: list[EvBurst] = []
    for t0, t1, item in sorted(items, key=lambda it: it[0]):
        if out and t0 <= out[-1].t1 + gap:
            b = out[-1]
            b.t1 = max(b.t1, t1)
        else:
            b = EvBurst(t0, t1)
            out.append(b)
        if isinstance(item, Jump):
            b.jumps.append(item)
        else:
            b.evictions.append(item)
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


def kcompactd_passes(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ticks in which kcompactd scanned PASS_SCAN_MIN pages or more.

    Each carries the daemon's wakes and kswapd's passes and scan over the
    PASS_LOOKBACK_S before it, and ``since_kswapd``: seconds from the last tick in
    which kswapd ran a pass (None if it never did). A pass of the proactive path
    (``vm.compaction_proactiveness``) comes with kswapd idle; one that kswapd
    asked for follows its passes.
    """
    out = []
    for i in range(1, len(ticks)):
        a, b = ticks[i - 1]["vm"], ticks[i]["vm"]
        key = "compact_daemon_migrate_scanned"
        scanned = b.get(key, 0) - a.get(key, 0)
        if scanned < PASS_SCAN_MIN:
            continue
        t = ticks[i]["t"]
        j = i
        while j > 0 and t - ticks[j]["t"] < PASS_LOOKBACK_S:
            j -= 1
        before = ticks[j]["vm"]
        since = None
        for k in range(i, 0, -1):
            prev = ticks[k - 1]["vm"].get("pageoutrun", 0)
            if ticks[k]["vm"].get("pageoutrun", 0) > prev:
                since = t - ticks[k]["t"]
                break
        out.append(
            {
                "t": t,
                "scanned": scanned,
                "migrated": b.get("pgmigrate_success", 0)
                - a.get("pgmigrate_success", 0),
                "wakes": b.get("compact_daemon_wake", 0)
                - before.get("compact_daemon_wake", 0),
                "kswapd_passes": b.get("pageoutrun", 0) - before.get("pageoutrun", 0),
                "kswapd_scan": b.get("pgscan_kswapd", 0)
                - before.get("pgscan_kswapd", 0),
                "since_kswapd": since,
            }
        )
    return out


def window_reclaim_rows(
    ticks: list[dict[str, Any]], windows: list[dict[str, Any]]
) -> list[list[Any]]:
    """One row per proof window the ticks cover: its evictions, the host counters."""
    rows: list[list[Any]] = []
    for w in windows:
        delta = vm_delta(ticks, w["t0"], w["t1"])
        if not delta:
            continue
        free = [
            t["vm"]["nr_free_pages"]
            for t in ticks
            if w["t0"] <= t["t"] <= w["t1"] and "nr_free_pages" in t["vm"]
        ]
        evicted = w.get("evicted_ms") or {}
        rows.append(
            [
                w["name"],
                datetime.fromtimestamp(w["t0"]).strftime("%H:%M:%S"),
                ", ".join(
                    f"{worker_label(k)} {ms}" for k, ms in sorted(evicted.items())
                )
                or "0",
                human(delta.get("pgscan_kswapd", 0)),
                human(delta.get("pgscan_direct", 0)),
                delta.get("compact_stall", 0),
                human(delta.get("pgmigrate_success", 0)),
                human(delta.get("pswpout", 0)),
                human(delta.get("compact_daemon_migrate_scanned", 0)),
                f"{min(free) * 4096 / GIB:.1f}" if free else "?",
            ]
        )
    return rows


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
    rep: Report,
    ticks: list[dict[str, Any]],
    bursts: list[Burst],
    t_first: float,
    windows: list[dict[str, Any]] | None = None,
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
    window_rows = window_reclaim_rows(ticks, windows) if windows else []
    if window_rows:
        rep.add("Per proof window: evicted ms per worker and the host's counters.")
        rep.add()
        rep.table(
            [
                "window",
                "start",
                "evicted ms",
                "kswapd scan",
                "direct scan",
                "compaction stalls",
                "migrated",
                "swapped out",
                "kcompactd scan",
                "lowest MemFree GiB",
            ],
            window_rows,
        )
    passes = kcompactd_passes(ticks)
    if passes:
        rep.add(
            "kcompactd passes: ticks in which the daemon scanned "
            f"{human(PASS_SCAN_MIN)} pages or more, with its wakes and kswapd's "
            f"work over the {PASS_LOOKBACK_S:.0f} s before."
        )
        rep.add()
        rep.table(
            [
                "tick",
                "daemon scan",
                "migrated in tick",
                "wakes",
                "kswapd passes",
                "kswapd scan",
                "since last kswapd pass",
            ],
            [
                [
                    fmt_t(p["t"], t_first),
                    human(p["scanned"]),
                    human(p["migrated"]),
                    p["wakes"],
                    p["kswapd_passes"],
                    human(p["kswapd_scan"]),
                    "never"
                    if p["since_kswapd"] is None
                    else f"{p['since_kswapd']:.0f} s",
                ]
                for p in passes
            ],
        )
        quiet = [p for p in passes if p["kswapd_passes"] == 0]
        rep.add(
            f"{len(passes) - len(quiet)} of {len(passes)} passes follow kswapd passes "
            f"within {PASS_LOOKBACK_S:.0f} s; {len(quiet)} came with kswapd idle "
            "(where a proactive pass would show)."
        )
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


def burst_pids(b: EvBurst) -> list[str]:
    return sorted({j.pid for j in b.jumps} | {str(v.start.pid) for v in b.evictions})


def is_teardown(
    b: EvBurst,
    workers: dict[str, dict[str, Any]],
    exits: dict[str, tuple[float, float]],
) -> bool:
    """A teardown: no counter jump, nothing was ever restored, and every followed
    worker left the samples within EXIT_WITHIN_S of its eviction."""
    followed = [p for p in burst_pids(b) if p in workers]
    return (
        bool(followed)
        and not b.jumps
        and all(
            at_exit(v, exits) if str(v.start.pid) in workers else v.end is None
            for v in b.evictions
        )
    )


def human(n: int) -> str:
    """2713924 -> 2.71M, 44007 -> 44k."""
    for unit, size in (("M", 10**6), ("k", 10**3)):
        if abs(n) >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


def window_label(t: float, windows: list[dict[str, Any]]) -> str:
    """The window of the proof's hook (idle1, load1, ...) that contains ``t``."""
    for w in windows:
        if w["t0"] <= t <= w["t1"]:
            if "t_end" not in w:
                return f"{w['name']} (idle)"
            phase = "neighbour running" if t <= w["t_end"] else "settling"
            return f"{w['name']} ({phase})"
    return "gap"


def burst_summary(
    rep: Report,
    ticks: list[dict[str, Any]],
    bursts: list[EvBurst],
    reclaim: list[Burst],
    pagemap: list[dict[str, Any]],
    t_first: float,
    slack: float,
    loads: list[Load],
    workers: dict[str, dict[str, Any]],
    exits: dict[str, tuple[float, float]],
    windows: list[dict[str, Any]] | None = None,
) -> None:
    """One row per burst: who was evicted, what the host did, which pages moved."""
    rows = []
    for n, b in enumerate(bursts, 1):
        pids = burst_pids(b)
        followed = [p for p in pids if p in workers]
        if is_teardown(b, workers, exits):
            who = f"teardown of {len(followed)} worker(s)"
        elif followed:
            who = ", ".join(
                f"{workers[p].get('label', p)} "
                f"{sum(j.ms for j in b.jumps if j.pid == p)} ms"
                for p in followed
            )
        else:
            who = "not followed"
        near = sorted(
            {r.kind for r in reclaim if r.t0 - 5 <= b.t1 and r.t1 + 5 >= b.t0}
        )
        vm = vm_delta(ticks, b.t0 - slack, b.t1 + slack)
        wakes = vm.get("compact_daemon_wake", 0)
        host = (
            f"scan {human(vm.get('pgscan_kswapd', 0))}/"
            f"{human(vm.get('pgscan_direct', 0))}, "
            f"migrated {human(vm.get('pgmigrate_success', 0))}, "
            f"compaction scan {human(vm.get('compact_migrate_scanned', 0))} "
            f"(kcompactd {human(vm.get('compact_daemon_migrate_scanned', 0))}, "
            f"{wakes} wake{'' if wakes == 1 else 's'}), "
            f"stalls {vm.get('compact_stall', 0)}, "
            f"swap-out {human(vm.get('pswpout', 0))}"
            if vm
            else ""
        )
        pages: dict[tuple[str, bool], list[int]] = defaultdict(lambda: [0, 0])
        for rec in pagemap:
            if overlaps(rec, b.t0, b.t1, slack):
                agg = pages[(rec["cls"], lock_state(rec) == "locked")]
                agg[0] += rec["moved"]
                agg[1] += rec["swapped_out"]

        def cell(cls: str, locked: bool, pages: dict = pages) -> str:
            moved, swapped = pages.get((cls, locked), (0, 0))
            return f"{moved} / {swapped}"

        servers = {workers.get(p, {}).get("ppid") for p in pids}
        running = "; ".join(
            f"{'same server' if ld.ppid in servers else 'neighbour'} {ld.gib:.0f} GiB"
            for ld in loads_near(loads, b.t0, b.t1)
        )
        row = [
            n,
            f"{fmt_t(b.t0, t_first)}..{fmt_t(b.t1, t_first)}",
            who,
            ", ".join(near) or "none",
            host,
            cell("pinned", False),
            cell("pinned", True),
            cell("shm", False),
            cell("anon", False),
            running or "none",
        ]
        if windows is not None:
            row.append(window_label(b.t0, windows))
        rows.append(row)
    note = (
        "Pages are moved / swapped out, in the records that overlap each burst "
        "(+- the slack); host counters are kswapd/direct pages scanned, pages "
        "migrated, pages scanned for migration by compaction (of them by the "
        "kcompactd daemon, and how often it was woken), direct compaction "
        "stalls and pages swapped out."
    )
    if any(is_teardown(b, workers, exits) for b in bursts):
        note += " A teardown is a worker exiting, not reclaim."
    rep.add(note)
    rep.add()
    rep.table(
        [
            "burst",
            "window",
            "evicted (counter)",
            "reclaim near",
            "host counters",
            "pinned unlocked",
            "pinned locked",
            "shm",
            "anon",
            "weight load",
        ]
        + (["proof window"] if windows is not None else []),
        rows,
    )


def section_bursts(
    rep: Report,
    ticks: list[dict[str, Any]],
    bursts: list[EvBurst],
    reclaim: list[Burst],
    pagemap: list[dict[str, Any]],
    t_first: float,
    slack: float,
    loads: list[Load],
    workers: dict[str, dict[str, Any]],
    exits: dict[str, tuple[float, float]],
    windows: list[dict[str, Any]] | None = None,
) -> None:
    rep.add("## 3. Eviction bursts and the mappings that moved")
    rep.add()
    server_label = {
        str(w["ppid"]): "server of the workers"
        for w in workers.values()
        if w.get("ppid")
    }
    if not bursts:
        rep.add("No jump of evicted_ms and no raw QUEUE_EVICTION event.")
        rep.add()
    if bursts:
        burst_summary(
            rep,
            ticks,
            bursts,
            reclaim,
            pagemap,
            t_first,
            slack,
            loads,
            workers,
            exits,
            windows,
        )
    for n, b in enumerate(bursts, 1):
        pids = burst_pids(b)
        followed = [p for p in pids if p in workers]
        teardown = is_teardown(b, workers, exits)
        near = sorted(
            {r.kind for r in reclaim if r.t0 - 5 <= b.t1 and r.t1 + 5 >= b.t0}
        )
        others = len(pids) - len(followed)
        what = (
            f"teardown: {len(followed)} worker(s) evicted as they exited"
            + (f", {others} other process(es) with them" if others else "")
            if teardown
            else f"{len(pids)} process(es)"
        )
        rep.add(
            f"### Burst {n}: {fmt_t(b.t0, t_first)} to {fmt_t(b.t1, t_first)}, "
            f"{what}, reclaim near: {near or 'none'}"
        )
        rep.add()
        rows = []
        for pid in pids:
            info = workers.get(pid, {})
            js = [j for j in b.jumps if j.pid == pid]
            vs = [v for v in b.evictions if str(v.start.pid) == pid]
            longest = [round((v.end.t - v.start.t) * 1000) for v in vs if v.end]
            rows.append(
                [
                    pid,
                    info.get("label", server_label.get(pid, "not followed")),
                    info.get("ppid", ""),
                    sum(j.ms for j in js) if js else "",
                    f"{len(vs)} on {len({v.start.gpu for v in vs})} GPUs" if vs else "",
                    "/".join(sorted({v.start.trigger for v in vs})),
                    max(longest, default=""),
                    sum(v.resched for v in vs) if vs else "",
                    sum(1 for v in vs if v.end is None) if vs else "",
                    f"+{exits[pid][1] - t_first:.1f}s" if pid in exits else "",
                ]
            )
        rep.table(
            [
                "pid",
                "worker",
                "server",
                "evicted ms (counter)",
                "raw evictions",
                "trigger",
                "longest restore ms",
                "rescheduled restores",
                "not restored",
                "left the samples",
            ],
            rows,
        )
        servers = {workers.get(p, {}).get("ppid") for p in pids}
        running = loads_near(loads, b.t0, b.t1)
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
        if b.evictions:
            shown = b.evictions[:RAW_ROWS]
            rep.table(
                [
                    "pid",
                    "GPU bus",
                    "evicted at",
                    "trigger",
                    "restored after ms",
                    "rescheduled",
                    "re-evicted",
                ],
                [
                    [
                        v.start.pid,
                        v.start.bus,
                        fmt_t(v.start.t, t_first),
                        v.start.trigger,
                        "" if v.end is None else round((v.end.t - v.start.t) * 1000),
                        v.resched,
                        v.again,
                    ]
                    for v in shown
                ],
            )
            if len(b.evictions) > len(shown):
                rep.add(f"... and {len(b.evictions) - len(shown)} more raw evictions.")
                rep.add()
        rep.add(
            "Host counters over the burst: "
            + nonzero(vm_delta(ticks, b.t0 - slack, b.t1 + slack))
            + " (unevictable_pgs_rescued is the locked pages of an exiting process)."
        )
        rep.add()
        sel: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0, 0])
        for rec in pagemap:
            if overlaps(rec, b.t0, b.t1, slack):
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


def move_seconds(
    ticks: list[dict[str, Any]], jumps: list[Jump], pagemap: list[dict[str, Any]]
) -> list[list[Any]]:
    """Seconds of the run by what moved in them, and how many had an eviction.

    A pagemap record (t, dt) says pages moved or were swapped out in
    (t - dt, t]: its whole-second bins count. A counter jump of ms between the
    samples at ta and tb is an eviction that ended in (ta, tb] and began up to
    ms before: the bins from ta - ms to tb count as under eviction. Pages
    dropped (unmapped, as at a worker's exit) are not moves.
    """
    pinned: set[int] = set()
    other: set[int] = set()
    for rec in pagemap:
        if not rec["moved"] + rec["swapped_out"]:
            continue
        bins = range(int(rec["t"] - rec["dt"]), int(rec["t"]) + 1)
        (pinned if rec["cls"] == "pinned" else other).update(bins)
    evicted: set[int] = set()
    for jump in jumps:
        evicted.update(range(int(jump.ta - jump.ms / 1000), int(jump.tb) + 1))
    seconds = range(int(ticks[0]["t"]), int(ticks[-1]["t"]) + 1)
    groups = (
        ("pinned pages moved or swapped out", [s for s in seconds if s in pinned]),
        (
            "only other pages moved or swapped out",
            [s for s in seconds if s not in pinned and s in other],
        ),
        (
            "nothing moved",
            [s for s in seconds if s not in pinned and s not in other],
        ),
    )
    return [
        [label, len(secs), sum(1 for s in secs if s in evicted)]
        for label, secs in groups
    ]


def section_totals(
    rep: Report,
    pagemap: list[dict[str, Any]],
    ticks: list[dict[str, Any]],
    jumps: list[Jump],
) -> None:
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
    if pagemap:
        rep.add()
        rep.add(
            "Seconds of the run by what moved in them, and how many of them had a "
            "worker under eviction (counter jumps, back to where each began):"
        )
        rep.add()
        rep.table(
            ["seconds in which", "seconds", "with a worker under eviction"],
            [
                [label, n, f"{e} ({100 * e / n:.0f} %)" if n else "0"]
                for label, n, e in move_seconds(ticks, jumps, pagemap)
            ],
        )


def build_report(
    meta: dict[str, Any],
    ticks: list[dict[str, Any]],
    pagemap: list[dict[str, Any]],
    events: list[KfdEvent],
    slack: float = 1.0,
    windows: list[dict[str, Any]] | None = None,
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
            info = workers.setdefault(pid, {})
            info["label"] = worker_label(w["title"])  # the rank may come late
            info["ppid"] = w.get("ppid")
            if w.get("vmlck_kb") is not None or "last" not in info:  # not a dying one
                info["last"] = w
    jumps = eviction_jumps(ticks)
    section_workers(rep, workers, jumps)
    evictions = pair_events(events)
    exits = worker_exits(ticks)
    if evictions:
        trig: dict[str, int] = defaultdict(int)
        for v in evictions:
            trig[v.start.trigger] += 1
        never = [v for v in evictions if v.end is None]
        rep.add(
            f"Raw KFD events: {len(evictions)} QUEUE_EVICTION (one per process and "
            f"GPU) by trigger (0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND) {dict(trig)}; "
            f"{len(evictions) - len(never)} restored, {len(never)} not restored "
            f"({sum(1 for v in never if at_exit(v, exits))} of them at a worker "
            f"exit); {sum(v.resched for v in evictions)} restores were rescheduled "
            "(the queues stay evicted until a later one) and "
            f"{sum(v.again for v in evictions)} evictions came while the queues "
            "were still evicted."
        )
        rep.add()
    reclaim = reclaim_bursts(ticks)
    loads = weight_loads(ticks)
    ppid_of = {pid: info["ppid"] for pid, info in workers.items()}
    section_reclaim(rep, ticks, reclaim, t_first, windows)
    section_loads(rep, ticks, loads, jumps, ppid_of, t_first)
    section_bursts(
        rep,
        ticks,
        eviction_bursts(jumps, evictions),
        reclaim,
        pagemap,
        t_first,
        slack,
        loads,
        workers,
        exits,
        windows,
    )
    section_totals(rep, pagemap, ticks, jumps)
    return rep.text()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir", help="pinned_sampler.py output directory")
    ap.add_argument("--kfd-events", help="kfd_smi_raw.py log")
    ap.add_argument("--slack", type=float, default=1.0, help="seconds")
    ap.add_argument(
        "--serve-log",
        action="append",
        default=[],
        help="vLLM server log: its '(Worker_TP0_EP0 pid=N)' lines name the workers "
        "whose recorded title lacks the rank (repeatable)",
    )
    ap.add_argument(
        "--windows",
        help="the proof hook's windows.json: bursts are labelled with the window "
        "(idle, neighbour load running, settling, gap) they start in",
    )
    ap.add_argument("--md", help="write the report here instead of stdout")
    args = ap.parse_args(argv)
    meta, ticks = load_samples(f"{args.dir}/samples.jsonl")
    pagemap = load_jsonl(f"{args.dir}/pagemap.jsonl")
    events = (
        parse_kfd_events(args.kfd_events, meta.get("start")) if args.kfd_events else []
    )
    apply_labels(ticks, parse_worker_labels(args.serve_log))
    windows = None
    if args.windows:
        with open(args.windows) as f:
            windows = json.load(f)["windows"]
    text = build_report(meta, ticks, pagemap, events, args.slack, windows)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
