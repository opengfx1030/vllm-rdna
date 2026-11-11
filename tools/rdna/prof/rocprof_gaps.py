# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find where the GPU waits, from rocprofv3 CSV traces of a serving window.

    python3 tools/rdna/prof/rocprof_gaps.py TRACE_DIR [--min-gap-us 50] [--top 25]

TRACE_DIR is a ``rocprofv3 --kernel-trace --hip-runtime-trace
--memory-copy-trace --output-format csv`` output (attach mode writes one file
set per process). For every GPU agent it reports busy fraction, the idle-gap
histogram, the kernel pairs that bracket the most idle time, and the blocking
host HIP calls (copies, syncs, queries) that overlap those gaps. Per process
it reports the host API time by function, so a per-step host wait shows up as
a large ``hipMemcpy*``/``*Synchronize`` share on the scheduling thread.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import os
from collections import Counter, defaultdict
from collections.abc import Iterator

import regex as re


def col(row: dict, *names: str) -> str:
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    raise KeyError(names)


def short(name: str, n: int = 60) -> str:
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"<.*", "", name)
    return name[-n:]


def read_csvs(trace_dir: str, suffix: str) -> Iterator[tuple[str, dict]]:
    for path in glob.glob(os.path.join(trace_dir, "**", f"*{suffix}"), recursive=True):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                yield path, r


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def report_steps(
    items: list[tuple[int, int, str, str]], marker: re.Pattern, decode_max_ms: float
) -> None:
    """Split the timeline at the step-marker kernel (one per engine step) and
    report decode-only vs mixed/prefill steps: period, kernels, busy share,
    inter-kernel gap buckets, and the gap from step end to the next kernel
    (near zero when the next step is already queued, i.e. async scheduling)."""
    idx = [i for i, it in enumerate(items) if marker.search(it[2])]
    if len(idx) < 3:
        print(f"steps: marker {marker.pattern!r} found {len(idx)} times; skipped")
        return
    rows = []
    boundary = []
    for a, b in zip(idx, idx[1:]):
        seg = items[a + 1 : b + 1]
        period = items[b][1] - items[a][1]
        busy = sum(e - s for s, e in merge([(s, e) for s, e, _, _ in seg]))
        gaps = [max(0, seg[i + 1][0] - seg[i][1]) for i in range(len(seg) - 1)]
        buckets = [
            sum(g for g in gaps if lo <= g < hi)
            for lo, hi in ((0, 10_000), (10_000, 50_000), (50_000, 1 << 62))
        ]
        rows.append((period, len(seg), busy, buckets, Counter(it[2] for it in seg)))
        boundary.append(max(0, items[a + 1][0] - items[a][1]))
    boundary.sort()
    nb = len(boundary)
    print(
        f"step boundary (marker end -> next kernel): med "
        f"{boundary[nb // 2] / 1e3:.1f}us p90 {boundary[int(nb * 0.9)] / 1e3:.1f}us "
        f"p99 {boundary[min(nb - 1, int(nb * 0.99))] / 1e3:.1f}us"
    )
    for label, keep in (
        ("decode-only", lambda r: r[0] < decode_max_ms * 1e6),
        ("mixed/prefill", lambda r: r[0] >= decode_max_ms * 1e6),
    ):
        sel = [r for r in rows if keep(r)]
        if not sel:
            continue
        n = len(sel)
        periods = sorted(r[0] for r in sel)
        period = sum(periods) / n
        busy = sum(r[2] for r in sel) / n
        g = [sum(r[3][i] for r in sel) / n for i in range(3)]
        print(
            f"{label} steps={n} period mean {period / 1e6:.1f}ms "
            f"(med {periods[n // 2] / 1e6:.1f}, max {periods[-1] / 1e6:.1f}) "
            f"kernels/step {sum(r[1] for r in sel) / n:.0f} busy {busy / 1e6:.1f}ms "
            f"({100 * busy / period:.0f}%) gaps <10us {g[0] / 1e6:.2f}ms "
            f"10-50us {g[1] / 1e6:.2f}ms >50us {g[2] / 1e6:.2f}ms"
        )
        top = Counter()
        for r in sel:
            top.update(r[4])
        print(
            "  kernels/step: "
            + ", ".join(f"{k}:{v / n:.0f}" for k, v in top.most_common(8))
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--min-gap-us", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--window-ms",
        type=float,
        default=0.0,
        help="analyse only the last N ms before the final kernel (0 = everything)",
    )
    ap.add_argument(
        "--step-marker",
        default="get_num_sampled_and_rejected",
        help="regex of a kernel that runs once per engine step ('' disables)",
    )
    ap.add_argument("--decode-max-ms", type=float, default=80.0)
    a = ap.parse_args()

    lo = 0
    if a.window_ms > 0:
        t_end = 0
        for _, r in read_csvs(a.trace_dir, "kernel_trace.csv"):
            t_end = max(t_end, int(col(r, "End_Timestamp")))
        lo = t_end - int(a.window_ms * 1e6)

    # (agent) -> list of (start, end, label, stream)
    work: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    agent_pid: dict[str, str] = {}
    for path, r in read_csvs(a.trace_dir, "kernel_trace.csv"):
        s, e = int(col(r, "Start_Timestamp")), int(col(r, "End_Timestamp"))
        if s < lo:
            continue
        agent = col(r, "Agent_Id", "Agent_Node_Id")
        stream = r.get("Stream_Id") or r.get("Queue_Id") or "?"
        work[agent].append((s, e, short(col(r, "Kernel_Name")), stream))
        agent_pid[agent] = r.get("Process_Id") or os.path.basename(path).split("_")[0]
    for _, r in read_csvs(a.trace_dir, "memory_copy_trace.csv"):
        d = r.get("Direction", "")
        agent = (
            r.get("Destination_Agent_Id")
            if "HtoD" in d or "H2D" in d
            else r.get("Source_Agent_Id")
        )
        s, e = int(col(r, "Start_Timestamp")), int(col(r, "End_Timestamp"))
        if agent is None or s < lo:
            continue
        work[agent].append((s, e, f"COPY_{d}", "copy"))

    # Host API, per process: (start, end, function, thread)
    api: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for path, r in read_csvs(a.trace_dir, "hip_api_trace.csv"):
        s, e = int(col(r, "Start_Timestamp")), int(col(r, "End_Timestamp"))
        if e < lo:
            continue
        pid = r.get("Process_Id") or os.path.basename(path).split("_")[0]
        api[pid].append(
            (s, e, col(r, "Function", "Operation"), r.get("Thread_Id", "?"))
        )
    for v in api.values():
        v.sort()

    min_gap = int(a.min_gap_us * 1000)
    for agent in sorted(work, key=lambda x: int(x) if x.isdigit() else 0):
        items = sorted(work[agent])
        if len(items) < 100:
            continue
        t0, t1 = items[0][0], max(e for _, e, _, _ in items)
        busy = merge([(s, e) for s, e, _, _ in items])
        busy_ns = sum(e - s for s, e in busy)
        span = t1 - t0
        print(
            f"\n=== agent {agent} (pid {agent_pid.get(agent)}): {len(items)} ops, "
            f"window {span / 1e6:.1f} ms, busy {100 * busy_ns / span:.1f}%"
        )
        streams = Counter(st for _, _, _, st in items)
        print("streams:", dict(streams.most_common(6)))
        # concurrency across streams
        by_stream = defaultdict(list)
        for s, e, _, st in items:
            by_stream[st].append((s, e))
        if len(by_stream) > 1:
            merged = {k: merge(v) for k, v in by_stream.items()}
            total_overlap = 0
            keys = list(merged)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    A, Bv = merged[keys[i]], merged[keys[j]]
                    ia = ib = 0
                    while ia < len(A) and ib < len(Bv):
                        lo = max(A[ia][0], Bv[ib][0])
                        hi = min(A[ia][1], Bv[ib][1])
                        if hi > lo:
                            total_overlap += hi - lo
                        if A[ia][1] < Bv[ib][1]:
                            ia += 1
                        else:
                            ib += 1
            print(f"cross-stream overlap: {total_overlap / 1e6:.2f} ms")

        # gaps with bracketing ops
        ends_sorted = sorted(items, key=lambda x: x[1])
        gaps = []
        for (s0, e0), (s1, _) in zip(busy, busy[1:]):
            g = s1 - e0
            if g >= min_gap:
                gaps.append((e0, s1))
        hist = Counter()
        for gs, ge in gaps:
            g = (ge - gs) / 1000
            b = (
                "50-200us"
                if g < 200
                else "0.2-1ms"
                if g < 1000
                else "1-5ms"
                if g < 5000
                else ">5ms"
            )
            hist[b] += ge - gs
        idle = span - busy_ns
        print(
            f"idle {idle / 1e6:.1f} ms; gaps>={a.min_gap_us:.0f}us: "
            f"{len(gaps)} totalling "
            f"{sum(ge - gs for gs, ge in gaps) / 1e6:.1f} ms; by size (ms): "
            + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in sorted(hist.items()))
        )

        end_keys = [x[1] for x in ends_sorted]
        start_items = sorted(items)
        start_keys = [x[0] for x in start_items]
        pair_time: Counter = Counter()
        pair_n: Counter = Counter()
        api_in_gap: Counter = Counter()
        api_n: Counter = Counter()
        pid_api = api.get(agent_pid.get(agent, ""), [])
        api_starts = [x[0] for x in pid_api]
        for gs, ge in gaps:
            i = bisect.bisect_right(end_keys, gs) - 1
            j = bisect.bisect_left(start_keys, ge)
            prev = ends_sorted[i][2] if i >= 0 else "?"
            nxt = start_items[j][2] if j < len(start_items) else "?"
            pair_time[(prev, nxt)] += ge - gs
            pair_n[(prev, nxt)] += 1
            # host calls overlapping the gap (search back 50 ms for long calls)
            k = bisect.bisect_left(api_starts, gs - 50_000_000)
            seen = set()
            while k < len(pid_api) and pid_api[k][0] < ge:
                s, e, fn, th = pid_api[k]
                if e > gs:
                    api_in_gap[(fn, th)] += min(e, ge) - max(s, gs)
                    if (fn, th) not in seen:
                        seen.add((fn, th))
                        api_n[(fn, th)] += 1
                k += 1
        print("top bracketing pairs by idle time (prev -> next): ")
        for (p, n), t in pair_time.most_common(a.top):
            print(f"  {t / 1e6:8.2f} ms  n={pair_n[(p, n)]:5d}  {p}  ->  {n}")
        print("host HIP calls overlapping gaps (fn, thread): ")
        for (fn, th), t in api_in_gap.most_common(12):
            print(f"  {t / 1e6:8.2f} ms  n={api_n[(fn, th)]:5d}  {fn}  tid={th}")
        if a.step_marker:
            report_steps(items, re.compile(a.step_marker), a.decode_max_ms)

    print("\n=== host HIP API time per process (top functions)")
    for pid, v in api.items():
        tot = Counter()
        n = Counter()
        threads = Counter()
        for s, e, fn, th in v:
            tot[fn] += e - s
            n[fn] += 1
            threads[th] += 1
        if not v:
            continue
        span = (max(e for _, e, _, _ in v) - v[0][0]) / 1e6
        print(
            f"-- pid {pid}: {len(v)} calls over {span:.0f} ms, threads {len(threads)}"
        )
        for fn, t in tot.most_common(10):
            print(f"  {t / 1e6:9.2f} ms  n={n[fn]:7d}  {fn}")


if __name__ == "__main__":
    main()
