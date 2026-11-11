# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bubble analysis of the expert-offload decode path from a torch profiler trace.

usage: offload_gaps.py TRACE.json[.gz] | ROCPROF_CSV_DIR [--marker REGEX] [--k2 REGEX]
                       [--top N] [--json OUT]

TRACE comes from ``serve_profile.sh PROFILER=torch`` (kernels, HIP runtime
events and Python stacks; the torch branch wires in the kineto shim), or pass a
``serve_profile.sh`` rocprofv3 ``trace`` directory (``*_kernel_trace.csv`` +
``*_hip_api_trace.csv`` per process; there are no Python stacks then, so host
calls are listed by API name only; the analysis runs on the process with the most
K2 stage kernels). Steps are the intervals between ``--marker`` kernels. In every
step the offloaded K2 is a chain per MoE layer: stage 1 (prefix, route ids) ->
host drain and expert-miss copies -> stage 2 (one launch per expert pass) ->
stage 3 (finish). For each layer this reports the GPU idle time between the
stages (the bubble), the host calls that block inside it (hipMemcpyAsync D2H,
hipStreamSynchronize, hipEventSynchronize, ...) with the repo frames that issued
them, and the expert-miss copies that run inside it.
"""

import argparse
import bisect
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import regex as re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_sites import GPU_CATS, Intervals, frame_label, load  # noqa: E402

BLOCKING = re.compile(
    r"Synchronize|hipMemcpy|hipMemcpyAsync|EventQuery|hipStreamWaitEvent|hipStreamQuery|hipFree|hipHostFree"
)
COPY = re.compile(r"__amd_rocclr_copyBuffer|Memcpy|memcpy")
DRAIN = re.compile(
    r"hipMemcpyWithStream|hipStreamSynchronize|hipDeviceSynchronize|hipMemcpyDtoH|hipMemcpy$|cudaStreamSynchronize|cudaMemcpy$"
)
H2D = re.compile(r"HtoD|HOST_TO_DEVICE")


def events_from_rocprof(trace_dir, k2_rx, proc=0):
    """Chrome-trace-like events from the rocprofv3 CSVs of the proc-th process.

    proc is 0-based; 0 = the process with the most K2 stage kernels.
    """
    import csv
    import glob

    cands = []
    for kf in glob.glob(
        os.path.join(trace_dir, "**", "*_kernel_trace.csv"), recursive=True
    ):
        n = 0
        with open(kf, newline="") as f:
            for r in csv.DictReader(f):
                if k2_rx.search(r["Kernel_Name"]):
                    n += 1
        cands.append((n, kf))
    cands.sort(key=lambda c: (-c[0], c[1]))
    if proc >= len(cands):
        return []
    best = cands[proc]
    kf = best[1]
    af = kf.replace("_kernel_trace.csv", "_hip_api_trace.csv")
    print(f"rocprof process files: {kf} ({best[0]} K2 stage kernels), {af}")
    ev = []
    keep = (
        set()
    )  # correlation ids whose HIP API call is needed: copies and K2 stage launches
    with open(kf, newline="") as f:
        for r in csv.DictReader(f):
            st, en = int(r["Start_Timestamp"]) / 1e3, int(r["End_Timestamp"]) / 1e3
            name = r["Kernel_Name"]
            c = int(r["Correlation_Id"])
            ev.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "ts": st,
                    "dur": en - st,
                    "tid": r["Queue_Id"],
                    "args": {"correlation": c},
                }
            )
            if COPY.search(name) or k2_rx.search(name):
                keep.add(c)
    mf = kf.replace("_kernel_trace.csv", "_memory_copy_trace.csv")
    if os.path.exists(
        mf
    ):  # SDMA copies (expert miss uploads, slot map) are not kernel dispatches
        n = 0
        with open(mf, newline="") as f:
            for r in csv.DictReader(f):
                st, en = int(r["Start_Timestamp"]) / 1e3, int(r["End_Timestamp"]) / 1e3
                c = int(r.get("Correlation_Id", -1))
                ev.append(
                    {
                        "ph": "X",
                        "cat": "gpu_memcpy",
                        "name": "memcpy:" + r.get("Direction", "?"),
                        "ts": st,
                        "dur": en - st,
                        "tid": "copy",
                        "args": {"correlation": c},
                    }
                )
                keep.add(c)
                n += 1
        print(f"memory copy records: {n} ({mf})")
    if not os.path.exists(af):
        # --hip-runtime-trace is optional (it can wedge native batched copies):
        # the GPU-side gaps stay, host calls are not attributed.
        print(f"no HIP API trace ({af}); host calls are not attributed")
        return ev
    with open(af, newline="") as f:
        for r in csv.DictReader(f):
            fn = r["Function"]
            c = int(r["Correlation_Id"])
            if BLOCKING.search(fn) or c in keep:
                st, en = int(r["Start_Timestamp"]) / 1e3, int(r["End_Timestamp"]) / 1e3
                ev.append(
                    {
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": fn,
                        "ts": st,
                        "dur": en - st,
                        "tid": r["Thread_Id"],
                        "args": {"correlation": c},
                    }
                )
    return ev


_BUSY = {}


def merge_busy(ivs, lo, hi):
    """GPU busy time inside [lo, hi] given sorted (start, end) intervals.

    The union is precomputed once; each query is O(log n).
    """
    key = id(ivs)
    if key not in _BUSY:
        starts, ends, cum = [], [], [0.0]
        cs = ce = None
        for s, e in ivs:
            if ce is None or s > ce:
                if ce is not None:
                    starts.append(cs)
                    ends.append(ce)
                    cum.append(cum[-1] + ce - cs)
                cs, ce = s, e
            elif e > ce:
                ce = e
        if ce is not None:
            starts.append(cs)
            ends.append(ce)
            cum.append(cum[-1] + ce - cs)
        _BUSY[key] = (starts, ends, cum)
    starts, ends, cum = _BUSY[key]
    if hi <= lo or not starts:
        return 0.0
    i = bisect.bisect_right(ends, lo)  # first merged interval ending after lo
    j = bisect.bisect_left(starts, hi)  # merged intervals [i, j) start before hi
    if i >= j:
        return 0.0
    busy = cum[j] - cum[i]
    busy -= max(0.0, lo - starts[i])  # clip the first interval's head
    busy -= max(0.0, ends[j - 1] - hi)  # clip the last interval's tail
    return busy


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def stats_line(xs, unit="us"):
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} mean {statistics.fmean(xs):.0f} p50 {pct(xs, 50):.0f} "
        f"p90 {pct(xs, 90):.0f} max {max(xs):.0f} {unit}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--marker", default="get_num_sampled_and_rejected")
    ap.add_argument("--k2", default=r"mono_ffn_kernel<\s*(\d+),\s*(\d+),\s*([123])\s*>")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument(
        "--proc",
        type=int,
        default=0,
        help=(
            "rocprofv3 directory: analyse the N-th process by K2 kernel count "
            "(0 = most)"
        ),
    )
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    k2_rx = re.compile(a.k2)
    marker_rx = re.compile(a.marker)

    events = (
        events_from_rocprof(a.trace, k2_rx, a.proc)
        if os.path.isdir(a.trace)
        else load(a.trace)
    )
    cats = Counter(e.get("cat") for e in events)
    print("event categories:", dict(cats))
    gpu = sorted(
        (
            (e["ts"], e["ts"] + e.get("dur", 0), e.get("name", ""), e)
            for e in events
            if e.get("cat") in GPU_CATS and e.get("ph") == "X"
        ),
        key=lambda x: x[0],
    )
    rt = [e for e in events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]
    print(f"gpu events {len(gpu)}  runtime events {len(rt)}")
    if not gpu:
        print(
            "no GPU events: boot with ROCPROFILER_REGISTER_ENABLED=1 "
            "(serve_profile.sh torch branch)"
        )
        return 1
    busy_iv = [(s, e) for s, e, _, _ in gpu]
    busy_iv.sort()

    # K2 chain: stage 1 .. stage 2 (n passes) .. stage 3, in launch order
    k2 = []
    for s, e, n, ev in gpu:
        m = k2_rx.search(n)
        if m:
            k2.append((s, e, int(m.group(3)), int(m.group(1)), ev))
    print(
        f"K2 offload stage kernels: {len(k2)} "
        f"(stage counts {dict(Counter(k[2] for k in k2))})"
    )

    layers = []
    cur = None
    for s, e, stage, mt, ev in k2:
        if stage == 1:
            cur = {"s1": (s, e), "s2": [], "ev2": [], "s3": None, "mt": mt}
        elif stage == 2 and cur is not None:
            cur["s2"].append((s, e))
            cur["ev2"].append(ev)
        elif stage == 3 and cur is not None:
            cur["s3"] = (s, e)
            layers.append(cur)
            cur = None
    print(f"complete K2 layers (1..2*..3): {len(layers)}")

    # steps
    marks = [e for s, e, n, _ in gpu if marker_rx.search(n)]
    steps = (
        list(zip(marks, marks[1:]))
        if len(marks) >= 3
        else [(gpu[0][0], max(e for _, e, _, _ in gpu))]
    )
    print(f"steps by marker {a.marker!r}: {len(steps)} ({len(marks)} markers)")
    starts = [s for s, _ in steps]

    def step_of(t):
        i = bisect.bisect_right(starts, t) - 1
        return i if 0 <= i < len(steps) and t <= steps[i][1] else None

    per_step = defaultdict(list)
    for L in layers:
        i = step_of(L["s1"][0])
        if i is not None:
            per_step[i].append(L)

    # per-layer bubbles
    g12, g23, gpass, wall, layer_busy, bubble, ncopy, copy_us, npass = (
        [] for _ in range(9)
    )
    host_q = defaultdict(list)  # tid -> [(ts, key)]
    host_ev = {}
    layer_hosts = []
    copies_all = [
        (s, e, n, ev)
        for s, e, n, ev in gpu
        if ev.get("cat") == "gpu_memcpy" and H2D.search(n)
    ]
    copy_starts = [c[0] for c in copies_all]
    rt_sorted = sorted(rt, key=lambda e: e["ts"])
    rt_ts = [e["ts"] for e in rt_sorted]
    corr_rt = {
        e.get("args", {}).get("correlation"): e
        for e in rt
        if e.get("args", {}).get("correlation") is not None
    }
    first_copy_after_drain = []
    seg = defaultdict(list)
    for L in layers:
        s1e = L["s1"][1]
        s3s = L["s3"][0]
        first2 = L["s2"][0][0] if L["s2"] else s3s
        g12.append(first2 - s1e)
        for (a_s, a_e), (b_s, _b_e) in zip(L["s2"], L["s2"][1:]):
            gpass.append(b_s - a_e)
        if L["s2"]:
            g23.append(s3s - L["s2"][-1][1])
        npass.append(len(L["s2"]))
        w = L["s3"][1] - L["s1"][0]
        b = merge_busy(busy_iv, L["s1"][0], L["s3"][1])
        wall.append(w)
        layer_busy.append(b)
        bubble.append(w - b)
        lo = bisect.bisect_left(copy_starts, s1e)
        hi = bisect.bisect_left(copy_starts, s3s)
        cs = copies_all[lo:hi]
        ncopy.append(len(cs))
        copy_us.append(sum(c[1] - c[0] for c in cs))
        hosts = []
        j = bisect.bisect_left(rt_ts, s1e - 2000)
        while j < len(rt_sorted) and rt_sorted[j]["ts"] < s3s:
            r = rt_sorted[j]
            j += 1
            if BLOCKING.search(r.get("name", "")) and r["ts"] + r.get("dur", 0) > s1e:
                hosts.append(r)
                key = id(r)
                host_q[r["tid"]].append((r["ts"], key))
                host_ev[key] = r
        layer_hosts.append(hosts)
        # g12 decomposition: A1 stage-1 end -> first drain returns, A2 -> last
        # drain returns, B -> stage-2 launch call (host plan + ensure + copy
        # issue), C -> stage-2 kernel start (GPU side: copy/event waits, launch
        # latency)
        launch2 = (
            corr_rt.get(L["ev2"][0].get("args", {}).get("correlation"))
            if L["ev2"]
            else None
        )
        h2d = copies_all[
            bisect.bisect_left(copy_starts, s1e) : bisect.bisect_left(
                copy_starts, first2
            )
        ]
        L["h2d_n"] = len(h2d)
        L["h2d_us"] = sum(c[1] - c[0] for c in h2d)
        if launch2 is not None:
            ends = sorted(
                h["ts"] + h.get("dur", 0)
                for h in hosts
                if DRAIN.search(h["name"]) and h["ts"] < launch2["ts"]
            )
            if ends:
                t1, t2, t3 = ends[0], ends[-1], launch2["ts"]
                last_copy_end = max((c[1] for c in h2d), default=0.0)
                seg["A1"].append(t1 - s1e)
                seg["A2"].append(t2 - t1)
                seg["B"].append(t3 - t2)
                seg["C"].append(first2 - t3)
                seg["Ccopy"].append(max(0.0, min(first2, last_copy_end) - t3))
                L["seg"] = (t1 - s1e, t2 - t1, t3 - t2, first2 - t3)
        if cs:
            c0 = corr_rt.get(cs[0][3].get("args", {}).get("correlation"))
            drains = [h for h in hosts if DRAIN.search(h["name"])]
            if c0 and drains:
                d_end = min(h["ts"] + h.get("dur", 0) for h in drains)
                first_copy_after_drain.append(c0["ts"] - d_end)

    # python frames for the blocking host calls (stack sweep on their threads)
    iv = Intervals()
    tids = set(host_q)
    for e in events:
        if (
            e.get("cat") == "python_function"
            and e.get("ph") == "X"
            and e.get("tid") in tids
        ):
            iv.add(e["tid"], e["ts"], e["ts"] + e.get("dur", 0), e["name"])
    stacks = iv.sweep(host_q)

    def site(r):
        st = [frame_label(f) for f in stacks.get(id(r), [])]
        repo = [lab for cl, lab in st if cl == "repo"]
        return " <- ".join(reversed(repo[-3:])) if repo else "(no repo frame)"

    n_steps = max(len(per_step), 1)
    print()
    print(
        f"layers/step: {len(layers) / n_steps:.1f}  "
        f"(steps with K2 layers: {len(per_step)})"
    )
    print(f"layer wall (stage1 start -> stage3 end): {stats_line(wall)}")
    print(f"layer GPU busy inside it:                {stats_line(layer_busy)}")
    print(f"layer bubble (wall - busy):              {stats_line(bubble)}")
    print(
        f"gap stage1 end -> first stage2 start:    {stats_line(g12)}   "
        "<- drain + plan + miss copies (H2/H3/H4)"
    )
    print(f"gap between stage-2 passes:              {stats_line(gpass)}")
    print(f"gap last stage2 end -> stage3 start:     {stats_line(g23)}")
    print(f"expert passes per layer: {dict(Counter(npass))}")
    print(
        f"layers with H2D copies between the stages: "
        f"{sum(1 for c in ncopy if c)}/{len(ncopy)}; "
        f"H2D records/layer {stats_line([float(c) for c in ncopy], 'copies')}"
    )
    print(f"H2D copy GPU time between the stages (us): {stats_line(copy_us)}")
    if first_copy_after_drain:
        print(
            f"first H2D copy launch - end of first drain (us): "
            f"{stats_line(first_copy_after_drain)}"
        )
    if seg["B"]:
        print()
        print("stage1-end -> stage2-start decomposition (us):")
        print(
            f"  A1 stage-1 end -> first drain call returns (D2H + wake-up): "
            f"{stats_line(seg['A1'])}"
        )
        print(
            f"  A2 first -> last drain call returns (padding .cpu()):        "
            f"{stats_line(seg['A2'])}"
        )
        print(
            f"  B  last drain return -> stage-2 launch call "
            f"(host plan, ensure, copy issue): {stats_line(seg['B'])}"
        )
        print(
            f"  C  stage-2 launch call -> stage-2 kernel start (GPU side):   "
            f"{stats_line(seg['C'])}"
        )
        print(
            f"     of which H2D copies still running after the launch call:  "
            f"{stats_line(seg['Ccopy'])}"
        )
        tot = [sum(L["seg"]) for L in layers if "seg" in L]
        print(f"  sum A1+A2+B+C: {stats_line(tot)}")
        buckets = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 10**9)]
        print(
            "by H2D copy records between the stages "
            "(count of memcpy H2D records inside the gap):"
        )
        print(
            f"  {'H2D n':>9} {'layers':>7} {'share%':>7} {'g12 mean':>9} "
            f"{'A1':>6} {'B':>6} {'C':>6} {'copy GPU us':>11}"
        )
        for lo_n, hi_n in [(0, 0)] + buckets:
            sel = [L for L in layers if "seg" in L and lo_n <= L["h2d_n"] <= hi_n]
            if not sel:
                continue
            m = lambda i, sel=sel: statistics.fmean(L["seg"][i] for L in sel)  # noqa: E731
            lab = (
                f"{lo_n}"
                if lo_n == hi_n
                else (f"{lo_n}-{hi_n}" if hi_n < 10**9 else f">={lo_n}")
            )
            print(
                f"  {lab:>9} {len(sel):7d} "
                f"{100 * len(sel) / len(layers):7.1f} "
                f"{sum(m(i) for i in range(4)):9.0f} "
                f"{m(0):6.0f} {m(2):6.0f} {m(3):6.0f} "
                f"{statistics.fmean(L['h2d_us'] for L in sel):11.0f}"
            )

    # per-step table
    print()
    print(
        "per step: period ms, GPU busy %, K2 layers, "
        "sum(stage1->2 gap) ms, sum(bubble) ms, bubble % of period"
    )
    rows = []
    for i, (s, e) in enumerate(steps):
        period = e - s
        busy = merge_busy(busy_iv, s, e)
        ls = per_step.get(i, [])
        sg12 = sum((L["s2"][0][0] if L["s2"] else L["s3"][0]) - L["s1"][1] for L in ls)
        sb = sum(
            (L["s3"][1] - L["s1"][0]) - merge_busy(busy_iv, L["s1"][0], L["s3"][1])
            for L in ls
        )
        rows.append(
            {
                "step": i,
                "period_ms": period / 1e3,
                "busy_pct": 100 * busy / period if period else 0,
                "layers": len(ls),
                "g12_ms": sg12 / 1e3,
                "bubble_ms": sb / 1e3,
            }
        )
    for r in rows[: a.top]:
        print(
            f"  step {r['step']:3d}  {r['period_ms']:8.1f} ms  "
            f"busy {r['busy_pct']:5.1f}%  layers {r['layers']:3d}  "
            f"g12 {r['g12_ms']:7.1f} ms  bubble {r['bubble_ms']:7.1f} ms "
            f"({100 * r['bubble_ms'] / r['period_ms'] if r['period_ms'] else 0:.0f}%)"
        )
    if rows:
        print(
            f"  median step {statistics.median(r['period_ms'] for r in rows):.1f} ms, "
            f"busy {statistics.median(r['busy_pct'] for r in rows):.1f}%, "
            f"bubble {statistics.median(r['bubble_ms'] for r in rows):.1f} ms"
        )

    # blocking host calls inside the bubbles, by site
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for hosts in layer_hosts:
        for r in hosts:
            k = (r["name"], site(r))
            d = r.get("dur", 0)
            agg[k][0] += 1
            agg[k][1] += d
            agg[k][2] = max(agg[k][2], d)
    print()
    print(
        f"blocking host calls between stage 1 end and stage 3 start "
        f"(per step: count, ms; top {a.top}):"
    )
    for (n, st), (c, tot, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(
            f"  {c / n_steps:7.1f}/step {tot / n_steps / 1e3:8.2f} ms/step  "
            f"mean {tot / c:8.0f} us  max {mx:8.0f} us  {n}  {st}"
        )

    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "steps": rows,
                    "layers": len(layers),
                    "g12_us": g12,
                    "g23_us": g23,
                    "bubble_us": bubble,
                    "copies": ncopy,
                },
                f,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
