# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host time per Python frame in a torch profiler trace (no GPU events needed).

usage: frame_time.py TRACE.json[.gz] [--step REGEX] [--match REGEX] [--top N]
                     [--layer REGEX ...]

The trace is a ``serve_profile.sh PROFILER=torch`` window. Python frames of the
busiest thread are aggregated by label (``file:line function``; builtin-method
frames such as ``<built-in method cpu of Tensor object>`` keep their name): count
per step, inclusive ms per step, mean and max. Steps are the ``--step`` frames
(default ``execute_model``). Blocking host calls show up directly as long
``cpu`` / ``tolist`` / ``item`` / ``synchronize`` frames, so the host drain of the
expert-offload path (k2.py ``route_ids.cpu()``, rdna_expert_cache.py
``_map_upload_event.synchronize()``) is measurable even when the trace has no
GPU activity events.

``--layer REGEX`` (repeatable) prints, for every frame whose label matches, the
timeline of its direct children: mean start offset, mean duration and count per
(child label, occurrence index), in start order. ``--layer 'k2.py:\\d+ run_offloaded'``
gives the host sequence of one offloaded MoE layer (launch stage 1, drain, plan,
ensure, passes, finish).
"""

import argparse
import statistics
import sys
from collections import Counter, defaultdict

import regex as re

sys.path.insert(
    0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))
)
from kernel_sites import load  # noqa: E402

ADDR = re.compile(r" at 0x[0-9a-f]+")
FRAME = re.compile(r"^(.*)\((\d+)\): (.*)$")


def label(raw):
    raw = ADDR.sub("", raw)
    m = FRAME.match(raw)
    if not m:
        return raw[:110]
    path, line, fn = m.groups()
    if "site-packages/" in path:
        path = path.split("site-packages/")[-1]
    elif "/vllm/" in path:
        path = "vllm/" + path.split("/vllm/", 1)[1]
    return f"{path}:{line} {fn}"


def child_timelines(ev, rx):
    """For frames matching rx: mean (offset, dur, count) of each direct child.

    Children are grouped by (label, occurrence index).
    """
    ev = sorted(ev, key=lambda e: (e["ts"], -e["dur"]))
    stack = []  # (end, index)
    parent = [-1] * len(ev)
    for i, e in enumerate(ev):
        while stack and stack[-1][0] <= e["ts"]:
            stack.pop()
        parent[i] = stack[-1][1] if stack else -1
        stack.append((e["ts"] + e["dur"], i))
    kids = defaultdict(list)
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    own = []
    for i, e in enumerate(ev):
        if not rx.search(label(e["name"])):
            continue
        own.append(e["dur"])
        seen = Counter()
        for k in kids.get(i, ()):
            lab = label(ev[k]["name"])
            g = agg[(lab, seen[lab])]
            seen[lab] += 1
            g[0] += 1
            g[1] += ev[k]["ts"] - e["ts"]
            g[2] += ev[k]["dur"]
    return own, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step", default="execute_model")
    ap.add_argument("--match", default=None, help="only labels matching this regex")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--layer",
        action="append",
        default=[],
        help="child timeline of frames matching this label regex",
    )
    a = ap.parse_args()
    ev = [
        e
        for e in load(a.trace)
        if e.get("cat") == "python_function" and e.get("ph") == "X"
    ]
    by_tid = Counter(e["tid"] for e in ev)
    tid = by_tid.most_common(1)[0][0]
    ev = [e for e in ev if e["tid"] == tid]
    step_rx = re.compile(a.step)
    steps = sorted(
        (e for e in ev if step_rx.search(label(e["name"]))), key=lambda e: e["ts"]
    )
    n_steps = max(len(steps), 1)
    print(
        f"python frames on busiest thread {tid}: {len(ev)}; "
        f"steps ({a.step!r}): {len(steps)}"
    )
    if steps:
        d = [s["dur"] / 1e3 for s in steps]
        print(
            f"step frame ms: mean {statistics.fmean(d):.1f} "
            f"p50 {statistics.median(d):.1f} max {max(d):.1f}"
        )
    rx = re.compile(a.match) if a.match else None
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for e in ev:
        lab = label(e["name"])
        if rx and not rx.search(lab):
            continue
        g = agg[lab]
        g[0] += 1
        g[1] += e["dur"]
        g[2] = max(g[2], e["dur"])
    for lr in a.layer:
        own, agg_l = child_timelines(ev, re.compile(lr))
        if not own:
            print(f"\n--layer {lr!r}: no matching frames")
            continue
        own_s = sorted(own)
        print(
            f"\n--layer {lr!r}: {len(own)} frames, "
            f"dur us mean {statistics.fmean(own):.0f} "
            f"p50 {own_s[len(own_s) // 2]:.0f} max {own_s[-1]:.0f}"
        )
        print(f"{'start us':>9} {'dur us':>8} {'n':>5}  child (occurrence index)")
        for (lab, idx), (c, off, dur) in sorted(
            agg_l.items(), key=lambda kv: kv[1][1] / kv[1][0]
        ):
            if c < 0.2 * len(own):
                continue
            print(f"{off / c:9.0f} {dur / c:8.0f} {c:5d}  {lab} #{idx}")
        print()
    print(f"{'count/step':>10} {'ms/step':>9} {'mean us':>9} {'max us':>9}  frame")
    for lab, (c, tot, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(
            f"{c / n_steps:10.1f} {tot / n_steps / 1e3:9.2f} "
            f"{tot / c:9.0f} {mx:9.0f}  {lab}"
        )


if __name__ == "__main__":
    main()
