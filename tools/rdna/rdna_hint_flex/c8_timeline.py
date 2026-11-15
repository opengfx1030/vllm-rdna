#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler timeline of a ``vllm bench serve`` cell, from its JSON alone.

    python3 tools/rdna/rdna_hint_flex/c8_timeline.py bench-16384x1024-c8.json \
        [--c1-prefill-tps 720] [--prefix OUT/c8_timeline]

Per request the JSON has the send time (``start_times``), the time to first token
(``ttfts``) and every inter-token latency (``itls``). From them:

* phases: how long the engine had a prefill pending and no decoder (``prefill``),
  a prefill pending and at least one decoder (``mixed``), and only decoders
  (``decode``, by decoder count);
* steps: the token times of all requests, clustered within ``--cluster-ms``, are
  the end times of engine steps; a step longer than ``--mixed-s`` carried a
  prefill chunk. Count, total and spread of both kinds, and the decoders in the
  decode-only steps;
* the gap to the idle-engine prefill rate (``--c1-prefill-tps``): the time the
  prompt tokens would take alone, the prefill-bearing phases above that, and the
  decode-only phases.

Writes ``PREFIX.md`` and ``PREFIX_steps.csv`` (one row per step). Standard
library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(p / 100 * len(s)))] if s else float("nan")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    t0 = min(d["start_times"])
    reqs = []
    for i, (st, ttft, itls, lat) in enumerate(
        zip(d["start_times"], d["ttfts"], d["itls"], d["latencies"])
    ):
        start = st - t0
        times = [start + ttft]
        for x in itls:
            times.append(times[-1] + x)
        reqs.append(
            dict(
                i=i,
                start=start,
                first=times[0],
                end=max(times[-1], start + lat),
                times=times,
                n_in=d["input_lens"][i],
                n_out=d["output_lens"][i],
            )
        )
    return reqs


def phases(reqs: list[dict], horizon: float) -> dict[str, float]:
    """Seconds per engine state: prefill / mixed / decode_<n> / idle."""
    ev: list[tuple[float, int, int]] = []
    for r in reqs:
        ev += [(r["start"], 1, 0), (r["first"], -1, 1), (r["end"], 0, -1)]
    ev.sort()
    out: dict[str, float] = defaultdict(float)
    pre = dec = 0
    prev = 0.0
    for t, dp, dd in ev:
        dt = t - prev
        if dt > 0:
            if pre and dec:
                out["mixed"] += dt
            elif pre:
                out["prefill"] += dt
            elif dec:
                out[f"decode_{dec}"] += dt
            else:
                out["idle"] += dt
        pre += dp
        dec += dd
        prev = t
    out["total"] = horizon
    return dict(out)


def steps(reqs: list[dict], cluster_s: float) -> list[tuple[float, int]]:
    """(end time, tokens emitted) of every engine step, from clustered token times."""
    out: list[tuple[float, int]] = []
    for t in sorted(t for r in reqs for t in r["times"]):
        if out and t - out[-1][0] <= cluster_s:
            end, n = out[-1]
            out[-1] = (max(end, t), n + 1)
        else:
            out.append((t, 1))
    return out


def step_row(name: str, v: list[float]) -> str:
    return (
        f"| {name} | {len(v)} | {sum(v):.1f} | {1e3 * statistics.mean(v):.1f} | "
        f"{1e3 * pct(v, 50):.1f} | {1e3 * pct(v, 99):.1f} |"
    )


def report(a: argparse.Namespace, reqs: list[dict]) -> tuple[str, list[tuple]]:
    horizon = max(r["end"] for r in reqs)
    ph = phases(reqs, horizon)
    st = steps(reqs, a.cluster_ms / 1e3)
    durs = [(st[k][0] - st[k - 1][0], st[k][1]) for k in range(1, len(st))]
    mixed = [d for d, _ in durs if d >= a.mixed_s]
    dec = [(d, n) for d, n in durs if d < a.mixed_s]
    n_in = sum(r["n_in"] for r in reqs)
    ideal = n_in / a.c1_prefill_tps
    pre_bearing = ph.get("prefill", 0.0) + ph.get("mixed", 0.0)
    dec_only = sum(v for k, v in ph.items() if k.startswith("decode_"))

    out = [f"# Timeline of {os.path.basename(a.bench_json)}", ""]
    out += [f"{len(reqs)} requests, {n_in} prompt tokens, duration {horizon:.1f} s"]
    out += ["", "| phase | seconds | share |", "| --- | ---: | ---: |"]
    for k in ("prefill", "mixed", "idle"):
        if k in ph:
            out.append(f"| {k} | {ph[k]:.2f} | {100 * ph[k] / horizon:.1f}% |")
    for k in sorted((k for k in ph if k.startswith("decode_")), key=lambda x: x[7:]):
        label = f"decode, {k[7:]} decoders"
        out.append(f"| {label} | {ph[k]:.2f} | {100 * ph[k] / horizon:.1f}% |")

    out += ["", "| steps | count | total s | mean ms | p50 ms | p99 ms |"]
    out += ["| --- | ---: | ---: | ---: | ---: | ---: |"]
    if mixed:
        out.append(step_row(f"prefill-carrying (>= {a.mixed_s} s)", mixed))
    if dec:
        out.append(step_row("decode-only", [d for d, _ in dec]))
        by_n: dict[int, list[float]] = defaultdict(list)
        for d, n in dec:
            by_n[n].append(d)
        out += ["", "| decode-only step: tokens | steps | mean ms |"]
        out += ["| ---: | ---: | ---: |"]
        for n in sorted(by_n):
            if len(by_n[n]) >= 5:
                mean_ms = 1e3 * statistics.mean(by_n[n])
                out.append(f"| {n} | {len(by_n[n])} | {mean_ms:.1f} |")

    out += ["", f"Idle-engine prefill at {a.c1_prefill_tps:.0f} tok/s: {ideal:.1f} s."]
    out += [
        f"Prefill-bearing phases: {pre_bearing:.1f} s = {n_in / pre_bearing:.0f} tok/s,"
        f" {pre_bearing - ideal:+.1f} s against the idle-engine time.",
        f"Decode-only phases: {dec_only:.1f} s. Cell duration above the idle-engine "
        f"prefill time: {horizon - ideal:.1f} s "
        f"({100 * (horizon - ideal) / horizon:.0f}%).",
    ]
    if mixed:
        mean = statistics.mean(mixed)
        out.append(
            f"A prefill-carrying step takes {1e3 * mean:.0f} ms; at {a.chunk} tokens "
            f"per step that is {a.chunk / mean:.0f} tok/s."
        )
    rows = [(st[k][0], st[k][0] - st[k - 1][0], st[k][1]) for k in range(1, len(st))]
    return "\n".join(out) + "\n", rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench_json")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--c1-prefill-tps", type=float, default=720.0)
    ap.add_argument("--cluster-ms", type=float, default=8.0)
    ap.add_argument("--mixed-s", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=1024, help="tokens per mixed step")
    a = ap.parse_args()
    text, rows = report(a, load(a.bench_json))
    print(text)
    if a.prefix:
        os.makedirs(os.path.dirname(os.path.abspath(a.prefix)), exist_ok=True)
        with open(a.prefix + ".md", "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(a.prefix + "_steps.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["end_s", "duration_s", "tokens", "kind"])
            for end, d, n in rows:
                kind = "mixed" if d >= a.mixed_s else "decode"
                w.writerow([f"{end:.4f}", f"{d:.4f}", n, kind])
    return 0


if __name__ == "__main__":
    sys.exit(main())
