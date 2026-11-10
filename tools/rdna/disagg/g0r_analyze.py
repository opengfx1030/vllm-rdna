#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline comparison of the ``g0r_trace.py`` traces of the DisaggRework arms.

  g0r_analyze.py --root ~/w4a8_runs/disagg/g0a --prod producer_mono \
                 --cons consumer_mono [--ids L2049_0,L3073_0] [--ranks 0,1,2,3]

``<root>/arms/<arm>/trace/<tag>/rank<r>/steps.json`` holds, per prefill step of one
request, the chunk (``computed``, ``sched``) and fingerprints of its state pages before
the request's first step (``pre``) and after every step (``post``). Tags: ``<id>_cold0``
and ``<id>_hit`` (``base_ds``), ``<id>_P`` (producer), ``<id>_D`` (consumer). Per prompt
of length N and rank this prints:

* the chunk boundaries of the baseline's cold prefill and of the producer's prefill, up
  to position N-1;
* ``P@N-1 vs B@N-1``: the producer's state after its last chunk against the baseline's
  in-flight state at the same position (the claim of the monolithic split);
* ``D pre vs P@N-1``: the state the decode instance starts from against what the
  producer left (what the transfer must preserve);
* ``Bhit pre vs B@pos``: the baseline's restored state after a prefix hit against its
  own in-flight state at that position;
* for the first mismatch of each comparison, the byte-level difference when both sides
  saved raw pages (state groups only).

Fingerprints are position-weighted wrapping sums of the page's int32 words; equal
fingerprints of every (group, column, layer) mean equal state up to a hash collision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


def load(root: str, arm: str, tag: str, rank: int) -> dict[str, Any] | None:
    path = os.path.join(root, "arms", arm, "trace", tag, f"rank{rank}", "steps.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    doc["dir"] = os.path.dirname(path)
    return doc


def chunks(doc: dict[str, Any], upto: int) -> list[tuple[int, int]]:
    """Prefill chunks [computed, computed+sched) that end at or before ``upto``."""
    return [
        (s["computed"], s["computed"] + s["sched"])
        for s in doc["steps"]
        if s["computed"] + s["sched"] <= upto
    ]


def fmt_chunks(cs: list[tuple[int, int]]) -> str:
    return " ".join(f"[{a},{b})" for a, b in cs) or "-"


def post_at(doc: dict[str, Any], pos: int) -> dict[str, Any] | None:
    for s in doc["steps"]:
        if s["computed"] + s["sched"] == pos and "post" in s:
            return s["post"]
    return None


def pre_of(doc: dict[str, Any]) -> dict[str, Any] | None:
    for s in doc["steps"]:
        if "pre" in s:
            return s["pre"]
    return None


def compare_snaps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per KV group: fingerprint entries compared / equal, and the first differences
    as (column, layer index)."""
    out: dict[str, Any] = {}
    for g in sorted(set(a["fp"]) | set(b["fp"]), key=int):
        ga, gb = a["fp"].get(g), b["fp"].get(g)
        if ga is None or gb is None:
            out[g] = {"missing": "A" if ga is None else "B"}
            continue
        total = equal = 0
        diffs: list[tuple[int, int]] = []
        for col in sorted(set(ga) & set(gb), key=int):
            la, lb = ga[col]["layers"], gb[col]["layers"]
            for li, (x, y) in enumerate(zip(la, lb)):
                total += 1
                if x == y:
                    equal += 1
                elif len(diffs) < 6:
                    diffs.append((int(col), li))
        out[g] = {
            "total": total,
            "equal": equal,
            "diffs": diffs,
            "cols": [sorted(ga, key=int), sorted(gb, key=int)],
        }
    return out


def fmt_cmp(cmp: dict[str, Any]) -> str:
    parts = []
    for g, r in cmp.items():
        if "missing" in r:
            parts.append(f"g{g}:missing-{r['missing']}")
        elif r["total"] == r["equal"]:
            parts.append(f"g{g}:{r['equal']}/{r['total']}")
        else:
            parts.append(f"g{g}:{r['equal']}/{r['total']} DIFF{r['diffs'][:3]}")
    return " ".join(parts)


RawKey = tuple[int, int, int]


def raw_items(doc_dir: str, name: str) -> dict[RawKey, tuple[str, int, int]]:
    base = os.path.join(doc_dir, name)
    if not os.path.exists(base + ".json"):
        return {}
    with open(base + ".json") as f:
        meta = json.load(f)
    return {
        (it["g"], it["col"], it["layer"]): (it["h"], it["off"], it["n"])
        for it in meta["items"]
    }


def raw_diff(dir_a: str, name_a: str, dir_b: str, name_b: str, limit: int = 4) -> str:
    """Byte-level difference of the state pages two traces saved (first ``limit``)."""
    ia, ib = raw_items(dir_a, name_a), raw_items(dir_b, name_b)
    if not ia or not ib:
        return "raw: not saved on both sides"
    da = np.memmap(os.path.join(dir_a, name_a + ".bin"), dtype=np.uint8, mode="r")
    db = np.memmap(os.path.join(dir_b, name_b + ".bin"), dtype=np.uint8, mode="r")
    lines = []
    equal = 0
    for key in sorted(set(ia) & set(ib)):
        (ha, oa, na), (hb, ob, nb) = ia[key], ib[key]
        if ha == hb:
            equal += 1
            continue
        if len(lines) >= limit:
            continue
        xa, xb = da[oa : oa + na], db[ob : ob + nb]
        n = min(na, nb)
        bad = np.flatnonzero(xa[:n] != xb[:n])
        lines.append(
            f"g{key[0]} col{key[1]} layer{key[2]}: {bad.size}/{n} bytes differ"
            + (f", first at {int(bad[0])}" if bad.size else "")
        )
    total = len(set(ia) & set(ib))
    return f"raw pages equal {equal}/{total}" + ("".join(f"\n      {x}" for x in lines))


def analyze(args: argparse.Namespace) -> int:
    ids = args.ids.split(",") if args.ids else None
    if ids is None:
        tdir = os.path.join(args.root, "arms", args.prod, "trace")
        ids = sorted(
            (d[:-2] for d in os.listdir(tdir) if d.endswith("_P")),
            key=lambda x: (int(x[1:].split("_")[0]), x),
        )
    report: dict[str, Any] = {}
    for pid in ids:
        n = int(pid[1:].split("_")[0])
        for rank in (int(r) for r in args.ranks.split(",")):
            cold = load(args.root, args.base, f"{pid}_cold0", rank)
            hit = load(args.root, args.base, f"{pid}_hit", rank)
            prod = load(args.root, args.prod, f"{pid}_P", rank)
            cons = load(args.root, args.cons, f"{pid}_D", rank) if args.cons else None
            row: dict[str, Any] = {}
            print(f"== {pid} (N={n}) rank {rank}")
            if cold:
                print(f"  B cold chunks to N-1: {fmt_chunks(chunks(cold, n - 1))}")
            if prod:
                print(f"  P      chunks:        {fmt_chunks(chunks(prod, n))}")
            if cold and prod:
                same = chunks(cold, n - 1) == chunks(prod, n)
                row["chunks_equal"] = same
                print(f"  chunk boundaries equal: {same}")
            if cold and prod:
                a, b = post_at(prod, n - 1), post_at(cold, n - 1)
                if a and b:
                    cmp = compare_snaps(a, b)
                    row["P_vs_B_state"] = cmp
                    print(f"  P@N-1 vs B@N-1:  {fmt_cmp(cmp)}")
                    if any(r.get("total") != r.get("equal") for r in cmp.values()):
                        print(
                            "    "
                            + raw_diff(
                                prod["dir"],
                                f"raw_post_{n - 1}",
                                cold["dir"],
                                f"raw_post_{n - 1}",
                            )
                        )
            if prod and cons:
                a, b = post_at(prod, n - 1), pre_of(cons)
                if a and b:
                    cmp = compare_snaps(a, b)
                    row["D_pre_vs_P_state"] = cmp
                    print(f"  D pre vs P@N-1:  {fmt_cmp(cmp)}")
                    if any(r.get("total") != r.get("equal") for r in cmp.values()):
                        print(
                            "    "
                            + raw_diff(
                                prod["dir"],
                                f"raw_post_{n - 1}",
                                cons["dir"],
                                f"raw_pre_{n - 1}",
                            )
                        )
            if hit and cold:
                pre = pre_of(hit)
                if pre:
                    pos = pre["pos"]
                    ref = post_at(cold, pos)
                    if ref:
                        cmp = compare_snaps(pre, ref)
                        row["Bhit_pre_vs_Bcold_state"] = cmp
                        print(f"  Bhit pre@{pos} vs Bcold@{pos}: {fmt_cmp(cmp)}")
            report[f"{pid}/rank{rank}"] = row
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="driver result dir (R)")
    ap.add_argument("--base", default="base_ds")
    ap.add_argument("--prod", default="producer_mono")
    ap.add_argument("--cons", default="consumer_mono")
    ap.add_argument("--ids", help="comma list of prompt ids (default: all in --prod)")
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--out")
    return analyze(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
