# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verdict of one replay boot of ``e_run.sh``: do the ranks disagree, on which shapes,
which rank is odd, and is it the same layers every time?

    python -m tools.rdna.eplb.replay_check <stage>/replay [--json out.json]

``<stage>/replay`` holds what ``e_replay.py`` wrote: ``replay.json`` (the prompt groups
in the order they ran), ``dump/`` (the four counter files). A dump directory without a
``replay.json`` gets the whole-window verdict only.

The ring is cut into forward passes (a pass writes one record per counted layer, in
layer order) and the passes with ``phase`` prefill into *runs* of consecutive passes: at
c=1 one run is one request, so the last ``sum(count)`` runs are the replayed requests
in order (the boot probes and the warm-up come before them). Per group it reports the
runs with a disagreeing record, the odd rank (the one off the majority hash), whether
loads differ too, and how often the same (pass, layer) cell disagrees across the
replays: a cell that disagrees in every replay is a deterministic function of the
inputs, a cell that disagrees now and then is data dependent noise.
"""

import argparse
import collections
import itertools
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.rdna.eplb import summary


@dataclass
class Pass:
    """One forward pass: ``stop - start`` consecutive records of the window."""

    start: int
    stop: int
    rows: int
    phase: int
    complete: bool


def _window(stats: list[summary.RankStats]) -> dict[str, np.ndarray] | None:
    """Per record of the window all rings still hold: layer, rows, phase, verdicts."""
    lo = max(s.first_record for s in stats)
    hi = min(s.ring_total for s in stats)
    if hi <= lo:
        return None
    tabs = [summary.ring_table(s) for s in stats]
    sl = [slice(lo - s.first_record, hi - s.first_record) for s in stats]
    h = np.stack([t["hash"][x] for t, x in zip(tabs, sl)])  # [ranks, n]
    agree = (h[:, None, :] == h[None, :, :]).sum(axis=1)
    major = h[np.argmax(agree, axis=0), np.arange(h.shape[1])]
    off = h != major
    n_off = off.sum(axis=0)
    odd = np.where(n_off == 0, -1, np.where(n_off == 1, np.argmax(off, axis=0), -2))
    t_all = np.stack([t["T"][x] for t, x in zip(tabs, sl)])
    p_all = np.stack([t["P"][x] for t, x in zip(tabs, sl)])
    loads = (t_all != t_all[0]).any(axis=(0, 2)) | (p_all != p_all[0]).any(axis=(0, 2))
    return {
        "first": np.array(lo),
        "layer": tabs[0]["layer"][sl[0]],
        "rows": tabs[0]["rows"][sl[0]],
        "phase": tabs[0]["phase"][sl[0]],
        "bad": n_off > 0,
        "odd": odd,
        "loads": loads & (n_off > 0),
    }


def split_passes(layer: np.ndarray, rows: np.ndarray, phase: np.ndarray, n_layers: int):
    """Cut the records into forward passes (the layer position restarts every pass)."""
    if layer.size == 0:
        return []
    new = np.r_[True, layer[1:] <= layer[:-1]]
    starts = np.flatnonzero(new)
    stops = np.r_[starts[1:], layer.size]
    passes = []
    for a, b in zip(starts, stops):
        full = b - a == n_layers and layer[a] == 0
        passes.append(Pass(int(a), int(b), int(rows[a]), int(phase[a]), bool(full)))
    return passes


def prefill_runs(passes: list[Pass]) -> list[list[int]]:
    """Maximal runs of consecutive complete prefill passes (indices into ``passes``)."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, p in enumerate(passes):
        if p.phase == 1 and p.complete:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def group_report(
    group: dict, runs: list[list[int]], passes: list[Pass], w: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Statistics of the runs of one prompt group."""
    sigs = [tuple(passes[i].rows for i in run) for run in runs]
    common = collections.Counter(sigs).most_common(1)[0][0] if sigs else ()
    cells: list[frozenset] = []
    per_run = []
    odd: collections.Counter = collections.Counter()
    loads_differ = 0
    for run in runs:
        found = set()
        n_rec = 0
        for j, i in enumerate(run):
            p = passes[i]
            sel = np.arange(p.start, p.stop)
            bad = sel[w["bad"][sel]]
            n_rec += int(bad.size)
            loads_differ += int(w["loads"][bad].sum())
            for k in bad:
                found.add((j, int(w["layer"][k])))
                odd[int(w["odd"][k])] += 1
        cells.append(frozenset(found))
        per_run.append(n_rec)
    hit = [c for c in cells if c]
    freq = collections.Counter(itertools.chain.from_iterable(cells))
    pairs = list(itertools.combinations(hit, 2))
    n = len(runs)
    return {
        "name": group["name"],
        "length": group["length"],
        "kind": group["kind"],
        "expected_runs": group["count"],
        "runs": n,
        "signature": list(common),
        "signature_consistent": all(s == common for s in sigs),
        "runs_with_mismatch": len(hit),
        "records_mismatch": int(sum(per_run)),
        "per_run_records": per_run,
        "odd_rank": {str(k): v for k, v in sorted(odd.items())},
        "loads_differ": loads_differ,
        "hash_only": int(sum(per_run)) - loads_differ,
        "distinct_cell_sets": len({c for c in cells if c}),
        "jaccard_mean": float(np.mean([_jaccard(a, b) for a, b in pairs]))
        if pairs
        else None,
        "cells_every_run": sum(1 for v in freq.values() if v == n) if n > 1 else None,
        "cells_some_runs": sum(1 for v in freq.values() if v < n) if n > 1 else None,
        "cell_max_freq": max(freq.values()) if freq else 0,
        "distinct_outputs": group.get("distinct_outputs"),
        "cached_tokens": group.get("cached_tokens"),
    }


def analyze(replay_dir: str) -> dict[str, Any]:
    dump = os.path.join(replay_dir, "dump")
    if not os.path.isdir(dump):
        dump = replay_dir
    stats = summary.load_dir(dump)
    rep: dict[str, Any] = {"consensus": summary.consensus(stats)}
    w = _window(stats)
    if w is None:
        return rep
    n_layers = len(stats[0].layers)
    passes = split_passes(w["layer"], w["rows"], w["phase"], n_layers)
    by_rows: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for p in passes:
        e = by_rows[p.rows]
        e[0] += 1
        e[1] += int(w["bad"][p.start : p.stop].sum())
        e[2] += int(p.phase == 1)
    rep["by_rows"] = {
        str(r): {"passes": e[0], "mismatching_records": e[1], "prefill": bool(e[2])}
        for r, e in sorted(by_rows.items())
    }
    rep["passes"] = len(passes)
    meta_path = os.path.join(replay_dir, "replay.json")
    if not os.path.isfile(meta_path):
        return rep
    with open(meta_path) as f:
        meta = json.load(f)
    runs = prefill_runs(passes)
    total = sum(g["count"] for g in meta["groups"])
    rep["prefill_runs"] = len(runs)
    rep["replay_requests"] = total
    used: set[int] = set()
    groups = []
    end = len(runs)
    for g in reversed(meta["groups"]):  # the newest runs are the last group's
        take = runs[max(0, end - g["count"]) : end]
        end -= g["count"]
        used.update(i for run in take for i in run)
        groups.append(group_report(g, take, passes, w))
    rep["groups"] = list(reversed(groups))
    outside = [
        p
        for i, p in enumerate(passes)
        if i not in used and int(w["bad"][p.start : p.stop].sum())
    ]
    rep["mismatching_records_outside_replay_runs"] = int(
        sum(w["bad"][p.start : p.stop].sum() for p in outside)
    )
    return rep


def format_report(rep: dict[str, Any]) -> str:
    c = rep["consensus"]
    out = [
        f"consensus: ok={c.get('ok')} compared={c.get('compared')} "
        f"mismatches={c.get('mismatches')} first={c.get('first_mismatch_record')}"
    ]
    if "detail" in c:
        out.append(f"  detail: {json.dumps(c['detail'])}")
    if "by_rows" in rep:
        out.append(
            f"passes in the window: {rep['passes']}; by valid rows (mismatching only):"
        )
        bad = {r: v for r, v in rep["by_rows"].items() if v["mismatching_records"]}
        for r, v in bad.items():
            out.append(
                f"  rows={r}: {v['passes']} passes, {v['mismatching_records']} records"
            )
        if not bad:
            out.append("  none")
    for g in rep.get("groups", []):
        n = g["runs"]
        out.append(
            f"[{g['name']}] {g['length']} tokens ({g['kind']}): {n} of "
            f"{g['expected_runs']} runs, prefill passes per request {g['signature']}"
            f"{'' if g['signature_consistent'] else ' (NOT the same in every run)'}"
        )
        out.append(
            f"    runs with a disagreeing record: {g['runs_with_mismatch']} of {n}; "
            f"records {g['records_mismatch']} (per run {g['per_run_records']})"
        )
        if g["records_mismatch"]:
            out.append(
                f"    odd rank {g['odd_rank']} (-2: no single odd rank); loads differ "
                f"in {g['loads_differ']}, hash only {g['hash_only']}"
            )
            out.append(
                f"    distinct (pass, layer) sets {g['distinct_cell_sets']}; mean "
                f"Jaccard {g['jaccard_mean']}; cells that recur in every run "
                f"{g['cells_every_run']}, in some runs {g['cells_some_runs']}, max "
                f"recurrence {g['cell_max_freq']}"
            )
        out.append(
            f"    distinct greedy outputs {g['distinct_outputs']}, "
            f"prefix-cache tokens {g['cached_tokens']}"
        )
    if "mismatching_records_outside_replay_runs" in rep:
        out.append(
            "mismatching records outside the replayed requests (probes, warm-up): "
            f"{rep['mismatching_records_outside_replay_runs']}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rep = analyze(args.dir)
    print(format_report(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
