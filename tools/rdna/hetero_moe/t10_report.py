#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks and tables of the T10 run (``t10_run.sh``) of the expert-stats collector.

    python -m tools.rdna.hetero_moe.t10_report itl RUN_DIR [--json out.json]
    python -m tools.rdna.hetero_moe.t10_report check ARM_DIR [--json out.json]

``itl``: per bench cell, the median ITL of every arm (``arms/A1 B1 B2 A2``), the
mean of the stats-off arms (A) against the mean of the stats-on arms (B), and the
within-condition spread (|A1 - A2|, |B1 - B2|). Decision rule, fixed beforehand:
the collector costs something only if ``mean(B) - mean(A)`` exceeds both spreads;
otherwise the overhead is reported as unresolved with ``delta + spread`` as its
upper bound (the convention of docs/rdna2/hwq-abba-2026-11-06.md).

``check``: GPU validation items of docs/rdna2/expert-stats-2026-11-07.md section 8
on one arm's ``stats/`` directory:

* snapshots: ``seq`` strictly increasing, ``elapsed_s`` increasing, every
  cumulative counter non-decreasing (snapshot ordering);
* ``sum(counts) == top_k * tokens`` per layer and phase (internal);
* decode counts against the independent row histogram:
  ``sum(counts[decode]) == top_k * sum(r * rows_hist[r])`` (a lost ``index_add_``
  update under CUDA-graph replay breaks it);
* target-layer tokens of ``final - snapshot 1`` against the server's
  ``vllm:iteration_tokens_total`` of the last /metrics poll (snapshot 1 precedes
  the first scheduled step: warm-up and capture runs call the model runner, not
  ``Worker.execute_model``, where the collector ticks);
* the final snapshot (written by ``close()`` on SIGTERM) is newer than the last
  periodic one and the file parses;
* with several rank files (``VLLM_EXPERT_STATS_ALL_RANKS=1``): counts of every
  rank against rank 0.
"""

import argparse
import glob
import json
import os
import statistics
import sys

import numpy as np

from tools.rdna.hetero_moe.stats_io import PHASES, parse_raw

CELLS = (
    "bench-1024x512-c1",
    "bench-1024x512-c6",
    "bench-1024x512-c8",
    "bench-16384x1024-c1",
    "bench-16384x1024-c8",
)


def _cell(arm_dir: str, cell: str) -> dict | None:
    path = os.path.join(arm_dir, cell + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def itl_table(run_dir: str) -> dict:
    arms = {a: os.path.join(run_dir, "arms", a) for a in ("A1", "B1", "B2", "A2")}
    out: dict[str, dict] = {}
    for cell in CELLS:
        row: dict[str, dict] = {}
        for metric in ("median_itl_ms", "mean_itl_ms", "output_throughput"):
            vals = {}
            for arm, d in arms.items():
                data = _cell(d, cell)
                if data is not None and metric in data:
                    vals[arm] = float(data[metric])
            row[metric] = vals
        out[cell] = row
    summary: dict[str, dict] = {}
    for cell, row in out.items():
        for metric, vals in row.items():
            if not all(a in vals for a in arms):
                continue
            a = [vals["A1"], vals["A2"]]
            b = [vals["B1"], vals["B2"]]
            mean_a, mean_b = statistics.mean(a), statistics.mean(b)
            spread_a, spread_b = abs(a[0] - a[1]), abs(b[0] - b[1])
            delta = mean_b - mean_a
            summary[f"{cell}:{metric}"] = {
                "mean_A": mean_a,
                "mean_B": mean_b,
                "delta": delta,
                "delta_pct": 100.0 * delta / mean_a,
                "spread_A": spread_a,
                "spread_B": spread_b,
                "resolved": abs(delta) > max(spread_a, spread_b),
            }
    return {"cells": out, "summary": summary}


def print_itl(table: dict) -> None:
    print("median ITL (ms) per arm, A = stats off, B = stats on (pairs)")
    print(
        f"{'cell':22}{'A1':>8}{'B1':>8}{'B2':>8}{'A2':>8} | {'mean A':>8}"
        f"{'mean B':>8}{'delta':>8}{'%':>7}{'sprA':>7}{'sprB':>7}  verdict"
    )
    for cell in CELLS:
        vals = table["cells"][cell]["median_itl_ms"]
        s = table["summary"].get(f"{cell}:median_itl_ms")
        cols = "".join(
            f"{vals.get(a, float('nan')):8.2f}" for a in ("A1", "B1", "B2", "A2")
        )
        if s is None:
            print(f"{cell:22}{cols} | incomplete")
            continue
        bound = abs(s["delta"]) + max(s["spread_A"], s["spread_B"])
        verdict = (
            f"resolved {s['delta']:+.2f} ms"
            if s["resolved"]
            else f"unresolved (<= {bound:.2f} ms)"
        )
        print(
            f"{cell:22}{cols} | {s['mean_A']:8.2f}{s['mean_B']:8.2f}"
            f"{s['delta']:+8.2f}{s['delta_pct']:+7.2f}{s['spread_A']:7.2f}"
            f"{s['spread_B']:7.2f}  {verdict}"
        )


def _last_metric(arm_dir: str, name: str) -> tuple[float | None, int | None]:
    """Value of the last block of ``metrics-poll.txt`` for a metric, and its epoch."""
    path = os.path.join(arm_dir, "metrics-poll.txt")
    if not os.path.exists(path):
        return None, None
    epoch, value = None, None
    with open(path) as f:
        for line in f:
            if line.startswith("epoch "):
                epoch = int(line.split()[1])
            elif line.startswith(name):
                value = float(line.rsplit(" ", 1)[1])
    return value, epoch


def _metric_file(path: str, name: str) -> float | None:
    if not os.path.exists(path):
        return None
    total = None
    with open(path) as f:
        for line in f:
            if line.startswith(name):
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    return total


def _flat(snap: dict, layer: str) -> list[int]:
    out: list[int] = []
    for p in PHASES:
        out += snap["counts"][layer][p]
        out.append(snap["tokens"][layer][p])
        out.append(snap["steps"][layer][p])
    out += snap["rows_hist"][layer]
    return out


def check_file(path: str) -> dict:
    meta, snaps = parse_raw(path)
    top_k = int(meta["top_k"])
    layers = [x["name"] for x in meta["layers"]]
    kinds = {x["name"]: x["kind"] for x in meta["layers"]}
    res: dict = {
        "file": os.path.basename(path),
        "tp_rank": meta.get("tp_rank"),
        "layers": len(layers),
        "targets": sum(k == "target" for k in kinds.values()),
        "snapshots": len(snaps),
        "top_k": top_k,
        "problems": [],
    }
    bad = res["problems"]
    for a, b in zip(snaps, snaps[1:], strict=False):
        if b["seq"] != a["seq"] + 1:
            bad.append(f"seq jump {a['seq']} -> {b['seq']}")
        if b["elapsed_s"] < a["elapsed_s"]:
            bad.append(f"elapsed_s decreases at seq {b['seq']}")
        for layer in layers:
            if layer not in a["counts"] or layer not in b["counts"]:
                continue
            if any(
                y < x for x, y in zip(_flat(a, layer), _flat(b, layer), strict=True)
            ):
                bad.append(f"counter decreased in {layer} at seq {b['seq']}")
                break
    last = snaps[-1]
    first = snaps[0]
    internal, hist_bad = 0, []
    for layer in layers:
        for p in PHASES:
            total = sum(last["counts"][layer][p])
            if total != top_k * last["tokens"][layer][p]:
                internal += 1
        rows = sum(r * c for r, c in enumerate(last["rows_hist"][layer]))
        dec = sum(last["counts"][layer]["decode"])
        if dec != top_k * rows:
            hist_bad.append((layer, dec, top_k * rows))
    res["sum_counts_vs_tokens_top_k_bad_layers"] = internal
    res["decode_counts_vs_rows_hist_bad_layers"] = len(hist_bad)
    res["decode_counts_vs_rows_hist_examples"] = hist_bad[:3]
    if internal:
        bad.append(f"{internal} layer/phase with sum(counts) != top_k*tokens")
    if hist_bad:
        bad.append(f"{len(hist_bad)} layers: decode counts != top_k * rows_hist rows")
    per_layer = {}
    for layer in layers:
        per_layer[layer] = sum(
            last["tokens"][layer][p] - first["tokens"][layer][p] for p in PHASES
        )
    tgt = {n: v for n, v in per_layer.items() if kinds[n] == "target"}
    res["target_tokens_final_minus_seq1"] = sorted(set(tgt.values()))
    res["mtp_tokens_final_minus_seq1"] = [
        v for n, v in per_layer.items() if kinds[n] != "target"
    ]
    res["first_snapshot_target_tokens"] = sorted(
        {
            sum(first["tokens"][n][p] for p in PHASES)
            for n in layers
            if kinds[n] == "target"
        }
    )
    res["final_elapsed_s"] = last["elapsed_s"]
    res["prev_elapsed_s"] = snaps[-2]["elapsed_s"] if len(snaps) > 1 else None
    res["final_newer_than_last_periodic"] = len(snaps) > 1 and _flat(
        last, layers[0]
    ) != _flat(snaps[-2], layers[0])
    return res


def check_arm(arm_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(arm_dir, "stats", "expert_stats-*.jsonl")))
    if not files:
        return {"error": f"no expert_stats-*.jsonl under {arm_dir}/stats"}
    results = [check_file(f) for f in files]
    out: dict = {"arm": arm_dir, "files": results}
    metric = "vllm:iteration_tokens_total_sum"
    last_val, last_epoch = _last_metric(arm_dir, metric)
    end_val = _metric_file(os.path.join(arm_dir, "metrics-load-end.txt"), metric)
    start_val = _metric_file(os.path.join(arm_dir, "metrics-load-start.txt"), metric)
    out["server_iteration_tokens_last_poll"] = last_val
    out["server_iteration_tokens_load_end"] = end_val
    out["server_iteration_tokens_load_start"] = start_val
    r0 = results[0]
    tt = r0["target_tokens_final_minus_seq1"]
    if last_val is not None and len(tt) == 1:
        out["target_tokens_vs_server"] = {
            "collector": tt[0],
            "server": last_val,
            "diff": tt[0] - last_val,
            "rel": (tt[0] - last_val) / max(last_val, 1.0),
        }
    if len(results) > 1:
        ref_meta, ref_snaps = parse_raw(files[0])
        agree = {}
        for f in files[1:]:
            meta, snaps = parse_raw(f)
            diff_layers = 0
            max_abs = 0
            max_rel = 0.0
            last, ref = snaps[-1], ref_snaps[-1]
            for layer in ref["counts"]:
                if layer not in last["counts"]:
                    diff_layers += 1
                    continue
                d = 0
                for p in PHASES:
                    d += int(
                        np.abs(
                            np.asarray(last["counts"][layer][p], dtype=np.int64)
                            - np.asarray(ref["counts"][layer][p], dtype=np.int64)
                        ).max()
                    )
                tot = max(sum(ref["counts"][layer]["decode"]), 1)
                if d:
                    diff_layers += 1
                max_abs = max(max_abs, d)
                max_rel = max(max_rel, d / tot)
            agree[f"rank{meta.get('tp_rank')}_vs_rank{ref_meta.get('tp_rank')}"] = {
                "layers_differing": diff_layers,
                "max_abs_count_diff": max_abs,
                "max_rel_to_layer_decode_total": max_rel,
                "snapshots": [len(snaps), len(ref_snaps)],
            }
        out["rank_agreement"] = agree
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("itl")
    a.add_argument("run_dir")
    a.add_argument("--json")
    c = sub.add_parser("check")
    c.add_argument("arm_dir")
    c.add_argument("--json")
    args = ap.parse_args(argv)
    if args.cmd == "itl":
        table = itl_table(args.run_dir)
        print_itl(table)
        result = table
    else:
        result = check_arm(args.arm_dir)
        print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    sys.exit(0)


if __name__ == "__main__":
    main()
