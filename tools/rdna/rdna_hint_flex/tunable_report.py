#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level table of a tunableop_hi2.sh campaign: heuristic vs tuned rocBLAS.

Joins the cold-weight graph timings (``measure heur|new``), the curation decisions
(``curate``) and the rowcheck numerics (``check``) per census key and prints a markdown
table (stdout) and, with ``--csv``, the same rows as CSV.

  python tunable_report.py --run $W [--csv table.csv]

Columns: slot, M, heuristic ms / TF/s, tuned ms / TF/s, gain, solver, decision, and the
rowcheck error against an fp64 matmul of the same fp16 inputs (gaussian activations:
max |err| / max |ref| and relative L2, rocBLAS default -> tuned) with whether the tuned
bits equal the default's.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SLOT = {
    (17408, 5120): "gate_up",
    (5120, 8704): "down",
    (8192, 5120): "in_proj_qkvz",
    (48, 5120): "in_proj_ba",
    (5120, 3072): "out_proj / o_proj",
    (7168, 5120): "qkv_proj",
    (2560, 10240): "mtp fc",
    (124160, 5120): "lm_head",
}


def parse_key(key):
    """(R, M, K) of a ``tn_<R>_<M>_<K>_ld_<lda>_<ldb>_<ldc>`` key."""
    _, r, m, k, *_ = key.split("_")
    return int(r), int(m), int(k)


def load(path):
    return json.loads(Path(path).read_text())


def solvers(rows_file):
    out = {}
    for line in Path(rows_file).read_text().splitlines():
        if line.startswith("Gemm"):
            _, key, solver, *_ = line.split(",")
            out[key] = solver
    return out


def build(run: Path):
    heur = load(run / "measure/heur.json")["results"]
    new = load(run / "measure/new.json")["results"]
    decisions = load(run / "curated/decisions.json")
    solver = solvers(run / "scratch/tunableop_results0.csv")
    check_d = load(run / "check/default.json")["keys"]
    check_t = load(run / "check/tuned.json")["keys"]
    rows = []
    for key, ms_h in heur.items():
        r, tokens, k = parse_key(key)
        ms_n = new.get(key, ms_h)
        flop = 2 * r * k * tokens
        row = {
            "slot": SLOT.get((r, k), f"{r}x{k}"),
            "R": r,
            "M": tokens,
            "K": k,
            "heur_ms": ms_h,
            "heur_tflops": flop / ms_h / 1e9,
            "tuned_ms": ms_n,
            "tuned_tflops": flop / ms_n / 1e9,
            "gain_pct": (ms_h / ms_n - 1) * 100,
            "solver": solver.get(key, "-"),
            "decision": decisions.get(key, "-"),
        }
        d, t = check_d.get(key), check_t.get(key)
        if d and t:
            g_d, g_t = d["gaussian"], t["gaussian"]
            row.update(
                err_default=g_d["max"],
                err_tuned=g_t["max"],
                l2_default=g_d["l2"],
                l2_tuned=g_t["l2"],
                bits_equal=g_d["eager"] == g_t["eager"]
                and d["outliers"]["eager"] == t["outliers"]["eager"],
            )
        rows.append(row)
    rows.sort(key=lambda x: (x["R"], x["K"], x["M"]))
    return rows


def markdown(rows):
    head = (
        "| slot | M | heuristic ms (TF/s) | tuned ms (TF/s) | gain | solver "
        "| decision | max err / max abs ref, default -> tuned "
        "| rel L2, default -> tuned | tuned bits == default |"
    )
    lines = [head, "|" + " --- |" * 10]
    for x in rows:
        err = (
            f"{x['err_default']:.2e} -> {x['err_tuned']:.2e}"
            if "err_default" in x
            else "-"
        )
        l2 = (
            f"{x['l2_default']:.2e} -> {x['l2_tuned']:.2e}"
            if "l2_default" in x
            else "-"
        )
        eq = "-" if "bits_equal" not in x else ("yes" if x["bits_equal"] else "no")
        lines.append(
            f"| {x['slot']} | {x['M']} | {x['heur_ms']:.3f} ({x['heur_tflops']:.1f}) "
            f"| {x['tuned_ms']:.3f} ({x['tuned_tflops']:.1f}) | {x['gain_pct']:+.1f} % "
            f"| {x['solver'].replace('Gemm_Rocblas_', '')} | {x['decision']} "
            f"| {err} | {l2} | {eq} |"
        )
    return "\n".join(lines)


# target GEMMs of one forward pass: (R, K) -> layers (64 layers: 48 GDN + 16 attention;
# out_proj and o_proj share a key)
LAYERS = {
    (17408, 5120): 64,
    (5120, 8704): 64,
    (8192, 5120): 48,
    (48, 5120): 48,
    (5120, 3072): 64,
    (7168, 5120): 16,
}


def forward(table, m):
    """(heuristic ms, rows ms, FLOP share of the slots whose row is adopted) of the
    target GEMMs of one forward at M tokens; None if a slot was not measured at M."""
    heur = rows = hit = 0.0
    for (r, k), layers in LAYERS.items():
        entry = table.get((r, m, k))
        if entry is None:
            return None
        ms_h, ms_t, adopted = entry
        heur += layers * ms_h
        rows += layers * (ms_t if adopted else ms_h)
        hit += layers * 2 * r * k * adopted
    return heur, rows, hit / sum(n * 2 * r * k for (r, k), n in LAYERS.items())


def expected(run: Path, census: Path, variant: str) -> str:
    """Model the GEMM-time saving of the rows per workload from the scheduler census.

    Per workload: sum over its steps with M > 48 of the target-GEMM time at that M,
    measured (cold-weight graph timing) where all six shapes were measured and scaled
    from the FLOPs at the mean measured TF/s elsewhere (the uncovered M). Only adopted
    rows count as tuned. Target layers only (the MTP drafter layer adds ~1.5 %).
    """
    heur = load(run / "measure/heur.json")["results"]
    new = load(run / "measure/new.json")["results"]
    decisions = load(run / "curated/decisions.json")
    table = {}
    flops = ms = 0.0
    for key, ms_h in heur.items():
        r, tokens, k = parse_key(key)
        adopted = decisions.get(key) in ("new", "current")
        table[(r, tokens, k)] = (ms_h, new.get(key, ms_h), adopted)
        flops += 2 * r * k * tokens
        ms += ms_h
    rate = flops / ms  # FLOP per ms over every measured key
    per_token = sum(layers * 2 * r * k for (r, k), layers in LAYERS.items())
    census_data = load(census)
    lines = [
        f"| workload ({variant}) | steps M>48 | rows hit (share of GEMM FLOPs) "
        "| GEMM time, heuristic | GEMM time, rows | saved |",
        "|" + " --- |" * 6,
    ]
    for name, per_variant in census_data["results"].items():
        res = per_variant.get(variant)
        if res is None:
            continue
        subs = res.items() if "segments" not in res else [(None, res)]
        for sub, r in subs:
            hist = (r.get("agg_run_only") or r["agg"])["hist_m_gt48"]
            t_h = t_t = hit_flops = all_flops = 0.0
            for m_str, n in hist.items():
                m = int(m_str)
                f = per_token * m
                all_flops += n * f
                got = forward(table, m)
                if got is None:
                    heur_ms = tuned_ms = f / rate
                else:
                    heur_ms, tuned_ms, share = got
                    hit_flops += n * f * share
                t_h += n * heur_ms
                t_t += n * tuned_ms
            if not t_h:
                continue
            label = name + (f"/{sub}" if sub else "")
            lines.append(
                f"| {label} | {sum(hist.values())} "
                f"| {100 * hit_flops / all_flops:.0f} % "
                f"| {t_h / 1e3:.2f} s | {t_t / 1e3:.2f} s "
                f"| {100 * (1 - t_t / t_h):.1f} % |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="the campaign's $W")
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--census",
        type=Path,
        help="sched_census.py census.json: print the modelled saving",
    )
    parser.add_argument("--variant", default="mtp0")
    args = parser.parse_args()
    if args.census:
        print(expected(args.run, args.census, args.variant))
        return 0
    rows = build(args.run)
    print(markdown(rows))
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(
                    dict.fromkeys(
                        [
                            *rows[0],
                            "err_default",
                            "err_tuned",
                            "l2_default",
                            "l2_tuned",
                            "bits_equal",
                        ]
                    )
                ),
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
