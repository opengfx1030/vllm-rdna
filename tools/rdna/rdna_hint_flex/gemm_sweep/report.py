# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Markdown tables from sweep.py CSVs (reference rows, best design per cell, ranking).

  python report.py run1.csv [run2.csv ...] [--top 12] [--bar 1.10 --floor 0.97]

Reads only the CSVs: no GPU. A design that appears in several files keeps its
fastest row per cell. ``vs_rb`` in the tables is always against the faster of
rocBLAS's heuristic and its TunableOp-tuned solution, both cold and measured in
the same interleaved window as the design.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

BIG3 = ("gate_up", "down", "in_proj_qkvz")  # the three largest N x K
ORDER = ["gate_up", "down", "in_proj_qkvz", "qkv_proj", "out_proj"]


def load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for k in ("N", "K", "M"):
            r[k] = int(r[k]) if r.get(k) else 0
        for k in ("us", "us_min", "tflops", "vs_rb", "max_abs", "vgpr", "lds", "scratch", "threads",
                  "blocks_per_wgp", "waves_per_simd", "gfx_mhz", "power_w"):  # fmt: skip
            try:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            except ValueError:
                r[k] = None
    return rows


def cells(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["shape"], r["M"])].append(r)
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def best_by(rs, kind, variant=None):
    c = [
        r
        for r in rs
        if r["kind"] == kind
        and (variant is None or r["variant"] == variant)
        and r["us"]
    ]
    return min(c, key=lambda r: r["us"]) if c else None


def sorted_cells(cs):
    return sorted(cs, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))


def reference_table(cs) -> str:
    out = [
        "| shape (N x K) | M | rocBLAS heur us | TF/s | tuned us | TF/s | hot (W0) heur us | fused_ref us | TF/s"
        " | dequant us | dequant+mm heur us | dequant+mm tuned us | clk MHz | W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted_cells(cs):
        rs = cs[key]
        h, t = best_by(rs, "rocblas_heur"), best_by(rs, "rocblas_tuned")
        hh, fu = best_by(rs, "rocblas_heur_hot"), best_by(rs, "fused_ref")
        dq, dh = best_by(rs, "dequant_only"), best_by(rs, "dequant_mm", "dequant+heur")
        dt = best_by(rs, "dequant_mm", "dequant+tuned")
        n, k = (h or t or fu or rs[0])["N"], (h or t or fu or rs[0])["K"]
        clk = (h or rs[0]).get("gfx_mhz"), (h or rs[0]).get("power_w")
        out.append(
            f"| {key[0]} ({n}x{k}) | {key[1]} | {fmt(h and h['us'])} | {fmt(h and h['tflops'])} | {fmt(t and t['us'])}"
            f" | {fmt(t and t['tflops'])} | {fmt(hh and hh['us'])} | {fmt(fu and fu['us'])} | {fmt(fu and fu['tflops'])}"
            f" | {fmt(dq and dq['us'])} | {fmt(dh and dh['us'])} | {fmt(dt and dt['us'])}"
            f" | {fmt(clk[0], 0)} | {fmt(clk[1], 0)} |"
        )
    return "\n".join(out)


def per_cell_best(cs, ok) -> str:
    out = [
        "| shape | M | best rocBLAS us (TF/s) | best design | us | TF/s | vs best rocBLAS | vs heuristic | VGPR | LDS B"
        " | waves/SIMD |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted_cells(cs):
        rs = cs[key]
        t, h = best_by(rs, "rocblas_tuned"), best_by(rs, "rocblas_heur")
        rb = min((x for x in (t, h) if x), key=lambda r: r["us"], default=None)
        dense = [r for r in rs if r["kind"] == "dense" and r["us"] and ok(r["variant"])]
        if not rb or not dense:
            continue
        b = min(dense, key=lambda r: r["us"])
        out.append(
            f"| {key[0]} | {key[1]} | {rb['us']:.1f} ({rb['tflops']:.2f}) | `{b['variant']}` | {b['us']:.1f} | {b['tflops']:.2f}"
            f" | {rb['us'] / b['us']:.3f} | {(h or rb)['us'] / b['us']:.3f} | {fmt(b['vgpr'], 0)} | {fmt(b['lds'], 0)}"
            f" | {fmt(b['waves_per_simd'], 1)} |"
        )
    return "\n".join(out)


def ranking(cs, ok, top, bar, floor):
    per = defaultdict(dict)  # design -> cell -> speed-up over the faster rocBLAS row
    perh = defaultdict(
        dict
    )  # design -> cell -> speed-up over the rocBLAS heuristic (what serving runs today)
    meta = {}
    for key, rs in cs.items():
        h, t = best_by(rs, "rocblas_heur"), best_by(rs, "rocblas_tuned")
        if not h or not t:
            continue
        rb = min(h["us"], t["us"])
        for r in rs:
            if r["kind"] == "dense" and r["us"] and ok(r["variant"]):
                per[r["variant"]][key] = rb / r["us"]
                perh[r["variant"]][key] = h["us"] / r["us"]
                meta[r["variant"]] = r
    allkeys = sorted_cells(cs)
    out = [
        "| design | cells | geomean vs best rocBLAS | min | worst cell | big-3 min | cells >= bar | geomean vs heuristic"
        " | min vs heuristic | VGPR | LDS B | waves/SIMD |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for v, d in per.items():
        if len(d) < max(1, len(allkeys) // 2):
            continue
        gm = math.exp(statistics.mean(math.log(x) for x in d.values()))
        worst = min(d, key=d.get)
        big = [x for k2, x in d.items() if k2[0] in BIG3]
        rows.append((gm, v, d, worst, big))
    rows.sort(reverse=True, key=lambda t: t[0])
    for gm, v, d, worst, big in rows[:top]:
        m, dh = meta[v], perh[v]
        gmh = math.exp(statistics.mean(math.log(x) for x in dh.values()))
        out.append(
            f"| `{v}` | {len(d)}/{len(allkeys)} | {gm:.3f} | {min(d.values()):.3f} | {worst[0]} M={worst[1]} "
            f"| {fmt(min(big) if big else None, 3)} | {sum(x >= bar for x in d.values())} | {gmh:.3f} "
            f"| {min(dh.values()):.3f} | {fmt(m['vgpr'], 0)} | {fmt(m['lds'], 0)} | {fmt(m['waves_per_simd'], 1)} |"
        )
    return "\n".join(out), rows


def matrix_table(cs, designs: list[str]) -> str:
    """One row per cell; us (TF/s) for rocBLAS heuristic / tuned, the existing fused kernel and each design."""
    cols = [("rocblas_heur", None, "rocBLAS heur"), ("rocblas_tuned", None, "rocBLAS tuned"),
            ("fused_ref", None, "fused_ref")] + [("dense", d, f"`{d}`") for d in designs]  # fmt: skip
    out = [
        "| shape | M | " + " | ".join(c[2] for c in cols) + " |",
        "|---|---:|" + "---:|" * len(cols),
    ]
    for key in sorted_cells(cs):
        cells_ = []
        for kind, variant, _ in cols:
            r = best_by(cs[key], kind, variant)
            cells_.append("-" if r is None else f"{r['us']:.0f} ({r['tflops']:.1f})")
        out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells_) + " |")
    return "\n".join(out)


def oracle_summary(cs, ok, bar: float, floor: float) -> str:
    """What a per-cell dispatcher (best dense design where it wins, rocBLAS elsewhere) would achieve."""
    gains, gains_h, n = [], [], defaultdict(int)
    for key in sorted_cells(cs):
        rs = cs[key]
        t, h = best_by(rs, "rocblas_tuned"), best_by(rs, "rocblas_heur")
        dense = [r for r in rs if r["kind"] == "dense" and r["us"] and ok(r["variant"])]
        if not (t and h and dense):
            continue
        rb, b = min(t["us"], h["us"]), min(r["us"] for r in dense)
        sp, sp_h = rb / b, h["us"] / b
        n["cells"] += 1
        n[">= bar"] += sp >= bar
        n["> 1.03"] += sp > 1.03
        n["within 3 %"] += floor <= sp <= 1.03
        n[f"< {floor}"] += sp < floor
        gains.append(max(sp, 1.0))
        gains_h.append(max(sp_h, 1.0))
    if not gains:
        return "(no cells)"
    gm = math.exp(statistics.mean(math.log(g) for g in gains))
    gmh = math.exp(statistics.mean(math.log(g) for g in gains_h))
    return (f"cells {n['cells']}: dense best >= {bar}x: {n['>= bar']}, > 1.03x: {n['> 1.03']}, within 3 %: "
            f"{n['within 3 %']}, < {floor}x: {n[f'< {floor}']}; per-cell dispatcher (dense only where it wins) "
            f"geomean {gm:.3f}x over the faster rocBLAS row, {gmh:.3f}x over the heuristic")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument(
        "--bar", type=float, default=1.10, help="required speed-up on the big shapes"
    )
    ap.add_argument("--floor", type=float, default=0.97, help="slowest acceptable cell")
    ap.add_argument(
        "--only-correct",
        default="",
        help="comma list of variants verified correct (default: all)",
    )
    ap.add_argument(
        "--check",
        help="check.csv from `sweep.py check --csv`: only designs with status ok count",
    )
    ap.add_argument(
        "--designs", default="", help="comma list of designs for the per-cell matrix"
    )
    args = ap.parse_args()
    rows = load(args.csv)
    good = {v for v in args.only_correct.split(",") if v}
    if args.check:
        with open(args.check, newline="") as f:
            good |= {r["variant"] for r in csv.DictReader(f) if r["status"] == "ok"}
    # candidates are MODE 0 designs (the m1..m5 builds are roofline / occupancy probes)
    ok = lambda v: "m0u" in v and (not good or v in good)  # noqa: E731
    cs = cells(rows)
    print("## Reference rows (cold rotating weights, sustained clocks, medians)\n")
    print(reference_table(cs))
    print("\n## Best dense design per cell\n")
    print(per_cell_best(cs, ok))
    print(f"\n## Ranking over all cells (top {args.top} by geometric mean)\n")
    tab, rows_ = ranking(cs, ok, args.top, args.bar, args.floor)
    print(tab)
    print("\n## Per-cell dispatcher (oracle over the designs in the CSVs)\n")
    print(oracle_summary(cs, ok, args.bar, args.floor))
    if args.designs:
        print("\n## Selected designs: us (TF/s) per cell\n")
        print(matrix_table(cs, [d for d in args.designs.split(",") if d]))
    if rows_:
        gm, v, d, worst, big = rows_[0]
        passed = (
            bool(big)
            and all(x >= args.bar for x in big)
            and min(d.values()) >= args.floor
        )
        print(f"\nBest single design: `{v}` geomean {gm:.3f}; big-3 min {min(big):.3f}; worst {min(d.values()):.3f}"
              f" -> bar (>= {args.bar} on every big-3 cell, >= {args.floor} everywhere): {'MET' if passed else 'NOT MET'}")  # fmt: skip


if __name__ == "__main__":
    main()
