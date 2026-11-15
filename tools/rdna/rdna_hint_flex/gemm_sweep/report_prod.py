# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Markdown tables from ``bench_prod.py`` CSVs (no GPU).

  python report_prod.py prod_a1.csv [more.csv ...] [--designs "D1,D4"] [--min-m 512]

Per cell (slot, M): the rocBLAS heuristic and TunableOp rows, the table-driven op
(``hip_auto``: what ``RDNA_HINT_FLEX_PREFILL_GEMM=hip`` runs), and the fastest
design of the cell (kinds ``hip`` / ``dev``). ``vs best`` = better of the two
rocBLAS rows / the row, ``vs heur`` = heuristic row / the row. A later file
overrides an earlier one for the same (kind, variant, N, K, M).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

ORDER = ["gate_up", "down", "in_proj_qkvz", "qkv_proj", "out_proj"]
DESIGN_KINDS = ("hip", "dev")


def load(paths: list[str]) -> dict[tuple, dict]:
    seen: dict[tuple, dict] = {}
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("us"):
                    continue
                r["N"], r["K"], r["M"] = int(r["N"]), int(r["K"]), int(r["M"])
                for key in ("us", "tflops", "gfx_mhz", "power_w"):
                    r[key] = float(r[key]) if r.get(key) else None
                seen[(r["kind"], r["variant"], r["N"], r["K"], r["M"])] = r
    return seen


def short(v: str) -> str:
    return v if len(v) < 14 else v.split("g16")[0]


def geomean(xs: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+")
    ap.add_argument(
        "--designs", default="", help="comma list of variants to keep (substring match)"
    )
    ap.add_argument("--min-m", type=int, default=0)
    args = ap.parse_args()
    keep = [s for s in args.designs.split(",") if s]
    rows = load(args.csv)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows.values():
        cells[(r["shape"], r["M"])].append(r)

    def key(c):
        return (ORDER.index(c[0]) if c[0] in ORDER else 99, c[1])

    out = [
        "| slot | M | heur us | tuned us | best rb TF/s | table pick | us | TF/s | MHz | vs best | vs heur | fastest design | us | vs best | vs heur |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    auto_best, auto_heur, top_best, top_heur = [], [], [], []
    worst = (9.0, None)
    for cell in sorted(cells, key=key):
        shape, m = cell
        if m < args.min_m:
            continue
        rs = cells[cell]
        heur = next((r for r in rs if r["kind"] == "rocblas_heur"), None)
        tuned = next((r for r in rs if r["kind"] == "rocblas_tuned"), None)
        if heur is None:
            continue
        rb = min(r["us"] for r in (heur, tuned) if r is not None)
        flop = 2.0 * m * heur["N"] * heur["K"]
        auto = next((r for r in rs if r["kind"] == "hip_auto"), None)
        designs = [
            r for r in rs
            if r["kind"] in DESIGN_KINDS and (not keep or any(s in r["variant"] for s in keep))
        ]  # fmt: skip
        top = min(designs, key=lambda r: r["us"]) if designs else None
        cols = [
            shape,
            str(m),
            f"{heur['us']:.0f}",
            f"{tuned['us']:.0f}" if tuned else "-",
            f"{flop / rb / 1e6:.1f}",
        ]
        if auto:
            pick = (auto.get("note") or "").split("pick ")[-1].split(" ")[0]
            cols += [
                "rocBLAS" if pick == "-1" else f"D{int(pick) + 1}",
                f"{auto['us']:.0f}", f"{auto['tflops']:.1f}" if auto["tflops"] else "-",
                f"{auto['gfx_mhz']:.0f}" if auto["gfx_mhz"] else "-",
                f"{rb / auto['us']:.3f}", f"{heur['us'] / auto['us']:.3f}",
            ]  # fmt: skip
            auto_best.append(rb / auto["us"])
            auto_heur.append(heur["us"] / auto["us"])
            if rb / auto["us"] < worst[0]:
                worst = (rb / auto["us"], cell)
        else:
            cols += ["-"] * 6
        if top:
            cols += [
                short(top["variant"]),
                f"{top['us']:.0f}",
                f"{rb / top['us']:.3f}",
                f"{heur['us'] / top['us']:.3f}",
            ]
            top_best.append(rb / top["us"])
            top_heur.append(heur["us"] / top["us"])
        else:
            cols += ["-"] * 4
        out.append("| " + " | ".join(cols) + " |")
    print("\n".join(out))
    print()
    if auto_best:
        print(
            f"table-driven op ({len(auto_best)} cells): geomean {geomean(auto_best):.3f} vs better rocBLAS row, "
            f"{geomean(auto_heur):.3f} vs heuristic; worst {worst[0]:.3f} at {worst[1]}; "
            f">= 1.10: {sum(x >= 1.10 for x in auto_best)}; < 1.00: {sum(x < 1.0 for x in auto_best)}; < 0.97: {sum(x < 0.97 for x in auto_best)}"
        )
    if top_best:
        print(
            f"fastest design per cell ({len(top_best)} cells): geomean {geomean(top_best):.3f} vs better rocBLAS row, "
            f"{geomean(top_heur):.3f} vs heuristic; >= 1.10: {sum(x >= 1.10 for x in top_best)}"
        )
    # matrix: every design against the better rocBLAS row
    variants = sorted(
        {
            r["variant"]
            for r in rows.values()
            if r["kind"] in DESIGN_KINDS
            and (not keep or any(s in r["variant"] for s in keep))
        }
    )
    if variants:
        print("\n| slot | M | " + " | ".join(short(v) for v in variants) + " |")
        print("| --- | ---: | " + " | ".join("---:" for _ in variants) + " |")
        for cell in sorted(cells, key=key):
            if cell[1] < args.min_m:
                continue
            rs = cells[cell]
            base = [
                r["us"] for r in rs if r["kind"] in ("rocblas_heur", "rocblas_tuned")
            ]
            if not base:
                continue
            rb = min(base)
            vals = {r["variant"]: rb / r["us"] for r in rs if r["kind"] in DESIGN_KINDS}
            print(
                f"| {cell[0]} | {cell[1]} | "
                + " | ".join(f"{vals[v]:.3f}" if v in vals else "-" for v in variants)
                + " |"
            )


if __name__ == "__main__":
    main()
