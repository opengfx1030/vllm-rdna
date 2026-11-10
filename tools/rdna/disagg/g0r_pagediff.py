# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte/element-level difference of one state page between two tracer raw dumps.

    g0r_pagediff.py A.bin:A.json B.bin:B.json [--group 0] [--col 0] [--layers 0,1]

Each side is a raw dump written by g0r_trace.py (``raw_<pre|post>_<pos>.bin`` plus its
json). For every (layer) page of the group and column present on both sides it prints
how many bytes differ, where (first and last offset, the 64 KiB regions that hold a
difference) and, reading the page as float32 and float16, the largest absolute and
relative element difference. It tells whether a whole-page fingerprint mismatch is a
numerical difference of the state or a few dead bytes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load(spec: str) -> dict[tuple[int, int, int], np.ndarray]:
    bin_path, json_path = spec.split(":")
    with open(json_path) as f:
        meta = json.load(f)
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r")
    out = {}
    for it in meta["items"]:
        out[(it["g"], it["col"], it["layer"])] = raw[it["off"] : it["off"] + it["n"]]
    return out


def elem_stats(a: np.ndarray, b: np.ndarray, dtype: str) -> str:
    n = len(a) // np.dtype(dtype).itemsize * np.dtype(dtype).itemsize
    x = a[:n].view(dtype).astype(np.float64)
    y = b[:n].view(dtype).astype(np.float64)
    fin = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x - y)[fin]
    if d.size == 0 or d.max() == 0:
        return f"{dtype}: equal"
    scale = np.maximum(np.abs(x), np.abs(y))[fin]
    rel = d / np.maximum(scale, 1e-30)
    return (
        f"{dtype}: {int((d > 0).sum())} elems differ, max abs {d.max():.3g}, "
        f"max rel {rel.max():.3g}, max |value| {scale.max():.3g}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--layers", default="0,1")
    args = ap.parse_args()
    ra, rb = load(args.a), load(args.b)
    for layer in (int(x) for x in args.layers.split(",")):
        key = (args.group, args.col, layer)
        if key not in ra or key not in rb:
            print(f"g{args.group} col{args.col} layer{layer}: missing on one side")
            continue
        a, b = np.asarray(ra[key]), np.asarray(rb[key])
        n = min(len(a), len(b))
        diff = np.nonzero(a[:n] != b[:n])[0]
        if diff.size == 0:
            print(f"g{args.group} col{args.col} layer{layer}: {n} bytes equal")
            continue
        regions = sorted({int(o) >> 16 for o in diff})
        print(
            f"g{args.group} col{args.col} layer{layer}: {diff.size}/{n} bytes differ, "
            f"offsets {int(diff[0])}..{int(diff[-1])}, 64KiB regions {regions}"
        )
        for dt in ("float32", "float16"):
            print("    " + elem_stats(a[:n], b[:n], dt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
