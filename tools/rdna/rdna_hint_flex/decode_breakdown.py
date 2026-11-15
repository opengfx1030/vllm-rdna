#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-decode-step GPU kernel-time breakdown from rocprofv3 kernel traces.

    python3 tools/rdna/rdna_hint_flex/decode_breakdown.py TRACE_DIR \
        [--labels c1,c4] [--hbm-gbs 512] [--md out.md] [--json out.json]

TRACE_DIR holds ``<pid>_kernel_trace.csv`` files of profiled decode windows
(``tools/rdna/rdna_hint_flex/decode_profile.py`` writes them with rocprofv3
``--kernel-trace --selected-regions``; CUDA-graph replays are recorded kernel
by kernel). Each file is one TP rank; windows are separated by idle gaps of
``--window-gap-s``. Steps run back to back on the GPU (async scheduling), so a
window is cut at a kernel that runs once per step (``--marker``, default the
v2 runner's ``scatter_num_accepted_kernel``); without markers it falls back to
idle gaps of ``--step-gap-us``. Steps whose kernel count differs from the
median by more than 1% are dropped.

Three tables per rank and window:

* kernel families (name regexes): launches, kernel time, and the idle time in
  front of the family's launches (launch gaps inside the replayed graph);
* the hi2 GEMVs by layer type, cut from their launch order (a GDN layer runs
  in_proj_qkvz, in_proj_ba, out_proj, gate_up, down; a full-attention layer
  runs qkv, o_proj, gate_up, down; layers repeat 3 GDN + 1 attention) with
  the achieved GB/s from the per-rank checkpoint bytes of that layer type
  (rank_bytes.py over the checkpoint header) and the time at the HBM peak;
* phases: every non-GEMV kernel is charged to the GEMV that precedes it
  ("after gdn in_proj_ba" is the whole GDN core, "after gdn out_proj" the
  all-reduce and residual/norm glue, ...), so the glue between the GEMVs is
  visible per layer position.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import regex as re

# Kernel families, first match wins. Names are the demangled rocprofv3 names.
FAMILIES: list[tuple[str, str]] = [
    ("hi2 gemv", r"(?:^|[\s:])gemv(?:_i8)?_kernel|rdna_hint_flex.*gemv"),
    ("hi2 act transform", r"act_(?:transform|quant)(?:64)?_kernel"),
    ("lm_head gemv (dense fp16)", r"gemv_f16_rdna2|gemv_act_k"),
    ("rocBLAS gemm", r"^Cijk_|rocblas"),
    ("all-reduce (rdna_ar one-shot)", r"rdna_ar"),
    ("all-reduce (RCCL)", r"nccl|rccl"),
    (
        "GDN recurrent decode",
        r"gdn_decode|fused_recurrent_gated_delta|fused_sigmoid_gating|"
        r"gdn_leapquant|mono_gdn|gdn_spec",
    ),
    ("GDN conv1d update", r"causal_conv1d"),
    ("GDN gated rmsnorm", r"gated_rms_norm"),
    ("attention decode", r"fa_decode|fa_rdna2|paged_attn|splitk"),
    ("kv cache write", r"reshape_and_cache"),
    ("rmsnorm (add + norm)", r"rms_norm|layer_norm|layernorm"),
    (
        "sampling / logits",
        r"gumbel|argmax|topk|top_k|topp|top_p|sort|softmax|rdna2_sample|"
        r"get_top|multinomial|sampler",
    ),
    (
        "runner glue (input prep, mamba align)",
        r"rdna2_|prepare_|expand_idx|batch_memcpy|mamba|post_update|"
        r"compute_slot|gather_block",
    ),
    (
        "index / gather / scatter (torch)",
        r"index_elementwise|index_copy|gather|scatter|indexSelect|index_select",
    ),
    ("triton pointwise (inductor)", r"^triton_"),
    ("copyBuffer / memset kernels", r"__amd_rocclr"),
    (
        "elementwise / reduce (torch)",
        r"elementwise|vectorized|unrolled|reduce_kernel|Functor|CatArray|"
        r"fill_|index_put|masked|where|clamp",
    ),
]
_FAM = [(n, re.compile(rx)) for n, rx in FAMILIES]

# Per-rank bytes of the hi2 arrays of one launch (TP=2), MiB: sums of the
# hi2_* tensors of that layer type over the checkpoint header / layer count.
MIB = 1 << 20
HI2_27B_TP2_MIB = {
    "gdn in_proj_qkvz": (408.2 + 283.3) / 48,
    "gdn in_proj_ba": (3.6 + 3.6) / 48,
    "gdn out_proj": 340.8 / 48,
    "attn qkv": (160.0 + 22.6 + 21.1) / 16,
    "attn o_proj": 115.3 / 16,
    "mlp gate_up": (977.7 + 1003.0) / 64,
    "mlp down": 1039.9 / 64,
}
GDN_LAYER = [
    "gdn in_proj_qkvz",
    "gdn in_proj_ba",
    "gdn out_proj",
    "mlp gate_up",
    "mlp down",
]
ATTN_LAYER = ["attn qkv", "attn o_proj", "mlp gate_up", "mlp down"]
GEMVS_PER_STEP = 48 * len(GDN_LAYER) + 16 * len(ATTN_LAYER)
LM_HEAD_BYTES_PER_RANK = 248320 / 2 * 5120 * 2  # fp16 vocab shard


def family(name: str) -> str:
    for fam, rx in _FAM:
        if rx.search(name):
            return fam
    return "other"


def short(name: str, n: int = 110) -> str:
    return re.sub(r"^void\s+", "", name)[:n]


class Row:
    __slots__ = ("s", "e", "name", "fam", "grid", "wg")


def read_trace(path: str) -> list[Row]:
    rows: list[Row] = []
    fam_cache: dict[str, str] = {}
    with open(path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {h: i for i, h in enumerate(head)}
        i_s, i_e, i_n = ix["Start_Timestamp"], ix["End_Timestamp"], ix["Kernel_Name"]
        i_gx = ix.get("Grid_Size_X")
        i_wx = ix.get("Workgroup_Size_X")
        for r in rd:
            row = Row()
            row.s, row.e, row.name = int(r[i_s]), int(r[i_e]), r[i_n]
            fam = fam_cache.get(row.name)
            if fam is None:
                fam = fam_cache[row.name] = family(row.name)
            row.fam = fam
            row.grid = int(r[i_gx]) if i_gx is not None else 0
            row.wg = int(r[i_wx]) if i_wx is not None else 0
            rows.append(row)
    rows.sort(key=lambda x: (x.s, x.e))
    return rows


def split_runs(rows: list[Row], gap_ns: int) -> list[list[Row]]:
    runs: list[list[Row]] = []
    cur: list[Row] = []
    last_end = None
    for r in rows:
        if last_end is not None and r.s - last_end >= gap_ns:
            runs.append(cur)
            cur = []
        cur.append(r)
        last_end = r.e if last_end is None else max(last_end, r.e)
    if cur:
        runs.append(cur)
    return runs


def cut_steps(rows: list[Row], marker, gap_ns: int):
    """Steps [start, next start) of one window and the wall period of each."""
    idx = [i for i, r in enumerate(rows) if marker.search(r.name)] if marker else []
    if len(idx) >= 3:
        cuts = idx
        how = "marker"
    else:
        runs = split_runs(rows, gap_ns)
        cuts = []
        pos = 0
        for run in runs:
            cuts.append(pos)
            pos += len(run)
        how = "gap"
    steps = [rows[a:b] for a, b in zip(cuts, cuts[1:])]
    walls = [rows[b].s - rows[a].s for a, b in zip(cuts, cuts[1:])]
    return steps, walls, how


def union_ns(rows: list[Row]) -> int:
    tot = 0
    cs = ce = None
    for r in rows:  # rows are start-sorted
        if cs is None:
            cs, ce = r.s, r.e
        elif r.s <= ce:
            ce = max(ce, r.e)
        else:
            tot += ce - cs
            cs, ce = r.s, r.e
    if cs is not None:
        tot += ce - cs
    return tot


def label_gemvs(step: list[Row]) -> dict[int, str]:
    """Layer-type label of every hi2 GEMV launch of one step, by launch order."""
    idx = [i for i, r in enumerate(step) if r.fam == "hi2 gemv"]
    if len(idx) != GEMVS_PER_STEP:
        return {}
    out: dict[int, str] = {}
    pos = 0
    layer = 0
    while pos < len(idx):
        for slot in ATTN_LAYER if layer % 4 == 3 else GDN_LAYER:
            out[idx[pos]] = slot
            pos += 1
        layer += 1
    return out


def analyse(rows: list[Row], marker, gap_us: float):
    steps, walls, how = cut_steps(rows, marker, int(gap_us * 1000))
    if len(steps) < 3:
        return None
    counts = [len(s) for s in steps]
    mid = statistics.median(counts)
    keep = [i for i, c in enumerate(counts) if abs(c - mid) <= max(3, 0.01 * mid)]
    acc = {
        k: defaultdict(list)
        for k in ("fam_t", "fam_n", "fam_g", "gv_t", "gv_n", "ph_t", "ph_n", "ph_g")
    }
    per = []
    sig: Counter = Counter()
    other: Counter = Counter()
    for i in keep:
        st = steps[i]
        span = max(r.e for r in st) - st[0].s
        per.append((walls[i], span, union_ns(st), sum(r.e - r.s for r in st)))
        lab = label_gemvs(st)
        cur = "prologue (input prep, embedding)"
        ft, fn, fg = defaultdict(int), defaultdict(int), defaultdict(int)
        gt, gn = defaultdict(int), defaultdict(int)
        pt, pn, pg = defaultdict(int), defaultdict(int), defaultdict(int)
        end = st[0].s
        last_gemv = max(lab) if lab else -1
        for j, r in enumerate(st):
            d = r.e - r.s
            gap = max(r.s - end, 0)
            end = max(end, r.e)
            ft[r.fam] += d
            fn[r.fam] += 1
            fg[r.fam] += gap
            if r.fam == "other":
                other[short(r.name)] += d
            if j in lab:
                gt[lab[j]] += d
                gn[lab[j]] += 1
            if lab:
                if j in lab:
                    key = "gemv " + lab[j]
                    cur = "after " + lab[j]
                    if j == last_gemv:
                        cur = "epilogue (norm, lm_head, sampling)"
                else:
                    key = cur
                pt[key] += d
                pn[key] += 1
                pg[key] += gap
        for name, src in (
            ("fam", (ft, fn, fg)),
            ("gv", (gt, gn)),
            ("ph", (pt, pn, pg)),
        ):
            keys = set().union(*[set(s) for s in src]) | set(acc[name + "_t"])
            for k in keys:
                acc[name + "_t"][k].append(src[0].get(k, 0))
                acc[name + "_n"][k].append(src[1].get(k, 0))
                if len(src) > 2:
                    acc[name + "_g"][k].append(src[2].get(k, 0))
        if i == keep[0]:
            for r in st:
                if r.fam == "hi2 gemv":
                    sig[(r.grid, r.wg)] += 1
    return {
        "how": how,
        "n_kept": len(keep),
        "n_all": len(steps),
        "kernels": int(mid),
        "per": per,
        "acc": acc,
        "sig": sig,
        "other": other,
    }


def med(v):
    return statistics.median(v) if v else 0.0


def render(title: str, a: dict, hbm_gbs: float) -> tuple[str, dict]:
    per = a["per"]
    wall = med([p[0] for p in per]) / 1e6
    span = med([p[1] for p in per]) / 1e6
    busy = med([p[2] for p in per]) / 1e6
    sumd = med([p[3] for p in per]) / 1e6
    acc = a["acc"]
    n = a["kernels"]
    lines = [
        f"### {title}: {a['n_kept']} steps kept of {a['n_all']} (cut by {a['how']}), "
        f"{n} kernels per step",
        "",
        f"step wall (start to next start) median {wall:.3f} ms; first start to "
        f"last end {span:.3f} ms; GPU busy {busy:.3f} ms; sum of kernel durations "
        f"{sumd:.3f} ms; idle {wall - busy:.3f} ms "
        f"({1000 * (wall - busy) / max(n, 1):.2f} us per launch)",
        "",
        "| family | launches / step | kernel time / step (ms) | idle before (ms) "
        "| share of kernel time | share of step wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = sorted(
        (
            (
                med(acc["fam_t"][f]) / 1e6,
                f,
                med(acc["fam_n"][f]),
                med(acc["fam_g"][f]) / 1e6,
            )
            for f in acc["fam_t"]
        ),
        reverse=True,
    )
    tot_t = sum(r[0] for r in rows)
    tot_g = sum(r[3] for r in rows)
    for t, fam, cnt, g in rows:
        lines.append(
            f"| {fam} | {cnt:.0f} | {t:.3f} | {g:.3f} | {100 * t / tot_t:.1f}% | "
            f"{100 * t / wall:.1f}% |"
        )
    lines.append(
        f"| **total** | {n} | {tot_t:.3f} | {tot_g:.3f} | 100% | "
        f"{100 * tot_t / wall:.1f}% |"
    )
    res = {
        "title": title,
        "steps": a["n_kept"],
        "kernels_per_step": n,
        "wall_ms": wall,
        "span_ms": span,
        "busy_ms": busy,
        "sum_kernel_ms": sumd,
        "families": {
            f: {"time_ms": t, "launches": c, "idle_before_ms": g} for t, f, c, g in rows
        },
    }
    if acc["gv_t"]:
        lines += [
            "",
            "hi2 GEMV by layer type (per TP=2 rank):",
            "",
            "| layer type | launches / step | time / step (ms) | us / launch "
            "| MiB / launch | GB/s | floor at HBM peak (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        gtot_t = gtot_b = 0.0
        gres = {}
        for name, mib in HI2_27B_TP2_MIB.items():
            if name not in acc["gv_t"]:
                continue
            t = med(acc["gv_t"][name]) / 1e6
            cnt = med(acc["gv_n"][name])
            nbytes = mib * MIB * cnt
            gb = nbytes / (t * 1e6) if t else 0.0  # bytes / ns = GB/s
            floor = nbytes / (hbm_gbs * 1e6)
            gtot_t += t
            gtot_b += nbytes
            gres[name] = {"launches": cnt, "time_ms": t, "gbs": gb, "floor_ms": floor}
            lines.append(
                f"| {name} | {cnt:.0f} | {t:.3f} | {1000 * t / cnt if cnt else 0:.1f} "
                f"| {mib:.2f} | {gb:.0f} | {floor:.3f} |"
            )
        n_all = sum(med(acc["gv_n"][k]) for k in acc["gv_n"])
        lines.append(
            f"| **all hi2 GEMV** | {n_all:.0f} | {gtot_t:.3f} | | {gtot_b / MIB:.0f} "
            f"| {gtot_b / (gtot_t * 1e6) if gtot_t else 0:.0f} "
            f"| {gtot_b / (hbm_gbs * 1e6):.3f} |"
        )
        lm_floor = LM_HEAD_BYTES_PER_RANK / (hbm_gbs * 1e6)
        lm_t = med(acc["fam_t"].get("lm_head gemv (dense fp16)", [0])) / 1e6
        floor_all = gtot_b / (hbm_gbs * 1e6) + lm_floor
        lines += [
            "",
            f"weight-read floor at {hbm_gbs:.0f} GB/s: hi2 "
            f"{gtot_b / (hbm_gbs * 1e6):.2f} ms + lm_head {lm_floor:.2f} ms "
            f"({LM_HEAD_BYTES_PER_RANK / 1e9:.2f} GB) = {floor_all:.2f} ms; "
            f"measured hi2 GEMV {gtot_t:.2f} ms + lm_head gemv {lm_t:.2f} ms "
            f"= {gtot_t + lm_t:.2f} ms",
        ]
        res["hi2_gemv_by_layer"] = gres
        res["weight_floor_ms"] = floor_all
        res["weight_read_measured_ms"] = gtot_t + lm_t
        lines += [
            "",
            "phases (a non-GEMV kernel is charged to the GEMV before it):",
            "",
            "| phase | launches / step | kernel time / step (ms) | idle before (ms) |",
            "| --- | ---: | ---: | ---: |",
        ]
        order = (
            ["prologue (input prep, embedding)"]
            + [f"gemv {k}" for k in HI2_27B_TP2_MIB]
            + [f"after {k}" for k in HI2_27B_TP2_MIB]
            + ["epilogue (norm, lm_head, sampling)"]
        )
        pres = {}
        for k in order:
            if k not in acc["ph_t"]:
                continue
            t = med(acc["ph_t"][k]) / 1e6
            cnt = med(acc["ph_n"][k])
            g = med(acc["ph_g"][k]) / 1e6
            pres[k] = {"time_ms": t, "launches": cnt, "idle_before_ms": g}
            lines.append(f"| {k} | {cnt:.0f} | {t:.3f} | {g:.3f} |")
        res["phases"] = pres
    if a["sig"]:
        sigs = ", ".join(f"({g},{w}): {c}" for (g, w), c in sorted(a["sig"].items()))
        lines += [
            "",
            "hi2 gemv launch signatures in the first kept step "
            f"(grid_x, workgroup_x: launches): {sigs}",
        ]
    if a["other"]:
        top = "; ".join(f"{k} {v / 1e3:.1f}" for k, v in a["other"].most_common(6))
        lines += ["", f"largest unclassified kernels (first kept step, us): {top}"]
    return "\n".join(lines), res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace_dir")
    ap.add_argument("--marker", default="scatter_num_accepted_kernel")
    ap.add_argument("--step-gap-us", type=float, default=120.0)
    ap.add_argument("--hbm-gbs", type=float, default=512.0)
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--top", type=int, default=0, help="also list the N largest kernel names"
    )
    ap.add_argument(
        "--window-gap-s",
        type=float,
        default=0.5,
        help="idle gap that separates profiler windows (one per profile cell)",
    )
    ap.add_argument("--labels", default="", help="names of the windows, in time order")
    args = ap.parse_args()

    pattern = os.path.join(args.trace_dir, "**", "*kernel_trace.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if os.path.getsize(f) > 2000]
    if not files:
        print("no kernel_trace.csv with data under", args.trace_dir, file=sys.stderr)
        return 1
    marker = re.compile(args.marker) if args.marker else None
    labels = [x for x in args.labels.split(",") if x]
    out = []
    results = []
    for f in files:
        rows = read_trace(f)
        wins = split_runs(rows, int(args.window_gap_s * 1e9))
        for k, wrows in enumerate(w for w in wins if len(w) >= 200):
            wname = labels[k] if k < len(labels) else f"window {k}"
            title = f"{wname} | {os.path.basename(f)}"
            a = analyse(wrows, marker, args.step_gap_us)
            if a is None or a["n_kept"] < 3:
                out.append(f"### {title}: no complete steps")
                continue
            text, res = render(title, a, args.hbm_gbs)
            out.append(text)
            res["window"] = wname
            res["file"] = os.path.basename(f)
            results.append(res)
            if args.top:
                names: Counter = Counter()
                cnt: Counter = Counter()
                for r in wrows:
                    names[short(r.name, 140)] += r.e - r.s
                    cnt[short(r.name, 140)] += 1
                out.append("")
                out.append("largest kernel names (ms over the window, launches):")
                for nm, t in names.most_common(args.top):
                    out.append(f"  {t / 1e6:9.2f} ms  n={cnt[nm]:7d}  {nm}")
    text = "\n\n".join(out)
    print(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
