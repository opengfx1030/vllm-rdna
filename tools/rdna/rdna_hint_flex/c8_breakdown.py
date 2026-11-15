#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step GPU time by category for prefill, mixed and decode steps (rocprofv3).

    python3 tools/rdna/rdna_hint_flex/c8_breakdown.py TRACE_DIR \
        [--ranks-json ranks.json] [--out-prefix OUT/c8] [--marker REGEX]

TRACE_DIR holds the ``<pid>_kernel_trace.csv`` files that ``c8_profile.py``
records (one per TP rank, windows separated by idle gaps of ``--window-gap-s``).
Every window is cut into model steps at a kernel that runs once per step (the
lm_head GEMV, ``--marker``); a step is typed by its attention kernels: prefill
kernels only = ``prefill``, prefill and decode kernels = ``mixed``, decode
kernels only = ``decode``. Per rank and step type the output has, per kernel
family and per roll-up category: launches, kernel time, and the GPU idle time in
front of the launches (a gap is charged to the kernel that waited), as ms per
step and as a share of the step wall (marker start to next marker start).

Files: ``PREFIX.md`` (tables), ``PREFIX_categories.csv``, ``PREFIX_steps.csv``
(one row per step: wall, busy, idle, ms per roll-up), ``PREFIX.json``.
Pure Python, no dependencies.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import regex as re

# (family, roll-up category, name regex): first match wins, demangled kernel names.
FAMILIES: list[tuple[str, str, str]] = [
    ("hi2 dequant", "HI2 GEMM", r"dequant_kernel"),
    ("dense GEMM (rocBLAS)", "HI2 GEMM", r"^Cijk_|rocblas|^Custom_Cijk"),
    ("dense GEMM (HIP prefill op)", "HI2 GEMM", r"(?:^|[\s:])gemm2?_kernel"),
    ("hi2 gemv", "HI2 GEMM", r"(?:^|[\s:])gemv2?(?:_i8)?_kernel"),
    ("hi2 act transform", "HI2 GEMM", r"act_(?:transform|quant)(?:64)?_kernel"),
    ("lm_head gemv", "embedding / lm_head / sampling", r"gemv_f16_rdna2|gemv_act_k"),
    ("attention prefill", "full attention", r"fa_prefill"),
    ("attention decode", "full attention", r"fa_decode|paged_attn|splitk"),
    ("kv cache write", "full attention", r"reshape_and_cache"),
    (
        "GDN prefill kernels",
        "GDN",
        r"gdn_prefill|chunk_gated_delta|chunk_scaled_dot|solve_tril|recompute_w_u|"
        r"chunk_fwd|chunk_local_cumsum|l2norm_fwd",
    ),
    ("GDN conv1d", "GDN", r"causal_conv1d"),
    (
        "GDN decode / recurrence",
        "GDN",
        r"gdn_decode|fused_recurrent|fused_sigmoid_gating|gdn_leapquant|mono_gdn|"
        r"gdn_spec",
    ),
    ("GDN gated rmsnorm", "GDN", r"gated_rms_norm|rmsnorm_gated|layer_norm_fwd"),
    ("all-reduce (rdna_ar)", "all-reduce", r"rdna_ar|rdna_ag"),
    ("all-reduce (RCCL)", "all-reduce", r"nccl|rccl"),
    ("rmsnorm (add + norm)", "norms / residual / elementwise", r"rms_norm|layernorm"),
    (
        "sampling / logits",
        "embedding / lm_head / sampling",
        r"gumbel|argmax|topk|top_k|topp|top_p|sort|softmax|rdna2_sample|get_top|"
        r"multinomial|sampler|scatter_num_accepted",
    ),
    (
        "embedding",
        "embedding / lm_head / sampling",
        r"embedding|indexSelectLargeIndex|index_select|IndexSelect",
    ),
    ("copyBuffer / memset", "copies", r"__amd_rocclr|[Mm]emcpy|[Mm]emset"),
    (
        "runner glue (input prep, mamba align)",
        "norms / residual / elementwise",
        r"rdna2_|prepare_|expand_idx|batch_memcpy|mamba|post_update|compute_slot|"
        r"gather_block",
    ),
    (
        "index / gather / scatter",
        "copies",
        r"index_elementwise|index_copy|gather|scatter|indexSelect",
    ),
    (
        "triton / inductor pointwise",
        "norms / residual / elementwise",
        r"^triton_|_kernel_[0-9]+d?$",
    ),
    (
        "elementwise / reduce (torch)",
        "norms / residual / elementwise",
        r"elementwise|vectorized|unrolled|reduce_kernel|Functor|CatArray|fill_|"
        r"index_put|masked|where|clamp|silu|act_and_mul|rotary|rope",
    ),
]
_FAM = [(n, r, re.compile(rx)) for n, r, rx in FAMILIES]
ROLLUPS = [
    "HI2 GEMM",
    "full attention",
    "GDN",
    "all-reduce",
    "norms / residual / elementwise",
    "embedding / lm_head / sampling",
    "copies",
    "other",
]
IDLE = "GPU idle gaps"


def classify(name: str, cache: dict[str, tuple[str, str]]) -> tuple[str, str]:
    hit = cache.get(name)
    if hit is None:
        hit = ("other", "other")
        for fam, roll, rx in _FAM:
            if rx.search(name):
                hit = (fam, roll)
                break
        cache[name] = hit
    return hit


class K:
    __slots__ = ("s", "e", "name", "fam", "roll")


def read_trace(path: str) -> list[K]:
    rows: list[K] = []
    cache: dict[str, tuple[str, str]] = {}
    with open(path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd, None)
        if not head:
            return rows
        ix = {h: i for i, h in enumerate(head)}
        i_s, i_e, i_n = ix["Start_Timestamp"], ix["End_Timestamp"], ix["Kernel_Name"]
        for r in rd:
            k = K()
            k.s, k.e, k.name = int(r[i_s]), int(r[i_e]), r[i_n]
            k.fam, k.roll = classify(k.name, cache)
            rows.append(k)
    rows.sort(key=lambda x: (x.s, x.e))
    return rows


def split_windows(rows: list[K], gap_ns: int) -> list[list[K]]:
    out: list[list[K]] = []
    cur: list[K] = []
    last = None
    for k in rows:
        if last is not None and k.s - last >= gap_ns:
            out.append(cur)
            cur = []
        cur.append(k)
        last = k.e if last is None else max(last, k.e)
    if cur:
        out.append(cur)
    return out


def cut_steps(win: list[K], marker: re.Pattern) -> list[list[K]]:
    idx = [i for i, k in enumerate(win) if marker.search(k.name)]
    return [win[a:b] for a, b in zip(idx, idx[1:])]


def step_type(st: list[K]) -> str:
    pf = any(k.fam == "attention prefill" for k in st)
    dc = any(k.fam == "attention decode" for k in st)
    return "mixed" if pf and dc else "prefill" if pf else "decode" if dc else "other"


def measure(st: list[K], wall_ns: int) -> dict:
    """Kernel time, launches and charged idle per family / roll-up of one step."""
    fam_t: dict[str, int] = defaultdict(int)
    fam_n: dict[str, int] = defaultdict(int)
    fam_g: dict[str, int] = defaultdict(int)
    roll_t: dict[str, int] = defaultdict(int)
    roll_g: dict[str, int] = defaultdict(int)
    end = st[0].s
    busy = 0
    cs = ce = None
    for k in st:
        gap = max(k.s - end, 0)
        end = max(end, k.e)
        d = k.e - k.s
        fam_t[k.fam] += d
        fam_n[k.fam] += 1
        fam_g[k.fam] += gap
        roll_t[k.roll] += d
        roll_g[k.roll] += gap
        if cs is None:
            cs, ce = k.s, k.e
        elif k.s <= ce:
            ce = max(ce, k.e)
        else:
            busy += ce - cs
            cs, ce = k.s, k.e
    if cs is not None:
        busy += ce - cs
    return {
        "wall_ns": wall_ns,
        "busy_ns": busy,
        "idle_ns": max(wall_ns - busy, 0),
        "n": len(st),
        "fam_t": fam_t,
        "fam_n": fam_n,
        "fam_g": fam_g,
        "roll_t": roll_t,
        "roll_g": roll_g,
    }


def mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def analyse_rank(path: str, marker: re.Pattern, gap_s: float) -> list[dict]:
    rows = read_trace(path)
    steps: list[dict] = []
    for w, win in enumerate(split_windows(rows, int(gap_s * 1e9))):
        cut = cut_steps(win, marker)
        idx = [i for i, k in enumerate(win) if marker.search(k.name)]
        for j, st in enumerate(cut):
            wall = win[idx[j + 1]].s - win[idx[j]].s
            m = measure(st, wall)
            m["window"] = w
            m["step"] = j
            m["type"] = step_type(st)
            m["t0_ns"] = st[0].s
            m["ar"] = [(k.s, k.e) for k in st if k.roll == "all-reduce"]
            steps.append(m)
    return steps


def aggregate(steps: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        by[s["type"]].append(s)
    out = {}
    for typ, ss in by.items():
        walls = [s["wall_ns"] / 1e6 for s in ss]
        fams = sorted({f for s in ss for f in s["fam_t"]})
        out[typ] = {
            "steps": len(ss),
            "wall_ms": mean(walls),
            "wall_ms_median": statistics.median(walls),
            "busy_ms": mean([s["busy_ns"] / 1e6 for s in ss]),
            "idle_ms": mean([s["idle_ns"] / 1e6 for s in ss]),
            "kernels": mean([s["n"] for s in ss]),
            "roll": {
                r: mean([s["roll_t"].get(r, 0) / 1e6 for s in ss]) for r in ROLLUPS
            },
            "roll_gap": {
                r: mean([s["roll_g"].get(r, 0) / 1e6 for s in ss]) for r in ROLLUPS
            },
            "fam": {
                f: {
                    "launches": mean([s["fam_n"].get(f, 0) for s in ss]),
                    "ms": mean([s["fam_t"].get(f, 0) / 1e6 for s in ss]),
                    "gap_ms": mean([s["fam_g"].get(f, 0) / 1e6 for s in ss]),
                    "roll": next((r for n, r, _ in _FAM if n == f), "other"),
                }
                for f in fams
            },
        }
    return out


def ar_pairs(steps0: list[dict], steps1: list[dict]) -> dict[str, dict]:
    """Per step type: all-reduce data time and rank skew from paired launches."""
    by: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for a, b in zip(steps0, steps1):
        if a["type"] != b["type"] or len(a["ar"]) != len(b["ar"]) or not a["ar"]:
            continue
        data = skew = busy = 0.0
        for (s0, e0), (s1, e1) in zip(a["ar"], b["ar"]):
            data += min(e0 - s0, e1 - s1)
            skew += abs(s0 - s1)
            busy += max(e0 - s0, e1 - s1)
        by[a["type"]].append((data / 1e6, skew / 1e6, busy / 1e6))
    return {
        t: {
            "pairs": len(v),
            "data_ms": mean([x[0] for x in v]),
            "skew_ms": mean([x[1] for x in v]),
            "longer_rank_ms": mean([x[2] for x in v]),
        }
        for t, v in by.items()
    }


def md_table(rank: int, typ: str, a: dict) -> list[str]:
    wall = a["wall_ms"]
    lines = [
        f"#### rank {rank}, {typ} steps: {a['steps']} steps, {a['kernels']:.0f} "
        f"kernels per step, wall {wall:.1f} ms (median {a['wall_ms_median']:.1f}), "
        f"GPU busy {a['busy_ms']:.1f} ms, idle {a['idle_ms']:.1f} ms "
        f"({100 * a['idle_ms'] / wall:.2f}%)",
        "",
        "| roll-up category | ms / step | % of step wall | idle charged (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in sorted(ROLLUPS, key=lambda r: -a["roll"][r]):
        if a["roll"][r] <= 0:
            continue
        lines.append(
            f"| {r} | {a['roll'][r]:.2f} | {100 * a['roll'][r] / wall:.1f}% | "
            f"{a['roll_gap'][r]:.2f} |"
        )
    lines.append(
        f"| {IDLE} | {a['idle_ms']:.2f} | {100 * a['idle_ms'] / wall:.1f}% | |"
    )
    lines += [
        "",
        "| family | roll-up | launches / step | ms / step | % of step wall | "
        "idle before (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for f, v in sorted(a["fam"].items(), key=lambda kv: -kv[1]["ms"]):
        lines.append(
            f"| {f} | {v['roll']} | {v['launches']:.0f} | {v['ms']:.2f} | "
            f"{100 * v['ms'] / wall:.1f}% | {v['gap_ms']:.2f} |"
        )
    return lines + [""]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace_dir")
    ap.add_argument("--ranks-json", default="", help='{"0": pid, "1": pid}')
    ap.add_argument("--marker", default=r"gemv_f16_rdna2")
    ap.add_argument("--window-gap-s", type=float, default=2.0)
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(a.trace_dir, "**", "*kernel_trace.csv"), recursive=True)
    )
    files = [f for f in files if os.path.getsize(f) > 200]
    pids = {}
    if a.ranks_json:
        with open(a.ranks_json) as fh:
            pids = {int(v): int(k) for k, v in json.load(fh).items()}
    marker = re.compile(a.marker)
    ranks: dict[int, list[dict]] = {}
    for n, f in enumerate(files):
        pid = int(os.path.basename(f).split("_")[0])
        rank = pids.get(pid, n)
        ranks[rank] = analyse_rank(f, marker, a.window_gap_s)
        print(
            f"rank {rank}: {os.path.basename(f)} {len(ranks[rank])} steps", flush=True
        )

    aggs = {r: aggregate(s) for r, s in ranks.items()}
    md = [f"### {a.title or 'c8 step breakdown'}", ""]
    for r in sorted(aggs):
        for typ in ("prefill", "mixed", "decode", "other"):
            if typ in aggs[r]:
                md += md_table(r, typ, aggs[r][typ])
    pairs = {}
    if 0 in ranks and 1 in ranks:
        pairs = ar_pairs(ranks[0], ranks[1])
        md += [
            "#### all-reduce, rank 0 / rank 1 launches paired by order",
            "",
            "| step type | steps paired | data time (min of the two durations, ms) | "
            "start skew (sum of |start0 - start1|, ms) | longer duration (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for t, v in pairs.items():
            md.append(
                f"| {t} | {v['pairs']} | {v['data_ms']:.2f} | {v['skew_ms']:.2f} | "
                f"{v['longer_rank_ms']:.2f} |"
            )
        md.append("")
    text = "\n".join(md)
    print(text)
    if a.out_prefix:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
        with open(a.out_prefix + ".md", "w") as fh:
            fh.write(text + "\n")
        with open(a.out_prefix + "_categories.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "rank", "step_type", "steps", "level", "name", "rollup",
                    "launches_per_step", "ms_per_step", "pct_of_step_wall",
                    "idle_before_ms",
                ]
            )  # fmt: skip
            for r in sorted(aggs):
                for typ, v in aggs[r].items():
                    wall = v["wall_ms"]
                    for ro in ROLLUPS:
                        if v["roll"][ro] > 0:
                            w.writerow(
                                [r, typ, v["steps"], "rollup", ro, ro, "",
                                 f"{v['roll'][ro]:.4f}",
                                 f"{100 * v['roll'][ro] / wall:.3f}",
                                 f"{v['roll_gap'][ro]:.4f}"]
                            )  # fmt: skip
                    w.writerow(
                        [r, typ, v["steps"], "rollup", IDLE, "", "",
                         f"{v['idle_ms']:.4f}", f"{100 * v['idle_ms'] / wall:.3f}", ""]
                    )  # fmt: skip
                    w.writerow(
                        [r, typ, v["steps"], "step", "wall", "", "",
                         f"{wall:.4f}", "100", ""]
                    )  # fmt: skip
                    for f, fv in v["fam"].items():
                        w.writerow(
                            [r, typ, v["steps"], "family", f, fv["roll"],
                             f"{fv['launches']:.1f}", f"{fv['ms']:.4f}",
                             f"{100 * fv['ms'] / wall:.3f}", f"{fv['gap_ms']:.4f}"]
                        )  # fmt: skip
        with open(a.out_prefix + "_steps.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["rank", "window", "step", "type", "t0_s", "wall_ms", "busy_ms",
                 "idle_ms", "kernels", *[r for r in ROLLUPS]]
            )  # fmt: skip
            for r in sorted(ranks):
                base = min((s["t0_ns"] for s in ranks[r]), default=0)
                for s in ranks[r]:
                    w.writerow(
                        [r, s["window"], s["step"], s["type"],
                         f"{(s['t0_ns'] - base) / 1e9:.3f}",
                         f"{s['wall_ns'] / 1e6:.3f}", f"{s['busy_ns'] / 1e6:.3f}",
                         f"{s['idle_ns'] / 1e6:.3f}", s["n"],
                         *[f"{s['roll_t'].get(ro, 0) / 1e6:.3f}" for ro in ROLLUPS]]
                    )  # fmt: skip
        with open(a.out_prefix + ".json", "w") as fh:
            json.dump({"ranks": aggs, "all_reduce_pairs": pairs}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
