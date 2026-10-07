#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate vLLM torch-profiler traces into a kernel-share table.

usage: step0_tp_parse.py <dir-with-*.pt.trace.json> [--json out.json]

Each TP worker writes its own trace; the table sums device kernel time across
all workers (a kernel's absolute time is wall-clock across concurrent ranks, so
this is per-rank-busy summed, i.e. a ranking of where GPU time goes).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

GROUPS = [
    ("mtp_stats", re.compile(r"compute_local_logits_stats", re.I)),
    ("mtp_reject", re.compile(r"rejection_kernel", re.I)),
    ("mtp_resample", re.compile(r"resample_kernel", re.I)),
    ("qsa", re.compile(r"qsa|sparse_paged_gqa|expand_qsa|compress_qsa", re.I)),
    ("causal_conv1d_triton", re.compile(r"causal_conv1d_(fwd|update)_kernel", re.I)),
    ("causal_conv1d_hip", re.compile(r"causal_conv1d_(fwd|update)_rdna2", re.I)),
    ("fa_rdna2", re.compile(r"fa_rdna2", re.I)),
    ("paged_mqa_indexer", re.compile(r"paged_mqa_logits", re.I)),
    ("rope", re.compile(r"rope|rotary", re.I)),
    ("triton", re.compile(r"triton_", re.I)),
    ("rocblas", re.compile(r"rocblas|Cijk_|gemm_|Gemm|GEMM", re.I)),
    ("w4a16_dot2", re.compile(r"gptq_gemm_rdna2|moe_gptq_gemm_rdna2|w4a8|dot2", re.I)),
    ("allreduce", re.compile(r"all_reduce|allreduce|rdna_ar|nccl", re.I)),
]


def group_of(name: str) -> str:
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def main() -> int:
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
    per_kernel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    nfiles = 0
    for f in files:
        try:
            obj = json.load(open(f))
        except Exception:
            continue
        evs = obj.get("traceEvents", obj if isinstance(obj, list) else [])
        if not evs:
            continue
        nfiles += 1
        for e in evs:
            if not isinstance(e, dict):
                continue
            cat = str(e.get("cat", ""))
            if cat.lower() not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            name = e.get("name", "") or ""
            dur = e.get("dur")
            if not isinstance(dur, (int, float)) or dur < 0:
                continue
            per_kernel[name][0] += int(dur)
            per_kernel[name][1] += 1
            per_group[group_of(name)][0] += int(dur)
            per_group[group_of(name)][1] += 1
            total += int(dur)

    def fmt(d_):
        return f"{d_/1000:,.1f}"

    print(f"# traces={nfiles} files_matched={len(files)} total_kernel_busy_ms={fmt(total)}")
    print()
    print("## by group")
    print(f"{'group':26} {'busy_ms':>13} {'share':>7} {'kernels':>10}")
    for g, (dd, n) in sorted(per_group.items(), key=lambda kv: -kv[1][0]):
        print(f"{g:26} {fmt(dd):>13} {100*dd/total:6.1f}% {n:>10}")
    print()
    print("## top kernels")
    print(f"{'busy_ms':>13} {'share':>7} {'n':>9}  kernel")
    for name, (dd, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:45]:
        short = name if len(name) <= 110 else name[:107] + "..."
        print(f"{fmt(dd):>13} {100*dd/total:6.1f}% {n:>9}  {short}")

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump(
            {
                "total_ms": total / 1000,
                "groups": {g: {"ms": dd / 1000, "n": n} for g, (dd, n) in per_group.items()},
                "kernels": {k: {"ms": dd / 1000, "n": n} for k, (dd, n) in per_kernel.items()},
            },
            open(out, "w"),
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
