#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate a rocprofv3 kernel_trace.csv into a kernel-share table.

usage: step0_parse.py <prof_kernel_trace.csv> [--json out.json]

Share = kernel busy time / total kernel dispatch time in the file. For a
clean-exit `bench throughput` run the file covers load + compile + the
measured prefill/decode, so treat the table as a ranking of where GPU time
goes, not a per-phase breakdown.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict

GROUPS = [
    ("mtp_stats", re.compile(r"compute_local_logits_stats", re.I)),
    ("mtp_reject", re.compile(r"rejection_kernel", re.I)),
    ("mtp_resample", re.compile(r"resample_kernel", re.I)),
    ("qsa", re.compile(r"qsa|sparse_paged_gqa|expand_qsa|compress_qsa", re.I)),
    ("causal_conv1d_triton", re.compile(r"causal_conv1d_fwd_kernel|causal_conv1d_update_kernel", re.I)),
    ("causal_conv1d_hip", re.compile(r"causal_conv1d_(fwd|update)_rdna2", re.I)),
    ("fa_rdna2", re.compile(r"fa_rdna2", re.I)),
    ("triton", re.compile(r"triton_", re.I)),
    ("rocblas", re.compile(r"rocblas|Cijk_|gemm|Gemm|GEMM", re.I)),
    ("dot2_w4a16", re.compile(r"gptq_gemm_rdna2|moe_gptq_gemm_rdna2|w4a8|dot2", re.I)),
]


def group_of(name: str) -> str:
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def main() -> int:
    path = sys.argv[1]
    rows = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if r.get("Kind") != "KERNEL_DISPATCH":
                continue
            try:
                dur = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            except (KeyError, ValueError):
                continue
            rows.append((r.get("Kernel_Name", ""), dur))

    per_kernel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    for name, dur in rows:
        if dur < 0:
            continue
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1
        per_group[group_of(name)][0] += dur
        per_group[group_of(name)][1] += 1
        total += dur

    def fmt(d: int) -> str:
        return f"{d/1e6:,.1f}"

    print(f"# {path}")
    print(f"# dispatches={len(rows)} total_kernel_busy_ms={fmt(total)}")
    print()
    print("## by group")
    print(f"{'group':26} {'busy_ms':>14} {'share':>7} {'dispatches':>11}")
    for g, (d, n) in sorted(per_group.items(), key=lambda kv: -kv[1][0]):
        print(f"{g:26} {fmt(d):>14} {100*d/total:6.1f}% {n:>11}")
    print()
    print("## top kernels")
    print(f"{'busy_ms':>14} {'share':>7} {'n':>9}  kernel")
    for name, (d, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:40]:
        short = name if len(name) <= 110 else name[:107] + "..."
        print(f"{fmt(d):>14} {100*d/total:6.1f}% {n:>9}  {short}")

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w") as fh:
            json.dump(
                {
                    "total_ms": total / 1e6,
                    "dispatches": len(rows),
                    "groups": {g: {"ms": d / 1e6, "n": n} for g, (d, n) in per_group.items()},
                    "kernels": {k: {"ms": d / 1e6, "n": n} for k, (d, n) in per_kernel.items()},
                },
                fh,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
