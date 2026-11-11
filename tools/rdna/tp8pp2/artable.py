#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Markdown table from benchmark_rdna_allreduce.py logs.

  artable.py LABEL=PATH [LABEL=PATH ...]     (run with the venv python: regex)

One block per log: rows are token counts, columns backend/mode in microseconds per
call.
"""

import sys
from pathlib import Path

import regex as re

LINE = re.compile(
    r"^(pynccl|rdna_ar|auto|rs\+ag)\s+(eager|graph)\s+(\d+)x(\d+)\s+([\d.]+) KiB\s+"
    r"([\d.]+) us\s+busbw\s+([\d.]+) GB/s\s+ok=(\w+)"
)


def parse(path):
    rows = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        be, mode, tok, hid, kib, us, bw, ok = m.groups()
        rows.setdefault((int(tok), int(hid), float(kib)), {})[(be, mode)] = (
            float(us),
            float(bw),
            ok == "True",
        )
    return rows


def main():
    for spec in sys.argv[1:]:
        label, _, path = spec.partition("=")
        rows = parse(path)
        cols = [
            ("pynccl", "eager"),
            ("pynccl", "graph"),
            ("rdna_ar", "eager"),
            ("rdna_ar", "graph"),
            ("auto", "graph"),
        ]
        print(f"\n### {label}\n")
        print(
            "| tokens x hidden | KiB | RCCL eager | RCCL graph | one-shot eager |"
            " one-shot graph | auto graph | busbw RCCL graph (GB/s) | all ok |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for (tok, hid, kib), d in sorted(rows.items()):
            cells = [f"{d[c][0]:.1f}" if c in d else "-" for c in cols]
            bw = d.get(("pynccl", "graph"), (0, 0, True))[1]
            ok = all(v[2] for v in d.values())
            print(
                f"| {tok} x {hid} | {kib:.0f} | "
                + " | ".join(cells)
                + f" | {bw:.1f} | {'yes' if ok else 'NO'} |"
            )


if __name__ == "__main__":
    main()
