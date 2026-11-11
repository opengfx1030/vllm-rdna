# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Emit the serving report for the 16k-input / 1k-output load.

Prints per-request and aggregated prefill and decode, TTFT and ITL for the
c=1 and c=8 cells, side by side for the FP16 reference and the INT4 candidate.

Usage: python report_16k.py <fp16-dir> <int4-dir> [-o out.md]
"""

import argparse
import json
from pathlib import Path

CELLS = (("bench-16384x1024-c1", "c=1"), ("bench-16384x1024-c8", "c=8"))
ROWS = (
    ("mean_request_prefill_tokens_per_s", "prefill per-request (tok/s)"),
    ("whole_run_input_tokens_per_s", "prefill aggregated (tok/s)"),
    ("wave_prefill_proxy_tokens_per_s", "prefill aggregated proxy (tok/s)"),
    ("mean_request_decode_tokens_per_s", "decode per-request (tok/s)"),
    ("aggregate_output_tokens_per_s", "decode aggregated (tok/s)"),
    ("mean_ttft_s", "TTFT (s)"),
    ("itl_chunk_median_ms", "ITL median (ms)"),
    ("itl_chunk_mean_ms", "ITL mean (ms)"),
    ("acceptance_length", "MTP acceptance length"),
)


def metrics(arm: Path):
    return json.loads((arm / "metrics.json").read_text())


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    ref, cand = metrics(args.reference), metrics(args.candidate)
    lines = [
        "# 16k input / 1k output - serving report",
        "",
        f"- FP16 reference: `{args.reference}`",
        f"- INT4 candidate: `{args.candidate}`",
        "",
    ]
    for cell, label in CELLS:
        lines += [
            f"## {label} ({cell})",
            "",
            "| metric | FP16 | INT4 | ratio | delta |",
            "| --- | --- | --- | --- | --- |",
        ]
        for key, name in ROWS:
            a, b = ref[cell].get(key), cand[cell].get(key)
            ratio = f"{b / a:.4f}" if isinstance(a, (int, float)) and a else "n/a"
            delta = (
                f"{(b - a) / a * 100:+.2f}%"
                if isinstance(a, (int, float)) and a
                else "n/a"
            )
            lines.append(f"| {name} | {fmt(a)} | {fmt(b)} | {ratio} | {delta} |")
        lines.append("")
    # MD047: the file ends with a single newline, so drop the blank separator
    # the per-cell grouping leaves behind.
    text = "\n".join(lines).rstrip("\n")
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
