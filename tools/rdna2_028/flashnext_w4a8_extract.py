#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flatten the bench-serve JSON result files of a Flash-Next W4A8 arm to CSV.

Usage: flashnext_w4a8_extract.py <arm_dir>

Finds <arm_dir>/cells/c<in>_<n>/openai-*.json (plus warmup dirs), picks the
newest JSON per dir, and prints one CSV row per measured cell. All metrics are
read defensively: missing keys become empty fields, never a crash.
"""
import csv
import glob
import json
import os
import re
import sys


def pick_json(d):
    cands = sorted(glob.glob(os.path.join(d, "*.json")), key=os.path.getmtime)
    return cands[-1] if cands else None


def dims_from_log(d):
    """The bench JSON omits the random lengths; the sibling .log args carry them."""
    p = d + ".log"
    try:
        with open(p) as f:
            head = f.read()
    except OSError:
        return "", ""
    m = re.search(r"random_input_len=(\d+),\s*random_output_len=(\d+)", head)
    return (m.group(1), m.group(2)) if m else ("", "")


def num(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return ""


def main() -> None:
    arm = sys.argv[1]
    arm_name = os.path.basename(arm)
    w = csv.writer(sys.stdout)
    w.writerow([
        "arm", "dir", "n", "in_len", "out_len", "ok", "dur_s", "in_tok", "out_tok",
        "ttft_mean_ms", "ttft_med_ms", "ttft_p99_ms",
        "tpot_mean_ms", "itl_med_ms", "itl_p99_ms",
        "out_tok_s_agg", "total_tok_s", "decode_tok_s_per_req", "prefill_tok_s",
        "accept_len", "accept_rate",
    ])
    dirs = sorted(glob.glob(os.path.join(arm, "cells", "c*_*")))
    dirs += sorted(glob.glob(os.path.join(arm, "warmup_*")))
    for d in dirs:
        p = pick_json(d)
        if not p:
            continue
        with open(p) as f:
            r = json.load(f)
        n = num(r, "num_prompts", "completed") or ""
        inlen = num(r, "random_input_len", "max_input_len") or ""
        outlen = num(r, "random_output_len", "max_output_len") or ""
        if not inlen or not outlen:
            ji, jo = dims_from_log(d)
            inlen = inlen or ji
            outlen = outlen or jo
        in_tok = num(r, "total_input_tokens")
        out_tok = num(r, "total_output_tokens")
        ttft = num(r, "mean_ttft_ms")
        tpot = num(r, "mean_tpot_ms")
        decode = round(1000.0 / tpot, 2) if isinstance(tpot, (int, float)) and tpot else ""
        prefill = ""
        if isinstance(ttft, (int, float)) and ttft and isinstance(in_tok, (int, float)):
            prefill = round(in_tok / (ttft / 1000.0), 1)
        w.writerow([
            arm_name, os.path.basename(d), n, inlen, outlen,
            num(r, "completed", "successful_requests"), num(r, "duration"),
            in_tok, out_tok,
            ttft, num(r, "median_ttft_ms"), num(r, "p99_ttft_ms"),
            tpot, num(r, "median_itl_ms"), num(r, "p99_itl_ms"),
            num(r, "output_throughput", "output_token_throughput"),
            num(r, "total_token_throughput"),
            decode, prefill,
            num(r, "acceptance_length"), num(r, "acceptance_rate"),
        ])


if __name__ == "__main__":
    main()
