# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare a completed FP16 reference arm with a completed INT4 candidate arm.

Both arms must be same-boot, same-config and same-manifest; this tool refuses
to compare arms that are not. Output is a JSON artifact plus a markdown table.

Usage: python compare_arms.py <fp16-dir> <int4-dir> [-o out.json]
"""

import argparse
import json
from pathlib import Path

CELLS = (
    "bench-1024x512-c1",
    "bench-1024x512-c8",
    "bench-16384x1024-c1",
    "bench-16384x1024-c8",
)
KEYS = (
    ("aggregate_output_tokens_per_s", "aggregate_output_tok/s"),
    ("mean_request_decode_tokens_per_s", "mean_request_decode_tok/s"),
    ("mean_request_prefill_tokens_per_s", "mean_request_prefill_tok/s"),
    ("whole_run_input_tokens_per_s", "whole_run_input_tok/s"),
    ("wave_prefill_proxy_tokens_per_s", "wave_prefill_proxy_tok/s"),
    ("mean_ttft_s", "mean_ttft_s"),
    ("itl_chunk_median_ms", "itl_chunk_median_ms"),
    ("itl_chunk_mean_ms", "itl_chunk_mean_ms"),
)


def load(arm: Path):
    metrics = json.loads((arm / "metrics.json").read_text())
    run = json.loads((arm / "run.json").read_text())
    status = json.loads((arm / "status.json").read_text())
    qualification_file = arm / "qualification.json"
    if qualification_file.exists():
        qualification = json.loads(qualification_file.read_text())
    else:
        # Arms run before the classification artifact existed: derive it from the
        # preserved kernel delta so legacy arms stay comparable.
        from kernel_events import classify

        delta_file = arm / "kernel.delta.log"
        events = (
            classify(delta_file.read_text(errors="replace"))
            if delta_file.exists()
            else None
        )
        mode = (
            "strict-pass"
            if status.get("status") == "PASS"
            else "diagnostic-only"
            if events and events["verdict"] == "corrected-only"
            else "fail"
        )
        qualification = {
            "qualification_mode": mode,
            "raw_status": status.get("status"),
            "raw_cleanup": status.get("cleanup"),
            "hardware_attribution": (
                "clean"
                if mode == "strict-pass"
                else "contaminated-corrected-link-noise"
                if mode == "diagnostic-only"
                else "unqualified"
            ),
            "kernel_events": {
                "verdict": events["verdict"] if events else "incomplete",
                "corrected_event_groups": events["corrected_event_groups"]
                if events
                else 0,
                "corrected_error_count": events["corrected_error_count"]
                if events
                else 0,
                "fatal_event_groups": events["fatal_event_groups"] if events else 0,
                "fatal_events": events["fatal_events"] if events else [],
                "endpoints_with_nonfatal_err": events["endpoints_with_nonfatal_err"]
                if events
                else 0,
                "endpoints_with_bad_tlp": events["endpoints_with_bad_tlp"]
                if events
                else 0,
                "aer_recovery_buffer_overflows": events["aer_recovery_buffer_overflows"]
                if events
                else [],
            },
        }
    return {
        "metrics": metrics,
        "run": run,
        "status": status,
        "qualification": qualification,
    }


def check_pairable(reference, candidate):
    problems = []
    if reference["run"]["boot_id"] != candidate["run"]["boot_id"]:
        problems.append("boot differs")
    for key in ("recipe", "kv_cache_layout", "model_config_sha256", "gpus"):
        if reference["run"].get(key) != candidate["run"].get(key):
            problems.append(f"{key} differs")
    ref_cmd = dict(x.split("=", 1) for x in reference["run"]["command"] if "=" in x)
    cand_cmd = dict(x.split("=", 1) for x in candidate["run"]["command"] if "=" in x)
    for key in (
        "RECIPE",
        "MODEL",
        "TP",
        "MTP",
        "SEQS",
        "MAXLEN",
        "KV",
        "BLOCK_SIZE",
        "GPU_MAX_HW_QUEUES",
        "OMP_NUM_THREADS",
        "TUNABLEOP",
        "COMPILE_MODE",
        "FEATURES",
        "PLE",
    ):
        if ref_cmd.get(key) != cand_cmd.get(key):
            problems.append(f"config {key} differs")
    # MAXBAT is deliberately not compared: the prefill budget is derived from
    # the attention block size, which the hybrid page alignment picks per KV
    # dtype (1024 for FP16 KV, 2048 for a packed KV cache with an FP16 SSM
    # state). A packed candidate and its FP16 reference therefore carry
    # different budgets for the same requested block size, and the invariant
    # that matters is per arm: the budget must cover two mamba state blocks.
    for key in ("status", "cleanup"):
        if reference["status"].get(key) != "PASS":
            problems.append(f"reference raw {key} is {reference['status'].get(key)}")
        if candidate["status"].get(key) != "PASS":
            problems.append(f"candidate raw {key} is {candidate['status'].get(key)}")
    return problems


def build(reference, candidate):
    rows = []
    for cell in CELLS:
        ref = reference["metrics"][cell]
        cand = candidate["metrics"][cell]
        row = {"cell": cell}
        for key, label in KEYS:
            ref_value, cand_value = ref[key], cand[key]
            row[label] = {
                "fp16": ref_value,
                "int4": cand_value,
                "ratio": (cand_value / ref_value) if ref_value else None,
                "delta_pct": ((cand_value - ref_value) / ref_value * 100.0)
                if ref_value
                else None,
            }
        rows.append(row)
    paired_ppl = None
    ppl_file = candidate.get("ppl-paired")
    if ppl_file and Path(ppl_file).exists():
        paired_ppl = json.loads(Path(ppl_file).read_text())
    return {
        "reference": {
            "dir": str(reference["path"]),
            "boot_id": reference["run"]["boot_id"],
            "raw_status": reference["status"],
            "qualification_mode": reference["qualification"]["qualification_mode"],
            "hardware_attribution": reference["qualification"]["hardware_attribution"],
            "kernel_events": reference["qualification"]["kernel_events"],
        },
        "candidate": {
            "dir": str(candidate["path"]),
            "boot_id": candidate["run"]["boot_id"],
            "raw_status": candidate["status"],
            "qualification_mode": candidate["qualification"]["qualification_mode"],
            "hardware_attribution": candidate["qualification"]["hardware_attribution"],
            "kernel_events": candidate["qualification"]["kernel_events"],
        },
        "same_boot": reference["run"]["boot_id"] == candidate["run"]["boot_id"],
        "paired_ppl": paired_ppl,
        "cells": rows,
    }


def markdown(report):
    lines = [
        "| cell | metric | FP16 | INT4 | ratio | delta % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["cells"]:
        for _, label in KEYS:
            entry = row[label]
            ratio = f"{entry['ratio']:.4f}" if entry["ratio"] is not None else "n/a"
            delta = (
                f"{entry['delta_pct']:+.2f}%"
                if entry["delta_pct"] is not None
                else "n/a"
            )
            lines.append(
                f"| {row['cell']} | {label} | {entry['fp16']:.3f} "
                f"| {entry['int4']:.3f} | {ratio} | {delta} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    reference["path"] = args.reference
    candidate["path"] = args.candidate
    candidate["ppl-paired"] = args.candidate / "ppl-paired.json"

    problems = check_pairable(reference, candidate)
    report = build(reference, candidate)
    report["pairing_problems"] = problems
    report["pairable"] = not problems
    if paired := report.get("paired_ppl"):
        report["model_quality"] = {
            "ppl_ratio": paired["ratio"],
            "max_ratio": paired["max_ratio"],
            "within_gate": paired["ratio"] <= paired["max_ratio"],
        }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print()
    print(markdown(report))
    if problems:
        print()
        print("PAIRING PROBLEMS:")
        for problem in problems:
            print("  -", problem)


if __name__ == "__main__":
    main()
