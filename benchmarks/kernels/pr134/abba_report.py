# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the counterbalanced ABBA artifact from the four serving arms.

The campaign runs FP16 -> INT4 -> INT4 -> FP16 so that run-order drift is
controlled: each configuration is measured twice, once in each half of the
sequence. This tool turns the four arms into one artifact that states the
per-arm gates, proves the arms are comparable, and gives the per-cell INT4
ratio for both repeats plus the pooled delta.

All four arms must share one boot, one configuration, one source/binary
manifest and a raw PASS. A corrected-link PCIe event in any of them is
reported, never silently dropped.

Usage:
    python abba_report.py <fp16-A> <int4-A> <int4-B> <fp16-B> -o out.json
"""

import argparse
import json
from pathlib import Path

from compare_arms import CELLS, load

# The metrics the ABBA artifact carries. wave_prefill_proxy is left out on
# purpose: it is a proxy, and whole_run_input covers the same question.
ABBA_METRICS = (
    ("aggregate_output_tokens_per_s", "aggregate_output_tok/s"),
    ("mean_request_decode_tokens_per_s", "mean_request_decode_tok/s"),
    ("mean_request_prefill_tokens_per_s", "mean_request_prefill_tok/s"),
    ("whole_run_input_tokens_per_s", "whole_run_input_tok/s"),
    ("mean_ttft_s", "mean_ttft_s"),
    ("itl_chunk_median_ms", "itl_chunk_median_ms"),
    ("itl_chunk_mean_ms", "itl_chunk_mean_ms"),
)

SEQUENCE = "FP16-A -> INT4-A -> INT4-B -> FP16-B"


def arm_summary(name: str, arm: Path, loaded: dict) -> dict:
    qualification = loaded["qualification"]
    events = qualification.get("kernel_events") or {}
    return {
        "dir": str(arm),
        "boot_id": loaded["run"].get("boot_id"),
        "gpus": loaded["run"].get("gpus"),
        "cache_dtype": loaded["run"].get("cache_dtype"),
        "raw_status": loaded["status"],
        "qualification_mode": qualification.get("qualification_mode"),
        "hardware_attribution": qualification.get("hardware_attribution"),
        "kernel_events_verdict": events.get("verdict", "incomplete"),
    }


def manifest_of(arm: Path) -> dict | None:
    manifest_file = arm / "manifest.before.json"
    if not manifest_file.exists():
        return None
    return json.loads(manifest_file.read_text())


def check_comparable(loaded: dict[str, dict]) -> list[str]:
    """Every arm must be same-boot, same-config, same-manifest and PASS."""
    problems: list[str] = []
    names = list(loaded)
    base_name = names[0]
    base = loaded[base_name]
    base_manifest = manifest_of(Path(base["dir"]))
    for name in names[1:]:
        other = loaded[name]
        if other["run"].get("boot_id") != base["run"].get("boot_id"):
            problems.append(f"{name}: boot differs from {base_name}")
        for key in ("recipe", "kv_cache_layout", "model_config_sha256", "gpus"):
            if other["run"].get(key) != base["run"].get(key):
                problems.append(f"{name}: {key} differs from {base_name}")
        base_cmd = dict(x.split("=", 1) for x in base["run"]["command"] if "=" in x)
        other_cmd = dict(x.split("=", 1) for x in other["run"]["command"] if "=" in x)
        for key in (
            "RECIPE",
            "MODEL",
            "TP",
            "MTP",
            "SEQS",
            "MAXLEN",
            "MAXBAT",
            "KV",
            "BLOCK_SIZE",
            "GPU_MAX_HW_QUEUES",
            "OMP_NUM_THREADS",
            "TUNABLEOP",
            "COMPILE_MODE",
            "FEATURES",
            "PLE",
        ):
            if base_cmd.get(key) != other_cmd.get(key):
                problems.append(f"{name}: config {key} differs from {base_name}")
        other_manifest = manifest_of(Path(other["dir"]))
        if base_manifest is None or other_manifest is None:
            missing = [
                name
                for name, arm in (  # noqa: B023
                    (base_name, base),
                    (name, other),
                )
                if manifest_of(Path(arm["dir"])) is None
            ]
            problems.append(
                f"{name}: source/binary manifest missing for {', '.join(missing)}"
            )
        elif other_manifest != base_manifest:
            problems.append(f"{name}: source/binary manifest differs from {base_name}")
        for key in ("status", "cleanup"):
            if other["status"].get(key) != "PASS":
                problems.append(f"{name}: raw {key} is {other['status'].get(key)}")
    return problems


def build(arms: dict[str, dict]) -> dict:
    cells: dict[str, list[dict]] = {}
    for cell in CELLS:
        if any(cell not in arms[name]["metrics"] for name in arms):
            continue
        rows = []
        for key, label in ABBA_METRICS:
            fp16_a = arms["FP16-A"]["metrics"][cell][key]
            fp16_b = arms["FP16-B"]["metrics"][cell][key]
            int4_a = arms["INT4-A"]["metrics"][cell][key]
            int4_b = arms["INT4-B"]["metrics"][cell][key]
            pooled_fp16 = (fp16_a + fp16_b) / 2.0
            pooled_int4 = (int4_a + int4_b) / 2.0
            rows.append(
                {
                    "metric": label,
                    "fp16_A": fp16_a,
                    "fp16_B": fp16_b,
                    "int4_A": int4_a,
                    "int4_B": int4_b,
                    "ratio_A": (int4_a / fp16_a) if fp16_a else None,
                    "ratio_B": (int4_b / fp16_b) if fp16_b else None,
                    "pooled_fp16": pooled_fp16,
                    "pooled_int4": pooled_int4,
                    "pooled_delta_pct": (
                        ((pooled_int4 - pooled_fp16) / pooled_fp16 * 100.0)
                        if pooled_fp16
                        else None
                    ),
                }
            )
        cells[cell] = rows
    return {"sequence": SEQUENCE, "arms": arms, "cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fp16_a", type=Path)
    parser.add_argument("int4_a", type=Path)
    parser.add_argument("int4_b", type=Path)
    parser.add_argument("fp16_b", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    dirs = {
        "FP16-A": args.fp16_a,
        "INT4-A": args.int4_a,
        "INT4-B": args.int4_b,
        "FP16-B": args.fp16_b,
    }
    loaded = {name: load(path) for name, path in dirs.items()}
    for name, path in dirs.items():
        loaded[name]["dir"] = str(path)

    arms = {name: arm_summary(name, dirs[name], loaded[name]) for name in dirs}
    problems = check_comparable(loaded)
    report = build(loaded)
    report["arms"] = arms
    report["comparable"] = not problems
    report["problems"] = problems

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)

    if problems:
        print("FAIL: the four arms are not comparable:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("OK: four arms are same-boot, same-config, same-manifest, raw PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
