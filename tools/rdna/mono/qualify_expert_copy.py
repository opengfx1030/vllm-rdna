#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quiescent, same-server expert-copy A/B with retained per-rank counters.

Requires ExpertCacheWorker and the development RPC endpoint on an isolated
server. GPU and external prefix state reset each cell; expert residency remains.
Alternating arms reduce order bias but do not give identical cache histories.
Performance requests force output lengths and are not semantic quality gates.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from tools.rdna.mono.qualify_offload_long import bench, post

CELLS = {
    "1k-c1": (1024, 512, 1, 1),
    "1k-c8": (1024, 512, 8, 8),
    "16k-c1": (16384, 1024, 1, 1),
    "16k-c8": (16384, 1024, 8, 8),
}
COUNTERS = (
    "expert_demands",
    "expert_misses",
    "committed_passes",
    "copied_bytes",
    "native_copy_calls",
    "batch_copy_calls",
    "map_uploads",
)


def check_control_result(result, expected_caches=48, mode=""):
    """Validate an expert-cache control reply for any rank count.

    Every rank must report the same nonzero cache count (one per MoE layer,
    48 on Qwen4Exp) and, when switching, the requested copy backend.
    """
    if not result:
        raise ValueError("expert-cache control returned no ranks")
    counts = {rank["caches"] for rank in result}
    if counts != {expected_caches} or expected_caches < 1:
        raise ValueError(
            f"expected {expected_caches} expert caches on every "
            f"({len(result)}-rank) server: {result}"
        )
    if mode and any(rank["copy_modes"] != [mode] for rank in result):
        raise ValueError(f"copy backend switch failed: {result}")
    return result


def control(args, mode=""):
    result = post(
        args,
        "/collective_rpc",
        {"method": "rdna_expert_cache_control", "kwargs": {"copy_mode": mode}},
    )["results"]
    return check_control_result(
        result, getattr(args, "expected_caches", 48), mode
    )


def reset_prefix(args):
    for _ in range(5):
        request = urllib.request.Request(
            args.base_url + "/reset_prefix_cache?reset_external=true", method="POST"
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 200 and json.load(response).get("success") is True:
                return
        time.sleep(1)
    raise RuntimeError("prefix reset did not acknowledge success")


def check_cold_prefix(result):
    for name in ("vllm:prefix_cache_hits", "vllm:external_prefix_cache_hits"):
        if result["counter_deltas"].get(name) != 0:
            raise RuntimeError(
                f"cold-prefix benchmark has hits or missing counter: {name}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=f"http://127.0.0.1:{os.getenv('PORT', '18766')}"
    )
    parser.add_argument("--model", default=os.getenv("SERVED", "flash-next"))
    parser.add_argument("--tokenizer", default=os.getenv("MODEL"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-caches",
        type=int,
        default=48,
        help="expert caches every rank must report (48 MoE layers on Qwen4Exp)",
    )
    parser.add_argument("--cell-timeout", type=float, default=900)
    parser.add_argument("--cells", choices=CELLS, nargs="+", default=list(CELLS))
    parser.add_argument(
        "--arms",
        choices=("tensor", "native"),
        nargs="+",
        default=["tensor", "native", "native", "tensor"],
    )
    args = parser.parse_args()
    if not args.tokenizer or not 0 < args.cell_timeout <= 3600:
        parser.error("tokenizer and a bounded positive cell timeout are required")
    root = args.out
    args.skip_ready_check = True
    args.optional_kv_counters = True
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "scope": (
            "same-server copy A/B; cold GPU/CPU prefixes, retained expert residency"
        ),
        "counter_scope": (
            "measured requests only; client ready-check request disabled after "
            "worker RPC readiness check; no client warmup requests"
        ),
        "semantic_quality": "run normal-EOS/PPL/prefix gates separately",
        "cells": [],
    }
    report_path = root / "qualification.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for name in args.cells:
            for index, mode in enumerate(args.arms):
                before = control(args, mode)
                reset_prefix(args)
                args.out = root / f"{name}-{index}-{mode}"
                args.out.mkdir()
                result = bench(args, CELLS[name], 12345)
                check_cold_prefix(result)
                after = control(args)
                deltas = []
                for old, new in zip(before, after):
                    if (
                        old["device_bytes"] != new["device_bytes"]
                        or old["host_bytes"] != new["host_bytes"]
                    ):
                        raise RuntimeError("expert pool size changed")
                    delta = {key: new[key] - old[key] for key in COUNTERS}
                    if any(value < 0 for value in delta.values()):
                        raise RuntimeError("cache counters decreased")
                    delta["pass_expert_hit_fraction"] = (
                        1 - delta["expert_misses"] / delta["expert_demands"]
                        if delta["expert_demands"]
                        else None
                    )
                    if (
                        mode == "native"
                        and delta["expert_misses"]
                        and not delta["native_copy_calls"]
                    ):
                        raise RuntimeError("native arm did not submit copies")
                    deltas.append(delta)
                report["cells"].append(
                    {
                        "cell": name,
                        "arm": index,
                        "copy_mode": mode,
                        "before": before,
                        "after": after,
                        "deltas": deltas,
                        "benchmark": result,
                    }
                )
                save()
                print(
                    f"{name} arm={index} mode={mode}: {result['headline']}", flush=True
                )
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
    print("PASS expert-copy A/B; inspect timings and separate quality gates")


if __name__ == "__main__":
    main()
