#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-line headline metrics from `vllm bench serve --save-result` JSON files.

    python bench_metrics.py bench-16384x1024-c8.json [...]

Per cell: prefill tok/s per request (mean input length / mean TTFT) and in
aggregate (all prompt tokens / p99 TTFT, i.e. until the last request of the
burst got its first token), decode tok/s per request (1000 / mean TPOT) and
aggregate output tok/s, TTFT (s), ITL (ms, median and mean). Stdlib only.
"""

import json
import sys


def line(path: str) -> str:
    with open(path) as f:
        d = json.load(f)
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    ttft_s = d.get("mean_ttft_ms", 0.0) / 1000.0
    tpot_ms = d.get("mean_tpot_ms", 0.0)
    prefill = in_len / ttft_s if ttft_s > 0 else 0.0
    p99_ttft_s = d.get("p99_ttft_ms", 0.0) / 1000.0
    total_in = d.get("total_input_tokens", 0)
    prefill_agg = total_in / p99_ttft_s if p99_ttft_s > 0 else 0.0
    decode = 1000.0 / tpot_ms if tpot_ms > 0 else 0.0
    name = path.rsplit("/", 1)[-1].removesuffix(".json")
    return (
        f"{name:28} prefill {prefill:8.1f} tok/s/req (agg {prefill_agg:7.1f}) | "
        f"decode {decode:6.2f} tok/s/req "
        f"(agg out {d.get('output_throughput', 0.0):7.2f}) | TTFT {ttft_s:7.2f} s | "
        f"ITL med {d.get('median_itl_ms', 0.0):7.1f} ms mean "
        f"{d.get('mean_itl_ms', 0.0):7.1f} ms | ok {done}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(line(p))
        except (OSError, ValueError, KeyError) as e:
            print(f"{p}: unreadable ({e})")
