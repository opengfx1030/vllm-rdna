#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-line headline metrics from `vllm bench serve --save-result` JSON files.

    python bench_metrics.py bench-16384x1024-c8.json [...]

Per cell: prefill tok/s per request (mean input length / mean TTFT) and in
aggregate (one wave of `max_concurrency` prompts / p99 TTFT, i.e. until the
last request of a wave got its first token), decode tok/s per request
(1000 / mean TPOT) and aggregate output tok/s, TTFT (s), ITL (ms, median and
mean). Stdlib only. ``metrics`` returns the same numbers as a dict (plus the
TTFT and ITL percentiles) for tools/rdna/pr_receipt_report.py.
"""

import json
import sys


def metrics(d: dict) -> dict[str, float]:
    """Headline numbers of one bench result (ms for latencies, tok/s for rates)."""
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    ttft_mean_ms = d.get("mean_ttft_ms", 0.0)
    ttft_p99_ms = d.get("p99_ttft_ms", 0.0)
    tpot_ms = d.get("mean_tpot_ms", 0.0)
    wave = min(int(d.get("max_concurrency") or 1), done)
    return {
        "completed": float(done),
        "in_len": in_len,
        "ttft_mean_ms": ttft_mean_ms,
        "ttft_p50_ms": d.get("median_ttft_ms", 0.0),
        "ttft_p99_ms": ttft_p99_ms,
        "prefill_req_tps": in_len / (ttft_mean_ms / 1000.0)
        if ttft_mean_ms > 0
        else 0.0,
        "prefill_agg_tps": wave * in_len / (ttft_p99_ms / 1000.0)
        if ttft_p99_ms > 0
        else 0.0,
        "itl_mean_ms": d.get("mean_itl_ms", 0.0),
        "itl_p50_ms": d.get("median_itl_ms", 0.0),
        "itl_p99_ms": d.get("p99_itl_ms", 0.0),
        "tpot_mean_ms": tpot_ms,
        "decode_req_tps": 1000.0 / tpot_ms if tpot_ms > 0 else 0.0,
        "out_tps": d.get("output_throughput", 0.0),
    }


def line(path: str) -> str:
    with open(path) as f:
        d = json.load(f)
    m = metrics(d)
    name = path.rsplit("/", 1)[-1].removesuffix(".json")
    return (
        f"{name:28} prefill {m['prefill_req_tps']:8.1f} tok/s/req "
        f"(agg {m['prefill_agg_tps']:7.1f}) | "
        f"decode {m['decode_req_tps']:6.2f} tok/s/req "
        f"(agg out {m['out_tps']:7.2f}) | TTFT {m['ttft_mean_ms'] / 1000.0:7.2f} s | "
        f"ITL med {m['itl_p50_ms']:7.1f} ms mean "
        f"{m['itl_mean_ms']:7.1f} ms | ok {int(m['completed'])}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(line(p))
        except (OSError, ValueError, KeyError) as e:
            print(f"{p}: unreadable ({e})")
