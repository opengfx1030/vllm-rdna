#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode / prefill / memory cells of one running Inkling-2 server (HTTP only).

One server per invocation (a port), so the same cells can be run against the stock
directory (a reference) and against the HI2 directory on the same lease:

    python tools/inkling_hi2/bench.py --url http://127.0.0.1:18231 --label stock \\
        --log ~/w4a8_runs/inkling-hi2-runtime/logs/stock.log
    python tools/inkling_hi2/bench.py --url http://127.0.0.1:18230 --label hi2 \\
        --log ~/w4a8_runs/inkling-hi2-runtime/logs/serve.log

Cells (all greedy, ``ignore_eos``):
* prefill ``--prefill-tokens`` at batch 1: prompt-tokens/s from ``usage`` and wall;
* decode ``--decode-tokens`` at batch 1 and 4: wall per decode step (the prefill is
  included, so it is an upper bound; the same prompt is used for stock and HI2);
* memory: the engine log lines ("Model loading took ... GiB", "GPU KV cache size",
  "Available KV cache memory", graph capture) and the served ``max_model_len``.

No pass/fail: the numbers are the record; ``--baseline bench.json`` prints the HI2
cell as a ratio of a stock result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import regex as re

PARAGRAPH = (
    "The ink dries slowly on the page while the reader follows the line of the "
    "argument through the margin notes. "
)
PROMPT = "Continue the text with one short sentence."


def call(url: str, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def models(url: str) -> dict:
    with urllib.request.urlopen(url + "/v1/models", timeout=60) as r:
        return json.loads(r.read())["data"][0]


def completion(url: str, model: str, prompt, tokens: int) -> tuple[dict, float]:
    t0 = time.time()
    out = call(
        url,
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    )
    return out, time.time() - t0


def prefill(url: str, model: str, approx_tokens: int) -> dict:
    text = PARAGRAPH * max(1, math.ceil(approx_tokens / 24))
    out, elapsed = completion(url, model, text, 1)
    used = out["usage"]["prompt_tokens"]
    return {
        "cell": "prefill",
        "prompt_tokens": used,
        "seconds": elapsed,
        "prompt_tokens_per_s": used / elapsed,
    }


def decode(url: str, model: str, batch: int, tokens: int) -> dict:
    def one(_):
        return completion(url, model, PROMPT, tokens)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=batch) as pool:
        outs = list(pool.map(one, range(batch)))
    wall = time.time() - t0
    generated = sum(o["usage"]["completion_tokens"] for o, _ in outs)
    steps = max(o["usage"]["completion_tokens"] for o, _ in outs)
    return {
        "cell": "decode",
        "batch": batch,
        "completion_tokens": generated,
        "seconds": wall,
        "ms_per_step": 1000.0 * wall / max(steps, 1),
        "tokens_per_s": generated / wall,
    }


def engine_log(path: str | None) -> dict:
    if not path:
        return {}
    keys = {
        "model_loading_gib": r"Model loading took ([\d.]+) GiB",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
        "graph_capture_s": r"Graph capturing finished in ([\d.]+) sec",
        "workspace_gib": r"workspace ([\d.]+) GiB",
    }
    found = {}
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return found
    for name, pattern in keys.items():
        m = re.findall(pattern, text)
        if m:
            found[name] = m[-1]
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--prefill-tokens", type=int, default=1000)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--log", default=None)
    ap.add_argument("--baseline", default=None, help="stock BENCH JSON to compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    info = models(args.url)
    name = info["id"]
    result = {
        "label": args.label,
        "model": name,
        "max_model_len": info.get("max_model_len"),
        "cells": [],
        "engine": engine_log(args.log),
    }
    print(
        f"bench[{args.label}]: {name} max_model_len={result['max_model_len']} "
        f"engine={result['engine']}"
    )
    cells = [prefill(args.url, name, args.prefill_tokens)]
    for batch in (int(b) for b in args.batches.split(",") if b):
        cells.append(decode(args.url, name, batch, args.decode_tokens))
    result["cells"] = cells
    for cell in cells:
        print("BENCH " + json.dumps(cell))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print(f"bench: baseline {base.get('label')} {base.get('max_model_len')}")
        for got, want in zip(cells, base.get("cells", [])):
            if got["cell"] == "prefill":
                print(
                    f"bench: prefill ratio "
                    f"{got['prompt_tokens_per_s'] / want['prompt_tokens_per_s']:.3f}x"
                )
            else:
                print(
                    f"bench: decode b={got['batch']} step ratio "
                    f"{got['ms_per_step'] / want['ms_per_step']:.3f}x "
                    f"(lower is better)"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
