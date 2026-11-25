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

Every window is journaled: GPU junction temperatures (``gpu_temps.py`` in watcher
mode: any sample >= 95 C or a 5 s mean >= 90 C kills the server tree and exits 3)
and the vLLM tree's KFD eviction counters (``--server-pid``, the server's tree
root; serve.sh writes it to ``SERVER_PID_FILE``).  A missing temperature sample,
a tripped watcher, no ``--server-pid``, no vLLM KFD context or an eviction makes
the window INVALID: it is recorded, printed and exits 3, so the numbers are never
a pass.  ``--baseline bench.json`` prints a cell as a ratio of a stock result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex as re
import slot  # tools/inkling_hi2/slot.py, next to this script

TREE = Path(__file__).resolve().parents[2]
TEMPS_TOOL = Path(
    os.environ.get("TEMPS_TOOL", TREE / "tools/rdna/mono/w4a8/gpu_temps.py")
)
KFD_TOOL = Path(
    os.environ.get("KFD_TOOL", TREE / "tools/rdna/port_v031/kfd_evictions.py")
)
RUN_DIR = Path.home() / "w4a8_runs/inkling-hi2-runtime"

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


def temps_peak(path: Path) -> float | None:
    """Highest junction temperature in a gpu_temps.py CSV, or None if empty."""
    peak = None
    try:
        with open(path, newline="") as f:
            rows = csv.reader(f)
            next(rows, None)
            for row in rows:
                for cell in row[2:]:
                    if cell:
                        peak = max(peak or float(cell), float(cell))
    except OSError:
        return None
    return peak


def window_record(
    peak: float | None,
    evictions: str | None,
    sampled: bool,
) -> dict:
    """Validity of a measured window; no eviction evidence is never a pass."""
    reasons: list[str] = []
    if peak is None:
        reasons.append("no temperature samples")
    elif peak >= 95:
        reasons.append(f"junction {peak:.1f} C >= 95 C")
    if not sampled:
        reasons.append("evictions not sampled (pass --server-pid)")
    elif not evictions or "no vLLM workers" in evictions:
        reasons.append("no vLLM worker KFD context")
    elif "INVALID" in evictions:
        reasons.append("KFD evicted the model")
    return {
        "max_junction_c": peak,
        "evictions": evictions,
        "valid": not reasons,
        "reasons": reasons,
    }


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
    ap.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help=(
            "vLLM tree root (serve.sh SERVER_PID_FILE); required for a valid "
            "window (KFD eviction check)"
        ),
    )
    ap.add_argument("--run-dir", default=str(RUN_DIR), help="where the window CSVs go")
    args = ap.parse_args()

    slot_name = os.environ.get(slot.ENV_SLOT) or slot.DEFAULT_SLOT
    hip, buses = slot.env_pair(slot_name)
    try:
        slot.check_declared(slot_name, hip, buses)
    except slot.SlotError as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    if slot_name != "a2":
        print(
            "bench: TP=2 belongs on a2 (HIP 0,1; PCI d1 d4); "
            f"{slot_name} is the TP4 lease",
            file=sys.stderr,
        )
        return 2

    info = models(args.url)
    name = info["id"]
    result = {
        "label": args.label,
        "slot": {"name": slot_name, "hip": hip, "pci_buses": buses},
        "model": name,
        "max_model_len": info.get("max_model_len"),
        "cells": [],
        "engine": engine_log(args.log),
    }
    print(
        f"bench[{args.label}]: slot={slot_name} HIP={hip} PCI={buses} {name} "
        f"max_model_len={result['max_model_len']} engine={result['engine']}"
    )
    logs = Path(args.run_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    temps_csv = logs / f"temps-bench-{args.label}-{stamp}.csv"
    temps_cmd = [
        sys.executable,
        str(TEMPS_TOOL),
        "--gpus",
        hip,
        "--period",
        "5",
        "--count",
        "0",
        "--out",
        str(temps_csv),
    ]
    if args.server_pid:
        temps_cmd += ["--kill-pid", str(args.server_pid)]
    temps = subprocess.Popen(
        temps_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    evict_before = logs / f"evict-bench-{args.label}-{stamp}.json"
    if args.server_pid:
        subprocess.run(
            [
                sys.executable,
                str(KFD_TOOL),
                str(args.server_pid),
                "--save",
                str(evict_before),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    cells: list[dict] = []
    premature: int | None = None
    try:
        cells.append(prefill(args.url, name, args.prefill_tokens))
        for batch in (int(b) for b in args.batches.split(",") if b):
            if temps.poll() is not None:
                premature = temps.returncode
                break
            cells.append(decode(args.url, name, batch, args.decode_tokens))
    finally:
        if temps.poll() is None:
            temps.terminate()
            try:
                temps.wait(timeout=15)
            except subprocess.TimeoutExpired:
                temps.kill()
                temps.wait()
        else:
            premature = temps.returncode
    result["cells"] = cells
    for cell in cells:
        print("BENCH " + json.dumps(cell))

    since = None
    if args.server_pid:
        since = subprocess.run(
            [
                sys.executable,
                str(KFD_TOOL),
                str(args.server_pid),
                "--since",
                str(evict_before),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    window = window_record(temps_peak(temps_csv), since, bool(args.server_pid))
    window["temps_csv"] = str(temps_csv)
    window["watcher_exit"] = premature
    if premature is not None:
        window["valid"] = False
        window["reasons"].append(f"temperature watcher exited {premature}")
    result["window"] = window
    print("BENCH window " + json.dumps(window))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    if not window["valid"]:
        print(
            "bench: the measured window is INVALID: " + "; ".join(window["reasons"]),
            file=sys.stderr,
        )
        return 3
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
