#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded long-context LOAD_HOOK; uses vllm bench serve and normal-EOS retrieval.

The performance cells force their requested lengths. Their random-prompt
outputs are retained and screened, but are not semantic correctness evidence.
The separate 32k-class retrieval request uses normal EOS and a known marker.
Run only against the isolated qualification server; no server is started here.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import regex as re

TREE = Path(__file__).resolve().parents[3]
CELLS = ((16384, 1024, 1, 1), (16384, 1024, 8, 8), (32256, 256, 1, 1))
CELL_NAMES = ("16k-c1", "16k-c8", "32k-c1")
CELL_CONFIGS = dict(zip(CELL_NAMES, CELLS))
_METRIC = re.compile(r"^(vllm:[\w:]+)(?:\{[^}]*\})?\s+([^\s]+)")
COUNTERS = (
    "vllm:num_preemptions",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:kv_offload_load_bytes",
    "vllm:kv_offload_store_bytes",
    "vllm:kv_offload_load_time",
    "vllm:kv_offload_store_time",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
)

REQUEST_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def load_tool(name, relative):
    spec = importlib.util.spec_from_file_location(name, TREE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match:
            name, raw = match.groups()
            if (
                name.startswith(COUNTERS)
                or name in REQUEST_GAUGES
                or "cache_usage_perc" in name
            ):
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite metric {name}")
                # Counters sum across engine labels; occupancy reports the
                # busiest engine rather than adding percentages.
                if "usage_perc" in name:
                    values[name] = max(values.get(name, 0), value)
                else:
                    values[name] = values.get(name, 0) + value
    return values


def get_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        return parse_metrics(response.read().decode())


def post(args, path, body):
    req = urllib.request.Request(
        args.base_url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.cell_timeout) as response:
        return json.load(response)


class MetricSamples:
    def __init__(self, base, path):
        self.base, self.path = base, path
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def sample(self):
        row = {"monotonic_seconds": time.monotonic()}
        try:
            row["metrics"] = get_metrics(self.base)
        except Exception as error:
            row["error"] = str(error)
        self.rows.append(row)

    def poll(self):
        while not self.stop.wait(1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)
        self.sample()
        self.path.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")

    def summary(self, require_boundary_counters=True):
        if len(self.rows) < 2 or any(
            "metrics" not in row for row in (self.rows[0], self.rows[-1])
        ):
            raise RuntimeError("missing successful before/after server metrics")
        samples = [row["metrics"] for row in self.rows if "metrics" in row]
        before, after = self.rows[0]["metrics"], self.rows[-1]["metrics"]
        delta = {}
        required = {
            "vllm:num_preemptions",
            "vllm:kv_offload_load_bytes",
            "vllm:kv_offload_store_bytes",
        }
        for key in COUNTERS:
            name = key + "_total" if key + "_total" in before else key
            if name not in before or name not in after:
                if key in required and (
                    require_boundary_counters or key == "vllm:num_preemptions"
                ):
                    raise RuntimeError(f"required boundary counter unavailable: {key}")
                delta[key] = None
                continue
            delta[key] = after[name] - before[name]
            if delta[key] < 0:
                raise RuntimeError(f"counter reset during qualification: {name}")
        return {
            "counter_deltas": delta,
            "peak_cache_usage": {
                key: max(sample.get(key, 0) for sample in samples)
                for key in {key for sample in samples for key in sample}
                if "usage_perc" in key
            },
            "peak_requests": {
                key: max(sample.get(key, 0) for sample in samples)
                for key in REQUEST_GAUGES
                if any(key in sample for sample in samples)
            },
            "missing_request_gauges": [
                key for key in REQUEST_GAUGES if not any(key in s for s in samples)
            ],
            "metric_sample_errors": sum("error" in row for row in self.rows),
        }


INPUT_LENGTH_TOLERANCE = 0.01
MAX_CONTEXT_TOKENS = 32768


def check_bench(result, requested, output_len, input_len):
    texts = result.get("generated_texts", [])
    if result.get("failed", 0) or result.get("completed") != requested:
        raise RuntimeError("benchmark did not complete every request")
    if (
        not isinstance(texts, list)
        or len(texts) != requested
        or any(not isinstance(text, str) for text in texts)
    ):
        raise RuntimeError("benchmark omitted or returned malformed detailed outputs")
    if result.get("output_lens") != [output_len] * requested:
        raise RuntimeError("forced-length benchmark did not generate requested tokens")
    actual_inputs = result.get("input_lens", [])
    if len(actual_inputs) != requested or any(
        not isinstance(length, int)
        or isinstance(length, bool)
        or abs(length - input_len) > input_len * INPUT_LENGTH_TOLERANCE
        or length + output_len > MAX_CONTEXT_TOKENS
        for length in actual_inputs
    ):
        raise RuntimeError(
            "actual prompt lengths violate 1% tolerance or context limit"
        )
    quality = load_tool("offload_long_quality", "tools/rdna/port_v031/ppl_probe.py")
    return [
        {
            "request": index,
            "garbage": quality.garbage_reason(text),
            "repetition_warning": quality.repeat_warning(text),
        }
        for index, text in enumerate(texts)
        if quality.garbage_reason(text) or quality.repeat_warning(text)
    ]


FIXED_RANGE_RATIO = 0.0  # Zero width around the requested input/output means.


def bench(args, cell, seed):
    input_len, output_len, concurrency, requests = cell
    name = f"bench-{input_len}x{output_len}-c{concurrency}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(FIXED_RANGE_RATIO),
        "--num-prompts",
        str(requests),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        str(getattr(args, "request_rate", "inf")),
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        str(seed),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(args.out),
        "--result-filename",
        name + ".json",
        "--disable-tqdm",
    ]
    if getattr(args, "skip_ready_check", False):
        # The same-server A/B already checks worker readiness via RPC. Avoid
        # an unmeasured full request warming prefix/expert state in each arm.
        command.extend(["--ready-check-timeout-sec", "0"])
    result_path = args.out / f"{name}.json"
    result_path.unlink(
        missing_ok=True
    )  # A successful client must write this run's result.
    with MetricSamples(args.base_url, args.out / f"{name}-metrics.jsonl") as samples:
        with (args.out / f"{name}.log").open("w") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=args.cell_timeout
            )
        if completed.returncode:
            raise RuntimeError(f"{name} exited {completed.returncode}; inspect its log")
    result = json.loads(result_path.read_text())
    warnings = check_bench(result, requests, output_len, input_len)
    formatter = load_tool(
        "offload_long_metrics", "tools/rdna/port_v031/bench_metrics.py"
    )
    return {
        "name": name,
        "command": command,
        "forced_ignore_eos": True,
        "requested_input_tokens": input_len,
        "input_length_relative_tolerance": INPUT_LENGTH_TOLERANCE,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "requested_output_tokens": output_len,
        "actual_input_tokens": result.get("input_lens"),
        "actual_output_tokens": result.get("output_lens"),
        "headline": formatter.line(str(args.out / f"{name}.json")),
        "output_quality_warnings": warnings,
        **samples.summary(
            require_boundary_counters=not getattr(args, "optional_kv_counters", False)
        ),
    }


def retrieval(args):
    nonce = f"long-{time.time_ns()}"
    marker = "ORBITAL_FENCE_4729"
    source = (
        TREE / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
    ).read_text()

    def encode(text):
        return post(
            args,
            "/tokenize",
            {"model": args.model, "prompt": text, "add_special_tokens": False},
        )["tokens"]

    prefix = encode(f"Session {nonce}. Deployment marker: {marker}.\nSource context:\n")
    suffix = encode(
        "\nWhat deployment marker was given at the start? "
        "Return only that exact marker, without explanation.\nDeployment marker:"
    )
    corpus = encode(source)
    remaining = 32256 - len(prefix) - len(suffix)
    if not corpus or remaining <= 0:
        raise RuntimeError("cannot build bounded long retrieval prompt")
    ids = prefix + (corpus * (remaining // len(corpus) + 1))[:remaining] + suffix
    with MetricSamples(args.base_url, args.out / "retrieval-metrics.jsonl") as samples:
        response = post(
            args,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": ids,
                "max_tokens": 256,
                "temperature": 0,
                "seed": 17,
                "cache_salt": nonce,
            },
        )
    (args.out / "retrieval.json").write_text(json.dumps(response, indent=2))
    text = response.get("choices", [{}])[0].get("text", "")
    if marker not in text:
        raise RuntimeError("32k-class normal-EOS retrieval omitted the known marker")
    usage = response.get("usage") or {}
    if not 32000 <= usage.get("prompt_tokens", 0) <= 32512:
        raise RuntimeError("retrieval prompt length is outside its bounded window")
    quality = load_tool("offload_long_quality", "tools/rdna/port_v031/ppl_probe.py")
    reason = quality.garbage_reason(text) or quality.repeat_warning(text)
    if reason:
        raise RuntimeError(f"normal-EOS retrieval output quality failed: {reason}")
    return {
        "normal_eos": True,
        "marker_found": True,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": usage,
        "text": text,
        **samples.summary(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=f"http://127.0.0.1:{os.getenv('PORT', '18766')}"
    )
    parser.add_argument("--model", default=os.getenv("SERVED", "flash-next"))
    parser.add_argument("--tokenizer", default=os.getenv("MODEL"))
    parser.add_argument(
        "--out", type=Path, default=Path(os.getenv("OUT", ".")) / "long"
    )
    parser.add_argument("--cell-timeout", type=float, default=900)
    parser.add_argument(
        "--cells", choices=CELL_NAMES, nargs="+", default=list(CELL_NAMES)
    )
    args = parser.parse_args()
    if (
        not args.tokenizer
        or not math.isfinite(args.cell_timeout)
        or args.cell_timeout <= 0
    ):
        parser.error(
            "--tokenizer/MODEL and a positive bounded --cell-timeout are required"
        )
    args.base_url = args.base_url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": False,
        "cells": [],
        "selected_cells": args.cells,
        "partial_scope": set(args.cells) != set(CELL_NAMES),
        "scope": (
            "selected performance cells plus normal-EOS retrieval; "
            "no baseline speedup claim"
        ),
        "transfer_contention": (
            "not attributed; transfer counters and per-request latency recorded"
        ),
    }
    seed = time.time_ns() % (1 << 31)
    try:
        for index, name in enumerate(args.cells):
            cell = CELL_CONFIGS[name]
            report["cells"].append(bench(args, cell, seed + index))
        report["retrieval"] = retrieval(args)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (args.out / "qualification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(
        f"PASS: selected long workload qualification ({', '.join(args.cells)})"
        if report["passed"]
        else "FAIL: long workload qualification"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
