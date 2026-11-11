#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove native KV store/reload and greedy parity after a GPU-only cache reset.

Run against an isolated test server: resetting its local prefix cache changes
server state. The external CPU tier is deliberately preserved.
"""

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request

import regex as re

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([^\s]+)(?:\s+.*)?$")


def request(base, path, timeout, body=None):
    headers = {"Content-Type": "application/json"} if body is not None else {}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"{path}: HTTP {error.code}: {detail}") from error
    return json.loads(text) if path != "/metrics" else text


def metrics(base, timeout):
    samples = {}
    for line in request(base, "/metrics", timeout).splitlines():
        match = _SAMPLE.match(line)
        if match:
            name, raw = match.groups()
            if name.startswith(("vllm:kv_offload_", "vllm:prefix_cache_hits")):
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite offload counter: {name}")
                samples[name] = samples.get(name, 0.0) + value
    result = {}
    for direction in ("store", "load"):
        name = f"vllm:kv_offload_{direction}_bytes"
        selected = name + "_total" if name + "_total" in samples else name
        result[f"{direction}_bytes"] = samples.get(selected, 0.0)
        result[f"{direction}_metric"] = selected if selected in samples else None
    name = "vllm:prefix_cache_hits"
    selected = name + "_total" if name + "_total" in samples else name
    result["prefix_cache_hits"] = samples.get(selected, 0.0)
    result["prefix_cache_hits_metric"] = selected if selected in samples else None
    return result


def wait_for(base, timeout, wait_timeout, predicate, description):
    deadline = time.monotonic() + wait_timeout
    while True:
        snapshot = metrics(base, min(timeout, 10))
        if predicate(snapshot):
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {description}: {snapshot}")
        time.sleep(0.5)


def prompt(tokens, nonce):
    header = f"Review session {nonce}.\n"
    lines = []
    for index in range(max(1, tokens // 45)):
        lines.append(
            f"def route_batch_{index}(tokens, experts, budget):\n"
            f"    # batch {index}: preserve each weighted pair and fence slot reuse\n"
            f"    return [item for item in tokens if experts[item] < budget]\n"
        )
    return (
        header
        + "\n".join(lines)
        + "\nExplain why an asynchronous expert cache needs consumer fences. "
        "Give one concrete race and its fix.\nAnswer:"
    )


def complete(args, text, salt, max_tokens, *, max_load_tokens=None):
    started = time.monotonic()
    body = {
        "model": args.model,
        "prompt": text,
        "cache_salt": salt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 17,
        "logprobs": 5,
        "return_tokens_as_token_ids": True,
    }
    if max_load_tokens is not None:
        body["kv_transfer_params"] = {"max_load_tokens": max_load_tokens}
    response = request(
        args.base_url,
        "/v1/completions",
        args.request_timeout,
        body,
    )
    choices = response.get("choices", [])
    if len(choices) != 1 or not choices[0].get("text", "").strip():
        raise RuntimeError("completion returned no nonempty single output")
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    values = logprobs.get("token_logprobs") or []
    if not tokens or len(tokens) != len(values):
        raise RuntimeError("completion omitted aligned output token logprobs")
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise RuntimeError("completion returned invalid token logprobs")
    top = logprobs.get("top_logprobs")
    if top is not None:
        if len(top) != len(tokens) or any(
            not isinstance(row, dict)
            or any(not math.isfinite(float(value)) for value in row.values())
            for row in top
        ):
            raise RuntimeError("completion returned invalid top token logprobs")
        top = [{token: float(value) for token, value in row.items()} for row in top]
    usage = response.get("usage") or {}
    if usage.get("completion_tokens", 0) <= 0:
        raise RuntimeError("completion reported no generated tokens")
    return {
        "text": choice["text"],
        "tokens": tokens,
        "token_logprobs": [float(value) for value in values],
        "top_logprobs": top,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "elapsed_seconds": time.monotonic() - started,
    }


def compare(cold, loaded, atol, *, enforce_logprobs=True):
    if cold["text"] != loaded["text"] or cold["tokens"] != loaded["tokens"]:
        raise RuntimeError("compared output text/token IDs differ")
    if cold["finish_reason"] != loaded["finish_reason"]:
        raise RuntimeError("reloaded completion finish reason changed")
    diffs = [
        abs(left - right)
        for left, right in zip(
            cold["token_logprobs"], loaded["token_logprobs"], strict=True
        )
    ]
    worst = max(diffs)
    if enforce_logprobs and worst > atol:
        raise RuntimeError(f"token logprob difference {worst:.6g} exceeds {atol}")
    return {
        "identical_tokens": len(diffs),
        "max_abs_token_logprob_delta": worst,
        "within_logprob_tolerance": worst <= atol,
        "logprob_tolerance_enforced": enforce_logprobs,
    }


def gpu_cached_reference(args, text, nonce, report, key):
    before = report[f"metrics_before_{key}"] = metrics(
        args.base_url, args.request_timeout
    )
    completion = report[key] = complete(args, text, nonce, args.max_tokens)
    cached_tokens = (completion["usage"].get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    after = report[f"metrics_after_{key}"] = wait_for(
        args.base_url,
        args.request_timeout,
        args.wait_timeout,
        lambda snapshot: (
            (cached_tokens is not None and cached_tokens > 0)
            or snapshot["prefix_cache_hits"] > before["prefix_cache_hits"]
        ),
        f"{key} GPU-resident cached-prefix hit evidence",
    )
    if after["load_bytes"] != before["load_bytes"]:
        raise RuntimeError(f"{key} unexpectedly loaded external KV")
    report[f"{key}_hit_evidence"] = {
        "usage_cached_tokens": cached_tokens,
        "prefix_cache_hits_delta": after["prefix_cache_hits"]
        - before["prefix_cache_hits"],
        "external_load_bytes_delta": 0,
    }
    return completion


def run(args, report):
    nonce = f"expert-kv-{time.time_ns()}"
    text = prompt(args.prompt_tokens, nonce)
    report.update(
        base_url=args.base_url,
        model=args.model,
        cache_salt=nonce,
        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        approximate_prompt_tokens=args.prompt_tokens,
        prompt_characters=len(text),
        logprob_atol=args.logprob_atol,
    )
    before = report["metrics_before"] = metrics(args.base_url, args.request_timeout)
    cold = report["cold"] = complete(args, text, nonce, args.max_tokens)
    if cold["usage"].get("prompt_tokens", 0) < 2048:
        raise RuntimeError("actual prompt is shorter than two 1024-token blocks")
    gpu_cached = gpu_cached_reference(args, text, nonce, report, "gpu_cached")
    gpu_repeat = gpu_cached_reference(args, text, nonce, report, "gpu_cached_repeat")
    report["gpu_cached_repeat_parity"] = compare(
        gpu_cached, gpu_repeat, args.logprob_atol, enforce_logprobs=False
    )
    reload_limit = None
    if getattr(args, "match_reload_boundary", False):
        reload_limit = (gpu_cached["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        if type(reload_limit) is not int or reload_limit <= 0:
            raise RuntimeError(
                "matching the reload boundary "
                "requires positive GPU usage cached_tokens; "
                "launch with --enable-prompt-tokens-details"
            )
    report["matched_reload_boundary_tokens"] = reload_limit
    request_cap = (
        reload_limit + getattr(args, "reload_cap_slack", 0)
        if reload_limit is not None
        else None
    )
    report["reload_request_max_load_tokens"] = request_cap
    # Cold prefill and cached-prefix execution may use different arithmetic
    # shapes. Keep token parity strict and report their numerical baseline;
    # CPU reload must match the GPU-cached reference at the original tolerance.
    report["cold_vs_gpu_cache_parity"] = compare(
        cold, gpu_cached, args.logprob_atol, enforce_logprobs=False
    )
    # A subsequent forward submits stores deferred from the final cached step.
    complete(args, "State the word ready.", nonce + "-flush", 8)
    stored = report["metrics_after_store"] = wait_for(
        args.base_url,
        args.request_timeout,
        args.wait_timeout,
        lambda snapshot: snapshot["store_bytes"] > before["store_bytes"],
        "positive native KV store bytes",
    )
    deadline = time.monotonic() + args.wait_timeout
    while True:
        reset = request(
            args.base_url,
            "/reset_prefix_cache?reset_running_requests=false&reset_external=false",
            args.request_timeout,
            {},
        )
        report["reset"] = reset
        if reset.get("success") is True:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("local prefix cache reset never succeeded")
        time.sleep(0.5)
    before_load = report["metrics_before_reload"] = metrics(
        args.base_url, args.request_timeout
    )
    if reload_limit is None:
        loaded = complete(args, text, nonce, args.max_tokens)
    else:
        loaded = complete(
            args, text, nonce, args.max_tokens, max_load_tokens=request_cap
        )
    report["reloaded"] = loaded
    after = report["metrics_after_reload"] = wait_for(
        args.base_url,
        args.request_timeout,
        args.wait_timeout,
        lambda snapshot: snapshot["load_bytes"] > before_load["load_bytes"],
        "positive native KV reload bytes after GPU-only reset",
    )
    report["parity_reference"] = "gpu_cached"
    report["cold_vs_reloaded_parity"] = compare(
        cold, loaded, args.logprob_atol, enforce_logprobs=False
    )
    report["gpu_repeat_vs_reloaded_parity"] = compare(
        gpu_repeat, loaded, args.logprob_atol, enforce_logprobs=False
    )
    if reload_limit is not None:
        actual_cached = (loaded["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        if actual_cached != reload_limit:
            raise RuntimeError(
                f"CPU reload cached {actual_cached} tokens; "
                f"expected exactly {reload_limit}"
            )
    report["parity"] = compare(
        gpu_cached, loaded, args.logprob_atol, enforce_logprobs=False
    )
    report["store_bytes_delta"] = stored["store_bytes"] - before["store_bytes"]
    report["load_bytes_delta"] = after["load_bytes"] - before_load["load_bytes"]
    report["external_cpu_cache_preserved"] = True
    failures = []
    for key, label in (
        ("gpu_cached_repeat_parity", "unstable GPU-resident reference"),
        ("parity", "CPU reload versus original GPU reference"),
    ):
        parity = report[key]
        parity["logprob_tolerance_enforced"] = True
        if not parity["within_logprob_tolerance"]:
            failures.append(
                f"{label}: token logprob difference "
                f"{parity['max_abs_token_logprob_delta']:.6g} "
                f"exceeds {args.logprob_atol}"
            )
    report["validation_failures"] = failures
    if failures:
        raise RuntimeError("; ".join(failures))
    report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18766")
    parser.add_argument("--model", default="flash-next")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--wait-timeout", type=float, default=30)
    parser.add_argument("--logprob-atol", type=float, default=0.05)
    parser.add_argument(
        "--match-reload-boundary",
        action="store_true",
        help="Cap CPU reload to the original GPU cached_tokens and require equal usage",
    )
    parser.add_argument(
        "--reload-cap-slack",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "With --match-reload-boundary, allow 1 token for native Mamba lookup's "
            "pre-alignment subtraction; actual restored usage must still match exactly"
        ),
    )
    parser.add_argument("--out", help="Also save the complete JSON report here")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    if (
        min(
            args.prompt_tokens, args.max_tokens, args.request_timeout, args.wait_timeout
        )
        <= 0
    ):
        parser.error("token counts and timeouts must be positive")
    if args.logprob_atol < 0 or not math.isfinite(args.logprob_atol):
        parser.error("logprob-atol must be finite and nonnegative")
    if args.reload_cap_slack and not args.match_reload_boundary:
        parser.error("reload-cap-slack requires --match-reload-boundary")
    report = {"passed": False, "scope": "isolated native KV store/reset/load probe"}
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w") as destination:
            destination.write(output + "\n")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
