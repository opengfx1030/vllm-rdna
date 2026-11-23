#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Send one long prompt twice and check the warm request's prefix-cache hits.

    python prefix_probe.py --url http://127.0.0.1:18120/v1/completions \
        --model flash-next --words 1800

The verdict is the prefix-cache hit rate the server reports for the two
requests, read from vllm:prefix_cache_queries_total and
vllm:prefix_cache_hits_total before and after them: the warm request must hit
more than half of its prompt tokens. TTFT is always printed, but it only gates
the verdict when --ratio is passed explicitly (warm TTFT <= ratio * cold
TTFT). The old default of 0.5 demanded a 2x speedup even when the cache was
working, because a hit only removes the prefill of the reused blocks.

Cache granularity belongs to the served model, not this probe. The probe asks
the server's /tokenize endpoint for the real prompt length and validates
--max-model-len with it; --words only sizes the generated prompt. Whether any
blocks are reusable is proved by the measured hits, never by a hardcoded
token threshold. Tokenize and metrics reads fail closed instead of inventing
a length or zero counters.
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass

import regex as re

# A warm request must reuse a strict majority of its prompt tokens.
_MIN_WARM_HIT_RATE = 0.5


class ProbeError(RuntimeError):
    """The server did not provide the evidence the probe needs."""


@dataclass(frozen=True)
class Verdict:
    """Measured prefix-cache reuse, with TTFT as an optional gate."""

    passed: bool
    hit_tokens: float
    warm_queries: float
    hit_rate: float
    # None when --ratio was not supplied: TTFT is context, not a gate.
    latency_met: bool | None


def ttft(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, str]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    first = None
    text = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            piece = chunk["choices"][0].get("text", "")
            if first is None and piece:
                first = time.monotonic() - t0
            text.append(piece)
    if first is None:
        raise ProbeError("completion stream ended without any output text")
    return first, "".join(text)


def _server_root(url: str) -> str:
    """Strip the OpenAI route off a vLLM endpoint, keeping scheme and port."""
    return re.sub(r"/v1(?:/.*)?$", "", url)


def tokenize_count(url: str, model: str, prompt: str) -> int:
    """Return the server's real token count for prompt via POST /tokenize."""
    tokenize_url = _server_root(url) + "/tokenize"
    body = json.dumps({"model": model, "prompt": prompt}).encode()
    req = urllib.request.Request(
        tokenize_url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except (OSError, ValueError) as exc:
        raise ProbeError(
            f"tokenize endpoint {tokenize_url} unavailable or malformed: {exc}"
        ) from exc
    count = payload.get("count") if isinstance(payload, dict) else None
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ProbeError(
            f"tokenize endpoint {tokenize_url} returned no positive integer token count"
        )
    return count


def prefix_counters(url: str) -> tuple[float, float]:
    """Return summed (queried tokens, hit tokens) from the server's /metrics."""
    metrics_url = _server_root(url) + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode()
    except (OSError, ValueError) as exc:
        raise ProbeError(
            f"metrics endpoint {metrics_url} unavailable or unreadable: {exc}"
        ) from exc
    counters = {}
    for name in ("queries_total", "hits_total"):
        values = re.findall(
            rf"^vllm:prefix_cache_{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
            text,
            re.MULTILINE,
        )
        if not values:
            raise ProbeError(
                f"metrics endpoint {metrics_url} has no "
                f"vllm:prefix_cache_{name} counter"
            )
        try:
            counters[name] = sum(float(value) for value in values)
        except ValueError as exc:
            raise ProbeError(
                f"metrics endpoint {metrics_url} has a malformed "
                f"vllm:prefix_cache_{name} value"
            ) from exc
    return counters["queries_total"], counters["hits_total"]


def check_prompt(prompt_tokens: int, max_tokens: int, max_model_len: int) -> None:
    """Reject a tokenized prompt that cannot fit the configured context."""
    if prompt_tokens + max_tokens > max_model_len:
        raise ProbeError(
            f"tokenized prompt {prompt_tokens} + --max-tokens {max_tokens} "
            f"exceeds --max-model-len {max_model_len}; the request would be "
            "rejected with HTTP 400"
        )


def evaluate(
    cold: float,
    warm: float,
    queries_delta: float,
    hits_delta: float,
    ratio: float | None,
) -> Verdict:
    """Judge the paired requests from their counter deltas."""
    if queries_delta <= 0:
        raise ProbeError(
            "prefix cache query counter did not advance across the two requests"
        )
    if hits_delta < 0:
        raise ProbeError(f"prefix cache hit counter went backwards ({hits_delta:+.0f})")
    if hits_delta > queries_delta:
        raise ProbeError(
            f"prefix cache deltas are inconsistent (hits +{hits_delta:.0f} > "
            f"queries +{queries_delta:.0f})"
        )
    # Both requests carry the same prompt, so the warm request's query share is
    # half the delta; its hits are all of the delta.
    warm_queries = queries_delta / 2.0
    hit_rate = hits_delta / warm_queries
    if hit_rate > 1.0:
        raise ProbeError(
            f"counter deltas imply a {hit_rate:.1%} warm hit rate; hits cannot "
            "be attributed to the warm request"
        )
    latency_met = None if ratio is None else warm <= ratio * cold
    passed = hit_rate > _MIN_WARM_HIT_RATE and latency_met is not False
    return Verdict(passed, hits_delta, warm_queries, hit_rate, latency_met)


def _ttft_line(cold: float, warm: float, passed: bool) -> str:
    return (
        f"cold TTFT {cold:.6f} s, warm TTFT {warm:.6f} s, "
        f"speedup {cold / max(warm, 1e-6):.2f}x; probe {'PASS' if passed else 'FAIL'}"
    )


def _latency_target(ratio: float | None, met: bool | None) -> str:
    if met is None or ratio is None:
        return "none (TTFT reported as context)"
    return f"warm <= {ratio:.2f}x cold ({'MET' if met else 'MISSED'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    # --words sizes the prompt; only the server knows its tokenized length.
    ap.add_argument("--words", type=int, default=1800)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    # None: report TTFT as context and gate only on the measured cache hits.
    # An explicit ratio keeps its strict warm <= ratio * cold target.
    ap.add_argument("--ratio", type=float, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(1234)
    vocab = [
        "kernel",
        "wavefront",
        "register",
        "cache",
        "latency",
        "memory",
        "shader",
        "vector",
        "scalar",
        "buffer",
        "texture",
        "queue",
        "fence",
        "atomic",
        "barrier",
        "launch",
        "stream",
        "graph",
        "tensor",
        "matrix",
    ]
    # Random nonce up front so a previous run's cached prefix can't hit.
    nonce = f"Session {time.time_ns()}.\n"
    filler = " ".join(rng.choice(vocab) for _ in range(args.words))
    prompt = nonce + filler + "\n\nSummarize the text above in one sentence:"

    try:
        prompt_tokens = tokenize_count(args.url, args.model, prompt)
    except ProbeError as exc:
        print(f"prompt length unknown: {exc}")
        return 1
    try:
        check_prompt(prompt_tokens, args.max_tokens, args.max_model_len)
    except ProbeError as exc:
        print(f"prompt rejected: {exc}")
        return 1

    try:
        q0, h0 = prefix_counters(args.url)
    except ProbeError as exc:
        print(f"prefix cache: unknown before the probe ({exc}) -> FAIL")
        return 1

    try:
        cold, cold_text = ttft(args.url, args.model, prompt, args.max_tokens)
        warm, warm_text = ttft(args.url, args.model, prompt, args.max_tokens)
    except (OSError, ValueError, ProbeError) as exc:
        print(f"completion evidence unavailable: {exc} -> FAIL")
        return 1

    try:
        q1, h1 = prefix_counters(args.url)
        verdict = evaluate(cold, warm, q1 - q0, h1 - h0, args.ratio)
    except ProbeError as exc:
        print(_ttft_line(cold, warm, False))
        print(f"prefix cache: unknown after the probe ({exc})")
        return 1

    target = _latency_target(args.ratio, verdict.latency_met)
    print(_ttft_line(cold, warm, verdict.passed))
    print(
        f"prefix cache: prompt {prompt_tokens} tokens, "
        f"queries +{q1 - q0:.0f}, hits +{h1 - h0:.0f}, "
        f"warm hit rate {verdict.hit_rate:.1%} "
        f"(required > {_MIN_WARM_HIT_RATE:.0%}); latency target: {target}"
    )
    print(f"cold text: {cold_text[:120]!r}")
    print(f"warm text: {warm_text[:120]!r}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
