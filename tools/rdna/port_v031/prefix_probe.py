#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Send one long prompt twice and compare TTFT (prefix-cache hit check).

    python prefix_probe.py --url http://127.0.0.1:18120/v1/completions \
        --model flash-next --words 2600

Prints cold/warm TTFT and exits 1 when the warm request is not clearly
faster (warm > ratio * cold). Stdlib only.

The hit granularity is the mamba state block: the served Flash-Next logs
"Mamba prefix checkpoints: aligned_split=True state_block=2048
scheduler_block=2048 hash_block=2048", so a prompt of at most 2048 tokens
can produce no full reusable block at all. The prompt must cross that
block (the default 2600 words is ~3.7k tokens and stays under the 4096
max_model_len together with --max-tokens). The probe also reads the
server's vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total
counters and prints the deltas, so a hit is confirmed by the metric and not
only by TTFT.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

import regex as re

_HIT_GRANULARITY_TOKENS = 2048


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
    return (first if first is not None else time.monotonic() - t0), "".join(text)


def prefix_counters(url: str) -> tuple[float, float]:
    """Return (queried tokens, hit tokens) from the server's /metrics."""
    metrics_url = re.sub(r"/v1/.*$", "/metrics", url)
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode()
    except OSError:
        return 0.0, 0.0
    counters = {}
    for name in ("queries_total", "hits_total"):
        m = re.search(
            rf"^vllm:prefix_cache_{name}\{{[^}}]*\}}\s+([0-9.eE+]+)$",
            text,
            re.MULTILINE,
        )
        counters[name] = float(m.group(1)) if m else 0.0
    return counters["queries_total"], counters["hits_total"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    # The prompt plus max_tokens must fit the served model's
    # max_model_len (4096 on the Flash-Next recipe); 12000 words
    # made every request fail with HTTP 400 on both arms. The prompt also
    # has to cross the 2048-token mamba state block, else the prefix cache
    # cannot produce a reusable block: 1200 words (~1.7k tokens) sat below
    # it and every warm request reported a 1.0x "speedup".
    ap.add_argument("--words", type=int, default=1800)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--ratio", type=float, default=0.5)
    args = ap.parse_args()

    if args.words + args.max_tokens + 64 > args.max_model_len:
        print(
            f"--words {args.words} plus --max-tokens {args.max_tokens} does not "
            f"fit --max-model-len {args.max_model_len}; the request would be "
            "rejected with HTTP 400."
        )
        return 1
    if args.words * 2 < _HIT_GRANULARITY_TOKENS:
        print(
            f"--words {args.words} is below the {_HIT_GRANULARITY_TOKENS}-token "
            "mamba state block; the prefix cache has no full block to reuse."
        )
        return 1

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

    q0, h0 = prefix_counters(args.url)
    cold, cold_text = ttft(args.url, args.model, prompt, args.max_tokens)
    warm, warm_text = ttft(args.url, args.model, prompt, args.max_tokens)
    q1, h1 = prefix_counters(args.url)
    # The deltas cover both requests, so the warm prompt's own token count is
    # half the queried delta.
    hit_rate = (h1 - h0) / max((q1 - q0) / 2.0, 1.0)
    ok = warm <= args.ratio * cold and hit_rate > 0.5
    print(
        f"cold TTFT {cold:.2f} s, warm TTFT {warm:.2f} s, "
        f"speedup {cold / max(warm, 1e-6):.1f}x -> {'PASS' if ok else 'FAIL'}"
    )
    print(
        f"prefix cache: queries +{q1 - q0:.0f}, hits +{h1 - h0:.0f}, "
        f"warm hit rate {hit_rate:.1%} "
        f"(hit granularity {_HIT_GRANULARITY_TOKENS} tokens)"
    )
    print(f"cold text: {cold_text[:120]!r}")
    print(f"warm text: {warm_text[:120]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
