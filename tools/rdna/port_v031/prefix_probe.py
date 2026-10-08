#!/usr/bin/env python3
"""Send one long prompt twice and compare TTFT (prefix-cache hit check).

    python prefix_probe.py --url http://127.0.0.1:18120/v1/completions \
        --model flash-next --words 12000

Prints cold/warm TTFT and exits 1 when the warm request is not clearly
faster (warm > ratio * cold). Stdlib only.
"""

import argparse
import json
import random
import sys
import time
import urllib.request


def ttft(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, str]:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--words", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--ratio", type=float, default=0.5)
    args = ap.parse_args()

    rng = random.Random(1234)
    vocab = [
        "kernel", "wavefront", "register", "cache", "latency", "memory",
        "shader", "vector", "scalar", "buffer", "texture", "queue", "fence",
        "atomic", "barrier", "launch", "stream", "graph", "tensor", "matrix",
    ]
    # Random nonce up front so a previous run's cached prefix can't hit.
    nonce = f"Session {time.time_ns()}.\n"
    filler = " ".join(rng.choice(vocab) for _ in range(args.words))
    prompt = nonce + filler + "\n\nSummarize the text above in one sentence:"

    cold, cold_text = ttft(args.url, args.model, prompt, args.max_tokens)
    warm, warm_text = ttft(args.url, args.model, prompt, args.max_tokens)
    ok = warm <= args.ratio * cold
    print(f"cold TTFT {cold:.2f} s, warm TTFT {warm:.2f} s, "
          f"speedup {cold / max(warm, 1e-6):.1f}x -> {'PASS' if ok else 'FAIL'}")
    print(f"cold text: {cold_text[:120]!r}")
    print(f"warm text: {warm_text[:120]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
