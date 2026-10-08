#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Perplexity + garbage probe for degraded (e.g. pruned) checkpoints.

    python ppl_probe.py --url http://127.0.0.1:18310/v1/completions \
        --model dsv4-flash [--max-ppl 60]

1. Perplexity: sends a fixed English paragraph with ``echo: true,
   max_tokens: 1, logprobs: 1`` and reports the mean NLL / perplexity over
   the prompt tokens. A broken attention or MoE path is near uniform over
   the vocabulary (perplexity in the hundreds to thousands); a working model,
   even a heavily pruned one, lands in the tens.
2. Garbage check: greedy continuations of a few English prompts fail when
   empty or mostly non-Latin script (token salad). Greedy loops are common
   on pruned checkpoints and are reported as WARN only. Exact answers are
   not required.

Prints ``PPL <value>`` and exits 1 when perplexity exceeds ``--max-ppl`` or
any continuation is flagged. Stdlib only.
"""

import argparse
import json
import math
import sys
import urllib.request

PARAGRAPH = (
    "The river town grew up around a wooden bridge that farmers used to "
    "bring grain to the market on the eastern bank. In the early years the "
    "market opened only on Saturdays, and most families walked for several "
    "hours to reach it. As the population increased, merchants built stone "
    "warehouses along the water, and a small school was founded next to the "
    "church. The first newspaper appeared about forty years later. It "
    "reported prices, weather, the arrival of boats, and long letters from "
    "readers who argued about whether the old bridge should be replaced. "
    "After a severe flood destroyed part of the town, the council finally "
    "agreed to build a new bridge made of iron. Construction took three "
    "years and employed hundreds of workers, many of whom later settled in "
    "the area with their families. The new bridge allowed heavier wagons to "
    "cross, and trade with the neighboring valleys expanded quickly. By the "
    "end of the century the town had a railway station, a hospital, two "
    "banks, and a public library that was open every day of the week. "
    "Today the historic center is protected, and visitors can still see the "
    "old warehouses, which now contain shops, restaurants, and a museum "
    "that describes how the town changed over time. Every summer a festival "
    "celebrates the opening of the iron bridge with music, food, and a "
    "parade of decorated boats on the river."
)

PROMPTS = [
    "The history of the printing press begins",
    "Here is a short recipe for vegetable soup:",
    "In mathematics, a prime number is",
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def perplexity(url: str, model: str) -> tuple[float, float, int]:
    out = post(
        url,
        {
            "model": model,
            "prompt": PARAGRAPH,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        },
    )
    lps = out["choices"][0]["logprobs"]["token_logprobs"]
    # The first prompt token has no context (None); also drop the trailing
    # generated token so only prompt tokens count.
    n_prompt = out["usage"]["prompt_tokens"]
    vals = [lp for lp in lps[1:n_prompt] if lp is not None]
    if not vals:
        raise RuntimeError("no prompt logprobs returned")
    nll = -sum(vals) / len(vals)
    return nll, math.exp(min(nll, 50.0)), len(vals)


def garbage_reason(text: str) -> str | None:
    """Token salad or empty output: the signature of a broken kernel."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return "no letters"
    non_latin = sum(1 for c in letters if ord(c) > 0x24F)
    if non_latin / len(letters) > 0.2:
        return f"non-Latin script {non_latin}/{len(letters)}"
    return None


def repeat_warning(text: str) -> str | None:
    """Greedy loops: common on pruned checkpoints, reported but not failed."""
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if len(words) >= 8:
        top = max(words.count(w) for w in set(words))
        if top / len(words) > 0.4:
            return "one word dominates"
    n = 24
    if len(text) >= 3 * n:
        grams = [text[i : i + n] for i in range(len(text) - n)]
        if max(grams.count(g) for g in set(grams)) >= 3:
            return f"repeated {n}-char span"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-ppl", type=float, default=60.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    ok = True
    nll, ppl, n = perplexity(args.url, args.model)
    verdict = "PASS" if ppl <= args.max_ppl else "FAIL"
    ok &= ppl <= args.max_ppl
    print(f"PPL {ppl:.2f} (mean NLL {nll:.3f} over {n} tokens) {verdict}")

    for prompt in PROMPTS:
        out = post(
            args.url,
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
        )
        text = out["choices"][0]["text"]
        why = garbage_reason(text)
        warn = repeat_warning(text)
        ok &= why is None
        tag = "GARBAGE" if why else ("WARN" if warn else "ok")
        note = why or warn
        print(f"{tag:7s} {prompt!r} -> {text!r}" + (f"  [{note}]" if note else ""))
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
