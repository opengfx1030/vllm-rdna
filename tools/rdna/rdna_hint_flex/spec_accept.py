#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mean speculative-decoding acceptance of a running vLLM server on natural text.

Sends greedy chat requests (a fixed set of varied prompts, EOS honoured) at a
fixed concurrency and reads the server's own counters from /metrics before and
after: ``vllm:spec_decode_num_drafts``, ``..._num_draft_tokens``,
``..._num_accepted_tokens`` and ``..._num_accepted_tokens_per_pos`` (what the
"SpecDecoding metrics" log line is built from). Mean acceptance length is
``1 + accepted / drafts``, the same definition as that log line.

Random-token prompts (``vllm bench serve --dataset-name random``) say nothing
about acceptance, so ``serve_validate.sh`` can run this as a LOAD_HOOK:

    LOAD_HOOK=<a script that execs this file> bash serve_validate.sh MTP=2

Usage: spec_accept.py --port 18211 --model q27hi2 [--conc 1] [--max-tokens 256]
Exit status 1 when the server exposes no speculative counters or no request
completed.
"""

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Explain how a transformer language model generates text, step by step.",
    "Write a Python function that merges two sorted lists and state its "
    "time complexity.",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "What is the difference between TCP and UDP? Give an example use of each.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "A train leaves at 3 pm at 80 km/h and a second train leaves the same "
    "station at 4 pm at 100 km/h. When does the second catch the first? "
    "Show the steps.",
    "Describe the water cycle for a ten-year-old.",
    "Write a SQL query that returns the five customers with the largest total "
    "order value, and explain each clause.",
    "Give a concise history of the printing press.",
    "Explain the rules of chess to a complete beginner.",
    "Compare the architectures of a CPU and a GPU in a table and a paragraph.",
    "Translate 'The quick brown fox jumps over the lazy dog' into French, "
    "German and Spanish and comment on the grammar.",
]


def counters(port: int) -> tuple[dict[str, float], dict[int, float]]:
    url = f"http://127.0.0.1:{port}/metrics"
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    tot = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    pos: dict[int, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_num_"):
            continue
        head, val = line.rsplit(" ", 1)
        name = head.split("{", 1)[0]
        if name == "vllm:spec_decode_num_drafts_total":
            tot["drafts"] += float(val)
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            tot["draft_tokens"] += float(val)
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            tot["accepted"] += float(val)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            p = int(head.split('position="', 1)[1].split('"', 1)[0])
            pos[p] = pos.get(p, 0.0) + float(val)
    return tot, pos


def complete(port: int, model: str, prompt: str, max_tokens: int) -> int:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)["usage"]["completion_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=1, help="passes over the prompts")
    a = ap.parse_args()

    before, before_pos = counters(a.port)
    prompts = PROMPTS * a.rounds
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.conc) as pool:
        futs = [
            pool.submit(complete, a.port, a.model, p, a.max_tokens) for p in prompts
        ]
        toks = [f.result() for f in futs]
    dt = time.time() - t0
    after, after_pos = counters(a.port)

    d = {k: after[k] - before[k] for k in after}
    print(
        f"spec_accept: conc={a.conc} requests={len(prompts)} "
        f"completion_tokens={sum(toks)} in {dt:.1f} s "
        f"({sum(toks) / dt:.1f} tok/s aggregate, not a benchmark)"
    )
    if not toks or d["drafts"] <= 0:
        print("spec_accept: no speculative-decoding counters moved: FAIL")
        return 1
    mean_len = 1 + d["accepted"] / d["drafts"]
    rate = d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else float("nan")
    per_pos = [
        (after_pos[p] - before_pos.get(p, 0.0)) / d["drafts"] for p in sorted(after_pos)
    ]
    print(
        f"spec_accept: drafts={d['drafts']:.0f} draft_tokens={d['draft_tokens']:.0f} "
        f"accepted={d['accepted']:.0f}"
    )
    print(
        f"spec_accept: mean acceptance length {mean_len:.3f}, draft acceptance "
        f"rate {rate * 100:.1f}%, per position {[round(x, 3) for x in per_pos]}"
    )
    print("spec_accept: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
