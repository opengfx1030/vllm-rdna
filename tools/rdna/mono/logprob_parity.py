#!/usr/bin/env python3
"""Greedy decode + top-5 logprobs for a fixed prompt set, saved to JSON; then
compare two runs (switch off vs on on the same build).

  python3 logprob_parity.py dump --url http://127.0.0.1:18790 --model flash-next --out off.json
  python3 logprob_parity.py compare off.json on.json

The prompts are short reasoning / arithmetic / factual items. Decode is greedy
(temperature 0), 64 tokens, 1 sequence at a time and then all at once (c=8),
so both the width-3 and the wider FULL decode graphs are exercised.
"""

import argparse
import concurrent.futures as cf
import json
import math
import sys
import urllib.request

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's think step by step.",
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Let's think step by step.",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer: Let's think step by step.",
    "Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nAnswer: Let's think step by step.",
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Translate to German: The weather is nice today, so we are going to the park.",
    "List the first ten prime numbers:",
]


def complete(url, model, prompt, max_tokens):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "logprobs": 5}).encode()
    req = urllib.request.Request(url + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ch = json.load(r)["choices"][0]
    lp = ch["logprobs"]
    return {"text": ch["text"], "tokens": lp["tokens"],
            "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"]}


def dump(a):
    out = {"solo": [], "c8": []}
    for p in PROMPTS:
        out["solo"].append(complete(a.url, a.model, p, a.max_tokens))
    with cf.ThreadPoolExecutor(len(PROMPTS)) as ex:
        out["c8"] = list(ex.map(lambda p: complete(a.url, a.model, p, a.max_tokens), PROMPTS))
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


def compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    worst = 0.0
    lines = []
    total_tok = same_prefix_tok = 0
    for mode in ("solo", "c8"):
        for i, (x, y) in enumerate(zip(A[mode], B[mode])):
            n = min(len(x["tokens"]), len(y["tokens"]))
            div = next((k for k in range(n) if x["tokens"][k] != y["tokens"][k]), n)
            # logprob of the same token while the two sequences agree
            d = [abs(x["token_logprobs"][k] - y["token_logprobs"][k]) for k in range(div)
                 if x["token_logprobs"][k] is not None and y["token_logprobs"][k] is not None]
            md = max(d) if d else 0.0
            worst = max(worst, md)
            total_tok += n
            same_prefix_tok += div
            near_tie = ""
            if div < n:
                top = x["top"][div]
                vals = sorted(top.values(), reverse=True)
                gap = vals[0] - vals[1] if len(vals) > 1 else math.inf
                near_tie = f" (diverges at {div}; top-2 logprob gap there {gap:.3f})"
            lines.append(f"{mode} p{i}: same tokens {div}/{n}, max |dlogprob| {md:.4f}{near_tie}")
    print("\n".join(lines))
    print(f"identical-prefix tokens {same_prefix_tok}/{total_tok}, worst |dlogprob| on shared prefix {worst:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--url", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-tokens", type=int, default=64)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    (dump if a.cmd == "dump" else compare)(a)


if __name__ == "__main__":
    sys.exit(main())
