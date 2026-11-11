#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Long-context numerical A/B between two server configurations.

The short PPL probe (``ppl_probe.py``, ~200 prompt tokens) never reaches the
prefill-only kernels that start at 512 tokens per MoE call (W4A8 MoE) or at
2048 tokens per chunk (large-tile paths), so it cannot judge them. This tool
runs the same long passages through a server and records, per passage:

* the teacher-forced token logprobs of the whole prompt (``echo`` + ``logprobs``),
  which are chunked-prefill outputs exactly as served (MAXBAT-sized chunks);
* a greedy continuation (temperature 0) after the passage.

``collect`` writes one JSON file per server; ``compare`` diffs two of them
(PPL, per-token logprob drift, greedy divergence point). Stdlib only.

    python quality_ab.py collect --url http://127.0.0.1:18410 --model <served> \
        --out base.json [--passages 6] [--passage-chars 12000] [--gen-tokens 96]
    python quality_ab.py compare base.json variant.json [--max-ppl-rel 0.02]

Passages come from ``--files`` (default: the markdown under ``docs/`` of this
tree), concatenated and cut into equal slices, so both runs see identical text.
"""

import argparse
import glob
import json
import math
import os
import sys
import urllib.request

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def load_passages(files: list[str], count: int, chars: int) -> list[str]:
    text = ""
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text += handle.read() + "\n\n"
    if len(text) < count * chars:
        raise SystemExit(
            f"need {count * chars} characters of text, the files hold {len(text)}"
        )
    stride = len(text) // count
    return [text[i * stride : i * stride + chars] for i in range(count)]


def collect(args: argparse.Namespace) -> int:
    files = sorted(
        f
        for pattern in args.files or [os.path.join(TREE, "docs", "**", "*.md")]
        for f in glob.glob(pattern, recursive=True)
    )
    passages = load_passages(files, args.passages, args.passage_chars)
    url = args.url.rstrip("/")
    if not url.endswith("/v1/completions"):
        url += "/v1/completions"
    result: dict = {"model": args.model, "passages": []}
    for index, passage in enumerate(passages):
        echo = post(
            url,
            {
                "model": args.model,
                "prompt": passage,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            },
        )
        choice = echo["choices"][0]
        n_prompt = echo["usage"]["prompt_tokens"]
        lps = [lp for lp in choice["logprobs"]["token_logprobs"][:n_prompt]]
        tokens = choice["logprobs"]["tokens"][:n_prompt]
        scored = [lp for lp in lps[1:] if lp is not None]
        nll = -sum(scored) / len(scored)
        greedy = post(
            url,
            {
                "model": args.model,
                "prompt": passage,
                "max_tokens": args.gen_tokens,
                "temperature": 0,
                "logprobs": 1,
            },
        )["choices"][0]
        entry = {
            "prompt_tokens": n_prompt,
            "nll": nll,
            "ppl": math.exp(min(nll, 50.0)),
            "token_logprobs": [None if lp is None else round(lp, 5) for lp in lps],
            "tokens": tokens,
            "greedy_text": greedy["text"],
            "greedy_tokens": greedy["logprobs"]["tokens"],
            "greedy_logprobs": [
                None if lp is None else round(lp, 5)
                for lp in greedy["logprobs"]["token_logprobs"]
            ],
        }
        result["passages"].append(entry)
        print(
            f"passage {index}: {n_prompt} tokens  PPL {entry['ppl']:.3f}  "
            f"greedy {len(entry['greedy_tokens'])} tokens",
            flush=True,
        )
    total = sum(p["nll"] * (p["prompt_tokens"] - 1) for p in result["passages"])
    count = sum(p["prompt_tokens"] - 1 for p in result["passages"])
    result["ppl"] = math.exp(min(total / count, 50.0))
    with open(args.out, "w") as handle:
        json.dump(result, handle)
    print(f"PPL_LONG {result['ppl']:.4f} over {count} tokens -> {args.out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    with open(args.a) as handle:
        a = json.load(handle)
    with open(args.b) as handle:
        b = json.load(handle)
    ok = True
    sum_abs = 0.0
    count = 0
    exact = 0
    divergence: list[int] = []
    for index, (pa, pb) in enumerate(zip(a["passages"], b["passages"])):
        if pa["tokens"] != pb["tokens"]:
            print(f"passage {index}: tokenization differs, cannot compare")
            ok = False
            continue
        deltas = [
            abs(x - y)
            for x, y in zip(pa["token_logprobs"], pb["token_logprobs"])
            if x is not None and y is not None
        ]
        mean_abs = sum(deltas) / len(deltas)
        big = sum(1 for d in deltas if d > 0.1) / len(deltas)
        ga, gb = pa["greedy_tokens"], pb["greedy_tokens"]
        first = next(
            (i for i, (x, y) in enumerate(zip(ga, gb)) if x != y),
            min(len(ga), len(gb)),
        )
        same = ga == gb
        exact += same
        divergence.append(first)
        sum_abs += sum(deltas)
        count += len(deltas)
        print(
            f"passage {index}: PPL {pa['ppl']:.3f} -> {pb['ppl']:.3f} "
            f"({pb['ppl'] / pa['ppl'] - 1:+.2%})  mean|dlogp| {mean_abs:.4f}  "
            f"|dlogp|>0.1 {big:.2%}  greedy first divergence "
            f"{'none' if same else first}/{len(ga)}"
        )
    rel = b["ppl"] / a["ppl"] - 1
    verdict = "PASS" if abs(rel) <= args.max_ppl_rel and ok else "FAIL"
    print(
        f"TOTAL PPL {a['ppl']:.4f} -> {b['ppl']:.4f} ({rel:+.3%})  "
        f"mean|dlogp| {sum_abs / count:.4f}  greedy exact {exact}/"
        f"{len(a['passages'])}  mean first divergence "
        f"{sum(divergence) / len(divergence):.1f} tokens  {verdict}"
    )
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True, help="server base or /v1/completions URL")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--files", nargs="*", help="text files or globs (default docs/)")
    c.add_argument("--passages", type=int, default=6)
    c.add_argument("--passage-chars", type=int, default=12000)
    c.add_argument("--gen-tokens", type=int, default=96)
    c.set_defaults(func=collect)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--max-ppl-rel", type=float, default=0.02)
    d.set_defaults(func=compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
