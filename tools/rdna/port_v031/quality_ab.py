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
        --out base.json [--passages 6] [--passage-chars 12000] [--gen-tokens 96] \
        [--key-file /path/to/api.key] [--require-cold-cache]
    python quality_ab.py compare base.json variant.json [--max-ppl-rel 0.02]

Passages come from ``--files`` (default: the markdown under ``docs/`` of this
tree), concatenated and cut into equal slices, so both runs see identical text.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import urllib.request
import uuid
from pathlib import Path

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def post(url: str, body: dict, *, api_key: str | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key is not None:
        req.add_header("Authorization", "Bearer " + api_key)
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


def _cached_tokens(response: dict, required: bool) -> int | None:
    details = response["usage"].get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is not None and (type(cached) is not int or cached < 0):
        raise ValueError("invalid cached-token usage")
    if required and cached != 0:
        raise ValueError("cold-cache collection requires explicit zero cached tokens")
    return cached


def _scores(tokens, logprobs, count):
    if type(count) is not int or count < 2:
        raise ValueError("prompt must contain at least two tokens")
    if len(tokens) != count or len(logprobs) != count:
        raise ValueError("prompt tokens and score positions must match reported usage")
    scored = logprobs[1:]
    if any(type(lp) not in (int, float) or not math.isfinite(lp) for lp in scored):
        raise ValueError("every noninitial prompt position requires a finite score")
    return scored


def _bundle_scores(bundle):
    passages = bundle["passages"]
    if not passages or bundle.get("complete") is not True:
        raise ValueError("quality collection is incomplete")
    nll_sum, count = 0.0, 0
    salts = [salt for passage in passages for salt in passage["cache_salts"].values()]
    if (len(salts) != 2 * len(passages) or len(set(salts)) != len(salts)
            or any(not isinstance(salt, str) or not salt for salt in salts)):
        raise ValueError("every quality request requires a distinct cache salt")
    for passage in passages:
        text = passage["passage"]
        if hashlib.sha256(text.encode()).hexdigest() != passage["passage_sha256"]:
            raise ValueError("quality passage hash differs")
        scored = _scores(
            passage["tokens"], passage["token_logprobs"], passage["prompt_tokens"]
        )
        _cached_tokens(passage["echo_response"], bundle["require_cold_cache"])
        _cached_tokens(passage["greedy_response"], bundle["require_cold_cache"])
        nll_sum -= math.fsum(scored)
        count += len(scored)
    return math.exp(nll_sum / count), count


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
    api_key = Path(args.key_file).read_text().strip() if args.key_file else None
    if api_key == "":
        raise ValueError("API key file is empty")
    result = {"model": args.model, "passages": [], "complete": False,
              "require_cold_cache": args.require_cold_cache}

    def save():
        with open(args.out, "w") as handle:
            json.dump(result, handle, allow_nan=False)

    save()
    for index, passage in enumerate(passages):
        salts = {"echo": uuid.uuid4().hex, "greedy": uuid.uuid4().hex}
        entry = {"passage": passage,
                 "passage_sha256": hashlib.sha256(passage.encode()).hexdigest(),
                 "cache_salts": salts}
        result["passages"].append(entry)
        echo = post(
            url,
            {"model": args.model, "prompt": passage, "max_tokens": 1,
             "temperature": 0, "echo": True, "logprobs": 1,
             "cache_salt": salts["echo"]},
            api_key=api_key,
        )
        entry["echo_response"] = echo
        save()
        entry["echo_cached_tokens"] = _cached_tokens(echo, args.require_cold_cache)
        choice = echo["choices"][0]
        n_prompt = echo["usage"]["prompt_tokens"]
        lps = choice["logprobs"]["token_logprobs"][:n_prompt]
        tokens = choice["logprobs"]["tokens"][:n_prompt]
        scored = _scores(tokens, lps, n_prompt)
        nll = -math.fsum(scored) / len(scored)
        entry.update(prompt_tokens=n_prompt, scored_tokens=len(scored), nll=nll,
                     ppl=math.exp(nll), token_logprobs=lps, tokens=tokens)
        save()
        greedy_response = post(
            url,
            {"model": args.model, "prompt": passage, "max_tokens": args.gen_tokens,
             "temperature": 0, "logprobs": 1, "cache_salt": salts["greedy"]},
            api_key=api_key,
        )
        entry["greedy_response"] = greedy_response
        save()
        entry["greedy_cached_tokens"] = _cached_tokens(
            greedy_response, args.require_cold_cache
        )
        greedy = greedy_response["choices"][0]
        entry.update(
            greedy_text=greedy["text"], greedy_tokens=greedy["logprobs"]["tokens"],
            greedy_logprobs=greedy["logprobs"]["token_logprobs"],
        )
        save()
        print(
            f"passage {index}: {n_prompt} tokens  PPL {entry['ppl']:.3f}  "
            f"greedy {len(entry['greedy_tokens'])} tokens", flush=True,
        )
    result["complete"] = True
    result["ppl"], result["scored_tokens"] = _bundle_scores(result)
    save()
    print(f"PPL_LONG {result['ppl']:.4f} over {result['scored_tokens']} tokens -> {args.out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    with open(args.a) as handle:
        a = json.load(handle)
    with open(args.b) as handle:
        b = json.load(handle)
    try:
        ppl_a, scores_a = _bundle_scores(a)
        ppl_b, scores_b = _bundle_scores(b)
        if len(a["passages"]) != len(b["passages"]) or scores_a != scores_b:
            raise ValueError("passage/scored-token counts differ")
        salts_a = {salt for passage in a["passages"]
                   for salt in passage["cache_salts"].values()}
        salts_b = {salt for passage in b["passages"]
                   for salt in passage["cache_salts"].values()}
        if not salts_a.isdisjoint(salts_b):
            raise ValueError("quality arms reuse cache salts")
        for pa, pb in zip(a["passages"], b["passages"]):
            if (pa["passage_sha256"] != pb["passage_sha256"]
                    or pa["prompt_tokens"] != pb["prompt_tokens"]
                    or pa["tokens"] != pb["tokens"]):
                raise ValueError("passage bytes/tokenization/usage differ")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        print(f"INCOMPARABLE: {error}")
        return 1
    sum_abs, count, exact, divergence = 0.0, 0, 0, []
    for index, (pa, pb) in enumerate(zip(a["passages"], b["passages"])):
        deltas = [
            abs(x - y)
            for x, y in zip(pa["token_logprobs"][1:], pb["token_logprobs"][1:])
        ]
        mean_abs = math.fsum(deltas) / len(deltas)
        big = sum(d > 0.1 for d in deltas) / len(deltas)
        ga, gb = pa["greedy_tokens"], pb["greedy_tokens"]
        first = next(
            (i for i, (x, y) in enumerate(zip(ga, gb)) if x != y),
            min(len(ga), len(gb)),
        )
        same = ga == gb
        exact += same
        divergence.append(first)
        sum_abs += math.fsum(deltas)
        count += len(deltas)
        print(
            f"passage {index}: PPL {pa['ppl']:.3f} -> {pb['ppl']:.3f} "
            f"({pb['ppl'] / pa['ppl'] - 1:+.2%})  mean|dlogp| {mean_abs:.4f}  "
            f"|dlogp|>0.1 {big:.2%}  greedy first divergence "
            f"{'none' if same else first}/{len(ga)}"
        )
    rel = ppl_b / ppl_a - 1
    verdict = "PASS" if abs(rel) <= args.max_ppl_rel else "FAIL"
    print(
        f"TOTAL PPL {ppl_a:.4f} -> {ppl_b:.4f} ({rel:+.3%})  "
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
    c.add_argument("--key-file", help="API bearer-key file; contents are never recorded")
    c.add_argument("--require-cold-cache", action="store_true",
                   help="require explicit zero cached-token usage for every request")
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
