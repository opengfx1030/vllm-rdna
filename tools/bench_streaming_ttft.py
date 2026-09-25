#!/usr/bin/env python3
"""Streaming TTFT / prefill / decode bench against a live vLLM serve.

Unique prefixes (unique bytes at the start of every prompt) so prefix-cache
cannot collapse concurrent prefills. Greedy (temperature=0) + ignore_eos.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

GARBAGE_SUBSTR = (
    "\ufffd",
    "{{{",
    "!!!!",
    "The The The",
    "|[/",
    "… |[",
)
ELLIPSIS_RE = re.compile(r"(\.\.\.){3,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")

CASES = (
    ("The capital of France is", "Paris", 32, "city"),
    ("1+1=", "2", 0, "starts"),
    ("The capital of Germany is", "Berlin", 40, "city"),
    ("The capital of Italy is", "Rome", 40, "city"),
    ("The capital of Spain is", "Madrid", 40, "city"),
    ("The capital of Japan is", "Tokyo", 40, "city"),
    ("Canada's capital is", "Ottawa", 40, "city"),
    ("The capital of the United Kingdom is", "London", 48, "city"),
)


def is_garbage(text: str) -> str | None:
    # Full-completion bar: glued "Parisduct" must fail, not only word-boundary.
    if "duct" in text.lower():
        return "substring 'duct'"
    for s in GARBAGE_SUBSTR:
        if s.lower() in text.lower() or s in text:
            return f"substring {s!r}"
    if ELLIPSIS_RE.search(text):
        return "repeated ellipsis"
    if REPEAT_RE.search(text):
        return "run of repeated chars"
    if text.count("!") > 6:
        return "bang storm"
    return None


def check_out(text: str, expect: str, window: int, kind: str) -> str:
    if not text:
        return "FAIL empty"
    g = is_garbage(text)
    if g:
        return f"FAIL garbage {g} (head={text.lstrip()[:48]!r})"
    head = text.lstrip()
    if kind == "starts":
        if not head.startswith(expect):
            return f"FAIL does not start with {expect!r} (head={head[:48]!r})"
    else:
        if expect not in head[:window]:
            return f"FAIL {expect!r} not in first {window} (head={head[:48]!r})"
    return "PASS"


def chunk_text(ch: dict) -> str:
    if ch.get("text"):
        return ch["text"]
    delta = ch.get("delta") or {}
    return delta.get("content") or delta.get("text") or ""


def build_prompt(tok, n_tokens: int, suffix: str, req_id: int, tag: str = "") -> str:
    # Unique bytes FIRST so prefix cache cannot share across concurrent slots
    # or across bench cells (tag).
    unique = f"Bench {tag} request {req_id} unique-prefix {suffix!r}. "
    filler = f"The river {tag}-{req_id} runs through the quiet valley toward the sea. "
    suffix_ids = tok.encode(unique + suffix, add_special_tokens=False)
    budget = max(n_tokens - len(suffix_ids), 8)
    text = filler * ((budget // 8) + 16)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < budget:
        ids = (ids * ((budget // max(len(ids), 1)) + 1))[:budget]
    else:
        ids = ids[:budget]
    # Unique bytes MUST lead the prompt so prefix-cache hashing cannot
    # share the filler prefix across concurrent slots (the comment above
    # said FIRST; putting unique last made 16k c=4 share "The river {tag}-").
    return unique + tok.decode(ids) + suffix


def one_stream(url, model, prompt, max_tokens, req_id, tok, expect, window, kind):
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "skip_special_tokens": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    parts = []
    n_out = 0
    n_prompt = 0
    err = None
    try:
        with urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or {}
                if usage.get("completion_tokens"):
                    n_out = int(usage["completion_tokens"])
                if usage.get("prompt_tokens"):
                    n_prompt = int(usage["prompt_tokens"])
                chs = obj.get("choices") or [{}]
                piece = chunk_text(chs[0] if chs else {})
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    except (URLError, HTTPError, TimeoutError, OSError) as e:
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    out = "".join(parts)
    if n_out <= 0 and out:
        n_out = len(tok.encode(out, add_special_tokens=False))
    verdict = check_out(out, expect, window, kind)
    if not out and err:
        verdict = f"FAIL empty ({err})"
    return {
        "id": req_id,
        "ttft_s": ttft if ttft is not None else wall,
        "wall_s": wall,
        "n_out": n_out,
        "n_prompt": n_prompt,
        "head": out[:160],
        "verdict": verdict,
        "err": err,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18094/v1/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument(
        "--tag",
        default="",
        help="Unique cell tag prepended so prefix cache cannot hit a prior bench.",
    )
    args = p.parse_args()
    if args.concurrency < 1 or args.concurrency > len(CASES):
        print(f"concurrency must be 1..{len(CASES)}", file=sys.stderr)
        return 2
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = CASES[: args.concurrency]
    prompts = []
    n_ins = []
    for i, (suf, _e, _w, _k) in enumerate(cases):
        pr = build_prompt(tok, args.input_len, suf, i, args.tag)
        prompts.append(pr)
        n_ins.append(len(tok.encode(pr, add_special_tokens=False)))
    print(f"INPUT_LEN_TARGET {args.input_len}", flush=True)
    print(f"OUTPUT_LEN {args.output_len}", flush=True)
    print(f"CONCURRENCY {args.concurrency}", flush=True)
    print(f"INPUT_TOKENS {n_ins}", flush=True)
    t_batch = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(
                one_stream,
                args.url,
                args.model,
                prompts[i],
                args.output_len,
                i,
                tok,
                cases[i][1],
                cases[i][2],
                cases[i][3],
            )
            for i in range(args.concurrency)
        ]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"req{r['id']} ttft={r['ttft_s']:.3f}s wall={r['wall_s']:.3f}s "
                f"n_out={r['n_out']} n_prompt={r['n_prompt']} "
                f"verdict={r['verdict']}",
                flush=True,
            )
            print(f"req{r['id']} HEAD {r['head']!r}", flush=True)
    results.sort(key=lambda x: x["id"])
    ttfts = [r["ttft_s"] for r in results]
    mean_ttft = sum(ttfts) / len(ttfts)
    mean_in = sum(n_ins) / len(n_ins)
    # Per-request prefill tok/s from that request's own prompt length / TTFT.
    prefills = []
    for r, n_in in zip(results, n_ins):
        n_p = r["n_prompt"] or n_in
        if r["ttft_s"] > 0:
            prefills.append(n_p / r["ttft_s"])
    mean_prefill = sum(prefills) / len(prefills) if prefills else float("nan")
    dec = []
    for r in results:
        gen = r["wall_s"] - r["ttft_s"]
        n_dec = max(r["n_out"] - 1, 0)
        if gen > 0.05 and n_dec > 0:
            dec.append(n_dec / gen)
    mean_dec = sum(dec) / len(dec) if dec else float("nan")
    span = time.perf_counter() - t_batch
    agg = sum(r["n_out"] for r in results) / max(span, 1e-6)
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"TTFT_S {mean_ttft:.4f}", flush=True)
    print(f"TTFT_MIN_S {min(ttfts):.4f}", flush=True)
    print(f"TTFT_MAX_S {max(ttfts):.4f}", flush=True)
    print(f"PREFILL_TOK_S {mean_prefill:.2f}", flush=True)
    print(
        f"DECODE_TOK_S_PER_REQ {mean_dec:.2f}" if dec else "DECODE_TOK_S_PER_REQ nan",
        flush=True,
    )
    print(f"AGG_OUTPUT_TOK_S {agg:.2f}", flush=True)
    print(f"OUTPUT_CHECK {n_pass}/{args.concurrency}", flush=True)
    print(f"BATCH_WALL_S {span:.3f}", flush=True)
    return 0 if n_pass == args.concurrency else 1


if __name__ == "__main__":
    raise SystemExit(main())
