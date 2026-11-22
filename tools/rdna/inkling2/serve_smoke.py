#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test of a running Inkling-2-Mini OpenAI server (the ``inkling2-mini-tp*``
recipes).

    bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1 MODEL=... VENV=...
    python tools/rdna/inkling2/serve_smoke.py --url http://127.0.0.1:18221

Plain HTTP (``urllib``), greedy decoding, no dependency but Pillow and pybase64 for
the image:

* ``models``: the served name and ``max_model_len`` the server reports;
* ``text``: a chat request, the answer has to name Paris (the model thinks first, so
  the word shows up in the first tokens);
* ``image``: the red disc on a blue background of ``vision_engine_check.py`` sent as a
  base64 data URL; the answer has to name red and blue;
* ``prefix``: one prompt of about ``--prompt-tokens`` tokens twice; the second request
  is a prefix-cache hit and has to be faster
  (``usage.prompt_tokens_details.cached_tokens`` when the server reports it);
* ``stream``: server-sent events, the number of tokens and tokens/s;
* ``concurrent``: ``--concurrency`` requests at once, all of them answered.

The script prints one ``RESULT`` line: ``PASS`` when every check holds.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pybase64 as base64


def call(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(base: str, model: str, content, tokens: int, **extra):
    t0 = time.time()
    out = call(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": tokens,
            "temperature": 0.0,
            **extra,
        },
    )
    return out, time.time() - t0


def disc_data_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 336), (20, 40, 200))
    ImageDraw.Draw(image).ellipse((130, 70, 330, 270), fill=(220, 30, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def text_of(out: dict) -> str:
    msg = out["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18221")
    ap.add_argument("--model", default="inkling2-mini")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-image", action="store_true", help="the server has no vision")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    ok = True

    models = call(f"{base}/v1/models")["data"]
    print(
        "[models]",
        [(m["id"], m.get("max_model_len")) for m in models],
        flush=True,
    )

    out, dt = chat(base, args.model, "What is the capital of France?", args.tokens)
    text = text_of(out)
    good = "paris" in text.lower()
    ok &= good
    print(f"[text] {dt:.2f}s ok={good}: {text[:160]!r}", flush=True)

    if not args.no_image:
        content = [
            {"type": "image_url", "image_url": {"url": disc_data_url()}},
            {"type": "text", "text": "Describe the image in one sentence."},
        ]
        out, dt = chat(base, args.model, content, args.tokens)
        text = text_of(out)
        good = "red" in text.lower() and "blue" in text.lower()
        ok &= good
        print(f"[image] {dt:.2f}s ok={good}: {text[:160]!r}", flush=True)

    facts = "\n".join(
        f"Fact {i}: the number {i * 7 % 1000} is on shelf {i}."
        for i in range(1, 1 + args.prompt_tokens // 14)
    )
    prompt = facts + "\nHow many facts were listed? Answer briefly."
    first, t_cold = chat(base, args.model, prompt, 8)
    second, t_warm = chat(base, args.model, prompt, 8)
    usage = second["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    ok &= t_warm < t_cold
    print(
        f"[prefix] {usage['prompt_tokens']} prompt tokens: cold {t_cold:.2f}s, warm "
        f"{t_warm:.2f}s, cached_tokens {cached}, same answer "
        f"{text_of(first) == text_of(second)}",
        flush=True,
    )

    t0 = time.time()
    n = 0
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Count from 1 to 40."}],
                "max_tokens": args.tokens,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_first = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk["choices"] and chunk["choices"][0]["delta"]:
                    n += 1
                    t_first = t_first or time.time() - t0
    dt = time.time() - t0
    good = n > 8
    ok &= good
    print(
        f"[stream] {n} chunks in {dt:.2f}s, first after {t_first or 0:.2f}s, "
        f"{n / max(dt - (t_first or 0), 1e-9):.1f} chunks/s ok={good}",
        flush=True,
    )

    def one(i: int):
        return chat(base, args.model, f"Write the number {i} in words.", args.tokens)

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, range(args.concurrency)))
    toks = sum(r[0]["usage"]["completion_tokens"] for r in results)
    dt = time.time() - t0
    good = all(
        r[0]["choices"][0]["finish_reason"] in ("stop", "length") for r in results
    )
    ok &= good
    print(
        f"[concurrent] {args.concurrency} requests, {toks} tokens in {dt:.2f}s "
        f"({toks / dt:.1f} tok/s) ok={good}",
        flush=True,
    )
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
