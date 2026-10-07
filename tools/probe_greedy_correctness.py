#!/usr/bin/env python3
"""Strict greedy correctness probe for gfx1030 serve.

First-token-only matches are FAIL. FPP16 printed Paris/2 then garbage
and a naive `\"Paris\" in text` check called that PASS.

Exit 0 only if every prompt stays coherent for the full completion.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request


GARBAGE_SUBSTR = (
    "duct",
    "\ufffd",
    "{{{",
    "!!!!",
    "The The The",
    "|[/",
    "… |[",
)
ELLIPSIS_RE = re.compile(r"(\.\.\.){3,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")


def is_garbage(text: str) -> str | None:
    low = text.lower()
    for s in GARBAGE_SUBSTR:
        if s.lower() in low or s in text:
            return f"substring {s!r}"
    if ELLIPSIS_RE.search(text):
        return "repeated ellipsis"
    if REPEAT_RE.search(text):
        return "run of repeated chars"
    if text.count("!") > 6:
        return "bang storm"
    return None


def check_paris(text: str) -> str | None:
    head = text.lstrip()[:32]
    if "Paris" not in head:
        return "Paris not in first 32 chars"
    return is_garbage(text)


def check_one_plus_one(text: str) -> str | None:
    head = text.lstrip()
    if not head.startswith("2"):
        return "does not start with 2"
    return is_garbage(text)


def check_germany(text: str) -> str | None:
    head = text.lstrip()[:40]
    if "Berlin" not in head:
        return "Berlin not in first 40 chars"
    return is_garbage(text)


CASES = (
    ("The capital of France is", check_paris),
    ("1+1=", check_one_plus_one),
    ("The capital of Germany is", check_germany),
)


def complete(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "skip_special_tokens": True,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["text"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18094/v1/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()
    failed = 0
    for prompt, checker in CASES:
        try:
            text = complete(args.url, args.model, prompt, args.max_tokens)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"FAIL  {prompt!r}: request error {e}")
            failed += 1
            continue
        reason = checker(text)
        status = "FAIL" if reason else "PASS"
        if reason:
            failed += 1
        print(f"{status}  prompt={prompt!r}")
        print(f"      reply={text!r}")
        if reason:
            print(f"      why={reason}")
    print("RESULT", "FAIL" if failed else "PASS", f"({len(CASES) - failed}/{len(CASES)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
