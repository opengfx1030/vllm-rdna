#!/usr/bin/env python3
"""Concurrent greedy probe with DIFFERENT prompts (not cached copies).

Full-completion rules: Paris/Berlin/Rome/... in the first window, 1+1=
starts with 2, plus the same garbage bar as probe_greedy_correctness.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

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
WORD_DUCT = re.compile(r"(?<![A-Za-z])duct(?![A-Za-z])", re.I)


def is_garbage(text: str) -> str | None:
    if WORD_DUCT.search(text):
        return "substring 'duct'"
    for s in GARBAGE_SUBSTR:
        if s == "duct":
            continue
        if s.lower() in text.lower() or s in text:
            return f"substring {s!r}"
    if ELLIPSIS_RE.search(text):
        return "repeated ellipsis"
    if REPEAT_RE.search(text):
        return "run of repeated chars"
    if text.count("!") > 6:
        return "bang storm"
    return None


def expect_city(name: str, window: int):
    def check(text: str) -> str | None:
        head = text.lstrip()[:window]
        if name not in head:
            return f"{name} not in first {window} chars"
        return is_garbage(text)

    return check


def expect_starts(ch: str):
    def check(text: str) -> str | None:
        head = text.lstrip()
        if not head.startswith(ch):
            return f"does not start with {ch!r}"
        return is_garbage(text)

    return check


ALL_CASES = (
    ("The capital of France is", expect_city("Paris", 32)),
    ("1+1=", expect_starts("2")),
    ("The capital of Germany is", expect_city("Berlin", 40)),
    ("The capital of Italy is", expect_city("Rome", 40)),
    ("The capital of Spain is", expect_city("Madrid", 40)),
    ("The capital of Japan is", expect_city("Tokyo", 40)),
    ("The capital of Canada is", expect_city("Ottawa", 40)),
    ("The capital of the United Kingdom is", expect_city("London", 40)),
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["text"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18094/v1/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    cases = ALL_CASES[: args.concurrency]
    if len(cases) < args.concurrency:
        print(f"FAIL  only {len(ALL_CASES)} distinct cases, need {args.concurrency}")
        return 1
    failed = 0
    results: list[tuple[str, str, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(complete, args.url, args.model, prompt, args.max_tokens): (
                prompt,
                checker,
            )
            for prompt, checker in cases
        }
        for fut in as_completed(futs):
            prompt, checker = futs[fut]
            try:
                text = fut.result()
                reason = checker(text)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                text, reason = "", f"request error {e}"
            results.append((prompt, text, reason))
    # print in original prompt order
    order = {prompt: i for i, (prompt, _) in enumerate(cases)}
    results.sort(key=lambda r: order[r[0]])
    for prompt, text, reason in results:
        status = "FAIL" if reason else "PASS"
        if reason:
            failed += 1
        print(f"{status}  prompt={prompt!r}")
        print(f"      reply={text!r}")
        if reason:
            print(f"      why={reason}")
    print(
        "RESULT",
        "FAIL" if failed else "PASS",
        f"({len(cases) - failed}/{len(cases)}) c={args.concurrency}",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
