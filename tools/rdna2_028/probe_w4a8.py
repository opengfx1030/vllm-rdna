#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coherence + garbage probe for the W4A8 A/B (Qwen3.8-27B-AWQ-INT4).

Sends deterministic prompts (temp=0) to a running server and reports, per
prompt, the raw text plus a coherence/garbage verdict. Used to prove the
W4A8=0 arm is coherent and to detect the W4A8=1 arm's collapse.

Usage:
  python probe_w4a8.py <base_url> <served_model_name> [n_repeats]
"""
import json
import re
import sys

from openai import OpenAI

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18210/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "q27d"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1

PROMPTS = [
    ("france", "The capital of France is", ["paris"]),
    ("math", "2 + 2 =", ["4"]),
    ("gpu", "In one sentence, what is a GPU?", []),
    ("python", "Write one sentence about Python.", []),
]

# A token repeated >= this many times in a row is a collapse signature.
REPEAT_THRESHOLD = 5
# If one whitespace token accounts for >= this share of the output, the model
# has degenerated onto a single token (e.g. "Register Register ..." or a run
# like "ductductduct" that has no whitespace at all).
DOMINANCE_THRESHOLD = 0.5
# Longest repeated substring (in chars) that signals a degenerate loop even
# without whitespace separators.
LOOP_MIN_LEN = 6


def _longest_repeat_span(text: str) -> int:
    """Return the longest run length of the repeated form s = unit * k, k >= 3."""
    n = len(text)
    best = 0
    for unit in range(1, max(2, n // 3 + 1)):
        if n % unit:
            continue
        rep = n // unit
        if rep < 3:
            continue
        if text == text[:unit] * rep:
            best = max(best, unit)
    return best


def garbage_flags(text: str) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty")
    toks = re.findall(r"\S+", text)
    if toks:
        top = max(set(toks), key=toks.count)
        if toks.count(top) / len(toks) >= DOMINANCE_THRESHOLD and len(toks) >= 4:
            flags.append(f"dominant:{top[:24]}")
    # repeated token collapse: same whitespace-delimited token >= threshold
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            if run >= REPEAT_THRESHOLD:
                flags.append(f"repeat:{toks[i][:24]}")
                break
        else:
            run = 1
    # whitespace-free degenerate loop, e.g. "ductductduct..."
    span = _longest_repeat_span(text.strip())
    if span >= LOOP_MIN_LEN:
        flags.append(f"char-loop:{text.strip()[:span][:24]}")
    # non-printable / replacement chars
    if re.search(r"[\ufffd]", text):
        flags.append("replacement-char")
    return flags


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    results = []
    for _ in range(N):
        for name, prompt, expect in PROMPTS:
            try:
                r = client.completions.create(
                    model=MODEL,
                    prompt=prompt,
                    max_tokens=32,
                    temperature=0.0,
                    extra_body={"ignore_eos": True},
                )
                text = r.choices[0].text or ""
            except Exception as e:  # noqa: BLE001
                text = ""
                results.append({"name": name, "prompt": prompt, "error": repr(e)})
                print(f"[{name}] ERROR: {e!r}")
                continue
            flags = garbage_flags(text)
            low = text.lower()
            coherent = any(e in low for e in expect) if expect else bool(text.strip())
            verdict = "OK" if (coherent and not flags) else "BAD"
            results.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "output": text,
                    "coherent": coherent,
                    "garbage": flags,
                    "verdict": verdict,
                }
            )
            print(f"[{name}] {verdict:3s} coherent={coherent} garbage={flags} | {text[:120]!r}")

    ok = sum(1 for r in results if r.get("verdict") == "OK")
    bad = sum(1 for r in results if r.get("verdict") == "BAD")
    print(f"\n=== {ok} OK / {bad} BAD / {len(results)} total ===")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
