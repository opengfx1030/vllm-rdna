#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context needle recall and a small held-out task set, one serve arm.

The RidgeFill fit prompts are plain repeated text; these tasks are held out.
Run it once per arm (flag off, then flag on) and compare ``RESULT`` lines:

    python3 quality_probe.py --model flash-next \
        --url http://127.0.0.1:18094/v1/chat/completions

Needle: one ~8K-token document with unique access codes before the tail cut
and in the last sentences. Every question is a fresh request, so the whole
document is prefilled each time; prefix caching must be off for this probe
(otherwise the second question reuses the first request's cache and the
number says nothing about the fill).

Tasks: twelve one-shot prompts (arithmetic, capitals, exact repetition).
Only an exact expected substring counts; a refusal or a wrong answer is 0.
Prints one JSON object and a PASS/FAIL line; exit 1 when any needle in the
filled region is missed or task accuracy is below ``--min-task-accuracy``.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

FILLER = (
    "The council met on Tuesday and reviewed the annual plan for the "
    "district. Several residents asked about the road repair schedule and "
    "the new bus route. The clerk recorded every question and promised a "
    "written answer before the next meeting. "
    "In the afternoon the committee walked along the river and inspected "
    "the old warehouses near the bridge. "
)

# Unique codes, placed before the tail cut and inside it.
FILLED_CODES = ("ORANGE-314", "SILVER-271", "VIOLET-828")
TAIL_CODES = ("COPPER-905", "INDIGO-667")
NEEDLE_TAIL_SENTENCES = 6
NEEDLE_FILLED_FRACTION = 0.6

TASKS = (
    ("What is 17 + 25?", "42"),
    ("What is 9 * 8?", "72"),
    ("What is 144 / 12?", "12"),
    ("What is 1000 - 457?", "543"),
    ("What is 2 ** 10?", "1024"),
    ("What is 37 + 58?", "95"),
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("Repeat exactly: ZX-4417-B", "ZX-4417-B"),
    ("Repeat exactly: QM-8310-K", "QM-8310-K"),
    ("Repeat exactly: LV-2206-T", "LV-2206-T"),
)


def ask(url: str, model: str, prompt: str, max_tokens: int) -> str:
    # Completions, not chat: the chat template on this checkpoint can spend
    # the whole budget in reasoning_content and return an empty content.
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
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["text"] or ""


def needle_document() -> tuple[str, dict[str, str]]:
    """A ~8K-token document and {code: region} for every planted code."""
    sentences = []
    for i in range(360):
        sentence = FILLER.split(". ")[i % 5]
        sentences.append(sentence.strip() + ".")
    planted: dict[str, str] = {}
    filled_at = int(len(sentences) * NEEDLE_FILLED_FRACTION)
    for i, code in enumerate(FILLED_CODES):
        index = filled_at - 4 * (i + 1)
        sentences[index] = f"The access code for site {i + 1} is {code}."
        planted[code] = "filled"
    for i, code in enumerate(TAIL_CODES):
        index = len(sentences) - NEEDLE_TAIL_SENTENCES + i
        site = len(FILLED_CODES) + i + 1
        sentences[index] = f"The access code for site {site} is {code}."
        planted[code] = "tail"
    return " ".join(sentences), planted


def score_needles(url: str, model: str) -> dict:
    document, planted = needle_document()
    hits = {"filled": [0, 0], "tail": [0, 0]}
    details = {}
    for number, (code, region) in enumerate(planted.items(), 1):
        prompt = (
            document + f"\nQuestion: what is the access code for site {number}? "
            "Answer with the code only."
        )
        try:
            reply = ask(url, model, prompt, 24)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            reply = f"<error {e}>"
        hit = code.lower() in reply.lower()
        hits[region][1] += 1
        hits[region][0] += int(hit)
        details[code] = {"region": region, "hit": hit, "reply": reply[:80]}
    return {"hits": hits, "chars": len(document), "details": details}


def score_tasks(url: str, model: str) -> dict:
    passed = 0
    details = {}
    for prompt, expected in TASKS:
        try:
            reply = ask(url, model, prompt + "\nAnswer only.", 32)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            reply = f"<error {e}>"
        hit = expected.lower() in reply.lower()
        passed += int(hit)
        details[prompt] = {
            "expected": expected,
            "hit": hit,
            "reply": reply[:80],
        }
    return {"passed": passed, "total": len(TASKS), "details": details}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18094/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--min-task-accuracy", type=float, default=0.75)
    args = p.parse_args()
    needle = score_needles(args.url, args.model)
    tasks = score_tasks(args.url, args.model)
    accuracy = tasks["passed"] / tasks["total"] if tasks["total"] else 0.0
    filled = needle["hits"]["filled"]
    tail = needle["hits"]["tail"]
    print(
        json.dumps(
            {
                "needle": needle,
                "tasks": tasks,
                "task_accuracy": accuracy,
            },
            indent=2,
        )
    )
    failed = bool(filled[1] and filled[0] < filled[1])
    failed = failed or accuracy < args.min_task_accuracy
    print(
        "RESULT",
        "FAIL" if failed else "PASS",
        f"filled {filled[0]}/{filled[1]}",
        f"tail {tail[0]}/{tail[1]}",
        f"tasks {tasks['passed']}/{tasks['total']}",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
