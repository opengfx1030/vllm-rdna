#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane handoff probe (R-S7-2): token streams from one running server.

Records, for each prompt length, the greedy token stream (tokens, their
logprobs and top-5 alternatives) of a served request. The orchestrator runs
this probe twice, one server after the other (never two servers at once):

  # flag off (baseline tree)
  python tools/rdna/port_v031/handoff_probe.py record \\
      --url http://127.0.0.1:18096/v1/completions --model flash-next \\
      --lengths 1,5,1023,1024,1025,2047,2048,3000,4097,16384,16385 \\
      --output baseline.json

  # flag on (two-lane tree), same recipe and prompts
  python tools/rdna/port_v031/handoff_probe.py record ... --output lane.json

  python tools/rdna/port_v031/handoff_probe.py compare \\
      --baseline baseline.json --lane lane.json

``compare`` applies ``concurrency_probe.py``'s near-tie rules: a greedy flip
at a near tie (margin <= --tie-nats) is a warning, a confident flip or a
quality drop is a failure. A lane handoff loses the prompt's drafter state;
the seeded run below catches a corrupted RNG/state path (it must match
exactly, not near-tie).

``record`` also records the same lengths with ``repetition_penalty 1.1`` and
a fixed ``seed`` at ``temperature 0.8`` (``--seeded-runs``, default on).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrency_probe

DEFAULT_LENGTHS = (
    1,
    5,
    1023,
    1024,
    1025,
    2047,
    2048,
    3000,
    4097,
    16384,
    16385,
)


def prompt_for(length: int, seed: int = 0) -> str:
    """A deterministic ~length-token prompt (distinct per length)."""
    words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    text = []
    state = seed * 2654435761 + length
    while len(text) < length:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        text.append(words[state % len(words)])
    return " ".join(text[:length])


def complete(
    args: argparse.Namespace,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "temperature": temperature if temperature is not None else 0,
        "ignore_eos": True,
        "logprobs": 5,
        "return_tokens_as_token_ids": True,
    }
    if args.repetition_penalty is not None:
        body["repetition_penalty"] = args.repetition_penalty
    if seed is not None:
        body["seed"] = seed
    import urllib.request

    req = urllib.request.Request(
        args.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        choice = json.load(resp)["choices"][0]
    elapsed = time.monotonic() - started
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or list(choice["text"])
    top = lp.get("top_logprobs") or [{} for _ in tokens]
    token_lps = lp.get("token_logprobs") or [0.0 for _ in tokens]
    return {
        "tokens": list(tokens),
        "top": [t or {} for t in top],
        "lps": [x if x is not None else 0.0 for x in token_lps],
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "elapsed_s": elapsed,
    }


def record(args: argparse.Namespace) -> int:
    report: dict = {
        "model": args.model,
        "url": args.url,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "seeded": args.seeded_runs,
        "lengths": {},
    }
    for length in args.lengths:
        prompt = prompt_for(length, args.prompt_seed)
        report["lengths"][str(length)] = {
            "greedy": complete(args, prompt),
            "penalty": complete(args, prompt, temperature=0.8, seed=7),
        }
        if args.seeded_runs:
            report["lengths"][str(length)]["seeded"] = complete(
                args, prompt, temperature=0.8, seed=1234
            )
        print(
            f"handoff_probe: recorded L={length} "
            f"({len(report['lengths'][str(length)]['greedy']['tokens'])} tokens)"
        )
    text = json.dumps(report, indent=1)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text + "\n")
    print(f"handoff_probe: wrote {args.output}")
    return 0


def compare(args: argparse.Namespace) -> int:
    with open(args.baseline, encoding="utf-8") as file:
        baseline = json.load(file)
    with open(args.lane, encoding="utf-8") as file:
        lane = json.load(file)
    counts = {"ok": 0, "tie": 0, "soft": 0, "noise": 0, "FAIL": 0}
    rows: list[str] = []
    for length, refs in baseline["lengths"].items():
        if length not in lane["lengths"]:
            counts["FAIL"] += 1
            rows.append(f"  L={length}: missing in the lane file")
            continue
        for kind, ref in refs.items():
            got = lane["lengths"][length].get(kind)
            if got is None:
                counts["FAIL"] += 1
                rows.append(f"  L={length} {kind}: missing in the lane file")
                continue
            if kind == "seeded" or args.exact:
                same = ref["tokens"] == got["tokens"]
                status = "ok" if same else "FAIL"
                detail = (
                    f"matching / {len(ref['tokens'])}"
                    if same
                    else f"seeded/exact tokens differ "
                    f"({ref['tokens']} vs {got['tokens']})"
                )
            else:
                status, detail = concurrency_probe.compare(ref, got, args)
            counts[status] += 1
            rows.append(f"  L={length} {kind:<7}: {status:<5} {detail}")
    print(
        f"handoff_probe compare: {counts['ok']} ok, {counts['tie']} near-tie, "
        f"{counts['soft']} soft, {counts['noise']} noise, "
        f"{counts['FAIL']} failing"
    )
    for row in rows:
        if not row.endswith(" ok matching") and "matching /" not in row:
            print(row)
    if args.timeline:
        with open(os.path.join(os.path.dirname(args.lane), "handoff.txt"), "w") as file:
            file.write("\n".join(rows) + "\n")
    return 0 if counts["FAIL"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--url", required=True)
    rec.add_argument("--model", required=True)
    rec.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    rec.add_argument("--max-tokens", type=int, default=64)
    rec.add_argument("--repetition-penalty", type=float, default=1.1)
    rec.add_argument("--seeded-runs", action="store_true", default=True)
    rec.add_argument("--no-seeded-runs", dest="seeded_runs", action="store_false")
    rec.add_argument("--prompt-seed", type=int, default=0)
    rec.add_argument("--output", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--baseline", required=True)
    cmp.add_argument("--lane", required=True)
    cmp.add_argument("--tie-nats", type=float, default=0.2)
    cmp.add_argument("--fail-nats", type=float, default=1.0)
    cmp.add_argument("--max-lp-drop", type=float, default=0.8)
    cmp.add_argument(
        "--exact",
        action="store_true",
        help="require identical token streams (seeded runs always are)",
    )
    cmp.add_argument("--timeline", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "record":
        args.lengths = [int(x) for x in args.lengths.split(",") if x]
        return record(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
