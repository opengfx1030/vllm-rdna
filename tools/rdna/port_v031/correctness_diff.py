#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two ``serve_validate.sh`` output dirs for two-lane correctness.

The flag-off arm is the reference: the flag-on engine must generate the same
text for every deterministic probe. Only generated text is compared; timings
are not part of the correctness contract.

- ``probe.txt``: every ``reply='...'`` continuation.
- ``ppl.txt``: the PPL value and every greedy ``-> '...'`` continuation.
- ``prefix.txt``: the PASS/FAIL verdict. The prompt starts with a
  ``time.time_ns()`` nonce, so its two texts are intentionally not comparable.
- ``concurrency.json``: every ``text`` field (not logprobs).
- bench result JSONs: no sampled text in this fork; the completion counters
  are compared and the timing fields are metrics-only.

Exit status is 0 when every comparable output matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import regex as re

_REPLY = re.compile(r"^\s+reply='(.*)'$", re.MULTILINE)
_PPL_VALUE = re.compile(r"^PPL (\S+)", re.MULTILINE)
_PPL_OK = re.compile(r"^ok\s+'.*?' -> '(.*)'$", re.MULTILINE)
_VERDICT = re.compile(r"-> (PASS|FAIL)")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _collect_texts(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            out.append(text)
        for item in value.values():
            _collect_texts(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_texts(item, out)


def _concurrency_texts(path: Path) -> list[str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    texts: list[str] = []
    for section in data.values():
        _collect_texts(section, texts)
    return texts


def _compare(name: str, off: list[str] | None, on: list[str] | None) -> bool:
    if off is None or on is None:
        print(f"{name}: MISSING")
        return False
    if off == on:
        print(f"{name}: {len(off)} outputs identical")
        return True
    first = next(
        (i for i, (a, b) in enumerate(zip(off, on)) if a != b),
        min(len(off), len(on)),
    )
    print(f"{name}: DIFFERS at item {first} ({len(off)} vs {len(on)})")
    if first < len(off) and first < len(on):
        print(f"  off: {off[first][:100]!r}")
        print(f"  on:  {on[first][:100]!r}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", required=True, help="flag-off output dir")
    parser.add_argument("--on", required=True, help="flag-on output dir")
    args = parser.parse_args(argv)
    off = Path(args.off)
    on = Path(args.on)
    ok = True
    print("two-lane correctness diff")
    print(f"off={off}")
    print(f"on={on}")

    off_probe = _read(off / "probe.txt")
    on_probe = _read(on / "probe.txt")
    ok &= _compare(
        "probe.txt",
        _REPLY.findall(off_probe or ""),
        _REPLY.findall(on_probe or ""),
    )

    off_ppl = _read(off / "ppl.txt")
    on_ppl = _read(on / "ppl.txt")
    ok &= _compare(
        "ppl.txt continuations",
        _PPL_OK.findall(off_ppl or ""),
        _PPL_OK.findall(on_ppl or ""),
    )
    off_value = _PPL_VALUE.search(off_ppl or "")
    on_value = _PPL_VALUE.search(on_ppl or "")
    if off_value and on_value and off_value.group(1) == on_value.group(1):
        print(f"ppl.txt value: {off_value.group(1)} identical")
    else:
        print(
            "ppl.txt value: DIFFERS "
            f"({off_value.group(1) if off_value else 'MISSING'} vs "
            f"{on_value.group(1) if on_value else 'MISSING'})"
        )
        ok = False

    off_prefix = _VERDICT.findall(_read(off / "prefix.txt") or "")
    on_prefix = _VERDICT.findall(_read(on / "prefix.txt") or "")
    ok &= _compare("prefix.txt verdicts", off_prefix, on_prefix)
    print("prefix.txt text: skipped (time-based nonce prompt)")

    ok &= _compare(
        "concurrency.json",
        _concurrency_texts(off / "concurrency.json"),
        _concurrency_texts(on / "concurrency.json"),
    )

    off_bench = sorted(off.glob("bench-*.json"))
    on_bench = {path.name: path for path in on.glob("bench-*.json")}
    if not off_bench and not on_bench:
        print("bench: none")
    for left_path in off_bench:
        right_path = on_bench.pop(left_path.name, None)
        if right_path is None:
            print(f"{left_path.name}: MISSING on")
            ok = False
            continue
        left = json.loads(left_path.read_text(encoding="utf-8"))
        right = json.loads(right_path.read_text(encoding="utf-8"))
        left_texts: list[str] = []
        right_texts: list[str] = []
        _collect_texts(left, left_texts)
        _collect_texts(right, right_texts)
        if left_texts or right_texts:
            ok &= _compare(left_path.name, left_texts, right_texts)
            continue
        left_done = left.get("completed")
        right_done = right.get("completed")
        if left_done != right_done:
            print(f"{left_path.name}: completed {left_done} != {right_done}")
            ok = False
        else:
            print(f"{left_path.name}: metrics-only, completed {left_done} equal")
    for name in on_bench:
        print(f"{name}: MISSING off")
        ok = False

    print("RESULT PASS" if ok else "RESULT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
