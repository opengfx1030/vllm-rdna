#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two ``serve_validate.sh`` output dirs for two-lane correctness.

The flag-off arm is the reference: the flag-on engine must generate the same
text for every deterministic probe. ``probe.txt``, ``ppl.txt`` and
``prefix.txt`` are compared byte for byte. ``concurrency.json`` is compared
by the sampled outputs of every section (its timings are not part of the
correctness contract). Bench result JSONs carry no sampled text in this fork;
for them the completion counters are compared and the timing fields are
reported as metrics-only, never as a correctness failure.

Exit status is 0 when every comparable output matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BYTE_FILES = ("probe.txt", "ppl.txt", "prefix.txt")


def _text_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", required=True, help="flag-off output dir")
    parser.add_argument("--on", required=True, help="flag-on output dir")
    args = parser.parse_args(argv)
    off = Path(args.off)
    on = Path(args.on)
    ok = True
    report = ["two-lane correctness diff", f"off={off}", f"on={on}"]

    for name in _BYTE_FILES:
        left = _text_bytes(off / name)
        right = _text_bytes(on / name)
        if left is None or right is None:
            report.append(f"{name}: MISSING")
            ok = False
        elif left == right:
            report.append(f"{name}: identical")
        else:
            report.append(f"{name}: DIFFERS")
            ok = False

    off_texts = _concurrency_texts(off / "concurrency.json")
    on_texts = _concurrency_texts(on / "concurrency.json")
    if off_texts is None or on_texts is None:
        report.append("concurrency.json: MISSING")
        ok = False
    elif off_texts == on_texts:
        report.append(f"concurrency.json: {len(off_texts)} sampled outputs identical")
    else:
        report.append("concurrency.json: DIFFERS")
        ok = False

    off_bench = sorted(off.glob("bench-*.json"))
    on_bench = {path.name: path for path in on.glob("bench-*.json")}
    if not off_bench and not on_bench:
        report.append("bench: none")
    for left_path in off_bench:
        right_path = on_bench.pop(left_path.name, None)
        if right_path is None:
            report.append(f"{left_path.name}: MISSING on")
            ok = False
            continue
        left = json.loads(left_path.read_text(encoding="utf-8"))
        right = json.loads(right_path.read_text(encoding="utf-8"))
        left_texts: list[str] = []
        right_texts: list[str] = []
        _collect_texts(left, left_texts)
        _collect_texts(right, right_texts)
        if left_texts or right_texts:
            if left_texts == right_texts:
                report.append(f"{left_path.name}: {len(left_texts)} texts identical")
            else:
                report.append(f"{left_path.name}: TEXTS DIFFER")
                ok = False
            continue
        left_done = left.get("completed")
        right_done = right.get("completed")
        if left_done != right_done:
            report.append(f"{left_path.name}: completed {left_done} != {right_done}")
            ok = False
        else:
            report.append(
                f"{left_path.name}: metrics-only, completed {left_done} equal"
            )
    for name in on_bench:
        report.append(f"{name}: MISSING off")
        ok = False

    report.append("RESULT PASS" if ok else "RESULT FAIL")
    print("\n".join(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
