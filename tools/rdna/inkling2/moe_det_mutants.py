#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kill the deterministic Inkling-2 expert-path mutants with the CPU test module.

Each mutant restores the *shipped* (nondeterministic) behavior in one place:
the switch is ignored, the decode kernel's atomic top-k epilogue is kept, the
decode w13 path is kept, or the top-k sum drops back to fp16.  The test module
must fail on every one of them.

The product file is edited in place and restored from the saved bytes in a
``finally`` block, so the checkout is left byte-identical (verified at the end).

  python tools/rdna/inkling2/moe_det_mutants.py [-v]
  (run with the box venv's python; no GPU needed)

Exit 0 only if the baseline passes and every mutant is killed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TARGET = REPO / "vllm/models/inkling2_loop/amd/moe.py"
TEST = "tests/models/inkling2_loop/test_moe_deterministic.py"
SUBSET = "deterministic or flag_"

# (name, [(old, new), ...]) -- every mutant must make the test module fail
MUTANTS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "switch_forgotten",  # the flag no longer selects the deterministic path
        [
            (
                "_DETERMINISTIC = _deterministic_switch(os.environ)",
                "_DETERMINISTIC = False",
            )
        ],
    ),
    (
        "atomic_topk_kept",  # det w2 keeps the fused atomic top-k epilogue
        [
            (
                (
                    "                    bm,\n"
                    "                    False,\n"
                    "                    0,\n"
                    "                    cta_info=cta,\n"
                    "                )\n"
                    "                _sum_topk_rows_f32"
                ),
                (
                    "                    bm,\n"
                    "                    True,\n"
                    "                    K,\n"
                    "                    cta_info=cta,\n"
                    "                )\n"
                    "                _sum_topk_rows_f32"
                ),
            )
        ],
    ),
    (
        "decode_w13_kept",  # the flag no longer forces the BM w13 kernel
        [
            (
                "and (tc >= _PREFILL_BM_MIN_TOKENS or det)",
                "and tc >= _PREFILL_BM_MIN_TOKENS",
            )
        ],
    ),
    (
        "fp16_topk_sum",  # the deterministic sum silently loses fp32
        [
            (
                "y = rows.view(t, k, rows.shape[-1]).float()",
                "y = rows.view(t, k, rows.shape[-1])",
            )
        ],
    ),
]


def run_tests(extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", TEST, "-q", "-k", SUBSET, *extra]
    return subprocess.run(cmd, cwd=REPO).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not TARGET.is_file():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2
    original = TARGET.read_bytes()
    try:
        rc = run_tests([])
        if rc != 0:
            print("baseline: FAIL (the tests must pass unmutated)", file=sys.stderr)
            return 1
        print("baseline: pass")
        killed = 0
        for name, edits in MUTANTS:
            text = original.decode("utf-8")
            for old, new in edits:
                if text.count(old) != 1:
                    print(
                        f"{name}: pattern count {text.count(old)} != 1", file=sys.stderr
                    )
                    return 2
                text = text.replace(old, new)
            TARGET.write_text(text, encoding="utf-8")
            try:
                rc = run_tests([] if args.verbose else ["--no-header", "-x"])
            finally:
                TARGET.write_bytes(original)
            status = "killed" if rc != 0 else "SURVIVED"
            killed += rc != 0
            print(f"{name}: {status}")
        ok = killed == len(MUTANTS) and TARGET.read_bytes() == original
        restored = TARGET.read_bytes() == original
        print(f"{killed}/{len(MUTANTS)} killed; checkout restored: {restored}")
        return 0 if ok else 1
    finally:
        TARGET.write_bytes(original)


if __name__ == "__main__":
    raise SystemExit(main())
