#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kill the final-card driver's control-flow mutants with the CPU test module.

``tools/inkling_hi2/validate_final_a2.sh`` is one long bash script; its risk is
control flow, not numerics.  Each mutant below restores one bug the driver must
never have: the disk guard off, a failed boot treated as success, and a PASSed
step re-run on resume.  The tests run against a scratch copy of ``tools/`` and
the test file, so the checkout is never modified.

  python tools/inkling_hi2/mutants_final.py [-v] [--only TEXT]

Exit 0 only if the baseline passes and every mutant is killed.  No GPU needed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEST = "tests/models/inkling2_loop/test_hi2_final_driver.py"
SUBSET = "dry_run or disk_guard or resume or first_failure or bench"

# (name, [(file, old, new), ...], pytest -k of the tests that must fail)
MUTANTS: list[tuple[str, list[tuple[str, str, str]], str]] = [
    (
        "disk_guard_off",
        [
            (
                "validate_final_a2.sh",
                "if ((free_kib < need_kib)); then",
                "if false; then",
            )
        ],
        "disk_guard",
    ),
    (
        "failed_step_not_fatal",
        [
            (
                "validate_final_a2.sh",
                '    if ((rc != 0)); then\n        log "FAIL $name',
                '    if false; then\n        log "FAIL $name',
            )
        ],
        "first_failure",
    ),
    (
        "resume_ignored",
        [
            (
                "validate_final_a2.sh",
                "if [[ $RESUME == 1 && -f $RUN/steps/$name.pass ]]; then",
                "if false; then",
            )
        ],
        "resume",
    ),
    (
        "bench_solo_flags_wrong",
        [
            (
                "validate_final_a2.sh",
                (
                    'num "solo_flags" "gmem=$GMEM_SINGLE maxlen=$CTX '
                    'maxbat=$MAXBAT seqs=$SEQS decode_tokens=$decode"'
                ),
                (
                    'num "solo_flags" "gmem=$GMEM_PAIR maxlen=$CTX '
                    'maxbat=$MAXBAT seqs=$SEQS decode_tokens=$decode"'
                ),
            )
        ],
        "bench",
    ),
    (
        "dry_run_skips_tool_help",
        [
            (
                "validate_final_a2.sh",
                '        "$V" "$tool" --help >>"$RUN/dryrun.log" 2>&1 ||',
                '        true >>"$RUN/dryrun.log" 2>&1 ||',
            )
        ],
        "dry_run",
    ),
]


def prepare(scratch: Path) -> None:
    for sub in ("tools", "tests/models/inkling2_loop"):
        shutil.copytree(
            REPO / sub,
            scratch / sub,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def pytest_run(scratch: Path, expr: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-x", "-k", expr],
        cwd=scratch,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--only", default="", help="run only mutants whose name contains this text"
    )
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree")
    args = ap.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="hi2-final-mutants-"))
    try:
        prepare(scratch)
        baseline = pytest_run(scratch, SUBSET)
        if baseline.returncode != 0:
            print(baseline.stdout[-4000:])
            print("baseline failed; mutants not run", file=sys.stderr)
            return 1
        print(f"baseline: {baseline.stdout.strip().splitlines()[-1]}")
        selected = [m for m in MUTANTS if args.only in m[0]]
        survived = []
        for name, edits, expr in selected:
            originals = {}
            for rel, old, new in edits:
                path = scratch / "tools" / "inkling_hi2" / rel
                if path not in originals:
                    originals[path] = path.read_text()
                text = path.read_text()
                if old not in text:
                    raise SystemExit(f"{name}: pattern not found in {rel}: {old!r}")
                path.write_text(text.replace(old, new, 1))
            result = pytest_run(scratch, expr)
            for path, text in originals.items():
                path.write_text(text)
            killed = result.returncode != 0
            print(f"{name}: {'KILLED' if killed else 'SURVIVED'} ({expr})")
            if args.verbose and not killed:
                print(result.stdout[-2000:])
            if not killed:
                survived.append(name)
        print(f"mutants: {len(selected) - len(survived)}/{len(selected)} killed")
        if survived:
            print("survived: " + ", ".join(survived), file=sys.stderr)
            return 1
        return 0
    finally:
        if args.keep:
            print(f"scratch kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
