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
SUBSET = (
    "dry_run or disk_guard or resume or first_failure or bench or "
    "lease_inherit or own_request or pair_restarted_once or "
    "pair_unhealthy_fails_cleanly or vision_ or kl_backfill or speculator or "
    "boot720_"
)

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
        "inherit_unconditional",
        [
            (
                "lease.sh",
                "if [[ $LEASE_INHERIT == 1 ]] && lease_inherit_check; then",
                "if [[ $LEASE_INHERIT == 1 ]]; then",
            )
        ],
        "lease_inherit_foreign",
    ),
    (
        "inherit_release_removes_holder",
        [
            (
                "lease.sh",
                (
                    "        # Never delete the owner's holder: the outer "
                    "driver releases it.\n"
                    "        LEASE_REQUEST=\n"
                    "        LEASE_HOLDER_CREATED=\n"
                    "        return 0"
                ),
                (
                    "        # Never delete the owner's holder: the outer "
                    "driver releases it.\n"
                    "        LEASE_REQUEST=\n"
                    '        rm -f "$LEASE_DIR/$OWNER.$SLOT"\n'
                    "        return 0"
                ),
            )
        ],
        "lease_release_inherit",
    ),
    (
        "own_request_self_queues",
        [
            (
                "lease.sh",
                'if [[ $(lease_line_pid "$LEASE_DIR/$SLOT.request") == "$$" ]]; then',
                "if false; then",
            )
        ],
        "own_request",
    ),
    (
        "pair_health_gate_ignored",
        [
            (
                "validate_final_a2.sh",
                "    if pair_health; then\n        return 0\n    fi",
                "    return 0",
            )
        ],
        "pair_restarted_once",
    ),
    (
        "pair_restart_retried",
        [
            (
                "validate_final_a2.sh",
                "    if restart_pair; then",
                "    if { restart_pair || restart_pair; }; then",
            )
        ],
        "pair_unhealthy_fails_cleanly",
    ),
    (
        "pair_unhealthy_ignored",
        [
            (
                "validate_final_a2.sh",
                (
                    '    log "pair-unhealthy: $step cannot run; the pair did not '
                    'come back after one restart (see $RUN/logs)"\n    return 1'
                ),
                (
                    '    log "pair-unhealthy: $step cannot run; the pair did not '
                    'come back after one restart (see $RUN/logs)"'
                ),
            )
        ],
        "pair_unhealthy_fails_cleanly",
    ),
    (
        "kl_old_key",
        [("validate_final_a2.sh", "d['kl_mean']", "d['mean_kl']")],
        "pair_restarted_once",
    ),
    (
        "kl_backfill_off",
        [
            (
                "validate_final_a2.sh",
                "run_step smoke s_smoke\nkl_backfill\nrun_step kl s_kl",
                "run_step smoke s_smoke\nrun_step kl s_kl",
            )
        ],
        "kl_backfill",
    ),
    (
        "dspark_spec_check_off",
        [
            (
                "validate_final_a2.sh",
                (
                    '    require_spec_config "$RUN/logs/serve-dspark.log" on dspark'
                    " || return 1\n"
                ),
                "",
            )
        ],
        "speculator",
    ),
    (
        "plain_spec_check_off",
        [
            (
                "validate_final_a2.sh",
                (
                    '    require_spec_config "$RUN/logs/serve-plain16k.log" off'
                    " plain16k || return 1\n"
                ),
                "",
            )
        ],
        "plain16k_boot_must_not",
    ),
    (
        "boot720_spec_check_off",
        [
            (
                "validate_final_a2.sh",
                (
                    '    require_spec_config "$RUN/logs/serve-boot720.log" on'
                    " boot720 || return 1\n"
                ),
                "",
            )
        ],
        "boot720_fails_when_the_speculator",
    ),
    (
        "vision_fallback_ignored",
        [
            (
                "validate_final_a2.sh",
                '    if summary=$(vision_eval_gate 2>>"$RUN/steps/vision.log"); then',
                "    if summary=$(echo ignored); then",
            )
        ],
        "vision_probe_defect_without",
    ),
    (
        "vision_check_failure_not_fatal",
        [
            (
                "validate_final_a2.sh",
                "    if [[ -f $RUN/steps/vision.json ]]; then",
                "    if false; then",
            )
        ],
        "vision_probe_check_failure",
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
