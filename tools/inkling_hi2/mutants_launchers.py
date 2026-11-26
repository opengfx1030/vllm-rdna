#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kill the HI2 kit's launcher/lease/window mutants with the CPU test module.

Each mutant is a small textual edit of the kit scripts or clients that restores a
bug the behavioural tests exist for (a foreign-holder pass, an early boot-lock
release, a skipped device check, a thermal/eviction window that passes, ...).
The test module runs against a scratch copy of ``tools/`` and the test file, so
the checkout is never modified.

  python tools/inkling_hi2/mutants_launchers.py [-v]

Exit 0 only if the baseline passes and every mutant is killed.  The mutants need
no GPU (the launchers run against the fake torch/stub server in the test module).
The model/loader battery is ``tools/inkling_hi2/mutants.py``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEST = "tests/models/inkling2_loop/test_hi2_slots.py"

# (name, [(file, old, new), ...], pytest -k of the tests that must fail)
MUTANTS: list[tuple[str, list[tuple[str, str, str]], str]] = [
    (
        "lease_gate_off",
        [("serve.sh", "\nlease_acquire\n", "\ntrue\n")],
        "another_jobs_holder",
    ),
    (
        "lease_owner_back_to_hi2inkling",
        [
            (
                "serve.sh",
                "OWNER=${OWNER:-HI2InklingRuntime}",
                "OWNER=${OWNER:-HI2Inkling}",
            )
        ],
        "another_jobs_holder",
    ),
    (
        "lease_vram_ignored",
        [("lease.sh", "((free >= want)); then", "((1)); then")],
        "free_vram",
    ),
    (
        "lease_request_clobbers",
        [
            (
                "lease.sh",
                "if [[ -z $request_blocked && -z $LEASE_REQUEST ]]; then",
                "if true; then",
            )
        ],
        "never_overwrites",
    ),
    (
        "lease_request_owner_ignored",
        [
            (
                "lease.sh",
                'if [[ $req_owner != "$OWNER" ]]; then',
                "if false; then",
            )
        ],
        "never_overwrites",
    ),
    (
        "lease_stale_holder_not_detected",
        [
            (
                "lease.sh",
                (
                    'if lease_pid_alive "$LEASE_DIR/$OWNER.$SLOT"; then\n'
                    "                own_holder=1"
                ),
                "if true; then\n                own_holder=1",
            )
        ],
        "stale_own_holder",
    ),
    (
        "boot_lock_released_early",
        [
            (
                "serve.sh",
                "    break\ndone\n\nif [[ $REGRESSION == 0 ]]; then",
                "    break\ndone\nexec 9>&-\n\nif [[ $REGRESSION == 0 ]]; then",
            ),
            (
                "serve.sh",
                "\nexec 9>&-\nif ((!ready)); then",
                "\ntrue\nif ((!ready)); then",
            ),
        ],
        "boot_lock_until_ready",
    ),
    (
        "second_device_check_off",
        [
            (
                "serve.sh",
                "check_devices\nif [[ -n $SERVER_PID_FILE ]]",
                "if [[ -n $SERVER_PID_FILE ]]",
            )
        ],
        "checks_the_devices_around_the_boot",
    ),
    (
        "post_health_check_off",
        [("serve.sh", "\ncheck_devices || {\n", "\ntrue || {\n")],
        "checks_the_devices_around_the_boot or rechecks_the_devices_after_health",
    ),
    (
        "per_cell_device_check_off",
        [
            (
                "smoke.sh",
                (
                    "measure_begin() {\n    touch_quiet\n    check_devices\n"
                    "    wait_boot_lock\n}"
                ),
                "measure_begin() {\n    touch_quiet\n    wait_boot_lock\n}",
            )
        ],
        "rechecks_the_devices",
    ),
    (
        "live_check_without_hip",
        [
            (
                "serve.sh",
                'HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"',
                "HIP_VISIBLE_DEVICES=",
            )
        ],
        "checks_the_devices_around_the_boot",
    ),
    (
        "server_keeps_rocr",
        [
            (
                "serve.sh",
                (
                    "env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \\\n"
                    '    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" EXPECT_PCI_BUSES='
                ),
                (
                    "env ROCR_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES= \\\n"
                    '    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" EXPECT_PCI_BUSES='
                ),
            )
        ],
        "checks_the_devices_around_the_boot",
    ),
    (
        "temp_threshold_never",
        [("smoke.sh", "sys.exit(1 if peak >= 95 else 0)", "sys.exit(0)")],
        "hot",
    ),
    (
        "eviction_missing_passes",
        [
            (
                "smoke.sh",
                "if [[ ${EVICT_MISSING:-0} == 1 ]]; then",
                "if false; then",
            ),
            (
                "smoke.sh",
                "[[ -n ${EVICT_BEFORE:-} ]] || return 1",
                "[[ -n ${EVICT_BEFORE:-} ]] || return 0",
            ),
        ],
        "without_kfd_workers",
    ),
    (
        "bench_b4_allowed",
        [("bench.py", 'if slot_name != "a2":', "if False:")],
        "refuse_b4",
    ),
    (
        "bench_missing_workers_ok",
        [
            (
                "bench.py",
                'elif not evictions or "no vLLM workers" in evictions:',
                "elif False:",
            )
        ],
        "bench_invalid_without_kfd_workers or bench_window_record",
    ),
    (
        "serve_passes_empty_spec_config",
        [
            (
                "serve.sh",
                'EXTRA_ARGS="$EXTRA" "${spec_args[@]}" 9>&- &',
                'EXTRA_ARGS="$EXTRA" SPEC_CONFIG="${SPEC_CONFIG:-}" 9>&- &',
            )
        ],
        "lets_the_recipe_set_the_spec_config",
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
        env={
            **__import__("os").environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--only", default="", help="run only mutants whose name contains this text"
    )
    ap.add_argument(
        "--keep", action="store_true", help="keep the scratch tree for inspection"
    )
    args = ap.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="hi2-kit-mutants-"))
    try:
        prepare(scratch)
        subset = (
            "another_jobs_holder or free_vram or boot_lock_until_ready or "
            "checks_the_devices_around_the_boot or rechecks_the_devices or hot or "
            "rechecks_the_devices_after_health or "
            "without_kfd_workers or refuse_b4 or bench_invalid_without_kfd_workers "
            "or bench_window_record or never_overwrites or stale_own_holder or "
            "live_own_holder"
        )
        baseline = pytest_run(scratch, subset)
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
