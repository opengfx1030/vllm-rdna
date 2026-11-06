#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Failure rate of repeated concurrency probes (hook.sh CONC_REPEAT) per arm.

  conctally.py ARMS_ROOT label=arm[:glob] [label=arm[:glob] ...]
                                               (run with the venv python)

Each arm directory holds conc/<glob>, default run*.txt (hook.sh CONC_REPEAT writes
run<N>.txt, CONC_REPEAT_NOEOS writes noeos<N>.txt): outputs of concurrency_probe.py.
The probe's own pass criterion decides: a run PASSES when it prints no FAIL line.
Prints one row per arm: runs, failing runs, failing checks / total checks, near-tie
and soft WARN counts, and the failing checks themselves. With two or more arms it
adds a two-sided Fisher exact test of the failing-run counts for every pair, so one
flip is not read as a verdict (greedy output is not bitwise reproducible across
boots).
"""

import itertools
import sys
from math import comb
from pathlib import Path

import regex as re

RESULT = re.compile(
    r"RESULT (PASS|FAIL): (\d+) failing, (\d+) near-tie, (\d+) soft, "
    r"(\d+) solo-noise, (\d+) identical of (\d+) checks"
)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """P-value for the 2x2 table [[a, b], [c, d]] (failing, passing per arm)."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    p_obs = prob(a)
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    return min(
        1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= p_obs * (1 + 1e-9))
    )


def tally(arm_dir: Path, pattern: str = "run*.txt") -> dict:
    runs = fails = checks = failing_checks = ties = softs = 0
    failing = []
    for txt in sorted((arm_dir / "conc").glob(pattern), key=lambda p: len(p.stem)):
        text = txt.read_text(errors="replace")
        m = RESULT.search(text)
        if not m:
            continue
        runs += 1
        verdict, nfail, nties, nsoft, _noise, _ok, ntot = m.groups()
        fails += verdict == "FAIL"
        failing_checks += int(nfail)
        ties += int(nties)
        softs += int(nsoft)
        checks += int(ntot)
        failing += [
            f"{txt.stem}: {ln[:110]}"
            for ln in text.splitlines()
            if ln.startswith("FAIL")
        ]
    return {
        "runs": runs,
        "fails": fails,
        "checks": checks,
        "failing_checks": failing_checks,
        "ties": ties,
        "softs": softs,
        "failing": failing,
    }


def main() -> int:
    root = Path(sys.argv[1])
    arms = {}
    for spec in sys.argv[2:]:
        label, _, rest = spec.partition("=")
        arm, _, pattern = rest.partition(":")
        arms[label] = tally(root / arm, pattern or "run*.txt")
    print("| arm | runs | failing runs | failing / total checks | near-tie | soft |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, t in arms.items():
        print(
            f"| {label} | {t['runs']} | {t['fails']} | {t['failing_checks']} / "
            f"{t['checks']} | {t['ties']} | {t['softs']} |"
        )
    for (la, ta), (lb, tb) in itertools.combinations(arms.items(), 2):
        if ta["runs"] and tb["runs"]:
            p = fisher_two_sided(
                ta["fails"],
                ta["runs"] - ta["fails"],
                tb["fails"],
                tb["runs"] - tb["fails"],
            )
            print(f"\nFisher exact, failing runs {la} vs {lb}: p = {p:.3f}")
    for label, t in arms.items():
        for line in t["failing"]:
            print(f"\n{label} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
