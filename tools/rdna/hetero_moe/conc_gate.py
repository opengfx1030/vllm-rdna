#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrency gate of a hetero-MoE arm: transport and determinism, not token
equality with the T=3 solo reference (HxServer's proposal, hx doc 8.9).

Input: an arm directory of ``tools/rdna/tp8pp2/driver.sh`` (``arms/NAME``, CFG=h1
or h1v) with the probe results ``concurrency.json`` (the boot probe) and
``conc/run*.json`` (``CONC_REPEAT=N``), the hub's ``serve.log`` and the expert
servers' ``hetero/<id>.log``.

PASS needs all of:

* the hx digest (``VLLM_HX_FLUSH`` bit 256) of the hub and of every server: the last
  counters line says 0 bad at the first read and 0 unhealed. The digest verifies x,
  ids, weights and y of every exchange end to end, so it is the transport test;
* ``solo`` bit-identical (tokens and logprobs) in every run, and ``solo2`` equal to
  ``solo`` in tokens and logprobs in every run: with a deterministic hub MoE
  (``VLLM_RDNA2_MOE_DETERMINISTIC=1``) the same request at the same width is exactly
  reproducible, so a difference is a non-deterministic kernel or a wrong exchange.

Reported, never failed: how many ``together``, ``churn`` and ``long_together`` items
diverge from the T=3 ``solo`` reference (the batch composition decides how a row's
pairs are split over devices, so some rows round differently), how many of them at or
right after a special token, how many are confident (the probe's FAIL class), and the
probe's own verdicts.

    python tools/rdna/hetero_moe/conc_gate.py ARMS/h1v-int-conc [--rows]

Exit code: 0 PASS, 1 FAIL, 2 no probe results. The first output line is the verdict.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import regex as re

SPECIAL = {248044: "<|endoftext|>", 248045: "<|im_start|>", 248046: "<|im_end|>"}
FIELDS = ("ok", "bad", "healed", "unhealed", "late", "rescans")
DIGEST = re.compile(
    r"\b(?P<role>hub|server) digest: (?P<ok>\d+) ok, (?P<bad>\d+) bad at the first "
    r"read, (?P<healed>\d+) healed, (?P<unhealed>\d+) unhealed, (?P<late>\d+) late "
    r"digest words, (?P<rescans>\d+) rescans"
)
VERDICT = "HETERO-CONC-GATE"
FLIP_SCENARIOS = (
    ("together", "solo"),
    ("churn", "solo"),
    ("long_together", "long_solo"),
)


def token_id(token: str) -> int | None:
    """``token_id:123`` (the probe asks for token ids) -> 123, else None."""
    head, _, tail = token.partition(":")
    return int(tail) if head == "token_id" and tail.lstrip("-").isdigit() else None


def prompt_of(key: str) -> str:
    """Prompt index of an item: churn items are keyed ``(slot, prompt)``."""
    return key.strip("()").split(", ")[1] if key.startswith("(") else key


def first_divergence(ref: dict, got: dict) -> int | None:
    a, b = ref["tokens"], got["tokens"]
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def signature(item: dict) -> tuple:
    return tuple(item["tokens"]), tuple(item["lps"])


def last_digest(path: str, role: str) -> dict[str, int] | None:
    """The last counters line of ``role`` in a log (the counters are cumulative)."""
    found = None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = DIGEST.search(line)
                if m is not None and m["role"] == role:
                    found = {k: int(m[k]) for k in FIELDS}
    except OSError:
        return None
    return found


def describe(c: dict[str, int]) -> str:
    return (
        f"{c['ok']} ok, {c['bad']} bad at the first read, {c['healed']} healed, "
        f"{c['unhealed']} unhealed, {c['late']} late digest words, "
        f"{c['rescans']} rescans"
    )


def check_digest(arm: str) -> tuple[list[str], list[str]]:
    """(reasons for FAIL, report lines): hub in serve.log, one line per server in
    hetero/<id>.log (the servers are those with a ready file)."""
    reasons: list[str] = []
    report: list[str] = []
    seen = {"hub": last_digest(os.path.join(arm, "serve.log"), "hub")}
    ids = sorted(
        os.path.basename(p)[: -len(".ready")]
        for p in glob.glob(os.path.join(arm, "hetero", "*.ready"))
    )
    for sid in ids:
        seen[sid] = last_digest(os.path.join(arm, "hetero", f"{sid}.log"), "server")
    if not ids:
        reasons.append("no server (hetero/*.ready) found")
    for who, counters in seen.items():
        if counters is None:
            reasons.append(f"no digest line of {who} (VLLM_HX_FLUSH without bit 256?)")
            report.append(f"  digest {who}: no counters line")
            continue
        report.append(f"  digest {who}: {describe(counters)}")
        if counters["bad"] or counters["unhealed"]:
            reasons.append(
                f"digest {who}: {counters['bad']} bad, {counters['unhealed']} unhealed"
            )
    return reasons, report


def load_runs(arm: str) -> list[tuple[str, dict]]:
    files = []
    boot = os.path.join(arm, "concurrency.json")
    if os.path.exists(boot):
        files.append(("boot", boot))
    repeats = glob.glob(os.path.join(arm, "conc", "run*.json"))
    repeats.sort(key=lambda f: int(os.path.basename(f)[3:-5]))
    files += [(os.path.basename(f)[:-5], f) for f in repeats]
    runs = []
    for name, path in files:
        with open(path) as fh:
            runs.append((name, json.load(fh)))
    return runs


def check_solo(runs: list[tuple[str, dict]]) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    seen: dict[str, set[tuple]] = defaultdict(set)
    keys = None
    solo2_items = solo2_equal = 0
    first_diff = None
    for name, d in runs:
        solo, solo2 = d.get("solo", {}), d.get("solo2", {})
        keys = set(solo) if keys is None else keys
        if set(solo) != keys or set(solo2) != keys:
            reasons.append(f"{name}: solo/solo2 items {sorted(set(solo2))} != {keys}")
        for key, item in solo.items():
            seen[key].add(signature(item))
        for key, item in solo2.items():
            solo2_items += 1
            ref = solo.get(key)
            if ref is not None and signature(ref) == signature(item):
                solo2_equal += 1
            elif first_diff is None:
                i = first_divergence(ref, item) if ref is not None else None
                first_diff = f"{name} item {key}, first token difference at {i}"
    unstable = sorted(k for k, s in seen.items() if len(s) > 1)
    if unstable:
        reasons.append(f"solo differs between runs for items {unstable}")
    if solo2_equal != solo2_items:
        reasons.append(
            f"solo2 differs from solo in {solo2_items - solo2_equal} of "
            f"{solo2_items} items (first: {first_diff})"
        )
    n = len(seen)
    report = [
        f"  solo: {n - len(unstable)}/{n} items bit-identical (tokens and logprobs) "
        f"in all {len(runs)} runs; solo2 equal to solo in {solo2_equal}/{solo2_items} "
        "items"
    ]
    return reasons, report


def flips(runs: list[tuple[str, dict]], fail_nats: float, rows: bool) -> list[str]:
    count: Counter[str] = Counter()
    detail: list[str] = []
    for name, d in runs:
        for scenario, ref_name in FLIP_SCENARIOS:
            for key, item in d.get(scenario, {}).items():
                ref = d.get(ref_name, {}).get(prompt_of(key))
                if ref is None:
                    continue
                count[f"{scenario} items"] += 1
                i = first_divergence(ref, item)
                if i is None:
                    continue
                count[f"{scenario} diverged"] += 1
                a, b = ref["tokens"][i], item["tokens"][i]
                prev = ref["tokens"][i - 1] if i else ""
                if {token_id(a), token_id(b), token_id(prev)} & set(SPECIAL):
                    count[f"{scenario} special"] += 1
                top = ref["top"][i] if i < len(ref.get("top", [])) else {}
                margin = top[a] - top[b] if a in top and b in top else None
                if margin is None or margin > fail_nats:
                    count[f"{scenario} confident"] += 1
                if rows:
                    shown = "not in top-5" if margin is None else f"{margin:.3f}"
                    detail.append(
                        f"    {name} {scenario} {key} token {i}: solo {a} got {b}, "
                        f"solo margin {shown}"
                    )
    lines = [
        f"  info, not a gate: {scenario} {count[f'{scenario} items']} items, "
        f"{count[f'{scenario} diverged']} diverge from the T=3 solo, "
        f"{count[f'{scenario} special']} at or after a special token, "
        f"{count[f'{scenario} confident']} confident (margin > {fail_nats} nats or "
        "outside the solo top-5)"
        for scenario, _ in FLIP_SCENARIOS
    ]
    return lines + detail


def probe_verdicts(arm: str) -> list[str]:
    boot = None
    try:
        with open(os.path.join(arm, "concurrency.txt"), errors="replace") as f:
            results = [ln.strip() for ln in f if ln.startswith("RESULT")]
        boot = results[-1].split(":")[0] if results else None
    except OSError:
        pass
    failed = total = 0
    try:
        with open(os.path.join(arm, "conc", "runs.tsv")) as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 3:
                    total += 1
                    failed += cols[2] != "0"
    except OSError:
        pass
    return [
        f"  info: the probe's own verdict (token flips against the T=3 solo): boot "
        f"{boot or 'n/a'}, repeats {failed} of {total} FAIL"
    ]


def judge(arm: str, require_digest: bool, fail_nats: float, rows: bool):
    """(verdict, lines): True PASS, False FAIL, None no probe results."""
    runs = load_runs(arm)
    if not runs:
        return None, [
            f"{VERDICT} NO DATA: no concurrency.json or conc/run*.json in {arm}"
        ]
    reasons, report = check_solo(runs)
    if require_digest:
        digest_reasons, digest_report = check_digest(arm)
        reasons += digest_reasons
        report += digest_report
    else:
        report.append("  digest: not required (--no-digest)")
    report += flips(runs, fail_nats, rows)
    report += probe_verdicts(arm)
    head = (
        f"{VERDICT} FAIL: " + "; ".join(reasons)
        if reasons
        else f"{VERDICT} PASS: solo and solo2 bit-identical in {len(runs)} runs"
        + (", hx digest clean on the hub and every server" if require_digest else "")
    )
    return not reasons, [head, *report]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arm", help="arms/NAME directory")
    ap.add_argument("--no-digest", action="store_true", help="do not require digest")
    ap.add_argument("--fail-nats", type=float, default=1.0)
    ap.add_argument("--rows", action="store_true", help="one line per flip")
    args = ap.parse_args()
    verdict, lines = judge(
        os.path.expanduser(args.arm), not args.no_digest, args.fail_nats, args.rows
    )
    print("\n".join(lines))
    return 2 if verdict is None else int(not verdict)


if __name__ == "__main__":
    sys.exit(main())
