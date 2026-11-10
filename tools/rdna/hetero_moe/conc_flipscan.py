#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where do the concurrency runs of an arm diverge from the solo reference?

Input: arm directories of ``tools/rdna/tp8pp2/driver.sh`` (``arms/NAME``) whose
``conc/runN.json`` were written by ``tools/rdna/port_v031/concurrency_probe.py``
(``CONC_REPEAT=N``). Output, per arm:

* the scenarios ``solo2``, ``together`` and ``churn`` against ``solo``: how many
  items diverged, how many of those at a special token or right after one
  (``<|im_start|>``, ``<|im_end|>``, ``<|endoftext|>``: out-of-distribution
  positions after an ignored EOS, where the model amplifies rounding noise), how
  many are confident (solo-top margin above ``--fail-nats`` or the other token
  outside the solo top-5: the probe's FAIL class);
* the cross-run identity of every (scenario, item): the number of distinct token
  sequences over the runs of the arm, and of distinct (tokens, logprobs) pairs.
  1 means bit-identical in every run: a deterministic stack shows 1 for
  ``solo``/``solo2`` and (with a deterministic schedule) for ``together`` and
  ``churn``, and any perturbation of one layer's output (a corrupted transfer
  too) moves some logprob; the run-to-run noise of the default MoE epilogues
  (``VLLM_RDNA2_MOE_DETERMINISTIC=0``) shows more.

``--rows`` also prints one line per divergence. Read-only.

    python tools/rdna/hetero_moe/conc_flipscan.py ARMS/h1v-int-concdig [...]
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

SPECIAL = {248044: "<|endoftext|>", 248045: "<|im_start|>", 248046: "<|im_end|>"}
SCENARIOS = ("solo2", "together", "churn")


def tok(token: str) -> int:
    return int(token.split(":")[1])


def name(token: str) -> str:
    n = tok(token)
    return SPECIAL.get(n, str(n))


def prompt_of(key: str) -> str:
    """Prompt index of an item: churn items are keyed ``(slot, prompt)``."""
    return key.strip("()").split(", ")[1] if key.startswith("(") else key


def first_divergence(solo: dict, got: dict) -> int | None:
    a, b = solo["tokens"], got["tokens"]
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def scan_arm(arm: str, fail_nats: float, rows: bool) -> None:
    files = sorted(
        glob.glob(os.path.join(arm, "conc", "run*.json")),
        key=lambda f: int(os.path.basename(f)[3:-5]),
    )
    if not files:
        print(f"{arm}: no conc/run*.json")
        return
    count: Counter[str] = Counter()
    seqs: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    lpseqs: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    for f in files:
        run = os.path.basename(f)
        with open(f) as fh:
            d = json.load(fh)
        solo = d["solo"]
        for key, e in solo.items():
            seqs[("solo", key)].add(tuple(e["tokens"]))
            lpseqs[("solo", key)].add((tuple(e["tokens"]), tuple(e["lps"])))
        for scenario in SCENARIOS:
            for key, e in d.get(scenario, {}).items():
                count[f"{scenario} items"] += 1
                seqs[(scenario, key)].add(tuple(e["tokens"]))
                lpseqs[(scenario, key)].add((tuple(e["tokens"]), tuple(e["lps"])))
                ref = solo[prompt_of(key)]
                i = first_divergence(ref, e)
                if i is None:
                    continue
                count[f"{scenario} diverged"] += 1
                a, b = ref["tokens"][i], e["tokens"][i]
                prev = ref["tokens"][i - 1] if i else "token_id:-1"
                special = {tok(a), tok(b), tok(prev)} & set(SPECIAL)
                if special:
                    count[f"{scenario} diverged at/after a special token"] += 1
                top = ref["top"][i]
                margin = top[a] - top[b] if a in top and b in top else None
                confident = margin is None or margin > fail_nats
                if confident:
                    count[f"{scenario} confident (FAIL class)"] += 1
                if rows:
                    m = "not in top-5" if margin is None else f"{margin:.3f}"
                    print(
                        f"  {run} {scenario} {key} token {i}: solo {name(a)} "
                        f"(prev {name(prev)}) got {name(b)}, solo margin {m}, "
                        f"lp solo {ref['lps'][i]:.3f} got {e['lps'][i]:.3f}"
                    )
    print(f"{arm}: {len(files)} runs")
    for scenario in SCENARIOS:
        items = count[f"{scenario} items"]
        print(
            f"  {scenario:9s} items {items:4d}  diverged from solo "
            f"{count[f'{scenario} diverged']:4d}  at/after special "
            f"{count[f'{scenario} diverged at/after a special token']:3d}  "
            f"confident {count[f'{scenario} confident (FAIL class)']:3d}"
        )
    for scenario in ("solo", *SCENARIOS):
        cells = [s for (sc, _), s in seqs.items() if sc == scenario]
        lp_cells = [s for (sc, _), s in lpseqs.items() if sc == scenario]
        print(
            f"  cross-run identity {scenario:9s}: {sum(len(s) == 1 for s in cells)}"
            f"/{len(cells)} items gave one token sequence in every run, "
            f"{sum(len(s) == 1 for s in lp_cells)}/{len(lp_cells)} also bit-identical "
            f"logprobs"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+", help="arms/NAME directories")
    ap.add_argument("--fail-nats", type=float, default=1.0)
    ap.add_argument("--rows", action="store_true", help="one line per divergence")
    args = ap.parse_args()
    for arm in args.arms:
        scan_arm(os.path.expanduser(arm), args.fail_nats, args.rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
