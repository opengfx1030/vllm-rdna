#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Greedy probe of mixed prefill/decode batches at long context.

N distinct long prompts (several thousand tokens each) are sent at once with
``--max-tokens`` decode steps each. The scheduler prefills them in chunks, so
later arrivals' prefill chunks share steps with earlier requests' decode rows:
mixed batches with up to N-1 decode rows at long context. Each output is
compared with its own solo greedy run on the same server (prefix cache reset in
between) with the concurrency probe's near-tie logic. The 2026-09-09 report of
the GQA decode kernel ("garbage from the first decode token" in mixed batches
with two or more decode rows) shows up as a FAIL at token 0 to 2.

    python mixed_decode_probe.py --url http://127.0.0.1:P/v1/completions \
        --model m --reset-url http://127.0.0.1:P/reset_prefix_cache

The reset endpoint needs ``VLLM_SERVER_DEV_MODE=1`` on the server. Without
``--reset-url`` the mixed phase reuses the solo run's prefix cache and the
batches are no longer mixed. Under ``serve_validate.sh`` use it as a
``LOAD_HOOK`` (``URL``, ``PORT`` and ``SERVED`` are exported to the hook).
Exit status is 1 when any prompt fails.
"""

import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concurrency_probe as cp  # noqa: E402


def reset_cache(url):
    req = urllib.request.Request(url, b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reset-url", default="")
    ap.add_argument("-n", "--num-long", type=int, default=6)
    ap.add_argument("--sentences", type=int, default=600, help="about 20 tokens each")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--tie-nats", type=float, default=0.2)
    ap.add_argument("--fail-nats", type=float, default=1.0)
    ap.add_argument("--max-lp-drop", type=float, default=1.5)
    ap.add_argument("--repeat-lp-slack", type=float, default=0.5)
    ap.add_argument("--min-mean-lp", type=float, default=-2.5)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a.ignore_eos = True
    prompts = [cp.long_prompt(100 + k, a.sentences) for k in range(a.num_long)]
    t0 = time.monotonic()
    solo = {}
    for k, p in enumerate(prompts):
        solo[k] = cp.complete(a.url, a.model, p, a.max_tokens, True)
        print(
            f"solo {k}: {len(solo[k]['tokens'])} tokens, "
            f"mean lp {cp.mean(solo[k]['lps']):.2f} ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
    if a.reset_url:
        print("reset_prefix_cache:", reset_cache(a.reset_url), flush=True)
    else:
        print("WARNING: no --reset-url; the mixed phase reuses the solo prefix cache")
    together, err = cp.run_parallel(
        [(k, p, a.max_tokens, 0.0) for k, p in enumerate(prompts)],
        a.url,
        a.model,
        True,
    )
    fails = 0
    for k in range(a.num_long):
        if k in err:
            print(f"FAIL mixed {k}: {err[k]}")
            fails += 1
            continue
        status, detail = cp.compare(solo[k], together[k], a, None)
        got_lp = cp.mean(together[k]["lps"])
        if got_lp < a.min_mean_lp:
            status, detail = "FAIL", f"{detail}; output mean lp {got_lp:.2f}"
        garbage = cp.looks_like_garbage(together[k]["text"])
        if garbage and not cp.looks_like_garbage(solo[k]["text"]):
            status, detail = "FAIL", f"{detail}; garbage: {garbage}"
        fails += status == "FAIL"
        label = "ok" if status == "ok" else ("FAIL" if status == "FAIL" else "WARN")
        print(f"{label:4s} mixed {k}: {status} {detail}")
        if status == "FAIL" and a.verbose:
            print(f"   solo: {solo[k]['text']!r}\n   got:  {together[k]['text']!r}")
    print(
        f"RESULT {'PASS' if fails == 0 else 'FAIL'}: {fails} failing of "
        f"{a.num_long} ({time.monotonic() - t0:.0f}s)"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
