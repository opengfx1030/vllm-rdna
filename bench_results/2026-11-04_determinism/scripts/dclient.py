#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Score fixed 2048-token windows repeatedly with prompt_logprobs.

Each request: the window's token ids as the prompt, max_tokens=1,
temperature 0, prompt_logprobs=TOP (same as tools/topk_kl.py collect). One
request at a time. Saves per request the actual-token logprob and rank and the
top-1 id/logprob for positions 1..L-1, and prints how each repeat compares
with the first pass of the same window.

  dclient.py --url http://127.0.0.1:18460 --model flash-next \
      --windows windows.json --seq mix:28,mix:26 --repeats 3 --out run.npz
"""

import argparse
import json
import time
import urllib.request

import numpy as np


def post(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def parse(window, plp):
    n = len(window) - 1
    assert len(plp) == len(window) and plp[0] is None, len(plp)
    act_lp = np.empty(n, np.float64)
    act_rank = np.empty(n, np.int64)
    top1 = np.empty(n, np.int64)
    top1_lp = np.empty(n, np.float64)
    for j in range(n):
        e = plp[j + 1]
        a = e[str(window[j + 1])]
        act_lp[j] = a["logprob"]
        act_rank[j] = a["rank"]
        best = min(e.items(), key=lambda kv: (kv[1]["rank"], -kv[1]["logprob"]))
        top1[j] = int(best[0])
        top1_lp[j] = best[1]["logprob"]
    return act_lp, act_rank, top1, top1_lp


def compare(ref, cur):
    d = np.abs(cur[0] - ref[0])
    ident = float(np.mean(cur[0] == ref[0]))
    agree = float(np.mean(cur[2] == ref[2]))

    def first(th):
        idx = np.nonzero(d >= th)[0]
        return int(idx[0]) if idx.size else -1

    return (
        f"ident {ident * 100:5.1f}%  top1 {agree * 100:5.1f}%  max|d| {d.max():6.3f} "
        f"@{int(d.argmax())}  first>=0.1 {first(0.1)}  >=1 {first(1.0)}  "
        f"n>=1 {int((d >= 1).sum())}  n>=5 {int((d >= 5).sum())}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--seq", required=True, help="domain:index,... in request order")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--order", choices=["interleave", "block"], default="interleave")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.windows) as f:
        spec = json.load(f)
    url = args.url.rstrip("/") + "/v1/completions"
    seq = []
    for item in args.seq.split(","):
        dom, idx = item.split(":")
        seq.append((dom, int(idx)))
    if args.order == "interleave":
        plan = [(r, d, i) for r in range(args.repeats) for d, i in seq]
    else:
        plan = [(r, d, i) for d, i in seq for r in range(args.repeats)]

    first = {}
    results = []
    for k, (r, dom, idx) in enumerate(plan):
        window = spec["domains"][dom]["windows"][idx]
        t0 = time.time()
        out = post(
            url,
            {
                "model": args.model,
                "prompt": window,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": args.top,
            },
            args.timeout,
        )
        dt = time.time() - t0
        assert out["usage"]["prompt_tokens"] == len(window)
        cur = parse(window, out["choices"][0]["prompt_logprobs"])
        key = (dom, idx)
        msg = f"[{k:3d}] rep {r} {dom}:{idx:3d} nll {-cur[0].mean():.4f} {dt:6.2f}s"
        if key in first:
            msg += "  vs pass0: " + compare(first[key], cur)
        else:
            first[key] = cur
        print(msg, flush=True)
        results.append((r, dom, idx, dt) + cur)

    np.savez_compressed(
        args.out,
        rep=np.array([x[0] for x in results]),
        dom=np.array([x[1] for x in results]),
        idx=np.array([x[2] for x in results]),
        latency=np.array([x[3] for x in results]),
        act_lp=np.stack([x[4] for x in results]),
        act_rank=np.stack([x[5] for x in results]),
        top1=np.stack([x[6] for x in results]),
        top1_lp=np.stack([x[7] for x in results]),
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
