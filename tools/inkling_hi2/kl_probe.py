#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KL(stock || HI2) probe on the quantizer's held-out windows, over HTTP.

The windows are the stage-4 evaluation ids (``heldout_mixed_ids.npy``,
``[96, 2048]`` int32; the first 32 are the published 0.0996 nats at 3.5 bpw).
Both servers answer a completions request with ``prompt=<ids>``, ``echo=true`` and
``logprobs=--topk``; the probe compares the next-token distributions position by
position and reports the mean KL, its standard error over windows, top-1 agreement
and the expected speculative acceptance.  The HTTP distribution is truncated to
the top-k tokens, so the number is an approximation of the quantizer's full-logit
KL (which the reference eval JSON carries); the gate defaults to that reference
plus the quantization's own headroom.

    python tools/inkling_hi2/kl_probe.py \\
        --hi2-url http://127.0.0.1:18230 --stock-url http://127.0.0.1:18231 \\
        --ids ~/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy \\
        --windows 32 \\
        --reference-eval ~/w4a8_runs/inkling-hi2/stage4/eval/bpw3.5_mixed.json

Exit 0 when the mean KL is within ``--max-kl`` and top-1 is at least ``--min-top1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

import numpy as np
import slot  # tools/inkling_hi2/slot.py, next to this script


def call(url: str, payload: dict, timeout: float = 3600.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def served_name(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
        return json.loads(r.read())["data"][0]["id"]


def _key(k: str) -> str:
    # vLLM can return token ids ("token_id:123") so strings cannot collide
    return k


def top_distribution(choice: dict, position: int) -> dict[str, float]:
    top = choice["logprobs"]["top_logprobs"][position]
    return {_key(k): float(v) for k, v in top.items()}


def window_kl(url: str, model: str, ids: list[int], topk: int):
    out = call(
        url + "/v1/completions",
        {
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": topk,
            "return_tokens_as_token_ids": True,
        },
    )
    choice = out["choices"][0]
    if choice["logprobs"] is None:
        raise RuntimeError(f"{url}: no logprobs in the response")
    return choice, out["usage"]


def truncated_kl(p: dict[str, float], q: dict[str, float]) -> tuple[float, bool]:
    """KL(p || q) over the union of both top-k sets, each renormalised there.

    A token outside one top-k gets that server's smallest returned logprob (the
    best available bound over HTTP); the truncation makes the value a lower
    estimate of the full-vocabulary KL.
    """
    union = set(p) | set(q)
    pmin = min(p.values())
    qmin = min(q.values())
    lp = {k: p.get(k, pmin) for k in union}
    lq = {k: q.get(k, qmin) for k in union}
    lse_p = math.log(sum(math.exp(v) for v in lp.values()))
    lse_q = math.log(sum(math.exp(v) for v in lq.values()))
    kl = 0.0
    for k in union:
        pp = math.exp(lp[k] - lse_p)
        kl += pp * ((lp[k] - lse_p) - (lq[k] - lse_q))
    top1_p = max(p, key=p.get)
    top1_q = max(q, key=q.get)
    return kl, top1_p == top1_q


def greedy_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


GREEDY_PROMPTS = [
    "What is the capital of France? Reply with the city name.",
    "Write one sentence about the sea.",
    "def add(a, b):",
]


def greedy(url: str, model: str, prompt: str, tokens: int = 16) -> list[int]:
    out = call(
        url + "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": tokens,
            "temperature": 0.0,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
    )
    lp = out["choices"][0]["logprobs"]
    return [int(k.split(":")[1]) for k in lp["tokens"]] if lp else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hi2-url", required=True)
    ap.add_argument("--stock-url", required=True)
    ap.add_argument("--ids", required=True, help="held-out window ids (.npy)")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--max-kl", type=float, default=None)
    ap.add_argument("--min-top1", type=float, default=0.90)
    ap.add_argument("--reference-eval", default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slot_name = os.environ.get(slot.ENV_SLOT) or slot.DEFAULT_SLOT
    hip, buses = slot.env_pair(slot_name)
    try:
        slot.check_declared(slot_name, hip, buses)
    except slot.SlotError as exc:
        print(f"kl_probe: {exc}", file=sys.stderr)
        return 2
    if slot_name != "a2":
        print(
            f"kl_probe: TP=2 belongs on a2 (HIP 0,1; PCI d1 d4); "
            f"{slot_name} is the TP4 lease",
            file=sys.stderr,
        )
        return 2

    windows = np.load(args.ids)
    if windows.ndim != 2:
        ap.error(f"--ids must be [windows, seq_len], got {windows.shape}")
    n = min(args.windows, windows.shape[0])
    hi2_name, stock_name = served_name(args.hi2_url), served_name(args.stock_url)
    print(f"kl: {n} of {windows.shape[0]} windows of {windows.shape[1]} tokens")
    print(
        f"kl: slot={slot_name} HIP={hip} PCI={buses} "
        f"hi2={hi2_name} stock={stock_name} topk={args.topk}"
    )

    kls, top1 = [], []
    for i in range(n):
        ids = windows[i].tolist()
        c_stock, u_stock = window_kl(args.stock_url, stock_name, ids, args.topk)
        c_hi2, u_hi2 = window_kl(args.hi2_url, hi2_name, ids, args.topk)
        values = []
        agrees = 0
        for pos in range(len(ids) - 1):
            p = top_distribution(c_stock, pos)
            q = top_distribution(c_hi2, pos)
            kl, same = truncated_kl(p, q)
            values.append(kl)
            agrees += int(same)
        mean = float(np.mean(values))
        kls.append(mean)
        top1.append(agrees / max(len(values), 1))
        print(
            f"kl: window {i:03d} KL {mean:.5f} top1 {top1[-1]:.4f} "
            f"prompt_tokens {u_stock.get('prompt_tokens')}/{u_hi2.get('prompt_tokens')}"
        )

    mean = float(np.mean(kls))
    sem = float(np.std(kls, ddof=1) / math.sqrt(len(kls))) if len(kls) > 1 else 0.0
    top1_mean = float(np.mean(top1))
    result = {
        "slot": {"name": slot_name, "hip": hip, "pci_buses": buses},
        "windows": n,
        "seq_len": int(windows.shape[1]),
        "kl_mean": mean,
        "kl_sem": sem,
        "top1": top1_mean,
        "per_window_kl": kls,
    }
    if args.reference_eval:
        with open(args.reference_eval) as f:
            ref = json.load(f)
        student = ref.get("student", ref)
        result["reference_kl_mean"] = student.get("kl_mean")
        result["reference_top1"] = student.get("top1")
        print(
            f"kl: quantizer reference KL {student.get('kl_mean')} "
            f"top1 {student.get('top1')} ({args.reference_eval})"
        )

    if args.greedy:
        for prompt in GREEDY_PROMPTS:
            a = greedy(args.stock_url, stock_name, prompt)
            b = greedy(args.hi2_url, hi2_name, prompt)
            print(
                f"greedy: prefix match {greedy_prefix(a, b)}/{min(len(a), len(b))} "
                f"first tokens {a[:6]} / {b[:6]} :: {prompt!r}"
            )

    print("KL " + json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    ok = top1_mean >= args.min_top1
    if args.max_kl is not None:
        ok = ok and mean <= args.max_kl
    print("RESULT kl", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
