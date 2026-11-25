#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KL(stock || HI2) probe on the quantizer's held-out windows, over HTTP.

The windows are the stage-4 evaluation ids (``heldout_mixed_ids.npy``,
``[96, 2048]`` int32; the first 32 are the published 0.0996 nats at 3.5 bpw).
Both servers answer a completions request with ``prompt=<ids>``, ``echo=true``,
``temperature=1`` and ``logprobs=--topk``; the probe scores the next-token
distributions from position 1 (position 0 predicts the first token with no
context, and vLLM returns ``None`` there), skips any position whose top-k entry a
server omitted, and caps the skipped share at ``--max-skipped-share`` (default 5 %
of a window).  It reports the mean KL, its standard error over windows, top-1
agreement and the expected speculative acceptance.  The served number is a top-k
truncated KL on the fp16 path (raw post-softcap logits, no renormalisation), so it
approximates rather than equals the quantizer's float32 full-logit eval: the
reference is 0.0996 nats and top-1 93.0 % on the 32 mixed windows, and the gate
(``--max-kl 0.15`` / ``--min-top1 0.90``) leaves that tolerance.

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


def top_distribution(choice: dict, position: int) -> dict[str, float] | None:
    """Top-k logprob dict at ``position``, or None when the server omitted it.

    vLLM returns ``None`` for the first prompt position (no context) and can omit
    an entry entirely; the caller skips those positions and counts them.
    """
    logprobs = choice.get("logprobs") or {}
    tops = logprobs.get("top_logprobs") or []
    if position >= len(tops) or tops[position] is None:
        return None
    return {_key(k): float(v) for k, v in tops[position].items()}


def top_entry_size(choice: dict) -> int:
    """Number of top-k entries the server returned (0 if it ignored logprobs).

    A response with fewer than the requested ``logprobs`` entries means the served
    KL is truncated further than asked, so the probe prints and records this.
    """
    tops = (choice.get("logprobs") or {}).get("top_logprobs") or []
    return next((len(t) for t in tops if t), 0)


def score_window(
    c_stock: dict, c_hi2: dict, seq_len: int
) -> tuple[list[float], int, int, int]:
    """Per-position KLs, top-1 agreements, skipped positions, considered positions.

    Positions are scored from 1: position 0 predicts the first prompt token with no
    context and its top-k entry is None on vLLM.  A position where either server
    omitted the entry is skipped and counted (the caller caps the share).
    """
    values, agrees, skipped = [], 0, 0
    for pos in range(1, seq_len):
        p = top_distribution(c_stock, pos)
        q = top_distribution(c_hi2, pos)
        if p is None or q is None:
            skipped += 1
            continue
        kl, same = truncated_kl(p, q)
        values.append(kl)
        agrees += int(same)
    return values, agrees, skipped, seq_len - 1


def window_kl(url: str, model: str, ids: list[int], topk: int):
    out = call(
        url + "/v1/completions",
        {
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            # T=1 scores the raw, post-softcap logits the quantizer's eval used;
            # logprobs are returned truncated to the top-k, not renormalised.
            "temperature": 1.0,
            "echo": True,
            "logprobs": topk,
            "return_tokens_as_token_ids": True,
        },
    )
    choice = out["choices"][0]
    if choice.get("logprobs") is None:
        raise RuntimeError(f"{url}: no logprobs in the response (echo=True?)")
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
    ap.add_argument(
        "--max-skipped-share",
        type=float,
        default=0.05,
        help="fail a window if more of its positions lack a top-k distribution",
    )
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
    skipped_total, considered_total = 0, 0
    topk_entries = None
    for i in range(n):
        ids = windows[i].tolist()
        c_stock, u_stock = window_kl(args.stock_url, stock_name, ids, args.topk)
        c_hi2, u_hi2 = window_kl(args.hi2_url, hi2_name, ids, args.topk)
        if i == 0:
            topk_entries = {
                "stock": top_entry_size(c_stock),
                "hi2": top_entry_size(c_hi2),
            }
            print(
                f"kl: top entries per position stock={topk_entries['stock']} "
                f"hi2={topk_entries['hi2']} (requested {args.topk})"
            )
        values, agrees, skipped, considered = score_window(c_stock, c_hi2, len(ids))
        if not values:
            raise RuntimeError(f"kl: window {i} has no usable prompt logprobs")
        if skipped > args.max_skipped_share * considered:
            raise RuntimeError(
                f"kl: window {i} skipped {skipped}/{considered} positions "
                f"(> {args.max_skipped_share:.0%})"
            )
        skipped_total += skipped
        considered_total += considered
        mean = float(np.mean(values))
        kls.append(mean)
        top1.append(agrees / len(values))
        print(
            f"kl: window {i:03d} KL {mean:.5f} top1 {top1[-1]:.4f} "
            f"skipped {skipped}/{considered} "
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
        "skipped_share": skipped_total / max(considered_total, 1),
        "topk_entries": topk_entries,
        "note": (
            "served KL: top-k truncated, temperature 1, post-softcap fp16 path; "
            "the reference is the quantizer's float32 full-logit eval"
        ),
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
