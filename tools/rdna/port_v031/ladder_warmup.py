#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm a server's capture ladder before measuring (stdlib only).

The first step of each new token count JIT-compiles / tunes on gfx1030 (up to
tens of seconds, and an rdna_ar "late peer" on the other ranks), which lands
in whichever probe or bench cell hits that size first. This sends one
prefill of exactly N prompt tokens per ladder size (fresh random token ids,
so the prefix cache cannot skip it), then decode batches of 1..max_seqs
concurrent requests, all with tiny outputs.

  python ladder_warmup.py --url http://127.0.0.1:18341 --model flash-next \
      --sizes 1,2,4,8,16,...,1032 --max-seqs 8
"""

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def complete(url: str, model: str, n_tokens: int, max_tokens: int, rng) -> float:
    ids = [rng.randrange(1000, 30000) for _ in range(max(1, n_tokens))]
    body = json.dumps(
        {
            "model": model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    sizes = sorted({int(s) for s in args.sizes.strip("[] ").split(",") if s})
    for n in sizes:
        dt = complete(args.url, args.model, n, 1, rng)
        print(f"prefill {n:>5} tok: {dt:6.2f} s", flush=True)
    for c in range(1, args.max_seqs + 1):
        t0 = time.monotonic()
        with ThreadPoolExecutor(c) as ex:
            list(
                ex.map(
                    lambda _: complete(
                        args.url, args.model, 64, args.decode_tokens, rng
                    ),
                    range(c),
                )
            )
        print(f"decode batch {c}: {time.monotonic() - t0:6.2f} s", flush=True)


if __name__ == "__main__":
    main()
