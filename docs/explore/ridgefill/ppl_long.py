#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced perplexity on long PG-19 windows, scored on the exact tail.

The 261-token ``ppl_probe`` paragraph moves by 0.06 between two boots of the
same arm, so it cannot resolve a map-precision change. This scores
``windows x (tail - 1)`` tokens (64 x 2047 by default) and reports the arm
difference with a paired interval, which is the noise floor of the A/B.

1. ``prepare`` (needs ``transformers`` and ``pyarrow``): one window of
   ``--length`` tokens per PG-19 test book, in file order, starting
   ``--skip`` tokens in (past the front matter); shorter books are skipped.
2. ``run`` (stdlib): each window is one token-ID prompt with ``echo: true,
   max_tokens: 1, logprobs: 1``; the server returns every prompt token's
   logprob given its prefix. RidgeFill fills the rows before
   ``length - tail``, and row ``i - 1`` predicts token ``i``, so only tokens
   ``length - tail + 1 .. length - 1`` are scored (exact rows only). Run with
   prefix caching off.
3. ``compare A.json B.json``: both perplexities, B - A, and a 95% interval
   from the per-window mean-NLL deltas (normal approximation; every window
   scores the same number of tokens).

    python3 ppl_long.py prepare --tokenizer <snapshot> \
        --parquet pg19-test/data/test-00000-of-00001-*.parquet --out win.json
    python3 ppl_long.py run --windows win.json --model flash-next \
        --url http://127.0.0.1:18096/v1/completions --out arm.json
    python3 ppl_long.py compare exact.json arm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def prepare(args) -> int:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = Path(args.parquet).read_bytes()
    table = pq.read_table(args.parquet)
    titles = table.column("short_book_title").to_pylist()
    texts = table.column("text").to_pylist()
    need = args.skip + args.length
    windows = []
    for book, (title, text) in enumerate(zip(titles, texts)):
        # Tokenize a prefix only: books run to 4.5M characters.
        chars = need * 8
        ids = tok(text[:chars], add_special_tokens=False)["input_ids"]
        if len(ids) < need and chars < len(text):
            ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < need:
            continue
        win = ids[args.skip : need]
        windows.append(
            dict(book=book, title=title, offset=args.skip, ids=win, sha=ids_sha(win))
        )
        if len(windows) == args.windows:
            break
    if len(windows) < args.windows:
        print(f"only {len(windows)} books hold {need} tokens", file=sys.stderr)
        return 1
    meta = dict(
        corpus="emozilla/pg19-test",
        parquet_sha256=hashlib.sha256(data).hexdigest(),
        tokenizer=str(args.tokenizer),
        length=args.length,
        skip=args.skip,
    )
    Path(args.out).write_text(json.dumps(dict(meta=meta, windows=windows)))
    print(f"{len(windows)} windows of {args.length} tokens -> {args.out}")
    return 0


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run(args) -> int:
    spec = json.loads(Path(args.windows).read_text())
    length = spec["meta"]["length"]
    if not 1 < args.tail < length:
        print(f"--tail {args.tail} must be in (1, {length})", file=sys.stderr)
        return 1
    first = length - args.tail + 1
    rows = []
    t_all = time.time()
    for n, win in enumerate(spec["windows"]):
        body = dict(
            model=args.model,
            prompt=win["ids"],
            max_tokens=1,
            temperature=0,
            echo=True,
            logprobs=1,
        )
        for attempt in (1, 2):
            t0 = time.time()
            try:
                out = post(args.url, body, args.timeout)
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise
                print(f"window {n}: {e}; retrying", file=sys.stderr)
        lp = out["choices"][0]["logprobs"]["token_logprobs"]
        if len(lp) < length:
            print(f"window {n}: {len(lp)} logprobs < {length}", file=sys.stderr)
            return 1
        scored = lp[first:length]
        if any(v is None or not math.isfinite(v) for v in scored):
            print(f"window {n}: non-finite tail logprob", file=sys.stderr)
            return 1
        nll = -sum(scored)
        rows.append(
            dict(
                book=win["book"],
                sha=win["sha"],
                nll=nll,
                tokens=len(scored),
                seconds=time.time() - t0,
            )
        )
        print(
            f"window {n:3d} book {win['book']:3d}: tail ppl "
            f"{math.exp(nll / len(scored)):8.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
    total = sum(r["nll"] for r in rows)
    count = sum(r["tokens"] for r in rows)
    result = dict(
        meta=dict(
            spec["meta"],
            url=args.url,
            model=args.model,
            tail=args.tail,
            label=args.label,
            seconds=time.time() - t_all,
        ),
        windows=rows,
        ppl=math.exp(total / count),
        tokens=count,
    )
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"PPL {result['ppl']:.4f} over {count} tokens, {len(rows)} windows")
    return 0


def compare(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    wa, wb = a["windows"], b["windows"]
    if [w["sha"] for w in wa] != [w["sha"] for w in wb]:
        print("the two runs scored different windows", file=sys.stderr)
        return 1
    if a["meta"]["tail"] != b["meta"]["tail"]:
        print("the two runs used different tails", file=sys.stderr)
        return 1
    d = [y["nll"] / y["tokens"] - x["nll"] / x["tokens"] for x, y in zip(wa, wb)]
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    pa, pb = a["ppl"], b["ppl"]
    lo = pa * (math.exp(mean - 1.96 * se) - 1.0)
    hi = pa * (math.exp(mean + 1.96 * se) - 1.0)
    worse = sum(1 for v in d if v > 0)
    print(
        json.dumps(
            dict(
                a=dict(label=a["meta"].get("label"), ppl=pa, tokens=a["tokens"]),
                b=dict(label=b["meta"].get("label"), ppl=pb, tokens=b["tokens"]),
                delta_ppl=pb - pa,
                delta_ppl_95=[lo, hi],
                delta_nll_per_token=mean,
                delta_nll_se=se,
                windows=len(d),
                windows_b_worse=worse,
            ),
            indent=1,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--tokenizer", required=True)
    pp.add_argument("--parquet", required=True)
    pp.add_argument("--windows", type=int, default=64)
    pp.add_argument("--length", type=int, default=16384)
    pp.add_argument("--skip", type=int, default=2048)
    pp.add_argument("--out", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--windows", required=True)
    pr.add_argument("--url", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--tail", type=int, default=2048)
    pr.add_argument("--label", default="")
    pr.add_argument("--timeout", type=float, default=900.0)
    pr.add_argument("--out", required=True)
    pc = sub.add_parser("compare")
    pc.add_argument("a")
    pc.add_argument("b")
    args = p.parse_args()
    return {"prepare": prepare, "run": run, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
