#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline: the draft vocabulary of the Inkling-2 DSpark draft (an explicit opt-in).

The greedy DSpark draft sweeps the target's whole LM head (922 MB) plus the folded
head (102 MB) for every proposal, which is half of its cost.  Restricting the draft
to the most frequent tokens shrinks both in proportion; a token outside the list is
never drafted, which can only cost acceptance (the target verifies every draft).

This tool ranks tokens by their frequency in text you give it (code, prose and chat
count equally), keeps the ``--size`` most frequent, always keeps the tokenizer's
special and added tokens (the chat template's structure tokens open and close every
reply, but no corpus has them), fills what the corpora never saw with the lowest
unused ids, and writes the ids ascending (so the draft's lowest-id tie rule is the
full-vocabulary one) as an ``.npy`` for

    --speculative-config '{..., "speculator_options": {"draft_vocab": "ids.npy"}}'

Rank on text that is not what you evaluate on; ``--heldout`` reports the share of
its tokens the list covers (the coverage bounds the acceptance loss: a position
whose target token is outside the list cannot be accepted).

    python draft_vocab.py --ckpt DIR --size 32768 --out ids.npy \\
        --code 'src/**/*.py' --prose 'docs/**/*.md' --chat calib.jsonl \\
        --heldout 'other/**/*.py'

With several ``--size`` values, ``--out`` holds ``{size}``.  ``--chat`` takes JSON
lines with a ``text`` string or ``messages`` (each ``content`` counts).
"""

import argparse
import glob
import json
import pathlib
import random
from collections.abc import Iterator

import numpy as np


def texts(patterns: list[str], jsonl: bool) -> Iterator[str]:
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    random.Random(0).shuffle(files)
    for name in files:
        try:
            if not jsonl:
                yield pathlib.Path(name).read_text(errors="ignore")[:200_000]
                continue
            with open(name, errors="ignore") as f:
                for line in f:
                    row = json.loads(line)
                    if "text" in row:
                        yield row["text"][:200_000]
                    for message in row.get("messages", []):
                        if isinstance(message.get("content"), str):
                            yield message["content"][:200_000]
        except OSError:
            continue


def count_tokens(
    tok, patterns: list[str], vocab: int, max_chars: int, jsonl: bool = False
) -> np.ndarray:
    counts = np.zeros(vocab, dtype=np.int64)
    chars = 0
    for text in texts(patterns, jsonl):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if ids:
            np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
            chars += len(text)
        if chars >= max_chars:
            break
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (the tokenizer)")
    ap.add_argument("--size", type=int, nargs="+", default=[32768], help="ids to keep")
    ap.add_argument("--out", required=True, help="the .npy ({size} for several sizes)")
    ap.add_argument("--code", nargs="*", default=[], help="globs of code files")
    ap.add_argument("--prose", nargs="*", default=[], help="globs of prose files")
    ap.add_argument("--chat", nargs="*", default=[], help="globs of JSON-lines chat")
    ap.add_argument("--heldout", nargs="*", default=[], help="globs to report on")
    ap.add_argument("--max-chars", type=int, default=30_000_000, help="per group")
    ap.add_argument("--score-cache", default="", help=".npy of the token scores")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    vocab = len(tok)
    if args.score_cache and pathlib.Path(args.score_cache).exists():
        score = np.load(args.score_cache)
    else:
        score = np.zeros(vocab, dtype=np.float64)
        for name, patterns, jsonl in (
            ("code", args.code, False),
            ("prose", args.prose, False),
            ("chat", args.chat, True),
        ):
            if patterns:
                counts = count_tokens(tok, patterns, vocab, args.max_chars, jsonl)
                print(f"{name}: {counts.sum()} tokens, {(counts > 0).sum()} distinct")
                score += counts / max(counts.sum(), 1)
        if args.score_cache:
            np.save(args.score_cache, score)
    if not score.any():
        raise SystemExit("no text: give --code, --prose and/or --chat")
    special = np.array(
        sorted({*tok.all_special_ids, *tok.get_added_vocab().values()}), dtype=np.int64
    )
    score[special] = np.inf
    seen = np.argsort(-score, kind="stable")[: int((score > 0).sum())]
    heldout = (
        count_tokens(tok, args.heldout, vocab, args.max_chars) if args.heldout else None
    )
    for size in args.size:
        keep = seen[:size]
        if keep.size < size:  # ids no corpus used: the lowest ones
            unused = np.setdiff1d(np.arange(vocab), keep, assume_unique=True)
            keep = np.concatenate([keep, unused[: size - keep.size]])
        ids = np.sort(keep).astype(np.int32)
        out = args.out.format(size=size)
        np.save(out, ids)
        print(f"wrote {ids.size} ids to {out} ({min(size, seen.size)} ranked)")
        if heldout is not None:
            covered = heldout[ids].sum() / max(heldout.sum(), 1)
            print(f"  held-out coverage: {100 * covered:.2f}% of {heldout.sum()} tokens")


if __name__ == "__main__":
    main()
