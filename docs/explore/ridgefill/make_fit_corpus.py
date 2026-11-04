#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit and held-out PG-19 windows for the RidgeFill capture, disjoint from the eval.

``ppl_long.py prepare`` cut the eval windows: one ``length``-token window per
PG-19 test book, in file order, ``skip`` tokens in. This script tokenizes the
books those windows do not use with the same tokenizer and conventions as
``ppl_long.prepare`` (``add_special_tokens=False``, an 8-characters-per-token
prefix first, the whole book when the prefix is short) and cuts consecutive
non-overlapping windows at ``skip + k * length``, at most ``--max-per-book``
per book. It first re-tokenizes every eval book and requires each eval
window's sha to match, which proves the conventions match.

Without ``--held-parquet`` the books that hold a window are split in file
order: the last ones (as many as ``--held-windows``, fewer if the fit would
run short) are held out, the rest are the fit books. The 36 books the 64 eval
windows leave in PG-19 test hold only 65 windows at three per book, short of
64 + 8, so the capture's held-out windows come from ``--held-parquet`` (the
PG-19 validation split): one window from each of its first books, in file
order, whose title is not an eval or fit title; every spare test book then
feeds the fit. Windows are listed round-robin over books (every book's first
window, then the second ones, ...), so any prefix of the fit file, such as
the capture's 8-prompt snapshot, spans as many books as it can.

Both files use the eval file's schema (``windows[i]``: book index in its
parquet, title, offset, ids, sha); ``meta`` adds the source ``parquet`` and
its sha256, ``role``, ``books`` and the eval file's sha. The script asserts
that eval, fit and held-out windows share no book (parquet sha and index), no
title and no window sha, and prints each set's book ids and titles.

    python3 make_fit_corpus.py --eval pg19_w64_l16384.json \\
        --parquet pg19-test/data/test-00000-of-00001-*.parquet \\
        --held-parquet pg19/data/validation-00000-of-00001-*.parquet \\
        --held-corpus "emozilla/pg19 validation" \\
        --fit-out pg19_fit_w64_l16384.json --held-out pg19_held_w8_l16384.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppl_long import ids_sha  # noqa: E402


def book_ids(tok, text: str, need: int) -> list[int]:
    """The book's first tokens, tokenized as ``ppl_long.prepare`` does.

    A prefix of ``need * 8`` characters first (books run to 4.5M characters);
    the whole book when that prefix holds fewer than ``need`` tokens.
    """
    chars = need * 8
    ids = tok(text[:chars], add_special_tokens=False)["input_ids"]
    if len(ids) < need and chars < len(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids


def round_robin(books: list[dict], count: int) -> list[dict]:
    """``count`` windows, every book's k-th window before any (k+1)-th."""
    out: list[dict] = []
    depth = max((len(b["windows"]) for b in books), default=0)
    for k in range(depth):
        for book in books:
            if k < len(book["windows"]) and len(out) < count:
                out.append(book["windows"][k])
    if len(out) < count:
        raise SystemExit(f"only {len(out)} windows available, need {count}")
    return out


def split_books(books: list[dict], fit: int, held: int) -> tuple[list, list]:
    """Hold out the last books in file order, keeping ``fit`` fit windows."""
    for count in range(min(held, len(books)), 0, -1):
        held_books = books[len(books) - count :]
        fit_books = books[: len(books) - count]
        if sum(len(b["windows"]) for b in held_books) < held:
            continue
        if sum(len(b["windows"]) for b in fit_books) >= fit:
            return fit_books, held_books
    raise SystemExit(
        f"{len(books)} books cannot hold {fit} fit and {held} held-out windows"
    )


def load_books(
    tok, path: str, *, skip: int, length: int, cap: int, exclude: set[int]
) -> tuple[list[dict], list[str], list[str]]:
    """Books of one parquet that hold a window, with up to ``cap`` windows each."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    titles = table.column("short_book_title").to_pylist()
    texts = table.column("text").to_pylist()
    need = skip + cap * length
    books = []
    for book, (title, text) in enumerate(zip(titles, texts)):
        if book in exclude:
            continue
        ids = book_ids(tok, text, need)
        count = min(cap, max(0, (len(ids) - skip) // length))
        print(f"  book {book:3d}: {len(ids):7d} tokens, {count} windows  {title}")
        if count == 0:
            continue
        windows = []
        for k in range(count):
            offset = skip + k * length
            win = ids[offset : offset + length]
            windows.append(
                dict(book=book, title=title, offset=offset, ids=win, sha=ids_sha(win))
            )
        books.append(dict(book=book, title=title, windows=windows))
    return books, titles, texts


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval", required=True, help="eval windows (ppl_long prepare)")
    p.add_argument("--parquet", required=True, help="the eval file's PG-19 parquet")
    p.add_argument(
        "--held-parquet",
        default="",
        help="held-out books from this parquet (one window per book) instead "
        "of the eval parquet's last spare books",
    )
    p.add_argument(
        "--held-corpus",
        default="emozilla/pg19 validation",
        help="corpus name recorded for --held-parquet",
    )
    p.add_argument("--tokenizer", default="", help="default: the eval file's")
    p.add_argument("--fit-windows", type=int, default=64)
    p.add_argument("--held-windows", type=int, default=8)
    p.add_argument("--max-per-book", type=int, default=3)
    p.add_argument("--fit-out", required=True)
    p.add_argument("--held-out", required=True)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    eval_path = Path(args.eval)
    eval_bytes = eval_path.read_bytes()
    spec = json.loads(eval_bytes)
    meta = spec["meta"]
    length, skip = int(meta["length"]), int(meta["skip"])
    parquet_sha = sha256(args.parquet)
    if parquet_sha != meta["parquet_sha256"]:
        raise SystemExit(f"{args.parquet} is not the eval file's parquet")
    tokenizer = args.tokenizer or meta["tokenizer"]
    tok = AutoTokenizer.from_pretrained(tokenizer)

    eval_ids = {int(w["book"]) for w in spec["windows"]}
    print(f"spare books of {args.parquet}:")
    books, titles, texts = load_books(
        tok,
        args.parquet,
        skip=skip,
        length=length,
        cap=args.max_per_book,
        exclude=eval_ids,
    )
    # Same tokenizer, same conventions: every eval window must reproduce.
    for win in spec["windows"]:
        book = int(win["book"])
        ids = book_ids(tok, texts[book], skip + length)
        got = ids_sha(ids[skip : skip + length])
        if got != win["sha"] or titles[book] != win["title"]:
            raise SystemExit(f"eval book {book} does not reproduce ({got})")
    print(f"eval: {len(spec['windows'])} windows reproduce (tokenizer {tokenizer})")

    own_path = str(Path(args.parquet).resolve())
    sources = {"eval": (own_path, parquet_sha), "fit": (own_path, parquet_sha)}
    if args.held_parquet:
        fit = round_robin(books, args.fit_windows)
        print(f"held-out books of {args.held_parquet}:")
        held_books, _, _ = load_books(
            tok, args.held_parquet, skip=skip, length=length, cap=1, exclude=set()
        )
        # A book can sit in two splits under different indices: titles decide.
        taken = {w["title"] for w in spec["windows"]} | {w["title"] for w in fit}
        held_books = [b for b in held_books if b["title"] not in taken]
        held = round_robin(held_books, args.held_windows)
        sources["held"] = (
            str(Path(args.held_parquet).resolve()),
            sha256(args.held_parquet),
        )
    else:
        fit_books, held_books = split_books(books, args.fit_windows, args.held_windows)
        fit = round_robin(fit_books, args.fit_windows)
        held = round_robin(held_books, args.held_windows)
        sources["held"] = (own_path, parquet_sha)

    # Disjointness by (parquet, book index), by title and by window sha.
    roles = {"eval": spec["windows"], "fit": fit, "held": held}
    groups = {
        role: (
            {(sources[role][1], int(w["book"])) for w in windows},
            {w["title"] for w in windows},
            {w["sha"] for w in windows},
        )
        for role, windows in roles.items()
    }
    for role, windows in roles.items():
        if len(groups[role][2]) != len(windows):
            raise SystemExit(f"a {role} window repeats")
        used: dict[int, str] = {}
        for w in windows:
            used.setdefault(int(w["book"]), w["title"])
        print(
            f"{role}: {len(windows)} windows from {len(used)} books of "
            f"{sources[role][0]} (sha256 {sources[role][1]})"
        )
        print(f"{role} book ids: {sorted(used)}")
        for book in sorted(used):
            print(f"  {role} book {book:3d}: {used[book]}")
    for a, b in (("eval", "fit"), ("eval", "held"), ("fit", "held")):
        common = [sorted(x & y) for x, y in zip(groups[a], groups[b])]
        print(
            f"disjoint {a}/{b}: common books {common[0]} titles {common[1]} "
            f"window shas {common[2]}"
        )
        assert not any(common), (a, b, common)
    print("disjointness: PASS (eval/fit/held share no book, title or window sha)")

    for role, windows, out in (
        ("fit", fit, args.fit_out),
        ("held", held, args.held_out),
    ):
        path, digest = sources[role]
        own = path == own_path
        role_meta = dict(
            corpus=meta["corpus"] if own else args.held_corpus,
            parquet=str(path),
            parquet_sha256=digest,
            tokenizer=str(tokenizer),
            length=length,
            skip=skip,
            role=role,
            max_per_book=args.max_per_book if own else 1,
            books=sorted({int(w["book"]) for w in windows}),
            eval_windows=str(eval_path),
            eval_sha256=hashlib.sha256(eval_bytes).hexdigest(),
        )
        Path(out).write_text(json.dumps(dict(meta=role_meta, windows=windows)))
        print(f"{role}: {len(windows)} windows of {length} tokens -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
