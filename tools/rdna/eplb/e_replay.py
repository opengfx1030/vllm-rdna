# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay fixed prompts at c=1 against a server that runs the EPLB counters.

The cross-rank top-k disagreement study (docs/rdna2/eplb-lean-design-2026-11-16.md,
section 11) needs the SAME token ids over and over with the prefix cache unable to
skip the prefill: every request carries its own ``cache_salt``, so the block hashes
differ while the tokens do not. Requests run strictly one after another (like
``bench serve --max-concurrency 1``), grouped by prompt length:

    --groups "mk:777:3,g1024:1024:24,g2048:2048:24,t1024:1024:12:txt"

``name:length:count[:kind]``; ``rnd`` (default) takes the first ``length`` ids of one
seeded random stream (so a shorter prompt is a prefix of a longer one), ``txt`` the
first ``length`` tokens of a text file (``--text-file``, tokenised with the model's
tokenizer). After the last group it sends one-token filler requests until every rank
file of ``--stats-dir`` carries a snapshot written after the last group (the counters
drain on the worker's step tick, so an idle server writes nothing) and copies the four
files to ``<out>/dump``.

Writes ``<out>/requests.jsonl`` (one line per request), ``<out>/replay.json`` (the
groups, the prefix-cache counters per group, the ring size at the end) and prints a
last line ``PASS`` or ``FAIL`` for ``serve_validate.sh`` (LOAD_HOOK).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

TOKEN_LOW, TOKEN_HIGH = 1000, 100000  # well inside the vocabulary, no special ids


def parse_groups(spec: str) -> list[dict[str, Any]]:
    groups = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"group {item!r}: want name:length:count[:kind]")
        kind = parts[3] if len(parts) == 4 else "rnd"
        if kind not in ("rnd", "txt"):
            raise ValueError(f"group {item!r}: kind must be rnd or txt")
        groups.append(
            {
                "name": parts[0],
                "length": int(parts[1]),
                "count": int(parts[2]),
                "kind": kind,
            }
        )
    return groups


def random_stream(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(TOKEN_LOW, TOKEN_HIGH, size=n).tolist()


def text_stream(path: str, tokenizer: str, n: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids: list[int] = []
    while len(ids) < n:
        ids += tok(text, add_special_tokens=False)["input_ids"]
    return ids[:n]


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def metric_total(base: str, name: str) -> float:
    """Sum of a Prometheus counter over its label sets (0.0 when absent)."""
    with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
        text = resp.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.rsplit(" ", 1)[1])
    return total


def snapshot_times(stats_dir: str) -> list[float]:
    """``meta['time']`` of every rank file (empty list while one is missing)."""
    times = []
    for path in sorted(glob.glob(os.path.join(stats_dir, "eplb_stats-rank*-pid*.npz"))):
        try:
            with np.load(path) as z:
                times.append(float(json.loads(str(z["meta"]))["time"]))
        except (OSError, ValueError, KeyError):
            return []
    return times


def flush_snapshot(
    base: str, model: str, stats_dir: str, since: float, ranks: int, timeout_s: float
) -> bool:
    """Step the engine with one-token requests until all ranks wrote after ``since``."""
    t_end = time.time() + timeout_s
    i = 0
    while time.time() < t_end:
        times = snapshot_times(stats_dir)
        if len(times) >= ranks and min(times) > since:
            return True
        i += 1
        post(
            base + "/v1/completions",
            {
                "model": model,
                "prompt": random_stream(8, 900000 + i),
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": f"flush-{os.getpid()}-{i}",
            },
        )
        time.sleep(2.0)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", help="tokenizer path (txt groups only)")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--text-file", help="source text of the txt groups")
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--flush-timeout", type=float, default=150.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    groups = parse_groups(args.groups)
    os.makedirs(args.out, exist_ok=True)
    longest = max(g["length"] for g in groups)
    rnd = random_stream(longest, args.seed)
    txt: list[int] | None = None
    if any(g["kind"] == "txt" for g in groups):
        if not (args.tokenizer and args.text_file):
            raise SystemExit("txt groups need --tokenizer and --text-file")
        txt = text_stream(args.text_file, args.tokenizer, longest)

    url = args.base_url + "/v1/completions"
    t_start = time.time()
    pid = os.getpid()
    failed = 0
    outputs: dict[str, set[str]] = {}
    summary: list[dict[str, Any]] = []
    with open(os.path.join(args.out, "requests.jsonl"), "w") as log:
        for g in groups:
            ids = (txt if g["kind"] == "txt" and txt is not None else rnd)[
                : g["length"]
            ]
            queried0 = metric_total(args.base_url, "vllm:prompt_tokens_total")
            cached0 = metric_total(args.base_url, "vllm:prompt_tokens_cached_total")
            lat: list[float] = []
            for i in range(g["count"]):
                t0 = time.time()
                rec: dict[str, Any] = {
                    "group": g["name"],
                    "i": i,
                    "length": g["length"],
                }
                try:
                    resp = post(
                        url,
                        {
                            "model": args.model,
                            "prompt": ids,
                            "max_tokens": args.max_tokens,
                            "temperature": 0,
                            "ignore_eos": True,
                            "cache_salt": f"e-{args.seed}-{g['name']}-{i}-{pid}",
                        },
                    )
                    text = resp["choices"][0]["text"]
                    rec["completion_tokens"] = resp["usage"]["completion_tokens"]
                    rec["text_sha1"] = hashlib.sha1(text.encode()).hexdigest()[:12]
                    outputs.setdefault(g["name"], set()).add(rec["text_sha1"])
                except (urllib.error.URLError, KeyError, ValueError) as exc:
                    failed += 1
                    rec["error"] = repr(exc)
                rec["t0"], rec["t1"] = t0, time.time()
                lat.append(rec["t1"] - t0)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            queried = metric_total(args.base_url, "vllm:prompt_tokens_total") - queried0
            cached = (
                metric_total(args.base_url, "vllm:prompt_tokens_cached_total") - cached0
            )
            entry = {
                **g,
                "prompt_tokens": queried,
                "cached_tokens": cached,
                "distinct_outputs": len(outputs.get(g["name"], ())),
                "median_s": float(np.median(lat)),
            }
            summary.append(entry)
            print(
                f"[{g['name']}] {g['count']} x {g['length']} tokens ({g['kind']}): "
                f"median {entry['median_s']:.2f} s, cached tokens {cached:.0f} of "
                f"{queried:.0f}, distinct outputs {entry['distinct_outputs']}",
                flush=True,
            )
    t_end = time.time()
    flushed = flush_snapshot(
        args.base_url, args.model, args.stats_dir, t_end, args.ranks, args.flush_timeout
    )
    dump = os.path.join(args.out, "dump")
    os.makedirs(dump, exist_ok=True)
    for path in glob.glob(os.path.join(args.stats_dir, "eplb_stats-rank*-pid*.npz")):
        shutil.copy2(path, dump)
    with open(os.path.join(args.out, "replay.json"), "w") as f:
        json.dump(
            {
                "groups": summary,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "t_start": t_start,
                "t_end": t_end,
                "flushed": flushed,
                "failed_requests": failed,
            },
            f,
            indent=1,
        )
    leaked = sum(s["cached_tokens"] for s in summary)
    ok = flushed and not failed and leaked == 0
    print(
        f"{'PASS' if ok else 'FAIL'} replay: {sum(g['count'] for g in groups)} "
        f"requests, {failed} failed, flushed={flushed}, prefix-cache tokens served "
        f"{leaked:.0f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
