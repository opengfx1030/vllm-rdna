#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decode-path top-k KL between served configurations.

``topk_kl.py`` and ``quality_ab.py`` score prompts, i.e. the prefill path. A
change that only acts across decode steps (a lossy recurrent-state format such
as the LeapQuant GDN state, which a single-chunk prefill never stores) is
invisible there. This tool greedy-generates from fixed passages with the
top-k logprobs of every generated token. Up to the first token where two runs
disagree, both runs decoded the same context, so the per-position KL there is a
decode-path KL; the divergence point itself is reported, and a second run of
the reference configuration (``--floor``) gives the noise floor of both.

    python decode_kl.py collect --url http://127.0.0.1:18096 --model flash-next \\
        --out A.json [--passages 32] [--passage-chars 6000] [--gen-tokens 512] \\
        [--workers 8]
    python decode_kl.py compare A.json B.json [--floor A2.json] [--json out.json]

``--workers`` requests run concurrently, so decode batches hold several
requests as in serving. KL(ref || cand) is taken on the partition of the
vocabulary into the tokens of both top-k lists plus one cell for the rest
(a lower bound on the full KL, as in ``topk_kl.py``).
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_passages(files: list[str], count: int, chars: int) -> list[str]:
    text = ""
    for path in sorted(files):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text += handle.read() + "\n\n"
    if len(text) < count * chars:
        raise SystemExit(
            f"need {count * chars} characters of text, the files hold {len(text)}"
        )
    stride = len(text) // count
    return [text[i * stride : i * stride + chars] for i in range(count)]


def token_id(tok: str) -> int:
    # return_tokens_as_token_ids renders tokens as "token_id:<id>".
    return int(tok.rsplit(":", 1)[1])


def generate(args: argparse.Namespace, prompt: str) -> dict:
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.gen_tokens,
        "temperature": 0.0,
        "seed": 12345,
        "ignore_eos": True,
        "logprobs": args.top,
        "return_tokens_as_token_ids": True,
    }
    url = args.url.rstrip("/")
    if not url.endswith("/v1/completions"):
        url += "/v1/completions"
    out = post(url, body, args.timeout)
    lp = out["choices"][0]["logprobs"]
    ids = [token_id(t) for t in lp["tokens"]]
    top = [
        {str(token_id(t)): v for t, v in (d or {}).items()} for d in lp["top_logprobs"]
    ]
    return {
        "ids": ids,
        "lp": lp["token_logprobs"],
        "top": top,
        "prompt_tokens": out["usage"]["prompt_tokens"],
    }


def cmd_collect(args: argparse.Namespace) -> int:
    files = args.files or glob.glob(
        os.path.join(TREE, "docs", "**", "*.md"), recursive=True
    )
    passages = load_passages(files, args.passages, args.passage_chars)
    with ThreadPoolExecutor(args.workers) as pool:
        runs = list(pool.map(lambda p: generate(args, p), passages))
    meta = {
        "url": args.url,
        "passages": args.passages,
        "passage_chars": args.passage_chars,
        "gen_tokens": args.gen_tokens,
        "top": args.top,
        "workers": args.workers,
    }
    with open(args.out, "w") as handle:
        json.dump({"meta": meta, "runs": runs}, handle)
    print(
        f"collect: {len(runs)} passages, prompt tokens "
        f"{min(r['prompt_tokens'] for r in runs)}-"
        f"{max(r['prompt_tokens'] for r in runs)}, {args.gen_tokens} generated each"
    )
    return 0


def kl_partition(ref: dict, cand: dict) -> tuple[float, float]:
    """KL(ref || cand) on (shared top-k tokens, rest); returns (kl, coverage)."""
    shared = [t for t in ref if t in cand]
    p = [math.exp(ref[t]) for t in shared]
    q = [math.exp(cand[t]) for t in shared]
    p_rest = max(1.0 - sum(p), 0.0)
    q_rest = max(1.0 - sum(q), 1e-12)
    kl = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 0)
    if p_rest > 0:
        kl += p_rest * (math.log(p_rest) - math.log(q_rest))
    return max(kl, 0.0), sum(p)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def pair(ref: dict, cand: dict) -> dict:
    kls, covs, divergence, flips = [], [], [], 0
    for r, c in zip(ref["runs"], cand["runs"]):
        n = min(len(r["ids"]), len(c["ids"]))
        d = next((i for i in range(n) if r["ids"][i] != c["ids"][i]), n)
        divergence.append(d)
        for i in range(d):
            kl, cov = kl_partition(r["top"][i], c["top"][i])
            kls.append(kl)
            covs.append(cov)
        if d < n:
            # A divergence at a confident reference position is a real flip,
            # not a near-tie: margin of the reference top-1 over its runner-up.
            lps = sorted(r["top"][d].values(), reverse=True)
            if len(lps) > 1 and lps[0] - lps[1] > 1.0:
                flips += 1
    gen = ref["meta"]["gen_tokens"]
    return {
        "positions": len(kls),
        "kl_mean": statistics.fmean(kls) if kls else float("nan"),
        "kl_p50": pct(kls, 0.5),
        "kl_p99": pct(kls, 0.99),
        "kl_max": max(kls) if kls else float("nan"),
        "coverage_mean": statistics.fmean(covs) if covs else float("nan"),
        "divergence_median": statistics.median(divergence),
        "identical": sum(d >= gen for d in divergence),
        "passages": len(divergence),
        "confident_flips": flips,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    def load(path: str) -> dict:
        with open(path) as handle:
            return json.load(handle)

    ref, cand = load(args.ref), load(args.cand)
    report = {"ref-cand": pair(ref, cand)}
    if args.floor:
        report["ref-floor"] = pair(ref, load(args.floor))
    for name, s in report.items():
        print(
            f"{name}: KL mean {s['kl_mean']:.3e} p50 {s['kl_p50']:.3e} "
            f"p99 {s['kl_p99']:.3e} max {s['kl_max']:.3e} over {s['positions']} "
            f"decode positions (coverage {s['coverage_mean']:.3f}); divergence "
            f"median {s['divergence_median']} tokens, {s['identical']}/"
            f"{s['passages']} identical, {s['confident_flips']} confident flips"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True, help="server base or /v1/completions URL")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--files", nargs="*")
    c.add_argument("--passages", type=int, default=32)
    c.add_argument("--passage-chars", type=int, default=6000)
    c.add_argument("--gen-tokens", type=int, default=512)
    c.add_argument("--top", type=int, default=20)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--timeout", type=float, default=1800)
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("--floor")
    p.add_argument("--json")
    args = ap.parse_args()
    return cmd_collect(args) if args.cmd == "collect" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
