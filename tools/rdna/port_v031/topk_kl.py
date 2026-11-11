#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Teacher-forced top-k log-probability A/B between served configurations.

``quality_ab.py`` diffs top-1 logprobs of a few character passages. This tool
scores fixed token windows with the top-k ``prompt_logprobs`` of vLLM and
compares runs position by position: NLL/PPL, the KL divergence over the shared
top-k support, top-1 agreement and the drift of the scored token's
log-probability, with bootstrap CIs over windows.

Requests go one at a time, so a window is one prefill of exactly ``--seq-len``
tokens, in one chunk when ``--max-num-batched-tokens`` >= seq-len and alone in
its batch. Prompt-logprob requests skip the prefix cache, so every position is
computed. A kernel that only runs on large prefill chunks (the W4A8 MoE GEMM
from 512 tokens) is on every scored position.

    # once (needs transformers): cut the token windows
    python topk_kl.py windows --tokenizer MODEL_DIR --out windows.json \\
        --domain docs=study_text_all.txt@800000 \\
        --domain mix=eval_mix.txt@0,calib_mix.txt@0 --count 128 --seq-len 2048
    # once per server configuration
    python topk_kl.py collect --url http://127.0.0.1:18096 --model flash-next \\
        --windows windows.json --top 20 --out A.npz
    # pairs, reference first
    python topk_kl.py compare --pair A-B=A.npz,B.npz --pair A-A2=A.npz,A2.npz \\
        --json kl.json --md kl.md

KL(ref || cand) is taken on a partition of the vocabulary: one cell per token of
S (the tokens in both top-k lists, plus the scored token, whose log-probability
is exact on both sides) and one cell for everything else. Merging cells can only
lower a KL divergence (data processing inequality), so the value is a lower
bound on the full-vocabulary KL; ``coverage`` is the reference mass inside S.
"""

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

FIELDS = ("act_lp", "act_rank", "top_ids", "top_lp")


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def ids_digest(domains: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(domains):
        h.update(name.encode())
        for window in domains[name]["windows"]:
            h.update(np.asarray(window, dtype=np.int32).tobytes())
    return h.hexdigest()


def cmd_windows(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    domains = {}
    for spec in args.domain:
        name, _, sources = spec.partition("=")
        ids: list[int] = []
        used = []
        for src in sources.split(","):
            path, _, offset = src.rpartition("@")
            raw = Path(path).read_bytes()
            text_ids = tok(raw.decode(), add_special_tokens=False).input_ids
            part = text_ids[int(offset) :]
            used.append(
                {
                    "file": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "offset": int(offset),
                    "tokens_after_offset": len(part),
                }
            )
            ids.extend(part)
        need = args.count * args.seq_len
        if len(ids) < need:
            raise SystemExit(f"{name}: {len(ids)} tokens, {need} needed")
        windows = [
            ids[i * args.seq_len : (i + 1) * args.seq_len] for i in range(args.count)
        ]
        domains[name] = {"sources": used, "windows": windows}
        print(f"{name}: {args.count} x {args.seq_len} tokens from {used}")
    out = {
        "seq_len": args.seq_len,
        "tokenizer": str(args.tokenizer),
        "sha256": ids_digest(domains),
        "domains": domains,
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"windows sha256 {out['sha256']} -> {args.out}")
    return 0


def parse_window(window: list[int], plp: list, top: int) -> dict:
    """Arrays for positions 1..L-1 of one window from vLLM's prompt_logprobs."""
    n = len(window) - 1
    if len(plp) != len(window) or plp[0] is not None:
        raise ValueError(f"prompt_logprobs has {len(plp)} entries for {len(window)}")
    act_lp = np.empty(n, np.float64)
    act_rank = np.empty(n, np.int32)
    top_ids = np.empty((n, top), np.int32)
    top_lp = np.empty((n, top), np.float64)
    for j in range(n):
        entry = plp[j + 1]
        key = str(window[j + 1])
        act = entry[key]
        act_lp[j] = act["logprob"]
        act_rank[j] = act["rank"]
        # the top entries, plus the scored token when it is outside them
        if len(entry) == top + 1:
            entry = {k: v for k, v in entry.items() if k != key}
        elif len(entry) != top:
            raise ValueError(f"position {j + 1}: {len(entry)} entries for top {top}")
        ranked = sorted((-v["logprob"], v["rank"], int(k)) for k, v in entry.items())
        top_ids[j] = [r[2] for r in ranked]
        top_lp[j] = [-r[0] for r in ranked]
    return {
        "act_lp": act_lp,
        "act_rank": act_rank,
        "top_ids": top_ids,
        "top_lp": top_lp,
    }


def cmd_collect(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.windows).read_text())
    url = args.url.rstrip("/")
    if not url.endswith("/v1/completions"):
        url += "/v1/completions"
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "url": url,
        "model": args.model,
        "windows_sha256": spec["sha256"],
        "seq_len": spec["seq_len"],
        "top": args.top,
        "domains": {},
    }
    for name, dom in spec["domains"].items():
        windows = dom["windows"][: args.limit] if args.limit else dom["windows"]
        rows = []
        latency = []
        for i, window in enumerate(windows):
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
            latency.append(time.time() - t0)
            prompt_tokens = out["usage"]["prompt_tokens"]
            if prompt_tokens != len(window):
                raise ValueError(f"{name}/{i}: server saw {prompt_tokens} tokens")
            row = parse_window(window, out["choices"][0]["prompt_logprobs"], args.top)
            rows.append(row)
            nll = -row["act_lp"].mean()
            print(f"{name} {i}: nll {nll:.4f} in {latency[-1]:.2f}s", flush=True)
        for field in FIELDS:
            arrays[f"{name}.{field}"] = np.stack([r[field] for r in rows])
        arrays[f"{name}.act_id"] = np.asarray([w[1:] for w in windows], np.int32)
        meta["domains"][name] = {
            "windows": len(windows),
            "latency_s": [round(x, 3) for x in latency],
        }
    arrays["meta"] = np.asarray(json.dumps(meta))
    np.savez_compressed(args.out, **arrays)
    print(f"saved {args.out}")
    return 0


def load(path: str) -> tuple[dict, dict]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        data = {k: z[k] for k in z.files if k != "meta"}
    return meta, data


def window_stats(ref: dict, cand: dict, w: int) -> dict:
    """Per-position KL lower bound, coverage, top-1 match and logprob drift."""
    ia, la = ref["top_ids"][w], ref["top_lp"][w]
    ib, lb = cand["top_ids"][w], cand["top_lp"][w]
    act = ref["act_id"][w]
    a_act, b_act = ref["act_lp"][w], cand["act_lp"][w]
    eq = ia[:, :, None] == ib[:, None, :]
    shared = eq.any(-1)
    lb_on_a = np.where(eq, lb[:, None, :], 0.0).sum(-1)
    pa = np.where(shared, np.exp(la), 0.0)
    pb = np.where(shared, np.exp(lb_on_a), 0.0)
    kl = (pa * np.where(shared, la - lb_on_a, 0.0)).sum(-1)
    in_both = (ia == act[:, None]).any(-1) & (ib == act[:, None]).any(-1)
    extra = ~in_both
    pa_act, pb_act = np.exp(a_act), np.exp(b_act)
    kl += np.where(extra, pa_act * (a_act - b_act), 0.0)
    mass_a = pa.sum(-1) + np.where(extra, pa_act, 0.0)
    mass_b = pb.sum(-1) + np.where(extra, pb_act, 0.0)
    ra = np.clip(1.0 - mass_a, 0.0, 1.0)
    rb = np.clip(1.0 - mass_b, 1e-12, 1.0)
    kl += np.where(ra > 0, ra * (np.log(np.maximum(ra, 1e-300)) - np.log(rb)), 0.0)
    return {
        "kl": kl,
        "coverage": np.minimum(mass_a, 1.0),
        "top1": ia[:, 0] == ib[:, 0],
        "dlp": b_act - a_act,
    }


def boot_ci(per_window: np.ndarray, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_window), size=(reps, len(per_window)))
    means = per_window[idx].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def domain_pair(ref: dict, cand: dict, reps: int, worst: int) -> dict:
    n_win = ref["act_lp"].shape[0]
    if not np.array_equal(ref["act_id"], cand["act_id"]):
        raise SystemExit("the two runs scored different tokens")
    per = [window_stats(ref, cand, w) for w in range(n_win)]
    kl = np.stack([p["kl"] for p in per])
    dlp = np.stack([p["dlp"] for p in per])
    top1 = np.stack([p["top1"] for p in per])
    cov = np.stack([p["coverage"] for p in per])
    nll_a = -ref["act_lp"].mean(1)
    nll_b = -cand["act_lp"].mean(1)
    absd = np.abs(dlp)
    w_kl = kl.mean(1)
    order = np.argsort(-w_kl)[:worst]
    return {
        "windows": int(n_win),
        "positions": int(kl.size),
        "nll_ref": float(nll_a.mean()),
        "nll_cand": float(nll_b.mean()),
        "ppl_ref": math.exp(float(nll_a.mean())),
        "ppl_cand": math.exp(float(nll_b.mean())),
        "dnll": float((nll_b - nll_a).mean()),
        "dnll_ci": boot_ci(nll_b - nll_a, reps, 1),
        "kl": float(kl.mean()),
        "kl_ci": boot_ci(w_kl, reps, 2),
        "kl_p99": float(np.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "coverage": float(cov.mean()),
        "top1": float(top1.mean()),
        "top1_ci": boot_ci(top1.mean(1), reps, 3),
        "absdlp_mean": float(absd.mean()),
        "absdlp_p50": float(np.quantile(absd, 0.5)),
        "absdlp_p99": float(np.quantile(absd, 0.99)),
        "absdlp_max": float(absd.max()),
        "identical_positions": float((dlp == 0).mean()),
        "identical_windows": int((absd.max(1) == 0).sum()),
        "worst": [
            {
                "window": int(w),
                "kl": float(w_kl[w]),
                "dnll": float(nll_b[w] - nll_a[w]),
                "top1": float(top1[w].mean()),
                "absdlp_max": float(absd[w].max()),
            }
            for w in order
        ],
    }


def cmd_compare(args: argparse.Namespace) -> int:
    report: dict = {"pairs": {}}
    cache: dict[str, tuple[dict, dict]] = {}
    for spec in args.pair:
        label, _, paths = spec.partition("=")
        ref_path, cand_path = paths.split(",")
        for p in (ref_path, cand_path):
            if p not in cache:
                cache[p] = load(p)
        (ma, da), (mb, db) = cache[ref_path], cache[cand_path]
        if ma["windows_sha256"] != mb["windows_sha256"] or ma["top"] != mb["top"]:
            raise SystemExit(f"{label}: runs used different windows or top-k")
        res = {"ref": ref_path, "cand": cand_path, "domains": {}}
        for name in ma["domains"]:
            sub = {f: da[f"{name}.{f}"] for f in (*FIELDS, "act_id")}
            subc = {f: db[f"{name}.{f}"] for f in (*FIELDS, "act_id")}
            res["domains"][name] = domain_pair(sub, subc, args.boot, args.worst)
        report["pairs"][label] = res
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    md = markdown(report)
    if args.md:
        Path(args.md).write_text(md)
    print(md)
    return 0


def markdown(report: dict) -> str:
    head = (
        "| pair | domain | windows | PPL ref | PPL cand | dNLL [95% CI] | "
        "KL nats/token [95% CI] | KL p99 | top-1 [95% CI] | "
        "abs dlogp p50 / p99 / max | identical tokens |\n"
        "|---|---|---:|---:|---:|---|---|---:|---|---|---:|\n"
    )
    lines = []
    for label, res in report["pairs"].items():
        for name, d in res["domains"].items():
            lines.append(
                f"| {label} | {name} | {d['windows']} | {d['ppl_ref']:.4f} | "
                f"{d['ppl_cand']:.4f} | {d['dnll']:+.5f} "
                f"[{d['dnll_ci'][0]:+.5f}, {d['dnll_ci'][1]:+.5f}] | "
                f"{d['kl']:.5f} [{d['kl_ci'][0]:.5f}, {d['kl_ci'][1]:.5f}] | "
                f"{d['kl_p99']:.4f} | {d['top1']:.4f} "
                f"[{d['top1_ci'][0]:.4f}, {d['top1_ci'][1]:.4f}] | "
                f"{d['absdlp_p50']:.4f} / {d['absdlp_p99']:.4f} / "
                f"{d['absdlp_max']:.3f} | {d['identical_positions']:.1%} |"
            )
    return head + "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows", help="cut token windows from text files")
    w.add_argument("--tokenizer", required=True)
    w.add_argument("--domain", action="append", required=True, help="NAME=FILE@OFF,...")
    w.add_argument("--count", type=int, default=128, help="windows per domain")
    w.add_argument("--seq-len", type=int, default=2048)
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_windows)
    c = sub.add_parser("collect", help="score the windows on one server")
    c.add_argument("--url", required=True, help="server base or /v1/completions URL")
    c.add_argument("--model", required=True)
    c.add_argument("--windows", required=True)
    c.add_argument("--top", type=int, default=20)
    c.add_argument("--limit", type=int, default=0, help="windows per domain (0: all)")
    c.add_argument("--timeout", type=float, default=900)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)
    d = sub.add_parser("compare", help="paired statistics of collected runs")
    d.add_argument("--pair", action="append", required=True, help="LABEL=REF,CAND")
    d.add_argument("--boot", type=int, default=10000)
    d.add_argument("--worst", type=int, default=5)
    d.add_argument("--json")
    d.add_argument("--md")
    d.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
