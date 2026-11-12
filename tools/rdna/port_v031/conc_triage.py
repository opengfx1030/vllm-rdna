#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Noise or corruption? Triage the flips of concurrency_probe.py on a live server.

  python conc_triage.py validity   --url http://127.0.0.1:8000 --model m --out DIR
  python conc_triage.py runs       --url ... --model m --out DIR --runs 15
  python conc_triage.py adjudicate --url ... --model m --out DIR

validity    the top-k logprobs a completion returns against ``prompt_logprobs`` of
            the same tokens (teacher forcing, one prefill), and two prefills of
            the same sequence against each other. If the first pair differs by
            no more than the second, the completion path (speculative decoding
            included) reports the model's own logprobs at the right positions and
            the rest is run-to-run noise.
runs        N probe runs (``--probe`` picks the script, extra probe options go
            after ``--``), saved as DIR/runNN.{txt,json}.
adjudicate  for every divergence of DIR/run*.json that the probe's rule or its
            pre-2026-11-12 rule (solo margin above --fail-nats, or a token
            outside the solo top-5) fails: the clean-context logprob of both
            tokens at the divergence (``--repeats`` prefills of the same prefix)
            and the clean score of both continuations, so a flip that is a tie
            under teacher forcing can be told from one that is not.

Stdlib only. The server must allow ``--max-logprobs`` >= 20 for ``prompt_logprobs``.
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import concurrency_probe as cp  # noqa: E402


class Thresholds:
    """The probe's defaults, as the probe's compare() reads them."""

    ignore_eos = True
    tie_nats = 0.2
    fail_nats = 1.0
    max_lp_drop = 1.5
    repeat_lp_slack = 0.5


def post(base: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def tokenize(base: str, model: str, text: str) -> list[int]:
    out = post(base, "/tokenize", {"model": model, "prompt": text})
    return [int(t) for t in out["tokens"]]


def detokenize(base: str, model: str, ids: list[int]) -> str:
    try:
        return post(base, "/detokenize", {"model": model, "tokens": ids}).get(
            "prompt", ""
        )
    except Exception as e:  # noqa: BLE001
        return f"<detokenize failed {e!r}>"


def tid(token) -> int:
    """``token_id:N`` -> N."""
    return int(token.split(":")[1])


def completion(base: str, model: str, prompt: str, max_tokens: int, top: int = 20):
    """A forced-continuation greedy completion with integer token ids."""
    res = cp.complete(base + "/v1/completions", model, prompt, max_tokens, True, top)
    res["tokens"] = [tid(t) for t in res["tokens"]]
    res["top"] = [{tid(k): v for k, v in t.items()} for t in res["top"]]
    return res


def teacher_force(base: str, model: str, ids: list[int], top: int = 20) -> list:
    """``prompt_logprobs`` of a token-id prompt: per position {token: (lp, rank)}."""
    out = post(
        base,
        "/v1/completions",
        {
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": top,
        },
    )
    entries = out["choices"][0]["prompt_logprobs"]
    assert len(entries) == len(ids), (len(entries), len(ids))
    return [
        None if e is None else {int(k): (v["logprob"], v["rank"]) for k, v in e.items()}
        for e in entries
    ]


def quantiles(xs: list[float], qs=(0.5, 0.9, 0.99, 0.999)) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    return {q: xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs} | {
        "max": xs[-1],
        "n": len(xs),
    }


# ------------------------------------------------------------------------ validity
def cmd_validity(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    c_vs_tf, tf_vs_tf, c_vs_c = [], [], []
    top_c_tf, top_tf_tf, top_c_c = [], [], []
    odds_tf_tf, odds_c_tf = [], []
    overlap = []
    argmax_bad = tf_argmax_bad = positions = 0
    rows = []
    for index, prompt in enumerate(cp.PROMPTS[: args.num_prompts]):
        ids = tokenize(args.url, args.model, prompt)
        c1 = completion(args.url, args.model, prompt, args.max_tokens)
        c2 = completion(args.url, args.model, prompt, args.max_tokens)
        seq = ids + c1["tokens"]
        t1 = teacher_force(args.url, args.model, seq)
        t2 = teacher_force(args.url, args.model, seq)
        for j, tok in enumerate(c1["tokens"]):
            e1, e2 = t1[len(ids) + j], t2[len(ids) + j]
            top = c1["top"][j]
            positions += 1
            argmax_bad += bool(top) and max(top, key=top.get) != tok
            tf_argmax_bad += max(e1, key=lambda x: e1[x][0]) != tok
            c_vs_tf.append(abs(c1["lps"][j] - e1[tok][0]))
            tf_vs_tf.append(abs(e1[tok][0] - e2[tok][0]))
            overlap.append(
                len(
                    set(sorted(top, key=top.get, reverse=True)[:5])
                    & set(sorted(e1, key=lambda x: -e1[x][0])[:5])
                )
            )
            same_prefix = c2["tokens"][: j + 1] == c1["tokens"][: j + 1]
            if same_prefix:
                c_vs_c.append(abs(c1["lps"][j] - c2["lps"][j]))
            for t, lp in top.items():
                if t in e1:
                    top_c_tf.append(abs(lp - e1[t][0]))
                if t in e1 and t in e2:
                    top_tf_tf.append(abs(e1[t][0] - e2[t][0]))
                if same_prefix and t in c2["top"][j]:
                    top_c_c.append(abs(lp - c2["top"][j][t]))
                if t != tok and t in e1 and tok in e1 and t in e2:
                    # log-odds of the sampled token over another shared top token
                    odds_tf_tf.append(
                        abs((e1[tok][0] - e1[t][0]) - (e2[tok][0] - e2[t][0]))
                    )
                    odds_c_tf.append(abs((c1["lps"][j] - lp) - (e1[tok][0] - e1[t][0])))
        rows.append({"prompt": prompt, "c1": c1, "c2": c2})
        print(f"validity prompt {index}: {len(c1['tokens'])} tokens", flush=True)
    summary = {
        "positions": positions,
        "completion_argmax_not_sampled": argmax_bad,
        "tf_argmax_not_sampled": tf_argmax_bad,
        "mean_top5_overlap_c_vs_tf": statistics.mean(overlap),
        "abs_dlp_sampled_completion_vs_tf": quantiles(c_vs_tf),
        "abs_dlp_sampled_tf_vs_tf": quantiles(tf_vs_tf),
        "abs_dlp_sampled_completion_vs_completion": quantiles(c_vs_c),
        "abs_dlp_shared_top_completion_vs_tf": quantiles(top_c_tf),
        "abs_dlp_shared_top_tf_vs_tf": quantiles(top_tf_tf),
        "abs_dlp_shared_top_completion_vs_completion": quantiles(top_c_c),
        "abs_dlp_logodds_sampled_vs_other_tf_vs_tf": quantiles(odds_tf_tf),
        "abs_dlp_logodds_sampled_vs_other_completion_vs_tf": quantiles(odds_c_tf),
    }
    with open(f"{args.out}/validity.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f)
    print(json.dumps(summary, indent=1))
    return 0


# ---------------------------------------------------------------------------- runs
def cmd_runs(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    start = len(glob.glob(f"{args.out}/run*.json"))
    probe = args.probe or os.path.join(HERE, "concurrency_probe.py")
    for r in range(start, start + args.runs):
        base = f"{args.out}/run{r:02d}"
        t0 = time.time()
        with open(base + ".txt", "w") as f:
            cmd = [
                sys.executable,
                probe,
                "--url",
                args.url + "/v1/completions",
                "--model",
                args.model,
                "--verbose",
                "--save-results",
                base + ".json",
                *args.probe_args,
            ]
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        with open(base + ".txt") as f:
            result = [ln for ln in f if ln.startswith("RESULT")]
        line = result[-1].strip() if result else "?"
        print(f"run{r:02d} rc={rc} {time.time() - t0:.0f}s {line}", flush=True)
    return 0


# ----------------------------------------------------------------------- adjudicate
def top5(res: dict) -> dict:
    """The result as the pre-2026-11-12 probe saw it: five logprobs per step."""
    return {
        **res,
        "top": [dict(sorted(t.items(), key=lambda kv: -kv[1])[:5]) for t in res["top"]],
    }


def legacy_fails(ref: dict, got: dict, i: int) -> bool:
    """The pre-2026-11-12 margin rule: the solo margin above fail_nats, or a
    token outside the solo top-5."""
    top = top5(ref)["top"][i]
    a, b = ref["tokens"][i], got["tokens"][i]
    return a not in top or b not in top or top[a] - top[b] > Thresholds.fail_nats


def checks(d: dict):
    """(phase, key, prompt index, solo, other, noise position, prompt text)."""
    prompts = d["prompts"]
    solo = {int(k): v for k, v in d["solo"].items()}
    solo2 = {int(k): v for k, v in d["solo2"].items()}
    noise_pos = {i: cp.first_divergence(solo[i], solo2[i]) for i in solo}
    for i in solo:
        yield "solo2", i, i, solo[i], solo2[i], noise_pos[i], prompts[i]
    for k, got in d["together"].items():
        i = int(k)
        yield "together", i, i, solo[i], got, noise_pos[i], prompts[i]
    for k, got in d["churn"].items():
        key = tuple(int(x) for x in k.strip("[]() ").split(","))
        i = key[1]
        yield "churn", key, i, solo[i], got, noise_pos[i], prompts[i]
    long_solo = {int(k): v for k, v in d.get("long_solo", {}).items()}
    for k, got in d.get("long_together", {}).items():
        i = int(k)
        if i in long_solo:
            yield "long", i, i, long_solo[i], got, None, d["long_prompts"][i]


def clean_margin(base: str, model: str, prefix: list[int], a: int, b: int) -> dict:
    """Clean-context logprob of a and b after ``prefix`` (one prefill each)."""
    ea = teacher_force(base, model, prefix + [a])[-1]
    eb = teacher_force(base, model, prefix + [b])[-1]
    best = max(ea, key=lambda x: ea[x][0])
    return {
        "lp_a": ea[a][0],
        "lp_b": eb[b][0],
        "rank_a": ea[a][1],
        "rank_b": eb[b][1],
        "top1": best,
        "top1_lp": ea[best][0],
        "margin": ea[a][0] - eb[b][0],
    }


def clean_score(base: str, model: str, ids: list[int], toks: list[int], start: int):
    """Sum and mean clean logprob of ``toks[start:]`` after ``ids + toks[:start]``."""
    plp = teacher_force(base, model, ids + toks)
    lps = [plp[len(ids) + j][toks[j]][0] for j in range(start, len(toks))]
    return {"sum": sum(lps), "n": len(lps), "mean": sum(lps) / max(1, len(lps))}


def cmd_adjudicate(args) -> int:
    ids_cache: dict[str, list[int]] = {}
    natural: dict[str, dict] = {}
    out = []
    for path in sorted(glob.glob(f"{args.out}/run*.json")):
        with open(path) as f:
            d = json.load(f)
        solo = {int(k): v for k, v in d["solo"].items()}
        solo2 = {int(k): v for k, v in d["solo2"].items()}
        spreads = [cp.same_input_noise(solo[i], solo2[i]) for i in solo]
        noisy = any(s > 0.0 or diverged for s, diverged in spreads)
        for phase, key, i, ref, got, noise_at, prompt in checks(d):
            dv = cp.first_divergence(ref, got)
            if dv is None:
                continue
            rule = cp.compare(ref, got, Thresholds, noise_at, noisy)[0]
            # the old probe called a margin-only flip at the solo2 flip "noise"
            legacy = legacy_fails(ref, got, dv) and noise_at != dv
            flipped = cp.flip_margins(ref, got, dv)
            wanted = rule == "FAIL" or legacy
            if not wanted and not args.only_fail:
                wanted = max(flipped.solo, flipped.got) > args.min_margin
            if not wanted:
                continue
            ref_ids = [tid(t) for t in ref["tokens"]]
            got_ids = [tid(t) for t in got["tokens"]]
            a, b = ref_ids[dv], got_ids[dv]
            if prompt not in ids_cache:
                ids_cache[prompt] = tokenize(args.url, args.model, prompt)
                natural[prompt] = cp.complete(
                    args.url + "/v1/completions",
                    args.model,
                    prompt,
                    args.max_tokens,
                    False,
                )
            ids = ids_cache[prompt]
            stop_at = cp.stop_index(ref, natural[prompt])
            repeats = [
                clean_margin(args.url, args.model, ids + ref_ids[:dv], a, b)
                for _ in range(args.repeats)
            ]
            entry = {
                "file": os.path.basename(path),
                "phase": phase,
                "key": list(key) if isinstance(key, tuple) else key,
                "prompt_idx": i,
                "dv": dv,
                "n": min(len(ref_ids), len(got_ids)),
                "rule": rule,
                "legacy_fail": legacy,
                "stop_at": stop_at,
                "after_stop": stop_at is not None and dv > stop_at,
                "a": a,
                "b": b,
                "a_text": detokenize(args.url, args.model, [a]),
                "b_text": detokenize(args.url, args.model, [b]),
                "context": detokenize(
                    args.url, args.model, ref_ids[max(0, dv - 6) : dv]
                ),
                "solo_margin": flipped.solo,
                "got_margin": flipped.got,
                "margins_exact": flipped.exact,
                "clean": repeats,
                "solo_continuation_clean": clean_score(
                    args.url, args.model, ids, ref_ids, dv
                ),
                "got_continuation_clean": clean_score(
                    args.url, args.model, ids, got_ids, dv
                ),
            }
            out.append(entry)
            margins = [round(r["margin"], 2) for r in repeats]
            print(
                f"{entry['file']} {phase} {key} p{i} dv={dv}/{entry['n']} "
                f"rule={rule} legacy={'FAIL' if legacy else 'ok'} "
                f"after_stop={entry['after_stop']} solo={flipped.solo:.2f} "
                f"got={flipped.got:.2f} clean={margins} "
                f"a={entry['a_text']!r} b={entry['b_text']!r}",
                flush=True,
            )
    with open(f"{args.out}/adjudicated.json", "w") as f:
        json.dump(out, f, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["validity", "runs", "adjudicate"])
    ap.add_argument("--url", required=True, help="server base URL, no path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-margin", type=float, default=0.5)
    ap.add_argument("--probe", help="probe script for `runs` (default: the sibling)")
    ap.add_argument(
        "--only-fail", action="store_true", help="adjudicate failing flips only"
    )
    ap.add_argument("probe_args", nargs="*", help="after --: options for the probe")
    args = ap.parse_args()
    return {"validity": cmd_validity, "runs": cmd_runs, "adjudicate": cmd_adjudicate}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
