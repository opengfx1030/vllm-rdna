#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy concurrency-correctness probe for a running OpenAI-compatible server.

The c=1 probes always land on request slot 0, so a bug that hands a request
another slot's state (batch row != slot) is invisible to them and to the
throughput benches. This probe runs N distinct prompts greedily:

  1. solo: one at a time (reference), then a second solo pass (run-to-run
     noise baseline),
  2. together: all N at once,
  3. churned: staggered starts and different max_tokens, so requests finish
     and new ones take freed slots while others are still running
     (batch-row order != slot order),
  4. long: ~1.5k-token prompts solo and together; they cross a 1024-token
     mamba block, so the state pre-copy between blocks runs.

Every output (solo included) must also keep a sane mean token logprob
(--min-mean-lp), which catches a request that is wrong even when alone.

Each output is compared token for token with its solo output. fp16 kernels
can flip a greedy choice when the batch shape changes, but only where the
top logits are nearly tied, and the alternative continuation stays fluent.
Corrupted state instead flips confident tokens and/or degrades the text.
At the first divergence the probe records the solo run's margin between its
choice and the other run's choice (top-5 logprobs), the mean logprob of both
continuations from that point, and a garbage check of the continuation:

  WARN near-tie : margin <= --tie-nats
  WARN soft     : margin <= --fail-nats, fluent continuation
  WARN noise    : margin > --fail-nats, but a second solo run of the same
                  prompt already flips at the same token (measured noise)
  FAIL          : margin > --fail-nats (or not in the solo top-5), or the
                  continuation's mean logprob drops by > --max-lp-drop nats
                  vs solo, or it looks like garbage (repeats, salad)

  python tools/rdna/port_v031/concurrency_probe.py \\
      --url http://127.0.0.1:8000/v1/completions --model m

Stdlib only. Exit code 0 = PASS (warnings allowed), 1 = FAIL.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    "List the first ten prime numbers:",
    "def fibonacci(n):\n    ",
    "Explain photosynthesis in one paragraph:",
    "The three laws of thermodynamics are",
    "Translate to German: 'The weather is nice today.'",
    "Once upon a time, in a small village by the sea,",
    "The difference between TCP and UDP is",
    "A recipe for pancakes needs",
    "In 1969, Apollo 11",
    "SELECT name, COUNT(*) FROM users",
]


LONG_TOPICS = [
    "the migration of arctic terns between the poles",
    "the construction of medieval cathedrals in France",
    "how lithium-ion batteries store and release energy",
    "the rules and history of the game of chess",
    "the water cycle and how clouds form",
    "the design of the TCP congestion control algorithm",
]


def long_prompt(k: int, sentences: int) -> str:
    """A distinct ~1.5k-token prompt: crosses a 1024-token mamba block."""
    topic = LONG_TOPICS[k % len(LONG_TOPICS)]
    body = " ".join(
        f"Note {j} on {topic}: this paragraph adds detail number {j} about {topic}."
        for j in range(sentences)
    )
    return f"Document {k} is about {topic}.\n{body}\nSummarize document {k}:"


def complete(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Greedy completion: tokens, their logprobs and the top-5 per step."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "logprobs": 5,
        "return_tokens_as_token_ids": True,
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        choice = json.load(resp)["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or list(choice["text"])
    top = lp.get("top_logprobs") or [{} for _ in tokens]
    token_lps = lp.get("token_logprobs") or [0.0 for _ in tokens]
    return {
        "tokens": list(tokens),
        "top": [t or {} for t in top],
        "lps": [x if x is not None else 0.0 for x in token_lps],
        "text": choice["text"],
    }


def run_parallel(jobs, url, model):
    """jobs: list of (key, prompt, max_tokens, start_delay_s)."""
    results: dict = {}
    errors: dict = {}
    t0 = time.monotonic()

    def worker(key, prompt, max_tokens, delay):
        time.sleep(max(0.0, delay - (time.monotonic() - t0)))
        try:
            results[key] = complete(url, model, prompt, max_tokens)
        except Exception as e:  # noqa: BLE001
            errors[key] = repr(e)

    threads = [threading.Thread(target=worker, args=job) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def looks_like_garbage(text: str) -> str | None:
    """Token salad / NaN-style failure signatures, judged on the text (so a
    run of EOS tokens kept by ignore_eos, which decode to nothing, is fine)."""
    words = text.split()
    if len(words) >= 8:
        run = best = 1
        for prev, cur in zip(words, words[1:]):
            run = run + 1 if cur == prev else 1
            best = max(best, run)
        if best >= 8:
            return f"same word repeated {best}x"
    if len(words) >= 20 and len(set(words)) < len(words) // 5:
        return f"only {len(set(words))} distinct words of {len(words)}"
    if text:
        bad = sum(1 for c in text if not c.isprintable() and c not in "\n\t\r")
        if bad / len(text) > 0.1:
            return f"{bad} non-printable chars"
    return None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def first_divergence(ref: dict, got: dict) -> int | None:
    n = min(len(ref["tokens"]), len(got["tokens"]))
    return next((i for i in range(n) if ref["tokens"][i] != got["tokens"][i]), None)


def compare(ref: dict, got: dict, args, noise_at: int | None = None) -> tuple[str, str]:
    """Return (status, detail) for ``got`` against the solo ``ref``.

    ``noise_at`` is where a second solo run of the same prompt already
    diverged; a margin-only divergence at that token is run-to-run noise.
    """
    n = min(len(ref["tokens"]), len(got["tokens"]))
    for i in range(n):
        a, b = ref["tokens"][i], got["tokens"][i]
        if a == b:
            continue
        top = ref["top"][i] if i < len(ref["top"]) else {}
        margin = top[a] - top[b] if (a in top and b in top) else None
        ref_lp = mean(ref["lps"][i:n])
        got_lp = mean(got["lps"][i:n])
        garbage = looks_like_garbage(got["text"])
        if garbage and looks_like_garbage(ref["text"]):
            garbage = None  # the solo text repeats the same way
        m = "not in solo top-5" if margin is None else f"margin {margin:.3f}"
        detail = (
            f"token {i}/{n}: {m}, continuation mean lp {got_lp:.2f} "
            f"vs solo {ref_lp:.2f}"
        )
        if garbage:
            return "FAIL", f"{detail}, garbage: {garbage}"
        if ref_lp - got_lp > args.max_lp_drop:
            return "FAIL", f"{detail}, quality drop"
        if margin is None or margin > args.fail_nats:
            if noise_at == i:
                return "noise", f"{detail}, same flip in solo2"
            return "FAIL", detail
        if margin <= args.tie_nats:
            return "tie", detail
        return "soft", detail
    return "ok", f"identical / {n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tie-nats", type=float, default=0.2)
    ap.add_argument("--fail-nats", type=float, default=1.0)
    ap.add_argument("--max-lp-drop", type=float, default=1.5)
    ap.add_argument(
        "--long-prompts",
        type=int,
        default=4,
        help="extra ~1.5k-token prompts run solo and together (0 = off); they "
        "cross a 1024-token mamba block, which exercises the state pre-copy",
    )
    ap.add_argument("--long-sentences", type=int, default=90)
    ap.add_argument(
        "--min-mean-lp",
        type=float,
        default=-2.5,
        help="FAIL any output (solo included) whose mean token logprob is lower",
    )
    ap.add_argument("--verbose", action="store_true", help="print texts on failure")
    args = ap.parse_args()
    prompts = PROMPTS[: args.num_prompts]
    n = len(prompts)
    t_start = time.monotonic()

    try:
        solo = {
            i: complete(args.url, args.model, p, args.max_tokens)
            for i, p in enumerate(prompts)
        }
    except Exception as e:  # noqa: BLE001
        # A solo request already failing (e.g. NaN logprobs -> HTTP 400) is
        # the strongest failure there is.
        print(f"FAIL solo request: {e!r}")
        print("RESULT FAIL: solo requests fail")
        return 1
    solo2 = {
        i: complete(args.url, args.model, p, args.max_tokens)
        for i, p in enumerate(prompts)
    }
    together, err_t = run_parallel(
        [(i, p, args.max_tokens, 0.0) for i, p in enumerate(prompts)],
        args.url,
        args.model,
    )
    # Churn: short and long requests interleaved with staggered starts, so
    # early finishers free slots that later arrivals take while long
    # requests are still decoding.
    lengths = [16, args.max_tokens, 24, args.max_tokens, 32, 40, 48, args.max_tokens]
    churn_jobs = []
    for k in range(2 * n):
        i = (k * 3) % n
        mt = min(lengths[k % len(lengths)], args.max_tokens)
        churn_jobs.append(((k, i), prompts[i], mt, 0.15 * k))
    churned, err_c = run_parallel(churn_jobs, args.url, args.model)

    longs = [long_prompt(k, args.long_sentences) for k in range(args.long_prompts)]
    long_solo = {
        k: complete(args.url, args.model, p, args.max_tokens)
        for k, p in enumerate(longs)
    }
    long_together, err_l = run_parallel(
        [(k, p, args.max_tokens, 0.0) for k, p in enumerate(longs)],
        args.url,
        args.model,
    )

    counts = {"ok": 0, "tie": 0, "soft": 0, "noise": 0, "FAIL": 0}
    lines = []
    # Where the second solo run diverged from the first: the measured
    # run-to-run noise of this server for each prompt.
    noise_pos = {i: first_divergence(solo[i], solo2[i]) for i in range(n)}

    def check(phase, key, i, got, ref_set=None):
        ref = (ref_set or solo)[i]
        noise_at = noise_pos.get(i) if ref_set is None else None
        status, detail = compare(ref, got, args, noise_at)
        got_lp = mean(got["lps"])
        if got_lp < args.min_mean_lp:
            status, detail = "FAIL", f"{detail}; output mean lp {got_lp:.2f}"
        counts[status] += 1
        label = {"ok": "ok", "FAIL": "FAIL"}.get(status, "WARN")
        kind = {"tie": "near-tie ", "soft": "soft ", "noise": "noise "}.get(status, "")
        lines.append(f"{label:4s} {phase:8s} {str(key):8s} prompt {i}: {kind}{detail}")
        if status == "FAIL" and args.verbose:
            lines.append(f"       solo: {ref['text']!r}")
            lines.append(f"       got:  {got['text']!r}")

    for i in range(n):
        check("solo", i, i, solo[i])
    for i in range(n):
        check("solo2", i, i, solo2[i])
    for i in range(n):
        if i in err_t:
            counts["FAIL"] += 1
            lines.append(f"FAIL together {i}: {err_t[i]}")
        else:
            check("together", i, i, together[i])
    for key, _prompt, _mt, _delay in churn_jobs:
        if key in err_c:
            counts["FAIL"] += 1
            lines.append(f"FAIL churn {key}: {err_c[key]}")
        else:
            check("churn", key, key[1], churned[key])

    for k in range(len(longs)):
        check("longsolo", k, k, long_solo[k], long_solo)
    for k in range(len(longs)):
        if k in err_l:
            counts["FAIL"] += 1
            lines.append(f"FAIL long {k}: {err_l[k]}")
        else:
            check("long", k, k, long_together[k], long_solo)

    print("\n".join(lines))
    total = sum(counts.values())
    print(
        f"RESULT {'PASS' if counts['FAIL'] == 0 else 'FAIL'}: {counts['FAIL']} "
        f"failing, {counts['tie']} near-tie, {counts['soft']} soft, "
        f"{counts['noise']} solo-noise, "
        f"{counts['ok']} identical of {total} checks "
        f"({time.monotonic() - t_start:.1f}s; tie <= {args.tie_nats}, "
        f"fail > {args.fail_nats} nats, lp drop > {args.max_lp_drop})"
    )
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
