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
can flip a greedy choice when the batch shape changes, and the packed-fp16
atomics of the production MoE epilogue do it between two identical requests,
because their sum depends on arrival order. A flip near a tie is therefore
expected, and a flat distribution can move its runner-up by a nat or more, so
one run being sure of its token proves nothing. Corrupted state flips tokens
both runs are sure of and/or degrades the text. At the first divergence the
probe reads each run's margin between its own token and the other run's token
from the top-20 logprobs (a token outside them is only bounded by the 20th
entry, shown as ">"), the mean logprob of both continuations from that point,
and a garbage check of the continuation:

  WARN near-tie : margin <= --tie-nats
  WARN soft     : margin <= --fail-nats, fluent continuation
  WARN soft     : the flip is after the solo run's stop token: --ignore-eos
                  forces tokens that serving never shows (looked up by one
                  normal-stop request, made only for a flip that would fail)
  WARN noise    : margin > --fail-nats, but a second solo run of the same
                  prompt already flips at the same token (measured noise)
  FAIL          : margin > --fail-nats, or the continuation's mean logprob
                  drops by > --max-lp-drop nats vs solo, or it looks like
                  garbage (repeats, salad)

The margin is min(solo, got) when the two solo runs differ in their logprobs
(a noisy server, the default MoE epilogue), and the solo margin alone when
they agree to the last bit (the deterministic MoE epilogue: there nothing but
the batch explains a flip).

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
from typing import NamedTuple

# Top logprobs requested per step. vLLM serves up to 20 by default.
TOP_LOGPROBS = 20

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


def complete(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool = True,
    top: int = TOP_LOGPROBS,
) -> dict:
    """Greedy completion: tokens, their logprobs and the top-``top`` per step."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "logprobs": top,
        "return_tokens_as_token_ids": True,
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        choice = json.load(resp)["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or list(choice["text"])
    top_lps = lp.get("top_logprobs") or [{} for _ in tokens]
    token_lps = lp.get("token_logprobs") or [0.0 for _ in tokens]
    return {
        "tokens": list(tokens),
        "top": [t or {} for t in top_lps],
        "lps": [x if x is not None else 0.0 for x in token_lps],
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
    }


def run_parallel(jobs, url, model, ignore_eos=True, top=TOP_LOGPROBS):
    """jobs: list of (key, prompt, max_tokens, start_delay_s)."""
    results: dict = {}
    errors: dict = {}
    t0 = time.monotonic()

    def worker(key, prompt, max_tokens, delay):
        time.sleep(max(0.0, delay - (time.monotonic() - t0)))
        try:
            results[key] = complete(url, model, prompt, max_tokens, ignore_eos, top)
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


class Margins(NamedTuple):
    """How sure each run was of its own token at a divergence, in nats."""

    solo: float
    got: float
    solo_exact: bool = True
    got_exact: bool = True

    @property
    def exact(self) -> bool:
        return self.solo_exact and self.got_exact


def flip_margins(ref: dict, got: dict, i: int) -> Margins:
    """How sure each run was of its own token at the first divergence ``i``.

    ``solo`` is the solo run's logprob of its token minus its logprob of the
    other run's token, ``got`` the same for the other run, both from the
    reported top-k. A token missing from a run's top-k only bounds that margin
    from below (by the distance to the k-th entry), which the ``*_exact``
    fields record. A run without logprobs is infinitely sure.
    """
    a, b = ref["tokens"][i], got["tokens"][i]

    def margin(top: dict, mine, other) -> tuple[float, bool]:
        if mine not in top:
            return float("inf"), False
        if other in top:
            return top[mine] - top[other], True
        return top[mine] - min(top.values()), False

    solo, solo_exact = margin(ref["top"][i] if i < len(ref["top"]) else {}, a, b)
    other, other_exact = margin(got["top"][i] if i < len(got["top"]) else {}, b, a)
    return Margins(solo, other, solo_exact, other_exact)


def same_input_noise(first: dict, second: dict) -> tuple[float, bool]:
    """Run-to-run noise of the server for one prompt, from two identical solo
    requests: ``(biggest, diverged)``, the largest logprob difference of one
    token over the steps where both runs share the prefix (the first diverging
    step included) and whether the two runs chose different tokens. A server
    that is run-to-run deterministic gives (0.0, False)."""
    n = min(len(first["tokens"]), len(second["tokens"]))
    stop = first_divergence(first, second)
    biggest = 0.0
    for j in range(n if stop is None else stop + 1):
        top1 = first["top"][j] if j < len(first["top"]) else {}
        top2 = second["top"][j] if j < len(second["top"]) else {}
        for tok, lp in top1.items():
            if tok in top2:
                biggest = max(biggest, abs(lp - top2[tok]))
    return biggest, stop is not None


def fmt_margin(value: float, exact: bool) -> str:
    if value == float("inf"):
        return "no logprobs"
    return f"{value:.3f}" if exact else f">{value:.3f}"


def stop_index(ref: dict, natural: dict) -> int | None:
    """Where the solo run ``ref``, which was forced past the stop token, would
    have stopped: the first occurrence of the stop token that ``natural`` (the
    same prompt run with normal stopping) ended on. None if it ran to the
    length limit or the forced run never produced that token."""
    if natural.get("finish_reason") != "stop" or not natural["tokens"]:
        return None
    return next(
        (j for j, t in enumerate(ref["tokens"]) if t == natural["tokens"][-1]), None
    )


def compare(
    ref: dict,
    got: dict,
    args,
    noise_at: int | None = None,
    noisy: bool = False,
    stop_at: int | None = None,
) -> tuple[str, str]:
    """Return (status, detail) for ``got`` against the solo ``ref``.

    ``noise_at`` is where a second solo run of the same prompt already
    diverged; a margin-only divergence at that token is run-to-run noise.

    ``noisy`` says the server is not run-to-run deterministic (identical solo
    requests give different logprobs, as with the packed-fp16 CAS MoE
    epilogue). A flip there is a failure only if both runs are sure of their
    own token, i.e. min(solo margin, got margin) > --fail-nats: one run
    sitting on a near-tie is what the noise does. On a bit-identical server
    any flip the solo run is sure about fails.

    ``stop_at`` is the index of the solo run's stop token (see stop_index). A
    flip after it is in the continuation --ignore-eos forces and serving never
    shows: reported, not judged.
    """
    n = min(len(ref["tokens"]), len(got["tokens"]))
    stop_note = ""
    if (
        not args.ignore_eos
        and "stop" in (ref.get("finish_reason"), got.get("finish_reason"))
        and (ref.get("finish_reason"), ref.get("stop_reason"), len(ref["tokens"]))
        != (got.get("finish_reason"), got.get("stop_reason"), len(got["tokens"]))
    ):
        stop_note = (
            f", normal-EOS stopping differs: solo {len(ref['tokens'])} tokens "
            f"({ref.get('finish_reason')}/{ref.get('stop_reason')}), "
            f"got {len(got['tokens'])} "
            f"({got.get('finish_reason')}/{got.get('stop_reason')})"
        )
    for i in range(n):
        a, b = ref["tokens"][i], got["tokens"][i]
        if a == b:
            continue
        margins = flip_margins(ref, got, i)
        margin = min(margins.solo, margins.got) if noisy else margins.solo
        ref_lp = mean(ref["lps"][i:n])
        got_lp = mean(got["lps"][i:n])
        garbage = looks_like_garbage(got["text"])
        if garbage and looks_like_garbage(ref["text"]):
            garbage = None  # the solo text repeats the same way
        repeat_note = ""
        if (
            garbage
            and "non-printable" not in garbage
            and ref_lp - got_lp <= args.repeat_lp_slack
        ):
            # A fluent loop (e.g. a list re-stating one sentence) after a
            # near-tie keeps the solo's confidence; corrupted state does not.
            repeat_note, garbage = f", repetitive but fluent ({garbage})", None
        m = (
            f"margin solo {fmt_margin(margins.solo, margins.solo_exact)} "
            f"got {fmt_margin(margins.got, margins.got_exact)}"
        )
        detail = (
            f"token {i}/{n}: {m}, continuation mean lp {got_lp:.2f} "
            f"vs solo {ref_lp:.2f}{repeat_note}{stop_note}"
        )
        if stop_at is not None and i > stop_at:
            note = f"after the solo stop token at {stop_at}, not judged"
            return "soft", f"{detail}; {note}"
        if garbage:
            return "FAIL", f"{detail}, garbage: {garbage}"
        if ref_lp - got_lp > args.max_lp_drop:
            return "FAIL", f"{detail}, quality drop"
        if margin > args.fail_nats:
            if noise_at == i:
                return "noise", f"{detail}, same flip in solo2"
            return "FAIL", detail
        if margin <= args.tie_nats:
            return "tie", detail
        return "soft", detail
    if not args.ignore_eos and len(ref["tokens"]) != len(got["tokens"]):
        shorter = ref if len(ref["tokens"]) < len(got["tokens"]) else got
        if shorter.get("finish_reason") == "stop":
            return "FAIL", f"different stopping point after {n} matching tokens"
    return "ok", f"matching / {n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="force continuation past EOS (default); disable to test normal stopping",
    )
    ap.add_argument(
        "--save-results", help="write token/logprob results for cross-run comparison"
    )
    ap.add_argument("--tie-nats", type=float, default=0.2)
    ap.add_argument("--fail-nats", type=float, default=1.0)
    ap.add_argument("--max-lp-drop", type=float, default=1.5)
    ap.add_argument(
        "--repeat-lp-slack",
        type=float,
        default=0.5,
        help="a repetitive (not non-printable) continuation is FAIL only if its "
        "mean logprob is more than this many nats below solo",
    )
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
    ap.add_argument(
        "--top-logprobs",
        type=int,
        default=TOP_LOGPROBS,
        help="top logprobs requested per step; a flip's margins are read from them",
    )
    ap.add_argument("--verbose", action="store_true", help="print texts on failure")
    args = ap.parse_args()
    prompts = PROMPTS[: args.num_prompts]
    n = len(prompts)
    t_start = time.monotonic()
    top = args.top_logprobs

    try:
        solo = {
            i: complete(args.url, args.model, p, args.max_tokens, args.ignore_eos, top)
            for i, p in enumerate(prompts)
        }
    except Exception as e:  # noqa: BLE001
        # A solo request already failing (e.g. NaN logprobs -> HTTP 400) is
        # the strongest failure there is.
        print(f"FAIL solo request: {e!r}")
        print("RESULT FAIL: solo requests fail")
        return 1
    solo2 = {
        i: complete(args.url, args.model, p, args.max_tokens, args.ignore_eos, top)
        for i, p in enumerate(prompts)
    }
    together, err_t = run_parallel(
        [(i, p, args.max_tokens, 0.0) for i, p in enumerate(prompts)],
        args.url,
        args.model,
        args.ignore_eos,
        top,
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
    churned, err_c = run_parallel(
        churn_jobs, args.url, args.model, args.ignore_eos, top
    )

    longs = [long_prompt(k, args.long_sentences) for k in range(args.long_prompts)]
    long_solo = {
        k: complete(args.url, args.model, p, args.max_tokens, args.ignore_eos, top)
        for k, p in enumerate(longs)
    }
    long_together, err_l = run_parallel(
        [(k, p, args.max_tokens, 0.0) for k, p in enumerate(longs)],
        args.url,
        args.model,
        args.ignore_eos,
        top,
    )

    if args.save_results:
        with open(args.save_results, "w") as output:
            json.dump(
                {
                    "ignore_eos": args.ignore_eos,
                    "prompts": prompts,
                    "solo": solo,
                    "solo2": solo2,
                    "together": together,
                    "churn": {str(k): v for k, v in churned.items()},
                    "long_prompts": longs,
                    "long_solo": long_solo,
                    "long_together": long_together,
                },
                output,
            )

    counts = {"ok": 0, "tie": 0, "soft": 0, "noise": 0, "FAIL": 0}
    lines = []
    # Where the second solo run diverged from the first: the measured
    # run-to-run noise of this server for each prompt.
    noise_pos = {i: first_divergence(solo[i], solo2[i]) for i in range(n)}
    # A server whose identical solo requests agree to the last bit (the
    # deterministic MoE epilogue) is judged on the solo margin alone; any other
    # server is noisy, and there a flip needs both runs to be sure of it.
    spreads = [same_input_noise(solo[i], solo2[i]) for i in range(n)]
    noise_max = max((s for s, _ in spreads), default=0.0)
    noisy = noise_max > 0.0 or any(diverged for _, diverged in spreads)

    natural: dict = {}

    def stop_after(prompt, ref):
        """Index of the stop token in the forced solo run ``ref``: one request
        per prompt that stops normally, made only for a flip that would fail."""
        if prompt not in natural:
            try:
                natural[prompt] = complete(
                    args.url, args.model, prompt, args.max_tokens, False, top
                )
            except Exception:  # noqa: BLE001
                natural[prompt] = None
        return None if natural[prompt] is None else stop_index(ref, natural[prompt])

    def check(phase, key, i, got, ref_set=None):
        ref = (ref_set or solo)[i]
        noise_at = noise_pos.get(i) if ref_set is None else None
        status, detail = compare(ref, got, args, noise_at, noisy)
        flip_at = first_divergence(ref, got)
        if status == "FAIL" and noisy and args.ignore_eos and flip_at is not None:
            stop_at = stop_after((prompts if ref_set is None else longs)[i], ref)
            if stop_at is not None and flip_at > stop_at:
                status, detail = compare(ref, got, args, noise_at, noisy, stop_at)
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
    server = (
        f"run-to-run noise {noise_max:.2f} nats, a flip fails if both runs are sure"
        if noisy
        else "run-to-run bit-identical, a flip fails if the solo run is sure"
    )
    print(
        f"RESULT {'PASS' if counts['FAIL'] == 0 else 'FAIL'}: {counts['FAIL']} "
        f"failing, {counts['tie']} near-tie, {counts['soft']} soft, "
        f"{counts['noise']} solo-noise, "
        f"{counts['ok']} identical of {total} checks "
        f"({time.monotonic() - t_start:.1f}s; tie <= {args.tie_nats}, "
        f"fail > {args.fail_nats} nats, lp drop > {args.max_lp_drop}; {server})"
    )
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
