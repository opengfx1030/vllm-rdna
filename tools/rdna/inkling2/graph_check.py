#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph / speculative-decoding check of Inkling-2 on the V2 runner (one GPU).

Boots ``vllm.LLM`` on a tiny or the real checkpoint with ``--mode`` ``eager``
(``enforce_eager``) or any ``CUDAGraphMode`` name (``FULL_DECODE_ONLY``,
``FULL_AND_PIECEWISE``, ...), optionally with the DSpark draft
(``--spec-k K``), and runs

* ``outputs``: greedy generations (ids and the logprob of every chosen token) of the
  prompts alone and in batches of 1, 2, 3, 5 and 7 (non-power-of-two batches are padded
  to the next captured graph size), written to ``--json``;
* ``bench``: for every batch size of ``--batches`` the wall time of ``--tokens`` greedy
  tokens and of one token, so ``(t_N - t_1) / (N - 1)`` is the decode step time seen by
  the user (scheduler and sampler included) and the speculative-decoding counters of
  the run (mean accepted tokens per step, per position);
* ``--prefix-check`` (real checkpoint, ``--prefix-caching``): a long prompt cold and
  on a warm cache must produce the same ids.

``--compare A.json B.json`` compares two runs (token equality, ``|dlogp|``) and exits.
The expert GEMMs add a token's expert outputs with fp16 CAS atomics, so two runs of the
same prompt differ by rounding noise (docs/rdna2/inkling2-port-spec.md); the comparison
prints the run-to-run floor next to the difference when ``--compare`` gets three files
(``A B A2``: A vs B and A vs A2).

    HIP_VISIBLE_DEVICES=2 python graph_check.py --ckpt DIR --mode FULL_AND_PIECEWISE \
        [--tiny] [--spec-k 7] --batches 1,4,8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_args import add_tp_args, apply_tp_env, graph_kwargs  # noqa: E402

PROMPTS = (
    "What is the capital of France?",
    "Write one word: hello",
    "List three primary colors.",
    "Explain what a prime number is in one sentence.",
    "Translate 'good morning' to Spanish.",
    "Name the largest planet of the solar system.",
    "What is 12 times 12?",
    "Give me a synonym for 'quick'.",
)


def tiny_prompts(vocab: int = 500) -> list[list[int]]:
    import torch

    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in (5, 40, 70)]


def render_prompts(ckpt: str, template_file: str) -> tuple[object, list[list[int]]]:
    from real_check import render
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    template = Path(template_file).read_text() if template_file else None
    return tok, [render(tok, template, p) for p in PROMPTS]


def spec_counters(llm, enabled: bool) -> dict[str, object]:
    """Cumulative speculative-decoding counters of the engine (empty without a draft:
    the metrics need the stat logger, which only the speculative runs switch on)."""
    out: dict[str, object] = {}
    if not enabled:
        return out
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            out["drafts"] = m.value
        elif m.name == "vllm:spec_decode_num_draft_tokens":
            out["draft_tokens"] = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            out["accepted"] = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            out["per_pos"] = list(m.values)
    return out


def spec_delta(a: dict, b: dict) -> dict:
    if not b:
        return {}
    drafts = b.get("drafts", 0) - a.get("drafts", 0)
    acc = b.get("accepted", 0) - a.get("accepted", 0)
    per_pos_a = a.get("per_pos") or [0] * len(b.get("per_pos", []))
    per_pos = [y - x for x, y in zip(per_pos_a, b.get("per_pos", []))]
    return {
        "drafts": drafts,
        "accepted": acc,
        "mean_accepted_per_step": acc / drafts if drafts else None,
        "mean_tokens_per_step": 1 + acc / drafts if drafts else None,
        "per_pos_rate": [p / drafts for p in per_pos] if drafts else [],
    }


def run_outputs(llm, prompts, tokens: int, long_ids: list[int]) -> dict:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tokens, logprobs=2, ignore_eos=True)
    res: dict[str, dict] = {}

    def collect(tag, outs):
        for i, o in enumerate(outs):
            c = o.outputs[0]
            res[f"{tag}/{i}"] = {
                "ids": list(c.token_ids),
                "lp": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)],
                # the two best candidates of every step (token -> logprob)
                "top": [
                    {str(t): x.logprob for t, x in lp.items()} for lp in c.logprobs
                ],
            }

    for i, p in enumerate(prompts):
        collect(
            f"alone{i}", llm.generate([{"prompt_token_ids": p}], sp, use_tqdm=False)
        )
    for n in (1, 2, 3, 5, 7):
        batch = [prompts[i % len(prompts)] for i in range(n)]
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in batch], sp, use_tqdm=False
        )
        collect(f"batch{n}", outs)
    # a short request decoding while the long one is still being prefilled in chunks:
    # the mixed (prefill + decode) batches
    for tag, order in (
        ("mixed", (prompts[0], long_ids)),
        ("mixedr", (long_ids, prompts[1])),
    ):
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in order], sp, use_tqdm=False
        )
        collect(tag, outs)
    return res


def run_bench(llm, prompts, batches, tokens: int, reps: int, spec_on: bool) -> dict:
    """Per batch size ``b``: groups of ``b`` distinct prompts (all prompts of the set
    are covered when ``b`` divides their number) generate ``tokens`` greedy tokens, and
    one token, every group a separate ``generate``.  ``(t_N - t_1) / (N - 1)`` is the
    time per generated token of one request as the user sees it (scheduler and sampler
    included; with a draft it is the effective time per token, not per engine step)."""
    from vllm import SamplingParams

    res: dict[str, dict] = {}
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    for b in batches:
        groups = [
            [
                {"prompt_token_ids": prompts[(g * b + i) % len(prompts)]}
                for i in range(b)
            ]
            for g in range(max(1, len(prompts) // b))
        ]
        llm.generate(groups[0], spn, use_tqdm=False)  # warm-up: shapes, allocator
        t1s, tns, specs = [], [], []
        for _ in range(reps):
            t1 = tn = 0.0
            for group in groups:
                t0 = time.perf_counter()
                llm.generate(group, sp1, use_tqdm=False)
                t1 += time.perf_counter() - t0
            mid = spec_counters(llm, spec_on)
            for group in groups:
                t0 = time.perf_counter()
                outs = llm.generate(group, spn, use_tqdm=False)
                tn += time.perf_counter() - t0
                assert sum(len(o.outputs[0].token_ids) for o in outs) == b * tokens
            t1s.append(t1)
            tns.append(tn)
            specs.append(spec_delta(mid, spec_counters(llm, spec_on)))
        t1, tn = statistics.median(t1s), statistics.median(tns)
        step_ms = 1e3 * (tn - t1) / (len(groups) * (tokens - 1))
        spec = specs[len(specs) // 2] if spec_on else {}
        res[str(b)] = {
            "groups": len(groups),
            "prefill_plus_first_ms": 1e3 * t1 / len(groups),
            "total_s": tn,
            "decode_step_ms": step_ms,
            "decode_tok_s_per_req": 1e3 / step_ms,
            "decode_tok_s_total": b * 1e3 / step_ms,
            "t1_list_ms": [1e3 * x / len(groups) for x in t1s],
            "tn_list_s": tns,
            "spec": spec,
        }
        print(
            f"[bench] batch {b} ({len(groups)} group(s) of distinct prompts): "
            f"{step_ms:.2f} ms per token per request "
            f"({b * 1e3 / step_ms:.1f} tok/s total, {1e3 / step_ms:.1f} per request); "
            f"1-token run {1e3 * t1 / len(groups):.0f} ms; spec {spec or '-'}",
            flush=True,
        )
    return res


def run_accept(llm, prompts, tokens: int) -> dict:
    """Speculative acceptance of every prompt alone (batch 1) over ``tokens`` tokens."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    llm.generate([{"prompt_token_ids": prompts[0]}], sp, use_tqdm=False)  # warm-up
    rows: dict[str, dict] = {}
    drafts = accepted = 0
    per_pos: list[float] = []
    for i, p in enumerate(prompts):
        before = spec_counters(llm, True)
        llm.generate([{"prompt_token_ids": p}], sp, use_tqdm=False)
        d = spec_delta(before, spec_counters(llm, True))
        rows[f"prompt{i}"] = d
        drafts += d["drafts"]
        accepted += d["accepted"]
        per_pos = [
            x + r * d["drafts"]
            for x, r in zip(
                per_pos or [0.0] * len(d["per_pos_rate"]), d["per_pos_rate"]
            )
        ]
        print(
            f"[accept] prompt {i}: {d['drafts']} drafts, mean accepted "
            f"{d['mean_accepted_per_step']:.2f}",
            flush=True,
        )
    total = {
        "drafts": drafts,
        "accepted": accepted,
        "mean_accepted_per_step": accepted / drafts,
        "mean_tokens_per_step": 1 + accepted / drafts,
        "per_pos_rate": [x / drafts for x in per_pos],
    }
    print(
        f"[accept] all {len(prompts)} prompts alone, {tokens} tokens each: {drafts} "
        f"drafts, mean accepted {total['mean_accepted_per_step']:.2f} (mean tokens per "
        f"step {total['mean_tokens_per_step']:.2f}), per position "
        f"{[round(p, 2) for p in total['per_pos_rate']]}",
        flush=True,
    )
    return {"prompts": rows, "total": total}


def prefix_check(
    llm, tok, template_file: str, target: int, tokens: int, spec_on: bool = False
) -> dict:
    """A long prompt cold, cold again and on a warm prefix cache.

    The expert GEMMs add with fp16 CAS atomics, so two runs differ by rounding noise
    and a near-tied argmax can flip: the report therefore gives the greedy ids and the
    logprob of every chosen token, and for the first divergence the two candidates'
    logprobs in both runs (a prefix-cache bug shows as a gap of whole nats at the first
    token, noise as a gap of the size of the cold-vs-cold difference)."""
    from real_check import long_prompt

    from vllm import SamplingParams

    template = Path(template_file).read_text() if template_file else None
    ids = long_prompt(tok, template, target)
    sp = SamplingParams(temperature=0.0, max_tokens=tokens, logprobs=2, ignore_eos=True)

    def run():
        before = spec_counters(llm, spec_on)
        t0 = time.perf_counter()
        o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0].outputs[0]
        dt = time.perf_counter() - t0
        top = [{t: lp.logprob for t, lp in step.items()} for step in o.logprobs]
        return (
            list(o.token_ids),
            top,
            dt,
            spec_delta(before, spec_counters(llm, spec_on)),
        )

    def first_gap(a, b):
        """(first divergent index or None, max |dlogp| of the chosen tokens before it,
        and the candidates' logprobs of both runs at it)."""
        n = min(len(a[0]), len(b[0]))
        div = next((i for i in range(n) if a[0][i] != b[0][i]), None)
        m = n if div is None else div
        worst = max(
            (abs(a[1][i][a[0][i]] - b[1][i][b[0][i]]) for i in range(m)), default=0.0
        )
        at = None
        if div is not None:
            at = {"run_a": a[1][div], "run_b": b[1][div]}
        return div, worst, at

    llm.reset_prefix_cache()
    cold = run()
    llm.reset_prefix_cache()
    cold2 = run()
    warm = run()
    cc = first_gap(cold, cold2)
    cw = first_gap(cold, warm)
    print(
        f"[prefix] {len(ids)}-token prompt: cold {cold[2]:.2f}s, warm {warm[2]:.2f}s; "
        f"cold vs cold again: first divergence {cc[0]}, "
        f"max |dlogp| before it {cc[1]:.3f}"
        f"; cold vs warm: first divergence {cw[0]}, max |dlogp| before it {cw[1]:.3f}, "
        f"candidates at it {cw[2]}",
        flush=True,
    )
    if spec_on:
        for name, run_ in (("cold", cold), ("cold again", cold2), ("warm", warm)):
            s = run_[3]
            print(
                f"[prefix] acceptance, {name}: {s['drafts']} drafts, mean accepted "
                f"{s['mean_accepted_per_step']:.2f}, per position "
                f"{[round(p, 2) for p in s['per_pos_rate']]}",
                flush=True,
            )
    return {
        "tokens": len(ids),
        "cold_s": cold[2],
        "warm_s": warm[2],
        "cold_vs_cold": {"first_divergence": cc[0], "max_dlogp": cc[1], "at": cc[2]},
        "cold_vs_warm": {"first_divergence": cw[0], "max_dlogp": cw[1], "at": cw[2]},
        "same_ids": cold[0] == warm[0],
        "cold_same_as_cold_again": cold[0] == cold2[0],
        "spec": {"cold": cold[3], "cold_again": cold2[3], "warm": warm[3]},
    }


def compare(paths: list[str]) -> int:
    runs = [json.loads(Path(p).read_text())["outputs"] for p in paths]
    base = runs[0]

    def gap(a, b, k, div):
        """How far each run's chosen token is above the other run's choice, in that
        run's own logprobs (small both ways: a near-tie flipped by rounding noise)."""
        ta, tb = str(a[k]["ids"][div]), str(b[k]["ids"][div])
        if "top" not in a[k] or "top" not in b[k]:
            return None
        pa, pb = a[k]["top"][div], b[k]["top"][div]
        if tb not in pa or ta not in pb:  # outside the recorded top-2
            return None
        return round(max(pa[ta] - pa[tb], pb[tb] - pb[ta]), 3)

    def diff(a, b):
        keys = sorted(a.keys() & b.keys())
        same = [k for k in keys if a[k]["ids"] == b[k]["ids"]]
        worst = 0.0
        first_div = []
        for k in keys:
            n = min(len(a[k]["ids"]), len(b[k]["ids"]))
            div = next((i for i in range(n) if a[k]["ids"][i] != b[k]["ids"][i]), n)
            if div < n:
                first_div.append((k, div, gap(a, b, k, div)))
            m = min(div, n)
            if m:
                d = np.abs(np.array(a[k]["lp"][:m]) - np.array(b[k]["lp"][:m]))
                worst = max(worst, float(d.max()))
        return len(keys), len(same), worst, first_div

    for p, other in zip(paths[1:], runs[1:]):
        n, same, worst, div = diff(base, other)
        print(
            f"{paths[0]} vs {p}: {same}/{n} generations identical; max |dlogp| of "
            f"the common prefix before the first divergence {worst:.4f}; first "
            f"divergences (key, position, near-tie gap in nats) {div[:12]}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", nargs="+", metavar="JSON")
    ap.add_argument("--ckpt")
    ap.add_argument("--mode", default="FULL_AND_PIECEWISE")
    ap.add_argument("--tiny", action="store_true", help="tiny checkpoint, no tokenizer")
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--spec-k", type=int, default=0, help="DSpark draft tokens")
    ap.add_argument(
        "--spec-schedule",
        default="",
        help="JSON [[first batch, last batch, verified draft tokens], ...] (dynamic "
        "speculative decoding); needs --spec-k, the tokens drafted per step",
    )
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-outputs", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--prefix-check", type=int, default=0, metavar="TOKENS")
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        default=16,
        help="tokens generated by every run of the prefix check (96+ gives the "
        "speculative acceptance of the cold and warm runs a sample)",
    )
    ap.add_argument(
        "--accept",
        type=int,
        default=0,
        metavar="TOKENS",
        help="speculative runs: every prompt alone for this many tokens, acceptance "
        "per prompt and in total",
    )
    ap.add_argument(
        "--prefill-sizes",
        default="",
        help="token counts of the prefill / mixed (PIECEWISE) graphs, e.g. "
        "16,64,256,1024",
    )
    ap.add_argument(
        "--decode-sizes",
        default="",
        help="request counts of the uniform-decode (FULL) graphs, e.g. 1,2,4,8",
    )
    add_tp_args(ap)
    ap.add_argument("--json")
    ap.add_argument(
        "--path", choices=["reference", "hip"], help="VLLM_INKLING2_PATH override"
    )
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    if not args.ckpt:
        ap.error("--ckpt is required")
    apply_tp_env(args.ar_backend, args.ar_max_kb)
    if args.path:
        os.environ["VLLM_INKLING2_PATH"] = args.path

    from vllm import LLM

    need_tok = not args.tiny
    tok = None
    if args.tiny:
        import torch

        prompts = tiny_prompts()
        gen = torch.Generator().manual_seed(99)
        long_ids = torch.randint(0, 500, (200,), generator=gen).tolist()
    else:
        tok, prompts = render_prompts(args.ckpt, args.chat_template_file)
        from real_check import long_prompt

        template = (
            Path(args.chat_template_file).read_text()
            if args.chat_template_file
            else None
        )
        long_ids = long_prompt(tok, template, 1300)
    kwargs: dict = dict(
        model=args.ckpt,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        skip_tokenizer_init=not (need_tok and args.vision),
        max_logprobs=8,
        enable_prefix_caching=args.prefix_caching,
        limit_mm_per_prompt={"image": int(args.vision)},
        seed=0,
        disable_log_stats=not args.spec_k,
    )
    if args.tiny:
        kwargs["hf_overrides"] = {"vision_config": None}
        kwargs["max_logprobs"] = 500
    kwargs.update(graph_kwargs(args.mode, args.prefill_sizes, args.decode_sizes))
    if args.spec_k:
        kwargs["speculative_config"] = {
            "method": "dspark",
            "num_speculative_tokens": args.spec_k,
        }
        if args.spec_schedule:
            # dynamic speculative decoding: the draft always proposes spec_k tokens,
            # the scheduler verifies as many as the batch size's entry allows
            kwargs["speculative_config"]["num_speculative_tokens_per_batch_size"] = (
                json.loads(args.spec_schedule)
            )
    t0 = time.time()
    llm = LLM(**kwargs)
    boot = time.time() - t0
    print(f"BOOT OK in {boot:.1f}s mode={args.mode} spec_k={args.spec_k}", flush=True)

    result: dict = {"args": vars(args), "boot_s": boot}
    if not args.skip_outputs:
        result["outputs"] = run_outputs(llm, prompts, min(args.tokens, 24), long_ids)
        for k in ("alone0", "alone1", "batch3/2"):
            if k in result["outputs"] or f"{k}/0" in result["outputs"]:
                key = k if k in result["outputs"] else f"{k}/0"
                print(f"[outputs] {key}: {result['outputs'][key]['ids'][:12]}")
    if not args.skip_bench:
        batches = [int(b) for b in args.batches.split(",")]
        result["bench"] = run_bench(
            llm, prompts, batches, args.tokens, args.reps, bool(args.spec_k)
        )
    if args.accept:
        assert args.spec_k, "--accept measures the speculative acceptance"
        result["accept"] = run_accept(llm, prompts, args.accept)
    if args.prefix_check:
        assert tok is not None and args.prefix_caching
        result["prefix"] = prefix_check(
            llm,
            tok,
            args.chat_template_file,
            args.prefix_check,
            args.prefix_tokens,
            bool(args.spec_k),
        )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
