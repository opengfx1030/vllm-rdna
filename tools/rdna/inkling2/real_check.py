#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline vLLM run of the real Inkling-2-Mini checkpoint (one GPU, or
``--tensor-parallel-size`` GPUs; fp16; eager unless ``--cudagraph-mode`` names a graph
mode: the serving recipe is ``FULL_AND_PIECEWISE``).

Renders each prompt exactly like ``ref_checkpoint.py`` (chat template file,
``add_generation_prompt``), prints the first-step top-8 tokens with their raw
logprobs so they can be compared with the CPU oracle's top-8, then greedy-decodes
``--tokens`` tokens and prints the text.

``--long-prompt-tokens N`` (with ``--prefix-caching``) also renders a prompt of about
``N`` tokens and runs it on a cold prefix cache and again on a warm one (everything but
the last partial block is a prefix-cache hit): the generated ids and the first-step
top-8 must be the same with and without the hit.  The prefill time of both runs is
printed.

``--tensor-parallel-size 2|4`` shards the model over that many GPUs
(``--ar-backend rdna``: the one-shot HIP all-reduce, ``--ar-max-kb`` its size gate).
``--dump FILE`` saves the first-step top-8 (id, raw logprob) and the generated ids of
every prompt; ``--against A [--floor B]`` compares the run with the dump ``A`` of
another run (the TP=1 run) and prints the largest ``|dlogp|`` over the top-8 tokens
both runs share; ``--floor B`` is a second dump of the *same* configuration as ``A``
(two TP=1 runs differ already: the expert GEMMs add the eight expert outputs with
fp16 atomics).  The run passes when its top-1 equals ``A``'s and the difference stays
within ``2 * floor + 0.1``.  One floor pair is a thin basis: measured on the real
checkpoint, TP=1 repeats against ``A`` reach 0.43 / 0.63 on the first step where the
pair A / B gave 0.07 / 0.05, so this limit false-fails TP=1 against itself
(``inkling2-tensor-parallel.md``); decide on the ``--nll-*`` result.

``--nll-file F`` teacher-forces the first ``--nll-tokens`` tokens of a text file
(prompt logprobs: the logprob of every token given its predecessors, a few hundred
natural positions instead of the near-deterministic first step); ``--nll-dump`` saves
them and ``--nll-against A [--nll-floor B]`` prints the mean / p99 / max ``|dlogp|``
against the run ``A``'s, and passes when the mean is within ``2 * floor + 0.01`` (the
mean of the same statistic between ``A`` and the second run ``B`` of its configuration).

    HIP_VISIBLE_DEVICES=2 python real_check.py --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja \
        [--prefix-caching --long-prompt-tokens 1500] [--path reference|hip] [--vision]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from engine_args import add_graph_args, add_tp_args, apply_tp_env, graph_kwargs


def render(tok, template, text: str) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        chat_template=template,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def long_prompt(tok, template, target: int) -> list[int]:
    """A deterministic prompt of at least ``target`` tokens (numbered facts + task)."""
    facts: list[str] = []
    while True:
        facts.append(f"Fact {len(facts) + 1}: the number {len(facts) * 7 + 3} is odd.")
        text = " ".join(facts) + "\nHow many facts were listed? Answer briefly."
        ids = render(tok, template, text)
        if len(ids) >= target:
            return ids


def compare_dumps(run: list[dict], base: list[dict], floor: list[dict] | None) -> bool:
    """``run`` against ``base`` (the TP=1 run); ``floor``: a second run of ``base``."""

    def top(rec: dict) -> dict[int, float]:
        return {t: lp for t, lp in rec["top8"]}

    def spread(a: dict[int, float], b: dict[int, float]) -> float:
        return max((abs(a[t] - b[t]) for t in a.keys() & b.keys()), default=0.0)

    def agree(a: list[int], b: list[int]) -> int:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    ok = True
    for i, (r, b) in enumerate(zip(run, base)):
        a_top, b_top = top(r), top(b)
        same_top1 = max(a_top, key=a_top.get) == max(b_top, key=b_top.get)
        worst = spread(a_top, b_top)
        noise = spread(b_top, top(floor[i])) if floor else None
        within = worst <= (2 * noise + 0.1 if noise is not None else float("inf"))
        ok &= same_top1 and within
        print(
            f"against prompt {i}: top-1 equal {same_top1}, top-8 sets share "
            f"{len(a_top.keys() & b_top.keys())}/8, max |dlogp| of the shared tokens "
            f"{worst:.4f}"
            + ("" if noise is None else f" (run to run floor {noise:.4f})")
            + f", generated ids equal for {agree(r['ids'], b['ids'])} of "
            f"{len(b['ids'])} tokens",
            flush=True,
        )
    print("AGAINST", "PASS" if ok else "FAIL", flush=True)
    return ok


def teacher_forced(llm, tok, path: str, n: int) -> list[float]:
    """The logprob of each of the first ``n`` tokens of the text file given its
    predecessors (the first token has none: dropped)."""
    from vllm import SamplingParams

    ids = tok(Path(path).read_text(), add_special_tokens=False)["input_ids"][:n]
    out = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        use_tqdm=False,
    )[0]
    return [float(out.prompt_logprobs[i][ids[i]].logprob) for i in range(1, len(ids))]


def compare_nll(run: list[float], base: list[float], floor: list[float] | None) -> bool:
    """Per-position ``|dlogp|`` of ``run`` against ``base``; ``floor``: a second run of
    ``base``'s configuration."""
    import numpy as np

    def stats(a: list[float], b: list[float]) -> tuple[float, float, float]:
        d = np.abs(np.asarray(a) - np.asarray(b))
        return float(d.mean()), float(np.percentile(d, 99)), float(d.max())

    mean, p99, worst = stats(run, base)
    line = (
        f"nll against: {len(run)} positions, mean logp {np.mean(run):.4f} vs "
        f"{np.mean(base):.4f}; |dlogp| mean {mean:.4f} p99 {p99:.4f} max {worst:.4f}"
    )
    ok = True
    if floor is not None:
        f_mean, f_p99, f_worst = stats(base, floor)
        ok = mean <= 2 * f_mean + 0.01
        line += (
            f" (run to run floor: mean {f_mean:.4f} p99 {f_p99:.4f} max {f_worst:.4f};"
            f" mean logp {np.mean(floor):.4f})"
        )
    print(line, flush=True)
    print("NLL AGAINST", "PASS" if ok else "FAIL", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument(
        "--prompts", default="What is the capital of France?|Write one word: hello"
    )
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument(
        "--long-prompt-tokens",
        type=int,
        default=0,
        help="also run a prompt of about this many tokens cold and on a warm cache",
    )
    ap.add_argument(
        "--path",
        choices=["reference", "hip"],
        help="VLLM_INKLING2_PATH (default: the native HIP kernels on a GPU)",
    )
    ap.add_argument(
        "--vision",
        action="store_true",
        help="boot with the vision tower (needs the tokenizer / processor), then send "
        "one synthetic image request after the text prompts",
    )
    add_tp_args(ap)
    add_graph_args(ap)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--dump", help="save the first-step top-8 and ids of every prompt")
    ap.add_argument("--against", help="--dump of the run to compare with")
    ap.add_argument(
        "--floor", help="--dump of a second run of the --against configuration"
    )
    ap.add_argument("--nll-file", help="text file to teacher-force (prompt logprobs)")
    ap.add_argument("--nll-tokens", type=int, default=800)
    ap.add_argument("--nll-dump", help="save the teacher-forced logprobs (json)")
    ap.add_argument("--nll-against", help="--nll-dump of the run to compare with")
    ap.add_argument(
        "--nll-floor", help="--nll-dump of a second run of the --nll-against run"
    )
    args = ap.parse_args()
    apply_tp_env(args.ar_backend, args.ar_max_kb)
    if args.path:
        os.environ["VLLM_INKLING2_PATH"] = args.path

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
    prompts = [render(tok, template, p) for p in args.prompts.split("|")]
    if args.long_prompt_tokens and not args.prefix_caching:
        raise SystemExit("--long-prompt-tokens needs --prefix-caching")

    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        **graph_kwargs(args.cudagraph_mode, args.prefill_sizes, args.decode_sizes),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        skip_tokenizer_init=not args.vision,
        max_logprobs=8,
        enable_prefix_caching=args.prefix_caching,
        limit_mm_per_prompt={"image": int(args.vision)},  # 0: no tower, no processor
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, logprobs=8)
    record = []
    for p, ids in zip(args.prompts.split("|"), prompts):
        t0 = time.time()
        out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        o = out.outputs[0]
        first = o.logprobs[0]
        top = sorted(first.items(), key=lambda kv: kv[1].rank)
        print(f"prompt {p!r} ({len(ids)} tokens) {time.time() - t0:.1f}s")
        print("  top-8:", [(tok.decode([t]), round(lp.logprob, 2)) for t, lp in top])
        print("  text:", repr(tok.decode(o.token_ids)), flush=True)
        record.append(
            {
                "prompt": p,
                "ids": [int(t) for t in o.token_ids],
                "top8": [[int(t), float(lp.logprob)] for t, lp in top],
            }
        )
    nll_ok = True
    if args.nll_file:
        t0 = time.time()
        logp = teacher_forced(llm, tok, args.nll_file, args.nll_tokens)
        print(
            f"teacher-forced {len(logp)} positions {time.time() - t0:.1f}s, mean logp "
            f"{sum(logp) / len(logp):.4f}",
            flush=True,
        )
        if args.nll_dump:
            Path(args.nll_dump).write_text(json.dumps({"logp": logp}))
        if args.nll_against:
            base_nll = json.loads(Path(args.nll_against).read_text())["logp"]
            floor_nll = (
                json.loads(Path(args.nll_floor).read_text())["logp"]
                if args.nll_floor
                else None
            )
            nll_ok = compare_nll(logp, base_nll, floor_nll)
    if args.dump:
        Path(args.dump).write_text(json.dumps(record, indent=1))
    top_ok = True
    if args.against:
        base = json.loads(Path(args.against).read_text())
        floor = json.loads(Path(args.floor).read_text()) if args.floor else None
        top_ok = compare_dumps(record, base, floor)
    if not (top_ok and nll_ok):
        return 1

    if args.vision:
        from PIL import Image, ImageDraw

        from vllm.models.inkling2_loop.amd.processing import IMAGE_PLACEHOLDER

        image = Image.new("RGB", (448, 336), (20, 40, 200))
        ImageDraw.Draw(image).ellipse((130, 70, 330, 270), fill=(220, 30, 30))
        text = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": IMAGE_PLACEHOLDER
                    + "Describe the image in one sentence.",
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
            chat_template=template,
        )
        t0 = time.time()
        out = llm.generate(
            {"prompt": text, "multi_modal_data": {"image": image}}, sp, use_tqdm=False
        )[0]
        print(
            f"image request ({len(out.prompt_token_ids)} prompt tokens) "
            f"{time.time() - t0:.1f}s"
        )
        print("  text:", repr(tok.decode(out.outputs[0].token_ids)), flush=True)

    if not args.long_prompt_tokens:
        return 0

    ids = long_prompt(tok, template, args.long_prompt_tokens)
    if len(ids) + args.tokens >= args.max_model_len:
        raise SystemExit(f"{len(ids)} tokens do not fit max_model_len")
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=8)

    def run(params):
        t0 = time.time()
        out = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        return out.outputs[0], time.time() - t0

    def top8(o):
        first = sorted(o.logprobs[0].items(), key=lambda kv: kv[1].rank)
        return {t: lp.logprob for t, lp in first}

    def spread(a, b):  # max |dlogp| over the top-8 tokens both runs share
        return max((abs(a[t] - b[t]) for t in a.keys() & b.keys()), default=0.0)

    llm.reset_prefix_cache()
    _, t_cold = run(sp1)  # prefill of the whole prompt (fills the prefix cache)
    _, t_hit = run(sp1)  # everything but the last partial block is a hit
    llm.reset_prefix_cache()
    cold, _ = run(sp)
    llm.reset_prefix_cache()
    cold2, _ = run(sp)  # the same computation again: the run-to-run noise floor
    hit, _ = run(sp)  # the cache holds the whole prompt again
    a, b, c = top8(cold), top8(cold2), top8(hit)
    same_tokens = list(cold.token_ids) == list(cold2.token_ids) == list(hit.token_ids)
    same_top1 = max(a, key=a.get) == max(b, key=b.get) == max(c, key=c.get)
    floor, worst = spread(a, b), spread(a, c)
    print(
        f"long prompt ({len(ids)} tokens): prefill cold {t_cold:.2f}s, "
        f"warm {t_hit:.2f}s"
    )
    print("  cold text:", repr(tok.decode(cold.token_ids)))
    print("  warm text:", repr(tok.decode(hit.token_ids)))
    print(
        f"  prefix hit vs cold: same generated ids {same_tokens}, same first-step "
        f"top-1 {same_top1}; max |dlogp| of the top-8 tokens both runs share: hit vs "
        f"cold {worst:.4f}, cold vs cold again {floor:.4f} (top-8 sets "
        f"{'equal' if set(a) == set(c) else 'differ'}: the tail ranks sit within about "
        f"a nat of each other)"
    )
    # two cold runs of the same prompt already differ: the expert GEMMs add a token's
    # eight expert outputs with fp16 CAS atomics (order dependent at fp16 precision)
    # and the router amplifies the rounding; the prefix hit must stay within that floor
    ok = same_tokens and same_top1 and worst <= 2 * floor + 0.1
    print("PREFIX HIT", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
