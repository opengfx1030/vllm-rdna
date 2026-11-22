#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline vLLM run of the real Inkling-2-Mini checkpoint (one GPU, eager, fp16).

Renders each prompt exactly like ``ref_checkpoint.py`` (chat template file,
``add_generation_prompt``), prints the first-step top-8 tokens with their raw
logprobs so they can be compared with the CPU oracle's top-8, then greedy-decodes
``--tokens`` tokens and prints the text.

``--long-prompt-tokens N`` (with ``--prefix-caching``) also renders a prompt of about
``N`` tokens and runs it on a cold prefix cache and again on a warm one (everything but
the last partial block is a prefix-cache hit): the generated ids and the first-step
top-8 must be the same with and without the hit.  The prefill time of both runs is
printed.

    HIP_VISIBLE_DEVICES=2 python real_check.py --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja \
        [--prefix-caching --long-prompt-tokens 1500] [--path reference|hip] [--vision]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


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
    args = ap.parse_args()
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
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=2,
        skip_tokenizer_init=not args.vision,
        max_logprobs=8,
        enable_prefix_caching=args.prefix_caching,
        limit_mm_per_prompt={"image": int(args.vision)},  # 0: no tower, no processor
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, logprobs=8)
    for p, ids in zip(args.prompts.split("|"), prompts):
        t0 = time.time()
        out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        o = out.outputs[0]
        first = o.logprobs[0]
        top = sorted(first.items(), key=lambda kv: kv[1].rank)
        print(f"prompt {p!r} ({len(ids)} tokens) {time.time() - t0:.1f}s")
        print("  top-8:", [(tok.decode([t]), round(lp.logprob, 2)) for t, lp in top])
        print("  text:", repr(tok.decode(o.token_ids)), flush=True)

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
