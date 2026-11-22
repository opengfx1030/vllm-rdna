#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline vLLM run of the real Inkling-2-Mini checkpoint (one GPU, eager, fp16).

Renders each prompt exactly like ``ref_checkpoint.py`` (chat template file,
``add_generation_prompt``), prints the first-step top-8 tokens with their raw
logprobs so they can be compared with the CPU oracle's top-8, then greedy-decodes
``--tokens`` tokens and prints the text.

    HIP_VISIBLE_DEVICES=2 python real_check.py --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


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
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
    prompts = []
    for p in args.prompts.split("|"):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            chat_template=template,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompts.append(list(ids))

    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=2,
        skip_tokenizer_init=True,
        max_logprobs=8,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
