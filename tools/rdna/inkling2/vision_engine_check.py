#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini image requests through the V2 engine: graphs, TP, prefix cache.

Boots the real checkpoint with the vision tower (one GPU, or ``--tensor-parallel-size``
GPUs; eager, or ``--cudagraph-mode FULL_AND_PIECEWISE``) and sends synthetic images
whose content a text answer can name: a red disc on blue, a green square on yellow, a
white triangle on black.

* ``single``: every image alone (``LLM.chat`` with an ``image_pil`` part, or ``--raw``:
  ``LLM.generate`` with the placeholder prompt of ``real_check.py``); the answer must
  name the colours and the shape.
* ``again``: the first image once more.  Its tokens are a prefix-cache hit (the image
  rows are keyed by the multimodal hash): ``num_cached_tokens`` must be above zero and
  the greedy answer must be the first one's (within the engine's run-to-run floor:
  the expert GEMMs add with fp16 atomics, so a near-tie can flip a token).
* ``batch``: two different images in one ``generate`` call, and the two images with a
  text-only prompt; every answer must be the single answer of that image.
* ``--dump F`` / ``--against F``: save the answers, compare with another run (the eager
  one): the text of every request is printed side by side and the first token at which
  the two differ is reported.

The tool prints one ``RESULT`` line (``PASS`` when every keyword check holds and the
prefix hit happened).  Greedy decoding, ``--tokens`` tokens (default 48).

    HIP_VISIBLE_DEVICES=3 python vision_engine_check.py --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja \
        --cudagraph-mode FULL_AND_PIECEWISE --dump graphs.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from engine_args import add_graph_args, add_tp_args, apply_tp_env, graph_kwargs

QUESTION = "Describe the image in one sentence."

# (name, size, background, shape, fill, keywords): every group of keywords must have one
# of its words in the answer
IMAGES = {
    "disc": ((448, 336), (20, 40, 200), "ellipse", (130, 70, 330, 270), (220, 30, 30)),
    "square": (
        (392, 392),
        (240, 220, 40),
        "rectangle",
        (100, 100, 290, 290),
        (20, 150, 40),
    ),
    "triangle": (
        (336, 448),
        (10, 10, 10),
        "polygon",
        [(168, 60), (60, 380), (276, 380)],
        (240, 240, 240),
    ),
}
KEYWORDS = {
    "disc": (("red",), ("circle", "disc", "disk", "round", "oval"), ("blue",)),
    "square": (("green",), ("square", "rectangle", "box"), ("yellow",)),
    "triangle": (("triangle",), ("white",), ("black", "dark")),
}


def make_image(name: str):
    from PIL import Image, ImageDraw

    size, background, shape, geometry, fill = IMAGES[name]
    image = Image.new("RGB", size, background)
    getattr(ImageDraw.Draw(image), shape)(geometry, fill=fill)
    return image


def keywords_ok(name: str, text: str) -> bool:
    low = text.lower()
    return all(any(w in low for w in group) for group in KEYWORDS[name])


def first_difference(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    at = next((i for i in range(n) if a[i] != b[i]), None)
    if at is None and len(a) != len(b):
        return n
    return at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument(
        "--raw",
        action="store_true",
        help="LLM.generate with the placeholder prompt instead of LLM.chat",
    )
    add_tp_args(ap)
    add_graph_args(ap)
    ap.add_argument("--dump")
    ap.add_argument("--against")
    args = ap.parse_args()
    apply_tp_env(args.ar_backend, args.ar_max_kb)

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.models.inkling2_loop.amd.processing import IMAGE_PLACEHOLDER

    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
    tok = AutoTokenizer.from_pretrained(args.ckpt)
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
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1},
        max_logprobs=2,
        seed=0,
    )
    print(
        f"BOOT OK in {time.time() - t0:.1f}s mode={args.cudagraph_mode} "
        f"tp={args.tensor_parallel_size}",
        flush=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=False)
    images = {name: make_image(name) for name in IMAGES}

    def image_prompt(name: str):
        text = tok.apply_chat_template(
            [{"role": "user", "content": IMAGE_PLACEHOLDER + QUESTION}],
            add_generation_prompt=True,
            tokenize=False,
            chat_template=template,
        )
        return {"prompt": text, "multi_modal_data": {"image": images[name]}}

    def chat_messages(name: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": images[name]},
                    {"type": "text", "text": QUESTION},
                ],
            }
        ]

    def text_prompt():
        text = tok.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            add_generation_prompt=True,
            tokenize=False,
            chat_template=template,
        )
        return {"prompt": text}

    def run(names: list[str], text_too: bool = False) -> list[dict]:
        """One engine call over the images ``names`` (and a text prompt); one record
        per request."""
        if args.raw or text_too:
            prompts = [image_prompt(n) for n in names] + (
                [text_prompt()] if text_too else []
            )
            outs = llm.generate(prompts, sp, use_tqdm=False)
        else:
            outs = llm.chat(
                [chat_messages(n) for n in names],
                sp,
                use_tqdm=False,
                chat_template=template,
            )
        records = []
        for name, out in zip([*names, "text"][: len(outs)], outs):
            o = out.outputs[0]
            records.append(
                {
                    "image": name,
                    "ids": [int(t) for t in o.token_ids],
                    "text": tok.decode(o.token_ids),
                    "prompt_tokens": len(out.prompt_token_ids),
                    "cached": out.num_cached_tokens,
                }
            )
        return records

    result: dict[str, list[dict]] = {}
    ok = True

    def check(tag: str, records: list[dict]) -> None:
        nonlocal ok
        result[tag] = records
        for r in records:
            good = r["image"] == "text" or keywords_ok(r["image"], r["text"])
            ok &= good
            print(
                f"[{tag}] {r['image']}: {r['prompt_tokens']} prompt tokens, cached "
                f"{r['cached']}, keywords {'ok' if good else 'MISSING'}: {r['text']!r}",
                flush=True,
            )

    llm.reset_prefix_cache()
    singles = {}
    for name in IMAGES:
        records = run([name])
        singles[name] = records[0]
        check(f"single/{name}", records)
    again = run(["disc"])
    check("again/disc", again)
    hit = again[0]["cached"] or 0
    print(
        f"[again] disc: {hit} of {again[0]['prompt_tokens']} prompt tokens served "
        f"from the prefix cache; ids equal to the first answer: "
        f"{again[0]['ids'] == singles['disc']['ids']}",
        flush=True,
    )
    ok &= hit > 0
    llm.reset_prefix_cache()
    check("batch/disc+square", run(["disc", "square"]))
    check("batch/triangle+disc+text", run(["triangle", "disc"], text_too=True))
    for tag in ("batch/disc+square", "batch/triangle+disc+text"):
        for r in result[tag]:
            if r["image"] == "text":
                continue
            at = first_difference(r["ids"], singles[r["image"]]["ids"])
            print(
                f"[{tag}] {r['image']} against its single answer: "
                f"{'identical' if at is None else f'first difference at token {at}'}",
                flush=True,
            )

    if args.dump:
        Path(args.dump).write_text(json.dumps(result, indent=1))
    if args.against:
        base = json.loads(Path(args.against).read_text())
        for tag, records in result.items():
            for r, b in zip(records, base.get(tag, [])):
                at = first_difference(r["ids"], b["ids"])
                print(
                    f"[against {os.path.basename(args.against)}] {tag}/{r['image']}: "
                    + ("identical" if at is None else f"first difference at token {at}")
                    + (
                        ""
                        if at is None
                        else f"\n    this : {r['text']!r}\n    other: {b['text']!r}"
                    ),
                    flush=True,
                )
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
