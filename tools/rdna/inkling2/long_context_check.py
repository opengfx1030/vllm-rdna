#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Needle-in-a-haystack run of the real Inkling-2-Mini checkpoint at a long context.

Boots the checkpoint (one GPU, or ``--tensor-parallel-size`` GPUs; fp16, prefix
caching; eager, or with ``--cudagraph-mode FULL_AND_PIECEWISE``) with
``--max-model-len`` (``-1``: the engine auto-fits the largest length the KV memory
holds and logs it; use ``--boot-only`` to read it), renders a chat prompt of about
``--frac`` of it (or ``--prompt-tokens`` tokens: numbered filler facts with one
passphrase hidden at ``--depth``) and prints

* the prefill throughput of the cold prompt,
* the prefill of the same prompt on a warm prefix cache (everything but the last
  partial block is a hit),
* the decode time per token at that depth (a warm long decode minus the warm
  one-token run),
* whether the generated text names the passphrase, next to a control with the same
  needle in a ``--control-tokens`` prompt (separates the model from the serving path).

    HIP_VISIBLE_DEVICES=3 python long_context_check.py --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja \
        --max-model-len 262144 --frac 0.9

Tensor parallel (``--ar-backend rdna`` selects the one-shot HIP all-reduce)::

    HIP_VISIBLE_DEVICES=4,5 python long_context_check.py \
        --ckpt ~/hfcache/inkling2-mini \
        --chat-template-file ~/hfcache/inkling2-mini/chat_template.jinja \
        --tensor-parallel-size 2 --ar-backend rdna --max-model-len 1048576 \
        --max-num-batched-tokens 184 --prompt-tokens 262144 --text-only
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from engine_args import add_graph_args, add_tp_args, apply_tp_env, graph_kwargs

NEEDLE = "maple-quartz-47"


class VramPeak:
    """Peak of the device's used VRAM while active, read from sysfs (this process
    never creates a GPU context, so it does not disturb what it measures)."""

    def __init__(self, path: str, period: float = 0.5) -> None:
        self.path = Path(path) if path else None
        self.period = period
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def used(self) -> int:
        return int(self.path.read_text()) if self.path else 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            self._stop.wait(self.period)

    def __enter__(self) -> VramPeak:
        self.peak = self.used()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.used())

    def gib(self) -> str:
        return f"{self.peak / 2**30:.2f}"


def build_prompt(tok, template, target: int, depth: float) -> list[int]:
    """Chat prompt of about ``target`` tokens (just below it) with the needle at
    ``depth`` of the facts."""

    def render(n_facts: int) -> list[int]:
        at = int(n_facts * depth)
        facts = [
            f"Fact {i + 1}: the number {i * 7 + 3} is odd."
            if i != at
            else f"Fact {i + 1}: the secret passphrase is {NEEDLE}."
            for i in range(n_facts)
        ]
        body = (
            "Below is a long list of facts. Exactly one of them states the secret "
            "passphrase.\n"
            + "\n".join(facts)
            + "\nWhat is the secret passphrase? Answer with the passphrase only."
        )
        ids = tok.apply_chat_template(
            [{"role": "user", "content": body}],
            add_generation_prompt=True,
            chat_template=template,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)

    n = max(8, target // 14)
    for _ in range(6):
        ids = render(n)
        if 0.995 * target <= len(ids) <= target:
            break
        n = max(8, int(n * target / len(ids)) - (1 if len(ids) > target else 0))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--prompt-tokens", type=int, default=0, help="instead of --frac")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--control-tokens", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--boot-only", action="store_true")
    add_tp_args(ap)
    add_graph_args(ap)
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="boot without the vision tower and the encoder cache "
        "(limit_mm_per_prompt image=0): their memory goes to the KV cache",
    )
    ap.add_argument(
        "--vram-file",
        default="",
        help="sysfs mem_info_vram_used of the GPU: prints the device's used VRAM "
        "(peak per phase) next to the engine's own accounting",
    )
    ap.add_argument(
        "--kv-margin-gib",
        type=float,
        default=None,
        help="experiment: replace the extensible KV cache's 3 GiB ROCm margin floor "
        "(runs the engine in this process so the constant can be patched)",
    )
    args = ap.parse_args()

    apply_tp_env(args.ar_backend, args.ar_max_kb)

    if args.kv_margin_gib is not None:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    if args.kv_margin_gib is not None:
        from vllm.v1.worker import extensible_kv_cache as ekv

        margin = int(args.kv_margin_gib * 2**30)
        ekv.KV_CACHE_MARGIN_FLOOR_BYTES = margin
        ekv.measure_kv_cache_bytes.__kwdefaults__["margin_floor_bytes"] = margin

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
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
        skip_tokenizer_init=True,
        enable_prefix_caching=True,
        **({"limit_mm_per_prompt": {"image": 0}} if args.text_only else {}),
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)
    if args.vram_file:
        used = VramPeak(args.vram_file).used()
        print(f"device VRAM used after boot: {used / 2**30:.2f} GiB", flush=True)
    if args.boot_only:
        return 0

    target = args.prompt_tokens or int(args.frac * args.max_model_len)
    assert target + args.tokens < args.max_model_len, (
        "prompt + output exceed the length"
    )
    t0 = time.time()
    ids = build_prompt(tok, template, target, args.depth)
    print(f"prompt: {len(ids)} tokens (built in {time.time() - t0:.0f}s)", flush=True)

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    spn = SamplingParams(temperature=0.0, max_tokens=args.tokens)

    def run(prompt_ids, params):
        t0 = time.time()
        with VramPeak(args.vram_file) as vram:
            out = llm.generate(
                [{"prompt_token_ids": prompt_ids}], params, use_tqdm=False
            )
        return out[0].outputs[0], time.time() - t0, vram.gib()

    def vram_note(peak: str) -> str:
        return f"; peak device VRAM {peak} GiB" if args.vram_file else ""

    llm.reset_prefix_cache()
    _, t_cold, v_cold = run(ids, sp1)
    print(
        f"prefill cold: {t_cold:.1f} s = {len(ids) / t_cold:.0f} tok/s "
        f"({len(ids)} tokens){vram_note(v_cold)}",
        flush=True,
    )
    _, t_warm, _ = run(ids, sp1)
    print(f"prefill warm (prefix hit): {t_warm:.2f} s", flush=True)
    out, t_n, v_dec = run(ids, spn)
    n = len(out.token_ids)
    per_token = (t_n - t_warm) / max(n - 1, 1) * 1e3
    text = tok.decode(out.token_ids)
    print(
        f"decode at {len(ids)} tokens of context: {per_token:.1f} ms/token "
        f"({n} tokens in {t_n:.1f} s, warm one-token run {t_warm:.2f} s)"
        f"{vram_note(v_dec)}",
        flush=True,
    )
    print("  text:", repr(text))
    found = NEEDLE in text
    print(
        f"needle at depth {args.depth}: {'FOUND' if found else 'MISSING'}", flush=True
    )

    control = build_prompt(tok, template, args.control_tokens, args.depth)
    out, t_c, _ = run(control, spn)
    ctext = tok.decode(out.token_ids)
    print(
        f"control ({len(control)} tokens, {t_c:.1f} s): "
        f"{'FOUND' if NEEDLE in ctext else 'MISSING'}: {ctext!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
