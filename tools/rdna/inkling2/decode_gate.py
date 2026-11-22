#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-path quality and speed gate of the weight-only quantized decode linears
(``VLLM_INKLING2_{LM_HEAD,ATTN_IN,O_PROJ,ATTN}_QUANT``, ``qlinear_pack.py``), one boot.

The quantized kernels serve 1 to 4 token rows; the ``real_check.py --nll-*`` gate feeds
the text as ONE 800-token prefill chunk and would never run them.  This gate boots the
engine with ``--chunk`` (default 4) ``max_num_batched_tokens``: every step of the
teacher-forced prompt is a chunk of at most 4 rows (the quantized attention packs and
the quantized lm_head, which writes the logits of the prompt logprobs), then it runs

* ``graph_check.run_outputs``: greedy generations alone and in batches of 1, 2, 3, 5, 7
  and mixed with a long prompt (ids and logprobs; compare runs with
  ``graph_check.py --compare A.json B.json [A2.json]``);
* ``graph_check.run_bench``: the decode step time of every batch of ``--batches``.

    HIP_VISIBLE_DEVICES=2 VLLM_INKLING2_ATTN_QUANT=w8 python decode_gate.py \
        --ckpt ~/hfcache/inkling2-mini --chat-template-file .../chat_template.jinja \
        --nll-file .../nll_text.md --nll-against tp1_a_nll.json \
        --nll-floor tp1_b_nll.json --json run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_check  # noqa: E402
import real_check  # noqa: E402
from engine_args import add_graph_args, graph_kwargs  # noqa: E402


def paired(run: list[float], base: list[float]) -> str:
    """The signed per-position difference of two runs: its mean is the shift of the NLL
    (a quantization bias shows here, run-to-run noise averages out), with its standard
    error; the mean |dlogp| is dominated by the run-to-run noise."""
    import numpy as np

    m = min(len(run), len(base))
    d = np.asarray(run[:m]) - np.asarray(base[:m])
    return (
        f"paired {m} positions: mean dlogp {d.mean():+.4f} +- {d.std(ddof=1) / m**0.5:.4f} "
        f"(se), mean |dlogp| {np.abs(d).mean():.4f}, p99 |dlogp| {np.percentile(np.abs(d), 99):.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--chunk", type=int, default=4, help="max_num_batched_tokens")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--nll-file")
    ap.add_argument("--nll-tokens", type=int, default=800)
    ap.add_argument(
        "--nll-against",
        action="append",
        default=[],
        help="reference dump {'logp': [...]} (repeatable): the first one gates, with "
        "--nll-floor; the others are reported (e.g. the flag-off run of this chunking)",
    )
    ap.add_argument("--nll-floor")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--long-prompt-tokens", type=int, default=300)
    ap.add_argument("--skip-outputs", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--json", required=True)
    add_graph_args(ap)
    args = ap.parse_args()

    from vllm import LLM

    tok, prompts = graph_check.render_prompts(args.ckpt, args.chat_template_file)
    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
    long_ids = real_check.long_prompt(tok, template, args.long_prompt_tokens)
    sizes = ",".join(str(s) for s in range(1, args.chunk + 1))
    kwargs = dict(
        model=args.ckpt,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.chunk,
        max_num_seqs=args.max_num_seqs,
        skip_tokenizer_init=True,
        max_logprobs=8,
        seed=0,
        disable_log_stats=True,
    )
    kwargs.update(
        graph_kwargs(
            args.cudagraph_mode, args.prefill_sizes or sizes, args.decode_sizes
        )
    )
    flags = {k: v for k, v in os.environ.items() if k.startswith("VLLM_INKLING2_")}
    print(f"flags {flags} chunk {args.chunk}", flush=True)
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)

    result: dict = {"flags": flags, "chunk": args.chunk}
    ok = True
    if args.nll_file:
        t0 = time.time()
        logp = real_check.teacher_forced(llm, tok, args.nll_file, args.nll_tokens)
        print(
            f"teacher-forced {len(logp)} positions in chunks of {args.chunk} "
            f"{time.time() - t0:.1f}s, mean logp {sum(logp) / len(logp):.4f}",
            flush=True,
        )
        result["logp"] = (
            logp  # the key of real_check.py --nll-dump, so dumps interchange
        )
        for i, path in enumerate(args.nll_against):
            if not Path(path).exists():
                print(f"nll against {path}: no such reference yet, skipped", flush=True)
                continue
            base = json.loads(Path(path).read_text())["logp"]
            m = min(len(logp), len(base))  # the reference may cover a shorter text
            floor = (
                json.loads(Path(args.nll_floor).read_text())["logp"][:m]
                if args.nll_floor and i == 0
                else None
            )
            print(f"nll against {path}", flush=True)
            good = real_check.compare_nll(logp[:m], base[:m], floor)
            print(paired(logp, base), flush=True)
            ok = ok and (good or i > 0)
    if not args.skip_outputs:
        result["outputs"] = graph_check.run_outputs(
            llm, prompts, min(args.tokens, 24), long_ids
        )
    if not args.skip_bench:
        batches = [int(b) for b in args.batches.split(",")]
        result["bench"] = graph_check.run_bench(
            llm, prompts, batches, args.tokens, args.reps, False
        )
    Path(args.json).write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {args.json}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
