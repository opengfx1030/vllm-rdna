#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM engine vs the transformers reference on a tiny Inkling-2 checkpoint (one GPU).

Boots ``vllm.LLM`` offline on a checkpoint of ``make_tiny_checkpoint.py`` (no tokenizer,
eager, fp16, TP=1) and compares it with what ``tiny_reference.py`` left next to the
weights: ``ref_logits.npy`` ``[P, V]`` (last-position logits), ``ref_logits_decode.npy``
``[P, N, V]`` (the logits that produced each greedy token) and the greedy ids in
``ref_logits.json``.  The engine returns the whole vocabulary as logprobs
(``logprobs=V``), so the comparison is on the log-softmax over the ``V`` real ids.

Checks (a failed check makes the exit code 1):

* ``last``: every prompt alone, last-position logprobs: top-1 agreement, max / mean
  ``|dlogp|``, top-5 overlap;
* ``decode``: every prompt alone, ``N`` cached greedy steps: ids vs the reference and,
  while the ids agree, per-step ``|dlogp|``;
* ``batch``: all prompts in one batch (concurrent requests) vs the runs above: same ids,
  logprob differences;
* ``--chunked``: ``max_num_batched_tokens`` small enough that the longest prompt is
  prefilled in several chunks, so ``last`` / ``decode`` / ``batch`` also cover chunked
  prefill and mixed chunk + chunk batches.

The prompts are the ones of the tiny harness (``torch.randint(0, 500)`` with seed 1234,
5 / 40 / 70 tokens); the 70-token one exceeds the sliding window (16) and the relative
position extent (32).

Usage (``HIP_VISIBLE_DEVICES`` must select exactly the GPU you were assigned)::

    HIP_VISIBLE_DEVICES=2 python tiny_vllm_check.py --ckpt DIR [--chunked] \
        [--prefix-caching] [--json out.json]

Thresholds: the reference runs fp32 with exactly dequantised experts, the engine fp16
activations on the W4A16 / W8A16 kernels, so the comparison is statistical.  A random
tiny model also flips near-tie top-k router decisions under fp16 rounding, and one flip
moves the logits of the tokens after it by O(1): ``tests/models/inkling2_loop/
test_paged_model.py`` shows the CPU fp16 emulation of this very checkpoint with *exact*
experts at last-position mean |dlogp| ``[0.006, 1.05, 0.14]`` for the three prompts and
at ``<= 0.015`` once the expert choice of the fp32 run is imposed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROMPT_LENS = (5, 40, 70)


def prompts(vocab: int = 500) -> list[list[int]]:
    import torch

    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in PROMPT_LENS]


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def to_vector(logprobs: dict, vocab: int) -> tuple[np.ndarray, int]:
    """Engine ``{token_id: Logprob}`` -> dense ``[V]`` (``-inf`` where absent)."""
    vec = np.full(vocab, -np.inf)
    for tok, lp in logprobs.items():
        vec[tok] = lp.logprob
    return vec, int(np.isinf(vec).sum())


def compare(got: np.ndarray, want_logits: np.ndarray) -> dict:
    """Metrics of engine logprobs ``got [V]`` vs the reference ``want_logits [V]``."""
    want = log_softmax(want_logits)
    d = np.abs(got - want)
    top5 = set(np.argsort(-got)[:5].tolist()) & set(np.argsort(-want)[:5].tolist())
    return {
        "top1": bool(got.argmax() == want.argmax()),
        "top5_overlap": len(top5),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--ckpt", required=True, help="tiny checkpoint dir with ref_logits*"
    )
    ap.add_argument(
        "--chunked",
        action="store_true",
        help="boot with max_num_batched_tokens=16 (several prefill chunks)",
    )
    ap.add_argument(
        "--prefix-caching",
        action="store_true",
        help="enable prefix caching (off by default: isolate the state code)",
    )
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument(
        "--tol-mean",
        type=float,
        default=0.05,
        help="max allowed mean |dlogp| (last position / decode steps)",
    )
    ap.add_argument("--json", help="write all metrics here")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    ref_last = np.load(ckpt / "ref_logits.npy")  # [P, V]
    ref_dec = np.load(ckpt / "ref_logits_decode.npy")  # [P, N, V]
    meta = json.loads((ckpt / "ref_logits.json").read_text())
    ref_ids = [p["greedy_ids"] for p in meta["prompts"]]
    num_prompts, steps, vocab = ref_dec.shape
    ids = prompts(vocab)
    assert [len(p) for p in ids] == list(PROMPT_LENS)

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(ckpt),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=256,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        max_logprobs=vocab,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.prefix_caching,
    )
    if args.chunked:
        kwargs["max_num_batched_tokens"] = 16
        kwargs["enable_chunked_prefill"] = True
    t0 = time.time()
    llm = LLM(**kwargs)
    print(
        f"BOOT OK in {time.time() - t0:.1f}s "
        f"({'chunked' if args.chunked else 'default'}"
        f", prefix caching {'on' if args.prefix_caching else 'off'})",
        flush=True,
    )

    def sp(n: int):
        return SamplingParams(
            temperature=0.0,
            max_tokens=n,
            logprobs=vocab,
            detokenize=False,
            ignore_eos=True,
        )

    def generate(batch: list[list[int]], n: int):
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in batch], sp(n), use_tqdm=False
        )
        res = []
        for out in outs:
            comp = out.outputs[0]
            vecs, missing = [], 0
            for lp in comp.logprobs:
                v, m = to_vector(lp, vocab)
                vecs.append(v)
                missing += m
            res.append(
                {
                    "ids": [int(t) for t in comp.token_ids],
                    "lp": np.stack(vecs),
                    "missing": missing,
                }
            )
        return res

    report: dict = {
        "ckpt": str(ckpt),
        "chunked": args.chunked,
        "prefix_caching": args.prefix_caching,
        "checks": {},
    }
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        report["checks"][name] = {"pass": passed, "detail": detail}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    # ---- last position, prompts one at a time --------------------------------
    last = [generate([p], 1)[0] for p in ids]
    rows = [compare(r["lp"][0], ref_last[i]) for i, r in enumerate(last)]
    report["last"] = rows
    for i, m in enumerate(rows):
        print(
            f"  last  n={len(ids[i]):3d}: top1={m['top1']} top5={m['top5_overlap']}/5 "
            f"max|dlogp|={m['max_abs']:.4f} mean|dlogp|={m['mean_abs']:.5f}",
            flush=True,
        )
    mean_all = float(np.mean([m["mean_abs"] for m in rows]))
    check(
        "last/top1",
        all(m["top1"] for m in rows),
        f"{sum(m['top1'] for m in rows)}/{len(rows)}",
    )
    check(
        "last/mean_abs",
        mean_all < args.tol_mean,
        f"mean over prompts {mean_all:.5f}, per prompt "
        f"{[round(m['mean_abs'], 5) for m in rows]} (tol {args.tol_mean})",
    )
    check(
        "last/full_vocab",
        all(r["missing"] == 0 for r in last),
        "every request returned all logprobs",
    )

    # ---- cached greedy decode, prompts one at a time -------------------------
    dec = [generate([p], steps)[0] for p in ids]
    report["decode"] = []
    for i, r in enumerate(dec):
        agree = 0
        while agree < steps and r["ids"][agree] == ref_ids[i][agree]:
            agree += 1
        per_step = [compare(r["lp"][s], ref_dec[i, s]) for s in range(agree)]
        mean = (
            float(np.mean([m["mean_abs"] for m in per_step]))
            if per_step
            else float("nan")
        )
        mx = (
            float(np.max([m["max_abs"] for m in per_step]))
            if per_step
            else float("nan")
        )
        report["decode"].append(
            {
                "ids": r["ids"],
                "ref_ids": ref_ids[i],
                "agree": agree,
                "mean_abs": mean,
                "max_abs": mx,
            }
        )
        print(
            f"  decode n={len(ids[i]):3d}: ids {r['ids']} vs ref {ref_ids[i]} "
            f"(first {agree}/{steps} equal) mean|dlogp|={mean:.5f} max={mx:.4f}",
            flush=True,
        )
    check(
        "decode/ids",
        all(d["agree"] == steps for d in report["decode"]),
        f"{[d['agree'] for d in report['decode']]} "
        f"of {steps} steps equal the reference",
    )
    check(
        "decode/mean_abs",
        all(d["mean_abs"] < args.tol_mean for d in report["decode"] if d["agree"]),
        f"{[round(d['mean_abs'], 5) for d in report['decode']]} (tol {args.tol_mean})",
    )
    # the decode run's first step is the last-position run again (state is not reused)
    same = all(
        np.allclose(dec[i]["lp"][0], last[i]["lp"][0], atol=1e-3)
        for i in range(num_prompts)
    )
    check("decode/step0_equals_last", same, "first decode step == the prefill-only run")

    # ---- all prompts concurrently in one batch -------------------------------
    batch = generate(ids, steps)
    worst = 0.0
    same_ids = True
    for i in range(num_prompts):
        same_ids &= batch[i]["ids"] == dec[i]["ids"]
        n = min(len(batch[i]["ids"]), len(dec[i]["ids"]))
        agree = 0
        while agree < n and batch[i]["ids"][agree] == dec[i]["ids"][agree]:
            agree += 1
        if agree:
            worst = max(
                worst,
                float(np.abs(batch[i]["lp"][:agree] - dec[i]["lp"][:agree]).max()),
            )
    report["batch"] = {"same_ids": same_ids, "max_abs_vs_single": worst}
    check("batch/ids", same_ids, "concurrent requests generate the single-request ids")
    check(
        "batch/logprobs",
        worst < 0.05,
        f"max |dlogp| vs the single-request runs {worst:.5f}",
    )

    report["pass"] = ok
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
