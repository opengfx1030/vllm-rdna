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
* ``--prefix-caching``: the engine's prefix cache is on, so ``decode`` / ``batch`` hit
  the blocks of the runs before them (the first decode step must equal the prefill-only
  run); ``prefix/*`` adds two identical requests in one batch on a cold cache (the
  second would hit blocks the first only writes in that very step: the conv-state
  manager defers it by one step), a warm repeat, and a new tail after a shared prefix
  with and without the hit;
* ``--path``: ``VLLM_INKLING2_PATH``: the torch reference or the native HIP kernels (the
  default on a GPU).  The HIP kernels need the attention shapes of
  ``make_tiny_checkpoint.py --head-dim 128 --d-rel 16 --num-heads 6 --num-kv-heads 2``;
  ``--router-bias-scale 100`` makes the routing fp16-stable (exact comparisons).

The prompts are the ones of the tiny harness (``torch.randint(0, 500)`` with seed 1234,
5 / 40 / 70 tokens); the 70-token one exceeds the sliding window (16) and the relative
position extent (32).

Usage (``HIP_VISIBLE_DEVICES`` must select exactly the GPU you were assigned)::

    HIP_VISIBLE_DEVICES=2 python tiny_vllm_check.py --ckpt DIR [--chunked] \
        [--prefix-caching] [--json out.json]

Thresholds: the reference runs fp32 with exactly dequantised experts, the engine fp16
activations on the W4A16 / W8A16 kernels, so the comparison is statistical.  A random
tiny model also flips near-tie top-k router decisions under fp16 rounding (fp32 margins
of 1e-4 .. 2e-3 against fp16 noise of ~1e-2), and one flip moves the logits of the
tokens after it by O(1).  Which decisions flip depends on the GEMM shapes, so the same
prompt can fail alone, chunked or in a batch.  ``tests/models/inkling2_loop/
test_paged_model.py`` shows the CPU fp16 emulation of these checkpoints with *exact*
experts at last-position mean |dlogp| ``[0.006, 1.05, 0.14]`` (tiny1) and
``[0.017, 6.0, 1.8]`` (tiny8) for the three prompts, and at ``0.006 .. 0.061`` once the
expert choice of the fp32 run is imposed.  On the engine the same imposition (see
``docs/rdna2/inkling2-port-spec.md``, 2026-11-22) gives the reference ids for 8/8 steps
in every scenario (alone, batched, shifted in a batch, chunked), so a failure on the 40
or 70 token prompt is a routing-stability statement about the prompt until that
forced-route check says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument(
        "--tol-batch",
        type=float,
        default=0.05,
        help="max allowed |dlogp| between runs of one prompt that differ only in batch "
        "composition / prefix hit (fp16 noise grows with the loop visits: use 0.15 "
        "on the 8-layer checkpoint)",
    )
    ap.add_argument("--json", help="write all metrics here")
    ap.add_argument(
        "--path",
        choices=["reference", "hip"],
        help="VLLM_INKLING2_PATH: the torch reference or the native HIP kernels "
        "(default: the HIP kernels on a GPU)",
    )
    args = ap.parse_args()
    if args.path:
        os.environ["VLLM_INKLING2_PATH"] = args.path

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
        limit_mm_per_prompt={"image": 0},  # the tiny checkpoints have no vision weights
        # no vision tower at all: its HIP kernels only take the real 20x20 patch
        # geometry, the tiny ``vision_config`` is there for the transformers reference
        hf_overrides={"vision_config": None},
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
        mean = float(np.mean([m["mean_abs"] for m in per_step])) if per_step else None
        mx = float(np.max([m["max_abs"] for m in per_step])) if per_step else None
        flip = None
        if agree < steps:  # first step whose greedy token differs from the reference
            want, got = ref_ids[i][agree], r["ids"][agree]
            flip = {
                "prompt": len(ids[i]),
                "step": agree,
                "ref_gap": float(ref_dec[i, agree][want] - ref_dec[i, agree][got]),
            }
        report["decode"].append(
            {
                "ids": r["ids"],
                "ref_ids": ref_ids[i],
                "agree": agree,
                "mean_abs": mean,
                "max_abs": mx,
                "flip": flip,
            }
        )
        shown = (
            "mean|dlogp| n/a (no step equal)"
            if mean is None
            else f"mean|dlogp|={mean:.5f} max={mx:.4f}"
        )
        print(
            f"  decode n={len(ids[i]):3d}: ids {r['ids']} vs ref {ref_ids[i]} "
            f"(first {agree}/{steps} equal) {shown}",
            flush=True,
        )
    flips = [d["flip"] for d in report["decode"] if d["flip"]]
    # a flip is fp16 noise when the reference's own gap between the two tokens is below
    # the largest |dlogp| of the agreeing steps; the random tiny weights give near-flat
    # distributions, so such ties are common
    noise = max(
        (d["max_abs"] for d in report["decode"] if d["max_abs"] is not None),
        default=0.0,
    )
    check(
        "decode/ids",
        all(f["ref_gap"] <= noise for f in flips),
        f"{[d['agree'] for d in report['decode']]} of {steps} steps equal the reference"
        + "".join(
            f"; n={f['prompt']} step {f['step']}: reference gap {f['ref_gap']:.4f}"
            for f in flips
        )
        + f" (noise {noise:.4f})",
    )
    means = [
        None if d["mean_abs"] is None else round(d["mean_abs"], 5)
        for d in report["decode"]
    ]
    check(
        "decode/mean_abs",
        all(d["mean_abs"] < args.tol_mean for d in report["decode"] if d["agree"]),
        f"{means} (tol {args.tol_mean}; None: no step equal)",
    )
    # the decode run's first step is the last-position run again (state is not reused);
    # with prefix caching it hits the blocks of the previous run, a different chunk
    # shape: fp16 noise, not bitwise equality
    atol = 0.1 if args.prefix_caching else 1e-3
    same = all(
        np.allclose(dec[i]["lp"][0], last[i]["lp"][0], atol=atol)
        for i in range(num_prompts)
    )
    check(
        "decode/step0_equals_last",
        same,
        f"first decode step == the prefill-only run (atol {atol})",
    )

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
        worst < args.tol_batch,
        f"max |dlogp| vs the single-request runs {worst:.5f} (tol {args.tol_batch})",
    )

    # ---- prefix caching: a hit must give the output of a cold run ------------
    if args.prefix_caching:
        longest = ids[-1]
        # identical requests in one batch on a cold cache: the second one would hit
        # blocks the first one only writes in this very step, so the scheduler
        # defers it (Inkling2ConvStateManager) and it hits them one step later
        llm.reset_prefix_cache()
        pair = generate([longest, longest], steps)
        worst = max(
            (
                float(np.abs(r["lp"] - dec[-1]["lp"]).max())
                for r in pair
                if r["ids"] == dec[-1]["ids"]
            ),
            default=float("inf"),
        )
        same = all(r["ids"] == dec[-1]["ids"] for r in pair)
        check(
            "prefix/same_step_hit",
            same and worst < args.tol_batch,
            f"two identical {len(longest)}-token requests in one batch (cold cache): "
            f"ids equal the single run {same}, max |dlogp| {worst:.5f}",
        )
        # a later request hits the blocks of finished ones
        again = generate([longest], steps)[0]
        same = again["ids"] == dec[-1]["ids"]
        worst = (
            float(np.abs(again["lp"] - dec[-1]["lp"]).max()) if same else float("inf")
        )
        check(
            "prefix/cross_step_hit",
            same and worst < args.tol_batch,
            f"warm cache, same prompt: ids equal {same}, max |dlogp| {worst:.5f}",
        )
        # a different tail after a shared prefix, with and without the hit
        branch = longest[:67] + [(longest[67] + 1) % vocab, longest[69]]
        llm.reset_prefix_cache()
        cold = generate([branch], steps)[0]
        llm.reset_prefix_cache()
        generate([longest], 1)  # primes the cache with the shared prefix
        warm = generate([branch], steps)[0]
        same = warm["ids"] == cold["ids"]
        worst = float(np.abs(warm["lp"] - cold["lp"]).max()) if same else float("inf")
        check(
            "prefix/hit_vs_cold",
            same and worst < args.tol_batch,
            f"67 shared tokens + a new tail: ids equal {same}, max |dlogp| {worst:.5f}",
        )

    report["pass"] = ok
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
