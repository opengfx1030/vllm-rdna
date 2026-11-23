#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced NLL of the decode path with and without the Inkling-2 DSpark draft.

Speculative decoding is lossless: whatever the draft proposes, a token is accepted
only when it is the one the target samples at that position.  This tool checks the
numbers behind the claim.  A reference text (the greedy generations of a no-draft
run, say) is forced through the engine's *decode* path with the logits processor of
``decode_check.py``: every decode step, and every verification row of a speculative
step, samples the next token of the text, and the raw logprob of that token (taken
before the processor runs) is the NLL of that position.  The draft proposes from the
forced context as in a real run, so its accepted and rejected proposals are real,
and every run attends the same context at every position: runs compare token by
token.  One process per engine (TP=1, in-process engine).

    python dspark_nll.py run --refs refs.json --out nodraft.json --reps 2
    python dspark_nll.py run --refs refs.json --out k2.json --spec-config \\
        '{"method":"dspark","num_speculative_tokens":7,
          "num_speculative_tokens_per_batch_size":[[1,2,2]]}'
    python dspark_nll.py compare nodraft.json k2.json

``refs.json`` is ``[{"name": "chat[0]", "prompt": [ids], "ids": [ids]}, ...]``.
``compare`` takes the two repetitions of the first file as the run-to-run floor of
the shipped configuration (two runs differ already: the expert GEMMs add the eight
expert outputs with fp16 atomics, so no two runs agree bit for bit) and the other
runs as the ones under test; a run passes when the mean ``|dlogp|`` of its forced
tokens against the first floor run is within ``2 * floor + 0.01``, the rule of
``decode_check.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = Path("~/hfcache/inkling2-mini").expanduser()


def spec_counters(llm) -> dict[str, float]:
    totals: dict[str, float] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name and hasattr(metric, "value"):
            totals[metric.name] = totals.get(metric.name, 0.0) + float(metric.value)
    return totals


def run(args) -> int:
    # the engine runs in this process so that the processor class needs no import path
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.path.insert(0, str(HERE))
    from decode_check import _force_next_processor

    from vllm import LLM, SamplingParams

    refs = json.loads(Path(args.refs).read_text())
    kwargs: dict = dict(
        model=args.ckpt,
        dtype="float16",
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        skip_tokenizer_init=True,
        max_logprobs=8,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
        tensor_parallel_size=1,
        seed=0,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        logits_processors=[_force_next_processor()],
    )
    if args.spec_config:
        kwargs["speculative_config"] = json.loads(args.spec_config)
    llm = LLM(**kwargs)

    result: dict = {"spec_config": args.spec_config, "reps": []}
    for rep in range(args.reps):
        before = spec_counters(llm)
        rows = []
        for ref in refs:
            prompt, ids = ref["prompt"], ref["ids"]
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=len(ids),
                ignore_eos=True,
                logprobs=1,
                extra_args={"force": {"start": len(prompt) - 1, "ids": ids}},
            )
            request = {"prompt_token_ids": prompt}
            out = llm.generate([request], sampling, use_tqdm=False)[0].outputs[0]
            if [int(t) for t in out.token_ids] != ids:
                raise SystemExit(f"{ref['name']}: the logits processor did not force")
            logp = [float(out.logprobs[i][t].logprob) for i, t in enumerate(ids)]
            rows.append({"name": ref["name"], "logp": logp})
        after = spec_counters(llm)
        spec = {k: v - before.get(k, 0.0) for k, v in after.items()}
        result["reps"].append({"rows": rows, "spec": spec})
        print(f"rep {rep}: {len(rows)} texts, spec counters {spec}", flush=True)
    Path(args.out).write_text(json.dumps(result))
    print("DSPARK NLL RUN DONE", flush=True)
    return 0


def flat(rep: dict) -> list[float]:
    return [x for row in rep["rows"] for x in row["logp"]]


def stats(test: list[float], ref: list[float]) -> dict:
    import numpy as np

    d = np.asarray(test) - np.asarray(ref)
    return {
        "nll": float(-np.mean(test)),
        "mean": float(d.mean()),
        "mean_abs": float(np.abs(d).mean()),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        "max_abs": float(np.abs(d).max()),
    }


def compare(args) -> int:
    runs = [json.loads(Path(p).read_text()) for p in args.files]
    floor_a, floor_b = runs[0]["reps"][0], runs[0]["reps"][1]
    ref = flat(floor_a)
    floor = stats(flat(floor_b), ref)
    limit = 2 * floor["mean_abs"] + 0.01
    print(f"{len(ref)} forced tokens; floor (two no-draft runs): {floor}")
    print(f"pass rule: mean|dlogp| <= 2 * floor + 0.01 = {limit:.4f}\n")
    print("| run | NLL | mean dlogp | mean abs dlogp | p99 abs | max abs | verdict |")
    print("|---|---|---|---|---|---|---|")
    ok = True
    for path, run_ in zip(args.files, runs):
        for i, rep in enumerate(run_["reps"]):
            if run_ is runs[0] and i == 0:
                s = stats(ref, ref)
                print(f"| {path} #{i} (reference) | {s['nll']:.4f} | - | - | - | - | |")
                continue
            s = stats(flat(rep), ref)
            good = s["mean_abs"] <= limit
            ok &= good
            print(
                f"| {path} #{i} | {s['nll']:.4f} | {s['mean']:+.4f} | "
                f"{s['mean_abs']:.4f} | {s['p99_abs']:.3f} | {s['max_abs']:.3f} | "
                f"{'pass' if good else 'FAIL'} |"
            )
    for path, run_ in zip(args.files, runs):
        spec = run_["reps"][-1]["spec"]
        drafts = sum(v for k, v in spec.items() if k.endswith("num_drafts"))
        accepted = sum(v for k, v in spec.items() if k.endswith("num_accepted_tokens"))
        if drafts:
            print(f"{path}: {drafts:.0f} drafts, {accepted / drafts:.2f} accepted each")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="force the references through one engine")
    r.add_argument("--ckpt", default=str(CKPT))
    r.add_argument("--refs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--spec-config", default="", help="JSON; none = no draft")
    c = sub.add_parser("compare", help="floor and runs under test (needs no engine)")
    c.add_argument("files", nargs="+")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
