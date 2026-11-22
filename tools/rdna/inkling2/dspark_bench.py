#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speed, acceptance and fidelity of the Inkling-2 DSpark draft on real chat prompts.

One ``vllm.LLM`` boot of the real checkpoint (``--tensor-parallel-size`` 1, 2 or 4)
with or without the draft (``--spec-k`` drafted tokens, ``--schedule`` the tokens the
scheduler verifies per batch size), then any of

* ``--bench``: for every batch size of ``--batches`` groups of distinct chat prompts
  (:data:`CHAT_PROMPTS`, rendered with the model's chat template) generate
  ``--tokens`` greedy tokens and one token; ``(t_N - t_1) / (N - 1)`` is the time per
  generated token of one request as the user sees it.  With a draft the counters give
  the accepted tokens per step and the cumulative acceptance per draft position;
* ``--gen FILE``: the greedy generations (ids and the two best logprobs of every step)
  of the prompts alone and in batches, written to ``FILE``;
* ``--teacher A.json,B.json``: the generated ids of earlier ``--gen`` runs forced
  through this engine (prompt logprobs): per generated position the logprob gap
  between the engine's own top-1 token and the token that was generated.  A greedy
  run of the same model has gaps of zero except at near-ties that rounding noise
  flips, so the gap statistics of a speculative run next to those of a no-draft run
  say whether the verification path changes the output beyond the noise floor (expert
  GEMMs add with fp16 CAS atomics, so no two runs agree bit for bit);
* ``--prefix-check``: a ~1000-token prompt and an overlapping one cold, as a
  prefix-cache hit, after a cache reset and as duplicates inside one step: the tokens
  served from the cache, the accepted tokens per step and the divergence of every hit
  from the cold generation;
* ``--compare A.json,B.json``: the first divergence of two ``--gen`` files and whether
  both engines saw a near tie there (needs no engine);
* ``--table F1.json F2.json ...``: milliseconds per token and speed-up of the ``--json``
  files of several runs against the one without a draft (needs no engine).

    HIP_VISIBLE_DEVICES=2 python dspark_bench.py --ckpt DIR --spec-k 7 \
        --schedule '[[1, 8, 5]]' --bench --batches 1,2,4,8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Single-turn prompts of the kinds a chat model sees: code, maths, explanation, writing,
# planning, structured output.  Most answers run well past a few hundred tokens (the
# model thinks first at its default reasoning effort).
CHAT_PROMPTS = (
    (
        "Write a Python function that merges two sorted lists into one sorted list, "
        "with a docstring and a few unit tests."
    ),
    (
        "Explain how the attention mechanism of a transformer works, as if to a "
        "software engineer who has never studied machine learning."
    ),
    (
        "I have three days in Lisbon in spring. Plan an itinerary with food "
        "recommendations and rough travel times."
    ),
    (
        "A train leaves city A at 9:00 at 80 km/h and another leaves city B towards A "
        "at 10:00 at 100 km/h; the cities are 460 km apart. When and where do they "
        "meet? Show the steps."
    ),
    (
        "Summarize the causes and consequences of the 2008 financial crisis in about "
        "five paragraphs."
    ),
    (
        "Write a short story of about 400 words about a lighthouse keeper who finds a "
        "message in a bottle."
    ),
    (
        "Give me a SQL query for the top 5 customers by total order value in the last "
        "90 days and explain each clause. Tables: customers(id, name), orders(id, "
        "customer_id, total, created_at)."
    ),
    (
        "Compare PostgreSQL and MongoDB for an application that stores user profiles "
        "and activity events; recommend one and justify it."
    ),
    (
        "Translate this sentence into French and explain the grammar of the result: "
        "'The committee decided to postpone the meeting because several members were "
        "unable to attend.'"
    ),
    (
        "Write a bash script that watches a directory and prints the name of every "
        "new file that appears, with a timestamp."
    ),
    (
        "What are the main differences between TCP and UDP, and when should an "
        "application choose each?"
    ),
    (
        "Rewrite this email so that it sounds polite and professional: 'hey, your "
        "report is late again. i need it today or we have a problem.'"
    ),
    "Derive the formula for the sum of the first n cubes and prove it by induction.",
    (
        "Explain the difference between processes and threads in an operating system "
        "and show how to create a thread in C."
    ),
    (
        "Create a JSON object describing a fictional library with three books, each "
        "with title, author, year and a list of genres. Output only JSON."
    ),
    (
        "How does photosynthesis work? Describe the light-dependent and "
        "light-independent reactions."
    ),
    (
        "Debug this Python code and explain the bug: def avg(xs): total = 0; for x in "
        "xs: total += x; return total / len(xs) crashes on an empty list. Suggest "
        "fixes."
    ),
    "Write five haiku about autumn rain and then explain the imagery.",
    (
        "Outline an eight-week study plan to learn linear algebra for a machine "
        "learning engineer, with weekly goals and resources."
    ),
    "Implement an LRU cache in C++ with O(1) get and put and explain the design.",
    (
        "Explain the CAP theorem and give examples of systems that choose CP and of "
        "systems that choose AP."
    ),
    (
        "What would happen to Earth's tides if the Moon were twice as far away? "
        "Reason it through."
    ),
    (
        "Write a regular expression that matches valid IPv4 addresses and explain how "
        "it works."
    ),
    (
        "Give me five tips for improving sleep quality, each with a brief scientific "
        "rationale."
    ),
)

EOS_ID = 200006  # generation_config.json of the checkpoint


def render_prompts(ckpt: str, template_file: str, effort: str | None, n: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    template = Path(template_file).read_text() if template_file else None
    extra = {} if effort is None else {"reasoning_effort": effort}
    prompts = []
    for text in CHAT_PROMPTS[:n]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            chat_template=template,
            **extra,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompts.append(list(ids))
    return tok, prompts


def build_llm(args):
    from vllm import LLM

    kwargs: dict = dict(
        model=args.ckpt,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        skip_tokenizer_init=True,
        max_logprobs=8,
        enable_prefix_caching=not args.no_prefix_caching,
        limit_mm_per_prompt={"image": 0},
        tensor_parallel_size=args.tensor_parallel_size,
        seed=0,
        disable_log_stats=not args.spec_k,
    )
    if args.async_scheduling != "auto":
        kwargs["async_scheduling"] = args.async_scheduling == "on"
    if args.mode == "eager":
        kwargs["enforce_eager"] = True
    else:
        kwargs["compilation_config"] = {"cudagraph_mode": args.mode}
    if args.spec_k:
        spec: dict = {"method": "dspark", "num_speculative_tokens": args.spec_k}
        if args.schedule:
            spec["num_speculative_tokens_per_batch_size"] = json.loads(args.schedule)
        kwargs["speculative_config"] = spec
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)
    return llm


def bench(llm, prompts, batches, tokens: int, reps: int, spec_on: bool) -> dict:
    from graph_check import spec_counters, spec_delta

    from vllm import SamplingParams

    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    res: dict[str, dict] = {}
    for b in batches:
        groups = [
            [
                {"prompt_token_ids": prompts[(g * b + i) % len(prompts)]}
                for i in range(b)
            ]
            for g in range(max(1, len(prompts) // b))
        ]
        llm.generate(groups[0], spn, use_tqdm=False)  # warm-up: shapes, allocator
        t1s, tns, specs, eos = [], [], [], 0
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
                eos += sum(EOS_ID in o.outputs[0].token_ids for o in outs)
            t1s.append(t1)
            tns.append(tn)
            specs.append(spec_delta(mid, spec_counters(llm, spec_on)))
        t1, tn = statistics.median(t1s), statistics.median(tns)
        step_ms = 1e3 * (tn - t1) / (len(groups) * (tokens - 1))
        spec = specs[len(specs) // 2] if spec_on else {}
        res[str(b)] = {
            "groups": len(groups),
            "decode_ms_per_token_per_request": step_ms,
            "tok_s_per_request": 1e3 / step_ms,
            "tok_s_total": b * 1e3 / step_ms,
            "t1_ms": [1e3 * x / len(groups) for x in t1s],
            "tn_s": tns,
            "requests_with_eos": eos,
            "spec": spec,
        }
        rates = [round(p, 3) for p in spec.get("per_pos_rate", [])] if spec else "-"
        print(
            f"[bench] batch {b} ({len(groups)} groups): {step_ms:.2f} ms/token/request "
            f"({b * 1e3 / step_ms:.1f} tok/s total); drafts "
            f"{spec.get('drafts', '-')}, accepted/step "
            f"{(spec.get('mean_accepted_per_step') or 0):.3f}, cumulative per "
            f"position {rates}; requests that sampled EOS: {eos}",
            flush=True,
        )
    return res


def generate(llm, prompts, tokens: int) -> dict:
    """Greedy generations alone and in batches of 2, 4 and 8 (distinct prompts)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tokens, logprobs=2, ignore_eos=True)
    res: dict[str, dict] = {}

    def keep(tag: str, i: int, prompt: list[int], out) -> None:
        c = out.outputs[0]
        res[f"{tag}/{i}"] = {
            "prompt": prompt,
            "ids": list(c.token_ids),
            "lp": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)],
            "top": [{str(t): x.logprob for t, x in lp.items()} for lp in c.logprobs],
        }

    for i, p in enumerate(prompts):
        out = llm.generate([{"prompt_token_ids": p}], sp, use_tqdm=False)[0]
        keep("alone", i, p, out)
    for b in (2, 4, 8):
        for g in range(max(1, len(prompts) // b)):
            batch = [prompts[(g * b + i) % len(prompts)] for i in range(b)]
            outs = llm.generate(
                [{"prompt_token_ids": p} for p in batch], sp, use_tqdm=False
            )
            for i, (p, o) in enumerate(zip(batch, outs)):
                keep(f"batch{b}g{g}", i, p, o)
    return res


def teacher(llm, runs: dict[str, dict]) -> dict:
    """Gap of every generated token to this engine's own top-1, per generation file."""
    import numpy as np

    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    report: dict[str, dict] = {}
    for name, gens in runs.items():
        gaps: list[float] = []
        flips = 0
        for key, rec in gens.items():
            if not key.startswith("alone/") and not key.startswith("batch8"):
                continue
            ids = rec["prompt"] + rec["ids"]
            out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
            for pos in range(len(rec["prompt"]), len(ids)):
                entry = out.prompt_logprobs[pos]
                actual = entry[ids[pos]].logprob
                best = max(x.logprob for x in entry.values())
                gaps.append(best - actual)
                flips += best > actual
        g = np.asarray(gaps)
        report[name] = {
            "positions": int(g.size),
            "not_top1": int(flips),
            "gap_mean": float(g.mean()),
            "gap_p99": float(np.percentile(g, 99)),
            "gap_max": float(g.max()),
            "gap_gt_0.1": int((g > 0.1).sum()),
            "gap_gt_0.3": int((g > 0.3).sum()),
        }
        print(f"[teacher] {name}: {report[name]}", flush=True)
    return report


def compare(a: dict, b: dict, near_tie: float = 0.1) -> dict:
    """First divergence of two greedy generation files, and whether it is a near tie.

    A divergence is explained by numerical noise when the two best candidates of
    *both* runs at that position are closer than ``near_tie`` nats: the runs then
    chose between tokens the model itself cannot tell apart.
    """
    rows = []
    for key in sorted(set(a) & set(b)):
        ia, ib = a[key]["ids"], b[key]["ids"]
        n = min(len(ia), len(ib))
        first = next((i for i in range(n) if ia[i] != ib[i]), None)
        row = {"key": key, "tokens": n, "first_divergence": first}
        if first is not None:
            for tag, gen in (("a", a[key]), ("b", b[key])):
                top = sorted(gen["top"][first].values(), reverse=True)
                row[f"{tag}_gap"] = top[0] - top[1] if len(top) > 1 else None
            row["near_tie"] = all(
                row[f"{t}_gap"] is not None and row[f"{t}_gap"] < near_tie
                for t in ("a", "b")
            )
        rows.append(row)
    split = [r for r in rows if r["first_divergence"] is not None]
    summary = {
        "generations": len(rows),
        "identical": len(rows) - len(split),
        "diverged": len(split),
        "diverged_at_near_tie": sum(r["near_tie"] for r in split),
        "diverged_not_near_tie": [
            (r["key"], r["first_divergence"], r["a_gap"], r["b_gap"])
            for r in split
            if not r["near_tie"]
        ],
        "mean_matched_prefix": (
            sum(
                (
                    r["first_divergence"]
                    if r["first_divergence"] is not None
                    else r["tokens"]
                )
                / r["tokens"]
                for r in rows
            )
            / max(1, len(rows))
        ),
    }
    print(f"[compare] {summary}", flush=True)
    return summary


def prefix_check(
    llm, prompts: list[list[int]], spec_on: bool, tokens: int = 48
) -> dict:
    """Prefix-cache hits with the draft on: a ~1000-token prompt and an overlapping one
    run cold, as a hit, after a cache reset and as duplicates inside one step.

    Per run: the tokens the engine served from the cache and, with a draft, the
    accepted tokens per step (a hit that left the draft's own KV blocks stale would
    show as a drop).  Every hit / duplicate generation is compared with the cold one
    (greedy, so equal up to the near-tie noise of ``compare``)."""
    from graph_check import spec_counters, spec_delta

    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True, logprobs=2)
    shared = [t for p in prompts[:-2] for t in p]
    a, b = shared + prompts[-2], shared + prompts[-1]
    runs: dict[str, dict] = {}

    def run(label: str, batch: list[list[int]]) -> None:
        before = spec_counters(llm, spec_on)
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in batch], sp, use_tqdm=False
        )
        spec = spec_delta(before, spec_counters(llm, spec_on))
        for i, o in enumerate(outs):
            c = o.outputs[0]
            runs[f"{label}/{i}"] = {
                "ids": list(c.token_ids),
                "top": [
                    {str(t): x.logprob for t, x in lp.items()} for lp in c.logprobs
                ],
                "cached": o.num_cached_tokens,
                "prompt": len(o.prompt_token_ids),
            }
        cached = [runs[f"{label}/{i}"]["cached"] for i in range(len(outs))]
        print(
            f"[prefix] {label}: prompt {[len(p) for p in batch]} tokens, served "
            f"from the cache {cached}; accepted/step "
            f"{(spec.get('mean_accepted_per_step') or 0):.3f}",
            flush=True,
        )

    llm.reset_prefix_cache()
    run("cold_a", [a])
    run("hit_a", [a])
    run("hit_b", [b])
    llm.reset_prefix_cache()
    run("cold_b", [b])
    llm.reset_prefix_cache()
    run("same_step", [a, a, b, b])
    pairs = {
        "hit_a vs cold_a": ("hit_a/0", "cold_a/0"),
        "hit_b vs cold_b": ("hit_b/0", "cold_b/0"),
        "same_step a vs cold_a": ("same_step/0", "cold_a/0"),
        "same_step a' vs cold_a": ("same_step/1", "cold_a/0"),
        "same_step b vs cold_b": ("same_step/2", "cold_b/0"),
        "same_step b' vs cold_b": ("same_step/3", "cold_b/0"),
    }
    report = {}
    for name, (x, y) in pairs.items():
        print(f"[prefix] {name}:", flush=True)
        report[name] = compare({"g": runs[x]}, {"g": runs[y]})
    return {
        "runs": {k: {f: v[f] for f in ("cached", "prompt")} for k, v in runs.items()},
        "compare": report,
    }


def table(paths: list[str]) -> None:
    """Markdown tables of ``--json`` files: time per token of one request, speed-up
    over the file without a draft, accepted tokens per step and the cumulative
    acceptance per draft position at every batch size."""
    runs = []
    for p in paths:
        doc = json.loads(Path(p).read_text())
        a = doc["args"]
        if a["spec_k"]:
            sched = (
                json.loads(a["schedule"])
                if a["schedule"]
                else [[1, 1 << 30, a["spec_k"]]]
            )
            label = "K=" + "/".join(sorted({str(s[2]) for s in sched}, key=int))
        else:
            label = "no draft"
        runs.append((label, doc["bench"]))
    base = next((b for label, b in runs if label == "no draft"), None)
    batches = sorted({int(b) for _, bench in runs for b in bench})
    print("ms per token of one request (speed-up over no draft)")
    print("| run | " + " | ".join(f"b{b}" for b in batches) + " |")
    print("|---|" + "---|" * len(batches))
    for label, bench in runs:
        cells = []
        for b in batches:
            r = bench.get(str(b))
            if r is None:
                cells.append("-")
                continue
            ms = r["decode_ms_per_token_per_request"]
            ref = base.get(str(b)) if base is not None and base is not bench else None
            ratio = (
                f" ({ref['decode_ms_per_token_per_request'] / ms:.2f}x)" if ref else ""
            )
            cells.append(f"{ms:.2f}{ratio}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    print("accepted drafted tokens per step; cumulative acceptance per position")
    for label, bench in runs:
        for b in batches:
            spec = (bench.get(str(b)) or {}).get("spec") or {}
            if not spec:
                continue
            rates = ", ".join(f"{x:.3f}" for x in spec["per_pos_rate"])
            print(
                f"- {label} b{b}: {spec['mean_accepted_per_step']:.3f} accepted/step "
                f"over {spec['drafts']:.0f} drafts; [{rates}]"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--effort", default=None, help="reasoning_effort of the template")
    ap.add_argument("--mode", default="FULL_AND_PIECEWISE")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument(
        "--async-scheduling",
        choices=("auto", "on", "off"),
        default="auto",
        help="engine async scheduling (auto: the engine default, on for this setup)",
    )
    ap.add_argument("--spec-k", type=int, default=0, help="DSpark drafted tokens")
    ap.add_argument(
        "--schedule",
        default="",
        help="JSON [[first batch, last batch, verified tokens], ...]",
    )
    ap.add_argument("--prompts", type=int, default=len(CHAT_PROMPTS))
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--gen", metavar="FILE")
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--teacher", default="", metavar="A.json,B.json")
    ap.add_argument(
        "--prefix-check",
        action="store_true",
        help="prefix-cache hits (cold / hit / reset / duplicates in one step) of the "
        "chat prompts concatenated into one long shared prefix",
    )
    ap.add_argument(
        "--compare",
        default="",
        metavar="A.json,B.json",
        help="compare two generation files (no engine is started)",
    )
    ap.add_argument("--json")
    ap.add_argument(
        "--table",
        nargs="+",
        metavar="F.json",
        help="print the tables of several --json files (no engine is started)",
    )
    args = ap.parse_args()
    if args.table:
        table(args.table)
        return 0
    if args.compare:
        left, right = (json.loads(Path(p).read_text()) for p in args.compare.split(","))
        compare(left, right)
        return 0
    _, prompts = render_prompts(
        args.ckpt, args.chat_template_file, args.effort, args.prompts
    )
    print(
        f"{len(prompts)} chat prompts of {min(map(len, prompts))}.."
        f"{max(map(len, prompts))} tokens",
        flush=True,
    )
    llm = build_llm(args)
    result: dict = {"args": vars(args)}
    if args.bench:
        result["bench"] = bench(
            llm,
            prompts,
            [int(b) for b in args.batches.split(",")],
            args.tokens,
            args.reps,
            bool(args.spec_k),
        )
    if args.gen:
        gens = generate(llm, prompts, args.gen_tokens)
        Path(args.gen).write_text(json.dumps(gens))
        print(f"wrote {args.gen}: {len(gens)} generations", flush=True)
    if args.teacher:
        runs = {p: json.loads(Path(p).read_text()) for p in args.teacher.split(",")}
        result["teacher"] = teacher(llm, runs)
    if args.prefix_check:
        result["prefix"] = prefix_check(llm, prompts, bool(args.spec_k))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
