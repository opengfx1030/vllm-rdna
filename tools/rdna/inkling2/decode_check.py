#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-path check of the real Inkling-2-Mini checkpoint: teacher-forced decode NLL,
greedy decode and decode ms/token, for A/B runs of a decode-kernel change.

``real_check.py --nll-file`` teacher-forces a text through *prefill* (prompt logprobs),
so it never runs a ``q_len == 1`` attention call.  This tool prefills a prefix and then
forces the next tokens of the text through the *decode* path one step at a time: a
logits processor (model runner V2 interface) makes every decode step sample the next
token of the file, and the raw logprob of that token (computed from the logits before
the processor runs) is the NLL of the decode step.  Every run attends the same context
at every step, so two runs are comparable token by token.

    # one process per flag value (graphs bake the split factors); TP=1,
    # FULL_AND_PIECEWISE
    VLLM_INKLING2_DECODE_FUSED=0 python decode_check.py run --out A1.json
    VLLM_INKLING2_DECODE_FUSED=1 python decode_check.py run --out B.json
    VLLM_INKLING2_DECODE_FUSED=0 python decode_check.py run --out A2.json   # the floor
    python decode_check.py compare A1.json A2.json B.json  # or label=path arguments

``run`` also greedy-decodes two prompts longer than the sliding window (ids and the
logprob of the chosen token per step) and times the decode (wall of ``N + 1`` tokens
minus wall of one token of the same prompt) at two context lengths.  ``compare`` takes
the first two dumps as the run-to-run floor of the shipped configuration (two TP=1 runs
differ already: the expert GEMMs add the eight expert outputs with fp16 atomics) and
the others as the runs under test; a run passes when the mean ``|dlogp|`` of its forced
tokens against the first floor run is within ``2 * floor + 0.01`` (the rule of
``real_check.py --nll-against``).  The greedy ids are reported, not gated: two shipped
runs already diverge at a near-tie within the first tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CKPT = Path("~/hfcache/inkling2-mini").expanduser()


def _force_next_processor():
    """The teacher-forcing logits processor class (imports vLLM)."""
    import torch

    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsContext,
        LogitsProcessor,
        LogitsProcRequestState,
    )

    class ForceNext(LogitsProcessor):
        """``SamplingParams.extra_args["force"] = {"start": p0, "ids": [...]}``: the
        row whose input token sits at position ``p0 + i`` samples ``ids[i]``."""

        def __init__(self, vllm_config, req_states: LogitsProcRequestState):
            self.device = req_states.device
            self.table = torch.full(
                (req_states.max_num_reqs, vllm_config.model_config.max_model_len + 1),
                -1,
                dtype=torch.int32,
                device=self.device,
            )

        def add_request(self, req_idx, sampling_params) -> bool:
            force = (sampling_params.extra_args or {}).get("force")
            self.table[req_idx].fill_(-1)
            if force is None:
                return False
            p0, ids = int(force["start"]), force["ids"]
            self.table[req_idx, p0 : p0 + len(ids)] = torch.tensor(
                ids, dtype=torch.int32, device=self.device
            )
            return True

        def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
            tgt = self.table[ctx.expanded_idx_mapping.long(), ctx.pos.long()].long()
            forced = tgt >= 0
            one_hot = torch.full_like(logits, float("-inf"))
            one_hot.scatter_(1, tgt.clamp(min=0)[:, None], 0.0)
            return torch.where(forced[:, None], one_hot, logits)

    return ForceNext


def run(args) -> int:
    # the engine runs in this process so that the processor class needs no import path
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from engine_args import graph_kwargs
    from real_check import long_prompt
    from transformers import AutoTokenizer

    import vllm._custom_ops as ops
    from vllm import LLM, SamplingParams

    ckpt = Path(args.ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt)
    template = (ckpt / "chat_template.jinja").read_text()
    text_ids = tok(Path(args.text).read_text(), add_special_tokens=False)["input_ids"]
    probe = {
        "VLLM_INKLING2_DECODE_FUSED": os.environ.get("VLLM_INKLING2_DECODE_FUSED"),
        "module_flag": ops._INKLING2_DECODE_FUSED,
        "splits_sliding_B1": list(
            ops.inkling2_attention_splits(1, 8, 1024, args.max_model_len, 1)
        ),
        "splits_full_B1": list(
            ops.inkling2_attention_splits(1, 8, 0, args.max_model_len, 1)
        ),
        "splits_sliding_B2": list(
            ops.inkling2_attention_splits(2, 8, 1024, args.max_model_len, 1)
        ),
    }
    print("PROBE", json.dumps(probe), flush=True)

    t0 = time.time()
    llm = LLM(
        model=str(ckpt),
        dtype="float16",
        **graph_kwargs(args.cudagraph_mode, "", args.decode_sizes),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        skip_tokenizer_init=True,
        max_logprobs=8,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
        logits_processors=[_force_next_processor()],
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)

    def gen(ids, sp):
        t = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        return out.outputs[0], time.perf_counter() - t

    gen(text_ids[:200], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))

    result: dict = {"probe": probe, "greedy": {}, "forced": [], "timing": {}}
    greedy_sp = SamplingParams(
        temperature=0.0, max_tokens=args.greedy_tokens, ignore_eos=True, logprobs=1
    )
    prompts = {
        "facts1700": long_prompt(tok, template, 1700),
        "text1400": text_ids[:1400],
    }
    for name, ids in prompts.items():
        out, dt = gen(ids, greedy_sp)
        result["greedy"][name] = {
            "prompt_tokens": len(ids),
            "ids": [int(t) for t in out.token_ids],
            "logp": [
                float(out.logprobs[i][t].logprob) for i, t in enumerate(out.token_ids)
            ],
        }
        print(
            f"greedy {name}: {len(ids)} prompt tokens, {dt:.2f}s: "
            f"{tok.decode(out.token_ids)[:80]!r}",
            flush=True,
        )

    for cut in args.cuts.split(","):
        p, k = (int(x) for x in cut.split(":"))
        prompt, target = text_ids[:p], text_ids[p : p + k]
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=len(target),
            ignore_eos=True,
            logprobs=1,
            extra_args={"force": {"start": p - 1, "ids": target}},
        )
        out, dt = gen(prompt, sp)
        if [int(t) for t in out.token_ids] != target:
            raise SystemExit("the logits processor did not force the text")
        logp = [float(out.logprobs[i][t].logprob) for i, t in enumerate(target)]
        result["forced"].append({"prefix": p, "tokens": len(target), "logp": logp})
        print(
            f"forced decode {p}+{len(target)}: {dt:.2f}s mean logp "
            f"{sum(logp) / len(logp):.4f} (decode steps 2..{len(target)}: "
            f"{sum(logp[1:]) / (len(logp) - 1):.4f})",
            flush=True,
        )

    n = args.time_tokens
    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    many = SamplingParams(temperature=0.0, max_tokens=n + 1, ignore_eos=True)
    for ctx in (int(c) for c in args.time_ctx.split(",")):
        ids = long_prompt(tok, template, ctx)
        gen(ids, one)
        t1, tn = [], []
        for _ in range(args.time_reps):
            t1.append(gen(ids, one)[1])
            tn.append(gen(ids, many)[1])
        ms = (min(tn) - min(t1)) / n * 1e3
        result["timing"][str(ctx)] = {
            "prompt_tokens": len(ids),
            "t1_s": t1,
            "tn_s": tn,
            "decode_ms_per_token": ms,
        }
        print(
            f"timing ctx {len(ids)}: prefill+1 {min(t1):.3f}s, +{n} tokens "
            f"{min(tn):.3f}s -> {ms:.2f} ms/token",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(result))
    print("DECODE CHECK RUN DONE", flush=True)
    return 0


def _stats(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    ad = sorted(abs(x) for x in d)
    n = len(d)
    return {
        "mean_abs": sum(ad) / n,
        "p99": ad[min(n - 1, int(0.99 * n))],
        "max": ad[-1],
        "mean_signed": sum(d) / n,
    }


def _agree(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare(args) -> int:
    runs = {}
    for i, item in enumerate(args.dumps):
        label, _, path = item.partition("=")
        if not path:
            label, path = ("A1", "A2", "B", "B2", "B3")[i], item
        runs[label] = json.loads(Path(path).read_text())
    names = list(runs)
    f1, f2 = names[0], names[1]  # the floor pair
    for n, r in runs.items():
        print(f"{n}: probe {r['probe']}")
    pairs = [(f1, f2)] + [
        (a, b) for b in names[2:] for a in (f1, f2)
    ]  # the floor, then every run under test against both floor runs
    print(
        "\n== teacher-forced decode NLL (raw logprob of the forced token; "
        "steps 2.. are q_len == 1 decode steps)"
    )
    flat = {n: [x for c in r["forced"] for x in c["logp"][1:]] for n, r in runs.items()}
    for n in runs:
        mean = sum(flat[n]) / len(flat[n])
        print(f"  {n}: mean logp {mean:.5f} over {len(flat[n])} steps")
    floor, ok = None, True
    for a, b in pairs:
        s = _stats(flat[a], flat[b])
        tag = ""
        if (a, b) == (f1, f2):
            floor, tag = s["mean_abs"], "  <- floor (shipped vs shipped)"
        elif floor is not None and a == f1:
            limit = 2 * floor + 0.01
            good = s["mean_abs"] <= limit
            ok &= good
            tag = f"  limit {limit:.5f} {'PASS' if good else 'FAIL'}"
        print(
            f"  {a} vs {b}: mean|dlogp| {s['mean_abs']:.5f} p99 {s['p99']:.4f} "
            f"max {s['max']:.4f} mean signed {s['mean_signed']:+.5f}{tag}"
        )
    print("\n== greedy decode: ids equal for N of the generated tokens, mean |dlogp|")
    print("   of the chosen token over the common prefix")
    for name in runs[f1]["greedy"]:
        for a, b in pairs:
            ga, gb = runs[a]["greedy"][name], runs[b]["greedy"][name]
            k = _agree(ga["ids"], gb["ids"])
            diffs = [abs(x - y) for x, y in zip(ga["logp"][:k], gb["logp"][:k])]
            print(
                f"  {name} {a} vs {b}: ids equal for {k} of {len(ga['ids'])}, "
                f"mean|dlogp| {sum(diffs) / max(k, 1):.5f}"
            )
    print("\n== decode ms/token (min over reps)")
    for ctx in runs[f1]["timing"]:
        row = ", ".join(
            f"{n} {r['timing'][ctx]['decode_ms_per_token']:.2f}"
            if ctx in r["timing"]
            else f"{n} -"
            for n, r in runs.items()
        )
        print(f"  ctx {ctx}: {row}")
    print("\nDECODE CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="boot the engine and write one dump")
    r.add_argument("--ckpt", default=str(CKPT))
    r.add_argument("--out", required=True)
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    r.add_argument("--decode-sizes", default="1")
    r.add_argument("--text", default=str(ROOT / "docs/rdna2/inkling2-port-spec.md"))
    r.add_argument(
        "--cuts",
        default="1200:300,2200:300,3200:300",
        help="prefix:continuation tokens",
    )
    r.add_argument("--greedy-tokens", type=int, default=96)
    r.add_argument("--time-ctx", default="1700,7000")
    r.add_argument("--time-tokens", type=int, default=96)
    r.add_argument("--time-reps", type=int, default=3)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare", help="A1 A2 (the floor pair) B [B2 ...]")
    c.add_argument(
        "dumps", nargs="+", help="[label=]dump.json; the first two are the floor pair"
    )
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    if args.cmd == "compare" and len(args.dumps) < 3:
        raise SystemExit("compare needs the floor pair and one run under test")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
